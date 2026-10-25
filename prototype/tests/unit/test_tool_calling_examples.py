"""
data/tool_calling_examples.py（ローカルツール一致の FC 教師データ合成）と
eval/fc_eval.py（FC 指標）の単体テスト。

最重要の検証: **合成した正例の tool_call が、本番と同じ AgentBase で実際に dispatch
できる（ok=True）こと**。教師データと推論器が食い違っていると、いくら学習しても
ok 率は上がらない（これが外部 FC データセットだけでは足りない理由そのもの）。
"""
import json
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.agent.base import AgentBase
from swi_llm.agent.tools import TOOL_SCHEMAS, calculate, get_weather, search_documents
from swi_llm.data.tool_calling_examples import (
    build_tool_calling_examples, build_fc_eval_set, SUPPORTED_TOOLS,
)
from swi_llm.eval.fc_eval import score_completion, aggregate
from swi_llm.utils.chat_template import TOOL_CALL_PATTERN, THINK_START, THINK_END
from swi_llm.utils.constants import TOOL_CALCULATE, TOOL_GET_WEATHER, TOOL_SEARCH_DOCUMENTS


def _make_agent(execute=False):
    """本番（web/routes/inference.py）と同じ登録セット。SEC-6 により execute_python は登録しない。

    execute=False のときは副作用（HTTP・FAISS 検索）を避けるため、実処理をスタブに差し替える。
    dispatch の可否判定（= last_call_ok / 戻り値 ok）はツール本体の中身に依存しないため、
    これで AgentBase の dispatch 経路を忠実に検証できる。
    """
    agent = AgentBase()
    if execute:
        funcs = {TOOL_CALCULATE: calculate, TOOL_GET_WEATHER: get_weather,
                 TOOL_SEARCH_DOCUMENTS: search_documents}
    else:
        funcs = {TOOL_CALCULATE: lambda expression: {"result": "stub"},
                 TOOL_GET_WEATHER: lambda location: {"condition": "stub"},
                 TOOL_SEARCH_DOCUMENTS: lambda query: {"results": "stub"}}
    for name, func in funcs.items():
        agent.register_tool(name, func, TOOL_SCHEMAS[name])
    return agent


class TestBuildToolCallingExamples(unittest.TestCase):
    def test_counts_and_determinism(self):
        a = build_tool_calling_examples(n_per_tool=5, n_negative=7)
        b = build_tool_calling_examples(n_per_tool=5, n_negative=7)
        self.assertEqual(len(a), 5 * len(SUPPORTED_TOOLS) + 7)
        self.assertEqual(a, b, "同じ seed で生成結果が変わる（学習の再現性が壊れる）")

    def test_positives_dispatch_on_real_agent(self):
        """合成正例の tool_call は、本番と同じ AgentBase でそのまま dispatch できる。"""
        agent = _make_agent()
        examples = build_tool_calling_examples(n_per_tool=10, n_negative=0)
        self.assertEqual(len(examples), 10 * len(SUPPORTED_TOOLS))

        for ex in examples:
            _result, ok = agent.process_llm_output_ex(ex["assistant"])
            self.assertTrue(ok, f"dispatch できない正例が混入している: {ex['assistant'][:120]!r}")

    def test_positives_use_registered_tool_names_and_required_args(self):
        """ツール名・必須引数は TOOL_SCHEMAS と一致する（文字列の独自ハードコードがない）。"""
        for ex in build_tool_calling_examples(n_per_tool=6, n_negative=0):
            m = TOOL_CALL_PATTERN.search(ex["assistant"])
            self.assertIsNotNone(m)
            payload = json.loads(m.group(1).strip())      # 壊れた JSON なら例外で落ちる
            self.assertIn(payload["name"], SUPPORTED_TOOLS)
            required = TOOL_SCHEMAS[payload["name"]]["parameters"]["required"]
            for r in required:
                self.assertIn(r, payload["arguments"])

    def test_negatives_contain_no_tool_call(self):
        """負例は tool_call を含まない（過剰発火を抑える教師）。"""
        for ex in build_tool_calling_examples(n_per_tool=0, n_negative=20):
            self.assertIsNone(TOOL_CALL_PATTERN.search(ex["assistant"]),
                              f"負例に tool_call が混入: {ex['assistant'][:120]!r}")
            _result, ok = _make_agent().process_llm_output_ex(ex["assistant"])
            self.assertFalse(ok)

    def test_cot_blocks_are_well_formed(self):
        """<think> を付ける場合は開始・終了タグが対で、tool_call はブロック外に出る。"""
        for ex in build_tool_calling_examples(n_per_tool=8, n_negative=8):
            text = ex["assistant"]
            self.assertEqual(text.count(THINK_START), text.count(THINK_END))
            if THINK_START in text and TOOL_CALL_PATTERN.search(text):
                # tool_call は </think> より後ろ（思考の中に隠れない = 推論時に抽出できる）
                self.assertLess(text.index(THINK_END), text.index("<tool_call>"))

    def test_calculate_expressions_are_evaluable(self):
        """calculate 正例の expression は実ツールが実際に計算できる（教師が不正式でない）。"""
        for ex in build_tool_calling_examples(n_per_tool=15, n_negative=0):
            m = TOOL_CALL_PATTERN.search(ex["assistant"])
            payload = json.loads(m.group(1).strip())
            if payload["name"] != TOOL_CALCULATE:
                continue
            result = calculate(expression=payload["arguments"]["expression"])
            self.assertNotIn("error", result, f"評価できない式: {payload['arguments']}")


class TestFcEvalSet(unittest.TestCase):
    def test_eval_set_labels_match_content(self):
        items = build_fc_eval_set(n_per_tool=4, n_negative=6)
        self.assertEqual(len(items), 4 * len(SUPPORTED_TOOLS) + 6)
        self.assertEqual(sum(1 for i in items if i["expect_tool"] is None), 6)
        for i in items:
            if i["expect_tool"] is not None:
                self.assertIn(i["expect_tool"], SUPPORTED_TOOLS)

    def test_eval_set_is_held_out_from_training_data(self):
        """評価セットは学習データとプロンプトが 1 件も重複しない（丸暗記ではなく般化を測る）。

        回帰の経緯: 当初は seed だけを変えていたが、get_weather / search_documents の
        正例は都市・話題を `i % len(pool)` で選ぶため seed に依存せず、評価セットの
        プロンプトが学習データに完全包含されていた（18 件中 12 件が重複）。
        語彙プールを train/eval で排他分割して解消した。
        """
        train_users = {ex["user"] for ex in build_tool_calling_examples()}
        eval_users = {i["user"] for i in build_fc_eval_set()}
        overlap = train_users & eval_users
        self.assertEqual(overlap, set(),
                         f"評価プロンプトが学習データに漏れている: {sorted(overlap)[:5]}")

    def test_train_and_eval_vocab_pools_are_disjoint(self):
        """語彙プール（都市・検索話題・負例文）が train/eval で排他であること。"""
        train = build_tool_calling_examples(n_per_tool=30, n_negative=30, split="train")
        ev = build_tool_calling_examples(n_per_tool=30, n_negative=30, split="eval")
        train_weather_or_search = {
            ex["user"] for ex in train
            if any(t in ex["assistant"] for t in (TOOL_GET_WEATHER, TOOL_SEARCH_DOCUMENTS))
        }
        eval_weather_or_search = {
            ex["user"] for ex in ev
            if any(t in ex["assistant"] for t in (TOOL_GET_WEATHER, TOOL_SEARCH_DOCUMENTS))
        }
        self.assertEqual(train_weather_or_search & eval_weather_or_search, set())

        train_neg = {ex["user"] for ex in train if not TOOL_CALL_PATTERN.search(ex["assistant"])}
        eval_neg = {ex["user"] for ex in ev if not TOOL_CALL_PATTERN.search(ex["assistant"])}
        self.assertEqual(train_neg & eval_neg, set())

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            build_tool_calling_examples(split="validation")


class TestScoreCompletion(unittest.TestCase):
    def setUp(self):
        self.agent = _make_agent()

    def test_correct_positive(self):
        c = '<tool_call>{"name": "calculate", "arguments": {"expression": "2 + 2"}}</tool_call>'
        s = score_completion(self.agent, c, expect_tool=TOOL_CALCULATE)
        self.assertTrue(s["has_tool_call"] and s["dispatch_ok"] and s["correct_tool"])

    def test_wrong_tool_is_dispatchable_but_not_correct(self):
        c = '<tool_call>{"name": "calculate", "arguments": {"expression": "1 + 1"}}</tool_call>'
        s = score_completion(self.agent, c, expect_tool=TOOL_GET_WEATHER)
        self.assertTrue(s["dispatch_ok"])
        self.assertFalse(s["correct_tool"])

    def test_unregistered_tool_is_not_dispatchable(self):
        c = '<tool_call>{"name": "send_email", "arguments": {"to": "x"}}</tool_call>'
        s = score_completion(self.agent, c, expect_tool=TOOL_CALCULATE)
        self.assertTrue(s["has_tool_call"])
        self.assertFalse(s["dispatch_ok"])

    def test_missing_required_argument_is_not_dispatchable(self):
        c = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
        s = score_completion(self.agent, c, expect_tool=TOOL_GET_WEATHER)
        self.assertFalse(s["dispatch_ok"])

    def test_broken_json_is_not_dispatchable(self):
        c = '<tool_call>{"name": "calculate", "arguments": </tool_call>'
        s = score_completion(self.agent, c, expect_tool=TOOL_CALCULATE)
        self.assertFalse(s["dispatch_ok"])

    def test_no_tool_call_on_negative(self):
        s = score_completion(self.agent, "日本の首都は東京です。", expect_tool=None)
        self.assertFalse(s["has_tool_call"])

    def test_dispatch_mode_matches_static_judgement(self):
        """dispatch=True（実 dispatch）と既定の静的判定が一致する（副作用なし判定の妥当性）。"""
        cases = [
            ('<tool_call>{"name": "calculate", "arguments": {"expression": "2 + 2"}}</tool_call>', True),
            ('<tool_call>{"name": "nope", "arguments": {}}</tool_call>', False),
            ('<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>', False),
        ]
        for c, expected in cases:
            static = score_completion(self.agent, c, expect_tool=TOOL_CALCULATE)["dispatch_ok"]
            live = score_completion(self.agent, c, expect_tool=TOOL_CALCULATE, dispatch=True)["dispatch_ok"]
            self.assertEqual(static, expected)
            self.assertEqual(static, live, f"静的判定と実 dispatch が食い違う: {c!r}")


class TestAggregate(unittest.TestCase):
    def test_rates(self):
        records = [
            {"expect_tool": "calculate", "has_tool_call": True, "dispatch_ok": True, "correct_tool": True},
            {"expect_tool": "calculate", "has_tool_call": True, "dispatch_ok": True, "correct_tool": False},
            {"expect_tool": "get_weather", "has_tool_call": True, "dispatch_ok": False, "correct_tool": False},
            {"expect_tool": None, "has_tool_call": False, "dispatch_ok": False, "correct_tool": False},
            {"expect_tool": None, "has_tool_call": True, "dispatch_ok": True, "correct_tool": False},
        ]
        m = aggregate(records)
        self.assertEqual(m["n_positive"], 3)
        self.assertEqual(m["n_negative"], 2)
        self.assertAlmostEqual(m["dispatch_ok_rate"], 2 / 3)
        self.assertAlmostEqual(m["correct_tool_rate"], 1 / 3)
        self.assertAlmostEqual(m["no_tool_on_negative_rate"], 1 / 2)

    def test_empty_is_zero_not_crash(self):
        m = aggregate([])
        self.assertEqual(m["dispatch_ok_rate"], 0.0)
        self.assertEqual(m["no_tool_on_negative_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
