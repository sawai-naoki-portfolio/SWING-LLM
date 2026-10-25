"""
評価スイート (swi_llm/eval/run_eval.py) の単体テスト（§14）。
モデルロード・生成を伴わない純粋部分（ベンチマーク読み込み・ckpt 探索）を対象とする。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.eval.run_eval import load_benchmark, find_latest_checkpoint, _CKPT_STAGE_PRIORITY
from swi_llm.eval.contamination import ContaminationChecker


class TestBenchmark(unittest.TestCase):
    def test_benchmark_loads(self):
        items = load_benchmark()
        self.assertGreater(len(items), 0)

    def test_benchmark_expanded(self):
        # §14.1: ベンチマークは拡充済み（20問以上）
        self.assertGreaterEqual(len(load_benchmark()), 20)

    def test_benchmark_schema(self):
        for item in load_benchmark():
            self.assertIn("category", item)
            self.assertIn("question", item)
            self.assertIn("answer", item)

    def test_categories_present(self):
        cats = {item["category"] for item in load_benchmark()}
        self.assertTrue({"math", "knowledge", "reasoning", "code"}.issubset(cats))


class TestContamination(unittest.TestCase):
    def setUp(self):
        self.items = [{"question": "日本の首都はどこですか？", "answer": "東京"},
                      {"question": "2 + 2 はいくつですか？", "answer": "4"}]
        self.checker = ContaminationChecker(self.items)

    def test_clean_corpus_no_contamination(self):
        rep = self.checker.report(["機械学習は人工知能の一分野であり応用が広い。"])
        self.assertEqual(rep["n_contaminated"], 0)

    def test_leaked_question_detected(self):
        rep = self.checker.report(["解説: 日本の首都はどこですか？ 答えは東京です。"])
        self.assertGreaterEqual(rep["n_contaminated"], 1)

    def test_doc_exclusion(self):
        self.assertTrue(self.checker.doc_is_contaminated("この文書には 2 + 2 はいくつですか？ が含まれる"))
        self.assertFalse(self.checker.doc_is_contaminated("全く無関係な内容の文書"))

    def test_real_benchmark_not_self_contaminated_by_clean_text(self):
        rep = ContaminationChecker(load_benchmark()).report(["これは一般的な説明文です。"])
        self.assertEqual(rep["n_contaminated"], 0)


class TestCheckpointDiscovery(unittest.TestCase):
    def test_stage_priority_order(self):
        # 後段ステージ (grpo) が先頭 = 優先
        self.assertEqual(_CKPT_STAGE_PRIORITY[0], "grpo_checkpoint")
        self.assertEqual(_CKPT_STAGE_PRIORITY[-1], "pretrain_checkpoint")

    def test_find_latest_returns_str_or_none(self):
        result = find_latest_checkpoint()
        self.assertTrue(result is None or isinstance(result, str))


if __name__ == "__main__":
    unittest.main()
