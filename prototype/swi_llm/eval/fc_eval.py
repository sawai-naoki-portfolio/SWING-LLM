"""
swi_llm/eval/fc_eval.py

モデル自発 Function Calling の評価（roadmap/llm_roadmap.md ステップ3 / fix 19）。

`web/intent.py` のルールベース先回り実行を撤去できるかは、「モデル自身が正しい
tool_call を出せるか」に懸かっている。ログ KPI（FC-KPI / INTENT-KPI）は本番トラフィックの
事後集計だが、本スクリプトは **hold-out セットに対する能動的な測定** を行い、
学習の前後で比較できる数値を出す。

測る 3 指標（いずれも AgentBase = 実推論器で判定する。教師データと推論器の整合を保証）:
  1. dispatch_ok_rate  : 正例で「実際に dispatch できる tool_call」を出せた割合
                         （= process_llm_output_ex の ok。未登録ツール名・壊れた JSON は不可）
  2. correct_tool_rate : 正例で dispatch でき、かつ **期待したツール** を選べた割合
  3. no_tool_on_negative_rate : 負例（ツール不要）で tool_call を出さなかった割合（過剰発火の抑制）

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m swi_llm.eval.fc_eval
    .venv\\Scripts\\python.exe -m swi_llm.eval.fc_eval --json fc_eval.json --max-new-tokens 96

終了コード: 常に 0（評価は情報提供でありパイプラインを止めない）。
ただし --min-dispatch-ok を指定した場合、dispatch_ok_rate が閾値未満なら 1。
"""
import os
import sys
import json
import argparse

PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROTOTYPE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.chat_template import build_prompt, TOOL_CALL_PATTERN
from swi_llm.utils.constants import TRAINING_SYSTEM_PROMPT
from swi_llm.data.tool_calling_examples import build_fc_eval_set, SUPPORTED_TOOLS

logger = setup_logger("eval.fc")


def _build_agent():
    """推論時（web/routes/inference.py）と同じ登録ツールを持つ AgentBase。

    SEC-6 により execute_python は登録しない（モデル発話由来の任意コード実行を遮断）。
    評価も本番と同じ登録状態で行う（本番で dispatch できない呼び出しを「成功」と数えない）。
    """
    from swi_llm.agent.base import AgentBase
    from swi_llm.agent.tools import get_weather, calculate, search_documents, TOOL_SCHEMAS
    from swi_llm.utils.constants import TOOL_GET_WEATHER, TOOL_CALCULATE, TOOL_SEARCH_DOCUMENTS

    agent = AgentBase()
    agent.register_tool(TOOL_GET_WEATHER, get_weather, TOOL_SCHEMAS[TOOL_GET_WEATHER])
    agent.register_tool(TOOL_CALCULATE, calculate, TOOL_SCHEMAS[TOOL_CALCULATE])
    agent.register_tool(TOOL_SEARCH_DOCUMENTS, search_documents, TOOL_SCHEMAS[TOOL_SEARCH_DOCUMENTS])
    return agent


def _requested_tool_name(completion):
    """出力の tool_call から要求ツール名を取り出す（JSON が壊れていれば None）。"""
    m = TOOL_CALL_PATTERN.search(completion or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip()).get("name")
    except (json.JSONDecodeError, AttributeError):
        return None


def score_completion(agent, completion, expect_tool, dispatch=False):
    """1 件分の採点（生成は行わない純関数。単体テスト可能）。

    Args:
        agent: 本番と同じツールを登録した AgentBase。
        completion: モデル出力（プロンプトを除いた生成部分）。
        expect_tool: 期待ツール名。None なら「tool_call を出してはいけない」負例。
        dispatch: True なら実際にツールを実行して ok を得る（HTTP/検索の副作用あり）。
            False（既定）なら **副作用なし** でツール名の登録有無から dispatch 可否を判定する。

    Returns:
        {"has_tool_call": bool, "dispatch_ok": bool, "correct_tool": bool, "tool": str|None}
    """
    has_call = bool(TOOL_CALL_PATTERN.search(completion or ""))
    tool = _requested_tool_name(completion)

    if not has_call:
        return {"has_tool_call": False, "dispatch_ok": False, "correct_tool": False, "tool": None}

    if dispatch:
        # 実 dispatch（get_weather の HTTP など副作用が走る）。ok は推論器の判定そのもの。
        _result, ok = agent.process_llm_output_ex(completion)
    else:
        # 副作用なしの等価判定: 「JSON が壊れておらず、登録済みツール名である」なら
        # AgentBase は dispatch に進む（引数不足はスキーマ検証で弾かれるため下で確認）。
        ok = tool in agent.tools
        if ok:
            required = agent.tools[tool]["schema"].get("parameters", {}).get("required", [])
            try:
                args = json.loads(TOOL_CALL_PATTERN.search(completion).group(1).strip()).get("arguments", {})
                ok = all(r in args for r in required)
            except (json.JSONDecodeError, AttributeError):
                ok = False

    return {
        "has_tool_call": True,
        "dispatch_ok": bool(ok),
        "correct_tool": bool(ok) and tool == expect_tool,
        "tool": tool,
    }


def aggregate(records):
    """採点済みレコード列 → 指標。正例/負例で分母を分ける。"""
    pos = [r for r in records if r["expect_tool"] is not None]
    neg = [r for r in records if r["expect_tool"] is None]

    def _rate(hits, total):
        return (hits / total) if total else 0.0

    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "dispatch_ok_rate": _rate(sum(1 for r in pos if r["dispatch_ok"]), len(pos)),
        "correct_tool_rate": _rate(sum(1 for r in pos if r["correct_tool"]), len(pos)),
        "no_tool_on_negative_rate": _rate(sum(1 for r in neg if not r["has_tool_call"]), len(neg)),
    }


def run_fc_eval(max_new_tokens=96, dispatch=False, checkpoint=None):
    """
    Args:
        checkpoint: 評価するチェックポイントのパス。None なら最新（ステージ優先順）を自動選択。
            学習前後の比較では **必ず明示指定する** こと。既定の自動選択は
            GRPO > DPO > SFT の優先順で拾うため、SFT だけ再学習しても古い GRPO
            チェックポイントが評価されて効果が見えない。
    """
    import torch
    from swi_llm.core.generation import generate
    from swi_llm.data.tokenizer import get_tokenizer
    from swi_llm.eval.run_eval import find_latest_checkpoint, load_model

    tokenizer = get_tokenizer()
    ckpt_path = checkpoint or find_latest_checkpoint()
    if checkpoint and not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    model = load_model(ckpt_path)
    agent = _build_agent()

    items = build_fc_eval_set()
    records = []
    for item in items:
        # 推論時と同じシステムプロンプト（TRAINING_SYSTEM_PROMPT = 既定ペルソナ）を使う。
        prompt = build_prompt(item["user"], system_text=TRAINING_SYSTEM_PROMPT)
        with torch.no_grad():
            output = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                              temperature=0.3, top_p=0.9)
        completion = output[len(prompt):]
        scored = score_completion(agent, completion, item["expect_tool"], dispatch=dispatch)
        scored["expect_tool"] = item["expect_tool"]
        records.append(scored)

    metrics = aggregate(records)
    metrics["checkpoint"] = os.path.basename(ckpt_path) if ckpt_path else "random"

    logger.info("=" * 56)
    logger.info(f"FC EVAL (checkpoint: {metrics['checkpoint']}, tools={list(SUPPORTED_TOOLS)})")
    logger.info("=" * 56)
    logger.info(f"  dispatch_ok_rate         {metrics['dispatch_ok_rate'] * 100:5.1f}%  "
                f"(正例 n={metrics['n_positive']}: 実際に dispatch できる tool_call を出せた割合)")
    logger.info(f"  correct_tool_rate        {metrics['correct_tool_rate'] * 100:5.1f}%  "
                f"(正例: 期待どおりのツールを選べた割合)")
    logger.info(f"  no_tool_on_negative_rate {metrics['no_tool_on_negative_rate'] * 100:5.1f}%  "
                f"(負例 n={metrics['n_negative']}: ツールを呼ばずに答えた割合＝過剰発火の抑制)")
    logger.info("=" * 56)
    logger.info("ルールベース先回り実行（web/intent.py）の撤去は、dispatch_ok_rate と "
                "no_tool_on_negative_rate が十分高いことを確認してから段階的に行う（fix 19）。")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--json", dest="json_path", default=None, help="指標を JSON で保存")
    ap.add_argument("--min-dispatch-ok", type=float, default=None,
                    help="dispatch_ok_rate がこの値未満なら終了コード 1")
    ap.add_argument("--dispatch", action="store_true",
                    help="実際にツールを実行して判定する（get_weather の HTTP 等の副作用あり）")
    ap.add_argument("--checkpoint", default=None,
                    help="評価するチェックポイント（学習前後の比較では明示指定する）")
    args = ap.parse_args()

    metrics = run_fc_eval(max_new_tokens=args.max_new_tokens, dispatch=args.dispatch,
                          checkpoint=args.checkpoint)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved FC eval metrics to {args.json_path}")

    if args.min_dispatch_ok is not None and metrics["dispatch_ok_rate"] < args.min_dispatch_ok:
        logger.warning(f"dispatch_ok_rate {metrics['dispatch_ok_rate']:.3f} < "
                       f"--min-dispatch-ok {args.min_dispatch_ok}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
