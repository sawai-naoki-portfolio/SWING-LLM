"""
swi_llm/eval/rag_eval.py

RAG 検索品質の回帰評価（roadmap/llm ステップ4 / fix 02 / 実装編 §11 の検索版）。

`rag_benchmark.jsonl` の各質問について SimpleRAG（実運用と同じ構成: チャンキング →
BM25+密検索の RRF 統合 → クロスエンコーダ再順位付け）で検索し、期待文字列が
上位 k 件に含まれる割合（recall@k）を報告する。RAG の変更（チャンカ・融合・
リランカ・埋め込みモデル）が検索品質を退行させていないかの物差し。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m swi_llm.eval.rag_eval
    .venv\\Scripts\\python.exe -m swi_llm.eval.rag_eval --top-k 2 --json out.json

終了コード: 常に 0（情報提供）。--min-recall 指定時のみ、recall@k が閾値未満なら 1。
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

logger = setup_logger("eval.rag")

_BENCHMARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_benchmark.jsonl")


def load_benchmark(path=_BENCHMARK_PATH):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_rag_eval(top_k=2):
    from swi_llm.agent.rag import SimpleRAG

    rag = SimpleRAG()          # 実運用と同じ既定構成（埋め込み + リランカの自動ロード）
    items = load_benchmark()

    details = []
    hits = 0
    for item in items:
        result_text = rag.search(item["question"], top_k=top_k)
        hit = item["expect"] in result_text
        hits += int(hit)
        details.append({"question": item["question"], "expect": item["expect"],
                        "source": item.get("source"), "hit": hit})
        logger.info(f"  [{'HIT ' if hit else 'MISS'}] expect={item['expect']!r} <- {item['question']}")

    recall = hits / len(items) if items else 0.0
    metrics = {
        "recall_at_k": recall,
        "k": top_k,
        "n_questions": len(items),
        "dense": rag.use_dense,
        "reranker": type(rag.reranker).__name__ if rag.reranker is not None else None,
        "n_chunks": len(rag.documents),
    }

    logger.info("=" * 56)
    logger.info(f"RAG EVAL: recall@{top_k} = {recall * 100:.1f}%  "
                f"({hits}/{len(items)}; chunks={metrics['n_chunks']}, "
                f"dense={metrics['dense']}, reranker={metrics['reranker']})")
    logger.info("=" * 56)
    return metrics, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--json", dest="json_path", default=None, help="指標を JSON で保存")
    ap.add_argument("--min-recall", type=float, default=None,
                    help="recall@k がこの値未満なら終了コード 1")
    args = ap.parse_args()

    metrics, details = run_rag_eval(top_k=args.top_k)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "details": details}, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved RAG eval metrics to {args.json_path}")

    if args.min_recall is not None and metrics["recall_at_k"] < args.min_recall:
        logger.warning(f"recall@{args.top_k} {metrics['recall_at_k']:.3f} < --min-recall {args.min_recall}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
