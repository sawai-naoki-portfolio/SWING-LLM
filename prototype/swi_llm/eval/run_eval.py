"""
swi_llm/eval/run_eval.py

小規模能力ベンチマークによる評価スイート（ブループリント §14）。

benchmark.jsonl の各問題をモデルに解かせ、検証器 (verifier.compute_reward) で
採点し、カテゴリ別正答率を報告する。改善（データミックス・QK-Norm・GRPO 等）が
実際に効いているかを測る「物差し」。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m swi_llm.eval.run_eval                # 最新チェックポイントを評価
    .venv\\Scripts\\python.exe -m swi_llm.eval.run_eval --max-new-tokens 128
    .venv\\Scripts\\python.exe -m swi_llm.eval.run_eval --json out.json

終了コード: 常に 0（評価は情報提供でありパイプラインを止めない）。
ただし --min-accuracy を指定した場合、総合正答率が閾値未満なら 1。
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
from swi_llm.utils.constants import (
    CKPT_GRPO, CKPT_DPO, CKPT_SFT, CKPT_PRETRAIN, EXT_PT,
)
from swi_llm.utils.chat_template import build_prompt
from swi_llm.data.verifier import compute_reward

logger = setup_logger("eval")

_BENCHMARK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.jsonl")
# 正答とみなす報酬の下限
CORRECT_THRESHOLD = 1.0
_CKPT_STAGE_PRIORITY = [CKPT_GRPO, CKPT_DPO, CKPT_SFT, CKPT_PRETRAIN]


def load_benchmark(path=_BENCHMARK_PATH):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _suffixes():
    from swi_llm.utils.config import config
    suffix = config.get("ckpt_suffix", "")
    out = [suffix]
    for fb in ["", "_SMOKE", "_TEST", "_MAX"]:
        if fb not in out:
            out.append(fb)
    return out


def find_latest_checkpoint():
    """モードサフィックス優先 → 学習ステージの新しい順で最初に存在する ckpt を返す。"""
    for s in _suffixes():
        for base in _CKPT_STAGE_PRIORITY:
            path = os.path.join(PROTOTYPE_DIR, f"{base}{s}{EXT_PT}")
            if os.path.exists(path):
                return path
    return None


def load_model(ckpt_path):
    import torch
    from swi_llm.utils.config import create_mini_llm, INFERENCE_CONFIG
    from swi_llm.data.tokenizer import VOCAB_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_mini_llm(vocab_size=VOCAB_SIZE, max_seq_len=INFERENCE_CONFIG["max_seq_len"]).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        sd = {k: v for k, v in sd.items() if k not in ("freqs_cos", "freqs_sin")}
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")
        del ckpt, sd
    else:
        logger.warning("No checkpoint found. Evaluating randomly-initialized model.")
    model.eval()
    return model


def run_eval(max_new_tokens=64):
    import torch
    from swi_llm.core.generation import generate
    from swi_llm.data.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    ckpt_path = find_latest_checkpoint()
    model = load_model(ckpt_path)

    items = load_benchmark()
    per_cat = {}
    details = []
    for item in items:
        cat = item.get("category", "general")
        prompt = build_prompt(item["question"])
        with torch.no_grad():
            output = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                              temperature=0.3, top_p=0.9)
        completion = output[len(prompt):]
        reward = compute_reward(item["question"], completion, item["answer"])
        correct = reward >= CORRECT_THRESHOLD
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][1] += 1
        if correct:
            per_cat[cat][0] += 1
        details.append({"category": cat, "question": item["question"],
                        "expected": item["answer"], "reward": reward, "correct": correct})

    total_correct = sum(v[0] for v in per_cat.values())
    total = sum(v[1] for v in per_cat.values())
    overall = total_correct / total if total else 0.0

    # 汚染検出 (§14.1): ベンチ問題が学習コーパスに混入していないか確認。
    # 混入があるとスコアが実力を過大評価するため警告する。
    contamination = _check_contamination(items)

    logger.info("=" * 50)
    logger.info(f"EVAL RESULTS (checkpoint: {os.path.basename(ckpt_path) if ckpt_path else 'random'})")
    logger.info("=" * 50)
    for cat, (c, n) in sorted(per_cat.items()):
        logger.info(f"  {cat:12} {c}/{n}  ({100.0 * c / n:.1f}%)")
    logger.info(f"  {'OVERALL':12} {total_correct}/{total}  ({100.0 * overall:.1f}%)")
    if contamination is not None:
        rate = contamination["contamination_rate"]
        msg = f"  {'CONTAM':12} {contamination['n_contaminated']}/{contamination['n_total']} benchmark items leaked into corpus ({100.0 * rate:.1f}%)"
        if contamination["n_contaminated"] > 0:
            logger.warning(msg + " — scores may be overestimated!")
        else:
            logger.info(msg)

    return {"overall": overall, "per_category": {k: v[0] / v[1] for k, v in per_cat.items()},
            "counts": per_cat, "details": details, "checkpoint": ckpt_path,
            "contamination": contamination}


def _check_contamination(items):
    """cleaned_wikipedia.jsonl に対しベンチ汚染を検出する。コーパス無ければ None。"""
    from swi_llm.eval.contamination import ContaminationChecker
    from swi_llm.utils.constants import DATASET_WIKIPEDIA_CLEANED
    corpus_path = os.path.join(PROTOTYPE_DIR, DATASET_WIKIPEDIA_CLEANED)
    if not os.path.exists(corpus_path):
        return None
    texts = []
    try:
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(json.loads(line).get("text", ""))
    except Exception as e:
        logger.warning(f"Could not load corpus for contamination check ({e}).")
        return None
    return ContaminationChecker(items).report(texts)


def main():
    parser = argparse.ArgumentParser(description="Run the SWI-LLM capability benchmark (§14).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--json", type=str, default=None, help="Write full results JSON to this path")
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="Exit 1 if overall accuracy is below this (default: never fail)")
    args = parser.parse_args()

    try:
        result = run_eval(max_new_tokens=args.max_new_tokens)
    except Exception:
        logger.exception("Evaluation failed.")
        # 評価自体の失敗はパイプラインを止めない（情報提供ステップ）
        sys.exit(0)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote results to {args.json}")

    if args.min_accuracy is not None and result["overall"] < args.min_accuracy:
        logger.warning(f"Overall accuracy {result['overall']:.3f} < threshold {args.min_accuracy}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
