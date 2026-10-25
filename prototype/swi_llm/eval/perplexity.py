"""
swi_llm/eval/perplexity.py

パープレキシティ評価（Phase 2 / roadmap「perplexity + 日本語ベンチ」）。

言語モデルの基本の物差し。「次のトークンを平均で何択まで絞れているか」に相当し、低いほど良い
（学習前のランダム重みは語彙数 32768 前後、よく学習した日本語モデルは 1 桁〜十数）。
これまでコード上どこにも無く、数週間の学習で賢くなったかを数値で確かめられなかった。

測る対象:
  1. heldout: リポジトリ同梱の固定テキスト（ppl_heldout_ja.txt。どの学習データにも入っていない
     書き下ろし）。コーパスや予算が変わっても同じ文章で測るので、run を跨いで比較できる。
     ゲート（--max-ppl）はこちらに掛ける。
  2. corpus_val: トークン予算コーパス（token_corpus）の検証分割があれば、その先頭
     --max-corpus-tokens トークン。学習データと同じ分布での汎化を見る。
結果は errorLog/metrics/eval_ppl_<MODE>.jsonl に追記する（成長グラフの元データ）。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m swi_llm.eval.perplexity                 # 最新の pretrain ckpt
    .venv\\Scripts\\python.exe -m swi_llm.eval.perplexity --max-ppl 5000  # ゲート

終了コード: 評価できて閾値以内なら 0。閾値超過・評価自体の失敗（ckpt 破損など）は 1
（run_eval と違い、失敗を緑にしない: 監査 G-2）。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROTOTYPE_DIR not in sys.path:
    sys.path.append(PROTOTYPE_DIR)

from swi_llm.utils.logger import setup_logger

logger = setup_logger("eval.perplexity")

HELDOUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppl_heldout_ja.txt")
DEFAULT_MAX_CORPUS_TOKENS = 200_000


def heldout_tokens(tokenizer, eot_id, path=HELDOUT_PATH):
    """固定の評価テキストを段落（空行区切り）ごとにトークン化し、EOT で区切って連結する。"""
    with open(path, encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    ids = []
    for p in paragraphs:
        ids.extend(tokenizer.encode(p))
        ids.append(int(eot_id))
    return np.asarray(ids, dtype=np.int64)


def compute_perplexity(model, tokens, seq_len, batch_size=4, max_tokens=None, device="cpu"):
    """tokens（1 次元）を seq_len の重なり無し窓に切り、次トークン予測の平均 NLL と PPL を返す。

    末尾の端数窓（2 トークン以上）も評価に含める。予測したトークン数で重み付けした平均。
    """
    import torch
    import torch.nn.functional as F

    tokens = np.asarray(tokens[: (max_tokens + 1) if max_tokens else None], dtype=np.int64)
    if len(tokens) < 2:
        raise ValueError("Need at least 2 tokens to compute perplexity.")
    starts = list(range(0, len(tokens) - 1, seq_len))
    full = [s for s in starts if s + seq_len + 1 <= len(tokens)]
    tail = starts[len(full):]

    total_nll, total_count = 0.0, 0
    model.eval()

    def run(batch_starts, length):
        nonlocal total_nll, total_count
        x = torch.tensor(np.stack([tokens[s: s + length] for s in batch_starts]), device=device)
        y = torch.tensor(np.stack([tokens[s + 1: s + length + 1] for s in batch_starts]), device=device)
        with torch.no_grad():
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += float(nll)
        total_count += int(y.numel())

    for i in range(0, len(full), batch_size):
        run(full[i: i + batch_size], seq_len)
    for s in tail:
        run([s], len(tokens) - 1 - s)

    mean_nll = total_nll / total_count
    return {"tokens": total_count, "nll": round(mean_nll, 6), "ppl": round(math.exp(min(mean_nll, 700.0)), 4)}


def find_pretrain_checkpoint():
    """現在のモードの pretrain ckpt を優先し、無ければ後段ステージも含めた最新 ckpt。"""
    from swi_llm.eval.run_eval import _suffixes, find_latest_checkpoint
    from swi_llm.utils.constants import CKPT_PRETRAIN, EXT_PT
    for s in _suffixes():
        path = os.path.join(PROTOTYPE_DIR, f"{CKPT_PRETRAIN}{s}{EXT_PT}")
        if os.path.exists(path):
            return path
    return find_latest_checkpoint()


def evaluate(ckpt_path=None, max_corpus_tokens=DEFAULT_MAX_CORPUS_TOKENS, batch_size=4):
    import torch
    from swi_llm.data.tokenizer import get_tokenizer, EOT_TOKEN_ID
    from swi_llm.eval.run_eval import load_model
    from swi_llm.utils.config import TRAINING_CONFIG

    ckpt_path = ckpt_path or find_pretrain_checkpoint()
    model = load_model(ckpt_path)
    device = next(model.parameters()).device
    seq_len = TRAINING_CONFIG["pretrain"]["seq_len"]
    tokenizer = get_tokenizer()

    result = {"ckpt": ckpt_path, "seq_len": seq_len}
    if ckpt_path and os.path.exists(ckpt_path):
        result["ckpt_mtime"] = round(os.path.getmtime(ckpt_path), 3)
    result["heldout"] = compute_perplexity(model, heldout_tokens(tokenizer, EOT_TOKEN_ID), seq_len,
                                           batch_size=batch_size, device=device)

    try:
        from swi_llm.data.token_corpus import load_token_corpus
        loaded = load_token_corpus()
    except Exception as e:
        logger.warning(f"Token corpus unavailable for corpus_val perplexity ({e}).")
        loaded = None
    if loaded is not None and len(loaded[1]) > seq_len + 1:
        result["corpus_val"] = compute_perplexity(model, loaded[1], seq_len, batch_size=batch_size,
                                                  max_tokens=max_corpus_tokens, device=device)
    else:
        result["corpus_val"] = None     # 検証分割が無い（未構築 or 小さすぎる）: 学習データで代用しない

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Perplexity of the SWI-LLM on held-out text (Phase 2).")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: latest pretrain checkpoint)")
    ap.add_argument("--max-ppl", type=float, default=None,
                    help="exit 1 if held-out perplexity is above this (lower is better)")
    ap.add_argument("--max-corpus-tokens", type=int, default=DEFAULT_MAX_CORPUS_TOKENS)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--json", default=None, help="also write the result JSON to this path")
    args = ap.parse_args(argv)

    try:
        result = evaluate(args.ckpt, max_corpus_tokens=args.max_corpus_tokens, batch_size=args.batch_size)
    except Exception:
        logger.exception("Perplexity evaluation failed.")
        return 1

    from swi_llm.utils.metrics import MetricsLogger
    with MetricsLogger("eval_ppl") as metrics:
        metrics.log_eval(ckpt=result["ckpt"], ckpt_mtime=result.get("ckpt_mtime"),
                         heldout_ppl=result["heldout"]["ppl"], heldout_tokens=result["heldout"]["tokens"],
                         corpus_val_ppl=(result["corpus_val"] or {}).get("ppl"),
                         corpus_val_tokens=(result["corpus_val"] or {}).get("tokens"))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    cv = result["corpus_val"]
    logger.info(f"Perplexity: heldout={result['heldout']['ppl']:.2f} ({result['heldout']['tokens']} tokens)"
                + (f", corpus_val={cv['ppl']:.2f} ({cv['tokens']} tokens)" if cv else ", corpus_val=n/a")
                + f" [ckpt={result['ckpt']}]")
    if args.max_ppl is not None and result["heldout"]["ppl"] > args.max_ppl:
        logger.error(f"Held-out perplexity {result['heldout']['ppl']:.2f} > gate {args.max_ppl}.")
        return 1
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
    sys.exit(main())
