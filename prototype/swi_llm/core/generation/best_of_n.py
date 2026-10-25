"""
swi_llm/core/generation/best_of_n.py

PRM (Process Reward Model) を用いた Best-of-N 推論（ブループリント §8.2）。

学習済みの PRM ヘッド（各トークンにスカラー報酬を出力）を推論時に活用する:
  1. 同一プロンプトから N 個の候補を生成（多様性のため温度を上げる）。
  2. 各候補を PRM でスコアリング（回答トークンの平均報酬）。
  3. 最高スコアの候補を採用する。

固定モデルの性能上限を推論時計算で押し上げる「第3のスケーリング軸」(§8.4)。
"""
import torch

from swi_llm.utils.logger import setup_logger
from .base import generate

logger = setup_logger("generate_best_of_n")


def _text_model(model):
    """VLM ラッパの場合は内部のテキストモデル (prm_head 保持) を取り出す。"""
    return model.text_model if hasattr(model, "text_model") else model


def score_with_prm(model, tokenizer, prompt, completion):
    """PRM ヘッドで完了 (回答部分) の平均報酬を計算する。値が高いほど良い。"""
    tm = _text_model(model)
    if not hasattr(tm, "prm_head"):
        return float("-inf")
    device = next(tm.parameters()).device
    # L-9: encode(prompt+completion) は境界のサブワード結合で
    # encode(prompt)+encode(completion) と一致しないことがあり、prompt_len が回答部の
    # 開始位置とずれて PRM スコア範囲が数トークン汚染される。prompt と completion を
    # 別々にエンコードして連結し、境界（prompt_len）を明示管理する。
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    prompt_len = len(prompt_ids)
    if len(completion_ids) == 0:
        return float("-inf")

    # 学習時 seq_len を超えないよう安全側で切り詰め
    max_len = getattr(tm, "max_seq_len", 512)
    full_tokens = (prompt_ids + completion_ids)[:max_len]
    if len(full_tokens) <= prompt_len:
        return float("-inf")
    input_ids = torch.tensor([full_tokens], dtype=torch.long, device=device)

    tm.eval()
    with torch.no_grad():
        outputs = tm(input_ids, return_prm=True)
    prm_logits = outputs[-1]  # (1, seqlen) — return_prm=True, use_cache=False
    answer_scores = prm_logits[0, prompt_len:]
    if answer_scores.numel() == 0:
        return float("-inf")
    return answer_scores.float().mean().item()


def generate_best_of_n(model, tokenizer, prompt, n=4, max_new_tokens=256,
                       temperature=0.9, top_k=50, top_p=0.95, repetition_penalty=1.1,
                       images=None):
    """N 個の候補を生成し、PRM スコア最高の完了を返す（§8.2）。

    Returns:
        (best_full_text, candidates): best_full_text は prompt+最良完了。
        candidates は [(score, completion), ...]（降順）。
    """
    tm = _text_model(model)
    candidates = []
    for i in range(max(1, n)):
        full = generate(tm, tokenizer, prompt, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_k=top_k, top_p=top_p,
                        repetition_penalty=repetition_penalty, images=images)
        completion = full[len(prompt):]
        score = score_with_prm(model, tokenizer, prompt, completion)
        candidates.append((score, completion))
        logger.info(f"Best-of-N candidate {i + 1}/{n}: PRM score={score:.4f}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_completion = candidates[0]
    logger.info(f"Best-of-N selected candidate with score={best_score:.4f}")
    return prompt + best_completion, candidates
