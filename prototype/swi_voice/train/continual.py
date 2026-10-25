"""
swi_voice/train/continual.py
継続学習: LoRA ベース話者追加とフィードバック収集 [音声編9.1, 実装編9.1]

新しい話者・スタイルをベースモデルの重みを凍結したまま小さな LoRA アダプタとして
追加学習する。ゼロショットクローニング（model/cloning 相当, 音声編4.1）だけでは
声質再現度が不足する用途（頻繁に使う特定話者等）の実務的な補完手段になる。

LoRA 本体は自前実装せず `swi_llm.training.lora`（apply_lora/merge_all_lora/
lora_state_dict）をそのまま流用する。voice 側のモデル（ARCodecLM/HybridARFM/
SpeechToSpeechBackbone）はいずれも swi_llm の TransformerBlock を共有しており、
対象線形層名（wq/w_kv_up/w_kv_down/wo/w1/w2/w3）が一致するため、target_modules の
変更なしにそのまま適用できる（実装編9.1で示唆されている通り）。

併せて、本番運用フィードバック（「再生成」を暗黙的ネガティブシグナルとする）から
継続 DPO（train/dpo_mos.tts_dpo_loss）向けの (text_ids, chosen_codes, rejected_codes)
ペアを組み立てる `collect_voice_feedback` を提供する [音声編9.2]。
"""
import torch

from swi_llm.utils.logger import setup_logger
from swi_llm.training.lora import (
    apply_lora, merge_all_lora, lora_state_dict, DEFAULT_LORA_TARGETS,
)

logger = setup_logger("voice.continual")


def add_speaker_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.0,
                      targets=DEFAULT_LORA_TARGETS):
    """ベースモデルの重みを凍結し、新規話者用の LoRA アダプタを追加する [音声編9.1]。

    Returns:
        置き換えた線形層数（0 の場合は `targets` がモデルの層名と一致していない
        可能性が高いため警告する）。
    """
    n = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout, targets=targets)
    if n == 0:
        logger.warning("add_speaker_lora: no matching linear layers found; check `targets` "
                        "against the model's module names.")
    else:
        logger.info(f"Speaker LoRA adapter attached: {n} layers (rank={rank}, alpha={alpha}).")
    return n


def save_speaker_adapter(model, path: str):
    """LoRA パラメータのみを保存する（話者ごとの小さなアダプタファイル）。"""
    state = lora_state_dict(model)
    if not state:
        logger.warning(f"save_speaker_adapter: no LoRA parameters found in model "
                        f"(did you call add_speaker_lora first?); saving empty adapter to {path}.")
    torch.save(state, path)
    logger.info(f"Saved speaker LoRA adapter: {path} ({len(state)} tensors).")
    return path


def load_speaker_adapter(model, path: str, map_location="cpu"):
    """保存済み LoRA アダプタを、既に `add_speaker_lora` 適用済みのモデルへ読み込む。"""
    state = torch.load(path, map_location=map_location)
    missing, _unexpected = model.load_state_dict(state, strict=False)
    still_missing_lora = [k for k in missing if "lora_" in k]
    if still_missing_lora:
        logger.warning(f"load_speaker_adapter: {len(still_missing_lora)} LoRA tensors still missing "
                        f"after load; model may not have add_speaker_lora applied yet.")
    logger.info(f"Loaded speaker LoRA adapter: {path} ({len(state)} tensors).")
    return model


def merge_speaker_lora(model):
    """検証後、LoRA の寄与をベース重みへマージして通常の推論用モデルに戻す。"""
    return merge_all_lora(model)


def collect_voice_feedback(interactions, text_key="text_ids", chosen_key="final_codes",
                            rejected_key="first_codes", regenerated_key="regenerated"):
    """本番ログから「再生成された」やり取りを暗黙的ネガティブシグナルとして抽出し、
    継続 DPO（train/dpo_mos.tts_dpo_loss / dpo_with_mos_step）向けの
    (text_ids, chosen_codes, rejected_codes) ペアを組み立てる [音声編9.2]。

    Args:
        interactions: dict のリスト。各要素は `regenerated_key`（bool）と、
            `text_key`/`chosen_key`/`rejected_key` を持つことを想定する
            （再生成された場合、最終的に採用された音声=chosen、最初に生成され
            棄却された音声=rejected とみなす）。
        欠損・不正な要素は例外を出さず安全にスキップする（V-2: 入力検証）。

    Returns:
        {"text_ids", "chosen_codes", "rejected_codes"} の dict のリスト。
    """
    pairs = []
    if not interactions:
        return pairs

    skipped = 0
    for rec in interactions:
        try:
            if rec is None or not rec.get(regenerated_key):
                continue
            text_ids = rec.get(text_key)
            chosen = rec.get(chosen_key)
            rejected = rec.get(rejected_key)
            if text_ids is None or chosen is None or rejected is None:
                skipped += 1
                continue
            pairs.append({"text_ids": text_ids, "chosen_codes": chosen, "rejected_codes": rejected})
        except Exception:
            logger.exception("collect_voice_feedback: failed to process one interaction record; skipping.")
            skipped += 1

    if skipped:
        logger.info(f"collect_voice_feedback: skipped {skipped} malformed/incomplete records.")
    logger.info(f"collect_voice_feedback: collected {len(pairs)} continual-DPO pairs "
                f"from {len(interactions)} interactions.")
    return pairs
