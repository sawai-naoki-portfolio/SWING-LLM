"""
swi_voice/data/quality_filter.py
学習クリップの音質フィルタ（SNR）[音声編1.1, NEW-V1 / 02_architecture_docs_gap_audit]

設計（03_Voice_Synthesis_Architecture §1）は「SNR（信号対雑音比）による低品質音源の除外」を
要求するが、従来のフィルタは長さ（0.5〜15 秒）だけで、雑音まみれ・無音同然のクリップが
学習に素通しだった（コーデック品質の上流要因）。

方式（依存ゼロ・百分位ベース）:
  クリップをフレーム（既定 30ms）に分割し、フレームパワーの下位分位（既定 5%）を
  ノイズフロア、上位分位（既定 90%）を音声レベルとみなして比を dB にする。
  通常の発話クリップは無音区間（呼吸・語間）を含むため下位分位が実際の録音ノイズを捉える。
  全編ノイズ・無音のクリップは上下の分位が近づき ~0 dB になる。

実測（2026-07-16, common_voice/reazon/jsut 実クリップ）: 実クリップ 16〜59 dB /
ノイズのみ ~0.6 dB / 強ノイズ付加 3.5〜14 dB。既定閾値 10 dB は実クリップを落とさない。

限界: 無音を全く含まない密な発話では下位分位に音声が混ざり SNR を過小評価する
（既定閾値を保守的な 10 dB に置く理由）。将来ニューラル VAD（fix 13 で導入済みの
silero）で音声/非音声フレームを分けて測る精緻化ができる。
"""
import math
import os

import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_SAMPLE_RATE, VOICE_MIN_SNR_DB, VOICE_MIN_SNR_ENV,
    VOICE_SNR_FRAME_MS, VOICE_SNR_NOISE_PCT, VOICE_SNR_SPEECH_PCT, VOICE_SNR_MAX_DB,
)

logger = setup_logger("voice.quality")

_EPS = 1e-10


def estimate_snr_db(wav, sample_rate=VOICE_SAMPLE_RATE, frame_ms=VOICE_SNR_FRAME_MS,
                    noise_pct=VOICE_SNR_NOISE_PCT, speech_pct=VOICE_SNR_SPEECH_PCT,
                    cap_db=VOICE_SNR_MAX_DB):
    """クリップの SNR [dB] を百分位方式で推定する。推定不能（短すぎる等）なら None。"""
    if wav is None:
        return None
    x = wav.detach().reshape(-1).float()
    frame = max(1, int(sample_rate * frame_ms / 1000))
    if x.numel() < frame * 4:                     # 分位が意味を持つ最低限のフレーム数
        return None
    n = x.numel() // frame
    power = x[: n * frame].reshape(n, frame).pow(2).mean(dim=1)
    noise = torch.quantile(power, noise_pct).item() + _EPS
    speech = torch.quantile(power, speech_pct).item() + _EPS
    return min(cap_db, 10.0 * math.log10(speech / noise))


def min_snr_threshold():
    """有効な SNR 閾値 [dB] を返す。無効化（SWI_VOICE_MIN_SNR=off）なら None。"""
    raw = os.environ.get(VOICE_MIN_SNR_ENV, "").strip().lower()
    if raw in ("off", "0", "false", "no", "none"):
        return None
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning(f"{VOICE_MIN_SNR_ENV}='{raw}' を数値として解釈できません。"
                           f"既定 {VOICE_MIN_SNR_DB} dB を使います。")
    return VOICE_MIN_SNR_DB


def passes_snr(wav, sample_rate=VOICE_SAMPLE_RATE, threshold=None):
    """SNR フィルタを通過するなら True。(推定不能なクリップは通過扱い = 長さフィルタに委ねる)。

    戻り値 (passed: bool, snr_db: float | None)。
    """
    thr = min_snr_threshold() if threshold is None else threshold
    if thr is None:
        return True, None
    snr = estimate_snr_db(wav, sample_rate=sample_rate)
    if snr is None:
        return True, None
    return snr >= thr, snr
