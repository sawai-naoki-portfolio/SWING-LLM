"""
swi_voice/safety/watermark.py

生成音声への電子透かし付与 (Audio Watermarking) [音声編8.2]。

拡散スペクトラム方式の実装:
  payload から決定的に生成した ±1 の擬似ノイズ列（PN 系列）を、微小振幅 α で波形へ
  加算する。PN 系列は音声本体と無相関なので可聴閾値以下（既定 α=0.006 ≒ -44dB）で
  埋め込め、検出時は同じ PN 系列との相関 s = mean(wav · pn) を取ることで
  「透かしの有無」と「ペイロード」を判定できる（s ≈ α で埋め込み検出、s ≈ 0 で非埋め込み）。

  生成音声が AI 由来であることの機械可読な証跡（来歴・悪用検知）を残すのが目的。
  堅牢な公開手法（AudioSeal 等）の代替となる自己完結・依存ゼロの実装。
"""
import hashlib

import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_WATERMARK_PAYLOAD, VOICE_WATERMARK_STRENGTH,
    VOICE_WATERMARK_DETECT_FACTOR, VOICE_WATERMARK_KNOWN_PAYLOADS,
)

logger = setup_logger("voice.watermark")


def _pn_sequence(payload: str, length: int, device, dtype):
    """payload から決定的に ±1 の擬似ノイズ列 [length] を生成する（CPU 生成→転送）。"""
    seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.rand(length, generator=g)
    pn = torch.where(r < 0.5, torch.tensor(-1.0), torch.tensor(1.0))
    return pn.to(device=device, dtype=dtype)


def apply_watermark(audio_waveform: torch.Tensor, payload: str = VOICE_WATERMARK_PAYLOAD,
                    strength: float = VOICE_WATERMARK_STRENGTH) -> torch.Tensor:
    """波形に拡散スペクトラム電子透かしを埋め込む。

    形状は不変（[..., T]）。α·PN を加算し [-1, 1] にクランプして返す。
    空/None はそのまま返す（呼び出し側の再生を壊さない）。
    """
    try:
        if audio_waveform is None or audio_waveform.numel() == 0:
            logger.warning("Empty audio_waveform passed to apply_watermark; returning as-is.")
            return audio_waveform
        T = audio_waveform.shape[-1]
        pn = _pn_sequence(payload, T, audio_waveform.device, audio_waveform.dtype)  # [T]（先頭次元へブロードキャスト）
        watermarked = audio_waveform + strength * pn
        watermarked = torch.clamp(watermarked, -1.0, 1.0)
        logger.debug(f"Applied watermark payload='{payload}' (T={T}, strength={strength}).")
        return watermarked
    except Exception:
        logger.exception("Failed to apply watermark to audio; returning original.")
        return audio_waveform


def _detection_statistic(audio_waveform: torch.Tensor, payload: str) -> float:
    """s = mean(wav · pn)。埋め込みなら ≈ strength、非埋め込みなら ≈ 0。"""
    T = audio_waveform.shape[-1]
    # マルチチャンネルは時間軸以外を平均して 1 系列に畳む（同一 PN を全チャンネルへ加算済み）。
    wav = audio_waveform.reshape(-1, T).mean(dim=0).float()
    pn = _pn_sequence(payload, T, wav.device, wav.dtype)
    return float((wav * pn).mean().item())


def detect_watermark(audio_waveform: torch.Tensor, payload: str = None) -> str:
    """波形から電子透かしを抽出する。検出できたペイロード文字列を返す（無ければ ""）。

    payload を指定するとそのペイロードのみを判定。None の場合は既知ペイロード集合を
    走査して最初に検出したものを返す。
    """
    try:
        if audio_waveform is None or audio_waveform.numel() == 0:
            logger.warning("Empty audio_waveform passed to detect_watermark.")
            return ""
        threshold = VOICE_WATERMARK_STRENGTH * VOICE_WATERMARK_DETECT_FACTOR
        candidates = [payload] if payload else list(VOICE_WATERMARK_KNOWN_PAYLOADS)
        best_payload, best_s = "", threshold
        for cand in candidates:
            s = _detection_statistic(audio_waveform, cand)
            if s >= best_s:
                best_payload, best_s = cand, s
        if best_payload:
            logger.debug(f"Watermark detected: payload='{best_payload}' (stat={best_s:.5f}).")
        return best_payload
    except Exception:
        logger.exception("Failed to detect watermark in audio.")
        return ""
