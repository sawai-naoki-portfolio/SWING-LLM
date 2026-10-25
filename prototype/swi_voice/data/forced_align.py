"""
swi_voice/data/forced_align.py
強制アライメント (Forced Alignment) [音声編1.2]

テキストと音声のフレーム単位アライメント（各テキスト単位が音声のどのフレーム区間に
対応するか）を推定する。継続時間モデル（model/duration.py）の教師信号や、ASR疑似
ラベリング（asr_pseudo_label.py）と並ぶデータ品質フィルタとして使う。

2 通りの経路を持つ:
  1. `aligner_model` を渡した場合: それに委譲する（Montreal Forced Aligner /
     wav2vec2 CTC アライナー等、外部の音響モデルを想定した DI）。想定インターフェイス
     は `aligner_model.align(audio, text) -> object`（`max_gap_seconds` 属性を持つ）。
     asr_pseudo_label.py の `asr_model` 注入と同じ設計思想で、これが「本物」の経路。
  2. `aligner_model` を渡さない場合: 依存ゼロの RMS エネルギーベースの最小実装へ
     フォールバックする。無音区間をトリムして発話区間を検出し、テキスト単位（文字/
     単語）へ長さ比例で単調に配分する近似アライメントである。エネルギーは音素境界の
     厳密な代理ではないため、フル版（wav2vec2/MFA 等の音響モデルによる真のアライメ
     ント）の代わりにはならない点に注意（`codec/semantic_acoustic.py` の意味トークン
     proxy と同種の「最小実装」）。
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.duplex.vad import compute_rms_energy, VAD_RMS_THRESHOLD
from swi_voice.utils.constants import VOICE_SAMPLE_RATE, VOICE_CODEC_HOP

logger = setup_logger("voice.forced_align")


@dataclass
class UnitAlignment:
    text: str
    start_frame: int
    end_frame: int          # exclusive
    start_sec: float
    end_sec: float


@dataclass
class Alignment:
    units: List[UnitAlignment] = field(default_factory=list)
    frame_rate: float = 0.0
    max_gap_seconds: float = 0.0


def _split_units(text: str):
    """整列対象のテキスト単位に分割する。空白があれば単語単位、無ければ文字単位
    （日本語等の CJK 向けフォールバック）。空白のみの断片は捨てる。"""
    text = (text or "").strip()
    if not text:
        return []
    if " " in text or "\t" in text:
        return [w for w in text.split() if w]
    return list(text)


def _frame_energies(audio: torch.Tensor, hop: int) -> torch.Tensor:
    """波形 [T] を hop サンプルごとに区切り、フレームごとの RMS エネルギー [n_frames] を返す。"""
    x = audio.detach().float().reshape(-1)
    n_frames = max(1, x.shape[0] // hop)
    energies = torch.zeros(n_frames)
    for i in range(n_frames):
        frame = x[i * hop:(i + 1) * hop]
        energies[i] = compute_rms_energy(frame)
    return energies


def _voiced_span(energies: torch.Tensor, threshold: float):
    """先頭/末尾の無音フレームをトリムし、発話区間 [start, end) を返す（無ければ None）。"""
    active = (energies >= threshold).nonzero(as_tuple=True)[0]
    if active.numel() == 0:
        return None
    return int(active[0].item()), int(active[-1].item()) + 1


def _distribute_frames(units: List[str], start: int, end: int) -> List[UnitAlignment]:
    """[start, end) のフレーム区間を、各単位の文字数に比例して単調に配分する
    （最大剰余法で合計が区間長と厳密に一致するようにする）。"""
    span = end - start
    weights = [max(1, len(u)) for u in units]
    total_w = sum(weights)
    raw = [span * w / total_w for w in weights]
    base = [int(r) for r in raw]
    remainder = span - sum(base)
    # 端数の大きい順に 1 フレームずつ配る（largest remainder method）
    order = sorted(range(len(units)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:max(0, remainder)]:
        base[i] += 1

    result = []
    cursor = start
    for text, n in zip(units, base):
        s, e = cursor, cursor + max(0, n)
        result.append((text, s, e))
        cursor = e
    return result


def align_text_and_audio(text: str, audio, aligner_model=None,
                          sample_rate: int = VOICE_SAMPLE_RATE, hop: int = VOICE_CODEC_HOP,
                          max_gap_seconds: float = 2.0) -> Optional[Alignment]:
    """テキストと音声のアライメント情報を返す。

    Args:
        text: 発話内容のテキスト。
        audio: 波形（torch.Tensor 相当、[-1,1] 正規化想定）。
        aligner_model: 指定時はこれに委譲する（`.align(audio, text)` を実装するオブジェクト）。
        sample_rate/hop: フォールバック経路でのフレーミングパラメータ（既定はコーデックと同じ
            50Hz フレームレート）。
        max_gap_seconds: アライメント結果の無音ギャップ（先頭/末尾トリム量の最大）がこれを
            超える場合は、ミスアライメントとみなして None を返す [音声編1.2]。

    Returns:
        Alignment、またはアライメント不能/著しいギャップがある場合は None。
    """
    if not (text or "").strip():
        logger.warning("align_text_and_audio: empty text; cannot align.")
        return None
    if audio is None or (hasattr(audio, "numel") and audio.numel() == 0):
        logger.warning("align_text_and_audio: empty/missing audio; cannot align.")
        return None

    if aligner_model is not None:
        try:
            alignment = aligner_model.align(audio, text)
        except Exception:
            logger.exception("align_text_and_audio: aligner_model.align raised; treating as failed alignment.")
            return None
        if alignment is None:
            return None
        if getattr(alignment, "max_gap_seconds", 0.0) > max_gap_seconds:
            logger.info(f"align_text_and_audio: aligner_model gap "
                        f"{alignment.max_gap_seconds:.2f}s > {max_gap_seconds:.2f}s; discarding pair.")
            return None
        return alignment

    # --- フォールバック: RMS エネルギーベースの最小実装（依存ゼロ） ---
    audio_t = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio, dtype=torch.float32)
    energies = _frame_energies(audio_t, hop)
    span = _voiced_span(energies, VAD_RMS_THRESHOLD)
    if span is None:
        logger.warning("align_text_and_audio: audio is entirely silent; cannot align.")
        return None
    start, end = span
    n_frames = energies.shape[0]
    frame_rate = sample_rate / hop
    leading_gap = start / frame_rate
    trailing_gap = (n_frames - end) / frame_rate
    gap = max(leading_gap, trailing_gap)
    if gap > max_gap_seconds:
        logger.info(f"align_text_and_audio: leading/trailing silence gap {gap:.2f}s > "
                    f"{max_gap_seconds:.2f}s; discarding pair (likely misaligned).")
        return None

    units = _split_units(text)
    if not units:
        logger.warning("align_text_and_audio: text has no alignable units after splitting.")
        return None
    if end <= start:
        logger.warning("align_text_and_audio: voiced span is empty after trimming; cannot align.")
        return None

    unit_alignments = [
        UnitAlignment(text=u, start_frame=s, end_frame=e,
                       start_sec=s / frame_rate, end_sec=e / frame_rate)
        for u, s, e in _distribute_frames(units, start, end)
    ]
    return Alignment(units=unit_alignments, frame_rate=frame_rate, max_gap_seconds=gap)
