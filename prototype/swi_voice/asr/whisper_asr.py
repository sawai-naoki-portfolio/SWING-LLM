"""
swi_voice/asr/whisper_asr.py
実 ASR: faster-whisper（CTranslate2 版 Whisper）の統合 [音声編10.1, fix 08 / voice roadmap step3]

既存の DI 契約 `.transcribe(audio) -> str`（asr_pseudo_label / eval.evaluate.transcription_wer /
eval.drift_monitor.DriftMonitor / agents.cascade_router.CascadeVoiceAgent が受け取る形）の
実装を提供する。これまでこの口に刺さる実モデルがプロジェクト内に存在しなかった。

- 入力波形は torch.Tensor / ndarray / list（[T] / [C,T] / [1,1,T]）を受け付け、既定で
  プロジェクト標準の 24kHz（VOICE_SAMPLE_RATE）とみなして内部で 16kHz へリサンプルする。
- モデルは遅延ロード。重みは初回のみ HF Hub から取得され、以後はローカルキャッシュ
  （ASCII パスの ~/.cache/huggingface）から読む。モデルサイズは SWI_ASR_MODEL（既定 small）。
- faster-whisper 未導入・ロード失敗・SWI_ASR=0 のときは get_default_asr() が None を返し、
  呼び出し側は従来どおり「ASR なし」で縮退する（C-4: オフラインでも機能を殺さない）。
"""
import os
import threading
import time

import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_SAMPLE_RATE, ASR_DISABLE_ENV, ASR_MODEL_ENV, ASR_DEFAULT_MODEL,
    ASR_SAMPLE_RATE, ASR_DEFAULT_LANGUAGE, ASR_BEAM_SIZE,
)

logger = setup_logger("voice.asr")


def asr_enabled() -> bool:
    """SWI_ASR=0 で実 ASR を明示的に無効化できる（既定は有効）。"""
    return os.environ.get(ASR_DISABLE_ENV, "1").strip().lower() not in ("0", "false", "no")


def asr_available() -> bool:
    """faster-whisper が import 可能かつ無効化されていないか（モデルロードはしない）。

    Web の status エンドポイント等、「重みを読み込まずに可用性だけ知りたい」用途向け。
    """
    if not asr_enabled():
        return False
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


class WhisperASR:
    """faster-whisper を `.transcribe(audio) -> str` 契約でラップする。

    Args:
        model_size: faster-whisper のモデルサイズ or ローカルパス
            （省略時 SWI_ASR_MODEL → 既定 "small"）。
        device: "cuda" / "cpu"（省略時は自動）。
        compute_type: CTranslate2 の量子化（省略時 cuda=float16 / cpu=int8）。
        language: 転写言語（既定 "ja"。fix 08 本番仕様の「日本語固定」）。
        beam_size: ビームサーチ幅。
    """

    def __init__(self, model_size=None, device=None, compute_type=None,
                 language=ASR_DEFAULT_LANGUAGE, beam_size=ASR_BEAM_SIZE):
        self.model_size = (model_size or os.environ.get(ASR_MODEL_ENV, "").strip()
                           or ASR_DEFAULT_MODEL)
        self.language = language
        self.beam_size = beam_size
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._load_lock = threading.Lock()

    def load(self):
        """モデルを（未ロードなら）ロードする。初回は HF Hub から重みを取得する。"""
        if self._model is not None:
            return self
        with self._load_lock:
            if self._model is not None:
                return self
            from faster_whisper import WhisperModel
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            compute = self._compute_type or ("float16" if device == "cuda" else "int8")
            logger.info(f"Loading faster-whisper '{self.model_size}' "
                        f"(device={device}, compute_type={compute})...")
            t0 = time.time()
            self._model = WhisperModel(self.model_size, device=device, compute_type=compute)
            logger.info(f"faster-whisper '{self.model_size}' ready in {time.time() - t0:.1f}s.")
        return self

    @staticmethod
    def _to_mono_f32(audio):
        """波形を [T] float32 Tensor へ正規化する。空・None は None を返す。"""
        if audio is None:
            return None
        if isinstance(audio, torch.Tensor):
            x = audio.detach().float()
        else:
            x = torch.as_tensor(audio, dtype=torch.float32)
        if x.numel() == 0:
            return None
        while x.dim() > 1 and x.shape[0] == 1:      # [1,1,T] → [T]
            x = x.squeeze(0)
        if x.dim() == 2:                            # [C,T] 多チャンネル → モノラル平均
            x = x.mean(dim=0)
        elif x.dim() > 2:
            logger.warning(f"WhisperASR: unexpected waveform shape {tuple(x.shape)}; "
                           "flattening. Pass a single utterance per call.")
            x = x.reshape(-1)
        return x

    def transcribe(self, audio, sample_rate=VOICE_SAMPLE_RATE, language=None) -> str:
        """音声波形をテキストへ転写する（DI 契約: `.transcribe(audio) -> str`）。

        Args:
            audio: 波形（Tensor/ndarray/list）。既定でプロジェクト標準 24kHz とみなす。
            sample_rate: 入力のサンプルレート。
            language: この呼び出しだけ転写言語を上書き（既定は self.language="ja"）。
        Returns:
            転写テキスト。空音声・無音は ""。
        """
        x = self._to_mono_f32(audio)
        if x is None:
            return ""
        if sample_rate != ASR_SAMPLE_RATE:
            import torchaudio.functional as AF
            x = AF.resample(x, int(sample_rate), ASR_SAMPLE_RATE)
        self.load()
        # faster-whisper の transcribe はスレッドセーフ（CTranslate2 が並列実行を扱う）だが、
        # segments はジェネレータで、消費し切るまで転写は完了しない。ここで join して返す。
        segments, _info = self._model.transcribe(
            x.cpu().numpy(), language=(language or self.language), beam_size=self.beam_size)
        return "".join(seg.text for seg in segments).strip()


_default_asr = None
_default_failed = False
_default_lock = threading.Lock()


def get_default_asr(eager=False):
    """既定 ASR（WhisperASR シングルトン）を返す。使えない環境では None（呼び出し側は縮退, C-4）。

    Args:
        eager: True なら重みロードまでこの場で行い、失敗したら None を返す
            （eval / pipeline_check 用。「後で毎サンプル例外」ではなく最初に 1 回で確定させる）。
            False なら import 可否のみ確認し、ロードは初回 transcribe まで遅延する（Web 用）。
    """
    global _default_asr, _default_failed
    if _default_failed or not asr_enabled():
        if not asr_enabled():
            logger.info(f"Real ASR disabled via {ASR_DISABLE_ENV}=0.")
        return None
    with _default_lock:
        if _default_failed:
            return None
        if _default_asr is None:
            if not asr_available():
                logger.warning("faster-whisper is not installed; ASR-dependent features "
                               "(WER eval, pseudo-labeling, cascade ASR) stay disabled (fix 08).")
                _default_failed = True
                return None
            _default_asr = WhisperASR()
        if eager:
            try:
                _default_asr.load()
            except Exception as e:
                logger.warning(f"faster-whisper model load failed ({type(e).__name__}: {e}); "
                               "continuing without real ASR (C-4).")
                _default_failed = True
                _default_asr = None
                return None
    return _default_asr


def reset_default_asr():
    """シングルトン状態を初期化する（テスト用）。"""
    global _default_asr, _default_failed
    with _default_lock:
        _default_asr = None
        _default_failed = False
