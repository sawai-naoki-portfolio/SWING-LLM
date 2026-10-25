"""
swi_voice/duplex/vad.py
発話区間検出 (Voice Activity Detection) [音声編10.2, fix 13]

2 つのバックエンドを持つ:
  - silero: ニューラル VAD（Silero VAD, JIT ~2MB, パッケージ同梱でオフライン可）。
    フレームごとの発話「確率」を返すため、騒音・非音声（咳/物音/BGM/トーン）と
    発話を区別できる。導入済みなら既定で選択される（fix 13 本番仕様）。
  - rms   : RMS エネルギー閾値の最小実装（依存ゼロ）。silero-vad 未導入・ロード失敗時の
    フォールバック（C-4: オフラインでも機能を殺さない）。原理的限界: 音量しか見ないため
    騒音下で誤発火し、音声/非音声を区別できない。

バックエンドは環境変数 SWI_VAD_BACKEND（auto|silero|rms, 既定 auto）または各関数の
`backend` 引数で選ぶ。`detect_voice_activity(chunk) -> bool` の契約は従来と同一。
"""
import io
import os
import threading

import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_SAMPLE_RATE, VAD_BACKEND_ENV, VAD_SILERO_SAMPLE_RATE,
    VAD_SILERO_WINDOW, VAD_SPEECH_PROB_THRESHOLD,
)

logger = setup_logger(__name__)

# 正規化音声([-1,1])に対する RMS 閾値。発話 RMS は概ね 0.05〜0.3、無音は ~0。
VAD_RMS_THRESHOLD = 0.02


def compute_rms_energy(audio_stream: torch.Tensor) -> float:
    """音声テンソルの RMS（二乗平均平方根）エネルギーを返す。"""
    x = audio_stream.detach().float().reshape(-1)
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x * x) + 1e-12))


class SileroVAD:
    """Silero VAD (JIT) のラッパー。波形チャンクの発話確率を返す。

    - モデルは silero_vad パッケージ同梱の JIT を **bytes 経由**（BytesIO）でロードする。
      パス文字列を直接 torch.jit.load へ渡すと、非 ASCII を含むインストールパス
      （例: OneDrive「デスクトップ」配下の .venv）で libtorch の fopen が
      errno 42 (Illegal byte sequence) で失敗する（2026-07-16 実測）。
    - モデルは LSTM 状態を持つ。呼び出しごとに reset_states() し、独立したチャンク判定に
      する（ストリーミングでの文脈保持は barge_in.BargeInDetector のヒステリシスが担う）。
    - 推論はロックで直列化する（JIT モデルの内部状態が並行 WS セッションで競合しないように）。
    """

    def __init__(self, jit_model):
        self._model = jit_model
        self._lock = threading.Lock()

    @torch.no_grad()
    def speech_probability(self, audio_stream, sample_rate=VOICE_SAMPLE_RATE) -> float:
        """波形全体の発話確率 = 512 サンプル窓 @16kHz ごとの確率の最大値。

        「チャンク内のどこかに発話が始まっていれば検知したい」（barge-in / 切り出しの用途）
        ため、平均ではなく最大を取る。窓に満たない末尾は捨て、全体が 1 窓未満なら
        ゼロパディングして 1 窓だけ評価する。
        """
        x = audio_stream.detach().float().reshape(-1)
        if x.numel() == 0:
            return 0.0
        if sample_rate != VAD_SILERO_SAMPLE_RATE:
            import torchaudio.functional as AF
            x = AF.resample(x, sample_rate, VAD_SILERO_SAMPLE_RATE)
        if x.numel() < VAD_SILERO_WINDOW:
            x = torch.nn.functional.pad(x, (0, VAD_SILERO_WINDOW - x.numel()))
        best = 0.0
        with self._lock:
            self._model.reset_states()
            for start in range(0, x.numel() - VAD_SILERO_WINDOW + 1, VAD_SILERO_WINDOW):
                p = float(self._model(x[start:start + VAD_SILERO_WINDOW],
                                      VAD_SILERO_SAMPLE_RATE).item())
                if p > best:
                    best = p
        return best


_silero_singleton = None
_silero_failed = False
_silero_init_lock = threading.Lock()


def get_silero_vad():
    """SileroVAD のシングルトンを返す。未導入・ロード失敗なら None（以後は再試行しない）。"""
    global _silero_singleton, _silero_failed
    if _silero_singleton is not None:
        return _silero_singleton
    if _silero_failed:
        return None
    with _silero_init_lock:
        if _silero_singleton is not None or _silero_failed:
            return _silero_singleton
        try:
            import silero_vad
            pkg_dir = os.path.dirname(os.path.abspath(silero_vad.__file__))
            jit_path = os.path.join(pkg_dir, "data", "silero_vad.jit")
            with open(jit_path, "rb") as f:
                model = torch.jit.load(io.BytesIO(f.read()), map_location="cpu")
            model.eval()
            _silero_singleton = SileroVAD(model)
            logger.info("Silero VAD loaded (neural backend enabled, fix 13).")
        except Exception as e:
            _silero_failed = True
            logger.warning(f"Silero VAD unavailable ({type(e).__name__}: {e}); "
                           "falling back to RMS energy VAD (fix 13).")
    return _silero_singleton


def resolve_backend(backend=None) -> str:
    """VAD バックエンド名を確定する（引数 > 環境変数 SWI_VAD_BACKEND > auto）。"""
    b = (backend or os.environ.get(VAD_BACKEND_ENV, "") or "auto").strip().lower()
    if b not in ("auto", "silero", "rms"):
        logger.warning(f"Unknown VAD backend '{b}'; using 'auto'.")
        b = "auto"
    return b


def _neural_vad_for(backend: str):
    """指定バックエンドでニューラル VAD を使うべきなら SileroVAD を返す（無ければ None）。"""
    if backend not in ("auto", "silero"):
        return None
    vad = get_silero_vad()
    if vad is None and backend == "silero":
        logger.warning("VAD backend 'silero' was requested but silero-vad is unavailable; "
                       "using RMS fallback.")
    return vad


def speech_probability(audio_stream, sample_rate=VOICE_SAMPLE_RATE, backend=None,
                       rms_threshold=VAD_RMS_THRESHOLD) -> float:
    """チャンクの発話確率 [0,1] を返す（barge_in.BargeInDetector のヒステリシス判定が共用）。

    silero バックエンドでは較正された確率、rms 縮退時は閾値判定を {0.0, 1.0} に写した
    二値プロキシを返す（ヒステリシスは退化するが契約は保たれる）。
    """
    try:
        if audio_stream is None or (hasattr(audio_stream, "numel") and audio_stream.numel() == 0):
            return 0.0
        vad = _neural_vad_for(resolve_backend(backend))
        if vad is not None:
            return vad.speech_probability(audio_stream, sample_rate=sample_rate)
        return 1.0 if compute_rms_energy(audio_stream) >= rms_threshold else 0.0
    except Exception:
        logger.exception("Error during VAD probability computation")
        return 0.0


def detect_voice_activity(audio_stream: torch.Tensor, threshold: float = VAD_RMS_THRESHOLD,
                          backend=None, prob_threshold: float = VAD_SPEECH_PROB_THRESHOLD,
                          sample_rate: int = VOICE_SAMPLE_RATE) -> bool:
    """ストリーミング音声チャンクに発話があるかを判定する（契約は従来と同一）。

    Args:
        audio_stream: [T] または [.., T] の音声波形テンソル（[-1,1] 正規化想定）。
        threshold: RMS バックエンド時の RMS しきい値（silero 時は使わない）。
        backend: "auto"（既定）/ "silero" / "rms"。None なら SWI_VAD_BACKEND → auto。
        prob_threshold: silero バックエンド時の発話確率しきい値。
        sample_rate: 入力波形のサンプルレート（既定はプロジェクト標準 24kHz）。
    Returns:
        発話が検出されたら True。
    """
    try:
        if audio_stream is None or audio_stream.numel() == 0:
            return False
        vad = _neural_vad_for(resolve_backend(backend))
        if vad is not None:
            p = vad.speech_probability(audio_stream, sample_rate=sample_rate)
            active = p >= prob_threshold
            logger.debug(f"VAD(silero): prob={p:.3f} threshold={prob_threshold:.2f} -> active={active}")
            return bool(active)
        energy = compute_rms_energy(audio_stream)
        active = energy >= threshold
        logger.debug(f"VAD(rms): rms={energy:.4f} threshold={threshold:.4f} -> active={active}")
        return bool(active)
    except Exception:
        logger.exception("Error during VAD processing")
        return False
