"""
swi_voice/duplex/wake_word.py
ウェイクワード検出 (Wake Word Detection) [音声編13.2]

正規化相互相関（Normalized Cross-Correlation）でテンプレート音声との類似度を計算する、
依存ゼロの最小-本物実装。テンプレート（ユーザーのウェイクワード発話サンプル等）を保持し、
ストリーミング音声をスライディングウィンドウで走査して相関ピークが閾値を超えたら検出とする。

テンプレートと各ウィンドウを振幅正規化してから内積を取ることで、音量差の影響を抑える
（`safety/watermark.py` の相関検出と同種の設計）。実運用では小型ニューラル KWS モデルへ
置換可能（`detect(audio_frame) -> bool` のインターフェイスのみ固定すれば差し替え自由）。
"""
import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.tensor_guard import ensure_waveform

logger = setup_logger("voice.wake_word")

WAKE_WORD_THRESHOLD = 0.6
WAKE_WORD_SCAN_HOP = 160


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().reshape(-1)
    norm = torch.linalg.norm(x)
    if norm < 1e-8:
        return x
    return x / norm


class WakeWordDetector:
    """テンプレート音声との正規化相互相関でウェイクワードを検出する。"""

    def __init__(self, template, threshold: float = WAKE_WORD_THRESHOLD,
                 hop: int = WAKE_WORD_SCAN_HOP):
        # V-14: ndarray/list のテンプレートも受け付ける（numel() 前提で落とさない）。
        template = ensure_waveform(template, name="WakeWordDetector.template")
        if template is None:
            raise ValueError("WakeWordDetector requires a non-empty template waveform.")
        self.template = _normalize(template)
        self.template_len = self.template.numel()
        self.threshold = threshold
        self.hop = max(1, hop)

    def _best_score(self, audio: torch.Tensor) -> float:
        # V-15: ウィンドウごとの _normalize（コピー + ノルム + float() 同期）をやめ、
        # unfold でビュー化した全ウィンドウを一括計算する。Python ループと GPU→CPU 同期が
        # 各 1 回になる（旧実装はウィンドウ数≈376 回/detect の再計算をしていた）。
        audio = audio.detach().float().reshape(-1)
        if audio.numel() < self.template_len:
            return 0.0
        windows = audio.unfold(0, self.template_len, self.hop)   # [n_win, L]（ビュー、コピーなし）
        norms = windows.norm(dim=1).clamp_min(1e-8)              # 無音ウィンドウのゼロ除算防止
        scores = (windows @ self.template) / norms               # template は __init__ で正規化済み
        # 旧実装は best=0.0 起点のため負の相関は 0.0 に丸められていた。挙動を維持する。
        return max(0.0, float(scores.max()))

    def detect(self, audio_frame) -> bool:
        """audio_frame にウェイクワードが含まれるか。

        テンプレート長未満の短いフレームや空/None/非 Tensor 入力は「検出不能」として安全に
        False を返す（例外を投げて呼び出し側の音声ループを止めない）。
        """
        try:
            # V-14: ndarray/list は Tensor へ正規化し、変換不能・空なら False（ホットパスで
            # スタックトレースを毎フレーム吐かない。警告は tensor_guard 側で型ごとに 1 回だけ）。
            frame = ensure_waveform(audio_frame, name="WakeWordDetector.audio_frame")
            if frame is None:
                return False
            score = self._best_score(frame)
            active = score >= self.threshold
            logger.debug(f"WakeWordDetector: score={score:.3f} threshold={self.threshold:.3f} -> {active}")
            return active
        except Exception:
            logger.exception("Error during wake-word detection")
            return False
