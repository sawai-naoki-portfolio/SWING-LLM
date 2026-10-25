"""
swi_voice/duplex/barge_in.py
割り込み処理 (Barge-in handling) [音声編10.2, fix 13(2)]
"""
import threading
from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_SAMPLE_RATE, VAD_PROB_ON, VAD_PROB_OFF, VAD_MIN_ACTIVE_CHUNKS,
)

logger = setup_logger(__name__)


class BargeInDetector:
    """ヒステリシス付き barge-in 判定（fix 13(2): 誤 barge-in の抑制）。

    チャンクごとの発話確率（duplex/vad.speech_probability）に対して
      - 発話開始: 確率 >= on_prob が min_active_chunks 回**連続**したとき
      - 発話終了: 確率 <= off_prob になったとき
    の 2 閾値 + 最小継続時間で判定する。単発ノイズ（咳・物音の 1 チャンク）では
    発火せず、発話中の短い揺らぎ（on と off の間の確率）では解除されない。

    silero 不在の RMS 縮退時は確率が {0.0, 1.0} の二値になるが、
    「連続 min_active_chunks 回」の条件はそのまま機能する（C-4）。

    状態を持つため、セッション（WebSocket 接続）ごとに 1 つ作ること
    （SessionStopSignal と同じ方針。グローバル共有はセッション間で干渉する）。

    Args:
        probability_fn: `callable(chunk) -> float`。テストや別 VAD の注入用。
            省略時は duplex/vad.speech_probability（sample_rate/backend を束縛）。
    """

    def __init__(self, on_prob=VAD_PROB_ON, off_prob=VAD_PROB_OFF,
                 min_active_chunks=VAD_MIN_ACTIVE_CHUNKS,
                 sample_rate=VOICE_SAMPLE_RATE, backend=None, probability_fn=None):
        if off_prob > on_prob:
            raise ValueError(f"off_prob ({off_prob}) must be <= on_prob ({on_prob}) "
                             "for hysteresis to work.")
        if min_active_chunks < 1:
            raise ValueError("min_active_chunks must be >= 1.")
        self.on_prob = on_prob
        self.off_prob = off_prob
        self.min_active_chunks = min_active_chunks
        if probability_fn is None:
            from swi_voice.duplex.vad import speech_probability
            probability_fn = lambda chunk: speech_probability(
                chunk, sample_rate=sample_rate, backend=backend)
        self._probability_fn = probability_fn
        self._consecutive_active = 0
        self._speaking = False

    def reset(self):
        """状態を初期化する（TTS の発話ターンを跨ぐときに呼ぶ）。"""
        self._consecutive_active = 0
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def update(self, audio_chunk) -> bool:
        """音声チャンクを 1 つ観測し、「ユーザーが発話中（= barge-in すべき）」かを返す。"""
        try:
            p = float(self._probability_fn(audio_chunk))
        except Exception:
            logger.exception("BargeInDetector: probability computation failed; treating as silence.")
            p = 0.0

        if self._speaking:
            if p <= self.off_prob:
                self._speaking = False
                self._consecutive_active = 0
        else:
            if p >= self.on_prob:
                self._consecutive_active += 1
                if self._consecutive_active >= self.min_active_chunks:
                    self._speaking = True
                    logger.info(f"Barge-in: speech onset detected "
                                f"(prob={p:.3f}, chunks={self._consecutive_active}).")
            else:
                self._consecutive_active = 0
        return self._speaking


class SessionStopSignal:
    """セッション（WebSocket 接続）ごとの停止シグナル (V-4)。

    グローバル状態を共有しないため、あるクライアントの "stop" が同時に生成中の他クライアントの
    TTS を止めてしまう干渉や、別クライアントの clear が停止を解除してしまうレースが起きない。
    LLM 側の `generate_stream(stop_event=...)`（リクエスト単位）と同じ方針に揃える。
    """
    def __init__(self):
        self._event = threading.Event()

    def set(self):
        logger.info("Barge-in: stop signal set (session-scoped).")
        self._event.set()

    def clear(self):
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()


# --- 後方互換のグローバル API（非推奨）。単一プロセスの旧デモ用途のみ。---
# 新規コード（統合サーバの WebSocket）はセッション単位の SessionStopSignal を使うこと。
_stop_event = threading.Event()

def set_stop_signal():
    """[非推奨] グローバル停止シグナルを送信する。マルチユーザーでは干渉するため
    SessionStopSignal を使うこと (V-4)。"""
    logger.info("Barge-in: Stop signal received (global, deprecated).")
    _stop_event.set()

def clear_stop_signal():
    """[非推奨] グローバル停止シグナルをクリアする。"""
    _stop_event.clear()

def check_stop_signal() -> bool:
    """[非推奨] グローバル停止シグナルが立っているか確認する。"""
    return _stop_event.is_set()

def handle_barge_in(audio_chunk, backend=None):
    """ユーザー音声チャンクを監視し、発話（バージイン）を検出したら True を返す。

    V-6: 従来は常に False を返すプレースホルダだったが、実装済みの VAD
    （duplex/vad.detect_voice_activity）に接続し、実際に発話区間を判定する。
    fix 13: VAD は silero-vad 導入済みならニューラル判定に自動昇格する（`backend` で固定可）。
    単発チャンクの無状態判定なので、ストリーミングでの誤発火抑制には
    ヒステリシス付きの BargeInDetector を使うこと。
    """
    try:
        if audio_chunk is None:
            logger.warning("Empty audio chunk provided for barge-in detection.")
            return False
        from swi_voice.duplex.vad import detect_voice_activity
        return bool(detect_voice_activity(audio_chunk, backend=backend))
    except Exception:
        logger.exception("Failed during barge-in handling.")
        return False
