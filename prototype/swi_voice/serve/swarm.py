"""
swi_voice/serve/swarm.py
デバイス×クラウド階層ルーティング (Voice Swarm) [音声編13.2]

ウェイクワード検出・簡単な定型応答はオンデバイス側（ここでは既存の学習済み TTS を
「デバイス側の小型モデル」役として流用）で即座に処理し、複雑な対話のみクラウド側
（cloud_agent、DI）へエスカレーションする。プライバシー（機微な音声データがデバイス外に
出る頻度を減らせる）・体感レイテンシ・運用コストの三方向で利点を持つ階層化設計。

on_device_tts は `.synthesize(text, **kwargs) -> bytes` を実装するオブジェクト
（`serve.tts.SWIVoiceTTS` 相当）、cloud_agent は `.handle(user_audio) -> ...` を実装する
オブジェクト（`agents.cascade_router.CascadeVoiceAgent`/`S2SVoiceAgent` 相当）を想定する。
"""
from swi_llm.utils.logger import setup_logger

logger = setup_logger("voice.swarm")

DEFAULT_WAKE_RESPONSE = "はい、ご用件をどうぞ"
DEFAULT_CONFIRMATION_RESPONSE = "承知しました"


class VoiceSwarmRouter:
    """デバイス側の小型モデルとクラウド側の大型モデルを使い分けるルーター [音声編13.2]。"""

    def __init__(self, wake_word_detector, on_device_tts, cloud_agent=None,
                 wake_response=DEFAULT_WAKE_RESPONSE,
                 confirmation_response=DEFAULT_CONFIRMATION_RESPONSE):
        self.wake_word = wake_word_detector
        self.on_device_tts = on_device_tts
        self.cloud_agent = cloud_agent
        self.wake_response = wake_response
        self.confirmation_response = confirmation_response

    def handle_frame(self, audio_frame, conversation_state=None):
        """1フレーム分の音声を処理する。

        優先順位: (1) ウェイクワード検出 → 即時オンデバイス応答、(2) 単純な確認応答
        （conversation_state["is_simple_confirmation"]）→ オンデバイス、(3) それ以外は
        cloud_agent へエスカレーション（未設定なら明示的にエラーにする＝silent failure回避）。
        """
        conversation_state = conversation_state or {}

        if self.wake_word is not None and self.wake_word.detect(audio_frame):
            logger.info("VoiceSwarmRouter: wake word detected; responding on-device.")
            return self.on_device_tts.synthesize(self.wake_response)

        if conversation_state.get("is_simple_confirmation"):
            logger.debug("VoiceSwarmRouter: simple confirmation handled on-device.")
            return self.on_device_tts.synthesize(self.confirmation_response)

        if self.cloud_agent is None:
            raise ValueError("VoiceSwarmRouter: no cloud_agent configured for complex requests.")
        logger.debug("VoiceSwarmRouter: escalating to cloud_agent.")
        return self.cloud_agent.handle(audio_frame)
