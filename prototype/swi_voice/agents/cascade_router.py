"""
swi_voice/agents/cascade_router.py
カスケード/S2S使い分けルーター (Cascade Router) [音声編10.1, 10.4]

音声対話エージェントの2方式を実装する:
  - CascadeVoiceAgent: ASR→LLM→TTS のカスケード方式。各段を DI で受け取り、
    正確性・監査可能性が要求される目標指向型の対話に向く [音声編10.1]。
  - S2SVoiceAgent: 学習済み `duplex.s2s_backbone.SpeechToSpeechBackbone` をラップした
    ネイティブ音声間方式。テキストを介さず低遅延・親しみやすい相槌に向く。

AgentRouter は、両エージェントと intent_classifier を注入すると発話内容に応じて動的に
経路を切り替える [音声編10.4]。cascade_agent/s2s_agent/intent_classifier のいずれかが
未指定の場合は、常に実装済みの "cascade" 経路を返す確定ルーティングにフォールバックする
（後方互換: 呼び出し側が何も注入しなくても壊れない）。
"""
import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.tensor_guard import ensure_waveform

logger = setup_logger(__name__)

ROUTE_CASCADE = "cascade"
ROUTE_S2S = "s2s"


class CascadeVoiceAgent:
    """ASR→LLM→TTS カスケード方式 [音声編10.1, 実装編8.2]。

    Args:
        asr_model: `.transcribe(audio) -> str` を実装するオブジェクト
            （`data.asr_pseudo_label` と同じ DI インターフェイス。Whisper 等を想定）。
        llm_generate_fn: `callable(text: str) -> str`。`swi_llm.core.generation.base.generate`
            をモデル/トークナイザで束縛したクロージャを渡す想定（実推論経路をそのまま利用）。
        tts_model: `.synthesize(text, **kwargs) -> bytes` を実装するオブジェクト
            （`serve.tts.SWIVoiceTTS` と同じインターフェイス）。
    """

    def __init__(self, asr_model, llm_generate_fn, tts_model):
        self.asr_model = asr_model
        self.llm_generate_fn = llm_generate_fn
        self.tts_model = tts_model

    def handle(self, user_audio, **tts_kwargs):
        """ユーザー音声 → (ASR)テキスト → (LLM)応答テキスト → (TTS)応答音声。

        いずれかの段が空/失敗した場合は例外を投げず None を返す（silent failure ではなく、
        原因をログに残した上で「応答なし」を明示する）。
        """
        if self.asr_model is None:
            raise ValueError("CascadeVoiceAgent requires an asr_model (DI).")
        try:
            text = (self.asr_model.transcribe(user_audio) or "").strip()
        except Exception:
            logger.exception("CascadeVoiceAgent: ASR transcription failed.")
            return None
        if not text:
            logger.warning("CascadeVoiceAgent: ASR produced empty transcript; nothing to respond to.")
            return None

        try:
            response_text = self.llm_generate_fn(text)
        except Exception:
            logger.exception("CascadeVoiceAgent: llm_generate_fn failed.")
            return None
        if not response_text:
            logger.warning("CascadeVoiceAgent: llm_generate_fn produced empty response.")
            return None

        return self.tts_model.synthesize(response_text, **tts_kwargs)


class S2SVoiceAgent:
    """ネイティブ音声間 (S2S) 方式 [音声編10.1]。学習済み `SpeechToSpeechBackbone` を
    ラップし、ユーザー音声コードから直接エージェント音声コードを同期生成する。"""

    def __init__(self, s2s_model, codec, temperature=0.8, top_k=50):
        self.s2s_model = s2s_model
        self.codec = codec
        self.temperature = temperature
        self.top_k = top_k

    @staticmethod
    def _to_batch(user_audio: torch.Tensor) -> torch.Tensor:
        """波形を [1,1,T] へ整形する（[T] / [1,T] / [1,1,T] のいずれを渡しても対応）。"""
        if user_audio.dim() == 1:
            return user_audio.view(1, 1, -1)
        if user_audio.dim() == 2:
            return user_audio.unsqueeze(0)
        return user_audio

    @torch.no_grad()
    def handle(self, user_audio):
        """ユーザー波形 → コーデックエンコード → S2Sモデル生成 → コーデックデコード。

        V-14: ndarray/list の波形も受け付ける（`.numel()` 前提の空チェックが
        AttributeError でクラッシュしないよう、まず Tensor へ正規化する）。
        """
        user_audio = ensure_waveform(user_audio, name="S2SVoiceAgent.user_audio")
        if user_audio is None:
            logger.warning("S2SVoiceAgent: empty or non-tensor user_audio; nothing to respond to.")
            return None
        wav = self._to_batch(user_audio)
        user_codes = self.codec.encode(wav)
        if user_codes.shape[1] == 0:
            logger.warning("S2SVoiceAgent: codec produced 0 frames for user_audio.")
            return None
        agent_codes = self.s2s_model.generate(user_codes, temperature=self.temperature, top_k=self.top_k)
        agent_wav = self.codec.decode(agent_codes)
        return agent_wav.squeeze(0).squeeze(0).detach().cpu()


class AgentRouter:
    """処理経路を決定するルーター [音声編10.4]。

    cascade_agent/s2s_agent/intent_classifier を注入すると、`intent_classifier(context)` が
    返す意図（"casual" なら S2S、それ以外は cascade）に基づいて動的にルーティングする。
    いずれか未指定の場合は、実装済みの唯一の確定経路 "cascade" を返す（後方互換の最小実装）。
    """

    def __init__(self, cascade_agent=None, s2s_agent=None, intent_classifier=None):
        self.cascade_agent = cascade_agent
        self.s2s_agent = s2s_agent
        self.intent_classifier = intent_classifier
        dynamic = self.intent_classifier is not None and self.s2s_agent is not None
        logger.debug(f"Initializing AgentRouter (dynamic_routing={dynamic}).")

    def route_request(self, context) -> str:
        """コンテキストに基づき処理経路を返す。動的ルーティングの前提が揃っていなければ
        常に "cascade"（唯一保証された経路）を返す。"""
        if self.intent_classifier is None or self.s2s_agent is None:
            return ROUTE_CASCADE
        intent = self.intent_classifier(context)
        route = ROUTE_S2S if intent == "casual" else ROUTE_CASCADE
        logger.debug(f"AgentRouter: intent={intent!r} -> route={route}")
        return route

    def handle(self, context, user_audio):
        """route_request で決定した経路の実エージェントへ処理を委譲する。"""
        route = self.route_request(context)
        agent = self.s2s_agent if route == ROUTE_S2S else self.cascade_agent
        if agent is None:
            raise ValueError(f"AgentRouter: no agent configured for route '{route}'.")
        return agent.handle(user_audio)
