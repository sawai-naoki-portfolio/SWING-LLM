"""
swi_voice/agents/cascade_router.py の単体テスト
（未実装解消: CascadeVoiceAgent / S2SVoiceAgent / 動的ルーティング §10.1, 10.4）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_voice.agents.cascade_router import (
    AgentRouter, CascadeVoiceAgent, S2SVoiceAgent, ROUTE_CASCADE, ROUTE_S2S,
)


class _FakeASR:
    def __init__(self, text="こんにちは"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.text


class _FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, **kwargs):
        self.calls.append(text)
        return f"WAV({text})"


class TestCascadeVoiceAgent(unittest.TestCase):
    def test_happy_path(self):
        asr = _FakeASR("今日の天気は？")
        tts = _FakeTTS()
        agent = CascadeVoiceAgent(asr, lambda text: f"応答: {text}", tts)
        out = agent.handle(torch.zeros(10))
        self.assertEqual(out, "WAV(応答: 今日の天気は？)")
        self.assertEqual(asr.calls, 1)
        self.assertEqual(tts.calls, ["応答: 今日の天気は？"])

    def test_none_asr_model_raises(self):
        agent = CascadeVoiceAgent(None, lambda t: t, _FakeTTS())
        with self.assertRaises(ValueError):
            agent.handle(torch.zeros(10))

    def test_empty_transcript_returns_none(self):
        agent = CascadeVoiceAgent(_FakeASR(""), lambda t: t, _FakeTTS())
        self.assertIsNone(agent.handle(torch.zeros(10)))

    def test_empty_llm_response_returns_none(self):
        agent = CascadeVoiceAgent(_FakeASR("hi"), lambda t: "", _FakeTTS())
        self.assertIsNone(agent.handle(torch.zeros(10)))

    def test_asr_exception_returns_none_without_raising(self):
        class _RaisingASR:
            def transcribe(self, audio):
                raise RuntimeError("asr backend down")
        agent = CascadeVoiceAgent(_RaisingASR(), lambda t: t, _FakeTTS())
        self.assertIsNone(agent.handle(torch.zeros(10)))

    def test_llm_exception_returns_none_without_raising(self):
        def broken_llm(text):
            raise RuntimeError("llm backend down")
        agent = CascadeVoiceAgent(_FakeASR("hi"), broken_llm, _FakeTTS())
        self.assertIsNone(agent.handle(torch.zeros(10)))


class TestS2SVoiceAgent(unittest.TestCase):
    def _make(self):
        from swi_voice.codec import NeuralAudioCodec
        from swi_voice.duplex.s2s_backbone import SpeechToSpeechBackbone
        torch.manual_seed(0)
        codec = NeuralAudioCodec(dim=16, hidden=8, n_quantizers=2, codebook_size=32)
        codec.eval()
        s2s = SpeechToSpeechBackbone(n_codebooks=2, codebook_size=32, dim=32, n_layers=1, n_heads=2)
        s2s.eval()
        return S2SVoiceAgent(s2s, codec)

    def test_handle_1d_audio_returns_finite_waveform(self):
        agent = self._make()
        user_audio = torch.randn(480 * 6) * 0.1
        out = agent.handle(user_audio)
        self.assertIsNotNone(out)
        self.assertGreater(out.numel(), 0)
        self.assertTrue(torch.isfinite(out).all())

    def test_handle_batched_audio_shape(self):
        agent = self._make()
        user_audio = torch.randn(1, 1, 480 * 4) * 0.1
        out = agent.handle(user_audio)
        self.assertEqual(out.dim(), 1)

    def test_empty_audio_returns_none(self):
        agent = self._make()
        self.assertIsNone(agent.handle(torch.zeros(0)))
        self.assertIsNone(agent.handle(None))

    def test_numpy_audio_is_accepted_not_crash(self):
        """V-14: soundfile 由来の ndarray を渡しても `.numel()` の AttributeError で
        クラッシュせず、Tensor へ正規化して処理される。"""
        import numpy as np
        agent = self._make()
        out = agent.handle((np.random.RandomState(0).randn(480 * 4) * 0.1).astype(np.float32))
        self.assertIsNotNone(out)
        self.assertTrue(torch.isfinite(out).all())

    def test_unconvertible_audio_returns_none_without_raising(self):
        """V-14: 波形として解釈できない入力は WARNING の上 None（安全フォールバック）。"""
        agent = self._make()
        self.assertIsNone(agent.handle("not audio"))


class TestDynamicAgentRouter(unittest.TestCase):
    def test_default_router_is_static_cascade(self):
        router = AgentRouter()
        self.assertEqual(router.route_request({"x": 1}), ROUTE_CASCADE)

    def test_dynamic_routing_by_intent(self):
        cascade = CascadeVoiceAgent(_FakeASR("procedure request"), lambda t: t, _FakeTTS())
        s2s_calls = []

        class _FakeS2S:
            def handle(self, audio):
                s2s_calls.append(audio)
                return "s2s-response"

        router = AgentRouter(cascade_agent=cascade, s2s_agent=_FakeS2S(),
                              intent_classifier=lambda ctx: ctx.get("intent"))
        self.assertEqual(router.route_request({"intent": "casual"}), ROUTE_S2S)
        self.assertEqual(router.route_request({"intent": "procedure"}), ROUTE_CASCADE)

        out = router.handle({"intent": "casual"}, torch.zeros(10))
        self.assertEqual(out, "s2s-response")
        self.assertEqual(len(s2s_calls), 1)

    def test_missing_agent_for_chosen_route_raises(self):
        router = AgentRouter(cascade_agent=None, s2s_agent=object(),
                              intent_classifier=lambda ctx: "procedure")
        with self.assertRaises(ValueError):
            router.handle({}, torch.zeros(10))

    def test_partial_configuration_falls_back_to_static(self):
        # s2s_agent はあるが intent_classifier が無い -> 動的ルーティングの前提を満たさない
        router = AgentRouter(s2s_agent=object())
        self.assertEqual(router.route_request({}), ROUTE_CASCADE)


if __name__ == "__main__":
    unittest.main()
