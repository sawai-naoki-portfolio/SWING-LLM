"""
swi_voice/serve/swarm.py の単体テスト（未実装解消: デバイス×クラウド階層ルーティング §13.2）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_voice.serve.swarm import VoiceSwarmRouter, DEFAULT_WAKE_RESPONSE, DEFAULT_CONFIRMATION_RESPONSE


class _FakeWakeWord:
    def __init__(self, active):
        self.active = active
        self.calls = 0

    def detect(self, audio_frame):
        self.calls += 1
        return self.active


class _FakeTTS:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, **kwargs):
        self.calls.append(text)
        return f"WAV({text})"


class _FakeCloudAgent:
    def __init__(self):
        self.calls = []

    def handle(self, audio_frame):
        self.calls.append(audio_frame)
        return "cloud-response"


class TestVoiceSwarmRouter(unittest.TestCase):
    def test_wake_word_handled_on_device(self):
        wake = _FakeWakeWord(active=True)
        tts = _FakeTTS()
        cloud = _FakeCloudAgent()
        router = VoiceSwarmRouter(wake, tts, cloud_agent=cloud)
        out = router.handle_frame("frame")
        self.assertEqual(out, f"WAV({DEFAULT_WAKE_RESPONSE})")
        self.assertEqual(cloud.calls, [])          # クラウドへは行かない

    def test_simple_confirmation_handled_on_device(self):
        wake = _FakeWakeWord(active=False)
        tts = _FakeTTS()
        cloud = _FakeCloudAgent()
        router = VoiceSwarmRouter(wake, tts, cloud_agent=cloud)
        out = router.handle_frame("frame", conversation_state={"is_simple_confirmation": True})
        self.assertEqual(out, f"WAV({DEFAULT_CONFIRMATION_RESPONSE})")
        self.assertEqual(cloud.calls, [])

    def test_complex_request_escalates_to_cloud(self):
        wake = _FakeWakeWord(active=False)
        tts = _FakeTTS()
        cloud = _FakeCloudAgent()
        router = VoiceSwarmRouter(wake, tts, cloud_agent=cloud)
        out = router.handle_frame("frame")
        self.assertEqual(out, "cloud-response")
        self.assertEqual(cloud.calls, ["frame"])
        self.assertEqual(tts.calls, [])

    def test_missing_cloud_agent_raises_for_complex_request(self):
        router = VoiceSwarmRouter(_FakeWakeWord(active=False), _FakeTTS(), cloud_agent=None)
        with self.assertRaises(ValueError):
            router.handle_frame("frame")

    def test_wake_word_checked_before_cloud_escalation(self):
        wake = _FakeWakeWord(active=True)
        tts = _FakeTTS()
        router = VoiceSwarmRouter(wake, tts, cloud_agent=None)   # cloud_agent 無しでも wake word は動く
        out = router.handle_frame("frame")
        self.assertEqual(out, f"WAV({DEFAULT_WAKE_RESPONSE})")

    def test_custom_responses(self):
        wake = _FakeWakeWord(active=True)
        tts = _FakeTTS()
        router = VoiceSwarmRouter(wake, tts, wake_response="やあ！")
        self.assertEqual(router.handle_frame("frame"), "WAV(やあ！)")


if __name__ == "__main__":
    unittest.main()
