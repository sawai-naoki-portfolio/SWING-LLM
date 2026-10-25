"""
音声合成 TTS 推論 (swi_voice/serve/tts.py) と WebUI 統合の単体テスト。
チェックポイント/トークナイザ未整備の環境でも壊れないよう skip ガードする。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.constants import TOKENIZER_MODEL_PATH

_PROTO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TOK = os.path.join(_PROTO, TOKENIZER_MODEL_PATH)


class TestVoiceTTSHelpers(unittest.TestCase):
    """純粋ロジック（チェックポイント探索順・可用性判定）。"""

    def test_suffix_priority_includes_fallbacks(self):
        from swi_voice.serve.tts import _suffix_priority
        order = _suffix_priority()
        self.assertIn("", order)               # PROD（無印）フォールバックを含む
        self.assertIn("_TEST", order)
        self.assertGreaterEqual(len(order), 4)

    def test_voice_available_returns_bool(self):
        from swi_voice.serve.tts import voice_available
        self.assertIsInstance(voice_available(), bool)


@unittest.skipUnless(os.path.exists(_TOK), "tokenizer.model not trained yet")
class TestVoiceTTSSynthesis(unittest.TestCase):
    """学習済みチェックポイントがある場合のみ、text→WAV の end-to-end を検証。"""

    def test_synthesize_produces_valid_wav(self):
        from swi_voice.serve.tts import voice_available, get_tts
        if not voice_available():
            self.skipTest("voice checkpoints not present (run swi_voice training first)")
        import io
        import soundfile as sf
        wav = get_tts().synthesize("テスト音声です。", lang="ja", max_frames=16)
        self.assertEqual(wav[:4], b"RIFF")     # 有効な WAV
        data, sr = sf.read(io.BytesIO(wav))
        self.assertEqual(sr, 24000)
        self.assertGreater(len(data), 0)       # 空音声にならない


if __name__ == "__main__":
    unittest.main()
