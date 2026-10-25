"""
tests/unit/test_voice_tts_engines.py

NEW-V2/V3（音声編3.2/3.3）: TTS serving のエンジン切替（ar | flow | hybrid）。

flow / hybrid は「実装済みだが製品経路に乗っていない」状態だったものを serve/tts.py へ
接続した。極小モデルを直接組み立てて（ckpt を介さず）合成経路そのものを検証する。
"""
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data.tokenizer import get_tokenizer
from swi_voice.codec import NeuralAudioCodec
from swi_voice.model import FlowTTS
from swi_voice.model.hybrid_ar_fm import HybridARFM
from swi_voice.serve import tts as serve_tts
from swi_voice.serve.tts import SWIVoiceTTS, resolve_tts_engine
from swi_voice.utils.constants import VOICE_SAMPLE_RATE


def _tiny_instance(engine):
    """ckpt を介さず極小モデルで SWIVoiceTTS を組み立てる（合成経路のユニット検証用）。"""
    tok = get_tokenizer()
    inst = SWIVoiceTTS.__new__(SWIVoiceTTS)
    inst.device = "cpu"
    inst.tokenizer = tok
    inst.codec = NeuralAudioCodec(dim=16, hidden=8, n_quantizers=2, codebook_size=32).eval()
    inst.model = None                     # ar 経路は使わない
    inst.engine = engine
    inst.flow_model = None
    inst.hybrid_model = None
    inst.flow_steps = 2
    inst.flow_frames = 8
    if engine == "flow":
        inst.flow_model = FlowTTS(text_vocab_size=tok.vocab_size, latent_dim=16,
                                  dim=32, n_layers=1, n_heads=2, max_frames=32).eval()
    elif engine == "hybrid":
        inst.hybrid_model = HybridARFM(
            text_vocab_size=tok.vocab_size, semantic_codebook_size=16,
            acoustic_latent_dim=16, ar_dim=32, ar_layers=1, ar_heads=2,
            fm_dim=32, fm_layers=1, fm_heads=2, fm_max_frames=32,
            max_sem_len=16).eval()
    return inst


def _parse_wav(wav_bytes):
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return data, sr


class TestEngineResolution(unittest.TestCase):
    def test_default_is_ar(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWI_TTS_ENGINE", None)
            self.assertEqual(resolve_tts_engine(), "ar")

    def test_env_selects_engine(self):
        for e in ("flow", "hybrid", "AR"):
            with mock.patch.dict(os.environ, {"SWI_TTS_ENGINE": e}):
                self.assertEqual(resolve_tts_engine(), e.lower())

    def test_unknown_engine_falls_back_to_ar(self):
        with mock.patch.dict(os.environ, {"SWI_TTS_ENGINE": "vocoder9000"}):
            self.assertEqual(resolve_tts_engine(), "ar")


class TestMissingCheckpointFallback(unittest.TestCase):
    """指定エンジンの ckpt が無ければ WARNING を出して ar へ縮退する（C-4）。"""

    def _bare_instance(self):
        inst = SWIVoiceTTS.__new__(SWIVoiceTTS)
        inst.device = "cpu"
        inst.tokenizer = get_tokenizer()
        inst.flow_model = None
        inst.hybrid_model = None
        return inst

    def test_flow_missing_falls_back(self):
        inst = self._bare_instance()
        inst.engine = "flow"
        with mock.patch.object(serve_tts, "_find_ckpt", return_value=None):
            with self.assertLogs("voice.tts", level="WARNING"):
                inst._load_flow_engine("cpu")
        self.assertEqual(inst.engine, "ar")
        self.assertIsNone(inst.flow_model)

    def test_hybrid_missing_falls_back(self):
        inst = self._bare_instance()
        inst.engine = "hybrid"
        with mock.patch.object(serve_tts, "_find_ckpt", return_value=None):
            with self.assertLogs("voice.tts", level="WARNING"):
                inst._load_hybrid_engine("cpu")
        self.assertEqual(inst.engine, "ar")
        self.assertIsNone(inst.hybrid_model)


class TestFlowEngineSynthesis(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tts = _tiny_instance("flow")

    def test_synthesize_returns_valid_wav(self):
        wav_bytes = self.tts.synthesize("こんにちは")
        data, sr = _parse_wav(wav_bytes)
        self.assertEqual(sr, VOICE_SAMPLE_RATE)
        self.assertGreater(data.shape[0], 0)
        self.assertTrue(np.isfinite(data).all())

    def test_stream_yields_valid_chunks(self):
        chunks = list(self.tts.synthesize_stream("こんにちは"))
        self.assertGreaterEqual(len(chunks), 1)
        for ch in chunks:
            data, sr = _parse_wav(ch)
            self.assertEqual(sr, VOICE_SAMPLE_RATE)
            self.assertGreater(data.shape[0], 0)

    def test_generation_length_respects_flow_frames(self):
        # n_frames = min(max_frames, flow_frames, model.max_frames) = 8 → 8 * hop サンプル
        wav_bytes = self.tts.synthesize("テスト", max_frames=200)
        data, _ = _parse_wav(wav_bytes)
        self.assertLessEqual(data.shape[0], 8 * self.tts.codec.hop + self.tts.codec.hop)


class TestHybridEngineSynthesis(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tts = _tiny_instance("hybrid")

    def test_synthesize_returns_valid_wav(self):
        wav_bytes = self.tts.synthesize("こんにちは", max_frames=8)
        data, sr = _parse_wav(wav_bytes)
        self.assertEqual(sr, VOICE_SAMPLE_RATE)
        self.assertGreater(data.shape[0], 0)
        self.assertTrue(np.isfinite(data).all())

    def test_stream_yields_valid_chunks(self):
        chunks = list(self.tts.synthesize_stream("テスト", max_frames=8))
        self.assertGreaterEqual(len(chunks), 1)
        data, sr = _parse_wav(chunks[0])
        self.assertEqual(sr, VOICE_SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
