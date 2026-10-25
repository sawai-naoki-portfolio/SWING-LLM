"""
tests/unit/test_voice_whisper_asr.py

fix 08 / voice roadmap step3: faster-whisper 統合（WhisperASR）のテスト。

実モデルはロードに数秒 + 初回は重みダウンロードを要するため、単体テストは
`faster_whisper` モジュールを偽物に差し替えて「契約」（波形正規化・リサンプル・
セグメント結合・縮退動作・Web ルート）を検証する。実モデルでの E2E は
`SWI_ASR_INTEGRATION_TESTS=1` のときだけ走る（重み取得済みの環境向け）。
"""
import asyncio
import base64
import io
import json
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_voice.asr import whisper_asr
from swi_voice.asr.whisper_asr import WhisperASR, get_default_asr, reset_default_asr, asr_available


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeWhisperModel:
    """faster_whisper.WhisperModel の記録付きフェイク。"""
    last = None

    def __init__(self, model_size, device=None, compute_type=None):
        _FakeWhisperModel.last = self
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.calls = []

    def transcribe(self, audio, language=None, beam_size=None):
        self.calls.append({"n_samples": int(len(audio)), "language": language,
                           "beam_size": beam_size})
        return iter([_FakeSegment(" こんにち"), _FakeSegment("は ")]), {"language": language}


def _fake_module():
    m = types.ModuleType("faster_whisper")
    m.WhisperModel = _FakeWhisperModel
    return m


class TestWhisperASRContract(unittest.TestCase):
    def setUp(self):
        _FakeWhisperModel.last = None
        self._patch = mock.patch.dict(sys.modules, {"faster_whisper": _fake_module()})
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_segments_joined_and_stripped(self):
        asr = WhisperASR(model_size="small")
        self.assertEqual(asr.transcribe(torch.zeros(24000)), "こんにちは")

    def test_resamples_24k_to_16k(self):
        asr = WhisperASR(model_size="small")
        asr.transcribe(torch.zeros(24000))                        # 1 秒 @24kHz
        self.assertEqual(_FakeWhisperModel.last.calls[0]["n_samples"], 16000)

    def test_16k_input_not_resampled(self):
        asr = WhisperASR(model_size="small")
        asr.transcribe(torch.zeros(16000), sample_rate=16000)
        self.assertEqual(_FakeWhisperModel.last.calls[0]["n_samples"], 16000)

    def test_accepts_batched_tensor_ndarray_and_list(self):
        asr = WhisperASR(model_size="small")
        self.assertEqual(asr.transcribe(torch.zeros(1, 1, 2400)), "こんにちは")   # [1,1,T]
        self.assertEqual(asr.transcribe(np.zeros(2400, dtype=np.float32)), "こんにちは")
        self.assertEqual(asr.transcribe([0.0] * 2400), "こんにちは")

    def test_multichannel_is_mono_mixed(self):
        asr = WhisperASR(model_size="small")
        stereo = torch.stack([torch.ones(24000), -torch.ones(24000)])   # [2,T]
        asr.transcribe(stereo)
        # モノラル化（平均）後にリサンプルされる（チャンネルが別サンプル扱いにならない）
        self.assertEqual(_FakeWhisperModel.last.calls[-1]["n_samples"], 16000)

    def test_empty_audio_returns_empty_without_loading_model(self):
        asr = WhisperASR(model_size="small")
        self.assertEqual(asr.transcribe(None), "")
        self.assertEqual(asr.transcribe(torch.zeros(0)), "")
        self.assertIsNone(_FakeWhisperModel.last)     # モデルは構築すらされない

    def test_language_default_ja_and_override(self):
        asr = WhisperASR(model_size="small")
        asr.transcribe(torch.zeros(2400))
        self.assertEqual(_FakeWhisperModel.last.calls[0]["language"], "ja")
        asr.transcribe(torch.zeros(2400), language="en")
        self.assertEqual(_FakeWhisperModel.last.calls[1]["language"], "en")

    def test_model_size_env_fallback(self):
        with mock.patch.dict(os.environ, {"SWI_ASR_MODEL": "tiny"}):
            self.assertEqual(WhisperASR().model_size, "tiny")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWI_ASR_MODEL", None)
            self.assertEqual(WhisperASR().model_size, "small")

    def test_model_loaded_once(self):
        asr = WhisperASR(model_size="small")
        asr.transcribe(torch.zeros(2400))
        first = _FakeWhisperModel.last
        asr.transcribe(torch.zeros(2400))
        self.assertIs(_FakeWhisperModel.last, first)


class TestDefaultASRSingleton(unittest.TestCase):
    def setUp(self):
        reset_default_asr()
        self.addCleanup(reset_default_asr)

    def test_disabled_via_env(self):
        with mock.patch.dict(os.environ, {"SWI_ASR": "0"}):
            self.assertIsNone(get_default_asr())
            self.assertFalse(asr_available())

    def test_unavailable_when_import_fails(self):
        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            self.assertFalse(asr_available())
            self.assertIsNone(get_default_asr())
        # 失敗は記憶され、以後も None（毎回 import を試みてログを吐き続けない）
        self.assertIsNone(get_default_asr())

    def test_eager_load_failure_degrades_to_none(self):
        m = types.ModuleType("faster_whisper")

        class _Boom:
            def __init__(self, *a, **kw):
                raise RuntimeError("no weights, offline")
        m.WhisperModel = _Boom
        with mock.patch.dict(sys.modules, {"faster_whisper": m}):
            self.assertIsNone(get_default_asr(eager=True))
            self.assertIsNone(get_default_asr(eager=True))   # 失敗はキャッシュされる

    def test_lazy_returns_instance_when_importable(self):
        with mock.patch.dict(sys.modules, {"faster_whisper": _fake_module()}):
            asr = get_default_asr()
            self.assertIsInstance(asr, WhisperASR)
            self.assertIs(get_default_asr(), asr)            # シングルトン


class TestASRRoute(unittest.TestCase):
    """/api/asr, /api/asr/status（web/routes/voice.py, fix 08(3)）。

    ルート関数を直接呼ぶ（test_llm_status_endpoint.py と同型）。ASR 本体は
    get_default_asr のモックで差し替え、経路の契約（400/503/200）だけを検証する。
    """

    @staticmethod
    def _wav_b64(seconds=0.1, sr=16000):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="WAV")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _call(coro):
        return asyncio.run(coro)

    def _post(self, audio_b64, lang="ja", asr=None):
        from web.routes import voice as voice_routes
        req = voice_routes.ASRRequest(audio=audio_b64, lang=lang)
        with mock.patch("swi_voice.asr.whisper_asr.get_default_asr", return_value=asr):
            return self._call(voice_routes.api_asr(req))

    def test_transcribes_valid_wav(self):
        class _ASR:
            model_size = "small"
            def transcribe(self, wav, sample_rate=24000, language=None):
                assert int(sample_rate) == 16000     # soundfile が返した実 SR が渡ること
                return "テスト転写"
        body = self._post(self._wav_b64(sr=16000), asr=_ASR())
        self.assertEqual(body, {"text": "テスト転写"})

    def test_503_when_asr_unavailable(self):
        resp = self._post(self._wav_b64(), asr=None)
        self.assertEqual(resp.status_code, 503)

    def test_400_on_missing_audio(self):
        resp = self._post("", asr=None)
        self.assertEqual(resp.status_code, 400)

    def test_400_on_invalid_base64(self):
        resp = self._post("!!!not-base64!!!", asr=object())
        self.assertEqual(resp.status_code, 400)

    def test_400_on_non_audio_bytes(self):
        garbage = base64.b64encode(b"this is not audio").decode("ascii")
        resp = self._post(garbage, asr=object())
        self.assertEqual(resp.status_code, 400)

    def test_data_url_prefix_accepted(self):
        class _ASR:
            model_size = "small"
            def transcribe(self, wav, sample_rate=24000, language=None):
                return "ok"
        body = self._post("data:audio/wav;base64," + self._wav_b64(), asr=_ASR())
        self.assertEqual(body["text"], "ok")

    def test_status_reports_unavailable_when_disabled(self):
        from web.routes import voice as voice_routes
        with mock.patch("swi_voice.asr.whisper_asr.asr_available", return_value=False):
            body = self._call(voice_routes.api_asr_status())
        self.assertEqual(body, {"available": False, "model": None})

    def test_status_reports_model_when_available(self):
        from web.routes import voice as voice_routes

        class _ASR:
            model_size = "small"
        with mock.patch("swi_voice.asr.whisper_asr.asr_available", return_value=True), \
             mock.patch("swi_voice.asr.whisper_asr.get_default_asr", return_value=_ASR()):
            body = self._call(voice_routes.api_asr_status())
        self.assertEqual(body, {"available": True, "model": "small"})


@unittest.skipUnless(os.environ.get("SWI_ASR_INTEGRATION_TESTS") == "1",
                     "set SWI_ASR_INTEGRATION_TESTS=1 to run the real faster-whisper E2E")
class TestRealWhisperIntegration(unittest.TestCase):
    """実モデル E2E（オプトイン）: 実クリップを転写し、参照との WER が閾値未満であること。"""

    def test_real_clip_transcription(self):
        from swi_voice.data.audio_dataset import load_voice_clips
        from swi_voice.data.asr_pseudo_label import word_error_rate
        reset_default_asr()
        asr = get_default_asr(eager=True)
        self.assertIsNotNone(asr, "faster-whisper が利用できません")
        clip = load_voice_clips(max_clips=1, sample_rate=24000)[0]
        hyp = asr.transcribe(clip["audio"])
        self.assertTrue(hyp.strip(), "実音声の転写が空です")
        wer = word_error_rate(hyp, clip.get("text") or "")
        self.assertLess(wer, 0.9, f"WER が異常に高い: {wer} (hyp={hyp!r})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
