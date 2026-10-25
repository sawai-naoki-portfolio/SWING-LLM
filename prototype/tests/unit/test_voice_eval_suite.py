"""
評価スイート (swi_voice/eval/evaluate.py) と WER ドリフト監視 (eval/drift_monitor.py) の
単体テスト [音声編12.1 / 12.3]。

  - transcription_wer: 完全一致で 0.0、不一致で >0。ASR 失敗/未注入は None。
  - speaker_similarity: 同一音声で ~1.0。空入力は None。
  - VoiceEvalSuite: 注入された軸だけを測り、平均は測れた値のみで取る。
  - DriftMonitor: MOS の低下と WER の上昇をそれぞれ検知する。
"""
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_voice.eval.evaluate import (
    VoiceEvalSuite, speaker_similarity, transcription_wer,
)
from swi_voice.eval.drift_monitor import DriftMonitor
from swi_voice.codec import NeuralAudioCodec


class _EchoASR:
    """与えた書き起こしをそのまま返すスタブ ASR。"""

    def __init__(self, transcript):
        self.transcript = transcript

    def transcribe(self, audio):
        return self.transcript


class _FailingASR:
    def transcribe(self, audio):
        raise RuntimeError("asr exploded")


class _ConstMOS:
    """常に一定の MOS を返すスタブ予測器。"""

    def __init__(self, value):
        self.value = value

    def __call__(self, waveform):
        return torch.full((waveform.shape[0],), float(self.value))


def _tiny_codec():
    # strides の積 = 8。極小構成で高速に回す。
    return NeuralAudioCodec(dim=8, hidden=4, n_quantizers=2, codebook_size=16, strides=(2, 4))


class TestTranscriptionWER(unittest.TestCase):
    def test_exact_match_is_zero(self):
        wav = torch.randn(800)
        self.assertEqual(transcription_wer(wav, "こんにちは", _EchoASR("こんにちは")), 0.0)

    def test_mismatch_is_positive(self):
        wav = torch.randn(800)
        self.assertGreater(transcription_wer(wav, "こんにちは", _EchoASR("さようなら")), 0.0)

    def test_no_asr_returns_none(self):
        self.assertIsNone(transcription_wer(torch.randn(800), "text", None))

    def test_asr_failure_returns_none(self):
        # 1 件の ASR 失敗で評価全体を落とさない
        self.assertIsNone(transcription_wer(torch.randn(800), "text", _FailingASR()))

    def test_empty_waveform_returns_none(self):
        self.assertIsNone(transcription_wer(torch.zeros(0), "text", _EchoASR("text")))


def _sine(freq, n=800, sr=800.0):
    t = torch.linspace(0, n / sr, n)
    return (0.4 * torch.sin(2 * torch.pi * freq * t)).view(1, 1, -1)


class TestSpeakerSimilarity(unittest.TestCase):
    def test_identical_audio_is_one(self):
        codec = _tiny_codec().eval()
        wav = torch.randn(1, 1, 800)
        sim = speaker_similarity(wav, wav, codec)
        self.assertIsNotNone(sim)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_discriminates_different_timbres(self):
        """回帰防止: 量子化 latent / DC 未除去だと、どの 2 音声でも 1.0 に飽和して弁別できない。"""
        codec = _tiny_codec().eval()
        a, b = _sine(20.0), _sine(90.0)
        same = speaker_similarity(a, a, codec)
        diff = speaker_similarity(a, b, codec)
        self.assertAlmostEqual(same, 1.0, places=4)
        self.assertLess(diff, 0.99)          # 飽和していないこと
        self.assertLess(diff, same)

    def test_empty_input_returns_none(self):
        codec = _tiny_codec().eval()
        self.assertIsNone(speaker_similarity(torch.zeros(0), torch.randn(1, 1, 800), codec))
        self.assertIsNone(speaker_similarity(torch.randn(1, 1, 800), None, codec))

    def test_silence_similarity_is_undefined(self):
        # 無音は話者情報を持たない -> 0.0（「似ていない」）ではなく None（未定義）
        codec = _tiny_codec().eval()
        self.assertIsNone(speaker_similarity(torch.zeros(1, 1, 800), _sine(20.0), codec))

    def test_too_short_waveform_returns_none(self):
        # エンコーダの受容野に満たない波形は RuntimeError になる（実コーデックは 5 段ダウン
        # サンプルのため 480 サンプル程度でも落ちる）。例外にせず None を返すこと。
        class _ShortRejectingCodec:
            def encode_latent(self, wav):
                raise RuntimeError("Kernel size can't be greater than actual input size")

        self.assertIsNone(speaker_similarity(torch.randn(1, 1, 2), _sine(20.0), _ShortRejectingCodec()))

    def test_similarity_in_valid_range(self):
        codec = _tiny_codec().eval()
        sim = speaker_similarity(torch.randn(1, 1, 800), torch.randn(1, 1, 800), codec)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)


class TestVoiceEvalSuite(unittest.TestCase):
    def test_measures_only_injected_axes(self):
        suite = VoiceEvalSuite(codec=None, asr_model=None, mos_predictor=_ConstMOS(4.0))
        r = suite.evaluate_one(torch.randn(1, 1, 800), expected_text="text")
        self.assertIsNone(r["wer"])
        self.assertIsNone(r["sim"])
        self.assertAlmostEqual(r["mos"], 4.0, places=5)

    def test_all_axes(self):
        codec = _tiny_codec().eval()
        suite = VoiceEvalSuite(codec=codec, asr_model=_EchoASR("あ"), mos_predictor=_ConstMOS(3.5))
        wav = torch.randn(1, 1, 800)
        r = suite.evaluate_one(wav, expected_text="あ", ref_waveform=wav)
        self.assertEqual(r["wer"], 0.0)
        self.assertGreater(r["sim"], 0.99)
        self.assertAlmostEqual(r["mos"], 3.5, places=5)

    def test_evaluate_aggregates_and_skips_unmeasurable(self):
        suite = VoiceEvalSuite(asr_model=_EchoASR("あ"), mos_predictor=_ConstMOS(3.0))
        samples = [
            {"audio": torch.randn(1, 1, 800), "text": "あ"},      # wer = 0.0
            {"audio": torch.randn(1, 1, 800)},                     # text 無し -> wer なし
        ]
        r = suite.evaluate(samples)
        self.assertEqual(r["n"], 2)
        self.assertEqual(r["wer"], 0.0)          # 測れた 1 件のみで平均
        self.assertIsNone(r["sim"])              # codec 未注入
        self.assertAlmostEqual(r["mos"], 3.0, places=5)

    def test_empty_samples(self):
        r = VoiceEvalSuite().evaluate([])
        self.assertEqual(r["n"], 0)
        self.assertIsNone(r["mos"])


class TestDriftMonitor(unittest.TestCase):
    def test_mos_drop_detected(self):
        dm = DriftMonitor(window=5, drop_threshold=0.3)
        dm.set_baseline([4.0, 4.0, 4.0])
        for _ in range(3):
            dm.observe(3.0)
        drifted, delta = dm.check_drift()
        self.assertTrue(drifted)
        self.assertAlmostEqual(delta, 1.0, places=5)

    def test_mos_stable_not_flagged(self):
        dm = DriftMonitor(window=5, drop_threshold=0.3)
        dm.set_baseline([4.0, 4.0])
        dm.observe(3.95)
        drifted, _ = dm.check_drift()
        self.assertFalse(drifted)

    def test_wer_rise_detected(self):
        dm = DriftMonitor(window=5, wer_rise_threshold=0.2)
        dm.set_wer_baseline([0.10, 0.10])
        dm.observe_wer(0.5)
        drifted, delta = dm.check_wer_drift()
        self.assertTrue(drifted)
        self.assertAlmostEqual(delta, 0.4, places=5)

    def test_wer_stable_not_flagged(self):
        dm = DriftMonitor(window=5, wer_rise_threshold=0.2)
        dm.set_wer_baseline([0.10, 0.10])
        dm.observe_wer(0.15)
        drifted, _ = dm.check_wer_drift()
        self.assertFalse(drifted)

    def test_no_baseline_never_drifts(self):
        dm = DriftMonitor()
        dm.observe(1.0)
        dm.observe_wer(0.9)
        self.assertEqual(dm.check_drift(), (False, 0.0))
        self.assertEqual(dm.check_wer_drift(), (False, 0.0))

    def test_score_wer_records_history(self):
        dm = DriftMonitor(asr_model=_EchoASR("さようなら"), window=5)
        wer = dm.score_wer(torch.randn(1, 1, 800), "こんにちは")
        self.assertGreater(wer, 0.0)
        self.assertEqual(dm.recent_wer_mean(), wer)

    def test_score_wer_ignores_asr_failure(self):
        dm = DriftMonitor(asr_model=_FailingASR(), window=5)
        self.assertIsNone(dm.score_wer(torch.randn(1, 1, 800), "text"))
        self.assertIsNone(dm.recent_wer_mean())    # 失敗は履歴に残さない

    def test_check_all_reports_both_axes(self):
        dm = DriftMonitor(window=5, drop_threshold=0.3, wer_rise_threshold=0.2)
        dm.set_baseline([4.0])
        dm.set_wer_baseline([0.1])
        dm.observe(3.0)
        dm.observe_wer(0.5)
        result = dm.check_all()
        self.assertTrue(result["mos"][0])
        self.assertTrue(result["wer"][0])

    def test_window_evicts_old_scores(self):
        dm = DriftMonitor(window=2)
        dm.observe(1.0)
        dm.observe(5.0)
        dm.observe(5.0)
        self.assertAlmostEqual(dm.recent_mean(), 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
