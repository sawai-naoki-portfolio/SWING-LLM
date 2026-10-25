"""
tests/unit/test_voice_quality_filter.py

NEW-V1（音声編1.1）: SNR による低品質音源フィルタ。
実測済みの分離特性（実クリップ 16〜59dB / ノイズのみ ~0.6dB）を合成信号で固定する。
"""
import os
import sys
import unittest
from unittest import mock

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_voice.data.quality_filter import estimate_snr_db, min_snr_threshold, passes_snr

SR = 24000


def _speech_like(seconds=2.0, sr=SR, noise_amp=0.001):
    """発話風の合成信号: 語間の無音を挟んだトーンバースト + 微小ノイズフロア。"""
    torch.manual_seed(0)
    t = torch.arange(int(seconds * sr), dtype=torch.float32) / sr
    tone = torch.sin(2 * torch.pi * 220.0 * t) * 0.3
    gate = (torch.sin(2 * torch.pi * 2.0 * t) > 0).float()     # 250ms オン/オフの発話ゲート
    return tone * gate + torch.randn_like(t) * noise_amp


class TestEstimateSnr(unittest.TestCase):
    def test_clean_speech_like_is_high(self):
        snr = estimate_snr_db(_speech_like(noise_amp=0.001))
        self.assertGreater(snr, 20.0)

    def test_noise_only_is_near_zero(self):
        torch.manual_seed(0)
        snr = estimate_snr_db(torch.randn(SR * 2) * 0.1)
        self.assertLess(snr, 3.0)

    def test_noisy_speech_lower_than_clean(self):
        clean = estimate_snr_db(_speech_like(noise_amp=0.001))
        noisy = estimate_snr_db(_speech_like(noise_amp=0.1))
        self.assertLess(noisy, clean - 10.0)

    def test_silence_is_capped_not_inf(self):
        snr = estimate_snr_db(torch.zeros(SR * 2))
        self.assertIsNotNone(snr)
        self.assertLessEqual(snr, 60.0)

    def test_too_short_returns_none(self):
        self.assertIsNone(estimate_snr_db(torch.randn(100)))
        self.assertIsNone(estimate_snr_db(None))

    def test_accepts_channel_dim(self):
        snr = estimate_snr_db(_speech_like().unsqueeze(0))     # [1, T]
        self.assertGreater(snr, 20.0)


class TestThresholdAndGate(unittest.TestCase):
    def test_default_threshold(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWI_VOICE_MIN_SNR", None)
            self.assertEqual(min_snr_threshold(), 10.0)

    def test_env_override_and_off(self):
        with mock.patch.dict(os.environ, {"SWI_VOICE_MIN_SNR": "25"}):
            self.assertEqual(min_snr_threshold(), 25.0)
        with mock.patch.dict(os.environ, {"SWI_VOICE_MIN_SNR": "off"}):
            self.assertIsNone(min_snr_threshold())

    def test_invalid_env_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"SWI_VOICE_MIN_SNR": "loud"}):
            self.assertEqual(min_snr_threshold(), 10.0)

    def test_passes_snr_separates_speech_from_noise(self):
        torch.manual_seed(0)
        ok, snr = passes_snr(_speech_like(), threshold=10.0)
        self.assertTrue(ok)
        self.assertGreater(snr, 10.0)
        bad, snr_bad = passes_snr(torch.randn(SR * 2) * 0.1, threshold=10.0)
        self.assertFalse(bad)
        self.assertLess(snr_bad, 10.0)

    def test_disabled_via_env_passes_everything(self):
        # threshold=None は「引数で未指定 → env/既定を引く」なので、無効化は env の off で行う
        torch.manual_seed(0)
        with mock.patch.dict(os.environ, {"SWI_VOICE_MIN_SNR": "off"}):
            ok, snr = passes_snr(torch.randn(SR * 2) * 0.1)
        self.assertTrue(ok)
        self.assertIsNone(snr)

    def test_unestimable_clip_passes(self):
        ok, snr = passes_snr(torch.randn(64), threshold=10.0)  # 短すぎて推定不能
        self.assertTrue(ok)
        self.assertIsNone(snr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
