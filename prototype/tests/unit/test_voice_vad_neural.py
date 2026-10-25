"""
tests/unit/test_voice_vad_neural.py

fix 13 / voice roadmap step2: ニューラル VAD（Silero）統合と barge-in ヒステリシスのテスト。

- silero-vad 導入済み環境では実モデルで「RMS が誤発火する非音声（大音量ノイズ・DC・トーン）を
  ニューラル VAD は正しく棄却する」回帰を検証する（fix 13 のステップ3 が要求する騒音回帰）。
- 未導入環境では該当テストをスキップし、RMS フォールバックの契約だけを検証する（C-4）。
- BargeInDetector のヒステリシスは確率注入（probability_fn DI）で VAD 実装から独立に検証する。
"""
import os
import unittest
from unittest import mock

import torch

from swi_voice.duplex import vad
from swi_voice.duplex.barge_in import BargeInDetector


def _silero_available():
    return vad.get_silero_vad() is not None


class TestBackendResolution(unittest.TestCase):
    def test_explicit_argument_wins(self):
        self.assertEqual(vad.resolve_backend("rms"), "rms")
        self.assertEqual(vad.resolve_backend("SILERO"), "silero")

    def test_env_var_selects_backend(self):
        with mock.patch.dict(os.environ, {"SWI_VAD_BACKEND": "rms"}):
            self.assertEqual(vad.resolve_backend(None), "rms")

    def test_unknown_backend_falls_back_to_auto(self):
        self.assertEqual(vad.resolve_backend("bogus"), "auto")

    def test_default_is_auto(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWI_VAD_BACKEND", None)
            self.assertEqual(vad.resolve_backend(None), "auto")


class TestRmsFallbackContract(unittest.TestCase):
    """silero が使えないときも従来の RMS 契約が生きていること（C-4）。"""

    def test_fallback_when_silero_unavailable(self):
        # get_silero_vad が None を返す状況を強制し、auto が RMS 判定になることを確認。
        with mock.patch.object(vad, "get_silero_vad", return_value=None):
            self.assertTrue(vad.detect_voice_activity(torch.ones(2400) * 0.3))   # RMS 0.3 > 0.02
            self.assertFalse(vad.detect_voice_activity(torch.zeros(2400)))
            # 確率プロキシは {0,1} の二値
            self.assertEqual(vad.speech_probability(torch.ones(2400) * 0.3), 1.0)
            self.assertEqual(vad.speech_probability(torch.zeros(2400)), 0.0)

    def test_empty_and_none_are_safe(self):
        self.assertFalse(vad.detect_voice_activity(None))
        self.assertFalse(vad.detect_voice_activity(torch.zeros(0)))
        self.assertEqual(vad.speech_probability(None), 0.0)
        self.assertEqual(vad.speech_probability(torch.zeros(0)), 0.0)


@unittest.skipUnless(_silero_available(), "silero-vad is not installed")
class TestSileroRejectsNonSpeech(unittest.TestCase):
    """fix 13 の核心: RMS が誤発火する非音声入力を、ニューラル VAD は棄却する。

    どのケースも RMS エネルギーは閾値 0.02 を大きく超える（= RMS 版なら「発話」と誤判定）
    ことを同時に主張し、「ニューラル版が実際に判定を変えている」ことを証明する。
    """

    def test_loud_white_noise(self):
        torch.manual_seed(0)
        noise = torch.randn(24000) * 0.3            # 1 秒 @24kHz, RMS ≈ 0.3
        self.assertTrue(vad.detect_voice_activity(noise, backend="rms"))      # 既知の誤発火
        self.assertFalse(vad.detect_voice_activity(noise, backend="silero"))  # 正しく棄却

    def test_dc_signal(self):
        dc = torch.ones(24000) * 0.3
        self.assertTrue(vad.detect_voice_activity(dc, backend="rms"))
        self.assertFalse(vad.detect_voice_activity(dc, backend="silero"))

    def test_pure_tone(self):
        t = torch.arange(24000, dtype=torch.float32) / 24000.0
        tone = torch.sin(2 * torch.pi * 220.0 * t) * 0.3
        self.assertTrue(vad.detect_voice_activity(tone, backend="rms"))
        self.assertFalse(vad.detect_voice_activity(tone, backend="silero"))

    def test_probability_is_calibrated_low_for_noise(self):
        torch.manual_seed(0)
        p = vad.speech_probability(torch.randn(4800) * 0.1, backend="silero")
        self.assertGreaterEqual(p, 0.0)
        self.assertLess(p, 0.5)

    def test_short_chunk_is_padded_not_crashed(self):
        # 1 判定窓（512 @16kHz = 768 @24kHz）未満のチャンクはゼロパディングで判定される。
        out = vad.detect_voice_activity(torch.randn(256) * 0.05, backend="silero")
        self.assertIsInstance(out, bool)

    def test_forced_silero_backend_returns_bool(self):
        self.assertFalse(vad.detect_voice_activity(torch.zeros(24000), backend="silero"))


class TestBargeInHysteresis(unittest.TestCase):
    """fix 13(2): ヒステリシス + 最小継続時間の状態機械（確率を注入して決定的に検証）。"""

    @staticmethod
    def _run(probs, **kw):
        seq = iter(probs)
        det = BargeInDetector(probability_fn=lambda chunk: next(seq), **kw)
        return [det.update(None) for _ in probs]

    def test_requires_consecutive_active_chunks(self):
        # 単発の高確率（咳など）では発火しない。2 連続で初めて発火（min_active_chunks=2）。
        self.assertEqual(self._run([0.9, 0.3, 0.9, 0.9], min_active_chunks=2),
                         [False, False, False, True])

    def test_hysteresis_keeps_speaking_between_thresholds(self):
        # 発話開始後、off(0.4) < p < on(0.6) の揺らぎでは解除されない。
        self.assertEqual(self._run([0.9, 0.9, 0.5, 0.45], min_active_chunks=2),
                         [False, True, True, True])

    def test_releases_below_off_threshold_and_counter_resets(self):
        # p <= off で解除。解除後は再び「連続 min_active_chunks 回」が必要。
        self.assertEqual(self._run([0.9, 0.9, 0.3, 0.9, 0.9], min_active_chunks=2),
                         [False, True, False, False, True])

    def test_min_active_chunks_one_fires_immediately(self):
        self.assertEqual(self._run([0.7], min_active_chunks=1), [True])

    def test_reset_clears_state(self):
        seq = iter([0.9, 0.9, 0.9])
        det = BargeInDetector(min_active_chunks=2, probability_fn=lambda c: next(seq))
        det.update(None)
        self.assertTrue(det.update(None))
        det.reset()
        self.assertFalse(det.speaking)
        # reset 後は 1 回の高確率では発火しない（カウンタも消えている）
        self.assertFalse(det.update(None))

    def test_probability_error_treated_as_silence(self):
        def boom(chunk):
            raise RuntimeError("probe failed")
        det = BargeInDetector(probability_fn=boom)
        self.assertFalse(det.update(None))

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            BargeInDetector(on_prob=0.4, off_prob=0.6)
        with self.assertRaises(ValueError):
            BargeInDetector(min_active_chunks=0)

    def test_rms_binary_proxy_still_works_end_to_end(self):
        # silero 不在の縮退（確率が {0,1}）でもヒステリシス機械は成立する。
        with mock.patch.object(vad, "get_silero_vad", return_value=None):
            det = BargeInDetector(min_active_chunks=2, backend="rms")
            loud = torch.ones(2400) * 0.3
            self.assertFalse(det.update(loud))     # 1 回目: まだ
            self.assertTrue(det.update(loud))      # 2 連続: 発火
            self.assertFalse(det.update(torch.zeros(2400)))   # 無音: 解除


if __name__ == "__main__":
    unittest.main(verbosity=2)
