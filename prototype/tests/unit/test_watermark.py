"""
電子透かし (swi_voice/safety/watermark.py) の単体テスト。

拡散スペクトラム方式:
  - apply → detect でペイロードが往復検出できる。
  - 透かし未挿入の波形は検出されない（誤検出しない）。
  - 形状・有限性が保たれ、振幅変化は微小（不可聴レベル）。
  - 異なるペイロードでは検出されない。
"""
import os
import sys
import math
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from swi_voice.safety.watermark import apply_watermark, detect_watermark
from swi_voice.utils.constants import VOICE_WATERMARK_PAYLOAD, VOICE_WATERMARK_STRENGTH


def _host_signal(T=48000):
    """決定的な音声風ホスト信号（正弦波の重ね合わせ）。"""
    t = torch.arange(T, dtype=torch.float32) / 24000.0
    return 0.3 * torch.sin(2 * math.pi * 220 * t) + 0.15 * torch.sin(2 * math.pi * 440 * t)


class TestWatermark(unittest.TestCase):
    def test_roundtrip_detects_payload(self):
        wav = _host_signal()
        wm = apply_watermark(wav, payload=VOICE_WATERMARK_PAYLOAD)
        self.assertEqual(detect_watermark(wm, payload=VOICE_WATERMARK_PAYLOAD), VOICE_WATERMARK_PAYLOAD)
        # payload 未指定（既知集合の走査）でも検出できる
        self.assertEqual(detect_watermark(wm), VOICE_WATERMARK_PAYLOAD)

    def test_clean_signal_not_detected(self):
        wav = _host_signal()
        self.assertEqual(detect_watermark(wav), "")
        self.assertEqual(detect_watermark(wav, payload=VOICE_WATERMARK_PAYLOAD), "")

    def test_wrong_payload_not_detected(self):
        wav = _host_signal()
        wm = apply_watermark(wav, payload=VOICE_WATERMARK_PAYLOAD)
        self.assertEqual(detect_watermark(wm, payload="SOME-OTHER-PAYLOAD"), "")

    def test_shape_and_finite_preserved(self):
        wav = _host_signal(T=1920).view(1, 1, -1)   # pipeline_check 相当の短い多次元入力
        wm = apply_watermark(wav)
        self.assertEqual(wm.shape, wav.shape)
        self.assertTrue(torch.isfinite(wm).all())

    def test_watermark_amplitude_is_small(self):
        wav = _host_signal()
        wm = apply_watermark(wav, payload=VOICE_WATERMARK_PAYLOAD)
        max_diff = (wm - wav).abs().max().item()
        # 加算振幅は strength 程度（クランプ分だけ小さくなることはあっても超えない）
        self.assertLessEqual(max_diff, VOICE_WATERMARK_STRENGTH + 1e-6)

    def test_empty_returns_as_is(self):
        empty = torch.zeros(0)
        self.assertEqual(apply_watermark(empty).numel(), 0)
        self.assertEqual(detect_watermark(empty), "")


if __name__ == "__main__":
    unittest.main()
