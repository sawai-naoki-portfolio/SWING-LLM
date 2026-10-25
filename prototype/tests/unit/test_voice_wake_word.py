"""
swi_voice/duplex/wake_word.py の単体テスト（未実装解消: ウェイクワード検出 §13.2）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_voice.duplex.wake_word import WakeWordDetector


def _template(n=480):
    # 単調ではない、はっきり区別できるパターン（矩形波っぽい形）
    t = torch.arange(n, dtype=torch.float32)
    return torch.sin(t * 0.3) + 0.5 * torch.sin(t * 0.07)


class TestWakeWordDetector(unittest.TestCase):
    def test_rejects_empty_template(self):
        with self.assertRaises(ValueError):
            WakeWordDetector(torch.zeros(0))
        with self.assertRaises(ValueError):
            WakeWordDetector(None)

    def test_detects_exact_template_embedded_in_silence(self):
        template = _template(480)
        audio = torch.cat([torch.zeros(960), template, torch.zeros(960)])
        det = WakeWordDetector(template, threshold=0.6, hop=80)
        self.assertTrue(det.detect(audio))

    def test_rejects_unrelated_audio(self):
        torch.manual_seed(0)
        template = _template(480)
        unrelated = torch.randn(2000) * 0.2
        det = WakeWordDetector(template, threshold=0.6, hop=80)
        self.assertFalse(det.detect(unrelated))

    def test_shorter_than_template_is_false_not_crash(self):
        template = _template(480)
        det = WakeWordDetector(template)
        self.assertFalse(det.detect(torch.randn(100)))

    def test_none_or_empty_audio_frame_is_false(self):
        det = WakeWordDetector(_template(480))
        self.assertFalse(det.detect(None))
        self.assertFalse(det.detect(torch.zeros(0)))

    def test_amplitude_invariant_detection(self):
        # 振幅（音量）が違っても正規化相関なら検出できる
        template = _template(480)
        quiet_copy = template * 0.1
        audio = torch.cat([torch.zeros(240), quiet_copy, torch.zeros(240)])
        det = WakeWordDetector(template, threshold=0.6, hop=80)
        self.assertTrue(det.detect(audio))


class TestWakeWordRobustInputs(unittest.TestCase):
    """V-14: 非 Tensor 入力（ndarray/list 等）でも `.numel()` の AttributeError で
    落ちず、Tensor へ正規化して動作する（変換不能は False）。"""

    def test_numpy_audio_frame_is_accepted(self):
        import numpy as np
        template = _template(480)
        det = WakeWordDetector(template, threshold=0.6, hop=80)
        audio = torch.cat([torch.zeros(240), template, torch.zeros(240)]).numpy()
        self.assertTrue(det.detect(audio))
        self.assertFalse(det.detect(np.zeros(2000, dtype=np.float32)))

    def test_list_and_unconvertible_inputs_return_false(self):
        det = WakeWordDetector(_template(480))
        self.assertFalse(det.detect([0.0] * 100))    # list（テンプレート長未満）→ False
        self.assertFalse(det.detect("not audio"))    # 変換不能 → False（例外を投げない）

    def test_numpy_template_is_accepted(self):
        template = _template(480)
        det = WakeWordDetector(template.numpy(), threshold=0.6, hop=80)
        audio = torch.cat([torch.zeros(240), template, torch.zeros(240)])
        self.assertTrue(det.detect(audio))


class TestBestScoreVectorization(unittest.TestCase):
    """V-15: unfold 一括計算が旧ループ実装と同じスコアを返す（等価性の回帰）。"""

    def test_matches_loop_reference(self):
        torch.manual_seed(1)
        template = _template(480)
        det = WakeWordDetector(template, threshold=0.6, hop=64)
        audio = torch.randn(3000)
        audio[777:777 + 480] += template * 2.0   # 非 hop 整列の位置に強い相関を仕込む

        # 旧実装（ウィンドウごとに正規化して内積）を参照実装として再現
        def loop_reference():
            a = audio.detach().float().reshape(-1)
            best = 0.0
            for start in range(0, a.numel() - det.template_len + 1, det.hop):
                w = a[start:start + det.template_len]
                norm = torch.linalg.norm(w)
                w = w if norm < 1e-8 else w / norm
                best = max(best, float(torch.dot(w, det.template)))
            return best

        self.assertAlmostEqual(det._best_score(audio), loop_reference(), places=4)


if __name__ == "__main__":
    unittest.main()
