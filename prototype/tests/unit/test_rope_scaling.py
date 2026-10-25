"""
長文外挿 (§9.1) の単体テスト: precompute_freqs_cis の YaRN / Dynamic NTK スケーリング。

要点:
  - スケーリング無指定、または end <= original_max_seq_len では従来と完全一致
    （＝既存チェックポイントの位置エンコーディング互換を壊さない）。
  - end が学習長を超えるときのみ YaRN / NTK が発動する。
  - 出力形状・値域（cos/sin ∈ [-1, 1]、有限）が健全。
"""
import os
import sys
import math
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_llm.core.models.transformer_blocks import precompute_freqs_cis


class TestRopeScaling(unittest.TestCase):
    DIM = 64
    ORIG = 512

    def test_no_scaling_shape_and_range(self):
        cos, sin = precompute_freqs_cis(self.DIM, self.ORIG)
        self.assertEqual(cos.shape, (self.ORIG, self.DIM // 2))
        self.assertEqual(sin.shape, (self.ORIG, self.DIM // 2))
        self.assertTrue(torch.isfinite(cos).all() and torch.isfinite(sin).all())
        self.assertLessEqual(cos.abs().max().item(), 1.0 + 1e-6)
        self.assertLessEqual(sin.abs().max().item(), 1.0 + 1e-6)

    def test_within_original_len_is_unchanged(self):
        # end <= original_max_seq_len では scaling 指定があっても従来と一致
        base_cos, base_sin = precompute_freqs_cis(self.DIM, self.ORIG)
        for st in ("yarn", "ntk"):
            cos, sin = precompute_freqs_cis(
                self.DIM, self.ORIG, original_max_seq_len=self.ORIG, scaling_type=st)
            self.assertTrue(torch.allclose(cos, base_cos), f"{st} altered within-range cos")
            self.assertTrue(torch.allclose(sin, base_sin), f"{st} altered within-range sin")

    def test_no_scaling_type_ignores_original_len(self):
        # scaling_type 無指定なら original_max_seq_len を渡しても従来計算のまま
        long = self.ORIG * 4
        plain_cos, plain_sin = precompute_freqs_cis(self.DIM, long)
        cos, sin = precompute_freqs_cis(self.DIM, long, original_max_seq_len=self.ORIG)
        self.assertTrue(torch.allclose(cos, plain_cos))
        self.assertTrue(torch.allclose(sin, plain_sin))

    def test_yarn_extrapolation_differs_and_finite(self):
        long = self.ORIG * 4
        plain_cos, plain_sin = precompute_freqs_cis(self.DIM, long)
        cos, sin = precompute_freqs_cis(
            self.DIM, long, original_max_seq_len=self.ORIG, scaling_type="yarn")
        self.assertEqual(cos.shape, (long, self.DIM // 2))
        self.assertTrue(torch.isfinite(cos).all() and torch.isfinite(sin).all())
        # スケーリングにより外挿域では従来値から変化しているはず
        self.assertFalse(torch.allclose(cos, plain_cos))
        # 先頭位置 (t=0) は角度 0 なので cos=1, sin=0 が保たれる
        self.assertTrue(torch.allclose(cos[0], torch.ones_like(cos[0])))
        self.assertTrue(torch.allclose(sin[0], torch.zeros_like(sin[0]), atol=1e-6))

    def test_ntk_extrapolation_differs_and_finite(self):
        long = self.ORIG * 4
        plain_cos, plain_sin = precompute_freqs_cis(self.DIM, long)
        cos, sin = precompute_freqs_cis(
            self.DIM, long, original_max_seq_len=self.ORIG, scaling_type="ntk")
        self.assertEqual(cos.shape, (long, self.DIM // 2))
        self.assertTrue(torch.isfinite(cos).all() and torch.isfinite(sin).all())
        self.assertFalse(torch.allclose(cos, plain_cos))

    def test_unknown_scaling_type_falls_back(self):
        # 未知の scaling_type は従来計算にフォールバック（例外を投げない）
        long = self.ORIG * 4
        plain_cos, plain_sin = precompute_freqs_cis(self.DIM, long)
        cos, sin = precompute_freqs_cis(
            self.DIM, long, original_max_seq_len=self.ORIG, scaling_type="bogus")
        self.assertTrue(torch.allclose(cos, plain_cos))
        self.assertTrue(torch.allclose(sin, plain_sin))


if __name__ == "__main__":
    unittest.main()
