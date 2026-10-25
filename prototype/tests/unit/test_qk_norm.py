"""
QK-Norm と SwiGLU Clamping (core/models/transformer_blocks.py) の単体テスト
（ブループリント §2.3 / §2.7-C）。tokenizer 非依存で小さいモデルを構築して検証する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_llm.core.models import MiniLLM
from swi_llm.core.models.transformer_blocks import SwiGLU, SWIGLU_CLAMP_VALUE

_KW = dict(dim=64, n_layers=2, n_heads=4, hidden_dim=128, num_experts=4,
           top_k=2, max_seq_len=32)


class TestQKNorm(unittest.TestCase):
    def test_qk_norm_adds_params(self):
        m = MiniLLM(vocab_size=256, use_qk_norm=True, **_KW)
        qk = [k for k in m.state_dict() if "q_norm" in k or "k_norm" in k]
        # 2 層 × (q_norm + k_norm) = 4
        self.assertEqual(len(qk), 4)

    def test_qk_norm_ablation_removes_params(self):
        m = MiniLLM(vocab_size=256, use_qk_norm=False, **_KW)
        qk = [k for k in m.state_dict() if "q_norm" in k or "k_norm" in k]
        self.assertEqual(len(qk), 0)

    def test_forward_finite_with_qk_norm(self):
        m = MiniLLM(vocab_size=256, use_qk_norm=True, **_KW)
        m.eval()
        x = torch.randint(0, 256, (2, 16))
        logits, aux = m(x)
        self.assertEqual(tuple(logits.shape), (2, 16, 256))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(aux).all())

    def test_cache_forward_with_qk_norm(self):
        from swi_llm.core.models.transformer_blocks import StaticMLACache
        m = MiniLLM(vocab_size=256, use_qk_norm=True, **_KW)
        m.eval()
        x = torch.randint(0, 256, (1, 8))
        cache = [StaticMLACache(1, 32, 512, torch.device("cpu"), torch.float32)
                 for _ in range(len(m.layers))]
        out = m(x, start_pos=0, use_cache=True, past_key_values=cache)
        self.assertTrue(torch.isfinite(out[0]).all())


class TestSwiGLUClamping(unittest.TestCase):
    def test_clamp_keeps_output_finite_on_extreme_input(self):
        torch.manual_seed(0)
        ffn = SwiGLU(dim=32, hidden_dim=64)
        # 極端に大きい入力でも clamp により出力が有限に保たれる
        x = torch.full((4, 32), 1e4)
        out = ffn(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_clamp_value_is_ten(self):
        self.assertEqual(SWIGLU_CLAMP_VALUE, 10.0)


if __name__ == "__main__":
    unittest.main()
