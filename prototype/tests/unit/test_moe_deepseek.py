"""
DeepSeekMoE 拡張と aux_loss の頑健化 (core/models/transformer_blocks.py) の単体テスト。

検証:
  - 問題1: 共有エキスパート（use_shared_experts）— パラメータ追加・出力への寄与。
  - 問題2: aux-loss-free バランシング（use_aux_free_balancing）— routing_bias の
    学習時更新 / 推論時不変。
  - L-4: aux_loss が NaN/Inf のとき、グラフ接続を保ったまま 0 化し発生回数を集計する。
  - C-1: 既定（両フラグ OFF）では state_dict 構造が従来と同一（新規キーが増えない）。
tokenizer 非依存で小さいモデル/モジュールを構築して検証する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_llm.core.models import MiniLLM
from swi_llm.core.models.transformer_blocks import SparseMoE

_KW = dict(dim=64, n_layers=2, n_heads=4, hidden_dim=128, num_experts=4,
           top_k=2, max_seq_len=32)


class TestDefaultCompatibility(unittest.TestCase):
    def test_default_off_has_no_deepseek_params(self):
        # C-1: 既定は共有エキスパート/ルーティングバイアスを持たず、旧 ckpt と同一構造。
        m = MiniLLM(vocab_size=256, **_KW)
        extra = [k for k in m.state_dict() if "shared_experts" in k or "routing_bias" in k]
        self.assertEqual(extra, [])

    def test_default_forward_finite(self):
        m = MiniLLM(vocab_size=256, **_KW)
        m.eval()
        logits, aux = m(torch.randint(0, 256, (2, 8)))
        self.assertEqual(tuple(logits.shape), (2, 8, 256))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(aux).all())


class TestSharedExperts(unittest.TestCase):
    def test_shared_experts_adds_params(self):
        m = MiniLLM(vocab_size=256, use_shared_experts=True, **_KW)
        keys = [k for k in m.state_dict() if "shared_experts" in k]
        self.assertTrue(len(keys) > 0)
        m.eval()
        logits, aux = m(torch.randint(0, 256, (2, 8)))
        self.assertTrue(torch.isfinite(logits).all())

    def test_shared_experts_contributes_to_output(self):
        # gate/experts を同一重みに揃え、共有エキスパートの有無で出力が変わることを確認。
        torch.manual_seed(0)
        off = SparseMoE(16, 32, 4, 2)
        torch.manual_seed(0)
        on = SparseMoE(16, 32, 4, 2, use_shared_experts=True)
        on.gate.load_state_dict(off.gate.state_dict())
        on.experts.load_state_dict(off.experts.state_dict())
        off.eval(); on.eval()
        x = torch.randn(2, 5, 16)
        o_off, _ = off(x)
        o_on, _ = on(x)
        self.assertFalse(torch.allclose(o_off, o_on))


class TestAuxFreeBalancing(unittest.TestCase):
    def test_bias_buffer_exists_and_starts_zero(self):
        moe = SparseMoE(16, 32, 4, 2, use_aux_free_balancing=True)
        self.assertTrue(hasattr(moe, "routing_bias"))
        self.assertTrue(torch.all(moe.routing_bias == 0))

    def test_bias_updates_in_train(self):
        torch.manual_seed(0)
        moe = SparseMoE(16, 32, 4, 2, use_aux_free_balancing=True)
        moe.train()
        out, aux = moe(torch.randn(4, 8, 16))
        # 学習時は負荷分散のためバイアスが 0 から動く
        self.assertTrue(torch.any(moe.routing_bias != 0))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(aux).all())

    def test_bias_frozen_in_eval(self):
        moe = SparseMoE(16, 32, 4, 2, use_aux_free_balancing=True)
        moe.eval()
        moe(torch.randn(4, 8, 16))
        self.assertTrue(torch.all(moe.routing_bias == 0))


class TestAuxLossNaNRobustness(unittest.TestCase):
    def test_nan_aux_loss_sanitized_connected_and_counted(self):
        # L-4: gate 重みを NaN にして aux_loss を NaN 化させ、フォールバック挙動を検証。
        moe = SparseMoE(16, 32, 4, 2)
        with torch.no_grad():
            moe.gate.weight.fill_(float("nan"))
        out, aux = moe(torch.randn(2, 3, 16))
        self.assertTrue(torch.isfinite(aux).all())       # 0 に無害化
        self.assertEqual(moe._nan_aux_count, 1)          # 発生を集計
        # 新規リーフではなくグラフに接続されたまま（旧実装は grad_fn=None のリーフを返していた）
        self.assertIsNotNone(aux.grad_fn)


if __name__ == "__main__":
    unittest.main()
