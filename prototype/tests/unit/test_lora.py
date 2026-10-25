"""
LoRA と継続学習支援 (swi_llm/training/lora.py, continual.py) の単体テスト（§11）。
"""
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_llm.core.models import MiniLLM
from swi_llm.training.lora import (
    apply_lora, lora_state_dict, count_trainable_parameters, merge_all_lora, LoRALinear,
)
from swi_llm.training.continual import ReplayBuffer, FeedbackStore

_KW = dict(dim=64, n_layers=2, n_heads=4, hidden_dim=128, num_experts=4, top_k=2, max_seq_len=64)


class TestLoRA(unittest.TestCase):
    def _model(self):
        return MiniLLM(vocab_size=256, **_KW)

    def test_apply_lora_replaces_layers(self):
        m = self._model()
        n = apply_lora(m, rank=4, alpha=8)
        self.assertGreater(n, 0)
        self.assertTrue(any(isinstance(mod, LoRALinear) for mod in m.modules()))

    def test_only_lora_trainable(self):
        m = self._model()
        apply_lora(m, rank=4)
        trainable_names = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertTrue(trainable_names)
        self.assertTrue(all("lora_" in n for n in trainable_names))

    def test_trainable_fraction_small(self):
        m = self._model()
        apply_lora(m, rank=4)
        tr, tot = count_trainable_parameters(m)
        self.assertLess(tr / tot, 0.2)  # LoRA は総パラメータのごく一部

    def test_forward_unchanged_at_init(self):
        # B=0 初期化のため LoRA 適用直後は forward がベースと一致する
        m = self._model()
        m.eval()
        x = torch.randint(0, 256, (1, 8))
        before = m(x)[0].clone()
        apply_lora(m, rank=4)
        m.eval()
        after = m(x)[0]
        self.assertTrue(torch.allclose(before, after, atol=1e-5))

    def test_lora_state_dict_only_lora(self):
        m = self._model()
        apply_lora(m, rank=4)
        self.assertTrue(all("lora_" in k for k in lora_state_dict(m)))

    def test_merge_removes_lora(self):
        m = self._model()
        apply_lora(m, rank=4)
        merge_all_lora(m)
        self.assertFalse(any(isinstance(mod, LoRALinear) for mod in m.modules()))


class TestReplayBuffer(unittest.TestCase):
    def test_add_and_sample(self):
        rb = ReplayBuffer(capacity=3)
        rb.extend(["a", "b", "c", "d"])  # 容量超過
        self.assertEqual(len(rb), 3)
        self.assertLessEqual(len(rb.sample(2)), 2)

    def test_mix_ratio(self):
        rb = ReplayBuffer()
        rb.extend(["p", "q", "r", "s"])
        mixed = rb.mix(["x", "y", "z", "w"], replay_ratio=0.5)
        self.assertEqual(len(mixed), 4 + 2)  # 新規4 + リプレイ2

    def test_empty_sample(self):
        self.assertEqual(ReplayBuffer().sample(3), [])


class TestFeedbackStore(unittest.TestCase):
    def test_add_and_load(self):
        path = os.path.join(tempfile.mkdtemp(), "fb.jsonl")
        fs = FeedbackStore(path)
        fs.add("q1", "r1", 1)
        fs.add("q2", "r2", -1)
        self.assertEqual(len(fs.load()), 2)

    def test_positive_filter(self):
        path = os.path.join(tempfile.mkdtemp(), "fb.jsonl")
        fs = FeedbackStore(path)
        fs.add("q1", "r1", 1)
        fs.add("q2", "r2", -1)
        fs.add("q3", "r3", 1)
        self.assertEqual(len(fs.positive_examples()), 2)


if __name__ == "__main__":
    unittest.main()
