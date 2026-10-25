"""
swi_voice/train/continual.py の単体テスト（未実装解消: LoRA話者追加/フィードバック収集 §9.1）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_voice.model import ARCodecLM
from swi_voice.train.continual import (
    add_speaker_lora, save_speaker_adapter, load_speaker_adapter,
    merge_speaker_lora, collect_voice_feedback,
)


def _make_model():
    torch.manual_seed(0)
    return ARCodecLM(text_vocab_size=64, n_codebooks=2, codebook_size=16,
                      dim=32, n_layers=2, n_heads=2)


class TestSpeakerLoRA(unittest.TestCase):
    def test_add_speaker_lora_freezes_base_and_attaches_adapters(self):
        m = _make_model()
        n_layers = add_speaker_lora(m, rank=4, alpha=8)
        self.assertGreater(n_layers, 0)
        trainable = [name for name, p in m.named_parameters() if p.requires_grad]
        self.assertTrue(all("lora_" in name for name in trainable))   # LoRA以外は凍結
        self.assertGreater(len(trainable), 0)

    def test_lora_adapter_changes_output_when_trained(self):
        m = _make_model()
        text = torch.randint(0, 64, (1, 5))
        codes = torch.randint(0, 16, (1, 6, 2))
        with torch.no_grad():
            _, loss_before = m(text, codes)

        add_speaker_lora(m, rank=4, alpha=8)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(50):
            opt.zero_grad()
            _, loss = m(text, codes)
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, loss_after = m(text, codes)
        self.assertLess(loss_after.item(), loss_before.item())        # 話者アダプタが学習される

    def test_save_and_load_adapter_roundtrip(self):
        m1 = _make_model()
        add_speaker_lora(m1, rank=4, alpha=8)
        text = torch.randint(0, 64, (1, 5))
        codes = torch.randint(0, 16, (1, 6, 2))
        opt = torch.optim.Adam([p for p in m1.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(20):
            opt.zero_grad()
            _, loss = m1(text, codes)
            loss.backward()
            opt.step()

        path = os.path.join(os.path.dirname(__file__), "_tmp_speaker_adapter.pt")
        try:
            save_speaker_adapter(m1, path)

            m2 = _make_model()
            add_speaker_lora(m2, rank=4, alpha=8)
            load_speaker_adapter(m2, path)

            m1.eval(); m2.eval()
            with torch.no_grad():
                logits1, _ = m1(text, codes)
                logits2, _ = m2(text, codes)
            for a, b in zip(logits1, logits2):
                self.assertTrue(torch.allclose(a, b, atol=1e-5))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_merge_speaker_lora_preserves_output(self):
        m = _make_model()
        add_speaker_lora(m, rank=4, alpha=8)
        text = torch.randint(0, 64, (1, 5))
        codes = torch.randint(0, 16, (1, 6, 2))
        m.eval()
        with torch.no_grad():
            logits_before, _ = m(text, codes)
        merged = merge_speaker_lora(m)
        self.assertGreater(merged, 0)
        with torch.no_grad():
            logits_after, _ = m(text, codes)
        for a, b in zip(logits_before, logits_after):
            self.assertTrue(torch.allclose(a, b, atol=1e-4))          # マージ前後で出力不変

    def test_zero_matching_layers_warns_and_returns_zero(self):
        m = _make_model()
        n = add_speaker_lora(m, targets=("no_such_layer_name",))
        self.assertEqual(n, 0)


class TestCollectVoiceFeedback(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(collect_voice_feedback([]), [])
        self.assertEqual(collect_voice_feedback(None), [])

    def test_only_regenerated_interactions_are_collected(self):
        interactions = [
            {"regenerated": True, "text_ids": [1, 2], "final_codes": "F1", "first_codes": "R1"},
            {"regenerated": False, "text_ids": [3, 4], "final_codes": "F2", "first_codes": "R2"},
        ]
        pairs = collect_voice_feedback(interactions)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen_codes"], "F1")
        self.assertEqual(pairs[0]["rejected_codes"], "R1")

    def test_incomplete_records_are_skipped_without_raising(self):
        interactions = [
            {"regenerated": True, "text_ids": [1, 2], "final_codes": "F1"},   # rejected 欠損
            None,                                                              # 不正レコード
            {"regenerated": True, "text_ids": [1], "final_codes": "F3", "first_codes": "R3"},
        ]
        pairs = collect_voice_feedback(interactions)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen_codes"], "F3")


if __name__ == "__main__":
    unittest.main()
