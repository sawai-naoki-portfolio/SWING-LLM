"""
swi_voice/data/asr_pseudo_label.py の単体テスト（未実装解消: データ品質パイプライン §1.2）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_voice.data.asr_pseudo_label import word_error_rate, pseudo_label_and_filter


class _FakeASR:
    def __init__(self, transcripts, raise_on=()):
        """transcripts: list（呼び出し順）または dict（audio値→text）。"""
        self.transcripts = transcripts
        self.raise_on = set(raise_on)
        self.n_calls = 0

    def transcribe(self, audio):
        idx = self.n_calls
        self.n_calls += 1
        if idx in self.raise_on:
            raise RuntimeError("ASR backend failure")
        return self.transcripts[idx]


class TestWordErrorRate(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(word_error_rate("hello world", "hello world"), 0.0)

    def test_one_word_substitution(self):
        self.assertAlmostEqual(word_error_rate("hello there", "hello world"), 0.5)

    def test_empty_reference_nonempty_hyp_is_one(self):
        self.assertEqual(word_error_rate("hello", ""), 1.0)

    def test_both_empty_is_zero(self):
        self.assertEqual(word_error_rate("", ""), 0.0)

    def test_japanese_falls_back_to_char_level(self):
        # 空白の無い日本語は文字単位（実質CER）で計算される
        wer = word_error_rate("こんばんは", "こんにちは")
        self.assertGreater(wer, 0.0)
        self.assertLess(wer, 1.0)


class TestPseudoLabelAndFilter(unittest.TestCase):
    def test_empty_batch_returns_empty(self):
        self.assertEqual(pseudo_label_and_filter([], _FakeASR([])), [])

    def test_none_model_returns_empty(self):
        batch = [{"audio": torch.zeros(10), "text": "hi"}]
        self.assertEqual(pseudo_label_and_filter(batch, None), [])

    def test_consistent_pair_is_kept_with_reference_text(self):
        batch = [{"audio": torch.zeros(10), "text": "hello world", "speaker_id": "s1"}]
        asr = _FakeASR(["hello world"])
        out = pseudo_label_and_filter(batch, asr, wer_threshold=0.15)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "hello world")
        self.assertEqual(out[0]["speaker_id"], "s1")   # 他フィールドは保持される

    def test_inconsistent_pair_is_dropped(self):
        batch = [{"audio": torch.zeros(10), "text": "completely different sentence"}]
        asr = _FakeASR(["hello world"])
        out = pseudo_label_and_filter(batch, asr, wer_threshold=0.15)
        self.assertEqual(out, [])

    def test_missing_transcript_adopts_asr_output_as_pseudo_label(self):
        batch = [{"audio": torch.zeros(10)}]     # text フィールド無し
        asr = _FakeASR(["ASR generated text"])
        out = pseudo_label_and_filter(batch, asr)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "ASR generated text")

    def test_existing_transcripts_dict_used_over_item_text(self):
        batch = [{"audio": torch.zeros(10), "text": "stale text"}]
        asr = _FakeASR(["fresh reference"])
        out = pseudo_label_and_filter(batch, asr, existing_transcripts={0: "fresh reference"})
        self.assertEqual(out[0]["text"], "fresh reference")

    def test_empty_or_missing_audio_is_skipped(self):
        batch = [{"audio": None, "text": "hi"}, {"audio": torch.zeros(0), "text": "hi"}]
        asr = _FakeASR(["hi", "hi"])
        out = pseudo_label_and_filter(batch, asr)
        self.assertEqual(out, [])

    def test_asr_exception_skips_item_without_raising(self):
        batch = [{"audio": torch.zeros(10), "text": "hi"},
                 {"audio": torch.zeros(10), "text": "there"}]
        asr = _FakeASR(["hi", "there"], raise_on={0})
        out = pseudo_label_and_filter(batch, asr, wer_threshold=0.5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "there")

    def test_original_batch_items_not_mutated(self):
        item = {"audio": torch.zeros(10), "text": "hello"}
        batch = [item]
        asr = _FakeASR(["hello"])
        out = pseudo_label_and_filter(batch, asr)
        self.assertEqual(item["text"], "hello")
        self.assertIsNot(out[0], item)


if __name__ == "__main__":
    unittest.main()
