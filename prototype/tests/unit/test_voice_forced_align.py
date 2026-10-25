"""
swi_voice/data/forced_align.py の単体テスト（未実装解消: データ品質パイプライン §1.2）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_voice.data.forced_align import align_text_and_audio
from swi_voice.utils.constants import VOICE_CODEC_HOP


def _tone(n_frames, hop=VOICE_CODEC_HOP, amp=0.3):
    """発話ありに判定される一定振幅の波形 [n_frames*hop] を作る。"""
    return torch.ones(n_frames * hop) * amp


def _silence(n_frames, hop=VOICE_CODEC_HOP):
    return torch.zeros(n_frames * hop)


class TestForcedAlignFallback(unittest.TestCase):
    """aligner_model 未指定時のエネルギーベース・フォールバック経路。"""

    def test_rejects_empty_text(self):
        self.assertIsNone(align_text_and_audio("", _tone(6)))

    def test_rejects_missing_audio(self):
        self.assertIsNone(align_text_and_audio("hello", None))
        self.assertIsNone(align_text_and_audio("hello", torch.zeros(0)))

    def test_rejects_all_silence(self):
        self.assertIsNone(align_text_and_audio("hello", _silence(10)))

    def test_word_level_split_and_monotonic_frames(self):
        audio = _tone(12)
        alignment = align_text_and_audio("hello world", audio, max_gap_seconds=100.0)
        self.assertIsNotNone(alignment)
        self.assertEqual([u.text for u in alignment.units], ["hello", "world"])
        # 単調増加かつフレーム範囲が音声全体をちょうどカバーする
        prev_end = 0
        for u in alignment.units:
            self.assertGreaterEqual(u.start_frame, prev_end)
            self.assertGreater(u.end_frame, u.start_frame)
            prev_end = u.end_frame
        self.assertEqual(alignment.units[0].start_frame, 0)
        self.assertEqual(alignment.units[-1].end_frame, 12)

    def test_char_level_split_for_whitespace_free_text(self):
        # 日本語等、空白の無いテキストは文字単位にフォールバックする
        audio = _tone(9)
        alignment = align_text_and_audio("こんにちは", audio, max_gap_seconds=100.0)
        self.assertEqual(len(alignment.units), 5)
        self.assertEqual("".join(u.text for u in alignment.units), "こんにちは")

    def test_durations_proportional_to_unit_length(self):
        # "a" (1文字) と "bbbb" (4文字) => 概ね 1:4 の比でフレームが配分される
        audio = _tone(20)
        alignment = align_text_and_audio("a bbbb", audio, max_gap_seconds=100.0)
        dur_a = alignment.units[0].end_frame - alignment.units[0].start_frame
        dur_b = alignment.units[1].end_frame - alignment.units[1].start_frame
        self.assertGreater(dur_b, dur_a)

    def test_leading_trailing_silence_trimmed_and_reflected_in_gap(self):
        audio = torch.cat([_silence(5), _tone(6), _silence(5)])
        alignment = align_text_and_audio("hi", audio, max_gap_seconds=100.0)
        self.assertIsNotNone(alignment)
        # 無音区間はトリムされ、発話区間の先頭付近から割り当てが始まる
        self.assertGreaterEqual(alignment.units[0].start_frame, 4)
        self.assertGreater(alignment.max_gap_seconds, 0.0)

    def test_large_gap_is_rejected(self):
        # 大部分が無音 => leading/trailing gap が閾値を超えて None になる
        audio = torch.cat([_silence(50), _tone(2), _silence(50)])
        self.assertIsNone(align_text_and_audio("hi", audio, max_gap_seconds=0.1))


class _FakeAligner:
    def __init__(self, gap_seconds=0.0, raise_on_align=False):
        self.gap_seconds = gap_seconds
        self.raise_on_align = raise_on_align
        self.calls = []

    def align(self, audio, text):
        self.calls.append((audio, text))
        if self.raise_on_align:
            raise RuntimeError("boom")

        class _Result:
            max_gap_seconds = self.gap_seconds

        return _Result()


class TestForcedAlignWithInjectedModel(unittest.TestCase):
    """aligner_model を注入した場合はそれに委譲する（asr_pseudo_label と同じ DI 方式）。"""

    def test_delegates_to_aligner_model(self):
        aligner = _FakeAligner(gap_seconds=0.05)
        audio = _tone(4)
        result = align_text_and_audio("hi", audio, aligner_model=aligner, max_gap_seconds=2.0)
        self.assertEqual(result.max_gap_seconds, aligner.gap_seconds)
        self.assertEqual(len(aligner.calls), 1)

    def test_rejects_large_gap_from_aligner_model(self):
        aligner = _FakeAligner(gap_seconds=5.0)
        result = align_text_and_audio("hi", _tone(4), aligner_model=aligner, max_gap_seconds=2.0)
        self.assertIsNone(result)

    def test_aligner_model_exception_yields_none(self):
        aligner = _FakeAligner(raise_on_align=True)
        result = align_text_and_audio("hi", _tone(4), aligner_model=aligner)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
