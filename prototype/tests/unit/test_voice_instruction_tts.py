"""
指示追従TTS (swi_voice/data/instruction_tts.py, model/style_control.py) の単体テスト
[音声編6.3, 実装編4.2 / 5.3]。

  - format_style_prompt: [STYLE]...[/STYLE] で指示を発話テキストと区別する。指示が空なら素通し。
  - build_instruction_tts_example: 空音声/0フレームは None、prompt_frames はターゲット長でクランプ。
  - StyleEncoder: 指示経路が形を保ち、指示内容が変われば条件ベクトルも変わる。
    instruction_dim 未指定なら追加パラメータを作らない（既存ckpt互換）。
"""
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_voice.data.instruction_tts import (
    build_instruction_tts_example, format_style_prompt, encode_instruction,
)
from swi_voice.model.style_control import StyleEncoder
from swi_voice.utils.constants import VOICE_STYLE_TAG_OPEN, VOICE_STYLE_TAG_CLOSE


class _CharTokenizer:
    """1 文字 = 1 トークンの決定的スタブトークナイザ。"""

    def encode(self, text):
        return [ord(c) % 64 + 1 for c in (text or "")]


class _StubCodec:
    def __init__(self, n_frames=6, n_q=2):
        self.n_frames, self.n_q = n_frames, n_q

    def encode(self, wav):
        return torch.zeros((1, self.n_frames, self.n_q), dtype=torch.long)


class TestFormatStylePrompt(unittest.TestCase):
    def test_wraps_instruction_in_tags(self):
        p = format_style_prompt("落ち着いた声で", "こんにちは")
        self.assertEqual(p, f"{VOICE_STYLE_TAG_OPEN}落ち着いた声で{VOICE_STYLE_TAG_CLOSE}こんにちは")

    def test_empty_instruction_passes_text_through(self):
        self.assertEqual(format_style_prompt("", "こんにちは"), "こんにちは")
        self.assertEqual(format_style_prompt(None, "こんにちは"), "こんにちは")

    def test_whitespace_only_instruction_is_empty(self):
        self.assertEqual(format_style_prompt("   ", "text"), "text")


class TestEncodeInstruction(unittest.TestCase):
    def test_returns_none_for_empty(self):
        self.assertIsNone(encode_instruction("", _CharTokenizer()))
        self.assertIsNone(encode_instruction(None, _CharTokenizer()))

    def test_truncates_to_max_tokens(self):
        ids = encode_instruction("あ" * 100, _CharTokenizer(), max_tokens=8)
        self.assertEqual(ids.shape[0], 8)


class TestBuildInstructionTTSExample(unittest.TestCase):
    def test_builds_example(self):
        ex = build_instruction_tts_example("ゆっくり", torch.randn(800), _StubCodec(),
                                           _CharTokenizer(), text="こんにちは")
        self.assertIsNotNone(ex)
        self.assertEqual(ex["target_codes"].shape, (6, 2))
        self.assertEqual(ex["input_ids"].dim(), 1)
        self.assertGreater(ex["input_ids"].shape[0], 0)
        self.assertIsNotNone(ex["instruction_ids"])
        self.assertEqual(ex["prompt_frames"], 0)

    def test_empty_audio_returns_none(self):
        self.assertIsNone(build_instruction_tts_example(
            "ゆっくり", torch.zeros(0), _StubCodec(), _CharTokenizer()))
        self.assertIsNone(build_instruction_tts_example(
            "ゆっくり", None, _StubCodec(), _CharTokenizer()))

    def test_zero_frame_codec_returns_none(self):
        self.assertIsNone(build_instruction_tts_example(
            "ゆっくり", torch.randn(800), _StubCodec(n_frames=0), _CharTokenizer()))

    def test_prompt_frames_clamped_below_target_length(self):
        # 全フレームが損失外になると学習信号が消えるため n_frames-1 でクランプ
        ex = build_instruction_tts_example("x", torch.randn(800), _StubCodec(n_frames=6),
                                           _CharTokenizer(), prompt_frames=99)
        self.assertEqual(ex["prompt_frames"], 5)

    def test_negative_prompt_frames_clamped_to_zero(self):
        ex = build_instruction_tts_example("x", torch.randn(800), _StubCodec(),
                                           _CharTokenizer(), prompt_frames=-3)
        self.assertEqual(ex["prompt_frames"], 0)

    def test_no_instruction_still_builds_with_none_instruction_ids(self):
        ex = build_instruction_tts_example("", torch.randn(800), _StubCodec(),
                                           _CharTokenizer(), text="こんにちは")
        self.assertIsNotNone(ex)
        self.assertIsNone(ex["instruction_ids"])


class TestStyleEncoderInstructionPath(unittest.TestCase):
    def test_instruction_path_shape(self):
        se = StyleEncoder(dim=32, n_emotions=4, instruction_dim=64)
        out = se(instruction_ids=torch.randint(1, 64, (2, 6)))
        self.assertEqual(tuple(out.shape), (2, 32))

    def test_different_instructions_give_different_conditioning(self):
        torch.manual_seed(0)
        se = StyleEncoder(dim=32, n_emotions=4, instruction_dim=64).eval()
        a = se(instruction_ids=torch.full((1, 5), 3)).detach()
        b = se(instruction_ids=torch.full((1, 5), 40)).detach()
        self.assertFalse(torch.allclose(a, b))

    def test_padding_is_masked_from_pooling(self):
        # 右パディング(=0)を足しても条件ベクトルは変わらない
        torch.manual_seed(0)
        se = StyleEncoder(dim=16, n_emotions=4, instruction_dim=64, pad_id=0).eval()
        base = se(instruction_ids=torch.tensor([[5, 6, 7]])).detach()
        padded = se(instruction_ids=torch.tensor([[5, 6, 7, 0, 0]])).detach()
        self.assertTrue(torch.allclose(base, padded, atol=1e-6))

    def test_all_padding_does_not_nan(self):
        se = StyleEncoder(dim=16, n_emotions=4, instruction_dim=64, pad_id=0).eval()
        out = se(instruction_ids=torch.zeros((1, 4), dtype=torch.long))
        self.assertTrue(torch.isfinite(out).all())

    def test_instruction_path_absent_by_default(self):
        # C-1: instruction_dim 未指定なら state_dict に指示経路のパラメータを追加しない
        se = StyleEncoder(dim=16, n_emotions=4)
        self.assertIsNone(se.instruction_encoder)
        self.assertFalse(any("instruction" in k for k in se.state_dict()))
        with self.assertRaises(ValueError):
            se(instruction_ids=torch.zeros((1, 3), dtype=torch.long))

    def test_generate_stream_applies_style_emb(self):
        """generate_stream の style_emb パススルー: 指示追従がストリーミング経路にも効くこと。"""
        from swi_voice.model import ARCodecLM
        torch.manual_seed(0)
        m = ARCodecLM(text_vocab_size=64, n_codebooks=2, codebook_size=16,
                      dim=32, n_layers=1, n_heads=2, max_seq_len=128, use_style=True).eval()
        text = torch.randint(0, 64, (1, 5))
        style = torch.randn(1, 32)

        torch.manual_seed(1)
        with_style = torch.cat(list(m.generate_stream(
            text, chunk_frames=2, max_frames=6, temperature=0.0, style_emb=style)), dim=1)
        torch.manual_seed(1)
        without = torch.cat(list(m.generate_stream(
            text, chunk_frames=2, max_frames=6, temperature=0.0,
            style_emb=torch.zeros_like(style))), dim=1)
        self.assertFalse(torch.equal(with_style, without))

        # 一括生成とストリーミング生成は同じ条件なら一致する（内部ジェネレータを共有）
        torch.manual_seed(1)
        batched = m.generate(text, max_frames=6, temperature=0.0, style_emb=style)
        self.assertTrue(torch.equal(batched, with_style))

    def test_three_paths_combine_as_mean(self):
        se = StyleEncoder(dim=8, n_emotions=4, ref_latent_dim=6, instruction_dim=32).eval()
        emo = torch.tensor([1])
        ref = torch.randn(1, 3, 6)
        ins = torch.randint(1, 32, (1, 4))
        combined = se(emotion_id=emo, ref_latent=ref, instruction_ids=ins).detach()
        expected = (se.from_emotion(emo) + se.from_reference(ref) + se.from_instruction(ins)).detach() / 3
        self.assertTrue(torch.allclose(combined, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
