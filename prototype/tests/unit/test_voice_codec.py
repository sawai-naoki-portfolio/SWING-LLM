"""
音声コーデック (swi_voice/codec/rvq.py) の単体テスト [音声編2.2]。
torch が必要なため未導入環境ではスキップ。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestVectorQuantizer(unittest.TestCase):
    def test_forward_shapes_and_valid_indices(self):
        from swi_voice.codec import VectorQuantizer
        vq = VectorQuantizer(codebook_size=32, dim=8)
        x = torch.randn(2, 5, 8)
        q, idx, loss = vq(x)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(idx.shape, (2, 5))
        self.assertTrue((idx >= 0).all() and (idx < 32).all())
        self.assertTrue(torch.isfinite(loss))

    def test_straight_through_gradient(self):
        from swi_voice.codec import VectorQuantizer
        vq = VectorQuantizer(codebook_size=16, dim=4)
        x = torch.randn(1, 3, 4, requires_grad=True)
        q, _, _ = vq(x)
        q.sum().backward()
        # STE により入力へ勾配が流れる
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestResidualVQ(unittest.TestCase):
    def test_codes_shape_and_decode_roundtrip(self):
        from swi_voice.codec import ResidualVQ
        rvq = ResidualVQ(n_quantizers=3, codebook_size=32, dim=8)
        x = torch.randn(2, 6, 8)
        q, codes, loss = rvq(x)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(codes.shape, (2, 6, 3))
        self.assertTrue(torch.isfinite(loss))
        # decode_codes は同じコードから潜在を復元する
        z = rvq.decode_codes(codes)
        self.assertEqual(z.shape, x.shape)


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestNeuralAudioCodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swi_voice.codec import NeuralAudioCodec
        cls.codec = NeuralAudioCodec(dim=16, hidden=8, n_quantizers=2, codebook_size=32)
        cls.hop = cls.codec.hop  # 480
        cls.wave = torch.randn(2, 1, cls.hop * 6)
        cls.n_frames = cls.codec.encode(cls.wave).shape[1]  # 実測フレーム数（>0）

    def test_frames_are_produced(self):
        self.assertGreater(self.n_frames, 0)

    def test_forward_recon_and_loss(self):
        recon, codes, loss = self.codec(self.wave)
        self.assertEqual(recon.shape[0], 2)
        self.assertEqual(recon.shape[1], 1)
        # 再構成長は入力長と概ね一致（数フレーム分の誤差以内）
        self.assertLessEqual(abs(recon.shape[-1] - self.wave.shape[-1]), 3 * self.hop)
        # codes: [B, T', n_quantizers]
        self.assertEqual(codes.shape[0], 2)
        self.assertEqual(codes.shape[2], 2)
        self.assertGreater(codes.shape[1], 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # 学習可能

    def test_encode_decode_shapes(self):
        codes = self.codec.encode(self.wave)
        self.assertEqual(codes.shape[0], 2)
        self.assertEqual(codes.shape[2], 2)
        self.assertTrue((codes >= 0).all() and (codes < 32).all())
        wav = self.codec.decode(codes)
        self.assertEqual(wav.shape[1], 1)
        self.assertGreater(wav.shape[-1], 0)

    def test_pad_to_hop_handles_arbitrary_length(self):
        # hop の倍数でない長さも扱える
        wave = torch.randn(1, 1, self.hop * 3 + 17)
        recon, codes, loss = self.codec(wave)
        self.assertEqual(codes.shape[-1], 2)
        self.assertGreater(codes.shape[1], 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
