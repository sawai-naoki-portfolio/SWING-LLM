"""
画像 VAE (swi_image/model/vae.py) の単体テスト [画像編2]。torch 必須のため未導入時 skip。
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
class TestConvVAE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swi_image.model import ConvVAE
        cls.size = 32
        cls.down = 4
        cls.vae = ConvVAE(in_ch=3, latent_ch=4, base_ch=8, down_factor=cls.down)
        cls.x = torch.rand(2, 3, cls.size, cls.size) * 2 - 1     # [-1, 1]

    def test_encode_shapes(self):
        mean, logvar = self.vae.encode(self.x)
        lat = self.size // self.down
        self.assertEqual(mean.shape, (2, 4, lat, lat))
        self.assertEqual(logvar.shape, (2, 4, lat, lat))
        self.assertTrue(torch.isfinite(mean).all() and torch.isfinite(logvar).all())

    def test_forward_recon_and_loss(self):
        recon, loss, parts = self.vae(self.x)
        self.assertEqual(recon.shape, self.x.shape)          # 入力と同形へ復元
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("recon", parts)
        self.assertIn("kl", parts)
        loss.backward()                                       # 学習可能

    def test_encode_decode_roundtrip_shapes(self):
        z = self.vae.encode_latent(self.x)                    # スケール済み潜在
        lat = self.size // self.down
        self.assertEqual(z.shape, (2, 4, lat, lat))
        img = self.vae.decode(z)                               # scaled=True でスケール戻し
        self.assertEqual(img.shape, self.x.shape)
        self.assertLessEqual(img.abs().max().item(), 1.0 + 1e-4)   # tanh 出力

    def test_overfits_single_image(self):
        # 1 枚を数十ステップで過学習でき、再構成損失が下がる（学習可能性）
        from swi_image.model import ConvVAE
        vae = ConvVAE(in_ch=3, latent_ch=4, base_ch=8, down_factor=4)
        x = torch.rand(1, 3, 32, 32) * 2 - 1
        opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
        first = None
        for _ in range(50):
            opt.zero_grad()
            _, loss, parts = vae(x)
            loss.backward(); opt.step()
            if first is None:
                first = parts["recon"].item()
        self.assertLess(parts["recon"].item(), first)


if __name__ == "__main__":
    unittest.main()
