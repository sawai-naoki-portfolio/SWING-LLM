"""
tests/unit/test_image_vae_gan.py

NEW-I4（画像編2.2）: VAE の知覚的損失 + 敵対的損失（PatchGAN + hinge）。
"""
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_image.model.discriminator import PatchDiscriminator, hinge_d_loss, hinge_g_loss


class TestPatchDiscriminator(unittest.TestCase):
    def test_outputs_patch_logit_map(self):
        d = PatchDiscriminator(base_ch=16, n_layers=2)
        out = d(torch.randn(2, 3, 32, 32))
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)
        self.assertGreater(out.shape[2] * out.shape[3], 1)     # 1 スカラーではなくパッチマップ
        self.assertTrue(torch.isfinite(out).all())

    def test_gradients_flow(self):
        d = PatchDiscriminator(base_ch=16, n_layers=2)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        hinge_g_loss(d(x)).backward()
        self.assertIsNotNone(x.grad)                           # 生成側へ勾配が流れる
        self.assertTrue(torch.isfinite(x.grad).all())


class TestHingeLosses(unittest.TestCase):
    def test_d_loss_is_zero_when_separated(self):
        real = torch.full((2, 1, 3, 3), 2.0)
        fake = torch.full((2, 1, 3, 3), -2.0)
        self.assertAlmostEqual(hinge_d_loss(real, fake).item(), 0.0)

    def test_d_loss_positive_when_confused(self):
        real = torch.zeros(2, 1, 3, 3)
        fake = torch.zeros(2, 1, 3, 3)
        self.assertGreater(hinge_d_loss(real, fake).item(), 0.0)

    def test_g_loss_decreases_as_fake_logits_rise(self):
        low = hinge_g_loss(torch.full((2, 1, 3, 3), -1.0))
        high = hinge_g_loss(torch.full((2, 1, 3, 3), 1.0))
        self.assertGreater(low.item(), high.item())


class TestPerceptualLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from swi_image.model.perceptual import PerceptualLoss
            cls.loss = PerceptualLoss(backbone="squeezenet1_1")
        except Exception as e:                                  # 重み取得不可（オフライン）
            raise unittest.SkipTest(f"perceptual backbone unavailable: {e}")

    def test_identical_images_are_zero(self):
        x = torch.rand(1, 3, 32, 32) * 2 - 1
        self.assertAlmostEqual(self.loss(x, x).item(), 0.0, places=6)

    def test_different_images_positive_and_backprops(self):
        torch.manual_seed(0)
        x = (torch.rand(1, 3, 32, 32) * 2 - 1).requires_grad_(True)
        y = torch.rand(1, 3, 32, 32) * 2 - 1
        val = self.loss(x, y)
        self.assertGreater(val.item(), 0.0)
        val.backward()
        self.assertIsNotNone(x.grad)                           # VAE の再構成へ勾配が流れる

    def test_backbone_is_frozen(self):
        self.assertTrue(all(not p.requires_grad for p in self.loss.net.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
