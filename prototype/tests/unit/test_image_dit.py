"""
画像 DiT / テキストエンコーダ / Flow Matching (swi_image/model) の単体テスト [画像編3・4・5]。
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
class TestTextEncoder(unittest.TestCase):
    def test_shapes(self):
        from swi_image.model import TextEncoder
        te = TextEncoder(vocab_size=100, dim=32, layers=1, heads=2, max_len=16)
        ids = torch.randint(0, 100, (2, 5))
        out = te(ids)
        self.assertEqual(out["sequence"].shape, (2, 5, 32))
        self.assertEqual(out["pooled"].shape, (2, 32))
        null = te.null_condition(2, torch.device("cpu"))
        self.assertEqual(null["pooled"].shape, (2, 32))


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestDiTFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swi_image.model import DiT, TextEncoder
        cls.torch = torch
        cls.latent_ch, cls.lat = 4, 8
        cls.te = TextEncoder(vocab_size=100, dim=32, layers=1, heads=2, max_len=16)
        cls.dit = DiT(latent_ch=cls.latent_ch, latent_size=cls.lat, patch_size=2,
                      dim=64, depth=2, heads=4, text_dim=32)
        cls.ids = torch.randint(0, 100, (2, 5))
        cls.latents = torch.randn(2, cls.latent_ch, cls.lat, cls.lat)

    def test_forward_velocity_shape(self):
        cond = self.te(self.ids)
        t = torch.rand(2)
        v = self.dit(self.latents, t, cond["pooled"], cond["sequence"])
        self.assertEqual(v.shape, self.latents.shape)
        self.assertTrue(torch.isfinite(v).all())

    def test_flow_loss_and_backward(self):
        from swi_image.model import rectified_flow_loss
        cond = self.te(self.ids)
        loss = rectified_flow_loss(self.dit, self.latents, cond)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_sample_euler_shape(self):
        from swi_image.model import sample_euler
        cond = self.te(self.ids)
        null = self.te.null_condition(2, torch.device("cpu"))
        out = sample_euler(self.dit, (2, self.latent_ch, self.lat, self.lat),
                           cond, uncond=null, n_steps=4, cfg_scale=3.0, device="cpu")
        self.assertEqual(out.shape, (2, self.latent_ch, self.lat, self.lat))
        self.assertTrue(torch.isfinite(out).all())

    def test_overfits_single_latent(self):
        from swi_image.model import DiT, TextEncoder, rectified_flow_loss
        te = TextEncoder(vocab_size=50, dim=32, layers=1, heads=2, max_len=8)
        dit = DiT(latent_ch=4, latent_size=8, patch_size=2, dim=64, depth=2, heads=4, text_dim=32)
        ids = torch.randint(0, 50, (1, 3))
        lat = torch.randn(1, 4, 8, 8)
        params = list(dit.parameters()) + list(te.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        torch.manual_seed(0)
        first = None
        for _ in range(60):
            opt.zero_grad()
            loss = rectified_flow_loss(dit, lat, te(ids))
            loss.backward(); opt.step()
            if first is None:
                first = loss.item()
        self.assertLess(loss.item(), first)     # 学習で損失低下


if __name__ == "__main__":
    unittest.main()
