"""
I-2: VAE の scale_factor をチェックポイントに保存し、読み込み側は ckpt の値を使うことの回帰テスト。

以前は scale_factor を保存しておらず、IMG_VAE_SCALE_FACTOR を変えると既存の VAE / DiT が
例外なしに（潜在のスケールがずれたまま）画質劣化した。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_image.model.vae import vae_from_ckpt_cfg
from swi_image.utils.constants import IMG_VAE_SCALE_FACTOR

_PROTO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CFG = {"latent_ch": 4, "base_ch": 8, "down_factor": 4}


class TestVaeScaleFactor(unittest.TestCase):
    def test_checkpoint_value_wins_over_constant(self):
        vae = vae_from_ckpt_cfg({**_CFG, "scale_factor": 0.5})
        self.assertEqual(vae.scale_factor, 0.5)
        self.assertNotEqual(vae.scale_factor, IMG_VAE_SCALE_FACTOR)

    def test_legacy_checkpoint_falls_back_to_constant_with_warning(self):
        with self.assertLogs(level="WARNING") as cm:
            vae = vae_from_ckpt_cfg(dict(_CFG))
        self.assertEqual(vae.scale_factor, IMG_VAE_SCALE_FACTOR)
        self.assertIn("legacy", "\n".join(cm.output))

    def test_trainer_saves_scale_factor(self):
        src = open(os.path.join(_PROTO, "swi_image", "train", "vae_train.py"), encoding="utf-8").read()
        self.assertIn('"scale_factor": vae.scale_factor', src)

    def test_every_loader_uses_the_checkpoint_value(self):
        """VAE ckpt を読む全経路が ConvVAE を直接組まず vae_from_ckpt_cfg を通ること。"""
        for rel in ("swi_image/serve/generate.py", "swi_image/train/dit_train.py",
                    "swi_image/eval/image_eval.py", "swi_image/pipeline_check.py"):
            src = open(os.path.join(_PROTO, rel), encoding="utf-8").read()
            self.assertNotIn("ConvVAE(", src, rel)
            self.assertIn("vae_from_ckpt_cfg(", src, rel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
