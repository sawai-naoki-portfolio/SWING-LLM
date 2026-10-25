"""
swi_image/model/vae.py

VAE（潜在空間エンコーダ/デコーダ）[画像編2]。

生画像（高次元）を DiT が扱いやすい低次元の潜在空間へ圧縮/復元する。DiT は
この潜在空間上で Flow Matching を学習する（ピクセル空間より計算量が大幅に減る）。

Phase 0 は自作の小型 Conv VAE（L1 再構成 + KL 損失）。実運用では高品質な公開
VAE を流用する方が費用対効果が高い（[画像編2 実装方針]）が、from-scratch 方針の
プロトタイプとして最小構成を用意する。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
from swi_image.utils.constants import IMG_VAE_KL_WEIGHT, IMG_VAE_SCALE_FACTOR

logger = setup_logger("image.vae")


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class VAEEncoder(nn.Module):
    def __init__(self, in_ch, latent_ch, base_ch, n_down):
        super().__init__()
        self.inp = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        chs = [min(base_ch * (2 ** i), 256) for i in range(n_down + 1)]
        blocks = []
        for i in range(n_down):
            blocks += [ResBlock(chs[i]), nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1)]
        self.body = nn.Sequential(*blocks)
        self.out = nn.Sequential(nn.GroupNorm(min(8, chs[-1]), chs[-1]), nn.SiLU(),
                                 nn.Conv2d(chs[-1], latent_ch * 2, 1))

    def forward(self, x):
        h = self.body(self.inp(x))
        mean, logvar = self.out(h).chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)


class VAEDecoder(nn.Module):
    def __init__(self, out_ch, latent_ch, base_ch, n_down):
        super().__init__()
        chs = [min(base_ch * (2 ** i), 256) for i in range(n_down + 1)]
        self.inp = nn.Conv2d(latent_ch, chs[-1], 3, padding=1)
        blocks = []
        for i in range(n_down, 0, -1):
            blocks += [ResBlock(chs[i]), nn.ConvTranspose2d(chs[i], chs[i - 1], 4, stride=2, padding=1)]
        self.body = nn.Sequential(*blocks)
        self.out = nn.Sequential(nn.GroupNorm(min(8, chs[0]), chs[0]), nn.SiLU(),
                                 nn.Conv2d(chs[0], out_ch, 3, padding=1))

    def forward(self, z):
        h = self.body(self.inp(z))
        return torch.tanh(self.out(h))          # 画像は [-1, 1]


def reparameterize(mean, logvar):
    std = (0.5 * logvar).exp()
    return mean + std * torch.randn_like(std)


def vae_loss(x, x_recon, mean, logvar, kl_weight=IMG_VAE_KL_WEIGHT):
    """L1 再構成損失 + KL 損失 [画像編2.2]。"""
    recon_loss = F.l1_loss(x_recon, x)
    kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


class ConvVAE(nn.Module):
    """小型 Conv VAE。encode → 潜在、decode → 画像。"""

    def __init__(self, in_ch=3, latent_ch=4, base_ch=64, down_factor=8,
                 scale_factor=IMG_VAE_SCALE_FACTOR):
        super().__init__()
        assert down_factor >= 2 and (down_factor & (down_factor - 1)) == 0, "down_factor は2の冪"
        n_down = int(math.log2(down_factor))
        self.encoder = VAEEncoder(in_ch, latent_ch, base_ch, n_down)
        self.decoder = VAEDecoder(in_ch, latent_ch, base_ch, n_down)
        self.latent_ch = latent_ch
        self.down_factor = down_factor
        self.scale_factor = scale_factor
        logger.info(f"Initializing ConvVAE: latent_ch={latent_ch}, base_ch={base_ch}, "
                    f"down_factor={down_factor}")

    def encode(self, x):
        return self.encoder(x)                  # mean, logvar

    def encode_latent(self, x):
        """DiT 学習用: スケール済み潜在（サンプリング）を返す。"""
        mean, logvar = self.encoder(x)
        return reparameterize(mean, logvar) * self.scale_factor

    def decode(self, z, scaled=True):
        if scaled:
            z = z / self.scale_factor
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = reparameterize(mean, logvar)
        recon = self.decoder(z)
        loss, recon_loss, kl_loss = vae_loss(x, recon, mean, logvar)
        return recon, loss, {"recon": recon_loss, "kl": kl_loss}


def vae_from_ckpt_cfg(vae_cfg):
    """チェックポイントの vae_cfg から ConvVAE を組み立てる（I-2）。

    scale_factor は ckpt に保存された値を使う。以前は保存しておらず、IMG_VAE_SCALE_FACTOR を
    変えると既存の VAE/DiT が例外なしに画質劣化した。キーの無い旧 ckpt だけ定数で補う。
    """
    scale = vae_cfg.get("scale_factor")
    if scale is None:
        logger.warning("VAE checkpoint has no scale_factor (legacy); assuming "
                       f"IMG_VAE_SCALE_FACTOR={IMG_VAE_SCALE_FACTOR}. Re-save the VAE to pin it.")
        scale = IMG_VAE_SCALE_FACTOR
    return ConvVAE(in_ch=3, latent_ch=vae_cfg["latent_ch"], base_ch=vae_cfg["base_ch"],
                   down_factor=vae_cfg["down_factor"], scale_factor=float(scale))
