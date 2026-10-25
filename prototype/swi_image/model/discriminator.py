"""
swi_image/model/discriminator.py
VAE-GAN 用の PatchGAN 判別器 + hinge 損失 [画像編2.2, NEW-I4]

画像全体を 1 スカラーで判定する代わりに、局所パッチごとの real/fake ロジットマップを
出力する（PatchGAN）。高周波のテクスチャ・エッジの「もっともらしさ」に敏感で、
VAE 再構成のぼやけ（recon 損失だけでは平均化する）を敵対的に補正する。

損失は hinge（SN-GAN / VQGAN 系の標準）:
    D: relu(1 - D(real)) + relu(1 + D(fake))
    G: -D(fake)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger

logger = setup_logger("image.discriminator")


class PatchDiscriminator(nn.Module):
    """PatchGAN 判別器。入力 [B,3,H,W] → パッチロジット [B,1,h,w]。"""

    def __init__(self, in_ch=3, base_ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        ch = base_ch
        for i in range(1, n_layers):
            nxt = min(ch * 2, base_ch * 8)
            stride = 2 if i < n_layers - 1 else 1
            layers += [nn.Conv2d(ch, nxt, 4, stride=stride, padding=1, bias=False),
                       nn.GroupNorm(min(8, nxt), nxt),
                       nn.LeakyReLU(0.2, inplace=True)]
            ch = nxt
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)
        logger.info(f"PatchDiscriminator: base_ch={base_ch}, n_layers={n_layers}")

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real_logits, fake_logits):
    """判別器の hinge 損失（real を +1 以上、fake を -1 以下へ押す）。"""
    return (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def hinge_g_loss(fake_logits):
    """生成側（VAE）の hinge 損失（fake のロジットを引き上げる）。"""
    return -fake_logits.mean()
