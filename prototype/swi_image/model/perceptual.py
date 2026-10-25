"""
swi_image/model/perceptual.py
VAE 学習の知覚的損失（LPIPS 系）[画像編2.2, NEW-I4 / 02_architecture_docs_gap_audit]

設計は VAE の学習損失として「再構成損失、KL、**知覚的損失（LPIPS）、敵対的損失**」を
要求するが、従来は recon(+KL) のみで高周波ディテールを担保する 2 損失が欠落していた。

方式: ImageNet 学習済み CNN の中間特徴space で再構成と原画像の距離を測る（LPIPS の骨格）。
既定 backbone は squeezenet1_1（重み ~5MB・初回のみ DL。LPIPS 公式も squeeze 変種を持つ）。
PROD で品質を追い込む場合は VGG16（~528MB）へ設定で切替できる。
バックボーンは凍結し、勾配は入力（VAE の再構成）へのみ流す。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger

logger = setup_logger("image.perceptual")


class PerceptualLoss(nn.Module):
    """凍結 CNN 特徴の L1 距離による知覚的損失。入力は [-1, 1] 正規化の画像。"""

    def __init__(self, backbone="squeezenet1_1"):
        super().__init__()
        from torchvision import models
        if backbone == "vgg16":
            net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        elif backbone == "squeezenet1_1":
            net = models.squeezenet1_1(
                weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).features[:8]
        else:
            raise ValueError(f"unknown perceptual backbone: {backbone}")
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        logger.info(f"PerceptualLoss ready (backbone={backbone}, frozen).")

    def _prep(self, x):
        # [-1,1] → [0,1] → ImageNet 正規化
        return ((x + 1.0) * 0.5 - self.mean) / self.std

    def forward(self, recon, target):
        return F.l1_loss(self.net(self._prep(recon)), self.net(self._prep(target)))
