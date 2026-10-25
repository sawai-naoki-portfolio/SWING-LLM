"""
swi_llm/training/lora.py

LoRA (Low-Rank Adaptation) による継続適応（ブループリント §11.1, §10.1）。

ベースモデルの重みを凍結したまま、各線形層に低ランクの学習可能アダプタ
(A·B) を追加する。少ない VRAM で継続学習でき、検証後にベース重みへ
マージすることでベースモデルの汎用能力を守りながら知識を取り込める。

初期状態では B=0 のため LoRA の寄与は 0（forward はベースと一致）。
"""
import math

import torch
import torch.nn as nn

from swi_llm.utils.logger import setup_logger

logger = setup_logger("lora")

# デフォルトで LoRA を挿入する線形層の名前（部分一致）。MLA と SwiGLU の主要射影。
DEFAULT_LORA_TARGETS = ("wq", "w_kv_up", "w_kv_down", "wo", "w1", "w2", "w3")


class LoRALinear(nn.Module):
    """nn.Linear をラップし、低ランクアダプタ A·B を加える。"""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        # A はランダム、B はゼロ初期化 → 学習開始時の寄与は 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    @torch.no_grad()
    def merge(self):
        """LoRA の寄与をベース重みへ畳み込み、通常の nn.Linear を返す。"""
        delta = (self.lora_B @ self.lora_A) * self.scaling  # (out, in)
        self.base.weight.data += delta
        return self.base


def _set_submodule(root, dotted_name, new_module):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def apply_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.0,
               targets=DEFAULT_LORA_TARGETS):
    """model 内の対象 nn.Linear を LoRALinear に置き換え、ベースを凍結する。

    Returns:
        置き換えた層数。
    """
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) or name.endswith("." + t) for t in targets):
            to_replace.append((name, module))

    for name, module in to_replace:
        _set_submodule(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))

    mark_only_lora_trainable(model)
    logger.info(f"Applied LoRA to {len(to_replace)} linear layers (rank={rank}, alpha={alpha}).")
    return len(to_replace)


def mark_only_lora_trainable(model):
    """LoRA パラメータ (lora_) 以外を全て凍結する。"""
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def lora_state_dict(model):
    """LoRA パラメータのみの state_dict（アダプタ保存用）。"""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_all_lora(model):
    """全 LoRALinear をベースへマージして通常 Linear に戻す。"""
    merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            _set_submodule(model, name, module.merge())
            merged += 1
    logger.info(f"Merged {merged} LoRA layers into base weights.")
    return merged
