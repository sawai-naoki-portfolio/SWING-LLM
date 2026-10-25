"""
swi_image/model/dit.py

Diffusion Transformer (DiT) [画像編4]。

VAE 潜在をパッチ列に変換し、タイムステップ + テキスト条件のもとで Flow Matching の
速度場を予測する。条件付けは AdaLN-Zero（時刻+プールしたテキスト）＋テキスト系列への
クロスアテンションで行う。SwiGLU FFN は LLM 実装編と同型 [画像編4.2/4.3]。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger

logger = setup_logger("image.dit")


def timestep_embedding(t, dim, max_period=10000):
    """スカラー時刻 t(0..1) を正弦波埋め込みへ [画像編4]。"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MultiHeadAttention(nn.Module):
    """双方向（因果マスクなし）自己/クロスアテンション。"""

    def __init__(self, dim, heads, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(kv_dim, dim, bias=False)
        self.v = nn.Linear(kv_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def _split(self, x):
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        q, k, v = self._split(self.q(x)), self._split(self.k(ctx)), self._split(self.v(ctx))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim * 4
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    """自己アテンション + テキストクロスアテンション + SwiGLU、各層を AdaLN-Zero で変調。"""

    def __init__(self, dim, heads, cond_dim, text_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = MultiHeadAttention(dim, heads, kv_dim=text_dim)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = SwiGLU(dim)
        # 3 サブ層 × (scale, shift, gate) = 9。ゼロ初期化で学習開始時は恒等に近い挙動。
        self.adaln = nn.Linear(cond_dim, dim * 9)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x, cond, text_seq):
        p = self.adaln(cond).chunk(9, dim=-1)
        s1, sh1, g1, s2, sh2, g2, s3, sh3, g3 = p
        x = x + g1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), s1, sh1))
        x = x + g2.unsqueeze(1) * self.cross(_modulate(self.norm2(x), s2, sh2), text_seq)
        x = x + g3.unsqueeze(1) * self.ffn(_modulate(self.norm3(x), s3, sh3))
        return x


class DiT(nn.Module):
    """潜在 [B, C, H, W] + 時刻 + テキスト → 速度場 [B, C, H, W]。"""

    def __init__(self, latent_ch, latent_size, patch_size, dim, depth, heads, text_dim):
        super().__init__()
        self.latent_ch = latent_ch
        self.patch_size = patch_size
        self.grid = latent_size // patch_size
        n_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(latent_ch, dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        cond_dim = dim
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, cond_dim))
        self.text_pool_proj = nn.Linear(text_dim, cond_dim)
        self.dim = dim

        self.blocks = nn.ModuleList([DiTBlock(dim, heads, cond_dim, text_dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaln_out = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.adaln_out.weight)
        nn.init.zeros_(self.adaln_out.bias)
        self.head = nn.Linear(dim, patch_size * patch_size * latent_ch)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        logger.info(f"Initializing DiT: dim={dim}, depth={depth}, heads={heads}, "
                    f"latent={latent_ch}x{latent_size}x{latent_size}, patches={n_patches}")

    def _unpatchify(self, x):
        # x: [B, N, p*p*C] -> [B, C, H, W]
        b = x.shape[0]
        p, c, g = self.patch_size, self.latent_ch, self.grid
        x = x.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, g * p, g * p)
        return x

    def forward(self, xt, t, text_pooled, text_seq):
        h = self.patch_embed(xt).flatten(2).transpose(1, 2) + self.pos_embed   # [B, N, dim]
        cond = self.t_mlp(timestep_embedding(t, self.dim)) + self.text_pool_proj(text_pooled)
        for blk in self.blocks:
            h = blk(h, cond, text_seq)
        s, sh = self.adaln_out(cond).chunk(2, dim=-1)
        h = _modulate(self.norm_out(h), s, sh)
        return self._unpatchify(self.head(h))                                  # 速度場
