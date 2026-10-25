"""
swi_voice/model/flow_tts.py

Flow Matching ベースの TTS（F5-TTS / E2-TTS 系）[音声編3.2]。

VALL-E 系（ar_codec_lm.py）が自己回帰で 1 フレームずつ生成するのに対し、本方式は
**非自己回帰**で系列全体を並列生成する。ニューラルコーデックの連続 latent
（`codec.encoder(wav)` の出力, [B, T', dim]）を対象に、ノイズ→データを結ぶ直線経路の
速度場を回帰する Rectified Flow（画像編4.1 と同型）で学習し、生成は Euler ODE で行う。

ゼロショットクローニングは「音声インペインティング」として自然に獲得する:
  参照音声の latent を文脈（context）として与え、マスクした区間をテキスト条件のもとで
  復元するタスクを学習する。専用の話者エンコーダを必要としない [音声編3.2]。

生成した latent は `codec.decoder(latent.transpose(1,2))` で波形へ復元する
（デコーダは量子化コードでなく連続 latent を受け取れる。rvq.py 参照）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger

logger = setup_logger("voice.flow_tts")


# =====================================================================
# 1D-DiT 部品（画像編 DiT と同型。swi_voice を swi_image へ依存させないため自前実装）
# =====================================================================
def timestep_embedding(t, dim, max_period=10000):
    """スカラー時刻 t(0..1) を正弦波埋め込み [B] -> [B, dim] へ。"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([args.cos(), args.sin()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _modulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _MHA(nn.Module):
    """双方向（因果マスク無し）自己/クロスアテンション。"""

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


class _SwiGLU(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim * 4
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _FlowDiTBlock(nn.Module):
    """自己アテンション + テキストクロスアテンション + SwiGLU、各層 AdaLN-Zero 変調。"""

    def __init__(self, dim, heads, text_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = _MHA(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = _MHA(dim, heads, kv_dim=text_dim)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = _SwiGLU(dim)
        # 3 サブ層 ×(scale, shift, gate)=9。ゼロ初期化で学習開始時は恒等に近い。
        self.adaln = nn.Linear(dim, dim * 9)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(self, x, cond, text_seq):
        s1, sh1, g1, s2, sh2, g2, s3, sh3, g3 = self.adaln(cond).chunk(9, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), s1, sh1))
        x = x + g2.unsqueeze(1) * self.cross(_modulate(self.norm2(x), s2, sh2), text_seq)
        x = x + g3.unsqueeze(1) * self.ffn(_modulate(self.norm3(x), s3, sh3))
        return x


# =====================================================================
# FlowTTS 本体
# =====================================================================
class FlowTTS(nn.Module):
    """コーデック連続 latent 上の Flow Matching TTS [音声編3.2]。

    forward(xt, t, text_ids, context) -> 速度場 [B, T', latent_dim]。
      xt:      ノイズ補間された latent [B, T', latent_dim]
      t:       時刻 [B]（0..1）
      text_ids:テキストトークン [B, Lt]
      context: 文脈 latent [B, T', latent_dim]（マスク区間は 0。クローニングの参照や
               インペインティングの可視区間を条件として与える）
    """

    def __init__(self, text_vocab_size, latent_dim, dim=256, n_layers=4, n_heads=4,
                 max_frames=512, max_text_len=256, use_style=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.max_frames = max_frames
        # スタイル/感情制御 [音声編4]。既定 False（C-1: 既存チェックポイント互換のため
        # style_proj を追加しない）。True のとき style_emb を条件付けへ加算する。
        self.use_style = use_style
        if use_style:
            self.style_proj = nn.Linear(dim, dim)

        self.text_emb = nn.Embedding(text_vocab_size, dim)
        self.text_pos = nn.Parameter(torch.zeros(1, max_text_len, dim))
        nn.init.normal_(self.text_pos, std=0.02)

        # 入力: [xt, context] を連結（速度を求める対象と可視文脈を同時に条件付け）
        self.in_proj = nn.Linear(latent_dim * 2, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text_pool_proj = nn.Linear(dim, dim)

        self.blocks = nn.ModuleList([_FlowDiTBlock(dim, n_heads, text_dim=dim) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaln_out = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.adaln_out.weight)
        nn.init.zeros_(self.adaln_out.bias)
        # 速度場ヘッド。ゼロ初期化で学習開始時の速度予測を 0（AdaLN-Zero の思想）に。
        self.out_proj = nn.Linear(dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        logger.info(f"Initializing FlowTTS: dim={dim}, layers={n_layers}, heads={n_heads}, "
                    f"latent_dim={latent_dim}, max_frames={max_frames}")

    def _encode_text(self, text_ids):
        Lt = text_ids.shape[1]
        seq = self.text_emb(text_ids) + self.text_pos[:, :Lt]     # [B, Lt, dim]
        pooled = seq.mean(dim=1)                                  # [B, dim]
        return seq, pooled

    def forward(self, xt, t, text_ids, context, style_emb=None):
        T = xt.shape[1]
        text_seq, text_pooled = self._encode_text(text_ids)
        h = self.in_proj(torch.cat([xt, context], dim=-1)) + self.pos_embed[:, :T]   # [B, T, dim]
        cond = self.t_mlp(timestep_embedding(t, self.dim)) + self.text_pool_proj(text_pooled)
        # スタイル/感情条件（任意）。use_style かつ style_emb 指定時のみ加算（既定は不変）。
        if self.use_style and style_emb is not None:
            cond = cond + self.style_proj(style_emb)
        for blk in self.blocks:
            h = blk(h, cond, text_seq)
        s, sh = self.adaln_out(cond).chunk(2, dim=-1)
        h = _modulate(self.norm_out(h), s, sh)
        return self.out_proj(h)                                  # 速度場 [B, T, latent_dim]


def _span_mask(B, T, mask_ratio, device):
    """各サンプルにランダムなマスク区間を作る。戻り値 mask[B,T,1]（1=マスク=損失対象）。"""
    mask_len = max(1, min(int(T * mask_ratio), T - 1)) if T > 1 else 1
    mask = torch.zeros(B, T, 1, device=device)
    for i in range(B):
        start = int(torch.randint(0, max(1, T - mask_len + 1), (1,)).item())
        mask[i, start:start + mask_len] = 1.0
    return mask


def speech_infilling_loss(model, latent, text_ids, mask_ratio=0.7, style_emb=None):
    """音声インペインティングによる Rectified Flow 損失 [音声編3.2]。

    latent [B, T, D] の一部区間をマスクし、周囲文脈(context)とテキストから復元する。
    Rectified Flow: xt=(1-t)x0 + t*noise, target_v = noise - x0。損失はマスク区間のみ。
    style_emb を渡すと感情/スタイル条件下で学習する [音声編4]（既定 None は無条件）。
    """
    B, T, D = latent.shape
    device = latent.device
    mask = _span_mask(B, T, mask_ratio, device)                  # [B, T, 1]
    context = latent * (1.0 - mask)                              # 可視区間のみ（マスク区間は 0）

    t = torch.rand(B, device=device)
    noise = torch.randn_like(latent)
    t_ = t.view(B, 1, 1)
    xt = (1 - t_) * latent + t_ * noise                         # 直線補間
    target_v = noise - latent                                    # 直線の速度は一定
    pred_v = model(xt, t, text_ids, context, style_emb=style_emb)
    # マスク区間のみで損失を測る（可視区間は文脈として与えているので学習対象外）
    denom = mask.sum() * D + 1e-8
    return ((pred_v - target_v).pow(2) * mask).sum() / denom


@torch.no_grad()
def sample(model, text_ids, n_frames, ref_latent=None, n_steps=16, device=None, style_emb=None):
    """テキスト（+参照音声 latent）から latent [B, n_frames, D] を Euler ODE で生成する。

    ref_latent [B, Lp, D] を渡すと、その区間を可視文脈として先頭に固定し、続きを生成する
    （ゼロショットクローニング）。None なら全区間をノイズから生成する。
    返り値は生成した latent（参照区間は含めない）。`codec.decoder(latent.transpose(1,2))` で波形化。
    """
    model.eval()
    B = text_ids.shape[0]
    device = device or text_ids.device
    D = model.latent_dim

    Lp = ref_latent.shape[1] if ref_latent is not None else 0
    total = Lp + n_frames
    # 文脈: 参照区間は可視(ref_latent)、生成区間は 0（マスク相当）
    context = torch.zeros(B, total, D, device=device)
    if Lp:
        context[:, :Lp] = ref_latent.to(device)

    x = torch.randn(B, total, D, device=device)
    if Lp:
        x[:, :Lp] = ref_latent.to(device)                       # 参照区間はノイズにせず固定
    dt = 1.0 / n_steps
    for i in range(n_steps, 0, -1):
        t = torch.full((B,), i / n_steps, device=device)
        v = model(x, t, text_ids, context, style_emb=style_emb)
        x = x - v * dt
        if Lp:
            x[:, :Lp] = ref_latent.to(device)                   # 参照区間は各ステップで固定し直す
    return x[:, Lp:]                                             # 生成区間のみ [B, n_frames, D]
