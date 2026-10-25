"""
swi_llm/core/models/transformer_blocks.py

Transformer 構成部品:
  - RMSNorm: 正規化層
  - precompute_freqs_cis / apply_rotary_emb: RoPE (回転位置エンコーディング)
  - StaticMLACache: MLA 用の INT8 量子化・事前確保 KV キャッシュ
  - MultiHeadLatentAttention (MLA): 低ランク潜在ベクトルによる KV 圧縮アテンション
  - SwiGLU: ゲート付き活性化 FFN
  - SparseMoE: Top-K ルーティングの Mixture of Experts (+ 負荷分散補助損失)
  - TransformerBlock: Pre-Norm 構成の 1 ブロック (MLA + SparseMoE)

注意 (C-1): 各モジュールの属性名・パラメータ形状は既存チェックポイントとの
互換性維持のため変更しないこと。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
logger = setup_logger("transformer_blocks")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

import math


def _yarn_find_correction_dim(num_rotations, dim, theta, orig_len):
    return (dim * math.log(orig_len / (num_rotations * 2 * math.pi))) / (2 * math.log(theta))


def _yarn_correction_range(beta_fast, beta_slow, dim, theta, orig_len):
    low = math.floor(_yarn_find_correction_dim(beta_fast, dim, theta, orig_len))
    high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, theta, orig_len))
    return max(low, 0), min(high, dim // 2 - 1)


def _yarn_linear_ramp(low, high, size):
    if low == high:
        high += 0.001  # ゼロ除算回避
    ramp = (torch.arange(size, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(ramp, 0.0, 1.0)


def _yarn_inv_freq(dim, theta, orig_len, new_len, beta_fast=32, beta_slow=1):
    """YaRN (NTK-by-parts): 低周波は内挿、高周波は外挿を保つ逆周波数を返す。"""
    idx = torch.arange(0, dim, 2)[: (dim // 2)].float()
    pos_freqs = theta ** (idx / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    scale = new_len / orig_len
    inv_freq_interpolation = 1.0 / (scale * pos_freqs)

    low, high = _yarn_correction_range(beta_fast, beta_slow, dim, theta, orig_len)
    # ramp=1 の帯域は外挿(高周波)、ramp=0 の帯域は内挿(低周波)
    extrap_mask = 1.0 - _yarn_linear_ramp(low, high, dim // 2)
    return inv_freq_interpolation * (1 - extrap_mask) + inv_freq_extrapolation * extrap_mask


def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0,
                         original_max_seq_len: int = None, scaling_type: str = None):
    """RoPE の cos/sin を事前計算する。

    長文外挿 (§9.1): scaling_type と original_max_seq_len を指定し、end が学習長を
    超える場合に位置エンコーディングを拡張する。
      - "ntk"  : Dynamic NTK（基底周波数を動的に引き上げる）
      - "yarn" : YaRN（周波数帯域ごとに内挿率を変える。長文安定性が高い）
    無指定または end <= original_max_seq_len のときは従来通り。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if scaling_type and original_max_seq_len and end > original_max_seq_len:
        if scaling_type == "ntk":
            ratio = end / original_max_seq_len
            new_theta = theta * (ratio ** (dim / (dim - 2)))
            inv_freq = 1.0 / (new_theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif scaling_type == "yarn":
            inv_freq = _yarn_inv_freq(dim, theta, original_max_seq_len, end)

    t = torch.arange(end, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class StaticMLACache:
    """Pre-allocated Cache for Multi-Head Latent Attention to avoid O(N^2) memory reallocation."""
    def __init__(self, bsz, max_seq_len, latent_dim, device, dtype):
        self.c_kv_int8 = torch.zeros((bsz, max_seq_len, latent_dim), device=device, dtype=torch.int8)
        self.scales = torch.zeros((bsz, max_seq_len, 1), device=device, dtype=dtype)
        self.seqlen = 0
        self.dtype = dtype

    def update(self, c_kv_new):
        bsz, seqlen, _ = c_kv_new.shape

        # 容量オーバーの防御 (defense in depth): 事前確保領域を越える書き込みは、呼び出し側
        # (generate_stream の容量ガード) が停止して防ぐべき状況。万一ここへ到達した場合、
        # ゼロ幅スライスへの代入による不明瞭な broadcast エラーではなく、原因の分かる
        # ValueError で fail-loud にする（StaticMLACache は推論専用で学習には影響しない）。
        max_seq_len = self.c_kv_int8.shape[1]
        if self.seqlen + seqlen > max_seq_len:
            raise ValueError(
                f"StaticMLACache overflow: writing {seqlen} token(s) at position {self.seqlen} "
                f"exceeds capacity {max_seq_len}. Generation must stop before the cache fills.")

        # INT8 Symmetric Quantization
        abs_max = c_kv_new.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        scale = abs_max / 127.0
        c_kv_int8 = (c_kv_new / scale).round().to(torch.int8)
        
        self.c_kv_int8[:, self.seqlen:self.seqlen+seqlen] = c_kv_int8
        self.scales[:, self.seqlen:self.seqlen+seqlen] = scale
        self.seqlen += seqlen
        
        # Dequantize and return full cache
        full_c_kv_int8 = self.c_kv_int8[:, :self.seqlen]
        full_scales = self.scales[:, :self.seqlen]

        # FIXME (問題6, 性能): 毎生成ステップで全系列長(seqlen)のキャッシュを BF16/FP32 へ
        # 逆量子化（.to(dtype) * scales）して Attention に渡している。INT8 でメモリ「容量」は
        # 節約できるが、128k コンテキストでは毎ステップ巨大テンソルを展開するためメモリ「帯域」と
        # 計算時間が圧迫され、INT8 の恩恵を半分しか受けられない。将来、量子化サービング導入時に
        # Attention カーネル（FlashAttention 拡張 / Triton）が INT8 入力+スケールを受け取り、
        # レジスタ上でオンザフライ逆量子化する実装へ置き換えて解消する。
        return full_c_kv_int8.to(self.dtype) * full_scales

    def rollback(self, length):
        """Rollback the cache by 'length' tokens. Used for Speculative Decoding rejection."""
        self.seqlen = max(0, self.seqlen - length)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) implementation for compressing KV cache.
    Reduces memory bottleneck for extremely long contexts (e.g. 128k+).

    QK-Norm (ブループリント §2.3): use_qk_norm=True のとき Query/Key の各ヘッド
    (head_dim 軸) を RMSNorm で正規化してから RoPE を適用する。深い層・大きい
    バッチでの学習安定性を低コストで引き上げる（Qwen3 が採用）。
    """
    def __init__(self, dim, n_heads, latent_dim=512, use_qk_norm=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.latent_dim = latent_dim
        self.use_qk_norm = use_qk_norm

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w_kv_down = nn.Linear(dim, latent_dim, bias=False)
        self.w_kv_up = nn.Linear(latent_dim, 2 * n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if use_qk_norm:
            # head_dim 軸で正規化（RMSNorm は最終次元に対して作用する）
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, freqs_cos, freqs_sin, is_causal=True, use_cache=False, past_key_value=None, attn_mask=None):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x)
        
        # Latent KV compression
        c_kv = self.w_kv_down(x)
        
        # Cache update
        if isinstance(past_key_value, StaticMLACache):
            full_c_kv = past_key_value.update(c_kv)
            new_cache = past_key_value if use_cache else None
        elif past_key_value is not None:
            full_c_kv = torch.cat([past_key_value, c_kv], dim=1)
            new_cache = full_c_kv if use_cache else None
        else:
            full_c_kv = c_kv
            new_cache = c_kv if use_cache else None

        # Decompress KV for attention
        kv_full = self.w_kv_up(full_c_kv)
        xk, xv = kv_full.chunk(2, dim=-1)
        
        full_seqlen = full_c_kv.shape[1]
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_heads, self.head_dim)

        # QK-Norm (RoPE の前に head_dim 軸で正規化) — §2.3
        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        # Apply RoPE
        xq_cos = freqs_cos[-seqlen:]
        xq_sin = freqs_sin[-seqlen:]
        xk_cos = freqs_cos[:full_seqlen]
        xk_sin = freqs_sin[:full_seqlen]
        
        xq, _ = apply_rotary_emb(xq, xq, xq_cos, xq_sin)
        xk, _ = apply_rotary_emb(xk, xk, xk_cos, xk_sin)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, is_causal=is_causal if attn_mask is None else False)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output), new_cache

# SwiGLU Clamping (§2.7-C): 内部の線形成分の値域を [-CLAMP, CLAMP] にクリップし、
# 極端な活性化値による学習不安定化 (ロススパイク) を防ぐシンプルな安定化策。
SWIGLU_CLAMP_VALUE = 10.0


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # ゲート前活性 (w1) と線形成分 (w3) を [-10, 10] にクランプ (§2.7-C)
        gate = torch.clamp(self.w1(x), -SWIGLU_CLAMP_VALUE, SWIGLU_CLAMP_VALUE)
        linear = torch.clamp(self.w3(x), -SWIGLU_CLAMP_VALUE, SWIGLU_CLAMP_VALUE)
        return self.w2(F.silu(gate) * linear)


try:
    import triton # type: ignore
    import triton.language as tl # type: ignore
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    
    class _DummyTriton:
        def jit(self, fn): return fn
        def __getattr__(self, name): return lambda *args, **kwargs: None
    
    class _DummyTL:
        constexpr = type("constexpr", (), {})
        def __getattr__(self, name): return lambda *args, **kwargs: None

    triton = _DummyTriton()
    tl = _DummyTL()

if HAS_TRITON:
    @triton.jit
    def moe_aggregation_kernel(
        out_ptr, exp_out_ptr, token_indices_ptr, token_weights_ptr,
        dim,
        stride_od, stride_ed,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        tok_idx = tl.load(token_indices_ptr + pid)
        weight = tl.load(token_weights_ptr + pid)
        
        offs_d = tl.arange(0, BLOCK_SIZE)
        mask = offs_d < dim
        
        exp_val = tl.load(exp_out_ptr + pid * stride_ed + offs_d, mask=mask)
        out_val = exp_val * weight
        
        out_ptrs = out_ptr + tok_idx * stride_od + offs_d
        tl.atomic_add(out_ptrs, out_val, mask=mask)

class SparseMoE(nn.Module):
    """Top-K ルーティングの Mixture of Experts。

    DeepSeekMoE 設計（ブループリント §2.4-C/§2.6）:
      - 共有エキスパート（問題1, use_shared_experts）: 全トークンが必ず通過する FFN を 1 本設け、
        一般知識・共通表現を担保する。ルーティング出力に加算する。
      - Auxiliary-loss-free バランシング（問題2, use_aux_free_balancing）: 補助損失で主目的損失を
        邪魔せず、学習不要のバイアス項 `routing_bias` のみで負荷分散する。Top-K 選択は
        (router_logits + routing_bias) で行い、重み係数には元の softmax を使う。

    互換性 (C-1): 上記 2 機能は既定 OFF。OFF のとき state_dict の構造・順伝播は従来と同一で、
    既存チェックポイントをそのまま読める。ON にすると `shared_experts`・`routing_bias` が
    増えるため、有効化する場合は新規に学習し直す必要がある（旧 ckpt からの継続は不可）。
    """
    def __init__(self, dim, hidden_dim, num_experts, top_k, z_loss_weight=1e-4,
                 use_shared_experts=False, use_aux_free_balancing=False,
                 bias_update_rate=1e-3):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.z_loss_weight = z_loss_weight
        self.use_shared_experts = use_shared_experts
        self.use_aux_free_balancing = use_aux_free_balancing
        self.bias_update_rate = bias_update_rate
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])
        # 問題1: 共有エキスパート（全トークン通過）。既定 OFF で旧 ckpt 互換。
        if use_shared_experts:
            self.shared_experts = SwiGLU(dim, hidden_dim)
        # 問題2: aux-loss-free 用のルーティングバイアス（勾配ではなくヒューリスティックで更新）。
        # 学習済みバランシングを推論へ引き継ぐため persistent=True（フラグ ON 時のみ増える）。
        if use_aux_free_balancing:
            self.register_buffer("routing_bias", torch.zeros(num_experts))
        # L-4: aux_loss が NaN になったフォールバック回数（可視化用, 非 state_dict）。
        self._nan_aux_count = 0

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        router_logits = self.gate(x_flat)
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() * self.z_loss_weight

        routing_probs = F.softmax(router_logits, dim=-1)
        if self.use_aux_free_balancing:
            # 選択はバイアス込みスコアで、重み係数は元の親和度(softmax)で（問題2）。
            biased = router_logits + self.routing_bias
            _, selected_experts = torch.topk(biased, self.top_k, dim=-1)
            routing_weights = torch.gather(routing_probs, -1, selected_experts)
        else:
            routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        final_output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            token_indices, topk_idx = torch.where(selected_experts == i)
            if token_indices.numel() == 0:
                continue
            expert_tokens = x_flat[token_indices]
            expert_out = expert(expert_tokens)
            token_weights = routing_weights[token_indices, topk_idx].squeeze(-1)

            if HAS_TRITON and expert_out.is_cuda:
                BLOCK_SIZE = triton.next_power_of_2(self.dim)
                grid = lambda meta: (expert_out.shape[0], )
                moe_aggregation_kernel[grid](
                    final_output, expert_out, token_indices, token_weights,
                    self.dim,
                    final_output.stride(0), expert_out.stride(0),
                    BLOCK_SIZE=BLOCK_SIZE
                )
            else:
                final_output.index_add_(0, token_indices, expert_out * token_weights.unsqueeze(-1))

        # 問題1: 全トークンを共有エキスパートに通し、ルーティング出力へ加算。
        if self.use_shared_experts:
            final_output = final_output + self.shared_experts(x_flat)

        final_output = final_output.view(bsz, seq_len, dim)

        tokens_per_expert = torch.bincount(selected_experts.flatten(), minlength=self.num_experts)
        fraction_tokens_per_expert = tokens_per_expert.float() / selected_experts.numel()

        if self.use_aux_free_balancing:
            # 問題2: 補助損失を加えず、バイアス項のみで負荷分散する。学習時のみ、目標頻度
            # (1/num_experts) を超えたエキスパートのバイアスを下げ、過少を上げる（勾配不使用）。
            if self.training:
                with torch.no_grad():
                    target = 1.0 / self.num_experts
                    self.routing_bias -= self.bias_update_rate * torch.sign(
                        fraction_tokens_per_expert - target)
            total_aux_loss = z_loss   # ルーター安定化目的の z_loss のみ残す
        else:
            prob_per_expert = routing_probs.mean(0)
            aux_loss = self.num_experts * torch.sum(fraction_tokens_per_expert * prob_per_expert)
            total_aux_loss = aux_loss + z_loss

        # L-4: NaN 時は「グラフから切り離した新規リーフ」ではなく nan_to_num で 0 化する。
        # 新規リーフに差し替えると backward で上流ルーターへ勾配が流れず、その step の
        # 負荷分散/ルーター学習が丸ごと無効化されるため。発生回数を集計し監視可能にする。
        if torch.isnan(total_aux_loss).any() or torch.isinf(total_aux_loss).any():
            self._nan_aux_count += 1
            logger.error(f"NaN/Inf detected in SparseMoE auxiliary loss (occurrence #{self._nan_aux_count})! "
                         f"Sanitizing to 0 while preserving graph connection.")
            total_aux_loss = torch.nan_to_num(total_aux_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return final_output, total_aux_loss

class TransformerBlock(nn.Module):
    """Pre-Norm 構成の Transformer ブロック (MLA + Sparse MoE)。

    Note:
        MLA 移行に伴い GQA 用の `n_kv_heads` 引数は廃止済み (REQ-16)。
    """
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, num_experts: int,
                 top_k: int, dropout: float = 0.1, z_loss_weight: float = 1e-4,
                 use_qk_norm: bool = True, use_shared_experts: bool = False,
                 use_aux_free_balancing: bool = False, moe_bias_update_rate: float = 1e-3):
        super().__init__()
        self.attention = MultiHeadLatentAttention(dim, n_heads, latent_dim=512, use_qk_norm=use_qk_norm)
        self.moe = SparseMoE(dim, hidden_dim, num_experts, top_k, z_loss_weight=z_loss_weight,
                             use_shared_experts=use_shared_experts,
                             use_aux_free_balancing=use_aux_free_balancing,
                             bias_update_rate=moe_bias_update_rate)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, freqs_cos, freqs_sin, is_causal=True, use_cache=False, past_key_value=None, attn_mask=None):
        attn_out, new_cache = self.attention(self.attention_norm(x), freqs_cos, freqs_sin, is_causal=is_causal,
                                              use_cache=use_cache, past_key_value=past_key_value, attn_mask=attn_mask)
        h = x + self.dropout1(attn_out)
        moe_out, aux_loss = self.moe(self.ffn_norm(h))
        out = h + self.dropout2(moe_out)
        return out, aux_loss, new_cache
