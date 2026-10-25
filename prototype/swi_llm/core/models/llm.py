"""
swi_llm/core/models/llm.py

SWI-LLM テキストモデル本体 (MiniLLM)。

構成: Token Embedding → TransformerBlock × n_layers (MLA + Sparse MoE)
      → RMSNorm → LM Head。加えて MTP ヘッド (Multi-Token Prediction) と
      PRM ヘッド (Process Reward Model) を持つ。

注意 (C-1): 本ファイルのモジュール属性名・state_dict のキー構造・テンソル形状は
既存チェックポイントとの互換性維持のため変更しないこと。
"""
import torch
import torch.nn as nn

from swi_llm.core.models.transformer_blocks import TransformerBlock, RMSNorm, precompute_freqs_cis
from swi_llm.utils.logger import setup_logger

logger = setup_logger("llm_model")

class MiniLLM(nn.Module):
    """MLA + Sparse MoE + MTP を備えた SWI-LLM のテキストモデル本体。

    Note:
        アテンションは Multi-Head Latent Attention (MLA) のため、GQA 時代の
        `n_kv_heads` 引数は廃止済み (REQ-16)。state_dict のキー構造・テンソル形状は
        従来と互換 (C-1)。
    """
    def __init__(self, vocab_size: int, dim: int = 512, n_layers: int = 4, n_heads: int = 8,
                 hidden_dim: int = 1433, num_experts: int = 8, top_k: int = 2,
                 max_seq_len: int = 1024, dropout: float = 0.1,
                 z_loss_weight: float = 1e-4, n_mtp_future: int = 0, use_qk_norm: bool = True,
                 rope_original_seq_len: int = None, rope_scaling_type: str = None,
                 use_shared_experts: bool = False, use_aux_free_balancing: bool = False,
                 moe_bias_update_rate: float = 1e-3):
        super().__init__()
        logger.info(f"Initializing MiniLLM: vocab={vocab_size}, dim={dim}, layers={n_layers}, heads={n_heads}, experts={num_experts}, mtp_future={n_mtp_future}, qk_norm={use_qk_norm}, rope_scaling={rope_scaling_type}")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.prm_head = nn.Linear(dim, 1, bias=False) # Process Reward Model head

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim, num_experts, top_k, dropout, z_loss_weight,
                             use_qk_norm=use_qk_norm, use_shared_experts=use_shared_experts,
                             use_aux_free_balancing=use_aux_free_balancing,
                             moe_bias_update_rate=moe_bias_update_rate)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        self.n_mtp_future = n_mtp_future
        if n_mtp_future > 0:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(dim, vocab_size, bias=False) for _ in range(n_mtp_future)
            ])
            
        # 長文外挿 (§9.1): 生成長が学習長を超える場合、YaRN/NTK で位置エンコーディングを拡張。
        # L-10: freqs は導出可能な定数テーブルのため persistent=False とし state_dict に含めない
        # （ARCodecLM と統一）。max_seq_len=131072 で構築すると ~67MB になり保存が無駄。旧
        # チェックポイント（freqs_cos/freqs_sin を含む）はロード側が該当キーを除外して読むため互換。
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim // n_heads, max_seq_len * 2,
            original_max_seq_len=rope_original_seq_len, scaling_type=rope_scaling_type)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, tokens, start_pos=0, use_cache=False, past_key_values=None, vision_features=None, return_mtp=False, return_prm=False):
        """順伝播。

        Args:
            tokens: (bsz, seqlen) のトークンID。
            start_pos: KVキャッシュ使用時の生成開始位置。
            use_cache: True なら KV キャッシュ (StaticMLACache) を更新して返す。
            past_key_values: 各層のキャッシュのリスト。
            vision_features: VLM 用の視覚特徴 (bsz, n_vision, dim)。テキスト埋め込みの前に連結される。
            return_mtp: True なら MTP ヘッドの logits も返す（投機的デコーディング用）。
            return_prm: True なら PRM ヘッドの logits も返す。

        Returns:
            (logits, aux_loss[, new_key_values][, mtp_logits][, prm_logits])
            返却タプルの構成は use_cache / return_mtp / return_prm の組み合わせで決まる。
        """
        bsz, seqlen = tokens.shape
        h_text = self.tok_embeddings(tokens)
        
        if vision_features is not None:
            h = torch.cat([vision_features, h_text], dim=1)
            total_seqlen = h.shape[1]
            num_vision = vision_features.shape[1]
        else:
            h = h_text
            total_seqlen = seqlen
            num_vision = 0
            
        h = self.emb_dropout(h)
        
        freqs_cos = self.freqs_cos
        freqs_sin = self.freqs_sin
        
        cache_num_vision = 0
        if use_cache and past_key_values is not None:
            if hasattr(past_key_values[0], 'seqlen'): # StaticMLACache
                cache_seqlen = past_key_values[0].seqlen
            else:
                cache_seqlen = past_key_values[0].shape[1]
            cache_num_vision = max(0, cache_seqlen - start_pos)
            
        total_num_vision = num_vision + cache_num_vision
        
        if total_num_vision > 0:
            vision_cos = torch.ones((total_num_vision, freqs_cos.shape[-1]), device=freqs_cos.device, dtype=freqs_cos.dtype)
            vision_sin = torch.zeros((total_num_vision, freqs_sin.shape[-1]), device=freqs_sin.device, dtype=freqs_sin.dtype)
            freqs_cos = torch.cat([vision_cos, freqs_cos], dim=0)
            freqs_sin = torch.cat([vision_sin, freqs_sin], dim=0)
        
        # Only pass the relevant slice to the layers
        total_len = total_num_vision + start_pos + seqlen
        freqs_cos_slice = freqs_cos[:total_len]
        freqs_sin_slice = freqs_sin[:total_len]

        if use_cache and past_key_values is not None:
            if seqlen > 1 and cache_seqlen > 0 and num_vision == 0:
                # L-8: 投機的デコードでチャンク（[直前トークン + MTPドラフト群]）をまとめて
                # 検証する際、チャンク内の各クエリが後続（未来）ドラフトへ注意しないよう因果
                # マスクを与える。これが無いと位置 i のクエリが未来ドラフトを覗き、単一トークン
                # 逐次生成と結果が食い違う（投機 ON/OFF で出力が不一致になる実バグ）。
                # 単一トークン生成や prefill（空キャッシュ）は従来挙動を維持するため対象外。
                # query i（絶対位置 cache_seqlen+i）は key 0..(cache_seqlen+i) のみ参照可。
                is_causal = False
                total_keys = cache_seqlen + seqlen
                q_pos = torch.arange(seqlen, device=tokens.device).unsqueeze(1) + cache_seqlen
                k_pos = torch.arange(total_keys, device=tokens.device).unsqueeze(0)
                attn_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)   # [1,1,seqlen,total_keys] bool
            else:
                is_causal = False
                attn_mask = None
        else:
            if num_vision > 0:
                is_causal = False
                attn_mask = torch.ones((total_seqlen, total_seqlen), device=tokens.device, dtype=torch.bool)
                attn_mask = torch.tril(attn_mask)
                attn_mask[:num_vision, :num_vision] = True
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                is_causal = True
                attn_mask = None
        
        total_aux_loss = 0.0
        new_key_values = []
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, aux_loss, new_cache = layer(h, freqs_cos_slice, freqs_sin_slice, is_causal=is_causal,
                                           use_cache=use_cache, past_key_value=past_kv,
                                           attn_mask=attn_mask)
            total_aux_loss += aux_loss
            if use_cache:
                new_key_values.append(new_cache)
            
        h = self.norm(h)
        logits = self.output(h)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error("NaN/Inf detected in model logits during forward pass! Sanitizing to finite values.")
            # NaN/Inf をそのまま返すと後段のサンプラ(softmax)が全域 NaN 化し空応答/クラッシュに
            # つながるため、有限値へ丸めて下流を保護する（発生自体はログで可視化）。
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            
        mtp_logits = None
        if return_mtp and self.n_mtp_future > 0:
            mtp_logits = [head(h) for head in self.mtp_heads]
            
        prm_logits = None
        if return_prm:
            prm_logits = self.prm_head(h).squeeze(-1)
        
        if use_cache:
            if return_mtp and return_prm:
                return logits, total_aux_loss, new_key_values, mtp_logits, prm_logits
            if return_mtp:
                return logits, total_aux_loss, new_key_values, mtp_logits
            if return_prm:
                return logits, total_aux_loss, new_key_values, prm_logits
            return logits, total_aux_loss, new_key_values
        
        if return_mtp and return_prm:
            return logits, total_aux_loss, mtp_logits, prm_logits
        if return_mtp:
            return logits, total_aux_loss, mtp_logits
        if return_prm:
            return logits, total_aux_loss, prm_logits
        return logits, total_aux_loss
