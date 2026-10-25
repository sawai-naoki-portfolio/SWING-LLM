"""
swi_image/model/text_encoder.py

テキスト条件付けエンコーダ [画像編6]。

実運用では CLIP/T5/VLM の凍結重みを使うが（[画像編6 実装方針]）、from-scratch 方針の
Phase 0 では swi_llm の SentencePiece(32k 日本語) トークナイザを流用し、小型の
双方向 Transformer でキャプションを埋め込む。DiT へは pooled（AdaLN 条件）と
sequence（クロスアテンション）の2系統を渡す。
"""
import torch
import torch.nn as nn

from swi_llm.utils.logger import setup_logger
from swi_image.model.dit import MultiHeadAttention, SwiGLU

logger = setup_logger("image.text_encoder")


class _EncLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim, layers, heads, max_len):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([_EncLayer(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        logger.info(f"Initializing TextEncoder: dim={dim}, layers={layers}, heads={heads}")

    def forward(self, token_ids):
        """token_ids: [B, L] -> {sequence:[B, L, dim], pooled:[B, dim]}。"""
        L = min(token_ids.shape[1], self.max_len)
        token_ids = token_ids[:, :L]
        x = self.tok_emb(token_ids) + self.pos_emb[:, :L]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {"sequence": x, "pooled": x.mean(dim=1)}

    def null_condition(self, batch_size, device):
        """CFG 用の無条件（空テキスト）。長さ1のゼロ埋め込み系列を返す。"""
        seq = torch.zeros(batch_size, 1, self.dim, device=device)
        return {"sequence": seq, "pooled": seq.mean(dim=1)}
