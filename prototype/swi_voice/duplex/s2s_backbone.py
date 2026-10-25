"""
swi_voice/duplex/s2s_backbone.py
Moshi型フルデュプレックス (Speech-to-Speech Backbone) [音声編10.2, アーキ乖離 問題9]

テキストを介さず、ユーザー音声とエージェント音声の 2 ストリームを同一タイムステップで
同時にモデル化する（「聞きながら話す」フルデュプレックス）。各フレームで「ユーザーコード」と
「エージェントコード（1 つ前）」の埋め込みを合成して時間 Transformer に通し、次フレームの
エージェントコード（全コードブック）を予測する。半二重（一問一答）の TTS と異なり、
ユーザー発話に重なって相槌・割り込み応答を生成できる。

swi_llm の TransformerBlock（MLA+MoE+QK-Norm）を流用。依存ゼロ・config でスケール可能な
最小-本物実装（overfit テストで学習を実証）。フル版は Temporal+Depth の 2 段 Transformer と
実対話コーパスでの学習（Phase 2 拡張）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
from swi_llm.core.models.transformer_blocks import TransformerBlock, RMSNorm, precompute_freqs_cis

logger = setup_logger("voice.s2s")


class SpeechToSpeechBackbone(nn.Module):
    """デュアルストリーム（user/agent）のフルデュプレックス音声対話モデル。"""

    def __init__(self, n_codebooks, codebook_size, dim=256, n_layers=4, n_heads=4,
                 num_experts=4, top_k=2, max_seq_len=2048):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.bos_id = codebook_size                          # エージェント入力系列の開始トークン
        # 2 ストリーム × コードブックの埋め込み（agent は BOS 用に +1）
        self.user_emb = nn.ModuleList(
            [nn.Embedding(codebook_size, dim) for _ in range(n_codebooks)])
        self.agent_emb = nn.ModuleList(
            [nn.Embedding(codebook_size + 1, dim) for _ in range(n_codebooks)])
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, dim * 4, num_experts, top_k) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.heads = nn.ModuleList(
            [nn.Linear(dim, codebook_size) for _ in range(n_codebooks)])
        self.max_seq_len = max_seq_len
        head_dim = dim // n_heads
        fc, fs = precompute_freqs_cis(head_dim, max_seq_len * 2)
        self.register_buffer("freqs_cos", fc, persistent=False)
        self.register_buffer("freqs_sin", fs, persistent=False)
        logger.info(f"Initializing SpeechToSpeechBackbone: dim={dim}, layers={n_layers}, "
                    f"n_codebooks={n_codebooks}, codebook={codebook_size}")

    def _embed_frame(self, user_codes, agent_codes):
        """user/agent コード [B,T,n_q] を合成した埋め込み [B,T,dim]（両ストリームの和）。"""
        u = sum(self.user_emb[i](user_codes[..., i]) for i in range(self.n_codebooks))
        a = sum(self.agent_emb[i](agent_codes[..., i]) for i in range(self.n_codebooks))
        return u + a

    def _run(self, x):
        L = x.shape[1]
        fc, fs = self.freqs_cos[:L], self.freqs_sin[:L]
        aux = 0.0
        for layer in self.layers:
            x, a, _ = layer(x, fc, fs, is_causal=True)
            aux = aux + a
        return self.norm(x), aux

    def forward(self, user_codes, agent_codes):
        """user_codes/agent_codes [B,T,n_q]。教師強制で次エージェントフレームを予測。

        入力エージェント系列は [BOS, agent[:-1]]（1 つ前まで）、ターゲットは agent。各位置 t で
        「その時刻のユーザーフレーム user[t]（聞く）＋エージェント履歴」から agent[t]（話す）を予測。
        """
        B, T, _ = agent_codes.shape
        device = agent_codes.device
        bos = torch.full((B, 1, self.n_codebooks), self.bos_id, dtype=torch.long, device=device)
        agent_in = torch.cat([bos, agent_codes[:, :-1]], dim=1)      # [B, T, n_q]
        x = self._embed_frame(user_codes, agent_in)
        h, aux = self._run(x)
        logits = [head(h) for head in self.heads]                    # each [B, T, codebook_size]
        loss = 0.0
        for i in range(self.n_codebooks):
            loss = loss + F.cross_entropy(
                logits[i].reshape(-1, self.codebook_size), agent_codes[..., i].reshape(-1))
        loss = loss / self.n_codebooks + 0.01 * aux
        return logits, loss

    @torch.no_grad()
    def generate(self, user_codes, temperature=0.8, top_k=50):
        """ユーザー音声コード [B,T,n_q] に同期して、エージェント音声コード [B,T,n_q] を逐次生成する。"""
        self.eval()
        B, T, _ = user_codes.shape
        device = user_codes.device
        agent_in = torch.full((B, 1, self.n_codebooks), self.bos_id, dtype=torch.long, device=device)
        generated = []
        for t in range(T):
            x = self._embed_frame(user_codes[:, :t + 1], agent_in)   # 長さ t+1 同士
            h, _ = self._run(x)
            last = h[:, -1, :]                                        # [B, dim]
            frame = []
            for i in range(self.n_codebooks):
                lg = self.heads[i](last)                             # [B, codebook_size]
                if temperature and temperature > 0:
                    lg = lg / temperature
                    if top_k and 0 < top_k < lg.shape[-1]:
                        v, _ = torch.topk(lg, top_k)
                        lg = lg.masked_fill(lg < v[:, [-1]], float("-inf"))
                    idx = torch.multinomial(F.softmax(lg, dim=-1), 1).squeeze(-1)
                else:
                    idx = lg.argmax(dim=-1)
                frame.append(idx)
            frame = torch.stack(frame, dim=-1)                       # [B, n_q]
            generated.append(frame)
            agent_in = torch.cat([agent_in, frame.unsqueeze(1)], dim=1)
        return torch.stack(generated, dim=1)                         # [B, T, n_q]
