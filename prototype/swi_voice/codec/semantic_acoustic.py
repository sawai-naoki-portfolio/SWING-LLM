"""
swi_voice/codec/semantic_acoustic.py
意味トークンと音響トークンの分離 (Semantic/Acoustic Split) [音声編2.3, アーキ乖離 問題1]

2026 年主流のハイブリッド構成（AR+Flow Matching）の土台。役割を 2 系統に分離する:

  - 意味トークン (Semantic): 学習済み音響コーデックのエンコーダ特徴を時間方向に
    downsample（VOICE_SEMANTIC_DOWNSAMPLE 倍）した **低フレームレートの単一 VQ**。
    言語的内容・大まかな韻律を粗く捉える（内容中心）。ハイブリッドの AR パートが予測する。
  - 音響トークン (Acoustic): 既存 RVQ の **高フレームレート連続 latent**（声質・微細な抑揚）。
    ハイブリッドの Flow Matching パートが意味トークン条件下で生成する。

注: ここでは ASR 教師蒸留の代わりに「音響エンコーダ特徴の低レート VQ」を意味トークンの
最小実装（プロキシ）として用いる。フル版は HuBERT / w2v-BERT からの蒸留（Phase 拡張）。
音響コーデック（NeuralAudioCodec）は学習済みを渡して凍結し、意味 VQ のみを学習する。
"""
import torch
import torch.nn as nn

from swi_llm.utils.logger import setup_logger
from swi_voice.codec.rvq import VectorQuantizer
from swi_voice.utils.constants import (
    VOICE_SEMANTIC_CODEBOOK_SIZE, VOICE_SEMANTIC_DOWNSAMPLE,
)

logger = setup_logger("voice.semantic_acoustic")


class SemanticAcousticCodec(nn.Module):
    """学習済み音響コーデック上に、低レート意味トークン化を付与する分離コーデック。"""

    def __init__(self, codec, semantic_codebook_size=VOICE_SEMANTIC_CODEBOOK_SIZE,
                 semantic_downsample=VOICE_SEMANTIC_DOWNSAMPLE):
        super().__init__()
        self.codec = codec                      # 音響（高レート, 既存/学習済み）
        self.semantic_downsample = semantic_downsample
        dim = codec.encoder.out.out_channels    # エンコーダ出力次元
        ds = semantic_downsample
        # 時間方向に ds 倍 downsample する strided 畳み込み（低フレームレート化）。
        self.semantic_proj = nn.Conv1d(dim, dim, kernel_size=2 * ds, stride=ds, padding=ds // 2)
        self.semantic_vq = VectorQuantizer(semantic_codebook_size, dim)
        self.semantic_codebook_size = semantic_codebook_size
        logger.info(f"Initializing SemanticAcousticCodec: dim={dim}, "
                    f"semantic_codebook={semantic_codebook_size}, downsample={ds}")

    def _encoder_feats(self, waveform):
        """波形 [B,1,T] -> 音響エンコーダ特徴 [B, dim, T_ac]。"""
        return self.codec.encoder(self.codec._pad_to_hop(waveform))

    def _semantic_feats(self, waveform):
        """低レート意味特徴 [B, T_sem, dim] を返す。"""
        z = self._encoder_feats(waveform)                       # [B, dim, T_ac]
        return self.semantic_proj(z).transpose(1, 2)            # [B, T_sem, dim]

    @torch.no_grad()
    def extract_acoustic_latent(self, waveform):
        """高フレームレートの音響連続 latent [B, T_ac, dim]（FM の生成対象）。"""
        return self._encoder_feats(waveform).transpose(1, 2)

    @torch.no_grad()
    def extract_acoustic_codes(self, waveform):
        """高フレームレートの音響 RVQ コード [B, T_ac, n_q]。"""
        return self.codec.encode(waveform)

    def extract_semantic_tokens(self, waveform):
        """低フレームレートの意味トークン [B, T_sem]（AR が予測する離散系列）。"""
        s = self._semantic_feats(waveform)
        _, idx, _ = self.semantic_vq(s)
        return idx

    def decode_acoustic(self, latent):
        """音響連続 latent [B, T_ac, dim] -> 波形 [B, 1, ~T]。"""
        return self.codec.decoder(latent.transpose(1, 2))

    def forward(self, waveform):
        """学習: 意味 VQ のコミットメント損失を返す（音響コーデックは凍結想定）。

        Returns: (semantic_tokens[B,T_sem], vq_loss)
        """
        s = self._semantic_feats(waveform)                      # [B, T_sem, dim]
        _, idx, vq_loss = self.semantic_vq(s)
        return idx, vq_loss
