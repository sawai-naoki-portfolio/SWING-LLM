"""
swi_voice/model/hybrid_ar_fm.py
自己回帰(AR) + Flow Matching(FM) のハイブリッド TTS [音声編3.3, アーキ乖離 問題2]

2026 年主流トレンドの 2 段構成:
  1. 意味 LM (AR): テキスト -> 低フレームレートの「意味トークン」を自己回帰生成。
     長距離の言語的一貫性を担う（ARCodecLM を単一コードブック=意味語彙で流用）。
  2. 音響 FM (Flow Matching): 意味トークン条件下で高フレームレートの「音響 latent」を
     非自己回帰生成。高音質・高速・安定を担う（FlowTTS を意味トークン列条件で流用）。

生成した音響 latent は SemanticAcousticCodec / NeuralAudioCodec のデコーダで波形へ復元する。
意味/音響トークンは codec/semantic_acoustic.SemanticAcousticCodec から得る（問題1）。
"""
import torch
import torch.nn as nn

from swi_llm.utils.logger import setup_logger
from swi_voice.model.ar_codec_lm import ARCodecLM
from swi_voice.model.flow_tts import FlowTTS, speech_infilling_loss, sample as flow_sample

logger = setup_logger("voice.hybrid_ar_fm")


class HybridARFM(nn.Module):
    """AR(意味) + Flow Matching(音響) のハイブリッド TTS。"""

    def __init__(self, text_vocab_size, semantic_codebook_size, acoustic_latent_dim,
                 ar_dim=256, ar_layers=4, ar_heads=4,
                 fm_dim=256, fm_layers=4, fm_heads=4, fm_max_frames=512,
                 max_text_len=256, max_sem_len=256, num_experts=4, top_k=2):
        super().__init__()
        self.semantic_codebook_size = semantic_codebook_size
        self.max_sem_len = max_sem_len
        # 意味 LM: テキスト -> 意味トークン（単一コードブック = 意味語彙の AR）。
        self.semantic_lm = ARCodecLM(
            text_vocab_size=text_vocab_size, n_codebooks=1,
            codebook_size=semantic_codebook_size,
            dim=ar_dim, n_layers=ar_layers, n_heads=ar_heads,
            num_experts=num_experts, top_k=top_k,
            max_seq_len=max_text_len + max_sem_len + 4)
        # 音響 FM: 意味トークン列を条件に音響 latent を生成（FlowTTS の text 条件を意味トークンに流用）。
        self.acoustic_fm = FlowTTS(
            text_vocab_size=semantic_codebook_size, latent_dim=acoustic_latent_dim,
            dim=fm_dim, n_layers=fm_layers, n_heads=fm_heads,
            max_frames=fm_max_frames, max_text_len=max_sem_len)
        logger.info(f"Initializing HybridARFM: sem_vocab={semantic_codebook_size}, "
                    f"acoustic_dim={acoustic_latent_dim}, max_sem_len={max_sem_len}")

    def _clamp_sem(self, semantic_tokens):
        """意味トークン列を FM の max_text_len 以内に収める（位置埋め込みの範囲保護）。"""
        return semantic_tokens[:, :self.max_sem_len]

    def semantic_loss(self, text_ids, semantic_tokens, code_lens=None, text_mask=None):
        """意味 LM の次トークン予測損失（テキスト -> 意味トークン）。"""
        codes = self._clamp_sem(semantic_tokens).unsqueeze(-1)      # [B, T_sem, 1]
        _, loss = self.semantic_lm(text_ids, codes, code_lens=code_lens, text_mask=text_mask)
        return loss

    def acoustic_loss(self, semantic_tokens, acoustic_latent, mask_ratio=0.7):
        """音響 FM の Rectified Flow 損失（意味トークン条件で音響 latent を復元）。"""
        return speech_infilling_loss(
            self.acoustic_fm, acoustic_latent, self._clamp_sem(semantic_tokens), mask_ratio=mask_ratio)

    @torch.no_grad()
    def generate(self, text_ids, n_frames, max_sem_frames=200,
                 temperature=0.8, top_k=50, n_steps=16):
        """テキスト -> 意味トークン(AR) -> 音響 latent(FM)。

        Returns:
            (acoustic_latent [B, n_frames, acoustic_dim], semantic_tokens [B, T_sem])。
            acoustic_latent はコーデックのデコーダで波形化する。
        """
        device = text_ids.device
        sem = self.semantic_lm.generate(text_ids, max_frames=max_sem_frames,
                                        temperature=temperature, top_k=top_k)   # [B, T_sem, 1]
        if sem.shape[1] == 0:                                       # 空生成の安全策（最低1トークン）
            sem = torch.zeros((text_ids.shape[0], 1, 1), dtype=torch.long, device=device)
        sem_tokens = self._clamp_sem(sem.squeeze(-1))              # [B, T_sem]
        latent = flow_sample(self.acoustic_fm, sem_tokens, n_frames=n_frames,
                             n_steps=n_steps, device=device)        # [B, n_frames, acoustic_dim]
        return latent, sem_tokens
