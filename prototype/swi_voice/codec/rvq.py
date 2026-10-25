"""
swi_voice/codec/rvq.py

ニューラル・オーディオ・コーデック（残差ベクトル量子化, RVQ）[音声編2.2]。

生波形を「音声のBPEトークン列」に相当する離散コード列へ圧縮し、逆変換で波形へ
復元する。これにより後段の TTS が LLM の技術基盤（Transformer・次トークン予測）を
ほぼそのまま転用できるようになる（土台となる最重要モジュール）。

構成: Conv1d エンコーダ（strided ダウンサンプル）→ ResidualVQ → Conv1d デコーダ。
損失: 多解像度 STFT 再構成損失 + VQ コミットメント損失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_CODEC_STRIDES, VOICE_CODEC_HOP, VOICE_VQ_COMMIT_WEIGHT, VOICE_STFT_SCALES,
)

logger = setup_logger("voice.codec")


class VectorQuantizer(nn.Module):
    """単一コードブックのベクトル量子化。Straight-Through Estimator で
    非微分な argmin をバイパスして勾配を流す [音声編2.2]。"""

    def __init__(self, codebook_size, dim):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x):
        # x: [B, T, dim] -> 各ベクトルを最近傍コードブックエントリへ量子化
        flat = x.reshape(-1, x.shape[-1])                      # [B*T, dim]
        # 距離 ||x - e||^2 = ||x||^2 - 2 x·e + ||e||^2
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        idx = dist.argmin(dim=1).view(x.shape[:-1])            # [B, T]
        quantized = self.codebook(idx)                          # [B, T, dim]
        # STE: 前向きは quantized、逆伝播は x の勾配をそのまま通す
        quantized_st = x + (quantized - x).detach()
        # コミットメント損失（エンコーダ出力をコードへ引き寄せる）
        commit_loss = F.mse_loss(x, quantized.detach())
        codebook_loss = F.mse_loss(quantized, x.detach())
        return quantized_st, idx, commit_loss + codebook_loss


class ResidualVQ(nn.Module):
    """複数段のコードブックで残差を順次量子化する [音声編2.2]。
    段数(n_quantizers)を増やすほど再構成品質は上がるがトークン数も増える。"""

    def __init__(self, n_quantizers, codebook_size, dim):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(codebook_size, dim) for _ in range(n_quantizers)])

    def forward(self, x):
        residual = x
        quantized_out = torch.zeros_like(x)
        all_idx, total_loss = [], 0.0
        for q in self.quantizers:
            q_out, idx, loss = q(residual)
            residual = residual - q_out
            quantized_out = quantized_out + q_out
            all_idx.append(idx)
            total_loss = total_loss + loss
        codes = torch.stack(all_idx, dim=-1)                    # [B, T, n_quantizers]
        return quantized_out, codes, total_loss

    def decode_codes(self, codes):
        """コード [B, T, n_q] を潜在ベクトル [B, T, dim] へ復元する（各段の和）。"""
        out = 0.0
        for i, q in enumerate(self.quantizers):
            out = out + q.codebook(codes[..., i])
        return out


class _ConvBlock(nn.Module):
    """1 段の strided ダウンサンプル畳み込み（残差接続付き）。"""

    def __init__(self, ch, stride):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv1d(ch, ch, kernel_size=1),
        )
        self.act = nn.ELU()
        self.down = nn.Conv1d(ch, ch, kernel_size=2 * stride, stride=stride, padding=stride // 2)

    def forward(self, x):
        x = x + self.res(x)
        return self.down(self.act(x))


class _ConvTransposeBlock(nn.Module):
    """1 段の strided アップサンプル転置畳み込み（残差接続付き）。"""

    def __init__(self, ch, stride):
        super().__init__()
        self.up = nn.ConvTranspose1d(ch, ch, kernel_size=2 * stride, stride=stride, padding=stride // 2)
        self.act = nn.ELU()
        self.res = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv1d(ch, ch, kernel_size=1),
        )

    def forward(self, x):
        x = self.up(self.act(x))
        return x + self.res(x)


class Conv1dEncoder(nn.Module):
    def __init__(self, hidden, dim, strides):
        super().__init__()
        self.inp = nn.Conv1d(1, hidden, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([_ConvBlock(hidden, s) for s in strides])
        self.out = nn.Conv1d(hidden, dim, kernel_size=3, padding=1)

    def forward(self, waveform):
        x = self.inp(waveform)                                  # [B, hidden, T]
        for b in self.blocks:
            x = b(x)
        return self.out(x)                                      # [B, dim, T']


class Conv1dDecoder(nn.Module):
    def __init__(self, hidden, dim, strides):
        super().__init__()
        self.inp = nn.Conv1d(dim, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([_ConvTransposeBlock(hidden, s) for s in reversed(strides)])
        self.out = nn.Conv1d(hidden, 1, kernel_size=7, padding=3)

    def forward(self, z):
        x = self.inp(z)
        for b in self.blocks:
            x = b(x)
        return torch.tanh(self.out(x))                          # [B, 1, ~T]


def multi_scale_stft_loss(recon, target, scales=VOICE_STFT_SCALES):
    """複数解像度の STFT 振幅スペクトルで再構成誤差を測る [音声編2.1]。
    recon/target: [B, 1, T]。長さがずれる場合は短い方に合わせて切り詰める。"""
    r = recon.squeeze(1)
    t = target.squeeze(1)
    n = min(r.shape[-1], t.shape[-1])
    r, t = r[..., :n], t[..., :n]
    loss = 0.0
    for n_fft, hop in scales:
        if n < n_fft:
            continue
        win = torch.hann_window(n_fft, device=r.device)
        R = torch.stft(r, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs()
        T = torch.stft(t, n_fft=n_fft, hop_length=hop, window=win, return_complex=True).abs()
        loss = loss + F.l1_loss(R, T) + F.mse_loss(R, T)
    # どのスケールも成立しない極短信号では時間領域 L1 にフォールバック
    if isinstance(loss, float):
        return F.l1_loss(r, t)
    return loss


class NeuralAudioCodec(nn.Module):
    """EnCodec/DAC 型の全体構成: Conv1d エンコーダ → RVQ → Conv1d デコーダ [音声編2.2]。"""

    def __init__(self, dim, hidden, n_quantizers, codebook_size,
                 strides=VOICE_CODEC_STRIDES, commit_weight=VOICE_VQ_COMMIT_WEIGHT):
        super().__init__()
        self.hop = 1
        for s in strides:
            self.hop *= s
        self.encoder = Conv1dEncoder(hidden, dim, strides)
        self.rvq = ResidualVQ(n_quantizers, codebook_size, dim)
        self.decoder = Conv1dDecoder(hidden, dim, strides)
        self.commit_weight = commit_weight
        self.n_quantizers = n_quantizers
        logger.info(f"Initializing NeuralAudioCodec: dim={dim}, hidden={hidden}, "
                    f"n_quantizers={n_quantizers}, codebook={codebook_size}, hop={self.hop}")

    def _pad_to_hop(self, waveform):
        # 長さを hop の倍数へパディング（エンコード/デコードの整合を安定させる）
        T = waveform.shape[-1]
        pad = (self.hop - T % self.hop) % self.hop
        if pad:
            waveform = F.pad(waveform, (0, pad))
        return waveform

    def encode_latent(self, waveform):
        """波形 [B, 1, T] -> 量子化**前**の連続 latent [B, T', dim]。

        RVQ を通さない連続表現。話者類似度など「コードブックの解像度に依存せず、
        音色の連続的な差を見たい」用途で使う（eval/evaluate.speaker_embedding）。
        """
        waveform = self._pad_to_hop(waveform)
        return self.encoder(waveform).transpose(1, 2)           # [B, T', dim]

    def encode(self, waveform):
        """波形 [B, 1, T] -> コード [B, T', n_quantizers]。"""
        z = self.encode_latent(waveform)                        # [B, T', dim]
        _, codes, _ = self.rvq(z)
        return codes

    def decode(self, codes):
        """コード [B, T', n_quantizers] -> 波形 [B, 1, ~T]。"""
        z = self.rvq.decode_codes(codes).transpose(1, 2)        # [B, dim, T']
        return self.decoder(z)

    def forward(self, waveform):
        waveform = self._pad_to_hop(waveform)
        z = self.encoder(waveform).transpose(1, 2)              # [B, T', dim]
        z_q, codes, vq_loss = self.rvq(z)
        recon = self.decoder(z_q.transpose(1, 2))               # [B, 1, ~T]
        recon_loss = multi_scale_stft_loss(recon, waveform)
        loss = recon_loss + self.commit_weight * vq_loss
        return recon, codes, loss
