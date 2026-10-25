"""
swi_voice/model/vocoder.py
波形生成ボコーダー (Vocoder) [音声編2.1 / 品質向上]

mel スペクトログラム → 波形を生成する HiFi-GAN / Vocos 系の軽量ニューラルボコーダ。
転置畳み込みによるアップサンプリング + 残差ブロック（MRF-lite）で構成し、多解像度 STFT 損失で
学習する（依存ゼロ・純 torch の最小-本物実装）。コーデック（NeuralAudioCodec）のデコーダとは別に、
mel 特徴からの高音質合成経路を提供する。フル版は判別器付き GAN 学習（Phase 拡張）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
from swi_voice.codec.rvq import multi_scale_stft_loss
from swi_voice.utils.constants import (
    VOICE_SAMPLE_RATE, VOICE_CODEC_STRIDES, VOICE_N_MELS, VOICE_MEL_N_FFT,
    VOICE_VOCODER_HIDDEN, VOICE_CODEC_HOP,
)

logger = setup_logger("voice.vocoder")


def waveform_to_mel(waveform, n_mels=VOICE_N_MELS, n_fft=VOICE_MEL_N_FFT, hop=VOICE_CODEC_HOP,
                    sample_rate=VOICE_SAMPLE_RATE):
    """波形 [B,1,T] または [B,T] -> log-mel スペクトログラム [B, n_mels, T']（hop=コーデックhop）。"""
    import torchaudio
    x = waveform.squeeze(1) if waveform.dim() == 3 else waveform
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=1.0
    ).to(x.device)
    mel = melspec(x)                                            # [B, n_mels, T']
    return torch.log(mel.clamp(min=1e-5))


class _ResBlock(nn.Module):
    """dilated Conv1d 2 本の残差ブロック（HiFi-GAN の MRF を軽量化）。"""

    def __init__(self, ch, kernel_size=3, dilations=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(ch, ch, kernel_size, dilation=d, padding=(kernel_size - 1) * d // 2)
            for d in dilations])

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(F.leaky_relu(x, 0.1))
        return x


class Vocoder(nn.Module):
    """mel [B, n_mels, T] -> 波形 [B, 1, ~T*hop]。転置畳み込みで段階的にアップサンプリング。"""

    def __init__(self, n_mels=VOICE_N_MELS, hidden=VOICE_VOCODER_HIDDEN,
                 upsample_rates=VOICE_CODEC_STRIDES):
        super().__init__()
        self.pre = nn.Conv1d(n_mels, hidden, kernel_size=7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        ch = hidden
        for r in upsample_rates:
            self.ups.append(nn.ConvTranspose1d(ch, ch // 2, kernel_size=2 * r, stride=r, padding=r // 2))
            ch = ch // 2
            self.resblocks.append(_ResBlock(ch))
        self.post = nn.Conv1d(ch, 1, kernel_size=7, padding=3)
        self.hop = 1
        for r in upsample_rates:
            self.hop *= r
        logger.info(f"Initializing Vocoder: n_mels={n_mels}, hidden={hidden}, "
                    f"upsample={tuple(upsample_rates)}, hop={self.hop}")

    def forward(self, mel):
        x = self.pre(mel)
        for up, rb in zip(self.ups, self.resblocks):
            x = up(F.leaky_relu(x, 0.1))
            x = rb(x)
        return torch.tanh(self.post(F.leaky_relu(x, 0.1)))      # [B, 1, ~T*hop]


def vocoder_loss(recon, target):
    """生成波形と目標波形の多解像度 STFT 再構成損失（コーデックと同じ指標）。"""
    return multi_scale_stft_loss(recon, target)
