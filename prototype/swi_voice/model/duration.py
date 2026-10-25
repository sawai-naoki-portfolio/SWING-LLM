"""
swi_voice/model/duration.py
継続時間モデル (Duration Model) [音声編3.4]

テキスト（音素/トークン）埋め込み列から、各トークンの発話フレーム数（継続時間）を
予測する軽量な回帰器。FastSpeech 系の duration predictor の最小実装で、Conv1d 2 段 +
線形射影で per-token の log-duration を出力する（学習時は log(1+duration) を回帰）。

非自己回帰 TTS（Flow Matching 等）でテキスト長→音響フレーム長のアラインメントに使う。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger

logger = setup_logger(__name__)


class DurationPredictor(nn.Module):
    """テキスト埋め込み [B, L, dim] から各トークンの log-duration [B, L] を予測する。"""

    def __init__(self, dim, hidden=256, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(dim, hidden, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=pad)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, 1)
        logger.info(f"Initializing DurationPredictor: dim={dim}, hidden={hidden}, kernel={kernel_size}")

    def forward(self, text_emb, text_mask=None):
        """text_emb: [B, L, dim] -> log_dur: [B, L]（log(1+frames) を回帰する想定）。

        text_mask [B, L]（True=実トークン）を渡すとパディング位置の出力を 0 にする。
        """
        x = text_emb.transpose(1, 2)                     # [B, dim, L]
        x = self.drop(F.relu(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2)))
        x = self.drop(F.relu(self.norm2(self.conv2(x).transpose(1, 2)).transpose(1, 2)))
        log_dur = self.proj(x.transpose(1, 2)).squeeze(-1)   # [B, L]
        if text_mask is not None:
            log_dur = log_dur.masked_fill(~text_mask.to(torch.bool), 0.0)
        return log_dur

    @torch.no_grad()
    def predict_frames(self, text_emb, text_mask=None):
        """推論用: 予測 log-duration を非負のフレーム数 [B, L]（整数）へ変換する。"""
        frames = torch.expm1(self.forward(text_emb, text_mask)).clamp(min=0).round().long()
        return frames


def duration_loss(pred_log_dur, target_frames, text_mask=None):
    """予測 log-duration と正解フレーム数から L2 損失（log 域）を計算する。"""
    target_log = torch.log1p(target_frames.float())
    diff = (pred_log_dur - target_log) ** 2
    if text_mask is not None:
        m = text_mask.to(diff.dtype)
        return (diff * m).sum() / (m.sum() + 1e-8)
    return diff.mean()


class LengthRegulator(nn.Module):
    """予測 duration に従って系列を時間方向へ伸長する（FastSpeech 系の Length Regulator）。

    非自己回帰 TTS で「テキスト/意味トークン長 → 音響フレーム長」のアラインメントを担う。
    各トークン i の表現をその duration[i] 回だけ繰り返して並べる。
    """

    def forward(self, x, durations, max_len=None):
        """x [B, L, dim], durations [B, L]（整数, >=0）-> (expanded [B, T, dim], lengths [B])。

        T は max_len または各サンプルの伸長長の最大値。伸長長が T 未満の分は 0 パディング。
        """
        B, L, dim = x.shape
        durations = durations.clamp(min=0).long()
        expanded = [torch.repeat_interleave(x[b], durations[b], dim=0) for b in range(B)]
        lengths = durations.sum(dim=1)                              # [B]
        target = int(max_len) if max_len is not None else int(max(1, int(lengths.max().item())))
        result = x.new_zeros(B, target, dim)
        for b, e in enumerate(expanded):
            n = min(e.shape[0], target)
            if n > 0:
                result[b, :n] = e[:n]
        return result, lengths


class DurationAdaptor(nn.Module):
    """DurationPredictor + LengthRegulator を束ね、テキスト埋め込みを音響フレーム長へ伸長する。"""

    def __init__(self, dim, hidden=256, kernel_size=3):
        super().__init__()
        self.predictor = DurationPredictor(dim, hidden=hidden, kernel_size=kernel_size)
        self.regulator = LengthRegulator()

    def forward(self, text_emb, text_mask=None, durations=None, max_len=None):
        """text_emb [B,L,dim] を（学習時は正解 durations、推論時は予測）で伸長する。

        Returns: (expanded [B,T,dim], log_dur [B,L], lengths [B])
        """
        log_dur = self.predictor(text_emb, text_mask)
        if durations is None:                                      # 推論: 予測 duration を使う
            durations = torch.expm1(log_dur).clamp(min=0).round().long()
            if text_mask is not None:
                durations = durations.masked_fill(~text_mask.to(torch.bool), 0)
        expanded, lengths = self.regulator(text_emb, durations, max_len=max_len)
        return expanded, log_dur, lengths
