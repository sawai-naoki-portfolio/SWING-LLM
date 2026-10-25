"""
swi_voice/eval/mos_predictor.py
自動品質推定（MOS 予測器）[音声編6.1, アーキ乖離 問題7]

生成音声の自然さを人手評価（MOS: Mean Opinion Score, 1〜5）の代理として自動推定する。
DPO/RLVR 型の選好アライメント（train/dpo_mos.py）の報酬として用いる。

ここでは依存ゼロの Conv1d スタックで最小実装する。実運用では wav2vec2 等の SSL 表現を
特徴抽出器に用いて相関を高める（Phase 拡張）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger

logger = setup_logger("voice.mos_predictor")


class MOSPredictor(nn.Module):
    """波形 [B,1,T]（または [B,T]）-> MOS スコア [B]（[1,5] にクランプ）を予測する回帰器。"""

    def __init__(self, hidden=64, n_layers=4, kernel_size=15, stride=4):
        super().__init__()
        pad = kernel_size // 2
        chans = [1] + [hidden] * n_layers
        self.convs = nn.ModuleList([
            nn.Conv1d(chans[i], chans[i + 1], kernel_size, stride=stride, padding=pad)
            for i in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        logger.info(f"Initializing MOSPredictor: hidden={hidden}, layers={n_layers}")

    def forward(self, waveform):
        x = waveform if waveform.dim() == 3 else waveform.unsqueeze(1)   # [B,1,T]
        for conv in self.convs:
            x = F.elu(conv(x))
        x = x.mean(dim=-1)                          # グローバル平均プーリング [B, hidden]
        x = self.norm(x)
        raw = self.head(x).squeeze(-1)              # [B]
        return 1.0 + 4.0 * torch.sigmoid(raw)       # MOS を [1,5] に写像


def mos_regression_loss(pred, target):
    """予測 MOS と正解 MOS の MSE 損失。"""
    return F.mse_loss(pred, target.float())
