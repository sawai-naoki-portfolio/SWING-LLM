"""
swi_voice/eval/drift_monitor.py
本番監視: 品質ドリフト検知と継続学習 [音声編9.1 / 12.3, アーキ乖離 問題11]

本番環境の音声品質を 2 軸で時系列に集計し、基準期間に対して直近期間が閾値以上悪化したら
「ドリフト」を検知してアラートする [音声編12.3, 実装編10.2]。

  - 自然さ (MOS): eval/mos_predictor.MOSPredictor による自動推定。低下＝品質劣化。
  - 明瞭度 (WER): 生成音声を再 ASR した単語誤り率（eval/evaluate.transcription_wer）。
    上昇＝明瞭度の劣化。新しいアクセント・スラング・雑音環境の混入で先に動くことが多い。

併せて、低 MOS の生成をネガティブサンプルとして収集し、継続的な選好アライメント（DPO/MOS,
train/dpo_mos）へ流し込む負信号として使う。
"""
import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.eval.evaluate import transcription_wer

logger = setup_logger("voice.drift_monitor")


class _Series:
    """基準値と直近 window 件の移動平均を保持する時系列（MOS / WER で共用）。"""

    def __init__(self, window):
        self.window = window
        self.history = []
        self.baseline = None

    def set_baseline(self, values):
        self.baseline = float(sum(values) / max(1, len(values)))

    def observe(self, value):
        self.history.append(float(value))
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

    def recent_mean(self):
        return sum(self.history) / len(self.history) if self.history else None


class DriftMonitor:
    """MOS（低下）と WER（上昇）の時系列から品質ドリフトを検知する [音声編12.3]。

    Args:
        mos_predictor: `MOSPredictor`。`score()` で使う。
        asr_model: `.transcribe(audio) -> str`。`score_wer()` で使う（無ければ WER 監視なし）。
        window: 直近何件の移動平均を「現在の品質」とみなすか。
        drop_threshold: MOS がこの分だけ基準を下回ったら drift。
        wer_rise_threshold: WER がこの分だけ基準を上回ったら drift。
    """

    def __init__(self, mos_predictor=None, asr_model=None, window=20,
                 drop_threshold=0.3, wer_rise_threshold=0.3):
        self.mos_predictor = mos_predictor
        self.asr_model = asr_model
        self.drop_threshold = drop_threshold          # 基準比でこの分だけ低下したら drift
        self.wer_rise_threshold = wer_rise_threshold  # 基準比でこの分だけ上昇したら drift
        self.window = window
        self._mos = _Series(window)
        self._wer = _Series(window)

    # MOS 系列は従来 `history` / `baseline` 属性で公開していたため、後方互換で残す。
    @property
    def history(self):
        return self._mos.history

    @property
    def baseline(self):
        return self._mos.baseline

    # ---- MOS（自然さ。低いほど悪い） -------------------------------------------------
    def set_baseline(self, scores):
        """基準期間の平均 MOS を設定する。"""
        self._mos.set_baseline(scores)
        logger.info(f"DriftMonitor MOS baseline set: {self._mos.baseline:.3f} (n={len(scores)})")

    def observe(self, mos_score):
        """MOS スコアを 1 件記録する（window を超えたら古いものを落とす）。"""
        self._mos.observe(mos_score)

    @torch.no_grad()
    def score(self, waveform):
        """波形 [B,1,T] の MOS を予測器で推定し、履歴へ記録して返す（バッチ平均）。"""
        if self.mos_predictor is None:
            raise ValueError("DriftMonitor.score requires a mos_predictor.")
        s = float(self.mos_predictor(waveform).mean())
        self.observe(s)
        return s

    def recent_mean(self):
        return self._mos.recent_mean()

    # ---- WER（明瞭度。高いほど悪い） -------------------------------------------------
    def set_wer_baseline(self, wers):
        """基準期間の平均 WER を設定する。"""
        self._wer.set_baseline(wers)
        logger.info(f"DriftMonitor WER baseline set: {self._wer.baseline:.3f} (n={len(wers)})")

    def observe_wer(self, wer):
        """WER を 1 件記録する。"""
        self._wer.observe(wer)

    def recent_wer_mean(self):
        return self._wer.recent_mean()

    def score_wer(self, waveform, expected_text):
        """生成音声を再 ASR して WER を測り、履歴へ記録して返す [実装編10.2]。

        ASR 未注入・書き起こし失敗時は None（記録もしない）。1 件の失敗で監視全体を落とさない。
        """
        if self.asr_model is None:
            raise ValueError("DriftMonitor.score_wer requires an asr_model.")
        wer = transcription_wer(waveform, expected_text, self.asr_model)
        if wer is None:
            return None
        self.observe_wer(wer)
        return wer

    # ---- 判定 -----------------------------------------------------------------------
    def check_drift(self):
        """基準と直近平均 MOS を比較。低下が閾値以上なら (True, delta) を返す（delta>0=劣化量）。"""
        if self._mos.baseline is None or not self._mos.history:
            return False, 0.0
        delta = self._mos.baseline - self._mos.recent_mean()
        drifted = delta >= self.drop_threshold
        if drifted:
            logger.warning(f"Quality drift detected (MOS): baseline={self._mos.baseline:.3f} "
                           f"recent={self._mos.recent_mean():.3f} drop={delta:.3f}")
        return drifted, delta

    def check_wer_drift(self):
        """基準と直近平均 WER を比較。上昇が閾値以上なら (True, delta) を返す（delta>0=劣化量）。"""
        if self._wer.baseline is None or not self._wer.history:
            return False, 0.0
        delta = self._wer.recent_mean() - self._wer.baseline
        drifted = delta >= self.wer_rise_threshold
        if drifted:
            logger.warning(f"Quality drift detected (WER): baseline={self._wer.baseline:.3f} "
                           f"recent={self._wer.recent_mean():.3f} rise={delta:.3f}")
        return drifted, delta

    def check_all(self):
        """MOS・WER 双方を判定する。戻り値 {"mos": (drifted, delta), "wer": (drifted, delta)}。"""
        return {"mos": self.check_drift(), "wer": self.check_wer_drift()}


def collect_negatives_for_continual(scored_items, mos_threshold=2.5):
    """(mos_score, item) の列から、MOS が閾値未満の item を負サンプルとして収集する [音声編9.1]。

    本番ログから拾ったネガティブ（再生停止/再生成/低 MOS）を継続 DPO（train/dpo_mos）へ
    流し込むための負信号。返り値は item のリスト。
    """
    return [item for (score, item) in scored_items if float(score) < mos_threshold]
