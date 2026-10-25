"""
swi_voice/train/dpo_mos.py
MOSスコアを用いたDPO学習 (Direct Preference Optimization with MOS) [音声編6.2, アーキ乖離 問題7]

自動 MOS 推定器（eval/mos_predictor.MOSPredictor）を報酬に、好ましい/好ましくない音声の
ペアから DPO で TTS モデル（ARCodecLM）をファインチューニングし、出力の自然さを高める。

DPO 損失: -log σ( β · [ (logπ(chosen) - logπ(rejected)) - (logπ_ref(chosen) - logπ_ref(rejected)) ] )
参照モデル（凍結 SFT）からの乖離を抑えつつ、chosen を rejected より相対的に高く評価させる。
系列対数尤度は AR モデルの平均 CE 損失の符号反転（-loss）を per-sample proxy として用いる。
"""
import torch
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger

logger = setup_logger("voice.dpo_mos")


def ar_logprob(model, text_ids, codes):
    """AR コーデック LM の系列対数尤度プロキシ（= -平均CE損失）。高いほど尤もらしい。"""
    _, loss = model(text_ids, codes)
    return -loss


def tts_dpo_loss(policy, reference, text_ids, chosen_codes, rejected_codes, beta=0.1):
    """chosen を rejected より相対的に高く評価するよう policy を更新する DPO 損失。

    reference は凍結参照モデル（KL アンカー）。戻り値はスカラー損失。
    """
    pol_chosen = ar_logprob(policy, text_ids, chosen_codes)
    pol_rejected = ar_logprob(policy, text_ids, rejected_codes)
    with torch.no_grad():
        ref_chosen = ar_logprob(reference, text_ids, chosen_codes)
        ref_rejected = ar_logprob(reference, text_ids, rejected_codes)
    logits = beta * ((pol_chosen - pol_rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(logits).mean()


@torch.no_grad()
def score_audio_mos(mos_predictor, codec, codes):
    """コード列 [B,T,n_q] をコーデックでデコードし、MOS 予測器で自然さを評価する [B]。"""
    wav = codec.decode(codes)
    return mos_predictor(wav)


def dpo_with_mos_step(policy, reference, mos_predictor, codec,
                      text_ids, codes_a, codes_b, beta=0.1):
    """2 候補 (codes_a, codes_b) を MOS で評価し、高い方を chosen として DPO 損失を返す。

    MOS 予測器（報酬）で選好ラベルを自動生成する RLVR/DPO 型のアライメント 1 ステップ。
    """
    mos_a = score_audio_mos(mos_predictor, codec, codes_a).mean()
    mos_b = score_audio_mos(mos_predictor, codec, codes_b).mean()
    if mos_a >= mos_b:
        chosen, rejected = codes_a, codes_b
    else:
        chosen, rejected = codes_b, codes_a
    logger.debug(f"DPO/MOS: mos_a={float(mos_a):.3f} mos_b={float(mos_b):.3f} "
                 f"-> chosen={'a' if chosen is codes_a else 'b'}")
    return tts_dpo_loss(policy, reference, text_ids, chosen, rejected, beta=beta)
