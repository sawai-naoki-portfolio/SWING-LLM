"""
swi_voice/eval/voice_eval.py

音声の品質評価ゲート（voice ロードマップの前提整備 / 監査 G-3）。

**なぜ必要か**: これまで voice レーンには eval ステージが 1 つも無かった（`run.ps1` の voice は
tests/codec/tts/flow/check のみ）。`VoiceEvalSuite` は実装済みだが `pipeline_check` からも
`run.ps1` からも呼ばれておらず、**MOS/WER の回帰監視が存在しなかった**。
ロードマップは「投資効果を測れない状態で大規模学習を始めない」を制約に掲げているため、
ボコーダ GAN（step7）や意味コーデック蒸留（step8）の前に、まず物差しを常設する。

**測る指標**:
  1. `codec_snr_db`      : コーデック再構成 SNR [dB]（**ホールドアウト**クリップ）。
                           **0 dB = 「無音を出力するだけ」と同じ**という自然な基準になる
                           （SNR = 10log10(||x||^2 / ||x - x_hat||^2) なので x_hat=0 で 0 dB）。
                           負なら無音より悪い = コーデックが壊れている。
  2. `codebook_usage`    : RVQ コードブックの使用率（量子化器ごとの平均）。
                           **コードブック崩壊の検知**用。崩壊すると全入力が同一コードへ潰れ、
                           「コード列ベースの話者指標が全入力で同一になる」という既知の障害が起きる。
                           voice ロードマップ step8 が「コードブック使用率ロギングで崩壊を監視」と
                           規定している当のもの。
  3. `mos` / `sim`       : `VoiceEvalSuite` による MOS 予測・話者類似度。**現状は参考値**
                           （MOS 予測器は自作 Conv1d、話者類似度は latent 平均のプロキシ。
                           step6 で WavLM / ECAPA へ差し替えたら初めてゲートに使える）。
  4. `wer`               : コーデック往復後の音声を実 ASR（faster-whisper, fix 08 / step3）で
                           再転写し、原文テキストとの誤り率を測る（内容保存性の軸）。
                           faster-whisper 未導入・SWI_ASR=0 の環境では従来どおりスキップ（C-4）。

終了コード: 0。ただし `--min-codec-snr` / `--min-codebook-usage` / `--max-wer` 指定時、
基準を満たさなければ 1（run.ps1 のゲート）。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m swi_voice.eval.voice_eval
    .venv\\Scripts\\python.exe -m swi_voice.eval.voice_eval --min-codec-snr 0.5 --json out.json
"""
import argparse
import math
import os
import sys

import torch

PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROTOTYPE_DIR not in sys.path:
    sys.path.append(PROTOTYPE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.config import config
from swi_voice.utils.constants import VOICE_SAMPLE_RATE
from swi_voice.data.audio_dataset import load_voice_clips, _fit_length
from swi_voice.train.codec_train import CODEC_CKPT

logger = setup_logger("voice.eval")

# 評価に使うホールドアウトのクリップ数（学習が使った件数の「次」から取る）。
VOICE_EVAL_HOLDOUT = 8
# 波形の長さを揃える（コーデックはバッチで処理する）。
VOICE_EVAL_CLIP_SEC = 2.0


def signal_to_noise_db(reference, estimate):
    """SNR [dB] = 10log10(||x||^2 / ||x - x_hat||^2)。

    x_hat = 0（無音を返すだけ）のとき厳密に 0 dB になるので、**0 が自明ベースライン**。
    負の値は「無音を返すより悪い」ことを意味する。
    """
    length = min(reference.shape[-1], estimate.shape[-1])
    ref = reference[..., :length]
    est = estimate[..., :length]
    signal = torch.sum(ref ** 2).item()
    noise = torch.sum((ref - est) ** 2).item()
    if noise <= 1e-12:
        return 99.0
    if signal <= 1e-12:
        return 0.0
    return 10.0 * math.log10(signal / noise)


def codebook_usage(codes, codebook_size):
    """RVQ コード [B, T', n_q] の使用率（量子化器ごとの「使われたコード数 / 全コード数」の平均）。

    崩壊すると 1 量子化器あたり 1〜数個しか使われず、使用率が 0 に近づく。
    """
    if codes.numel() == 0:
        return 0.0, []
    per_quantizer = []
    n_q = codes.shape[-1]
    for q in range(n_q):
        used = torch.unique(codes[..., q]).numel()
        per_quantizer.append(used / codebook_size)
    return sum(per_quantizer) / len(per_quantizer), per_quantizer


def _load_codec(device):
    from swi_voice.codec.rvq import NeuralAudioCodec
    ckpt = torch.load(CODEC_CKPT, map_location=device, weights_only=False)
    cfg = ckpt["codec_cfg"]
    codec = NeuralAudioCodec(dim=cfg["dim"], hidden=cfg["hidden"],
                             n_quantizers=cfg["n_quantizers"],
                             codebook_size=cfg["codebook_size"])
    codec.load_state_dict(ckpt["model"])
    codec.eval().to(device)
    return codec, cfg


def _holdout_clips(train_clips):
    """学習に使っていないクリップを取る（各ソースの先頭 train_clips 件をスキップ）。"""
    clips = load_voice_clips(max_clips=VOICE_EVAL_HOLDOUT, sample_rate=VOICE_SAMPLE_RATE,
                             skip_per_source=train_clips)
    target_len = int(VOICE_EVAL_CLIP_SEC * VOICE_SAMPLE_RATE)
    waveforms = torch.stack([_fit_length(c["audio"], target_len) for c in clips])   # [B, 1, T]
    return clips, waveforms


def run_voice_eval(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    codec_cfg = config["codec"]
    train_clips = codec_cfg["max_clips"]

    logger.info("=" * 60)
    logger.info(f"VOICE EVAL (mode={config['MODE']}, device={device})")

    codec, ckpt_cfg = _load_codec(device)
    clips, waveforms = _holdout_clips(train_clips)
    waveforms = waveforms.to(device)

    # 実運用と同じ経路（量子化を通す）で往復させる。forward は損失も返すため encode/decode を使う。
    with torch.no_grad():
        codes = codec.encode(waveforms)                  # [B, T', n_q]
        recon = codec.decode(codes)                      # [B, 1, ~T]

    snr = signal_to_noise_db(waveforms, recon)
    usage, per_quantizer = codebook_usage(codes.cpu(), ckpt_cfg["codebook_size"])

    logger.info(f"  コーデック再構成 SNR : {snr:+.2f} dB  "
                f"(ホールドアウト {len(clips)} クリップ, skip={train_clips})  "
                "[0 dB = 無音を返すのと同じ]")
    logger.info(f"  コードブック使用率   : {usage * 100:.1f}%  "
                f"(量子化器別: {[f'{u * 100:.1f}%' for u in per_quantizer]}, "
                f"codebook={ckpt_cfg['codebook_size']})")
    if snr <= 0:
        logger.warning(f"コーデックの再構成 SNR が {snr:+.2f} dB です（0 dB = 無音出力と同等）。"
                       "コーデックが音声を再構成できていません "
                       "（voice ロードマップ step7/8: ボコーダ GAN・意味コーデック蒸留で解消する想定）。")
    if usage < 0.05:
        logger.warning(f"コードブック使用率が {usage * 100:.1f}% です。コードブック崩壊の疑いがあります"
                       "（全入力が同一コードへ潰れると、コード列ベースの話者指標が無意味になります）。")

    # MOS / 話者類似度 / WER: 現状は参考値（予測器が未学習プロキシのため）。
    # 「生成音声」= コーデック往復後の再構成、「参照」= 原音。こうしないと sim が
    # 同一波形どうしの比較になり、常に 1.000 という無意味な値になる。
    suite_metrics = _run_suite(codec, recon, waveforms, clips, device)

    metrics = {
        "codec_snr_db": snr,
        "codebook_usage": usage,
        "codebook_usage_per_quantizer": per_quantizer,
        "n_holdout": len(clips),
        "mode": config["MODE"],
        **suite_metrics,
    }
    logger.info("=" * 60)
    return metrics


def _run_suite(codec, generated, reference, clips, device):
    """VoiceEvalSuite を走らせる（未配線だったものをここで初めて常設する）。

    generated: コーデック往復後の再構成波形 [B, 1, T]（= 評価対象）
    reference: 原音 [B, 1, T]（= 参照。話者類似度はこの 2 つの間で測る）

    WER は実 ASR（faster-whisper, fix 08）が使える環境でのみ測る。
    MOS/SIM は未学習プロキシのため参考値として出す。
    """
    from swi_voice.asr.whisper_asr import get_default_asr
    from swi_voice.eval.evaluate import VoiceEvalSuite
    from swi_voice.eval.mos_predictor import MOSPredictor

    # 波形は device 上にあるので、予測器も同じ device へ載せる（CPU のままだと device 不一致で落ちる）。
    mos_predictor = MOSPredictor().to(device).eval()
    # eager=True: ここでロードまで確定させる（不可なら None → WER 軸はスキップ。
    # 遅延ロードのままだと 8 サンプルそれぞれで例外→WARNING が繰り返されるため）。
    asr_model = get_default_asr(eager=True)
    suite = VoiceEvalSuite(codec=codec, asr_model=asr_model, mos_predictor=mos_predictor)
    samples = [{"audio": generated[i], "text": clips[i].get("text"),
                "ref_audio": reference[i]} for i in range(len(clips))]
    result = suite.evaluate(samples)

    logger.info(f"  MOS (参考値)         : {result['mos']:.2f}" if result["mos"] is not None
                else "  MOS (参考値)         : 計測不可")
    logger.info(f"  話者類似度 (参考値)  : {result['sim']:.3f}  (原音 vs コーデック再構成)"
                if result["sim"] is not None else "  話者類似度 (参考値)  : 計測不可")
    if result["wer"] is not None:
        logger.info(f"  WER (内容保存性)     : {result['wer']:.3f}  "
                    f"(コーデック再構成を faster-whisper '{asr_model.model_size}' で再転写 vs 原文。"
                    "1.0 = 全て誤り)")
    else:
        logger.info("  WER                  : 計測不可（faster-whisper 未導入 or SWI_ASR=0。fix 08）")
    return {"mos_proxy": result["mos"], "speaker_sim_proxy": result["sim"], "wer": result["wer"],
            "asr_model": asr_model.model_size if asr_model is not None else None}


def main():
    ap = argparse.ArgumentParser(description="音声の品質評価（voice ロードマップの前提整備）")
    ap.add_argument("--json", dest="json_path", default=None, help="指標を JSON で保存")
    ap.add_argument("--min-codec-snr", type=float, default=None,
                    help="コーデック再構成 SNR [dB] がこの値未満なら終了コード 1（0 = 無音出力と同等）")
    ap.add_argument("--min-codebook-usage", type=float, default=None,
                    help="コードブック使用率がこの値未満（=崩壊）なら終了コード 1")
    ap.add_argument("--max-wer", type=float, default=None,
                    help="コーデック往復の WER がこの値を超えたら終了コード 1"
                         "（実 ASR が使えない環境では WER 未計測のため判定しない）")
    args = ap.parse_args()

    metrics = run_voice_eval()

    if args.json_path:
        import json
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved voice eval metrics to {args.json_path}")

    failed = False
    if args.min_codec_snr is not None and metrics["codec_snr_db"] < args.min_codec_snr:
        logger.error(f"コーデック再構成 SNR {metrics['codec_snr_db']:+.2f} dB < 閾値 "
                     f"{args.min_codec_snr} dB（コーデックが音声を再構成できていません）")
        failed = True
    if args.min_codebook_usage is not None and metrics["codebook_usage"] < args.min_codebook_usage:
        logger.error(f"コードブック使用率 {metrics['codebook_usage'] * 100:.1f}% < 閾値 "
                     f"{args.min_codebook_usage * 100:.1f}%（コードブック崩壊）")
        failed = True
    if args.max_wer is not None and metrics.get("wer") is not None and metrics["wer"] > args.max_wer:
        logger.error(f"コーデック往復の WER {metrics['wer']:.3f} > 閾値 {args.max_wer}"
                     "（内容保存性の退行）")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
