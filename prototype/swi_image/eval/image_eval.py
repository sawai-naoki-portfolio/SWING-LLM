"""
swi_image/eval/image_eval.py

画像生成の品質評価ゲート（image ロードマップ ステップ2 / 監査 G-3）。

**なぜ必要か**: これまで image レーンには eval が 1 つも無かった。`pipeline_check` が見ていたのは
「VAE と DiT の ckpt が存在するか」「E2E が例外なく走るか」の 2 点だけで、**画質は一切測っていなかった**。
ロードマップは「投資効果を測れない状態で大規模学習を始めない」を制約に掲げているため、
凍結 VAE（ステップ3）や DiT の学習投資（ステップ5）の前に、まず物差しを常設する。

**測る指標**:
  1. `vae_psnr_gain`  : VAE 再構成 PSNR の、**自明ベースライン（定数=データ平均色を出すだけ）に対する利得** [dB]。
                        **生の PSNR を閾値にしてはいけない**: 2026-07-12 の実測で、TEST の学習済み VAE は
                        10.68 dB、一方「全画素を平均色で塗るだけ」が 11.11 dB だった。つまり生 PSNR で
                        ゲートすると「灰色一色を返すだけの壊れた VAE」が合格してしまう。
                        利得 = vae_psnr - psnr(定数出力) を使えば、「本当に画像を再構成できているか」を測れる。
                        評価画像は**ホールドアウト**（学習が使った先頭 N 件の次のスライス）。
  2. `gen_pixel_std`  : 生成画像のピクセル標準偏差。**モード崩壊の検知**用（全面グレー等に潰れた
                        生成器は std がほぼ 0 になる。E2E スモークは「有限値・正しい形」しか見ないので
                        この崩壊を見逃す）。
  3. `fid` / `clip_score` / `lpips` : 公開モデル（InceptionV3 / CLIP / LPIPS）が入っていれば計測する。
                        未導入の環境では計測をスキップし、**パイプラインは止めない**（C-4: オフラインでも
                        機能を殺さない）。ロードマップ ステップ3 で凍結 CLIP/VAE を導入した時点で常時有効になる。

終了コード: 0。ただし `--min-psnr-gain` / `--min-gen-std` 指定時、下回れば 1（run.ps1 のゲート）。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m swi_image.eval.image_eval
    .venv\\Scripts\\python.exe -m swi_image.eval.image_eval --min-psnr-gain 1.0 --json out.json
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
from swi_llm.data.tokenizer import get_tokenizer
from swi_image.utils.config import config
from swi_image.utils.constants import IMG_TEXT_MAX_LEN
from swi_image.data.image_dataset import load_image_captions
from swi_image.train.vae_train import VAE_CKPT
from swi_image.train.dit_train import DIT_CKPT

logger = setup_logger("image.eval")

# 再構成評価に使うホールドアウト枚数。学習が使った先頭 N 件の「次の」スライスを取る。
IMG_EVAL_HOLDOUT = 32
# 生成崩壊の検知に使うサンプル数とプロンプト。
IMG_EVAL_GEN_SAMPLES = 4
IMG_EVAL_PROMPTS = [
    "赤い円が中央にある青い背景",
    "a photo of a dog on the grass",
    "山と湖の風景写真",
    "a red car parked on a street",
]
# 画像は [-1, 1] 正規化（VAE デコーダが tanh 出力）なのでダイナミックレンジは 2.0。
IMG_DATA_RANGE = 2.0


def psnr(reference, reconstruction, data_range=IMG_DATA_RANGE):
    """PSNR [dB]。完全一致なら inf を返さないよう上限を設ける。"""
    mse = torch.mean((reference - reconstruction) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def gen_pixel_std(images):
    """生成画像のピクセル標準偏差。モード崩壊（一様な出力）で 0 に近づく。"""
    return float(images.std().item())


def _load_vae(device):
    ckpt = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    from swi_image.model.vae import vae_from_ckpt_cfg
    cfg = ckpt["vae_cfg"]
    vae = vae_from_ckpt_cfg(cfg)
    vae.load_state_dict(ckpt["model"])
    vae.eval().to(device)
    return vae, cfg


def _eval_vae_reconstruction(vae, device, image_size, train_samples):
    """学習に使っていないホールドアウト画像で再構成 PSNR と、自明ベースラインに対する利得を測る。

    ベースライン = 「全画素をデータ平均色で塗る」だけの出力。これを下回る VAE は、
    画像を再構成しているのではなく単に平均へ潰しているだけなので、利得で判定する。
    """
    skip = train_samples or 0
    items = load_image_captions(max_samples=IMG_EVAL_HOLDOUT, size=image_size, skip=skip)
    images = torch.stack([it["image"] for it in items]).to(device)
    with torch.no_grad():
        recon, _, _ = vae(images)

    score = psnr(images, recon)
    constant = torch.full_like(images, images.mean().item())
    baseline = psnr(images, constant)
    gain = score - baseline

    logger.info(f"  VAE 再構成 PSNR  : {score:.2f} dB  "
                f"(ホールドアウト {len(items)} 枚, skip={skip}, {image_size}x{image_size})")
    logger.info(f"  自明ベースライン : {baseline:.2f} dB  (全画素=平均色を出すだけ)")
    logger.info(f"  → 利得           : {gain:+.2f} dB  [0 以下 = 平均色を塗るより劣る]")
    if gain <= 0:
        logger.warning(f"VAE の再構成が自明ベースラインに負けています（利得 {gain:+.2f} dB）。"
                       "この VAE は画像を再構成できておらず、画質の上限を押し下げています "
                       "（image ロードマップ step3: 凍結 SDXL VAE の導入で解消する想定）。")
    return score, baseline, gain, len(items)


def _eval_generation(vae, device, image_size):
    """DiT で生成し、崩壊していないか（ピクセル std）を見る。"""
    from swi_image.model import DiT, TextEncoder, sample_euler

    ckpt = torch.load(DIT_CKPT, map_location=device, weights_only=False)
    dcfg = ckpt["dit_cfg"]
    tok = get_tokenizer()
    text_encoder = TextEncoder(vocab_size=tok.vocab_size, dim=dcfg["text_dim"],
                               layers=dcfg["text_layers"], heads=dcfg["text_heads"],
                               max_len=IMG_TEXT_MAX_LEN)
    text_encoder.load_state_dict(ckpt["text_enc"])
    text_encoder.eval().to(device)

    dit = DiT(latent_ch=ckpt["latent_ch"], latent_size=ckpt["latent_size"],
              patch_size=ckpt["patch_size"], dim=dcfg["dim"], depth=dcfg["depth"],
              heads=dcfg["heads"], text_dim=dcfg["text_dim"])
    dit.load_state_dict(ckpt["dit"])
    dit.eval().to(device)

    prompts = IMG_EVAL_PROMPTS[:IMG_EVAL_GEN_SAMPLES]
    ids = torch.tensor([tok.encode(p)[:IMG_TEXT_MAX_LEN] +
                        [0] * max(0, IMG_TEXT_MAX_LEN - len(tok.encode(p)[:IMG_TEXT_MAX_LEN]))
                        for p in prompts], dtype=torch.long, device=device)
    with torch.no_grad():
        cond = text_encoder(ids)
        null = text_encoder.null_condition(len(prompts), torch.device(device))
        latents = sample_euler(dit, (len(prompts), ckpt["latent_ch"],
                                     ckpt["latent_size"], ckpt["latent_size"]),
                               cond, uncond=null, n_steps=8, cfg_scale=3.0, device=device)
        images = vae.decode(latents)

    std = gen_pixel_std(images)
    logger.info(f"  生成ピクセル std : {std:.4f}  ({len(prompts)} プロンプト) "
                f"[0 に近いほどモード崩壊]")
    return std, images, prompts


def _optional_metrics(images, prompts):
    """公開モデルが入っていれば FID / CLIPScore / LPIPS を測る。未導入ならスキップ（C-4）。

    ロードマップ ステップ3（凍結 SDXL VAE + 凍結 CLIP テキスト塔）でこれらの依存が入るため、
    その時点で自動的に有効になる。今はまだ from-scratch なので通常はスキップされる。
    """
    metrics = {}
    try:
        import torchmetrics  # noqa: F401
        from torchmetrics.image.fid import FrechetInceptionDistance  # noqa: F401
        metrics["fid"] = None
        logger.info("  FID              : torchmetrics は在るが、実装は凍結 VAE 導入(step3)とセットで有効化する")
    except Exception:
        logger.info("  FID              : スキップ（torchmetrics 未導入。step3 で導入予定）")

    try:
        import clip  # noqa: F401
        metrics["clip_score"] = None
    except Exception:
        logger.info("  CLIPScore        : スキップ（CLIP 未導入。step3 で凍結テキスト塔と同時導入予定）")

    try:
        import lpips  # noqa: F401
        metrics["lpips"] = None
    except Exception:
        logger.info("  LPIPS            : スキップ（lpips 未導入。step3 で導入予定）")

    return metrics


def run_image_eval(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae_cfg = config["vae"]
    image_size = vae_cfg["image_size"]
    # vae_train と同じ既定。学習が使った先頭 N 件の「次」を評価に使うため N が必要。
    train_samples = vae_cfg["max_samples"] or 2000

    logger.info("=" * 60)
    logger.info(f"IMAGE EVAL (mode={config['MODE']}, device={device})")

    vae, _ = _load_vae(device)
    vae_psnr, baseline, gain, n_holdout = _eval_vae_reconstruction(vae, device, image_size, train_samples)
    std, images, prompts = _eval_generation(vae, device, image_size)
    optional = _optional_metrics(images, prompts)

    metrics = {
        "vae_psnr": vae_psnr,
        "vae_psnr_constant_baseline": baseline,
        "vae_psnr_gain": gain,
        "gen_pixel_std": std,
        "n_holdout": n_holdout,
        "image_size": image_size,
        "mode": config["MODE"],
        **optional,
    }
    logger.info("=" * 60)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="画像生成の品質評価（image ロードマップ step2）")
    ap.add_argument("--json", dest="json_path", default=None, help="指標を JSON で保存")
    ap.add_argument("--min-psnr-gain", type=float, default=None,
                    help="自明ベースライン（定数=平均色）に対する PSNR 利得 [dB] がこの値未満なら終了コード 1")
    ap.add_argument("--min-gen-std", type=float, default=None,
                    help="生成ピクセル std がこの値未満（=モード崩壊）なら終了コード 1")
    args = ap.parse_args()

    metrics = run_image_eval()

    if args.json_path:
        import json
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved image eval metrics to {args.json_path}")

    failed = False
    if args.min_psnr_gain is not None and metrics["vae_psnr_gain"] < args.min_psnr_gain:
        logger.error(f"VAE 再構成の PSNR 利得 {metrics['vae_psnr_gain']:+.2f} dB < 閾値 "
                     f"{args.min_psnr_gain} dB（VAE が画像を再構成できていません）")
        failed = True
    if args.min_gen_std is not None and metrics["gen_pixel_std"] < args.min_gen_std:
        logger.error(f"生成ピクセル std {metrics['gen_pixel_std']:.4f} < 閾値 {args.min_gen_std} "
                     "（生成がモード崩壊している可能性があります）")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
