"""
swi_image/pipeline_check.py

画像生成パイプラインの成功基準を自動判定する（swi_llm の pipeline_check 相当）。
- VAE / DiT チェックポイントが生成されているか
- text → サンプリング(潜在) → VAE 復号 → 画像 の end-to-end が有限値・正形で通るか
非ゼロ終了で run_pipeline.ps1 が失敗検知する。
"""
import argparse
import os
import sys

import torch

from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer
from swi_image.utils.config import config
from swi_image.utils.constants import IMG_TEXT_MAX_LEN
from swi_image.train.vae_train import VAE_CKPT
from swi_image.train.dit_train import DIT_CKPT

logger = setup_logger("image.pipeline_check")


def _e2e_smoke():
    from swi_image.model import DiT, TextEncoder, sample_euler
    from swi_image.model.vae import vae_from_ckpt_cfg

    device = "cpu"
    vc = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    v = vc["vae_cfg"]
    vae = vae_from_ckpt_cfg(v)
    vae.load_state_dict(vc["model"])
    vae.eval()

    dt = torch.load(DIT_CKPT, map_location=device, weights_only=False)
    dcfg = dt["dit_cfg"]
    tok = get_tokenizer()
    te = TextEncoder(vocab_size=tok.vocab_size, dim=dcfg["text_dim"], layers=dcfg["text_layers"],
                     heads=dcfg["text_heads"], max_len=IMG_TEXT_MAX_LEN)
    te.load_state_dict(dt["text_enc"])
    te.eval()
    dit = DiT(latent_ch=dt["latent_ch"], latent_size=dt["latent_size"], patch_size=dt["patch_size"],
              dim=dcfg["dim"], depth=dcfg["depth"], heads=dcfg["heads"], text_dim=dcfg["text_dim"])
    dit.load_state_dict(dt["dit"])
    dit.eval()

    ids = torch.tensor([tok.encode("赤い円が中央にある青い背景")[:IMG_TEXT_MAX_LEN]], dtype=torch.long)
    with torch.no_grad():
        cond = te(ids)
        null = te.null_condition(1, torch.device(device))
        lat = sample_euler(dit, (1, dt["latent_ch"], dt["latent_size"], dt["latent_size"]),
                           cond, uncond=null, n_steps=4, cfg_scale=3.0, device=device)
        img = vae.decode(lat)
    return img.shape[-1] == dt["image_size"] and bool(torch.isfinite(img).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=config["MODE"])
    ap.parse_args()

    checks = [
        ("vae checkpoint generated", os.path.exists(VAE_CKPT)),
        ("dit checkpoint generated", os.path.exists(DIT_CKPT)),
    ]
    if all(ok for _, ok in checks):
        try:
            checks.append(("e2e text->latent->image", _e2e_smoke()))
        except Exception:
            logger.exception("E2E smoke failed")
            checks.append(("e2e text->latent->image", False))

    for name, ok in checks:
        logger.info(f"[{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(ok for _, ok in checks)
    n_pass = sum(1 for _, ok in checks if ok)
    logger.info(f"IMAGE PIPELINE CHECK: {'PASS' if passed else 'FAIL'} ({n_pass}/{len(checks)} checks passed)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
