"""
swi_image/train/dit_train.py

DiT（Diffusion Transformer）の Flow Matching 学習エントリ [画像編3・4・5]。
学習済み VAE で画像を潜在へ変換し、テキスト条件付きで速度場を学習する。
swi_llm の SentencePiece トークナイザを流用。CFG のため一定確率で条件を落とす。

実行: python -m swi_image.train.dit_train （vae_train の後に実行）
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.metrics import log_epoch_metric
from swi_llm.data.tokenizer import get_tokenizer
from swi_image.utils.config import config
from swi_image.utils.constants import IMG_TEXT_MAX_LEN, IMG_CFG_DROPOUT
from swi_image.data.image_dataset import load_image_captions, DiTDataset
from swi_image.model import DiT, TextEncoder, rectified_flow_loss
from swi_image.model.vae import vae_from_ckpt_cfg
from swi_image.train.vae_train import VAE_CKPT

logger = setup_logger("image.dit_train")

DIT_CKPT = f"image_dit_checkpoint{config['ckpt_suffix']}.pt"


def _collate(batch):
    ids, lats = zip(*batch)
    maxlen = max(t.shape[0] for t in ids)
    ids_pad = torch.stack([F.pad(t, (0, maxlen - t.shape[0])) for t in ids])
    return ids_pad, torch.stack(lats)


def _load_vae(device):
    if not os.path.exists(VAE_CKPT):
        logger.error(f"VAE checkpoint not found: {VAE_CKPT}. Run vae_train first.")
        return None
    ckpt = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    v = ckpt["vae_cfg"]
    vae = vae_from_ckpt_cfg(v).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    return vae, v


def main():
    cfg = config["dit"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"DiT training (mode={config['MODE']}, device={device}, cfg={cfg})")

    loaded = _load_vae(device)
    if loaded is None:
        return 1
    vae, vcfg = loaded

    tokenizer = get_tokenizer()
    size = config["image_size"]
    items = load_image_captions(max_samples=cfg["max_samples"] or 2000, size=size)
    ds = DiTDataset(items, vae, tokenizer, max_text_len=IMG_TEXT_MAX_LEN)
    if len(ds) == 0:
        logger.error("No samples for DiT training.")
        return 1
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=len(ds) > cfg["batch_size"], collate_fn=_collate)

    latent_size = size // vae.down_factor
    text_enc = TextEncoder(vocab_size=tokenizer.vocab_size, dim=cfg["text_dim"],
                           layers=cfg["text_layers"], heads=cfg["text_heads"],
                           max_len=IMG_TEXT_MAX_LEN).to(device)
    dit = DiT(latent_ch=vae.latent_ch, latent_size=latent_size, patch_size=cfg["patch_size"],
              dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"], text_dim=cfg["text_dim"]).to(device)
    opt = torch.optim.Adam(list(dit.parameters()) + list(text_enc.parameters()), lr=cfg["lr"])

    dit.train()
    text_enc.train()
    for epoch in range(cfg["epochs"]):
        total, nb = 0.0, 0
        for ids, lat in loader:
            ids, lat = ids.to(device), lat.to(device)
            cond = text_enc(ids)
            # CFG 用: 一定確率でバッチ全体の条件を無条件に落とす [画像編7.1]
            if torch.rand(1).item() < IMG_CFG_DROPOUT:
                cond = text_enc.null_condition(ids.shape[0], device)
            opt.zero_grad()
            loss = rectified_flow_loss(dit, lat, cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(dit.parameters()) + list(text_enc.parameters()), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        logger.info(f"DiT epoch {epoch + 1}/{cfg['epochs']} avg_loss={total / max(nb, 1):.4f}")
        log_epoch_metric("image_dit", epoch + 1, total / max(nb, 1))

    torch.save({"dit": dit.state_dict(), "text_enc": text_enc.state_dict(), "dit_cfg": cfg,
                "latent_ch": vae.latent_ch, "latent_size": latent_size,
                "patch_size": cfg["patch_size"], "image_size": size,
                "vae_cfg": vcfg}, DIT_CKPT)
    logger.info(f"Saved DiT checkpoint: {DIT_CKPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
