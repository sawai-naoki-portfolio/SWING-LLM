"""
swi_image/train/vae_train.py

VAE（潜在空間）の学習エントリ [画像編2]。画像で再構成を学習し、後段 DiT が使う潜在空間を
用意する。

NEW-I4 [画像編2.2]: recon(+KL) に加えて**知覚的損失**（凍結 CNN 特徴距離）と
**敵対的損失**（PatchGAN + hinge）を config フラグで併用できる（高周波ディテールの担保）。
TEST/SMOKE は既定 OFF、PROD/MAX は既定 ON（swi_image/utils/config.py 参照）。

実行: python -m swi_image.train.vae_train
"""
import sys

import torch
from torch.utils.data import DataLoader

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.metrics import log_epoch_metric
from swi_image.utils.config import config
from swi_image.data.image_dataset import load_image_captions, VAEImageDataset
from swi_image.model import ConvVAE

logger = setup_logger("image.vae_train")

VAE_CKPT = f"image_vae_checkpoint{config['ckpt_suffix']}.pt"


def _setup_perceptual(cfg, device):
    """知覚的損失（NEW-I4）。重み取得不可などの失敗は WARNING を出して無効で続行（C-4）。"""
    if not cfg.get("use_perceptual"):
        return None
    try:
        from swi_image.model.perceptual import PerceptualLoss
        return PerceptualLoss(backbone=cfg.get("perceptual_backbone", "squeezenet1_1")).to(device)
    except Exception as e:
        logger.warning(f"PerceptualLoss unavailable ({e}); training without it (NEW-I4).")
        return None


def _setup_discriminator(cfg, device):
    """敵対的損失（NEW-I4）。判別器と optimizer を返す（無効なら (None, None)）。"""
    if not cfg.get("use_adversarial"):
        return None, None
    from swi_image.model.discriminator import PatchDiscriminator
    disc = PatchDiscriminator(base_ch=cfg.get("disc_base_ch", 64),
                              n_layers=cfg.get("disc_layers", 3)).to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=cfg["lr"], betas=(0.5, 0.9))
    return disc, disc_opt


def main():
    cfg = config["vae"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"VAE training (mode={config['MODE']}, device={device}, cfg={cfg})")

    items = load_image_captions(max_samples=cfg["max_samples"] or 2000, size=cfg["image_size"])
    ds = VAEImageDataset(items)
    if len(ds) == 0:
        logger.error("No images for VAE training.")
        return 1
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=len(ds) > cfg["batch_size"])

    vae = ConvVAE(in_ch=3, latent_ch=cfg["latent_ch"], base_ch=cfg["base_ch"],
                  down_factor=cfg["down_factor"]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg["lr"])

    perceptual = _setup_perceptual(cfg, device)
    disc, disc_opt = _setup_discriminator(cfg, device)
    if perceptual is not None or disc is not None:
        from swi_image.model.discriminator import hinge_d_loss, hinge_g_loss
        logger.info(f"VAE-GAN losses enabled: perceptual={perceptual is not None} "
                    f"adversarial={disc is not None} "
                    f"(w_p={cfg['perceptual_weight']}, w_adv={cfg['adv_weight']}, "
                    f"disc_start={cfg['disc_start_steps']})")
    disc_start = cfg.get("disc_start_steps", 0)
    global_step = 0

    vae.train()
    for epoch in range(cfg["epochs"]):
        total, tot_p, tot_g, tot_d, nb = 0.0, 0.0, 0.0, 0.0, 0
        for img in loader:
            img = img.to(device)
            opt.zero_grad()
            recon, loss, parts = vae(img)
            if perceptual is not None:
                p_loss = perceptual(recon, img)
                loss = loss + cfg["perceptual_weight"] * p_loss
                tot_p += p_loss.item()
            adv_active = disc is not None and global_step >= disc_start
            if adv_active:
                # G ステップ: D は凍結して勾配を VAE 側だけへ流す
                for p_ in disc.parameters():
                    p_.requires_grad_(False)
                g_loss = hinge_g_loss(disc(recon))
                loss = loss + cfg["adv_weight"] * g_loss
                tot_g += g_loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()

            if adv_active:
                # D ステップ: 再構成は detach（VAE へ D の勾配を流さない）
                for p_ in disc.parameters():
                    p_.requires_grad_(True)
                disc_opt.zero_grad()
                d_loss = hinge_d_loss(disc(img), disc(recon.detach()))
                d_loss.backward()
                disc_opt.step()
                tot_d += d_loss.item()

            total += parts["recon"].item()
            nb += 1
            global_step += 1
        extras = ""
        if perceptual is not None:
            extras += f" avg_perc={tot_p / max(nb, 1):.4f}"
        if disc is not None:
            extras += f" avg_g={tot_g / max(nb, 1):.4f} avg_d={tot_d / max(nb, 1):.4f}"
        logger.info(f"VAE epoch {epoch + 1}/{cfg['epochs']} avg_recon={total / max(nb, 1):.4f}{extras}")
        log_epoch_metric("image_vae", epoch + 1, total / max(nb, 1),
                         perc_loss=tot_p / max(nb, 1) if perceptual is not None else None,
                         adv_g_loss=tot_g / max(nb, 1) if disc is not None else None,
                         adv_d_loss=tot_d / max(nb, 1) if disc is not None else None)

    # 判別器は推論に不要なため保存しない（PROD の長期 VAE-GAN 学習を resume 可能にする際は
    # disc/disc_opt も ckpt へ含めること）。
    # I-2: scale_factor も保存する（定数を変えても既存 VAE/DiT が黙って壊れないよう ckpt 側を正とする）。
    torch.save({"model": vae.state_dict(), "vae_cfg": {**cfg, "scale_factor": vae.scale_factor}}, VAE_CKPT)
    logger.info(f"Saved VAE checkpoint: {VAE_CKPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
