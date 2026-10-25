"""
swi_image/utils/config.py

画像生成AIのモード別設定。swi_llm/swi_voice と同じ SWI_MODE (TEST/SMOKE/PROD/MAX) を
参照し、段階別の画像解像度・データ量・学習ステップ・モデル規模を一元管理する（S-7）。
"""
import os
import logging

logger = logging.getLogger(__name__)

from swi_image.utils.constants import (
    MODE_TEST, MODE_SMOKE, MODE_PROD, MODE_MAX,
    IMG_VAE_LATENT_CH, IMG_VAE_DOWN_FACTOR, IMG_VAE_BASE_CH,
    IMG_DIT_DIM, IMG_DIT_DEPTH, IMG_DIT_HEADS, IMG_PATCH_SIZE,
    IMG_TEXT_ENC_DIM, IMG_TEXT_ENC_LAYERS, IMG_TEXT_ENC_HEADS,
)

# =====================================================================
# モード検出（環境変数 SWI_MODE）
# =====================================================================
MODE = os.environ.get("SWI_MODE", "").strip().upper()
if MODE not in (MODE_TEST, MODE_SMOKE, MODE_PROD, MODE_MAX):
    if MODE:
        logger.warning(f"Unknown SWI_MODE '{MODE}', falling back to {MODE_TEST}.")
    MODE = MODE_TEST

_small = MODE in (MODE_TEST, MODE_SMOKE)

# =====================================================================
# 解像度・データ・学習量（モード別）
# =====================================================================
if MODE == MODE_TEST:
    image_size = 32
    max_samples = 64
    vae_epochs = 1
    dit_epochs = 1
elif MODE == MODE_SMOKE:
    image_size = 64
    max_samples = 500
    vae_epochs = 1
    dit_epochs = 1
elif MODE == MODE_PROD:
    image_size = 128
    max_samples = 50000
    vae_epochs = 3
    dit_epochs = 3
else:  # MAX
    image_size = 256
    max_samples = None
    vae_epochs = 5
    dit_epochs = 5

def _env_flag(name, default):
    """環境変数で bool フラグを強制上書きする（未設定なら default）。"""
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


VAE_CONFIG = {
    "image_size": image_size,
    "latent_ch": IMG_VAE_LATENT_CH,
    "down_factor": 4 if _small else IMG_VAE_DOWN_FACTOR,   # 小画像では控えめに縮小
    "base_ch": 16 if _small else IMG_VAE_BASE_CH,
    "batch_size": 4 if _small else 16,
    "lr": 1e-4,
    "epochs": vae_epochs,
    "max_samples": max_samples,
    # NEW-I4 [画像編2.2]: 知覚的損失 + 敵対的損失（高周波ディテールの担保）。
    # TEST/SMOKE は既定 OFF（速度・決定性。SWI_VAE_PERCEPTUAL / SWI_VAE_ADVERSARIAL=1 で強制可）。
    # PROD/MAX は既定 ON。※ step3 で凍結公開 VAE を採用した場合、この 2 損失は不要になる。
    "use_perceptual": _env_flag("SWI_VAE_PERCEPTUAL", not _small),
    "use_adversarial": _env_flag("SWI_VAE_ADVERSARIAL", not _small),
    "perceptual_backbone": "squeezenet1_1" if _small else "vgg16",
    "perceptual_weight": 0.1,
    "adv_weight": 0.05,
    "disc_base_ch": 16 if _small else 64,
    "disc_layers": 2 if _small else 3,
    # 再構成がある程度形になるまで敵対的項を寝かせる（G が D に序盤から潰されないように）
    "disc_start_steps": 0 if _small else 500,
}

DIT_CONFIG = {
    "dim": 128 if _small else IMG_DIT_DIM,
    "depth": 2 if _small else IMG_DIT_DEPTH,
    "heads": 4 if _small else IMG_DIT_HEADS,
    "patch_size": IMG_PATCH_SIZE,
    "text_dim": 64 if _small else IMG_TEXT_ENC_DIM,
    "text_layers": 1 if _small else IMG_TEXT_ENC_LAYERS,
    "text_heads": 2 if _small else IMG_TEXT_ENC_HEADS,
    "batch_size": 4 if _small else 16,
    "lr": 1e-4,
    "epochs": dit_epochs,
    "max_samples": max_samples,
}

CKPT_SUFFIX = "" if MODE == MODE_PROD else f"_{MODE}"

config = {
    "MODE": MODE,
    "image_size": image_size,
    "vae": VAE_CONFIG,
    "dit": DIT_CONFIG,
    "ckpt_suffix": CKPT_SUFFIX,
}
