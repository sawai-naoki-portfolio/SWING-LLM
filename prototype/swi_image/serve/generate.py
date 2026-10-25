"""
swi_image/serve/generate.py

WebUI 用の画像生成推論（プロンプト → 画像 PNG バイト列）[画像編5]。

学習済み VAE + DiT + TextEncoder を（初回のみ）遅延ロードし、テキストから潜在を
Flow Matching サンプリング → VAE で画像へ復元 → PNG にエンコードして返す。
3AI 統合 WebUI から /api/generate_image で呼ばれる。

- スレッドセーフな遅延ロード（web/models.py と同型）。
- VAE と DiT はモードサフィックスが一致する組を探索する（同一モードで対に学習される）。
- モデル未学習でも WebUI 全体は起動できるよう、ロード失敗は呼び出し側で処理する。
"""
import io
import os
import threading

import numpy as np
import torch
from PIL import Image

from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer
from swi_image.utils.config import config
from swi_image.utils.constants import IMG_TEXT_MAX_LEN, IMG_SAMPLE_STEPS, IMG_CFG_SCALE
from swi_image.model import DiT, TextEncoder, sample_euler
from swi_image.model.vae import vae_from_ckpt_cfg

logger = setup_logger("image.generate")

# swi_image/serve/generate.py → swi_image → prototype（チェックポイント保存先）
_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DISPLAY_SIZE = 256          # 小さな生成画像を表示用に拡大する一辺
_gen_instance = None
_lock = threading.Lock()


def _suffix_priority():
    suffix = config.get("ckpt_suffix", "")
    order = [suffix]
    for fb in ["", "_SMOKE", "_TEST", "_MAX"]:
        if fb not in order:
            order.append(fb)
    return order


def _find_matching_ckpts():
    """VAE と DiT のサフィックスが一致する最初の組を返す。"""
    for s in _suffix_priority():
        vae_p = os.path.join(_PROTOTYPE_DIR, f"image_vae_checkpoint{s}.pt")
        dit_p = os.path.join(_PROTOTYPE_DIR, f"image_dit_checkpoint{s}.pt")
        if os.path.exists(vae_p) and os.path.exists(dit_p):
            return vae_p, dit_p
    return None, None


def image_available():
    """画像生成モデルの VAE/DiT チェックポイントが揃っているか。"""
    vae_p, dit_p = _find_matching_ckpts()
    return vae_p is not None and dit_p is not None


class SWIImageGenerator:
    def __init__(self, device):
        self.device = device
        self.tokenizer = get_tokenizer()

        vae_path, dit_path = _find_matching_ckpts()
        if not vae_path or not dit_path:
            raise FileNotFoundError(
                "image checkpoints not found (run swi_image vae_train / dit_train first)")

        # SEC-2: pickle 由来の任意コード実行を防ぐ（Web 到達経路。web/models.py と同方針）。
        # 保存内容は state_dict + プリミティブ設定 dict のみのため weights_only=True で読める。
        vc = torch.load(vae_path, map_location=device, weights_only=True)
        v = vc["vae_cfg"]
        self.vae = vae_from_ckpt_cfg(v).to(device)
        self.vae.load_state_dict(vc["model"])
        self.vae.eval()

        dt = torch.load(dit_path, map_location=device, weights_only=True)   # SEC-2
        dcfg = dt["dit_cfg"]
        self.te = TextEncoder(vocab_size=self.tokenizer.vocab_size, dim=dcfg["text_dim"],
                              layers=dcfg["text_layers"], heads=dcfg["text_heads"],
                              max_len=IMG_TEXT_MAX_LEN).to(device)
        self.te.load_state_dict(dt["text_enc"])
        self.te.eval()
        self.dit = DiT(latent_ch=dt["latent_ch"], latent_size=dt["latent_size"],
                       patch_size=dt["patch_size"], dim=dcfg["dim"], depth=dcfg["depth"],
                       heads=dcfg["heads"], text_dim=dcfg["text_dim"]).to(device)
        self.dit.load_state_dict(dt["dit"])
        self.dit.eval()
        self.latent_ch = dt["latent_ch"]
        self.latent_size = dt["latent_size"]
        logger.info(f"SWIImageGenerator loaded (vae={os.path.basename(vae_path)}, "
                    f"dit={os.path.basename(dit_path)}, device={device}).")

    @torch.no_grad()
    def generate(self, prompt, steps=None, cfg_scale=None):
        """プロンプト → PNG バイト列（表示用に拡大）。"""
        steps = steps or IMG_SAMPLE_STEPS
        cfg_scale = IMG_CFG_SCALE if cfg_scale is None else cfg_scale
        ids = self.tokenizer.encode(str(prompt))[:IMG_TEXT_MAX_LEN]
        if not ids:
            ids = self.tokenizer.encode("画像")[:IMG_TEXT_MAX_LEN]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        cond = self.te(input_ids)
        null = self.te.null_condition(1, self.device)
        lat = sample_euler(self.dit, (1, self.latent_ch, self.latent_size, self.latent_size),
                           cond, uncond=null, n_steps=steps, cfg_scale=cfg_scale, device=self.device)
        img = self.vae.decode(lat)[0].clamp(-1, 1)             # [3, S, S] ∈ [-1,1]
        arr = ((img + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()   # [S, S, 3]
        pil = Image.fromarray(arr, mode="RGB")
        if pil.width < _DISPLAY_SIZE:
            pil = pil.resize((_DISPLAY_SIZE, _DISPLAY_SIZE), Image.NEAREST)  # ドットを保って拡大
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()


def get_generator():
    """SWIImageGenerator を（初回のみ）ロードして返す。スレッドセーフ。"""
    global _gen_instance
    if _gen_instance is not None:
        return _gen_instance
    with _lock:
        if _gen_instance is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _gen_instance = SWIImageGenerator(device)
    return _gen_instance
