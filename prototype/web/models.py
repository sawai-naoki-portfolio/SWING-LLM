"""
web/models.py

WebUI 用の VLM モデルのロードと画像前処理。

load_model はスレッドセーフ:
  - グローバル `model` は重みのロードが完全に終わってからのみ公開する
    （ロード途中のモデルでリクエストが生成を始める競合を防止）。
  - チェックポイントは現在のモード (SWI_MODE) のサフィックスを最優先に、
    学習ステージの新しい順 (grpo → dpo → sft → pretrain) で決定的に選択する。
"""
import os
import threading
import torch
import base64
from io import BytesIO
from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer, VOCAB_SIZE
from swi_llm.utils.config import config, INFERENCE_CONFIG, VISION_CONFIG
from swi_llm.utils.constants import (
    CKPT_GRPO, CKPT_DPO, CKPT_SFT, CKPT_PRETRAIN, CKPT_VISION, EXT_PT,
    MAX_IMAGE_BASE64_CHARS, MAX_IMAGE_PIXELS,
)
from swi_llm.core.models import MiniLLM, SWIVisionLanguageModel
from swi_llm.core.vision import VisionEncoder, VisionProjector

logger = setup_logger("web_models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer()
vocab_size = VOCAB_SIZE
model = None
_model_lock = threading.Lock()

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prototype_dir = current_dir

# ロード優先順: 後段の学習ステージほど優先
_CKPT_STAGE_PRIORITY = [CKPT_GRPO, CKPT_DPO, CKPT_SFT, CKPT_PRETRAIN]


def _suffix_priority():
    """現在モードのサフィックスを最優先に、フォールバック順を返す。"""
    suffix = config.get("ckpt_suffix", "")
    suffixes = [suffix]
    for fallback in ["", "_SMOKE", "_TEST", "_MAX"]:
        if fallback not in suffixes:
            suffixes.append(fallback)
    return suffixes


def _find_checkpoints(bases, suffixes):
    """モードサフィックス優先 → ステージ優先の順で、存在するチェックポイントを列挙する。
    先頭が最優先。破損時に次点へフォールバックできるよう「最初の1件」ではなく全件返す。"""
    found = []
    for s in suffixes:
        for base in bases:
            candidate = os.path.join(prototype_dir, f"{base}{s}{EXT_PT}")
            if os.path.exists(candidate) and candidate not in found:
                found.append(candidate)
    return found


def _find_checkpoint(bases, suffixes):
    """後方互換: 最優先の1件を返す（存在しなければ None）。"""
    found = _find_checkpoints(bases, suffixes)
    return found[0] if found else None


def load_model():
    """VLM モデルを（初回のみ）構築・ロードして返す。

    スレッドセーフ: 複数リクエストが同時に呼んでもロードは一度だけ実行され、
    他のスレッドはロード完了まで待機する。
    """
    global model
    if model is not None:
        return model

    with _model_lock:
        if model is not None:
            return model

        logger.info(f"Loading VLM model on {device}...")

        from swi_llm.utils.config import create_mini_llm
        text_model = create_mini_llm(vocab_size=vocab_size, max_seq_len=INFERENCE_CONFIG["max_seq_len"])

        vision_encoder = VisionEncoder(embed_dim=768)
        vision_projector = VisionProjector(vision_dim=768, llm_dim=768)

        vlm = SWIVisionLanguageModel(text_model, vision_encoder, vision_projector).to(device)

        ckpt_paths = _find_checkpoints(_CKPT_STAGE_PRIORITY, _suffix_priority())

        loaded = False
        for ckpt_path in ckpt_paths:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            try:
                # 6GB 級のフルチェックポイント (optimizer state 込み) を VRAM に
                # 直接展開しないよう、必ず CPU にロードしてから重みだけ転送する。
                # weights_only=True: pickle 経由の任意コード実行を防ぐ（ネットワーク公開
                # されるプロセスなので、悪意ある .pt を読んでも安全側に倒す）。
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt['model_state_dict']
                else:
                    state_dict = ckpt

                state_dict = {k: v for k, v in state_dict.items() if k not in ["freqs_cos", "freqs_sin"]}
                vlm.text_model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded checkpoint from {ckpt_path}")
                del ckpt, state_dict
                loaded = True
                break
            except Exception:
                # 破損/不完全な .pt でランダム重みへ黙って劣化しないよう、次点へフォールバック。
                logger.exception(f"Failed to load checkpoint {ckpt_path}; trying next candidate.")
        if not loaded:
            # 有効なチェックポイントが 1 つも読めず、ランダム重みで応答を返す状態。
            # サービスは起動するが出力は無意味＝機能提供が成立しない致命的障害のため CRITICAL。
            logger.critical("No valid checkpoint could be loaded; serving with RANDOM weights "
                            "(model output is meaningless). Check checkpoint files / SWI_MODE.")

        vision_ckpt_path = _find_checkpoint([CKPT_VISION], _suffix_priority())

        if vision_ckpt_path:
            logger.info(f"Loading Vision weights from {vision_ckpt_path}")
            try:
                ckpt = torch.load(vision_ckpt_path, map_location="cpu", weights_only=True)
                if 'vision_encoder' in ckpt:
                    vision_state_dict = ckpt['vision_encoder']
                    if 'patch_embed.proj.weight' in vision_state_dict:
                        weight = vision_state_dict['patch_embed.proj.weight']
                        if weight.ndim == 4:
                            vision_state_dict['patch_embed.proj.weight'] = weight.view(weight.shape[0], -1)
                    vlm.vision_encoder.load_state_dict(vision_state_dict)
                if 'vision_projector' in ckpt:
                    vlm.vision_projector.load_state_dict(ckpt['vision_projector'])
                del ckpt
            except Exception:
                logger.exception(f"Failed to load vision checkpoint {vision_ckpt_path}")

        vlm.eval()
        # 完全にロードし終えてから初めてグローバルに公開する（競合防止）
        model = vlm
        return model


def parse_images(base64_images):
    if not base64_images:
        return None
    try:
        from PIL import Image
        import numpy as np
        # decompression bomb 対策: PIL が閾値超の巨大画像で DecompressionBombError を投げる。
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img_str = base64_images[0]
        if "," in img_str:
            img_data = img_str.split(",")[1]
        else:
            img_data = img_str

        # 復号前に base64 長で上限チェック（巨大入力による OOM/DoS の防止）
        if len(img_data) > MAX_IMAGE_BASE64_CHARS:
            raise ValueError("画像データが大きすぎます。")

        img = Image.open(BytesIO(base64.b64decode(img_data)))
        img.verify()                                   # 破損/爆弾の早期検出
        img = Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')

        mean = torch.tensor(VISION_CONFIG["image_mean"]).view(1, 3, 1, 1)
        std = torch.tensor(VISION_CONFIG["image_std"]).view(1, 3, 1, 1)

        # L-5: AnyRes 有効時は縦横比を保ってタイル分割し、細部を保持する（既定 OFF）。
        if VISION_CONFIG.get("use_anyres"):
            from swi_llm.core.vision import process_image_anyres
            max_side = VISION_CONFIG.get("anyres_max_side", 448)
            w, h = img.size
            scale = min(1.0, max_side / max(w, h))     # 長辺を max_side 以下へ縮小（タイル数の暴発防止）
            if scale < 1.0:
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            arr = np.transpose(np.array(img).astype(np.float32) / 255.0, (2, 0, 1))
            tensor = (torch.tensor(arr).unsqueeze(0) - mean) / std       # [1,C,H,W]（正規化）
            tiles = process_image_anyres(tensor, target_size=tuple(VISION_CONFIG["image_size"]))
            return tiles.to(device)                    # [1, num_tiles, C, 224, 224]

        # 01(小): 前処理は core/vision.preprocess_pil_image を単一情報源として使う
        # （リサイズ方針 = VISION_CONFIG["resize_policy"]。既定 squash は旧実装と同一挙動）。
        from swi_llm.core.vision import preprocess_pil_image
        tensor = preprocess_pil_image(img)
        return tensor.to(device)
    except Exception as e:
        raise ValueError(f"無効な画像データです: {str(e)}")
