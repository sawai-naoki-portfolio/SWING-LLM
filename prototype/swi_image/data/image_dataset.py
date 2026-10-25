"""
swi_image/data/image_dataset.py

画像-キャプションコーパスの読み込み [画像編1]。

HuggingFace の実 image-caption データセット（COCO captions 等）を streaming で取得し、
[-1,1] 正規化テンソルへ整える。

方針: 全モード（TEST/SMOKE/PROD/MAX）で必ず実データのみを使う。合成/ダミー画像は
使用しない。実データが 1 件も取得できない場合は例外を送出して学習を止める
（誤って合成画像で学習しないための安全策）。

安全性フィルタ（NSFW）[画像編1.3, NEW-I1] は `data/safety_filter.py` で実体化済みで、
このロード経路に常時配線されている（分類器が使えない環境では ERROR で明示して続行。
CSAM ハッシュ照合は組織向けの DI 差し込み口のみ — safety_filter.py 参照）。
"""
import itertools

import numpy as np
import torch
from torch.utils.data import Dataset

from swi_llm.utils.logger import setup_logger
from swi_image.utils.constants import IMG_DATASET_CANDIDATES, IMG_MIN_CAPTION_LEN

logger = setup_logger("image.data")


def _load_from_hf(max_samples, size, skip=0):
    try:
        from datasets import load_dataset
        from PIL import Image  # noqa: F401
    except Exception as e:
        logger.warning(f"Could not load image source (deps unavailable: {e}).")
        return []
    from swi_image.data.safety_filter import get_safety_filter
    from swi_image.data.phash import PHashDeduper, dedup_enabled
    safety = get_safety_filter()          # None = 明示無効化 or 分類器不在（ERROR 済み）
    for name, kwargs, img_field, txt_field, split in IMG_DATASET_CANDIDATES:
        try:
            ds = load_dataset(name, split=split, streaming=True, **kwargs)
            items = []
            # pHash 近重複排除（NEW-I3）。ロード 1 回分の状態なのでソースごとに作り直す。
            deduper = PHashDeduper() if dedup_enabled() else None
            # skip: ストリーム先頭から読み飛ばす件数。評価で「学習に使っていない画像」を
            # 取るためこの値は生ストリーム行で数える（学習は skip=0 で先頭から取るので、
            # 受理率がほぼ 100% なら skip=N が実質のホールドアウト境界になる。安全フィルタの
            # 除外数はログで可視化し、境界のずれが疑えるようにする）。
            for row in itertools.islice(ds, skip, skip + max_samples * 3):
                img = row.get(img_field)
                cap = row.get(txt_field)
                if isinstance(cap, (list, tuple)):
                    cap = cap[0] if cap else ""
                cap = str(cap).strip()
                if img is None or len(cap) < IMG_MIN_CAPTION_LEN:
                    continue
                try:
                    pil = img.convert("RGB")
                except Exception:
                    continue                                  # デコード不能な画像はスキップ
                # 安全フィルタは元解像度で判定する（縮小前の方が分類精度が高い）。
                if safety is not None and not safety.is_safe(pil):
                    continue
                # pHash 近重複排除（NEW-I3）。外部 VLM による再キャプション（旧 NEW-I2）は
                # 完全自作方針（2026-10-25）で撤去したため、キャプションは元データのまま使う。
                if deduper is not None and deduper.is_duplicate(pil):
                    continue
                pil = pil.resize((size, size))
                arr = np.asarray(pil, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
                items.append({"image": t.float(), "caption": cap})
                if len(items) >= max_samples:
                    break
            if items:
                if safety is not None:
                    safety.log_stats(context=name)
                if deduper is not None:
                    deduper.log_stats(context=name)
                logger.info(f"Image corpus loaded from '{name}' ({len(items)} items, streaming).")
                return items
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("gated", "401", "403", "authentication", "authorized", "access to")):
                logger.warning(f"Image dataset '{name}' requires HF auth/terms acceptance "
                               f"(run `huggingface_hub.login` and accept terms on its HF page). Trying next.")
            else:
                logger.warning(f"Image dataset '{name}' unavailable ({e}). Trying next candidate.")
    return []


def load_image_captions(max_samples=64, size=32, skip=0):
    """[{image:[3,S,S], caption}] を返す。

    全モード（TEST/SMOKE/PROD/MAX）で実 image-caption コーパスのみを使う。合成/ダミー
    画像は使わない。実データが 1 件も取得できない場合は RuntimeError を送出して学習を
    止める（ネットワーク不通 / HF 認証切れ / 全ソース失敗などを黙って合成で埋めない）。

    skip: ストリーム先頭から読み飛ばす件数。学習は skip=0（先頭 N 件）を使うため、
          評価側で skip=N とすると学習に使っていないホールドアウトになる。
    """
    items = _load_from_hf(max_samples, size, skip=skip)
    if not items:
        raise RuntimeError(
            "実 image-caption コーパスを 1 件も取得できませんでした。ネットワーク接続 / "
            "HuggingFace 認証 (`huggingface_hub.login`) / データセットの公開状況を確認してください。"
            "（方針により合成・ダミー画像へのフォールバックは無効化されています）")
    return items


class VAEImageDataset(Dataset):
    """VAE 学習用: 画像テンソル [3, S, S] を返す。"""

    def __init__(self, items):
        self.images = [it["image"] for it in items]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class DiTDataset(Dataset):
    """DiT 学習用: (token_ids[L], latent[C,h,w]) を返す。VAE で潜在を事前計算。"""

    def __init__(self, items, vae, tokenizer, max_text_len=32):
        self.samples = []
        vae.eval()
        device = next(vae.parameters()).device
        with torch.no_grad():
            for it in items:
                img = it["image"].unsqueeze(0).to(device)
                latent = vae.encode_latent(img)[0].cpu()         # [C, h, w]
                ids = tokenizer.encode(it["caption"])[:max_text_len]
                if not ids:
                    continue
                self.samples.append((torch.tensor(ids, dtype=torch.long), latent))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
