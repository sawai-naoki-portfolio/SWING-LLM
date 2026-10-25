"""
swi_image/data/phash.py
知覚的ハッシュ（pHash）による画像の近重複排除 [画像編1.1, NEW-I3 / 02_architecture_docs_gap_audit]

設計（02_Image_Generation_Architecture §1）は「知覚的ハッシュ (pHash) を用いてほぼ同一の
画像を除外し、記憶（Memorization）リスクを低減」を要求するが、実装が存在しなかった。

方式（依存ゼロ・標準的な DCT pHash）:
  1. 32x32 グレースケールへ縮小
  2. 2D DCT-II（numpy の行列積で実装。scipy 不要）
  3. 低周波 8x8 ブロックの AC 係数 63 個を中央値で二値化 → 63bit ハッシュ
  リサイズ・再圧縮・軽微な色調変化に頑健で、ハミング距離 ≤ IMG_PHASH_MAX_HAMMING を
  「ほぼ同一」と判定する。

スケール: 照合は受理済みハッシュとの線形走査（O(N) / 枚）。現行のロード規模
（数千枚）では十分速い。数百万枚規模では BK-tree / マルチインデックスへ置換する
（LLM 側 NEW-L3 の LSH 永続化と同じ発想）。
"""
import os

import numpy as np

from swi_llm.utils.logger import setup_logger
from swi_image.utils.constants import IMG_PHASH_MAX_HAMMING, IMG_DEDUP_ENV

logger = setup_logger("image.phash")

_HASH_SIZE = 8          # 低周波ブロックの一辺（63bit ハッシュ）
_DCT_SIZE = 32          # DCT 入力の一辺


def dedup_enabled() -> bool:
    """SWI_IMG_DEDUP=0 で明示的に無効化できる（既定は有効）。"""
    return os.environ.get(IMG_DEDUP_ENV, "1").strip().lower() not in ("0", "false", "no")


def _dct_matrix(n):
    """DCT-II の変換行列 [n, n]（正規直交）。"""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    mat = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    mat[0] /= np.sqrt(2.0)
    return mat.astype(np.float64)


_DCT = _dct_matrix(_DCT_SIZE)


def phash64(pil_image) -> int:
    """PIL 画像の pHash（63bit 整数）を返す。"""
    from PIL import Image
    g = pil_image.convert("L").resize((_DCT_SIZE, _DCT_SIZE), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.float64)
    coeffs = _DCT @ a @ _DCT.T                     # 2D DCT-II
    block = coeffs[:_HASH_SIZE, :_HASH_SIZE].reshape(-1)
    ac = block[1:]                                  # DC 成分（全体輝度）は除外
    median = np.median(ac)
    bits = ac > median
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    """2 つのハッシュ間のハミング距離。"""
    return int(bin(a ^ b).count("1"))


class PHashDeduper:
    """受理済み画像の pHash を保持し、近重複を検出する（ロード 1 回分の状態）。"""

    def __init__(self, max_hamming=IMG_PHASH_MAX_HAMMING):
        self.max_hamming = max_hamming
        self._hashes = []
        self.stats = {"checked": 0, "dropped": 0}

    def is_duplicate(self, pil_image=None, precomputed_hash=None) -> bool:
        """近重複なら True。新規なら登録して False。

        precomputed_hash を渡すと再計算しない（呼び出し側で計算済みのハッシュを共有するため）。
        """
        self.stats["checked"] += 1
        h = phash64(pil_image) if precomputed_hash is None else precomputed_hash
        for seen in self._hashes:
            if hamming(h, seen) <= self.max_hamming:
                self.stats["dropped"] += 1
                return True
        self._hashes.append(h)
        return False

    def log_stats(self, context=""):
        s = self.stats
        if s["dropped"]:
            logger.info(f"pHash dedup{f' ({context})' if context else ''}: "
                        f"dropped {s['dropped']}/{s['checked']} near-duplicates "
                        f"(max_hamming={self.max_hamming}, NEW-I3).")
