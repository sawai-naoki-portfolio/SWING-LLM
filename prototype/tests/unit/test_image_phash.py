"""
tests/unit/test_image_phash.py

NEW-I3（画像編1.1）: pHash 知覚的ハッシュによる近重複排除。
「同一・軽微な変化 = 衝突 / 別画像 = 非衝突」という判定特性を合成画像で固定する。
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_image.data.phash import phash64, hamming, PHashDeduper, dedup_enabled


def _photo_like(seed, size=128):
    """自然画像風の低周波パターン（乱数そのままだと pHash の前提が崩れるため滑らかにする）。"""
    rng = np.random.RandomState(seed)
    small = rng.rand(8, 8, 3) * 255
    img = Image.fromarray(small.astype(np.uint8)).resize((size, size), Image.BILINEAR)
    return img


class TestPhash(unittest.TestCase):
    def test_identical_images_have_zero_distance(self):
        img = _photo_like(0)
        self.assertEqual(hamming(phash64(img), phash64(img.copy())), 0)

    def test_resized_image_is_near_duplicate(self):
        img = _photo_like(1, size=256)
        smaller = img.resize((96, 96), Image.BILINEAR)
        self.assertLessEqual(hamming(phash64(img), phash64(smaller)), 6)

    def test_brightness_shift_is_near_duplicate(self):
        img = _photo_like(2)
        arr = np.asarray(img, dtype=np.int16) + 12                 # 一様な明度シフト
        shifted = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        self.assertLessEqual(hamming(phash64(img), phash64(shifted)), 6)

    def test_different_images_are_far(self):
        d = hamming(phash64(_photo_like(10)), phash64(_photo_like(11)))
        self.assertGreater(d, 6)

    def test_hash_is_63_bits(self):
        h = phash64(_photo_like(3))
        self.assertGreaterEqual(h, 0)
        self.assertLess(h, 1 << 63)


class TestDeduper(unittest.TestCase):
    def test_drops_duplicates_keeps_distinct(self):
        d = PHashDeduper(max_hamming=6)
        a, b = _photo_like(20), _photo_like(21)
        self.assertFalse(d.is_duplicate(a))                        # 新規
        self.assertTrue(d.is_duplicate(a.resize((96, 96))))        # 近重複 → 除外
        self.assertFalse(d.is_duplicate(b))                        # 別画像 → 通過
        self.assertEqual(d.stats, {"checked": 3, "dropped": 1})

    def test_exact_only_mode(self):
        d = PHashDeduper(max_hamming=0)
        a = _photo_like(30)
        self.assertFalse(d.is_duplicate(a))
        self.assertTrue(d.is_duplicate(a.copy()))                  # 完全一致は距離 0

    def test_env_disable(self):
        with mock.patch.dict(os.environ, {"SWI_IMG_DEDUP": "0"}):
            self.assertFalse(dedup_enabled())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWI_IMG_DEDUP", None)
            self.assertTrue(dedup_enabled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
