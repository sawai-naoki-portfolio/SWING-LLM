"""
core/vision.py の前処理統合（01(小): preprocess_pil_image / fail-loud 化）の単体テスト。

- squash（既定）が旧実装（PIL resize 直 + CLIP 統計正規化）と数値一致すること
  （= web/models.parse_images の従来挙動が保存されていることの証明）。
- shortside_crop がアスペクト比を保持したセンタークロップになっていること。
- PIL 欠如時に乱数ダミーではなく RuntimeError（fail-loud）になること。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from PIL import Image

from swi_llm.core.vision import preprocess_pil_image, process_image
from swi_llm.utils.config import VISION_CONFIG


def _h_gradient_img(w, h):
    """左→右へ 0..255 の水平グラデーション画像（R チャンネル）。"""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, w).astype(np.uint8)[None, :]
    return Image.fromarray(arr)


def _denorm_r(tensor):
    """正規化済みテンソルの R チャンネルを [0,255] スケールへ戻す。"""
    mean = VISION_CONFIG["image_mean"][0]
    std = VISION_CONFIG["image_std"][0]
    return (tensor[0, 0] * std + mean) * 255.0


class TestPreprocessPilImage(unittest.TestCase):
    def test_squash_matches_legacy_math(self):
        """既定 squash は旧実装（resize 直 + CLIP 正規化）と数値一致する。"""
        img = _h_gradient_img(100, 50)
        out = preprocess_pil_image(img, target_size=(224, 224), policy="squash")

        legacy = img.convert("RGB").resize((224, 224))
        arr = np.transpose(np.array(legacy).astype(np.float32) / 255.0, (2, 0, 1))
        t = torch.tensor(arr).unsqueeze(0)
        mean = torch.tensor(VISION_CONFIG["image_mean"]).view(1, 3, 1, 1)
        std = torch.tensor(VISION_CONFIG["image_std"]).view(1, 3, 1, 1)
        expected = (t - mean) / std

        self.assertEqual(tuple(out.shape), (1, 3, 224, 224))
        self.assertTrue(torch.allclose(out, expected))

    def test_default_policy_comes_from_config(self):
        """policy 未指定時は VISION_CONFIG["resize_policy"]（既定 squash）が使われる。"""
        img = _h_gradient_img(100, 50)
        self.assertTrue(torch.allclose(
            preprocess_pil_image(img, target_size=(224, 224)),
            preprocess_pil_image(img, target_size=(224, 224),
                                 policy=VISION_CONFIG.get("resize_policy", "squash")),
        ))

    def test_shortside_crop_center_crops_without_distortion(self):
        """横長 2:1 画像の shortside_crop は水平中央を切り出す（歪ませない）。"""
        img = _h_gradient_img(448, 224)   # 短辺 224 → scale=1 → 中央 224px を切り出すはず
        out = preprocess_pil_image(img, target_size=(224, 224), policy="shortside_crop")
        self.assertEqual(tuple(out.shape), (1, 3, 224, 224))

        r = _denorm_r(out)
        # 元画像の水平グラデーション（0..255）の中央 1/2 区間 ≈ [64, 191] が残る
        self.assertAlmostEqual(float(r[:, 0].mean()), 448 / 4 / 447 * 255, delta=4.0)
        self.assertAlmostEqual(float(r[:, -1].mean()), (448 * 3 / 4 - 1) / 447 * 255, delta=4.0)

        # squash（全幅を圧縮: 0..255 が残る）とは異なる結果になる
        squash = preprocess_pil_image(img, target_size=(224, 224), policy="squash")
        self.assertFalse(torch.allclose(out, squash))


class TestProcessImage(unittest.TestCase):
    def test_tensor_passthrough(self):
        t = torch.randn(1, 3, 224, 224)
        self.assertIs(process_image(t), t)

    def test_missing_pillow_fails_loud(self):
        """01(小): PIL 欠如時は乱数ダミーを返さず RuntimeError を送出する。"""
        with mock.patch.dict(sys.modules, {"PIL": None}):
            with self.assertRaises(RuntimeError):
                process_image("nonexistent.png")


if __name__ == "__main__":
    unittest.main()
