"""
VLM の AnyRes（複数タイル）対応の単体テスト（L-5, core/models/vlm.py）。

検証:
  - 4D 画像 [B,C,H,W]（従来）で forward が有限 logits を返す。
  - 5D 画像 [B,num_tiles,C,H,W]（AnyRes）で各タイルがエンコードされ、vision トークンが
    タイル数分だけ増える（MiniLLM が可変長 vision_features を扱える）。
tokenizer 非依存。小さい VisionEncoder/MiniLLM を構築して検証する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_llm.core.models import MiniLLM
from swi_llm.core.models.vlm import SWIVisionLanguageModel
from swi_llm.core.vision import VisionEncoder, VisionProjector

_TOKENS_PER_TILE = (224 // 16) ** 2 + 1   # パッチ196 + cls = 197


def _vlm():
    torch.manual_seed(0)
    text = MiniLLM(vocab_size=256, dim=64, n_layers=2, n_heads=4, hidden_dim=128,
                   num_experts=4, top_k=2, max_seq_len=1024)
    venc = VisionEncoder(image_size=224, patch_size=16, embed_dim=64, depth=2,
                         n_heads=4, mlp_hidden_dim=128)
    vproj = VisionProjector(vision_dim=64, llm_dim=64)
    return SWIVisionLanguageModel(text, venc, vproj).eval()


class TestVLMAnyRes(unittest.TestCase):
    def test_single_image_4d(self):
        vlm = _vlm()
        ids = torch.randint(0, 256, (1, 5))
        img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits, _ = vlm(ids, images=img)
        self.assertEqual(logits.shape[1], _TOKENS_PER_TILE + 5)
        self.assertTrue(torch.isfinite(logits).all())

    def test_anyres_5d_scales_vision_tokens(self):
        vlm = _vlm()
        ids = torch.randint(0, 256, (1, 5))
        n_tiles = 3
        tiles = torch.randn(1, n_tiles, 3, 224, 224)      # AnyRes: グローバル+タイル
        with torch.no_grad():
            logits, _ = vlm(ids, images=tiles)
        # vision トークンがタイル数分だけ増える
        self.assertEqual(logits.shape[1], n_tiles * _TOKENS_PER_TILE + 5)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
