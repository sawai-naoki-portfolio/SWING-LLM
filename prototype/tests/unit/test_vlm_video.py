"""
tests/unit/test_vlm_video.py

NEW-L5（02_multimodal 展望2）: 動画理解の最小拡張（VideoFrameEncoder + videos 経路）。

検証:
  - フレーム列 [B,T,C,H,W] が時空間エンコードされ [B,T,D] のフレームトークンになる。
  - VLM の videos= 経路で有限 logits が返り、vision トークン数 = フレーム数（1 フレーム
    1 トークン = コンテキスト線形消費）になる。
  - max_frames 超過は等間隔の間引きで処理される。
tokenizer 非依存。test_vlm_anyres.py と同じ極小構成で検証する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_llm.core.models import MiniLLM
from swi_llm.core.models.vlm import SWIVisionLanguageModel
from swi_llm.core.vision import VisionEncoder, VisionProjector, VideoFrameEncoder


def _vlm(max_frames=8):
    torch.manual_seed(0)
    text = MiniLLM(vocab_size=256, dim=64, n_layers=2, n_heads=4, hidden_dim=128,
                   num_experts=4, top_k=2, max_seq_len=1024)
    venc = VisionEncoder(image_size=224, patch_size=16, embed_dim=64, depth=2,
                         n_heads=4, mlp_hidden_dim=128)
    vproj = VisionProjector(vision_dim=64, llm_dim=64)
    video = VideoFrameEncoder(venc, embed_dim=64, max_frames=max_frames,
                              temporal_depth=1, n_heads=4)
    return SWIVisionLanguageModel(text, venc, vproj, video_encoder=video).eval()


class TestVideoFrameEncoder(unittest.TestCase):
    def test_output_is_one_token_per_frame(self):
        vlm = _vlm()
        frames = torch.randn(2, 4, 3, 224, 224)
        with torch.no_grad():
            out = vlm.video_encoder(frames)
        self.assertEqual(tuple(out.shape), (2, 4, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_too_many_frames_subsampled(self):
        vlm = _vlm(max_frames=4)
        frames = torch.randn(1, 9, 3, 224, 224)
        with torch.no_grad():
            out = vlm.video_encoder(frames)
        self.assertEqual(out.shape[1], 4)                      # 等間隔間引きで上限内へ

    def test_rejects_non_5d(self):
        vlm = _vlm()
        with self.assertRaises(ValueError):
            vlm.video_encoder(torch.randn(1, 3, 224, 224))


class TestVLMVideoPath(unittest.TestCase):
    def test_forward_with_videos(self):
        vlm = _vlm()
        ids = torch.randint(0, 256, (1, 5))
        frames = torch.randn(1, 4, 3, 224, 224)
        with torch.no_grad():
            logits, _ = vlm(ids, videos=frames)
        # 1 フレーム = 1 vision トークン（画像 1 枚の 197 トークンに対し線形消費）
        self.assertEqual(logits.shape[1], 4 + 5)
        self.assertTrue(torch.isfinite(logits).all())

    def test_encode_video_requires_injection(self):
        torch.manual_seed(0)
        text = MiniLLM(vocab_size=256, dim=64, n_layers=2, n_heads=4, hidden_dim=128,
                       num_experts=4, top_k=2, max_seq_len=1024)
        venc = VisionEncoder(image_size=224, patch_size=16, embed_dim=64, depth=2,
                             n_heads=4, mlp_hidden_dim=128)
        vlm = SWIVisionLanguageModel(text, venc, VisionProjector(64, 64))
        with self.assertRaises(ValueError):
            vlm.encode_video(torch.randn(1, 2, 3, 224, 224))

    def test_image_path_unchanged(self):
        vlm = _vlm()
        ids = torch.randint(0, 256, (1, 5))
        img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits, _ = vlm(ids, images=img)
        self.assertEqual(logits.shape[1], (224 // 16) ** 2 + 1 + 5)   # 従来どおり


if __name__ == "__main__":
    unittest.main(verbosity=2)
