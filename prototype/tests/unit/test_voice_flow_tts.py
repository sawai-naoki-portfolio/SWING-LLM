"""
Flow Matching TTS (swi_voice/model/flow_tts.py) の単体テスト [音声編3.2]。

検証:
  - forward が正しい形状の速度場を返す。
  - speech_infilling_loss がスカラー有限値で backward できる。
  - 単一バッチに overfit させると損失が有意に減少する（Rectified Flow 学習が機能）。
  - sample が正しい形状の有限 latent を返す（無条件生成 / 参照付きクローニング）。
  - 生成 latent を NeuralAudioCodec.decoder に通すと有限波形になる（端から端）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from swi_voice.model.flow_tts import FlowTTS, speech_infilling_loss, sample
from swi_voice.codec import NeuralAudioCodec


def _tiny_model(latent_dim=16, vocab=64):
    torch.manual_seed(0)
    return FlowTTS(text_vocab_size=vocab, latent_dim=latent_dim, dim=32, n_layers=2,
                   n_heads=2, max_frames=64, max_text_len=32)


class TestFlowTTS(unittest.TestCase):
    def test_forward_shape(self):
        m = _tiny_model()
        B, T, D = 2, 12, 16
        xt = torch.randn(B, T, D)
        t = torch.rand(B)
        text = torch.randint(0, 64, (B, 8))
        ctx = torch.zeros(B, T, D)
        v = m(xt, t, text, ctx)
        self.assertEqual(v.shape, (B, T, D))
        self.assertTrue(torch.isfinite(v).all())

    def test_infilling_loss_and_backward(self):
        m = _tiny_model()
        latent = torch.randn(2, 12, 16)
        text = torch.randint(0, 64, (2, 8))
        loss = speech_infilling_loss(m, latent, text, mask_ratio=0.7)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()   # 勾配が流れる（例外なし）
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)

    def test_overfits_single_batch(self):
        # 単一の (latent, text) を記憶できるか＝Rectified Flow 学習が機能しているか。
        torch.manual_seed(1)
        m = _tiny_model()
        latent = torch.randn(1, 16, 16)
        text = torch.randint(0, 64, (1, 6))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)

        def avg_loss(n=8):
            with torch.no_grad():
                return sum(speech_infilling_loss(m, latent, text).item() for _ in range(n)) / n

        init = avg_loss()
        for _ in range(300):
            opt.zero_grad()
            loss = speech_infilling_loss(m, latent, text)
            loss.backward()
            opt.step()
        final = avg_loss()
        # 有意に減少していれば学習している（ノイズ由来の下限があるので 0 にはならない）
        self.assertLess(final, init * 0.6, f"loss did not drop enough: {init:.4f} -> {final:.4f}")

    def test_sample_shape_and_finite(self):
        m = _tiny_model()
        text = torch.randint(0, 64, (2, 8))
        out = sample(m, text, n_frames=20, n_steps=8)
        self.assertEqual(out.shape, (2, 20, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_sample_with_reference_cloning(self):
        m = _tiny_model()
        text = torch.randint(0, 64, (1, 8))
        ref = torch.randn(1, 5, 16)          # 参照音声の latent（可視文脈）
        out = sample(m, text, n_frames=10, ref_latent=ref, n_steps=8)
        self.assertEqual(out.shape, (1, 10, 16))   # 参照区間は含めず生成区間のみ
        self.assertTrue(torch.isfinite(out).all())

    def test_end_to_end_latent_to_waveform(self):
        # FlowTTS の生成 latent を実コーデックのデコーダに通し、有限波形になることを確認。
        codec = NeuralAudioCodec(dim=16, hidden=8, n_quantizers=2, codebook_size=32)
        codec.eval()
        m = _tiny_model(latent_dim=16)
        text = torch.randint(0, 64, (1, 6))
        gen_latent = sample(m, text, n_frames=8, n_steps=6)      # [1, 8, 16]
        with torch.no_grad():
            wav = codec.decoder(gen_latent.transpose(1, 2))       # [1, 1, ~8*hop]
        # 長さは 8 フレーム相当 ±1 フレーム未満（コーデックの転置畳み込み境界オフセット分）。
        self.assertGreaterEqual(wav.shape[-1], 8 * codec.hop)
        self.assertLess(wav.shape[-1], 9 * codec.hop)
        self.assertTrue(torch.isfinite(wav).all())
        self.assertLessEqual(wav.abs().max().item(), 1.0 + 1e-4)  # tanh 出力なので [-1,1]


if __name__ == "__main__":
    unittest.main()
