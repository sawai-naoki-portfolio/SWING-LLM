"""
AR コーデックLM (swi_voice/model/ar_codec_lm.py) の単体テスト [音声編3.1]。
torch が必要なため未導入環境ではスキップ。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not available")
class TestARCodecLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swi_voice.model import ARCodecLM
        cls.torch = torch
        cls.model = ARCodecLM(text_vocab_size=100, n_codebooks=2, codebook_size=16,
                              dim=32, n_layers=1, n_heads=2, num_experts=4, top_k=2,
                              max_seq_len=128)
        cls.text_ids = torch.randint(0, 100, (2, 5))
        cls.codes = torch.randint(0, 16, (2, 6, 2))

    def test_forward_loss_and_backward(self):
        logits, loss = self.model(self.text_ids, self.codes)
        self.assertEqual(len(logits), 2)                          # コードブック数
        # 停止制御: 予測長は T+1（末尾に EOS）、出力語彙は codebook_size+1（EOS クラス込み）
        self.assertEqual(logits[0].shape, (2, 6 + 1, 16 + 1))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_forward_variable_length_masks_padding(self):
        # code_lens 指定（右パディング）で損失が有限＆逆伝播可能（パディング区間は損失外）
        code_lens = torch.tensor([6, 3])                          # 2 サンプル目は 3 フレームで終端
        logits, loss = self.model(self.text_ids, self.codes, code_lens=code_lens)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_generate_shape_and_valid_indices(self):
        codes = self.model.generate(self.text_ids, max_frames=8)
        self.assertEqual(codes.shape[0], 2)
        self.assertEqual(codes.shape[2], 2)
        self.assertLessEqual(codes.shape[1], 8)                   # EOS で早期停止しうる
        # 生成コードは実コード範囲（EOS クラスは出力に含めない）
        self.assertTrue((codes >= 0).all() and (codes < 16).all())

    def test_eos_stops_generation(self):
        # EOS を強制（codebook0 のヘッドバイアスを EOS クラスへ）すると即停止して空になる
        from swi_voice.model import ARCodecLM
        m = ARCodecLM(text_vocab_size=100, n_codebooks=2, codebook_size=16,
                      dim=32, n_layers=1, n_heads=2, num_experts=4, top_k=2, max_seq_len=128)
        with torch.no_grad():
            m.heads[0].bias[m.eos_class] = 1e4                    # codebook0 が常に EOS
        out = m.generate(self.text_ids, max_frames=20, temperature=0.0)
        self.assertEqual(out.shape[1], 0)                         # 即 EOS → 0 フレーム
        # 逆に EOS を抑制すると max_frames まで生成する
        with torch.no_grad():
            m.heads[0].bias[m.eos_class] = -1e4
        out2 = m.generate(self.text_ids, max_frames=12, temperature=0.0)
        self.assertEqual(out2.shape[1], 12)

    def test_cloning_forward_masks_prompt(self):
        # prompt_frames 指定で損失が有限＆逆伝播可能（音響プロンプト区間は損失外）
        logits, loss = self.model(self.text_ids, self.codes, prompt_frames=3)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_cloning_generate_excludes_prompt_echo(self):
        # 参照音声プロンプトを与えた生成は「新規フレームのみ」を返す（プロンプトのエコー無し）
        prompt = torch.randint(0, 16, (2, 4, 2))
        out = self.model.generate(self.text_ids, prompt_codes=prompt, max_frames=8, temperature=0.0)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[2], 2)
        self.assertLessEqual(out.shape[1], 8)
        self.assertTrue((out >= 0).all() and (out < 16).all())

    def test_kv_cache_matches_uncached_greedy(self):
        # KV(INT8量子化)キャッシュ経路と非キャッシュ(全精度)経路の貪欲生成の一致を確認する。
        # StaticMLACache は KV を INT8 量子化するため厳密一致(torch.equal)は保証されず、
        # 入力/重み依存で稀に貪欲argmaxが反転する（旧テストが順序依存でフレーキーだった原因）。
        # ここでは重みと入力を固定シードで決定化し、大半のフレームが一致することを検証する。
        from swi_voice.model import ARCodecLM
        torch.manual_seed(0)
        m = ARCodecLM(text_vocab_size=100, n_codebooks=2, codebook_size=16, dim=32,
                      n_layers=1, n_heads=2, num_experts=4, top_k=2, max_seq_len=128)
        m.eval()
        text = torch.randint(0, 100, (2, 5))
        prompt = torch.randint(0, 16, (2, 3, 2))
        g_cached = m.generate(text, prompt_codes=prompt, max_frames=10, temperature=0.0, use_cache=True)
        g_uncached = m.generate(text, prompt_codes=prompt, max_frames=10, temperature=0.0, use_cache=False)
        # 生成長はほぼ一致（早期EOSタイミングの反転で最大1フレーム差を許容）
        self.assertLessEqual(abs(g_cached.shape[1] - g_uncached.shape[1]), 1)
        n = min(g_cached.shape[1], g_uncached.shape[1])
        self.assertGreater(n, 0)
        agree = (g_cached[:, :n] == g_uncached[:, :n]).float().mean().item()
        self.assertGreaterEqual(agree, 0.8, f"cached/uncached agreement too low: {agree:.2f}")

    def test_text_mask_blocks_padding_leak(self):
        # 04#1: text_mask を渡すと、右パディングトークンの「値」を変えてもフレーム予測が
        # 不変（フレームがパディングに注意しない＝漏れが塞がれた）。
        m = self.model
        m.eval()
        real = torch.randint(0, 100, (2, 4))
        codes = torch.randint(0, 16, (2, 5, 2))
        K = 3
        mask = torch.cat([torch.ones(2, 4, dtype=torch.bool),
                          torch.zeros(2, K, dtype=torch.bool)], dim=1)
        padA = torch.randint(0, 100, (2, K))
        padB = torch.randint(0, 100, (2, K))
        with torch.no_grad():
            logitsA, _ = m(torch.cat([real, padA], dim=1), codes, text_mask=mask)
            logitsB, _ = m(torch.cat([real, padB], dim=1), codes, text_mask=mask)
        for i in range(len(logitsA)):
            self.assertTrue(torch.allclose(logitsA[i], logitsB[i], atol=1e-5))

    def test_without_mask_padding_leaks(self):
        # 対照: マスク無し（従来）だとパディング値がフレーム予測に影響する＝マスクの必要性を実証。
        m = self.model
        m.eval()
        real = torch.randint(0, 100, (2, 4))
        codes = torch.randint(0, 16, (2, 5, 2))
        K = 3
        padA = torch.zeros(2, K, dtype=torch.long)
        padB = torch.full((2, K), 7, dtype=torch.long)
        with torch.no_grad():
            lA, _ = m(torch.cat([real, padA], dim=1), codes)   # text_mask=None（従来）
            lB, _ = m(torch.cat([real, padB], dim=1), codes)
        leaks = any(not torch.allclose(lA[i], lB[i], atol=1e-5) for i in range(len(lA)))
        self.assertTrue(leaks)

    def test_sampling_produces_valid_indices(self):
        out = self.model.generate(self.text_ids, max_frames=8, temperature=0.8, top_k=8)
        self.assertTrue((out >= 0).all() and (out < 16).all())

    def test_generate_stream_concatenates_to_generate(self):
        # ストリーミング(チャンク)の連結が一括生成と一致すること（貪欲で決定論的）
        text = self.text_ids[:1]                                  # B=1
        full = self.model.generate(text, max_frames=12, temperature=0.0)
        chunks = list(self.model.generate_stream(text, chunk_frames=4, max_frames=12, temperature=0.0))
        for c in chunks:
            self.assertLessEqual(c.shape[1], 4)                   # 各チャンクは chunk_frames 以下
            self.assertEqual(c.shape[2], 2)
        streamed = torch.cat(chunks, dim=1) if chunks else torch.zeros_like(full)
        self.assertTrue(torch.equal(streamed, full))             # 連結 == 一括

    def test_overfits_single_batch(self):
        # 1 バッチを数十ステップで過学習でき、損失が下がることを確認（学習可能性）
        from swi_voice.model import ARCodecLM
        m = ARCodecLM(text_vocab_size=50, n_codebooks=1, codebook_size=8,
                      dim=32, n_layers=1, n_heads=2, num_experts=4, top_k=2, max_seq_len=64)
        text = torch.randint(0, 50, (1, 3))
        codes = torch.randint(0, 8, (1, 4, 1))
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        first = None
        for _ in range(60):
            opt.zero_grad()
            _, loss = m(text, codes)
            loss.backward(); opt.step()
            if first is None:
                first = loss.item()
        self.assertLess(loss.item(), first)                       # 学習で損失が低下


if __name__ == "__main__":
    unittest.main()
