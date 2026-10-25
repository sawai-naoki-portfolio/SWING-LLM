"""
StaticMLACache (swi_llm/core/models/transformer_blocks.py) の容量境界テスト。

回帰防止の対象: MTP 投機デコードで 1 回の forward が複数トークンを書き込むとき、
事前確保した KV キャッシュの容量境界を越えると、以前は
「ゼロ幅スライスへの代入」による不明瞭な RuntimeError でクラッシュしていた
（web serve の長文生成が [ERROR] Error during streaming を出す原因）。

  - 容量ちょうどまでの書き込みは成功し、逆量子化した full cache を正しい形で返す。
  - 容量を越える書き込み（単発の大きな write / 複数回の write の累積）は、
    原因の分かる ValueError で fail-loud する（broadcast RuntimeError ではない）。
"""
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.core.models.transformer_blocks import StaticMLACache


def _cache(capacity, latent=16, bsz=1):
    return StaticMLACache(bsz, capacity, latent, device="cpu", dtype=torch.float32)


class TestStaticMLACacheCapacity(unittest.TestCase):
    def test_incremental_fill_to_capacity(self):
        cache = _cache(capacity=8, latent=16)
        for i in range(8):                       # 1 トークンずつ容量ちょうどまで
            out = cache.update(torch.randn(1, 1, 16))
            self.assertEqual(out.shape, (1, i + 1, 16))   # full cache を返す
        self.assertEqual(cache.seqlen, 8)

    def test_single_write_exceeding_capacity_raises_valueerror(self):
        cache = _cache(capacity=4, latent=16)
        with self.assertRaises(ValueError):
            cache.update(torch.randn(1, 5, 16))   # 一度に容量超

    def test_accumulated_write_overflow_raises_valueerror(self):
        # MTP の再現: 単発 write で境界近くまで進み、次の複数トークン write で溢れる
        cache = _cache(capacity=6, latent=16)
        cache.update(torch.randn(1, 4, 16))       # seqlen=4
        with self.assertRaises(ValueError):
            cache.update(torch.randn(1, 3, 16))   # 4+3=7 > 6 → 溢れ

    def test_overflow_is_valueerror_not_runtimeerror(self):
        # 以前の不具合は RuntimeError（ゼロ幅スライスの broadcast）だった。
        # 明示的な ValueError であること（＝原因が分かる fail-loud）を保証する。
        cache = _cache(capacity=3, latent=8)
        cache.update(torch.randn(1, 3, 8))        # seqlen=3（満杯）
        try:
            cache.update(torch.randn(1, 3, 8))    # ゼロ幅スライスに 3 トークン
            self.fail("expected ValueError on overflow")
        except ValueError as e:
            self.assertIn("overflow", str(e).lower())
        except RuntimeError:
            self.fail("overflow raised cryptic RuntimeError instead of clear ValueError")

    def test_exact_fit_multi_token_write_succeeds(self):
        cache = _cache(capacity=6, latent=16)
        cache.update(torch.randn(1, 4, 16))       # seqlen=4
        out = cache.update(torch.randn(1, 2, 16))  # 4+2=6 ちょうど → OK
        self.assertEqual(cache.seqlen, 6)
        self.assertEqual(out.shape, (1, 6, 16))

    def test_rollback_frees_capacity(self):
        cache = _cache(capacity=4, latent=8)
        cache.update(torch.randn(1, 4, 8))        # 満杯
        cache.rollback(2)                          # 2 トークン戻す
        self.assertEqual(cache.seqlen, 2)
        out = cache.update(torch.randn(1, 2, 8))   # 再び書けること
        self.assertEqual(cache.seqlen, 4)
        self.assertEqual(out.shape, (1, 4, 8))


if __name__ == "__main__":
    unittest.main()
