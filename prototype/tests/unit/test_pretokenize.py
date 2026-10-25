"""
data/pretokenize.py（事前トークン化キャッシュ — ステップ5 / fix 03）の単体テスト。

トークナイザは呼び出し回数を数える決定的フェイクを注入し、
「キャッシュヒット時にトークン化が丸ごと省略される」ことを直接検証する。
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from swi_llm.data.pretokenize import get_or_build_token_cache
from swi_llm.data.pretrain import HFPretrainDataset

_EOT = 2


class FakeTokenizer:
    """文字コードをそのままトークン ID にする決定的トークナイザ（呼び出し回数を記録）。"""

    def __init__(self, offset=0):
        self.encode_calls = 0
        self.offset = offset

    def encode(self, text):
        self.encode_calls += 1
        return [min(60000, ord(c) + self.offset) for c in text]


class OverflowTokenizer:
    def encode(self, text):
        return [70000]   # uint16 範囲外


class TestTokenCache(unittest.TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp(prefix="swi_tokcache_")
        self.docs = ["こんにちは", "テスト文書です", "三つ目"]

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _build(self, tokenizer, docs=None, fp="fp-a", tag="t"):
        return get_or_build_token_cache(docs or self.docs, tokenizer, _EOT,
                                        tag=tag, cache_dir=self.cache_dir, fingerprint=fp)

    def test_build_matches_legacy_tokenization(self):
        """キャッシュ経由のトークン列が従来実装（文書ごと encode + EOT）と完全一致する。"""
        tok = FakeTokenizer()
        tokens = self._build(tok)
        expected = []
        for d in self.docs:
            expected.extend(FakeTokenizer().encode(d))
            expected.append(_EOT)
        np.testing.assert_array_equal(np.asarray(tokens, dtype=np.int64), np.array(expected))
        self.assertEqual(tokens.dtype, np.uint16)

    def test_cache_hit_skips_tokenization(self):
        self._build(FakeTokenizer())
        tok2 = FakeTokenizer()
        tokens = self._build(tok2)
        self.assertEqual(tok2.encode_calls, 0, "キャッシュヒットなのに再トークン化した")
        self.assertGreater(len(tokens), 0)

    def test_corpus_change_invalidates(self):
        self._build(FakeTokenizer())
        tok2 = FakeTokenizer()
        self._build(tok2, docs=self.docs + ["追加文書"])
        self.assertGreater(tok2.encode_calls, 0, "コーパス変更後もキャッシュを使ってしまった")

    def test_tokenizer_fingerprint_change_invalidates(self):
        self._build(FakeTokenizer(), fp="fp-a")
        tok2 = FakeTokenizer()
        self._build(tok2, fp="fp-b")
        self.assertGreater(tok2.encode_calls, 0, "トークナイザ刷新後もキャッシュを使ってしまった")

    def test_uint16_overflow_fails_loud(self):
        with self.assertRaises(ValueError):
            self._build(OverflowTokenizer())

    def test_empty_docs_are_skipped(self):
        tokens = self._build(FakeTokenizer(), docs=["", "  ", "実文書"])
        # 実文書ぶん + EOT のみ
        self.assertEqual(len(tokens), len("実文書") + 1)

    def test_memmap_is_readonly(self):
        tokens = self._build(FakeTokenizer())
        with self.assertRaises((ValueError, TypeError)):
            tokens[0] = 1

    def test_dataset_serves_long_tensors_from_uint16_memmap(self):
        """HFPretrainDataset が uint16 memmap から torch.long の (x, y) を返せる
        （torch は uint16 非対応のため int64 へ明示変換している回帰確認）。"""
        docs = ["a" * 50]
        tokens = self._build(FakeTokenizer(), docs=docs)
        ds = HFPretrainDataset(tokens, seq_len=8, curriculum=True)
        x, y = ds[0]
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(len(x), 8)
        # x を 1 つずらしたものが y（次トークン予測の教師）
        self.assertTrue(torch.equal(x[1:], y[:-1]))


if __name__ == "__main__":
    unittest.main()
