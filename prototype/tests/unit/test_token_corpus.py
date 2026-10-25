"""
tests/unit/test_token_corpus.py

Phase 1（roadmap A-1）: トークン予算ベースのストリーミング事前学習コーパスの検証。
ネットワーク（HF）に依存しないよう、ソースは合成ストリームで、トークナイザは
1 文字 = 1 トークン（ord）の偽物で差し替える（ファイル内容から文書を復元して検証できる）。
"""
import os
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data import token_corpus
from swi_llm.data.token_corpus import (
    SourceUnavailable, build_token_corpus, load_token_corpus,
)
from swi_llm.utils.config import parse_token_count

EOT = 1


class CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]


def _body(name, i, length):
    seed = f"{name}{i}"
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = []
    h = sum(map(ord, seed))
    for k in range(length):
        h = (h * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(letters[h % 26] if k % 7 else " ")
    return "".join(out)


class FakeSource:
    """決定的な合成ストリーム。fail_at で途中のネットワーク断を再現できる。"""

    def __init__(self, name, n_docs=10 ** 9, doc_len=200, fail_at=None, unavailable=False):
        self.name, self.n_docs, self.doc_len = name, n_docs, doc_len
        self.fail_at, self.unavailable = fail_at, unavailable
        self.opens = 0

    def doc(self, i):
        return f"{self.name}-{i:07d} " + _body(self.name, i, self.doc_len)

    def __call__(self, skip, prefer=None):
        self.opens += 1
        if self.unavailable:
            raise SourceUnavailable("offline")

        def gen():
            for i in range(skip, self.n_docs):
                if self.fail_at is not None and i >= self.fail_at:
                    raise ConnectionError("network dropped")
                yield self.doc(i)
        return gen(), f"fake:{self.name}"


def _docs_in(path):
    """bin を EOT で区切って文書文字列に戻す。"""
    arr = np.fromfile(path, dtype=np.uint16)
    docs, cur = [], []
    for t in arr.tolist():
        if t == EOT:
            docs.append("".join(map(chr, cur)))
            cur = []
        else:
            cur.append(t)
    return docs


def _build(out_dir, budget, sources, **kw):
    kw.setdefault("mix", {"web": 0.5, "wikipedia": 0.3, "code": 0.2})
    kw.setdefault("safety", False)
    kw.setdefault("flush_tokens", 5_000)
    kw.setdefault("batch_docs", 8)
    kw.setdefault("val_fraction", 0.05)
    kw.setdefault("max_val_tokens", 10_000)
    kw.setdefault("anneal_ratio", 0.2)
    kw.setdefault("high_quality", ("code",))
    return build_token_corpus(budget, out_dir=out_dir, sources=sources, tokenizer=CharTokenizer(),
                              eot_id=EOT, tokenizer_fp="test", **kw)


def _sources(**overrides):
    srcs = {n: FakeSource(n) for n in ("web", "wikipedia", "code")}
    srcs.update(overrides)
    return srcs


class TestBudgetAndMix(unittest.TestCase):
    def test_budget_met_and_mix_followed(self):
        with tempfile.TemporaryDirectory() as d:
            meta = _build(d, 100_000, _sources(), anneal_ratio=0.0)
            self.assertGreaterEqual(meta["train_tokens"], 100_000)
            self.assertLess(meta["train_tokens"], 100_000 + 220)            # 超過は最後の 1 文書ぶんまで
            for src, target in {"web": 0.5, "wikipedia": 0.3, "code": 0.2}.items():
                self.assertAlmostEqual(meta["realized_mix"][src], target, delta=0.03)
            self.assertEqual(os.path.getsize(os.path.join(d, "train.bin")), meta["train_tokens"] * 2)

    def test_long_documents_do_not_overshoot_the_budget(self):
        """論文のような長文（1 文書 2 万字）でも、バッチ単位ではなく文書単位で予算に止まる。"""
        long_docs = {n: FakeSource(n, doc_len=20_000) for n in ("web", "wikipedia", "code")}
        with tempfile.TemporaryDirectory() as d:
            meta = _build(d, 30_000, long_docs, batch_docs=64, batch_chars=50_000, val_fraction=0.0)
        self.assertGreaterEqual(meta["train_tokens"], 30_000)
        self.assertLess(meta["train_tokens"], 30_000 + 20_020)

    def test_sources_are_interleaved_not_concatenated(self):
        """逐次読みの学習でも混ざるよう、先頭付近から複数ソースが現れる。"""
        with tempfile.TemporaryDirectory() as d:
            _build(d, 50_000, _sources(), anneal_ratio=0.0)
            head = _docs_in(os.path.join(d, "train.bin"))[:60]
            self.assertEqual({doc.split("-")[0] for doc in head}, {"web", "wikipedia", "code"})

    def test_val_is_disjoint_from_train(self):
        with tempfile.TemporaryDirectory() as d:
            meta = _build(d, 60_000, _sources())
            train = set(_docs_in(os.path.join(d, "train.bin")))
            val = set(_docs_in(os.path.join(d, "val.bin")))
            self.assertGreater(len(val), 0)
            self.assertEqual(train & val, set())
            self.assertEqual(meta["val_tokens"], os.path.getsize(os.path.join(d, "val.bin")) // 2)

    def test_exhausted_source_share_goes_to_others(self):
        with tempfile.TemporaryDirectory() as d:
            meta = _build(d, 60_000, _sources(code=FakeSource("code", n_docs=5)), anneal_ratio=0.0)
            self.assertGreaterEqual(meta["train_tokens"], 60_000)
            self.assertEqual(meta["sources"]["code"]["status"], "exhausted")
            self.assertLess(meta["realized_mix"]["code"], 0.05)

    def test_unavailable_source_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            meta = _build(d, 40_000, _sources(code=FakeSource("code", unavailable=True)))
            self.assertEqual(meta["sources"]["code"]["status"], "unavailable")
            self.assertGreaterEqual(meta["train_tokens"], 40_000)

    def test_all_sources_short_still_completes_with_warning(self):
        tiny = {n: FakeSource(n, n_docs=3) for n in ("web", "wikipedia", "code")}
        with tempfile.TemporaryDirectory() as d:
            with self.assertLogs("data.token_corpus", level="WARNING"):
                meta = _build(d, 1_000_000, tiny)
            self.assertTrue(meta["complete"])
            self.assertLess(meta["train_tokens"], 1_000_000)

    def test_no_tokens_at_all_raises(self):
        dead = {n: FakeSource(n, unavailable=True) for n in ("web", "wikipedia", "code")}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                _build(d, 10_000, dead)


class TestAnnealing(unittest.TestCase):
    def test_tail_upweights_high_quality_sources_but_keeps_general(self):
        with tempfile.TemporaryDirectory() as d:
            meta = _build(d, 100_000, _sources(), anneal_ratio=0.3, anneal_hq_weight=3.0)
            code = meta["sources"]["code"]["phase_tokens"]
            main_share = code["main"] / meta["phase_tokens"]["main"]
            anneal_share = code["anneal"] / meta["phase_tokens"]["anneal"]
            self.assertAlmostEqual(main_share, 0.2, delta=0.03)
            self.assertAlmostEqual(anneal_share, 0.6 / 1.4, delta=0.04)    # 0.2*3 / (0.5+0.3+0.6)
            self.assertGreater(meta["sources"]["web"]["phase_tokens"]["anneal"], 0)


class TestResume(unittest.TestCase):
    def test_interrupted_build_resumes_to_identical_bytes(self):
        with tempfile.TemporaryDirectory() as ref, tempfile.TemporaryDirectory() as d:
            _build(ref, 80_000, _sources())

            flaky = FakeSource("web", fail_at=150)
            with self.assertRaises(ConnectionError):
                _build(d, 80_000, _sources(web=flaky))
            with open(os.path.join(d, "train.bin"), "ab") as f:
                f.write(b"\xff\xff" * 37)                  # 最終保存点より後の中途半端な追記

            flaky.fail_at = None                            # ネットワーク復旧
            meta = _build(d, 80_000, _sources(web=flaky))
            self.assertTrue(meta["complete"])
            for name in ("train.bin", "val.bin"):
                with open(os.path.join(ref, name), "rb") as a, open(os.path.join(d, name), "rb") as b:
                    self.assertEqual(a.read(), b.read(), name)

    def test_complete_corpus_is_reused_without_reading_sources(self):
        with tempfile.TemporaryDirectory() as d:
            _build(d, 20_000, _sources())
            srcs = _sources()
            _build(d, 20_000, srcs)
            self.assertEqual(sum(s.opens for s in srcs.values()), 0)

    def test_budget_change_rebuilds(self):
        with tempfile.TemporaryDirectory() as d:
            _build(d, 20_000, _sources())
            meta = _build(d, 40_000, _sources())
            self.assertGreaterEqual(meta["train_tokens"], 40_000)
            self.assertEqual(os.path.getsize(os.path.join(d, "train.bin")), meta["train_tokens"] * 2)

    def test_source_that_worked_before_must_not_silently_change(self):
        """再開時に以前使えたソースが開けないなら、構成を黙って変えずに止める。"""
        with tempfile.TemporaryDirectory() as d:
            flaky = FakeSource("web", fail_at=100)
            with self.assertRaises(ConnectionError):
                _build(d, 60_000, _sources(web=flaky))
            with self.assertRaises(RuntimeError):
                _build(d, 60_000, _sources(web=FakeSource("web", unavailable=True)))


class TestBoundedMemory(unittest.TestCase):
    def _peak(self, budget):
        with tempfile.TemporaryDirectory() as d:
            tracemalloc.start()
            _build(d, budget, _sources(), flush_tokens=20_000)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        return peak

    def test_peak_memory_does_not_grow_with_budget(self):
        small, large = self._peak(100_000), self._peak(1_000_000)
        self.assertLess(large, 8 * 1024 * 1024)             # 10 倍の予算でも数 MB に収まる
        self.assertLess(large, small * 2.0)                 # 予算に比例しない


class TestSafetyFilters(unittest.TestCase):
    def test_pii_masked_and_toxic_dropped(self):
        class Docs:
            opens = 0

            def __call__(self, skip, prefer=None):
                docs = ["連絡先は taro@example.com です。" + "本文" * 40,
                        "死ね " * 60,
                        "普通の百科事典的な記述です。" * 10] * 50
                return iter(docs[skip:]), "fake:docs"
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"SWI_TOXICITY_FILTER": "1"}):
            meta = build_token_corpus(2_000, out_dir=d, sources={"web": Docs()}, mix={"web": 1.0},
                                      tokenizer=CharTokenizer(), eot_id=EOT, tokenizer_fp="t",
                                      safety=True, val_fraction=0.0, anneal_ratio=0.0)
            text = "".join(_docs_in(os.path.join(d, "train.bin")))
        self.assertNotIn("taro@example.com", text)
        self.assertIn("<EMAIL>", text)
        self.assertNotIn("死ね", text)
        self.assertGreater(meta["sources"]["web"]["dropped_docs"], 0)


class TestLoad(unittest.TestCase):
    def test_load_returns_memmaps_for_complete_corpus_only(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_token_corpus(d))
            meta = _build(d, 20_000, _sources())
            train, val, loaded_meta = load_token_corpus(d)
            self.assertEqual(len(train), meta["train_tokens"])
            self.assertEqual(len(val), meta["val_tokens"])
            self.assertEqual(train.dtype, np.uint16)


class TestPretrainIntegration(unittest.TestCase):
    def test_budget_mode_reads_corpus_sequentially_without_document_lists(self):
        from swi_llm.data import pretrain
        tokens = np.arange(5_000, dtype=np.uint16)
        val = np.arange(2_000, dtype=np.uint16)
        meta = {"realized_mix": {"web": 1.0}}
        with mock.patch.dict(pretrain.config, {"pretrain_token_budget": 5_000}), \
             mock.patch.object(token_corpus, "load_or_build_token_corpus",
                               return_value=(tokens, val, meta)) as build, \
             mock.patch.object(pretrain, "build_mixed_documents",
                               side_effect=AssertionError("document lists must not be built")):
            train_loader, val_loader = pretrain.get_text_dataloaders(seq_len=16, batch_size=2)
            build.assert_called_once_with(5_000)
            x, y = next(iter(train_loader))
        self.assertEqual(x[0].tolist(), list(range(0, 16)))          # 先頭から順に
        self.assertEqual(x[1].tolist(), list(range(16, 32)))
        self.assertEqual(y[0].tolist(), list(range(1, 17)))


class TestParseTokenCount(unittest.TestCase):
    def test_units_and_forms(self):
        self.assertEqual(parse_token_count("4B"), 4_000_000_000)
        self.assertEqual(parse_token_count("500M"), 500_000_000)
        self.assertEqual(parse_token_count("200k"), 200_000)
        self.assertEqual(parse_token_count("4e9"), 4_000_000_000)
        self.assertEqual(parse_token_count("1_000"), 1_000)
        self.assertEqual(parse_token_count(3e6), 3_000_000)
        self.assertIsNone(parse_token_count(""))
        self.assertIsNone(parse_token_count(None))

    def test_invalid_raises(self):
        for bad in ("abc", "0", "-5M", "4X"):
            with self.assertRaises(ValueError, msg=bad):
                parse_token_count(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
