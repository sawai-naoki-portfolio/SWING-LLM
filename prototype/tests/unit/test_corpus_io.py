"""
data/corpus_io.py（Parquet シャード I/O — ステップ5 / fix 03）の単体テスト。
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data.corpus_io import (
    write_parquet_shards, iter_parquet_texts, has_parquet_corpus, MANIFEST_FILE,
)


class TestCorpusIO(unittest.TestCase):
    def setUp(self):
        self.out_dir = tempfile.mkdtemp(prefix="swi_corpus_io_")

    def tearDown(self):
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_roundtrip_preserves_order_and_content(self):
        docs = [f"文書{i}の内容です。日本語テキスト。" for i in range(10)]
        n = write_parquet_shards(({"id": i, "text": t} for i, t in enumerate(docs)), self.out_dir)
        self.assertEqual(n, 10)
        self.assertTrue(has_parquet_corpus(self.out_dir))
        self.assertEqual(list(iter_parquet_texts(self.out_dir)), docs)

    def test_sharding_splits_files_and_keeps_order(self):
        docs = [f"doc-{i:03d}" for i in range(25)]
        write_parquet_shards(({"text": t} for t in docs), self.out_dir, shard_rows=10)
        shards = [f for f in os.listdir(self.out_dir) if f.endswith(".parquet")]
        self.assertEqual(len(shards), 3)          # 10+10+5
        self.assertEqual(list(iter_parquet_texts(self.out_dir)), docs)

    def test_empty_and_missing_text_rows_are_skipped(self):
        n = write_parquet_shards(iter([{"text": ""}, {"id": 1}, {"text": "有効"}]), self.out_dir)
        self.assertEqual(n, 1)
        self.assertEqual(list(iter_parquet_texts(self.out_dir)), ["有効"])

    def test_missing_manifest_is_not_a_corpus(self):
        self.assertFalse(has_parquet_corpus(self.out_dir))
        self.assertFalse(has_parquet_corpus(os.path.join(self.out_dir, "nope")))

    def test_batch_iteration_smaller_than_shard(self):
        docs = [f"batch-{i}" for i in range(30)]
        write_parquet_shards(({"text": t} for t in docs), self.out_dir, shard_rows=30)
        self.assertEqual(list(iter_parquet_texts(self.out_dir, batch_rows=7)), docs)

    def test_manifest_records_counts(self):
        import json
        write_parquet_shards(({"text": f"t{i}"} for i in range(5)), self.out_dir, shard_rows=2)
        with open(os.path.join(self.out_dir, MANIFEST_FILE), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["n_rows"], 5)
        self.assertEqual(sum(s["rows"] for s in manifest["shards"]), 5)


class TestDataMixParquetPreference(unittest.TestCase):
    """ステップ5: data_mix が Parquet コーパスを JSONL より優先して読むこと。"""

    def test_collect_wikipedia_prefers_parquet(self):
        from unittest import mock
        import swi_llm.data.data_mix as dm

        out_dir = tempfile.mkdtemp(prefix="swi_dm_parquet_")
        try:
            docs = [("パーケット由来の文書です。" * 30) + f"番号{i}" for i in range(3)]
            write_parquet_shards(({"text": t} for t in docs), out_dir)
            with mock.patch.multiple(
                    dm,
                    DATASET_WIKIPEDIA_CLEANED_PARQUET=out_dir,
                    DATASET_WIKIPEDIA_CLEANED=os.path.join(out_dir, "no_such.jsonl")):
                collected = dm._collect_wikipedia(limit=2)
            self.assertEqual(len(collected), 2)
            self.assertTrue(all("パーケット由来" in d for d in collected))
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
