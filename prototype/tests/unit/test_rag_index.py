"""
SimpleRAG の 02(小) 対応（コサイン化 IndexFlatIP / 密インデックスのディスク永続化 /
埋め込みモデル DI）の単体テスト。

重い SentenceTransformer はロードせず、決定的なフェイク埋め込みモデルを DI で注入して
高速・独立に検証する（faiss は本体を使用）。
"""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from swi_llm.agent.rag import SimpleRAG


class FakeEmbedder:
    """「犬」「猫」を直交軸に写す決定的な 8 次元埋め込み（呼び出し回数を記録）。"""

    def __init__(self):
        self.corpus_calls = 0   # 複数テキスト（コーパス構築）の encode 回数
        self.query_calls = 0    # 単一テキスト（クエリ）の encode 回数

    def encode(self, texts, convert_to_numpy=True):
        if len(texts) == 1:
            self.query_calls += 1
        else:
            self.corpus_calls += 1
        out = []
        for t in texts:
            v = np.zeros(8, dtype="float32")
            v[0] = 1.0 if "犬" in t else 0.0
            v[1] = 1.0 if "猫" in t else 0.0
            v[2] = 0.1   # 全ベクトルを非ゼロに保つ
            out.append(v)
        return np.stack(out)


def _write_kb(kb_dir):
    with open(os.path.join(kb_dir, "a_dogs.txt"), "w", encoding="utf-8") as f:
        f.write("犬は忠実な動物です。毎日の散歩が大好きです。\n")
    with open(os.path.join(kb_dir, "b_cats.txt"), "w", encoding="utf-8") as f:
        f.write("猫は気まぐれな動物です。日向での昼寝が大好きです。\n")


class TestRagDenseIndex(unittest.TestCase):
    def setUp(self):
        self.kb_dir = tempfile.mkdtemp(prefix="swi_rag_test_")
        _write_kb(self.kb_dir)

    def tearDown(self):
        shutil.rmtree(self.kb_dir, ignore_errors=True)

    def test_cosine_ranking_prefers_semantic_match(self):
        """IndexFlatIP（正規化内積 = コサイン）で意味的に近い文書が上位になる。
        単文字クエリ「犬」は BM25（バイグラム）では当たらないため、密検索の寄与を直接検証できる。"""
        rag = SimpleRAG(kb_dir=self.kb_dir, embedding_model=FakeEmbedder())
        self.assertTrue(rag.use_dense)
        result = rag.search("犬", top_k=1)
        self.assertIn("犬は忠実", result)
        self.assertNotIn("猫は気まぐれ", result)

    def test_index_cache_skips_reencoding(self):
        """2 回目の起動はディスクキャッシュを使い、コーパスの再エンベッドをしない。"""
        emb1 = FakeEmbedder()
        SimpleRAG(kb_dir=self.kb_dir, embedding_model=emb1)
        self.assertEqual(emb1.corpus_calls, 1)

        emb2 = FakeEmbedder()
        rag2 = SimpleRAG(kb_dir=self.kb_dir, embedding_model=emb2)
        self.assertEqual(emb2.corpus_calls, 0, "キャッシュがあるのに再エンベッドした")
        # キャッシュロード後も検索は機能する（クエリのみ encode される）
        result = rag2.search("犬", top_k=1)
        self.assertIn("犬は忠実", result)
        self.assertEqual(emb2.query_calls, 1)

    def test_kb_change_invalidates_cache(self):
        """KB へのファイル追加でキャッシュが無効化され、再構築される。"""
        SimpleRAG(kb_dir=self.kb_dir, embedding_model=FakeEmbedder())
        with open(os.path.join(self.kb_dir, "c_birds.txt"), "w", encoding="utf-8") as f:
            f.write("鳥は空を飛ぶ動物です。さえずりがきれいです。\n")

        emb = FakeEmbedder()
        rag = SimpleRAG(kb_dir=self.kb_dir, embedding_model=emb)
        self.assertEqual(emb.corpus_calls, 1, "KB 変更後もキャッシュを使ってしまった")
        self.assertEqual(rag.index.ntotal, 3)

    def test_cache_dir_is_not_indexed_as_document(self):
        """.rag_cache 配下（meta.json 等）が KB 文書として誤って索引されない。"""
        rag1 = SimpleRAG(kb_dir=self.kb_dir, embedding_model=FakeEmbedder())
        n_docs = len(rag1.documents)
        rag2 = SimpleRAG(kb_dir=self.kb_dir, embedding_model=FakeEmbedder())
        self.assertEqual(len(rag2.documents), n_docs)

    def test_cache_works_under_non_ascii_path(self):
        """回帰: faiss.write_index の C++ fopen は Windows の非 ASCII パス
        （本番 KB は「デスクトップ」配下）で 'Illegal byte sequence' となり、
        永続化が黙って無効になっていた。serialize_index + Python IO で解消。"""
        kb = tempfile.mkdtemp(prefix="swi_rag_日本語パス_検証_")
        try:
            _write_kb(kb)
            emb1 = FakeEmbedder()
            SimpleRAG(kb_dir=kb, embedding_model=emb1)
            self.assertEqual(emb1.corpus_calls, 1)
            self.assertTrue(
                os.path.exists(os.path.join(kb, ".rag_cache", "index.faiss")),
                "非 ASCII パスでキャッシュが書けていない")

            emb2 = FakeEmbedder()
            rag2 = SimpleRAG(kb_dir=kb, embedding_model=emb2)
            self.assertEqual(emb2.corpus_calls, 0, "非 ASCII パスのキャッシュを読めていない")
            self.assertIn("犬は忠実", rag2.search("犬", top_k=1))
        finally:
            shutil.rmtree(kb, ignore_errors=True)

    def test_bm25_only_fallback_still_searches(self):
        """faiss 不在の環境では BM25 のみで動作する（依存欠如で全機能を殺さない）。"""
        with mock.patch.dict(sys.modules, {"faiss": None}):
            rag = SimpleRAG(kb_dir=self.kb_dir, embedding_model=FakeEmbedder())
        self.assertFalse(rag.use_dense)
        result = rag.search("犬は忠実な動物", top_k=1)
        self.assertIn("犬は忠実", result)


if __name__ == "__main__":
    unittest.main()
