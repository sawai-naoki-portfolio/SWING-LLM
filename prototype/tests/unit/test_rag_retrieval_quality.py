"""
RAG 検索段（RRF 統合 + クロスエンコーダ再順位付け — 実装編 §10.2 / Q-12 /
roadmap ステップ4）の単体テスト。

重いモデルはロードせず、決定的なフェイク埋め込み・フェイクリランカを DI で注入する。
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from swi_llm.agent.rag import SimpleRAG, reciprocal_rank_fusion, RAG_RRF_K


class KeywordEmbedder:
    """キーワード → 直交軸の決定的埋め込み（8 次元）。"""

    KEYWORDS = ["犬", "猫", "天気", "経費"]

    def encode(self, texts, convert_to_numpy=True):
        out = []
        for t in texts:
            v = np.zeros(8, dtype="float32")
            for i, kw in enumerate(self.KEYWORDS):
                if kw in t:
                    v[i] = 1.0
            v[7] = 0.05   # 非ゼロ保証
            out.append(v)
        return np.stack(out)


class SubstringReranker:
    """クエリの完全一致フレーズ数でスコアするフェイク・クロスエンコーダ。"""

    def __init__(self, phrase):
        self.phrase = phrase
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [float(doc.count(self.phrase)) for _query, doc in pairs]


class BrokenReranker:
    def predict(self, pairs):
        raise RuntimeError("cross-encoder backend down")


_DOCS = {
    "a_dogs.txt": "犬は忠実な動物です。毎日の散歩が大好きです。",
    "b_cats.txt": "猫は気まぐれな動物です。日向での昼寝が大好きです。",
    "c_weather.txt": "天気の観測は毎朝行います。雨の日は記録を二重に取ります。",
    "d_expense.txt": "経費精算の締め切りは毎月末です。領収書の原本が必要です。",
}


class TestReciprocalRankFusion(unittest.TestCase):
    def test_formula(self):
        scores = reciprocal_rank_fusion([[0, 1, 2], [2, 0]], k=60)
        self.assertAlmostEqual(scores[0], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores[1], 1 / 62)
        self.assertAlmostEqual(scores[2], 1 / 63 + 1 / 61)

    def test_both_lists_beat_single_list(self):
        """両方のランキングに現れる文書は、片方だけの同順位文書より上に来る。"""
        scores = reciprocal_rank_fusion([[0], [0, 1]], k=RAG_RRF_K)
        self.assertGreater(scores[0], scores[1])

    def test_empty_rankings(self):
        self.assertEqual(reciprocal_rank_fusion([[], []]), {})


class TestRetrievalQuality(unittest.TestCase):
    """固定 KB + 決定的埋め込みでの recall（検索品質の回帰ゲート）。"""

    @classmethod
    def setUpClass(cls):
        cls.kb_dir = tempfile.mkdtemp(prefix="swi_rag_quality_")
        for name, text in _DOCS.items():
            with open(os.path.join(cls.kb_dir, name), "w", encoding="utf-8") as f:
                f.write(text + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.kb_dir, ignore_errors=True)

    def _rag(self, **kwargs):
        return SimpleRAG(kb_dir=self.kb_dir, embedding_model=KeywordEmbedder(), **kwargs)

    def test_recall_at_2_is_full_on_benchmark(self):
        cases = [
            ("犬の散歩について", "犬は忠実"),
            ("猫の昼寝", "猫は気まぐれ"),
            ("天気の観測方法", "天気の観測"),
            ("経費精算の締め切りはいつ？", "経費精算"),
        ]
        rag = self._rag()
        hits = sum(1 for q, expect in cases if expect in rag.search(q, top_k=2))
        self.assertEqual(hits, len(cases), "recall@2 が 1.0 を割った（検索品質の回帰）")

    def test_reranker_overrides_rrf_order(self):
        """RRF では負けている文書を、リランカが一致度で最上位へ引き上げる。"""
        # クエリ「動物」: BM25 は犬・猫の両方にヒットし、埋め込みは無反応 → RRF はほぼ同格
        # フェイクリランカは「昼寝」を含む文書（猫）を強くスコアする
        rag = self._rag(reranker=SubstringReranker("昼寝"))
        result = rag.search("動物の昼寝", top_k=1)
        self.assertIn("猫は気まぐれ", result)
        self.assertNotIn("犬は忠実", result)
        self.assertEqual(rag.reranker.calls, 1)

    def test_without_reranker_rrf_order_is_used(self):
        rag = self._rag()
        self.assertIsNone(rag.reranker, "DI 埋め込み時に既定リランカを自動ロードしてはならない")
        result = rag.search("犬の散歩", top_k=1)
        self.assertIn("犬は忠実", result)

    def test_broken_reranker_falls_back_to_rrf(self):
        """リランカ障害時は RRF 順で継続する（C-4: 追加段の障害で検索を殺さない）。

        クエリは複数文書にヒットさせる（候補 1 件ではリランカ自体が呼ばれないため、
        「動物」で犬・猫の両文書を候補に載せ、「散歩」で RRF が犬を上位にする）。"""
        rag = self._rag(reranker=BrokenReranker())
        with self.assertLogs("rag", level="WARNING") as cm:
            result = rag.search("動物の散歩", top_k=1)
        self.assertIn("犬は忠実", result)
        self.assertTrue(any("falling back to RRF" in m for m in cm.output))

    def test_unrelated_query_returns_not_found(self):
        """語彙ヒットも密類似も無いクエリは「見つからない」を返す（候補ゲート）。"""
        rag = self._rag()
        result = rag.search("量子力学のシュレディンガー方程式", top_k=2)
        self.assertIn("見つかりませんでした", result)

    def test_fetch_k_limits_rerank_candidates(self):
        reranker = SubstringReranker("動物")
        rag = self._rag(reranker=reranker)
        rag.search("動物", top_k=1, fetch_k=1)
        # fetch_k=1 だと候補 1 件 → リランカは呼ばれない（len(ordered)>1 のガード）
        self.assertEqual(reranker.calls, 0)


if __name__ == "__main__":
    unittest.main()
