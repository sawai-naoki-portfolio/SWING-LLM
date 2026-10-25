"""
Self-RAG (swi_llm/agent/self_rag.py) の純粋ロジックの単体テスト（§13.2）。
SimpleRAG（埋め込みモデルロード）を伴わない should_retrieve / grounding_score を対象とする。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.agent.self_rag import should_retrieve, grounding_score, _char_bigrams


class TestShouldRetrieve(unittest.TestCase):
    def test_knowledge_query(self):
        self.assertTrue(should_retrieve("SWI-LLMのアーキテクチャとは？"))

    def test_search_keyword(self):
        self.assertTrue(should_retrieve("RAGについて検索して"))

    def test_question_marker(self):
        self.assertTrue(should_retrieve("水の化学式は何ですか"))

    def test_chitchat_skipped(self):
        self.assertFalse(should_retrieve("こんにちは"))
        self.assertFalse(should_retrieve("ありがとう"))


class TestGroundingScore(unittest.TestCase):
    def test_high_overlap(self):
        s = grounding_score("機械学習のアーキテクチャ", "機械学習のアーキテクチャは重要な概念です")
        self.assertGreater(s, 0.08)

    def test_low_overlap(self):
        s = grounding_score("東京の天気", "全く無関係なXYZABCの内容")
        self.assertLess(s, 0.08)

    def test_empty(self):
        self.assertEqual(grounding_score("", "something"), 0.0)
        self.assertEqual(grounding_score("q", ""), 0.0)

    def test_bigrams(self):
        self.assertEqual(_char_bigrams("abc"), {"ab", "bc"})
        self.assertEqual(_char_bigrams(""), set())


if __name__ == "__main__":
    unittest.main()
