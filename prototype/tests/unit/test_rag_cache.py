"""
S5: SelfRAG のキャッシュ（シングルトン化）の単体テスト。

search_documents が呼ばれる度に SelfRAG(=SimpleRAG) を new すると SentenceTransformer
ロード + FAISS 構築が毎回走り致命的に遅い。get_self_rag は kb_dir ごとに 1 度だけ
構築してキャッシュすることを検証する（重い SelfRAG はスタブに差し替えて高速・独立に検証）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import swi_llm.agent.self_rag as sr


class _StubRAG:
    """SelfRAG の代わりに生成回数を数える軽量スタブ。"""
    instances = 0

    def __init__(self, kb_dir=None):
        _StubRAG.instances += 1
        self.kb_dir = kb_dir


class TestRagCache(unittest.TestCase):
    def setUp(self):
        self._orig = sr.SelfRAG
        sr.SelfRAG = _StubRAG
        _StubRAG.instances = 0
        sr.reset_self_rag()

    def tearDown(self):
        sr.SelfRAG = self._orig
        sr.reset_self_rag()

    def test_same_kb_dir_builds_once(self):
        a = sr.get_self_rag("kb1")
        b = sr.get_self_rag("kb1")
        c = sr.get_self_rag("kb1")
        self.assertIs(a, b)
        self.assertIs(b, c)
        self.assertEqual(_StubRAG.instances, 1)   # 3 回呼んでも構築は 1 回

    def test_different_kb_dir_separate_instances(self):
        a = sr.get_self_rag("kb1")
        b = sr.get_self_rag("kb2")
        self.assertIsNot(a, b)
        self.assertEqual(_StubRAG.instances, 2)

    def test_default_key_shared(self):
        a = sr.get_self_rag()
        b = sr.get_self_rag(None)
        self.assertIs(a, b)
        self.assertEqual(_StubRAG.instances, 1)

    def test_reset_forces_rebuild(self):
        a = sr.get_self_rag("kb1")
        sr.reset_self_rag()
        b = sr.get_self_rag("kb1")
        self.assertIsNot(a, b)
        self.assertEqual(_StubRAG.instances, 2)


if __name__ == "__main__":
    unittest.main()
