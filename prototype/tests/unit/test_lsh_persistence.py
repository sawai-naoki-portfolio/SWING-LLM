"""
tests/unit/test_lsh_persistence.py

NEW-L3（04_data 展望2）: MinHash LSH インデックスの sqlite 永続化と増分（run 跨ぎ）重複検知。
"""
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data.advanced_pipeline import MinHashLSHLocal

DOC = "量子力学とは、微視的な系を記述する物理学の基礎理論である。プランクの量子仮説に始まる。" * 3
NEAR_DUP = DOC + "（一部加筆）"
OTHER = "経済学は希少な資源の配分を研究する社会科学である。需要と供給の均衡が中心概念となる。" * 3


class TestInMemoryUnchanged(unittest.TestCase):
    def test_exact_and_near_duplicates_detected(self):
        lsh = MinHashLSHLocal()
        self.assertFalse(lsh.check_duplicate_and_insert(1, DOC))
        self.assertTrue(lsh.check_duplicate_and_insert(2, DOC))        # 完全一致
        self.assertTrue(lsh.check_duplicate_and_insert(3, NEAR_DUP))   # 近重複
        self.assertFalse(lsh.check_duplicate_and_insert(4, OTHER))     # 別文書

    def test_close_is_noop_for_memory(self):
        lsh = MinHashLSHLocal()
        lsh.close()                                                     # 例外なし


class TestSqlitePersistence(unittest.TestCase):
    def test_duplicates_detected_across_instances(self):
        """run を跨いだ増分 dedup: 前 run で見た文書を次 run のインスタンスが検知する。"""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lsh.sqlite")
            run1 = MinHashLSHLocal(store_path=path)
            self.assertFalse(run1.check_duplicate_and_insert("r1-doc", DOC))
            run1.close()

            run2 = MinHashLSHLocal(store_path=path)                     # 別インスタンス = 次の run
            self.assertTrue(run2.check_duplicate_and_insert("r2-dup", DOC))
            self.assertTrue(run2.check_duplicate_and_insert("r2-near", NEAR_DUP))
            self.assertFalse(run2.check_duplicate_and_insert("r2-new", OTHER))
            run2.close()

    def test_new_doc_registered_persistently(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lsh.sqlite")
            a = MinHashLSHLocal(store_path=path)
            a.check_duplicate_and_insert(1, DOC)
            a.check_duplicate_and_insert(2, OTHER)
            a.close()
            b = MinHashLSHLocal(store_path=path)
            self.assertTrue(b.check_duplicate_and_insert(3, OTHER))     # 2 run 目でも既知
            b.close()

    def test_memory_and_sqlite_agree(self):
        """同じ入力列に対してメモリ版と sqlite 版の判定が一致する（バックエンド差なし）。"""
        docs = [DOC, OTHER, DOC, NEAR_DUP, OTHER + "追記です。" * 5]
        mem = MinHashLSHLocal()
        mem_results = [mem.check_duplicate_and_insert(i, t) for i, t in enumerate(docs)]
        with tempfile.TemporaryDirectory() as d:
            db = MinHashLSHLocal(store_path=os.path.join(d, "l.sqlite"))
            db_results = [db.check_duplicate_and_insert(i, t) for i, t in enumerate(docs)]
            db.close()
        self.assertEqual(mem_results, db_results)


if __name__ == "__main__":
    unittest.main(verbosity=2)
