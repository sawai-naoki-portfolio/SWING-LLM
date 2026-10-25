"""
データミックス (swi_llm/data/data_mix.py) の設定ロジックの単体テスト（§1.5）。
ネットワーク不要な純粋ロジック（配分計算・比率）のみを対象とする。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data.data_mix import _allocate_doc_counts
from swi_llm.utils.constants import (
    PRETRAIN_DATA_MIX, PRETRAIN_MAX_DOCS_PER_SOURCE, PRETRAIN_ANNEAL_RATIO,
)


class TestDataMix(unittest.TestCase):
    def test_ratios_sum_to_one(self):
        self.assertAlmostEqual(sum(PRETRAIN_DATA_MIX.values()), 1.0, places=6)

    def test_allocation_respects_ratios(self):
        counts = _allocate_doc_counts(1000)
        # §1.5 準拠の混合比率（web30/wiki15/code20/book10/arxiv10/dolly15）
        self.assertEqual(counts["web"], 300)
        self.assertEqual(counts["wikipedia"], 150)
        self.assertEqual(counts["code"], 200)
        self.assertEqual(counts["book"], 100)
        self.assertEqual(counts["arxiv"], 100)
        self.assertEqual(counts["dolly"], 150)

    def test_code_source_present(self):
        # 専用コードコーパスが混合比率に含まれる (§1.5)
        self.assertIn("code", PRETRAIN_DATA_MIX)

    def test_web_source_present(self):
        # CC/mC4 系 Web テキストが混合比率に含まれる (§1.5 の Web 枠)
        self.assertIn("web", PRETRAIN_DATA_MIX)

    def test_book_source_present(self):
        # 書籍（青空文庫）が混合比率に含まれる (§1.5 の書籍枠)
        self.assertIn("book", PRETRAIN_DATA_MIX)

    def test_web_is_largest_share(self):
        # Web が最大シェア（§1.5: Web 45-55% を最大のテコとする）
        self.assertEqual(max(PRETRAIN_DATA_MIX, key=PRETRAIN_DATA_MIX.get), "web")

    def test_allocation_small_budget_min_one(self):
        counts = _allocate_doc_counts(1)
        # どのソースも最低 1 件は割り当てる
        for src in PRETRAIN_DATA_MIX:
            self.assertGreaterEqual(counts[src], 1)

    def test_allocation_none_uses_source_cap(self):
        counts = _allocate_doc_counts(None)
        for src in PRETRAIN_DATA_MIX:
            self.assertEqual(counts[src], PRETRAIN_MAX_DOCS_PER_SOURCE)

    def test_anneal_ratio_in_range(self):
        self.assertTrue(0.0 < PRETRAIN_ANNEAL_RATIO < 1.0)


if __name__ == "__main__":
    unittest.main()
