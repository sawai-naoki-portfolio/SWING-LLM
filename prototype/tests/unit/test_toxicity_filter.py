"""
tests/unit/test_toxicity_filter.py

NEW-L2（01_LLM_Architecture §2「安全性 (PII・Toxicity)」）: 有害コンテンツフィルタ。
密度方式の要点 — 「単発の言及は通し、有害語で飽和した文書を落とす」— を固定する。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data.advanced_pipeline import ToxicityFilter, toxicity_filter_enabled


class TestToxicityFilter(unittest.TestCase):
    def setUp(self):
        self.f = ToxicityFilter()

    def test_clean_educational_text_passes(self):
        text = ("量子力学とは、原子や電子などの微視的な系を記述する物理学の理論である。"
                "例えば、電子のエネルギーは離散的な値をとる。したがって古典力学では説明できない"
                "現象を説明できる。1900 年のプランクの量子仮説に始まり、多くの実験で検証された。") * 3
        self.assertFalse(self.f.is_toxic(text))

    def test_block_term_drops_regardless_of_length(self):
        long_clean = "これは長い普通の文章です。" * 200
        self.assertTrue(self.f.is_toxic(long_clean + " nigger"))

    def test_english_block_requires_word_boundary(self):
        # 部分文字列では誤爆しない（\b 境界。Scunthorpe 問題の類）
        self.assertFalse(self.f.is_toxic("The snigger of the crowd echoed. " * 20))

    def test_saturated_profanity_drops(self):
        text = "死ね クソ野郎 死ね 死ね ぶっ殺すぞ " * 20      # 高密度の罵倒
        self.assertTrue(self.f.is_toxic(text))

    def test_single_mention_in_long_doc_passes(self):
        # 報道・百科事典的な単発の言及（密度が閾値未満）は通す
        text = ("この事件では被告が被害者を脅迫した。目撃者によれば「死ね」と叫んだという。"
                + "その後の裁判では、犯行の経緯と社会的背景が詳細に検討された。" * 30)
        self.assertFalse(self.f.is_toxic(text))

    def test_english_profanity_density(self):
        self.assertTrue(self.f.is_toxic("fuck this shit you bitch " * 20))
        # 単発の言及 + 長い通常文（密度が閾値未満）は通す
        self.assertFalse(self.f.is_toxic(
            "The film contains one use of the word fuck, noted by reviewers. "
            + "The rest of the review discusses cinematography and pacing in detail. " * 30))

    def test_empty_and_none_are_safe(self):
        self.assertFalse(self.f.is_toxic(""))
        self.assertFalse(self.f.is_toxic(None))

    def test_case_insensitive_english(self):
        self.assertTrue(self.f.is_toxic("FUCK FUCK FUCK " * 30))


class TestToxicityEnabledFlag(unittest.TestCase):
    def test_default_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWI_TOXICITY_FILTER", None)
            self.assertTrue(toxicity_filter_enabled())

    def test_env_disable(self):
        with mock.patch.dict(os.environ, {"SWI_TOXICITY_FILTER": "0"}):
            self.assertFalse(toxicity_filter_enabled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
