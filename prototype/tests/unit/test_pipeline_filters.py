"""
データパイプラインの純粋ロジック (data/advanced_pipeline.py) の単体テスト。
PIIMasker / QualityScorer / MinHashLSHLocal を対象とする（torch 不要）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data.advanced_pipeline import (
    PIIMasker, QualityScorer, EducationalValueScorer, MinHashLSHLocal,
)
from swi_llm.utils.constants import PIPELINE_QUALITY_MIN_SCORE, QUALITY_MAX_SCORE


class TestPIIMasker(unittest.TestCase):
    def setUp(self):
        self.masker = PIIMasker()

    def test_masks_email(self):
        self.assertIn("<EMAIL>", self.masker.mask("連絡先は test.user@example.com です"))

    def test_masks_phone(self):
        self.assertIn("<PHONE>", self.masker.mask("電話は 090-1234-5678 まで"))

    def test_masks_ip_address(self):
        self.assertIn("<IP_ADDRESS>", self.masker.mask("サーバーは 192.168.10.1 にある"))

    def test_leaves_normal_text(self):
        text = "これは個人情報を含まない普通の文章です。"
        self.assertEqual(self.masker.mask(text), text)


class TestQualityScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = QualityScorer()

    def test_short_text_scores_zero(self):
        self.assertEqual(self.scorer.score("短い"), 0)

    def test_empty_text_scores_zero(self):
        self.assertEqual(self.scorer.score(""), 0)

    def test_educational_text_passes_threshold(self):
        text = (
            "The Pythagorean theorem states that in a right triangle, "
            "the square of the hypotenuse equals the sum of the squares of the other two sides. "
            "For example, if the legs measure 3 and 4, then the hypotenuse measures 5, "
            "because 3*3 + 4*4 = 25. This fundamental relationship underpins geometry, "
            "trigonometry, and countless engineering calculations used throughout history."
        )
        self.assertGreaterEqual(self.scorer.score(text), PIPELINE_QUALITY_MIN_SCORE)

    def test_score_never_exceeds_max(self):
        text = "calculation " * 100 + "def compute(): pass. Why? Because 1+1, logic and math!"
        self.assertLessEqual(self.scorer.score(text), QUALITY_MAX_SCORE)


class TestEducationalValueScorer(unittest.TestCase):
    """教育的価値フィルタ (§1.1 / Q-02)。"""

    def setUp(self):
        self.scorer = EducationalValueScorer()

    def test_empty_and_short_score_zero(self):
        self.assertEqual(self.scorer.score(""), 0)
        self.assertEqual(self.scorer.score("短い"), 0)

    def test_educational_text_passes_threshold(self):
        text = (
            "光合成とは、植物が光エネルギーを利用して二酸化炭素と水から有機物を"
            "合成する反応である。なぜなら、葉緑体に存在するクロロフィルが光を吸収し、"
            "そのエネルギーで水を分解するためだ。例えば、6分子の二酸化炭素と6分子の水から、"
            "1分子のグルコースと6分子の酸素が生成される。この反応は地球上の生態系を"
            "支える基盤であり、食物連鎖の出発点となっている。"
        )
        self.assertGreaterEqual(self.scorer.score(text), PIPELINE_QUALITY_MIN_SCORE)

    def test_spam_boilerplate_scores_low(self):
        text = ("今すぐ登録！ 会員登録は https://example.com/signup から！ "
                "click here!!! buy now!!! ★★★★★ >>> !!! <<< @@@ ### $$$ %%%") * 3
        self.assertLess(self.scorer.score(text), PIPELINE_QUALITY_MIN_SCORE)

    def test_repetitive_lines_penalized(self):
        text = "\n".join(["これは全く同じ行の繰り返しでスパム的な羅列です。"] * 20)
        self.assertLess(self.scorer.score(text), PIPELINE_QUALITY_MIN_SCORE)

    def test_score_within_range(self):
        for t in ["", "短", "普通の文章です。", "a" * 5000]:
            s = self.scorer.score(t)
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, QUALITY_MAX_SCORE)


class TestMinHashLSHLocal(unittest.TestCase):
    def setUp(self):
        self.lsh = MinHashLSHLocal()

    def test_first_insert_is_not_duplicate(self):
        self.assertFalse(self.lsh.check_duplicate_and_insert(1, "これは最初のドキュメントです。" * 5))

    def test_exact_duplicate_is_detected(self):
        text = "全く同じ内容のドキュメントは重複として検出されるべきです。" * 5
        self.assertFalse(self.lsh.check_duplicate_and_insert(1, text))
        self.assertTrue(self.lsh.check_duplicate_and_insert(2, text))

    def test_different_documents_are_kept(self):
        self.assertFalse(self.lsh.check_duplicate_and_insert(1, "機械学習は人工知能の一分野です。" * 5))
        self.assertFalse(self.lsh.check_duplicate_and_insert(2, "今日の東京の天気は快晴で気温は28度です。" * 5))


if __name__ == "__main__":
    unittest.main()
