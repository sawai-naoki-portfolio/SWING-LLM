"""
RLVR 検証器 (swi_llm/data/verifier.py) の単体テスト（§7.2-C）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data.verifier import (
    compute_reward, verify_math, _extract_final_answer, _looks_like_code,
)


class TestVerifyMath(unittest.TestCase):
    def test_fraction_vs_decimal(self):
        self.assertTrue(verify_math("1/2", "0.5"))

    def test_commutative_multiplication(self):
        self.assertTrue(verify_math("2*x", "x*2"))

    def test_implicit_multiplication(self):
        self.assertTrue(verify_math("E = m c^2", "E = m*c**2"))

    def test_power_notation(self):
        self.assertTrue(verify_math("7^2", "49"))

    def test_unequal(self):
        self.assertFalse(verify_math("3", "4"))

    def test_unparseable_returns_false(self):
        self.assertFalse(verify_math("東京", "大阪"))


class TestExtractAnswer(unittest.TestCase):
    def test_boxed(self):
        self.assertEqual(_extract_final_answer(r"reasoning \boxed{42}"), "42")

    def test_after_think(self):
        got = _extract_final_answer("<think>長い推論...</think>\n答え: 5")
        self.assertIn("5", got)

    def test_answer_label(self):
        self.assertEqual(_extract_final_answer("answer: 100"), "100")


class TestComputeReward(unittest.TestCase):
    def test_clean_final_answer_full(self):
        # 明確な最終回答（答え: 42）が期待値と一致 → 満点
        self.assertEqual(compute_reward("q", "答え: 42", "42"), 1.0)

    def test_boxed_final_answer_full(self):
        self.assertEqual(compute_reward("q", r"推論... \boxed{42}", "42"), 1.0)

    def test_buried_answer_not_full(self):
        # 本文に答えが紛れ込むだけでは満点にしない（報酬ハッキング対策）。部分点に留める。
        r = compute_reward("q", "the answer is 42 indeed", "42")
        self.assertLess(r, 1.0)
        self.assertGreater(r, 0.0)

    def test_sympy_equivalent(self):
        self.assertEqual(compute_reward("q", "</think>\n答え: 0.5", "1/2"), 1.0)

    def test_numeric_partial(self):
        self.assertEqual(compute_reward("q", "</think>\n答え: 3 and 4", "3 4"), 0.5)

    def test_wrong_answer(self):
        self.assertEqual(compute_reward("q", "totally unrelated text", "42"), 0.0)

    def test_length_penalty(self):
        self.assertEqual(compute_reward("q", "x" * 1001, "42"), -0.5)


class TestLooksLikeCode(unittest.TestCase):
    def test_detects_code(self):
        self.assertTrue(_looks_like_code("def f():\n    return 1"))
        self.assertTrue(_looks_like_code("print('hello')"))

    def test_plain_text_not_code(self):
        self.assertFalse(_looks_like_code("これは普通の文章です。"))


if __name__ == "__main__":
    unittest.main()
