"""
L-13 / L-18: GRPO の系列構築と後始末の回帰テスト。

L-13: プロンプトが seq_len 以上だと、切り詰め後の target が全て -100 になり、学習信号ゼロの
系列が表示 loss を歪めていた（実測: logps=0・勾配 0・loss=-advantage）。
"""
import ast
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.training import grpo_train
from swi_llm.training.grpo_train import build_grpo_sequence


class TestBuildGrpoSequence(unittest.TestCase):
    def test_prompt_longer_than_seq_len_is_skipped(self):
        tokens = list(range(1, 21))                 # prompt 15 + completion 5
        self.assertIsNone(build_grpo_sequence(tokens, prompt_len=15, seq_len=10))
        self.assertIsNone(build_grpo_sequence(tokens, prompt_len=10, seq_len=10))  # ちょうど境界

    def test_masks_prompt_and_keeps_completion_targets(self):
        tokens = [10, 11, 12, 20, 21]               # prompt 3 + completion 2
        inp, tgt = build_grpo_sequence(tokens, prompt_len=3, seq_len=512)
        self.assertEqual(inp, [10, 11, 12, 20])
        self.assertEqual(tgt, [-100, -100, 20, 21])  # completion の 2 トークンだけが学習対象

    def test_truncation_keeps_remaining_completion(self):
        tokens = [10, 11, 12, 20, 21, 22, 23]
        inp, tgt = build_grpo_sequence(tokens, prompt_len=3, seq_len=5)
        self.assertEqual(tgt, [-100, -100, 20, 21])
        self.assertTrue(any(t != -100 for t in tgt))

    def test_every_returned_sequence_has_a_valid_target(self):
        for seq_len in range(1, 12):
            for prompt_len in range(0, 12):
                out = build_grpo_sequence(list(range(1, 11)), prompt_len, seq_len)
                if out is not None:
                    inp, tgt = out
                    self.assertEqual(len(inp), len(tgt))
                    self.assertTrue(any(t != -100 for t in tgt), (seq_len, prompt_len))

    def test_zero_prompt_len_masks_nothing(self):
        """prompt_len=0 で旧実装の `[:prompt_len-1]` = `[:-1]` が completion まで消さないこと。"""
        inp, tgt = build_grpo_sequence([1, 2, 3, 4], prompt_len=0, seq_len=512)
        self.assertEqual(tgt, [2, 3, 4])


class TestNoBareExcept(unittest.TestCase):
    def test_grpo_train_has_no_bare_except(self):
        """L-18: 裸の except は KeyboardInterrupt / SystemExit まで握り潰す。"""
        tree = ast.parse(open(grpo_train.__file__, encoding="utf-8").read())
        bare = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.type is None]
        self.assertEqual(bare, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
