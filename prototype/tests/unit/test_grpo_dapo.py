"""
GRPO/DAPO のフィルタリング条件 (問題4) の単体テスト。

DAPO は「グループ内の報酬分散がゼロ（全員正解 or 全員不正解）」のとき、有害な
ポリシー更新（KL のみが効く退化）を避けるため学習をスキップする。ここでは
その発火条件—compute_advantages が zero-variance グループで全ゼロを返し、
grpo_train のスキップ判定 `all(|advantage| < GRPO_DAPO_ADV_EPS)` が成立すること—を検証する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_llm.data.grpo import RewardVerifier
from swi_llm.training.grpo_train import GRPO_DAPO_ADV_EPS


class TestDAPOFiltering(unittest.TestCase):
    def setUp(self):
        self.verifier = RewardVerifier()

    def _is_skipped(self, rewards):
        adv = self.verifier.compute_advantages(rewards)
        return bool(torch.all(adv.abs() < GRPO_DAPO_ADV_EPS))

    def test_all_correct_group_is_skipped(self):
        # 全員正解 → 分散ゼロ → advantage 全ゼロ → DAPO スキップ対象
        self.assertTrue(self._is_skipped([1.0, 1.0, 1.0, 1.0]))

    def test_all_wrong_group_is_skipped(self):
        self.assertTrue(self._is_skipped([0.0, 0.0, 0.0, 0.0]))

    def test_mixed_group_is_not_skipped(self):
        # 正誤が混在 → 学習信号あり → スキップしない
        self.assertFalse(self._is_skipped([1.0, 0.0, 1.0, 0.0]))

    def test_advantages_zero_for_zero_variance(self):
        adv = self.verifier.compute_advantages([0.5, 0.5, 0.5])
        self.assertTrue(torch.allclose(adv, torch.zeros_like(adv)))


if __name__ == "__main__":
    unittest.main()
