"""
PRM モンテカルロ / コントラスト・ステップ報酬 (swi_llm/data/prm_mc.py) の単体テスト（§8.2 / Q-07）。
モデル生成を伴わない純粋ロジック（mc_step_rewards / contrastive_step_rewards）を対象とする。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data.prm_mc import mc_step_rewards, contrastive_step_rewards


class TestMCStepRewards(unittest.TestCase):
    def test_fraction_correct(self):
        # ステップ0: 4件中2件正解 -> 0.5、ステップ1: 全正解 -> 1.0
        outcomes = {0: [True, False, True, False], 1: [True, True, True, True]}
        rewards = mc_step_rewards(lambda k: outcomes[k], n_steps=2)
        self.assertEqual(rewards, [0.5, 1.0])

    def test_all_wrong_is_zero(self):
        rewards = mc_step_rewards(lambda k: [False, False], n_steps=3)
        self.assertEqual(rewards, [0.0, 0.0, 0.0])

    def test_empty_rollout_is_zero(self):
        rewards = mc_step_rewards(lambda k: [], n_steps=2)
        self.assertEqual(rewards, [0.0, 0.0])

    def test_rewards_in_unit_range(self):
        import random
        rng = random.Random(0)
        fn = lambda k: [rng.random() > 0.5 for _ in range(8)]
        for r in mc_step_rewards(fn, n_steps=5):
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)


class TestContrastiveStepRewards(unittest.TestCase):
    def test_correct_all_ones(self):
        self.assertEqual(contrastive_step_rewards(4, correct=True), [1.0, 1.0, 1.0, 1.0])

    def test_incorrect_decays(self):
        r = contrastive_step_rewards(5, correct=False, reward_start=0.5, reward_end=0.0)
        # 単調非増加、先頭=start、末尾=end
        self.assertAlmostEqual(r[0], 0.5)
        self.assertAlmostEqual(r[-1], 0.0)
        self.assertTrue(all(r[i] >= r[i + 1] for i in range(len(r) - 1)))

    def test_single_incorrect_step_is_end(self):
        self.assertEqual(contrastive_step_rewards(1, correct=False, reward_end=0.0), [0.0])

    def test_empty(self):
        self.assertEqual(contrastive_step_rewards(0, correct=True), [])

    def test_contrastive_signal_distinguishes(self):
        # 正解軌跡の平均 > 不正解軌跡の平均（PRM に識別信号を与える）
        pos = contrastive_step_rewards(4, correct=True)
        neg = contrastive_step_rewards(4, correct=False)
        self.assertGreater(sum(pos) / len(pos), sum(neg) / len(neg))


if __name__ == "__main__":
    unittest.main()
