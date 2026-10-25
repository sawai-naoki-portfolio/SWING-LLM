"""
swi_llm/data/prm_mc.py

PRM ステップ報酬のモンテカルロ推定（§8.2 / Q-07, MATH-Shepherd 方式）。

従来 PRM は「正解軌跡の全ステップ reward=1.0 固定」だった。本モジュールは
各推論ステップ prefix から続きを複数生成し、外部検証器で正答率を測ることで
「そのステップから正解に到達する確率」＝ステップ報酬 (0..1) を推定する。
誤りが混入した段階を低く評価できるようになり、PRM の識別能力が向上する。

- mc_step_rewards(): MC 推定の中核（純粋ロジック・モデル非依存でテスト可能）。
- estimate_step_rewards_mc(): モデル + 検証器で実 MC 推定（policy モデルが必要）。
- contrastive_step_rewards(): モデルなしの軽量コントラスト・ラベル
  （正解=1.0 / 不正解=後半へ減衰）。uniform 1.0 固定からの脱却。
"""
from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    PRM_MC_ROLLOUTS, PRM_REJECTED_REWARD_START, PRM_REJECTED_REWARD_END,
)

logger = setup_logger("data.prm_mc")


def mc_step_rewards(rollout_correctness_fn, n_steps):
    """モンテカルロ・ステップ報酬推定の中核（純粋ロジック）。

    Args:
        rollout_correctness_fn: ステップ index k(0始まり) を受け取り、その prefix から
            生成した各ロールアウトの正誤（bool の反復可能）を返す関数。
        n_steps: ステップ数。

    Returns:
        各ステップの MC 報酬 (0.0..1.0) のリスト。ロールアウトが空なら 0.0。
    """
    rewards = []
    for k in range(n_steps):
        outcomes = list(rollout_correctness_fn(k))
        if outcomes:
            rewards.append(sum(1.0 for o in outcomes if o) / len(outcomes))
        else:
            rewards.append(0.0)
    return rewards


def contrastive_step_rewards(n_steps, correct,
                             reward_start=PRM_REJECTED_REWARD_START,
                             reward_end=PRM_REJECTED_REWARD_END):
    """モデルなしのコントラスト・ラベル（MC 推定の軽量代替）。

    - 正解軌跡: 全ステップ 1.0（正解に至る prefix は正解確率 ~1、MC と整合）。
    - 不正解軌跡: reward_start → reward_end へ線形減衰（誤りは後半で顕在化しやすい
      という事前分布）。uniform 1.0 固定を脱し、良い/悪いステップを区別する信号を与える。
    """
    if n_steps <= 0:
        return []
    if correct:
        return [1.0] * n_steps
    if n_steps == 1:
        return [reward_end]
    return [reward_start + (reward_end - reward_start) * (k / (n_steps - 1))
            for k in range(n_steps)]


def estimate_step_rewards_mc(model, tokenizer, prompt, steps, gold_answer,
                             n_rollouts=PRM_MC_ROLLOUTS, max_new_tokens=64, question=""):
    """モデル + 検証器でステップ報酬を MC 推定する（MATH-Shepherd 方式）。

    steps[:k+1] を prefix として prompt に連結し、そこから n_rollouts 個の続きを
    生成、compute_reward(gold) が正なら correct と数え、その割合を報酬にする。
    policy モデルが利用できる環境（SFT 後）でのみ使用する。
    """
    from swi_llm.core.generation import generate
    from swi_llm.data.verifier import compute_reward

    tm = model.text_model if hasattr(model, "text_model") else model

    def rollout_correctness(k):
        prefix = prompt + "\n".join(steps[:k + 1]) + "\n"
        outcomes = []
        for _ in range(n_rollouts):
            full = generate(tm, tokenizer, prefix, max_new_tokens)
            cont = full[len(prefix):] if full.startswith(prefix) else full
            answer_text = "\n".join(steps[:k + 1]) + "\n" + cont
            outcomes.append(compute_reward(question, answer_text, gold_answer) >= 1.0)
        return outcomes

    return mc_step_rewards(rollout_correctness, len(steps))
