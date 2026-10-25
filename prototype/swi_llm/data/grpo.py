import json
import os
import re
import torch
from torch.utils.data import Dataset

class GRPODataset(Dataset):
    def __init__(self, data_path=None, max_samples=None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, "math_sample.jsonl")
            
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    self.data.append(item)
                    if max_samples is not None and len(self.data) >= max_samples:
                        break

        if len(self.data) == 0:
            # 方針: 全モードで実データのみを使う。モック/ダミーのプロンプトへは
            # フォールバックせず、実データが無ければ例外で学習を止める。
            raise RuntimeError(
                f"GRPO 用の実プロンプトデータが見つかりません: {data_path}. "
                f"実データ (question/answer の JSONL) を配置してください。"
                f"（方針によりモック/ダミーへのフォールバックは無効化されています）")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["question"], item["answer"]


class RewardVerifier:
    """RLVR の検証器（§7.2-C）。数式等価 (SymPy)・コード実行・文字列一致を
    段階的に試みる。実装は swi_llm/data/verifier.py に委譲。"""
    def __init__(self):
        pass

    def verify(self, question: str, completion: str, expected_answer: str) -> float:
        """完了に対する報酬 (0.0〜1.0、無限生成には負値) を返す。"""
        from swi_llm.data.verifier import compute_reward
        return compute_reward(question, completion, expected_answer)

    def compute_advantages(self, rewards: list) -> torch.Tensor:
        """
        Computes group-relative advantages (GRPO).
        A_i = (r_i - mean(r)) / (std(r) + 1e-8)
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards_tensor) <= 1:
            return torch.zeros_like(rewards_tensor)
            
        mean = rewards_tensor.mean()
        std = rewards_tensor.std()
        
        if std < 1e-8:
            return torch.zeros_like(rewards_tensor)
            
        advantages = (rewards_tensor - mean) / (std + 1e-8)
        return advantages
