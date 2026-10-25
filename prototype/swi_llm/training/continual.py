"""
swi_llm/training/continual.py

継続学習の支援コンポーネント（ブループリント §11）。

- FeedbackStore: 本番運用フィードバック（高評価/低評価・編集要求など）を JSONL に
  蓄積するフライホイールのシグナル収集層 (§11.2)。
- ReplayBuffer: 過去の高品質データを保持し、継続学習時に再混入して
  破局的忘却 (Catastrophic Forgetting) を防ぐ (§11.1)。

実際の LoRA 継続微調整は lora.py と組み合わせて行う。
"""
import os
import json
import random

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import MAX_FEEDBACK_FIELD_CHARS, MAX_FEEDBACK_FILE_BYTES

logger = setup_logger("continual")

_DEFAULT_FEEDBACK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "feedback.jsonl"
)


class FeedbackStore:
    """本番フィードバックを JSONL に追記・読み出しする (§11.2)。"""

    def __init__(self, path=_DEFAULT_FEEDBACK_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def add(self, query: str, response: str, rating: int, meta=None):
        """フィードバックを 1 件追記する。rating: 例) +1=good, -1=bad。

        SEC-4: ディスク枯渇 DoS を防ぐためファイル総サイズ上限を超えたら追記を拒否し、
        各フィールドは長さ上限で切り詰める（無検証な巨大入力の永続化を防ぐ）。
        """
        # ファイルサイズ上限（枯渇DoS対策）
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) >= MAX_FEEDBACK_FILE_BYTES:
                logger.warning(f"Feedback store size limit reached ({MAX_FEEDBACK_FILE_BYTES} bytes); rejecting append.")
                raise ValueError("feedback store is full")
        except OSError:
            pass
        # フィールド長の切り詰め（多層防御。呼び出し側でも検証するが最終防波堤として）
        q = (query or "")[:MAX_FEEDBACK_FIELD_CHARS]
        r = (response or "")[:MAX_FEEDBACK_FIELD_CHARS]
        record = {"query": q, "response": r, "rating": int(rating)}
        if meta:
            record["meta"] = meta
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Feedback stored (rating={rating}) for query={q[:40]!r}")
        return record

    def load(self, min_rating=None):
        """蓄積済みフィードバックを読み出す。min_rating 指定でフィルタ。"""
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if min_rating is None or rec.get("rating", 0) >= min_rating:
                    records.append(rec)
        return records

    def positive_examples(self):
        """高評価 (rating>0) のみを学習候補として返す（迎合的回答への過学習を避けるため
        監査は別途必要。§11.2 の安全上の注意）。"""
        return self.load(min_rating=1)


class ReplayBuffer:
    """破局的忘却対策のリプレイバッファ (§11.1)。

    過去の高品質サンプル（テキスト列）を保持し、継続学習バッチへ再混入する。
    """

    def __init__(self, capacity: int = 10000, seed: int = 42):
        self.capacity = capacity
        self.items = []
        self._rng = random.Random(seed)

    def add(self, text: str):
        if len(self.items) < self.capacity:
            self.items.append(text)
        else:
            # 容量超過時はランダム置換（reservoir 風）
            idx = self._rng.randint(0, len(self.items) - 1)
            self.items[idx] = text

    def extend(self, texts):
        for t in texts:
            self.add(t)

    def sample(self, k: int):
        if not self.items:
            return []
        k = min(k, len(self.items))
        return self._rng.sample(self.items, k)

    def mix(self, new_texts, replay_ratio: float = 0.25):
        """新規データに、その replay_ratio 割合ぶんの過去データを混ぜて返す (§11.1)。"""
        n_replay = int(len(new_texts) * replay_ratio)
        replayed = self.sample(n_replay)
        combined = list(new_texts) + replayed
        self._rng.shuffle(combined)
        return combined

    def __len__(self):
        return len(self.items)
