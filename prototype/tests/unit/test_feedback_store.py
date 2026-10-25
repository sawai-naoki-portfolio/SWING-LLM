"""
SEC-4: FeedbackStore（継続学習フィードバックの永続化層）の入力検証・サイズ上限テスト。

無認証 /api/feedback からのデータ汚染・ディスク枯渇を防ぐため、FeedbackStore.add が
フィールド長を切り詰め、ファイル総サイズ上限を超えたら追記を拒否することを検証する。
"""
import os
import sys
import json
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.training.continual import FeedbackStore
from swi_llm.utils.constants import MAX_FEEDBACK_FIELD_CHARS, MAX_FEEDBACK_FILE_BYTES


class TestFeedbackStoreSecurity(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "feedback.jsonl")
        self.store = FeedbackStore(path=self.path)

    def test_stores_valid_record(self):
        rec = self.store.add("質問", "回答", 1)
        self.assertEqual(rec["rating"], 1)
        self.assertEqual(rec["query"], "質問")
        loaded = self.store.load()
        self.assertEqual(len(loaded), 1)

    def test_truncates_oversized_fields(self):
        rec = self.store.add("q" * (MAX_FEEDBACK_FIELD_CHARS + 5000),
                             "r" * (MAX_FEEDBACK_FIELD_CHARS + 5000), -1)
        self.assertEqual(len(rec["query"]), MAX_FEEDBACK_FIELD_CHARS)
        self.assertEqual(len(rec["response"]), MAX_FEEDBACK_FIELD_CHARS)

    def test_rejects_when_file_full(self):
        # ファイルを上限まで膨らませると追記が拒否される（ディスク枯渇DoS防止）
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("x" * MAX_FEEDBACK_FILE_BYTES)
        with self.assertRaises(ValueError):
            self.store.add("q", "r", 1)

    def test_record_is_valid_json_line(self):
        self.store.add("あ\"'", "い\n改行", 1)
        with open(self.path, encoding="utf-8") as f:
            line = f.readline()
        obj = json.loads(line)   # 破損せず 1 行 = 1 JSON
        self.assertIn("query", obj)
        self.assertIn("rating", obj)


if __name__ == "__main__":
    unittest.main()
