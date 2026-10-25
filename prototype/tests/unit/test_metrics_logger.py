"""
tests/unit/test_metrics_logger.py

NEW-Q2（04_Quality_and_Standards S-3）: 学習メトリクスの JSONL 記録。
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.metrics import MetricsLogger, metrics_enabled


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestMetricsLogger(unittest.TestCase):
    def test_step_records_loss_lr_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            with MetricsLogger("sft", mode="TEST", out_dir=d, enabled=True) as m:
                m.log_step(epoch=0, step=1, loss=2.5, lr=3e-4, n_samples=8)
                m.log_step(epoch=0, step=2, loss=2.4, lr=3e-4, n_samples=8)
                path = m.path
            rows = _read_jsonl(path)
        self.assertEqual(len(rows), 2)
        r = rows[0]
        self.assertEqual(r["kind"], "step")
        self.assertEqual(r["run"], "sft")
        self.assertEqual(r["mode"], "TEST")
        self.assertEqual(r["loss"], 2.5)
        self.assertEqual(r["lr"], 3e-4)
        self.assertIn("ts", r)
        self.assertIn("run_id", r)
        # スループットは 2 ステップ目から算出できる（1 ステップ目は基準時刻のみ）
        self.assertIsNone(rows[0]["samples_per_sec"])
        self.assertIsNotNone(rows[1]["samples_per_sec"])

    def test_epoch_records(self):
        with tempfile.TemporaryDirectory() as d:
            with MetricsLogger("voice_codec", mode="TEST", out_dir=d, enabled=True) as m:
                m.log_epoch(epoch=1, loss=0.12, sem_loss=0.05)
                path = m.path
            rows = _read_jsonl(path)
        self.assertEqual(rows[0]["kind"], "epoch")
        self.assertEqual(rows[0]["loss"], 0.12)
        self.assertEqual(rows[0]["sem_loss"], 0.05)

    def test_disabled_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            m = MetricsLogger("sft", mode="TEST", out_dir=d, enabled=False)
            m.log_step(epoch=0, step=1, loss=1.0)         # 例外なく no-op
            m.log_epoch(epoch=1, loss=1.0)
            m.close()
            self.assertEqual(os.listdir(d), [])           # 何も書かれない

    def test_env_disable(self):
        with mock.patch.dict(os.environ, {"SWI_METRICS": "0"}):
            self.assertFalse(metrics_enabled())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SWI_METRICS", None)
            self.assertTrue(metrics_enabled())

    def test_appends_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            for _ in range(2):
                with MetricsLogger("sft", mode="TEST", out_dir=d, enabled=True) as m:
                    m.log_epoch(epoch=1, loss=1.0)
                    path = m.path
            rows = _read_jsonl(path)
        self.assertEqual(len(rows), 2)                     # run を跨いで追記される
        self.assertEqual(len({r["run_id"] for r in rows}), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
