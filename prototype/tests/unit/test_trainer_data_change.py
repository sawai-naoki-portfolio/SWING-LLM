"""
BaseTrainer.is_training_complete(data_size=...) の回帰テスト。

実際に踏んだ静かな失敗モード（2026-07-11, roadmap ステップ3）:
  SFT へ tool_call 教師データ 180 件を追加して再学習したところ、既存の完了済み
  チェックポイント（epoch 3/3）により `Training already complete` で即終了し、
  **新データが 1 件も学習されないまま exit 0 で成功扱い**になった。
  TEST/SMOKE は run.ps1 が既定 fresh のため露見しないが、PROD/MAX は既定 resume。

対策: チェックポイントに data_size を保存し、次回ロード時に件数が変わっていれば
「スキップされる。新データは学習されない。--fresh せよ」と WARNING で可視化する
（自動再学習はしない = PROD で長時間学習が意図せず走る方が危険）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.trainer_utils import BaseTrainer


class _FakeTrainer(BaseTrainer):
    """モデル・オプティマイザの構築（重い）を避け、判定ロジックだけを検証する。"""

    def __init__(self, start_epoch, epochs, loaded_data_size, main_ckpt_path="sft_checkpoint_TEST.pt"):
        import logging
        self.logger = logging.getLogger("test_trainer")
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.loaded_data_size = loaded_data_size
        self.data_size = None
        self.main_ckpt_path = main_ckpt_path


class TestIsTrainingComplete(unittest.TestCase):
    def test_incomplete_training_runs(self):
        t = _FakeTrainer(start_epoch=1, epochs=3, loaded_data_size=100)
        self.assertFalse(t.is_training_complete(data_size=280))

    def test_complete_with_same_data_is_silent(self):
        """データが変わっていなければ従来どおり静かに完了扱い（警告なし）。"""
        t = _FakeTrainer(start_epoch=3, epochs=3, loaded_data_size=100)
        with self.assertLogs("test_trainer", level="INFO") as cm:
            self.assertTrue(t.is_training_complete(data_size=100))
        self.assertFalse(any("WARNING" in line for line in cm.output))

    def test_complete_with_changed_data_warns(self):
        """データ件数が変わったのに完了扱いでスキップされる場合は WARNING で可視化する。"""
        t = _FakeTrainer(start_epoch=3, epochs=3, loaded_data_size=100)
        with self.assertLogs("test_trainer", level="WARNING") as cm:
            self.assertTrue(t.is_training_complete(data_size=280))
        joined = "\n".join(cm.output)
        self.assertIn("100 -> 280", joined)
        self.assertIn("--fresh", joined)            # 対処方法を必ず案内する
        self.assertIn("NOT be learned", joined)     # 何が起きるかを明言する

    def test_legacy_checkpoint_without_data_size_does_not_warn(self):
        """data_size を持たない旧 ckpt では比較せず、従来どおり動く（後方互換）。"""
        t = _FakeTrainer(start_epoch=3, epochs=3, loaded_data_size=None)
        with self.assertLogs("test_trainer", level="INFO") as cm:
            self.assertTrue(t.is_training_complete(data_size=280))
        self.assertFalse(any("WARNING" in line for line in cm.output))

    def test_data_size_is_recorded_for_subsequent_saves(self):
        """渡した data_size は trainer に記録され、以後の save_checkpoint に伝播できる。"""
        t = _FakeTrainer(start_epoch=0, epochs=3, loaded_data_size=None)
        t.is_training_complete(data_size=280)
        self.assertEqual(t.data_size, 280)

    def test_no_data_size_argument_keeps_legacy_behaviour(self):
        """引数なし呼び出し（旧シグネチャ）でも動く。"""
        t = _FakeTrainer(start_epoch=3, epochs=3, loaded_data_size=100)
        self.assertTrue(t.is_training_complete())
        self.assertIsNone(t.data_size)


class TestSaveCheckpointDataSize(unittest.TestCase):
    def test_data_size_roundtrip(self):
        """save_checkpoint(data_size=...) が ckpt に保存され、BaseTrainer が読み戻せる形になっている。"""
        import tempfile
        import torch
        from swi_llm.utils.trainer_utils import save_checkpoint

        model = torch.nn.Linear(2, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        scaler = torch.amp.GradScaler('cuda', enabled=False)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            save_checkpoint(path, model.state_dict(), opt, sched, scaler, 3, -1, 0.5, data_size=280)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["data_size"], 280)

            # data_size 未指定なら保存しない（旧 ckpt と同じ形 = 後方互換）
            path2 = os.path.join(d, "ckpt2.pt")
            save_checkpoint(path2, model.state_dict(), opt, sched, scaler, 3, -1, 0.5)
            self.assertNotIn("data_size", torch.load(path2, map_location="cpu", weights_only=False))


if __name__ == "__main__":
    unittest.main()
