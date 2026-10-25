"""
L-17: チェックポイントの保存・読み込みで「黙ってスクラッチ開始」しないことの回帰テスト。

数週間かかる事前学習では、次のどれか 1 つで全成果を失いうる:
  - 保存中の停電・強制終了で ckpt が途中まで書かれて壊れる
  - 壊れた/構造の合わない ckpt を「読めなかったので最初から」と黙って学習し直し、次の保存で上書きする
  - 保存失敗（ディスク満杯）を握り潰し、何日も保存されないまま進む
  - 比率ベースの保存だけで、長い 1 エポックでは保存が数日おきになる
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils import trainer_utils
from swi_llm.utils.trainer_utils import (
    BaseTrainer, CheckpointLoadError, load_checkpoint, save_checkpoint,
)


def _parts(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return opt, sched, scaler


class TestAtomicSave(unittest.TestCase):
    def test_interrupted_save_keeps_previous_checkpoint(self):
        model = torch.nn.Linear(4, 4)
        opt, sched, scaler = _parts(model)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pretrain_checkpoint.pt")
            save_checkpoint(path, model.state_dict(), opt, sched, scaler, 0, 10, 1.0)
            good_weight = model.weight.detach().clone()

            def crash_midway(obj, f, *a, **kw):
                with open(f, "wb") as fh:
                    fh.write(b"partial")       # 途中まで書いたところで落ちた
                raise OSError("power loss")

            with torch.no_grad():
                model.weight.add_(1.0)
            with mock.patch.object(trainer_utils.torch, "save", side_effect=crash_midway):
                with self.assertRaises(OSError):      # 保存失敗は握り潰さない
                    save_checkpoint(path, model.state_dict(), opt, sched, scaler, 0, 20, 1.0)

            self.assertFalse(os.path.exists(path + ".tmp"))
            fresh = torch.nn.Linear(4, 4)
            _, start_step, _, ckpt = load_checkpoint(path, "cpu", fresh)
            self.assertIsNotNone(ckpt)
            self.assertEqual(start_step, 11)                    # 前回の保存点から再開できる
            self.assertTrue(torch.equal(fresh.weight, good_weight))


class TestLoudLoadFailure(unittest.TestCase):
    def test_missing_file_starts_from_scratch(self):
        with tempfile.TemporaryDirectory() as d:
            out = load_checkpoint(os.path.join(d, "none.pt"), "cpu", torch.nn.Linear(4, 4))
        self.assertEqual(out[:2], (0, 0))
        self.assertIsNone(out[3])

    def test_corrupt_file_raises_instead_of_scratch(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pretrain_checkpoint.pt")
            with open(path, "wb") as fh:
                fh.write(b"\x00garbage")
            with self.assertRaises(CheckpointLoadError) as cm:
                load_checkpoint(path, "cpu", torch.nn.Linear(4, 4))
        self.assertIn("--fresh", str(cm.exception))             # 対処法を案内する

    def test_no_matching_tensors_raises(self):
        """構造が変わって 1 つも一致しない ckpt を「ロード成功」扱いにしない。"""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pretrain_checkpoint.pt")
            torch.save({"model_state_dict": {"other.weight": torch.zeros(2, 2)}}, path)
            with self.assertRaises(CheckpointLoadError):
                load_checkpoint(path, "cpu", torch.nn.Linear(4, 4))

    def test_compiled_model_resumes_real_weights(self):
        """torch.compile 済みモデル（'_orig_mod.' 接頭辞）でも保存済みの重みが実際に戻る。"""
        src = torch.nn.Linear(4, 4)
        opt, sched, scaler = _parts(src)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pretrain_checkpoint.pt")
            save_checkpoint(path, src.state_dict(), opt, sched, scaler, 0, 5, 1.0)
            inner = torch.nn.Linear(4, 4)
            compiled = torch.compile(inner)
            load_checkpoint(path, "cpu", compiled)
        self.assertTrue(torch.equal(inner.weight, src.weight))


class _SaveTrainer(BaseTrainer):
    """step_and_save の保存判定だけを検証する（モデル構築は避ける）。"""

    def __init__(self, interval_seconds):
        import logging
        self.logger = logging.getLogger("test_ckpt_trainer")
        self.model = torch.nn.Linear(2, 2)
        self.optimizer, self.scheduler, self.scaler = _parts(self.model)
        self.grad_accum_steps = 1
        self.batch_size = 1
        self.best_val_loss = float("inf")
        self.data_size = None
        self.main_ckpt_path = "unused.pt"
        self.metrics = mock.Mock()
        self.save_interval_seconds = interval_seconds
        self._last_save_time = 0.0


class TestTimeBasedSave(unittest.TestCase):
    def _step(self, trainer, step):
        loss = trainer.model(torch.ones(1, 2)).sum()
        trainer.step_and_save(loss, step, dataloader_len=10_000, epoch=0, save_interval_steps=10_000)

    def test_saves_when_interval_elapsed_even_if_step_ratio_not_reached(self):
        t = _SaveTrainer(interval_seconds=1800)
        with mock.patch.object(trainer_utils, "save_checkpoint") as save, \
             mock.patch.object(trainer_utils.time, "monotonic", return_value=100.0):
            self._step(t, 0)
            save.assert_not_called()                     # 30 分経っていない
        with mock.patch.object(trainer_utils, "save_checkpoint") as save, \
             mock.patch.object(trainer_utils.time, "monotonic", return_value=1900.0):
            self._step(t, 1)
            save.assert_called_once()                    # 30 分経過 → 比率に関係なく保存
            self.assertEqual(save.call_args.args[6], 1)  # 保存した step（再開点）
        self.assertEqual(t._last_save_time, 1900.0)

    def test_disabled_when_zero(self):
        t = _SaveTrainer(interval_seconds=0)
        with mock.patch.object(trainer_utils, "save_checkpoint") as save, \
             mock.patch.object(trainer_utils.time, "monotonic", return_value=10 ** 9):
            self._step(t, 0)
            save.assert_not_called()

    def test_env_override(self):
        with mock.patch.dict(os.environ, {"SWI_CKPT_SAVE_MINUTES": "5"}):
            self.assertEqual(trainer_utils.ckpt_save_interval_seconds(), 300.0)
        with mock.patch.dict(os.environ, {"SWI_CKPT_SAVE_MINUTES": "abc"}):
            self.assertEqual(trainer_utils.ckpt_save_interval_seconds(), 30 * 60.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
