"""
スケール/効率化ヘルパー (問題5) の単体テスト（utils/trainer_utils.py）。

FSDP ラップと torch.compile は「分散未初期化・フラグ未設定（=通常運用・テスト）では
素通し（従来挙動を完全維持）」であることを検証する。torch.compile 済み state_dict の
'_orig_mod.' 接頭辞除去も検証する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch.nn as nn
from swi_llm.utils.trainer_utils import maybe_wrap_fsdp, maybe_compile, _strip_compiled_prefix


class TestScalingHelpers(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("SWI_USE_TORCH_COMPILE")
        os.environ.pop("SWI_USE_TORCH_COMPILE", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["SWI_USE_TORCH_COMPILE"] = self._saved

    def test_fsdp_noop_when_not_distributed(self):
        # torch.distributed 未初期化（単一プロセス）では wrap せず同一オブジェクトを返す
        m = nn.Linear(4, 4)
        self.assertIs(maybe_wrap_fsdp(m), m)

    def test_compile_noop_when_flag_off(self):
        m = nn.Linear(4, 4)
        self.assertIs(maybe_compile(m), m)

    def test_strip_compiled_prefix(self):
        sd = {"_orig_mod.layers.0.w": 1, "_orig_mod.norm.weight": 2, "plain": 3}
        out = _strip_compiled_prefix(sd)
        self.assertEqual(set(out.keys()), {"layers.0.w", "norm.weight", "plain"})

    def test_strip_compiled_prefix_noop_without_prefix(self):
        sd = {"layers.0.w": 1, "norm.weight": 2}
        self.assertEqual(_strip_compiled_prefix(sd), sd)


if __name__ == "__main__":
    unittest.main()
