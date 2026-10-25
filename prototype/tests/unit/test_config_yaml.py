"""
YAML 設定オーバーライド層 (問題3) の単体テスト（utils/config.py）。

検証:
  - _deep_merge がネスト dict を再帰マージし、非 dict は上書きする。
  - load_config_overrides が SWI_CONFIG 指定の YAML を読み、辞書を返す。
  - 不在/不正パスでは {} を返す（Python 既定値へフォールバック＝非破壊）。
  - 同梱の configs/default.yaml は全項目コメントアウトのため何も上書きしない（no-op）。
"""
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.config import _deep_merge, load_config_overrides


class TestDeepMerge(unittest.TestCase):
    def test_recursive_merge_and_override(self):
        base = {"a": 1, "nested": {"x": 1, "y": 2}}
        _deep_merge(base, {"a": 9, "nested": {"y": 20, "z": 30}})
        self.assertEqual(base, {"a": 9, "nested": {"x": 1, "y": 20, "z": 30}})


class TestLoadOverrides(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("SWI_CONFIG")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("SWI_CONFIG", None)
        else:
            os.environ["SWI_CONFIG"] = self._saved

    def test_loads_yaml_from_env(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "over.yaml")
            with open(p, "w", encoding="utf-8") as f:
                f.write("model_arch:\n  dim: 1234\ninference:\n  use_speculative_decoding: false\n")
            os.environ["SWI_CONFIG"] = p
            ov = load_config_overrides()
            self.assertEqual(ov["model_arch"]["dim"], 1234)
            self.assertFalse(ov["inference"]["use_speculative_decoding"])

    def test_missing_path_returns_empty(self):
        os.environ["SWI_CONFIG"] = os.path.join(tempfile.gettempdir(), "no_such_swi_config_12345.yaml")
        self.assertEqual(load_config_overrides(), {})

    def test_shipped_default_yaml_is_noop(self):
        # 同梱 configs/default.yaml は全コメントのため空辞書（何も上書きしない）
        os.environ.pop("SWI_CONFIG", None)
        proto = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_path = os.path.join(proto, "configs", "default.yaml")
        self.assertEqual(load_config_overrides(default_path), {})


if __name__ == "__main__":
    unittest.main()
