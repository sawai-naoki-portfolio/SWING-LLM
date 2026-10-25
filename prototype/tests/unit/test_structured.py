"""
構造化出力 (swi_llm/core/generation/structured.py) の JSON 検証器の単体テスト（§6.3）。
モデル生成を伴わない純粋ロジックを対象とする。
"""
import json
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.core.generation.structured import is_valid_json_prefix, is_complete_json
from swi_llm.utils.constants import TOKENIZER_MODEL_PATH

_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_PROTOTYPE_DIR, TOKENIZER_MODEL_PATH)


class TestValidJSONPrefix(unittest.TestCase):
    def test_valid_prefixes(self):
        for s in ["", "{", '{"', '{"name"', '{"name":', '{"name": "', '{"name": "calc"',
                  '{"name": "calc",', "[", "[1", "[1,", "[1, 2", "[1, 2]", "tr", "true",
                  "-1", "-1.5", "3.14e", '{"a": [1, {"b": nu']:
            self.assertTrue(is_valid_json_prefix(s), f"should be valid prefix: {s!r}")

    def test_invalid_prefixes(self):
        for s in ["}", "{}}", "{\"a\": :", "[,", '{"a" "b"}', "{1: 2}", "trux",
                  "[1 2]", '{"a": 1 "b": 2}', "nulx"]:
            self.assertFalse(is_valid_json_prefix(s), f"should be invalid: {s!r}")

    def test_nested(self):
        self.assertTrue(is_valid_json_prefix('{"name": "calc", "arguments": {"x": 1, "y": [2, 3'))


class TestCompleteJSON(unittest.TestCase):
    def test_complete(self):
        for s in ["{}", '{"a": 1}', "[1, 2, 3]", "true", '"hello"', "123",
                  '{"name": "calc", "arguments": {"x": 1}}']:
            self.assertTrue(is_complete_json(s), f"should be complete: {s!r}")

    def test_incomplete(self):
        for s in ["{", '{"a":', "[1,", "tru"]:
            self.assertFalse(is_complete_json(s))


@unittest.skipUnless(os.path.exists(_MODEL_PATH), "tokenizer.model not trained yet")
class TestConstrainedToolCall(unittest.TestCase):
    """制約デコーディングによる tool_call 生成 (§6.3 / Q-10)。
    ランダム小型モデルでも常に構文的に有効な JSON になることを検証する。"""

    @classmethod
    def setUpClass(cls):
        import torch
        from swi_llm.core.models import MiniLLM
        from swi_llm.data.tokenizer import get_tokenizer
        cls.torch = torch
        cls.tok = get_tokenizer()
        cls.model = MiniLLM(vocab_size=cls.tok.vocab_size, dim=64, n_layers=2, n_heads=4,
                            hidden_dim=128, num_experts=4, top_k=2, max_seq_len=128)
        cls.model.eval()
        cls.prompt = "<|im_start|>user\n東京の天気は？<|im_end|>\n<|im_start|>assistant\n"

    def test_tool_call_is_wrapped_and_valid_json(self):
        from swi_llm.core.generation import generate_tool_call
        from swi_llm.utils.chat_template import TOOL_CALL_START, TOOL_CALL_END
        for seed in range(4):
            self.torch.manual_seed(seed)
            out = generate_tool_call(self.model, self.tok, self.prompt, max_new_tokens=32)
            self.assertTrue(out.startswith(TOOL_CALL_START), out)
            self.assertTrue(out.endswith(TOOL_CALL_END), out)
            inner = out[len(TOOL_CALL_START):-len(TOOL_CALL_END)]
            # 常に構文的に有効な JSON オブジェクト
            obj = json.loads(inner)
            self.assertIsInstance(obj, dict)

    def test_generate_json_prefix_forces_object(self):
        # prefix="{" は常に「JSON オブジェクトの有効な接頭辞」を返す契約
        # （完成すれば dict。小型モデルで未完なら接頭辞のまま = generate_tool_call が
        #  {} フォールバックで完全化する）。
        from swi_llm.core.generation import generate_json
        self.torch.manual_seed(0)
        out = generate_json(self.model, self.tok, self.prompt, max_new_tokens=32, prefix="{")
        self.assertTrue(out.startswith("{"))
        self.assertTrue(is_valid_json_prefix(out))
        if is_complete_json(out):
            self.assertIsInstance(json.loads(out), dict)


if __name__ == "__main__":
    unittest.main()
