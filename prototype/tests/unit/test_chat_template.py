"""
ChatML テンプレート組み立て (utils/chat_template.py) の単体テスト。

出力はリファクタ前の f-string とバイト単位で同一であることが
チェックポイント互換性 (C-1) の前提のため、期待値は生文字列で固定する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.chat_template import (
    build_prompt, build_conversation_prompt, format_message, assistant_header,
    format_assistant_response, wrap_with_think, format_tool_call, format_tool_result,
    TOOL_CALL_PATTERN, ROLE_USER, ROLE_ASSISTANT,
)


class TestChatTemplate(unittest.TestCase):
    def test_build_prompt_plain(self):
        self.assertEqual(
            build_prompt("質問"),
            "<|im_start|>user\n質問<|im_end|>\n<|im_start|>assistant\n")

    def test_build_prompt_cot(self):
        self.assertEqual(
            build_prompt("質問", cot=True),
            "<|im_start|>user\n質問<|im_end|>\n<|im_start|>assistant\n<think>\n")

    def test_build_prompt_with_system(self):
        self.assertEqual(
            build_prompt("質問", system_text="SYS"),
            "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\n質問<|im_end|>\n<|im_start|>assistant\n")

    def test_format_message(self):
        self.assertEqual(format_message(ROLE_USER, "A"), "<|im_start|>user\nA<|im_end|>\n")

    def test_assistant_header(self):
        self.assertEqual(assistant_header(), "<|im_start|>assistant\n")

    def test_format_assistant_response(self):
        self.assertEqual(format_assistant_response("回答"), "回答<|im_end|>\n")

    def test_wrap_with_think(self):
        self.assertEqual(
            wrap_with_think("推論", "回答"),
            "<think>\n推論\n</think>\n回答")

    def test_format_tool_call(self):
        self.assertEqual(format_tool_call('{"a":1}'), '<tool_call>{"a":1}</tool_call>')

    def test_format_tool_result(self):
        self.assertEqual(
            format_tool_result("RES"),
            "<|im_start|>tool\nRES<|im_end|>\n<|im_start|>assistant\n")

    def test_tool_call_pattern_extracts_payload(self):
        text = 'before <tool_call>{"name": "calculate"}</tool_call> after'
        m = TOOL_CALL_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), '{"name": "calculate"}')

    def test_tool_call_pattern_multiline(self):
        text = '<tool_call>{\n"name": "x"\n}</tool_call>'
        m = TOOL_CALL_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertIn('"name": "x"', m.group(1))

    def test_tool_call_pattern_findall_returns_last(self):
        text = '<tool_call>{"a":1}</tool_call> mid <tool_call>{"b":2}</tool_call>'
        matches = TOOL_CALL_PATTERN.findall(text)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[-1], '{"b":2}')


class TestConversationPrompt(unittest.TestCase):
    """マルチターン会話履歴 (§12.3)。"""

    def test_empty_history_equals_build_prompt(self):
        # 履歴なしは build_prompt と完全一致（既存プロンプト互換の担保）
        self.assertEqual(
            build_conversation_prompt([], "質問"),
            build_prompt("質問"))
        self.assertEqual(
            build_conversation_prompt(None, "質問", system_text="SYS", cot=True),
            build_prompt("質問", system_text="SYS", cot=True))

    def test_history_turns_prepended(self):
        history = [
            {"role": "user", "content": "1回目"},
            {"role": "assistant", "content": "1回目の返答"},
        ]
        self.assertEqual(
            build_conversation_prompt(history, "2回目"),
            "<|im_start|>user\n1回目<|im_end|>\n"
            "<|im_start|>assistant\n1回目の返答<|im_end|>\n"
            "<|im_start|>user\n2回目<|im_end|>\n"
            "<|im_start|>assistant\n")

    def test_max_turns_keeps_most_recent(self):
        history = [
            {"role": "user", "content": "古い"},
            {"role": "assistant", "content": "古い返答"},
            {"role": "user", "content": "新しい"},
            {"role": "assistant", "content": "新しい返答"},
        ]
        out = build_conversation_prompt(history, "現在", max_turns=2)
        self.assertNotIn("古い", out)
        self.assertIn("新しい", out)
        self.assertIn("現在", out)

    def test_invalid_roles_and_empty_content_skipped(self):
        history = [
            {"role": "system", "content": "無視される"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "有効"},
        ]
        out = build_conversation_prompt(history, "Q")
        self.assertNotIn("無視される", out)
        self.assertIn("有効", out)

    def test_max_turns_zero_drops_all_history(self):
        history = [{"role": "user", "content": "過去"}]
        out = build_conversation_prompt(history, "現在", max_turns=0)
        self.assertNotIn("過去", out)
        self.assertEqual(out, build_prompt("現在"))


if __name__ == "__main__":
    unittest.main()
