"""
AgentBase (agent/base.py) のツール呼び出しパース・実行・救済ロジックの単体テスト。
"""
import os
import sys
import threading
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.agent.base import AgentBase

ECHO_SCHEMA = {
    "name": "echo",
    "description": "echo tool",
    "parameters": {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    },
}


def echo_tool(msg):
    return {"echoed": msg}


class TestAgentBase(unittest.TestCase):
    def setUp(self):
        self.agent = AgentBase()
        self.agent.register_tool("echo", echo_tool, ECHO_SCHEMA)

    def test_no_tool_call_returns_none(self):
        self.assertIsNone(self.agent.process_llm_output("ただの返答です。"))

    def test_valid_tool_call_executes(self):
        out = self.agent.process_llm_output('<tool_call>{"name": "echo", "arguments": {"msg": "hi"}}</tool_call>')
        self.assertIsNotNone(out)
        self.assertIn('"echoed": "hi"', out)
        # ツール結果は ChatML の tool ブロック + assistant ヘッダで返る
        self.assertTrue(out.startswith("<|im_start|>tool\n"))
        self.assertTrue(out.endswith("<|im_start|>assistant\n"))
        # 登録済みツールを実際に dispatch できた → last_call_ok は True
        self.assertTrue(self.agent.last_call_ok)

    def test_unknown_tool_reports_error(self):
        out = self.agent.process_llm_output('<tool_call>{"name": "nope", "arguments": {}}</tool_call>')
        self.assertIsNotNone(out)
        self.assertIn("not found", out)
        # 未登録ツール（ハルシネーション）は dispatch 失敗 → last_call_ok は False
        self.assertFalse(self.agent.last_call_ok)

    def test_missing_required_argument_reports_error(self):
        out = self.agent.process_llm_output('<tool_call>{"name": "echo", "arguments": {}}</tool_call>')
        self.assertIsNotNone(out)
        self.assertIn("error", out.lower())
        # 必須引数欠落でツールは実行されていない → last_call_ok は False
        self.assertFalse(self.agent.last_call_ok)

    def test_malformed_json_calculate_recovery(self):
        # JSON が壊れていても calculate は式を正規表現で救済する
        out = self.agent.process_llm_output('<tool_call>{"name": "calculate", "expression": "2 + 3"</tool_call>')
        self.assertIsNotNone(out)
        self.assertIn("5", out)
        # 正規表現救済で calculate を実行できた → last_call_ok は True
        self.assertTrue(self.agent.last_call_ok)

    def test_malformed_json_without_recovery_reports_error(self):
        out = self.agent.process_llm_output("<tool_call>garbage-that-matches-nothing</tool_call>")
        self.assertIsNotNone(out)
        self.assertIn("error", out.lower())
        # 救済できない不正 JSON は dispatch 失敗 → last_call_ok は False
        self.assertFalse(self.agent.last_call_ok)

    def test_last_call_ok_resets_between_calls(self):
        # 成功呼び出しで True になった後、失敗呼び出しで False に戻る（毎回リセットされる）
        self.agent.process_llm_output('<tool_call>{"name": "echo", "arguments": {"msg": "hi"}}</tool_call>')
        self.assertTrue(self.agent.last_call_ok)
        self.agent.process_llm_output('<tool_call>{"name": "nope", "arguments": {}}</tool_call>')
        self.assertFalse(self.agent.last_call_ok)

    def test_truncates_long_result(self):
        long_msg = "x" * 500
        out = self.agent.process_llm_output(
            f'<tool_call>{{"name": "echo", "arguments": {{"msg": "{long_msg}"}}}}</tool_call>')
        self.assertIsNotNone(out)
        self.assertIn("...(truncated)", out)


class TestProcessLlmOutputEx(unittest.TestCase):
    """V-12: 並行実行下でも競合しない成否判定（戻り値 ok）の回帰テスト。"""

    def setUp(self):
        self.agent = AgentBase()
        self.agent.register_tool("echo", echo_tool, ECHO_SCHEMA)

    def test_ex_returns_result_and_ok_on_success(self):
        out, ok = self.agent.process_llm_output_ex(
            '<tool_call>{"name": "echo", "arguments": {"msg": "hi"}}</tool_call>')
        self.assertIn('"echoed": "hi"', out)
        self.assertTrue(ok)

    def test_ex_unknown_tool_returns_error_text_with_ok_false(self):
        out, ok = self.agent.process_llm_output_ex(
            '<tool_call>{"name": "nope", "arguments": {}}</tool_call>')
        # エラーでも <tool_result> 文字列は返る（真偽値では成否を判定できない）が ok は False
        self.assertIn("not found", out)
        self.assertFalse(ok)

    def test_ex_no_tool_call_returns_none_false(self):
        out, ok = self.agent.process_llm_output_ex("ただの返答です。")
        self.assertIsNone(out)
        self.assertFalse(ok)

    def test_ok_is_isolated_under_concurrent_calls(self):
        """V-12 の競合再現: 共有インスタンスへの並行呼び出しで last_call_ok 属性は
        他セッションに上書きされるが、戻り値 ok は自分の呼び出しの成否を保持する。"""
        release = threading.Event()
        started = threading.Event()

        def slow_echo(msg):
            started.set()
            release.wait(2)   # 失敗側の呼び出しが完了するまでツール実行中で待機
            return {"echoed": msg}

        agent = AgentBase()
        agent.register_tool("echo", slow_echo, ECHO_SCHEMA)
        results = {}

        t = threading.Thread(target=lambda: results.__setitem__(
            "ok_call", agent.process_llm_output_ex(
                '<tool_call>{"name": "echo", "arguments": {"msg": "hi"}}</tool_call>')))
        t.start()
        self.assertTrue(started.wait(2), "成功側のツール実行が開始されていない")
        # 成功側がツール実行中の間に、同じ共有インスタンスで失敗呼び出しが完了する
        results["fail_call"] = agent.process_llm_output_ex(
            '<tool_call>{"name": "nope", "arguments": {}}</tool_call>')
        release.set()
        t.join(timeout=10)

        self.assertTrue(results["ok_call"][1], "成功側の ok が True でない")
        self.assertFalse(results["fail_call"][1], "失敗側の ok が他セッションに上書きされた")
        # 共有属性は最後に完了した成功呼び出しの True になる。失敗セッションがこれを
        # 読むと成功と誤認する — これが V-12 の競合であり、属性を読まない理由。
        self.assertTrue(agent.last_call_ok)


if __name__ == "__main__":
    unittest.main()
