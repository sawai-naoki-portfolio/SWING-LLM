"""
swi_voice/duplex/tool_bridge.py の単体テスト（未実装解消: つなぎ発話 §10.3）。
"""
import asyncio
import os
import sys
import threading
import time
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.agent.base import AgentBase
from swi_llm.agent.tools import calculate, TOOL_SCHEMAS
from swi_llm.utils.constants import TOOL_CALCULATE
from swi_voice.duplex.tool_bridge import handle_tool_call_with_filler, has_pending_tool_call


def _make_agent(func=calculate):
    agent = AgentBase()
    agent.register_tool(TOOL_CALCULATE, func, TOOL_SCHEMAS[TOOL_CALCULATE])
    return agent


def _tool_call_text(expr="2 + 2"):
    return f'<tool_call>{{"name": "{TOOL_CALCULATE}", "arguments": {{"expression": "{expr}"}}}}</tool_call>'


class TestHasPendingToolCall(unittest.TestCase):
    def test_detects_tool_call(self):
        self.assertTrue(has_pending_tool_call(_tool_call_text()))

    def test_plain_text_has_none(self):
        self.assertFalse(has_pending_tool_call("こんにちは、元気ですか？"))

    def test_empty_or_none(self):
        self.assertFalse(has_pending_tool_call(""))
        self.assertFalse(has_pending_tool_call(None))


class TestHandleToolCallWithFiller(unittest.TestCase):
    def test_no_tool_call_skips_filler(self):
        agent = _make_agent()
        spoken = []

        async def speak(text):
            spoken.append(text)

        result, filler_spoken = asyncio.run(
            handle_tool_call_with_filler(agent, "plain response, no tool", speak, ["wait"]))
        self.assertIsNone(result)
        self.assertFalse(filler_spoken)
        self.assertEqual(spoken, [])

    def test_tool_call_triggers_filler_and_returns_result(self):
        agent = _make_agent()
        spoken = []

        async def speak(text):
            spoken.append(text)

        result, filler_spoken = asyncio.run(
            handle_tool_call_with_filler(agent, _tool_call_text("2 + 2"), speak, ["少々お待ちください"]))
        self.assertTrue(filler_spoken)
        self.assertEqual(spoken, ["少々お待ちください"])
        self.assertIn("4", result)          # calculate(2+2)=4 がツール結果に含まれる

    def test_empty_filler_phrases_still_executes_tool(self):
        agent = _make_agent()
        result, filler_spoken = asyncio.run(
            handle_tool_call_with_filler(agent, _tool_call_text("3 + 3"), None, []))
        self.assertFalse(filler_spoken)
        self.assertIn("6", result)

    def test_filler_runs_concurrently_with_tool_execution(self):
        order = []

        async def speak(text):
            order.append("filler_start")
            await asyncio.sleep(0.01)
            order.append("filler_done")

        def slow_calculate(expression):
            order.append("tool_start")
            time.sleep(0.03)
            order.append("tool_done")
            return calculate(expression)

        agent = _make_agent(func=slow_calculate)
        result, filler_spoken = asyncio.run(
            handle_tool_call_with_filler(agent, _tool_call_text("1 + 1"), speak, ["wait"]))
        self.assertTrue(filler_spoken)
        self.assertIn("2", result)
        # つなぎ発話はツール完了を待たずに開始されている(=並行実行の証拠)
        self.assertLess(order.index("filler_start"), order.index("tool_done"))

    def test_tool_exception_is_handled_without_raising(self):
        def broken_tool(expression):
            raise RuntimeError("boom")

        agent = _make_agent(func=broken_tool)
        spoken = []

        async def speak(text):
            spoken.append(text)

        result, filler_spoken = asyncio.run(
            handle_tool_call_with_filler(agent, _tool_call_text("1 + 1"), speak, ["wait"]))
        # ToolExecutor 自体が例外を握りつぶしてエラーJSONを返すため、result は None にならない。
        # 呼び出し側（tool_bridge）がクラッシュしないことのみをここでは確認する。
        self.assertTrue(filler_spoken)
        self.assertIsNotNone(result)

    def test_speak_fn_none_with_default_phrases_still_executes_tool(self):
        """V-13: 「フレーズは既定のまま speak_fn=None」という素直な呼び出しが
        TypeError にならず、つなぎ無しでツールを実行する。"""
        agent = _make_agent()
        result, filler_spoken = asyncio.run(
            handle_tool_call_with_filler(agent, _tool_call_text("3 + 4"), None))
        self.assertFalse(filler_spoken)
        self.assertIn("7", result)

    def test_filler_failure_does_not_discard_tool_result(self):
        """V-11: つなぎ発話の再生失敗（音声デバイス切断等）で、成功したツール結果を失わない。"""
        async def broken_speak(text):
            raise RuntimeError("audio device disconnected")

        agent = _make_agent()
        with self.assertLogs("voice.tool_bridge", level="ERROR") as cm:
            result, filler_spoken = asyncio.run(
                handle_tool_call_with_filler(agent, _tool_call_text("2 + 2"), broken_speak, ["wait"]))
        self.assertTrue(filler_spoken)
        self.assertIn("4", result)   # ツール結果は保持される（旧実装は例外で失われた）
        self.assertTrue(any("filler speech failed" in m for m in cm.output))

    def test_cancellation_propagates_and_stops_filler(self):
        """V-11: 上位キャンセル（バージイン相当）は握り潰されず伝播し、
        つなぎ発話タスクもキャンセルされる（タスクリークしない）。"""
        cancelled = {"filler": False}

        async def speak(text):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled["filler"] = True
                raise

        def slow_tool(expression):
            time.sleep(0.2)
            return calculate(expression)

        agent = _make_agent(func=slow_tool)

        async def scenario():
            task = asyncio.create_task(
                handle_tool_call_with_filler(agent, _tool_call_text("1 + 1"), speak, ["wait"]))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return True
            return False

        self.assertTrue(asyncio.run(scenario()), "CancelledError が伝播していない")
        self.assertTrue(cancelled["filler"], "つなぎ発話タスクがキャンセルされていない")

    def test_timeout_returns_none_and_warns_tool_still_running(self):
        """V-10: timeout は待機の打ち切りであり、ツール実行は中断されない。
        その事実が WARNING でログに残ることを確認する。"""
        release = threading.Event()

        def slow_tool(expression):
            release.wait(1)
            return calculate(expression)

        agent = _make_agent(func=slow_tool)

        async def speak(text):
            pass

        async def scenario():
            try:
                return await handle_tool_call_with_filler(
                    agent, _tool_call_text("1 + 1"), speak, ["wait"], timeout=0.05)
            finally:
                release.set()   # バックグラウンドスレッドを速やかに終わらせる（テスト高速化）

        with self.assertLogs("voice.tool_bridge", level="WARNING") as cm:
            result, filler_spoken = asyncio.run(scenario())
        self.assertIsNone(result)
        self.assertTrue(filler_spoken)
        self.assertTrue(any("keeps running in a background thread" in m for m in cm.output))


if __name__ == "__main__":
    unittest.main()
