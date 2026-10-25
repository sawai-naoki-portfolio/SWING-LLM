"""
AST ホワイトリスト・サンドボックス (agent/sandbox.py) と
ツール実行境界 (agent/executor.py)・calculate (agent/tools.py) の単体テスト (REQ-20/21)。
torch 不要の純粋ロジックのみを対象とする。
"""
import os
import sys
import time
import types
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.agent.base import AgentBase
from swi_llm.agent.executor import ToolExecutor
from swi_llm.agent.sandbox import _validate_code_ast, execute_python
from swi_llm.agent.tools import calculate, TOOL_SCHEMAS
from swi_llm.utils.chat_template import format_tool_call
from swi_llm.utils.constants import (
    code_exec_tool_enabled, TOOL_EXECUTE_PYTHON,
    DOCKER_SANDBOX_USER, DOCKER_SANDBOX_PIDS_LIMIT, DOCKER_SANDBOX_NANO_CPUS,
)


class TestSandboxValidation(unittest.TestCase):
    """_validate_code_ast: None = 許可, 文字列 = 拒否理由"""

    # ---- 許可されるべきコード ----
    def test_allows_whitelisted_module(self):
        self.assertIsNone(_validate_code_ast("import math\nprint(math.pi)"))

    def test_allows_multiple_whitelisted_modules(self):
        self.assertIsNone(_validate_code_ast("import json, re\nfrom collections import Counter"))

    def test_allows_plain_computation(self):
        self.assertIsNone(_validate_code_ast("x = [i**2 for i in range(10)]\nprint(sum(x))"))

    # ---- import 制限 (REQ-20: ホワイトリスト方式) ----
    def test_rejects_os_import(self):
        self.assertIsNotNone(_validate_code_ast("import os"))

    def test_rejects_subprocess_import(self):
        self.assertIsNotNone(_validate_code_ast("import subprocess"))

    def test_rejects_from_import_outside_whitelist(self):
        self.assertIsNotNone(_validate_code_ast("from shutil import rmtree"))

    def test_rejects_unknown_third_party(self):
        # ホワイトリストに無いものは全部拒否（ブラックリスト方式との違い）
        self.assertIsNotNone(_validate_code_ast("import requests"))
        self.assertIsNotNone(_validate_code_ast("import socket"))
        self.assertIsNotNone(_validate_code_ast("import ctypes"))

    def test_rejects_submodule_of_forbidden_root(self):
        self.assertIsNotNone(_validate_code_ast("import os.path"))

    # ---- SEC-1: 文字列ベースの属性トラバースで脱出可能なモジュールを除外 ----
    def test_rejects_operator_import(self):
        # operator.attrgetter('__class__…') は dunder 静的検査を回避し RCE に悪用可能
        self.assertIsNotNone(_validate_code_ast("import operator"))

    def test_rejects_string_module_import(self):
        # string.Formatter().get_field('0.__class__…') も同様の属性トラバース経路
        self.assertIsNotNone(_validate_code_ast("import string"))

    def test_rejects_operator_attrgetter_escape_payload(self):
        # 実証済み脱出 PoC。operator が非許可になったことで import 段で拒否される
        payload = ("import operator\n"
                   "operator.attrgetter('__class__.__base__.__subclasses__')(())()")
        self.assertIsNotNone(_validate_code_ast(payload))

    # ---- dunder / 属性エスケープ (REQ-21) ----
    def test_rejects_dunder_class(self):
        self.assertIsNotNone(_validate_code_ast("().__class__"))

    def test_rejects_dunder_bases_chain(self):
        self.assertIsNotNone(_validate_code_ast("().__class__.__bases__[0].__subclasses__()"))

    def test_rejects_dunder_globals(self):
        self.assertIsNotNone(_validate_code_ast("(lambda: 0).__globals__"))

    def test_rejects_private_attribute(self):
        self.assertIsNotNone(_validate_code_ast("x._private"))

    def test_rejects_builtins_name(self):
        self.assertIsNotNone(_validate_code_ast("__builtins__"))

    # ---- 危険な組み込み関数 ----
    def test_rejects_eval_exec_compile(self):
        self.assertIsNotNone(_validate_code_ast("eval('1')"))
        self.assertIsNotNone(_validate_code_ast("exec('x=1')"))
        self.assertIsNotNone(_validate_code_ast("compile('x', 'f', 'exec')"))

    def test_rejects_getattr_family(self):
        self.assertIsNotNone(_validate_code_ast("getattr(1, 'real')"))
        self.assertIsNotNone(_validate_code_ast("setattr(x, 'a', 1)"))
        self.assertIsNotNone(_validate_code_ast("vars()"))
        self.assertIsNotNone(_validate_code_ast("globals()"))

    def test_rejects_open(self):
        self.assertIsNotNone(_validate_code_ast("open('secret.txt')"))

    def test_rejects_syntax_error(self):
        self.assertIsNotNone(_validate_code_ast("def broken(:"))


class TestCalculate(unittest.TestCase):
    """calculate: AST ベースの安全な式評価（eval 不使用）"""

    def test_basic_arithmetic(self):
        self.assertEqual(calculate("12345 * 6789")["result"], 12345 * 6789)
        self.assertEqual(calculate("2 + 2")["result"], 4)
        self.assertEqual(calculate("(1 + 2) * 3 - 4 / 2")["result"], 7.0)

    def test_unary_minus(self):
        self.assertEqual(calculate("-5 + 3")["result"], -2)

    def test_rejects_power_operator(self):
        self.assertIn("error", calculate("2 ** 100"))

    def test_rejects_too_long_expression(self):
        self.assertIn("error", calculate("1+" * 40 + "1"))

    def test_rejects_function_call(self):
        self.assertIn("error", calculate("__import__('os')"))

    def test_rejects_name_access(self):
        self.assertIn("error", calculate("x + 1"))


class _FakeContainer:
    def wait(self, timeout=None):
        return {"StatusCode": 0}

    def logs(self, stdout=True, stderr=False):
        return b""

    def stop(self):
        pass

    def remove(self, force=False):
        pass


class _FakeContainers:
    def __init__(self, captured):
        self._captured = captured

    def run(self, image, **kwargs):
        self._captured["image"] = image
        self._captured.update(kwargs)
        return _FakeContainer()


class _FakeClient:
    def __init__(self, captured):
        self.containers = _FakeContainers(captured)


class TestDockerSandboxHardening(unittest.TestCase):
    """SEC-5: Docker 実行が多層防御のハードニング引数付きで起動されることを検証する。"""

    def _run_capturing(self):
        captured = {}
        fake_docker = types.ModuleType("docker")
        fake_docker.from_env = lambda: _FakeClient(captured)
        with mock.patch.dict(sys.modules, {"docker": fake_docker}):
            result = execute_python("print(1)")
        return captured, result

    def test_docker_run_receives_hardening_options(self):
        captured, result = self._run_capturing()
        # run() まで到達し、成功応答が返っている（フォールバック経路ではない）
        self.assertTrue(captured, "containers.run was not called (Docker path not exercised)")
        self.assertEqual(result.get("status"), "success")
        # ネットワーク遮断・メモリ制限（既存）
        self.assertTrue(captured.get("network_disabled"))
        self.assertIn("mem_limit", captured)
        # SEC-5 で追加したハードニング
        self.assertEqual(captured.get("user"), DOCKER_SANDBOX_USER)
        self.assertEqual(list(captured.get("cap_drop", [])), ["ALL"])
        self.assertIn("no-new-privileges", captured.get("security_opt", []))
        self.assertEqual(captured.get("pids_limit"), DOCKER_SANDBOX_PIDS_LIMIT)
        self.assertEqual(captured.get("nano_cpus"), DOCKER_SANDBOX_NANO_CPUS)
        self.assertTrue(captured.get("read_only"))
        # read_only 下でも matplotlib 等が rw マウント /app にキャッシュを書けるよう環境変数を注入
        env = captured.get("environment", {})
        self.assertEqual(env.get("HOME"), "/app")
        self.assertEqual(env.get("MPLCONFIGDIR"), "/app")


class TestCodeExecToolGate(unittest.TestCase):
    """SEC-6: モデル発話由来の execute_python は既定オフ。ゲート無効時は実行されない。"""

    def setUp(self):
        self._saved = os.environ.get("SWI_ENABLE_CODE_EXEC_TOOL")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("SWI_ENABLE_CODE_EXEC_TOOL", None)
        else:
            os.environ["SWI_ENABLE_CODE_EXEC_TOOL"] = self._saved

    def test_gate_default_off(self):
        os.environ.pop("SWI_ENABLE_CODE_EXEC_TOOL", None)
        self.assertFalse(code_exec_tool_enabled())

    def test_gate_opt_in(self):
        for v in ("1", "true", "yes", "YES"):
            os.environ["SWI_ENABLE_CODE_EXEC_TOOL"] = v
            self.assertTrue(code_exec_tool_enabled(), f"value {v!r} should enable the tool")
        os.environ["SWI_ENABLE_CODE_EXEC_TOOL"] = "0"
        self.assertFalse(code_exec_tool_enabled())

    def _agent_like_serving(self):
        """web 経路（stream.py / inference.py）と同じゲート論理でツールを登録する。"""
        agent = AgentBase()
        if code_exec_tool_enabled():
            agent.register_tool(TOOL_EXECUTE_PYTHON, execute_python, TOOL_SCHEMAS[TOOL_EXECUTE_PYTHON])
        return agent

    def test_model_triggered_exec_rejected_when_gate_off(self):
        # プロンプトインジェクションで実行させようとする典型的な tool_call。
        os.environ.pop("SWI_ENABLE_CODE_EXEC_TOOL", None)
        agent = self._agent_like_serving()
        tool_call = format_tool_call(
            '{"name": "execute_python", "arguments": {"code": "import os"}}')
        resp = agent.process_llm_output(tool_call)
        # 未登録なので dispatch されず（コード実行されず）、not found を返す。
        self.assertFalse(agent.last_call_ok)
        self.assertIn("not found", resp)

    def test_model_triggered_exec_registered_when_gate_on(self):
        os.environ["SWI_ENABLE_CODE_EXEC_TOOL"] = "1"
        agent = self._agent_like_serving()
        self.assertIn(TOOL_EXECUTE_PYTHON, agent.tools)


class TestToolExecutor(unittest.TestCase):
    def test_success_result_passthrough(self):
        result = ToolExecutor.execute_with_timeout(lambda **kw: {"ok": 1}, {}, timeout_sec=5)
        self.assertEqual(result, {"ok": 1})

    def test_timeout_returns_error(self):
        def slow(**kw):
            time.sleep(3)
            return {"ok": 1}
        result = ToolExecutor.execute_with_timeout(slow, {}, timeout_sec=1)
        self.assertIn("error", result)
        self.assertIn("timed out", result["error"])

    def test_exception_returns_error(self):
        def boom(**kw):
            raise ValueError("boom")
        result = ToolExecutor.execute_with_timeout(boom, {}, timeout_sec=5)
        self.assertIn("error", result)
        self.assertIn("boom", result["error"])


if __name__ == "__main__":
    unittest.main()
