"""
swi_llm/agent/executor.py

Deterministic Control Plane (Control Plane Sandbox):
ツール実行にタイムアウトを強制する実行境界。エージェントの暴走を防ぐ。

ツールの実装は tools.py、Python 実行サンドボックスは sandbox.py を参照 (REQ-01)。
"""
import queue
import threading

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    ERR_TOOL_TIMEOUT, ERR_TOOL_EXECUTION, ERR_TOOL_SILENT, TOOL_TIMEOUT_SEC,
)

logger = setup_logger("agent.executor")


class ToolExecutor:
    """
    Deterministic Control Plane (Control Plane Sandbox):
    Wraps tool executions to enforce timeouts and constraints.
    Prevents Agent runaway by providing a strict execution boundary.
    """
    @staticmethod
    def execute_with_timeout(func, kwargs, timeout_sec=TOOL_TIMEOUT_SEC):
        result_queue = queue.Queue()

        def wrapper():
            try:
                res = func(**kwargs)
                result_queue.put(("success", res))
            except Exception as e:
                result_queue.put(("error", e))

        thread = threading.Thread(target=wrapper)
        thread.daemon = True
        thread.start()
        thread.join(timeout_sec)

        if thread.is_alive():
            logger.warning(f"Tool '{getattr(func, '__name__', func)}' timed out after {timeout_sec}s. kwargs={kwargs}")
            return {"error": ERR_TOOL_TIMEOUT.format(timeout_sec=timeout_sec)}

        try:
            status, res = result_queue.get_nowait()
            if status == "error":
                logger.error(f"Tool '{getattr(func, '__name__', func)}' raised: {res}")
                return {"error": ERR_TOOL_EXECUTION.format(error=str(res))}
            logger.info(f"Tool '{getattr(func, '__name__', func)}' executed successfully.")
            return res
        except queue.Empty:
            logger.error(f"Tool '{getattr(func, '__name__', func)}' returned no result.")
            return {"error": ERR_TOOL_SILENT}
