"""
agent/base.py

Function Calling を行うためのエージェント基盤クラス。
ツールの登録、LLM 出力のパース、関数の実行、フォーマット変換を担う。
"""
import json
import re

from .executor import ToolExecutor
from .tools import get_weather, calculate
from swi_llm.utils.logger import setup_logger
from swi_llm.utils.chat_template import (
    TOOL_CALL_PATTERN, TOOL_CALL_START, TOOL_CALL_END, format_tool_result,
)
from swi_llm.utils.constants import (
    TOOL_TIMEOUT_SEC, TOOL_RESPONSE_TRUNCATE_LEN,
    TOOL_CALCULATE, TOOL_GET_WEATHER,
)

logger = setup_logger("agent.base")


class AgentBase:
    """
    Function Callingを行うためのエージェント基盤クラス。
    ツールの登録、LLMの出力パース、関数の実行、フォーマット変換を担います。

    スレッド安全性 (V-12): このクラスは**スレッドセーフではない**。`last_call_ok` は
    呼び出しごとに書き換わるインスタンス状態のため、同一インスタンスを複数スレッド/
    セッションで共有して並行に呼ぶと成否が混線する。並行実行する場合はセッションごとに
    インスタンスを生成するか、`process_llm_output_ex` の戻り値 ok（呼び出しローカル）を
    使うこと。
    """
    def __init__(self):
        self.tools = {}
        # 直近の process_llm_output で「登録済みツールを実際に dispatch/実行できたか」。
        # 未登録ツール・不正 JSON（救済失敗）などの dispatch 失敗時は False のまま。
        # 注意 (V-12): 並行実行下ではセッション間で上書きされ競合する。新規コードは
        # process_llm_output_ex の戻り値 ok を使うこと（本属性は後方互換のために残す）。
        self.last_call_ok = False

    def register_tool(self, name, func, schema):
        """
        利用可能なツール（Python関数）と、そのJSON Schemaを登録します。
        """
        self.tools[name] = {
            "func": func,
            "schema": schema
        }

    def get_system_prompt_addition(self):
        """
        システムプロンプトに注入するための、利用可能なツール一覧と使用方法の文字列を生成します。
        """
        schemas = [info["schema"] for info in self.tools.values()]
        return (
            "Available tools:\n"
            f"{json.dumps(schemas, ensure_ascii=False, indent=2)}\n\n"
            "To use a tool, format your response EXACTLY as:\n"
            f"{TOOL_CALL_START}{{\"name\": \"tool_name\", \"arguments\": {{\"arg1\": \"val1\"}}}}{TOOL_CALL_END}"
        )

    def process_llm_output(self, llm_text):
        """
        LLMの出力テキストを受け取り、<tool_call>タグが含まれていれば
        そのツールを実行して、結果をLLMに返すフォーマットに変換します。

        後方互換 API。成否は `last_call_ok` 属性に残るが、並行実行下では競合するため
        (V-12)、成否が必要な呼び出し側は `process_llm_output_ex` を使うこと。
        """
        result, _ok = self.process_llm_output_ex(llm_text)
        return result

    def process_llm_output_ex(self, llm_text):
        """`process_llm_output` の本体。成否を戻り値で返す（V-12: 競合しない判定手段）。

        Returns:
            (formatted_result, call_ok):
                formatted_result: <tool_result> 形式の文字列（tool_call が無ければ None）。
                    未登録ツール・不正 JSON でもエラー内容の <tool_result> を返すため、
                    真偽値としては成功と区別できない点に注意。
                call_ok: 登録済みツールを実際に dispatch/実行できたか。`last_call_ok` と
                    同じ意味だが、インスタンス状態ではなく呼び出しローカルなので
                    並行実行でも他セッションに上書きされない。
        """
        match = TOOL_CALL_PATTERN.search(llm_text)
        if match:
            # このツール呼び出しを実際に dispatch できたかを毎回リセットしてから判定する。
            # 属性は後方互換のために更新し続けるが、判定は戻り値 ok（ローカル）で行う。
            self.last_call_ok = False
            tool_json_str = match.group(1).strip()
            tool_name = None
            try:
                tool_req = json.loads(tool_json_str)
                tool_name = tool_req.get("name")
                arguments = tool_req.get("arguments", {})

                # ツールが登録されているか確認して実行 (Control Plane Sandbox経由)
                if tool_name in self.tools:
                    schema = self.tools[tool_name]["schema"]

                    # Schema validation
                    try:
                        import jsonschema
                        jsonschema.validate(instance=arguments, schema=schema.get("parameters", {}))
                    except ImportError:
                        # Fallback simple validation
                        required = schema.get("parameters", {}).get("required", [])
                        for req in required:
                            if req not in arguments:
                                logger.warning(f"Tool call '{tool_name}' missing required argument: {req}")
                                return self._format_tool_response({"error": f"Missing required argument: {req}"}), False
                    except Exception as e:
                        logger.warning(f"Tool call '{tool_name}' failed schema validation: {e}")
                        return self._format_tool_response({"error": f"Schema validation error: {str(e)}"}), False

                    logger.info(f"Executing tool '{tool_name}' with arguments: {arguments}")
                    func = self.tools[tool_name]["func"]
                    result = ToolExecutor.execute_with_timeout(func, arguments, timeout_sec=TOOL_TIMEOUT_SEC)
                    # 登録済みツールを dispatch・実行できた（ツール自体がエラーを返しても
                    # 「呼び出しは成立」とみなす。ハルシネーション由来の未登録呼び出しと区別する）。
                    self.last_call_ok = True
                    return self._format_tool_response(result), True
                else:
                    logger.warning(f"Requested tool '{tool_name}' is not registered.")
                    return self._format_tool_response({"error": f"Tool '{tool_name}' not found."}), False

            except json.JSONDecodeError:
                # 150Mモデルのハルシネーション対策：JSONが壊れていても正規表現で計算式等を推測して救済する
                logger.warning(f"Malformed tool_call JSON. Attempting regex-based recovery: {tool_json_str[:200]!r}")
                target_str = tool_name if tool_name else tool_json_str  # 明確化: None チェックを分離
                if TOOL_CALCULATE in target_str:
                    expr_match = re.search(r'"expression"\s*:\s*"([^"]+)"', tool_json_str)
                    if not expr_match:
                        expr_match = re.search(r'([0-9\s\+\-\*\/\(\)]+)', tool_json_str)
                    if expr_match:
                        try:
                            result = calculate(expression=expr_match.group(1).strip())
                            self.last_call_ok = True   # 正規表現救済で calculate を実行できた
                            return self._format_tool_response(result), True
                        except Exception:
                            logger.exception("Regex-based calculate recovery failed.")
                elif "weather" in tool_json_str:
                    loc_match = re.search(r'"location"\s*:\s*"([^"]+)"', tool_json_str)
                    if loc_match:
                        try:
                            result = get_weather(location=loc_match.group(1).strip())
                            self.last_call_ok = True   # 正規表現救済で get_weather を実行できた
                            return self._format_tool_response(result), True
                        except Exception:
                            logger.exception("Regex-based get_weather recovery failed.")

                # モデルが壊れた tool_call を出すのは想定内の回復可能事象（呼び出し側が
                # リトライ/直接回答へフォールバックする）。継続不能ではないため WARNING (REQ-09)。
                logger.warning("Invalid JSON in tool_call and fallback parser failed.")
                return self._format_tool_response({"error": "Invalid JSON format in tool_call, and fallback parser failed."}), False

        # ツール呼び出しがなかった場合は None
        return None, False

    def _format_tool_response(self, result_obj):
        """
        ツール実行結果のPython辞書を、ブループリント指定の <|im_start|>tool フォーマットに変換します。
        """
        result_json = json.dumps(result_obj, ensure_ascii=False)
        if len(result_json) > TOOL_RESPONSE_TRUNCATE_LEN:
            result_json = result_json[:TOOL_RESPONSE_TRUNCATE_LEN] + "...(truncated)"
        return format_tool_result(result_json)
