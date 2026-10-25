"""
web/intent.py

ルールベースの意図検知とツール自動実行 (REQ-01 で web/routes/inference.py から分離)。

小規模モデルは自発的な <tool_call> 出力が不安定なため、プロンプトの
パターンマッチで計算・天気・検索の意図を検知し、該当ツールを
先回りで実行して結果をプロンプトに注入する。

結果はプロンプトへ注入するだけだと、モデルが弱いうちは回答に反映されず
ユーザーに届かない（計算は正しく実行されたのに答えが見えない）。そのため
`format_tool_result_for_display` で人が読める形に整え、ストリームにも直接流す。
"""
import re
from typing import NamedTuple, Optional

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    INTENT_CALC_TRIGGER_PATTERN, INTENT_NUMBER_PATTERN, INTENT_CALC_EXPR_PATTERN,
    INTENT_WEATHER_LOC_PATTERN, INTENT_KEYWORD_WEATHER, INTENT_KEYWORD_CALC,
    INTENT_KEYWORDS_SEARCH, INTENT_WEATHER_DEFAULT_LOCATION, INTENT_TOOL_DISPLAY_MAX_CHARS,
)
from swi_llm.agent.tools import get_weather, calculate, search_documents

logger = setup_logger("web_intent")

# 表示テキストに混ざるとクライアントのストリーム表示マークアップ（docs/03）と衝突するタグ。
# 検索結果の本文など外部由来のテキストが <tool_result> の枠を途中で閉じないよう除去する。
_STREAM_MARKUP = re.compile(r"</?(?:tool_result|tool_call|think)>")


class IntentToolOutcome(NamedTuple):
    tool: str            # 実行したツール名
    result: dict         # ツールの生の戻り値（表示用の整形に使う）
    tool_response: str   # プロンプトへ注入する ChatML ブロック


def detect_and_execute_tool_detailed(prompt_text, agent) -> Optional[IntentToolOutcome]:
    """プロンプトから意図を検知し、該当ツールを自動実行する。

    Args:
        prompt_text: ユーザー入力。
        agent: ツール実行結果を ChatML 形式へ変換するための AgentBase。

    Returns:
        実行したツール名・生の結果・ChatML ブロックの組。意図が検知されなければ None。
    """
    outcome = None

    if re.search(INTENT_CALC_TRIGGER_PATTERN, prompt_text) and re.search(INTENT_NUMBER_PATTERN, prompt_text):
        expr_match = re.search(INTENT_CALC_EXPR_PATTERN, prompt_text)
        if expr_match:
            expr = expr_match.group(1).replace('×', '*').replace('÷', '/').strip()
            if expr and re.search(r"\d", expr):
                try:
                    res = calculate(expression=expr)
                    outcome = IntentToolOutcome("calculate", res, agent._format_tool_response(res))
                except Exception:
                    logger.exception("Auto-invoked calculate tool failed.")
    elif INTENT_KEYWORD_WEATHER in prompt_text:
        loc_match = re.search(INTENT_WEATHER_LOC_PATTERN, prompt_text)
        loc = loc_match.group(1) if loc_match else INTENT_WEATHER_DEFAULT_LOCATION
        res = get_weather(location=loc)
        outcome = IntentToolOutcome("get_weather", res, agent._format_tool_response(res))
    elif any(kw in prompt_text for kw in INTENT_KEYWORDS_SEARCH):
        if INTENT_KEYWORD_WEATHER not in prompt_text and INTENT_KEYWORD_CALC not in prompt_text:
            res = search_documents(query=prompt_text)
            outcome = IntentToolOutcome("search_documents", res, agent._format_tool_response(res))

    # 19(1): ルールベース先回り実行の発火率 KPI。集計は
    #   発火数   = この行の grep 件数（tool= 別に内訳可能）
    #   分母     = web_inference の "Incoming Request" 行数
    # で行う（ペイロードは記録しない: REQ-09/SEC-7）。モデル自発の Function Calling
    # 成功率（FC-KPI, core/generation/stream.py）と併せて、ルール撤去（19(3)）の
    # 判断材料にする。
    if outcome is not None:
        logger.info(f"INTENT-KPI: preemptive_tool_fired tool={outcome.tool}")

    return outcome


def detect_and_execute_tool(prompt_text, agent):
    """後方互換 API。ツール実行結果の ChatML ブロック文字列（意図が無ければ None）を返す。"""
    outcome = detect_and_execute_tool_detailed(prompt_text, agent)
    return outcome.tool_response if outcome is not None else None


def _format_number(value):
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.10g}"
    return str(value)


def format_tool_result_for_display(tool, result, max_chars=INTENT_TOOL_DISPLAY_MAX_CHARS):
    """自動実行したツールの結果を、ユーザーに見せる 1〜数行のテキストへ整える。

    ツールの生の戻り値（dict）はモデル向けの形式で、そのまま見せると JSON や
    モデルへの指示文（「事実を断定せず…」）が露出するため、ツールごとに整形する。
    表示すべき内容が無ければ None。
    """
    if not isinstance(result, dict):
        return None

    if "error" in result:
        label = {"calculate": "計算", "get_weather": "天気の取得",
                 "search_documents": "文書検索"}.get(tool, "ツールの実行")
        text = f"{label}に失敗しました: {result['error']}"
    elif tool == "calculate" and "result" in result:
        text = f"計算結果: {result.get('expression', '')} = {_format_number(result['result'])}"
    elif tool == "get_weather" and result.get("condition"):
        text = f"天気: {result['condition']}"
    elif tool == "search_documents":
        # grounded=False の results はモデル向けの指示文なので、利用者向けの文言に置き換える。
        if result.get("grounded") and isinstance(result.get("results"), str):
            text = "検索結果:\n" + result["results"]
        else:
            text = "関連性の高い文書は見つかりませんでした。"
    else:
        return None

    text = _STREAM_MARKUP.sub("", text)
    # 空行はクライアントの markdown 描画で枠（HTML ブロック）を途中で終わらせるため 1 改行に詰める。
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text or None
