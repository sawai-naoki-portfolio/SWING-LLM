"""
utils/chat_template.py

ChatML 特殊トークンとプロンプト組み立ての「単一情報源」モジュール (REQ-02 / REQ-04)。

`<|im_start|>` / `<think>` / `<tool_call>` 等のテンプレート文字列は、
本モジュール以外の Python コードにリテラルとして書かないこと。
プロンプトの組み立て・分解は必ずここの定数／関数を経由する。

注意: 各ビルダー関数の出力はリファクタ前の f-string と「バイト単位で同一」に
なるよう設計されている。変更すると学習データ・推論プロンプトの互換性が壊れる。
"""
import re

# =====================================================================
# ChatML 特殊トークン
# =====================================================================
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

# 思考 (Chain-of-Thought) ブロック
THINK_START = "<think>"
THINK_END = "</think>"

# ツール呼び出しブロック
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

# 外部データセット (DataPilot/Function-Calling-Japanese-Dataset) の旧タグ。
# 学習データ読み込み時に TOOL_CALL_* へ変換するためにのみ使用する。
LEGACY_FUNCTIONCALL_START = "<functioncall>"
LEGACY_FUNCTIONCALL_END = "</functioncall>"

# LLM 出力から <tool_call>...</tool_call> を抽出する共通パターン
TOOL_CALL_PATTERN = re.compile(
    re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END),
    re.DOTALL,
)


# =====================================================================
# プロンプト組み立てビルダー
# =====================================================================
def format_message(role: str, content: str) -> str:
    """1 ターン分の ChatML ブロックを組み立てる。

    例: format_message(ROLE_USER, "こんにちは")
        -> "<|im_start|>user\\nこんにちは<|im_end|>\\n"
    """
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def assistant_header() -> str:
    """アシスタント応答の書き出しヘッダ (生成開始位置)。"""
    return f"{IM_START}{ROLE_ASSISTANT}\n"


def build_prompt(user_text: str, system_text: str = None, cot: bool = False) -> str:
    """推論・学習共通のプロンプトを組み立てる。

    Args:
        user_text: ユーザー発話。
        system_text: システムプロンプト。None なら system ブロックを省略。
        cot: True なら末尾に思考ブロックの開始タグ "<think>\\n" を付与する。
    """
    parts = []
    if system_text is not None:
        parts.append(format_message(ROLE_SYSTEM, system_text))
    parts.append(format_message(ROLE_USER, user_text))
    parts.append(assistant_header())
    if cot:
        parts.append(f"{THINK_START}\n")
    return "".join(parts)


def build_conversation_prompt(history, user_text: str, system_text: str = None,
                              cot: bool = False, max_turns: int = None) -> str:
    """マルチターン会話のプロンプトを組み立てる (§12.3)。

    Args:
        history: [{"role": "user"|"assistant", "content": str}, ...] の過去ターン列。
        user_text: 今回のユーザー発話。
        system_text: システムプロンプト。None なら省略。
        cot: True なら末尾に思考ブロック開始タグを付与。
        max_turns: 直近この件数の履歴のみ使用（None なら全件）。作業記憶の肥大抑制。
    """
    parts = []
    if system_text is not None:
        parts.append(format_message(ROLE_SYSTEM, system_text))

    turns = history or []
    if max_turns is not None and max_turns >= 0:
        # 注: turns[-0:] は全件になるため 0 は明示的に空扱いにする
        turns = turns[-max_turns:] if max_turns > 0 else []
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role in (ROLE_USER, ROLE_ASSISTANT) and content:
            parts.append(format_message(role, content))

    parts.append(format_message(ROLE_USER, user_text))
    parts.append(assistant_header())
    if cot:
        parts.append(f"{THINK_START}\n")
    return "".join(parts)


def format_assistant_response(text: str) -> str:
    """学習データ用: アシスタント応答本文を終端タグ付きで返す。"""
    return f"{text}{IM_END}\n"


def wrap_with_think(reasoning: str, answer: str) -> str:
    """学習データ用: 応答本文の前に思考ブロックを付与する。"""
    return f"{THINK_START}\n{reasoning}\n{THINK_END}\n{answer}"


def format_tool_call(payload_json: str) -> str:
    """ツール呼び出し JSON をタグで包む。"""
    return f"{TOOL_CALL_START}{payload_json}{TOOL_CALL_END}"


def format_tool_result(result_text: str) -> str:
    """ツール実行結果を tool ロールのブロックに変換し、
    続けてアシスタントの生成を再開させるヘッダを付与する。"""
    return f"{IM_START}{ROLE_TOOL}\n{result_text}{IM_END}\n{assistant_header()}"
