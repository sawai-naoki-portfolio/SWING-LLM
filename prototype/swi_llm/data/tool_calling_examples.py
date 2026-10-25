"""
swi_llm/data/tool_calling_examples.py

ローカルツールに厳密一致した Function Calling の SFT 教師データ（正例・負例）を
決定的に合成する（roadmap/llm_roadmap.md ステップ3 / fix 19: ルールベース意図検知の撤去）。

なぜ必要か:
  既存 SFT は外部 FC データセット（DataPilot/Function-Calling-Japanese）を取り込むが、
  そこに現れるツール名・引数名は **本プロジェクトに実際に登録されるツール**
  （agent/tools.py の TOOL_SCHEMAS: calculate / get_weather / search_documents）と
  一致しない。そのためモデルは「tool_call というタグの形」は学べても、
  「呼べば実際に dispatch されるツール呼び出し」を学べていない
  （= AgentBase.process_llm_output_ex が ok=False になる出力を量産する）。

  さらに負例（ツール不要な入力にはツールを呼ばず直接答える）が明示的に無いため、
  過剰発火（何にでも tool_call を出す）を抑えられない。

設計:
  - ツール名・引数名は TOOL_SCHEMAS を単一情報源として参照する（文字列ハードコード禁止）。
  - 生成は決定的（同じ seed なら同じ出力）。学習の再現性とテスト容易性のため。
  - 正例の assistant 応答は「tool_call を出すところまで」（ツール実行結果の後続応答は
    推論時に ReAct ループが注入するため、SFT では tool_call の生成のみを学ばせる）。
  - 生成した正例は `AgentBase.process_llm_output_ex` でそのまま dispatch できる
    （テストで検証。教師データと推論器の整合を機械的に保証する）。
"""
import json
import random

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    TOOL_CALCULATE, TOOL_GET_WEATHER, TOOL_SEARCH_DOCUMENTS,
)
from swi_llm.utils.chat_template import format_tool_call, wrap_with_think

logger = setup_logger("data.tool_calling")

# 正例に使うツール（execute_python は SEC-6 により既定で未登録なので教師にしない）
SUPPORTED_TOOLS = (TOOL_CALCULATE, TOOL_GET_WEATHER, TOOL_SEARCH_DOCUMENTS)

# --- 正例の素材（ユーザー発話のテンプレート, ツール引数） -------------------
_CALC_CASES = [
    ("{a} + {b} はいくつですか？", "{a} + {b}"),
    ("{a} かける {b} を計算してください。", "{a} * {b}"),
    ("{a} から {b} を引くといくつ？", "{a} - {b}"),
    ("{a} ÷ {b} の答えを教えて。", "{a} / {b}"),
    ("({a} + {b}) * 2 を計算して。", "({a} + {b}) * 2"),
]
# 語彙プールは train / eval で分割する（hold-out）。
# 分割しないと、i % len(pool) 由来の同一プロンプトが学習と評価の両方に現れ、
# 「丸暗記」を評価してしまう（汎化が測れない）。テンプレート（言い回し）は共有してよい
# ——学ばせたいのは形式であり、未知の都市・話題へ般化できるかを測る。
_TRAIN_CITIES = ["東京", "大阪", "札幌", "福岡", "名古屋", "京都", "仙台", "那覇", "横浜", "広島"]
_EVAL_CITIES = ["金沢", "松山", "長崎", "熊本", "岐阜"]
_WEATHER_TEMPLATES = [
    "{city}の天気を教えてください。",
    "今日の{city}の天気はどうですか？",
    "{city}は今どんな天気？",
]
_TRAIN_SEARCH_TOPICS = [
    "SWI プロジェクトのアーキテクチャ",
    "社内ドキュメントのセキュリティ方針",
    "ナレッジベースに登録された運用手順",
    "設計書に書かれているモデル構成",
    "登録済みドキュメントの評価指標",
]
_EVAL_SEARCH_TOPICS = [
    "オンボーディング資料の記載内容",
    "障害対応マニュアルの手順",
    "リリースノートに記録された変更点",
]
_SEARCH_TEMPLATES = [
    "{topic}について調べてください。",
    "{topic}を検索して教えて。",
    "ドキュメントから{topic}を探して。",
]

# --- 負例（ツール不要 = tool_call を出さずに直接答える） --------------------
# 過剰発火の抑制が目的。「数字を含むが計算要求ではない」「天気の話題だが取得が不要」など、
# 正例と紛らわしいケースを意図的に含める（境界を学ばせる）。負例も train/eval で分割する。
_TRAIN_NEGATIVE_CASES = [
    ("こんにちは。今日もよろしくお願いします。", "こんにちは。本日もお手伝いします。ご用件をどうぞ。"),
    ("ありがとう、助かりました。", "どういたしまして。またいつでもお声がけください。"),
    ("あなたは誰ですか？", "私は SWI のアシスタントです。質問への回答や、必要に応じたツールの実行を行います。"),
    ("日本の首都はどこですか？", "日本の首都は東京です。"),
    ("2 は偶数ですか？", "はい、2 は偶数です。2 で割り切れる整数を偶数と呼びます。"),
    ("昨日は雨で大変でしたね。", "それは大変でしたね。今日は落ち着いて過ごせるとよいですね。"),
    ("計算が得意なんですね。", "はい、必要に応じて計算ツールを使って正確にお答えします。"),
    ("面白い話をしてください。", "では一つ。プログラマは電球を替えません。それはハードウェアの問題だからです。"),
    ("元気ですか？", "はい、問題なく稼働しています。ご用件をお知らせください。"),
    ("この前の話の続きをしましょう。", "承知しました。どの話題から再開しますか？"),
]
_EVAL_NEGATIVE_CASES = [
    ("おはようございます。", "おはようございます。本日のご用件をお知らせください。"),
    ("助かりました、感謝します。", "お役に立てて何よりです。またご相談ください。"),
    ("富士山は美しい山ですね。", "はい、日本を代表する山として広く親しまれています。"),
    ("3 は素数ですよね？", "はい、3 は素数です。1 と自分自身以外に約数を持ちません。"),
    ("最近は暑い日が続きますね。", "そうですね。水分補給を忘れずにお過ごしください。"),
]

# CoT（<think>）を付ける確率。既存 SFT（data/sft.py）と同じ 0.5 で揃える。
_COT_PROB = 0.5

_POSITIVE_REASONING = (
    "ユーザーの要求は「{short}...」である。\n"
    "この要求は登録済みツール '{tool}' で正確に解決できる。引数を抽出して tool_call を出力する。"
)
_NEGATIVE_REASONING = (
    "ユーザーの発話は「{short}...」である。\n"
    "これは登録済みツールを必要としない（計算・天気取得・文書検索のいずれでもない）。"
    "ツールを呼ばずに直接回答する。"
)


def _tool_call_text(tool_name: str, arguments: dict) -> str:
    """TOOL_SCHEMAS に一致する tool_call ブロックを組み立てる（AgentBase が解釈できる形）。"""
    payload = json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)
    return format_tool_call(payload)


def _maybe_think(rng, user_text: str, body: str, reasoning_template: str, **fmt) -> str:
    if rng.random() >= _COT_PROB:
        return body
    short = user_text[:30].replace("\n", " ")
    return wrap_with_think(reasoning_template.format(short=short, **fmt), body)


def build_tool_calling_examples(n_per_tool: int = 40, n_negative: int = 60,
                                seed: int = 20260711, split: str = "train"):
    """ローカルツールに一致した tool_call 正例と、ツール不要の負例を合成する。

    Args:
        n_per_tool: ツールごとの正例数（calculate / get_weather / search_documents）。
        n_negative: 負例（tool_call を出さない応答）の数。
        seed: 乱数シード（決定的生成）。
        split: "train"（学習用）/ "eval"（hold-out 評価用）。語彙プール（都市・検索話題・
            負例文）が排他になるよう分割されており、同一プロンプトが両者に現れない
            （= 丸暗記ではなく未知語彙への般化を測れる）。calculate の数値は乱数。

    Returns:
        [{"user": str, "assistant": str}, ...]。assistant は正例なら <tool_call> を含み、
        負例なら含まない。CoT（<think>）は確率 0.5 で付与（既存 SFT と同条件）。
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
    is_eval = split == "eval"
    cities = _EVAL_CITIES if is_eval else _TRAIN_CITIES
    topics = _EVAL_SEARCH_TOPICS if is_eval else _TRAIN_SEARCH_TOPICS
    negatives = _EVAL_NEGATIVE_CASES if is_eval else _TRAIN_NEGATIVE_CASES

    rng = random.Random(seed)
    examples = []

    # --- 正例: calculate（数値は乱数。train/eval で同一プロンプトになる確率は無視できる） ---
    for i in range(n_per_tool):
        template, expr_template = _CALC_CASES[i % len(_CALC_CASES)]
        a, b = rng.randint(2, 999), rng.randint(2, 99)
        user_text = template.format(a=a, b=b)
        expression = expr_template.format(a=a, b=b)
        body = _tool_call_text(TOOL_CALCULATE, {"expression": expression})
        examples.append({
            "user": user_text,
            "assistant": _maybe_think(rng, user_text, body, _POSITIVE_REASONING, tool=TOOL_CALCULATE),
        })

    # --- 正例: get_weather ---
    for i in range(n_per_tool):
        city = cities[i % len(cities)]
        user_text = _WEATHER_TEMPLATES[i % len(_WEATHER_TEMPLATES)].format(city=city)
        body = _tool_call_text(TOOL_GET_WEATHER, {"location": city})
        examples.append({
            "user": user_text,
            "assistant": _maybe_think(rng, user_text, body, _POSITIVE_REASONING, tool=TOOL_GET_WEATHER),
        })

    # --- 正例: search_documents ---
    for i in range(n_per_tool):
        topic = topics[i % len(topics)]
        user_text = _SEARCH_TEMPLATES[i % len(_SEARCH_TEMPLATES)].format(topic=topic)
        body = _tool_call_text(TOOL_SEARCH_DOCUMENTS, {"query": topic})
        examples.append({
            "user": user_text,
            "assistant": _maybe_think(rng, user_text, body, _POSITIVE_REASONING, tool=TOOL_SEARCH_DOCUMENTS),
        })

    # --- 負例: ツールを呼ばない ---
    for i in range(n_negative):
        user_text, answer = negatives[i % len(negatives)]
        examples.append({
            "user": user_text,
            "assistant": _maybe_think(rng, user_text, answer, _NEGATIVE_REASONING),
        })

    rng.shuffle(examples)
    logger.info(f"Built {len(examples)} local tool-calling examples [{split}] "
                f"(positive={n_per_tool * len(SUPPORTED_TOOLS)}, negative={n_negative}, seed={seed}).")
    return examples


def build_fc_eval_set(n_per_tool: int = 6, n_negative: int = 12, seed: int = 777):
    """FC 評価用の hold-out セット（学習では使わない語彙プールから生成）。

    Returns:
        [{"user": str, "expect_tool": str | None}, ...]
        expect_tool が None の項目は「tool_call を出してはいけない」負例。
    """
    items = []
    examples = build_tool_calling_examples(
        n_per_tool=n_per_tool, n_negative=n_negative, seed=seed, split="eval")
    for ex in examples:
        expect_tool = None
        for tool in SUPPORTED_TOOLS:
            if f'"name": "{tool}"' in ex["assistant"]:
                expect_tool = tool
                break
        items.append({"user": ex["user"], "expect_tool": expect_tool})
    return items
