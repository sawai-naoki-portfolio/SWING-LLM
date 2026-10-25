"""
swi_llm/agent/tools.py

エージェントが呼び出すツールの実装と JSON Schema 定義 (REQ-01 で executor.py から分離)。

- get_weather: wttr.in による天気取得
- calculate: AST ベースの安全な四則演算（eval 不使用）
- execute_python: sandbox.py のサンドボックス実行（本モジュールから再公開）
- search_documents: knowledge_base に対するハイブリッド RAG 検索
- wikipedia_search: Wikipedia API による Web 検索（NEW-L6 / 03_agentic 展望2）
"""
import os
import ast
import json
import operator
import re
import urllib.parse
import urllib.request

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    TOOL_GET_WEATHER, TOOL_CALCULATE, TOOL_EXECUTE_PYTHON, TOOL_SEARCH_DOCUMENTS,
    TOOL_WIKIPEDIA_SEARCH,
    CALC_MAX_EXPR_LEN,
    WEATHER_API_URL_TEMPLATE, WEATHER_API_TIMEOUT_SEC,
    WIKIPEDIA_API_URL_TEMPLATE, WIKIPEDIA_ALLOWED_LANGS, WIKIPEDIA_API_TIMEOUT_SEC,
    WIKIPEDIA_MAX_RESULTS, WIKIPEDIA_SNIPPET_MAX_CHARS,
)
from swi_llm.agent.sandbox import execute_python  # noqa: F401  (ツール一式として再公開)

logger = setup_logger("agent.tools")


TOOL_SCHEMAS = {
    TOOL_GET_WEATHER: {
        "name": TOOL_GET_WEATHER,
        "description": "Get the current weather for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name, e.g., 'Tokyo' or 'San Francisco'."}
            },
            "required": ["location"]
        }
    },
    TOOL_CALCULATE: {
        "name": TOOL_CALCULATE,
        "description": "Evaluate a mathematical expression. Use for math queries.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A mathematical expression, e.g., '12345 * 6789' or '2 + 2'."}
            },
            "required": ["expression"]
        }
    },
    TOOL_EXECUTE_PYTHON: {
        "name": TOOL_EXECUTE_PYTHON,
        "description": "Execute Python code and return the output. Useful for complex logic or scripting.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The python code string to execute."}
            },
            "required": ["code"]
        }
    },
    TOOL_SEARCH_DOCUMENTS: {
        "name": TOOL_SEARCH_DOCUMENTS,
        "description": "Search the knowledge base for documents matching the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search term or question to find in documents."}
            },
            "required": ["query"]
        }
    },
    TOOL_WIKIPEDIA_SEARCH: {
        "name": TOOL_WIKIPEDIA_SEARCH,
        "description": ("Search Wikipedia for up-to-date encyclopedic information. "
                        "Use when the local knowledge base does not cover the topic."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The topic or question to look up on Wikipedia."},
                "lang": {"type": "string", "description": "Wikipedia language code: 'ja' (default) or 'en'."}
            },
            "required": ["query"]
        }
    }
}


def get_weather(location, date="today"):
    """指定された場所の天気を取得する関数（wttr.inを使用）"""
    try:
        encoded_location = urllib.parse.quote(location)
        url = WEATHER_API_URL_TEMPLATE.format(location=encoded_location)
        # wttr.in はブラウザ系 UA に HTML ページを返すことがある（実測 2026-07-12:
        # ?format=3 でも <!DOCTYPE html> 全文が返り、LLM のコンテキストを汚染した）。
        # curl 系 UA でプレーンテキストを要求する。
        req = urllib.request.Request(url, headers={'User-Agent': 'curl/8.0'})
        with urllib.request.urlopen(req, timeout=WEATHER_API_TIMEOUT_SEC) as response:
            weather_data = response.read().decode('utf-8').strip()
        # HTML が返った場合は「取得失敗」として扱う（巨大な HTML を LLM に流し込まない）
        if weather_data[:200].lstrip().lower().startswith(("<!doctype", "<html")):
            logger.warning(f"get_weather({location}) returned HTML instead of plain text; treating as failure.")
            return {"error": "Weather service returned an unexpected (HTML) response."}
        logger.info(f"get_weather({location}) -> {weather_data}")
        return {"location": location, "condition": weather_data}
    except Exception as e:
        logger.error(f"get_weather({location}) failed: {e}")
        return {"error": f"Failed to get weather: {e}"}


_HTML_TAG = re.compile(r"<[^>]+>")


def wikipedia_search(query, lang="ja", max_results=WIKIPEDIA_MAX_RESULTS):
    """Wikipedia API で検索し、上位ヒットのタイトル・要約・URL を返す（NEW-L6）。

    設計（03_agentic 展望2「Webブラウジングツールの追加」）の実装。API キー不要で
    安定しており、ローカル RAG（search_documents）がカバーしない一般知識を補う。

    SSRF 対策: URL はホスト固定のテンプレート + 言語コード許可リストから組み立てる
    （モデル出力から任意 URL を構成しない）。スニペットは HTML タグを除去し長さ制限する
    （LLM のコンテキスト汚染防止。get_weather の HTML 対策と同方針）。
    """
    query = (query or "").strip()
    if not query:
        return {"error": "query is required."}
    if lang not in WIKIPEDIA_ALLOWED_LANGS:
        logger.warning(f"wikipedia_search: unsupported lang '{lang}'; using 'ja'.")
        lang = "ja"
    try:
        url = WIKIPEDIA_API_URL_TEMPLATE.format(
            lang=lang, query=urllib.parse.quote(query), limit=int(max_results))
        # Wikipedia API はボット識別可能な UA を要求する（無記名 UA は 403 になり得る）
        req = urllib.request.Request(url, headers={"User-Agent": "SWI-LLM/1.0 (research prototype)"})
        with urllib.request.urlopen(req, timeout=WIKIPEDIA_API_TIMEOUT_SEC) as response:
            data = json.loads(response.read().decode("utf-8"))
        hits = data.get("query", {}).get("search", [])
        results = []
        for hit in hits[:max_results]:
            snippet = _HTML_TAG.sub("", str(hit.get("snippet", "")))[:WIKIPEDIA_SNIPPET_MAX_CHARS]
            results.append({
                "title": hit.get("title", ""),
                "snippet": snippet,
                "url": f"https://{lang}.wikipedia.org/?curid={hit.get('pageid', '')}",
            })
        logger.info(f"wikipedia_search({query!r}, lang={lang}) -> {len(results)} hits")
        if not results:
            return {"query": query, "results": [], "note": "No Wikipedia articles matched."}
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"wikipedia_search({query!r}) failed: {e}")
        return {"error": f"Failed to search Wikipedia: {e}"}


# 四則演算のみを AST で安全に評価する（eval は使わない）
_CALC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_expr(expr):
    """数値リテラルと四則演算のみ許可した AST ベースの式評価。"""
    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
            return _CALC_OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
            return _CALC_OPS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"Unsupported operation: {type(node).__name__}")
    return evaluate(ast.parse(expr, mode="eval").body)


def calculate(expression):
    """計算式を評価する関数（AST ベースの安全な評価器を使用）"""
    try:
        if len(expression) > CALC_MAX_EXPR_LEN or "**" in expression:
            logger.warning(f"calculate rejected expression (too long or forbidden op): {expression!r}")
            return {"error": "Expression too long or contains forbidden operators."}

        result = _safe_eval_expr(expression)
        logger.info(f"calculate({expression}) -> {result}")
        return {"expression": expression, "result": result}
    except Exception as e:
        logger.error(f"calculate({expression!r}) failed: {e}")
        return {"error": str(e)}


def search_documents(query):
    """
    外部の知識ベースから、キーワードに一致するドキュメントを検索して返します。
    """
    try:
        from .self_rag import get_self_rag
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # S5: キャッシュ済みシングルトンを使い、毎回の SentenceTransformer ロード +
        # FAISS 再構築（数百ms〜数秒/回）を回避する。
        rag = get_self_rag(os.path.join(base_dir, "knowledge_base"))

        # Self-RAG: 取得根拠の自己検証 (§13.2/§13.4)。force=True — 本関数は「検索する」と
        # 判断済みの主体（モデルの tool_call / intent のキーワード検知）からのみ呼ばれるため、
        # 生ユーザー発話向けの検索要否ヒューリスティックで二重に弾かない。
        outcome = rag.retrieve(query, top_k=2, force=True)
        # SEC-7: 検索クエリ(ユーザー入力)を平文でログに残さない。長さ・結果メタのみ記録する。
        logger.info(f"search_documents(query_len={len(query or '')}) -> retrieved={outcome['retrieved']}, "
                    f"grounded={outcome['grounded']}, score={outcome['grounding_score']:.3f}")
        return {
            "status": "success",
            "results": outcome["results"] if outcome["results"] is not None else "No relevant documents found.",
            "grounded": outcome["grounded"],
        }

    except Exception as e:
        logger.exception(f"search_documents(query_len={len(query or '')}) failed.")   # SEC-7
        return {"error": f"Search failed: {str(e)}"}
