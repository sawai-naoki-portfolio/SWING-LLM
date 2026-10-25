"""
swi_llm/agent/self_rag.py

Self-RAG（エージェント型 RAG / ブループリント §13.2, §13.4）。

固定的な「検索→生成」ではなく、
  1. 検索要否の自己判断（should_retrieve）
  2. 取得根拠が問いを実際に裏付けているかの自己検証（grounding verification）
を挟み、根拠のない主張（ハルシネーション）を抑制する。

検証で根拠不十分と判断した場合は、結果に grounded=False を明示して
「根拠が無いことは断定しない」という制約をモデルに伝える。
"""
import re
import threading

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import INTENT_KEYWORDS_SEARCH

logger = setup_logger("agent.self_rag")

# S5: SelfRAG(=SimpleRAG) の生成は SentenceTransformer ロード + FAISS 構築を伴い重い。
# search_documents が呼ばれる度に new すると毎回この初期化が走り致命的に遅い。
# kb_dir ごとに 1 度だけ構築してキャッシュする（Web は複数スレッドから呼ぶためロックで保護）。
_INSTANCES = {}
_INSTANCES_LOCK = threading.Lock()

# 検索が有用と判断する疑問・知識要求の手がかり
_QUESTION_MARKERS = ("？", "?", "とは", "教えて", "何", "誰", "どこ", "いつ", "なぜ", "どうやって")
# 取得結果が問いを裏付けていると見なす最小の語彙重なりスコア
DEFAULT_MIN_GROUNDING = 0.08
# 取得結果テキスト中で「見つからなかった」ことを示す既知メッセージの手がかり
_NO_RESULT_MARKERS = ("見つかりません", "登録されていません", "No relevant", "検索エラー")


def _char_bigrams(text):
    text = re.sub(r"\s+", "", text.lower())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


# --- 純粋関数（RAG インスタンス非依存・単体テスト可能） ---
def should_retrieve(query: str) -> bool:
    """検索が有用そうかを判定する（知識要求・疑問文・検索キーワード）。"""
    if any(kw in query for kw in INTENT_KEYWORDS_SEARCH):
        return True
    return any(marker in query for marker in _QUESTION_MARKERS)


def grounding_score(query: str, results_text: str) -> float:
    """クエリと取得テキストの文字バイグラム重なり（Jaccard 風）。"""
    q = _char_bigrams(query)
    r = _char_bigrams(results_text)
    if not q or not r:
        return 0.0
    return len(q & r) / len(q)


class SelfRAG:
    """SimpleRAG に検索要否判断と根拠自己検証を加えたラッパ。"""

    def __init__(self, kb_dir=None):
        from .rag import SimpleRAG
        self.rag = SimpleRAG(kb_dir) if kb_dir else SimpleRAG()

    def should_retrieve(self, query: str) -> bool:
        return should_retrieve(query)

    def _grounding_score(self, query: str, results_text: str) -> float:
        return grounding_score(query, results_text)

    def retrieve(self, query: str, top_k: int = 2, min_grounding: float = DEFAULT_MIN_GROUNDING,
                 force: bool = False):
        """検索し、根拠の自己検証結果を付けて返す (§13.2/§13.4)。

        Args:
            force: True なら should_retrieve の要否判断をスキップして必ず検索する。
                ツール呼び出し（モデル/intent が既に「検索する」と決定済み）用。
                実測 (2026-07-12): モデルが search_documents を明示的に呼んだのに、
                生ユーザー発話向けのヒューリスティック（疑問符・「教えて」等）が
                短い検索クエリ（例:「SWI-LLM パラメータ数」）を弾いて retrieved=False に
                なる誤動作があった。判断主体が既に決めた検索は拒否しない。

        Returns dict:
          - retrieved: 検索を実行したか
          - grounded:  取得根拠が問いを裏付けているか
          - results:   取得テキスト（または根拠不十分メッセージ）
          - grounding_score: 数値スコア
        """
        if not force and not self.should_retrieve(query):
            logger.info(f"Self-RAG: retrieval skipped for non-knowledge query: query_len={len(query or '')}")
            return {"retrieved": False, "grounded": False, "results": None, "grounding_score": 0.0}

        results_text = self.rag.search(query, top_k=top_k)
        # SimpleRAG が「見つからない」系メッセージを返した場合は非接地
        if not results_text or any(m in results_text for m in _NO_RESULT_MARKERS):
            logger.info(f"Self-RAG: no results for query_len={len(query or '')}")
            return {"retrieved": True, "grounded": False,
                    "results": "根拠となる文書が見つかりませんでした。事実を断定せず、その旨を伝えてください。",
                    "grounding_score": 0.0}

        score = self._grounding_score(query, results_text)
        grounded = score >= min_grounding
        if not grounded:
            logger.info(f"Self-RAG: retrieved but low grounding (score={score:.3f}) for query_len={len(query or '')}")
            return {"retrieved": True, "grounded": False,
                    "results": ("取得した文書はクエリとの関連が弱いため信頼できる根拠になりません。"
                                "推測で断定しないでください。"),
                    "grounding_score": score}

        logger.info(f"Self-RAG: grounded answer (score={score:.3f}) for query_len={len(query or '')}")
        return {"retrieved": True, "grounded": True, "results": results_text, "grounding_score": score}


def get_self_rag(kb_dir=None) -> "SelfRAG":
    """SelfRAG を kb_dir ごとにキャッシュして返す（S5: 毎回の重い再構築を回避）。

    初回のみ SentenceTransformer ロード + FAISS 構築を行い、以降は同一インスタンスを
    使い回す。スレッドセーフ（二重チェックロック）。知識ベースを更新した場合は
    reset_self_rag() でキャッシュを破棄して再構築させる。
    """
    key = kb_dir or "__default__"
    inst = _INSTANCES.get(key)
    if inst is not None:
        return inst
    with _INSTANCES_LOCK:
        inst = _INSTANCES.get(key)
        if inst is None:
            logger.info(f"Building SelfRAG (cached singleton) for kb_dir={key!r} ...")
            inst = SelfRAG(kb_dir)
            _INSTANCES[key] = inst
        return inst


def reset_self_rag():
    """キャッシュ済み SelfRAG を破棄する（知識ベース更新時などに再構築させる）。"""
    with _INSTANCES_LOCK:
        _INSTANCES.clear()
