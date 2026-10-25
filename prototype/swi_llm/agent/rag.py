import hashlib
import json
import os
import re
import math
from collections import Counter
from swi_llm.utils.logger import setup_logger
from swi_llm.agent.chunking import semantic_chunk, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS

logger = setup_logger("rag")

# Hybrid RAG の密ベクトル埋め込みモデル (HuggingFace リポジトリID)
DENSE_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 02(中)/Q-12: クロスエンコーダ・リランカ（日本語特化の軽量モデル）。
# ロード失敗時は RRF 順をそのまま使う（C-4: オフラインでも検索機能を殺さない）。
RERANKER_MODEL = "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1"

# swi_llm パッケージ直下の knowledge_base をデフォルトとする（cwd 非依存）
DEFAULT_KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")

# 02(小): 密インデックスのディスクキャッシュ。起動のたびに KB 全文を再エンベッドする
# コストを排除する（KB が数万段落規模になると起動が分単位になるため）。
_CACHE_DIR_NAME = ".rag_cache"
_CACHE_INDEX_FILE = "index.faiss"
_CACHE_META_FILE = "meta.json"
# インデックスの互換性バージョン。距離空間・正規化・チャンキング方式を変えたらここを上げる
# （旧キャッシュは自動的に無効化されて再構築される）。
_INDEX_SCHEMA = "ip-normalized-v1+semchunk-v1"

# 02(中): 検索段の定数（実装編 §10.2 hybrid_search 準拠）
RAG_FETCH_K = 30           # 一次候補数（RRF 統合後、リランカへ渡す上限）
RAG_RRF_K = 60             # Reciprocal Rank Fusion の平滑化定数（標準値）
RAG_MIN_DENSE_SIM = 0.20   # 密検索を候補と見なすコサイン類似度の下限（「見つからない」判定に使用）


def reciprocal_rank_fusion(rankings, k=RAG_RRF_K):
    """Reciprocal Rank Fusion（実装編 §10.2）。

    Args:
        rankings: 「良い順に並んだ doc index のリスト」の列（BM25 順位・密検索順位など）。
            片方のランキングにしか現れない doc はその分の寄与のみを得る。
    Returns:
        {doc_index: rrf_score}。score = Σ_ranking 1 / (k + rank)（rank は 1 始まり）。
    """
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


class SimpleRAG:
    def __init__(self, kb_dir=DEFAULT_KB_DIR, embedding_model=None, reranker=None):
        """
        Args:
            kb_dir: ナレッジベースディレクトリ（.txt/.md を semantic_chunk で索引化）。
            embedding_model: `.encode(list[str], convert_to_numpy=True) -> np.ndarray` を持つ
                埋め込みモデル（テスト・差し替え用の DI）。None なら
                sentence-transformers の既定モデルをロードする。
            reranker: `.predict(list[(query, doc)]) -> list[float]` を持つクロスエンコーダ
                （sentence_transformers.CrossEncoder 互換）。None のとき、embedding_model も
                None（= 本番経路）なら既定リランカのロードを試みる。DI でモデルを注入した
                テスト/カスタム構成では既定ロードを行わない（ネットワークを踏まない）。
                環境変数 SWI_RAG_RERANKER=0 で既定ロードを無効化できる。
        """
        self.kb_dir = kb_dir
        self.documents = []
        self.idf = {}
        self.use_dense = False
        self.index = None
        self.reranker = reranker

        # Dense Embedding init
        try:
            import faiss  # noqa: F401  （利用可否の確認。実利用は後段）
            if embedding_model is not None:
                self.model = embedding_model
                self._model_id = f"injected:{type(embedding_model).__name__}"
            else:
                from sentence_transformers import SentenceTransformer
                # 軽量な多言語モデルを使用
                self.model = SentenceTransformer(DENSE_EMBEDDING_MODEL)
                self._model_id = DENSE_EMBEDDING_MODEL
            self.use_dense = True
            logger.info("FAISS loaded successfully for Hybrid RAG "
                        f"(embedding={self._model_id}).")
        except ImportError:
            logger.warning("FAISS or sentence-transformers not found. Falling back to BM25-only RAG.")
        except Exception as e:
            # モデルのダウンロード失敗（オフライン・リポジトリ不存在等）でも
            # 検索機能全体を殺さず BM25 のみで継続する
            logger.warning(f"Failed to load dense embedding model ({e}). Falling back to BM25-only RAG.")

        # 02(中)/Q-12: クロスエンコーダ・リランカ（本番経路のみ既定ロード）
        if (self.reranker is None and embedding_model is None
                and os.environ.get("SWI_RAG_RERANKER", "1") != "0"):
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANKER_MODEL)
                logger.info(f"RAG reranker loaded: {RERANKER_MODEL}")
            except Exception as e:
                # リランカは品質向上の追加段であり必須ではない（C-4）。RRF 順で継続する。
                logger.warning(f"RAG reranker unavailable ({e}); using RRF order as-is.")
                self.reranker = None

        self._load_and_index()

    def _tokenize(self, text):
        """
        日本語対応のため、外部の形態素解析ライブラリ（MeCab等）を使わず、
        文字のバイグラム（N-gram, N=2）でトークン化する軽量アプローチ (BM25用)。
        """
        text = text.lower()
        text = re.sub(r'\s+', '', text)
        if len(text) < 2:
            return [text]
        return [text[i:i+2] for i in range(len(text)-1)]

    # ------------------------------------------------------------------
    # 02(小): 密インデックスの永続化
    # ------------------------------------------------------------------
    def _cache_dir(self):
        return os.path.join(self.kb_dir, _CACHE_DIR_NAME)

    @staticmethod
    def _corpus_digest(paragraphs):
        h = hashlib.sha256()
        for p in paragraphs:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _cache_manifest(self, paragraphs):
        return {
            "schema": _INDEX_SCHEMA,
            "model": self._model_id,
            "n_docs": len(paragraphs),
            "corpus_digest": self._corpus_digest(paragraphs),
            # チャンク設定を変えたらキャッシュを作り直す
            "chunk": [CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS],
        }

    def _try_load_cached_index(self, paragraphs):
        """キャッシュが現在の KB 内容・モデル・スキーマと一致すればロードする。

        注意: faiss の read_index/write_index（C++ 側の fopen）は Windows の非 ASCII
        パス（例: デスクトップ配下）を開けず 'Illegal byte sequence' で失敗する。
        serialize_index/deserialize_index + Python 側のファイル IO（Unicode 対応）を使う。
        """
        import faiss
        import numpy as np
        index_path = os.path.join(self._cache_dir(), _CACHE_INDEX_FILE)
        meta_path = os.path.join(self._cache_dir(), _CACHE_META_FILE)
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta != self._cache_manifest(paragraphs):
                logger.info("RAG index cache is stale (KB/model changed); rebuilding.")
                return False
            index = faiss.deserialize_index(np.fromfile(index_path, dtype=np.uint8))
            if index.ntotal != len(paragraphs):
                logger.warning("RAG index cache size mismatch; rebuilding.")
                return False
            self.index = index
            logger.info(f"RAG dense index loaded from cache ({index.ntotal} docs; encode skipped).")
            return True
        except Exception as e:
            logger.warning(f"Failed to load RAG index cache ({e}); rebuilding.")
            return False

    def _save_index_cache(self, paragraphs):
        """キャッシュ書込は best-effort（失敗しても検索機能は継続する）。
        非 ASCII パス対応のため faiss.write_index ではなく serialize_index を使う（上記注意）。"""
        import faiss
        try:
            os.makedirs(self._cache_dir(), exist_ok=True)
            buf = faiss.serialize_index(self.index)          # np.uint8 配列
            with open(os.path.join(self._cache_dir(), _CACHE_INDEX_FILE), "wb") as f:
                f.write(buf.tobytes())
            with open(os.path.join(self._cache_dir(), _CACHE_META_FILE), "w", encoding="utf-8") as f:
                json.dump(self._cache_manifest(paragraphs), f, ensure_ascii=False, indent=1)
            logger.info("RAG dense index cache written.")
        except Exception as e:
            logger.warning(f"Failed to write RAG index cache ({e}); continuing without cache.")

    @staticmethod
    def _normalize_rows(x):
        """L2 正規化（コサイン類似度を内積として計算するため）。ゼロベクトルは保護。"""
        import numpy as np
        x = np.asarray(x, dtype="float32")
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, 1e-8, None)

    def _load_and_index(self):
        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir)
            logger.info(f"Created knowledge base directory at {self.kb_dir}")

        paragraphs = []
        try:
            # 02(小): 段落順（= インデックスの行順）を OS 依存にしないため必ずソートする。
            # 旧実装の os.listdir 素通しは環境によって順序が変わり、キャッシュ照合を壊す。
            for filename in sorted(os.listdir(self.kb_dir)):
                if filename.endswith(".txt") or filename.endswith(".md"):
                    filepath = os.path.join(self.kb_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    # 02(中): 単純な空行分割 → semantic_chunk（見出し優先 + トークン長上限 +
                    # オーバーラップ。実装編 §10.1）。出典は従来どおり "[filename] " を前置。
                    for chunk in semantic_chunk(text):
                        chunk = chunk.strip()
                        if len(chunk) > 10:
                            paragraphs.append(f"[{filename}] {chunk}")
        except Exception:
            logger.exception(f"Error reading knowledge base directory: {self.kb_dir}")

        # 簡易BM25インデックスの構築（ハイブリッドの語彙側 + 依存欠如時のフォールバック）
        df = Counter()
        self.documents = []
        for p in paragraphs:
            tokens = self._tokenize(p)
            self.documents.append({"text": p, "tokens": tokens})
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] += 1

        N = len(self.documents)
        if N > 0:
            for t, count in df.items():
                # BM25-like IDF計算
                self.idf[t] = math.log((N - count + 0.5) / (count + 0.5) + 1)

            # FAISS Index (密ベクトル) の構築 — キャッシュがあれば再エンベッドを省略
            if self.use_dense:
                import faiss
                if not self._try_load_cached_index(paragraphs):
                    logger.info("Computing dense embeddings for knowledge base...")
                    embeddings = self.model.encode(paragraphs, convert_to_numpy=True)
                    # 02(小): L2 正規化 + 内積（IndexFlatIP）= コサイン類似度。
                    # 旧実装の L2 距離 + 1/(1+dist) 変換は距離スケールに敏感だった。
                    embeddings = self._normalize_rows(embeddings)
                    d = embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(d)
                    self.index.add(embeddings)
                    self._save_index_cache(paragraphs)

        logger.info(f"RAG initialized: Loaded {N} document snippets from {self.kb_dir}.")

    def search(self, query, top_k=2, fetch_k=RAG_FETCH_K):
        """ハイブリッド検索（実装編 §10.2 hybrid_search 準拠）。

        BM25 順位と密検索順位を Reciprocal Rank Fusion で統合して一次候補
        （最大 fetch_k 件）を作り、リランカ（クロスエンコーダ）があれば
        並べ替えて top_k 件を返す。リランカ不在/失敗時は RRF 順をそのまま使う。
        """
        if not self.documents:
            logger.warning(f"Search aborted. Knowledge base is empty. Query: query_len={len(query or '')}")
            return "検索エラー: ナレッジベースにドキュメントが登録されていません。"

        # 1. BM25 Scoring
        query_tokens = self._tokenize(query)
        bm25_scores = []

        for doc in self.documents:
            doc_tokens = doc["tokens"]
            doc_len = len(doc_tokens)
            score = 0
            doc_counter = Counter(doc_tokens)

            for qt in query_tokens:
                if qt in doc_counter:
                    tf = doc_counter[qt]
                    score += self.idf.get(qt, 0) * (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * (doc_len / 50.0)))
            bm25_scores.append(score)

        # 2. Dense Scoring (FAISS: 正規化済み内積 = コサイン類似度)
        dense_sims = [0.0] * len(self.documents)
        if self.use_dense and self.index is not None:
            query_emb = self._normalize_rows(self.model.encode([query], convert_to_numpy=True))
            D, I = self.index.search(query_emb, len(self.documents))
            for sim, idx in zip(D[0], I[0]):
                if idx >= 0:
                    dense_sims[idx] = max(0.0, float(sim))   # 負相関は 0 に丸める

        # 3. 候補ゲート: BM25 に語彙ヒットがある、または密類似度が下限以上の文書のみを
        #    候補とする（RRF は順位のみを見るため、このゲートが「見つからない」判定を担う）。
        bm25_ranking = [i for i in sorted(range(len(self.documents)),
                                          key=lambda i: bm25_scores[i], reverse=True)
                        if bm25_scores[i] > 0]
        dense_ranking = [i for i in sorted(range(len(self.documents)),
                                           key=lambda i: dense_sims[i], reverse=True)
                         if dense_sims[i] >= RAG_MIN_DENSE_SIM]

        if not bm25_ranking and not dense_ranking:
            logger.warning(f"No results found for query: query_len={len(query or '')}")
            return "検索結果: 関連する情報が見つかりませんでした。"

        # 4. Reciprocal Rank Fusion（実装編 §10.2）で一次候補を統合
        rrf_scores = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
        ordered = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:max(fetch_k, top_k)]

        # 5. クロスエンコーダ再順位付け（Q-12）。失敗しても RRF 順で継続（C-4）。
        if self.reranker is not None and len(ordered) > 1:
            try:
                pairs = [(query, self.documents[i]["text"]) for i in ordered]
                ce_scores = self.reranker.predict(pairs)
                ordered = [i for _, i in sorted(zip(ce_scores, ordered),
                                                key=lambda t: float(t[0]), reverse=True)]
                logger.debug(f"Reranked {len(pairs)} candidates with cross-encoder.")
            except Exception as e:
                logger.warning(f"Reranker failed ({e}); falling back to RRF order.")

        results = [self.documents[i]["text"] for i in ordered[:top_k]]

        logger.info(f"Search successful for query: query_len={len(query or '')}. Returned {len(results)} snippets.")
        return "\n\n".join(results)

# モジュールとして直接呼ばれた場合のテストコード
if __name__ == "__main__":
    rag = SimpleRAG()
    res = rag.search("SWI_LLMの機能")
    logger.info(f"Test search result: {res}")
