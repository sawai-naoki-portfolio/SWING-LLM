"""
swi_llm/agent/chunking.py

RAG チャンキング（実装編 §10.1 `semantic_chunk` の実装。roadmap/llm ステップ4 / fix 02）。

設計（実装編 §10.1 に準拠）:
  - Markdown 見出し（行頭 #）の構造を優先してセクション分割し、セクションを跨ぐ
    チャンクを作らない（無関係な話題の混入防止）。
  - セクション内は 段落 → 文 → 文字 の順で崩しながら、`max_tokens` 以下になるよう
    貪欲に詰める。隣接チャンクには `overlap` トークンぶんの末尾を持ち越す
    （境界で文脈が切れて検索に掛からなくなるのを防ぐ）。
  - 長さの単位は既定で SentencePiece トークン数（実装編は max_tokens 指定。
    プロンプト注入時の予算と同じ単位で管理する）。`length_fn` の注入でテストや
    文字数ベース運用にも差し替え可能。

実装編からの意図的な差分:
  - ファイルパスは blueprint の `swi_llm/rag/chunking.py` ではなく、既存の検索実装
    （agent/rag.py・agent/self_rag.py）と同じ `swi_llm/agent/` 配下に置く。
  - 既定 max_tokens は blueprint の 512 ではなく 160。本モデルの学習系列長が 512 で、
    検索結果はプロンプトへ複数件注入されるため、512 トークンのチャンクは
    コンテキスト予算を食い潰す（top_k=2 でも本文の大半が検索結果になる）。
"""
import re

from swi_llm.utils.logger import setup_logger

logger = setup_logger("rag.chunking")

# 既定値（rag.py から参照される。変更するとインデックスキャッシュは自動で再構築される）
CHUNK_MAX_TOKENS = 160
CHUNK_OVERLAP_TOKENS = 24

_HEADING_RE = re.compile(r"^#{1,6}\s")
_SENTENCE_END_RE = re.compile(r"(?<=[。．！？!?])")


def _default_length_fn():
    """SentencePiece トークン数で長さを測る（プロンプト予算と同じ単位）。"""
    from swi_llm.data.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    return lambda text: len(tokenizer.encode(text))


def split_by_headings(text):
    """Markdown 見出し（行頭 #）でセクションへ分割する。

    見出し行はそのセクション本文の先頭に残す（チャンク単体でも出典文脈が分かるように）。
    見出しが無いテキストは全体を 1 セクションとして返す。
    """
    sections = []
    current = []
    for line in text.splitlines():
        if _HEADING_RE.match(line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return [s for s in sections if s.strip()]


def _split_sentences(paragraph):
    return [s for s in _SENTENCE_END_RE.split(paragraph) if s.strip()]


def _hard_split(piece, max_len, length_fn):
    """1 文すら max_len を超える場合の最終手段: 必ず max_len 以下になるまで
    文字ウィンドウを半減させながら前から切り出す。"""
    out = []
    start = 0
    while start < len(piece):
        end = len(piece)
        window = piece[start:end]
        while length_fn(window) > max_len and (end - start) > 1:
            end = start + max(1, (end - start) // 2)
            window = piece[start:end]
        out.append(window)
        start = end
    return out


def _pack(pieces, max_len, overlap, length_fn):
    """断片（段落/文）を max_len 以下へ貪欲に詰める。フラッシュ時は末尾の断片を
    合計 overlap 以下の範囲で次チャンクへ持ち越す（隣接チャンクの重なり）。"""
    chunks = []
    cur, cur_len = [], 0
    for piece in pieces:
        p_len = length_fn(piece)
        join_cost = 1 if cur else 0          # "\n" 結合ぶんの概算（SP では通常 1 トークン）
        if cur and cur_len + join_cost + p_len > max_len:
            chunks.append("\n".join(cur))
            # overlap: 末尾から合計 overlap 以下の断片を持ち越す
            keep, keep_len = [], 0
            for q in reversed(cur):
                q_len = length_fn(q)
                if keep_len + q_len > overlap:
                    break
                keep.insert(0, q)
                keep_len += q_len
            cur, cur_len = keep, keep_len
            # 持ち越し + 新断片が max を超える場合は持ち越しを諦める（overlap が大きい設定への防御）
            if cur and cur_len + 1 + p_len > max_len:
                cur, cur_len = [], 0
        cur.append(piece)
        cur_len += p_len + (1 if len(cur) > 1 else 0)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def semantic_chunk(document, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP_TOKENS,
                   length_fn=None):
    """文書 → 検索用チャンク列（実装編 §10.1）。

    Args:
        document: 入力テキスト（Markdown / プレーンテキスト）。
        max_tokens: チャンク長の上限（length_fn の単位。既定は SP トークン）。
        overlap: 隣接チャンク間で持ち越す末尾の長さ。
        length_fn: 長さ計測関数。None なら SentencePiece トークン数。

    Returns:
        list[str]。空/空白のみの入力は []。
    """
    if not document or not document.strip():
        return []
    if length_fn is None:
        length_fn = _default_length_fn()

    # 同一文字列の長さ計測を繰り返すため（pack の overlap 巻き戻し）メモ化する
    cache = {}

    def cached_len(s):
        if s not in cache:
            cache[s] = length_fn(s)
        return cache[s]

    chunks = []
    for section in split_by_headings(document):
        pieces = []
        for paragraph in re.split(r"\n\s*\n", section):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if cached_len(paragraph) <= max_tokens:
                pieces.append(paragraph)
                continue
            for sentence in _split_sentences(paragraph):
                if cached_len(sentence) <= max_tokens:
                    pieces.append(sentence)
                else:
                    pieces.extend(_hard_split(sentence, max_tokens, cached_len))
        chunks.extend(_pack(pieces, max_tokens, overlap, cached_len))
    return chunks
