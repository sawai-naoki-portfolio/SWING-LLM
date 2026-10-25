"""
swi_llm/data/pretokenize.py

事前学習コーパスの事前トークン化キャッシュ（roadmap/llm ステップ5 / fix 03）。

従来はエポック（学習実行）のたびに全ドキュメントをトークナイズし、int32 の
フラット配列を RAM に保持していた。数十億トークン規模ではトークン化だけで
数時間・RAM 数十 GB になるため、
  1. トークン列を uint16 バイナリ（vocab 32768 < 65536）でディスクへ一度だけ書き、
  2. 以後は np.memmap（読み取り専用・オンデマンドページング）で供給する。
キャッシュは「コーパス内容のダイジェスト + トークナイザ指紋 + フォーマット版数」で
検証し、どれかが変われば自動で作り直す（黙って古いトークン列を使わない —
trainer の data_size ガードと同じ思想）。

環境変数 SWI_PRETOKENIZE=0 で無効化でき、従来のオンザフライ経路に戻せる（退避弁）。
"""
import hashlib
import json
import os

import numpy as np

from swi_llm.utils.logger import setup_logger

logger = setup_logger("data.pretokenize")

_CACHE_VERSION = "uint16-eot-v1"
_DTYPE = np.uint16
_DTYPE_MAX = np.iinfo(_DTYPE).max

# 既定キャッシュ置き場: swi_llm/data/.token_cache/
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".token_cache")


def pretokenize_enabled():
    return os.environ.get("SWI_PRETOKENIZE", "1") != "0"


def corpus_digest(docs):
    """ドキュメント列の内容ダイジェスト（順序込み）。"""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    h.update(str(len(docs)).encode())
    return h.hexdigest()


def tokenizer_fingerprint():
    """SentencePiece モデルファイルの指紋。トークナイザ刷新でキャッシュを無効化する。"""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.model")
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def _paths(cache_dir, tag):
    return (os.path.join(cache_dir, f"{tag}.bin"),
            os.path.join(cache_dir, f"{tag}.json"))


def _expected_meta(docs, eot_id, fingerprint):
    return {
        "version": _CACHE_VERSION,
        "tokenizer": fingerprint,
        "corpus_digest": corpus_digest(docs),
        "n_docs": len(docs),
        "eot_id": int(eot_id),
    }


def get_or_build_token_cache(docs, tokenizer, eot_id, tag, cache_dir=None, fingerprint=None):
    """トークン列キャッシュを返す（ヒット時は memmap、ミス時は構築して memmap）。

    Args:
        docs: ドキュメント文字列のリスト。
        tokenizer: `.encode(text) -> list[int]`。
        eot_id: 文書区切りトークン ID（各文書の末尾に付与。従来実装と同一の並び）。
        tag: キャッシュ名（例: "pretrain_train" / "pretrain_val"）。
        cache_dir: 省略時は swi_llm/data/.token_cache/。
        fingerprint: トークナイザ指紋（テスト用 DI。省略時は tokenizer.model のハッシュ）。

    Returns:
        np.memmap（dtype=uint16, mode='r'）。呼び出し側は読み取り専用として扱うこと。
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    fingerprint = fingerprint or tokenizer_fingerprint()
    bin_path, meta_path = _paths(cache_dir, tag)
    expected = _expected_meta(docs, eot_id, fingerprint)

    # --- キャッシュヒット判定 ---
    if os.path.exists(bin_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            n_tokens = meta.pop("n_tokens", None)
            if meta == expected and n_tokens is not None:
                mm = np.memmap(bin_path, dtype=_DTYPE, mode="r")
                if len(mm) == n_tokens:
                    logger.info(f"Token cache hit [{tag}]: {n_tokens} tokens memmapped "
                                f"(tokenization skipped).")
                    return mm
            logger.info(f"Token cache stale [{tag}] (corpus/tokenizer changed); rebuilding.")
        except Exception as e:
            logger.warning(f"Failed to read token cache [{tag}] ({e}); rebuilding.")

    # --- 構築（1 回だけのトークン化）---
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = bin_path + ".tmp"
    n_tokens = 0
    logger.info(f"Building token cache [{tag}]: tokenizing {len(docs)} docs -> {bin_path}")
    with open(tmp_path, "wb") as f:
        for text in docs:
            text = text.strip()
            if not text:
                continue
            ids = tokenizer.encode(text)
            ids.append(int(eot_id))
            arr = np.asarray(ids, dtype=np.int64)
            if arr.size and int(arr.max()) > _DTYPE_MAX:
                raise ValueError(
                    f"Token id {int(arr.max())} exceeds uint16 range; "
                    f"increase cache dtype before using vocab > {_DTYPE_MAX + 1}.")
            if arr.size and int(arr.min()) < 0:
                raise ValueError("Negative token id encountered during pretokenization.")
            arr.astype(_DTYPE).tofile(f)
            n_tokens += arr.size
    os.replace(tmp_path, bin_path)

    meta = dict(expected)
    meta["n_tokens"] = n_tokens
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    logger.info(f"Token cache built [{tag}]: {n_tokens} tokens "
                f"({os.path.getsize(bin_path) / 1048576:.1f} MB, dtype={_DTYPE.__name__}).")
    return np.memmap(bin_path, dtype=_DTYPE, mode="r")
