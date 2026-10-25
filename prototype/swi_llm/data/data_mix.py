"""
swi_llm/data/data_mix.py

事前学習コーパスのデータミックス構築（ブループリント §1.5）。

複数ソース（Wikipedia-ja / arxiv / dolly-ja）を目標比率で混合し、
カリキュラム学習（短文→長文）とアニーリング（終盤に高品質データ集中投入）に
対応したドキュメント列を返す。

- build_mixed_documents(max_docs): (train_docs, val_docs) を返す。
  train_docs は curriculum=True のとき「短文→長文 + 末尾に高品質データ」順。
"""
import os
import json
import random

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    PRETRAIN_DATA_MIX, PRETRAIN_CURRICULUM, PRETRAIN_ANNEAL_RATIO,
    PRETRAIN_HIGH_QUALITY_SOURCES, PRETRAIN_MAX_DOCS_PER_SOURCE,
    DATASET_WIKIPEDIA_CLEANED, DATASET_WIKIPEDIA_CLEANED_PARQUET,
    DATASET_WIKIPEDIA_JA, DATASET_DOLLY_JA,
    DATASET_CODE_CANDIDATES, DATASET_WEB_CANDIDATES, DATASET_DIALOGUE_CANDIDATES,
    DATASET_BOOK_CANDIDATES, PIPELINE_MIN_DOC_LENGTH,
)

logger = setup_logger("data.data_mix")

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_ARXIV_DIR = os.path.join(_DATA_DIR, "arxiv_papers")

# 再現性のための固定シード
_SEED = 42


def _collect_wikipedia(limit):
    """優先順: Parquet シャード → cleaned_wikipedia.jsonl → HF wikipedia-ja。

    ステップ5 (fix 03): Parquet シャード（advanced_pipeline の出力）はシャード単位で
    ストリーム読みでき、TB 級でも全量をメモリへ載せない。
    """
    docs = []
    try:
        from swi_llm.data.corpus_io import has_parquet_corpus, iter_parquet_texts
        if has_parquet_corpus(DATASET_WIKIPEDIA_CLEANED_PARQUET):
            for text in iter_parquet_texts(DATASET_WIKIPEDIA_CLEANED_PARQUET):
                text = (text or "").strip()
                if len(text) >= PIPELINE_MIN_DOC_LENGTH:
                    docs.append(text)
                    if len(docs) >= limit:
                        return docs
            if docs:
                return docs
    except Exception as e:
        logger.warning(f"Parquet corpus unavailable ({e}); falling back to JSONL/HF.")
        docs = []
    if os.path.exists(DATASET_WIKIPEDIA_CLEANED):
        with open(DATASET_WIKIPEDIA_CLEANED, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    text = json.loads(line).get("text", "").strip()
                except json.JSONDecodeError:
                    continue
                if len(text) >= PIPELINE_MIN_DOC_LENGTH:
                    docs.append(text)
                    if len(docs) >= limit:
                        return docs
    if not docs:
        try:
            from datasets import load_dataset
            ds = load_dataset(DATASET_WIKIPEDIA_JA, split=f"train[:{limit}]")
            for row in ds:
                text = str(row.get("text", "")).strip()
                if len(text) >= PIPELINE_MIN_DOC_LENGTH:
                    docs.append(text)
        except Exception as e:
            logger.warning(f"Could not load wikipedia source ({e}).")
    return docs[:limit]


def _collect_arxiv(limit):
    """arxiv_papers 配下の .txt を 1 ファイル = 1 ドキュメントとして収集。"""
    import glob
    files = glob.glob(os.path.join(_ARXIV_DIR, "**", "*.txt"), recursive=True)
    rng = random.Random(_SEED)
    rng.shuffle(files)
    docs = []
    for path in files[: limit * 2]:  # 空ファイル等を見越して多めに走査
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if len(text) >= PIPELINE_MIN_DOC_LENGTH:
                docs.append(text)
                if len(docs) >= limit:
                    break
        except Exception:
            continue
    return docs


def _format_dialogue_row(row):
    """対話行を 1 ドキュメントに整形。instruction/input/output 形式を優先し、
    無ければ text フィールドを使う（データセットごとのスキーマ差を吸収）。"""
    if row.get("output") or row.get("instruction"):
        parts = [str(row.get("instruction", "")).strip()]
        if str(row.get("input", "")).strip():
            parts.append(str(row["input"]).strip())
        parts.append(str(row.get("output", "")).strip())
        return "\n".join(p for p in parts if p)
    return str(row.get("text", "")).strip()


def _collect_dolly(limit):
    """対話・指示応答コーパス (§1.5)。dolly-ja を主軸に、追加候補（oasst-ja 等）を
    上乗せして多様化する。全体で limit 件に達するまで各ソースから順に収集する。"""
    docs = []
    try:
        from datasets import load_dataset
    except Exception as e:
        logger.warning(f"datasets unavailable for dialogue source ({e}).")
        return docs

    sources = [(DATASET_DOLLY_JA, {})] + [(n, kw) for n, kw in DATASET_DIALOGUE_CANDIDATES]
    for name, kwargs in sources:
        if len(docs) >= limit:
            break
        try:
            ds = load_dataset(name, split="train", **kwargs)
            for row in ds:
                text = _format_dialogue_row(row)
                if len(text) >= PIPELINE_MIN_DOC_LENGTH:
                    docs.append(text)
                    if len(docs) >= limit:
                        break
        except Exception as e:
            logger.warning(f"Could not load dialogue source '{name}' ({e}).")
    return docs[:limit]


def _collect_streaming_candidates(candidates, limit, label):
    """大規模コーパス（コード / Web）の共通コレクタ (§1.5)。

    候補データセットを順に streaming=True で試し、最初に成功したものから
    先頭 limit 件を収集する。streaming により全体ダウンロードを避ける。
    どれもロードできなければ空を返す（他ソースで学習は継続）。
    """
    import itertools
    docs = []
    try:
        from datasets import load_dataset
    except Exception as e:
        logger.warning(f"datasets unavailable for {label} source ({e}).")
        return docs

    for name, kwargs, field in candidates:
        try:
            ds = load_dataset(name, split="train", streaming=True, **kwargs)
            for row in itertools.islice(ds, limit * 3):
                text = str(row.get(field, "")).strip()
                if len(text) >= PIPELINE_MIN_DOC_LENGTH:
                    docs.append(text)
                    if len(docs) >= limit:
                        break
            if docs:
                logger.info(f"{label} source loaded from '{name}' ({len(docs)} docs, streaming).")
                return docs
        except Exception as e:
            logger.warning(f"{label} dataset '{name}' unavailable ({e}). Trying next candidate.")
    if not docs:
        logger.warning(f"No {label} dataset could be loaded. {label} source will be empty.")
    return docs


def _collect_code(limit):
    """専用コードコーパス (§1.5)。"""
    return _collect_streaming_candidates(DATASET_CODE_CANDIDATES, limit, "Code")


def _collect_web(limit):
    """CC/mC4 系 + fineweb-2(jpn) Web テキスト (§1.5 の Web 枠)。流暢さ向上の最大のテコ。"""
    return _collect_streaming_candidates(DATASET_WEB_CANDIDATES, limit, "Web")


def _collect_book(limit):
    """書籍コーパス（青空文庫, §1.5「長文一貫推論の学習に不可欠」）。"""
    return _collect_streaming_candidates(DATASET_BOOK_CANDIDATES, limit, "Book")


_COLLECTORS = {
    "web": _collect_web,
    "wikipedia": _collect_wikipedia,
    "code": _collect_code,
    "book": _collect_book,
    "arxiv": _collect_arxiv,
    "dolly": _collect_dolly,
}


def _allocate_doc_counts(max_docs):
    """max_docs を PRETRAIN_DATA_MIX の比率でソースへ配分する。"""
    if max_docs is None:
        return {src: PRETRAIN_MAX_DOCS_PER_SOURCE for src in PRETRAIN_DATA_MIX}
    counts = {}
    for src, ratio in PRETRAIN_DATA_MIX.items():
        counts[src] = max(1, int(round(max_docs * ratio)))
    return counts


def build_mixed_documents(max_docs, val_ratio=0.05, curriculum=PRETRAIN_CURRICULUM):
    """データミックスした train/val ドキュメント列を返す。

    Args:
        max_docs: 収集するドキュメント総数の目安（None なら各ソース上限まで）。
        val_ratio: 検証用に取り分ける割合。
        curriculum: True なら train を短文→長文順に並べ、末尾に高品質データ
            （PRETRAIN_HIGH_QUALITY_SOURCES）を集中配置する（アニーリング）。

    Returns:
        (train_docs, val_docs): それぞれ str のリスト。
    """
    counts = _allocate_doc_counts(max_docs)
    per_source = {}
    for src, n in counts.items():
        n = min(n, PRETRAIN_MAX_DOCS_PER_SOURCE)
        docs = _COLLECTORS[src](n)
        per_source[src] = docs
        logger.info(f"Data mix: collected {len(docs)} docs from '{src}' (target {n}).")

    rng = random.Random(_SEED)

    # val は各ソースから比率分を held-out（分布を train と揃える）
    val_docs = []
    train_by_source = {}
    for src, docs in per_source.items():
        docs = list(docs)
        rng.shuffle(docs)
        n_val = int(len(docs) * val_ratio)
        val_docs.extend(docs[:n_val])
        train_by_source[src] = docs[n_val:]

    if curriculum:
        # 通常ソース（web 等）を短文→長文に整列 → 末尾に高品質データを集中投入
        general, high_quality = [], []
        for src, docs in train_by_source.items():
            (high_quality if src in PRETRAIN_HIGH_QUALITY_SOURCES else general).extend(docs)
        general.sort(key=len)          # 短文 → 長文（カリキュラム）
        high_quality.sort(key=len)
        # アニーリング区間の目標長に足りなければ general 末尾も high_quality 側へ回す
        train_docs = general + high_quality
        anneal_len = int(len(train_docs) * PRETRAIN_ANNEAL_RATIO)
        logger.info(
            f"Curriculum enabled: {len(general)} general (short->long) + "
            f"{len(high_quality)} high-quality tail (anneal ~{anneal_len} docs)."
        )
    else:
        train_docs = [d for docs in train_by_source.values() for d in docs]
        rng.shuffle(train_docs)

    rng.shuffle(val_docs)
    if not train_docs:
        raise RuntimeError("Data mix produced no training documents. Check sources/network.")
    logger.info(f"Data mix ready: {len(train_docs)} train docs, {len(val_docs)} val docs.")
    return train_docs, val_docs
