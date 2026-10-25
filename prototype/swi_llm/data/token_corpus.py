"""
swi_llm/data/token_corpus.py
トークン予算ベースのストリーミング事前学習コーパス（roadmap A-1 / Phase 1）。

従来の build_mixed_documents は全文書を Python のリストとしてメモリに載せ、ソースごとに
PRETRAIN_MAX_DOCS_PER_SOURCE 件で頭打ちだった。Chinchilla 級（40〜80 億トークン ≈ 16〜32GB の
テキスト）ではメモリに載らず、上限で 5〜10 億トークン程度にしか届かない。

本モジュールは「目標トークン数」を受け取り、
  1. 各ソースをストリームで読み（HF datasets の streaming / ローカルファイル）、
  2. 最小長・PII マスク・Toxicity（既定 ON）を通し、
  3. まとめてトークン化して uint16 のままディスクへ追記する。
メモリに残るのは書き出し前のバッファ（既定 ~1M トークン）だけで、予算に比例しない。

並びと混合:
  - 主区間（予算の 1 - PRETRAIN_ANNEAL_RATIO）は PRETRAIN_DATA_MIX の比率でトークン単位に交互に
    取り出す（逐次読みでも局所的に混ざる）。末尾のアニーリング区間は高品質ソースの重みを
    PRETRAIN_ANNEAL_HQ_WEIGHT 倍にする。学習は先頭から順に読むので、この並びがそのまま効く。
  - 枯渇・取得不能なソースの分は、残りのソースへ比率どおり自動で回る（実際の構成は meta に残す）。
  - 検証データは文書内容のハッシュで決定的に選び（PRETRAIN_VAL_FRACTION、上限
    PRETRAIN_MAX_VAL_TOKENS）、学習側には決して入らない。予算には数えない。

中断と再開:
  ビルドは数時間かかりうる（大半はダウンロード）。書き出しのたびに progress.json を
  アトミックに更新し、次回は同じ設定なら記録位置までファイルを切り詰め、各ソースを消費済み
  件数だけ読み飛ばして続きから書く（中断しない場合とバイト単位で同一の結果になる）。
  設定（予算・混合比・トークナイザ等）が変わったら作り直す。

使い方:
    python -m swi_llm.data.token_corpus --tokens 4B          # 事前に構築
    SWI_PRETRAIN_TOKENS=4B  run.bat llm MAX --stage pretrain   # 学習時に未構築なら構築して使う
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys

import numpy as np

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    PRETRAIN_DATA_MIX, PRETRAIN_ANNEAL_RATIO, PRETRAIN_HIGH_QUALITY_SOURCES,
    PRETRAIN_ANNEAL_HQ_WEIGHT, PRETRAIN_VAL_FRACTION, PRETRAIN_MAX_VAL_TOKENS,
    PIPELINE_MIN_DOC_LENGTH,
    DATASET_WEB_CANDIDATES, DATASET_CODE_CANDIDATES, DATASET_BOOK_CANDIDATES,
    DATASET_WIKIPEDIA_JA, DATASET_WIKIPEDIA_CLEANED, DATASET_WIKIPEDIA_CLEANED_PARQUET,
    DATASET_DOLLY_JA, DATASET_DIALOGUE_CANDIDATES,
)

logger = setup_logger("data.token_corpus")

CORPUS_VERSION = "token-corpus-v1"
DEFAULT_CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".token_corpus")
_DTYPE = np.uint16
_DTYPE_MAX = int(np.iinfo(_DTYPE).max)
_ARXIV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv_papers")
_SEED = 42

TRAIN_FILE = "train.bin"
VAL_FILE = "val.bin"
PROGRESS_FILE = "progress.json"


class SourceUnavailable(RuntimeError):
    """ソースを開けなかった（未導入・ゲート付き・ネットワーク不通）。そのソースは使わずに続行する。"""


# ---------------------------------------------------------------------------
# ソース（open(skip, prefer) -> (iterator of str, origin)）
# ---------------------------------------------------------------------------
def _peeked(iterator):
    """先頭 1 件を取り出してから元の並びで返す（ストリームの遅延エラーを「開く時点」で検出する）。"""
    first = next(iterator)
    return itertools.chain([first], iterator)


def _hf_stream_source(candidates):
    """[(dataset, kwargs, extract(row)->str), ...] を先頭から試すストリームソース。

    prefer（前回使った候補名）があればそれを最初に試す。再開時に別候補へ切り替わると
    「消費済み件数ぶん読み飛ばす」位置がずれるため。
    """
    def open_source(skip, prefer=None):
        try:
            from datasets import load_dataset
        except Exception as e:
            raise SourceUnavailable(f"datasets not installed ({e})")
        ordered = sorted(candidates, key=lambda c: c[0] != prefer) if prefer else list(candidates)
        errors = []
        for name, kwargs, extract in ordered:
            try:
                ds = load_dataset(name, split="train", streaming=True, **kwargs)
                rows = itertools.islice(iter(ds), skip, None)
                return _peeked(extract(row) for row in rows), name
            except StopIteration:
                return iter(()), name                       # 読み飛ばした結果ちょうど末尾
            except Exception as e:
                errors.append(f"{name}: {e}")
        raise SourceUnavailable("; ".join(errors) or "no candidates")
    return open_source


def _field(name):
    return lambda row: str(row.get(name, "") or "")


def _dialogue_text(row):
    if row.get("output") or row.get("instruction"):
        parts = [str(row.get("instruction", "")).strip()]
        if str(row.get("input", "")).strip():
            parts.append(str(row["input"]).strip())
        parts.append(str(row.get("output", "")).strip())
        return "\n".join(p for p in parts if p)
    return str(row.get("text", "") or "")


def _local_wikipedia_texts():
    try:
        from swi_llm.data.corpus_io import has_parquet_corpus, iter_parquet_texts
        if has_parquet_corpus(DATASET_WIKIPEDIA_CLEANED_PARQUET):
            yield from iter_parquet_texts(DATASET_WIKIPEDIA_CLEANED_PARQUET)
            return
    except Exception as e:
        logger.warning(f"Local Parquet wikipedia unavailable ({e}).")
    if os.path.exists(DATASET_WIKIPEDIA_CLEANED):
        with open(DATASET_WIKIPEDIA_CLEANED, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield str(json.loads(line).get("text", "") or "")
                    except json.JSONDecodeError:
                        continue


def _wikipedia_source():
    """全量の HF wikipedia-ja（ストリーム）を優先し、取れなければローカルの前処理済みコーパス。"""
    hf = _hf_stream_source([(DATASET_WIKIPEDIA_JA, {}, _field("text"))])

    def open_source(skip, prefer=None):
        if prefer != "local:wikipedia":
            try:
                return hf(skip, prefer)
            except SourceUnavailable as e:
                if prefer == DATASET_WIKIPEDIA_JA:
                    raise                                   # 再開時は同じ候補でないと位置がずれる
                logger.warning(f"HF wikipedia unavailable ({e}); using the local cleaned corpus.")
        try:
            return _peeked(itertools.islice(_local_wikipedia_texts(), skip, None)), "local:wikipedia"
        except StopIteration:
            if skip:
                return iter(()), "local:wikipedia"
            raise SourceUnavailable("no local wikipedia corpus") from None
    return open_source


def _arxiv_source(root=_ARXIV_DIR):
    def texts(files):
        for path in files:
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    yield f.read()
            except OSError:
                yield ""

    def open_source(skip, prefer=None):
        import glob
        files = sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))
        if not files:
            raise SourceUnavailable(f"no .txt under {root}")
        random.Random(_SEED).shuffle(files)                 # 決定的な並び（再開で同じ順になる）
        return itertools.islice(texts(files), skip, None), "local:arxiv"
    return open_source


def _dialogue_source():
    """dolly-ja → 追加の対話コーパスを順に連結する（候補の切り替えではなく全部使う）。"""
    sets = [(DATASET_DOLLY_JA, {})] + [(n, kw) for n, kw in DATASET_DIALOGUE_CANDIDATES]

    def chained():
        from datasets import load_dataset
        for name, kwargs in sets:
            try:
                ds = load_dataset(name, split="train", streaming=True, **kwargs)
                for row in ds:
                    yield _dialogue_text(row)
            except Exception as e:
                logger.warning(f"Dialogue set '{name}' unavailable ({e}); skipping it.")

    def open_source(skip, prefer=None):
        try:
            import datasets  # noqa: F401
        except Exception as e:
            raise SourceUnavailable(f"datasets not installed ({e})")
        try:
            return _peeked(itertools.islice(chained(), skip, None)), "dialogue-chain"
        except StopIteration:
            if skip:
                return iter(()), "dialogue-chain"
            raise SourceUnavailable("no dialogue dataset could be loaded") from None
    return open_source


def default_sources():
    """PRETRAIN_DATA_MIX のソース名 → open(skip, prefer) の対応。"""
    def candidates(cands):
        return [(name, kwargs, _field(field)) for name, kwargs, field in cands]
    return {
        "web": _hf_stream_source(candidates(DATASET_WEB_CANDIDATES)),
        "wikipedia": _wikipedia_source(),
        "code": _hf_stream_source(candidates(DATASET_CODE_CANDIDATES)),
        "book": _hf_stream_source(candidates(DATASET_BOOK_CANDIDATES)),
        "arxiv": _arxiv_source(),
        "dolly": _dialogue_source(),
    }


# ---------------------------------------------------------------------------
# 構築
# ---------------------------------------------------------------------------
def _stable_unit(text):
    """文書内容から [0, 1) の決定的な値（val 選別用。実行や順序に依存しない）。"""
    return int.from_bytes(hashlib.sha1(text.encode("utf-8", "ignore")).digest()[:8], "big") / 2.0 ** 64


def _encode_many(tokenizer, texts):
    if hasattr(tokenizer, "encode_batch"):
        return tokenizer.encode_batch(texts)
    return [tokenizer.encode(t) for t in texts]


def _atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _phase_weights(mix, phase, high_quality, hq_weight):
    if phase == "main":
        return dict(mix)
    return {s: w * (hq_weight if s in high_quality else 1.0) for s, w in mix.items()}


def _config_fingerprint(token_budget, mix, anneal_ratio, high_quality, hq_weight, val_fraction,
                        max_val_tokens, tokenizer_fp, eot_id, source_names, safety):
    return {
        "version": CORPUS_VERSION, "token_budget": int(token_budget),
        "mix": {k: float(v) for k, v in sorted(mix.items())},
        "anneal_ratio": float(anneal_ratio), "high_quality": sorted(high_quality),
        "anneal_hq_weight": float(hq_weight), "val_fraction": float(val_fraction),
        "max_val_tokens": int(max_val_tokens), "tokenizer": tokenizer_fp, "eot_id": int(eot_id),
        "sources": sorted(source_names), "min_doc_length": PIPELINE_MIN_DOC_LENGTH,
        "safety": bool(safety),
    }


class _Filters:
    """最小長・PII マスク・Toxicity（既存の advanced_pipeline の自作フィルタを流用）。"""

    def __init__(self, enabled):
        self.pii = self.tox = None
        if enabled:
            from swi_llm.data.advanced_pipeline import PIIMasker, ToxicityFilter, toxicity_filter_enabled
            self.pii = PIIMasker()
            self.tox = ToxicityFilter() if toxicity_filter_enabled() else None
            if self.tox is None:
                logger.warning("Toxicity filter disabled (SWI_TOXICITY_FILTER=0): toxic documents "
                               "may enter the pretraining corpus (NEW-L2).")

    def apply(self, text):
        """通すなら整形済みテキスト、落とすなら None。"""
        text = (text or "").strip()
        if len(text) < PIPELINE_MIN_DOC_LENGTH:
            return None
        if self.tox is not None and self.tox.is_toxic(text):
            return None
        if self.pii is not None:
            text = self.pii.mask(text)
        return text


def build_token_corpus(token_budget, out_dir=None, sources=None, mix=None, tokenizer=None,
                       eot_id=None, tokenizer_fp=None, val_fraction=PRETRAIN_VAL_FRACTION,
                       max_val_tokens=PRETRAIN_MAX_VAL_TOKENS, anneal_ratio=PRETRAIN_ANNEAL_RATIO,
                       high_quality=PRETRAIN_HIGH_QUALITY_SOURCES,
                       anneal_hq_weight=PRETRAIN_ANNEAL_HQ_WEIGHT, flush_tokens=1 << 20,
                       batch_docs=64, batch_chars=1 << 19, safety=True, fresh=False):
    """トークン予算ぶんの事前学習コーパスを out_dir に構築し、メタ情報（dict）を返す。

    完成済みで設定が同じなら何もせず返す。途中まで作られていれば続きから再開する。

    Args:
        token_budget: 学習用トークン数の目標（検証データは別枠）。
        sources: {名前: open(skip, prefer)}。省略時は default_sources()（テストでは差し替える）。
        mix: {名前: 比率}。省略時は PRETRAIN_DATA_MIX。
        tokenizer / eot_id / tokenizer_fp: 省略時は SentencePiece トークナイザとその指紋。
        flush_tokens: この量が溜まるたびにディスクへ書き、進捗を保存する（= メモリ上限の目安）。
        batch_docs / batch_chars: 1 回に取り出す文書数・文字数の上限（論文のような長文でも
            1 バッチのメモリと予算の超過を小さく保つ）。
        fresh: True なら既存の構築物を捨てて作り直す。
    """
    token_budget = int(token_budget)
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    out_dir = out_dir or DEFAULT_CORPUS_DIR
    sources = sources if sources is not None else default_sources()
    mix = {s: w for s, w in (mix or PRETRAIN_DATA_MIX).items() if w > 0 and s in sources}
    if not mix:
        raise ValueError("No source in the mix has an opener.")
    if tokenizer is None:
        from swi_llm.data.tokenizer import get_tokenizer, EOT_TOKEN_ID
        tokenizer = get_tokenizer()
        eot_id = EOT_TOKEN_ID if eot_id is None else eot_id
    if eot_id is None:
        raise ValueError("eot_id is required with a custom tokenizer")
    if tokenizer_fp is None:
        from swi_llm.data.pretokenize import tokenizer_fingerprint
        tokenizer_fp = tokenizer_fingerprint()

    cfg = _config_fingerprint(token_budget, mix, anneal_ratio, high_quality, anneal_hq_weight,
                              val_fraction, max_val_tokens, tokenizer_fp, eot_id, mix.keys(), safety)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, TRAIN_FILE)
    val_path = os.path.join(out_dir, VAL_FILE)
    progress_path = os.path.join(out_dir, PROGRESS_FILE)

    state = None
    if not fresh and os.path.exists(progress_path):
        try:
            with open(progress_path, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Unreadable corpus progress ({e}); rebuilding from scratch.")
            state = None
        if state is not None and state.get("config") != cfg:
            logger.info("Corpus settings changed (budget/mix/tokenizer/...); rebuilding from scratch.")
            state = None
    if state is not None and state.get("complete"):
        sizes_ok = (os.path.exists(train_path) and os.path.exists(val_path)
                    and os.path.getsize(train_path) == state["train_tokens"] * 2
                    and os.path.getsize(val_path) == state["val_tokens"] * 2)
        if sizes_ok:
            logger.info(f"Token corpus ready: {state['train_tokens']:,} train / "
                        f"{state['val_tokens']:,} val tokens ({out_dir}).")
            return state
        logger.warning("Token corpus files do not match the recorded sizes; rebuilding.")
        state = None

    if state is None:
        state = {"config": cfg, "complete": False, "train_tokens": 0, "val_tokens": 0,
                 "phase_tokens": {"main": 0, "anneal": 0},
                 "sources": {s: {"consumed": 0, "kept_docs": 0, "dropped_docs": 0, "tokens": 0,
                                 "phase_tokens": {"main": 0, "anneal": 0}, "origin": None,
                                 "status": "pending"} for s in mix}}
        for p in (train_path, val_path):
            open(p, "wb").close()
        logger.info(f"Building token corpus: budget={token_budget:,} tokens, mix={mix} -> {out_dir}")
    else:
        # 前回の最終保存点まで巻き戻す（その後に書かれた中途半端な追記を捨てる）。
        for p, n in ((train_path, state["train_tokens"]), (val_path, state["val_tokens"])):
            with open(p, "ab") as f:
                f.truncate(n * 2)
        logger.info(f"Resuming token corpus build at {state['train_tokens']:,}/{token_budget:,} tokens.")

    filters = _Filters(safety)
    main_target = int(token_budget * (1.0 - anneal_ratio))
    iters = {}

    def open_iter(name):
        st = state["sources"][name]
        try:
            it, origin = sources[name](st["consumed"], st["origin"])
        except SourceUnavailable as e:
            if st["origin"] is not None:
                # 前回使えたソースが開けない = 一時的な障害の可能性。黙って構成を変えずに止める。
                raise RuntimeError(f"Source '{name}' ({st['origin']}) used earlier is unavailable "
                                   f"now ({e}). Re-run to resume once it is reachable.") from e
            logger.warning(f"Source '{name}' unavailable ({e}); its share goes to the other sources.")
            st["status"] = "unavailable"
            return None
        st["origin"] = origin
        st["status"] = "active"
        return it

    for name, st in state["sources"].items():
        if st["status"] in ("pending", "active"):
            it = open_iter(name)
            if it is not None:
                iters[name] = it

    train_buf, val_buf = [], []
    buffered = 0

    def flush():
        nonlocal buffered
        for path, buf in ((train_path, train_buf), (val_path, val_buf)):
            if buf:
                with open(path, "ab") as f:
                    np.concatenate(buf).astype(_DTYPE).tofile(f)
                    f.flush()
                    os.fsync(f.fileno())
                buf.clear()
        buffered = 0
        _atomic_write_json(progress_path, state)

    def next_batch(name):
        """1 バッチを取り出して検証・トークン化まで済ませる。state とバッファはまだ触らない。

        Returns:
            None（枯渇）または生文書ごとの [(整形済み text, uint16 配列), ...]（落とした文書は (None, None)）。
        """
        raw, chars = [], 0
        for text in iters[name]:
            raw.append(text)
            chars += len(text or "")
            if len(raw) >= batch_docs or chars >= batch_chars:
                break
        if not raw:
            return None
        filtered = [filters.apply(text) for text in raw]
        kept = [t for t in filtered if t is not None]
        encoded = iter(_encode_many(tokenizer, kept) if kept else [])
        items = []
        for text in filtered:
            if text is None:
                items.append((None, None))
                continue
            arr = np.asarray(list(next(encoded)) + [int(eot_id)], dtype=np.int64)
            if int(arr.max()) > _DTYPE_MAX or int(arr.min()) < 0:
                raise ValueError(f"Token id out of uint16 range in source '{name}'.")
            items.append((text, arr.astype(_DTYPE)))
        return items

    def commit(name, phase, items):
        """バッチ分をまとめて state とバッファへ反映する（途中で例外が出ても両者がずれない）。

        学習側が予算に達したらそこで打ち切り、以降の文書は「未消費」として残す。
        """
        nonlocal buffered
        st = state["sources"][name]
        for text, arr in items:
            if state["train_tokens"] >= token_budget:
                break
            st["consumed"] += 1
            if arr is None:
                st["dropped_docs"] += 1
                continue
            st["kept_docs"] += 1
            n = int(arr.size)
            if _stable_unit(text) < val_fraction and state["val_tokens"] < max_val_tokens:
                val_buf.append(arr)
                state["val_tokens"] += n
            else:
                train_buf.append(arr)
                state["train_tokens"] += n
                st["tokens"] += n
                st["phase_tokens"][phase] += n
                state["phase_tokens"][phase] += n
            buffered += n

    try:
        while state["train_tokens"] < token_budget:
            phase = "main" if state["train_tokens"] < main_target else "anneal"
            weights = _phase_weights(mix, phase, high_quality, anneal_hq_weight)
            alive = [s for s in iters if weights.get(s, 0) > 0]
            if not alive:
                break
            # 重みあたりの産出量が最も少ないソースから取る（トークン単位の重み付きラウンドロビン）。
            name = min(alive, key=lambda s: state["sources"][s]["phase_tokens"][phase] / weights[s])
            batch = next_batch(name)
            if batch is None:
                st = state["sources"][name]
                st["status"] = "exhausted"
                del iters[name]
                logger.info(f"Source '{name}' exhausted after {st['tokens']:,} tokens.")
                continue
            commit(name, phase, batch)
            if buffered >= flush_tokens:
                flush()
    except BaseException:
        # ネットワーク断・Ctrl+C でも、確定済みのバッチまでは保存して再開できるようにする。
        try:
            flush()
            logger.error(f"Corpus build interrupted at {state['train_tokens']:,} tokens; "
                         "progress saved. Re-run to resume.")
        except Exception:
            logger.exception("Could not save corpus progress after the interruption.")
        raise

    if state["train_tokens"] == 0:
        flush()
        raise RuntimeError("Token corpus build produced no training tokens. Check sources/network.")
    if state["train_tokens"] < token_budget:
        logger.warning(f"All sources exhausted at {state['train_tokens']:,} tokens "
                       f"(budget {token_budget:,}). Training will use what was collected.")
    state["complete"] = True
    state["realized_mix"] = {s: round(v["tokens"] / state["train_tokens"], 4)
                             for s, v in state["sources"].items()}
    flush()
    logger.info(f"Token corpus built: {state['train_tokens']:,} train / {state['val_tokens']:,} val "
                f"tokens. Realized mix: {state['realized_mix']}")
    return state


def load_token_corpus(out_dir=None):
    """完成済みコーパスを (train_memmap, val_memmap, meta) で返す。未完成・不整合なら None。"""
    out_dir = out_dir or DEFAULT_CORPUS_DIR
    progress_path = os.path.join(out_dir, PROGRESS_FILE)
    if not os.path.exists(progress_path):
        return None
    with open(progress_path, encoding="utf-8") as f:
        meta = json.load(f)
    if not meta.get("complete"):
        return None
    train = np.memmap(os.path.join(out_dir, TRAIN_FILE), dtype=_DTYPE, mode="r")
    val_path = os.path.join(out_dir, VAL_FILE)
    val = (np.memmap(val_path, dtype=_DTYPE, mode="r")
           if meta["val_tokens"] > 0 else np.zeros(0, dtype=_DTYPE))
    if len(train) != meta["train_tokens"] or len(val) != meta["val_tokens"]:
        return None
    return train, val, meta


def load_or_build_token_corpus(token_budget, out_dir=None, **kwargs):
    """予算どおりのコーパスを（必要なら構築・再開して）memmap で返す。"""
    build_token_corpus(token_budget, out_dir=out_dir, **kwargs)
    loaded = load_token_corpus(out_dir)
    if loaded is None:
        raise RuntimeError(f"Token corpus at {out_dir or DEFAULT_CORPUS_DIR} is incomplete after build.")
    return loaded


def main(argv=None):
    from swi_llm.utils.config import parse_token_count, resolve_pretrain_token_budget
    ap = argparse.ArgumentParser(description="Build the token-budget pretraining corpus (roadmap A-1).")
    ap.add_argument("--tokens", help="target training tokens, e.g. 4B / 500M / 4e9 "
                                     "(default: SWI_PRETRAIN_TOKENS or run.pretrain_token_budget)")
    ap.add_argument("--out", default=None, help=f"output directory (default {DEFAULT_CORPUS_DIR})")
    ap.add_argument("--fresh", action="store_true", help="discard any partial/complete build first")
    ap.add_argument("--if-configured", action="store_true",
                    help="exit 0 without doing anything when no token budget is configured "
                         "(used by the run.bat 'corpus' stage)")
    args = ap.parse_args(argv)

    budget = parse_token_count(args.tokens) if args.tokens else resolve_pretrain_token_budget()
    if not budget:
        if args.if_configured:
            logger.info("No pretrain token budget configured (SWI_PRETRAIN_TOKENS); skipping corpus build.")
            return 0
        ap.error("no token budget: pass --tokens or set SWI_PRETRAIN_TOKENS")
    meta = build_token_corpus(budget, out_dir=args.out, fresh=args.fresh)
    print(json.dumps({k: meta[k] for k in ("train_tokens", "val_tokens", "realized_mix")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
    sys.exit(main())
