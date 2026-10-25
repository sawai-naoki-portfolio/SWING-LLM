"""
swi_llm/data/train_tokenizer.py

SentencePiece BPE トークナイザ（語彙 32,768・日本語最適化）の学習スクリプト
（ブループリント §1.4 準拠）。

生成物: swi_llm/data/tokenizer.model / tokenizer.vocab

設計:
  - model_type=bpe, vocab_size=32768, character_coverage=0.9995（日本語向け）
  - byte_fallback=True（未知 UTF-8 をバイト分解して OOV を出さない）
  - ChatML / 思考 / ツール系の特殊トークンを user_defined_symbols として
    語彙上のアトミックトークンに登録（学習・推論プロンプト中でも 1 トークン化）
  - 日本語の主要句読点・記号も user_defined_symbols に登録し語彙効率を最適化
  - 予約 ID: unk=0, bos=1, eos=2(=EOT), pad=3

コーパス（存在するものを結合。ネットワーク不可なら該当分をスキップ）:
  - swi_llm/data/arxiv_papers/**/*.txt （英語・技術文書）
  - cleaned_wikipedia.jsonl があればそれ、無ければ HuggingFace wikipedia-ja サンプル
  - dolly-ja（対話・指示応答体）

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m swi_llm.data.train_tokenizer
    .venv\\Scripts\\python.exe -m swi_llm.data.train_tokenizer --force   # 既存を上書き再学習
"""
import os
import sys
import json
import glob
import random
import argparse
import tempfile

# Add prototype directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    TOKENIZER_VOCAB_SIZE, TOKENIZER_MODEL_PATH, TOKENIZER_SPECIAL_TOKENS,
    TOKENIZER_JA_SYMBOLS, DATASET_WIKIPEDIA_CLEANED, DATASET_WIKIPEDIA_JA,
    DATASET_DOLLY_JA,
)

logger = setup_logger("train_tokenizer")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ARXIV_DIR = os.path.join(DATA_DIR, "arxiv_papers")

# 学習コーパスの上限（行数）。SentencePiece は C++ で高速だが、
# 巨大コーパスをそのまま食わせると時間・メモリを消費するため上限を設ける。
MAX_LINES_PER_SOURCE = {
    "arxiv": 40000,
    "wikipedia": 30000,
    "dolly": 15000,
}
MIN_LINE_LENGTH = 10


def _iter_arxiv_lines(limit):
    """arxiv_papers 配下の .txt から行を収集する。"""
    files = glob.glob(os.path.join(ARXIV_DIR, "**", "*.txt"), recursive=True)
    random.shuffle(files)
    count = 0
    for path in files:
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= MIN_LINE_LENGTH:
                        yield line
                        count += 1
                        if count >= limit:
                            return
        except Exception as e:
            logger.warning(f"Skipping unreadable file {path}: {e}")


def _iter_wikipedia_lines(limit):
    """cleaned_wikipedia.jsonl があればそれを、無ければ HF wikipedia-ja を使う。"""
    cleaned = os.path.join(os.path.dirname(DATA_DIR), "data", "cleaned_wikipedia.jsonl")
    # DATASET_WIKIPEDIA_CLEANED は 'swi_llm/data/...' 相対。prototype cwd 前提で解決。
    candidate = DATASET_WIKIPEDIA_CLEANED
    count = 0
    if os.path.exists(candidate):
        logger.info(f"Using local cleaned corpus: {candidate}")
        with open(candidate, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    text = json.loads(line).get("text", "").strip()
                except json.JSONDecodeError:
                    continue
                for seg in text.split("\n"):
                    seg = seg.strip()
                    if len(seg) >= MIN_LINE_LENGTH:
                        yield seg
                        count += 1
                        if count >= limit:
                            return
        if count > 0:
            return
    # フォールバック: HuggingFace から少量サンプル
    try:
        from datasets import load_dataset
        logger.info(f"Loading {DATASET_WIKIPEDIA_JA} sample for tokenizer training...")
        ds = load_dataset(DATASET_WIKIPEDIA_JA, split=f"train[:{limit}]")
        for row in ds:
            for seg in str(row.get("text", "")).split("\n"):
                seg = seg.strip()
                if len(seg) >= MIN_LINE_LENGTH:
                    yield seg
                    count += 1
                    if count >= limit:
                        return
    except Exception as e:
        logger.warning(f"Could not load wikipedia-ja ({e}). Skipping Japanese wiki source.")


def _iter_dolly_lines(limit):
    """dolly-ja から指示・応答文を収集（対話体の学習）。"""
    try:
        from datasets import load_dataset
        logger.info(f"Loading {DATASET_DOLLY_JA} sample for tokenizer training...")
        ds = load_dataset(DATASET_DOLLY_JA, split="train")
        count = 0
        for row in ds:
            for key in ("instruction", "input", "output"):
                val = str(row.get(key, "")).strip()
                if len(val) >= MIN_LINE_LENGTH:
                    yield val
                    count += 1
                    if count >= limit:
                        return
    except Exception as e:
        logger.warning(f"Could not load dolly-ja ({e}). Skipping dialogue source.")


def build_corpus(corpus_path):
    """各ソースを結合してシャッフルし、1 行 1 文のコーパスファイルを書き出す。"""
    lines = []
    for src, it in [
        ("arxiv", _iter_arxiv_lines(MAX_LINES_PER_SOURCE["arxiv"])),
        ("wikipedia", _iter_wikipedia_lines(MAX_LINES_PER_SOURCE["wikipedia"])),
        ("dolly", _iter_dolly_lines(MAX_LINES_PER_SOURCE["dolly"])),
    ]:
        before = len(lines)
        lines.extend(it)
        logger.info(f"Collected {len(lines) - before} lines from {src}.")

    if len(lines) < 1000:
        raise RuntimeError(
            f"Corpus too small ({len(lines)} lines). Need arxiv_papers/ or network access "
            "to wikipedia-ja / dolly-ja to train a 32k tokenizer."
        )

    random.shuffle(lines)
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Wrote corpus with {len(lines)} lines to {corpus_path}.")
    return len(lines)


def train(force=False):
    import sentencepiece as spm

    model_prefix = os.path.splitext(TOKENIZER_MODEL_PATH)[0]
    abs_model = os.path.join(os.path.dirname(os.path.dirname(DATA_DIR)), TOKENIZER_MODEL_PATH)
    # TOKENIZER_MODEL_PATH は 'swi_llm/data/tokenizer.model' 相対。prototype cwd 前提。
    if os.path.exists(TOKENIZER_MODEL_PATH) and not force:
        logger.info(f"Tokenizer already exists at {TOKENIZER_MODEL_PATH}. Use --force to retrain.")
        return

    random.seed(42)
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = os.path.join(tmp, "corpus.txt")
        build_corpus(corpus_path)

        logger.info(f"Training SentencePiece BPE: vocab_size={TOKENIZER_VOCAB_SIZE}...")
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            model_type="bpe",
            vocab_size=TOKENIZER_VOCAB_SIZE,
            character_coverage=0.9995,
            byte_fallback=True,
            # このビルドの sentencepiece には nmt_nfkc の charsmap が同梱されて
            # いないため正規化を無効化（原文をそのまま保持。ChatML タグや
            # 日本語の異体字を壊さない利点もある）
            normalization_rule_name="identity",
            # 先頭ダミー空白を付けない（tiktoken 同様、encode を連結安全にする。
            # 学習データは prompt/response を個別に encode して結合するため必須）
            add_dummy_prefix=False,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3,
            unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
            user_defined_symbols=list(TOKENIZER_SPECIAL_TOKENS) + list(TOKENIZER_JA_SYMBOLS),
            input_sentence_size=1_000_000,
            shuffle_input_sentence=True,
            num_threads=os.cpu_count() or 4,
            train_extremely_large_corpus=False,
        )

    logger.info(f"Tokenizer trained. Model: {TOKENIZER_MODEL_PATH}")
    # 検証
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL_PATH)
    logger.info(f"Vocab size: {sp.get_piece_size()}")
    sample = "<|im_start|>user\n計算してください: 12 + 34 は？<|im_end|>\n<|im_start|>assistant\n<think>"
    ids = sp.encode(sample, out_type=int)
    logger.info(f"Round-trip OK: {sp.decode(ids) == sample} ({len(ids)} tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SWI-LLM SentencePiece tokenizer.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing tokenizer.model")
    args = parser.parse_args()
    train(force=args.force)
