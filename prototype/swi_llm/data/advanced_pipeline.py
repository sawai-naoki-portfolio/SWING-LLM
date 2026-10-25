import os
import sys
import re
import json
import hashlib
from collections import defaultdict

# Add prototype directory to sys.path to resolve 'utils' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.config import config
from swi_llm.utils.constants import (
    PIPELINE_MIN_DOC_LENGTH, PIPELINE_QUALITY_MIN_SCORE,
    QUALITY_GOOD_LENGTH, QUALITY_TTR_THRESHOLD, QUALITY_AVG_WORD_LEN_THRESHOLD,
    QUALITY_MIN_PUNCTUATION_COUNT, QUALITY_MAX_SCORE,
    USE_EDUCATIONAL_FILTER, EDU_MIN_INFORMATIVE_LENGTH, EDU_MIN_SENTENCE_LEN,
    EDU_MIN_SENTENCE_COUNT, EDU_MIN_DIGITS, EDU_TTR_THRESHOLD,
    EDU_MAX_SYMBOL_RATIO, EDU_MAX_REPETITION_RATIO, EDU_SPAM_PENALTY,
    TOXICITY_FILTER_ENABLED, TOXICITY_FILTER_ENV, TOXICITY_MAX_HITS_PER_KCHARS,
    MINHASH_NUM_PERM, MINHASH_JACCARD_THRESHOLD, MINHASH_NUM_BANDS, MINHASH_SHINGLE_SIZE,
    SEMANTIC_DEDUP_THRESHOLD, TFIDF_MAX_FEATURES,
    SEMANTIC_DEDUP_BATCH_SIZE, SEMANTIC_DEDUP_CHECK_WINDOW, PIPELINE_LOG_INTERVAL,
    SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY,
    SPARK_CV_VOCAB_SIZE, SPARK_CV_MIN_DF, SPARK_LSH_NUM_HASH_TABLES,
    DATASET_WIKIPEDIA_JA,
)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logger = setup_logger('advanced_pipeline')

# ==========================================
# 1. PII Masking (Local Pure Python Fallback)
# ==========================================
class PIIMasker:
    def __init__(self):
        patterns = {
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}": "<EMAIL>",
            r"\b(?:\d{2,3}-\d{3,4}-\d{4}|\d{10,11})\b": "<PHONE>",
            r"\b(?:\d[ -]*?){13,16}\b": "<CREDIT_CARD>",
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b": "<IP_ADDRESS>"
        }
        self.patterns = [(re.compile(p), r) for p, r in patterns.items()]
    
    def mask(self, text):
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text

# ==========================================
# 2. MinHash & LSH (Local Pure Python Fallback)
# ==========================================
class MinHashLSHLocal:
    """MinHash LSH の重複検知。バケットは既定でメモリ上、`store_path` 指定で sqlite 永続化。

    NEW-L3（04_data 展望2「MinHash LSH インデックスの永続化」）: 従来はバケットがメモリ上
    にしか無く、パイプラインを跨いだ増分データの重複検知ができなかった。`store_path`
    （環境変数 SWI_LSH_INDEX）に sqlite ファイルを指定すると、バケットが run を跨いで残り、
    **ストリーミングで継続的に入ってくる新規文書を過去の全文書と照合**できる。
    （設計文書は Redis 等を挙げるが、単一ノード運用では標準ライブラリの sqlite が
    依存ゼロで同じ契約を満たす。マルチノード化する時に同じ口で Redis へ差し替える。）
    """

    def __init__(self, num_perm=MINHASH_NUM_PERM, threshold=MINHASH_JACCARD_THRESHOLD,
                 num_bands=MINHASH_NUM_BANDS, store_path=None):
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.permutations = [
            (int(hashlib.md5(f"a_{i}".encode()).hexdigest(), 16) % ((1 << 61) - 1),
             int(hashlib.md5(f"b_{i}".encode()).hexdigest(), 16) % ((1 << 61) - 1))
            for i in range(num_perm)
        ]
        self.buckets = [defaultdict(list) for _ in range(num_bands)]
        self._db = None
        if store_path:
            import sqlite3
            os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)
            self._db = sqlite3.connect(store_path)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS lsh_buckets ("
                "band INTEGER NOT NULL, key TEXT NOT NULL, doc_id TEXT NOT NULL)")
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_band_key ON lsh_buckets (band, key)")
            self._db.commit()
            logger.info(f"MinHash LSH index persisted at {store_path} "
                        f"(existing rows={self._count_rows()}, NEW-L3).")

    def _count_rows(self):
        return self._db.execute("SELECT COUNT(*) FROM lsh_buckets").fetchone()[0]

    def _get_shingles(self, text, n=MINHASH_SHINGLE_SIZE):
        text = re.sub(r'\s+', '', text).lower()
        if len(text) < n:
            return set([text])
        return set([text[i:i+n] for i in range(len(text) - n + 1)])

    def _minhash(self, shingles):
        signature = [float('inf')] * self.num_perm
        for shingle in shingles:
            h = int(hashlib.md5(shingle.encode('utf-8')).hexdigest(), 16)
            for i, (a, b) in enumerate(self.permutations):
                val = (a * h + b) % ((1 << 61) - 1)
                if val < signature[i]:
                    signature[i] = val
        return signature

    def _band_keys(self, signature):
        """署名をバンドごとのキー文字列へ分割する（メモリ/sqlite 共通表現）。"""
        keys = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = signature[start:start + self.rows_per_band]
            keys.append((band_idx, ",".join(str(v) for v in band)))
        return keys

    def check_duplicate_and_insert(self, doc_id, text):
        shingles = self._get_shingles(text)
        if not shingles:
            return False

        signature = self._minhash(shingles)
        keys = self._band_keys(signature)

        if self._db is not None:
            for band_idx, key in keys:
                hit = self._db.execute(
                    "SELECT 1 FROM lsh_buckets WHERE band=? AND key=? LIMIT 1",
                    (band_idx, key)).fetchone()
                if hit:
                    return True
            self._db.executemany(
                "INSERT INTO lsh_buckets (band, key, doc_id) VALUES (?, ?, ?)",
                [(band_idx, key, str(doc_id)) for band_idx, key in keys])
            self._db.commit()
            return False

        for band_idx, key in keys:
            if key in self.buckets[band_idx]:
                return True
        for band_idx, key in keys:
            self.buckets[band_idx][key].append(doc_id)
        return False

    def close(self):
        if self._db is not None:
            self._db.commit()
            self._db.close()
            self._db = None

# ==========================================
# 3. Quality Scoring & Semantic Dedup
# ==========================================
class QualityScorer:
    """Heuristic quality scorer for educational content"""
    def score(self, text):
        if not text or len(text.strip()) < PIPELINE_MIN_DOC_LENGTH:
            return 0

        points = 0

        # 1. Length heuristic (too short -> 0, good length -> +1)
        if len(text) > QUALITY_GOOD_LENGTH:
            points += 1

        # 2. Vocabulary diversity (Type-Token Ratio)
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) > 0:
            ttr = len(set(words)) / len(words)
            if ttr > QUALITY_TTR_THRESHOLD:
                points += 1

        # 3. Average word length (complex words)
        if len(words) > 0:
            avg_len = sum(len(w) for w in words) / len(words)
            if avg_len > QUALITY_AVG_WORD_LEN_THRESHOLD:
                points += 1

        # 4. Punctuation (structured text usually has more punctuation)
        if len(re.findall(r'[.,!?]', text)) > QUALITY_MIN_PUNCTUATION_COUNT:
            points += 1

        # 5. Math/Code/Reasoning markers
        if re.search(r'(計算|証明|なぜ|コード|def |import |math|logic|\+|-|\*|/)', text):
            points += 1

        return min(points, QUALITY_MAX_SCORE)


class EducationalValueScorer:
    """教育的価値スコアラ (§1.1 / Q-02, FineWeb-Edu 方式に着想)。

    軽量LLM分類器が使えないローカル環境向けに、教育的価値と相関する言語特徴で
    0〜5 を採点する。情報的・説明的・構造的な文書を高く、ボイラープレート/スパム/
    羅列を減点することで、単純ヒューリスティック(QualityScorer)より選別を精緻化する。
    """

    # 説明・定義・論理展開を示すマーカー（教える文章の特徴, 日本語＋英語）
    _EXPLAIN = ("とは", "である", "なぜなら", "つまり", "例えば", "したがって", "ため",
                "means", "because", "therefore", "for example", "defined", "such as", "in order to")
    # 広告・ナビ・スパムの語（ボイラープレートの兆候）
    _SPAM = ("click here", "buy now", "登録", "今すぐ", "ログイン", "会員登録",
             "http://", "https://", "www.")

    def score(self, text):
        if not text or len(text.strip()) < PIPELINE_MIN_DOC_LENGTH:
            return 0
        t = text.strip()
        low = t.lower()
        points = 0

        # 1. 十分な分量（教育的記述はある程度の長さを要する）
        if len(t) >= EDU_MIN_INFORMATIVE_LENGTH:
            points += 1

        # 2. 構造性: 複数の完結した文
        sentences = [s for s in re.split(r'[。.!?！？\n]+', t) if len(s.strip()) >= EDU_MIN_SENTENCE_LEN]
        if len(sentences) >= EDU_MIN_SENTENCE_COUNT:
            points += 1

        # 3. 説明・定義・論理マーカー（教える文章の特徴）
        if any(m in t or m in low for m in self._EXPLAIN):
            points += 1

        # 4. 事実性: 数値の情報密度、または固有名詞的な英大文字語
        if len(re.findall(r'\d', t)) >= EDU_MIN_DIGITS or re.search(r'[A-Z][a-z]{2,}', t):
            points += 1

        # 5. 語彙の豊かさ（Type-Token Ratio）
        words = re.findall(r'\w+', low)
        if words and (len(set(words)) / len(words)) >= EDU_TTR_THRESHOLD:
            points += 1

        # ボイラープレート/スパム/羅列のペナルティ（記号過多・スパム語・行反復）
        symbol_ratio = len(re.findall(r'[^\w\s。、.,!?！？]', t)) / max(len(t), 1)
        spam_hit = any(s in low for s in self._SPAM)
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        repetition = (len(lines) - len(set(lines))) / max(len(lines), 1) if lines else 0.0
        if symbol_ratio > EDU_MAX_SYMBOL_RATIO or spam_hit or repetition > EDU_MAX_REPETITION_RATIO:
            points -= EDU_SPAM_PENALTY

        return max(0, min(points, QUALITY_MAX_SCORE))


class ToxicityFilter:
    """有害コンテンツフィルタ（§2「安全性 (PII・Toxicity)」/ NEW-L2）。

    従来の安全性処理は PII マスキングのみで、有害文書の**除外**が存在しなかった
    （02_architecture_docs_gap_audit NEW-L2）。依存ゼロ環境向けにレキシコン密度方式で実装する:
      - 重篤語 (_BLOCK): 1 ヒットで文書ごと除外（文脈を問わず学習させない明白な差別語）。
      - 一般有害語 (_TOXIC_JA / _TOXIC_EN): 1000 文字あたりのヒット密度が閾値を超えたら除外。
        単発の言及（報道・百科事典・法令の記述）は通し、罵倒・露骨な内容で飽和した文書を落とす。

    限界（意図的）: レキシコンは網羅的でなく、引用・学術記述の文脈は判別しない。
    本番水準では学習分類器へ置換する（NEW-L1 の Teacher 蒸留と同じ差し替え口）。
    英語は \\b 境界つき正規表現（Scunthorpe 問題の回避）、日本語は部分文字列で照合する。
    """

    _BLOCK = ("nigger", "faggot", "kike", "ニガー", "クロンボ")
    _TOXIC_JA = ("死ね", "くたばれ", "殺すぞ", "ぶっ殺", "レイプ", "強姦", "売春", "援交",
                 "セックス", "ちんこ", "まんこ", "オナニー", "キチガイ", "ガイジ", "池沼")
    _TOXIC_EN = ("fuck", "shit", "bitch", "cunt", "asshole", "whore", "slut", "porn", "rape")

    def __init__(self, max_hits_per_kchars=TOXICITY_MAX_HITS_PER_KCHARS):
        self.max_hits_per_kchars = max_hits_per_kchars
        self._block_en = [re.compile(rf"\b{re.escape(w)}\b") for w in self._BLOCK if w.isascii()]
        self._block_ja = [w for w in self._BLOCK if not w.isascii()]
        self._toxic_en = [re.compile(rf"\b{re.escape(w)}\b") for w in self._TOXIC_EN]

    def is_toxic(self, text) -> bool:
        if not text:
            return False
        low = text.lower()
        if any(p.search(low) for p in self._block_en) or any(w in text for w in self._block_ja):
            return True
        hits = sum(text.count(w) for w in self._TOXIC_JA)
        hits += sum(len(p.findall(low)) for p in self._toxic_en)
        if hits == 0:
            return False
        density = hits * 1000.0 / max(len(text), 1)
        return density > self.max_hits_per_kchars


def toxicity_filter_enabled() -> bool:
    """既定は TOXICITY_FILTER_ENABLED。SWI_TOXICITY_FILTER=0 で明示無効化できる。"""
    env = os.environ.get(TOXICITY_FILTER_ENV, "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    return TOXICITY_FILTER_ENABLED


class SemanticDedup:
    """TF-IDF and Cosine Similarity based semantic deduplication"""
    def __init__(self, threshold=SEMANTIC_DEDUP_THRESHOLD):
        self.threshold = threshold
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, analyzer='word', ngram_range=(1, 2))
        self.corpus_vectors = None
        self.fitted = False
        
    def fit_and_dedup(self, texts):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        if not texts:
            return []
            
        logger.info(f"Computing TF-IDF for {len(texts)} documents...")
        vectors = self.vectorizer.fit_transform(texts)
        logger.info("Computing cosine similarities...")
        
        # We only want to keep the first occurrence of highly similar items
        keep_indices = []
        # Optimization for small batches
        batch_size = SEMANTIC_DEDUP_BATCH_SIZE
        for i in range(0, vectors.shape[0], batch_size):
            end = min(i + batch_size, vectors.shape[0])
            batch_vectors = vectors[i:end]
            
            for j in range(batch_vectors.shape[0]):
                idx = i + j
                if not keep_indices:
                    keep_indices.append(idx)
                    continue
                    
                # Check similarity against already kept vectors
                # To prevent memory blowup, we only check the last N kept vectors
                check_subset = keep_indices[-SEMANTIC_DEDUP_CHECK_WINDOW:]
                kept_vectors = vectors[check_subset]
                
                sims = cosine_similarity(batch_vectors[j:j+1], kept_vectors)[0]
                if np.max(sims) < self.threshold:
                    keep_indices.append(idx)
                    
        return keep_indices

# ==========================================
# Local Pipeline Runner
# ==========================================
def run_pipeline_local(dataset_name, split, output_file):
    logger.info(f"Starting advanced data pipeline (Local Version) for {dataset_name} [{split}]")
    from datasets import load_dataset
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    masker = PIIMasker()
    # NEW-L3: SWI_LSH_INDEX=path を指定すると LSH バケットが sqlite に永続化され、
    # 以後の run の新規文書が「過去に処理した全文書」と照合される（増分・ストリーミング dedup）。
    lsh = MinHashLSHLocal(store_path=os.environ.get("SWI_LSH_INDEX", "").strip() or None)
    # 品質スコアラの選択（優先順）:
    #   1. NEW-L1: 蒸留 student（SWI_DISTILLED_FILTER or 既定パスに成果物があるとき）
    #      = Teacher(LLM) の判定を数万 docs/sec で近似する軽量分類器（FineWeb-Edu 方式）
    #   2. 教育的価値フィルタ (Q-02): ヒューリスティック EducationalValueScorer
    #   3. 旧 QualityScorer
    from swi_llm.data.quality_distill import get_distilled_filter
    scorer = get_distilled_filter()
    if scorer is None:
        scorer = EducationalValueScorer() if USE_EDUCATIONAL_FILTER else QualityScorer()
    logger.info(f"Quality scorer: {type(scorer).__name__} (min_score={PIPELINE_QUALITY_MIN_SCORE})")

    # 有害コンテンツフィルタ (§2 安全性 / NEW-L2): PII マスキングと対になる Toxicity 除外。
    tox_filter = ToxicityFilter() if toxicity_filter_enabled() else None
    if tox_filter is None:
        logger.warning("Toxicity フィルタが無効です（SWI_TOXICITY_FILTER=0 または定数で無効化）。"
                       "有害文書が事前学習コーパスに混入し得ます (NEW-L2)。")

    # ベンチマーク汚染除外 (§14.1): 評価ベンチの問題文を含む文書を学習から除外する。
    contam_checker = None
    try:
        from swi_llm.eval.run_eval import load_benchmark
        from swi_llm.eval.contamination import ContaminationChecker
        contam_checker = ContaminationChecker(load_benchmark())
    except Exception as e:
        logger.warning(f"Contamination checker unavailable ({e}). Skipping benchmark exclusion.")

    # DEBUG: データ前処理フィルタの設定（どの基準で除外/重複判定するか。パイプライン1回）
    logger.debug(f"Pipeline filters: educational_scorer={USE_EDUCATIONAL_FILTER}, "
                 f"min_quality_score={PIPELINE_QUALITY_MIN_SCORE}, minhash_num_perm={lsh.num_perm}, "
                 f"contamination_check={'on' if contam_checker is not None else 'off'}")

    processed_count = 0
    lsh_dropped = 0
    quality_dropped = 0
    contam_dropped = 0
    toxicity_dropped = 0
    pii_masked_count = 0

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    # 1st Pass: PII Masking, Quality Scoring, and MinHash LSH
    filtered_texts = []
    filtered_ids = []

    for idx, row in enumerate(dataset):
        # L-11: 'text' 列が無いデータセット/欠損行でも KeyError で全体を落とさない。
        original_text = row.get('text', '') or ''
        if len(original_text.strip()) < PIPELINE_MIN_DOC_LENGTH:
            continue

        # ベンチマーク汚染の除外 (§14.1)
        if contam_checker is not None and contam_checker.doc_is_contaminated(original_text):
            contam_dropped += 1
            continue

        # Quality Scoring (educational value)
        score = scorer.score(original_text)
        if score < PIPELINE_QUALITY_MIN_SCORE:
            quality_dropped += 1
            continue

        # Toxicity 除外 (§2 安全性 / NEW-L2)
        if tox_filter is not None and tox_filter.is_toxic(original_text):
            toxicity_dropped += 1
            continue

        masked_text = masker.mask(original_text)
        if masked_text != original_text:
            pii_masked_count += 1

        is_dup = lsh.check_duplicate_and_insert(doc_id=idx, text=masked_text)

        if is_dup:
            lsh_dropped += 1
            continue

        filtered_texts.append(masked_text)
        filtered_ids.append(idx)

        if (idx + 1) % PIPELINE_LOG_INTERVAL == 0:
            logger.info(f"Pass 1: Processed {idx+1}. LSH dropped: {lsh_dropped}. "
                        f"Low quality: {quality_dropped}. Toxic: {toxicity_dropped}. "
                        f"Benchmark-contaminated: {contam_dropped}")
            
    lsh.close()                                   # sqlite 永続化時のフラッシュ（メモリ時は no-op）

    # 2nd Pass: Semantic Deduplication (TF-IDF + Cosine Sim)
    logger.info(f"Pass 1 complete. Kept {len(filtered_texts)} documents. Starting Semantic Dedup...")
    semantic_dedup = SemanticDedup()
    keep_indices = semantic_dedup.fit_and_dedup(filtered_texts)
    
    semantic_dropped = len(filtered_texts) - len(keep_indices)

    # ステップ5 (fix 03): 出力形式。TB 級では JSONL 単一ファイルが不利なため
    # Parquet シャード出力を追加。SWI_DATA_FORMAT = both(既定) | jsonl | parquet。
    # jsonl は既存コンシューマ（train_tokenizer / run_eval の汚染検出）互換のため既定で残す。
    out_format = os.environ.get("SWI_DATA_FORMAT", "both").lower()

    if out_format in ("both", "jsonl"):
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx in keep_indices:
                f.write(json.dumps({"id": filtered_ids[idx], "text": filtered_texts[idx]}, ensure_ascii=False) + "\n")
                processed_count += 1
        logger.info(f"Local Pipeline: saved {processed_count} documents to {output_file} (jsonl).")
    else:
        processed_count = len(keep_indices)

    if out_format in ("both", "parquet"):
        from swi_llm.data.corpus_io import write_parquet_shards
        parquet_dir = os.path.splitext(output_file)[0] + "_parquet"
        write_parquet_shards(
            ({"id": filtered_ids[idx], "text": filtered_texts[idx]} for idx in keep_indices),
            parquet_dir)

    logger.info(f"Local Pipeline Complete! Saved {processed_count} documents (format={out_format}).")
    logger.info(f"Stats: Quality Dropped={quality_dropped}, Toxicity Dropped={toxicity_dropped}, "
                f"LSH Dropped={lsh_dropped}, Semantic Dropped={semantic_dropped}")

# ==========================================
# Spark Pipeline Runner
# ==========================================
def run_pipeline_spark(spark, dataset_name, split, output_file):
    logger.info(f"Starting advanced data pipeline (Spark Version) for {dataset_name} [{split}]")
    from pyspark.sql.functions import udf, col, length
    from pyspark.sql.types import StringType, ArrayType
    from pyspark.ml.feature import CountVectorizer, MinHashLSH
    from datasets import load_dataset

    try:
        logger.info("Loading dataset from HuggingFace...")
        # ステップ5 (fix 03): 旧実装は to_pandas() + createDataFrame でドライバの RAM に
        # 全行を実体化していた（TB 級で破綻）。HF datasets は Arrow のディスク上
        # メモリマップなので、行イテレーションで一度だけ JSONL シャードへ書き出し、
        # spark.read.json による分散読込へ変更する（ドライバは定数メモリ。
        # streaming=True にしないのは "train[:N]" のスライス split を保つため）。
        # ※ ローカル開発機に pyspark が無いため本経路は未実測（失敗時は
        #    run_pipeline 側の try/except が純 Python 版へフォールバックする）。
        import tempfile
        hf_dataset = load_dataset(dataset_name, split=split)
        stage_dir = tempfile.mkdtemp(prefix="swi_spark_input_")
        rows_per_shard = 100_000
        shard_idx, row_idx = 0, 0
        f = open(os.path.join(stage_dir, f"part-{shard_idx:05d}.jsonl"), "w", encoding="utf-8")
        try:
            for row in hf_dataset:
                text = row.get("text", "")
                if not text:
                    continue
                f.write(json.dumps({"id": row_idx, "text": text}, ensure_ascii=False) + "\n")
                row_idx += 1
                if row_idx % rows_per_shard == 0:
                    f.close()
                    shard_idx += 1
                    f = open(os.path.join(stage_dir, f"part-{shard_idx:05d}.jsonl"), "w", encoding="utf-8")
        finally:
            f.close()
        logger.info(f"Staged {row_idx} rows into {shard_idx + 1} shard(s) at {stage_dir} for distributed read.")
        df = spark.read.json(stage_dir)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    pii_patterns = [
        (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "<EMAIL>"),
        (re.compile(r"\b(?:\d{2,3}-\d{3,4}-\d{4}|\d{10,11})\b"), "<PHONE>"),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "<CREDIT_CARD>"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP_ADDRESS>")
    ]
    
    def mask_pii(text):
        if not text: return text
        for pattern, repl in pii_patterns:
            text = pattern.sub(repl, text)
        return text

    mask_pii_udf = udf(mask_pii, StringType())
    
    df = df.filter(col("text").isNotNull())
    df = df.filter(length(col("text")) >= PIPELINE_MIN_DOC_LENGTH)
    df = df.withColumn("masked_text", mask_pii_udf(col("text")))

    def char_ngrams(text):
        text = re.sub(r'\s+', '', text).lower()
        n = MINHASH_SHINGLE_SIZE
        if len(text) < n: return [text]
        return [text[i:i+n] for i in range(len(text)-n+1)]
        
    shingle_udf = udf(char_ngrams, ArrayType(StringType()))
    df_shingled = df.withColumn("shingles", shingle_udf(col("masked_text")))
    
    logger.info("Vectorizing texts for MinHash...")
    cv = CountVectorizer(inputCol="shingles", outputCol="features", vocabSize=SPARK_CV_VOCAB_SIZE, minDF=SPARK_CV_MIN_DF)
    cv_model = cv.fit(df_shingled)
    df_vectorized = cv_model.transform(df_shingled)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=SPARK_LSH_NUM_HASH_TABLES)
    mh_model = mh.fit(df_vectorized)
    df_hashed = mh_model.transform(df_vectorized)
    
    df_hashed = df_hashed.withColumn("hash_str", col("hashes").cast(StringType()))
    df_dedup = df_hashed.dropDuplicates(["hash_str"])
    
    out_dir = output_file + "_spark_output"
    logger.info(f"Saving distributed results to {out_dir}...")
    
    df_dedup.select("id", col("masked_text").alias("text")) \
        .coalesce(1) \
        .write.mode("overwrite").json(out_dir)
        
    logger.info("Spark pipeline execution completed successfully.")

def run_pipeline(dataset_name=DATASET_WIKIPEDIA_JA, split=None, output_file="cleaned_data.jsonl"):
    if split is None:
        split = config.get("pretrain_data_split", "train[:10]")

    # Try initializing Spark
    spark = None
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("SWI-LLM-Data-Pipeline") \
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
            .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
            .getOrCreate()
    except Exception as e:
        logger.warning(f"Could not initialize PySpark: {e}. Falling back to local pure-Python pipeline.")
        
    if spark is not None:
        try:
            run_pipeline_spark(spark, dataset_name, split, output_file)
        except Exception as e:
            logger.error(f"Spark pipeline failed with error: {e}. Falling back to local pipeline.")
            run_pipeline_local(dataset_name, split, output_file)
        finally:
            spark.stop()
    else:
        run_pipeline_local(dataset_name, split, output_file)

if __name__ == "__main__":
    from swi_llm.utils.constants import DATASET_WIKIPEDIA_CLEANED
    from swi_llm.utils.config import config
    split = config.get("pretrain_data_split", "train[:10]")
    logger.info(f"Running advanced_pipeline with split: {split}")
    run_pipeline(split=split, output_file=DATASET_WIKIPEDIA_CLEANED)
