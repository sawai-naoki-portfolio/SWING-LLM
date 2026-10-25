"""
utils/constants.py

Centralized string literals and constants for SWI-LLM.

ChatML 特殊トークン (<|im_start|> 等) は utils/chat_template.py が単一情報源。
本モジュールはそれ以外の定数（データセット名、ツール設定、サンドボックス設定、
Web UI 設定、データパイプライン閾値など）を集約する (REQ-05/06/07)。
"""
import os

from swi_llm.utils.chat_template import TOOL_CALL_START, TOOL_CALL_END, THINK_END

# =====================================================================
# Mode Names
# =====================================================================
MODE_TEST = "TEST"
MODE_SMOKE = "SMOKE"
MODE_PROD = "PROD"
MODE_MAX = "MAX"

# =====================================================================
# Dataset and Split Names
# =====================================================================
DATASET_WIKIPEDIA_CLEANED = "swi_llm/data/cleaned_wikipedia.jsonl"
# ステップ5 (fix 03): 前処理済みコーパスの Parquet シャード版（TB 級スケールの保存形式）。
# data_mix はこちらを優先して読み、無ければ JSONL → HF の順にフォールバックする。
DATASET_WIKIPEDIA_CLEANED_PARQUET = "swi_llm/data/cleaned_wikipedia_parquet"
DATASET_DOLLY_JA = "kunishou/databricks-dolly-15k-ja"
DATASET_FUNCTION_CALLING = "DataPilot/Function-Calling-Japanese-Dataset"
DATASET_HH_RLHF_JA = "kunishou/hh-rlhf-49k-ja"
DATASET_CIFAR10 = "uoft-cs/cifar10"

SPLIT_TRAIN = "train"
SPLIT_TRAIN_10 = "train[:100]"
SPLIT_TRAIN_100 = "train[:1000]"
SPLIT_TRAIN_1000 = "train[:500000]"

# =====================================================================
# Checkpoint Base Names
# =====================================================================
CKPT_PRETRAIN = "pretrain_checkpoint"
CKPT_PRETRAIN_LATEST = "pretrain_checkpoint_latest"
# 時間ベースの定期保存間隔（分）。数週間かかる長期学習でも、落ちたときの損失をこの時間以内に抑える。
# 0 以下で無効（ステップ比率ベースの保存のみ）。環境変数 SWI_CKPT_SAVE_MINUTES で上書き可。
CKPT_SAVE_INTERVAL_MINUTES = 30
CKPT_SFT = "sft_checkpoint"
CKPT_SFT_LATEST = "sft_checkpoint_latest"
CKPT_DPO = "dpo_checkpoint"
CKPT_DPO_LATEST = "dpo_checkpoint_latest"
CKPT_GRPO = "grpo_checkpoint"
CKPT_GRPO_LATEST = "grpo_checkpoint_latest"
CKPT_RFT = "rft_checkpoint"
CKPT_RFT_LATEST = "rft_checkpoint_latest"
CKPT_PRM = "prm_checkpoint"
CKPT_PRM_LATEST = "prm_checkpoint_latest"
CKPT_VISION = "vision_checkpoint"

# =====================================================================
# File Extensions
# =====================================================================
EXT_PT = ".pt"

# =====================================================================
# System Prompts and Messages
# =====================================================================
PROMPT_BUDGET_HIGH = "[System: You have a STRICT thinking budget of {tokens} tokens. Plan your thoughts accordingly and output " + THINK_END + " when done.]\n"
PROMPT_BUDGET_LOW = "[System: You have a small thinking budget of {tokens} tokens. Keep thoughts brief and output " + THINK_END + " quickly.]\n"

# =====================================================================
# Error Messages
# =====================================================================
ERR_TOOL_TIMEOUT = "Tool execution timed out after {timeout_sec} seconds."
ERR_TOOL_EXECUTION = "Execution error: {error}"
ERR_TOOL_SILENT = "Tool execution failed silently."

# =====================================================================
# Tool Names (REQ-05)
# =====================================================================
TOOL_GET_WEATHER = "get_weather"
TOOL_CALCULATE = "calculate"
TOOL_EXECUTE_PYTHON = "execute_python"
TOOL_SEARCH_DOCUMENTS = "search_documents"
TOOL_WIKIPEDIA_SEARCH = "wikipedia_search"   # NEW-L6: Web 検索（Wikipedia API）

# --- wikipedia_search (NEW-L6 / 03_agentic 展望2) ---
# ホストはテンプレート固定 + 言語コード許可リストで SSRF を防ぐ（任意 URL は組み立てない）。
WIKIPEDIA_API_URL_TEMPLATE = ("https://{lang}.wikipedia.org/w/api.php"
                              "?action=query&list=search&srsearch={query}"
                              "&format=json&utf8=1&srlimit={limit}")
WIKIPEDIA_ALLOWED_LANGS = ("ja", "en")
WIKIPEDIA_API_TIMEOUT_SEC = 8
WIKIPEDIA_MAX_RESULTS = 3
WIKIPEDIA_SNIPPET_MAX_CHARS = 300

# =====================================================================
# Tool / Sandbox Settings (REQ-05, REQ-20, REQ-21)
# =====================================================================
# ツール実行全体のタイムアウト（ToolExecutor が強制する上限）
TOOL_TIMEOUT_SEC = 10
# execute_python 内のスクリプト実行タイムアウト（Docker / サブプロセス共通）
PYTHON_EXEC_TIMEOUT_SEC = 5
# Docker サンドボックス設定
DOCKER_SANDBOX_IMAGE = "python:3.9-slim"
DOCKER_SANDBOX_MEM_LIMIT = "128m"
# SEC-5: Docker サンドボックスのハードニング（多層防御）。network_disabled + mem_limit に加え、
# 非root実行・全ケーパビリティ剥奪・権限昇格禁止・PID/CPU 制限・rootfs 読取専用を課す。
# これにより、静的検査を回避したコードがコンテナ内で実行されても、コンテナ脱出・資源枯渇・
# ホストマウント汚染の被害を最小化する。
DOCKER_SANDBOX_USER = "65534:65534"          # nobody:nogroup（非root）
DOCKER_SANDBOX_PIDS_LIMIT = 64               # フォークボム対策（プロセス数上限）
DOCKER_SANDBOX_NANO_CPUS = 1_000_000_000     # CPU 上限（1e9 nanoCPU = 1 コア）
DOCKER_SANDBOX_CAP_DROP = ("ALL",)           # 全 Linux ケーパビリティを剥奪
DOCKER_SANDBOX_SECURITY_OPT = ("no-new-privileges",)  # setuid 等での権限昇格を禁止
# calculate ツールの式の最大長
CALC_MAX_EXPR_LEN = 50
# ツール実行結果をプロンプトに埋め込む際の最大文字数
TOOL_RESPONSE_TRUNCATE_LEN = 150
# ReAct ループでツール呼び出しの連続失敗を許容する回数
MAX_REACT_RETRIES = 3
# 天気ツール (wttr.in)
WEATHER_API_URL_TEMPLATE = "https://wttr.in/{location}?format=3"
WEATHER_API_TIMEOUT_SEC = 5

# AST フォールバック・サンドボックスの「許可」モジュール（ホワイトリスト方式）。
# ここに無いモジュールの import は全て拒否される (REQ-20)。
# SEC-1: `operator`（attrgetter/methodcaller）と `string`（Formatter.get_field）は
# 文字列ベースの属性トラバースで AST の dunder 静的検査を回避し、脱出（RCE）に悪用
# できるためホワイトリストから除外した（PoC 実証済み）。
SANDBOX_ALLOWED_MODULES = frozenset({
    "math", "cmath", "random", "statistics", "decimal", "fractions", "numbers",
    "datetime", "calendar", "time",
    "json", "re", "textwrap", "unicodedata",
    "itertools", "functools", "collections", "heapq", "bisect",
    "array", "copy", "enum", "dataclasses", "typing", "abc",
})

# サンドボックス内で呼び出しを禁止する組み込み関数 (REQ-20, REQ-21)
SANDBOX_FORBIDDEN_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "breakpoint", "exit", "quit", "memoryview",
})


def code_exec_tool_enabled():
    """SEC-6: モデル出力（プロンプトインジェクション経路）由来の `execute_python` を
    ReAct ループのツールとして登録・実行してよいかを返す。既定は False。

    ユーザーは自由文プロンプトでモデルに任意コードを生成・実行させられる（SEC-1 の起爆経路）。
    そのため Web 提供経路では既定でコード実行ツールを登録せず、信頼できる環境でのみ
    環境変数 SWI_ENABLE_CODE_EXEC_TOOL=1 で明示的にオプトインさせる。SEC-1 の
    SWI_ALLOW_UNSANDBOXED_EXEC（Docker 不在時に隔離なし実行を許すか）とは独立で、
    両者とも既定オフにすることで多層防御を成す。呼び出し毎に環境を読むため、
    プロセス起動後の切替（テスト含む）にも追従する。
    """
    return os.environ.get("SWI_ENABLE_CODE_EXEC_TOOL", "").strip().lower() in ("1", "true", "yes")

# =====================================================================
# Web UI / Serving Settings (REQ-05, REQ-06)
# =====================================================================
# 既定は localhost バインド（execute_python 等を無認証で LAN へ公開しないため）。
# LAN 公開が必要な場合のみ環境変数 SWI_WEB_HOST=0.0.0.0 で明示的に許可する。
WEB_HOST = os.environ.get("SWI_WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("SWI_WEB_PORT", "5000"))

# --- SEC-3: リクエスト境界の防御（Host/Origin 検証 + 任意トークン認証）---
# Host 検証の既定許可ホスト（DNS リバインディング遮断）。SWI_WEB_HOST が localhost 以外
# （LAN 公開）の場合はその値も許可し、追加は SWI_ALLOWED_HOSTS（カンマ区切り, "*" で無効化）で指定。
WEB_BASE_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})
# 任意の API トークン。設定時のみ /api/* に `Authorization: Bearer <token>` or `X-API-Token` を要求。
# 既定（未設定）はローカル開発の利便のため無認証（Host/Origin 検証は常に有効）。
WEB_API_TOKEN = os.environ.get("SWI_API_TOKEN", "").strip()

# --- SEC-4: フィードバック収集の入力検証・レート制限・サイズ上限 ---
MAX_FEEDBACK_FIELD_CHARS = 4000          # query/response 各フィールドの最大文字数
MAX_FEEDBACK_FILE_BYTES = 50 * 1024 * 1024   # feedback.jsonl の総サイズ上限（ディスク枯渇DoS防止）
FEEDBACK_RATE_LIMIT_PER_MIN = 30         # 1分あたりの許容フィードバック件数（プロセス共有）

# --- SEC-12 / WebUI ロードマップ step5: 生成系エンドポイントのレート制限 ---
# 従来レート制限は /api/feedback にしか無く、GPU を消費する生成系 6 経路
# (/api/generate, /api/generate_stream, /api/generate_image, /api/tts, /api/tts_stream,
#  /ws/voice-stream) は無制限だった。MAX_CONCURRENT_INFERENCE のセマフォは「同時実行数」の
# 制御であって「流量」の制御ではないため、待ち行列を無限に伸ばせた（DoS 面）。
# 方式: クライアント（トークン or IP）ごとのトークンバケット。容量=burst / 補充速度=per_min/60。
RATE_LIMIT_ENABLED = os.environ.get("SWI_RATE_LIMIT", "1").strip().lower() not in ("0", "false", "no")
# クラス別の (1分あたり許容数, バースト). コストの重い経路ほど絞る。
#
# burst の決め方（2026-07-12 実測に基づく）: E2E スイート（tests/webui_test.py）は
# 正規利用でありながら generate 系を **20 回**叩く（基本3 + ペルソナ2×ツール4 + CoT1 +
# 異常系2 + 不正画像1 + 並列4 + Best-of-N1）。burst をこれ以下にすると、攻撃ではなく
# 正規のヘビー利用が 429 で落ちる。よって burst は「正規のバースト利用（=テスト）を通す」値に置き、
# **持続レート（per_min）で洪水を止める**設計にする。
RATE_LIMIT_RULES = {
    "generate": (60, 30),    # LLM 推論（ストリーム含む）。E2E の 20 連射を通しつつ、持続 1req/s で頭打ち
    "image":    (10, 4),     # 拡散モデル生成: 最も重い
    "tts":      (60, 30),    # 音声合成 / 音声 WS（barge-in で再接続が続くため burst は広め）
    "default":  (300, 60),   # status 等の軽量 API（/api/*, /ws/* のその他）
}
# 追跡するクライアント鍵の上限。多数の IP からの流入でメモリを食い潰されないよう LRU で打ち切る。
RATE_LIMIT_MAX_KEYS = 4096
# リバースプロキシ（WebUI step3）導入後は X-Forwarded-For を信頼してよいが、
# 既定は False。プロキシ無しで信頼するとヘッダ偽装でレート制限を自明に回避できる。
TRUST_PROXY_HEADERS = os.environ.get("SWI_TRUST_PROXY_HEADERS", "").strip().lower() in ("1", "true", "yes")
# リクエストボディの上限（Content-Length）。画像入力（MAX_IMAGE_BASE64_CHARS=8M 文字）+
# プロンプト + 履歴が収まる大きさ。parse_images は先頭 1 枚しか使わないため、
# images 配列に巨大な base64 を並べる攻撃はここで弾く。
MAX_REQUEST_BODY_BYTES = 12 * 1024 * 1024

# OOM 防止のための同時推論ストリーム数上限
MAX_CONCURRENT_INFERENCE = 2
# 推論用スレッドプールのワーカー数
INFERENCE_THREAD_POOL_WORKERS = 4
# 受け付けるプロンプトの最大文字数
MAX_PROMPT_LENGTH = 4000
# 会話履歴 (history) 全体で受け付ける最大文字数（プロンプト制限の回避=OOM/DoS 対策）
MAX_HISTORY_CHARS = 16000
# 画像入力の上限（decompression bomb 対策）: base64 文字数と復号後の総ピクセル数
MAX_IMAGE_BASE64_CHARS = 8_000_000            # ~6MB 相当の base64
MAX_IMAGE_PIXELS = 24_000_000                 # 復号後の幅×高さ上限（約 6000x4000）
# prefix キャッシュ（KV 再利用）の LRU 上限プロンプト数。無制限増加による OOM を防ぐ。
PREFIX_CACHE_MAX_ENTRIES = 32
# KV キャッシュ確保時に「プロンプト長+生成上限」へ加える余裕（ツール結果/思考フェーズの追記分）。
# 128k 文脈をフル確保せず、実際に必要な長さ+余裕だけを確保してメモリを節約する。
KV_CACHE_EXTRA_MARGIN = 4096
# L-12: repetition penalty を適用する直近トークン窓。生成が長くなるほど set(generated) 全走査は
# 線形に重くなるため直近 N トークンのみを対象にする（遠い過去の反復抑制は不要かつ品質面でも妥当）。
REPETITION_PENALTY_WINDOW = 256
# 画像のみのリクエスト時に補うデフォルトプロンプト
DEFAULT_IMAGE_PROMPT = "この画像について説明してください。"
# マルチターン会話履歴で推論に引き継ぐ直近ターン数 (§12.3)。
# 作業記憶(コンテキスト)の肥大とレイテンシを抑えるため上限を設ける。
MAX_HISTORY_TURNS = 8

# ツール使用方法のシステムプロンプト注入文（学習・推論で共通）
TOOL_INSTRUCTIONS = (
    "\n\n[Tools]\n"
    f"To use tools, output JSON in {TOOL_CALL_START}.\n"
    "EX:\n"
    f"{TOOL_CALL_START}{{\"name\": \"calculate\", \"arguments\": {{\"expression\": \"1+1\"}}}}{TOOL_CALL_END}\n"
)

# ペルソナ定義（学習 (SFT) と推論 (Web UI) で共通のシステムプロンプト）
PERSONA_JARVIS = "jarvis"
PERSONA_NAVI = "navi"
DEFAULT_PERSONA = PERSONA_JARVIS
PERSONA_SYSTEM_PROMPTS = {
    PERSONA_JARVIS: (
        "You are JARVIS, a highly capable, cool, and logical AI assistant created by Project Zenith.\n"
        "You speak in a polite, concise, and professional Japanese tone "
        "(e.g., '承知いたしました。', '〜と推測されます。')." + TOOL_INSTRUCTIONS
    ),
    PERSONA_NAVI: (
        "You are NAVI, a friendly and energetic AI navigator created by Project Zenith.\n"
        "You speak in a cheerful, casual Japanese tone "
        "(e.g., '〜だよ！', '〜だね！', '任せて！')." + TOOL_INSTRUCTIONS
    ),
}

# 学習（SFT/DPO/GRPO/RFT/PRM）で共通に付与する system プロンプト。
# 推論(Web UI)は選択ペルソナの system を使うため、学習全段階を既定ペルソナ(JARVIS)へ
# 揃えることで「学習と推論でプロンプト書式（system 有無）が食い違う」問題を解消する。
# これを単一情報源とし、各学習ステージはこの定数を参照すること。
TRAINING_SYSTEM_PROMPT = PERSONA_SYSTEM_PROMPTS[DEFAULT_PERSONA]

# 意図検知（ルールベースのツール自動起動）用パターン (REQ-06)
INTENT_CALC_TRIGGER_PATTERN = r"計算|[+*\-/×÷]"
INTENT_NUMBER_PATTERN = r"\d+"
INTENT_CALC_EXPR_PATTERN = r"([0-9\s\+\-\*\/\(\)×÷]{3,})"
INTENT_WEATHER_LOC_PATTERN = r"([a-zA-Z]+|東京|大阪|京都|福岡)の天気"
INTENT_KEYWORD_WEATHER = "天気"
INTENT_KEYWORD_CALC = "計算"
INTENT_KEYWORDS_SEARCH = ("検索", "RAG", "教えて")
INTENT_WEATHER_DEFAULT_LOCATION = "Tokyo"
# 自動実行したツールの結果をユーザーへ直接表示するときの最大文字数（検索結果の全文で画面を埋めない）
INTENT_TOOL_DISPLAY_MAX_CHARS = 600
# CoT 難易度自動判定（高難度と見なすキーワード）
COT_HIGH_DIFFICULTY_PATTERN = r"(計算|証明|なぜ|理由|コード|code|math|logic|分析|解析|詳細)"

# =====================================================================
# Data Pipeline Thresholds (REQ-07)
# =====================================================================
# ドキュメントとして採用する最小文字数
PIPELINE_MIN_DOC_LENGTH = 50
# 教育的価値スコアの採用下限 (0〜5 のうち 3 以上を保持: FineWeb-Edu 方式)
PIPELINE_QUALITY_MIN_SCORE = 3
# 品質スコアリングのヒューリスティック閾値
QUALITY_GOOD_LENGTH = 200
QUALITY_TTR_THRESHOLD = 0.4          # 語彙多様性 (Type-Token Ratio)
QUALITY_AVG_WORD_LEN_THRESHOLD = 4.5  # 平均単語長
QUALITY_MIN_PUNCTUATION_COUNT = 2     # 句読点数
QUALITY_MAX_SCORE = 5
# 教育的価値フィルタ (§1.1 / Q-02, FineWeb-Edu 方式)。True で QualityScorer より
# 精緻な EducationalValueScorer を使用（説明性・情報密度を評価しスパム/羅列を減点）。
USE_EDUCATIONAL_FILTER = True
EDU_MIN_INFORMATIVE_LENGTH = 300      # 教育的記述に必要な最小分量
EDU_MIN_SENTENCE_LEN = 10             # 「文」とみなす最小長
EDU_MIN_SENTENCE_COUNT = 3            # 構造性: 完結文の最小数
EDU_MIN_DIGITS = 3                    # 事実性: 数値の最小出現数
EDU_TTR_THRESHOLD = 0.35             # 語彙多様性の下限
EDU_MAX_SYMBOL_RATIO = 0.15          # 記号過多（ボイラープレート）の上限
EDU_MAX_REPETITION_RATIO = 0.3       # 行の反復（スパム/羅列）の上限
EDU_SPAM_PENALTY = 2                  # スパム/ボイラープレート検出時の減点
# 有害コンテンツフィルタ (§2「PII・Toxicity」/ NEW-L2)。従来は PII のみで Toxicity 除外が
# 無かった。レキシコン密度方式 (advanced_pipeline.ToxicityFilter): 重篤語は 1 ヒットで除外、
# 一般有害語は密度 (ヒット/1000文字) の閾値超過で除外 (単発の言及・報道文体は通す)。
# 本番では学習分類器へ置換する (NEW-L1 の蒸留経路と同じ差し替え口)。
TOXICITY_FILTER_ENABLED = True        # SWI_TOXICITY_FILTER=0 で無効化
TOXICITY_FILTER_ENV = "SWI_TOXICITY_FILTER"
TOXICITY_MAX_HITS_PER_KCHARS = 2.0    # 一般有害語の許容密度 (ヒット / 1000 文字)
# PRM ステップ報酬のラベリング (§8.2 / Q-07)。従来は全ステップ 1.0 固定だったが、
# コントラスト・ラベル（正解=1.0 / 不正解=減衰）で良い/悪いステップを区別させる。
# policy モデルがある環境では prm_mc.estimate_step_rewards_mc で実 MC 推定も可能。
PRM_USE_CONTRASTIVE_LABELS = True     # rejected 軌跡を減衰報酬で PRM 学習に加える
PRM_MC_ROLLOUTS = 4                   # 実 MC 推定時の1ステップあたりロールアウト数
PRM_REJECTED_REWARD_START = 0.5       # 不正解軌跡・先頭ステップの報酬
PRM_REJECTED_REWARD_END = 0.0         # 不正解軌跡・末尾ステップの報酬
# MinHash + LSH（ドキュメントレベル重複排除）
MINHASH_NUM_PERM = 100
MINHASH_JACCARD_THRESHOLD = 0.8
MINHASH_NUM_BANDS = 20
MINHASH_SHINGLE_SIZE = 3
# セマンティック重複排除 (TF-IDF + コサイン類似度)
SEMANTIC_DEDUP_THRESHOLD = 0.9
TFIDF_MAX_FEATURES = 10000
SEMANTIC_DEDUP_BATCH_SIZE = 2000
SEMANTIC_DEDUP_CHECK_WINDOW = 5000
# 進捗ログの出力間隔（処理ドキュメント数）
PIPELINE_LOG_INTERVAL = 500
# Spark 設定
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "4g"
SPARK_CV_VOCAB_SIZE = 10000
SPARK_CV_MIN_DF = 2.0
SPARK_LSH_NUM_HASH_TABLES = 20
# 事前学習用データセット（HuggingFace）
DATASET_WIKIPEDIA_JA = "izumi-lab/wikipedia-ja-20230720"
# コードコーパスの候補（先頭から順に streaming load を試み、最初に成功したものを使用）。
# (dataset_name, load_kwargs, text_field) の形式。streaming=True で全体DLを避ける。
# 注: datasets 5.0 で trust_remote_code（スクリプト方式）は廃止 → Parquet 配布のみ可。
DATASET_CODE_CANDIDATES = [
    ("codeparrot/codeparrot-clean-valid", {}, "content"),
    ("bigcode/the-stack-smol", {"data_dir": "data/python"}, "content"),
]
# Web テキストコーパスの候補（§1.5 の「Web 45-55%」枠 = 流暢さの最大のテコ）。
# fineweb-2(jpn) は教育的にフィルタ済みの高品質 Web（先頭 = 最優先）。
# (dataset_name, load_kwargs, text_field) 形式。すべて Parquet ネイティブ。
DATASET_WEB_CANDIDATES = [
    ("HuggingFaceFW/fineweb-2", {"name": "jpn_Jpan"}, "text"),
    ("range3/cc100-ja", {}, "text"),
    ("allenai/c4", {"name": "ja"}, "text"),
]
# 書籍コーパスの候補（§1.5「長文一貫推論の学習に不可欠」）。青空文庫（パブリックドメイン）。
DATASET_BOOK_CANDIDATES = [
    ("globis-university/aozorabunko-clean", {}, "text"),
]
# 対話・指示応答コーパスの追加候補（§1.5 の「対話 3-15%」枠）。dolly-ja に上乗せして
# 多様化する。instruction/input/output もしくは text フィールドを柔軟に吸収する。
# (dataset_name, load_kwargs) 形式（field はフォーマッタが自動判定するため不要）。
DATASET_DIALOGUE_CANDIDATES = [
    ("kunishou/oasst1-89k-ja", {}),
]

# =====================================================================
# Pretrain データミックス / カリキュラム (ブループリント §1.5)
# =====================================================================
# 事前学習コーパスのソース別混合比率（ドキュメント数ベースの目安）。
# §1.5 の推奨: Web 45-55% / コード20-25% / 学術8-12% / 対話3-15% / 日本語10-15%。
#   - web      : CC/mC4 系 + fineweb-2(jpn) 高品質 Web（§1.5 の Web 枠 = 流暢さの最大のテコ）
#   - wikipedia: 日本語 + 百科事典的テキスト（§1.5 の日本語枠）
#   - code     : 専用コードコーパス（§1.5「コード比率を増やすと数学・論理推論が向上」）
#   - book     : 書籍（青空文庫, §1.5「長文一貫推論の学習に不可欠」）
#   - arxiv    : 学術・技術文書
#   - dolly    : 対話・指示応答（dolly-ja + oasst-ja 等, §1.5 の対話/Q&A 枠）
# あるソースがロードできない環境でも、他ソースの実収集分で学習は継続できる（空許容）。
PRETRAIN_DATA_MIX = {
    "web": 0.30,
    "wikipedia": 0.15,
    "code": 0.20,
    "book": 0.10,
    "arxiv": 0.10,
    "dolly": 0.15,
}
# カリキュラム学習: True で train を短文→長文順に並べ、末尾に高品質データを
# 集中投入する（アニーリング）。§1.5 の「学習初期は短文、終盤に高品質データ」。
PRETRAIN_CURRICULUM = True
# アニーリング区間の割合（学習コーパス末尾のこの割合を高品質データにする）
PRETRAIN_ANNEAL_RATIO = 0.15
# カリキュラム時に「高品質」とみなすソース（アニーリング末尾へ回す）。書籍は長文一貫性が高い。
PRETRAIN_HIGH_QUALITY_SOURCES = ("arxiv", "dolly", "book")
# 1 ソースあたりの最大収集ドキュメント数の安全上限（MAX モードの暴走防止）
PRETRAIN_MAX_DOCS_PER_SOURCE = 200000

# --- トークン予算モード（swi_llm/data/token_corpus.py, roadmap A-1） ---
# SWI_PRETRAIN_TOKENS（例 "4B" / "500M" / "4e9"）か YAML の run.pretrain_token_budget で有効化。
# 文書をメモリに溜めず、ソースをストリーム → トークン化 → ディスク追記する（RAM は予算に比例しない）。
PRETRAIN_TOKENS_ENV = "SWI_PRETRAIN_TOKENS"
# 予算モードの事前学習エポック数（Chinchilla 流の 1 パス）。SWI_PRETRAIN_EPOCHS で上書き可。
PRETRAIN_BUDGET_EPOCHS = 1
# 検証用に取り分ける文書の割合（文書内容のハッシュで決定的に選ぶ）と、その上限トークン数。
PRETRAIN_VAL_FRACTION = 0.005
PRETRAIN_MAX_VAL_TOKENS = 5_000_000
# アニーリング区間（学習末尾の PRETRAIN_ANNEAL_RATIO）で高品質ソースの重みを何倍にするか。
# 汎用データも混ぜ続ける（高品質だけに切り替えると分布が急変して忘却を招く）。
PRETRAIN_ANNEAL_HQ_WEIGHT = 3.0

# =====================================================================
# Tokenizer (SentencePiece BPE, 日本語最適化 / ブループリント §1.4)
# =====================================================================
# 語彙サイズ。小規模モデルでは Embedding/LM Head のコストを抑えるため
# 3.2 万程度が現実的（§1.4 の推奨）。
TOKENIZER_VOCAB_SIZE = 32768
# 学習済みモデルの配置（prototype ディレクトリからの相対パス）
TOKENIZER_MODEL_PATH = "swi_llm/data/tokenizer.model"

# 予約 ID（train_tokenizer.py の学習設定と一致させること）
TOKEN_UNK_ID = 0
TOKEN_BOS_ID = 1
TOKEN_EOS_ID = 2   # 文書終端 (EOT) として使用
TOKEN_PAD_ID = 3

# 語彙上のアトミックトークンとして登録する特殊トークン。
# chat_template.py のタグと一致させること（学習・推論プロンプトで 1 トークン化）。
TOKENIZER_SPECIAL_TOKENS = (
    "<|im_start|>", "<|im_end|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>",
    "<tool_result>", "</tool_result>",
    "<|step|>",
)

# 日本語の主要句読点・記号（単一トークン化して語彙効率を最適化）
TOKENIZER_JA_SYMBOLS = (
    "、", "。", "！", "？", "「", "」", "『", "』",
    "（", "）", "・", "…", "ー", "〜", "：", "；",
)
