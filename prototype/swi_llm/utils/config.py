"""
utils/config.py

This module centralizes all configurations, hyperparameters, and mode logic.
Replaces both mode_config.py and model_config.py.
"""
import os
import glob

from swi_llm.utils.logger import setup_logger

# setup_logger 経由にして app.log/error.log へ確実に出力し、SWI_LOG_LEVEL=DEBUG で構成解決を追える
# ようにする（従来の logging.getLogger はハンドラ未設定で自動モード検出ログ等が app.log に届かなかった）。
logger = setup_logger("config")

from swi_llm.utils.constants import (
    MODE_TEST, MODE_SMOKE, MODE_PROD, MODE_MAX,
    SPLIT_TRAIN_10, SPLIT_TRAIN_100, SPLIT_TRAIN_1000, SPLIT_TRAIN
)

# =====================================================================
# 1. MODE DETECTION
# =====================================================================
MODE = os.environ.get("SWI_MODE", "").strip().upper()

if not MODE:
    # チェックポイントは prototype 直下 (swi_llm/utils/ から 2 階層上) に置かれる
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # voice_ 等の他機能のテスト時チェックポイントに引っ張られないよう LLM の主要チェックポイントのみを対象にする
    all_pts = glob.glob(os.path.join(base_dir, "*.pt"))
    checkpoints = [f for f in all_pts if any(prefix in os.path.basename(f) for prefix in ["sft_checkpoint", "dpo_checkpoint", "grpo_checkpoint", "pretrain_checkpoint"])]
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getmtime)
        filename = os.path.basename(latest_ckpt)
        if f"_{MODE_TEST}" in filename: MODE = MODE_TEST
        elif f"_{MODE_SMOKE}" in filename: MODE = MODE_SMOKE
        elif f"_{MODE_MAX}" in filename: MODE = MODE_MAX
        else: MODE = MODE_PROD
        logger.info(f"Auto-detected mode {MODE} from latest checkpoint: {filename}")
    else:
        MODE = MODE_TEST
        logger.info(f"No checkpoints found. Defaulting to {MODE_TEST} mode.")

if MODE not in [MODE_TEST, MODE_SMOKE, MODE_PROD, MODE_MAX]:
    logger.warning(f"Unknown mode: {MODE}, falling back to {MODE_TEST} mode")
    MODE = MODE_TEST

logger.debug(f"Resolved SWI_MODE={MODE} (source={'env' if os.environ.get('SWI_MODE','').strip() else 'auto/default'})")

# =====================================================================
# 2. MODEL ARCHITECTURE
# =====================================================================
# 学習時の実効コンテキスト長（長文外挿の基準）。pretrain の seq_len と一致させる。
ROPE_ORIGINAL_SEQ_LEN = 512
# 長文外挿方式 (§9.1): "yarn" / "ntk" / None（無効）。
#
# 【重要・既定 None の理由】YaRN/NTK は inv_freq を「スケール比」で変形するが、
# 現在の precompute_freqs_cis はそのスケール比を end(=max_seq_len*2) から算出する。
# 学習は max_seq_len=512、推論(WebUI)は max_seq_len=131072 でモデルを構築するため、
# scaling を有効にすると同一位置でも学習時と推論時で freqs が食い違い（実測 pos=100 で
# |train-inf|≈1.4、cos値域[-1,1]）、生成が破綻する（思考ボックスが空になる等）。
# また 512 までしか学習していないモデルには長文外挿の実利が無い（長文の学習データが無い）。
# → 長文学習データを導入し、かつ scaling 比を固定基準で train/推論一致させるまでは None。
# YaRN/NTK の実装本体とテストは transformer_blocks.py / test_rope_scaling.py に温存。
ROPE_SCALING_TYPE = None

MODEL_ARCH = {
    "dim": 768,
    "n_layers": 8,
    "n_heads": 12,
    "hidden_dim": 2048,
    "num_experts": 8,
    "top_k": 2,
    "dropout": 0.1,
    "z_loss_weight": 1e-4,
    "n_mtp_future": 2,
    "use_qk_norm": True,  # QK-Norm による学習安定化 (§2.3)
    "rope_original_seq_len": ROPE_ORIGINAL_SEQ_LEN,
    "rope_scaling_type": ROPE_SCALING_TYPE,  # 長文外挿 (§9.1)
    # DeepSeekMoE 設計 (§2.4-C/§2.6)。既定 False は C-1（既存チェックポイント互換）維持のため。
    # True にすると共有エキスパート/ルーティングバイアスのパラメータが増えるので新規学習が必要。
    "use_shared_experts": False,       # 問題1: 全トークンが通過する共有エキスパート
    "use_aux_free_balancing": False,   # 問題2: 補助損失を使わずバイアス項のみで負荷分散
}

def create_mini_llm(vocab_size: int, max_seq_len: int = 512):
    """Factory function to create a standardized MiniLLM model."""
    from swi_llm.core.models import MiniLLM
    return MiniLLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        **MODEL_ARCH
    )

# =====================================================================
# 3. TRAINING & DATA CONFIGURATIONS
# =====================================================================
if MODE == MODE_TEST:
    # 1~2 min
    pretrain_data_split = SPLIT_TRAIN_10
    pretrain_max_samples = 100
    sft_data_max_samples = 100
    dpo_data_max_samples = 100
    grpo_data_max_samples = 20
    epochs_override = None
    max_train_steps = None
    max_val_steps = None
elif MODE == MODE_SMOKE:
    # 3~5 min
    pretrain_data_split = SPLIT_TRAIN_100
    pretrain_max_samples = 1000
    sft_data_max_samples = 1000
    dpo_data_max_samples = 1000
    grpo_data_max_samples = 200
    epochs_override = None
    max_train_steps = None
    max_val_steps = None
elif MODE == MODE_PROD:
    # 4~8 hours
    pretrain_data_split = SPLIT_TRAIN_1000
    pretrain_max_samples = 500000
    sft_data_max_samples = 500000
    dpo_data_max_samples = 500000
    grpo_data_max_samples = 100000
    epochs_override = None
    max_train_steps = None
    max_val_steps = None
else: # MAX
    # 1 week
    pretrain_data_split = SPLIT_TRAIN
    pretrain_max_samples = None
    sft_data_max_samples = None
    dpo_data_max_samples = None
    grpo_data_max_samples = None
    epochs_override = None # Keep PROD epochs
    max_train_steps = None
    max_val_steps = None

# Global config dictionary matching the old mode_config.py structure for backward compatibility
config = {
    "MODE": MODE,
    "pretrain_data_split": pretrain_data_split,
    "pretrain_max_samples": pretrain_max_samples,
    "sft_data_max_samples": sft_data_max_samples,
    "dpo_data_max_samples": dpo_data_max_samples,
    "grpo_data_max_samples": grpo_data_max_samples,
    "rft_data_max_samples": sft_data_max_samples, # RFT reuses SFT dataset max limits for now
    "prm_data_max_samples": dpo_data_max_samples, # PRM reuses DPO limits for step data
    "epochs": epochs_override if epochs_override is not None else 3, # Default PROD epochs is 3
    "max_train_steps": max_train_steps,
    "max_val_steps": max_val_steps,
    "ckpt_suffix": f"_{MODE}" if MODE != MODE_PROD else ""
}

# DEBUG: 解決した学習/データ規模の要約（SWI_LOG_LEVEL=DEBUG で「どの規模で走っているか」を把握）
logger.debug(
    f"Config resolved: MODE={MODE}, ckpt_suffix='{config['ckpt_suffix']}', "
    f"pretrain_split={pretrain_data_split}, pretrain_max={pretrain_max_samples}, "
    f"sft_max={sft_data_max_samples}, dpo_max={dpo_data_max_samples}, grpo_max={grpo_data_max_samples}, "
    f"epochs={config['epochs']}, max_train_steps={max_train_steps}")

# Centralized Hyperparameters for each Training phase
# save_interval_ratio: Fraction of the epoch at which an intra-epoch save happens. 
# e.g., 0.25 = save 4 times per epoch.
TRAINING_CONFIG = {
    "pretrain": {
        "seq_len": 512,
        "batch_size": 2,
        "grad_accum_steps": 16,
        "lr": 1.5e-4,
        "epochs": config["epochs"],
        "save_interval_ratio": 0.15, 
    },
    "sft": {
        "seq_len": 512,
        "batch_size": 2,
        "grad_accum_steps": 16,
        "lr": 1e-4,
        "epochs": config["epochs"],
        "save_interval_ratio": 0.15,
    },
    "dpo": {
        "seq_len": 512,
        "batch_size": 2,
        "grad_accum_steps": 16,
        "lr": 1e-6,
        "beta": 0.1,
        "epochs": config["epochs"],
        "save_interval_ratio": 0.15,
    },
    "grpo": {
        "seq_len": 512,
        "batch_size": 1,
        "group_size": 4 if MODE in [MODE_TEST, MODE_SMOKE] else 8,
        "grad_accum_steps": 16,
        "lr": 1e-6,
        "beta": 0.04,
        "epsilon": 0.2,
        "epochs": config["epochs"],
        "save_interval_ratio": 0.15,
        "rollout_queue_size": 2, # Number of pre-generated rollouts
    },
    "rft": {
        "seq_len": 512,
        "batch_size": 2,
        "grad_accum_steps": 16,
        "lr": 1e-4,
        "epochs": config["epochs"],
        "save_interval_ratio": 0.15,
    },
    "prm": {
        "seq_len": 512,
        "batch_size": 2,
        "grad_accum_steps": 16,
        "lr": 5e-5,
        "epochs": config["epochs"],
        "save_interval_ratio": 0.15,
    },
    "vision": {
        "seq_len": 256,
        "batch_size": 4,
        "grad_accum_steps": 8,
        "lr": 1e-4,
        "epochs": config["epochs"],
        "save_interval_ratio": 0.15,
    }
}

# =====================================================================
# 4. INFERENCE CONFIGURATIONS
# =====================================================================
INFERENCE_CONFIG = {
    "max_seq_len": 131072,         # 128k context length
    "max_new_tokens": 2048,        # Default generation limit
    "stream_max_new_tokens": 4096, # Streaming generation limit
    "max_thought_tokens": 4096,    # Default CoT thinking limit
    "max_answer_tokens": 4096,     # Default CoT answering limit
    "use_speculative_decoding": True, # Self-Speculative Decoding via MTP
    "best_of_n": 1,                # PRM Best-of-N (§8.2)。1 で無効、>1 で有効化
    "constrained_tool_calls": True, # 不正 tool_call を制約デコーディングで有効JSON化 (§6.3 / Q-10)
}

if MODE in [MODE_TEST, MODE_SMOKE]:
    INFERENCE_CONFIG["max_new_tokens"] = 128
    INFERENCE_CONFIG["stream_max_new_tokens"] = 256
    INFERENCE_CONFIG["max_thought_tokens"] = 64
    INFERENCE_CONFIG["max_answer_tokens"] = 128

# =====================================================================
# 5. VISION CONFIGURATIONS
# =====================================================================
VISION_CONFIG = {
    "image_size": (224, 224),
    "image_mean": [0.48145466, 0.4578275, 0.40821073], # CLIP mean
    "image_std": [0.26862954, 0.26130258, 0.27577711],  # CLIP std
    # 01(小): リサイズ方針の単一情報源（学習 vision_data.py / 推論 web/models.py /
    # core/vision.py preprocess_pil_image が共通で参照する）。
    #   "squash"        : アスペクト比を無視して直接リサイズ（現行チェックポイントの学習分布）
    #   "shortside_crop": 短辺基準リサイズ + センタークロップ（CLIP 標準・歪みなし）
    # 現行の VLM は squash で学習済みのため、切替は vision 再学習とセットで行うこと
    # （roadmap/llm_roadmap.md ステップ7）。推論だけ切替えると学習分布と乖離して劣化する。
    "resize_policy": "squash",
    # L-5: AnyRes（高解像度・非正方形画像をタイル分割して細部を保持）。既定 OFF。
    # 現行の学習は単一 224 画像（vision トークン固定長）で行っているため、ON にすると
    # vision トークン数が (タイル数×パッチ数) に増え、学習分布と乖離して品質が劣化する。
    # 実運用で有効化するには AnyRes 対応の VLM 学習＋長文脈化が前提（整合を取ってから有効化）。
    "use_anyres": False,
    # AnyRes 有効時、元画像の長辺をこの値以下へ縮小してからタイル分割する（タイル数=OOM/系列長の暴発防止）。
    "anyres_max_side": 448,   # 224×2 → 最大 2×2 タイル + グローバル = 5 タイル
}

# =====================================================================
# 6. YAML CONFIG OVERRIDES（問題3: Config 駆動アーキテクチャ）
# =====================================================================
# ブループリント 0.1 の「Config 駆動（YAML で宣言）」に対応する軽量なオーバーライド層。
# OmegaConf/Hydra は新規依存になるため、既に導入済みの pyyaml で同等の宣言的オーバーライドを
# 実現する。既定（YAML 不在 / pyyaml 不在 / パース失敗）では Python 既定値をそのまま使い、
# 挙動は一切変わらない（非破壊）。優先順位: 引数 path > 環境変数 SWI_CONFIG > configs/default.yaml。

def _deep_merge(base: dict, override: dict) -> dict:
    """override を base へ再帰マージする（in-place）。ネストした dict は再帰、それ以外は上書き。"""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config_overrides(path: str = None) -> dict:
    """YAML 設定オーバーライドを読み込む。読めない場合は {}（=Python 既定値へフォールバック）。"""
    try:
        import yaml
    except Exception:
        return {}
    if path is None:
        path = os.environ.get("SWI_CONFIG")
    if not path:
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs", "default.yaml")
        path = default_path if os.path.exists(default_path) else None
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict) or not data:
            return {}
        logger.info(f"Applying YAML config overrides from {path}.")
        return data
    except Exception as e:
        logger.warning(f"Failed to load YAML config overrides from {path}: {e}. Using Python defaults.")
        return {}


# 全 config 定義後にオーバーライドを適用（create_mini_llm は MODEL_ARCH を実行時参照するため反映される）。
_CONFIG_OVERRIDES = load_config_overrides()
if _CONFIG_OVERRIDES:
    _deep_merge(MODEL_ARCH, _CONFIG_OVERRIDES.get("model_arch", {}) or {})
    _deep_merge(TRAINING_CONFIG, _CONFIG_OVERRIDES.get("training", {}) or {})
    _deep_merge(INFERENCE_CONFIG, _CONFIG_OVERRIDES.get("inference", {}) or {})
    _deep_merge(VISION_CONFIG, _CONFIG_OVERRIDES.get("vision", {}) or {})
    _deep_merge(config, _CONFIG_OVERRIDES.get("run", {}) or {})


def parse_token_count(raw):
    """"4B" / "500M" / "200k" / "4e9" / 4000000000 をトークン数（int）へ。空・None は None。"""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip().replace("_", "").replace(",", "")
        if not text:
            return None
        units = {"K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}
        mult = units.get(text[-1].upper(), 1.0)
        if mult != 1.0:
            text = text[:-1]
        try:
            value = float(text) * mult
        except ValueError:
            raise ValueError(f"Invalid token count {raw!r} (examples: 4B, 500M, 200k, 4e9).") from None
    if value <= 0:
        raise ValueError(f"Token count must be positive: {raw!r}")
    return int(value)


def resolve_pretrain_token_budget():
    """事前学習のトークン予算。環境変数 SWI_PRETRAIN_TOKENS が YAML の run.pretrain_token_budget に優先する。"""
    from swi_llm.utils.constants import PRETRAIN_TOKENS_ENV
    env = os.environ.get(PRETRAIN_TOKENS_ENV, "").strip()
    return parse_token_count(env) if env else parse_token_count(config.get("pretrain_token_budget"))


# トークン予算モード（roadmap A-1）: 予算が決まっていれば事前学習は既定 1 パス。
config["pretrain_token_budget"] = resolve_pretrain_token_budget()
if config["pretrain_token_budget"]:
    from swi_llm.utils.constants import PRETRAIN_BUDGET_EPOCHS
    TRAINING_CONFIG["pretrain"]["epochs"] = int(os.environ.get("SWI_PRETRAIN_EPOCHS", "").strip()
                                                or PRETRAIN_BUDGET_EPOCHS)
    logger.info(f"Pretrain token budget: {config['pretrain_token_budget']:,} tokens "
                f"(epochs={TRAINING_CONFIG['pretrain']['epochs']}).")
