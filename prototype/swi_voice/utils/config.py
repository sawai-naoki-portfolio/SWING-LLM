"""
swi_voice/utils/config.py

音声合成AIのモード別設定。swi_llm と同じ SWI_MODE (TEST/SMOKE/PROD/MAX) を参照し、
段階別のデータ量・学習ステップ・モデル規模を一元管理する（swi_llm 規約 S-7）。
"""
import os
import logging

logger = logging.getLogger(__name__)

from swi_voice.utils.constants import (
    MODE_TEST, MODE_SMOKE, MODE_PROD, MODE_MAX,
    VOICE_CODEC_DIM, VOICE_CODEC_HIDDEN, VOICE_N_QUANTIZERS, VOICE_CODEBOOK_SIZE,
    VOICE_TTS_DIM, VOICE_TTS_N_LAYERS, VOICE_TTS_N_HEADS, VOICE_TTS_MAX_SEQ_LEN,
    VOICE_FLOW_DIM, VOICE_FLOW_N_LAYERS, VOICE_FLOW_N_HEADS, VOICE_FLOW_MAX_FRAMES,
    VOICE_FLOW_SAMPLE_STEPS,
)

# =====================================================================
# モード検出（swi_llm と完全同期。swi_llm 側の自動判別に相乗りする）
# =====================================================================
try:
    from swi_llm.utils.config import MODE
except ImportError:
    MODE = os.environ.get("SWI_MODE", "").strip().upper()
    if MODE not in (MODE_TEST, MODE_SMOKE, MODE_PROD, MODE_MAX):
        if MODE:
            logger.warning(f"Unknown SWI_MODE '{MODE}', falling back to {MODE_TEST}.")
        MODE = MODE_TEST

# =====================================================================
# コーデック学習設定（モード別）
# =====================================================================
# clip_frames: 1 クリップのコーデックフレーム数（= 音声長 / VOICE_CODEC_HOP）。
#   ゼロショットクローニングでは「前半=音響プロンプト / 後半=ターゲット」に分割するため、
#   意味のある分割ができる長さが必要。TEST/SMOKE は高速化のため短め、PROD/MAX は実用長。
# prompt_frames: クローニング学習時に音響プロンプトへ割り当てるフレーム数（損失対象外）。
if MODE == MODE_TEST:
    codec_max_clips = 32
    codec_epochs = 1
    tts_max_clips = 32
    tts_epochs = 1
    clip_frames = 16          # ~0.32s（プロンプト8 + ターゲット8 のスモーク検証）
    prompt_frames = 8
elif MODE == MODE_SMOKE:
    codec_max_clips = 500
    codec_epochs = 1
    tts_max_clips = 500
    tts_epochs = 1
    clip_frames = 32          # ~0.64s
    prompt_frames = 12
elif MODE == MODE_PROD:
    codec_max_clips = 50000
    codec_epochs = 3
    tts_max_clips = 50000
    tts_epochs = 3
    clip_frames = 150         # ~3.0s
    prompt_frames = 75        # ~1.5s の音響プロンプト
else:  # MAX
    codec_max_clips = None
    codec_epochs = 5
    tts_max_clips = None
    tts_epochs = 5
    clip_frames = 300         # ~6.0s
    prompt_frames = 150       # ~3.0s の音響プロンプト

# コーデックのモデル規模（TEST/SMOKE は小型で高速に）
_small = MODE in (MODE_TEST, MODE_SMOKE)

CODEC_CONFIG = {
    "dim": 64 if _small else VOICE_CODEC_DIM,
    "hidden": 16 if _small else VOICE_CODEC_HIDDEN,
    "n_quantizers": 2 if _small else VOICE_N_QUANTIZERS,
    "codebook_size": 256 if _small else VOICE_CODEBOOK_SIZE,
    "batch_size": 2 if _small else 8,
    "lr": 3e-4,
    "epochs": codec_epochs,
    "max_clips": codec_max_clips,
}

TTS_CONFIG = {
    "dim": 128 if _small else VOICE_TTS_DIM,
    "n_layers": 2 if _small else VOICE_TTS_N_LAYERS,
    "n_heads": 4 if _small else VOICE_TTS_N_HEADS,
    "max_seq_len": 512 if _small else VOICE_TTS_MAX_SEQ_LEN,
    "batch_size": 2 if _small else 8,
    "lr": 3e-4,
    "epochs": tts_epochs,
    "max_clips": tts_max_clips,
}

# Flow Matching TTS [音声編3.2]。学習/生成のフレーム長はクリップ長(clip_frames)に揃える。
FLOW_CONFIG = {
    "dim": 128 if _small else VOICE_FLOW_DIM,
    "n_layers": 2 if _small else VOICE_FLOW_N_LAYERS,
    "n_heads": 4 if _small else VOICE_FLOW_N_HEADS,
    "max_frames": VOICE_FLOW_MAX_FRAMES,
    "n_frames": clip_frames,            # 固定長 latent のフレーム数（= クリップ長）
    "sample_steps": VOICE_FLOW_SAMPLE_STEPS,
    "batch_size": 2 if _small else 8,
    "lr": 3e-4,
    "epochs": tts_epochs,
    "max_clips": tts_max_clips,
}

# ハイブリッド AR+FM [音声編3.3, NEW-V3]。意味 LM(AR) は TTS 系の縮小版、音響 FM は
# FLOW 系に揃える。意味 VQ（SemanticAcousticCodec）→ ハイブリッド本体の 2 段で学習する。
HYBRID_CONFIG = {
    "ar_dim": 128 if _small else 256,
    "ar_layers": 2 if _small else 4,
    "ar_heads": 4,
    "fm_dim": 128 if _small else VOICE_FLOW_DIM,
    "fm_layers": 2 if _small else 4,
    "fm_heads": 4,
    "fm_max_frames": VOICE_FLOW_MAX_FRAMES,
    "n_frames": clip_frames,            # 音響 latent のフレーム数（= クリップ長）
    "max_sem_len": 256,                 # 意味トークン列の上限（FM の条件長）
    "sample_steps": VOICE_FLOW_SAMPLE_STEPS,
    "sem_vq_epochs": tts_epochs,        # 段1: 意味 VQ の学習エポック
    "batch_size": 2 if _small else 8,
    "lr": 3e-4,
    "epochs": tts_epochs,               # 段2: ハイブリッド本体の学習エポック
    "max_clips": tts_max_clips,
}

# チェックポイント接尾辞（PROD は無印、それ以外はモード名）
CKPT_SUFFIX = "" if MODE == MODE_PROD else f"_{MODE}"

config = {
    "MODE": MODE,
    "codec": CODEC_CONFIG,
    "tts": TTS_CONFIG,
    "flow": FLOW_CONFIG,
    "hybrid": HYBRID_CONFIG,
    "ckpt_suffix": CKPT_SUFFIX,
    "clip_frames": clip_frames,        # コーデック/TTS 共通のクリップ長（フレーム）
    "prompt_frames": prompt_frames,    # クローニング学習の音響プロンプト長（フレーム）
}

# =====================================================================
# YAML 設定オーバーライド（問題5: Config 駆動アーキテクチャ）
# =====================================================================
# swi_llm と同じ pyyaml ベースのオーバーライド機構を再利用し、YAML の `voice` セクションで
# codec/tts/flow のハイパーパラメータを宣言的に上書きできるようにする。既定（YAML 不在/
# pyyaml 不在/パース失敗）では Python 既定値をそのまま使い、挙動は変わらない（非破壊）。
try:
    from swi_llm.utils.config import load_config_overrides, _deep_merge
    _VOICE_OVERRIDES = (load_config_overrides().get("voice") or {})
    if _VOICE_OVERRIDES:
        _deep_merge(CODEC_CONFIG, _VOICE_OVERRIDES.get("codec", {}) or {})
        _deep_merge(TTS_CONFIG, _VOICE_OVERRIDES.get("tts", {}) or {})
        _deep_merge(FLOW_CONFIG, _VOICE_OVERRIDES.get("flow", {}) or {})
        _deep_merge(HYBRID_CONFIG, _VOICE_OVERRIDES.get("hybrid", {}) or {})
        _deep_merge(config, {k: v for k, v in _VOICE_OVERRIDES.items()
                             if k not in ("codec", "tts", "flow", "hybrid")})
        logger.info("Applied YAML 'voice' config overrides.")
except Exception as _e:  # オーバーライド層の失敗が起動を壊さないよう握りつぶす
    logger.warning(f"Voice YAML override skipped ({_e}); using Python defaults.")
