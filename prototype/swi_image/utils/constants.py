"""
swi_image/utils/constants.py

画像生成AIのマジックナンバー・マジックストリングの集約（swi_llm 規約 S-2 に準拠）。
"""

# =====================================================================
# モード（swi_llm/swi_voice と共有: 環境変数 SWI_MODE）
# =====================================================================
MODE_TEST = "TEST"
MODE_SMOKE = "SMOKE"
MODE_PROD = "PROD"
MODE_MAX = "MAX"

# =====================================================================
# VAE（潜在空間）[画像編2]
# =====================================================================
IMG_VAE_LATENT_CH = 4                 # 潜在チャンネル数
IMG_VAE_DOWN_FACTOR = 8               # 空間ダウンサンプル率（8=SD系標準）
IMG_VAE_BASE_CH = 64                  # エンコーダ/デコーダの基底チャンネル数
IMG_VAE_KL_WEIGHT = 1e-6             # KL 損失の重み [画像編2.2]
IMG_VAE_SCALE_FACTOR = 0.18215       # 潜在のスケール係数（SD 慣習値）

# =====================================================================
# DiT（Diffusion Transformer）[画像編4]
# =====================================================================
IMG_PATCH_SIZE = 2                    # 潜在上のパッチサイズ
IMG_DIT_DIM = 384
IMG_DIT_DEPTH = 6
IMG_DIT_HEADS = 6

# =====================================================================
# Flow Matching / サンプリング [画像編3・5]
# =====================================================================
IMG_SAMPLE_STEPS = 28                 # Euler ODE ステップ数（Flow Matching で少なめでOK）
IMG_CFG_SCALE = 4.0                   # Classifier-Free Guidance の強度 [画像編7.1]
IMG_CFG_DROPOUT = 0.1                 # 学習時に条件を落として無条件も学習する確率

# =====================================================================
# テキスト条件付け [画像編6]
# =====================================================================
# テキストトークナイザは swi_llm の SentencePiece(32k 日本語) を流用する。
IMG_TEXT_MAX_LEN = 32
IMG_TEXT_ENC_DIM = 256
IMG_TEXT_ENC_LAYERS = 2
IMG_TEXT_ENC_HEADS = 4

# =====================================================================
# データセット [画像編1]
# =====================================================================
# 実 image-caption コーパス候補（すべて Parquet ネイティブ・画像バイト埋め込み・streaming）。
# 上から順に試し、最初に取得できたソースを使う。全モードで実データのみ（合成なし）。
# (dataset_name, load_kwargs, image_field, text_field, split)
#   - jxie/coco_captions              : COCO 566,747 枚（英キャプション, 640x480 埋め込み）主力
#   - sayakpaul/coco-30-val-2014      : COCO val 30,000 枚（英キャプション）予備
#   - svjack/pokemon-blip-captions-en-ja : 833 枚（日本語キャプション ja_text）予備
IMG_DATASET_CANDIDATES = [
    ("jxie/coco_captions", {}, "image", "caption", "train"),
    ("sayakpaul/coco-30-val-2014", {}, "image", "caption", "train"),
    ("svjack/pokemon-blip-captions-en-ja", {}, "image", "ja_text", "train"),
]
IMG_MIN_CAPTION_LEN = 3

# =====================================================================
# 安全性 [画像編1.3 / 11, NEW-I1]
# =====================================================================
# 学習データの NSFW フィルタ（data/safety_filter.py で実体化済み）。
# nsfw スコアがこの閾値以上の画像を学習データから除外する。
IMG_NSFW_THRESHOLD = 0.85
# "0" で学習データ安全フィルタを明示的に無効化（既定は有効。分類器が使えない環境では
# フィルタなしで続行するが ERROR ログで明示する — 静かな縮退にしない）。
IMG_SAFETY_ENV = "SWI_IMG_SAFETY"
# NSFW 分類器（HF image-classification モデル）。初回のみ重みを DL（~350MB）。
IMG_NSFW_MODEL_ENV = "SWI_IMG_NSFW_MODEL"
IMG_NSFW_DEFAULT_MODEL = "Falconsai/nsfw_image_detection"

# =====================================================================
# 重複排除 [画像編1.1, NEW-I3]
# =====================================================================
# pHash（DCT ベース 64bit 知覚的ハッシュ）による近重複の除外。記憶（Memorization）
# リスクと学習データの偏りを抑える。ハミング距離がこの値以下なら「ほぼ同一」とみなす
# （0 = 完全一致のみ。6 は再圧縮/リサイズ程度の変化を同一と判定する保守的な値）。
IMG_PHASH_MAX_HAMMING = 6
IMG_DEDUP_ENV = "SWI_IMG_DEDUP"      # "0" で無効化（既定は有効）
