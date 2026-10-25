"""
swi_voice/utils/constants.py

音声合成AIのマジックナンバー・マジックストリングの集約（swi_llm 規約 S-2 に準拠）。
音響パラメータ・コーデック構成・データセット等をここで名前付き定数として定義する。
"""

# =====================================================================
# モード（swi_llm と共有: 環境変数 SWI_MODE）
# =====================================================================
MODE_TEST = "TEST"
MODE_SMOKE = "SMOKE"
MODE_PROD = "PROD"
MODE_MAX = "MAX"

# =====================================================================
# 音響パラメータ [音声編2.1]
# =====================================================================
VOICE_SAMPLE_RATE = 24000            # サンプリングレート (Hz)
VOICE_FRAME_RATE = 50                # コーデックのフレームレート (Hz)
# エンコーダの総ダウンサンプル率 = sample_rate / frame_rate = 480。
# strided Conv1d の各段ストライドの積が 480 になるよう分解する。
VOICE_CODEC_STRIDES = (2, 4, 4, 5, 3)   # 積 = 480
VOICE_CODEC_HOP = VOICE_SAMPLE_RATE // VOICE_FRAME_RATE  # 480
# V-5: ストリーミング時、直前チャンク末尾のこのフレーム数をデコーダへ左文脈として渡し、
# 連続デコード後に重複分をトリムしてチャンク境界の不連続（クリック/段差）を軽減する。
VOICE_STREAM_OVERLAP_FRAMES = 2

# =====================================================================
# ニューラルコーデック (RVQ) 構成 [音声編2.2]
# =====================================================================
VOICE_CODEC_DIM = 256                # コーデック潜在次元
VOICE_CODEC_HIDDEN = 64              # Conv スタックの基底チャンネル数
VOICE_N_QUANTIZERS = 8               # RVQ 段数（多いほど高品質・長系列）
VOICE_CODEBOOK_SIZE = 1024           # 各コードブックのエントリ数
VOICE_VQ_COMMIT_WEIGHT = 0.25        # コミットメント損失の重み

# 意味/音響トークンの分離 [音声編2.3]（ハイブリッドAR+FMの土台, 問題1）。
# 意味トークン: 音響エンコーダ特徴を時間方向に downsample した低フレームレートの単一VQ。
# 音響トークン: 既存 RVQ の高フレームレート連続latent（NeuralAudioCodec）。
VOICE_SEMANTIC_CODEBOOK_SIZE = 512   # 意味トークンのコードブックサイズ（音響より粗い＝内容中心）
VOICE_SEMANTIC_DOWNSAMPLE = 4        # 音響フレームレートに対する意味フレームの間引き率（50Hz→12.5Hz）

# 多解像度 STFT 損失の (n_fft, hop_length) スケール [音声編2.1 multi_scale_stft_loss]
VOICE_STFT_SCALES = ((512, 128), (1024, 256), (2048, 512))

# ニューラルボコーダ（HiFi-GAN/Vocos 系）[音声編品質]。mel スペクトログラム→波形。
# mel の hop はコーデック hop(480) に合わせ、フレームレートを 50Hz に揃える。
VOICE_N_MELS = 80                    # mel フィルタバンク数
VOICE_MEL_N_FFT = 1024               # mel 抽出の FFT 窓長
VOICE_VOCODER_HIDDEN = 128           # ボコーダ生成器の基底チャンネル数

# =====================================================================
# TTS (AR コーデック言語モデル) 構成 [音声編3.1]
# =====================================================================
# テキストトークナイザは swi_llm の SentencePiece を流用する（日本語対応・32k）。
VOICE_TTS_DIM = 512
VOICE_TTS_N_LAYERS = 6
VOICE_TTS_N_HEADS = 8
VOICE_TTS_MAX_SEQ_LEN = 2048

# Flow Matching TTS (F5-TTS/E2-TTS 系) [音声編3.2] のモデル規模。
# コーデックの連続 latent [B, T', codec_dim] 上で非自己回帰生成する 1D-DiT。
VOICE_FLOW_DIM = 512                 # DiT の内部次元
VOICE_FLOW_N_LAYERS = 6             # DiT ブロック数
VOICE_FLOW_N_HEADS = 8             # アテンションヘッド数
VOICE_FLOW_MAX_FRAMES = 1024       # 位置埋め込みの最大フレーム数（~20s @ 50fps）
VOICE_FLOW_SAMPLE_STEPS = 16      # 生成時の Euler ODE ステップ数（多いほど高品質・低速）

# TTS serving エンジン [NEW-V2/V3]: 学習済みの 3 経路を serving で切り替える。
#   ar     (既定): AR コーデック LM（VALL-E 系）。ストリーミング・クローニング対応の現行主経路。
#   flow   : Flow Matching TTS（F5/E2 系, 非自己回帰）。voice_flow_checkpoint が必要。
#   hybrid : AR(意味) + FM(音響) の 2 段構成 [音声編3.3]。voice_hybrid_checkpoint が必要。
# 指定エンジンの ckpt が無い場合は WARNING を出して ar へ縮退する（C-4）。
VOICE_TTS_ENGINE_ENV = "SWI_TTS_ENGINE"
# 非自己回帰エンジン（flow/hybrid）のストリーミング応答で 1 チャンクにする秒数。
# 生成自体は一括（非 AR）なので、これは転送・逐次再生の粒度。
VOICE_NONAR_STREAM_CHUNK_SEC = 0.5

# 特殊トークン（コーデック系列の境界）
VOICE_BOS = "<|voice_bos|>"          # 音声生成開始
VOICE_EOS = "<|voice_eos|>"          # 音声生成終了

# スタイル/感情制御 [音声編4]。明示的な感情タグの種類数（neutral/happy/sad/angry 等）。
# 参照音声からの暗黙的スタイル転写も併用できる（StyleEncoder）。
VOICE_N_EMOTIONS = 8
VOICE_EMOTIONS = ("neutral", "happy", "sad", "angry", "fear", "surprise", "disgust", "calm")

# 指示追従TTS [音声編6.3, 実装編5.3]。自然言語のスタイル指示（「落ち着いた低い声で
# ゆっくり」等）をテキスト列の先頭にこのタグで囲んで与え、発話内容と区別させる。
VOICE_STYLE_TAG_OPEN = "[STYLE]"
VOICE_STYLE_TAG_CLOSE = "[/STYLE]"
# 指示追従SFT でスタイル指示に割り当てるトークン数の上限（長い指示でテキスト枠を
# 食い潰さないよう制限する）。
VOICE_MAX_INSTRUCTION_TOKENS = 32

# =====================================================================
# データセット [音声編1.1]
# =====================================================================
# 事前学習用の音声-テキストコーパス候補。§1.1 の「話者・多様性がクローニング性能に
# 直結」に従い、アクセスできる全ソースを streaming で混合する（swi_llm のデータミックス
# と同型）。入手不可（ゲート/未認証/不存在）のソースは自動スキップされる。
# (dataset_name, load_kwargs, audio_field, text_field, lang, split)
# 注: datasets 5.0 はスクリプト方式（trust_remote_code）を廃止 → Parquet ネイティブのみ可。
# 元の mozilla-foundation/common_voice・reazon-research/reazonspeech・lj_speech はスクリプト
# 方式で読めないため、Parquet 版ミラー（kotoba-tech の japanese-asr/*）へ差し替えている。
# 音声デコードは torchcodec ではなく soundfile（ffmpeg 非依存）で行う（audio_dataset.py 参照）。
#   - common_voice_8_0 : common_voice 日本語の Parquet 版（split=test・高速）。
#   - reazon_speech_all: ReazonSpeech（放送由来・大規模）の Parquet 版（config subset_0・高速）。
#   - jsut_basic5000   : JSUT・高品質単一話者TTSコーパスの Parquet 版（split=test・高速）。
#   - fleurs           : 公開・多言語読み上げの日本語分（train）。parquet が巨大で低速なため最後。
# 高速な3ソースを先頭に置き、混合予算を素早く満たす（TEST/SMOKE も実データで学習）。
VOICE_DATASET_CANDIDATES = [
    ("japanese-asr/ja_asr.common_voice_8_0", {}, "audio", "transcription", "ja", "test"),
    ("japanese-asr/ja_asr.reazon_speech_all", {"name": "subset_0"}, "audio", "transcription", "ja", "train"),
    ("japanese-asr/ja_asr.jsut_basic5000", {}, "audio", "transcription", "ja", "test"),
    ("google/fleurs", {"name": "ja_jp"}, "audio", "transcription", "ja", "train"),
]
VOICE_MIN_AUDIO_SEC = 0.5            # 短すぎるクリップは除外
VOICE_MAX_AUDIO_SEC = 15.0          # 長すぎるクリップは除外
VOICE_ASR_WER_THRESHOLD = 0.15      # ASR疑似ラベルの整合しきい値 [音声編1.2]

# --- SNR フィルタ [音声編1.1「SNR による低品質音源の除外」, NEW-V1] ---
# 推定 SNR がこの値 [dB] 未満のクリップを学習から除外する。実測（2026-07-16, 3 ソース）:
# 実クリップは 16〜59dB、ノイズのみ ~0.6dB、強ノイズ付加 3.5〜14dB。10dB は実クリップを
# 落とさず（最小 16.1dB）、無音・ノイズ支配のクリップを弾く保守的な既定値。
VOICE_MIN_SNR_DB = 10.0
VOICE_MIN_SNR_ENV = "SWI_VOICE_MIN_SNR"   # 数値で上書き / "off" で無効化
VOICE_SNR_FRAME_MS = 30              # SNR 推定のフレーム長 [ms]
VOICE_SNR_NOISE_PCT = 0.05           # ノイズフロア = フレームパワーの下位分位
VOICE_SNR_SPEECH_PCT = 0.90          # 音声レベル = フレームパワーの上位分位
VOICE_SNR_MAX_DB = 60.0              # 推定の上限（デジタル無音で発散しないためのキャップ）
# 多言語対応 [音声編5.2]: TTS 学習時にテキスト先頭へ付与する言語IDタグ。
VOICE_LANG_TAG = "[{}] "

# =====================================================================
# 同意チェーン [音声編8.1]
# =====================================================================
CONSENT_SCOPE_PRETRAIN = "pretrain"
CONSENT_SCOPE_TTS = "commercial_tts"
CONSENT_SCOPE_CLONING = "voice_cloning"
# audio_dataset が学習データ収集時に渡す purpose 文字列。
CONSENT_SCOPE_TRAINING = "training"

# 公開ライセンス済みコーパス（VOICE_DATASET_CANDIDATES）に対して「データセット単位の
# 同意」として自動付与するスコープ。これらの公開音声コーパスは研究/ASR/TTS 用途で
# 明示ライセンスされているため、収集パイプラインでは許諾済みとして扱う。ただし
# 「特定個人の声のクローニング(voice_cloning)」は含めない（個別の明示同意が必要）。
CONSENT_PUBLIC_SOURCE_SCOPES = (
    CONSENT_SCOPE_TRAINING, CONSENT_SCOPE_PRETRAIN, CONSENT_SCOPE_TTS)
# 同意台帳に明示登録の無い話者/ソースを既定で拒否するか（consent-first の担保）。
# False にすると従来どおり全通しになるため、通常は True のまま運用する。
CONSENT_ENFORCE = True

# ゼロショットクローニングの同意ゲート [音声編4.1 / 8.1, 実装編4.1・10.1]。
# 参照音声による声色の模倣は「話者個別の明示同意」を必須とする（上記のとおり公開
# コーパスのデータセット単位同意には voice_cloning を含めない）。自社録音・出演者
# 同意済みの話者を起動時に台帳へ登録するための環境変数（カンマ区切りの話者ID）。
#   例: SWI_VOICE_CLONING_CONSENTED="actor_001,actor_002"
VOICE_CLONING_CONSENT_ENV = "SWI_VOICE_CLONING_CONSENTED"

# 参照音声プロンプトの最大フレーム数（50Hz で ~3s 相当）。長すぎる参照で AR の
# コンテキストが肥大化しないよう制限する [音声編4.1]。
VOICE_MAX_PROMPT_FRAMES = 150

# =====================================================================
# VAD（発話区間検出）[音声編10.2, fix 13: RMS → ニューラル VAD]
# =====================================================================
# バックエンド選択の環境変数。auto（既定: silero が使えれば silero, 無ければ rms）
# / silero（強制。不在なら警告して rms 縮退）/ rms（従来の RMS エネルギー方式）。
VAD_BACKEND_ENV = "SWI_VAD_BACKEND"
VAD_SILERO_SAMPLE_RATE = 16000       # Silero VAD の入力サンプルレート
VAD_SILERO_WINDOW = 512              # 1 判定窓のサンプル数（Silero の仕様値, 32ms @16kHz）
VAD_SPEECH_PROB_THRESHOLD = 0.5      # 単発判定（detect_voice_activity）の発話確率しきい値
# barge-in のヒステリシス（fix 13(2)）: 開始は高め・終了は低めの 2 閾値 + 最小継続で
# 単発ノイズ（咳・物音）による誤 barge-in と、発話中の短い息継ぎでの解除チャタリングを抑える。
VAD_PROB_ON = 0.6                    # 発話開始とみなす確率しきい値
VAD_PROB_OFF = 0.4                   # 発話終了とみなす確率しきい値
VAD_MIN_ACTIVE_CHUNKS = 2            # 発話開始と認める連続アクティブチャンク数

# =====================================================================
# ASR（音声認識）[音声編10.1, fix 08: faster-whisper 統合]
# =====================================================================
ASR_DISABLE_ENV = "SWI_ASR"          # "0" で実 ASR を無効化（従来の「ASR なし」縮退動作）
ASR_MODEL_ENV = "SWI_ASR_MODEL"      # faster-whisper のモデルサイズ or ローカルパス
ASR_DEFAULT_MODEL = "small"          # 日本語で実用最小（CPU int8 で ~1s / 2s クリップを実測）
ASR_SAMPLE_RATE = 16000              # Whisper 系の入力サンプルレート
ASR_DEFAULT_LANGUAGE = "ja"          # 既定は日本語固定（fix 08 本番仕様）
ASR_BEAM_SIZE = 5                    # faster-whisper の beam_size（既定値と同じ・明示）

# =====================================================================
# 電子透かし（音声の出所証明）[音声編8.2]
# =====================================================================
# 拡散スペクトラム方式: payload から決定的に生成した ±1 擬似ノイズ列を微小振幅で
# 加算し、相関で検出する。可聴閾値以下（-48dB 相当）で埋め込み、AI 生成の証跡を残す。
VOICE_WATERMARK_PAYLOAD = "SWI-VOICE"   # TTS 出力に埋め込む既定ペイロード
VOICE_WATERMARK_STRENGTH = 0.006        # 加算振幅 α（フルスケール比。-44dB 相当で不可聴）
VOICE_WATERMARK_DETECT_FACTOR = 0.4     # 検出しきい値 = α × この係数（相関統計量の下限）
VOICE_WATERMARK_KNOWN_PAYLOADS = ("SWI-VOICE", "SWI_STREAM", "SWI_TEST", "SWI")
