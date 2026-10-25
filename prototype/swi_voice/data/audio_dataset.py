"""
swi_voice/data/audio_dataset.py

音声-テキストコーパスの読み込み [音声編1.1/1.2]。

HuggingFace の実音声コーパス（common_voice / reazon_speech / jsut / fleurs 等）を
streaming で取得し、torchaudio で VOICE_SAMPLE_RATE のモノラルへ整える。

方針: 全モード（TEST/SMOKE/PROD/MAX）で必ず実データのみを使う。合成/ダミー音声は
使用しない。実データが 1 件も取得できない場合は例外を送出して学習を止める
（誤って合成データで学習しないための安全策）。
"""
import random
import itertools

import torch
from torch.utils.data import Dataset

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_SAMPLE_RATE, VOICE_CODEC_HOP, VOICE_DATASET_CANDIDATES,
    VOICE_MIN_AUDIO_SEC, VOICE_MAX_AUDIO_SEC, VOICE_LANG_TAG,
)
from swi_voice.safety.consent_check import verify_speaker_consent

logger = setup_logger("voice.data")


def _collect_one_source(name, kwargs, audio_field, text_field, lang, split, target, sample_rate,
                        skip=0):
    """1 つの HF コーパスから streaming で最大 target 件を収集する。失敗時は空。

    datasets 5.0 の Audio 自動デコードは torchcodec（FFmpeg 共有ライブラリ）依存で
    環境差に弱いため、Audio(decode=False) で生バイトを取り出し soundfile
    （libsndfile 同梱・ffmpeg 非依存、wav/flac/mp3 対応）で自前デコードする。
    リサンプルは torchaudio.functional.resample（純 torch）で行う。

    skip: ストリーム先頭から読み飛ばす件数。評価で「学習に使っていないクリップ」を取るために使う。
    """
    import io
    import soundfile as sf
    import torchaudio
    from datasets import load_dataset, Audio
    from swi_voice.data.quality_filter import passes_snr, min_snr_threshold

    ds = load_dataset(name, split=split, streaming=True, **kwargs)
    ds = ds.cast_column(audio_field, Audio(decode=False))     # torchcodec を介さない
    clips = []
    snr_threshold = min_snr_threshold()                        # None = SNR フィルタ無効
    snr_dropped = 0
    for row in itertools.islice(ds, skip, skip + target * 3):
        audio = row.get(audio_field)
        text = str(row.get(text_field, "")).strip()
        if not audio or not text:
            continue
        try:
            if audio.get("bytes"):
                data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
            elif audio.get("path"):
                data, sr = sf.read(audio["path"], dtype="float32", always_2d=False)
            else:
                continue
        except Exception:
            continue                                          # デコード不能なクリップはスキップ
        wav = torch.as_tensor(data, dtype=torch.float32)
        if wav.ndim > 1:
            wav = wav.mean(dim=-1)                             # モノラル化
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sec = wav.shape[-1] / sample_rate
        if sec < VOICE_MIN_AUDIO_SEC or sec > VOICE_MAX_AUDIO_SEC:
            continue

        # SNR フィルタ [音声編1.1, NEW-V1]: 雑音支配・無音同然の低品質クリップを除外する。
        passed, snr = passes_snr(wav, sample_rate=sample_rate, threshold=snr_threshold)
        if not passed:
            snr_dropped += 1
            logger.debug(f"SNR filtered: {snr:.1f} dB < {snr_threshold} dB ({name})")
            continue

        speaker_id = str(row.get("client_id", row.get("speaker_id", name)))
        # [V-4] 安全基盤の強制適用: 学習データの利用許諾チェック（consent-first）。
        # source=name を渡し、公開ライセンス済みコーパスはデータセット単位の同意として
        # 許可、未知/未同意のソース・話者は除外する（consent_check.py 参照）。
        if not verify_speaker_consent(speaker_id, purpose="training", source=name):
            continue
            
        clips.append({"audio": wav.unsqueeze(0), "text": text, "lang": lang,
                      "speaker_id": speaker_id})
        if len(clips) >= target:
            break
    if snr_dropped:
        logger.info(f"SNR filter ({name}): dropped {snr_dropped} low-quality clips "
                    f"(< {snr_threshold} dB, NEW-V1).")
    return clips


def _mix_from_hf(max_clips, sample_rate, skip_per_source=0):
    """アクセスできる全コーパスを混合する（§1.1 多様性）。ソース欠落は自動スキップ。
    max_clips を総予算とし、各ソースへ均等配分して収集する。

    skip_per_source: 各ソースのストリーム先頭から読み飛ばす件数（評価のホールドアウト用）。
    """
    try:
        import torchaudio  # noqa: F401
        from datasets import load_dataset  # noqa: F401
    except Exception as e:
        logger.warning(f"Could not load audio source (deps unavailable: {e}).")
        return []

    n = max(1, len(VOICE_DATASET_CANDIDATES))
    per_source = max(1, (max_clips + n - 1) // n)
    mixed = []
    for name, kwargs, audio_field, text_field, lang, split in VOICE_DATASET_CANDIDATES:
        remaining = max_clips - len(mixed)
        if remaining <= 0:
            break
        target = min(per_source, remaining) if len(mixed) < max_clips else 0
        # 予算に余裕があれば残りを埋められるよう、少なくとも per_source は狙う
        target = max(target, min(per_source, remaining))
        try:
            clips = _collect_one_source(name, kwargs, audio_field, text_field, lang, split, target,
                                        sample_rate, skip=skip_per_source)
            if clips:
                logger.info(f"Voice source '{name}' ({lang}): {len(clips)} clips (streaming).")
                mixed.extend(clips)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("gated", "401", "403", "authentication", "authorized", "access to")):
                # ゲート付きデータセット: HF ログイン + 規約同意が必要
                logger.warning(f"Voice dataset '{name}' requires HF auth/terms acceptance "
                               f"(run `huggingface-cli login` and accept terms on its HF page). Skipping.")
            else:
                logger.warning(f"Voice dataset '{name}' unavailable ({e}). Skipping (mix continues).")
    if mixed:
        random.Random(0).shuffle(mixed)
        logger.info(f"Voice corpus mixed: {len(mixed)} clips from "
                    f"{len({c['speaker_id'].split(':')[0] for c in mixed})} sources.")
    return mixed[:max_clips]


def load_voice_clips(max_clips=32, sample_rate=VOICE_SAMPLE_RATE, skip_per_source=0):
    """音声クリップ [{audio:[1,T], text, lang, speaker_id}] を返す。

    全モード（TEST/SMOKE/PROD/MAX）で実コーパスの混合のみを使う。合成/ダミー音声は
    使わない。実データが 1 件も取得できない場合は RuntimeError を送出して学習を止める
    （ネットワーク不通 / HF 認証切れ / 全ソース失敗などを黙って合成で埋めない）。

    skip_per_source: 各ソースの先頭から読み飛ばす件数。学習は 0（先頭から）を使うため、
                     評価側で学習件数分だけ skip するとホールドアウトになる。
    """
    clips = _mix_from_hf(max_clips, sample_rate, skip_per_source=skip_per_source)
    if not clips:
        raise RuntimeError(
            "実音声コーパスを 1 件も取得できませんでした。ネットワーク接続 / HuggingFace 認証 "
            "(`huggingface_hub.login`) / データセットの公開状況を確認してください。"
            "（方針により合成・ダミー音声へのフォールバックは無効化されています）")
    return clips


def _fit_length(wav, target_len):
    """波形 [1, T] を target_len にクロップ/パディング。"""
    T = wav.shape[-1]
    if T >= target_len:
        return wav[..., :target_len]
    return torch.nn.functional.pad(wav, (0, target_len - T))


class CodecDataset(Dataset):
    """コーデック学習用: 固定長の波形 [1, L] を返す。"""

    def __init__(self, clips, clip_frames=8):
        self.target_len = clip_frames * VOICE_CODEC_HOP
        self.waves = [_fit_length(c["audio"], self.target_len) for c in clips]

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, idx):
        return self.waves[idx]


def _fit_frames(z, n_frames):
    """latent [T', dim] を n_frames にクロップ/ゼロパディングして [n_frames, dim] にする。"""
    T = z.shape[0]
    if T >= n_frames:
        return z[:n_frames]
    return torch.nn.functional.pad(z, (0, 0, 0, n_frames - T))


class FlowTTSDataset(Dataset):
    """Flow Matching TTS 学習用: (text_ids[L_txt], latent[n_frames, dim]) を返す [音声編3.2]。

    コーデックの連続 latent（`encoder` 出力, 量子化前）を固定長へ整える。Flow Matching は
    固定長の系列を対象に非自己回帰で速度場を学習するため、可変長コード列を返す TTSDataset とは
    別に用意する（latent は連続値で、生成後に `codec.decoder` へ直接渡せる）。
    """

    def __init__(self, clips, codec, tokenizer, clip_frames=8, n_frames=None, max_text_len=32):
        self.samples = []
        codec.eval()
        device = next(codec.parameters()).device
        n_frames = n_frames or clip_frames
        max_len = clip_frames * VOICE_CODEC_HOP
        with torch.no_grad():
            for c in clips:
                wav = c["audio"][..., :max_len]
                if wav.shape[-1] < VOICE_CODEC_HOP:
                    continue                                             # 1 フレーム未満は除外
                wav = wav.unsqueeze(0).to(device)                        # [1,1,L]
                z = codec.encoder(codec._pad_to_hop(wav)).transpose(1, 2)[0].cpu()  # [T', dim]（連続）
                z = _fit_frames(z, n_frames)                             # [n_frames, dim]
                tagged = VOICE_LANG_TAG.format(c.get("lang", "ja")) + c["text"]
                ids = tokenizer.encode(tagged)[:max_text_len]
                if not ids:
                    continue
                self.samples.append((torch.tensor(ids, dtype=torch.long), z))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TTSDataset(Dataset):
    """AR コーデックLM 学習用: (text_ids[L_txt], codes[T', n_q]) を返す。
    codec でクリップをエンコードしてコード列を得る（codec は学習済み想定）。"""

    def __init__(self, clips, codec, tokenizer, clip_frames=8, max_text_len=32):
        self.samples = []
        codec.eval()
        device = next(codec.parameters()).device
        # 継続時間学習 [音声編3.4]: 固定長へパディングせず、各クリップの実長（上限 clip_frames）
        # を保つ。これにより EOS が「発話の実際の末尾」に付与され、テキスト適応的な長さを学べる。
        max_len = clip_frames * VOICE_CODEC_HOP
        with torch.no_grad():
            for c in clips:
                wav = c["audio"][..., :max_len].unsqueeze(0).to(device)  # [1,1,L]（上限のみクロップ）
                if wav.shape[-1] < VOICE_CODEC_HOP:
                    continue                                             # 1 フレーム未満は除外
                codes = codec.encode(wav)[0].cpu()                       # [T', n_q]（可変長）
                # 多言語対応 [音声編5.2]: テキスト先頭に言語IDタグを付与
                tagged = VOICE_LANG_TAG.format(c.get("lang", "ja")) + c["text"]
                ids = tokenizer.encode(tagged)[:max_text_len]
                if not ids:
                    continue
                self.samples.append((torch.tensor(ids, dtype=torch.long), codes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
