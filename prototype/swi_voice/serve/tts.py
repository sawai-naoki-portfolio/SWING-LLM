"""
swi_voice/serve/tts.py

WebUI 用の TTS 推論（テキスト → 音声波形 → WAV バイト列）[音声編7]。

学習済みの swi_voice コーデック + TTS モデルを（初回のみ）遅延ロードし、テキストから
波形を生成 → soundfile で WAV にエンコードして返す。3AI 統合 WebUI から /api/tts で呼ばれる。

エンジン（SWI_TTS_ENGINE, NEW-V2/V3）:
  - ar     (既定): AR コーデック LM（VALL-E 系）。逐次生成のストリーミング・クローニング対応。
  - flow   : Flow Matching TTS（F5/E2 系, 非自己回帰）。voice_flow_checkpoint を使用。
             クローニングは参照 latent のインペインティング文脈で対応。
  - hybrid : AR(意味) + FM(音響) の 2 段 [音声編3.3]。voice_hybrid_checkpoint を使用。
  指定エンジンの ckpt が無ければ WARNING を出して ar へ縮退する（C-4）。

- スレッドセーフな遅延ロード（web/models.py と同型）。
- チェックポイントは現在の SWI_MODE サフィックスを優先し、無ければフォールバック。
- 音声モデルが未学習でも WebUI 全体は起動できるよう、ロード失敗は呼び出し側で処理する。
"""
import io
import os
import threading

import numpy as np
import torch
import soundfile as sf

from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer
from swi_voice.utils.config import config
from swi_voice.utils.constants import (
    VOICE_SAMPLE_RATE, VOICE_LANG_TAG, VOICE_WATERMARK_PAYLOAD,
    VOICE_STREAM_OVERLAP_FRAMES, VOICE_CODEC_HOP, VOICE_MAX_PROMPT_FRAMES,
    VOICE_TTS_ENGINE_ENV, VOICE_NONAR_STREAM_CHUNK_SEC, VOICE_FLOW_SAMPLE_STEPS)
from swi_voice.codec import NeuralAudioCodec
from swi_voice.model import ARCodecLM
from swi_voice.model.cloning import require_reference_consent
from swi_voice.safety.watermark import apply_watermark

logger = setup_logger("voice.tts")

# swi_voice/serve/tts.py → swi_voice → prototype（チェックポイント保存先）
_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_CODEC_CKPT_BASE = "voice_codec_checkpoint"
_TTS_CKPT_BASE = "voice_tts_checkpoint"
_FLOW_CKPT_BASE = "voice_flow_checkpoint"
_HYBRID_CKPT_BASE = "voice_hybrid_checkpoint"

_tts_instance = None
_lock = threading.Lock()


def resolve_tts_engine():
    """SWI_TTS_ENGINE から serving エンジンを確定する（ar | flow | hybrid, 既定 ar）。"""
    engine = os.environ.get(VOICE_TTS_ENGINE_ENV, "").strip().lower() or "ar"
    if engine not in ("ar", "flow", "hybrid"):
        logger.warning(f"Unknown TTS engine '{engine}' ({VOICE_TTS_ENGINE_ENV}); using 'ar'.")
        engine = "ar"
    return engine


def _suffix_priority():
    suffix = config.get("ckpt_suffix", "")
    order = [suffix]
    for fb in ["", "_SMOKE", "_TEST", "_MAX"]:
        if fb not in order:
            order.append(fb)
    return order


def _find_ckpt(base):
    for s in _suffix_priority():
        p = os.path.join(_PROTOTYPE_DIR, f"{base}{s}.pt")
        if os.path.exists(p):
            return p
    return None


def voice_available():
    """音声モデルのチェックポイントが揃っているか（フロントの機能出し分け用）。"""
    return _find_ckpt(_CODEC_CKPT_BASE) is not None and _find_ckpt(_TTS_CKPT_BASE) is not None


class SWIVoiceTTS:
    def __init__(self, device):
        self.device = device
        self.tokenizer = get_tokenizer()

        codec_path = _find_ckpt(_CODEC_CKPT_BASE)
        tts_path = _find_ckpt(_TTS_CKPT_BASE)
        if not codec_path or not tts_path:
            raise FileNotFoundError(
                "voice checkpoints not found (run swi_voice codec_train / tts_train first)")

        # SEC-2: pickle 由来の任意コード実行を防ぐ（Web 到達経路。web/models.py と同方針）。
        # 保存内容は state_dict + プリミティブ設定 dict のみのため weights_only=True で読める。
        cc = torch.load(codec_path, map_location=device, weights_only=True)
        c = cc["codec_cfg"]
        self.codec = NeuralAudioCodec(dim=c["dim"], hidden=c["hidden"],
                                      n_quantizers=c["n_quantizers"], codebook_size=c["codebook_size"]).to(device)
        self.codec.load_state_dict(cc["model"])
        self.codec.eval()

        tc = torch.load(tts_path, map_location=device, weights_only=True)   # SEC-2
        tcfg = tc["tts_cfg"]
        self.model = ARCodecLM(text_vocab_size=self.tokenizer.vocab_size,
                               n_codebooks=tc["n_codebooks"], codebook_size=tc["codebook_size"],
                               dim=tcfg["dim"], n_layers=tcfg["n_layers"], n_heads=tcfg["n_heads"],
                               max_seq_len=tcfg["max_seq_len"]).to(device)
        self.model.load_state_dict(tc["model"])
        self.model.eval()

        # NEW-V2/V3: 非自己回帰エンジン（flow / hybrid）。指定エンジンの ckpt が無ければ
        # ar へ縮退する（C-4。「学習していないのに指定した」構成ミスを WARNING で可視化）。
        self.engine = resolve_tts_engine()
        self.flow_model = None
        self.hybrid_model = None
        self.flow_steps = VOICE_FLOW_SAMPLE_STEPS
        if self.engine == "flow":
            self._load_flow_engine(device)
        elif self.engine == "hybrid":
            self._load_hybrid_engine(device)
        logger.info(f"SWIVoiceTTS loaded (engine={self.engine}, "
                    f"codec={os.path.basename(codec_path)}, "
                    f"tts={os.path.basename(tts_path)}, device={device}).")

    def _load_flow_engine(self, device):
        """Flow Matching TTS（NEW-V2）をロードする。ckpt 不在なら ar へ縮退。"""
        path = _find_ckpt(_FLOW_CKPT_BASE)
        if path is None:
            logger.warning("SWI_TTS_ENGINE=flow but voice_flow_checkpoint is missing "
                           "(run `run.bat voice --stage flow` first). Falling back to 'ar'.")
            self.engine = "ar"
            return
        from swi_voice.model import FlowTTS
        fc = torch.load(path, map_location=device, weights_only=True)       # SEC-2
        cfg = fc["flow_cfg"]
        self.flow_model = FlowTTS(text_vocab_size=self.tokenizer.vocab_size,
                                  latent_dim=fc["latent_dim"], dim=cfg["dim"],
                                  n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                                  max_frames=cfg["max_frames"]).to(device)
        self.flow_model.load_state_dict(fc["model"])
        self.flow_model.eval()
        self.flow_steps = cfg.get("sample_steps", VOICE_FLOW_SAMPLE_STEPS)
        self.flow_frames = cfg.get("n_frames") or 64          # 学習時のクリップ長（生成の既定長）
        logger.info(f"Flow TTS engine ready ({os.path.basename(path)}, "
                    f"n_frames={self.flow_frames}, steps={self.flow_steps}).")

    def _load_hybrid_engine(self, device):
        """ハイブリッド AR+FM（NEW-V3）をロードする。ckpt 不在なら ar へ縮退。"""
        path = _find_ckpt(_HYBRID_CKPT_BASE)
        if path is None:
            logger.warning("SWI_TTS_ENGINE=hybrid but voice_hybrid_checkpoint is missing "
                           "(run `run.bat voice --stage hybrid` first). Falling back to 'ar'.")
            self.engine = "ar"
            return
        from swi_voice.model.hybrid_ar_fm import HybridARFM
        hc = torch.load(path, map_location=device, weights_only=True)       # SEC-2
        cfg = hc["hybrid_cfg"]
        self.hybrid_model = HybridARFM(
            text_vocab_size=self.tokenizer.vocab_size,
            semantic_codebook_size=hc["semantic_codebook_size"],
            acoustic_latent_dim=hc["latent_dim"],
            ar_dim=cfg["ar_dim"], ar_layers=cfg["ar_layers"], ar_heads=cfg["ar_heads"],
            fm_dim=cfg["fm_dim"], fm_layers=cfg["fm_layers"], fm_heads=cfg["fm_heads"],
            fm_max_frames=cfg["fm_max_frames"], max_sem_len=cfg["max_sem_len"]).to(device)
        self.hybrid_model.load_state_dict(hc["hybrid"])
        self.hybrid_model.eval()
        self.flow_steps = cfg.get("sample_steps", VOICE_FLOW_SAMPLE_STEPS)
        self.flow_frames = cfg.get("n_frames") or 64
        logger.info(f"Hybrid AR+FM engine ready ({os.path.basename(path)}, "
                    f"n_frames={self.flow_frames}, steps={self.flow_steps}).")

    def _decode_ref_wav(self, ref_audio_bytes):
        """参照音声バイト列 → 波形 [1, 1, T]（VOICE_SAMPLE_RATE・モノラル・正規化）。

        soundfile で復号（ffmpeg 非依存）し、audio_dataset と同じ前処理で整える。
        AR（コード列プロンプト）と flow（連続 latent 文脈）で共用する。
        """
        import soundfile as sf
        import torchaudio
        data, sr = sf.read(io.BytesIO(ref_audio_bytes), dtype="float32", always_2d=False)
        wav = torch.as_tensor(data, dtype=torch.float32)
        if wav.ndim > 1:
            wav = wav.mean(dim=-1)                                 # モノラル化
        if sr != VOICE_SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, VOICE_SAMPLE_RATE)
        peak = wav.abs().max()
        if peak > 1e-6:
            wav = wav / peak * 0.95                                # 音量差の正規化
        return wav.view(1, 1, -1).to(self.device)

    def _ref_audio_to_codes(self, ref_audio_bytes):
        """参照音声バイト列 → コーデックコード列 [1, Lp, n_q]（AR クローニングの音響プロンプト）。"""
        codes = self.codec.encode(self._decode_ref_wav(ref_audio_bytes))   # [1, Lp, n_q]
        if codes.shape[1] > VOICE_MAX_PROMPT_FRAMES:
            codes = codes[:, :VOICE_MAX_PROMPT_FRAMES]            # プロンプト長を制限
        return codes

    def _ref_audio_to_latent(self, ref_audio_bytes):
        """参照音声バイト列 → 連続 latent [1, Lp, dim]（flow クローニングの可視文脈）。"""
        wav = self._decode_ref_wav(ref_audio_bytes)
        z = self.codec.encoder(self.codec._pad_to_hop(wav)).transpose(1, 2)   # [1, Lp, dim]
        if z.shape[1] > VOICE_MAX_PROMPT_FRAMES:
            z = z[:, :VOICE_MAX_PROMPT_FRAMES]
        return z

    def _prepare_inputs(self, text, lang, ref_audio, ref_speaker_id=None):
        """テキスト/参照音声を (input_ids, prompt_codes) に整える（synthesize 系で共有）。

        参照音声が指定された場合は、波形を復号する前に voice_cloning の同意ゲートを通す
        [音声編8.1, 実装編4.1]。同意が無ければ `CloningConsentError` を送出する。ゲートは
        下の try/except より前に置き、復号失敗のフォールバックに巻き込まれないようにする
        （同意拒否を無言でクローニングなし合成に落とさない）。
        """
        tagged = VOICE_LANG_TAG.format(lang) + str(text)
        ids = self.tokenizer.encode(tagged)[:64]
        if not ids:
            ids = self.tokenizer.encode(VOICE_LANG_TAG.format(lang) + "。")[:64]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        prompt_codes = None
        if ref_audio:
            require_reference_consent(ref_speaker_id)              # [音声編8.1] 同意ファースト
            try:
                prompt_codes = self._ref_audio_to_codes(ref_audio)
                logger.info(f"Zero-shot cloning: reference prompt {prompt_codes.shape[1]} frames.")
            except Exception as e:
                logger.warning(f"Reference audio decode failed ({e}); falling back to non-cloning TTS.")
                prompt_codes = None
        return input_ids, prompt_codes

    @staticmethod
    def _wav_bytes(wav_np):
        # 安全基盤の強制適用 [音声編8.2]: 全 TTS 出力（一括/ストリーミング）の唯一の
        # 出口で電子透かしを埋め込み、AI 生成音声であることの機械可読な証跡を残す。
        try:
            wav_t = torch.from_numpy(np.ascontiguousarray(wav_np, dtype=np.float32))
            wav_np = apply_watermark(wav_t, payload=VOICE_WATERMARK_PAYLOAD).cpu().numpy()
        except Exception:
            logger.exception("Watermark embedding failed; emitting audio without watermark.")
        buf = io.BytesIO()
        sf.write(buf, wav_np, VOICE_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # ---- 非自己回帰エンジン（flow / hybrid, NEW-V2/V3）の生成 --------------------------
    def _prepare_nonar_inputs(self, text, lang, ref_audio, ref_speaker_id):
        """(input_ids, ref_latent) を整える。同意ゲートの順序は _prepare_inputs と同一。"""
        tagged = VOICE_LANG_TAG.format(lang) + str(text)
        ids = self.tokenizer.encode(tagged)[:64]
        if not ids:
            ids = self.tokenizer.encode(VOICE_LANG_TAG.format(lang) + "。")[:64]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        ref_latent = None
        if ref_audio:
            require_reference_consent(ref_speaker_id)              # [音声編8.1] 同意ファースト
            if self.engine == "hybrid":
                # 現行の HybridARFM.generate は音響プロンプト未対応（意味→音響の 2 段のみ）。
                # 無言でクローニングなしに落とさず、未対応であることを明示する。
                logger.warning("hybrid エンジンは参照音声クローニング未対応です。"
                               "参照を無視して通常合成します（ar / flow を使ってください）。")
            else:
                try:
                    ref_latent = self._ref_audio_to_latent(ref_audio)
                    logger.info(f"Zero-shot cloning (flow): reference context "
                                f"{ref_latent.shape[1]} frames.")
                except Exception as e:
                    logger.warning(f"Reference audio decode failed ({e}); "
                                   "falling back to non-cloning TTS.")
        return input_ids, ref_latent

    @torch.no_grad()
    def _generate_nonar_wav(self, input_ids, ref_latent, max_frames):
        """flow / hybrid で波形 [T] (numpy, 正規化済み) を一括生成する。"""
        if self.engine == "flow":
            from swi_voice.model.flow_tts import sample as flow_sample
            n_frames = min(max_frames, self.flow_frames, self.flow_model.max_frames)
            latent = flow_sample(self.flow_model, input_ids, n_frames=n_frames,
                                 ref_latent=ref_latent, n_steps=self.flow_steps,
                                 device=self.device)               # [1, n_frames, dim]
        else:                                                       # hybrid
            n_frames = min(max_frames, self.flow_frames, self.hybrid_model.acoustic_fm.max_frames)
            latent, _sem = self.hybrid_model.generate(input_ids, n_frames=n_frames,
                                                      n_steps=self.flow_steps)
        wav = self.codec.decoder(latent.transpose(1, 2)).squeeze().detach().cpu().float().numpy()
        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = (wav / peak) * 0.95                              # クリップ防止の正規化
        return wav

    @torch.no_grad()
    def synthesize(self, text, lang="ja", max_frames=200, ref_audio=None,
                   temperature=0.8, top_k=50, ref_speaker_id=None):
        """テキスト → WAV バイト列（16-bit PCM, VOICE_SAMPLE_RATE）。

        ref_audio（バイト列）を渡すと、その声色を模倣するゼロショットクローニングで
        生成する [音声編4.1]。この経路は `ref_speaker_id` に対する voice_cloning の
        明示同意を必須とし、無ければ `CloningConsentError` を送出する [音声編8.1]。
        復号失敗時のみ通常 TTS へフォールバックする。

        エンジンが flow / hybrid のときは非自己回帰の一括生成（NEW-V2/V3）。
        temperature / top_k は意味 LM（hybrid）にのみ意味を持ち、flow では使わない。
        """
        if self.engine != "ar":
            input_ids, ref_latent = self._prepare_nonar_inputs(text, lang, ref_audio, ref_speaker_id)
            return self._wav_bytes(self._generate_nonar_wav(input_ids, ref_latent, max_frames))

        input_ids, prompt_codes = self._prepare_inputs(text, lang, ref_audio, ref_speaker_id)
        codes = self.model.generate(input_ids, prompt_codes=prompt_codes, max_frames=max_frames,
                                    temperature=temperature, top_k=top_k)   # [1, T', n_q]
        if codes.shape[1] == 0:
            # 生成が空なら 0.3 秒の無音を返す（フロントの再生が失敗しないよう保証）
            wav = np.zeros(int(VOICE_SAMPLE_RATE * 0.3), dtype=np.float32)
        else:
            wav = self.codec.decode(codes).squeeze().detach().cpu().float().numpy()
            peak = np.abs(wav).max()
            if peak > 1e-6:
                wav = (wav / peak) * 0.95                                  # クリップ防止の正規化
        return self._wav_bytes(wav)

    @torch.no_grad()
    def synthesize_stream(self, text, lang="ja", chunk_frames=16, max_frames=200,
                          ref_audio=None, temperature=0.8, top_k=50, ref_speaker_id=None):
        """テキスト → WAV チャンクを逐次 yield（ストリーミング推論 [音声編7.1]）。

        コードをチャンク生成し、各チャンクを独立した WAV として即座に返す。フロントは
        受信順に逐次再生でき、TTFB（初回音声までの時間）が短縮される。デコーダ出力は
        tanh で [-1,1] のため、チャンク間の音量段差を避けて固定ゲインで整える
        （全体ピーク正規化はストリーミングでは不可能なため）。

        V-5: 各チャンクを独立デコードするとデコーダが境界で状態を失いクリック/段差が出る。
        直前チャンク末尾 VOICE_STREAM_OVERLAP_FRAMES フレームを左文脈としてデコーダへ渡し
        （連続文脈を供給）、その重複区間をトリムして出力することで境界の不連続を軽減する。

        入力の確定（＝クローニングの同意ゲート）は呼び出し時点で行い、チャンク生成本体は
        `_stream_chunks` に委ねる。ジェネレータ本体に同意チェックを置くと最初の next() まで
        発火せず、ストリーム開始後にしか拒否できないため [音声編8.1]。

        エンジンが flow / hybrid のときは非自己回帰の一括生成後に固定長チャンクへ分割して
        返す（NEW-V2/V3。TTFB は全生成時間になるが、応答プロトコルは ar と互換）。
        """
        if self.engine != "ar":
            input_ids, ref_latent = self._prepare_nonar_inputs(text, lang, ref_audio, ref_speaker_id)
            return self._stream_nonar(input_ids, ref_latent, max_frames)

        input_ids, prompt_codes = self._prepare_inputs(text, lang, ref_audio, ref_speaker_id)
        return self._stream_chunks(input_ids, prompt_codes, chunk_frames, max_frames,
                                   temperature, top_k)

    @torch.no_grad()
    def _stream_nonar(self, input_ids, ref_latent, max_frames):
        """flow / hybrid: 一括生成した波形を固定長チャンクで逐次 yield する。"""
        wav = self._generate_nonar_wav(input_ids, ref_latent, max_frames)
        chunk = max(1, int(VOICE_SAMPLE_RATE * VOICE_NONAR_STREAM_CHUNK_SEC))
        if wav.shape[0] == 0:
            yield self._wav_bytes(np.zeros(int(VOICE_SAMPLE_RATE * 0.2), dtype=np.float32))
            return
        for start in range(0, wav.shape[0], chunk):
            yield self._wav_bytes(wav[start:start + chunk])

    @torch.no_grad()
    def _stream_chunks(self, input_ids, prompt_codes, chunk_frames, max_frames, temperature, top_k):
        """確定済み入力から WAV チャンクを逐次 yield する（`synthesize_stream` の本体）。"""
        overlap = VOICE_STREAM_OVERLAP_FRAMES
        hop = getattr(self.codec, "hop", VOICE_CODEC_HOP)
        prev_tail = None                         # 直前チャンク末尾 overlap フレーム（左文脈）
        any_chunk = False
        for codes_chunk in self.model.generate_stream(
                input_ids, prompt_codes=prompt_codes, chunk_frames=chunk_frames,
                max_frames=max_frames, temperature=temperature, top_k=top_k):
            if prev_tail is not None and overlap > 0:
                dec_in = torch.cat([prev_tail, codes_chunk], dim=1)   # 左文脈 + 本チャンク
                wav = self.codec.decode(dec_in).squeeze().detach().cpu().float().numpy()
                trim = min(overlap * hop, max(0, wav.shape[0] - 1))   # 重複した先頭を除去
                wav = wav[trim:]
            else:
                wav = self.codec.decode(codes_chunk).squeeze().detach().cpu().float().numpy()
            wav = np.clip(wav, -1.0, 1.0) * 0.95
            prev_tail = codes_chunk[:, -overlap:] if overlap > 0 else None
            any_chunk = True
            yield self._wav_bytes(wav)
        if not any_chunk:
            # 生成が空（即 EOS 等）でも再生が失敗しないよう短い無音を1チャンク返す
            yield self._wav_bytes(np.zeros(int(VOICE_SAMPLE_RATE * 0.2), dtype=np.float32))


def get_tts():
    """SWIVoiceTTS を（初回のみ）ロードして返す。スレッドセーフ。"""
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance
    with _lock:
        if _tts_instance is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _tts_instance = SWIVoiceTTS(device)
    return _tts_instance
