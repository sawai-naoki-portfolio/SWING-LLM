"""
swi_voice/pipeline_check.py

音声パイプラインの成功基準を自動判定する（swi_llm の pipeline_check 相当）。
- コーデック / TTS チェックポイントが生成されているか
- text → コード列 → 波形 の end-to-end スモークが有限値で通るか
非ゼロ終了で run_pipeline.ps1 が失敗検知する。
"""
import argparse
import os
import sys

# Parse args before importing swi_voice modules so SWI_MODE is set correctly
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default=os.environ.get("SWI_MODE", "TEST"))
args, _ = ap.parse_known_args()
os.environ["SWI_MODE"] = args.mode

import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.config import config
from swi_voice.utils.constants import VOICE_SAMPLE_RATE
from swi_voice.train.codec_train import CODEC_CKPT
from swi_voice.train.tts_train import TTS_CKPT
from swi_voice.safety.watermark import apply_watermark
from swi_voice.duplex.vad import detect_voice_activity

logger = setup_logger("voice.pipeline_check")


def _load_codec():
    """学習済みコーデックを CPU にロードする（e2e / 新規モジュールスモークで共用）。"""
    from swi_voice.codec import NeuralAudioCodec

    cc = torch.load(CODEC_CKPT, map_location="cpu", weights_only=False)
    c = cc["codec_cfg"]
    codec = NeuralAudioCodec(dim=c["dim"], hidden=c["hidden"],
                             n_quantizers=c["n_quantizers"], codebook_size=c["codebook_size"])
    codec.load_state_dict(cc["model"])
    codec.eval()
    return codec, c


def _e2e_smoke():
    from swi_voice.model import ARCodecLM
    from swi_llm.data.tokenizer import get_tokenizer

    device = "cpu"
    codec, _ = _load_codec()

    tc = torch.load(TTS_CKPT, map_location=device, weights_only=False)
    tcfg = tc["tts_cfg"]
    tok = get_tokenizer()
    model = ARCodecLM(text_vocab_size=tok.vocab_size, n_codebooks=tc["n_codebooks"],
                      codebook_size=tc["codebook_size"], dim=tcfg["dim"],
                      n_layers=tcfg["n_layers"], n_heads=tcfg["n_heads"], max_seq_len=tcfg["max_seq_len"])
    model.load_state_dict(tc["model"])
    model.eval()

    ids = torch.tensor([tok.encode("こんにちは")], dtype=torch.long)
    # 停止制御 [音声編3.4] 導入後、EOS で早期停止して 0 フレームになるのは正常系。
    # ここでは「生成→デコード経路が有限波形を返す」健全性を確認する（空なら probe で経路検証）。
    codes = model.generate(ids, max_frames=32)
    probe = codes if codes.shape[1] > 0 else torch.zeros((1, 4, model.n_codebooks), dtype=torch.long)
    wav = codec.decode(probe)
    
    # [V-4] 安全基盤の強制適用: 電子透かし埋め込み
    watermarked_wav = apply_watermark(wav, payload="SWI_TEST")
    
    # VAD の動作確認 (クラッシュせずにboolを返すか)
    vad_result = detect_voice_activity(watermarked_wav, threshold=0.1)
    logger.info(f"VAD Test Result on synthetic waveform: {vad_result}")
    
    return watermarked_wav.shape[-1] > 0 and bool(torch.isfinite(watermarked_wav).all())


def _new_module_checks():
    """V-16: 2026-07-10 追加モジュール（wake_word / cascade_router / s2s / tool_bridge）の
    配線スモーク。学習品質ではなく「経路が例外なく有限値で通るか」を検証する。
    各項目を個別の (name, ok) で返し、失敗箇所をログから特定できるようにする。
    """
    import asyncio

    checks = []

    # 1) WakeWordDetector: テンプレートを埋め込んだ音声で検出できる + ndarray 入力に耐える (V-14)
    try:
        import numpy as np
        from swi_voice.duplex.wake_word import WakeWordDetector

        t = torch.linspace(0.0, 1.0, 800)
        template = torch.sin(2 * torch.pi * 440.0 * t)
        # オフセット 400 は hop=80 の倍数なので完全一致ウィンドウが必ず走査される（score=1.0）
        audio = torch.cat([torch.zeros(400), template, torch.zeros(400)])
        det = WakeWordDetector(template, threshold=0.9, hop=80)
        hit = det.detect(audio)
        ndarray_safe = det.detect(np.zeros(1600, dtype=np.float32)) is False
        checks.append(("wake_word detect smoke", bool(hit) and ndarray_safe))
    except Exception:
        logger.exception("wake_word smoke failed")
        checks.append(("wake_word detect smoke", False))

    # 2) AgentRouter: 静的（未注入 → cascade）/ 動的（casual → s2s）の両ルーティング
    try:
        from swi_voice.agents.cascade_router import AgentRouter, ROUTE_CASCADE, ROUTE_S2S

        static_ok = AgentRouter().route_request("今日の天気を調べて") == ROUTE_CASCADE
        dynamic = AgentRouter(cascade_agent=object(), s2s_agent=object(),
                              intent_classifier=lambda ctx: "casual")
        dynamic_ok = (dynamic.route_request("ねえねえ") == ROUTE_S2S and
                      AgentRouter(cascade_agent=object(), s2s_agent=object(),
                                  intent_classifier=lambda ctx: "task").route_request("計算して") == ROUTE_CASCADE)
        checks.append(("agent_router static+dynamic smoke", static_ok and dynamic_ok))
    except Exception:
        logger.exception("agent_router smoke failed")
        checks.append(("agent_router static+dynamic smoke", False))

    # 3) S2SVoiceAgent: 学習済み codec + 極小 S2S バックボーン（未学習）で波形→波形の配線検証
    try:
        from swi_voice.agents.cascade_router import S2SVoiceAgent
        from swi_voice.duplex.s2s_backbone import SpeechToSpeechBackbone

        codec, ccfg = _load_codec()
        backbone = SpeechToSpeechBackbone(
            n_codebooks=ccfg["n_quantizers"], codebook_size=ccfg["codebook_size"],
            dim=64, n_layers=1, n_heads=2, num_experts=2, top_k=1, max_seq_len=64)
        s2s = S2SVoiceAgent(backbone, codec, temperature=0.0)   # argmax で決定的に
        # codec.hop の 8 倍 = ちょうど 8 フレームの入力（モード非依存でフレーム数を保証）
        n = codec.hop * 8
        wav_in = torch.sin(2 * torch.pi * 220.0 * torch.linspace(0.0, 0.1, n))
        out = s2s.handle(wav_in)
        checks.append(("s2s agent waveform smoke",
                       out is not None and out.numel() > 0 and bool(torch.isfinite(out).all())))
    except Exception:
        logger.exception("s2s agent smoke failed")
        checks.append(("s2s agent waveform smoke", False))

    # 4) CascadeVoiceAgent: 実 ASR（faster-whisper, fix 08）があれば実物 + 実音声クリップで、
    #    無ければ従来どおりモック ASR で ASR→LLM→TTS の配線を検証する（明示ログ）。
    label = "cascade agent wiring smoke (mock ASR)"
    try:
        from swi_voice.agents.cascade_router import CascadeVoiceAgent
        from swi_voice.asr.whisper_asr import get_default_asr

        class _MockTTS:
            def synthesize(self, text, **kwargs):
                return b"WAV" + text.encode("utf-8")

        asr = get_default_asr(eager=True)
        user_audio = None
        if asr is not None:
            # 実クリップの取得に失敗したら（オフライン等）モック経路へ縮退する（C-4）。
            try:
                from swi_voice.data.audio_dataset import load_voice_clips
                user_audio = load_voice_clips(max_clips=1, sample_rate=VOICE_SAMPLE_RATE)[0]["audio"]
                label = "cascade agent wiring smoke (real ASR)"
                logger.info(f"cascade smoke: real ASR (faster-whisper '{asr.model_size}') on a real "
                            "clip — the ASR->LLM->TTS cascade runs on actual speech (fix 08).")
            except Exception:
                logger.exception("cascade smoke: could not load a real clip; "
                                 "falling back to the mock-ASR wiring check.")
                asr = None
        if asr is None:
            logger.info("cascade smoke: real ASR unavailable; "
                        "using a mock ASR to verify the ASR->LLM->TTS wiring only (V-16).")

            class _MockASR:
                def transcribe(self, audio):
                    return "こんにちは"

            asr = _MockASR()
            user_audio = torch.zeros(1600)

        cascade = CascadeVoiceAgent(asr, lambda text: f"応答: {text}", _MockTTS())
        wav_bytes = cascade.handle(user_audio)
        checks.append((label,
                       isinstance(wav_bytes, (bytes, bytearray)) and len(wav_bytes) > 0))
    except Exception:
        logger.exception("cascade agent smoke failed")
        checks.append((label, False))

    # 5) tool_bridge: 実 AgentBase + calculate で、つなぎ発話→ツール実行→結果返却の一連
    try:
        from swi_llm.agent.base import AgentBase
        from swi_llm.agent.tools import calculate, TOOL_SCHEMAS
        from swi_llm.utils.constants import TOOL_CALCULATE
        from swi_voice.duplex.tool_bridge import handle_tool_call_with_filler

        agent = AgentBase()
        agent.register_tool(TOOL_CALCULATE, calculate, TOOL_SCHEMAS[TOOL_CALCULATE])
        spoken = []

        async def _speak(text):
            spoken.append(text)

        tool_call = ('<tool_call>{"name": "%s", "arguments": {"expression": "2 + 2"}}</tool_call>'
                     % TOOL_CALCULATE)
        result, filler_spoken = asyncio.run(
            handle_tool_call_with_filler(agent, tool_call, _speak))
        checks.append(("tool_bridge filler+tool smoke",
                       filler_spoken is True and len(spoken) == 1
                       and result is not None and "4" in result))
    except Exception:
        logger.exception("tool_bridge smoke failed")
        checks.append(("tool_bridge filler+tool smoke", False))

    return checks


def main():
    checks = [
        ("codec checkpoint generated", os.path.exists(CODEC_CKPT)),
        ("tts checkpoint generated", os.path.exists(TTS_CKPT)),
    ]
    if all(ok for _, ok in checks):
        try:
            checks.append(("e2e text->codes->waveform", _e2e_smoke()))
        except Exception:
            logger.exception("E2E smoke failed")
            checks.append(("e2e text->codes->waveform", False))
        # V-16: 新規モジュール（項目ごとに内部で例外処理し (name, ok) を返す）
        checks.extend(_new_module_checks())

    for name, ok in checks:
        logger.info(f"[{'PASS' if ok else 'FAIL'}] {name}")
    passed = all(ok for _, ok in checks)
    n_pass = sum(1 for _, ok in checks if ok)
    logger.info(f"VOICE PIPELINE CHECK: {'PASS' if passed else 'FAIL'} ({n_pass}/{len(checks)} checks passed)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
