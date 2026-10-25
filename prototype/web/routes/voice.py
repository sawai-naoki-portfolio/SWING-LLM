"""
web/routes/voice.py

3AI 統合 WebUI の音声合成（TTS）エンドポイント。
学習済み swi_voice モデルでテキストを音声（WAV）に変換して返す。
音声モデル未学習時は 503 を返し、フロントはブラウザ音声にフォールバックする。
"""
import asyncio
import base64
import threading
import json
from functools import partial

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import MAX_PROMPT_LENGTH

logger = setup_logger("web_voice")
router = APIRouter()

# 参照音声 base64 の最大長（~9MB 相当、数十秒の WAV）。巨大入力による OOM/DoS を防ぐ。
_MAX_REF_AUDIO_B64 = 12_000_000


class TTSRequest(BaseModel):
    text: str = ''
    lang: str = 'ja'
    ref_audio: str = ''   # 参照音声（base64 / data URL）。ゼロショットクローニング用 [音声編4.1]。
    # 参照音声の話者ID。ref_audio を使う場合は必須で、この話者に voice_cloning の明示同意が
    # 登録されていなければ 403 で拒否する（同意ファースト [音声編8.1]）。
    ref_speaker_id: str = ''


def _clean_text(req):
    """(text, error_response)。text が空なら error を返す。"""
    text = (req.text or '').strip()
    if not text:
        return None, JSONResponse({'error': 'text is required'}, status_code=400)
    if len(text) > MAX_PROMPT_LENGTH:
        text = text[:MAX_PROMPT_LENGTH]
    return text, None


def _extract_ref(req):
    """参照音声 base64 を復号。(ref_bytes|None, error_response|None)。"""
    if not req.ref_audio:
        return None, None
    b64 = req.ref_audio.strip()
    if b64.startswith('data:') and ',' in b64:
        b64 = b64.split(',', 1)[1]
    if len(b64) > _MAX_REF_AUDIO_B64:
        return None, JSONResponse({'error': 'reference audio too large'}, status_code=400)
    try:
        return base64.b64decode(b64), None
    except Exception:
        return None, JSONResponse({'error': 'invalid reference audio'}, status_code=400)


def _consent_denied_response(exc):
    """クローニング同意の拒否 → 403（クライアントは参照音声なしで再試行できる）。"""
    logger.warning(f"Voice cloning request denied: {exc}")
    return JSONResponse({'error': str(exc)}, status_code=403)


@router.get("/api/voice/status")
async def api_voice_status():
    """音声モデルが利用可能か（フロントの機能出し分け用）。

    engine は構成上の選択値（SWI_TTS_ENGINE, NEW-V2/V3）。指定エンジンの ckpt が無い場合、
    実際のロード時に ar へ縮退することがある（ログで可視化される）。
    """
    try:
        from swi_voice.serve.tts import voice_available, resolve_tts_engine
        return {"available": bool(voice_available()), "engine": resolve_tts_engine()}
    except Exception:
        logger.exception("voice status check failed")
        return {"available": False, "engine": "ar"}


# =====================================================================
# ASR（音声 → テキスト）: fix 08(3) / voice roadmap step3
# 既存 /api/generate・/api/tts と組み合わせることで、クライアントから
# ASR→LLM→TTS のカスケード音声対話を構成できる（API 契約は追加のみ）。
# =====================================================================

class ASRRequest(BaseModel):
    # base64（data URL 可）の音声ファイルバイト列。soundfile が読める形式（wav/flac/ogg 等）。
    audio: str = ''
    # 転写言語（既定 ja）。faster-whisper にそのまま渡す。
    lang: str = 'ja'


@router.get("/api/asr/status")
async def api_asr_status():
    """実 ASR（faster-whisper）が利用可能か（フロント/モバイルの機能出し分け用）。

    import 可否のみを見る軽量チェックで、モデル重みのロードは行わない。
    """
    try:
        from swi_voice.asr.whisper_asr import asr_available, get_default_asr
        available = bool(asr_available())
        model = None
        if available:
            asr = get_default_asr()          # 遅延ロードのままインスタンスだけ得る
            model = asr.model_size if asr is not None else None
            available = asr is not None
        return {"available": available, "model": model}
    except Exception:
        logger.exception("asr status check failed")
        return {"available": False, "model": None}


@router.post("/api/asr")
async def api_asr(req: ASRRequest):
    """音声（base64）→ テキスト（faster-whisper による実転写）。

    ASR が使えない環境（faster-whisper 未導入 / SWI_ASR=0）では 503 を返し、
    クライアントは音声入力 UI を隠す（/api/asr/status で事前判定できる）。
    """
    if not req.audio:
        return JSONResponse({'error': 'audio is required'}, status_code=400)
    b64 = req.audio.strip()
    if b64.startswith('data:') and ',' in b64:
        b64 = b64.split(',', 1)[1]
    if len(b64) > _MAX_REF_AUDIO_B64:
        return JSONResponse({'error': 'audio too large'}, status_code=400)
    try:
        audio_bytes = base64.b64decode(b64)
    except Exception:
        return JSONResponse({'error': 'invalid audio'}, status_code=400)

    from swi_voice.asr.whisper_asr import get_default_asr
    asr = get_default_asr()
    if asr is None:
        return JSONResponse({'error': 'ASR is not available on this server'}, status_code=503)

    try:
        import io
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=True)
    except Exception:
        return JSONResponse({'error': 'unsupported audio format'}, status_code=400)

    import torch
    wav = torch.from_numpy(data).mean(dim=1)      # [T] モノラル化

    loop = asyncio.get_event_loop()
    try:
        # 初回はモデルロード（数秒〜、重み未取得なら DL）を含む。executor で回して
        # イベントループを塞がない。ロード失敗はここで 500 として返る。
        text = await loop.run_in_executor(
            None, partial(asr.transcribe, wav, sample_rate=int(sr),
                          language=(req.lang or None)))
    except Exception as e:
        logger.exception("ASR transcription failed")
        return JSONResponse({'error': str(e)}, status_code=500)
    return {"text": text}


@router.post("/api/tts")
async def api_tts(req: TTSRequest):
    """テキスト → swi_voice で音声合成 → WAV を返す（一括）。

    ref_audio（base64）を渡すと、その声色を模倣するゼロショットクローニングで合成する。
    その場合 ref_speaker_id は必須で、voice_cloning の明示同意が無ければ 403 [音声編8.1]。
    """
    text, err = _clean_text(req)
    if err:
        return err
    ref_bytes, err = _extract_ref(req)
    if err:
        return err

    from swi_voice.model.cloning import CloningConsentError
    try:
        from swi_voice.serve.tts import get_tts
        tts = get_tts()
        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(
            None, partial(tts.synthesize, text, req.lang, ref_audio=ref_bytes,
                          ref_speaker_id=req.ref_speaker_id))
        return Response(content=wav_bytes, media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})
    except CloningConsentError as e:
        return _consent_denied_response(e)
    except FileNotFoundError as e:
        # 音声モデル未学習: フロントはブラウザ音声にフォールバックする
        logger.warning(f"TTS unavailable: {e}")
        return JSONResponse({'error': 'voice model not trained yet'}, status_code=503)
    except Exception as e:
        logger.exception("TTS synthesis failed")
        return JSONResponse({'error': str(e)}, status_code=500)


@router.post("/api/tts_stream")
async def api_tts_stream(req: TTSRequest):
    """テキスト → WAV チャンクを逐次ストリーミング（ストリーミング推論 [音声編7.1]）。

    改行区切りで各行 = 1 チャンクの base64(WAV)。フロントは受信順に逐次再生でき、
    TTFB（初回音声までの時間）が短縮される。ref_audio でクローニングも併用可
    （その場合 ref_speaker_id は必須。同意が無ければストリーム開始前に 403）。
    """
    text, err = _clean_text(req)
    if err:
        return err
    ref_bytes, err = _extract_ref(req)
    if err:
        return err

    from swi_voice.model.cloning import CloningConsentError
    try:
        from swi_voice.serve.tts import get_tts
        tts = get_tts()            # 未学習なら FileNotFoundError（下でフロントがフォールバック）
        # 同意ゲートは synthesize_stream の呼び出し時点で発火する（生成はまだ走らない）。
        # ストリーム応答を返す前にここで拒否を確定させ、403 を返せるようにする。
        chunks = tts.synthesize_stream(text, req.lang, ref_audio=ref_bytes,
                                       ref_speaker_id=req.ref_speaker_id)
    except CloningConsentError as e:
        return _consent_denied_response(e)
    except FileNotFoundError as e:
        logger.warning(f"TTS unavailable: {e}")
        return JSONResponse({'error': 'voice model not trained yet'}, status_code=503)

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        # 同期の重い生成をスレッドで回し、チャンクを base64 行としてキューへ送る
        try:
            for wav_chunk in chunks:
                b64 = base64.b64encode(wav_chunk).decode('ascii')
                asyncio.run_coroutine_threadsafe(queue.put(b64 + "\n"), loop)
        except Exception as e:
            logger.exception("TTS streaming failed")
            asyncio.run_coroutine_threadsafe(queue.put(f"__ERROR__{e}\n"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    async def gen():
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(gen(), media_type="text/plain",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.websocket("/ws/voice-stream")
async def websocket_voice_stream(websocket: WebSocket):
    """
    フロントエンドとリアルタイム双方向通信を行うためのWebSocketエンドポイント
    swi_voice/serve/api.py から移植され、統合サーバー (port 5000) で稼働する
    """
    await websocket.accept()
    logger.info("WebSocket connection established.")

    try:
        from swi_voice.serve.tts import get_tts
        tts = get_tts()
        actual_model = tts.model
        actual_codec = tts.codec
    except FileNotFoundError:
        logger.warning("WebSocket TTS unavailable: voice model not trained yet")
        await websocket.close(code=1011)
        return
    except Exception:
        logger.exception("Error loading TTS model for WebSocket")
        await websocket.close(code=1011)
        return

    # V-4: 停止シグナルは接続（セッション）ごとに保持し、他クライアントへ干渉させない。
    from swi_voice.duplex.barge_in import SessionStopSignal
    stop_signal = SessionStopSignal()

    async def receive_messages():
        try:
            while True:
                data_str = await websocket.receive_text()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                action = data.get("action", "generate")
                if action == "stop":
                    logger.info("Received stop request from client.")
                    stop_signal.set()
                elif action == "generate":
                    text = data.get("text", "")
                    if text:
                        logger.info(f"Received generate request (text_len={len(text)})")   # SEC-7: 内容は記録しない
                        stop_signal.clear()
                        asyncio.create_task(send_audio(text))
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected by client.")
        except Exception:
            logger.exception("Error in receive loop.")

    async def send_audio(text: str):
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                for wav_chunk in tts.synthesize_stream(text, "ja"):
                    if stop_signal.is_set():                 # V-4: このセッションの停止のみを見る
                        logger.info("Barge-in detected, stopping TTS generation.")
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(wav_chunk), loop)
            except Exception as e:
                logger.exception("Error in TTS streaming worker.")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        import threading
        threading.Thread(target=worker, daemon=True).start()

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                await websocket.send_bytes(chunk)
            await websocket.send_text(json.dumps({"status": "done"}))
        except Exception:
            logger.exception("Error in send loop.")

    await receive_messages()
