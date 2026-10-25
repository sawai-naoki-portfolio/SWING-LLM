import os
import re
import time
import asyncio
import threading
from collections import deque
from typing import List
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from concurrent.futures import ThreadPoolExecutor

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.config import INFERENCE_CONFIG
from swi_llm.utils.chat_template import build_conversation_prompt, format_tool_call
from swi_llm.utils.constants import (
    TOOL_GET_WEATHER, TOOL_CALCULATE, TOOL_EXECUTE_PYTHON, TOOL_SEARCH_DOCUMENTS,
    TOOL_WIKIPEDIA_SEARCH,
    MAX_CONCURRENT_INFERENCE, INFERENCE_THREAD_POOL_WORKERS, MAX_PROMPT_LENGTH,
    DEFAULT_IMAGE_PROMPT, DEFAULT_PERSONA, PERSONA_SYSTEM_PROMPTS, MAX_HISTORY_TURNS,
    MAX_HISTORY_CHARS, MAX_FEEDBACK_FIELD_CHARS, FEEDBACK_RATE_LIMIT_PER_MIN,
    code_exec_tool_enabled,
)
from swi_llm.core.generation import generate, generate_with_thought, generate_stream, generate_o1_stream, generate_best_of_n
from swi_llm.agent.base import AgentBase
from swi_llm.agent.tools import (
    get_weather, calculate, execute_python, search_documents, wikipedia_search, TOOL_SCHEMAS)
from web.intent import detect_and_execute_tool_detailed, format_tool_result_for_display
from web.models import load_model, parse_images, tokenizer

logger = setup_logger("web_inference")
router = APIRouter()

# OOM 防止のため同時推論ストリーム数を制限
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
executor = ThreadPoolExecutor(max_workers=INFERENCE_THREAD_POOL_WORKERS)

class HistoryTurn(BaseModel):
    role: str = ''
    content: str = ''

class GenerateRequest(BaseModel):
    prompt: str = ''
    use_cot: bool = False
    images: List[str] = []
    persona: str = DEFAULT_PERSONA
    client: str = 'web'
    best_of_n: int = 0  # >1 で PRM Best-of-N (§8.2)。0 は INFERENCE_CONFIG の既定に従う
    history: List[HistoryTurn] = []  # マルチターン会話履歴 (§12.3)

class FeedbackRequest(BaseModel):
    query: str = ''
    response: str = ''
    rating: int = 0  # +1 = good, -1 = bad


def _history_dicts(req: GenerateRequest):
    """req.history (List[HistoryTurn]) を build_conversation_prompt 用の dict 列に変換 (§12.3)。

    MAX_PROMPT_LENGTH を history 経由で回避した巨大入力(OOM/DoS)を防ぐため、直近ターンから
    合計 MAX_HISTORY_CHARS 文字までに制限する（古いターンを落とす）。
    """
    turns = list(req.history or [])
    kept, total = [], 0
    for t in reversed(turns):                    # 新しいターンを優先して残す
        c = t.content or ""
        total += len(c)
        if total > MAX_HISTORY_CHARS:
            break
        kept.append({"role": t.role, "content": c})
    kept.reverse()
    return kept


# SEC-4: フィードバック収集のレート制限（プロセス共有・直近60秒の件数）。
_feedback_times = deque()
_feedback_lock = threading.Lock()


def _feedback_rate_ok():
    """直近 60 秒のフィードバック件数が上限内なら True。無制限追記による汚染/DoS を抑止。"""
    now = time.time()
    with _feedback_lock:
        while _feedback_times and now - _feedback_times[0] > 60.0:
            _feedback_times.popleft()
        if len(_feedback_times) >= FEEDBACK_RATE_LIMIT_PER_MIN:
            return False
        _feedback_times.append(now)
        return True


@router.get("/api/llm/status")
async def api_llm_status():
    """稼働中の推論バックエンドと、そこで**実際に使える機能**を返す。

    追加のみの GET（既存エンドポイントの契約は不変）。/api/voice/status,
    /api/image/status と同じ「フロントの機能出し分け」用途。

    完全自作方針（2026-10-25）により外部 LLM ゲートウェイ（旧 web/llm_backend.py）は撤去済み。
    推論は常に自作モデルだが、クライアントが読むレスポンス形状は互換のため維持する。
    """
    return {
        "backend": "local",
        "model": "SWI-LLM (MiniLLM / from-scratch)",
        "capabilities": {"vision": True, "cot": True, "best_of_n": True, "tools": True},
    }


@router.post("/api/feedback")
async def api_feedback(req: FeedbackRequest):
    """本番運用フィードバックを収集する（継続学習フライホイール, §11.2）。

    SEC-4: 無認証エンドポイントゆえ、継続学習へのデータ汚染・ディスク枯渇を防ぐため
    入力検証（rating は ±1、query/response は長さ上限）とレート制限・ファイルサイズ上限を課す。
    """
    # rating は good(+1)/bad(-1) のみ許可（任意値の注入を拒否）
    if req.rating not in (-1, 1):
        return JSONResponse({'error': 'rating must be +1 (good) or -1 (bad)'}, status_code=400)
    # query/response の長さ上限（巨大入力によるファイル肥大・ログ汚染を防止）
    if len(req.query or "") > MAX_FEEDBACK_FIELD_CHARS or len(req.response or "") > MAX_FEEDBACK_FIELD_CHARS:
        return JSONResponse({'error': f'query/response too long (max {MAX_FEEDBACK_FIELD_CHARS} chars)'},
                            status_code=400)
    # レート制限（無認証のため、短時間の大量注入を抑止）
    if not _feedback_rate_ok():
        return JSONResponse({'error': 'feedback rate limit exceeded, try again later'}, status_code=429)
    try:
        from swi_llm.training.continual import FeedbackStore
        FeedbackStore().add(req.query, req.response, req.rating)
        return {"status": "ok"}
    except ValueError as e:
        # FeedbackStore のサイズ上限超過など（想定内の拒否）
        return JSONResponse({'error': str(e)}, status_code=507)
    except Exception as e:
        logger.exception("Failed to store feedback")
        return JSONResponse({'error': str(e)}, status_code=500)


@router.post("/api/generate")
async def api_generate(req: GenerateRequest):
    prompt_text = req.prompt
    use_cot = req.use_cot
    images_base64 = req.images

    # SEC-7: ユーザー入力(PII等)を平文でログに残さない。長さ・件数のみ記録する。
    logger.info(f"Incoming Request (Sync) - prompt_len={len(prompt_text or '')}, images={len(images_base64)}")

    if not prompt_text and not images_base64:
        return JSONResponse({'error': 'Prompt or image is required'}, status_code=400)

    if len(prompt_text) > MAX_PROMPT_LENGTH:
        return JSONResponse({'error': f"Prompt length exceeds maximum allowed ({MAX_PROMPT_LENGTH})"}, status_code=400)

    model = load_model()

    if not prompt_text and images_base64:
        prompt_text = DEFAULT_IMAGE_PROMPT

    try:
        image_tensor = parse_images(images_base64)
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=400)

    try:
        async with inference_semaphore:
            if use_cot:
                sft_prompt_cot = build_conversation_prompt(
                    _history_dicts(req), prompt_text, cot=True, max_turns=MAX_HISTORY_TURNS)
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    executor,
                    generate_with_thought,
                    model, tokenizer, sft_prompt_cot,
                    INFERENCE_CONFIG["max_thought_tokens"],
                    INFERENCE_CONFIG["max_answer_tokens"]
                )
            else:
                sft_prompt = build_conversation_prompt(
                    _history_dicts(req), prompt_text, max_turns=MAX_HISTORY_TURNS)
                loop = asyncio.get_event_loop()
                n = req.best_of_n if req.best_of_n and req.best_of_n > 0 else INFERENCE_CONFIG.get("best_of_n", 1)
                if n > 1:
                    # PRM Best-of-N: N 候補を生成し PRM スコア最高を採用 (§8.2)
                    def _best_of_n():
                        best_text, _ = generate_best_of_n(
                            model, tokenizer, sft_prompt, n=n,
                            max_new_tokens=INFERENCE_CONFIG["max_new_tokens"])
                        return best_text
                    output = await loop.run_in_executor(executor, _best_of_n)
                else:
                    output = await loop.run_in_executor(
                        executor,
                        generate,
                        model.text_model, tokenizer, sft_prompt, INFERENCE_CONFIG["max_new_tokens"]
                    )
        logger.info(f"Sync generation completed. response_len={len(output or '')}")   # SEC-7: 内容は記録しない
        return {"response": output}
    except Exception as e:
        logger.exception("Error during generation")
        return JSONResponse({'error': str(e)}, status_code=500)

@router.post("/api/generate_stream")
async def api_generate_stream(req: GenerateRequest):
    prompt_text = req.prompt
    use_cot = req.use_cot
    images_base64 = req.images
    persona = req.persona
    client_type = req.client

    # SEC-7: ユーザー入力を平文でログに残さない。長さ・件数・ペルソナのみ記録する。
    logger.info(f"Incoming Request - prompt_len={len(prompt_text or '')}, images={len(images_base64)}, persona={persona}")

    if not prompt_text and not images_base64:
        return JSONResponse({'error': 'Prompt or image is required'}, status_code=400)

    if len(prompt_text) > MAX_PROMPT_LENGTH:
        return JSONResponse({'error': f"Prompt length exceeds maximum allowed ({MAX_PROMPT_LENGTH})"}, status_code=400)

    agent = AgentBase()
    agent.register_tool(TOOL_GET_WEATHER, get_weather, TOOL_SCHEMAS[TOOL_GET_WEATHER])
    agent.register_tool(TOOL_CALCULATE, calculate, TOOL_SCHEMAS[TOOL_CALCULATE])
    # SEC-6: 既定では execute_python を登録しない（信頼できない入力からの任意コード実行を遮断）。
    # SWI_ENABLE_CODE_EXEC_TOOL=1 のときのみ登録する。
    if code_exec_tool_enabled():
        agent.register_tool(TOOL_EXECUTE_PYTHON, execute_python, TOOL_SCHEMAS[TOOL_EXECUTE_PYTHON])
    agent.register_tool(TOOL_SEARCH_DOCUMENTS, search_documents, TOOL_SCHEMAS[TOOL_SEARCH_DOCUMENTS])
    # NEW-L6: Web 検索（Wikipedia API・キー不要・ホスト固定）。ローカル KB 外の一般知識を補う。
    agent.register_tool(TOOL_WIKIPEDIA_SEARCH, wikipedia_search, TOOL_SCHEMAS[TOOL_WIKIPEDIA_SEARCH])

    system_prompt = PERSONA_SYSTEM_PROMPTS.get(persona, PERSONA_SYSTEM_PROMPTS[DEFAULT_PERSONA])

    model = load_model()

    if not prompt_text and images_base64:
        prompt_text = DEFAULT_IMAGE_PROMPT

    async def stream_generator():
        loop = asyncio.get_event_loop()
        # クライアント切断/正常終了でワーカースレッドへ停止を伝える協調キャンセル用フラグ。
        stop_event = threading.Event()
        try:
            try:
                image_tensor = parse_images(images_base64)
            except ValueError as e:
                yield f"Error: {str(e)}"
                return

            accumulated_response = ""
            sft_prompt = build_conversation_prompt(
                _history_dicts(req), prompt_text, system_text=system_prompt,
                max_turns=MAX_HISTORY_TURNS)

            # ルールベース意図検知によるツール自動実行 (web/intent.py)。
            # ツールは HTTP/サブプロセス等でブロックしうるため、イベントループを
            # 塞がないようスレッドプールで実行する。
            intent = await loop.run_in_executor(
                executor, detect_and_execute_tool_detailed, prompt_text, agent)

            if intent is not None:
                # 結果はプロンプトへ注入するだけでなく利用者にも直接見せる。モデルが弱いうちは
                # 回答に反映されず、正しく実行できたツールの答えが埋もれてしまうため。
                display = format_tool_result_for_display(intent.tool, intent.result)
                if client_type == 'flutter':
                    yield "\n\n[🔧✨ 意図を検知しツールを自動実行しました]\n\n"
                    if display:
                        yield f"🔧 {display}\n\n"
                else:
                    # tool-check は span（script.js の CHIP_OK と同形）。div を入れ子にすると、CDN 不通時の
                    # 簡易描画（renderProse の protect 正規表現）が内側の </div> で枠を閉じ損ね、後続を飲み込む。
                    yield "\n\n<div class=\"tool-box success\"><span class=\"tool-check\">✔</span><span>✨ 意図を検知しツールを自動実行しました</span></div>\n\n"
                    if display:
                        # WebUI は <tool_result> をエスケープして結果ボックスに描画する（script.js toolResult）。
                        yield f"<tool_result>{display}</tool_result>\n\n"
                sft_prompt += format_tool_call("{\"name\": \"system_auto\", \"arguments\": {}}") + f"\n{intent.tool_response}"

            queue = asyncio.Queue()

            def run_generation():
                try:
                    if use_cot:
                        gen = generate_o1_stream(
                            model, tokenizer, sft_prompt,
                            max_thought_tokens=INFERENCE_CONFIG["max_thought_tokens"],
                            max_answer_tokens=INFERENCE_CONFIG["max_answer_tokens"],
                            images=image_tensor,
                            difficulty="auto",
                            stop_event=stop_event,
                        )
                    else:
                        gen = generate_stream(
                            model, tokenizer, sft_prompt,
                            max_new_tokens=INFERENCE_CONFIG["stream_max_new_tokens"],
                            images=image_tensor,
                            stop_event=stop_event,
                        )
                    for chunk in gen:
                        if stop_event.is_set():
                            break   # 消費側が去った → これ以上 enqueue しない
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(e), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            async with inference_semaphore:
                executor.submit(run_generation)
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        raise chunk
                    if isinstance(chunk, str):
                        accumulated_response += chunk
                    yield chunk

            logger.info(f"Stream completed. response_len={len(accumulated_response)}")   # SEC-7: 内容は記録しない

        except Exception as e:
            logger.exception("Error during streaming")
            yield f"\n[ERROR] {str(e)}"
        finally:
            # 正常終了・例外・クライアント切断(GeneratorExit) のいずれでもワーカーへ停止を通知。
            stop_event.set()

    return StreamingResponse(stream_generator(), media_type="text/plain")
