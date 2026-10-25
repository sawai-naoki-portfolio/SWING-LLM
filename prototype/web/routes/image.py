"""
web/routes/image.py

3AI 統合 WebUI の画像生成エンドポイント。
学習済み swi_image モデル（VAE + DiT）でプロンプトから画像（PNG）を生成して返す。
画像モデル未学習時は 503 を返し、フロントはメッセージ表示にフォールバックする。
"""
import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import MAX_PROMPT_LENGTH

logger = setup_logger("web_image")
router = APIRouter()


class ImageGenRequest(BaseModel):
    prompt: str = ''
    steps: int = 0          # 0 = 既定ステップ


@router.get("/api/image/status")
async def api_image_status():
    """画像生成モデルが利用可能か（フロントの機能出し分け用）。"""
    try:
        from swi_image.serve.generate import image_available
        return {"available": bool(image_available())}
    except Exception:
        logger.exception("image status check failed")
        return {"available": False}


@router.post("/api/generate_image")
async def api_generate_image(req: ImageGenRequest):
    """プロンプト → swi_image で画像生成 → PNG を返す。"""
    prompt = (req.prompt or '').strip()
    if not prompt:
        return JSONResponse({'error': 'prompt is required'}, status_code=400)
    if len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[:MAX_PROMPT_LENGTH]

    try:
        from swi_image.serve.generate import get_generator
        gen = get_generator()
        steps = req.steps if req.steps and req.steps > 0 else None
        loop = asyncio.get_event_loop()
        png = await loop.run_in_executor(None, gen.generate, prompt, steps)
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})
    except FileNotFoundError as e:
        logger.warning(f"Image generation unavailable: {e}")
        return JSONResponse({'error': 'image model not trained yet'}, status_code=503)
    except Exception as e:
        logger.exception("Image generation failed")
        return JSONResponse({'error': str(e)}, status_code=500)
