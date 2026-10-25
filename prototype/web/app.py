import os
import sys
import json
import hmac
import hashlib
from urllib.parse import urlparse, parse_qs

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
# stderr も UTF-8 にする。ここを漏らすと日本語を含む例外メッセージ/トレースバックが
# cp932 コンソールで文字化けし、「起動を拒否した理由」が読めなくなる（対処法が伝わらない）。
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
prototype_dir = os.path.dirname(current_dir)
if prototype_dir not in sys.path:
    sys.path.insert(0, prototype_dir)

import torch
torch.backends.cudnn.enabled = False

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    WEB_HOST, WEB_PORT, WEB_BASE_ALLOWED_HOSTS, WEB_API_TOKEN,
    RATE_LIMIT_ENABLED, RATE_LIMIT_RULES, MAX_REQUEST_BODY_BYTES,
)
from web.models import load_model
from web.rate_limit import RateLimitMiddleware, TokenBucketLimiter, client_ip
from web.routes.inference import router as inference_router
from web.routes.voice import router as voice_router
from web.routes.image import router as image_router

logger = setup_logger("app")


# =====================================================================
# SEC-3: リクエスト境界の防御（Host/Origin 検証 + 任意トークン認証）
# =====================================================================
def _compute_allowed_hosts():
    hosts = set(WEB_BASE_ALLOWED_HOSTS)
    if WEB_HOST and WEB_HOST not in ("0.0.0.0", "::"):
        hosts.add(WEB_HOST.lower())
    extra = os.environ.get("SWI_ALLOWED_HOSTS", "").strip()
    if extra:
        hosts.update(h.strip().lower() for h in extra.split(",") if h.strip())
    return hosts


_ALLOWED_HOSTS = _compute_allowed_hosts()
_WILDCARD_HOSTS = "*" in _ALLOWED_HOSTS


def _is_public_bind():
    return WEB_HOST in ("0.0.0.0", "::")


def validate_bind_config():
    """WebUI ロードマップ step2 / `12_web_production_deployment.md` §3: 公開バインドを fail-closed にする。

    旧実装はこの関数の代わりに `_ENFORCE_HOST` を持ち、「0.0.0.0/:: バインド ∧ SWI_ALLOWED_HOSTS 未設定」
    のとき Host 検証を**無効化**していた（fail-open）。締め出し防止が動機だったが、
    0.0.0.0 で待ち受けるとは「任意のホスト名で到達しうる」ということであり、そこで Host 検証を切ると
    DNS リバインディング（悪意サイトが被害者のブラウザを踏み台に LAN 内 API を叩く）を素通しする。
    「うっかり全開」は起動時に止める。公開を意図しているなら、到達に使うホスト名を運用者に明示させる。
    """
    if not _is_public_bind() or os.environ.get("SWI_ALLOWED_HOSTS", "").strip():
        return
    # 対処法は logger 経由で出す（error.log に残り、run.ps1 の Show-ErrorTail でも拾われる）。
    for line in (
        f"起動を拒否しました (fail-closed): 公開バインド({WEB_HOST}) なのに SWI_ALLOWED_HOSTS が未設定です。",
        "  0.0.0.0/:: で待ち受ける構成では Host 検証の許可リストが必須です",
        "  （未設定のまま Host 検証を外すと DNS リバインディングを防げません）。",
        "  対処のいずれか:",
        "    1) 到達に使うホスト名/IP を明示する:  $env:SWI_ALLOWED_HOSTS = 'swi.example.com,192.168.1.10'",
        "    2) ローカル専用に戻す:                Remove-Item Env:SWI_WEB_HOST   # 既定 127.0.0.1",
        "    3) 検証を意図的に無効化する(非推奨):  $env:SWI_ALLOWED_HOSTS = '*'",
        "  公開運用では SWI_API_TOKEN の設定も併せて必須です。",
    ):
        logger.error(line)
    raise RuntimeError(
        f"Refusing to start: public bind ({WEB_HOST}) without SWI_ALLOWED_HOSTS (fail-closed). "
        "See the error log above for remediation."
    )


def _host_of(header_value):
    """Host/Origin ヘッダからホスト名（ポート/ブラケット除去・小文字化）を取り出す。"""
    if not header_value:
        return ""
    h = header_value.rsplit(":", 1)[0] if header_value.count(":") == 1 else header_value
    # IPv6 "[::1]:port" 等の簡易対応
    if header_value.startswith("["):
        h = header_value[1:header_value.find("]")] if "]" in header_value else header_value
    return h.strip("[]").lower()


# X-2: トークン認証を要求するパス接頭辞。従来は "/api/" のみで、WebSocket
# ("/ws/voice-stream") は認証を素通りしていた（GPU を消費する TTS を無認証で起動できた）。
_PROTECTED_PREFIXES = ("/api/", "/ws/")


def _is_protected(path):
    return any(path.startswith(p) for p in _PROTECTED_PREFIXES)


def _token_matches(candidate):
    """X-1: 定数時間比較。`!=` は先頭不一致で早期リターンするため、応答時間の差から
    トークンを 1 バイトずつ復元されうる（タイミング攻撃）。"""
    if not WEB_API_TOKEN:
        return True
    return hmac.compare_digest(candidate or "", WEB_API_TOKEN)


def _extract_token(headers, scope):
    """Authorization: Bearer / X-API-Token / (WS のみ) クエリ `?token=` からトークンを取り出す。

    ブラウザの WebSocket API はハンドシェイクに任意ヘッダを付けられないため、WS に限り
    クエリパラメータを許可する。ヘッダを送れるクライアント（Flutter の IOWebSocketChannel、
    Python の websockets 等）はヘッダを使うこと（クエリは URL としてログに残りうるため）。
    """
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    header_token = headers.get("x-api-token", "")
    if header_token:
        return header_token
    if scope.get("type") == "websocket":
        qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        values = qs.get("token") or []
        if values:
            return values[0]
    return ""


class SecurityGuardMiddleware:
    """純 ASGI ミドルウェア（ストリーミング応答に干渉しない）。
    - Host 検証: DNS リバインディング遮断（許可ホスト以外を 400）。
    - Origin 検証: 状態変更 HTTP と WebSocket のクロスオリジン副作用を遮断（403/close）。
    - 任意トークン認証: SWI_API_TOKEN 設定時、/api/* と /ws/* に Bearer/X-API-Token を要求
      （HTTP は 401、WebSocket は accept 前に close 1008）。WS はブラウザがヘッダを付けられない
      ため `?token=` クエリも受け付ける（X-2）。比較は定数時間（X-1）。
    既定（トークン未設定・localhost バインド）はローカル開発の利便のため無認証だが、
    Host/Origin 検証は常に有効で、悪意サイトからの localhost API 悪用を抑止する。
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        stype = scope.get("type")
        if stype not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # 1) Host 検証（DNS リバインディング遮断）。常時有効。
        # 「公開バインド ∧ allowlist 未設定」で検証を切る旧経路は撤去した（validate_bind_config が
        # そもそも起動を拒否する）。無効化は SWI_ALLOWED_HOSTS="*" の明示指定でのみ可能。
        if not _WILDCARD_HOSTS:
            host = _host_of(headers.get("host", ""))
            if host and host not in _ALLOWED_HOSTS:
                await self._reject(scope, send, 400, "invalid host"); return

        # 2) Origin 検証（クロスオリジン副作用の遮断）: 状態変更 HTTP と WS を対象
        needs_origin_check = (stype == "websocket") or (stype == "http" and method not in ("GET", "HEAD", "OPTIONS"))
        if needs_origin_check and not _WILDCARD_HOSTS:
            origin = headers.get("origin")
            if origin:
                if _host_of(urlparse(origin).netloc or origin) not in _ALLOWED_HOSTS:
                    await self._reject(scope, send, 403, "cross-origin request blocked"); return

        # 3) 任意トークン認証（設定時のみ・/api/* と /ws/*・プリフライト OPTIONS は除外）
        # X-2: WebSocket も対象に含める。従来は stype=="http" 限定で、/ws/voice-stream が
        # 完全に無認証だった（トークンを設定しても TTS を誰でも起動できた）。
        # X-1: 比較は hmac.compare_digest による定数時間比較。
        if WEB_API_TOKEN and method != "OPTIONS" and _is_protected(path):
            if not _token_matches(_extract_token(headers, scope)):
                await self._reject(scope, send, 401, "unauthorized"); return

        await self.app(scope, receive, send)

    async def _reject(self, scope, send, status, msg):
        if scope.get("type") == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": msg}).encode("utf-8")
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時にモデルを初期化する。FastAPI 0.109+ で @app.on_event("startup") は非推奨に
    # なったため lifespan ハンドラへ移行（DeprecationWarning の解消）。挙動は従来と同一で、
    # 初期化失敗はプロセスを落とさずログに残す（ここで raise すると起動不可になるため）。
    try:
        load_model()
    except Exception:
        logger.exception("Failed to initialize model on startup.")

    # RAG シングルトンの起動時ウォーム。初回の search_documents で
    # SentenceTransformer + リランカのロード（数秒〜十数秒）が走ると
    # TOOL_TIMEOUT_SEC(10s) を超えてツールがタイムアウトする（2026-07-12 実測）。
    # リクエストをブロックしないよう別スレッドで先行ロードする。
    import threading

    def _warm_rag():
        try:
            import os as _os
            from swi_llm.agent.self_rag import get_self_rag
            base_dir = _os.path.dirname(_os.path.abspath(__import__("swi_llm").__file__))
            get_self_rag(_os.path.join(base_dir, "knowledge_base"))
            logger.info("RAG singleton warmed (search_documents は初回からタイムアウトしません).")
        except Exception:
            logger.exception("RAG warmup failed (search_documents の初回呼び出しが遅くなる可能性).")
    threading.Thread(target=_warm_rag, name="rag-warmup", daemon=True).start()

    yield
    # 終了時のクリーンアップは現状不要。


def rate_limit_key(scope, headers):
    """レート制限のキー。トークンがあればそれを優先し（同一トークンの多端末を 1 枠に束ねる）、
    無ければクライアント IP を使う。

    トークンは平文のまま保持せずハッシュ化する（バケット辞書やログに秘密が残らないように）。
    """
    token = _extract_token(headers, scope)
    if token:
        return "tok:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return "ip:" + client_ip(scope, headers)


def log_startup_checklist():
    """起動時の実効セキュリティ構成を運用チェックリストとして 1 か所に出す（WebUI ロードマップ step2）。

    「設定したつもり」を防ぐのが目的なので、意図ではなく**実効値**だけを出す。
    """
    public = _is_public_bind()
    lines = [
        "===== SWI Web セキュリティ構成（実効値） =====",
        f"  bind        : {WEB_HOST}:{WEB_PORT}" + ("  [PUBLIC: LAN/公衆網から到達可能]" if public else "  [loopback のみ]"),
        f"  host 検証   : " + ("無効 (SWI_ALLOWED_HOSTS='*')" if _WILDCARD_HOSTS
                              else f"有効 allowlist={sorted(_ALLOWED_HOSTS)}"),
        "  origin 検証 : " + ("無効 (SWI_ALLOWED_HOSTS='*')" if _WILDCARD_HOSTS
                              else "有効 (状態変更 HTTP + WebSocket)"),
        f"  token 認証  : {'有効 (SWI_API_TOKEN)' if WEB_API_TOKEN else '無効'}",
        "  レート制限  : " + (f"有効 generate={RATE_LIMIT_RULES['generate'][0]}/min "
                              f"image={RATE_LIMIT_RULES['image'][0]}/min "
                              f"tts={RATE_LIMIT_RULES['tts'][0]}/min "
                              f"(body上限 {MAX_REQUEST_BODY_BYTES // (1024 * 1024)}MB)"
                              if RATE_LIMIT_ENABLED else "無効 (SWI_RATE_LIMIT=0)"),
        "  TLS         : なし (uvicorn 直起動)",
        "=============================================",
    ]
    for line in lines:
        logger.info(line)

    if public and not WEB_API_TOKEN:
        logger.warning("公開バインドですが SWI_API_TOKEN が未設定です。/api/* と /ws/* が無認証で到達可能です。")
    if public and _WILDCARD_HOSTS:
        logger.warning("公開バインド + SWI_ALLOWED_HOSTS='*' です。Host/Origin 検証がどちらも無効で、"
                       "DNS リバインディングとクロスオリジン副作用に対して無防備です。")
    if public:
        logger.warning("公開運用では TLS 終端（リバースプロキシ）が前提です。"
                       "uvicorn を直接インターネットへ露出しないでください。")
    if not RATE_LIMIT_ENABLED:
        logger.warning("レート制限が無効です (SWI_RATE_LIMIT=0)。生成系エンドポイントが無制限に叩けます。")


# fail-closed: 「公開バインド ∧ allowlist 未設定」ならここで起動を拒否する（app を作らない）。
validate_bind_config()

app = FastAPI(title="SWI-LLM Web UI", lifespan=lifespan)
# CORS: 本 API は Cookie/認証情報を用いないため allow_credentials=False とする。
# allow_origins=["*"] と allow_credentials=True の併用は「任意サイトへ資格情報付きで
# 応答を晒す」危険な組み合わせ（ブラウザ仕様上ワイルドレスポンスが資格情報付きで返る）
# になるため禁止。資格情報を使わない開かれた API として * を許可しつつ credentials は無効化する。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# SEC-12: レート制限 + ボディサイズ上限。SecurityGuard の**内側**に置く（= 後に追加する
# SecurityGuard が最外層になり、先に評価される）。不正 Host / 不正トークンの洪水は
# SecurityGuard が先に弾くので、バケット鍵が汚染されない。
app.add_middleware(
    RateLimitMiddleware,
    limiter=TokenBucketLimiter(),
    key_func=rate_limit_key,
    enabled=RATE_LIMIT_ENABLED,
)
# SEC-3: Host/Origin/トークン検証。Starlette は「後に追加したものが最外層＝先に評価」なので
# 最後に追加して最外層に置き、不正 Host/クロスオリジンを最初に弾く。
app.add_middleware(SecurityGuardMiddleware)
log_startup_checklist()

app.include_router(inference_router)
app.include_router(voice_router)   # 3AI統合: swi_voice TTS (/api/tts, /api/voice/status)
app.include_router(image_router)   # 3AI統合: swi_image 生成 (/api/generate_image, /api/image/status)

# Mount static files at root so index.html, style.css, script.js are served
static_folder = os.path.join(current_dir, 'static')
if os.path.exists(static_folder):
    app.mount("/", StaticFiles(directory=static_folder, html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    logger.info("Starting FastAPI Web UI...")
    uvicorn.run("web.app:app", host=WEB_HOST, port=WEB_PORT, reload=False, workers=1)
