"""SEC-12 / WebUI ロードマップ step5: レート制限とリクエストサイズ上限（純 ASGI ミドルウェア）。

**なぜ必要か**: レート制限は `/api/feedback` にしか掛かっておらず、GPU を消費する生成系
（`/api/generate`, `/api/generate_stream`, `/api/generate_image`, `/api/tts`, `/api/tts_stream`,
`/ws/voice-stream`）は無制限だった。`MAX_CONCURRENT_INFERENCE` のセマフォは**同時実行数**の制御で
あって**流量**の制御ではないため、1 クライアントが待ち行列を無限に伸ばして GPU を占有できた。

**方式**: クライアント（トークン or IP）ごとのトークンバケット。
  - 容量 = burst（瞬間的に許す連射数）、補充速度 = per_min / 60 [tokens/sec]
  - 枯渇時は HTTP 429 + `Retry-After`、WebSocket は accept 前に close(1013 = Try Again Later)

**制約（意図的）**:
  - プロセス内メモリで完結する。ワーカーを多重化すると実効上限はワーカー数倍になる
    （厳密な全体制限が要るなら共有ストア = Redis 等が必要。WebUI step6 のプロセス管理と併せて判断）。
  - ボディ上限は `Content-Length` で判定する。Content-Length 無し（chunked）の場合は
    受信バイトを数えて上限超過で接続を切る。**恒久的な上限はリバースプロキシ（step3）側の
    `client_max_body_size` 等で掛けるのが本来の層**であり、ここはその手前の保険。
"""
import json
import time
import threading
from collections import OrderedDict

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    RATE_LIMIT_RULES, RATE_LIMIT_MAX_KEYS, MAX_REQUEST_BODY_BYTES, TRUST_PROXY_HEADERS,
)

logger = setup_logger("web.rate_limit")

# パス → コストクラス。完全一致で引く（"/api/generate" は "/api/generate_stream" の接頭辞なので
# 接頭辞マッチだと取り違える）。
PATH_CLASS = {
    "/api/generate": "generate",
    "/api/generate_stream": "generate",
    "/api/generate_image": "image",
    "/api/tts": "tts",
    "/api/tts_stream": "tts",
    "/ws/voice-stream": "tts",
    # fix 08(3): 実 ASR（faster-whisper）。CPU/GPU を消費する生成系に準じて tts クラスで絞る。
    "/api/asr": "tts",
}


def classify_path(path):
    """パスのコストクラスを返す。レート制限の対象外（静的ファイル等）は None。"""
    cost_class = PATH_CLASS.get(path)
    if cost_class:
        return cost_class
    if path.startswith("/api/") or path.startswith("/ws/"):
        return "default"
    return None


def client_ip(scope, headers=None, trust_proxy=None):
    """レート制限のキーに使うクライアント IP。

    `X-Forwarded-For` は既定で**信頼しない**。プロキシが前段に居ない状態で信頼すると、
    攻撃者がヘッダを詐称するだけでキーを無限に変えられ、レート制限を自明に回避できる。
    リバースプロキシ導入後（step3）に SWI_TRUST_PROXY_HEADERS=1 で有効化する。
    """
    if trust_proxy is None:
        trust_proxy = TRUST_PROXY_HEADERS
    if trust_proxy and headers:
        forwarded = headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    client = scope.get("client")
    if client and client[0]:
        return client[0]
    return "unknown"


class TokenBucketLimiter:
    """クライアント鍵 × コストクラスごとのトークンバケット。

    `RATE_LIMIT_MAX_KEYS` を超えたら最も古い鍵から捨てる（多数の IP による
    メモリ枯渇 DoS を防ぐ）。捨てられた鍵は次回アクセス時に満タンで作り直されるだけで、
    正規利用者への影響は「制限がゆるむ方向」なので安全側。
    """

    def __init__(self, rules=None, max_keys=RATE_LIMIT_MAX_KEYS, clock=time.monotonic):
        self.rules = dict(rules or RATE_LIMIT_RULES)
        self.max_keys = max_keys
        self._clock = clock
        self._buckets = OrderedDict()          # (key, cost_class) -> [tokens, last_ts]
        self._lock = threading.Lock()

    def _rule(self, cost_class):
        return self.rules.get(cost_class) or self.rules["default"]

    def check(self, key, cost_class):
        """1 トークン消費を試みる。戻り値 (allowed, retry_after_seconds)。"""
        per_min, burst = self._rule(cost_class)
        refill_per_sec = per_min / 60.0
        now = self._clock()
        bucket_id = (key, cost_class)

        with self._lock:
            bucket = self._buckets.get(bucket_id)
            if bucket is None:
                bucket = [float(burst), now]
                self._buckets[bucket_id] = bucket
                while len(self._buckets) > self.max_keys:
                    self._buckets.popitem(last=False)      # LRU: 最も古いものから捨てる
            else:
                self._buckets.move_to_end(bucket_id)
                elapsed = max(0.0, now - bucket[1])
                bucket[0] = min(float(burst), bucket[0] + elapsed * refill_per_sec)
                bucket[1] = now

            if bucket[0] >= 1.0:
                bucket[0] -= 1.0
                return True, 0.0

            # 1 トークン貯まるまでの待ち時間（切り上げて最低 1 秒）。
            deficit = 1.0 - bucket[0]
            retry_after = deficit / refill_per_sec if refill_per_sec > 0 else 60.0
            return False, max(1.0, retry_after)


class RateLimitMiddleware:
    """純 ASGI ミドルウェア（ストリーミング応答に干渉しない）。

    SecurityGuardMiddleware の**内側**に置く（= 先に Host/Origin/トークン検証を通す）。
    そうすることで、不正 Host や不正トークンの洪水はバケットを一切消費せずに弾かれ、
    バケット鍵が汚染されない。
    """

    # 413 を返す前に読み捨てるボディの上限（max_body_bytes の何倍まで付き合うか）。
    DRAIN_LIMIT_FACTOR = 2

    def __init__(self, app, limiter=None, key_func=None, enabled=True,
                 max_body_bytes=MAX_REQUEST_BODY_BYTES):
        self.app = app
        self.limiter = limiter or TokenBucketLimiter()
        self.key_func = key_func or (lambda scope, headers: "ip:" + client_ip(scope, headers))
        self.enabled = enabled
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        stype = scope.get("type")
        if stype not in ("http", "websocket") or not self.enabled:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        cost_class = classify_path(path)
        if cost_class is None:                      # 静的ファイル等は対象外
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

        # 1) ボディサイズ上限（Content-Length で判定。アプリにはボディを渡さない = パースさせない）
        if stype == "http":
            declared = headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > self.max_body_bytes:
                logger.warning(f"413 body too large: path={path} content_length={declared}")
                # 413 を返す前に送信中のボディを読み捨てる。ここを省くとクライアントは
                # ボディ送信の途中で接続を切られ、**413 を読めずに ConnectionAborted になる**
                # （実測: 13MB の POST が status ではなく接続エラーで返った）。
                # 読み捨てはメモリに貯めないので OOM にはならず、上限も設けて悪用を防ぐ。
                await self._drain_body(receive)
                await self._reject_http(send, 413, "request body too large")
                return

        # 2) レート制限
        key = self.key_func(scope, headers)
        allowed, retry_after = self.limiter.check(key, cost_class)
        if not allowed:
            logger.warning(f"429 rate limited: path={path} class={cost_class} "
                           f"key={key} retry_after={retry_after:.1f}s")
            if stype == "websocket":
                # 1013 = Try Again Later. accept 前に閉じる（TTS を起動させない）。
                await send({"type": "websocket.close", "code": 1013})
            else:
                await self._reject_http(send, 429, "rate limit exceeded",
                                        extra_headers=[(b"retry-after", str(int(retry_after)).encode())])
            return

        # 3) Content-Length 無し（chunked）の保険: 受信バイトを数え、上限超過で切断する。
        if stype == "http" and not headers.get("content-length"):
            receive = self._counting_receive(receive, path)

        await self.app(scope, receive, send)

    async def _drain_body(self, receive):
        """拒否応答を返す前に、送信中のリクエストボディを読み捨てる（メモリには貯めない）。

        読み捨て量には上限を設ける（`DRAIN_LIMIT_FACTOR` 倍まで）。青天井にすると、
        巨大ボディを送り続けるだけで接続を占有され続ける（slowloris 類似）。
        上限に達したら読み捨てをやめ、応答して接続を閉じる（相手は RST を見るが、
        すでに上限超過を宣言している不正クライアントなのでそれでよい）。
        """
        drained = 0
        limit = self.max_body_bytes * self.DRAIN_LIMIT_FACTOR
        while drained < limit:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            drained += len(message.get("body", b"") or b"")
            if not message.get("more_body"):
                return

    def _counting_receive(self, receive, path):
        state = {"total": 0}

        async def wrapped():
            message = await receive()
            if message.get("type") == "http.request":
                state["total"] += len(message.get("body", b"") or b"")
                if state["total"] > self.max_body_bytes:
                    logger.warning(f"body cap exceeded on chunked request: path={path} "
                                   f"bytes={state['total']} — 接続を切断します")
                    return {"type": "http.disconnect"}
            return message

        return wrapped

    @staticmethod
    async def _reject_http(send, status, message, extra_headers=None):
        body = json.dumps({"error": message}).encode("utf-8")
        headers = [(b"content-type", b"application/json")]
        if extra_headers:
            headers.extend(extra_headers)
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
