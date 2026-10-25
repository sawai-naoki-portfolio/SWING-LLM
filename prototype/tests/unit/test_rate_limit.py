"""SEC-12 / WebUI ロードマップ step5: レート制限とボディサイズ上限の単体テスト。

修正前は生成系 6 経路（/api/generate, /api/generate_stream, /api/generate_image,
/api/tts, /api/tts_stream, /ws/voice-stream）が完全に無制限だった。
本テストは「バケットが枯渇したら 429 / WS close(1013) で止まる」ことを固定する。
"""
import asyncio
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from web.rate_limit import (
    TokenBucketLimiter, RateLimitMiddleware, classify_path, client_ip,
)


class FakeClock:
    """時間を進められる時計。sleep せずに補充を検証するため。"""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _run_asgi(make_middleware, scope, body_chunks=None):
    """ミドルウェアを直接叩き (送信メッセージ列, 内部アプリ到達フラグ) を返す。

    `make_middleware` は「内部アプリを受け取ってミドルウェアを返す」ファクトリ。
    バケットの状態は呼び出しをまたいで共有したいので、limiter はファクトリ側が閉じ込める。
    """
    sent = []
    called = {"inner": False}

    async def inner_app(scope, receive, send):
        called["inner"] = True
        if scope["type"] == "http" and body_chunks is not None:
            while True:                       # ボディを読み切る（counting_receive を通す）
                message = await receive()
                if message["type"] == "http.disconnect":
                    called["disconnected"] = True
                    return
                if not message.get("more_body"):
                    return

    async def send(message):
        sent.append(message)

    queue = list(body_chunks or [])

    async def receive():
        if queue:
            return queue.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(make_middleware(inner_app)(scope, receive, send))
    called["queue_left"] = len(queue)          # 読み捨て（drain）できているかの確認用
    return sent, called


def _http_scope(path="/api/generate", headers=None, method="POST"):
    return {"type": "http", "path": path, "method": method,
            "headers": headers or [], "query_string": b"", "client": ("1.2.3.4", 51000)}


def _ws_scope(path="/ws/voice-stream"):
    return {"type": "websocket", "path": path, "headers": [],
            "query_string": b"", "client": ("1.2.3.4", 51000)}


class TestClassifyPath(unittest.TestCase):
    def test_generate_variants_are_not_confused(self):
        # "/api/generate" は "/api/generate_stream" / "/api/generate_image" の接頭辞なので、
        # 接頭辞マッチだと画像生成が安いクラスに落ちてしまう。完全一致であることを固定する。
        self.assertEqual(classify_path("/api/generate"), "generate")
        self.assertEqual(classify_path("/api/generate_stream"), "generate")
        self.assertEqual(classify_path("/api/generate_image"), "image")

    def test_voice_paths(self):
        self.assertEqual(classify_path("/api/tts"), "tts")
        self.assertEqual(classify_path("/api/tts_stream"), "tts")
        self.assertEqual(classify_path("/ws/voice-stream"), "tts")

    def test_other_api_paths_get_default_class(self):
        self.assertEqual(classify_path("/api/llm/status"), "default")
        self.assertEqual(classify_path("/api/feedback"), "default")

    def test_static_paths_are_not_limited(self):
        self.assertIsNone(classify_path("/"))
        self.assertIsNone(classify_path("/script.js"))


class TestClientIp(unittest.TestCase):
    def test_x_forwarded_for_is_not_trusted_by_default(self):
        # 信頼するとヘッダ偽装だけでキーを変えられ、レート制限が自明に回避される。
        scope = _http_scope()
        ip = client_ip(scope, {"x-forwarded-for": "9.9.9.9"}, trust_proxy=False)
        self.assertEqual(ip, "1.2.3.4")

    def test_x_forwarded_for_used_when_proxy_trusted(self):
        scope = _http_scope()
        ip = client_ip(scope, {"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, trust_proxy=True)
        self.assertEqual(ip, "9.9.9.9")


class TestTokenBucketLimiter(unittest.TestCase):
    def test_burst_is_allowed_then_blocked(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rules={"generate": (60, 3), "default": (120, 30)}, clock=clock)
        for i in range(3):
            allowed, _ = limiter.check("ip:a", "generate")
            self.assertTrue(allowed, f"burst {i + 1} 回目が拒否された")
        allowed, retry_after = limiter.check("ip:a", "generate")
        self.assertFalse(allowed, "バースト超過が通ってしまった")
        self.assertGreaterEqual(retry_after, 1.0)

    def test_bucket_refills_over_time(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rules={"generate": (60, 2), "default": (120, 30)}, clock=clock)
        limiter.check("ip:a", "generate")
        limiter.check("ip:a", "generate")
        self.assertFalse(limiter.check("ip:a", "generate")[0])

        clock.advance(1.0)                 # 60/min = 1 tok/sec -> 1 個補充
        self.assertTrue(limiter.check("ip:a", "generate")[0])
        self.assertFalse(limiter.check("ip:a", "generate")[0])

    def test_refill_is_capped_at_burst(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rules={"generate": (60, 2), "default": (120, 30)}, clock=clock)
        limiter.check("ip:a", "generate")
        clock.advance(10_000)              # 長時間放置しても容量を超えて貯まらない
        self.assertTrue(limiter.check("ip:a", "generate")[0])
        self.assertTrue(limiter.check("ip:a", "generate")[0])
        self.assertFalse(limiter.check("ip:a", "generate")[0])

    def test_keys_are_independent(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rules={"generate": (60, 1), "default": (120, 30)}, clock=clock)
        self.assertTrue(limiter.check("ip:a", "generate")[0])
        self.assertFalse(limiter.check("ip:a", "generate")[0])
        self.assertTrue(limiter.check("ip:b", "generate")[0], "別クライアントが巻き添えで拒否された")

    def test_cost_classes_are_independent(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rules={"generate": (60, 1), "image": (60, 1), "default": (120, 30)},
                                     clock=clock)
        self.assertTrue(limiter.check("ip:a", "generate")[0])
        self.assertTrue(limiter.check("ip:a", "image")[0], "クラスをまたいでバケットが共有されている")

    def test_key_table_is_bounded(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rules={"default": (120, 30)}, max_keys=10, clock=clock)
        for i in range(100):
            limiter.check(f"ip:{i}", "default")
        self.assertLessEqual(len(limiter._buckets), 10, "鍵表が無制限に増える（メモリ枯渇 DoS）")


class TestRateLimitMiddleware(unittest.TestCase):
    def _factory(self, rules=None, clock=None, max_body_bytes=1000, enabled=True):
        """limiter を 1 つ閉じ込めたミドルウェアファクトリを返す（呼び出し間でバケットを共有する）。"""
        limiter = TokenBucketLimiter(rules=rules or {"generate": (60, 1), "tts": (60, 1),
                                                     "image": (60, 1), "default": (120, 30)},
                                     clock=clock or FakeClock())

        def make(app):
            return RateLimitMiddleware(app, limiter=limiter, enabled=enabled,
                                       max_body_bytes=max_body_bytes)
        return make

    def test_http_over_limit_returns_429_with_retry_after(self):
        mw = self._factory()
        sent, called = _run_asgi(mw, _http_scope())
        self.assertTrue(called["inner"], "1 回目が通らない")

        sent, called = _run_asgi(mw, _http_scope())
        self.assertFalse(called["inner"], "レート超過なのに推論に到達した")
        self.assertEqual(sent[0]["status"], 429)
        header_names = [k for k, _ in sent[0]["headers"]]
        self.assertIn(b"retry-after", header_names)

    def test_websocket_over_limit_is_closed_1013(self):
        mw = self._factory()
        sent, called = _run_asgi(mw, _ws_scope())
        self.assertTrue(called["inner"])

        sent, called = _run_asgi(mw, _ws_scope())
        self.assertFalse(called["inner"], "レート超過なのに TTS WS が起動した")
        self.assertEqual(sent, [{"type": "websocket.close", "code": 1013}])

    def test_static_paths_bypass_the_limiter(self):
        mw = self._factory()
        for _ in range(50):
            _, called = _run_asgi(mw, _http_scope(path="/", method="GET"))
            self.assertTrue(called["inner"])

    def test_oversized_content_length_is_413(self):
        mw = self._factory(max_body_bytes=1000)
        scope = _http_scope(headers=[(b"content-length", b"999999")])
        sent, called = _run_asgi(mw, scope)
        self.assertFalse(called["inner"], "上限超過のボディが推論に到達した")
        self.assertEqual(sent[0]["status"], 413)

    def test_body_within_limit_passes(self):
        mw = self._factory(max_body_bytes=1000)
        scope = _http_scope(headers=[(b"content-length", b"500")])
        _, called = _run_asgi(mw, scope)
        self.assertTrue(called["inner"])

    def test_oversized_body_is_drained_before_responding_413(self):
        """413 を返す前に送信中のボディを読み捨てること。

        これを省くと、クライアントはボディ送信の途中で接続を切られ **413 を読めない**
        （実測: 12MB 上限に対する 13MB の POST が、status ではなく ConnectionAbortedError で返った）。
        ボディはメモリに貯めずに捨てるので OOM にはならない。
        """
        # 宣言 120B > 上限 100B なので 413。合計 120B は読み捨て上限(100×2=200B)の内側なので全量捨てられる。
        mw = self._factory(max_body_bytes=100)
        chunks = [{"type": "http.request", "body": b"x" * 60, "more_body": True},
                  {"type": "http.request", "body": b"x" * 60, "more_body": False}]
        scope = _http_scope(headers=[(b"content-length", b"120")])
        sent, called = _run_asgi(mw, scope, body_chunks=chunks)

        self.assertFalse(called["inner"], "上限超過のボディがアプリに渡った")
        self.assertEqual(sent[0]["status"], 413)
        self.assertEqual(called["queue_left"], 0,
                         "ボディを読み捨てずに 413 を返している（クライアントが応答を読めない）")

    def test_drain_itself_is_bounded(self):
        """読み捨てにも上限があること（青天井だと巨大ボディで接続を占有され続ける）。

        上限（max_body_bytes × DRAIN_LIMIT_FACTOR）を超えたら読み捨てをやめ、
        413 を返して接続を閉じる。相手は応答を読めないかもしれないが、
        すでに上限超過を宣言している不正クライアントなのでそれでよい。
        """
        mw = self._factory(max_body_bytes=100)          # 読み捨て上限 = 200B
        chunks = [{"type": "http.request", "body": b"x" * 500, "more_body": True},
                  {"type": "http.request", "body": b"x" * 500, "more_body": True},
                  {"type": "http.request", "body": b"x" * 500, "more_body": False}]
        scope = _http_scope(headers=[(b"content-length", b"1500")])
        sent, called = _run_asgi(mw, scope, body_chunks=chunks)

        self.assertEqual(sent[0]["status"], 413)
        self.assertGreater(called["queue_left"], 0, "読み捨てが無制限に続いている")

    def test_chunked_body_over_cap_is_disconnected(self):
        # Content-Length を付けない（chunked）経路。受信バイトを数えて切断する。
        mw = self._factory(max_body_bytes=100)
        chunks = [{"type": "http.request", "body": b"x" * 80, "more_body": True},
                  {"type": "http.request", "body": b"x" * 80, "more_body": False}]
        _, called = _run_asgi(mw, _http_scope(), body_chunks=chunks)
        self.assertTrue(called["inner"])                      # ハンドラには入る
        self.assertTrue(called.get("disconnected"), "上限超過の chunked ボディが読み切られた")

    def test_disabled_limiter_passes_everything(self):
        mw = self._factory(enabled=False)
        for _ in range(20):
            _, called = _run_asgi(mw, _http_scope())
            self.assertTrue(called["inner"])


if __name__ == "__main__":
    unittest.main()
