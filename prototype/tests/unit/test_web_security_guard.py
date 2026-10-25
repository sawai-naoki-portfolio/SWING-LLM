"""
web/app.py の SecurityGuardMiddleware / 起動時バインド検証の単体テスト。

- X-1: API トークン比較が定数時間（hmac.compare_digest）であること。
- X-2: トークン認証が /api/* だけでなく /ws/* にも適用されること
       （従来 WebSocket は完全に無認証だった）。
- G-1 (WebUI ロードマップ step2): 「公開バインド ∧ allowlist 未設定」で起動を拒否すること
       （旧実装はこの組み合わせで Host 検証を*無効化*していた = fail-open）。

`WEB_API_TOKEN` / `WEB_HOST` はモジュール import 時に環境変数から解決される定数のため、
テストでは環境変数を設定してから `web.app` を import し直す。
"""
import importlib
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_TOKEN = "s3cret-token-value"

_ENV_KEYS = ("SWI_API_TOKEN", "SWI_WEB_HOST", "SWI_ALLOWED_HOSTS")
_ORIGINAL_ENV = {k: os.environ.get(k) for k in _ENV_KEYS}


def _purge_modules():
    for mod in ("swi_llm.utils.constants", "web.app"):
        sys.modules.pop(mod, None)


def _apply_env(values):
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_app_module(token, web_host=None, allowed_hosts=None):
    """指定の環境変数で web.app を再ロードして返す（None のキーは未設定にする）。"""
    _apply_env({"SWI_API_TOKEN": token, "SWI_WEB_HOST": web_host, "SWI_ALLOWED_HOSTS": allowed_hosts})
    _purge_modules()
    return importlib.import_module("web.app")


def _run_http(app_mod, headers, path="/api/generate", method="POST"):
    """SecurityGuardMiddleware を直接叩き (送信メッセージ列, 内部アプリに到達したか) を返す。"""
    import asyncio
    sent = []
    called = {"inner": False}

    async def inner_app(scope, receive, send):
        called["inner"] = True

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "path": path, "method": method,
             "headers": headers, "query_string": b""}
    mw = app_mod.SecurityGuardMiddleware(inner_app)
    asyncio.run(mw(scope, None, send))
    return sent, called["inner"]


def tearDownModule():
    """環境変数とモジュールキャッシュを元に戻し、同一プロセスで走る他テストを汚染しない。"""
    _apply_env(_ORIGINAL_ENV)
    _purge_modules()


class TestTokenComparisonIsConstantTime(unittest.TestCase):
    """X-1: `!=` ではなく hmac.compare_digest を使っていること。"""

    @classmethod
    def setUpClass(cls):
        cls.app_mod = _load_app_module(_TOKEN)

    def test_correct_token_matches(self):
        self.assertTrue(self.app_mod._token_matches(_TOKEN))

    def test_wrong_token_rejected(self):
        self.assertFalse(self.app_mod._token_matches("wrong"))
        self.assertFalse(self.app_mod._token_matches(""))
        self.assertFalse(self.app_mod._token_matches(None))

    def test_prefix_of_correct_token_rejected(self):
        # タイミング攻撃で狙われる「先頭一致」ケース
        self.assertFalse(self.app_mod._token_matches(_TOKEN[:-1]))

    def test_uses_compare_digest_not_plain_equality(self):
        import inspect
        src = inspect.getsource(self.app_mod._token_matches)
        self.assertIn("compare_digest", src)
        self.assertNotIn("!= WEB_API_TOKEN", src)

    def test_no_token_configured_allows_all(self):
        mod = _load_app_module("")           # トークン未設定 = 従来どおり無認証
        self.assertTrue(mod._token_matches(""))
        self.assertTrue(mod._token_matches("anything"))


class TestProtectedPaths(unittest.TestCase):
    """X-2: /ws/* が保護対象に含まれること。"""

    @classmethod
    def setUpClass(cls):
        cls.app_mod = _load_app_module(_TOKEN)

    def test_api_paths_protected(self):
        self.assertTrue(self.app_mod._is_protected("/api/generate"))

    def test_ws_paths_protected(self):
        # 旧実装ではここが False 相当（stype=="http" and path.startswith("/api/")）だった
        self.assertTrue(self.app_mod._is_protected("/ws/voice-stream"))

    def test_static_paths_not_protected(self):
        self.assertFalse(self.app_mod._is_protected("/"))
        self.assertFalse(self.app_mod._is_protected("/static/script.js"))


class TestTokenExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_mod = _load_app_module(_TOKEN)

    def test_bearer_header(self):
        t = self.app_mod._extract_token({"authorization": f"Bearer {_TOKEN}"}, {"type": "http"})
        self.assertEqual(t, _TOKEN)

    def test_x_api_token_header(self):
        t = self.app_mod._extract_token({"x-api-token": _TOKEN}, {"type": "http"})
        self.assertEqual(t, _TOKEN)

    def test_ws_query_param_accepted(self):
        scope = {"type": "websocket", "query_string": f"token={_TOKEN}".encode()}
        self.assertEqual(self.app_mod._extract_token({}, scope), _TOKEN)

    def test_http_query_param_ignored(self):
        # HTTP はヘッダを送れるため、クエリ経由のトークンは受け付けない（URL がログに残るため）
        scope = {"type": "http", "query_string": f"token={_TOKEN}".encode()}
        self.assertEqual(self.app_mod._extract_token({}, scope), "")

    def test_missing_token_returns_empty(self):
        self.assertEqual(self.app_mod._extract_token({}, {"type": "http"}), "")


class TestMiddlewareRejectsUnauthenticatedWebSocket(unittest.TestCase):
    """X-2 の本丸: トークン設定時、認証なしの WS ハンドシェイクが close 1008 で拒否される。"""

    @classmethod
    def setUpClass(cls):
        cls.app_mod = _load_app_module(_TOKEN)

    def _run_ws(self, scope_extra=None, headers=None):
        import asyncio
        sent = []
        called = {"inner": False}

        async def inner_app(scope, receive, send):
            called["inner"] = True

        async def send(msg):
            sent.append(msg)

        scope = {
            "type": "websocket",
            "path": "/ws/voice-stream",
            "headers": [(b"host", b"127.0.0.1:5000")],
            "query_string": b"",
        }
        if headers is not None:
            scope["headers"] = headers
        if scope_extra:
            scope.update(scope_extra)

        mw = self.app_mod.SecurityGuardMiddleware(inner_app)
        asyncio.run(mw(scope, None, send))
        return sent, called["inner"]

    def test_ws_without_token_is_closed(self):
        sent, inner_called = self._run_ws()
        self.assertFalse(inner_called, "unauthenticated WS reached the app (auth bypass)")
        self.assertEqual(sent, [{"type": "websocket.close", "code": 1008}])

    def test_ws_with_valid_query_token_passes(self):
        sent, inner_called = self._run_ws(scope_extra={"query_string": f"token={_TOKEN}".encode()})
        self.assertTrue(inner_called)
        self.assertEqual(sent, [])

    def test_ws_with_valid_header_token_passes(self):
        headers = [(b"host", b"127.0.0.1:5000"), (b"authorization", f"Bearer {_TOKEN}".encode())]
        sent, inner_called = self._run_ws(headers=headers)
        self.assertTrue(inner_called)
        self.assertEqual(sent, [])

    def test_ws_with_wrong_token_is_closed(self):
        sent, inner_called = self._run_ws(scope_extra={"query_string": b"token=wrong"})
        self.assertFalse(inner_called)
        self.assertEqual(sent, [{"type": "websocket.close", "code": 1008}])


class TestMiddlewareRejectsUnauthenticatedHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_mod = _load_app_module(_TOKEN)

    def _run_http(self, headers):
        import asyncio
        sent = []
        called = {"inner": False}

        async def inner_app(scope, receive, send):
            called["inner"] = True

        async def send(msg):
            sent.append(msg)

        scope = {"type": "http", "path": "/api/generate", "method": "POST",
                 "headers": headers, "query_string": b""}
        mw = self.app_mod.SecurityGuardMiddleware(inner_app)
        asyncio.run(mw(scope, None, send))
        return sent, called["inner"]

    def test_api_without_token_is_401(self):
        sent, inner_called = self._run_http([(b"host", b"127.0.0.1:5000")])
        self.assertFalse(inner_called)
        self.assertEqual(sent[0]["status"], 401)

    def test_api_with_token_passes(self):
        headers = [(b"host", b"127.0.0.1:5000"), (b"authorization", f"Bearer {_TOKEN}".encode())]
        sent, inner_called = self._run_http(headers)
        self.assertTrue(inner_called)
        self.assertEqual(sent, [])


class TestPublicBindFailsClosed(unittest.TestCase):
    """G-1 / WebUI ロードマップ step2: 公開バインド × allowlist 未設定は**起動を拒否**する。

    旧実装（fail-open）はこの組み合わせで `_ENFORCE_HOST = False` とし、Host 検証を丸ごと切っていた。
    その場合 web.app の import は成功してしまうため、本クラスの
    `test_public_bind_without_allowlist_refuses_startup` は旧コードでは必ず落ちる。
    """

    def test_public_bind_without_allowlist_refuses_startup(self):
        for host in ("0.0.0.0", "::"):
            with self.subTest(bind=host):
                with self.assertRaises(RuntimeError) as ctx:
                    _load_app_module("", web_host=host, allowed_hosts=None)
                self.assertIn("SWI_ALLOWED_HOSTS", str(ctx.exception))

    def test_public_bind_with_allowlist_starts_and_still_enforces_host(self):
        mod = _load_app_module("", web_host="0.0.0.0", allowed_hosts="swi.example.com")
        self.assertIn("swi.example.com", mod._ALLOWED_HOSTS)
        self.assertFalse(mod._WILDCARD_HOSTS)

        sent, inner_called = _run_http(mod, [(b"host", b"evil.example.net")])
        self.assertFalse(inner_called, "許可外 Host が素通りした（DNS リバインディング面が開いている）")
        self.assertEqual(sent[0]["status"], 400)

        sent, inner_called = _run_http(mod, [(b"host", b"swi.example.com")])
        self.assertTrue(inner_called, "許可済み Host が拒否された（正規利用の締め出し）")

    def test_wildcard_is_the_only_escape_hatch(self):
        # 検証を切りたい運用者は "*" を明示する。暗黙に切れる経路は残さない。
        mod = _load_app_module("", web_host="0.0.0.0", allowed_hosts="*")
        self.assertTrue(mod._WILDCARD_HOSTS)
        _, inner_called = _run_http(mod, [(b"host", b"anything.example.net")])
        self.assertTrue(inner_called)

    def test_loopback_bind_starts_without_allowlist(self):
        # 既定のローカル開発フロー（127.0.0.1）は従来どおり無設定で起動できる。
        mod = _load_app_module("", web_host="127.0.0.1", allowed_hosts=None)
        self.assertIn("127.0.0.1", mod._ALLOWED_HOSTS)
        _, inner_called = _run_http(mod, [(b"host", b"127.0.0.1:5000")])
        self.assertTrue(inner_called)

    def test_startup_checklist_flags_public_bind_without_token(self):
        mod = _load_app_module("", web_host="0.0.0.0", allowed_hosts="swi.example.com")
        with self.assertLogs(mod.logger, level="WARNING") as captured:
            mod.log_startup_checklist()
        joined = "\n".join(captured.output)
        self.assertIn("SWI_API_TOKEN", joined)


if __name__ == "__main__":
    unittest.main()
