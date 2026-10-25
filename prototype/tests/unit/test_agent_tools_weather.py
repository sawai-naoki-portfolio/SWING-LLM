"""
agent/tools.py get_weather の応答検証（2026-07-12 実測の回帰）。

wttr.in がプレーンテキストではなく HTML ページ全文を返すことがあり、
それをそのまま condition に入れると LLM のコンテキストを汚染して無関係な生成
（HTML 出力）へ暴走した。HTML 応答は「取得失敗」として扱う。
"""
import io
import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.agent.tools import get_weather


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestGetWeatherResponseGuard(unittest.TestCase):
    def _with_body(self, body: bytes):
        return mock.patch("urllib.request.urlopen", return_value=_FakeResponse(body))

    def test_plain_text_response_is_returned(self):
        with self._with_body("東京: ☀️ +30°C".encode("utf-8")):
            result = get_weather("東京")
        self.assertEqual(result.get("location"), "東京")
        self.assertIn("+30°C", result.get("condition", ""))

    def test_html_response_is_treated_as_failure(self):
        html = b"<!DOCTYPE html>\n<html><head><title>wttr.in</title></head><body>...</body></html>"
        with self._with_body(html):
            result = get_weather("Tokyo")
        self.assertIn("error", result)
        self.assertNotIn("condition", result)

    def test_html_response_lowercase_variant(self):
        with self._with_body(b"  <html><body>page</body></html>"):
            result = get_weather("Tokyo")
        self.assertIn("error", result)

    def test_network_failure_returns_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("net down")):
            result = get_weather("Tokyo")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
