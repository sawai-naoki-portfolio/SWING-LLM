"""
tests/unit/test_wikipedia_tool.py

NEW-L6（03_agentic 展望2）: wikipedia_search ツール。
HTTP はモックし、パース・HTML 除去・言語許可リスト・エラー縮退の契約を検証する。
実 API の疎通は SWI_WIKI_INTEGRATION_TESTS=1 のオプトイン。
"""
import io
import json
import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.agent import tools
from swi_llm.agent.tools import wikipedia_search, TOOL_SCHEMAS
from swi_llm.utils.constants import TOOL_WIKIPEDIA_SEARCH


def _fake_response(payload):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False
    return _Resp(json.dumps(payload).encode("utf-8"))


_PAYLOAD = {"query": {"search": [
    {"title": "量子力学", "pageid": 100,
     "snippet": '<span class="searchmatch">量子力学</span>は物理学の理論。'},
    {"title": "量子コンピュータ", "pageid": 200, "snippet": "計算機の一種。"},
]}}


class TestWikipediaSearch(unittest.TestCase):
    def test_parses_hits_and_strips_html(self):
        with mock.patch.object(tools.urllib.request, "urlopen",
                               return_value=_fake_response(_PAYLOAD)) as m:
            out = wikipedia_search("量子力学")
        self.assertEqual(len(out["results"]), 2)
        first = out["results"][0]
        self.assertEqual(first["title"], "量子力学")
        self.assertNotIn("<", first["snippet"])                 # HTML タグ除去
        self.assertEqual(first["url"], "https://ja.wikipedia.org/?curid=100")
        req = m.call_args[0][0]
        self.assertIn("ja.wikipedia.org", req.full_url)         # 既定は日本語版
        self.assertIn("User-agent", req.headers)                # UA 明示（API 規約）

    def test_lang_en_and_whitelist_fallback(self):
        with mock.patch.object(tools.urllib.request, "urlopen",
                               return_value=_fake_response(_PAYLOAD)) as m:
            wikipedia_search("quantum", lang="en")
        self.assertIn("en.wikipedia.org", m.call_args[0][0].full_url)
        # 許可リスト外 → ja へフォールバック（任意ホストを組み立てない = SSRF 対策）
        with mock.patch.object(tools.urllib.request, "urlopen",
                               return_value=_fake_response(_PAYLOAD)) as m:
            wikipedia_search("test", lang="evil.example.com/")
        self.assertIn("ja.wikipedia.org", m.call_args[0][0].full_url)

    def test_empty_query_is_error(self):
        self.assertIn("error", wikipedia_search(""))
        self.assertIn("error", wikipedia_search(None))

    def test_no_hits_returns_note(self):
        with mock.patch.object(tools.urllib.request, "urlopen",
                               return_value=_fake_response({"query": {"search": []}})):
            out = wikipedia_search("zzz_no_such_topic")
        self.assertEqual(out["results"], [])
        self.assertIn("note", out)

    def test_network_failure_degrades_to_error_dict(self):
        with mock.patch.object(tools.urllib.request, "urlopen",
                               side_effect=OSError("offline")):
            out = wikipedia_search("量子力学")
        self.assertIn("error", out)

    def test_schema_registered(self):
        schema = TOOL_SCHEMAS[TOOL_WIKIPEDIA_SEARCH]
        self.assertEqual(schema["name"], "wikipedia_search")
        self.assertIn("query", schema["parameters"]["properties"])
        self.assertEqual(schema["parameters"]["required"], ["query"])


@unittest.skipUnless(os.environ.get("SWI_WIKI_INTEGRATION_TESTS") == "1",
                     "set SWI_WIKI_INTEGRATION_TESTS=1 for the live API test")
class TestLiveWikipedia(unittest.TestCase):
    def test_live_search(self):
        out = wikipedia_search("量子力学")
        self.assertNotIn("error", out)
        self.assertGreater(len(out["results"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
