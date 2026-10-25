"""
/api/llm/status（web/routes/inference.py）の単体テスト。

このエンドポイントは WebUI の「機能出し分け」の唯一の情報源である。
完全自作方針（2026-10-25）で外部 LLM ゲートウェイを撤去したため、推論は常に自作モデル。
レスポンス形状（backend / model / capabilities のキー）はクライアント互換のため維持する。
"""
import asyncio
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from web.routes import inference


def _status():
    return asyncio.run(inference.api_llm_status())


class TestLlmStatusEndpoint(unittest.TestCase):
    def test_reports_from_scratch_model_with_all_capabilities(self):
        body = _status()
        self.assertEqual(body["backend"], "local")
        self.assertIn("from-scratch", body["model"])
        caps = body["capabilities"]
        self.assertTrue(caps["vision"])
        self.assertTrue(caps["cot"])
        self.assertTrue(caps["best_of_n"])
        self.assertTrue(caps["tools"])

    def test_response_keys_are_stable(self):
        """WebUI / モバイルが参照するキー名の契約。欠けるとフロントが機能を出し分けられない。"""
        body = _status()
        self.assertEqual(set(body), {"backend", "model", "capabilities"})
        self.assertEqual(set(body["capabilities"]), {"vision", "cot", "best_of_n", "tools"})

    def test_external_gateway_is_gone(self):
        """外部 LLM ゲートウェイへの経路が復活していないこと（完全自作方針の回帰検知）。"""
        self.assertFalse(hasattr(inference, "backend_enabled"))
        with self.assertRaises(ImportError):
            import web.llm_backend  # noqa: F401


if __name__ == "__main__":
    unittest.main()
