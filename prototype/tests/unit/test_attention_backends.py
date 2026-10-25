"""
tests/unit/test_attention_backends.py

NEW-L4（01_core 展望2）: SDPA バックエンドの明示有効化と実行プローブ。
"""
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.attention_backends import configure_sdp_backends, probe_flash_attention


class TestAttentionBackends(unittest.TestCase):
    def test_configure_returns_status_and_is_idempotent(self):
        s1 = configure_sdp_backends()
        s2 = configure_sdp_backends()
        self.assertIn("cuda", s1)
        self.assertEqual(s1["cuda"], s2["cuda"])
        if s1["cuda"]:
            self.assertTrue(s1["flash_enabled"])           # 明示有効化されている
            self.assertTrue(s1["mem_efficient_enabled"])
            self.assertIn("flash_kernel_works", s1)        # 実行プローブの事実が記録される

    def test_probe_never_raises(self):
        result = probe_flash_attention()
        self.assertIsInstance(result, bool)
        if not torch.cuda.is_available():
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
