"""
PRM Best-of-N (swi_llm/core/generation/best_of_n.py) の単体テスト（§8.2）。
tokenizer.model が必要なため未生成環境ではスキップ。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.constants import TOKENIZER_MODEL_PATH

_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_PROTOTYPE_DIR, TOKENIZER_MODEL_PATH)


@unittest.skipUnless(os.path.exists(_MODEL_PATH), "tokenizer.model not trained yet")
class TestBestOfN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from swi_llm.core.models import MiniLLM
        from swi_llm.data.tokenizer import get_tokenizer
        cls.torch = torch
        cls.tok = get_tokenizer()
        cls.model = MiniLLM(vocab_size=cls.tok.vocab_size, dim=64, n_layers=2, n_heads=4,
                            hidden_dim=128, num_experts=4, top_k=2, max_seq_len=128, n_mtp_future=2)
        cls.model.eval()
        cls.prompt = "<|im_start|>user\n2+2は？<|im_end|>\n<|im_start|>assistant\n"

    def test_score_is_float(self):
        from swi_llm.core.generation import score_with_prm
        s = score_with_prm(self.model, self.tok, self.prompt, "答えは4です。")
        self.assertIsInstance(s, float)

    def test_best_of_n_selects_highest(self):
        from swi_llm.core.generation import generate_best_of_n
        self.torch.manual_seed(0)
        best, cands = generate_best_of_n(self.model, self.tok, self.prompt, n=3, max_new_tokens=10)
        self.assertEqual(len(cands), 3)
        # 降順ソートされ、先頭が最高スコア
        self.assertEqual(cands[0][0], max(c[0] for c in cands))
        self.assertTrue(best.startswith(self.prompt))


if __name__ == "__main__":
    unittest.main()
