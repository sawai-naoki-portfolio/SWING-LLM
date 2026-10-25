"""
投機的デコーディング（MTP ドラフト）の整合性テスト（L-8, core/generation/stream.py）。

投機 ON と OFF で「貪欲（greedy）生成の出力が完全一致する」ことを検証する。投機は
1 回のモデル前向きで複数ドラフトを受理/棄却するため、受理数・KV rollback・インデックスの
どこかにズレがあると生成が破綻（重複/欠落）する。サンプリングの確率性を排除するため
sampler を決定的 argmax に差し替え、投機ロジックのみを隔離して検証する。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.constants import TOKENIZER_MODEL_PATH

_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_PROTOTYPE_DIR, TOKENIZER_MODEL_PATH)


def _greedy_sampler(logits, generated, temperature, repetition_penalty, top_k, top_p,
                    tokenizer, EOT_TOKEN_ID, prevent_eot=False):
    """確率性を排した貪欲サンプラ。投機 ON/OFF で同一に使い、RNG 依存を除去する。"""
    lg = logits[:, -1, :].clone()
    if prevent_eot:
        lg[0, EOT_TOKEN_ID] = float("-inf")
        eot2 = getattr(tokenizer, "eot_token_id", None)
        if eot2 is not None:
            lg[0, eot2] = float("-inf")
    return int(lg[0].argmax().item())


@unittest.skipUnless(os.path.exists(_MODEL_PATH), "tokenizer.model not trained yet")
class TestSpeculativeDecodingEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from swi_llm.core.models import MiniLLM
        from swi_llm.data.tokenizer import get_tokenizer
        cls.torch = torch
        cls.tok = get_tokenizer()
        torch.manual_seed(0)
        # n_mtp_future>0 で投機ドラフトが有効になる
        cls.model = MiniLLM(vocab_size=cls.tok.vocab_size, dim=64, n_layers=2, n_heads=4,
                            hidden_dim=128, num_experts=4, top_k=2, max_seq_len=128, n_mtp_future=2)
        cls.model.eval()
        cls.prompt = "<|im_start|>user\nこんにちは<|im_end|>\n<|im_start|>assistant\n"

    def _run(self, use_spec, max_new_tokens=30):
        from swi_llm.core.generation import stream as stream_mod
        from swi_llm.utils.config import INFERENCE_CONFIG
        stream_mod.clear_prefix_cache()   # 実行間で prefix キャッシュを共有しない
        saved = INFERENCE_CONFIG.get("use_speculative_decoding")
        INFERENCE_CONFIG["use_speculative_decoding"] = use_spec
        try:
            with mock.patch.object(stream_mod, "sample_next_token", _greedy_sampler):
                out = "".join(stream_mod.generate_stream(
                    self.model, self.tok, self.prompt, max_new_tokens=max_new_tokens,
                    temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0))
            return out
        finally:
            if saved is None:
                INFERENCE_CONFIG.pop("use_speculative_decoding", None)
            else:
                INFERENCE_CONFIG["use_speculative_decoding"] = saved

    def test_spec_on_off_greedy_outputs_match(self):
        out_spec = self._run(use_spec=True)
        out_plain = self._run(use_spec=False)
        self.assertEqual(out_spec, out_plain,
                         "投機 ON/OFF で貪欲生成の出力が一致しない（受理数/rollback/index のズレの疑い）")

    def test_generation_is_nonempty(self):
        # 20 トークンは EOT 抑止のため必ず生成される（比較が自明な空一致にならないことの担保）
        out = self._run(use_spec=False)
        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
