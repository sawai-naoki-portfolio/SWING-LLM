"""
SentencePiece トークナイザ (swi_llm/data/tokenizer.py) の単体テスト（ブループリント §1.4）。

tokenizer.model が未生成の環境ではスキップする（学習は train_tokenizer.py）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.constants import (
    TOKENIZER_MODEL_PATH, TOKENIZER_VOCAB_SIZE, TOKENIZER_SPECIAL_TOKENS,
    TOKEN_EOS_ID, TOKEN_PAD_ID,
)
from swi_llm.utils.chat_template import build_prompt

_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_PROTOTYPE_DIR, TOKENIZER_MODEL_PATH)


@unittest.skipUnless(os.path.exists(_MODEL_PATH), "tokenizer.model not trained yet")
class TestSPTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swi_llm.data.tokenizer import get_tokenizer, EOT_TOKEN_ID, PAD_TOKEN_ID, VOCAB_SIZE
        cls.tok = get_tokenizer()
        cls.EOT = EOT_TOKEN_ID
        cls.PAD = PAD_TOKEN_ID
        cls.VOCAB = VOCAB_SIZE

    def test_vocab_size(self):
        self.assertEqual(self.VOCAB, TOKENIZER_VOCAB_SIZE)

    def test_reserved_ids(self):
        self.assertEqual(self.EOT, TOKEN_EOS_ID)
        self.assertEqual(self.PAD, TOKEN_PAD_ID)

    def test_roundtrip_japanese(self):
        text = "日本語のトークン化テスト、句読点。「引用」も含む。"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_roundtrip_chatml(self):
        text = build_prompt("計算して: 12 + 34", system_text="あなたはアシスタントです", cot=True)
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_byte_fallback_emoji(self):
        # 学習コーパスに無い絵文字も byte_fallback で往復できる
        text = "絵文字😀🎉のテスト"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_special_tokens_are_atomic(self):
        # ChatML / 思考 / ツール系の特殊トークンは 1 ID に収まる
        for t in TOKENIZER_SPECIAL_TOKENS:
            ids = self.tok.encode(t)
            self.assertEqual(len(ids), 1, f"{t!r} should be a single token, got {ids}")

    def test_encode_concatenation_safe(self):
        # prompt と response を個別に encode して連結しても、まとめて encode した
        # 結果と一致する（学習データ組み立ての前提。add_dummy_prefix=False）
        a = build_prompt("こんにちは")
        b = "はい、こんにちは。<|im_end|>\n"
        self.assertEqual(self.tok.encode(a) + self.tok.encode(b), self.tok.encode(a + b))

    def test_all_ids_in_range(self):
        ids = self.tok.encode("任意のテキスト abc 123 <|im_start|>")
        self.assertTrue(all(0 <= i < self.VOCAB for i in ids))


if __name__ == "__main__":
    unittest.main()
