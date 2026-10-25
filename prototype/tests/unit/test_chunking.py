"""
agent/chunking.py（semantic_chunk — 実装編 §10.1）の単体テスト。

長さ計測はテスト安定性のため文字数（len）を注入する（既定の SP トークン計測は
単位が違うだけで経路は同一）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.agent.chunking import semantic_chunk, split_by_headings

# 結合コスト（"\n" 1 個 = length_fn で 1）を見込んだ許容スラック
_JOIN_SLACK = 4


class TestSplitByHeadings(unittest.TestCase):
    def test_no_heading_is_single_section(self):
        self.assertEqual(len(split_by_headings("段落1です。\n\n段落2です。")), 1)

    def test_headings_split_and_keep_heading_line(self):
        text = "# 概要\n本文A。\n\n## 詳細\n本文B。"
        sections = split_by_headings(text)
        self.assertEqual(len(sections), 2)
        self.assertTrue(sections[0].startswith("# 概要"))
        self.assertTrue(sections[1].startswith("## 詳細"))

    def test_leading_body_before_first_heading(self):
        sections = split_by_headings("前書きです。\n# 章1\n本文。")
        self.assertEqual(len(sections), 2)
        self.assertIn("前書き", sections[0])


class TestSemanticChunk(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(semantic_chunk("", length_fn=len), [])
        self.assertEqual(semantic_chunk("   \n  ", length_fn=len), [])

    def test_short_document_is_single_chunk(self):
        chunks = semantic_chunk("短い文書です。", max_tokens=100, overlap=10, length_fn=len)
        self.assertEqual(chunks, ["短い文書です。"])

    def test_chunks_respect_max_length(self):
        paragraphs = [f"段落{i}の本文がここに入ります。" for i in range(30)]
        doc = "\n\n".join(paragraphs)
        chunks = semantic_chunk(doc, max_tokens=60, overlap=15, length_fn=len)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 60 + _JOIN_SLACK, f"チャンクが上限超過: {len(c)}")

    def test_adjacent_chunks_share_overlap(self):
        paragraphs = [f"共有内容{i}。" for i in range(20)]
        doc = "\n\n".join(paragraphs)
        chunks = semantic_chunk(doc, max_tokens=40, overlap=12, length_fn=len)
        self.assertGreater(len(chunks), 1)
        # 各チャンクの先頭断片は、直前チャンクの末尾に含まれている（持ち越し）
        for prev, cur in zip(chunks, chunks[1:]):
            head = cur.split("\n")[0]
            self.assertIn(head, prev, "隣接チャンクにオーバーラップが無い")

    def test_chunks_do_not_cross_headings(self):
        text = "# 犬の章\n" + "犬の話。" * 5 + "\n\n# 猫の章\n" + "猫の話。" * 5
        chunks = semantic_chunk(text, max_tokens=1000, overlap=0, length_fn=len)
        self.assertEqual(len(chunks), 2)
        self.assertNotIn("猫", chunks[0])
        self.assertNotIn("犬", chunks[1])

    def test_oversized_paragraph_splits_by_sentence(self):
        doc = "。".join([f"これは長い文{i}" for i in range(40)]) + "。"
        chunks = semantic_chunk(doc, max_tokens=50, overlap=0, length_fn=len)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 50 + _JOIN_SLACK)
        # 内容が失われていない（全文再構成に十分な文が残る）
        self.assertIn("これは長い文0", chunks[0])
        self.assertIn("これは長い文39", chunks[-1])

    def test_oversized_sentence_hard_splits(self):
        doc = "あ" * 300   # 句点なしの巨大 1 文
        chunks = semantic_chunk(doc, max_tokens=64, overlap=0, length_fn=len)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 64 + _JOIN_SLACK)
        self.assertEqual("".join(chunks).count("あ"), 300, "強制分割で文字が失われた")

    def test_deterministic(self):
        doc = "# H\n" + "\n\n".join(f"文{i}。" for i in range(25))
        a = semantic_chunk(doc, max_tokens=30, overlap=8, length_fn=len)
        b = semantic_chunk(doc, max_tokens=30, overlap=8, length_fn=len)
        self.assertEqual(a, b)

    def test_default_tokenizer_length_fn_works(self):
        """既定（SentencePiece トークン計測）でも例外なくチャンクできる（疎通）。"""
        chunks = semantic_chunk("これはトークナイザで長さを測る確認です。\n\n二つ目の段落。")
        self.assertGreaterEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
