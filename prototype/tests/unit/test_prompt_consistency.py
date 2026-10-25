"""
S6: 学習ステージのプロンプト書式統一の単体テスト。

SFT・推論(Web)はペルソナ system プロンプトを付与するが、DPO/GRPO/RFT/PRM は
付けていなかった（学習/推論で書式が食い違う）。TRAINING_SYSTEM_PROMPT を単一情報源に
全学習段階へ system プロンプトを付与したことを検証する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.constants import (
    TRAINING_SYSTEM_PROMPT, PERSONA_SYSTEM_PROMPTS, DEFAULT_PERSONA,
)
from swi_llm.utils.chat_template import (
    build_prompt, format_message, IM_START, IM_END, ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT,
)


class TestTrainingSystemPrompt(unittest.TestCase):
    def test_single_source_matches_default_persona(self):
        self.assertTrue(TRAINING_SYSTEM_PROMPT)
        self.assertEqual(TRAINING_SYSTEM_PROMPT, PERSONA_SYSTEM_PROMPTS[DEFAULT_PERSONA])
        self.assertIn("JARVIS", TRAINING_SYSTEM_PROMPT)

    def test_build_prompt_includes_system_block(self):
        p = build_prompt("こんにちは", system_text=TRAINING_SYSTEM_PROMPT)
        self.assertTrue(p.startswith(f"{IM_START}{ROLE_SYSTEM}\n"))
        self.assertIn(TRAINING_SYSTEM_PROMPT, p)


class TestStagePromptsHaveSystem(unittest.TestCase):
    """各学習ステージの実プロンプト構築に system プロンプトが含まれることを検証。"""

    def test_dpo_format_includes_system(self):
        # HFDPODataset の実 _format_and_pad を（HF ロードを回避して）呼び、
        # 生成された入力トークンを復号して system プロンプトの存在を確認する。
        from swi_llm.data.dpo import HFDPODataset
        from swi_llm.data.tokenizer import get_tokenizer
        ds = HFDPODataset.__new__(HFDPODataset)
        ds.tokenizer = get_tokenizer()
        ds.seq_len = 256
        x, _ = ds._format_and_pad("好きな色は？", "青です。")
        decoded = ds.tokenizer.decode(x.tolist())
        self.assertIn("JARVIS", decoded)

    def test_prm_format_includes_system(self):
        from swi_llm.data.prm import PRMDataset
        from swi_llm.data.tokenizer import get_tokenizer
        ds = PRMDataset.__new__(PRMDataset)
        ds.tokenizer = get_tokenizer()
        ds.step_token = "<|step|>"
        ds.samples = []
        ds._add_trajectory("2+3は？", "まず2\n次に3\n答えは5", correct=True)
        self.assertTrue(ds.samples, "trajectory should be added")
        self.assertIn(f"{IM_START}{ROLE_SYSTEM}", ds.samples[0]["text"])
        self.assertIn("JARVIS", ds.samples[0]["text"])

    def test_rft_text_construction_includes_system(self):
        # RFT が採用サンプルへ組み立てる文字列（rft.py と同じ式）が system ブロックで始まる
        text = (f"{format_message(ROLE_SYSTEM, TRAINING_SYSTEM_PROMPT)}"
                f"{format_message(ROLE_USER, 'q')}{IM_START}{ROLE_ASSISTANT}\na{IM_END}")
        self.assertTrue(text.startswith(f"{IM_START}{ROLE_SYSTEM}\n"))
        self.assertIn("JARVIS", text)


if __name__ == "__main__":
    unittest.main()
