import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer, EOT_TOKEN_ID, PAD_TOKEN_ID
from swi_llm.utils.config import config
from swi_llm.utils.constants import DATASET_DOLLY_JA, DATASET_FUNCTION_CALLING, TRAINING_SYSTEM_PROMPT
from swi_llm.utils.chat_template import (
    build_prompt, format_assistant_response, wrap_with_think,
    TOOL_CALL_START, TOOL_CALL_END, LEGACY_FUNCTIONCALL_START, LEGACY_FUNCTIONCALL_END,
)
from swi_llm.data.tool_calling_examples import build_tool_calling_examples

logger = setup_logger("data.sft")

# ローカルツール（calculate/get_weather/search_documents）に一致した合成 FC 教師データの件数。
# 正例 = 3 ツール × PER_TOOL、負例 = NEGATIVE（ツールを呼ばずに直接答える例）。
LOCAL_TOOL_EXAMPLES_PER_TOOL = 40
LOCAL_TOOL_NEGATIVE_EXAMPLES = 60

class HFSFTDataset(Dataset):
    def __init__(self, seq_len: int, split='train', max_samples=None):
        self.seq_len = seq_len
        self.tokenizer = get_tokenizer()
        logger.info(f"Loading {DATASET_DOLLY_JA} dataset...")
        try:
            dolly_ds = load_dataset(DATASET_DOLLY_JA, split=split)
            if max_samples and len(dolly_ds) > max_samples:
                dolly_ds = dolly_ds.select(range(max_samples))
                
            # Convert Dolly to our standard format
            rng = random.Random(42)
            self.dataset = []
            for row in dolly_ds:
                user_text = str(row['instruction']).strip()
                if row.get('input'):
                    user_text += "\n" + str(row['input']).strip()
                response_text = str(row['output']).strip()
                
                if rng.random() < 0.5:
                    short_user = user_text[:30].replace('\n', ' ')
                    reasoning = f"ユーザーの質問は「{short_user}...」に関連している。\n背景知識を整理し、事実に基づいた的確で分かりやすい回答を構成する必要がある。"
                    response_text = wrap_with_think(reasoning, response_text)
                    
                self.dataset.append({"user": user_text, "assistant": response_text})
                
        except Exception as e:
            logger.exception("Failed to load Dolly dataset.")
            raise
            
        logger.info(f"Loading {DATASET_FUNCTION_CALLING} for real tool calling data...")
        try:
            tool_ds = load_dataset(DATASET_FUNCTION_CALLING, split='train')
            if max_samples and len(tool_ds) > max_samples:
                tool_ds = tool_ds.select(range(max_samples))
                tool_samples = len(tool_ds)
            else:
                tool_samples = min(2000, len(tool_ds))
            for i in range(tool_samples):
                row = tool_ds[i]
                messages = row.get('messages', [])
                if len(messages) >= 3 and messages[1]['role'] == 'user' and messages[2]['role'] == 'assistant':
                    user_text = messages[1]['content'].strip()
                    response_text = (
                        messages[2]['content'].strip()
                        .replace(LEGACY_FUNCTIONCALL_START, TOOL_CALL_START)
                        .replace(LEGACY_FUNCTIONCALL_END, TOOL_CALL_END)
                    )

                    if rng.random() < 0.5:
                        short_user = user_text[:30].replace('\n', ' ')
                        reasoning = f"ユーザーの要求は「{short_user}...」である。\n利用可能なツールセットから適切なツールを選択し、引数を正確に抽出して関数呼び出しを行うのが最適だ。"
                        response_text = wrap_with_think(reasoning, response_text)
                        
                    self.dataset.append({"user": user_text, "assistant": response_text})
            logger.info(f"Appended {tool_samples} real tool-calling examples.")
        except Exception as e:
            logger.warning(f"Failed to load tool calling dataset: {e}. Skipping tool data.")

        # roadmap ステップ3 / fix 19: 外部 FC データセットのツール名・引数名は本プロジェクトに
        # 実際に登録されるツール（TOOL_SCHEMAS）と一致しないため、「呼べば実際に dispatch
        # される tool_call」を学べない。ローカルツールに厳密一致した正例と、ツール不要時に
        # tool_call を出さない負例（過剰発火の抑制）を合成して必ず追加する。
        # 外部データセットの取得失敗（オフライン等）に依存しないよう try の外で行う。
        local_tool_examples = build_tool_calling_examples(
            n_per_tool=LOCAL_TOOL_EXAMPLES_PER_TOOL, n_negative=LOCAL_TOOL_NEGATIVE_EXAMPLES)
        self.dataset.extend(local_tool_examples)
        logger.info(f"Appended {len(local_tool_examples)} local tool-calling examples "
                    f"(schema-aligned positives + no-tool negatives).")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        user_text = item["user"]
        response_text = item["assistant"]

        # 学習時のシステムプロンプトは推論 (Web UI) と共通（S6: 全学習段階で統一）
        prompt_str = build_prompt(user_text, system_text=TRAINING_SYSTEM_PROMPT)
        response_str = format_assistant_response(response_text)
            
        prompt_tokens = self.tokenizer.encode(prompt_str)
        response_tokens = self.tokenizer.encode(response_str) + [EOT_TOKEN_ID]
        
        tokens = prompt_tokens + response_tokens
        labels = [-100] * len(prompt_tokens) + response_tokens
        
        if len(tokens) > self.seq_len + 1:
            tokens = tokens[:self.seq_len + 1]
            labels = labels[:self.seq_len + 1]
        else:
            padding_len = self.seq_len + 1 - len(tokens)
            tokens = tokens + [PAD_TOKEN_ID] * padding_len
            labels = labels + [-100] * padding_len
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y

def get_sft_dataloaders(seq_len: int, batch_size: int, val_split: float = 0.1, max_samples=None):
    dataset = HFSFTDataset(seq_len, max_samples=max_samples)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
