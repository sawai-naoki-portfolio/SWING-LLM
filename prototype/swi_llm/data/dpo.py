import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer, EOT_TOKEN_ID, PAD_TOKEN_ID
from swi_llm.utils.constants import DATASET_HH_RLHF_JA, TRAINING_SYSTEM_PROMPT
from swi_llm.utils.chat_template import build_prompt, format_assistant_response

logger = setup_logger("data.dpo")

class HFDPODataset(Dataset):
    def __init__(self, seq_len: int, split='train', max_samples=None):
        self.seq_len = seq_len
        self.tokenizer = get_tokenizer()
        logger.info(f"Loading {DATASET_HH_RLHF_JA} dataset for DPO...")
        try:
            raw_dataset = load_dataset(DATASET_HH_RLHF_JA, split=split)
        except Exception as e:
            logger.exception(f"Failed to load {DATASET_HH_RLHF_JA} dataset.")
            raise
            
        # Filter out rows with invalid or missing rejected responses (e.g. 0.0)
        valid_data = []
        for row in raw_dataset:
            ng = row.get('ng_translation')
            if isinstance(ng, str) and len(ng.strip()) > 0:
                valid_data.append(row)
            if max_samples and len(valid_data) >= max_samples:
                break
                
        self.dataset = valid_data
        logger.info(f"Loaded {len(self.dataset)} valid DPO examples.")

    def __len__(self):
        return len(self.dataset)

    def _format_and_pad(self, user_text, response_text):
        # S6: SFT・推論と同じ system プロンプトを付与し、学習/推論の書式ズレを解消。
        # DPO の参照モデル(SFT)ログ確率も同一書式で計算され、KL アンカーが正しく機能する。
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

    def __getitem__(self, idx):
        row = self.dataset[idx]
        user_text = str(row['instruction']).strip()
        if 'input' in row and isinstance(row['input'], str) and row['input'].strip():
            user_text += "\n" + row['input'].strip()
            
        chosen = str(row['output']).strip()
        rejected = str(row['ng_translation']).strip()
        
        x_c, y_c = self._format_and_pad(user_text, chosen)
        x_r, y_r = self._format_and_pad(user_text, rejected)
        
        return x_c, y_c, x_r, y_r

def get_dpo_dataloader(seq_len: int, batch_size: int):
    dataset = HFDPODataset(seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
