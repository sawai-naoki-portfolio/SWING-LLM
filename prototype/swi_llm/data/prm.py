import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from swi_llm.data.tokenizer import get_tokenizer
from swi_llm.data.prm_mc import contrastive_step_rewards
from swi_llm.utils.logger import setup_logger
from swi_llm.utils.chat_template import build_prompt, THINK_END, IM_END
from swi_llm.utils.constants import PRM_USE_CONTRASTIVE_LABELS, TRAINING_SYSTEM_PROMPT

logger = setup_logger("data.prm")

class PRMDataset(Dataset):
    """
    PRM (Process Reward Model) Dataset.
    This dataset should contain step-by-step reasoning paths and a reward for each step.
    For this implementation, we will simulate this by using math data or generic text,
    splitting it by newlines, and appending a <step> token.
    We will assign a generic reward to each step (e.g., 1.0 if it's from a correct solution).
    """
    def __init__(self, seq_len=512, max_samples=None):
        self.seq_len = seq_len
        self.tokenizer = get_tokenizer()
        self.samples = []
        
        # We need a special step token. We'll use a placeholder or define one.
        # Ideally, this should be added to the tokenizer, but we can reuse a rare token
        # or assume the tokenizer has a way to represent it. Let's use <|step|>
        self.step_token = "<|step|>"
        
        from swi_llm.data.dpo import HFDPODataset
        base_dataset = HFDPODataset(seq_len=seq_len, max_samples=max_samples)
        
        logger.info("Formatting PRM samples (Monte-Carlo / contrastive step rewards, Q-07)...")
        for i in range(len(base_dataset.dataset)):
            row = base_dataset.dataset[i]
            prompt = str(row['instruction']).strip()

            # 正解軌跡（chosen）は MC 整合で全ステップ 1.0
            self._add_trajectory(prompt, str(row.get('output', '')).strip(), correct=True)

            # 不正解軌跡（rejected）を減衰報酬で加え、良い/悪いステップを識別させる (Q-07)。
            # uniform 1.0 固定と異なり、PRM に「どこで推論が崩れたか」の信号を与える。
            if PRM_USE_CONTRASTIVE_LABELS:
                self._add_trajectory(prompt, str(row.get('ng_translation', '')).strip(), correct=False)

            if max_samples and len(self.samples) >= max_samples:
                break

        logger.info(f"PRM Dataset loaded {len(self.samples)} samples.")

    def _add_trajectory(self, prompt, answer, correct):
        """1 軌跡を「ステップ分割 + ステップ別報酬」に整形して samples へ追加する。

        correct=True は正解軌跡（報酬 1.0）、False は不正解軌跡（後半へ減衰）。
        報酬は prm_mc.contrastive_step_rewards が算出する（MC 推定の軽量代替）。
        """
        if not answer:
            return
        steps = [s.strip() for s in answer.split('\n') if s.strip()]
        if not steps:
            return

        step_rewards = contrastive_step_rewards(len(steps), correct=correct)

        # S6: SFT・推論と同じ system プロンプトを付与し、PRM が採点する軌跡の書式を揃える。
        formatted_text = build_prompt(prompt, system_text=TRAINING_SYSTEM_PROMPT, cot=True)
        for step in steps:
            formatted_text += step + f" {self.step_token}\n"
        formatted_text += f"{THINK_END}\n{IM_END}"

        self.samples.append({"text": formatted_text, "step_rewards": step_rewards})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        step_rewards = sample["step_rewards"]
        
        # We tokenize the text. We also need to find the indices of the step token.
        # Since we might not have a real <|step|> token in the vocab, we can find it by string matching 
        # before encoding, or we encode it and find its ID.
        # Let's assume the tokenizer handles it or we find it by ID.
        # A simpler way: just encode and then we will manually align.
        # For simplicity, let's just train the model to predict the reward at the end of every token,
        # but we only calculate loss on the tokens that correspond to the end of a step.
        
        tokens = self.tokenizer.encode(text)
        from swi_llm.data.tokenizer import EOT_TOKEN_ID
        
        if len(tokens) > self.seq_len + 1:
            tokens = tokens[:self.seq_len + 1]
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y_text = torch.tensor(tokens[1:], dtype=torch.long) # Next token prediction
        
        # Create a reward tensor of same length, default to -100 (ignore)
        rewards = torch.full((len(tokens) - 1,), -100.0, dtype=torch.float)
        
        # Find newline tokens (as a proxy for step ends if we don't have a strict step token)
        newline_id = self.tokenizer.encode("\n")[-1]
        
        step_idx = 0
        for i in range(len(tokens) - 1):
            if tokens[i+1] == newline_id and step_idx < len(step_rewards):
                rewards[i] = step_rewards[step_idx]
                step_idx += 1
                
        # Pad to seq_len
        if len(tokens) < self.seq_len + 1:
            pad_len = self.seq_len + 1 - len(tokens)
            x = torch.cat([x, torch.tensor([EOT_TOKEN_ID] * pad_len, dtype=torch.long)])
            y_text = torch.cat([y_text, torch.tensor([-100] * pad_len, dtype=torch.long)])
            rewards = torch.cat([rewards, torch.tensor([-100.0] * pad_len, dtype=torch.float)])
            
        return x, y_text, rewards

def get_prm_dataloaders(seq_len=512, batch_size=2, max_samples=None):
    dataset = PRMDataset(seq_len=seq_len, max_samples=max_samples)
    
    if len(dataset) == 0:
        return [], []
        
    train_size = max(1, int(0.9 * len(dataset)))
    val_size = len(dataset) - train_size
    
    if val_size == 0:
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True if len(train_dataset) > batch_size else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
