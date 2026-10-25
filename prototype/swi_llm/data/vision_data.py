import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer, EOT_TOKEN_ID, PAD_TOKEN_ID
from swi_llm.utils.constants import DATASET_CIFAR10
from swi_llm.utils.chat_template import build_prompt, format_assistant_response

# 画像キャプション学習用の固定プロンプト
VISION_CAPTION_PROMPT = "この画像には何が写っていますか？"

logger = setup_logger("data.vision")

class HFVisionTextDataset(Dataset):
    def __init__(self, seq_len: int, split='train', max_samples=None):
        self.seq_len = seq_len
        self.tokenizer = get_tokenizer()
        logger.info(f"Loading {DATASET_CIFAR10} dataset for vision training...")
        try:
            self.dataset = load_dataset(DATASET_CIFAR10, split=split)
            self.classes = self.dataset.features['label'].names
        except Exception as e:
            logger.exception(f"Failed to load {DATASET_CIFAR10} dataset.")
            raise
        
        if max_samples and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
            
        from swi_llm.utils.config import VISION_CONFIG
        # 01(小): リサイズ方針は VISION_CONFIG["resize_policy"] を単一情報源として
        # 推論側（core/vision.preprocess_pil_image）と一致させる。squash（既定）は従来挙動。
        size = VISION_CONFIG["image_size"]
        if VISION_CONFIG.get("resize_policy", "squash") == "shortside_crop":
            resize_ops = [transforms.Resize(min(size)), transforms.CenterCrop(size)]
        else:
            resize_ops = [transforms.Resize(size)]
        self.transform = transforms.Compose(resize_ops + [
            transforms.ToTensor(),
            transforms.Normalize(mean=VISION_CONFIG["image_mean"],
                                 std=VISION_CONFIG["image_std"])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row['img']
        text = self.classes[row['label']]
        
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        image_tensor = self.transform(image)
        
        prompt_str = build_prompt(VISION_CAPTION_PROMPT)
        response_str = format_assistant_response(f"これは {text} です。")
            
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
        return image_tensor, x, y

def get_vision_dataloaders(seq_len: int, batch_size: int, val_split: float = 0.1, max_samples: int = None):
    dataset = HFVisionTextDataset(seq_len, max_samples=max_samples)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
