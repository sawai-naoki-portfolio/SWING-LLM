"""
swi_llm/data/pretrain.py

事前学習用データセット。data_mix.py で構築した複数ソース混合コーパス
（Wikipedia / arxiv / dolly）をトークン列へ平坦化して供給する（ブループリント §1.5）。

カリキュラム学習 (PRETRAIN_CURRICULUM):
  - train ドキュメントは data_mix 側で「短文→長文 + 末尾に高品質データ」に整列済み。
  - 本データセットはその順序を保つため、カリキュラム時は窓をシーケンシャルに
    取り出す（DataLoader も shuffle=False）。無効時は従来通りランダム窓 + shuffle。
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer, EOT_TOKEN_ID
from swi_llm.data.data_mix import build_mixed_documents
from swi_llm.utils.config import config
from swi_llm.utils.constants import PRETRAIN_CURRICULUM

logger = setup_logger("data.pretrain")


def _tokenize_documents(docs, tokenizer):
    """各ドキュメントを個別にトークナイズし EOT で区切って結合する。

    L-7（設計意図の明記）: 複数文書を EOT で連結した 1 本のトークン列から `seq_len` 窓を切り出す
    GPT 系の標準的なパッキングを採用している。このため 1 つの窓が文書境界（EOT）を跨ぐことがあり、
    その場合は文書をまたいで causal attention が及ぶ（EOT 直後トークンも次トークン予測の学習対象）。
    実運用では大きな害は無いが、境界を跨がせたくない場合は intra-document attention マスク
    （document masking）が必要。現状は MiniLLM.forward が text 経路で is_causal=True 固定のため未対応で、
    将来 packing 対応の attention マスクを導入する際の課題として明記する。
    """
    all_tokens = []
    for text in docs:
        text = text.strip()
        if not text:
            continue
        all_tokens.extend(tokenizer.encode(text))
        all_tokens.append(EOT_TOKEN_ID)
    return np.array(all_tokens, dtype=np.int32)


class HFPretrainDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int, curriculum: bool):
        self.tokens = tokens
        self.seq_len = seq_len
        self.curriculum = curriculum

    def __len__(self):
        return max(1, (len(self.tokens) - self.seq_len) // self.seq_len)

    def __getitem__(self, idx):
        if self.curriculum:
            # ドキュメント整列順を保つためシーケンシャルに窓を取り出す
            start_idx = min(idx * self.seq_len, max(0, len(self.tokens) - self.seq_len - 1))
        else:
            start_idx = np.random.randint(0, max(1, len(self.tokens) - self.seq_len - 1))
        # ステップ5: tokens は np.memmap(uint16) の場合がある。torch は uint16 を
        # 直接扱えないため、ここで int64 へ明示コピーしてから Tensor 化する
        # （memmap からの読みはこのスライス分だけがページングされる）。
        chunk = np.asarray(self.tokens[start_idx: start_idx + self.seq_len + 1], dtype=np.int64)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def _token_budget_dataloaders(token_budget, seq_len, batch_size):
    """トークン予算モード（roadmap A-1）: ストリーム構築したコーパスを memmap で供給する。

    文書リスト（build_mixed_documents）は一切作らない。コーパスは「主区間 → アニーリング区間」の
    順に並んでいるので、先頭から順に窓を取り出す（shuffle しない）。
    """
    from swi_llm.data.token_corpus import load_or_build_token_corpus
    train_tokens, val_tokens, meta = load_or_build_token_corpus(token_budget)
    if len(val_tokens) <= seq_len + 1:
        logger.warning("Token corpus has (almost) no validation tokens; validating on the head of train.")
        val_tokens = train_tokens[: seq_len * 4]
    logger.info(f"Pretrain (token budget {token_budget:,}): {len(train_tokens):,} train / "
                f"{len(val_tokens):,} val tokens. Realized mix: {meta.get('realized_mix')}")
    train_dataset = HFPretrainDataset(train_tokens, seq_len, curriculum=True)
    val_dataset = HFPretrainDataset(val_tokens, seq_len, curriculum=True)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False))


def get_text_dataloaders(seq_len: int, batch_size: int):
    token_budget = config.get("pretrain_token_budget")
    if token_budget:
        return _token_budget_dataloaders(token_budget, seq_len, batch_size)

    tokenizer = get_tokenizer()
    max_docs = config.get("pretrain_max_samples")
    curriculum = PRETRAIN_CURRICULUM

    logger.info(f"Building pretrain data mix (max_docs={max_docs}, curriculum={curriculum})...")
    train_docs, val_docs = build_mixed_documents(max_docs, curriculum=curriculum)

    # ステップ5 (fix 03): 事前トークン化キャッシュ（uint16 バイナリ + memmap）。
    # コーパス/トークナイザが変わらない限り 2 回目以降のトークン化を丸ごと省略し、
    # RAM 常駐も int32 全量 → ページング（uint16）へ削減する。SWI_PRETOKENIZE=0 で
    # 従来のオンザフライ経路に戻せる。
    from swi_llm.data.pretokenize import pretokenize_enabled, get_or_build_token_cache
    if pretokenize_enabled():
        train_tokens = get_or_build_token_cache(train_docs, tokenizer, EOT_TOKEN_ID, tag="pretrain_train")
        val_tokens = (get_or_build_token_cache(val_docs, tokenizer, EOT_TOKEN_ID, tag="pretrain_val")
                      if val_docs else train_tokens[: seq_len * 4])
    else:
        train_tokens = _tokenize_documents(train_docs, tokenizer)
        val_tokens = _tokenize_documents(val_docs, tokenizer) if val_docs else train_tokens[: seq_len * 4]
    logger.info(f"Loaded {len(train_tokens)} train / {len(val_tokens)} val tokens.")

    train_dataset = HFPretrainDataset(train_tokens, seq_len, curriculum=curriculum)
    val_dataset = HFPretrainDataset(val_tokens, seq_len, curriculum=False)

    # DEBUG: 事前学習データの構築結果（窓数・シャッフル方針。データ構築1回）
    logger.debug(f"Pretrain data built: train_docs={len(train_docs)}, val_docs={len(val_docs) if val_docs else 0}, "
                 f"train_windows={len(train_dataset)}, val_windows={len(val_dataset)}, seq_len={seq_len}, "
                 f"curriculum={curriculum}, shuffle={not curriculum}")

    # カリキュラム時は短文→長文の提示順を壊さないよう shuffle=False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not curriculum)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
