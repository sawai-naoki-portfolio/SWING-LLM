"""
swi_llm/training/continual_train.py

LoRA による継続微調整のエントリポイント（ブループリント §11）。

フロー:
  1. 最新のベースチェックポイントをロードし、LoRA アダプタを適用（ベース凍結）。
  2. 蓄積フィードバック（高評価例）＋リプレイバッファ（過去高品質データ）で
     継続学習（破局的忘却を抑制）。
  3. LoRA アダプタのみを保存（検証後に merge_all_lora でベースへマージ可能）。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m swi_llm.training.continual_train
    .venv\\Scripts\\python.exe -m swi_llm.training.continual_train --steps 50 --rank 8

フィードバックが無い場合は SFT データの一部を新規データとして代用する（デモ用）。
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.config import config, create_mini_llm, TRAINING_CONFIG
from swi_llm.utils.constants import CKPT_GRPO, CKPT_DPO, CKPT_SFT, CKPT_PRETRAIN, EXT_PT
from swi_llm.data.tokenizer import get_tokenizer, VOCAB_SIZE, EOT_TOKEN_ID, PAD_TOKEN_ID
from swi_llm.training.lora import apply_lora, lora_state_dict, count_trainable_parameters
from swi_llm.training.continual import FeedbackStore, ReplayBuffer

logger = setup_logger("continual_train")

_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STAGE_PRIORITY = [CKPT_GRPO, CKPT_DPO, CKPT_SFT, CKPT_PRETRAIN]


def _find_base_checkpoint():
    suffix = config.get("ckpt_suffix", "")
    for s in [suffix, "", "_SMOKE", "_TEST", "_MAX"]:
        for base in _STAGE_PRIORITY:
            path = os.path.join(_PROTOTYPE_DIR, f"{base}{s}{EXT_PT}")
            if os.path.exists(path):
                return path
    return None


def _seed_texts(max_items):
    """新規学習テキストを用意する。フィードバックが無ければ SFT データで代用。"""
    fb = FeedbackStore().positive_examples()
    if fb:
        logger.info(f"Using {len(fb)} positive feedback examples as new data.")
        return [f"{r['query']}\n{r['response']}" for r in fb[:max_items]]

    logger.info("No feedback found. Using SFT samples as demo new-data.")
    from swi_llm.data.sft import HFSFTDataset
    ds = HFSFTDataset(seq_len=256, max_samples=max_items)
    return [f"{d['user']}\n{d['assistant']}" for d in ds.dataset[:max_items]]


def continual_finetune(steps=20, rank=8, alpha=16, lr=1e-4, replay_ratio=0.25, max_items=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    seq_len = TRAINING_CONFIG["sft"]["seq_len"]

    model = create_mini_llm(vocab_size=VOCAB_SIZE, max_seq_len=seq_len).to(device)
    ckpt_path = _find_base_checkpoint()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        sd = {k: v for k, v in sd.items() if k not in ("freqs_cos", "freqs_sin")}
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded base checkpoint: {ckpt_path}")
        del ckpt, sd
    else:
        logger.warning("No base checkpoint found; continually adapting a random model (demo).")

    # LoRA 適用（ベース凍結）
    apply_lora(model, rank=rank, alpha=alpha)
    model.to(device)
    trainable, total = count_trainable_parameters(model)
    logger.info(f"Trainable (LoRA) params: {trainable:,} / {total:,} "
                f"({100.0 * trainable / total:.3f}%)")

    # 新規データ + リプレイ（破局的忘却対策）
    new_texts = _seed_texts(max_items)
    replay = ReplayBuffer()
    replay.extend(new_texts)  # デモ: 同じ集合を過去データとしても使う
    train_texts = replay.mix(new_texts, replay_ratio=replay_ratio)
    if not train_texts:
        logger.warning("No training texts available. Aborting continual finetune.")
        return None

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()

    def _batch(text):
        tokens = tokenizer.encode(text)[: seq_len + 1]
        if len(tokens) < 2:
            tokens = tokens + [EOT_TOKEN_ID]
        if len(tokens) < seq_len + 1:
            tokens = tokens + [PAD_TOKEN_ID] * (seq_len + 1 - len(tokens))
        x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
        y[y == PAD_TOKEN_ID] = -100
        return x, y

    step = 0
    for _ in range(steps):
        text = train_texts[step % len(train_texts)]
        x, y = _batch(text)
        logits, aux = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        loss = loss + 0.01 * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 5 == 0:
            logger.info(f"Continual step {step}/{steps} loss={loss.item():.4f}")
        step += 1

    # LoRA アダプタのみ保存
    suffix = config.get("ckpt_suffix", "")
    out_path = os.path.join(_PROTOTYPE_DIR, f"lora_adapter{suffix}{EXT_PT}")
    torch.save(lora_state_dict(model), out_path)
    logger.info(f"Saved LoRA adapter to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="LoRA continual fine-tuning (§11).")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    args = parser.parse_args()
    try:
        continual_finetune(steps=args.steps, rank=args.rank, alpha=args.alpha,
                           lr=args.lr, replay_ratio=args.replay_ratio)
    except Exception:
        logger.exception("Continual fine-tuning failed.")
        raise


if __name__ == "__main__":
    main()
