import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F
from tqdm import tqdm

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from swi_llm.utils.config import config
from swi_llm.utils.trainer_utils import BaseTrainer
from swi_llm.utils.logger import setup_logger
from swi_llm.data.pretrain import get_text_dataloaders
from swi_llm.utils.constants import CKPT_PRETRAIN, EXT_PT

logger = setup_logger("train")

def train():
    ckpt_path = f"{CKPT_PRETRAIN}{config['ckpt_suffix']}{EXT_PT}"
    trainer = BaseTrainer("pretrain", ckpt_path)
    
    train_loader, val_loader = get_text_dataloaders(seq_len=trainer.seq_len, batch_size=trainer.batch_size)
    trainer.setup_scheduler(len(train_loader))
    trainer.load_checkpoints()
    
    # データ変更が「完了扱い」で握り潰されるのを検知する（trainer_utils 参照）。
    if trainer.is_training_complete(data_size=len(train_loader.dataset)):
        return
            
    save_interval_steps = trainer.get_save_interval(len(train_loader))
            
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{trainer.epochs}")
        
        for step, (x, y) in enumerate(pbar):
            if config["max_train_steps"] is not None and step >= config["max_train_steps"]: break
            if epoch == trainer.start_epoch and step < trainer.start_step: continue
                
            x, y = x.to(trainer.device), y.to(trainer.device)
            
            with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                logits, aux_loss, mtp_logits = trainer.model(x, return_mtp=True)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
            total_loss = loss + 0.01 * aux_loss
            
            if mtp_logits is not None:
                mtp_loss = 0.0
                for i, mtp_logit in enumerate(mtp_logits):
                    shift = i + 1
                    if y.shape[1] > shift:
                        shifted_y = y[:, shift:]
                        shifted_logits = mtp_logit[:, :-shift]
                        mtp_loss += F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.size(-1)).float(), shifted_y.reshape(-1))
                total_loss += 0.3 * mtp_loss
                
            total_loss = total_loss / trainer.grad_accum_steps
            trainer.step_and_save(total_loss, step, len(train_loader), epoch, save_interval_steps)
            
            current_lr = trainer.scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f"{(total_loss.item() * trainer.grad_accum_steps):.4f}", 'lr': f"{current_lr:.2e}"})
            
        # Validation
        trainer.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, (x_val, y_val) in enumerate(tqdm(val_loader, desc="Validation")):
                if config["max_val_steps"] is not None and val_step >= config["max_val_steps"]: break
                x_val, y_val = x_val.to(trainer.device), y_val.to(trainer.device)
                with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                    logits, _, _ = trainer.model(x_val, return_mtp=True)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y_val.view(-1))
                val_loss += loss.item()
        
        actual_val_steps = min(len(val_loader), config["max_val_steps"]) if config["max_val_steps"] is not None else len(val_loader)
        avg_val_loss = val_loss / actual_val_steps if actual_val_steps > 0 else float('inf')
        val_ppl = math.exp(min(avg_val_loss, 700.0))
        logger.info(f"Epoch {epoch+1} completed. Validation Loss: {avg_val_loss:.4f} (perplexity {val_ppl:.2f})")
        # Phase 2: 成長の推移を JSONL に残す（errorLog/metrics/pretrain_<MODE>.jsonl, kind=epoch）。
        trainer.metrics.log_epoch(epoch=epoch + 1, loss=avg_val_loss, val_ppl=round(val_ppl, 4))
        
        if avg_val_loss < trainer.best_val_loss:
            trainer.best_val_loss = avg_val_loss
            from swi_llm.utils.trainer_utils import save_checkpoint
            save_checkpoint(trainer.main_ckpt_path, trainer.model.state_dict(), trainer.optimizer, trainer.scheduler, trainer.scaler, epoch + 1, -1, trainer.best_val_loss, data_size=trainer.data_size)
            
        trainer.start_step = 0

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.critical("A fatal error occurred during pretraining.", exc_info=True)
        raise
