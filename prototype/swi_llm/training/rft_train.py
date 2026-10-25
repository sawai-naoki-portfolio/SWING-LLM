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
from swi_llm.data.rft import get_rft_dataloaders
from swi_llm.utils.constants import CKPT_RFT, CKPT_RFT_LATEST, CKPT_SFT, CKPT_SFT_LATEST, EXT_PT

logger = setup_logger("rft_train")

def train_rft():
    ckpt_path = f"{CKPT_RFT}{config['ckpt_suffix']}{EXT_PT}"
    latest_path = f"{CKPT_RFT_LATEST}{config['ckpt_suffix']}{EXT_PT}"
    sft_path = f"{CKPT_SFT}{config['ckpt_suffix']}{EXT_PT}"
    sft_latest_path = f"{CKPT_SFT_LATEST}{config['ckpt_suffix']}{EXT_PT}"
    
    trainer = BaseTrainer("rft", ckpt_path, fallback_ckpt_paths=[latest_path, sft_path, sft_latest_path])
    
    train_loader, val_loader = get_rft_dataloaders(
        seq_len=trainer.seq_len, 
        batch_size=trainer.batch_size, 
        max_samples=config.get("rft_data_max_samples")
    )
    
    if len(train_loader) == 0:
        logger.warning("No RFT data available. Exiting.")
        return

    trainer.setup_scheduler(len(train_loader))
    trainer.load_checkpoints()

    # データ変更が「完了扱い」で握り潰されるのを検知する（trainer_utils 参照）。
    if trainer.is_training_complete(data_size=len(train_loader.dataset)):
        return
        
    save_interval_steps = trainer.get_save_interval(len(train_loader))

    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.model.train()
        pbar = tqdm(train_loader, desc=f"RFT Epoch {epoch+1}/{trainer.epochs}")
        trainer.optimizer.zero_grad(set_to_none=True)
        
        for step, (x, y) in enumerate(pbar):
            if config["max_train_steps"] is not None and step >= config["max_train_steps"]: break
            if epoch == trainer.start_epoch and step < trainer.start_step: continue
                
            x, y = x.to(trainer.device), y.to(trainer.device)
            
            with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                logits, aux_loss, mtp_logits = trainer.model(x, return_mtp=True)
                logits_flat = logits.view(-1, logits.size(-1)).float()
                y_flat = y.view(-1)
                
                loss = F.cross_entropy(logits_flat, y_flat, ignore_index=-100)
                total_loss = loss + 0.01 * aux_loss
                
                if mtp_logits is not None:
                    mtp_loss = 0.0
                    for i, mtp_logit in enumerate(mtp_logits):
                        shift = i + 1
                        if y.shape[1] > shift:
                            shifted_y = y[:, shift:]
                            shifted_logits = mtp_logit[:, :-shift]
                            mtp_loss += F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.size(-1)).float(), shifted_y.reshape(-1), ignore_index=-100)
                    total_loss += 0.3 * mtp_loss
                    
                total_loss = total_loss / trainer.grad_accum_steps
                
            trainer.step_and_save(total_loss, step, len(train_loader), epoch, save_interval_steps, save_path=latest_path)
            
            current_lr = trainer.scheduler.get_last_lr()[0]
            pbar.set_postfix({'rft_loss': f"{(total_loss.item() * trainer.grad_accum_steps):.4f}", 'lr': f"{current_lr:.2e}"})
            
        # Validation
        trainer.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, (x_val, y_val) in enumerate(val_loader):
                if config["max_val_steps"] is not None and val_step >= config["max_val_steps"]: break
                x_val, y_val = x_val.to(trainer.device), y_val.to(trainer.device)
                with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                    logits, aux_loss, _ = trainer.model(x_val, use_cache=False, return_mtp=True)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y_val.view(-1), ignore_index=-100)
                    val_loss += (loss + 0.01 * aux_loss).item()
        
        actual_val_steps = min(len(val_loader), config["max_val_steps"]) if config["max_val_steps"] is not None else len(val_loader)
        avg_val_loss = val_loss / actual_val_steps if actual_val_steps > 0 else 1
        logger.info(f"Epoch {epoch+1} completed. Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < trainer.best_val_loss:
            trainer.best_val_loss = avg_val_loss
            from swi_llm.utils.trainer_utils import save_checkpoint
            save_checkpoint(trainer.main_ckpt_path, trainer.model.state_dict(), trainer.optimizer, trainer.scheduler, trainer.scaler, epoch + 1, -1, trainer.best_val_loss, data_size=trainer.data_size)
            
        trainer.start_step = 0
            
    logger.info("RFT Training complete.")

if __name__ == "__main__":
    try:
        train_rft()
    except Exception as e:
        logger.critical("A fatal error occurred during RFT training.", exc_info=True)
        raise
