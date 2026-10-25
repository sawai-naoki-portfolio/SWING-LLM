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
from swi_llm.data.prm import get_prm_dataloaders
from swi_llm.utils.constants import CKPT_PRM, CKPT_PRM_LATEST, CKPT_SFT, EXT_PT

logger = setup_logger("prm_train")

def train_prm():
    ckpt_path = f"{CKPT_PRM}{config['ckpt_suffix']}{EXT_PT}"
    latest_path = f"{CKPT_PRM_LATEST}{config['ckpt_suffix']}{EXT_PT}"
    sft_path = f"{CKPT_SFT}{config['ckpt_suffix']}{EXT_PT}"
    
    trainer = BaseTrainer("prm", ckpt_path, fallback_ckpt_paths=[latest_path, sft_path])
    
    train_loader, val_loader = get_prm_dataloaders(
        seq_len=trainer.seq_len, 
        batch_size=trainer.batch_size, 
        max_samples=config.get("prm_data_max_samples")
    )
    
    if len(train_loader) == 0:
        logger.warning("No PRM data available. Exiting.")
        return

    trainer.setup_scheduler(len(train_loader))
    trainer.load_checkpoints()

    # データ変更が「完了扱い」で握り潰されるのを検知する（trainer_utils 参照）。
    if trainer.is_training_complete(data_size=len(train_loader.dataset)):
        return
        
    save_interval_steps = trainer.get_save_interval(len(train_loader))

    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.model.train()
        pbar = tqdm(train_loader, desc=f"PRM Epoch {epoch+1}/{trainer.epochs}")
        trainer.optimizer.zero_grad(set_to_none=True)
        
        for step, (x, y_text, rewards) in enumerate(pbar):
            if config["max_train_steps"] is not None and step >= config["max_train_steps"]: break
            if epoch == trainer.start_epoch and step < trainer.start_step: continue
                
            x, y_text, rewards = x.to(trainer.device), y_text.to(trainer.device), rewards.to(trainer.device)
            
            with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                # return_prm=True to get prm_logits
                logits, aux_loss, prm_logits = trainer.model(x, return_prm=True)
                
                # Language modeling loss (optional, usually good to keep)
                lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y_text.view(-1), ignore_index=-100)
                
                # PRM Loss: Mean Squared Error on the predicted reward vs target reward
                # Ignore tokens where reward is -100
                prm_mask = (rewards != -100)
                if prm_mask.any():
                    prm_loss = F.mse_loss(prm_logits[prm_mask].float(), rewards[prm_mask].float())
                else:
                    prm_loss = torch.tensor(0.0, device=trainer.device)
                
                total_loss = prm_loss + 0.1 * lm_loss + 0.01 * aux_loss
                total_loss = total_loss / trainer.grad_accum_steps
                
            trainer.step_and_save(total_loss, step, len(train_loader), epoch, save_interval_steps, save_path=latest_path)
            
            current_lr = trainer.scheduler.get_last_lr()[0]
            pbar.set_postfix({'prm_loss': f"{(total_loss.item() * trainer.grad_accum_steps):.4f}", 'lr': f"{current_lr:.2e}"})
            
        # Validation
        trainer.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, (x_val, y_val, r_val) in enumerate(val_loader):
                if config["max_val_steps"] is not None and val_step >= config["max_val_steps"]: break
                x_val, y_val, r_val = x_val.to(trainer.device), y_val.to(trainer.device), r_val.to(trainer.device)
                with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                    logits, aux_loss, prm_logits = trainer.model(x_val, use_cache=False, return_prm=True)
                    lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y_val.view(-1), ignore_index=-100)
                    
                    prm_mask = (r_val != -100)
                    if prm_mask.any():
                        prm_loss = F.mse_loss(prm_logits[prm_mask].float(), r_val[prm_mask].float())
                    else:
                        prm_loss = torch.tensor(0.0, device=trainer.device)
                        
                    val_loss += (prm_loss + 0.1 * lm_loss + 0.01 * aux_loss).item()
        
        actual_val_steps = min(len(val_loader), config["max_val_steps"]) if config["max_val_steps"] is not None else len(val_loader)
        avg_val_loss = val_loss / actual_val_steps if actual_val_steps > 0 else 1
        logger.info(f"Epoch {epoch+1} completed. Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < trainer.best_val_loss:
            trainer.best_val_loss = avg_val_loss
            from swi_llm.utils.trainer_utils import save_checkpoint
            save_checkpoint(trainer.main_ckpt_path, trainer.model.state_dict(), trainer.optimizer, trainer.scheduler, trainer.scaler, epoch + 1, -1, trainer.best_val_loss, data_size=trainer.data_size)
            
        trainer.start_step = 0
            
    logger.info("PRM Training complete.")

if __name__ == "__main__":
    try:
        train_prm()
    except Exception as e:
        logger.critical("A fatal error occurred during PRM training.", exc_info=True)
        raise
