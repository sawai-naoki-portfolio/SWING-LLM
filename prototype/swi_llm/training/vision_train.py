import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
torch.backends.cudnn.enabled = False  # RTX 5070 (Blackwell) 等でのConv2dクラッシュを回避
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from swi_llm.utils.config import config, create_mini_llm
from swi_llm.utils.trainer_utils import BaseTrainer
from swi_llm.utils.logger import setup_logger
from swi_llm.core.models import SWIVisionLanguageModel
from swi_llm.core.vision import VisionEncoder, VisionProjector
from swi_llm.data.vision_data import get_vision_dataloaders
from swi_llm.data.tokenizer import VOCAB_SIZE
from swi_llm.utils.constants import CKPT_SFT, CKPT_VISION, EXT_PT

logger = setup_logger("vision_train")

def train_vision():
    text_ckpt_path = f"{CKPT_SFT}{config['ckpt_suffix']}{EXT_PT}"
    ckpt_path = f"{CKPT_VISION}{config['ckpt_suffix']}{EXT_PT}"
    
    def model_fn():
        cfg_seq_len = config.get("vision_seq_len", 256)
        text_model = create_mini_llm(vocab_size=VOCAB_SIZE, max_seq_len=cfg_seq_len)
        vision_encoder = VisionEncoder(image_size=224, patch_size=16, embed_dim=768, depth=12)
        vision_projector = VisionProjector(vision_dim=768, llm_dim=768)
        return SWIVisionLanguageModel(text_model, vision_encoder, vision_projector)

    trainer = BaseTrainer("vision", ckpt_path, model_fn=model_fn)
    
    # Load base text model weights manually before vision ckpt
    if os.path.exists(text_ckpt_path):
        logger.info(f"Loading pretrained text model from {text_ckpt_path}...")
        try:
            ckpt = torch.load(text_ckpt_path, map_location=trainer.device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = {k: v for k, v in ckpt['model_state_dict'].items() if not k.endswith("freqs_cos") and not k.endswith("freqs_sin")}
                trainer.model.text_model.load_state_dict(state_dict, strict=False)
            else:
                state_dict = {k: v for k, v in ckpt.items() if not k.endswith("freqs_cos") and not k.endswith("freqs_sin")}
                trainer.model.text_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {text_ckpt_path}: {e}")
            
    for param in trainer.model.text_model.parameters():
        param.requires_grad = False
    for param in trainer.model.vision_encoder.parameters():
        param.requires_grad = True
    for param in trainer.model.vision_projector.parameters():
        param.requires_grad = True
        
    trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]
    trainer.optimizer = optim.AdamW(trainable_params, lr=trainer.lr)
    
    train_loader, val_loader = get_vision_dataloaders(
        seq_len=trainer.seq_len, 
        batch_size=trainer.batch_size, 
        max_samples=config.get("sft_data_max_samples")
    )
    
    trainer.setup_scheduler(len(train_loader))
    trainer.load_checkpoints(disable_pretrain_fallback=True)

    # データ変更が「完了扱い」で握り潰されるのを検知する（trainer_utils 参照）。
    if trainer.is_training_complete(data_size=len(train_loader.dataset)):
        return
        
    num_vision_tokens = (224 // 16) ** 2 + 1 
    save_interval_steps = trainer.get_save_interval(len(train_loader))
            
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.model.train()
        trainer.model.text_model.eval()
        trainer.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{trainer.epochs}")
        
        for step, (images, x, y) in enumerate(pbar):
            if config["max_train_steps"] is not None and step >= config["max_train_steps"]: break
            if epoch == trainer.start_epoch and step < trainer.start_step: continue
                
            images = images.to(trainer.device)
            x, y = x.to(trainer.device), y.to(trainer.device)
            
            with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                logits, aux_loss = trainer.model(x, images=images)
                text_logits = logits[:, num_vision_tokens:, :]
                
                loss = F.cross_entropy(text_logits.reshape(-1, text_logits.size(-1)).float(), y.view(-1))
                total_loss = loss / trainer.grad_accum_steps
                
            trainer.scaler.scale(total_loss).backward()
            
            if (step + 1) % trainer.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                trainer.scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
                trainer.scheduler.step()
                trainer.optimizer.zero_grad(set_to_none=True)
                
                opt_step = (step + 1) // trainer.grad_accum_steps
                if opt_step > 0 and opt_step % save_interval_steps == 0:
                    state_dict = {
                        'vision_encoder': trainer.model.vision_encoder.state_dict(),
                        'vision_projector': trainer.model.vision_projector.state_dict()
                    }
                    from swi_llm.utils.trainer_utils import save_checkpoint
                    save_checkpoint(ckpt_path, state_dict, trainer.optimizer, trainer.scheduler, trainer.scaler, epoch, step, trainer.best_val_loss, data_size=trainer.data_size)
                
            current_lr = trainer.scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f"{(total_loss.item() * trainer.grad_accum_steps):.4f}", 'lr': f"{current_lr:.2e}"})
            
        # Validation
        trainer.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_step, (images_val, x_val, y_val) in enumerate(tqdm(val_loader, desc="Validation")):
                if config["max_val_steps"] is not None and val_step >= config["max_val_steps"]: break
                images_val = images_val.to(trainer.device)
                x_val, y_val = x_val.to(trainer.device), y_val.to(trainer.device)
                with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                    logits, _ = trainer.model(x_val, images=images_val)
                    text_logits = logits[:, num_vision_tokens:, :]
                    loss = F.cross_entropy(text_logits.reshape(-1, text_logits.size(-1)).float(), y_val.view(-1))
                val_loss += loss.item()
        
        actual_val_steps = min(len(val_loader), config["max_val_steps"]) if config["max_val_steps"] is not None else len(val_loader)
        avg_val_loss = val_loss / actual_val_steps if actual_val_steps > 0 else 1
        logger.info(f"Epoch {epoch+1} completed. Validation Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < trainer.best_val_loss:
            trainer.best_val_loss = avg_val_loss
            state_dict = {
                'vision_encoder': trainer.model.vision_encoder.state_dict(),
                'vision_projector': trainer.model.vision_projector.state_dict()
            }
            from swi_llm.utils.trainer_utils import save_checkpoint
            save_checkpoint(ckpt_path, state_dict, trainer.optimizer, trainer.scheduler, trainer.scaler, epoch + 1, -1, trainer.best_val_loss, data_size=trainer.data_size)
            
        trainer.start_step = 0

if __name__ == "__main__":
    try:
        train_vision()
    except Exception as e:
        logger.critical("A fatal error occurred during vision training.", exc_info=True)
        raise
