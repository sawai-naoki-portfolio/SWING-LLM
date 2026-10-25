import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from swi_llm.utils.config import config
from swi_llm.utils.trainer_utils import BaseTrainer
from swi_llm.utils.logger import setup_logger
from swi_llm.data.dpo import HFDPODataset
from swi_llm.utils.constants import CKPT_DPO, CKPT_DPO_LATEST, CKPT_SFT, CKPT_SFT_LATEST, EXT_PT

logger = setup_logger("dpo_train")

def get_logprobs(logits, labels):
    loss_mask = (labels != -100)
    labels_cloned = labels.clone()
    labels_cloned[~loss_mask] = 0
    
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gathered_log_probs = torch.gather(log_probs, 2, labels_cloned.unsqueeze(2)).squeeze(2)
    return (gathered_log_probs * loss_mask).sum(dim=1)

def train_dpo():
    ckpt_path = f"{CKPT_DPO}{config['ckpt_suffix']}{EXT_PT}"
    latest_path = f"{CKPT_DPO_LATEST}{config['ckpt_suffix']}{EXT_PT}"
    sft_ckpt_path = f"{CKPT_SFT}{config['ckpt_suffix']}{EXT_PT}"
    sft_latest_path = f"{CKPT_SFT_LATEST}{config['ckpt_suffix']}{EXT_PT}"
    
    trainer = BaseTrainer("dpo", ckpt_path, fallback_ckpt_paths=[latest_path, sft_ckpt_path, sft_latest_path])
    beta = trainer.cfg["beta"]
    
    dataset = HFDPODataset(seq_len=trainer.seq_len, max_samples=config["dpo_data_max_samples"])
    dataloader = DataLoader(dataset, batch_size=trainer.batch_size, shuffle=True, drop_last=True)
    
    trainer.setup_scheduler(len(dataloader))
    trainer.load_checkpoints()

    # データ変更が「完了扱い」で握り潰されるのを検知する（trainer_utils 参照）。
    if trainer.is_training_complete(data_size=len(dataloader.dataset)):
        return

    # 参照モデルは常に SFT（DPO の開始点）で固定する (#3)。deepcopy(policy) のままだと DPO
    # 再開時に policy が進んだ重みになり、KL アンカーが本来の SFT からドリフトして DPO 目的が
    # 崩れる。SFT チェックポイントを独立にロードして固定する（無ければ現在重みで代替）。
    from swi_llm.utils.trainer_utils import load_checkpoint
    ref_model = copy.deepcopy(trainer.model)
    ref_loaded = False
    for ref_src in [sft_ckpt_path, sft_latest_path]:
        if os.path.exists(ref_src):
            _, _, _, ck = load_checkpoint(ref_src, trainer.device, ref_model)
            if ck is not None:
                logger.info(f"DPO reference model fixed to SFT checkpoint: {ref_src}")
                ref_loaded = True
                break
    if not ref_loaded:
        logger.warning("SFT checkpoint not found/loadable; DPO reference falls back to current policy weights.")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    
    save_interval_steps = trainer.get_save_interval(len(dataloader))
        
    trainer.model.train()
    
    for epoch in range(trainer.start_epoch, trainer.epochs):
        pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch+1}/{trainer.epochs}")
        trainer.optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(pbar):
            if config["max_train_steps"] is not None and step >= config["max_train_steps"]: break
            if epoch == trainer.start_epoch and step < trainer.start_step: continue
                
            x_c, y_c, x_r, y_r = [tensor.to(trainer.device) for tensor in batch]
            
            with torch.no_grad():
                with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                    ref_logits_c, _ = ref_model(x_c)
                    ref_logits_r, _ = ref_model(x_r)
                    
                    ref_chosen_logps = get_logprobs(ref_logits_c, y_c)
                    ref_rejected_logps = get_logprobs(ref_logits_r, y_r)
                
            with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                policy_logits_c, policy_aux_c = trainer.model(x_c)
                policy_logits_r, policy_aux_r = trainer.model(x_r)
                
                policy_chosen_logps = get_logprobs(policy_logits_c, y_c)
                policy_rejected_logps = get_logprobs(policy_logits_r, y_r)
                
                chosen_ratio = policy_chosen_logps - ref_chosen_logps
                rejected_ratio = policy_rejected_logps - ref_rejected_logps
                
                logits = chosen_ratio - rejected_ratio
                loss = -F.logsigmoid(beta * logits).mean()
                
                total_aux_loss = policy_aux_c + policy_aux_r
                total_loss = (loss + 0.01 * total_aux_loss) / trainer.grad_accum_steps
                
            trainer.step_and_save(total_loss, step, len(dataloader), epoch, save_interval_steps, save_path=latest_path)
            pbar.set_postfix({"loss": f"{loss.item() * trainer.grad_accum_steps:.4f}", "lr": f"{trainer.scheduler.get_last_lr()[0]:.2e}"})
            
        trainer.start_step = 0
            
    logger.info("DPO Training complete.")
    from swi_llm.utils.trainer_utils import save_checkpoint
    save_checkpoint(ckpt_path, trainer.model.state_dict(), trainer.optimizer, trainer.scheduler, trainer.scaler, trainer.epochs, -1, trainer.best_val_loss, data_size=trainer.data_size)

if __name__ == "__main__":
    try:
        train_dpo()
    except Exception as e:
        logger.critical("A fatal error occurred during DPO training.", exc_info=True)
        raise
