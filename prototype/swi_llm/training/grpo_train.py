import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import threading
import queue

# Add prototype directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from swi_llm.utils.config import config
from swi_llm.utils.trainer_utils import BaseTrainer
from swi_llm.utils.logger import setup_logger
from swi_llm.data.tokenizer import get_tokenizer
from swi_llm.data.grpo import GRPODataset, RewardVerifier
from swi_llm.core.generation import generate
from swi_llm.utils.constants import CKPT_GRPO, CKPT_GRPO_LATEST, CKPT_DPO, CKPT_SFT, EXT_PT, TRAINING_SYSTEM_PROMPT
from swi_llm.utils.chat_template import build_prompt

logger = setup_logger("grpo_train")

# 問題4 (DAPO): グループ内 advantage の絶対値がこの閾値未満なら「分散ゼロ（全員正解/全員不正解）」
# とみなし、そのグループのポリシー更新をスキップする。
GRPO_DAPO_ADV_EPS = 1e-8

def build_grpo_sequence(tokens, prompt_len, seq_len):
    """1 つの completion から (input_ids, target_ids) を作る。学習できるトークンが無ければ None。

    L-13: 以前は「プロンプトより長いか」を**切り詰め前**に判定してから seq_len で切っていたため、
    prompt_len >= seq_len のとき切り詰め後の target が全て -100（プロンプト扱い）になり、
    学習信号ゼロの系列が表示 loss に -advantage だけを足していた。
    切り詰め → プロンプト長のクランプ → 有効ターゲットの有無、の順で判定する。

    Args:
        tokens: プロンプト + completion のトークン列。
        prompt_len: プロンプト部分のトークン数（target 側でマスクする）。
        seq_len: 系列長の上限。
    Returns:
        (input_ids, target_ids)（list[int]。target のプロンプト部分は -100）または None。
    """
    tokens = list(tokens[:seq_len])
    prompt_len = max(0, min(prompt_len, len(tokens)))
    # 有効ターゲット数 = len(tokens) - max(prompt_len, 1)。completion が 1 トークンも残らなければ捨てる。
    if len(tokens) < 2 or len(tokens) <= prompt_len:
        return None
    input_ids = tokens[:-1]
    target_ids = tokens[1:]
    n_mask = max(prompt_len - 1, 0)
    target_ids = [-100] * n_mask + target_ids[n_mask:]
    return input_ids, target_ids


def get_logprobs(logits, labels):
    loss_mask = (labels != -100)
    labels_cloned = labels.clone()
    labels_cloned[~loss_mask] = 0
    
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gathered_log_probs = torch.gather(log_probs, 2, labels_cloned.unsqueeze(2)).squeeze(2)
    return (gathered_log_probs * loss_mask).sum(dim=1)

class RolloutWorker(threading.Thread):
    def __init__(self, dataloader, tokenizer, group_size, device, verifier):
        super().__init__()
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.device = device
        self.verifier = verifier
        
        self.output_queue = queue.Queue(maxsize=config.get("rollout_queue_size", 2))
        self.sync_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.daemon = True

    def run(self):
        initial_state = self.sync_queue.get()
        if initial_state is None:
            return
            
        from swi_llm.utils.config import create_mini_llm
        from swi_llm.data.tokenizer import VOCAB_SIZE
        
        actor_model = create_mini_llm(vocab_size=VOCAB_SIZE, max_seq_len=512).to(self.device)
        actor_model.load_state_dict(initial_state)
        actor_model.eval()
        
        while not self.stop_event.is_set():
            for questions, expected_answers in self.dataloader:
                if self.stop_event.is_set():
                    return
                    
                try:
                    new_state = self.sync_queue.get_nowait()
                    if new_state is None:
                        return
                    actor_model.load_state_dict(new_state)
                except queue.Empty:
                    pass
                    
                question = questions[0]
                expected_answer = expected_answers[0]
                # S6: SFT・推論と同じ system プロンプトを付与し、ロールアウトを推論条件に揃える。
                prompt = build_prompt(question, system_text=TRAINING_SYSTEM_PROMPT, cot=True)
                
                completions = []
                with torch.no_grad():
                    for _ in range(self.group_size):
                        output = generate(actor_model, self.tokenizer, prompt, max_new_tokens=150, temperature=0.9, top_p=0.95)
                        completions.append(output[len(prompt):])
                        
                rewards = [self.verifier.verify(question, comp, expected_answer) for comp in completions]
                # 崩壊防止 (#1): グループ内に正報酬が1つも無い場合、報酬信号は実質「長さ
                # ペナルティのみ」となり、モデルが空応答/短文（改行のみ等）へ退化する。
                # この場合は advantage を全て0にして有害なポリシー更新をスキップする
                # （KL 正則化のみが残り、モデルは参照=SFT/DPO 近傍に保たれる）。
                if max(rewards) <= 0.0:
                    logger.debug(f"GRPO: no positive reward in group (max={max(rewards):.3f}); skipping policy update.")
                    advantages = torch.zeros(len(rewards), device=self.device)
                else:
                    advantages = self.verifier.compute_advantages(rewards).to(self.device)
                
                while not self.stop_event.is_set():
                    try:
                        self.output_queue.put((prompt, completions, rewards, advantages), timeout=1)
                        break
                    except queue.Full:
                        continue

def train_grpo():
    ckpt_path = f"{CKPT_GRPO}{config['ckpt_suffix']}{EXT_PT}"
    latest_path = f"{CKPT_GRPO_LATEST}{config['ckpt_suffix']}{EXT_PT}"
    dpo_path = f"{CKPT_DPO}{config['ckpt_suffix']}{EXT_PT}"
    sft_path = f"{CKPT_SFT}{config['ckpt_suffix']}{EXT_PT}"
    
    trainer = BaseTrainer("grpo", ckpt_path, fallback_ckpt_paths=[latest_path, dpo_path, sft_path])
    
    group_size = trainer.cfg["group_size"] # Number of samples (N) per prompt
    beta = trainer.cfg["beta"]
    epsilon = trainer.cfg["epsilon"]
    
    tokenizer = get_tokenizer()
    
    dataset = GRPODataset(max_samples=config.get("grpo_data_max_samples"))
    # We use batch_size=1 for prompts to easily generate group_size completions
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    verifier = RewardVerifier()
    
    trainer.setup_scheduler(len(dataloader))
    trainer.load_checkpoints()

    # データ変更が「完了扱い」で握り潰されるのを検知する（trainer_utils 参照）。
    if trainer.is_training_complete(data_size=len(dataloader.dataset)):
        return
        
    # L-3: GRPO は policy(勾配あり) / ref(KLアンカー) / actor(RolloutWorker のロールアウト生成) の
    # 3 モデルを載せ、加えて policy の optimizer 状態（AdamW ≒ パラメータの約2倍）も載る。
    # PROD/MAX の実サイズでは 3 体同時 GPU 常駐で OOM しやすい（TEST/SMOKE では顕在化しない）。
    # メモリ制約時は SWI_GRPO_REF_ON_CPU=1 で参照モデルを CPU に退避できる（既定は GPU＝従来どおり）。
    # 参照モデルは no_grad の eval 用途のみのため CPU 退避の許容度が高い（その分やや低速）。
    _ref_on_cpu = os.environ.get("SWI_GRPO_REF_ON_CPU", "").strip().lower() in ("1", "true", "yes")
    ref_device = torch.device("cpu") if _ref_on_cpu else trainer.device
    ref_model = copy.deepcopy(trainer.model).to(ref_device)
    ref_model.eval()
    if _ref_on_cpu:
        logger.info("GRPO: reference model placed on CPU (SWI_GRPO_REF_ON_CPU) to reduce GPU residency.")
    
    save_interval_steps = trainer.get_save_interval(len(dataloader))
    
    rollout_worker = RolloutWorker(dataloader, tokenizer, group_size, trainer.device, verifier)
    rollout_worker.start()
    
    # Initial state sync
    rollout_worker.sync_queue.put({k: v.cpu() for k, v in trainer.model.state_dict().items()})
    
    try:
        for epoch in range(trainer.start_epoch, trainer.epochs):
            pbar = tqdm(range(len(dataloader)), desc=f"GRPO Epoch {epoch+1}/{trainer.epochs}")
            trainer.optimizer.zero_grad(set_to_none=True)
            
            for step in pbar:
                if config["max_train_steps"] is not None and step >= config["max_train_steps"]: break
                if epoch == trainer.start_epoch and step < trainer.start_step: continue
                
                # Fetch pre-generated rollouts from the pipeline
                prompt, completions, rewards, advantages = rollout_worker.output_queue.get()

                # 問題4 (DAPO): グループ内 advantage が全て ~0（全員正解 or 全員不正解で分散ゼロ）の
                # 場合は学習信号が無い。このまま forward/backward まで進めると KL ペナルティだけが
                # 効いてポリシーが参照モデルへ不要に引き戻される（退化）。よってこのグループの
                # ポリシー更新（forward/backward）を完全にスキップする。step 進行・定期保存の整合の
                # ため step_and_save は呼ぶ（今回の勾配寄与はなし）。
                if torch.all(advantages.abs() < GRPO_DAPO_ADV_EPS):
                    logger.debug("GRPO/DAPO: zero-variance group (advantages ~0); skipping policy update.")
                    trainer.step_and_save(torch.tensor(0.0), step, len(dataloader), epoch,
                                          save_interval_steps, save_path=latest_path, skip_backward=True)
                    pbar.set_postfix({"loss": "skip(DAPO)", "reward": f"{sum(rewards)/len(rewards):.2f}"})
                    continue

                # Tokenize and format for forward pass
                batch_loss = 0.0

                # To compute logprobs correctly, we run each completion through the model
                for i, comp in enumerate(completions):
                    # Construct sequence
                    full_text = prompt + comp
                    tokens = tokenizer.encode(full_text)
                    prompt_len = len(tokenizer.encode(prompt))

                    # L-13: 切り詰め後に completion が残らない系列は学習信号ゼロなので捨てる。
                    seq = build_grpo_sequence(tokens, prompt_len, trainer.seq_len)
                    if seq is None:
                        logger.debug("GRPO: completion truncated away (prompt_len >= seq_len); skipping.")
                        continue
                    input_ids = torch.tensor([seq[0]], dtype=torch.long).to(trainer.device)
                    # プロンプト部分は -100 (ignore) でマスク済み
                    target_ids = torch.tensor([seq[1]], dtype=torch.long).to(trainer.device)
                    
                    # Old policy logprobs (from the generation we just did, so we can approximate by running no_grad on policy_model or ref_model if no older policy kept)
                    # In standard GRPO, old_policy is the policy before this update step.
                    # old_logps / ref_logps は重要度比の基準値のため、dropout を無効化した
                    # eval モードで決定論的に計算する (REQ-18)
                    trainer.model.eval()
                    with torch.no_grad():
                        with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                            old_logits, _ = trainer.model(input_ids)
                            old_logps = get_logprobs(old_logits, target_ids)

                            # L-3: ref がCPU退避時はそのデバイスで計算し、結果を学習デバイスへ戻す
                            # （既定は ref_device==trainer.device のため .to() は no-op で従来と同一）。
                            ref_logits, _ = ref_model(input_ids.to(ref_device))
                            ref_logps = get_logprobs(ref_logits, target_ids.to(ref_device)).to(trainer.device)

                    # Current policy logprobs (requires_grad)
                    trainer.model.train()
                    with torch.autocast(device_type=trainer.device.type, dtype=trainer.amp_dtype, enabled=trainer.device.type=='cuda'):
                        logits, aux_loss = trainer.model(input_ids)
                        current_logps = get_logprobs(logits, target_ids)
                        
                        # Policy Ratio
                        ratio = torch.exp(current_logps - old_logps.detach())
                        
                        # GRPO Clipped Surrogate Objective
                        surrogate1 = ratio * advantages[i]
                        surrogate2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages[i]
                        policy_loss = -torch.min(surrogate1, surrogate2).mean()
                        
                        # KL Divergence Penalty (against ref model to prevent mode collapse)
                        kl_div = torch.exp(ref_logps - current_logps.detach()) - (ref_logps - current_logps.detach()) - 1
                        kl_loss = beta * kl_div.mean()
                        
                        loss = (policy_loss + kl_loss + 0.01 * aux_loss) / (group_size * trainer.grad_accum_steps)
                        batch_loss += loss.item()
                        
                    trainer.scaler.scale(loss).backward()
                
                trainer.step_and_save(torch.tensor(0.0), step, len(dataloader), epoch, save_interval_steps, save_path=latest_path, skip_backward=True)
                
                if (step + 1) % trainer.grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    opt_step = (step + 1) // trainer.grad_accum_steps
                    if opt_step > 0 and opt_step % save_interval_steps == 0:
                        # Sync state to actor periodically
                        try:
                            rollout_worker.sync_queue.put_nowait({k: v.cpu() for k, v in trainer.model.state_dict().items()})
                        except queue.Full:
                            pass
                    
                pbar.set_postfix({"loss": f"{batch_loss * trainer.grad_accum_steps:.4f}", "reward": f"{sum(rewards)/len(rewards):.2f}"})
                
            trainer.start_step = 0
            logger.info("GRPO Training complete.")
            from swi_llm.utils.trainer_utils import save_checkpoint
            save_checkpoint(ckpt_path, trainer.model.state_dict(), trainer.optimizer, trainer.scheduler, trainer.scaler, trainer.epochs, -1, trainer.best_val_loss, data_size=trainer.data_size)
    finally:
        rollout_worker.stop_event.set()
        rollout_worker.sync_queue.put(None)
        
        # Flush output queue to unblock worker
        while not rollout_worker.output_queue.empty():
            try:
                rollout_worker.output_queue.get_nowait()
            except queue.Empty:   # L-18: 裸の except は KeyboardInterrupt まで握り潰す
                break
        rollout_worker.join(timeout=5)

if __name__ == "__main__":
    try:
        train_grpo()
    except Exception as e:
        logger.critical("A fatal error occurred during GRPO training.", exc_info=True)
        raise
