import torch
import torch.nn.functional as F

from swi_llm.utils.constants import REPETITION_PENALTY_WINDOW

def sample_next_token(logits, generated, temperature, repetition_penalty, top_k, top_p, tokenizer, EOT_TOKEN_ID, prevent_eot=False):
    # Prevent division by zero if temperature is exactly 0
    temp = max(temperature, 1e-5)
    next_token_logits = logits[:, -1, :] / temp

    if repetition_penalty != 1.0:
        # L-12: 直近 REPETITION_PENALTY_WINDOW トークンのみを対象にし、長文生成での
        # set(generated) 全走査による線形コストを避ける（短い生成では全体と同挙動）。
        recent = generated[-REPETITION_PENALTY_WINDOW:] if len(generated) > REPETITION_PENALTY_WINDOW else generated
        for token in set(recent):
            if next_token_logits[0, token] < 0:
                next_token_logits[0, token] *= repetition_penalty
            else:
                next_token_logits[0, token] /= repetition_penalty
                
    if prevent_eot:
        next_token_logits[0, EOT_TOKEN_ID] = float('-inf')
        if hasattr(tokenizer, 'eot_token_id') and tokenizer.eot_token_id is not None:
            next_token_logits[0, tokenizer.eot_token_id] = float('-inf')
            
    if top_k > 0:
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_logits[indices_to_remove] = float('-inf')
        
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = float('-inf')
        
    probs = F.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token
