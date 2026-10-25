from .base import generate
from swi_llm.utils.logger import setup_logger
from swi_llm.utils.chat_template import THINK_START, THINK_END

logger = setup_logger("generate_thought")

def generate_with_thought(model, tokenizer, prompt, max_thought_tokens=50, max_answer_tokens=50):
    """
    Implementation of Extended Chain-of-Thought at Inference without mock structures.
    Forces the model to generate <think> tokens first.
    """
    cot_prompt = prompt + f"\n{THINK_START}\n"

    logger.info("generate_with_thought: Phase 1/2 - Thinking...")
    full_output = generate(model, tokenizer, cot_prompt, max_new_tokens=max_thought_tokens)

    if THINK_END not in full_output:
        full_output += f"\n{THINK_END}\n"
        
    logger.info("generate_with_thought: Phase 2/2 - Answering...")
    final_output = generate(model, tokenizer, full_output, max_new_tokens=max_answer_tokens)
    
    return final_output
