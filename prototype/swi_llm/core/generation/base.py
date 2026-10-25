from .stream import generate_stream
from swi_llm.utils.logger import setup_logger

logger = setup_logger("generate_base")

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1, images=None):
    """
    generate_stream を内部で呼び出し、結果をまとめて返す非ストリーミング生成関数
    """
    generator = generate_stream(
        model, tokenizer, prompt, max_new_tokens, temperature, 
        top_k, top_p, repetition_penalty, images
    )
    result = ""
    for chunk in generator:
        result += chunk
    return prompt + result
