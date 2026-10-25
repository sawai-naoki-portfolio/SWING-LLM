import json
import os
import sys
import argparse

# Add prototype directory to sys.path to resolve 'utils' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.chat_template import format_message, ROLE_USER, ROLE_ASSISTANT

logger = setup_logger("data.formatter")


def format_dataset(input_file, output_file):
    """
    HuggingFace等で一般的な QAペアの JSONL ファイルを、
    SWI-LLMの学習用フォーマット（ChatML形式）に変換します。
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file '{input_file}' not found.")
        return

    processed_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                question = data.get("question")
                answer = data.get("answer")

                if question and answer:
                    # SWI-LLMの学習用フォーマット（ChatML）に変換
                    formatted_text = format_message(ROLE_USER, question) + format_message(ROLE_ASSISTANT, answer)

                    # 学習パイプライン用に {"text": "..."} の形式で出力
                    out_record = {"text": formatted_text}
                    outfile.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    processed_count += 1
                else:
                    logger.warning(f"Missing 'question' or 'answer' in line: {line}")

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON line: {line}")

    logger.info("==========================================")
    logger.info("Conversion complete!")
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"Processed {processed_count} records.")
    logger.info("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert QA dataset to SWI-LLM ChatML format.")
    parser.add_argument("--input", type=str, default="data/math_sample.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output", type=str, default="data/formatted_dataset.jsonl", help="Path to output JSONL file")

    args = parser.parse_args()

    # スクリプトがあるディレクトリを基準にする
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output

    format_dataset(input_path, output_path)
