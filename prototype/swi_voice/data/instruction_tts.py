"""
swi_voice/data/instruction_tts.py
指示追従TTS のSFTデータ構築 [音声編6.3, 実装編5.3]

「落ち着いた低い声でゆっくり読んで」のような自然言語のスタイル指示に従って発話させる
ため、指示文と発話テキストを 1 本のトークン列にまとめた学習例を組み立てる。

  input_ids = tokenize("[STYLE]{指示}[/STYLE]" + "{発話テキスト}")
  target_codes = codec.encode(target_audio)     # 教師となる音声のコーデックコード列

ロスマスキング [実装編5.3]:
  LLM実装編5.1 の format_and_mask と同型に、「モデルが生成すべき部分だけ」に損失をかける。
  音声トークン側では ARCodecLM の `prompt_frames`（先頭 N フレームを損失外にする）が
  その役割を担うため、ここでは音響プロンプト長を一緒に返す。テキスト側は条件入力であり
  そもそも損失対象ではない（ARCodecLM.forward はフレーム位置のみを予測する）。

`instruction_ids` も返すため、StyleEncoder の指示経路 [実装編4.2] に
`style_emb = style_encoder(instruction_ids=...)` として渡す使い方もできる。
"""
import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_STYLE_TAG_OPEN, VOICE_STYLE_TAG_CLOSE, VOICE_MAX_INSTRUCTION_TOKENS,
)

logger = setup_logger("voice.instruction_tts")


def format_style_prompt(instruction_text, text=""):
    """スタイル指示と発話テキストを 1 本の文字列へ整形する [実装編5.3]。

    指示が空なら発話テキストのみを返す（無条件TTSと同じ形式に落ちる）。
    """
    instruction = (instruction_text or "").strip()
    if not instruction:
        return str(text)
    return f"{VOICE_STYLE_TAG_OPEN}{instruction}{VOICE_STYLE_TAG_CLOSE}{text}"


def encode_instruction(instruction_text, tokenizer, max_tokens=VOICE_MAX_INSTRUCTION_TOKENS):
    """スタイル指示のみをトークン化する（StyleEncoder の指示経路へ渡す用）[実装編4.2]。

    空の指示や、トークン化結果が空になった場合は None を返す（無条件生成へフォールバック）。
    """
    instruction = (instruction_text or "").strip()
    if not instruction:
        return None
    ids = tokenizer.encode(instruction)[:max_tokens]
    if not ids:
        logger.warning("encode_instruction: tokenizer produced no tokens; no style conditioning.")
        return None
    return torch.tensor(ids, dtype=torch.long)


def build_instruction_tts_example(instruction_text, target_audio, codec, tokenizer,
                                  text="", prompt_frames=0,
                                  max_instruction_tokens=VOICE_MAX_INSTRUCTION_TOKENS):
    """指示追従TTS の学習例を 1 件組み立てる [音声編6.3, 実装編5.3]。

    Args:
        instruction_text: 自然言語のスタイル指示（例: "落ち着いた低い声でゆっくり"）。
        target_audio: 教師音声の波形（[T] / [1,T] / [1,1,T]）。
        codec: `.encode(wav) -> [B, T', n_q]` を実装する NeuralAudioCodec。
        tokenizer: `.encode(str) -> list[int]` を実装するトークナイザ（swi_llm の SentencePiece）。
        text: 発話テキスト（省略時はスタイル指示のみを条件にする）。
        prompt_frames: 音響プロンプトとして損失から除外する先頭フレーム数
            （ARCodecLM.forward の `prompt_frames` にそのまま渡す）。

    Returns:
        {"input_ids": LongTensor[Lt], "target_codes": LongTensor[T', n_q],
         "instruction_ids": LongTensor[Li] | None, "prompt_frames": int}
        教師音声が空、またはコーデックが 0 フレームしか返さない場合は None
        （V-2: 想定内の WARNING を出して呼び出し側がスキップできるようにする）。
    """
    if target_audio is None or target_audio.numel() == 0:
        logger.warning("build_instruction_tts_example: empty target_audio; skipping example.")
        return None

    wav = target_audio
    if wav.dim() == 1:
        wav = wav.view(1, 1, -1)
    elif wav.dim() == 2:
        wav = wav.unsqueeze(0)

    with torch.no_grad():
        target_codes = codec.encode(wav)                     # [1, T', n_q]
    if target_codes.shape[1] == 0:
        logger.warning("build_instruction_tts_example: codec produced 0 frames; skipping example.")
        return None
    target_codes = target_codes.squeeze(0)                   # [T', n_q]

    prompt = format_style_prompt(instruction_text, text)
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        logger.warning("build_instruction_tts_example: tokenizer produced no tokens; skipping example.")
        return None

    # prompt_frames はターゲット長を超えられない（全フレームが損失外になるのを防ぐ）
    n_frames = target_codes.shape[0]
    clamped_prompt_frames = max(0, min(int(prompt_frames), n_frames - 1))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "target_codes": target_codes,
        "instruction_ids": encode_instruction(instruction_text, tokenizer, max_instruction_tokens),
        "prompt_frames": clamped_prompt_frames,
    }
