"""
swi_llm/core/generation/structured.py

構造化出力 / JSON Mode（ブループリント §6.3）。

BPE トークナイザ環境での制約デコーディングを、
「JSON プレフィックス検証器 + top-k 候補フィルタリング」で実現する:
  - 各ステップで次トークン候補（top-k）のうち、追加しても出力が「有効な JSON の
    接頭辞」であり続けるトークンだけを許可（それ以外の logits を -inf にマスク）。
  - 完全な JSON が形成された時点で停止。

これにより小型モデルでも 100% 構文的に有効な JSON / tool_call を生成できる。
"""
import json

import torch
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.chat_template import TOOL_CALL_START, TOOL_CALL_END

logger = setup_logger("generate_structured")

_WHITESPACE = " \t\n\r"
_LITERALS = ("true", "false", "null")


def is_valid_json_prefix(s: str) -> bool:
    """s が「ある有効な JSON 文書の接頭辞」であり得るなら True。

    末尾で未完のトークン（閉じていない文字列・途中の数値・途中のリテラル）は
    許容し、構文的に確定した誤りのみ False とする。
    """
    # expectation: 何を次に期待しているか
    #   'value'            : 値の開始
    #   'obj_key_or_end'   : '{' 直後（キー文字列 or '}'）
    #   'obj_key'          : ',' 直後（キー文字列）
    #   'colon'            : キーの後（':'）
    #   'obj_sep_or_end'   : オブジェクト内の値の後（',' or '}'）
    #   'arr_sep_or_end'   : 配列内の値の後（',' or ']'）
    #   'done'             : トップレベル値が完成（以降は空白のみ）
    expectation = "value"
    stack = []  # 'obj' / 'arr'
    i = 0
    n = len(s)

    def after_value():
        # 値を 1 つ読み終えた後の期待状態へ遷移
        if not stack:
            return "done"
        return "obj_sep_or_end" if stack[-1] == "obj" else "arr_sep_or_end"

    while i < n:
        c = s[i]
        if c in _WHITESPACE:
            i += 1
            continue

        if expectation in ("value", "obj_key_or_end", "obj_key"):
            if expectation == "obj_key_or_end" and c == "}":
                stack.pop()
                expectation = after_value()
                i += 1
                continue
            if expectation in ("obj_key_or_end", "obj_key"):
                # キーは文字列でなければならない
                if c != '"':
                    return False
                ok, i, closed = _scan_string(s, i)
                if not ok:
                    return False
                if not closed:
                    return True  # 未完文字列 = 有効な接頭辞
                expectation = "colon"
                continue
            # expectation == 'value'
            if c == "{":
                stack.append("obj")
                expectation = "obj_key_or_end"
                i += 1
                continue
            if c == "[":
                stack.append("arr")
                expectation = "value_or_end"
                i += 1
                continue
            if c == '"':
                ok, i, closed = _scan_string(s, i)
                if not ok:
                    return False
                if not closed:
                    return True
                expectation = after_value()
                continue
            if c == "-" or c.isdigit():
                i, complete = _scan_number(s, i)
                if not complete:
                    return True  # 数値が末尾で途中 = 有効な接頭辞
                expectation = after_value()
                continue
            if c in "tfn":
                ok, i, complete = _scan_literal(s, i)
                if not ok:
                    return False
                if not complete:
                    return True
                expectation = after_value()
                continue
            return False

        elif expectation == "value_or_end":
            # 配列の最初の要素 or ']'
            if c == "]":
                stack.pop()
                expectation = after_value()
                i += 1
                continue
            expectation = "value"
            continue  # 同じ文字を value として再評価

        elif expectation == "colon":
            if c != ":":
                return False
            expectation = "value"
            i += 1
            continue

        elif expectation == "obj_sep_or_end":
            if c == ",":
                expectation = "obj_key"
                i += 1
                continue
            if c == "}":
                stack.pop()
                expectation = after_value()
                i += 1
                continue
            return False

        elif expectation == "arr_sep_or_end":
            if c == ",":
                expectation = "value"
                i += 1
                continue
            if c == "]":
                stack.pop()
                expectation = after_value()
                i += 1
                continue
            return False

        elif expectation == "done":
            return False  # トップレベル値の後に非空白が来たら誤り

    return True


def _scan_string(s, i):
    """s[i]=='"' から文字列を走査。(ok, next_i, closed) を返す。"""
    assert s[i] == '"'
    i += 1
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            if i + 1 >= n:
                return True, n, False  # エスケープ途中 = 未完
            i += 2
            continue
        if c == '"':
            return True, i + 1, True
        i += 1
    return True, n, False  # 閉じていない = 未完（有効な接頭辞）


def _scan_number(s, i):
    """数値を走査。(next_i, complete) を返す。complete=False は末尾で途中。"""
    n = len(s)
    start = i
    if i < n and s[i] == "-":
        i += 1
    while i < n and (s[i].isdigit() or s[i] in ".eE+-"):
        i += 1
    # 末尾まで数値文字が続いた場合は「途中の可能性」あり
    complete = i < n
    return i, complete


def _scan_literal(s, i):
    """true/false/null を走査。(ok, next_i, complete) を返す。"""
    for lit in _LITERALS:
        if s.startswith(lit, i):
            return True, i + len(lit), True
        # 末尾までがリテラルの接頭辞なら未完
        remaining = s[i:]
        if lit.startswith(remaining):
            return True, len(s), False
    return False, i, False


def is_complete_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def generate_json(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7,
                  top_k=40, prefix=""):
    """JSON 制約付き生成（§6.3）。有効な JSON が完成した時点で停止する。

    Args:
        prefix: 生成前に「既に出力済みの JSON 断片」として扱う文字列。
            プロンプト末尾に付与してモデルを誘導しつつ、接頭辞検証・完成判定・
            戻り値にも含める。例: prefix="{" で payload を必ず JSON オブジェクトに制約。

    Returns:
        生成された JSON 文字列（prompt は含まないが prefix は含む）。
    """
    tm = model.text_model if hasattr(model, "text_model") else model
    device = next(tm.parameters()).device
    prompt_ids = tokenizer.encode(prompt + prefix)
    generated = []

    tm.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
            input_ids = input_ids[:, -getattr(tm, "max_seq_len", 512):]
            outputs = tm(input_ids)
            logits = outputs[0][0, -1, :] / max(temperature, 1e-5)

            # top-k 候補に絞る
            k = min(top_k, logits.shape[-1])
            top_vals, top_idx = torch.topk(logits, k)

            current_text = prefix + tokenizer.decode(generated)
            # 有効な JSON 接頭辞を保つ候補だけ残す
            allowed_vals, allowed_idx = [], []
            for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                cand_text = current_text + tokenizer.decode([idx])
                if is_valid_json_prefix(cand_text):
                    allowed_vals.append(val)
                    allowed_idx.append(idx)

            if not allowed_idx:
                # 有効な継続が無ければ打ち切り
                break

            probs = F.softmax(torch.tensor(allowed_vals, device=device), dim=-1)
            choice = torch.multinomial(probs, num_samples=1).item()
            next_id = allowed_idx[choice]
            generated.append(next_id)

            text = prefix + tokenizer.decode(generated)
            if is_complete_json(text):
                # 完全な JSON になったら終了（末尾の余分生成を防ぐ）
                break

    return prefix + tokenizer.decode(generated)


def generate_tool_call(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7,
                       top_k=40):
    """制約デコーディングで構文的に必ず有効な tool_call を生成する（§6.3 / Q-10）。

    prompt の続きに `<tool_call>{...}</tool_call>` を生成する。JSON 本体は
    generate_json（JSON 接頭辞検証 + top-k フィルタ）で構築し、prefix="{" で
    必ず JSON オブジェクト始まりに制約するため、小型モデルでも壊れた tool_call に
    ならない（100% 構文的に有効な JSON）。ReAct ループの不正 tool_call 修復に用いる。

    Returns:
        "<tool_call>{...}</tool_call>" 形式の文字列（payload は常に有効な JSON）。
    """
    json_prompt = prompt + TOOL_CALL_START
    payload = generate_json(model, tokenizer, json_prompt, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k, prefix="{").strip()
    if not is_complete_json(payload):
        # 生成が途中終了した場合は構文有効性を最優先し空オブジェクトへフォールバック
        payload = "{}"
    return f"{TOOL_CALL_START}{payload}{TOOL_CALL_END}"
