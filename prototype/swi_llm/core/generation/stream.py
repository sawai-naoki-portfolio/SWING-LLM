import torch
import copy
import threading
from collections import OrderedDict
from swi_llm.data.tokenizer import EOT_TOKEN_ID
from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    PROMPT_BUDGET_HIGH, PROMPT_BUDGET_LOW, MAX_REACT_RETRIES,
    COT_HIGH_DIFFICULTY_PATTERN, PREFIX_CACHE_MAX_ENTRIES, KV_CACHE_EXTRA_MARGIN,
)
from swi_llm.utils.chat_template import TOOL_CALL_START, TOOL_CALL_END, TOOL_CALL_PATTERN, THINK_START, THINK_END, format_tool_call, format_tool_result
from swi_llm.utils.config import INFERENCE_CONFIG, TRAINING_CONFIG
from .sampler import sample_next_token

logger = setup_logger("generate_stream")

# prefix キャッシュ（プロンプト KV の再利用）。LRU 上限つき OrderedDict で
# 無制限なメモリ増加（＝サーバ長時間運用での OOM）を防ぐ。
# 同時推論（ThreadPool + semaphore）で共有されるため read/write をロックで保護する
# （無保護だと同一プロンプトの並行要求で deepcopy 中に破損 KV を読むレースが起きる）。
PREFIX_CACHE_STORE = OrderedDict()
_PREFIX_CACHE_LOCK = threading.Lock()


def clear_prefix_cache():
    """prefix キャッシュを全消去する。モデル重み再ロード（継続学習/LoRA）後に
    古い KV を使い続けないよう呼び出す。"""
    with _PREFIX_CACHE_LOCK:
        PREFIX_CACHE_STORE.clear()


def _prefix_cache_get(key):
    """LRU: ヒットしたら最近使用として末尾へ移動し、一貫スナップショット（deepcopy）を返す。"""
    with _PREFIX_CACHE_LOCK:
        cached = PREFIX_CACHE_STORE.get(key)
        if cached is None:
            return None
        PREFIX_CACHE_STORE.move_to_end(key)
        return copy.deepcopy(cached)   # ロック下で一貫スナップショットを取得


def _prefix_cache_put(key, snapshot):
    """LRU 上限を超えたら最古のエントリを退避してからスナップショットを格納する。"""
    with _PREFIX_CACHE_LOCK:
        PREFIX_CACHE_STORE[key] = snapshot
        PREFIX_CACHE_STORE.move_to_end(key)
        while len(PREFIX_CACHE_STORE) > PREFIX_CACHE_MAX_ENTRIES:
            PREFIX_CACHE_STORE.popitem(last=False)   # 最古(LRU)を退避

# 学習時のシーケンス長。これを超えた生成は学習分布の外となり
# 品質が劣化しうるため、超過時に WARNING を出す (REQ-19)。
TRAINED_SEQ_LEN = TRAINING_CONFIG["pretrain"]["seq_len"]


def _warn_if_beyond_trained_length(position, already_warned):
    """生成位置が学習時シーケンス長を超えた最初の時点で WARNING を出す。"""
    if not already_warned and position > TRAINED_SEQ_LEN:
        logger.warning(
            f"Generation position ({position}) exceeded trained sequence length "
            f"({TRAINED_SEQ_LEN}). Output quality may degrade beyond this point."
        )
        return True
    return already_warned

def generate_stream(model, tokenizer, prompt, max_new_tokens=100, temperature=0.3, top_k=50, top_p=0.85, repetition_penalty=1.15, images=None, stop_event=None):
    """
    Generator that yields the newly generated text delta. Uses KV cache.

    stop_event: threading.Event。set されたら次の反復で生成を打ち切る（クライアント
    切断時にワーカースレッドを早期停止し、スレッドプールのゾンビ占有を防ぐ）。
    """
    model.eval()
    device = next(model.parameters()).device
    
    tokens = tokenizer.encode(prompt)

    generated = list(tokens)
    prompt_len = len(tokens)
    previous_text = ""

    from swi_llm.core.models.transformer_blocks import StaticMLACache
    bsz = 1
    model_max_seq_len = getattr(model, 'max_seq_len', 512)
    # 128k 文脈をフル確保せず、実際に必要な長さ（プロンプト+生成上限+余裕）だけを確保する。
    # StaticMLACache は事前確保サイズが固定なので、これを生成ループの上限にも用いる。
    cache_capacity = min(model_max_seq_len, prompt_len + max_new_tokens + KV_CACHE_EXTRA_MARGIN)
    layers = model.layers if hasattr(model, 'layers') else model.text_model.layers
    latent_dim = getattr(layers[0].attention, 'latent_dim', 512)
    model_dtype = next(model.parameters()).dtype

    past_key_values = [
        StaticMLACache(bsz, cache_capacity, latent_dim, device, model_dtype)
        for _ in range(len(layers))
    ]

    recompute_cache = True
    # prefix キャッシュは画像なしの純テキスト要求のみ対象（VLM は画像で KV が変わるため
    # 同一テキストでも衝突する。キーに画像を含めない代わりに使用条件で除外する）。
    use_prefix_cache = images is None

    # DEBUG: 要求単位で解決した生成設定（ホットパスのループ外＝1リクエスト1回）
    logger.debug(
        f"generate_stream: prompt_len={prompt_len}, max_new_tokens={max_new_tokens}, "
        f"images={'yes' if images is not None else 'no'}, prefix_cache={use_prefix_cache}, "
        f"cache_capacity={cache_capacity}/{model_max_seq_len}, "
        f"sampling(temp={temperature}, top_k={top_k}, top_p={top_p}, rep_pen={repetition_penalty})")
    prompt_tuple = tuple(tokens)

    if use_prefix_cache:
        cached = _prefix_cache_get(prompt_tuple)
        if cached is not None:
            logger.debug("Prefix cache HIT")   # REQ-09: キャッシュ詳細は DEBUG（生成ホットパス）
            past_key_values = cached   # prompt[:-1] までの KV（最後の1トークンはこれから処理）
            recompute_cache = False
        else:
            logger.debug("Prefix cache MISS")
    
    react_retry_count = 0
    mtp_draft_tokens = []
    warned_trained_len = False
    yielded_visible = False   # 非空白（可視）テキストを1度でも yield したか（空白のみ退化の検出用）

    with torch.no_grad():
        for i in range(max_new_tokens):
            if stop_event is not None and stop_event.is_set():
                break   # クライアント切断等でキャンセル要求 → 早期停止
            if recompute_cache:
                inp = torch.tensor([generated], dtype=torch.long).to(device)
                start_pos = 0
                recompute_cache = False
                current_images = images
            else:
                if len(mtp_draft_tokens) > 0:
                    inp = torch.tensor([[generated[-1]] + mtp_draft_tokens], dtype=torch.long).to(device)
                    start_pos = len(generated) - 1
                else:
                    inp = torch.tensor([[generated[-1]]], dtype=torch.long).to(device)
                    start_pos = len(generated) - 1
                current_images = None

            # MTP 投機デコードでは 1 回の forward が複数トークン（1 + ドラフト数）を KV へ
            # 書き込む。ガードが 1 トークン分（cache_capacity - 1）しか見ないと、境界付近で
            # StaticMLACache の事前確保領域をはみ出し、ゼロ幅スライスへの代入で RuntimeError
            # クラッシュする。実際の書き込み幅 inp.shape[1] を予約する（write_width=1 のとき
            # 従来の `start_pos >= cache_capacity - 1` と厳密に一致する）。
            write_width = inp.shape[1]
            if start_pos + write_width >= cache_capacity:
                logger.warning(f"Reached KV cache capacity ({start_pos}+{write_width}/{cache_capacity}).")
                yield "\n[Warning: Reached maximum sequence length]"
                break

            warned_trained_len = _warn_if_beyond_trained_length(len(generated), warned_trained_len)

            use_spec = INFERENCE_CONFIG.get("use_speculative_decoding", False)
            return_mtp = use_spec and (getattr(model, 'n_mtp_future', 0) > 0 or (hasattr(model, 'text_model') and getattr(model.text_model, 'n_mtp_future', 0) > 0))
            if hasattr(model, 'vision_encoder'):
                outputs = model(inp, start_pos=start_pos, use_cache=True, past_key_values=past_key_values, images=current_images, return_mtp=return_mtp)
            else:
                outputs = model(inp, start_pos=start_pos, use_cache=True, past_key_values=past_key_values, return_mtp=return_mtp)
                
            if return_mtp:
                logits, _, past_key_values, mtp_logits = outputs
            else:
                logits, _, past_key_values = outputs
                mtp_logits = None
                
            if use_prefix_cache and start_pos == 0 and len(generated) > 1:
                # プロンプト全体を1回で処理した直後の KV から、末尾1トークン分を rollback して
                # 「prompt[:-1] までの KV」を保存する。HIT 時はこのキャッシュに対して最後の
                # プロンプトトークンを start_pos=prompt_len-1 で1回だけ処理するため、
                # cache_seqlen == start_pos が成立し、KV 二重追記・RoPE 位置ずれ・
                # cache_num_vision 誤検知（HIT と MISS で出力が食い違う不具合）を回避する。
                snapshot = copy.deepcopy(past_key_values)     # ロック外で重い deepcopy
                for c in snapshot:
                    if hasattr(c, 'rollback'):
                        c.rollback(1)
                _prefix_cache_put(prompt_tuple, snapshot)
                
            if len(mtp_draft_tokens) > 0:
                target_logits = logits[0, :-1, :]
                accepted_count = 0
                
                for j, draft_tok in enumerate(mtp_draft_tokens):
                    pred_tok = target_logits[j].argmax(dim=-1).item()
                    if pred_tok == draft_tok:
                        accepted_count += 1
                        generated.append(draft_tok)
                    else:
                        break
                        
                if accepted_count < len(mtp_draft_tokens):
                    reject_count = len(mtp_draft_tokens) - accepted_count
                    for cache in past_key_values:
                        if hasattr(cache, 'rollback'):
                            cache.rollback(reject_count)
                            
                next_token = sample_next_token(
                    logits[:, accepted_count:accepted_count+1, :], generated, temperature, repetition_penalty, 
                    top_k, top_p, tokenizer, EOT_TOKEN_ID, prevent_eot=(i < 20)
                )
                generated.append(next_token)
                mtp_draft_tokens = []
            else:
                next_token = sample_next_token(
                    logits[:, -1:, :], generated, temperature, repetition_penalty, 
                    top_k, top_p, tokenizer, EOT_TOKEN_ID, prevent_eot=(i < 20)
                )
                generated.append(next_token)
                
            if mtp_logits is not None:
                mtp_draft_tokens = []
                for head_logits in mtp_logits:
                    draft_tok = head_logits[0, -1, :].argmax(dim=-1).item()
                    mtp_draft_tokens.append(draft_tok)
            
            if generated[-1] == EOT_TOKEN_ID or (hasattr(tokenizer, 'eot_token') and generated[-1] == tokenizer.eot_token):
                break
            
            # SentencePiece では全 ID が有効。特殊トークンは surface 文字列
            # （</tool_call> 等）へ復元され、後段の ReAct 検知にも利用される。
            current_text = tokenizer.decode(generated[prompt_len:])
            if current_text.endswith('\ufffd'):
                continue
                
            if len(current_text) > len(previous_text):
                delta = current_text[len(previous_text):]
                previous_text = current_text
                if delta.strip():
                    yielded_visible = True
                yield delta

            # L-12: TOOL_CALL_PATTERN はモジュール先頭で import 済み（旧: ループ内で毎反復 import）。
            # 完全な <tool_call>...</tool_call> が揃った場合のみツール実行を試みる。
            # 未学習モデルが終了タグ単独 (</tool_call>) を出力するのは正常系として
            # 無視する（アトミックトークン化で発生しやすいため raise/ERROR にしない）。
            tool_matches = TOOL_CALL_PATTERN.findall(current_text) if TOOL_CALL_END in current_text else []
            if tool_matches and react_retry_count < MAX_REACT_RETRIES:
                from swi_llm.agent.base import AgentBase
                from swi_llm.agent.tools import TOOL_SCHEMAS, get_weather, calculate, execute_python, search_documents
                from swi_llm.utils.constants import (
                    TOOL_GET_WEATHER, TOOL_CALCULATE, TOOL_EXECUTE_PYTHON, TOOL_SEARCH_DOCUMENTS,
                    code_exec_tool_enabled,
                )

                agent = AgentBase()
                agent.register_tool(TOOL_GET_WEATHER, get_weather, TOOL_SCHEMAS[TOOL_GET_WEATHER])
                agent.register_tool(TOOL_CALCULATE, calculate, TOOL_SCHEMAS[TOOL_CALCULATE])
                # SEC-6: モデル発話由来の execute_python は既定で登録しない（プロンプト
                # インジェクションからの任意コード実行を遮断）。SWI_ENABLE_CODE_EXEC_TOOL=1 の
                # ときのみ登録する。未登録時はモデルの execute_python 呼び出しは
                # process_llm_output が「Tool not found」を返し、ReAct ループは通常応答へ継続する。
                if code_exec_tool_enabled():
                    agent.register_tool(TOOL_EXECUTE_PYTHON, execute_python, TOOL_SCHEMAS[TOOL_EXECUTE_PYTHON])
                else:
                    # 理由の説明用（DEBUG）。実際にモデルが execute_python を呼んだ場合は
                    # base.process_llm_output が「not registered」を WARNING で記録する。
                    logger.debug("execute_python not registered for model-triggered calls (SEC-6). "
                                 "Set SWI_ENABLE_CODE_EXEC_TOOL=1 to enable in a trusted environment.")
                agent.register_tool(TOOL_SEARCH_DOCUMENTS, search_documents, TOOL_SCHEMAS[TOOL_SEARCH_DOCUMENTS])

                try:
                    tool_json_str = tool_matches[-1].strip()

                    # Q-10 (§6.3): 壊れた JSON tool_call は制約デコーディングで有効な
                    # JSON へ再生成する。100% 構文的に有効な tool_call を保証し、
                    # 小型モデルのハルシネーションによるパース失敗を未然に防ぐ。
                    from swi_llm.core.generation.structured import generate_tool_call, is_complete_json
                    if INFERENCE_CONFIG.get("constrained_tool_calls", True) and not is_complete_json(tool_json_str):
                        full_text = tokenizer.decode(generated[prompt_len:])
                        pos = full_text.rfind(TOOL_CALL_START)
                        pre_text = tokenizer.decode(generated[:prompt_len]) + (full_text[:pos] if pos >= 0 else full_text)
                        try:
                            repaired = generate_tool_call(model, tokenizer, pre_text)
                            m = TOOL_CALL_PATTERN.search(repaired)
                            if m:
                                tool_json_str = m.group(1).strip()
                                logger.info("Repaired malformed tool_call via constrained JSON decoding (Q-10).")
                        except Exception:
                            logger.exception("Constrained tool_call regeneration failed; using raw output.")

                    mock_output = format_tool_call(tool_json_str)
                    # REQ-09/SEC-7: ツール呼び出しの生ペイロード詳細は DEBUG（内容を app.log に残さない）。
                    logger.debug(f"Detected tool call: {mock_output}")

                    # V-12: 成否は戻り値 call_ok（呼び出しローカル）で受け取る。共有インスタンスの
                    # last_call_ok 属性は、音声側 tool_bridge 等の並行呼び出しで上書きされうるため
                    # 読まない（process_llm_output_ex は tool_call がある限り、未登録ツール/不正
                    # JSON でもエラー用 <tool_result> 文字列を返すため、戻り値の真偽だけでは
                    # 成功/失敗を区別できない）。
                    tool_response, call_ok = agent.process_llm_output_ex(mock_output)

                    # 19(2): モデル自発 Function Calling の成功率 KPI（ログ集計用）。
                    # ok 率 = grep "FC-KPI" のうち ok=True の割合。ルールベース先回り
                    # （INTENT-KPI, web/intent.py）の撤去判断（19(3)）の材料。ペイロードは
                    # 記録しない（REQ-09/SEC-7）。
                    logger.info(f"FC-KPI: model_tool_call ok={call_ok}")

                    if call_ok:
                        yield "\n\n⚙️ [システム: ツールを実行し、結果を推論に反映しました]\n"
                        react_retry_count = 0
                    else:
                        # tool_call は検出されたが dispatch 失敗（未登録ツール / 不正な形式）。
                        # tool_response には具体的なエラー <tool_result> が入っているので
                        # それをそのままモデルへ注入し、ユーザーには失敗を正直に伝える。
                        react_retry_count += 1
                        yield f"\n\n⚠️ [システム: ツール呼び出しに失敗しました（未登録ツールまたは不正な形式） ({react_retry_count}/{MAX_REACT_RETRIES})]\n"
                        if not tool_response:
                            tool_response = format_tool_result("Error: Invalid tool call format or unknown tool.\n")

                        if react_retry_count >= MAX_REACT_RETRIES:
                            yield "\n\n🛑 [システム: ツールの連続失敗が上限に達したため、通常回答モードに移行します]\n"
                            tool_response += "Error: Tool execution failed multiple times. Please answer the user directly without using tools.\n"

                    result_tokens = tokenizer.encode(tool_response)
                    
                    if len(result_tokens) > 1 and past_key_values is not None:
                        inp_tool = torch.tensor([result_tokens[:-1]], dtype=torch.long).to(device)
                        start_pos_tool = len(generated)
                        if hasattr(model, 'vision_encoder'):
                            _, _, past_key_values = model(inp_tool, start_pos=start_pos_tool, use_cache=True, past_key_values=past_key_values, images=None)
                        else:
                            _, _, past_key_values = model(inp_tool, start_pos=start_pos_tool, use_cache=True, past_key_values=past_key_values)
                            
                    generated.extend(result_tokens)
                    prompt_len = len(generated)
                    previous_text = ""
                    
                except Exception as e:
                    logger.exception(f"Error executing ReAct loop: {e}")
                    yield f"\n[Error executing tool: {e}]\n"
                    prompt_len = len(generated)
                    previous_text = ""

    # 空応答/空白のみ応答のセーフガード。可視（非空白）テキストを一度も yield して
    # いない場合に発火する。次の 2 ケースを救済する:
    #   (1) バイトフォールバックのみで全て � としてスキップ → 最終デコードを返す
    #   (2) モデルが空白（改行等）のみに退化 → 明示メッセージを返し、UI 空白表示や
    #       「空応答」誤検知を防ぐ（学習済みモデルでは通常この経路は通らない）
    if not yielded_visible:
        fallback = tokenizer.decode(generated[prompt_len:])
        if fallback.strip():
            yield fallback
        else:
            yield "（有効な応答を生成できませんでした。モデルの学習が不十分な可能性があります。）"


def generate_o1_stream(model, tokenizer, prompt, max_thought_tokens=200, max_answer_tokens=200, temperature=0.3, top_k=50, top_p=0.85, repetition_penalty=1.15, images=None, difficulty="auto", stop_event=None):
    """
    Native implementation of Extended Chain-of-Thought at Inference (o1-like).

    stop_event: threading.Event。set されたら生成を打ち切る（クライアント切断時の早期停止）。
    """
    import re
    if difficulty == "auto":
        if re.search(COT_HIGH_DIFFICULTY_PATTERN, prompt):
            difficulty = "high"
        else:
            difficulty = "low"
            
    if difficulty == "high":
        max_thought_tokens = max(max_thought_tokens, 1024)
        budget_prompt = PROMPT_BUDGET_HIGH.format(tokens=max_thought_tokens)
    else:
        max_thought_tokens = min(max_thought_tokens, 256)
        budget_prompt = PROMPT_BUDGET_LOW.format(tokens=max_thought_tokens)

    # DEBUG: 動的思考予算の決定（難易度判定→思考トークン上限）。要求単位で1回。
    logger.debug(f"generate_o1_stream: difficulty={difficulty} -> max_thought_tokens={max_thought_tokens}, "
                 f"max_answer_tokens={max_answer_tokens}")

    prompt = budget_prompt + prompt
        
    model.eval()
    device = next(model.parameters()).device
    
    if not prompt.endswith(f"{THINK_START}\n"):
        if prompt.endswith(THINK_START):
            prompt += "\n"
        else:
            prompt += f"{THINK_START}\n"

    yield f"{THINK_START}\n"
    
    tokens = tokenizer.encode(prompt)

    generated = list(tokens)
    prompt_len = len(tokens)
    previous_text = ""

    from swi_llm.core.models.transformer_blocks import StaticMLACache
    bsz = 1
    model_max_seq_len = getattr(model, 'max_seq_len', 512)
    # 思考+回答フェーズ分の長さ+余裕だけを確保（128k フル確保を避ける）。
    cache_capacity = min(model_max_seq_len,
                         prompt_len + max_thought_tokens + max_answer_tokens + KV_CACHE_EXTRA_MARGIN)
    layers = model.layers if hasattr(model, 'layers') else model.text_model.layers
    latent_dim = getattr(layers[0].attention, 'latent_dim', 512)
    model_dtype = next(model.parameters()).dtype

    past_key_values = [
        StaticMLACache(bsz, cache_capacity, latent_dim, device, model_dtype)
        for _ in range(len(layers))
    ]
    
    phase = "thinking"
    tokens_in_current_phase = 0
    max_tokens = max_thought_tokens
    warned_trained_len = False

    with torch.no_grad():
        while True:
            if stop_event is not None and stop_event.is_set():
                break   # クライアント切断等でキャンセル要求 → 早期停止
            if tokens_in_current_phase >= max_tokens:
                if phase == "thinking":
                    yield f"\n{THINK_END}\n"
                    end_think_tokens = tokenizer.encode(f"\n{THINK_END}\n")
                    generated.extend(end_think_tokens)
                    prompt_len = len(generated)
                    previous_text = ""
                    
                    phase = "answering"
                    tokens_in_current_phase = 0
                    max_tokens = max_answer_tokens
                    
                    if len(end_think_tokens) > 1 and past_key_values is not None:
                        inp_tool = torch.tensor([end_think_tokens[:-1]], dtype=torch.long).to(device)
                        start_pos_tool = len(generated) - len(end_think_tokens)
                        if hasattr(model, 'vision_encoder'):
                            _, _, past_key_values = model(inp_tool, start_pos=start_pos_tool, use_cache=True, past_key_values=past_key_values, images=None)
                        else:
                            _, _, past_key_values = model(inp_tool, start_pos=start_pos_tool, use_cache=True, past_key_values=past_key_values)
                    continue
                else:
                    break
            
            if past_key_values[0].seqlen == 0:
                inp = torch.tensor([generated], dtype=torch.long).to(device)
                start_pos = 0
                current_images = images
            else:
                inp = torch.tensor([[generated[-1]]], dtype=torch.long).to(device)
                start_pos = len(generated) - 1
                current_images = None
                
            if start_pos >= cache_capacity - 1:
                logger.warning(f"Reached KV cache capacity ({start_pos}/{cache_capacity}).")
                yield "\n[Warning: Reached maximum sequence length]"
                break

            warned_trained_len = _warn_if_beyond_trained_length(len(generated), warned_trained_len)

            if hasattr(model, 'vision_encoder'):
                logits, _, past_key_values = model(inp, start_pos=start_pos, use_cache=True, past_key_values=past_key_values, images=current_images)
            else:
                logits, _, past_key_values = model(inp, start_pos=start_pos, use_cache=True, past_key_values=past_key_values)
                
            next_token = sample_next_token(
                logits, generated, temperature, repetition_penalty, 
                top_k, top_p, tokenizer, EOT_TOKEN_ID, prevent_eot=(tokens_in_current_phase < 5)
            )
            
            generated.append(next_token)
            tokens_in_current_phase += 1
            
            if next_token == EOT_TOKEN_ID or (hasattr(tokenizer, 'eot_token') and next_token == tokenizer.eot_token):
                if phase == "thinking":
                    yield f"\n{THINK_END}\n"
                    generated.pop()
                    end_think_tokens = tokenizer.encode(f"\n{THINK_END}\n")
                    generated.extend(end_think_tokens)
                    prompt_len = len(generated)
                    previous_text = ""
                    phase = "answering"
                    tokens_in_current_phase = 0
                    max_tokens = max_answer_tokens
                    
                    if len(end_think_tokens) > 1 and past_key_values is not None:
                        inp_tool = torch.tensor([end_think_tokens[:-1]], dtype=torch.long).to(device)
                        start_pos_tool = len(generated) - len(end_think_tokens)
                        if hasattr(model, 'vision_encoder'):
                            _, _, past_key_values = model(inp_tool, start_pos=start_pos_tool, use_cache=True, past_key_values=past_key_values, images=None)
                        else:
                            _, _, past_key_values = model(inp_tool, start_pos=start_pos_tool, use_cache=True, past_key_values=past_key_values)
                    continue
                else:
                    break
                    
            # SentencePiece では全 ID が有効。特殊トークンは surface 文字列
            # （</tool_call> 等）へ復元され、後段の ReAct 検知にも利用される。
            current_text = tokenizer.decode(generated[prompt_len:])
            if current_text.endswith('\ufffd'):
                continue
                
            if len(current_text) > len(previous_text):
                delta = current_text[len(previous_text):]
                previous_text = current_text
                yield delta
                
            if phase == "thinking" and THINK_END in current_text:
                prompt_len = len(generated)
                previous_text = ""
                phase = "answering"
                tokens_in_current_phase = 0
                max_tokens = max_answer_tokens
