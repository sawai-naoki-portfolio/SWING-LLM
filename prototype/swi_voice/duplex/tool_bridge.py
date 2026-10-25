"""
swi_voice/duplex/tool_bridge.py
ツール呼び出し中の「つなぎ」発話 (Filler Speech During Tool Calls) [音声編10.3, 実装編7.3]

音声対話でツール呼び出し（Function Calling）が発生すると、その実行完了を待つ間に不自然な
沈黙が生じる。これを緩和するため、「少々お待ちください」等のつなぎ発話を即座に合成・再生
しつつ、裏でツール呼び出しを並行実行する。

`swi_llm.agent.base.AgentBase.process_llm_output` は同期実装（`ToolExecutor.execute_with_timeout`
がブロッキングI/O）のため、`asyncio.to_thread` でスレッドオフロードして非同期化する。ツール
呼び出しの有無は `AgentBase` と同じ `TOOL_CALL_PATTERN` で事前判定し、実際にツールが呼ばれる
場合のみつなぎ発話を挟む（呼び出しが無い通常応答でまで相槌を挟まない）。

スレッド安全性 (V-12): `AgentBase` はスレッドセーフではない（`last_call_ok` が呼び出しごとに
書き換わるインスタンス状態）。同一 agent インスタンスを複数セッションで共有して本関数を並行に
呼ぶと `last_call_ok` が競合するため、**agent はセッションごとに生成すること**。本関数自体は
共有状態を読まず、結果は戻り値のみで返す（成否の判定が必要なら
`AgentBase.process_llm_output_ex` の戻り値 ok を使う）。
"""
import asyncio
import contextlib
import random

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.chat_template import TOOL_CALL_PATTERN

logger = setup_logger("voice.tool_bridge")

DEFAULT_FILLER_PHRASES = ("少々お待ちください。", "確認しています。", "少しお時間をいただきます。")


def has_pending_tool_call(llm_output: str) -> bool:
    """LLM出力に未処理の <tool_call> タグが含まれるか（AgentBase と同じ判定基準）。"""
    return bool(TOOL_CALL_PATTERN.search(llm_output or ""))


async def handle_tool_call_with_filler(agent, llm_output, speak_fn,
                                        filler_phrases=DEFAULT_FILLER_PHRASES, timeout=None):
    """ツール呼び出しを検出したら、つなぎ発話を即時再生しつつ裏で実行する [音声編10.3]。

    Args:
        agent: `.process_llm_output(llm_text) -> str | None` を実装するオブジェクト
            （`swi_llm.agent.base.AgentBase` 相当。同期実装・セッション専用インスタンスを渡す）。
        llm_output: LLM の生出力テキスト（<tool_call> タグを含みうる）。
        speak_fn: `async def speak_fn(text) -> None`。つなぎフレーズを合成/再生する。
            None ならつなぎ無しでツールのみ実行する (V-13)。
        filler_phrases: つなぎフレーズの候補（ランダムに1つ選ぶ）。空/None ならつなぎ無しで実行。
        timeout: ツール完了を**待つ**上限秒（None なら `ToolExecutor` 側のタイムアウトに委ねる）。

    注意 — timeout の意味 (V-10): `asyncio.wait_for` は「待つのをやめる」だけで、
    `asyncio.to_thread` が起動したワーカースレッド上のツール実行そのものは**中断できない**
    （CPython のスレッドは外部からキャンセル不能）。タイムアウト後もツールはバックグラウンドで
    完走し、副作用（ファイル書き込み・外部 API 呼び出し）は発生しうる。またそのスレッドは
    既定の ThreadPoolExecutor を占有し続けるため、恒常的にタイムアウトする長時間ツールへ
    短い timeout を与え続けるとプールが枯渇する。実行自体の打ち切りは
    `ToolExecutor.execute_with_timeout`（TOOL_TIMEOUT_SEC）が担う。

    Returns:
        (tool_result, filler_spoken): tool_result は `agent.process_llm_output` の戻り値
        （ツール呼び出しが無い/失敗時は None）。filler_spoken はつなぎ発話を開始したか。

    例外: 上位からのキャンセル（バージイン等の `asyncio.CancelledError`）は握り潰さず、
    つなぎ発話タスクを止めてから伝播する (V-11)。
    """
    if not has_pending_tool_call(llm_output):
        return None, False   # ツール呼び出しが無ければ何もしない（不要な相槌を挟まない）

    async def _run_tool():
        coro = asyncio.to_thread(agent.process_llm_output, llm_output)
        if timeout is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            # V-10: 待機を打ち切っただけで、ツール実行スレッドは停止していない。
            logger.warning(
                f"handle_tool_call_with_filler: tool wait timed out after {timeout}s; "
                "the tool keeps running in a background thread and its side effects may still occur.")
            raise

    if not filler_phrases or speak_fn is None:
        # V-13: 「フレーズはあるが speak_fn=None」でも TypeError にせず、つなぎ無しで実行する。
        if not filler_phrases:
            logger.warning("handle_tool_call_with_filler: no filler_phrases provided; executing without filler.")
        else:
            logger.debug("handle_tool_call_with_filler: speak_fn is None; executing without filler.")
        try:
            return await _run_tool(), False
        except Exception:
            logger.exception("handle_tool_call_with_filler: tool execution failed.")
            return None, False

    filler_task = asyncio.create_task(speak_fn(random.choice(filler_phrases)))
    try:
        result = await _run_tool()
    except asyncio.CancelledError:
        # V-11: キャンセル（BaseException 派生）は握り潰さない。つなぎ発話も
        # 止めてから上位へ伝播する（発話タスクのリークを防ぐ）。
        filler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await filler_task
        raise
    except Exception:
        logger.exception("handle_tool_call_with_filler: tool execution failed.")
        result = None
        # 失敗が確定した後に「少々お待ちください」を最後まで再生しても意味がなく、
        # 体感レイテンシを伸ばすだけなので打ち切る (V-11)。
        filler_task.cancel()
    # V-11: 再生側（音声デバイス切断等）の例外で、成功したツール結果を失わない。
    # 正常時は自然な間を保つため再生完了を待つが、失敗はログに留めて result は必ず返す。
    try:
        await filler_task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception(
            "handle_tool_call_with_filler: filler speech failed (tool result is preserved).")
    return result, True
