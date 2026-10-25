"""
tests/webui_test.py

Web UI の包括的 E2E テスト (REQ-11 / REQ-12)。

カバレッジ:
  - 基本推論パス（テキスト / 画像のみ / 画像+テキスト / CoT）
  - 全ペルソナ (jarvis, navi) × 全ツール (weather, calculate, search, execute_python)
  - 異常系: 空入力、巨大入力（上限超過）、不正な画像データ、同時リクエスト上限超過
  - ログベース自動判定 (REQ-12): テスト実行中に error.log へ想定外の
    WARNING / ERROR が追記された場合、テスト全体を不合格として扱う。

前提: web.app サーバーが http://127.0.0.1:5000 で起動していること
（run_pipeline.ps1 が TEST/SMOKE モードで自動起動する）。
"""
import urllib.request
import urllib.error
import json
import time
import sys
import os
import threading

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure we log to the right places
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swi_llm.utils.logger import setup_logger
from swi_llm.utils.chat_template import THINK_START
from swi_llm.utils.constants import (
    MAX_PROMPT_LENGTH, MAX_CONCURRENT_INFERENCE,
    PERSONA_JARVIS, PERSONA_NAVI, WEB_PORT,
)
from tests import log_check

logger = setup_logger("webui_test")

BASE_URL = f"http://127.0.0.1:{WEB_PORT}"
DUMMY_IMAGE_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
INVALID_IMAGE_B64 = "data:image/png;base64,!!!this-is-not-base64!!!"

# サーバー側 (web/routes/inference.py) がツール自動実行時に出す表示文
TOOL_INDICATOR = "意図を検知しツールを自動実行しました"

DEFAULT_TIMEOUT_SEC = 180
COT_TIMEOUT_SEC = 300

ALL_PERSONAS = [PERSONA_JARVIS, PERSONA_NAVI]

# ツールごとの (プロンプト, 期待文字列 or None)。
# weather / calculate / search はルールベースの意図検知で必ず自動実行される。
# execute_python は意図検知の対象外のため、応答が返ること（非空）のみを確認する。
TOOL_TEST_CASES = {
    "get_weather": ("東京の天気を教えて", TOOL_INDICATOR),
    "calculate": ("123 * 456 の計算をお願い", TOOL_INDICATOR),
    "search_documents": ("SWI-LLMのアーキテクチャについて検索して教えて", TOOL_INDICATOR),
    "execute_python": ("Pythonでフィボナッチ数列を出力するコードを書いて実行してください", None),
}

PROMPT_GREETING = "こんにちは、元気ですか？"
PROMPT_IMAGE_QUESTION = "この画像は何色ですか？"
PROMPT_COT = "フェルマーの最終定理の証明を計算して"
PROMPT_CONCURRENCY = "自己紹介をしてください"


def wait_for_server():
    max_retries = 30
    for i in range(max_retries):
        try:
            req = urllib.request.Request(BASE_URL + "/")
            with urllib.request.urlopen(req, timeout=1):
                logger.info("Web UI server is up and running!")
                return True
        except Exception:
            time.sleep(1)
    logger.error("Timed out waiting for Web UI server to start.")
    return False


def stream_request(prompt, images=None, use_cot=False, persona=PERSONA_JARVIS,
                   timeout=DEFAULT_TIMEOUT_SEC):
    """/api/generate_stream にリクエストし、ストリーム全文を返す。
    HTTP エラー時は urllib.error.HTTPError を送出する。"""
    url = f"{BASE_URL}/api/generate_stream"
    data = {
        "prompt": prompt,
        "use_cot": use_cot,
        "images": images or [],
        "persona": persona,
        "client": "web",
    }
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    response_text = ""
    with urllib.request.urlopen(req, timeout=timeout) as response:
        while True:
            chunk = response.read(1024)
            if not chunk:
                break
            response_text += chunk.decode('utf-8', errors='replace')
    return response_text.strip().encode('utf-8', 'replace').decode('utf-8')


def check_response(name, prompt, images=None, use_cot=False, persona=PERSONA_JARVIS,
                   expected_in_response=None, timeout=DEFAULT_TIMEOUT_SEC):
    """正常系テスト: 応答が非空で、期待文字列（あれば）を含むこと。"""
    logger.info(f"[{name}] Prompt: '{prompt[:60]}', persona={persona}, cot={use_cot}, images={len(images or [])}")
    try:
        text = stream_request(prompt, images=images, use_cot=use_cot, persona=persona, timeout=timeout)
    except Exception as e:
        logger.error(f"[{name}] Request failed: {e}")
        return False

    logger.info(f"[{name}] Response ({len(text)} chars): {text[:200]}")
    if not text:
        logger.error(f"[{name}] Response is empty.")
        return False
    if expected_in_response and expected_in_response not in text:
        logger.error(f"[{name}] Expected '{expected_in_response}' not found in response.")
        return False
    return True


def check_http_error(name, prompt, images=None, expected_status=400):
    """異常系テスト: 指定の HTTP ステータスで拒否されること。"""
    logger.info(f"[{name}] Expecting HTTP {expected_status}...")
    try:
        stream_request(prompt, images=images, timeout=60)
        logger.error(f"[{name}] Request unexpectedly succeeded (expected HTTP {expected_status}).")
        return False
    except urllib.error.HTTPError as e:
        if e.code == expected_status:
            logger.info(f"[{name}] Correctly rejected with HTTP {e.code}.")
            return True
        logger.error(f"[{name}] Unexpected HTTP status {e.code} (expected {expected_status}).")
        return False
    except Exception as e:
        logger.error(f"[{name}] Unexpected error type: {e}")
        return False


def check_invalid_image(name):
    """異常系テスト: 不正な画像データはストリーム内のエラー通知で処理されること。"""
    logger.info(f"[{name}] Sending broken base64 image...")
    try:
        text = stream_request(PROMPT_IMAGE_QUESTION, images=[INVALID_IMAGE_B64], timeout=60)
    except Exception as e:
        logger.error(f"[{name}] Request failed unexpectedly: {e}")
        return False
    if "Error" in text or "無効な画像" in text:
        logger.info(f"[{name}] Invalid image correctly reported: {text[:120]}")
        return True
    logger.error(f"[{name}] No error notification for invalid image. Response: {text[:200]}")
    return False


def check_concurrency(name, num_requests=None):
    """異常系テスト: 同時リクエスト数が上限 (semaphore) を超えても、
    全リクエストがエラーなく最終的に処理されること（キューイングされる設計）。"""
    if num_requests is None:
        num_requests = MAX_CONCURRENT_INFERENCE + 2
    logger.info(f"[{name}] Sending {num_requests} concurrent requests "
                f"(limit={MAX_CONCURRENT_INFERENCE})...")
    results = [None] * num_requests

    def worker(idx):
        try:
            text = stream_request(PROMPT_CONCURRENCY, timeout=COT_TIMEOUT_SEC)
            results[idx] = bool(text)
        except Exception as e:
            logger.error(f"[{name}] Concurrent request {idx} failed: {e}")
            results[idx] = False

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_requests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=COT_TIMEOUT_SEC + 30)

    ok = all(r is True for r in results)
    if ok:
        logger.info(f"[{name}] All {num_requests} concurrent requests completed successfully.")
    else:
        logger.error(f"[{name}] Concurrency results: {results}")
    return ok


def check_feedback_endpoint(name):
    """WebUI のフィードバック収集 (/api/feedback, §11.2) が 200 を返すこと。"""
    logger.info(f"[{name}] Posting feedback...")
    url = f"{BASE_URL}/api/feedback"
    data = {"query": "テスト質問", "response": "テスト回答", "rating": 1}
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        ok = body.get("status") == "ok"
        logger.info(f"[{name}] Feedback stored: {body}")
        return ok
    except Exception as e:
        logger.error(f"[{name}] Feedback request failed: {e}")
        return False


def check_best_of_n(name):
    """WebUI 高品質モードの裏側 (/api/generate best_of_n, §8.2) が非空応答を返すこと。"""
    logger.info(f"[{name}] Requesting PRM Best-of-N generation...")
    url = f"{BASE_URL}/api/generate"
    data = {"prompt": "自己紹介してください", "use_cot": False, "images": [],
            "persona": PERSONA_JARVIS, "best_of_n": 2}
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=COT_TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode('utf-8'))
        text = body.get("response", "")
        ok = bool(text)
        logger.info(f"[{name}] Best-of-N response ({len(text)} chars): {text[:120]}")
        return ok
    except Exception as e:
        logger.error(f"[{name}] Best-of-N request failed: {e}")
        return False


def check_rate_limit(name):
    """SEC-12: レート制限が実際に 429 を返すこと（WebUI ロードマップ step5 / 監査 G-4）。

    修正前は生成系エンドポイントが完全に無制限だった。ここでは GPU を一切消費しない
    軽量な `/api/llm/status`（default クラス: burst=60, 300/min）を上限超えまで叩き、
    429 と `Retry-After` が返ることを確認する。

    **必ず他のチェックの後に実行すること**（バケットを消費するため）。生成系のバケットは
    別クラスなので、この検査は他の推論系チェックに影響しない。
    """
    from swi_llm.utils.constants import RATE_LIMIT_ENABLED, RATE_LIMIT_RULES

    if not RATE_LIMIT_ENABLED:
        logger.warning(f"[{name}] SWI_RATE_LIMIT=0 のためレート制限の検証をスキップします。")
        return True

    burst = RATE_LIMIT_RULES["default"][1]
    attempts = burst + 20
    logger.info(f"[{name}] Sending {attempts} requests to /api/llm/status (burst={burst})...")

    saw_429, retry_after = False, None
    for _ in range(attempts):
        req = urllib.request.Request(f"{BASE_URL}/api/llm/status")
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except urllib.error.HTTPError as e:
            if e.code == 429:
                saw_429 = True
                retry_after = e.headers.get("Retry-After")
                break
        except Exception as e:
            logger.error(f"[{name}] Unexpected transport error: {e}")
            return False

    if not saw_429:
        logger.error(f"[{name}] {attempts} 連射しても 429 が返りませんでした "
                     "（レート制限が効いていない = 生成系が無制限に叩ける）")
        return False
    if not retry_after:
        logger.error(f"[{name}] 429 は返ったが Retry-After ヘッダがありません")
        return False
    logger.info(f"[{name}] Rate limit fired as designed (429, Retry-After={retry_after}).")
    return True


def main():
    logger.info("Starting Web UI Comprehensive E2E Tests...")

    if not wait_for_server():
        sys.exit(1)

    time.sleep(2)  # Give it an extra moment to fully initialize

    # REQ-12: テスト開始時点の error.log オフセットを記録し、
    # テスト中に追記された想定外の WARNING/ERROR を最後に判定する。
    log_offset = log_check.current_offset()

    results = {}

    # ---- 1. 基本推論パス ----
    results["basic_text"] = check_response("basic_text", PROMPT_GREETING)
    results["image_only"] = check_response("image_only", "", images=[DUMMY_IMAGE_B64])
    results["image_and_text"] = check_response("image_and_text", PROMPT_IMAGE_QUESTION, images=[DUMMY_IMAGE_B64])

    # ---- 2. 全ペルソナ × 全ツール (REQ-11) ----
    for persona in ALL_PERSONAS:
        for tool_name, (prompt, expected) in TOOL_TEST_CASES.items():
            test_name = f"tool_{tool_name}_{persona}"
            results[test_name] = check_response(
                test_name, prompt, persona=persona, expected_in_response=expected)

    # ---- 3. Dynamic Thinking Budget (CoT) ----
    results["cot_thinking"] = check_response(
        "cot_thinking", PROMPT_COT, use_cot=True,
        expected_in_response=THINK_START, timeout=COT_TIMEOUT_SEC)

    # ---- 4. 異常系 (REQ-11) ----
    results["err_empty_input"] = check_http_error("err_empty_input", "", images=[])
    results["err_oversized_input"] = check_http_error(
        "err_oversized_input", "あ" * (MAX_PROMPT_LENGTH + 1))
    results["err_invalid_image"] = check_invalid_image("err_invalid_image")
    results["err_concurrency_over_limit"] = check_concurrency("err_concurrency_over_limit")

    # ---- 5. WebUI が使う追加機能のエンドポイント ----
    results["feedback_endpoint"] = check_feedback_endpoint("feedback_endpoint")  # §11.2
    results["best_of_n_endpoint"] = check_best_of_n("best_of_n_endpoint")        # §8.2

    # ---- 5b. レート制限 (SEC-12)。バケットを消費するので推論系チェックの後に置く ----
    results["rate_limit_enforced"] = check_rate_limit("rate_limit_enforced")

    # ---- 6. ログベース自動判定 (REQ-12) ----
    log_report = log_check.analyze_log_file(since_offset=log_offset)
    results["log_no_unexpected_events"] = log_report.passed
    if not log_report.passed:
        logger.error("Unexpected WARNING/ERROR log events detected during E2E run:")
        for line in log_report.summary_lines():
            logger.error(f"  {line}")
    else:
        logger.info(f"Log check OK (allowed warnings: {len(log_report.allowed_warnings)})")

    # ---- サマリ ----
    failed = [name for name, ok in results.items() if not ok]
    logger.info("=" * 60)
    logger.info(f"E2E SUMMARY: {len(results) - len(failed)}/{len(results)} passed")
    for name, ok in results.items():
        logger.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    logger.info("=" * 60)

    if not failed:
        logger.info("All Comprehensive Web UI E2E tests passed successfully!")
        sys.exit(0)
    else:
        logger.error(f"{len(failed)}/{len(results)} Web UI E2E tests failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
