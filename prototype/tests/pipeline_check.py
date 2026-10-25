"""
tests/pipeline_check.py

run_pipeline.ps1 実行後に、各段階の成功基準を自動判定するCLI (REQ-13 / REQ-14)。

判定項目:
  1. チェックポイント生成: pretrain / sft / vision / dpo / grpo の各 .pt が存在し、
     パイプライン開始マーカー (errorLog/.pipeline_start) 以降に更新されていること。
  2. 損失の健全性: errorLog/app.log の "Validation Loss: X" / "Val Loss: X" が
     すべて有限値（NaN / inf でない）であること。
  3. 各学習ステージの完了: app.log に各ステージの完了ログが存在すること。
  4. WebUI E2E: app.log に E2E 全件合格のログが存在すること (--skip-e2e で省略可)。
  5. ログ健全性: error.log に ERROR および想定外 WARNING がないこと
     (tests/log_check.py に委譲。--strict で WARNING 全件を不合格要因にできる)。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m tests.pipeline_check                # SWI_MODE から自動判別
    .venv\\Scripts\\python.exe -m tests.pipeline_check --mode SMOKE
    .venv\\Scripts\\python.exe -m tests.pipeline_check --mode TEST --strict
終了コード: 0 = 全項目合格, 1 = 不合格あり
"""
import argparse
import math
import os
import re
import sys

# Add prototype directory to sys.path
PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROTOTYPE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from tests.log_check import analyze_log_file, DEFAULT_LOG_PATH
from swi_llm.utils.constants import (
    CKPT_PRETRAIN, CKPT_SFT, CKPT_VISION, CKPT_DPO, CKPT_GRPO, EXT_PT,
    MODE_TEST, MODE_SMOKE, MODE_PROD, MODE_MAX,
)

APP_LOG_PATH = os.path.join(PROTOTYPE_DIR, "errorLog", "app.log")
START_MARKER_PATH = os.path.join(PROTOTYPE_DIR, "errorLog", ".pipeline_start")

# パイプラインが生成すべきチェックポイント (REQ-13)
REQUIRED_CKPT_BASES = [CKPT_PRETRAIN, CKPT_SFT, CKPT_VISION, CKPT_DPO, CKPT_GRPO]

# app.log に現れるべきステージ完了ログ
STAGE_COMPLETION_MARKERS = [
    ("Pretrain (val loss logged)", r"\[train\] Epoch \d+ completed"),
    ("SFT complete", r"SFT Training complete"),
    ("DPO complete", r"DPO Training complete"),
    ("GRPO complete", r"GRPO Training complete"),
]

E2E_PASS_MARKER = r"All Comprehensive Web UI E2E tests passed successfully!"

LOSS_PATTERN = re.compile(r"(?:Validation Loss|Val Loss):\s*([^\s,]+)")


class CheckResult:
    def __init__(self, name, passed, detail=""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def line(self):
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f" - {self.detail}" if self.detail else "")


def resolve_suffix(mode):
    return f"_{mode}" if mode and mode != MODE_PROD else ""


def read_app_log():
    if not os.path.exists(APP_LOG_PATH):
        return ""
    with open(APP_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def check_checkpoints(suffix, start_time):
    """必須チェックポイントの存在と、パイプライン開始以降の更新を確認する。"""
    results = []
    for base in REQUIRED_CKPT_BASES:
        path = os.path.join(PROTOTYPE_DIR, f"{base}{suffix}{EXT_PT}")
        if not os.path.exists(path):
            results.append(CheckResult(f"Checkpoint exists: {os.path.basename(path)}", False, "file not found"))
            continue
        mtime = os.path.getmtime(path)
        if start_time is not None and mtime < start_time:
            results.append(CheckResult(
                f"Checkpoint fresh: {os.path.basename(path)}", False,
                "not updated by this pipeline run (stale file)"))
        else:
            results.append(CheckResult(f"Checkpoint OK: {os.path.basename(path)}", True))
    return results


def check_losses(app_log_text):
    """app.log 中の全 val loss が有限値であることを確認する (REQ-13: loss が NaN でない)。"""
    values = LOSS_PATTERN.findall(app_log_text)
    if not values:
        return [CheckResult("Loss values logged", False, "no 'Validation Loss'/'Val Loss' lines found in app.log")]
    bad = []
    for v in values:
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                bad.append(v)
        except ValueError:
            bad.append(v)
    if bad:
        return [CheckResult(f"All losses finite ({len(values)} values)", False, f"invalid values: {bad[:5]}")]
    return [CheckResult(f"All losses finite ({len(values)} values)", True)]


def check_stages(app_log_text):
    results = []
    for name, pattern in STAGE_COMPLETION_MARKERS:
        found = re.search(pattern, app_log_text) is not None
        results.append(CheckResult(f"Stage: {name}", found, "" if found else f"pattern not found: {pattern}"))
    return results


def check_e2e(app_log_text):
    found = re.search(E2E_PASS_MARKER, app_log_text) is not None
    return [CheckResult("WebUI E2E all passed", found,
                        "" if found else "E2E pass marker not found in app.log")]


def check_error_log(strict):
    report = analyze_log_file(DEFAULT_LOG_PATH, strict=strict)
    detail_parts = [f"errors={len(report.errors)}", f"unexpected_warnings={len(report.unexpected_warnings)}",
                    f"allowed_warnings={len(report.allowed_warnings)}"]
    result = CheckResult(f"error.log clean (strict={strict})", report.passed, ", ".join(detail_parts))
    return [result], report


def main():
    parser = argparse.ArgumentParser(description="Judge pipeline success criteria (REQ-13/REQ-14).")
    parser.add_argument("--mode", default=os.environ.get("SWI_MODE", "").strip().upper() or MODE_TEST,
                        choices=[MODE_TEST, MODE_SMOKE, MODE_PROD, MODE_MAX])
    parser.add_argument("--strict", action="store_true", help="Fail on ANY warning in error.log")
    parser.add_argument("--skip-e2e", action="store_true", help="Do not require the WebUI E2E pass marker")
    args = parser.parse_args()

    suffix = resolve_suffix(args.mode)
    start_time = os.path.getmtime(START_MARKER_PATH) if os.path.exists(START_MARKER_PATH) else None
    app_log_text = read_app_log()

    results = []
    results += check_checkpoints(suffix, start_time)
    results += check_losses(app_log_text)
    results += check_stages(app_log_text)
    if not args.skip_e2e:
        results += check_e2e(app_log_text)
    log_results, log_report = check_error_log(args.strict)
    results += log_results

    print("=" * 70)
    print(f"Pipeline check  mode={args.mode}  suffix='{suffix}'  "
          f"start_marker={'found' if start_time else 'none (freshness check skipped)'}")
    print("=" * 70)
    for r in results:
        print(r.line())
    if not log_report.passed:
        print("-" * 70)
        print("error.log details:")
        for line in log_report.summary_lines():
            print(line)

    all_passed = all(r.passed for r in results)
    print("-" * 70)
    print(f"PIPELINE CHECK RESULT: {'PASS' if all_passed else 'FAIL'} "
          f"({sum(r.passed for r in results)}/{len(results)} checks passed)")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
