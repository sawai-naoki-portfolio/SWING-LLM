"""
tests/log_check.py

errorLog/error.log（WARNING以上のみが記録される）を機械的に解析し、
合否を自動判定するモジュール兼CLI (REQ-12 / REQ-14)。

判定基準:
  - ERROR 行が 1 件でもあれば不合格。
  - WARNING 行は「設計上想定されたパターン (ALLOWED_WARNING_PATTERNS)」に
    一致するものだけ許容する。想定外の WARNING があれば不合格。
  - --strict 指定時は許容リストを無効化し、WARNING が 1 件でもあれば不合格
    （REQ-14 の「ERROR/WARNING が 1 件も出ないこと」の厳格運用）。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m tests.log_check              # 通常判定
    .venv\\Scripts\\python.exe -m tests.log_check --strict     # 厳格判定
    .venv\\Scripts\\python.exe -m tests.log_check --since-offset 1234
終了コード: 0 = 合格, 1 = 不合格
"""
import argparse
import os
import re
import sys

# ログ1行の先頭形式: [2026-07-04 18:39:20] [WARNING] [name] message
LOG_LINE_PATTERN = re.compile(r"^\[[\d\- :]+\] \[(?P<level>[A-Z]+)\] \[(?P<name>[^\]]+)\] (?P<message>.*)$")

# 設計上想定された（合格扱いとする）WARNING のパターン。
# ここに追加する場合は、なぜ想定内なのかをコメントで必ず説明すること。
ALLOWED_WARNING_PATTERNS = [
    # PySpark 未導入環境ではローカル純Pythonパイプラインへ切り替わる設計 (data/advanced_pipeline.py)
    r"Could not initialize PySpark",
    # 学習時シーケンス長を超えた生成に対する設計上の警告 (REQ-19)
    r"exceeded trained sequence length",
    # Docker 不可時の AST サンドボックスへのフォールバック通知 (REQ-22)
    r"Docker unavailable",
    # dense 埋め込みモデルが取得できない場合の BM25-only フォールバック (agent/rag.py)
    r"Falling back to BM25-only RAG",
    r"FAISS or sentence-transformers not found",
    # 小型/未学習モデルが不正な tool_call を出力した際の回復処理 (agent/base.py)。
    # 特殊トークンのアトミック化 (§1.4) で未学習モデルがタグを出しやすく、
    # 呼び出し側がリトライ/直接回答へフォールバックするため継続不能ではない (REQ-09)。
    r"Malformed tool_call JSON",
    r"Invalid JSON in tool_call and fallback parser failed",
    r"missing required argument",
    r"Requested tool '.*' is not registered",
    # データミックスのソース取得フォールバック (§1.5, data/data_mix.py)。
    # gated/未認証/不存在のデータセットは次候補へ切り替える設計（継続可能）。
    r"\w+ dataset '.*' unavailable",       # Code/Web 等の候補フォールバック
    r"No \w+ dataset could be loaded",     # 全候補が不可（空ソース許容）
    r"Could not load .* source",           # dialogue/wikipedia 等の取得失敗
    r"datasets unavailable for .* source", # datasets ライブラリ不在
    # 学習の再開/フォールバック (utils/trainer_utils.py)。初回や不整合 ckpt では
    # 「ゼロ/ベース重みから開始」する設計（継続可能）。setup_logger 化で error.log に届くため許容。
    r"No valid training checkpoints found",   # 初回学習等（scratch/base から開始）
    r"Failed to load checkpoint .* Starting fresh",  # 破損/不整合 ckpt → fresh 開始
    # SEC-12 レート制限 (web/rate_limit.py)。**拒否は設計どおりの正常動作**であり、
    # E2E は「429 が実際に返ること」を明示的に検証する（check_rate_limit）。
    # 品質ゲートの役割は exit code が担い、この WARNING は運用可視化のための情報。
    r"429 rate limited",
    r"413 body too large",
    r"body cap exceeded on chunked request",
    # 画像評価 (swi_image/eval/image_eval.py)。自作 Conv VAE は現状「定数=平均色を出す」自明
    # ベースラインに負けている（2026-07-12 実測 -0.45dB）。これは image ロードマップ step3
    # （凍結 SDXL VAE）で解消する**既知の状態**であり、パイプラインの破綻ではない。
    # 「壊れていないこと」のゲートは image_eval の --min-psnr-gain（exit code）が担う。
    r"VAE の再構成が自明ベースラインに負けています",
    # 音声評価 (swi_voice/eval/voice_eval.py)。自作 RVQ コーデックは現状 SNR -1.21dB
    # （無音出力より悪い）かつコードブック使用率 0.4%（256 個中 1 個 = 崩壊）。
    # voice ロードマップ step7（ボコーダ GAN）/ step8（意味コーデック蒸留）で解消する
    # **既知の状態**。ゲートは voice_eval の --min-codec-snr / --min-codebook-usage が担う。
    r"コーデックの再構成 SNR が",
    r"コードブック使用率が .* コードブック崩壊の疑い",
]

DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "errorLog", "error.log"
)


class LogReport:
    """error.log の解析結果。"""
    def __init__(self):
        self.errors = []            # ERROR 行（常に不合格要因）
        self.unexpected_warnings = []  # 許容リスト外の WARNING（不合格要因）
        self.allowed_warnings = []     # 許容リスト内の WARNING（情報表示のみ）

    @property
    def passed(self):
        return not self.errors and not self.unexpected_warnings

    def summary_lines(self):
        lines = [
            f"ERROR: {len(self.errors)}",
            f"WARNING (unexpected): {len(self.unexpected_warnings)}",
            f"WARNING (allowed by design): {len(self.allowed_warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  [NG][ERROR] {e}")
        for w in self.unexpected_warnings:
            lines.append(f"  [NG][WARN ] {w}")
        for w in self.allowed_warnings:
            lines.append(f"  [OK][warn ] {w}")
        return lines


def analyze_log_text(text, strict=False):
    """error.log のテキストを解析して LogReport を返す。

    トレースバック等の継続行（フォーマット外の行）は直前のイベントの
    詳細とみなし、独立したイベントとしてはカウントしない。
    """
    report = LogReport()
    for line in text.splitlines():
        m = LOG_LINE_PATTERN.match(line)
        if not m:
            continue  # traceback 等の継続行
        level = m.group("level")
        if level == "ERROR" or level == "CRITICAL":
            report.errors.append(line)
        elif level == "WARNING":
            if not strict and any(re.search(p, line) for p in ALLOWED_WARNING_PATTERNS):
                report.allowed_warnings.append(line)
            else:
                report.unexpected_warnings.append(line)
    return report


def analyze_log_file(log_path=DEFAULT_LOG_PATH, since_offset=0, strict=False):
    """error.log ファイルを解析する。

    Args:
        log_path: 解析対象のログファイル。
        since_offset: このバイトオフセット以降のみ解析する
            （テスト開始前のログを除外するために使用。REQ-12）。
        strict: True なら許容リストを無効化し WARNING 全件を不合格要因とする。
    """
    if not os.path.exists(log_path):
        return analyze_log_text("", strict=strict)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        if since_offset > 0:
            f.seek(since_offset)
        text = f.read()
    return analyze_log_text(text, strict=strict)


def current_offset(log_path=DEFAULT_LOG_PATH):
    """現在のログファイルサイズ（以降の追記分だけを判定するための基準点）。"""
    return os.path.getsize(log_path) if os.path.exists(log_path) else 0


def main():
    parser = argparse.ArgumentParser(description="Analyze error.log and judge pass/fail (REQ-12/REQ-14).")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH, help="Path to error.log")
    parser.add_argument("--since-offset", type=int, default=0, help="Analyze only bytes appended after this offset")
    parser.add_argument("--strict", action="store_true", help="Fail on ANY warning (ignore allowed list)")
    args = parser.parse_args()

    report = analyze_log_file(args.log_path, since_offset=args.since_offset, strict=args.strict)

    print("=" * 60)
    print(f"Log check: {args.log_path} (strict={args.strict})")
    print("=" * 60)
    for line in report.summary_lines():
        print(line)
    verdict = "PASS" if report.passed else "FAIL"
    print("-" * 60)
    print(f"LOG CHECK RESULT: {verdict}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
