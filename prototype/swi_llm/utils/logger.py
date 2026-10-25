"""
utils/logger.py

SWI-LLM 共通のロガーセットアップ。

■ ログレベルの使い分け基準 (REQ-09)
    DEBUG   : 詳細トレース。開発時のみ有用な内部状態（テンソル形状、キャッシュ
              ヒット詳細など）。本番の判断には使わない。
    INFO    : 正常フローの節目。データセットのロード完了、チェックポイントの
              保存/復元、ツール実行の成功、リクエストの受理など。
    WARNING : 想定内の異常系・フォールバック。Docker が使えず AST サンドボックスへ
              切替、サンドボックスによるコード拒否、学習時シーケンス長の超過、
              チェックポイント未発見（ランダム初期化で継続）など、処理は継続する
              が運用者が知るべき事象。
    ERROR   : 処理継続不能・結果が信頼できない事象。ツール実行の例外、logits の
              NaN 検出、データセットのロード失敗など。
    （exception() は ERROR + スタックトレース。except 節では原則こちらを使う）
    CRITICAL: プログラムの実行自体を妨げる致命的事象。学習エントリの最上位で捕捉した
              致命例外（この後 re-raise してプロセスが停止する）、サーバ起動時に有効な
              チェックポイントを 1 つも読めずランダム重みで稼働継続してしまう状態など、
              「機能提供が成立しない」レベルの障害。critical(msg, exc_info=True) で
              スタックトレースを付与できる。error.log の合否判定でも ERROR と同等に不合格扱い。

■ ホットパスでのログ出力方針 (REQ-10)
    学習ループ・モデルの forward() などの 1 ステップごとに通過するホットパスでは、
    無条件のログ出力を置かないこと。進捗は tqdm / save_interval 単位の INFO に
    集約する。ホットパス内で許されるのは「異常検知時のみ発火する条件付きログ」
    （例: NaN 検出時の ERROR）だけである。
"""
import os
import logging
import sys
from logging.handlers import RotatingFileHandler


def _ensure_utf8_console():
    """コンソール stream を UTF-8 化する（cp932 のままだとログ行が丸ごと消える）。

    Windows の既定 stdout は cp932 で、em dash (U+2014) のような cp932 に無い文字を含む
    ログを出そうとすると `UnicodeEncodeError` になり、**そのログレコードが出力されない**。
    error.log はパイプラインの合否判定の入力（REQ-12/14）なので、警告が黙って消えるのは
    「異常を見逃す」ことに直結する。

    各エントリポイントが個別に `sys.stdout.reconfigure(encoding='utf-8')` する既存の慣習
    （advanced_pipeline / *_train / *_eval など）を、全モジュールが必ず通る setup_logger に
    集約する。unittest のように reconfigure しない実行経路でも保護される。
    `errors="backslashreplace"` により、万一表現できない文字が来てもレコードを落とさず degrade する。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
            if encoding != "utf8":
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass          # 再構成できない stream（パイプ/キャプチャ等）は現状のまま使う


def setup_logger(name: str):
    """
    Setup a logger that writes to console, app.log, and error.log.
    Creates an errorLog directory if it doesn't exist.
    """
    _ensure_utf8_console()
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.hasHandlers():
        return logger
        
    # Set base logger level to lowest we care about (DEBUG)
    logger.setLevel(logging.DEBUG)

    # Parse SWI_LOG_LEVEL
    level_str = os.environ.get("SWI_LOG_LEVEL", "INFO").strip().upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    console_level = level_map.get(level_str, logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (overridable via SWI_LOG_LEVEL)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 単体テストなど「意図的に異常系を通す」実行では、共有ログファイル
    # (errorLog/*.log) を汚染しないようファイル出力を無効化できる。
    # error.log はログベース合否判定 (REQ-12/14) の入力になるため重要。
    if os.environ.get("SWI_DISABLE_FILE_LOGS", "").strip() == "1":
        return logger

    # Create errorLog directory (prototype 直下に共有。swi_llm/utils/ から 2 階層上)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prototype_dir = os.path.dirname(os.path.dirname(current_dir))
    log_dir = os.path.join(prototype_dir, "errorLog")

    # Clear old logs once per run (run_pipeline.ps1 が実行冒頭でクリアする)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. General Log File (INFO and above)
    app_log_path = os.path.join(log_dir, "app.log")
    file_handler = RotatingFileHandler(app_log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 3. Error Log File (WARNING and above)
    error_log_path = os.path.join(log_dir, "error.log")
    error_handler = RotatingFileHandler(error_log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger
