"""
tests/run_unit_tests.py

純粋ロジック部分の単体テストを一括実行するランナー (torch 不要)。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m tests.run_unit_tests
終了コード: 0 = 全テスト合格, 1 = 失敗あり
"""
import os
import sys
import unittest

# 単体テストは意図的に異常系（サンドボックス拒否・タイムアウト等）を通すため、
# その WARNING/ERROR が共有ログ (errorLog/*.log) に書かれると
# ログベース合否判定 (REQ-12/14) を汚染する。ファイルログを無効化する。
# ※ テスト対象モジュールの import (= logger 生成) より前に設定すること。
os.environ["SWI_DISABLE_FILE_LOGS"] = "1"

PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROTOTYPE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(PROTOTYPE_DIR, "tests", "unit"),
                            pattern="test_*.py",
                            top_level_dir=PROTOTYPE_DIR)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
