"""
tests/run_voice_unit_tests.py

swi_voice の単体テスト（tests/unit/test_voice_*.py）を一括実行するランナー。

V-16: run.ps1 の voice ステージは従来 2 モジュール（codec / ar_lm）だけを名指しで実行して
おり、新規追加されたテストファイルが `run.bat voice TEST` の網から漏れていた。llm ステージの
`tests.run_unit_tests` と同じ discover 方式（ただし test_voice_* に限定）にすることで、
以後テストファイルを追加するだけで自動的に実行対象へ入る。

使い方 (prototype ディレクトリで実行):
    .venv\\Scripts\\python.exe -m tests.run_voice_unit_tests
終了コード: 0 = 全テスト合格, 1 = 失敗あり
"""
import os
import sys
import unittest

# 単体テストは意図的に異常系（空入力・タイムアウト等）を通すため、その WARNING/ERROR が
# 共有ログ (errorLog/*.log) に書かれるとログベース合否判定 (REQ-12/14) を汚染する。
# ※ テスト対象モジュールの import (= logger 生成) より前に設定すること。
os.environ["SWI_DISABLE_FILE_LOGS"] = "1"

PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROTOTYPE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(PROTOTYPE_DIR, "tests", "unit"),
                            pattern="test_voice_*.py",
                            top_level_dir=PROTOTYPE_DIR)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
