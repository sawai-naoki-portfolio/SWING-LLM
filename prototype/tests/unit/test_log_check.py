"""
ログ自動合否判定 (tests/log_check.py) の単体テスト (REQ-12/REQ-14)。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.log_check import analyze_log_text

ERROR_LINE = "[2026-07-04 18:15:37] [ERROR] [agent.executor] search_documents('x') failed."
ALLOWED_WARN_PYSPARK = "[2026-07-04 18:39:20] [WARNING] [advanced_pipeline] Could not initialize PySpark: No module named 'pyspark'. Falling back to local pure-Python pipeline."
ALLOWED_WARN_SEQLEN = "[2026-07-04 18:46:20] [WARNING] [generate_stream] Generation position (513) exceeded trained sequence length (512). Output quality may degrade beyond this point."
UNEXPECTED_WARN = "[2026-07-04 18:00:00] [WARNING] [web_models] No checkpoint found, using random weights."
TRACEBACK_LINES = "Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError: boom"


class TestAnalyzeLogText(unittest.TestCase):
    def test_empty_log_passes(self):
        report = analyze_log_text("")
        self.assertTrue(report.passed)

    def test_error_fails(self):
        report = analyze_log_text(ERROR_LINE)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.errors), 1)

    def test_allowed_warnings_pass(self):
        report = analyze_log_text(ALLOWED_WARN_PYSPARK + "\n" + ALLOWED_WARN_SEQLEN)
        self.assertTrue(report.passed)
        self.assertEqual(len(report.allowed_warnings), 2)
        self.assertEqual(len(report.unexpected_warnings), 0)

    def test_unexpected_warning_fails(self):
        report = analyze_log_text(UNEXPECTED_WARN)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.unexpected_warnings), 1)

    def test_strict_mode_fails_on_allowed_warning(self):
        report = analyze_log_text(ALLOWED_WARN_PYSPARK, strict=True)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.unexpected_warnings), 1)
        self.assertEqual(len(report.allowed_warnings), 0)

    def test_traceback_continuation_lines_not_counted(self):
        # トレースバックの継続行は独立イベントとしてカウントしない
        report = analyze_log_text(ERROR_LINE + "\n" + TRACEBACK_LINES)
        self.assertEqual(len(report.errors), 1)

    def test_mixed_log(self):
        text = "\n".join([ALLOWED_WARN_PYSPARK, ERROR_LINE, UNEXPECTED_WARN, ALLOWED_WARN_SEQLEN])
        report = analyze_log_text(text)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(len(report.unexpected_warnings), 1)
        self.assertEqual(len(report.allowed_warnings), 2)

    def test_info_lines_ignored(self):
        report = analyze_log_text("[2026-07-04 12:00:00] [INFO] [x] normal flow")
        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
