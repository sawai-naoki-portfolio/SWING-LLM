"""setup_logger のコンソール encoding 回帰テスト。

**背景**: Windows の既定 stdout は cp932。em dash (U+2014) のように cp932 に存在しない文字を
含むログを出すと `UnicodeEncodeError` になり、**そのログレコードが丸ごと出力されない**。
error.log はパイプラインの合否判定の入力（REQ-12/14）なので、WARNING/ERROR が黙って
消えるのは「異常を見逃す」ことに直結する（実際にレート制限の WARNING が消えていた）。

ここでは「cp932 に無い文字を含む警告でもレコードが失われない」ことを固定する。
"""
import io
import logging
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.utils.logger import _ensure_utf8_console, setup_logger

# cp932 に無い文字たち（U+2014 EM DASH は典型例。cp932 にあるのは U+2015 HORIZONTAL BAR）。
_NON_CP932 = "— ✅ 𝄞"


class TestConsoleIsUtf8(unittest.TestCase):
    def test_stdout_and_stderr_are_utf8_after_setup(self):
        setup_logger("test.encoding.setup")
        for stream in (sys.stdout, sys.stderr):
            encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
            self.assertEqual(encoding, "utf8", f"{stream!r} が UTF-8 になっていない")

    def test_non_cp932_text_survives_encoding(self):
        # 実際に cp932 で encode しようとすると失敗する文字であることを確認しておく
        # （この前提が崩れたらテストの意味がなくなる）。
        with self.assertRaises(UnicodeEncodeError):
            _NON_CP932.encode("cp932")
        self.assertTrue(_NON_CP932.encode("utf-8"))


class TestLogRecordIsNotDropped(unittest.TestCase):
    """cp932 stream を模して、レコードが失われないことを確かめる。"""

    def test_cp932_stream_would_drop_the_record(self):
        # 対照実験: errors 指定なしの cp932 stream に書くと logging がエラーを出す（= 消える）。
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict")
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger("test.encoding.cp932")
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        with self.assertRaises(UnicodeEncodeError):
            handler.format(logging.LogRecord("x", logging.WARNING, __file__, 1, _NON_CP932, None, None))
            stream.write(_NON_CP932)
            stream.flush()

    def test_backslashreplace_stream_keeps_the_record(self):
        # 本番の設定（errors="backslashreplace"）なら、表現できない文字が来てもレコードは残る。
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="backslashreplace")
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger = logging.getLogger("test.encoding.replace")
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        logger.warning("rate limited %s", _NON_CP932)
        handler.flush()
        written = stream.buffer.getvalue().decode("cp932", errors="replace")
        self.assertIn("rate limited", written, "警告レコードが失われた")

    def test_ensure_utf8_console_is_idempotent(self):
        _ensure_utf8_console()
        _ensure_utf8_console()
        encoding = (sys.stdout.encoding or "").lower().replace("-", "")
        self.assertEqual(encoding, "utf8")


if __name__ == "__main__":
    unittest.main()
