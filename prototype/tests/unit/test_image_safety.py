"""
tests/unit/test_image_safety.py

NEW-I1（画像編1.3）: 学習データの画像安全フィルタ。
分類器はフェイク DI で差し替え、フィルタの契約（閾値・ハッシュ照合・縮退・統計）を検証する。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_image.data import safety_filter as sf
from swi_image.data.safety_filter import ImageSafetyFilter, get_safety_filter, reset_safety_filter


class _FakeClassifier:
    """nsfw_score をキューから返すフェイク。"""

    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = 0

    def nsfw_score(self, pil_image):
        self.calls += 1
        return self._scores.pop(0)


class TestImageSafetyFilter(unittest.TestCase):
    def test_drops_at_or_above_threshold_keeps_below(self):
        f = ImageSafetyFilter(classifier=_FakeClassifier([0.9, 0.85, 0.84, 0.1]), threshold=0.85)
        self.assertFalse(f.is_safe(object()))     # 0.90 >= 0.85 → 除外
        self.assertFalse(f.is_safe(object()))     # 0.85 >= 0.85 → 除外（境界は除外側）
        self.assertTrue(f.is_safe(object()))      # 0.84 → 通過
        self.assertTrue(f.is_safe(object()))      # 0.10 → 通過
        self.assertEqual(f.stats["checked"], 4)
        self.assertEqual(f.stats["nsfw_dropped"], 2)

    def test_hash_checker_hit_drops_regardless_of_nsfw(self):
        clf = _FakeClassifier([0.0])
        f = ImageSafetyFilter(classifier=clf, hash_checker=lambda img: True)
        self.assertFalse(f.is_safe(object()))
        self.assertEqual(f.stats["hash_dropped"], 1)
        self.assertEqual(clf.calls, 0)            # ハッシュヒット時は分類器を呼ばない

    def test_hash_checker_error_is_fail_closed(self):
        def boom(img):
            raise RuntimeError("hash db unavailable")
        f = ImageSafetyFilter(classifier=_FakeClassifier([0.0]), hash_checker=boom)
        self.assertFalse(f.is_safe(object()))     # 照合できない画像は安全側（除外）
        self.assertEqual(f.stats["hash_dropped"], 1)

    def test_classifier_error_degrades_open_but_loud(self):
        class _Boom:
            def nsfw_score(self, img):
                raise RuntimeError("cuda oom")
        f = ImageSafetyFilter(classifier=_Boom())
        with self.assertLogs("image.safety", level="ERROR"):
            self.assertTrue(f.is_safe(object()))  # データロードは止めない（ERROR で明示）
        self.assertTrue(f.is_safe(object()))      # 以後は分類チェックなしで通過
        self.assertEqual(f.stats["errors"], 1)

    def test_no_classifier_no_hash_passes_everything(self):
        f = ImageSafetyFilter(classifier=None)
        self.assertTrue(f.is_safe(object()))


class TestSafetyFilterSingleton(unittest.TestCase):
    def setUp(self):
        reset_safety_filter()
        self.addCleanup(reset_safety_filter)

    def test_disabled_via_env_returns_none_with_warning(self):
        with mock.patch.dict(os.environ, {"SWI_IMG_SAFETY": "0"}):
            with self.assertLogs("image.safety", level="WARNING"):
                self.assertIsNone(get_safety_filter())

    def test_classifier_load_failure_returns_none_with_error(self):
        with mock.patch.object(sf.NSFWClassifier, "load", side_effect=RuntimeError("offline")):
            with self.assertLogs("image.safety", level="ERROR"):
                self.assertIsNone(get_safety_filter())
        # 失敗は記憶され、以後も None（毎回ロードを試みてログを吐き続けない）
        self.assertIsNone(get_safety_filter())

    def test_singleton_reused_when_available(self):
        with mock.patch.object(sf.NSFWClassifier, "load", return_value=None):
            f1 = get_safety_filter()
            f2 = get_safety_filter()
        self.assertIsNotNone(f1)
        self.assertIs(f1, f2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
