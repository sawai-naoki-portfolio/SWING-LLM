"""
tests/unit/test_quality_distill.py

NEW-L1（01_LLM §2）: Teacher スコア → 軽量分類器の蒸留品質フィルタ。
完全自作方針（2026-10-25）で教師は自作ヒューリスティックのみ。蒸留の骨格（採点・学習・
契約適合・成果物の自動検出）と、外部 LLM 教師の旧成果物を拒否することを検証する。
"""
import os
import pickle
import sys
import tempfile
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_llm.data import quality_distill
from swi_llm.data.quality_distill import (
    TeacherScorer, StudentClassifier, DistilledQualityFilter, get_distilled_filter,
)

_EDU = ("量子力学とは、原子や電子など微視的な系を記述する物理学の理論である。"
        "例えば電子のエネルギーは離散的な値をとる。したがって古典力学では説明できない。")
_SPAM = "今すぐクリック！ 登録 登録 会員登録 http://example.com 今すぐ！！ " * 3


class TestTeacherScorer(unittest.TestCase):
    def test_scores_with_self_made_heuristic(self):
        t = TeacherScorer()
        self.assertEqual(t.kind, "heuristic")
        edu, spam = t.score(_EDU), t.score(_SPAM)
        for score in (edu, spam):
            self.assertIsInstance(score, int)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 5)
        self.assertGreater(edu, spam)

    def test_no_external_llm_teacher_path(self):
        """外部 LLM（OpenAI 互換 / Qwen）へ問い合わせる経路が復活していないこと。"""
        src = open(quality_distill.__file__, encoding="utf-8").read()
        for marker in ("SWI_OPENAI", "chat/completions", "qwen", "requests"):
            self.assertNotIn(marker, src)


class TestStudentAndFilter(unittest.TestCase):
    def _trained(self):
        texts = [_EDU + str(i) for i in range(10)] + [_SPAM + str(i) for i in range(10)]
        labels = [1] * 10 + [0] * 10
        return StudentClassifier().fit(texts, labels)

    def test_student_separates_families(self):
        s = self._trained()
        self.assertGreater(s.predict_proba_positive([_EDU])[0], 0.5)
        self.assertLess(s.predict_proba_positive([_SPAM])[0], 0.5)

    def test_save_load_roundtrip(self):
        s = self._trained()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "student.pkl")
            s.save(path)
            loaded = StudentClassifier.load(path)
            self.assertAlmostEqual(float(loaded.predict_proba_positive([_EDU])[0]),
                                   float(s.predict_proba_positive([_EDU])[0]), places=6)

    def test_filter_matches_pipeline_scorer_contract(self):
        f = DistilledQualityFilter(self._trained())
        score = f.score(_EDU)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 5.0)
        self.assertGreater(score, f.score(_SPAM))
        self.assertEqual(f.score(""), 0.0)
        self.assertEqual(f.score(None), 0.0)

    def test_get_distilled_filter_detects_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "student.pkl")
            with mock.patch.dict(os.environ, {"SWI_DISTILLED_FILTER": path}):
                self.assertIsNone(get_distilled_filter())      # 成果物なし → None
                self._trained().save(path)
                f = get_distilled_filter()                     # 成果物あり → ロード
            self.assertIsInstance(f, DistilledQualityFilter)

    def test_rejects_legacy_artifact_without_teacher_record(self):
        """teacher 記録の無い旧成果物（外部 LLM 教師で作られた可能性）は使わない。"""
        s = self._trained()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "student.pkl")
            with open(path, "wb") as fh:
                pickle.dump({"vectorizer": s.vectorizer, "clf": s.clf}, fh)
            with self.assertRaises(ValueError):
                StudentClassifier.load(path)
            with mock.patch.dict(os.environ, {"SWI_DISTILLED_FILTER": path}):
                self.assertIsNone(get_distilled_filter())     # パイプラインは自作スコアラへ縮退


if __name__ == "__main__":
    unittest.main(verbosity=2)
