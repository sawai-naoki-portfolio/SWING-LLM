"""
swi_llm/data/quality_distill.py
Teacher スコア → 軽量分類器の蒸留による品質フィルタ [01_LLM §2, NEW-L1 / FineWeb-Edu 方式]

設計は「軽量 LLM（Teacher）で教育的価値をスコアリングし、FastText 等の軽量分類器で
TB 級コーパスを高速にフィルタリング」を要求する。従来はヒューリスティック
（EducationalValueScorer）による代用のみで、蒸留の経路が存在しなかった。

構成:
  1. **TeacherScorer**: 自作のヒューリスティック EducationalValueScorer を教師として
     文書の教育的価値を 0〜5 で採点する。完全自作方針（2026-10-25）により外部 LLM
     （旧: OpenAI 互換エンドポイント上の Qwen2.5）を教師にする経路は撤去した。
     自作 LLM が採点できる水準に育ったら、ここを差し替える。
  2. **StudentClassifier**: HashingVectorizer（文字 n-gram）+ SGDClassifier(logistic)。
     FastText と同じ「bag-of-ngram + 線形」クラスの構成で、Windows でネイティブビルド不要
     （scikit-learn は既存依存）。推論は数万 docs/sec 級で TB 級の一次フィルタに使える。
  3. **DistilledQualityFilter**: 学習済み student を advanced_pipeline のスコアラ契約
     （`score(text) -> 0..5`）で提供する。SWI_DISTILLED_FILTER=path で差し替わる。

CLI（教師採点 → 蒸留 → 保存 → ホールドアウト一致率の報告）:
    .venv\\Scripts\\python.exe -m swi_llm.data.quality_distill --sample 200
"""
import argparse
import os
import pickle
import sys

from swi_llm.utils.logger import setup_logger

PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROTOTYPE_DIR not in sys.path:
    sys.path.append(PROTOTYPE_DIR)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')   # type: ignore[union-attr]

logger = setup_logger("quality_distill")

DISTILLED_FILTER_ENV = "SWI_DISTILLED_FILTER"
DEFAULT_STUDENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "quality_student.pkl")
# 成果物に記録する教師の種別。これを持たない旧成果物は外部 LLM（Qwen2.5）教師で
# 作られた可能性があるため、完全自作方針の下では読み込まない。
TEACHER_KIND = "heuristic"


class TeacherScorer:
    """自作ヒューリスティック教師（EducationalValueScorer）で 0〜5 を採点する。"""

    kind = TEACHER_KIND

    def __init__(self):
        from swi_llm.data.advanced_pipeline import EducationalValueScorer
        self._heuristic = EducationalValueScorer()

    def score(self, text) -> int:
        """0〜5 の教育的価値（整数）。"""
        return int(round(self._heuristic.score(text)))


class StudentClassifier:
    """ハッシュ化文字 n-gram + 線形分類器（FastText 相当クラスの軽量 student）。"""

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import SGDClassifier
        # 文字 2-4gram: 日本語で形態素解析なしに機能する（MinHash のシングリングと同じ発想）
        self.vectorizer = HashingVectorizer(analyzer="char", ngram_range=(2, 4),
                                            n_features=2 ** 18, alternate_sign=False)
        self.clf = SGDClassifier(loss="log_loss", random_state=0)

    def fit(self, texts, labels):
        self.clf.fit(self.vectorizer.transform(texts), labels)
        return self

    def predict_proba_positive(self, texts):
        import numpy as np
        proba = self.clf.predict_proba(self.vectorizer.transform(texts))
        pos_idx = list(self.clf.classes_).index(1)
        return np.asarray(proba)[:, pos_idx]

    def save(self, path, teacher=TEACHER_KIND):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf, "teacher": teacher}, f)
        logger.info(f"Saved distilled quality student to {path}")

    @classmethod
    def load(cls, path):
        # SEC 注: pickle は任意コード実行になり得る。本ファイルは自プロジェクトの CLI が
        # 生成するローカル成果物のみを読む想定（外部から受け取った .pkl を置かないこと）。
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("teacher") != TEACHER_KIND:
            raise ValueError(
                f"{path} は自作教師で作られた成果物ではありません（teacher={data.get('teacher')!r}）。"
                "外部 LLM 教師の旧成果物の可能性があるため使用しません。削除して "
                "`python -m swi_llm.data.quality_distill` で再生成してください。")
        obj = cls.__new__(cls)
        obj.vectorizer = data["vectorizer"]
        obj.clf = data["clf"]
        return obj


class DistilledQualityFilter:
    """advanced_pipeline のスコアラ契約 `score(text) -> 0..5` で蒸留 student を提供する。

    student は「教育的（teacher スコア >= 3）」の確率を出すので、0〜5 スケールへ写像して
    既存の PIPELINE_QUALITY_MIN_SCORE 閾値と互換にする。
    """

    def __init__(self, student):
        self.student = student

    @classmethod
    def load(cls, path):
        return cls(StudentClassifier.load(path))

    def score(self, text) -> float:
        if not text or not text.strip():
            return 0.0
        return float(self.student.predict_proba_positive([text])[0] * 5.0)


def get_distilled_filter():
    """SWI_DISTILLED_FILTER=path（または既定パスの成果物）が在れば DistilledQualityFilter。"""
    path = os.environ.get(DISTILLED_FILTER_ENV, "").strip() or DEFAULT_STUDENT_PATH
    if not os.path.exists(path):
        return None
    try:
        f = DistilledQualityFilter.load(path)
        logger.info(f"Distilled quality filter loaded from {path} (NEW-L1).")
        return f
    except Exception:
        logger.exception(f"Failed to load distilled quality filter {path}; "
                         "falling back to the heuristic scorer.")
        return None


def _load_sample_docs(n):
    """教師採点用の文書サンプルを事前学習ソースから取る。"""
    from datasets import load_dataset
    from swi_llm.utils.constants import DATASET_WIKIPEDIA_JA
    ds = load_dataset(DATASET_WIKIPEDIA_JA, split=f"train[:{n}]")
    return [row.get("text", "") for row in ds if (row.get("text") or "").strip()]


def main():
    ap = argparse.ArgumentParser(description="Teacher→Student 蒸留品質フィルタの学習（NEW-L1）")
    ap.add_argument("--sample", type=int, default=200, help="teacher 採点する文書数")
    ap.add_argument("--out", default=DEFAULT_STUDENT_PATH, help="student の保存先 .pkl")
    ap.add_argument("--positive-threshold", type=int, default=3,
                    help="teacher スコアがこの値以上を正例（教育的）とする")
    args = ap.parse_args()

    docs = _load_sample_docs(args.sample)
    if len(docs) < 20:
        logger.error(f"教師採点に足る文書を取得できませんでした（{len(docs)} 件）。")
        return 1

    teacher = TeacherScorer()
    logger.info(f"Scoring {len(docs)} docs with the self-made {teacher.kind} teacher...")
    labels = [1 if teacher.score(d) >= args.positive_threshold else 0 for d in docs]
    pos = sum(labels)
    logger.info(f"Teacher labels: positive={pos} negative={len(labels) - pos}")
    if pos == 0 or pos == len(labels):
        logger.error("teacher ラベルが単一クラスです。--sample を増やすか閾値を調整してください。")
        return 1

    split = max(10, int(len(docs) * 0.8))
    student = StudentClassifier().fit(docs[:split], labels[:split])

    import numpy as np
    proba = student.predict_proba_positive(docs[split:])
    pred = (proba >= 0.5).astype(int)
    truth = np.asarray(labels[split:])
    agreement = float((pred == truth).mean()) if len(truth) else float("nan")
    logger.info(f"Holdout agreement with teacher: {agreement:.3f} (n={len(truth)})")

    student.save(args.out)
    logger.info(f"有効化するには SWI_DISTILLED_FILTER={args.out} を設定して "
                "advanced_pipeline を実行してください（未設定でも既定パスなら自動検出）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
