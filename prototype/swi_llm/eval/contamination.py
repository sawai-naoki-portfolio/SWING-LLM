"""
swi_llm/eval/contamination.py

ベンチマーク汚染検出（ブループリント §14.1）。

事前学習データにベンチマーク問題文が混入すると、スコアが実力を過大評価する。
空白正規化後の部分文字列一致で、
  1. ベンチ問題が学習コーパスに逐語混入しているか（汚染レポート）
  2. 学習コーパスのドキュメントがベンチ問題文を含むか（重複排除での除外）
を判定する。文字数が短い問題でも安定して機能するよう、n-gram ではなく
正規化部分文字列一致を採用している。
"""
import re

# この文字数未満のベンチ問題は短すぎて誤検出しやすいため汚染判定から除外する
DEFAULT_MIN_LEN = 8


def _norm(text):
    """空白を除去して比較用に正規化する。"""
    return re.sub(r"\s+", "", text)


class ContaminationChecker:
    """ベンチマーク問題文に基づく汚染検出。"""

    def __init__(self, benchmark_items, min_len=DEFAULT_MIN_LEN):
        self.min_len = min_len
        self.questions = [it.get("question", "") for it in benchmark_items]
        self.questions_norm = [_norm(q) for q in self.questions]

    def doc_is_contaminated(self, doc_text):
        """corpus ドキュメントがいずれかのベンチ問題文を逐語で含むなら True（学習除外対象）。"""
        d = _norm(doc_text)
        return any(len(q) >= self.min_len and q in d for q in self.questions_norm)

    def report(self, corpus_texts):
        """各ベンチ問題が corpus に混入しているかを判定し、汚染サマリを返す。"""
        corpus_norm = "\n".join(_norm(t) for t in corpus_texts)
        leakage = []
        for q_raw, q in zip(self.questions, self.questions_norm):
            leaked = len(q) >= self.min_len and q in corpus_norm
            leakage.append({"question": q_raw, "contaminated": leaked})
        n_contaminated = sum(1 for x in leakage if x["contaminated"])
        return {
            "n_total": len(leakage),
            "n_contaminated": n_contaminated,
            "contamination_rate": n_contaminated / len(leakage) if leakage else 0.0,
            "details": leakage,
        }
