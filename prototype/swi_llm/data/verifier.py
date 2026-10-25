"""
swi_llm/data/verifier.py

RLVR（検証可能な報酬）の検証器（ブループリント §7.2-C）。

3 種の検証を段階的に試みる:
  1. 数式等価 (SymPy): 期待値と抽出した回答を記号的/数値的に比較。
     例: "1/2" と "0.5"、"2x" と "x*2" を等価と判定。
  2. コード実行 (unit-test): 期待値がコード実行結果を示す場合、サンドボックスで
     実行して出力を照合（execute_python を利用）。
  3. 文字列/数値一致: 上記が使えない一般タスク向けのフォールバック。

いずれも失敗しても例外を投げず 0.0 相当を返す（RL ループを止めない）。
"""
import re

from swi_llm.utils.logger import setup_logger

logger = setup_logger("data.verifier")

# 完了テキストから最終回答を抽出するためのパターン
_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_LABEL_PATTERN = re.compile(r"(?:答え|回答|answer|=)\s*[:：]?\s*([^\n。]+)", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
# </think> 以降を最終回答とみなす
_THINK_END = "</think>"

# SymPy にかける式の最大長。これを超える文字列は数式ではないとみなしパースを
# スキップする。長大な入力での SymPy 暴走（実 GRPO のロングロールアウトで
# 検証器がハングする問題）を防ぐための安全弁。
_MAX_SYMPY_EXPR_LEN = 128


def _extract_final_answer(completion: str) -> str:
    """完了テキストから最終回答部分を抽出する。"""
    text = completion
    if _THINK_END in text:
        text = text.split(_THINK_END, 1)[1]
    text = text.strip()
    # \boxed{...} を最優先
    m = _BOXED_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    # 「答え: ...」形式
    m = _ANSWER_LABEL_PATTERN.search(text)
    if m:
        return m.group(1).strip()
    return text


def _to_sympy(expr_str):
    """文字列を SymPy 式へ変換（失敗時 None）。LaTeX 風表記も軽く正規化する。"""
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application,
        )
    except ImportError:
        return None
    s = expr_str.strip()
    if not s or len(s) > _MAX_SYMPY_EXPR_LEN:
        # 空、または数式にしては長すぎる文字列は SymPy に渡さない（暴走防止）
        return None
    # よくある表記ゆれの正規化
    s = s.replace("^", "**").replace("×", "*").replace("÷", "/")
    s = s.replace("\\pi", "pi").replace("π", "pi").replace("\\times", "*")
    s = s.replace("{", "(").replace("}", ")")
    # 暗黙の乗算 (例: "m c" -> "m*c", "2x" -> "2*x") を解釈する
    transforms = standard_transformations + (implicit_multiplication_application,)

    def _p(expr):
        return parse_expr(expr, transformations=transforms, evaluate=True)

    # 等式 "a = b" は左辺-右辺の差にして 0 と比較できるようにする
    try:
        if "=" in s and "==" not in s and "<=" not in s and ">=" not in s:
            lhs, rhs = s.split("=", 1)
            return _p(lhs) - _p(rhs)
        return _p(s)
    except Exception:
        return None


def verify_math(expected: str, extracted: str) -> bool:
    """SymPy による数式等価判定。判定不能なら False。"""
    try:
        import sympy
    except ImportError:
        return False
    e_exp = _to_sympy(expected)
    e_got = _to_sympy(extracted)
    if e_exp is None or e_got is None:
        return False
    try:
        diff = sympy.simplify(e_exp - e_got)
        if diff == 0:
            return True
        # 数値評価でも比較（記号的に簡約できないケース）
        val = complex(sympy.N(diff))
        return abs(val) < 1e-6
    except Exception:
        return False


def verify_code(expected: str, code: str) -> bool:
    """コードをサンドボックス実行し、標準出力に期待値が含まれるか判定。"""
    try:
        from swi_llm.agent.sandbox import execute_python
        result = execute_python(code)
        if result.get("status") != "success":
            return False
        output = str(result.get("output", ""))
        return expected.strip() in output
    except Exception as e:
        logger.warning(f"Code verification failed: {e}")
        return False


def _looks_like_code(text: str) -> bool:
    return bool(re.search(r"\b(def |print\(|import |return )", text))


def _normalize_answer(s: str) -> str:
    """回答比較用の正規化: 小文字化・空白圧縮・前後の句読点/記号除去。"""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" 　。.!?！？:：,，、")


def compute_reward(question: str, completion: str, expected_answer: str) -> float:
    """段階的検証で 0.0〜1.0 の報酬を返す（§7.2-C）。

    報酬ハッキング対策: 満点(1.0)は「抽出した最終回答」が期待値と一致する場合のみ与える。
    completion 本文のどこかに期待文字列/数値が現れるだけでは満点にしない（推論文に紛れ込ませて
    報酬を稼ぐ退行を防ぐ）。部分一致は限定的な部分点(0.5)に留める。
    """
    extracted = _extract_final_answer(completion)

    # 1. 数式等価（数値・記号）: 抽出した最終回答と期待値を比較
    if verify_math(expected_answer, extracted):
        return 1.0

    # 2. コード実行（完了がコードを含み、期待値が出力を示す場合）
    if _looks_like_code(completion):
        code_match = re.search(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
        code = code_match.group(1) if code_match else completion
        if verify_code(expected_answer, code):
            return 1.0

    # 3. 最終回答の正規化一致（completion 全体の部分一致ではなく、抽出回答と比較）
    exp_n = _normalize_answer(expected_answer)
    got_n = _normalize_answer(extracted)
    if exp_n and exp_n == got_n:
        return 1.0

    # 部分点: 期待回答が抽出回答内に「独立した語」として現れる（本文への紛れ込みは対象外）
    if exp_n and exp_n in got_n.split():
        return 0.5
    # 部分点: 期待値の数値集合が抽出回答の数値集合に含まれる（本文全体ではなく最終回答で判定）
    exp_nums = set(_NUMBER_PATTERN.findall(expected_answer))
    got_nums = set(_NUMBER_PATTERN.findall(extracted))
    if exp_nums and exp_nums <= got_nums:
        return 0.5

    # 長さペナルティ（無限生成の抑制）
    if len(completion) > 1000:
        return -0.5
    return 0.0
