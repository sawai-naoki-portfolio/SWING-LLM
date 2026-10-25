"""
swi_llm/data/tokenizer.py

SentencePiece BPE トークナイザ（語彙 32,768・日本語最適化 / ブループリント §1.4）の
ロードとラッパ。学習は train_tokenizer.py を参照。

公開インターフェース（旧 tiktoken 実装と互換）:
  - get_tokenizer() -> SPTokenizer
  - EOT_TOKEN_ID: 文書終端 (= eos) の ID
  - VOCAB_SIZE:   語彙サイズ（Embedding / LM Head のサイズ決定に使用）
  - PAD_TOKEN_ID: パディング用 ID

SPTokenizer は .encode(str)->list[int] / .decode(list[int])->str を提供し、
.eot_token_id / .pad_token_id 属性を持つ。
"""
import os

import sentencepiece as spm

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.constants import (
    TOKENIZER_MODEL_PATH, TOKEN_EOS_ID, TOKEN_PAD_ID, TOKEN_UNK_ID, TOKEN_BOS_ID,
)

logger = setup_logger("tokenizer")

# prototype ディレクトリからの相対パス（TOKENIZER_MODEL_PATH）を絶対パスへ解決。
# tokenizer.py は swi_llm/data/ にあるので 2 階層上が prototype。
_PROTOTYPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_ABS_PATH = os.path.join(_PROTOTYPE_DIR, TOKENIZER_MODEL_PATH)


class SPTokenizer:
    """SentencePiece モデルのラッパ。"""

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}\n"
                "Train it first: `.venv\\Scripts\\python.exe -m swi_llm.data.train_tokenizer`"
            )
        self.sp = spm.SentencePieceProcessor()
        # SentencePiece の C++ ローダは非 ASCII を含む絶対パス（例: 「デスクトップ」）を
        # 扱えない (Illegal byte sequence)。Python 側でバイト読み込みし、
        # シリアライズ済み proto として渡すことで回避する。
        with open(model_path, "rb") as f:
            self.sp.load_from_serialized_proto(f.read())
        self.eot_token_id = TOKEN_EOS_ID
        self.pad_token_id = TOKEN_PAD_ID
        self.unk_token_id = TOKEN_UNK_ID
        self.bos_token_id = TOKEN_BOS_ID
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str) -> list:
        """文字列をトークン ID 列へ変換する（特殊トークンはアトミックに 1 ID 化）。"""
        return self.sp.encode(text, out_type=int)

    def encode_batch(self, texts) -> list:
        """複数文書をまとめてトークン化する（C++ 側のマルチスレッド。encode の逐次呼び出しと同一結果）。"""
        return self.sp.encode(list(texts), out_type=int, num_threads=os.cpu_count() or 1)

    def decode(self, ids) -> str:
        """トークン ID 列を文字列へ復元する。"""
        return self.sp.decode(list(ids))


# シングルトン（プロセス内で 1 度だけロード）
_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = SPTokenizer(_MODEL_ABS_PATH)
        # DEBUG: トークナイザのロード結果（1度だけ。語彙・特殊トークンIDの確認）
        logger.debug(f"Tokenizer loaded: vocab_size={_TOKENIZER.vocab_size}, "
                     f"eos={_TOKENIZER.eot_token_id}, pad={_TOKENIZER.pad_token_id}, "
                     f"bos={_TOKENIZER.bos_token_id}, unk={_TOKENIZER.unk_token_id}")
    return _TOKENIZER


# モジュールロード時に確定させる定数（多くのモジュールが直接 import する）
_sp_probe = get_tokenizer()
EOT_TOKEN_ID = _sp_probe.eot_token_id
PAD_TOKEN_ID = _sp_probe.pad_token_id
VOCAB_SIZE = _sp_probe.vocab_size
