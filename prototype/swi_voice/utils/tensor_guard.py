"""
swi_voice/utils/tensor_guard.py
テンソル安全性チェックの共通ヘルパ（規約 V-2 / 指摘 V-14）。

規約 V-2 は「テンソルを受け取る関数の冒頭で空チェックを行い安全にフォールバックする」ことを
求めるが、`.numel()` は torch.Tensor 固有のため、`numpy.ndarray` / `list` 等が渡ると
チェック自体が AttributeError で落ちる（V-14）。実運用の音声フレームは `soundfile` 由来の
ndarray であることが多く、ここで Tensor へ正規化してから空チェックする。
"""
import torch

from swi_llm.utils.logger import setup_logger

logger = setup_logger("voice.tensor_guard")

# 変換不能型の警告は (name, 型) ごとに 1 回だけ出す。ウェイクワード検出などの常時起動
# ホットパスから毎フレーム呼ばれるため、無条件の連続ログは規約 V-1 に抵触する。
_warned_types = set()


def ensure_waveform(x, name="waveform"):
    """入力を torch.Tensor へ正規化する。None・空・変換不能は None を返す。

    Args:
        x: torch.Tensor / numpy.ndarray / list など波形として解釈可能な値。
        name: 警告ログ用の呼び出し元コンテキスト名。

    Returns:
        非空の torch.Tensor、または None（呼び出し側が「入力なし」としてフォールバックする）。
    """
    if x is None:
        return None
    if not isinstance(x, torch.Tensor):
        try:
            x = torch.as_tensor(x)
        except (TypeError, ValueError, RuntimeError) as e:
            key = (name, type(x).__name__)
            if key not in _warned_types:
                _warned_types.add(key)
                logger.warning(
                    f"{name}: cannot interpret {type(x).__name__} as a waveform ({e}); "
                    "treating as empty input (further occurrences suppressed).")
            return None
    if x.numel() == 0:
        return None
    return x
