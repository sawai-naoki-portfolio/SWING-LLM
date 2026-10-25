"""
swi_llm/utils/attention_backends.py
SDPA アテンションバックエンドの明示有効化と可視化 [01_core 展望2, NEW-L4]

設計（01_core_llm_architecture 展望2）は「Triton ベースのカスタムカーネル（FlashAttention）」
を挙げるが、**Triton は Windows を公式サポートしない**ため、本開発環境ではカスタムカーネル
路線は採れない（Linux 学習環境へ移行する際の選択肢として残る）。

実効的な等価物: 本プロジェクトの全アテンションは `F.scaled_dot_product_attention` 経由で
実装されており、PyTorch は条件（CUDA・fp16/bf16・対応 head_dim）を満たすと **内蔵の
FlashAttention カーネル**へ自動ディスパッチする。本モジュールはそれを「確実に・見える形で」
使うための 2 つを提供する:
  1. `configure_sdp_backends()`: flash / memory-efficient / math を明示有効化し、
     実効状態を INFO ログに出す（BaseTrainer 起動時に 1 回呼ばれる）。
  2. `probe_flash_attention()`: 実際に flash カーネル限定で 1 回 SDPA を実行してみて、
     この環境・dtype で flash が動くかを事実で確認する（推測ではなく実行結果）。
"""
import torch

from swi_llm.utils.logger import setup_logger

logger = setup_logger("attention_backends")

_configured = False


def probe_flash_attention(head_dim=64, dtype=torch.float16):
    """flash カーネル限定で SDPA を 1 回実行し、成功可否を返す（CUDA 不在は False）。"""
    if not torch.cuda.is_available():
        return False
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        q = torch.randn(1, 4, 128, head_dim, device="cuda", dtype=dtype)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return True
    except Exception as e:
        logger.debug(f"flash SDPA probe failed: {e}")
        return False


def configure_sdp_backends(force_log=False):
    """SDPA の全バックエンドを明示有効化し、実効状態を返す（ログは初回のみ）。

    既定でも有効だが、「有効化されているつもり」を防ぐため実効値を観測して記録する
    （log_startup_checklist と同じ思想）。math は数値検証用のフォールバックとして残す。
    """
    global _configured
    status = {"cuda": torch.cuda.is_available()}
    if status["cuda"]:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        status["flash_enabled"] = torch.backends.cuda.flash_sdp_enabled()
        status["mem_efficient_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        status["math_enabled"] = torch.backends.cuda.math_sdp_enabled()
        status["flash_kernel_works"] = probe_flash_attention()
    if not _configured or force_log:
        if status["cuda"]:
            logger.info(
                f"SDPA backends: flash={status['flash_enabled']} "
                f"mem_efficient={status['mem_efficient_enabled']} math={status['math_enabled']} "
                f"| flash カーネル実行プローブ: {'OK' if status['flash_kernel_works'] else 'NG (mem_efficient/math へフォールバック)'} "
                "(NEW-L4: Triton カスタムカーネルは Windows 非対応のため SDPA 内蔵 flash を使う)")
        else:
            logger.info("SDPA backends: CUDA なし（CPU math 実装。flash は対象外, NEW-L4）")
        _configured = True
    return status
