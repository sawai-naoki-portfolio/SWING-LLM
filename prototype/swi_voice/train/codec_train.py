"""
swi_voice/train/codec_train.py

ニューラルコーデック（RVQ）の事前学習エントリ [音声編2.2]。
音声クリップ（オフライン時は合成）で波形再構成を学習し、後段 TTS が使う
「音声トークナイザ」を用意する。SWI_MODE で規模を制御する。

実行: python -m swi_voice.train.codec_train
"""
import os
import sys

import torch
from torch.utils.data import DataLoader

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.metrics import log_epoch_metric
from swi_llm.utils.trainer_utils import maybe_wrap_fsdp, maybe_compile, _strip_compiled_prefix
from swi_voice.utils.config import config
from swi_voice.data.audio_dataset import load_voice_clips, CodecDataset
from swi_voice.codec import NeuralAudioCodec

logger = setup_logger("voice.codec_train")

# クリップ長（フレーム）はモード別設定から取得（クローニングのプロンプト分割に足る長さ）
CLIP_FRAMES = config["clip_frames"]
CODEC_CKPT = f"voice_codec_checkpoint{config['ckpt_suffix']}.pt"


def main():
    cfg = config["codec"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Codec training (mode={config['MODE']}, device={device}, cfg={cfg})")

    clips = load_voice_clips(max_clips=cfg["max_clips"] or 2000)
    ds = CodecDataset(clips, clip_frames=CLIP_FRAMES)
    if len(ds) == 0:
        logger.error("No clips available for codec training.")
        return 1
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=len(ds) > cfg["batch_size"])

    codec = NeuralAudioCodec(dim=cfg["dim"], hidden=cfg["hidden"],
                             n_quantizers=cfg["n_quantizers"],
                             codebook_size=cfg["codebook_size"]).to(device)
    codec = maybe_wrap_fsdp(codec)                      # 問題6: 分散時のみFSDP（既定素通し）
    opt = torch.optim.Adam(codec.parameters(), lr=cfg["lr"])
    codec = maybe_compile(codec)                        # 問題6: SWI_USE_TORCH_COMPILE=1のみ

    codec.train()
    for epoch in range(cfg["epochs"]):
        total, nb = 0.0, 0
        for wav in loader:
            wav = wav.to(device)
            opt.zero_grad()
            _, _, loss = codec(wav)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        logger.info(f"Codec epoch {epoch + 1}/{cfg['epochs']} avg_loss={total / max(nb, 1):.4f}")
        log_epoch_metric("voice_codec", epoch + 1, total / max(nb, 1))

    torch.save({"model": _strip_compiled_prefix(codec.state_dict()), "codec_cfg": cfg}, CODEC_CKPT)
    logger.info(f"Saved codec checkpoint: {CODEC_CKPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
