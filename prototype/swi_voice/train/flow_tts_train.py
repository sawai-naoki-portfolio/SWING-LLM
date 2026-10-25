"""
swi_voice/train/flow_tts_train.py

Flow Matching TTS（F5-TTS/E2-TTS 系）の学習エントリ [音声編3.2]。

学習済みコーデックで音声を連続 latent（量子化前, [T', dim]）へ変換し、テキスト条件下で
音声インペインティング（Rectified Flow）を学習する。非自己回帰のため系列全体を並列生成でき、
VALL-E 系（tts_train.py）より高速・安定・高音質を狙う。swi_llm のトークナイザを流用。

実行: python -m swi_voice.train.flow_tts_train （codec_train の後に実行）
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.metrics import log_epoch_metric
from swi_llm.utils.trainer_utils import maybe_wrap_fsdp, maybe_compile, _strip_compiled_prefix
from swi_llm.data.tokenizer import get_tokenizer
from swi_voice.utils.config import config
from swi_voice.data.audio_dataset import load_voice_clips, FlowTTSDataset
from swi_voice.codec import NeuralAudioCodec
from swi_voice.model import FlowTTS, speech_infilling_loss
from swi_voice.train.codec_train import CODEC_CKPT, CLIP_FRAMES

logger = setup_logger("voice.flow_tts_train")

FLOW_CKPT = f"voice_flow_checkpoint{config['ckpt_suffix']}.pt"


def _collate(batch):
    """text_ids は右パディング、latent は固定長 [n_frames, dim] なのでそのまま stack。"""
    texts, latents = zip(*batch)
    maxlen_t = max(t.shape[0] for t in texts)
    text_pad = torch.stack([F.pad(t, (0, maxlen_t - t.shape[0])) for t in texts])
    latents = torch.stack(latents)                       # [B, n_frames, dim]
    return text_pad, latents


def _load_codec(device):
    if not os.path.exists(CODEC_CKPT):
        logger.error(f"Codec checkpoint not found: {CODEC_CKPT}. Run codec_train first.")
        return None
    ckpt = torch.load(CODEC_CKPT, map_location=device, weights_only=False)
    c = ckpt["codec_cfg"]
    codec = NeuralAudioCodec(dim=c["dim"], hidden=c["hidden"],
                             n_quantizers=c["n_quantizers"], codebook_size=c["codebook_size"]).to(device)
    codec.load_state_dict(ckpt["model"])
    codec.eval()
    return codec, c["dim"]


def main():
    cfg = config["flow"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Flow-TTS training (mode={config['MODE']}, device={device}, cfg={cfg})")

    loaded = _load_codec(device)
    if loaded is None:
        return 1
    codec, latent_dim = loaded

    tokenizer = get_tokenizer()
    clips = load_voice_clips(max_clips=cfg["max_clips"] or 2000)
    ds = FlowTTSDataset(clips, codec, tokenizer, clip_frames=CLIP_FRAMES,
                        n_frames=cfg["n_frames"], max_text_len=32)
    if len(ds) == 0:
        logger.error("No samples available for Flow-TTS training.")
        return 1
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=len(ds) > cfg["batch_size"], collate_fn=_collate)

    model = FlowTTS(text_vocab_size=tokenizer.vocab_size, latent_dim=latent_dim,
                    dim=cfg["dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                    max_frames=cfg["max_frames"]).to(device)
    model = maybe_wrap_fsdp(model)                      # 問題6: 分散時のみFSDP（既定素通し）
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model = maybe_compile(model)                        # 問題6: SWI_USE_TORCH_COMPILE=1のみ

    model.train()
    for epoch in range(cfg["epochs"]):
        total, nb = 0.0, 0
        for text_ids, latents in loader:
            text_ids, latents = text_ids.to(device), latents.to(device)
            opt.zero_grad()
            loss = speech_infilling_loss(model, latents, text_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        logger.info(f"Flow-TTS epoch {epoch + 1}/{cfg['epochs']} avg_loss={total / max(nb, 1):.4f}")
        log_epoch_metric("voice_flow", epoch + 1, total / max(nb, 1))

    torch.save({"model": _strip_compiled_prefix(model.state_dict()), "flow_cfg": cfg,
                "latent_dim": latent_dim}, FLOW_CKPT)
    logger.info(f"Saved Flow-TTS checkpoint: {FLOW_CKPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
