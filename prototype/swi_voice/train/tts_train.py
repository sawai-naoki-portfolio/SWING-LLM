"""
swi_voice/train/tts_train.py

AR コーデックLM（VALL-E 系 TTS）の学習エントリ [音声編3.1]。
学習済みコーデックで音声をコード列へ変換し、テキスト→コード列の次トークン
予測を学習する。swi_llm の SentencePiece トークナイザ・TransformerBlock を流用。

実行: python -m swi_voice.train.tts_train （codec_train の後に実行）
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
from swi_voice.data.audio_dataset import load_voice_clips, TTSDataset
from swi_voice.codec import NeuralAudioCodec
from swi_voice.model import ARCodecLM
from swi_voice.train.codec_train import CODEC_CKPT, CLIP_FRAMES

logger = setup_logger("voice.tts_train")

TTS_CKPT = f"voice_tts_checkpoint{config['ckpt_suffix']}.pt"


def _collate(batch):
    texts, codes = zip(*batch)
    maxlen_t = max(t.shape[0] for t in texts)
    text_pad = torch.stack([F.pad(t, (0, maxlen_t - t.shape[0])) for t in texts])
    # 04#1: テキストの実長を返し、右パディング位置をアテンションから除外できるようにする。
    text_lens = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
    # 可変長コード列を最大長へ右パディングし、実長 code_lens を返す（損失で末尾を除外）
    code_lens = torch.tensor([c.shape[0] for c in codes], dtype=torch.long)
    maxlen_c = int(code_lens.max())
    codes_pad = torch.stack([F.pad(c, (0, 0, 0, maxlen_c - c.shape[0])) for c in codes])   # [B, Tmax, n_q]
    return text_pad, codes_pad, code_lens, text_lens


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
    return codec, c["codebook_size"]


def main():
    cfg = config["tts"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"TTS training (mode={config['MODE']}, device={device}, cfg={cfg})")

    loaded = _load_codec(device)
    if loaded is None:
        return 1
    codec, codebook_size = loaded

    tokenizer = get_tokenizer()
    # クローニング学習: クリップ前半 prompt_frames を音響プロンプト、後半をターゲットとし、
    # ターゲット区間のみに損失をかける（参照音声の声色を保った続きの生成を学習）。
    prompt_frames = config["prompt_frames"]
    clips = load_voice_clips(max_clips=cfg["max_clips"] or 2000)
    ds = TTSDataset(clips, codec, tokenizer, clip_frames=CLIP_FRAMES, max_text_len=32)
    if len(ds) == 0:
        logger.error("No samples available for TTS training.")
        return 1
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=len(ds) > cfg["batch_size"], collate_fn=_collate)

    model = ARCodecLM(text_vocab_size=tokenizer.vocab_size,
                      n_codebooks=codec.n_quantizers, codebook_size=codebook_size,
                      dim=cfg["dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                      max_seq_len=cfg["max_seq_len"]).to(device)
    # 問題6: 分散環境では FSDP でシャーディング（optimizer 生成前）。単一デバイス/既定は素通し。
    model = maybe_wrap_fsdp(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model = maybe_compile(model)   # 問題6: SWI_USE_TORCH_COMPILE=1 のみ（既定素通し）

    model.train()
    for epoch in range(cfg["epochs"]):
        total, nb = 0.0, 0
        for text_ids, codes, code_lens, text_lens in loader:
            text_ids, codes, code_lens = text_ids.to(device), codes.to(device), code_lens.to(device)
            text_lens = text_lens.to(device)
            opt.zero_grad()
            # 04#1: 右パディング位置を False とするテキストマスク（フレームがパディングに注意しない）
            text_mask = torch.arange(text_ids.shape[1], device=device).unsqueeze(0) < text_lens.unsqueeze(1)
            # 最短クリップでも最低1フレームはターゲットが残るよう prompt_frames をクランプ
            pf = min(prompt_frames, max(0, int(code_lens.min()) - 1))
            _, loss = model(text_ids, codes, code_lens=code_lens, prompt_frames=pf, text_mask=text_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        logger.info(f"TTS epoch {epoch + 1}/{cfg['epochs']} avg_loss={total / max(nb, 1):.4f}")
        log_epoch_metric("voice_tts", epoch + 1, total / max(nb, 1))

    # 問題6: torch.compile 済みなら '_orig_mod.' 接頭辞を除去して ckpt 互換を保つ。
    torch.save({"model": _strip_compiled_prefix(model.state_dict()), "tts_cfg": cfg,
                "n_codebooks": codec.n_quantizers, "codebook_size": codebook_size}, TTS_CKPT)
    logger.info(f"Saved TTS checkpoint: {TTS_CKPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
