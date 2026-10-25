"""
swi_voice/train/hybrid_train.py

ハイブリッド AR+FM TTS（意味 LM + 音響 Flow Matching）の学習エントリ
[音声編3.3, NEW-V3 / 02_architecture_docs_gap_audit]。

設計文書が「2026 年主流トレンド」と位置づける 2 段構成のモデル（model/hybrid_ar_fm.py）は
実装・テスト済みだったが、**学習ステージも serving 接続も無く製品経路に乗っていなかった**。
本スクリプトが学習を、serve/tts.py の `SWI_TTS_ENGINE=hybrid` が serving を配線する。

2 段で学習する:
  段1: 意味 VQ（SemanticAcousticCodec の semantic_proj + semantic_vq）を学習。
       学習済み音響コーデックは凍結し、低フレームレートの意味トークン化だけを獲得する。
       ※ 意味表現は現状「音響特徴の低レート自己 VQ」プロキシ（fix 16 / voice step8 の
       HuBERT/w2v-BERT 蒸留で本物化する。その際も本スクリプトの骨格は変わらない）。
  段2: HybridARFM を学習。
       - 意味 LM: テキスト → 意味トークン列の次トークン予測（semantic_loss）
       - 音響 FM: 意味トークン条件で音響連続 latent を復元（acoustic_loss）

実行: python -m swi_voice.train.hybrid_train （codec_train の後に実行）
"""
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from swi_llm.utils.logger import setup_logger
from swi_llm.utils.metrics import log_epoch_metric
from swi_llm.utils.trainer_utils import maybe_compile, _strip_compiled_prefix
from swi_llm.data.tokenizer import get_tokenizer
from swi_voice.utils.config import config
from swi_voice.utils.constants import VOICE_CODEC_HOP, VOICE_LANG_TAG
from swi_voice.data.audio_dataset import load_voice_clips, _fit_length
from swi_voice.codec import NeuralAudioCodec
from swi_voice.codec.semantic_acoustic import SemanticAcousticCodec
from swi_voice.model.hybrid_ar_fm import HybridARFM
from swi_voice.train.codec_train import CODEC_CKPT, CLIP_FRAMES

logger = setup_logger("voice.hybrid_train")

HYBRID_CKPT = f"voice_hybrid_checkpoint{config['ckpt_suffix']}.pt"


class _TextWaveDataset(Dataset):
    """(text_ids[L], waveform[1, clip_frames*hop]) を返す。意味/音響の抽出は学習ループ側で
    バッチ単位に行う（段1 で意味 VQ が学習中に変わるため事前計算できない）。"""

    def __init__(self, clips, tokenizer, clip_frames, max_text_len=32):
        self.samples = []
        target_len = clip_frames * VOICE_CODEC_HOP
        for c in clips:
            tagged = VOICE_LANG_TAG.format(c.get("lang", "ja")) + c["text"]
            ids = tokenizer.encode(tagged)[:max_text_len]
            if not ids:
                continue
            wav = _fit_length(c["audio"], target_len)
            self.samples.append((torch.tensor(ids, dtype=torch.long), wav))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate(batch):
    texts, waves = zip(*batch)
    maxlen_t = max(t.shape[0] for t in texts)
    text_pad = torch.stack([F.pad(t, (0, maxlen_t - t.shape[0])) for t in texts])
    text_lens = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
    return text_pad, text_lens, torch.stack(waves)


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
    for p in codec.parameters():                 # 音響コーデックは凍結
        p.requires_grad_(False)
    return codec, c["dim"]


def main():
    cfg = config["hybrid"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Hybrid AR+FM training (mode={config['MODE']}, device={device}, cfg={cfg})")

    loaded = _load_codec(device)
    if loaded is None:
        return 1
    codec, latent_dim = loaded

    tokenizer = get_tokenizer()
    clips = load_voice_clips(max_clips=cfg["max_clips"] or 2000)
    ds = _TextWaveDataset(clips, tokenizer, clip_frames=CLIP_FRAMES)
    if len(ds) == 0:
        logger.error("No samples available for hybrid training.")
        return 1
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        drop_last=len(ds) > cfg["batch_size"], collate_fn=_collate)

    # ---- 段1: 意味 VQ（低フレームレートの意味トークン化）----------------------------
    sem_codec = SemanticAcousticCodec(codec).to(device)
    sem_params = [p for n, p in sem_codec.named_parameters() if not n.startswith("codec.")]
    opt_sem = torch.optim.Adam(sem_params, lr=cfg["lr"])
    sem_codec.train()
    for epoch in range(cfg["sem_vq_epochs"]):
        total, nb = 0.0, 0
        for _text, _tl, wav in loader:
            wav = wav.to(device)
            opt_sem.zero_grad()
            _idx, vq_loss = sem_codec(wav)
            vq_loss.backward()
            opt_sem.step()
            total += vq_loss.item()
            nb += 1
        logger.info(f"Semantic-VQ epoch {epoch + 1}/{cfg['sem_vq_epochs']} "
                    f"avg_vq_loss={total / max(nb, 1):.4f}")
        log_epoch_metric("voice_hybrid_semvq", epoch + 1, total / max(nb, 1))
    sem_codec.eval()

    # ---- 段2: HybridARFM（意味 LM + 音響 FM）----------------------------------------
    model = HybridARFM(
        text_vocab_size=tokenizer.vocab_size,
        semantic_codebook_size=sem_codec.semantic_codebook_size,
        acoustic_latent_dim=latent_dim,
        ar_dim=cfg["ar_dim"], ar_layers=cfg["ar_layers"], ar_heads=cfg["ar_heads"],
        fm_dim=cfg["fm_dim"], fm_layers=cfg["fm_layers"], fm_heads=cfg["fm_heads"],
        fm_max_frames=cfg["fm_max_frames"], max_sem_len=cfg["max_sem_len"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model = maybe_compile(model)                 # SWI_USE_TORCH_COMPILE=1 のみ（既定素通し）

    model.train()
    for epoch in range(cfg["epochs"]):
        tot_sem, tot_ac, nb = 0.0, 0.0, 0
        for text_ids, text_lens, wav in loader:
            text_ids, text_lens, wav = text_ids.to(device), text_lens.to(device), wav.to(device)
            with torch.no_grad():
                sem_tokens = sem_codec.extract_semantic_tokens(wav)      # [B, T_sem]
                acoustic = sem_codec.extract_acoustic_latent(wav)        # [B, T_ac, dim]
            text_mask = (torch.arange(text_ids.shape[1], device=device).unsqueeze(0)
                         < text_lens.unsqueeze(1))
            sem_lens = torch.full((sem_tokens.shape[0],), sem_tokens.shape[1],
                                  dtype=torch.long, device=device)
            opt.zero_grad()
            loss_sem = model.semantic_loss(text_ids, sem_tokens, code_lens=sem_lens,
                                           text_mask=text_mask)
            loss_ac = model.acoustic_loss(sem_tokens, acoustic)
            (loss_sem + loss_ac).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_sem += loss_sem.item()
            tot_ac += loss_ac.item()
            nb += 1
        logger.info(f"Hybrid epoch {epoch + 1}/{cfg['epochs']} "
                    f"avg_sem_loss={tot_sem / max(nb, 1):.4f} avg_fm_loss={tot_ac / max(nb, 1):.4f}")
        log_epoch_metric("voice_hybrid", epoch + 1, (tot_sem + tot_ac) / max(nb, 1),
                         sem_loss=tot_sem / max(nb, 1), fm_loss=tot_ac / max(nb, 1))

    torch.save({
        "hybrid": _strip_compiled_prefix(model.state_dict()),
        "sem_proj": sem_codec.semantic_proj.state_dict(),
        "sem_vq": sem_codec.semantic_vq.state_dict(),
        "hybrid_cfg": cfg,
        "latent_dim": latent_dim,
        "semantic_codebook_size": sem_codec.semantic_codebook_size,
    }, HYBRID_CKPT)
    logger.info(f"Saved hybrid checkpoint: {HYBRID_CKPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
