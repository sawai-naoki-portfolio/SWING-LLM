"""
swi_voice/model/style_control.py
スタイル/感情の制御 [音声編4.3 / 6.3, 実装編4.2]

TTS の出力に感情・話し方のスタイルを条件付ける。実装編4.2 の StyleConditioner と同じく
3 経路を統一的に条件ベクトルへエンコードする:
  - 明示制御: 感情タグ (emotion_id: neutral/happy/sad/... = VOICE_EMOTIONS) を埋め込む。
  - 暗黙制御: 参照音声の音響 latent をプールしてスタイルベクトルへ写像する
    （参照話者の話し方・トーンの暗黙的転写。ゼロショットクローニングと併用可能）。
  - 指示追従: 「落ち着いた低い声でゆっくり」等の自然言語スタイル指示をトークン列として
    受け取り、テキストエンコーダでプールして条件ベクトルにする [音声編6.3 指示追従TTS]。

出力は TTS モデルの条件付け次元 dim のスタイル埋め込み [B, dim]。ARCodecLM / FlowTTS の
`use_style=True` 経路に `style_emb=` として渡すと生成に反映される（既定 None は無条件）。

C-1（チェックポイント互換）: 指示経路のパラメータは `instruction_dim` を指定したときのみ
生成する。既定 None では state_dict の形が従来と一致し、既存チェックポイントを読める。
"""
import torch
import torch.nn as nn

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import VOICE_N_EMOTIONS

logger = setup_logger("voice.style_control")


class InstructionEncoder(nn.Module):
    """自然言語スタイル指示のトークン列 -> プール済み条件ベクトル [B, dim] [実装編4.2]。

    埋め込み + 1 層の双方向 GRU 相当の軽量エンコーダではなく、依存を増やさないため
    埋め込み → Transformer 風の平均プーリング MLP で最小実装する。`pad_id` の位置は
    マスクしてプールから除外する（右パディングが平均を薄めないように）。
    """

    def __init__(self, vocab_size, dim, pad_id=0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pad_id = pad_id

    def forward(self, instruction_ids):
        """instruction_ids [B, L]（long）-> [B, dim]。パディングを除いた平均プーリング。"""
        x = self.token_emb(instruction_ids)                      # [B, L, dim]
        mask = (instruction_ids != self.pad_id).unsqueeze(-1).to(x.dtype)   # [B, L, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)                   # 全パディングでも 0 除算しない
        pooled = (x * mask).sum(dim=1) / denom                   # [B, dim]
        return self.proj(pooled)


class StyleEncoder(nn.Module):
    """感情タグ / 参照音声 / 自然言語指示からスタイル埋め込み [B, dim] を生成する [実装編4.2]。

    Args:
        dim: 出力スタイル埋め込みの次元（TTS の条件付け次元に合わせる）。
        n_emotions: 感情タグの種類数。
        ref_latent_dim: 参照音響 latent の次元。None なら参照経路を持たない。
        instruction_dim: 指示テキストの語彙サイズ。None なら指示経路を持たない
            （既存チェックポイント互換のため、既定では追加パラメータを作らない）。
        pad_id: 指示トークン列のパディングID。
    """

    def __init__(self, dim, n_emotions=VOICE_N_EMOTIONS, ref_latent_dim=None,
                 instruction_dim=None, pad_id=0):
        super().__init__()
        self.emotion_emb = nn.Embedding(n_emotions, dim)
        self.ref_proj = None
        if ref_latent_dim is not None:
            self.ref_proj = nn.Sequential(
                nn.Linear(ref_latent_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.instruction_encoder = None
        if instruction_dim is not None:
            self.instruction_encoder = InstructionEncoder(instruction_dim, dim, pad_id=pad_id)
        self.n_emotions = n_emotions
        logger.info(f"Initializing StyleEncoder: dim={dim}, n_emotions={n_emotions}, "
                    f"ref={'on' if ref_latent_dim else 'off'}, "
                    f"instruction={'on' if instruction_dim else 'off'}")

    def from_emotion(self, emotion_id):
        """emotion_id [B]（long）-> スタイル埋め込み [B, dim]。"""
        return self.emotion_emb(emotion_id)

    def from_reference(self, ref_latent):
        """参照音響 latent [B, Lp, ref_latent_dim] -> スタイル埋め込み [B, dim]（時間平均）。"""
        if self.ref_proj is None:
            raise ValueError("StyleEncoder was created without ref_latent_dim; cannot use reference style.")
        return self.ref_proj(ref_latent.mean(dim=1))

    def from_instruction(self, instruction_ids):
        """自然言語スタイル指示 [B, L]（long）-> スタイル埋め込み [B, dim] [音声編6.3]。"""
        if self.instruction_encoder is None:
            raise ValueError("StyleEncoder was created without instruction_dim; cannot use instruction style.")
        return self.instruction_encoder(instruction_ids)

    def forward(self, emotion_id=None, ref_latent=None, instruction_ids=None):
        """3 経路のいずれか/複数からスタイル埋め込みを返す。

        実装編4.2 に合わせ、複数指定時は条件ベクトルの**平均**を取る（経路数が増えても
        スケールが一定に保たれる）。何も指定しなければ None（無条件生成）。
        """
        parts = []
        if emotion_id is not None:
            parts.append(self.from_emotion(emotion_id))
        if ref_latent is not None:
            parts.append(self.from_reference(ref_latent))
        if instruction_ids is not None:
            parts.append(self.from_instruction(instruction_ids))
        if not parts:
            return None
        return torch.stack(parts, dim=0).mean(dim=0)
