"""
swi_voice/model/cloning.py
ゼロショット音声クローニングと、その同意ゲート [音声編4.1 / 8.1 / 12.2]

参照音声（数秒）のコーデックコード列を音響プロンプトとして与え、追加学習なしに
その声色でテキストを発話させる [音声編4.1]。

同意ファーストの強制 [音声編8.1, 実装編4.1・10.1, 品質要件 V-4]:
  「同意なしに任意話者の声を模倣しない」を仕組みで担保するため、参照音声を取り扱う
  すべての経路はこのモジュールのゲートを通す。クローニングは**話者個別の明示同意**を
  必須とし、公開ライセンス済みコーパスのデータセット単位同意
  (CONSENT_PUBLIC_SOURCE_SCOPES) には voice_cloning を含めない。同意が確認できない
  参照音声は `CloningConsentError` で明示的に拒否する（無言のフォールバックはしない）。

クローニング悪用監視 [音声編12.2]:
  拒否した試行を話者ID単位で集計し、`denied_cloning_attempts()` で参照できる。
  無許可の参照音声アップロードの反復（なりすまし試行）の検知に使う。

同意の登録は `swi_voice.safety.consent_check.grant_consent(speaker_id, ["voice_cloning"])`、
または環境変数 `SWI_VOICE_CLONING_CONSENTED`（カンマ区切りの話者ID）で行う。
"""
import torch

from swi_llm.utils.logger import setup_logger
from swi_voice.safety.consent_check import verify_speaker_consent
from swi_voice.utils.constants import (
    CONSENT_SCOPE_CLONING, CONSENT_ENFORCE, VOICE_MAX_PROMPT_FRAMES,
)

logger = setup_logger("voice.cloning")


class CloningConsentError(PermissionError):
    """参照音声の `voice_cloning` 同意が確認できないときに送出する [実装編4.1]。"""


# [音声編12.2] 拒否したクローニング試行の話者ID単位カウンタ（悪用監視）。
_denied_attempts = {}

_UNSPECIFIED = "<unspecified>"


def denied_cloning_attempts() -> dict:
    """拒否したクローニング試行の {speaker_id: 回数}（悪用監視用のスナップショット）。"""
    return dict(_denied_attempts)


def reset_denied_cloning_attempts():
    """悪用監視カウンタを初期化する（テスト・監視ウィンドウの切り替え用）。"""
    _denied_attempts.clear()


def verify_reference_consent(speaker_id, source=None) -> bool:
    """参照音声の話者が `voice_cloning` に同意しているか [音声編8.1]。

    speaker_id が空なら拒否（誰の声か特定できない参照音声は模倣しない）。
    後方互換フラグ CONSENT_ENFORCE=False のときのみ全通しに戻る（非推奨）。
    """
    if not CONSENT_ENFORCE:
        return True
    if not speaker_id:
        return False
    return verify_speaker_consent(speaker_id, purpose=CONSENT_SCOPE_CLONING, source=source)


def require_reference_consent(speaker_id, source=None):
    """同意が無ければ `CloningConsentError` を送出する（クローニング経路の必須ゲート）。

    拒否は例外で明示する。呼び出し側が握り潰して「同意なしのクローニング」に
    フォールバックすることを防ぐため、戻り値では表現しない [品質要件 V-4]。
    """
    if verify_reference_consent(speaker_id, source=source):
        return
    key = speaker_id or _UNSPECIFIED
    _denied_attempts[key] = _denied_attempts.get(key, 0) + 1
    logger.warning(
        f"Voice cloning denied: speaker='{key}' has no explicit 'voice_cloning' consent "
        f"(attempt #{_denied_attempts[key]}).")
    raise CloningConsentError(
        "reference audio requires a per-speaker 'voice_cloning' consent record. "
        "Register it via swi_voice.safety.consent_check.grant_consent(speaker_id, "
        "['voice_cloning']) or the SWI_VOICE_CLONING_CONSENTED environment variable.")


def _to_batch(waveform: torch.Tensor) -> torch.Tensor:
    """波形を [1,1,T] へ整形する（[T] / [1,T] / [B,1,T] のいずれを渡しても対応）。"""
    if waveform.dim() == 1:
        return waveform.view(1, 1, -1)
    if waveform.dim() == 2:
        return waveform.unsqueeze(0)
    return waveform


@torch.no_grad()
def encode_reference_prompt(ref_audio, codec, max_prompt_frames=VOICE_MAX_PROMPT_FRAMES):
    """参照波形 → 音響プロンプトのコード列 [1, Lp, n_q]。空/None なら None を返す。

    V-2: テンソルの空チェックを行い、想定内の WARNING を出して安全にフォールバックする。
    同意チェックは行わない（呼び出し側が `require_reference_consent` を通す前提）。
    """
    if ref_audio is None or ref_audio.numel() == 0:
        logger.warning("encode_reference_prompt: empty reference audio; no acoustic prompt.")
        return None
    codes = codec.encode(_to_batch(ref_audio))                 # [1, Lp, n_q]
    if codes.shape[1] == 0:
        logger.warning("encode_reference_prompt: codec produced 0 frames; no acoustic prompt.")
        return None
    if codes.shape[1] > max_prompt_frames:
        codes = codes[:, :max_prompt_frames]                   # コンテキスト肥大化を防ぐ
    return codes


@torch.no_grad()
def zero_shot_clone_inference(model, text_ids, ref_audio, codec, speaker_id,
                              source=None, consent_check_fn=None,
                              max_prompt_frames=VOICE_MAX_PROMPT_FRAMES,
                              max_frames=200, temperature=0.8, top_k=50, style_emb=None):
    """参照音声の声色を模倣してテキストを発話する [音声編4.1, 実装編4.1]。

    設計方針どおり、**呼び出し口で必ず同意チェックを通す**。同意が無ければ波形の
    デコードにも到達せず `CloningConsentError` を送出する。

    Args:
        model: `.generate(text_ids, prompt_codes=..., ...)` を実装する TTS モデル（ARCodecLM）。
        text_ids: [B, Lt] のテキストトークン列。
        ref_audio: 参照波形テンソル（[T] / [1,T] / [1,1,T]）。
        codec: `.encode(wav)` / `.decode(codes)` を実装する NeuralAudioCodec。
        speaker_id: 参照音声の話者ID。同意台帳の照合キー（必須）。
        source: データ収集元（公開コーパス判定用。クローニングでは通常 None）。
        consent_check_fn: `callable(speaker_id) -> bool`。差し替え用（既定は台帳照合）。

    Returns:
        生成波形 [B, 1, T]。生成が空なら None。

    Raises:
        CloningConsentError: 参照音声の voice_cloning 同意が確認できない場合。
    """
    if consent_check_fn is None:
        require_reference_consent(speaker_id, source=source)          # [音声編8.1] 同意ファースト
    elif not consent_check_fn(speaker_id):
        raise CloningConsentError(f"consent check failed for speaker '{speaker_id}'")

    prompt_codes = encode_reference_prompt(ref_audio, codec, max_prompt_frames)
    generated = model.generate(text_ids, prompt_codes=prompt_codes, max_frames=max_frames,
                               temperature=temperature, top_k=top_k, style_emb=style_emb)
    if generated.shape[1] == 0:
        logger.warning("zero_shot_clone_inference: model generated 0 frames.")
        return None
    return codec.decode(generated)
