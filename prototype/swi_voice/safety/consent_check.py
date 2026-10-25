"""
swi_voice/safety/consent_check.py

利用許諾の確認処理 (Consent Check) [音声編8.1]。

同意ファーストの設計原則（同意なしに任意話者の声を模倣・学習しない）を仕組みで
担保する。従来はプレースホルダで「話者IDが非空なら常に許可」だったが、本実装では
`data.consent_ledger.ConsentLedger` を実際に参照して許諾を判定する。

判定ポリシー（consent-first・既定拒否）:
  1. 話者IDが空 → 拒否。
  2. `source`（データ収集元のデータセット名）が「公開ライセンス済みコーパス」で、
     かつ purpose がそのデータセット単位の同意スコープに含まれる → 許可。
     ※ VOICE_DATASET_CANDIDATES の公開コーパスは研究/ASR/TTS 用途で明示ライセンス
       されているため、データセット単位の同意として扱う（voice_cloning は含めない）。
  3. 台帳に話者ID個別の有効な明示同意があり purpose を含む → 許可。
  4. 上記いずれにも該当しない → 拒否（未知/未同意の話者は既定で除外）。

CONSENT_ENFORCE=False の場合のみ、後方互換のため全通し（従来挙動）に戻す。
"""
import os

from swi_llm.utils.logger import setup_logger
from swi_voice.utils.constants import (
    VOICE_DATASET_CANDIDATES, CONSENT_PUBLIC_SOURCE_SCOPES, CONSENT_ENFORCE,
    VOICE_CLONING_CONSENT_ENV, CONSENT_SCOPE_CLONING, CONSENT_SCOPE_TTS,
)
from swi_voice.data.consent_ledger import ConsentLedger

logger = setup_logger("voice.consent")

# プロセス共有の同意台帳（個別の明示同意はここへ grant する）。
_ledger = ConsentLedger()

# 公開ライセンス済みコーパスのソース名集合（データセット単位の同意）。
# VOICE_DATASET_CANDIDATES に登録されたデータセットは、研究/ASR/TTS 用途で
# 明示ライセンスされている前提で許諾済みソースとして扱う。
_CONSENTED_SOURCES = {entry[0] for entry in VOICE_DATASET_CANDIDATES}

# 拒否のログを過剰に出さないためのサマリ用カウンタ（ソース単位）。
_denied_summary = {}


def get_ledger() -> ConsentLedger:
    """プロセス共有の同意台帳を返す（個別同意の登録/失効に使う）。"""
    return _ledger


def grant_consent(speaker_id, scope, expires_at=None):
    """特定話者に明示同意を付与する（例: 自社録音・出演者同意済みの声）。"""
    _ledger.grant(speaker_id, scope, expires_at=expires_at)
    logger.info(f"Consent granted: speaker='{speaker_id}' scope={list(scope)}")


def revoke_consent(speaker_id):
    """特定話者の同意を撤回する（以降の利用を拒否）。"""
    _ledger.revoke(speaker_id)
    logger.info(f"Consent revoked: speaker='{speaker_id}'")


def register_consented_source(source_name):
    """公開ライセンス済みソースを追加登録する（データセット単位の同意）。"""
    _CONSENTED_SOURCES.add(source_name)


def _source_is_consented(source, purpose):
    """source が公開ライセンス済みコーパスで、purpose がその同意スコープに含まれるか。"""
    if not source:
        return False
    if source in _CONSENTED_SOURCES:
        return purpose in CONSENT_PUBLIC_SOURCE_SCOPES
    # 前方一致（config subset 等でソース名にサフィックスが付くケースを許容）
    for consented in _CONSENTED_SOURCES:
        if source.startswith(consented):
            return purpose in CONSENT_PUBLIC_SOURCE_SCOPES
    return False


def verify_speaker_consent(speaker_id: str, purpose: str = "training", source: str = None) -> bool:
    """発話者の利用許諾を確認する（consent-first・既定拒否）。

    Args:
        speaker_id: 話者ID（Common Voice の client_id 等。無ければデータセット名）。
        purpose:    利用目的（"training"/"pretrain"/"commercial_tts"/"voice_cloning"）。
        source:     データ収集元のデータセット名。公開ライセンス済みコーパスの
                    データセット単位同意を判定するために使う。

    Returns:
        許諾されていれば True。未同意/未知の話者・ソースは False。
    """
    try:
        if not speaker_id:
            logger.warning("Empty speaker_id provided for consent check; denying.")
            return False

        # 後方互換フラグ: 明示的に無効化された場合のみ全通し（非推奨）。
        if not CONSENT_ENFORCE:
            return True

        # 1) データセット単位の同意（公開ライセンス済みコーパス）
        if _source_is_consented(source, purpose):
            return True

        # 2) 話者個別の明示同意（台帳）
        if _ledger.is_usage_allowed(speaker_id, purpose):
            return True

        # 3) consent-first の既定拒否。ソース単位で件数を集計し、まとめて警告する。
        key = source or speaker_id
        n = _denied_summary.get(key, 0) + 1
        _denied_summary[key] = n
        if n == 1:
            logger.warning(
                f"Consent denied: source='{source}' speaker='{speaker_id}' purpose='{purpose}' "
                f"(no dataset-level or per-speaker consent). Such items are excluded from collection.")
        return False
    except Exception:
        logger.exception(f"Unexpected error during consent verification for speaker {speaker_id}; denying.")
        return False


def _seed_cloning_consent_from_env():
    """`VOICE_CLONING_CONSENT_ENV` に列挙された話者へ voice_cloning 同意を付与する。

    クローニングは既定拒否のため、同意取得済みの話者（自社録音・出演者等）を運用側が
    プロセス起動時に登録するための唯一のサポート経路。既定値は空＝何も許可しない。
    """
    raw = os.environ.get(VOICE_CLONING_CONSENT_ENV, "")
    speaker_ids = [s.strip() for s in raw.split(",") if s.strip()]
    for sid in speaker_ids:
        _ledger.grant(sid, [CONSENT_SCOPE_CLONING, CONSENT_SCOPE_TTS])
    if speaker_ids:
        logger.info(f"Seeded voice_cloning consent for {len(speaker_ids)} speaker(s) "
                    f"from {VOICE_CLONING_CONSENT_ENV}.")


_seed_cloning_consent_from_env()
