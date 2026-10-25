"""
swi_voice/data/consent_ledger.py

同意チェーン [音声編8.1]。「誰が・いつ・どの範囲の利用に」同意したかを追跡可能な
形で保持し、学習・クローニングパイプラインの入口で必ず参照する。同意ファーストの
設計原則（同意なしに任意話者の声を模倣する機能は提供しない）を仕組みで担保する。
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from swi_llm.utils.logger import setup_logger

logger = setup_logger("voice.consent")


@dataclass
class ConsentRecord:
    speaker_id: str
    granted_at: datetime
    scope: list                       # ["pretrain", "commercial_tts", "voice_cloning"]
    expires_at: Optional[datetime] = None
    revoked: bool = False


class ConsentLedger:
    """同意記録の台帳。失効・撤回にも対応する。"""

    def __init__(self):
        self._db = {}

    def grant(self, speaker_id, scope, expires_at=None):
        self._db[speaker_id] = ConsentRecord(
            speaker_id=speaker_id, granted_at=datetime.now(),
            scope=list(scope), expires_at=expires_at)

    def revoke(self, speaker_id):
        if speaker_id in self._db:
            self._db[speaker_id].revoked = True

    def is_usage_allowed(self, speaker_id, purpose):
        record = self._db.get(speaker_id)
        if record is None or record.revoked:
            return False
        if record.expires_at and datetime.now() > record.expires_at:
            return False
        return purpose in record.scope


def filter_by_consent(items, consent_ledger, purpose):
    """同意が確認できない話者データを収集パイプラインの入口で除外する [音声編8.1]。
    items: {"speaker_id": ...} を持つ dict のリスト。"""
    allowed, dropped = [], 0
    for item in items:
        if consent_ledger.is_usage_allowed(item.get("speaker_id", ""), purpose):
            allowed.append(item)
        else:
            dropped += 1
    if dropped:
        logger.info(f"Consent filter: dropped {dropped} items lacking '{purpose}' consent.")
    return allowed
