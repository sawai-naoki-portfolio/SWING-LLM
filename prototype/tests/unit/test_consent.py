"""
同意チェーン (swi_voice/safety/consent_check.py) の単体テスト。

consent-first・既定拒否のポリシー:
  - 公開ライセンス済みコーパス(source)は training/pretrain/commercial_tts で許可。
  - 未知ソース・未知話者は既定で拒否。
  - 台帳への明示 grant で許可、revoke で拒否。
  - voice_cloning は公開ソースでも拒否（個別同意が必要）。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_voice.safety.consent_check import (
    verify_speaker_consent, grant_consent, revoke_consent, register_consented_source,
)
from swi_voice.utils.constants import VOICE_DATASET_CANDIDATES

# 実在の公開コーパス名（VOICE_DATASET_CANDIDATES の先頭）を使う
PUBLIC_SOURCE = VOICE_DATASET_CANDIDATES[0][0]


class TestConsent(unittest.TestCase):
    def test_empty_speaker_denied(self):
        self.assertFalse(verify_speaker_consent("", purpose="training", source=PUBLIC_SOURCE))

    def test_public_source_allowed_for_training(self):
        # 公開ライセンス済みコーパスはデータセット単位の同意として training を許可
        self.assertTrue(verify_speaker_consent("spk_x", purpose="training", source=PUBLIC_SOURCE))

    def test_public_source_prefix_match(self):
        # config subset 等でサフィックスが付いても前方一致で許可
        self.assertTrue(verify_speaker_consent("spk_y", purpose="training", source=PUBLIC_SOURCE + ":subset_0"))

    def test_public_source_denies_cloning(self):
        # 公開ソースでも voice_cloning は許可スコープ外 → 拒否（個別同意が必要）
        self.assertFalse(verify_speaker_consent("spk_z", purpose="voice_cloning", source=PUBLIC_SOURCE))

    def test_unknown_source_denied(self):
        self.assertFalse(verify_speaker_consent("spk_a", purpose="training", source="some/unlicensed_corpus"))

    def test_no_source_denied_without_grant(self):
        # source 無し・台帳未登録 → 既定拒否
        self.assertFalse(verify_speaker_consent("unregistered_speaker", purpose="training"))

    def test_explicit_grant_then_revoke(self):
        sid = "consented_actor_001"
        self.assertFalse(verify_speaker_consent(sid, purpose="voice_cloning"))
        grant_consent(sid, scope=["voice_cloning", "commercial_tts"])
        self.assertTrue(verify_speaker_consent(sid, purpose="voice_cloning"))
        self.assertTrue(verify_speaker_consent(sid, purpose="commercial_tts"))
        # 付与していない purpose は拒否
        self.assertFalse(verify_speaker_consent(sid, purpose="training"))
        revoke_consent(sid)
        self.assertFalse(verify_speaker_consent(sid, purpose="voice_cloning"))

    def test_register_new_consented_source(self):
        src = "my-org/internal_consented_corpus"
        self.assertFalse(verify_speaker_consent("spk_b", purpose="training", source=src))
        register_consented_source(src)
        self.assertTrue(verify_speaker_consent("spk_b", purpose="training", source=src))


if __name__ == "__main__":
    unittest.main()
