"""
ゼロショットクローニングの同意ゲート (swi_voice/model/cloning.py) の単体テスト [音声編4.1 / 8.1]。

  - 話者IDなしの参照音声は拒否（誰の声か特定できない声は模倣しない）。
  - 未登録話者は拒否。公開コーパスの source を添えても voice_cloning は許可されない。
  - 明示 grant で許可、revoke で再び拒否。
  - 拒否は例外（CloningConsentError）で表面化し、無言でクローニングなし合成に落ちない。
  - 拒否した試行は悪用監視カウンタに記録される [音声編12.2]。
  - 同意が無ければ参照波形のエンコードにも到達しない（ゲートが最初に走る）。
"""
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swi_voice.model.cloning import (
    CloningConsentError, verify_reference_consent, require_reference_consent,
    zero_shot_clone_inference, encode_reference_prompt,
    denied_cloning_attempts, reset_denied_cloning_attempts,
)
from swi_voice.safety.consent_check import grant_consent, revoke_consent
from swi_voice.utils.constants import VOICE_DATASET_CANDIDATES

PUBLIC_SOURCE = VOICE_DATASET_CANDIDATES[0][0]


class _SpyCodec:
    """encode/decode が呼ばれたかを記録するスタブコーデック。"""

    def __init__(self, n_frames=4, n_q=2):
        self.n_frames, self.n_q = n_frames, n_q
        self.encode_calls, self.decode_calls = 0, 0

    def encode(self, wav):
        self.encode_calls += 1
        return torch.zeros((1, self.n_frames, self.n_q), dtype=torch.long)

    def decode(self, codes):
        self.decode_calls += 1
        return torch.zeros((1, 1, codes.shape[1] * 8))


class _SpyModel:
    def __init__(self, n_frames=3, n_q=2):
        self.n_frames, self.n_q = n_frames, n_q
        self.generate_calls = 0

    def generate(self, text_ids, prompt_codes=None, max_frames=200,
                 temperature=0.8, top_k=50, style_emb=None):
        self.generate_calls += 1
        return torch.zeros((1, self.n_frames, self.n_q), dtype=torch.long)


class TestCloningConsentGate(unittest.TestCase):
    def setUp(self):
        reset_denied_cloning_attempts()

    def test_empty_speaker_id_denied(self):
        self.assertFalse(verify_reference_consent(""))
        self.assertFalse(verify_reference_consent(None))

    def test_unregistered_speaker_denied(self):
        self.assertFalse(verify_reference_consent("some_random_person"))

    def test_public_source_does_not_grant_cloning(self):
        # 公開コーパスのデータセット単位同意には voice_cloning を含めない
        self.assertFalse(verify_reference_consent("spk_public", source=PUBLIC_SOURCE))

    def test_grant_then_revoke(self):
        sid = "cloning_actor_gate_001"
        self.assertFalse(verify_reference_consent(sid))
        grant_consent(sid, scope=["voice_cloning"])
        self.assertTrue(verify_reference_consent(sid))
        revoke_consent(sid)
        self.assertFalse(verify_reference_consent(sid))

    def test_require_raises_and_counts_denials(self):
        with self.assertRaises(CloningConsentError):
            require_reference_consent("denied_speaker_x")
        with self.assertRaises(CloningConsentError):
            require_reference_consent("denied_speaker_x")
        self.assertEqual(denied_cloning_attempts().get("denied_speaker_x"), 2)

    def test_require_records_unspecified_speaker(self):
        with self.assertRaises(CloningConsentError):
            require_reference_consent("")
        self.assertEqual(denied_cloning_attempts().get("<unspecified>"), 1)

    def test_require_passes_for_consented_speaker(self):
        sid = "cloning_actor_gate_002"
        grant_consent(sid, scope=["voice_cloning"])
        require_reference_consent(sid)           # 例外を投げなければ合格
        revoke_consent(sid)

    def test_clone_inference_denied_before_touching_audio(self):
        """同意が無ければ codec.encode / model.generate に到達しない（ゲートが最初）。"""
        codec, model = _SpyCodec(), _SpyModel()
        with self.assertRaises(CloningConsentError):
            zero_shot_clone_inference(model, torch.zeros((1, 4), dtype=torch.long),
                                      torch.randn(1600), codec, speaker_id="no_consent_person")
        self.assertEqual(codec.encode_calls, 0)
        self.assertEqual(model.generate_calls, 0)
        self.assertEqual(codec.decode_calls, 0)

    def test_clone_inference_runs_for_consented_speaker(self):
        sid = "cloning_actor_gate_003"
        grant_consent(sid, scope=["voice_cloning"])
        codec, model = _SpyCodec(), _SpyModel()
        out = zero_shot_clone_inference(model, torch.zeros((1, 4), dtype=torch.long),
                                        torch.randn(1600), codec, speaker_id=sid)
        self.assertIsNotNone(out)
        self.assertEqual(codec.encode_calls, 1)
        self.assertEqual(model.generate_calls, 1)
        revoke_consent(sid)

    def test_custom_consent_check_fn_can_deny(self):
        codec, model = _SpyCodec(), _SpyModel()
        with self.assertRaises(CloningConsentError):
            zero_shot_clone_inference(model, torch.zeros((1, 4), dtype=torch.long),
                                      torch.randn(1600), codec, speaker_id="anyone",
                                      consent_check_fn=lambda sid: False)
        self.assertEqual(codec.encode_calls, 0)

    def test_prompt_truncated_to_max_frames(self):
        codec = _SpyCodec(n_frames=500)
        codes = encode_reference_prompt(torch.randn(1600), codec, max_prompt_frames=150)
        self.assertEqual(codes.shape[1], 150)

    def test_empty_reference_returns_none(self):
        # V-2: 空テンソルは WARNING を出して安全にフォールバック（例外にしない）
        self.assertIsNone(encode_reference_prompt(torch.zeros(0), _SpyCodec()))
        self.assertIsNone(encode_reference_prompt(None, _SpyCodec()))


if __name__ == "__main__":
    unittest.main()
