"""
swi_voice のコードレビュー修正（V-4 barge-in セッション化 / V-5 クロスフェード /
V-6 プレースホルダーの明示化）の単体テスト。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestSessionStopSignal(unittest.TestCase):
    """V-4: 停止シグナルがセッションごとに独立していること。"""

    def test_sessions_are_independent(self):
        from swi_voice.duplex.barge_in import SessionStopSignal
        a = SessionStopSignal()
        b = SessionStopSignal()
        self.assertFalse(a.is_set())
        self.assertFalse(b.is_set())
        a.set()                                  # セッション A だけ停止
        self.assertTrue(a.is_set())
        self.assertFalse(b.is_set())             # B は影響を受けない（旧グローバルなら止まっていた）
        a.clear()
        self.assertFalse(a.is_set())


class TestStreamOverlapDecode(unittest.TestCase):
    """V-5: デコーダオーバーラップ（左文脈供給＋トリム）で出力長が非オーバーラップと一致すること。"""

    def test_overlap_trim_length_matches_plain(self):
        import numpy as np
        import torch
        from swi_voice.codec import NeuralAudioCodec
        from swi_voice.utils.constants import VOICE_STREAM_OVERLAP_FRAMES
        codec = NeuralAudioCodec(dim=16, hidden=8, n_quantizers=2, codebook_size=32)
        codec.eval()
        hop = codec.hop
        overlap = VOICE_STREAM_OVERLAP_FRAMES
        chunk = torch.randint(0, 32, (1, 8, 2))
        tail = torch.randint(0, 32, (1, overlap, 2))
        with torch.no_grad():
            w_plain = codec.decode(chunk).squeeze().cpu().numpy()
            w_ov = codec.decode(torch.cat([tail, chunk], dim=1)).squeeze().cpu().numpy()
        trim = min(overlap * hop, max(0, w_ov.shape[0] - 1))
        w_trimmed = w_ov[trim:]
        self.assertEqual(w_trimmed.shape[0], w_plain.shape[0])   # トリム後は非オーバーラップと同長
        self.assertTrue(np.isfinite(w_trimmed).all())


class TestVAD(unittest.TestCase):
    """V-6: RMS ベース VAD の最小実装が発話/無音を判定できること。

    fix 13 でニューラル VAD（silero）が既定になったため、これらは backend="rms" を
    明示してフォールバック経路の契約を固定する（DC 0.3 は「音量が大きい」だけで
    音声ではない — silero は正しく棄却するので、auto のままだと期待が反転する）。
    ニューラル側の挙動は test_voice_vad_neural.py が検証する。
    """

    def test_silence_is_inactive(self):
        import torch
        from swi_voice.duplex.vad import detect_voice_activity
        self.assertFalse(detect_voice_activity(torch.zeros(2400), backend="rms"))

    def test_loud_signal_is_active(self):
        import torch
        from swi_voice.duplex.vad import detect_voice_activity, compute_rms_energy
        sig = torch.ones(2400) * 0.3
        self.assertGreater(compute_rms_energy(sig), 0.02)
        self.assertTrue(detect_voice_activity(sig, backend="rms"))

    def test_empty_and_none(self):
        import torch
        from swi_voice.duplex.vad import detect_voice_activity
        self.assertFalse(detect_voice_activity(None))
        self.assertFalse(detect_voice_activity(torch.zeros(0)))

    def test_handle_barge_in_uses_vad(self):
        import torch
        from swi_voice.duplex.barge_in import handle_barge_in
        self.assertTrue(handle_barge_in(torch.ones(2400) * 0.3, backend="rms"))  # 大音量 → 割り込み
        self.assertFalse(handle_barge_in(torch.zeros(2400), backend="rms"))      # 無音 → 継続


class TestDurationPredictor(unittest.TestCase):
    """V-6: 継続時間予測器の最小実装が学習・推論できること。"""

    def test_forward_shape_and_backward(self):
        import torch
        from swi_voice.model.duration import DurationPredictor, duration_loss
        m = DurationPredictor(dim=32, hidden=16)
        emb = torch.randn(2, 7, 32)
        log_dur = m(emb)
        self.assertEqual(tuple(log_dur.shape), (2, 7))
        target = torch.randint(1, 10, (2, 7))
        loss = duration_loss(log_dur, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_predict_frames_nonnegative_int(self):
        import torch
        from swi_voice.model.duration import DurationPredictor
        m = DurationPredictor(dim=32, hidden=16)
        frames = m.predict_frames(torch.randn(1, 5, 32))
        self.assertEqual(frames.dtype, torch.long)
        self.assertTrue((frames >= 0).all())


class TestPlaceholdersFailLoudly(unittest.TestCase):
    """データ品質モジュール（forced_align/asr_pseudo_label）が実装され、silent failure
    ではなく明示的な None/[] 返しや警告で不正入力を扱うことを確認する
    （実装内容そのものの検証は test_voice_forced_align.py / test_voice_asr_pseudo_label.py）。"""

    def test_forced_align_rejects_invalid_input(self):
        import torch
        from swi_voice.data.forced_align import align_text_and_audio
        self.assertIsNone(align_text_and_audio("", torch.zeros(480 * 4)))   # 空テキスト
        self.assertIsNone(align_text_and_audio("a", None))                  # 音声なし

    def test_asr_pseudo_label_rejects_missing_model(self):
        import torch
        from swi_voice.data.asr_pseudo_label import pseudo_label_and_filter
        self.assertEqual(pseudo_label_and_filter([{"audio": torch.zeros(10)}], None), [])

    def test_router_returns_cascade(self):
        # cascade_router は「実装済み経路へ確定ルーティング」の最小実装（silent failure ではない）
        from swi_voice.agents.cascade_router import AgentRouter, ROUTE_CASCADE
        self.assertEqual(AgentRouter().route_request({"x": 1}), ROUTE_CASCADE)


if __name__ == "__main__":
    unittest.main()
