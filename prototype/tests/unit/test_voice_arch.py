"""
音声合成アーキテクチャの新規実装（2026 ハイブリッド系）の単体テスト。

  - 問題1: SemanticAcousticCodec — 意味(低レート離散)/音響(高レート連続)の分離。
  - 問題2: HybridARFM — 意味LM(AR) → 音響FM(Flow Matching) の合成 TTS。
  - 問題7: MOSPredictor / tts_dpo_loss — 自動品質推定と選好アライメント。
tokenizer 非依存の小さなモデルで検証する。
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from swi_voice.codec import NeuralAudioCodec


class TestSemanticAcousticCodec(unittest.TestCase):
    def _make(self, sem_cb=64, ds=4):
        from swi_voice.codec.semantic_acoustic import SemanticAcousticCodec
        codec = NeuralAudioCodec(dim=16, hidden=8, n_quantizers=2, codebook_size=32)
        codec.eval()
        return SemanticAcousticCodec(codec, semantic_codebook_size=sem_cb, semantic_downsample=ds)

    def test_semantic_is_lower_rate_than_acoustic(self):
        sac = self._make(sem_cb=64, ds=4)
        wav = torch.randn(2, 1, 480 * 8)
        ac = sac.extract_acoustic_latent(wav)              # [2, T_ac, 16]
        sem = sac.extract_semantic_tokens(wav)             # [2, T_sem]
        self.assertEqual(ac.shape[-1], 16)
        self.assertLess(sem.shape[1], ac.shape[1])         # 意味は低フレームレート
        self.assertTrue((sem >= 0).all() and (sem < 64).all())

    def test_decode_acoustic_is_finite(self):
        sac = self._make()
        wav = torch.randn(1, 1, 480 * 8)
        ac = sac.extract_acoustic_latent(wav)
        out = sac.decode_acoustic(ac)
        self.assertTrue(torch.isfinite(out).all())

    def test_semantic_vq_commitment_trains(self):
        torch.manual_seed(0)
        sac = self._make(sem_cb=32, ds=2)
        wav = torch.randn(2, 1, 480 * 8)
        params = list(sac.semantic_proj.parameters()) + list(sac.semantic_vq.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)

        def loss_val():
            with torch.no_grad():
                return sac(wav)[1].item()
        init = loss_val()
        for _ in range(50):
            opt.zero_grad()
            _, vq_loss = sac(wav)
            vq_loss.backward()
            opt.step()
        self.assertLess(loss_val(), init * 0.8)            # コミットメント損失が有意に減少


class TestHybridARFM(unittest.TestCase):
    def _make(self):
        from swi_voice.model.hybrid_ar_fm import HybridARFM
        torch.manual_seed(0)
        return HybridARFM(text_vocab_size=64, semantic_codebook_size=32, acoustic_latent_dim=16,
                          ar_dim=32, ar_layers=1, ar_heads=2,
                          fm_dim=32, fm_layers=2, fm_heads=2, fm_max_frames=64,
                          max_text_len=32, max_sem_len=16)

    def test_semantic_and_acoustic_losses_train(self):
        # 単一 (text, semantic, acoustic) を overfit し、両段の損失が有意に下がる（学習が機能）。
        m = self._make()
        text = torch.randint(0, 64, (1, 6))
        sem = torch.randint(0, 32, (1, 5))
        ac = torch.randn(1, 12, 16)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)

        def sem_loss():
            with torch.no_grad():
                return m.semantic_loss(text, sem).item()

        def ac_loss():
            with torch.no_grad():
                return sum(m.acoustic_loss(sem, ac).item() for _ in range(4)) / 4

        si, ai = sem_loss(), ac_loss()
        for _ in range(200):
            opt.zero_grad()
            loss = m.semantic_loss(text, sem) + m.acoustic_loss(sem, ac)
            loss.backward()
            opt.step()
        self.assertLess(sem_loss(), si * 0.7, "semantic LM did not learn")
        self.assertLess(ac_loss(), ai * 0.7, "acoustic FM did not learn")

    def test_generate_end_to_end_finite(self):
        # テキスト -> 意味(AR) -> 音響latent(FM) -> コーデックデコードで有限波形（端から端）。
        m = self._make()
        m.eval()
        codec = NeuralAudioCodec(dim=16, hidden=8, n_quantizers=2, codebook_size=32)
        codec.eval()
        text = torch.randint(0, 64, (1, 6))
        latent, sem = m.generate(text, n_frames=8, max_sem_frames=10, n_steps=6)
        self.assertEqual(tuple(latent.shape), (1, 8, 16))
        self.assertTrue(torch.isfinite(latent).all())
        self.assertGreaterEqual(sem.shape[1], 1)
        with torch.no_grad():
            wav = codec.decoder(latent.transpose(1, 2))
        self.assertTrue(torch.isfinite(wav).all())


class TestMOSAndDPO(unittest.TestCase):
    def test_mos_predictor_range_and_regresses(self):
        from swi_voice.eval.mos_predictor import MOSPredictor, mos_regression_loss
        torch.manual_seed(0)
        m = MOSPredictor(hidden=16, n_layers=3)
        wav = torch.randn(4, 1, 480 * 8)
        score = m(wav)
        self.assertEqual(tuple(score.shape), (4,))
        self.assertTrue((score >= 1.0).all() and (score <= 5.0).all())   # MOS は [1,5]
        target = torch.tensor([4.5, 2.0, 3.0, 1.5])
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)

        def loss_v():
            with torch.no_grad():
                return mos_regression_loss(m(wav), target).item()
        init = loss_v()
        for _ in range(120):
            opt.zero_grad()
            mos_regression_loss(m(wav), target).backward()
            opt.step()
        self.assertLess(loss_v(), init * 0.5)                            # 目標MOSへ回帰

    def test_dpo_loss_increases_chosen_margin(self):
        import copy
        from swi_voice.model import ARCodecLM
        from swi_voice.train.dpo_mos import tts_dpo_loss, ar_logprob
        torch.manual_seed(0)
        policy = ARCodecLM(text_vocab_size=64, n_codebooks=2, codebook_size=16,
                           dim=32, n_layers=1, n_heads=2)
        reference = copy.deepcopy(policy)
        for p in reference.parameters():
            p.requires_grad_(False)
        text = torch.randint(0, 64, (1, 5))
        chosen = torch.randint(0, 16, (1, 6, 2))
        rejected = torch.randint(0, 16, (1, 6, 2))
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

        def margin():
            with torch.no_grad():
                return (ar_logprob(policy, text, chosen) - ar_logprob(policy, text, rejected)).item()
        m0 = margin()
        for _ in range(120):
            opt.zero_grad()
            tts_dpo_loss(policy, reference, text, chosen, rejected, beta=0.5).backward()
            opt.step()
        self.assertGreater(margin(), m0)                                 # chosen を相対的に選好

    def test_dpo_with_mos_step_end_to_end(self):
        import copy
        from swi_voice.model import ARCodecLM
        from swi_voice.eval.mos_predictor import MOSPredictor
        from swi_voice.train.dpo_mos import dpo_with_mos_step
        codec = NeuralAudioCodec(dim=16, hidden=8, n_quantizers=2, codebook_size=16)
        codec.eval()
        policy = ARCodecLM(text_vocab_size=64, n_codebooks=2, codebook_size=16,
                           dim=32, n_layers=1, n_heads=2)
        reference = copy.deepcopy(policy)
        mos = MOSPredictor(hidden=16, n_layers=3)
        text = torch.randint(0, 64, (1, 5))
        a = torch.randint(0, 16, (1, 6, 2))
        b = torch.randint(0, 16, (1, 6, 2))
        loss = dpo_with_mos_step(policy, reference, mos, codec, text, a, b, beta=0.1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()                                                  # 勾配が流れる


class TestStyleControl(unittest.TestCase):
    def test_style_encoder_paths(self):
        from swi_voice.model.style_control import StyleEncoder
        se = StyleEncoder(dim=32, n_emotions=8, ref_latent_dim=16)
        self.assertEqual(tuple(se(emotion_id=torch.tensor([0, 3])).shape), (2, 32))
        self.assertEqual(tuple(se(ref_latent=torch.randn(2, 5, 16)).shape), (2, 32))
        self.assertEqual(tuple(se(emotion_id=torch.tensor([1, 2]),
                                  ref_latent=torch.randn(2, 5, 16)).shape), (2, 32))
        self.assertIsNone(se())                                   # 指定なしは無条件(None)

    def test_flowtts_style_changes_output(self):
        from swi_voice.model import FlowTTS, speech_infilling_loss
        from swi_voice.model.style_control import StyleEncoder
        torch.manual_seed(0)
        m = FlowTTS(text_vocab_size=32, latent_dim=16, dim=32, n_layers=2, n_heads=2,
                    max_frames=32, max_text_len=16, use_style=True)
        se = StyleEncoder(dim=32, n_emotions=4)
        text = torch.randint(0, 32, (2, 5))
        s0 = se(emotion_id=torch.tensor([0, 0])).detach()        # StyleEncoder は本テストで固定
        s1 = se(emotion_id=torch.tensor([1, 1])).detach()
        # FlowTTS は AdaLN-Zero で初期出力が 0（条件付け不感）のため、数ステップ学習して
        # 条件付け経路（style を含む）を活性化してから感情差を検証する。
        lat = torch.randn(2, 8, 16)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(40):
            opt.zero_grad()
            speech_infilling_loss(m, lat, text, style_emb=s0).backward()
            opt.step()
        m.eval()
        xt = torch.randn(2, 8, 16); t = torch.rand(2); ctx = torch.zeros(2, 8, 16)
        with torch.no_grad():
            v_s0 = m(xt, t, text, ctx, style_emb=s0)
            v_s1 = m(xt, t, text, ctx, style_emb=s1)
        self.assertFalse(torch.allclose(v_s0, v_s1))             # 感情で速度場が変わる

    def test_arcodec_style_and_backward_compat(self):
        from swi_voice.model import ARCodecLM
        from swi_voice.model.style_control import StyleEncoder
        # C-1: 既定(use_style=False)は style_proj を持たない＝旧チェックポイント互換
        m_off = ARCodecLM(text_vocab_size=64, n_codebooks=2, codebook_size=16,
                          dim=32, n_layers=1, n_heads=2)
        self.assertFalse(any("style_proj" in k for k in m_off.state_dict()))
        torch.manual_seed(0)
        m = ARCodecLM(text_vocab_size=64, n_codebooks=2, codebook_size=16,
                      dim=32, n_layers=1, n_heads=2, use_style=True)
        m.eval()
        se = StyleEncoder(dim=32, n_emotions=4)
        text = torch.randint(0, 64, (1, 5)); codes = torch.randint(0, 16, (1, 6, 2))
        with torch.no_grad():
            l0, _ = m(text, codes, style_emb=se(emotion_id=torch.tensor([0])))
            l1, _ = m(text, codes, style_emb=se(emotion_id=torch.tensor([3])))
        self.assertFalse(torch.allclose(l0[0], l1[0]))           # 感情で logits が変わる


class TestDurationRegulator(unittest.TestCase):
    def test_length_regulator_expands_by_durations(self):
        from swi_voice.model.duration import LengthRegulator
        lr = LengthRegulator()
        x = torch.randn(1, 3, 8)
        dur = torch.tensor([[2, 1, 3]])
        out, lengths = lr(x, dur)
        self.assertEqual(tuple(out.shape), (1, 6, 8))              # sum(dur)=6
        self.assertEqual(int(lengths[0]), 6)
        self.assertTrue(torch.allclose(out[0, 0], x[0, 0]))       # token0 を 2 回
        self.assertTrue(torch.allclose(out[0, 1], x[0, 0]))
        self.assertTrue(torch.allclose(out[0, 2], x[0, 1]))       # token1 を 1 回
        self.assertTrue(torch.allclose(out[0, 3], x[0, 2]))       # token2 を 3 回

    def test_duration_predictor_trains(self):
        from swi_voice.model.duration import DurationPredictor, duration_loss
        torch.manual_seed(0)
        dp = DurationPredictor(dim=16, hidden=16)
        emb = torch.randn(2, 5, 16)
        target = torch.randint(1, 8, (2, 5))
        opt = torch.optim.Adam(dp.parameters(), lr=1e-2)

        def lv():
            with torch.no_grad():
                return duration_loss(dp(emb), target).item()
        init = lv()
        for _ in range(80):
            opt.zero_grad()
            duration_loss(dp(emb), target).backward()
            opt.step()
        self.assertLess(lv(), init * 0.5)                          # 継続時間の回帰が学習

    def test_duration_adaptor_end_to_end(self):
        from swi_voice.model.duration import DurationAdaptor
        da = DurationAdaptor(dim=16, hidden=16)
        da.eval()
        emb = torch.randn(1, 4, 16)
        expanded, log_dur, lengths = da(emb)
        self.assertEqual(tuple(log_dur.shape), (1, 4))
        self.assertEqual(expanded.shape[0], 1)
        self.assertEqual(expanded.shape[2], 16)
        self.assertEqual(expanded.shape[1], max(1, int(lengths[0])))   # 伸長長 = 予測 duration 総和
        self.assertTrue(torch.isfinite(expanded).all())


class TestS2SBackbone(unittest.TestCase):
    def test_forward_overfits(self):
        from swi_voice.duplex.s2s_backbone import SpeechToSpeechBackbone
        torch.manual_seed(0)
        m = SpeechToSpeechBackbone(n_codebooks=2, codebook_size=16, dim=32, n_layers=1, n_heads=2)
        user = torch.randint(0, 16, (1, 6, 2))
        agent = torch.randint(0, 16, (1, 6, 2))
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)

        def lv():
            with torch.no_grad():
                return m(user, agent)[1].item()
        init = lv()
        for _ in range(150):
            opt.zero_grad()
            _, loss = m(user, agent)
            loss.backward()
            opt.step()
        self.assertLess(lv(), init * 0.5)                          # dual-stream 次フレーム予測が学習

    def test_generate_shape_and_valid(self):
        from swi_voice.duplex.s2s_backbone import SpeechToSpeechBackbone
        m = SpeechToSpeechBackbone(n_codebooks=2, codebook_size=16, dim=32, n_layers=1, n_heads=2)
        user = torch.randint(0, 16, (1, 5, 2))
        agent = m.generate(user, temperature=0.0)
        self.assertEqual(tuple(agent.shape), (1, 5, 2))            # ユーザー長に同期
        self.assertTrue((agent >= 0).all() and (agent < 16).all())


class TestVocoder(unittest.TestCase):
    def test_mel_to_waveform_and_overfit(self):
        from swi_voice.model.vocoder import Vocoder, waveform_to_mel, vocoder_loss
        torch.manual_seed(0)
        voc = Vocoder(n_mels=80, hidden=64, upsample_rates=(2, 4, 4, 5, 3))
        target = torch.randn(1, 1, 480 * 6) * 0.3
        mel = waveform_to_mel(target)                              # [1, 80, ~7]
        out = voc(mel)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 1)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(out.abs().max().item(), 1.0 + 1e-4)  # tanh 出力
        opt = torch.optim.Adam(voc.parameters(), lr=1e-3)

        def lv():
            with torch.no_grad():
                return vocoder_loss(voc(mel), target).item()
        init = lv()
        for _ in range(60):
            opt.zero_grad()
            vocoder_loss(voc(mel), target).backward()
            opt.step()
        self.assertLess(lv(), init * 0.85)                        # mel→波形の再構成が学習


class TestFlowDistill(unittest.TestCase):
    def test_one_step_distillation_reduces_gap(self):
        import copy
        from swi_voice.model import FlowTTS, speech_infilling_loss
        from swi_voice.train.flow_distill import flow_distill_loss
        torch.manual_seed(0)
        teacher = FlowTTS(text_vocab_size=32, latent_dim=16, dim=32, n_layers=2, n_heads=2,
                          max_frames=32, max_text_len=16)
        text = torch.randint(0, 32, (1, 5))
        lat = torch.randn(1, 8, 16)
        topt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
        for _ in range(60):                                       # teacher を軽く学習
            topt.zero_grad()
            speech_infilling_loss(teacher, lat, text).backward()
            topt.step()
        student = copy.deepcopy(teacher)
        sopt = torch.optim.Adam(student.parameters(), lr=1e-3)
        fixed_noise = [torch.randn(1, 8, 16) for _ in range(8)]

        def gap():                                               # 1step vs teacher 16step の平均ギャップ
            with torch.no_grad():
                return sum(flow_distill_loss(student, teacher, text, 8, teacher_steps=16, noise=n).item()
                           for n in fixed_noise) / len(fixed_noise)
        init = gap()
        for _ in range(150):
            sopt.zero_grad()
            flow_distill_loss(student, teacher, text, 8, teacher_steps=16).backward()
            sopt.step()
        self.assertLess(gap(), init * 0.7)                       # 1ステップ生成が多ステップ生成に接近


class TestDriftMonitor(unittest.TestCase):
    def test_detects_quality_drop(self):
        from swi_voice.eval.drift_monitor import DriftMonitor
        dm = DriftMonitor(window=5, drop_threshold=0.5)
        dm.set_baseline([4.0, 4.2, 3.8])                          # 基準 ~4.0
        for s in [4.0, 3.9, 4.1, 4.0, 3.9]:                       # 正常域
            dm.observe(s)
        drifted, _ = dm.check_drift()
        self.assertFalse(drifted)
        for s in [3.0, 2.8, 3.1, 2.9, 3.0]:                       # 劣化
            dm.observe(s)
        drifted, delta = dm.check_drift()
        self.assertTrue(drifted)
        self.assertGreater(delta, 0.5)

    def test_score_uses_mos_predictor(self):
        from swi_voice.eval.mos_predictor import MOSPredictor
        from swi_voice.eval.drift_monitor import DriftMonitor
        dm = DriftMonitor(mos_predictor=MOSPredictor(hidden=16, n_layers=3), window=10)
        s = dm.score(torch.randn(1, 1, 480 * 4))
        self.assertTrue(1.0 <= s <= 5.0)
        self.assertEqual(len(dm.history), 1)

    def test_collect_negatives_for_continual(self):
        from swi_voice.eval.drift_monitor import collect_negatives_for_continual
        negs = collect_negatives_for_continual([(4.0, "a"), (2.0, "b"), (1.5, "c")], mos_threshold=2.5)
        self.assertEqual(negs, ["b", "c"])                        # 低MOS のみ負サンプルに


if __name__ == "__main__":
    unittest.main()
