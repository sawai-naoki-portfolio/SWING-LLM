"""
swi_voice/train/flow_distill.py
低遅延ストリーミングのための Flow Matching 蒸留 [音声編7.2, アーキ乖離 問題8]

Flow Matching(FlowTTS) は既定 16 ステップの Euler ODE で高音質だが、対話用途の TTFB
（初回音声までの時間）を詰めるにはステップ数を減らしたい。Rectified-Flow の reflow 蒸留で、
教師（多ステップ）の生成軌跡に生徒（1 ステップ）の生成を一致させ、**1 ステップ生成**を獲得する。

student_out = noise - student(noise, t=1, cond)   （dt=1 の単一 Euler ステップ）
loss = MSE(student_out, teacher_multi_step_out)   （同一ノイズ・同一条件で）
"""
import torch
import torch.nn.functional as F

from swi_llm.utils.logger import setup_logger

logger = setup_logger("voice.flow_distill")


@torch.no_grad()
def ode_from_noise(model, text_ids, noise, n_steps, style_emb=None):
    """与えたノイズから Euler ODE を n_steps 実行して latent を返す（flow_tts.sample と同じ更新則）。"""
    model.eval()
    B = noise.shape[0]
    device = noise.device
    x = noise.clone()
    context = torch.zeros_like(noise)
    dt = 1.0 / n_steps
    for i in range(n_steps, 0, -1):
        t = torch.full((B,), i / n_steps, device=device)
        v = model(x, t, text_ids, context, style_emb=style_emb)
        x = x - v * dt
    return x


def flow_distill_loss(student, teacher, text_ids, n_frames, teacher_steps=16, style_emb=None, noise=None):
    """1-step reflow 蒸留損失。teacher の多ステップ生成に student の 1 ステップ生成を一致させる。"""
    device = text_ids.device
    B = text_ids.shape[0]
    D = student.latent_dim
    if noise is None:
        noise = torch.randn(B, n_frames, D, device=device)
    with torch.no_grad():
        teacher_out = ode_from_noise(teacher, text_ids, noise, teacher_steps, style_emb=style_emb)
    context = torch.zeros(B, n_frames, D, device=device)
    t1 = torch.ones(B, device=device)
    v = student(noise, t1, text_ids, context, style_emb=style_emb)   # t=1 の速度
    student_out = noise - v                                          # 単一 Euler ステップ(dt=1)
    return F.mse_loss(student_out, teacher_out)


@torch.no_grad()
def sample_one_step(student, text_ids, n_frames, style_emb=None, device=None):
    """蒸留済み student で 1 ステップ生成する（低遅延推論）。返り値 latent [B, n_frames, D]。"""
    device = device or text_ids.device
    B = text_ids.shape[0]
    D = student.latent_dim
    noise = torch.randn(B, n_frames, D, device=device)
    context = torch.zeros(B, n_frames, D, device=device)
    t1 = torch.ones(B, device=device)
    return noise - student(noise, t1, text_ids, context, style_emb=style_emb)
