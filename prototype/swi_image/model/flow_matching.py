"""
swi_image/model/flow_matching.py

Rectified Flow の学習損失とサンプラー [画像編3・5]。

ノイズ x1 とデータ x0 を結ぶ直線経路上の（一定な）速度場 v = x1 - x0 を回帰する。
DDPM の epsilon 予測より単純で、経路が直線的な分サンプリング効率が良い。
サンプリングは Euler ODE ソルバー + Classifier-Free Guidance [画像編7.1]。
"""
import torch
import torch.nn.functional as F


def rectified_flow_loss(dit, latents, text_cond):
    """潜在 x0 に対する Rectified Flow 損失 [画像編5.1]。"""
    b = latents.size(0)
    t = torch.rand(b, device=latents.device)
    noise = torch.randn_like(latents)
    t_ = t.view(b, 1, 1, 1)
    xt = (1 - t_) * latents + t_ * noise            # 直線補間
    target_v = noise - latents                       # 直線の速度は一定
    pred_v = dit(xt, t, text_cond["pooled"], text_cond["sequence"])
    return F.mse_loss(pred_v, target_v)


@torch.no_grad()
def sample_euler(dit, shape, text_cond, uncond=None, n_steps=28, cfg_scale=4.0, device="cuda"):
    """ノイズ(t=1)からデータ(t=0)へ Euler で逆積分。返り値はスケール済み潜在 [画像編5]。"""
    x = torch.randn(shape, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps, 0, -1):
        t = torch.full((shape[0],), i / n_steps, device=device)
        v_cond = dit(x, t, text_cond["pooled"], text_cond["sequence"])
        if uncond is not None and cfg_scale != 1.0:
            v_uncond = dit(x, t, uncond["pooled"], uncond["sequence"])
            v = v_uncond + cfg_scale * (v_cond - v_uncond)      # CFG
        else:
            v = v_cond
        x = x - v * dt
    return x
