import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import deepinv as dinv

from src.generate.generate import ddim_sticky_classifier_guidance
from src.generate.classifier_model import WideResNet_noisy

torch.set_default_dtype(torch.float32)

def logits_fn(x):
    x1, x2 = x[:, 0], x[:, 1]
    l_y = x1**2 + x2**2
    l_r = -x1**2 - x2**2 + 4
    return torch.stack([l_y, l_r], dim=1)

def margin(x):
    l = logits_fn(x)
    return l[:, 0] - l[:, 1]

def grad_margin(x):
    x = x.clone().detach().requires_grad_(True)
    m = margin(x).sum()
    (g,) = torch.autograd.grad(m, x)
    return g.detach()

def step_to_band(x, lr=0.12, eps=0.05):
    m = margin(x)
    g = grad_margin(x)
    return x - lr * (m-eps) * g

def make_trajectories(starts, steps=400, lr=0.12, eps=0.5):
    trajs = []
    for s in starts:
        x = torch.tensor(s[None, :], dtype=torch.float32)
        path = [x.squeeze(0).numpy().copy()]
        for _ in range(steps):
            x = step_to_band(x, lr=lr, eps=eps)
            path.append(x.squeeze(0).numpy().copy())
        trajs.append(np.stack(path, 0))
    return trajs

def make_grid(xlim=(-4,4), ylim=(-4,4), n=400):
    gx, gy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    g = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    m_grid = margin(g).view(gx.shape).numpy()
    return gx, gy, m_grid

def render_gif(out_path="./approach_boundary.gif", steps=60, eps=0.8, xlim=(-4,4), ylim=(-4,4), lr=0.12):
    starts = np.array([
        [x, y] for x in np.linspace(-3, -3, 6) for y in np.linspace(4, -3, 6)
    ], dtype=np.float32)

    trajs = make_trajectories(starts, steps=steps, lr=lr, eps=eps)
    gx, gy, m_grid = make_grid(xlim, ylim, n=500)

    frames = []
    for t in range(1, steps+1):
        fig, ax = plt.subplots(figsize=(6,5))
        cf = ax.contourf(gx, gy, m_grid, levels=[-1e9, 0, 1e9], alpha=0.25)
        ax.contour(gx, gy, m_grid, levels=[0.0], linewidths=2)
        ax.contour(gx, gy, m_grid, levels=[eps], linestyles="--")

        # Plot up to time t for each trajectory
        for traj in trajs:
            seg = traj[:t]
            ax.plot(seg[:,0], seg[:,1], "o:", markersize=1)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title("Points approaching the decision boundary")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close(fig)

    imageio.mimsave(out_path, frames, duration=0.3)
    return out_path


if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # load models
    print("Loading models...")
    denoiser = dinv.models.DiffUNet(in_channels=3, out_channels=3, pretrained=None).to(device)
    checkpoint = torch.load("./models_checkpoint/denoiser_199.pt", weights_only=False, map_location=device)
    denoiser.load_state_dict(checkpoint)
    denoiser = torch.nn.DataParallel(denoiser)

    classifier = WideResNet_noisy(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
    classifier = torch.nn.DataParallel(classifier)
    checkpoint = torch.load("./models_checkpoint/best_epoch210.pt", weights_only=False, map_location=device)
    classifier.load_state_dict(checkpoint["model"])

    
