import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from deepinv.models import DiffUNet

from src.generate.generate import ddim_sticky_classifier_guidance
from src.generate.classifier_model import WideResNet_noisy

def to_img(x): return (x.clamp(-1, 1) + 1) / 2

def test_boundry(classifier, denoiser):
     # CIFAR-10 label map
    labels = {"airplane":0,"automobile":1,"bird":2,"cat":3,"deer":4,"dog":5,"frog":6,"horse":7,"ship":8,"truck":9}
    target_label = "automobile"
    target_idx = labels[target_label]

    # sweep settings
    boundary_scales = torch.linspace(0, 1, 10)
    num_samples    = 50
    guidance_scale = 4
    T_plot         = 1  # only for plotting axis readability
    margin_eps = 0
    band = 1

    # 2) SWEEP WITH CALIBRATED ε
    stats = []
    all_margins_plot = []
    images_dict = []

    for bs in boundary_scales.tolist():
        x_hat = ddim_sticky_classifier_guidance(
                classifier, denoiser,
                target_label=target_label,
                guidance_scale=guidance_scale,
                boundary_scale=bs,
                margin_eps=margin_eps,
                num_samples=num_samples,
                num_steps_denoiser=100,
                eta=0,
                anneal_boundary=False,

                # sticky knobs
                tether_width=1,
                recover_gain=5,
                clip_norm=False,
                stick_tol=0.1,
                freeze_last_steps=0,
                alpha=0.85,
            )

        # save images
        images_dict.append({
            "boundary_scale": bs,
            "images": x_hat.detach().cpu().numpy()
        })
        


        with torch.no_grad():
            t0 = torch.zeros(x_hat.size(0), dtype=torch.long, device=x_hat.device)
            logits = classifier(x_hat, t0)

            target = torch.full((x_hat.size(0),), target_idx, device=x_hat.device, dtype=torch.long)
            logit_y = logits.gather(1, target[:, None]).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, target[:, None], False)
            logit_rival = logits.masked_fill(~mask, float('-inf')).max(dim=1).values
            m_target = logit_y - logit_rival


            hit_rate = (logits.argmax(dim=1) == target).float().mean().item()
            neg_rate = (m_target < 0).float().mean().item()
            near_lo, near_hi = margin_eps - band/2, margin_eps + band/2
            near_rate = ((m_target >= near_lo) & (m_target <= near_hi)).float().mean().item()

            m_plot = (m_target / T_plot).detach().cpu().numpy()
            stats.append(dict(
                boundary_scale=float(bs),
                margin_mean_plot=float(m_plot.mean()),
                margin_std_plot=float(m_plot.std()),
                hit_rate=hit_rate,
                near_rate=near_rate,
                neg_rate=neg_rate,
            ))
            all_margins_plot.append(m_plot)

    # 3) PLOTS
    xs = [s["boundary_scale"] for s in stats]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot margin mean and standard deviation with error bars
    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_ylabel("target margin m = y*-y_rival", color=color)
    ax1.errorbar(xs, [s["margin_mean_plot"] for s in stats],
                 yerr=[s["margin_std_plot"] for s in stats],
                 marker='o', linestyle='-', color=color, label="Margin mean ± std")
    ax1.axhline(margin_eps / T_plot, linestyle='--', color='gray', label="ε")
    ax1.axhline((margin_eps - band/2) / T_plot, linestyle=':', linewidth=1, color='gray')
    ax1.axhline((margin_eps + band/2) / T_plot, linestyle=':', linewidth=1, color='gray')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper left")
    ax1.set_title("Margin Analysis")

    # Plot class consistency and boundary proximity
    ax2 = axes[1]
    color = 'tab:orange'
    ax2.set_xlabel("boundary_scale")
    ax2.set_ylabel("fraction", color=color)
    ax2.plot(xs, [s["hit_rate"] for s in stats], marker='o', label="target hit-rate", color='tab:blue')
    ax2.plot(xs, [s["near_rate"] for s in stats], marker='o', label=f"within band [ε±{band/2:.2f}]", color='tab:orange')
    ax2.plot(xs, [s["neg_rate"] for s in stats], marker='x', label="m < 0 (flipped)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    ax2.set_title("Class Consistency and Boundary Proximity")

    # plot some images
    num_images_to_plot = 5
    fig, img_axes = plt.subplots(len(images_dict), num_images_to_plot, figsize=(10, 3 * len(images_dict)))

    for i, img_dict in enumerate(images_dict):
        bs = img_dict["boundary_scale"]
        images = img_dict["images"][:num_images_to_plot]  # select few images to plot
        for j, img in enumerate(images):
            ax = img_axes[i, j] if len(images_dict) > 1 else img_axes[j]
            ax.imshow(to_img(torch.tensor(img)).permute(1, 2, 0).numpy())
            if j == 0:
                ax.set_ylabel(f"bs={bs:.2f}", fontsize=12, rotation=0, labelpad=40, va='center')
    
    plt.tight_layout()
    plt.show()


def test_alpha_scale(classifier, denoiser):
    # CIFAR-10 label map
    labels = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
    target_label = "automobile"
    target_idx = labels[target_label]

    # sweep settings
    alpha_scales = torch.linspace(0, 1, 10)
    num_samples = 50
    guidance_scale = 4
    T_plot = 1  # only for plotting axis readability
    margin_eps = 1
    band = 1

    # 2) SWEEP WITH CALIBRATED ε
    stats = []
    all_margins_plot = []
    images_dict = []

    for alpha in alpha_scales.tolist():
        x_hat = ddim_sticky_classifier_guidance(
            classifier, denoiser,
            target_label=target_label,
            guidance_scale=guidance_scale,
            boundary_scale=0.3,
            margin_eps=margin_eps,
            num_samples=num_samples,
            num_steps_denoiser=5,
            eta=0,
            anneal_boundary=False,

            # sticky knobs
            tether_width=1,
            recover_gain=5,
            clip_norm=False,
            stick_tol=0.1,
            freeze_last_steps=0,
            alpha=alpha,
        )

        # save images
        images_dict.append({
            "alpha_scale": alpha,
            "images": x_hat.detach().cpu().numpy()
        })

        with torch.no_grad():
            t0 = torch.zeros(x_hat.size(0), dtype=torch.long, device=x_hat.device)
            logits = classifier(x_hat, t0)

            target = torch.full((x_hat.size(0),), target_idx, device=x_hat.device, dtype=torch.long)
            logit_y = logits.gather(1, target[:, None]).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, target[:, None], False)
            logit_rival = logits.masked_fill(~mask, float('-inf')).max(dim=1).values
            m_target = logit_y - logit_rival

            hit_rate = (logits.argmax(dim=1) == target).float().mean().item()
            neg_rate = (m_target < 0).float().mean().item()
            near_lo, near_hi = margin_eps - band / 2, margin_eps + band / 2
            near_rate = ((m_target >= near_lo) & (m_target <= near_hi)).float().mean().item()

            m_plot = (m_target / T_plot).detach().cpu().numpy()
            stats.append(dict(
                alpha_scale=float(alpha),
                margin_mean_plot=float(m_plot.mean()),
                margin_std_plot=float(m_plot.std()),
                hit_rate=hit_rate,
                near_rate=near_rate,
                neg_rate=neg_rate,
            ))
            all_margins_plot.append(m_plot)

    # 3) PLOTS
    xs = [s["alpha_scale"] for s in stats]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot margin mean and standard deviation with error bars
    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_ylabel("target margin m = y*-y_rival", color=color)
    ax1.errorbar(xs, [s["margin_mean_plot"] for s in stats],
                    yerr=[s["margin_std_plot"] for s in stats],
                    marker='o', linestyle='-', color=color, label="Margin mean ± std")
    ax1.axhline(margin_eps / T_plot, linestyle='--', color='gray', label="ε")
    ax1.axhline((margin_eps - band / 2) / T_plot, linestyle=':', linewidth=1, color='gray')
    ax1.axhline((margin_eps + band / 2) / T_plot, linestyle=':', linewidth=1, color='gray')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper left")
    ax1.set_title("Margin Analysis")

    # Plot class consistency and boundary proximity
    ax2 = axes[1]
    color = 'tab:orange'
    ax2.set_xlabel("alpha_scale")
    ax2.set_ylabel("fraction", color=color)
    ax2.plot(xs, [s["hit_rate"] for s in stats], marker='o', label="target hit-rate", color='tab:blue')
    ax2.plot(xs, [s["near_rate"] for s in stats], marker='o', label=f"within band [ε±{band / 2:.2f}]", color='tab:orange')
    ax2.plot(xs, [s["neg_rate"] for s in stats], marker='x', label="m < 0 (flipped)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    ax2.set_title("Class Consistency and Alpha Proximity")

    # plot some images
    num_images_to_plot = 5
    fig, img_axes = plt.subplots(len(images_dict), num_images_to_plot, figsize=(10, 3 * len(images_dict)))

    for i, img_dict in enumerate(images_dict):
        alpha = img_dict["alpha_scale"]
        images = img_dict["images"][:num_images_to_plot]  # select few images to plot
        for j, img in enumerate(images):
            ax = img_axes[i, j] if len(images_dict) > 1 else img_axes[j]
            ax.imshow(to_img(torch.tensor(img)).permute(1, 2, 0).numpy())
            if j == 0:
                ax.set_ylabel(f"α={alpha:.2f}", fontsize=12, rotation=0, labelpad=40, va='center')

    plt.tight_layout()
    plt.show()

def test_recovery_scale(classifier, denoiser):
    # CIFAR-10 label map
    labels = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
    target_label = "automobile"
    target_idx = labels[target_label]

    # sweep settings
    recovery_scales = torch.linspace(0, 10, 10)
    num_samples = 50
    guidance_scale = 4
    T_plot = 1  # only for plotting axis readability
    margin_eps = 0
    band = 1

    # 2) SWEEP WITH CALIBRATED ε
    stats = []
    all_margins_plot = []
    images_dict = []

    for recovery_gain in recovery_scales.tolist():
        x_hat = ddim_sticky_classifier_guidance(
            classifier, denoiser,
            target_label=target_label,
            guidance_scale=guidance_scale,
            boundary_scale=0.3,
            margin_eps=margin_eps,
            num_samples=num_samples,
            num_steps_denoiser=5,
            eta=0,
            anneal_boundary=False,

            # sticky knobs
            tether_width=1,
            recover_gain=recovery_gain,
            clip_norm=False,
            stick_tol=0.1,
            freeze_last_steps=0,
            alpha=0.85,
        )

        # save images
        images_dict.append({
            "recovery_scale": recovery_gain,
            "images": x_hat.detach().cpu().numpy()
        })

        with torch.no_grad():
            t0 = torch.zeros(x_hat.size(0), dtype=torch.long, device=x_hat.device)
            logits = classifier(x_hat, t0)

            target = torch.full((x_hat.size(0),), target_idx, device=x_hat.device, dtype=torch.long)
            logit_y = logits.gather(1, target[:, None]).squeeze(1)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, target[:, None], False)
            logit_rival = logits.masked_fill(~mask, float('-inf')).max(dim=1).values
            m_target = logit_y - logit_rival

            hit_rate = (logits.argmax(dim=1) == target).float().mean().item()
            neg_rate = (m_target < 0).float().mean().item()
            near_lo, near_hi = margin_eps - band / 2, margin_eps + band / 2
            near_rate = ((m_target >= near_lo) & (m_target <= near_hi)).float().mean().item()

            m_plot = (m_target / T_plot).detach().cpu().numpy()
            stats.append(dict(
                recovery_scale=float(recovery_gain),
                margin_mean_plot=float(m_plot.mean()),
                margin_std_plot=float(m_plot.std()),
                hit_rate=hit_rate,
                near_rate=near_rate,
                neg_rate=neg_rate,
            ))
            all_margins_plot.append(m_plot)

    # 3) PLOTS
    xs = [s["recovery_scale"] for s in stats]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot margin mean and standard deviation with error bars
    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_ylabel("target margin m = y*-y_rival", color=color)
    ax1.errorbar(xs, [s["margin_mean_plot"] for s in stats],
                    yerr=[s["margin_std_plot"] for s in stats],
                    marker='o', linestyle='-', color=color, label="Margin mean ± std")
    ax1.axhline(margin_eps / T_plot, linestyle='--', color='gray', label="ε")
    ax1.axhline((margin_eps - band / 2) / T_plot, linestyle=':', linewidth=1, color='gray')
    ax1.axhline((margin_eps + band / 2) / T_plot, linestyle=':', linewidth=1, color='gray')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper left")
    ax1.set_title("Margin Analysis")

    # Plot class consistency and recovery proximity
    ax2 = axes[1]
    color = 'tab:orange'
    ax2.set_xlabel("recovery_scale")
    ax2.set_ylabel("fraction", color=color)
    ax2.plot(xs, [s["hit_rate"] for s in stats], marker='o', label="target hit-rate", color='tab:blue')
    ax2.plot(xs, [s["near_rate"] for s in stats], marker='o', label=f"within band [ε±{band / 2:.2f}]", color='tab:orange')
    ax2.plot(xs, [s["neg_rate"] for s in stats], marker='x', label="m < 0 (flipped)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    ax2.set_title("Class Consistency and Recovery Proximity")

    # plot some images
    num_images_to_plot = 5
    fig, img_axes = plt.subplots(len(images_dict), num_images_to_plot, figsize=(10, 3 * len(images_dict)))

    for i, img_dict in enumerate(images_dict):
        recovery_gain = img_dict["recovery_scale"]
        images = img_dict["images"][:num_images_to_plot]  # select few images to plot
        for j, img in enumerate(images):
            ax = img_axes[i, j] if len(images_dict) > 1 else img_axes[j]
            ax.imshow(to_img(torch.tensor(img)).permute(1, 2, 0).numpy())
            if j == 0:
                ax.set_ylabel(f"rec={recovery_gain:.2f}", fontsize=12, rotation=0, labelpad=40, va='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # set mps device for mac
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    # load models
    print("Loading models...")
    denoiser = DiffUNet(in_channels=3, out_channels=3, pretrained=None).to(device)
    checkpoint = torch.load("./models_checkpoint/denoiser_199.pt", weights_only=False, map_location=device)
    denoiser.load_state_dict(checkpoint)
    denoiser = torch.nn.DataParallel(denoiser)

    classifier = WideResNet_noisy(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
    classifier = torch.nn.DataParallel(classifier)
    checkpoint = torch.load("./models_checkpoint/best_epoch210.pt", weights_only=False, map_location=device)
    classifier.load_state_dict(checkpoint["model"])

    denoiser.to(device)
    classifier.to(device)

    denoiser.eval()
    classifier.eval()

    print("Testing boundary approach...")
    test_boundry(classifier, denoiser)
    #test_alpha_scale(classifier, denoiser)
    # test_recovery_scale(classifier, denoiser)