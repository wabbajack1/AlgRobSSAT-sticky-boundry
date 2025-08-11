import deepinv as dinv
from deepinv.utils import plot
import torch
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Grayscale, Normalize
from deepinv.physics import GaussianNoise
from src.utils.data_loader import get_train_valid_loader, get_test_loader
from src.utils.save_datapoints import classify_clean_batch, accept_mask, save_batch, SynWriter
import os
from datetime import datetime
import os, json, math, itertools
import hashlib
import torch.nn.functional as F
from tqdm.notebook import trange, tqdm
from src.generate.hyperparams import Generation
from src.generate.classifier_model import WideResNet_noisy
import math
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def seed_for(class_id:int, idx:int, run_seed:int)->int:
    # stateless 64-bit seed from (class, index, run_seed)
    h = hashlib.sha256(f"{class_id}-{idx}-{run_seed}".encode()).digest()
    return int.from_bytes(h[:8], "little", signed=False)

def randn_batch_from_seeds(B, C, H, W, seeds, device):
    xs = []
    for s in seeds:
        g = torch.Generator(device=device).manual_seed(s)
        xs.append(torch.randn((1, C, H, W), generator=g, device=device))
    return torch.cat(xs, dim=0)


########################### DDIM Classifier Guidance ###########################


def ddim_classifier_guidance(classifier, denoiser, guidance_scale=4, target_label="automobile", num_steps_denoiser=20, eta=0, save_every=5, num_samples=10, init_seeds=None):

    ### diffusion variances; THESE ARE FIXED; DO NOT ALTER ######################
    beta_start = 1e-4
    beta_end = 0.02
    num_timesteps = 1000

    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
    #############################################################################

    #-------
    # CONFIG: classifier guidance settings and models
    #-------
    labels = {"airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,}
    
    classifier.eval()                    # pretrained noisy-aware classifier
    denoiser.eval()
    classifier.to(device)
    denoiser.to(device)

    guidance_scale = guidance_scale      # w: 0 = off, 1–5 typical
    target_class = labels[target_label]    # <-- desired class name here, gets mapped to index (int or LongTensor)

    # how many reverse steps
    num_steps_denoiser = num_steps_denoiser      # try 20, 50, 100
    eta = 0             # 0 = deterministic DDIM; >0 adds noise like DDPM
    save_every = 5      # save every k-th step into `res`
    T = len(betas)      # number of Timesteps --> gets subsampled (skipped)

    def to_img(x):  # x in [-1,1] -> [0,1] for display only
        return (x.clamp(-1,1) + 1) / 2


    #  build a subsampled schedule t_0 < t_1 < ... < t_{S-1} in [0, T-1]
    timesteps = torch.linspace(0, T-1, steps=num_steps_denoiser, dtype=torch.long, device=device)

    res = []
    with torch.no_grad():
        if init_seeds is not None:
            x_t = randn_batch_from_seeds(num_samples, 3, 32, 32, init_seeds, device=device)
        else:
            x_t = torch.randn(num_samples, 3, 32, 32, device=device)

        # walk the schedule in reverse: t = t_{S-1}, t_{S-2}, ..., t_0
        for k in range(num_steps_denoiser-1, -1, -1):
            t = timesteps[k].item()
            prev_t = timesteps[k-1].item() if k > 0 else -1  # -1 means ᾱ_prev = 1

            t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long, device=device)

            # gather coefficients for current and "previous" (subsampled) step
            alpha_t        = alphas[t].view(1,1,1,1)
            alpha_bar_t    = alphas_bar[t].view(1,1,1,1)
            alpha_bar_prev = (torch.ones_like(alpha_bar_t) if prev_t < 0
                              else alphas_bar[prev_t].view(1,1,1,1))

            #---------
            # DDIM: predict noise epsilon_t
            #---------
            eps = denoiser(x_t, t_tensor, type_t="timestep")[:, :3, ...]

            #---------
            # eps_guided = eps - w * sqrt(1 - alpha_bar_t) * ∇_x log p(y|x_t,t)
            #---------
            # Buidl p per-sample target class tensor
            if isinstance(target_class, int):
                y = torch.full((x_t.size(0),), target_class, device=device, dtype=torch.long)
            else:
                y = target_class.to(device).long()  # allow a LongTensor of size [B]

            # Compute classifier gradient wrt x_t at this timestep
            with torch.enable_grad():
                x_in = x_t.detach().clone().requires_grad_(True).to(device)
                logits = classifier(x_in, t_tensor)          # shape [B, num_classes]
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs.gather(1, y.view(-1,1)).sum()  # sum over batch for a scalar
                grad = torch.autograd.grad(selected, x_in, create_graph=False, retain_graph=False)[0]

            eps = eps - guidance_scale * torch.sqrt(1.0 - alpha_bar_t) * grad
            # no graph needed beyond this point
            #---------

            # reconstruct x0
            x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # DDIM variance and direction for (t -> prev_t) jump
            # sigma_t = eta * sqrt( (1-ᾱ_prev)/(1-ᾱ_t) * (1-α_t) )
            sigma_t = eta * torch.sqrt(((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * (1.0 - alpha_t))
            c_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0))

            # DDIM update:
            # x_{prev} = sqrt(ᾱ_prev) * x0_hat + c_t * eps + sigma_t * z
            x_prev = torch.sqrt(alpha_bar_prev) * x0_hat + c_t * eps
            if prev_t >= 0 and eta > 0:
                x_prev = x_prev + sigma_t * torch.randn_like(x_t)

            x_t = x_prev

            if ((k % save_every == 0) or (k == num_steps_denoiser-1) or (k == 0)) and save_every>0:
                res.append(to_img(x_t.detach().clone().cpu()))

    return x_t # [-1, 1]


########################### DDIM Entropy Guidance ###########################

def ddim_entropy_guidance(
    classifier,
    denoiser,
    guidance_scale=4.0,                 # overall multiplier for guidance
    target_label="automobile",
    num_steps_denoiser=20,
    eta=0.0,
    save_every=5,
    num_samples=10,
    init_seeds=None,

    #### entropy guidance knobs ####
    ent_weight=1.0,                     # λ: push up entropy
    constraint_weight=1.0,              # μ: enforce p(y*) >= τ
    tau=0.5,                            # τ: min prob for y*
    use_midstep_anneal=True,            # stronger guidance in the middle
    device="cuda"
):
    """
    DDIM sampler with entropy-max under class constraint guidance.
    Assumes `classifier(x, t_long)` is noise-aware (takes timestep) and returns logits.
    Assumes `denoiser(x, t_long, type_t="timestep")` returns ε prediction (channels-first).
    """

    # fixed linear beta schedule
    beta_start, beta_end, num_timesteps = 1e-4, 0.02, 1000
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

    # config
    labels = {"airplane":0,"automobile":1,"bird":2,"cat":3,"deer":4,
              "dog":5,"frog":6,"horse":7,"ship":8,"truck":9}

    classifier.eval()
    denoiser.eval()
    classifier.to(device)
    denoiser.to(device)

    target_class = labels[target_label]
    T = len(betas)

    # subsampled schedule
    timesteps = torch.linspace(0, T-1, steps=num_steps_denoiser, dtype=torch.long, device=device)

    def to_img(x):  # [-1,1] -> [0,1] for viewing
        return (x.clamp(-1,1) + 1) / 2

    res = []

    # init noise
    if init_seeds is not None:
        x_t = randn_batch_from_seeds(num_samples, 3, 32, 32, init_seeds, device=device)
    else:
        x_t = torch.randn(num_samples, 3, 32, 32, device=device)

    for k in range(num_steps_denoiser - 1, -1, -1):
        t = timesteps[k].item()
        prev_t = timesteps[k-1].item() if k > 0 else -1
        t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long, device=device)

        alpha_t        = alphas[t].view(1,1,1,1)
        alpha_bar_t    = alphas_bar[t].view(1,1,1,1)
        alpha_bar_prev = (torch.ones_like(alpha_bar_t) if prev_t < 0
                          else alphas_bar[prev_t].view(1,1,1,1))

        # base denoiser eps
        eps = denoiser(x_t, t_tensor, type_t="timestep")[:, :3, ...]

        ###### entropy guidance ######
        # Build per-sample target class tensor
        if isinstance(target_class, int):
            y = torch.full((x_t.size(0),), target_class, device=device, dtype=torch.long)
        else:
            y = target_class.to(device).long()

        # Compute ∇_x L_t where:
        # L_t = -λ H(p) + μ [ReLU(τ - p_y)]^2
        with torch.enable_grad():
            x_in = x_t.detach().clone().requires_grad_(True)
            logits = classifier(x_in, t_tensor)              # [B, K]
            log_probs = F.log_softmax(logits, dim=-1)        # stable
            probs = log_probs.exp()

            # Entropy H(p) = -sum p log p  (mean over batch)
            entropy = -(probs * log_probs).sum(dim=1).mean()

            # p_y for the target class
            p_y = probs.gather(1, y.view(-1,1)).squeeze(1)   # [B]
            c = torch.clamp(tau - p_y, min=0.0)              # [B]
            constraint_term = (c * c).mean()

            L_t = (-ent_weight * entropy) + (constraint_weight * constraint_term)

            grad = torch.autograd.grad(L_t, x_in, create_graph=False, retain_graph=False)[0]

        # strongest in the middle of the trajectory
        if use_midstep_anneal:
            # s in [0,1], peak near 0.5 using a sine^2 bell
            s = (k + 1) / num_steps_denoiser
            anneal = torch.sin(torch.tensor(torch.pi * s, device=device)) ** 2
            scale = guidance_scale * anneal
        else:
            scale = guidance_scale

        # inject entropy guidance
        eps = eps + scale * torch.sqrt(1.0 - alpha_bar_t) * grad
        #----------------------------------------------------

        # reconstruct x0 and perform DDIM step
        x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        sigma_t = eta * torch.sqrt(((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * (1.0 - alpha_t))
        c_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0))

        x_prev = torch.sqrt(alpha_bar_prev) * x0_hat + c_t * eps
        if prev_t >= 0 and eta > 0:
            x_prev = x_prev + sigma_t * torch.randn_like(x_t)

        x_t = x_prev

        if ((k % save_every == 0) or (k == num_steps_denoiser-1) or (k == 0)) and save_every>0:
            res.append(to_img(x_t.detach().clone().cpu()))

    return x_t  # [-1,1]


########################### DDIM Sticky Guidance ###########################

def ddim_sticky_classifier_guidance(
    classifier,
    denoiser,
    guidance_scale=10,
    target_label="automobile",
    num_steps_denoiser=20,
    eta=0.0,
    save_every=0,
    num_samples=10,
    init_seeds=None,

    #  sticky boundary knobs 
    boundary_scale=3.0,      # overall weight of boundary term
    margin_eps=2.0,          # center of the band (LOGIT units, calibrated)
    tether_width=1.0,        # band width in logits; hi = eps + width, lo = eps
    recover_gain=3.0,        # how strongly to push back if m < eps (anti-flip)
    clip_norm=1.0,           # per-sample grad clip for boundary term
    anneal_boundary=True,    # gate boundary force to the middle of the trajectory
    gate_low=0.2, gate_high=0.85, gate_center=0.55, gate_width=0.20,
    stick_tol=0.05,          # reflect updates that would push m below eps by more than this
    freeze_last_steps=3,     # turn off boundary term in the last k steps
    alpha=0.8,                  # balance between classifier and boundary guidance
):
    """
    Classifier guidance + STICKY boundary guidance.

    Margin: m = ℓ_y* - max_{k≠y*} ℓ_k  (logits).
    Tether band: [eps_lo, eps_hi] with eps_lo = margin_eps, eps_hi = margin_eps + tether_width.
      - If m > eps_hi:      push along -∇m  (reduce margin)
      - If m < eps_lo:      push along +∇m  (recover; strong, via recover_gain)
      - Else:               no boundary force

    Reflection: if m <= eps_lo + stick_tol and the combined update would *decrease* m,
    project out that component so we do not cross further.
    """

    device = next(denoiser.parameters()).device

    # fixed schedule
    beta_start, beta_end, num_timesteps = 1e-4, 0.02, 1000
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    labels = {"airplane":0,"automobile":1,"bird":2,"cat":3,"deer":4,"dog":5,"frog":6,"horse":7,"ship":8,"truck":9}
    target_class = labels[target_label]

    classifier.eval(); denoiser.eval()

    T = len(betas)
    timesteps = torch.linspace(0, T - 1, steps=num_steps_denoiser, dtype=torch.long, device=device)

    def to_img(x): return (x.clamp(-1,1) + 1) / 2

    with torch.no_grad():
        if init_seeds is not None:
            x_t = randn_batch_from_seeds(num_samples, 3, 32, 32, init_seeds, device=device)
        else:
            x_t = torch.randn(num_samples, 3, 32, 32, device=device)

        for k in range(num_steps_denoiser - 1, -1, -1):
            t = timesteps[k].item()
            prev_t = timesteps[k - 1].item() if k > 0 else -1
            t_tensor = torch.full((x_t.size(0),), t, dtype=torch.long, device=device)

            alpha_t        = alphas[t].view(1,1,1,1)
            alpha_bar_t    = alphas_bar[t].view(1,1,1,1)
            alpha_bar_prev = (torch.ones_like(alpha_bar_t) if prev_t < 0 else alphas_bar[prev_t].view(1,1,1,1))

            # base denoiser
            eps = denoiser(x_t, t_tensor, type_t="timestep")[:, :3, ...]

            # grads (enable autograd)
            with torch.enable_grad():
                x_in = x_t.detach().clone().requires_grad_(True) # [B,C,H,W]
                logits = classifier(x_in, t_tensor)  # [B,K]

                # classifier guidance
                y = torch.full((x_t.size(0),), int(target_class), device=device, dtype=torch.long)
                log_probs = F.log_softmax(logits, dim=-1) # [B,K]
                selected = log_probs.gather(1, y[:,None]).sum() # sum over batch for scalar
                g_cls = torch.autograd.grad(selected, x_in, retain_graph=True)[0] # [B,C,H,W], since we take the gradient wrt x_in

                # sticky boundary guidance
                if boundary_scale != 0.0:
                    # rival and margin
                    logits_masked = logits.clone()
                    logits_masked.scatter_(1, y[:,None], float('-inf'))
                    y_hat = logits_masked.argmax(dim=1) # take the rival class, i.e. max_{k≠y*} L_k

                    ly = logits.gather(1, y[:,None]).squeeze(1) # true logit ℓ_y*
                    lr = logits.gather(1, y_hat[:,None]).squeeze(1) # rival logit, i.e. max_{k≠y*} L_k
                    m = ly - lr  # margin

                    # grad m
                    g_m = torch.autograd.grad(m.sum(), x_in, retain_graph=True)[0]  # [B,C,H,W]
                    print(f"Norm of margin grad: {g_m.flatten(1).norm(p=2, dim=-1).mean().item():.3f}")

                    # clip boundary grad
                    if clip_norm is not None and clip_norm > 0:
                        gn = g_m.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-12)
                        scale = (clip_norm / gn).clamp(max=1.0)
                        g_m = g_m * scale.view(-1,1,1,1)

                    # push down if m > eps_hi; push up if m < eps_lo
                    first_term = boundary_scale * (m-margin_eps >= tether_width).float().view(-1,1,1,1) * -g_m # m ≥ ε+W; push down
                    second_term = boundary_scale * (m-margin_eps < 0).float().view(-1,1,1,1) * recover_gain * g_m # m ≤ ε; push up (recover)

                    total_grad = (1-alpha) * guidance_scale * g_cls + \
                                alpha * (first_term + second_term)

                else:
                    total_grad = guidance_scale * g_cls

            # guided score
            eps = eps - torch.sqrt(1.0 - alpha_bar_t) * total_grad

            # ddim update
            x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            sigma_t = eta * torch.sqrt(((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * (1.0 - alpha_t))
            c_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0))
            x_prev = torch.sqrt(alpha_bar_prev) * x0_hat + c_t * eps
            if prev_t >= 0 and eta > 0:
                x_prev = x_prev + sigma_t * torch.randn_like(x_t)

            x_t = x_prev

            if save_every > 0 and ((k % save_every) == 0 or k in {0, num_steps_denoiser-1}):
                _ = to_img(x_t.detach().clone().cpu())

    return x_t

def generate_balanced_dataset(
    per_class=50_000,
    num_classes=10,
    batch_gen=128,
    guidance_scale=4.0,
    eta=0.0,
    min_margin=None,          # e.g., 0.05  (optional boundary control)
    max_margin=None,          # e.g., 0.20
    run_seed=1234,
    device="cuda",
    writer=None,              # your SynWriter
    num_timesteps=None,           # denoising/generative/reverse/sampling timesteps
    denoiser=None,
    classifier=None,
    denoiser_ckpt="", classifier_ckpt=""
):
    labels = {"airplane":
     0,"automobile": 1,
     "bird": 2,
     "cat": 3,
     "deer": 4,
     "dog": 5,
     "frog": 6,
     "horse": 7,
     "ship": 8,
     "truck": 9,}
     
    denoiser.eval()
    classifier.eval()
    counts = {c: 0 for c in range(num_classes)}
    global_idx = {c: 0 for c in range(num_classes)}  # for seed derivation per class

    while True:
        remaining = [(c, per_class - counts[c]) for c in range(num_classes)]
        remaining = [(c, r) for c, r in remaining if r > 0]
        if not remaining:
            break

        for c, r in remaining:
            B = min(batch_gen, max(r * 2, batch_gen))  # oversample a bit to offset rejections
            # Prepare targets and seeds
            seeds = [seed_for(c, global_idx[c] + i, run_seed) for i in range(B)]
            global_idx[c] += B

            # Sample batch -> [B, C, H, W]
            x0_batch = ddim_sticky_classifier_guidance(
                classifier=classifier,
                denoiser=denoiser,
                target_label=list(labels)[c],
                guidance_scale=guidance_scale,
                boundary_scale=4,
                margin_eps=0.2,
                num_samples=batch_gen,
                num_steps_denoiser=5,
                eta=0,
                anneal_boundary=False,

                # sticky knobs
                tether_width=1,
                recover_gain=5,
                clip_norm=None,
                stick_tol=1,
                freeze_last_steps=0,
                init_seeds=seeds
            )

            # eval and filter
            logits, probs, pred, margin = classify_clean_batch(x0_batch, classifier, device)
            mask = accept_mask(pred, c, margin, min_margin, max_margin)
            keep_idx = mask.nonzero(as_tuple=False).view(-1).tolist()

            # save accepted samples
            for i in keep_idx:
                if counts[c] >= per_class:
                    break
                writer.write(
                    x0=x0_batch[i].cpu(),
                    seed=seeds[i],
                    target_label=int(c),
                    guidance_scale=guidance_scale,
                    eta=eta,
                    num_steps=num_timesteps,
                    denoiser_ckpt=denoiser_ckpt,
                    classifier_ckpt=classifier_ckpt,
                    logits=logits[i].cpu(),
                    probs=probs[i].cpu(),
                    predicted=int(pred[i].cpu()),
                    margin=float(margin[i].cpu()),
                )
                counts[c] += 1
            print(f"Progress per class", {c: counts[c] for c in range(num_classes)})
    return counts




if __name__ == "__main__":

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

    denoiser.to(device)
    classifier.to(device)

    
    # create a SynWriter to save the generated datapoints
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SynWriter(f"./conditional_generated_data_sticky/{now}")

    # generate a balanced dataset
    print("Generating balanced dataset...")
    counts = generate_balanced_dataset(
        per_class=Generation.per_class,
        num_classes=Generation.num_classes,
        batch_gen=Generation.batch_gen,
        guidance_scale=Generation.guidance_scale,
        eta=Generation.eta,
        run_seed=Generation.run_seed,
        device=Generation.device,
        writer=writer,
        num_timesteps=Generation.denoising_steps,
        denoiser=denoiser,
        classifier=classifier,
        denoiser_ckpt="model/denoiser_199.pt",
        classifier_ckpt="model/best_epoch210.pt"
    )

    print("Generated counts:", counts)
    writer.close()
    print(f"Datapoints saved to {writer.root}")

