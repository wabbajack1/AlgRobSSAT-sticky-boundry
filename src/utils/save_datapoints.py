import os, json, hashlib, time
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

class SynWriter:
    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        self.manifest_path = os.path.join(root, "manifest.jsonl")
        self._fh = open(self.manifest_path, "a", buffering=1)
        self._idx = 0

    def write(self, x0, *, seed, target_label, guidance_scale, eta, num_steps,
                denoiser_ckpt, classifier_ckpt,
                logits=None, probs=None, predicted=None, margin=None):
        """
        x0: CHW tensor in [-1,1]
        """
        self._idx += 1
        img_id = f"{self._idx:08d}"
        img_path = os.path.join(self.root, "images", f"{img_id}.png")

        # save lossless PNG (assumes x0 in [-1,1])
        save_image((x0.clamp(-1,1)+1)/2.0, img_path)

        rec = {
            "id": img_id,
            "path": f"images/{img_id}.png",
            "seed": int(seed),
            "target_label": int(target_label),
            "guidance_scale": float(guidance_scale),
            "eta": float(eta),
            "num_steps": int(num_steps),
            "denoiser_ckpt": str(denoiser_ckpt),
            "classifier_ckpt": str(classifier_ckpt),
            "created_utc": int(time.time()),
            "norm": "[-1,1]->PNG",
        }
        if logits is not None:
            rec["logits"] = logits.detach().cpu().tolist()
            rec["logits_top5"] = torch.topk(logits, k=min(5, logits.numel()), dim=-1).values.detach().cpu().tolist()
        if probs is not None:
            top5 = torch.topk(probs, k=min(5, probs.numel()), dim=-1)
            rec["top5_idx"] = top5.indices.detach().cpu().tolist()
            rec["top5_prob"] = top5.values.detach().cpu().tolist()
        if predicted is not None:
            rec["pred_label"] = int(predicted)
        if margin is not None:
            rec["margin_top1_top2"] = float(margin)

        self._fh.write(json.dumps(rec) + "\n")

    def close(self):
        self._fh.close()


@torch.no_grad()
def save_batch(
    writer,
    x0_batch,                 # [B, C, H, W] in [-1,1]
    seeds,                    # list/1D tensor length B (per-sample seeds you used)
    guidance_scale,
    eta,
    num_steps,
    denoiser_ckpt,
    classifier_ckpt,
    classifier,
    device,
    target_labels=None,            # int or LongTensor [B]
    ):
    B = x0_batch.size(0)

    # classify the clean images (t=0)
    print(f"Classifying {B} datapoints...")
    if B > 1000:
        # classify in batches to avoid OOM
        logits, probs, pred, margins = [], [], [], []
        for i in tqdm(range(0, B, 1000)):
            x0_batch_i = x0_batch[i:i+1000]
            logits_i, probs_i, pred_i, margins_i = classify_clean_batch(x0_batch_i, classifier, device)
            logits.append(logits_i)
            probs.append(probs_i)
            pred.append(pred_i)
            margins.append(margins_i)
        logits = torch.cat(logits, dim=0)
        probs = torch.cat(probs, dim=0)
        pred = torch.cat(pred, dim=0)
        margins = torch.cat(margins, dim=0)
    else:
        logits, probs, pred, margins = classify_clean_batch(x0_batch, classifier, device)
    
    # allow scalar or vector guidance scale/labels
    if isinstance(target_labels, int):
        target_labels = torch.full((B,), target_labels, dtype=torch.long)
    if isinstance(guidance_scale, (int, float)):
        guidance_scale = torch.full((B,), float(guidance_scale))

    if target_labels is None:
        target_labels = pred # use predicted labels if not provided
    guidance_scale = torch.tensor(guidance_scale).cpu()

    print(f"Saving {B} datapoints to {writer.root}...")
    for i in tqdm(range(B)):
        writer.write(
            x0=x0_batch[i].cpu(),                     # CHW in [-1,1]
            seed=int(seeds[i]),
            target_label=int(target_labels[i].cpu()),
            guidance_scale=float(guidance_scale[i]),
            eta=float(eta),
            num_steps=int(num_steps),
            denoiser_ckpt=denoiser_ckpt,
            classifier_ckpt=classifier_ckpt,
            logits=logits[i].cpu(),
            probs=probs[i].cpu(),
            predicted=int(pred[i].cpu()),
            margin=float(margins[i].cpu()),
        )


@torch.no_grad()
def classify_clean_batch(x0_batch, classifier, device):
    B = x0_batch.size(0)
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    logits = classifier(x0_batch.to(device), t0)    # [B,K]
    probs  = logits.softmax(dim=-1)
    pred   = probs.argmax(dim=-1)
    top2   = probs.topk(2, dim=-1).values
    margin = (top2[:,0] - top2[:,1])
    return logits, probs, pred, margin

def accept_mask(pred, target, margin, min_margin=None, max_margin=None):
    m = (pred == target)
    if min_margin is not None:
        m = m & (margin >= min_margin)
    if max_margin is not None:
        m = m & (margin <= max_margin)
    return m