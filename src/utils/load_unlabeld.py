import os, json, math
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torchvision.transforms as T

normalize = T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

# define transforms
tf = T.Compose([
        T.ToTensor(),
        normalize,
])

class CombineManifestDataset(Dataset):
    def __init__(self, root1, root2, manifest_glob="manifest.jsonl", 
            mmin=0, 
            mmax=1, 
            tau_keep=0, 
            use_pred_if_no_target=True,
            beta=0.5,
            alpha=0.5):

            # root1 and root2 are paths to the directories containing the manifest files of different datasets
            # beta is a hyperparameter for selecting samples from the two datasets
            # alpha is a hyperparameter for size of the dataset to be used
            self.root1 = Path(root1)
            self.root2 = Path(root2)
            self.beta = beta
            self.alpha = alpha
            self.items = []

            # Load items from root1
            items1 = self._load_items(self.root1, manifest_glob, mmin, mmax, tau_keep, use_pred_if_no_target)
            # Load items from root2
            items2 = self._load_items(self.root2, manifest_glob, mmin, mmax, tau_keep, use_pred_if_no_target)

            # choose a subset of items from both datasets based on alpha
            len1 = int(len(items1) * alpha)
            len2 = int(len(items2) * alpha)
            items1 = items1[:len1]
            items2 = items2[:len2]

            # Combine items from both datasets based on beta
            len1 = int(len(items1) * beta)
            len2 = int(len(items2) * (1 - beta))
            combined_items = items1[:len1] + items2[:len2]
            self.items = combined_items

            if len(self.items) == 0:
                raise RuntimeError("No items matched filters; relax mmin/mmax/tau_keep.")
    
    def _load_items(self, root, manifest_glob, mmin, mmax, tau_keep, use_pred_if_no_target):
        items = []
        for mpath in sorted(root.glob(manifest_glob)):
            with open(mpath, "r") as fh:
                for line in fh:
                    rec = json.loads(line)
                    # filters (boundary + confidence)
                    margin = float(rec.get("margin_top1_top2", 1.0))
                    top = max(rec.get("top5_prob", [1.0]))
                    if not (mmin <= margin <= mmax and top >= tau_keep):
                        continue
                    y = rec.get("target_label", None)
                    if y is None and use_pred_if_no_target:
                        y = rec.get("pred_label", None)
                    if y is None:
                        continue
                    rel = rec["path"] if "path" in rec else f"{rec['id']}.png"
                    items.append((str(root / rel), int(y)))
        return items

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        x = tf(img)
        return x, y

class ManifestDataset(Dataset):
    def __init__(self, root, mmin=0, mmax=1, tau_keep=0.9, 
                 use_pred_if_no_target=True,
                 alpha=0.5 # alpha is a hyperparameter for size of the dataset to be used
                ):
        # root is the path to the directory containing the manifest file
        self.use_pred_if_no_target = use_pred_if_no_target
        self.root = Path(root)
        self.mmin = mmin
        self.mmax = mmax
        self.tau_keep = tau_keep
        self.alpha = alpha
        self.items = []



        manifest_path = self.root / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

        with open(manifest_path, "r") as fh:
            for line in fh:
                rec = json.loads(line)
                margin = float(rec.get("margin_top1_top2", 1.0))
                top = max(rec.get("top5_prob", [1.0]))
                if not (mmin <= margin <= mmax and top >= tau_keep):
                    continue
                y = rec.get("target_label", None)
                if y is None:
                    y = rec.get("pred_label", None)
                if y is None:
                    continue
                rel = rec["path"] if "path" in rec else f"{rec['id']}.png"
                self.items.append((str(self.root / rel), int(y)))

        # Adjust the size of the dataset based on alpha
        max_items = int(len(self.items) * alpha)
        self.items = self.items[:max_items]
        
        if len(self.items) == 0:
            raise RuntimeError("No items matched filters; relax mmin/mmax/tau_keep.")

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        x = tf(img)
        return x, y

def make_unlabeled_loader_from_manifest(steps_per_epoch, 
                                        batch_size,
                                        ds):


    # 50k samples per epoch: sample exactly S*B_u with replacement
    unlabeled_sampler = RandomSampler(ds, replacement=False, num_samples=steps_per_epoch * batch_size)
    
    return DataLoader(ds, batch_size=batch_size, sampler=unlabeled_sampler,drop_last=False)
