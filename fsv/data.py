"""Datasets: CIFAR-10/100 with a fixed held-out split carved from the training set.

The held-out split (default 5,000 images) is used for per-step temperature
scaling and for validating halting policies; the official test set is only
used for final reporting.

Data source: by default the parquet copies on the Hugging Face Hub
(uoft-cs/cifar10, uoft-cs/cifar100), decoded once and cached as
``<root>/<name>_<split>.npz``. The original Toronto mirror used by torchvision
is very slow from some hosts (about 50 kB/s from Kaggle); it remains available
with ``source="torchvision"`` and as a fallback.
"""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms as T

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def train_transform(name: str, randaug: bool = True, erase: float = 0.25):
    mean, std = STATS[name]
    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaug:
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if erase > 0:
        ops.append(T.RandomErasing(p=erase))
    return T.Compose(ops)


def eval_transform(name: str):
    mean, std = STATS[name]
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


HF_SOURCES = {
    # name: (repo, subdir, label column)
    "cifar10": ("uoft-cs/cifar10", "plain_text", "label"),
    "cifar100": ("uoft-cs/cifar100", "cifar100", "fine_label"),
}


class ArrayDataset(Dataset):
    """uint8 images [N, H, W, 3] + integer targets."""

    def __init__(self, images: np.ndarray, targets, transform):
        self.images = images
        self.targets = list(map(int, targets))
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), self.targets[i]


def _hf_arrays(name: str, root: str, split: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
    root_p = Path(root)
    cache = root_p / f"{name}_{split}.npz"
    if not cache.exists():
        if not download:
            raise FileNotFoundError(cache)
        import pyarrow.parquet as pq

        repo, sub, label_col = HF_SOURCES[name]
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{sub}/{split}-00000-of-00001.parquet"
        root_p.mkdir(parents=True, exist_ok=True)
        print(f"[data] downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            table = pq.read_table(io.BytesIO(r.read()))
        img_col = next(c for c in table.column_names if c in ("img", "image"))
        images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                           for d in table.column(img_col).to_pylist()])
        labels = np.asarray(table.column(label_col).to_pylist(), dtype=np.int64)
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, images=images, labels=labels)
        tmp.replace(cache)
    z = np.load(cache)
    return z["images"], z["labels"]


class _FakeCifar(Dataset):
    """Deterministic random images for smoke tests (no download)."""

    def __init__(self, n: int, num_classes: int, transform, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.images = (torch.rand(n, 32, 32, 3, generator=g) * 255).to(torch.uint8).numpy()
        self.targets = torch.randint(0, num_classes, (n,), generator=g).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), self.targets[i]


def _base(name, root, train, transform, download, fake, source="hf"):
    if fake:
        return _FakeCifar(2000 if train else 500, NUM_CLASSES[name], transform, seed=0 if train else 1)
    if source == "hf":
        try:
            images, labels = _hf_arrays(name, root, "train" if train else "test", download)
            return ArrayDataset(images, labels, transform)
        except Exception as e:  # fall back to torchvision's mirror
            print(f"[data] Hugging Face source failed ({e!r}); falling back to torchvision", flush=True)
    cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
    return cls(root, train=train, transform=transform, download=download)


def build_splits(
    name: str = "cifar100",
    root: str = "data",
    holdout: int = 5000,
    seed: int = 0,
    download: bool = True,
    fake: bool = False,
    randaug: bool = True,
    erase: float = 0.25,
    source: str = "hf",
) -> dict[str, Dataset]:
    """Return {'train', 'holdout', 'test', 'train_eval'} datasets.

    'train_eval' is the training portion with eval transforms (for caching).
    """
    if name not in STATS:
        raise ValueError(f"unknown dataset {name!r}")
    if fake:
        holdout = min(holdout, 400)
    train_aug = _base(name, root, True, train_transform(name, randaug, erase), download, fake, source)
    train_plain = _base(name, root, True, eval_transform(name), download, fake, source)
    test = _base(name, root, False, eval_transform(name), download, fake, source)
    perm = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(seed)).tolist()
    ho_idx, tr_idx = perm[:holdout], perm[holdout:]
    return {
        "train": Subset(train_aug, tr_idx),
        "holdout": Subset(train_plain, ho_idx),
        "test": test,
        "train_eval": Subset(train_plain, tr_idx),
    }
