"""Datasets: CIFAR-10/100 with a fixed held-out split carved from the training set.

The held-out split (default 5,000 images) is used for per-step temperature
scaling and for validating halting policies; the official test set is only
used for final reporting.
"""

from __future__ import annotations

import torch
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
        from PIL import Image

        return self.transform(Image.fromarray(self.images[i])), self.targets[i]


def _base(name, root, train, transform, download, fake):
    if fake:
        return _FakeCifar(2000 if train else 500, NUM_CLASSES[name], transform, seed=0 if train else 1)
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
) -> dict[str, Dataset]:
    """Return {'train', 'holdout', 'test', 'train_eval'} datasets.

    'train_eval' is the training portion with eval transforms (for caching).
    """
    if name not in STATS:
        raise ValueError(f"unknown dataset {name!r}")
    if fake:
        holdout = min(holdout, 400)
    train_aug = _base(name, root, True, train_transform(name, randaug, erase), download, fake)
    train_plain = _base(name, root, True, eval_transform(name), download, fake)
    test = _base(name, root, False, eval_transform(name), download, fake)
    perm = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(seed)).tolist()
    ho_idx, tr_idx = perm[:holdout], perm[holdout:]
    return {
        "train": Subset(train_aug, tr_idx),
        "holdout": Subset(train_plain, ho_idx),
        "test": test,
        "train_eval": Subset(train_plain, tr_idx),
    }
