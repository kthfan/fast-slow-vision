"""Batch-level Mixup / CutMix producing soft targets."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class Mixup:
    def __init__(self, num_classes: int, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0,
                 prob: float = 1.0, switch_prob: float = 0.5):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    @property
    def enabled(self) -> bool:
        return self.prob > 0 and (self.mixup_alpha > 0 or self.cutmix_alpha > 0)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = F.one_hot(y, self.num_classes).float()
        if not self.enabled or np.random.rand() > self.prob:
            return x, target
        use_cutmix = self.cutmix_alpha > 0 and (self.mixup_alpha <= 0 or np.random.rand() < self.switch_prob)
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            H, W = x.shape[-2:]
            r = np.sqrt(1.0 - lam)
            ch, cw = int(H * r), int(W * r)
            cy, cx = np.random.randint(H), np.random.randint(W)
            y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
            x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
            x = x.clone()
            x[..., y1:y2, x1:x2] = x.flip(0)[..., y1:y2, x1:x2]
            lam = 1.0 - (y2 - y1) * (x2 - x1) / float(H * W)
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            x = lam * x + (1.0 - lam) * x.flip(0)
        return x, lam * target + (1.0 - lam) * target.flip(0)
