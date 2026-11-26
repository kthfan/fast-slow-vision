"""CIFAR ResNet (He, Zhang, Ren & Sun, 2016, section 4.2): the feed-forward CNN baseline of phase 1.

depth = 6n + 2: a 3x3 stem with 16 channels, three stages of n basic blocks (16, 32, 64 channels at
32, 16, 8 pixels; the first block of stages 2 and 3 has stride 2), global average pooling, one linear
layer. ResNet-20 / 56 / 110 = n 3 / 9 / 18. Shortcuts are option A of the paper (identity; where the
shape changes, subsample by 2 and zero-pad the new channels), so ResNet-56 has 0.86M parameters with a
100-class head, as in the paper's 0.85M (10 classes).

Exposed through the same step interface as bViT and VCNet with a single step (``t_max = 1``):
``embed`` = stem, ``forward_step`` = the three stages, ``readout`` = pool + linear. The training,
evaluation and FLOP-counting code then runs unchanged; halting curves have one point.

``norm``: "bn" (BatchNorm, the paper) or "gn" (GroupNorm, as VCNet), to separate the normalisation from
the architecture when the two are compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResNetConfig:
    arch: str = "resnet"
    img_size: int = 32
    in_chans: int = 3
    num_classes: int = 100
    t_max: int = 1  # one feed-forward pass
    depth: int = 56  # 6n + 2
    width: int = 16  # channels of the first stage (x2, x4 in stages 2, 3)
    norm: str = "bn"  # "bn" | "gn"
    gn_groups: int = 8

    @property
    def dim(self) -> int:
        return 4 * self.width


PRESETS = {
    "resnet20": dict(depth=20),
    "resnet56": dict(depth=56),
    "resnet110": dict(depth=110),
}


def _norm(c: int, kind: str, groups: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(c)
    if kind == "gn":
        return nn.GroupNorm(max(d for d in range(1, groups + 1) if c % d == 0), c)
    raise ValueError(f"unknown norm {kind!r}")


class BasicBlock(nn.Module):
    """conv3x3 - norm - ReLU - conv3x3 - norm, + shortcut, ReLU."""

    def __init__(self, cin: int, cout: int, stride: int, norm: str, groups: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _norm(cout, norm, groups)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _norm(cout, norm, groups)
        self.stride, self.pad = stride, cout - cin

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        """Option A: identity, or subsample + zero-pad the extra channels (no parameters)."""
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, cfg: ResNetConfig):
        super().__init__()
        self.cfg = c = cfg
        assert (c.depth - 2) % 6 == 0, "CIFAR ResNet depth must be 6n + 2"
        assert c.t_max == 1, "ResNet is feed-forward: t_max must be 1"
        n, w = (c.depth - 2) // 6, c.width
        self.stem = nn.Sequential(nn.Conv2d(c.in_chans, w, 3, 1, 1, bias=False), _norm(w, c.norm, c.gn_groups),
                                  nn.ReLU(inplace=True))
        blocks, cin = [], w
        for i, cout in enumerate((w, 2 * w, 4 * w)):
            for j in range(n):
                blocks.append(BasicBlock(cin, cout, 2 if (j == 0 and i > 0) else 1, c.norm, c.gn_groups))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(cin, c.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def no_weight_decay(self) -> set[str]:
        return set()

    # ------------------------------------------------------- step interface (one step)
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)

    def forward_step(self, state: torch.Tensor, t: int = 0, x0: torch.Tensor | None = None,
                     out: torch.Tensor | None = None) -> torch.Tensor:
        assert isinstance(t, int) and t == 0, "ResNet has a single step"
        res = self.blocks(state)
        return res if out is None else out.copy_(res)

    def readout(self, state: torch.Tensor, t: int = 0) -> torch.Tensor:
        return self.head(state.mean((2, 3)))

    def features(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cls": state.mean((2, 3)), "pool": state.amax((2, 3))}

    def forward(self, x: torch.Tensor, t_max: int | None = None, return_feats: bool = False):
        """Logits [1, B, C] (the step axis of the recurrent models, with one step)."""
        h = self.blocks(self.stem(x))
        logits = self.readout(h)[None]
        if return_feats:
            return logits, {k: v[None] for k, v in self.features(h).items()}
        return logits


def build_resnet(name: str = "resnet56", **overrides) -> ResNet:
    if name not in PRESETS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(PRESETS)}")
    kw = {**PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}}
    return ResNet(ResNetConfig(**kw))
