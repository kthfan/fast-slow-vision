"""Stage A loss (TODO.md section 3).

    L = sum_t w_t * [CE(p_t, y) + beta * Brier(p_t, y)]
        + gamma * L_conf
        + delta * mean_{t<T} KL(sg(p_T) || p_t)
        + eta * L_distill(teacher)

L_conf follows the Gatekeeper idea (Rabanser et al., 2025): correctly
classified samples keep cross-entropy, misclassified samples are pushed towards
the uniform distribution via KL(U || p). The exact original weighting still
needs to be checked against the paper (TODO-2); ``gk_alpha`` mixes the two terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StageALossConfig:
    beta: float = 1.0  # Brier weight
    gamma: float = 0.5  # Gatekeeper confidence loss weight
    delta: float = 0.5  # last-step self-distillation weight
    eta: float = 0.0  # teacher distillation weight (not used on CIFAR yet)
    gk_alpha: float = 0.5  # Gatekeeper: alpha * correct-CE + (1 - alpha) * incorrect-KL(U||p)
    step_weights: str = "uniform"  # "uniform" | "linear" (later steps weigh more)
    label_smoothing: float = 0.0


def step_weights(T: int, kind: str, device=None) -> torch.Tensor:
    if kind == "uniform":
        w = torch.ones(T, device=device)
    elif kind == "linear":
        w = torch.arange(1, T + 1, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown step weighting {kind!r}")
    return w / w.sum()


def stage_a_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    cfg: StageALossConfig,
    distill_loss: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """logits: [T, B, C]; target: soft targets [B, C] (one-hot or mixed).

    Returns the loss and detached scalar components (tensors, to avoid a GPU sync per step).
    """
    T, B, C = logits.shape
    logits = logits.float()
    target = target.float()
    if cfg.label_smoothing > 0:
        target = target * (1 - cfg.label_smoothing) + cfg.label_smoothing / C
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    w = step_weights(T, cfg.step_weights, logits.device)  # [T]

    ce = -(target * logp).sum(-1)  # [T, B]
    brier = ((p - target) ** 2).sum(-1)  # [T, B]
    main = (w[:, None] * (ce + cfg.beta * brier)).sum(0).mean()

    total = main
    parts = {"ce": (w[:, None] * ce).sum(0).mean().detach(), "brier": (w[:, None] * brier).sum(0).mean().detach()}

    if cfg.gamma > 0:
        correct = (p.argmax(-1) == target.argmax(-1)).float()  # [T, B]
        kl_uniform = -logp.mean(-1) - math.log(C)  # KL(U || p), [T, B]
        conf = cfg.gk_alpha * correct * ce + (1 - cfg.gk_alpha) * (1 - correct) * kl_uniform
        l_conf = (w[:, None] * conf).sum(0).mean()
        total = total + cfg.gamma * l_conf
        parts["conf"] = l_conf.detach()

    if cfg.delta > 0 and T > 1:
        p_last = p[-1].detach()
        logp_last = logp[-1].detach()
        kl = (p_last * (logp_last - logp[:-1])).sum(-1)  # [T-1, B]
        l_sd = kl.mean()
        total = total + cfg.delta * l_sd
        parts["self_distill"] = l_sd.detach()

    if cfg.eta > 0 and distill_loss is not None:
        total = total + cfg.eta * distill_loss
        parts["distill"] = distill_loss.detach()

    parts["total"] = total.detach()
    return total, parts
