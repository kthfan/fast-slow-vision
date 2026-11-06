"""Accuracy and calibration metrics (TODO.md section 7). All inputs are torch tensors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    total = torch.zeros((), device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(total)


def brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
    onehot = F.one_hot(labels, probs.shape[-1]).float()
    return float(((probs - onehot) ** 2).sum(-1).mean())


def nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return float(-torch.log(probs.gather(-1, labels[:, None]).clamp_min(1e-12)).mean())


def aurc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the risk-coverage curve, confidence = max probability (lower is better)."""
    conf, pred = probs.max(-1)
    order = conf.argsort(descending=True)
    errors = (pred[order] != labels[order]).float()
    risk = errors.cumsum(0) / torch.arange(1, len(errors) + 1, device=errors.device)
    return float(risk.mean())


def classification_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = probs.float()
    return {
        "acc": float((probs.argmax(-1) == labels).float().mean()),
        "ece": ece(probs, labels),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
        "aurc": aurc(probs, labels),
    }


def per_step_metrics(logits: torch.Tensor, labels: torch.Tensor, temps: torch.Tensor | None = None) -> list[dict]:
    """logits [T, N, C] -> one metrics dict per step (optionally temperature-scaled)."""
    out = []
    for t in range(logits.shape[0]):
        z = logits[t].float()
        if temps is not None:
            z = z / temps[t]
        out.append({"step": t, **classification_metrics(z.softmax(-1), labels)})
    return out
