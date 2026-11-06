"""Stage B: per-step temperature scaling (Guo et al., 2017), fitted on a held-out split."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """Return the scalar T > 0 minimising NLL(softmax(logits / T), labels)."""
    logits = logits.detach().float()
    log_t = torch.zeros(1, requires_grad=True, device=logits.device)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def fit_step_temperatures(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits [T, N, C] -> temperatures [T]."""
    return torch.tensor([fit_temperature(logits[t], labels) for t in range(logits.shape[0])])
