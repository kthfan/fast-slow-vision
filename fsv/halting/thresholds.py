"""Training-free halting baselines evaluated on cached per-step probabilities.

Covers Stage C item 1 (max-prob / entropy thresholds), the fixed-depth exits
and the oracle stopping step t* (TODO-1e).
"""

from __future__ import annotations

import numpy as np
import torch


def _point(stop: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor, cum_flops: torch.Tensor) -> dict:
    N = labels.shape[0]
    final_pred = probs.argmax(-1)[stop, torch.arange(N)]
    return {
        "acc": float((final_pred == labels).float().mean()),
        "gflops": float(cum_flops[stop].double().mean()) / 1e9,
        "mean_steps": float(stop.float().mean()) + 1.0,
    }


def first_true(mask: torch.Tensor) -> torch.Tensor:
    """mask [T, N] -> index of first True per column; T-1 if none (forced stop at T_max)."""
    T = mask.shape[0]
    mask = mask.clone()
    mask[-1] = True
    return mask.float().argmax(0).clamp(max=T - 1)


def threshold_curve(probs: torch.Tensor, labels: torch.Tensor, cum_flops, kind: str, n: int = 50) -> list[dict]:
    """probs [T, N, C]. kind: 'maxprob' (stop if p_max >= tau) or 'entropy' (stop if H <= tau)."""
    cum_flops = torch.as_tensor(cum_flops, dtype=torch.float64)
    if kind == "maxprob":
        score = probs.max(-1).values
        taus = np.unique(np.concatenate([np.linspace(0.0, 1.0, n), [1.01]]))
        rule = lambda tau: score >= tau  # noqa: E731
    elif kind == "entropy":
        score = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        taus = np.unique(np.concatenate([np.linspace(0.0, float(score.max()) + 1e-6, n), [-1.0]]))
        rule = lambda tau: score <= tau  # noqa: E731
    else:
        raise ValueError(kind)
    return [{"tau": float(tau), **_point(first_true(rule(tau)), probs, labels, cum_flops)} for tau in taus]


def fixed_depth_curve(probs: torch.Tensor, labels: torch.Tensor, cum_flops) -> list[dict]:
    cum_flops = torch.as_tensor(cum_flops, dtype=torch.float64)
    T, N = probs.shape[:2]
    return [{"step": t, **_point(torch.full((N,), t, dtype=torch.long), probs, labels, cum_flops)} for t in range(T)]


def oracle_steps(probs: torch.Tensor, labels: torch.Tensor, never_correct: str = "full") -> torch.Tensor:
    """t* = earliest step after which every step is correct.

    never_correct: what to do when the last step is wrong:
      'full' -> run to T_max (t* = T-1); 'any' -> stop immediately (t* = 0).
    """
    correct = probs.argmax(-1) == labels[None]  # [T, N]
    T = correct.shape[0]
    # suffix_all[t] = all(correct[t:])
    suffix_all = correct.flip(0).float().cumprod(0).flip(0).bool()
    has = suffix_all.any(0)
    t_star = suffix_all.float().argmax(0)
    fallback = T - 1 if never_correct == "full" else 0
    return torch.where(has, t_star, torch.full_like(t_star, fallback))


def oracle_point(probs: torch.Tensor, labels: torch.Tensor, cum_flops, never_correct: str = "full") -> dict:
    cum_flops = torch.as_tensor(cum_flops, dtype=torch.float64)
    return _point(oracle_steps(probs, labels, never_correct), probs, labels, cum_flops)


def curve_auc(points: list[dict], lo: float, hi: float, anchor: dict | None = None) -> float:
    """Mean accuracy over [lo, hi] GFLOPs under the Pareto envelope of a method's points.

    ``anchor`` (normally "stop everything at step 0") is added to the points: it is
    always available, and any point between it and the method's cheapest point is
    reachable by randomly mixing the two settings, so the envelope is linear there.
    Above the most expensive point accuracy is held flat (spending less is allowed).
    Without an anchor the cheapest point is extended flat to the left, which
    over-credits methods that cannot reach low compute.
    """
    xs, ys = envelope(points + ([anchor] if anchor else []))
    grid = np.linspace(lo, hi, 512)
    vals = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    return float(np.trapezoid(vals, grid) / (hi - lo)) if hi > lo else float(vals[0])


def envelope(points: list[dict]) -> tuple[list[float], list[float]]:
    """Pareto envelope (best accuracy at <= compute) as sorted (gflops, acc) lists."""
    xs, ys, best = [], [], -1.0
    for x, y in sorted((p["gflops"], p["acc"]) for p in points):
        best = max(best, y)
        xs.append(x)
        ys.append(best)
    return xs, ys


def acc_at_compute(points: list[dict], gflops: float, anchor: dict | None = None) -> float:
    """Best accuracy reachable at a given mean compute (see ``curve_auc`` for ``anchor``)."""
    xs, ys = envelope(points + ([anchor] if anchor else []))
    return float(np.interp(gflops, xs, ys, left=ys[0], right=ys[-1]))
