"""Per-step inputs for halting policies (TODO-1a/1b), computed from cached outputs.

question (S0): probability statistics of the temperature-scaled p_t
    top-5 probabilities, normalised entropy, top1-top2 margin,
    top-1 changed vs t-1, KL(p_t || p_{t-1}), run length of an unchanged top-1 (/T),
    one-hot step index.
state (S3): representation convergence, for both the CLS and the pooled patch feature
    cos(h_t, h_{t-1}), ||h_t - h_{t-1}|| / ||h_t||, cos(h_t, h_0).
state (S2): S0 + the CLS feature h_t itself (D dims); the MLP's first layer is then a
    learned projection D -> 64, i.e. the S6 "learned linear reduction" of TODO-1b.
state (S5): S0 + a fixed Gaussian random projection of h_t to 64 dims (no training).

The price lambda is appended by the policy itself (it is not a property of the image).
All features are causal: step t only uses steps <= t.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FEATURE_SETS = ("S0", "S3")  # default grid; "S2" and "S5" are available on request
ALL_FEATURE_SETS = ("S0", "S2", "S3", "S5")
RP_DIM = 64


def prob_features(probs: torch.Tensor) -> torch.Tensor:
    """probs [T, N, C] -> [T, N, 10 + T]."""
    T, N, C = probs.shape
    p = probs.float().clamp_min(1e-12)
    logp = p.log()
    top5 = p.topk(min(5, C), dim=-1).values
    if C < 5:
        top5 = F.pad(top5, (0, 5 - C))
    ent = -(p * logp).sum(-1, keepdim=True) / math.log(C)
    margin = (top5[..., 0] - top5[..., 1]).unsqueeze(-1)
    pred = p.argmax(-1)
    prev_logp = torch.cat([logp[:1], logp[:-1]])
    changed = (pred != torch.cat([pred[:1], pred[:-1]])).float()
    kl = (p * (logp - prev_logp)).sum(-1)
    kl[0] = 0.0
    run = torch.zeros(T, N)
    for t in range(1, T):
        run[t] = (run[t - 1] + 1) * (1 - changed[t])
    onehot = torch.eye(T)[:, None, :].expand(T, N, T)
    return torch.cat([top5, ent, margin, changed[..., None], kl[..., None], (run / T)[..., None], onehot], -1)


def convergence_features(feats: torch.Tensor) -> torch.Tensor:
    """feats [T, N, D] -> [T, N, 3]: cos to previous step, relative change, cos to step 0."""
    f = feats.float()
    prev = torch.cat([f[:1], f[:-1]])
    cos_prev = F.cosine_similarity(f, prev, dim=-1)
    rel = (f - prev).norm(dim=-1) / f.norm(dim=-1).clamp_min(1e-6)
    cos0 = F.cosine_similarity(f, f[:1].expand_as(f), dim=-1)
    return torch.stack([cos_prev, rel, cos0], -1)


def build_features(cache: dict, temps: torch.Tensor, feature_set: str) -> torch.Tensor:
    """cache: output of evaluate.py --cache (logits [T,N,C], cls/pool [T,N,D])."""
    probs = (cache["logits"].float() / temps[:, None, None]).softmax(-1)
    parts = [prob_features(probs)]
    if feature_set == "S3":
        parts += [convergence_features(cache["cls"]), convergence_features(cache["pool"])]
    elif feature_set == "S2":
        parts.append(cache["cls"].float())
    elif feature_set == "S5":
        D = cache["cls"].shape[-1]
        proj = torch.randn(D, RP_DIM, generator=torch.Generator().manual_seed(1234)) / math.sqrt(RP_DIM)
        parts.append(cache["cls"].float() @ proj)
    elif feature_set != "S0":
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {ALL_FEATURE_SETS}")
    return torch.cat(parts, -1)


def feature_flops(feature_set: str, T: int, C: int, D: int) -> int:
    """Rough per-step FLOPs to compute the features (softmax already exists for the readout)."""
    flops = 10 * C + T  # top-5, entropy, KL, margin, bookkeeping
    if feature_set == "S3":
        flops += 2 * 3 * 4 * D  # 2 feature types x 3 similarity terms x ~4 ops per dim
    elif feature_set == "S5":
        flops += 2 * D * RP_DIM  # the projection (S2's projection is counted in the MLP)
    return flops
