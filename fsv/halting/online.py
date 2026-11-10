"""Online adaptive inference: run the recurrent model step by step and stop per sample.

``AdaptiveRunner`` computes the S0 policy features incrementally (identical to
``features.prob_features`` on the full sequence) and, after every step, removes
the samples whose policy said "stop" from the batch ("compaction"), so later
steps process fewer images. This is the deployable version of the offline
study and is what TODO-4 times.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..engine import autocast_ctx
from .policy import HaltMLP


def online_s0(p: torch.Tensor, logp: torch.Tensor, prev_logp, prev_pred, run, t: int, T: int):
    """One step of S0 features for the active samples. Returns (features [N, 10+T], pred, run)."""
    N, C = p.shape
    top5 = p.topk(min(5, C), dim=-1).values
    if C < 5:
        top5 = F.pad(top5, (0, 5 - C))
    ent = -(p * logp).sum(-1, keepdim=True) / math.log(C)
    margin = (top5[:, 0] - top5[:, 1]).unsqueeze(-1)
    pred = p.argmax(-1)
    if t == 0:
        changed = torch.zeros(N, device=p.device)
        kl = torch.zeros(N, device=p.device)
        run = torch.zeros(N, device=p.device)
    else:
        changed = (pred != prev_pred).float()
        kl = (p * (logp - prev_logp)).sum(-1)
        run = (run + 1) * (1 - changed)
    onehot = torch.zeros(N, T, device=p.device)
    onehot[:, t] = 1
    feats = torch.cat([top5, ent, margin, changed[:, None], kl[:, None], (run / T)[:, None], onehot], -1)
    return feats, pred, run


class AdaptiveRunner:
    def __init__(self, model, policy: HaltMLP, temps: torch.Tensor, device, amp: str = "none"):
        self.model, self.policy = model.eval(), policy.eval()
        self.temps = temps.float().to(device)
        self.device, self.amp = device, amp
        self.T = model.cfg.t_max

    @torch.no_grad()
    def run(self, x: torch.Tensor, lam: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, 3, H, W] on device -> (predictions [B], stop step [B]) on device."""
        B, T = x.shape[0], self.T
        preds = torch.empty(B, dtype=torch.long, device=x.device)
        steps = torch.empty(B, dtype=torch.long, device=x.device)
        idx = torch.arange(B, device=x.device)
        with autocast_ctx(self.device, self.amp):
            x0 = tok = self.model.embed(x)
        prev_logp = prev_pred = run = None
        for t in range(T):
            with autocast_ctx(self.device, self.amp):
                tok = self.model.forward_step(tok, t, x0)
                z = self.model.readout(tok, t)
            p = (z.float() / self.temps[t]).softmax(-1).clamp_min(1e-12)
            logp = p.log()
            feats, pred, run = online_s0(p, logp, prev_logp, prev_pred, run, t, T)
            if t == T - 1:
                stop = torch.ones(len(idx), dtype=torch.bool, device=x.device)
            else:
                lam_vec = torch.full((len(idx),), float(lam), device=x.device) if self.policy.use_lambda else None
                stop = self.policy(feats[None], lam_vec)[0] > 0  # sigmoid(logit) > 0.5
            preds[idx[stop]] = pred[stop]
            steps[idx[stop]] = t
            keep = ~stop
            if not bool(keep.any()):
                break
            idx, tok, x0 = idx[keep], tok[keep], x0[keep]
            prev_logp, prev_pred, run = logp[keep], pred[keep], run[keep]
        return preds, steps

    @torch.no_grad()
    def run_fixed(self, x: torch.Tensor, t_stop: int) -> torch.Tensor:
        """Fixed-depth baseline: every sample stops after step t_stop (readout at every step, as counted in FLOPs)."""
        with autocast_ctx(self.device, self.amp):
            x0 = tok = self.model.embed(x)
            for t in range(t_stop + 1):
                tok = self.model.forward_step(tok, t, x0)
                z = self.model.readout(tok, t)
        return z.argmax(-1)
