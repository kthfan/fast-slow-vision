"""Online adaptive inference: run the recurrent model step by step and stop per sample.

``AdaptiveRunner.run`` computes the S0 policy features incrementally (identical
to ``features.prob_features`` on the full sequence) and, after every step,
removes the finished images from the batch ("compaction"), so later steps
process fewer images. It synchronises with the GPU once per step (one
``nonzero``): results are written with ``index_copy_`` and the batch is compacted
with index gathers, which do not synchronise. ``run_naive`` is the first
version (boolean-mask indexing, ~9 synchronisations per step), kept to measure
how much that costs.

With ``compile=True`` the per-step model call and the per-step
feature + policy computation are wrapped in ``torch.compile`` (dynamic shapes),
which fuses the many small kernels; the fixed-depth baseline uses the same
compiled model step, so the comparison stays fair.
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
    def __init__(self, model, policy: HaltMLP, temps: torch.Tensor, device, amp: str = "none",
                 compile: bool = False):
        self.model, self.policy = model.eval(), policy.eval()
        self.temps = temps.float().to(device)
        self.device, self.amp = device, amp
        self.T = model.cfg.t_max
        self.compiled = compile
        if compile:
            # one graph per step index t (and per batch-size regime): raise dynamo's small default cache
            # limit so it does not silently fall back to eager after 8 variants
            import torch._dynamo as dynamo

            for key in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
                        "accumulated_recompile_limit"):
                if hasattr(dynamo.config, key):
                    setattr(dynamo.config, key, max(getattr(dynamo.config, key), 256))
            self._step = torch.compile(self._step_impl, dynamic=True)
            self._decide = torch.compile(self._decide_impl, dynamic=True)
        else:
            self._step, self._decide = self._step_impl, self._decide_impl

    # ------------------------------------------------------------------ pieces
    def _step_impl(self, tok, x0, t: int):
        with autocast_ctx(self.device, self.amp):
            tok = self.model.forward_step(tok, t, x0)
            z = self.model.readout(tok, t)
        return tok, z

    def _decide_impl(self, z, prev_logp, prev_pred, run, lam_vec, t: int):
        p = (z.float() / self.temps[t]).softmax(-1).clamp_min(1e-12)
        logp = p.log()
        feats, pred, run = online_s0(p, logp, prev_logp, prev_pred, run, t, self.T)
        stop = self.policy(feats[None], lam_vec)[0] > 0  # sigmoid(logit) > 0.5
        return stop, pred, logp, run

    def _lam(self, n: int, lam):
        return torch.full((n,), float(lam), device=self.device) if self.policy.use_lambda else None

    # --------------------------------------------------------------- runners
    @torch.no_grad()
    def run(self, x: torch.Tensor, lam: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, 3, H, W] on device -> (predictions [B], stop step [B]); one host sync per step."""
        B, T = x.shape[0], self.T
        preds = torch.zeros(B, dtype=torch.long, device=x.device)
        steps = torch.full((B,), T - 1, dtype=torch.long, device=x.device)
        idx = torch.arange(B, device=x.device)
        with autocast_ctx(self.device, self.amp):
            x0 = tok = self.model.embed(x)
        lam_all = self._lam(B, lam)
        prev_logp = prev_pred = run = None
        for t in range(T):
            tok, z = self._step(tok, x0, t)
            if t == T - 1:  # everything still active stops here
                preds.index_copy_(0, idx, z.argmax(-1))
                break
            n = idx.shape[0]
            stop, pred, logp, run = self._decide(z, prev_logp, prev_pred, run,
                                                 None if lam_all is None else lam_all[:n], t)
            # record the finished images without synchronising
            preds.index_copy_(0, idx, torch.where(stop, pred, preds.index_select(0, idx)))
            steps.index_copy_(0, idx, torch.where(stop, torch.full_like(idx, t), steps.index_select(0, idx)))
            keep = (~stop).nonzero().squeeze(1)  # the one synchronisation of this step
            n_keep = keep.shape[0]
            if n_keep == 0:
                break
            if n_keep < n:
                idx, tok, x0 = idx.index_select(0, keep), tok.index_select(0, keep), x0.index_select(0, keep)
                logp, pred, run = logp.index_select(0, keep), pred.index_select(0, keep), run.index_select(0, keep)
            prev_logp, prev_pred = logp, pred
        return preds, steps

    @torch.no_grad()
    def run_naive(self, x: torch.Tensor, lam: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """First version: boolean-mask indexing (several host syncs per step). Same results as ``run``."""
        B, T = x.shape[0], self.T
        preds = torch.empty(B, dtype=torch.long, device=x.device)
        steps = torch.empty(B, dtype=torch.long, device=x.device)
        idx = torch.arange(B, device=x.device)
        with autocast_ctx(self.device, self.amp):
            x0 = tok = self.model.embed(x)
        prev_logp = prev_pred = run = None
        for t in range(T):
            tok, z = self._step_impl(tok, x0, t)
            p = (z.float() / self.temps[t]).softmax(-1).clamp_min(1e-12)
            logp = p.log()
            feats, pred, run = online_s0(p, logp, prev_logp, prev_pred, run, t, T)
            if t == T - 1:
                stop = torch.ones(len(idx), dtype=torch.bool, device=x.device)
            else:
                stop = self.policy(feats[None], self._lam(len(idx), lam))[0] > 0
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
        """Fixed-depth baseline: every image stops after step t_stop (readout at every step, as counted in FLOPs)."""
        with autocast_ctx(self.device, self.amp):
            x0 = tok = self.model.embed(x)
        for t in range(t_stop + 1):
            tok, z = self._step(tok, x0, t)
        return z.argmax(-1)


class O3Runner:
    """O3 inference: run steps 0..k on the whole batch, predict every image's stop step once
    (one host synchronisation), sort the batch deepest-first and run each later step on a
    prefix of known size, so no further synchronisation is needed."""

    def __init__(self, model, step_policy, temps: torch.Tensor, device, amp: str = "none", step_fn=None):
        self.model, self.policy = model.eval(), step_policy.eval()
        self.temps = temps.float().to(device)
        self.device, self.amp = device, amp
        self.T, self.k = model.cfg.t_max, step_policy.k
        self._step = step_fn or self._step_impl  # e.g. AdaptiveRunner(compile=True)._step

    def _step_impl(self, tok, x0, t: int):
        with autocast_ctx(self.device, self.amp):
            tok = self.model.forward_step(tok, t, x0)
            z = self.model.readout(tok, t)
        return tok, z

    @torch.no_grad()
    def run(self, x: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, k = x.shape[0], self.T, self.k
        with autocast_ctx(self.device, self.amp):
            x0 = tok = self.model.embed(x)
        feats, preds_hist = [], []
        prev_logp = prev_pred = run = None
        for t in range(k + 1):
            tok, z = self._step(tok, x0, t)
            p = (z.float() / self.temps[t]).softmax(-1).clamp_min(1e-12)
            logp = p.log()
            f, prev_pred, run = online_s0(p, logp, prev_logp, prev_pred, run, t, T)
            prev_logp = logp
            feats.append(f)
            preds_hist.append(prev_pred)
        lam_vec = torch.full((B,), float(lam), device=x.device)
        s = self.policy(torch.cat(feats, -1), lam_vec).argmax(-1)  # stop step per image
        preds = torch.stack(preds_hist)[s.clamp(max=k), torch.arange(B, device=x.device)]
        if k == T - 1:
            return preds, s
        counts = torch.bincount(s, minlength=T).tolist()  # the one synchronisation
        alive = [sum(counts[t:]) for t in range(T)] + [0]  # images with stop step >= t
        n = alive[k + 1]
        if n == 0:
            return preds, s
        sel = torch.argsort(s, descending=True)[:n]  # deepest first; rows [alive[t+1], alive[t]) stop at t
        tok, x0 = tok.index_select(0, sel), x0.index_select(0, sel)
        out = torch.empty(n, dtype=torch.long, device=x.device)
        for t in range(k + 1, T):
            m = alive[t]
            if m == 0:
                break
            tok, z = self._step(tok[:m], x0[:m], t)
            lo = alive[t + 1]
            if m > lo:
                out[lo:m] = z[lo:m].argmax(-1)
        preds.index_copy_(0, sel, out)
        return preds, s
