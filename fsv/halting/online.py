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


# --------------------------------------------------------------------------- batch 1 with CUDA graphs
class _Replay:
    """A function captured as a CUDA graph (``capture=True``) or simply called (CPU, tests)."""

    def __init__(self, fn, capture: bool):
        self.fn, self.graph = fn, None
        if capture:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warm-up outside the capture, as CUDA graphs require
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                fn()

    def __call__(self):
        if self.graph is None:
            self.fn()
        else:
            self.graph.replay()


class CudaGraphBS1:
    """Batch-1 inference with the kernel-launch overhead removed by CUDA graphs.

    * fixed depth t: the whole forward pass (embed + steps 0..t) is ONE graph;
    * O1 policy: one graph per step (model step + features + policy), then one host read of
      the stop flag per step;
    * O3 policy: one graph for steps 0..k and the depth decision, one host read of the
      chosen depth s, then one graph that runs steps k+1..s.
    All inputs and outputs live in static buffers. On a CPU device nothing is captured and
    the same functions are called directly, which is how the logic is tested.
    """

    def __init__(self, model, temps: torch.Tensor, device, amp: str = "none", policy=None, o3_policy=None):
        cfg = model.cfg
        self.model, self.device, self.amp, self.T = model.eval(), device, amp, cfg.t_max
        self.temps = temps.float().to(device)
        self.policy = policy.eval() if policy is not None else None
        self.o3 = o3_policy.eval() if o3_policy is not None else None
        self.captured = device.type == "cuda"
        C = cfg.num_classes
        with torch.no_grad():
            self.x = torch.zeros(1, cfg.in_chans, cfg.img_size, cfg.img_size, device=device)
            with autocast_ctx(device, amp, cache_enabled=False):
                e = model.embed(self.x)
                probe = model.forward_step(e, 0, e)
            # VCNet: the step state (AIT map) has a different shape from the embedding (V1 map)
            self.x0, self.tok = torch.zeros_like(e), torch.zeros_like(probe)
            self.pred = torch.zeros(1, dtype=torch.long, device=device)
            self.s = torch.zeros(1, dtype=torch.long, device=device)
            self.stop = torch.zeros(1, dtype=torch.bool, device=device)
            self.prev_logp = torch.zeros(1, C, device=device)
            self.prev_pred = torch.zeros(1, dtype=torch.long, device=device)
            self.run_ = torch.zeros(1, device=device)
            self.lam = torch.ones(1, device=device)
            cap = self.captured
            self.g_fixed = [_Replay(lambda t=t: self._fixed_fn(t), cap) for t in range(self.T)]
            if self.policy is not None:
                self.g_embed = _Replay(self._embed_fn, cap)
                self.g_step = [_Replay(lambda t=t: self._step_fn(t), cap) for t in range(self.T)]
            if self.o3 is not None:
                k = self.o3.k
                self.g_o3_head = _Replay(self._o3_head_fn, cap)
                self.g_o3_tail = {j: _Replay(lambda j=j: self._o3_tail_fn(j), cap) for j in range(k + 1, self.T)}

    # ---- captured pieces (they read and write only the static buffers)
    def _model_step(self, tok, x0, t):
        with autocast_ctx(self.device, self.amp, cache_enabled=False):
            tok = self.model.forward_step(tok, t, x0)
            z = self.model.readout(tok, t)
        return tok, z

    def _probs(self, z, t):
        p = (z.float() / self.temps[t]).softmax(-1).clamp_min(1e-12)
        return p, p.log()

    def _fixed_fn(self, t_stop: int):
        with autocast_ctx(self.device, self.amp, cache_enabled=False):
            x0 = tok = self.model.embed(self.x)
        for t in range(t_stop + 1):
            tok, z = self._model_step(tok, x0, t)
        self.pred.copy_(z.argmax(-1))

    def _embed_fn(self):
        with autocast_ctx(self.device, self.amp, cache_enabled=False):
            x0 = self.model.embed(self.x)
        self.x0.copy_(x0)
        if self.tok.shape == x0.shape:  # ViT: step 0 reads the embedding from tok; VCNet reads x0
            self.tok.copy_(x0)

    def _step_fn(self, t: int):
        tok, z = self._model_step(self.tok, self.x0, t)
        self.tok.copy_(tok)
        if t == self.T - 1:
            self.pred.copy_(z.argmax(-1))
            return
        p, logp = self._probs(z, t)
        prev = (self.prev_logp, self.prev_pred, self.run_) if t > 0 else (None, None, None)
        feats, pred, run = online_s0(p, logp, *prev, t, self.T)
        stop = self.policy(feats[None], self.lam if self.policy.use_lambda else None)[0] > 0
        self.prev_logp.copy_(logp)
        self.prev_pred.copy_(pred)
        self.run_.copy_(run)
        self.stop.copy_(stop)
        self.pred.copy_(pred)

    def _o3_head_fn(self):
        k = self.o3.k
        with autocast_ctx(self.device, self.amp, cache_enabled=False):
            x0 = tok = self.model.embed(self.x)
        feats, preds = [], []
        prev_logp = prev_pred = run = None
        for t in range(k + 1):
            tok, z = self._model_step(tok, x0, t)
            p, logp = self._probs(z, t)
            f, prev_pred, run = online_s0(p, logp, prev_logp, prev_pred, run, t, self.T)
            prev_logp = logp
            feats.append(f)
            preds.append(prev_pred)
        s = self.o3(torch.cat(feats, -1), self.lam).argmax(-1)
        self.s.copy_(s)
        self.pred.copy_(torch.stack(preds)[s.clamp(max=k), torch.zeros_like(s)])
        self.x0.copy_(x0)
        self.tok.copy_(tok)

    def _o3_tail_fn(self, j: int):
        tok = self.tok
        for t in range(self.o3.k + 1, j + 1):
            tok, z = self._model_step(tok, self.x0, t)
        self.pred.copy_(z.argmax(-1))

    # ---- runners (x: [1, C, H, W] on the device)
    @torch.no_grad()
    def run_fixed(self, x: torch.Tensor, t_stop: int) -> torch.Tensor:
        self.x.copy_(x)
        self.g_fixed[t_stop]()
        return self.pred.clone()

    @torch.no_grad()
    def run(self, x: torch.Tensor, lam: float | None = None) -> tuple[torch.Tensor, int]:
        self.x.copy_(x)
        if lam is not None:
            self.lam.fill_(float(lam))
        self.g_embed()
        for t in range(self.T):
            self.g_step[t]()
            if t < self.T - 1 and bool(self.stop.item()):  # one host read per step
                return self.pred.clone(), t
        return self.pred.clone(), self.T - 1

    @torch.no_grad()
    def run_o3(self, x: torch.Tensor, lam: float) -> tuple[torch.Tensor, int]:
        self.x.copy_(x)
        self.lam.fill_(float(lam))
        self.g_o3_head()
        s = int(self.s.item())  # the one host read
        if s > self.o3.k:
            self.g_o3_tail[s]()
        return self.pred.clone(), s
