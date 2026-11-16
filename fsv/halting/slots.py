"""Fixed-size slot pool ("continuous batching") for O1 adaptive inference.

``AdaptiveRunner`` runs one batch from step 0 to the end and drops finished images, so
the batch shrinks: late steps have little GPU work, and every step needs a host read to
know the new size. Here ``slots`` images are in flight at all times, each at its own step:

  tick:  1. refill   free slots take the next images of the queue (embedded, step 0)
         2. step     one recurrent step for every slot, each at its own step t (the block
                     is shared across steps, so one batched call covers all of them)
         3. decide   S0 features + halting policy per slot (step T-1 always stops)
         4. record   stopped slots write their prediction and stop step, and become free

All shapes are fixed, so a tick is ONE CUDA graph; the queue sits on the GPU and the
next-image pointer is a device tensor, so the CPU never needs to know which slots
stopped. It replays the tick graph and reads the number of finished images one tick
late (pinned memory + event), so the GPU always has the next tick queued.

Costs: slots that are free (queue exhausted, at the very end) are still computed;
``refill`` caps how many slots one tick can refill (embedding cost per tick).
Same decisions as ``AdaptiveRunner.run`` up to float rounding (different batch shapes).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..engine import autocast_ctx
from .online import _Replay


def s0_rows(p, logp, prev_logp, prev_pred, run, t: torch.Tensor, T: int):
    """``online_s0`` with every row at its own step ``t`` (long [N]); rows at step 0 ignore
    the previous-step state. Same values as ``online_s0`` row by row."""
    N, C = p.shape
    top5 = p.topk(min(5, C), dim=-1).values
    if C < 5:
        top5 = F.pad(top5, (0, 5 - C))
    ent = -(p * logp).sum(-1, keepdim=True) / math.log(C)
    margin = (top5[:, 0] - top5[:, 1]).unsqueeze(-1)
    pred = p.argmax(-1)
    first = t == 0
    zero = torch.zeros(N, device=p.device)
    changed = torch.where(first, zero, (pred != prev_pred).float())
    kl = torch.where(first, zero, (p * (logp - prev_logp)).sum(-1))
    run = torch.where(first, zero, (run + 1) * (1 - changed))
    onehot = F.one_hot(t, T).float()
    feats = torch.cat([top5, ent, margin, changed[:, None], kl[:, None], (run / T)[:, None], onehot], -1)
    return feats, pred, run


class SlotPoolRunner:
    def __init__(self, model, policy, temps: torch.Tensor, device, amp: str = "none", slots: int = 256,
                 refill: int | None = None, fused: bool = False, graph: bool | None = None):
        if len(model.blocks) != 1:
            raise NotImplementedError("the slot pool needs one block shared across steps (n_blocks == 1)")
        self.model, self.policy = model.eval(), policy.eval()
        self.device, self.amp = device, amp
        self.T, self.S = model.cfg.t_max, slots
        self.R = min(refill or slots, slots)
        self.temps = temps.float().to(device)
        self.graph = (device.type == "cuda") if graph is None else graph
        self.fd = None
        if fused:
            from .fused import FusedDecide

            self.fd = FusedDecide(self.policy, temps, self.T)
        self.Q = 0  # queue capacity; buffers (and the graph) are built on the first run
        self.last_ticks = 0

    # ------------------------------------------------------------------ buffers
    def _build(self, Q: int, shape) -> None:
        dev, S, cfg = self.device, self.S, self.model.cfg
        with torch.no_grad():
            with autocast_ctx(dev, self.amp, cache_enabled=False):
                probe = self.model.forward_step(self.model.embed(torch.zeros(1, *shape, device=dev)), 0,
                                                self.model.embed(torch.zeros(1, *shape, device=dev)))
            self.Q = Q
            self.queue = torch.zeros(Q, *shape, device=dev)
            self.n_total = torch.zeros((), dtype=torch.long, device=dev)
            self.next_ptr = torch.zeros((), dtype=torch.long, device=dev)
            self.done = torch.zeros((), dtype=torch.long, device=dev)
            self.out_pred = torch.zeros(Q + 1, dtype=torch.long, device=dev)  # row Q: sink for idle slots
            self.out_step = torch.zeros(Q + 1, dtype=torch.long, device=dev)
            self.tok = torch.zeros(S, *probe.shape[1:], dtype=probe.dtype, device=dev)
            self.x0 = torch.zeros_like(self.tok)
            self.t = torch.zeros(S, dtype=torch.long, device=dev)
            self.img = torch.full((S,), Q, dtype=torch.long, device=dev)
            self.active = torch.zeros(S, dtype=torch.bool, device=dev)
            self.stop = torch.zeros(S, dtype=torch.bool, device=dev)
            self.prev_logp = torch.zeros(S, cfg.num_classes, device=dev)
            self.prev_pred = torch.zeros(S, dtype=torch.long, device=dev)
            self.run_ = torch.zeros(S, device=dev)
            self.lam = torch.ones(S, device=dev)
            self.n_total.fill_(0)  # the warm-up / capture ticks below then do nothing useful
            self.g_tick = _Replay(self._tick, self.graph)
            self.done_host = torch.zeros(2, dtype=torch.long, pin_memory=self.graph)

    def _reset(self, N: int) -> None:
        self.n_total.fill_(N)
        self.next_ptr.zero_()
        self.done.zero_()
        self.active.zero_()
        self.t.zero_()
        self.img.fill_(self.Q)

    # ------------------------------------------------------------------ one tick (captured)
    def _tick(self) -> None:
        T, Q, R = self.T, self.Q, self.R
        # 1. refill: the first R free slots take the next images of the queue
        order = torch.argsort(self.active.to(torch.uint8), stable=True)[:R]  # free slots first
        free = ~self.active.index_select(0, order)
        qi = self.next_ptr + torch.cumsum(free.long(), 0) - 1
        take = free & (qi < self.n_total)
        qi = torch.where(take, qi, torch.zeros_like(qi))
        with autocast_ctx(self.device, self.amp, cache_enabled=False):
            emb = self.model.embed(self.queue.index_select(0, qi)).to(self.tok.dtype)
        tk = take[:, None, None]
        self.tok.index_copy_(0, order, torch.where(tk, emb, self.tok.index_select(0, order)))
        self.x0.index_copy_(0, order, torch.where(tk, emb, self.x0.index_select(0, order)))
        self.t.index_copy_(0, order, torch.where(take, torch.zeros_like(qi), self.t.index_select(0, order)))
        self.img.index_copy_(0, order, torch.where(take, qi, self.img.index_select(0, order)))
        self.active.index_copy_(0, order, self.active.index_select(0, order) | take)
        self.next_ptr.add_(take.sum())
        # 2. one recurrent step for every slot, each at its own step
        with autocast_ctx(self.device, self.amp, cache_enabled=False):
            tok = self.model.forward_step(self.tok, self.t, self.x0)
            z = self.model.readout(tok, self.t)
        self.tok.copy_(tok)
        # 3. decide (state updated in place; step T-1 always stops)
        if self.fd is not None:
            self.fd.rows(z, self.prev_logp, self.prev_pred, self.run_, self.lam, self.t,
                         out={"stop": self.stop, "pred": self.prev_pred, "logp": self.prev_logp, "run": self.run_})
        else:
            p = (z.float() / self.temps[self.t][:, None]).softmax(-1).clamp_min(1e-12)
            logp = p.log()
            feats, pred, run = s0_rows(p, logp, self.prev_logp, self.prev_pred, self.run_, self.t, T)
            lam = self.lam if self.policy.use_lambda else None
            self.stop.copy_((self.policy(feats[None], lam)[0] > 0) | (self.t == T - 1))
            self.prev_logp.copy_(logp)
            self.prev_pred.copy_(pred)
            self.run_.copy_(run)
        # 4. record the stopped slots (idle slots write to the sink row Q) and free them
        stop = self.stop & self.active
        sink = torch.where(stop, self.img, torch.full_like(self.img, Q))
        self.out_pred.index_copy_(0, sink, self.prev_pred)
        self.out_step.index_copy_(0, sink, self.t)
        self.done.add_(stop.sum())
        self.active.copy_(self.active & ~stop)
        self.t.copy_((self.t + 1).clamp(max=T - 1))

    # ------------------------------------------------------------------ runner
    @torch.no_grad()
    def run(self, x: torch.Tensor, lam: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x [N, C, H, W] on the device -> (predictions [N], stop steps [N])."""
        N = x.shape[0]
        if N > self.Q:
            self._build(max(N, 2 * self.Q), tuple(x.shape[1:]))
        self.queue[:N].copy_(x)
        if lam is not None:
            self.lam.fill_(float(lam))
        self._reset(N)
        ticks = 0
        if not self.graph:  # CPU / debugging: read the counter every tick
            while int(self.done) < N:
                self.g_tick()
                ticks += 1
        else:  # read the counter one tick late, so the next tick is always queued on the GPU
            evs = [torch.cuda.Event(), torch.cuda.Event()]
            prev = None
            while True:
                self.g_tick()
                ticks += 1
                i = ticks % 2
                self.done_host[i].copy_(self.done, non_blocking=True)
                evs[i].record()
                if prev is not None:
                    evs[prev].synchronize()
                    if int(self.done_host[prev]) >= N:
                        break
                prev = i
        self.last_ticks = ticks
        return self.out_pred[:N].clone(), self.out_step[:N].clone()

    def occupancy(self, steps: torch.Tensor) -> float:
        """Fraction of slot-steps of the last run that did useful work."""
        return float((steps + 1).sum()) / max(1, self.last_ticks * self.S)
