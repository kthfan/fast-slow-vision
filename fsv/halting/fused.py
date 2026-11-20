"""Features + halting policy of one O1 step fused into ONE GPU kernel (Triton).

The unfused step runs ~40 small kernels (softmax, log, top-5, entropy, KL, concatenation,
normalisation, two linear layers, GELU, comparison, state copies). At batch 1 each costs a
launch and a few microseconds of GPU time, which together is about as long as a recurrent
step itself. ``fused_decide`` computes, for every row (image), in one program:

    p = softmax(z / T_t).clamp_min(1e-12), logp = log p
    S0 features: top-5 of p, entropy / log C, top1 - top2, top-1 changed, KL(p || p_prev),
                 run length / T, one-hot(t)                  (as ``online.online_s0``)
    policy:      x = (f - mu) / sd [, scaled log lambda];  h = GELU(W1 x + b1);  stop = w2 h + b2 > 0

and writes stop, pred (argmax), logp and the run length. The outputs may alias the inputs
(``prev_logp`` -> ``logp`` etc.): one program reads its whole row before writing it, so the
state of a CUDA-graph runner is updated in place. Results equal the PyTorch path up to
float rounding (sums are reduced in a different order), so a decision can flip only when the
policy logit is within ~1e-6 of 0.

``TRITON_INTERPRET=1`` runs the kernel on the CPU (slow; used by the tests).
"""

from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice as _ld
    except Exception:  # pragma: no cover - very old triton
        _ld = None
    HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only installs without triton
    HAVE_TRITON = False


def triton_usable(device: torch.device) -> bool:
    """True if the fused kernel can run on tensors of this device."""
    if not HAVE_TRITON:
        return False
    if os.environ.get("TRITON_INTERPRET") == "1":
        return device.type == "cpu"
    return device.type == "cuda"


if HAVE_TRITON:
    _ACCURATE = _ld is not None and os.environ.get("TRITON_INTERPRET") != "1"  # CUDA libdevice math

    @triton.jit
    def _exp(x, ACC: tl.constexpr):
        if ACC:
            return _ld.exp(x)
        else:
            return tl.exp(x)

    @triton.jit
    def _log(x, ACC: tl.constexpr):
        if ACC:
            return _ld.log(x)
        else:
            return tl.log(x)

    @triton.jit
    def _erf(x, ACC: tl.constexpr):
        if ACC:
            return _ld.erf(x)
        else:
            return tl.math.erf(x)

    @triton.jit
    def _decide_kernel(z_ptr, z_stride, plogp_ptr, ppred_ptr, prun_ptr, lam_ptr, lam_stride,
                       mu_ptr, sd_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
                       stop_ptr, pred_ptr, pred2_ptr, logp_ptr, run_ptr, feats_ptr, t_ptr, temps_ptr,
                       temp, t, inv_T, inv_logC, log_lo, inv_log_range,
                       C: tl.constexpr, T: tl.constexpr, F: tl.constexpr, IN: tl.constexpr, H: tl.constexpr,
                       BLOCK_C: tl.constexpr, BLOCK_F: tl.constexpr,
                       HAS_PREV: tl.constexpr, USE_LAM: tl.constexpr, HAS_PRED2: tl.constexpr,
                       WRITE_FEATS: tl.constexpr, ACC: tl.constexpr, ROWS: tl.constexpr):
        row = tl.program_id(0)
        if ROWS:  # every row at its own step (slot pool): t and its temperature per row
            tt = tl.load(t_ptr + row)
            tmp = tl.load(temps_ptr + tt)
        else:
            tt = t
            tmp = temp
        oc = tl.arange(0, BLOCK_C)
        cm = oc < C
        # ---- calibrated probabilities
        z = tl.load(z_ptr + row * z_stride + oc, mask=cm, other=float("-inf")).to(tl.float32) / tmp
        zmax = tl.max(z, axis=0)
        e = tl.where(cm, _exp(z - zmax, ACC), 0.0)
        p = tl.maximum(e / tl.sum(e, axis=0), 1e-12)
        logp = _log(p, ACC)
        p = tl.where(cm, p, 0.0)
        logp = tl.where(cm, logp, 0.0)
        ent = -tl.sum(p * logp, axis=0) * inv_logC
        pred = tl.argmax(tl.where(cm, p, -1.0), axis=0)
        # ---- top-5 (values only); zero padding when C < 5, as F.pad in online_s0
        q = tl.where(cm, p, -1.0)
        k0 = tl.max(q, axis=0)
        q = tl.where(oc == tl.argmax(q, axis=0), -1.0, q)
        k1 = tl.max(q, axis=0)
        q = tl.where(oc == tl.argmax(q, axis=0), -1.0, q)
        k2 = tl.max(q, axis=0)
        q = tl.where(oc == tl.argmax(q, axis=0), -1.0, q)
        k3 = tl.max(q, axis=0)
        q = tl.where(oc == tl.argmax(q, axis=0), -1.0, q)
        k4 = tl.max(q, axis=0)
        k1 = tl.where(C > 1, k1, 0.0)
        k2 = tl.where(C > 2, k2, 0.0)
        k3 = tl.where(C > 3, k3, 0.0)
        k4 = tl.where(C > 4, k4, 0.0)
        # ---- change features
        if HAS_PREV:
            plogp = tl.load(plogp_ptr + row * C + oc, mask=cm, other=0.0)
            changed = (pred != tl.load(ppred_ptr + row)).to(tl.float32)
            kl = tl.sum(p * (logp - plogp), axis=0)
            run = (tl.load(prun_ptr + row) + 1.0) * (1.0 - changed)
            if ROWS:  # a row at step 0 has no previous step (its state buffers hold another image)
                first = tt == 0
                changed = tl.where(first, 0.0, changed)
                kl = tl.where(first, 0.0, kl)
                run = tl.where(first, 0.0, run)
        else:
            changed = 0.0
            kl = 0.0
            run = 0.0
        # ---- feature vector [top5, ent, margin, changed, kl, run/T, onehot(T)] (+ lambda slot)
        of = tl.arange(0, BLOCK_F)
        f = tl.where(of == 0, k0, 0.0)
        f = tl.where(of == 1, k1, f)
        f = tl.where(of == 2, k2, f)
        f = tl.where(of == 3, k3, f)
        f = tl.where(of == 4, k4, f)
        f = tl.where(of == 5, ent, f)
        f = tl.where(of == 6, k0 - k1, f)
        f = tl.where(of == 7, changed, f)
        f = tl.where(of == 8, kl, f)
        f = tl.where(of == 9, run * inv_T, f)
        f = tl.where(of == 10 + tt, 1.0, f)
        if WRITE_FEATS:
            tl.store(feats_ptr + row * F + of, f, mask=of < F)
        fm = of < F
        x = (f - tl.load(mu_ptr + of, mask=fm, other=0.0)) / tl.load(sd_ptr + of, mask=fm, other=1.0)
        x = tl.where(fm, x, 0.0)
        if USE_LAM:
            lam = tl.load(lam_ptr + row * lam_stride)
            x = tl.where(of == F, (_log(lam, ACC) - log_lo) * inv_log_range * 2.0 - 1.0, x)
        # ---- MLP: h = GELU(W1 x + b1), logit = w2 . h + b2
        oh = tl.arange(0, H)
        w1 = tl.load(w1_ptr + oh[:, None] * IN + of[None, :], mask=of[None, :] < IN, other=0.0)
        h = tl.sum(w1 * x[None, :], axis=1) + tl.load(b1_ptr + oh)
        h = 0.5 * h * (1.0 + _erf(h * 0.7071067811865476, ACC))
        logit = tl.sum(h * tl.load(w2_ptr + oh), axis=0) + tl.load(b2_ptr)
        # ---- outputs (each program owns its row, so outputs may alias the inputs)
        if ROWS:  # the last step always stops
            tl.store(stop_ptr + row, (logit > 0) | (tt == T - 1))
        else:
            tl.store(stop_ptr + row, logit > 0)
        tl.store(pred_ptr + row, pred)
        if HAS_PRED2:
            tl.store(pred2_ptr + row, pred)
        tl.store(logp_ptr + row * C + oc, logp, mask=cm)
        tl.store(run_ptr + row, run)


class FusedDecide:
    """Callable fused features + policy for one ``HaltMLP`` and its temperatures."""

    def __init__(self, policy, temps: torch.Tensor, T: int):
        net = policy.net
        lin1, act, lin2 = net[0], net[1], net[2]
        assert isinstance(lin1, torch.nn.Linear) and isinstance(lin2, torch.nn.Linear) and lin2.out_features == 1
        assert getattr(act, "approximate", "none") == "none", "fused kernel implements the exact (erf) GELU"
        self.policy, self.T = policy, T
        self.H = lin1.out_features
        assert self.H & (self.H - 1) == 0, "hidden size must be a power of two"
        self.F = 10 + T
        self.IN = lin1.in_features
        self.use_lambda = policy.use_lambda
        self.temps = [float(v) for v in temps.float().cpu()]  # host floats: no device read per call
        self.log_lo = float(policy.log_lo)
        self.inv_log_range = 1.0 / float(policy.log_hi - policy.log_lo)
        self.BLOCK_F = triton.next_power_of_2(self.IN) if HAVE_TRITON else 0
        self._temps_dev = None

    def rows(self, z, prev_logp, prev_pred, run, lam, t: torch.Tensor, out: dict | None = None,
             feats_out: torch.Tensor | None = None):
        """As ``__call__`` with every row at its own step ``t`` (long tensor [N]); rows at step 0
        ignore the previous-step state, and rows at step T-1 always stop (slot-pool runner)."""
        if self._temps_dev is None or self._temps_dev.device != z.device:
            self._temps_dev = torch.tensor(self.temps, dtype=torch.float32, device=z.device)
        return self(z, prev_logp, prev_pred, run, lam, 1, out, feats_out, t_rows=t)

    def __call__(self, z, prev_logp, prev_pred, run, lam, t: int, out: dict | None = None,
                 feats_out: torch.Tensor | None = None, t_rows: torch.Tensor | None = None):
        """z [N, C] logits -> (stop bool [N], pred long [N], logp float [N, C], run float [N]).

        ``out`` may give destination tensors ("stop", "pred", "logp", "run", optional "pred2");
        they may be the same tensors as prev_logp / prev_pred / run (updated in place)."""
        N, C = z.shape
        dev = z.device
        if z.stride(1) != 1:
            z = z.contiguous()
        o = out or {}
        stop = o.get("stop")
        stop = torch.empty(N, dtype=torch.bool, device=dev) if stop is None else stop
        pred = o.get("pred")
        pred = torch.empty(N, dtype=torch.long, device=dev) if pred is None else pred
        logp = o.get("logp")
        logp = torch.empty(N, C, dtype=torch.float32, device=dev) if logp is None else logp
        run_o = o.get("run")
        run_o = torch.empty(N, dtype=torch.float32, device=dev) if run_o is None else run_o
        pred2 = o.get("pred2")
        has_prev = t > 0
        dummy = logp
        w1, b1 = self.policy.net[0].weight, self.policy.net[0].bias
        w2, b2 = self.policy.net[2].weight, self.policy.net[2].bias
        use_lam = self.use_lambda and lam is not None
        _decide_kernel[(N,)](
            z, z.stride(0), prev_logp if has_prev else dummy, prev_pred if has_prev else pred,
            run if has_prev else run_o, lam if use_lam else dummy, (lam.stride(0) if lam.numel() > 1 else 0)
            if use_lam else 0,
            self.policy.mu, self.policy.sd, w1, b1, w2, b2,
            stop, pred, pred2 if pred2 is not None else pred, logp, run_o,
            feats_out if feats_out is not None else dummy,
            t_rows if t_rows is not None else pred, self._temps_dev if t_rows is not None else logp,
            self.temps[0] if t_rows is not None else self.temps[t], 0 if t_rows is not None else t, 1.0 / self.T, 1.0 / math.log(C), self.log_lo, self.inv_log_range,
            C=C, T=self.T, F=self.F, IN=self.IN, H=self.H, BLOCK_C=triton.next_power_of_2(C),
            BLOCK_F=self.BLOCK_F, HAS_PREV=has_prev, USE_LAM=use_lam, HAS_PRED2=pred2 is not None,
            WRITE_FEATS=feats_out is not None, ACC=_ACCURATE, ROWS=t_rows is not None, num_warps=4,
        )
        return stop, pred, logp, run_o
