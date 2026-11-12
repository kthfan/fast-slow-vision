"""Learned halting policies trained offline on cached per-step outputs (TODO-1, output O1).

Policy: a 2-layer MLP (hidden 64) mapping the per-step features (and, for the
cost-aware objective, log lambda) to a halting logit; at inference the model
stops at the first step whose halting probability exceeds a threshold, and
always at T_max.

Objectives
  * ``reward``: lambda-conditioned expected reward. Because every step's output
    is cached, the expected value under the halting distribution
        pi_t = h_t * prod_{s<t} (1 - h_s)            (h_{T-1} := 1)
    is computed exactly and maximised by gradient descent:
        loss = E_n E_{t~pi} [ Brier(p_t, y) + lambda * (t + 1) / T ]
    with lambda ~ log-uniform on LAM_RANGE per sample (TODO.md 2.4 and the Stage C reward).
    Inference sweeps lambda with threshold 0.5.
  * ``reward_acc``: the same with the 0/1 error in place of the Brier score,
        loss = E_n E_{t~pi} [ 1[wrong at t] + lambda * (t + 1) / T ]
    i.e. exactly the accuracy-compute trade-off being evaluated.
  * ``oracle_full`` / ``oracle_any``: supervised BCE on stop labels 1[t >= t*]
    (TODO-1e), t* from ``oracle_steps``; inference sweeps the threshold.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .thresholds import oracle_steps

OBJECTIVES = ("reward", "reward_acc", "oracle_full", "oracle_any")
REWARD_OBJECTIVES = ("reward", "reward_acc")
# Price range: at lambda = 10 one extra step costs 1.25 Brier units, enough to stop almost everything at step 0.
LAM_RANGE = (1e-4, 10.0)


class HaltMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, use_lambda: bool = False,
                 lam_range: tuple[float, float] = LAM_RANGE):
        super().__init__()
        self.use_lambda = use_lambda
        self.log_lo, self.log_hi = math.log(lam_range[0]), math.log(lam_range[1])
        self.net = nn.Sequential(nn.Linear(in_dim + int(use_lambda), hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def set_normalization(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])
        self.mu.copy_(flat.mean(0))
        self.sd.copy_(flat.std(0).clamp_min(1e-4))

    def forward(self, x: torch.Tensor, lam: torch.Tensor | None = None) -> torch.Tensor:
        """x [T, N, F]; lam [N] (only when use_lambda). Returns halting logits [T, N]."""
        z = (x - self.mu) / self.sd
        if self.use_lambda:
            ll = (lam.log() - self.log_lo) / (self.log_hi - self.log_lo) * 2 - 1  # in [-1, 1]
            z = torch.cat([z, ll[None, :, None].expand(z.shape[0], -1, 1)], -1)
        return self.net(z).squeeze(-1)

    def flops(self) -> int:
        return sum(2 * m.in_features * m.out_features for m in self.net if isinstance(m, nn.Linear))


def halting_distribution(h: torch.Tensor) -> torch.Tensor:
    """h [T, N] halting probabilities -> pi [T, N] (stop forced at the last step)."""
    h = torch.cat([h[:-1], torch.ones_like(h[-1:])])
    survive = torch.cumprod(torch.cat([torch.ones_like(h[:1]), 1 - h[:-1]]), 0)
    return h * survive


def per_step_brier(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    onehot = F.one_hot(labels, probs.shape[-1]).float()
    return ((probs.float() - onehot[None]) ** 2).sum(-1)  # [T, N]


def stop_steps(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """First step with sigmoid(logit) > threshold; T-1 if none."""
    mask = torch.sigmoid(logits) > threshold
    mask[-1] = True
    return mask.float().argmax(0)


def train_policy(x: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor, objective: str,
                 seed: int = 0, iters: int = 1500, lr: float = 3e-3, val_frac: float = 0.2,
                 lam_range: tuple[float, float] = LAM_RANGE, n_val_last: int | None = None,
                 weight_decay: float = 1e-4, device: str = "cpu") -> tuple[HaltMLP, dict]:
    """Full-batch training; the state with the best validation loss is kept.

    By default a random ``val_frac`` of the samples is the validation set. With
    ``n_val_last`` the last ``n_val_last`` samples are (e.g. held-out images
    appended after training-set images).
    """
    if objective not in OBJECTIVES:
        raise ValueError(objective)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    T, N, _ = x.shape
    if n_val_last:
        idx_tr, idx_val = torch.arange(N - n_val_last), torch.arange(N - n_val_last, N)
        idx_tr = idx_tr[torch.randperm(len(idx_tr), generator=g)]
    else:
        perm = torch.randperm(N, generator=g)
        idx_val, idx_tr = perm[:int(N * val_frac)], perm[int(N * val_frac):]
    n_val = len(idx_val)
    use_lambda = objective in REWARD_OBJECTIVES
    model = HaltMLP(x.shape[-1], use_lambda=use_lambda, lam_range=lam_range)
    model.set_normalization(x[:, idx_tr])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if objective == "reward":
        step_loss = per_step_brier(probs, labels)
    elif objective == "reward_acc":
        step_loss = (probs.argmax(-1) != labels[None]).float()
    else:
        t_star = oracle_steps(probs, labels, objective.split("_")[1])
        target = (torch.arange(T)[:, None] >= t_star[None]).float()
    lo, hi = math.log(lam_range[0]), math.log(lam_range[1])
    steps_cost = ((torch.arange(T, dtype=torch.float32)[:, None] + 1) / T).to(device)
    # full-batch training: put everything on the device once
    x = x.to(device)
    idx_tr, idx_val = idx_tr.to(device), idx_val.to(device)
    model.to(device)
    if use_lambda:
        step_loss = step_loss.to(device)
    else:
        target = target.to(device)

    def loss_on(idx, lam=None):
        logits = model(x[:, idx], lam)
        if use_lambda:
            pi = halting_distribution(torch.sigmoid(logits))
            return (pi * (step_loss[:, idx] + lam[None] * steps_cost)).sum(0).mean()
        return F.binary_cross_entropy_with_logits(logits, target[:, idx])

    lam_val = torch.exp(torch.empty(n_val).uniform_(lo, hi, generator=g)).to(device) if use_lambda else None
    best, best_state, history = float("inf"), None, []
    for it in range(iters):
        model.train()
        lam = torch.exp(torch.empty(len(idx_tr)).uniform_(lo, hi, generator=g)).to(device) if use_lambda else None
        loss = loss_on(idx_tr, lam)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == iters - 1:
            model.eval()
            with torch.no_grad():
                v = float(loss_on(idx_val, lam_val))
            history.append((it, float(loss.detach()), v))
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model.cpu(), {"best_val_loss": best, "history": history[-5:]}


@torch.no_grad()
def policy_curve(model: HaltMLP, x: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor, cum_flops,
                 extra_flops_per_step: float, n: int = 41, lam_range=LAM_RANGE, device: str = "cpu") -> list[dict]:
    """Accuracy-compute points: sweep lambda (reward policy) or the threshold (supervised)."""
    T, N = x.shape[:2]
    model = model.to(device)
    xd = x.to(device)
    cum = torch.as_tensor(cum_flops, dtype=torch.float64)
    pred = probs.argmax(-1)
    if model.use_lambda:
        grid = np.exp(np.linspace(math.log(lam_range[0]), math.log(lam_range[1]), n))
        settings = [("lam", float(l), model(xd, torch.full((N,), float(l), device=device)).cpu(), 0.5)
                    for l in grid]
    else:
        logits = model(xd).cpu()
        settings = [("tau", float(t), logits, float(t)) for t in np.linspace(0.02, 0.98, n)]
    out = []
    for key, val, logits, thr in settings:
        stop = stop_steps(logits, thr)
        correct = pred[stop, torch.arange(N)] == labels
        g = (cum[stop] + (stop.double() + 1) * extra_flops_per_step).mean()
        out.append({key: val, "acc": float(correct.float().mean()), "gflops": float(g) / 1e9,
                    "mean_steps": float(stop.float().mean()) + 1})
    return out


# --------------------------------------------------------------------------- O3: decide the depth once
class StepMLP(nn.Module):
    """O3 (TODO-1d): after step ``k``, predict the stop step for the whole run in one go.

    Input: the per-step features of steps 0..k, concatenated (and log lambda).
    Output: logits over the stop step s in {0..T-1}; inference takes the argmax.
    Steps s <= k reuse readouts that were already computed (their cost is k+1 steps).
    Deciding once means a batch needs a single host synchronisation: the images can be
    sorted by depth and every later step runs on a known-size prefix of the batch.
    """

    def __init__(self, step_dim: int, T: int, k: int, hidden: int = 64, lam_range=LAM_RANGE):
        super().__init__()
        self.T, self.k, self.step_dim = T, k, step_dim
        self.use_lambda = True
        self.log_lo, self.log_hi = math.log(lam_range[0]), math.log(lam_range[1])
        in_dim = (k + 1) * step_dim
        self.net = nn.Sequential(nn.Linear(in_dim + 1, hidden), nn.GELU(), nn.Linear(hidden, T))
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def set_normalization(self, x: torch.Tensor) -> None:
        self.mu.copy_(x.mean(0))
        self.sd.copy_(x.std(0).clamp_min(1e-4))

    def forward(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """x [N, (k+1)*F] -> logits over the stop step [N, T]."""
        ll = (lam.log() - self.log_lo) / (self.log_hi - self.log_lo) * 2 - 1
        return self.net(torch.cat([(x - self.mu) / self.sd, ll[:, None]], -1))

    def flops(self) -> int:
        return sum(2 * m.in_features * m.out_features for m in self.net if isinstance(m, nn.Linear))


def o3_inputs(x_steps: torch.Tensor, k: int) -> torch.Tensor:
    """Per-step features [T, N, F] -> the O3 input [N, (k+1)*F] (steps 0..k, step-major)."""
    return x_steps[:k + 1].permute(1, 0, 2).reshape(x_steps.shape[1], -1)


def _split(N: int, g: torch.Generator, val_frac: float, n_val_last: int | None):
    if n_val_last:
        idx_tr = torch.arange(N - n_val_last)
        return idx_tr[torch.randperm(len(idx_tr), generator=g)], torch.arange(N - n_val_last, N)
    perm = torch.randperm(N, generator=g)
    return perm[int(N * val_frac):], perm[:int(N * val_frac)]


def train_step_predictor(x_steps: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor, k: int,
                         objective: str = "reward_acc", seed: int = 0, iters: int = 1500, lr: float = 3e-3,
                         val_frac: float = 0.2, n_val_last: int | None = None, weight_decay: float = 1e-4,
                         lam_range=LAM_RANGE, device: str = "cpu") -> tuple[StepMLP, dict]:
    """Exact expected reward under the predicted step distribution:
        loss = E_n sum_s softmax(logits)_s [ err_s + lambda * (max(s, k) + 1) / T ]."""
    if objective not in REWARD_OBJECTIVES:
        raise ValueError(f"O3 supports {REWARD_OBJECTIVES}, got {objective!r}")
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    T, N, F_ = x_steps.shape
    x = o3_inputs(x_steps, k)
    idx_tr, idx_val = _split(N, g, val_frac, n_val_last)
    model = StepMLP(F_, T, k, lam_range=lam_range)
    model.set_normalization(x[idx_tr])
    step_loss = per_step_brier(probs, labels) if objective == "reward" else \
        (probs.argmax(-1) != labels[None]).float()  # [T, N]
    cost = (torch.clamp(torch.arange(T), min=k).float() + 1) / T  # [T]
    lo, hi = math.log(lam_range[0]), math.log(lam_range[1])
    x, step_loss, cost = x.to(device), step_loss.to(device), cost.to(device)
    idx_tr, idx_val = idx_tr.to(device), idx_val.to(device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def loss_on(idx, lam):
        pi = model(x[idx], lam).softmax(-1)  # [n, T]
        return (pi * (step_loss[:, idx].T + lam[:, None] * cost[None])).sum(-1).mean()

    lam_val = torch.exp(torch.empty(len(idx_val)).uniform_(lo, hi, generator=g)).to(device)
    best, best_state = float("inf"), None
    for it in range(iters):
        model.train()
        lam = torch.exp(torch.empty(len(idx_tr)).uniform_(lo, hi, generator=g)).to(device)
        loss = loss_on(idx_tr, lam)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == iters - 1:
            model.eval()
            with torch.no_grad():
                v = float(loss_on(idx_val, lam_val))
            if v < best:
                best, best_state = v, {kk: t.clone() for kk, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model.cpu(), {"best_val_loss": best}


@torch.no_grad()
def step_predictor_curve(model: StepMLP, x_steps: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor,
                         cum_flops, feature_flops_per_step: float, n: int = 41, lam_range=LAM_RANGE,
                         device: str = "cpu") -> list[dict]:
    T, N = x_steps.shape[:2]
    k = model.k
    cum = torch.as_tensor(cum_flops, dtype=torch.float64)
    pred = probs.argmax(-1)
    model = model.to(device)
    x = o3_inputs(x_steps, k).to(device)
    extra = feature_flops_per_step * (k + 1) + model.flops()  # features up to step k, one MLP call
    out = []
    for lam in np.exp(np.linspace(math.log(lam_range[0]), math.log(lam_range[1]), n)):
        s = model(x, torch.full((N,), float(lam), device=device)).argmax(-1).cpu()
        eff = s.clamp(min=k)
        correct = pred[s, torch.arange(N)] == labels
        out.append({"lam": float(lam), "acc": float(correct.float().mean()),
                    "gflops": float((cum[eff] + extra).mean()) / 1e9, "mean_steps": float(eff.float().mean()) + 1})
    return out
