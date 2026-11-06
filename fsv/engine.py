"""Shared inference helpers: collect per-step outputs over a loader, and latency timing."""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch


def auto_amp(device: torch.device) -> str:
    """bf16 on Ampere/Hopper (e.g. H100, no loss scaling needed), fp16 on older GPUs, none on CPU."""
    if device.type != "cuda":
        return "none"
    return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"


def autocast_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.float16 if amp == "fp16" else torch.bfloat16)


@torch.no_grad()
def collect_outputs(model, loader, device, amp: str = "none", feats: bool = False) -> dict[str, torch.Tensor]:
    """Return {'logits': [T, N, C] float32, 'labels': [N], optional 'cls'/'pool': [T, N, D] float16} on CPU."""
    model.eval()
    logits, labels, fcls, fpool = [], [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            if feats:
                z, f = model(x, return_feats=True)
                fcls.append(f["cls"].half().cpu())
                fpool.append(f["pool"].half().cpu())
            else:
                z = model(x)
        logits.append(z.float().cpu())
        labels.append(y)
    out = {"logits": torch.cat(logits, 1), "labels": torch.cat(labels)}
    if feats:
        out["cls"] = torch.cat(fcls, 1)
        out["pool"] = torch.cat(fpool, 1)
    return out


@torch.no_grad()
def step_latency(model, device, batch_size: int, amp: str = "none", warmup: int = 5, iters: int = 20) -> dict:
    """Wall-clock latency (ms) of embed + t steps + readout at each t, for one batch size.

    Measures the cumulative time to reach and read out step t, i.e. the
    latency of a sample that stops after step t (no halting-policy cost yet).
    """
    model.eval()
    c = model.cfg
    x = torch.randn(batch_size, c.in_chans, c.img_size, c.img_size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)

    def run(T):
        with autocast_ctx(device, amp):
            tok = model.embed(x)
            for t in range(T):
                tok = model.forward_step(tok, t)
                model.readout(tok, t)

    res = []
    for T in range(1, c.t_max + 1):
        for _ in range(warmup):
            run(T)
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            run(T)
        sync()
        res.append((time.perf_counter() - t0) / iters * 1e3)
    return {"batch_size": batch_size, "ms_cumulative": res,
            "throughput_full_img_per_s": batch_size / (res[-1] / 1e3)}
