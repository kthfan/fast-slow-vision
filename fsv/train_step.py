"""One optimisation step, shared by scripts/train.py and scripts/bench_train.py.

    forward (all steps) + Stage A loss   <- optionally torch.compile'd as ONE function
    backward, gradient clipping, optimizer step   (eager; the fused AdamW is one kernel per group)

``compile_mode``:
    None               eager (the recipe of rounds 1-3)
    "default"          Inductor: fuses the many small VCNet kernels (GroupNorm, CBAM, attention...)
    "reduce-overhead"  + CUDA graphs (cudagraph trees) for the forward and the backward: the CPU
                       launches one graph instead of thousands of kernels. Needs fixed shapes, i.e.
                       drop_last batches of one size (both loaders do that).
Mixup / CutMix stay outside the compiled function: their lambda is a fresh Python float every
step and would force a recompile each time.
"""

from __future__ import annotations

import torch

from .engine import autocast_ctx
from .losses import StageALossConfig, stage_a_loss

COMPILE_MODES = ("none", "default", "reduce-overhead", "max-autotune-no-cudagraphs")


class TrainStep:
    def __init__(self, net, params, opt, scaler, loss_cfg: StageALossConfig, device, amp: str, clip: float,
                 loss_steps: str = "all", compile_mode: str | None = None, channels_last: bool = False):
        device = torch.device(device)
        self.params = [p for p in params if p.requires_grad]
        self.opt, self.scaler, self.clip = opt, scaler, clip
        self.cudagraphs = compile_mode == "reduce-overhead" and device.type == "cuda"

        def fwd_loss(x, target):
            if channels_last:  # NHWC activations: cuDNN's preferred layout for bf16 convolutions
                x = x.contiguous(memory_format=torch.channels_last)
            with autocast_ctx(device, amp):
                logits = net(x)
            if loss_steps == "last":
                logits = logits[-1:]
            return stage_a_loss(logits, target, loss_cfg)

        if compile_mode and compile_mode != "none":
            self.fwd_loss = torch.compile(fwd_loss, mode=None if compile_mode == "default" else compile_mode)
        else:
            self.fwd_loss = fwd_loss

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        loss, parts = self.fwd_loss(x, target)
        if self.cudagraphs:  # graph outputs are overwritten by the next replay
            parts = {k: v.clone() for k, v in parts.items()}
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.clip > 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        return parts


def make_optimizer(groups, lr: float, device, impl: str = "foreach") -> torch.optim.Optimizer:
    """AdamW; ``impl="fused"`` uses the single-kernel CUDA implementation (CPU: falls back to foreach)."""
    kw = {}
    if impl == "fused" and torch.device(device).type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), **kw)
