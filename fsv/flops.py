"""FLOP counts per step (1 multiply-add = 2 FLOPs).

Recurrent ViT: analytic. VCNet: measured once with ``torch.utils.flop_counter``
on a single image, which counts the same operations as the analytic ViT count
(convolutions and matrix products; normalisation, activations and pooling are
not counted).

TODO.md section 7 requires the readout head and the halting policy to be
counted. The readout is applied at every executed step, so the cost of
stopping after step t (0-indexed) is

    embed + (t + 1) * (block + readout) + policy_cost(t)

The halting policy cost is added by the caller once a policy exists.
"""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode

from .models.bvit import RecurrentViT


def block_flops(dim: int, n_tokens: int, mlp_ratio: float) -> int:
    hidden = int(dim * mlp_ratio)
    macs = (
        n_tokens * dim * 3 * dim  # qkv
        + 2 * n_tokens * n_tokens * dim  # q@k^T and attn@v
        + n_tokens * dim * dim  # output projection
        + 2 * n_tokens * dim * hidden  # FFN
    )
    return 2 * macs


def model_flops(model) -> dict:
    if not isinstance(model, RecurrentViT):
        return measured_flops(model)
    c = model.cfg
    n = c.num_patches + 1
    embed = 2 * c.num_patches * (c.in_chans * c.patch_size**2) * c.dim
    block = block_flops(c.dim, n, c.mlp_ratio)
    readout = 2 * c.dim * c.num_classes
    cumulative = [embed + (t + 1) * (block + readout) for t in range(c.t_max)]
    return {
        "embed": embed,
        "block": block,
        "readout": readout,
        "cumulative": cumulative,  # FLOPs if we stop after step t
    }


@torch.no_grad()
def measured_flops(model) -> dict:
    """Per-step FLOPs of any model with the step interface, counted on one image.

    Step costs can differ per step (VCNet: step 0 has no top-down prediction), so
    ``step`` is a list; ``block`` is the cost of a later step, for reports."""
    c = model.cfg
    p = next(model.parameters())
    x = torch.zeros(1, c.in_chans, c.img_size, c.img_size, device=p.device, dtype=p.dtype)

    def count(fn):
        with FlopCounterMode(display=False) as fc:
            out = fn()
        return fc.get_total_flops(), out

    was_training = model.training
    model.eval()
    embed, x0 = count(lambda: model.embed(x))
    state, steps, readout = x0, [], 0
    for t in range(c.t_max):
        f, state = count(lambda: model.forward_step(state, t, x0))
        readout, _ = count(lambda: model.readout(state, t))
        steps.append(f)
    model.train(was_training)
    cumulative, acc = [], embed
    for f in steps:
        acc += f + readout
        cumulative.append(acc)
    return {"embed": embed, "block": steps[-1], "step": steps, "readout": readout, "cumulative": cumulative}
