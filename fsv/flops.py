"""Analytic FLOP counts for the recurrent ViT (1 multiply-add = 2 FLOPs).

TODO.md section 7 requires the readout head and the halting policy to be
counted. The readout is applied at every executed step, so the cost of
stopping after step t (0-indexed) is

    embed + (t + 1) * (block + readout) + policy_cost(t)

The halting policy cost is added by the caller once a policy exists.
"""

from __future__ import annotations

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


def model_flops(model: RecurrentViT) -> dict:
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
