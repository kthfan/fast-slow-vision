"""bViT-style recurrent Vision Transformer with per-step readout.

One Transformer block is applied T_max times with shared weights (Byra et al.,
2026, "Investigating Single-Block Recurrence in Vision Transformers").
``n_blocks > 1`` cycles several distinct blocks; ``n_blocks = t_max`` is a
standard (non-shared) ViT with an exit after every block, used as a baseline. After
every step a shared linear head reads out class logits from
``LN(CLS_t) + step_emb(t)`` (TODO.md section 2.2).

Step indexing: t = 0 is the state after the first pass through the block.

Normalisation is per-sample only (LayerNorm by default, RMSNorm optional, as
in the paper). BatchNorm with per-step statistics is deliberately not offered
(TODO.md section 2.1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BViTConfig:
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 100
    dim: int = 384
    heads: int = 6
    mlp_ratio: float = 4.0
    t_max: int = 8
    n_blocks: int = 1  # distinct blocks, cycled: step t uses block t % n_blocks (1 = bViT)
    norm: str = "layernorm"  # "layernorm" | "rmsnorm"
    qkv_bias: bool = True
    step_embed_in: bool = False  # bViT-TE: add a learned per-step vector before the block
    layerscale_init: float = 0.0  # 0 disables LayerScale (paper default)

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2


PRESETS = {
    "bvit_tiny": dict(dim=192, heads=3),
    "bvit_s": dict(dim=384, heads=6),
    "bvit_b": dict(dim=768, heads=12),
}


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=1e-6)
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=1e-6)
    raise ValueError(f"unsupported norm {kind!r} (BatchNorm is intentionally not allowed)")


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm Transformer block: x + MHSA(N(x)), then x + FFN(N(x))."""

    def __init__(self, cfg: BViTConfig):
        super().__init__()
        self.norm1 = make_norm(cfg.norm, cfg.dim)
        self.attn = Attention(cfg.dim, cfg.heads, cfg.qkv_bias)
        self.norm2 = make_norm(cfg.norm, cfg.dim)
        self.mlp = Mlp(cfg.dim, int(cfg.dim * cfg.mlp_ratio))
        if cfg.layerscale_init > 0:
            self.ls1 = nn.Parameter(cfg.layerscale_init * torch.ones(cfg.dim))
            self.ls2 = nn.Parameter(cfg.layerscale_init * torch.ones(cfg.dim))
        else:
            self.ls1 = self.ls2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.norm1(x))
        x = x + (a * self.ls1 if self.ls1 is not None else a)
        m = self.mlp(self.norm2(x))
        return x + (m * self.ls2 if self.ls2 is not None else m)


class RecurrentViT(nn.Module):
    """Single shared block unrolled for up to ``t_max`` steps, read out at every step."""

    def __init__(self, cfg: BViTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.patch_embed = nn.Conv2d(cfg.in_chans, d, cfg.patch_size, cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches + 1, d))
        assert 1 <= cfg.n_blocks <= cfg.t_max
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        # Optional bViT-TE input time embedding (zero-init: starts identical to plain bViT).
        self.step_in = nn.Parameter(torch.zeros(cfg.t_max, d)) if cfg.step_embed_in else None
        # Shared readout: head(LN(CLS_t) + step_emb(t)).
        self.head_norm = make_norm(cfg.norm, d)
        self.step_out = nn.Parameter(torch.zeros(cfg.t_max, d))
        self.head = nn.Linear(d, cfg.num_classes)
        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        w = self.patch_embed.weight
        nn.init.trunc_normal_(w.view(w.shape[0], -1), std=0.02)
        nn.init.zeros_(self.patch_embed.bias)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Checkpoints from before n_blocks existed store the shared block as "block.*".
        state_dict = {("blocks.0." + k[len("block."):] if k.startswith("block.") else k): v
                      for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def no_weight_decay(self) -> set[str]:
        return {"cls_token", "pos_embed", "step_in", "step_out"}

    # ------------------------------------------------------- step interface
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos_embed

    def forward_step(self, tokens: torch.Tensor, t: int) -> torch.Tensor:
        if self.step_in is not None:
            tokens = tokens + self.step_in[t]
        return self.blocks[t % len(self.blocks)](tokens)

    def readout(self, tokens: torch.Tensor, t: int) -> torch.Tensor:
        """Logits at step t (before any temperature scaling)."""
        return self.head(self.head_norm(tokens[:, 0]) + self.step_out[t])

    def features(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-step features for halting-policy research (TODO-1 state sources)."""
        return {
            "cls": self.head_norm(tokens[:, 0]),
            "pool": self.head_norm(tokens[:, 1:].mean(dim=1)),
        }

    # ----------------------------------------------------------- full pass
    def forward(self, x: torch.Tensor, t_max: int | None = None, return_feats: bool = False):
        """Run all steps. Returns logits [T, B, C] (and per-step features if asked)."""
        T = t_max or self.cfg.t_max
        assert T <= self.cfg.t_max
        tokens = self.embed(x)
        logits, feats = [], {"cls": [], "pool": []}
        for t in range(T):
            tokens = self.forward_step(tokens, t)
            logits.append(self.readout(tokens, t))
            if return_feats:
                for k, v in self.features(tokens).items():
                    feats[k].append(v)
        out = torch.stack(logits)
        if return_feats:
            return out, {k: torch.stack(v) for k, v in feats.items()}
        return out


def build_model(name: str = "bvit_s", **overrides) -> RecurrentViT:
    if name not in PRESETS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(PRESETS)}")
    kw = {**PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}}
    return RecurrentViT(BViTConfig(**kw))


def config_dict(model: RecurrentViT) -> dict:
    return asdict(model.cfg)
