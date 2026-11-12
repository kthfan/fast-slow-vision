"""VCNet: a visual-cortex-inspired CNN (Hill, Zhang & Prasetio, 2025, arXiv 2508.02995).

The paper gives the macro-architecture but no code, channel widths, strides,
normalisation or wiring of the top-down loop. Everything the paper states is
implemented as stated; every gap is filled by an explicit assumption, listed in
``docs/vcnet.md`` and marked ``[assumption]`` below.

Stated in the paper (v1 / v3):
* ventral stream V1 -> V2 (interstripe, thin stripe) -> V4 -> PIT -> CIT -> AIT;
  dorsal stream V1 -> V2 (thick stripe) -> MT -> MST; streams interconnected;
  AIT "receives convergent inputs and feeds into the classification layer";
* V1: three parallel depthwise-separable convolutions (3x3, 5x5, 7x7), concatenated;
  a Lateral Interaction block (convolution followed by channel-wise
  self-attention, inside a residual connection);
* CBAM and Neuromodulation (learnable channel-wise multiplicative scale) in V1, MT, V4;
* MT and MST contain Recurrent Blocks: one shared convolution applied t = 3 times,
  z_{i+1} = f(z_i) + z_0;
* top-down predictive coding: AIT predicts the V1 activity, error
  eps = ReLU(V1_bottom-up - AIT_top-down), used as "an additional learning signal".

Steps (the adaptive-computation axis of this project): the outer predictive-coding
loop. Step t = 0 is the purely bottom-up sweep. Every step t >= 1 lets AIT_{t-1}
predict V1, forms eps_t, corrects the V1 input (z_t = V1 + W eps_t) and runs
V2 ... AIT again. V1 itself is computed once (``embed``). The original model
has ``t_max = 2`` (one feedback correction) and is read out at its last step
only; the adaptive variant unrolls more steps with a readout after every step.

Normalisation is GroupNorm (TODO.md sections 2.1 and 4: no BatchNorm with
per-step statistics). The readout is the same shared head as bViT:
``Linear(LN(avgpool(AIT_t)) + step_emb(t))``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VCNetConfig:
    arch: str = "vcnet"
    img_size: int = 32
    in_chans: int = 3
    num_classes: int = 100
    t_max: int = 2  # steps of the predictive-coding loop (1 = feedforward only)
    # channel widths [assumption: the paper gives none; these give ~1.0M params / 3.9 MB,
    # close to the 3.52 MB the paper reports for full VCNet]
    c_v1: int = 48  # 3 branches x 16
    c_v2: int = 64  # ventral V2 (interstripe + thin stripe, half each)
    c_v2d: int = 32  # dorsal V2 (thick stripe)
    c_v4: int = 80
    c_pit: int = 80
    c_cit: int = 128
    c_ait: int = 160
    c_mt: int = 64
    c_mst: int = 80
    rec_iters: int = 3  # Recurrent Block iterations in MT / MST (paper: t = 3)
    gn_groups: int = 8  # GroupNorm groups (largest divisor of the width <= this)
    cbam_reduction: int = 8
    ait_carry: bool = False  # AIT_t += g * AIT_{t-1} (zero-init gate): state carried across steps

    @property
    def dim(self) -> int:  # feature size of the readout input (halting features use it)
        return self.c_ait


def _scaled(width: float, **base) -> dict:
    out = {}
    for k, v in base.items():
        mult = 3 if k == "c_v1" else (2 if k == "c_v2" else 1)  # keep the V1 / V2 branch splits even
        out[k] = max(mult * 4, int(round(v * width / (mult * 4))) * mult * 4)
    return out


_FULL = dict(c_v1=48, c_v2=64, c_v2d=32, c_v4=80, c_pit=80, c_cit=128, c_ait=160, c_mt=64, c_mst=80)

PRESETS = {
    "vcnet": dict(_FULL),
    # "we reduced VCNet's hidden-layer widths to form the Mini variant" [assumption: x0.25]
    "vcnet_mini": _scaled(0.25, **_FULL),
    "vcnet_wide": _scaled(2.0, **_FULL),
}


# --------------------------------------------------------------------------- building blocks
def gn(c: int, groups: int) -> nn.GroupNorm:
    g = max(d for d in range(1, groups + 1) if c % d == 0)
    return nn.GroupNorm(g, c, eps=1e-5)


class ConvGN(nn.Sequential):
    """conv (no bias) -> GroupNorm -> ReLU."""

    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 8):
        super().__init__(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), gn(cout, groups), nn.ReLU(inplace=True))


class DSConv(nn.Sequential):
    """Depthwise k x k + pointwise 1 x 1 -> GroupNorm -> ReLU (V1 branches)."""

    def __init__(self, cin: int, cout: int, k: int, groups: int = 8):
        super().__init__(nn.Conv2d(cin, cin, k, 1, k // 2, groups=cin, bias=False),
                         nn.Conv2d(cin, cout, 1, bias=False), gn(cout, groups), nn.ReLU(inplace=True))


class ChannelSelfAttention(nn.Module):
    """Channel-wise self-attention (channels attend to channels, as in DANet's CAM).

    A = softmax(X X^T / sqrt(HW)) over channels, out = gamma * A X; gamma is zero-init
    [assumption: the paper only names the operation].
    """

    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        X = x.flatten(2)  # [B, C, N]
        A = torch.softmax(X @ X.transpose(1, 2) / (H * W) ** 0.5, dim=-1)  # [B, C, C]
        return self.gamma * (A @ X).view(B, C, H, W)


class LateralInteraction(nn.Module):
    """x + ChannelSelfAttention(conv(x)): 'a convolution followed by channel-wise
    self-attention within a residual connection'."""

    def __init__(self, c: int, groups: int):
        super().__init__()
        self.conv = ConvGN(c, c, 3, 1, groups)
        self.attn = ChannelSelfAttention()

    def forward(self, x):
        return x + self.attn(self.conv(x))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018): channel then spatial attention."""

    def __init__(self, c: int, reduction: int = 8, k: int = 7):
        super().__init__()
        h = max(4, c // reduction)
        self.mlp = nn.Sequential(nn.Conv2d(c, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, c, 1))
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class Neuromodulation(nn.Module):
    """Learnable channel-wise multiplicative gain (init 1)."""

    def __init__(self, c: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, c, 1, 1))

    def forward(self, x):
        return x * self.scale


class RecurrentBlock(nn.Module):
    """z_0 = x; z_{i+1} = f(z_i) + z_0 for ``iters`` iterations with one shared f."""

    def __init__(self, c: int, iters: int, groups: int):
        super().__init__()
        self.f = ConvGN(c, c, 3, 1, groups)
        self.iters = iters

    def forward(self, x):
        z = x
        for _ in range(self.iters):
            z = self.f(z) + x
        return z


class Residual(nn.Module):
    def __init__(self, body: nn.Module):
        super().__init__()
        self.body = body

    def forward(self, x):
        return x + self.body(x)


# --------------------------------------------------------------------------- the network
class VCNet(nn.Module):
    """VCNet with the predictive-coding loop exposed as recurrent steps and a readout per step."""

    def __init__(self, cfg: VCNetConfig):
        super().__init__()
        self.cfg = c = cfg
        g = cfg.gn_groups
        assert c.c_v1 % 3 == 0 and c.c_v2 % 2 == 0 and c.t_max >= 1
        b = c.c_v1 // 3

        # V1 (full resolution): multi-scale DS convs -> lateral interaction -> CBAM -> neuromodulation
        self.v1_branches = nn.ModuleList(DSConv(c.in_chans, b, k, g) for k in (3, 5, 7))
        self.v1_post = nn.Sequential(LateralInteraction(c.c_v1, g), CBAM(c.c_v1, c.cbam_reduction),
                                     Neuromodulation(c.c_v1))

        # ventral stream [assumption: stride-2 at V2, V4, CIT; 32 -> 16 -> 8 -> 4]
        half = c.c_v2 // 2
        self.v2_inter = ConvGN(c.c_v1, half, 3, 2, g)
        self.v2_thin = ConvGN(c.c_v1, half, 3, 2, g)
        self.v2_mix = ConvGN(c.c_v2, c.c_v2, 3, 1, g)
        self.v4_in = ConvGN(c.c_v2, c.c_v4, 3, 2, g)
        self.mt_to_v4 = nn.Conv2d(c.c_mt, c.c_v4, 1, bias=False)  # dorsal -> ventral interconnection
        self.v4_post = nn.Sequential(ConvGN(c.c_v4, c.c_v4, 3, 1, g), CBAM(c.c_v4, c.cbam_reduction),
                                     Neuromodulation(c.c_v4))
        self.pit = nn.Sequential(ConvGN(c.c_v4, c.c_pit, 1, 1, g) if c.c_pit != c.c_v4 else nn.Identity(),
                                 Residual(nn.Sequential(ConvGN(c.c_pit, c.c_pit, 3, 1, g),
                                                        ConvGN(c.c_pit, c.c_pit, 3, 1, g))))
        self.cit = nn.Sequential(ConvGN(c.c_pit, c.c_cit, 3, 2, g), ConvGN(c.c_cit, c.c_cit, 3, 1, g))

        # dorsal stream
        self.v2_thick = ConvGN(c.c_v1, c.c_v2d, 3, 2, g)
        self.mt = nn.Sequential(ConvGN(c.c_v2d, c.c_mt, 3, 2, g), RecurrentBlock(c.c_mt, c.rec_iters, g),
                                CBAM(c.c_mt, c.cbam_reduction), Neuromodulation(c.c_mt))
        self.mst = nn.Sequential(ConvGN(c.c_mt, c.c_mst, 3, 2, g), RecurrentBlock(c.c_mst, c.rec_iters, g))

        # AIT: convergence of CIT (ventral) and MST (dorsal)
        self.ait_in = ConvGN(c.c_cit + c.c_mst, c.c_ait, 1, 1, g)
        self.ait = Residual(ConvGN(c.c_ait, c.c_ait, 3, 1, g))

        # top-down predictive coding AIT -> V1 [assumption: 1x1 conv, bilinear upsampling to V1 size]
        self.td_pred = nn.Conv2d(c.c_ait, c.c_v1, 1)
        # the error corrects the V1 input of the next sweep; zero-init so step t>=1 starts equal to step 0
        self.err_in = nn.Conv2d(c.c_v1, c.c_v1, 1, bias=False)
        self.carry_gate = nn.Parameter(torch.zeros(1, c.c_ait, 1, 1)) if c.ait_carry else None

        # shared readout: head(LN(avgpool AIT_t) + step_emb(t))
        self.head_norm = nn.LayerNorm(c.c_ait, eps=1e-6)
        self.step_out = nn.Parameter(torch.zeros(c.t_max, c.c_ait))
        self.head = nn.Linear(c.c_ait, c.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.err_in.weight)
        nn.init.trunc_normal_(self.td_pred.weight, std=0.02)

    def no_weight_decay(self) -> set[str]:
        return {"step_out", "carry_gate"}

    # ------------------------------------------------------- step interface (same as RecurrentViT)
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """V1 bottom-up activity [B, c_v1, H, W]; computed once per image."""
        v1 = torch.cat([br(x) for br in self.v1_branches], 1)
        return self.v1_post(v1)

    def predict_v1(self, ait: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(self.td_pred(ait), size=size, mode="bilinear", align_corners=False)

    def prediction_error(self, v1: torch.Tensor, ait: torch.Tensor) -> torch.Tensor:
        """eps = ReLU(V1_bottom-up - AIT_top-down)."""
        return F.relu(v1 - self.predict_v1(ait, v1.shape[-2:]))

    def sweep(self, z: torch.Tensor) -> torch.Tensor:
        """One bottom-up pass V2 ... AIT from the (corrected) V1 activity z."""
        v2 = self.v2_mix(torch.cat([self.v2_inter(z), self.v2_thin(z)], 1))
        mt = self.mt(self.v2_thick(z))
        v4 = self.v4_post(self.v4_in(v2) + self.mt_to_v4(mt))
        cit = self.cit(self.pit(v4))
        mst = self.mst(mt)
        return self.ait(self.ait_in(torch.cat([cit, mst], 1)))

    def forward_step(self, state: torch.Tensor, t: int, x0: torch.Tensor | None = None) -> torch.Tensor:
        """Step t. ``x0`` is the V1 activity from ``embed``; ``state`` is AIT_{t-1} for t >= 1
        (at t = 0 it is ignored, the runners pass the embedding there). Returns AIT_t [B, c_ait, h, w]."""
        if t == 0:
            v1 = x0 if x0 is not None else state
            return self.sweep(v1)
        assert x0 is not None, "VCNet steps t >= 1 need x0 = model.embed(x)"
        z = x0 + self.err_in(self.prediction_error(x0, state))
        ait = self.sweep(z)
        if self.carry_gate is not None:
            ait = ait + self.carry_gate * state
        return ait

    def readout(self, state: torch.Tensor, t: int) -> torch.Tensor:
        return self.head(self.head_norm(state.mean((2, 3))) + self.step_out[t])

    def features(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-step features for halting research: 'cls' = what the head reads (avg-pooled AIT),
        'pool' = max-pooled AIT."""
        return {"cls": self.head_norm(state.mean((2, 3))), "pool": self.head_norm(state.amax((2, 3)))}

    # ----------------------------------------------------------- full pass
    def forward(self, x: torch.Tensor, t_max: int | None = None, return_feats: bool = False):
        """Run all steps. Returns logits [T, B, C] (and per-step features if asked)."""
        T = t_max or self.cfg.t_max
        assert T <= self.cfg.t_max
        x0 = state = self.embed(x)
        logits, feats = [], {"cls": [], "pool": []}
        for t in range(T):
            state = self.forward_step(state, t, x0)
            logits.append(self.readout(state, t))
            if return_feats:
                for k, v in self.features(state).items():
                    feats[k].append(v)
        out = torch.stack(logits)
        if return_feats:
            return out, {k: torch.stack(v) for k, v in feats.items()}
        return out


def build_vcnet(name: str = "vcnet", **overrides) -> VCNet:
    if name not in PRESETS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(PRESETS)}")
    kw = {**PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}}
    return VCNet(VCNetConfig(**kw))


def vcnet_config_dict(model: VCNet) -> dict:
    return asdict(model.cfg)
