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

Extensions (off by default, ``lateral`` / ``topdown``): every area V2, MT, V4,
PIT, CIT, MST, AIT keeps a state across steps through a zero-init 1 x 1
convolution (CORnet-RT-style lateral recurrence), and every area receives the
previous step of the area above it (BLT-style top-down). The step state then
holds all seven area maps, packed into one tensor so the runners stay unchanged.

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
    # Extensions beyond the paper (docs/vcnet.md, "跨步狀態"); both zero-init, so a new model starts
    # out identical to the plain loop:
    lateral: bool = False  # every area keeps its own state: A_t = F(input_t) + R_A(A_{t-1}) (CORnet-RT style)
    topdown: bool = False  # BLT-style top-down: A_t += T_A(up(B_{t-1})), B = the next area up (Spoerer et al.)

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
AREAS = ("v2", "mt", "v4", "pit", "cit", "mst", "ait")  # packing order (AIT last)
# top-down source of each area (BLT): the next area up in its stream
TOPDOWN = {"v2": "v4", "v4": "pit", "pit": "cit", "cit": "ait", "mt": "mst", "mst": "ait"}

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

        # per-area state across steps (lateral) and top-down from the area above (BLT)
        self.area_ch = {"v2": c.c_v2, "mt": c.c_mt, "v4": c.c_v4, "pit": c.c_pit, "cit": c.c_cit,
                        "mst": c.c_mst, "ait": c.c_ait}
        h = c.img_size
        h2, h4, h8 = -(-h // 2), -(-h // 4), -(-h // 8)
        self.area_hw = {"v2": h2, "mt": h4, "v4": h4, "pit": h4, "cit": h8, "mst": h8, "ait": h8}
        self.packed = c.lateral or c.topdown
        self.lat = nn.ModuleDict({a: nn.Conv2d(ch, ch, 1, bias=False) for a, ch in self.area_ch.items()}) \
            if c.lateral else None
        self.td = nn.ModuleDict({a: nn.Conv2d(self.area_ch[src], self.area_ch[a], 1, bias=False)
                                 for a, src in TOPDOWN.items()}) if c.topdown else None

        # bilinear upsampling as two small matmuls (same values as F.interpolate, see bilinear_matrix):
        # non-persistent buffers, so checkpoints are unchanged
        pairs = {(h8, h)} | ({(self.area_hw[src], self.area_hw[a]) for a, src in TOPDOWN.items()} if c.topdown else set())
        for n_in, n_out in pairs:
            if n_in != n_out:
                self.register_buffer(f"up_{n_in}_{n_out}", bilinear_matrix(n_in, n_out), persistent=False)

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
        for mods in (self.lat, self.td):
            for m in (mods.values() if mods is not None else []):
                nn.init.zeros_(m.weight)
        nn.init.trunc_normal_(self.td_pred.weight, std=0.02)

    def no_weight_decay(self) -> set[str]:
        return {"step_out", "carry_gate"}

    def _upsample(self, x: torch.Tensor, size) -> torch.Tensor:
        """F.interpolate(x, size, mode="bilinear", align_corners=False), in training as U_h @ x @ U_w^T.

        Same values (up to float rounding). Why: Inductor turns the backward of upsample_bilinear2d
        into an index_put with atomic adds, which took ~9% of the blt training step on the H100
        (525 us per call for AIT -> V1, 4x4 -> 32x32); the backward of a matmul is a matmul."""
        (hi, wi), (ho, wo) = x.shape[-2:], tuple(size)
        if (hi, wi) == (ho, wo):
            return x
        uh, uw = getattr(self, f"up_{hi}_{ho}", None), getattr(self, f"up_{wi}_{wo}", None)
        # training only: inference keeps F.interpolate, so evaluation, latency and the measured FLOPs
        # (which do not count the interpolation) stay exactly as in rounds 1-3
        if uh is None or uw is None or not self.training:
            return F.interpolate(x, size=(ho, wo), mode="bilinear", align_corners=False)
        return uh.to(x.dtype) @ x @ uw.to(x.dtype).T

    # ------------------------------------------------------- step interface (same as RecurrentViT)
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """V1 bottom-up activity [B, c_v1, H, W]; computed once per image."""
        v1 = torch.cat([br(x) for br in self.v1_branches], 1)
        return self.v1_post(v1)

    def predict_v1(self, ait: torch.Tensor, size) -> torch.Tensor:
        return self._upsample(self.td_pred(ait), size)

    def prediction_error(self, v1: torch.Tensor, ait: torch.Tensor) -> torch.Tensor:
        """eps = ReLU(V1_bottom-up - AIT_top-down)."""
        return F.relu(v1 - self.predict_v1(ait, v1.shape[-2:]))

    # ------------------------------------------------------- step state (packed when areas keep state)
    def pack(self, areas: dict[str, torch.Tensor]) -> torch.Tensor:
        """All area maps -> one [B, N] tensor (AIT last), so the runners can index and buffer it."""
        return torch.cat([areas[a].flatten(1) for a in AREAS], 1)

    def unpack(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        sizes = [self.area_ch[a] * self.area_hw[a] ** 2 for a in AREAS]
        parts = state.split(sizes, 1)
        return {a: p.view(-1, self.area_ch[a], self.area_hw[a], self.area_hw[a]) for a, p in zip(AREAS, parts)}

    def ait_of(self, state: torch.Tensor) -> torch.Tensor:
        if not self.packed:
            return state
        hw = self.area_hw["ait"]
        n = self.area_ch["ait"] * hw * hw
        return state[:, -n:].view(-1, self.area_ch["ait"], hw, hw)

    def _recur(self, name: str, a: torch.Tensor, prev: dict | None, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Add the area's own previous state (lateral) and the previous state of the area above (top-down).
        ``mask`` [B, 1, 1, 1] (per-row steps): only rows where it is True get them (rows at step 0)."""
        if prev is None:
            return a
        out = a
        if self.lat is not None:
            out = out + self.lat[name](prev[name])
        if self.td is not None and name in self.td:
            above = self.td[name](prev[TOPDOWN[name]])
            if above.shape[-1] != a.shape[-1]:
                above = self._upsample(above, a.shape[-2:])
            out = out + above
        return out if mask is None else torch.where(mask, out, a)

    def sweep(self, z: torch.Tensor, prev: dict | None = None, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """One bottom-up pass V2 ... AIT from the (corrected) V1 activity z; ``prev`` = the
        previous step's area maps (used only with ``lateral`` / ``topdown``); ``mask`` see ``_recur``."""
        v2 = self._recur("v2", self.v2_mix(torch.cat([self.v2_inter(z), self.v2_thin(z)], 1)), prev, mask)
        mt = self._recur("mt", self.mt(self.v2_thick(z)), prev, mask)
        v4 = self._recur("v4", self.v4_post(self.v4_in(v2) + self.mt_to_v4(mt)), prev, mask)
        pit = self._recur("pit", self.pit(v4), prev, mask)
        cit = self._recur("cit", self.cit(pit), prev, mask)
        mst = self._recur("mst", self.mst(mt), prev, mask)
        ait = self._recur("ait", self.ait(self.ait_in(torch.cat([cit, mst], 1))), prev, mask)
        return {"v2": v2, "mt": mt, "v4": v4, "pit": pit, "cit": cit, "mst": mst, "ait": ait}

    def _step_areas(self, prev: dict | None, t: int, x0: torch.Tensor) -> dict[str, torch.Tensor]:
        """Step t on unpacked area maps: ``prev`` = the previous step's areas (None at t = 0)."""
        if t == 0:
            return self.sweep(x0)
        ait_prev = prev["ait"]
        z = x0 + self.err_in(self.prediction_error(x0, ait_prev))
        areas = self.sweep(z, prev if self.packed else None)
        if self.carry_gate is not None:
            areas["ait"] = areas["ait"] + self.carry_gate * ait_prev
        return areas

    def _step_areas_rows(self, prev: dict, t: torch.Tensor, x0: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every row at its own step ``t`` (long [B]): the feedback / state terms are computed for all
        rows and kept only where t > 0, so a row at step 0 gets exactly the plain bottom-up sweep
        (its stale previous state is ignored by ``torch.where``, never multiplied)."""
        m = (t > 0).view(-1, 1, 1, 1)
        ait_prev = prev["ait"]
        z = torch.where(m, x0 + self.err_in(self.prediction_error(x0, ait_prev)), x0)
        areas = self.sweep(z, prev if self.packed else None, m)
        if self.carry_gate is not None:
            areas["ait"] = torch.where(m, areas["ait"] + self.carry_gate * ait_prev, areas["ait"])
        return areas

    def forward_step(self, state: torch.Tensor, t: int | torch.Tensor, x0: torch.Tensor | None = None,
                     out: torch.Tensor | None = None) -> torch.Tensor:
        """Step t. ``x0`` is the V1 activity from ``embed``; ``state`` is the previous step's state for
        t >= 1 (at t = 0 it is ignored, the runners pass the embedding there).

        The state is AIT_t [B, c_ait, h, w], or, with ``lateral`` / ``topdown``, all area maps packed
        into [B, N] (see ``pack``).
        ``t`` is the step of the whole batch (int) or of every row (long tensor [B]; slot-pool runner).
        ``out``: tensor to copy the new state into (may be ``state`` itself)."""
        if isinstance(t, torch.Tensor):
            assert x0 is not None, "per-row VCNet steps need x0 = model.embed(x)"
            prev = self.unpack(state) if self.packed else {"ait": state}
            areas = self._step_areas_rows(prev, t, x0)
        elif t == 0:
            areas = self.sweep(x0 if x0 is not None else state)
        else:
            assert x0 is not None, "VCNet steps t >= 1 need x0 = model.embed(x)"
            prev = self.unpack(state) if self.packed else {"ait": state}
            areas = self._step_areas(prev, t, x0)
        res = self.pack(areas) if self.packed else areas["ait"]
        if out is None:
            return res
        return out.copy_(res)

    def _readout_ait(self, ait: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        # steps past the trained length (evaluate.py --steps) reuse the last trained step embedding;
        # t may be a long tensor [B] (per-row steps)
        last = self.step_out.shape[0] - 1
        e = self.step_out[t.clamp(max=last) if isinstance(t, torch.Tensor) else min(t, last)]
        return self.head(self.head_norm(ait.mean((2, 3))) + e)

    def readout(self, state: torch.Tensor, t: int | torch.Tensor) -> torch.Tensor:
        return self._readout_ait(self.ait_of(state), t)

    def extend_steps(self, t_max: int) -> None:
        """Run the predictive-coding loop for ``t_max`` steps at inference, also past the length the
        model was trained with (the weights are shared across steps; extra steps reuse the last step
        embedding). Inference only: the stored config keeps the trained length in ``trained_t_max``."""
        self.trained_t_max = self.step_out.shape[0]
        self.cfg.t_max = t_max

    def _features_ait(self, ait: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cls": self.head_norm(ait.mean((2, 3))), "pool": self.head_norm(ait.amax((2, 3)))}

    def features(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-step features for halting research: 'cls' = what the head reads (avg-pooled AIT),
        'pool' = max-pooled AIT."""
        return self._features_ait(self.ait_of(state))

    # ----------------------------------------------------------- full pass
    def forward(self, x: torch.Tensor, t_max: int | None = None, return_feats: bool = False):
        """Run all steps. Returns logits [T, B, C] (and per-step features if asked).

        Same computation as embed + forward_step + readout, but the area maps stay unpacked
        between steps (no pack / unpack per step; this is the training path)."""
        T = t_max or self.cfg.t_max
        assert T <= self.cfg.t_max
        x0 = self.embed(x)
        areas = None
        logits, feats = [], {"cls": [], "pool": []}
        for t in range(T):
            areas = self._step_areas(areas, t, x0)
            logits.append(self._readout_ait(areas["ait"], t))
            if return_feats:
                for k, v in self._features_ait(areas["ait"]).items():
                    feats[k].append(v)
        out = torch.stack(logits)
        if return_feats:
            return out, {k: torch.stack(v) for k, v in feats.items()}
        return out


def bilinear_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """[n_out, n_in] weights of 1-D linear interpolation as F.interpolate(mode="bilinear",
    align_corners=False) computes them: source index max(0, (i + 0.5) * n_in / n_out - 0.5)."""
    u_mat = torch.zeros(n_out, n_in)
    for i in range(n_out):
        u = max(0.0, (i + 0.5) * n_in / n_out - 0.5)
        i0 = min(int(u), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        w = u - i0
        u_mat[i, i0] += 1 - w
        u_mat[i, i1] += w
    return u_mat


def build_vcnet(name: str = "vcnet", **overrides) -> VCNet:
    if name not in PRESETS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(PRESETS)}")
    kw = {**PRESETS[name], **{k: v for k, v in overrides.items() if v is not None}}
    return VCNet(VCNetConfig(**kw))


def vcnet_config_dict(model: VCNet) -> dict:
    return asdict(model.cfg)
