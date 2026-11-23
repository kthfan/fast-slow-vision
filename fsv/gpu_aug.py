"""CIFAR training data on the GPU: the uint8 training set lives on the device and every batch is
augmented there, per image, with the recipe of ``data.train_transform``:

    RandomCrop(32, padding=4) -> RandomHorizontalFlip -> RandAugment(num_ops=2, magnitude=9)
    -> ToTensor -> Normalize -> RandomErasing(p, value 0)

Why: VCNet trains at ~1-10k img/s on an H100 with < 1% of its compute in use. Once the model step
is compiled, the CPU DataLoader (PIL + RandAugment, 8 cores) becomes the ceiling, and three seeds
in parallel would share those cores. Here the per-batch cost is a few hundred small GPU kernels.

Faithfulness to torchvision (``RandAugment`` on PIL images, 31 magnitude bins):
* the same 14 ops, chosen uniformly and independently per image and per round, with the same
  magnitudes at bin 9 (shear 0.09, translate int(150/331 * 32 * 0.3) = 4 px, rotate 9 deg,
  enhance factors 1 +- 0.27, posterize 7 bits, solarize threshold 178.5) and a random sign for
  the signed ops; geometric ops use nearest-neighbour sampling and fill 0;
* images stay on the 0..255 integer grid after every op (floor, as torchvision's uint8 tensor path),
  so posterize / solarize / equalize see the same kind of values;
* shear is about the top-left pixel (torchvision passes center=[0, 0]), rotation about the centre.
It is a re-implementation, not bit-identical: the random streams differ, and PIL's exact rounding
and pixel-centre conventions are not reproduced. Validate accuracy once before relying on it
(experiment ``c100_vcnet_adapt_blt_fast``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

N_BINS = 31
# op ids, in torchvision's order
IDENTITY, SHEAR_X, SHEAR_Y, TRANS_X, TRANS_Y, ROTATE, BRIGHTNESS, COLOR, CONTRAST, SHARPNESS, \
    POSTERIZE, SOLARIZE, AUTOCONTRAST, EQUALIZE = range(14)
N_OPS = 14
GEOMETRIC = (SHEAR_X, SHEAR_Y, TRANS_X, TRANS_Y, ROTATE)


def _floor_clamp(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 255).floor()


def _blend(a: torch.Tensor, b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """torchvision _blend on uint8: (f * a + (1 - f) * b).clamp(0, 255), truncated. f: [B, 1, 1, 1]."""
    return _floor_clamp(f * a + (1 - f) * b)


def _gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x.unbind(1)
    return (0.2989 * r + 0.587 * g + 0.114 * b).floor().unsqueeze(1)  # [B, 1, H, W]


def _sharpness_degenerate(x: torch.Tensor) -> torch.Tensor:
    k = torch.ones(3, 3, device=x.device, dtype=x.dtype)
    k[1, 1] = 5.0
    k = (k / k.sum()).expand(x.shape[1], 1, 3, 3)
    d = F.conv2d(x, k, groups=x.shape[1]).round()  # interior only
    out = x.clone()
    out[..., 1:-1, 1:-1] = d
    return out


def _autocontrast(x: torch.Tensor) -> torch.Tensor:
    lo = x.amin((2, 3), keepdim=True)
    hi = x.amax((2, 3), keepdim=True)
    flat = hi == lo
    scale = torch.where(flat, torch.ones_like(hi), 255.0 / (hi - lo).clamp_min(1e-6))
    lo = torch.where(flat, torch.zeros_like(lo), lo)
    return _floor_clamp((x - lo) * scale)


def _equalize(x: torch.Tensor) -> torch.Tensor:
    """Per-image, per-channel histogram equalisation (torchvision ``_scale_channel``), vectorised."""
    B, C, H, W = x.shape
    v = x.long().view(B * C, H * W)
    hist = torch.zeros(B * C, 256, device=x.device, dtype=torch.float32)
    hist.scatter_add_(1, v, torch.ones_like(v, dtype=torch.float32))
    last = hist.gather(1, v.amax(1, keepdim=True))  # count of the last non-zero bin
    step = torch.div(H * W - last, 255, rounding_mode="floor")  # [BC, 1]
    lut = torch.div(hist.cumsum(1) + torch.div(step, 2, rounding_mode="floor"), step.clamp_min(1),
                    rounding_mode="floor")
    lut = torch.cat([torch.zeros_like(lut[:, :1]), lut[:, :-1]], 1).clamp(0, 255)
    out = torch.where(step > 0, lut.gather(1, v), v.float())
    return out.view(B, C, H, W)


class GPUAugment:
    """Batched, per-image augmentation of uint8 [B, 3, H, W] images on any device."""

    def __init__(self, mean, std, randaug: bool = True, num_ops: int = 2, magnitude: int = 9,
                 erase: float = 0.25, pad: int = 4, flip: bool = True, generator: torch.Generator | None = None):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.randaug, self.num_ops, self.erase, self.pad, self.flip = randaug, num_ops, erase, pad, flip
        self.g = generator
        frac = magnitude / (N_BINS - 1)
        self.shear = 0.3 * frac
        self.translate_frac = 150.0 / 331.0 * frac  # times the image size, truncated to int pixels
        self.rotate = 30.0 * frac
        self.enhance = 0.9 * frac
        self.posterize_bits = int(8 - round(magnitude / ((N_BINS - 1) / 4)))
        self.solarize = 255.0 * (1 - frac)

    # ------------------------------------------------------------------ random helpers
    def _rand(self, *shape, device):
        return torch.rand(*shape, device=device, generator=self.g)

    def _randint(self, high, shape, device):
        return torch.randint(0, high, shape, device=device, generator=self.g)

    # ------------------------------------------------------------------ pieces
    def crop_flip(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dev = x.device
        if self.pad > 0:
            xp = F.pad(x, (self.pad,) * 4)
            oy = self._randint(2 * self.pad + 1, (B,), dev)
            ox = self._randint(2 * self.pad + 1, (B,), dev)
            rows = (oy[:, None] + torch.arange(H, device=dev))[:, None, :, None]
            cols = (ox[:, None] + torch.arange(W, device=dev))[:, None, None, :]
            x = xp[torch.arange(B, device=dev)[:, None, None, None], torch.arange(C, device=dev)[None, :, None, None],
                   rows, cols]
        if self.flip:
            m = self._rand(B, device=dev) < 0.5
            x = torch.where(m[:, None, None, None], x.flip(-1), x)
        return x

    def _affine(self, x: torch.Tensor, op: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """All geometric ops of one round as ONE grid_sample (identity for the other images).
        Inverse map in pixel coordinates, src = A @ dst + b, converted to affine_grid's
        normalised coordinates (align_corners=False)."""
        B, _, H, W = x.shape
        dev = x.device
        A = torch.eye(2, device=dev).repeat(B, 1, 1)
        b = torch.zeros(B, 2, device=dev)
        s = sign.float()
        A[:, 0, 1] = torch.where(op == SHEAR_X, s * self.shear, A[:, 0, 1])
        A[:, 1, 0] = torch.where(op == SHEAR_Y, s * self.shear, A[:, 1, 0])
        b[:, 0] = torch.where(op == TRANS_X, -s * int(self.translate_frac * W), b[:, 0])
        b[:, 1] = torch.where(op == TRANS_Y, -s * int(self.translate_frac * H), b[:, 1])
        rot = op == ROTATE
        th = s * math.radians(self.rotate)
        c, si = torch.cos(th), torch.sin(th)
        R = torch.stack([torch.stack([c, -si], -1), torch.stack([si, c], -1)], -2)  # [B, 2, 2]
        ctr = torch.tensor([(W - 1) / 2, (H - 1) / 2], device=dev)
        A = torch.where(rot[:, None, None], R, A)
        b = torch.where(rot[:, None], ctr - (R @ ctr), b)
        # normalised: u = D^-1 (p - c), D = diag(W/2, H/2)
        D = torch.tensor([W / 2, H / 2], device=dev)
        tA = A * D[None, None, :] / D[None, :, None]
        tb = ((A @ ctr) + b - ctr) / D
        theta = torch.cat([tA, tb[:, :, None]], 2)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    def randaug_round(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        dev = x.device
        op = self._randint(N_OPS, (B,), dev)
        sign = torch.where(self._rand(B, device=dev) < 0.5, -1, 1)
        geo = (op >= SHEAR_X) & (op <= ROTATE)
        x = torch.where(geo[:, None, None, None], self._affine(x, op, sign), x)

        f = (1 + sign.float() * self.enhance).view(B, 1, 1, 1)
        gray = _gray(x)
        mask = 255 - (2 ** (8 - self.posterize_bits) - 1)
        candidates = {
            BRIGHTNESS: lambda: _blend(x, torch.zeros_like(x), f),
            COLOR: lambda: _blend(x, gray, f),
            CONTRAST: lambda: _blend(x, gray.mean((1, 2, 3), keepdim=True), f),
            SHARPNESS: lambda: _blend(x, _sharpness_degenerate(x), f),
            POSTERIZE: lambda: (x.long() & mask).float(),
            SOLARIZE: lambda: torch.where(x >= self.solarize, 255 - x, x),
            AUTOCONTRAST: lambda: _autocontrast(x),
            EQUALIZE: lambda: _equalize(x),
        }
        out = x
        for k, fn in candidates.items():
            sel = op == k
            out = torch.where(sel[:, None, None, None], fn(), out)
        return out

    def random_erase(self, x: torch.Tensor) -> torch.Tensor:
        """torchvision RandomErasing(p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0): up to 10
        attempts per image, the first box that fits is erased."""
        B, _, H, W = x.shape
        dev = x.device
        n_try = 10
        area = H * W * (0.02 + (0.33 - 0.02) * self._rand(B, n_try, device=dev))
        logr = math.log(0.3) + (math.log(3.3) - math.log(0.3)) * self._rand(B, n_try, device=dev)
        r = torch.exp(logr)
        h = torch.round(torch.sqrt(area * r))
        w = torch.round(torch.sqrt(area / r))
        ok = (h < H) & (w < W)
        first = torch.argmax(ok.int(), 1)  # first fitting attempt (0 if none; masked below)
        any_ok = ok.any(1) & (self._rand(B, device=dev) < self.erase)
        h = h.gather(1, first[:, None])[:, 0]
        w = w.gather(1, first[:, None])[:, 0]
        i = torch.floor(self._rand(B, device=dev) * (H - h + 1))
        j = torch.floor(self._rand(B, device=dev) * (W - w + 1))
        yy = torch.arange(H, device=dev)[None, :, None]
        xx = torch.arange(W, device=dev)[None, None, :]
        box = ((yy >= i[:, None, None]) & (yy < (i + h)[:, None, None]) &
               (xx >= j[:, None, None]) & (xx < (j + w)[:, None, None]) & any_ok[:, None, None])
        return x.masked_fill(box[:, None], 0.0)

    @torch.no_grad()
    def __call__(self, x_u8: torch.Tensor) -> torch.Tensor:
        """uint8 [B, 3, H, W] -> normalised float32 [B, 3, H, W]."""
        x = x_u8.float()
        x = self.crop_flip(x)
        if self.randaug:
            for _ in range(self.num_ops):
                x = self.randaug_round(x)
        x = (x / 255.0 - self.mean.to(x.device)) / self.std.to(x.device)
        if self.erase > 0:
            x = self.random_erase(x)
        return x


def subset_arrays(ds) -> tuple[np.ndarray, np.ndarray]:
    """uint8 [N, H, W, 3] images and int64 labels of a dataset from ``data.build_splits`` (a Subset of
    ArrayDataset, the fake dataset or a torchvision CIFAR)."""
    idx = None
    if isinstance(ds, torch.utils.data.Subset):
        idx, ds = np.asarray(ds.indices), ds.dataset
    images = ds.images if hasattr(ds, "images") else ds.data
    labels = np.asarray(ds.targets, dtype=np.int64)
    if idx is not None:
        images, labels = images[idx], labels[idx]
    return np.ascontiguousarray(images), labels


class GPULoader:
    """Drop-in for the training DataLoader: shuffled, drop_last, batches augmented on the device.

    With several ranks every rank draws the same permutation (seeded) and takes its own shard;
    the augmentation stream is seeded per rank."""

    def __init__(self, ds, batch_size: int, device, mean, std, seed: int = 0, rank: int = 0, world: int = 1,
                 **aug_kw):
        images, labels = subset_arrays(ds)
        self.x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).contiguous()  # uint8 on the device
        self.y = torch.from_numpy(labels).to(device)
        self.bs, self.rank, self.world = batch_size, rank, world
        self.n = len(labels) // world
        dev = torch.device(device)
        self.g_perm = torch.Generator(device=dev).manual_seed(seed)
        self.aug = GPUAugment(mean, std, generator=torch.Generator(device=dev).manual_seed(10_007 * seed + rank + 1),
                              **aug_kw)

    def __len__(self) -> int:
        return self.n // self.bs

    def set_epoch(self, epoch: int) -> None:  # parity with DistributedSampler; the stream is already per epoch
        pass

    def __iter__(self):
        perm = torch.randperm(len(self.y), device=self.x.device, generator=self.g_perm)
        perm = perm[self.rank::self.world][: self.n]
        for k in range(len(self)):
            idx = perm[k * self.bs:(k + 1) * self.bs]
            yield self.aug(self.x[idx]), self.y[idx]
