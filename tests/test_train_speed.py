"""Training-speed pipeline: GPU augmentation, the (compiled) train step, parallel runs, bench_train."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode as IM

from fsv import gpu_aug as G
from fsv.cpu import available_cpus, split_cpus
from fsv.data import build_splits
from fsv.losses import StageALossConfig
from fsv.models import build_model
from fsv.train_step import TrainStep, make_optimizer

ROOT = Path(__file__).resolve().parents[1]


def _images(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n, 3, 8, 8, generator=g) * 255
    return torch.nn.functional.interpolate(base, size=32, mode="bilinear").round().to(torch.uint8)


@pytest.mark.parametrize("sign", [1, -1])
def test_randaugment_ops_match_torchvision(sign):
    x = _images()
    a = G.GPUAugment((0.5,) * 3, (0.5,) * 3)
    m = a.shear * sign
    tr = int(a.translate_frac * 32 * sign)
    ref = {
        G.SHEAR_X: lambda im: TF.affine(im, 0.0, [0, 0], 1.0, [math.degrees(math.atan(m)), 0.0], IM.NEAREST, [0.],
                                        center=[0, 0]),
        G.SHEAR_Y: lambda im: TF.affine(im, 0.0, [0, 0], 1.0, [0.0, math.degrees(math.atan(m))], IM.NEAREST, [0.],
                                        center=[0, 0]),
        G.TRANS_X: lambda im: TF.affine(im, 0.0, [tr, 0], 1.0, [0., 0.], IM.NEAREST, [0.]),
        G.TRANS_Y: lambda im: TF.affine(im, 0.0, [0, tr], 1.0, [0., 0.], IM.NEAREST, [0.]),
        G.ROTATE: lambda im: TF.rotate(im, a.rotate * sign, IM.NEAREST, fill=[0.]),
        G.BRIGHTNESS: lambda im: TF.adjust_brightness(im, 1 + a.enhance * sign),
        G.COLOR: lambda im: TF.adjust_saturation(im, 1 + a.enhance * sign),
        G.CONTRAST: lambda im: TF.adjust_contrast(im, 1 + a.enhance * sign),
        G.SHARPNESS: lambda im: TF.adjust_sharpness(im, 1 + a.enhance * sign),
        G.POSTERIZE: lambda im: TF.posterize(im, a.posterize_bits),
        G.SOLARIZE: lambda im: TF.solarize(im, a.solarize),
        G.AUTOCONTRAST: TF.autocontrast,
        G.EQUALIZE: TF.equalize,
        G.IDENTITY: lambda im: im,
    }
    assert len(ref) == G.N_OPS
    for k, fn in ref.items():
        a._randint = lambda high, shape, device, k=k: torch.full(shape, k)  # every image gets op k
        a._rand = lambda *shape, device: torch.full(shape, 0.0 if sign < 0 else 0.9)  # fixed sign
        d = (a.randaug_round(x.float()) - torch.stack([fn(im) for im in x]).float()).abs()
        assert d.max() <= 1, (k, d.max())  # rounding of blends only
        assert (d == 0).float().mean() > 0.9, k


def test_gpu_augment_statistics_and_determinism():
    x = torch.full((4000, 3, 32, 32), 200, dtype=torch.uint8)
    mk = lambda: G.GPUAugment((0.,) * 3, (1.,) * 3, randaug=False, erase=0.25, flip=False, pad=0,  # noqa: E731
                              generator=torch.Generator().manual_seed(1))
    y = mk()(x)
    assert y.shape == x.shape and y.dtype == torch.float32
    assert torch.equal(y, mk()(x))  # same seed, same batch
    # erasing: ~25% of images (no crop padding here, so zeros come from erasing only)
    erased = ((y[:, 0] == 0).float().mean((1, 2)) > 0.05).float().mean()
    assert 0.2 < erased < 0.35
    full = G.GPUAugment((0.5,) * 3, (0.25,) * 3, generator=torch.Generator().manual_seed(0))(_images(256))
    assert torch.isfinite(full).all() and full.abs().max() < 3


def test_gpu_loader_epoch_and_shards():
    splits = build_splits("cifar10", "data", 400, fake=True)
    ld = G.GPULoader(splits["train"], 64, "cpu", (0.5,) * 3, (0.25,) * 3, seed=3, randaug=False, erase=0)
    n = len(splits["train"])
    assert len(ld) == n // 64
    ys = torch.cat([y for _, y in ld])
    assert ys.numel() == len(ld) * 64
    imgs, labels = G.subset_arrays(splits["train"])
    assert imgs.shape == (n, 32, 32, 3) and labels.shape == (n,)
    r0 = G.GPULoader(splits["train"], 8, "cpu", (0.5,) * 3, (0.25,) * 3, seed=3, rank=0, world=2)
    r1 = G.GPULoader(splits["train"], 8, "cpu", (0.5,) * 3, (0.25,) * 3, seed=3, rank=1, world=2)
    p0 = torch.randperm(n, generator=r0.g_perm)[0::2]
    p1 = torch.randperm(n, generator=r1.g_perm)[1::2]
    assert not set(p0.tolist()) & set(p1.tolist())


def test_compiled_train_step_matches_eager():
    torch.manual_seed(0)
    x, target = torch.randn(8, 3, 32, 32), torch.softmax(torch.randn(8, 10), -1)
    out = {}
    for mode in (None, "default"):
        torch.manual_seed(0)
        m = build_model("vcnet_mini", num_classes=10, t_max=2).train()
        # SGD: the parameter change is the gradient itself (Adam would amplify the sign of ~0 gradients
        # of the zero-init feedback weights, which differ by rounding between eager and compiled)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        step = TrainStep(m, m.parameters(), opt, torch.amp.GradScaler("cuda", enabled=False), StageALossConfig(),
                         "cpu", "none", 1.0, "all", mode)
        losses = [float(step(x, target)["total"]) for _ in range(2)]
        out[mode] = (losses, torch.cat([p.detach().flatten() for p in m.parameters()]))
        torch._dynamo.reset()
    assert out[None][0] == pytest.approx(out["default"][0], rel=1e-4)
    assert torch.allclose(out[None][1], out["default"][1], atol=1e-5)


def test_make_optimizer_falls_back_on_cpu():
    opt = make_optimizer([torch.nn.Parameter(torch.zeros(2))], 1e-3, "cpu", "fused")
    assert isinstance(opt, torch.optim.AdamW) and not opt.defaults.get("fused")


def test_split_cpus_and_cpu_set(monkeypatch):
    cpus = available_cpus()
    parts = split_cpus(len(cpus), 2)
    if len(cpus) >= 2:
        assert not set(parts[0]) & set(parts[1])
    monkeypatch.setenv("FSV_CPU_SET", str(cpus[-1]))
    assert available_cpus() == [cpus[-1]]


def test_train_gpu_aug_and_parallel_runs(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, str(ROOT / "scripts/run_experiment.py"), "smoke", "--fake-data", "--seeds", "0",
                    "1", "--parallel", "2", "--max-cpus", "2", "--out-root", str(tmp_path), "--gpu", "",
                    "--extra", "--model vcnet_mini --t-max 2 --max-steps 2 --batch-size 16 --holdout 64 --gpu-aug "
                               "--opt-impl fused"], check=True, cwd=ROOT, env=env)
    for name in ("smoke", "smoke_s1"):
        info = json.loads((tmp_path / "reports" / name / "env.json").read_text())
        assert info["train_exit"] == 0 and info["eval_exit"] == 0 and info["parallel"] == 2
        cfg = json.loads((tmp_path / "reports" / name / "config.json").read_text())
        assert cfg["args"]["gpu_aug"]


def test_bench_train_smoke(tmp_path):
    out = tmp_path / "bench.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/bench_train.py"), "c100_vcnet_orig", "--fake-data",
                    "--extra", "--model vcnet_mini --batch-size 16", "--configs", "cpu-eager", "gpu-fused",
                    "--iters", "2", "--repeats", "1", "--warmup", "1", "--max-cpus", "2", "--procs", "2",
                    "--json", str(out)], check=True, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    res = json.loads(out.read_text())
    assert [r["config"] for r in res["runs"]] == ["cpu-eager", "gpu-fused"]
    assert all(r["procs"] == 2 and len(r["img_per_s_per_copy"]) == 2 and r["img_per_s"] > 0 for r in res["runs"])
    assert {d["loader"] for d in res["loaders"]["c100_vcnet_orig"]} == {"cpu", "gpu"}
