"""Phase 1 baseline: CIFAR ResNet behind the step interface, and VCNet matched to ResNet-56's size."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsv.experiments import EXPERIMENTS  # noqa: E402
from fsv.flops import model_flops  # noqa: E402
from fsv.models import build_model, config_dict, model_from_config  # noqa: E402
from fsv.train_step import make_optimizer  # noqa: E402


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_resnet_sizes_match_the_paper():
    # He et al. 2016, table 6 (10 classes): ResNet-20 0.27M, ResNet-56 0.85M, ResNet-110 1.7M
    for name, ref in (("resnet20", 0.27e6), ("resnet56", 0.85e6), ("resnet110", 1.7e6)):
        m = build_model(name, num_classes=10)
        assert abs(n_params(m) - ref) / ref < 0.03, (name, n_params(m))
    m = build_model("resnet56", num_classes=100)
    assert 0.855e6 < n_params(m) < 0.865e6
    fl = model_flops(m)
    assert len(fl["cumulative"]) == 1 and 0.24e9 < fl["cumulative"][0] < 0.26e9


@pytest.mark.parametrize("norm", ["bn", "gn"])
def test_resnet_step_interface_and_roundtrip(norm):
    torch.manual_seed(0)
    m = build_model("resnet20", num_classes=7, norm=norm).eval()
    x = torch.randn(3, 3, 32, 32)
    z = m(x)
    assert z.shape == (1, 3, 7)
    h = m.forward_step(m.embed(x), 0)
    assert torch.allclose(m.readout(h, 0), z[0], atol=1e-5)
    z2, feats = m(x, return_feats=True)
    assert torch.equal(z2, z) and feats["cls"].shape == (1, 3, 64)
    m2 = model_from_config(config_dict(m))
    m2.load_state_dict(m.state_dict())
    assert torch.allclose(m2.eval()(x), z)


def test_option_a_shortcut_has_no_parameters():
    m = build_model("resnet20")
    blk = m.blocks[3]  # first block of stage 2: 16 -> 32 channels, stride 2
    x = torch.randn(2, 16, 32, 32)
    s = blk.shortcut(x)
    assert s.shape == (2, 32, 16, 16)
    assert torch.equal(s[:, 8:24], x[:, :, ::2, ::2]) and s[:, :8].abs().sum() == 0 and s[:, 24:].abs().sum() == 0
    assert not any("shortcut" in n for n, _ in m.named_parameters())


def test_vcnet_presets_matched_to_resnet56():
    r56 = build_model("resnet56")
    p_ref, f_ref = n_params(r56), model_flops(r56)["cumulative"][0]
    p = build_model("vcnet_r56p", t_max=4, lateral=True, topdown=True)
    assert abs(n_params(p) - p_ref) / p_ref < 0.01
    cum = model_flops(p)["cumulative"]
    assert cum[1] < f_ref < cum[2]  # ResNet-56's compute lies between 2 and 3 steps
    f4 = build_model("vcnet_r56f4", t_max=4, lateral=True, topdown=True)
    assert abs(model_flops(f4)["cumulative"][3] - f_ref) / f_ref < 0.02


def test_sgd_optimizer():
    w = torch.nn.Parameter(torch.ones(2))
    opt = make_optimizer([{"params": [w], "weight_decay": 0.0}], 0.1, "cpu", "fused", "sgd", 0.9)
    assert isinstance(opt, torch.optim.SGD) and opt.defaults["nesterov"]
    with pytest.raises(ValueError):
        make_optimizer([w], 0.1, "cpu", kind="lion")


def test_phase1_recipes():
    rn = EXPERIMENTS["c100_resnet56_fast"]["train"]
    vc = EXPERIMENTS["c100_vcnet_orig_fast"]["train"]
    # same recipe as VCNet except the model and the step count
    diff = {(a, b) for a, b in zip(rn, vc) if a != b}
    assert len(rn) == len(vc) and diff == {("resnet56", "vcnet"), ("1", "2")}
    sgd = EXPERIMENTS["c100_resnet56_sgd"]["train"]
    for flag, value in (("--lr", "0.2"), ("--wd", "5e-4"), ("--opt", "sgd"), ("--mixup", "0")):
        assert sgd[sgd.index(flag) + 1] == value and sgd.count(flag) == 1


def test_train_then_evaluate_resnet(tmp_path):
    out = tmp_path / "run"
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--fake-data", "--model", "resnet20",
                    "--t-max", "1", "--opt", "sgd", "--lr", "0.1", "--wd", "5e-4", "--beta", "0", "--gamma", "0",
                    "--delta", "0", "--epochs", "1", "--max-steps", "3", "--batch-size", "16", "--workers", "0",
                    "--holdout", "64", "--out", str(out), "--eval-every", "1"], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--ckpt", str(out / "model.pt"),
                    "--fake-data", "--workers", "0", "--latency", "--latency-batch", "4"], check=True, cwd=ROOT)
    res = json.loads((out / "results.json").read_text())
    assert res["model_cfg"]["arch"] == "resnet" and len(res["test_ts"]) == 1
    assert 0.07 < res["flops"]["cumulative"][0] / 1e9 < 0.09  # ResNet-20: 0.082 GFLOPs
