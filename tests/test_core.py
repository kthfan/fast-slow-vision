import math

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from fsv.calibration import fit_temperature
from fsv.flops import model_flops
from fsv.halting import curve_auc, fixed_depth_curve, oracle_steps, threshold_curve
from fsv.losses import StageALossConfig, stage_a_loss
from fsv.metrics import classification_metrics, ece
from fsv.mixup import Mixup
from fsv.models import build_model


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model("bvit_tiny", num_classes=10, t_max=4).eval()


def test_forward_shapes_and_step_interface(model):
    x = torch.randn(2, 3, 32, 32)
    logits, feats = model(x, return_feats=True)
    assert logits.shape == (4, 2, 10)
    assert feats["cls"].shape == (4, 2, 192) and feats["pool"].shape == (4, 2, 192)
    tok = model.embed(x)
    for t in range(4):
        tok = model.forward_step(tok, t)
        torch.testing.assert_close(model.readout(tok, t), logits[t])
    assert model(x, t_max=2).shape == (2, 2, 10)


def test_weights_are_shared_across_steps(model):
    # Only one block: parameter count does not grow with t_max.
    a = sum(p.numel() for p in build_model("bvit_tiny", num_classes=10, t_max=4).block.parameters())
    b = sum(p.numel() for p in build_model("bvit_tiny", num_classes=10, t_max=12).block.parameters())
    assert a == b


def test_no_batchnorm(model):
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_rmsnorm_and_te_variants():
    m = build_model("bvit_tiny", num_classes=10, t_max=3, norm="rmsnorm", step_embed_in=True, layerscale_init=1e-4)
    assert m(torch.randn(1, 3, 32, 32)).shape == (3, 1, 10)


def test_stage_a_loss_finite_and_backprop(model):
    model.train()
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    x, target = Mixup(10)(x, y)
    loss, parts = stage_a_loss(model(x), target, StageALossConfig())
    loss.backward()
    assert torch.isfinite(loss)
    assert {"ce", "brier", "conf", "self_distill", "total"} <= parts.keys()
    assert model.block.attn.qkv.weight.grad is not None
    assert model.step_out.grad is not None


def test_self_distill_zero_when_steps_agree():
    logits = torch.randn(1, 5, 7).repeat(3, 1, 1)
    target = torch.nn.functional.one_hot(torch.randint(0, 7, (5,)), 7).float()
    _, parts = stage_a_loss(logits, target, StageALossConfig(gamma=0))
    assert abs(float(parts["self_distill"])) < 1e-6


def test_metrics_basic():
    labels = torch.tensor([0, 1, 2, 3])
    probs = torch.nn.functional.one_hot(labels, 4).float()
    m = classification_metrics(probs, labels)
    assert m["acc"] == 1.0 and m["ece"] < 1e-6 and m["brier"] < 1e-6 and m["aurc"] == 0.0
    # Over-confident wrong predictions -> large ECE
    assert ece(torch.nn.functional.one_hot((labels + 1) % 4, 4).float(), labels) > 0.99


def test_temperature_fixes_overconfidence():
    torch.manual_seed(0)
    N, C = 4000, 10
    labels = torch.randint(0, C, (N,))
    clean = torch.randn(N, C)
    clean[torch.arange(N), labels] += 1.5
    logits = clean * 4.0  # sharpen -> over-confident
    T = fit_temperature(logits, labels)
    assert 2.5 < T < 6.0
    before = classification_metrics(logits.softmax(-1), labels)
    after = classification_metrics((logits / T).softmax(-1), labels)
    assert after["nll"] < before["nll"] and after["ece"] < before["ece"]


def test_analytic_flops_match_counter(model):
    fl = model_flops(model)
    with FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, 32, 32))
    counted = fc.get_total_flops()
    assert math.isclose(fl["cumulative"][-1], counted, rel_tol=0.05), (fl["cumulative"][-1], counted)


def test_threshold_curves_and_oracle():
    torch.manual_seed(0)
    T, N, C = 4, 200, 5
    labels = torch.randint(0, C, (N,))
    logits = torch.randn(T, N, C)
    for t in range(T):  # later steps are more accurate
        logits[t, torch.arange(N), labels] += 0.8 * t
    probs = logits.softmax(-1)
    cum = [1e9 * (t + 1) for t in range(T)]
    fixed = fixed_depth_curve(probs, labels, cum)
    assert [p["gflops"] for p in fixed] == [1.0, 2.0, 3.0, 4.0]
    mp = threshold_curve(probs, labels, cum, "maxprob")
    # tau = 0 stops at step 0; tau > 1 runs to the end
    assert mp[0]["mean_steps"] == 1.0 and mp[-1]["mean_steps"] == T
    ent = threshold_curve(probs, labels, cum, "entropy")
    assert min(p["mean_steps"] for p in ent) == 1.0 and max(p["mean_steps"] for p in ent) == T
    t_star = oracle_steps(probs, labels, "full")
    correct = probs.argmax(-1) == labels
    for i in range(N):
        s = int(t_star[i])
        if correct[-1, i]:
            assert correct[s:, i].all() and (s == 0 or not correct[s - 1, i])
        else:
            assert s == T - 1
    assert 0 <= curve_auc(mp, 1.0, 4.0) <= 1


def test_mixup_targets_sum_to_one():
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
    for _ in range(10):
        _, t = Mixup(10)(x, y)
        torch.testing.assert_close(t.sum(-1), torch.ones(8))
