"""VCNet: step interface, paper-stated structure, FLOPs, and the adaptive runners."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from fsv.flops import model_flops
from fsv.models import build_model, load_checkpoint_model, config_dict
from fsv.models.vcnet import RecurrentBlock

ROOT = Path(__file__).resolve().parents[1]


def _model(T=4, **kw):
    torch.manual_seed(0)
    m = build_model("vcnet_mini", num_classes=10, t_max=T, **kw).eval()
    with torch.no_grad():  # move the zero-init feedback path off zero so steps differ
        m.err_in.weight.normal_(0, 0.1)
        m.step_out.normal_(0, 0.1)
    return m


def test_forward_matches_step_interface():
    m = _model()
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        logits, feats = m(x, return_feats=True)
        assert logits.shape == (4, 3, 10)
        assert feats["cls"].shape == (4, 3, m.cfg.c_ait) and feats["pool"].shape == (4, 3, m.cfg.c_ait)
        x0 = state = m.embed(x)
        for t in range(4):
            state = m.forward_step(state, t, x0)
            torch.testing.assert_close(m.readout(state, t), logits[t])
        assert m(x, t_max=2).shape == (2, 3, 10)
    assert not torch.allclose(logits[0], logits[1])  # the feedback changes the later sweeps


def test_feedback_is_identity_at_init_and_error_formula():
    torch.manual_seed(0)
    m = build_model("vcnet_mini", num_classes=10, t_max=3).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        z = m(x)
        torch.testing.assert_close(z[0], z[1])  # zero-init err_in and step_out: step 1 == step 0
        v1 = m.embed(x)
        ait = m.forward_step(v1, 0, v1)
        eps = m.prediction_error(v1, ait)
        pred = m.predict_v1(ait, v1.shape[-2:])
        torch.testing.assert_close(eps, torch.relu(v1 - pred))  # eps = ReLU(V1 - AIT prediction)
        assert pred.shape == v1.shape and (eps >= 0).all()


def test_paper_structure():
    m = build_model("vcnet", num_classes=100, t_max=2)
    assert not any(isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) for mod in m.modules())
    ks = sorted(br[0].kernel_size[0] for br in m.v1_branches)
    assert ks == [3, 5, 7] and all(br[0].groups == 3 for br in m.v1_branches)  # depthwise separable
    recs = [mod for mod in m.modules() if isinstance(mod, RecurrentBlock)]
    assert len(recs) == 2 and all(r.iters == 3 for r in recs)  # MT and MST, t = 3
    # parameters do not grow with the number of predictive-coding steps (only the step embedding)
    a = sum(p.numel() for p in build_model("vcnet", num_classes=100, t_max=2).parameters())
    b = sum(p.numel() for p in build_model("vcnet", num_classes=100, t_max=8).parameters())
    assert b - a == 6 * m.cfg.c_ait
    assert 0.8e6 < a < 1.3e6  # paper: full VCNet 3.52 MB (~0.9M fp32 parameters)


def test_measured_flops_match_full_forward():
    m = _model()
    fl = model_flops(m)
    assert len(fl["cumulative"]) == 4 and fl["step"][1] > fl["step"][0]  # step 0 has no top-down prediction
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(torch.randn(1, 3, 32, 32))
    assert fl["cumulative"][-1] == fc.get_total_flops()


def test_ait_carry_and_checkpoint_roundtrip():
    m = _model(T=3, ait_carry=True)
    ck = {"model": m.state_dict(), "model_cfg": config_dict(m)}
    m2 = load_checkpoint_model(ck).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(m(x), m2(x))
    assert m2.carry_gate is not None


def test_adaptive_runners_with_vcnet():
    from fsv.halting.features import build_features
    from fsv.halting.online import AdaptiveRunner, CudaGraphBS1, O3Runner
    from fsv.halting.policy import stop_steps, train_policy, train_step_predictor

    T, N, C = 4, 24, 10
    m = _model(T)
    x = torch.randn(N, 3, 32, 32)
    labels = torch.randint(0, C, (N,))
    temps = torch.tensor([1.2, 1.0, 0.9, 0.8])
    with torch.no_grad():
        logits = m(x)
    feats = build_features({"logits": logits}, temps, "S0")
    p_off = (logits / temps[:, None, None]).softmax(-1)
    pol, _ = train_policy(feats, p_off, labels, "reward_acc", iters=60)
    o3, _ = train_step_predictor(feats, p_off, labels, 1, iters=60)
    dev = torch.device("cpu")
    eager, o3_eager = AdaptiveRunner(m, pol, temps, dev), O3Runner(m, o3, temps, dev)
    cg = CudaGraphBS1(m, temps, dev, policy=pol, o3_policy=o3)
    for lam in (1e-3, 0.3):
        with torch.no_grad():
            stop = stop_steps(pol(feats, torch.full((N,), lam)))
        preds, steps = eager.run(x, lam)
        assert torch.equal(steps, stop)
        assert torch.equal(preds, p_off.argmax(-1)[stop, torch.arange(N)])
        po, so = o3_eager.run(x, lam)
        for i in range(0, N, 4):
            p1, s1 = cg.run(x[i:i + 1], lam)
            assert int(p1) == int(preds[i]) and s1 == int(steps[i])
            p3, s3 = cg.run_o3(x[i:i + 1], lam)
            assert int(p3) == int(po[i]) and s3 == int(so[i])
    for t in range(T):
        assert torch.equal(eager.run_fixed(x, t), logits[t].argmax(-1))
        assert int(cg.run_fixed(x[:1], t)) == int(logits[t, 0].argmax())


@pytest.mark.parametrize("loss_steps,T", [("last", 2), ("all", 3)])
def test_train_then_evaluate_vcnet(tmp_path, loss_steps, T):
    out = tmp_path / "run"
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--fake-data", "--model", "vcnet_mini",
                    "--t-max", str(T), "--loss-steps", loss_steps, "--epochs", "1", "--max-steps", "3",
                    "--batch-size", "16", "--workers", "0", "--holdout", "64", "--out", str(out),
                    "--eval-every", "1"], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--ckpt", str(out / "model.pt"),
                    "--fake-data", "--workers", "0", "--cache", "--cache-train", "32", "--latency",
                    "--latency-batch", "4"], check=True, cwd=ROOT)
    res = json.loads((out / "results.json").read_text())
    assert res["model_cfg"]["arch"] == "vcnet" and len(res["test_ts"]) == T
    assert len(res["flops"]["cumulative"]) == T
