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


@pytest.mark.parametrize("loss_steps,T,extra", [("last", 2, []), ("all", 3, ["--lateral", "--topdown"])])
def test_train_then_evaluate_vcnet(tmp_path, loss_steps, T, extra):
    out = tmp_path / "run"
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--fake-data", "--model", "vcnet_mini",
                    "--t-max", str(T), "--loss-steps", loss_steps, "--epochs", "1", "--max-steps", "3",
                    "--batch-size", "16", "--workers", "0", "--holdout", "64", "--out", str(out),
                    "--eval-every", "1", *extra], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--ckpt", str(out / "model.pt"),
                    "--fake-data", "--workers", "0", "--cache", "--cache-train", "32", "--latency",
                    "--latency-batch", "4"], check=True, cwd=ROOT)
    res = json.loads((out / "results.json").read_text())
    assert res["model_cfg"]["arch"] == "vcnet" and len(res["test_ts"]) == T
    assert len(res["flops"]["cumulative"]) == T


@pytest.mark.parametrize("kw", [dict(lateral=True), dict(lateral=True, topdown=True), dict(topdown=True)])
def test_area_state_variants(kw):
    torch.manual_seed(0)
    m = build_model("vcnet_mini", num_classes=10, t_max=4, **kw).eval()
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        z = m(x)
        torch.testing.assert_close(z[0], z[1])  # zero-init: identical to the plain loop at the start
        for mods in (m.lat, m.td):
            for conv in (mods.values() if mods is not None else []):
                conv.weight.normal_(0, 0.1)
        z, feats = m(x, return_feats=True)
        assert not torch.allclose(z[1], z[2]) and feats["cls"].shape == (4, 3, m.cfg.c_ait)
        # packed state round-trips, and the step interface reproduces forward()
        x0 = state = m.embed(x)
        for t in range(4):
            state = m.forward_step(state, t, x0)
            torch.testing.assert_close(m.readout(state, t), z[t])
        areas = m.unpack(state)
        assert set(areas) == {"v2", "mt", "v4", "pit", "cit", "mst", "ait"}
        torch.testing.assert_close(m.pack(areas), state)
        torch.testing.assert_close(m.ait_of(state), areas["ait"])
    fl = model_flops(m)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(torch.randn(1, 3, 32, 32))
    assert fl["cumulative"][-1] == fc.get_total_flops()


def test_runners_with_packed_state():
    from fsv.halting.features import build_features
    from fsv.halting.online import AdaptiveRunner, CudaGraphBS1
    from fsv.halting.policy import stop_steps, train_policy

    torch.manual_seed(0)
    T, N, C = 4, 16, 10
    m = build_model("vcnet_mini", num_classes=C, t_max=T, lateral=True, topdown=True).eval()
    with torch.no_grad():
        for mods in (m.lat, m.td):
            for conv in mods.values():
                conv.weight.normal_(0, 0.1)
        m.err_in.weight.normal_(0, 0.1)
    x, labels = torch.randn(N, 3, 32, 32), torch.randint(0, C, (N,))
    temps = torch.ones(T)
    with torch.no_grad():
        logits = m(x)
    feats = build_features({"logits": logits}, temps, "S0")
    p_off = logits.softmax(-1)
    pol, _ = train_policy(feats, p_off, labels, "reward_acc", iters=60)
    dev = torch.device("cpu")
    eager, cg = AdaptiveRunner(m, pol, temps, dev), CudaGraphBS1(m, temps, dev, policy=pol)
    for lam in (1e-3, 0.3):
        with torch.no_grad():
            stop = stop_steps(pol(feats, torch.full((N,), lam)))
        preds, steps = eager.run(x, lam)
        assert torch.equal(steps, stop)
        for i in range(0, N, 4):
            p1, s1 = cg.run(x[i:i + 1], lam)
            assert int(p1) == int(preds[i]) and s1 == int(steps[i])


def test_benchmark_fixed_depth_only_without_policy(tmp_path):
    # the original VCNet has no halting policy: the benchmark still times its fixed depths
    import os

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.run([sys.executable, str(ROOT / "scripts/run_experiment.py"), "smoke", "--fake-data",
                    "--max-cpus", "2", "--out-root", str(tmp_path), "--gpu", "",
                    "--extra", "--model vcnet_mini --t-max 2 --loss-steps last --max-steps 2 --batch-size 16 "
                               "--holdout 64"], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/benchmark_adaptive.py"), "smoke", "--fake-data",
                    "--out-root", str(tmp_path), "--repeats", "1", "--bs1-images", "3", "--batch-size", "64",
                    "--max-cpus", "2", "--no-compile"], check=True, cwd=ROOT, env=env)
    bench = json.loads((tmp_path / "reports" / "smoke" / "benchmark.json").read_text())
    assert bench["policy"] is None
    for mode in ("eager", "cudagraph-bs1"):
        assert len(bench["modes"][mode]["fixed"]) == 2 and bench["modes"][mode]["adaptive"] == []
    summary = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), "smoke", "--reports",
                              str(tmp_path / "reports")], check=True, capture_output=True, text=True, cwd=ROOT).stdout
    assert "[benchmark smoke]" in summary and "fixed t=1" in summary


def test_extend_steps_past_trained_length(tmp_path):
    m = _model(T=2)
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        base = m(x)
        m.extend_steps(5)
        ext = m(x)
    assert ext.shape == (5, 3, 10) and m.trained_t_max == 2
    torch.testing.assert_close(ext[:2], base)  # the trained steps are unchanged
    # CLI: evaluate an original-style checkpoint (2 steps, last readout) for 4 steps
    out = tmp_path / "run"
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--fake-data", "--model", "vcnet_mini",
                    "--t-max", "2", "--loss-steps", "last", "--epochs", "1", "--max-steps", "2",
                    "--batch-size", "16", "--workers", "0", "--holdout", "64", "--out", str(out)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--ckpt", str(out / "model.pt"),
                    "--fake-data", "--workers", "0", "--steps", "4"], check=True, cwd=ROOT)
    res = json.loads((out / "steps4" / "results.json").read_text())
    assert res["steps_run"] == 4 and len(res["test_ts"]) == 4 and len(res["flops"]["cumulative"]) == 4
    assert not (out / "results.json").exists()  # the normal evaluation output is not overwritten


def test_training_upsample_matches_interpolate():
    # training replaces F.interpolate by two matmuls (fast backward under Inductor); same values,
    # nothing new in the checkpoint
    import torch.nn.functional as F

    from fsv.models.vcnet import bilinear_matrix

    for n_in, n_out in [(4, 32), (4, 8), (8, 16), (3, 7)]:
        x = torch.randn(2, 3, n_in, n_in)
        u = bilinear_matrix(n_in, n_out)
        ref = F.interpolate(x, size=(n_out, n_out), mode="bilinear", align_corners=False)
        assert torch.allclose(u @ x @ u.T, ref, atol=1e-6)
    m = _model(T=3, lateral=True, topdown=True)
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim > 1:
                p.normal_(0, 0.05)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ev = m.eval()(x)
        tr = m.train()(x)
    assert torch.allclose(ev, tr, atol=1e-4)
    assert not [k for k in m.state_dict() if k.startswith("up_")]
