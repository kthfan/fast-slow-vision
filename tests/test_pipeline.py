"""End-to-end smoke test: a few training iterations on fake data, then evaluation."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_then_evaluate(tmp_path):
    out = tmp_path / "run"
    subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--fake-data", "--model", "bvit_tiny",
                    "--t-max", "3", "--epochs", "1", "--max-steps", "3", "--batch-size", "16", "--workers", "0",
                    "--holdout", "64", "--out", str(out), "--eval-every", "1"], check=True, cwd=ROOT)
    assert (out / "model.pt").exists()
    hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    assert len(hist[-1]["holdout"]) == 3
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--ckpt", str(out / "model.pt"),
                    "--fake-data", "--workers", "0", "--cache", "--cache-train", "32", "--latency",
                    "--latency-batch", "4"], check=True, cwd=ROOT)
    res = json.loads((out / "results.json").read_text())
    assert len(res["temperatures"]) == 3 and len(res["test_ts"]) == 3
    assert {"maxprob", "entropy", "fixed_depth"} <= res["curve_auc_acc"].keys()
    assert (out / "cache_test.pt").exists() and (out / "cache_train.pt").exists()


def test_run_experiment_writes_report(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "smoke", "--seeds", "0", "1",
           "--fake-data", "--max-cpus", "2", "--halting", "--cache-train", "64",
           "--out-root", str(tmp_path), "--gpu", "",
           "--extra", "--model bvit_tiny --t-max 3 --max-steps 3 --batch-size 16 --holdout 64"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    rep = tmp_path / "reports" / "smoke"
    for f in ("report.md", "results.json", "history.jsonl", "config.json", "env.json", "train.log", "eval.log"):
        assert (rep / f).exists(), f
    env = json.loads((rep / "env.json").read_text())
    assert env["train_exit"] == 0 and env["eval_exit"] == 0 and env["max_cpus"] <= 2
    cfg = json.loads((rep / "config.json").read_text())
    assert cfg["args"]["max_cpus"] <= 2 and cfg["args"]["workers"] <= 2
    text = (rep / "report.md").read_text()
    assert "Test set, per step" in text and "ECE TS %" in text
    assert env["code_version"].startswith("0.") and len(env["code_hash"]) == 12
    seed1 = json.loads((tmp_path / "reports" / "smoke_s1" / "env.json").read_text())
    assert seed1["seed"] == 1 and seed1["train_exit"] == 0
    summary = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), "smoke",
                              "--reports", str(tmp_path / "reports")], check=True, capture_output=True,
                             text=True, cwd=ROOT).stdout
    assert "BEGIN FSV SUMMARY" in summary and "[smoke] seeds 0,1" in summary and "test acc%" in summary
    assert "±" in summary
    assert (rep / "halting.json").exists() and "[halting smoke]" in summary and "S3-reward" in summary
    # halting-only on existing caches adds policies and keeps the earlier ones
    subprocess.run(cmd[:2] + ["smoke", "--fake-data", "--max-cpus", "2", "--out-root", str(tmp_path), "--gpu", "",
                              "--halting-only", "--halting-args",
                              "--feature-sets S2 S5 --objectives reward_acc --iters 30 --policy-seeds 2"],
                   check=True, cwd=ROOT)
    subprocess.run(cmd[:2] + ["smoke", "--fake-data", "--max-cpus", "2", "--out-root", str(tmp_path), "--gpu", "",
                              "--halting-only", "--halting-args",
                              "--feature-sets S0 --objectives reward_acc --iters 30 --policy-seeds 1 --policy-train train"],
                   check=True, cwd=ROOT)
    subprocess.run(cmd[:2] + ["smoke", "--fake-data", "--max-cpus", "2", "--out-root", str(tmp_path), "--gpu", "",
                              "--halting-only", "--halting-args",
                              "--feature-sets S0 --objectives --o3-ks 0 1 --iters 30 --policy-seeds 1"],
                   check=True, cwd=ROOT)
    halt = json.loads((rep / "halting.json").read_text())
    pol = halt["policies"]
    assert {"S2-reward_acc", "S5-reward_acc", "S0-reward", "S3-oracle_any", "S0-reward_acc@train"} <= pol.keys()
    assert halt["policy_train"]["split"] == "train"
    assert (rep / "policies" / "S0-reward_acc.pt").exists() and "cost@full" in pol["S0-reward_acc"]
    subprocess.run([sys.executable, str(ROOT / "scripts/benchmark_adaptive.py"), "smoke", "--fake-data",
                    "--out-root", str(tmp_path), "--repeats", "1", "--bs1-images", "3", "--batch-size", "64",
                    "--lams", "0.01", "1.0", "--max-cpus", "2", "--no-compile", "--naive"], check=True, cwd=ROOT)
    bench = json.loads((rep / "benchmark.json").read_text())
    assert set(bench["modes"]) == {"eager", "eager-naive", "o3k0-eager", "o3k1-eager", "cudagraph-bs1",
                                   "o3k0-cudagraph-bs1", "o3k1-cudagraph-bs1"}
    assert bench["cudagraph_captured"] is False  # CPU: logic only
    assert all(a["agree_with_eager"] == 1.0 for m in bench["modes"] if "cudagraph" in m
               for a in bench["modes"][m]["adaptive"])
    assert all(1 <= a["mean_steps"] <= 3 for a in bench["modes"]["o3k1-eager"]["adaptive"])
    assert len(bench["modes"]["eager"]["fixed"]) == 3 and len(bench["modes"]["eager"]["adaptive"]) == 2
    # both runners make the same decisions
    assert [a["acc"] for a in bench["modes"]["eager"]["adaptive"]] == \
        [a["acc"] for a in bench["modes"]["eager-naive"]["adaptive"]]
    summary = subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), "smoke",
                              "--reports", str(tmp_path / "reports")], check=True, capture_output=True,
                             text=True, cwd=ROOT).stdout
    assert "[benchmark smoke]" in summary and "cost@full" in summary
    # re-evaluating with --skip-train keeps the training record
    subprocess.run(cmd[:2] + ["smoke", "--fake-data", "--max-cpus", "2", "--out-root", str(tmp_path), "--gpu", "",
                              "--skip-train"], check=True, cwd=ROOT)
    env2 = json.loads((rep / "env.json").read_text())
    assert env2["train_minutes"] == env["train_minutes"] and "eval_code_version" in env2
