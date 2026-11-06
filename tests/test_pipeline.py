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
    cmd = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "smoke", "--fake-data", "--max-cpus", "2",
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
