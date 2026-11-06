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
