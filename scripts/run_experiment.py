"""Run a named experiment end to end and write an uploadable report.

    python scripts/run_experiment.py --list
    python scripts/run_experiment.py smoke --max-cpus 8
    python scripts/run_experiment.py c100_bvit_s --max-cpus 8

Outputs:
    runs/<name>/      checkpoints (model.pt, last.pt) - large, stay on your machine
    reports/<name>/   report.md, results.json, history.jsonl, config.json, env.json,
                      train.log, eval.log - small, this is what you upload

An interrupted run resumes from runs/<name>/last.pt when started again;
pass --fresh to discard it and start over.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsv.cpu import available_cpus, limit_cpus, thread_env  # noqa: E402
from fsv.experiments import EXPERIMENTS  # noqa: E402
from fsv.report import write_report  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?", help="experiment name (see --list)")
    p.add_argument("--list", action="store_true", help="list experiments and exit")
    p.add_argument("--max-cpus", type=int, default=None, help="hard CPU-core cap (default: $FSV_MAX_CPUS or 8)")
    p.add_argument("--gpu", default="0", help="value for CUDA_VISIBLE_DEVICES (default 0)")
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--fresh", action="store_true", help="delete runs/<name> and reports/<name> first")
    p.add_argument("--cache", action="store_true", help="also save per-step logits/features (stays in runs/)")
    p.add_argument("--skip-train", action="store_true", help="only (re)evaluate runs/<name>/model.pt")
    p.add_argument("--out-root", default=str(ROOT), help="where runs/ and reports/ are created")
    p.add_argument("--fake-data", action="store_true", help=argparse.SUPPRESS)  # tests only
    p.add_argument("--extra", default="", help="extra train.py flags, recorded in the report (avoid for final runs)")
    return p.parse_args(argv)


def git_info() -> dict:
    def g(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "?"
    return {"git_commit": g("rev-parse", "HEAD"), "git_branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def torch_info(gpu: str) -> dict:
    code = ("import torch, json; c = torch.cuda.is_available(); print(json.dumps({'torch': torch.__version__, "
            "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if c else 'none (CPU)'}))")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}).stdout
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:  # pragma: no cover
        return {"torch": f"error: {e}"}


def run_logged(cmd: list[str], log_path: Path, env: dict) -> int:
    """Run cmd, streaming its output to the console and to log_path."""
    print(f"\n$ {shlex.join(cmd)}\n", flush=True)
    with open(log_path, "a") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def main(argv=None):
    args = parse_args(argv)
    if args.list or not args.name:
        for k, v in EXPERIMENTS.items():
            print(f"{k:<24} {v['desc']}")
        return 0
    if args.name not in EXPERIMENTS:
        sys.exit(f"unknown experiment {args.name!r}; run with --list")
    exp = EXPERIMENTS[args.name]

    machine_cpus = len(available_cpus())
    n_cpus = limit_cpus(args.max_cpus)  # pins this process; children inherit the affinity
    out_root = Path(args.out_root)
    run_dir, report_dir = out_root / "runs" / args.name, out_root / "reports" / args.name
    if args.fresh:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(report_dir, ignore_errors=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, **thread_env(n_cpus), "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1"}
    py = sys.executable
    train_cmd = [py, "scripts/train.py", *exp["train"], *shlex.split(args.extra), "--data-dir", args.data_dir,
                 "--out", str(run_dir), "--max-cpus", str(n_cpus), "--resume"]
    eval_cmd = [py, "scripts/evaluate.py", "--ckpt", str(run_dir / "model.pt"), "--data-dir", args.data_dir,
                "--max-cpus", str(n_cpus), "--latency"] + (["--cache"] if args.cache else [])
    if args.fake_data:
        train_cmd.append("--fake-data")
        eval_cmd += ["--fake-data", "--latency-batch", "8"]

    info = {"experiment": args.name, "description": exp["desc"], **git_info(), **torch_info(args.gpu),
            "python": platform.python_version(), "max_cpus": n_cpus, "machine_cpus": machine_cpus,
            "started": dt.datetime.now().isoformat(timespec="seconds"), "extra": args.extra,
            "train_cmd": shlex.join(train_cmd[1:]), "eval_cmd": shlex.join(eval_cmd[1:])}
    print(f"[run_experiment] {args.name}: {exp['desc']}")
    print(f"[run_experiment] CPU cap {n_cpus}/{machine_cpus} cores, GPU {info.get('gpu')}, "
          f"commit {info['git_commit'][:8]}{' (dirty)' if info['git_dirty'] else ''}")
    env_path = report_dir / "env.json"

    def save_env():
        env_path.write_text(json.dumps(info, indent=2))

    save_env()
    if not args.skip_train:
        t0 = time.time()
        info["train_exit"] = run_logged(train_cmd, report_dir / "train.log", env)
        info["train_minutes"] = (time.time() - t0) / 60
        save_env()
        if info["train_exit"] != 0:
            print(f"[run_experiment] training failed (exit {info['train_exit']}); see {report_dir / 'train.log'}")
    if (run_dir / "model.pt").exists() and info.get("train_exit", 0) == 0:
        t0 = time.time()
        info["eval_exit"] = run_logged(eval_cmd, report_dir / "eval.log", env)
        info["eval_minutes"] = (time.time() - t0) / 60
    info["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    save_env()

    for f in ("config.json", "history.jsonl", "results.json"):
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, report_dir / f)
    report = write_report(report_dir)
    ok = info.get("train_exit", 0) == 0 and info.get("eval_exit") == 0
    print(f"\n[run_experiment] {'done' if ok else 'FINISHED WITH ERRORS'}; report: {report}")
    print(f"[run_experiment] upload it with:\n  git add {report_dir.relative_to(out_root)} && "
          f"git commit -m 'Report: {args.name}' && git push")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
