"""Run named experiments end to end and write reports.

    python scripts/run_experiment.py --list
    python scripts/run_experiment.py smoke
    python scripts/run_experiment.py c100_bvit_s --seeds 0 1 2
    python scripts/run_experiment.py c100_bvit_s c100_vit_s_8blk --seeds 1 2

Each (experiment, seed) pair writes
    runs/<name>[_s<seed>]/      checkpoints (model.pt, last.pt): large, stay on your machine
    reports/<name>[_s<seed>]/   report.md, results.json, history.jsonl, config.json, env.json, logs
Seed 0 uses the bare experiment name.

Afterwards print one compact, paste-friendly block for everything in reports/:
    python scripts/summarize.py

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
from fsv.version import code_info  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="experiment names (see --list)")
    p.add_argument("--list", action="store_true", help="list experiments and exit")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="training seeds (default: 0)")
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


def run_name(exp: str, seed: int) -> str:
    return exp if seed == 0 else f"{exp}_s{seed}"


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


def run_one(args, exp_name: str, seed: int, n_cpus: int, machine_cpus: int, tinfo: dict, cinfo: dict) -> bool:
    exp = EXPERIMENTS[exp_name]
    name = run_name(exp_name, seed)
    out_root = Path(args.out_root)
    run_dir, report_dir = out_root / "runs" / name, out_root / "reports" / name
    if args.fresh:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(report_dir, ignore_errors=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, **thread_env(n_cpus), "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1"}
    py = sys.executable
    train_cmd = [py, "scripts/train.py", *exp["train"], *shlex.split(args.extra), "--seed", str(seed),
                 "--data-dir", args.data_dir, "--out", str(run_dir), "--max-cpus", str(n_cpus), "--resume"]
    eval_cmd = [py, "scripts/evaluate.py", "--ckpt", str(run_dir / "model.pt"), "--data-dir", args.data_dir,
                "--max-cpus", str(n_cpus), "--latency"] + (["--cache"] if args.cache else [])
    if args.fake_data:
        train_cmd.append("--fake-data")
        eval_cmd += ["--fake-data", "--latency-batch", "8"]

    info = {"experiment": exp_name, "run": name, "seed": seed, "description": exp["desc"], **cinfo, **tinfo,
            "python": platform.python_version(), "max_cpus": n_cpus, "machine_cpus": machine_cpus,
            "started": dt.datetime.now().isoformat(timespec="seconds"), "extra": args.extra,
            "train_cmd": shlex.join(train_cmd[1:]), "eval_cmd": shlex.join(eval_cmd[1:])}
    print(f"\n[run_experiment] ===== {name}: {exp['desc']}")
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
    print(f"[run_experiment] {name}: {'done' if ok else 'FINISHED WITH ERRORS'}; report: {report}")
    return ok


def main(argv=None):
    args = parse_args(argv)
    if args.list or not args.names:
        for k, v in EXPERIMENTS.items():
            print(f"{k:<24} {v['desc']}")
        return 0
    unknown = [n for n in args.names if n not in EXPERIMENTS]
    if unknown:
        sys.exit(f"unknown experiment(s) {unknown}; run with --list")

    machine_cpus = len(available_cpus())
    n_cpus = limit_cpus(args.max_cpus)  # pins this process; children inherit the affinity
    tinfo, cinfo = torch_info(args.gpu), code_info()
    print(f"[run_experiment] code {cinfo['code_version']}, CPU cap {n_cpus}/{machine_cpus} cores, "
          f"GPU {tinfo.get('gpu')}; runs: " + ", ".join(run_name(e, s) for e in args.names for s in args.seeds))

    failed = [run_name(e, s) for e in args.names for s in args.seeds
              if not run_one(args, e, s, n_cpus, machine_cpus, tinfo, cinfo)]
    print("\n[run_experiment] all runs finished" + (f"; FAILED: {failed}" if failed else ""))
    print("[run_experiment] paste the output of:  python scripts/summarize.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
