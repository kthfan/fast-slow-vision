"""Training-speed benchmark: images per second of the real training step (mixup + forward of all
steps + Stage A loss + backward + clip + AdamW) under different pipelines, for named experiments.

    python scripts/bench_train.py c100_vcnet_orig c100_vcnet_adapt_blt
    python scripts/bench_train.py c100_vcnet_adapt_blt --configs gpu-cg gpu-cg-bs512 --procs 3

Configs (``--configs``; preset names, or key=value lists such as "data=gpu,compile=reduce-overhead,bs=512"):
    cpu-eager      CPU DataLoader (PIL + RandAugment), eager, foreach AdamW  = the recipe of rounds 1-3
    cpu-cg         CPU DataLoader + compiled step with CUDA graphs: is the CPU loader the ceiling?
    cpu-cg-cl / gpu-cg-cl   + channels_last (NHWC) weights and activations
    gpu-eager      augmentation on the GPU (fsv/gpu_aug.py), eager
    gpu-fused      + fused AdamW
    gpu-compile    + torch.compile (Inductor)
    gpu-cg         + CUDA graphs (compile mode reduce-overhead); the next batch is augmented on a side
                   stream while the step runs (0.15.1; gpu-cg-noprefetch = the same on one stream)
    gpu-cg-bs512 / gpu-cg-bs1024   bigger batches (the lr would need re-tuning; speed only)
Keys: data=cpu|gpu, compile=none|default|reduce-overhead|max-autotune-no-cudagraphs,
      opt=foreach|fused, bs=<global batch>, cl=0|1 (channels_last), prefetch=0|1 (GPU data only).

``--profile N`` adds a torch.profiler table of the top GPU kernels over N iterations of every config
(use it with configs without CUDA graphs, e.g. gpu-compile: inside a graph the kernels are not listed).

``--procs N`` runs N copies of every config at once on the same GPU (disjoint CPU cores, measured
over the same window), i.e. N seeds in parallel; the img/s column is then the sum over copies.
Also measured once per experiment: the CPU DataLoader alone and the GPU augmentation alone.

Not included: the held-out evaluation every 10 epochs and the final evaluate.py pass.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train as T  # noqa: E402
from fsv.cpu import default_workers, limit_cpus, split_cpus  # noqa: E402
from fsv.data import NUM_CLASSES, build_splits  # noqa: E402
from fsv.engine import auto_amp  # noqa: E402
from fsv.experiments import EXPERIMENTS  # noqa: E402
from fsv.mixup import Mixup  # noqa: E402
from fsv.train_step import TrainStep, make_optimizer  # noqa: E402

PRESETS = {
    "cpu-eager": {"data": "cpu", "compile": "none", "opt": "foreach"},
    "cpu-cg": {"data": "cpu", "compile": "reduce-overhead", "opt": "fused"},
    "gpu-eager": {"data": "gpu", "compile": "none", "opt": "foreach"},
    "gpu-fused": {"data": "gpu", "compile": "none", "opt": "fused"},
    "gpu-compile": {"data": "gpu", "compile": "default", "opt": "fused"},
    "gpu-cg": {"data": "gpu", "compile": "reduce-overhead", "opt": "fused"},
    "gpu-cg-noprefetch": {"data": "gpu", "compile": "reduce-overhead", "opt": "fused", "prefetch": 0},
    "gpu-cg-cl": {"data": "gpu", "compile": "reduce-overhead", "opt": "fused", "cl": 1},
    "cpu-cg-cl": {"data": "cpu", "compile": "reduce-overhead", "opt": "fused", "cl": 1},
    "gpu-cg-bs512": {"data": "gpu", "compile": "reduce-overhead", "opt": "fused", "bs": 512},
    "gpu-cg-bs1024": {"data": "gpu", "compile": "reduce-overhead", "opt": "fused", "bs": 1024},
}
DEFAULT_CONFIGS = ["cpu-eager", "cpu-cg", "gpu-cg-noprefetch", "gpu-cg", "cpu-cg-cl", "gpu-cg-cl", "gpu-cg-bs512",
                   "gpu-cg-bs1024"]


def parse_config(spec: str) -> dict:
    if spec in PRESETS:
        return {"name": spec, **PRESETS[spec]}
    cfg = {"name": spec, "data": "gpu", "compile": "none", "opt": "foreach"}
    for kv in spec.split(","):
        k, v = kv.split("=")
        cfg[k.strip()] = int(v) if k.strip() in ("bs", "cl", "prefetch") else v.strip()
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", help="experiment names (scripts/run_experiment.py --list)")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p.add_argument("--iters", type=int, default=50, help="timed iterations per repeat")
    p.add_argument("--repeats", type=int, default=3, help="timed repeats (median reported)")
    p.add_argument("--warmup", type=int, default=15, help="untimed iterations first (includes compilation)")
    p.add_argument("--procs", type=int, default=1, help="copies of each config run at once (parallel seeds)")
    p.add_argument("--max-cpus", type=int, default=None)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--extra", default="", help="extra train.py flags on top of the recipe (e.g. for tests)")
    p.add_argument("--fake-data", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--json", default=None, help="also write all results to this JSON file")
    p.add_argument("--no-data-only", action="store_true", help="skip the loader-only measurements")
    p.add_argument("--profile", type=int, default=0, help="profile N extra iterations per config (top GPU kernels)")
    # internal: one copy of a --procs run
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)  # "<barrier dir>:<id>:<n>"
    return p.parse_args(argv)


# --------------------------------------------------------------------------- one measurement
_SPLITS = {}


def splits_for(targs):
    key = (targs.dataset, targs.data_dir, targs.fake_data, targs.no_randaug, targs.erase)
    if key not in _SPLITS:
        _SPLITS[key] = build_splits(targs.dataset, targs.data_dir, targs.holdout, seed=0, download=True,
                                    fake=targs.fake_data, randaug=not targs.no_randaug, erase=targs.erase)
    return _SPLITS[key]


def infinite(loader):
    while True:
        yield from loader


def make_loader(targs, cfg, device, workers):
    ds = splits_for(targs)["train"]
    if cfg["data"] == "gpu":
        return T.gpu_loader(targs, ds, targs.batch_size, device, prefetch=bool(cfg.get("prefetch", 1)))
    return DataLoader(ds, batch_size=targs.batch_size, shuffle=True, num_workers=workers,
                      pin_memory=device.type == "cuda", drop_last=True, persistent_workers=workers > 0)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def barrier_wait(child: str | None, tag: str):
    """--procs copies start their timed window together."""
    if not child:
        return
    d, i, n = child.split(":")
    Path(d, f"{tag}.{i}").touch()
    t0 = time.time()
    while sum(Path(d, f"{tag}.{k}").exists() for k in range(int(n))) < int(n):
        if time.time() - t0 > 1800:
            raise TimeoutError("other copies did not reach the barrier")
        time.sleep(0.05)


def measure(exp: str, cfg: dict, args, device, workers) -> dict:
    targv = [*EXPERIMENTS[exp]["train"], *shlex.split(args.extra), "--data-dir", args.data_dir]
    if "bs" in cfg:
        targv += ["--batch-size", str(cfg["bs"])]
    if args.fake_data:
        targv.append("--fake-data")
    targs = T.parse_args(targv)
    amp = targs.amp if targs.amp != "auto" else auto_amp(device)
    torch.manual_seed(0)
    model = T.model_from_args(targs, NUM_CLASSES[targs.dataset]).to(device)
    cl = bool(cfg.get("cl", 0))
    if cl:
        model = model.to(memory_format=torch.channels_last)
    opt = make_optimizer(T.param_groups(model, targs.wd), targs.lr, device, cfg["opt"], targs.opt, targs.momentum)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    mixup = Mixup(NUM_CLASSES[targs.dataset], targs.mixup, targs.cutmix, targs.mix_prob, groups=targs.mix_groups)
    step = TrainStep(model, model.parameters(), opt, scaler, T.loss_config(targs), device, amp, targs.clip,
                     targs.loss_steps, None if cfg["compile"] == "none" else cfg["compile"], cl)
    model.train()
    data = infinite(make_loader(targs, cfg, device, workers))

    def one():
        x, y = next(data)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, target = mixup(x, y)
        return step(x, target)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    t0 = time.perf_counter()
    one()
    sync(device)
    first = time.perf_counter() - t0
    for _ in range(max(0, args.warmup - 1)):
        one()
    sync(device)
    warm = time.perf_counter() - t0
    barrier_wait(args.child, f"{exp}.{cfg['name']}")
    reps = []
    for _ in range(args.repeats):
        sync(device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            parts = one()
        float(parts["total"])  # also checks the loss is finite below
        sync(device)
        reps.append((time.perf_counter() - t0) / args.iters)
    reps.sort()
    ms = reps[len(reps) // 2] * 1e3
    loss = float(parts["total"])
    n_train = len(splits_for(targs)["train"])
    res = {"experiment": exp, "config": cfg["name"], **{k: cfg[k] for k in ("data", "compile", "opt")},
           "batch_size": targs.batch_size, "ms_per_iter": ms, "img_per_s": targs.batch_size / ms * 1e3,
           "first_iter_s": first, "warmup_s": warm, "max_over_min": reps[-1] / reps[0], "loss": loss,
           "iters_per_epoch": n_train // targs.batch_size, "epochs": targs.epochs, "amp": amp,
           "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0}
    res["minutes_per_run"] = res["epochs"] * res["iters_per_epoch"] * ms / 1e3 / 60
    res["channels_last"] = cl
    if args.profile > 0:
        res["profile"] = profile_table(one, args.profile, device)
    del step, opt, model, data
    torch._dynamo.reset()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def profile_table(one, n: int, device, rows: int = 25) -> str:
    from torch.autograd.profiler_util import FunctionEventAvg
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(n):
            one()
        sync(device)
    if device.type != "cuda":
        key = "self_cpu_time_total"
    else:
        key = "self_device_time_total" if hasattr(FunctionEventAvg, "self_device_time_total") else "self_cuda_time_total"
    return prof.key_averages().table(sort_by=key, row_limit=rows, max_name_column_width=70)


def data_only(exp: str, args, device, workers, n_batches: int = 60) -> list[dict]:
    targs = T.parse_args([*EXPERIMENTS[exp]["train"], *shlex.split(args.extra), "--data-dir", args.data_dir,
                          *(["--fake-data"] if args.fake_data else [])])
    out = []
    for kind in ("cpu", "gpu"):
        data = infinite(make_loader(targs, {"data": kind}, device, workers))
        for _ in range(5):
            next(data)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n_batches):
            x, _ = next(data)
            x = x.to(device, non_blocking=True)
        sync(device)
        dt = time.perf_counter() - t0
        out.append({"loader": kind, "img_per_s": n_batches * targs.batch_size / dt, "workers": workers})
        del data
    return out


# --------------------------------------------------------------------------- --procs: copies side by side
def run_copies(exp: str, spec: str, args, n_cpus: int) -> dict:
    slots = split_cpus(n_cpus, args.procs)
    with tempfile.TemporaryDirectory() as d:
        procs = []
        for i, cores in enumerate(slots):
            cmd = [sys.executable, __file__, exp, "--configs", spec, "--iters", str(args.iters),
                   "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--max-cpus", str(len(cores)),
                   "--data-dir", args.data_dir, "--extra", args.extra, "--no-data-only",
                   "--json", str(Path(d, f"out{i}.json")), "--child", f"{d}:{i}:{args.procs}"]
            if args.fake_data:
                cmd.append("--fake-data")
            env = {**os.environ, "FSV_CPU_SET": ",".join(map(str, cores))}
            procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
        errs = [p.communicate()[1] for p in procs]
        if any(p.returncode for p in procs):
            raise RuntimeError(f"a copy failed:\n{next(e for p, e in zip(procs, errs) if p.returncode)[-3000:]}")
        rows = [json.loads(Path(d, f"out{i}.json").read_text())["runs"][0] for i in range(args.procs)]
    agg = dict(rows[0])
    agg["procs"] = args.procs
    agg["img_per_s"] = sum(r["img_per_s"] for r in rows)
    agg["img_per_s_per_copy"] = [r["img_per_s"] for r in rows]
    agg["ms_per_iter"] = sum(r["ms_per_iter"] for r in rows) / len(rows)
    agg["minutes_per_run"] = max(r["minutes_per_run"] for r in rows)
    agg["peak_mem_gb"] = sum(r["peak_mem_gb"] for r in rows)
    return agg


# --------------------------------------------------------------------------- main
def print_table(exp: str, rows: list[dict], loaders: list[dict], device_name: str, procs: int):
    base = next((r for r in rows if r["config"] == "cpu-eager"), rows[0])
    seeds = max(3, procs)
    print(f"\n[bench_train {exp}] {device_name} | amp {rows[0]['amp']} | {rows[0]['epochs']} epochs | "
          f"copies at once: {procs} | speed-up vs {base['config']}")
    w = max(16, *(len(r["config"]) for r in rows))
    head = (f"  {'config':<{w}} {'bs':>5} {'img/s':>9} {'ms/it':>8} {'x':>6} {'1st it s':>9} {'peak GB':>8} "
            f"{'min/run':>8} {f'{seeds} seeds min':>13} {'spread':>7}")
    print(head)
    for r in rows:
        waves = -(-seeds // procs)
        print(f"  {r['config']:<{w}} {r['batch_size']:>5} {r['img_per_s']:>9,.0f} {r['ms_per_iter']:>8.2f} "
              f"{r['img_per_s'] / base['img_per_s']:>5.2f}x {r['first_iter_s']:>9.1f} {r['peak_mem_gb']:>8.2f} "
              f"{r['minutes_per_run']:>8.1f} {waves * r['minutes_per_run']:>13.1f} {r['max_over_min']:>7.2f}")
    if loaders:
        print("  loader alone: " + " | ".join(f"{d['loader']} {d['img_per_s']:,.0f} img/s" for d in loaders)
              + f" (CPU workers {loaders[0]['workers']})")
    print("  min/run = epochs x iters/epoch x ms/it (no evaluation); spread = slowest / fastest timed repeat")


def main(argv=None):
    args = parse_args(argv)
    unknown = [n for n in args.names if n not in EXPERIMENTS]
    if unknown:
        sys.exit(f"unknown experiment(s) {unknown}")
    n_cpus = limit_cpus(args.max_cpus)
    workers = default_workers(n_cpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    out = {"device": dev_name, "torch": torch.__version__, "procs": args.procs, "runs": [], "loaders": {}}
    for exp in args.names:
        rows = []
        for spec in args.configs:
            cfg = parse_config(spec)
            if args.procs > 1 and not args.child:
                r = run_copies(exp, spec, args, n_cpus)
            else:
                r = measure(exp, cfg, args, device, workers)
            rows.append(r)
            out["runs"].append(r)
            if not args.child:
                print(f"[bench_train] {exp} {spec}: {r['img_per_s']:,.0f} img/s, {r['ms_per_iter']:.2f} ms/it, "
                      f"first iteration {r['first_iter_s']:.1f} s, loss {r['loss']:.3f}", flush=True)
                if r.get("profile"):
                    print(f"[profile {exp} {spec}] {args.profile} iterations\n{r['profile']}", flush=True)
        loaders = [] if (args.no_data_only or args.child) else data_only(exp, args, device, workers)
        out["loaders"][exp] = loaders
        if not args.child:
            print_table(exp, rows, loaders, dev_name, args.procs)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
