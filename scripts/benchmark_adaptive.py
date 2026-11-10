"""TODO-4: wall-clock latency and throughput of adaptive vs fixed-depth inference.

    python scripts/benchmark_adaptive.py c100_bvit_s_quadw_lr5e4 c100_vit_s_8blk --seeds 0 1 2

All runs are timed back to back in ONE process, so their numbers are comparable
(timings from different sessions on a shared GPU are not). For every run it uses
runs/<name>/model.pt, the temperatures in reports/<name>/results.json and the
saved policy reports/<name>/policies/<policy>.pt (written by halting_study.py).

Measured on the test set, preloaded on the GPU (no data-loading time):
  * fixed depth t = 0..T-1: batched throughput (batch 256) and batch-1 latency;
  * adaptive, for several prices lambda: the policy decides after each step and
    finished images are removed from the batch ("compaction"), so later steps
    run on fewer images. Batch-1 latency uses the same code with one image.
Modes: "eager" (always), "compiled" (torch.compile of the per-step model call and
of the feature + policy computation, for both fixed and adaptive; skipped with
--no-compile or if compilation fails) and "eager-naive" (--naive: the first
runner, with several host syncs per step).
Accuracy and mean steps come from the same runs. Writes reports/<name>/benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsv.cpu import limit_cpus  # noqa: E402
from fsv.data import build_splits  # noqa: E402
from fsv.engine import auto_amp  # noqa: E402
from fsv.halting.online import AdaptiveRunner  # noqa: E402
from fsv.halting.policy import HaltMLP  # noqa: E402
from fsv.models import BViTConfig, RecurrentViT  # noqa: E402
from fsv.version import code_info  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", help="experiment names")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--policy", default="S0-reward_acc")
    p.add_argument("--lams", type=float, nargs="+", default=[0.001, 0.01, 0.03, 0.1, 0.3])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bs1-images", type=int, default=500, help="test images timed one at a time")
    p.add_argument("--repeats", type=int, default=5, help="timed passes over the data; the median is reported")
    p.add_argument("--max-images", type=int, default=0, help="use only the first N test images (quick checks)")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="skip the torch.compile variant (measured by default; eager is always measured)")
    p.add_argument("--naive", action="store_true", help="also time the first runner (several syncs per step)")
    p.add_argument("--out-root", default=str(ROOT))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--fake-data", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--max-cpus", type=int, default=None)
    return p.parse_args(argv)


def timed(fn, repeats: int, sync) -> tuple[float, object]:
    out = fn()  # warm-up
    sync()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return st.median(times), out


def main(argv=None):
    args = parse_args(argv)
    limit_cpus(args.max_cpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    test = build_splits("cifar100", args.data_dir, download=True, fake=args.fake_data)["test"]
    n_img = min(args.max_images, len(test)) if args.max_images else len(test)
    xs, ys = zip(*(test[i] for i in range(n_img)))
    X, Y = torch.stack(xs).to(device), torch.tensor(ys, device=device)
    N, bs = len(Y), args.batch_size
    n1 = min(args.bs1_images, N)
    print(f"[benchmark] device {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}, "
          f"{N} test images, batch {bs}, batch-1 on {n1} images, {args.repeats} repeats")

    for name in args.names:
        for seed in args.seeds:
            run = name if seed == 0 else f"{name}_s{seed}"
            run_dir, rep = Path(args.out_root) / "runs" / run, Path(args.out_root) / "reports" / run
            pol_path = rep / "policies" / f"{args.policy}.pt"
            if not (run_dir / "model.pt").exists() or not pol_path.exists():
                print(f"[benchmark] {run}: missing model.pt or {pol_path.name}; skipped")
                continue
            ck = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
            model = RecurrentViT(BViTConfig(**ck["model_cfg"]))
            model.load_state_dict(ck["model"])
            model.to(device).eval()
            amp = ck.get("amp") if device.type == "cuda" and ck.get("amp") else auto_amp(device)
            pc = torch.load(pol_path, map_location="cpu", weights_only=False)
            policy = HaltMLP(pc["in_dim"], use_lambda=pc["use_lambda"], lam_range=tuple(pc["lam_range"]))
            policy.load_state_dict(pc["state_dict"])
            policy.to(device)
            T = model.cfg.t_max

            def batched(fn):
                return lambda: [fn(X[i:i + bs]) for i in range(0, N, bs)]

            def single(fn):
                return lambda: [fn(X[i:i + 1]) for i in range(n1)]

            lams = args.lams if policy.use_lambda else [None]

            def measure(runner, method: str) -> dict:
                fixed, adaptive = [], []
                for t in range(T):
                    sec, out = timed(batched(lambda xb, t=t: runner.run_fixed(xb, t)), args.repeats, sync)
                    acc = float((torch.cat(out) == Y).float().mean())
                    sec1, _ = timed(single(lambda xb, t=t: runner.run_fixed(xb, t)), max(1, args.repeats // 2), sync)
                    fixed.append({"step": t, "acc": acc, "img_per_s": N / sec, "bs1_ms": sec1 / n1 * 1e3})
                fn = getattr(runner, method)
                for lam in lams:
                    sec, out = timed(batched(lambda xb: fn(xb, lam)), args.repeats, sync)
                    pred = torch.cat([o[0] for o in out])
                    steps = torch.cat([o[1] for o in out]).float()
                    sec1, _ = timed(single(lambda xb: fn(xb, lam)), max(1, args.repeats // 2), sync)
                    adaptive.append({"lam": lam, "acc": float((pred == Y).float().mean()),
                                     "mean_steps": float(steps.mean()) + 1, "img_per_s": N / sec,
                                     "bs1_ms": sec1 / n1 * 1e3,
                                     "steps_hist": torch.bincount(steps.long(), minlength=T).tolist()})
                return {"fixed": fixed, "adaptive": adaptive}

            res = {"run": run, "policy": args.policy, "amp": amp, "batch_size": bs, "bs1_images": n1,
                   "code_version": code_info()["code_version"],
                   "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "modes": {}}
            eager = AdaptiveRunner(model, policy, pc["temperatures"], device, amp)
            res["modes"]["eager"] = measure(eager, "run")
            if args.naive:  # the first runner: boolean-mask indexing, several host syncs per step
                res["modes"]["eager-naive"] = measure(eager, "run_naive")
            if args.compile:
                try:
                    res["modes"]["compiled"] = measure(
                        AdaptiveRunner(model, policy, pc["temperatures"], device, amp, compile=True), "run")
                except Exception as e:  # keep the eager numbers if compilation is unavailable
                    res["compile_error"] = repr(e)[:500]
                    print(f"[benchmark] torch.compile failed, skipped: {res['compile_error']}")
            rep.mkdir(parents=True, exist_ok=True)
            (rep / "benchmark.json").write_text(json.dumps(res, indent=2))
            print(f"\n[benchmark] {run} (AMP {amp}, policy {args.policy})")
            for mode, m in res["modes"].items():
                print(f"  [{mode}] setting   acc %   steps   img/s (bs {bs})   bs1 ms")
                for r in m["fixed"]:
                    print(f"  fixed t={r['step']}      {r['acc'] * 100:6.2f}  {r['step'] + 1:6.2f}  "
                          f"{r['img_per_s']:14.0f}  {r['bs1_ms']:7.3f}")
                for r in m["adaptive"]:
                    print(f"  lam={r['lam']!s:<10}  {r['acc'] * 100:6.2f}  {r['mean_steps']:6.2f}  "
                          f"{r['img_per_s']:14.0f}  {r['bs1_ms']:7.3f}")


if __name__ == "__main__":
    main()
