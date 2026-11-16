"""Synthetic timing of the per-step CPU-GPU synchronisation (no checkpoint needed).

    python scripts/bench_sync.py --model bvit_s --steps 3.5

Random weights and a random S0 halting policy whose bias is tuned so the mean number of
steps is about --steps. Times, back to back in one process:
  * fixed depth (the number of steps closest to the adaptive mean, and one more);
  * O1 with one wait per step (``AdaptiveRunner.run``, ``CudaGraphBS1.run``);
  * O1 with features + policy fused into one Triton kernel;
  * O1 "no sync" bound: the step graphs replayed for a known depth (what sync costs);
  * O1 slot pool (``SlotPoolRunner``): --slots images in flight, each at its own step, one CUDA
    graph per tick (batch-256 column: throughput over all images; occupancy = useful slot-steps).
Batch 256 throughput (eager) and batch-1 latency (eager and CUDA graphs). Accuracy is
meaningless here; only the timing and the agreement of the decisions are reported.
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

from fsv.engine import auto_amp  # noqa: E402
from fsv.halting.online import AdaptiveRunner, CudaGraphBS1  # noqa: E402
from fsv.halting.policy import HaltMLP  # noqa: E402
from fsv.models.bvit import build_model  # noqa: E402


def timed(fn, repeats, sync):
    fn()
    sync()
    ts = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return st.median(ts), out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bvit_s")
    p.add_argument("--t-max", type=int, default=8)
    p.add_argument("--classes", type=int, default=100)
    p.add_argument("--steps", type=float, default=3.5, help="target mean number of steps of the policy")
    p.add_argument("--images", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bs1-images", type=int, default=300)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--slots", type=int, nargs="+", default=[256])
    p.add_argument("--refill", type=int, nargs="*", default=[0], help="slot pool refill cap per tick (0 = slots)")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    torch.manual_seed(0)
    T = a.t_max
    model = build_model(a.model, num_classes=a.classes, t_max=T).to(dev).eval()
    amp = auto_amp(dev)
    temps = torch.ones(T)
    policy = HaltMLP(10 + T).to(dev).eval()
    X = torch.randn(a.images, 3, 32, 32, device=dev)
    runner = AdaptiveRunner(model, policy, temps, dev, amp)

    lo, hi = -20.0, 20.0  # tune the output bias: higher bias -> earlier stop
    for _ in range(25):
        mid = (lo + hi) / 2
        policy.net[-1].bias.data.fill_(mid)
        m = float(runner.run(X[:1024])[1].float().mean()) + 1
        lo, hi = (lo, mid) if m < a.steps else (mid, hi)
    policy.net[-1].bias.data.fill_((lo + hi) / 2)
    _, s_ref = runner.run(X)
    mean_steps = float(s_ref.float().mean()) + 1
    res = {"device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "model": a.model,
           "amp": amp, "T": T, "mean_steps": mean_steps, "rows": []}
    print(f"[bench_sync] {res['device']} | {a.model} T={T} AMP {amp} | policy mean steps {mean_steps:.2f}")

    bs, N, n1 = a.batch_size, a.images, a.bs1_images
    batched = lambda fn: (lambda: [fn(X[i:i + bs]) for i in range(0, N, bs)])  # noqa: E731
    single = lambda fn: (lambda: [fn(X[i:i + 1]) for i in range(n1)])  # noqa: E731
    cg = CudaGraphBS1(model, temps, dev, amp, policy=policy)
    ref1 = [runner.run(X[i:i + 1]) for i in range(n1)]
    s1 = [int(r[1]) for r in ref1]
    from fsv.halting.fused import triton_usable

    rft = AdaptiveRunner(model, policy, temps, dev, amp, fused=True) if triton_usable(dev) else None
    cgt = CudaGraphBS1(model, temps, dev, amp, policy=policy, fused=True) if rft is not None else None

    chunks = [X[i:i + bs] for i in range(0, N, bs)]

    def row(name, f_b=None, f_1=None, f_g=None, steps=None, check=None, indexed=False, f_all=None):
        r = {"setting": name, "steps": steps}
        if f_b or f_all:  # f_all: one call over all batches (e.g. several batches in flight)
            sec, out = timed((lambda: f_all(chunks)) if f_all else batched(f_b), a.repeats, sync)
            r["img_per_s"] = N / sec
            if check:
                r["agree_bs256"] = float((torch.cat([o[1] for o in out]) == s_ref).float().mean())
        if f_1:
            r["bs1_ms"] = timed(single(f_1), max(1, a.repeats // 2), sync)[0] / n1 * 1e3
        if f_g:
            g = (lambda: [f_g(X[i:i + 1], i) for i in range(n1)]) if indexed else single(f_g)
            sec, out = timed(g, max(1, a.repeats // 2), sync)
            r["cg_bs1_ms"] = sec / n1 * 1e3
            if check:
                r["agree_cg"] = sum(int(o[1]) == int(rf[1]) and int(o[0]) == int(rf[0])
                                    for o, rf in zip(out, ref1)) / n1
        res["rows"].append(r)
        print(f"  {name:<24} steps {r['steps'] or 0:5.2f}  bs256 {r.get('img_per_s', float('nan')):9.0f} img/s  "
              f"bs1 eager {r.get('bs1_ms', float('nan')):6.3f} ms  bs1 graph {r.get('cg_bs1_ms', float('nan')):6.3f} ms"
              + (f"  agree {r.get('agree_bs256', 1):.3f}/{r.get('agree_cg', 1):.3f}" if check else ""))

    with torch.no_grad():
        for t in sorted({max(0, round(mean_steps) - 1), min(T - 1, round(mean_steps))}):
            row(f"fixed t={t}", lambda xb, t=t: runner.run_fixed(xb, t), lambda xb, t=t: runner.run_fixed(xb, t),
                lambda xb, t=t: cg.run_fixed(xb, t), steps=t + 1)
        row("O1 sync per step", runner.run, runner.run, cg.run, steps=mean_steps, check=True)
        from fsv.halting.slots import SlotPoolRunner

        for S in a.slots:  # continuous batching: fixed shapes, one graph per tick
            for R in a.refill:
                for fz in (False, True) if rft is not None else (False,):
                    sp = SlotPoolRunner(model, policy, temps, dev, amp, slots=S, refill=R or None, fused=fz)
                    name = f"slot pool S={S}" + (f" R={R}" if R else "") + (" triton" if fz else "")
                    row(name, f_all=lambda c, sp=sp: [sp.run(X, None)], steps=mean_steps, check=True)
                    r_ = res["rows"][-1]
                    r_["ticks"], r_["occupancy"] = sp.last_ticks, sp.occupancy(sp.run(X, None)[1])
                    if dev.type == "cuda":  # GPU time of one tick -> GPU idle of the run
                        sync()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for _ in range(50):
                            sp.g_tick()
                        e1.record()
                        sync()
                        r_["tick_gpu_ms"] = e0.elapsed_time(e1) / 50
                        wall_ms = N / r_["img_per_s"] * 1e3
                        r_["gpu_idle"] = 1 - r_["tick_gpu_ms"] * r_["ticks"] / wall_ms
                    print(f"    ticks {r_['ticks']}, occupancy {r_['occupancy']:.3f}, tick GPU "
                          f"{r_.get('tick_gpu_ms', float('nan')):.3f} ms, GPU idle {r_.get('gpu_idle', float('nan')):.1%}")
        row("O1 graphs, no sync (bound)", f_g=lambda xb, i: cg.run_known_depth(xb, s1[i]), steps=mean_steps,
            indexed=True)
        if cgt is not None:  # features + policy in ONE Triton kernel
            row("O1 triton-fused sync", rft.run, rft.run, cgt.run, steps=mean_steps, check=True)
            row("O1 triton-fused bound", f_g=lambda xb, i: cgt.run_known_depth(xb, s1[i]), steps=mean_steps,
                indexed=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
