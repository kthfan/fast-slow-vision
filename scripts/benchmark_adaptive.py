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
CUDA-graph modes ("cudagraph-bs1", "o3k1-cudagraph-bs1"; skip with --no-cudagraph) time batch 1
with every per-step computation captured as a graph: fixed depth = one graph for the whole
forward pass, O1 = one graph per step + one host read of the stop flag, O3 = two graphs +
one host read. They report latency only (accuracy is taken from the eager modes).
"eager-triton" / "cudagraph-bs1-triton" compute each step's features + policy in ONE Triton kernel
(fsv/halting/fused.py; GPU only).
Slot-pool modes ("slots", "slots-triton"; --slots) keep --slots images in flight, each at its own step,
and run one CUDA graph per tick over the whole test set (throughput only; fsv/halting/slots.py).
Batch CUDA-graph modes (0.16.0; skip with --no-graph-batch), the fair batch-256 comparison:
  "graph-bs<B>":          fixed depth = embed + steps 0..t as ONE graph per batch of B (FixedDepthGraph),
                          vs the slot pool (+ fused kernel when Triton is available); eager kernels in both;
  "graph-compiled-bs<B>": the same with the model step compiled by Inductor before capture, in both.
Each mode carries its own fixed-depth rows, so the speed-ups printed for it compare like with like.
O3 modes ("o3k0-eager", "o3k1-compiled", ...) time step predictors that decide every
image's depth once after step k (one host sync per batch); they need the O3 policy
files written by ``halting_study.py --o3-ks``.
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
from fsv.halting.online import AdaptiveRunner, CudaGraphBS1, O3Runner  # noqa: E402
from fsv.halting.policy import HaltMLP, StepMLP  # noqa: E402
from fsv.models import load_checkpoint_model  # noqa: E402
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
    p.add_argument("--slots", type=int, default=256,
                   help="slot-pool modes (\"slots\", \"slots-triton\"): images in flight; 0 skips them")
    p.add_argument("--refill", type=int, default=None,
                   help="slot pool: refill cap per tick (default S/2 for bViT, S for VCNet)")
    p.add_argument("--graph-only", action="store_true",
                   help="only the mode graph-compiled-bs<B> (fixed depth as one compiled CUDA graph vs the compiled "
                        "slot pool): the fair batch comparison, without the slower eager / compaction modes")
    p.add_argument("--no-graph-batch", dest="graph_batch", action="store_false",
                   help="skip the modes graph-bs<B> / graph-compiled-bs<B> (fixed depth as one CUDA graph per batch "
                        "vs the slot pool)")
    p.add_argument("--no-cudagraph", dest="cudagraph", action="store_false",
                   help="skip the batch-1 CUDA-graph modes (measured by default)")
    p.add_argument("--o3-policies", nargs="*", default=["S0-o3k0-reward_acc", "S0-o3k1-reward_acc"],
                   help="O3 step predictors to time (decide the depth once; skipped if the file is missing)")
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


def save_and_print(res, rep, run, args, bs) -> None:
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "benchmark.json").write_text(json.dumps(res, indent=2))
    print(f"\n[benchmark] {run} (AMP {res['amp']}, policy {args.policy})")
    for mode, m in res["modes"].items():
        print(f"  [{mode}] setting   acc %   steps   img/s (bs {bs})   bs1 ms")
        for r in m["fixed"]:
            print(f"  fixed t={r['step']}      {r['acc'] * 100:6.2f}  {r['step'] + 1:6.2f}  "
                  f"{(r['img_per_s'] or float('nan')):14.0f}  {(r['bs1_ms'] or float('nan')):7.3f}")
        for r in m["adaptive"]:
            print(f"  lam={r['lam']!s:<10}  {r['acc'] * 100:6.2f}  {r['mean_steps']:6.2f}  "
                  f"{(r['img_per_s'] or float('nan')):14.0f}  {(r['bs1_ms'] or float('nan')):7.3f}"
                  + (f"   V1 {r['embed_ms']:.1f} ms, tick {r['tick_ms']:.3f} ms x {r['ticks']}" if r.get("tick_ms") else ""))


def slot_rows(sp, X, Y, N, lams, T, args, sync) -> list[dict]:
    rows = []
    for lam in lams:
        sec, (pred, steps) = timed(lambda: sp.run(X, lam), args.repeats, sync)
        rows.append({"lam": lam, "acc": float((pred == Y).float().mean()),
                     "mean_steps": float(steps.float().mean()) + 1, "img_per_s": N / sec,
                     "bs1_ms": None, "ticks": sp.last_ticks, "occupancy": sp.occupancy(steps),
                     "embed_ms": sp.last_embed_ms, "tick_ms": sp.last_tick_ms / max(1, sp.last_ticks),
                     "steps_hist": torch.bincount(steps, minlength=T).tolist()})
    return rows


def measure_graph_batch(res, model, policy, pc, device, amp, T, X, Y, N, sync, args, lams):
    """Modes graph-bs<B> / graph-compiled-bs<B>: fixed depth as one CUDA graph per batch vs the slot pool."""
    from fsv.halting.fused import triton_usable
    from fsv.halting.online import FixedDepthGraph
    from fsv.halting.slots import SlotPoolRunner, supports_slots

    bs = args.batch_size
    for tag, comp in ((f"graph-bs{bs}", False), (f"graph-compiled-bs{bs}", True)):
        if (comp and not args.compile) or (args.graph_only and not comp):
            continue
        try:
            fg = FixedDepthGraph(model, device, amp, batch=bs, compile=comp)
            fixed = []
            for t in range(T):
                sec, pred = timed(lambda t=t: fg.run_fixed(X, t), args.repeats, sync)
                fixed.append({"step": t, "acc": float((pred == Y).float().mean()), "img_per_s": N / sec,
                              "bs1_ms": None})
            del fg
            mode = {"fixed": fixed, "adaptive": []}
            if policy is not None and args.slots and supports_slots(model):
                fz = triton_usable(device)
                sp = SlotPoolRunner(model, policy, pc["temperatures"], device, amp, slots=args.slots,
                                    refill=args.refill, fused=fz, compile=comp)
                mode["adaptive"] = slot_rows(sp, X, Y, N, lams, T, args, sync)
                mode.update({"slots": args.slots, "refill": sp.R, "fused": fz})
                del sp
            res["modes"][tag] = mode
        except Exception as e:
            res[f"{tag}_error"] = repr(e)[:500]
            print(f"[benchmark] {tag} failed, skipped: {res[f'{tag}_error']}")
        if device.type == "cuda":
            torch.cuda.empty_cache()


def measure_cudagraph(res, model, policy, pc, o3_loaded, device, amp, T, X, n1, single, sync, args, lams):
    """Modes "cudagraph-bs1" (fixed depth + O1) and "<o3k>-cudagraph-bs1": batch-1 latency only.
    Accuracy and mean steps are copied from the eager modes (full test set, same decisions);
    ``agree_with_eager`` checks the decisions on the timed images."""
    reps = max(1, args.repeats // 2)
    cg = CudaGraphBS1(model, pc["temperatures"], device, amp, policy=policy)
    res["cudagraph_captured"] = cg.captured
    eager = AdaptiveRunner(model, policy, pc["temperatures"], device, amp)
    fixed = []
    for t in range(T):
        sec1, _ = timed(single(lambda xb, t=t: cg.run_fixed(xb, t)), reps, sync)
        ref = res["modes"]["eager"]["fixed"][t]
        fixed.append({"step": t, "acc": ref["acc"], "img_per_s": None, "bs1_ms": sec1 / n1 * 1e3})

    def adaptive_rows(fn, ref_rows, ref_runner, lam_list):
        rows = []
        for i, lam in enumerate(lam_list):
            sec1, out = timed(single(lambda xb: fn(xb, lam)), reps, sync)
            pred = torch.cat([o[0] for o in out])
            # same batch size (1) as the graph, so the same kernels: decisions should match exactly
            ref_pred = torch.cat([ref_runner.run(X[i:i + 1], lam)[0] for i in range(n1)])
            rows.append({"lam": lam, "acc": ref_rows[i]["acc"], "mean_steps": ref_rows[i]["mean_steps"],
                         "img_per_s": None, "bs1_ms": sec1 / n1 * 1e3,
                         "agree_with_eager": float((pred == ref_pred).float().mean())})
        return rows

    res["modes"]["cudagraph-bs1"] = {"fixed": fixed, "adaptive": adaptive_rows(
        cg.run, res["modes"]["eager"]["adaptive"], eager, lams)}
    from fsv.halting.fused import triton_usable

    if policy is not None and triton_usable(device):  # features + policy as ONE Triton kernel in each step graph
        try:
            cgt = CudaGraphBS1(model, pc["temperatures"], device, amp, policy=policy, fused=True)
            res["modes"]["cudagraph-bs1-triton"] = {"fixed": fixed, "adaptive": adaptive_rows(
                cgt.run, res["modes"]["eager"]["adaptive"], eager, lams)}
        except Exception as e:
            res["cudagraph_triton_error"] = repr(e)[:500]
            print(f"[benchmark] cudagraph-bs1-triton failed, skipped: {res['cudagraph_triton_error']}")
    for tag, (sp, oc) in o3_loaded.items():
        cg3 = CudaGraphBS1(model, oc["temperatures"], device, amp, o3_policy=sp)
        cg3.g_fixed = cg.g_fixed  # same fixed-depth graphs
        res["modes"][f"{tag}-cudagraph-bs1"] = {"fixed": fixed, "adaptive": adaptive_rows(
            cg3.run_o3, res["modes"][f"{tag}-eager"]["adaptive"], O3Runner(model, sp, oc["temperatures"], device, amp),
            args.lams)}


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
            if not (run_dir / "model.pt").exists():
                print(f"[benchmark] {run}: missing model.pt; skipped")
                continue
            if pol_path.exists():
                pc = torch.load(pol_path, map_location="cpu", weights_only=False)
                policy = HaltMLP(pc["in_dim"], use_lambda=pc["use_lambda"], lam_range=tuple(pc["lam_range"]))
                policy.load_state_dict(pc["state_dict"])
                policy.to(device)
            elif (rep / "results.json").exists():  # no halting policy (e.g. the original VCNet): fixed depths only
                print(f"[benchmark] {run}: no {pol_path.name}; timing fixed depths only")
                pc = {"temperatures": torch.tensor(json.loads((rep / "results.json").read_text())["temperatures"])}
                policy = None
            else:
                print(f"[benchmark] {run}: missing {pol_path.name} and results.json; skipped")
                continue
            ck = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
            model = load_checkpoint_model(ck).to(device).eval()
            amp = ck.get("amp") if device.type == "cuda" and ck.get("amp") else auto_amp(device)
            T = model.cfg.t_max

            def batched(fn):
                return lambda: [fn(X[i:i + bs]) for i in range(0, N, bs)]

            def single(fn):
                return lambda: [fn(X[i:i + 1]) for i in range(n1)]

            lams = [] if policy is None else args.lams if policy.use_lambda else [None]

            def measure(runner, method: str, fixed=None) -> dict:
                adaptive = []
                for t in range(T) if fixed is None else []:
                    sec, out = timed(batched(lambda xb, t=t: runner.run_fixed(xb, t)), args.repeats, sync)
                    acc = float((torch.cat(out) == Y).float().mean())
                    sec1, _ = timed(single(lambda xb, t=t: runner.run_fixed(xb, t)), max(1, args.repeats // 2), sync)
                    fixed = (fixed or []) + [{"step": t, "acc": acc, "img_per_s": N / sec, "bs1_ms": sec1 / n1 * 1e3}]
                fn = method if callable(method) else runner.run if method == "o3" else getattr(runner, method)
                for lam in (lams if method != "o3" else args.lams):
                    sec, out = timed(batched(lambda xb: fn(xb, lam)), args.repeats, sync)
                    pred = torch.cat([o[0] for o in out])
                    steps = torch.cat([o[1] for o in out]).float()
                    if method == "o3":  # steps <= k reuse readouts that were computed anyway
                        steps = steps.clamp(min=runner.k)
                    sec1, _ = timed(single(lambda xb: fn(xb, lam)), max(1, args.repeats // 2), sync)
                    adaptive.append({"lam": lam, "acc": float((pred == Y).float().mean()),
                                     "mean_steps": float(steps.mean()) + 1, "img_per_s": N / sec,
                                     "bs1_ms": sec1 / n1 * 1e3,
                                     "steps_hist": torch.bincount(steps.long(), minlength=T).tolist()})
                return {"fixed": fixed, "adaptive": adaptive}

            res = {"run": run, "policy": args.policy if policy is not None else None, "amp": amp, "batch_size": bs,
                   "bs1_images": n1,
                   "code_version": code_info()["code_version"],
                   "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu", "modes": {}}
            if args.graph_only:
                measure_graph_batch(res, model, policy, pc, device, amp, T, X, Y, N, sync, args, lams)
                save_and_print(res, rep, run, args, bs)
                continue
            eager = AdaptiveRunner(model, policy, pc["temperatures"], device, amp)
            res["modes"]["eager"] = measure(eager, "run")
            from fsv.halting.fused import triton_usable

            if policy is not None and triton_usable(device):  # features + policy as ONE Triton kernel per step
                try:
                    tri = AdaptiveRunner(model, policy, pc["temperatures"], device, amp, fused=True)
                    res["modes"]["eager-triton"] = measure(tri, "run", fixed=res["modes"]["eager"]["fixed"])
                except Exception as e:
                    res["triton_error"] = repr(e)[:500]
                    print(f"[benchmark] eager-triton failed, skipped: {res['triton_error']}")
            from fsv.halting.slots import SlotPoolRunner, supports_slots

            if args.slots and policy is not None and supports_slots(model):  # slot pool: one CUDA graph per tick
                variants = [("slots", False)] + ([("slots-triton", True)] if triton_usable(device) else [])
                for tag, fz in variants:
                    try:
                        sp = SlotPoolRunner(model, policy, pc["temperatures"], device, amp, slots=args.slots,
                                            refill=args.refill, fused=fz)
                        rows = slot_rows(sp, X, Y, N, lams, T, args, sync)
                        res["modes"][tag] = {"fixed": res["modes"]["eager"]["fixed"], "adaptive": rows,
                                             "slots": args.slots, "refill": sp.R}
                    except Exception as e:
                        res[f"{tag}_error"] = repr(e)[:500]
                        print(f"[benchmark] {tag} failed, skipped: {res[f'{tag}_error']}")
            if args.graph_batch:  # fair batch comparison: fixed depth as one CUDA graph vs the slot pool
                measure_graph_batch(res, model, policy, pc, device, amp, T, X, Y, N, sync, args, lams)
            if args.naive:  # the first runner: boolean-mask indexing, several host syncs per step
                res["modes"]["eager-naive"] = measure(eager, "run_naive")
            compiled = None
            if args.compile:
                try:
                    compiled = AdaptiveRunner(model, policy, pc["temperatures"], device, amp, compile=True)
                    res["modes"]["compiled"] = measure(compiled, "run")
                except Exception as e:  # keep the eager numbers if compilation is unavailable
                    compiled = None
                    res["compile_error"] = repr(e)[:500]
                    print(f"[benchmark] torch.compile failed, skipped: {res['compile_error']}")
            o3_loaded = {}
            for o3_name in (args.o3_policies if policy is not None else []):  # decide the depth once: one sync per batch
                o3_path = rep / "policies" / f"{o3_name}.pt"
                if not o3_path.exists():
                    continue
                oc = torch.load(o3_path, map_location="cpu", weights_only=False)
                if oc.get("feature_set") != "S0":
                    print(f"[benchmark] {o3_name}: only S0 O3 policies are supported online; skipped")
                    continue
                sp = StepMLP(oc["step_dim"], oc["T"], oc["k"], lam_range=tuple(oc["lam_range"]))
                sp.load_state_dict(oc["state_dict"])
                sp.to(device)
                tag = o3_name.split("-")[1]  # e.g. o3k1
                o3_loaded[tag] = (sp, oc)
                res["modes"][f"{tag}-eager"] = measure(O3Runner(model, sp, oc["temperatures"], device, amp), "o3",
                                                       fixed=res["modes"]["eager"]["fixed"])
                if compiled is not None:
                    res["modes"][f"{tag}-compiled"] = measure(
                        O3Runner(model, sp, oc["temperatures"], device, amp, step_fn=compiled._step), "o3",
                        fixed=res["modes"]["compiled"]["fixed"])
            if args.cudagraph:  # batch 1 with CUDA graphs: launch overhead removed for every method
                try:
                    measure_cudagraph(res, model, policy, pc, o3_loaded, device, amp, T, X, n1, single, sync, args,
                                      lams)
                except Exception as e:
                    res["cudagraph_error"] = repr(e)[:500]
                    print(f"[benchmark] CUDA-graph modes failed, skipped: {res['cudagraph_error']}")
            save_and_print(res, rep, run, args, bs)


if __name__ == "__main__":
    main()
