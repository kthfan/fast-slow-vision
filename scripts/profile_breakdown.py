"""Where does the time go? Per-part timing of fixed-depth vs adaptive O1 inference.

    python scripts/profile_breakdown.py                         # synthetic: random bvit_s and bvit_b
    python scripts/profile_breakdown.py --run c100_bvit_s_quadw_lr5e4 --lam 0.03   # a trained run + policy

Both methods are split into the same parts and run with the same code as the runners in
``fsv/halting/online.py`` (the predictions are checked against them):

  embed        patch embedding                    model_step   one recurrent block
  readout      head at step t                     features     softmax / log / S0 features
  policy       halting MLP                        record       write finished predictions
  sync         ``nonzero`` (GPU -> CPU read)      compaction   drop finished images
  argmax       final prediction (fixed depth)

For each part it reports, per forward call (one batch of 256, or one image at batch 1):
  * cpu_us   host time spent inside the part, without the profiler (kernel launches, Python,
             and for ``sync`` the time the CPU waits for the GPU);
  * gpu_us   GPU kernel time of the part (torch.profiler, kernels attributed to the part);
  * kernels  number of GPU kernels / copies it launches.
plus wall time (no profiler), total GPU busy time and GPU idle = wall - busy. The GPU is warmed up
before timing (clocks ramp up under load); the kernel times come from a separate profiled pass,
so busy and wall can disagree by a few % when the clock drifts.

Batch 1 with CUDA graphs ("graph-bs1") cannot be profiled per kernel inside one graph, so
every part is captured as its own graph: model step (+ readout), features, policy; each
graph's GPU time comes from back-to-back replays timed with CUDA events, its CPU time is the
launch cost, and ``sync`` is the host read of the stop flag. The real (single graph per step)
O1 and fixed-depth graphs of ``CudaGraphBS1`` are timed as a reference.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsv.engine import auto_amp, autocast_ctx  # noqa: E402
from fsv.halting.online import AdaptiveRunner, CudaGraphBS1, _Replay, online_s0  # noqa: E402
from fsv.halting.policy import HaltMLP  # noqa: E402
from fsv.models.bvit import build_model  # noqa: E402

PARTS = ["embed", "model_step", "readout", "features", "policy", "record", "sync", "compaction", "argmax"]


# --------------------------------------------------------------------------- instrumentation
class Parts:
    """``with P("name"):`` adds host time to the part (and a profiler range when profiling)."""

    def __init__(self, profile: bool = False):
        self.profile = profile
        self.cpu = defaultdict(float)

    @contextmanager
    def __call__(self, name):
        rf = torch.profiler.record_function(name) if self.profile else nullcontext()
        with rf:
            t0 = time.perf_counter()
            yield
            self.cpu[name] += time.perf_counter() - t0


def o1_eager(r: AdaptiveRunner, x, lam, P):
    """``AdaptiveRunner.run`` with every part timed (same operations, same order)."""
    B, T, dev = x.shape[0], r.T, x.device
    preds = torch.zeros(B, dtype=torch.long, device=dev)
    steps = torch.full((B,), T - 1, dtype=torch.long, device=dev)
    idx = torch.arange(B, device=dev)
    with P("embed"), autocast_ctx(r.device, r.amp):
        x0 = tok = r.model.embed(x)
    lam_all = r._lam(B, lam)
    prev_logp = prev_pred = run = None
    for t in range(T):
        with autocast_ctx(r.device, r.amp):  # one autocast per step, as in AdaptiveRunner._step_impl
            with P("model_step"):
                tok = r.model.forward_step(tok, t, x0)
            with P("readout"):
                z = r.model.readout(tok, t)
        if t == T - 1:
            with P("record"):
                preds.index_copy_(0, idx, z.argmax(-1))
            break
        n = idx.shape[0]
        with P("features"):
            p = (z.float() / r.temps[t]).softmax(-1).clamp_min(1e-12)
            logp = p.log()
            feats, pred, run = online_s0(p, logp, prev_logp, prev_pred, run, t, T)
        with P("policy"):
            stop = r.policy(feats[None], None if lam_all is None else lam_all[:n])[0] > 0
        with P("record"):
            preds.index_copy_(0, idx, torch.where(stop, pred, preds.index_select(0, idx)))
            steps.index_copy_(0, idx, torch.where(stop, torch.full_like(idx, t), steps.index_select(0, idx)))
        with P("sync"):
            keep = (~stop).nonzero().squeeze(1)
            n_keep = keep.shape[0]
        if n_keep == 0:
            break
        if n_keep < n:
            with P("compaction"):
                idx, tok, x0 = idx.index_select(0, keep), tok.index_select(0, keep), x0.index_select(0, keep)
                logp, pred, run = logp.index_select(0, keep), pred.index_select(0, keep), run.index_select(0, keep)
        prev_logp, prev_pred = logp, pred
    return preds, steps


def fixed_eager(r: AdaptiveRunner, x, t_stop, P):
    """``AdaptiveRunner.run_fixed`` with every part timed."""
    with P("embed"), autocast_ctx(r.device, r.amp):
        x0 = tok = r.model.embed(x)
    for t in range(t_stop + 1):
        with autocast_ctx(r.device, r.amp):
            with P("model_step"):
                tok = r.model.forward_step(tok, t, x0)
            with P("readout"):
                z = r.model.readout(tok, t)
    with P("argmax"):
        return z.argmax(-1)


def gpu_parts(prof) -> tuple[dict, dict, float]:
    """(gpu_us per part, kernel count per part, total GPU busy us) from a profiler run."""
    from torch.autograd import DeviceType

    gpu, nk, busy = defaultdict(float), defaultdict(int), 0.0

    def kernels(e):
        return len(e.kernels) + sum(kernels(c) for c in e.cpu_children)

    for e in prof.events():
        if e.device_type != DeviceType.CPU:
            continue
        if e.name in PARTS:
            gpu[e.name] += e.device_time_total
            nk[e.name] += kernels(e)
        busy += sum(k.duration for k in e.kernels)
    return gpu, nk, busy


def breakdown_eager(fn, chunks, sync, repeats, cuda):
    """fn(chunk, P) over all chunks. Returns per-call cpu/gpu/kernels per part + wall/busy/idle."""
    run_all = lambda P: [fn(c, P) for c in chunks]  # noqa: E731
    run_all(Parts())
    sync()
    walls, cpus = [], []
    for _ in range(repeats):
        P = Parts()
        sync()
        t0 = time.perf_counter()
        run_all(P)
        sync()
        walls.append(time.perf_counter() - t0)
        cpus.append(P.cpu)
    n = len(chunks)
    wall = st.median(walls) / n * 1e6
    cpu = {k: st.median([c.get(k, 0.0) for c in cpus]) / n * 1e6 for k in PARTS}
    gpu, nk, busy = defaultdict(float), defaultdict(int), float("nan")
    if cuda:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_all(Parts(profile=True))
            sync()
        g, k, b = gpu_parts(prof)
        gpu = {p: g.get(p, 0.0) / n for p in PARTS}
        nk = {p: k.get(p, 0) / n for p in PARTS}
        busy = b / n
    rows = {p: {"cpu_us": cpu[p], "gpu_us": gpu.get(p, float("nan")) if cuda else float("nan"),
                "kernels": nk.get(p, float("nan")) if cuda else float("nan")} for p in PARTS if cpu[p] > 0}
    return {"parts": rows, "wall_us": wall, "gpu_busy_us": busy, "gpu_idle_us": wall - busy,
            "cpu_parts_sum_us": sum(cpu.values())}


# --------------------------------------------------------------------------- CUDA graphs, batch 1
class SplitGraphs:
    """The O1 step of ``CudaGraphBS1`` captured as three graphs per step: model step (+ readout),
    features, policy. Shares the static buffers of ``cg``."""

    def __init__(self, cg: CudaGraphBS1):
        self.cg = cg
        T, dev, cap = cg.T, cg.device, cg.captured
        with torch.no_grad():
            with autocast_ctx(dev, cg.amp, cache_enabled=False):
                z = cg.model.readout(cg.tok, 0)
            self.z = torch.zeros_like(z)
            self.feats = torch.zeros(1, 10 + T, device=dev)
            self.g_model = [_Replay(lambda t=t: self._model_fn(t), cap) for t in range(T)]
            self.g_feat = [_Replay(lambda t=t: self._feat_fn(t), cap) for t in range(T - 1)]
            self.g_pol = _Replay(self._pol_fn, cap)

    def _model_fn(self, t):
        cg = self.cg
        tok, z = cg._model_step(cg.tok, cg.x0, t)
        cg.tok.copy_(tok)
        self.z.copy_(z)
        if t == cg.T - 1:
            cg.pred.copy_(z.argmax(-1))

    def _feat_fn(self, t):
        cg = self.cg
        p, logp = cg._probs(self.z, t)
        prev = (cg.prev_logp, cg.prev_pred, cg.run_) if t > 0 else (None, None, None)
        feats, pred, run = online_s0(p, logp, *prev, t, cg.T)
        self.feats.copy_(feats)
        cg.prev_logp.copy_(logp)
        cg.prev_pred.copy_(pred)
        cg.run_.copy_(run)
        cg.pred.copy_(pred)

    def _pol_fn(self):
        cg = self.cg
        cg.stop.copy_(cg.policy(self.feats[None], cg.lam if cg.policy.use_lambda else None)[0] > 0)

    @torch.no_grad()
    def run(self, x, lam, P):
        cg = self.cg
        cg.x.copy_(x)
        if lam is not None:
            cg.lam.fill_(float(lam))
        with P("embed"):
            cg.g_embed()
        for t in range(cg.T):
            with P("model_step"):
                self.g_model[t]()
            if t == cg.T - 1:
                break
            with P("features"):
                self.g_feat[t]()
            with P("policy"):
                self.g_pol()
            with P("sync"):
                if bool(cg.stop.item()):
                    return cg.pred.clone(), t
        return cg.pred.clone(), cg.T - 1


def warm_gpu(fn, seconds: float, sync):
    """Run fn until `seconds` have passed, so the GPU clock has ramped up before timing."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        sync()


def graph_gpu_us(g, n=200, cuda=True, rounds=3) -> tuple[float, float]:
    """(GPU us, CPU launch us) per replay of one graph: n back-to-back replays, min over rounds."""
    if not cuda:
        return float("nan"), float("nan")
    gpu, cpu = [], []
    for _ in range(rounds):
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        t0 = time.perf_counter()
        for _ in range(n):
            g()
        t1 = time.perf_counter()
        e1.record()
        torch.cuda.synchronize()
        gpu.append(e0.elapsed_time(e1) * 1e3 / n)
        cpu.append((t1 - t0) * 1e6 / n)
    return min(gpu), min(cpu)


def breakdown_graph(cg: CudaGraphBS1, sg: SplitGraphs, X, lam, s_ref, fixed_ts, sync, repeats, cuda):
    n1, T = X.shape[0], cg.T
    single = lambda fn: (lambda: [fn(X[i:i + 1]) for i in range(n1)])  # noqa: E731

    def wall(fn):
        fn()
        ts = []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            out = fn()
            sync()
            ts.append(time.perf_counter() - t0)
        return st.median(ts) / n1 * 1e6, out

    warm_gpu(lambda: [gr() for gr in cg.g_fixed], 2.0 if cuda else 0.0, sync)
    # GPU time and launch cost of every graph (state of the buffers does not matter for timing)
    g = {"embed": graph_gpu_us(cg.g_embed, cuda=cuda)}
    per_t = {k: [graph_gpu_us(gr, cuda=cuda) for gr in lst] for k, lst in
             (("model_step", sg.g_model), ("features", sg.g_feat), ("o1_step_graph", cg.g_step),
              ("fixed_graph", cg.g_fixed))}
    g["policy"] = graph_gpu_us(sg.g_pol, cuda=cuda)

    # split O1 run: host time per part, and how many times each part ran per image
    walls, cpus = [], []
    with torch.no_grad():
        for _ in range(repeats):
            P = Parts()
            sync()
            t0 = time.perf_counter()
            out = [sg.run(X[i:i + 1], lam, P) for i in range(n1)]
            sync()
            walls.append(time.perf_counter() - t0)
            cpus.append(P.cpu)
    agree = sum(int(o[1]) == int(s) for o, s in zip(out, s_ref)) / n1
    steps = [int(s) for s in s_ref]  # stop step per image
    n_model = sum(s + 1 for s in steps) / n1
    n_dec = sum(min(s, T - 2) + 1 for s in steps) / n1  # features/policy/sync run at steps 0..min(s, T-2)
    mean_model = lambda k: st.mean(v[0] for v in per_t[k])  # noqa: E731
    gpu = {"embed": g["embed"][0], "model_step": mean_model("model_step") * n_model,
           "features": st.mean(v[0] for v in per_t["features"]) * n_dec, "policy": g["policy"][0] * n_dec,
           "sync": 0.0}
    cpu = {k: st.median([c.get(k, 0.0) for c in cpus]) / n1 * 1e6 for k in gpu}
    w = st.median(walls) / n1 * 1e6
    busy = sum(gpu.values())
    o1 = {"parts": {k: {"cpu_us": cpu[k], "gpu_us": gpu[k]} for k in gpu}, "wall_us": w, "gpu_busy_us": busy,
          "gpu_idle_us": w - busy, "agree_with_eager": agree, "model_steps_per_image": n_model,
          "decisions_per_image": n_dec,
          "per_graph_us": {"embed": g["embed"], "policy": g["policy"],
                           **{k: [round(a, 2) for a, _ in v] for k, v in per_t.items()}}}
    o1["reference_wall_us"] = {"o1_one_graph_per_step": wall(single(lambda xb: cg.run(xb, lam)))[0],
                               **{f"fixed_t{t}": wall(single(lambda xb, t=t: cg.run_fixed(xb, t)))[0]
                                  for t in fixed_ts}}
    o1["reference_gpu_us"] = {
        "o1_one_graph_per_step": g["embed"][0] + st.mean(sum(per_t["o1_step_graph"][t][0] for t in range(s + 1))
                                                         for s in steps),
        **{f"fixed_t{t}": per_t["fixed_graph"][t][0] for t in fixed_ts}}
    o1["launch_us"] = {"embed": g["embed"][1], "policy": g["policy"][1],
                       **{k: st.mean(v[1] for v in per_t[k]) for k in per_t}}
    return o1


# --------------------------------------------------------------------------- setup and report
def setup(a, dev):
    if a.run:
        from fsv.data import build_splits
        from fsv.models import BViTConfig, RecurrentViT

        run_dir, rep = ROOT / "runs" / a.run, ROOT / "reports" / a.run
        ck = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
        model = RecurrentViT(BViTConfig(**ck["model_cfg"]))
        model.load_state_dict(ck["model"])
        amp = ck.get("amp") if dev.type == "cuda" and ck.get("amp") else auto_amp(dev)
        pc = torch.load(rep / "policies" / f"{a.policy}.pt", map_location="cpu", weights_only=False)
        policy = HaltMLP(pc["in_dim"], use_lambda=pc["use_lambda"], lam_range=tuple(pc["lam_range"]))
        policy.load_state_dict(pc["state_dict"])
        test = build_splits("cifar100", str(ROOT / "data"), download=True)["test"]
        n = min(a.images, len(test))
        X = torch.stack([test[i][0] for i in range(n)])
        lam = a.lam if policy.use_lambda else None
        return model.to(dev).eval(), policy.to(dev).eval(), pc["temperatures"], amp, X.to(dev), lam, a.run
    torch.manual_seed(0)
    model = build_model(a.model, num_classes=a.classes, t_max=a.t_max).to(dev).eval()
    T = model.cfg.t_max
    policy = HaltMLP(10 + T).to(dev).eval()
    temps = torch.ones(T)
    X = torch.randn(a.images, 3, 32, 32, device=dev)
    r = AdaptiveRunner(model, policy, temps, dev, auto_amp(dev))
    lo, hi = -20.0, 20.0  # tune the output bias so the mean number of steps is about a.steps
    for _ in range(25):
        mid = (lo + hi) / 2
        policy.net[-1].bias.data.fill_(mid)
        m = float(r.run(X[:1024])[1].float().mean()) + 1
        lo, hi = (lo, mid) if m < a.steps else (mid, hi)
    policy.net[-1].bias.data.fill_((lo + hi) / 2)
    return model, policy, temps, auto_amp(dev), X, None, f"{a.model} (random weights)"


def print_table(title, res):
    print(f"\n  [{title}]  wall {res['wall_us']:.1f} us | GPU busy {res['gpu_busy_us']:.1f} us | "
          f"GPU idle {res['gpu_idle_us']:.1f} us ({res['gpu_idle_us'] / res['wall_us'] * 100:.0f}% of wall)")
    print(f"    {'part':<12}{'cpu_us':>10}{'gpu_us':>10}{'kernels':>9}")
    for p, r in res["parts"].items():
        print(f"    {p:<12}{r['cpu_us']:10.1f}{r['gpu_us']:10.1f}{r.get('kernels', float('nan')):9.1f}")
    if "cpu_parts_sum_us" in res:
        print(f"    {'(rest)':<12}{res['wall_us'] - res['cpu_parts_sum_us']:10.1f}"
              "   <- host time outside the parts: Python loop, and waiting for the GPU at the end")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="", help="trained run under runs/ (default: synthetic random model)")
    p.add_argument("--policy", default="S0-reward_acc")
    p.add_argument("--lam", type=float, default=0.03)
    p.add_argument("--model", nargs="+", default=["bvit_s", "bvit_b"], help="synthetic models")
    p.add_argument("--t-max", type=int, default=8)
    p.add_argument("--classes", type=int, default=100)
    p.add_argument("--steps", type=float, default=3.5, help="synthetic: target mean number of steps")
    p.add_argument("--fixed-t", type=int, nargs="*", default=None,
                   help="fixed depths to compare (default: the two around the O1 mean)")
    p.add_argument("--images", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--bs1-images", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    all_res = []
    for model_name in ([a.run] if a.run else a.model):
        a.model = model_name
        model, policy, temps, amp, X, lam, label = setup(a, dev)
        T = model.cfg.t_max
        r = AdaptiveRunner(model, policy, temps, dev, amp)
        with torch.no_grad():
            _, s_all = r.run(X, lam)
        mean_steps = float(s_all.float().mean()) + 1
        fixed_ts = a.fixed_t if a.fixed_t is not None else sorted(
            {max(0, int(mean_steps) - 1), min(T - 1, int(mean_steps))})
        res = {"model": label, "device": torch.cuda.get_device_name() if cuda else "cpu", "amp": amp, "T": T,
               "lam": lam, "o1_mean_steps_bs256": mean_steps, "fixed_ts": fixed_ts, "modes": {}}
        print(f"\n===== {label} | {res['device']} | AMP {amp} | O1 mean steps {mean_steps:.2f} "
              f"| fixed depths {[t + 1 for t in fixed_ts]} steps =====")
        bs, n1 = a.batch_size, min(a.bs1_images, X.shape[0])
        chunks = {"eager-bs256": [X[i:i + bs] for i in range(0, X.shape[0], bs)],
                  "eager-bs1": [X[i:i + 1] for i in range(n1)]}
        with torch.no_grad():
            warm_gpu(lambda: r.run_fixed(X[:bs], T - 1), 3.0 if cuda else 0.0, sync)
            for mode, ch in chunks.items():
                ref = [r.run(c, lam) for c in ch]
                o1 = breakdown_eager(lambda c, P: o1_eager(r, c, lam, P), ch, sync, a.repeats, cuda)
                o1["agree_with_runner"] = float(torch.cat([o1_eager(r, c, lam, Parts())[1] for c in ch]).eq(
                    torch.cat([x[1] for x in ref])).float().mean())
                res["modes"][f"{mode}/O1"] = o1
                print_table(f"{mode} O1, lam={lam}, mean steps "
                            f"{float(torch.cat([x[1] for x in ref]).float().mean()) + 1:.2f}", o1)
                for t in fixed_ts:
                    fx = breakdown_eager(lambda c, P, t=t: fixed_eager(r, c, t, P), ch, sync, a.repeats, cuda)
                    res["modes"][f"{mode}/fixed_t{t}"] = fx
                    print_table(f"{mode} fixed {t + 1} steps", fx)
            cg = CudaGraphBS1(model, temps, dev, amp, policy=policy)
            sg = SplitGraphs(cg)
            Xs = X[:n1]
            s_ref = [int(r.run(Xs[i:i + 1], lam)[1]) for i in range(n1)]
            gr = breakdown_graph(cg, sg, Xs, lam, s_ref, fixed_ts, sync, a.repeats, cuda)
            res["modes"]["graph-bs1/O1"] = gr
            print_table(f"graph-bs1 O1 split into part graphs, mean steps {gr['model_steps_per_image']:.2f}, "
                        f"decisions agree {gr['agree_with_eager'] * 100:.0f}%", gr)
            print("    reference wall us/img (real graphs): "
                  + ", ".join(f"{k} {v:.1f}" for k, v in gr["reference_wall_us"].items()))
            print("    reference GPU us/img (real graphs):  "
                  + ", ".join(f"{k} {v:.1f}" for k, v in gr["reference_gpu_us"].items()))
            pg = gr["per_graph_us"]
            print(f"    per graph GPU us: embed {pg['embed'][0]:.1f}, model step {pg['model_step']}, "
                  f"features {pg['features']}, policy {pg['policy'][0]:.1f}, "
                  f"O1 step graph {pg['o1_step_graph']}")
            print("    per graph launch us (CPU): " + ", ".join(f"{k} {v:.1f}" for k, v in gr["launch_us"].items()))
        all_res.append(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(all_res, indent=2))


if __name__ == "__main__":
    main()
