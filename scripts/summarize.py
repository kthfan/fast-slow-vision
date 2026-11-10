"""Print one compact, paste-friendly summary of finished runs.

    python scripts/summarize.py                      # every run under reports/
    python scripts/summarize.py c100_bvit_s c100_bvit_b   # only these experiments (all seeds)

Runs of the same experiment with different seeds are aggregated as
mean ± std (sample std; "±" omitted for a single seed). Copy the whole output,
including the BEGIN/END lines, into the conversation.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(d: Path):
    def j(name):
        p = d / name
        return json.loads(p.read_text()) if p.exists() else None

    hist = d / "history.jsonl"
    return {"dir": d.name, "env": j("env.json") or {}, "cfg": j("config.json") or {}, "res": j("results.json"),
            "halt": j("halting.json"), "bench": j("benchmark.json"),
            "hist": [json.loads(x) for x in hist.read_text().splitlines() if x.strip()] if hist.exists() else []}


def ms(vals, prec=1, scale=1.0):
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return "-"
    m = st.mean(vals)
    return f"{m:.{prec}f}" if len(vals) == 1 else f"{m:.{prec}f}±{st.stdev(vals):.{prec}f}"


def row(runs, getter, prec=1, scale=1.0):
    per_run = [getter(r) for r in runs]
    T = max(len(x) for x in per_run)
    return " ".join(ms([x[t] for x in per_run if t < len(x)], prec, scale) for t in range(T))


def benchmark_lines(runs_by_exp: dict[str, list[dict]]) -> list[str]:
    """TODO-4 wall-clock results; speed-ups are against the cheapest fixed depth at least as accurate."""
    out = []
    for exp, runs in runs_by_exp.items():
        bs = [r["bench"] for r in runs if r.get("bench")]
        if not bs:
            continue
        b0 = bs[0]
        out += ["", f"[benchmark {exp}] seeds {len(bs)} | {b0['device']} | AMP {b0['amp']} | policy {b0['policy']} | "
                    f"batch {b0['batch_size']} throughput (compaction), batch-1 latency on {b0['bs1_images']} images"]
        out.append("  setting        acc%          steps   img/s           bs1 ms        speed-up vs fixed (bs/bs1)")
        T = len(b0["fixed"])
        for t in range(T):
            rows = [b["fixed"][t] for b in bs]
            out.append(f"  fixed t={t:<6}" + ms([r["acc"] for r in rows], 2, 100).rjust(12) + f"{t + 1:8.2f}"
                       + ms([r["img_per_s"] for r in rows], 0).rjust(14) + ms([r["bs1_ms"] for r in rows], 3).rjust(14))
        for i, a0 in enumerate(b0["adaptive"]):
            rows = [b["adaptive"][i] for b in bs]
            ratios_b, ratios_1 = [], []
            for b, r in zip(bs, rows):  # cheapest fixed depth with accuracy >= this setting
                ok = [f for f in b["fixed"] if f["acc"] >= r["acc"]]
                f = ok[0] if ok else b["fixed"][-1]
                ratios_b.append(r["img_per_s"] / f["img_per_s"])
                ratios_1.append(f["bs1_ms"] / r["bs1_ms"])
            out.append(f"  lam={str(a0['lam']):<9}" + ms([r["acc"] for r in rows], 2, 100).rjust(12)
                       + ms([r["mean_steps"] for r in rows], 2).rjust(8) + ms([r["img_per_s"] for r in rows], 0).rjust(14)
                       + ms([r["bs1_ms"] for r in rows], 3).rjust(14)
                       + f"   {ms(ratios_b, 2)}x / {ms(ratios_1, 2)}x")
    return out


def halting_lines(runs_by_exp: dict[str, list[dict]]) -> list[str]:
    """TODO-1 study: AUC and accuracy at the compute of fixed depth t=1..3, averaged over backbone seeds."""
    out = []
    for exp, runs in runs_by_exp.items():
        halts = [r["halt"] for r in runs if r.get("halt")]
        if not halts:
            continue
        keys = list(halts[0]["matched_compute_gflops"])
        out += ["", f"[halting {exp}] backbone seeds {len(halts)} | test set, policies trained on held-out | "
                    f"AUC and acc@<fixed-depth compute> in %"]
        has_cost = "cost@full" in halts[0]["baselines"]["fixed_depth"]
        ckeys = ["cost@full", "cost@best"] if has_cost else []
        out.append("  " + "method".ljust(20) + "AUC".rjust(12) + "".join(f"acc@{k}".rjust(12) for k in keys)
                   + "".join(k.rjust(12) for k in ckeys))
        names = list(halts[0]["baselines"]) + list(dict.fromkeys(k for h in halts for k in h["policies"]))
        if "cost@full" in halts[0]["baselines"]["fixed_depth"]:
            out.append("  cost@full / cost@best = compute needed to match the full-depth / best fixed-depth accuracy, "
                       "% of full-depth compute ('-' = never)")
        for name in names:
            def val(h, k):
                v = h["baselines"].get(name) or h["policies"].get(name) or {}
                v = v.get(k)
                return v["mean"] if isinstance(v, dict) else v
            out.append("  " + name.ljust(20) + ms([val(h, "auc") for h in halts], 2, 100).rjust(12)
                       + "".join(ms([val(h, f"acc@{k}") for h in halts], 2, 100).rjust(12) for k in keys)
                       + "".join(ms([val(h, k) for h in halts], 1, 100).rjust(12) for k in ckeys))
        pol = halts[0]["policies"]
        rw = [k for k in pol if "reward" in k]
        if rw:
            viol = [h["policies"][k]["lambda_monotone_violations"]["mean"] for h in halts for k in rw
                    if k in h["policies"]]
            out.append(f"  lambda sweep: mean steps non-monotone in {st.mean(viol):.1f} of 40 adjacent pairs (reward policies)")
        cost = max(v["policy_cost_fraction_of_step"] for h in halts for v in h["policies"].values())
        out.append(f"  policy cost <= {cost * 100:.4f}% of one recurrent step; oracle any "
                   f"{ms([h['oracle']['any']['acc'] for h in halts], 2, 100)}% @ "
                   f"{ms([h['oracle']['any']['gflops'] for h in halts], 3)} GFLOPs")
    return out


def summarize(runs_by_exp: dict[str, list[dict]]) -> str:
    all_runs = [r for rs in runs_by_exp.values() for r in rs]
    e = max(all_runs, key=lambda r: (r["env"].get("code_version") is not None, r["env"].get("started", "")))["env"]
    out = ["===== BEGIN FSV SUMMARY =====",
           f"code {e.get('code_version', '?')} | {e.get('gpu', '?')} | torch {e.get('torch', '?')} "
           f"| cpus {e.get('max_cpus', '?')}/{e.get('machine_cpus', '?')}"]
    versions = {r["env"].get("code_version") for rs in runs_by_exp.values() for r in rs}
    if len(versions) > 1:
        out.append(f"NOTE: runs use different code versions: {sorted(map(str, versions))}")
    for exp, runs in runs_by_exp.items():
        done = [r for r in runs if r["res"]]
        seeds = ",".join(str(r["env"].get("seed", 0)) for r in runs)
        failed = [r["dir"] for r in runs if not r["res"]]
        cfg = runs[0]["cfg"]
        out.append("")
        out.append(f"[{exp}] seeds {seeds} | params {cfg.get('params', 0) / 1e6:.2f}M | "
                   f"train {ms([r['env'].get('train_minutes') for r in runs])} min | "
                   f"epoch {ms([st.mean(h['epoch_time_s'] for h in r['hist']) for r in runs if r['hist']])}s"
                   + (f" | FAILED/unfinished: {failed}" if failed else ""))
        if not done:
            continue
        res = [r["res"] for r in done]
        out.append("  test acc%  " + row(res, lambda r: [x["acc"] for x in r["test_raw"]], 2, 100))
        out.append("  ECE raw%   " + row(res, lambda r: [x["ece"] for x in r["test_raw"]], 2, 100))
        out.append("  ECE TS%    " + row(res, lambda r: [x["ece"] for x in r["test_ts"]], 2, 100))
        out.append("  NLL TS     " + row(res, lambda r: [x["nll"] for x in r["test_ts"]], 3))
        out.append("  AURC TS%   " + row(res, lambda r: [x["aurc"] for x in r["test_ts"]], 2, 100))
        out.append("  temp       " + row(res, lambda r: r["temperatures"], 3))
        out.append("  GFLOPs     " + " ".join(f"{c / 1e9:.3f}" for c in res[0]["flops"]["cumulative"]))
        lat = [r.get("latency", {}) for r in res]
        for k in [k for k in lat[0] if k.startswith("bs")]:
            out.append(f"  lat {k} ms " + row(lat, lambda x: x[k]["ms_cumulative"], 2))
        hold = [r["hist"][-1]["holdout"] for r in done if r["hist"] and "holdout" in r["hist"][-1]]
        if hold:
            out.append("  holdout%   " + row(hold, lambda h: [x["acc"] for x in h], 1, 100))
        last_loss = [r["hist"][-1].get("train_total") for r in done if r["hist"]]
        if last_loss:
            out.append(f"  final train loss {ms(last_loss, 3)} | AMP {cfg.get('amp', '?')}")
        best = [max(x["acc"] for x in r["test_raw"]) for r in res]
        out.append(f"  best-step acc {ms(best, 2, 100)} | last-step acc "
                   f"{ms([r['test_raw'][-1]['acc'] for r in res], 2, 100)}")
        out.append("  AUC " + " ".join(f"{k} {ms([r['curve_auc_acc'][k] for r in res], 2, 100)}"
                                        for k in res[0]["curve_auc_acc"]))
        out.append("  oracle " + " | ".join(
            f"{k} {ms([r['oracle'][k]['acc'] for r in res], 2, 100)}% @ "
            f"{ms([r['oracle'][k]['gflops'] for r in res], 3)} GFLOPs" for k in res[0]["oracle"]))
    out += halting_lines(runs_by_exp)
    out += benchmark_lines(runs_by_exp)
    out.append("===== END FSV SUMMARY =====")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", help="experiment names (default: all under reports/)")
    p.add_argument("--reports", default=str(ROOT / "reports"))
    args = p.parse_args(argv)

    runs_by_exp: dict[str, list[dict]] = {}
    for d in sorted(Path(args.reports).iterdir() if Path(args.reports).exists() else []):
        if not (d / "env.json").exists():
            continue
        r = load(d)
        exp = r["env"].get("experiment", d.name)
        if exp == "smoke" and not args.experiments:
            continue
        if args.experiments and exp not in args.experiments:
            continue
        runs_by_exp.setdefault(exp, []).append(r)
    if not runs_by_exp:
        sys.exit(f"no finished runs found under {args.reports}")
    for runs in runs_by_exp.values():
        runs.sort(key=lambda r: r["env"].get("seed", 0))
    print(summarize(runs_by_exp))


if __name__ == "__main__":
    main()
