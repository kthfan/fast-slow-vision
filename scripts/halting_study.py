"""TODO-1 halting-policy study on cached per-step outputs (backbone frozen).

    python scripts/halting_study.py --run-dir runs/c100_bvit_s_linw --out reports/c100_bvit_s_linw

Needs runs/<name>/cache_holdout.pt and cache_test.pt (``evaluate.py --cache``;
``run_experiment.py --halting`` creates them). Policies are trained on the
held-out split (the split the temperatures were fitted on) and evaluated on
the test set, alongside the training-free baselines and the oracle.

Writes <out>/halting.json (merged with an existing one: policies not re-run
are kept) and prints a table. Default grid: feature sets {S0, S3} x objectives
{reward, reward_acc, oracle_full, oracle_any}, several policy seeds each;
--feature-sets S2 S5 adds the state-feature variants of TODO-1b.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsv.cpu import limit_cpus  # noqa: E402
from fsv.halting import (acc_at_compute, curve_auc, fixed_depth_curve, oracle_point,  # noqa: E402
                         threshold_curve)
from fsv.halting.features import ALL_FEATURE_SETS, FEATURE_SETS, build_features, feature_flops  # noqa: E402
from fsv.halting.policy import OBJECTIVES, policy_curve, train_policy  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policy-seeds", type=int, default=3)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(ALL_FEATURE_SETS))
    p.add_argument("--objectives", nargs="+", default=list(OBJECTIVES))
    p.add_argument("--policy-train", default="holdout", choices=["holdout", "train", "both"],
                   help="data the policy is trained on: held-out outputs (default), training-image outputs "
                        "(needs cache_train.pt; validated on held-out), or both. Non-default runs are named <set>-<obj>@<split>")
    p.add_argument("--device", default="auto", help="where policies are trained: auto (cuda if available) | cpu | cuda")
    p.add_argument("--max-cpus", type=int, default=None)
    return p.parse_args(argv)


def mean_sd(v):
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "values": v}


def main(argv=None):
    args = parse_args(argv)
    limit_cpus(args.max_cpus)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    run, out = Path(args.run_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ho = torch.load(run / "cache_holdout.pt", map_location="cpu", weights_only=False)
    te = torch.load(run / "cache_test.pt", map_location="cpu", weights_only=False)
    temps = ho["temperatures"].float()
    cum = [float(c) for c in ho["flops_cumulative"]]
    T, _, C = te["logits"].shape
    D = te["cls"].shape[-1]
    step_flops = cum[1] - cum[0] if T > 1 else cum[0]
    lo, hi = cum[0] / 1e9, cum[-1] / 1e9
    probs = {k: (c["logits"].float() / temps[:, None, None]).softmax(-1) for k, c in (("ho", ho), ("te", te))}
    labels_ho, labels_te = ho["labels"], te["labels"]
    fixed = fixed_depth_curve(probs["te"], labels_te, cum)
    match = {f"t{t}": fixed[t]["gflops"] for t in range(1, min(4, T))}  # compute of fixed depth t

    anchor = fixed[0]  # "stop everything at step 0" is always available

    def summarize_curve(points):
        return {"auc": curve_auc(points, lo, hi, anchor),
                **{f"acc@{k}": acc_at_compute(points, g, anchor) for k, g in match.items()}}

    t0 = time.time()
    res = {"run": run.name, "T": T, "range_gflops": [lo, hi], "matched_compute_gflops": match,
           "auc_anchor": "step-0 fixed-depth point; linear (random-mixing) interpolation below each curve",
           "baselines": {}, "policies": {}, "step_gflops": step_flops / 1e9}
    for name, pts in (("fixed_depth", fixed),
                      ("maxprob", threshold_curve(probs["te"], labels_te, cum, "maxprob")),
                      ("entropy", threshold_curve(probs["te"], labels_te, cum, "entropy"))):
        res["baselines"][name] = {**summarize_curve(pts), "curve": pts}
    res["oracle"] = {k: oracle_point(probs["te"], labels_te, cum, k) for k in ("full", "any")}

    tr = None
    if args.policy_train != "holdout":
        tr = torch.load(run / "cache_train.pt", map_location="cpu", weights_only=False)
        probs["tr"] = (tr["logits"].float() / temps[:, None, None]).softmax(-1)
        res["policy_train"] = {"split": args.policy_train, "n_train_images": int(tr["labels"].shape[0]),
                               "train_image_acc_last_step": float((probs["tr"][-1].argmax(-1) == tr["labels"])
                                                                  .float().mean())}
    suffix = "" if args.policy_train == "holdout" else f"@{args.policy_train}"

    for fs in args.feature_sets:
        x_ho, x_te = build_features(ho, temps, fs), build_features(te, temps, fs)
        if tr is None:
            x_fit, p_fit, y_fit, n_val_last = x_ho, probs["ho"], labels_ho, None
        else:
            x_tr = build_features(tr, temps, fs)
            if args.policy_train == "train":  # train on training images, select on all held-out images
                x_fit = torch.cat([x_tr, x_ho], 1)
                p_fit, y_fit = torch.cat([probs["tr"], probs["ho"]], 1), torch.cat([tr["labels"], labels_ho])
                n_val_last = len(labels_ho)
            else:  # both: training images + 80% of held-out; select on the other 20%
                perm = torch.randperm(len(labels_ho), generator=torch.Generator().manual_seed(0))
                cut = int(0.8 * len(perm))
                a, b = perm[:cut], perm[cut:]
                x_fit = torch.cat([x_tr, x_ho[:, a], x_ho[:, b]], 1)
                p_fit = torch.cat([probs["tr"], probs["ho"][:, a], probs["ho"][:, b]], 1)
                y_fit = torch.cat([tr["labels"], labels_ho[a], labels_ho[b]])
                n_val_last = len(b)
        for obj in args.objectives:
            runs = []
            for seed in range(args.policy_seeds):
                model, info = train_policy(x_fit, p_fit, y_fit, obj, seed=seed, iters=args.iters,
                                           n_val_last=n_val_last, device=device)
                extra = feature_flops(fs, T, C, D) + model.flops()
                pts = policy_curve(model, x_te, probs["te"], labels_te, cum, extra, device=device)
                summary = summarize_curve(pts)
                if obj.startswith("reward"):  # TODO-3: does more compute price mean fewer steps?
                    steps = [p["mean_steps"] for p in pts]
                    summary["lambda_monotone_violations"] = sum(b > a + 1e-9 for a, b in zip(steps, steps[1:]))
                runs.append({**summary, "curve": pts, "val_loss": info["best_val_loss"],
                             "policy_flops_per_step": extra})
            name = f"{fs}-{obj}{suffix}"
            agg = {k: mean_sd([r[k] for r in runs]) for k in runs[0] if k not in ("curve",)}
            agg["policy_cost_fraction_of_step"] = runs[0]["policy_flops_per_step"] / step_flops
            agg["curve"] = runs[0]["curve"]
            res["policies"][name] = agg
            print(f"[halting] {name:<18} AUC {agg['auc']['mean'] * 100:.2f}±{agg['auc']['sd'] * 100:.2f}  "
                  + "  ".join(f"{k} {agg[k]['mean'] * 100:.2f}" for k in agg if k.startswith("acc@")), flush=True)
    res["minutes"] = (time.time() - t0) / 60
    res["device"] = device
    path = out / "halting.json"
    if path.exists():  # merge: keep policies from earlier studies that this call did not re-run
        old = json.loads(path.read_text())
        if old.get("run") == res["run"]:
            res["policies"] = {**{k: v for k, v in old.get("policies", {}).items() if k not in res["policies"]},
                               **res["policies"]}
    path.write_text(json.dumps(res, indent=2))

    print(f"\nhalting study: {run.name} (test set; policies trained on held-out; AUC over [{lo:.3f}, {hi:.3f}] GFLOPs)")
    hdr = "method".ljust(20) + "AUC %".rjust(13) + "".join(f"{'acc@' + k + ' %':>13}" for k in match)
    print(hdr)
    for name, b in res["baselines"].items():
        print(name.ljust(20) + f"{b['auc'] * 100:13.2f}" + "".join(f"{b['acc@' + k] * 100:13.2f}" for k in match))
    for name, a in res["policies"].items():
        print(name.ljust(20) + f"{a['auc']['mean'] * 100:9.2f}±{a['auc']['sd'] * 100:.2f}"
              + "".join(f"{a['acc@' + k]['mean'] * 100:13.2f}" for k in match))
    for k, v in res["oracle"].items():
        print(f"oracle[{k}]: {v['acc'] * 100:.2f}% @ {v['gflops']:.3f} GFLOPs")
    print(f"policy cost per step: {max(a['policy_cost_fraction_of_step'] for a in res['policies'].values()) * 100:.4f}% "
          f"of one recurrent step (budget 2%)")
    print(f"[halting] wrote {out / 'halting.json'} in {res['minutes']:.1f} min")


if __name__ == "__main__":
    main()
