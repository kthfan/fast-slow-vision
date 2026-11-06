"""Evaluate a trained recurrent ViT.

1. Collect per-step logits on the held-out split and the test set.
2. Stage B: fit one temperature per step on held-out.
3. Per-step accuracy / ECE / Brier / NLL / AURC on test, raw and temperature-scaled.
4. Analytic FLOPs per step (readout included).
5. Training-free halting baselines (max-prob / entropy thresholds), fixed-depth
   exits and oracle stopping, with area under the accuracy-compute curve.
6. Optional: per-step GPU latency (batch 1 and a deployment batch) and a cache
   of per-step logits + features for offline halting-policy research (Stage C).

    python scripts/evaluate.py --ckpt runs/c100_bvit_s/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsv.calibration import fit_step_temperatures  # noqa: E402
from fsv.data import build_splits  # noqa: E402
from fsv.engine import collect_outputs, step_latency  # noqa: E402
from fsv.flops import model_flops  # noqa: E402
from fsv.halting import curve_auc, fixed_depth_curve, oracle_point, threshold_curve  # noqa: E402
from fsv.metrics import per_step_metrics  # noqa: E402
from fsv.models import BViTConfig, RecurrentViT  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="output dir (default: checkpoint dir)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "none"])
    p.add_argument("--latency", action="store_true", help="measure per-step latency")
    p.add_argument("--latency-batch", type=int, default=256)
    p.add_argument("--cache", action="store_true", help="save per-step logits and features")
    p.add_argument("--cache-train", type=int, default=10000, help="train images to cache (0 = none)")
    return p.parse_args(argv)


def fmt_row(name, rows, key, scale=1.0, prec=1):
    return f"{name:<14}" + " ".join(f"{r[key] * scale:6.{prec}f}" for r in rows)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp if args.amp != "auto" else ("fp16" if device.type == "cuda" else "none")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = RecurrentViT(BViTConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    train_args = ck.get("args", {})
    dataset = train_args.get("dataset", "cifar100")
    holdout = train_args.get("holdout", 5000)
    out = Path(args.out or Path(args.ckpt).parent)
    out.mkdir(parents=True, exist_ok=True)

    splits = build_splits(dataset, args.data_dir, holdout, seed=0, download=True, fake=args.fake_data)
    loader = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,  # noqa: E731
                                   num_workers=args.workers, pin_memory=device.type == "cuda")

    ho = collect_outputs(model, loader(splits["holdout"]), device, amp, feats=args.cache)
    te = collect_outputs(model, loader(splits["test"]), device, amp, feats=args.cache)

    temps = fit_step_temperatures(ho["logits"], ho["labels"])
    raw = per_step_metrics(te["logits"], te["labels"])
    ts = per_step_metrics(te["logits"], te["labels"], temps)
    ho_ts = per_step_metrics(ho["logits"], ho["labels"], temps)

    fl = model_flops(model)
    cum = fl["cumulative"]
    probs = (te["logits"] / temps[:, None, None]).softmax(-1)
    labels = te["labels"]
    curves = {
        "maxprob": threshold_curve(probs, labels, cum, "maxprob"),
        "entropy": threshold_curve(probs, labels, cum, "entropy"),
        "fixed_depth": fixed_depth_curve(probs, labels, cum),
    }
    lo, hi = cum[0] / 1e9, cum[-1] / 1e9
    auc = {k: curve_auc(v, lo, hi) for k, v in curves.items()}
    oracle = {k: oracle_point(probs, labels, cum, k) for k in ("full", "any")}

    results = {
        "ckpt": str(args.ckpt),
        "dataset": dataset,
        "model_cfg": ck["model_cfg"],
        "params": sum(p.numel() for p in model.parameters()),
        "temperatures": temps.tolist(),
        "test_raw": raw,
        "test_ts": ts,
        "holdout_ts": ho_ts,
        "flops": fl,
        "curves": curves,
        "curve_auc_acc": auc,
        "curve_auc_range_gflops": [lo, hi],
        "oracle": oracle,
    }

    if args.latency:
        results["latency"] = {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "amp": amp,
            "bs1": step_latency(model, device, 1, amp),
            f"bs{args.latency_batch}": step_latency(model, device, args.latency_batch, amp),
        }

    (out / "results.json").write_text(json.dumps(results, indent=2))

    if args.cache:
        cache = {"temperatures": temps, "flops_cumulative": torch.tensor(cum), "model_cfg": ck["model_cfg"]}
        for name, res in (("holdout", ho), ("test", te)):
            torch.save({**cache, **res}, out / f"cache_{name}.pt")
        if args.cache_train > 0:
            n = min(args.cache_train, len(splits["train_eval"]))
            g = torch.Generator().manual_seed(0)
            idx = torch.randperm(len(splits["train_eval"]), generator=g)[:n].tolist()
            tr = collect_outputs(model, loader(Subset(splits["train_eval"], idx)), device, amp, feats=True)
            torch.save({**cache, **tr, "indices": torch.tensor(idx)}, out / "cache_train.pt")

    # ---- console summary
    T = len(raw)
    print("step          " + " ".join(f"{t:6d}" for t in range(T)))
    print(fmt_row("acc %", raw, "acc", 100))
    print(fmt_row("ECE raw %", raw, "ece", 100, 2))
    print(fmt_row("ECE TS %", ts, "ece", 100, 2))
    print(fmt_row("NLL TS", ts, "nll", 1, 3))
    print("temperature   " + " ".join(f"{t:6.3f}" for t in temps.tolist()))
    print("GFLOPs cum    " + " ".join(f"{c / 1e9:6.3f}" for c in cum))
    print(f"AUC (mean acc over [{lo:.3f}, {hi:.3f}] GFLOPs): " + ", ".join(f"{k} {v * 100:.2f}" for k, v in auc.items()))
    for k, v in oracle.items():
        print(f"oracle[{k}]: acc {v['acc'] * 100:.2f} gflops {v['gflops']:.3f} steps {v['mean_steps']:.2f}")
    if args.latency:
        for k in ("bs1", f"bs{args.latency_batch}"):
            lat = results["latency"][k]
            print(f"latency {k} ms: " + " ".join(f"{m:.2f}" for m in lat["ms_cumulative"]))
    print(f"[evaluate] wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
