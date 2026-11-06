"""Render a Markdown report from a report directory.

Expected files (written by scripts/run_experiment.py):
    env.json, config.json, history.jsonl, results.json
"""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def _row(label: str, values, fmt: str) -> str:
    return f"| {label} | " + " | ".join(format(v, fmt) for v in values) + " |"


def render_report(report_dir: str | Path) -> str:
    d = Path(report_dir)
    env = _load(d / "env.json") or {}
    cfg = _load(d / "config.json") or {}
    hist = _load(d / "history.jsonl") or []
    res = _load(d / "results.json")

    lines = [f"# Report: {env.get('experiment', d.name)}", ""]
    if env.get("description"):
        lines += [env["description"], ""]
    lines += [
        "## Environment",
        "",
        f"- git: `{env.get('git_commit', '?')}` on `{env.get('git_branch', '?')}`"
        + (" (**uncommitted changes**)" if env.get("git_dirty") else ""),
        f"- GPU: {env.get('gpu', '?')} | torch {env.get('torch', '?')} (CUDA {env.get('cuda', '?')}) "
        f"| Python {env.get('python', '?')}",
        f"- CPU cap: {env.get('max_cpus', '?')} cores (machine has {env.get('machine_cpus', '?')}); "
        f"DataLoader workers {cfg.get('args', {}).get('workers', '?')}",
        f"- AMP: {cfg.get('amp', '?')} | params: {cfg.get('params', 0) / 1e6:.2f}M",
        f"- started {env.get('started', '?')}, train {env.get('train_minutes', 0):.1f} min, "
        f"eval {env.get('eval_minutes', 0):.1f} min, exit codes train={env.get('train_exit')} "
        f"eval={env.get('eval_exit')}",
        f"- train command: `{env.get('train_cmd', '')}`",
        "",
    ]

    if hist:
        times = [h["epoch_time_s"] for h in hist if "epoch_time_s" in h]
        ips = [h["imgs_per_s"] for h in hist if "imgs_per_s" in h]
        lines += ["## Training", "",
                  f"- epochs run: {len(hist)}; mean epoch {sum(times) / len(times):.1f}s; "
                  f"mean throughput {sum(ips) / len(ips):.0f} img/s", ""]
        evals = [h for h in hist if "holdout" in h]
        if evals:
            T = len(evals[0]["holdout"])
            lines += ["Held-out top-1 % per step (no temperature scaling):", "",
                      "| epoch | train loss | " + " | ".join(f"t={t}" for t in range(T)) + " |",
                      "|---|---|" + "---|" * T]
            for h in evals:
                lines.append(f"| {h['epoch']} | {h.get('train_total', float('nan')):.3f} | "
                             + " | ".join(f"{r['acc'] * 100:.1f}" for r in h["holdout"]) + " |")
            lines.append("")

    if res:
        T = len(res["test_raw"])
        lat = res.get("latency", {})
        lat_keys = [k for k in lat if k.startswith("bs")]
        lines += ["## Test set, per step", "",
                  "| | " + " | ".join(f"t={t}" for t in range(T)) + " |", "|---|" + "---|" * T,
                  _row("top-1 %", [r["acc"] * 100 for r in res["test_raw"]], ".2f"),
                  _row("ECE raw %", [r["ece"] * 100 for r in res["test_raw"]], ".2f"),
                  _row("ECE TS %", [r["ece"] * 100 for r in res["test_ts"]], ".2f"),
                  _row("NLL TS", [r["nll"] for r in res["test_ts"]], ".3f"),
                  _row("Brier TS", [r["brier"] for r in res["test_ts"]], ".4f"),
                  _row("AURC TS %", [r["aurc"] * 100 for r in res["test_ts"]], ".2f"),
                  _row("temperature", res["temperatures"], ".3f"),
                  _row("cum. GFLOPs", [c / 1e9 for c in res["flops"]["cumulative"]], ".3f")]
        for k in lat_keys:
            lines.append(_row(f"latency {k} ms", lat[k]["ms_cumulative"], ".2f"))
        lo, hi = res["curve_auc_range_gflops"]
        lines += ["", f"Accuracy-compute AUC (mean top-1 % over [{lo:.3f}, {hi:.3f}] GFLOPs, TS probabilities):", ""]
        lines += [f"- {k}: {v * 100:.2f}" for k, v in res["curve_auc_acc"].items()]
        lines += ["", "Oracle stopping (t* = earliest step after which every step is correct):", ""]
        lines += [f"- {k}: top-1 {v['acc'] * 100:.2f}%, {v['gflops']:.3f} GFLOPs, {v['mean_steps']:.2f} steps"
                  for k, v in res["oracle"].items()]
        if lat_keys:
            lines += ["", f"Latency device: {lat.get('device', '?')}, AMP {lat.get('amp', '?')}; "
                      + "; ".join(f"{k} full-depth throughput {lat[k]['throughput_full_img_per_s']:.0f} img/s"
                                  for k in lat_keys)]
        lines.append("")
    else:
        lines += ["## Test set", "", "results.json missing (evaluation did not finish).", ""]
    return "\n".join(lines)


def write_report(report_dir: str | Path) -> Path:
    path = Path(report_dir) / "report.md"
    path.write_text(render_report(report_dir))
    return path
