"""Draw the CPU / GPU timelines and the per-part time breakdown written by profile_breakdown.py.

    python scripts/plot_timeline.py results/t4_profile_breakdown/breakdown.json --out results/t4_profile_breakdown

For every model in the JSON it writes
  * timeline_<model>_<panel>.png  one call (one image, or one batch of 256): for each method a
    CPU lane (host time in each part) and a GPU lane (kernels, coloured by the part that
    launched them). Empty space on the GPU lane is GPU idle time.
  * breakdown_<model>.png         per method, GPU time per part + GPU idle = wall time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# categorical slots 1-5 of the reference palette (validated: adjacent CVD dE >= 9.1, normal >= 19.6)
GROUPS = [  # (label, colour, parts)
    ("model (embed / step / readout)", "#2a78d6", ("embed", "model_step", "readout", "argmax")),
    ("features", "#eb6834", ("features",)),
    ("policy MLP", "#1baf7a", ("policy",)),
    ("features + policy, one Triton kernel", "#eda100", ("fused_decide",)),
    ("bookkeeping (record / compaction)", "#e87ba4", ("record", "compaction")),
    ("sync (GPU to CPU read)", "#8f8e89", ("sync",)),
]
OTHER = ("other", "#d4d3cd")
COLOR = {p: c for _, c, ps in GROUPS for p in ps}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
PANEL_TITLE = {"graph-bs1": "batch 1, CUDA graphs", "eager-bs1": "batch 1, eager", "eager-bs256": "batch 256, eager"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def merge(segs, gap=0.3):
    """Join consecutive segments of the same part closer than `gap` us (many tiny kernels)."""
    out = []
    for part, a, b in sorted(segs, key=lambda s: s[1]):
        if out and out[-1][0] == part and a - out[-1][2] <= gap:
            out[-1][2] = max(out[-1][2], b)
        else:
            out.append([part, a, b])
    return out


def used_groups(parts):
    return [(lab, c) for lab, c, ps in GROUPS if any(p in parts for p in ps)] + (
        [OTHER] if any(p not in COLOR for p in parts) else [])


def plot_timeline(model: str, panel: str, cfgs: dict, out: Path, unit_ms: bool):
    names = list(cfgs)
    fig_h = 1.6 + 1.05 * len(names)
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    scale = 1e-3 if unit_ms else 1.0
    end = max(c["end_us"] for c in cfgs.values()) * scale
    yt, yl, parts = [], [], set()
    lane_h = 0.36
    for i, name in enumerate(names):
        c = cfgs[name]
        base = (len(names) - 1 - i) * 1.05
        for lane, (segs, y) in enumerate(((c["cpu"], base + 0.42), (c["gpu"], base))):
            # CPU parts can nest (e.g. none here) - draw outer first; GPU: merged kernels
            segs = merge(segs) if lane == 1 else sorted(segs, key=lambda s: s[2] - s[1], reverse=True)
            for part, a, b in segs:
                parts.add(part)
                w = max((b - a) * scale, end * 0.0015)  # keep microsecond kernels visible
                thin = w < end * 0.004  # no surface-coloured gap on slivers, it would hide them
                ax.broken_barh([(a * scale, w)], (y, lane_h), facecolors=COLOR.get(part, OTHER[1]),
                               edgecolor=SURFACE, linewidth=0 if thin else 0.6)
            yt.append(y + lane_h / 2)
            yl.append(f"{name}  {'CPU' if lane == 0 else 'GPU'}")
        busy = c["gpu_busy_us"] / c["end_us"] * 100
        ax.text(end * 1.01, base + 0.4, f"{c['end_us'] * scale:.2f} {'ms' if unit_ms else 'us'}\n"
                f"GPU busy {busy:.0f}%", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(yt, yl)
    ax.set_xlim(0, end * 1.16)
    ax.set_xlabel("time since the call started (" + ("ms" if unit_ms else "µs") + ")", color=INK2, fontsize=9)
    ax.set_title(f"{model} - {PANEL_TITLE.get(panel, panel)}: {'one batch of 256' if '256' in panel else 'one image'}"
                 " (under torch.profiler: host times inflated)", loc="left", fontsize=10.5, color=INK)
    handles = [Patch(facecolor=c, label=lab) for lab, c in used_groups(parts)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=min(4, len(handles)),
               frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.4, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.72 / fig_h, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_breakdown(res: dict, out: Path):
    """Horizontal stacked bars: GPU time per part group + idle, one bar per method, one panel per mode."""
    fixed_ts = res["fixed_ts"]
    panels = [("graph-bs1", "batch 1, CUDA graphs (µs / image)"), ("eager-bs1", "batch 1, eager (µs / image)"),
              ("eager-bs256", "batch 256, eager (ms / batch)")]
    n256 = 4
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 5.6 + 0.5 * n256), dpi=150,
                             gridspec_kw={"height_ratios": [4, 4, n256]})
    fig.patch.set_facecolor(SURFACE)
    for ax, (panel, title) in zip(axes, panels):
        style(ax)
        scale = 1e-3 if panel == "eager-bs256" else 1.0
        labels, bars = [], []
        for t in fixed_ts:
            if panel == "graph-bs1":  # fixed depth is ONE graph: only its total GPU and wall time are known
                ref = res["modes"]["graph-bs1/O1"]
                gpu, wall = {"model_step": ref["reference_gpu_us"][f"fixed_t{t}"]}, ref["reference_wall_us"][f"fixed_t{t}"]
            else:
                m = res["modes"][f"{panel}/fixed_t{t}"]
                gpu, wall = {q: v["gpu_us"] for q, v in m["parts"].items()}, m["wall_us"]
            labels.append(f"fixed {t + 1} steps")
            bars.append((gpu, wall))
        for k, name in (("O1", "O1"), ("O1-fused", "O1, features + policy fused")):
            m = res["modes"].get(f"{panel}/{k}")
            if m:
                labels.append(name)
                bars.append(({q: v["gpu_us"] for q, v in m["parts"].items()}, m["wall_us"]))
        ys = list(range(len(labels)))[::-1]
        parts = set()
        scaled = False
        for y, (gpu, wall) in zip(ys, bars):
            x = 0.0
            busy = sum(gpu.values())
            if busy > wall:  # profiled kernel time above the (unprofiled) wall time: clock drift on the GPU
                gpu = {q: v * wall / busy for q, v in gpu.items()}
                scaled = True
            for lab, c, ps in GROUPS:
                v = sum(gpu.get(p, 0.0) for p in ps) * scale
                if v <= 0:
                    continue
                parts.update(p for p in ps if p in gpu)
                ax.barh(y, v, left=x, height=0.55, color=c, edgecolor=SURFACE, linewidth=1.2)
                x += v
            idle = max(0.0, wall * scale - x)
            ax.barh(y, idle, left=x, height=0.55, color=SURFACE, edgecolor="#b5b4ae", linewidth=0.8, hatch="////")
            ax.text(max(x + idle, x) + wall * scale * 0.01, y, f"{wall * scale:,.{2 if scale < 1 else 0}f}",
                    va="center", fontsize=9, color=INK)
        ax.set_yticks(ys, labels)
        if panel == "graph-bs1":  # the O1 bars come from one graph per part (a few extra launches)
            real = [res["modes"][k]["reference_wall_us"]["o1_one_graph_per_step"] for k in
                    ("graph-bs1/O1", "graph-bs1/O1-fused") if k in res["modes"]]
            title += "; O1 bars: one graph per part, real O1 graphs " + " / ".join(f"{v:,.0f}" for v in real) + " µs"
        ax.set_title(title + ("  - bars scaled to wall where the profiled GPU time exceeded it (clock drift)"
                              if scaled else ""), loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, max(b[1] for b in bars) * scale * 1.12)
    handles = [Patch(facecolor=c, label=lab) for lab, c, _ in GROUPS] + [
        Patch(facecolor=SURFACE, edgecolor="#b5b4ae", hatch="////", label="GPU idle")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.01, 1.0),
               labelcolor=INK)
    fig.suptitle(f"{res['model']} on {res['device']}: GPU time per part + GPU idle = wall time (number at bar end)",
                 x=0.01, y=1.045, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    data = json.loads(Path(a.json).read_text())
    out = Path(a.out or Path(a.json).parent)
    out.mkdir(parents=True, exist_ok=True)
    for res in data:
        tag = res["model"].split()[0]
        plot_breakdown(res, out / f"breakdown_{tag}.png")
        for panel, cfgs in res.get("timelines", {}).get("panels", {}).items():
            cfgs = {k: v for k, v in cfgs.items() if v}
            if cfgs:
                plot_timeline(res["model"], panel, cfgs, out / f"timeline_{tag}_{panel}.png", "256" in panel)
        print(f"[plot] {tag}: written to {out}")


if __name__ == "__main__":
    main()
