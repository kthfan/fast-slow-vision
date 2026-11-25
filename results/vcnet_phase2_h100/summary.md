# Phase 2: slot pool for VCNet vs fixed depth as one CUDA graph (H100, pasted by the user)

`python scripts/benchmark_adaptive.py c100_vcnet_orig c100_vcnet_adapt_blt --seeds 0 1 2 --lams 0.003 0.01 0.03 0.1 0.3
--no-cudagraph --o3-policies --bs1-images 100`, code 0.16.0, idle H100 NVL, bf16, batch 256 / 256 slots (refill cap
256, V1 pre-computed). Raw output: [`raw.txt`](raw.txt).

## The fair comparison: `graph-compiled-bs256` (model step compiled, then captured as CUDA graphs, in both)

| setting | acc % | mean steps | img/s | us / image | vs matched fixed | vs mix of fixed depths |
|---|---|---|---|---|---|---|
| fixed 1 step | 63.72 | 1 | 180,759 | 5.53 | | |
| **fixed 2 steps** | 67.33 | 2 | **105,289** | 9.50 | | |
| fixed 3 steps | 68.30 | 3 | 76,032 | 13.15 | | |
| fixed 4 steps | 68.65 | 4 | 60,074 | 16.65 | | |
| slots, lam=0.3 | 66.70 | 1.40 | 79,684 | 12.55 | 0.76x (2 steps) | 0.70x |
| slots, lam=0.1 | 67.83 | 1.74 | 70,475 | 14.19 | 0.93x (3 steps) | 0.80x |
| slots, lam=0.03 | 68.24 | 2.09 | 62,075 | 16.11 | 0.87x (3 steps) | 0.80x |
| slots, lam=0.01 | 68.38 | 2.41 | 56,754 | 17.62 | 0.85x (4 steps) | 0.79x |
| slots, lam=0.003 | 68.49 | 2.90 | 49,745 | 20.10 | 0.74x (4 steps) | |

"vs mix of fixed depths": the throughput of sending each image to one of the two neighbouring fixed depths at
random so that the mean accuracy equals the policy's (time per image interpolated linearly). It is the frontier a
halting policy has to beat; matching only "the cheapest fixed depth at least as accurate" flatters the policy
when its accuracy sits just above a fixed depth.

Reading:

* **Adaptive still loses at batch 256 against the strongest baseline**: 0.70-0.80x of the fixed-depth frontier,
  0.93x of the matched fixed depth at best (lam=0.1).
* **But the gap shrank a lot.** The compaction runner reached 27-42% of fixed depth; the compiled slot pool is 5.3x
  faster than the compiled compaction runner at lam=0.1 (70,475 vs 13,241 img/s), with 92-95% of the slot-steps
  doing useful work.
* **Where the time goes** (lam=0.1, per image): ideal = V1 once (~1.8 us, a third of step 0) + 1.84 slot-steps
  x ~3.8 us = ~8.8 us, i.e. ~113k img/s, 1.28x the frontier; measured 14.2 us. About 5.3 us per image is overhead.
  A clear candidate: the V1 pre-computation ran EAGERLY (not compiled, not captured) even in the compiled mode, and
  eager blt kernels are about 2x slower than compiled ones even inside a graph (fixed 1 step: 87k vs 181k img/s),
  plus launch overhead outside a graph. 0.16.1 compiles it and captures each batch as a CUDA graph, and reports the
  split (V1 time, time per tick, ticks).
* **The fixed-depth baseline itself got much faster.** blt at 2 steps: eager 12,572 img/s, compiled (no graph)
  40,590, eager kernels in one graph 12,778, compiled + one graph **105,289** (8.4x eager). The earlier batch-256
  numbers in the report (compiled, no graph) understate what fixed depth can do: blt at 2 steps reaches 91% of the
  original VCNet's throughput (115,122 img/s) at +2.8 points.
* Eager-kernel modes (`slots`, `slots-triton`, `graph-bs256`) are slow and unstable (one seed 2-3x slower in several
  rows); they are not used for conclusions.
* Accuracy and mean steps of the slot pool match the compaction runner (e.g. lam=0.1: 67.77-67.83%, 1.74 steps).
