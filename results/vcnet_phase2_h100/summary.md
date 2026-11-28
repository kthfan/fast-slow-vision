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

## Second measurement (0.16.1: V1 pre-embedding compiled and captured)

| setting | acc % | steps | img/s (0.16.0 -> 0.16.1) | vs matched fixed | vs mix of fixed depths | V1 | tick x ticks |
|---|---|---|---|---|---|---|---|
| lam=0.3 | 66.75 | 1.40 | 79,684 -> **92,653** | 0.88x (2 steps) | 0.82x | 29.0 ms | 1.337 ms x 59 |
| lam=0.1 | 67.78 | 1.74 | 70,475 -> **79,210** | **1.05x** (3 steps) | 0.89x | 29.0 ms | 1.336 ms x 73 |
| lam=0.03 | 68.21 | 2.08 | 62,075 -> **68,636** | 1.00x (3 steps) | 0.89x | 29.0 ms | 1.334 ms x 88 |
| lam=0.01 | 68.37 | 2.41 | 56,754 -> **62,068** | 0.98x (4 steps) | 0.87x | 29.0 ms | 1.335 ms x 100 |

(10,000 test images; fixed depth in the same run: 178,226 / 105,170 / 75,391 / 58,555 img/s at 1-4 steps.)

Time budget per image at lam=0.1 (126 ms per 10,000 images = 29.0 ms V1 + 73 ticks x 1.336 ms):

| part | us / image | note |
|---|---|---|
| V1, once | 2.90 | real compute, also in every fixed-depth pass (fixed step 0 = 5.61 us = V1 + first sweep) |
| ticks | 9.75 | 73 ticks x 256 slots; 93.5% of the slot-steps useful |
| of which: the step itself | 7.29 | a fixed step >= 1 at batch 256 takes ~1.00 ms (105,170 vs 178,226 img/s) |
| of which: tick overhead | 2.46 | 0.34 ms per tick: readout + decision + refill + record + per-row masks and the packed state |

Reading:

* The compiled, captured V1 pre-embedding gave +10-16%. The slot pool now beats the matched fixed depth at lam=0.1
  (1.05x vs 3 steps) and ties it at lam=0.03, but stays below the fixed-depth frontier (0.82-0.89x).
* Bound on what is left: without any tick overhead (tick = 1.00 ms) lam=0.1 would reach ~98k img/s, 1.10x the
  frontier. The ceiling is low because (1) the mean depth is only 1.74 steps, so adaptivity can save at most part of
  the gap between 2 and 3 fixed steps; (2) V1 (2.9 us, a quarter of the per-image time) is paid by every image
  whatever its depth; (3) in a slot every row pays a feedback step (~3.9 us) even at its step 0, which fixed depth
  does more cheaply as a plain sweep (~2.7 us).
* Conclusion for CIFAR-100: at batch 256, fixed depth compiled + captured as one CUDA graph is the best way to
  serve this model; the adaptive slot pool is within 11-18% of it and at most ~10% ahead even with zero overhead.
  Adaptivity pays at batch 1 (6-9% over blt_orig2 at equal accuracy, and 1.4x over blt's own fixed 3 steps).
  Larger images / more steps per image (phase 4) leave more room.
