# VCNet benchmark on an idle GPU (H100, pasted by the user)

`scripts/benchmark_adaptive.py c100_vcnet_orig c100_vcnet_adapt_carry c100_vcnet_adapt_blt --seeds 0 1 2
--lams 0.003 0.01 0.03 0.1 0.3`, run with no other programs on the GPU, after round 2. The raw per-seed output is
[`raw.txt`](raw.txt).

The modes `eager-triton` and `cudagraph-bs1-triton` come from branch `claude/admiring-franklin-6zlsn0`, which fuses
the halting features and the policy MLP into one Triton kernel and had merged the VCNet code from `main`. So this run
used that branch, not `dev/vcnet`. The model, policy and fixed-depth code paths are the same; the `-triton` modes only
change how the stop decision is computed.

Summary (3 seeds: accuracy, mean steps and throughput are means; batch-1 CUDA-graph latency is the median over seeds,
because a single seed sometimes runs slow, e.g. blt seed 1 at lam=0.1: 3.095 ms vs 1.825 / 1.811 ms):

| model, setting | acc % | mean steps | bs256 eager img/s | bs256 compiled img/s | bs1 CUDA graph ms | + Triton ms |
|---|---|---|---|---|---|---|
| orig, 1 step (untrained readout) | 61.00 | 1 | 60,679 | 76,681 | 1.059 | - |
| **orig, 2 steps** | **64.56** | 2 | 29,377 | 41,499 | 1.908 | - |
| adapt_carry, fixed 2 steps | 65.33 | 2 | 29,212 | 42,182 | 1.809 | - |
| adapt_blt, fixed 1 step | 63.71 | 1 | 65,990 | 73,840 | 1.008 | - |
| adapt_blt, fixed 2 steps | 67.27 | 2 | 11,610 | 39,804 | 1.949 | - |
| adapt_blt, fixed 3 steps | 68.29 | 3 | 5,784 | 27,081 | 2.915 | - |
| adapt_blt, fixed 4 steps | 68.68 | 4 | 4,369 | 20,954 | 3.882 | - |
| adapt_blt + policy, lam=0.3 | 66.74 | 1.40 | 12,718 | 16,774 | 1.583 | 1.486 |
| adapt_blt + policy, lam=0.1 | 67.77 | 1.73 | 9,198 | 13,567 | 1.825 | 1.776 |
| adapt_blt + policy, lam=0.03 | 68.20 | 2.09 | 6,898 | 10,644 | 2.125 | 1.989 |
| adapt_blt + policy, lam=0.01 | 68.34 | 2.41 | 5,983 | 10,224 | 2.598 | 2.432 |

Reading:

* Batch 1 (CUDA graph): adaptive helps. blt + policy at lam=0.1 is 1.78-1.83 ms at 67.77%, the original VCNet at
  2 steps 1.91 ms at 64.56%: 4-7% lower latency, +3.2 points. Against blt's own fixed depth, lam=0.03 reaches 68.20%
  at 1.99-2.13 ms where fixed 3 steps needs 2.92 ms for 68.29% (about 1.4x faster).
* Batch 256 (compiled): adaptive hurts; fixed depth is better. blt fixed 2 steps reaches 96% of the original's
  throughput with +2.7 points, but the halting policy drops throughput to 27-42% of fixed 2 steps (the batch is
  compacted after every step, so later steps run on ever smaller batches), and no policy setting beats fixed 3 steps.
* Eager blt is slow (fixed 2 steps 11,610 img/s, 29% of compiled), matching the evaluate.py measurement (3.2x the
  per-step time of adapt at batch 256). Deploy blt with torch.compile.
* The fused Triton kernel lowers the batch-1 CUDA-graph policy latency by a further 3-9% (median).
* Fixed-depth CUDA-graph latencies agree across seeds to < 0.1 ms; the round-2 values for the original
  (3.20 +- 1.25 ms at 2 steps) were disturbed by other jobs.
* adapt_carry seed 2: the policy stops every image at step 0 for every lambda (as adapt did in round 1).
