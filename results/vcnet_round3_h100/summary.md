# VCNet round 3: the blt architecture trained like the original VCNet (H100, pasted by the user)

`python scripts/run_experiment.py c100_vcnet_blt_orig2 c100_vcnet_blt_orig4 --seeds 0 1 2 --parallel 3`, code 0.15.1,
old training pipeline (CPU augmentation, eager). Three runs at a time shared the 8-core CPU cap (2 cores each), which
starved the CPU loader: training times (68 / 199 min) are not comparable with earlier rounds. With `--parallel`,
evaluate.py measured no latency.

```
[c100_vcnet_blt_orig2] seeds 0,1,2 | params 1.15M | train 68.1±0.1 min | epoch 24.2±0.0s
  test acc%  41.06±3.45 67.61±0.74
  ECE raw%   10.53±2.04 12.17±0.15
  ECE TS%    1.81±0.72 1.60±0.36
  NLL TS     2.367±0.139 1.170±0.031
  AURC TS%   36.75±5.51 11.30±0.38
  temp       0.799±0.019 0.757±0.004
  GFLOPs     0.161 0.284
  holdout%   40.3±3.2 66.9±0.6
  final train loss 2.543±0.109 | AMP bf16
  best-step acc 67.61±0.74 | last-step acc 67.61±0.74
  AUC maxprob 57.46±1.74 entropy 57.07±1.69 fixed_depth 54.34±1.66
  oracle full 67.61±0.74% @ 0.238±0.004 GFLOPs | any 71.16±0.76% @ 0.198±0.004 GFLOPs

[c100_vcnet_blt_orig4] seeds 0,1,2 | params 1.15M | train 198.6±0.3 min | epoch 59.0±0.1s
  test acc%  19.01±4.01 44.57±3.19 48.57±2.01 68.90±0.62
  ECE raw%   2.26±1.17 13.44±2.25 14.10±0.27 12.12±0.14
  ECE TS%    1.65±0.54 2.09±0.07 1.35±0.14 1.46±0.23
  NLL TS     3.482±0.185 2.143±0.151 2.079±0.114 1.121±0.022
  AURC TS%   67.97±7.41 32.66±3.51 26.85±1.85 10.53±0.32
  temp       0.957±0.062 0.739±0.031 0.757±0.005 0.756±0.009
  GFLOPs     0.161 0.284 0.406 0.528
  holdout%   18.7±3.8 44.2±2.6 47.0±1.8 67.8±0.5
  final train loss 2.474±0.024 | AMP bf16
  best-step acc 68.90±0.62 | last-step acc 68.90±0.62
  AUC maxprob 51.04±0.24 entropy 50.99±0.35 fixed_depth 45.70±1.08
  oracle full 68.90±0.62% @ 0.420±0.005 GFLOPs | any 71.20±0.59% @ 0.306±0.005 GFLOPs
```

## Architecture vs adaptivity (CIFAR-100 test, 3 seeds)

| model | training | steps | GFLOPs | acc % |
|---|---|---|---|---|
| orig (round 1) | one readout, CE | 2 | 0.275 | 64.55±0.96 |
| orig_fast (phase 0) | one readout, CE | 2 | 0.275 | 65.48±1.77 |
| **blt_orig2** | one readout, CE | 2 | 0.284 | **67.61±0.74** |
| adapt_blt (round 2), fixed 2 steps | every step, Stage A | 2 | 0.284 | 67.27±0.57 |
| adapt_blt_fast, fixed 2 steps | every step, Stage A | 2 | 0.284 | 67.62±0.44 |
| **blt_orig4** | one readout, CE | 4 | 0.528 | **68.90±0.62** |
| adapt_blt (round 2), fixed 4 steps | every step, Stage A | 4 | 0.528 | 68.68±0.58 |
| adapt_blt_fast, fixed 4 steps | every step, Stage A | 4 | 0.528 | 68.85±0.49 |
| adapt_blt + policy, lambda=0.1 | every step, Stage A | 1.73 | ~0.251 | 67.77±0.42 |
| adapt_blt + policy, lambda=0.03 | every step, Stage A | 2.09 | ~0.295 | 68.20±0.58 |
| adapt_blt + policy, lambda=0.01 | every step, Stage A | 2.41 | ~0.334 | 68.34±0.55 |

Reading:

* The architecture carries the gain. At the same compute, blt_orig2 is +3.1 points over the original VCNet
  (67.61 vs 64.55; +2.1 against orig_fast, whose seed spread is larger).
* Reading out and training every step costs nothing at the final depth: adapt_blt at fixed 2 / 4 steps equals
  blt_orig2 / blt_orig4 (67.27-67.62 vs 67.61; 68.68-68.85 vs 68.90), all within one seed standard deviation.
* What adaptivity adds on CIFAR is modest. One adapt_blt model matches each fixed-depth specialist at its own depth
  (one model for every compute budget, where the original-style models need one training per depth and their
  intermediate readouts are useless: blt_orig4 at step 1-3 is 19 / 45 / 49%). With the policy, lambda=0.1 matches
  blt_orig2 (67.77 vs 67.61) at 88% of its compute; lambda=0.03 is +0.6 points for +4% compute. No policy setting
  reaches blt_orig4 (best 68.34 at 63% of its compute, -0.56 points).
* Latency needs no new measurement: blt_orig2 / blt_orig4 run the same computation as adapt_blt at fixed 2 / 4
  steps (idle-GPU benchmark: batch 1 CUDA graph 1.949 / 3.882 ms, batch 256 compiled 39,804 / 20,954 img/s). At
  batch 1, adapt_blt + policy lambda=0.1 (1.78-1.83 ms) is 6-9% faster than blt_orig2 at the same accuracy; at
  batch 256 the fixed-depth model wins.
* Calibration after temperature scaling: the original-style training is a little better (ECE 1.46-1.60% vs
  2.3-2.6%, NLL 1.170 vs 1.179-1.187 at 2 steps, 1.121 vs 1.135-1.144 at 4), as in round 1: the Stage A terms do
  not improve post-scaling calibration.
