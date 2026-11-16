# VCNet round 1, original model (H100, pasted by the user)

Code 0.9.0. `python scripts/summarize.py c100_vcnet_orig c100_vcnet_orig_ff`, 3 seeds each. The adaptive runs of the
same round are in [`adaptive.md`](adaptive.md).

Notes on reading it:

* `train 0.1 min` is the time of the last `train.py` process, which apparently resumed already-finished runs; the
  epoch times (4.8 s for 2 steps, 3.0 s for 1 step) are the real ones (200 epochs = about 16 and 10 minutes).
* `c100_vcnet_orig` is trained on its last readout only, so its step-0 row (60.99%) is an untrained readout through the
  shared head; only the step-1 row (64.55%) is the original model. Its halting AUCs are not meaningful.
* No benchmark: the benchmark script skipped runs without a halting policy before 0.10.0.

Comparison with the adaptive runs (test top-1, mean ± sd over seeds 0, 1, 2):

| model | mean steps | GFLOPs | acc % | ECE after TS % |
|---|---|---|---|---|
| orig_ff | 1 | 0.161 | 63.72±1.22 | 1.89 |
| orig | 2 | 0.275 | 64.55±0.96 | 1.79 |
| adapt, fixed 2 steps | 2 | 0.275 | 64.91±0.94 | 2.50 |
| adapt_carry, fixed 2 steps | 2 | 0.275 | 65.30±0.41 | 2.41 |
| adapt_carry, S0-reward_acc, lam=0.1 | 1.23 | 0.187 | 64.71±0.69 | - |
| adapt_carry, S0-reward_acc, lam=0.03 | 1.36 | 0.202 | 64.84±0.85 | - |

Adaptive GFLOPs = 0.161 + (mean steps - 1) x 0.1134 (policy cost < 0.01% of a step).

```
===== BEGIN FSV SUMMARY =====
code 0.9.0+dec7d6e42181 | NVIDIA H100 NVL | torch 2.6.0+cu124 | cpus 8/48

[c100_vcnet_orig] seeds 0,1,2 | params 1.02M | train 0.1±0.0 min | epoch 4.8±0.4s
  test acc%  60.99±0.93 64.55±0.96
  ECE raw%   11.60±0.78 12.17±0.34
  ECE TS%    1.91±0.34 1.79±0.26
  NLL TS     1.424±0.034 1.266±0.041
  AURC TS%   15.81±0.85 13.12±0.78
  temp       0.768±0.008 0.758±0.004
  GFLOPs     0.161 0.275
  lat bs1 ms 3.48±0.14 5.81±0.08
  lat bs256 ms 3.51±0.07 8.61±0.00
  holdout%   60.3±0.6 63.9±0.9
  final train loss 2.628±0.018 | AMP bf16
  best-step acc 64.55±0.96 | last-step acc 64.55±0.96
  AUC maxprob 63.80±0.76 entropy 63.66±0.75 fixed_depth 62.77±0.74
  oracle full 64.55±0.96% @ 0.210±0.001 GFLOPs | any 68.76±1.49% @ 0.170±0.002 GFLOPs

[c100_vcnet_orig_ff] seeds 0,1,2 | params 1.02M | train 0.1±0.0 min | epoch 3.0±0.0s
  test acc%  63.72±1.22
  ECE raw%   12.35±0.69
  ECE TS%    1.89±0.38
  NLL TS     1.310±0.039
  AURC TS%   13.71±0.68
  temp       0.755±0.008
  GFLOPs     0.161
  lat bs1 ms 3.33±0.07
  lat bs256 ms 3.49±0.04
  holdout%   62.8±1.5
  final train loss 2.664±0.019 | AMP bf16
  best-step acc 63.72±1.22 | last-step acc 63.72±1.22
  AUC maxprob 63.72±1.22 entropy 63.72±1.22 fixed_depth 63.72±1.22
  oracle full 63.72±1.22% @ 0.161±0.000 GFLOPs | any 63.72±1.22% @ 0.161±0.000 GFLOPs
===== END FSV SUMMARY =====
```
