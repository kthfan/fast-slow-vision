# VCNet round 2 (H100, pasted by the user)

Code 0.10.0. New: `c100_vcnet_adapt_lat` (every area keeps its state across steps) and `c100_vcnet_adapt_blt`
(state + BLT top-down), 3 seeds each with `--halting`; then `scripts/benchmark_adaptive.py c100_vcnet_orig
c100_vcnet_adapt_carry c100_vcnet_adapt_lat c100_vcnet_adapt_blt --seeds 0 1 2 --lams 0.0001 ... 0.3`.
The user reports that other programs may have been running on the GPU during the benchmark.

Notes on reading it:

* The per-area state removes the saturation. lat: 64.75 -> 67.39 -> 67.83 -> 67.90 (t=0..3).
  blt: 63.72 -> 67.27 -> 68.30 -> 68.68 (t=0..3), then flat at ~68.6. Best fixed depth: blt 68.68 at 4 steps.
* Against the original VCNet (64.55% at 0.275 GFLOPs), fixed depth, same compute (2 steps, 0.28 GFLOPs):
  lat 67.39 (+2.8), blt 67.27 (+2.7). lat at 1 step (0.161 GFLOPs, 59% of the original's compute) is 64.75%.
* Halting now beats fixed depth inside the same model (it did not in round 1). blt, S0-reward_acc: AUC 68.35 vs
  68.12; at the compute of 2 steps 68.18 vs 67.27; at 3 steps 68.59 vs 68.30. Benchmark rows, blt: lam=0.1 67.77%
  at 1.74 steps (0.252 GFLOPs, 91% of the original), lam=0.03 68.20% at 2.09 steps, lam=0.01 68.34% at 2.41 steps
  (fixed 3 steps: 68.29%). lat gains less (AUC +0.14).
  Adaptive GFLOPs = 0.161 + (mean steps - 1) x step cost (blt 0.1223, lat 0.1195).
* Most of the gain over the original comes from the architecture (state across steps), not from halting. The
  original was trained with one readout and CE only; an architecture-matched original (blt trained with fixed steps
  and one readout) is needed to separate the two: round 3.
* Calibration: step 0 is strongly under-confident (temperature ~0.69, raw ECE ~17%); after per-step temperature
  scaling ECE is 2.2-2.5% for every step. NLL after TS improves with the architecture (blt 1.144, orig 1.266).
* Timing is disturbed (the user's note): several rows have standard deviations of 30-100% of the mean, e.g.
  carry cudagraph fixed t=1 3.065+-1.124 ms (round 1, same model: 1.818+-0.083), orig cudagraph t=1 3.204+-1.247 ms.
  blt batch-1 CUDA-graph rows are the most consistent: fixed 2/3/4 steps 2.10/3.12/4.17 ms, adaptive lam=0.1
  1.84 ms (67.77%), lam=0.03 2.22 ms (68.20%). Batch 256: adaptive is still slower than fixed depth (compiled
  0.34-0.44x for blt).
* Training: blt epoch 36.8+-10.7 s and lat 21.7+-4.6 s vs 16.2 s for carry, for +5-8% FLOPs; part of it may be the
  same interference, part is the packing / top-down upsampling overhead.

```
===== BEGIN FSV SUMMARY =====
code 0.10.0+85e59bca386a | NVIDIA H100 NVL | torch 2.6.0+cu124 | cpus 8/48
NOTE: runs use different code versions: ['0.10.0+85e59bca386a', '0.9.0+dec7d6e42181']

[c100_vcnet_adapt_blt] seeds 0,1,2 | params 1.15M | train 124.1±36.0 min | epoch 36.8±10.7s
  test acc%  63.72±0.77 67.27±0.57 68.30±0.48 68.68±0.58 68.65±0.39 68.60±0.29 68.56±0.30 68.61±0.30
  ECE raw%   17.60±0.10 12.28±0.22 11.74±0.35 11.79±0.42 11.64±0.39 11.57±0.32 11.53±0.28 11.58±0.29
  ECE TS%    2.22±0.30 2.32±0.62 2.43±0.54 2.34±0.61 2.49±0.72 2.49±0.56 2.47±0.61 2.43±0.50
  NLL TS     1.319±0.026 1.187±0.013 1.153±0.012 1.144±0.012 1.143±0.011 1.143±0.010 1.144±0.010 1.144±0.010
  AURC TS%   13.67±0.46 11.24±0.30 10.57±0.27 10.43±0.29 10.39±0.23 10.41±0.20 10.43±0.22 10.43±0.21
  temp       0.688±0.004 0.751±0.004 0.758±0.003 0.760±0.004 0.762±0.005 0.762±0.005 0.763±0.005 0.763±0.005
  GFLOPs     0.161 0.284 0.406 0.528 0.650 0.773 0.895 1.017
  lat bs1 ms 3.21±0.06 6.15±0.11 9.09±0.16 11.75±0.14 14.62±0.24 17.42±0.32 19.85±1.13 22.36±1.22
  lat bs256 ms 3.29±0.03 20.07±0.01 36.81±0.00 53.55±0.01 70.30±0.02 87.03±0.02 103.72±0.07 120.39±0.08
  holdout%   63.4±0.6 66.6±0.3 67.5±0.2 67.6±0.2 67.7±0.2 67.6±0.1 67.7±0.1 67.7±0.1
  final train loss 3.252±0.036 | AMP bf16
  best-step acc 68.79±0.45 | last-step acc 68.61±0.30
  AUC maxprob 67.87±0.31 entropy 67.62±0.31 fixed_depth 68.12±0.48
  oracle full 68.61±0.30% @ 0.454±0.003 GFLOPs | any 71.83±0.25% @ 0.185±0.001 GFLOPs

[c100_vcnet_adapt_carry] seeds 0,1,2 | params 1.02M | train 54.7±2.3 min | epoch 16.2±0.7s
  test acc%  63.69±0.61 65.30±0.41 65.44±0.53 65.40±0.53 65.43±0.54 65.40±0.53 65.42±0.54 65.39±0.51
  ECE raw%   12.83±0.42 12.95±0.31 13.03±0.21 12.99±0.18 13.02±0.20 13.00±0.20 13.01±0.21 12.99±0.19
  ECE TS%    2.72±0.05 2.41±0.29 2.41±0.07 2.51±0.19 2.53±0.14 2.52±0.12 2.48±0.15 2.47±0.15
  NLL TS     1.333±0.032 1.268±0.031 1.265±0.030 1.264±0.030 1.264±0.030 1.264±0.030 1.264±0.030 1.264±0.030
  AURC TS%   13.64±0.35 12.48±0.28 12.41±0.32 12.41±0.32 12.40±0.32 12.40±0.32 12.40±0.32 12.40±0.31
  temp       0.738±0.007 0.741±0.007 0.741±0.007 0.740±0.007 0.741±0.007 0.741±0.007 0.740±0.007 0.740±0.007
  GFLOPs     0.161 0.275 0.388 0.502 0.615 0.729 0.842 0.955
  lat bs1 ms 3.41±0.03 5.79±0.09 8.23±0.18 10.46±0.32 12.89±0.18 15.35±0.29 17.87±0.30 21.37±2.14
  lat bs256 ms 4.76±2.18 12.81±7.18 21.11±12.24 29.17±16.91 37.45±21.94 45.57±26.75 53.63±31.42 61.77±36.14
  holdout%   62.9±0.6 64.6±0.3 64.7±0.0 64.6±0.1 64.7±0.1 64.6±0.0 64.7±0.1 64.7±0.1
  final train loss 3.377±0.022 | AMP bf16
  best-step acc 65.48±0.48 | last-step acc 65.39±0.51
  AUC maxprob 65.17±0.48 entropy 65.07±0.50 fixed_depth 65.32±0.48
  oracle full 65.39±0.51% @ 0.444±0.004 GFLOPs | any 68.33±0.63% @ 0.170±0.000 GFLOPs

[c100_vcnet_adapt_lat] seeds 0,1,2 | params 1.09M | train 73.2±15.4 min | epoch 21.7±4.6s
  test acc%  64.75±0.49 67.39±0.48 67.83±0.47 67.90±0.48 67.79±0.54 67.80±0.54 67.83±0.55 67.80±0.52
  ECE raw%   17.19±0.23 12.46±0.14 12.41±0.29 12.26±0.33 12.11±0.39 12.12±0.45 12.16±0.38 12.12±0.38
  ECE TS%    2.27±0.17 2.40±0.18 2.50±0.31 2.28±0.07 2.35±0.07 2.36±0.19 2.38±0.04 2.40±0.10
  NLL TS     1.281±0.015 1.191±0.019 1.177±0.018 1.176±0.016 1.177±0.017 1.177±0.017 1.176±0.017 1.176±0.017
  AURC TS%   12.94±0.41 11.25±0.36 10.99±0.36 10.98±0.32 11.00±0.32 11.00±0.32 11.00±0.33 11.00±0.33
  temp       0.691±0.002 0.748±0.004 0.751±0.003 0.754±0.003 0.754±0.003 0.754±0.003 0.754±0.003 0.754±0.003
  GFLOPs     0.161 0.281 0.400 0.519 0.638 0.758 0.877 0.996
  lat bs1 ms 3.40±0.25 6.23±0.42 8.95±0.57 11.68±0.56 14.34±0.21 16.87±0.30 19.08±0.85 21.82±1.33
  lat bs256 ms 4.24±0.78 10.85±1.91 17.98±3.53 24.79±4.82 31.33±5.52 39.21±8.46 45.92±9.26 52.26±9.80
  holdout%   64.0±0.2 66.5±0.7 66.7±0.3 66.7±0.4 66.8±0.5 66.9±0.5 66.8±0.5 66.8±0.4
  final train loss 3.279±0.019 | AMP bf16
  best-step acc 67.90±0.48 | last-step acc 67.80±0.52
  AUC maxprob 67.40±0.58 entropy 67.23±0.57 fixed_depth 67.59±0.47
  oracle full 67.80±0.52% @ 0.445±0.005 GFLOPs | any 70.72±0.59% @ 0.177±0.001 GFLOPs

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

[halting c100_vcnet_adapt_blt] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                          AUC       acc@t1       acc@t2       acc@t3    cost@full    cost@best
  cost@full / cost@best = compute needed to match the full-depth / best fixed-depth accuracy, % of full-depth compute ('-' = never reached, blank = not computed for that policy)
  fixed_depth              68.12±0.48   67.27±0.57   68.30±0.48   68.68±0.58     50.8±9.0     55.9±6.9
  maxprob                  67.87±0.31   66.40±0.38   67.83±0.35   68.43±0.23     71.5±5.8            -
  entropy                  67.62±0.31   65.75±0.36   67.22±0.30   68.13±0.29     69.7±6.6            -
  S0-reward                68.35±0.52   67.93±0.42   68.58±0.57   68.68±0.59    44.5±14.6    53.1±14.9
  S0-reward_acc            68.35±0.51   68.18±0.43   68.59±0.52   68.64±0.54    42.4±13.9         39.7
  S0-oracle_full           67.98±0.32   67.00±0.50   68.06±0.37   68.43±0.22     73.0±9.1            -
  S0-oracle_any            68.27±0.34   68.31±0.34   68.51±0.32   68.51±0.32         32.3            -
  S3-reward                68.32±0.49   68.00±0.50   68.51±0.47   68.66±0.52     40.1±2.8     47.5±3.7
  S3-reward_acc            67.86±0.91   67.78±0.89   68.06±0.99   68.11±1.00    44.9±10.8            -
  S3-oracle_full           67.98±0.32   66.97±0.47   68.08±0.34   68.45±0.21     72.8±9.8            -
  S3-oracle_any            68.26±0.33   68.27±0.32   68.49±0.32   68.49±0.32            -            -
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0067% of one recurrent step; oracle any 71.83±0.25% @ 0.185±0.001 GFLOPs

[halting c100_vcnet_adapt_carry] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                          AUC       acc@t1       acc@t2       acc@t3    cost@full    cost@best
  cost@full / cost@best = compute needed to match the full-depth / best fixed-depth accuracy, % of full-depth compute ('-' = never reached, blank = not computed for that policy)
  fixed_depth              65.32±0.48   65.30±0.41   65.47±0.48   65.48±0.48     33.2±5.6    40.6±11.9
  maxprob                  65.17±0.48   64.66±0.47   65.17±0.43   65.39±0.48     54.0±9.6            -
  entropy                  65.07±0.50   64.46±0.53   64.89±0.44   65.23±0.44     62.4±5.7            -
  S0-reward                64.62±0.51   64.65±0.49   64.66±0.51   64.67±0.52     27.1±4.6    39.8±22.1
  S0-reward_acc            65.13±0.48   65.16±0.48   65.16±0.48   65.16±0.48     22.0±0.3     22.8±0.1
  S0-oracle_full           65.25±0.49   64.96±0.43   65.28±0.43   65.40±0.51     52.0±9.9         52.7
  S0-oracle_any            65.36±0.50   65.41±0.50   65.41±0.50   65.41±0.50     23.2±1.8         24.9
  S3-reward                63.90±0.24   63.91±0.23   63.91±0.23   63.91±0.23         22.8         23.5
  S3-reward_acc            64.98±0.61   65.02±0.62   65.02±0.62   65.02±0.62     21.7±0.5         22.3
  S3-oracle_full           65.24±0.48   64.94±0.45   65.27±0.45   65.39±0.49     53.5±7.9            -
  S3-oracle_any            65.34±0.48   65.38±0.48   65.38±0.48   65.38±0.48     23.1±2.8            -
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0072% of one recurrent step; oracle any 68.33±0.63% @ 0.170±0.000 GFLOPs

[halting c100_vcnet_adapt_lat] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                          AUC       acc@t1       acc@t2       acc@t3    cost@full    cost@best
  cost@full / cost@best = compute needed to match the full-depth / best fixed-depth accuracy, % of full-depth compute ('-' = never reached, blank = not computed for that policy)
  fixed_depth              67.59±0.47   67.39±0.48   67.83±0.47   67.90±0.48     39.8±7.4     52.1±0.0
  maxprob                  67.40±0.58   66.49±0.66   67.54±0.70   67.78±0.54     52.7±2.2            -
  entropy                  67.23±0.57   66.09±0.67   67.03±0.66   67.61±0.61     64.0±7.0            -
  S0-reward                67.59±0.45   67.50±0.41   67.70±0.40   67.75±0.45    41.9±14.8         41.4
  S0-reward_acc            67.73±0.54   67.74±0.53   67.84±0.53   67.85±0.53    30.7±11.3         31.7
  S0-oracle_full           67.49±0.57   66.88±0.65   67.59±0.61   67.78±0.52     52.9±0.5            -
  S0-oracle_any            67.70±0.50   67.79±0.50   67.80±0.49   67.80±0.49     25.6±2.1            -
  S3-reward                65.38±1.04   65.35±1.01   65.38±1.04   65.41±1.07    54.1±16.4            -
  S3-reward_acc            66.58±2.08   66.60±2.10   66.62±2.12   66.63±2.13    45.2±30.9            -
  S3-oracle_full           67.49±0.57   66.89±0.65   67.59±0.62   67.78±0.52     53.1±0.6            -
  S3-oracle_any            67.72±0.53   67.80±0.53   67.82±0.53   67.82±0.53     27.0±2.9            -
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0069% of one recurrent step; oracle any 70.72±0.59% @ 0.177±0.001 GFLOPs

[benchmark c100_vcnet_adapt_blt] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput with compaction, batch-1 latency on 500 images
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.71±0.78           1.00     56896±19427     3.639±0.256
  fixed t=1              67.27±0.56           2.00       6933±4903     7.165±0.198
  fixed t=2              68.29±0.47           3.00       5296±2702    10.868±0.199
  fixed t=3              68.68±0.55           4.00          4708±3    14.117±0.586
  fixed t=4              68.64±0.41           5.00       2882±1217    18.082±0.223
  fixed t=5              68.65±0.31           6.00          2893±2    20.980±0.203
  fixed t=6              68.56±0.28           7.00          2429±3    24.748±0.296
  fixed t=7              68.58±0.28           8.00        1885±362    28.285±0.263
  lam=0.0001             68.52±0.36      4.80±1.63       2935±2009    20.510±6.171   0.62±0.43x / 0.74±0.26x
  lam=0.0003             68.43±0.44      4.15±1.33        2976±676    18.911±2.941   0.63±0.14x / 0.76±0.11x
  lam=0.001              68.49±0.61      3.27±0.77        3925±331    14.338±2.786   0.83±0.07x / 1.01±0.21x
  lam=0.003              68.46±0.65      2.89±0.62        4975±793    13.222±1.037   1.06±0.17x / 1.07±0.11x
  lam=0.01               68.34±0.55      2.41±0.40       5407±1112    11.610±2.005   1.06±0.39x / 1.15±0.25x
  lam=0.03               68.20±0.58      2.09±0.26       6443±1313     9.174±1.705   1.80±1.15x / 1.30±0.07x
  lam=0.1                67.77±0.42      1.74±0.05       7454±2866     7.503±0.603   2.13±2.14x / 1.46±0.13x
  lam=0.3                66.74±0.45      1.40±0.08      12471±1131     5.920±0.371   2.31±1.10x / 1.21±0.08x
  [compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.71±0.75           1.00      74853±2343     3.176±0.098
  fixed t=1              67.28±0.55           2.00      34270±8568     5.844±0.193
  fixed t=2              68.34±0.44           3.00      24550±6214     8.449±0.230
  fixed t=3              68.70±0.55           4.00       20652±705    11.111±0.290
  fixed t=4              68.68±0.38           5.00      15114±3597    13.664±0.254
  fixed t=5              68.62±0.25           6.00       14145±206    16.401±0.370
  fixed t=6              68.56±0.28           7.00       12017±309    19.031±0.587
  fixed t=7              68.57±0.27           8.00       10635±476    21.960±0.885
  lam=0.0001             68.50±0.38      4.79±1.63        8667±331    15.456±4.703   0.42±0.01x / 0.78±0.29x
  lam=0.0003             68.45±0.42      4.15±1.33        8360±781    14.353±3.315   0.40±0.04x / 0.81±0.22x
  lam=0.001              68.51±0.59      3.27±0.77       8218±1201    13.941±4.055   0.40±0.05x / 0.84±0.20x
  lam=0.003              68.49±0.62      2.90±0.62       7725±1053    11.902±4.576   0.34±0.08x / 0.92±0.22x
  lam=0.01               68.40±0.57      2.41±0.40       8474±1988    10.712±3.618   0.38±0.14x / 1.01±0.31x
  lam=0.03               68.22±0.61      2.08±0.26       9193±1322     8.122±2.675   0.37±0.09x / 1.19±0.16x
  lam=0.1                67.78±0.43      1.74±0.05      10470±3863     7.841±1.920   0.44±0.15x / 1.12±0.28x
  lam=0.3                66.74±0.48      1.40±0.08      12339±4842     5.804±1.902   0.37±0.12x / 1.07±0.31x
  [cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.71±0.78           1.00               -     1.056±0.128
  fixed t=1              67.27±0.56           2.00               -     2.098±0.257
  fixed t=2              68.29±0.47           3.00               -     3.124±0.365
  fixed t=3              68.68±0.55           4.00               -     4.167±0.496
  fixed t=4              68.64±0.41           5.00               -     5.206±0.620
  fixed t=5              68.65±0.31           6.00               -     6.249±0.743
  fixed t=6              68.56±0.28           7.00               -     7.056±0.482
  fixed t=7              68.58±0.28           8.00               -     8.612±1.501
  lam=0.0001             68.52±0.36      4.80±1.63               -     5.116±1.724   -x / 0.93±0.51x
  lam=0.0003             68.43±0.44      4.15±1.33               -     4.444±1.399   -x / 1.05±0.53x
  lam=0.001              68.49±0.61      3.27±0.77               -     5.404±2.440   -x / 0.84±0.23x
  lam=0.003              68.46±0.65      2.89±0.62               -     3.101±0.692   -x / 1.42±0.53x
  lam=0.01               68.34±0.55      2.41±0.40               -     2.609±0.433   -x / 1.46±0.18x
  lam=0.03               68.20±0.58      2.09±0.26               -     2.217±0.265   -x / 1.56±0.20x
  lam=0.1                67.77±0.42      1.74±0.05               -     1.841±0.042   -x / 1.70±0.23x
  lam=0.3                66.74±0.45      1.40±0.08               -     1.485±0.106   -x / 1.41±0.11x

[benchmark c100_vcnet_adapt_carry] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput with compaction, batch-1 latency on 500 images
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.69±0.61           1.00       68710±443     3.564±0.055
  fixed t=1              65.33±0.41           2.00       29199±197     6.413±0.146
  fixed t=2              65.43±0.52           3.00      14376±6238     9.236±0.159
  fixed t=3              65.37±0.55           4.00       7735±4465    12.137±0.447
  fixed t=4              65.43±0.56           5.00       8094±3516    14.891±0.758
  fixed t=5              65.38±0.55           6.00       5000±2921    18.082±0.420
  fixed t=6              65.44±0.51           7.00          2803±3    21.460±0.276
  fixed t=7              65.40±0.53           8.00       3665±2134    23.761±0.718
  lam=0.0001             64.91±0.91      1.92±0.90     15698±12089     8.251±2.168   0.65±0.51x / 0.79±0.38x
  lam=0.0003             64.89±0.92      1.79±0.83     18254±10691     8.184±2.739   0.73±0.41x / 0.83±0.48x
  lam=0.001              64.89±0.92      1.66±0.72     15963±11948     6.707±2.535   0.69±0.57x / 0.99±0.45x
  lam=0.003              64.89±0.92      1.55±0.59     29258±22394     7.015±3.222   1.11±0.66x / 0.99±0.52x
  lam=0.01               64.84±0.85      1.45±0.48     28523±22946     6.830±3.032   0.62±0.24x / 0.85±0.27x
  lam=0.03               64.84±0.85      1.36±0.37      20960±9663     6.890±2.674   0.52±0.22x / 0.83±0.28x
  lam=0.1                64.71±0.69      1.23±0.22     30044±21997     6.066±2.288   0.67±0.21x / 0.94±0.30x
  lam=0.3                64.22±0.55      1.09±0.08     31701±21315     5.411±1.377   0.72±0.22x / 1.02±0.30x
  [compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.70±0.57           1.00     52313±18697     3.178±0.042
  fixed t=1              65.28±0.42           2.00      30018±6179     5.388±0.158
  fixed t=2              65.46±0.53           3.00      23743±7636     7.495±0.344
  fixed t=3              65.42±0.54           4.00      18595±5667     9.558±0.725
  fixed t=4              65.41±0.55           5.00      18174±4839    11.534±0.715
  fixed t=5              65.45±0.55           6.00      15260±3901    13.626±0.560
  fixed t=6              65.41±0.52           7.00      13248±3356    15.492±0.067
  fixed t=7              65.40±0.51           8.00      11660±2912    17.827±0.588
  lam=0.0001             64.85±0.88      1.92±0.90     18400±11995     6.572±1.265   0.49±0.07x / 0.83±0.40x
  lam=0.0003             64.85±0.89      1.79±0.83     21632±12377     8.436±2.490   0.59±0.29x / 0.64±0.21x
  lam=0.001              64.86±0.90      1.66±0.72     19102±11925     6.953±0.890   0.50±0.09x / 0.75±0.22x
  lam=0.003              64.87±0.92      1.55±0.59     23007±10388     6.808±1.014   0.66±0.26x / 0.77±0.21x
  lam=0.01               64.83±0.85      1.45±0.48      21885±8978     6.096±1.212   0.57±0.11x / 0.76±0.14x
  lam=0.03               64.83±0.87      1.36±0.37     35450±23661     6.209±2.203   0.84±0.18x / 0.77±0.08x
  lam=0.1                64.69±0.70      1.23±0.22     30109±28725     5.935±2.071   0.63±0.20x / 0.81±0.07x
  lam=0.3                64.24±0.57      1.09±0.08     34247±26535     4.566±1.604   0.77±0.10x / 1.07±0.34x
  [cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.69±0.61           1.00               -     1.432±0.645
  fixed t=1              65.33±0.41           2.00               -     3.065±1.124
  fixed t=2              65.43±0.52           3.00               -     3.599±1.799
  fixed t=3              65.37±0.55           4.00               -     4.701±2.345
  fixed t=4              65.43±0.56           5.00               -     5.810±2.897
  fixed t=5              65.38±0.55           6.00               -     6.926±3.454
  fixed t=6              65.44±0.51           7.00               -     8.022±3.995
  fixed t=7              65.40±0.53           8.00               -     6.511±0.002
  lam=0.0001             64.91±0.91      1.92±0.90               -     1.891±0.796   -x / 1.26±0.26x
  lam=0.0003             64.89±0.92      1.79±0.83               -     2.226±1.440   -x / 1.15±0.27x
  lam=0.001              64.89±0.92      1.66±0.72               -     1.665±0.623   -x / 1.43±0.39x
  lam=0.003              64.89±0.92      1.55±0.59               -     1.571±0.519   -x / 1.51±0.47x
  lam=0.01               64.84±0.85      1.45±0.48               -     1.489±0.430   -x / 1.42±0.55x
  lam=0.03               64.84±0.85      1.36±0.37               -     1.403±0.328   -x / 1.51±0.67x
  lam=0.1                64.71±0.69      1.23±0.22               -     1.295±0.206   -x / 1.66±0.86x
  lam=0.3                64.22±0.55      1.09±0.08               -     1.159±0.064   -x / 1.92±1.20x

[benchmark c100_vcnet_adapt_lat] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput with compaction, batch-1 latency on 500 images
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              64.74±0.49           1.00      68591±1100     3.604±0.016
  fixed t=1              67.38±0.47           2.00       28335±129     6.813±0.219
  fixed t=2              67.82±0.52           3.00      13925±6021     9.887±0.448
  fixed t=3              67.88±0.51           4.00        12542±77    13.230±0.279
  fixed t=4              67.77±0.52           5.00       5895±3403    16.896±0.297
  fixed t=5              67.79±0.53           6.00       4826±2781    19.644±0.674
  fixed t=6              67.83±0.50           7.00       4106±2407    22.815±1.057
  fixed t=7              67.82±0.54           8.00       3532±2030    26.099±0.541
  lam=0.0001             67.72±0.66      3.66±2.09       5891±2662   18.517±13.496   0.36±0.12x / 0.75±0.33x
  lam=0.0003             67.63±0.74      2.94±1.36        7279±713    15.452±6.916   0.55±0.37x / 0.64±0.27x
  lam=0.001              67.58±0.83      2.53±1.07       7398±1869    10.592±3.968   0.45±0.24x / 0.95±0.09x
  lam=0.003              67.50±0.94      2.23±1.00      12485±5152    12.464±7.267   0.97±0.64x / 0.83±0.29x
  lam=0.01               67.40±1.08      1.90±0.71       8764±2582    10.675±5.498   0.60±0.24x / 0.94±0.29x
  lam=0.03               67.25±1.38      1.66±0.49      16577±3878     7.170±1.522   1.16±0.54x / 1.25±0.20x
  lam=0.1                66.90±1.68      1.40±0.27      17920±5470     7.096±2.863   1.21±0.47x / 1.33±0.30x
  lam=0.3                66.27±1.50      1.18±0.13      22357±8832     5.072±0.632   0.79±0.31x / 1.36±0.15x
  [compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              64.73±0.47           1.00     63201±19459     3.265±0.161
  fixed t=1              67.39±0.47           2.00      30074±8587     5.992±0.359
  fixed t=2              67.85±0.50           3.00      23508±5379     8.840±0.876
  fixed t=3              67.86±0.52           4.00      18075±3795    11.380±0.635
  fixed t=4              67.78±0.54           5.00      15654±3569    13.930±1.329
  fixed t=5              67.80±0.50           6.00      11601±3127    16.298±1.176
  fixed t=6              67.83±0.51           7.00       9948±2506    18.891±1.554
  fixed t=7              67.85±0.50           8.00         7524±17    21.306±1.142
  lam=0.0001             67.72±0.66      3.66±2.09       8022±1443    14.200±6.791   0.35±0.06x / 0.72±0.29x
  lam=0.0003             67.64±0.69      2.95±1.36        7565±806    11.814±4.258   0.33±0.08x / 0.81±0.27x
  lam=0.001              67.57±0.82      2.53±1.08       9234±1868     9.684±3.218   0.43±0.16x / 0.85±0.20x
  lam=0.003              67.48±0.94      2.23±1.00      14168±7081     9.193±3.920   0.64±0.29x / 0.97±0.51x
  lam=0.01               67.40±1.09      1.90±0.71      14547±7924     8.154±2.920   0.65±0.32x / 1.08±0.55x
  lam=0.03               67.23±1.38      1.66±0.49      16054±6884     7.496±2.186   0.71±0.26x / 1.13±0.50x
  lam=0.1                66.91±1.73      1.40±0.27      20235±8543     5.791±2.150   0.88±0.26x / 1.45±0.47x
  lam=0.3                66.26±1.52      1.18±0.13     23897±11916     5.085±1.525   0.90±0.56x / 1.24±0.32x
  [cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              64.74±0.49           1.00               -     1.433±0.645
  fixed t=1              67.38±0.47           2.00               -     2.856±1.137
  fixed t=2              67.82±0.52           3.00               -     4.892±1.901
  fixed t=3              67.88±0.51           4.00               -     6.447±2.502
  fixed t=4              67.77±0.52           5.00               -     7.998±3.100
  fixed t=5              67.79±0.53           6.00               -     9.559±3.705
  fixed t=6              67.83±0.50           7.00               -     9.803±3.687
  fixed t=7              67.82±0.54           8.00               -     6.998±0.000
  lam=0.0001             67.72±0.66      3.66±2.09               -     5.078±2.051   -x / 1.08±0.23x
  lam=0.0003             67.63±0.74      2.94±1.36               -     4.502±2.085   -x / 0.95±0.49x
  lam=0.001              67.58±0.83      2.53±1.07               -     3.215±0.893   -x / 1.36±0.73x
  lam=0.003              67.50±0.94      2.23±1.00               -     2.256±0.921   -x / 1.64±0.20x
  lam=0.01               67.40±1.08      1.90±0.71               -     2.606±0.841   -x / 1.53±0.75x
  lam=0.03               67.25±1.38      1.66±0.49               -     2.785±1.341   -x / 1.56±0.89x
  lam=0.1                66.90±1.68      1.40±0.27               -     1.458±0.258   -x / 2.52±0.89x
  lam=0.3                66.27±1.50      1.18±0.13               -     2.251±1.801   -x / 1.57±0.73x

[benchmark c100_vcnet_orig] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy None | batch 256 throughput with compaction, batch-1 latency on 500 images

  [eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              61.00±0.94           1.00      37825±5161     3.589±0.085
  fixed t=1              64.56±0.93           2.00      17658±9993     6.630±0.453
  [compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              61.01±0.92           1.00      53935±21401     3.112±0.075
  fixed t=1              64.60±0.97           2.00        26479±42     5.281±0.030
  [cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              61.00±0.94           1.00               -     1.781±0.694
  fixed t=1              64.56±0.93           2.00               -     3.204±1.247
===== END FSV SUMMARY =====
```
