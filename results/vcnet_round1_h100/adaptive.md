# VCNet round 1, adaptive runs (H100, pasted by the user)

Code 0.9.0 (`dev/vcnet`). Adaptive VCNet runs only: `c100_vcnet_adapt` and `c100_vcnet_adapt_carry`, 3 seeds each,
with `--halting`, then `scripts/benchmark_adaptive.py` on both. The original-model runs
(`c100_vcnet_orig`, `c100_vcnet_orig_ff`) were pasted separately: see [`original.md`](original.md).

Notes on reading it:

* The predictive-coding loop saturates after one correction. adapt: 63.75 (t=0) -> 64.91 (t=1), then at most +0.02.
  carry: 63.69 -> 65.30 (t=1) -> 65.44 (t=2), then flat. Two steps (0.275 GFLOPs, 29% of the 8-step cost) already
  give the full-depth accuracy; steps 3-8 add nothing.
* The carried AIT state helps a little: +0.4-0.6 points from t=1 on, with a smaller seed spread.
* Raw ECE ~12.7% with fitted temperatures ~0.745 (< 1): the model is under-confident (mixup/CutMix + Brier +
  Gatekeeper); per-step temperature scaling brings ECE to ~2.5%.
* The halting table reports AUC over [0.161, 0.955] GFLOPs and accuracy at 2-4 steps, where every curve is flat,
  so it hides the only region where halting can matter (1-2 steps). cost@ columns average only over the seeds
  that reach the target.
* Benchmark, adapt: for every lambda in the benchmark grid the S0-reward_acc policy stops all images at step 0
  (63.75%). carry: 1.09-1.66 mean steps, 64.22-64.89%. Against random mixing of fixed depths 0 and 1 at the same
  mean steps (63.69 + f x 1.64), the carry policy is +0.4 to +0.6 points at 1.09-1.36 steps and -0.1 at 1.66 steps;
  it never reaches the 2-step accuracy (65.33%).
* Batch-1 CUDA graphs, carry: fixed t=0 1.049 ms, t=1 1.818 ms; adaptive 1.16-1.67 ms.
  Batch 256: adaptive is not faster (speed-up column 0.35-0.83x). Several bs256 throughputs have standard
  deviations of 30-70% of the mean (and the carry `lat bs256` row from evaluate.py has +-36 ms), so at least one
  seed's timing was disturbed.

```
===== BEGIN FSV SUMMARY =====
code 0.9.0+dec7d6e42181 | NVIDIA H100 NVL | torch 2.6.0+cu124 | cpus 8/48

[c100_vcnet_adapt] seeds 0,1,2 | params 1.02M | train 56.0±2.6 min | epoch 16.6±0.8s
  test acc%  63.75±1.13 64.91±0.94 64.85±1.11 64.87±0.95 64.93±1.03 64.89±1.06 64.93±1.00 64.88±1.03
  ECE raw%   12.65±0.80 12.77±0.37 12.64±0.59 12.66±0.44 12.72±0.51 12.67±0.54 12.71±0.48 12.66±0.51
  ECE TS%    2.65±0.55 2.50±0.34 2.58±0.43 2.59±0.35 2.49±0.34 2.54±0.37 2.47±0.35 2.55±0.38
  NLL TS     1.341±0.053 1.280±0.046 1.278±0.045 1.278±0.045 1.278±0.045 1.278±0.045 1.278±0.045 1.278±0.045
  AURC TS%   13.70±1.05 12.73±0.76 12.72±0.82 12.72±0.78 12.71±0.80 12.71±0.80 12.71±0.79 12.72±0.80
  temp       0.747±0.006 0.745±0.005 0.745±0.005 0.745±0.005 0.745±0.005 0.745±0.005 0.745±0.005 0.745±0.005
  GFLOPs     0.161 0.275 0.388 0.502 0.615 0.729 0.842 0.955
  lat bs1 ms 3.25±0.10 5.71±0.11 7.98±0.24 10.22±0.18 12.60±0.16 14.97±0.21 17.15±0.13 19.40±0.16
  lat bs256 ms 3.46±0.07 8.65±0.01 14.01±0.00 19.38±0.01 24.73±0.01 30.09±0.00 35.42±0.00 40.79±0.01
  holdout%   62.8±1.0 64.3±1.0 64.5±1.0 64.5±0.9 64.5±1.0 64.5±0.9 64.5±0.9 64.6±0.9
  final train loss 3.379±0.018 | AMP bf16
  best-step acc 64.98±1.02 | last-step acc 64.88±1.03
  AUC maxprob 64.76±1.08 entropy 64.67±1.11 fixed_depth 64.88±1.01
  oracle full 64.88±1.03% @ 0.448±0.008 GFLOPs | any 67.73±1.23% @ 0.169±0.000 GFLOPs

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

[halting c100_vcnet_adapt] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                          AUC       acc@t1       acc@t2       acc@t3    cost@full    cost@best
  cost@full / cost@best = compute needed to match the full-depth / best fixed-depth accuracy, % of full-depth compute ('-' = never reached, blank = not computed for that policy)
  fixed_depth              64.88±1.01   64.91±0.94   64.97±1.03   64.98±1.02     30.8±5.1    40.6±11.9
  maxprob                  64.76±1.08   64.47±1.22   64.75±1.13   64.89±1.06     47.7±5.1            -
  entropy                  64.67±1.11   64.31±1.21   64.55±1.15   64.76±1.08     60.9±7.7         54.5
  S0-reward                64.32±0.74   64.34±0.74   64.35±0.72   64.35±0.72     23.8±1.1         24.0
  S0-reward_acc            64.47±0.90   64.48±0.89   64.49±0.89   64.49±0.89     21.8±0.2     24.0±1.1
  S0-oracle_full           64.80±1.08   64.62±1.15   64.83±1.08   64.91±1.04     44.8±7.0         50.5
  S0-oracle_any            64.88±1.03   64.91±1.03   64.91±1.03   64.91±1.03     22.3±2.2            -
  S3-reward                63.75±1.13   63.75±1.13   63.75±1.13   63.75±1.13            -            -
  S3-reward_acc            63.85±1.29   63.86±1.30   63.86±1.30   63.86±1.30            -            -
  S3-oracle_full           64.80±1.08   64.62±1.17   64.80±1.11   64.90±1.03     45.1±5.8            -
  S3-oracle_any            64.89±1.04   64.92±1.04   64.92±1.04   64.92±1.04     22.0±1.7         21.9
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0072% of one recurrent step; oracle any 67.73±1.23% @ 0.169±0.000 GFLOPs

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

[benchmark c100_vcnet_adapt] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput with compaction, batch-1 latency on 500 images
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.75±1.13           1.00       72106±981     3.429±0.069
  fixed t=1              64.91±0.92           2.00       29159±219     6.065±0.151
  fixed t=2              64.86±1.11           3.00      14666±5591     8.857±0.176
  fixed t=3              64.87±0.97           4.00      10389±4485    11.300±0.318
  fixed t=4              64.91±1.02           5.00       10218±103    13.895±0.791
  fixed t=5              64.91±1.03           6.00       6682±2919    16.752±0.627
  fixed t=6              64.92±1.02           7.00         7116±71    19.119±0.162
  fixed t=7              64.88±1.05           8.00         6175±57    21.972±0.319
  lam=0.001              63.75±1.13      1.00±0.00        57291±54     4.203±0.074   0.79±0.01x / 0.82±0.03x
  lam=0.01               63.75±1.13      1.00±0.00      58686±1925     4.131±0.142   0.81±0.02x / 0.83±0.04x
  lam=0.03               63.75±1.13      1.00±0.00      59480±1889     4.349±0.239   0.82±0.02x / 0.79±0.06x
  lam=0.1                63.75±1.13      1.00±0.00      57250±1957     4.162±0.099   0.79±0.04x / 0.82±0.03x
  lam=0.3                63.75±1.13      1.00±0.00      57682±1675     4.151±0.064   0.80±0.03x / 0.83±0.03x
  [compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.73±1.12           1.00      80422±1653     3.013±0.051
  fixed t=1              64.90±0.93           2.00      45897±1006     5.063±0.103
  fixed t=2              64.87±1.09           3.00       33156±640     7.070±0.122
  fixed t=3              64.91±0.98           4.00       25268±588     9.028±0.053
  fixed t=4              64.90±1.03           5.00       21542±586    10.874±0.023
  fixed t=5              64.88±1.04           6.00       17986±147    13.265±0.636
  fixed t=6              64.90±1.01           7.00       15880±341    15.127±0.360
  fixed t=7              64.89±1.01           8.00       13873±141    17.050±0.166
  lam=0.001              63.73±1.12      1.00±0.00     56032±20728     4.251±1.218   0.69±0.25x / 0.74±0.18x
  lam=0.01               63.73±1.12      1.00±0.00     54701±19460     4.241±1.185   0.68±0.23x / 0.74±0.18x
  lam=0.03               63.73±1.12      1.00±0.00     46176±18341     4.211±1.230   0.57±0.21x / 0.75±0.19x
  lam=0.1                63.73±1.12      1.00±0.00     54912±19644     4.224±1.216   0.68±0.24x / 0.75±0.18x
  lam=0.3                63.73±1.12      1.00±0.00     55345±19994     4.258±1.313   0.69±0.24x / 0.75±0.19x
  [cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.75±1.13           1.00               -     1.433±0.647
  fixed t=1              64.91±0.92           2.00               -     2.537±1.212
  fixed t=2              64.86±1.11           3.00               -     3.595±1.802
  fixed t=3              64.87±0.97           4.00               -     3.768±0.741
  fixed t=4              64.91±1.02           5.00               -     4.130±0.000
  fixed t=5              64.91±1.03           6.00               -     4.913±0.002
  fixed t=6              64.92±1.02           7.00               -     5.697±0.006
  fixed t=7              64.88±1.05           8.00               -     6.607±0.212
  lam=0.001              63.75±1.13      1.00±0.00               -     1.085±0.002   -x / 1.32±0.59x
  lam=0.01               63.75±1.13      1.00±0.00               -     1.086±0.001   -x / 1.32±0.59x
  lam=0.03               63.75±1.13      1.00±0.00               -     1.099±0.016   -x / 1.30±0.59x
  lam=0.1                63.75±1.13      1.00±0.00               -     1.614±0.914   -x / 0.92±0.09x
  lam=0.3                63.75±1.13      1.00±0.00               -     1.291±0.351   -x / 1.08±0.18x

[benchmark c100_vcnet_adapt_carry] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput with compaction, batch-1 latency on 500 images
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.69±0.61           1.00      69973±1301     3.474±0.144
  fixed t=1              65.33±0.41           2.00       29217±113     6.129±0.313
  fixed t=2              65.43±0.52           3.00      15347±2588     9.125±0.412
  fixed t=3              65.37±0.55           4.00       9179±3977    11.748±0.422
  fixed t=4              65.43±0.56           5.00       7234±3089    14.779±0.952
  fixed t=5              65.38±0.55           6.00       5014±2939    17.130±0.905
  fixed t=6              65.44±0.51           7.00       4247±2493    20.367±1.401
  fixed t=7              65.40±0.53           8.00       4939±2161    22.888±0.652
  lam=0.001              64.89±0.92      1.66±0.72     28365±24937     6.469±2.422   0.73±0.08x / 0.99±0.46x
  lam=0.01               64.84±0.85      1.45±0.48     33990±21123     5.818±1.681   0.77±0.05x / 0.91±0.21x
  lam=0.03               64.84±0.85      1.36±0.37     32983±19952     5.438±1.258   0.75±0.04x / 0.96±0.20x
  lam=0.1                64.71±0.69      1.23±0.22     34901±20748     5.026±0.833   0.80±0.04x / 1.03±0.23x
  lam=0.3                64.22±0.55      1.09±0.08     35514±18905     4.425±0.334   0.83±0.11x / 1.17±0.28x
  [compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.70±0.57           1.00      80049±1215     3.018±0.066
  fixed t=1              65.28±0.42           2.00      46574±1586     5.054±0.124
  fixed t=2              65.46±0.53           3.00      33067±1193     6.967±0.143
  fixed t=3              65.42±0.54           4.00       25860±546     8.901±0.129
  fixed t=4              65.41±0.55           5.00       21216±436    11.326±0.290
  fixed t=5              65.45±0.55           6.00      15728±4319    13.512±0.885
  fixed t=6              65.41±0.52           7.00      11620±3878    15.466±0.510
  fixed t=7              65.40±0.51           8.00          8305±3    18.053±0.759
  lam=0.001              64.86±0.90      1.66±0.72     19153±11836     8.096±2.371   0.35±0.06x / 0.63±0.24x
  lam=0.01               64.83±0.85      1.45±0.48      25548±7861     6.444±0.828   0.46±0.13x / 0.68±0.19x
  lam=0.03               64.83±0.87      1.36±0.37      25492±7564     5.909±1.099   0.46±0.12x / 0.75±0.20x
  lam=0.1                64.69±0.70      1.23±0.22     37615±26434     4.955±1.624   0.61±0.22x / 0.90±0.18x
  lam=0.3                64.24±0.57      1.09±0.08     38457±25017     4.635±1.563   0.63±0.19x / 0.98±0.30x
  [cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              63.69±0.61           1.00               -     1.049±0.019
  fixed t=1              65.33±0.41           2.00               -     1.818±0.083
  fixed t=2              65.43±0.52           3.00               -     2.561±0.001
  fixed t=3              65.37±0.55           4.00               -     3.351±0.001
  fixed t=4              65.43±0.56           5.00               -     4.141±0.001
  fixed t=5              65.38±0.55           6.00               -     4.933±0.001
  fixed t=6              65.44±0.51           7.00               -     5.722±0.001
  fixed t=7              65.40±0.53           8.00               -     6.511±0.001
  lam=0.001              64.89±0.92      1.66±0.72               -     1.665±0.625   -x / 1.13±0.43x
  lam=0.01               64.84±0.85      1.45±0.48               -     1.487±0.432   -x / 1.06±0.16x
  lam=0.03               64.84±0.85      1.36±0.37               -     1.405±0.324   -x / 1.11±0.17x
  lam=0.1                64.71±0.69      1.23±0.22               -     1.294±0.206   -x / 1.20±0.22x
  lam=0.3                64.22±0.55      1.09±0.08               -     1.159±0.063   -x / 1.34±0.35x
===== END FSV SUMMARY =====
```
