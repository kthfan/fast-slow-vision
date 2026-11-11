# Round 9 (H100, pasted by the user)

Output of `python scripts/summarize.py c100_bvit_s_quadw_lr5e4 c100_vit_s_8blk c100_bvit_b --sections halting bench`
after training O3 step predictors (`--halting-only --halting-args "--feature-sets S0 --objectives --o3-ks 0 1"`),
re-evaluating c100_bvit_b with halting, and `python scripts/benchmark_adaptive.py c100_bvit_s_quadw_lr5e4
c100_vit_s_8blk c100_bvit_b --seeds 0 1 2` (code 0.8.2, all 23 tests incl. the GPU CUDA-graph test passed).

Caveat on the speed-up column: when an adaptive setting is more accurate than every fixed depth (c100_bvit_b,
lam=0.001: 58.07-58.08% vs at most 58.04%), code 0.8.2 compared it with the slowest (full-depth) setting, which
inflates the ratio (1.71x, 2.05x, 1.46x, 1.19x, 1.13x, 1.12x, 1.90x). From 0.8.3 such rows are compared with the
most accurate fixed depth and marked with `*`. Some bs1 CUDA-graph entries have very large standard deviations
(e.g. 1.354±1.393 ms): one seed's measurement was disturbed; medians per seed are otherwise ~0.4-0.6 ms.

```
===== BEGIN FSV SUMMARY =====
code 0.6.1+447e8f02f81a | NVIDIA H100 NVL | torch 2.6.0+cu124 | cpus 8/48
NOTE: runs use different code versions: ['0.2.0+4533fb7e05f9', '0.6.1+447e8f02f81a', 'None']

[halting c100_bvit_b] backbone seeds 1 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                          AUC       acc@t1       acc@t2       acc@t3    cost@full    cost@best
  fixed_depth                   56.38        55.59        57.85        58.03         35.2         50.0
  maxprob                       54.51        49.73        54.04        56.02         74.8            -
  entropy                       54.17        48.72        53.14        55.58         75.6            -
  S0-reward_acc                 56.79        56.32        57.97        58.03         29.0         38.4
  S0-o3k0-reward_acc            56.47        55.51        57.11        57.76         42.2         47.9
  S0-o3k1-reward_acc            56.42        55.63        57.90        57.96         30.4         39.6
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0004% of one recurrent step; oracle any 61.78% @ 1.296 GFLOPs

[halting c100_bvit_s_quadw_lr5e4] backbone seeds 3 (rows unchanged from rounds 6-7 omitted)
  method                          AUC       acc@t1       acc@t2       acc@t3    cost@full    cost@best
  fixed_depth              57.54±0.39   52.16±0.44   59.40±0.44   61.40±0.64     62.0±8.4     79.2±7.2
  maxprob                  55.83±0.17   45.88±0.33   54.70±0.21   59.06±0.14     83.4±6.9            -
  S0-reward_acc            58.26±0.40   52.55±0.40   60.89±0.59   62.01±0.42     43.2±0.2         46.7
  S0-oracle_any            58.25±0.29   52.70±0.17   60.81±0.52   61.96±0.28     48.3±2.4            -
  S0-o3k0-reward_acc       57.54±0.47   51.85±0.31   58.69±0.73   60.91±0.57     59.3±1.6            -
  S0-o3k1-reward_acc       57.45±0.56   52.15±0.44   59.64±0.79   61.31±0.70         51.2            -
  lambda sweep: mean steps non-monotone in 0.1 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0257% of one recurrent step; oracle any 64.76±0.36% @ 0.408±0.003 GFLOPs

[halting c100_vit_s_8blk] backbone seeds 3 (rows unchanged from rounds 6-7 omitted)
  method                          AUC       acc@t1       acc@t2       acc@t3    cost@full    cost@best
  fixed_depth              58.30±0.39   53.17±0.45   58.98±0.49   61.28±0.11     95.3±8.1     95.8±7.2
  maxprob                  57.67±0.45   49.15±0.39   56.55±0.29   60.47±0.57     72.4±2.4     72.6±2.3
  S0-reward_acc            59.39±0.47   54.30±0.53   60.95±0.51   62.27±0.50     53.8±2.5     54.1±2.0
  S0-oracle_any            59.44±0.50   54.35±0.42   61.00±0.27   62.42±0.67     51.1±0.2     51.4±0.6
  S0-o3k0-reward_acc       58.16±0.48   53.09±0.44   58.12±0.84   60.82±0.50            -            -
  S0-o3k1-reward_acc       58.17±0.38   53.16±0.44   58.86±0.62   61.50±0.52            -            -
  lambda sweep: mean steps non-monotone in 0.3 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0257% of one recurrent step; oracle any 66.40±0.44% @ 0.422±0.004 GFLOPs

[benchmark c100_bvit_b] seeds 1 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput with compaction, batch-1 latency on 500 images
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  fixed t=0                   40.08           1.00          184545           0.567
  fixed t=1                   55.60           2.00          102558           0.883
  fixed t=2                   57.87           3.00           71149           1.232
  fixed t=3                   58.04           4.00           54338           1.588
  fixed t=4                   57.84           5.00           43370           1.955
  fixed t=5                   57.86           6.00           36857           2.244
  fixed t=6                   57.60           7.00           31385           2.642
  fixed t=7                   57.47           8.00           27532           2.956
  lam=0.001                   58.07           3.43           47022           4.678   1.71x / 0.63x
  lam=0.01                    58.03           3.11           48526           3.338   0.89x / 0.48x
  lam=0.03                    58.01           2.87           47638           3.133   0.88x / 0.51x
  lam=0.1                     57.73           2.41           48385           2.640   0.68x / 0.47x
  lam=0.3                     56.59           2.05           49480           2.299   0.70x / 0.54x
  [compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  fixed t=0                   40.18           1.00          211229           0.578
  fixed t=1                   55.60           2.00          119627           0.926
  fixed t=2                   57.81           3.00           82943           1.316
  fixed t=3                   58.05           4.00           63587           1.605
  fixed t=4                   57.84           5.00           51752           1.921
  fixed t=5                   57.83           6.00           43157           2.225
  fixed t=6                   57.57           7.00           37517           2.897
  fixed t=7                   57.47           8.00           32923           2.861
  lam=0.001                   58.04           3.43           55971           3.028   0.88x / 0.53x
  lam=0.01                    57.99           3.12           56429           2.765   0.89x / 0.58x
  lam=0.03                    57.94           2.87           56352           2.527   0.89x / 0.63x
  lam=0.1                     57.63           2.41           58159           2.190   0.70x / 0.60x
  lam=0.3                     56.47           2.05           58125           1.866   0.70x / 0.70x
  [o3k0-eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  lam=0.001                   58.03           3.83           51185           2.381   0.94x / 0.67x
  lam=0.01                    57.90           3.71           52831           2.377   0.97x / 0.67x
  lam=0.03                    57.84           3.48           55416           2.311   0.78x / 0.53x
  lam=0.1                     57.20           3.01           62109           2.099   0.87x / 0.59x
  lam=0.3                     55.08           1.88           80521           1.682   0.79x / 0.52x
  [o3k0-compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  lam=0.001                   57.99           3.83           59370           2.486   0.93x / 0.65x
  lam=0.01                    57.89           3.71           60623           2.482   0.95x / 0.65x
  lam=0.03                    57.77           3.48           63999           2.359   0.77x / 0.56x
  lam=0.1                     57.07           3.01           71316           2.183   0.86x / 0.60x
  lam=0.3                     55.05           1.87           84431           6.515   0.71x / 0.14x
  [o3k1-eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  lam=0.001                   58.08           3.23           56372           2.484   2.05x / 1.19x
  lam=0.01                    57.95           2.97           58831           2.428   1.08x / 0.65x
  lam=0.03                    57.82           2.79           59447           2.330   0.84x / 0.53x
  lam=0.1                     57.52           2.52           60340           2.174   0.85x / 0.57x
  lam=0.3                     56.88           2.23           61022           2.065   0.86x / 0.60x
  [o3k1-compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  lam=0.001                   58.08           3.23           62563           2.561   1.90x / 1.12x
  lam=0.01                    57.96           2.97           62323           2.505   0.98x / 0.64x
  lam=0.03                    57.75           2.79           63049           2.414   0.76x / 0.55x
  lam=0.1                     57.48           2.52           64276           2.264   0.77x / 0.58x
  lam=0.3                     56.84           2.23           66560           2.137   0.80x / 0.62x
  [cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  fixed t=0                   40.08           1.00               -           0.147
  fixed t=1                   55.60           2.00               -           0.255
  fixed t=2                   57.87           3.00               -           0.358
  fixed t=3                   58.04           4.00               -           0.469
  fixed t=4                   57.84           5.00               -           0.550
  fixed t=5                   57.86           6.00               -           0.638
  fixed t=6                   57.60           7.00               -           0.732
  fixed t=7                   57.47           8.00               -           0.834
  lam=0.001                   58.07           3.43               -           0.737   -x / 1.13x
  lam=0.01                    58.03           3.11               -           0.674   -x / 0.70x
  lam=0.03                    58.01           2.87               -           0.631   -x / 0.74x
  lam=0.1                     57.73           2.41               -           0.532   -x / 0.67x
  lam=0.3                     56.59           2.05               -           0.461   -x / 0.78x
  [o3k0-cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  lam=0.001                   58.03           3.83               -           0.580   -x / 0.81x
  lam=0.01                    57.90           3.71               -           0.564   -x / 0.83x
  lam=0.03                    57.84           3.48               -           0.486   -x / 0.74x
  lam=0.1                     57.20           3.01               -           0.442   -x / 0.81x
  lam=0.3                     55.08           1.88               -           0.326   -x / 0.78x
  [o3k1-cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1)
  lam=0.001                   58.08           3.23               -           0.572   -x / 1.46x
  lam=0.01                    57.95           2.97               -           0.550   -x / 0.85x
  lam=0.03                    57.82           2.79               -           0.498   -x / 0.72x
  lam=0.1                     57.52           2.52               -           0.438   -x / 0.82x
  lam=0.3                     56.88           2.23               -           0.410   -x / 0.87x

[benchmark c100_bvit_s_quadw_lr5e4] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  fixed t=4 61.97±0.40 98112±891 img/s 1.825±0.050 ms; fixed t=7 61.93±0.24 62617±385 img/s 2.797±0.036 ms
  lam=0.03 (eager)           61.95±0.42 3.70±0.05 steps 44141±1085 img/s 3.809±0.051 ms  0.48x / 0.51x
  lam=0.03 (compiled)        61.98±0.41 3.70±0.05 steps 43410±12982 img/s 4.858±1.642 ms 0.40x / 0.43x
  lam=0.03 (o3k0-eager)      61.68±0.59 4.58±0.18 steps 81659±2478 img/s 2.542±0.066 ms  0.83x / 0.72x
  lam=0.03 (o3k0-compiled)   61.74±0.57 4.58±0.18 steps 76031±1658 img/s 2.670±0.145 ms  0.65x / 0.68x
  lam=0.03 (o3k1-eager)      61.15±0.74 3.78±0.33 steps 70234±4928 img/s 2.551±0.099 ms  0.58x / 0.59x
  [cudagraph-bs1] fixed t=0..7 ms: 0.124 0.211 0.297 0.383 0.469 0.554 0.641 0.711
  lam=0.001 0.891±0.055 ms 0.53x | lam=0.01 0.783±0.021 0.63x | lam=0.03 0.719±0.011 0.69x | lam=0.1 0.632±0.011 0.70x | lam=0.3 2.525±3.430 0.49x
  [o3k0-cudagraph-bs1] lam=0.001 2.167±1.378 | 0.01 1.342±1.386 | 0.03 0.517±0.032 0.91x | 0.1 0.459±0.031 0.90x | 0.3 0.574±0.328
  [o3k1-cudagraph-bs1] lam=0.001 1.375±1.433 | 0.01 1.185±1.138 | 0.03 1.057±1.010 | 0.1 0.456±0.015 0.84x | 0.3 0.417±0.001 0.78x

[benchmark c100_vit_s_8blk] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  fixed t=4 62.01±0.54 98054±431 img/s 1.852±0.011 ms; fixed t=7 62.62±0.71 62426±126 img/s 2.890±0.041 ms
  lam=0.03 (eager)           62.49±0.54 4.22±0.05 steps 30621±1181 img/s 4.361±0.370 ms  0.47x / 0.64x
  lam=0.03 (compiled)        62.48±0.56 4.22±0.05 steps 31304±1060 img/s 4.098±0.345 ms  0.48x / 0.81x
  lam=0.03 (o3k0-eager)      61.52±0.75 4.49±0.27 steps 77791±1270 img/s 2.609±0.092 ms  0.74x / 0.67x
  lam=0.03 (o3k1-eager)      61.49±0.27 3.95±0.06 steps 67226±3350 img/s 2.544±0.015 ms  0.64x / 0.68x
  [cudagraph-bs1] fixed t=0..7 ms: 0.124 0.211 0.297 0.395 0.470 0.569 0.659 0.743
  lam=0.001 1.005±0.039 0.71x | lam=0.01 0.879±0.016 0.81x | lam=0.03 0.812±0.014 0.88x | lam=0.1 1.912±2.128 | lam=0.3 0.579±0.026 0.68x
  [o3k0-cudagraph-bs1] all lam ~1.2-1.35±1.4 ms (one seed disturbed)
  [o3k1-cudagraph-bs1] lam=0.001 0.612±0.027 0.87x | 0.01 0.550±0.041 0.86x | 0.03 0.507±0.007 0.89x | 0.1 0.479±0.003 0.82x | 0.3 0.403±0.025 0.74x
===== END FSV SUMMARY =====
```

The two bViT-S / 8-block benchmark blocks are condensed here (key rows only); the full tables are in the
conversation in which they were pasted.
