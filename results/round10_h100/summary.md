# Round 10 (H100, pasted by the user)

Code 0.14.0 (branch `claude/admiring-franklin-6zlsn0` rebased on main). Two runs, one seed, NVIDIA H100 NVL, AMP bf16:

1. `python scripts/bench_sync.py --model bvit_s --images 10240 --slots 256` (random weights, synthetic policy);
2. `python scripts/benchmark_adaptive.py c100_bvit_s_quadw_lr5e4 --seeds 0` (trained backbone, policy S0-reward_acc).

## Speed-up against the accuracy-matched fixed depth (computed from the output below)

The fixed depth used for each lambda is the cheapest one at least as accurate. For lam=0.01 and 0.03 that is
6 steps, because the adaptive accuracy (62.04%) exceeds 5 steps (62.00%) by 4 test images; the ratios against
5 steps are given in brackets.

Batch 256, throughput:

| lam | acc % | steps | fixed match (img/s) | O1 eager | eager-triton | slots | slots-triton |
|---|---|---|---|---|---|---|---|
| 0.001 | 61.93 | 4.62 | 5 steps (99,143) | 0.40x | 0.57x | 0.69x | 0.80x |
| 0.01 | 62.04 | 4.11 | 6 steps (83,480) | 0.54x | 0.76x | 0.91x | 1.06x (0.89x) |
| 0.03 | 62.04 | 3.71 | 6 steps (83,480) | 0.53x | 0.76x | 1.01x | 1.18x (0.99x) |
| 0.1 | 61.53 | 3.26 | 5 steps (99,143) | 0.44x | 0.63x | 0.96x | **1.12x** |
| 0.3 | 60.05 | 2.74 | 4 steps (122,204) | 0.37x | 0.52x | 0.98x | **1.07x** |

Batch 1, CUDA graphs, latency:

| lam | fixed match (ms) | O1 | O1 + Triton kernel |
|---|---|---|---|
| 0.001 | 5 steps (0.467) | 0.52x | 0.77x |
| 0.01 | 6 steps (0.520) | 0.65x | 1.01x (0.91x) |
| 0.03 | 6 steps (0.520) | 0.72x | 1.09x (0.98x) |
| 0.1 | 5 steps (0.467) | 0.73x | **1.12x** |
| 0.3 | 4 steps (0.383) | 0.71x | **1.07x** |

## Raw output

```
[bench_sync] NVIDIA H100 NVL | bvit_s T=8 AMP bf16 | policy mean steps 3.51
  fixed t=3                steps  4.00  bs256    121824 img/s  bs1 eager  1.409 ms  bs1 graph  0.357 ms
  fixed t=4                steps  5.00  bs256     99759 img/s  bs1 eager  1.766 ms  bs1 graph  0.438 ms
  O1 sync per step         steps  3.51  bs256     35190 img/s  bs1 eager  3.137 ms  bs1 graph  0.617 ms  agree 1.000/1.000
  slot pool S=256          steps  3.51  bs256     87019 img/s  bs1 eager    nan ms  bs1 graph    nan ms  agree 1.000/1.000
    ticks 147, occupancy 0.956, tick GPU 0.791 ms, GPU idle 1.2%
  slot pool S=256 triton   steps  3.51  bs256    102474 img/s  bs1 eager    nan ms  bs1 graph    nan ms  agree 1.000/1.000
    ticks 147, occupancy 0.956, tick GPU 0.670 ms, GPU idle 1.5%
  O1 graphs, no sync (bound) steps  3.51  bs256       nan img/s  bs1 eager    nan ms  bs1 graph  0.617 ms
  O1 triton-fused sync     steps  3.51  bs256     45668 img/s  bs1 eager  2.382 ms  bs1 graph  0.425 ms  agree 1.000/1.000
  O1 triton-fused bound    steps  3.51  bs256       nan img/s  bs1 eager    nan ms  bs1 graph  0.368 ms
```

```
[benchmark] device NVIDIA H100 NVL, 10000 test images, batch 256, batch-1 on 500 images, 5 repeats

[benchmark] c100_bvit_s_quadw_lr5e4 (AMP bf16, policy S0-reward_acc)
  [eager] setting   acc %   steps   img/s (bs 256)   bs1 ms
  fixed t=0       25.92    1.00          402375    0.453
  fixed t=1       52.15    2.00          227292    0.814
  fixed t=2       59.27    3.00          159440    1.118
  fixed t=3       61.26    4.00          122204    1.380
  fixed t=4       62.00    5.00           99143    1.739
  fixed t=5       62.06    6.00           83480    2.123
  fixed t=6       61.86    7.00           72054    2.460
  fixed t=7       61.79    8.00           63022    2.812
  lam=0.001        61.93    4.62           39971    4.681
  lam=0.01         62.04    4.11           44721    4.212
  lam=0.03         62.04    3.71           44147    3.825
  lam=0.1          61.53    3.26           43889    3.267
  lam=0.3          60.05    2.74           44863    2.739
  [eager-triton] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.92    4.62           56653    3.175
  lam=0.01         62.04    4.11           63234    2.860
  lam=0.03         62.04    3.71           63278    2.598
  lam=0.1          61.53    3.26           62064    2.291
  lam=0.3          60.05    2.74           64044    1.952
  [slots] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.93    4.62           68198      nan
  lam=0.01         62.04    4.11           76033      nan
  lam=0.03         62.04    3.71           84189      nan
  lam=0.1          61.53    3.26           94976      nan
  lam=0.3          60.05    2.74          119373      nan
  [slots-triton] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.92    4.62           79697      nan
  lam=0.01         62.04    4.11           88668      nan
  lam=0.03         62.04    3.71           98454      nan
  lam=0.1          61.53    3.26          111132      nan
  lam=0.3          60.05    2.74          130841      nan
  [compiled] setting   acc %   steps   img/s (bs 256)   bs1 ms
  fixed t=0       25.93    1.00          385335    0.552
  fixed t=1       52.17    2.00          242948    0.879
  fixed t=2       59.38    3.00          184572    1.187
  fixed t=3       61.25    4.00          141035    1.503
  fixed t=4       62.03    5.00          121918    1.792
  fixed t=5       62.08    6.00          102068    2.097
  fixed t=6       61.81    7.00           90639    2.478
  fixed t=7       61.78    8.00           79977    2.654
  lam=0.001        61.97    4.62           47561    3.740
  lam=0.01         62.07    4.11           51553    3.394
  lam=0.03         62.06    3.71           52445    3.093
  lam=0.1          61.57    3.26           50317    2.743
  lam=0.3          60.08    2.74           50865    2.335
  [o3k0-eager] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.95    4.91           80482    2.577
  lam=0.01         61.75    4.79           82827    2.570
  lam=0.03         61.60    4.67           84364    2.504
  lam=0.1          60.06    3.58           83464    2.123
  lam=0.3          58.65    2.89           83725    1.908
  [o3k0-compiled] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.99    4.91           77861    2.694
  lam=0.01         61.80    4.79           77453    2.720
  lam=0.03         61.65    4.67           78025    2.674
  lam=0.1          60.15    3.58           77759    2.233
  lam=0.3          58.74    2.89           78109    2.052
  [o3k1-eager] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.36    4.92           66826    2.920
  lam=0.01         61.20    4.61           65579    2.731
  lam=0.03         61.03    4.16           67868    2.640
  lam=0.1          60.55    3.54           67424    2.428
  lam=0.3          59.15    2.89           68032    2.234
  [o3k1-compiled] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.44    4.92           62173    3.055
  lam=0.01         61.24    4.61           63196    3.030
  lam=0.03         61.09    4.16           60712    2.972
  lam=0.1          60.57    3.54           58000    2.780
  lam=0.3          59.21    2.89           56003    2.418
  [cudagraph-bs1] setting   acc %   steps   img/s (bs 256)   bs1 ms
  fixed t=0       25.92    1.00             nan    0.124
  fixed t=1       52.15    2.00             nan    0.210
  fixed t=2       59.27    3.00             nan    0.296
  fixed t=3       61.26    4.00             nan    0.383
  fixed t=4       62.00    5.00             nan    0.467
  fixed t=5       62.06    6.00             nan    0.520
  fixed t=6       61.86    7.00             nan    0.600
  fixed t=7       61.79    8.00             nan    0.681
  lam=0.001        61.93    4.62             nan    0.897
  lam=0.01         62.04    4.11             nan    0.806
  lam=0.03         62.04    3.71             nan    0.726
  lam=0.1          61.53    3.26             nan    0.640
  lam=0.3          60.05    2.74             nan    0.542
  [cudagraph-bs1-triton] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.93    4.62             nan    0.606
  lam=0.01         62.04    4.11             nan    0.514
  lam=0.03         62.04    3.71             nan    0.477
  lam=0.1          61.53    3.26             nan    0.416
  lam=0.3          60.05    2.74             nan    0.357
  [o3k0-cudagraph-bs1] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.95    4.91             nan    0.571
  lam=0.01         61.75    4.79             nan    0.562
  lam=0.03         61.60    4.67             nan    0.550
  lam=0.1          60.06    3.58             nan    0.424
  lam=0.3          58.65    2.89             nan    0.367
  [o3k1-cudagraph-bs1] setting   acc %   steps   img/s (bs 256)   bs1 ms
  lam=0.001        61.36    4.92             nan    0.585
  lam=0.01         61.20    4.61             nan    0.560
  lam=0.03         61.03    4.16             nan    0.523
  lam=0.1          60.55    3.54             nan    0.476
  lam=0.3          59.15    2.89             nan    0.419
```

(The fixed-depth rows repeated under every mode are omitted after their first occurrence; they are identical.)
