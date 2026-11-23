# Round 10, 3 seeds (H100, pasted by the user)

`python scripts/benchmark_adaptive.py c100_bvit_s_quadw_lr5e4 --seeds 0 1 2` with code 0.14.0, then
`python scripts/summarize.py c100_bvit_s_quadw_lr5e4 --sections bench`. Speed-ups in the summary are computed per
seed against that seed's cheapest eager fixed depth at least as accurate, then averaged.

## Analysis

Accuracy-matched comparison from the 3-seed means (fixed depth = cheapest one whose mean accuracy is at least the
adaptive mean accuracy):

| lam | acc % | steps | matched fixed | slots-triton img/s | vs O1 (eager) | vs eager fixed | vs compiled fixed |
|---|---|---|---|---|---|---|---|
| 0.001 | 61.94 | 4.42 | 5 steps | 82,940 | 2.58x | 0.91x | 0.73x |
| 0.01 | 61.98 | 4.04 | 6 steps | 90,375 | 2.07x | 1.18x | 0.93x |
| 0.03 | 61.95 | 3.70 | 5 steps | 98,427 | 2.29x | 1.08x | 0.87x |
| 0.1 | 61.48 | 3.24 | 5 steps | 111,647 | 2.60x | 1.22x | 0.98x |
| 0.3 | 60.35 | 2.79 | 4 steps | 128,942 | 2.99x | 1.22x | 0.96x |

- Confirmed with 3 seeds: the slot pool with the fused kernel is 2.1-3.0x faster than the compaction runner (O1)
  at batch 256, with small spread (+-1.5% throughput across seeds).
- Correction to the single-seed conclusion: the "beats fixed depth" ratios compared a CUDA-graph runner with the
  EAGER fixed-depth baseline, which on the H100 is itself limited by kernel launches: torch.compile makes fixed depth
  1.24-1.27x faster at 4-6 steps, and its timings are far more stable across seeds (+-1-2% vs +-14% for eager).
  Against the compiled fixed depth the slot pool is at parity at best (0.96-0.98x at lam 0.1 / 0.3, 0.87-0.93x at
  lam 0.01 / 0.03). The fair baseline, fixed depth captured as a CUDA graph at batch 256, has not been measured yet.
- Batch 1, CUDA graphs + fused kernel: 0.81x / 0.94x / 1.03x at lam 0.001 / 0.01 / 0.03 against fixed depth (both
  graph-captured, so this comparison is fair). At lam 0.1 / 0.3 one seed was disturbed (0.581+-0.296 ms,
  0.502+-0.238 ms; the other modes' fixed rows are stable), so those two entries are not usable.
- O3 runners remain behind the slot pool and below the fixed depth at equal accuracy.

## Raw output

```
===== BEGIN FSV SUMMARY =====
code 0.6.1+447e8f02f81a | NVIDIA H100 NVL | torch 2.6.0+cu124 | cpus 8/48

[benchmark c100_bvit_s_quadw_lr5e4] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput with compaction, batch-1 latency on 500 images
  CUDA graphs captured: True; decisions agree with eager on 100.0%+ of timed images
  [eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              24.98±0.84           1.00    364700±67955     0.559±0.112
  fixed t=1              52.16±0.44           2.00    209671±31710     0.849±0.017
  fixed t=2              59.40±0.44           3.00    146708±21988     1.203±0.009
  fixed t=3              61.40±0.64           4.00    106104±14962     1.807±0.251
  fixed t=4              61.97±0.40           5.00     91220±13443     1.883±0.038
  fixed t=5              62.12±0.27           6.00     76835±11413     2.235±0.051
  fixed t=6              62.07±0.35           7.00      66088±9788     2.669±0.154
  fixed t=7              61.93±0.24           8.00      58051±8514     2.930±0.061
  lam=0.001              61.95±0.40      4.42±0.18      32092±6113     4.670±0.186   0.35±0.04x / 0.40±0.01x
  lam=0.01               61.98±0.42      4.04±0.08      43574±1346     4.288±0.255   0.51±0.05x / 0.47±0.05x
  lam=0.03               61.95±0.42      3.70±0.05      43067±1125     3.896±0.099   0.51±0.05x / 0.51±0.06x
  lam=0.1                61.48±0.52      3.24±0.05      42866±1098     3.458±0.064   0.47±0.07x / 0.57±0.04x
  lam=0.3                60.35±0.77      2.79±0.05      43181±1346     2.994±0.096   0.41±0.05x / 0.60±0.09x
  [eager-triton]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.94±0.40      4.42±0.18      52361±6883     3.315±0.149   0.59±0.15x / 0.57±0.04x
  lam=0.01               61.98±0.42      4.04±0.08       60907±409     2.939±0.076   0.72±0.09x / 0.68±0.06x
  lam=0.03               61.95±0.42      3.70±0.05       60187±941     2.716±0.029   0.71±0.10x / 0.74±0.09x
  lam=0.1                61.48±0.52      3.24±0.05       60235±713     2.422±0.036   0.66±0.11x / 0.81±0.05x
  lam=0.3                60.35±0.77      2.79±0.05       60780±967     2.120±0.065   0.58±0.07x / 0.85±0.12x
  [slots]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.95±0.40      4.42±0.18      70685±2421               -   0.79±0.14x / -x
  lam=0.01               61.98±0.42      4.04±0.08      77063±1397               -   0.91±0.11x / -x
  lam=0.03               61.95±0.42      3.70±0.05      85946±4347               -   1.01±0.09x / -x
  lam=0.1                61.48±0.52      3.24±0.05      99554±4505               -   1.09±0.20x / -x
  lam=0.3                60.35±0.77      2.79±0.05     114923±2949               -   1.10±0.17x / -x
  [slots-triton]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.94±0.40      4.42±0.18      82940±2914               -   0.92±0.16x / -x
  lam=0.01               61.98±0.42      4.04±0.08      90375±1754               -   1.06±0.13x / -x
  lam=0.03               61.95±0.42      3.70±0.05      98427±1327               -   1.16±0.14x / -x
  lam=0.1                61.48±0.52      3.24±0.05     111647±1810               -   1.23±0.20x / -x
  lam=0.3                60.35±0.77      2.79±0.05     128942±1666               -   1.23±0.15x / -x
  [compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              25.03±0.81           1.00     375750±5418     0.582±0.003
  fixed t=1              52.16±0.47           2.00     233723±6849     0.907±0.013
  fixed t=2              59.38±0.53           3.00     172300±3071     1.262±0.030
  fixed t=3              61.40±0.58           4.00     134669±1569     1.584±0.032
  fixed t=4              61.99±0.41           5.00     113550±1650     1.899±0.013
  fixed t=5              62.16±0.28           6.00       96948±492     2.216±0.030
  fixed t=6              62.06±0.36           7.00       84740±650     2.538±0.040
  fixed t=7              61.92±0.23           8.00      77002±2113     2.853±0.023
  lam=0.001              61.97±0.41      4.42±0.18      43187±7064     3.748±0.106   0.38±0.06x / 0.51±0.02x
  lam=0.01               62.00±0.42      4.04±0.08       49994±479     3.568±0.197   0.47±0.05x / 0.57±0.07x
  lam=0.03               61.98±0.41      3.70±0.05       48934±989     3.160±0.052   0.46±0.04x / 0.64±0.05x
  lam=0.1                61.50±0.49      3.25±0.05       49788±646     2.820±0.018   0.41±0.03x / 0.64±0.06x
  lam=0.3                60.35±0.71      2.80±0.05      51022±1550     2.470±0.057   0.38±0.01x / 0.64±0.00x
  [o3k0-eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.84±0.54      4.82±0.15      78946±1541     4.509±3.122   0.88±0.13x / 0.54±0.26x
  lam=0.01               61.74±0.54      4.69±0.17     59280±34185     4.521±3.154   0.62±0.31x / 0.54±0.26x
  lam=0.03               61.68±0.59      4.58±0.18      77364±1010     2.668±0.065   0.86±0.13x / 0.71±0.03x
  lam=0.1                60.92±0.87      4.01±0.37       77124±769     2.430±0.166   0.80±0.20x / 0.79±0.03x
  lam=0.3                58.89±0.32      3.03±0.13      80463±1723     2.061±0.096   0.56±0.08x / 0.58±0.03x
  [o3k0-compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.89±0.51      4.82±0.15       74025±681     2.802±0.062   0.65±0.01x / 0.68±0.02x
  lam=0.01               61.79±0.51      4.69±0.17      74709±2261     2.890±0.163   0.66±0.03x / 0.66±0.04x
  lam=0.03               61.74±0.57      4.58±0.18      73825±1707     2.702±0.077   0.65±0.02x / 0.70±0.02x
  lam=0.1                60.99±0.84      4.01±0.37      74242±1233     2.503±0.118   0.59±0.06x / 0.67±0.07x
  lam=0.3                58.92±0.31      3.03±0.13      75196±1099     2.155±0.028   0.44±0.00x / 0.59±0.01x
  [o3k1-eager]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.45±0.73      4.34±0.50      70270±6599     2.788±0.240   0.73±0.08x / 0.64±0.06x
  lam=0.01               61.32±0.76      4.06±0.48      68913±4652     2.695±0.208   0.67±0.13x / 0.64±0.06x
  lam=0.03               61.15±0.74      3.78±0.33      69572±4611     2.551±0.101   0.67±0.13x / 0.71±0.11x
  lam=0.1                60.73±0.66      3.39±0.15      70196±4241     2.411±0.103   0.67±0.12x / 0.75±0.13x
  lam=0.3                59.31±0.65      2.90±0.03      70470±4347     2.176±0.070   0.57±0.16x / 0.69±0.26x
  [o3k1-compiled]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.46±0.68      4.34±0.50      66581±4280     2.909±0.196   0.55±0.06x / 0.62±0.07x
  lam=0.01               61.33±0.71      4.06±0.48      65572±5069     2.814±0.190   0.52±0.09x / 0.60±0.10x
  lam=0.03               61.15±0.70      3.78±0.33      67096±4263     2.697±0.140   0.50±0.04x / 0.59±0.04x
  lam=0.1                60.72±0.66      3.39±0.15     59181±16812     2.556±0.098   0.44±0.13x / 0.62±0.04x
  lam=0.3                59.33±0.63      2.90±0.03      65496±5346     2.340±0.048   0.42±0.09x / 0.58±0.08x
  [cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  fixed t=0              24.98±0.84           1.00               -     0.121±0.005
  fixed t=1              52.16±0.44           2.00               -     0.205±0.008
  fixed t=2              59.40±0.44           3.00               -     0.290±0.012
  fixed t=3              61.40±0.64           4.00               -     0.366±0.015
  fixed t=4              61.97±0.40           5.00               -     0.448±0.017
  fixed t=5              62.12±0.27           6.00               -     0.530±0.020
  fixed t=6              62.07±0.35           7.00               -     0.613±0.023
  fixed t=7              61.93±0.24           8.00               -     0.697±0.026
  lam=0.001              61.95±0.40      4.42±0.18               -     0.842±0.036   -x / 0.53±0.04x
  lam=0.01               61.98±0.42      4.04±0.08               -     0.774±0.019   -x / 0.61±0.04x
  lam=0.03               61.95±0.42      3.70±0.05               -     0.708±0.010   -x / 0.67±0.05x
  lam=0.1                61.48±0.52      3.24±0.05               -     0.625±0.011   -x / 0.67±0.08x
  lam=0.3                60.35±0.77      2.79±0.05               -     0.544±0.011   -x / 0.67±0.02x
  [cudagraph-bs1-triton]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.95±0.40      4.42±0.18               -     0.557±0.018   -x / 0.81±0.05x
  lam=0.01               61.98±0.42      4.04±0.08               -     0.507±0.014   -x / 0.94±0.06x
  lam=0.03               61.95±0.42      3.70±0.05               -     0.462±0.006   -x / 1.03±0.08x
  lam=0.1                61.48±0.52      3.24±0.05               -     0.581±0.296   -x / 0.83±0.33x
  lam=0.3                60.35±0.77      2.79±0.05               -     0.502±0.238   -x / 0.83±0.32x
  [o3k0-cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.84±0.54      4.82±0.15               -     0.679±0.211   -x / 0.70±0.20x
  lam=0.01               61.74±0.54      4.69±0.17               -     0.653±0.216   -x / 0.73±0.22x
  lam=0.03               61.68±0.59      4.58±0.18               -     0.628±0.220   -x / 0.77±0.24x
  lam=0.1                60.92±0.87      4.01±0.37               -     0.557±0.141   -x / 0.74±0.25x
  lam=0.3                58.89±0.32      3.03±0.13               -     0.477±0.160   -x / 0.65±0.20x
  [o3k1-cudagraph-bs1]  setting      acc%           steps         img/s            bs1 ms        speed-up vs fixed (bs256 / bs1); * = more accurate than every fixed depth, compared with the most accurate one
  lam=0.001              61.45±0.73      4.34±0.50               -     0.684±0.235   -x / 0.65±0.18x
  lam=0.01               61.32±0.76      4.06±0.48               -     0.675±0.247   -x / 0.64±0.24x
  lam=0.03               61.15±0.74      3.78±0.33               -     0.627±0.236   -x / 0.64±0.21x
  lam=0.1                60.73±0.66      3.39±0.15               -     0.572±0.216   -x / 0.70±0.23x
  lam=0.3                59.31±0.65      2.90±0.03               -     0.517±0.173   -x / 0.65±0.24x
===== END FSV SUMMARY =====
```
