# Round 7 (H100, pasted by the user)

Output of `python scripts/summarize.py c100_bvit_s_quadw_lr5e4 c100_vit_s_8blk c100_bvit_s_lr5e4`.
Halting policies S0-reward_acc / S0-oracle_any were re-trained (code 0.7.0, `--halting-only`) to save the
policies and compute cost@full / cost@best; the other policy rows come from round 6 and have no cost metrics
(shown as "-", which the 0.7.1 summary shows as blank instead). The training-section entries are identical to
round 6 and are omitted here.

Benchmark (code 0.7.0, runner with boolean-mask indexing = "eager-naive" in 0.7.1): adaptive inference was
2-4x SLOWER than fixed depth on the GPU, both for batch 256 and batch 1, although it uses fewer FLOPs.
The column formatting bug ("59.97±0.814.07±0.66" = acc 59.97±0.81, steps 4.07±0.66) is fixed in 0.7.1.

```
[halting c100_bvit_s_lr5e4] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                       AUC      acc@t1      acc@t2      acc@t3   cost@full   cost@best
  cost@full / cost@best = compute needed to match the full-depth / best fixed-depth accuracy, % of full-depth compute ('-' = never)
  fixed_depth           57.17±0.61  54.73±0.38  58.74±0.62  59.87±0.64    48.6±6.4    62.5±0.0
  maxprob               55.26±0.84  48.06±0.75  54.26±0.89  57.62±0.86    83.2±3.9           -
  entropy               54.86±0.78  47.18±0.62  53.43±0.74  56.92±0.69    84.1±6.3           -
  S0-reward_acc         57.80±0.59  55.62±0.36  59.76±0.69  60.01±0.66   41.3±11.6    42.9±1.0
  S0-oracle_any         57.64±0.75  55.53±0.51  59.55±0.73  59.84±0.89    38.7±4.3           -
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0257% of one recurrent step; oracle any 63.17±0.87% @ 0.363±0.008 GFLOPs

[halting c100_bvit_s_quadw_lr5e4] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                       AUC      acc@t1      acc@t2      acc@t3   cost@full   cost@best
  fixed_depth           57.54±0.39  52.16±0.44  59.40±0.44  61.40±0.64    62.0±8.4    79.2±7.2
  maxprob               55.83±0.17  45.88±0.33  54.70±0.21  59.06±0.14    83.4±6.9           -
  entropy               55.33±0.22  44.87±0.18  53.40±0.28  58.26±0.26    87.0±6.2           -
  S0-reward_acc         58.26±0.40  52.55±0.40  60.89±0.59  62.01±0.42    43.2±0.2        46.7
  S0-oracle_any         58.25±0.29  52.70±0.17  60.81±0.52  61.96±0.28    48.3±2.4           -
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0257% of one recurrent step; oracle any 64.76±0.36% @ 0.408±0.003 GFLOPs

[halting c100_vit_s_8blk] backbone seeds 3 | test set, policies trained on held-out | AUC and acc@<fixed-depth compute> in %
  method                       AUC      acc@t1      acc@t2      acc@t3   cost@full   cost@best
  fixed_depth           58.30±0.39  53.17±0.45  58.98±0.49  61.28±0.11    95.3±8.1    95.8±7.2
  maxprob               57.67±0.45  49.15±0.39  56.55±0.29  60.47±0.57    72.4±2.4    72.6±2.3
  entropy               57.11±0.48  47.98±0.41  55.22±0.49  59.56±0.56    72.7±2.8    72.8±2.6
  S0-reward_acc         59.39±0.47  54.30±0.53  60.95±0.51  62.27±0.50    53.8±2.5    54.1±2.0
  S0-oracle_any         59.44±0.50  54.35±0.42  61.00±0.27  62.42±0.67    51.1±0.2    51.4±0.6
  lambda sweep: mean steps non-monotone in 0.0 of 40 adjacent pairs (reward policies)
  policy cost <= 0.0257% of one recurrent step; oracle any 66.40±0.44% @ 0.422±0.004 GFLOPs

[benchmark c100_bvit_s_lr5e4] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput (compaction), batch-1 latency on 500 images
  setting        acc%          steps   img/s           bs1 ms        speed-up vs fixed (bs/bs1)
  fixed t=0       33.05±0.28    1.00    406926±531   0.435±0.016
  fixed t=1       54.73±0.38    2.00    233116±864   0.688±0.036
  fixed t=2       58.74±0.62    3.00    163387±509   0.911±0.017
  fixed t=3       59.87±0.64    4.00    126248±219   1.121±0.020
  fixed t=4       60.09±0.84    5.00     102646±77   1.351±0.013
  fixed t=5       60.04±0.88    6.00      86479±78   1.540±0.034
  fixed t=6       59.80±0.97    7.00      74561±39   1.801±0.048
  fixed t=7       59.64±1.02    8.00      65514±68   2.020±0.054
  lam=0.0003     59.97±0.814.07±0.66    25200±1801   4.806±0.856   0.23±0.01x / 0.27±0.02x
  lam=0.001      59.88±0.823.85±0.60   30043±10186   4.494±0.696   0.27±0.06x / 0.29±0.02x
  lam=0.003      59.82±0.893.66±0.56   31986±13465   7.004±5.073   0.28±0.08x / 0.24±0.11x
  lam=0.01       59.78±0.983.42±0.47   32723±11598   5.007±1.222   0.28±0.09x / 0.25±0.07x
  lam=0.03       59.66±0.983.12±0.33    31640±7574   6.476±4.496   0.27±0.06x / 0.25±0.14x
  lam=0.1        59.28±1.012.75±0.20   18090±13417   6.442±5.670   0.14±0.11x / 0.27±0.16x
  lam=0.3        57.90±1.072.32±0.11   35279±24212   2.849±0.374   0.22±0.15x / 0.32±0.05x

[benchmark c100_bvit_s_quadw_lr5e4] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput (compaction), batch-1 latency on 500 images
  setting        acc%          steps   img/s           bs1 ms        speed-up vs fixed (bs/bs1)
  fixed t=0       24.98±0.84    1.00   405128±1185   0.436±0.014
  fixed t=1       52.16±0.44    2.00   234581±1018   0.672±0.013
  fixed t=2       59.40±0.44    3.00    164161±594   0.894±0.020
  fixed t=3       61.40±0.64    4.00    126200±374   1.135±0.012
  fixed t=4       61.97±0.40    5.00    102911±174   1.358±0.029
  fixed t=5       62.12±0.27    6.00     86910±301   1.592±0.030
  fixed t=6       62.07±0.35    7.00     75092±201   1.809±0.019
  fixed t=7       61.93±0.24    8.00     65918±389   2.014±0.004
  lam=0.0003     62.01±0.404.70±0.18    29023±8059   5.508±0.206   0.33±0.08x / 0.28±0.06x
  lam=0.001      61.95±0.404.42±0.18    31556±5859   5.210±0.206   0.31±0.06x / 0.26±0.01x
  lam=0.003      61.97±0.404.24±0.15     37531±963   5.012±0.238   0.36±0.01x / 0.27±0.01x
  lam=0.01       61.98±0.424.04±0.08     38248±585   4.755±0.094   0.39±0.03x / 0.30±0.03x
  lam=0.03       61.95±0.423.70±0.05     37770±593   4.350±0.138   0.39±0.03x / 0.33±0.03x
  lam=0.1        61.48±0.523.24±0.05     38052±152   3.802±0.059   0.35±0.04x / 0.34±0.03x
  lam=0.3        60.35±0.772.79±0.05     38962±851   3.225±0.048   0.31±0.01x / 0.35±0.01x

[benchmark c100_vit_s_8blk] seeds 3 | NVIDIA H100 NVL | AMP bf16 | policy S0-reward_acc | batch 256 throughput (compaction), batch-1 latency on 500 images
  setting        acc%          steps   img/s           bs1 ms        speed-up vs fixed (bs/bs1)
  fixed t=0       33.05±0.19    1.00  364886±68183   0.440±0.010
  fixed t=1       53.17±0.45    2.00  206927±38232   0.750±0.012
  fixed t=2       58.98±0.49    3.00  143759±26672   1.020±0.042
  fixed t=3       61.28±0.11    4.00  110130±20673   1.341±0.090
  fixed t=4       62.01±0.54    5.00   83855±14465   1.601±0.073
  fixed t=5       62.30±0.64    6.00   75272±14124   1.858±0.092
  fixed t=6       62.45±0.53    7.00   64887±12048   2.123±0.109
  fixed t=7       62.62±0.71    8.00   56989±10470   2.387±0.090
  lam=0.0003     62.62±0.645.29±0.26   20162±12228  11.279±9.169   0.32±0.16x / 0.29±0.15x
  lam=0.001      62.65±0.625.09±0.26    26045±2261   5.932±0.409   0.45±0.07x / 0.39±0.00x
  lam=0.003      62.67±0.664.90±0.22    26321±2286   5.787±0.540   0.45±0.07x / 0.40±0.01x
  lam=0.01       62.65±0.604.59±0.09    25759±1738   5.237±0.185   0.44±0.08x / 0.44±0.01x
  lam=0.03       62.49±0.544.22±0.05    26653±1977   4.903±0.151   0.46±0.08x / 0.47±0.04x
  lam=0.1        61.97±0.353.59±0.25    26320±1163   4.238±0.267   0.33±0.02x / 0.42±0.02x
  lam=0.3        60.70±0.112.90±0.15    32631±6211   4.993±2.883   0.31±0.13x / 0.33±0.15x
```
