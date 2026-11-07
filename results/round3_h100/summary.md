# Round 3 (H100, pasted by the user)

Output of `python scripts/summarize.py c100_bvit_s_fp16 c100_bvit_s_te c100_bvit_s_inject c100_bvit_s_linw c100_bvit_s_nosd`.

```
===== BEGIN FSV SUMMARY =====
code 0.3.0+938e443f1e34 | NVIDIA H100 NVL | torch 2.6.0+cu124 | cpus 8/48

[c100_bvit_s_fp16] seeds 0 | params 1.86M | train 12.9 min | epoch 3.7s
  test acc%  33.07 54.69 58.34 59.24 59.47 59.16 58.92 58.49
  ECE raw%   10.06 9.34 7.30 7.00 7.02 6.88 6.97 7.07
  ECE TS%    1.99 2.11 2.79 2.67 2.97 3.26 3.41 3.74
  NLL TS     2.700 1.754 1.594 1.567 1.568 1.578 1.592 1.606
  AURC TS%   44.97 20.49 17.29 16.76 16.73 16.90 17.10 17.37
  temp       0.765 0.803 0.823 0.827 0.828 0.828 0.826 0.822
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.41 0.61 0.83 1.11 1.32 1.52 1.70 1.92
  lat bs256 ms 0.64 1.14 1.63 2.11 2.65 3.08 3.64 4.07
  holdout%   31.8 52.9 56.9 58.0 58.4 58.5 58.4 58.1
  final train loss 3.795 | AMP fp16
  best-step acc 59.47 | last-step acc 58.49
  AUC maxprob 54.42 entropy 54.03 fixed_depth 56.71
  oracle full 58.49% @ 1.043 GFLOPs | any 62.23% @ 0.356 GFLOPs

[c100_bvit_s_inject] seeds 0 | params 1.86M | train 10.9 min | epoch 3.1s
  test acc%  33.23 54.50 58.69 59.53 59.75 59.59 59.46 59.24
  ECE raw%   9.97 9.00 7.06 6.60 6.59 6.60 6.85 7.13
  ECE TS%    1.86 1.78 2.29 2.73 3.02 2.96 3.01 3.05
  NLL TS     2.709 1.755 1.585 1.553 1.552 1.560 1.572 1.585
  AURC TS%   44.79 20.50 17.03 16.41 16.31 16.38 16.52 16.72
  temp       0.767 0.810 0.833 0.834 0.833 0.831 0.828 0.824
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.40 0.66 0.95 1.09 1.34 1.64 1.77 1.96
  lat bs256 ms 0.62 1.15 1.67 2.19 2.80 3.24 3.84 4.27
  holdout%   32.1 53.4 57.2 58.2 58.5 58.7 58.6 58.3
  final train loss 3.780 | AMP bf16
  best-step acc 59.75 | last-step acc 59.24
  AUC maxprob 54.94 entropy 54.50 fixed_depth 56.92
  oracle full 59.24% @ 1.036 GFLOPs | any 62.78% @ 0.361 GFLOPs

[c100_bvit_s_linw] seeds 0 | params 1.86M | train 10.2 min | epoch 2.9s
  test acc%  27.32 53.86 59.60 61.39 62.05 62.14 61.97 61.70
  ECE raw%   10.17 11.44 7.43 7.02 7.15 7.31 7.55 7.89
  ECE TS%    2.16 1.72 2.45 2.52 2.31 2.58 2.81 2.60
  NLL TS     2.974 1.788 1.538 1.474 1.459 1.462 1.472 1.487
  AURC TS%   53.50 21.62 16.57 15.32 15.01 15.04 15.22 15.46
  temp       0.726 0.778 0.824 0.830 0.831 0.830 0.827 0.823
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.37 0.69 0.78 1.03 1.25 1.40 1.54 1.82
  lat bs256 ms 0.62 1.10 1.58 2.04 2.58 2.99 3.53 3.96
  holdout%   26.9 52.0 58.4 60.1 60.1 60.1 60.0 59.5
  final train loss 3.620 | AMP bf16
  best-step acc 62.14 | last-step acc 61.70
  AUC maxprob 56.17 entropy 55.70 fixed_depth 57.99
  oracle full 61.70% @ 1.027 GFLOPs | any 64.71% @ 0.392 GFLOPs

[c100_bvit_s_nosd] seeds 0 | params 1.86M | train 10.1 min | epoch 2.9s
  test acc%  31.76 53.37 58.08 59.17 59.49 59.34 59.09 58.97
  ECE raw%   10.95 9.00 7.02 6.55 6.46 6.41 6.53 6.95
  ECE TS%    1.62 2.39 2.09 2.41 2.43 2.76 2.98 2.99
  NLL TS     2.789 1.798 1.604 1.557 1.548 1.553 1.563 1.576
  AURC TS%   47.13 21.51 17.50 16.64 16.48 16.58 16.78 16.98
  temp       0.743 0.808 0.838 0.841 0.840 0.837 0.832 0.827
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.35 0.52 0.75 0.97 1.14 1.30 1.48 1.69
  lat bs256 ms 0.62 1.11 1.58 2.04 2.62 2.99 3.55 3.94
  holdout%   30.4 52.0 56.8 58.1 58.0 57.9 57.9 57.8
  final train loss 3.704 | AMP bf16
  best-step acc 59.49 | last-step acc 58.97
  AUC maxprob 54.50 entropy 54.10 fixed_depth 56.39
  oracle full 58.97% @ 1.047 GFLOPs | any 62.37% @ 0.367 GFLOPs

[c100_bvit_s_te] seeds 0 | params 1.86M | train 10.4 min | epoch 3.0s
  test acc%  32.56 54.44 58.62 59.46 59.54 59.61 59.40 59.19
  ECE raw%   9.32 9.14 6.92 6.51 6.38 6.53 6.53 6.60
  ECE TS%    1.69 2.14 2.67 3.10 3.25 3.53 3.59 3.69
  NLL TS     2.732 1.744 1.574 1.543 1.541 1.547 1.555 1.566
  AURC TS%   45.59 20.48 17.14 16.51 16.47 16.57 16.73 16.91
  temp       0.777 0.807 0.830 0.832 0.831 0.829 0.827 0.826
  GFLOPs     0.239 0.476 0.712 0.949 1.185 1.422 1.659 1.895
  lat bs1 ms 0.40 0.62 0.86 1.05 1.24 1.42 1.68 1.82
  lat bs256 ms 0.65 1.16 1.64 2.13 2.71 3.13 3.73 4.13
  holdout%   31.2 52.7 57.1 57.8 58.0 58.3 58.2 58.0
  final train loss 3.785 | AMP bf16
  best-step acc 59.61 | last-step acc 59.19
  AUC maxprob 54.85 entropy 54.44 fixed_depth 56.77
  oracle full 59.19% @ 1.035 GFLOPs | any 62.80% @ 0.359 GFLOPs
===== END FSV SUMMARY =====
```
