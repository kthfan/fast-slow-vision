# Report: c100_vit_s_8blk

Non-shared baseline: 8 distinct blocks with an exit after each (same FLOPs, ~8x block params); measures the cost of weight sharing.

## Environment

- git: `?` on `?` (**uncommitted changes**)
- GPU: NVIDIA H100 NVL | torch 2.6.0+cu124 (CUDA 12.4) | Python 3.10.12
- CPU cap: 8 cores (machine has 48); DataLoader workers 7
- AMP: bf16 | params: 14.28M
- started 2026-11-06T21:53:56, train 19.5 min, eval 0.1 min, exit codes train=0 eval=0
- train command: `scripts/train.py --dataset cifar100 --model bvit_s --t-max 8 --epochs 200 --batch-size 256 --lr 1e-3 --wd 0.05 --warmup-epochs 10 --n-blocks 8 --data-dir /workspace/fast-slow-vision/data --out /workspace/fast-slow-vision/runs/c100_vit_s_8blk --max-cpus 8 --resume`

## Training

- epochs run: 200; mean epoch 4.6s; mean throughput 10911 img/s

Held-out top-1 % per step (no temperature scaling):

| epoch | train loss | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 4.892 | 17.2 | 19.4 | 19.9 | 19.6 | 19.3 | 19.2 | 19.0 | 19.1 |
| 19 | 4.724 | 20.5 | 23.4 | 23.8 | 23.9 | 24.1 | 24.2 | 24.3 | 24.2 |
| 29 | 4.655 | 22.8 | 28.0 | 28.7 | 29.2 | 29.4 | 29.3 | 29.3 | 29.3 |
| 39 | 4.530 | 25.6 | 32.1 | 33.4 | 34.1 | 34.1 | 34.1 | 34.2 | 34.1 |
| 49 | 4.465 | 25.5 | 34.1 | 36.0 | 36.7 | 36.9 | 36.8 | 37.1 | 36.9 |
| 59 | 4.422 | 26.6 | 36.3 | 38.3 | 39.9 | 40.0 | 40.2 | 40.6 | 40.4 |
| 69 | 4.270 | 27.4 | 38.6 | 41.7 | 43.3 | 43.6 | 44.3 | 44.5 | 44.5 |
| 79 | 4.194 | 27.7 | 41.4 | 44.7 | 46.4 | 47.4 | 48.3 | 48.3 | 48.4 |
| 89 | 4.079 | 27.6 | 43.3 | 48.0 | 50.1 | 51.4 | 51.8 | 52.5 | 52.4 |
| 99 | 3.989 | 28.6 | 45.0 | 49.9 | 51.9 | 53.2 | 53.2 | 53.6 | 53.8 |
| 109 | 3.872 | 29.3 | 46.6 | 51.8 | 54.4 | 55.1 | 55.7 | 56.1 | 56.2 |
| 119 | 3.834 | 29.3 | 47.4 | 52.4 | 54.8 | 56.3 | 56.4 | 56.6 | 56.6 |
| 129 | 3.764 | 29.1 | 48.3 | 53.5 | 55.9 | 57.4 | 57.8 | 57.8 | 57.9 |
| 139 | 3.641 | 29.9 | 49.0 | 54.4 | 56.7 | 57.7 | 57.9 | 58.6 | 58.6 |
| 149 | 3.539 | 30.9 | 50.1 | 55.6 | 57.9 | 58.6 | 59.5 | 59.7 | 59.7 |
| 159 | 3.525 | 30.4 | 50.2 | 55.7 | 58.2 | 59.4 | 59.9 | 60.4 | 60.4 |
| 169 | 3.555 | 31.1 | 51.1 | 56.3 | 58.5 | 59.8 | 60.4 | 60.5 | 60.1 |
| 179 | 3.647 | 31.0 | 50.9 | 57.0 | 58.9 | 59.8 | 60.0 | 60.4 | 60.7 |
| 189 | 3.505 | 30.9 | 51.2 | 57.0 | 59.5 | 60.2 | 60.4 | 61.0 | 60.9 |
| 199 | 3.567 | 31.3 | 51.5 | 57.0 | 59.3 | 60.3 | 60.5 | 60.8 | 60.6 |

## Test set, per step

| | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 |
|---|---|---|---|---|---|---|---|---|
| top-1 % | 32.94 | 53.14 | 58.50 | 61.28 | 62.10 | 62.41 | 62.76 | 62.92 |
| ECE raw % | 8.21 | 8.07 | 5.12 | 4.98 | 5.17 | 4.99 | 4.45 | 3.81 |
| ECE TS % | 1.29 | 1.05 | 2.00 | 2.45 | 2.44 | 2.81 | 3.00 | 3.54 |
| NLL TS | 2.706 | 1.809 | 1.601 | 1.530 | 1.527 | 1.530 | 1.537 | 1.547 |
| Brier TS | 0.8011 | 0.5974 | 0.5364 | 0.5099 | 0.5035 | 0.5022 | 0.5009 | 0.5022 |
| AURC TS % | 44.90 | 21.80 | 17.15 | 15.46 | 14.99 | 15.01 | 14.90 | 15.02 |
| temperature | 0.808 | 0.833 | 0.866 | 0.870 | 0.875 | 0.878 | 0.888 | 0.896 |
| cum. GFLOPs | 0.239 | 0.476 | 0.712 | 0.949 | 1.185 | 1.422 | 1.659 | 1.895 |
| latency bs1 ms | 0.45 | 0.71 | 0.97 | 1.33 | 1.63 | 1.88 | 2.09 | 2.54 |
| latency bs256 ms | 0.62 | 1.09 | 1.58 | 2.09 | 2.64 | 3.10 | 3.59 | 4.09 |

Accuracy-compute AUC (mean top-1 % over [0.239, 1.895] GFLOPs, TS probabilities):

- maxprob: 57.77
- entropy: 57.23
- fixed_depth: 58.30

Oracle stopping (t* = earliest step after which every step is correct):

- full: top-1 62.92%, 1.039 GFLOPs, 4.38 steps
- any: top-1 66.43%, 0.425 GFLOPs, 1.79 steps

Latency device: NVIDIA H100 NVL, AMP bf16; bs1 full-depth throughput 394 img/s; bs256 full-depth throughput 62604 img/s
