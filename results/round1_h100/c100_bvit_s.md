# Report: c100_bvit_s

bViT-S baseline, same recipe as the Kaggle run (reproduction on a single GPU).

## Environment

- git: `?` on `?` (**uncommitted changes**)
- GPU: NVIDIA H100 NVL | torch 2.6.0+cu124 (CUDA 12.4) | Python 3.10.12
- CPU cap: 8 cores (machine has 48); DataLoader workers 7
- AMP: bf16 | params: 1.86M
- started 2026-11-06T21:25:25, train 13.2 min, eval 0.1 min, exit codes train=0 eval=0
- train command: `scripts/train.py --dataset cifar100 --model bvit_s --t-max 8 --epochs 200 --batch-size 256 --lr 1e-3 --wd 0.05 --warmup-epochs 10 --data-dir /workspace/fast-slow-vision/data --out /workspace/fast-slow-vision/runs/c100_bvit_s --max-cpus 8 --resume`

## Training

- epochs run: 200; mean epoch 3.8s; mean throughput 12966 img/s

Held-out top-1 % per step (no temperature scaling):

| epoch | train loss | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 4.882 | 17.5 | 18.4 | 18.9 | 18.9 | 18.6 | 18.5 | 18.5 | 18.5 |
| 19 | 4.704 | 22.0 | 25.6 | 26.0 | 25.8 | 25.7 | 25.7 | 25.7 | 25.5 |
| 29 | 4.632 | 24.3 | 28.9 | 29.7 | 30.0 | 30.1 | 30.1 | 30.2 | 29.8 |
| 39 | 4.515 | 25.1 | 33.6 | 35.1 | 35.7 | 35.8 | 35.6 | 35.5 | 35.5 |
| 49 | 4.459 | 26.9 | 35.9 | 37.1 | 37.4 | 37.7 | 37.7 | 37.4 | 37.2 |
| 59 | 4.419 | 26.5 | 38.6 | 40.2 | 41.0 | 41.0 | 41.0 | 40.6 | 40.5 |
| 69 | 4.287 | 28.4 | 40.8 | 42.1 | 43.0 | 42.9 | 42.9 | 43.1 | 42.9 |
| 79 | 4.239 | 28.6 | 43.4 | 45.2 | 45.7 | 45.7 | 45.4 | 45.3 | 45.1 |
| 89 | 4.162 | 28.8 | 44.3 | 47.1 | 47.6 | 47.9 | 47.6 | 47.2 | 47.0 |
| 99 | 4.083 | 29.9 | 46.1 | 48.9 | 49.7 | 50.1 | 50.0 | 50.0 | 49.9 |
| 109 | 4.009 | 29.7 | 46.9 | 50.6 | 51.6 | 51.9 | 52.1 | 52.0 | 51.7 |
| 119 | 3.987 | 28.6 | 47.0 | 51.1 | 51.5 | 51.8 | 51.6 | 51.4 | 51.3 |
| 129 | 3.926 | 29.7 | 49.8 | 53.6 | 54.5 | 54.9 | 55.0 | 54.7 | 54.7 |
| 139 | 3.838 | 30.7 | 50.2 | 53.9 | 54.8 | 55.5 | 55.3 | 55.1 | 54.9 |
| 149 | 3.757 | 31.0 | 51.2 | 55.2 | 55.9 | 56.3 | 56.2 | 56.1 | 56.0 |
| 159 | 3.747 | 31.1 | 52.3 | 56.6 | 57.3 | 57.5 | 57.6 | 57.2 | 56.9 |
| 169 | 3.768 | 31.6 | 52.7 | 57.1 | 58.0 | 57.9 | 58.1 | 57.9 | 57.8 |
| 179 | 3.846 | 31.5 | 52.8 | 57.0 | 58.1 | 58.0 | 57.9 | 57.8 | 57.6 |
| 189 | 3.730 | 31.7 | 53.1 | 57.6 | 58.3 | 58.5 | 58.4 | 58.0 | 57.7 |
| 199 | 3.784 | 31.5 | 53.3 | 57.5 | 58.6 | 58.5 | 58.3 | 58.2 | 57.8 |

## Test set, per step

| | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 |
|---|---|---|---|---|---|---|---|---|
| top-1 % | 33.33 | 54.09 | 58.46 | 59.31 | 59.41 | 59.34 | 59.25 | 59.19 |
| ECE raw % | 9.99 | 8.88 | 7.18 | 6.79 | 6.66 | 6.68 | 6.92 | 7.37 |
| ECE TS % | 1.96 | 2.62 | 2.85 | 3.17 | 3.24 | 3.23 | 3.24 | 3.08 |
| NLL TS | 2.707 | 1.747 | 1.588 | 1.561 | 1.562 | 1.573 | 1.586 | 1.600 |
| Brier TS | 0.8020 | 0.5839 | 0.5390 | 0.5302 | 0.5295 | 0.5313 | 0.5340 | 0.5370 |
| AURC TS % | 45.18 | 20.81 | 17.38 | 16.71 | 16.66 | 16.77 | 16.96 | 17.17 |
| temperature | 0.769 | 0.800 | 0.826 | 0.829 | 0.830 | 0.830 | 0.829 | 0.825 |
| cum. GFLOPs | 0.239 | 0.476 | 0.712 | 0.949 | 1.185 | 1.422 | 1.659 | 1.895 |
| latency bs1 ms | 0.48 | 0.60 | 0.78 | 0.99 | 1.48 | 1.72 | 1.78 | 1.87 |
| latency bs256 ms | 0.62 | 1.20 | 1.51 | 2.03 | 2.53 | 2.97 | 3.43 | 3.89 |

Accuracy-compute AUC (mean top-1 % over [0.239, 1.895] GFLOPs, TS probabilities):

- maxprob: 54.65
- entropy: 54.28
- fixed_depth: 56.64

Oracle stopping (t* = earliest step after which every step is correct):

- full: top-1 59.19%, 1.036 GFLOPs, 4.37 steps
- any: top-1 62.56%, 0.360 GFLOPs, 1.51 steps

Latency device: NVIDIA H100 NVL, AMP bf16; bs1 full-depth throughput 536 img/s; bs256 full-depth throughput 65824 img/s
