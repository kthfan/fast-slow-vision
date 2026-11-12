# Report: c100_bvit_s_lightaug

bViT-S with lighter augmentation (mixup 0.2 only, no CutMix, no random erasing); tests whether the baseline underfits because of over-regularisation.

## Environment

- git: `?` on `?` (**uncommitted changes**)
- GPU: NVIDIA H100 NVL | torch 2.6.0+cu124 (CUDA 12.4) | Python 3.10.12
- CPU cap: 8 cores (machine has 48); DataLoader workers 7
- AMP: bf16 | params: 1.86M
- started 2026-11-06T21:38:46, train 15.0 min, eval 0.1 min, exit codes train=0 eval=0
- train command: `scripts/train.py --dataset cifar100 --model bvit_s --t-max 8 --epochs 200 --batch-size 256 --lr 1e-3 --wd 0.05 --warmup-epochs 10 --mixup 0.2 --cutmix 0 --erase 0 --data-dir /workspace/fast-slow-vision/data --out /workspace/fast-slow-vision/runs/c100_bvit_s_lightaug --max-cpus 8 --resume`

## Training

- epochs run: 200; mean epoch 4.3s; mean throughput 11680 img/s

Held-out top-1 % per step (no temperature scaling):

| epoch | train loss | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 4.647 | 21.2 | 23.9 | 23.9 | 23.7 | 23.4 | 23.2 | 23.1 | 22.9 |
| 19 | 4.309 | 26.5 | 32.3 | 33.3 | 33.4 | 32.9 | 32.7 | 32.6 | 32.5 |
| 29 | 4.056 | 28.1 | 36.8 | 37.7 | 37.5 | 37.2 | 37.1 | 36.9 | 36.8 |
| 39 | 3.945 | 30.3 | 38.8 | 40.3 | 41.0 | 41.2 | 41.2 | 41.0 | 40.8 |
| 49 | 3.763 | 29.1 | 40.8 | 43.1 | 43.7 | 43.8 | 43.6 | 43.3 | 43.0 |
| 59 | 3.560 | 29.4 | 41.9 | 44.0 | 44.4 | 44.2 | 44.0 | 43.9 | 43.7 |
| 69 | 3.312 | 31.1 | 46.1 | 48.4 | 49.0 | 49.2 | 48.9 | 48.5 | 48.5 |
| 79 | 3.298 | 30.7 | 46.6 | 50.5 | 50.7 | 50.8 | 50.7 | 50.6 | 50.4 |
| 89 | 3.070 | 30.9 | 47.4 | 50.4 | 51.4 | 51.1 | 50.8 | 50.4 | 50.4 |
| 99 | 3.082 | 30.9 | 47.4 | 51.4 | 52.3 | 51.7 | 51.4 | 51.1 | 51.3 |
| 109 | 2.839 | 30.7 | 49.3 | 52.3 | 52.7 | 52.4 | 52.1 | 51.6 | 51.0 |
| 119 | 2.733 | 31.6 | 49.9 | 53.1 | 53.5 | 53.0 | 53.0 | 53.0 | 52.7 |
| 129 | 2.653 | 32.0 | 49.8 | 54.1 | 54.3 | 53.9 | 53.9 | 53.7 | 53.3 |
| 139 | 2.596 | 32.9 | 51.0 | 54.5 | 54.6 | 54.2 | 54.0 | 53.5 | 53.2 |
| 149 | 2.384 | 33.8 | 51.3 | 54.4 | 54.3 | 54.2 | 53.6 | 53.3 | 53.1 |
| 159 | 2.350 | 33.5 | 50.8 | 54.5 | 54.9 | 54.8 | 54.2 | 53.7 | 53.4 |
| 169 | 2.237 | 33.7 | 51.1 | 55.4 | 55.6 | 55.1 | 54.8 | 54.8 | 54.4 |
| 179 | 2.317 | 33.7 | 51.5 | 55.3 | 55.8 | 55.4 | 55.0 | 54.8 | 54.7 |
| 189 | 2.366 | 33.9 | 51.9 | 55.7 | 56.1 | 55.8 | 55.1 | 54.9 | 54.4 |
| 199 | 2.267 | 34.1 | 52.2 | 55.8 | 56.3 | 55.8 | 55.3 | 54.9 | 54.6 |

## Test set, per step

| | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 |
|---|---|---|---|---|---|---|---|---|
| top-1 % | 35.69 | 53.54 | 56.87 | 56.94 | 56.95 | 56.74 | 56.72 | 56.49 |
| ECE raw % | 2.56 | 4.89 | 5.50 | 5.44 | 4.96 | 4.89 | 4.62 | 4.54 |
| ECE TS % | 1.37 | 1.44 | 1.38 | 1.47 | 1.54 | 1.70 | 1.66 | 1.63 |
| NLL TS | 2.639 | 1.878 | 1.762 | 1.760 | 1.775 | 1.790 | 1.802 | 1.812 |
| Brier TS | 0.7794 | 0.5976 | 0.5624 | 0.5594 | 0.5622 | 0.5659 | 0.5691 | 0.5719 |
| AURC TS % | 41.39 | 21.74 | 18.88 | 18.79 | 18.94 | 19.22 | 19.44 | 19.66 |
| temperature | 0.961 | 1.101 | 1.113 | 1.102 | 1.093 | 1.088 | 1.086 | 1.084 |
| cum. GFLOPs | 0.239 | 0.476 | 0.712 | 0.949 | 1.185 | 1.422 | 1.659 | 1.895 |
| latency bs1 ms | 0.51 | 0.71 | 1.04 | 1.24 | 1.28 | 1.31 | 1.44 | 1.70 |
| latency bs256 ms | 0.62 | 1.06 | 1.54 | 2.03 | 2.50 | 2.99 | 3.45 | 3.92 |

Accuracy-compute AUC (mean top-1 % over [0.239, 1.895] GFLOPs, TS probabilities):

- maxprob: 53.50
- entropy: 52.98
- fixed_depth: 54.93

Oracle stopping (t* = earliest step after which every step is correct):

- full: top-1 56.49%, 1.074 GFLOPs, 4.53 steps
- any: top-1 61.75%, 0.353 GFLOPs, 1.48 steps

Latency device: NVIDIA H100 NVL, AMP bf16; bs1 full-depth throughput 590 img/s; bs256 full-depth throughput 65333 img/s
