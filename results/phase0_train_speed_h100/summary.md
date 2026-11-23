# Phase 0: training speed and validation of the fast pipeline (H100, pasted by the user)

Code 0.15.0 (`dev/vcnet`), NVIDIA H100 NVL, torch 2.6.0+cu124, CPU cap 8 of the machine's 48 cores.
Raw output: [`raw.txt`](raw.txt).

## Training speed (`scripts/bench_train.py`, img/s of the full training step, batch 256 unless noted)

| config | orig (2 steps) | x | adapt_blt (8 steps) | x |
|---|---|---|---|---|
| cpu-eager (rounds 1-3) | 9,034 | 1.00 | 1,483 | 1.00 |
| gpu-eager | 8,129 | 0.90 | 1,411 | 0.95 |
| gpu-fused | 9,360 | 1.04 | 1,384 | 0.93 |
| gpu-compile | 8,305 | 0.92 | 3,664 | 2.47 |
| **gpu-cg** (compile + CUDA graphs) | **15,494** | **1.72** | 6,105 | 4.12 |
| cpu-cg | 11,895 | 1.32 | **6,867** | **4.63** |
| gpu-cg, batch 512 | 25,205 | 2.79 | 8,011 | 5.40 |
| gpu-cg, batch 1024 | 33,891 | 3.75 | 9,919 | 6.69 |
| data pipeline alone: CPU (7 workers) / GPU | 11,465 / 64,903 | | 10,947 / 58,190 | |

Three copies at once (`--procs 3`, adapt_blt, total img/s over the copies):

| config | 1 copy | 3 copies (sum) |
|---|---|---|
| cpu-eager | 1,483 | 1,050 |
| gpu-compile | 3,664 | 5,095 |
| gpu-cg | 6,105 | 5,776 |

Reading:

* CUDA graphs are the main gain: 1.7x (orig) and 4.1-4.6x (blt). Compiling without graphs helps blt (2.5x) but
  not orig (0.92x): the 2-step model is bound by kernel launches, which only the graph removes. Fused AdamW alone
  changes little.
* orig with CUDA graphs needs the GPU augmentation: cpu-cg (11,895) sits at the CPU loader's ceiling (11,465 img/s
  with 7 workers), gpu-cg reaches 15,494.
* blt: gpu-cg is 4.65 ms/iteration slower than cpu-cg (41.93 vs 37.28 ms). Cause found in the code: the GPU
  augmentation copied a few constants from host to device every batch (`mean.to(device)`, `torch.tensor(..., device)`),
  each a blocking copy, so the CPU waited for the previous training step before queueing the augmentation and the
  two never overlapped. Fixed in 0.15.1 (constants cached on the device, next batch augmented on a side stream).
* Parallel copies do not add throughput once each copy runs CUDA graphs (5,776 vs 6,105 img/s): without MPS the GPU
  time-slices between processes, so their kernels do not run concurrently. Without graphs, parallel copies hide
  each other's launch gaps (gpu-compile 3,664 -> 5,095). With the CPU loader, 3 copies share 8 cores (2 each) and
  collapse to 1,050 img/s.
* Bigger batches still scale (blt 6,105 -> 9,919 img/s at 1024, orig 15,494 -> 33,891), so a batch-256 step is
  far from filling the GPU; using them needs a learning-rate change and a new validation.
* Compilation costs 25-145 s once per run.

## Validation: the fast pipeline reproduces rounds 1-2 (3 seeds each, `--parallel 3`)

`_FAST` = `--gpu-aug --compile-mode reduce-overhead --opt-impl fused`.

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| adapt_blt (round 2) | 63.72±0.77 | 67.27±0.57 | 68.30±0.48 | 68.68±0.58 | 68.65±0.39 | 68.60±0.29 | 68.56±0.30 | 68.61±0.30 |
| adapt_blt_fast | 63.84±0.53 | 67.62±0.44 | 68.51±0.31 | 68.85±0.49 | 68.85±0.36 | 68.88±0.49 | 68.78±0.47 | 68.85±0.44 |
| difference | +0.12 | +0.35 | +0.21 | +0.17 | +0.20 | +0.28 | +0.22 | +0.24 |

| | step 1 | step 2 | ECE TS step 2 | NLL TS step 2 |
|---|---|---|---|---|
| orig (round 1) | 60.99±0.93 | 64.55±0.96 | 1.79 | 1.266 |
| orig_fast | 61.83±2.26 | 65.48±1.77 | 1.48 | 1.233 |

* adapt_blt_fast is 0.1-0.35 points above round 2 at every step, inside one seed standard deviation (0.3-0.6);
  calibration (ECE / NLL after temperature scaling, temperatures 0.69-0.76) is the same. Pass.
* orig_fast is 0.9 points higher with a larger seed spread (1.8 vs 1.0); the difference is not significant
  (Welch t ~ 0.8). Pass, but the per-seed values would show whether one seed is an outlier.
* Wall time for 3 seeds: adapt_blt_fast 82.6 min (epoch 23.6 s, all three at once) vs about 300 min with the old
  pipeline on an idle GPU (3 x 100.7 min, bench cpu-eager), i.e. 3.7x; orig_fast 30.5 min vs about 50 min (1.6x).
* The evaluate.py latencies of these runs (e.g. blt bs1 4.19 ms at step 1 vs 3.21 ms in round 2) were measured with
  the other runs on the GPU and are not valid; from 0.15.1 `--parallel` runs skip that measurement.

Decision: new experiments use `_FAST`. Rounds 1-3 keep their recipes, so their numbers stay reproducible.
