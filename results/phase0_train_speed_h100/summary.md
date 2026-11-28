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

## Second measurement (0.15.1: no host syncs in the GPU augmentation, side-stream prefetch, channels_last)

| config | orig img/s | x | adapt_blt img/s | x |
|---|---|---|---|---|
| cpu-eager | 8,382 | 1.00 | 1,484 | 1.00 |
| cpu-cg | 12,994 | 1.55 | **6,874** | **4.63** |
| gpu-cg-noprefetch (syncs removed) | 21,884 | 2.61 | 6,616 | 4.46 |
| **gpu-cg** (+ side stream) | **24,387** | **2.91** | 6,577 | 4.43 |
| cpu-cg-cl / gpu-cg-cl (channels_last) | 13,955 / 14,451 | 1.66 / 1.72 | 4,489 / 4,363 | 3.03 / 2.94 |
| gpu-cg, batch 512 | 31,024 | 3.70 | 8,280 | 5.58 |
| gpu-cg, batch 1024 | 37,641 | 4.49 | 10,219 | 6.89 |

* The fix worked for orig: gpu-cg 15,494 -> 24,387 img/s (2.9x the old pipeline; 6 min per run). Removing the host
  syncs gives most of it, the side stream another 11%.
* blt: the GPU augmentation is now within 4% of the CPU one (6,577 vs 6,874). The blt step is bound by GPU time,
  not by the CPU, so the augmentation's own GPU work (43k img/s alone) is no longer hidden.
* channels_last is slower for both (-35 to -41%). Not used.
* Profile of the blt step (compiled, no graph; 40 ms of GPU time per iteration): convolutions dominate, as ~360
  small convolutions forward and ~360 backward per iteration (8 steps), 20-50 us each; cuDNN converts layouts
  NCHW <-> NHWC around them (~20% of GPU time, but channels_last made the rest slower). The single most expensive
  kernel is the backward of the AIT -> V1 bilinear upsampling (Inductor: index_put with atomic adds, 526 us per call,
  7 per iteration = ~9%). 0.15.2 computes the upsampling in training as two small matmuls (same values; inference
  unchanged).
* Bigger batches remain the largest lever (blt 1.55x, orig 1.54x from 256 to 1024); validation experiments
  c100_vcnet_adapt_blt_fast_bs512 / _bs1024 (sqrt-scaled learning rate).
* Parallel copies: no gain once CUDA graphs are on (first measurement), so the fast runs should go one after another.

## Third measurement and the batch-size validation (0.15.2)

Speed of the blt step after the upsampling change: gpu-cg 6,577 -> 6,858 img/s (+4%), cpu-cg 6,874 -> 6,986 (+2%),
batch 1024 10,219 -> 11,102 (+9%). Smaller than the ~9% of GPU time the profile attributed to the upsampling
backward at batch 256; the two paths are now within 2%.

Larger batches lose accuracy (learning rate scaled with sqrt(batch / 256), same epochs and warmup epochs):

| step | 1 | 2 | 3 | 4 | 8 | train min / run |
|---|---|---|---|---|---|---|
| adapt_blt_fast, batch 256 (reference) | 63.84±0.53 | 67.62±0.44 | 68.51±0.31 | 68.85±0.49 | 68.85±0.44 | 27.5 (82.6 / 3 in parallel) |
| batch 512, lr 1.4e-3 | 62.36±0.30 | 66.67±0.39 | 67.58±0.53 | 67.83±0.37 | 67.87±0.43 | 20.8 |
| batch 1024, lr 2e-3 | 60.24±0.32 | 63.96±0.44 | 65.11±0.47 | 65.42±0.40 | 65.35±0.26 | 15.5 |

* Batch 512 is 1.0 point below batch 256 at every step from step 2 on (about two seed standard deviations);
  batch 1024 is 3.4 points below. Both fail the validation.
* The final training loss and the held-out accuracy point to under-training (4x fewer updates at batch 1024 with
  the same 200 epochs), not to instability: loss 3.34 / 3.29 vs 3.24, smooth per-step curves, calibration
  unchanged after temperature scaling.
* Recovering it would need a higher learning rate (linear scaling) and/or more epochs, which eats most of the
  1.2x (512) / 1.6x (1024) speed gain. Not pursued: the training recipe stays at batch 256.
* The bs512 run's evaluate.py batch-256 latencies are disturbed (standard deviation up to 80% of the mean); not used.

## Phase 0 outcome

The default training recipe for new experiments is `_FAST` at batch 256: augmentation on the GPU, forward + loss
compiled with CUDA graphs, fused AdamW, runs one after another. Against the recipe of rounds 1-3 on the same idle
GPU: orig 2.9x (about 6 min per run), adapt_blt 4.6x (about 22 min per run; 3 seeds about 65 min instead of 300).
Accuracy reproduces rounds 1-2 within the seed spread.
