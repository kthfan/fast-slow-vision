# Phase 1, first round: ResNet-56 vs VCNet at the same size (H100, pasted by the user)

Code 0.17.0 (`dev/vcnet`), NVIDIA H100 NVL, torch 2.6.0+cu124. Raw output: [`raw.txt`](raw.txt).
`c100_resnet56_fast` seed 0 has no results, so its numbers are seeds 1-2 only. Its training died with
`CUDA error: an illegal memory access was encountered` (exit -6 after 32.7 min, vs ~9.5 min for a full run of the
other seeds; the error is asynchronous, so the reported line in `GPUAugment.random_erase` is only where the CPU
noticed it). One failure in about 30 runs with the same pipeline (GPU augmentation + CUDA graphs), including the
other 5 ResNet-56 runs with BatchNorm; to be rerun with `--fresh` to see whether it is reproducible.

## CIFAR-100 test accuracy (3 seeds; 45k training images)

| model | params | GFLOPs | acc % | ECE TS % | NLL TS |
|---|---|---|---|---|---|
| **ResNet-56, SGD recipe** | 0.862M | 0.251 | **71.25±0.20** | 1.81 | 1.074 |
| ResNet-56, VCNet recipe (2 seeds) | 0.862M | 0.251 | 70.68±0.04 | 1.32 | 1.053 |
| vcnet_r56p blt, 2 steps | 0.864M | 0.232 | 65.49±0.37 | 2.30 | 1.256 |
| vcnet_r56p blt, 3 steps | 0.864M | 0.327 | 66.55±0.33 | 2.36 | 1.216 |
| vcnet_r56p blt, best (5 steps) | 0.864M | 0.517 | 67.17±0.13 | 2.14 | 1.202 |
| vcnet_r56f4 blt, 4 steps | 0.545M | 0.249 | 63.01±0.86 | 2.63 | 1.338 |
| blt full width, 4 steps (phase 0, adapt_blt_fast) | 1.150M | 0.528 | 68.85±0.49 | | |

* ResNet-56 reproduces the literature (71-72% on 50k images) and beats every VCNet: -5.8 points at the same
  parameters and ~the same compute (r56p, 2 steps), -8.2 points at the same compute (r56f4, 4 steps), and still
  -2.4 points for the full-width blt with 1.33x the parameters and 2.1x the compute.
* Even oracle stopping (the best step per image, known labels) of r56p reaches 70.01%, below ResNet-56.
* The training recipe is not the cause on the ResNet side: the VCNet recipe costs ResNet-56 only 0.57 points.
  It may matter on the VCNet side (open: c100_vcnet_r56p_blt_sgd).
* Calibration after temperature scaling is also better for ResNet-56 (ECE 1.3-1.8 vs 2.1-2.6 %, NLL 1.05 vs 1.20).
* Wall time: ResNet-56 trains in 8 min (2.3 s/epoch) vs 31 min for r56p; evaluate.py eager latency at batch 256
  4.1 ms vs 18.9 ms for r56p at 2 steps.
* The narrower VCNets are slower to train than the full-width one (r56p 8.8 s/epoch vs about 6.6 s): their channel
  counts (28, 36, 44, 68, 108, 136) are not multiples of 8 or 16, which bf16 convolutions need for fast kernels.

Verification item 1 (beat ResNet-56 at the same parameters and compute) fails by a wide margin.

## Second paste: seed-0 rerun, GroupNorm control, VCNet with the SGD recipe

* The `--fresh` rerun of `c100_resnet56_fast` seed 0 finished normally (9.6 min, the same augmentation stream), so
  the illegal memory access was a one-off. With 3 seeds: 70.44±0.42% (seed 0 about 69.96%).

| model (normalisation) | params | GFLOPs | acc % | NLL TS |
|---|---|---|---|---|
| ResNet-56 BN, SGD recipe | 0.862M | 0.251 | 71.25±0.20 | 1.074 |
| ResNet-56 BN, VCNet recipe | 0.862M | 0.251 | 70.44±0.42 | 1.059 |
| **ResNet-56 GN, VCNet recipe** | 0.862M | 0.251 | **62.84±0.45** | 1.321 |
| vcnet_r56p (GN), 2 / 3 steps | 0.864M | 0.232 / 0.327 | 65.49 / 66.55 | 1.256 / 1.216 |
| vcnet_r56f4 (GN), 4 steps | 0.545M | 0.249 | 63.01±0.86 | 1.338 |
| vcnet_r56p (GN), SGD recipe, best step | 0.864M | | 62.27±3.60 | |

* GroupNorm costs ResNet-56 7.6 points (and 1.56x the batch-256 latency, 6.32 vs 4.05 ms). With the same
  normalisation, VCNet is ahead: vcnet_r56p beats ResNet-56-GN by 2.65 points at 2 steps (with 8% fewer FLOPs) and
  by 3.7 points at 3 steps; vcnet_r56f4 ties it at the same FLOPs with 0.63x the parameters.
* So most of the 5.8-point gap of the first paste is the normalisation, not the architecture. Caveat: GN may hurt
  a 56-layer plain-depth network more than the shallower VCNet sweep, so "VCNet > ResNet at equal GN" is not yet
  "VCNet can beat the best ResNet".
* The SGD recipe does not suit VCNet: 4.7 points lower at the best step, one unstable seed (step-1 accuracy
  56.8±8.5), the final training loss 0.81±0.54 (overfitting without Mixup / CutMix) and over-confident
  (temperatures 1.1-1.4). VCNet keeps the AdamW recipe.

Next (0.18.0): GroupNorm + weight standardisation (keeps the TODO.md no-BatchNorm rule) for ResNet-56 and VCNet.

## Third paste (0.18.0): GroupNorm + weight standardisation

| model | params | GFLOPs | GN | GN + WS | WS gain |
|---|---|---|---|---|---|
| ResNet-56 | 0.862M | 0.251 | 62.84±0.45 | **68.64±0.33** | +5.80 |
| vcnet_r56p, 2 steps | 0.864M | 0.232 | 65.49±0.37 | 66.84±0.40 | +1.35 |
| vcnet_r56p, 3 steps | 0.864M | 0.327 | 66.55±0.33 | 67.82±0.32 | +1.27 |
| vcnet_r56p, best (4-5 steps) | 0.864M | 0.42-0.52 | 67.17±0.13 | 67.94±0.39 | +0.77 |
| vcnet_r56f4, 4 steps | 0.545M | 0.249 | 63.01±0.86 | 63.86±0.33 | +0.85 |
| (ResNet-56 BN, VCNet recipe) | 0.862M | 0.251 | 70.44±0.42 | | |

* WS recovers 5.8 of the 7.6 points GroupNorm cost ResNet-56; it helps VCNet by only 0.8-1.4 points.
* With the same normalisation (GN + WS, allowed by TODO.md), ResNet-56 leads again: by 1.8 points over vcnet_r56p
  at the same parameters and ~the same compute (2 steps, 0.232 GFLOPs), by 0.8 points even when VCNet spends 1.3x the
  compute (3 steps), and by 4.8 points over vcnet_r56f4 at the same compute. Verification item 1 still fails, but the
  gap at matched parameters is now 1-2 points instead of 5.8.
* The oracle stopping of vcnet_r56p + WS (71.00% at 0.153 GFLOPs) exceeds ResNet-56 GN + WS, so per-image
  adaptivity has room, but no trained policy realises it yet.
* Compute allocation (vcnet_r56p, 2 steps, flop counter): V1's lateral interaction 22.5% (full resolution, once per
  image), V2 28.4% (16x16), MT 10.9%, PIT 9.2%; CIT + AIT (4x4) only 9.6%. ResNet-56 splits its compute evenly over
  32 / 16 / 8 pixels and ends at 8x8 with 54 conv layers in sequence.
* The batch-256 latencies of c100_vcnet_r56p_blt_ws_fast have standard deviations up to 80% of the mean
  (disturbed measurement), and its training time varies (32.1±8.5 min); not used.
