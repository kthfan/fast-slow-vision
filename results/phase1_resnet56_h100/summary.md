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
