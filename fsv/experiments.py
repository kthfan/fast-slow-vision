"""Named experiments run by ``scripts/run_experiment.py``.

Each entry fixes the full training recipe so that a run on the user's machine
is reproducible from its name alone. Add new entries here instead of passing
ad-hoc flags, so every uploaded report maps to a recipe in the repository.
"""

from __future__ import annotations

# Recipe of the first run (Kaggle 2x T4, Nov 2026): 61.3% best-step test top-1.
_BASE = [
    "--dataset", "cifar100", "--model", "bvit_s", "--t-max", "8",
    "--epochs", "200", "--batch-size", "256", "--lr", "1e-3", "--wd", "0.05", "--warmup-epochs", "10",
]

# VCNet (docs/vcnet.md): same CIFAR recipe for the original and the adaptive model.
_VC = [
    "--dataset", "cifar100", "--model", "vcnet",
    "--epochs", "200", "--batch-size", "256", "--lr", "1e-3", "--wd", "0.05", "--warmup-epochs", "10",
]
# original VCNet: bottom-up sweep + one predictive-coding correction, read out once, plain cross-entropy
_VC_ORIG = _VC + ["--t-max", "2", "--loss-steps", "last", "--beta", "0", "--gamma", "0", "--delta", "0"]
# adaptive VCNet: 8 predictive-coding steps, a readout after every step, Stage A loss
_VC_ADAPT = _VC + ["--t-max", "8", "--step-weights", "quadratic"]
# phase 0 (0.15.0) fast training pipeline: augmentation on the GPU (fsv/gpu_aug.py), forward + loss
# compiled with CUDA graphs, fused AdamW. Same recipe otherwise; validated by the *_fast runs below
# against rounds 1-2 before new experiments adopt it.
_FAST = ["--gpu-aug", "--compile-mode", "reduce-overhead", "--opt-impl", "fused"]


def _set(args: list[str], flag: str, value: str) -> list[str]:
    """Copy of a recipe with ``flag``'s value replaced."""
    i = args.index(flag)
    return [*args[:i + 1], value, *args[i + 2:]]


# phase 1 baselines: the CIFAR ResNet with one step and plain cross-entropy
# (a) the VCNet recipe unchanged (AdamW, RandAugment, random erasing, Mixup / CutMix): same training for every model
_RN_FAST = _set(_set(_VC_ORIG, "--model", "resnet56"), "--t-max", "1") + _FAST


def _sgd(args: list[str]) -> list[str]:
    """(b) the standard CIFAR ResNet recipe (He et al. 2016, cosine schedule): SGD + Nesterov, lr 0.1 per 128
    images (0.2 at batch 256), weight decay 5e-4, crop + flip only, no clipping. Checks the implementation
    against the literature (ResNet-56 on CIFAR-100: about 71-72%; here trained on 45k images, 5k held out)."""
    for flag, value in (("--lr", "0.2"), ("--wd", "5e-4"), ("--warmup-epochs", "1")):
        args = _set(args, flag, value)
    return args + ["--opt", "sgd", "--clip", "0", "--no-randaug", "--erase", "0", "--mixup", "0", "--cutmix", "0"]


_RN_SGD = _sgd(_RN_FAST)

EXPERIMENTS: dict[str, dict] = {
    "smoke": {
        "desc": "Pipeline check on real CIFAR-100: 1 epoch capped at 100 iterations (~1-2 min on one GPU).",
        "train": ["--dataset", "cifar100", "--model", "bvit_s", "--t-max", "8", "--epochs", "1",
                  "--max-steps", "100", "--eval-every", "1"],
    },
    "c100_bvit_s": {
        "desc": "bViT-S baseline, same recipe as the Kaggle run (reproduction on a single GPU).",
        "train": _BASE,
    },
    "c100_bvit_s_lightaug": {
        "desc": "bViT-S with lighter augmentation (mixup 0.2 only, no CutMix, no random erasing); "
                "tests whether the baseline underfits because of over-regularisation.",
        "train": _BASE + ["--mixup", "0.2", "--cutmix", "0", "--erase", "0"],
    },
    "c100_bvit_b": {
        "desc": "Wider shared block (bViT-B: dim 768, 12 heads, ~7.2M params, ~4x FLOPs per step); "
                "bViT paper: width closes most of the gap to a non-shared ViT.",
        "train": [*_BASE[:3], "bvit_b", *_BASE[4:]],
    },
    # ---- round 3: precision check and attempts to make later recurrent steps useful
    "c100_bvit_s_fp16": {
        "desc": "bViT-S baseline trained in fp16 (+GradScaler) instead of bf16; the Kaggle fp16 run was ~2 points higher.",
        "train": _BASE + ["--amp", "fp16"],
    },
    "c100_bvit_s_te": {
        "desc": "bViT-S + learned per-step input time embedding (bViT-TE), so the shared block knows which step it is.",
        "train": _BASE + ["--step-embed-in"],
    },
    "c100_bvit_s_inject": {
        "desc": "bViT-S + gated re-injection of the patch embeddings before every step (zero-init gate).",
        "train": _BASE + ["--input-inject"],
    },
    "c100_bvit_s_linw": {
        "desc": "bViT-S with per-step loss weights growing linearly with t (later steps weigh more).",
        "train": _BASE + ["--step-weights", "linear"],
    },
    "c100_bvit_s_nosd": {
        "desc": "bViT-S without last-step self-distillation (delta = 0).",
        "train": _BASE + ["--delta", "0"],
    },
    # ---- round 4: build on linear step weights (round 3: +2.8 points); explain the Kaggle gap
    "c100_bvit_s_quadw": {
        "desc": "bViT-S with quadratic per-step loss weights, w_t ∝ (t+1)^2 (even more weight on late steps).",
        "train": _BASE + ["--step-weights", "quadratic"],
    },
    "c100_bvit_s_linw_te_inject": {
        "desc": "bViT-S with linear step weights + time embedding + input injection (stacks the round-3 gains).",
        "train": _BASE + ["--step-weights", "linear", "--step-embed-in", "--input-inject"],
    },
    "c100_vit_s_8blk_linw": {
        "desc": "Non-shared 8-block ViT with linear step weights (fair comparison with c100_bvit_s_linw).",
        "train": _BASE + ["--n-blocks", "8", "--step-weights", "linear"],
    },
    "c100_bvit_s_mix2": {
        "desc": "bViT-S baseline with 2 independent mixup/CutMix groups per batch, emulating the 2-GPU Kaggle run.",
        "train": _BASE + ["--mix-groups", "2"],
    },
    # ---- round 5: linear step weights help on average (+1.6) but one seed in three gained nothing
    "c100_bvit_s_linw_lr5e4": {
        "desc": "bViT-S, linear step weights, learning rate 5e-4 instead of 1e-3 (is the seed variance an optimisation issue?).",
        "train": _BASE + ["--step-weights", "linear", "--lr", "5e-4"],
    },
    # ---- round 6: lr 5e-4 stabilised linear weights (61.25±0.36); controls for the learning rate
    "c100_bvit_s_lr5e4": {
        "desc": "bViT-S baseline (uniform step weights) at learning rate 5e-4: control for c100_bvit_s_linw_lr5e4.",
        "train": _BASE + ["--lr", "5e-4"],
    },
    "c100_bvit_s_quadw_lr5e4": {
        "desc": "bViT-S, quadratic step weights, learning rate 5e-4 (quadw had the best mean but high variance at 1e-3).",
        "train": _BASE + ["--step-weights", "quadratic", "--lr", "5e-4"],
    },
    "c100_vit_s_8blk": {
        "desc": "Non-shared baseline: 8 distinct blocks with an exit after each (same FLOPs, ~8x block params); "
                "measures the cost of weight sharing.",
        "train": _BASE + ["--n-blocks", "8"],
    },
    # ---- VCNet: original (fixed computation) vs adaptive predictive-coding steps
    "vcnet_smoke": {
        "desc": "VCNet pipeline check on real CIFAR-100: adaptive variant, 1 epoch capped at 100 iterations.",
        "train": ["--dataset", "cifar100", "--model", "vcnet", "--t-max", "8", "--epochs", "1",
                  "--max-steps", "100", "--eval-every", "1"],
    },
    "c100_vcnet_orig": {
        "desc": "Original VCNet (~1.0M params): bottom-up sweep + one AIT->V1 predictive-coding correction, "
                "single readout, cross-entropy only.",
        "train": _VC_ORIG,
    },
    "c100_vcnet_orig_ff": {
        "desc": "Original VCNet without the top-down loop at inference (one bottom-up sweep): "
                "how much does the feedback correction add?",
        "train": [*_VC_ORIG[:_VC_ORIG.index("--t-max") + 1], "1", *_VC_ORIG[_VC_ORIG.index("--t-max") + 2:]],
    },
    "c100_vcnet_adapt": {
        "desc": "Adaptive VCNet: up to 8 predictive-coding steps, shared readout after every step, "
                "Stage A loss (CE + Brier + Gatekeeper + self-distillation, quadratic step weights).",
        "train": _VC_ADAPT,
    },
    "c100_vcnet_adapt_carry": {
        "desc": "Adaptive VCNet + AIT state carried across steps (AIT_t += g * AIT_{t-1}, zero-init gate).",
        "train": _VC_ADAPT + ["--ait-carry"],
    },
    # ---- VCNet round 2: state across steps in every area (docs/vcnet.md, "跨步狀態")
    "c100_vcnet_adapt_lat": {
        "desc": "Adaptive VCNet + every area (V2, MT, V4, PIT, CIT, MST, AIT) keeps its own state across steps "
                "(A_t = F(input_t) + R(A_{t-1}), 1x1 zero-init).",
        "train": _VC_ADAPT + ["--lateral"],
    },
    "c100_vcnet_adapt_blt": {
        "desc": "Adaptive VCNet + per-area state + BLT-style top-down from the next area up (previous step).",
        "train": _VC_ADAPT + ["--lateral", "--topdown"],
    },
    # ---- VCNet round 3: the blt architecture trained like the original VCNet (fixed steps, one readout, CE only),
    # to separate the architecture gain from the per-step-readout + halting gain
    "c100_vcnet_blt_orig2": {
        "desc": "Same architecture as c100_vcnet_adapt_blt, trained like the original VCNet: "
                "2 fixed steps, readout at the last step only, cross-entropy only.",
        "train": [*_VC_ORIG, "--lateral", "--topdown"],
    },
    "c100_vcnet_blt_orig4": {
        "desc": "Same as c100_vcnet_blt_orig2 with 4 fixed steps (the best fixed depth of c100_vcnet_adapt_blt).",
        "train": [*_VC_ORIG[:_VC_ORIG.index("--t-max") + 1], "4", *_VC_ORIG[_VC_ORIG.index("--t-max") + 2:],
                  "--lateral", "--topdown"],
    },
    # ---- phase 0: the fast training pipeline must reproduce rounds 1-2 (run 3 seeds with --parallel 3)
    "c100_vcnet_orig_fast": {
        "desc": "c100_vcnet_orig with the fast pipeline (GPU augmentation + compiled step with CUDA graphs + "
                "fused AdamW); must match 64.55% (round 1, 3 seeds).",
        "train": _VC_ORIG + _FAST,
    },
    "c100_vcnet_adapt_blt_fast": {
        "desc": "c100_vcnet_adapt_blt with the fast pipeline; must match round 2 (67.27% at 2 steps, "
                "68.68% at 4 steps, 3 seeds).",
        "train": _VC_ADAPT + ["--lateral", "--topdown"] + _FAST,
    },
    # ---- phase 0, step 2: bigger batches (bench_train: blt 6.6k -> 8.3k / 10.2k img/s at 512 / 1024).
    # AdamW learning rate scaled with sqrt(batch / 256); same epochs and warmup epochs (fewer updates).
    # Result (results/phase0_train_speed_h100): -1.0 (512) and -3.4 (1024) points; batch 256 stays the recipe.
    "c100_vcnet_adapt_blt_fast_bs512": {
        "desc": "c100_vcnet_adapt_blt_fast with batch 512, lr 1.4e-3 (sqrt scaling); must match the batch-256 run.",
        "train": _VC_ADAPT + ["--lateral", "--topdown"] + _FAST + ["--batch-size", "512", "--lr", "1.4e-3"],
    },
    "c100_vcnet_adapt_blt_fast_bs1024": {
        "desc": "c100_vcnet_adapt_blt_fast with batch 1024, lr 2e-3 (sqrt scaling); must match the batch-256 run.",
        "train": _VC_ADAPT + ["--lateral", "--topdown"] + _FAST + ["--batch-size", "1024", "--lr", "2e-3"],
    },
    # ---- phase 1: feed-forward CNN baseline (ResNet-56: 0.862M params, 0.252 GFLOPs) and VCNet at the same size
    "c100_resnet56_fast": {
        "desc": "ResNet-56 (He et al. 2016, option-A shortcuts, BatchNorm) with the VCNet recipe and the fast "
                "pipeline: AdamW, RandAugment, erasing, Mixup / CutMix, cross-entropy.",
        "train": _RN_FAST,
    },
    "c100_resnet56_sgd": {
        "desc": "ResNet-56 with the standard CIFAR recipe (SGD + Nesterov lr 0.2 at batch 256, wd 5e-4, crop + flip "
                "only); reference to the literature (~71-72% on CIFAR-100).",
        "train": _RN_SGD,
    },
    "c100_resnet56_gn_fast": {
        "desc": "c100_resnet56_fast with GroupNorm instead of BatchNorm (VCNet uses GroupNorm): "
                "separates the normalisation from the architecture.",
        "train": _RN_FAST + ["--resnet-norm", "gn"],
    },
    "c100_vcnet_r56p_blt_fast": {
        "desc": "VCNet blt (adaptive, 8 steps) at ResNet-56's parameter count (vcnet_r56p: 0.864M; "
                "0.232 / 0.327 GFLOPs at 2 / 3 steps), fast pipeline.",
        "train": _set(_VC_ADAPT, "--model", "vcnet_r56p") + ["--lateral", "--topdown"] + _FAST,
    },
    "c100_vcnet_r56f4_blt_fast": {
        "desc": "VCNet blt (adaptive, 8 steps) at ResNet-56's FLOPs after 4 steps (vcnet_r56f4: 0.545M params, "
                "0.249 GFLOPs at 4 steps), fast pipeline.",
        "train": _set(_VC_ADAPT, "--model", "vcnet_r56f4") + ["--lateral", "--topdown"] + _FAST,
    },
    "c100_vcnet_r56p_blt_sgd": {
        "desc": "c100_vcnet_r56p_blt_fast with the standard ResNet recipe (SGD, crop + flip only); "
                "only needed if c100_resnet56_sgd beats c100_resnet56_fast.",
        "train": _sgd(_set(_VC_ADAPT, "--model", "vcnet_r56p") + ["--lateral", "--topdown"] + _FAST),
    },
}
