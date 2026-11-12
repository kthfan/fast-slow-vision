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
}
