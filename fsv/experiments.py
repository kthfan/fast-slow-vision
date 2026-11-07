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
    "c100_vit_s_8blk": {
        "desc": "Non-shared baseline: 8 distinct blocks with an exit after each (same FLOPs, ~8x block params); "
                "measures the cost of weight sharing.",
        "train": _BASE + ["--n-blocks", "8"],
    },
}
