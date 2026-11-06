"""Check that this machine can run the experiments.

    python scripts/check_env.py --max-cpus 8

Prints the torch / CUDA / GPU setup and the CPU cap, then runs a short
bf16 forward/backward pass of bViT-S on the GPU. Exit code 0 means ready.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsv.cpu import available_cpus, limit_cpus  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-cpus", type=int, default=None)
    p.add_argument("--allow-cpu", action="store_true", help="do not fail when no GPU is visible")
    args = p.parse_args()

    machine = len(available_cpus())
    n = limit_cpus(args.max_cpus)
    import torch

    from fsv.engine import auto_amp, autocast_ctx
    from fsv.losses import StageALossConfig, stage_a_loss
    from fsv.models import build_model

    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__} (CUDA build {torch.version.cuda})")
    print(f"CPU cap     {n} of {machine} cores; affinity {len(os.sched_getaffinity(0))} cores; "
          f"torch threads {torch.get_num_threads()}")
    if not torch.cuda.is_available():
        print("GPU         none visible")
        if not args.allow_cpu:
            sys.exit("FAIL: no CUDA GPU visible. Install a CUDA build of torch (see README) or check drivers.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        cap = torch.cuda.get_device_capability()
        print(f"GPU         {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB)")
    amp = auto_amp(device)
    print(f"AMP         {amp}")

    model = build_model("bvit_s", num_classes=100, t_max=8).to(device)
    x = torch.randn(256, 3, 32, 32, device=device)
    target = torch.nn.functional.one_hot(torch.randint(0, 100, (256,), device=device), 100).float()
    for i in range(3):
        t0 = time.time()
        with autocast_ctx(device, amp):
            logits = model(x)
        loss, _ = stage_a_loss(logits, target, StageALossConfig())
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"step {i}      loss {loss.item():.3f}  {(time.time() - t0) * 1e3:.0f} ms (batch 256, 8 steps)")
    if not torch.isfinite(loss):
        sys.exit("FAIL: non-finite loss")
    print("OK: environment ready")


if __name__ == "__main__":
    main()
