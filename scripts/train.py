"""Stage A training: recurrent bViT with per-step readout on CIFAR.

Single GPU (e.g. one H100) or CPU:
    python scripts/train.py --out runs/debug --max-cpus 8
Multi-GPU:
    torchrun --nproc_per_node=2 scripts/train.py --out runs/c100_bvit_s

CPU usage is capped by --max-cpus (or FSV_MAX_CPUS, default 8); see fsv/cpu.py.
Normally launched through scripts/run_experiment.py, which also writes the report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsv.cpu import default_workers, limit_cpus  # noqa: E402
from fsv.data import NUM_CLASSES, build_splits  # noqa: E402
from fsv.engine import auto_amp, autocast_ctx, collect_outputs  # noqa: E402
from fsv.losses import StageALossConfig, stage_a_loss  # noqa: E402
from fsv.metrics import per_step_metrics  # noqa: E402
from fsv.mixup import Mixup  # noqa: E402
from fsv.models import build_model, config_dict  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--dataset", default="cifar100", choices=list(NUM_CLASSES))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--holdout", type=int, default=5000)
    p.add_argument("--fake-data", action="store_true", help="random images, for smoke tests")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: max-cpus - 1)")
    p.add_argument("--max-cpus", type=int, default=None, help="hard CPU-core cap (default: $FSV_MAX_CPUS or 8)")
    p.add_argument("--no-randaug", action="store_true")
    p.add_argument("--erase", type=float, default=0.25)
    # model
    p.add_argument("--model", default="bvit_s")
    p.add_argument("--t-max", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=1,
                   help="distinct blocks cycled over steps: 1 = bViT (shared), t_max = standard ViT with exits")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--norm", default="layernorm", choices=["layernorm", "rmsnorm"])
    p.add_argument("--step-embed-in", action="store_true", help="bViT-TE input time embedding")
    p.add_argument("--layerscale-init", type=float, default=0.0)
    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256, help="global batch size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=10)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay; 0 disables")
    p.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "none"])
    p.add_argument("--compile", action="store_true")
    # augmentation
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--mix-prob", type=float, default=1.0)
    # loss (TODO.md section 3, Stage A)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--gk-alpha", type=float, default=0.5)
    p.add_argument("--step-weights", default="uniform", choices=["uniform", "linear"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    # bookkeeping
    p.add_argument("--out", default="runs/debug")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many iterations (probe runs)")
    p.add_argument("--time-limit", type=float, default=0, help="minutes; stop cleanly after the epoch that exceeds it")
    p.add_argument("--resume", action="store_true", help="resume from <out>/last.pt if present")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- utils
def setup_dist():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def barrier(world):
    if world > 1:
        dist.barrier()


def param_groups(model, wd):
    skip = model.no_weight_decay()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n in skip or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(it, total, warmup, base, min_lr):
    if it < warmup:
        return base * (0.033 + (1 - 0.033) * it / max(1, warmup))
    progress = (it - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def summarize_steps(rows):
    return " ".join(f"{r['acc'] * 100:.1f}" for r in rows)


# --------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    rank, world, local = setup_dist()
    is_main = rank == 0
    n_cpus = limit_cpus(args.max_cpus, local, int(os.environ.get("LOCAL_WORLD_SIZE", world)))
    args.workers = default_workers(n_cpus, args.workers)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    amp = args.amp if args.amp != "auto" else auto_amp(device)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)  # Mixup / CutMix sampling
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    # data: rank 0 downloads first
    if is_main:
        splits = build_splits(args.dataset, args.data_dir, args.holdout, seed=0, download=True,
                              fake=args.fake_data, randaug=not args.no_randaug, erase=args.erase)
    barrier(world)
    if not is_main:
        splits = build_splits(args.dataset, args.data_dir, args.holdout, seed=0, download=False,
                              fake=args.fake_data, randaug=not args.no_randaug, erase=args.erase)
    per_gpu_bs = args.batch_size // world
    sampler = DistributedSampler(splits["train"], shuffle=True, seed=args.seed) if world > 1 else None
    train_loader = DataLoader(splits["train"], batch_size=per_gpu_bs, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True,
                              persistent_workers=args.workers > 0)
    holdout_loader = DataLoader(splits["holdout"], batch_size=512, shuffle=False, num_workers=args.workers,
                                persistent_workers=args.workers > 0)

    num_classes = NUM_CLASSES[args.dataset]
    model = build_model(args.model, num_classes=num_classes, t_max=args.t_max, n_blocks=args.n_blocks,
                        patch_size=args.patch_size,
                        norm=args.norm, step_embed_in=args.step_embed_in,
                        layerscale_init=args.layerscale_init).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    ema = EMA(model, args.ema) if args.ema > 0 else None
    loss_cfg = StageALossConfig(beta=args.beta, gamma=args.gamma, delta=args.delta, gk_alpha=args.gk_alpha,
                                step_weights=args.step_weights, label_smoothing=args.label_smoothing)
    mixup = Mixup(num_classes, args.mixup, args.cutmix, args.mix_prob)

    start_epoch = 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"):
            ema.model.load_state_dict(ck["ema"])
        start_epoch = ck["epoch"] + 1

    train_model = torch.compile(model) if args.compile else model
    if world > 1:
        train_model = DDP(train_model, device_ids=[local] if device.type == "cuda" else None)

    iters_per_epoch = len(train_loader)
    total_iters = args.epochs * iters_per_epoch
    warmup_iters = int(args.warmup_epochs * iters_per_epoch)
    if is_main:
        meta = {"args": vars(args), "model_cfg": config_dict(model), "params": n_params, "world": world,
                "amp": amp, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "torch": torch.__version__}
        (out / "config.json").write_text(json.dumps(meta, indent=2))
        print(f"[train] params={n_params / 1e6:.2f}M world={world} amp={amp} iters/epoch={iters_per_epoch} "
              f"cpus/rank={n_cpus} workers/rank={args.workers}",
              flush=True)

    t_start = time.time()
    it = start_epoch * iters_per_epoch
    stop = False
    for epoch in range(start_epoch, args.epochs):
        train_model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        t_ep = time.time()
        running, n_seen = {}, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, target = mixup(x, y)
            lr = lr_at(it, total_iters, warmup_iters, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            with autocast_ctx(device, amp):
                logits = train_model(x)
            loss, parts = stage_a_loss(logits, target, loss_cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v.float()
            n_seen += 1
            it += 1
            if is_main and it % args.log_every == 0:
                print(f"[train] ep {epoch} it {it}/{total_iters} lr {lr:.2e} loss {parts['total'].item():.4f}", flush=True)
            if args.max_steps and it >= args.max_steps:
                stop = True
                break

        # time-limit decision is made on rank 0 and broadcast so all ranks agree
        over = torch.tensor(float(args.time_limit > 0 and (time.time() - t_start) / 60 > args.time_limit),
                            device=device)
        if world > 1:
            dist.broadcast(over, 0)
        stop = stop or bool(over.item())
        last = stop or epoch == args.epochs - 1

        if is_main:
            ep_time = time.time() - t_ep
            rec = {"epoch": epoch, "lr": lr, "epoch_time_s": ep_time, "imgs_per_s": n_seen * args.batch_size / ep_time,
                   **{f"train_{k}": float(v) / max(1, n_seen) for k, v in running.items()}}
            if (epoch + 1) % args.eval_every == 0 or last:
                eval_model = ema.model if ema else model
                res = collect_outputs(eval_model, holdout_loader, device, amp)
                rows = per_step_metrics(res["logits"], res["labels"])
                rec["holdout"] = rows
                print(f"[eval] ep {epoch} holdout acc/step: {summarize_steps(rows)} | "
                      f"ECE last {rows[-1]['ece']:.3f}", flush=True)
            print(f"[epoch] {epoch} time {ep_time:.1f}s {rec['imgs_per_s']:.0f} img/s "
                  f"loss {rec.get('train_total', float('nan')):.4f}", flush=True)
            with open(out / "history.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "ema": ema.model.state_dict() if ema else None, "epoch": epoch,
                     "model_name": args.model, "model_cfg": config_dict(model)}
            torch.save(state, out / "last.pt")
            if last:
                final = {"model": (ema.model if ema else model).state_dict(), "model_name": args.model,
                         "model_cfg": config_dict(model), "epoch": epoch, "args": vars(args)}
                torch.save(final, out / "model.pt")
        barrier(world)
        if stop:
            break

    if is_main:
        print(f"[train] done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
