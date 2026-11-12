"""Hard cap on the number of CPU cores a run may use.

The cap comes from (in order): an explicit argument, the ``FSV_MAX_CPUS``
environment variable, or ``DEFAULT_MAX_CPUS``. It is never larger than the
cores already available to the process.

``limit_cpus`` enforces it three ways:
  1. CPU affinity (Linux ``sched_setaffinity``): the process and every child it
     starts later (DataLoader workers, subprocesses) can only run on those cores.
  2. Thread-pool sizes: ``torch.set_num_threads`` and the OMP/MKL/OpenBLAS env
     variables, so libraries do not oversubscribe the allowed cores.
  3. ``num_workers`` for DataLoaders is derived from the cap (see ``default_workers``).
"""

from __future__ import annotations

import os

DEFAULT_MAX_CPUS = 8
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def available_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def resolve_max_cpus(requested: int | None = None) -> int:
    n = requested or int(os.environ.get("FSV_MAX_CPUS", "0") or 0) or DEFAULT_MAX_CPUS
    return max(1, min(int(n), len(available_cpus())))


def limit_cpus(requested: int | None = None, local_rank: int = 0, local_world: int = 1) -> int:
    """Restrict this process to at most ``requested`` cores; returns the number granted.

    With several processes on one machine (torchrun), the cores are split
    between local ranks so that the total stays within the cap.
    """
    n_total = resolve_max_cpus(requested)
    cpus = available_cpus()[:n_total]
    per_rank = max(1, n_total // max(1, local_world))
    mine = cpus[local_rank * per_rank:(local_rank + 1) * per_rank] or cpus[:per_rank]
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, mine)
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(len(mine))
    os.environ["FSV_MAX_CPUS"] = str(n_total)  # inherited by subprocesses
    try:
        import torch

        torch.set_num_threads(len(mine))
    except Exception:
        pass
    return len(mine)


def default_workers(n_cpus: int, requested: int | None = None) -> int:
    """DataLoader workers: leave one core for the main process, never exceed the cap."""
    auto = max(1, n_cpus - 1) if n_cpus > 1 else 0
    return auto if requested is None else max(0, min(requested, n_cpus))


def thread_env(n: int) -> dict[str, str]:
    """Environment overrides for launching a subprocess under the same cap."""
    return {**{v: str(n) for v in THREAD_ENV_VARS}, "FSV_MAX_CPUS": str(n)}
