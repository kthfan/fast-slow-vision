"""Identify the exact code a run used, with or without git.

``code_hash`` is a short SHA-256 over the source files (fsv/, scripts/,
pyproject.toml), so it is the same for a git checkout and for a copy made
from the GitHub zip of the same commit.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from . import __version__

ROOT = Path(__file__).resolve().parents[1]


def code_hash(root: Path = ROOT) -> str:
    files = sorted([*root.glob("fsv/**/*.py"), *root.glob("scripts/*.py"), *root.glob("scripts/*.sh"),
                    root / "pyproject.toml"])
    h = hashlib.sha256()
    for f in files:
        if f.is_file():
            h.update(f.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
            h.update(b"\0")
    return h.hexdigest()[:12]


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def code_info() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if commit else None
    h = code_hash()
    return {
        "fsv_version": __version__,
        "code_hash": h,
        "code_version": f"{__version__}+{h}",
        "git_commit": commit,  # None when git or the .git directory is unavailable
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") if commit else None,
        "git_dirty": bool(status) if status is not None else None,
    }
