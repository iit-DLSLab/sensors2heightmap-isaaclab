"""Where a dataset or a model comes from, saved in its metadata to reproduce it: the versions of the code it ran on and
the md5 of the files it used."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def code_version(path: str | Path) -> str | None:
    """git describe of the checkout holding path (-dirty with uncommitted changes), or the commit that a copy of a
    checkout records in its SNAPSHOT_OF or COMMIT file."""
    path = Path(path).resolve()
    for directory in (path, *path.parents):
        if (directory / ".git").exists():
            result = subprocess.run(["git", "-C", str(directory), "describe", "--always", "--dirty", "--abbrev=12"],
                                    capture_output=True, text=True)
            return result.stdout.strip() or None
        for name in ("SNAPSHOT_OF", "COMMIT"):
            if (directory / name).is_file():
                return (directory / name).read_text().split()[0] + " (copy)"
    return None


def _package_version(package: str) -> str | None:
    spec = importlib.util.find_spec(package)
    return code_version(spec.origin) if spec is not None and spec.origin else None


def versions() -> dict:
    """sensors2heightmap, basic-locomotion and Isaac Lab as they are imported, and torch."""
    return {
        "sensors2heightmap": code_version(REPO),
        "basic_locomotion": _package_version("basic_locomotion_isaaclab"),
        "isaaclab": _package_version("isaaclab"),
        "torch": str(torch.__version__),  # a plain str: the data loader reads the chunks with weights_only=True
    }


def md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
