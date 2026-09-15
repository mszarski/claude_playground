"""Small shared helpers for the examples: seeding, figure output, timing."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Let the examples run straight from a checkout, without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # examples write files; they never need a display
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

FIGURE_DIR = Path(__file__).resolve().parent / "figures"


def setup(seed: int = 0, *, double: bool = True, threads: int | None = None) -> None:
    """Deterministic, CPU-friendly defaults shared by every example."""
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64 if double else torch.float32)
    torch.set_num_threads(threads or min(4, os.cpu_count() or 1))


def save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return path


class timed:
    """``with timed('label'):`` -- prints wall time on exit."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        print(f"  {self.label}: {time.perf_counter() - self.t0:.2f} s")
        return False


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Print a pass/fail line for an acceptance criterion."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok
