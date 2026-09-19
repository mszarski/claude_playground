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


_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def setup(seed: int = 0, *, double: bool = True, threads: int | None = None) -> None:
    """Deterministic, CPU-friendly defaults shared by every example.

    ``HYDROPT_EXAMPLE_DTYPE=float32`` overrides the precision, which is how an
    example gets run in single precision without editing it -- the same switch
    ``HYDROPT_TEST_DTYPE`` gives the test suite.  It matters for more than
    speed: fp64 throughput is half of fp32 on a datacentre GPU and a
    sixty-fourth of it on a consumer card, so whether the imaging path holds up
    in float32 decides whether a GPU is worth anything here.
    """
    name = os.environ.get("HYDROPT_EXAMPLE_DTYPE")
    if name is not None and name not in _DTYPES:
        raise ValueError(f"HYDROPT_EXAMPLE_DTYPE must be one of "
                         f"{sorted(_DTYPES)}, got {name!r}")
    dtype = (_DTYPES[name] if name
             else (torch.float64 if double else torch.float32))
    torch.manual_seed(seed)
    torch.set_default_dtype(dtype)
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
