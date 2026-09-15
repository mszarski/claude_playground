"""Time and memory for the reference workload: 5,000 rays x 4,000 RK4 steps.

Reports the forward trace, and a forward+backward pass with and without
gradient checkpointing, on CPU and -- if one is present -- on GPU.

Run: ``python scripts/benchmark.py [--rays 5000] [--steps 4000]``
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydropt import (  # noqa: E402
    ConstantLoss, FlatHeight, MunkProfile, Scene, make_time_grid, octave_bands,
    spherical_fan, vertical_line_array,
)


def peak_rss_mib() -> float:
    """Peak resident set size so far, in MiB (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build(device: str, dtype: torch.dtype, steps: int, checkpoint_every: int) -> Scene:
    scene = Scene(
        field=MunkProfile(),
        bottom=FlatHeight(5000.0),
        surface=FlatHeight(0.0),
        source=(0.0, 0.0, 1000.0),
        receivers=vertical_line_array(50_000.0, 0.0, 600.0, 2000.0, 4),
        surface_loss=ConstantLoss(0.5),
        bottom_loss=ConstantLoss(6.0),
        freqs_khz=octave_bands(0.5, 4),
        step_size=20.0,
        n_steps=steps,
        checkpoint_every=checkpoint_every,
    )
    return scene.to(device=device, dtype=dtype)


def path_bytes(result) -> float:
    tensors = (result.pos, result.tau, result.arclen, result.refl_db, result.alive)
    return sum(t.numel() * t.element_size() for t in tensors) / 2**20


def run(device: str, rays: int, steps: int, dtype: torch.dtype) -> None:
    """Orchestrate the measurements.

    Every case runs in its *own* process and this parent holds nothing but the
    subprocess handles.  Sharing a process gives a misleading answer and can
    kill the run outright: the allocator does not return one case's peak to the
    OS, so the next starts from a high baseline and, at this size, is
    OOM-killed before it prints anything.
    """
    print(f"\n{device.upper()}  ({torch.get_num_threads()} threads, {dtype})"
          if device == "cpu" else f"\n{device.upper()}  ({dtype})")
    print(f"  {rays} rays x {steps} steps")

    for case, label in (("forward", "forward trace (no_grad)"),
                        ("0", "fwd+bwd, no checkpointing"),
                        ("100", "fwd+bwd, checkpoint_every=100")):
        proc = subprocess.run(
            [sys.executable, "-u", __file__, "--case", case,
             "--rays", str(rays), "--steps", str(steps),
             "--threads", str(torch.get_num_threads()),
             *(["--float64"] if dtype is torch.float64 else [])],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            print(proc.stdout.rstrip())
        else:
            tail = proc.stdout.strip() or (proc.stderr.strip().splitlines() or ["no output"])[-1]
            if proc.returncode < 0:
                tail = "killed by the OS, almost certainly out of memory"
            print(f"  {label:32s} failed: {tail}")


def run_case(device: str, rays: int, steps: int, dtype: torch.dtype,
             case: str) -> None:
    """One measurement, run as its own process.

    ``case`` is ``"forward"`` for a no_grad trace, or the ``checkpoint_every``
    value to use for a forward+backward pass.
    """
    directions = spherical_fan(rays // 25, 25, (-16.0, 16.0), (-10.0, 10.0)).to(
        device=device, dtype=dtype)
    grid = make_time_grid(32.0, 36.0, 400).to(device=device, dtype=dtype)
    chunk = 0 if case == "forward" else int(case)
    scene = build(device, dtype, steps, chunk)

    if case == "forward":
        t0 = time.perf_counter()
        with torch.no_grad():
            result = scene.trace(directions)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"  forward trace (no_grad)          {time.perf_counter() - t0:7.2f} s   "
              f"stored path {path_bytes(result):8.1f} MiB")
        return
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    base = peak_rss_mib()
    t0 = time.perf_counter()
    etc = scene.render(directions, grid, sigma_d=150.0, sigma_t=5e-3, ray_chunk=500)
    if not etc.requires_grad:
        # Nothing reached a receiver, so there is no graph to walk.  At small
        # --steps the rays simply run out of path before the 50 km array.
        raise SystemExit(
            f"  no energy reached the array at {steps} steps "
            f"({scene.max_path_length / 1e3:.0f} km of path, array at 50 km); "
            "nothing to differentiate"
        )
    etc.sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    label = f"checkpoint_every={chunk}" if chunk else "no checkpointing"
    line = f"  fwd+bwd, {label:22s} {dt:7.2f} s"
    if device == "cuda":
        line += f"   peak CUDA {torch.cuda.max_memory_allocated() / 2**20:8.1f} MiB"
    else:
        line += f"   peak RSS {peak_rss_mib():9.1f} MiB (baseline {base:.1f})"
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--case", type=str, default=None,
                        help="internal: run one case in this process -- "
                             "'forward', or a checkpoint_every value")
    parser.add_argument("--float64", action="store_true",
                        help="benchmark in double precision (the default is float32, "
                             "which is what production runs should use)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.float64 else torch.float32
    torch.set_default_dtype(dtype)

    if args.case is not None:
        run_case("cpu", args.rays, args.steps, dtype, args.case)
        return 0

    print(f"hydropt benchmark -- torch {torch.__version__}")
    run("cpu", args.rays, args.steps, dtype)
    if torch.cuda.is_available():
        run("cuda", args.rays, args.steps, dtype)
    else:
        print("\nCUDA  not available on this machine; GPU figures not measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
