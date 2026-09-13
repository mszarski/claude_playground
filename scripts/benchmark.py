"""Time and memory for the reference workload: 5,000 rays x 4,000 RK4 steps.

Reports the forward trace, and a forward+backward pass with and without
gradient checkpointing, on CPU and -- if one is present -- on GPU.

Run: ``python scripts/benchmark.py [--rays 5000] [--steps 4000]``
"""

from __future__ import annotations

import argparse
import gc
import resource
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
    print(f"\n{device.upper()}  ({torch.get_num_threads()} threads, {dtype})"
          if device == "cpu" else f"\n{device.upper()}  ({dtype})")
    print(f"  {rays} rays x {steps} steps")
    directions = spherical_fan(rays // 25, 25, (-16.0, 16.0), (-10.0, 10.0)).to(
        device=device, dtype=dtype)
    grid = make_time_grid(32.0, 36.0, 400).to(device=device, dtype=dtype)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    # -- forward only ------------------------------------------------------- #
    scene = build(device, dtype, steps, 0)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache(), torch.cuda.reset_peak_memory_stats()
    base_rss = peak_rss_mib()
    t0 = time.perf_counter()
    with torch.no_grad():
        result = scene.trace(directions)
    sync()
    dt = time.perf_counter() - t0
    print(f"  forward trace (no_grad)          {dt:7.2f} s   "
          f"stored path {path_bytes(result):8.1f} MiB")
    if device == "cuda":
        print(f"      peak CUDA memory             {torch.cuda.max_memory_allocated() / 2**20:8.1f} MiB")
    del result
    gc.collect()

    # -- forward + backward, with and without checkpointing ----------------- #
    for label, chunk in (("no checkpointing", 0), ("checkpoint_every=100", 100)):
        scene = build(device, dtype, steps, chunk)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache(), torch.cuda.reset_peak_memory_stats()
        rss_before = peak_rss_mib()
        t0 = time.perf_counter()
        try:
            etc = scene.render(directions, grid, sigma_d=150.0, sigma_t=5e-3,
                               ray_chunk=500)
            etc.sum().backward()
            sync()
            dt = time.perf_counter() - t0
            extra = peak_rss_mib() - rss_before
            line = f"  fwd+bwd, {label:22s} {dt:7.2f} s"
            if device == "cuda":
                line += f"   peak CUDA {torch.cuda.max_memory_allocated() / 2**20:8.1f} MiB"
            else:
                line += f"   peak RSS increase {max(extra, 0.0):8.1f} MiB"
            print(line)
        except (RuntimeError, MemoryError) as exc:
            print(f"  fwd+bwd, {label:22s}   out of memory ({type(exc).__name__})")
        del scene
        gc.collect()
    print(f"  process peak RSS overall         {peak_rss_mib():8.1f} MiB "
          f"(baseline at entry {base_rss:.1f} MiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--float64", action="store_true",
                        help="benchmark in double precision (the default is float32, "
                             "which is what production runs should use)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.float64 else torch.float32
    torch.set_default_dtype(dtype)

    print(f"hydropt benchmark -- torch {torch.__version__}")
    run("cpu", args.rays, args.steps, dtype)
    if torch.cuda.is_available():
        run("cuda", args.rays, args.steps, dtype)
    else:
        print("\nCUDA  not available on this machine; GPU figures not measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
