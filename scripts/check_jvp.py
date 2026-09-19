"""Does forward-mode AD work through the tracer?

The brief proposed obtaining ray-tube spreading from "forward-mode derivatives
of the ray position with respect to launch angles".  hydropt instead takes
neighbour differences across a structured fan, which is what production ray
codes do.  This script is the evidence for that choice rather than an assertion
about it: it tries the forward-mode route and reports what happens.

Run: ``python scripts/check_jvp.py``
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydropt import ConstantLoss, FlatHeight, IsoProfile, MunkProfile, Scene  # noqa: E402
from hydropt.launch import directions_from_angles  # noqa: E402
from hydropt.tracer import trace  # noqa: E402


def _scene(bounded: bool) -> Scene:
    return Scene(
        field=MunkProfile(learnable=False) if not bounded else IsoProfile(1500.0),
        bottom=FlatHeight(200.0 if bounded else 1e6),
        surface=FlatHeight(0.0 if bounded else -1e6),
        source=(0.0, 0.0, 1000.0 if not bounded else 50.0),
        surface_loss=ConstantLoss(0.5, learnable=False),
        bottom_loss=ConstantLoss(3.0, learnable=False),
        step_size=20.0, n_steps=120,
    )


def _positions(elev: torch.Tensor, azim: torch.Tensor, scene: Scene) -> torch.Tensor:
    dirs = directions_from_angles(elev, azim)
    return trace(scene, dirs).pos


def main() -> int:
    torch.set_default_dtype(torch.float64)
    elev = torch.full((16,), 0.05)
    azim = torch.linspace(-0.1, 0.1, 16)

    for label, bounded in (("refracting, no boundary contact", False),
                           ("shallow, with reflections", True)):
        scene = _scene(bounded)
        print(f"\n{label}")
        try:
            t0 = time.perf_counter()
            primal, tangent = torch.func.jvp(
                lambda e: _positions(e, azim, scene), (elev,), (torch.ones_like(elev),))
            dt = time.perf_counter() - t0
            finite = bool(torch.isfinite(tangent).all())
            print(f"  torch.func.jvp: SUCCEEDED in {dt:.2f} s, "
                  f"tangent finite={finite}, |dr/de| max={tangent.abs().max():.3e}")

            # Cross-check against a central difference of two extra traces.
            d = 1e-6
            hi = _positions(elev + d, azim, scene)
            lo = _positions(elev - d, azim, scene)
            fd = (hi - lo) / (2 * d)
            rel = ((tangent - fd).abs().max() / fd.abs().max()).item()
            print(f"  agreement with a central difference: {rel:.3e} relative")
        except Exception as exc:  # noqa: BLE001 - the point is to report anything
            print(f"  torch.func.jvp: FAILED -- {type(exc).__name__}: {exc}")
            tb = traceback.format_exc().strip().splitlines()
            print(f"      {tb[-2].strip() if len(tb) > 1 else ''}")

    print("\nWhy hydropt uses neighbour differences regardless of the above:")
    print("  * cost: jvp gives one tangent per pass, so the two launch-angle")
    print("    derivatives are two extra traces on top of the primal -- 3x.")
    print("    Neighbour differences reuse the fan that was traced anyway.")
    print("  * inversion: spreading must stay reverse-mode differentiable in the")
    print("    scene parameters, so this would be reverse-over-forward.")
    print("  * accuracy: neighbour differences are O(dtheta^2) and converge at")
    print("    exactly that rate (see tests/test_spreading.py), which is well")
    print("    below the error already in a discretised ray fan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
