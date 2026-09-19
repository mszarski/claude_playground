#!/usr/bin/env python3
"""Gaussian beam summation: normalisation, beta-independence, and where it stops.

Prints the evidence behind the README's beam-summation section. Too slow for the
test suite because the Pekeris comparisons trace 3,700 rays through
``gaussian_beams`` twice.

Run: ``python3 scripts/validate_beamsum.py``
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydropt import (  # noqa: E402
    ConstantLoss, FlatHeight, IsoProfile, RayleighBottomLoss, Scene,
    beam_sum_kwargs, gaussian_beams, make_time_grid, splat_etc, structured_fan,
)
from hydropt.pekeris import PekerisWaveguide  # noqa: E402

C = 1500.0
FREQ_HZ = 200.0
OMEGA = 2 * math.pi * FREQ_HZ
NO_ABSORPTION = lambda f: torch.zeros_like(f)  # noqa: E731


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def fan(n_e: int, n_a: int, elev: float, azim: float):
    dirs, e, a = structured_fan(n_e, n_a, elev_range_deg=(-elev, elev),
                               azim_range_deg=(-azim, azim))
    d_e = math.radians(2 * elev) / (n_e - 1)
    d_a = math.radians(2 * azim) / (n_a - 1)
    w = (torch.cos(e).reshape(-1, 1) * d_e * d_a).expand(n_e, n_a).reshape(-1)
    ev = e.reshape(-1, 1).expand(n_e, n_a).reshape(-1).contiguous()
    az = a.reshape(1, -1).expand(n_e, n_a).reshape(-1).contiguous()
    return dirs, w.contiguous(), ev, az


def render(beams, weights, receivers, grid, chunk=400, **override):
    kw = beam_sum_kwargs(beams)
    kw.update(override)
    etc = splat_etc(beams.result, receivers, grid, torch.tensor([FREQ_HZ / 1e3]),
                    sigma_t=4e-3, ray_weights=weights, absorption=NO_ABSORPTION,
                    ray_chunk=chunk, **kw)
    return etc[:, 0].sum(-1) * float(grid[1] - grid[0])


def free_space() -> None:
    banner("free space: absolute normalisation, with nothing fitted")
    z_src = 500.0
    scene = Scene(field=IsoProfile(C, learnable=False), surface=FlatHeight(-1e5),
                  bottom=FlatHeight(1e5), source=(0.0, 0.0, z_src),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  freqs_khz=torch.tensor([FREQ_HZ / 1e3]), step_size=25.0,
                  n_steps=200, max_bounces=0)
    dirs, w, ev, az = fan(61, 61, 24.0, 24.0)
    ranges = [500.0, 1000.0, 2000.0, 4000.0]
    grid = make_time_grid(0.2, 3.2, 1600)
    rx = torch.stack([torch.tensor(ranges), torch.zeros(len(ranges)),
                      torch.full((len(ranges),), z_src)], dim=-1)
    print("  E s^2 omega beta / (pi c) should be 1 at every range and every beta")
    print(f"  {'beta':>7s} {'W(1km)':>8s} | "
          + " ".join(f"{int(r):>8d}" for r in ranges) + "   exponent")
    for beta in (150.0, 400.0, 1000.0):
        beams = gaussian_beams(scene, ev, az, beam_width=beta,
                               freq_khz=FREQ_HZ / 1e3)
        e = render(beams, w, rx, grid, chunk=4000)
        norm = [float(e[i]) * ranges[i] ** 2 * OMEGA * beta / (math.pi * C)
                for i in range(len(ranges))]
        expo = math.log(float(e[0] / e[-1])) / math.log(ranges[-1] / ranges[0])
        width = math.sqrt(C * (1e6 + beta * beta) / (OMEGA * beta))
        print(f"  {beta:7.0f} {width:8.1f} | "
              + " ".join(f"{v:8.4f}" for v in norm) + f"   {expo:8.4f}")
    print("\n  Independent of beta, which parameterises the decomposition and not")
    print("  the physics -- the strongest sign the sum reconstructs the field.")


def waveguide(depth: float, z_src: float, depths, label: str) -> None:
    banner(f"Pekeris {label}: beam sum against the modes, depth-averaged")
    wg = PekerisWaveguide(depth, C, 1800.0, 1000.0, 1800.0)
    fresnel = math.sqrt(2 * C * 2000.0 / OMEGA)
    print(f"  {wg.n_modes(FREQ_HZ)} trapped modes; Fresnel scale at 2 km is "
          f"{fresnel:.0f} m against {depth:.0f} m of water "
          f"({'fits' if fresnel < depth / 3 else 'does NOT fit'})")
    scene = Scene(field=IsoProfile(C, learnable=False), surface=FlatHeight(0.0),
                  bottom=FlatHeight(depth), source=(0.0, 0.0, z_src),
                  surface_loss=ConstantLoss(0.0, learnable=False,
                                            pressure_release=True),
                  bottom_loss=RayleighBottomLoss(1800.0, 1800.0, 0.0, rho1=1000.0,
                                                 c1=C, learnable=False),
                  freqs_khz=torch.tensor([FREQ_HZ / 1e3]), step_size=10.0,
                  n_steps=450, max_bounces=300)
    dirs, w, ev, az = fan(220, 17, 40.0, 3.0)
    beta = 2000.0
    beams = gaussian_beams(scene, ev, az, beam_width=beta, freq_khz=FREQ_HZ / 1e3)
    grid = make_time_grid(0.60, 2.90, 2400)
    ranges = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
    rx = torch.tensor([[r, 0.0, z] for r in ranges for z in depths])
    e = render(beams, w, rx, grid)
    intensity = (e * OMEGA * beta / (math.pi * C)).reshape(len(ranges),
                                                           len(depths)).numpy()
    print(f"  {'range':>7s} {'beam sum':>10s} {'modes':>9s} {'diff':>8s}")
    for i, r in enumerate(ranges):
        tl_mode = np.array([wg.transmission_loss(FREQ_HZ, np.array([r]), z_src, z,
                                                 coherent=False)[0] for z in depths])
        a = -10 * math.log10(intensity[i].mean())
        b = -10 * math.log10((10 ** (-tl_mode / 10)).mean())
        print(f"  {r:7.0f} {a:10.3f} {b:9.3f} {a - b:+8.3f}")


def main() -> int:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    free_space()
    waveguide(1000.0, 500.0, [float(z) for z in np.arange(100.0, 951.0, 50.0)],
              "deep (beams fit)")
    waveguide(100.0, 50.0, [float(z) for z in np.arange(10.0, 95.1, 5.0)],
              "shallow (beams do not fit)")
    banner("conclusion")
    print("  The beam sum is absolutely normalised and beta-independent, and it")
    print("  agrees with the modes to a tenth of a dB where the beam fits inside")
    print("  the waveguide.  Where it does not -- shallow water at low frequency,")
    print("  where the Fresnel scale exceeds the channel depth -- the transverse")
    print("  profile spills through both boundaries and the sum is a couple of dB")
    print("  high with a range trend.  Folding the beams at the boundaries (image")
    print("  beams) is what that regime needs, and is not implemented.")
    print(f"\ntotal {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
