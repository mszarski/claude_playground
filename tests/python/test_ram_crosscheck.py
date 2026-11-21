"""Milestone 12: the eigenray tier (Bellhop) against a parabolic equation (PyRAM) in the flank array's band.

PyRAM 1.3.0 (github.com/marcuskd/pyram, BSD-3; a Python/Numba port of M. D. Collins' RAM v1.5, its docstring
says) is a TEST ORACLE ONLY: it is imported here and nowhere in the library
(test_the_library_never_imports_pyram, which runs without it). The cross-check skips without it
(pip install pyram==1.3.0) and without Bellhop (GPUSONAR_BELLHOP); oracle_gate.py counts both skips, and
GPUSONAR_REQUIRE_ORACLES=1 turns them into failures.

What is compared. Our tier's coherent transmission loss, |sum_k a_k exp(-j 2 pi f tau_k)| over the arrivals that
`BellhopProvider.eigenrays` returns (its default arrival selection), against PyRAM's TL at the same ranges and
depths, in two 2-D waveguides a flank array sees: a Pekeris waveguide (100 m, 1500 m/s over a 1600 m/s,
1.8 g/cm^3, 0.5 dB/wavelength fluid half-space, as test_kraken_crosscheck.py) and an upslope wedge (the same
water and bottom, 100 m at the source to 30 m at 5 km, 0.8 degrees), at 6, 8 and 10 kHz (flank_32.yaml: LFM 6 to
10 kHz), source 30 m, in 500 m windows from 0.5 to 5 km.

Why the comparison is of 500 m range averages of intensity, and what tolerance (ours, from the physics below and
the measured spread; measured here on 2026-11-21 with PyRAM 1.3.0, numba 0.68.1, and Bellhop of the Acoustics
Toolbox at HEAD 8b4682b):

* Both solve the same point-source Helmholtz problem in (range, depth), by different approximations. Bellhop is a
  high-frequency beam method (kH = 2 pi f H / c = 2500 to 4200 here: far into its regime), with the plane-wave
  reflection coefficient of the half-space at each bottom bounce (Bellhop/ReflectMod.f90, Reflect2D, opened). RAM is a one-way (no backscatter) split-step Pade
  solution of the wave equation, so it carries diffraction, the field in the sediment and the beam displacement
  at near-critical reflection that a plane-wave coefficient leaves out (critical grazing angle acos(1500/1600) =
  20.4 degrees). Agreement is expected where the energy travels at grazing angles below critical, which beyond a
  few water depths is all of it (steeper paths have left through the bottom), and where the wedge is gentle
  enough for the one-way, stair-stepped bottom of RAM (PyRAM.updat moves the bottom index once per range step)
  to lose nothing: 0.8 degrees.
* Point by point they do not agree and are not expected to: at 8 kHz the field is the coherent sum of tens of
  arrivals and its nulls sit where their phases cancel, so a phase error of a fraction of a cycle moves a null
  by metres. Measured at 8 kHz from 0.5 to 5 km: half the point differences exceed 4.6 to 4.8 dB, a tenth
  exceed 11.9 to 13.4 dB (by case and depth).
  The comparison is therefore of the intensity averaged over 500 m windows of range at a fixed depth, of the
  coherent ray sum (the quantity PyRAM computes; the incoherent ray sum gives window means within 0.25 dB of
  the same PE averages at 8 kHz, measured).
* Convergence of the oracle (measured, 8 kHz, from a 1.5 m range step and a 0.02 m depth step): halving PyRAM's
  range and depth steps, or its range step alone four times, or 6 instead of 8 Pade terms, moves these 500 m
  averages by at most 0.14 dB. Bellhop's receivers sit on the provider's grid nodes, so its first-order delay
  correction does not enter. Receivers stay 8 m (over 40 wavelengths at 8 kHz) or more from the boundaries:
  the Bellhop documentation (doc/bellhop.htm, opened) warns of artifacts for receivers very close to them.
* Measured, all three frequencies and both cases, windows 0.5 to 5 km: the largest window difference is 1.42 dB
  (Pekeris, 6 kHz, 82 m), the mean over the windows of one depth at most 0.44 dB in magnitude (Pekeris, 8 kHz,
  50 m); the provider's default arrival selection drops at most 0.14 % of a point's arrival energy. Bounds:
  every window within 3 dB, each depth's mean within 1 dB: about twice the largest difference seen, which is
  ours (there is no published bound for this pair of models at these frequencies that we opened).
* What this does NOT test: the bottom density. Beyond 0.5 km the energy travels below the critical angle,
  where a 0.5 dB/wavelength bottom loses little per bounce whatever its density; density sets the reflection of
  steep paths (at normal incidence (rho c_b - c_w) / (rho c_b + c_w) = 0.32 at 1.8 g/cm^3, 0.03 at 1.0), and
  those have left by then. Measured (8 kHz, Pekeris): giving the provider 1.0 or 2.2 g/cm^3 instead of the PE's
  1.8 moves the window differences by up to 1.7 dB and the depth means by at most 0.25 dB, inside the bounds.
  A wrong density in our Bellhop input would pass here. Our own Rayleigh coefficient's density dependence is
  tested in test_propagation.py (test_rayleigh_reflection_limits); the density Bellhop is given is not checked
  against an oracle.

PyRAM's inputs are in PYRAM_INPUTS and PyRAM is pinned to 1.3.0 (another version skips): the tolerances are for it.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import re

import numpy as np
import pytest

from gpusonar.sim import find_bellhop
from gpusonar.sim.bellhop import BellhopProvider, OceanEnvironment

PYRAM_VERSION = "1.3.0"
PYRAM_MISSING = f"needs the PE oracle PyRAM {PYRAM_VERSION} (module pyram: pip install pyram=={PYRAM_VERSION})"


def pyram_wrong_version(found: str) -> str:
    return f"PyRAM {found} is installed; the cross-check is for {PYRAM_VERSION} (module pyram: pip install pyram=={PYRAM_VERSION})"


try:
    from pyram.PyRAM import PyRAM
    _found = importlib.metadata.version("pyram")
    _pyram_reason = None if _found == PYRAM_VERSION else pyram_wrong_version(_found)
except ImportError:
    PyRAM = None
    _pyram_reason = PYRAM_MISSING

needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
needs_pyram = pytest.mark.skipif(_pyram_reason is not None, reason=_pyram_reason or "")
LIBRARY = pathlib.Path(__file__).resolve().parents[2] / "python" / "gpusonar"

WATER, BOTTOM = 1500.0, dict(speed=1600.0, density=1.8, attenuation=0.5)
DEPTH, SOURCE_DEPTH, MAX_RANGE = 100.0, 30.0, 5000.0
CASES = {
    "pekeris": dict(bathymetry=None, receivers=(18.0, 50.0, 82.0)),
    "upslope_wedge": dict(bathymetry=[[0.0, 100.0], [5000.0, 30.0]], receivers=(10.0, 22.0)),
}
# Inputs to PyRAM besides the environment (its keyword names). dr = np x wavelength and dz = wavelength / 10 are
# PyRAM's own defaults; dz is rounded down so that the 2 m output grid (ndz steps) holds every receiver depth.
PYRAM_INPUTS = dict(np=8, ns=1, lyrw=20, c0=WATER, sediment_thickness=50.0, output_depth_step=2.0)
WINDOW, FIRST = 500.0, 500.0
MAX_WINDOW_DB, MAX_MEAN_DB = 3.0, 1.0
FREQUENCIES = (6000.0, 8000.0, 10000.0)


def pe_loss(frequency, bathymetry):
    """PyRAM TL [depth, range] dB re 1 m, its ranges and its output depths."""
    lam = WATER / frequency
    ndz = int(np.ceil(PYRAM_INPUTS["output_depth_step"] / (0.1 * lam)))
    dz = PYRAM_INPUTS["output_depth_step"] / ndz
    rbzb = np.array(bathymetry if bathymetry is not None else [[0.0, DEPTH]], dtype=float)
    one = lambda v: np.full((2, 1), float(v))                                   # noqa: E731
    run = PyRAM(frequency, SOURCE_DEPTH, 50.0, np.array([0.0, DEPTH]), np.array([0.0]), one(WATER),
                np.array([0.0, PYRAM_INPUTS["sediment_thickness"]]), np.array([0.0]),
                one(BOTTOM["speed"]), one(BOTTOM["density"]), one(BOTTOM["attenuation"]), rbzb,
                np=PYRAM_INPUTS["np"], ns=PYRAM_INPUTS["ns"], lyrw=PYRAM_INPUTS["lyrw"], c0=PYRAM_INPUTS["c0"],
                dr=PYRAM_INPUTS["np"] * lam, dz=dz, ndz=ndz, zmplt=DEPTH, rmax=MAX_RANGE).run()
    return run["TL Grid"], run["Ranges"], run["Depths"]


def window_average(ranges, loss):
    """dB of the intensity averaged over WINDOW-long range windows from FIRST to MAX_RANGE (each window holds points)."""
    edges = np.arange(FIRST, MAX_RANGE + 1e-6, WINDOW)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        keep = (ranges >= a) & (ranges < b)
        assert keep.sum() > 10, (a, b)
        out.append(-10.0 * np.log10(np.mean(10.0 ** (-loss[keep] / 10.0))))
    return np.array(out)


def test_the_library_never_imports_pyram():
    """PyRAM is approved as a test oracle only (docs/OPEN_SOURCE.md): no module of the package imports it."""
    found = [str(p.relative_to(LIBRARY)) for p in LIBRARY.rglob("*.py")
             if re.search(r"^\s*(import|from)\s+pyram\b", p.read_text(encoding="utf-8"), re.M)]
    assert not found, found


@needs_bellhop
@needs_pyram
@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("case", sorted(CASES))
def test_ray_tier_against_parabolic_equation(case, frequency, capsys):
    spec = CASES[case]
    pe, pe_ranges, pe_depths = pe_loss(frequency, spec["bathymetry"])
    # Our points: every fifth PE range, which are the provider's own range nodes; depths on its 4 m grid (2, 6, ...).
    step = 5 * PYRAM_INPUTS["np"] * WATER / frequency
    env = OceanEnvironment(depth=DEPTH, sound_speed=WATER, bottom_speed=BOTTOM["speed"], bottom_density=BOTTOM["density"],
                           bottom_attenuation=BOTTOM["attenuation"], bathymetry=spec["bathymetry"])
    provider = BellhopProvider(env, frequency, max_range=MAX_RANGE + 2.0 * step, range_step=step, min_range=step, depth_step=4.0)
    pick = np.arange(4, len(pe_ranges), 5)
    ranges = pe_ranges[pick]
    assert np.allclose(provider.ranges[np.rint(ranges / step).astype(int) - 1], ranges)
    lines = []
    for depth in spec["receivers"]:
        assert np.isclose(np.mod(depth - 2.0, 4.0), 0.0)                     # a provider depth node
        rays = provider.eigenrays(np.array([0.0, 0.0, SOURCE_DEPTH]),
                                  np.column_stack([ranges, np.zeros_like(ranges), np.full_like(ranges, depth)]))
        p = np.sum(np.where(rays.valid, rays.amplitude * np.exp(-2j * np.pi * frequency * rays.delay), 0.0), axis=1)
        ray_loss = -20.0 * np.log10(np.maximum(np.abs(p), 1e-30))
        pe_line = pe[int(np.argmin(np.abs(pe_depths - depth)))][pick]
        assert abs(pe_depths[int(np.argmin(np.abs(pe_depths - depth)))] - depth) < 1e-6
        diff = window_average(ranges, ray_loss) - window_average(ranges, pe_line)
        lines.append(f"{depth:.0f} m: windows max |diff| {np.abs(diff).max():.2f} dB, mean {diff.mean():+.2f} dB, "
                     f"PE {window_average(ranges, pe_line)[-1]:.1f} dB at {MAX_RANGE / 1000:.0f} km")
        assert np.abs(diff).max() < MAX_WINDOW_DB, (depth, diff)
        assert abs(diff.mean()) < MAX_MEAN_DB, (depth, diff)
    with capsys.disabled():
        print(f"\n[pyram {PYRAM_VERSION}] {case} {frequency / 1000:.0f} kHz, rays minus PE, {WINDOW:.0f} m windows: " + "; ".join(lines)
              + f" (provider dropped {100 * provider.dropped_energy:.2f} % of a point's arrival energy)")
