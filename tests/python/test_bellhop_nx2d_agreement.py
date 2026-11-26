"""sonar-6c8: where the 1 to 2.5 dB between our Bellhop runs and Bellhop3D's N x 2-D mode comes from.

test_seamount_3d.py measured our eigenray tier 1.0 to 2.5 dB louder than Bellhop3D's N x 2-D mode on the seamount
bearings (incoherent TL averaged over 3.5 to 6.5 km) and up to 1.0 dB louder over a level 200 m bed at 7 km. This file
pins what that difference is made of (measured here 2026-11-26, Acoustics Toolbox HEAD 8b4682b, PyRAM 1.3.0; the
whole account with the commands is docs/tasks/PROPAGATION_3D.md section 2). Characterisations, not requirements:
each test skips without its oracle.

1. Over a level bed the two programs are the same program. Bellhop 2-D and Bellhop3D's N x 2-D mode agree to 0.001 dB
   at the same beam type and count, and Bellhop3D's 3-D mode agrees with both. The level-bed 1 dB was the ORACLE's
   beam count: Gaussian beams ('B', which the seamount oracle uses) with 121 beams over +-30 degrees are not converged
   (1.12 dB too quiet at 7 km against the image-source closed form; 0.26 dB with 241, 0.00 with 481). Hat beams ('G',
   Bellhop's default, which our provider uses) are converged at 241 (0.08 dB at 121). Our provider is within 0.19 dB of
   the image sources from 2 to 7 km.
2. KRAKEN's incoherent mode sum is not the incoherent ray sum when the receiver is at the source's depth: it is
   3/2 (1.76 dB) louder (the mean of sin^4 against the square of the mean of sin^2 over the mode functions; ours).
   Off the source depth KRAKEN and the image sources agree within 0.07 dB from 3 km.
3. Over relief, Bellhop's ARRIVALS output (what our provider reads) depends on how the bed is written. With the
   piecewise-linear bed ('L', what sim/bellhop.py write_env writes) the arrivals summed in power are 1.0 to 1.8 dB
   louder than Bellhop's own incoherent TL run on the gently sloping bearings beside the seamount; with Bellhop's
   curvilinear bed ('C', the same points) they agree within 0.1 dB. A parabolic equation (PyRAM) sides with the TL
   run (within 0.3 dB), and Bellhop's coherent field over 'L' is 1.9 to 2.5 dB louder than the PE where over 'C'
   it is within 0.5 dB. The normal of a piecewise-linear bed jumps at every node; our reading (not traced in
   Bellhop's source) is that the reflected beams fold there and Bellhop's arrival merge (ArrMod.f90 AddArr: same
   phase within 0.05 rad, delay within 0.05 / omega) adds the folded contributions in amplitude. This changes nothing shipped over a level bed;
   which shipped scenarios it touches is in PROPAGATION_3D.md (the three fjord scenarios have a bathymetry profile).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import tempfile

import numpy as np
import pytest

from gpusonar.sim import bathymetry, find_bellhop
from gpusonar.sim.bellhop import BellhopProvider, OceanEnvironment, read_arrivals, write_env
from gpusonar.sim.propagation import ImageSourceProvider
from bellhop3d_oracle import find_bellhop3d, run_bellhop3d
from kraken_oracle import find_kraken, read_shade, run_kraken

needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
needs_bellhop3d = pytest.mark.skipif(find_bellhop3d() is None,
                                     reason="Bellhop3D not installed (GPUSONAR_BELLHOP3D; bellhop3d.exe beside the Bellhop executable is found too)")
needs_kraken = pytest.mark.skipif(find_kraken() is None, reason="KRAKEN not installed (GPUSONAR_KRAKEN_DIR)")

# The seamount case of test_seamount_3d.py without the seamount.
DEPTH, WATER, BOTTOM = 200.0, 1500.0, (1700.0, 1.9, 0.5)
FREQUENCY, SOURCE_DEPTH = 8000.0, 50.0
RANGES = np.arange(100.0, 7000.01, 100.0)
FAR = RANGES >= 2000.0
FAN = (-30.0, 30.0)
LEVEL = OceanEnvironment(depth=DEPTH, sound_speed=WATER, bottom_speed=BOTTOM[0], bottom_density=BOTTOM[1],
                         bottom_attenuation=BOTTOM[2])


def image_loss(receiver_depth):
    """Incoherent sum of the image sources (Rayleigh coefficient of the half-space at each image's grazing angle)."""
    img = ImageSourceProvider(WATER, 0.0, DEPTH, bottom=dict(speed=BOTTOM[0], density=BOTTOM[1], attenuation=BOTTOM[2]),
                              max_order=400, engine="numpy")
    es = img.eigenrays(np.array([0.0, 0.0, SOURCE_DEPTH]), np.column_stack([RANGES, 0 * RANGES, 0 * RANGES + receiver_depth]))
    return -10.0 * np.log10(np.sum(np.abs(es.amplitude) ** 2, axis=1))


def bellhop_field(env, run_type, n_beams, depths, ranges, bed_type="L"):
    """Bellhop 2-D: TL [n_depth, n_range] (dB) of a TL run, or the arrivals of an 'A' run. `bed_type` is the letter
    the bathymetry file is written with (the provider writes 'L')."""
    with tempfile.TemporaryDirectory(prefix="gpusonar_6c8_") as tmp:
        write_env(os.path.join(tmp, "run.env"), env, FREQUENCY, SOURCE_DEPTH, depths, ranges, run_type, n_beams, FAN)
        bty = os.path.join(tmp, "run.bty")
        if bed_type != "L" and os.path.exists(bty):
            with open(bty) as f:
                rest = f.read().split("\n", 1)[1]
            with open(bty, "w") as f:
                f.write(f"'{bed_type}'\n" + rest)
        subprocess.run([find_bellhop(), "run"], cwd=tmp, capture_output=True, check=True, timeout=600)
        if run_type.startswith("A"):
            return read_arrivals(os.path.join(tmp, "run.arr"))
        p, _, _ = read_shade(os.path.join(tmp, "run.shd"))
    return -20.0 * np.log10(np.maximum(np.abs(p), 1e-30))


@needs_bellhop
@needs_bellhop3d
def test_level_bed_bellhop_and_bellhop3d_are_the_same_program(capsys):
    """Same beams, level bed: Bellhop 2-D, Bellhop3D N x 2-D and Bellhop3D 3-D. Measured: 0.00 dB between the first
    two for both beam types; 3-D (hat beams, 121 x 21 over +-10 degrees of azimuth) within 0.01 dB of N x 2-D beyond
    2 km. Bounds: 0.05 and 0.1 dB."""
    flat = lambda x, y: np.full(np.broadcast(x, y).shape, DEPTH)                   # noqa: E731
    gx, gy = np.arange(-1000.0, 8000.01, 500.0), np.arange(-2000.0, 2000.01, 500.0)
    diffs = {}
    for beam in ("G", "B"):
        two = bellhop_field(LEVEL, "I" + beam, 121, [SOURCE_DEPTH], RANGES)[0]
        nx2d = run_bellhop3d(flat, gx, gy, DEPTH, WATER, BOTTOM, FREQUENCY, SOURCE_DEPTH, [SOURCE_DEPTH], [0.0], RANGES, "2",
                             121, FAN, 1, (0.0, 0.0), (7.9, 1.9), beam=beam)[0, 0]
        diffs[beam] = np.abs(two - nx2d).max()
    nx2d_g = run_bellhop3d(flat, gx, gy, DEPTH, WATER, BOTTOM, FREQUENCY, SOURCE_DEPTH, [SOURCE_DEPTH], [0.0], RANGES, "2",
                           121, FAN, 1, (0.0, 0.0), (7.9, 1.9), beam="G")[0, 0]
    full_g = run_bellhop3d(flat, gx, gy, DEPTH, WATER, BOTTOM, FREQUENCY, SOURCE_DEPTH, [SOURCE_DEPTH], [0.0], RANGES, "3",
                           121, FAN, 21, (-10.0, 10.0), (7.9, 1.9), beam="G")[0, 0]
    three = np.abs(full_g - nx2d_g)[FAR].max()
    with capsys.disabled():
        print(f"\n[6c8] level bed, Bellhop 2-D minus Bellhop3D N x 2-D, max |diff|: hat {diffs['G']:.3f} dB, Gaussian "
              f"{diffs['B']:.3f} dB; Bellhop3D 3-D minus N x 2-D (hat) from 2 km: {three:.3f} dB")
    assert diffs["G"] < 0.05 and diffs["B"] < 0.05, diffs
    assert three < 0.1


@needs_bellhop
def test_gaussian_beams_need_more_than_121(capsys):
    """The seamount oracle's beams (Gaussian 'B', 121 over +-30 degrees) against the image sources over a level bed:
    measured 1.12 dB too quiet at 7 km (0.67 at 5 km); 481 Gaussian or 241 hat beams within 0.07 dB of the image sources
    from 2 to 7 km (0.00 at 7 km), and hat beams at 121 within 0.08 dB at 7 km. Bounds: more than 0.8 dB at 7 km for the first (a
    characterisation of the non-convergence), 0.1 dB for the converged runs."""
    ref = image_loss(SOURCE_DEPTH)
    gauss121 = bellhop_field(LEVEL, "IB", 121, [SOURCE_DEPTH], RANGES)[0] - ref
    gauss481 = bellhop_field(LEVEL, "IB", 481, [SOURCE_DEPTH], RANGES)[0] - ref
    hat241 = bellhop_field(LEVEL, "IG", 241, [SOURCE_DEPTH], RANGES)[0] - ref
    with capsys.disabled():
        print(f"\n[6c8] level bed, Bellhop minus image sources at 7 km: Gaussian 121 beams {gauss121[-1]:+.2f} dB, "
              f"481 {gauss481[-1]:+.2f} dB, hat 241 {hat241[-1]:+.2f} dB; max |diff| 2 to 7 km of the last two "
              f"{np.abs(gauss481[FAR]).max():.3f}, {np.abs(hat241[FAR]).max():.3f} dB")
    assert gauss121[-1] > 0.8
    assert np.abs(gauss481[FAR]).max() < 0.1 and np.abs(hat241[FAR]).max() < 0.1


@needs_bellhop
def test_provider_against_image_sources_over_a_level_bed(capsys):
    """Our provider (the seamount test's settings: 100 m by 20 m lookup, +-30 degrees, its default arrival selection)
    summed in power, against the image-source closed form. Measured: within 0.19 dB from 2 to 7 km (0.13 dB at 1 km);
    nearer, Bellhop's +-30 degree fan leaves out the steep bottom paths the image series keeps (1.87 dB at 100 m).
    Bound: 0.3 dB from 2 km."""
    provider = BellhopProvider(LEVEL, FREQUENCY, max_range=RANGES[-1] + 100.0, range_step=100.0, min_range=100.0,
                               depth_step=20.0, angles=FAN)
    rays = provider.eigenrays(np.array([0.0, 0.0, SOURCE_DEPTH]),
                              np.column_stack([RANGES, 0 * RANGES, 0 * RANGES + SOURCE_DEPTH]))
    ours = -10.0 * np.log10(np.sum(np.where(rays.valid, np.abs(rays.amplitude) ** 2, 0.0), axis=1))
    diff = ours - image_loss(SOURCE_DEPTH)
    with capsys.disabled():
        print(f"\n[6c8] level bed, provider minus image sources: max |diff| 2 to 7 km {np.abs(diff[FAR]).max():.2f} dB, "
              f"mean {diff[FAR].mean():+.2f} dB")
    assert np.abs(diff[FAR]).max() < 0.3


@needs_kraken
def test_kraken_incoherent_sum_at_the_source_depth(capsys):
    """KRAKEN's incoherent mode sum against the image sources: off the source depth (70 m) within 0.07 dB from 3 km
    (measured); at the source depth (50 m) 1.68 to 1.75 dB louder from 3 km, the 3/2 of the module docstring. Bounds:
    0.1 dB; louder by 1.5 to 2.0 dB."""
    k = run_kraken(DEPTH, WATER, *BOTTOM, frequency=FREQUENCY, source_depth=SOURCE_DEPTH, receiver_depths=[50.0, 70.0],
                   ranges=RANGES, incoherent=True)
    beyond = RANGES >= 3000.0
    at, off = (k[0] - image_loss(50.0))[beyond], (k[1] - image_loss(70.0))[beyond]
    with capsys.disabled():
        print(f"\n[6c8] KRAKEN incoherent minus image sources from 3 km: at the source depth {at.min():+.2f} to "
              f"{at.max():+.2f} dB, 20 m below it max |diff| {np.abs(off).max():.3f} dB")
    assert np.abs(off).max() < 0.1
    assert np.all((at < -1.5) & (at > -2.0)), at


SEAMOUNT = dict(depth=DEPTH, height=180.0, center=[2500.0, 300.0], sigma=500.0)
BESIDE = 32.0                                         # degrees: a gently sloping bearing beside the seamount
FAR_RANGES = np.arange(3500.0, 6500.01, 20.0)
PROBE = 90.0                                          # m: off the source depth (see test_kraken_incoherent_...)
# Hat beams for the runs over relief: Bellhop's automatic count here is about 10,400 (28 s for the four runs); with 4001
# the TL run is the same within 0.02 dB, and the arrivals have converged to within 0.2 dB of it (measured: arrivals
# minus TL run over 'L' +0.88, -0.22, -0.95, -1.28, -1.46, -1.50 dB with 481, 1001, 2001, 4001, automatic, 20001 beams;
# over 'C' +1.15, +0.58, +0.24, +0.08, +0.05, +0.04 dB; over the level bed +0.05 and +0.03 dB with 4001 and 20001).
N_BEAMS = 4001


def beside_the_seamount():
    """The provider's environment over the 32 degree bearing's profile (25 m samples, as test_seamount_3d.py)."""
    bed = bathymetry.make_bathymetry({"seamount": SEAMOUNT})
    u = np.array([np.cos(np.radians(BESIDE)), np.sin(np.radians(BESIDE))])
    r = np.arange(0.0, 6700.0, 25.0)
    return dataclasses.replace(LEVEL, bathymetry=[[float(a), float(bed.depth(a * u[0], a * u[1]))] for a in r]), r, bed, u


def average(loss):
    return -10.0 * np.log10(np.mean(10.0 ** (-np.asarray(loss) / 10.0)))


@needs_bellhop
def test_arrivals_over_a_piecewise_linear_bed(capsys):
    """Bellhop's arrivals summed in power against its own incoherent TL run, at 90 m, 3.5 to 6.5 km, on the 32 degree
    bearing, 4001 hat beams. Measured: with the provider's 'L' bed the arrivals are 1.28 dB louder (1.46 with the
    automatic count the provider uses); with 'C' (the same points) 0.08 dB. Over the level bed: 0.05 dB.
    Bounds: louder by more than 1.0 dB with 'L' (CHARACTERISATION of the artefact:
    when the provider's bed representation changes this must become an agreement bound); within 0.2 dB with 'C'."""
    env, _, _, _ = beside_the_seamount()
    out = {}
    for bed_type in ("L", "C"):
        arr = bellhop_field(env, "A", N_BEAMS, [PROBE], FAR_RANGES, bed_type)
        tl = bellhop_field(env, "I", N_BEAMS, [PROBE], FAR_RANGES, bed_type)[0]
        out[bed_type] = average(arr.transmission_loss(coherent=False)[0]) - average(tl)
    with capsys.disabled():
        print(f"\n[6c8] 32 deg beside the seamount, arrivals minus TL run (3.5 to 6.5 km, 90 m): 'L' bed {out['L']:+.2f} dB, "
              f"'C' bed {out['C']:+.2f} dB")
    assert out["L"] < -1.0
    assert abs(out["C"]) < 0.2


def _pyram_reason():
    try:
        import pyram  # noqa: F401
    except ImportError:
        return "PyRAM not installed (pyram; pip install pyram==1.3.0)"
    return None


@needs_bellhop
@pytest.mark.skipif(_pyram_reason() is not None, reason=str(_pyram_reason()))
def test_parabolic_equation_sides_with_the_tl_run(capsys):
    """The same bearing against PyRAM (the inputs of test_ram_crosscheck.py: 8 Pade terms, dr = 8 wavelengths, dz
    a tenth): its intensity averaged over 3.5 to 6.5 km at 90 m against Bellhop's (4001 hat beams). Measured (negative:
    Bellhop louder): incoherent TL run -0.07 dB, arrivals in power -1.35 dB; coherent TL run over 'L' -2.51 dB, over 'C'
    -0.53 dB (with receivers every 10 m instead of 20 and automatic beams: -2.20 and -0.30 dB; a coherent field sampled
    every 20 m averages to within about 0.3 dB). Bounds: TL run within 0.5 dB; arrivals louder by more than 1.0 dB;
    coherent over 'C' within 1.0 dB, over 'L' louder than over 'C' by more than 1.0 dB (measured 1.98)."""
    from pyram.PyRAM import PyRAM
    env, r, bed, u = beside_the_seamount()
    lam = WATER / FREQUENCY
    ndz = int(np.ceil(2.0 / (0.1 * lam)))
    one = lambda v: np.full((2, 1), float(v))                                      # noqa: E731
    run = PyRAM(FREQUENCY, SOURCE_DEPTH, 50.0, np.array([0.0, DEPTH]), np.array([0.0]), one(WATER), np.array([0.0, 50.0]),
                np.array([0.0]), one(BOTTOM[0]), one(BOTTOM[1]), one(BOTTOM[2]), np.array(env.bathymetry), np=8, ns=1,
                lyrw=20, c0=WATER, dr=8 * lam, dz=2.0 / ndz, ndz=ndz, zmplt=DEPTH, rmax=6600.0).run()
    keep = (run["Ranges"] >= FAR_RANGES[0]) & (run["Ranges"] <= FAR_RANGES[-1])
    pe = average(run["TL Grid"][int(np.argmin(np.abs(run["Depths"] - PROBE))), keep])
    got = {"TL run": bellhop_field(env, "I", N_BEAMS, [PROBE], FAR_RANGES)[0],
           "arrivals": bellhop_field(env, "A", N_BEAMS, [PROBE], FAR_RANGES).transmission_loss(coherent=False)[0],
           "coherent L": bellhop_field(env, "C", N_BEAMS, [PROBE], FAR_RANGES)[0],
           "coherent C": bellhop_field(env, "C", N_BEAMS, [PROBE], FAR_RANGES, "C")[0]}
    d = {k: average(v) - pe for k, v in got.items()}
    with capsys.disabled():
        print("\n[6c8] 32 deg beside the seamount, Bellhop minus PyRAM (3.5 to 6.5 km, 90 m): "
              + ", ".join(f"{k} {v:+.2f} dB" for k, v in d.items()))
    assert abs(d["TL run"]) < 0.5 and d["arrivals"] < -1.0, d
    assert abs(d["coherent C"]) < 1.0 and d["coherent L"] - d["coherent C"] < -1.0, d
