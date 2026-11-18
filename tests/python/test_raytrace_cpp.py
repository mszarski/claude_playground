"""The chain's ray code in C++ (libs/live raytrace.hpp) against its Python reference (gpusonar/raytrace.py), against
closed forms (isovelocity images), and its fan with wavefront interpolation against its own eigenray search; with
timings printed (-s). The reference itself is held to closed forms, a Runge-Kutta integration and Bellhop in
test_raytrace.py."""

import time

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import elevation, raytrace

core = gs.core
PRESSURE = 0.017
SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}


def family(layer, thermocline, thickness, bottom=300.0):
    d, c = core.live_family_profile({"family": "mixed_layer", "family_gradient": PRESSURE, "layer_depth": layer, "thermocline_gradient": thermocline,
                                     "thermocline_thickness": thickness}, 0.0, 1500.0, bottom)
    return list(d), list(c)


# the stage-2 study's illustrative profiles (200 m of water, vehicle at 50 m) and the deep one (1000 m, vehicle at 225 m)
PROFILES = {"isothermal": (*map(list, core.live_family_profile({"family": "isothermal", "family_gradient": PRESSURE}, 0.0, 1500.0, 300.0)), 200.0, 50.0),
            "strong thermocline": (*family(30.0, -0.15, 40.0), 200.0, 50.0),
            "weak thermocline": (*family(50.0, -0.05, 60.0), 200.0, 50.0),
            "no mixed layer": (*family(1.0, -0.05, 99.0), 200.0, 50.0),
            "deep": (*family(30.0, -0.15, 40.0, 1100.0), 1000.0, 225.0)}


@pytest.mark.parametrize("name", list(PROFILES))
def test_trace_and_ray_depth_are_the_reference(name):
    d, c, water, own = PROFILES[name]
    prof = raytrace.Profile(d, c, water)
    for launch in np.radians([-20.0, -5.0, -1.0, 0.3, 2.0, 7.0, 15.0]):
        for r in (500.0, 3000.0, 12000.0):
            p, q = raytrace.trace(prof, own, launch, r), core.ray_trace(d, c, water, own, float(launch), r)
            assert (p.n_surface, p.n_bottom, p.turns, p.ok) == (q["n_surface"], q["n_bottom"], q["turns"], q["ok"])
            assert q["z"] == pytest.approx(p.z, abs=1e-6) and q["time"] == pytest.approx(p.time, abs=1e-9) and q["s"] == pytest.approx(p.s, abs=1e-9)
    for angle, slant in ((0.02, 3000.0), (-0.05, 6000.0), (0.01, 12000.0), (0.3, 800.0)):
        ours = core.ray_depth_arcs(own, angle, slant, d, c, water)
        assert ours == pytest.approx(raytrace.ray_depth_reference(own, angle, slant, d, c, water), abs=1e-6)
        assert raytrace.ray_depth(own, angle, slant, d, c, water) == ours                      # the thin call
        assert elevation.ray_depth(own, angle, slant, d, c, water, method="arcs") == ours    # and the elevation module's


def test_eigenrays_are_the_reference_and_far_faster():
    d, c, water, own = PROFILES["strong thermocline"]
    prof = raytrace.Profile(d, c, water)
    t0 = time.perf_counter()
    ref = raytrace.eigenrays(prof, own, 120.0, 3000.0, bottom=SAND)
    t1 = time.perf_counter()
    got = core.ray_eigenrays(d, c, water, own, 120.0, 3000.0, bottom=SAND)
    t2 = time.perf_counter()
    assert len(ref) == len(got) >= 6
    for e, f in zip(ref, got):
        assert (e.n_surface, e.n_bottom, e.turns, e.caustic) == (f["n_surface"], f["n_bottom"], f["turns"], f["caustic"])
        assert f["time"] == pytest.approx(e.time, abs=1e-10) and f["amplitude"] == pytest.approx(e.amplitude, rel=1e-6)
    print(f"\neigenrays, 2401-ray fan, 3 km, {len(got)} found: Python {t1 - t0:.3f} s, C++ {(t2 - t1) * 1e3:.2f} ms")
    assert (t2 - t1) < 0.2 * (t1 - t0)


def test_fan_in_isovelocity_water_is_the_image_solution():
    """Straight rays: every arrival read off the fan is an image path, its time the path length over c and its
    amplitude one over the path length; the edge of a sheet at the surface is reached (a shallow target far away)."""
    for own, z, r in ((225.0, 150.0, 12000.0), (225.0, 4.0, 12000.0), (50.0, 20.0, 1000.0), (30.0, 300.0, 6000.0)):
        fan = core.RayFan([0.0, 5000.0], [1500.0, 1500.0], 0.0, own, [r], 12.0, 600, 1)
        got = {(a["n_surface"], a["n_bottom"]): a for a in fan.arrivals(0, z)}
        for key, path in (((0, 0), np.hypot(r, z - own)), ((1, 0), np.hypot(r, z + own))):
            assert key in got, (own, z, r, key)
            assert got[key]["time"] == pytest.approx(path / 1500.0, abs=2e-9)          # nanoseconds: far below a carrier period
            assert got[key]["amplitude"] == pytest.approx(1.0 / path, rel=2e-3)
            assert got[key]["p"] == pytest.approx(r / path / 1500.0, rel=1e-6)          # d time / d range


@pytest.mark.parametrize("name", list(PROFILES))
def test_fan_arrivals_are_the_eigenrays(name):
    """Refracting water: the arrivals interpolated along the wavefront have the times of the eigenrays found by
    bisection to a tenth of a microsecond, and their amplitudes away from caustics."""
    d, c, water, own = PROFILES[name]
    checked = 0
    for z, r in ((120.0, 3000.0), (20.0, 1000.0), (150.0, 6000.0)):
        eig = core.ray_eigenrays(d, c, water, own, z, r, max_angle_deg=40.0, n_fan=4801, max_bounces=2, bottom=SAND)
        fan = core.RayFan(d, c, water, own, [r], 40.0, 2400, 2)
        arr = fan.arrivals(0, z)
        for e in eig:
            same = [a for a in arr if (a["n_surface"], a["n_bottom"]) == (e["n_surface"], e["n_bottom"])]
            assert same, (name, z, r, e)
            a = min(same, key=lambda a: abs(a["time"] - e["time"]))
            assert a["time"] == pytest.approx(e["time"], abs=1e-7), (name, z, r, e)
            if not (a["caustic"] or e["caustic"]) and e["n_bottom"] == 0:
                # geometric spreading; the surface coefficient (-1) is the caller's
                if abs(20.0 * np.log10(a["amplitude"] / abs(e["amplitude"]))) < 1.0:
                    checked += 1
    assert checked >= 2, name                                                         # most; at kinks of the profile the tubes differ


def test_timings():
    d, c, water, own = PROFILES["deep"]
    t0 = time.perf_counter()
    for _ in range(20):
        core.RayFan(d, c, water, own, [11900.0, 12060.0], 8.0, 1200, 2)
    fan_s = (time.perf_counter() - t0) / 20
    fan = core.RayFan(d, c, water, own, [11900.0, 12060.0], 8.0, 1200, 2)
    t0 = time.perf_counter()
    for z in np.linspace(5.0, 500.0, 200):
        fan.arrivals(0, float(z))
    look_s = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        core.ray_depth_arcs(own, 0.01, 12000.0, d, c, water)
    depth_s = (time.perf_counter() - t0) / 200
    print(f"\nfan of 1200 rays to 12 km at two ranges: {fan_s * 1e3:.2f} ms; arrivals at one depth: {look_s * 1e6:.0f} us; ray_depth_arcs: {depth_s * 1e6:.0f} us")
    assert fan_s < 0.5 and depth_s < 0.05
