"""The layered-profile ray code (python/gpusonar/raytrace.py) against closed forms and independent integration:
isovelocity images (times, 1 / R amplitudes, the Lloyd-mirror pattern), the circle and the travel time of a ray in a
constant gradient, an adaptive Runge-Kutta integration of the ray equations through a thermocline with turning points
and reflections, the chain's own `ray_depth`, and Bellhop's arrival times as an outside oracle (skipped without it).
"""

import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import gpusonar as gs
from gpusonar import elevation, lloyd, raytrace

core = gs.core
SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}
# An illustrative profile (defined here, not data): 30 m mixed layer, a thermocline of -0.15 /s for 40 m, pressure gradient below.
THERMO_Z = np.array([0.0, 30.0, 70.0, 400.0])
THERMO_C = np.array([1500.0, 1500.51, 1494.51, 1500.12])


def test_isovelocity_images_times_amplitudes_and_lloyd_mirror():
    prof = raytrace.Profile([0.0, 100.0], [1500.0, 1500.0], 100.0)
    zs, zr, r = 30.0, 70.0, 800.0
    rays = raytrace.eigenrays(prof, zs, zr, r, max_bounces=2, bottom=SAND)
    want = {(0, 0): np.hypot(r, zr - zs), (1, 0): np.hypot(r, zr + zs), (0, 1): np.hypot(r, 200.0 - zr - zs),
            (1, 1): None, (2, 0): None, (0, 2): None}
    seen = {}
    for e in rays:
        seen.setdefault((e.n_surface, e.n_bottom), []).append(e)
    for key in ((0, 0), (1, 0), (0, 1)):
        (e,) = seen[key]
        assert e.time == pytest.approx(want[key] / 1500.0, abs=2e-7)
        assert abs(e.amplitude) == pytest.approx((1.0 / want[key]) * (1.0 if key != (0, 1) else abs(raytrace._rayleigh(np.arctan2(200.0 - zr - zs, r), 1500.0, SAND))), rel=2e-3)
    assert seen[(1, 0)][0].amplitude.real < 0.0                    # the surface inverts
    # the two rays with one bounce off each boundary: images at 2 D +- (z_r - z_s) ... by their path lengths
    lengths = sorted(e.time * 1500.0 for e in seen[(1, 1)])
    assert lengths == pytest.approx(sorted([np.hypot(r, 200.0 + zr - zs), np.hypot(r, 200.0 - zr + zs)]), abs=1e-3)
    # no bottom given: only the direct path and the surface image, and their coherent sum is the Lloyd-mirror pattern
    deep = raytrace.Profile([0.0, 5000.0], [1500.0, 1500.0])
    for z1, z2, rng in ((225.0, 150.0, 12000.0), (225.0, 4.0, 12000.0), (50.0, 20.0, 3000.0)):
        pair = raytrace.eigenrays(deep, z1, z2, rng, max_bounces=1)
        assert [(e.n_surface, e.n_bottom) for e in pair] == [(0, 0), (1, 0)]
        for f in (4000.0, 5000.0, 6000.0):
            h, _ = lloyd.one_way([f], z1, z2, rng)
            direct_tl = 20.0 * np.log10(np.hypot(rng, z1 - z2))
            assert raytrace.transmission_loss(pair, f) - direct_tl == pytest.approx(-20.0 * np.log10(abs(h[0])), abs=0.05)
        assert raytrace.transmission_loss(pair) == pytest.approx(-10.0 * np.log10(1 / np.hypot(rng, z1 - z2) ** 2 + 1 / np.hypot(rng, z1 + z2) ** 2), abs=0.02)


def test_constant_gradient_ray_is_a_circle_with_the_closed_form_time():
    g, c0, z0 = 0.05, 1480.0, 2000.0
    prof = raytrace.Profile([0.0, 5000.0], [c0 - g * z0, c0 + g * (5000.0 - z0)])
    for launch_deg in (-3.0, -1.0, 2.0, 6.0):
        th = np.radians(launch_deg)
        radius = c0 / (g * np.cos(th))
        z_c, r_c = z0 - c0 / g, (c0 / g) * np.tan(th)              # the centre: where c would be zero
        for r in (500.0, 3000.0, 9000.0):
            ray = raytrace.trace(prof, z0, th, r)
            assert ray.n_surface == 0 and ray.n_bottom == 0
            assert np.hypot(r - r_c, ray.z - z_c) == pytest.approx(radius, rel=1e-10)
            # t = (1 / g) [atanh(sin th_0) - atanh(sin th)], sin th = sin th_0 - p g r
            s_end = np.sin(th) - np.cos(th) / c0 * g * r
            assert ray.time == pytest.approx((np.arctanh(np.sin(th)) - np.arctanh(s_end)) / g, rel=1e-12)
            assert ray.s == pytest.approx(s_end, abs=1e-12)
    # upward-refracting water turns a downward ray: one turning point, at the depth where c = c_0 / cos(theta_0)
    path = []
    ray = raytrace.trace(prof, z0, np.radians(6.0), 9000.0, path=path)
    assert ray.turns == 1 and max(p[1] for p in path) == pytest.approx(z0 + c0 * (1.0 / np.cos(np.radians(6.0)) - 1.0) / g, rel=1e-9)


def integrate(z0, launch_down, r_end, depth, speed, water):
    """The ray equations in range by an adaptive Runge-Kutta integrator, mirrored at the boundaries: nothing in common
    with the closed-form stepping. Returns (z, t)."""
    grad = np.diff(speed) / np.diff(depth)

    def rhs(r, y):                              # y = (z down, theta down, t)
        z, th, _ = y
        i = int(np.clip(np.searchsorted(depth, z, side="right") - 1, 0, len(grad) - 1))
        c = speed[i] + grad[i] * (z - depth[i])
        return [np.tan(th), -grad[i] / c, 1.0 / (c * np.cos(th))]

    def top(r, y):
        return y[0]

    def bottom(r, y):
        return y[0] - water

    top.terminal = bottom.terminal = True
    top.direction, bottom.direction = -1.0, 1.0
    r, y = 0.0, [z0, launch_down, 0.0]
    while r < r_end:
        sol = solve_ivp(rhs, (r, r_end), y, rtol=1e-11, atol=1e-11, max_step=1.0, events=[top, bottom])
        r, y = float(sol.t[-1]), [float(v) for v in sol.y[:, -1]]
        if sol.status == 1:
            y[1] = -y[1]                         # mirrored
            y[0] = min(max(y[0], 1e-9), water - 1e-9)
    return y[0], y[2]


def test_thermocline_against_runge_kutta_with_turns_and_bounces():
    prof = raytrace.Profile(THERMO_Z, THERMO_C, 200.0)
    for z0, launch_deg, r in ((50.0, -3.0, 3000.0), (50.0, 2.0, 6000.0), (20.0, 0.7, 3000.0), (120.0, -6.0, 6000.0), (50.0, 9.0, 2500.0)):
        ray = raytrace.trace(prof, z0, np.radians(launch_deg), r)
        z, t = integrate(z0, np.radians(launch_deg), r, prof.depth, prof.speed, 200.0)
        assert ray.z == pytest.approx(z, abs=0.02), (z0, launch_deg, r)
        assert ray.time == pytest.approx(t, abs=2e-6)


def test_ray_depth_is_the_chains_ray_depth():
    """`elevation.ray_depth` (small Snell steps, both chain engines) and the closed-form `raytrace.ray_depth` answer the same
    question: within a metre or two over ranges with turning points and bounces (the stepping's own error)."""
    for own, el_deg, slant in ((50.0, 1.0, 1000.0), (50.0, -2.0, 3000.0), (120.0, 3.0, 3000.0), (50.0, 4.0, 6000.0), (225.0, -1.5, 6000.0)):
        a = elevation.ray_depth(own, np.radians(el_deg), slant, THERMO_Z, THERMO_C, 300.0)
        b = raytrace.ray_depth(own, np.radians(el_deg), slant, THERMO_Z, THERMO_C, 300.0)
        c = core.live_ray_depth(own, np.radians(el_deg), slant, list(THERMO_Z), list(THERMO_C), 300.0)
        assert b == pytest.approx(a, abs=2.0) and b == pytest.approx(c, abs=2.0), (own, el_deg, slant)


def test_surface_delay_isovelocity_and_its_inversion():
    deep = raytrace.Profile([0.0, 5000.0], [1500.0, 1500.0])
    tau = raytrace.surface_delay(deep, 225.0, 150.0, 12000.0)
    assert tau == pytest.approx((np.hypot(12000.0, 375.0) - np.hypot(12000.0, 75.0)) / 1500.0, abs=1e-8)
    assert raytrace.flat_depth_from_delay(tau, 225.0, 12000.0, 1500.0) == pytest.approx(150.0, abs=0.01)
    # a shadow: under a strong thermocline a shallow source has no direct path to a shallow target far away
    prof = raytrace.Profile(THERMO_Z, THERMO_C, 200.0)
    assert raytrace.surface_delay(prof, 60.0, 65.0, 12000.0) is None


@pytest.mark.skipif(not os.environ.get("GPUSONAR_BELLHOP"), reason="Bellhop not available (GPUSONAR_BELLHOP)")
def test_arrival_times_against_bellhop():
    """Bellhop (run as a separate program) through the same thermocline: every arrival of ours with at most two bounces
    has one of Bellhop's with the same bounce counts within 20 microseconds, and its amplitude within 1.5 dB where
    neither is at a caustic."""
    from gpusonar.sim.bellhop import OceanEnvironment, run_bellhop
    env = OceanEnvironment(depth=200.0, ssp_depth=THERMO_Z[:3].tolist() + [200.0], ssp_speed=THERMO_C[:3].tolist() + [float(np.interp(200.0, THERMO_Z, THERMO_C))],
                           bottom_speed=SAND["speed"], bottom_density=SAND["density"], bottom_attenuation=SAND["attenuation"])
    prof = raytrace.Profile(THERMO_Z, THERMO_C, 200.0)
    checked = 0
    for zs, zr, rng in ((50.0, 120.0, 3000.0), (50.0, 20.0, 1000.0), (120.0, 150.0, 6000.0)):
        arr = run_bellhop(env, 5000.0, zs, [zr], [rng], n_beams=4001, angles=(-40.0, 40.0))
        t_b = np.asarray(arr.delay[0][0]).real
        ns_b, nb_b = np.asarray(arr.n_surface[0][0]), np.asarray(arr.n_bottom[0][0])
        a_b = np.abs(np.asarray(arr.amplitude[0][0]))
        for e in raytrace.eigenrays(prof, zs, zr, rng, max_bounces=2, max_angle_deg=40.0, bottom=SAND):
            same = (ns_b == e.n_surface) & (nb_b == e.n_bottom)
            assert same.any(), (zs, zr, rng, e)
            k = np.flatnonzero(same)[np.argmin(np.abs(t_b[same] - e.time))]
            assert abs(t_b[k] - e.time) < 2e-5, (zs, zr, rng, e, t_b[k])
            if not e.caustic and e.n_bottom == 0:
                assert 20.0 * np.log10(abs(e.amplitude) / a_b[k]) == pytest.approx(0.0, abs=1.5), (zs, zr, rng, e)
            checked += 1
    assert checked >= 6
