"""Depth from multipath timing (python/gpusonar/multipath_depth.py, libs/live multipath_depth.hpp): the predicted
pattern against its closed forms, the C++ stage against the NumPy reference, depth against truth on simulated pings
(one-row face build and test-AUV prototype, image-source surface; shallow water with bottom images; a rough surface;
an extended target; Bellhop's refracting water), the error against a Cramer-Rao bound, how often a depth is reported
where there is none (noise, single-arrival echoes), the reasons it gives when it cannot work, a track's accumulated
depth, the combination with a stacked-row depth, and the two chain engines reporting the same members.
"""

import copy
import os

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import multipath_depth as mp
from gpusonar import recon
from gpusonar.live import LiveChain
from gpusonar.sim import run_scenario

core = gs.core
SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}     # the popoto_12km scenario's bottom
KNOWN = {"gradient_sigma_per_s": 0.0}                            # the simulated water is isovelocity and taken as known: the sigma tested
                                                                 # is the fit's own (the gradient prior has its own test below)


# D-256: popoto_long_range sends the 2 s hyperbolic sweep since. A test whose levels were tuned to the 1 s linear sweep it sent
# before (its echo energy and structure) states that sweep with `former=True`.
FORMER_SWEEP = {"waveform": {"type": "lfm", "source": None, "duration": 1.0}}


def scenario(own, target, rng, ts=25.0, seed=3, sonar="popoto_long_range", half=600.0, surface=True, bottom_z=None, spread=None, scene=None,
             noise=True, pings=1, former=False):
    """One point target at broadside of a one-row sonar, straight rays, built here with overrides (no scenario file)."""
    cfg = {"name": "multipath_depth_test", "sonar": sonar, "fidelity": "image_source" if surface else "free_field",
           "sonar_overrides": {"receive": {"kind": "baseband", "sample_rate": 8.0e3, "window": {"range": [max(rng - half, 80.0), rng + half]}}},
           "environment": {"sound_speed": 1500.0, "temperature": 12.0, "salinity": 35.0},
           "platform": {"position": [0.0, 0.0, float(own)], "rpy": [0, 0, 0], "velocity": [0.0, 0.0, 0.0]},
           "pings": {"count": pings, "interval": 20.0},
           "scene": scene or [{"type": "points", "label": "target", "positions": [[0.0, float(rng), float(target)]], "target_strength_db": float(ts)}],
           "absorption": "ainslie_mccolm", "receiver": {"adc": False}, "seed": seed}
    if surface:
        cfg["propagation"] = {"surface_z": 0.0, "bottom_z": bottom_z, "max_order": 1 if bottom_z is None else 2}
        if bottom_z is not None:
            cfg["propagation"]["bottom"] = dict(SAND)
    if spread:
        cfg["time_spread"] = spread
    if noise:
        cfg["receiver"]["ambient_noise"] = {"wind_speed": 5.0, "shipping": 0.5}
    if former:
        cfg["sonar_overrides"]["transmit"] = dict(FORMER_SWEEP)
    return cfg


def prepared(cfg):
    rec = run_scenario(copy.deepcopy(cfg), engine="cpp").records[0]
    recon.prepare([rec])
    return rec, rec.rx_positions().mean(axis=0)


def detect(cfg, own, target, rng, opts=None, offset=1.0):
    """The C++ stage's estimate for the target of `scenario`, asked for at its slant range plus `offset` m."""
    rec, origin = prepared(cfg)
    return core.live_multipath_detection(rec, tuple(origin), 0.0, float(np.hypot(rng, target - own)) + offset, mp.cpp_options(dict(KNOWN, **(opts or {}))))


GEOM = mp.Geometry(range_m=12000.0, array_depth=225.0, projector_depth=225.0, sound_speed=1500.0, f_centre=5000.0, bandwidth=2000.0)


# --- the pattern ----------------------------------------------------------------------------------------------------------

def test_pattern_is_the_triplet_with_the_textbook_delay():
    """Projector and array at one depth, flat surface: +1, -2 (two coincident one-leg echoes), +1 at 0, tau, 2 tau with
    tau = 2 z z_t / (r c) to first order (3.75 ms at 12 km for 225 m and 150 m) and exactly the image geometry."""
    cfg = mp.options(KNOWN)
    delay, amp = mp.pattern(150.0, 2.0 * 12000.0 / 1500.0, GEOM, cfg)
    r_h = np.sqrt(12000.0**2 - 75.0**2)
    exact = (np.hypot(r_h, 375.0) - np.hypot(r_h, 75.0)) / 1500.0
    assert delay[:, 0] == pytest.approx([0.0, exact, exact, 2.0 * exact], abs=1e-12)
    assert exact == pytest.approx(2.0 * 225.0 * 150.0 / (12000.0 * 1500.0), rel=2e-3) and exact == pytest.approx(3.75e-3, rel=2e-3)
    assert amp[0, 0] == 1.0 and amp[1, 0].real == pytest.approx(-1.0, abs=2e-3) and amp[3, 0].real == pytest.approx(1.0, abs=4e-3)
    # the pattern's energy: resolved arrivals add in power, 1 + |2|^2 + 1 for the coincident pair
    assert mp.pattern_energy(delay, amp, 5000.0, 2000.0)[0] == pytest.approx(6.0, rel=0.05)
    # a lever arm: the projector 1.5 m below the array splits the middle arrival by 2 x 1.5 x z_t / (r c)
    low = mp.Geometry(**{**GEOM.__dict__, "projector_depth": 226.5})
    d2, _ = mp.pattern(150.0, 16.0, low, cfg)
    assert d2[2, 0] - d2[1, 0] == pytest.approx(2.0 * 1.5 * 150.0 / (12000.0 * 1500.0), rel=5e-3)
    # the spectral ripple of the echo has period 1 / tau = c r / (2 z z_t)
    f = np.linspace(4000.0, 6000.0, 4001)
    h = np.abs(np.sum(amp[:, 0, None] * np.exp(-2j * np.pi * f[None, :] * delay[:, 0, None]), axis=0))
    nulls = f[1:-1][(h[1:-1] < h[:-2]) & (h[1:-1] < h[2:])]
    assert np.diff(nulls) == pytest.approx(1.0 / exact, rel=0.01)


def test_cpp_pattern_grid_and_roughness_are_the_reference():
    for opts, water in (({}, None), ({"surface": {"rms_height_m": 0.3, "loss_db": 1.0}}, None), ({"bottom": SAND, "bottom_order": 2}, 300.0)):
        cfg = mp.options(opts)
        g = mp.Geometry(**{**GEOM.__dict__, "range_m": 3000.0, "projector_depth": 226.5, "water_depth": water})
        delay, amp = mp.pattern(120.0, 4.001, g, cfg)
        d2, a2 = core.live_multipath_pattern(120.0, 4.001, mp.geometry_dict(g), mp.cpp_options(opts))
        assert np.allclose(d2, delay[:, 0], atol=1e-13) and np.allclose(a2, amp[:, 0], atol=1e-12)
        grid, _, _ = mp.depth_grid(g, cfg)
        assert np.allclose(core.live_multipath_depth_grid(mp.geometry_dict(g), mp.cpp_options(opts)), grid, atol=1e-9)
    # a rough surface: the one-leg echo is exp(-Gamma^2 / 2) of the flat one, Gamma = 2 k h sin(grazing)
    _, amp = mp.pattern(150.0, 4.0, mp.Geometry(**{**GEOM.__dict__, "range_m": 3000.0}), mp.options({"surface": {"rms_height_m": 0.3}}))
    r_h = np.sqrt(3000.0**2 - 75.0**2)
    gamma = 2.0 * (2.0 * np.pi * 5000.0 / 1500.0) * 0.3 * 375.0 / np.hypot(r_h, 375.0)
    assert abs(amp[1, 0]) == pytest.approx(np.exp(-0.5 * gamma**2) * np.hypot(r_h, 75.0) / np.hypot(r_h, 375.0), rel=1e-9)


@pytest.mark.parametrize("own, target, rng, ts", [(225.0, 150.0, 12000.0, 26.0), (225.0, 300.0, 6000.0, 15.0), (50.0, 60.0, 3000.0, 10.0)])
def test_cpp_stage_is_the_numpy_reference(own, target, rng, ts):
    rec, origin = prepared(scenario(own, target, rng, ts=ts))
    slant = float(np.hypot(rng, target - own)) + 1.0
    ref = mp.estimate_detection(rec, origin, 0.0, slant, mp.options(KNOWN))
    got = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, mp.cpp_options(KNOWN))
    assert ref.valid and got["valid"] and got["method"] == ref.method
    assert got["depth"] == pytest.approx(ref.depth, abs=1e-6) and got["sigma"] == pytest.approx(ref.sigma, rel=1e-6)
    assert np.allclose(got["envelope"], ref.envelope, atol=1e-5 * max(1.0, ref.gain)) and np.allclose(got["grid"], ref.grid, atol=1e-9)


# --- depth against truth --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sonar, own, target, rng, ts", [
    ("popoto_long_range", 225.0, 150.0, 12000.0, 28.0),    # the face row at its design range, a deep target
    ("popoto_long_range", 225.0, 40.0, 12000.0, 28.0),     # ... and the shallowest it can do there
    ("popoto_long_range", 225.0, 300.0, 6000.0, 15.0),
    ("popoto_long_range", 225.0, 30.0, 6000.0, 15.0),
    ("popoto_long_range", 225.0, 150.0, 3000.0, 3.0),
    ("popoto_long_range", 100.0, 30.0, 3000.0, 3.0),
    ("popoto_long_range", 225.0, 60.0, 1000.0, -12.0),
    ("popoto_testauv_1row", 30.0, 60.0, 1000.0, 5.0),      # the test-AUV prototype (0.1 s pulse, to 2 km)
    ("popoto_testauv_1row", 30.0, 15.0, 500.0, -5.0),
    ("popoto_testauv_1row", 50.0, 120.0, 1800.0, 12.0),
])
def test_depth_of_a_point_target_under_a_flat_surface(sonar, own, target, rng, ts):
    worst = 0.0
    for seed in (11, 12, 13):
        d = detect(scenario(own, target, rng, ts=ts, seed=seed, sonar=sonar), own, target, rng)
        assert d["valid"] and not d["ambiguous"], d["reason"]
        assert abs(d["depth"] - target) <= max(3.0 * d["sigma"], 0.3), (seed, d["depth"], d["sigma"])
        assert d["arrivals"] == 4
        worst = max(worst, abs(d["depth"] - target))
    # one resolution cell of the surface echo's delay in depth, r c / (2 B (z_a + z_p)), is the coarsest it can be
    assert worst <= rng * 1500.0 / (2.0 * 2000.0 * 2.0 * own)


def test_the_detection_may_sit_on_any_arrival_or_come_with_its_extent():
    """Asked for at the direct arrival, at the surface-surface one, or with a merging detector's first_range and extent."""
    own, target, rng = 225.0, 150.0, 6000.0
    rec, origin = prepared(scenario(own, target, rng, ts=18.0))
    slant = float(np.hypot(rng, target - own))
    later = 2.0 * own * target / rng                          # extra path of one leg by the surface, m
    o = mp.cpp_options(KNOWN)
    depths = [core.live_multipath_detection(rec, tuple(origin), 0.0, slant + off, o)["depth"] for off in (0.0, 0.5 * later, later)]
    depths.append(core.live_multipath_detection(rec, tuple(origin), 0.0, slant + 0.7 * later, o, first_range=slant - 1.0, extent=later + 2.0)["depth"])
    assert depths == pytest.approx([target] * 4, abs=0.5)


def test_shallow_water_needs_the_bottom_in_the_pattern():
    """100 m of water over sand, target at 70 m seen from 50 m at 2 km: with the bottom's images in the pattern the depth is
    right; with the surface-only pattern the fit locks onto bottom echoes, and says that it may have (ambiguous, a note)."""
    own, target, rng, water = 50.0, 70.0, 2000.0, 100.0
    cfg = scenario(own, target, rng, ts=0.0, bottom_z=water)
    good = detect(cfg, own, target, rng, {"water_depth": water, "bottom": SAND, "bottom_order": 2})
    assert good["valid"] and abs(good["depth"] - target) <= max(3.0 * good["sigma"], 0.5) and good["arrivals"] > 4
    blind = detect(cfg, own, target, rng, {"water_depth": water})
    assert blind["valid"] and blind["ambiguous"] and blind["reason"] == "bottom_unmodelled"
    assert abs(blind["depth"] - target) > 10.0                 # which is why it is flagged
    low = detect(cfg, own, target, rng, {"water_depth": water, "bottom": SAND, "bottom_order": 1})
    assert low["ambiguous"] and low["reason"] == "bottom_order"
    # deep water whose depth is known: the bottom path is steep, the surface-only pattern is not doubted
    deep = detect(scenario(225.0, 150.0, 1000.0, ts=-10.0), 225.0, 150.0, 1000.0, {"water_depth": 1500.0})
    assert deep["valid"] and not deep["ambiguous"] and abs(deep["depth"] - 150.0) < 0.5


def test_rough_surface_with_time_spreading():
    """The simulator's rough surface (coherent loss and scattered taps). At 12 km the grazing angle is 1.8 degrees and a
    0.3 m rms surface is a mirror: the depth holds. At 1 km (21 degrees) the coherent echo is gone: told the roughness, the
    stage reports no depth rather than a wrong one."""
    spread = {"surface": {"rms_height": 0.3}, "taps": 3}
    for seed in (3, 4):
        far = detect(scenario(225.0, 150.0, 12000.0, ts=30.0, seed=seed, spread=spread), 225.0, 150.0, 12000.0, {"surface": {"rms_height_m": 0.3}})
        assert far["valid"] and abs(far["depth"] - 150.0) <= max(3.0 * far["sigma"], 1.0)
        near = detect(scenario(225.0, 150.0, 1000.0, ts=-10.0, seed=seed, spread=spread), 225.0, 150.0, 1000.0, {"surface": {"rms_height_m": 0.3}})
        assert not near["valid"] and near["reason"] in ("no_surface_path", "single_arrival")


def test_an_extended_target_is_not_taken_for_multipath():
    """Five highlights of one sign along 20 to 100 m of range, no surface: no depth. The same body under a surface: its depth."""
    for rng, ts, length in ((6000.0, 20.0, 20.0), (6000.0, 20.0, 60.0), (3000.0, 5.0, 100.0)):
        hl = [{"position": [0.0, float(x), 0.0], "target_strength_db": 0.0} for x in np.linspace(-length / 2, length / 2, 5)]
        scene = [{"type": "target", "label": "body", "origin": [0.0, rng, 150.0], "velocity": [0.0, 0.0, 0.0], "heading_deg": 90.0,
                  "aspect_table": {"aspect_deg": [0, 180], "ts_db": [ts, ts]}, "highlights": hl}]
        free = detect(scenario(225.0, 150.0, rng, scene=scene, surface=False), 225.0, 150.0, rng)
        assert not free["valid"] and free["reason"] == "single_arrival"
        under = detect(scenario(225.0, 150.0, rng, scene=scene), 225.0, 150.0, rng)
        assert under["valid"] and abs(under["depth"] - 150.0) <= max(3.0 * under["sigma"], 1.0)


def test_reasons_when_it_cannot_work():
    # sonar too shallow for any target depth at this range: tau B < 3 down to the search limit (the surface-cancellation regime)
    d = detect(scenario(10.0, 150.0, 12000.0, ts=40.0), 10.0, 150.0, 12000.0)
    assert not d["valid"] and d["reason"] == "merged"
    # target too shallow: its depth is above the shallowest the band can resolve, 3 r c / (4 B z_own) = 30 m here
    g = mp.Geometry(**{**GEOM.__dict__})
    grid, _, _ = mp.depth_grid(g, mp.options(KNOWN))
    assert grid[0] == pytest.approx(3.0 * 12000.0 * 1500.0 / (4.0 * 2000.0 * 225.0), rel=0.02)
    d = detect(scenario(225.0, 5.0, 12000.0, ts=45.0), 225.0, 5.0, 12000.0)
    assert not d["valid"] and d["reason"] == "single_arrival"
    # no surface path: bubbles at a high sea state (a loss per bounce the user evaluates from the named law)
    d = detect(scenario(225.0, 150.0, 6000.0, ts=18.0), 225.0, 150.0, 6000.0, {"surface": {"loss_db": 10.0}})
    assert not d["valid"] and d["reason"] == "no_surface_path"
    # the window does not fit in the record
    d = detect(scenario(225.0, 150.0, 6000.0, ts=18.0, half=60.0), 225.0, 150.0, 6000.0)
    assert not d["valid"] and d["reason"] == "window"
    with pytest.raises(ValueError):
        mp.options({"no_such_key": 1})


def test_refraction_first_order_against_the_ray_code():
    """In water with a constant gradient g the surface echo's delay is the flat one times 1 + r^2 g / (2 c (z_own + z_t)) to
    first order: against the closed-form ray code over 1 to 3 km in isothermal water (g = 0.017 /s). Taken out with
    `gradient_per_s`, the flat formula's depth comes back to truth; left in, it is the bias; and the gradient prior
    (the elevation depth's 0.05 /s unless told) is in the reported sigma, in both implementations."""
    from gpusonar import raytrace
    g_iso, own = 0.017, 50.0
    prof = raytrace.Profile([0.0, 3000.0], [1500.0, 1500.0 + g_iso * 3000.0])
    c_own = float(prof.c(own))
    for target, rng, tol in ((120.0, 1000.0, 0.004), (120.0, 2000.0, 0.02), (120.0, 3000.0, 0.05), (60.0, 1000.0, 0.01), (20.0, 1000.0, 0.02)):
        tau = raytrace.surface_delay(prof, own, target, rng)
        flat = (np.hypot(rng, target + own) - np.hypot(rng, target - own)) / c_own
        first = rng**2 * g_iso / (2.0 * c_own * (own + target))
        assert tau / flat - 1.0 == pytest.approx(first, abs=tol)
        geo = mp.Geometry(range_m=rng, array_depth=own, projector_depth=own, sound_speed=c_own, f_centre=5000.0, bandwidth=2000.0)
        z_flat = raytrace.flat_depth_from_delay(tau, own, rng, c_own)
        z, _ = mp.refraction(z_flat, geo, mp.options({"gradient_per_s": g_iso}))
        assert abs(z - target) <= max(3.0 * tol, 0.01) * target and abs(z - target) < 0.35 * abs(z_flat - target) + 0.3
        # the prior: sigma = z r^2 sigma_g / (2 c (z_own + z)), and the C++ function is the same
        z0, sigma = mp.refraction(target, geo, mp.options({}))
        assert z0 == target and sigma == pytest.approx(target * rng**2 * 0.05 / (2.0 * c_own * (own + target)))
        assert core.live_multipath_refraction(z_flat, mp.geometry_dict(geo), mp.cpp_options({"gradient_per_s": g_iso})) == pytest.approx((z, mp.refraction(z_flat, geo, mp.options({"gradient_per_s": g_iso}))[1]))
    # through the stage: the default sigma carries the prior (12 km, 225 m and 150 m: hundreds of metres), told-known water does not
    d = detect(scenario(225.0, 150.0, 12000.0, ts=30.0), 225.0, 150.0, 12000.0, {"gradient_sigma_per_s": None})
    assert d["valid"] and d["refraction_sigma"] == pytest.approx(150.0 * d["refraction_sigma"] / d["refraction_sigma"] * 12000.0**2 * 0.05 / (2.0 * 1500.0 * 375.0), rel=0.02)
    assert d["sigma"] >= d["refraction_sigma"] > 900.0
    # with the rows' depth: the gradient is one error in both, it is not averaged down
    f = mp.fuse(150.0, 60.0, 156.0, 80.0, refraction_mp=59.0, refraction_el=79.0)
    assert f["consistent"] and 59.0 < f["sigma"] < 80.0
    assert core.live_multipath_fuse(150.0, 60.0, 156.0, 80.0, [], 3.0, 59.0, 79.0)["sigma"] == pytest.approx(f["sigma"])


# --- noise, single arrivals, and the bound --------------------------------------------------------------------------------

def test_false_depth_rate_on_noise_and_on_single_arrival_echoes():
    """No surface (free field): one arrival, at detection-level and at high SNR, at 12, 3 and 1 km; and no target at all.
    With `min_gain` 12 a depth is reported for none of the echoes and for at most one window of noise in forty."""
    o = mp.cpp_options(KNOWN)
    false_echo = false_noise = 0
    for own, rng, levels in ((225.0, 12000.0, (27.0, 36.0, 46.0)), (225.0, 3000.0, (5.0, 20.0)), (50.0, 1000.0, (-10.0, 10.0))):
        for ts in levels:
            for seed in range(20):
                rec, origin = prepared(scenario(own, 150.0, rng, ts=ts, seed=500 + seed, surface=False))
                d = core.live_multipath_detection(rec, tuple(origin), 0.0, float(np.hypot(rng, 150.0 - own)) + 1.0, o)
                assert d["snr_db"] > 9.0
                false_echo += bool(d["valid"])
        for seed in range(40):
            rec, origin = prepared(scenario(own, 150.0, rng, ts=-80.0, seed=700 + seed, surface=False))
            false_noise += bool(core.live_multipath_detection(rec, tuple(origin), 0.0, rng, o)["valid"])
    assert false_echo == 0
    assert false_noise <= 3


def test_error_against_the_cramer_rao_bound():
    """At high SNR the fine (fringe) depth scatters about the truth by about the bound for this pattern in white noise;
    the reported sigma is never below what was measured. 6 km, 225 m and 150 m: direct-arrival SNR about 25 dB."""
    own, target, rng = 225.0, 150.0, 6000.0
    errors, sigmas, snr = [], [], []
    for seed in range(24):
        d = detect(scenario(own, target, rng, ts=26.0, seed=900 + seed), own, target, rng)
        assert d["valid"]
        errors.append(d["depth"] - target)
        sigmas.append(d["sigma"])
        snr.append(d["snr_db"])
    # the strongest sample is the coincident pair, 6 dB above the direct arrival
    g = mp.Geometry(range_m=float(np.hypot(rng, 75.0)), array_depth=own, projector_depth=own + 1.5, sound_speed=1500.0, f_centre=5000.0, bandwidth=2000.0)
    bound = mp.depth_bound(target, g, mp.options(KNOWN), 10.0 ** ((np.mean(snr) - 6.0) / 10.0))
    scatter = float(np.std(errors))
    assert bound < 0.05                                            # centimetres: the carrier phase of a mirror
    assert scatter < 4.0 * bound + 0.02                            # within the depth grid's step of the bound
    assert abs(np.mean(errors)) < 0.1                              # the model's own bias (projector lever arm, interpolation)
    assert min(sigmas) >= scatter                                  # the reported sigma (fringe ambiguity included) is cautious


def test_sigma_covers_the_error_near_the_threshold():
    """Detection-level SNR at 12 km: where a depth is given, truth is within two sigma in at least 80% of pings and
    never further than one resolution cell of the delay in depth (10 m here)."""
    own, target, rng = 225.0, 150.0, 12000.0
    inside = total = 0
    for seed in range(40):
        d = detect(scenario(own, target, rng, ts=23.0, seed=1200 + seed), own, target, rng)
        if d["valid"]:
            total += 1
            inside += abs(d["depth"] - target) <= 2.0 * d["sigma"]
            assert abs(d["depth"] - target) < 12.0
    assert total >= 20 and inside >= 0.8 * total


# --- along a track, and with a stacked-row depth --------------------------------------------------------------------------

def test_a_track_adds_up_pings_that_are_each_too_weak():
    own, target, rng = 225.0, 150.0, 12000.0
    stage = core.MultipathDepth(mp.cpp_options(KNOWN))
    o = mp.cpp_options(KNOWN)
    cfg = mp.options(KNOWN)
    single = 0
    for seed in range(8):
        rec, origin = prepared(scenario(own, target, rng, ts=16.0, seed=1500 + seed, former=True))      # 3 dB below the 19 it was tuned at: source levels are RMS now (D-102), echoes 3 dB up; the 1 s sweep it was tuned with (D-256)
        slant = float(np.hypot(rng, target - own)) + 1.0
        d = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, o)
        single += bool(d["valid"])
        stage.add(7, d, mp.geometry_dict(mp.geometry(rec, origin, slant, cfg)))
    t = stage.track(7)
    assert single <= 3                                             # most pings give no depth on their own
    assert t["valid"] and t["method"] == "track" and t["arrivals"] == 8
    assert abs(t["depth"] - target) <= max(3.0 * t["sigma"], 2.0) and t["sigma"] < 6.0
    assert not stage.track(8)["valid"]
    # the Python reference's accumulator is the same sum
    ref = mp.TrackDepth(cfg)
    stage2 = core.MultipathDepth(mp.cpp_options(KNOWN))
    for seed in range(3):
        rec, origin = prepared(scenario(own, target, rng, ts=26.0, seed=1600 + seed, former=True))
        slant = float(np.hypot(rng, target - own)) + 1.0
        e = mp.estimate_detection(rec, origin, 0.0, slant, cfg)
        g = mp.geometry(rec, origin, slant, cfg)
        ref.add(e, g)
        stage2.add(1, {"grid": e.grid, "envelope": e.envelope}, mp.geometry_dict(g))
    a, b = ref.result(), stage2.track(1)
    assert b["depth"] == pytest.approx(a.depth, abs=1e-9) and b["sigma"] == pytest.approx(a.sigma, rel=1e-6) and b["gain"] == pytest.approx(a.gain, rel=1e-9)


def test_fuse_with_a_stacked_row_depth_and_pick_among_two_row_candidates():
    # agreeing: inverse-variance mean
    f = mp.fuse(150.0, 2.0, 156.0, 6.0)
    assert f["consistent"] and f["depth"] == pytest.approx((150.0 / 4 + 156.0 / 36) / (1 / 4 + 1 / 36)) and f["sigma"] == pytest.approx(1.0 / np.sqrt(1 / 4 + 1 / 36))
    # disagreeing: both stand, said
    f = mp.fuse(150.0, 2.0, 260.0, 6.0)
    assert not f["consistent"] and f["depth"] == 150.0 and f["sigma"] == 2.0
    # two rows: the elevation has several candidates (here 12 km x sin of lobes 1.9 degrees apart = 400 m apart in depth);
    # the multipath depth is far finer than their spacing and selects the true one
    f = mp.fuse(150.0, 2.0, 548.0, 30.0, alternatives=[148.0, -252.0])
    assert f["consistent"] and f["selected"] == 1 and abs(f["depth"] - 150.0) < 1.0
    for args in ((150.0, 2.0, 156.0, 6.0, [], 3.0), (150.0, 2.0, 548.0, 30.0, [148.0, -252.0], 3.0), (150.0, 2.0, 260.0, 6.0, [], 3.0)):
        a, b = mp.fuse(args[0], args[1], args[2], args[3], args[4], args[5]), core.live_multipath_fuse(*args)
        assert b["depth"] == pytest.approx(a["depth"]) and b["sigma"] == pytest.approx(a["sigma"]) and b["consistent"] == a["consistent"] and b["selected"] == a["selected"]


# --- through the chain, both engines --------------------------------------------------------------------------------------

def chain_scenario(range_m, ts_db, sonar_rows=1, pings=4, former=False):
    from gpusonar.config import load_scenario_config
    cfg = load_scenario_config({1: "popoto_12km", 2: "popoto_12km_2row"}[sonar_rows])
    cfg.pop("truth", None)
    cfg.pop("reverberation", None)
    cfg["fidelity"] = "image_source"
    cfg["propagation"] = {"surface_z": 0.0, "bottom_z": None, "max_order": 1}
    cfg["sonar_overrides"]["receive"]["window"] = {"range": [range_m - 500.0, range_m + 500.0]}
    cfg["platform"] = {"position": [0.0, 0.0, 225.0], "rpy": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}
    cfg["pings"] = {"count": pings, "interval": 18.0}
    cfg["scene"] = [{"type": "points", "label": "submerged", "positions": [[300.0, range_m, 150.0]], "target_strength_db": ts_db}]
    if former:
        cfg["sonar_overrides"]["transmit"] = dict(FORMER_SWEEP)
    return cfg


NEW = ("depth_multipath_m", "depth_multipath_sigma_m", "depth_multipath_refraction_sigma_m", "multipath_arrivals", "multipath_method", "multipath_gain", "multipath_reason",
       "multipath_pings", "multipath_ambiguous", "multipath_alternatives_m", "multipath_note", "depth_fused_m", "depth_fused_sigma_m",
       "depth_consistent", "elevation_selected")


def strip(report):
    out = copy.deepcopy(report)
    out.pop("processing_seconds", None)
    for item in out["detections"] + out["tracks"]:
        for key in NEW:
            item.pop(key, None)
    return out


@pytest.mark.parametrize("range_m, ts_db", [(6000.0, 15.0), (12000.0, 36.0)])
def test_chain_reports_depth_on_both_engines_and_changes_nothing_else(range_m, ts_db):
    """The face row at 225 m, a target at 150 m, the surface above: every ping's one detection carries the depth, the
    track carries the accumulated one, both engines give the same members, and with the key absent the report is the
    one the chain gave before (every other member identical with it present)."""
    records = run_scenario(chain_scenario(range_m, ts_db)).records
    base = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host"}
    got = {}
    for engine in ("cpp", "python"):
        with LiveChain(dict(base, engine=engine)) as chain:
            plain = chain.run([r.copy() for r in records])
        with LiveChain(dict(base, engine=engine, multipath_depth=dict(KNOWN))) as chain:
            got[engine] = chain.run([r.copy() for r in records])
        for a, b in zip(plain, got[engine]):
            assert not any(k in d for d in a["detections"] + a["tracks"] for k in NEW)
            assert strip(a) == strip(b), engine
        for rep in got[engine]:
            d = rep["detections"][0]
            assert d["depth_multipath_m"] == pytest.approx(150.0, abs=max(3.0 * d["depth_multipath_sigma_m"], 1.0)), (engine, d)
            assert d["multipath_arrivals"] == 4 and d["multipath_method"] == "delay"
        t = got[engine][-1]["tracks"][0]
        assert t["depth_multipath_m"] == pytest.approx(150.0, abs=max(3.0 * t["depth_multipath_sigma_m"], 2.0)) and t["multipath_pings"] >= 2
    for a, b in zip(got["cpp"], got["python"]):
        for x, y in zip(a["detections"] + a["tracks"], b["detections"] + b["tracks"]):
            assert {k for k in x if k in NEW} == {k for k in y if k in NEW}
            assert x["depth_multipath_m"] == pytest.approx(y["depth_multipath_m"], abs=0.05)
            assert x["depth_multipath_sigma_m"] == pytest.approx(y["depth_multipath_sigma_m"], rel=0.05, abs=0.02)
    # the option names of the standalone program are the ones the chain's options take
    from gpusonar import live
    o = live.cpp_chain_options(dict(base, multipath_depth={"min_gain": 9.0, "bottom": SAND, "water_depth": 100.0}))
    assert o["multipath_depth"]["min_gain"] == 9.0 and o["multipath_depth"]["bottom"] == 1 and o["multipath_depth"]["bottom_speed"] == 1650.0
    assert "multipath_depth" not in live.cpp_chain_options(base)


def test_two_rows_multipath_depth_selects_the_elevation():
    """Two stacked rows give several possible elevations; the depth from multipath timing picks the one that is true.
    6 km, array at 225 m, target at 150 m: the candidates are hundreds of metres apart in depth."""
    records = run_scenario(chain_scenario(6000.0, 18.0, sonar_rows=2, former=True)).records      # the 1 s sweep its figures were measured with (D-256)
    for engine in ("cpp", "python"):
        with LiveChain({"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "engine": engine, "elevation": {"correct_bearing": False},
                        "multipath_depth": dict(KNOWN)}) as chain:
            reps = chain.run([r.copy() for r in records])
        seen = 0
        for rep in reps:
            for d in rep["detections"]:
                if d.get("depth_multipath_m") is None or d.get("elevation_deg") is None:
                    continue
                assert d["depth_multipath_m"] == pytest.approx(150.0, abs=1.5)
                if d.get("depth_consistent"):
                    seen += 1
                    assert abs(d["depth_fused_m"] - 150.0) < 3.0 and d["depth_fused_sigma_m"] < 1.5     # the rows add their gradient prior, weighted (1.13 m with RMS source levels and coloured, coherent noise: D-102, D-103)
                    if "elevation_selected" in d:
                        cands = [d["elevation_deg"]] + d.get("elevation_alternatives_deg", [])
                        true = np.degrees(np.arcsin((225.0 - 150.0) / d["range_m"]))
                        assert abs(cands[d["elevation_selected"]] - true) == min(abs(c - true) for c in cands)
        assert seen >= 1, engine


# --- refracting water: Bellhop as the oracle ------------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("GPUSONAR_BELLHOP"), reason="Bellhop not available (GPUSONAR_BELLHOP)")
def test_depth_in_refracting_water_from_bellhop_arrivals():
    """Bellhop's eigenrays through a mixed layer over a thermocline, 1000 m of water whose bottom returns little: the
    flat, straight-ray pattern still finds the depth within the model error the ray code predicts for this water
    (tools/multipath_depth_study.py), which `model_sigma_fraction` carries into the sigma."""
    from gpusonar import raytrace
    depth = [0.0, 30.0, 70.0, 1000.0]
    speed = [1500.0, 1500.5, 1494.5, 1510.3]
    own, target, rng = 225.0, 150.0, 6000.0
    cfg = scenario(own, target, rng, ts=18.0, former=True)          # the 1 s sweep its figures were measured with (D-256)
    cfg["fidelity"] = "eigenray"
    cfg["propagation"] = {"environment": {"depth": 1000.0, "ssp_depth": depth, "ssp_speed": speed, "bottom_speed": 1480.0, "bottom_density": 1.2,
                                          "bottom_attenuation": 1.0},
                          "frequency": 5000.0, "max_range": 7000.0, "range_step": 50.0, "depth_step": 5.0, "max_arrivals": 8}
    prof = raytrace.Profile(depth, speed, 1000.0)
    tau = raytrace.surface_delay(prof, own, target, rng)
    assert tau is not None
    predicted = raytrace.flat_depth_from_delay(tau, own, rng, float(prof.c(own)))
    bias = abs(predicted - target) / target
    d = detect(cfg, own, target, rng, {"model_sigma_fraction": max(bias, 0.005)})
    assert d["valid"], d["reason"]
    assert abs(d["depth"] - target) <= 3.0 * d["sigma"] + 1.0
    assert abs(d["depth"] - predicted) <= 0.02 * target + 1.0          # the ray code says where the flat formula lands
