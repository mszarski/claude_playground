"""The sound-speed ladder for the depth estimate: the ray-traced depth against an independent
ray integration, the rule that turns a profile's uncertainty into a depth sigma (and its
limit, the gradient prior), the rungs and their order, the profile a vehicle builds as it
dives (bins, age and distance, extrapolation, persistence, a cast), a simulated dive
through a thermocline whose depth estimates improve as the profile grows, the formats of a
supplied profile, and the two chain engines reporting the same depths, sigmas and rung.
"""

import json
import pathlib

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import gpusonar as gs
from gpusonar import elevation, live
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
BASEBAND = {"kind": "baseband", "sample_rate": 8.0e3}
# An illustrative profile (defined here, not data): 30 m mixed layer, a thermocline of -0.15 /s for 40 m, pressure gradient below.
FAMILY = {"family": "mixed_layer", "family_gradient": 0.017, "layer_depth": 30.0, "thermocline_gradient": -0.15, "thermocline_thickness": 40.0}
DEPTH, SPEED = (np.array(v) for v in core.live_family_profile(FAMILY, 0.0, 1500.0, 400.0))


def integrate_ray(z0, elevation_up, travel_time, depth=DEPTH, speed=SPEED):
    """Depth and horizontal range after `travel_time` s along a ray leaving depth z0 at an elevation
    (rad, up positive): the ray equations in path length, by an adaptive Runge-Kutta integrator. Nothing
    in common with `ray_depth`, which steps by Snell's law. No boundaries (the cases below stay in the water)."""
    grad = np.diff(speed) / np.diff(depth)
    c_of = lambda z: np.interp(z, depth, speed)                                          # noqa: E731
    g_of = lambda z: grad[int(np.clip(np.searchsorted(depth, z, side="right") - 1, 0, len(grad) - 1))]    # noqa: E731

    def rhs(t, y):                              # y = (z down, theta up, r); d/dt = c d/ds
        z, th, _ = y
        c = c_of(z)
        return [-c * np.sin(th), np.cos(th) * g_of(z), c * np.cos(th)]

    sol = solve_ivp(rhs, (0.0, travel_time), [z0, elevation_up, 0.0], rtol=1e-10, atol=1e-9, max_step=travel_time / 4000.0)
    return float(sol.y[0, -1]), float(sol.y[2, -1])


def integrate_to_range(z0, elevation_up, horizontal_range, depth=DEPTH, speed=SPEED):
    """Depth and travel time where the same ray has covered `horizontal_range` m (the ray equations in range)."""
    grad = np.diff(speed) / np.diff(depth)
    c_of = lambda z: np.interp(z, depth, speed)                                          # noqa: E731
    g_of = lambda z: grad[int(np.clip(np.searchsorted(depth, z, side="right") - 1, 0, len(grad) - 1))]    # noqa: E731

    def rhs(r, y):                              # y = (z down, theta up, t)
        z, th, _ = y
        return [-np.tan(th), g_of(z) / c_of(z), 1.0 / (c_of(z) * np.cos(th))]

    sol = solve_ivp(rhs, (0.0, horizontal_range), [z0, elevation_up, 0.0], rtol=1e-9, atol=1e-9, max_step=2.0)
    return float(sol.y[0, -1]), float(sol.y[2, -1])


@pytest.mark.parametrize("z0,angle_deg,slant", [(50.0, 2.0, 800.0), (50.0, -3.0, 1500.0), (20.0, -1.0, 1000.0), (120.0, 4.0, 2500.0),
                                                 (60.0, -2.5, 2000.0), (50.0, 0.5, 600.0)])
def test_ray_traced_depth_agrees_with_an_independent_ray_integration(z0, angle_deg, slant):
    c0 = float(np.interp(z0, DEPTH, SPEED))
    want, _ = integrate_ray(z0, np.radians(angle_deg), slant / c0)
    assert 0.0 < want < 400.0                                              # the case stays clear of the boundaries
    py = elevation.ray_depth(z0, np.radians(angle_deg), slant, DEPTH, SPEED)
    cpp = core.live_ray_depth(z0, np.radians(angle_deg), slant, list(DEPTH), list(SPEED))
    assert py == pytest.approx(cpp, abs=1e-6)
    assert py == pytest.approx(want, abs=0.5)
    # The straight line is wrong by many times that through this profile.
    straight = z0 - slant * np.sin(np.radians(angle_deg))
    if slant >= 1500.0:
        assert abs(straight - want) > 5.0


def test_depth_sigma_of_a_profile_nobody_vouches_for_is_the_gradient_prior():
    """sigma(z) infinite: the tilt is the prior's gradient, and the depth sigma range^2 sigma_g / 2c."""
    depth, speed, no_claim = [0.0, 5000.0], [1500.0, 1500.0], [1e9, 1e9]
    for own, level, rng, g in ((60.0, np.radians(1.0), 1000.0, 0.05), (200.0, np.radians(-2.0), 3000.0, 0.02), (60.0, np.radians(0.5), 500.0, 0.05)):
        want = elevation.refraction_sigma(rng, 1500.0, g)
        py = elevation.profile_depth_sigma(own, level, rng, depth, speed, no_claim, g)
        cpp = core.live_profile_depth_sigma(own, level, rng, depth, speed, no_claim, g)
        assert py == pytest.approx(cpp, rel=1e-6, abs=1e-6)
        assert py == pytest.approx(want, rel=0.03, abs=0.05)
    # Both engines build the same tilted profiles, kinked where the rung's claim takes over from the prior.
    d, c, s = [0.0, 40.0, 80.0, 300.0], [1500.0, 1500.5, 1495.0, 1498.0], [0.3, 0.2, 0.2, 2.0]
    knots, fast, slow = elevation.tilted_profiles(60.0, d, c, s, 0.05)
    k2, f2, s2 = core.live_tilted_profiles(60.0, d, c, s, 0.05)
    assert np.allclose(knots, k2) and np.allclose(fast, f2) and np.allclose(slow, s2)
    assert 60.0 in knots and 56.0 in knots and 64.0 in knots               # 0.2 m/s is reached 4 m from the vehicle at 0.05 /s
    e = 0.5 * (np.asarray(fast) - np.asarray(slow))
    assert e[list(knots).index(60.0)] == 0.0                               # nothing is uncertain where the sound speed is measured
    assert np.all(np.abs(e) <= np.interp(knots, d, s) + 1e-12) and np.all(np.abs(e) <= 0.05 * np.abs(np.asarray(knots) - 60.0) + 1e-12)
    assert np.all(e[np.asarray(knots) < 60.0] <= 0.0) and np.all(e[np.asarray(knots) > 60.0] >= 0.0)      # one tilt, not a bulge
    # A profile that is vouched for gives a smaller depth sigma than the prior alone, never a larger one.
    tight = elevation.profile_depth_sigma(60.0, np.radians(-1.0), 2000.0, d, c, [0.1] * 4, 0.05)
    loose = elevation.profile_depth_sigma(60.0, np.radians(-1.0), 2000.0, d, c, [1e9] * 4, 0.05)
    assert 0.0 < tight < 0.5 * loose


def test_rungs_are_tried_in_order_and_named():
    kw = dict(time_ns=0, x=0.0, y=0.0, own_depth=50.0, sound_speed=1490.0, water_depth=200.0, gradient_sigma=0.05)
    lad = core.SoundSpeedLadder({"sample_sigma": 0.1})
    assert lad.evaluate(measured=False, **kw)["rung"] == "prior"
    local = lad.evaluate(measured=True, **kw)
    assert local["rung"] == "local" and local["depth"] == []               # no profile: the straight line and the prior's sigma
    fam = core.SoundSpeedLadder(dict(FAMILY, sample_sigma=0.1, family_sigma=0.3)).evaluate(measured=True, **kw)
    assert fam["rung"] == "family:mixed_layer" and fam["sigma"] == [0.3] * 4
    assert np.interp(50.0, fam["depth"], fam["speed"]) == pytest.approx(1490.0)       # hung on the sound speed measured at the vehicle
    assert np.allclose(np.diff(fam["speed"]) / np.diff(fam["depth"]), [0.017, -0.15, 0.017])
    # A built profile outranks the family once it spans enough depth; a supplied one outranks both.
    both = core.SoundSpeedLadder(dict(FAMILY, sample_sigma=0.1, min_span_m=6.0))
    for z in (48.0, 50.0, 52.0):
        both.add(0, 0.0, 0.0, z, 1490.0)
    assert both.evaluate(measured=True, **kw)["rung"] == "family:mixed_layer"          # 4 m sampled: not yet
    both.add(0, 0.0, 0.0, 55.0, 1489.0)
    built = both.evaluate(measured=True, **kw)
    assert built["rung"] == "built" and built["bins"] == 4 and (built["span_lo"], built["span_hi"]) == (48.0, 55.0)
    sup = both.evaluate(measured=True, supplied_depth=[0.0, 200.0], supplied_speed=[1500.0, 1503.4], **kw)
    assert sup["rung"] == "supplied" and sup["sigma"][0] > 1e6             # no sigma given for it: no claim, the prior decides
    # The order is the configuration's; a family first is an assumption preferred to the data, and says so.
    first = core.SoundSpeedLadder(dict(FAMILY, sample_sigma=0.1, ladder=["family", "built"]))
    assert first.evaluate(measured=True, **kw)["rung"] == "family:mixed_layer"
    with pytest.raises(ValueError):
        core.SoundSpeedLadder({"ladder": ["guess"]})
    # The built rung has no default for the sensor's accuracy; a family has none for its parameters.
    unset = core.SoundSpeedLadder({})
    for z in (40.0, 50.0):
        unset.add(0, 0.0, 0.0, z, 1490.0)
    with pytest.raises(ValueError, match="sample_sigma"):
        unset.evaluate(measured=True, **kw)
    with pytest.raises(ValueError, match="mixed_layer needs"):
        core.SoundSpeedLadder({"family": "mixed_layer", "ladder": ["family"]}).evaluate(measured=True, **kw)
    with pytest.raises(ValueError):
        elevation.profile_options({"built": {}, "famly": {}})


def test_built_profile_bins_ages_and_widens_outside_what_was_sampled(tmp_path):
    hour, km = int(3600e9), 1000.0
    opts = {"ladder": ["built"], "bin_m": 2.0, "sample_sigma": 0.1, "age_sigma_per_hour": 0.5, "distance_sigma_per_km": 0.4, "min_span_m": 5.0}
    lad = core.SoundSpeedLadder(dict(opts, file=str(tmp_path / "profile.txt")))
    # Three samples in one 2 m bin are one knot at their mean; bins come out in depth order.
    for z, c in ((30.2, 1500.0), (31.0, 1500.2), (31.6, 1500.4), (40.5, 1495.0), (20.1, 1500.1)):
        lad.add(0, 0.0, 0.0, z, c)
    assert lad.bins() == 3
    got = lad.evaluate(0, 0.0, 0.0, 30.0, 1500.2, True, 100.0, gradient=0.017, gradient_sigma=0.05)
    assert got["rung"] == "built" and got["bins"] == 3
    d, c, s = (np.array(got[k]) for k in ("depth", "speed", "sigma"))
    assert np.allclose(d[1:4], [20.1, (30.2 + 31.0 + 31.6) / 3, 40.5]) and c[2] == pytest.approx(1500.2)
    assert np.allclose(s[1:4], 0.1)                                         # fresh, here: the sensor's accuracy
    # Outside the sampled depths: the assumed gradient, and a sigma that grows by gradient_sigma per metre from the edge bin's.
    assert d[0] == 0.0 and c[0] == pytest.approx(1500.1 - 0.017 * 20.1) and s[0] == pytest.approx(0.1 + 0.05 * 20.1)
    assert d[-1] > 10000.0 and c[-1] == pytest.approx(1495.0 + 0.017 * (d[-1] - 40.5)) and s[-1] == pytest.approx(0.1 + 0.05 * (d[-1] - 40.5))
    wide = core.SoundSpeedLadder(dict(opts, extrapolation_sigma_per_s=0.2))
    for z in (20.0, 30.0, 40.0):
        wide.add(0, 0.0, 0.0, z, 1500.0)
    assert wide.evaluate(0, 0.0, 0.0, 30.0, 1500.0, True, 100.0)["sigma"][0] == pytest.approx(0.1 + 0.2 * 20.0)
    # Two hours and three kilometres later the same bins are worth less: in quadrature.
    later = lad.evaluate(2 * hour, 3 * km, 0.0, 30.0, 1500.2, True, 100.0)
    assert later["sigma"][2] == pytest.approx(np.sqrt(0.1**2 + (0.5 * 2) ** 2 + (0.4 * 3) ** 2))
    # A bin visited again is fresh again and follows the water.
    lad.add(2 * hour, 3 * km, 0.0, 31.0, 1499.0)
    again = lad.evaluate(2 * hour, 3 * km, 0.0, 30.0, 1499.0, True, 100.0)
    assert again["sigma"][2] == pytest.approx(0.1) and 1499.0 < again["speed"][2] < 1500.2
    # Bins too old or too far are left out, and with too little left the rung is not used.
    old = core.SoundSpeedLadder(dict(opts, max_age_s=3600.0, ladder=["built", "local"]))
    for z in (20.0, 30.0, 40.0):
        old.add(0, 0.0, 0.0, z, 1500.0)
    assert old.evaluate(hour // 2, 0.0, 0.0, 30.0, 1500.0, True)["rung"] == "built"
    assert old.evaluate(2 * hour, 0.0, 0.0, 30.0, 1500.0, True)["rung"] == "local"
    # A restart reads the file back; a cast (a static array's) is loaded once and does not age.
    reborn = core.SoundSpeedLadder(dict(opts, file=str(tmp_path / "profile.txt")))
    assert reborn.bins() == 3
    assert reborn.evaluate(2 * hour, 3 * km, 0.0, 30.0, 1499.0, True, 100.0)["speed"] == again["speed"]
    cast = tmp_path / "cast.csv"
    cast.write_text("depth_m,speed_m_s\n# taken at deployment\n0.5,1501.0\n10,1500.6\n25,1497.0\n60,1490.2\n")
    static = core.SoundSpeedLadder(dict(opts, cast=str(cast)))
    got = static.evaluate(50 * hour, 0.0, 0.0, 10.0, 1500.6, True, 80.0)
    assert got["rung"] == "built" and got["bins"] == 4 and np.allclose(got["sigma"][1:5], 0.1)
    assert np.allclose(got["depth"][1:5], [0.5, 10.0, 25.0, 60.0]) and np.allclose(got["speed"][1:5], [1501.0, 1500.6, 1497.0, 1490.2])


def test_a_vehicle_diving_through_a_thermocline_builds_a_profile_and_its_depths_improve():
    """Truth by the independent integrator: a target at 1 km seen from 20, 40 and 80 m as the vehicle dives. The
    arrival angle and travel time are exact; what differs is what the vehicle knows about the water."""
    lad = core.SoundSpeedLadder({"ladder": ["built", "local"], "sample_sigma": 0.1, "age_sigma_per_hour": 0.0, "distance_sigma_per_km": 0.0,
                                 "min_span_m": 5.0})
    g_sigma, water = 0.05, 400.0
    target_depth, angle_at = 60.0, {}
    errors, sigmas, rungs = {}, {}, {}
    sampled_to = 0.0
    for own in (20.0, 40.0, 80.0):
        # The dive so far: a sample every metre from the surface down to the vehicle.
        for z in np.arange(max(sampled_to, 1.0), own + 0.5, 1.0):
            lad.add(0, 0.0, 0.0, float(z), float(np.interp(z, DEPTH, SPEED)))
        sampled_to = own + 1.0
        c_own = float(np.interp(own, DEPTH, SPEED))
        # The ray from the vehicle that reaches the target's depth at 1 km: a scan for the first launch angle
        # (from steeply down) whose ray passes the target's depth, then bisection.
        scan = np.radians(np.arange(-12.0, 4.01, 0.5))
        deep = [integrate_to_range(own, a, 1000.0)[0] - target_depth for a in scan]
        k = next(i for i in range(len(scan) - 1) if deep[i] > 0.0 >= deep[i + 1])
        lo, hi = scan[k], scan[k + 1]
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if integrate_to_range(own, mid, 1000.0)[0] > target_depth:
                lo = mid                                                     # too deep: aim higher
            else:
                hi = mid
        angle = 0.5 * (lo + hi)
        z_end, t = integrate_to_range(own, angle, 1000.0)
        assert abs(z_end - target_depth) < 0.05
        slant = c_own * t
        got = lad.evaluate(0, 0.0, 0.0, own, c_own, True, water, gradient_sigma=g_sigma)
        rungs[own] = got["rung"]
        errors[own] = core.live_ray_depth(own, angle, slant, got["depth"], got["speed"], water) - target_depth
        sigmas[own] = core.live_profile_depth_sigma(own, angle, slant, got["depth"], got["speed"], got["sigma"], g_sigma, water)
        # What the rung below would have said: the straight line, with the prior's sigma.
        straight = own - slant * np.sin(angle) - target_depth
        angle_at[own] = (straight, elevation.refraction_sigma(slant, c_own, g_sigma))
    assert set(rungs.values()) == {"built"}
    # From 20 m the target is below everything sampled: the ray runs through extrapolated water, the depth is the straight
    # line's and its sigma the prior's.
    assert abs(errors[20.0]) > 10.0 and errors[20.0] == pytest.approx(angle_at[20.0][0], abs=0.01) and sigmas[20.0] > 0.9 * angle_at[20.0][1]
    # From 40 m the top of the thermocline is known and the rest of the way down is not: better, and less sure than it will be.
    assert abs(errors[40.0]) < 0.25 * abs(angle_at[40.0][0]) and sigmas[40.0] < 0.5 * angle_at[40.0][1]
    # From 80 m the target is above the vehicle, in water the dive went through: a depth good to centimetres where the
    # straight line is wrong by 30 m, and a sigma a tenth of the prior's.
    assert abs(errors[80.0]) < 0.5 and abs(angle_at[80.0][0]) > 25.0
    assert sigmas[80.0] < 0.1 * angle_at[80.0][1]
    assert abs(errors[80.0]) < abs(errors[40.0]) < abs(errors[20.0]) and sigmas[80.0] < sigmas[40.0] < sigmas[20.0]
    # What is left is inside what is reported (from 40 m only just: the thermocline below the vehicle, -0.15 /s, is three
    # times what the 0.05 /s prior allows for, and a sigma is no better than its prior).
    assert all(abs(errors[z]) <= 2.0 * sigmas[z] + 0.5 for z in errors)


def test_supplied_profile_formats(tmp_path):
    want = {"depth": [0.0, 30.0, 70.0, 200.0], "speed": [1500.0, 1500.5, 1494.5, 1496.7]}
    assert elevation.normalise_ssp(want) == want
    assert elevation.normalise_ssp([[30, 1500.5], [0, 1500], [200, 1496.7], [70.0, 1494.5]]) == want     # a table, in any order
    path = tmp_path / "profile.csv"
    path.write_text("depth_m, speed_m_s\n0, 1500\n30, 1500.5\n70; 1494.5   # the thermocline's foot\n200 1496.7\n")
    assert elevation.normalise_ssp({"file": str(path)}) == want and elevation.normalise_ssp(str(path)) == want
    assert elevation.normalise_ssp(None) is None
    with pytest.raises(ValueError):
        elevation.normalise_ssp({"depth": [0.0, 0.0], "speed": [1500.0, 1500.0]})
    o = live.cpp_chain_options({"elevation": {"ssp": {"file": str(path)}, "profile": {"supplied": {"sigma_m_s": 0.3},
                                                                                    "family": {"kind": "isothermal", "gradient_per_s": 0.017}}}})
    assert o["elevation"]["ssp_depth"] == want["depth"] and o["elevation"]["profile"]["supplied_sigma"] == 0.3
    assert o["elevation"]["profile"]["family"] == "isothermal" and o["elevation"]["profile"]["family_gradient"] == 0.017
    assert "profile" not in live.cpp_chain_options({"elevation": {}})["elevation"]


# --- through the chain: both engines --------------------------------------------------------------

CHAIN = {"bearings_deg": [-50.0, 50.0, 0.5], "range_cell_m": 2.0, "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 100.0},
         "min_snr_db": 15.0, "backend": "host", "tracker": {"range_sigma": 5.0, "bearing_sigma_deg": 1.0, "confirm_hits": 2, "max_misses": 2},
         "classify": {"min_speed": 1.0, "static_speed": 0.6, "max_fit_rms": 10.0, "multipath": None}}


@pytest.fixture(scope="module")
def diving_pings():
    """Four pings of a vehicle diving from 30 to 60 m in isovelocity water (the simulator's), with what navigation
    would have put into each record: the sound speed as a measurement and the samples taken on the way down."""
    sonar = SonarModel.load("popoto_long_range_3row", {"receive": dict(BASEBAND, window={"range": [900.0, 1100.0]}),
                                                       "transmit": {"waveform": {"duration": 1.0}}})
    target = [100.0, 1000.0, 45.0]
    recs = []
    for i, own in enumerate((30.0, 40.0, 50.0, 60.0)):
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, own)), Scene().add_points([target], target_strength_db=10.0),
                            environment={"sound_speed": 1500.0}, sequence=i, time_ns=int(20e9 * i), noise_rms=0.0)
        samples = [[-20.0 + 2.0 * k, 0.0, 0.0, own - 10.0 + k, 1500.0] for k in range(10)] if i else []
        live.apply_nav_facts(rec, {"sound_speed_measured": 1, "sound_speed_samples": samples})
        recs.append(rec)
    return recs, target


def run(engine, recs, elevation_cfg):
    with LiveChain(dict(CHAIN, engine=engine, elevation=elevation_cfg)) as chain:
        return chain.run([r.copy() for r in recs])


def test_both_engines_report_the_same_depth_sigma_and_rung_as_the_profile_grows(diving_pings):
    recs, target = diving_pings
    cfg = {"gradient_sigma_per_s": 0.05, "profile": {"built": {"sample_sigma_m_s": 0.1, "min_span_m": 15.0}}}
    py, cpp = run("python", recs, cfg), run("cpp", recs, cfg)
    for a, b in zip(py, cpp):
        assert a["profile"]["rung"] == b["profile"]["rung"]
        assert a["profile"].get("bins") == b["profile"].get("bins")
        da, db = (max(r["detections"], key=lambda d: d["snr_db"]) for r in (a, b))
        assert da["depth_profile_source"] == db["depth_profile_source"] == a["profile"]["rung"]
        assert da["depth_m"] == pytest.approx(db["depth_m"], abs=0.05)
        assert da["depth_sigma_m"] == pytest.approx(db["depth_sigma_m"], rel=0.03, abs=0.05)
        assert da["depth_refraction_sigma_m"] == pytest.approx(db["depth_refraction_sigma_m"], rel=0.03, abs=0.05)
        for ta, tb in zip(a["tracks"], b["tracks"]):
            assert ta.get("depth_profile_source") == tb.get("depth_profile_source") == a["profile"]["rung"]
            assert ta["depth_sigma_m"] == pytest.approx(tb["depth_sigma_m"], rel=0.03, abs=0.05)
    # The first ping knows the sound speed at the vehicle only; by the third the dive has spanned enough depth.
    assert [r["profile"]["rung"] for r in cpp] == ["local", "local", "built", "built"]
    assert cpp[-1]["profile"]["bins"] >= 15 and cpp[-1]["profile"]["span_m"][0] == pytest.approx(30.5, abs=1.1)
    best = [max(r["detections"], key=lambda d: d["snr_db"]) for r in cpp]
    assert all(d["depth_m"] == pytest.approx(target[2], abs=1.5) for d in best)     # the water is isovelocity and the profile says so
    # On the local rung the refraction sigma is the prior's; with the built profile it is what the profile leaves open,
    # which is less for a target at a depth the dive has sampled.
    prior = elevation.refraction_sigma(best[0]["range_m"], 1500.0, 0.05)
    assert best[0]["depth_refraction_sigma_m"] == pytest.approx(prior, rel=2e-3)
    assert best[-1]["depth_refraction_sigma_m"] < 0.8 * prior


def test_ladder_off_changes_nothing_and_the_lower_rungs_keep_the_old_numbers(diving_pings):
    recs, _ = diving_pings
    for engine in ("python", "cpp"):
        plain = run(engine, recs[:2], {"gradient_sigma_per_s": 0.05, "gradient_per_s": 0.017})
        laddered = run(engine, recs[:2], {"gradient_sigma_per_s": 0.05, "gradient_per_s": 0.017, "profile": {"ladder": ["local", "prior"]}})
        for a, b in zip(plain, laddered):
            assert "profile" not in a and b["profile"] == {"rung": "local"}
            assert all("depth_profile_source" not in d for d in a["detections"])
            for da, db in zip(a["detections"], b["detections"]):
                assert db.pop("depth_profile_source", "local") == "local"
                assert da == db                                              # the same depth, sigma and everything else
    # Without a measured sound speed the rung is the prior, and says so.
    bare = [r.copy() for r in recs[:1]]
    for r in bare:
        m = r.metadata
        m.extra_json = "{}"
        r.metadata = m
    for engine in ("python", "cpp"):
        assert run(engine, bare, {"profile": {}})[0]["profile"]["rung"] == "prior"
        assert run(engine, bare, {"profile": {"family": {"kind": "isothermal", "gradient_per_s": 0.017}}})[0]["profile"]["rung"] == "family:isothermal"
    with LiveChain(dict(CHAIN, engine="cpp", sound_speed=1500.0, elevation={"profile": {}})) as chain:
        assert chain.run(bare)[0]["profile"]["rung"] == "local"             # a sound speed set by hand counts as measured


def test_study_writes_the_ladder_numbers_the_plan_reads():
    path = ROOT / "docs" / "design" / "popoto_stage2.json"
    data = json.loads(path.read_text())
    lad = data["profile_ladder"]
    assert "illustrative" in lad["note"] and len(lad["profiles"]) >= 3
    s = lad["summary"]
    # Knowing the profile: metres. Not knowing it: tens of metres and more beyond a few kilometres.
    assert s["supplied: the true profile"]["6000"]["worst_abs_error_m"] < 3.0
    assert s["prior / local (straight line)"]["6000"]["median_abs_error_m"] > 20.0
    assert s["built: dived through the whole column"]["3000"]["median_abs_error_m"] < 2.0
