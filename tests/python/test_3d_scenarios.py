"""The 3-D scenarios of sonar-d7sw ("add some 3d scenarios"; docs/SCENARIOS.md, docs/tasks/PROPAGATION_3D.md): four
scenes at the opt-in `eigenray3d` tier (Bellhop3D), each with an `eigenray` twin (Bellhop 2-D, N x 2-D), and a
level-bed control.

    flank_seamount_3d_opening   / _nx2d    a target opening along the seamount's flank bearing, 1.5 to 6 km
    flank_bank_3d, flank_level_3d / flank_level_nx2d   a bed sloping 1 degree across the look direction; level control
    flank_ridge_3d              / _nx2d    a sill crossing the bearing at 45 degrees
    flank_canyon_3d             / _nx2d    a canyon along the look direction

Without an oracle: the new bed shape (`bathymetry.ridge`) and `profile_along`; every new file loads, expands and builds
its provider; each twin differs from its 3-D scenario only in the tier and the bed's spelling, and its profile is the
3-D bed along the target's bearing; what the cross-slope warning says about each bed. With Bellhop3D and Bellhop: each
scenario's headline number at its target, one source position (the array), with a measured bound.
"""

import os
import warnings

import numpy as np
import pytest

from gpusonar import environment
from gpusonar.config import load_scenario_config
from gpusonar.sim import bathymetry, find_bellhop, make_provider
from gpusonar.sim.bellhop3d import find_bellhop3d
from gpusonar.sim.reverb_relief import cross_slope_report
from gpusonar.sonar import SonarModel

PAIRS = {"flank_seamount_3d_opening": "flank_seamount_3d_opening_nx2d", "flank_bank_3d": "flank_level_nx2d",
         "flank_level_3d": "flank_level_nx2d", "flank_ridge_3d": "flank_ridge_nx2d", "flank_canyon_3d": "flank_canyon_nx2d"}
TIER_KEYS = ("name", "fidelity", "bathymetry", "propagation")


def expand(name):
    cfg = load_scenario_config(name, expand=False)
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))
    env = environment.resolve(cfg["environment"], strict=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*unconfirmed.*")
        warnings.filterwarnings("ignore", message=".*placeholder.*")
        return cfg, environment.apply_to_scenario(cfg, env, sonar), env


def target_points(cfg):
    """Every target's position at every ping: [(label, ping, xyz)]."""
    out = []
    n, dt = int(cfg["pings"]["count"]), float(cfg["pings"]["interval"])
    for item in cfg["scene"]:
        p0, v = np.asarray(item["positions"][0], float), np.asarray(item.get("velocity", (0.0, 0.0, 0.0)), float)
        out += [(item["label"], i, p0 + v * i * dt) for i in range(n)]
    return out


# --- the bed ----------------------------------------------------------------------------------------------------

def test_ridge_and_trough():
    bed = bathymetry.ridge(120.0, 60.0, (0.0, 2500.0), 45.0, 250.0)
    assert bed.depth(0.0, 2500.0) == pytest.approx(60.0)                                  # the crest
    assert bed.depth(1000.0, 3500.0) == pytest.approx(60.0)                              # along the axis (45 deg)
    assert bed.depth(0.0, 0.0) == pytest.approx(120.0, abs=1e-6)                         # far from it
    # 250 m off the axis: one sigma, exp(-1/2) of the height
    assert bed.depth(-250.0 / np.sqrt(2), 2500.0 + 250.0 / np.sqrt(2)) == pytest.approx(120.0 - 60.0 * np.exp(-0.5))
    # the analytic gradient against the numerical one, and its direction (normal to the axis)
    x, y = np.array([100.0, -300.0, 50.0]), np.array([2600.0, 2300.0, 2900.0])
    num = bathymetry.Bathymetry(bed.depth, step=0.01)
    assert np.allclose(np.stack(bed.gradient(x, y)), np.stack(num.gradient(x, y)), atol=1e-6)
    gx, gy = bed.gradient(x, y)
    assert np.allclose(gx * np.cos(np.radians(45.0)) + gy * np.sin(np.radians(45.0)), 0.0, atol=1e-12)
    trough = bathymetry.make_bathymetry({"ridge": {"depth": 100.0, "height": -150.0, "point": [0.0, 0.0], "azimuth_deg": 90.0,
                                                   "sigma": 400.0}})
    assert trough.depth(0.0, 5000.0) == pytest.approx(250.0) and trough.depth(3000.0, 0.0) == pytest.approx(100.0, abs=1e-6)
    assert bathymetry.ridge(100.0, 0.0, (0, 0), 0.0, 10.0).flat_depth == 100.0 and trough.flat_depth is None
    for bad in (dict(height=100.0, sigma=10.0), dict(height=10.0, sigma=0.0)):
        with pytest.raises(ValueError, match="ridge"):
            bathymetry.ridge(100.0, bad["height"], (0, 0), 0.0, bad["sigma"])


def test_profile_along_a_bearing():
    bed = bathymetry.seamount(200.0, 180.0, (300.0, 2500.0), 500.0)
    prof = bathymetry.profile_along(bed, (0.0, 0.0), 83.15, [0.0, 2517.9])              # through the summit
    assert prof[0] == [0.0, 200.0] and prof[1][1] == pytest.approx(20.0, abs=0.1)
    plane = bathymetry.plane(100.0, (0.02, 0.0))
    assert [d for _, d in bathymetry.profile_along(plane, (0.0, 0.0), 90.0, [0.0, 4000.0])] == [100.0, 100.0]   # along a contour
    assert bathymetry.profile_along(plane, (0.0, 0.0), 0.0, [1000.0])[0][1] == pytest.approx(120.0)


# --- the files ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(set(PAIRS) | set(PAIRS.values())))
def test_each_file_loads_and_builds_its_provider(name):
    typed, cfg, env = expand(name)
    provider = make_provider(cfg["fidelity"], env["sound_speed"], cfg["propagation"])          # builds; runs nothing
    assert provider.tier == typed["fidelity"]
    assert "reverberation" not in typed                                                    # target in noise only (the tier's limit)
    if typed["fidelity"] == "eigenray3d":
        assert cfg["propagation"]["bathymetry"] == typed["bathymetry"]
        x, y = typed["platform"]["position"][:2]
        assert provider.bed.depth(x, y) == pytest.approx(typed["environment"]["water_depth_m"], abs=0.01)   # the bed agrees with the site
    # every target lies in the water, above the bed under it
    bed = bathymetry.make_bathymetry(typed["bathymetry"]) if "bathymetry" in typed else None
    for _, _, p in target_points(typed):
        floor = bed.depth(p[0], p[1]) if bed is not None else cfg["propagation"]["environment"]["depth"]
        assert 0.0 < p[2] < floor - 10.0


@pytest.mark.parametrize("name3d", sorted(PAIRS))
def test_the_twin_differs_only_in_the_tier(name3d):
    """Same sonar, window, environment, platform, pings, targets, noise and seed; the same lookup grid; the twin's bed
    is the 3-D bed along the target's bearing (N x 2-D on that bearing), except for the bank's off-broadside targets,
    which no one profile can serve (docs/SCENARIOS.md)."""
    a, cfg3, _ = expand(name3d)
    b, cfg2, _ = expand(PAIRS[name3d])
    assert (a["fidelity"], b["fidelity"]) == ("eigenray3d", "eigenray")
    assert {k: v for k, v in a.items() if k not in TIER_KEYS} == {k: v for k, v in b.items() if k not in TIER_KEYS}
    for key in ("range_step", "depth_step"):
        assert a["propagation"][key] == b["propagation"][key]
    bed = bathymetry.make_bathymetry(a["bathymetry"])
    origin = np.asarray(a["platform"]["position"][:2], float)
    ocean = cfg2["propagation"]["environment"]
    for label, _, p in target_points(a):
        # the bearing of the node the 3-D provider looks the target up at (its bearing_step, 0.5 degree)
        bearing = 0.5 * round(2.0 * np.degrees(np.arctan2(p[1] - origin[1], p[0] - origin[0])))
        if name3d == "flank_bank_3d" and label != "broadside":
            continue
        if ocean.get("bathymetry") is None:                                               # a level twin: the bed is level on the bearing
            r = np.linspace(0.0, cfg2["propagation"]["max_range"], 200)
            assert np.allclose([d for _, d in bathymetry.profile_along(bed, origin, bearing, r)], ocean["depth"], atol=0.05)
        else:
            prof = np.asarray(ocean["bathymetry"], float)
            assert prof[-1, 0] >= cfg2["propagation"]["max_range"]
            ours = bathymetry.profile_along(bed, origin, bearing, prof[:, 0])
            assert np.allclose([d for _, d in ours], prof[:, 1], atol=0.051)


def test_what_the_cross_slope_warning_says_about_each_bed():
    """The warning (P1) reads the slope across each bearing ON that bearing: it flags the seamount's flank bearing and
    the oblique sill, not the 1 degree bank (0.35 degree a bounce at 10 degrees grazing, under its 1.8) and NOT the canyon's
    axis, where the slope across the bearing is zero although the walls are 400 m away (a case it cannot see)."""
    flagged = {}
    for name3d in ("flank_seamount_3d_opening", "flank_bank_3d", "flank_ridge_3d", "flank_canyon_3d"):
        typed = load_scenario_config(name3d, expand=False)
        bed = bathymetry.make_bathymetry(typed["bathymetry"])
        origin = np.asarray(typed["platform"]["position"][:2], float)
        last = [p for label, _, p in target_points(typed) if label in ("target", "broadside")][-1]
        bearing = np.arctan2(last[1] - origin[1], last[0] - origin[0])
        flagged[name3d] = bool(cross_slope_report(bed, origin, [bearing], float(np.hypot(*(last[:2] - origin))))["flagged"][0])
    assert flagged == {"flank_seamount_3d_opening": True, "flank_bank_3d": False, "flank_ridge_3d": True, "flank_canyon_3d": False}


# --- with the oracles ------------------------------------------------------------------------------------------
#
# Each test builds the scenario's own providers (its `propagation`, its bed) and asks for the eigenrays of every target
# position from the array (one source position: the projector's leg, 0.5 m lower, is the same to within the bounds and
# doubles the time). "N x 2-D" is Bellhop3D's own N x 2-D mode ('2', incoherent TL, the same hat-beam fan and bed) at
# the same node, which separates the bed's horizontal refraction from the two programs' differences; "twin" is the
# scenario's `eigenray` twin (Bellhop 2-D over the bearing's profile, our N x 2-D tier). Measured 2026-11-26 on the
# shared 4-core CPU (docs/SCENARIOS.md, docs/tasks/PROPAGATION_3D.md).

needs_oracles = pytest.mark.skipif(find_bellhop3d() is None or find_bellhop() is None,
                                   reason="Bellhop3D and Bellhop not installed (GPUSONAR_BELLHOP3D, GPUSONAR_BELLHOP)")
slow = pytest.mark.skipif(os.environ.get("GPUSONAR_SLOW_ORACLES", "") in ("", "0"),
                          reason="slow 3-D scenario check (2 to 3 minutes): set GPUSONAR_SLOW_ORACLES=1 to run it")


def _nx2d_tl_db(provider, source, points):
    """Bellhop3D N x 2-D mode, incoherent TL (dB re 1 m, negative), at the 3-D provider's node of each point."""
    import tempfile
    from gpusonar.sim.bellhop3d import _read_tl, _run, write_bty3d, write_env3d
    key = provider._key(np.asarray(source, float))
    it, iz, ir = provider._nodes(np.array(key), np.atleast_2d(points))
    out = []
    for a, b, c in zip(it, iz, ir):
        bearing, (sx, sy, sz) = a * provider.bearing_step, key
        half = 1.05 * provider.ranges[c] + 300.0
        step = provider.grid_step if provider.bed.flat_depth is None else max(provider.grid_step, 500.0)
        g = np.arange(np.floor(-half / step), np.ceil(half / step) + 1) * step
        bed = lambda x, y: provider.bed.depth(x + sx, y + sy)                         # noqa: E731
        deepest = float(max(provider.env.depth, np.max(bed(g[None, :], g[:, None]))))
        z, cc = provider.env.profile()
        if deepest > z[-1]:
            z, cc = np.r_[z, deepest], np.r_[cc, cc[-1]]
        with tempfile.TemporaryDirectory(prefix="gpusonar_nx2d_") as tmp:
            write_bty3d(os.path.join(tmp, "run.bty"), bed, g, g)
            write_env3d(os.path.join(tmp, "run.env"), frequency=provider.frequency, profile=(z, cc), depth=deepest,
                        bottom=(provider.env.bottom_speed, provider.env.bottom_density, provider.env.bottom_attenuation),
                        source_depth=sz, receiver_depths=[provider.depths[b]], ranges=[provider.ranges[c]], bearings_deg=[bearing],
                        run_type="I", mode="2", n_alpha=provider.n_alpha, alpha_deg=provider.angles, n_beta=provider.n_beta,
                        beta_deg=(bearing - 0.5 * provider.azimuth_span, bearing + 0.5 * provider.azimuth_span),
                        box_km=(half / 1000.0, half / 1000.0), beam="G")
            _run(find_bellhop3d(), tmp, 3600.0)
            out.append(20.0 * np.log10(float(_read_tl(os.path.join(tmp, "run.shd"))[0, 0, 0])))
    return np.array(out)


def _measure(name3d):
    """Per target position: one-way level (dB re 1 m) at the 3-D tier, Bellhop3D N x 2-D and the twin; the arrivals'
    energy-weighted mean and largest (within 20 dB of the strongest) azimuth off the true bearing at the array (deg,
    positive clockwise seen from above, i.e. from north towards east)."""
    a, cfg3, env = expand(name3d)
    _, cfg2, _ = expand(PAIRS[name3d])
    p3 = make_provider("eigenray3d", env["sound_speed"], cfg3["propagation"])
    p2 = make_provider("eigenray", env["sound_speed"], cfg2["propagation"])
    src = np.asarray(a["platform"]["position"], float)
    pts = np.array([p for _, _, p in target_points(a)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        e3, e2 = p3.eigenrays(src, pts), p2.eigenrays(src, pts)
    level = lambda es: 10.0 * np.log10(np.sum(np.where(es.valid, np.abs(es.amplitude) ** 2, 0.0), axis=1))   # noqa: E731
    off_mean, off_max = [], []
    for i, p in enumerate(pts):
        v = e3.valid[i]
        la, w = e3.launch[i][v], np.abs(e3.amplitude[i][v]) ** 2
        off = np.degrees(np.angle(np.exp(1j * (np.arctan2(la[:, 1], la[:, 0]) - np.arctan2(p[1] - src[1], p[0] - src[0])))))
        off_mean.append(float(np.sum(off * w) / np.sum(w)))
        off_max.append(float(np.abs(off[w > 0.01 * w.max()]).max()))
    return dict(d3=level(e3), nx2d=_nx2d_tl_db(p3, src, pts), twin=level(e2), off_mean=np.array(off_mean),
                off_max=np.array(off_max), runs=p3.runs)


@needs_oracles
def test_level_bed_control_the_tier_adds_nothing():
    """flank_level_3d against flank_level_nx2d: three targets at 5 km over a level 100 m bed. Measured: the 3-D tier,
    Bellhop3D's N x 2-D and the twin within 0.005 dB one-way; arrivals off their bearing by under 0.001 degree. Bound
    0.2 dB (D-197). About 40 s (two Bellhop3D runs of 17 s)."""
    m = _measure("flank_level_3d")
    assert np.all(np.abs(m["d3"] - m["twin"]) < 0.2) and np.all(np.abs(m["d3"] - m["nx2d"]) < 0.2)
    assert np.all(np.abs(m["off_max"]) < 0.05)


@needs_oracles
@slow
def test_bank_turns_paths_but_does_not_move_the_level():
    """flank_bank_3d: 1 degree across the look direction. Measured (one-way, 5 km): 3-D minus Bellhop3D's N x 2-D
    +0.02, +0.01, -0.00 dB (north, broadside, south): no level effect. The arrivals come from 0.41 to 0.48 degree
    up-slope of the true bearing (energy-weighted; the formula's 0.17 degree a bounce at 5 degrees grazing, more for the
    steeper paths) and up to 3.0 to 3.3 degrees for the strong steep ones; the beamformed peak moves 0.015 degree, under
    a twentieth of the 3.6 degree beam. Against the level twin the off-broadside targets differ by -0.20 and +0.21 dB:
    the slope ALONG their bearings, which the twin's one profile does not have. About 2.3 minutes."""
    m = _measure("flank_bank_3d")                                   # north, broadside, south
    assert np.all(np.abs(m["d3"] - m["nx2d"]) < 0.3)
    assert abs(m["d3"][1] - m["twin"][1]) < 0.3
    # positive: clockwise of the bearing, i.e. south of east: up-slope (the bed deepens to the north)
    assert np.all((0.2 < m["off_mean"]) & (m["off_mean"] < 0.8)) and np.all(m["off_max"] < 4.0)


@needs_oracles
@slow
def test_seamount_opening_the_level_falls_behind_the_flank():
    """flank_seamount_3d_opening: the target at 1.5, 3, 4.5 and 6 km on the flank bearing. Measured one-way, 3-D minus
    Bellhop3D's N x 2-D: -0.00, -0.18, -3.98, -2.33 dB; minus the twin (our 2-D tier): +0.07, +0.71, -2.65, -2.06 dB
    (the twin's own arrivals over a piecewise-linear bed are louder than Bellhop's TL there: PROPAGATION_3D.md
    section 2). About 3 minutes (two Bellhop3D runs of 90 s here; the scenario runs four)."""
    m = _measure("flank_seamount_3d_opening")
    effect = m["d3"] - m["nx2d"]
    assert abs(effect[0]) < 0.3                                     # before the seamount: N x 2-D is the whole answer
    assert effect[2] < -3.0 and effect[3] < -1.5                    # behind the flank
    assert (m["d3"] - m["twin"])[2] < -2.0


@needs_oracles
@slow
def test_oblique_sill_3d_adds_little_and_the_twin_is_louder_for_its_own_reasons():
    """flank_ridge_3d: a sill crossing the bearing at 45 degrees, target 5 km behind it. Measured one-way: 3-D minus
    Bellhop3D's N x 2-D -0.20 dB (the bed's horizontal refraction is small here although the warning flags the
    bearing: arrivals 0.20 degree off on average, at most 0.65); 3-D minus the twin -1.86 dB, which is the twin's own
    arrivals over its piecewise-linear profile, louder than Bellhop's TL (docs/tasks/PROPAGATION_3D.md section 2, item
    4), not a 3-D effect. A characterisation, as test_seamount_3d.py's. About 2.2 minutes."""
    m = _measure("flank_ridge_3d")
    assert abs(m["d3"][0] - m["nx2d"][0]) < 0.6
    assert -3.0 < m["d3"][0] - m["twin"][0] < -1.0
    assert np.all(m["off_max"] < 2.0)


@needs_oracles
@slow
def test_canyon_channels_the_field_along_its_axis():
    """flank_canyon_3d: a target 5 km down a canyon's axis. Measured one-way: 3-D minus Bellhop3D's N x 2-D +6.16 dB
    (the walls turn paths back towards the axis: arrivals up to 13.9 degrees off the bearing); the twin (level 250 m)
    equals Bellhop3D's N x 2-D within 0.01 dB. The cross-slope warning does not flag the axis bearing (no slope across
    it ON the bearing). The provider rescaled this receiver's merged arrivals by up to 10.4 dB to Bellhop3D's TL: the
    level is the TL run's, the split between arrivals is uncertain (docs/SCENARIOS.md). About 2 minutes."""
    m = _measure("flank_canyon_3d")
    assert m["d3"][0] - m["nx2d"][0] > 4.0
    assert abs(m["twin"][0] - m["nx2d"][0]) < 0.3
    assert m["off_max"][0] > 5.0
