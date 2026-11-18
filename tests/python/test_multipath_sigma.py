"""The depth sigma of stacked rows where boundary echoes share the range cell (elevation.multipath_sigma).

A target's echo returns by every pair of paths. A pair and its mirror have one length, so they add and the rows read
the mean of the two elevations: the sea surface (direct with surface) or the sea bed (direct with bottom), whatever the
target's depth. `libs/live` `multipath_depth_limit` (elevation.hpp) says what that does to the depth; both chain engines
call it. Checked here:

1. one resolved echo: the term is the closed form of the pull of the far echoes, and small;
2. a reading that is a boundary: the depths that can give it are the ones geometry says, and the term is tens of metres;
3. a reading no echo gives: nothing is claimed;
4. without a water depth the report says what stands in for it;
5. the two-row face at 3 km in 100 m of water (the generator's `rehearsal`): the truth is 37 to 49 m from the reported
   depth; without the key the sigma is 11 to 15 m, with it the truth is within two sigma on every ping, in both engines,
   and nothing else in the line moves;
6. three rows at 1 km: resolved, the term stays under the angle's own sigma.
"""
import importlib.util
import pathlib
import sys
import warnings

import numpy as np
import pytest

from gpusonar import _gpusonar as core
from gpusonar import buildspec as bs
from gpusonar import config

ROOT = pathlib.Path(__file__).resolve().parents[2]
C, BAND, FC, HEIGHT = 1500.0, 2000.0, 5000.0, 1.5        # the face build: 4 to 6 kHz, rows 1.5 m apart
RESOLUTION, APERTURE = 0.5 * C / BAND, (C / FC) / HEIGHT


def limit(own, slant, elevation, sigma_deg, reported, water=0.0, bounces=2, max_depth=500.0):
    return core.elevation_multipath_limit([0.0, 5000.0], [C, C], water, own, slant, elevation, np.radians(sigma_deg), reported, RESOLUTION, APERTURE, True, 8.0, 3.0,
                                          bounces, max_depth)


@pytest.mark.parametrize("own, depth, slant, sigma_deg", [(200.0, 260.0, 3000.0, 0.05), (100.0, 150.0, 2000.0, 0.05), (60.0, 90.0, 1500.0, 0.1)])
def test_one_resolved_echo_gains_only_the_pull_of_the_far_ones(own, depth, slant, sigma_deg):
    """Deep water, surface only, one reflection a leg: three echoes. The direct one is first and alone in its cell; the
    surface pair (twice the pressure, at the mean elevation) is x cells later and the twice-reflected one 2 x. The rows
    are pulled by each by (a / a0)^2 E2 d^2 / 2, E2 = 1 / (2 pi^2 x^2), d no more than lambda / (2 pi D)."""
    el = np.arcsin((own - depth) / slant)
    got = limit(own, slant, el, sigma_deg, depth, bounces=1)
    flat = np.sqrt(slant**2 - (depth - own) ** 2)
    excess = np.hypot(flat, own + depth) - slant
    surface = np.arctan2(own + depth, flat)
    cap = APERTURE / (2.0 * np.pi)
    pull2 = 0.0
    for pressure, cells, away in ((2.0, 0.5 * excess / RESOLUTION, 0.5 * (surface - el)), (1.0, excess / RESOLUTION, surface - el)):
        assert cells > 5.0                                    # resolved by a wide margin
        pull2 += 0.5 * pressure**2 / (2.0 * np.pi**2 * cells**2) * min(abs(away), cap) ** 2
    own_sigma = slant * np.cos(el) * np.radians(sigma_deg)
    assert got["explained"] and not got["limited"] and got["echoes"] == 1
    assert got["extra"] == pytest.approx(slant * np.cos(el) * np.sqrt(pull2), rel=0.05)
    assert got["extra"] < 0.7 * own_sigma and got["depth_sigma"] == pytest.approx(np.hypot(own_sigma, got["extra"]))
    assert got["depth_lo"] < depth < got["depth_hi"] and got["depth_hi"] - got["depth_lo"] < 12.0 * np.hypot(own_sigma, got["extra"])     # three sigma each way, on a grid of depths


def test_a_reading_at_a_boundary_says_only_which_part_of_the_water_the_target_is_in():
    """30 m deep in 100 m of water, 3 km: the bottom pair reads -atan(70 / 3000) and the surface pair +atan(30 / 3000),
    for any target depth at which that pair is in the clump the stage takes. Geometry: the bottom pair is within a
    resolution cell of the direct echo below z = 100 - 0.375 x 3000 / 70 = 84 m, and from 57 m down it is in one clump
    with the surface pair that the direct echo is more than 8 dB under; the surface pair is in the first clump for every
    depth down to the bottom's reach."""
    own, water, slant = 30.0, 100.0, 3000.0
    bottom = limit(own, slant, -np.arctan2(water - own, slant), 0.2, 99.0, water)
    assert bottom["explained"] and bottom["limited"] and bottom["bottom"]
    assert 50.0 < bottom["depth_lo"] < 60.0 and bottom["depth_hi"] == pytest.approx(water)       # a target at 60 m reads the sea bed: 39 m of error
    assert 15.0 < bottom["extra"] < 30.0
    surface = limit(own, slant, np.arctan2(own, slant), 0.2, 0.5, water)
    assert surface["limited"] and surface["depth_lo"] == pytest.approx(0.0) and surface["depth_hi"] > 85.0
    assert 35.0 < surface["extra"] < 58.0                 # no more than a uniform column about its end, 100 / sqrt(3)
    # the same target at 300 m: every echo is resolved, and the same code says so
    near = limit(own, 300.0, np.arcsin((own - 60.0) / 300.0), 0.2, 60.0, water)
    assert not near["limited"] and near["extra"] < 0.3 and near["echoes"] == 1


def test_a_reading_no_echo_gives_is_not_claimed_and_no_water_depth_is_said():
    steep = limit(30.0, 3000.0, np.radians(10.0), 0.2, 20.0, 100.0)       # no path of two bounces arrives from 10 degrees up at 3 km in 100 m
    assert not steep["explained"] and steep["extra"] == 0.0 and '"depth_unresolved_sigma_m": null' in steep["members"]
    open_water = limit(30.0, 1000.0, np.arcsin((30.0 - 60.0) / 1000.0), 0.2, 60.0)
    assert not open_water["bottom"] and open_water["deepest"] == 500.0
    assert '"depth_unresolved_assumed": "water depth unknown: no bottom echoes, surface echoes only, target anywhere down to 500 m"' in open_water["members"]
    known = limit(30.0, 1000.0, np.arcsin((30.0 - 60.0) / 1000.0), 0.2, 60.0, 100.0)
    assert known["bottom"] and "assumed" not in known["members"] and '"depth_multipath_limited": false' in known["members"]


def generator():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("build_from_spec", ROOT / "tools" / "build_from_spec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rehearsal(gen, build, tmp_path):
    """The build's `rehearsal` trial as the generator's self-test runs it: records, the chain's settings, the truth."""
    from gpusonar.sim import run_scenario
    spec = bs.load(build)
    gen.generate(spec, tmp_path, diagrams=False)
    config.add_config_dir(tmp_path / "configs")
    try:
        sc = bs.scenario_configs(spec)["rehearsal"]
        truth = sc.pop("truth")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            records = run_scenario(sc).records
    finally:
        config.remove_config_dir(tmp_path / "configs")
    chain = bs.chain_config(spec, sync=bool(truth.get("sync")))
    chain["tracker"] = dict(chain.get("tracker") or {}, confirm_hits=2)
    chain["elevation"] = dict(chain["elevation"], gradient_per_s=None, gradient_sigma_per_s=0.0, water_depth=truth["water_depth"])
    return records, chain, truth


def test_two_rows_at_three_kilometres_say_that_they_read_a_boundary(tmp_path):
    gen = generator()
    records, chain, truth = rehearsal(gen, "popoto_face_2x32", tmp_path)
    target = truth["targets"][0]
    assert chain["elevation"]["multipath_sigma"] == {}                      # the generator switches it on
    without = dict(chain, elevation={k: v for k, v in chain["elevation"].items() if k != "multipath_sigma"})
    before = gen._run(records, without, "cpp")
    old = [gen._check_depth(r, target, truth["water_depth"]) for r in before]
    assert [round(c["depth_m"]) for c in old] == [98, 97, 11] and target[2] == 60.0      # the sea bed twice, then near the surface
    assert all(c["depth_error_sigmas"] > 2.5 and c["depth_sigma_m"] < 15.0 and not c["honest"] for c in old)      # the defect: 11 to 15 m stated, 37 to 49 m wrong
    after = {engine: gen._run(records, chain, engine) for engine in ("python", "cpp")}
    for engine, reports in after.items():
        new = [gen._check_depth(r, target, truth["water_depth"]) for r in reports]
        assert all(c["honest"] and c["multipath_limited"] and c["depth_error_sigmas"] < 1.7 for c in new), engine
        assert all(19.0 < c["multipath_sigma_m"] < 41.0 for c in new), engine
        for b, r in zip(before, reports):                                    # nothing else in the line moves
            d0, d1 = gen._nearest(b, target)[0], gen._nearest(r, target)[0]
            assert all(d1[k] == pytest.approx(d0[k], rel=1e-4, abs=1e-4) for k in ("range_m", "snr_db", "elevation_deg", "elevation_sigma_deg", "depth_m")), engine
            assert d1["depth_sigma_m"] == pytest.approx(np.hypot(d0["depth_sigma_m"], d1["depth_unresolved_sigma_m"]), rel=1e-4)
            assert d1["depth_unresolved_span_m"][0] <= target[2] <= d1["depth_unresolved_span_m"][1]
    for a, b in zip(after["python"], after["cpp"]):                          # one function, two engines
        da, db = gen._nearest(a, target)[0], gen._nearest(b, target)[0]
        for key in ("depth_sigma_m", "depth_unresolved_sigma_m", "depth_unresolved_echoes", "depth_multipath_limited"):
            assert da[key] == pytest.approx(db[key], rel=1e-4), key        # 1e-4: on a CUDA build the two engines hand the function elevations that differ in single precision
        assert da["depth_unresolved_span_m"] == pytest.approx(db["depth_unresolved_span_m"])
    # a track's depth: the same boundary ping after ping is not averaged away
    tracks = [t for t in after["cpp"][-1]["tracks"] if t.get("depth_m") is not None]
    tracks0 = [t for t in before[-1]["tracks"] if t.get("depth_m") is not None]
    assert tracks and len(tracks) == len(tracks0)
    for t, t0 in zip(tracks, tracks0):
        assert t["depth_m"] == pytest.approx(t0["depth_m"]) and t["depth_sigma_m"] >= t0["depth_sigma_m"]
        if abs(t["range_m"] - 3000.0) < 20.0:
            assert abs(t["depth_m"] - target[2]) <= 2.0 * t["depth_sigma_m"] < 2.0 * 60.0 and abs(t["depth_m"] - target[2]) > 2.0 * t0["depth_sigma_m"]


def test_three_rows_at_one_kilometre_resolve_the_target_and_the_term_stays_small(tmp_path):
    gen = generator()
    records, chain, truth = rehearsal(gen, "popoto_face_3x32", tmp_path)
    target = truth["targets"][0]
    for c in (gen._check_depth(r, target, truth["water_depth"]) for r in gen._run(records, chain, "cpp")):
        assert c["honest"] and not c["multipath_limited"] and not c["ambiguous"]
        assert abs(c["depth_error_m"]) < 3.0 and c["multipath_sigma_m"] < 2.5 and c["depth_sigma_m"] < 5.0       # 3.5 m of it is the 0.2 degree roll sigma at 1 km
