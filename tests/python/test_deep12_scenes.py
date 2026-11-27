"""The trade study's noise-limited site through the simulator (issue sonar-rctz; docs/tasks/DOPPLER.md "Deep site at 12 km"):

    popoto_12km_deep               the planning site deep_no_bottom_return (5 m/s, no bubble law), by name
    popoto_12km_deep_sea_state_5   the planning site deep_no_bottom_return_sea_state_5 (sea state 5, the apl_1994 law), by name;
                                   sea states 3 and 4 by --set environment.sea_state=N

What is held here, so that a measured margin can be set against the planner's (gpusonar.depthplan.plan, the trade study's
`predict_point`) with nothing but the chain between them:

- the scenes take their site from configs/builds/environments/illustrative.yaml (the same entries the planner plans in), and
  the site's bubble law reaches the simulator (time_spread.surface.bubble_loss_db) equal to the site's `bubble_model`;
- ambient noise: the simulator's spectrum at the band centre IS the planner's noise (Wenz at the same wind and shipping), at
  sea states 3, 4, 5 and the sites as stated (to 1e-9 dB); across the band the simulator's spectrum varies, the planner's
  does not: the band's mean power is within 0.3 dB of the centre's (the number is printed);
- absorption at the band centre IS the planner's (Ainslie-McColm at 12 C, 50 m); across the band it is not (printed);
- the surface: the rough surface's height and the bubble loss at one bounce are the planner's lloyd.Surface's;
- the paths: direct and surface image only (image_source, no bottom), the planner's "direct path and surface image";
- the geometry: the array at the own depth the depth plan recommends (docs/design/popoto_depth_planner.json), the targets'
  range rates as the header states;
- one thinned ping runs on both synthesis engines, equal to test_sampled_engine.py's tolerance.
"""

import copy
import json

import numpy as np
import pytest
import yaml

from gpusonar import acoustics, buildspec as bs, depthplan, environment, lloyd, target_classes
from gpusonar.config import config_dir, load_scenario_config
from gpusonar.sim import run_scenario
from gpusonar.sim import noise as noise_models
from gpusonar.sim import timespread
from gpusonar.sim.simulator import _noise_spectrum
from gpusonar.sonar import SonarModel

from test_sampled_engine import assert_same
from test_yaml_numbers import numberish_strings

DEEP, SS5 = "popoto_12km_deep", "popoto_12km_deep_sea_state_5"
SITE = {DEEP: "deep_no_bottom_return", SS5: "deep_no_bottom_return_sea_state_5"}
LABELS = ("closing", "oblique", "crossing", "ship", "point")
ROOT = config_dir().parent


def _site(name):
    (e,) = [e for e in environment.load_list("illustrative") if e["name"] == name]
    return e


def _cases():
    """(scene, overrides, the planning environment it must equal): each file as it is, and the sea-state-5 file at 3 and 4."""
    out = [(DEEP, None, _site(SITE[DEEP])), (SS5, None, _site(SITE[SS5]))]
    for n in (3, 4):
        e = {k: v for k, v in _site(SITE[SS5]).items() if k not in ("sea_state", "wind_speed_m_s")}
        out.append((SS5, {"environment": {"sea_state": n}}, {**e, "sea_state": n}))
    return out


def _budget():
    return bs.derive(bs.load("popoto_face_3x32"))


@pytest.mark.parametrize("name", [DEEP, SS5])
def test_the_scene_loads_and_takes_its_site_by_name(name):
    path = config_dir() / "scenarios" / f"{name}.yaml"
    assert list(numberish_strings(yaml.safe_load(path.read_text()))) == []
    raw = load_scenario_config(name, expand=False)
    assert raw["environment"]["base"] == SITE[name]                       # the planning site itself, not a copy of its values
    cfg = load_scenario_config(name)
    assert cfg["name"] == name and cfg["fidelity"] == "image_source" and [it["label"] for it in cfg["scene"]] == list(LABELS)
    assert cfg["sonar"] == "popoto_long_range_3row"
    env, site = cfg["environment"], _site(SITE[name])
    assert env["wind_speed"] == pytest.approx(depthplan.wind_of(site))
    assert env.get("bubble_model") == site.get("bubble_model")
    assert "water_depth" not in env and "site_water_depth" not in env and "profile" not in env     # no bottom, isovelocity
    assert cfg["propagation"].get("bottom_z") is None and cfg["propagation"]["max_order"] == 1
    # the site's bubble law reaches the simulator, at the site's wind
    surf = cfg["time_spread"]["surface"]
    assert (surf.get("bubble_loss_db") or {}).get("model") == site.get("bubble_model")
    assert surf["wind_speed"] == pytest.approx(env["wind_speed"])
    # the boundaries' reverberation by the one rule: the surface at the site's wind, no bottom
    assert cfg["reverberation"] == [{"model": "chapman_harris", "wind_speed": env["wind_speed"], "boundary_z": 0.0}]
    assert cfg["receiver"]["ambient_noise"] == {"model": "wenz", "wind_speed": env["wind_speed"], "shipping": 0.5}


def test_the_two_scenes_differ_only_in_their_site_and_its_bubble_law():
    a, b = load_scenario_config(DEEP, expand=False), load_scenario_config(SS5, expand=False)
    assert a["environment"] == {**b["environment"], "base": SITE[DEEP]}
    assert b["time_spread"] == {"surface": {"bubble_loss_db": {"model": "apl_1994"}}} and a["time_spread"] == {"surface": True}
    for c in (a, b):
        for k in ("name", "environment", "time_spread"):
            c.pop(k)
    assert a == b


@pytest.mark.parametrize("case", range(4))
def test_the_ambient_noise_is_the_planners_for_the_same_wind(case):
    """The planner's noise (buildspec.budget: Wenz at the band centre, the site's wind, shipping 0.5) and the simulator's spectrum
    for the scene at the same sea state: equal at the centre. Across 4-6 kHz the simulator's spectrum falls with frequency and the
    planner's is flat: the band's mean power is printed and held within 0.3 dB of the centre's."""
    name, over, site = _cases()[case]
    cfg = load_scenario_config(name, over)
    d = _budget()
    _, gains = bs.budget(d, depthplan.wind_of(site))
    spec = _noise_spectrum(cfg["receiver"]["ambient_noise"])
    fc = float(d["centre_hz"])
    assert float(spec(np.array([fc]))[0]) == pytest.approx(gains["noise_spectrum_db"], abs=1e-9)
    f = np.linspace(*d["band_hz"], 201)
    mean_db = 10.0 * np.log10(np.mean(10.0 ** (spec(f) / 10.0)))
    print(f"{name} {over}: wind {cfg['environment']['wind_speed']} m/s, noise at {fc:g} Hz {gains['noise_spectrum_db']:.2f} dB; "
          f"band mean {mean_db - gains['noise_spectrum_db']:+.3f} dB against the centre; at 4 / 6 kHz {spec(f[[0, -1]]) - gains['noise_spectrum_db']}")
    assert abs(mean_db - gains["noise_spectrum_db"]) < 0.3
    assert np.array_equal(spec(f), noise_models.wenz_spectrum_db(f, depthplan.wind_of(site), 0.5))


def test_the_absorption_at_the_band_centre_is_the_planners():
    cfg = load_scenario_config(DEEP)
    d = _budget()
    eq, _ = bs.budget(d, 5.0)
    fc = float(d["centre_hz"])
    got = environment.absorption_db_per_m(cfg["environment"], fc, cfg["absorption"])
    assert cfg["absorption"] == eq.absorption_model == "ainslie_mccolm"
    assert got == pytest.approx(eq.absorption_db_per_m(), rel=1e-12)
    # across the band (the planner takes the centre's): two-way at 12 km
    lo, hi = (2.0 * 12000.0 * environment.absorption_db_per_m(cfg["environment"], f, cfg["absorption"]) for f in d["band_hz"])
    print(f"two-way absorption at 12 km: {lo:.2f} dB at 4 kHz, {2 * 12000 * got:.2f} at 5 kHz, {hi:.2f} at 6 kHz")
    assert lo < 2 * 12000 * got < hi


@pytest.mark.parametrize("case", range(4))
def test_the_surface_is_the_planners(case):
    """The rough surface's rms height and the bubble loss at one bounce (5 kHz, grazing 0.5 to 5 degrees) as the planner's
    lloyd.Surface has them for the same site."""
    name, over, site = _cases()[case]
    cfg = load_scenario_config(name, over)
    planner = depthplan.surface_of(site)
    par = timespread.boundary_parameters(cfg["time_spread"]["surface"], surface=True, frequency=SonarModel.load(cfg["sonar"]).center_frequency)
    assert par["rms_height"] == pytest.approx(planner.height(), rel=1e-12)
    g = np.radians([0.5, 1.0, 2.0, 5.0])
    want = planner.bubble_loss_db(g, 5000.0)
    if site.get("bubble_model"):
        got = np.interp(g, *par["bubble_loss"])
        assert np.all(want > 0.0)
        np.testing.assert_allclose(got, want, rtol=0.02)                # the simulator interpolates a table of the same law
    else:
        assert "bubble_loss" not in par and np.all(want == 0.0)


def test_the_array_is_at_the_planners_own_depth_and_the_targets_as_stated():
    P = json.loads((ROOT / "docs" / "design" / "popoto_depth_planner.json").read_text())["builds"]["popoto_face_3x32"]["plans"]
    for name in (DEEP, SS5):
        cfg = load_scenario_config(name, expand=False)
        z = cfg["platform"]["position"][2]
        assert z == P[SITE[name]]["classes"]["ship"]["recommended"]["own_depth_m"] == 225.0
    assert P[SITE[DEEP]]["together"]["ship + large_submerged"]["best_own_depth_m"] == 225.0
    cfg = load_scenario_config(DEEP, expand=False)
    p0, vp = np.asarray(cfg["platform"]["position"], float), np.asarray(cfg["platform"]["velocity"], float)
    want = {"closing": (-15.0, -5.61), "oblique": (0.0, -0.50), "crossing": (15.0, -0.39), "ship": (29.17, -3.65), "point": (-30.0, 0.0)}
    classes = {c["name"]: c for c in target_classes.load("reference_classes")}
    for it in cfg["scene"]:
        c, v = np.mean(np.asarray(it["positions"], float), axis=0), np.asarray(it["velocity"], float)
        d = c - p0
        u = d / np.linalg.norm(d)
        b, r = want[it["label"]]
        assert np.degrees(np.arctan2(d[0], d[1])) == pytest.approx(b, abs=0.05) and (v - vp) @ u == pytest.approx(r, abs=0.01), it["label"]
        assert 11300.0 < np.linalg.norm(d) < 12700.0
        total = it["target_strength_db"] + 10.0 * np.log10(len(it["positions"]))
        cls = classes["ship" if it["label"] == "ship" else "large_submerged"]
        assert total == pytest.approx(min(np.atleast_1d(cls["target_strength_db"])), abs=0.02), it["label"]
        assert cls["depth_m"][0] <= c[2] <= cls["depth_m"][1], it["label"]
    base = load_scenario_config("popoto_12km_large_submerged", expand=False)
    assert [it for it in cfg["scene"] if it["label"] in ("closing", "oblique", "crossing")] == base["scene"]


@pytest.mark.parametrize("name, label", [(SS5, "point"), (DEEP, "ship")])
def test_a_deep_scene_runs_a_ping_on_both_engines(name, label):
    """Two hydrophones, one target, a window of +-60 m about it, the scene's surface (rough, with the bubble law), reverberation
    and noise: the C++ and NumPy engines agree to the tolerance of test_sampled_engine.py."""
    cfg = load_scenario_config(name, expand=False)
    cfg["pings"] = dict(cfg["pings"], count=1)
    cfg["scene"] = [it for it in cfg["scene"] if it["label"] == label]
    p = np.mean(np.asarray(cfg["scene"][0]["positions"], float), axis=0)
    r = float(np.linalg.norm(p - np.asarray(cfg["platform"]["position"], float)))
    cfg["sonar"] = "popoto_long_range"
    cfg["sonar_overrides"] = {"array": {"count": 2}, "receive": {"kind": "baseband", "sample_rate": 8.0e+3, "window": {"range": [r - 60.0, r + 60.0]}}}
    a = run_scenario(copy.deepcopy(cfg), engine="cpp").records[0]
    b = run_scenario(copy.deepcopy(cfg), engine="numpy").records[0]
    assert np.abs(a.samples.array).max() > 0
    assert_same(a.samples.array, b.samples.array, f"{name}: one ping, C++ vs NumPy")
