"""Rays through the ladder's profile inside the two depth estimators (libs/live raytrace.hpp, multipath_rays.cpp; chain
keys multipath_depth.rays, elevation.ray_method, elevation.profile.ensemble): the rays' pattern against the flat one in
isovelocity water, the profile's sigma against the first-order rule, the explicit reasons where a path does not exist,
truth from an independent ray integration (tools/refraction_depth_study.py) and from Bellhop-simulated pings, and the two
chain engines reporting the same members with everything else unchanged."""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import multipath_depth as mp
from gpusonar.live import LiveChain
from gpusonar.sim import run_scenario

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import test_multipath_depth as T  # noqa: E402

core = gs.core
ISO = {"depth": [0.0, 5000.0], "speed": [1500.0, 1500.0], "sigma": [0.0, 0.0], "gradient_sigma": 0.0, "rung": "supplied"}
DEEP = {"depth": [0.0, 30.0, 70.0, 1000.0], "speed": [1500.0, 1500.5, 1494.5, 1510.3]}           # the deep illustrative profile
RAYS = dict(mp.cpp_options({}), rays=1)


def test_rays_in_isovelocity_water_are_the_flat_pattern_and_the_flat_depth():
    g = mp.geometry_dict(T.GEOM)
    for rng in (1000.0, 3000.0, 12000.0):
        for z in (40.0, 150.0, 400.0):
            for dt in (0.0, 0.002):
                flat = core.live_multipath_pattern(z, 2.0 * rng / 1500.0 + dt, dict(g, range_m=rng), mp.cpp_options({}))
                rays = core.live_multipath_pattern(z, 2.0 * rng / 1500.0 + dt, dict(g, range_m=rng, profile=ISO), RAYS)
                assert np.allclose(flat[0], rays[0], atol=5e-8)                 # delays: a twentieth of a microsecond
                assert np.allclose(flat[1], rays[1], atol=2e-3)
    rec, origin = T.prepared(T.scenario(225.0, 150.0, 12000.0, ts=28.0))
    slant = float(np.hypot(12000.0, 75.0)) + 1.0
    flat = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, mp.cpp_options(T.KNOWN))
    rays = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, dict(mp.cpp_options(T.KNOWN), rays=1), profile=ISO)
    none = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, dict(mp.cpp_options(T.KNOWN), rays=1))      # no profile handed over: the prior
    assert flat["valid"] and rays["valid"] and none["valid"]
    assert rays["depth"] == pytest.approx(150.0, abs=1.0) and rays["depth"] == pytest.approx(flat["depth"], abs=0.5)
    assert rays["profile_source"] == "supplied" and none["profile_source"] == "prior" and none["depth"] == pytest.approx(rays["depth"], abs=1e-3)
    assert rays["refraction_sigma"] == 0.0 and np.isnan(rays["shadow"][0])                   # the profile is taken as exact; no shadow in this water


def test_the_profiles_sigma_is_the_first_order_rule_under_the_prior():
    """Nothing known of the water: the tilt of the prior through the rays is z r^2 sigma_g / (2 c (z_own + z))."""
    for rng, z, rel in ((1000.0, 150.0, 0.25), (3000.0, 150.0, 0.12), (3000.0, 40.0, 0.12), (6000.0, 150.0, 0.12), (12000.0, 150.0, 0.12)):
        g = dict(mp.geometry_dict(T.GEOM), range_m=rng, profile=dict(ISO, sigma=None, gradient_sigma=0.005))
        got = core.live_multipath_rays(g, RAYS, [z])[0]
        rule = z * rng**2 * 0.005 / (2.0 * 1500.0 * (225.0 + z))        # first order in depth over range: looser at 1 km
        assert got["paths"] == 3 and got["sigma"] == pytest.approx(rule, rel=rel), (rng, z, got["sigma"], rule)


def test_no_direct_path_is_said_and_no_depth_is_given():
    """A vehicle at 50 m in a strong thermocline has no direct path to anything at 6 km: the stage says so instead of
    fitting what arrives; in the same water at 1 km it gives the depth of a direct-path target and names the shadow."""
    from refraction_depth_study import STRONG, family
    depth, speed = family(STRONG, 300.0)
    water = {"depth": list(depth), "speed": list(speed), "sigma": [0.1] * len(depth), "gradient_sigma": 0.05, "rung": "built"}
    opts = dict(mp.cpp_options({"water_depth": 200.0, "max_depth_m": 200.0}), rays=1)
    rec, origin = T.prepared(T.scenario(50.0, 120.0, 6000.0, ts=25.0))
    far = core.live_multipath_detection(rec, tuple(origin), 0.0, float(np.hypot(6000.0, 70.0)) + 1.0, opts, profile=water)
    assert not far["valid"] and far["reason"] == "no_direct_path" and np.isnan(far["depth"]) and far["profile_source"] == "built"
    flat = core.live_multipath_detection(rec, tuple(origin), 0.0, float(np.hypot(6000.0, 70.0)) + 1.0, mp.cpp_options({"water_depth": 200.0, "max_depth_m": 200.0}))
    assert flat["valid"]                                 # what the flat pattern did with the same ping: a depth
    info = core.live_multipath_rays(dict(mp.geometry_dict(T.GEOM), range_m=6000.0, array_depth=50.0, projector_depth=50.0, water_depth=200.0, profile=water), opts,
                                    [float(z) for z in np.linspace(1.0, 199.0, 100)])
    assert not any(i["paths"] & 1 for i in info)


def test_depth_against_the_independent_integration_by_rung():
    """One water of the study (strong thermocline over 1000 m, vehicle at 225 m, target at 150 m, 3 km): through the true
    profile and through one built on a dive both estimators give the true depth; through nothing but the vehicle's own
    sound speed they are tens of metres off and their sigma says so."""
    import refraction_depth_study as S
    name, p, water, own, _ = S.WATERS[4]
    depth, speed = S.family(p, water + 100.0)
    truth = S.true_paths(depth, speed, own, 150.0, water)[3000.0]
    assert truth["direct"] is not None and truth["surface"] is not None
    c_own = float(np.interp(own, depth, speed))
    slant, tau, angle = c_own * truth["direct"][0], 2.0 * (truth["surface"][0] - truth["direct"][0]), truth["direct"][1]
    rungs = S.rungs_of(depth, speed, own, water)
    for label, tol in (("supplied: the true profile", 0.3), ("built: dived through the whole column", 0.5)):
        z, sigma, _, reason = S.multipath_through(rungs[label], own, slant, tau, water)
        assert reason == "" and abs(z - 150.0) < tol and sigma < 5.0, (label, z, sigma)
        assert abs(core.ray_depth_arcs(own, angle, slant, rungs[label]["depth"], rungs[label]["speed"], water) - 150.0) < tol
    w = rungs["prior / local: nothing but the sound speed at the vehicle"]
    z, sigma, ens, _ = S.multipath_through(w, own, slant, tau, water)
    assert 2.0 < abs(z - 150.0) <= 2.0 * sigma and len(ens) >= 8          # off, and owned up to
    ze = core.ray_depth_arcs(own, angle, slant, w["depth"], w["speed"], water)
    se = core.live_profile_depth_sigma(own, angle, slant, w["depth"], w["speed"], [1e9] * len(w["depth"]), S.GRADIENT_SIGMA, water, True)
    assert 10.0 < abs(ze - 150.0) <= 2.0 * se
    assert abs(z - 150.0) < abs(ze - 150.0)                               # the multipath depth is the less sensitive of the two


@pytest.mark.skipif(not os.environ.get("GPUSONAR_BELLHOP"), reason="Bellhop not available (GPUSONAR_BELLHOP)")
def test_bellhop_ping_through_the_true_profile_and_through_none():
    """A Bellhop-simulated ping at 6 km through the deep profile (target at 150 m): the flat pattern is 19 m off; the
    rays through the true profile are within their sigma; through no profile the answer is the flat one and its sigma
    (the prior's) covers the error."""
    own, target, rng = 225.0, 150.0, 6000.0
    cfg = T.scenario(own, target, rng, ts=18.0)
    cfg["fidelity"] = "eigenray"
    cfg["propagation"] = {"environment": {"depth": 1000.0, "ssp_depth": DEEP["depth"], "ssp_speed": DEEP["speed"], "bottom_speed": 1480.0, "bottom_density": 1.2,
                                          "bottom_attenuation": 1.0},
                          "frequency": 5000.0, "max_range": 7000.0, "range_step": 50.0, "depth_step": 5.0, "max_arrivals": 8}
    rec, origin = T.prepared(cfg)
    slant = float(np.hypot(rng, target - own)) + 1.0
    opts = dict(mp.cpp_options({"water_depth": 1000.0}), rays=1)
    flat = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, mp.cpp_options(dict(T.KNOWN, water_depth=1000.0)))
    true = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, opts, profile=dict(DEEP, sigma=[0.1] * 4, gradient_sigma=0.05, rung="supplied"))
    none = core.live_multipath_detection(rec, tuple(origin), 0.0, slant, opts)
    assert flat["valid"] and abs(flat["depth"] - target) > 10.0
    assert true["valid"] and abs(true["depth"] - target) < max(2.0 * true["sigma"], 4.0) and true["sigma"] < 10.0
    assert none["valid"] and abs(none["depth"] - target) <= 2.0 * none["sigma"] and none["sigma"] > 50.0


NEW = T.NEW + ("multipath_profile_source", "multipath_searched_m", "multipath_shadow_m", "depth_multipath_ensemble_m", "depth_multipath_ensemble_complete",
               "depth_ensemble_m")


def strip(report):
    out = copy.deepcopy(report)
    for key in ("processing_seconds", "coverage", "profile"):
        out.pop(key, None)
    for item in out["detections"] + out["tracks"]:
        for key in NEW:
            item.pop(key, None)
    return out


def test_one_row_chain_with_rays_on_both_engines_changes_nothing_else():
    """The face row at 225 m, a target at 150 m at 6 km under a flat surface, the water supplied as a profile with the
    ladder on: both engines report the depth through the rays with the rung it rests on; every other member of every
    line is what the chain gave without the keys."""
    records = run_scenario(T.chain_scenario(6000.0, 15.0)).records
    base = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host"}
    water = {"ssp": {"depth": [0.0, 2000.0], "speed": [1500.0, 1500.0]}, "profile": {"supplied": {"sigma_m_s": 0.1}}}
    got = {}
    for engine in ("cpp", "python"):
        with LiveChain(dict(base, engine=engine)) as chain:
            plain = chain.run([r.copy() for r in records])
        with LiveChain(dict(base, engine=engine, elevation=water, multipath_depth={"rays": True})) as chain:
            got[engine] = chain.run([r.copy() for r in records])
        for a, b in zip(plain, got[engine]):
            assert strip(a) == strip(b), engine
            assert b["profile"] == {"rung": "supplied"} and "profile" not in a
        for rep in got[engine]:
            d = rep["detections"][0]
            assert d["multipath_profile_source"] == "supplied" and "multipath_shadow_m" not in d
            assert d["depth_multipath_m"] == pytest.approx(150.0, abs=max(3.0 * d["depth_multipath_sigma_m"], 1.0)), (engine, d)
            assert 0.0 < d["depth_multipath_refraction_sigma_m"] < 5.0                # 0.1 m/s over the column, through the rays
        t = got[engine][-1]["tracks"][0]
        assert t["depth_multipath_m"] == pytest.approx(150.0, abs=max(3.0 * t["depth_multipath_sigma_m"], 2.0)) and t["multipath_profile_source"] == "supplied"
    for a, b in zip(got["cpp"], got["python"]):
        for x, y in zip(a["detections"] + a["tracks"], b["detections"] + b["tracks"]):
            assert {k for k in x if k in NEW} == {k for k in y if k in NEW}
            assert x["depth_multipath_m"] == pytest.approx(y["depth_multipath_m"], abs=0.05)
            assert x["depth_multipath_sigma_m"] == pytest.approx(y["depth_multipath_sigma_m"], rel=0.05, abs=0.02)


def test_two_rows_arcs_and_ensemble_on_both_engines():
    """Two stacked rows at 6 km with an assumed gradient: the depth by the arcs is the stepped ray's within 2 m and the
    rest of the line is unchanged; with the ensemble every depth also carries the span over its profiles; the engines agree."""
    records = run_scenario(T.chain_scenario(6000.0, 18.0, sonar_rows=2, pings=2)).records
    base = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host"}
    el = {"correct_bearing": False, "gradient_per_s": 0.017, "profile": {"ladder": ["local", "prior"]}}
    ens = {"gradients_per_s": [-0.05, 0.017], "extent_m": 100.0}
    out = {}
    for engine in ("cpp", "python"):
        runs = {}
        for key, cfg in (("steps", el), ("arcs", dict(el, ray_method="arcs")), ("ensemble", dict(el, ray_method="arcs", profile=dict(el["profile"], ensemble=ens)))):
            with LiveChain(dict(base, engine=engine, elevation=cfg, multipath_depth={"rays": True} if key == "ensemble" else None)) as chain:
                runs[key] = chain.run([r.copy() for r in records])
        seen = 0
        for a, b, c in zip(runs["steps"], runs["arcs"], runs["ensemble"]):
            for x, y, z in zip(a["detections"], b["detections"], c["detections"]):
                if x.get("depth_m") is None:
                    continue
                seen += 1
                assert y["depth_m"] == pytest.approx(x["depth_m"], abs=2.0)
                assert {k: v for k, v in x.items() if k not in ("depth_m", "depth_sigma_m", "depth_refraction_sigma_m")} == \
                       {k: v for k, v in y.items() if k not in ("depth_m", "depth_sigma_m", "depth_refraction_sigma_m")}
                lo, hi = z["depth_ensemble_m"]
                assert lo <= z["depth_m"] <= hi and hi - lo > 1.0
                if z.get("depth_multipath_m") is not None:
                    assert z["depth_multipath_ensemble_m"][0] <= z["depth_multipath_m"] <= z["depth_multipath_ensemble_m"][1]
        assert seen >= 1, engine
        out[engine] = runs["ensemble"]
    for a, b in zip(out["cpp"], out["python"]):
        for x, y in zip(a["detections"], b["detections"]):
            assert ("depth_ensemble_m" in x) == ("depth_ensemble_m" in y)
            if "depth_ensemble_m" in x:
                assert x["depth_ensemble_m"] == pytest.approx(y["depth_ensemble_m"], abs=0.5)


# --- leftovers: the bubble law by name, the standalone program, SBL bearings -----------------------------------------------

def test_bubble_loss_comes_from_the_named_law():
    """`surface.bubble_model` evaluates sim/surface.py's law (wind speed and frequency given) into a loss against grazing
    angle for the pattern, flat or through rays."""
    from gpusonar.sim import surface as surface_model
    o = mp.cpp_options({"surface": {"bubble_model": "apl_1994", "wind_speed_m_s": 9.8, "frequency_hz": 5000.0}})
    grazing, loss = surface_model.bubble_loss_table("apl_1994", 9.8, 5000.0, 1500.0)
    assert np.allclose(o["surface_loss_grazing"], grazing) and np.allclose(o["surface_loss_table_db"][5:], loss[5:])
    with pytest.raises(ValueError):
        mp.cpp_options({"surface": {"bubble_model": "apl_1994"}})
    g = mp.geometry_dict(T.GEOM)
    calm = core.live_multipath_pattern(150.0, 16.0, g, mp.cpp_options({}))[1]
    rough = core.live_multipath_pattern(150.0, 16.0, g, o)[1]
    at = float(np.interp(np.arcsin(375.0 / 12000.0), o["surface_loss_grazing"], o["surface_loss_table_db"]))
    assert at > 0.0 and abs(rough[1]) == pytest.approx(abs(calm[1]) * 10.0 ** (-at / 20.0), rel=0.02)
    rays = core.live_multipath_pattern(150.0, 16.0, dict(g, profile=ISO), dict(o, rays=1))[1]
    assert np.allclose(rays, rough, atol=5e-3)


EXE = Path(__file__).resolve().parents[2] / "build" / "bin" / "sonar_live"


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")
def test_the_program_takes_the_new_keys(tmp_path):
    """build/bin/sonar_live replays a recording with multipath_depth_rays, the ladder and coverage given as key=value: it
    ends with return code 0 and prints what the Python-wrapped C++ engine prints."""
    import json
    import subprocess

    from gpusonar import buildspec as bs
    from gpusonar import coverage
    d = bs.derive(bs.load("popoto_face_1x32"))
    cov = {"classes": [{"name": "sub", "ts_db": 15.0, "depth_m": [100.0, 200.0]}], "budget": coverage.budget_of(d, 5.0)}
    records = run_scenario(T.chain_scenario(6000.0, 15.0, pings=2)).records
    cfg = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "engine": "cpp", "multipath_depth": {"rays": True}, "coverage": cov,
           "elevation": {"ssp": {"depth": [0.0, 2000.0], "speed": [1500.0, 1500.0]}, "profile": {"supplied": {"sigma_m_s": 0.1}}}}
    with LiveChain(cfg, recording=str(tmp_path / "rec.h5")) as chain:
        want = chain.run([r.copy() for r in records])
    snr, pd = coverage.pd_table(records[0], {k: v for k, v in cfg.items() if k != "engine"})
    flat = coverage.cpp_options(cov)
    args = ["bearing_first_deg=-60", "bearing_last_deg=60", "bearing_step_deg=0.5", "multipath_depth=1", "multipath_depth_rays=1",
            "elevation_ssp_depth=0,2000", "elevation_ssp_speed=1500,1500", "elevation_profile=1", "elevation_profile_supplied_sigma=0.1", "coverage=1",
            "coverage_pd_snr_db=" + ",".join(repr(v) for v in snr), "coverage_pd_value=" + ",".join(repr(v) for v in pd)]
    args += [f"coverage_{k}={v}" for k, v in flat.items()]
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), *args], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    got = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    assert len(got) == len(want) == 2
    for a, b in zip(got, want):
        assert a["coverage"] == b["coverage"] and a["profile"] == b["profile"] == {"rung": "supplied"}
        assert a["detections"][0]["depth_multipath_m"] == pytest.approx(b["detections"][0]["depth_multipath_m"], abs=1e-6)
        assert a["detections"][0]["multipath_profile_source"] == "supplied"


def test_with_sbl_bearings_the_stage_still_gives_the_depth():
    """doa: sbl replaces the map's bearings by SBL's at each range gate (the C++ chain runs it for either engine): the
    detection of the target still carries the depth through the rays."""
    records = run_scenario(T.chain_scenario(6000.0, 15.0, pings=2)).records
    with LiveChain({"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "doa": "sbl", "multipath_depth": {"rays": True}}) as chain:
        reps = chain.run([r.copy() for r in records])
    depths = [d["depth_multipath_m"] for rep in reps for d in rep["detections"] if d.get("depth_multipath_m") is not None]
    assert depths and min(abs(z - 150.0) for z in depths) < 2.0, [d for rep in reps for d in rep["detections"]]
