"""The coverage picture (libs/live coverage.hpp, python/gpusonar/coverage.py, tools/coverage_map.py, chain.coverage):
its two-way factor against gpusonar.lloyd in isovelocity water (flat and rough surface, a bottom), against Bellhop in the
stage-2 study's refracting waters, its levels against buildspec.budget, its detection probability against the detector's
own helper, the summary against a direct evaluation of its grid, the refresh rule, "could I have seen it", the two chain
engines printing the same member with every other member unchanged, and the tool as a program."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import buildspec as bs
from gpusonar import coverage, live, lloyd
from gpusonar.live import LiveChain
from gpusonar.sim import make_pose, run_scenario
from gpusonar.sonar import SonarModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_multipath_depth as T  # noqa: E402

core = gs.core
ROOT = Path(__file__).resolve().parents[2]
SPEC = bs.load("popoto_face_1x32")
D = bs.derive(SPEC)
SONAR = {"f_centre": float(D["centre_hz"]), "bandwidth": float(D["band_hz"][1] - D["band_hz"][0]), "pulse": float(D["pulse_s"])}
CLASSES = coverage.classes_of(bs.target_classes_of(SPEC, range_m=12000.0))
CFG = {"classes": CLASSES, "budget": coverage.budget_of(D, 5.0)}
ISO = coverage.water([0.0, 5000.0], [1500.0, 1500.0])
RANGES = np.linspace(1000.0, 13000.0, 25)
SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}


def test_factor_is_lloyds_in_isovelocity_water():
    """Direct path and surface image, straight rays: the two-way factor is gpusonar.lloyd's peak figure, nulls and lobes
    included, for a shallow and a deep sonar and targets from 4 to 300 m."""
    zt = np.array([4.0, 40.0, 150.0, 300.0])
    for own in (30.0, 225.0):
        f = coverage.field(ISO, None, own, zt, RANGES, CFG, SONAR)
        ref = lloyd.two_way(D["band_hz"][0], D["band_hz"][1], own, zt[:, None], RANGES[None, :])["peak_db"]
        assert np.max(np.abs(f["factor_db"] - ref)) < 0.3 and np.median(np.abs(f["factor_db"] - ref)) < 0.1
        assert f["direct"].all()
    # with the projector 4 m below the array the two legs differ (the face build's lever arm)
    f = coverage.field(ISO, None, 30.0, zt, RANGES, CFG, SONAR, projector_below_m=4.0)
    ref = lloyd.two_way(D["band_hz"][0], D["band_hz"][1], 30.0, zt[:, None], RANGES[None, :], projector_below_m=4.0)["peak_db"]
    assert np.max(np.abs(f["factor_db"] - ref)) < 0.6


def test_rough_surface_and_bubbles_lie_between_lloyds_coherent_peak_and_its_energy():
    """Sea state 5 with the apl_1994 bubble law evaluated from sim/surface.py: lloyd's peak leaves the scattered energy
    out and its energy figure counts all of it whenever it arrives; ours adds what is scattered as power under the peak."""
    surf = lloyd.Surface.from_config({"wind_speed": 9.8, "bubble_model": "apl_1994"})
    cfg = dict(CFG, surface={"rms_height_m": surf.height(), "bubble_model": "apl_1994", "wind_speed_m_s": 9.8, "frequency_hz": SONAR["f_centre"]})
    zt = np.array([4.0, 40.0, 150.0])
    f = coverage.field(ISO, None, 225.0, zt, RANGES, cfg, SONAR)["factor_db"]
    ref = lloyd.two_way(D["band_hz"][0], D["band_hz"][1], 225.0, zt[:, None], RANGES[None, :], surface=surf)
    assert np.all(f >= ref["peak_db"] - 0.3) and np.all(f <= np.maximum(ref["energy_db"], ref["peak_db"]) + 3.1)
    assert np.median(np.abs(f - ref["peak_db"])) < 1.0


def test_bottom_reflections_are_the_image_series_with_the_named_sediments_coefficient():
    """100 m of isovelocity water over sand: the one-way sum of powers over the paths with at most two boundary
    reflections is the image series' to the same order (Rayleigh coefficient at each image's grazing angle); the bottom
    by name resolves through configs/sediments.yaml as lloyd does."""
    w = coverage.water([0.0, 100.0], [1500.0, 1500.0])
    cfg = dict(CFG, bottom=SAND, rays={"max_bounces": 2, "max_arrivals": 12, "fan": 1440})
    zt, own = np.array([30.0, 70.0]), 50.0
    got = coverage.field(w, 100.0, own, zt, RANGES, cfg, SONAR)["incoherent_db"]
    _, s = lloyd.one_way([SONAR["f_centre"]], own, zt[:, None], RANGES[None, :], layer=lloyd.Layer(depth=100.0, bottom=SAND, max_order=2))
    h_parts = []
    from gpusonar.sim.propagation import rayleigh_reflection
    for img, ns, nb in lloyd._images(zt[:, None], 100.0, 2):
        r_d, r_j = np.hypot(RANGES[None, :], own - zt[:, None]), np.hypot(RANGES[None, :], own - img)
        grazing = np.arcsin(np.abs(own - img) / r_j)
        h_parts.append((r_d / r_j) ** 2 * np.abs(rayleigh_reflection(grazing, 1500.0, **SAND)) ** (2 * nb))
    assert np.max(np.abs(got - 10.0 * np.log10(sum(h_parts)))) < 0.3
    named = coverage.halfspace({"sediment": "medium_sand"}) if "medium_sand" in open(ROOT / "configs" / "sediments.yaml").read() else None
    if named is not None:
        assert named["speed"] > 1500.0 and coverage.cpp_options(dict(CFG, bottom={"sediment": "medium_sand"}))["bottom_speed"] == pytest.approx(named["speed"])


@pytest.mark.skipif(not os.environ.get("GPUSONAR_BELLHOP"), reason="Bellhop not available (GPUSONAR_BELLHOP)")
def test_loss_and_shadows_against_bellhop_in_the_stage2_waters():
    """Bellhop (a separate program) as the oracle in refracting water: where it has a path that touches no boundary so
    have we, and the reverse, on nine in ten points of each water; the one-way sum of powers agrees within 3 dB on
    the median point and within 6 dB on four in five (geometric beams against Gaussian ones differ at caustics and
    shadow edges)."""
    from gpusonar.sim.bellhop import OceanEnvironment, run_bellhop
    from test_raytrace_cpp import PROFILES
    ranges = [1000.0, 2000.0, 3000.0, 4500.0, 6000.0, 9000.0, 12000.0]
    for name in ("isothermal", "strong thermocline", "weak thermocline", "no mixed layer"):
        d, c, water, own = PROFILES[name]
        zs = [20.0, 60.0, 120.0, 170.0]
        keep = [i for i, z in enumerate(d) if z < water]
        env = OceanEnvironment(depth=water, ssp_depth=[d[i] for i in keep] + [water], ssp_speed=[c[i] for i in keep] + [float(np.interp(water, d, c))],
                               bottom_speed=SAND["speed"], bottom_density=SAND["density"], bottom_attenuation=SAND["attenuation"])
        arr = run_bellhop(env, SONAR["f_centre"], own, zs, ranges, n_beams=6001, angles=(-45.0, 45.0))
        cfg = dict(CFG, bottom=SAND, rays={"max_bounces": 3, "max_arrivals": 40, "fan": 2880})
        f = coverage.field(coverage.water(d, c), water, own, zs, ranges, cfg, SONAR)
        diff, agree, total = [], 0, 0
        for i in range(len(zs)):
            for j, r in enumerate(ranges):
                a = np.abs(np.asarray(arr.amplitude[i][j]))
                ns, nb = np.asarray(arr.n_surface[i][j]), np.asarray(arr.n_bottom[i][j])
                few = (ns + nb) <= 3
                direct = bool(np.any((ns == 0) & (nb == 0) & (a > 0.0)))
                total += 1
                agree += direct == bool(f["direct"][i, j])
                if few.any() and np.isfinite(f["incoherent_db"][i, j]):
                    slant = np.hypot(r, zs[i] - own)
                    diff.append(f["incoherent_db"][i, j] - 10.0 * np.log10(np.sum(a[few] ** 2) * slant**2))
        diff = np.abs(np.array(diff))
        assert agree >= 0.9 * total, (name, agree, total)
        assert np.median(diff) < 3.0 and np.mean(diff < 6.0) >= 0.8, (name, np.median(diff), np.mean(diff < 6.0))


def record():
    return SonarModel.load("popoto_long_range").make_record(make_pose())


def test_levels_are_the_builds_budget_and_detection_is_the_detectors():
    eq, _ = bs.budget(D, 5.0)
    f = coverage.field(ISO, None, 225.0, [150.0], RANGES, CFG, SONAR)
    assert np.allclose(f["snr_0db_target_db"][0] - f["factor_db"][0], eq.snr(f["slant"][0]), atol=1e-6)      # one row: no rows' gain in either
    snr, pd = coverage.pd_table(record(), {})
    for s in (8.0, 14.0, 16.0, 20.0):
        assert np.interp(s, snr, pd) == pytest.approx(live.detection_probability(s, record(), {}), abs=1e-6)
    assert np.interp(0.5, pd, snr) == pytest.approx(live.budget_threshold_db(record(), {}, pd=0.5), abs=0.3)
    swerling = coverage.pd_table(record(), {"coverage": {"fluctuation": "swerling1"}})[1]
    assert np.interp(25.0, snr, swerling) < np.interp(25.0, snr, pd)


def test_summary_is_its_grid_and_could_have_seen_reads_it():
    """Isovelocity deep water, the face row at 225 m: each class's greatest range is where the worst of its depths
    crosses the stated probability on a direct evaluation of the same field, a shallow ship has its Lloyd nulls as blind
    bands, and the report's member answers "could I have seen it" at a range."""
    table = coverage.pd_table(record(), {})
    s = coverage.summary(ISO, None, 225.0, CFG, SONAR, table, D["window_m"], float(D["projector_m"][2]))
    assert s["profile"] == "supplied" and s["own_depth_m"] == 225 and s["water_depth_m"] is None and s["age_pings"] == 0
    assert len(json.dumps(s)) < 1500
    ranges = np.linspace(D["window_m"][0], D["window_m"][1], 48)
    for c, got in zip(CLASSES, s["classes"]):
        zs = np.linspace(c["depth_m"][0], c["depth_m"][1], 5)
        f = coverage.field(ISO, None, 225.0, zs, ranges, CFG, SONAR, float(D["projector_m"][2]))
        worst = np.interp(f["snr_0db_target_db"] + c["ts_db"], *table).min(axis=0)
        held = np.flatnonzero(worst >= 0.5)
        assert got["max_range_m"] == pytest.approx(ranges[held[-1]], abs=ranges[1] - ranges[0] + 1.0)
        assert got["direct_to_m"] == pytest.approx(ranges[-1], abs=1.0)
        seen = coverage.could_have_seen(s, 5000.0)[c["name"]]
        assert seen == pytest.approx(float(np.interp(5000.0, ranges, worst)), abs=0.25)
    xl = next(c for c in s["classes"] if c["name"] == "xl_auv")
    assert 3000.0 < xl["max_range_m"] < 9000.0 and coverage.could_have_seen(s, 11000.0)["xl_auv"] < 0.1 and coverage.could_have_seen(s, 2000.0)["xl_auv"] > 0.9
    assert coverage.could_have_seen(s, 20000.0)["ship"] is None                       # beyond the window
    # nothing invented: without the budget or the table the member says what is missing
    bare = core.Coverage(coverage.cpp_options({"classes": CLASSES}))
    assert "link budget" in bare.update_numbers(SONAR, 750.0, 12700.0, 225.0, 225.0, 0.0, ISO)
    no_table = core.Coverage(coverage.cpp_options(CFG))
    assert "detection-probability table" in no_table.update_numbers(SONAR, 750.0, 12700.0, 225.0, 225.0, 0.0, ISO)


def test_refreshed_when_something_changed_and_recommends_a_depth():
    """A shallow sonar under a strong thermocline (the stage-2 water): the summary is repeated while nothing moves, made
    again when the depth, the water depth or the profile does; the direct path ends inside the window, and a deeper own
    depth is recommended with what it gains."""
    from test_raytrace_cpp import PROFILES
    d, c, water, own = PROFILES["strong thermocline"]
    w = coverage.water(d, c, rung="built")
    cov = core.Coverage(coverage.cpp_options(dict(CFG, bottom=SAND, own_depth_limits_m=[5.0, 500.0])))
    cov.set_pd_table(*coverage.pd_table(record(), {}))
    first = json.loads("{" + cov.update_numbers(SONAR, 750.0, 12700.0, 20.0, 24.0, water, w) + "}")["coverage"]
    assert cov.refreshed() and first["age_pings"] == 0 and first["profile"] == "built" and first["water_depth_m"] == 200
    print(f"\ncoverage refresh, 3 classes, 10 own depths, 200 m of water with a bottom: {cov.seconds() * 1e3:.0f} ms")
    again = json.loads("{" + cov.update_numbers(SONAR, 750.0, 12700.0, 22.0, 26.0, water, w) + "}")["coverage"]
    assert not cov.refreshed() and again["age_pings"] == 1 and {k: v for k, v in again.items() if k != "age_pings"} == {k: v for k, v in first.items() if k != "age_pings"}
    cov.update_numbers(SONAR, 750.0, 12700.0, 27.0, 31.0, water, w)
    assert cov.refreshed()                                                           # 7 m deeper
    cov.update_numbers(SONAR, 750.0, 12700.0, 27.0, 31.0, water + 15.0, w)
    assert cov.refreshed()
    moved = coverage.water(d, [v + (1.0 if z > 100.0 else 0.0) for z, v in zip(d, c)], rung="built")
    cov.update_numbers(SONAR, 750.0, 12700.0, 27.0, 31.0, water + 15.0, moved)
    assert cov.refreshed()
    sub = next(k for k in first["classes"] if k["name"] == "large_submerged")
    assert sub["direct_to_m"] < 6000                                                 # the shadow of the thermocline
    assert first["recommend"]["own_depth_m"] <= 200.0 - 10.0 - 4.0 and first["recommend"]["gain_m"] >= 0
    assert cov.seen(0, 3000.0) == pytest.approx(coverage.could_have_seen(first, 3000.0)[first["classes"][0]["name"]], abs=0.3)


def strip(report):
    out = copy.deepcopy(report)
    out.pop("processing_seconds", None)
    out.pop("coverage", None)
    return out


def test_both_engines_print_the_same_member_and_nothing_else_changes():
    records = run_scenario(T.chain_scenario(6000.0, 15.0, pings=3)).records
    base = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host"}
    got = {}
    for engine in ("cpp", "python"):
        with LiveChain(dict(base, engine=engine)) as chain:
            plain = chain.run([r.copy() for r in records])
        with LiveChain(dict(base, engine=engine, coverage=CFG)) as chain:
            got[engine] = chain.run([r.copy() for r in records])
        for a, b in zip(plain, got[engine]):
            assert "coverage" not in a and strip(a) == strip(b), engine              # detections and tracks bit for bit
        assert [r["coverage"]["age_pings"] for r in got[engine]] == [0, 1, 2]        # made once: nothing moved
        c = got[engine][0]["coverage"]
        assert c["own_depth_m"] == pytest.approx(225.0, abs=0.5) and c["profile"] == "none" and [k["name"] for k in c["classes"]] == [k["name"] for k in CLASSES]
        assert c["ranges_m"][0] >= 5500 and c["ranges_m"][-1] <= 6500                # the ping's own window
    assert [r["coverage"] for r in got["cpp"]] == [r["coverage"] for r in got["python"]]
    o = live.cpp_chain_options(dict(base, coverage=CFG))
    assert o["coverage"]["classes"].startswith("ship:25.0:2.0:6.0;") and "coverage" not in live.cpp_chain_options(base)
    with pytest.raises(ValueError):
        coverage.options({"threshold_db": 13.0})                                     # there is no typed threshold to give


def test_the_tool_runs_as_a_program(tmp_path):
    out = tmp_path / "coverage.json"
    fig = tmp_path / "coverage.png"
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "coverage_map.py"), "--build", "popoto_face_1x32", "--environment", "deep_no_bottom_return", "--own-depth", "225",
                        "--profile", "prior", "--ensemble=-0.05,0.017", "--extent", "100", "--json", str(out), "--figure", str(fig), "--chain-config"],
                       capture_output=True, text=True, timeout=600, env=dict(os.environ, PYTHONPATH=str(ROOT / "python")))
    assert p.returncode == 0, p.stderr[-2000:]
    R = json.loads(out.read_text())
    assert R["profile_rung"] == "local" and R["ensemble_members"] >= 3 and fig.stat().st_size > 10000
    assert all("max_range_ensemble_m" in c for c in R["summary"]["classes"]) and '"coverage"' in p.stdout
    grid = R["grid"]
    assert np.array(grid["pd"]["ship"]).shape == (len(grid["target_depths_m"]), len(grid["ranges_m"]))
