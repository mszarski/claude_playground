"""tools/depth_range_sweep.py: its closed forms, its judge, and one cheap case through both engines (the plumbing).

What the tool measures is in docs/tasks/DEPTH_RANGE.md; these tests hold only that it computes what it says."""

import functools
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("depth_range_sweep", ROOT / "tools" / "depth_range_sweep.py")
drs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drs)


def test_physics_closed_forms():
    out = drs.physics(ranges=(1000.0, 12000.0), target_depths=(60.0,), own_depths=(30.0, 225.0))
    c = out["sound_speed_m_s"]
    assert out["rows_aperture_m"] == pytest.approx(1.5)
    assert out["aperture_wavelengths"] == pytest.approx(1.5 / (c / 5000.0))
    assert out["range_cell_m"] == pytest.approx(c / 4000.0)                  # c / 2B, B = 2 kHz
    by = {(m["own_depth_m"], m["range_m"]): m for m in out["multipath"]}
    m = by[(30.0, 1000.0)]
    assert m["two_way_delay_ms"] == pytest.approx(1e3 * 4.0 * 30.0 * 60.0 / (1000.0 * c))
    assert m["direct_echo_ahead_m"] == pytest.approx(1.8)                     # z_o z_t / R
    assert m["direct_echo_resolved"] and m["multipath_searchable"]
    m = by[(30.0, 12000.0)]
    assert not m["direct_echo_resolved"] and not m["multipath_searchable"]
    assert by[(225.0, 12000.0)]["direct_echo_resolved"]
    rows = {r["range_m"]: r for r in out["stacked_rows"]}
    # the gradient prior's term is the chain's own refraction rule, r^2 sigma_g / 2c, with the deployed 0.05 / s
    assert rows[12000.0]["from_gradient_m"] == pytest.approx(12000.0**2 * 0.05 / (2.0 * c))
    assert rows[12000.0]["depth_sigma_gradient_known_m"] < rows[12000.0]["depth_sigma_deployed_m"]


def test_the_rows_read_the_surface_when_direct_and_surface_share_the_cell():
    # The derivation the tool states: two plane waves at theta_d and theta_s, the second with mu = -1 and any phase, read
    # across a small height by the phase slope, give the mean elevation, whatever the phase.
    k, th_d, th_s = 2 * math.pi / 0.3, -0.004, 0.012
    h = np.array([-0.01, 0.01])                                # a small aperture: the slope at the centre
    for phi in (0.3, 1.0, 2.0, 2.9, 4.0):
        field = np.exp(1j * k * h * th_d) - np.exp(1j * phi) * np.exp(1j * k * h * th_s)
        slope = np.angle(field[1] * np.conj(field[0])) / (k * (h[1] - h[0]))
        assert slope == pytest.approx(0.5 * (th_d + th_s), rel=1e-3)


def test_judge():
    assert drs.judge(None, None, 60.0)["call"] == "none"
    j = drs.judge(58.0, 5.0, 60.0)
    assert j["honest"] and j["call"] == "submerged" and j["call_right"]
    j = drs.judge(3.0, 4.0, 60.0)                              # a wrong depth stated confidently
    assert not j["honest"] and j["call"] == "surface" and j["call_right"] is False
    j = drs.judge(30.0, 40.0, 60.0)
    assert j["honest"] and j["call"] == "undecided" and j["call_right"] is None
    assert drs.judge(4.0, 3.0, 4.0)["call"] == "surface"


def test_blind_range_is_skipped():
    case = drs.run_case(drs.bs.load(drs.SPEC), range_m=1000.0, target_z=60.0, own_z=30.0, water=None, waveform="hfm2", seed=1,
                        pings=1, ts_db=25.0, engines=("cpp",), multipath=False)
    assert "blind" in case["skipped"]


def test_quick_case_both_engines(tmp_path):
    out = tmp_path / "q.json"
    assert drs.main(["--quick", "--json", str(out)]) == 0
    import json
    case = json.loads(out.read_text())["cases"][0]
    py, cpp = case["python"]["pings"][0], case["cpp"]["pings"][0]
    assert py["detected"] and cpp["detected"]
    assert py["depth_m"] == pytest.approx(cpp["depth_m"], abs=drs.ENGINE_TOLERANCE_M)          # single-precision maps: millimetres
    assert py["depth_sigma_m"] == pytest.approx(cpp["depth_sigma_m"], abs=drs.ENGINE_TOLERANCE_M)
    for key in ("rows", "rows_deployed", "sigma_deployed_m", "multipath_reason", "timing"):
        assert key in cpp


def test_the_gpu_job_has_the_mode():
    job = (ROOT / "tools" / "gpu_job.sh").read_text()
    assert 'GPUSONAR_JOB:-tests}" = "depthrange"' in job and "tools/depth_range_sweep.py --backend cuda" in job
    for sweep in ("deep_own225:--water deep --own 225", "deep_own30:--water deep --own 30", "shelf100_own30:--water 100 --own 30"):
        assert sweep in job
    common = job.split('common="${GPUSONAR_DEPTHRANGE_ARGS:-')[1].split('}"')[0].split()
    ap_args = common + ["--water", "deep", "--own", "225"]                 # the job's arguments parse as the tool's
    flags = {a for a in ap_args if a.startswith("--")}
    assert flags <= {"--ranges", "--depths", "--waveform", "--pings", "--seeds", "--multipath", "--water", "--own"}


@functools.lru_cache(maxsize=None)
def _rows_depths(range_m, target_z=60.0, ts_db=10.0, pings=2):
    """The face's deployed stacked-row chain (told the water, as its self-test is) on the 1 s linear sweep in 100 m of water,
    own depth 30 m, a target at `range_m` and `target_z` (the trial's 3 degree roll), with the attitude history navigation sends."""
    case = drs.run_case(drs.bs.load(drs.SPEC), range_m=range_m, target_z=target_z, own_z=30.0, water=100.0, waveform="lfm1", seed=1, pings=pings,
                        ts_db=ts_db, engines=("python", "cpp"), multipath=False, attitude="trial")
    return case["python"]["pings"], case["cpp"]["pings"]


SHALLOW = [(1000.0, 60.0, 10.0), (1700.0, 60.0, 10.0), (1700.0, 4.0, 25.0)]      # range, depth, target strength


@pytest.mark.parametrize("range_m, target_z, ts_db", SHALLOW)
def test_the_engines_agree_on_the_shallow_cases(range_m, target_z, ts_db):
    py, cpp = _rows_depths(range_m, target_z, ts_db)
    for a, b in zip(py, cpp):
        assert a["detected"] and b["detected"]
        assert a["depth_m"] == pytest.approx(b["depth_m"], abs=drs.ENGINE_TOLERANCE_M) and a["depth_sigma_m"] == pytest.approx(b["depth_sigma_m"], abs=drs.ENGINE_TOLERANCE_M)
        assert a.get("multipath_term_m") == (pytest.approx(b["multipath_term_m"], abs=drs.ENGINE_TOLERANCE_M) if b.get("multipath_term_m") is not None else None)


@pytest.mark.xfail(strict=True, reason=(
    "the stacked rows' depth is dishonest in 100 m of water at 1.0 and 1.7 km once the attitude is known at the echo "
    "(docs/tasks/DEPTH_RANGE.md, F1): the rows read a mixture of the direct echo with the direct-surface and direct-bottom "
    "pairs a few range cells behind it, the multipath term (libs/live multipath_depth_limit) finds no echo that explains the "
    "reading (`depth_unresolved_sigma_m: null`) or only a narrow band without the truth, adds nothing, and the angle's own "
    "sigma (roll 0.2 deg, and the fold through the bottom) is stated: 68.5 +- 3.5 m and 29.2 +- 11.7 m for 60 m, and a ship "
    "4 m deep at 1.7 km reads the sea bed, 93 to 99 +- 26 to 29 m: called submerged. The remedy moves a default result "
    "(the term is on in every stacked-row deployment file): the owner's decision"))
@pytest.mark.parametrize("range_m, target_z, ts_db", SHALLOW)
def test_stacked_row_depth_is_honest_in_shallow_water(range_m, target_z, ts_db):
    pings = _rows_depths(range_m, target_z, ts_db)[1]
    assert all(p["rows"]["honest"] for p in pings), [(round(p["depth_m"], 1), round(p["depth_sigma_m"], 1), p.get("multipath_term_m")) for p in pings]


@functools.lru_cache(maxsize=None)
def _ship_timing(range_m):
    """The multipath-timing stage as a build would deploy it (rays, flagged depths withheld) on a ship-sized point 4 m deep,
    the sonar 225 m deep in deep water (the planner's own depth at the deep site), 1 s linear sweep, one ping."""
    case = drs.run_case(drs.bs.load(drs.SPEC), range_m=range_m, target_z=4.0, own_z=225.0, water=None, waveform="lfm1", seed=1, pings=1,
                        ts_db=25.0, engines=("python", "cpp"), multipath=True)
    return case["python"]["pings"][0], case["cpp"]["pings"][0]


@pytest.mark.xfail(strict=True, reason=(
    "the multipath-timing stage gives a target shallower than its search's floor the floor's depth with the floor's "
    "small sigma: 4 m reads 5.1 +- 0.2 m at 1.7 km and 8.0 +- 0.3 m at 3 km (docs/tasks/DEPTH_RANGE.md, F2). The search "
    "starts where the surface echo's delay is min_delay_bandwidth / B (z_min = 3 R c / (4 B z_own): 4.25 m at 1.7 km, "
    "7.5 m at 3 km), and the posterior spread over a grid cut there cannot cover a truth above it. Not deployed (no shipped "
    "build enables chain.multipath_depth); the remedy (an upper bound, or a flag that withholds it) changes the opt-in "
    "stage's output: reported, not fixed here"))
@pytest.mark.parametrize("range_m", [1700.0, 3000.0])
def test_multipath_timing_is_honest_above_its_search_floor(range_m):
    py, cpp = _ship_timing(range_m)
    assert py.get("multipath_depth_m") == (pytest.approx(cpp["multipath_depth_m"], abs=drs.ENGINE_TOLERANCE_M) if cpp.get("multipath_depth_m") is not None else None)
    assert cpp["timing"]["honest"] is not False, (cpp.get("multipath_depth_m"), cpp.get("multipath_sigma_m"))
