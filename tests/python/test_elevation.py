"""Stacked rows on the long-range array: elevation and depth as well as bearing.

Free-field truth and the Cramer-Rao bound, the second lobe of two rows and its cure with
three, the cone-angle correction, depth with the vehicle rolled, a channel with surface and
bottom bounces, dead channels, passive sources, and the two chain engines against each
other. One row must come out exactly as before.
"""

import json
import pathlib
import subprocess

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import detect, elevation, flank, recon, sync
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, run_scenario, simulate_ping
from gpusonar.sim.passive import passive_record
from gpusonar.sonar import SonarModel

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "bin" / "sonar_live"
C = 1500.0
BASEBAND = {"kind": "baseband", "sample_rate": 8.0e3}
CHAIN = {"bearings_deg": [-50.0, 50.0, 0.5], "range_cell_m": 2.0, "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 100.0},
         "min_snr_db": 15.0, "backend": "host", "tracker": {"range_sigma": 5.0, "bearing_sigma_deg": 1.0, "confirm_hits": 2, "max_misses": 2},
         "classify": {"min_speed": 1.0, "static_speed": 0.6, "max_fit_rms": 10.0, "multipath": None}}


def sonar_rows(name="popoto_long_range_2row", window=(900.0, 1100.0), **over):
    return SonarModel.load(name, {"receive": dict(BASEBAND, window={"range": list(window)}), **over})


def truth(record, rows, target):
    """(slant range, cone bearing, elevation in the array frame, level elevation) of a point."""
    frame = elevation.array_frame(record, rows)
    d = np.asarray(target, dtype=float) - frame.origin
    r = float(np.linalg.norm(d))
    u = d / r
    return r, float(np.arcsin(u @ frame.along)), float(np.arcsin(u @ frame.up)), float(np.arcsin(-u[2]))


def one_ping(sonar, target, *, pose=None, noise_rms=0.0, seed=1, ts=10.0):
    scene = Scene().add_points([target], target_strength_db=ts) if target is not None else Scene()
    return simulate_ping(sonar, pose or make_pose(position=(0.0, 0.0, 60.0)), scene, environment={"sound_speed": C},
                         noise_rms=noise_rms, rng=np.random.default_rng(seed))


def estimate_one(record, target, *, cfg=None, background=1e-30, offset=0.6):
    recon.prepare([record])
    rows = elevation.find_rows(record)
    frame = elevation.array_frame(record, rows)
    r, cone, el_true, _ = truth(record, rows, target)
    cfg = dict(elevation.DEFAULTS, **(cfg or {}))
    est = elevation.estimate(record, rows, elevation.row_records(record, rows), [(r + offset, cone, float("nan"))], [background],
                             frame, cfg, noise_weight=float(sum(len(x) for x in rows)), range_cell=2.0)[0]
    return est, el_true, frame


# --- layout and rows --------------------------------------------------------------------------------

def test_stacked_layout_puts_rows_at_their_heights_and_boards_follow():
    two, three, one = (SonarModel.load(n) for n in ("popoto_long_range_2row", "popoto_long_range_3row", "popoto_long_range"))
    assert (two.n_elements, len(two.boards), three.n_elements, len(three.boards)) == (64, 8, 96, 12)
    assert np.allclose(two.positions[:32, [0, 1]], one.positions[:, [0, 1]])          # each row is the one-row array
    assert np.allclose(two.positions[:32, 2], 0.19) and np.allclose(two.positions[32:, 2], -0.19)   # z is down: the first row is the lower
    assert np.allclose(three.positions[32:64], one.positions + [0.0, 0.079, 0.0])     # the middle row, further out
    # Four from each row in turn: a board of eight inputs then holds the same stations of both rows.
    inter = SonarModel.load("popoto_long_range_2row", {"array": {"interleave": 4}})
    assert np.allclose(inter.positions[:4], two.positions[:4]) and np.allclose(inter.positions[4:8], two.positions[32:36])
    assert sorted(map(tuple, np.round(inter.positions, 6))) == sorted(map(tuple, np.round(two.positions, 6)))


def test_rows_are_found_from_the_positions_whatever_the_wiring_and_with_channels_missing():
    rec = sonar_rows().make_record(make_pose())
    assert elevation.find_rows(rec) == [list(range(32)), list(range(32, 64))] == core.live_find_rows(rec)
    assert len(elevation.find_rows(sonar_rows("popoto_long_range").make_record(make_pose()))) == 1
    assert [len(r) for r in elevation.find_rows(sonar_rows("popoto_long_range_3row").make_record(make_pose()))] == [32, 32, 32]
    inter = sonar_rows(array={"interleave": 4}).make_record(make_pose())
    rows = elevation.find_rows(inter)
    assert rows == core.live_find_rows(inter) and rows[0][:8] == [0, 1, 2, 3, 8, 9, 10, 11] and rows[1][:4] == [4, 5, 6, 7]
    # A dead hydrophone and a dead board (eight inputs of the upper row) leave two usable rows.
    less = sync.select_channels(rec, [r for r in range(64) if r != 5 and not 40 <= r < 48])
    rows = elevation.find_rows(less)
    assert [len(r) for r in rows] == [31, 24] and rows == core.live_find_rows(less)
    frame = elevation.array_frame(less, rows)
    assert np.allclose(frame.along, [1.0, 0.0, 0.0]) and np.allclose(frame.up, [0.0, 0.0, -1.0]) and np.allclose(frame.look, [0.0, 1.0, 0.0])


# --- accuracy ---------------------------------------------------------------------------------------

def test_elevation_is_exact_without_noise_whatever_the_roll_and_the_projector_offset():
    """The projector hangs 1.5 m below the array: the search must hold the two-way path, not the
    range from the array, or the pulse's envelope drags the peak (an early version read 1.6 degrees off)."""
    for roll_deg, target in ((0.0, [200.0, 980.0, -40.0]), (4.0, [200.0, 980.0, 210.0]), (-6.0, [-400.0, 900.0, 60.0])):
        rec = one_ping(sonar_rows(), target, pose=make_pose(position=(0.0, 0.0, 60.0), rpy=(np.radians(roll_deg), 0.0, 0.0)))
        est, el_true, frame = estimate_one(rec, target)
        assert abs(np.degrees(np.arcsin(est.sin_elevation) - el_true)) < 0.02
        assert est.quality > 0.999
        assert frame.origin[2] + est.range * est.world[2] == pytest.approx(target[2], abs=0.5)    # depth, roll taken out


def test_elevation_is_unbiased_and_meets_the_cramer_rao_bound_in_noise():
    sonar = sonar_rows(window=(960.0, 1040.0))
    target = [100.0, 1000.0, 110.0]
    clean = one_ping(sonar, target)
    recon.prepare([clean])
    peak = float((np.abs(clean.samples.array) ** 2).max(axis=1).mean())           # echo power at a hydrophone
    rows = elevation.find_rows(clean)
    heights = elevation.array_frame(clean, rows).heights
    for snr_db, factor in ((3.0, 1.3), (13.0, 1.25)):                            # at one hydrophone: 18 and 28 dB after a row's beamforming
        noise = np.sqrt(peak / 10.0 ** (snr_db / 10.0))
        quiet = one_ping(sonar, None, noise_rms=1.0, seed=99)
        recon.prepare([quiet])
        # noise_rms that gives that level after compression, where the replica overlaps the data fully
        scale = noise / float(np.sqrt((np.abs(quiet.samples.array[:, :flank._n_valid(quiet)]) ** 2).mean()))
        bound = np.degrees(elevation.sin_elevation_sigma(heights, 10.0 ** (snr_db / 10.0), 4000.0, 6000.0, C))
        errors, sigmas = [], []
        for seed in range(60):
            rec = one_ping(sonar, target, noise_rms=scale, seed=seed)
            est, el_true, _ = estimate_one(rec, target, background=64.0 * noise**2)
            errors.append(np.degrees(np.arcsin(est.sin_elevation) - el_true))
            sigmas.append(np.degrees(est.sigma))
        errors = np.array(errors)
        assert abs(errors.mean()) < 3.0 * bound / np.sqrt(len(errors))            # unbiased
        assert bound / factor < errors.std() < bound * factor                     # at the bound, within the scatter of 60 trials
        assert np.median(sigmas) == pytest.approx(bound, rel=0.25)                # and the sigma it reports is that bound


def test_bound_formula_and_unambiguous_sector():
    # Two rows d apart, one row's SNR rho_row: sigma(sin e) = lambda_eff / (pi d sqrt(2 rho_row)).
    d, rho_row = 0.38, 100.0
    heights = np.repeat([-d / 2, d / 2], 32)
    k = elevation.rms_wavenumber(4000.0, 6000.0, C)
    assert k == pytest.approx(2 * np.pi / C * np.sqrt(5000.0**2 + 2000.0**2 / 12))
    assert elevation.sin_elevation_sigma(heights, rho_row / 32, 4000.0, 6000.0, C) == pytest.approx(1.0 / (k * (d / 2) * np.sqrt(2 * 2 * rho_row)))
    assert elevation.unambiguous_elevation(0.15, 5000.0) == pytest.approx(np.pi / 2)       # half a wavelength: no second lobe
    assert np.degrees(elevation.unambiguous_elevation(0.38, 6000.0)) == pytest.approx(19.2, abs=0.1)
    assert np.degrees(elevation.cone_bias(np.radians(30.0), np.radians(10.0))) == pytest.approx(0.512, abs=0.002)


def test_two_rows_flag_a_steep_arrival_as_ambiguous_and_three_rows_resolve_it():
    target = [0.0, 940.0, 60.0 - 342.0]                                            # 20 degrees up, 1 km
    rec = one_ping(sonar_rows(), target)
    est, el_true, _ = estimate_one(rec, target)
    assert est.ambiguous                                                           # a second lobe within 1.5 dB inside +-30 degrees
    rec = one_ping(sonar_rows("popoto_long_range_3row"), target)
    est, el_true, _ = estimate_one(rec, target)
    assert not est.ambiguous and abs(np.degrees(np.arcsin(est.sin_elevation) - el_true)) < 0.05
    # A level arrival is not ambiguous with two rows either.
    level = [0.0, 1000.0, 60.0]
    assert not estimate_one(one_ping(sonar_rows(), level), level)[0].ambiguous


# --- depth, folded ----------------------------------------------------------------------------------

def test_fold_depth_and_the_outside_water_test_match_the_cpp_ones():
    for z, water, want in ((50.0, 100.0, (50.0, "direct")), (-50.0, 100.0, (50.0, "surface")), (150.0, 100.0, (50.0, "bottom")),
                           (-150.0, 100.0, (50.0, "multiple")), (250.0, 100.0, (50.0, "multiple")), (-20.0, None, (20.0, "surface")),
                           (300.0, None, (300.0, "direct"))):
        got = elevation.fold_depth(z, water)
        assert got[0] == pytest.approx(want[0]) and got[1] == want[1]
        cpp = core.live_fold_depth(z, water or 0.0)
        assert cpp[0] == pytest.approx(want[0]) and cpp[1] == want[1]
    assert elevation.outside_water(-30.0, 10.0, 100.0) and not elevation.outside_water(-15.0, 10.0, 100.0)
    assert elevation.outside_water(125.0, 10.0, 100.0) and not elevation.outside_water(125.0, 10.0, None)


def test_ray_depth_is_the_straight_line_in_constant_sound_speed_and_obeys_snell_in_a_gradient():
    assert elevation.ray_depth(30.0, np.radians(-2.0), 1000.0, [0, 100], [C, C]) == pytest.approx(30.0 + 1000.0 * np.sin(np.radians(2.0)), abs=1e-6)
    # Up to the surface and back down: the fold.
    assert elevation.ray_depth(30.0, np.radians(5.0), 1000.0, [0, 100], [C, C], 100.0) == pytest.approx(1000.0 * np.sin(np.radians(5.0)) - 30.0, abs=1e-6)
    # Constant gradient g: travel time between depths is (1/g) ln[c2 (1 + sin t1) / (c1 (1 + sin t2))].
    g, z0, theta = 0.05, 100.0, np.radians(8.0)
    z = elevation.ray_depth(z0, -theta, 3000.0, [0.0, 2000.0], [C - g * z0, C + g * (2000.0 - z0)], step=0.5)
    c2 = C + g * (z - z0)
    theta2 = np.arccos(np.cos(theta) * c2 / C)
    assert np.log(c2 * (1 + np.sin(theta)) / (C * (1 + np.sin(theta2)))) / g == pytest.approx(3000.0 / C, rel=2e-4)
    assert z < z0 + 3000.0 * np.sin(theta) - 100.0                                 # the ray bends up: the straight line is 150 m too deep


def test_tracker_carries_depth_from_the_detections_that_have_one():
    tracker = detect.Tracker(confirm_hits=2, depth_process_noise=0.0)
    cpp = core.LiveChainCpp     # noqa: F841  (the C++ tracker's depth is checked through the chain below)
    for i, (depth, sigma) in enumerate(((40.0, 4.0), (float("nan"), float("nan")), (60.0, 4.0))):
        tracks = tracker.update(float(i), [detect.Detection(500.0, 0.1, 20.0, 1, depth, sigma)])
    assert tracks[0].depth == pytest.approx(50.0) and tracks[0].depth_var == pytest.approx(8.0)   # two equal measurements


# --- the chain --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def free_field():
    return run_scenario("popoto_elevation_free").records


def run(records, chain, engine="python"):
    with LiveChain(dict(chain, engine=engine)) as live:
        return live.run([r.copy() for r in records])


def test_chain_reports_depth_with_the_roll_taken_out_and_the_bearing_corrected(free_field):
    chain = dict(CHAIN, elevation={"roll_sigma_deg": 0.2})
    reports = run(free_field, chain)
    last = reports[-1]
    assert last["rows"] == 2
    targets = {(-300.0, 900.0): 10.0, (100.0, 1000.0): 60.0, (500.0, 1100.0): 160.0}     # the scenario's, with their depths
    assert len(last["tracks"]) == 3
    for t in last["tracks"]:
        (x, y), depth = min(targets.items(), key=lambda kv: np.hypot(kv[0][0] - t["x"], kv[0][1] - t["y"]))
        assert np.hypot(x - t["x"], y - t["y"]) < 5.0
        assert t["depth_m"] == pytest.approx(depth, abs=4.0) and t["path"] == "direct" and t["depth_sigma_m"] < 3.0
    for d in last["detections"]:
        assert d["elevation_quality"] > 0.99 and not d["elevation_ambiguous"]
        assert d["elevation_sigma_deg"] == pytest.approx(0.2, abs=0.02)            # the navigation's roll accuracy dominates here
    deep = max(last["detections"], key=lambda d: d["depth_m"])
    assert abs(deep["cone_bearing_deg"] - deep["bearing_deg"]) > 0.05              # 4.9 degrees down at 23 degrees off broadside: 0.09 degrees
    # Each use of the elevation has its own switch.
    plain = run(free_field, dict(CHAIN, elevation={"correct_bearing": False, "depth": False}))[-1]
    assert all("cone_bearing_deg" not in d and "depth_m" not in d and "elevation_deg" in d for d in plain["detections"])
    assert all("depth_m" not in t for t in plain["tracks"])
    assert max(plain["detections"], key=lambda d: d["range_m"])["bearing_deg"] == pytest.approx(deep["cone_bearing_deg"], abs=1e-6)


def test_bearing_is_corrected_for_the_cone_angle_in_both_engines():
    """10 degrees up at 30 degrees off broadside: one line reports the cone angle, 0.5 degrees short
    (105 m of cross-range at 12 km). With the elevation the horizontal bearing is recovered."""
    sonar = sonar_rows("popoto_long_range_3row", window=(800.0, 1200.0))
    el, bearing, r = np.radians(10.0), np.radians(30.0), 1000.0
    target = [r * np.cos(el) * np.sin(bearing), r * np.cos(el) * np.cos(bearing), 60.0 - r * np.sin(el)]
    rec = one_ping(sonar, target, noise_rms=1e-6)
    for engine in ("python", "cpp"):
        for correct in (True, False):
            d = max(run([rec], dict(CHAIN, elevation={"correct_bearing": correct}), engine)[0]["detections"], key=lambda x: x["snr_db"])
            cone = np.degrees(np.arcsin(np.cos(el) * np.sin(bearing)))             # 29.50
            assert d["elevation_deg"] == pytest.approx(10.0, abs=0.05) and d["image_depth_m"] == pytest.approx(target[2], abs=1.0)
            if correct:
                assert d["bearing_deg"] == pytest.approx(30.0, abs=0.06) and d["cone_bearing_deg"] == pytest.approx(cone, abs=0.06)
            else:
                assert d["bearing_deg"] == pytest.approx(cone, abs=0.06) and "cone_bearing_deg" not in d
    assert np.degrees(elevation.cone_bias(np.arcsin(np.cos(el) * np.sin(bearing)), el)) == pytest.approx(30.0 - cone, abs=1e-9)


def test_cpp_chain_gives_the_python_chains_elevation_and_depth(free_field):
    chain = dict(CHAIN, elevation={"roll_sigma_deg": 0.2})
    want, got = run(free_field, chain), run(free_field, chain, "cpp")
    assert set(want[-1]) == set(got[-1])
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"]) == 3
        for a, b in zip(w["detections"], g["detections"]):
            assert set(a) == set(b)
            for key, tol in (("range_m", 0.05), ("bearing_deg", 0.01), ("cone_bearing_deg", 0.01), ("elevation_deg", 0.01),
                             ("elevation_sigma_deg", 0.002), ("elevation_quality", 0.001), ("depth_m", 0.2), ("depth_sigma_m", 0.05),
                             ("image_depth_m", 0.2)):
                assert a[key] == pytest.approx(b[key], abs=tol), key
            assert (a["path"], a["elevation_ambiguous"]) == (b["path"], b["elevation_ambiguous"])
        assert [(t["id"], t["kind"], t.get("path")) for t in w["tracks"]] == [(t["id"], t["kind"], t.get("path")) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            assert set(a) == set(b)
            for key in ("x", "y", "depth_m", "depth_sigma_m"):
                assert a[key] == pytest.approx(b[key], abs=0.2), key


def test_one_row_is_processed_exactly_as_without_the_option():
    """The option on a one-row array, and the option off, give the same report lines in both engines."""
    sonar = SonarModel.load("popoto_long_range", {"receive": dict(BASEBAND, window={"range": [800.0, 1200.0]})})
    rng = np.random.default_rng(4)
    pings = [simulate_ping(sonar, make_pose(position=(1.5 * i, 0.0, 30.0), velocity=(1.5, 0.0, 0.0)),
                           Scene().add_points([[100.0, 1000.0, 80.0], [-200.0, 950.0, 30.0]], target_strength_db=10.0),
                           environment={"sound_speed": C}, sequence=i, time_ns=int(4e9 * i), noise_rms=1e-4, rng=rng) for i in range(3)]
    for engine in ("python", "cpp"):
        off, on = run(pings, CHAIN, engine), run(pings, dict(CHAIN, elevation={}), engine)
        for a, b in zip(off, on):
            a.pop("processing_seconds"), b.pop("processing_seconds")
            assert json.dumps(a) == json.dumps(b)
        assert len(off[-1]["detections"]) == 2 and len(off[-1]["tracks"]) == 2 and "rows" not in on[-1]


def test_dead_channels_and_a_dead_board_do_not_break_the_estimate(free_field):
    dead = [5] + list(range(40, 48))                       # one hydrophone of the lower row, one board of the upper
    chain = dict(CHAIN, elevation={}, dead_channels=dead)
    want, got = run(free_field, chain), run(free_field, chain, "cpp")
    depths = sorted(t["depth_m"] for t in want[-1]["tracks"])
    assert depths == pytest.approx([10.0, 60.0, 160.0], abs=5.0)
    assert sorted(t["depth_m"] for t in got[-1]["tracks"]) == pytest.approx(depths, abs=0.3)
    with pytest.raises(Exception, match="dead_channels"):
        run(free_field, dict(CHAIN, elevation={}, dead_channels=[64]), "cpp")


def test_shallow_channel_direct_surface_and_bottom_arrivals_fold_to_the_targets_depth():
    """Image-source multipath: 100 m of water, array at 30 m, target at 50 m, 1 km. The simulator
    gives every bounce path its own arrival angle at each element (its image position), so the
    surface echo comes from above and the bottom echo from below, as geometry says."""
    records = run_scenario("popoto_elevation_shallow", {"pings": {"count": 2}}).records
    rec = records[0].copy()
    recon.prepare([rec])
    rows = elevation.find_rows(rec)
    frame = elevation.array_frame(rec, rows)
    subs = elevation.row_records(rec, rows)
    r0 = float(np.hypot(1000.0, 20.0))
    ranges = r0 + np.arange(-1.0, 9.0, 0.1875)
    el, power, quality = elevation.profile(rec, subs, frame, 0.0, ranges)
    # Same-path arrivals (out and back by the same path) at their image ranges: direct, surface, bottom.
    for name, image_z, tol_deg in (("direct", 50.0, 0.6), ("surface", -50.0, 1.0), ("bottom", 150.0, 1.5)):
        slant = float(np.hypot(1000.0, image_z - 30.0))
        want = np.degrees(np.arctan2(-(image_z - 30.0), 1000.0))
        near = np.abs(ranges - slant) < 0.2
        j = np.nonzero(near)[0][np.argmax(power[near])]
        assert power[j] > 0.1 * power.max(), name
        # Range sidelobes of the neighbouring arrivals (-13 dB, unweighted pulse) pull each by up to a degree or so.
        assert np.degrees(el[j]) == pytest.approx(want, abs=tol_deg), name
        z = frame.origin[2] - ranges[j] * np.sin(el[j])
        folded, path = elevation.fold_depth(z, 100.0)
        assert path == name and folded == pytest.approx(50.0, abs=1000.0 * np.radians(tol_deg)), name
    # The chain takes the first arrival of the cluster, the direct one, and reports the target's depth.
    for engine in ("python", "cpp"):
        last = run(records, dict(CHAIN, elevation={"water_depth": 100.0}), engine)[-1]
        best = max(last["detections"], key=lambda d: d["snr_db"])
        assert best["path"] == "direct" and best["depth_m"] == pytest.approx(50.0, abs=10.0), engine
        assert best["elevation_deg"] == pytest.approx(-1.15, abs=0.6), engine


# --- passive ----------------------------------------------------------------------------------------

def listening(seconds=2.0, **kw):
    sonar = sonar_rows(window=(0.0, 0.5 * C * seconds))
    pose = make_pose(position=(0.0, 0.0, 100.0), rpy=(np.radians(3.0), 0.0, 0.0))
    return passive_record(sonar, pose, [{"position": [-400.0, 900.0, 20.0], "rms": 1.0}, {"position": [300.0, 800.0, 250.0], "rms": 0.7}],
                          noise_rms=1.0, rng=np.random.default_rng(2), **kw)


def test_passive_elevation_of_sources_at_depth_numpy_and_cpp():
    rec = listening()
    rows = elevation.find_rows(rec)
    for source in ([-400.0, 900.0, 20.0], [300.0, 800.0, 250.0]):
        _, cone, el_true, _ = truth(rec, rows, source)
        ref, q_ref = elevation.passive_elevation(rec, rows, cone, 4000.0, 6000.0, n_fft=256, hop=128)
        got, q = core.live_passive_elevation(rec, rows, cone, 4000.0, 6000.0, n_fft=256, hop=128)
        assert got == pytest.approx(ref, abs=1e-5) and q == pytest.approx(q_ref, abs=1e-4)
        assert np.degrees(got - el_true) == pytest.approx(0.0, abs=0.5)
    frames, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, first_deg=-60.0, last_deg=60.0,
                                 max_sources=2, min_snr_db=3.0, elevation=True)
    assert len(frames) >= 5
    for f in frames:
        found = sorted(zip((np.degrees(p[0]) for p in f["peaks"]), np.degrees(f["elevations"])))
        assert len(found) == 2
        # Level-referenced (the 3 degree roll taken out): 4.6 degrees up to the shallow source, 9.9 down to the deep one.
        for (bearing, level), source in zip(found, ([-400.0, 900.0, 20.0], [300.0, 800.0, 250.0])):
            _, cone, _, level_true = truth(rec, rows, source)
            assert bearing == pytest.approx(np.degrees(cone), abs=0.6) and level == pytest.approx(np.degrees(level_true), abs=1.5)
    # Without the option the frames are as they were; one row has no elevation to give.
    plain, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, max_sources=2)
    assert "elevations" not in plain[0]


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")
def test_program_reports_elevation_actively_and_passively(free_field, tmp_path):
    w = gs.RecordWriter(str(tmp_path / "active.h5"))
    for r in free_field:
        w.write(r)
    w.close()
    args = ["bearing_first_deg=-50", "bearing_last_deg=50", "cfar_pfa=1e-6", "cfar_guard_m=20", "cfar_train_m=100", "min_snr_db=15",
            "tracker_range_sigma=5", "tracker_confirm_hits=2", "multipath=0", "elevation=1", "elevation_roll_sigma_deg=0.2"]
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "active.h5"), *args], capture_output=True, text=True, check=True)
    last = json.loads(out.stdout.splitlines()[-1])
    assert last["rows"] == 2 and sorted(round(t["depth_m"] / 10.0) for t in last["tracks"]) == [1, 6, 16]
    w = gs.RecordWriter(str(tmp_path / "passive.h5"))
    w.write(listening())
    w.close()
    out = subprocess.run([str(EXE), "--passive", str(tmp_path / "passive.h5"), "--band", "4000", "6000", "n_fft=256", "hop=128",
                          "n_average=12", "sbl_max_sources=2", "elevation=1"], capture_output=True, text=True, check=True)
    frames = [json.loads(line) for line in out.stdout.splitlines()]
    assert len(frames) >= 5 and all(len(f["elevations_deg"]) == len(f["bearings_deg"]) == 2 for f in frames)
    assert sorted(frames[0]["elevations_deg"]) == pytest.approx([-9.9, 4.6], abs=1.5)


def test_exported_command_carries_the_elevation_settings(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import sonar_live
    from gpusonar.config import load_yaml
    cmd = sonar_live.export_template(load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle_elevation.yaml"), str(tmp_path / "t.h5"))
    assert "elevation=1" in cmd and "elevation_first_deg=-30" in cmd and "elevation_roll_sigma_deg=0.2" in cmd
    assert gs.RecordReader(str(tmp_path / "t.h5")).read(0).samples.n_channels == 64
    live = pytest.importorskip("gpusonar.live")
    opts = live.cpp_chain_options(load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle_elevation.yaml")["chain"])
    core.LiveChainCpp(dict(opts, backend=core.Backend.Host))                       # every key is one the C++ chain takes
