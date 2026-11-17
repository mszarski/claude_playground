"""Stacked rows on the long-range array: elevation and depth as well as bearing, for every
stage of the build: one row and two rows on the test AUV, two and three rows on the 2 m x 9 m face.

Free-field truth and the Cramer-Rao bound; the repeats of rows several wavelengths apart, the
candidates a track chooses among and what a third row or the water depth does to them; the
cone-angle correction; depth with the vehicle rolled and rolling; local sound speed, water depth
and the unknown gradient in the depth sigma; a channel with surface and bottom bounces; dead
channels; calibration of a row out of place; passive sources; every shipped stage through
simulate -> chain; and the two chain engines against each other. One row must come out exactly
as before.
"""

import json
import pathlib
import subprocess

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import detect, elevation, flank, live, recon, sync
from gpusonar.config import list_configs, load_scenario_config, load_yaml
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, run_scenario, simulate_ping
from gpusonar.sim.passive import passive_record
from gpusonar.sonar import SonarModel

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "bin" / "sonar_live"
C = 1500.0
BASEBAND = {"kind": "baseband", "sample_rate": 8.0e3}
FACE2, FACE3, STATIC2, STATIC3 = "popoto_long_range_2row", "popoto_long_range_3row", "popoto_static_2row", "popoto_static_3row"
NEAR2 = "two rows 0.30 m apart"       # rows of the static frame a wavelength apart: no repeats
CHAIN = {"bearings_deg": [-50.0, 50.0, 0.5], "range_cell_m": 2.0, "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 100.0},
         "min_snr_db": 15.0, "backend": "host", "tracker": {"range_sigma": 5.0, "bearing_sigma_deg": 1.0, "confirm_hits": 2, "max_misses": 2},
         "classify": {"min_speed": 1.0, "static_speed": 0.6, "max_fit_rms": 10.0, "multipath": None}}
QUIET = {"gradient_sigma_per_s": 0.0}          # depth sigma from the angle alone


def sonar_rows(name=FACE3, window=(900.0, 1100.0), **over):
    if name == NEAR2:
        name = STATIC2
        over = dict(over, array=dict({"layouts": {"two_rows": {"rows": [{"up": -0.15}, {"up": 0.15}]}}}, **over.get("array", {})))
    return SonarModel.load(name, {"receive": dict(BASEBAND, window={"range": list(window)}),
                                  "transmit": {"waveform": {"duration": 1.0}}, **over})


def truth(record, rows, target):
    """(slant range, cone bearing, elevation in the array frame, level elevation) of a point."""
    frame = elevation.array_frame(record, rows)
    d = np.asarray(target, dtype=float) - frame.origin
    r = float(np.linalg.norm(d))
    u = d / r
    return r, float(np.arcsin(u @ frame.along)), float(np.arcsin(u @ frame.up)), float(np.arcsin(-u[2]))


def one_ping(sonar, target, *, pose=None, noise_rms=0.0, seed=1, ts=10.0, sound_speed=C):
    scene = Scene().add_points([target], target_strength_db=ts) if target is not None else Scene()
    return simulate_ping(sonar, pose or make_pose(position=(0.0, 0.0, 60.0)), scene, environment={"sound_speed": sound_speed},
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


def level_deg(world):
    return float(np.degrees(np.arcsin(-world[2])))


def at(bearing_deg, el_deg, r=1000.0, own=60.0):
    b, e = np.radians(bearing_deg), np.radians(el_deg)
    return np.array([r * np.cos(e) * np.sin(b), r * np.cos(e) * np.cos(b), own - r * np.sin(e)])


def run(records, chain, engine="python"):
    with LiveChain(dict(chain, engine=engine)) as chain_:
        return chain_.run([r.copy() for r in records])


def strongest(report):
    return max(report["detections"], key=lambda d: d["snr_db"] if d["snr_db"] is not None else -99.0)


# --- the static drop-in array -------------------------------------------------------------------------

def static_chain(scenario):
    """The static array's own deployment file, as a chain for baseband pings of a scenario's water."""
    chain = load_yaml(ROOT / "configs" / "deploy" / "popoto_static_3row.yaml")
    assert "static" in chain["nav"] and chain["chain"]["elevation"]["attitude_time"] == "transmit"      # a fixed pose: no history
    chain = chain["chain"]
    chain.pop("sync")
    chain["tracker"] = dict(chain["tracker"], confirm_hits=2)
    chain["elevation"] = dict(chain["elevation"], gradient_per_s=None)                                   # the simulated water is isovelocity
    if scenario["environment"].get("water_depth"):
        chain["elevation"]["water_depth"] = float(scenario["environment"]["water_depth"])
    return dict(chain, backend="host")


def test_static_array_reports_depths_in_free_field_and_tracks_a_passing_target():
    """The frame does not move. Free field, tilted 2 degrees: the three targets' depths. A target crossing at
    3 m/s in 30 m of water: one mover at that speed and near its depth, and nothing else moving."""
    sc = load_scenario_config("popoto_static_free")
    assert sc["platform"]["velocity"] == [0, 0, 0] and SonarModel.load(sc["sonar"]).config["platform"]["static"] is True
    truth_depths = sorted(float(o["positions"][0][2]) for o in sc["scene"])
    records = run_scenario(dict(sc, pings=dict(sc["pings"], count=3))).records
    for engine in ("python", "cpp"):
        last = run(records, static_chain(sc), engine)[-1]
        assert last["rows"] == 3 and sorted(t["depth_m"] for t in last["tracks"]) == pytest.approx(truth_depths, abs=1.0), engine
        assert all(t["kind"] != "mover" and t["speed_m_s"] < 0.5 for t in last["tracks"]), engine
        assert all(d["elevation_sigma_deg"] >= 1.0 for d in last["detections"]), engine      # the frame's assumed tilt accuracy is owned up to
    sc = load_scenario_config("popoto_static_passing")
    mover = sc["scene"][0]
    records = run_scenario(dict(sc, pings=dict(sc["pings"], count=6))).records
    for engine in ("python", "cpp"):
        reps = run(records, static_chain(sc), engine)
        assert np.allclose([r["ownship_xyz"] for r in reps], reps[0]["ownship_xyz"]), engine           # the array stays where it is
        tracks = reps[-1]["tracks"]
        movers = [t for t in tracks if t["kind"] == "mover"]
        assert len(movers) == 1 and all(t["kind"] in ("mover", "multipath") for t in tracks), engine
        assert movers[0]["vx"] == pytest.approx(mover["velocity"][0], abs=0.5) and abs(movers[0]["vy"]) < 0.5, engine
        assert movers[0]["depth_m"] == pytest.approx(mover["positions"][0][2], abs=6.0), engine


def test_static_array_hears_the_elevation_of_noise_sources():
    """Listening with the three-row frame, tilted 2 degrees (sim/passive.py: free field, no ping)."""
    sonar = sonar_rows(STATIC3, window=(0.0, 1500.0))
    pose = make_pose(position=(0.0, 0.0, 40.0), rpy=(np.radians(2.0), 0.0, 0.0))
    sources = [[-200.0, 500.0, 10.0], [150.0, 450.0, 110.0]]
    rec = passive_record(sonar, pose, [{"position": sources[0], "rms": 1.0}, {"position": sources[1], "rms": 0.7}], noise_rms=1.0,
                         rng=np.random.default_rng(2))
    rows = elevation.find_rows(rec)
    frames, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, first_deg=-60.0, last_deg=60.0,
                                 max_sources=2, min_snr_db=3.0, elevation=True, elevation_step_deg=0.25)
    both = [f for f in frames if len(f["peaks"]) == 2]
    assert len(frames) >= 5 and len(both) >= 0.8 * len(frames)
    for f in both:
        found = sorted(zip((np.degrees(p[0]) for p in f["peaks"]), np.degrees(f["elevations"])))
        for (bearing, level), source in zip(found, sources):
            _, cone, _, level_true = truth(rec, rows, source)
            assert bearing == pytest.approx(np.degrees(cone), abs=1.0) and level == pytest.approx(np.degrees(level_true), abs=0.8)


# --- layout and rows --------------------------------------------------------------------------------

def test_stage_layouts_put_rows_at_their_heights_and_boards_follow():
    one, two, three = (SonarModel.load(n) for n in ("popoto_long_range", FACE2, FACE3))
    assert (two.n_elements, len(two.boards), three.n_elements, len(three.boards)) == (64, 8, 96, 12)
    assert np.allclose(two.positions[:32, [0, 1]], one.positions[:, [0, 1]])          # each row is the one-row array
    assert np.allclose(two.positions[:32, 2], 0.75) and np.allclose(two.positions[32:, 2], -0.75)   # z is down: the first row is the lower
    assert [float(three.positions[k, 2]) for k in (0, 32, 64)] == pytest.approx([0.75, 0.20, -0.75])
    # Inside the owner's face (2 m high, 9 m long) and inside the working values: 7 m of its length, outer rows 1.5 m apart.
    for m in (two, three):
        plat = m.config["platform"]
        assert (plat["span_m"], plat["height_m"], plat["usable_span_m"], plat["row_baseline_m"]) == (9.0, 2.0, 7.0, 1.5)
        assert np.ptp(m.positions[:, 2]) == pytest.approx(plat["row_baseline_m"]) and np.ptp(m.positions[:, 0]) <= plat["usable_span_m"]
        tx = np.asarray(m.config["transmit"]["lever_arm"])                       # the projector is on the face, clear of the rows
        assert abs(tx[0]) <= plat["span_m"] / 2 and abs(tx[2]) <= plat["height_m"] / 2 and abs(tx[0]) > np.abs(m.positions[:, 0]).max() + 0.5
    # The test AUV: 32 hydrophones over 3 m (2 m by override), ONE row. The static drop-in array: two or three rows.
    p1 = SonarModel.load("popoto_testauv_1row")
    assert (p1.n_elements, len(p1.boards)) == (32, 4) and 32 * p1.config["array"]["pitch"] <= p1.config["platform"]["span_m"]
    assert "popoto_testauv_2row" not in list_configs("sonars")
    s2, s3 = SonarModel.load(STATIC2), SonarModel.load(STATIC3)
    assert (s2.n_elements, len(s2.boards), s3.n_elements, len(s3.boards)) == (64, 8, 96, 12)
    for m in (s2, s3):
        plat = m.config["platform"]
        assert plat["static"] is True and np.ptp(m.positions[:, 0]) <= plat["span_m"] and np.ptp(m.positions[:, 2]) <= plat["height_m"]
        assert np.allclose(m.positions[:32, [0, 1]], p1.positions[:, [0, 1]])     # each row is the test AUV's row
    assert sorted(set(np.round(-s3.positions[:, 2], 3))) == pytest.approx([-0.75, -0.20, 0.75])
    assert np.ptp(sonar_rows(NEAR2).positions[:, 2]) == pytest.approx(0.30)
    short = SonarModel.load("popoto_testauv_1row", {"array": {"pitch": 0.0625}})
    assert np.ptp(short.positions[:, 0]) == pytest.approx(31 * 0.0625)
    # Four from each row in turn: a board of eight inputs then holds the same stations of both rows.
    inter = SonarModel.load(FACE2, {"array": {"interleave": 4}})
    assert np.allclose(inter.positions[:4], two.positions[:4]) and np.allclose(inter.positions[4:8], two.positions[32:36])
    assert sorted(map(tuple, np.round(inter.positions, 6))) == sorted(map(tuple, np.round(two.positions, 6)))
    for name, boards in (("popoto_testauv_1row_clocks", 4), ("popoto_static_2row_clocks", 8), ("popoto_static_3row_clocks", 12),
                         ("popoto_long_range_2row_clocks", 8), ("popoto_long_range_3row_clocks", 12)):
        m = SonarModel.load(name)
        assert len(m.boards) == boards and len(m.config["receive"]["reference"]["channels"]) == boards


def test_rows_are_found_from_the_positions_whatever_the_wiring_and_with_channels_missing():
    rec = sonar_rows(FACE2).make_record(make_pose())
    assert elevation.find_rows(rec) == [list(range(32)), list(range(32, 64))] == core.live_find_rows(rec)
    assert len(elevation.find_rows(sonar_rows("popoto_long_range").make_record(make_pose()))) == 1
    assert [len(r) for r in elevation.find_rows(sonar_rows(FACE3).make_record(make_pose()))] == [32, 32, 32]
    inter = sonar_rows(FACE2, array={"interleave": 4}).make_record(make_pose())
    rows = elevation.find_rows(inter)
    assert rows == core.live_find_rows(inter) and rows[0][:8] == [0, 1, 2, 3, 8, 9, 10, 11] and rows[1][:4] == [4, 5, 6, 7]
    # A dead hydrophone and a dead board (eight inputs of the upper row) leave two usable rows.
    less = sync.select_channels(rec, [r for r in range(64) if r != 5 and not 40 <= r < 48])
    rows = elevation.find_rows(less)
    assert [len(r) for r in rows] == [31, 24] and rows == core.live_find_rows(less)
    frame = elevation.array_frame(less, rows)
    assert np.allclose(frame.along, [1.0, 0.0, 0.0]) and np.allclose(frame.up, [0.0, 0.0, -1.0]) and np.allclose(frame.look, [0.0, 1.0, 0.0])


# --- accuracy ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [NEAR2, STATIC3, FACE2, FACE3])
def test_elevation_is_exact_without_noise_whatever_the_roll_and_the_projector_offset(name):
    """The projector is not at the array's centre: the search must hold the two-way path, not the
    range from the array, or the pulse's envelope drags the peak (an early version read 1.6 degrees off)."""
    for roll_deg, target in ((0.0, [200.0, 980.0, 30.0]), (4.0, [200.0, 980.0, 110.0]), (-6.0, [-400.0, 900.0, 60.0])):
        rec = one_ping(sonar_rows(name), target, pose=make_pose(position=(0.0, 0.0, 60.0), rpy=(np.radians(roll_deg), 0.0, 0.0)))
        est, el_true, frame = estimate_one(rec, target)
        assert abs(np.degrees(np.arcsin(est.sin_elevation) - el_true)) < 0.02
        assert est.quality > 0.999
        assert frame.origin[2] + est.range * est.world[2] == pytest.approx(target[2], abs=0.5)    # depth, roll taken out


@pytest.mark.parametrize("name", [NEAR2, FACE3])
def test_elevation_is_unbiased_and_meets_the_cramer_rao_bound_in_noise(name):
    sonar = sonar_rows(name, window=(960.0, 1040.0))
    target = [100.0, 1000.0, 110.0]
    clean = one_ping(sonar, target)
    recon.prepare([clean])
    peak = float((np.abs(clean.samples.array) ** 2).max(axis=1).mean())           # echo power at a hydrophone
    rows = elevation.find_rows(clean)
    heights = elevation.array_frame(clean, rows).heights
    for snr_db, factor in ((3.0, 1.3), (13.0, 1.25)):                             # at one hydrophone: 18 and 28 dB after a row's beamforming
        noise = np.sqrt(peak / 10.0 ** (snr_db / 10.0))
        quiet = one_ping(sonar, None, noise_rms=1.0, seed=99)
        recon.prepare([quiet])
        # noise_rms that gives that level after compression, where the replica overlaps the data fully
        scale = noise / float(np.sqrt((np.abs(quiet.samples.array[:, :flank._n_valid(quiet)]) ** 2).mean()))
        bound = np.degrees(elevation.sin_elevation_sigma(heights, 10.0 ** (snr_db / 10.0), 4000.0, 6000.0, C))
        errors, sigmas, flagged = [], [], 0
        for seed in range(60):
            rec = one_ping(sonar, target, noise_rms=scale, seed=seed)
            est, el_true, _ = estimate_one(rec, target, background=len(heights) * noise**2)
            errors.append(np.degrees(np.arcsin(est.sin_elevation) - el_true))
            sigmas.append(np.degrees(est.sigma))
            flagged += est.ambiguous
        errors = np.array(errors)
        if name == FACE3:
            assert np.abs(errors).max() < 3.0                                     # never another lobe
        assert abs(errors.mean()) < 3.0 * bound / np.sqrt(len(errors))            # unbiased
        assert bound / factor < errors.std() < bound * factor                     # at the bound, within the scatter of 60 trials
        assert np.median(sigmas) == pytest.approx(bound, rel=0.25)                # and the sigma it reports is that bound
        assert flagged <= 3
    # Outer rows 1.5 m apart: five times finer than a 0.30 m pair at the same SNR.
    if name == FACE3:
        assert bound == pytest.approx(0.070, abs=0.005)                            # degrees at 28 dB a row (0.30 m apart: 0.36)


def test_bound_formula_and_unambiguous_sector():
    # Two rows d apart, one row's SNR rho_row: sigma(sin e) = lambda_eff / (pi d sqrt(2 rho_row)).
    d, rho_row = 1.8, 100.0
    heights = np.repeat([-d / 2, d / 2], 32)
    k = elevation.rms_wavenumber(4000.0, 6000.0, C)
    assert k == pytest.approx(2 * np.pi / C * np.sqrt(5000.0**2 + 2000.0**2 / 12))
    assert elevation.sin_elevation_sigma(heights, rho_row / 32, 4000.0, 6000.0, C) == pytest.approx(1.0 / (k * (d / 2) * np.sqrt(2 * 2 * rho_row)))
    assert elevation.unambiguous_elevation(0.15, 5000.0) == pytest.approx(np.pi / 2)       # half a wavelength: no second lobe
    assert np.degrees(elevation.unambiguous_elevation(1.8, 5000.0)) == pytest.approx(4.8, abs=0.05)
    assert np.degrees(elevation.cone_bias(np.radians(30.0), np.radians(10.0))) == pytest.approx(0.512, abs=0.002)
    assert elevation.refraction_sigma(3000.0, C, 0.017) == pytest.approx(51.0)             # isothermal water, 3 km


def test_two_rows_on_the_face_offer_candidates_and_three_rows_or_a_bounded_sector_decide():
    target = at(10.0, 5.0)                                                          # 5 degrees up, 1 km
    rec = one_ping(sonar_rows(FACE2), target)
    est, el_true, _ = estimate_one(rec, target)
    # Rows 1.5 m apart repeat every 11.5 degrees; the band leaves the nearest repeats 0.6 dB down. Never trusted alone.
    assert est.ambiguous and len(est.alternatives) >= 2
    levels = sorted([level_deg(est.world)] + [level_deg(w) for w in est.alternatives])
    assert min(abs(v - 5.0) for v in levels) < 0.05                                # the truth is among the candidates
    assert np.diff(levels) == pytest.approx(11.5, abs=1.2)                         # 9.6 to 14.5 degrees over the band; 11.5 at 5 kHz
    # Where nothing can arrive from outside half a repeat each side, there is one candidate.
    est, el_true, _ = estimate_one(one_ping(sonar_rows(FACE2), target), target, cfg={"sector_deg": [0.5, 9.5, 0.25]})
    assert not est.ambiguous and abs(np.degrees(np.arcsin(est.sin_elevation) - el_true)) < 0.05
    # The third row, 0.55 m above the lowest, leaves the highest repeat 4 dB down: one answer at any useful SNR.
    est, el_true, _ = estimate_one(one_ping(sonar_rows(FACE3), target), target)
    assert not est.ambiguous and est.alternatives == () and abs(np.degrees(np.arcsin(est.sin_elevation) - el_true)) < 0.05
    # In noise the margin is weighed against the noise: three rows at 6 dB a row are flagged, at 20 dB not.
    clean = one_ping(sonar_rows(FACE3), target)
    recon.prepare([clean])
    peak = float((np.abs(clean.samples.array) ** 2).max())
    for row_snr_db, want in ((6.0, True), (20.0, False)):
        est, _, _ = estimate_one(one_ping(sonar_rows(FACE3), target), target, background=96.0 * peak * 32.0 / 10.0 ** (row_snr_db / 10.0))
        assert est.ambiguous is want
    # Rows 0.30 m apart (a wavelength) repeat far outside +-25 degrees: no candidates.
    est, el_true, _ = estimate_one(one_ping(sonar_rows(NEAR2), target), target, cfg={"sector_deg": [-25.0, 25.0, 0.25]})
    assert not est.ambiguous and abs(np.degrees(np.arcsin(est.sin_elevation) - el_true)) < 0.05


def test_calibration_finds_a_row_mounted_out_of_place_and_removes_the_elevation_bias():
    """The upper row 5 mm further out than the drawing says reads as 0.19 degrees of elevation with rows
    1.5 m apart (0.75 degrees with 0.38 m). Sources at several bearings AND elevations give the existing
    calibration (calibrate.py, planar=False) each hydrophone's offset in three dimensions."""
    from gpusonar import calibrate
    sonar = sonar_rows(FACE2, window=(960.0, 1040.0))
    rng = np.random.default_rng(3)
    position = np.zeros((64, 3))
    position[32:, 1] += 5e-3
    errors = {"gain": np.ones(64), "phase": np.radians(rng.normal(0.0, 3.0, 64)), "position": position}

    def ping(target, err):
        return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 60.0)), Scene().add_points([target], target_strength_db=20.0),
                             environment={"sound_speed": C}, absorption_model="none", adc=False, channel_errors=err)

    sources = [at(b, e) for b, e in ((-30.0, -10.0), (-10.0, 12.0), (0.0, 0.0), (10.0, -12.0), (30.0, 10.0), (20.0, 0.0))]
    est = calibrate.estimate_channel_errors([ping(src, errors) for src in sources], sources, planar=False)
    assert est.position[32:, 1].mean() - est.position[:32, 1].mean() == pytest.approx(5e-3, abs=0.4e-3)
    target = at(5.0, 2.0)
    narrow = {"sector_deg": [-2.0, 6.0, 0.1]}
    raw, el_true, _ = estimate_one(ping(target, errors), target, cfg=narrow)
    assert abs(np.degrees(np.arcsin(raw.sin_elevation) - el_true)) == pytest.approx(np.degrees(5e-3 / 1.5), abs=0.06)    # the bias
    rec = ping(target, errors)
    recon.prepare([rec])
    calibrate.store_calibration(rec, est)
    core.apply_calibration(rec)
    fixed, el_true, _ = estimate_one(rec, target, cfg=narrow)
    assert abs(np.degrees(np.arcsin(fixed.sin_elevation) - el_true)) < 0.03


# --- depth ------------------------------------------------------------------------------------------

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
    # "Some idea of water depth": a bottom known to 10 m does not condemn a depth 25 m below it at 2 sigma.
    assert not elevation.outside_water(125.0, 10.0, 100.0, water_depth_sigma=10.0)


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
    # The assumed-gradient profile is that same line through the vehicle's own sound speed.
    depths, speeds = elevation.gradient_profile(z0, C, g, 300.0)
    assert np.interp(z0, depths, speeds) == pytest.approx(C) and (speeds[1] - speeds[0]) / (depths[1] - depths[0]) == pytest.approx(g)
    assert elevation.ray_depth(z0, -theta, 3000.0, depths, speeds, step=0.5) == pytest.approx(z, abs=1e-6)
    assert core.live_ray_depth(z0, -theta, 3000.0, depths, speeds) == pytest.approx(elevation.ray_depth(z0, -theta, 3000.0, depths, speeds), abs=1e-9)


def test_tracker_carries_depth_and_chooses_among_an_ambiguous_detections_candidates():
    tracker = detect.Tracker(confirm_hits=2, depth_process_noise=0.0)
    for i, (depth, sigma) in enumerate(((40.0, 4.0), (float("nan"), float("nan")), (60.0, 4.0))):
        tracks = tracker.update(float(i), [detect.Detection(500.0, 0.1, 20.0, 1, depth, sigma)])
    assert tracks[0].depth == pytest.approx(50.0) and tracks[0].depth_var == pytest.approx(8.0)   # two equal measurements
    # An ambiguous detection (repeats 84 m apart at 500 m): the candidate that fits the track is taken ...
    tracks = tracker.update(3.0, [detect.Detection(500.0, 0.1, 20.0, 1, 136.0, 4.0, depth_alternatives=[52.0, -32.0])])
    assert tracks[0].depth == pytest.approx(50.0 + (52.0 - 50.0) * 8.0 / 24.0)
    # ... none of them when none fits ...
    before = tracks[0].depth
    tracks = tracker.update(4.0, [detect.Detection(500.0, 0.1, 20.0, 1, 136.0, 4.0, depth_alternatives=[220.0])])
    assert tracks[0].depth == before
    # ... and a track that has no depth yet does not start on a guess.
    fresh = detect.Tracker(confirm_hits=1)
    fresh.update(0.0, [detect.Detection(500.0, 0.1, 20.0, 1, 136.0, 4.0, depth_alternatives=[52.0])])
    assert np.isnan(fresh.tracks[0].depth)


# --- the chain --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def free_field():
    return run_scenario("popoto_elevation_free").records


def test_chain_reports_depth_with_the_roll_taken_out_and_the_bearing_corrected(free_field):
    chain = dict(CHAIN, elevation=dict(QUIET, roll_sigma_deg=0.2))
    last = run(free_field, chain)[-1]
    assert last["rows"] == 3
    targets = {(-300.0, 900.0): 10.0, (100.0, 1000.0): 60.0, (500.0, 1100.0): 160.0}     # the scenario's, with their depths
    assert len(last["tracks"]) == 3
    for t in last["tracks"]:
        (x, y), depth = min(targets.items(), key=lambda kv: np.hypot(kv[0][0] - t["x"], kv[0][1] - t["y"]))
        assert np.hypot(x - t["x"], y - t["y"]) < 5.0
        assert t["depth_m"] == pytest.approx(depth, abs=2.0) and t["path"] == "direct" and t["depth_sigma_m"] < 3.0
    for d in last["detections"]:
        assert d["elevation_quality"] > 0.99 and not d["elevation_ambiguous"] and "elevation_alternatives_deg" not in d
        assert d["elevation_sigma_deg"] == pytest.approx(0.2, abs=0.02)            # the navigation's roll accuracy is all of it here
    deep = max(last["detections"], key=lambda d: d["depth_m"])
    assert abs(deep["cone_bearing_deg"] - deep["bearing_deg"]) > 0.05              # 4.8 degrees down at 23 degrees off broadside: 0.09 degrees
    # Each use of the elevation has its own switch.
    plain = run(free_field, dict(CHAIN, elevation={"correct_bearing": False, "depth": False}))[-1]
    assert all("cone_bearing_deg" not in d and "depth_m" not in d and "elevation_deg" in d for d in plain["detections"])
    assert all("depth_m" not in t for t in plain["tracks"])
    assert max(plain["detections"], key=lambda d: d["range_m"])["bearing_deg"] == pytest.approx(deep["cone_bearing_deg"], abs=1e-6)


def test_depth_sigma_owns_up_to_the_gradient_nobody_measured(free_field):
    """No long-range profile: a mean gradient known only to 0.05 /s is range^2 x 0.05 / 2c of depth, 17 m at 1 km.
    It is in the reported sigma of detections and of tracks (which cannot average it away), not in the filter."""
    quiet = run(free_field, dict(CHAIN, elevation=QUIET))[-1]
    honest = run(free_field, dict(CHAIN, elevation={}))[-1]                         # the default prior: 0.05 /s
    for a, b in zip(quiet["detections"], honest["detections"]):
        bent = a["range_m"] ** 2 * 0.05 / (2 * C)
        assert b["depth_refraction_sigma_m"] == pytest.approx(bent, rel=0.01) and a["depth_refraction_sigma_m"] == 0.0
        assert b["depth_sigma_m"] == pytest.approx(np.hypot(a["depth_sigma_m"], b["depth_refraction_sigma_m"]), rel=1e-6) and b["depth_m"] == a["depth_m"]
    for a, b in zip(quiet["tracks"], honest["tracks"]):
        assert b["depth_m"] == pytest.approx(a["depth_m"], abs=1e-9) and b["depth_sigma_m"] > 10.0 > a["depth_sigma_m"]
    # An assumed gradient moves the depth itself (the ray through c = local speed + g (z - own depth)); both engines.
    for engine in ("python", "cpp"):
        bent = run(free_field, dict(CHAIN, elevation=dict(QUIET, gradient_per_s=0.05)), engine)[-1]
        for a, b in zip(quiet["detections"], bent["detections"]):
            assert b["depth_m"] == pytest.approx(abs(a["depth_m"] - a["range_m"] ** 2 * 0.05 / (2 * C)), abs=1.2), engine
    # A measured profile, when there is one, does the same through `ssp`.
    profile = {"depth": [0.0, 400.0], "speed": [C - 0.05 * 60.0, C + 0.05 * 340.0]}
    for a, b in zip(bent["detections"], run(free_field, dict(CHAIN, elevation=dict(QUIET, ssp=profile)), "cpp")[-1]["detections"]):
        assert b["depth_m"] == pytest.approx(a["depth_m"], abs=0.5)


def test_bearing_is_corrected_for_the_cone_angle_in_both_engines():
    """10 degrees up at 30 degrees off broadside: one line reports the cone angle, 0.5 degrees short
    (105 m of cross-range at 12 km). With the elevation the horizontal bearing is recovered."""
    sonar = sonar_rows(FACE3, window=(800.0, 1200.0))
    el, bearing = np.radians(10.0), np.radians(30.0)
    target = at(30.0, 10.0)
    rec = one_ping(sonar, target, noise_rms=1e-6)
    cone = np.degrees(np.arcsin(np.cos(el) * np.sin(bearing)))                     # 29.50
    for engine in ("python", "cpp"):
        for correct in (True, False):
            d = strongest(run([rec], dict(CHAIN, elevation={"correct_bearing": correct}), engine)[0])
            assert d["elevation_deg"] == pytest.approx(10.0, abs=0.05) and d["image_depth_m"] == pytest.approx(target[2], abs=1.0)
            if correct:
                # The map's own bearing is good to a tenth of a degree; the correction is exact on top of it.
                assert d["bearing_deg"] == pytest.approx(30.0, abs=0.15) and d["cone_bearing_deg"] == pytest.approx(cone, abs=0.15)
                assert d["bearing_deg"] - d["cone_bearing_deg"] == pytest.approx(30.0 - cone, abs=0.01)
            else:
                assert d["bearing_deg"] == pytest.approx(cone, abs=0.15) and "cone_bearing_deg" not in d
    assert np.degrees(elevation.cone_bias(np.arcsin(np.cos(el) * np.sin(bearing)), el)) == pytest.approx(30.0 - cone, abs=1e-9)


def rolling_ping(sonar, target, roll0_deg, rate_deg_s):
    """A ping from a vehicle that rolls while the echo is on its way: the echo is made with the turning
    array; the record then holds what navigation gives a processor, the attitude at the transmit and
    samples of it afterwards, not the rate."""
    pose = make_pose(position=(0.0, 0.0, 60.0), rpy=(np.radians(roll0_deg), 0.0, 0.0), rates=(np.radians(rate_deg_s), 0.0, 0.0))
    rec = one_ping(sonar, target, pose=pose, noise_rms=1e-6)
    out = []
    for t in np.arange(0.0, 3.0, 0.25):
        a = core.InsSample()
        a.t, a.rpy = float(t), (float(np.radians(roll0_deg + rate_deg_s * t)), 0.0, 0.0)
        out.append(a)
    nav = rec.navigation
    nav.ins = out
    rec.navigation = nav
    g = rec.geometry
    for name in ("tx_pose", "rx_pose"):
        q = getattr(g, name)
        q.rates = (0.0, 0.0, 0.0)
        setattr(g, name, q)
    rec.geometry = g
    return rec


def test_attitude_at_the_echos_arrival_removes_the_bias_of_a_rolling_vehicle():
    """2 degrees a second of roll and a target at 1 km: the 1 s echo arrives from 1.33 s after the ping, its middle
    3.67 degrees of roll later. With the attitude of the transmit that is 64 m of depth; with the attitude at
    the arrival, none."""
    sonar = sonar_rows(FACE3, window=(800.0, 1200.0))
    target = at(8.0, -2.0)                                                          # 35 m below the vehicle
    rec = rolling_ping(sonar, target, 1.0, 2.0)
    flight = 2.0 * 1000.0 / C + 0.5                                                 # to the middle of the 1 s pulse
    bias = 1000.0 * np.radians(2.0 * flight)
    for engine in ("python", "cpp"):
        then = strongest(run([rec], dict(CHAIN, elevation=dict(QUIET, attitude_time="transmit")), engine)[0])
        now = strongest(run([rec], dict(CHAIN, elevation=dict(QUIET, attitude_time="arrival")), engine)[0])
        assert abs(then["image_depth_m"] - target[2]) == pytest.approx(bias, rel=0.05), engine
        assert now["image_depth_m"] == pytest.approx(target[2], abs=3.0), engine     # what is left: the roll during the 1 s pulse itself
        assert now["elevation_deg"] == pytest.approx(-2.0, abs=0.15), engine
    # Without attitude samples the arrival-time attitude cannot be had: the transmit's is used and the reported
    # sigma grows by the rate the vehicle may have rolled at, times the echo's time of flight.
    bare = rec.copy()
    nav = bare.navigation
    nav.ins = []
    bare.navigation = nav
    cfg = dict(QUIET, attitude_time="arrival", attitude_rate_sigma_deg_s=1.0)
    a, b = (strongest(run([bare], dict(CHAIN, elevation=cfg), engine)[0]) for engine in ("python", "cpp"))
    assert a["elevation_sigma_deg"] == pytest.approx(1.0 * flight, rel=0.02) and b["elevation_sigma_deg"] == pytest.approx(a["elevation_sigma_deg"], rel=1e-3)
    assert abs(a["image_depth_m"] - target[2]) == pytest.approx(bias, rel=0.05)
    # The interpolation itself, and the history a navigation listener writes into a record.
    rpy, gap = elevation.attitude_at(rec, 1.1)
    assert np.degrees(rpy[0]) == pytest.approx(1.0 + 2.2) and gap == 0.0
    rpy, gap = elevation.attitude_at(rec, 4.0)
    assert np.degrees(rpy[0]) == pytest.approx(1.0 + 2.0 * 2.75) and gap == pytest.approx(1.25)
    assert elevation.attitude_at(bare, 1.0) == (None, None)


def test_navigation_brings_attitude_history_sound_speed_and_water_depth_into_the_record():
    nav = live.NavListener()
    t0 = 1_700_000_000_000_000_000
    for i in range(8):
        nav.push({"time_ns": t0 + int(0.5e9 * i), "x": 1.0 * i, "z": 30.0, "roll": 0.01 * i, "sound_speed": 1487.0, "water_depth": 92.0})
    rec = sonar_rows(FACE2).make_record(make_pose(), time_ns=t0 + int(1e9))
    with LiveChain(dict(CHAIN, elevation={}), nav=nav) as chain:
        fix = nav.at(rec.transmit.time_ns)
        live.apply_pose(rec, *fix)
        live.apply_environment(rec, *nav.extras(rec.transmit.time_ns))
        live.apply_history(rec, nav.history(rec.transmit.time_ns, rec.transmit.time_ns + int(2e9)))
        assert chain.nav is nav
    assert rec.environment.sound_speed == 1487.0 and rec.environment.water_depth == 92.0
    assert [round(a.t, 3) for a in rec.navigation.ins] == [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    rpy, gap = elevation.attitude_at(rec, 0.75)                                    # 1.75 s after the first fix: between the fourth and fifth
    assert rpy[0] == pytest.approx(0.035) and gap == 0.0
    # Datagrams without the two fields leave the record's own values alone.
    plain = live.NavListener(static={"x": 0.0, "z": 10.0})
    assert all(np.isnan(v) for v in plain.extras(0)) and len(plain.history(0, 10)) == 1


def test_local_sound_speed_from_config_replaces_the_records_in_both_engines():
    """The vehicle measures the sound speed where it is. A record whose environment says 1500 m/s while the
    water is at 1470 puts a target at 1 km 20 m too far; `chain.sound_speed` puts it back."""
    sonar = sonar_rows("popoto_long_range", window=(800.0, 1200.0))
    rec = one_ping(sonar, [0.0, 1000.0, 60.0], noise_rms=1e-6, sound_speed=1470.0)
    env = rec.environment
    env.sound_speed = 1500.0
    rec.environment = env
    for engine in ("python", "cpp"):
        assert strongest(run([rec], CHAIN, engine)[0])["range_m"] == pytest.approx(1000.0 * 1500.0 / 1470.0, abs=1.0), engine
        assert strongest(run([rec], dict(CHAIN, sound_speed=1470.0), engine)[0])["range_m"] == pytest.approx(1000.0, abs=1.0), engine


def test_cpp_chain_gives_the_python_chains_elevation_and_depth(free_field):
    chain = dict(CHAIN, elevation={"roll_sigma_deg": 0.2, "water_depth": 300.0, "water_depth_sigma_m": 5.0})
    want, got = run(free_field, chain), run(free_field, chain, "cpp")
    assert set(want[-1]) == set(got[-1])
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"]) == 3
        for a, b in zip(w["detections"], g["detections"]):
            assert set(a) == set(b)
            for key, tol in (("range_m", 0.05), ("bearing_deg", 0.01), ("cone_bearing_deg", 0.01), ("elevation_deg", 0.01),
                             ("elevation_sigma_deg", 0.002), ("elevation_quality", 0.001), ("depth_m", 0.2), ("depth_sigma_m", 0.05),
                             ("depth_refraction_sigma_m", 0.01), ("image_depth_m", 0.2)):
                assert a[key] == pytest.approx(b[key], abs=tol), key
            assert (a["path"], a["elevation_ambiguous"]) == (b["path"], b["elevation_ambiguous"])
        assert [(t["id"], t["kind"], t.get("path")) for t in w["tracks"]] == [(t["id"], t["kind"], t.get("path")) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            assert set(a) == set(b)
            for key in ("x", "y", "depth_m", "depth_sigma_m"):
                assert a[key] == pytest.approx(b[key], abs=0.2), key


def test_two_rows_on_the_face_candidates_reach_the_track_and_the_water_depth_decides():
    """Two rows on the face. 100 m of water, a target at 60 m depth 3 km away: every arrival is within 4 degrees of level,
    less than half a repeat (5.7 degrees). Searching only where the water lets an echo come from, two rows
    give one answer; searching +-30 degrees they give candidates, which a track that has a depth chooses among."""
    records = run_scenario("popoto_face_2row_shallow").records
    for engine in ("python", "cpp"):
        wide = run(records, dict(CHAIN, elevation=dict(QUIET, water_depth=100.0)), engine)
        d = strongest(wide[-1])
        assert d["elevation_ambiguous"] and len(d["elevation_alternatives_deg"]) >= 2, engine
        assert all(t["depth_m"] is None for t in wide[-1]["tracks"]), engine         # no track starts on a guess
        bounded = run(records, dict(CHAIN, elevation=dict(QUIET, water_depth=100.0, sector_from_depth={"bounces": 1, "margin_deg": 1.0})), engine)
        d = strongest(bounded[-1])
        assert not d["elevation_ambiguous"] and d["path"] == "direct", engine
        # Direct path from 0.57 degrees below. The echoes with one surface or bottom bounce on one leg arrive 0.6 to 0.9 m
        # behind it (two range cells) and pull the estimate by up to half a degree, 26 m at 3 km; by how much depends on
        # their phases, so on the geometry to a fraction of a wavelength (with the projector 1.5 m below the array's
        # centre instead of beside it the same target reads 48 to 53 m). In free field it reads 60 m.
        assert d["elevation_deg"] == pytest.approx(-0.57, abs=0.7) and d["depth_m"] == pytest.approx(60.0, abs=36.0), engine
        track = bounded[-1]["tracks"][0]
        assert track["depth_m"] == pytest.approx(60.0, abs=36.0) and track["path"] == "direct", engine
    assert strongest(wide[-1])["elevation_alternatives_deg"] == pytest.approx(strongest(run(records, dict(CHAIN, elevation=dict(QUIET, water_depth=100.0)))[-1])["elevation_alternatives_deg"], abs=0.02)


def test_interleaved_wiring_gives_the_same_depths():
    """Four hydrophones of each row per board (the rows of a board share its clock): the chain finds
    the rows from the positions, so the channel order does not matter."""
    sc = load_scenario_config("popoto_elevation_free")
    sc["sonar_overrides"]["array"] = {"interleave": 4}
    sc["pings"]["count"] = 3
    records = run_scenario(sc).records
    for engine in ("python", "cpp"):
        last = run(records, dict(CHAIN, elevation={}), engine)[-1]
        assert last["rows"] == 3 and sorted(t["depth_m"] for t in last["tracks"]) == pytest.approx([10.0, 60.0, 160.0], abs=2.0), engine


def test_one_row_is_processed_exactly_as_without_the_option():
    """The option on a one-row array, and the option off, give the same report lines in both engines:
    the long-range row and the prototype's."""
    for name in ("popoto_long_range", "popoto_testauv_1row"):
        sonar = sonar_rows(name, window=(800.0, 1200.0))
        rng = np.random.default_rng(4)
        pings = [simulate_ping(sonar, make_pose(position=(1.5 * i, 0.0, 30.0), velocity=(1.5, 0.0, 0.0)),
                               Scene().add_points([[100.0, 1000.0, 80.0], [-200.0, 950.0, 30.0]], target_strength_db=10.0),
                               environment={"sound_speed": C}, sequence=i, time_ns=int(4e9 * i), noise_rms=1e-4, rng=rng) for i in range(3)]
        for engine in ("python", "cpp"):
            off, on = run(pings, CHAIN, engine), run(pings, dict(CHAIN, elevation={"attitude_time": "arrival", "gradient_per_s": 0.017}), engine)
            for a, b in zip(off, on):
                a.pop("processing_seconds"), b.pop("processing_seconds")
                assert json.dumps(a) == json.dumps(b)
            assert len(off[-1]["detections"]) == 2 and len(off[-1]["tracks"]) == 2 and "rows" not in on[-1]


def test_dead_channels_and_a_dead_board_do_not_break_the_estimate(free_field):
    dead = [5] + list(range(40, 48))                       # one hydrophone of the lower row, one board of the middle row
    chain = dict(CHAIN, elevation={}, dead_channels=dead)
    want, got = run(free_field, chain), run(free_field, chain, "cpp")
    depths = sorted(t["depth_m"] for t in want[-1]["tracks"])
    assert depths == pytest.approx([10.0, 60.0, 160.0], abs=3.0)
    assert sorted(t["depth_m"] for t in got[-1]["tracks"]) == pytest.approx(depths, abs=0.3)
    # A whole row gone (four boards): the two that are left still give the angle, as candidates.
    gone = run(free_field, dict(CHAIN, elevation={}, dead_channels=list(range(32, 64))))[-1]
    assert gone["rows"] == 2 and all(d["elevation_ambiguous"] for d in gone["detections"])
    with pytest.raises(Exception, match="dead_channels"):
        run(free_field, dict(CHAIN, elevation={}, dead_channels=[96]), "cpp")


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
    strongest_power = elevation.profile(rec, subs, frame, 0.0, ranges, sector_deg=(-30.0, 30.0, 0.25))[1].max()
    # Same-path arrivals (out and back by the same path) at their image ranges: direct, surface, bottom. Each is
    # looked for within 4 degrees of where geometry puts it (the rows' beam is 5.6 degrees wide now, and a
    # stronger neighbour's side lobe elsewhere in elevation is not the arrival).
    for name, image_z, tol_deg in (("direct", 50.0, 0.6), ("surface", -50.0, 1.5), ("bottom", 150.0, 1.5)):
        slant = float(np.hypot(1000.0, image_z - 30.0))
        want = np.degrees(np.arctan2(-(image_z - 30.0), 1000.0))
        el, power, quality = elevation.profile(rec, subs, frame, 0.0, ranges, sector_deg=(want - 4.0, want + 4.0, 0.1))
        near = np.abs(ranges - slant) < 0.2
        j = np.nonzero(near)[0][np.argmax(power[near])]
        assert power[j] > 0.1 * strongest_power, name
        # Range sidelobes of the neighbouring arrivals (-13 dB, unweighted pulse) pull each by up to a degree or so.
        assert np.degrees(el[j]) == pytest.approx(want, abs=tol_deg), name
        z = frame.origin[2] - ranges[j] * np.sin(el[j])
        folded, path = elevation.fold_depth(z, 100.0)
        assert path == name and folded == pytest.approx(50.0, abs=1000.0 * np.radians(tol_deg)), name
    # The chain takes the first arrival of the cluster, the direct one, and reports the target's depth.
    for engine in ("python", "cpp"):
        best = strongest(run(records, dict(CHAIN, elevation={"water_depth": 100.0}), engine)[-1])
        assert best["path"] == "direct" and best["depth_m"] == pytest.approx(50.0, abs=10.0), engine
        assert best["elevation_deg"] == pytest.approx(-1.15, abs=0.6), engine


# --- every stage, from its own files ----------------------------------------------------------------

STAGES = [   # scenario, deployment file, rows, check
    ("popoto_testauv_trial_1row", "popoto_testauv_1row", 1, None),
    ("popoto_static_trial_2row", "popoto_static_2row", 2, 60.0),
    ("popoto_static_trial_3row", "popoto_static_3row", 3, 18.0),
    ("popoto_elevation_rehearsal", "popoto_elevation_rehearsal", 3, 60.0),
]


@pytest.mark.parametrize("scenario, deploy, n_rows, depth", STAGES)
def test_every_stage_runs_from_its_files_through_both_engines(scenario, deploy, n_rows, depth):
    """Raw 102.4 kHz samples, every board on its own clock, the vehicle rolled and rolling: simulate, then
    the stage's own deployment file through the Python chain and the C++ chain."""
    sc = load_scenario_config(scenario)
    sc["pings"]["count"] = 2
    result = run_scenario(sc)
    target = np.asarray(sc["scene"][0]["positions"][0])
    chain = load_yaml(ROOT / "configs" / "deploy" / f"{deploy}.yaml")["chain"]
    chain["tracker"] = dict(chain["tracker"], confirm_hits=2)
    if chain.get("elevation"):
        chain["elevation"] = dict(chain["elevation"], gradient_per_s=None, water_depth=float(sc["environment"]["water_depth"]))   # the simulated water is isovelocity
    reports = {engine: run(result.records, dict(chain, backend="host"), engine) for engine in ("python", "cpp")}
    picked = {}
    for engine, reps in reports.items():
        last = reps[-1]
        assert last["sync"]["ok"] and last["sync"]["worst_residual_us"] < 1.0, engine
        own = np.asarray(last["ownship_xyz"])
        d = min((x for x in last["detections"] if x["snr_db"] is not None and x["snr_db"] > 25.0), key=lambda x: abs(x["range_m"] - np.linalg.norm(target - own)))
        picked[engine] = d
        assert d["range_m"] == pytest.approx(np.linalg.norm(target - own), abs=3.0), engine
        assert d["bearing_deg"] == pytest.approx(np.degrees(np.arctan2(target[0] - own[0], target[1] - own[1])), abs=1.0), engine
        if n_rows == 1:
            assert "rows" not in last and "elevation_deg" not in d, engine
        else:
            assert last["rows"] == n_rows and not d["elevation_ambiguous"] and d["path"] == "direct", engine
            # The roll at the echo's arrival (the record's attitude samples) is in the depth.
            assert d["depth_m"] == pytest.approx(depth, abs=max(6.0, 0.01 * d["range_m"])), engine
            # The sigma owns up to the attitude accuracy the file states (1 degree for a frame without a tilt sensor).
            assert d["elevation_sigma_deg"] < 0.5 + chain["elevation"]["roll_sigma_deg"], engine
    a, b = picked["python"], picked["cpp"]
    assert a["range_m"] == pytest.approx(b["range_m"], abs=0.1) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.02)
    if n_rows > 1:
        assert a["depth_m"] == pytest.approx(b["depth_m"], abs=0.3) and a["elevation_deg"] == pytest.approx(b["elevation_deg"], abs=0.02)


# --- passive ----------------------------------------------------------------------------------------

SOURCES = ([-400.0, 900.0, 20.0], [300.0, 800.0, 250.0])


def listening(seconds=2.0, name=FACE3, seed=2):
    """Two broadband sources about 1 km away, at 20 m and 250 m depth, heard from 100 m with the vehicle rolled 3 degrees
    (sim/passive.py: free field, no ping)."""
    sonar = sonar_rows(name, window=(0.0, 0.5 * C * seconds))
    pose = make_pose(position=(0.0, 0.0, 100.0), rpy=(np.radians(3.0), 0.0, 0.0))
    return passive_record(sonar, pose, [{"position": SOURCES[0], "rms": 1.0}, {"position": SOURCES[1], "rms": 0.7}],
                          noise_rms=1.0, rng=np.random.default_rng(seed))


def test_passive_elevation_of_sources_at_depth_numpy_and_cpp():
    rec = listening()
    assert rec.samples.array.shape[0] == 96
    rows = elevation.find_rows(rec)
    for source in SOURCES:
        _, cone, el_true, _ = truth(rec, rows, source)
        ref, q_ref = elevation.passive_elevation(rec, rows, cone, 4000.0, 6000.0, n_fft=256, hop=128, sector_deg=(-30.0, 30.0, 0.25))
        got, q = core.live_passive_elevation(rec, rows, cone, 4000.0, 6000.0, n_fft=256, hop=128, step_deg=0.25)
        assert got == pytest.approx(ref, abs=1e-5) and q == pytest.approx(q_ref, abs=1e-4)
        assert np.degrees(got - el_true) == pytest.approx(0.0, abs=0.2)
    frames, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, first_deg=-60.0, last_deg=60.0,
                                 max_sources=2, min_snr_db=3.0, elevation=True, elevation_step_deg=0.25)
    assert len(frames) >= 5
    for f in frames:
        found = sorted(zip((np.degrees(p[0]) for p in f["peaks"]), np.degrees(f["elevations"])))
        assert len(found) == 2
        # Level-referenced (the 3 degree roll taken out): 4.6 degrees up to the shallow source, 9.9 down to the deep one.
        for (bearing, level), source in zip(found, SOURCES):
            _, cone, _, level_true = truth(rec, rows, source)
            assert bearing == pytest.approx(np.degrees(cone), abs=0.6) and level == pytest.approx(np.degrees(level_true), abs=0.7)
    # Without the option the frames are as they were; a scenario without the key is an active one.
    plain, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, max_sources=2, method="conventional")
    assert "elevations" not in plain[0]
    # A source under a surface: its mirror image sings along, in antiphase.
    mirrored = passive_record(sonar_rows(FACE3, window=(0.0, 750.0)), make_pose(position=(0.0, 0.0, 100.0)),
                              [{"position": SOURCES[1], "rms": 1.0}], noise_rms=0.0, surface_z=0.0, rng=np.random.default_rng(1))
    alone = passive_record(sonar_rows(FACE3, window=(0.0, 750.0)), make_pose(position=(0.0, 0.0, 100.0)),
                           [{"position": SOURCES[1], "rms": 1.0}], noise_rms=0.0, rng=np.random.default_rng(1))
    assert not np.allclose(mirrored.samples.array, alone.samples.array)


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")
def test_program_reports_elevation_actively_and_passively(free_field, tmp_path):
    w = gs.RecordWriter(str(tmp_path / "active.h5"))
    for r in free_field:
        w.write(r)
    w.close()
    args = ["bearing_first_deg=-50", "bearing_last_deg=50", "cfar_pfa=1e-6", "cfar_guard_m=20", "cfar_train_m=100", "min_snr_db=15",
            "tracker_range_sigma=5", "tracker_confirm_hits=2", "multipath=0", "elevation=1", "elevation_roll_sigma_deg=0.2",
            "elevation_attitude_arrival=1", "elevation_gradient_sigma=0.05"]
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "active.h5"), *args], capture_output=True, text=True, check=True)
    last = json.loads(out.stdout.splitlines()[-1])
    assert last["rows"] == 3 and sorted(round(t["depth_m"] / 10.0) for t in last["tracks"]) == [1, 6, 16]
    assert all(t["depth_sigma_m"] > 10.0 for t in last["tracks"])                  # the gradient prior, at 1 km
    w = gs.RecordWriter(str(tmp_path / "passive.h5"))
    w.write(listening())
    w.close()
    out = subprocess.run([str(EXE), "--passive", str(tmp_path / "passive.h5"), "--band", "4000", "6000", "n_fft=256", "hop=128",
                          "n_average=12", "sbl_max_sources=2", "elevation=1"], capture_output=True, text=True, check=True)
    frames = [json.loads(line) for line in out.stdout.splitlines()]
    assert len(frames) >= 5 and all(len(f["elevations_deg"]) == len(f["bearings_deg"]) == 2 for f in frames)
    assert sorted(frames[0]["elevations_deg"]) == pytest.approx([-9.9, 4.6], abs=0.7)


def test_exported_command_carries_the_elevation_settings(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import sonar_live
    cfg = load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle_elevation.yaml")
    cmd = sonar_live.export_template(cfg, str(tmp_path / "t.h5"))
    for part in ("elevation=1", "elevation_first_deg=-30", "elevation_step_deg=0.25", "elevation_roll_sigma_deg=0.2", "elevation_attitude_arrival=1",
                 "elevation_use_gradient=1", "elevation_gradient=0.017", "elevation_gradient_sigma=0.05", "elevation_water_depth_sigma=5", "sync=1"):
        assert part in cmd, part
    assert gs.RecordReader(str(tmp_path / "t.h5")).read(0).samples.n_channels == 96
    for name in ("popoto_testauv_1row", "popoto_static_2row", "popoto_static_3row", "popoto_face_2row", "popoto_vehicle_elevation", "popoto_elevation_rehearsal"):
        chain = load_yaml(ROOT / "configs" / "deploy" / f"{name}.yaml")["chain"]
        opts = live.cpp_chain_options(chain, live.sync_reference(chain))
        core.LiveChainCpp(dict(opts, backend=core.Backend.Host))                   # every key is one the C++ chain takes
    two = sonar_live.export_template(load_yaml(ROOT / "configs" / "deploy" / "popoto_face_2row.yaml"), str(tmp_path / "u.h5"))
    assert "elevation_sector_from_depth=1" in two and "elevation_sector_bounces=1" in two
