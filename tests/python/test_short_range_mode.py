"""The short-range mode (opt-in; docs/design/short_range_mode.md): the long-range sonar's 2 s hyperbolic sweep (D-256), blind
inside c T / 2 = 1.5 km, alternating with the same sweep cut short, on a schedule stated in the sonar's `short_range` block.

Held here: the schedule; the short pings' sonar (the same sweep, level, band and array: only the length and the window
differ); the records of each kind (each carries its own replica and window: no record field is added); the long pings of a
run with the mode are the pings of a run without it at the same transmit times; the refusals; both live-chain engines on the
mixed stream (one tracker fed by both kinds, the detector planned for each ping's own pulse, the same report in both); the
blind range (without the mode the near target is not seen); the hand-over (a mover crossing 1.5 km is one track); and the
stacked rows' elevation on the short pings at about 1 km. Ours."""

import copy
import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.config import load_scenario_config
from gpusonar.live import LiveChain
from gpusonar.sim.scene import Scene
from gpusonar.sim.simulator import make_pose, run_scenario, simulate_ping
from gpusonar.sonar import SonarModel, ping_schedule, short_range_block, short_range_config, window_end_s

core = gs.core
SCENE = "popoto_short_range_mode"
SHIP, NEAR, CROSSING = (400.0, 12000.0), (-150.0, 800.0), (0.0, 1300.0)    # the scene's targets at t = 0 (x, y); the ship at -6 m/s in x
SR = {"waveform": {"duration": 0.25}, "window": {"range": [190.0, 3000.0]}, "every": 1, "interval_s": 4.3}


def scene_config(short_range=True, **pings):
    sc = load_scenario_config(SCENE, expand=False)
    if not short_range:
        sc["sonar_overrides"]["short_range"] = None
    sc["pings"].update(pings)
    return sc


@pytest.fixture(scope="module")
def mixed():
    return run_scenario(scene_config())


@pytest.fixture(scope="module")
def alone(mixed):
    """The baseline alone: the same scene without the block, its pings at the long pings' times of `mixed`."""
    sc = load_scenario_config(SCENE, expand=False)
    cycle = sc["pings"]["interval"] + sc["sonar_overrides"]["short_range"]["interval_s"]
    return run_scenario(scene_config(short_range=False, count=mixed.kinds.count("long"), interval=cycle))


def chain_run(records, engine, cfg=None):
    with LiveChain(dict(cfg or {}, engine=engine)) as chain:
        return [chain.process(r.copy()) for r in records]


@pytest.fixture(scope="module")
def reports(mixed):
    return {engine: chain_run(mixed.records, engine) for engine in ("python", "cpp")}


@pytest.fixture(scope="module")
def alone_reports(alone):
    return {engine: chain_run(alone.records, engine) for engine in ("python", "cpp")}


def near(xy, report_item, tol):
    return np.hypot(report_item["x"] - xy[0], report_item["y"] - xy[1]) < tol


def at_range(dets, r, tol):
    return [d for d in dets if abs(d["range_m"] - r) < tol]


# --- the schedule and the short pings' sonar -----------------------------------------------------------------------

def test_without_the_block_every_ping_is_long_and_evenly_spaced():
    assert ping_schedule(4, 19.0, None) == [("long", 0.0), ("long", 19.0), ("long", 38.0), ("long", 57.0)]
    assert short_range_block(SonarModel.load("popoto_long_range").config) is None       # the shipped sonar has no short pings


def test_the_schedule_alternates_and_the_short_ping_takes_its_own_interval():
    got = ping_schedule(5, 19.0, SR)
    assert [k for k, _ in got] == ["long", "short", "long", "short", "long"]
    assert [t for _, t in got] == pytest.approx([0.0, 19.0, 23.3, 42.3, 46.6])
    got = ping_schedule(6, 19.0, dict(SR, every=2))                                    # one short ping after every two long ones
    assert [k for k, _ in got] == ["long", "long", "short", "long", "long", "short"]
    assert [t for _, t in got] == pytest.approx([0.0, 19.0, 38.0, 42.3, 61.3, 80.3])


@pytest.mark.parametrize("block, words", [({"window": {"range": [190.0, 3000.0]}}, "interval_s must be given"),
                                          ({"interval_s": 4.0}, "window must be given"),
                                          (dict(SR, every=0), "every"),
                                          (dict(SR, interval_s=0.0), "positive"),
                                          (dict(SR, pulse=0.1), "unknown key")])
def test_the_block_is_checked_by_name(block, words):
    with pytest.raises(ValueError, match=words):
        short_range_block({"short_range": block})


def test_the_short_pings_sonar_is_the_same_sweep_cut_short():
    sc = load_scenario_config(SCENE, expand=False)
    long = SonarModel.load(sc["sonar"], sc["sonar_overrides"])
    short = SonarModel.from_config(short_range_config(long.config))
    lw, sw = long.waveform_config, short.waveform_config
    assert (sw["type"], sw["source"], sw["f_start"], sw["f_stop"]) == (lw["type"], lw["source"], lw["f_start"], lw["f_stop"]) == ("sampled", "hfm", 4000.0, 6000.0)
    assert (lw["duration"], sw["duration"]) == (2.0, 0.25)
    assert short.source_level_db_rms == long.source_level_db_rms == 197.0                # the same drive: the energy falls with the length
    assert long.source_energy_db - short.source_energy_db == pytest.approx(10.0 * np.log10(2.0 / 0.25), abs=0.05)
    assert np.array_equal(short.positions, long.positions) and "short_range" not in short.config
    c = 1500.0
    assert short.window(c)[0] == pytest.approx(2.0 * 190.0 / c) and long.window(c)[0] == pytest.approx(2.0 * 1500.0 / c)
    assert window_end_s(short, c) == pytest.approx(2.0 * 3000.0 / c + 0.25, abs=1e-3)


# --- the simulator ----------------------------------------------------------------------------------------------------

def test_the_records_carry_their_own_pulse_and_window(mixed):
    assert mixed.kinds == ["long", "short"] * 5
    assert np.diff(mixed.times) == pytest.approx([19.0, 4.3] * 4 + [19.0])
    c = mixed.records[0].environment.sound_speed
    for kind, r in zip(mixed.kinds, mixed.records):
        w, s = r.transmit.waveform, r.samples
        assert w.type == core.WaveformType.Arbitrary and (w.f_start, w.f_stop) == (4000.0, 6000.0)
        assert w.duration == (2.0 if kind == "long" else 0.25)
        assert s.start_time == pytest.approx(2.0 * (1500.0 if kind == "long" else 190.0) / c)
        assert len(w.replica) == (48001 if kind == "long" else 6001)                     # the replica as sent, at its own rate
    assert len({r.transmit.waveform.source_level_db for r in mixed.records}) == 1


def test_the_long_pings_are_the_baseline_pings_at_the_same_times(mixed, alone):
    """The short pings draw their noise from a generator of their own: with the mode on, every long ping is, to 4e-8 of
    its peak, the ping the baseline alone sends at that time. In this simulation the mode costs the long mode nothing on a
    ping (it does not model one ping's echoes arriving in the next one's window: docs/design/short_range_mode.md); the cost
    is the update rate."""
    longs = [r for k, r in zip(mixed.kinds, mixed.records) if k == "long"]
    assert len(longs) == len(alone.records) == 5
    for a, b in zip(longs, alone.records):
        assert a.transmit.time_ns == b.transmit.time_ns
        # Equal but for the last bits of the transmit time (a sum of the two intervals against a product): 4e-8 of the peak.
        assert np.max(np.abs(a.samples.array - b.samples.array)) <= 1e-6 * np.max(np.abs(b.samples.array))


def test_the_simulator_refuses_a_schedule_whose_windows_overlap():
    with pytest.raises(ValueError, match="long ping's window closes"):
        run_scenario(scene_config(count=2, interval=18.0))
    sc = scene_config(count=2)
    sc["sonar_overrides"]["short_range"]["interval_s"] = 4.0
    with pytest.raises(ValueError, match="short ping's window closes"):
        run_scenario(sc)
    sc = scene_config(count=2)
    sc["platform"]["raster"] = {"x": [0.0, 2.0, 1.0], "y": [0.0, 1.0, 1.0]}
    with pytest.raises(ValueError, match="straight run only"):
        run_scenario(sc)


# --- the chain on a mixed stream (both engines) ----------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_detector_is_planned_for_each_pings_own_pulse(engine):
    """Two pings of one window and two pulses: the detector's plan (its thresholds follow from the pulse's autocorrelation)
    after a long ping then a short one is the plan a chain makes for the short ping alone. Before, both engines kept the
    long ping's plan when the two maps had the same number of cells (thresholds 0.002 dB apart for these two pulses)."""
    over = {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [1500.0, 1800.0]}},
            "short_range": {"waveform": {"duration": 0.25}, "window": {"range": [1500.0, 1800.0]}, "interval_s": 3.0}}
    long = SonarModel.load("popoto_long_range", over)
    rng = np.random.default_rng(1)
    scene = Scene().add_points([[0.0, 1650.0, 30.0]], target_strength_db=10.0)

    def ping(model, i):
        return simulate_ping(model, make_pose(position=(0.0, 0.0, 30.0)), scene, environment={"sound_speed": 1500.0}, sequence=i,
                             time_ns=int(5e9 * i), noise_rms=1e-4, rng=rng)

    def plan_and_cells(records):
        with LiveChain({"engine": engine}) as chain:
            for r in records:
                chain.process(r.copy())
            return chain.detector, len(chain.last["ranges"])

    first = ping(long, 0)
    _, cells = plan_and_cells([first])
    short = None
    for r1 in np.arange(1797.0, 1803.0, 0.25):             # the short window whose map has the long one's number of cells
        over["short_range"]["window"]["range"][1] = float(r1)
        cand = ping(SonarModel.from_config(short_range_config(SonarModel.load("popoto_long_range", over).config)), 1)
        if plan_and_cells([cand])[1] == cells:
            short = cand
            break
    assert short is not None
    mixed_plan, _ = plan_and_cells([first, short])
    alone_plan, _ = plan_and_cells([short])
    long_plan, _ = plan_and_cells([first])
    assert long_plan != alone_plan                          # the two pulses have different plans
    assert mixed_plan == alone_plan


def test_both_engines_give_the_same_report_on_the_mixed_stream(reports):
    want, got = reports["python"], reports["cpp"]
    assert len(want) == len(got) == 10
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"])
        for a, b in zip(w["detections"], g["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
            assert a["snr_db"] == pytest.approx(b["snr_db"], abs=0.05)
        assert [(t["id"], t["kind"]) for t in w["tracks"]] == [(t["id"], t["kind"]) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "range_m"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_with_the_mode_the_near_and_far_targets_are_detected_and_tracked(mixed, reports, engine):
    reps = reports[engine]
    for kind, rep in zip(mixed.kinds, reps):
        dets = rep["detections"]
        if kind == "short":                                  # the near object on every short ping, never on a long one
            assert at_range(dets, 800.0, 80.0) and all(d["range_m"] < 3000.0 for d in dets)
        else:
            assert at_range(dets, 12000.0, 30.0) and all(d["range_m"] >= 1500.0 for d in dets)
    last = reps[-1]["tracks"]
    assert any(abs(t["range_m"] - 12000.0) < 30.0 for t in last)                            # the ship, 12 km
    assert any(near(NEAR, t, 40.0) for t in last)            # the near object (track positions are in the first ping's frame)


def test_without_the_mode_the_near_target_is_not_seen(alone, alone_reports):
    """The blind range is real: the baseline's window opens at c T / 2 = 1.5 km (it cannot listen while it sends its 2 s
    pulse), so the object at 800 m is in no record and no detection or track is near it, in either engine."""
    c = alone.records[0].environment.sound_speed
    assert all(r.samples.start_time == pytest.approx(2.0 * 1500.0 / c) for r in alone.records)
    for engine in ("python", "cpp"):
        reps = alone_reports[engine]
        assert all(d["range_m"] >= 1500.0 for rep in reps for d in rep["detections"])
        assert not any(abs(t["range_m"] - 800.0) < 300.0 for rep in reps for t in rep["tracks"])


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_mover_crossing_the_hand_over_range_is_one_track(mixed, reports, engine):
    """The mover opens from 1300 m at 4 m/s: the short pings see it from the first, the long ones once it is beyond 1.5 km.
    Its detections from both kinds of ping go to one track, which keeps its id from its confirmation to the last ping."""
    reps = reports[engine]
    kinds_seen, ids = set(), []
    for kind, t_, rep in zip(mixed.kinds, mixed.times, reps):
        truth = np.hypot(CROSSING[0] - 1.5 * t_, CROSSING[1] + 4.0 * t_)
        if at_range(rep["detections"], truth, 40.0):
            kinds_seen.add(kind)
        here = [t for t in rep["tracks"] if abs(t["range_m"] - truth) < 60.0]
        assert len(here) <= 1
        if here:
            ids.append(here[0]["id"])
    assert kinds_seen == {"long", "short"}
    assert len(ids) >= 5 and len(set(ids)) == 1
    assert np.hypot(CROSSING[0] - 1.5 * mixed.times[-1], CROSSING[1] + 4.0 * mixed.times[-1]) > 1700.0     # it did cross


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_long_targets_detections_are_the_baselines(mixed, alone_reports, reports, engine):
    """The cost to the 12 km ship on a ping: none in this simulation (its long pings are the baseline's, above, and the chain
    plans each ping for its own pulse): its detections are the baseline's (within 1e-4 dB and 1e-4 degrees: the records
    differ by 4e-8 of their peak, above)."""
    base = alone_reports[engine]
    longs = [rep for k, rep in zip(mixed.kinds, reports[engine]) if k == "long"]
    for a, b in zip(longs, base):
        da, db = at_range(a["detections"], 12000.0, 30.0), at_range(b["detections"], 12000.0, 30.0)
        assert len(da) == len(db) == 1
        assert da[0]["range_m"] == pytest.approx(db[0]["range_m"], abs=0.01)
        assert da[0]["bearing_deg"] == pytest.approx(db[0]["bearing_deg"], abs=1e-4) and da[0]["snr_db"] == pytest.approx(db[0]["snr_db"], abs=1e-4)


# --- the stacked rows on the short pings ------------------------------------------------------------------------------

def test_the_rows_elevation_on_the_short_pings_at_1km():
    """popoto_elevation_free's three targets near 1 km (inside the 2 s pulse's blind range) with the shipped long pulse and the
    short-range mode: the short pings give each its depth as popoto_elevation_free does with the former 1 s linear sweep (within
    2 m; tests/python/test_elevation.py), in both engines."""
    from test_elevation import CHAIN, QUIET
    res = run_scenario("popoto_elevation_short_range")
    assert res.kinds == ["long", "short"] * 3
    chain = dict(CHAIN, elevation=dict(QUIET, roll_sigma_deg=0.2))
    targets = {(-300.0, 900.0): 10.0, (100.0, 1000.0): 60.0, (500.0, 1100.0): 160.0}
    for engine in ("python", "cpp"):
        reps = chain_run(res.records, engine, chain)
        assert all(not rep["detections"] for k, rep in zip(res.kinds, reps) if k == "long")      # blind inside 1.5 km
        last = reps[-1]
        assert last["rows"] == 3 and len(last["tracks"]) == 3, engine
        for t in last["tracks"]:
            (x, y), depth = min(targets.items(), key=lambda kv: np.hypot(kv[0][0] - t["x"], kv[0][1] - t["y"]))
            assert t["depth_m"] == pytest.approx(depth, abs=2.0), engine
