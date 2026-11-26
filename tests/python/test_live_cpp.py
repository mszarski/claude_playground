"""The C++ live library against the Python stages it replaces, and the standalone program."""

import json
import time
import pathlib
import subprocess

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import classify, detect
from gpusonar.live import LiveChain
from gpusonar.sources import FlankBoardEmulator, PingAssembler, make_source
from test_live import CHAIN, INTERVAL, N, OVERRIDES, copies, pings  # noqa: F401  (pings is a fixture)

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "bin" / "sonar_live"
CPP_CHAIN = {"bearing_first_deg": -40.0, "bearing_last_deg": 40.0, "bearing_step_deg": 1.0, "range_cell_m": 1.0, "cfar_pfa": 1e-6,
             "cfar_guard_m": 15.0, "cfar_train_m": 50.0, "min_snr_db": 12.0, "tracker_range_sigma": 3.0,
             "tracker_bearing_sigma": np.radians(1.0), "tracker_confirm_hits": 3, "tracker_max_misses": 2,
             "classify_min_speed": 1.0, "classify_static_speed": 0.6, "classify_max_fit_rms": 10.0, "multipath": False}
EXE_OPTIONS = ["bearing_first_deg=-40", "bearing_last_deg=40", "bearing_step_deg=1", "range_cell_m=1", "cfar_pfa=1e-6",
               "cfar_guard_m=15", "cfar_train_m=50", "min_snr_db=12", "tracker_range_sigma=3", "tracker_bearing_sigma_deg=1",
               "tracker_confirm_hits=3", "tracker_max_misses=2", "classify_max_fit_rms=10", "multipath=0"]


def test_cluster_matches_python():
    rng = np.random.default_rng(3)
    power = rng.exponential(1.0, (40, 300)).astype(np.float32)
    power[10:13, 100:104] += 60.0
    power[30, 250] += 80.0
    ranges, bearings = np.linspace(100.0, 400.0, 300), np.radians(np.linspace(-40.0, 40.0, 40))
    det, ratio = core.cfar(power, pfa=1e-3, guard=4, train=20, return_ratio=True)
    want = detect.cluster(det, power, ranges, bearings, guard=4, train=20)
    got = core.live_cluster(det, power, ratio, list(ranges), list(bearings))
    assert len(got) == len(want) > 5
    for g, w in zip(got, want):
        assert g[0] == pytest.approx(w.range, abs=1e-4) and g[1] == pytest.approx(w.bearing, abs=1e-7)   # float32 sums
        assert g[3] == w.cells and g[2] == pytest.approx(w.snr_db, abs=0.05)


@pytest.mark.parametrize("timed", [False, True, "nan"])
@pytest.mark.parametrize("association", ["likelihood", "in_order"])
def test_tracker_and_classifier_match_python(association, timed):
    """The two trackers and classifiers on the same detections; `timed`: each detection with its own time, transmit +
    range / c (D-156), and the classes' positions at the ping's time; "nan": every detection's time and the classes'
    time NaN, which both engines take as not given (the ping's time; each track's last point)."""
    rng = np.random.default_rng(8)
    opts = dict(process_noise=0.3, range_sigma=1.5, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0,
                association=association)
    py_tracker, cpp_tracker = detect.Tracker(**opts), core.LiveTracker(**opts)
    # Three movers, a row of five sliding along a line, clutter that comes and goes.
    for i in range(9):
        t = 0.5 * i
        truth = [(200.0 + 4.0 * t, 30.0), (150.0, -40.0 + 3.0 * t), (260.0 - 2.0 * t, 80.0 - 2.0 * t)]
        truth += [(120.0 + 12.0 * k + 2.0 * t, -90.0) for k in range(5)]
        truth += [(300.0, -20.0), (180.0, 100.0)]
        dets = []
        for x, y in truth:
            x, y = x + 0.3 * rng.standard_normal(), y + 0.3 * rng.standard_normal()
            dets.append((float(np.hypot(x, y)), float(np.arctan2(y, x)), float(15.0 + 5.0 * rng.random()), 2))
        for _ in range(3):
            if rng.random() < 0.5:
                dets.append((float(rng.uniform(100, 350)), float(rng.uniform(-0.6, 0.6)), 13.0, 1))
        times = None if not timed else [float("nan")] * len(dets) if timed == "nan" else [t + d[0] / 1500.0 for d in dets]
        a = py_tracker.update(t, [detect.Detection(*d, time=None if times is None else times[j]) for j, d in enumerate(dets)])
        b = cpp_tracker.update(t, dets, times)
        assert [tr.id for tr in a] == [tr["id"] for tr in b]
        for tr, cr in zip(a, b):
            assert np.allclose(tr.state, cr["state"], atol=1e-8) and tr.hits == cr["hits"] and tr.misses == cr["misses"]
            assert np.allclose(np.array(tr.history), np.array(cr["history"]), atol=1e-8) and np.allclose(tr.snr, cr["snr"])
            # D-157: the measured position of each row, NaN on the same rows (the misses).
            assert np.allclose(np.array(tr.measured), np.array(cr["measured"]), atol=1e-8, equal_nan=True)
            assert tr.time == cr["time"] == tr.history[-1][0]
        if timed is True and a:                             # the detections' times reach the tracks
            assert any(tr.time > t for tr in a)
        if timed == "nan":
            assert all(tr.time == t for tr in a)
    kw = dict(min_speed=0.5, max_fit_rms=2.5, time=t if timed is True else float("nan") if timed == "nan" else None)
    want = classify.classify_tracks(a, linear={"length": 40.0, "width": 2.0}, **kw)
    got = cpp_tracker.classify(linear=True, linear_length=40.0, linear_width=2.0, **kw)
    assert [c.kind for c in want] == [c["kind"] for c in got]
    assert {"mover", "static", "line"} <= {c.kind for c in want}          # the scene exercises each rule
    for w, g in zip(want, got):
        assert np.allclose(w.position, g["position"], atol=1e-8) and np.allclose(w.velocity, g["velocity"], atol=1e-8)
        assert w.speed_sigma == pytest.approx(g["speed_sigma"], rel=1e-8) and w.seen == pytest.approx(g["seen"])
    mp = {"max_excess_range": 60.0, "bearing": 0.2, "velocity": 3.0, "origin": (0.0, 0.0)}
    want = classify.classify_tracks(a, multipath=mp, **kw)
    got = cpp_tracker.classify(multipath=True, max_excess_range=60.0, multipath_bearing=0.2, multipath_velocity=3.0, **kw)
    assert [(c.kind, c.parent) for c in want] == [(c["kind"], c["parent"]) for c in got]


def _close_to_golden(actual, desired):
    """Equal to the golden to 1e-12 relative: bit for bit where it was made, a few ulps on another CPU."""
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(desired, dtype=float), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("association", ["likelihood", "in_order"])
@pytest.mark.parametrize("stamp", ["none", "ping"])
def test_cpp_detections_without_their_own_time_track_exactly_as_before(association, stamp):
    """D-156: core.LiveTracker with detections that carry no time, or the ping's, gives what it gave before (bit for bit on the CPU that made the golden; to 1e-12 on
    others, whose vector instructions round in another order)
    (tests/python/data/tracker_before_d156.json, made with the C++ tracker at commit 1be89d4)."""
    from test_detect import SCENARIO_TRACKER, before_d156, tracker_scenario
    want = before_d156()[association]
    cpp = core.LiveTracker(**SCENARIO_TRACKER, association=association)
    for (t, dets), ping in zip(tracker_scenario(), want["pings"]):
        got = cpp.update(t, [d[:4] for d in dets], None if stamp == "none" else [t] * len(dets))
        assert [tr["id"] for tr in got] == [w["id"] for w in ping["cpp"]]
        for tr, w in zip(got, ping["cpp"]):
            _close_to_golden(tr["state"], w["state"])
            _close_to_golden(np.array(tr["history"][-len(w["history"]):]), np.array(w["history"]))   # full on the last ping
            assert (tr["hits"], tr["misses"]) == (w["hits"], w["misses"])
    # The golden's classes are the fit through the filtered history (before D-157): fit_measured=False. The default
    # (the measured positions) is held by the parity tests in this file.
    for at in (None, t):
        got = cpp.classify(min_speed=0.5, max_fit_rms=2.5, time=at, fit_measured=False)
        assert [(c["id"], c["kind"]) for c in got] == [(w["id"], w["kind"]) for w in want["cpp_classes"]]
        _close_to_golden([c["position"] for c in got], [w["position"] for w in want["cpp_classes"]])
        _close_to_golden([c["velocity"] for c in got], [w["velocity"] for w in want["cpp_classes"]])



@pytest.mark.parametrize("association", ["likelihood", "in_order"])
def test_cpp_default_classes_of_the_tracker_scenario_are_held(association):
    """D-157 review: core.LiveTracker's default classes (the line through the measured positions) on the tracker
    scenario, held at 1e-12 against tests/python/data/classes_d157.json, and the Python engine's within 1e-6."""
    from test_detect import SCENARIO_TRACKER, classes_d157, tracker_scenario
    want = classes_d157()[association]
    cpp = core.LiveTracker(**SCENARIO_TRACKER, association=association)
    for t, dets in tracker_scenario():
        cpp.update(t, [d[:4] for d in dets], None)
    for name, at in (("none", None), ("last_ping", t)):
        w = want[name]["cpp"]
        got = cpp.classify(min_speed=0.5, max_fit_rms=2.5, time=at)
        assert [(c["id"], c["kind"]) for c in got] == [(c["id"], c["kind"]) for c in w] == [(c["id"], c["kind"]) for c in want[name]["python"]]
        _close_to_golden([c["position"] for c in got], [c["position"] for c in w])
        _close_to_golden([c["velocity"] for c in got], [c["velocity"] for c in w])
        _close_to_golden([c["fit_rms"] for c in got], [c["fit_rms"] for c in w])
        np.testing.assert_allclose([c["position"] for c in w], [c["position"] for c in want[name]["python"]], atol=1e-6)

def test_both_trackers_place_a_long_range_mover_where_its_echoes_show_it():
    """D-156, both engines on the same detections of a ship at 12 km crossing at 6 m/s (test_detect.long_range_mover),
    each stamped transmit + R/c: the same tracks, and the position at the ping's time within 3 m of the ship there."""
    from test_detect import LONG_RANGE_TRACKER, long_range_mover
    pings = long_range_mover(drop=(4,))
    py, cpp = detect.Tracker(**LONG_RANGE_TRACKER), core.LiveTracker(**LONG_RANGE_TRACKER)
    for tp, d, _, _ in pings:
        dets = [] if d is None else [d]
        a = py.update(tp, dets)
        b = cpp.update(tp, [(q.range, q.bearing, q.snr_db, q.cells) for q in dets], [q.time for q in dets])
        assert [tr.id for tr in a] == [tr["id"] for tr in b]
        for tr, cr in zip(a, b):
            assert np.allclose(tr.state, cr["state"], atol=1e-6) and np.allclose(np.array(tr.history), np.array(cr["history"]), atol=1e-6)
            assert tr.time == cr["time"]
    tp, _, at_transmit, _ = pings[-1]
    (want,) = classify.classify_tracks(a, time=tp)
    (got,) = cpp.classify(time=tp)
    assert np.allclose(want.position, got["position"], atol=1e-6) and np.hypot(*(np.asarray(got["position"]) - at_transmit)) < 3.0


@pytest.mark.parametrize("association", ["likelihood", "in_order"])
def test_both_trackers_never_move_a_missed_track_back_in_time(association):
    """D-156 with a ping interval (5 s) shorter than R/c (8 s at 12 km): a track that took a detection at t_k + R/c and
    misses ping k+1 holds at a time later than t_{k+1}. Both engines keep it there (its state and time) instead of
    predicting it back to the ping: every history is in time order, and the engines agree."""
    from test_detect import LONG_RANGE_TRACKER, long_range_mover
    pings = long_range_mover(n=14, interval=5.0, drop=(4, 7, 8))
    py, cpp = detect.Tracker(**LONG_RANGE_TRACKER, association=association), core.LiveTracker(**LONG_RANGE_TRACKER, association=association)
    held = 0
    for tp, d, _, _ in pings:
        dets = [] if d is None else [d]
        before = {tr.id: tr.time for tr in py.tracks}
        a = py.update(tp, dets)
        b = cpp.update(tp, [(q.range, q.bearing, q.snr_db, q.cells) for q in dets], [q.time for q in dets])
        assert [tr.id for tr in a] == [cr["id"] for cr in b]
        for tr in py.tracks:
            times = [h[0] for h in tr.history]
            assert times == sorted(times)
        for tr, cr in zip(a, b):
            times = [h[0] for h in tr.history]
            assert times == sorted(times) and [h[0] for h in cr["history"]] == times
            assert np.allclose(tr.state, cr["state"], atol=1e-6) and tr.time == cr["time"]
            if d is None and before.get(tr.id, -np.inf) > tp:
                assert tr.time == before[tr.id] and tr.misses >= 1        # kept, not moved back to the ping
                held += 1
    assert held >= 2


@pytest.mark.parametrize("association", ["likelihood", "in_order"])
@pytest.mark.parametrize("case", ["earlier_arrival", "one_transmit_time"])
def test_both_trackers_take_an_earlier_detection_at_the_tracks_own_time(association, case):
    """sonar-uc5 (a), both engines: a detection timed before the time a track holds is taken there (held, dt = 0), so
    every history stays in time order; the engines agree in state, history and time (test_detect has the cases)."""
    from test_detect import EARLIER_TRACKER, SCENARIO_TRACKER, earlier_arrival_pings, same_transmit_time_pings
    pings, opts = (earlier_arrival_pings(), EARLIER_TRACKER) if case == "earlier_arrival" else (same_transmit_time_pings(), SCENARIO_TRACKER)
    py, cpp = detect.Tracker(**opts, association=association), core.LiveTracker(**opts, association=association)
    for tp, dets in pings:
        a = py.update(tp, dets)
        b = cpp.update(tp, [(d.range, d.bearing, d.snr_db, d.cells) for d in dets], [d.time for d in dets])
        assert [tr.id for tr in a] == [cr["id"] for cr in b]
        for tr, cr in zip(a, b):
            times = [h[0] for h in tr.history]
            assert times == sorted(times) and [h[0] for h in cr["history"]] == times
            assert np.allclose(tr.state, cr["state"], atol=1e-6) and tr.time == cr["time"]
    assert len(a) == (1 if case == "earlier_arrival" else 2)


def test_both_classifiers_fit_a_held_missed_row_once():
    """sonar-uc5 (b), both engines: the row a held missed track repeats (ping interval 5 s < R/c 8 s) records no
    measurement, and the C++ classifier gives what the Python one gives for the track (test_detect: equal to the track
    without that row)."""
    from test_detect import LONG_RANGE_TRACKER, long_range_mover
    pings = long_range_mover(n=14, interval=5.0, drop=(4,))
    py, cpp = detect.Tracker(**LONG_RANGE_TRACKER), core.LiveTracker(**LONG_RANGE_TRACKER)
    for tp, d, _, _ in pings:
        dets = [] if d is None else [d]
        a = py.update(tp, dets)
        b = cpp.update(tp, [(q.range, q.bearing, q.snr_db, q.cells) for q in dets], [q.time for q in dets])
    assert b[0]["history"][4] == b[0]["history"][3]
    tp = pings[-1][0]
    (want,) = classify.classify_tracks(a, time=tp, max_fit_rms=10.0)
    (got,) = cpp.classify(time=tp, max_fit_rms=10.0)
    assert np.allclose(want.position, got["position"], atol=1e-6) and np.allclose(want.velocity, got["velocity"], atol=1e-9)
    assert want.fit_rms == pytest.approx(got["fit_rms"], rel=1e-9) and want.kind == got["kind"]


@pytest.mark.parametrize("interval", [0.0, 2.0])
def test_both_classifiers_refuse_a_velocity_from_echo_delays_alone(interval):
    """sonar-054, both engines: pings at one transmit time (interval 0, a source without timestamps), detections at
    transmit + R/c. Each track records the ping time of each row (Track.pings, HistoryPoint::ping), and neither classifier
    fits a velocity (0, sigma infinite) where the transmit times span no more than the echo delays; with pings 2 s apart
    both fit, alike. (Before the fix both fitted about 1.5 km/s and more at interval 0: test_detect.)"""
    from test_detect import SCENARIO_TRACKER, same_transmit_time_pings
    py, cpp = detect.Tracker(**SCENARIO_TRACKER), core.LiveTracker(**SCENARIO_TRACKER)
    for k, (t, dets) in enumerate(same_transmit_time_pings()):
        tp = t + interval * k
        dets = [detect.Detection(d.range, d.bearing, d.snr_db, d.cells, time=tp + (d.time - t)) for d in dets]
        a = py.update(tp, dets)
        b = cpp.update(tp, [(d.range, d.bearing, d.snr_db, d.cells) for d in dets], [d.time for d in dets])
        assert [tr.id for tr in a] == [cr["id"] for cr in b]
        for tr, cr in zip(a, b):
            assert tr.pings == cr["pings"] == [interval * j for j in range(len(tr.history))]
    assert len(a) == 2
    want = classify.classify_tracks(a, min_speed=0.5, max_fit_rms=2.5, time=tp)
    got = cpp.classify(min_speed=0.5, max_fit_rms=2.5, time=tp)
    assert [(c.id, c.kind) for c in want] == [(c["id"], c["kind"]) for c in got]
    for w, g in zip(want, got):
        assert np.allclose(w.position, g["position"], atol=1e-8) and np.allclose(w.velocity, g["velocity"], atol=1e-8)
        assert w.speed_sigma == pytest.approx(g["speed_sigma"], rel=1e-8)
        if interval == 0.0:
            assert g["velocity"] == (0.0, 0.0) and g["speed_sigma"] == np.inf and g["kind"] != "mover"
        else:
            assert 0.0 < g["speed"] < 0.5


def test_both_classifiers_move_a_short_track_to_the_time_asked():
    """A confirmed track with fewer than three points has no fitted line; with `time` given both engines report its last
    point moved to that time by its Kalman state's velocity, not the last point itself (D-156: that is the ship
    v R / c = 6 m/s x 8 s = 48 m along its course from where it is at the ping). Without a time (None, NaN), the last
    point, as before."""
    from test_detect import LONG_RANGE_TRACKER, long_range_mover
    pings = long_range_mover(n=2)
    opts = dict(LONG_RANGE_TRACKER, confirm_hits=2)
    py, cpp = detect.Tracker(**opts), core.LiveTracker(**opts)
    for tp, d, _, _ in pings:
        a = py.update(tp, [d])
        cpp.update(tp, [(d.range, d.bearing, d.snr_db, d.cells)], [d.time])
    tp, _, at_transmit, _ = pings[-1]
    assert len(a) == 1 and len(a[0].history) == 2
    (want,) = classify.classify_tracks(a, time=tp)
    (got,) = cpp.classify(time=tp)
    assert np.allclose(want.position, got["position"], atol=1e-6)
    assert np.hypot(*(np.asarray(want.position) - at_transmit)) < 5.0
    for at in (None, float("nan")):
        (w,) = classify.classify_tracks(a, time=at)
        (g,) = cpp.classify(time=at)
        assert w.position == tuple(a[0].history[-1][1:]) and tuple(g["position"]) == w.position
        assert np.hypot(*(np.asarray(w.position) - at_transmit)) > 40.0


def test_detection_rules_match_python():
    """The opt-in rules from the detections themselves (classify.py `detections`; detections.hpp ClassifyOptions
    min_mover_cells, row_snr_db): the same tracks, cells and classes in both engines, with each rule off and on."""
    rng = np.random.default_rng(11)
    opts = dict(process_noise=0.3, range_sigma=1.5, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0)
    py_tracker, cpp_tracker = detect.Tracker(**opts), core.LiveTracker(**opts)
    for i in range(9):
        t = 0.5 * i
        # (x forward, y across, snr, cells): a row of weak small echoes sliding along a pipeline, a strong large boat
        # running along it, a small fast glint, a large boat clear of everything, an anchored boat.
        truth = [(100.0 + 12.0 * k + 2.0 * t, -30.0, 15.0, 2 + (i + k) % 2) for k in range(5)]
        truth += [(110.0 + 4.0 * t, -29.0, 26.0, 40 + i % 5), (60.0 + 1.0 * t, 80.0 - 6.0 * t, 14.0, 1 + i % 3),
                  (200.0 - 4.0 * t, 60.0, 22.0, 12 + i % 4), (220.0, -100.0, 24.0, 20)]
        dets = [(float(np.hypot(x, y)), float(np.arctan2(y, x)), float(s + rng.random()), int(c))
                for x, y, s, c in [(x + 0.2 * rng.standard_normal(), y + 0.2 * rng.standard_normal(), s, c) for x, y, s, c in truth]]
        a = py_tracker.update(t, [detect.Detection(*d) for d in dets])
        b = cpp_tracker.update(t, dets)
        assert [tr.id for tr in a] == [tr["id"] for tr in b]
        assert all(tr.cells == cr["cells"] for tr, cr in zip(a, b))
    kw = dict(min_speed=0.5, max_fit_rms=2.5, linear={"length": 40.0, "width": 3.0})
    kinds_by_rule = {}
    for rules in (None, {"min_mover_cells": 6.0}, {"row_snr_db": 6.0}, {"min_mover_cells": 6.0, "row_snr_db": 6.0}):
        want = classify.classify_tracks(a, detections=rules, **kw)
        r = rules or {}
        got = cpp_tracker.classify(min_speed=0.5, max_fit_rms=2.5, linear=True, linear_length=40.0, linear_width=3.0,
                                   min_mover_cells=r.get("min_mover_cells"), row_snr_db=r.get("row_snr_db"))
        assert [(c.id, c.kind) for c in want] == [(c["id"], c["kind"]) for c in got], rules
        assert [c.extent_cells for c in want] == pytest.approx([c["extent_cells"] for c in got])
        kinds_by_rule[str(rules)] = [c.kind for c in want]
    # Each rule changes something in this scene, so the comparison above exercises it.
    off = kinds_by_rule["None"]
    assert all(kinds_by_rule[k] != off for k in kinds_by_rule if k != "None")
    # Negative or NaN refused in both, as classify.py does (None is off).
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="zero or more"):
            cpp_tracker.classify(min_mover_cells=bad)
        with pytest.raises(ValueError, match="zero or more"):
            classify.classify_tracks(a, detections={"row_snr_db": bad})


def test_both_engines_leave_the_same_glinting_measurement_out_of_the_line():
    """sonar-ljh, both engines: the 12 km crossing (test_detect.long_range_mover, six pings) with its last detection moved
    50 m ahead along track in bearing (a glint), through the live chain's 1 deg tracker. Both classifiers leave that
    measurement out and agree to 1e-6 m; with outlier_alpha 0 both keep it, agree, and are pulled ahead."""
    from test_classify import LIVE_TRACKER_12KM
    from test_detect import long_range_mover
    pings = long_range_mover(n=6)
    py, cpp = detect.Tracker(**LIVE_TRACKER_12KM), core.LiveTracker(**LIVE_TRACKER_12KM)
    for k, (tp, d, _, _) in enumerate(pings):
        if k == len(pings) - 1:
            d = detect.Detection(d.range, d.bearing + 50.0 / d.range, d.snr_db, d.cells, time=d.time)
        a = py.update(tp, [d])
        cpp.update(tp, [(d.range, d.bearing, d.snr_db, d.cells)], [d.time])
    tp, _, at_transmit, _ = pings[-1]
    ahead = {}
    for alpha in (0.01, 0.0):
        (want,) = classify.classify_tracks(a, time=tp, outlier_alpha=alpha)
        (got,) = cpp.classify(time=tp, outlier_alpha=alpha)
        assert got["kind"] == want.kind == "mover"
        assert got["position"] == pytest.approx(want.position, abs=1e-6) and got["velocity"] == pytest.approx(want.velocity, abs=1e-6)
        assert got["fit_rms"] == pytest.approx(want.fit_rms, abs=1e-6) and got["speed_sigma"] == pytest.approx(want.speed_sigma, rel=1e-6)
        ahead[alpha] = want.position[0] - at_transmit[0]
    assert classify.fitted_points(a[0], outlier_alpha=0.01) == 5 and classify.fitted_points(a[0]) == 6
    # Off by default in both engines (owner, D-159).
    (dflt,) = cpp.classify(time=tp)
    assert dflt["position"] == pytest.approx(classify.classify_tracks(a, time=tp, outlier_alpha=0.0)[0].position, abs=1e-6)
    assert dflt["position"] == pytest.approx(classify.classify_tracks(a, time=tp)[0].position, abs=1e-6)
    assert abs(ahead[0.01]) < 3.0 and ahead[0.0] > 15.0
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError, match="outlier_alpha"):
            cpp.classify(outlier_alpha=bad)
        with pytest.raises(ValueError, match="outlier_alpha"):
            core.LiveChainCpp(dict(CPP_CHAIN, classify_outlier_alpha=bad))


def test_the_chain_says_its_outlier_alpha_to_the_cpp_engine():
    """sonar-ljh: the configuration's classify.outlier_alpha (default 0: off, D-159) always goes to the C++ chain's options,
    whose own default is 0 too, and a value out of [0, 1) is refused by the Python chain."""
    from gpusonar.live import cpp_chain_options
    assert cpp_chain_options(dict(CHAIN))["classify_outlier_alpha"] == 0.0
    assert json.loads(core.LiveChainCpp(dict(CPP_CHAIN)).options())["classify_outlier_alpha"] == 0.0
    on = dict(CHAIN, classify=dict(CHAIN["classify"], outlier_alpha=0.01))
    assert cpp_chain_options(on)["classify_outlier_alpha"] == 0.01
    with pytest.raises(ValueError, match="outlier_alpha"):
        LiveChain(dict(CHAIN, classify=dict(CHAIN["classify"], outlier_alpha=1.5)))


def test_both_engines_grow_the_fit_limit_with_range_alike():
    """sonar-4jm, both engines: the 12 km crossing (ten pings) with its bearings alternately 0.45 deg either side (94 m
    across, fit RMS about 74 m). Under a fixed 60 m limit both classify it uncertain; with the limit grown by the
    Popoto array's derived 0.63 deg x range (132 m at 12 km) both a mover; positions and fit agree to 1e-6."""
    from test_classify import LIVE_TRACKER_12KM, POPOTO_FIT_RMS_BEARING
    from test_detect import long_range_mover
    py, cpp = detect.Tracker(**LIVE_TRACKER_12KM), core.LiveTracker(**LIVE_TRACKER_12KM)
    for k, (tp, d, _, _) in enumerate(long_range_mover(n=10)):
        d = detect.Detection(d.range, d.bearing + np.radians(0.45) * (-1) ** k, d.snr_db, d.cells, time=d.time)
        a = py.update(tp, [d])
        cpp.update(tp, [(d.range, d.bearing, d.snr_db, d.cells)], [d.time])
    kinds = []
    for grow in (0.0, POPOTO_FIT_RMS_BEARING):
        (want,) = classify.classify_tracks(a, time=tp, max_fit_rms=60.0, max_fit_rms_bearing=grow, origin=(0.0, 0.0))
        (got,) = cpp.classify(time=tp, max_fit_rms=60.0, max_fit_rms_bearing=grow)
        assert got["kind"] == want.kind and got["fit_rms"] == pytest.approx(want.fit_rms, abs=1e-6)
        assert got["position"] == pytest.approx(want.position, abs=1e-6)
        kinds.append(want.kind)
        assert 60.0 < want.fit_rms < 132.0
    assert kinds == ["uncertain", "mover"]


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_chain_derives_the_fit_limits_growth_from_the_beam(pings, engine):
    """sonar-4jm: each engine resolves classify.max_fit_rms_bearing_deg from the record (four times a sixteenth of the
    beamwidth over sqrt 2), and a value in the configuration wins and is listed as given."""
    with LiveChain(dict(CHAIN, engine=engine)) as chain:
        chain.run(copies(pings[:1]))
        r = chain.resolved
    beam = r["sonar"]["beamwidth_deg"]
    assert r["classify_max_fit_rms_bearing_deg"] == pytest.approx(4.0 * beam / 16.0 / np.sqrt(2.0), rel=1e-9)
    given = dict(CHAIN, engine=engine, classify=dict(CHAIN["classify"], max_fit_rms_bearing_deg=0.0))
    with LiveChain(given) as chain:
        chain.run(copies(pings[:1]))
        assert chain.resolved["classify_max_fit_rms_bearing_deg"] == 0.0
        assert "classify.max_fit_rms_bearing_deg" in chain.resolved["given"]


def _kinds(report):
    return sorted(t["kind"] for t in report["tracks"])


def test_the_fit_settings_given_to_the_cpp_chain_reach_its_classifier(pings):
    """Review of sonar-4jm and sonar-ljh: a classify_max_fit_rms_bearing given to LiveChainCpp decides its classes. Under
    a fit limit of 1 mm every track is uncertain; grown by 5 deg x range (about 22 m at these 250 m ranges) the two targets
    are a mover and a static object again, as without the tiny limit. outlier_alpha given reaches the chain's options."""
    def run(**kw):
        cpp = core.LiveChainCpp(dict(CPP_CHAIN, classify_max_fit_rms=1e-3, **kw))
        return [json.loads(cpp.process(r)) for r in copies(pings)], cpp
    tiny, _ = run(classify_max_fit_rms_bearing=0.0)
    grown, cpp = run(classify_max_fit_rms_bearing=float(np.radians(5.0)), classify_outlier_alpha=0.2)
    assert set(_kinds(tiny[-1])) == {"uncertain"}
    assert _kinds(grown[-1]) == ["mover", "static"]
    opts = json.loads(cpp.options())
    assert opts["classify_outlier_alpha"] == 0.2 and opts["classify_max_fit_rms_bearing_deg"] == pytest.approx(5.0)


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")
def test_the_fit_settings_given_to_sonar_live_reach_its_classifier(pings, tmp_path):
    """The same through build/bin/sonar_live's keys (degrees), and its refusal of an angle that is not finite and zero or
    more, as LiveChainCpp's, LiveChain's, cpp_chain_options's and classify_tracks's."""
    with LiveChain(CHAIN, recording=str(tmp_path / "rec.h5")) as chain:
        chain.run(copies(pings))
    def run(*extra):
        rep = tmp_path / f"rep{len(list(tmp_path.glob('rep*')))}.jsonl"
        out = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), "--report", str(rep), *EXE_OPTIONS,
                              "classify_min_speed=1", "classify_static_speed=0.6", "classify_max_fit_rms=0.001", *extra],
                             capture_output=True, text=True, timeout=300)
        return out, rep
    out, rep = run("classify_max_fit_rms_bearing_deg=0")
    assert out.returncode == 0, out.stderr
    assert set(_kinds(json.loads(rep.read_text().splitlines()[-1]))) == {"uncertain"}
    out, rep = run("classify_max_fit_rms_bearing_deg=5", "classify_outlier_alpha=0.2")
    assert out.returncode == 0, out.stderr
    assert _kinds(json.loads(rep.read_text().splitlines()[-1])) == ["mover", "static"]
    for bad in ("-1", "inf", "nan"):
        out, _ = run(f"classify_max_fit_rms_bearing_deg={bad}")
        assert out.returncode != 0 and "classify_max_fit_rms_bearing_deg" in out.stderr


def test_the_fit_limits_growth_is_refused_alike_out_of_range():
    """Review of sonar-4jm: -1 (once silently off), infinity (once switched the steadiness test off) and NaN (once
    silently 'derive') are refused by every entry: LiveChainCpp, tracker.classify, LiveChain and cpp_chain_options."""
    from gpusonar.live import cpp_chain_options
    from test_classify import LIVE_TRACKER_12KM
    tr = core.LiveTracker(**LIVE_TRACKER_12KM)
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="max_fit_rms_bearing"):
            core.LiveChainCpp(dict(CPP_CHAIN, classify_max_fit_rms_bearing=bad))
        with pytest.raises(ValueError, match="max_fit_rms_bearing"):
            tr.classify(max_fit_rms_bearing=bad)
        cfg = dict(CHAIN, classify=dict(CHAIN["classify"], max_fit_rms_bearing_deg=bad))
        with pytest.raises(ValueError, match="max_fit_rms_bearing_deg"):
            LiveChain(cfg)
        with pytest.raises(ValueError, match="max_fit_rms_bearing_deg"):
            cpp_chain_options(cfg)


def test_cpp_chain_detection_rules_off_by_default_and_equal_when_on(pings):
    """The live chain in both engines: without `classify.detections` the C++ chain's options carry no rule; with the
    extent rule on, the two engines' reports give the same classes. Each target here is 15 cells on every ping: at 16
    the crossing target, a mover without the rule, is uncertain in both. The row rule is refused by both: the chain
    never runs the row test it acts in."""
    from gpusonar.live import cpp_chain_options
    assert not {"classify_min_mover_cells", "classify_row_snr_db"} & set(cpp_chain_options(dict(CHAIN)))
    on = {"min_mover_cells": 16.0}
    cfg = dict(CHAIN, classify=dict(CHAIN["classify"], detections=on))
    with LiveChain(cfg) as chain:
        want = chain.run(copies(pings))
    cpp = core.LiveChainCpp(dict(CPP_CHAIN, classify_min_mover_cells=16.0))
    got = [json.loads(cpp.process(r)) for r in copies(pings)]
    for w, g in zip(want, got):
        assert [(t["id"], t["kind"], t["parent"]) for t in w["tracks"]] == [(t["id"], t["kind"], t["parent"]) for t in g["tracks"]]
    assert sorted(t["kind"] for t in got[-1]["tracks"]) == ["static", "uncertain"]      # off: a mover and a static object
    assert cpp_chain_options(cfg)["classify_min_mover_cells"] == 16.0
    row = dict(CHAIN, classify=dict(CHAIN["classify"], detections={"row_snr_db": 6.0}))
    for make in (lambda: LiveChain(row), lambda: cpp_chain_options(row),
                 lambda: core.LiveChainCpp(dict(CPP_CHAIN, classify_row_snr_db=6.0))):
        with pytest.raises(ValueError, match="row_snr_db.*never runs the row"):
            make()
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="zero or more"):
            core.LiveChainCpp(dict(CPP_CHAIN, classify_min_mover_cells=bad))


@pytest.mark.parametrize("key, value, where, nearest", [
    ("classify_min_mover_cell", 16.0, "", "classify_min_mover_cells"),       # the misspelling sonar-6hf names
    ("tracker_confirm_hit", 4, "", "tracker_confirm_hits"),
    ("min_snr", 12.0, "", "min_snr_db"),
    ("no_such_setting_at_all", 1.0, "", None),
    ("elevation", {"first_deg": -20.0, "max_detection": 5}, "elevation.", "max_detections"),
    ("blanking", {"recovery_s": 0.002, "taper": 0.001}, "blanking.", "taper_s"),
    ("decimate", {"output_rate": 2.0e4, "tap": 101}, "decimate.", "taps"),
])
def test_cpp_chain_refuses_an_unknown_key_by_name(key, value, where, nearest):
    """sonar-6hf: LiveChainCpp read its dictionary key by key and passed over any other, so a misspelt key left its
    setting at the default without a word (classify_min_mover_cell: the extent rule off). It now refuses a key it
    does not read, at the top and in each section it reads, by name and with the nearest key it does read."""
    bad = list(value)[-1] if where else key                  # in a section: its last key is the misspelt one
    match = f"LiveChainCpp: {where}{bad}: unknown key" + (f"; did you mean '{nearest}'" if nearest else "")
    with pytest.raises(ValueError, match=match.replace(".", r"\.")):
        core.LiveChainCpp(dict(CPP_CHAIN, **{key: value}))


def test_cpp_chain_refuses_an_unknown_key_in_sync_and_profile():
    """The sync section and elevation.profile: the same rule (sonar-6hf)."""
    sync = {"f_start": 38e3, "f_stop": 48e3, "duration": 4e-3, "lead": 5e-3, "tail": 10e-3}
    core.LiveChainCpp(dict(CPP_CHAIN, sync=sync))
    with pytest.raises(ValueError, match=r"LiveChainCpp: sync\.tolerance: unknown key; did you mean 'tolerance_us'"):
        core.LiveChainCpp(dict(CPP_CHAIN, sync=dict(sync, tolerance=2.0)))
    with pytest.raises(ValueError, match=r"LiveChainCpp: elevation\.profile\.bin: unknown key; did you mean 'bin_m'"):
        core.LiveChainCpp(dict(CPP_CHAIN, elevation={"profile": {"ladder": ["prior"], "bin": 2.0}}))


@pytest.mark.parametrize("section, nearest", [
    ({"min_snr": 12.0}, "min_snr_db"),
    ({"classify": {"min_sped": 2.0}}, "min_speed"),
    ({"tracker": {"confirm_hit": 4}}, "confirm_hits"),
    ({"cfar": {"false_alarm_per_ping": 1e-2}}, "false_alarms_per_ping"),
    ({"sbl": {"max_source": 3}}, "max_sources"),
    ({"classify": {"multipath": {"velocit": 2.0}}}, "velocity"),
    ({"sync": {"reference": {}, "tolerance": 2.0}}, "tolerance_us"),
    ({"elevation": {"max_detection": 5}}, "max_detections"),
    ({"elevation": {"sector_from_rays": {"max_depth": 300.0}}}, "max_depth_m"),
])
def test_python_chain_refuses_an_unknown_key_by_name(section, nearest):
    """sonar-6hf, the Python side: LiveChain (either engine) and cpp_chain_options merged the configuration over the
    defaults and passed over a key the defaults do not have, so `classify: {min_sped: 2}` ran with min_speed 1. Both
    refuse it now, by name and with the nearest key, as LiveChainCpp does."""
    from gpusonar.live import cpp_chain_options
    cfg = dict(CHAIN, **{k: (dict(CHAIN.get(k) or {}, **v) if isinstance(v, dict) and k != "sync" else v) for k, v in section.items()})
    for make in (lambda: LiveChain(cfg), lambda: LiveChain(dict(cfg, engine="cpp", backend="host")), lambda: cpp_chain_options(cfg)):
        with pytest.raises(ValueError, match=f"unknown key; did you mean '{nearest}'"):
            make()


def test_echo_grouping_matches_python_and_takes_the_earliest_established_arrival():
    """Echo grouping in both engines on the same tracks (D-155): a ship whose strongest arrival is the later one (1040 m)
    and a weaker one beyond it (1080 m); a short track before it (980 m, three pings: uncertain); a second ship at another
    bearing with its own echo. Each ship's object is its earliest established arrival and every other member is its
    multipath. Before D-155 the 1000 m arrival stayed a second mover beside the strong 1040 m one, and the short track
    stayed uncertain."""
    opts = dict(process_noise=0.3, range_sigma=6.0, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0)
    py_tracker, cpp_tracker = detect.Tracker(**opts), core.LiveTracker(**opts)
    for i in range(8):
        t = 2.0 * i
        dets = [(1040.0 + 3.0 * t, 0.2, 30.0, 2), (1000.0 + 3.0 * t, 0.2, 18.0, 2), (1080.0 + 3.0 * t, 0.2, 14.0, 2),
                (1500.0 - 2.0 * t, -0.3, 25.0, 2), (1530.0 - 2.0 * t, -0.3, 15.0, 2)]
        if i >= 5:
            dets.append((980.0 + 3.0 * t, 0.2, 13.0, 2))
        a = py_tracker.update(t, [detect.Detection(*d) for d in dets])
        cpp_tracker.update(t, dets)
    kw = dict(min_speed=1.0, static_speed=1.0, max_fit_rms=3.0)
    mp = {"max_excess_range": 800.0, "bearing": np.radians(3.0), "velocity": 2.5, "origin": (0.0, 0.0)}
    want = classify.classify_tracks(a, multipath=mp, **kw)
    got = cpp_tracker.classify(multipath=True, max_excess_range=800.0, multipath_bearing=np.radians(3.0), multipath_velocity=2.5, **kw)
    assert [(c.id, c.kind, c.parent) for c in want] == [(c["id"], c["kind"], c["parent"]) for c in got]
    by_range = {round(float(np.hypot(*c.position)) / 10.0) * 10: c for c in want}          # start range + 3 m/s * 14 s
    ship, strong, far, short = by_range[1040], by_range[1080], by_range[1120], by_range[1020]
    other, other_echo = by_range[1470], by_range[1500]
    assert ship.kind == "mover" and ship.parent is None and other.kind == "mover" and other.parent is None
    assert all((c.kind, c.parent) == ("multipath", ship.id) for c in (strong, far, short))
    assert (other_echo.kind, other_echo.parent) == ("multipath", other.id)
    assert strong.mean_snr_db > ship.mean_snr_db                    # the object is not the strongest arrival


def test_transitive_echo_grouping_matches_python():
    """A chain of arrivals 600 m apart in range with the gate at 800 m (the first and the third 1200 m apart), and a
    chain in velocity, on one tracker in both engines. The chain in velocity is one group, its nearest established
    arrival the object; the chain in range reaches no farther than one gate from its object (sonar-a5x), so the third
    arrival is an object of its own (before: the first's multipath). With a NaN origin nothing is grouped in either
    engine and neither raises."""
    opts = dict(process_noise=0.3, range_sigma=6.0, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0)
    py_tracker, cpp_tracker = detect.Tracker(**opts), core.LiveTracker(**opts)
    for i in range(8):
        t = 2.0 * i
        dets = [(3000.0 + 3.0 * t, 0.2, 30.0, 2), (3600.0 + 3.0 * t, 0.2, 18.0, 2), (4200.0 + 3.0 * t, 0.2, 15.0, 2),
                (5000.0 - 1.0 * t, -0.3, 25.0, 2), (5040.0 - 3.0 * t, -0.3, 16.0, 2), (5080.0 - 5.0 * t, -0.3, 14.0, 2)]
        a = py_tracker.update(t, [detect.Detection(*d) for d in dets])
        cpp_tracker.update(t, dets)
    kw = dict(min_speed=0.5, static_speed=0.5, max_fit_rms=3.0)
    for origin in ((0.0, 0.0), (float("nan"), float("nan"))):
        mp = {"max_excess_range": 800.0, "bearing": np.radians(3.0), "velocity": 2.5, "origin": origin}
        want = classify.classify_tracks(a, multipath=mp, **kw)
        got = cpp_tracker.classify(multipath=True, max_excess_range=800.0, multipath_bearing=np.radians(3.0), multipath_velocity=2.5,
                                   origin_x=origin[0], origin_y=origin[1], **kw)
        assert [(c.id, c.kind, c.parent) for c in want] == [(c["id"], c["kind"], c["parent"]) for c in got]
        if np.isnan(origin[0]):
            assert all(c.parent is None for c in want) and [c.kind for c in want].count("mover") == 6
        else:
            assert sorted(c.kind for c in want) == ["mover", "mover", "mover", "multipath", "multipath", "multipath"]
            third = min(want, key=lambda c: abs(np.hypot(*c.position) - 4242.0))
            assert third.kind == "mover" and third.parent is None
            nearest = {k: min((c for c in want if abs(c.position[0] / np.hypot(*c.position) - np.sin(b)) < 0.01),
                              key=lambda c: np.hypot(*c.position)) for k, b in (("first", 0.2), ("second", -0.3))}
            assert all(c.kind == "mover" and c.parent is None for c in nearest.values())


@pytest.mark.parametrize("strong_in_b", [False, True])
def test_two_ships_at_one_bearing_beyond_one_gate_are_two_objects_in_both_engines(strong_in_b):
    """sonar-a5x on one tracker in both engines: two ships at one bearing and velocity, B 850 m beyond A, each seen as
    three arrivals over 120 m; A's last arrival is within the gate (800 m) of B's first. Each ship is its own object
    with its own two echoes, also when B holds the strongest arrival (the anchor then lies beyond A's object and groups
    again from what is left). Before, the transitive growth made both one object."""
    opts = dict(process_noise=0.3, range_sigma=6.0, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0)
    py_tracker, cpp_tracker = detect.Tracker(**opts), core.LiveTracker(**opts)
    snr_a, snr_b = ((30.0, 18.0, 14.0), (20.0, 16.0, 13.0)) if not strong_in_b else ((20.0, 16.0, 13.0), (17.0, 35.0, 14.0))
    for i in range(8):
        t = 2.0 * i
        dets = [(3000.0 + 60.0 * k + 3.0 * t, 0.2, s, 2) for k, s in enumerate(snr_a)]
        dets += [(3850.0 + 60.0 * k + 3.0 * t, 0.2, s, 2) for k, s in enumerate(snr_b)]
        a = py_tracker.update(t, [detect.Detection(*d) for d in dets])
        cpp_tracker.update(t, dets)
    kw = dict(min_speed=0.5, static_speed=0.5, max_fit_rms=3.0)
    mp = {"max_excess_range": 800.0, "bearing": np.radians(3.0), "velocity": 2.5, "origin": (0.0, 0.0)}
    want = classify.classify_tracks(a, multipath=mp, **kw)
    got = cpp_tracker.classify(multipath=True, max_excess_range=800.0, multipath_bearing=np.radians(3.0), multipath_velocity=2.5, **kw)
    assert [(c.id, c.kind, c.parent) for c in want] == [(c["id"], c["kind"], c["parent"]) for c in got]
    by_range = sorted(want, key=lambda c: np.hypot(*c.position))
    assert [c.kind for c in by_range] == ["mover", "multipath", "multipath"] * 2
    assert [c.parent for c in by_range] == [None, by_range[0].id, by_range[0].id, None, by_range[3].id, by_range[3].id]


def test_both_chains_group_echoes_alike(pings):
    """The live chain in both engines with echo grouping on, over pings with a target crossing at 3 m/s, its strongest
    arrival 25 m beyond it and a weaker one 50 m beyond (point echoes on its bearing, moving with it). Both engines give
    one mover, the target's own (earliest) arrival, and the other two as its multipath; before D-155 the strong later
    arrival was a second mover."""
    from gpusonar.sim import Scene, make_pose, simulate_ping
    from gpusonar.sonar import SonarModel
    from test_live import OVERRIDES
    sonar = SonarModel.load("flank_32", OVERRIDES)
    rng = np.random.default_rng(5)
    echoes = []
    for i in range(8):
        t = i * INTERVAL
        p = np.array([40.0 - 3.0 * t, 220.0, 0.0])
        r = float(np.hypot(p[0], p[1]))
        scene = (Scene().add_points([p], target_strength_db=6.0).add_points([p * (r + 25.0) / r], target_strength_db=14.0)
                 .add_points([p * (r + 50.0) / r], target_strength_db=4.0).add_points([[-60.0, 320.0, 0.0]], target_strength_db=10.0))
        echoes.append(simulate_ping(sonar, make_pose(position=(1.5 * t, 0.0, 0.0), velocity=(1.5, 0.0, 0.0)), scene,
                                    environment={"sound_speed": 1500.0}, sequence=i, time_ns=int(t * 1e9), noise_rms=1e-5, rng=rng))
    cfg = dict(CHAIN, classify=dict(CHAIN["classify"], multipath={"max_excess_range": 80.0, "bearing_deg": 3.0, "velocity": 2.5}))
    reports = {}
    for engine in ("python", "cpp"):
        with LiveChain(dict(cfg, engine=engine)) as chain:
            reports[engine] = chain.run(copies(echoes))
    for w, g in zip(reports["python"], reports["cpp"]):
        assert [(t["id"], t["kind"], t["parent"]) for t in w["tracks"]] == [(t["id"], t["kind"], t["parent"]) for t in g["tracks"]]
    last = sorted(reports["python"][-1]["tracks"], key=lambda t: t["range_m"])
    moving = [t for t in last if t["kind"] in ("mover", "multipath")]
    assert [t["kind"] for t in moving] == ["mover", "multipath", "multipath"]
    assert all(t["parent"] == moving[0]["id"] for t in moving[1:])
    assert abs(moving[0]["range_m"] - 220.0) < 6.0 and moving[1]["mean_snr_db"] > moving[0]["mean_snr_db"]


def test_cpp_tracker_refuses_an_unknown_association():
    with pytest.raises(ValueError, match="association"):
        core.LiveTracker(association="nearest")


def test_cpp_chain_gives_the_python_chains_report(pings):
    with LiveChain(CHAIN) as chain:
        want = chain.run(copies(pings))
    cpp = core.LiveChainCpp(CPP_CHAIN)
    got = [json.loads(cpp.process(r)) for r in copies(pings)]
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"])
        for a, b in zip(w["detections"], g["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
        assert [(t["id"], t["kind"], t["parent"]) for t in w["tracks"]] == [(t["id"], t["kind"], t["parent"]) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "range_m"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key
    assert any(t["kind"] == "mover" for t in got[-1]["tracks"]) and any(t["kind"] == "static" for t in got[-1]["tracks"])
    power, ranges, bearings = cpp.last_map()
    assert power.shape == (len(bearings), len(ranges)) and power.max() > 0


def test_both_chains_report_track_positions_at_the_ping_time(pings):
    """D-156, the live chain in both engines: each detection is given to the tracker at transmit + range / c (c the
    record's sound speed, here 1500 m/s), so a track's last point is its detection's time, and the report gives its
    position on the fitted line at the report's time_ns (the transmit time), not at that last point. The two engines'
    reports agree to 0.05 m, less than the 0.5 m the crossing target (3 m/s at 250 m) moves in R/c."""
    with LiveChain(dict(CHAIN, engine="python")) as chain:
        want = chain.run(copies(pings))
        tracker, origin, t0 = chain.tracker, chain._origin, chain._t0
    t = (want[-1]["time_ns"] - t0) * 1e-9
    by_id = {tr.id: tr for tr in tracker.tracks}
    moved = 0
    for rt in want[-1]["tracks"]:
        tr = by_id[rt["id"]]
        r = float(np.hypot(*tr.history[-1][1:]))
        if tr.misses == 0:                                # took a detection on this ping: at its own time
            assert tr.history[-1][0] == pytest.approx(t + r / 1500.0, abs=0.01)
        # The fitted line is through the track's measured positions (D-157), as classify_tracks fits it: the outlier test
        # about the sonar's place at the ping, in the tracker's (across, forward) frame, as the chain gives it.
        own = np.asarray(want[-1]["ownship_xyz"][:2]) - origin[:2]
        pos = classify.fit_velocity(tr.history, t, tr.state[2:4], tr.measured, origin=(own[1], own[0]))[0]
        assert (rt["x"], rt["y"]) == pytest.approx((pos[1] + origin[0], pos[0] + origin[1]), abs=1e-6)
        last = classify.fit_velocity(tr.history, None, tr.state[2:4], tr.measured, origin=(own[1], own[0]))[0]
        if rt["kind"] == "mover":
            moved += 1
            assert np.hypot(*(pos - last)) == pytest.approx(rt["speed_m_s"] * (tr.history[-1][0] - t), rel=1e-6)
            assert np.hypot(*(pos - last)) > 0.3
    assert moved >= 1
    with LiveChain(dict(CHAIN, engine="cpp")) as chain:
        got = chain.run(copies(pings))
    for w, g in zip(want, got):
        assert [(q["id"], q["kind"]) for q in w["tracks"]] == [(q["id"], q["kind"]) for q in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "range_m"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key


def test_cpp_assembler_matches_python(pings):
    emulator = FlankBoardEmulator(pings[:3], INTERVAL, block_samples=128, start_ns=10**18, gain_code=3)
    blocks = list(emulator.blocks())
    template = pings[0].copy()
    py_asm = PingAssembler(gs.sonar.SonarModel.load("flank_32", OVERRIDES))
    cpp_asm = core.PingAssemblerCpp(template)
    a, b = [], []
    for i, block in enumerate(blocks):
        if i == len(blocks) // 2:
            continue                                               # one lost block: both must drop the same ping
        a += py_asm.feed(block)
        b += cpp_asm.feed(block)
    assert len(a) == len(b) == 2 and cpp_asm.dropped_pings == py_asm.dropped_pings == 1
    assert cpp_asm.lost_samples == py_asm.lost_samples == 128
    for x, y in zip(a, b):
        assert np.array_equal(x.samples.array, y.samples.array)
        assert x.transmit.time_ns == y.transmit.time_ns and x.metadata.sequence == y.metadata.sequence
        assert x.samples.start_time == pytest.approx(y.samples.start_time)
    assert cpp_asm.feed(b"not a block") == [] and cpp_asm.bad_blocks == 1


# --- the standalone program ------------------------------------------------------------------

needs_exe = pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")


def _without_timestamps(pings):
    """The pings with every transmit time 0: a source without timestamps (sonar-4bs)."""
    out = copies(pings)
    for r in out:
        r.transmit.time_ns = 0
    return out


def _chain_run(cfg, records):
    with LiveChain(cfg) as chain:
        return chain.run(records)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_source_without_timestamps_is_counted_and_clocked_by_a_configured_ping_interval(pings, engine):
    """sonar-4bs, both engines. Pings whose transmit time does not increase over the ping before (a source without
    timestamps: all 0) give the tracker no clock. Each such ping is counted in the report's `pings_without_time` (on a
    line only once above 0, as pings_without_velocity) with one RuntimeWarning. With `tracker.ping_interval_s` given, the
    chain's clock advances by it on such a ping instead: the pings at 0 then give exactly the reports of the same pings
    with their own transmit times 2 s apart (tracks, detections). Ordinary pings (increasing transmit times) are not
    touched by the setting (bit for bit) and carry no new member."""
    cfg = dict(CHAIN, engine=engine)
    timed = _chain_run(cfg, copies(pings))
    clocked_cfg = dict(cfg, tracker=dict(CHAIN["tracker"], ping_interval_s=INTERVAL))
    assert all("pings_without_time" not in r for r in timed)
    for r_timed, r_set in zip(timed, _chain_run(clocked_cfg, copies(pings))):
        assert r_set["tracks"] == r_timed["tracks"] and r_set["detections"] == r_timed["detections"]
        assert "pings_without_time" not in r_set
    with pytest.warns(RuntimeWarning, match="pings_without_time"):
        bare = _chain_run(cfg, _without_timestamps(pings))
    assert [r.get("pings_without_time", 0) for r in bare] == list(range(N))
    keys = list(bare[-1])
    assert keys.index("pings_without_time") == keys.index("pings_without_nav") + 1
    with pytest.warns(RuntimeWarning, match="pings_without_time"):
        clocked = _chain_run(clocked_cfg, _without_timestamps(pings))
    assert [r.get("pings_without_time", 0) for r in clocked] == list(range(N))
    for r_timed, r_clocked in zip(timed, clocked):
        assert r_clocked["tracks"] == r_timed["tracks"] and r_clocked["detections"] == r_timed["detections"]
    # What happens without the setting: the crossing target (3 m/s at 250 m) is never a mover.
    assert any(t["kind"] == "mover" for t in timed[-1]["tracks"])
    assert not any(t["kind"] == "mover" for t in bare[-1]["tracks"])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_the_ping_interval_is_refused_unless_positive(bad):
    """tracker.ping_interval_s: None (off) or a finite number of seconds above 0, in both engines."""
    for engine in ("python", "cpp"):
        with pytest.raises(ValueError, match="ping_interval_s"):
            LiveChain(dict(CHAIN, engine=engine, tracker=dict(CHAIN["tracker"], ping_interval_s=bad)))
    with pytest.raises(ValueError, match="ping_interval_s"):
        core.LiveChainCpp(dict(CPP_CHAIN, tracker_ping_interval_s=bad))


@needs_exe
def test_program_counts_pings_without_time_and_takes_the_ping_interval(pings, tmp_path):
    """sonar-4bs through build/bin/sonar_live: a recording of pings all at transmit time 0 is counted
    (pings_without_time, a line on stderr once) and, with tracker_ping_interval_s, tracked as the timed pings are."""
    def replay(records, name, *extra):
        w = gs.RecordWriter(str(tmp_path / f"{name}.h5"))
        for r in records:
            w.write(r)
        w.close()
        out = subprocess.run([str(EXE), "--replay", str(tmp_path / f"{name}.h5"), "--report", str(tmp_path / f"{name}.jsonl"),
                              *EXE_OPTIONS, "classify_min_speed=1", *extra], capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        return [json.loads(line) for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()], out.stderr
    timed, err = replay(copies(pings), "timed")
    assert all("pings_without_time" not in r for r in timed) and "pings_without_time" not in err
    bare, err = replay(_without_timestamps(pings), "bare")
    assert [r.get("pings_without_time", 0) for r in bare] == list(range(N)) and err.count("pings_without_time") == 1
    clocked, _ = replay(_without_timestamps(pings), "clocked", f"tracker_ping_interval_s={INTERVAL}")
    for a, b in zip(timed, clocked):
        assert a["tracks"] == b["tracks"]


@needs_exe
def test_program_replays_a_recording_like_the_python_chain(pings, tmp_path):
    with LiveChain(CHAIN, recording=str(tmp_path / "rec.h5")) as chain:
        want = chain.run(copies(pings))
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), *EXE_OPTIONS,
                          "classify_min_speed=1"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    got = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    assert len(got) == N
    movers = [t for t in got[-1]["tracks"] if t["kind"] == "mover"]
    want_movers = [t for t in want[-1]["tracks"] if t["kind"] == "mover"]
    assert len(movers) == len(want_movers) == 1
    assert movers[0]["x"] == pytest.approx(want_movers[0]["x"], abs=0.1) and movers[0]["vx"] == pytest.approx(-3.0, abs=0.6)


@needs_exe
def test_program_takes_blocks_over_udp_with_navigation(pings, tmp_path):
    import socket
    import time
    template = pings[0].copy()
    w = gs.RecordWriter(str(tmp_path / "template.h5"))
    w.write(template)
    w.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    nav_port = port + 1
    proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", str(tmp_path / "template.h5"), "--nav-port", str(nav_port),
                             "--recording", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), "--pings", str(N),
                             *EXE_OPTIONS], stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(N + 1):                                     # navigation first: 1.5 m/s north
            t = i * INTERVAL
            tx.sendto(json.dumps({"time_ns": int(t * 1e9), "x": 1.5 * t, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 1.5}).encode(),
                      ("127.0.0.1", nav_port))
        sender = FlankBoardEmulator(pings, INTERVAL, block_samples=128).send(port, pace=0.25)
        sender.join(timeout=120.0)
        _, err = proc.communicate(timeout=120.0)
    finally:
        proc.kill()
    assert proc.returncode == 0, f"sonar_live ended abnormally ({proc.returncode}): {err.strip()[-400:]}"   # a crash is not lost data
    reports = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    if "0 dropped, 0 samples lost" not in err or len(reports) < N:
        pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    assert "0 pings without navigation" in err
    assert reports[-1]["ownship_xyz"][0] == pytest.approx(1.5 * (N - 1) * INTERVAL, abs=0.6)   # array centre, from navigation
    kinds = {t["kind"] for t in reports[-1]["tracks"]}
    assert "mover" in kinds and "static" in kinds
    raw = list(make_source({"type": "container", "path": str(tmp_path / "rec.h5")}))
    assert len(raw) == N and not np.iscomplexobj(raw[0].samples.array)   # recorded before conditioning


@needs_exe
def test_program_moves_navigation_to_the_array_by_the_lever_arm(pings, tmp_path):
    """sonar-3rv: the program's navigation (nav_lever_arm_m, nav_mount_rpy_deg) puts the array where the Python
    NavListener puts it: here 5 m ahead of the navigation unit on a vehicle heading north, so the array's reference
    point is 5 m further north than in test_program_takes_blocks_over_udp_with_navigation, and the recording holds
    the array's pose (a replay applies no navigation, so the offset is applied once)."""
    import socket
    import time
    w = gs.RecordWriter(str(tmp_path / "template.h5"))
    w.write(pings[0].copy())
    w.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    nav_port = port + 1
    fixes = [{"time_ns": int(i * INTERVAL * 1e9), "x": 1.5 * i * INTERVAL, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 1.5} for i in range(N + 1)]
    proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", str(tmp_path / "template.h5"), "--nav-port", str(nav_port),
                             "--recording", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), "--pings", str(N),
                             *EXE_OPTIONS, "nav_lever_arm_m=5,0,0"], stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for f in fixes:
            tx.sendto(json.dumps(f).encode(), ("127.0.0.1", nav_port))
        sender = FlankBoardEmulator(pings, INTERVAL, block_samples=128).send(port, pace=0.25)
        sender.join(timeout=120.0)
        _, err = proc.communicate(timeout=120.0)
    finally:
        proc.kill()
    assert proc.returncode == 0, f"sonar_live ended abnormally ({proc.returncode}): {err.strip()[-400:]}"
    reports = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    if "0 dropped, 0 samples lost" not in err or len(reports) < N:
        pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    assert reports[-1]["ownship_xyz"][0] == pytest.approx(1.5 * (N - 1) * INTERVAL + 5.0, abs=0.6)
    from gpusonar.live import NavListener
    raw = list(make_source({"type": "container", "path": str(tmp_path / "rec.h5")}))
    nav = NavListener(lever_arm_m=[5.0, 0.0, 0.0])
    for f in fixes:
        nav.push(f)
    for rec in raw:
        pos, rpy, _ = nav.at(rec.transmit.time_ns)
        assert np.allclose(rec.geometry.rx_pose.position, pos, atol=1e-9) and np.allclose(rec.geometry.rx_pose.rpy, rpy, atol=1e-12)


@needs_exe
def test_program_counts_pings_whose_navigation_has_no_velocity_like_the_python_chain(pings, tmp_path):
    """sonar-8is: the program applies navigation in its UDP path; a ping whose fix had no velocity over ground is counted
    in pings_without_velocity on its lines and in its summary, as the Python LiveChain counts it on the same navigation
    stream (the first two fixes without velocity, the rest with it)."""
    import socket
    import time
    from gpusonar.live import NavListener
    template = pings[0].copy()
    w = gs.RecordWriter(str(tmp_path / "template.h5"))
    w.write(template)
    w.close()
    fixes = []
    for i in range(N + 1):
        t = i * INTERVAL
        fix = {"time_ns": int(t * 1e9), "x": 1.5 * t, "y": 0.0, "z": 0.0, "yaw": 0.0}
        fixes.append(fix if i < 2 else dict(fix, vx=1.5, vy=0.0, vz=0.0))
    nav = NavListener()
    for f in fixes:
        nav.push(f)
    stripped = copies(pings)
    from gpusonar.live import apply_pose
    for r in stripped:
        apply_pose(r, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.warns(RuntimeWarning, match="pings_without_velocity"):
        with LiveChain(CHAIN, nav=nav) as chain:
            want = [r.get("pings_without_velocity", 0) for r in chain.run(stripped)]
    # The pings at 0, 2 and 4 s take a fix without velocity (held from the first; interpolated with one of the first two).
    assert want == [1, 2, 3] + [3] * (N - 3)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    nav_port = port + 1
    proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", str(tmp_path / "template.h5"), "--nav-port", str(nav_port),
                             "--report", str(tmp_path / "rep.jsonl"), "--pings", str(N), *EXE_OPTIONS], stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for f in fixes:
            tx.sendto(json.dumps(f).encode(), ("127.0.0.1", nav_port))
        time.sleep(0.2)
        sender = FlankBoardEmulator(pings, INTERVAL, block_samples=128).send(port, pace=0.25)
        sender.join(timeout=120.0)
        _, err = proc.communicate(timeout=120.0)
    finally:
        proc.kill()
    assert proc.returncode == 0, f"sonar_live ended abnormally ({proc.returncode}): {err.strip()[-400:]}"
    reports = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    if "0 dropped, 0 samples lost" not in err or len(reports) < N:
        pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    assert [r.get("pings_without_velocity", 0) for r in reports] == want
    assert "navigation sends no velocity" in err and f"0 pings without navigation, {want[-1]} pings without velocity" in err


def test_both_engines_resample_with_the_same_weights():
    """sonar-fy6: the resampler's weights from three trig calls a sample, in both engines (libs/live resample_weights,
    sync._sinc_weights), over a dense set of fractions, the ones at the ends and the ones the chain makes: the same to
    a bit or two (each engine's own sine and cosine), and each within a bit or two of the formula."""
    from gpusonar import sync
    from test_calibration_sync import WEIGHT_BOUND, _chain_fractions, _direct_weights
    frac = np.concatenate([np.arange(100000) / 100000.0, _chain_fractions(),
                           [0.0, 1e-300, 1e-12, 0.5, np.nextafter(0.5, 1.0), np.nextafter(1.0, 0.0), 1.0 - 1e-12]])
    cpp, py = np.asarray(core.live_resample_weights(frac)), sync._sinc_weights(frac)
    assert cpp.shape == py.shape == (16, len(frac))
    assert np.abs(cpp - py).max() <= WEIGHT_BOUND, np.abs(cpp - py).max()
    assert np.abs(cpp - _direct_weights(frac)).max() <= WEIGHT_BOUND
    assert np.array_equal(cpp[:, len(frac) - 7], (np.arange(-7, 9) == 0).astype(float))      # frac = 0: the sample itself


def test_cpp_sync_matches_python_and_focuses_the_four_board_array():
    from gpusonar import sync
    from test_calibration_sync import four_board_ping
    sonar, rec, target, truth = four_board_ping(bearing_deg=10.0)
    x = rec.samples.array
    s_ = rec.samples
    s_.array = x + 2e-7 * np.random.default_rng(1).standard_normal(x.shape)
    rec.samples = s_
    ref = sonar.config["receive"]["reference"]
    want = sync.synchronise(rec.copy(), ref)
    opts = {"bearing_first_deg": -30.0, "bearing_last_deg": 30.0, "bearing_step_deg": 0.5, "range_cell_m": 0.5, "cfar_pfa": 1e-6,
            "cfar_guard_m": 5.0, "cfar_train_m": 20.0, "min_snr_db": 12.0, "multipath": False,
            "sync": {k: ref[k] for k in ("f_start", "f_stop", "duration", "lead", "tail")}}
    chain = core.LiveChainCpp(opts)
    rep = json.loads(chain.process(rec.copy()))
    assert rep["sync"]["ok"] and rep["sync"]["worst_residual_us"] < 1.0
    assert np.allclose(rep["sync"]["offsets_us"], want.estimate.offset * 1e6, atol=0.05)       # 50 ns
    assert np.allclose(rep["sync"]["drift_ppm"], want.estimate.drift_ppm, atol=0.2)
    assert max(abs(v) for v in rep["sync"]["offsets_us"]) > 100.0
    power, ranges, bearings = chain.last_map()
    ranges = np.array(ranges)
    inside = (ranges > 520.0) & (ranges < 680.0)
    ib, ir = np.unravel_index(np.argmax(power[:, inside]), power[:, inside].shape)
    assert ranges[inside][ir] == pytest.approx(600.0, abs=1.5) and np.degrees(bearings[ib]) == pytest.approx(10.0, abs=1.0)
    assert all(d["range_m"] > 450.0 + 0.5 * 1500.0 * (ref["lead"] + ref["duration"]) for d in rep["detections"])
    unsynced = core.LiveChainCpp({k: v for k, v in opts.items() if k != "sync"})
    unsynced.process(rec.copy())
    assert unsynced.last_map()[0][:, inside].max() < 0.5 * power[:, inside].max()


@needs_exe
def test_program_resolves_bearings_with_sbl_and_listens_passively(tmp_path):
    from test_doa import R, active_ping, passive_record
    w = gs.RecordWriter(str(tmp_path / "active.h5"))
    w.write(active_ping([2.0, 4.0]))
    w.close()
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "active.h5"), "doa=sbl", "bearing_first_deg=-30", "bearing_last_deg=30",
                          "cfar_pfa=1e-6", "cfar_guard_m=20", "cfar_train_m=60", "min_snr_db=12", "sbl_min_snr_db=10",
                          "sbl_dynamic_range_db=15"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    rep = json.loads(out.stdout.splitlines()[0])
    got = sorted(d["bearing_deg"] for d in rep["detections"] if abs(d["range_m"] - R) < 10.0)
    assert len(got) == 2 and got[0] == pytest.approx(2.0, abs=0.4) and got[1] == pytest.approx(4.0, abs=0.4)
    w = gs.RecordWriter(str(tmp_path / "passive.h5"))
    w.write(passive_record([(-20.0, 1.0), (15.0, 0.5)], seconds=2.0))
    w.close()
    out = subprocess.run([str(EXE), "--passive", str(tmp_path / "passive.h5"), "--band", "4000", "6000", "n_fft=256", "hop=128",
                          "n_average=12", "bearing_first_deg=-60", "bearing_last_deg=60", "sbl_min_snr_db=3"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    frames = [json.loads(line) for line in out.stdout.splitlines()]
    assert len(frames) >= 7 and frames[1]["time_s"] > frames[0]["time_s"]
    good = sum(1 for f in frames if len(f["bearings_deg"]) == 2 and abs(min(f["bearings_deg"]) + 20.0) < 0.6
               and abs(max(f["bearings_deg"]) - 15.0) < 0.6)
    assert good >= 0.8 * len(frames)


def test_one_deployment_file_runs_either_engine(pings, tmp_path):
    """engine: cpp puts libs/live behind the same LiveChain: same report fields, same tracks, taps and recording."""
    with LiveChain(CHAIN) as chain:
        want = chain.run(copies(pings))
    with LiveChain(dict(CHAIN, engine="cpp"), recording=str(tmp_path / "rec.h5"), report_path=str(tmp_path / "rep.jsonl"),
                   taps={"dir": str(tmp_path / "taps"), "pings": 1}) as chain:
        got = chain.run(copies(pings))
        assert chain.last["power"].shape[0] == len(chain.bearings)
    assert set(want[-1]) == set(got[-1]) and set(want[-1]["tracks"][0]) == set(got[-1]["tracks"][0])
    for w, g in zip(want, got):
        assert g["ping"] == w["ping"]
        assert [(t["id"], t["kind"]) for t in w["tracks"]] == [(t["id"], t["kind"]) for t in g["tracks"]]
    assert len((tmp_path / "rep.jsonl").read_text().splitlines()) == N
    stages = {p.name.split(".")[0].split("_", 1)[-1] for p in (tmp_path / "taps" / "ping0000").iterdir()}
    assert stages >= {"raw", "prepare", "beamform", "report"}
    with pytest.raises(ValueError, match="backprojection"):
        LiveChain(dict(CHAIN, engine="cpp", beamformer="capon"))    # farfield and mvdr run in C++ too: test_beamform_cpp.py


def test_python_chain_takes_sbl_bearings():
    from test_doa import R, active_ping
    cfg = {"bearings_deg": [-30.0, 30.0, 0.5], "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 60.0}, "min_snr_db": 12.0,
           "backend": "host", "doa": "sbl", "sbl": {"min_snr_db": 10.0, "dynamic_range_db": 15.0}}
    with LiveChain(cfg) as chain:
        rep = chain.process(active_ping([2.0, 4.0]))
    got = sorted(d["bearing_deg"] for d in rep["detections"] if abs(d["range_m"] - R) < 10.0)
    assert len(got) == 2 and got[0] == pytest.approx(2.0, abs=0.4) and got[1] == pytest.approx(4.0, abs=0.4)


@needs_exe
def test_exported_template_and_command_run_the_program(pings, tmp_path):
    """tools/sonar_live.py --export-template writes what build/bin/sonar_live needs and prints its command."""
    import shlex
    import sys
    import yaml
    sys.path.insert(0, str(ROOT / "tools"))
    import sonar_live as tool
    cfg = {"source": {"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": OVERRIDES, "port": 5799},
           "chain": {k: v for k, v in CHAIN.items() if k != "backend"},
           "output": {"report": str(tmp_path / "rep.jsonl")}}
    (tmp_path / "deploy.yaml").write_text(yaml.safe_dump(cfg))
    assert tool.main([str(tmp_path / "deploy.yaml"), "--export-template", str(tmp_path / "template.h5")]) == 0
    command = shlex.split(tool.export_template(cfg, str(tmp_path / "template.h5")))
    assert command[:5] == ["build/bin/sonar_live", "--udp", "5799", "--template", str(tmp_path / "template.h5")]
    t = gs.RecordReader(str(tmp_path / "template.h5")).read(0)
    assert t.samples.array.shape == pings[0].samples.array.shape and t.samples.sample_rate == pings[0].samples.sample_rate
    assert np.allclose(t.rx_positions() - t.rx_positions().mean(axis=0), pings[0].rx_positions() - pings[0].rx_positions().mean(axis=0))
    # Every option the tool prints is one the program accepts, and gives the Python chain's tracks on a replay.
    w = gs.RecordWriter(str(tmp_path / "pings.h5"))
    for r in copies(pings):
        w.write(r)
    w.close()
    options = [c for c in command if "=" in c]
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "pings.h5"), "--recording", str(tmp_path / "run_%Y.h5"), *options],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / f"run_{time.gmtime().tm_year}.h5").exists()          # strftime fields in output paths are filled
    with LiveChain(CHAIN) as chain:
        want = chain.run(copies(pings))
    got = [json.loads(line) for line in out.stdout.splitlines()]
    assert [(t["id"], t["kind"]) for t in want[-1]["tracks"]] == [(t["id"], t["kind"]) for t in got[-1]["tracks"]]


def test_sbl_mode_keeps_the_maps_detection_where_it_resolves_nothing():
    """A gate SBL is not run on (here: contrast threshold out of reach, or no gates allowed) keeps the map's detections."""
    from test_doa import active_ping
    base = {"bearings_deg": [-30.0, 30.0, 0.5], "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 60.0}, "min_snr_db": 12.0,
            "backend": "host", "engine": "cpp"}
    with LiveChain(base) as chain:
        want = chain.process(active_ping([2.0, 4.0]))["detections"]
    assert want
    for sbl in ({"min_contrast_db": 200.0}, {"max_gates": 0}):
        with LiveChain(dict(base, doa="sbl", sbl=sbl)) as chain:
            got = chain.process(active_ping([2.0, 4.0]))["detections"]
        key = lambda d: (round(d["range_m"], 3), round(d["bearing_deg"], 3))  # noqa: E731
        assert sorted(map(key, got)) == sorted(map(key, want))


@needs_exe
def test_program_listens_to_an_untriggered_board_stream(tmp_path):
    """--listen: blocks with no transmit mark, over UDP, become passive bearing frames as they arrive."""
    import socket
    import time
    from gpusonar.sources.flank_stream import FULL_SCALE_COUNTS, encode_block
    from test_doa import passive_record
    base = passive_record([(-20.0, 1.0), (15.0, 0.5)], seconds=2.0)
    # The boards send real samples: put the baseband record on its carrier at four times the rate.
    xb, fs_b, fc = base.samples.array, base.samples.sample_rate, base.samples.center_frequency
    up, n = 4, base.samples.array.shape[1]
    spec = np.zeros((xb.shape[0], up * n), dtype=complex)
    spec[:, :n // 2], spec[:, -(n // 2):] = np.fft.fft(xb, axis=1)[:, :n // 2], np.fft.fft(xb, axis=1)[:, -(n // 2):]
    t = np.arange(up * n) / (up * fs_b)
    real = np.real(up * np.fft.ifft(spec, axis=1) * np.exp(2j * np.pi * fc * t)[None, :]).astype(np.float32)
    rec = base.copy()
    s_ = rec.samples
    s_.array = real
    s_.kind = core.SampleKind.RealPassband
    s_.sample_rate = up * fs_b
    s_.start_time = 0.0
    rec.samples = s_
    w = gs.RecordWriter(str(tmp_path / "template.h5"))
    w.write(rec)
    w.close()
    full_scale = rec.metadata.adc_full_scale
    scale = FULL_SCALE_COUNTS / (full_scale if np.isfinite(full_scale) else 1.0)
    x = real.T                                                     # [samples, channels]
    counts = np.rint(x * min(scale, 0.2 * FULL_SCALE_COUNTS / np.abs(x).max())).astype(np.int32)
    N_FFT, HOP = 1024, 512                                         # the same 32 ms blocks at the higher rate
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    n_frames = len(counts) // (N_FFT + 11 * HOP)
    proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", str(tmp_path / "template.h5"), "--listen", "--band", "4000", "6000",
                             "--recording", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), "--pings", str(n_frames),
                             f"n_fft={N_FFT}", f"hop={HOP}", "n_average=12", "bearing_first_deg=-60", "bearing_last_deg=60", "sbl_min_snr_db=3"],
                            stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i, first in enumerate(range(0, len(counts), 128)):
            tx.sendto(encode_block(counts[first:first + 128], first, -1, 0, 0, 0, 5_000_000_000), ("127.0.0.1", port))
            if i % 4 == 3:
                time.sleep(0.002)
        _, err = proc.communicate(timeout=120.0)
    finally:
        proc.kill()
    assert proc.returncode == 0, f"sonar_live ended abnormally ({proc.returncode}): {err.strip()[-400:]}"   # a crash is not lost data
    frames = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    if "0 dropped, 0 samples lost" not in err or len(frames) < n_frames:
        pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    assert n_frames >= 5 and frames[0]["time_ns"] >= 5_000_000_000 and frames[1]["time_ns"] > frames[0]["time_ns"]
    good = sum(1 for f in frames if len(f["bearings_deg"]) == 2 and abs(min(f["bearings_deg"]) + 20.0) < 0.6
               and abs(max(f["bearings_deg"]) - 15.0) < 0.6)
    assert good >= 0.8 * len(frames)
    raw = list(make_source({"type": "container", "path": str(tmp_path / "rec.h5")}))
    assert len(raw) == n_frames and raw[0].samples.array.shape[1] == N_FFT + 11 * HOP


# --- the program's key=value interface and its report lines ---------------------------------------

def _every_option_chain(tmp_path):
    """A chain section that makes cpp_chain_options emit every key it can, each away from its default where it can be."""
    from gpusonar import buildspec as bs
    chain = bs.chain_config(bs.load("popoto_face_1x32"))         # coverage (with the site), sync, decimate
    chain.update(
        look="port", window="rect", backend="host", beamformer="farfield", doa="sbl", cell_statistic="multiscale", min_snr_db=10.5,
        range_cell_m=1.5, sound_speed=1490.0, dead_channels=[3, 7], range_doppler="full",
        cfar={"pfa": 2e-6, "guard_m": 90.0, "train_m": 280.0, "false_alarms_per_ping": 2e-3, "integrate_cells": [1, 4, 16], "merge_m": 7.0,
              "search_factor": 2.5, "sidelobe_margin_db": 9.0, "sidelobe_floor_db": -35.0},
        sbl={"grid_step_deg": 0.5, "max_sources": 5, "min_snr_db": 7.0, "dynamic_range_db": 18.0, "max_gates": 20, "min_contrast_db": 7.0,
             "max_iterations": 25, "tolerance": 0.02},
        tracker={"range_sigma": 4.0, "bearing_sigma_deg": 0.8, "confirm_hits": 4, "max_misses": 3, "initial_speed_sigma": 6.0, "process_noise": 0.07, "ping_interval_s": 1.5,
                 "association": "in_order"},
        classify={"min_speed": 1.2, "static_speed": 0.8, "max_fit_rms": 50.0,
                  "multipath": {"max_excess_range": 700.0, "bearing_deg": 2.5, "velocity": 2.0}},
        decimate={"output_rate": 20000.0, "taps": 101, "oversample": 2.5, "band_margin": 1.4},
        blanking={"recovery_s": 0.002, "taper_s": 0.001, "mode": "ignore"},
        multipath_depth={"bottom": {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}, "rays": True},
        elevation={"sector_deg": [-20.0, 25.0, 0.5], "sector_from_depth": {"bounces": 2, "margin_deg": 1.5},
                   "sector_from_rays": {"bounces": 3, "margin_deg": 2.0, "max_depth_m": 300.0},
                   "multipath_sigma": {"bounces": 1, "max_depth_m": 250.0}, "range_window_m": 6.0, "max_detections": 20, "arrival": "strongest",
                   "first_arrival_db": 7.0, "first_arrival_snr_db": 12.0, "min_quality": 0.5, "ambiguity_db": 1.5, "ambiguity_sigma": 2.5,
                   "alternatives_db": 5.0, "correct_bearing": False, "depth": True, "fold": False, "classify": False, "classify_sigma": 3.0,
                   "water_depth": 90.0, "water_depth_sigma_m": 4.0, "attitude_time": "arrival", "attitude_rate_sigma_deg_s": 0.3,
                   "gradient_per_s": 0.017, "gradient_sigma_per_s": 0.04, "surface_z": 0.5,
                   "ssp": {"depth": [0.0, 50.0, 100.0], "speed": [1500.0, 1495.5, 1493.25]}, "ray_method": "steps", "roll_sigma_deg": 0.2,
                   "depth_process_noise": 0.4, "row_tolerance_m": 0.03, "min_row_elements": 4,
                   "profile": {"ladder": ["built", "family", "prior"],
                               "built": {"bin_m": 3.0, "sample_sigma_m_s": 0.3, "age_sigma_m_s_per_hour": 0.6, "distance_sigma_m_s_per_km": 0.7,
                                         "max_age_s": 3600.0, "max_distance_m": 5000.0, "min_span_m": 6.0, "extrapolation_sigma_per_s": 0.01,
                                         "file": str(tmp_path / "profile_bins.txt"), "cast": str(tmp_path / "cast.txt")},
                               "supplied": {"sigma_m_s": 0.4},
                               "ensemble": {"gradients_per_s": [0.0, 0.017, -0.05], "extent_m": [20.0, 50.0], "sigma_members": False},
                               "family": {"kind": "mixed_layer", "gradient_per_s": 0.016, "layer_depth_m": 30.0, "thermocline_gradient_per_s": -0.2,
                                          "thermocline_thickness_m": 40.0, "sigma_m_s": 0.5}}})
    (tmp_path / "cast.txt").write_text("0 1500\n50 1495.5\n100 1493.25\n")
    chain["sync"] = dict(chain["sync"], min_quality=0.5, search_time=0.003, tolerance_us=2.0)
    chain["sync"]["reference"] = dict(chain["sync"]["reference"], dedicated=True, clip_level=1.5)
    return chain


def _same_options(a: dict, b: dict) -> list:
    """The keys on which two options_json dumps differ (numbers to 1e-9 relative: key=value text has 10 digits)."""
    def same(x, y):
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
        if isinstance(x, float) or isinstance(y, float):
            return x is not None and y is not None and float(x) == pytest.approx(float(y), rel=1e-9, abs=1e-12)
        return x == y
    return sorted(k for k in set(a) | set(b) if not same(a.get(k), b.get(k)))


@pytest.mark.parametrize("cell_statistic", ["multiscale", "peak"])
def test_every_chain_option_round_trips_through_binding_and_text(cell_statistic, tmp_path):
    """Every key cpp_chain_options emits gives the same C++ chain through the binding (LiveChainCpp's dictionary) and
    through the key=value text the tools write (program_options -> set_pair / load_options, as build/bin/sonar_live
    reads chain.options or its command line). Before: `decimate={}` and the profile's `True` / `[0.0, 0.017]` were read
    as 0 by the program, so the decimate stage and the ensemble were off and the first gradient read 0."""
    from gpusonar import live
    chain = dict(_every_option_chain(tmp_path), cell_statistic=cell_statistic)
    o = live.cpp_chain_options(chain, live.sync_reference(chain))
    flat = live.program_options(chain)                           # no record: the binding is given no detection-probability table either
    via_binding = json.loads(core.LiveChainCpp(dict(o)).options())
    via_text = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    assert _same_options(via_binding, via_text) == []
    defaults = json.loads(core.live_options_from_text(""))
    # Every key written is one the dump holds (cell_statistic is the dump's cell_peak / cell_multiscale) and every
    # value written is one the program ends up with (a key that read as 0 would differ from what was written).
    assert set(flat) - {"cell_statistic"} <= set(via_text)
    for key, text in flat.items():
        if key == "cell_statistic":
            continue
        got = via_text[key]
        if isinstance(got, bool):
            assert int(text) == int(got), key
        elif isinstance(got, list):
            want = [w for w in str(text).split(",") if w]
            assert len(want) == len(got) and all(str(g) == w if isinstance(g, str) else float(w) == pytest.approx(g, rel=1e-9) for g, w in zip(got, want)), key
        elif isinstance(got, (int, float)):
            assert float(text) == pytest.approx(got, rel=1e-9, abs=1e-12), key
        elif key == "coverage_classes":                             # name:ts_db:depth_lo:depth_hi;...
            parse = lambda t: [(c.split(":")[0], *map(float, c.split(":")[1:])) for c in t.split(";")]  # noqa: E731
            assert parse(text) == parse(got), key
        else:
            assert str(text) == got, key
    # The stages a deployment can switch on are on, and most settings are away from the defaults (the test means something).
    for key in ("decimate", "blanking", "sync", "elevation", "elevation_profile", "elevation_profile_ensemble", "multipath_depth", "coverage"):
        assert via_text[key] is True and defaults[key] is False, key
    assert via_text["elevation_profile_ensemble_gradients"] == pytest.approx([0.0, 0.017, -0.05])
    assert via_text["backend"] == "host" and via_text["look"] == "port" and via_text["window"] == "rect"
    assert via_text["range_doppler"] == "full" and defaults["range_doppler"] == "off"
    assert len(_same_options(via_text, defaults)) > 100


def test_program_options_carry_navigation_and_the_backend():
    """nav.surface_z, the water depth from altitude and the backend reach the program's key=value settings."""
    from gpusonar import live
    deploy = {"source": {"decimate": {"output_rate": 16000.0}, "sample_bytes": 2, "timeout": 12.0, "host": "127.0.0.1"},
              "nav": {"surface_z": -1.25, "keep_s": 90.0, "water_depth": {"source": "altitude", "dvl_offset_m": 0.4}},
              "output": {"report_udp_max_bytes": 9000}}
    for backend in ("auto", "host", "cuda"):
        flat = live.program_options({"backend": backend}, None, deploy)
        assert flat["backend"] == backend
    flat = live.program_options({"backend": "auto"}, None, deploy)
    got = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    assert got["nav_surface_z"] == -1.25 and got["nav_keep_s"] == 90.0 and got["nav_water_depth_from_altitude"] is True
    assert got["nav_dvl_offset_m"] == pytest.approx(0.4) and got["decimate"] is True and got["decimate_streaming"] is True
    assert got["decimate_output_rate"] == 16000.0 and got["stream_sample_bytes"] == 2 and got["idle_seconds"] == 12.0
    assert got["bind"] == "127.0.0.1" and got["report_udp_max_bytes"] == 9000
    # auto means what LiveChain makes of it: CUDA when the build has it and a device is present.
    assert got["backend"] == ("cuda" if core.cuda_available() else "host")
    # The program without the key: auto, as the Python chain (owner's ruling 2026-11-21).
    assert json.loads(core.live_options_from_text(""))["backend"] == ("cuda" if core.cuda_available() else "host")


@pytest.mark.parametrize("pair", ["decimate={}", "elevation_profile_ensemble=None", "cfar_pfa=1e-5x", "range_cell_m=", "tracker_confirm_hits=2.5",
                                  "elevation_ssp_depth=0,abc", "look=Starboard", "window=hamming", "window=none", "backend=gpu",
                                  "blanking_mode=zeros", "beamformer=Farfield", "multipath_depth_max_depth=deep", "coverage_n_ranges=1.5",
                                  "tonal_goertzel_lines=300,x", "nav_surface_z=sea"])
def test_program_options_refuse_what_they_cannot_read(pair):
    key = pair.split("=")[0]
    with pytest.raises(ValueError, match=key):
        core.live_options_from_text(pair)


def test_program_options_take_booleans_as_words_or_numbers():
    for text in ("1", "true", "True", "2"):
        assert json.loads(core.live_options_from_text(f"elevation_profile_ensemble={text}"))["elevation_profile_ensemble"] is True
    for text in ("0", "false", "False"):
        got = json.loads(core.live_options_from_text(f"cell_multiscale={text}\nmultipath={text}"))
        assert got["cell_multiscale"] is False and got["multipath"] is False
    got = json.loads(core.live_options_from_text("cfar_integrate_cells=[1, 4, 16]\nelevation_profile_ensemble_gradients=[0.0, 0.017]"))
    assert got["cfar_integrate_cells"] == [1, 4, 16] and got["elevation_profile_ensemble_gradients"] == [0.0, 0.017]


@pytest.mark.parametrize("key,value", [("look", "Starboard"), ("window", "hamming"), ("backend", "gpu"), ("beamformer", "Farfield")])
@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_both_engines_refuse_unknown_words(key, value, engine):
    """`look: Starboard` silently looked to port and a misspelt window was rectangular; now both engines refuse them."""
    from gpusonar import live
    with pytest.raises(ValueError, match=key):
        LiveChain({key: value, "engine": engine})
    with pytest.raises(ValueError, match=key):
        live.cpp_chain_options({key: value})


def test_report_fields_match_the_python_chain(pings, tmp_path):
    """The program's raw report lines (and the binding's) carry the Python chain's fields, pings_without_nav included."""
    from gpusonar import live
    with LiveChain(dict(CHAIN, engine="python"), recording=str(tmp_path / "rec.h5")) as chain:
        want = chain.run(copies(pings))
    cpp = core.LiveChainCpp(dict(live.cpp_chain_options(CHAIN), backend=core.Backend.Host))
    raw = [json.loads(cpp.process(p)) for p in copies(pings)]
    lines = {"binding": raw}
    if EXE.exists():
        flat = live.program_options(CHAIN)
        out = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), *[f"{k}={v}" for k, v in flat.items()]],
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        lines["program"] = [json.loads(line) for line in out.stdout.splitlines()]
    for name, got in lines.items():
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert set(g) == set(w), (name, set(g) ^ set(w))
            assert g["pings_without_nav"] == w["pings_without_nav"] == 0
            for part in ("detections", "tracks"):
                for gi, wi in zip(sorted(g[part], key=lambda d: d.get("range_m", 0.0)), sorted(w[part], key=lambda d: d.get("range_m", 0.0))):
                    assert set(gi) == set(wi), (name, part, set(gi) ^ set(wi))


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_backend_cuda_without_a_device_is_an_error_and_auto_falls_back(engine):
    """Owner's ruling 2026-11-21: asking for cuda means cuda; only auto runs on the host when there is no device."""
    from gpusonar import pipeline
    if gs.core.cuda_available():
        assert LiveChain({"engine": engine, "backend": "cuda"}).backend == gs.core.Backend.Cuda
        return
    with pytest.raises(RuntimeError, match="no CUDA device is usable"):
        LiveChain({"engine": engine, "backend": "cuda"})
    assert LiveChain({"engine": engine, "backend": "auto"}).backend == gs.core.Backend.Host
    with pytest.raises(RuntimeError, match="no CUDA device is usable"):
        pipeline.run({"backend": "cuda", "steps": []})


@pytest.mark.parametrize("drop", [(), (3,), (5,)])
def test_both_engines_fit_the_measured_positions_of_a_noise_free_12km_crossing(drop):
    """D-157 in both engines (test_classify.test_a_noise_free_12km_crossing_is_reported_where_the_ship_is): the same
    measured rows (NaN on the missed ping), the same class, position and velocity, within 2 m of the ship at the last
    ping's transmit time (the fit through the filtered history, before D-157: 38.7 m behind)."""
    from test_classify import LIVE_TRACKER_12KM, popoto_12km_noise_free
    py, cpp = detect.Tracker(**LIVE_TRACKER_12KM), core.LiveTracker(**LIVE_TRACKER_12KM)
    for t, d in popoto_12km_noise_free(drop=drop):
        dets = [] if d is None else [d]
        a = py.update(t, dets)
        b = cpp.update(t, [(q.range, q.bearing, q.snr_db, q.cells) for q in dets], [q.time for q in dets])
    (tr,), (cr,) = a, b
    assert np.allclose(np.array(tr.measured), np.array(cr["measured"]), atol=1e-6, equal_nan=True)
    assert [bool(np.isnan(m[0])) for m in cr["measured"]] == [i in drop for i in range(6)]
    (want,) = classify.classify_tracks(a, time=99.5)
    (got,) = cpp.classify(time=99.5)
    assert want.kind == got["kind"]
    assert np.allclose(want.position, got["position"], atol=1e-6) and np.allclose(want.velocity, got["velocity"], atol=1e-8)
    assert np.hypot(*(np.asarray(got["position"]) - (12000.0, 800.0 - 6.0 * 99.5))) < 2.0
    # No time given: at the last history row's time (its detection's, or the ping's, 99.5 s, when it missed the last
    # ping: drop (5,)), both engines.
    last = tr.history[-1][0]
    assert (last == 99.5) == (5 in drop)
    (want,) = classify.classify_tracks(a)
    (got,) = cpp.classify()
    assert np.allclose(want.position, got["position"], atol=1e-6)
    assert np.hypot(*(np.asarray(got["position"]) - (12000.0, 800.0 - 6.0 * last))) < 2.0


# --- sonar-fyj: the LFM range-Doppler coupling correction, opt-in, both engines --------------------------------------

FLANK_32_FACTOR = 0.5 * (6.0e3 + 10.0e3) * 1.0 / (10.0e3 - 6.0e3)     # f0 T / B of flank_32's up-chirp: 2 s


def _range_doppler_run(pings, engine, mode):
    cfg = dict(CHAIN, engine=engine) if mode is None else dict(CHAIN, engine=engine, range_doppler=mode)
    with LiveChain(cfg) as chain:
        return chain.run(copies(pings))


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_range_doppler_is_off_by_default_and_said_only_when_on(pings, engine):
    """Not given and `off` are the chain as before, byte for byte in the report (processing time aside)."""
    from gpusonar import live
    assert "range_doppler" not in live.cpp_chain_options(CHAIN) and live.cpp_chain_options(dict(CHAIN, range_doppler="full"))["range_doppler"] == "full"
    a, b = _range_doppler_run(pings, engine, None), _range_doppler_run(pings, engine, "off")
    strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731
    assert [strip(r) for r in a] == [strip(r) for r in b]


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_range_doppler_ownship_adds_the_factor_times_the_ownships_closing_speed_to_each_range(pings, engine):
    """`ownship`: each detection's range_m is the detector's plus f0 T / B times the ownship's speed along its line of
    sight (1.5 m/s along the array, x; a bearing b from broadside looks along (sin b, cos b)): the fixture's crossing
    target and static object, a few degrees either side of broadside, move by up to a metre, opposite ways."""
    off, on = _range_doppler_run(pings, engine, "off"), _range_doppler_run(pings, engine, "ownship")
    moved = []
    for a, b in zip(off, on):
        assert len(a["detections"]) == len(b["detections"])
        for da, db in zip(a["detections"], b["detections"]):
            assert db["bearing_deg"] == da["bearing_deg"]
            shift = FLANK_32_FACTOR * 1.5 * np.sin(np.radians(da["bearing_deg"]))
            assert db["range_m"] - da["range_m"] == pytest.approx(shift, abs=1e-6)
            moved.append(shift)
    assert max(moved) > 0.2 and min(moved) < -0.2


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_range_doppler_full_moves_each_track_by_its_own_closing_speed(pings, engine):
    """`full` is `ownship` with each reported track moved along the line of sight from the ownship by f0 T / B times
    its own closing speed, -v . u, from the fitted velocity; kinds, velocities and detections are those of `ownship`."""
    on, full = _range_doppler_run(pings, engine, "ownship"), _range_doppler_run(pings, engine, "full")
    shifts = []
    for a, b in zip(on, full):
        assert a["detections"] == b["detections"]
        assert [(t["id"], t["kind"], t["parent"]) for t in a["tracks"]] == [(t["id"], t["kind"], t["parent"]) for t in b["tracks"]]
        own = np.asarray(a["ownship_xyz"][:2])
        for ta, tb in zip(a["tracks"], b["tracks"]):
            d = np.array([ta["x"], ta["y"]]) - own
            u = d / np.hypot(*d)
            shift = FLANK_32_FACTOR * -float(np.array([ta["vx"], ta["vy"]]) @ u)
            assert (tb["x"], tb["y"]) == pytest.approx(tuple(np.array([ta["x"], ta["y"]]) + shift * u), abs=1e-6)
            assert (tb["vx"], tb["vy"]) == (ta["vx"], ta["vy"])
            assert tb["range_m"] == pytest.approx(ta["range_m"] + shift, abs=1e-6)
            shifts.append(shift)
    assert max(abs(v) for v in shifts) > 0.2          # the crossing target closes at about 0.5 m/s: about 1 m


def test_range_doppler_full_gives_the_same_report_in_both_engines(pings):
    want, got = _range_doppler_run(pings, "python", "full"), _range_doppler_run(pings, "cpp", "full")
    for w, g in zip(want, got):
        for a, b in zip(w["detections"], g["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
        assert [(t["id"], t["kind"], t["parent"]) for t in w["tracks"]] == [(t["id"], t["kind"], t["parent"]) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "range_m"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_range_doppler_refuses_an_unknown_word(engine):
    with pytest.raises(ValueError, match="range_doppler"):
        if engine == "python":
            LiveChain(dict(CHAIN, range_doppler="on"))
        else:
            core.LiveChainCpp(dict(CPP_CHAIN, range_doppler="on"))


# --- an elastic bottom (rock) in the depth and multipath models: both engines ---------------------------------------

def test_rock_bottom_is_elastic_and_equal_in_both_engines():
    """100 m of water over `rock` (the simulator's elastic bottom, A8), one row at 30 m, a target at 50 m depth 1 km
    away: the chain's coverage picture and multipath depth with `bottom: {sediment: rock}` give the same members in
    both engines (coverage exactly, the multipath depth to 0.05 m as test_multipath_depth.py holds sand), the options
    carry the row's shear to the C++ stage and the program's key=value text, and the elastic rock's coverage differs
    from the fluid rock's (`shear_speed: 0`): a weak class is seen less far. Before the port the multipath stage refused a named bottom and the
    coverage took rock as a fluid."""
    import warnings

    from gpusonar import live
    from gpusonar.sim import run_scenario
    from test_multipath_depth import KNOWN, scenario
    cfg = scenario(30.0, 50.0, 1000.0, ts=20.0, bottom_z=100.0, pings=3)
    cfg["propagation"]["bottom"] = {"sediment": "rock"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = run_scenario(cfg, engine="cpp").records
    cov = {"classes": [{"name": "sub", "ts_db": -30.0, "depth_m": [20.0, 90.0]}],        # weak: its range is set by the bottom's return
           "budget": {"source_level_db": 197.0, "noise_spectrum_db": 51.46, "array_gain_db": 15.05, "absorption_db_per_km": 0.32},
           "water_depth_m": 100.0, "bottom": {"sediment": "rock"}}
    rock = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "coverage": cov,
            "multipath_depth": dict(KNOWN, bottom={"sediment": "rock"}, water_depth=100.0, bottom_order=2)}
    fluid = dict(rock, coverage=dict(cov, bottom={"sediment": "rock", "shear_speed": 0.0}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = live.cpp_chain_options(rock)
        flat = live.program_options(rock)
    assert o["multipath_depth"]["bottom_shear_speed"] > 0.0 and o["coverage"]["bottom_shear_speed"] == o["multipath_depth"]["bottom_shear_speed"]
    text = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    for key in ("multipath_depth_bottom_shear_speed", "multipath_depth_bottom_shear_attenuation", "coverage_bottom_shear_speed", "coverage_bottom_shear_attenuation"):
        assert float(flat[key]) == pytest.approx(text[key], rel=1e-9) and text[key] > 0.0, key
    got = {}
    for engine in ("cpp", "python"):
        for name, chain in (("rock", rock), ("fluid", fluid)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with LiveChain(dict(chain, engine=engine)) as c:
                    got[engine, name] = c.run([r.copy() for r in records])
    for name in ("rock", "fluid"):
        a, b = got["cpp", name], got["python", name]
        assert [r["coverage"] for r in a] == [r["coverage"] for r in b], name
        n = 0
        for ra, rb in zip(a, b):
            assert len(ra["detections"]) == len(rb["detections"])
            for x, y in zip(ra["detections"] + ra["tracks"], rb["detections"] + rb["tracks"]):
                assert ("depth_multipath_m" in x) == ("depth_multipath_m" in y)
                if "depth_multipath_m" in x:
                    n += 1
                    assert x["depth_multipath_m"] == pytest.approx(y["depth_multipath_m"], abs=0.05)
                    assert x["depth_multipath_sigma_m"] == pytest.approx(y["depth_multipath_sigma_m"], rel=0.05, abs=0.02)
        assert n >= 1, name
    # the elastic rock returns less than the fluid one: the class is seen less far (755 against 897 m when written)
    reach = {name: got["cpp", name][0]["coverage"]["classes"][0]["max_range_m"] for name in ("rock", "fluid")}
    assert reach["rock"] < reach["fluid"] - 50.0, reach


def test_coverage_reverberation_is_equal_in_both_engines_and_off_is_as_before():
    """The coverage picture's reverberation term (sonar-w2w3): 100 m of water over sand, one row at 30 m, a Lambert
    -27 dB bottom and a Chapman-Harris surface at 5 m/s. Both engines print the same member (exactly: one C++ class),
    with `reverberation: included`, each class's noise-only reach beside its reach and which limits; the tables reach
    the program's key=value text. Without `coverage.reverberation` the member says the term is not included; with
    `false` it is the member as before the term existed (the same text less that one key)."""
    import warnings

    from gpusonar import live
    from gpusonar.sim import run_scenario
    from test_multipath_depth import scenario
    cfg = scenario(30.0, 50.0, 1000.0, ts=20.0, bottom_z=100.0, pings=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = run_scenario(cfg, engine="cpp").records
    cov = {"classes": [{"name": "sub", "ts_db": -10.0, "depth_m": [20.0, 90.0]}, {"name": "ship", "ts_db": 20.0, "depth_m": [2.0, 6.0]}],
           "budget": {"source_level_db": 197.0, "noise_spectrum_db": 51.46, "array_gain_db": 15.05, "absorption_db_per_km": 0.32},
           "water_depth_m": 100.0, "bottom": {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}}
    items = [{"mu_db": -27.0}, {"model": "chapman_harris", "wind_speed": 5.0}]
    base = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host"}
    chains = {"on": dict(base, coverage=dict(cov, reverberation={"items": items, "frequency_hz": 5000.0})),
              "unsaid": dict(base, coverage=cov), "off": dict(base, coverage=dict(cov, reverberation=False))}
    flat = live.program_options(chains["on"])
    text = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    assert text["coverage_reverb"] == 1 and len(text["coverage_reverb_bottom_strength"]) == len(text["coverage_reverb_grazing"]) ** 2
    assert text["coverage_reverb_surface_strength"] == pytest.approx([float(v) for v in flat["coverage_reverb_surface_strength"].split(",")], rel=1e-9)
    got = {}
    for engine in ("cpp", "python"):
        for name, chain in chains.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with LiveChain(dict(chain, engine=engine)) as c:
                    got[engine, name] = [r["coverage"] for r in c.run([r.copy() for r in records])]
    for name in chains:
        assert got["cpp", name] == got["python", name], name
    on = got["cpp", "on"][0]
    assert on["reverberation"] == "included" and on["beam_integral_deg"] > 0.0
    for c in on["classes"]:
        assert c["limited_by"] in ("noise", "reverberation") and c["max_range_m"] <= c["max_range_noise_only_m"]
    assert got["cpp", "unsaid"][0]["reverberation"].startswith("not included")
    assert "reverberation" not in got["cpp", "off"][0]
    assert all({k: v for k, v in a.items() if k != "reverberation"} == b for a, b in zip(got["cpp", "unsaid"], got["cpp", "off"]))


# --- the sweep's Doppler bank (D4; libs/live doppler.hpp, gpusonar/doppler.py): both engines ----------------------------

BANK = {"mode": "sweep_bank", "max_range_rate_m_s": 1.0, "report": True}


def _bank_run(pings, engine, rd=None, bank=BANK):
    cfg = dict(CHAIN, engine=engine, doppler=bank)
    if rd is not None:
        cfg["range_doppler"] = rd
    with LiveChain(cfg) as chain:
        return chain.run(copies(pings))


def _same_bank_reports(want, got):
    assert len(want) == len(got)
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"])
        for a, b in zip(w["detections"], g["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
            assert a["snr_db"] == pytest.approx(b["snr_db"], abs=0.05)
            assert a["range_rate_m_s"] == pytest.approx(b["range_rate_m_s"], abs=1e-9)
            assert a["range_rate_step_m_s"] == pytest.approx(b["range_rate_step_m_s"], abs=1e-12)
        assert [(t["id"], t["kind"], t["parent"]) for t in w["tracks"]] == [(t["id"], t["kind"], t["parent"]) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "range_m"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key


def test_the_doppler_bank_gives_the_same_report_in_both_engines(pings):
    """The fixture's pings (ownship at 1.5 m/s along the array: every beam's hypotheses centred on its own rate) through the
    Python chain and the C++ chain (the binding): the same detections, range rates and tracks; the stage's maps alike."""
    want, got = _bank_run(pings, "python"), _bank_run(pings, "cpp")
    _same_bank_reports(want, got)
    assert all("range_rate_m_s" in d for r in got for d in r["detections"])
    # The static object at -60 m across, 320 m out: a still scatterer seen from the array moving along +x has the range
    # rate -1.5 sin(b) (opening on this side).
    rates = [d["range_rate_m_s"] + 1.5 * np.sin(np.radians(d["bearing_deg"])) for d in got[-1]["detections"] if d["bearing_deg"] < -5.0]
    assert rates and all(abs(r) <= 0.5 * got[-1]["detections"][0]["range_rate_step_m_s"] + 1e-9 for r in rates)


@needs_exe
def test_program_runs_the_doppler_bank_like_the_python_chain(pings, tmp_path):
    """build/bin/sonar_live with the deployment's chain section as key=value options (program_options): the Python chain's
    detections and range rates on the replayed recording."""
    cfg = dict(CHAIN, doppler=BANK)
    with LiveChain(dict(cfg, engine="python"), recording=str(tmp_path / "rec.h5")) as chain:
        want = chain.run(copies(pings))
    flat = live_program_options(cfg)
    assert flat["doppler_mode"] == "sweep_bank" and flat["doppler_report"] == 1
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), *[f"{k}={v}" for k, v in flat.items()]],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    got = [json.loads(line) for line in out.stdout.splitlines()]
    _same_bank_reports(want, got)


def live_program_options(cfg):
    from gpusonar import live
    return live.program_options(cfg)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_with_the_bank_range_doppler_takes_the_banks_estimate_and_corrects_once(pings, engine):
    """With the bank on, a detection shows where it shows without the bank (every hypothesis aligned to the zero one):
    `ownship` then adds f0 T / B x the bank's closing speed (-range_rate: the target's part and the own ship's together),
    and `full` moves no track further (its target part is in each detection already): no double correction."""
    plain, own, full = _bank_run(pings, engine), _bank_run(pings, engine, "ownship"), _bank_run(pings, engine, "full")
    off = _range_doppler_run(pings, engine, None)
    moved = []
    for p, o, f, z in zip(plain, own, full, off):
        assert len(p["detections"]) == len(o["detections"])
        for dz in z["detections"]:                     # every detection without the bank is there with it, where it was
            near = [d for d in p["detections"] if abs(d["bearing_deg"] - dz["bearing_deg"]) < 0.5]
            assert near and min(abs(d["range_m"] - dz["range_m"]) for d in near) < 0.05
        # What the bank adds on these pings (targets 80 to 90 dB over the noise) are sidelobes of the mismatched hypotheses
        # 60 dB and more below the strongest echo, which the detector's sidelobe envelope (the matched pulse's) does not
        # explain (DOPPLER.md D4, what got worse).
        top = max(d["snr_db"] for d in p["detections"])
        for d in p["detections"]:
            if not any(abs(d["bearing_deg"] - dz["bearing_deg"]) < 0.5 and abs(d["range_m"] - dz["range_m"]) < 0.05 for dz in z["detections"]):
                assert d["snr_db"] < top - 50.0, d
        for dp, do in zip(p["detections"], o["detections"]):
            shift = FLANK_32_FACTOR * -dp["range_rate_m_s"]
            assert do["range_m"] - dp["range_m"] == pytest.approx(shift, abs=1e-6)
            moved.append(shift)
        assert {k: v for k, v in f.items() if k != "processing_seconds"} == {k: v for k, v in o.items() if k != "processing_seconds"}
    assert max(moved) > 0.2 and min(moved) < -0.2


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_doppler_bank_is_off_by_default_and_off_is_as_before(pings, engine):
    """Not given and `mode: off` are the chain as before, byte for byte in the report (processing time aside); nothing of
    the bank reaches the C++ options or the report line unless it is on; the range rate reaches the line only with
    `report` (owner question 6)."""
    from gpusonar import live
    assert not any(k.startswith("doppler") for k in live.cpp_chain_options(CHAIN))
    a = _range_doppler_run(pings, engine, None)
    with LiveChain(dict(CHAIN, engine=engine, doppler={"mode": "off"})) as chain:
        b = chain.run(copies(pings))
    strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731
    assert json.dumps([strip(r) for r in a]) == json.dumps([strip(r) for r in b])
    quiet = _bank_run(pings[:2], engine, bank=dict(BANK, report=False))
    assert not any("range_rate_m_s" in d for r in quiet for d in r["detections"])
