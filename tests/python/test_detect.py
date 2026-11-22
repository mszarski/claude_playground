"""Milestone 10 (part): adaptive beamforming, CFAR detection and tracking."""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import detect, flank
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sonar_equation import noise_rms_volts

C = 1500.0


def scene_ping(targets, ts_db, velocities=None, t=0.0, noise_db=None, seed=0, centre=2000.0, half=150.0,
               duration=0.1):
    """One flank_32 ping on targets given as (range, bearing in degrees)."""
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [centre - half, centre + half]}},
                                          "transmit": {"waveform": {"duration": duration}}})
    scene = Scene()
    for i, ((r, b), ts) in enumerate(zip(targets, ts_db)):
        b = np.radians(b)
        v = (0.0, 0.0, 0.0) if velocities is None else velocities[i]
        scene.add_points([r * np.array([np.sin(b), np.cos(b), 0.0])], target_strength_db=ts, velocity=v)
    rms = 0.0 if noise_db is None else noise_rms_volts(noise_db, -180.0, sonar.sample_rate)
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C}, noise_rms=rms,
                        adc=False, absorption_model="none", scenario_time=t, rng=np.random.default_rng(seed))
    return sonar, rec


# --- MVDR ------------------------------------------------------------------------------------------

def target_snr_db(bm, i_bearing, target_range):
    """Peak at the target over the mean power elsewhere along the same beam."""
    bm = bm.valid()
    row = np.abs(bm.data[i_bearing])
    i = int(np.argmin(np.abs(bm.ranges - target_range)))
    peak = row[i - 3:i + 4].max()
    away = np.r_[0:i - 300, i + 300:len(row)]
    return 10 * np.log10(peak**2 / np.mean(row[away] ** 2)), peak


def test_mvdr_is_distortionless():
    """With nothing to reject (white noise only), MVDR passes a target at the level the
    conventional beamformer gives, and its noise floor is no better."""
    sonar, rec = scene_ping([(2000.0, 6.0)], [0.0], noise_db=50.0, seed=2)
    bearings = np.radians(np.arange(-40.0, 40.01, 1.0))
    i_t = int(np.argmin(np.abs(bearings - np.radians(6.0))))
    snr_c, peak_c = target_snr_db(flank.farfield_beamform(rec, bearings), i_t, 2000.0)
    snr_m, peak_m = target_snr_db(flank.mvdr_beamform(rec, bearings), i_t, 2000.0)
    assert peak_m == pytest.approx(peak_c, rel=0.1)
    # Estimating a 32 x 32 covariance per sub-band from a single ping costs a few dB of
    # noise gain when there is nothing directional to reject.
    assert snr_c - 4.0 < snr_m < snr_c + 1.0


def test_mvdr_rejects_a_jammer_that_swamps_the_conventional_beamformer():
    """A broadband noise source 20 degrees off the bow, 30 dB above the ambient noise.
    Its sidelobe leakage hides a target in the conventional map; MVDR nulls it."""
    sonar, rec = scene_ping([(2000.0, 6.0)], [0.0], noise_db=50.0, seed=2)
    ambient = noise_rms_volts(50.0, -180.0, sonar.sample_rate)
    flank.add_jammer(rec, np.radians(-20.0), ambient * 10 ** (30 / 20), np.random.default_rng(9))
    bearings = np.radians(np.arange(-40.0, 40.01, 1.0))
    i_t = int(np.argmin(np.abs(bearings - np.radians(6.0))))
    i_j = int(np.argmin(np.abs(bearings - np.radians(-20.0))))
    conv = flank.farfield_beamform(rec, bearings)
    mvdr = flank.mvdr_beamform(rec, bearings)
    snr_c, peak_c = target_snr_db(conv, i_t, 2000.0)
    snr_m, peak_m = target_snr_db(mvdr, i_t, 2000.0)
    assert snr_m > snr_c + 10.0, (snr_c, snr_m)
    assert peak_m == pytest.approx(peak_c, rel=0.25)          # the target itself is untouched
    # Looking straight at the jammer, nothing is rejected: that is the look direction.
    jam_c = np.mean(np.abs(conv.valid().data[i_j]) ** 2)
    jam_m = np.mean(np.abs(mvdr.valid().data[i_j]) ** 2)
    assert 10 * np.log10(jam_m / jam_c) == pytest.approx(0.0, abs=1.5)


# --- CFAR ------------------------------------------------------------------------------------------

def test_cfar_false_alarm_rate_matches_its_design():
    rng = np.random.default_rng(0)
    # Noise only, with the mean level changing 20 dB across range: CFAR must not care.
    level = 10 ** (np.linspace(0, 2, 4000) / 10)
    power = rng.exponential(1.0, (200, 4000)) * level
    for pfa in (1e-2, 1e-3):
        hits = detect.cfar(power, pfa=pfa, guard=4, train=32)
        measured = hits[:, 100:-100].mean()
        assert measured == pytest.approx(pfa, rel=0.25), (pfa, measured)
    assert detect.cfar_threshold_factor(1e-3, 64) == pytest.approx(64 * (1e-3 ** (-1 / 64) - 1))


def test_normaliser_flattens_the_background():
    rng = np.random.default_rng(1)
    level = 10 ** (np.linspace(0, 3, 3000) / 10)
    snr = detect.normalise(rng.exponential(1.0, (50, 3000)) * level)
    quarters = snr.reshape(50, 4, -1).mean(axis=(0, 2))
    assert np.ptp(10 * np.log10(quarters)) < 0.5
    assert snr.mean() == pytest.approx(1.0, rel=0.05)


def test_detection_chain_finds_a_target_in_noise():
    sonar, rec = scene_ping([(2040.0, 12.0)], [10.0], noise_db=50.0, seed=3)
    bearings = np.radians(np.arange(-60.0, 60.01, 1.0))
    bm = flank.farfield_beamform(rec, bearings).valid()
    power = np.abs(bm.data) ** 2
    hits = detect.cfar(power, pfa=1e-6)
    found = detect.cluster(hits, power, bm.ranges, bearings)
    assert len(found) >= 1
    best = max(found, key=lambda d: d.snr_db)
    assert best.range == pytest.approx(2040.0, abs=1.0)
    assert np.degrees(best.bearing) == pytest.approx(12.0, abs=1.0)
    assert best.snr_db > 20.0
    # The target's own sidelobes also cross the threshold: same range, other bearings.
    # Nothing is detected at any other range (about 0.6 false alarms expected in this map).
    strays = [d for d in found if abs(d.range - 2040.0) > 5.0]
    assert len(strays) <= 2


def test_the_tail_of_a_compressed_record_is_not_valid_for_detection():
    """After pulse compression the last pulse length of the record has only part of the
    replica under it. Its noise level falls away, which a CFAR reads as a step."""
    sonar, rec = scene_ping([(2040.0, 12.0)], [10.0], noise_db=50.0, seed=3)
    bearings = np.radians(np.arange(-60.0, 60.01, 4.0))
    bm = flank.farfield_beamform(rec, bearings)
    assert bm.n_valid == bm.data.shape[1] - int(np.ceil(0.1 * rec.samples.sample_rate))
    tail_hits = detect.cfar(np.abs(bm.data) ** 2, pfa=1e-6)[:, bm.n_valid - 40:].sum()
    valid_far = detect.cfar(np.abs(bm.valid().data) ** 2, pfa=1e-6)[:, -400:].sum()
    assert tail_hits > 0 and valid_far == 0


# --- tracking --------------------------------------------------------------------------------------

def test_tracks_follow_a_moving_target():
    """A target crossing at 5 m/s, pinged every 4 s in noise, with false alarms about."""
    start = np.array([-150.0, 2000.0])
    velocity = np.array([5.0, -1.5])
    tracker = detect.Tracker(range_sigma=1.0, bearing_sigma=np.radians(0.7))
    bearings = np.radians(np.arange(-30.0, 30.01, 0.25))
    truth, confirmed_history = [], []
    rng = np.random.default_rng(0)
    for k in range(20):
        t = 4.0 * k
        p = start + velocity * t
        r, b = np.hypot(*p), np.degrees(np.arctan2(p[0], p[1]))
        sonar, rec = scene_ping([(r, b)], [8.0], velocities=[(velocity[0], velocity[1], 0.0)],
                                noise_db=52.0, seed=100 + k, centre=2000.0, half=200.0)
        bm = flank.farfield_beamform(rec, bearings).valid()
        power = np.abs(bm.data) ** 2
        found = detect.cluster(detect.cfar(power, pfa=1e-5), power, bm.ranges, bearings)
        # Keep the strongest detection per ping from the target's neighbourhood: its
        # sidelobe detections at the same range would otherwise start tracks of their own.
        found = sorted(found, key=lambda d: -d.snr_db)[:1]
        # Inject two false alarms per ping at random places, as clutter would.
        for _ in range(2):
            found.append(detect.Detection(float(rng.uniform(1850, 2150)), float(rng.uniform(-0.5, 0.5)), 15.0, 1))
        confirmed = tracker.update(t, found)
        # The echo left the target half a round trip after the ping: that is where it was seen.
        truth.append(p + velocity * (r / C))
        confirmed_history.append(confirmed)
    final = confirmed_history[-1]
    assert len(final) >= 1
    truth = np.array(truth)
    track = min(final, key=lambda tr: np.linalg.norm(tr.state[:2] - truth[-1]))
    # Position within a few metres at 2 km, velocity within 1 m/s.
    assert np.linalg.norm(track.state[:2] - truth[-1]) < 10.0
    assert np.linalg.norm(track.state[2:] - velocity) < 1.0
    assert track.hits >= 15
    # Random false alarms do not persist: at most one spurious confirmed track survives.
    assert len(final) <= 2
    # The track existed for most of the run.
    assert sum(any(tr.id == track.id for tr in c) for c in confirmed_history) >= 14


def _crossing_boat_tracks(association):
    """The crossing boat of `fls_170k_traffic` (seed 63, full size), its hull and its wake: the
    detections within 100 m of it, tracked with the FLS scenes' settings (tools/gallery.py)."""
    import json, pathlib
    from gpusonar import classify
    d = json.loads((pathlib.Path(__file__).parent / "data" / "crossing_boat_wake.json").read_text())
    tracker = detect.Tracker(range_sigma=1.5, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2,
                             initial_speed_sigma=6.0, process_noise=0.3, association=association)
    for t, hits in zip(d["times"], d["detections"]):
        tracks = tracker.update(t, [detect.Detection(float(np.hypot(x, y)), float(np.arctan2(y, x)), s, 1) for x, y, s in hits])
    end = np.array(d["boat_position_at_t0"]) + np.array(d["boat_velocity"]) * d["times"][-1]
    near = min(tracks, key=lambda q: np.hypot(q.state[1] - end[0], q.state[0] - end[1]))   # state: across, forward
    c = {k.id: k for k in classify.classify_tracks(tracks, min_speed=0.5, max_fit_rms=2.5)}[near.id]
    return float(np.hypot(near.state[1] - end[0], near.state[0] - end[1])), c.kind, c.speed


def test_the_tracker_keeps_a_crossing_boat_whose_wake_would_steal_its_detection():
    """D-141. The hull gives one strong detection a ping; the wake leaves detections fixed in the water 5 to 10 m apart.
    New tracks climbing the wake used to take the hull's detection, because each track in turn took its nearest
    detection ("in_order"): the track at the boat then read 15.4 m/s for a boat doing 4.0. Most probable pairs first
    ("likelihood", the default) keeps the hull's track on the hull."""
    err, kind, speed = _crossing_boat_tracks("likelihood")
    assert err < 1.5 and kind == "mover" and speed == pytest.approx(4.0, abs=0.6)
    err, kind, speed = _crossing_boat_tracks("in_order")          # the rule before D-141, kept by name
    assert speed > 10.0


def test_the_tracker_refuses_an_unknown_association():
    with pytest.raises(ValueError, match="association"):
        detect.Tracker(association="nearest")


def tracker_scenario():
    """Pings (time, [(range, bearing, snr, cells, depth, depth_sigma)]): three movers, a row of four, two standing points,
    clutter, detections dropped now and then (misses), one mover that leaves, depth on some, an uneven ping interval.
    tests/python/data/tracker_before_d156.json holds what the trackers gave for it before D-156 (python/gpusonar/detect.py
    and core.LiveTracker at commit 1be89d4)."""
    rng = np.random.default_rng(156)
    out = []
    for i in range(12):
        t = 0.5 * i + (0.1 if i > 6 else 0.0)
        truth = [(200.0 + 4.0 * t, 30.0), (150.0, -40.0 + 3.0 * t), (260.0 - 2.0 * t, 80.0 - 2.0 * t)]
        truth += [(120.0 + 12.0 * k + 2.0 * t, -90.0) for k in range(4)]
        truth += [(300.0, -20.0), (180.0, 100.0)]
        if i >= 8:
            truth = truth[1:]
        dets = []
        for k, (x, y) in enumerate(truth):
            if rng.random() < 0.15:
                continue
            x, y = x + 0.3 * rng.standard_normal(), y + 0.3 * rng.standard_normal()
            depth = (float(20.0 + rng.standard_normal()), 1.5) if k % 3 == 0 else (float("nan"), float("nan"))
            dets.append((float(np.hypot(x, y)), float(np.arctan2(y, x)), float(15.0 + 5.0 * rng.random()), 2 + k % 3) + depth)
        for _ in range(3):
            if rng.random() < 0.5:
                dets.append((float(rng.uniform(100, 350)), float(rng.uniform(-0.6, 0.6)), 13.0, 1, float("nan"), float("nan")))
        out.append((t, dets))
    return out


SCENARIO_TRACKER = dict(process_noise=0.3, range_sigma=1.5, bearing_sigma=float(np.radians(0.4)), confirm_hits=3, max_misses=2,
                        initial_speed_sigma=6.0)


def before_d156():
    import json
    import pathlib
    return json.loads((pathlib.Path(__file__).parent / "data" / "tracker_before_d156.json").read_text())


def _close_to_golden(actual, desired):
    """Equal to the golden to 1e-12 relative: bit for bit where it was made, a few ulps on another CPU."""
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(desired, dtype=float), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("association", ["likelihood", "in_order"])
@pytest.mark.parametrize("stamp", ["none", "ping"])
def test_detections_without_their_own_time_track_exactly_as_before(association, stamp):
    """D-156: a detection that carries no time (`time` None, the default) is taken at the ping's time, and the tracker
    gives what it gave before (bit for bit on the CPU that made the golden; to 1e-12 on
    others, whose vector instructions round in another order) (states, covariances, histories, depth, classes), as does a detection given
    the ping's time explicitly."""
    from gpusonar import classify
    want = before_d156()[association]
    tracker = detect.Tracker(**SCENARIO_TRACKER, association=association)
    for (t, dets), ping in zip(tracker_scenario(), want["pings"]):
        got = tracker.update(t, [detect.Detection(*d, time=None if stamp == "none" else t) for d in dets])
        assert [tr.id for tr in got] == [w["id"] for w in ping["python"]]
        for tr, w in zip(got, ping["python"]):
            _close_to_golden(tr.state, w["state"])
            _close_to_golden(tr.cov.ravel(), w["cov"])
            _close_to_golden(np.array(tr.history[-len(w["history"]):]), np.array(w["history"]))   # full on the last ping
            assert (tr.hits, tr.misses) == (w["hits"], w["misses"])
            _close_to_golden([tr.depth, tr.depth_var], [w["depth"], w["depth_var"]])
    # The classes of the golden are the fit through the filtered history (before D-157): the same tracks with their
    # measured positions left out, which classify_tracks then fits as it did. The default (the measured positions) is
    # held by test_classify and test_live_cpp.
    import dataclasses
    filtered = [dataclasses.replace(tr, measured=[]) for tr in got]
    for at in (None, t):
        classes = classify.classify_tracks(filtered, min_speed=0.5, max_fit_rms=2.5, time=at)
        assert [(c.id, c.kind) for c in classes] == [(w["id"], w["kind"]) for w in want["classes"]]
        _close_to_golden([c.position for c in classes], [w["position"] for w in want["classes"]])
        _close_to_golden([c.velocity for c in classes], [w["velocity"] for w in want["classes"]])



def classes_d157():
    import json
    import pathlib
    return json.loads((pathlib.Path(__file__).parent / "data" / "classes_d157.json").read_text())


@pytest.mark.parametrize("association", ["likelihood", "in_order"])
def test_the_default_classes_of_the_tracker_scenario_are_held(association):
    """D-157 review: the default classifier (the line through the measured positions) on the tracker scenario, held at
    1e-12 against tests/python/data/classes_d157.json (made by both engines with data/make_classes_d157.py, run with the repository root; the golden of
    tracker_before_d156.json holds the fit through the filtered history). The kinds are those of that golden."""
    from gpusonar import classify
    want = classes_d157()[association]
    tracker = detect.Tracker(**SCENARIO_TRACKER, association=association)
    for t, dets in tracker_scenario():
        got = tracker.update(t, [detect.Detection(*d) for d in dets])
    assert [(c["id"], c["kind"]) for c in want["last_ping"]["python"]] == [(w["id"], w["kind"]) for w in before_d156()[association]["classes"]]
    for name, at in (("none", None), ("last_ping", t)):
        w = want[name]["python"]
        classes = classify.classify_tracks(got, min_speed=0.5, max_fit_rms=2.5, time=at)
        assert [(c.id, c.kind) for c in classes] == [(c["id"], c["kind"]) for c in w]
        _close_to_golden([c.position for c in classes], [c["position"] for c in w])
        _close_to_golden([c.velocity for c in classes], [c["velocity"] for c in w])
        _close_to_golden([c.fit_rms for c in classes], [c["fit_rms"] for c in w])

def long_range_mover(n=10, interval=20.0, c=1500.0, start=(-60.0, 12000.0), velocity=(6.0, 0.0), drop=(), seed=4):
    """A ship at 12 km crossing at 6 m/s, seen by a monostatic sonar standing at the origin every `interval` s
    (more than the 16 s round trip): per ping (transmit time, the detection, where the ship is at the transmit time,
    when the sound reached it). The detection is the ship where it was then, transmit + R/c, R its range at that moment
    (solved by iteration), with 1 m of range and 0.005 deg of bearing noise. (across, forward) frame."""
    rng = np.random.default_rng(seed)
    start, velocity = np.asarray(start), np.asarray(velocity)
    out = []
    for k in range(n):
        tp = interval * k
        tr = tp
        for _ in range(8):
            tr = tp + np.hypot(*(start + velocity * tr)) / c
        p = start + velocity * tr
        r = float(np.hypot(*p)) + rng.normal(0.0, 1.0)
        b = float(np.arctan2(p[0], p[1])) + np.radians(0.005) * rng.standard_normal()
        out.append((tp, None if k in drop else detect.Detection(r, b, 20.0, 4, time=tp + r / c), start + velocity * tp, tr))
    return out


LONG_RANGE_TRACKER = dict(range_sigma=2.0, bearing_sigma=float(np.radians(0.01)), process_noise=0.05, initial_speed_sigma=8.0)


def test_a_long_range_mover_is_where_its_echoes_show_it():
    """D-156. Detections stamped at transmit + R/c: the track's position at the ping's transmit time (the report's,
    classify_tracks(time=...)) is within 3 m of the ship there, and its state, which holds at its last detection's
    time, within 3 m of the ship then. The same detections stamped at the transmit time (the rule before) put the ship
    v R / c = 6 m/s x 8 s = 48 m ahead of where it is at the transmit time, along its course."""
    from gpusonar import classify
    pings = long_range_mover()
    v = np.array([6.0, 0.0])
    errors = {}
    for stamp in ("reflection", "transmit"):
        tracker = detect.Tracker(**LONG_RANGE_TRACKER)
        for tp, d, _, _ in pings:
            if stamp == "transmit":
                d = detect.Detection(d.range, d.bearing, d.snr_db, d.cells)
            tracks = tracker.update(tp, [d])
        assert len(tracks) == 1
        tr = tracks[0]
        tp, d, at_transmit, t_reflect = pings[-1]
        (c,) = classify.classify_tracks(tracks, time=tp)
        errors[stamp] = np.asarray(c.position) - at_transmit
        assert c.speed == pytest.approx(6.0, abs=0.3)
        if stamp == "reflection":
            assert tr.time == pytest.approx(t_reflect, abs=0.01) and tr.history[-1][0] == tr.time
            assert np.hypot(*(tr.state[:2] - (at_transmit + v * (tr.time - tp)))) < 3.0
            assert [h[0] for h in tr.history] == pytest.approx([q[1].time for q in pings])
    assert np.hypot(*errors["reflection"]) < 3.0
    bias = 6.0 * 8.0
    assert errors["transmit"][0] == pytest.approx(bias, abs=4.0) and abs(errors["transmit"][1]) < 3.0


def test_misses_are_counted_per_ping_and_a_missed_track_waits_at_the_ping():
    """D-156: a track that takes no detection on a ping counts a miss and is predicted to that ping's time; its history
    then holds the ping's time, the others its detections' times; a new track is born at its detection's time."""
    pings = long_range_mover(drop=(5,))
    tracker = detect.Tracker(**LONG_RANGE_TRACKER, max_misses=2)
    for tp, d, _, _ in pings:
        tracks = tracker.update(tp, [] if d is None else [d])
        if tp == pings[0][0]:
            assert tracker.tracks[0].time == d.time and tracker.tracks[0].history == [(d.time, *d.xy())]
        if d is None:
            tr = tracker.tracks[0]
            assert tr.misses == 1 and tr.time == tp and tr.history[-1][0] == tp
    tr = tracks[0]
    assert tr.hits == 9 and tr.misses == 0 and len(tr.history) == 10
    want = [q[0] if q[1] is None else q[1].time for q in pings]
    assert [h[0] for h in tr.history] == want


@pytest.mark.parametrize("association", ["likelihood", "in_order"])
def test_a_detection_time_that_is_not_finite_is_the_pings(association):
    """A detection whose time is NaN (or infinite) is taken at the ping's time, as one with None, bit for bit; the C++
    tracker maps NaN so too (test_live_cpp.test_tracker_and_classifier_match_python, "nan")."""
    from gpusonar import classify
    stamps = [None, float("nan"), float("inf")]           # (a dict keyed by NaN would not find it again)
    trackers = [detect.Tracker(**SCENARIO_TRACKER, association=association) for _ in stamps]
    for t, dets in tracker_scenario():
        got = [tr.update(t, [detect.Detection(*d, time=k) for d in dets]) for k, tr in zip(stamps, trackers)]
    assert len(trackers[0].tracks) > 3
    for i, k in enumerate(stamps[1:], 1):
        assert len(trackers[i].tracks) == len(trackers[0].tracks)
        for a, b in zip(trackers[0].tracks, trackers[i].tracks):
            np.testing.assert_array_equal(a.state, b.state)
            np.testing.assert_array_equal(a.cov, b.cov)
            assert a.history == b.history and a.time == b.time
            np.testing.assert_array_equal([a.depth, a.depth_var], [b.depth, b.depth_var])
        want = classify.classify_tracks(got[0], min_speed=0.5, max_fit_rms=2.5)
        nan = classify.classify_tracks(got[i], min_speed=0.5, max_fit_rms=2.5, time=k)
        assert [(c.id, c.kind, c.position, c.velocity) for c in want] == [(c.id, c.kind, c.position, c.velocity) for c in nan]


def test_a_missed_track_is_never_moved_back_and_its_depth_variance_stays_positive():
    """D-156 with a ping interval (5 s) shorter than R/c (8 s at 12 km): a track that took a detection at t_k + R/c and
    misses ping k+1 already holds later than t_{k+1}; it keeps its state, time and depth variance (it is not predicted
    back, which shrank the depth variance by q_d^2 x 3 s, below zero, and put its history out of time order)."""
    pings = long_range_mover(n=14, interval=5.0, drop=(4, 7, 8))
    rng = np.random.default_rng(3)
    tracker = detect.Tracker(**LONG_RANGE_TRACKER)
    held = 0
    for tp, d, _, _ in pings:
        dets = [] if d is None else [detect.Detection(d.range, d.bearing, d.snr_db, d.cells, depth=30.0 + 0.3 * rng.standard_normal(),
                                                      depth_sigma=0.3, time=d.time)]
        before = {tr.id: (tr.time, tr.depth_var, tr.state.copy()) for tr in tracker.tracks}
        tracker.update(tp, dets)
        for tr in tracker.tracks:
            times = [h[0] for h in tr.history]
            assert times == sorted(times)
            assert tr.depth_var > 0.0
            if d is None and before.get(tr.id, (-np.inf,))[0] > tp:
                assert tr.time == before[tr.id][0] and tr.depth_var == before[tr.id][1]
                np.testing.assert_array_equal(tr.state, before[tr.id][2])
                held += 1
    assert held == 2


def test_prediction_to_an_earlier_time_still_grows_the_uncertainty():
    """Within one ping a detection can be earlier than the time a track holds (a nearer echo, D-156). The prediction to
    it grows the depth variance by q_d^2 |dt| (ours: a prediction never shrinks a variance), and the position process
    noise G G^T q^2 is positive semi-definite for either sign of dt."""
    tracker = detect.Tracker(process_noise=0.3, depth_process_noise=0.5)
    tr = detect.Track(1, np.array([0.0, 100.0, 2.0, -1.0]), np.diag([1.0, 1.0, 0.25, 0.25]), depth=20.0, depth_var=0.1, time=10.0)
    for to in (12.0, 8.0):
        x, cov, depth_var = tracker._predict(tr, to)
        assert depth_var == pytest.approx(0.1 + 0.25 * 2.0)
        assert np.linalg.eigvalsh(cov).min() > 0.0
        np.testing.assert_allclose(x, [2.0 * (to - 10.0), 100.0 - (to - 10.0), 2.0, -1.0])


def test_tracker_drops_a_target_that_disappears():
    tracker = detect.Tracker(confirm_hits=2, max_misses=2)
    for k in range(4):
        tracker.update(float(k), [detect.Detection(1000.0 + 2.0 * k, 0.1, 20.0, 3)])
    assert len(tracker.tracks) == 1 and tracker.tracks[0].confirmed
    for k in range(4, 8):
        tracker.update(float(k), [])
    assert tracker.tracks == []


# --- the C++ / CUDA CFAR kernel against the NumPy one ----------------------------------------

def _cfar_case():
    rng = np.random.default_rng(12)
    power = rng.exponential(1.0, size=(61, 3000)).astype(np.float32)
    power[30, 1500] = 400.0
    power[5, 3] = 300.0                      # near the edge: a short training window
    power[50, 2995] = 300.0
    return power


@pytest.mark.parametrize("guard, train", [(4, 32), (0, 8), (10, 100)])
def test_cfar_kernel_matches_numpy(guard, train):
    import gpusonar as gs
    power = _cfar_case()
    want = detect.cfar(power, pfa=1e-5, guard=guard, train=train)
    got, ratio = gs.core.cfar(power, pfa=1e-5, guard=guard, train=train, return_ratio=True)
    assert got.dtype == bool and got.shape == power.shape
    np.testing.assert_array_equal(got, want)
    assert got[30, 1500] and got[5, 3] and got[50, 2995]
    np.testing.assert_allclose(ratio, detect.normalise(power.astype(float), guard, train), rtol=2e-6)
    np.testing.assert_array_equal(detect.cfar(power, pfa=1e-5, guard=guard, train=train,
                                              backend=gs.core.Backend.Host), want)
    assert gs.core.cfar_threshold_factor(1e-5, 2 * train) == pytest.approx(
        detect.cfar_threshold_factor(1e-5, 2 * train))


def test_cfar_kernel_false_alarm_rate():
    import gpusonar as gs
    rng = np.random.default_rng(3)
    power = rng.exponential(1.0, size=(200, 5000)).astype(np.float32)
    rate = gs.core.cfar(power, pfa=1e-3, guard=4, train=32).mean()
    assert rate == pytest.approx(1e-3, rel=0.15)


@pytest.mark.skipif(not __import__("gpusonar").cuda_available(), reason="no CUDA device")
def test_cfar_kernel_on_the_gpu_is_identical():
    import gpusonar as gs
    power = _cfar_case()
    host, host_ratio = gs.core.cfar(power, pfa=1e-5, return_ratio=True)
    gpu, gpu_ratio = gs.core.cfar(power, pfa=1e-5, return_ratio=True, backend=gs.core.Backend.Cuda)
    np.testing.assert_array_equal(gpu, host)
    np.testing.assert_array_equal(gpu_ratio, host_ratio)


def test_flagged_cells_are_grouped_by_the_detectors_rule_and_cluster_merges_only_when_asked():
    """`detect.group_flagged`: two flagged cells are one detection when no further apart than merge_cells + 1 cells in
    range and merge_u in sin(bearing); groups are closed under that; touching cells always belong together. The
    multiscale detector groups by it (tests/python/test_detector.py holds that to the C++ one); `detect.cluster` does so
    only with `merge=`, and is the touching-cells grouping it always was without."""
    bearings = np.radians(np.arange(-10.0, 10.01, 0.5))
    step = float(np.abs(np.diff(np.sin(bearings))).max())
    rows, cols = np.array([5, 5, 5, 5, 9, 20, 21]), np.array([10, 13, 17, 30, 13, 40, 41])
    def groups(merge_cells, merge_u):
        g = detect.group_flagged(rows, cols, bearings, merge_cells, merge_u)
        return [sorted(np.flatnonzero(g == k).tolist()) for k in sorted(set(g.tolist()), key=lambda k: np.flatnonzero(g == k)[0])]
    assert groups(0, 0.0) == [[0], [1], [2], [3], [4], [5, 6]]                      # only the touching pair (a diagonal)
    assert groups(2, 0.0) == [[0, 1], [2], [3], [4], [5, 6]]                        # 3 cells apart: within 2 + 1; 4 apart: not
    assert groups(3, 0.0) == [[0, 1, 2], [3], [4], [5, 6]]                          # ... closed under the rule: 10-13-17
    assert groups(3, 4.2 * step) == [[0, 1, 2, 4], [3], [5, 6]]                     # four bearing steps away at the same range
    assert groups(3, 3.8 * step) == [[0, 1, 2], [3], [4], [5, 6]]
    ranges = 100.0 + 0.5 * np.arange(60)
    power = np.ones((len(bearings), len(ranges)))
    mask = np.zeros(power.shape, dtype=bool)
    mask[rows, cols] = True
    power[mask] = 100.0
    plain = detect.cluster(mask, power, ranges, bearings, guard=2, train=8)
    assert [h.cells for h in plain] == [1, 1, 1, 1, 1, 2]
    merged = detect.cluster(mask, power, ranges, bearings, guard=2, train=8, merge=(3, 0.0))
    assert sorted(h.cells for h in merged) == [1, 1, 2, 3]
    three = max(merged, key=lambda h: h.cells)
    assert three.range == pytest.approx(float(ranges[[10, 13, 17]].mean())) and three.first_range == pytest.approx(float(ranges[10]))
