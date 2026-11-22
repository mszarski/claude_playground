"""Track classes from the track alone: mover, static, line, multipath, uncertain."""

import numpy as np
import pytest

from gpusonar import classify
from gpusonar.detect import Track


def track(tid, start, velocity, n=8, dt=0.5, sigma=0.2, hits=None, snr=20.0, seed=0):
    """A confirmed track whose history is a straight line plus position noise (across, forward order)."""
    rng = np.random.default_rng(seed + tid)
    t = np.arange(n) * dt
    xy = np.asarray(start, dtype=float) + t[:, None] * np.asarray(velocity, dtype=float) + sigma * rng.standard_normal((n, 2))
    tr = Track(tid, np.array([xy[-1, 0], xy[-1, 1], velocity[0], velocity[1]]), np.eye(4), hits=n if hits is None else hits,
               confirmed=True)
    tr.history = [(float(a), float(b), float(c)) for a, (b, c) in zip(t, xy)]
    tr.snr = [snr] * tr.hits
    return tr


def kinds(tracks, **kw):
    return {c.id: c.kind for c in classify.classify_tracks(tracks, **kw)}


def test_fit_recovers_velocity_and_its_uncertainty():
    tr = track(1, (10.0, 200.0), (3.0, -1.0), n=10, sigma=0.3)
    pos, v, rms, sigma = classify.fit_velocity(tr.history)
    assert v == pytest.approx([3.0, -1.0], abs=0.25) and rms == pytest.approx(0.3, rel=0.5)
    assert sigma == pytest.approx(rms / np.sqrt(np.sum((np.arange(10) * 0.5 - 2.25) ** 2)))
    assert classify.fit_velocity(tr.history[:2])[3] == np.inf        # two points say nothing


def test_movers_and_static_objects():
    k = kinds([track(1, (0.0, 200.0), (4.0, 0.0)), track(2, (50.0, 150.0), (0.0, 0.0)), track(3, (-40.0, 120.0), (0.05, 0.1))])
    assert k == {1: "mover", 2: "static", 3: "static"}


def test_noise_alone_does_not_make_a_mover():
    # Twenty tracks standing still, positions known to 1 m, eight pings half a second apart:
    # fitted speeds of a few tenths of a metre a second arise by chance and must not count.
    still = [track(i, (10.0 * i, 200.0), (0.0, 0.0), sigma=1.0, seed=7) for i in range(20)]
    assert sum(k == "mover" for k in kinds(still).values()) == 0


def test_too_few_points_or_jumping_positions_are_uncertain():
    assert kinds([track(1, (0.0, 100.0), (4.0, 0.0), n=3)]) == {1: "uncertain"}
    assert kinds([track(1, (0.0, 100.0), (4.0, 0.0), sigma=6.0)], max_fit_rms=2.5) == {1: "uncertain"}
    seldom = track(2, (0.0, 100.0), (0.0, 0.0), hits=3)              # seen on three pings of eight
    assert kinds([seldom]) == {2: "uncertain"}


def test_sliding_along_a_row_of_tracks_is_a_line_not_a_mover():
    # A pipeline along the forward axis: five detections sliding along it at 2 m/s.
    row = [track(i, (0.0, 100.0 + 12.0 * i), (0.0, 2.0)) for i in range(5)]
    boat = track(9, (60.0, 150.0), (-4.0, 0.0))                      # crossing, well clear of the row
    k = kinds(row + [boat], linear={"length": 40.0, "width": 3.0})
    assert [k[i] for i in range(5)] == ["mover", "line", "line", "line", "mover"]   # the ends have a neighbour on one side only
    assert k[9] == "mover"
    assert set(kinds(row + [boat]).values()) == {"mover"}             # without the test they are all movers


def test_a_boat_with_its_wake_astern_stays_a_mover():
    boat = track(1, (0.0, 200.0), (5.0, 0.0))
    wake = [track(10 + i, (-8.0 - 8.0 * i, 200.0), (0.0, 0.0)) for i in range(4)]   # astern only, not moving
    k = kinds([boat] + wake, linear={"length": 40.0, "width": 3.0})
    assert k[1] == "mover" and all(k[10 + i] == "static" for i in range(4))


def test_echoes_by_a_longer_path_are_attached_to_the_stronger_track():
    mp = {"max_excess_range": 800.0, "bearing": np.radians(3.0), "velocity": 2.5, "origin": (0.0, 0.0)}
    ship = track(1, (6000.0, 400.0), (0.0, -6.0), dt=12.0, n=6, sigma=5.0, snr=22.0)
    echo1 = track(2, (6080.0, 405.0), (0.0, -6.2), dt=12.0, n=6, sigma=5.0, snr=14.0)
    echo2 = track(3, (6150.0, 410.0), (0.0, -5.7), dt=12.0, n=6, sigma=5.0, snr=12.0)
    other = track(4, (6100.0, 1800.0), (0.0, -6.0), dt=12.0, n=6, sigma=5.0, snr=13.0)      # same speed, another bearing
    slow = track(5, (6090.0, 402.0), (0.0, 0.0), dt=12.0, n=6, sigma=5.0, snr=13.0)         # same bearing, not moving with it
    out = {c.id: c for c in classify.classify_tracks([ship, echo1, echo2, other, slow], min_speed=1.0, static_speed=1.0,
                                                      max_fit_rms=60.0, multipath=mp)}
    assert out[1].kind == "mover" and out[1].parent is None
    assert out[2].kind == out[3].kind == "multipath" and out[2].parent == out[3].parent == 1
    assert out[4].kind == "mover" and out[5].kind == "static"
    assert classify.summary(list(out.values())) == {"mover": 2, "static": 1, "line": 0, "multipath": 2, "uncertain": 0}


MP = {"max_excess_range": 800.0, "bearing": np.radians(3.0), "velocity": 2.5, "origin": (0.0, 0.0)}
SHIP_KW = dict(min_speed=1.0, static_speed=1.0, max_fit_rms=60.0, multipath=MP)


def arrival(tid, rng_m, bearing_deg, snr, n=6, velocity=(0.0, -6.0), seed=0):
    """A ship's arrival at `rng_m` (at the start) and bearing (from forward, to starboard), twelve seconds a ping."""
    b = np.radians(bearing_deg)
    return track(tid, (rng_m * np.sin(b), rng_m * np.cos(b)), velocity, dt=12.0, n=n, sigma=5.0, snr=snr, seed=seed)


def test_the_earliest_established_arrival_is_the_object_not_the_strongest():
    """D-155: the strongest arrival in a shallow channel can be a later path (popoto_12km: 50-83 m beyond the true range).
    Before, the strongest (2) was the object and the direct arrival (1), nearer than it, stayed a second mover."""
    out = {c.id: c for c in classify.classify_tracks([arrival(1, 6000.0, 4.0, 14.0), arrival(2, 6070.0, 4.0, 22.0),
                                                      arrival(3, 6130.0, 4.0, 12.0)], **SHIP_KW)}
    assert out[1].kind == "mover" and out[1].parent is None
    assert (out[2].kind, out[2].parent) == ("multipath", 1) and (out[3].kind, out[3].parent) == ("multipath", 1)
    assert classify.summary(list(out.values()))["mover"] == 1


def test_an_earlier_uncertain_arrival_is_multipath_not_the_object():
    """A three-point track nearer in range than the established ones is grouped with them, but is not made the object:
    the object is the earliest ESTABLISHED arrival (a short track's fit is poor). Before, it stayed uncertain."""
    early = arrival(4, 5990.0, 4.0, 13.0, n=3)
    out = {c.id: c for c in classify.classify_tracks([early, arrival(1, 6000.0, 4.0, 14.0), arrival(2, 6070.0, 4.0, 22.0)],
                                                     **SHIP_KW)}
    assert (out[4].kind, out[4].parent) == ("multipath", 1)
    assert out[1].kind == "mover" and (out[2].kind, out[2].parent) == ("multipath", 1)


def test_an_uncertain_track_alone_is_not_an_anchor():
    """No established track: nothing is grouped (an uncertain track explains nothing away)."""
    out = classify.classify_tracks([arrival(4, 5990.0, 4.0, 22.0, n=3), arrival(5, 6050.0, 4.0, 13.0, n=3)], **SHIP_KW)
    assert [(c.kind, c.parent) for c in out] == [("uncertain", None), ("uncertain", None)]


def test_two_ships_at_different_bearings_keep_their_own_groups():
    tracks = [arrival(1, 6000.0, 4.0, 14.0), arrival(2, 6070.0, 4.0, 22.0),
              arrival(11, 6040.0, 12.0, 13.0), arrival(12, 6100.0, 12.0, 20.0)]
    out = {c.id: c for c in classify.classify_tracks(tracks, **SHIP_KW)}
    assert out[1].kind == out[11].kind == "mover"
    assert (out[2].kind, out[2].parent) == ("multipath", 1) and (out[12].kind, out[12].parent) == ("multipath", 11)


def test_a_group_grows_through_any_member_not_only_the_anchor():
    """A chain: the anchor A (strongest) and B within the gates of each other, A and C not (1200 m apart in range, the
    gate 800), B and C within. The group grows from A through B to C: one object. Measured to the anchor alone, C was
    left a second mover (popoto_12km seeds 3 and 9)."""
    tracks = [arrival(1, 6000.0, 4.0, 22.0), arrival(2, 6600.0, 4.0, 14.0), arrival(3, 7200.0, 4.0, 13.0)]
    out = {c.id: c for c in classify.classify_tracks(tracks, **SHIP_KW)}
    assert out[1].kind == "mover" and out[1].parent is None
    assert (out[2].kind, out[2].parent) == ("multipath", 1) and (out[3].kind, out[3].parent) == ("multipath", 1)
    # The same chain in velocity: C differs from A by 3 m/s (gate 2.5), from B by 1.5.
    tracks = [arrival(1, 6000.0, 4.0, 22.0), arrival(2, 6200.0, 4.0, 14.0, velocity=(0.0, -7.5)),
              arrival(3, 6400.0, 4.0, 13.0, velocity=(0.0, -9.0))]
    out = {c.id: c for c in classify.classify_tracks(tracks, **SHIP_KW)}
    assert [(out[i].kind, out[i].parent) for i in (1, 2, 3)] == [("mover", None), ("multipath", 1), ("multipath", 1)]


def test_an_anchor_without_a_range_is_left_alone():
    """A NaN origin gives every track a NaN range: nothing is grouped and nothing raises (C++ does the same)."""
    out = classify.classify_tracks([arrival(1, 6000.0, 4.0, 14.0), arrival(2, 6070.0, 4.0, 22.0)],
                                   **dict(SHIP_KW, multipath=dict(MP, origin=(float("nan"), float("nan")))))
    assert [(c.kind, c.parent) for c in out] == [("mover", None), ("mover", None)]


def test_equal_ranges_take_the_lowest_id_as_the_object():
    a, b = arrival(7, 6000.0, 4.0, 14.0), arrival(3, 6000.0, 4.0, 20.0)
    b.history = [(t, x, y) for t, x, y in a.history]               # the same positions, so the same range exactly
    out = {c.id: c for c in classify.classify_tracks([a, b], **SHIP_KW)}
    assert out[3].kind == "mover" and (out[7].kind, out[7].parent) == ("multipath", 3)


# --- opt-in rules from the detections themselves (classify.py `detections`) ----------------------------------------

def with_cells(tr, cells):
    tr.cells = [int(c) for c in np.resize(np.asarray(cells), tr.hits)]
    return tr


def scene():
    """A row of weak small echoes sliding along a pipeline, a strong large boat running along it (a row of seabed
    clutter ahead and astern of it), a small fast glint, a large boat under way clear of everything, an anchored boat."""
    row = [with_cells(track(i, (0.0, 100.0 + 12.0 * i), (0.0, 2.0), snr=15.0), [2, 3]) for i in range(5)]
    crossing = with_cells(track(10, (1.0, 110.0), (0.0, 4.0), snr=26.0), [40, 44])
    glint = with_cells(track(11, (80.0, 60.0), (-6.0, 1.0), snr=14.0), [2, 1, 3])
    boat = with_cells(track(12, (-90.0, 200.0), (0.0, -4.0), snr=22.0), [12, 15])
    anchored = with_cells(track(13, (40.0, 220.0), (0.0, 0.0), snr=24.0), [20])
    return row + [crossing, glint, boat, anchored]


def test_detection_rules_off_change_nothing():
    tracks = scene()
    lin = {"length": 40.0, "width": 3.0}
    base = classify.classify_tracks(tracks, linear=lin)
    for off in (None, {}, {"min_mover_cells": None, "row_snr_db": None}):
        got = classify.classify_tracks(tracks, linear=lin, detections=off)
        assert [(c.id, c.kind, c.parent, c.speed) for c in got] == [(c.id, c.kind, c.parent, c.speed) for c in base]


def test_a_mover_with_small_detections_is_uncertain():
    k = kinds(scene(), linear={"length": 40.0, "width": 3.0}, detections={"min_mover_cells": 6})
    assert k[11] == "uncertain"                                      # the glint: 2 cells median
    assert k[12] == "mover" and k[13] == "static"                    # the large boat stays a mover; static untouched
    assert k[1] == k[2] == k[3] == "line"                            # the pipeline's row stays a line (rule after the row test)
    assert kinds(scene(), linear={"length": 40.0, "width": 3.0})[11] == "mover"     # off: the glint is a mover
    no_cells = track(20, (80.0, 60.0), (-6.0, 1.0))                  # nothing recorded: the rule cannot judge it
    assert kinds([no_cells], detections={"min_mover_cells": 6}) == {20: "mover"}
    assert classify.classify_tracks(scene())[5].extent_cells == pytest.approx(42.0)


def test_a_strong_boat_running_along_a_row_of_weak_echoes_is_not_part_of_the_row():
    lin = {"length": 40.0, "width": 3.0}
    assert kinds(scene(), linear=lin)[10] == "line"                  # off: the corridor test takes it for the row
    k = kinds(scene(), linear=lin, detections={"row_snr_db": 6.0})
    assert k[10] == "mover"                                          # 11 dB above the row
    assert k[1] == k[2] == k[3] == "line"                            # the row's own echoes are within 6 dB of each other


def test_detection_rules_refuse_unknown_keys_and_negative_values():
    with pytest.raises(ValueError, match="unknown"):
        classify.classify_tracks(scene(), detections={"min_cells": 6})
    with pytest.raises(ValueError, match="zero or more"):
        classify.classify_tracks(scene(), detections={"row_snr_db": -1.0})


def test_positions_are_read_at_the_time_asked():
    """D-156: classify_tracks(time=...) gives each track's fitted position at that time (the live report's, the ping's
    transmit time), while a track's last point is its last detection's time, transmit + R/c; without `time`, at the
    last point, as before."""
    hist = [(0.0 + 8.0 + 20.0 * k, 100.0 + 6.0 * (8.0 + 20.0 * k), 12000.0) for k in range(6)]
    tr = Track(1, np.array([hist[-1][1], hist[-1][2], 6.0, 0.0]), np.eye(4), hits=6, history=hist)
    (at_last,) = classify.classify_tracks([tr])
    (at_ping,) = classify.classify_tracks([tr], time=100.0)
    assert at_last.position == pytest.approx((100.0 + 6.0 * 108.0, 12000.0))
    assert at_ping.position == pytest.approx((100.0 + 6.0 * 100.0, 12000.0))
    assert at_ping.velocity == at_last.velocity and at_ping.kind == at_last.kind



def test_a_time_that_is_not_finite_is_not_given():
    """time=NaN is time=None (the C++ ClassifyOptions::time NaN): each track's fitted position at its last point."""
    tracks = [track(1, (10.0, 200.0), (3.0, -1.0)), track(2, (50.0, 80.0), (0.0, 0.0))]
    for at in (float("nan"), float("inf")):
        assert classify.fit_velocity(tracks[0].history, at=at)[0] == pytest.approx(classify.fit_velocity(tracks[0].history)[0])
        want, got = classify.classify_tracks(tracks), classify.classify_tracks(tracks, time=at)
        assert [(c.kind, c.position, c.velocity) for c in want] == [(c.kind, c.position, c.velocity) for c in got]


def test_a_track_too_short_for_a_fit_is_moved_to_the_time_asked_by_its_state():
    """Fewer than three points: no fitted line, and with `time` the last point moved there by the track's Kalman state
    velocity; without a time (None, NaN), or without a state, the last point."""
    tr = Track(1, np.array([106.0, 50.0, 6.0, 0.0]), np.eye(4), hits=2, history=[(0.0, 46.0, 50.0), (10.0, 106.0, 50.0)])
    (c,) = classify.classify_tracks([tr], time=2.0)
    assert c.position == pytest.approx((58.0, 50.0)) and c.velocity == (0.0, 0.0) and c.kind == "uncertain"
    for at in (None, float("nan")):
        assert classify.classify_tracks([tr], time=at)[0].position == (106.0, 50.0)
    assert tuple(classify.fit_velocity(tr.history, at=2.0)[0]) == (106.0, 50.0)


def test_the_fit_needs_time_between_the_points_not_between_the_first_and_last():
    """The guard is max - min of the times (numpy.ptp; classify.cpp the same): points out of time order whose first and
    last share a time still give a line."""
    pos, v, rms, sigma = classify.fit_velocity([(0.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 0.0, 0.0)])
    assert v == pytest.approx([1.0, 0.0]) and np.isfinite(sigma)


# --- D-157: the fit goes through the measured positions ------------------------------------------------------------

C_12KM = 1496.82
LIVE_TRACKER_12KM = dict(range_sigma=15.0, bearing_sigma=float(np.radians(1.0)), confirm_hits=3, max_misses=2,
                         initial_speed_sigma=8.0, process_noise=0.05, association="likelihood")


def popoto_12km_noise_free(drop=()):
    """The popoto_12km geometry without noise: a ship crossing at 6 m/s 12 km away, six pings 19.9 s apart from an
    ownship making 1.5 m/s, each detection exact at transmit + R/c (R from the ownship, solved by iteration), placed
    about the first ping's position as the live chain places it. Per ping (transmit time, detection or None); the
    ship's true (across, forward) position at time T is (12000, 800 - 6 T). The tracker the live chain derives for that
    array (LIVE_TRACKER_12KM: 15 m and 1 deg sigmas, the deployment's other tracker settings). Ours (D-157), after the
    sonar-2hz diagnosis (scratchpad diag2hz/noisefree.py)."""
    from gpusonar import detect
    out = []
    for i in range(6):
        t = 19.9 * i
        own = np.array([1.5 * t, 0.0])
        tau = np.hypot(800 - 6 * t - own[0], 12000) / C_12KM
        for _ in range(4):
            tau = np.hypot(800 - 6 * (t + tau) - own[0], 12000) / C_12KM
        p = np.array([800 - 6 * (t + tau), 12000.0])            # world (x forward, y across) about the first ping's origin
        r = np.hypot(*(p - own))
        d = detect.Detection(float(np.hypot(*p)), float(np.arctan2(p[1], p[0])), 20.0, 1, time=t + r / C_12KM)
        out.append((t, None if i in drop else d))
    return out


def test_a_noise_free_12km_crossing_is_reported_where_the_ship_is():
    """D-157: exact detections of the popoto_12km ship through the live chain's tracker, classified at the last ping's
    transmit time: the reported position within 2 m of the ship along track and across. A track is born at zero
    velocity with a 1 deg bearing sigma (210 m at 12 km), so its filtered positions lag the ship; the fit through them
    (before D-157) put the ship 38.7 m behind along track and 1.1 m off across. The fit through the measured positions
    is exact, and its rms is zero."""
    from gpusonar import detect
    tracker = detect.Tracker(**LIVE_TRACKER_12KM)
    for t, d in popoto_12km_noise_free():
        tracks = tracker.update(t, [d])
    (tr,) = tracks
    assert len(tr.measured) == len(tr.history) == 6
    (c,) = classify.classify_tracks([tr], time=99.5)
    err = np.asarray(c.position) - (12000.0, 800.0 - 6.0 * 99.5)
    assert abs(err[1]) < 2.0 and abs(err[0]) < 2.0
    assert c.speed == pytest.approx(6.0, abs=0.01) and c.fit_rms < 1e-6
    # The filtered history alone (the fit before D-157) lags: the same tracker, the measurements left out.
    lag = np.asarray(classify.fit_velocity(tr.history, 99.5)[0]) - (12000.0, 800.0 - 6.0 * 99.5)
    assert lag[1] > 30.0


def test_rows_of_misses_are_left_out_of_the_fit():
    """D-157: a row of a ping the track missed carries no measurement (NaN) and is not fitted; the fit's degrees of
    freedom count the measured rows. Fewer than three measured rows: the short-track rule, from the last filtered point
    moved by the state's velocity."""
    from gpusonar import detect
    tracker = detect.Tracker(**LIVE_TRACKER_12KM)
    for t, d in popoto_12km_noise_free(drop=(3,)):
        tracks = tracker.update(t, [] if d is None else [d])
    (tr,) = tracks
    assert np.isnan(tr.measured[3]).all() and np.isfinite(np.asarray(tr.measured)[[0, 1, 2, 4, 5]]).all()
    assert classify.fitted_points(tr) == 5
    (c,) = classify.classify_tracks([tr], time=99.5)
    assert np.hypot(*(np.asarray(c.position) - (12000.0, 800.0 - 6.0 * 99.5))) < 1e-6
    # Two measured rows of five: no line; the last history point moved by the state's velocity.
    short = Track(1, np.array([10.0, 60.0, 2.0, 0.0]), np.eye(4), hits=2, history=[(0.0, 0.0, 50.0), (1.0, 3.0, 55.0), (2.0, 10.0, 60.0)],
                  measured=[(0.0, 50.0), (float("nan"), float("nan")), (11.0, 61.0)])
    pos, v, rms, sigma = classify.fit_velocity(short.history, 3.0, short.state[2:4], short.measured)
    assert tuple(pos) == (12.0, 60.0) and tuple(v) == (0.0, 0.0) and sigma == np.inf
    with pytest.raises(ValueError, match="measured rows"):
        classify.fit_velocity(short.history, None, None, short.measured[:2])


def test_without_a_time_a_track_that_missed_its_last_ping_is_reported_at_that_ping():
    """D-157 review: with no time given the position is the fitted line's at the track's last history row (its last
    ping), not at its last measured row. The ship missed the last ping (transmit 99.5 s): reported where it is at that
    row's time, not 11.9 s (71 m) behind, where it was at its last detection."""
    from gpusonar import detect
    tracker = detect.Tracker(**LIVE_TRACKER_12KM)
    for t, d in popoto_12km_noise_free(drop=(5,)):
        tracks = tracker.update(t, [] if d is None else [d])
    (tr,) = tracks
    assert np.isnan(tr.measured[-1]).all() and tr.history[-1][0] == 99.5
    for at in (None, float("nan")):
        (c,) = classify.classify_tracks([tr], time=at)
        assert np.hypot(*(np.asarray(c.position) - (12000.0, 800.0 - 6.0 * 99.5))) < 1e-6


# --- sonar-ljh: one glinting measurement is left out of the line --------------------------------------------------

def glinting_12km_track(glint=-50.0):
    """The noise-free popoto_12km crossing through the live chain's tracker, its measured positions given a small fixed
    scatter (+-3 m along track, +-1 m in range) and the last one moved `glint` m along track (negative: ahead of the
    ship, which runs toward -forward). The live chain measured the last ping 23 to 66 m ahead on popoto_12km seeds 51-60
    (the ship side-on, a coherent sum of its five points; D-157, sonar-ljh)."""
    from gpusonar import detect
    tracker = detect.Tracker(**LIVE_TRACKER_12KM)
    for t, d in popoto_12km_noise_free():
        tracks = tracker.update(t, [d])
    (tr,) = tracks
    scatter = np.array([(1.0, -3.0), (-1.0, 3.0), (0.5, 2.0), (-0.5, -2.0), (1.0, 1.0), (0.0, glint)])
    tr.measured = [tuple(np.asarray(m) + s) for m, s in zip(tr.measured, scatter)]
    return tr


def test_a_glinting_last_measurement_is_left_out_of_the_line():
    """sonar-ljh: a last measurement 80 m ahead along track (cross-range at 12 km) pulls the least-squares line through
    six points by its leverage at the end of the track (more than 15 m ahead, more than 6.4 m/s for 6.0 with every point
    in); the per-axis deleted-residual test leaves it out at the default false-alarm probability, and the line through the
    other five is within 3 m and 0.1 m/s. Changed in review: with each axis's own variance the six-point test has 3
    degrees of freedom and needs |t| > 13.7, so the 50 m glint this test used before (|t| = 11.9 over +-3 m of scatter)
    is now kept, and that is asserted too."""
    truth = np.array((12000.0, 800.0 - 6.0 * 99.5))
    tr = glinting_12km_track(glint=-80.0)
    (c,) = classify.classify_tracks([tr], time=99.5)
    assert classify.fitted_points(tr) == 5
    assert np.hypot(*(np.asarray(c.position) - truth)) < 3.0 and c.speed == pytest.approx(6.0, abs=0.1)
    assert c.kind == "mover"
    (every,) = classify.classify_tracks([tr], time=99.5, outlier_alpha=0.0)
    assert classify.fitted_points(tr, outlier_alpha=0.0) == 6
    assert truth[1] - every.position[1] > 15.0 and every.speed > 6.4          # pulled ahead and faster
    assert classify.fitted_points(glinting_12km_track(glint=-50.0)) == 6     # too little evidence in six points
    # The filtered history is not tested (its points are not independent): the measurements left out, nothing changes.
    hist = classify.fit_velocity(tr.history, 99.5)
    assert np.array_equal(hist[0], classify.fit_velocity(tr.history, 99.5, outlier_alpha=0.0)[0])


def test_the_outlier_test_leaves_gaussian_scatter_and_exact_lines_alone():
    """sonar-ljh: on straight tracks with Gaussian scatter of one variance in both coordinates the test leaves a point out
    of about alpha of them (Bonferroni over the points: at most alpha); on an exact line, and the noise-free 12 km
    crossing, of none."""
    rng = np.random.default_rng(7)
    t = np.arange(8) * 2.0
    hist = [(float(a), 0.0, 0.0) for a in t]
    for alpha, most in ((0.01, 0.02), (0.1, 0.12)):
        out = 0
        for _ in range(3000):
            xy = np.array([100.0, 2000.0]) + t[:, None] * [3.0, -1.0] + 5.0 * rng.standard_normal((8, 2))
            out += classify._fit(hist, None, None, xy, alpha)[4] == 7
        assert 0.3 * alpha < out / 3000 < most
    exact = [(float(a), 100.0 + 3.0 * a, 2000.0 - a) for a in t]
    assert classify._fit(exact, None, None, [p[1:] for p in exact], 0.5)[4] == 8
    from gpusonar import detect
    tracker = detect.Tracker(**LIVE_TRACKER_12KM)
    for at, d in popoto_12km_noise_free():
        tracks = tracker.update(at, [d])
    assert classify.fitted_points(tracks[0]) == 6


def test_the_t_tail_is_scipys():
    """classify.t_two_sided (the finite sums libs/live classify.cpp computes too) against scipy's Student's t, wherever
    the probability is above 1e-12 (the test compares it with alpha / m, 1e-2 to 1e-6)."""
    from scipy import stats
    for nu in range(1, 120):
        for t in (0.0, 0.3, 1.0, 2.5, 5.0, 13.7, 40.0, 300.0):
            want = 2.0 * stats.t.sf(t, nu)
            if want > 1e-12:
                assert classify.t_two_sided(t, nu) == pytest.approx(want, rel=1e-9, abs=1e-15), (nu, t)
    assert classify.t_two_sided(float("inf"), 5) == 0.0


def _false_drops(n, range_sigma, cross_sigma, alpha, trials, rng, pooled=False):
    """The fraction of clean straight 12 km crossings (n pings 19.9 s apart, the sonar at the origin, errors drawn in range
    and bearing) from which the outlier test drops a point. `pooled`: the test before review (one variance for both
    coordinates: F(2, 2(n-1)-4), Bonferroni over n), written out here to show what it did."""
    t = np.arange(n) * 19.9
    out = 0
    for _ in range(trials):
        p = np.c_[-60.0 + 6.0 * t, np.full(n, 12000.0)]
        r = np.hypot(p[:, 0], p[:, 1]) + range_sigma * rng.standard_normal(n)
        b = np.arctan2(p[:, 0], p[:, 1]) + (cross_sigma / 12000.0) * rng.standard_normal(n)
        xy = np.c_[r * np.sin(b), r * np.cos(b)]
        if pooled:
            tc = t - t.mean()
            res = xy - (xy.mean(axis=0) + tc[:, None] * ((tc[:, None] * xy).sum(axis=0) / (tc**2).sum()))
            r2 = (res**2).sum(axis=1)
            d2 = r2 / (1.0 - (1.0 / n + tc**2 / (tc**2).sum()))
            nu = 2 * (n - 1) - 4
            f = (0.5 * d2) / ((r2.sum() - d2) / nu)
            out += f.max() > 0.5 * nu * ((alpha / n) ** (-2.0 / nu) - 1.0)
        else:
            out += classify._outlier(t, xy, alpha, (0.0, 0.0)) is not None
    return out / trials


@pytest.mark.parametrize("n", [4, 6, 10, 20])
def test_the_outlier_test_drops_about_alpha_of_clean_tracks_whatever_the_error_shape(n):
    """Review of sonar-ljh: measured positions have their own error in range (about 1 m) and across the line of sight
    (R x bearing error, tens of metres at 12 km). On clean straight tracks the per-axis test drops a point from at most
    about alpha of them for errors of one size (1:1), cross-range 10 and 50 times the range error, range 10 times the
    cross-range error, and no cross-range error at all, within 4 Monte Carlo sigmas (2000 tracks each, alpha 0.05). The
    pooled test it replaced drops 18 to 31 % at 10:1 and 1:10 (asserted for 1:10)."""
    rng = np.random.default_rng(100 + n)
    alpha, trials = 0.05, 2000
    tol = 4.0 * np.sqrt(alpha * (1 - alpha) / trials)
    for sr, sc in ((1.0, 1.0), (1.0, 10.0), (1.0, 50.0), (10.0, 1.0), (1.0, 0.0)):
        rate = _false_drops(n, sr, sc, alpha, trials, rng)
        assert rate < alpha + tol and rate > 0.5 * alpha, (sr, sc, rate)
    assert _false_drops(n, 1.0, 10.0, alpha, trials, rng, pooled=True) > alpha + tol


def test_outlier_alpha_and_the_fit_limits_growth_are_checked():
    tr = glinting_12km_track()
    for bad in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError, match="outlier_alpha"):
            classify.classify_tracks([tr], outlier_alpha=bad)
    for bad in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="max_fit_rms_bearing"):
            classify.classify_tracks([tr], max_fit_rms=60.0, max_fit_rms_bearing=bad)


# --- sonar-4jm: the fit limit grows with range where asked ---------------------------------------------------------

POPOTO_FIT_RMS_BEARING = 4.0 * (0.2993648934 / 4.8 / 16.0) / np.sqrt(2.0)   # rad: libs/live derives it for popoto_long_range


def crossing(rng, bearing_sigma, rng_m=20000.0, n=6):
    """A ship crossing at 6 m/s at `rng_m`, n pings 19.9 s apart, measured with Gaussian bearing error (rad) and 1 m of
    range error; (across, forward) frame, the sonar at the origin."""
    t = 19.9 * np.arange(n)
    fwd = 800.0 - 6.0 * t
    b = np.arctan2(fwd, rng_m) + bearing_sigma * rng.standard_normal(n)
    r = np.hypot(fwd, rng_m) + rng.standard_normal(n)
    m = np.c_[r * np.cos(b), r * np.sin(b)]
    tr = Track(1, np.array([m[-1, 0], m[-1, 1], 0.0, -6.0]), np.eye(4), hits=n, confirmed=True)
    tr.history = [(float(a), float(p[0]), float(p[1])) for a, p in zip(t, m)]
    tr.measured = [tuple(map(float, p)) for p in m]
    tr.snr = [20.0] * n
    return tr


def test_the_fit_limit_grows_with_range_where_asked():
    """sonar-4jm: the measured positions scatter across the line of sight by R x the bearing error, which a fit limit
    from range resolution alone (60 m for the Popoto array) does not follow. At 20 km with 0.2 deg of bearing error
    (70 m across) the fixed limit makes uncertain 6 % of ships that are movers without any limit; with the limit grown
    by libs/live's derived 0.63 deg per metre of range (132 m at 12 km, 221 m at 20 km) it binds on none of them. At
    12 km and 0.05 to 0.2 deg (the scatter measured: 0.03-0.05 deg for a point, 0.1-0.3 deg for the ship side-on) the fixed
    limit binds on at most 1 % (seeded draws, 400 ships each)."""
    kw = dict(min_speed=1.0, static_speed=1.0)
    def movers(deg, rng_m, **more):
        rng = np.random.default_rng(1)
        return sum(classify.classify_tracks([crossing(rng, np.radians(deg), rng_m)], **kw, **more)[0].kind == "mover"
                   for _ in range(400))
    fixed, grown, none = (movers(0.2, 20000.0, max_fit_rms=60.0), movers(0.2, 20000.0, max_fit_rms=60.0, max_fit_rms_bearing=POPOTO_FIT_RMS_BEARING),
                          movers(0.2, 20000.0, max_fit_rms=None))
    assert grown == none and none - fixed >= 16
    for deg in (0.05, 0.1, 0.2):
        assert movers(deg, 12000.0, max_fit_rms=None) - movers(deg, 12000.0, max_fit_rms=60.0) <= 4


def test_a_grown_fit_limit_still_refuses_positions_that_jump_about():
    """sonar-4jm: the grown limit is R x 0.63 deg (221 m at 20 km); ten pings whose positions jump 400 m in range about
    the line (RMS 316 m) are still uncertain, while 80 m jumps (RMS 63 m), uncertain under the fixed 60 m, are a mover
    under the grown limit. classify_tracks's default (0) leaves the limit as max_fit_rms gives it."""
    jump = np.array([(400.0, 0.0), (-400.0, 0.0)] * 5)
    def jumping(scale):
        tr = crossing(np.random.default_rng(3), 0.0, n=10)
        tr.measured = [tuple(np.asarray(m) + scale * j) for m, j in zip(tr.measured, jump)]
        return tr
    (c,) = classify.classify_tracks([jumping(1.0)], max_fit_rms=60.0, max_fit_rms_bearing=POPOTO_FIT_RMS_BEARING)
    assert c.fit_rms > 300.0 and c.kind == "uncertain"
    near = jumping(0.2)
    (fixed,) = classify.classify_tracks([near], max_fit_rms=60.0)
    assert 60.0 < fixed.fit_rms < 70.0 and fixed.kind == "uncertain"
    assert [c.kind for c in classify.classify_tracks([near], max_fit_rms=60.0, max_fit_rms_bearing=POPOTO_FIT_RMS_BEARING,
                                                      origin=(0.0, 0.0))] == ["mover"]
    # The range is from `origin`: the sonar 19 km nearer the ship, the limit is 60 m again.
    assert [c.kind for c in classify.classify_tracks([near], max_fit_rms=60.0, max_fit_rms_bearing=POPOTO_FIT_RMS_BEARING,
                                                      origin=(19000.0, 0.0))] == ["uncertain"]
