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
