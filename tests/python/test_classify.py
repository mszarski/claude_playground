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
