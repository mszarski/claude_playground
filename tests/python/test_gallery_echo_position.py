"""tools/gallery.py echo_position: an echo is placed from where the array is when it arrives (sonar-960).

The simulator receives with a moving array where it is at the echo's arrival; the gallery placed every Popoto
detection from the array at transmit, 24 m behind along the track at 12 km and 1.5 m/s."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gallery():
    spec = importlib.util.spec_from_file_location("gallery_under_test", ROOT / "tools" / "gallery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def two_way(q, p0, v, c):
    """The exact two-way time of a monostatic echo from point q: transmit at p0, received by the array moving at v."""
    t = 2.0 * np.linalg.norm(q - p0) / c
    for _ in range(50):
        t = (np.linalg.norm(q - p0) + np.linalg.norm(q - (p0 + v * t))) / c
    return t


@pytest.mark.parametrize("q", [(800.0, 12000.0), (-2500.0, 11700.0), (40.0, 300.0)])
def test_the_echo_is_placed_where_the_target_is_from_a_moving_array(gallery, q):
    c, v, p0 = 1500.0, np.array([1.5, 0.0]), np.array([10.0, -3.0])
    q = np.asarray(q)
    t = two_way(q, p0, v, c)
    p_rx = p0 + v * t
    u = (q - p_rx) / np.linalg.norm(q - p_rx)
    rng, bearing = 0.5 * c * t, float(np.arctan2(u[0], u[1]))
    got = np.array(gallery.echo_position(p0, (1.5, 0.0, 0.0), rng, bearing, c))
    assert np.linalg.norm(got - q) < 0.06                        # second order: |d|^2 / rng
    old = p0 + rng * np.array([np.sin(bearing), np.cos(bearing)])  # from the array at transmit, as before
    d = v * t                                                     # the array's travel over the round trip
    assert np.allclose(old - q, -d + 0.5 * (u @ d) * u, atol=0.06)
    if q[1] > 10000.0 and abs(q[0]) < 1000.0:
        assert old[0] - q[0] == pytest.approx(-24.0, abs=0.2)     # 24 m behind at 12 km near broadside
    still = gallery.echo_position(p0, (0.0, 0.0, 0.0), rng, bearing, c)
    assert np.allclose(still, old, atol=1e-9)                     # an array at rest: the old placement exactly


def test_the_simulated_echo_of_a_fast_array_is_placed_on_its_target(gallery):
    """Through the simulator and the gallery's beamformer: a point 300 m out, the array making 25 m/s along itself
    (10 m over the round trip), a 10 ms pulse (the motion during a long pulse is left out: echo_position). The beam's
    peak bearing (parabolic on a 0.1 degree grid), placed by echo_position, lies on the target; placed from the array at
    transmit it lay about 10 m behind."""
    from gpusonar import flank
    from gpusonar.sim import Scene, make_pose, simulate_ping
    from gpusonar.sonar import SonarModel
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [100.0, 400.0]}},
                                         "transmit": {"waveform": {"type": "lfm", "f_start": 6000.0, "f_stop": 10000.0,
                                                                   "duration": 0.01, "window_alpha": 0.05,
                                                                   "source_level_db_rms": 200.0}}})
    c, v = 1500.0, np.array([25.0, 0.0, 0.0])
    q = np.array([30.0, 300.0, 0.0])
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0), velocity=tuple(v)), Scene().add_points([q], target_strength_db=10.0),
                        environment={"sound_speed": c}, sequence=0, time_ns=0, noise_rms=1e-6, rng=np.random.default_rng(1))
    origin = rec.rx_positions().mean(axis=0)
    bearings = np.radians(np.arange(-20.0, 20.001, 0.1))
    bm = flank.farfield_beamform(rec, bearings, window="hann").valid()
    power = np.abs(bm.data) ** 2
    ib, ir = np.unravel_index(np.argmax(power), power.shape)
    a, b, d = np.log(power[ib - 1, ir]), np.log(power[ib, ir]), np.log(power[ib + 1, ir])
    bearing = bearings[ib] + 0.5 * (a - d) / (a - 2.0 * b + d) * (bearings[1] - bearings[0])
    rng = float(bm.ranges[ir])
    got = np.array(gallery.echo_position(origin, rec.geometry.rx_pose.velocity, rng, bearing, rec.environment.sound_speed))
    assert abs(got[0] - q[0]) < 0.5 and abs(got[1] - q[1]) < 0.5, (got, q)
    old = origin[:2] + rng * np.array([np.sin(bearing), np.cos(bearing)])
    assert old[0] - q[0] < -8.0                                       # from the array at transmit: about 10 m behind


def test_the_leading_edge_of_a_merged_multipath_echo_is_its_direct_arrival(gallery):
    """sonar-ca3 / sonar-a98: in the Popoto and flank scenes one object's echo is several arrivals within tens of
    metres, the later ones (surface and bottom paths) the stronger, merged into one detection by detect.cluster. Its
    centroid lies beyond the object; `set_range_estimate("first")` (gallery.RANGE_ESTIMATE, the default since the
    owner's choice on those issues) gives the nearest cell, the direct arrival. A map of 2 m cells: the direct arrival at 6000 m, 6 dB weaker than one at +34 m, the
    cells between filled by the arrivals in between (above the threshold); background 1."""
    from gpusonar import detect
    ranges = np.arange(5000.0, 7000.0, 2.0)
    bearings = np.radians(np.arange(-10.0, 10.01, 0.5))
    power = np.ones((len(bearings), len(ranges)))
    ib = len(bearings) // 2
    on = (ranges >= 6000.0) & (ranges <= 6040.0)
    power[ib, on] = 1e3
    power[ib, ranges == 6000.0] = 1e4
    power[ib, ranges == 6034.0] = 4e4
    flags = power > 100.0
    hits = detect.cluster(flags, power, ranges, bearings, guard=50, train=150)
    assert len(hits) == 1 and hits[0].range > 6015.0               # the centroid: well beyond the direct arrival
    assert gallery.set_range_estimate(hits, "centroid")[0].range > 6015.0
    assert gallery.set_range_estimate(hits, "first")[0].range == pytest.approx(6000.0)
    with pytest.raises(ValueError):
        gallery.set_range_estimate(hits, "peak")
    assert gallery.RANGE_ESTIMATE == "first"                        # the default (was "centroid")
    assert gallery.set_range_estimate(detect.cluster(flags, power, ranges, bearings, guard=50, train=150))[0].range == pytest.approx(6000.0)
