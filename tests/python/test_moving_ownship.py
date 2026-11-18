"""Moving ownship over a seabed and under a surface that are fixed in the world; survey
legs; hull and cable objects; noise in baseband records; the synthesis beam floor."""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import acoustics
from gpusonar.config import load_scenario_config
from gpusonar.sim import FreeFieldProvider, Scene, build_scene, make_pose, reverb, run_scenario, simulate_ping
from gpusonar.sim.noise import wenz_spectrum_db
from gpusonar.sonar import SonarModel

C = 1500.0
core = gs.core


def test_reverberation_carries_ownship_doppler_by_bearing():
    """A line array moving along its own axis over a fixed seabed: reverberation arriving
    from ahead is shifted up by 2 v f / c, from astern down by the same."""
    speed, fc = 6.0, 8000.0
    sonar = SonarModel.load("flank_32", {
        "receive": {"kind": "baseband", "sample_rate": 4000.0, "window": {"range": [150.0, 900.0]}, "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": fc, "f_stop": fc, "duration": 0.2,
                                                          "window_alpha": 0.0, "source_level_db_rms": None, "amplitude": 1.0}}})
    centroid = {}
    for v in (0.0, speed):
        rec = simulate_ping(sonar, make_pose(position=(0, 0, -30.0), velocity=(v, 0, 0)), Scene(),
                            environment={"sound_speed": C}, adc=False, absorption_model="none",
                            reverberation=[{"boundary_z": 0.0, "mu_db": -27.0}], rng=np.random.default_rng(3))
        x = rec.samples.array
        pos = rec.rx_positions()
        along = (pos - pos.mean(axis=0))[:, 0]
        f = np.fft.fftfreq(x.shape[1], 1.0 / rec.samples.sample_rate)
        for name, sin_b in (("ahead", np.sin(np.radians(70.0))), ("astern", -np.sin(np.radians(70.0)))):
            beam = (x * np.exp(-2j * np.pi * fc / C * along * sin_b)[:, None]).sum(axis=0)
            p = np.abs(np.fft.fft(beam)) ** 2
            centroid[(v, name)] = float((f * p).sum() / p.sum())
    ceiling = 2 * speed * fc / C                                            # 64 Hz, dead ahead
    assert abs(centroid[(0.0, "ahead")]) < 6.0 and abs(centroid[(0.0, "astern")]) < 6.0
    # A 3 m array's beam towards 70 degrees takes in a wide cone, so the mean shift is
    # somewhat below the ceiling.
    assert 0.6 * ceiling < centroid[(speed, "ahead")] < ceiling
    assert centroid[(speed, "astern")] == pytest.approx(-centroid[(speed, "ahead")], rel=0.15)


def test_beam_patterns_shape_the_reverberation():
    """A forward-looking sonar hears no reverberation from astern, and a narrow beam tilted
    up hears far less of the seabed than a wide level one."""
    provider = FreeFieldProvider(C)
    times = np.arange(0.05, 0.4, 0.002)
    azimuths = np.radians([0.0, 45.0, 90.0, 180.0])
    source = np.array([0.0, 0.0, 20.0])
    seabed = {}
    for name in ("fls_120k", "fls_330k"):
        rec = SonarModel.load(name).make_record(make_pose(position=source))
        rot = np.asarray(core.rotation_ned(*rec.geometry.rx_pose.rpy))
        tx = reverb.element_power_pattern(rec.geometry.tx, rot)
        rx = reverb.element_power_pattern(rec.geometry.rx[0], rot)
        seabed[name] = reverb.reverberation_power(provider, source, 60.0, times, -27.0, pattern=(azimuths, tx, rx))
        assert seabed[name].shape == (4, len(times))
        assert seabed[name][3].sum() == 0.0                                 # nothing from astern
        assert seabed[name][0].sum() > seabed[name][1].sum() > seabed[name][2].sum() >= 0.0
    ratio_db = 10 * np.log10(seabed["fls_120k"][0].sum() / seabed["fls_330k"][0].sum())
    assert ratio_db > 20.0
    # Omnidirectional hydrophones have no pattern.
    rec = SonarModel.load("popoto_8").make_record(make_pose())
    assert reverb.element_power_pattern(rec.geometry.rx[0], np.eye(3)) is None


def test_ambient_noise_in_a_baseband_record_has_the_stated_level():
    sonar = SonarModel.load("fls_120k", {"receive": {"kind": "baseband", "sample_rate": 40e3, "window": {"range": [20.0, 60.0]}}})
    spectrum = lambda f: wenz_spectrum_db(f, 5.0, 0.5)                      # noqa: E731
    rec = simulate_ping(sonar, make_pose(position=(0, 0, 20.0)), Scene(), adc=False, noise_spectrum=spectrum,
                        rng=np.random.default_rng(0))
    x = rec.samples.array
    level = float(wenz_spectrum_db(np.array([120e3]), 5.0, 0.5)[0])
    expected = np.sqrt(2 * 10 ** (level / 10) * 40e3) * 10 ** ((-190.0 + 20.0) / 20)
    assert np.sqrt(np.mean(np.abs(x) ** 2)) == pytest.approx(expected, rel=0.02)
    assert abs(np.mean(x[0] * np.conj(x[1]))) < 0.05 * expected**2          # independent between channels


def test_survey_legs_look_at_one_patch_from_both_sides():
    cfg = load_scenario_config("sas_minsas_survey")
    # One marker abeam of each leg's first pings (the transmit beam is 5.4 degrees wide).
    cfg["scene"] = [{"type": "points", "label": "marker", "positions": [[0.4, 50.0, 0.0], [39.6, 50.0, 0.0]]}]
    for leg in cfg["platform"]["legs"]:
        leg["pings"] = 3
    result = run_scenario(cfg)
    assert result.legs == [0, 0, 0, 1, 1, 1]
    yaw = [r.geometry.rx_pose.rpy[2] for r in result.records]
    assert yaw[:3] == [0.0] * 3 and yaw[3:] == pytest.approx([np.pi] * 3)
    step = np.diff(result.true_positions, axis=0)
    assert np.allclose(step[0], [0.544, 0.0, 0.0]) and np.allclose(step[4], [-0.544, 0.0, 0.0])   # half the array per ping
    assert result.times[3] - result.times[2] > 20.0                         # the turn takes time
    # Starboard of each leg points at the marker: the boresight's y component flips.
    looks = [np.asarray(core.rotation_ned(*r.geometry.rx_pose.rpy)) @ np.asarray(r.geometry.rx[0].boresight) for r in result.records]
    assert looks[0][1] > 0.9 and looks[5][1] < -0.9
    assert all(np.abs(r.samples.array).max() > 0 for r in result.records)   # both see it
    # The array's elements run along the direction of travel on both legs.
    p = result.records[5].rx_positions()
    assert p[-1, 0] - p[0, 0] < -1.0


def test_hull_object():
    item = {"type": "hull", "label": "boat", "origin": [100.0, 0.0, 0.0], "length": 12.0, "beam": 3.5, "depth": 1.2,
            "heading_deg": 90.0, "count": 6000, "mu_db": -8.0}
    scene = build_scene([item], np.random.default_rng(1))
    p, n = scene.positions, scene.normals
    assert p[:, 1].min() < -5.5 and p[:, 1].max() > 5.5                     # 12 m along y
    assert np.abs(p[:, 0] - 100.0).max() <= 1.75 + 1e-9 and 0.0 <= p[:, 2].min() and p[:, 2].max() <= 1.2 + 1e-9
    assert np.all(((p - np.array([100.0, 0.0, 0.0])) * n).sum(axis=1) > -1e-6)   # normals point out of the hull
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    # Summed cross-section is mu times the wetted area (about 45 square metres here).
    total = np.sum(np.abs(scene.amplitudes) ** 2)
    assert 10 * np.log10(total) == pytest.approx(-8.0 + 10 * np.log10(45.0), abs=1.5)
    # A wreck may cast a shadow; a moving hull may not ask for one.
    wreck = build_scene([{**item, "occlude": True}], np.random.default_rng(1))
    assert wreck.occluders[0][0] == "capsule"
    with pytest.raises(ValueError, match="do not move"):
        build_scene([{**item, "occlude": True, "velocity": [1.0, 0.0, 0.0]}], np.random.default_rng(1))


def test_cable_follows_its_waypoints():
    way = [[0.0, 0.0, -0.04], [5.0, 1.0, -0.04], [10.0, 0.0, -0.04]]
    scene = build_scene([{"type": "cable", "label": "cable", "radius": 0.04, "density": 2000.0, "waypoints": way}],
                        np.random.default_rng(2))
    assert len(scene.occluders) == 2 and set(scene.labels) == {"cable"}
    w = np.asarray(way)
    def to_segment(p, a, b):
        t = np.clip(((p - a) @ (b - a)) / ((b - a) @ (b - a)), 0.0, 1.0)
        return np.linalg.norm(p - a - t[:, None] * (b - a), axis=1)

    per_segment = np.stack([to_segment(scene.positions, a, b) for a, b in zip(w[:-1], w[1:])])
    d = per_segment.max(axis=0)
    d = np.where(np.isclose(per_segment, 0.04, atol=1e-6).any(axis=0), 0.04, d)
    assert np.allclose(d, 0.04, atol=1e-6)                                  # every point on one of the two segments


def test_beam_floor_changes_little():
    sonar = SonarModel.load("sas_minsas120", {"receive": {"kind": "baseband", "sample_rate": 60e3, "window": {"range": [45.0, 60.0]}},
                                              "transmit": {"waveform": {"duration": 1e-3}}})
    rng = np.random.default_rng(5)
    pts = np.column_stack([rng.uniform(-25, 25, 400), rng.uniform(48, 56, 400), np.zeros(400)])
    scene = Scene().add_points(pts, rng.standard_normal(400) + 1j * rng.standard_normal(400))
    pose = make_pose(position=(0, 0, -12.0))
    full = simulate_ping(sonar, pose, scene, adc=False, absorption_model="none").samples.array
    cut = simulate_ping(sonar, pose, scene, adc=False, absorption_model="none", beam_floor_db=-40.0).samples.array
    assert np.sqrt(np.mean(np.abs(full - cut) ** 2)) < 0.03 * np.sqrt(np.mean(np.abs(full) ** 2))
    assert not np.array_equal(full, cut)                                    # some paths were skipped
    assert np.abs(full).max() > 0
