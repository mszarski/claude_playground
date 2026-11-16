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


def test_tracker_drops_a_target_that_disappears():
    tracker = detect.Tracker(confirm_hits=2, max_misses=2)
    for k in range(4):
        tracker.update(float(k), [detect.Detection(1000.0 + 2.0 * k, 0.1, 20.0, 3)])
    assert len(tracker.tracks) == 1 and tracker.tracks[0].confirmed
    for k in range(4, 8):
        tracker.update(float(k), [])
    assert tracker.tracks == []
