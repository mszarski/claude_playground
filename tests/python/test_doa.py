"""Bearings by sparse Bayesian learning: active range gates and passive frames, against
scenes whose bearings are known, and against the conventional beamformer on the same data."""

import json
import time

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
GRID = np.radians(np.arange(-40.0, 40.01, 0.25))
R = 3000.0


def popoto(extra=None):
    over = {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [R - 150.0, R + 150.0]}},
            "transmit": {"waveform": {"duration": 0.2}}}
    if extra:
        over = gs.config.deep_merge(over, extra)
    return SonarModel.load("popoto_long_range", over)


def active_ping(bearings_deg, strengths_db=None, noise=3e-9, seed=0, ranges=None):
    sonar = popoto()
    strengths_db = strengths_db or [10.0] * len(bearings_deg)
    ranges = ranges or [R] * len(bearings_deg)
    scene = Scene()
    for b, ts, r in zip(bearings_deg, strengths_db, ranges):
        scene.add_points([[r * np.sin(np.radians(b)), r * np.cos(np.radians(b)), 0.0]], target_strength_db=ts)
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C}, absorption_model="none", adc=False,
                        noise_rms=noise, rng=np.random.default_rng(seed))
    recon.prepare([rec])
    return rec


def peaks_deg(result):
    return sorted(np.degrees(p[0]) for p in result["peaks"])


def test_beamwidth_of_the_array_is_about_three_degrees():
    rec = active_ping([0.0])
    u = np.array(core.element_offsets(rec))
    assert len(u) == 32 and np.allclose(np.diff(u), 0.15) and u.mean() == pytest.approx(0.0, abs=1e-9)
    assert np.degrees(0.886 * (C / 5000.0) / (32 * 0.15)) == pytest.approx(3.2, abs=0.1)


def test_one_target_is_found_where_it_is_by_both_methods():
    rec = active_ping([7.3])
    for method in ("sbl", "conventional"):
        res = core.doa_gate(rec, 2 * R / C, 128, list(GRID), method=method, max_sources=1)
        assert len(res["peaks"]) == 1 and np.degrees(res["peaks"][0][0]) == pytest.approx(7.3, abs=0.2), method
    res = core.doa_gate(rec, 2 * R / C, 128, list(GRID))
    assert res["peaks"][0][2] > 20.0 and res["iterations"] < 60        # well above the noise, and converged


@pytest.mark.parametrize("pair", [(2.0, 4.0), (-11.0, -9.2), (20.0, 22.5)])
def test_two_echoes_closer_than_a_beamwidth_at_one_range(pair):
    """The same pulse back from two bearings at the same range: coherent, 1.8 to 2.5 degrees
    apart under a 3.2 degree beam (wider off broadside)."""
    rec = active_ping(list(pair))
    sbl = core.doa_gate(rec, 2 * R / C, 128, list(GRID), method="sbl", dynamic_range_db=15.0)
    got = peaks_deg(sbl)
    assert len(got) == 2 and got[0] == pytest.approx(pair[0], abs=0.4) and got[1] == pytest.approx(pair[1], abs=0.4)
    # The conventional beamformer on the same snapshots shows one lobe between them.
    spectrum = core.doa_gate(rec, 2 * R / C, 128, list(GRID), method="conventional")["spectrum"]
    inside = (np.degrees(GRID) > pair[0] - 1.0) & (np.degrees(GRID) < pair[1] + 1.0)
    s = spectrum[inside]
    interior_maxima = int(np.sum((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])))
    assert interior_maxima == 1


def test_a_weak_echo_beside_a_strong_one():
    rec = active_ping([0.0, 6.0], strengths_db=[10.0, -5.0])       # 15 dB down, two beamwidths away
    got = peaks_deg(core.doa_gate(rec, 2 * R / C, 128, list(GRID), dynamic_range_db=25.0, min_snr_db=6.0))
    assert len(got) == 2 and got[0] == pytest.approx(0.0, abs=0.3) and got[1] == pytest.approx(6.0, abs=0.5)
    # The conventional spectrum's first sidelobe (-13 dB) is about as strong as that echo; SBL has none to report.
    alone = core.doa_gate(active_ping([0.0]), 2 * R / C, 128, list(GRID), dynamic_range_db=25.0, min_snr_db=6.0)
    assert len(alone["peaks"]) == 1


def test_noise_alone_gives_nothing_above_the_floor():
    sonar = popoto()
    rec = simulate_ping(sonar, make_pose(), Scene(), environment={"sound_speed": C}, adc=False, noise_rms=3e-9,
                        rng=np.random.default_rng(4))
    recon.prepare([rec])
    res = core.doa_gate(rec, 2 * R / C, 128, list(GRID), min_snr_db=12.0)
    assert res["peaks"] == []


# --- passive ---------------------------------------------------------------------------------

def passive_record(sources, seconds=4.0, noise=1.0, seed=1):
    """Broadband noise sources as plane waves across the array, in a baseband record of the
    Popoto array: [(bearing_deg, rms amplitude), ...] plus independent sensor noise."""
    sonar = popoto({"receive": {"window": {"range": [0.0, 0.5 * C * seconds]}}})
    rec = sonar.make_record(make_pose(), environment={"sound_speed": C})
    s = rec.samples
    n, fs, fc = s.n_samples, s.sample_rate, s.center_frequency
    u = np.array(core.element_offsets(rec))
    f = np.fft.fftfreq(n, 1.0 / fs)
    rng = np.random.default_rng(seed)
    x = noise * (rng.standard_normal((len(u), n)) + 1j * rng.standard_normal((len(u), n))) / np.sqrt(2)
    for bearing, rms in sources:
        src = np.fft.fft(rms * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2))
        # A wave from that bearing reaches an element at offset u earlier by u sin(b) / c.
        lead = u[:, None] * np.sin(np.radians(bearing)) / C
        x = x + np.fft.ifft(src[None, :] * np.exp(2j * np.pi * (fc + f)[None, :] * lead), axis=1)
    kind = s.kind
    s.array = x.astype(np.complex64)
    s.kind = kind
    rec.samples = s
    return rec


def test_passive_bearing_time_record_separates_two_noise_sources():
    rec = passive_record([(-20.0, 1.0), (-17.2, 0.7)])              # 2.8 degrees apart, the second 3 dB down
    frames, grid = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, first_deg=-60.0, last_deg=60.0,
                                    step_deg=0.25, dynamic_range_db=12.0, min_snr_db=3.0)
    assert len(frames) >= 15 and len(grid) == 481
    both = 0
    for fr in frames:
        got = sorted(np.degrees(p[0]) for p in fr["peaks"])
        if len(got) == 2 and abs(got[0] + 20.0) < 0.6 and abs(got[1] + 17.2) < 0.6:
            both += 1
    assert both >= 0.8 * len(frames)
    assert np.all(np.diff([fr["time"] for fr in frames]) > 0)
    # Conventional beam power on the same frames: one lobe over the pair.
    conv, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, first_deg=-60.0, last_deg=60.0,
                               step_deg=0.25, method="conventional")
    deg = np.degrees(np.array(grid))
    s = np.mean([fr["spectrum"] for fr in conv], axis=0)[(deg > -21.5) & (deg < -15.5)]
    assert int(np.sum((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))) == 1


def test_passive_source_below_the_element_noise_is_still_found():
    rec = passive_record([(12.0, 0.3)])                             # -10 dB at each hydrophone; the array and time add up
    frames, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, max_sources=1, min_snr_db=0.0)
    found = [np.degrees(fr["peaks"][0][0]) for fr in frames if fr["peaks"]]
    assert len(found) >= 0.9 * len(frames) and np.median(found) == pytest.approx(12.0, abs=0.3)
    powers = [fr["peaks"][0][1] for fr in frames if fr["peaks"]]
    # Its power comes back as what was put in: 0.3^2 at each hydrophone, in the units of a Hann-weighted
    # 256-point transform (times 256 x 0.375); and the noise as 1.0 in the same units.
    assert np.median(powers) / (256 * 0.375) == pytest.approx(0.09, rel=0.2)
    assert np.median([fr["noise"] for fr in frames]) / (256 * 0.375) == pytest.approx(1.0, rel=0.1)


def test_passive_frames_of_noise_alone_are_empty():
    frames, _ = core.doa_passive(passive_record([]), 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, min_snr_db=6.0)
    assert sum(len(fr["peaks"]) for fr in frames) <= 0.1 * len(frames)


# --- in the chain ----------------------------------------------------------------------------

CHAIN = {"bearing_first_deg": -30.0, "bearing_last_deg": 30.0, "bearing_step_deg": 0.5, "range_cell_m": 2.0, "cfar_pfa": 1e-6,
         "cfar_guard_m": 20.0, "cfar_train_m": 60.0, "min_snr_db": 12.0, "multipath": False, "sbl_min_snr_db": 10.0,
         "sbl_dynamic_range_db": 15.0}


def test_chain_with_sbl_reports_both_bearings_where_the_map_reports_one():
    rec = active_ping([2.0, 4.0])
    by_map = json.loads(core.LiveChainCpp(dict(CHAIN, doa="map")).process(rec.copy()))
    by_sbl = json.loads(core.LiveChainCpp(dict(CHAIN, doa="sbl")).process(rec.copy()))
    near = lambda rep: sorted(d["bearing_deg"] for d in rep["detections"] if abs(d["range_m"] - R) < 10.0)   # noqa: E731
    assert len(near(by_map)) == 1 and 2.0 < near(by_map)[0] < 4.0
    got = near(by_sbl)
    assert len(got) == 2 and got[0] == pytest.approx(2.0, abs=0.4) and got[1] == pytest.approx(4.0, abs=0.4)


def test_sbl_gate_is_fast_enough_for_a_few_gates_per_ping():
    rec = active_ping([2.0, 4.0])
    core.doa_gate(rec, 2 * R / C, 128, list(GRID))
    t = time.perf_counter()
    for _ in range(5):
        core.doa_gate(rec, 2 * R / C, 128, list(GRID))
    per_gate = (time.perf_counter() - t) / 5
    print(f"[sbl] 32 channels, 321 bearings, one range gate: {per_gate * 1e3:.0f} ms")
    assert per_gate < 2.0
