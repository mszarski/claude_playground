"""The two Doppler paths side by side, both engines: what the plain sweep path (the default chain, every Doppler option
off) states about itself, and what the tone path (`doppler.tone: bank`) refuses (docs/design/doppler_tradeoffs.md,
the coverage table in docs/tasks/DOPPLER.md "Grating-lobe notch"). Cheap gaps of that table closed here; the rest of
each path is held by test_doppler_tone.py, test_doppler_tone_lobes.py, test_doppler_bank.py, test_doppler_cfar.py,
test_doppler_device.py and test_live_cpp.py.

Held here:
- the default chain's report line carries no Doppler field (range_rate_m_s, range_rate_step_m_s, doppler_path), on a
  sweep ping and on a train ping, both engines: the report line is a contract (DOPPLER.md section 5, question 6);
- the default chain on a train compresses with the WHOLE train as its replica (D-237: on popoto_12km_doppler the ship on
  0 of 6 pings): its snr on a still target stays at least 3 dB under the tone path's sweep-only replica, so the limit
  cannot change silently, and the record's replica is not touched;
- the default chain gives a mover no range rate (the plain sweep has none to give);
- the tone path is refused with `elevation` in the C++ chain itself (the Python check is test_doppler_tone.py's for
  range_doppler; elevation had none) and in the Python chain, and with `range_doppler` in the C++ chain itself.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
T = 0.5
LFM = {"type": "lfm", "f_start": 6.0e+3, "f_stop": 10.0e+3, "duration": T, "window_alpha": 0.05}
TRAIN = {"type": "sampled", "source": "train", "f_start": 6.0e+3, "f_stop": 10.6e+3,
         "segments": [{"kind": "cw", "frequency": 10.5e+3, "duration": T, "window_alpha": 1.0},
                      {"kind": "gap", "duration": 0.0},
                      {"kind": "lfm", "f_start": 6.0e+3, "f_stop": 10.0e+3, "duration": T, "window_alpha": 0.05}]}
TONE_TRAIN = {"tone": "bank", "report": True, "tone_frequency_hz": 10.5e+3, "tone_start_s": 0.0, "tone_duration_s": T, "tone_window_alpha": 1.0}
CHAIN = {"bearings_deg": [-50.0, 50.0, 1.0], "backend": "host"}
DOPPLER_FIELDS = ("range_rate_m_s", "range_rate_step_m_s", "doppler_path")
STILL = (400.0 * np.sin(np.radians(-20.0)), 400.0 * np.cos(np.radians(-20.0)), 0.0)


def ping(wave):
    """flank_32 (test_doppler_tone.py's records): a mover opening at 1.5 m/s at 500 m, a still target at 400 m, -20 deg."""
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [100.0, 900.0]}}, "transmit": {"waveform": wave}})
    scene = Scene()
    scene.add_points([[0.0, 500.0, 0.0]], velocity=(0.0, 1.5, 0.0), target_strength_db=-20.0)
    scene.add_points([list(STILL)], velocity=(0.0, 0.0, 0.0), target_strength_db=-20.0)
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)), scene, environment={"sound_speed": C},
                         noise_rms=1e-4, adc=False, absorption_model="none", rng=np.random.default_rng(1))


def near(rep, rng_m, bearing_deg):
    got = [d for d in rep["detections"] if abs(d["range_m"] - rng_m) < 60.0 and abs(d["bearing_deg"] - bearing_deg) < 2.0]
    return max(got, key=lambda d: d["snr_db"]) if got else None


def run(cfg, rec):
    with LiveChain(cfg) as chain:
        return chain.process(rec.copy())


@pytest.mark.parametrize("engine", ["python", "cpp"])
@pytest.mark.parametrize("wave", ["lfm", "train"])
def test_the_default_chain_reports_no_doppler_field_and_gives_a_mover_no_rate(engine, wave):
    rep = run(dict(CHAIN, engine=engine), ping(LFM if wave == "lfm" else TRAIN))
    assert rep["detections"], "nothing detected: the record no longer shows its targets"
    assert near(rep, 500.0, 0.0) is not None
    for d in rep["detections"]:
        assert not any(k in d for k in DOPPLER_FIELDS), d


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_default_chain_compresses_a_train_whole_and_pays_for_it(engine):
    """D-237's limit of the default, stated: with every Doppler option off the train is the replica as sent (the tone's
    half of the energy and its c T / 2 pedestal in the sweep's map); the tone path's sweep part alone gains at least 3 dB
    on the still target (measured when test_doppler_tone.py was written: 24.9 -> 31.1 dB). A change of the default's
    replica would fail here."""
    rec = ping(TRAIN)
    before = np.asarray(rec.transmit.waveform.replica).copy()
    off = run(dict(CHAIN, engine=engine), rec)
    on = run(dict(CHAIN, engine=engine, doppler=TONE_TRAIN), rec)
    assert np.array_equal(np.asarray(rec.transmit.waveform.replica), before)
    a, b = near(off, 400.0, -20.0), near(on, 400.0, -20.0)
    assert a is not None and b is not None
    assert b["snr_db"] > a["snr_db"] + 3.0


def test_the_tone_path_is_refused_with_elevation_and_range_doppler_in_both_engines():
    for engine in ("python", "cpp"):
        with pytest.raises(ValueError, match="elevation"):
            LiveChain(dict(CHAIN, engine=engine, elevation={}, doppler={"tone": "bank"}))
    with pytest.raises(ValueError, match="elevation"):
        core.LiveChainCpp({"doppler_tone": "bank", "elevation": {}})
    with pytest.raises(ValueError, match="range_doppler"):
        core.LiveChainCpp({"doppler_tone": "bank", "range_doppler": "ownship"})
