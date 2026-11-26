"""D3 part B: the tone's Doppler filter bank in the live chain, `chain.doppler: {tone: bank}` (libs/live doppler.hpp
plan_tone, tone_stage, merge_tone; gpusonar/doppler.py), opt-in, both engines. The scene acceptance is in
test_doppler_scenes.py (D3 section); the engine-parity cases of the chain are also in test_live_cpp.py.

What is held here:
- the bank is the matched filter of the tone at each range-rate hypothesis (against a NumPy reference written here, on
  the same beam, at the full sample rate: the bank's decimation is an exact band-limited cut);
- today's chain does not see a CW mover (the baseband keeps the tone's 1/T band only, DOPPLER.md section 0 item 2), the
  tone bank does, with its range rate; a still target is not given a Doppler (the notch: the stationary reverberation
  lies at each beam's own-ship rate and so does every still target; the sweep keeps those);
- a train with the tone above the sweep's band (the arrangement D2 measured to work): the mover is one detection with the
  sweep's range and the tone's rate (`doppler_path` both), the still target the sweep's alone;
- alternating tone-only and sweep-only pings: the tone ping's detections are the tone's, and the tracker takes them;
- the false-alarm budget on noise-only pings;
- the option: off by default and off is the chain as before (test_live_cpp.py holds the bytes), unknown keys refused.
"""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import detect, doppler, live
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
T = 0.5
F_TONE = 8000.0
WINDOW = {"receive": {"window": {"range": [100.0, 900.0]}}}
CW = {"type": "cw", "f_start": F_TONE, "f_stop": F_TONE, "duration": T, "window_alpha": 1.0}
LFM = {"type": "lfm", "f_start": 6.0e+3, "f_stop": 10.0e+3, "duration": T, "window_alpha": 0.05}
TRAIN = {"type": "sampled", "source": "train", "f_start": 6.0e+3, "f_stop": 10.6e+3,
         "segments": [{"kind": "cw", "frequency": 10.5e+3, "duration": T, "window_alpha": 1.0},
                      {"kind": "gap", "duration": 0.0},
                      {"kind": "lfm", "f_start": 6.0e+3, "f_stop": 10.0e+3, "duration": T, "window_alpha": 0.05}]}
CHAIN = {"bearings_deg": [-50.0, 50.0, 1.0], "backend": "host"}
TONE = {"tone": "bank", "report": True}
TONE_TRAIN = dict(TONE, tone_frequency_hz=10.5e+3, tone_start_s=0.0, tone_duration_s=T, tone_window_alpha=1.0)
strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731


def sonar(wave):
    return SonarModel.load("flank_32", dict(WINDOW, transmit={"waveform": wave}))


def ping(wave, targets=(((0.0, 500.0, 0.0), (0.0, 1.0, 0.0)),), ts_db=-20.0, noise_rms=1e-4, seed=1, sequence=0, own=(0.0, 0.0, 0.0)):
    """flank_32 (the array along x, broadside +y) with point targets [(position, velocity)] at the given pose time."""
    scene = Scene()
    for pos, vel in targets:
        scene.add_points([list(pos)], velocity=vel, target_strength_db=ts_db)
    return simulate_ping(sonar(wave), make_pose(position=(0.0, 0.0, 0.0), velocity=own), scene, environment={"sound_speed": C},
                         noise_rms=noise_rms, adc=False, absorption_model="none", rng=np.random.default_rng(seed), sequence=sequence,
                         time_ns=sequence * 2_000_000_000)


def run(cfg, recs):
    with LiveChain(cfg) as chain:
        return [chain.process(r.copy()) for r in recs]


def near(rep, rng_m, bearing_deg, gate_m=60.0, gate_deg=2.0):
    got = [d for d in rep["detections"] if abs(d["range_m"] - rng_m) < gate_m and abs(d["bearing_deg"] - bearing_deg) < gate_deg]
    return max(got, key=lambda d: d["snr_db"]) if got else None


def baseband(rec, allowance=6.0):
    r = rec.copy()
    core.to_baseband(r, max_range_rate=allowance)
    return r


# --- the bank against a reference ---------------------------------------------------------------------------------------

def test_the_bank_is_the_matched_filter_of_the_tone_at_each_rate():
    """One beam (broadside) of a CW ping with a target opening at 1 m/s: the bank's power at every hypothesis and lag
    within 25 dB of its peak is the matched filter of the Hann tone at that rate, sum_n y[n] w((t_n - tau_l) / T)
    exp(-i 2 pi (f (c - r) / (c + r) - fc) t_n), computed here on the beam at the record's full rate, scaled by the
    decimation's fs_d / fs (both normalised by sum w^2: white noise reads its variance at either rate). Within 0.1 dB.
    The beam here is the Hann-weighted channel sum (broadside of a line has no steering delays), not the library's."""
    rec = baseband(ping(CW))
    st = core.live_tone_stage(rec, [0.0], {"doppler_tone": "bank"}, keep_map=True)
    p, pw = st["plan"], st["power"][0]
    # Broadside of a line: no steering delays; the beam is the Hann-weighted sum of the channels (gpusonar.flank's weights).
    from gpusonar import flank
    x = np.asarray(rec.samples.array)
    y = (flank.aperture_weights(rec, "hann")[:, None] * x).sum(axis=0)
    t = rec.samples.start_time + np.arange(x.shape[1]) / rec.samples.sample_rate
    fc = rec.samples.center_frequency
    peak = pw.max()
    checked = 0
    for j in range(pw.shape[0]):
        r = p["rates"][p["first"][0] + j]
        for lag in range(pw.shape[1]):
            if pw[j, lag] < peak * 10 ** -2.5:
                continue
            tau = 2.0 * st["lag_ranges"][lag] / C            # the lag's start after the transmit (the tone starts the pulse)
            u = (t - tau) / T
            inside = (u >= 0.0) & (u < 1.0)
            w = np.where(inside, 0.5 - 0.5 * np.cos(2.0 * np.pi * np.clip(u, 0.0, 1.0)), 0.0)
            nu = F_TONE * (C - r) / (C + r) - fc
            xx = np.sum(y * w * np.exp(-2j * np.pi * nu * t))
            ref = abs(xx) ** 2 / np.sum(w ** 2) * st["fs"] / rec.samples.sample_rate
            assert 10 * np.log10(pw[j, lag] / ref) == pytest.approx(0.0, abs=0.1), (r, lag)
            checked += 1
    assert checked >= 10
    j, lag = np.unravel_index(np.argmax(pw), pw.shape)
    assert p["rates"][p["first"][0] + j] == pytest.approx(1.0, abs=p["step"])


def test_the_plan_spaces_the_hypotheses_at_half_the_resolution_and_derives_the_notch_and_the_allowance():
    rec = ping(CW, targets=())
    bearings = np.radians(np.arange(-50.0, 50.5, 1.0))
    p = core.live_tone_plan(rec, list(bearings), doppler.tone_options(dict(TONE, max_range_rate_m_s=3.0)))
    st = core.live_tone_stage(baseband(rec), list(bearings), doppler.tone_options(dict(TONE, max_range_rate_m_s=3.0)))
    q = st["plan"]
    assert p["on"] and p["tone_only"] and q["tone_only"]
    res = C / (2.0 * F_TONE * T)
    assert q["resolution"] == pytest.approx(res) and q["step"] == pytest.approx(0.5 * res)
    assert q["per_beam"] == 2 * int(np.ceil(3.0 / q["step"] - 1e-9)) + 1
    assert q["guard"] == 4 and q["train"] == 32
    # A still platform: every beam's own-ship rate 0, the notch three resolution cells.
    assert np.allclose(q["ownship"], 0.0) and np.allclose(q["notch"], 3.0 * res)
    # The baseband must keep 2 x 3 m/s x 8 kHz / c + 8 / T = 48 Hz either side of the tone in its untapered part.
    assert p["baseband_max_range_rate"] > 0.0
    assert q["gate_range"] == pytest.approx(0.25 * C * T) and q["range_sigma"] == pytest.approx(C * T / 8.0)
    # Under way along the array at 1.5 m/s: each beam centred on -v . u_b, the notch widened by the beam's own-motion span.
    moving = ping(CW, targets=(), own=(1.5, 0.0, 0.0))
    m = core.live_tone_stage(baseband(moving), list(bearings), doppler.tone_options(dict(TONE, max_range_rate_m_s=3.0)))["plan"]
    assert m["ownship"] == pytest.approx(list(-1.5 * np.sin(bearings)), abs=1e-6)
    assert all(n > 3.0 * res for n, b in zip(m["notch"], bearings) if abs(b) < np.radians(80.0))
    # A sweep-only ping (the other half of an alternating pair): no tone path.
    assert not core.live_tone_plan(ping(LFM, targets=()), list(bearings), doppler.tone_options(TONE))["on"]


# --- what it detects ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_cw_mover_is_missed_today_and_found_by_the_tone_bank_with_its_range_rate(engine):
    """Today's chain keeps the tone's 1 / T band at baseband (DOPPLER.md section 0 item 2), 8 Hz of sample rate here: the
    C++ chain refuses the ping (fewer than three range cells), the Python chain makes a map of it whose detections are
    nonsense (snr of thousands of dB at the window's start: a divergence of the engines on a CW record that predates D3,
    left for a follow-up) and none near the target."""
    rec = ping(CW, targets=(((0.0, 500.0, 0.0), (0.0, 1.0, 0.0)),))
    if engine == "cpp":
        with pytest.raises(ValueError, match="three range cells"):
            run(dict(CHAIN, engine=engine), [rec])
    else:
        (off,) = run(dict(CHAIN, engine=engine), [rec])
        assert near(off, 500.0, 0.0, gate_m=0.25 * C * T) is None
    (on,) = run(dict(CHAIN, engine=engine, doppler=TONE), [rec])
    d = near(on, 500.0, 0.0, gate_m=0.25 * C * T)
    assert d is not None, on["detections"]
    step = d["range_rate_step_m_s"]
    assert step == pytest.approx(C / (4.0 * F_TONE * T))
    assert d["range_rate_m_s"] == pytest.approx(1.0, abs=0.5 * step) and d["doppler_path"] == "tone"
    assert len(on["detections"]) == 1


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_still_target_is_not_given_a_doppler(engine):
    """The same target standing still: inside the notch; the tone path reports nothing (and the tone-only ping's
    zero-Doppler map is not reported either)."""
    (on,) = run(dict(CHAIN, engine=engine, doppler=TONE), [ping(CW, targets=(((0.0, 500.0, 0.0), (0.0, 0.0, 0.0)),))])
    assert on["detections"] == []


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_train_with_the_tone_above_the_band_gives_the_mover_the_sweeps_range_and_the_tones_rate(engine):
    """A Hann tone at 10.5 kHz then the 6-10 kHz sweep: a mover opening at 1.5 m/s at 500 m, a still target at 400 m and
    -20 degrees. The still one is the sweep's alone (path sweep, no rate); the mover is ONE detection, at the range the
    sweep gives without the tone path (within its own coupling) with the tone's rate (path both)."""
    still = (400.0 * np.sin(np.radians(-20.0)), 400.0 * np.cos(np.radians(-20.0)), 0.0)
    rec = ping(TRAIN, targets=(((0.0, 500.0, 0.0), (0.0, 1.5, 0.0)), (still, (0.0, 0.0, 0.0))))
    (off,) = run(dict(CHAIN, engine=engine), [rec])
    (on,) = run(dict(CHAIN, engine=engine, doppler=TONE_TRAIN), [rec])
    mover_off, mover = near(off, 500.0, 0.0), near(on, 500.0, 0.0)
    fixed = near(on, 400.0, -20.0)
    assert mover_off is not None and mover is not None and fixed is not None, on["detections"]
    assert mover["doppler_path"] == "both" and mover["range_rate_m_s"] == pytest.approx(1.5, abs=0.5 * mover["range_rate_step_m_s"])
    assert mover["range_m"] == pytest.approx(mover_off["range_m"], abs=1e-6)
    assert fixed["doppler_path"] == "sweep" and fixed["range_rate_m_s"] is None
    assert len([d for d in on["detections"] if d["doppler_path"] == "tone"]) == 0


def test_a_train_without_its_tone_named_is_refused():
    rec = ping(TRAIN, targets=())
    for engine in ("python", "cpp"):
        with pytest.raises(ValueError, match="tone_frequency_hz"):
            run(dict(CHAIN, engine=engine, doppler=TONE), [rec])


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_alternating_tone_and_sweep_pings_feed_one_tracker(engine):
    """Six pings, every other one a tone alone and the others the sweep alone, a target opening at 1 m/s from 500 m. The
    tone pings report the tone's detection (path tone, coarse range, its rate), the sweep pings the sweep's (path sweep);
    the tracker takes both (the tone's with its own range sigma, c T / 8) and holds one track on the target."""
    recs = []
    for i in range(6):
        x = 500.0 + 1.0 * 2.0 * i
        recs.append(ping(CW if i % 2 == 0 else LFM, targets=(((0.0, x, 0.0), (0.0, 1.0, 0.0)),), seed=10 + i, sequence=i))
    reps = run(dict(CHAIN, engine=engine, doppler=TONE), recs)
    for i, rep in enumerate(reps):
        d = near(rep, 500.0 + 2.0 * i, 0.0, gate_m=0.25 * C * T)
        assert d is not None, (i, rep["detections"])
        assert d["doppler_path"] == ("tone" if i % 2 == 0 else "sweep")
        if i % 2 == 0:
            assert d["range_rate_m_s"] == pytest.approx(1.0, abs=0.5 * d["range_rate_step_m_s"])
    tracks = [t for t in reps[-1]["tracks"] if abs(np.hypot(t["x"], t["y"]) - 510.0) < 30.0]
    assert len(tracks) == 1, reps[-1]["tracks"]


def test_alternating_pings_give_the_same_report_in_both_engines():
    recs = [ping(CW if i % 2 == 0 else LFM, targets=(((0.0, 500.0 + 2.0 * i, 0.0), (0.0, 1.0, 0.0)),), seed=20 + i, sequence=i) for i in range(4)]
    want, got = run(dict(CHAIN, engine="python", doppler=TONE), recs), run(dict(CHAIN, engine="cpp", doppler=TONE), recs)
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"])
        for a, b in zip(w["detections"], g["detections"]):
            for key in ("range_m", "bearing_deg", "snr_db"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key
            assert a["doppler_path"] == b["doppler_path"]
            if a["range_rate_m_s"] is not None:
                assert a["range_rate_m_s"] == pytest.approx(b["range_rate_m_s"], abs=0.01)
        assert [(t["id"], t["kind"]) for t in w["tracks"]] == [(t["id"], t["kind"]) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key


# --- the false-alarm rate -----------------------------------------------------------------------------------------------

def test_on_noise_the_tone_path_stays_inside_its_budget():
    """Noise-only tone pings (nothing in the water), the tone path's budget loosened to 3 false detections a ping: the
    count over 16 pings is at most the budget's 48 plus three Poisson sigmas (the union bound over every cell is on the
    safe side), and the threshold is not so high that nothing ever crosses at 30 a ping."""
    recs = [ping(CW, targets=(), seed=300 + i, sequence=i) for i in range(16)]
    counts = {}
    for budget in (3.0, 30.0):
        reps = run(dict(CHAIN, engine="cpp", doppler=dict(TONE, tone_false_alarms_per_ping=budget)), recs)
        counts[budget] = sum(len(r["detections"]) for r in reps)
    print(f"\nnoise only, 16 tone pings: {counts[3.0]} at 3 a ping, {counts[30.0]} at 30 a ping")
    assert counts[3.0] <= 48 + 3.0 * np.sqrt(48)
    assert counts[30.0] > 0


# --- the option and the tracker -----------------------------------------------------------------------------------------

def test_the_option_is_off_by_default_said_only_when_on_and_unknown_keys_refused():
    assert live.DEFAULTS["doppler"]["tone"] == "off"
    assert not any(k.startswith("doppler") for k in live.cpp_chain_options({}))
    o = live.cpp_chain_options({"doppler": dict(TONE_TRAIN)})
    assert o["doppler_tone"] == "bank" and o["doppler_tone_frequency_hz"] == 10.5e+3 and "doppler_mode" not in o
    with pytest.raises(ValueError, match="tone"):
        doppler.merged({"tone": "on"})
    with pytest.raises(ValueError, match=r"chain\.doppler\.tone_frequency: unknown key; did you mean 'tone_frequency_hz'\?"):
        live.check_keys(dict(live.DEFAULTS, doppler={"tone": "bank", "tone_frequency": 1.0}))
    with pytest.raises(ValueError, match="doppler_tone"):
        core.LiveChainCpp({"doppler_tone": "on"})
    text = json.loads(core.live_options_from_text("doppler_tone=bank\ndoppler_tone_frequency_hz=10500\ndoppler_tone_gate_bearing_deg=3\n"))
    assert text["doppler_tone"] == "bank" and text["doppler_tone_frequency_hz"] == 10500 and text["doppler_tone_gate_bearing_deg"] == pytest.approx(3.0)
    assert json.loads(core.live_options_from_text(""))["doppler_tone"] == "off"
    for engine in ("python", "cpp"):
        with pytest.raises(ValueError, match="range_doppler"):
            LiveChain(dict(CHAIN, engine=engine, range_doppler="ownship", doppler=TONE))


def test_a_detections_own_range_sigma_is_the_trackers_measurement_noise_and_nan_changes_nothing():
    tr = detect.Tracker(range_sigma=2.0, bearing_sigma=np.radians(1.0))
    plain = detect.Detection(500.0, 0.1, 20.0, 1)
    assert np.array_equal(tr._measurement_cov(plain), tr._measurement_cov(detect.Detection(500.0, 0.1, 20.0, 1, range_sigma=float("nan"))))
    coarse = tr._measurement_cov(detect.Detection(500.0, 0.1, 20.0, 1, range_sigma=100.0))
    s, c = np.sin(0.1), np.cos(0.1)
    radial = np.array([s, c])
    assert radial @ coarse @ radial == pytest.approx(100.0 ** 2, rel=1e-9)
    assert radial @ tr._measurement_cov(plain) @ radial == pytest.approx(2.0 ** 2, rel=1e-9)
