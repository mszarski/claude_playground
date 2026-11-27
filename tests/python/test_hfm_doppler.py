"""The hyperbolic sweep (`transmit.waveform: {type: sampled, source: hfm}`) on a mover, with the plain matched filter
(every Doppler option off): the closed forms of gpusonar.doppler `hfm_doppler` / `hfm_range_factor` (ours, derived there)
held to a direct correlation of a time-scaled pulse with the unscaled replica, as test_doppler_bank.py holds the linear
sweep's; the simulator's plain HFM echo of a mover (both engines); and both live-chain engines on an HFM record (the
replica is the record's, type Arbitrary: not rebuilt as a linear sweep from f_start / f_stop). docs/tasks/DOPPLER.md "HFM".

Derivation (ours; gpusonar/doppler.py): the HFM's phase is phi(u) = ln(1 + k f0 u) / k, k = (1/f1 - 1/f0) / T, and
phi(eta t) = ln(eta) / k + phi(t + tau) with tau = (1/eta - 1) f1 T / (f0 - f1): the time-scaled echo is the pulse advanced by
tau, in phase, over the band the two share. So the peak reads c tau / 2 nearer (f1 T / B metres per m/s of closing speed) and
loses only the ends outside the replica: 1 + (1/eta - 1) f1 / B of the matched peak (closing, up-sweep), whatever T.
"""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import doppler
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sim import waveform as wfm
from gpusonar.sonar import SonarModel

from test_sampled_waveform import rel_rms, sonar_with

core = gs.core
C = 1500.0
F0, F1 = 4.0e3, 6.0e3            # the shipped long-range band (configs/sonars/popoto_long_range.yaml)


def tukey(x, alpha):
    return doppler._tukey(np.asarray(x, dtype=float), alpha)


def pulse(kind, u, T, alpha):
    """The analytic signal of the swept pulse at emission times u (as sim/waveform.py builds `lfm` and `hfm`)."""
    u = np.asarray(u, dtype=float)
    x = np.clip(u, 0.0, T)
    if kind == "hfm":
        k = (1.0 / F1 - 1.0 / F0) / T
        phase = np.log1p(k * F0 * x) / k
    else:
        phase = (F0 + 0.5 * (F1 - F0) / T * x) * x
    return tukey(u / T, alpha) * np.exp(2j * np.pi * phase)


def correlate(kind, r, T, alpha, fs=6000.0, fc=5000.0):
    """The matched filter of the unscaled replica on the echo of range rate r (opening positive), complex baseband about fc:
    (peak over the matched peak, lag of the peak in s, positive late). The peak is found on the FFT's lags and then refined
    by direct sums at continuous lags (the replica evaluated exactly at t - L), so no sampling loss enters."""
    eta = (C - r) / (C + r)
    t = np.arange(int((1.05 * T + 0.02) * fs)) / fs
    mix = np.exp(-2j * np.pi * fc * t)
    echo, rep = pulse(kind, eta * t, T, alpha) * mix, pulse(kind, t, T, alpha) * mix
    norm = np.sum(np.abs(rep) ** 2)
    n = 1 << int(np.ceil(np.log2(2 * len(t))))
    out = np.abs(np.fft.ifft(np.fft.fft(echo, n) * np.conj(np.fft.fft(rep, n))))
    i = int(np.argmax(out))
    lag0 = (i if i < n // 2 else i - n) / fs

    def at(lag):
        s = pulse(kind, t - lag, T, alpha) * np.where((t - lag >= 0.0) & (t - lag <= T), 1.0, 0.0) * np.exp(-2j * np.pi * fc * t)
        return abs(np.sum(echo * np.conj(s))) / norm

    lags = lag0 + np.linspace(-1.5, 1.5, 61) / fs
    vals = np.array([at(L) for L in lags])
    j = int(np.argmax(vals))
    fine = lags[j] + np.linspace(-1.0, 1.0, 41) * (lags[1] - lags[0])
    vf = np.array([at(L) for L in fine])
    return float(vf.max()), float(fine[int(np.argmax(vf))]), float(out.max() / norm)


# --- the closed forms against a direct correlation ---------------------------------------------------------------------

@pytest.mark.parametrize("T", [1.0, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("alpha", [0.0, 0.1])
def test_the_hfm_echo_of_a_mover_peaks_where_and_as_high_as_the_closed_form_says(T, alpha):
    """4-6 kHz, the four pulse lengths of the trade study, closing and opening at 2, 5.6 and 8 m/s: the peak within 0.02 dB of
    the closed form and its lag within 10 microseconds (7.5 mm of range; an untapered pulse's cut edge ripples the top of the
    peak by a few microseconds) of -tau, against biases up to 193 m."""
    for r in (-8.0, -5.6, -2.0, 2.0, 8.0):
        want = doppler.hfm_doppler(r, F0, F1, T, C, window_alpha=alpha)
        peak, lag, coarse = correlate("hfm", r, T, alpha)
        assert -20.0 * np.log10(peak) == pytest.approx(want["loss_db"], abs=0.02), (T, alpha, r)
        assert lag == pytest.approx(-want["time_shift_s"], abs=1e-5), (T, alpha, r)
        assert 0.5 * C * lag == pytest.approx(want["range_error_m"], abs=7.5e-3)       # the range the chain would report, less the truth
        assert coarse <= peak + 1e-9                                       # no other lag of the FFT beats it
        # First order: f1 T / B metres per m/s (3 T at 4-6 kHz), within (r / c) of itself.
        assert want["range_error_m"] == pytest.approx(doppler.hfm_range_factor(F0, F1, T) * r, rel=abs(r) / C * 1.01 + 1e-12)


def test_the_closed_form_numbers_the_study_quotes():
    """The table of DOPPLER.md "HFM" (4-6 kHz, c = 1500 m/s): the loss does not grow with the pulse, the bias does."""
    for T in (1.0, 2.0, 4.0, 8.0):
        rect = {r: doppler.hfm_doppler(r, F0, F1, T, C)["loss_db"] for r in (-8.0, -2.0, 2.0, 8.0)}
        assert rect[-8.0] == pytest.approx(0.281, abs=0.001) and rect[8.0] == pytest.approx(0.188, abs=0.001)
        assert rect[-2.0] == pytest.approx(0.070, abs=0.001) and rect[2.0] == pytest.approx(0.047, abs=0.001)
        assert doppler.hfm_doppler(-8.0, F0, F1, T, C, window_alpha=0.1)["loss_db"] == pytest.approx(0.184, abs=0.002)
        assert doppler.hfm_range_factor(F0, F1, T) == pytest.approx(3.0 * T)
        # Untapered: the closed form itself, closing 1 + (1/eta - 1) f1 / B, opening 1 - (1/eta - 1) f0 / B.
        for r in (-8.0, 8.0):
            eps = (C + r) / (C - r) - 1.0
            ratio = 1.0 + eps * F1 / (F1 - F0) if r < 0 else 1.0 - eps * F0 / (F1 - F0)
            assert doppler.hfm_doppler(r, F0, F1, T, C)["peak_ratio"] == pytest.approx(ratio, rel=1e-12)
    assert doppler.hfm_doppler(0.0, F0, F1, 1.0, C, window_alpha=0.1)["loss_db"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        doppler.hfm_range_factor(F0, F0, 1.0)


@pytest.mark.parametrize("T", [1.0, 4.0])
def test_the_linear_sweeps_loss_grows_with_the_pulse_and_the_hyperbolic_ones_does_not(T):
    """The same correlation for the linear sweep of the same band and taper: the library's closed form (free delay,
    test_doppler_bank.py) within 0.5 dB untapered, and the taper only lowers it; the HFM loses under 0.3 dB at every T."""
    for r in (-8.0, -2.0):
        lfm_peak, _, _ = correlate("lfm", r, T, 0.0)
        want = -20.0 * np.log10(core.live_sweep_mismatch(np.pi * (F1 - F0) * T * abs(r) / C))
        assert -20.0 * np.log10(lfm_peak) == pytest.approx(want, abs=0.5), (T, r)
        tapered, _, _ = correlate("lfm", r, T, 0.1)
        assert -20.0 * np.log10(tapered) < want + 0.1
        hfm_peak, _, _ = correlate("hfm", r, T, 0.1)
        assert -20.0 * np.log10(hfm_peak) < 0.3
        assert -20.0 * np.log10(lfm_peak) > 8.0                  # 8.8 dB at 2 m/s for 1 s (the transmit table's)


def test_what_the_hfm_costs_at_zero_speed_resolution_sidelobes_and_spectrum():
    """At zero speed, 1 s, Tukey 0.1, unwindowed filter: the HFM's compressed main lobe, its peak sidelobe and its spectrum's
    tilt against the linear sweep's (numbers DOPPLER.md "HFM" quotes; ours)."""
    fs, T = 24000.0, 1.0
    t = np.arange(int(T * fs) + 1) / fs
    out = {}
    for kind in ("lfm", "hfm"):
        s = pulse(kind, t, T, 0.1) * np.exp(-2j * np.pi * 5000.0 * t)
        n = 1 << int(np.ceil(np.log2(8 * len(t))))
        sp = np.fft.fft(s, n)
        up = 16                                                      # lags at 1 / (16 fs): the spectrum padded about its edges
        power = np.fft.fftshift(np.abs(sp) ** 2)
        ac = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.pad(power, ((up - 1) * n // 2, (up - 1) * n // 2))))))
        ac /= ac.max()
        k = int(np.argmax(ac))
        above = np.nonzero(ac[k - 40 * up:k + 40 * up] >= 10 ** (-3 / 20))[0]
        width = (above.max() - above.min()) / (up * fs)              # the -3 dB width of |output|, s
        # first nulls, then the largest sidelobe beyond them
        j = k
        while ac[j + 1] < ac[j]:
            j += 1
        out[kind] = {"width_m": 0.5 * C * width, "sidelobe_db": float(20 * np.log10(ac[j:].max()))}
        f = np.fft.fftfreq(n, 1 / fs) + 5000.0
        band = lambda lo, hi: float(np.mean(np.abs(sp[(f > lo) & (f < hi)]) ** 2))   # noqa: E731
        out[kind]["tilt_db"] = 10 * np.log10(band(4150.0, 4350.0) / band(5650.0, 5850.0))
    print("\n" + json.dumps(out, indent=1))
    # Resolution: within 10 % of the linear sweep's (the HFM spends longer at the low end: a slightly narrower rms band).
    assert out["hfm"]["width_m"] == pytest.approx(out["lfm"]["width_m"], rel=0.10)
    # Spectrum: the HFM's energy density goes as 1 / f^2 (dwell 1 / (df/du) = T f0 f1 / (B f^2)): 4.25 against 5.75 kHz is
    # 20 log10(5.75 / 4.25) = 2.6 dB; the linear sweep's is flat.
    assert out["hfm"]["tilt_db"] == pytest.approx(20 * np.log10(5750.0 / 4250.0), abs=0.3)
    assert abs(out["lfm"]["tilt_db"]) < 0.3
    # Sidelobes: both well below the main lobe; the HFM's within 3 dB of the linear sweep's.
    assert out["hfm"]["sidelobe_db"] < -10.0 and out["hfm"]["sidelobe_db"] == pytest.approx(out["lfm"]["sidelobe_db"], abs=3.0)


# --- the simulator: a plain HFM's echo of a mover, both engines --------------------------------------------------------

@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_the_simulated_hfm_echo_of_a_mover_is_the_pulse_time_scaled_and_so_the_pulse_advanced(engine):
    """A 0.25 s, 4-6 kHz HFM (Tukey 0.1) from a point closing at 8 m/s at 400 m, one omni hydrophone, no spreading or
    absorption: the echo is b(eta (t - t0)) on a carrier scaled by eta (Doppler as time scaling: the existing
    test_sampled_waveform.py test does this for a sampled burst pair, test_pulse_train.py for trains; this is the plain HFM),
    and, over the band both share, the UNscaled pulse advanced by tau (the property the closed forms rest on)."""
    r0, v, T = 400.0, 8.0, 0.25
    wave = {"type": "sampled", "source": "hfm", "f_start": F0, "f_stop": F1, "duration": T, "window_alpha": 0.1, "gain_knots": 0}
    sonar = sonar_with("popoto_long_range", wave, r0 - 20.0, r0 + 40.0, "baseband", omni=True,
                       transmit={"lever_arm": [0.0, 0.0, 0.0]}, array={"count": 1}, receive={"sample_rate": 8.0e+3})
    target = sonar.boresight * r0
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target], velocity=-v * sonar.boresight),
                        environment={"sound_speed": C}, absorption_model="none", adc=False, spreading=False, engine=engine)
    s = rec.samples
    x = s.array[0] / rec.transmit.waveform.amplitude
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    eta = (C + v) / (C - v)                     # closing at v: the bank's convention (c - r) / (c + r) with r = -v
    t0 = 2.0 * r0 / (C + v)
    w = sonar.sampled_waveform
    mix = np.exp(-2j * np.pi * s.center_frequency * t)
    scaled = w.evaluate(eta * (t - t0)) * np.exp(2j * np.pi * w.carrier * eta * (t - t0)) * mix
    err = rel_rms(scaled, x)
    print(f"\n{engine}: echo against the time-scaled pulse {err:.2e}")
    assert err < 1e-6
    # The same echo as the unscaled pulse advanced by tau, where both are inside the pulse (the shared band), up to the
    # constant phase ln(eta) / k; the taper is scaled, not advanced, so compare away from the tapered ends.
    tau = doppler.hfm_doppler(-v, F0, F1, T, C)["time_shift_s"]
    u = t - t0
    inside = (u > 0.06 * T) & (u + tau < 0.94 * T)
    k = (1.0 / F1 - 1.0 / F0) / T
    advanced = pulse("hfm", u + tau, T, 0.0) * np.exp(1j * np.log(eta) / k * 2 * np.pi) * mix
    assert rel_rms(advanced[inside], x[inside]) < 1e-5
    assert tau == pytest.approx((1.0 - 1.0 / eta) * F1 * T / (F1 - F0), rel=1e-12) and tau > 0.0


def test_both_simulator_engines_agree_on_a_one_second_hfm_mover():
    r0, v = 1300.0, 5.6
    wave = {"type": "sampled", "source": "hfm", "f_start": F0, "f_stop": F1, "duration": 1.0, "window_alpha": 0.1}
    sonar = sonar_with("popoto_long_range", wave, r0 - 100.0, r0 + 100.0, "baseband", receive={"sample_rate": 8.0e+3})
    scene = Scene().add_points([sonar.boresight * r0], velocity=-v * sonar.boresight)
    a, b = (simulate_ping(sonar, make_pose(velocity=(1.5, 0.0, 0.0)), scene, environment={"sound_speed": C}, adc=False,
                          absorption_model="francois_garrison", engine=e) for e in ("cpp", "numpy"))
    err = rel_rms(b.samples.array, a.samples.array)
    print(f"\nC++ against NumPy, 1 s HFM, 32 channels: {err:.2e}")
    assert err < 2e-6


# --- both live-chain engines on an HFM record, every Doppler option off ------------------------------------------------

WINDOW = [1000.0, 1600.0]
CHAIN = {"bearings_deg": [-30.0, 30.0, 0.5], "backend": "host"}


def chain_ping(waveform, v, seed=3, sequence=0):
    """popoto_long_range (one row of 32, 4-6 kHz) at 8 kHz baseband, a point at 1300 m and 10 degrees closing at v along the
    line of sight, still platform, noise only, free field."""
    over = {"transmit": {"waveform": waveform}, "receive": {"kind": "baseband", "sample_rate": 8.0e+3, "window": {"range": WINDOW}}}
    sonar = SonarModel.load("popoto_long_range", over)
    b = np.radians(10.0)
    los = np.array([np.sin(b), np.cos(b), 0.0])
    scene = Scene().add_points([list(1300.0 * los)], velocity=list(-v * los), target_strength_db=-12.0)
    return simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C}, noise_rms=3e-6, adc=False, absorption_model="none",
                         rng=np.random.default_rng(seed), sequence=sequence, time_ns=sequence * 2_000_000_000)


HFM_1S = {"type": "sampled", "source": "hfm", "f_start": F0, "f_stop": F1, "duration": 1.0, "window_alpha": 0.1}
LFM_1S = {"type": "lfm", "f_start": F0, "f_stop": F1, "duration": 1.0, "window_alpha": 0.1}


def detection(report, want_range, gate=60.0):
    near = [d for d in report["detections"] if abs(d["bearing_deg"] - 10.0) < 1.5 and abs(d["range_m"] - want_range) < gate]
    return max(near, key=lambda d: d["snr_db"]) if near else None


def same_reports(a, b, rel=1e-6):
    """Two reports equal to the C++ engine's printed digits (it writes numbers with 10 significant digits): every key and
    every string the same, every number within `rel` (1e-6 relative, or 1e-9 absolute about zero)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_reports(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_reports(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= max(rel * max(abs(a), abs(b)), 1e-9)
    return a == b


def peak_range(d):
    """The detection's strongest arrival's range (the report's `range_m` is the detector cell's, coarser)."""
    return float(max(d["arrivals"], key=lambda a: a[1])[0])


def test_an_hfm_record_is_compressed_with_its_own_replica_in_both_engines_and_they_agree():
    """The record carries the HFM as type Arbitrary with its replica; pulse compression (libs/dsp build_arbitrary) builds the
    filter from that replica, not a linear sweep from f_start / f_stop. Proof: at zero speed the HFM's detection is as
    strong as the linear sweep's of the same energy (a linear replica on an HFM echo would lose tens of dB), and both
    engines give the same report for 2 pings of each (to the C++ engine's printed digits: `same_reports`)."""
    strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731
    rec = chain_ping(HFM_1S, 0.0)
    assert rec.transmit.waveform.type == core.WaveformType.Arbitrary and len(rec.transmit.waveform.replica) > 0
    snr = {}
    for name, wave in (("hfm", HFM_1S), ("lfm", LFM_1S)):
        recs = [chain_ping(wave, 0.0, seed=s, sequence=s) for s in range(2)]
        reports = {}
        for engine in ("python", "cpp"):
            with LiveChain(dict(CHAIN, engine=engine)) as ch:
                reports[engine] = [strip(ch.process(r.copy())) for r in recs]
        assert same_reports(reports["python"], reports["cpp"]), name
        dets = [detection(r, 1300.0) for r in reports["cpp"]]
        assert all(d is not None for d in dets), name
        snr[name] = float(np.mean([d["snr_db"] for d in dets]))
    print(f"\nzero speed: snr HFM {snr['hfm']:.2f} dB, LFM {snr['lfm']:.2f} dB")
    assert snr["hfm"] == pytest.approx(snr["lfm"], abs=1.0)


def test_the_chain_on_an_hfm_mover_keeps_its_peak_and_reads_the_closed_forms_bias():
    """Closing at 5.6 m/s, 1 s, plain filter, C++ engine (the engines agree: the test above): the HFM's snr within 0.7 dB of its
    zero-speed snr on the same noise, the linear sweep's at least 3 dB down (its closed-form peak loss is 13.6 dB; the
    multiscale detector's cells take some back, DOPPLER.md LS.3); the HFM's strongest arrival nearer than the zero-speed one by
    the closed form's 16.7 m plus the 4.8 m the target closes while the pulse reaches it (v R / (c + v)), within the map's
    0.25 m samples and the simulated echo's own scaling (0.5 m)."""
    out = {}
    for name, wave in (("hfm", HFM_1S), ("lfm", LFM_1S)):
        with LiveChain(dict(CHAIN, engine="cpp")) as ch:
            d0 = detection(ch.process(chain_ping(wave, 0.0)), 1300.0)
        with LiveChain(dict(CHAIN, engine="cpp")) as ch:
            d1 = detection(ch.process(chain_ping(wave, 5.6)), 1300.0)
        assert d0 is not None and d1 is not None, name
        out[name] = (d0, d1)
    (h0, h1), (l0, l1) = out["hfm"], out["lfm"]
    print(f"\nHFM snr {h0['snr_db']:.2f} -> {h1['snr_db']:.2f} dB, range {h0['range_m']:.2f} -> {h1['range_m']:.2f} m; "
          f"LFM snr {l0['snr_db']:.2f} -> {l1['snr_db']:.2f} dB, range {l0['range_m']:.2f} -> {l1['range_m']:.2f} m")
    assert h1["snr_db"] > h0["snr_db"] - 0.7
    assert l1["snr_db"] < l0["snr_db"] - 3.0
    bias = doppler.hfm_doppler(-5.6, F0, F1, 1.0, C)["range_error_m"]
    assert bias == pytest.approx(-16.74, abs=0.01)
    moved = -5.6 * 1300.0 / (C + 5.6)
    assert peak_range(h1) - peak_range(h0) == pytest.approx(bias + moved, abs=0.5)


def test_a_sampled_hfm_fits_the_frozen_record_and_the_bank_refuses_it():
    """No schema change: the HFM travels in the Arbitrary waveform's existing fields (type, band, duration, replica and its
    rate). The sweep bank takes a linear sweep only (libs/live plan_doppler): it refuses an HFM record, in both engines."""
    rec = chain_ping(HFM_1S, 0.0)
    w = rec.transmit.waveform
    assert (w.f_start, w.f_stop, w.duration) == (F0, F1, 1.0) and w.replica_sample_rate == pytest.approx(24000.0)
    back = wfm.SampledWaveform.from_record(w)
    built = wfm.build_waveform(HFM_1S)
    u = np.linspace(0.05, 0.95, 2001)
    assert rel_rms(built.evaluate(u), back.evaluate(u)) < 1e-4
    for engine in ("python", "cpp"):
        with pytest.raises((ValueError, RuntimeError), match="linear sweep"):
            with LiveChain(dict(CHAIN, engine=engine, doppler={"mode": "sweep_bank", "max_range_rate_m_s": 1.0})) as ch:
                ch.process(rec.copy())


def test_the_eval_tool_runs_a_small_case(tmp_path):
    """tools/hfm_doppler_eval.py (the measurement of DOPPLER.md "HFM") on one ping of each waveform at 0 and 5.6 m/s, 1 s, both
    engines: its scene (free field, still platform, the two targets closing along their lines of sight), the engines equal, the
    HFM within 1 dB of its zero-speed margin on the point and the LFM more than 3 dB under, the HFM's range error the closed
    form's within 1 m."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
    import hfm_doppler_eval as ev
    cfg = ev.scene_config("hfm", 1.0, 5.6, 1)
    assert cfg["fidelity"] == "free_field" and cfg["reverberation"] == [] and cfg["platform"]["velocity"] == [0.0, 0.0, 0.0]
    assert cfg["sonar_overrides"]["transmit"]["waveform"] == {"f_start": F0, "f_stop": F1, "duration": 1.0, "window_alpha": 0.1,
                                                              "type": "sampled", "source": "hfm"}
    tr = ev.truth(cfg, 1, C)
    # the 40 m of depth between the platform (10 m) and the targets (50 m) over 12 km: cos 0.19 deg
    assert tr["point"][0]["range_rate"] == pytest.approx(-5.6, abs=1e-4) and tr["hull"][0]["range_rate"] == pytest.approx(-5.6, abs=1e-4)
    out = tmp_path / "e.json"
    assert ev.main(["--speeds", "0", "5.6", "--pings", "1", "--ts-offset-db", "20", "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert all(c["plain"]["engines_equal"] for c in res["cases"])
    rel = {(x["waveform"], x["target"]): x for x in res["relative"] if x["speed_m_s"] == 5.6 and x["engine"] == "cpp"}
    assert rel[("hfm", "point")]["relative_db"] > -1.0 and rel[("lfm", "point")]["relative_db"] < -3.0
    assert rel[("hfm", "point")]["range_error_m"] == pytest.approx(doppler.hfm_doppler(-5.6, F0, F1, 1.0, C)["range_error_m"], abs=1.0)
