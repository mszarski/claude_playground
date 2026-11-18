"""Sampled (arbitrary) transmit waveforms: python/gpusonar/sim/waveform.py.

- the interpolator and the variable-delay resampler against TR 0702 (R. P. Goddard, "The
  Sonar Simulation Toolset, Release 4.6: Science, Mathematics, and Algorithms", APL-UW
  TR 0702, 2008): the VarDelay script of Sec. 4.3 (p. 20-21) and the design rule eq. 19;
- a sampled copy of an LFM against the analytic LFM, in every propagation tier;
- phase-coded pulses through pulse compression (peak, sidelobes, range);
- Doppler as time scaling against the closed-form factor of TR 0702 eqs. 63 and 68;
- the record carries the pulse in its existing fields, and survives the container file.

Measured numbers are printed (run with -s) and quoted in the assertions' comments.
"""

import json
import os
import warnings

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.processing import doppler_bank, doppler_scale
from gpusonar.sim import Scene, make_pose, run_scenario, simulate_ping
from gpusonar.sim import waveform as wfm
from gpusonar.sim.bellhop import find_bellhop
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
OMNI = {k: None for k in ("beamwidth_az", "beamwidth_el")}


def sonar_with(name, waveform, r0, r1, kind="baseband", omni=False, **extra):
    over = {"transmit": {"waveform": waveform}, "receive": {"window": {"range": [r0, r1]}, "kind": kind}}
    if omni:
        over["array"] = {**OMNI, "element_width": None, "element_height": None}
        over["transmit"].update({**OMNI, "aperture_width": None, "aperture_height": None})
        over["fov"] = None
    if extra:
        over = gs.config.deep_merge(over, extra)
    return SonarModel.load(name, over)


def rel_rms(a, b):
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2) / np.mean(np.abs(a) ** 2)))


def db(x):
    return 20.0 * np.log10(max(float(x), 1e-300))


# --- TR 0702 Sec. 4.3: the VarDelay script ------------------------------------------------

def test_vardelay_script_of_tr0702_lowers_each_tone_by_one_minus_delay_rate():
    """The script on p. 20-21: a HarmonicFamily (fundamental 220 Hz; harmonic 1 at -3 dB and
    0 deg, harmonic 2 at -6 dB and 90 deg; eq. 29: sum A_n cos(2 pi n f1 t + phi_n)) sampled
    as real samples at 8000 Hz from -2 s to 5 s, delayed by a delay given once a second from
    0 to 5 s as (1.00 1.10 1.25 1.45 1.7 2.0) s, output from 0 to 5 s at 8000 Hz. The delay is
    linear between its samples, so in second k the tones are at n f1 (1 - dT/dt), with
    dT/dt = 0.10, 0.15, 0.20, 0.25, 0.30."""
    fs, f1 = 8000.0, 220.0
    amps = [10 ** (-3 / 20), 10 ** (-6 / 20)]
    phases = [0.0, np.pi / 2]

    def signal(t):
        return sum(a * np.cos(2 * np.pi * (n + 1) * f1 * t + ph) for n, (a, ph) in enumerate(zip(amps, phases)))

    t_in = -2.0 + np.arange(int(7 * fs) + 1) / fs
    delay_t = np.arange(6.0)
    delay = np.array([1.00, 1.10, 1.25, 1.45, 1.7, 2.0])
    t_out = np.arange(int(5 * fs) + 1) / fs
    for taps, beta, tol in ((6, None, 1e-2), (16, None, 5e-4), (16, 12.0, 5e-6)):
        y = wfm.var_delay(signal(t_in), fs, t_in[0], t_out, delay_t, delay, taps=taps, beta=beta)
        exact = signal(t_out - np.interp(t_out, delay_t, delay))
        err = np.abs(y - exact).max()
        print(f"VarDelay script, {taps} taps, {'Hann' if beta is None else 'Kaiser'}: max error {err:.2e}")
        assert err < tol            # measured 5.1e-3 (SST's default order 5), 2.3e-4, 1.5e-6
    # Instantaneous frequency in each second, from the spectrum of its middle 0.8 s.
    for k in range(5):
        rate = delay[k + 1] - delay[k]
        seg = y[int((k + 0.1) * fs):int((k + 0.9) * fs)]
        n_fft = 1 << 20
        spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
        f = np.fft.rfftfreq(n_fft, 1 / fs)
        for n in (1, 2):
            expect = n * f1 * (1.0 - rate)
            band = (f > expect - 30) & (f < expect + 30)
            measured = f[band][np.argmax(spectrum[band])]
            assert measured == pytest.approx(expect, abs=0.02)
            assert measured < n * f1        # "tones get lower as the rate of change of delay gets larger"


# --- TR 0702 eq. 19: the interpolator design rule -------------------------------------------

def _response_error(taps, alpha, beta=None):
    """Worst error of the interpolator seen as a filter, for a signal filling the fraction
    alpha of the Nyquist band: departure from 1 in the band, and leakage of the copies of
    the spectrum centred on multiples of the sample rate (the nearest starts at 1 - alpha/2
    cycles per sample)."""
    passband = np.abs(wfm.interpolator_response(taps, np.linspace(0.0, alpha / 2, 201), beta) - 1.0).max()
    images = np.abs(wfm.interpolator_response(taps, np.linspace(1.0 - alpha / 2, 4.0, 3001), beta)).max()
    return passband, images


def test_interpolator_error_against_the_design_rule_eq19():
    """Eq. 19: with the Hann-windowed sinc of eq. 17 use M >= 2.5 / (1 - alpha) taps. The TR
    gives no error figure; this measures what the rule buys with our implementation of
    eq. 17: about -30 to -40 dB at the rule, -54 dB at twice the rule, and no better than
    -13 dB at half of it."""
    for alpha, at_rule in ((0.5, 6), (0.8, 14), (0.9, 26)):      # even M just above 2.5 / (1 - alpha)
        assert at_rule - 2 < 2.5 / (1 - alpha) <= at_rule
        row = []
        for taps in (at_rule // 2 // 2 * 2, at_rule, 2 * at_rule):
            passband, images = _response_error(taps, alpha)
            row.append(max(passband, images))
            print(f"eq. 19, alpha={alpha}, M={taps} (rule: M >= {2.5 / (1 - alpha):.1f}): "
                  f"passband {db(passband):.1f} dB, images {db(images):.1f} dB")
        assert db(row[0]) > -20.0                 # measured -11 to -13 dB: the transition covers the band edge
        assert -45.0 < db(row[1]) < -25.0         # measured -39.1, -33.6, -29.4 dB at the rule
        assert db(row[2]) < -50.0                 # measured -54.3, -54.6, -54.6 dB at twice the rule
    # SST's default order 5 (M = 6), "appropriate for no more than 50 % band coverage".
    assert db(max(_response_error(6, 0.5))) == pytest.approx(-39.1, abs=0.5)
    # The response is what a signal sees: a tone at the band edge, interpolated at random
    # places, is wrong by the passband error plus the leakage of its images.
    rng = np.random.default_rng(0)
    n = np.arange(4000)
    pos = rng.uniform(100, 3900, 4000)
    for alpha, taps in ((0.8, 14), (0.8, 28)):
        err = wfm.interpolate(np.exp(1j * np.pi * alpha * n), pos, taps) - np.exp(1j * np.pi * alpha * pos)
        passband, images = _response_error(taps, alpha)
        assert np.abs(err).max() < 2.5 * max(passband, images)
        assert np.abs(err).max() > 0.5 * max(passband, images)
    # The interpolator the simulator uses by default (16 taps, Kaiser window): the envelope
    # is sampled at 8 times its bandwidth (alpha = 0.125), and up to alpha = 0.5 the error
    # stays below -110 dB (measured -117 dB); the Hann window with 16 taps gives -62 dB.
    for alpha in (0.125, 0.25, 0.5):
        ours = max(_response_error(wfm.DEFAULT_TAPS, alpha, wfm.DEFAULT_BETA))
        hann = max(_response_error(wfm.DEFAULT_TAPS, alpha))
        print(f"default interpolator, alpha={alpha}: Kaiser {db(ours):.1f} dB, Hann {db(hann):.1f} dB")
        assert db(ours) < -110.0 and db(hann) > -80.0


# --- built-in pulses --------------------------------------------------------------------------

def test_barker_and_m_sequence_codes_have_their_defining_autocorrelation():
    for length, code in wfm.BARKER.items():
        acf = np.correlate(code, code, mode="full")
        assert acf[length - 1] == length
        assert np.abs(np.delete(acf, length - 1)).max() <= 1          # the definition of a Barker code
    for degree in (3, 5, 7, 10):
        chips = wfm.mseq(degree)
        n = 2 ** degree - 1
        assert len(chips) == n and abs(chips.sum()) == 1
        periodic = np.array([np.sum(chips * np.roll(chips, k)) for k in range(n)])
        assert periodic[0] == n and np.all(periodic[1:] == -1)        # two-valued: maximal length
    with pytest.raises(ValueError, match="maximal"):
        wfm.mseq(4, taps=[2, 4])                                      # period 6, not 15


def test_hfm_period_is_linear_in_time_and_stepped_fm_dwells_on_each_frequency():
    f0, f1, T = 6e3, 10e3, 0.05
    w = wfm.build_waveform({"source": "hfm", "f_start": f0, "f_stop": f1, "duration": T, "oversample": 16})
    u = np.arange(len(w.envelope)) / w.sample_rate
    f_inst = w.carrier + np.diff(np.unwrap(np.angle(w.envelope))) * w.sample_rate / (2 * np.pi)
    mid = 0.5 * (u[1:] + u[:-1])
    np.testing.assert_allclose(1.0 / f_inst, 1 / f0 + (1 / f1 - 1 / f0) * mid / T, rtol=2e-5)
    s = wfm.build_waveform({"source": "stepped_fm", "f_start": f0, "f_stop": f1, "duration": T, "steps": 5,
                            "oversample": 16})
    f_inst = s.carrier + np.diff(np.unwrap(np.angle(s.envelope))) * s.sample_rate / (2 * np.pi)
    per = len(f_inst) // 5
    for i in range(5):
        assert np.median(f_inst[i * per + 2:(i + 1) * per - 2]) == pytest.approx(f0 + (f1 - f0) * i / 4, rel=1e-9)


def test_sources_file_and_inline_samples_give_the_same_pulse(tmp_path):
    """A pulse given as complex envelope samples (.npy, HDF5, inline) or as real passband
    samples (.wav, .npy) is the pulse the built-in makes."""
    import h5py
    from scipy.io import wavfile
    f0, f1, T = 6e3, 10e3, 0.02
    base = {"type": "sampled", "f_start": f0, "f_stop": f1}
    ref = wfm.build_waveform({**base, "source": "lfm", "duration": T, "window_alpha": 0.3})
    u_test = np.linspace(0.001, T - 0.001, 500)
    env_rate = 32e3
    env = ref.evaluate(np.arange(int(T * env_rate) + 1) / env_rate)
    np.save(tmp_path / "env.npy", env)
    with h5py.File(tmp_path / "pulse.h5", "w") as f:
        d = f.create_dataset("tx/envelope", data=env)
        d.attrs["sample_rate"] = env_rate
    pass_rate = 48e3
    real = ref.passband(pass_rate)
    wavfile.write(tmp_path / "pulse.wav", int(pass_rate), real.astype(np.float32))
    np.save(tmp_path / "real.npy", real)
    cases = {
        "npy envelope": {"source": "file", "file": str(tmp_path / "env.npy"), "sample_rate": env_rate},
        "hdf5 envelope": {"source": "file", "file": str(tmp_path / "pulse.h5"), "dataset": "tx/envelope"},
        "inline envelope": {"source": "samples", "samples": np.column_stack([env.real, env.imag]).tolist(),
                            "sample_rate": env_rate},
        "wav passband": {"source": "file", "file": str(tmp_path / "pulse.wav")},
        "npy passband": {"source": "file", "file": str(tmp_path / "real.npy"), "sample_rate": pass_rate},
        "inline passband": {"source": "samples", "samples": real.tolist(), "sample_rate": pass_rate},
    }
    for label, cfg in cases.items():
        w = wfm.build_waveform({**base, **cfg})
        assert w.duration == pytest.approx(T, abs=1.0 / env_rate)
        err = np.abs(w.evaluate(u_test) - ref.evaluate(u_test)).max()
        print(f"{label}: max envelope error {err:.2e}")
        assert err < (1e-9 if "envelope" in label else 3e-4)     # measured 6e-16; 4.6e-5 (analytic signal of a burst)
    with pytest.raises(ValueError, match="sample_rate"):
        wfm.build_waveform({**base, "source": "file", "file": str(tmp_path / "env.npy")})
    with pytest.raises(ValueError, match="twice the top"):
        wfm.build_waveform({**base, "source": "samples", "samples": real.tolist(), "sample_rate": 15e3})
    with pytest.raises(ValueError, match="unknown source"):
        wfm.build_waveform({**base, "source": "chirpy", "duration": T})


# --- the record ---------------------------------------------------------------------------------

def test_record_carries_the_pulse_in_its_existing_fields_and_survives_the_file(tmp_path):
    sonar = sonar_with("popoto_8", {"type": "sampled", "source": "barker", "code": 13, "duration": 6.5e-3},
                       95.0, 105.0)
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([sonar.boresight * 100.0]),
                        environment={"sound_speed": C}, absorption_model="none", adc=False)
    report = gs.validate(rec)
    assert report.ok, str(report)
    w = rec.transmit.waveform
    assert w.type == core.WaveformType.Arbitrary
    assert w.replica_sample_rate == 4.0 * sonar.waveform_config["f_stop"]
    assert len(w.replica) == int(np.floor(w.duration * w.replica_sample_rate)) + 1
    path = str(tmp_path / "ping.h5")
    with core.RecordWriter(path, core.WriterOptions()) as writer:
        writer.write(rec)
    back = core.RecordReader(path).read(0)
    assert back.transmit.waveform.type == core.WaveformType.Arbitrary
    np.testing.assert_array_equal(np.asarray(back.transmit.waveform.replica), np.asarray(w.replica))
    assert back.transmit.waveform.replica_sample_rate == w.replica_sample_rate
    # The envelope recovered from the record's real replica is the one that was transmitted.
    original = sonar.sampled_waveform
    recovered = wfm.SampledWaveform.from_record(back.transmit.waveform)
    u = np.linspace(0.0, w.duration, 2000)[5:-5]
    err = rel_rms(original.evaluate(u), recovered.evaluate(u))
    print(f"envelope recovered from the record's replica: relative rms error {err:.2e}")
    assert err < 1e-2                                                # measured 3.4e-3 (the code's abrupt ends)
    # Processing needs nothing but the record: compressing the file's copy gives the same result.
    a, b = rec.copy(), back
    recon.prepare([a, b])
    np.testing.assert_allclose(a.samples.array, b.samples.array, atol=1e-6 * np.abs(a.samples.array).max())


# --- a sampled copy of an LFM is the analytic LFM ----------------------------------------------

def _five_targets(sonar, r0):
    rng = np.random.default_rng(1)
    pts = sonar.boresight * r0 + rng.uniform(-1, 1, (5, 3))
    return Scene().add_points(pts, amplitude=np.exp(1j * rng.uniform(0, 6, 5)))


@pytest.mark.parametrize("kind", ["baseband", "real"])
def test_sampled_lfm_matches_analytic_lfm_to_interpolation_accuracy(kind):
    """Omnidirectional elements and no absorption, so the only difference is interpolation
    of the sampled envelope. Tukey-tapered LFM (alpha 0.2), sas_auv (180-220 kHz, 2 ms).
    Measured relative rms error against envelope oversampling (samples per cycle of
    bandwidth): 1.25 -> 1.3e-2, 2 -> 1.9e-4, 4 -> 1.4e-5, 8 -> 2.4e-6, 16 -> 1.3e-6 (the
    last is the float32 rounding of the two records)."""
    errors = {}
    for oversample in (None, 1.25, 2, 4, 8, 16):
        wave = {"window_alpha": 0.2} if oversample is None else {
            "type": "sampled", "source": "lfm", "window_alpha": 0.2, "oversample": oversample}
        sonar = sonar_with("sas_auv", wave, 17.0, 23.0, kind, omni=True)
        rec = simulate_ping(sonar, make_pose(velocity=(1.5, 0.0, 0.0)), _five_targets(sonar, 20.0),
                            environment={"sound_speed": C}, absorption_model="none", adc=False)
        if oversample is None:
            ref = rec.samples.array
        else:
            errors[oversample] = rel_rms(ref, rec.samples.array)
            print(f"sampled LFM vs analytic ({kind}), oversample {oversample}: {errors[oversample]:.2e} "
                  f"({db(errors[oversample]):.1f} dB)")
    assert errors[1.25] < 3e-2 and errors[2] < 5e-4 and errors[4] < 5e-5 and errors[8] < 1e-5 and errors[16] < 5e-6
    assert errors[1.25] > errors[2] > errors[4] > errors[8] > errors[16]


def test_sampled_lfm_with_rectangular_edges_differs_only_at_the_edges():
    """A pulse that switches on and off abruptly is not band-limited, so its samples do not
    define it between samples near the edges. The error is confined there and falls slowly
    (measured 1.5e-2, 8.3e-3, 5.6e-3, 3.9e-3 for oversampling 2, 4, 8, 16)."""
    out = {}
    for oversample in (None, 2, 8):
        wave = {"window_alpha": 0.0} if oversample is None else {
            "type": "sampled", "source": "lfm", "window_alpha": 0.0, "oversample": oversample}
        sonar = sonar_with("sas_auv", wave, 17.0, 23.0, omni=True)
        scene = Scene().add_points([sonar.boresight * 20.0])
        out[oversample] = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C},
                                        absorption_model="none", adc=False).samples.array[0]
    ref = out[None]
    on = np.nonzero(np.abs(ref) > 0)[0]
    edge = int(0.05 * len(on))
    inner = slice(on[0] + edge, on[-1] - edge)
    for oversample in (2, 8):
        assert rel_rms(ref, out[oversample]) < 3e-2
        assert rel_rms(ref[inner], out[oversample][inner]) < 2e-3
    assert rel_rms(ref, out[8]) < rel_rms(ref, out[2])


@pytest.mark.parametrize("name, r0", [("fls_300k", 20.0), ("flank_32", 300.0)])
def test_sampled_lfm_with_beam_patterns_and_absorption(name, r0):
    """With beam patterns and absorption the analytic kernel applies the loss at the
    instantaneous frequency of the sweep, the sampled path applies it across the spectrum
    (piecewise linear between `gain_knots`). For an LFM with a large time-bandwidth product
    the two agree: measured 8.5e-4 (fls_300k, BT = 60) and 4.8e-4 (flank_32 cut to 20 ms, BT = 80)."""
    dur = min(SonarModel.load(name).waveform_config["duration"], 0.02)
    recs = []
    for wave in ({"duration": dur, "window_alpha": 0.2},
                 {"type": "sampled", "source": "lfm", "duration": dur, "window_alpha": 0.2}):
        sonar = sonar_with(name, wave, r0 - 3.0, r0 + 3.0)
        recs.append(simulate_ping(sonar, make_pose(), _five_targets(sonar, r0), environment={"sound_speed": C},
                                  absorption_model="francois_garrison", adc=False).samples.array)
    err = rel_rms(*recs)
    print(f"{name}: sampled vs analytic LFM with beams and absorption: {err:.2e}")
    assert err < 2e-3


def _scenario(tier, waveform, **extra):
    cfg = {
        "name": "t", "sonar": "flank_32", "fidelity": tier, "seed": 3,
        "sonar_overrides": {"array": {"count": 4},        # four hydrophones keep the NumPy runs short
                            "transmit": {"waveform": {"duration": 0.02, "window_alpha": 0.2, **waveform}},
                            "receive": {"window": {"range": [250.0, 420.0]}, "kind": extra.pop("kind", "baseband")}},
        "environment": {"sound_speed": C, "temperature": 12.0, "salinity": 35.0, "overrides": {"sound_speed_m_s": "a nominal speed for the test geometry"}},
        "platform": {"position": [0.0, 0.0, 30.0], "velocity": [2.0, 0.0, 0.0]},
        "pings": {"count": 1},
        "scene": [{"type": "points", "positions": [[5.0, 300.0, 50.0]], "velocity": [0.0, -3.0, 0.0],
                   "target_strength_db": 10.0}],
        "receiver": {"adc": False},
    }
    cfg.update(extra)
    return cfg


SAMPLED_LFM = {"type": "sampled", "source": "lfm"}
IMAGE = {"surface_z": 0.0, "bottom_z": 100.0, "max_order": 2,
         "bottom": {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}}
TIERS = {
    "free_field": {},
    "image_source": {"propagation": IMAGE},
    "image_source + time spread": {"propagation": IMAGE,
                                   "time_spread": {"surface": {"wind_speed": 6.0}, "taps": 4}},
    "image_source + reverberation": {"propagation": IMAGE,
                                     "reverberation": [{"boundary_z": 100.0, "mu_db": -27.0},
                                                       {"boundary_z": 0.0, "model": "chapman_harris", "wind_speed": 6.0}]},
    "sediment": {"propagation": {"sediment_speed": 1650.0, "interface_z": 100.0},
                 "scene": [{"type": "points", "positions": [[5.0, 300.0, 100.5]], "target_strength_db": 10.0}]},
    "eigenray": {"propagation": {"environment": {"depth": 100.0, "sound_speed": 1500.0}, "frequency": 8000.0,
                                 "max_range": 500.0, "range_step": 25.0, "depth_step": 2.0}},
}


@pytest.mark.parametrize("label", list(TIERS))
@pytest.mark.parametrize("kind", ["baseband", "real"])
def test_sampled_lfm_matches_analytic_lfm_in_every_propagation_tier(label, kind):
    """Every tier that synthesises an LFM synthesises its sampled copy, with a moving sonar
    and a moving target, as complex baseband and as real passband records. Tiers hand the
    synthesiser two-way paths, and the reverberation generator the pulse's samples, so
    nothing in a tier depends on the pulse being analytic. Measured relative rms difference
    3.6e-4 to 3.8e-4 (echoes; the beam patterns and absorption of flank_32, see the test
    above) and 1.3e-4 with reverberation."""
    tier = label.split(" ")[0]
    if tier == "eigenray" and find_bellhop() is None:
        pytest.skip("Bellhop not installed (GPUSONAR_BELLHOP)")
    extra = dict(TIERS[label])
    ref = run_scenario(_scenario(tier, {}, kind=kind, **dict(extra))).records[0]
    rec = run_scenario(_scenario(tier, SAMPLED_LFM, kind=kind, **dict(extra))).records[0]
    assert rec.transmit.waveform.type == core.WaveformType.Arbitrary
    assert np.abs(ref.samples.array).max() > 0
    err = rel_rms(ref.samples.array, rec.samples.array)
    print(f"{label} ({kind}): sampled vs analytic LFM {err:.2e}")
    assert err < 3e-3


# --- phase-coded pulses through pulse compression ---------------------------------------------

@pytest.mark.parametrize("kind", ["baseband", "real"])
@pytest.mark.parametrize("source, n_chips, extra", [("barker", 13, {"code": 13}), ("mseq", 127, {"degree": 7})])
def test_phase_coded_pulse_compresses_to_the_expected_peak_and_sidelobes_at_the_right_range(kind, source, n_chips, extra):
    """popoto_8's band (20-35 kHz), chip rate 7.5 kHz (the band is twice the chip rate).
    Expected: a peak equal to the echo amplitude at the two-way delay, and the code's
    sidelobes: 1/13 (-22.3 dB) for Barker 13; for the m-sequence the aperiodic sidelobes of
    this 127-chip code computed from the chips themselves."""
    chip_rate = 7.5e3
    wave = {"type": "sampled", "source": source, "duration": n_chips / chip_rate, **extra}
    r0 = 100.0
    sonar = sonar_with("popoto_8", wave, r0 - 10.0, r0 + 14.0, kind, omni=True)
    target = sonar.boresight * r0 + np.array([0.3, 0.0, 0.2])
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target]), environment={"sound_speed": C},
                        absorption_model="none", adc=False)
    recon.prepare([rec])                                   # to baseband if needed, then core.pulse_compress
    s = rec.samples
    y = np.abs(s.array)
    tx, rx = np.array(rec.tx_position()), rec.rx_positions()
    chips = np.array(wfm.BARKER[13]) if source == "barker" else wfm.mseq(7)
    acf = np.abs(np.correlate(chips, chips, mode="full")).astype(float)
    ideal_psl = db(np.delete(acf, len(chips) - 1).max() / len(chips))
    for ch in (0, rx.shape[0] - 1):
        r_out, r_back = np.linalg.norm(target - tx), np.linalg.norm(target - rx[ch])
        i = int(np.argmax(y[ch]))
        a, b, c = y[ch, i - 1], y[ch, i], y[ch, i + 1]
        frac = 0.5 * (a - c) / (a - 2 * b + c)
        t_peak = s.start_time + (i + frac) / s.sample_rate
        assert t_peak == pytest.approx((r_out + r_back) / C, abs=0.02 / chip_rate)       # 1/50 of a chip
        amp = rec.transmit.waveform.amplitude                                            # echo of a 0 dB target at 1 m
        assert y[ch, i] == pytest.approx(amp / (r_out * r_back), rel=0.02)               # spreading only
        # Sidelobes: everything further than one chip from the peak.
        guard = int(np.ceil(s.sample_rate / chip_rate))
        side = np.r_[y[ch, :i - guard], y[ch, i + guard:]].max()
        psl = db(side / y[ch, i])
        print(f"{source} ({kind}) ch {ch}: peak sidelobe {psl:.1f} dB, ideal chips {ideal_psl:.1f} dB")
        assert psl == pytest.approx(ideal_psl, abs=1.5)


def test_cpp_pulse_compression_of_a_sampled_pulse_matches_the_numpy_reference():
    wave = {"type": "sampled", "source": "mseq", "degree": 6, "duration": 63 / 7.5e3}
    sonar = sonar_with("popoto_8", wave, 90.0, 110.0, omni=True)
    scene = Scene().add_points([sonar.boresight * 97.0, sonar.boresight * 103.0], amplitude=[1.0, 0.5j])
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C}, absorption_model="none", adc=False)
    for scale in (1.0, 1.004):
        ref = wfm.pulse_compress_reference(rec, doppler_scale=scale)
        out = rec.copy()
        core.pulse_compress(out, doppler_scale=scale)
        err = np.abs(out.samples.array - ref).max() / np.abs(ref).max()
        print(f"C++ pulse_compress vs NumPy reference, doppler_scale {scale}: {err:.2e}")
        assert err < 1e-4
    # A replica window is applied as for the analytic pulses, and the gain stays normalised.
    win = rec.copy()
    core.pulse_compress(win, window="hamming")
    plain = rec.copy()
    core.pulse_compress(plain)
    assert np.abs(win.samples.array).max() == pytest.approx(np.abs(plain.samples.array).max(), rel=0.05)


# --- Doppler is a time scaling ------------------------------------------------------------------

def _gamma_eq68(c, v_receiver, v_source, v_target, source, target, receiver):
    """TR 0702 eq. 68: Gamma = (1 - V_R.S_Rp) / (1 + V_S.S_Sq) x (1 - V_T.S_Tq) / (1 + V_T.S_Tp),
    with the slowness vectors written for straight paths in water of sound speed c. Leg q
    runs source -> target, leg p target -> receiver. S_Sq and S_Tp are the gradients of the
    leg's travel time with respect to the position of its start (they point back along the
    path, against the sound); S_Tq and S_Rp those with respect to its end (along the sound)."""
    def unit(a):
        return a / np.linalg.norm(a)
    q = unit(np.asarray(target, float) - np.asarray(source, float)) / c      # along the sound on the way out
    p = unit(np.asarray(receiver, float) - np.asarray(target, float)) / c    # along the sound on the way back
    return ((1 - np.dot(v_receiver, p)) / (1 + np.dot(v_source, -q))) * ((1 - np.dot(v_target, q)) / (1 + np.dot(v_target, -p)))


def test_eq68_reduces_to_the_exact_two_way_factor_for_radial_motion():
    """Monostatic, target closing at v along the line of sight: the echo of a pulse is
    compressed by exactly (c + v) / (c - v) (emission at u reaches the target at
    t_s = (u + R0/c) / (1 + v/c) and returns at t = 2 t_s - u)."""
    v = 7.0
    gamma = _gamma_eq68(C, np.zeros(3), np.zeros(3), np.array([-v, 0, 0]), [0, 0, 0], [500.0, 0, 0], [0, 0, 0])
    assert gamma == pytest.approx((C + v) / (C - v), rel=1e-12)
    assert gamma == pytest.approx(doppler_scale(-v, C), rel=1e-12)           # the convention of processing.py
    # Receiver and source both closing at v on a still target: (1 + v/c) / (1 - v/c) again.
    gamma = _gamma_eq68(C, np.array([v, 0, 0]), np.array([v, 0, 0]), np.zeros(3), [0, 0, 0], [500.0, 0, 0], [0, 0, 0])
    assert gamma == pytest.approx((1 + v / C) / (1 - v / C), rel=1e-12)


@pytest.mark.parametrize("kind", ["baseband", "real"])
@pytest.mark.parametrize("mover", ["target", "sonar"])
def test_moving_target_echo_is_the_pulse_time_scaled_by_the_closed_form_doppler_factor(kind, mover):
    """A wideband pulse (two short bursts 80 ms apart, 6-10 kHz) from a target closing at
    12 m/s, or from a still target seen by a sonar closing at 12 m/s. The echo must be
    b(Gamma (t - t0)) on a carrier at Gamma F: envelope and carrier both scaled by the
    factor of TR 0702 eq. 68. A model that only shifts the carrier leaves the bursts 80 ms
    apart; time scaling brings them 1.27 ms closer (10 samples of the envelope's bandwidth)."""
    v, r0, gap, F = 12.0, 400.0, 0.080, 8e3
    rate = 32e3
    u = np.arange(int(0.1 * rate) + 1) / rate
    burst = lambda centre: np.exp(-0.5 * ((u - centre) / 0.6e-3) ** 2)        # noqa: E731
    env = burst(0.010) + 1j * burst(0.010 + gap)
    wave = {"type": "sampled", "source": "samples", "sample_rate": rate, "f_start": 6e3, "f_stop": 10e3,
            "samples": np.column_stack([env.real, env.imag]).tolist(), "gain_knots": 0}
    sonar = sonar_with("flank_32", wave, r0 - 20.0, r0 + 20.0, kind, omni=True,
                       transmit={"lever_arm": [0.0, 0.0, 0.0]}, array={"count": 1})
    target = sonar.boresight * r0
    if mover == "target":
        scene = Scene().add_points([target], velocity=-v * sonar.boresight)
        pose, v_t, v_s = make_pose(), -v * sonar.boresight, np.zeros(3)
    else:
        scene = Scene().add_points([target])
        pose, v_t, v_s = make_pose(velocity=v * sonar.boresight), np.zeros(3), v * sonar.boresight
    rec = simulate_ping(sonar, pose, scene, environment={"sound_speed": C}, absorption_model="none",
                        adc=False, spreading=False)
    s = rec.samples
    x = s.array[0] / rec.transmit.waveform.amplitude
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    gamma = _gamma_eq68(C, v_s, v_s, v_t, np.zeros(3), target, np.zeros(3))
    assert gamma == pytest.approx((C + v) / (C - v), rel=1e-9)
    # Arrival of the pulse's start (emitted at 0): exact for constant radial speed.
    t0 = 2.0 * r0 / (C + v)
    w = sonar.sampled_waveform
    arg = gamma * (t - t0)
    scaled = w.evaluate(arg) * np.exp(2j * np.pi * F * arg)                      # time-scaled analytic echo
    shifted = w.evaluate(t - t0) * np.exp(2j * np.pi * F * arg)                  # carrier shift only
    if kind == "baseband":
        mix = np.exp(-2j * np.pi * s.center_frequency * t)
        scaled, shifted = scaled * mix, shifted * mix
    else:
        scaled, shifted = scaled.real, shifted.real
    err, err_shift = rel_rms(scaled, x), rel_rms(shifted, x)
    print(f"{mover} moving ({kind}): echo vs time-scaled pulse {err:.2e}; vs carrier-shift-only model {err_shift:.2e}")
    assert err < 1e-6                 # measured 7.3e-8 (baseband), 6.7e-8 (real)
    assert err_shift > 0.5            # the shift-only model misplaces the second burst entirely
    # The spacing of the two bursts in the echo is gap / Gamma.
    e = np.abs(x) if kind == "baseband" else np.abs(wfm.analytic_envelope(x, s.sample_rate, F))
    half = int(np.argmin(np.abs(t - (t0 + 0.05))))
    peaks = []
    for seg, offset in ((e[:half], 0), (e[half:], half)):
        i = int(np.argmax(seg))
        a, b, c = seg[i - 1], seg[i], seg[i + 1]
        peaks.append(t[0] + (offset + i + 0.5 * (a - c) / (a - 2 * b + c)) / s.sample_rate)
    assert peaks[1] - peaks[0] == pytest.approx(gap / gamma, abs=2e-6)
    assert abs((peaks[1] - peaks[0]) - gap) > 5e-4


def test_doppler_bank_finds_the_range_rate_of_a_sampled_pulse():
    """Processing side: the matched filter is built from the record's replica with the
    hypothesis' time scale, so a Doppler-sensitive code picks out the range rate."""
    v_true = -6.0                                                    # closing
    wave = {"type": "sampled", "source": "mseq", "degree": 9, "duration": 0.25}
    sonar = sonar_with("flank_32", wave, 1400.0, 1600.0, omni=True)
    target = sonar.boresight * 1500.0
    scene = Scene().add_points([target], velocity=-v_true * -sonar.boresight)
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": C}, absorption_model="none", adc=False)
    rates = np.arange(-10.0, 10.5, 1.0)
    best, v_hat, peaks = doppler_bank(rec, rates)
    assert v_hat == pytest.approx(v_true, abs=0.5)
    assert peaks[list(rates).index(0.0)] < 0.3 * peaks.max()         # the unscaled replica loses the peak
    assert peaks.max() == pytest.approx(rec.transmit.waveform.amplitude / 1500.0 ** 2, rel=0.03)


# --- engines --------------------------------------------------------------------------------------

def test_both_engines_synthesise_a_sampled_pulse_and_the_record_says_when_numpy_did():
    """The C++ kernel (host or CUDA) synthesises a sampled pulse by default; the NumPy reference
    runs when selected, and then the record says so. tests/python/test_sampled_engine.py
    compares the two on the cases of this file. Path pairing and reverberation follow
    `engine` too."""
    wave = {"type": "sampled", "source": "hfm", "duration": 0.02, "window_alpha": 0.2}
    cfg = _scenario("image_source", wave, propagation=IMAGE, reverberation=[{"boundary_z": 100.0, "mu_db": -27.0}])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = run_scenario(dict(cfg), engine="cpp").records[0]
        b = run_scenario(dict(cfg), engine="numpy").records[0]
    assert "echo_synthesis" not in json.loads(a.metadata.extra_json or "{}").get("gpusonar", {})
    assert json.loads(b.metadata.extra_json)["gpusonar"]["echo_synthesis"] == "numpy_sampled"
    scale = np.abs(a.samples.array).max()
    np.testing.assert_allclose(a.samples.array, b.samples.array, atol=5e-6 * scale)
    # Analytic pulses are untouched: no note in the record.
    plain = run_scenario(_scenario("free_field", {})).records[0]
    assert "echo_synthesis" not in json.loads(plain.metadata.extra_json or "{}").get("gpusonar", {})


def test_analytic_output_and_bad_configs_are_refused_clearly():
    with pytest.raises(ValueError, match="f_start and f_stop must differ"):
        SonarModel.load("flank_32", {"transmit": {"waveform": {"type": "sampled", "source": "lfm",
                                                                  "f_start": 8e3, "f_stop": 8e3}}})
    with pytest.raises(ValueError, match="chip rate"):
        SonarModel.load("flank_32", {"transmit": {"waveform": {"type": "sampled", "source": "barker",
                                                                  "duration": 1e-3}}})
    sonar = sonar_with("flank_32", {"type": "sampled", "source": "lfm", "duration": 0.02}, 290.0, 310.0)
    rec = sonar.make_record(make_pose())
    s = rec.samples
    s.kind = core.SampleKind.ComplexAnalytic
    rec.samples = s
    with pytest.raises(ValueError, match="analytic output is not supported"):
        wfm.synthesize_into(rec, None, [[0.0, 300.0, 0.0]], [[0.0, 300.0, 0.0]], [1.0])
