"""Passive tonal analysis (gpusonar/tonal.py, libs/live tonal.hpp): Goertzel lines and DEMON.

The hydrophone signals are made here, not by the simulator: plane waves across the array's
real element positions (delays as phase ramps over frequency), a propeller as broadband noise
amplitude-modulated at the shaft rate and its harmonics, tonals with Doppler from a moving
source. Checked: the Goertzel sum against an FFT bin; amplitude, signal to noise and frequency
of a tone in white noise against their closed forms and the Cramer-Rao bound; DEMON's shaft
rate and blade count at stated signal-to-noise ratios and its false alarms on noise alone; two
propellers at two bearings; a tonal tracked through a pass; stacked rows; the C++ analyzer
against the NumPy one; a stream cut into pieces against the whole; the program's report lines
with and without the option."""

import json
import pathlib
import subprocess

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import elevation, tonal
from gpusonar.sim import make_pose
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
EXE = pathlib.Path(__file__).resolve().parents[2] / "build" / "bin" / "sonar_live"
needs_exe = pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live is not built")


# --- signals -----------------------------------------------------------------------------------

def blank(seconds, sonar="popoto_long_range", fs=8000.0, kind="baseband", time_ns=0):
    over = {"receive": {"kind": kind, "sample_rate": fs, "window": {"range": [0.0, 0.5 * C * seconds]}},
            "transmit": {"waveform": {"duration": 0.2}}}
    return SonarModel.load(sonar, over).make_record(make_pose(), environment={"sound_speed": C}, time_ns=time_ns)


def fill(rec, x):
    s = rec.samples
    kind = s.kind
    s.array = x.astype(np.complex64 if s.is_complex() else np.float32)
    s.kind = kind
    rec.samples = s
    return rec


class Array:
    """The record's hydrophones in the array's own frame: along the rows, and up."""

    def __init__(self, rec):
        s = rec.samples
        self.n, self.fs, self.complex = s.n_samples, s.sample_rate, s.is_complex()
        self.fc = s.center_frequency if self.complex else 0.0
        pos = rec.rx_positions()
        self.channels = len(pos)
        row = elevation.find_rows(rec)[0]
        axis = pos[row[-1]] - pos[row[0]]
        axis /= np.linalg.norm(axis)
        up = np.array([0.0, 0.0, -1.0])                       # z is down
        up = up - (up @ axis) * axis
        rel = pos - pos.mean(axis=0)
        self.along, self.height = rel @ axis, rel @ (up / np.linalg.norm(up))
        self.t = np.arange(self.n) / self.fs
        self.f = np.fft.fftfreq(self.n, 1.0 / self.fs) + self.fc if self.complex else np.fft.rfftfreq(self.n, 1.0 / self.fs)

    def lead(self, bearing_deg, elevation_deg=0.0):
        """[channels] s by which a plane wave from a cone bearing, arriving from `elevation_deg` above the
        horizontal about the array's axis, reaches each hydrophone before the array's centre."""
        sb = np.sin(np.radians(bearing_deg))
        return (self.along * sb + self.height * np.sqrt(1.0 - sb * sb) * np.sin(np.radians(elevation_deg))) / C

    def noise(self, rng, rms=1.0, channels=None):
        shape = (self.channels if channels is None else channels, self.n)
        if self.complex:
            return rms * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
        return rms * rng.standard_normal(shape)

    def plane_wave(self, waveform, bearing_deg, elevation_deg=0.0):
        """One waveform (as sampled at the array's centre) on every hydrophone with its lead."""
        ramp = np.exp(2j * np.pi * self.f[None, :] * self.lead(bearing_deg, elevation_deg)[:, None])
        if self.complex:
            return np.fft.ifft(np.fft.fft(waveform)[None, :] * ramp, axis=1)
        return np.fft.irfft(np.fft.rfft(waveform)[None, :] * ramp, n=self.n, axis=1)

    def propeller(self, rng, bearing_deg, rms, shaft_hz, blades, depths=(0.15, 0.4, 0.2), elevation_deg=0.0):
        """Broadband noise of rms `rms` at a hydrophone, its amplitude modulated at the shaft rate, the
        blade rate and twice the blade rate with the depths given."""
        env = np.ones(self.n)
        for depth, harmonic, phase in zip(depths, (1, blades, 2 * blades), (0.3, 1.0, 2.0)):
            env = env + depth * np.cos(2.0 * np.pi * harmonic * shaft_hz * self.t + phase)
        return self.plane_wave(env * self.noise(rng, rms, channels=1)[0], bearing_deg, elevation_deg)

    def tone(self, frequency, amplitude, bearing_deg, phase=0.0, elevation_deg=0.0):
        """A steady tone: amplitude of the cosine (real samples) or magnitude (complex baseband)."""
        arg = 2.0 * np.pi * (frequency * (self.t[None, :] + self.lead(bearing_deg, elevation_deg)[:, None])) + phase
        if self.complex:
            return amplitude * np.exp(1j * (arg - 2.0 * np.pi * self.fc * self.t[None, :]))
        return amplitude * np.cos(arg)


def by_kind(events, kind):
    return [e for e in events if e["kind"] == kind]


def close(a, b, tolerances, path=""):
    """Two results equal: structure exactly, numbers within the tolerance of their key."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            close(a[k], b[k], tolerances, f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for x, y in zip(a, b):
            close(x, y, tolerances, path)
    elif isinstance(a, float) and b is not None:
        assert abs(a - b) <= tolerances.get(path.rsplit("/", 1)[-1], 1e-6), (path, a, b)
    else:
        assert a == b, (path, a, b)


TOLERANCES = {"frequency_hz": 1e-4, "track_hz": 1e-4, "rate_hz_s": 1e-4, "snr_db": 1e-3, "noise_db": 1e-3, "level_db": 1e-3,
              "amplitude": 1e-6, "phase_rad": 1e-4, "bearing_deg": 1e-3, "modulation": 1e-5, "shaft_hz": 1e-5, "blade_hz": 1e-4,
              "confidence": 1e-6, "threshold_db": 1e-9, "independent": 1e-9, "block_s": 1e-12, "nominal_hz": 0.0}


# --- Goertzel: the sum itself ------------------------------------------------------------------

def test_goertzel_is_the_fft_bin_and_the_fourier_sum_between_bins():
    rng = np.random.default_rng(0)
    n, fs = 512, 2000.0
    for x in (rng.standard_normal(n), rng.standard_normal(n) + 1j * rng.standard_normal(n)):
        spectrum = np.fft.fft(x)
        for k in (3, 64, 200):
            assert tonal.goertzel(x, k * fs / n, fs) == pytest.approx(spectrum[k], rel=1e-9, abs=1e-9)
        f = 301.37                                                   # not a bin
        direct = (x * np.exp(-2j * np.pi * f * np.arange(n) / fs)).sum()
        assert tonal.goertzel(x, f, fs) == pytest.approx(direct, rel=1e-9)
        assert tonal.fourier_sums(x, [f, 64 * fs / n], fs) == pytest.approx([direct, spectrum[64]], rel=1e-9)
        # the C++ recurrence takes single-precision samples
        assert core.tonal_goertzel(x.astype(np.complex64 if np.iscomplexobj(x) else np.float32), f, fs) == pytest.approx(direct, rel=1e-5)
    # a cosine of amplitude A on the frequency: |X| = A N / 2
    assert abs(tonal.goertzel(0.7 * np.cos(2 * np.pi * 250.0 * np.arange(n) / fs + 0.4), 250.0, fs)) == pytest.approx(0.7 * n / 2, rel=1e-6)


def test_goertzel_agrees_with_arlpy_where_it_is_installed():
    arl = pytest.importorskip("arlpy.signal")
    rng = np.random.default_rng(1)
    x = rng.standard_normal(256) + np.cos(2 * np.pi * 64.0 * np.arange(256) / 512.0)
    for f in (64.0, 70.3):
        assert abs(tonal.goertzel(x, f, 512.0)) == pytest.approx(arl.goertzel(f, x, 512.0), rel=1e-9)


# --- Goertzel: a tone in white noise against the closed forms ----------------------------------

def tone_record(seconds=1.0, fs=2000.0):
    rec = blank(seconds, fs=fs)
    return rec, Array(rec)


def test_tone_in_white_noise_amplitude_phase_and_snr_follow_the_closed_form():
    """A tone of magnitude A on all 32 hydrophones from 12 degrees, complex noise of variance s2 on each.
    Steered there with uniform weights and a Hann window over N samples the line's bin holds the
    tone's power A^2 and noise s2 / (32 N / 1.5): 1.5 is the Hann window's noise bandwidth in bins."""
    rec, arr = tone_record(seconds=4.0)
    amplitude, sigma, f0, phase = 0.1, 1.0, 4712.3, 0.7
    rng = np.random.default_rng(3)
    cfg = {"window": "rect", "bearings_deg": [-40.0, 40.0, 1.0],
           "goertzel": {"lines": [4712.0], "block_s": 0.5, "track": None, "noise_bins": 16}}
    n = int(0.5 * arr.fs)
    want_snr = 10 * np.log10(amplitude**2 / sigma**2 * arr.channels * n / 1.5)
    got = []
    for trial in range(6):
        fill(rec, arr.tone(f0, amplitude, 12.0, phase) + arr.noise(rng, sigma))
        for e in tonal.TonalAnalyzer(cfg).push(rec):
            got.extend(e["lines"])
            middle = e["time_ns"] * 1e-9 - 0.25                      # the phase is the tone's at the middle of the block
            want_phase = phase + 2 * np.pi * (f0 - arr.fc) * middle
            assert abs(np.angle(np.exp(1j * (e["lines"][0]["phase_rad"] - want_phase)))) < 0.15     # 29 dB: 0.024 rad rms
    assert len(got) == 48 and all(q["detected"] for q in got)
    assert np.mean([q["amplitude"] for q in got]) == pytest.approx(amplitude, rel=0.02)
    assert np.mean([q["snr_db"] for q in got]) == pytest.approx(want_snr, abs=0.5)      # 29.3 dB
    assert np.mean([q["bearing_deg"] for q in got]) == pytest.approx(12.0, abs=0.1)
    assert np.mean([q["frequency_hz"] for q in got]) == pytest.approx(f0, abs=0.01)
    # the noise in the bin, as amplitude squared: s2 x (sum w^2) / (sum w)^2 / channels
    assert np.mean([10 ** (q["noise_db"] / 10) for q in got]) == pytest.approx(sigma**2 * 1.5 / n / arr.channels, rel=0.1)
    # real samples: a cosine of amplitude A reads A as well
    real = blank(1.0, fs=16000.0, kind="real")
    ra = Array(real)
    fill(real, ra.tone(f0, 0.3, -25.0, 0.2) + ra.noise(rng, 0.5))
    line = tonal.TonalAnalyzer(dict(cfg, goertzel=dict(cfg["goertzel"], block_s=1.0))).push(real)[0]["lines"][0]
    assert line["amplitude"] == pytest.approx(0.3, rel=0.02) and line["bearing_deg"] == pytest.approx(-25.0, abs=0.1)
    assert line["frequency_hz"] == pytest.approx(f0, abs=0.02)
    assert abs(np.angle(np.exp(1j * (line["phase_rad"] - 0.2 - 2 * np.pi * f0 * 0.5)))) < 0.1


def test_frequency_estimate_is_within_a_small_factor_of_the_cramer_rao_bound():
    """One steered beam (the tone from broadside, the same on every hydrophone), 300 noise draws, the
    line anywhere between two bins. The interpolated Hann bin is not the maximum-likelihood
    estimator (the window gives away some of the block's ends), so it sits above the bound: by a
    factor under two, and the error has no bias to speak of."""
    rec, arr = tone_record(seconds=0.5)
    n = arr.n
    cfg = {"window": "rect", "bearings_deg": [0.0, 0.0, 1.0],
           "goertzel": {"lines": [4700.0], "block_s": n / arr.fs, "track": None, "search_bins": 1, "guard_bins": 2, "noise_bins": 4}}
    rng = np.random.default_rng(5)
    for amplitude in (0.05, 0.2):
        snr = amplitude**2 * arr.channels                            # of the beam, per sample (noise variance 1 per hydrophone)
        errors = []
        for _ in range(300):
            f0 = 4700.0 + rng.uniform(-1.0, 1.0)                     # bins are 2 Hz apart
            fill(rec, arr.tone(f0, amplitude, 0.0, rng.uniform(0, 2 * np.pi)) + arr.noise(rng))
            errors.append(tonal.TonalAnalyzer(cfg).push(rec)[0]["lines"][0]["frequency_hz"] - f0)
        bound = tonal.crb_frequency(snr, n, arr.fs)
        rms = float(np.sqrt(np.mean(np.square(errors))))
        assert 0.9 * bound < rms < 2.0 * bound, (amplitude, rms, bound)
        assert abs(np.mean(errors)) < 0.3 * bound
    # the bound itself, against its textbook form for a complex tone: sqrt(6 / (snr N (N^2 - 1))) fs / 2 pi
    assert tonal.crb_frequency(2.0, 100, 1000.0) == pytest.approx(np.sqrt(6.0 / (2.0 * 100 * (100**2 - 1))) * 1000.0 / (2 * np.pi))


def test_line_on_the_hydrophones_without_steering():
    rec, arr = tone_record(seconds=2.0)
    rng = np.random.default_rng(7)
    fill(rec, arr.tone(4300.4, 0.1, 35.0) + arr.noise(rng))
    cfg = {"goertzel": {"lines": [4300.0, 5500.0], "on": "hydrophones", "track": None}}
    lines = tonal.TonalAnalyzer(cfg).push(rec)[0]["lines"]
    assert lines[0]["detected"] and lines[0]["bearing_deg"] is None
    assert lines[0]["frequency_hz"] == pytest.approx(4300.4, abs=0.05)
    # channel powers are added, not the channels: the tone's A^2 = 0.01 plus the bin's noise 1.5 / N, N = 2000
    assert lines[0]["amplitude"] ** 2 == pytest.approx(0.01 + 1.5 / 2000, rel=0.05)
    assert lines[0]["snr_db"] == pytest.approx(10 * np.log10(0.01 * 2000 / 1.5), abs=1.0)
    assert not lines[1]["detected"] and lines[1]["snr_db"] < 6.0     # nothing at 5500 Hz


# --- Doppler ------------------------------------------------------------------------------------

def passing_tonal(arr, f0, amplitude, speed, cpa, t_cpa):
    """A source of a steady tone passing on a straight line parallel to the array at `cpa` metres:
    phase 2 pi f0 (t - R(t) / c), bearing from where it is. Returns the samples, the frequency
    received and the bearing (degrees) at every sample time."""
    x = speed * (arr.t - t_cpa)
    r = np.hypot(cpa, x)
    sin_b = x / r
    arg = 2 * np.pi * f0 * (arr.t - r / C)[None, :] + 2 * np.pi * f0 * arr.along[:, None] * sin_b[None, :] / C
    received = f0 * (1.0 - speed * sin_b / C)
    return amplitude * np.exp(1j * (arg - 2 * np.pi * arr.fc * arr.t[None, :])), received, np.degrees(np.arcsin(sin_b))


DOPPLER = {"bearings_deg": [-70.0, 70.0, 1.0],
           "goertzel": {"lines": [4700.0], "block_s": 0.5, "search_bins": 14, "min_snr_db": 12.0}}


def test_doppler_of_a_passing_tonal_is_tracked():
    """10 m/s past the array at 200 m: the 4700 Hz line arrives 26 Hz high, falls through the rest
    frequency at 1.57 Hz/s at the closest point, and leaves 26 Hz low; the bearing sweeps 112 degrees."""
    rec = blank(60.0, fs=2000.0)
    arr = Array(rec)
    rng = np.random.default_rng(11)
    x, received, bearing = passing_tonal(arr, 4700.0, 0.15, 10.0, 200.0, 30.0)
    fill(rec, x + arr.noise(rng))
    events = tonal.TonalAnalyzer(DOPPLER).push(rec)
    assert len(events) == 120
    errors, bearing_errors, rates = [], [], []
    for e in events:
        q = e["lines"][0]
        centre = int((e["time_ns"] * 1e-9 - 0.25) * arr.fs)          # the middle of the block
        if q["detected"]:
            errors.append(q["frequency_hz"] - received[centre])
            bearing_errors.append(q["bearing_deg"] - bearing[centre])
            if abs(arr.t[centre] - 30.0) < 3.0:
                rates.append(q["rate_hz_s"])
    assert len(errors) >= 115                                        # held from the first blocks to the last
    assert np.sqrt(np.mean(np.square(errors))) < 0.15 and np.max(np.abs(errors)) < 0.6
    assert np.sqrt(np.mean(np.square(bearing_errors))) < 0.5
    assert events[2]["lines"][0]["frequency_hz"] > 4720.0 and events[-1]["lines"][0]["frequency_hz"] < 4680.0
    assert np.mean(rates) == pytest.approx(-4700.0 * 10.0**2 / (200.0 * C), rel=0.25)     # -1.57 Hz/s
    # without the track the search stays on the nominal frequency: same detections here (the span covers the shift)
    fixed = tonal.TonalAnalyzer(dict(DOPPLER, goertzel=dict(DOPPLER["goertzel"], track=None))).push(rec)
    assert sum(e["lines"][0]["detected"] for e in fixed) >= 115 and fixed[5]["lines"][0]["track_hz"] is None
    # a narrow search (4 bins) loses the line unless it follows it
    narrow = dict(DOPPLER["goertzel"], search_bins=4, lines=[4726.0])
    followed = tonal.TonalAnalyzer(dict(DOPPLER, goertzel=narrow)).push(rec)
    stayed = tonal.TonalAnalyzer(dict(DOPPLER, goertzel=dict(narrow, track=None))).push(rec)
    assert sum(e["lines"][0]["detected"] for e in followed) >= 110
    assert sum(e["lines"][0]["detected"] for e in stayed) < 60       # only while the line crosses the 16 Hz it watches


# --- DEMON -------------------------------------------------------------------------------------

DEMON = {"bearings_deg": [-60.0, 60.0, 2.0], "demon": {"band": [4000.0, 6000.0], "segment_s": 4.0}}


def demon_scene(sources, seconds=40.0, seed=0, sonar="popoto_long_range", elevation_deg=0.0, time_ns=0):
    """[(bearing_deg, rms at a hydrophone, shaft_hz, blades)] in unit noise on every hydrophone."""
    rec = blank(seconds, sonar=sonar, fs=4000.0, time_ns=time_ns)      # 3 to 7 kHz sampled; the band analysed is 4 to 6
    arr = Array(rec)
    rng = np.random.default_rng(seed)
    x = arr.noise(rng)
    for bearing, rms, shaft, blades in sources:
        x = x + arr.propeller(rng, bearing, rms, shaft, blades, elevation_deg=elevation_deg)
    return fill(rec, x), arr


def test_demon_threshold_and_grouping_pieces():
    # one spectrum: exponential, P(x > t) = exp(-t)
    t, n = tonal.demon_threshold(1e-3, 1, 0.0)
    assert n == 1.0 and t == pytest.approx(-np.log(1e-3), rel=1e-9)
    # eight: the Gamma tail, checked by drawing
    t8, n8 = tonal.demon_threshold(1e-2, 8, 0.0)
    draws = np.random.default_rng(0).exponential(size=(400000, 8)).mean(axis=1)
    assert n8 == 8.0 and np.mean(draws > t8) == pytest.approx(1e-2, rel=0.1)
    # over a background that is itself the mean of 6 independent such bins: the ratio's tail, checked by drawing
    background = np.random.default_rng(1).exponential(size=(400000, 48)).mean(axis=1)
    assert np.mean(draws / background > 2.0) == pytest.approx(tonal.ratio_tail(8, 48.0, 2.0), rel=0.1)
    assert tonal.ratio_tail(8, 48.0, 2.0) > tonal.gamma_tail(8, 16.0)                     # heavier than with the background known
    assert tonal.ratio_tail(1, 10.0, 3.0) == pytest.approx((1 + 3.0 / 10.0) ** -10)       # one over the mean of ten
    # 35 neighbouring Hann-weighted bins count as 18 independent ones
    assert tonal.ratio_tail(8, 8 * 18.0, tonal.demon_threshold(1e-2, 8, 0.0, 35)[0]) == pytest.approx(1e-2, rel=1e-6)
    # overlapped segments count for fewer, and the threshold lies between those of the whole numbers either side
    t_half, n_half = tonal.demon_threshold(1e-2, 8, 0.25, 32)
    assert n_half == pytest.approx(8 / (1 + 2 * 0.25 * 7 / 8)) and 5 < n_half < 6
    assert tonal.demon_threshold(1e-2, 6, 0.0, 32)[0] < t_half < tonal.demon_threshold(1e-2, 5, 0.0, 32)[0]
    assert core.tonal_demon_threshold(1e-2, 8, 0.25, 32) == pytest.approx((t_half, n_half), rel=1e-12)
    d = tonal.configure({"demon": {}})["demon"]
    # shaft 3 Hz, five blades: lines at 3, 15 (strongest), 30 and a stray one
    hs = tonal.harmonic_set([15.01, 3.0, 29.98, 22.4], [30.0, 15.0, 22.0, 6.0], 0.125, d)
    assert hs["shaft_hz"] == pytest.approx(3.0, abs=0.005) and hs["blades"] == 5 and hs["blade_hz"] == pytest.approx(15.0, abs=0.02)
    assert hs["harmonics"] == [5, 1, 10, 0] and hs["confidence"] == pytest.approx(67.0 / 73.0 * 0.75)
    # only the blade rate and its second harmonic seen: a rate, but no blade count
    hs = tonal.harmonic_set([17.6, 35.2], [20.0, 12.0], 0.125, d)
    assert hs["shaft_hz"] == pytest.approx(17.6, abs=0.01) and hs["blades"] is None and hs["harmonics"] == [1, 2]
    # one line is not a rate
    hs = tonal.harmonic_set([9.0], [20.0], 0.125, d)
    assert hs["shaft_hz"] is None and hs["confidence"] == 0.0
    opts = tonal.cpp_options({"demon": {}})
    for freqs, weights in (([15.01, 3.0, 29.98, 22.4], [30.0, 15.0, 22.0, 6.0]), ([17.6, 35.2], [20.0, 12.0]), ([9.0], [20.0])):
        py = tonal.harmonic_set(freqs, weights, 0.125, d)
        shaft, blade, blades, confidence, harmonics = core.tonal_harmonic_set(freqs, weights, 0.125, opts)
        assert shaft == pytest.approx(py["shaft_hz"] or 0.0) and blades == (py["blades"] or 0) and harmonics == py["harmonics"]
        assert blade == pytest.approx(py["blade_hz"] or 0.0) and confidence == pytest.approx(py["confidence"])


@pytest.mark.parametrize("snr_db, envelope", [(-6.0, "square"), (-12.0, "square"), (-12.0, "hilbert")])
def test_demon_recovers_shaft_rate_and_blade_count(snr_db, envelope):
    """A five-bladed propeller at 3 Hz on bearing -20 degrees, its noise `snr_db` below the noise at
    each hydrophone in the band; 40 s of data. The modulation depths are 0.15 (shaft), 0.4 (blade
    rate) and 0.2 (twice the blade rate) of the amplitude."""
    rec, _ = demon_scene([(-20.0, 10 ** (snr_db / 20), 3.0, 5)], seed=2)
    cfg = dict(DEMON, demon=dict(DEMON["demon"], envelope=envelope))
    events = by_kind(tonal.TonalAnalyzer(cfg).push(rec), "demon")
    assert len(events) == 16 and events[0]["segments"] == 4 and events[-1]["segments"] == 8
    last = events[-1]
    assert len(last["sources"]) == 1
    src = last["sources"][0]
    assert src["bearing_deg"] == pytest.approx(-20.0, abs=1.0)
    assert src["shaft_hz"] == pytest.approx(3.0, abs=0.01) and src["blades"] == 5 and src["blade_hz"] == pytest.approx(15.0, abs=0.05)
    assert src["confidence"] > 0.7
    got = {q["harmonic"]: q for q in src["lines"]}
    assert {1, 5, 10} <= set(got) and got[5]["snr_db"] == max(q["snr_db"] for q in src["lines"])
    if envelope == "square" and snr_db == -6.0:
        # Square law: the power's modulation is twice the amplitude's, less the share of the beam that is
        # not the source: source 0.25 x (1 + sum depths^2 / 2) = 0.278 against noise 1.5 / 32 (Hann aperture).
        share = 0.278 / (0.278 + 1.5 / 32)
        assert got[5]["modulation"] == pytest.approx(2 * 0.4 * share / 1.11, rel=0.1)
        assert got[1]["modulation"] == pytest.approx(2 * 0.15 * share / 1.11, rel=0.2)
        # and the line's level above the background is what the range study's expression says (tonal.demon_snr_db):
        # the source 10 log10(0.278) dB against unit noise, 2 kHz of band, 4 s segments
        model = tonal.demon_snr_db(10 * np.log10(0.278), 10 * np.log10(32 / 1.5), 0.4 / 1.11, 2000.0, 4.0)
        assert got[5]["snr_db"] == pytest.approx(model, abs=1.5), (got[5]["snr_db"], model)
    f = tonal.features(src)
    assert f["demon_blades"] == 5.0 and f["demon_shaft_hz"] == src["shaft_hz"] and f["demon_lines"] == len(src["lines"])


def test_demon_false_alarms_on_noise_follow_the_threshold():
    """Noise alone. With the threshold set for 1e-3 per bearing and modulation bin, the cells of the
    normalised spectra above it are counted over every result of 60 s; and at the default 1e-6 nothing is reported."""
    rec, _ = demon_scene([], seconds=60.0, seed=4)
    cfg = dict(DEMON, demon=dict(DEMON["demon"], pfa=1e-3, gram=True, max_sources=50))
    events = by_kind(tonal.TonalAnalyzer(cfg).push(rec), "demon")
    over = cells = 0
    for e in events[::4]:                                            # every fourth: no segment shared
        gram = np.array(e["gram"]["db"])
        assert gram.shape == (61, len(e["gram"]["frequencies_hz"])) and e["gram"]["frequencies_hz"][0] == 1.0
        over += int((gram >= e["threshold_db"]).sum())
        cells += gram.size
    assert cells > 50000
    print(f"cells above the threshold: {over} of {cells} = {over / cells:.2e} for pfa 1e-3")
    assert 0.5e-3 < over / cells < 2.0e-3, over / cells
    # At the default 1e-6: 26 results x 61 bearings x 157 bins, successive results sharing seven of their eight
    # spectra. Four such records gave three false lines between them (this one: one, at 26.9 Hz, for three results),
    # each within 0.15 dB of the threshold: more than 1e-6 would give (that far out the tail is not the model's). Never a rate.
    quiet = by_kind(tonal.TonalAnalyzer(DEMON).push(rec), "demon")
    false_lines = {round(q["frequency_hz"]) for e in quiet for src in e["sources"] for q in src["lines"]}
    assert len(false_lines) <= 1 and all(src["shaft_hz"] is None for e in quiet for src in e["sources"])
    # a propeller 25 dB down at the hydrophones is not found either: no rate is made up
    weak, _ = demon_scene([(-20.0, 10 ** (-25 / 20), 3.0, 5)], seed=5)
    assert all(src["shaft_hz"] is None for e in by_kind(tonal.TonalAnalyzer(DEMON).push(weak), "demon") for src in e["sources"])


def test_two_propellers_at_two_bearings_are_separated():
    rec, _ = demon_scene([(-20.0, 0.5, 3.0, 5), (15.0, 0.4, 4.4, 4)], seed=6)
    analyzer = tonal.TonalAnalyzer(DEMON)
    last = by_kind(analyzer.push(rec), "demon")[-1]
    found = sorted((s for s in last["sources"] if s["shaft_hz"]), key=lambda s: s["bearing_deg"])
    assert len(found) == 2
    assert found[0]["bearing_deg"] == pytest.approx(-20.0, abs=1.0) and found[1]["bearing_deg"] == pytest.approx(15.0, abs=1.0)
    assert found[0]["shaft_hz"] == pytest.approx(3.0, abs=0.01) and found[0]["blades"] == 5
    assert found[1]["shaft_hz"] == pytest.approx(4.4, abs=0.01) and found[1]["blades"] == 4
    # neither source's lines are reported at the other's bearing
    assert not any(abs(q["frequency_hz"] - 17.6) < 0.2 for q in found[0]["lines"])
    assert not any(abs(q["frequency_hz"] - 15.0) < 0.2 for q in found[1]["lines"])
    # the display product: bearing x modulation frequency, each blade-rate line on its own bearing
    gram = analyzer.gram
    b = lambda deg: int(np.argmin(np.abs(gram["bearings_deg"] - deg)))                  # noqa: E731
    k = lambda hz: int(np.argmin(np.abs(gram["frequencies_hz"] - hz)))                  # noqa: E731
    assert gram["db"].shape == (61, len(gram["frequencies_hz"]))
    assert gram["db"][b(-20.0), k(15.0)] > 20.0 and gram["db"][b(15.0), k(17.6)] > 20.0
    assert gram["db"][b(15.0), k(15.0)] < gram["db"][b(-20.0), k(15.0)] - 15.0


# --- the C++ analyzer against this one ---------------------------------------------------------

MIXED = {"bearings_deg": [-60.0, 60.0, 2.0],
         "goertzel": {"lines": [4700.0, 5300.0], "block_s": 0.5},
         "demon": {"band": [4000.0, 6000.0], "segment_s": 4.0, "gram": True}}


def mixed_scene(sonar="popoto_long_range", seconds=24.0, kind="baseband", fs=4000.0, elevation_deg=0.0, time_ns=0):
    rec = blank(seconds, sonar=sonar, fs=fs, kind=kind, time_ns=time_ns)
    arr = Array(rec)
    rng = np.random.default_rng(8)
    x = arr.noise(rng) + arr.propeller(rng, -20.0, 0.5, 3.0, 5, elevation_deg=elevation_deg)
    x = x + arr.propeller(rng, 15.0, 0.4, 4.4, 4, elevation_deg=elevation_deg)
    x = x + arr.tone(4700.3, 0.2, 30.0, 0.7, elevation_deg=elevation_deg)
    return fill(rec, x), arr


def run_cpp(cfg, records, **kw):
    analyzer = core.TonalAnalyzerCpp(tonal.cpp_options(cfg, **kw))
    return [e for rec in records for e in json.loads(analyzer.push(rec))], analyzer


@pytest.mark.parametrize("kind, fs", [("baseband", 4000.0), ("real", 12800.0)])
def test_cpp_analyzer_gives_the_numpy_results(kind, fs):
    rec, _ = mixed_scene(kind=kind, fs=fs, seconds=24.0 if kind == "baseband" else 14.0)
    want = tonal.TonalAnalyzer(MIXED).push(rec)
    got, analyzer = run_cpp(MIXED, [rec])
    assert len(want) > 30 and len(by_kind(want, "demon")) >= 3
    for a, b in zip(want, got):
        ga, gb = a.pop("gram", None), b.pop("gram", None)
        if ga:
            assert np.allclose(ga["bearings_deg"], gb["bearings_deg"]) and np.allclose(ga["frequencies_hz"], gb["frequencies_hz"])
            assert np.abs(np.array(ga["db"]) - np.array(gb["db"])).max() <= 0.11       # both rounded to 0.1 dB
    close(want, got, TOLERANCES)
    bearings, freqs, db = analyzer.gram()
    assert db.shape == (61, len(freqs)) and np.degrees(bearings[0]) == pytest.approx(-60.0)
    line = [e for e in got if e["kind"] == "goertzel"][-1]["lines"]
    assert line[0]["detected"] and line[0]["frequency_hz"] == pytest.approx(4700.3, abs=0.15) and not line[1]["detected"]
    src = sorted(by_kind(got, "demon")[-1]["sources"], key=lambda s: s["bearing_deg"])
    assert [s["blades"] for s in src if s["shaft_hz"]] == [5, 4]
    # other settings of every stage, the same way
    other = {"window": "rect", "bearings_deg": [-40.0, 40.0, 4.0],
             "goertzel": {"lines": [4700.0], "on": "hydrophones", "block_s": 0.4, "hop_s": 0.25, "search_bins": 2, "guard_bins": 1,
                          "noise_bins": 4, "track": {"alpha": 0.3, "beta": 0.05, "max_misses": 1, "max_shift_hz": 3.0}},
             "demon": {"envelope": "hilbert", "envelope_rate_hz": 80.0, "window_hops": 4, "segment_s": 3.0, "segment_overlap": 0.75,
                       "n_average": 5, "min_segments": 2, "f_min_hz": 2.0, "f_max_hz": 30.0, "background_bins": 8, "guard_bins": 1,
                       "clip": 2.0, "pfa": 1e-4, "max_lines": 4, "max_sources": 2, "min_separation_deg": 8.0, "shaft_hz": [1.0, 10.0],
                       "max_harmonic": 6, "harmonic_tolerance_bins": 0.5, "score_fraction": 0.8, "blades": [3, 7]}}
    want = tonal.TonalAnalyzer(other, 4200.0, 5800.0).push(rec)       # DEMON in the passive band given
    got, _ = run_cpp(other, [rec], f_low=4200.0, f_high=5800.0)
    assert len(by_kind(want, "demon")) >= 5
    close(want, got, TOLERANCES)


@pytest.mark.parametrize("sonar, rows", [("popoto_long_range_2row", 2), ("popoto_long_range_3row", 3)])
def test_stacked_rows_give_the_same_answers_as_one_row(sonar, rows):
    """Two and three rows of 32, the sources 3 degrees above the horizontal: each row is steered on its
    own and the rows' powers added, so the elevation (unknown to this stage) costs nothing."""
    rec, arr = mixed_scene(sonar=sonar, seconds=20.0, elevation_deg=3.0)
    assert arr.channels == 32 * rows and len(set(np.round(arr.height, 3))) == rows
    analyzer = tonal.TonalAnalyzer(MIXED)
    want = analyzer.push(rec)
    assert len(analyzer.rows) == rows
    line = by_kind(want, "goertzel")[-1]["lines"][0]
    assert line["detected"] and line["frequency_hz"] == pytest.approx(4700.3, abs=0.1) and line["bearing_deg"] == pytest.approx(30.0, abs=0.3)
    assert line["amplitude"] == pytest.approx(0.2, rel=0.1)
    src = sorted((s for s in by_kind(want, "demon")[-1]["sources"] if s["shaft_hz"]), key=lambda s: s["bearing_deg"])
    assert [(round(s["shaft_hz"], 1), s["blades"]) for s in src] == [(3.0, 5), (4.4, 4)]
    assert src[0]["bearing_deg"] == pytest.approx(-20.0, abs=1.0) and src[1]["bearing_deg"] == pytest.approx(15.0, abs=1.0)
    got, _ = run_cpp(MIXED, [rec])
    for a, b in zip(want, got):
        a.pop("gram", None), b.pop("gram", None)
    close(want, got, TOLERANCES)


# --- a stream cut into pieces ------------------------------------------------------------------

def pieces(rec, lengths):
    """The record cut into consecutive records of the lengths given (cycled), each with its own time."""
    x = np.asarray(rec.samples.array)
    fs = rec.samples.sample_rate
    out, at, i = [], 0, 0
    while at < x.shape[1]:
        n = min(lengths[i % len(lengths)], x.shape[1] - at)
        part = rec.copy()
        s = part.samples
        kind = s.kind
        s.array = x[:, at:at + n].copy()
        s.kind = kind
        s.start_time = 0.0
        part.samples = s
        tx = part.transmit
        tx.time_ns = rec.transmit.time_ns + int(round(rec.samples.start_time * 1e9)) + int(round(at / fs * 1e9))
        part.transmit = tx
        out.append(part)
        at, i = at + n, i + 1
    return out


def test_stream_in_pieces_equals_the_whole():
    rec, _ = mixed_scene(seconds=20.0, time_ns=7_000_000_000)
    whole = tonal.TonalAnalyzer(MIXED).push(rec)
    parts = pieces(rec, [1664, 8000, 333, 20001])                    # of any lengths
    analyzer = tonal.TonalAnalyzer(MIXED)
    stream = [e for p in parts for e in analyzer.push(p)]
    assert len(whole) > 40 and whole[0]["time_ns"] == 7_500_000_000
    assert stream == whole                                           # the same blocks, the same arithmetic: identical
    cpp_whole, _ = run_cpp(MIXED, [rec])
    cpp_stream, _ = run_cpp(MIXED, parts)
    assert cpp_stream == cpp_whole
    # a gap drops what was half filled and nothing else: the second stretch starts afresh in time
    late = pieces(rec, [20000])
    gap_ns = 3_000_000_000
    for p in late[1:]:
        tx = p.transmit
        tx.time_ns += gap_ns
        p.transmit = tx
    analyzer = tonal.TonalAnalyzer(MIXED)
    first, rest = analyzer.push(late[0]), [e for p in late[1:] for e in analyzer.push(p)]
    assert [e["time_ns"] for e in by_kind(first, "goertzel")] == [7_500_000_000 + 500_000_000 * i for i in range(10)]
    assert by_kind(rest, "goertzel")[0]["time_ns"] == 7_000_000_000 + 5_000_000_000 + gap_ns + 500_000_000
    assert by_kind(rest, "goertzel")[0]["lines"][0]["track_hz"] is not None            # the track was kept across the gap


# --- with the passive bearings: the program's report lines -------------------------------------

# The plain beamformer's bearings (the three strongest lobes): the bearing stage is not what is tested here.
PASSIVE = {"f_low": 4000.0, "f_high": 6000.0, "n_fft": 256, "hop": 128, "n_average": 24, "first_deg": -60.0, "last_deg": 60.0,
           "method": "conventional", "max_sources": 3}
PROGRAM = ["--band", "4000", "6000", "n_fft=256", "hop=128", "n_average=24", "bearing_first_deg=-60", "bearing_last_deg=60",
           "passive_method=conventional", "sbl_max_sources=3"]
SPAN = 256 + 23 * 128                                                # samples in a passive frame
REPORT = dict(MIXED, goertzel=dict(MIXED["goertzel"], block_s=1.0), demon=dict(MIXED["demon"], gram=False))


def write(path, records):
    w = gs.RecordWriter(str(path))
    for r in records:
        w.write(r)
    w.close()
    return str(path)


def frames_of_blocks(n_samples, block):
    """The passive frame each Goertzel block of `block` samples ends in."""
    n_frames = n_samples // SPAN
    return [min((end - 1) // SPAN, n_frames - 1) for end in range(block, n_samples + 1, block)]


def test_results_go_to_the_frame_they_ended_in_and_the_nearest_bearing():
    frames = [{"bearings_deg": [-20.0, 15.0]}, {"bearings_deg": []}, {"bearings_deg": [14.0, 40.0]}]
    line = {"nominal_hz": 300.0, "detected": True, "bearing_deg": 16.5}
    events = [{"kind": "goertzel", "time_ns": 1_900_000_000, "lines": [line, dict(line, detected=False)]},
              {"kind": "goertzel", "time_ns": 2_100_000_000, "lines": [line]},
              {"kind": "goertzel", "time_ns": 2_900_000_000, "lines": [dict(line, bearing_deg=None)]},
              {"kind": "demon", "time_ns": 3_000_000_000, "sources": [{"bearing_deg": 17.5}, {"bearing_deg": 36.0}]},
              {"kind": "demon", "time_ns": 9_000_000_000, "sources": [{"bearing_deg": 41.0}]}]
    out = tonal.attach(frames, events, 1_000_000_000, [1.0, 1.5, 2.0], attach_deg=3.0)
    assert out[0]["tonal"] == {"goertzel": [{"time_ns": 1_900_000_000, "lines": [dict(line, peak=1)]}]}     # detected lines only
    assert out[1]["tonal"] == {"goertzel": [{"time_ns": 2_100_000_000, "lines": [dict(line, peak=-1)]}]}    # no bearing in that frame
    assert out[2]["tonal"]["goertzel"][0]["lines"][0]["peak"] == -1                                         # a line without a bearing
    assert [q["peak"] for q in out[2]["tonal"]["demon"][0]["sources"]] == [-1, -1]                          # 3.5 and 4 degrees off
    assert out[2]["tonal"]["demon"][1]["sources"][0]["peak"] == 1                                           # later than the last frame: in the last


def test_report_lines_carry_the_results_on_the_bearings_they_belong_to():
    rec, arr = mixed_scene(seconds=20.0)
    plain = tonal.passive_report([rec], PASSIVE)
    lines = tonal.passive_report([rec], dict(PASSIVE, tonal=REPORT))
    assert len(lines) == len(plain) == arr.n // SPAN == 25
    # nothing but the new key is different
    assert [{k: v for k, v in line.items() if k != "tonal"} for line in lines] == plain
    with_lines = [i for i, line in enumerate(lines) if "goertzel" in line.get("tonal", {})]
    want = frames_of_blocks(arr.n, 4000)                             # a block every second: 20, in 20 of the 25 frames
    assert with_lines == sorted(set(want)) and sum(len(lines[i]["tonal"]["goertzel"]) for i in with_lines) == len(want) == 20
    q = lines[with_lines[-1]]["tonal"]["goertzel"][0]["lines"]
    assert len(q) == 1 and q[0]["nominal_hz"] == 4700.0              # the 5300 Hz line was not detected: not reported
    last = lines[with_lines[-1]]                                     # the tone on the passive bearing at 30 degrees
    assert q[0]["peak"] >= 0 and last["bearings_deg"][q[0]["peak"]] == pytest.approx(30.0, abs=0.5)
    assert q[0]["bearing_deg"] == pytest.approx(30.0, abs=0.3)
    with_demon = [line for line in lines if "demon" in line.get("tonal", {})]
    assert sum(len(line["tonal"]["demon"]) for line in with_demon) == 6
    frame = with_demon[-1]
    found = [src for src in frame["tonal"]["demon"][0]["sources"] if src["shaft_hz"]]
    assert {src["blades"] for src in found} == {4, 5}
    for src in found:                                                # each propeller on its passive bearing
        assert src["peak"] >= 0 and abs(frame["bearings_deg"][src["peak"]] - src["bearing_deg"]) <= 3.0
    assert sorted(round(frame["bearings_deg"][src["peak"]]) for src in found) == [-20, 15]
    f = tonal.features(found[0], lines[with_lines[-1]]["tonal"]["goertzel"][0]["lines"])
    assert f["demon_blades"] in (4.0, 5.0) and f["tonal_lines"] == 1 and f["tonal_mean_shift_hz"] == pytest.approx(0.3, abs=0.1)
    assert tonal.features()["demon_lines"] == 0 and np.isnan(tonal.features()["demon_shaft_hz"])
    # the C++ analyzer behind the same report
    cpp = tonal.passive_report([rec], dict(PASSIVE, tonal=REPORT), engine="cpp")
    close(lines, cpp, TOLERANCES)


@needs_exe
def test_program_reports_tonal_results_and_is_unchanged_without_the_option(tmp_path):
    rec, arr = mixed_scene(seconds=20.0, time_ns=5_000_000_000)
    rec = pieces(rec, [25 * SPAN])[0]                                # 25 whole frames: in a stream nothing can follow the last one
    whole = write(tmp_path / "whole.h5", [rec])
    options = [f"{k}={v}" for k, v in tonal.cpp_options(REPORT).items()]

    def run(path, extra):
        out = subprocess.run([str(EXE), "--passive", path, *PROGRAM, *extra], capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stderr
        return out.stdout.splitlines()

    plain = run(whole, [])
    with_tonal = run(whole, options)
    assert len(plain) == len(with_tonal) == 25
    changed = []
    for i, (a, b) in enumerate(zip(plain, with_tonal)):
        if a == b:
            continue                                                 # frames without a result: the same bytes
        changed.append(i)
        at = b.index(', "tonal": ')
        assert b[:at] + "}" == a                                     # the rest of the line: the same bytes
        assert set(json.loads(b)["tonal"]) <= {"goertzel", "demon"}
    assert set(frames_of_blocks(25 * SPAN, 4000)) <= set(changed) and len(changed) < 25
    assert options[0] == "tonal=1" and run(whole, ["tonal=0"] + options[1:]) == plain          # the keys given but switched off
    # the same lines as the Python report makes (bearings by the same C++ either way; tonal by NumPy)
    want = tonal.passive_report([rec], dict(PASSIVE, tonal=REPORT))
    close(want, [json.loads(line) for line in with_tonal], dict(TOLERANCES, time_s=1e-9, noise_db=1e-6, power_db=1e-6))
    # --listen hands the program the stream one frame at a time: a file of such stretches gives the same results
    stream = write(tmp_path / "stream.h5", pieces(rec, [SPAN]))
    streamed = run(stream, options)
    tonal_of = lambda lines: [json.loads(line)["tonal"] for line in lines if '"tonal"' in line]     # noqa: E731
    assert len(streamed) == 25 and tonal_of(streamed) == tonal_of(with_tonal)
    assert [json.loads(line)["time_ns"] for line in streamed] == [json.loads(line)["time_ns"] for line in with_tonal]


def test_deployment_file_section_becomes_the_programs_arguments(tmp_path):
    """`passive: {tonal: ...}` of a deployment file, printed by tools/sonar_live.py --export-template; absent, nothing is added."""
    import shlex
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
    import sonar_live as tool
    cfg = {"source": {"type": "flank_udp", "sonar": "popoto_long_range", "port": 5700}, "chain": {}}
    plain = shlex.split(tool.export_template(cfg, str(tmp_path / "t.h5")))
    assert not any(c.startswith("tonal") for c in plain)
    cfg["passive"] = {"f_low": 4000.0, "f_high": 6000.0, "tonal": {"goertzel": {"lines": [4700.0, 5200.5], "track": None}, "demon": {}}}
    command = shlex.split(tool.export_template(cfg, str(tmp_path / "t.h5")))
    added = command[len(plain):]
    assert command[:len(plain)] == plain and added[0] == "tonal=1"
    assert "tonal_goertzel_lines=4700.0,5200.5" in added and "tonal_goertzel_track=0" in added
    assert "tonal_demon=1" in added and "tonal_demon_f_low=4000.0" in added and "tonal_demon_f_high=6000.0" in added
    core.TonalAnalyzerCpp(dict(c.split("=", 1) for c in added))      # every one of them is a key the C++ side knows
    with pytest.raises(ValueError):
        tonal.configure({"goertzel": {"lines": [300.0], "blok_s": 1.0}})                 # a misspelt key is an error, not a default
    with pytest.raises(ValueError):
        tonal.configure({"goertzel": {}})                            # Goertzel without lines to watch


def test_range_figures_follow_the_two_expressions():
    """The study's model figures (tools/tonal_range_study.py): the sonar equation solved for range."""
    sonar = SonarModel.load("popoto_long_range")
    out = tonal.detection_ranges(sonar, 50.0, tonal_levels_db=[130.0, 140.0], tonal_hz=4700.0, broadband_levels_db=[110.0, 120.0], depth=0.2)
    gain = 10 * np.log10(32 / 1.5)
    assert out["array_gain_db"] == pytest.approx(gain) and out["band_hz"] == [4000.0, 6000.0]
    from gpusonar import acoustics
    for row in out["tonal"]:
        r = row["range_m"]
        loss = 20 * np.log10(r) + float(acoustics.absorption(4700.0, 10.0, 35.0, 0.0)) * r
        # at the range given the bin holds the threshold: SL - TL - NL0 - 10 log10(1.5 / block) + AG = 10 dB
        assert row["source_level_db"] - loss - 50.0 - 10 * np.log10(1.5) + gain == pytest.approx(10.0, abs=0.01)
    assert out["tonal"][1]["range_m"] > 2 * out["tonal"][0]["range_m"]                  # 10 dB more: more than twice as far here
    for row in out["demon"]:
        r = row["range_m"]
        loss = 20 * np.log10(r) + float(acoustics.absorption(5000.0, 10.0, 35.0, 0.0)) * r
        assert tonal.demon_snr_db(row["spectrum_level_db"] - loss - 50.0, gain, 0.2, 2000.0, 8.0) == pytest.approx(row["threshold_db"], abs=0.01)
    # close in, the modulation line saturates at 1 + (2 depth)^2 B T / 6 above the background, whatever the level
    assert tonal.demon_snr_db(60.0, gain, 0.2, 2000.0, 8.0) == pytest.approx(10 * np.log10(1 + 0.16 * 2000 * 8 / 6), abs=0.01)
    assert tonal.line_snr_db(100.0, 50.0, 1.0, 13.0) == pytest.approx(100 - 50 - 1.761 + 13, abs=1e-3)


@needs_exe
def test_program_listening_to_a_board_stream_gives_the_batch_results(tmp_path):
    """--listen: untriggered blocks over UDP, cut by the program into stretches of one passive frame.
    The tonal results in its report are those of the recording it wrote, analysed in one piece."""
    import socket
    import time

    from gpusonar.sources import make_source
    from gpusonar.sources.flank_stream import FULL_SCALE_COUNTS, encode_block
    rec = blank(5.0, fs=16000.0, kind="real")
    arr = Array(rec)
    rng = np.random.default_rng(9)
    fill(rec, 0.05 * (arr.noise(rng) + arr.propeller(rng, -20.0, 0.7, 4.0, 3) + arr.tone(4700.3, 0.2, 30.0)))
    write(tmp_path / "template.h5", [rec])
    full_scale = rec.metadata.adc_full_scale
    counts = np.rint(np.asarray(rec.samples.array).T * FULL_SCALE_COUNTS / (full_scale if np.isfinite(full_scale) else 1.0)).astype(np.int32)
    assert np.abs(counts).max() < 0.5 * FULL_SCALE_COUNTS
    cfg = {"bearings_deg": [-60.0, 60.0, 2.0], "goertzel": {"lines": [4700.0], "block_s": 0.5},
           "demon": {"band": [4000.0, 6000.0], "segment_s": 1.0, "n_average": 4, "min_segments": 2}}
    n_fft, hop = 512, 256
    span = n_fft + 11 * hop
    n_frames = len(counts) // span
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    settings = [f"n_fft={n_fft}", f"hop={hop}", "n_average=12", "bearing_first_deg=-60", "bearing_last_deg=60", "passive_method=conventional"]
    proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", str(tmp_path / "template.h5"), "--listen", "--band", "4000", "6000",
                             "--recording", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), "--pings", str(n_frames),
                             *settings, *[f"{k}={v}" for k, v in tonal.cpp_options(cfg).items()]], stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i, first in enumerate(range(0, len(counts), 128)):
            tx.sendto(encode_block(counts[first:first + 128], first, -1, 0, 0, 0, 5_000_000_000), ("127.0.0.1", port))
            if i % 4 == 3:
                time.sleep(0.002)
        _, err = proc.communicate(timeout=300.0)
    finally:
        proc.kill()
    frames = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    if "0 dropped, 0 samples lost" not in err or len(frames) < n_frames:
        pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    chunks = list(make_source({"type": "container", "path": str(tmp_path / "rec.h5")}))
    assert len(chunks) == n_frames >= 24 and chunks[0].samples.array.shape[1] == span
    # the recording in one piece, by the NumPy analyzer
    whole = chunks[0].copy()
    fill(whole, np.concatenate([np.asarray(c.samples.array) for c in chunks], axis=1))
    events = tonal.TonalAnalyzer(cfg).push(whole)
    assert len(by_kind(events, "goertzel")) == n_frames * span // 8000 and len(by_kind(events, "demon")) >= 6
    want = {"goertzel": [dict({k: v for k, v in e.items() if k != "kind"}, lines=[q for q in e["lines"] if q["detected"]])
                         for e in by_kind(events, "goertzel")],
            "demon": [{k: v for k, v in e.items() if k != "kind"} for e in by_kind(events, "demon")]}
    got = {"goertzel": [], "demon": []}
    for fr in frames:
        for kind, results in fr.get("tonal", {}).items():
            for r in results:
                for item in r.get("lines", []) + r.get("sources", []):
                    item.pop("peak")
                got[kind].append(r)
    close(want, got, TOLERANCES)
    line = got["goertzel"][-1]["lines"][0]
    assert line["frequency_hz"] == pytest.approx(4700.3, abs=0.1) and line["amplitude"] == pytest.approx(0.01, rel=0.1)
    assert any(abs(q["frequency_hz"] - 12.0) < 0.3 for src in got["demon"][-1]["sources"] for q in src["lines"])   # the blade rate
