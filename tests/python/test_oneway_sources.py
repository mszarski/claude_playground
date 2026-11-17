"""Radiating sources heard one way (sim/sources.py; scenario key `sources:`).

Closed forms used: the passive sonar equation per path (received level = SL - 20 log r -
absorption), image-source arrival times, the Doppler factor of TR 0702 eq. 63, the spectrum
level of noise at the hydrophone, and Lloyd's mirror (a source and its negative image).
"""
import json
import pathlib
import subprocess

import numpy as np
import pytest
from scipy.signal import welch

import gpusonar as gs
from gpusonar import acoustics, elevation
from gpusonar.sim import Scene, make_pose, make_provider, run_scenario, simulate_ping
from gpusonar.sim import sources as src
from gpusonar.sim import timespread
from gpusonar.sonar import SonarModel

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "bin" / "sonar_live"
C = 1500.0
SL = 190.0
ENV = {"sound_speed": C, "temperature": 12.0, "salinity": 35.0}


def sonar(kind="real", fs=64000.0, window=(0.0, 600.0), count=4, **wave):
    w = dict(type="cw", f_start=8000.0, f_stop=8000.0, duration=0.004, window_alpha=0.0, source_level_db=SL)
    w.update(wave)
    return SonarModel.load("flank_32", {"array": {"count": count, "pitch": 0.094}, "transmit": {"waveform": w},
                                        "receive": {"kind": kind, "sample_rate": fs, "window": {"range": list(window)}}})


def listen(model, source_items, *, pose=None, provider=None, absorption="none", seed=1, t=0.0, **kw):
    return simulate_ping(model, pose or make_pose(), Scene(), environment=ENV, provider=provider, absorption_model=absorption,
                         adc=False, sources=src.build_sources(source_items, np.random.default_rng(seed)), transmit=False,
                         rng=np.random.default_rng(seed), scenario_time=t, **kw)


def times(rec):
    s = rec.samples
    return s.start_time + np.arange(s.n_samples) / s.sample_rate


def tone_fit(x, t, f):
    """Complex amplitude a of a real or complex tone: x = Re{a exp(2 pi i f t)} (or a exp(...))."""
    e = np.exp(-2j * np.pi * f * t)
    a = np.sum(x * e) / len(t)
    return a if np.iscomplexobj(x) else 2.0 * a


# --- pulses through the kernel ---------------------------------------------------------------

@pytest.mark.parametrize("kind,fs", [("real", 64000.0), ("baseband", 8000.0)])
def test_pulse_free_field_arrival_time_and_level_follow_the_passive_sonar_equation(kind, fs):
    model = sonar(kind, fs)
    position = np.array([300.0, 40.0, 10.0])
    for absorption in ("none", "francois_garrison"):
        rec = listen(model, [{"origin": position.tolist(), "signal": {"type": "waveform"}}], absorption=absorption)
        x, t = rec.samples.array, times(rec)
        alpha = 0.0 if absorption == "none" else float(acoustics.absorption(8000.0, 12.0, 35.0, 0.0, 8.0, absorption))
        for ch in range(x.shape[0]):
            r = np.linalg.norm(position - rec.rx_positions()[ch])
            inside = (t > r / C + 5e-4) & (t < r / C + 0.004 - 5e-4)
            before, after = t < r / C - 2.0 / fs, t > r / C + 0.004 + 2.0 / fs
            level = np.sqrt(np.mean(np.abs(x[ch, inside]) ** 2)) * (np.sqrt(2.0) if kind == "real" else 1.0)
            want = model.amplitude_at_1m() / r * 10.0 ** (-alpha * r / 20.0)             # SL - TL, in sample units
            assert 20 * np.log10(level / want) == pytest.approx(0.0, abs=0.01)
            assert np.abs(x[ch, before]).max() < 1e-3 * want and np.abs(x[ch, after]).max() < 1e-3 * want


def test_pulse_emission_times_and_repeats():
    model = sonar("baseband", 8000.0, window=(0.0, 1500.0))
    position = [450.0, 0.0, 0.0]
    travel = 450.0 / C

    def onsets(signal, t0=0.0):
        rec = listen(model, [{"origin": position, "signal": dict(signal, type="waveform")}], t=t0)
        env, t = np.abs(rec.samples.array[0]), times(rec)
        edges = np.nonzero(np.diff((env > 0.5 * env.max()).astype(int)) == 1)[0]
        return t[edges + 1]

    assert onsets({}) == pytest.approx([travel], abs=2e-4)                            # with the ping
    assert onsets({"ping_offset": 0.25}) == pytest.approx([travel + 0.25], abs=2e-4)
    assert onsets({"times": [10.1, 10.6, 30.0]}, t0=10.0) == pytest.approx([travel + 0.1, travel + 0.6], abs=2e-4)
    # A pinger every 0.3 s since long before: the emissions before the ping that are still in flight are heard too.
    got = onsets({"repeat": {"first": 0.05, "interval": 0.3}}, t0=100.0)
    k = np.arange(330, 340)
    want = 0.05 + 0.3 * k - 100.0 + travel
    assert got == pytest.approx(want[(want > 0.0) & (want < 2.0 - 0.004)], abs=2e-4)


def test_pulse_through_image_paths_matches_the_image_formulas():
    """Source 30 m deep, receiver 50 m, water 100 m, 500 m apart: arrival time sqrt(x^2 + dz^2)/c
    of each image and level (product of coefficients) / distance."""
    model = sonar("baseband", 16000.0, window=(246.0, 285.0), count=1, duration=0.002)
    provider = make_provider("image_source", C, {"surface_z": 0.0, "bottom_z": 100.0, "max_order": 2, "bottom_coefficient": 0.5})
    rec = listen(model, [{"origin": [500.0, 0.0, 30.0], "signal": {"type": "waveform"}}],
                 pose=make_pose(position=(0.0, 0.0, 50.0)), provider=provider)
    x, t = np.abs(rec.samples.array[0]), times(rec)
    images = {"direct": (30.0 - 50.0, 1.0), "surface": (-30.0 - 50.0, -1.0), "bottom": (170.0 - 50.0, 0.5),
              "surface-bottom": (230.0 - 50.0, -0.5), "bottom-surface": (-170.0 - 50.0, -0.5)}
    for name, (dz, coefficient) in images.items():
        r = np.hypot(500.0, dz)
        onset = t[np.nonzero((x > 0.5 * abs(coefficient) * model.amplitude_at_1m() / r) & (t > r / C - 1e-3) & (t < r / C + 1e-3))[0][0]]
        assert onset == pytest.approx(r / C, abs=1.0 / 16000.0), name
        plateau = x[(t > r / C + 4e-4) & (t < r / C + 0.002 - 4e-4)].mean()
        assert plateau == pytest.approx(abs(coefficient) * model.amplitude_at_1m() / r, rel=2e-3), name


@pytest.mark.parametrize("kind,fs", [("real", 64000.0), ("baseband", 8000.0)])
def test_kernel_pulse_equals_the_streamed_tone_with_source_and_receiver_moving(kind, fs):
    """Two implementations of the same one-way signal: a CW pulse through the C++ kernel and a
    tone of the same level through the NumPy streaming synthesiser, both with the source and
    the receiver moving. Inside the pulse they are the same samples (delay, level, exact
    Doppler, board clock)."""
    model = sonar(kind, fs)
    item = {"origin": [300.0, 40.0, 5.0], "velocity": [-12.0, 3.0, 0.5]}
    pose = make_pose(position=(0.0, 0.0, 20.0), velocity=(5.0, 1.0, 0.0))
    pulse = listen(model, [dict(item, signal={"type": "waveform"})], pose=pose)
    tone = listen(model, [dict(item, signal={"type": "tonals", "lines": [{"frequency": 8000.0, "level_db": SL - 10 * np.log10(2.0), "phase_deg": 0.0}]})], pose=pose)
    x, y, t = pulse.samples.array, tone.samples.array, times(pulse)
    inside = (t > 0.2025) & (t < 0.2040)                              # the pulse covers 0.2012 to 0.2052 s on every channel
    assert np.abs(y[:, inside]).max() > 0.9 * model.amplitude_at_1m() / 305.0
    assert np.abs(x[:, inside] - y[:, inside]).max() < 2e-4 * np.abs(y[:, inside]).max()


def test_sampled_pulse_runs_one_way_on_both_engines():
    """A source with a waveform of its own (a sampled FM pulse, not the sonar's CW): the C++
    kernel with the envelope table and the NumPy reference give the same record."""
    wave = {"type": "sampled", "source": "lfm", "f_start": 7000.0, "f_stop": 9000.0, "duration": 0.013, "window_alpha": 0.2, "source_level_db": SL}
    model = sonar("baseband", 8000.0)
    item = [{"origin": [300.0, 0.0, 0.0], "velocity": [-8.0, 0.0, 0.0], "signal": {"type": "waveform", "waveform": wave}}]
    a = listen(model, item, engine="cpp").samples.array
    b = listen(model, item, engine="numpy").samples.array
    assert np.abs(a).max() == pytest.approx(model.amplitude_at_1m() / 300.0, rel=0.05)
    assert np.abs(a - b).max() < 2e-3 * np.abs(a).max()
    t = times(listen(model, item))
    on = t[np.abs(a[0]) > 0.5 * np.abs(a[0]).max()]
    assert on[0] == pytest.approx(0.2, abs=1.5e-3) and on[-1] - on[0] == pytest.approx(0.013 / (1 + 8.0 / C), abs=2.5e-3)


# --- tonals ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind,fs", [("real", 32000.0), ("baseband", 4000.0)])
def test_tonal_doppler_of_a_moving_source_and_receiver_is_eq_63(kind, fs):
    """A 8 kHz line from a source closing at 15 m/s on a receiver moving at 4 m/s: the received
    frequency is f (1 - V_R.n/c) / (1 - V_S.n/c), n from source to receiver (eq. 63)."""
    model = sonar(kind, fs, window=(0.0, 1500.0), count=1)
    position, vs, vr = np.array([2000.0, 300.0, 30.0]), np.array([-15.0, 2.0, 0.0]), np.array([4.0, -1.0, 0.0])
    rec = listen(model, [{"origin": position.tolist(), "velocity": vs.tolist(),
                          "signal": {"type": "tonals", "lines": [{"frequency": 8000.0, "level_db": 150.0, "phase_deg": 20.0}]}}],
                 pose=make_pose(velocity=tuple(vr)))
    x, t = rec.samples.array[0], times(rec)
    mid = 0.5 * (t[0] + t[-1])
    # Geometry at the middle of the record: the receiver then, the source one travel time earlier.
    rx = rec.rx_positions()[0] + vr * mid
    delay = np.linalg.norm(position + vs * (mid - np.linalg.norm(position - rx) / C) - rx) / C
    n = rx - (position + vs * (mid - delay))
    want = 8000.0 * src.doppler_factor(vs, vr, n, C)
    assert want - 8000.0 > 90.0                                        # about +101 Hz
    # Frequency from the phase slope of the analytic signal over the middle of the record.
    z = x if kind == "baseband" else np.fft.ifft(np.where(np.fft.fftfreq(len(x)) > 0, 2.0, 0.0) * np.fft.fft(x))
    sel = (t > mid - 0.2) & (t < mid + 0.2)
    phase = np.unwrap(np.angle(z[sel]))
    measured = np.polyfit(t[sel] - mid, phase, 2)[1] / (2 * np.pi) + (8000.0 if kind == "baseband" else 0.0)
    assert measured == pytest.approx(want, abs=0.02)
    # First order would give f (1 + closing speed / c): the exact factor differs by 0.2 Hz here... by (v/c)^2 f.
    first_order = 8000.0 * (1.0 + (np.dot(vs, n) - np.dot(vr, n)) / (np.linalg.norm(n) * C))
    assert abs(want - first_order) > 0.3 and abs(measured - first_order) > 0.3


def test_tonal_level_per_line_with_absorption_and_harmonic_family():
    """Received level of each line = SL - 20 log r - alpha(f) r; the family of eq. 29 gives
    lines at n f1 with peak amplitudes A_n and the phases asked for."""
    model = sonar("real", 64000.0, window=(0.0, 900.0), count=1)
    r = 1000.0
    family = {"fundamental": 1500.0, "harmonics": [[1, 150.0, 0.0], [2, 144.0, 90.0], [5, 140.0, -30.0]]}
    rec = listen(model, [{"origin": [r, 0.0, 0.0], "signal": {"type": "tonals", "family": family,
                                                              "lines": [{"frequency": 20000.0, "level_db": 160.0, "phase_deg": 0.0}]}}],
                 absorption="francois_garrison")
    x, t = rec.samples.array[0].astype(float), times(rec)
    rx_cfg = model.config["receive"]
    sens = 10.0 ** ((float(rx_cfg.get("sensitivity_db") or 0.0) + float(rx_cfg.get("gain_db", 0.0))) / 20.0)
    sel = slice(0, 64000)                                              # one second: every line on a bin
    for f, peak_db, phase_deg in ((1500.0, 150.0, 0.0), (3000.0, 144.0, 90.0), (7500.0, 140.0, -30.0), (20000.0, 160.0 + 10 * np.log10(2.0), 0.0)):
        alpha = float(acoustics.absorption(f, 12.0, 35.0, 0.0, 8.0, "francois_garrison"))
        a = tone_fit(x[sel], t[sel] - r / C, f)                        # phase referred to the emission time
        want = 10.0 ** (peak_db / 20.0) * sens / r * 10.0 ** (-alpha * r / 20.0)
        assert 20 * np.log10(abs(a) / want) == pytest.approx(0.0, abs=0.01), f
        assert np.degrees(np.angle(a * np.exp(-1j * np.radians(phase_deg)))) == pytest.approx(0.0, abs=0.05), f
    assert float(acoustics.absorption(20000.0, 12.0, 35.0, 0.0, 8.0, "francois_garrison")) * r > 2.0     # the 20 kHz line lost over 2 dB


def test_tone_is_continuous_from_ping_to_ping():
    model = sonar("real", 32000.0, window=(0.0, 150.0), count=1)
    items = src.build_sources([{"origin": [60.0, 0.0, 0.0], "signal": {"type": "tonals", "lines": [{"frequency": 3000.0, "level_db": 150.0}]}}],
                              np.random.default_rng(4))
    phases = []
    for t0 in (0.0, 0.7):
        rec = simulate_ping(model, make_pose(), Scene(), environment=ENV, absorption_model="none", adc=False, sources=items,
                            transmit=False, scenario_time=t0)
        phases.append(np.angle(tone_fit(rec.samples.array[0][:3200], times(rec)[:3200] + t0, 3000.0)))
    assert np.angle(np.exp(1j * (phases[1] - phases[0]))) == pytest.approx(0.0, abs=1e-3)


# --- Lloyd's mirror ----------------------------------------------------------------------------

@pytest.mark.parametrize("kind,fs", [("real", 16000.0), ("baseband", 2000.0)])
def test_lloyds_mirror_interference_of_direct_and_surface_paths(kind, fs):
    """A tone 6 m under a pressure-release surface heard at 1 km over a column of depths: the
    amplitude is |exp(i k r1)/r1 - exp(i k r2)/r2|, lobes and nulls included."""
    f, depth_s, x_r = 2000.0, 6.0, 1000.0
    k = 2 * np.pi * f / C
    provider = make_provider("image_source", C, {"surface_z": 0.0, "bottom_z": None, "max_order": 1})
    model = sonar(kind, fs, window=(700.0, 760.0), count=1, f_start=f, f_stop=f)
    got, want = [], []
    for depth_r in np.linspace(5.0, 400.0, 28):
        rec = listen(model, [{"origin": [x_r, 0.0, depth_s], "signal": {"type": "tonals", "lines": [{"frequency": f, "level_db": 170.0, "phase_deg": 0.0}]}}],
                     pose=make_pose(position=(0.0, 0.0, depth_r)), provider=provider)
        x, t = rec.samples.array[0], times(rec)
        got.append(abs(tone_fit(x[:int(fs * 0.05)], t[:int(fs * 0.05)], f)))
        r1, r2 = np.hypot(x_r, depth_r - depth_s), np.hypot(x_r, depth_r + depth_s)
        want.append(abs(np.exp(1j * k * r1) / r1 - np.exp(1j * k * r2) / r2))
    rx_cfg = model.config["receive"]
    unit = np.sqrt(2.0) * 10.0 ** ((170.0 + float(rx_cfg.get("sensitivity_db") or 0.0) + float(rx_cfg.get("gain_db", 0.0))) / 20.0)
    got, want = np.array(got) / unit, np.array(want)
    assert want.max() > 1.9e-3 and want.min() < 0.2e-3                  # the column crosses lobes (2/r) and nulls
    assert np.abs(got - want).max() < 0.01 * want.max()


# --- broadband noise ----------------------------------------------------------------------------

@pytest.mark.parametrize("kind,fs", [("real", 32000.0), ("baseband", 4000.0)])
def test_noise_spectrum_level_at_the_hydrophone(kind, fs):
    """Spectrum level at the hydrophone = source spectrum level - 20 log r - alpha(f) r, with
    the slope asked for."""
    model = sonar(kind, fs, window=(0.0, 3000.0), count=2)
    r = 200.0
    level, slope = 130.0, -6.0
    rec = listen(model, [{"origin": [r, 0.0, 0.0], "signal": {"type": "noise", "level_db": level, "slope_db_per_octave": slope, "reference_frequency": 8000.0}}],
                 absorption="francois_garrison")
    x = rec.samples.array
    rx_cfg = model.config["receive"]
    sens = float(rx_cfg.get("sensitivity_db") or 0.0) + float(rx_cfg.get("gain_db", 0.0))
    if kind == "real":
        f, pxx = welch(x[0], fs=fs, nperseg=512)
        band = (f > 1000.0) & (f < 15000.0)
    else:
        f, pxx = welch(x[0], fs=fs, nperseg=128, return_onesided=False)
        f, pxx = f + 8000.0, pxx / 2.0                                  # the envelope carries twice the passband density
        band = (np.abs(f - 8000.0) < 1700.0) & (np.abs(f - 8000.0) > 100.0)      # welch removes the mean: not the bin at the carrier
    alpha = np.asarray(acoustics.absorption(f[band], 12.0, 35.0, 0.0, 8.0, "francois_garrison"))
    want = level + slope * np.log2(f[band] / 8000.0) - 20 * np.log10(r) - alpha * r + sens
    assert np.abs(10 * np.log10(pxx[band]) - want).max() < 1.2
    assert np.mean(10 * np.log10(pxx[band]) - want) == pytest.approx(0.0, abs=0.25)
    # The two hydrophones hear the same noise, one sample period or so apart: fully coherent.
    assert abs(np.corrcoef(np.real(x[0]), np.real(x[1]))[0, 1]) > 0.8


def test_noise_band_and_table_and_file_source(tmp_path):
    model = sonar("real", 32000.0, window=(0.0, 1500.0), count=1)
    rec = listen(model, [{"origin": [100.0, 0.0, 0.0], "signal": {"type": "noise", "band": [3000.0, 6000.0],
                                                                  "spectrum": {"frequency": [1000.0, 10000.0], "level_db": [140.0, 120.0]}}}])
    f, pxx = welch(rec.samples.array[0], fs=32000.0, nperseg=1024)
    rx_cfg = model.config["receive"]
    sens = float(rx_cfg.get("sensitivity_db") or 0.0) + float(rx_cfg.get("gain_db", 0.0))
    inside, outside = (f > 3300.0) & (f < 5700.0), ((f > 200.0) & (f < 2500.0)) | (f > 6500.0)
    want = 140.0 - 20.0 * np.log10(f[inside] / 1000.0) - 40.0 + sens       # the table is interpolated in log frequency
    assert np.abs(10 * np.log10(pxx[inside]) - want).max() < 1.5
    assert pxx[outside].max() < 1e-4 * pxx[inside].min()
    # A file: a 5 kHz burst recorded at 48 kHz, scaled to an rms source level, starting at 0.1 s.
    rate = 48000.0
    m = np.arange(int(0.05 * rate))
    burst = np.sin(2 * np.pi * 5000.0 * m / rate) * np.hanning(len(m))
    np.save(tmp_path / "burst.npy", burst)
    item = [{"origin": [150.0, 0.0, 0.0], "signal": {"type": "file", "path": str(tmp_path / "burst.npy"), "sample_rate": rate, "level_db": 160.0, "start": 0.1}}]
    for kind, fs in (("real", 32000.0), ("baseband", 8000.0)):
        model = sonar(kind, fs, window=(0.0, 400.0), count=1, f_start=5000.0, f_stop=5000.0)
        rec = listen(model, item)
        x, t = rec.samples.array[0], times(rec)
        scale = 10.0 ** ((160.0 + sens) / 20.0) / np.sqrt(np.mean(burst**2)) / 150.0
        want = np.interp(t - 0.1 - 150.0 / C, m / rate, np.hanning(len(m)), left=0.0, right=0.0) * scale
        env = np.abs(x) if kind == "baseband" else np.abs(np.fft.ifft(np.where(np.fft.fftfreq(len(x)) > 0, 2.0, 0.0) * np.fft.fft(x)))
        assert np.abs(env - want).max() < 0.02 * want.max(), kind


# --- time spreading, clocks, echoes in the same record -------------------------------------------

def test_time_spread_bounce_keeps_the_energy_and_leaves_the_direct_path_alone():
    """With `time_spread` on, the surface-reflected one-way pulse becomes a coherent part plus a
    tail of taps; the direct arrival is the same samples, and the mean energy of the bounce over
    realisations is that of the flat-surface bounce (the spread provider conserves it)."""
    model = sonar("baseband", 16000.0, window=(246.0, 268.0), count=1, duration=0.002)
    inner = make_provider("image_source", C, {"surface_z": 0.0, "bottom_z": None, "max_order": 1})
    item = [{"origin": [500.0, 0.0, 30.0], "signal": {"type": "waveform"}}]
    pose = make_pose(position=(0.0, 0.0, 50.0))
    flat = listen(model, item, pose=pose, provider=inner)
    t = times(flat)
    direct, bounce = (t > 0.3330) & (t < 0.3365), (t > 0.3370) & (t < 0.3520)
    energies = []
    for seed in range(24):
        rough = timespread.wrap(inner, {"surface": {"wind_speed": 6.0}, "taps": 12, "seed": seed}, frequency=8000.0)
        rec = listen(model, item, pose=pose, provider=rough)
        assert np.abs(rec.samples.array[0][direct] - flat.samples.array[0][direct]).max() < 1e-5 * np.abs(flat.samples.array[0]).max()
        energies.append(np.sum(np.abs(rec.samples.array[0][bounce]) ** 2))
    reference = np.sum(np.abs(flat.samples.array[0][bounce]) ** 2)
    assert np.std(energies) > 0.05 * reference                           # it fades from realisation to realisation
    assert np.mean(energies) == pytest.approx(reference, rel=0.25)


def test_board_clocks_apply_to_pulses_and_to_streamed_signals_alike():
    """Two boards, the second 0.2 ms late and 50 ppm fast. A board that samples late sees the
    same sound at an earlier sample time: the tone's phase on its channels moves by
    -2 pi f offset. The kernel (pulse) and the streaming synthesiser (tone) agree sample for
    sample on both boards."""
    def model(offset, drift):
        return SonarModel.load("flank_32", {
            "array": {"count": 4, "pitch": 0.094},
            "transmit": {"waveform": {"type": "cw", "f_start": 8000.0, "f_stop": 8000.0, "duration": 0.004, "window_alpha": 0.0, "source_level_db": SL}},
            "receive": {"kind": "real", "sample_rate": 64000.0, "window": {"range": [0.0, 400.0]},
                        "boards": [{"first": 0, "count": 2}, {"first": 2, "count": 2, "clock_offset": offset, "clock_drift_ppm": drift}]}})
    place = {"origin": [0.0, 250.0, 0.0]}
    tone = [dict(place, signal={"type": "tonals", "lines": [{"frequency": 8000.0, "level_db": SL - 10 * np.log10(2.0), "phase_deg": 0.0}]})]
    pulse = [dict(place, signal={"type": "waveform"})]
    plain, late = listen(model(0.0, 0.0), tone), listen(model(2e-4, 0.0), tone)
    assert list(late.metadata.channel_board) == [0, 0, 1, 1]
    assert np.array_equal(plain.samples.array[:2], late.samples.array[:2])
    t = times(plain)
    a, b = tone_fit(plain.samples.array[2][:6400].astype(float), t[:6400], 8000.0), tone_fit(late.samples.array[2][:6400].astype(float), t[:6400], 8000.0)
    assert np.angle(b / a * np.exp(-2j * np.pi * 8000.0 * 2e-4)) == pytest.approx(0.0, abs=1e-3)       # sampled 0.2 ms later: the phase has run on
    x, y = listen(model(2e-4, 50.0), pulse).samples.array, listen(model(2e-4, 50.0), tone).samples.array
    inside = (t > 0.1680) & (t < 0.1700)
    assert np.abs(x[:, inside] - y[:, inside]).max() < 2e-4 * np.abs(y[:, inside]).max()
    assert np.abs(y[2, inside] - listen(model(2e-4, 0.0), tone).samples.array[2, inside]).max() > 0.01 * np.abs(y[2, inside]).max()   # the drift is felt


def test_same_record_as_echoes_and_reverberation_and_default_unchanged():
    base = {"sonar": "flank_32", "environment": {"sound_speed": C}, "seed": 5, "pings": {"count": 2, "interval": 1.0},
            "sonar_overrides": {"receive": {"window": {"range": [50.0, 400.0]}}},
            "platform": {"position": [0.0, 0.0, 30.0], "velocity": [1.0, 0.0, 0.0]}, "receiver": {"adc": False},
            "scene": [{"type": "points", "positions": [[0.0, 200.0, 30.0]], "target_strength_db": 0.0}],
            "reverberation": [{"boundary_z": 80.0, "mu_db": -27.0}]}
    place = {"label": "target noise", "origin": [0.0, 200.0, 30.0], "velocity": [0.0, -3.0, 0.0]}
    tone = dict(place, signal={"type": "tonals", "lines": [{"frequency": 7000.0, "level_db": 150.0, "phase_deg": 0.0}]})
    noisy = dict(place, signal=[{"type": "noise", "level_db": 120.0}, tone["signal"]])
    passive = {"count": 2, "interval": 1.0, "transmit": False}
    plain = run_scenario(dict(base))
    again = run_scenario(dict(base, sources=[]))
    both = run_scenario(dict(base, sources=[tone]))
    alone = run_scenario(dict(base, sources=[tone], pings=passive))
    for i in range(2):
        e, s, b = plain.records[i].samples.array, alone.records[i].samples.array, both.records[i].samples.array
        assert np.array_equal(e, again.records[i].samples.array)        # no sources: nothing changes
        assert np.abs(s).max() > 0 and np.abs(e).max() > 0
        # The pinging record is the echoes and reverberation plus what the passive capture holds.
        assert np.abs(b - e - s).max() < 1e-5 * np.abs(s).max()
    # With noise as well (its own realisation each run): the same power, on top of the same echoes.
    both, alone = run_scenario(dict(base, sources=[noisy])), run_scenario(dict(base, sources=[noisy], pings=passive))
    e, s, b = plain.records[0].samples.array, alone.records[0].samples.array, both.records[0].samples.array
    f, p_s = welch(s[0], fs=plain.records[0].samples.sample_rate, nperseg=2048)
    _, p_b = welch(b[0] - e[0], fs=plain.records[0].samples.sample_rate, nperseg=2048)
    band = (f > 1000.0) & (f < 6000.0)
    assert np.mean(p_b[band]) == pytest.approx(np.mean(p_s[band]), rel=0.15)
    assert alone.records[0].samples.array.shape == plain.records[0].samples.array.shape
    with pytest.raises(ValueError, match="unknown source signal type"):
        run_scenario(dict(base, sources=[{"origin": [0, 0, 0], "signal": {"type": "chirps"}}]))


# --- the passive chain -------------------------------------------------------------------------

def two_rows(seconds=2.0):
    return SonarModel.load("popoto_long_range_2row", {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [0.0, 0.5 * C * seconds]}}})


def passive_capture(provider=None):
    model = two_rows()
    places = ([-400.0, 900.0, 20.0], [300.0, 800.0, 250.0])
    items = [{"origin": p, "signal": {"type": "noise", "level_db": level}} for p, level in zip(places, (118.0, 115.0))]
    pose = make_pose(position=(0.0, 0.0, 100.0), rpy=(np.radians(3.0), 0.0, 0.0))
    rec = simulate_ping(model, pose, Scene(), environment=ENV, absorption_model="none", adc=False, provider=provider,
                        sources=src.build_sources(items), transmit=False, rng=np.random.default_rng(2),
                        noise_spectrum=lambda f: np.full_like(np.asarray(f, dtype=float), 50.0))
    return rec, places


def truth(record, rows, target):
    frame = elevation.array_frame(record, rows)
    d = np.asarray(target, dtype=float) - frame.origin
    u = d / np.linalg.norm(d)
    return float(np.arcsin(u @ frame.along)), float(np.arcsin(u @ frame.up)), float(np.arcsin(-u[2]))


def test_bearing_and_elevation_of_noise_sources_through_the_passive_chain_python_and_cpp():
    rec, places = passive_capture()
    rows = elevation.find_rows(rec)
    frames, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, first_deg=-60.0, last_deg=60.0,
                                 max_sources=2, min_snr_db=3.0, elevation=True)
    assert len(frames) >= 5
    for f in frames:
        found = sorted(zip((np.degrees(p[0]) for p in f["peaks"]), np.degrees(f["elevations"])))
        assert len(found) == 2
        for (bearing, level), place in zip(found, places):
            cone, _, level_true = truth(rec, rows, place)
            assert bearing == pytest.approx(np.degrees(cone), abs=0.6) and level == pytest.approx(np.degrees(level_true), abs=1.5)
    for place in places:
        cone, el_true, _ = truth(rec, rows, place)
        ref, q_ref = elevation.passive_elevation(rec, rows, cone, 4000.0, 6000.0, n_fft=256, hop=128)        # Python
        got, q = core.live_passive_elevation(rec, rows, cone, 4000.0, 6000.0, n_fft=256, hop=128)            # C++
        assert got == pytest.approx(ref, abs=1e-5) and q == pytest.approx(q_ref, abs=1e-4)
        assert np.degrees(got - el_true) == pytest.approx(0.0, abs=0.5)


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")
def test_program_listens_to_a_scenario_record(tmp_path):
    rec, places = passive_capture()
    w = gs.RecordWriter(str(tmp_path / "passive.h5"))
    w.write(rec)
    w.close()
    out = subprocess.run([str(EXE), "--passive", str(tmp_path / "passive.h5"), "--band", "4000", "6000", "n_fft=256", "hop=128",
                          "n_average=12", "sbl_max_sources=2", "elevation=1"], capture_output=True, text=True, check=True)
    frames = [json.loads(line) for line in out.stdout.splitlines()]
    rows = elevation.find_rows(rec)
    want = sorted(np.degrees(truth(rec, rows, p)[0]) for p in places)
    assert len(frames) >= 5 and all(sorted(f["bearings_deg"]) == pytest.approx(want, abs=0.6) for f in frames)
