"""Channel sounding from raw recordings (sim/channel_sounding.py): every estimator on a synthetic
recording whose channel is known, and the SODAC tool on the dataset when it is there."""
import io
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from gpusonar.sim import channel_replay as cr
from gpusonar.sim import channel_sounding as cs

FS, FC, BAND = 12000.0, 6000.0, 5000.0


def chirp_period(n: int, band: float = 4500.0) -> np.ndarray:
    t = np.arange(n) / FS
    T = n / FS
    return np.exp(1j * np.pi * band / T * (t - T / 2) ** 2)


def bpsk(n_chips: int, seed: int, chip_rate: float = 4500.0) -> np.ndarray:
    """A band-limited random BPSK envelope at FS (aperiodic)."""
    rng = np.random.default_rng(seed)
    n = int(n_chips * FS / chip_rate)
    chips = rng.choice([-1.0, 1.0], n_chips)
    x = chips[np.minimum((np.arange(n) * chip_rate / FS).astype(int), n_chips - 1)]
    f = np.fft.fftfreq(n, 1 / FS)
    return np.fft.ifft(np.fft.fft(x) * (np.abs(f) < 2400.0))


def through(x: np.ndarray, paths: list, n_out: int, start: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """y[n] = sum_i a_i(n) x(n - start - d_i): delays in samples (fractional), a_i a constant or a series."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n_out, dtype=complex)
    f = np.fft.fftfreq(n_out)
    xf = np.fft.fft(np.concatenate([x, np.zeros(n_out - len(x))]))
    for d, a in paths:
        y += a * np.fft.ifft(xf * np.exp(-2j * np.pi * f * (start + d)))
    if noise > 0:
        y += np.sqrt(noise / 2) * (rng.standard_normal(n_out) + 1j * rng.standard_normal(n_out))
    return y


def test_read_wav_24_bit_with_comment_and_float():
    x = np.array([0, 1, -1, 8388607, -8388608, 12345], dtype=np.int32)
    raw = b"".join(struct.pack("<i", int(v))[:3] for v in x)
    comment = b"14-Nov-2024  16:59:54.417\0"
    info = b"INFO" + b"ICMT" + struct.pack("<I", len(comment)) + comment
    body = (b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 96000, 288000, 3, 24)
            + b"LIST" + struct.pack("<I", len(info)) + info + b"data" + struct.pack("<I", len(raw)) + raw)
    got, fs, meta = cs.read_wav(b"RIFF" + struct.pack("<I", len(body)) + body)
    assert fs == 96000.0
    np.testing.assert_allclose(got, x / 8388608.0)
    assert cs.sodac_time(meta) == pytest.approx(16 * 3600 + 59 * 60 + 54.417)
    xf = np.array([0.25, -0.5, 1.0], dtype="<f4").tobytes()
    body = b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 90000, 360000, 4, 32) + b"data" + struct.pack("<I", len(xf)) + xf
    got, fs, _ = cs.read_wav(io.BytesIO(b"RIFF" + struct.pack("<I", len(body)) + body))
    assert fs == 90000.0
    np.testing.assert_allclose(got, [0.25, -0.5, 1.0])


def test_to_baseband_keeps_amplitude_and_offset_frequency():
    fs = 96000.0
    t = np.arange(int(fs)) / fs
    x = 0.7 * np.cos(2 * np.pi * (FC + 300.0) * t + 0.4)
    b = cs.to_baseband(x, fs, FC, FS)
    mid = b[1000:-1000]
    assert np.abs(mid).mean() == pytest.approx(0.7, rel=1e-3)
    tb = np.arange(len(b))[1000:-1000] / FS
    np.testing.assert_allclose(np.angle(mid * np.exp(-1j * (2 * np.pi * 300.0 * tb + 0.4))), 0.0, atol=2e-3)
    # out of band (more than 0.5 fs_out from fc) is rejected by 60 dB or more
    far = cs.to_baseband(np.cos(2 * np.pi * (FC + 7000.0) * t), fs, FC, FS)
    assert np.abs(far[1000:-1000]).max() < 1e-3


def test_periodic_sounding_recovers_paths_and_doppler():
    per, n_per = 300, 80
    p = chirp_period(per)
    x = np.tile(p, n_per)
    n = (n_per + 4) * per
    t = np.arange(n) / FS
    fading = 0.5j * np.exp(2j * np.pi * 2.0 * t)                       # a path with 2 Hz Doppler
    y = through(x, [(0.0, 1.0), (31.0, fading), (55.4, 0.2)], n, start=600, noise=1e-4)
    k, quality = cs.find_probe(y, x)
    assert k == 600 and quality > 20
    ch = cs.sound_periodic(y, p, FS, FC, start=k + per, n_periods=n_per - 2, band=BAND, lead=0.005)
    lead = 60
    assert ch.fs_time == pytest.approx(40.0) and ch.n_delay == per
    h = ch.h[:, 0]
    np.testing.assert_allclose(h[:, lead], 1.0, atol=0.02)               # amplitude kept at the peak
    assert np.abs(h[:, lead + 31]).mean() == pytest.approx(0.5, rel=0.03)
    nu, s = cr.doppler_spectrum(h[:, lead + 31][:, None], ch.fs_time, nfft=64)
    assert nu[np.argmax(s)] == pytest.approx(2.0, abs=40.0 / 64)
    # a path between bins: the peak of the interpolated pulse is at 55.4 samples, amplitude 0.2
    seg = np.fft.ifft(np.fft.fft(h.mean(axis=0)), n=per)
    fine = np.fft.ifft(np.concatenate([np.fft.fft(seg)[:per // 2], np.zeros(9 * per), np.fft.fft(seg)[per // 2:]])) * 10
    j = np.argmax(np.abs(fine[(lead + 50) * 10:(lead + 60) * 10])) + (lead + 50) * 10
    assert j / 10 - lead == pytest.approx(55.4, abs=0.15) and abs(fine[j]) == pytest.approx(0.2, rel=0.05)
    # nothing where there is no path: the raised-cosine sidelobes and the noise only
    assert np.abs(h[:, lead + 120:lead + 200]).max() < 0.02


def test_sliding_least_squares_follows_a_fading_tap_without_the_probes_sidelobes():
    x = bpsk(4500 * 12, seed=1)                                           # 12 s, aperiodic
    n = len(x) + 6000
    t = np.arange(n) / FS
    fade = 0.4 * (1.0 + 0.5 * np.cos(2 * np.pi * 0.3 * t))                 # 0.3 Hz amplitude modulation
    y = through(x, [(0.0, 1.0), (40.0, fade), (160.0, 0.3j)], n, start=1000, noise=1e-4)
    k, _ = cs.find_probe(y, x, head=12000)
    assert k == 1000
    ch = cs.sound_sliding(y, x, FS, FC, start=k, block=2400, hop=1200, n_delay=300, band=BAND, lead=0.005)
    assert ch.fs_time == pytest.approx(10.0)
    h = ch.h[:, 0]
    lead = 60
    np.testing.assert_allclose(h[:, lead], 1.0, atol=0.02)
    np.testing.assert_allclose(h[:, lead + 160], 0.3j, atol=0.02)
    tt = (300 + 1200 * np.arange(ch.n_time) + 1200) / FS                  # block centres (first block starts at n_delay)
    want = 0.4 * (1.0 + 0.5 * np.cos(2 * np.pi * 0.3 * (tt + 1040 / FS)) * np.sinc(0.3 * 0.2))
    np.testing.assert_allclose(np.abs(h[:, lead + 40]), want, atol=0.02)
    # A matched filter over the same block would leave the probe's own sidelobes at about
    # 1 / sqrt(chips in the block) = 0.033 of the total; least squares leaves the noise only.
    quiet = np.abs(h[:, lead + 200:lead + 230])
    assert np.sqrt((quiet ** 2).mean()) < 0.005


def test_sliding_least_squares_floor_from_energy_beyond_the_window():
    """What the channel does beyond the delay window is noise for the fit: its power divided by
    (about) the number of independent probe samples in the block per tap."""
    x = bpsk(4500 * 8, seed=2)
    n = len(x) + 6000
    y = through(x, [(0.0, 1.0), (1500.0, 0.5)], n, start=1000)             # a path 125 ms late, outside a 25 ms window
    ch = cs.sound_sliding(y, x, FS, FC, start=1000, block=2400, hop=2400, n_delay=300, band=BAND, lead=0.005)
    floor = (np.abs(ch.h[:, 0, 150:290]) ** 2).mean()
    chips = 2400 * 4500.0 / FS
    assert 0.2 * 0.25 / chips < floor < 5.0 * 0.25 / chips


def test_remove_drift_finds_a_clock_rate():
    per, n_per = 3000, 40
    p = chirp_period(per)
    x = np.tile(p, n_per)
    rate = 2e-6                                                           # the arrival slides later by 2 us per second
    n = (n_per + 2) * per
    t = np.arange(n) / FS
    # y(t) = x(t - tau0 - rate t) exp(-2 pi i fc rate t): resample by interpolation of the periodic probe
    arg = (t - 0.05 - rate * t) * FS
    i0 = np.floor(arg).astype(int)
    fr = arg - i0
    xp = np.concatenate([np.zeros(2 * per, complex), x, np.zeros(4 * per, complex)])
    y = ((1 - fr) * xp[i0 + 2 * per] + fr * xp[i0 + 2 * per + 1]) * np.exp(-2j * np.pi * FC * rate * t)
    ch = cs.sound_periodic(y, p, FS, FC, start=600 + per, n_periods=n_per - 3, band=BAND, lead=0.005)
    out, info = cs.remove_drift(ch)
    assert info["rate"] == pytest.approx(rate, rel=0.02)
    assert info["doppler_hz"] == pytest.approx(-FC * rate, rel=0.02)
    ph = np.angle(out.h[:, 0, np.argmax(np.abs(out.h[0, 0]))])
    assert np.std(ph) < 0.01


@pytest.mark.parametrize("decay", [0.3e-3, 0.8e-3])
def test_tail_decay_with_the_pulse_deconvolved(decay):
    """An arrival that is 70 % steady and 30 % scattered with an exponential tail, seen through a
    0.29 ms pulse: the straight-line fit of channel_replay needs a tail several cells long, the
    pulse-deconvolving fit recovers 0.3 ms and 0.8 ms to 25 %."""
    k = 96
    win = cs.band_window(k, FS, BAND, "hann")
    pulse = np.abs(np.fft.ifft(win * np.exp(-2j * np.pi * np.fft.fftfreq(k) * 20))) ** 2      # a path at bin 20
    rng = np.random.default_rng(3)
    n_t = 4000
    lags = np.arange(0.0, 8 * decay, 1 / (4 * FS))
    w = np.exp(-lags / decay)
    w = 0.3 * w / w.sum()
    hf = np.zeros((n_t, k), complex)
    f = np.fft.fftfreq(k)
    for lag, wi in zip(lags, w):
        g = (rng.standard_normal(n_t) + 1j * rng.standard_normal(n_t)) * np.sqrt(wi / 2)
        hf += g[:, None] * np.exp(-2j * np.pi * f * (20 + lag * FS))[None, :]
    hf += np.sqrt(0.7) * np.exp(-2j * np.pi * f * 20)[None, :]
    h = np.fft.ifft(hf * win[None, :], axis=1)
    h += 1e-3 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    fluct = (np.abs(h) ** 2).mean(axis=0) - np.abs(h.mean(axis=0)) ** 2
    fit = cs.tail_decay(fluct[10:70], pulse[10:70], FS, floor=2e-6)
    assert fit["resolved"]
    assert fit["decay_time_s"] == pytest.approx(decay, rel=0.25)
    assert fit["tail_share"] > 0.8


def test_tail_decay_says_when_there_is_no_tail():
    """Fluctuation that is a pure phase or amplitude modulation of the path (a heaving mirror, a
    swaying source) has the pulse's own shape: no tail is reported."""
    k = 96
    win = cs.band_window(k, FS, BAND, "hann")
    pulse = np.abs(np.fft.ifft(win * np.exp(-2j * np.pi * np.fft.fftfreq(k) * 20))) ** 2
    rng = np.random.default_rng(4)
    noisy = 0.05 * pulse * (1.0 + 0.03 * rng.standard_normal(k)) + 1e-7
    fit = cs.tail_decay(noisy[10:70], pulse[10:70], FS, floor=1e-7)
    assert not fit["resolved"] or fit["tail_share"] < 0.1 or fit["decay_time_s"] < 0.05e-3


def test_arrival_split_coherent_fraction_and_floor():
    ch = cr.synthetic_channel([{"delay": 2e-3, "amplitude": 1.0, "incoherent": 0.25, "doppler_width": 0.4}],
                              n_time=4000, fs_time=4.0, fs_delay=FS, n_delay=64, fc=FC, bandwidth=4500.0, noise=0.01, seed=5)
    k = int(round(2e-3 * FS))
    got = cs.arrival_split(ch, 0, slice(k, k + 1), floor=0.01)
    assert got["coherent_fraction"] == pytest.approx(0.8, abs=0.03)
    assert got["total"] == pytest.approx(1.25, rel=0.05)
    assert got["coherent_fraction_se"] < 0.05
    # without the floor correction the fraction is biased low by floor / total
    raw = cs.arrival_split(ch, 0, slice(k, k + 1))
    assert raw["coherent_fraction"] < got["coherent_fraction"]
    # a slow drift of the steady part is not scattering when detrended over a window shorter than the drift
    drift = np.exp(1j * 0.8 * np.sin(2 * np.pi * np.arange(4000) / 4000.0))
    ch.h = ch.h * drift[:, None, None]
    assert cs.arrival_split(ch, 0, slice(k, k + 1), floor=0.01)["coherent_fraction"] < 0.75
    assert cs.arrival_split(ch, 0, slice(k, k + 1), floor=0.01, detrend=60.0)["coherent_fraction"] == pytest.approx(0.8, abs=0.04)


def test_load_ctd_splits_casts_and_skips_air(tmp_path):
    rows = ["Date, Time (UTC), Salinity (ppt), Temperature (C), Depth (m), Sound speed (m/s)"]
    for date, base in (("14/11/2024", 1480.0), ("15/11/2024", 1482.0)):
        rows += [f"{date},17:00:0{i},0.01,8.0,0,1440.0" for i in range(3)]                      # in air
        depths = list(np.arange(1.0, 40.0, 1.0)) + list(np.arange(39.0, 0.5, -1.0))
        for j, d in enumerate(depths):
            lag = -0.5 if j < 39 else 0.5                                                       # the way down reads low, the way up high
            rows.append(f"{date},17:01:{j:02d},30.0,10.0,{d:.2f},{base + 0.5 * d + lag:.2f}")
        rows += [f"{date},17:05:00,0.0,9.0,0,1445.0"]
    p = tmp_path / "CTD.csv"
    p.write_text("\n".join(rows) + "\n", encoding="latin-1")
    casts = cs.load_ctd(str(p))
    assert [c["date"] for c in casts] == ["14/11/2024", "15/11/2024"]
    c = casts[0]
    assert c["max_depth"] == pytest.approx(39.0)
    np.testing.assert_allclose(c["speed"], 1480.0 + 0.5 * np.interp(c["depth"], [0, 100], [0, 100]), atol=0.6)
    np.testing.assert_allclose(c["hysteresis"][1:-1], 1.0, atol=0.3)
    assert casts[1]["speed"][0] - c["speed"][0] == pytest.approx(2.0, abs=0.01)


# -- the dataset itself (skipped without it) -----------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))


def _dataset():
    d = os.environ.get("GPUSONAR_SMARTOCEAN")
    return Path(d) if d and (Path(d) / "TX-waveforms.zip").exists() else None


@pytest.mark.skipif(_dataset() is None, reason="GPUSONAR_SMARTOCEAN is not set to the SODAC files")
def test_sodac_probe_periods_are_what_the_tool_assumes():
    import channel_smartocean as tool
    for name, per in tool.PERIODIC.items():
        p = cs.load_sodac_probe(str(_dataset() / "TX-waveforms.zip"), name, tool.FC, tool.FS)
        a, b = p[per:2 * per], p[2 * per:3 * per]
        assert np.abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real) > 0.999, name


@pytest.mark.skipif(_dataset() is None or not (_dataset() / "CTD.csv").exists(), reason="no SODAC CTD.csv")
def test_sodac_ctd_has_two_casts_with_a_maximum_near_the_source_depth():
    casts = cs.load_ctd(str(_dataset() / "CTD.csv"))
    assert len(casts) == 2
    for c in casts:
        assert 12.0 < c["depth"][np.argmax(c["speed"])] < 24.0
        assert c["speed"].max() - c["speed"][0] > 10.0
