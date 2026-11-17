"""Statistical properties of the reverberation generator (sim/reverb.py, libs/sim/src/reverb.cpp).

1. K-distributed reverberation: a gamma texture of unit mean per resolution cell on top of
   the Gaussian speckle. Checked against the closed-form intensity moments of that compound
   (K) distribution, the Rayleigh limit and the unchanged mean power.
2. The spread-against-coherence results of TR 0702 Sec. 10.6, which any reverberation
   generator should satisfy (docs/PHYSICS_NOTES.md Sec. 5.1 and 9): a Gaussian spread in time
   gives a Gaussian coherence in frequency of width 1 / (2 pi sigma_T) (eq. 89), in Doppler
   a coherence in time of width 1 / (2 pi f sigma_gamma) (eq. 96), in direction a coherence
   along the array of width lambda / (2 pi sigma_theta) (eq. 103).
3. Replica correlation gains 3 dB of signal-to-reverberation ratio per octave of bandwidth
   (TR 0702 p. 70).

The closed forms are checked on both engines where the generator itself is what is tested.
"""

import numpy as np
import pytest

from gpusonar import _gpusonar as core
from gpusonar.sim import FreeFieldProvider, Scene, make_pose, reverb, simulate_ping
from gpusonar.sim import engine as sim_engine
from gpusonar.sonar import SonarModel

C = 1500.0
ENGINES = ["numpy"] + (["cpp"] if sim_engine.HAVE_CPP else [])


def generate(engine, rng, density, coarse, *, n_low, fs, n_azimuth, fc=10e3, pulse=(1.0,), offsets=((0.0, 0.0, 0.0),),
             velocity=(0.0, 0.0, 0.0), components=None, texture=None):
    """The generator alone: the series [channels, n_low] of a given power density
    [rows, len(coarse)] (per radian and per second), at the rate `fs`, no decimation."""
    offsets = np.asarray(offsets, dtype=float)
    pulse = np.asarray(pulse, dtype=complex)
    density = np.asarray(density, dtype=float)
    if engine == "cpp":
        return reverb._series_cpp(rng, coarse, density, False, pulse, offsets, fs, n_low, 0.0, 0.0, fc, C, n_azimuth, 1,
                                  n_low, velocity, False, False, core.Backend.Host, components, texture)
    return reverb._series_numpy(rng, coarse, density, pulse, offsets, np.zeros(3), fs, n_low, 0.0,
                                np.array([0.0, 1.0 / fs]), fc, C, n_azimuth, 1, n_low, np.asarray(velocity), False,
                                False, components, texture)


def flat(n_rows, level=1.0):
    return np.array([-1.0, 1e6]), np.full((n_rows, 2), level)


# --- K-distributed reverberation ---------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("shape", [0.5, 2.0, 10.0])
def test_k_distributed_intensity_moments(engine, shape):
    """One cell per sample: intensity = gamma texture (unit mean, shape nu) x exponential
    speckle. Its moments are those of the K distribution:

        E{I^2} / E{I}^2 = 2 (1 + 1/nu),     E{I^3} / E{I}^3 = 6 (1 + 1/nu) (1 + 2/nu)

    (moments of a gamma variate, E{tau^n} = prod_{k<n} (1 + k/nu), times n! for the
    exponential), and the mean power is the Rayleigh one."""
    n, fs = 400_000, 1000.0
    coarse, density = flat(1, level=fs / (2 * np.pi))             # power 1 per sample: sigma^2 = 1/2
    rng = np.random.default_rng(int(10 * shape))
    texture = reverb.CellTexture(rng, shape, n, 1, [0])
    x = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=1, texture=texture)[0]
    inten = np.abs(x) ** 2
    mean = inten.mean()
    assert mean == pytest.approx(1.0, rel=0.02 if shape > 1 else 0.03)       # unit-mean texture: power unchanged
    m2, m3 = (inten**2).mean() / mean**2, (inten**3).mean() / mean**3
    assert m2 == pytest.approx(2.0 * (1.0 + 1.0 / shape), rel=0.06)
    if shape >= 2.0:
        assert m3 == pytest.approx(6.0 * (1.0 + 1.0 / shape) * (1.0 + 2.0 / shape), rel=0.15)
    assert m2 > 2.15                                                         # heavier tailed than Rayleigh


@pytest.mark.parametrize("engine", ENGINES)
def test_large_shape_recovers_rayleigh(engine):
    n, fs = 200_000, 1000.0
    coarse, density = flat(1, level=fs / (2 * np.pi))
    out = {}
    for shape in (1e7, None):
        rng = np.random.default_rng(2)
        texture = None if shape is None else reverb.CellTexture(rng, shape, n, 1, [0])
        if shape is None:
            rng.integers(0, 2**63 - 1)                                       # the draw that seeds a texture
        out[shape] = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=1, texture=texture)[0]
    inten = np.abs(out[1e7]) ** 2
    assert (inten**2).mean() / inten.mean() ** 2 == pytest.approx(2.0, rel=0.02)
    # Sample for sample the Gaussian sequence, to the 1 / sqrt(shape) of the texture.
    assert np.abs(out[1e7] - out[None]).max() < 5e-3 * np.abs(out[None]).max()
    # Rayleigh envelope: mean / rms = sqrt(pi) / 2.
    assert np.abs(out[1e7]).mean() / np.sqrt(inten.mean()) == pytest.approx(np.sqrt(np.pi) / 2, rel=0.01)


@pytest.mark.parametrize("engine", ENGINES)
def test_texture_is_shared_within_a_cell_and_independent_between_cells(engine):
    """Cells in time: samples of one cell share the texture, so their intensities are
    correlated (E{I_a I_b} / E{I}^2 = 1 + 1/nu), and samples of different cells are not.
    Cells in azimuth: a hydrophone that sums N cells of equal power sees
    E{I^2} / E{I}^2 = 2 (1 + 1 / (N nu)), on its way to Rayleigh."""
    shape, n, fs, cell = 1.5, 400_000, 1000.0, 50
    coarse, density = flat(1, level=fs / (2 * np.pi))
    rng = np.random.default_rng(8)
    x = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=1,
                 texture=reverb.CellTexture(rng, shape, n, cell, [0]))[0]
    inten = (np.abs(x) ** 2).reshape(-1, cell)
    mean = inten.mean()
    same = (inten[:, :-1] * inten[:, 1:]).mean() / mean**2          # neighbours within a cell
    other = (inten[:-1] * inten[1:]).mean() / mean**2               # the same place in the next cell
    assert same == pytest.approx(1.0 + 1.0 / shape, rel=0.05)
    assert other == pytest.approx(1.0, rel=0.03)
    # Eight sectors at one hydrophone, grouped into 8, 2 and 1 azimuth cells.
    n = 200_000
    coarse, density = flat(8, level=fs / (2 * np.pi))
    for per_cell, cells in ((1, 8), (4, 2), (8, 1)):
        rng = np.random.default_rng(9)
        texture = reverb.CellTexture(rng, shape, n, 1, np.arange(8) // per_cell)
        y = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=8, texture=texture)[0]
        inten = np.abs(y) ** 2
        assert (inten**2).mean() / inten.mean() ** 2 == pytest.approx(2.0 * (1.0 + 1.0 / (cells * shape)), rel=0.05)


@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
def test_texture_is_the_same_on_both_engines():
    n, fs = 4096, 1000.0
    coarse, density = flat(6, level=3.0)
    out = []
    for engine in ("numpy", "cpp"):
        rng = np.random.default_rng(5)
        texture = reverb.CellTexture(rng, 0.8, n, 16, np.arange(6) // 2)
        out.append((generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=6, texture=texture,
                             offsets=[[0, 0, 0], [0.05, 0, 0], [0.1, 0.02, 0]]), rng.bit_generator.state))
    assert np.abs(out[0][0] - out[1][0]).max() < 5e-6 * np.abs(out[0][0]).max()
    assert out[0][1] == out[1][1]


def empty_ping(ranges=(25.0, 230.0), fs=20e3, waveform=None, name="popoto_8"):
    wf = {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3, "duration": 2e-3, "window_alpha": 0.0,
          "source_level_db": None, "amplitude": 1.0}
    wf.update(waveform or {})
    sonar = SonarModel.load(name, {
        "receive": {"kind": "baseband", "sample_rate": fs, "window": {"range": list(ranges)}, "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": wf}})
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, -20.0)), Scene(), environment={"sound_speed": C},
                         adc=False)


def test_k_statistics_in_a_ping():
    """Through `add_reverberation`: one azimuth cell, cells of 40 ms (twenty pulse lengths).
    The intensity, with the range trend taken out, has the K moment (a little below it:
    a sample whose pulse straddles two cells averages their textures); the default is
    Rayleigh; the mean level is the same. (`band_taper`: the level falls by 59 dB over this
    record, and without it the generator's leakage from the start raises the last samples.)"""
    stats = {"model": "k", "shape": 1.0, "cell_time": 0.04}
    moment, level = {}, {}
    for label, statistics in (("k", stats), ("rayleigh", None)):
        norm = []
        for seed in range(12):
            rec = empty_ping(ranges=(100.0, 1500.0))
            power = reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(seed),
                                             n_azimuth=1, statistics=statistics, band_taper=True)
            norm.append(np.abs(rec.samples.array[0]) ** 2 / power)
        norm = np.concatenate(norm)
        moment[label] = (norm**2).mean() / norm.mean() ** 2
        level[label] = norm.mean()
    assert moment["rayleigh"] == pytest.approx(2.0, rel=0.05)
    assert 3.2 < moment["k"] < 4.4                                   # 2 (1 + 1/nu) = 4
    assert level["k"] == pytest.approx(level["rayleigh"], rel=0.1)


def test_statistics_options():
    def run(statistics, **kw):
        rec = empty_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(1), n_azimuth=12,
                                 statistics=statistics, **kw)
        return rec.samples.array

    plain = run(None)
    assert np.array_equal(run({"model": "rayleigh"}), plain)
    assert np.array_equal(run({"model": "k", "shape": float("inf")}), plain)      # no texture, nothing drawn
    assert not np.array_equal(run({"model": "k", "shape": 1.0}), plain)
    if sim_engine.HAVE_CPP:
        a, b = run({"model": "k", "shape": 1.0}, engine="cpp"), run({"model": "k", "shape": 1.0}, engine="numpy")
        assert np.abs(a - b).max() < 5e-6 * np.abs(b).max()
    with pytest.raises(ValueError, match="shape"):
        run({"model": "k"})
    with pytest.raises(ValueError, match="positive"):
        run({"model": "k", "shape": 0.0})
    with pytest.raises(ValueError, match="statistics"):
        run({"model": "weibull"})


# --- spread against coherence (TR 0702 Sec. 10.6) ----------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_time_spread_gives_the_coherence_bandwidth(engine):
    """Eq. (89): scattering strength Gaussian in two-way time with RMS width sigma_T gives a
    frequency coherence exp(-df^2 / (2 sigma_f^2)) with sigma_f = 1 / (2 pi sigma_T)."""
    fs, n, sigma_t = 1000.0, 1024, 0.02
    coarse = np.arange(0.0, n / fs, 1e-3)
    density = np.exp(-0.5 * ((coarse - 0.5) / sigma_t) ** 2)[None, :]
    rng = np.random.default_rng(1)
    spectra = np.array([np.fft.fft(generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=1)[0])
                        for _ in range(200)])
    lags = np.arange(0, 21)
    coherence = np.array([np.abs(np.mean(spectra * np.conj(np.roll(spectra, -k, axis=1)))) for k in lags])
    coherence /= coherence[0]
    sigma_f = 1.0 / (2.0 * np.pi * sigma_t)                        # 7.96 Hz
    expected = np.exp(-0.5 * (lags * fs / n / sigma_f) ** 2)
    assert np.abs(coherence - expected).max() < 0.03, coherence - expected
    # The width itself, from the lag where the coherence falls to exp(-1/2).
    width = np.interp(np.exp(-0.5), coherence[::-1], (lags * fs / n)[::-1])
    assert width == pytest.approx(sigma_f, rel=0.05)


@pytest.mark.parametrize("engine", ENGINES)
def test_doppler_spread_gives_the_coherence_time(engine):
    """Eq. (96): scattering strength Gaussian in Doppler with RMS width sigma_gamma gives,
    for a tone of frequency f, a time coherence of RMS width 1 / (2 pi f sigma_gamma).

    Here the Doppler spread is the generator's own: a sonar moving at v along x and
    reverberation arriving in a Gaussian fan of azimuths (RMS sigma_phi) about broadside,
    so gamma = 1 + (2 v / c) cos(phi) has sigma_gamma = 2 v sigma_phi / c. The tone has a
    finite length T, which multiplies the coherence by its own autocorrelation 1 - |s| / T."""
    fs, n, fc, v, sigma_phi, n_az, tone = 400.0, 4096, 10e3, 3.0, 0.1, 360, 0.5
    phi = (np.arange(n_az) + 0.5) * 2 * np.pi / n_az
    weight = np.exp(-0.5 * ((phi - 0.5 * np.pi) / sigma_phi) ** 2)
    weight[weight < 1e-8] = 0.0
    coarse = np.array([-1.0, 1e6])
    density = weight[:, None] * np.ones((1, 2))
    pulse = np.ones(int(tone * fs))
    lags = np.arange(0, 41)
    rng = np.random.default_rng(3)
    cov = 0.0
    for _ in range(12):
        x = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=n_az, fc=fc, pulse=pulse,
                     velocity=(v, 0.0, 0.0))[0]
        cov = cov + np.array([np.mean(x * np.conj(np.roll(x, -k))) for k in lags])
    coherence = cov / cov[0].real
    sigma_gamma = 2.0 * v * sigma_phi / C
    sigma_time = 1.0 / (2.0 * np.pi * fc * sigma_gamma)            # 39.8 ms
    s = lags / fs
    expected = np.exp(-0.5 * (s / sigma_time) ** 2) * (1.0 - s / tone)
    assert np.abs(coherence - expected).max() < 0.06, np.abs(coherence - expected).max()
    measured = np.interp(np.exp(-0.5), (np.abs(coherence) / (1.0 - s / tone))[::-1], s[::-1])
    assert measured == pytest.approx(sigma_time, rel=0.1)


@pytest.mark.parametrize("engine", ENGINES)
def test_direction_spread_gives_the_coherence_length(engine):
    """Eq. (103): arrivals Gaussian in sin(theta) about broadside with RMS width
    sigma_theta give a coherence between hydrophones d apart of exp(-d^2 / (2 sigma_d^2)),
    sigma_d = lambda / (2 pi sigma_theta)."""
    fs, n, fc, n_az, sigma_theta = 1000.0, 4096, 10e3, 720, 0.08
    phi = (np.arange(n_az) + 0.5) * 2 * np.pi / n_az
    weight = np.where(np.sin(phi) > 0.0, np.exp(-0.5 * (np.cos(phi) / sigma_theta) ** 2), 0.0)   # sin(theta) = cos(phi)
    weight[weight < 1e-8] = 0.0
    coarse = np.array([-1.0, 1e6])
    density = weight[:, None] * np.ones((1, 2))
    spacing, channels = 0.05, 12
    offsets = np.column_stack([np.arange(channels) * spacing, np.zeros(channels), np.zeros(channels)])
    rng = np.random.default_rng(4)
    num, den = np.zeros(channels, dtype=complex), 0.0
    for _ in range(4):
        x = generate(engine, rng, density, coarse, n_low=n, fs=fs, n_azimuth=n_az, fc=fc, offsets=offsets)
        den += np.mean(np.abs(x) ** 2)
        for sep in range(channels):
            num[sep] += np.mean(x[: channels - sep] * np.conj(x[sep:]))
    coherence = num / den
    d = np.arange(channels) * spacing
    sigma_d = (C / fc) / (2.0 * np.pi * sigma_theta)               # 0.298 m
    expected = np.exp(-0.5 * (d / sigma_d) ** 2)
    assert np.abs(coherence - expected).max() < 0.05, np.abs(coherence - expected).max()
    measured = np.interp(np.exp(-0.5), np.abs(coherence)[::-1], d[::-1])
    assert measured == pytest.approx(sigma_d, rel=0.08)


# --- replica correlation -----------------------------------------------------------------------

def test_replica_correlation_gains_3_db_per_octave_of_bandwidth():
    """Pulses of one length and energy, bandwidth doubling: the matched filter's peak on an
    echo is the pulse energy whatever the bandwidth, and its output on reverberation falls
    by half per octave (the reverberation is white over the band, the filter passes a
    resolution cell of 1 / B)."""
    fc, duration = 27.5e3, 0.02
    out_power, energy = {}, {}
    for band in (500.0, 1000.0, 2000.0, 4000.0):
        wf = {"type": "lfm", "f_start": fc - 0.5 * band, "f_stop": fc + 0.5 * band, "duration": duration}
        total = 0.0
        for seed in range(4):
            rec = empty_ping(ranges=(60.0, 400.0), waveform=wf)
            reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(seed))
            s = rec.samples
            replica = reverb.baseband_pulse(rec.transmit.waveform, s.sample_rate, s.center_frequency)
            compressed = np.array([np.convolve(row, np.conj(replica[::-1]), mode="valid") for row in s.array])
            total += np.mean(np.abs(compressed) ** 2)
        out_power[band] = total / 4
        energy[band] = float(np.sum(np.abs(replica) ** 2))
    bands = sorted(out_power)
    assert max(energy.values()) / min(energy.values()) == pytest.approx(1.0, abs=1e-9)
    # Signal-to-reverberation ratio after the filter, relative to the narrowest pulse.
    gain = np.array([10 * np.log10(out_power[bands[0]] / out_power[b]) for b in bands])
    steps = np.diff(gain)
    assert np.abs(steps - 10 * np.log10(2.0)).max() < 0.5, steps
    assert gain[-1] == pytest.approx(30 * np.log10(2.0), abs=0.6)   # three octaves: 9 dB


# --- leakage along a long record ---------------------------------------------------------------

def test_band_taper_keeps_the_far_end_of_a_long_record_at_its_own_level():
    """Over 30 m to 3 km the reverberation falls by 100 dB. With `band_taper` the generated
    level follows the expected power to the end of the record. (Without it the ideal delays
    and interpolation of the generator leak the strong start into the far end: measured
    7 dB too high over 50-75 % of this record and 25 dB over 90-97 %, on both engines.)"""
    excess = {}
    for taper in (True, False):
        ratio = 0.0
        for seed in range(4):
            rec = empty_ping(ranges=(30.0, 3000.0))
            power = reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(seed),
                                             band_taper=taper)
            ratio = ratio + (np.abs(rec.samples.array) ** 2).mean(axis=0) / power / 4
        n = len(ratio)
        assert 10 * np.log10(power[0] / power[-1]) > 95.0
        reference = np.median(ratio[n // 8: n // 4])
        excess[taper] = [10 * np.log10(ratio[int(n * a): int(n * b)].mean() / reference)
                         for a, b in ((0.5, 0.75), (0.75, 0.9), (0.9, 0.97), (0.97, 1.0))]
    assert np.abs(excess[True]).max() < 0.5, excess[True]
    assert min(excess[False]) > 3.0, excess[False]                  # the limit of the default, kept for old results


def test_band_taper_leaves_the_band_alone():
    """The taper touches the pulse's spectrum only above half the reduced rate's Nyquist
    frequency; the rate is at least four times the pulse's band, so the band is inside."""
    pulse = np.exp(2j * np.pi * 0.02 * np.arange(64) ** 2 / 64) * np.hanning(64)      # an LFM within +-0.1 of the rate
    out = reverb.taper_pulse(pulse, 1024)
    f = np.abs(np.fft.fftfreq(1024))
    before, after = np.fft.fft(pulse, 1024), np.fft.fft(out)
    assert np.abs(after - before)[f <= 0.25].max() < 1e-12
    assert np.abs(after[f > 0.499]).max() < 1e-4 * np.abs(before).max()
    assert np.sum(np.abs(out) ** 2) == pytest.approx(np.sum(np.abs(pulse) ** 2), rel=1e-3)


@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
def test_band_taper_is_the_same_on_both_engines():
    out = []
    for engine in ("cpp", "numpy"):
        sonar = SonarModel.load("popoto_8", {
            "receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": [25.0, 230.0]},
                        "sensitivity_db": None},
            "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3,
                                                              "duration": 2e-3, "window_alpha": 0.0,
                                                              "source_level_db": None, "amplitude": 1.0}}})
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, -20.0), velocity=(3.0, -2.0, 0.0)), Scene(),
                            environment={"sound_speed": C}, adc=False)
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, -27.0, np.random.default_rng(2), n_azimuth=24,
                                 band_taper=True, engine=engine)
        out.append(rec.samples.array)
    assert np.abs(out[0] - out[1]).max() < 5e-6 * np.abs(out[1]).max()
