"""Measured-channel replay and analysis (sim/channel_replay.py).

The estimators are tested on synthetic channels with known spreads, and on our own
time-spreading model seen through them (closed forms of sim/timespread.py). Tests on the
public measured channels (UACR, Zenodo doi:10.5281/zenodo.21287414) need the data:
`tools/fetch_channels.py <dir>` then GPUSONAR_UACR_DIR=<dir>; they are skipped without it.
"""

import os
import pathlib

import numpy as np
import pytest

from gpusonar import acoustics
from gpusonar.sim import ImageSourceProvider
from gpusonar.sim import channel_replay as cr
from gpusonar.sim.timespread import wrap

FS_T, FS_D, FC, BW = 65.1, 13020.0, 12500.0, 6510.0
GRID = dict(n_time=3400, fs_time=FS_T, fs_delay=FS_D, n_delay=128, fc=FC, bandwidth=BW)
_DIR = os.environ.get("GPUSONAR_UACR_DIR", "")
needs_data = pytest.mark.skipif(not (_DIR and (pathlib.Path(_DIR) / "purple_5.mat").exists()),
                                reason="UACR channel data not available (tools/fetch_channels.py, GPUSONAR_UACR_DIR)")


def _bin(delay):
    return int(round(delay * FS_D))


# -- estimators on known channels ---------------------------------------------------------------------

@pytest.mark.parametrize("decay", [0.3e-3, 0.8e-3])
def test_delay_spread_recovered(decay):
    ch = cr.synthetic_channel([{"delay": 2e-3, "incoherent": 1.0, "decay_time": decay, "doppler_width": 1.5}], n_receivers=4, **GRID)
    got = [cr.arrival_statistics(ch, m, _bin(2e-3), tail_bins=int(4 * decay * FS_D)) for m in range(4)]
    assert np.mean([g["decay_time_s"] for g in got]) == pytest.approx(decay, rel=0.15)
    # Mean excess delay over the fitted window: that of an exponential cut at 4 decay times, 0.91 T,
    # to within the pulse's own width.
    assert np.mean([g["mean_excess_delay_s"] for g in got]) == pytest.approx(0.91 * decay, abs=0.6 / BW)
    # A single path has no tail: the profile falls to the side lobes of the band-limited pulse at once.
    one = cr.synthetic_channel([{"delay": 2e-3, "incoherent": 1.0, "decay_time": 0.0, "doppler_width": 1.5}], **GRID)
    p = cr.arrival_statistics(one, 0, _bin(2e-3))["tail_profile"]
    assert p[3:].max() < 0.05 * p[0]


def test_doppler_width_and_shape_recovered():
    sigma, a = 1.5, 1.0
    gauss = cr.synthetic_channel([{"delay": 2e-3, "incoherent": 1.0, "doppler_width": sigma, "doppler_spectrum": "gaussian"}], n_receivers=8, **GRID)
    lor = cr.synthetic_channel([{"delay": 2e-3, "incoherent": 1.0, "doppler_width": a, "doppler_spectrum": "lorentzian"}], n_receivers=8, **GRID)
    g = [cr.arrival_statistics(gauss, m, _bin(2e-3), line_half_width=0.0) for m in range(8)]
    lo = [cr.arrival_statistics(lor, m, _bin(2e-3), line_half_width=0.0) for m in range(8)]
    mean = lambda rows, key: float(np.mean([r["doppler"][key] for r in rows]))  # noqa: E731
    assert mean(g, "rms") == pytest.approx(sigma, rel=0.08)
    assert mean(g, "w50") == pytest.approx(0.6745 * sigma, rel=0.1)
    assert mean(g, "w90") == pytest.approx(1.645 * sigma, rel=0.1)
    assert mean(g, "hwhm") == pytest.approx(1.1774 * sigma, rel=0.2)
    assert mean(g, "shape") == pytest.approx(2.44, abs=0.3)
    assert mean(lo, "w50") == pytest.approx(a, rel=0.15)
    assert mean(lo, "hwhm") == pytest.approx(a, rel=0.25)
    # Lorentzian on an axis of +-fs/2: the 90 % width is tan(0.9 atan(N / a)), N the Nyquist frequency.
    nyq = FS_T / 2.0
    assert mean(lo, "w90") == pytest.approx(a * np.tan(0.9 * np.arctan(nyq / a)), rel=0.2)
    assert mean(lo, "shape") > 4.0                           # tells the two apart
    # Coherence times: exp(-2 pi^2 s^2 t^2) and exp(-2 pi a |t|) fall to 1/e at 1/(sqrt 2 pi s) and 1/(2 pi a).
    assert np.mean([r["coherence_time_s"] for r in g]) == pytest.approx(1.0 / (np.sqrt(2.0) * np.pi * sigma), rel=0.12)
    assert np.mean([r["coherence_time_s"] for r in lo]) == pytest.approx(1.0 / (2.0 * np.pi * a), rel=0.2)


@pytest.mark.parametrize("coherent", [0.0, 0.3, 0.8])
def test_coherent_fraction_and_scintillation_recovered(coherent):
    """A partly coherent arrival behind a steady reference path, with a slow phase drift common to
    both and estimation noise: the reference removes the drift, the floor is subtracted, and the
    energy statistics are Rician: scintillation index 1 - c^2."""
    paths = [{"delay": 1e-3, "amplitude": 3.0},
             {"delay": 4e-3, "amplitude": np.sqrt(coherent) * np.exp(0.7j), "incoherent": 1.0 - coherent, "doppler_width": 1.0}]
    ch = cr.synthetic_channel(paths, n_receivers=6, noise=0.01, common_phase_rms=6.0, seed=3, **GRID)
    ref = cr.reference_phase(ch)
    floor = cr.delay_profile(ch)["floor"]
    assert floor == pytest.approx(0.01, rel=0.1)
    with_ref = [cr.arrival_statistics(ch, m, _bin(4e-3), reference=ref, floor=floor[m]) for m in range(6)]
    without = [cr.arrival_statistics(ch, m, _bin(4e-3), floor=floor[m]) for m in range(6)]
    assert np.mean([s["coherent_fraction"] for s in with_ref]) == pytest.approx(coherent, abs=0.05)
    assert np.mean([s["coherent_line_fraction"] for s in with_ref]) == pytest.approx(coherent, abs=0.12)
    assert np.mean([s["scintillation_index"] for s in with_ref]) == pytest.approx(1.0 - coherent**2, abs=0.12)
    if coherent > 0.0:
        assert np.mean([s["coherent_fraction"] for s in without]) < 0.5 * coherent       # the drift hides it
    # The reference path itself: steady.
    d = cr.arrival_statistics(ch, 0, _bin(1e-3), reference=ref, floor=floor[0])
    assert d["coherent_fraction"] == pytest.approx(1.0, abs=0.02) and d["scintillation_index"] < 0.01


def test_find_arrivals_and_scattering_function():
    paths = [{"delay": 1e-3, "amplitude": 1.0}, {"delay": 3e-3, "incoherent": 0.5, "doppler_width": 2.0},
             {"delay": 6e-3, "incoherent": 0.1, "doppler_width": 4.0}]
    ch = cr.synthetic_channel(paths, noise=1e-4, **GRID)
    prof = cr.delay_profile(ch)
    assert sorted(cr.find_arrivals(prof["total"][0], n=3)) == [_bin(1e-3), _bin(3e-3), _bin(6e-3)]
    nu, tau, s = cr.scattering_function(ch, nfft=256)
    assert s.shape == (256, 128) and tau[_bin(3e-3)] == pytest.approx(3e-3, abs=1 / FS_D)
    # Total power: sum over Doppler times dnu is the delay profile.
    assert s.sum(axis=0) * (nu[1] - nu[0]) == pytest.approx(prof["total"][0], rel=0.1, abs=2e-4)
    for b, width in ((_bin(3e-3), 2.0), (_bin(6e-3), 4.0)):
        col = s[:, b]
        assert np.sqrt((col * nu**2).sum() / col.sum()) == pytest.approx(width, rel=0.2)
    col = s[:, _bin(1e-3)]
    assert col[np.abs(nu) < 0.5].sum() > 0.99 * col.sum()                               # the steady path is a line


def test_estimator_filter_is_a_one_pole_low_pass():
    ch = cr.synthetic_channel([{"delay": 2e-3, "incoherent": 1.0, "doppler_width": 200.0}], n_receivers=16, **GRID)     # white in time
    f3 = 6.0
    out = cr.estimator_filter(ch, f3)
    nu, s0 = cr.doppler_spectrum(ch.h[:, :, _bin(2e-3)], FS_T, 256)
    _, s1 = cr.doppler_spectrum(out.h[:, :, _bin(2e-3)], FS_T, 256)
    ratio = s1 / s0
    for f, want in ((0.0, 1.0), (f3, 0.5), (2 * f3, 0.2)):
        k = int(np.argmin(np.abs(nu - f)))
        assert ratio[k - 2:k + 3].mean() == pytest.approx(want, rel=0.25)
    assert cr.TimeVaryingChannel(h=ch.h, fs_time=FS_T, fs_delay=13020.83, fc=FC, meta={"lambda": 0.995}).estimator_bandwidth() \
        == pytest.approx(0.005 * 13020.83 / (2 * np.pi))


# -- replay ----------------------------------------------------------------------------------------------

def _chirp(fs, duration=0.25, f0=10500.0, f1=14500.0):
    t = np.arange(int(duration * fs)) / fs
    return np.cos(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t**2)) * np.hanning(len(t))


def test_replay_static_channel_is_a_convolution():
    """Two fixed taps: the output is the input delayed and scaled twice, with the carrier phase
    each tap's delay implies."""
    fs = 4 * FS_D
    k1, k2, a2 = 10, 37, 0.5 * np.exp(1.1j)
    h = np.zeros((40, 1, 64), dtype=complex)
    h[:, 0, k1], h[:, 0, k2] = 1.0, a2
    ch = cr.TimeVaryingChannel(h=h, fs_time=FS_T, fs_delay=FS_D, fc=FC)
    x = _chirp(fs)
    y = cr.replay(ch, x, fs)[0]
    # Expected, by hand, on the analytic signal: tap k delays the envelope by k / fs_delay and multiplies it by the tap.
    from scipy.signal import hilbert
    xa = hilbert(np.concatenate([x, np.zeros(len(y) - len(x))]))
    t = np.arange(len(y)) / fs
    env = xa * np.exp(-2j * np.pi * FC * t)
    want = np.zeros(len(y), dtype=complex)
    for k, a in ((k1, 1.0), (k2, a2)):
        want += a * np.roll(env, 4 * k)
    want = np.real(want * np.exp(2j * np.pi * FC * t))
    assert np.sqrt(np.mean((y - want) ** 2)) < 0.03 * np.sqrt(np.mean(want**2))


def test_replay_follows_a_time_varying_tap_and_the_tracked_phase():
    fs = 4 * FS_D
    n_t = 60
    shift = 3.0                                                # Hz: a tap that rotates
    h = np.zeros((n_t, 2, 32), dtype=complex)
    h[:, 0, 5] = np.exp(2j * np.pi * shift * np.arange(n_t) / FS_T)
    h[:, 1, 5] = 1.0
    theta = np.zeros((int(n_t * FS_D / FS_T), 2))
    theta[:, 1] = 2 * np.pi * (-5.0) * np.arange(theta.shape[0]) / FS_D + 0.3          # a tracked -5 Hz on receiver 1
    ch = cr.TimeVaryingChannel(h=h, fs_time=FS_T, fs_delay=FS_D, fc=FC, phase=theta, phase_kind="theta")
    t = np.arange(int(0.6 * fs)) / fs
    y = cr.replay(ch, np.cos(2 * np.pi * 12000.0 * t), fs)
    f = np.fft.rfftfreq(8 * len(y[0]), 1 / fs)
    peak = lambda v: f[np.argmax(np.abs(np.fft.rfft(v * np.hanning(len(v)), 8 * len(v))))]  # noqa: E731
    assert peak(y[0]) == pytest.approx(12000.0 + shift, abs=0.4)
    assert peak(y[1]) == pytest.approx(12000.0 - 5.0, abs=0.4)
    assert peak(cr.replay(ch, np.cos(2 * np.pi * 12000.0 * t), fs, apply_phase=False)[1]) == pytest.approx(12000.0, abs=0.4)
    with pytest.raises(ValueError, match="runs past"):
        cr.replay(ch, np.zeros(int(2.0 * fs)), fs)
    # Baseband in, baseband out; a later start uses later snapshots.
    z = cr.replay(ch, np.ones(2000, dtype=complex), FS_D, baseband=True, start=0.2, receivers=[0])
    assert z.shape[0] == 1 and np.iscomplexobj(z)
    rot = np.angle(z[0, 1500] / z[0, 500]) / (2 * np.pi * 1000 / FS_D)
    assert rot == pytest.approx(shift, abs=0.2)


def test_replay_phi_puts_the_delay_drift_back():
    """`phi_hat` stands for a delay drift phi = -2 pi fc dtau: a steadily growing delay is a
    Doppler shift of the whole signal, carrier and envelope."""
    fs = 4 * FS_D
    n_t = 60
    v_over_c = 2e-4                                            # opening: delay grows by 2e-4 s per s
    h = np.zeros((n_t, 1, 32), dtype=complex)
    h[:, 0, 5] = 1.0
    n = int(n_t * FS_D / FS_T)
    phi = (-2 * np.pi * FC * v_over_c * np.arange(n) / FS_D)[:, None]
    ch = cr.TimeVaryingChannel(h=h, fs_time=FS_T, fs_delay=FS_D, fc=FC, phase=phi, phase_kind="phi")
    t = np.arange(int(0.6 * fs)) / fs
    y = cr.replay(ch, np.cos(2 * np.pi * 13500.0 * t), fs)[0]
    f = np.fft.rfftfreq(8 * len(y), 1 / fs)
    got = f[np.argmax(np.abs(np.fft.rfft(y * np.hanning(len(y)), 8 * len(y))))]
    assert got == pytest.approx(13500.0 * (1.0 - v_over_c), abs=0.4)       # -2.7 Hz, not the carrier's -2.5


# -- our model through the same estimators ------------------------------------------------------------------

DEPTH, RANGE, ZS, ZR, C0 = 15.0, 60.0, 11.0, 11.7, 1500.0


def _model(wind, n_rx=6, n_time=2400, bottom=True, **options):
    inner = ImageSourceProvider(C0, surface_z=0.0, bottom_z=DEPTH if bottom else None, max_order=2 if bottom else 1,
                                bottom=dict(speed=1700.0, density=1.9, attenuation=0.5))
    prov = wrap(inner, dict({"surface": {"wind_speed": wind}, "taps": 24, "coherence_length": 0.01, "seed": 5}, **options), frequency=FC)
    rx = np.array([[RANGE, 0.0, ZR + 0.05 * i] for i in range(n_rx)])
    return cr.model_channel(prov, [0.0, 0.0, ZS], rx, n_time=n_time, fs_time=FS_T, fs_delay=FS_D, n_delay=128, fc=FC, bandwidth=BW,
                            lead=2e-3, return_rays=True)


def _surface_arrival(rays, m=0):
    k = int(np.flatnonzero((rays["n_surface"][m] == 1) & (rays["n_bottom"][m] == 0) & rays["valid"][m])[0])
    return rays["delay"][m, k], np.arcsin(abs(rays["launch"][m, k, 2]))


def test_model_channel_arrivals_are_the_image_geometry():
    ch, rays = _model(0.0, n_rx=1, n_time=8)
    prof = cr.delay_profile(ch)["total"][0]
    direct = np.hypot(RANGE, ZR - ZS)
    want = {"direct": 0.0, "bottom": (np.hypot(RANGE, 2 * DEPTH - ZS - ZR) - direct) / C0, "surface": (np.hypot(RANGE, ZS + ZR) - direct) / C0}
    peaks = sorted(cr.find_arrivals(prof, n=3))
    for b, name in zip(peaks, ("direct", "bottom", "surface")):
        assert b / FS_D - 2e-3 == pytest.approx(want[name], abs=0.6 / FS_D)
    tau_s, grazing = _surface_arrival(rays)
    assert tau_s - 2e-3 == pytest.approx(want["surface"], abs=1e-9)
    assert grazing == pytest.approx(np.arctan2(ZS + ZR, RANGE), abs=1e-9)
    # Flat sea: every arrival steady.
    assert cr.arrival_statistics(ch, 0, peaks[2])["scintillation_index"] < 1e-6


def test_model_surface_arrival_statistics_match_the_closed_forms():
    """The surface-bounce arrival of sim/timespread.py seen through the estimators: coherent
    share exp(-Gamma^2), Gaussian Doppler spectrum of rms width 2 f v sin(g) / c /
    sqrt(1 - exp(-Gamma^2)), Lorentzian of that half width without the root, and the decay time
    of the rule of thumb, 2 mss t1 t2 / (t1 + t2), or of the facets, half of it at low grazing
    angles times (1 + sin^2 g)."""
    wind = 4.0
    h_rms, mss, v = acoustics.rms_wave_height(wind), acoustics.mean_square_slope(wind), acoustics.rms_surface_vertical_speed(wind)
    out = {}
    for key, opts in (("gauss_facet", dict(doppler_spectrum="gaussian", decay_time="facet")),
                      ("lorentz_rule", dict(doppler_spectrum="lorentzian", decay_time="rule_of_thumb"))):
        ch, rays = _model(wind, **opts)
        rows = []
        for m in range(ch.n_receivers):
            tau, g = _surface_arrival(rays, m)
            rows.append(cr.arrival_statistics(ch, m, int(round(tau * FS_D)), tail_bins=14, line_half_width=0.0))
        out[key] = (rows, g, tau + rays["delay"][0].min())
    rows, g, _ = out["gauss_facet"]
    gamma2 = (2.0 * 2.0 * np.pi * FC / C0 * h_rms * np.sin(g)) ** 2
    assert gamma2 > 10.0                                       # fully rough at 60 m: no coherent part
    assert np.mean([r["coherent_fraction"] for r in rows]) < 0.03
    assert np.mean([r["scintillation_index"] for r in rows]) == pytest.approx(1.0, abs=0.25)
    width = 2.0 * FC * v * np.sin(g) / C0
    assert np.mean([r["doppler"]["rms"] for r in rows]) == pytest.approx(width, rel=0.15)
    assert np.mean([r["doppler"]["shape"] for r in rows]) == pytest.approx(2.44, abs=0.5)
    t_total = np.hypot(RANGE, ZS + ZR) / C0
    lever = t_total * (ZS / (ZS + ZR)) * (ZR / (ZS + ZR))
    assert np.mean([r["decay_time_s"] for r in rows]) == pytest.approx(mss * (1.0 + np.sin(g) ** 2) * lever, rel=0.3)
    rows, g, _ = out["lorentz_rule"]
    assert np.mean([r["doppler"]["w50"] for r in rows]) == pytest.approx(width, rel=0.3)
    assert np.mean([r["doppler"]["shape"] for r in rows]) > 3.5
    assert np.mean([r["decay_time_s"] for r in rows]) == pytest.approx(2.0 * mss * lever, rel=0.3)


def test_model_partly_coherent_surface_arrival():
    """At a low grazing angle the surface arrival keeps a coherent part exp(-Gamma^2)."""
    global RANGE
    keep = RANGE
    try:
        RANGE = 200.0
        wind = 4.0
        ch, rays = _model(wind, n_rx=8, doppler_spectrum="gaussian", decay_time="facet", bottom=False)
        got, peak = [], []
        for m in range(ch.n_receivers):
            tau, g = _surface_arrival(rays, m)
            st = cr.arrival_statistics(ch, m, int(round(tau * FS_D)), half_width=1, tail_bins=40)
            got.append(st["coherent_fraction_arrival"])
            peak.append(st["coherent_fraction"])
        gamma2 = (2.0 * 2.0 * np.pi * FC / C0 * acoustics.rms_wave_height(wind) * np.sin(g)) ** 2
        assert 0.2 < np.exp(-gamma2) < 0.8
        assert np.mean(got) == pytest.approx(np.exp(-gamma2), abs=0.1)
        # At the peak alone the coherent share is larger: the scattered part is spread over the tail.
        assert np.mean(peak) > np.mean(got) + 0.2
    finally:
        RANGE = keep


# -- the public measured channels ------------------------------------------------------------------------------

@needs_data
def test_uacr_file_loads_with_its_normalisation():
    """UACR file specification: the energy of h_hat summed over taps and receivers and averaged
    over time is M / nsd."""
    import h5py
    path = pathlib.Path(_DIR) / "purple_5.mat"
    with h5py.File(path, "r") as f:
        m_all = f["h_hat"].shape[1]
    ch = cr.load_uacr(str(path))
    assert ch.n_receivers == m_all == 12 and ch.phase_kind == "theta"
    assert ch.fc == 12500.0 and ch.fs_delay == pytest.approx(1e7 / 768.0) and ch.fs_time == pytest.approx(1e7 / 1536.0 / 100.0)
    assert ch.phase.shape[0] / ch.fs_delay == pytest.approx(ch.duration, rel=1e-3)
    assert (np.abs(ch.h) ** 2).sum() / ch.n_time == pytest.approx(m_all / ch.meta["nsd"], rel=0.02)
    assert ch.estimator_bandwidth() == pytest.approx(10.36, abs=0.05)
    sub = cr.load_uacr(str(path), receivers=[3, 7])
    assert np.array_equal(sub.h, ch.h[:, [3, 7]])


@needs_data
def test_uacr_replay_power_is_the_documented_one():
    """"A passband signal of unit power, in the band the channel was estimated in and replayed
    through all M receivers ... gives output powers that sum to M" (over the whole channel; a
    few seconds see only part of the fading, so the tolerance is wide)."""
    ch = cr.load_uacr(str(pathlib.Path(_DIR) / "purple_5.mat"))
    from scipy.signal import firwin, lfilter
    fs = 4.0 * ch.fs_delay
    rng = np.random.default_rng(0)
    half = ch.fs_delay / 4.0 * 0.8                             # inside the estimate's band (fs_delay / 2 wide)
    x = lfilter(firwin(513, [ch.fc - half, ch.fc + half], pass_zero=False, fs=fs), 1.0, rng.standard_normal(int(4.0 * fs)))
    x /= x.std()
    y = cr.replay(ch, x, fs, start=5.0)
    assert y.shape[0] == 12
    assert y.var(axis=1).sum() * y.shape[1] / len(x) == pytest.approx(12.0, rel=0.35)


@needs_data
def test_uacr_purple_60m_has_the_image_source_arrivals():
    """SPACE'08 at 60 m (purple_1): bottom bounce and surface bounce where straight rays in 15 m
    of water put them; the bottom bounce steady, the surface bounce fully fluctuating."""
    path = pathlib.Path(_DIR) / "purple_1.mat"
    if not path.exists():
        pytest.skip("purple_1.mat not fetched")
    ch = cr.load_uacr(str(path), receivers=[0])
    ref = cr.reference_phase(ch)
    prof = cr.delay_profile(ch, ref)
    p = prof["total"][0]
    d = int(np.argmax(p))
    direct = np.hypot(60.0, 0.7)
    b = d + int(round((np.hypot(60.0, 4.0 + 3.3) - direct) / 1500.0 * ch.fs_delay))
    s = d + int(round((np.hypot(60.0, 11.0 + 11.7) - direct) / 1500.0 * ch.fs_delay))
    b_found = b - 2 + int(np.argmax(p[b - 2:b + 3]))
    s_found = s - 5 + int(np.argmax(p[s - 5:s + 6]))
    assert abs(b_found - b) <= 1 and abs(s_found - s) <= 4
    floor = float(prof["floor"][0])
    bottom = cr.arrival_statistics(ch, 0, b_found, reference=ref, floor=floor, half_width=0)
    surface = cr.arrival_statistics(ch, 0, s_found, reference=ref, floor=floor)
    assert bottom["coherent_fraction"] > 0.9 and bottom["scintillation_index"] < 0.05
    assert surface["coherent_fraction"] < 0.05 and surface["scintillation_index"] == pytest.approx(1.0, abs=0.35)
    assert 2.0 < surface["doppler"]["shape"] < 3.2             # a Gaussian-like Doppler spectrum, not a Lorentzian


# -- the validation tool -------------------------------------------------------------------------------------

def _tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("channel_validation", pathlib.Path(__file__).resolve().parents[2] / "tools" / "channel_validation.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_validation_procedure_recovers_a_known_wind_and_tells_the_options_apart(tmp_path):
    """tools/channel_validation.py on a "measurement" made by the model itself (wind 5 m/s,
    Gaussian Doppler spectrum, facet decay time): the one-parameter fit returns the wind, the
    arrivals are found where the images are, model A (the truth) agrees on the Doppler shape and
    the decay time, and model B (Lorentzian, rule of thumb) does not."""
    tool = _tool()
    ch, geo, rx = tool.synthetic_measured(1, 3, wind=5.0, n_time=1740)
    result = tool.run([("synthetic_1", 0, ch, geo, rx)], seeds=2, log=lambda *_: None)
    site = result["sites"][0]
    assert result["free_parameters"] == 1
    assert site["wind_fit"]["wind_speed"] == pytest.approx(5.0, rel=0.15)
    groups = {g["name"]: g for g in site["groups"]}
    first = site["groups"][0]                                  # the direct path, with the bottom bounce 0.29 ms behind it
    assert first["name"] in ("D", "D+B") and "S" in groups and first["n_surface"] == 0
    s = groups["S"]
    direct = np.hypot(60.0, 0.7)
    assert s["measured"]["delay_ms"][0] == pytest.approx((np.hypot(60.0, 22.7) - direct) / 1500.0 * 1e3, abs=0.16)
    assert first["measured"]["coherent"][0] > 0.95 and s["measured"]["coherent"][0] < 0.05
    meas, a, b = s["measured"], s["model"]["A"], s["model"]["B"]
    # (One seed of the model is a finite set of spectral lines: its shape scatters by a few tenths.)
    assert a["doppler_shape"][0] == pytest.approx(meas["doppler_shape"][0], abs=1.0)
    assert b["doppler_shape"][0] > a["doppler_shape"][0] + 0.8
    assert a["doppler_rms"][0] == pytest.approx(meas["doppler_rms"][0], rel=0.2)
    assert a["decay_ms"][0] == pytest.approx(meas["decay_ms"][0], rel=0.25)
    assert b["decay_ms"][0] > 1.4 * meas["decay_ms"][0]
    assert tool.verdict(meas["doppler_shape"], a["doppler_shape"], "doppler_shape") == "in"
    text = tool.markdown(result)
    assert "| S | Doppler rms width (Hz) FIT |" in text and "closed forms" in text
    paths = tool.figures(result, tmp_path, "t")
    assert len(paths) == 3 and all(p.stat().st_size > 10_000 for p in paths)


def test_subband_doppler_scales_with_frequency_for_a_moving_path():
    """A path whose delay wanders has a phase 2 pi f delay(t): its Doppler width is proportional to
    the frequency, so the upper half of the band is wider than the lower by the ratio of their
    centre frequencies. A path with fading that does not depend on frequency gives a ratio of 1."""
    rng = np.random.default_rng(8)
    n_time, n_delay = 3400, 96
    t = np.arange(n_time) / FS_T
    wander = sum(a * np.sin(2 * np.pi * f * t + p) for a, f, p in zip(rng.normal(0, 1, 40), rng.uniform(0.05, 1.5, 40), rng.uniform(0, 6.28, 40)))
    wander *= 4e-5 / wander.std()                                # 40 us rms: 3 rad at 12.5 kHz
    h = np.zeros((n_time, 1, n_delay), dtype=complex)
    for i in range(n_time):
        h[i, 0] = cr.band_pulse(n_delay, FS_D, BW, 2e-3 + wander[i]) * np.exp(-2j * np.pi * FC * wander[i])
    moving = cr.TimeVaryingChannel(h=h, fs_time=FS_T, fs_delay=FS_D, fc=FC)
    out = cr.subband_statistics(moving, 0, _bin(2e-3), line_half_width=0.0)
    assert out["frequency_ratio"] == pytest.approx(1.2, abs=0.08) and out["upper"]["frequency"] > FC > out["lower"]["frequency"]
    assert out["doppler_ratio"] == pytest.approx(out["frequency_ratio"], rel=0.05)
    rate = np.gradient(wander, 1.0 / FS_T).std()
    assert out["upper"]["doppler_rms"] == pytest.approx(out["upper"]["frequency"] * rate, rel=0.1)
    flat = cr.synthetic_channel([{"delay": 2e-3, "incoherent": 1.0, "doppler_width": 1.5}], n_receivers=6, **GRID)
    ratios = [cr.subband_statistics(flat, m, _bin(2e-3), line_half_width=0.0)["doppler_ratio"] for m in range(6)]
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.06)


def _sites_tool():
    import importlib.util
    import sys
    tools = pathlib.Path(__file__).resolve().parents[2] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("channel_sites", tools / "channel_sites.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_all_sites_table_survey_and_dependencies(tmp_path):
    """tools/channel_sites.py without the data: every site has a verdict and a reason; the
    model-free survey finds the arrivals of a synthetic channel with their statistics; the
    closed forms used for the cross-arrival comparison are those of sim/timespread.py."""
    tool = _sites_tool()
    assert set(tool.SITES) == {"purple", "green", "black", "red", "yellow", "blue", "pink", "brown"}
    assert all(s["protocol"] and s["why"] and s["geometry"] for s in tool.SITES.values())
    assert [n for n, s in tool.SITES.items() if s["protocol"] == "full"] == ["purple"]
    rls = cr.TimeVaryingChannel(h=np.zeros((2, 1, 4), complex), fs_time=1.0, fs_delay=13020.83, fc=FC, meta={"lambda": 0.995})
    lms = cr.TimeVaryingChannel(h=np.zeros((2, 1, 4), complex), fs_time=1.0, fs_delay=9000.0, fc=6e3, meta={"mu": 1.0 / 14400.0})
    assert tool.tracker_bandwidth(rls)[0] == pytest.approx(10.36, abs=0.01) and "stated" in tool.tracker_bandwidth(rls)[1]
    assert tool.tracker_bandwidth(lms)[0] == pytest.approx(0.0497, abs=0.001) and "assumed" in tool.tracker_bandwidth(lms)[1]
    ch = cr.synthetic_channel([{"delay": 1e-3, "amplitude": 2.0}, {"delay": 3e-3, "incoherent": 1.0, "doppler_width": 1.5},
                               {"delay": 6e-3, "incoherent": 0.5, "doppler_width": 1.0, "doppler_spectrum": "lorentzian"}], n_receivers=2, noise=1e-4, **GRID)
    rows = [r for r in tool.survey(ch) if not r["is_reference"]]
    assert len(rows) == 4
    gauss = [r for r in rows if abs(r["delay_ms"] - 2.0) < 0.2]
    lor = [r for r in rows if abs(r["delay_ms"] - 5.0) < 0.2]
    assert np.mean([r["doppler_rms"] for r in gauss]) == pytest.approx(1.5, rel=0.15)
    assert np.mean([r["doppler_shape"] for r in gauss]) < 3.0 < np.mean([r["doppler_shape"] for r in lor])
    assert np.mean([r["scintillation"] for r in gauss]) == pytest.approx(1.0, abs=0.3)
    # The cross-arrival closed forms: one surface bounce at 60 m, wind 4 m/s.
    val = lambda v: [v, 0.0]  # noqa: E731
    meas = {k: val(1.0) for k in tool.cv.METRICS}
    meas.update(level_db=val(-10.0), doppler_rms=val(1.6), coherent_arrival=val(0.01))
    site = {"label": "purple_1", "recording": 0, "range": 60.0, "wind_speed": 4.0, "floor_db_re_direct": -35.0, "wind_fit": {},
            "groups": [{"name": "S", "n_surface": 1, "measured": meas, "model": {"A": meas, "C": meas}}]}
    dep = tool.dependencies({"sites": [site]})
    assert len(dep) == 1 and dep[0]["fitted_here"] and dep[0]["n_surface"] == 1
    g = np.arctan2(11.0 + 11.7 + 0.0375 * 8, 60.0)
    assert np.radians(dep[0]["grazing_deg"]) == pytest.approx(g, abs=2e-3)
    assert dep[0]["doppler_closed_form"] == pytest.approx(2.0 * 12500.0 * acoustics.rms_surface_vertical_speed(4.0) * np.sin(g) / 1500.0, rel=0.01)
    assert dep[0]["gamma2"] == pytest.approx((4.0 * np.pi * 12500.0 / 1500.0 * acoustics.rms_wave_height(4.0) * np.sin(g)) ** 2, rel=0.02)
    text = tool.report(dep, [], {"red": {"tracker": "2 (LMS)", "files": {"x": tool.survey(ch)}}}, {1: 4.0})
    assert "What each site allows" in text and "| red | x |" in text and "recording 1: 4.00 m/s" in text
    tool.figure(dep, [], {}, tmp_path / "d.png")
    assert (tmp_path / "d.png").stat().st_size > 10_000
    assert tool.main(["--data", str(tmp_path / "nothing")]) == 2


@needs_data
def test_purple_doppler_width_grows_with_frequency_inside_the_band():
    """SPACE'08 at 60 m: the surface arrivals are wider in the upper half of the band than in the
    lower, by about the ratio of the frequencies (a width independent of frequency would give 1)."""
    if not (pathlib.Path(_DIR) / "purple_1.mat").exists():
        pytest.skip("purple_1.mat not fetched")
    rows = _sites_tool().frequency_scaling(pathlib.Path(_DIR), ["purple_1"], n_receivers=4)
    assert len(rows) >= 3
    ratio = np.mean([r["doppler_ratio"] for r in rows])
    assert rows[0]["frequency_ratio"] == pytest.approx(1.216, abs=0.01)
    assert 1.12 < ratio < 1.38


def test_validation_tool_says_what_is_missing_without_data(tmp_path, capsys):
    assert _tool().main(["--data", str(tmp_path)]) == 2
    assert "fetch_channels.py" in capsys.readouterr().out
