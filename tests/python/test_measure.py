"""Estimators for measured seafloor data (python/gpusonar/measure.py; G-9 of docs/tasks/SST_GAPS_JACKSON.md):
the statistical error of an average of N independent samples against the numbers Jackson and Richardson
(2007) print, the dB-against-intensity averaging bias, the number of independent samples, and the
monostatic estimator (G.13) with Psi from (G.28)-(G.29), inverted on our own simulator's reverberation.

Sources: docs/references/jackson_richardson_2007/03_wave_theories_reflection_penetration.md Sec. 2.9 and
Sec. 6 (App. G), 02_roughness_heterogeneity_statistics.md Sec. 2.6 (Ch. 16), and
docs/references/surface_papers_mcdaniel_dahl.md Sec. 2.9 (Dahl and Plant 1997)."""

import math

import numpy as np
import pytest

from gpusonar import measure
from gpusonar.sim import reverb
from gpusonar.sim.propagation import FreeFieldProvider

C = 1500.0


def test_statistical_error_at_16_and_67_samples_is_the_printed_one():
    """App. G.2.3: "if N = 16, the standard deviation is one-fourth of the mean, or roughly 1 dB"; the note's
    dB form 10 log10(1 + 1/sqrt(N)) gives 0.97 dB at 16 and 0.5 dB at 67 (printed in the note), and
    0.261 dB at 260, which the note rounds to 0.25."""
    assert measure.sigma_db_from_samples(16) == pytest.approx(0.97, abs=0.005)        # 0.9691
    assert measure.sigma_db_from_samples(67) == pytest.approx(0.50, abs=0.005)        # 0.5006
    assert measure.sigma_db_from_samples(260) == pytest.approx(0.261, abs=0.001)      # the note: "0.25"
    assert 10 ** (measure.sigma_db_from_samples(16) / 10) - 1 == pytest.approx(0.25, rel=1e-12)   # one-fourth of the mean
    lower, upper = measure.error_bar_db(16)
    assert upper == measure.sigma_db_from_samples(16) and lower == pytest.approx(-1.249, abs=1e-3)
    # The inverse: the least N for a stated error.
    assert [measure.samples_for_error_db(s) for s in (0.97, 0.5, 0.26)] == [16, 68, 263]
    for n in (1, 2, 16, 67, 260, 1000):
        assert measure.samples_for_error_db(measure.sigma_db_from_samples(n)) == n
    with pytest.raises(ValueError):
        measure.sigma_db_from_samples(0.5)


def test_rayleigh_speckle_averaged_over_n_cells_has_the_stated_spread():
    """Exponential intensities (Gaussian field, (16.14)), averaged over N independent cells: the relative
    standard deviation is 1/sqrt(N) (G.2.3); the dB of the average has the gamma-distribution mean and
    standard deviation of `intensity_mean_db_bias` / `_std` (ours); the mean of dB values is 2.507 dB low
    for every N with standard deviation 5.57/sqrt(N) ((16.15), (16.17); [DP97] eqs. (6)-(7) with
    sigma_T = 0). 200 000 trials, fixed seed; tolerances are about four Monte Carlo standard errors."""
    rng = np.random.default_rng(7)
    trials = 200_000
    for n in (1, 4, 16, 67):
        x = rng.exponential(1.0, (trials, n))
        mean = x.mean(axis=1)
        assert mean.std() == pytest.approx(1.0 / math.sqrt(n), rel=0.01)
        db = 10 * np.log10(mean)
        assert db.mean() == pytest.approx(measure.intensity_mean_db_bias(n), abs=0.02)
        assert db.std() == pytest.approx(measure.intensity_mean_db_std(n), rel=0.01)
        db_mean = (10 * np.log10(x)).mean(axis=1)
        assert db_mean.mean() == pytest.approx(measure.db_mean_bias(n), abs=0.03)
        assert db_mean.std() == pytest.approx(measure.db_mean_std(n), rel=0.01)
        # The book's +1 sigma is the 1/sqrt(N) step of the intensity, upward: 84 % of the averages lie
        # below it for a Gaussian; the skewed gamma puts 80-86 % there at these N.
        below = np.mean(db < measure.sigma_db_from_samples(n))
        assert 0.80 < below < 0.87, (n, below)
    assert measure.db_mean_bias() == pytest.approx(-2.507, abs=5e-4)
    assert measure.db_mean_std(1) == pytest.approx(5.57, abs=5e-3)
    # At N = 1 the two averages coincide: the intensity mean's dB bias is the dB mean's.
    assert measure.intensity_mean_db_bias(1) == pytest.approx(measure.DB_MEAN_BIAS_DB, abs=1e-12)
    # Averaging dB instead of intensity, the book's warning (Sect. 11.1): intensity_average_db is unbiased.
    levels = 10 * np.log10(rng.exponential(1.0, 400_000))
    assert measure.intensity_average_db(levels) == pytest.approx(0.0, abs=0.02)
    assert levels.mean() == pytest.approx(-2.507, abs=0.03)


def test_number_of_independent_samples():
    """p. 429: independent samples are a resolution apart, the pulse length (CW) or about 1/bandwidth;
    (16.26) gives the effective count of correlated samples, N^2 / sum rho_s^2."""
    assert measure.time_resolution(1e-3) == 1e-3
    assert measure.time_resolution(10e-3, 20e3) == pytest.approx(5e-5)
    assert measure.independent_samples(20e-3, 1e-3) == pytest.approx(20.0)
    assert measure.independent_samples(20e-3, 10e-3, 20e3, beams=4, pings=3) == pytest.approx(400 * 12)
    assert measure.independent_samples(1e-4, 1e-3) == 1.0                       # at least one per beam and ping
    rho = measure.rho_cw(1e-3)
    # Samples one pulse length apart are uncorrelated; identical samples count once.
    assert measure.effective_looks(np.arange(10) * 1e-3, rho) == pytest.approx(10.0, rel=1e-12)
    assert measure.effective_looks(np.zeros(5), rho) == pytest.approx(1.0, rel=1e-12)
    # Dense sampling of a window W of a CW return: W / (integral of rho_s^2) = 1.5 W / T (triangle).
    t = np.arange(0.0, 20e-3, 2e-5)
    assert measure.effective_looks(t, rho) == pytest.approx(30.0, rel=0.04)
    # An FM pulse of bandwidth B resolves about 1/B: dense sampling of 2 ms at B = 10 kHz gives about 2 ms x B.
    n_fm = measure.effective_looks(np.arange(0.0, 2e-3, 1e-5), measure.rho_lfm(10e-3, 10e3))
    assert 15.0 < n_fm < 25.0, n_fm
    assert measure.two_way_time(10.0, math.radians(30.0), C) == pytest.approx(2 * 20.0 / C)


def test_g13_round_trip_and_its_corrected_terms():
    """(G.13) as a forward level and solved for S_b: an exact round trip; the absorption term is the
    WATER's, two-way (2 alpha_w r: the printed "alpha_p" is a misprint, note 03 Sec. 6); the area is
    r c tau Psi / 2 ((G.11)); the range law -30 log10 r (with the area's r, -40 log10 r)."""
    psi = measure.psi_gaussian(math.radians(20.0), math.radians(8.0), math.radians(6.0))
    kw = dict(source_level_db=210.0, receive_sensitivity_db=-180.0, tx_directivity_db=-1.0, rx_directivity_db=-0.5,
              sound_speed=C)
    r = np.array([30.0, 60.0, 120.0])
    level = measure.reverberation_level_db(-30.0, r, 1e-3, psi, absorption_db_per_m=0.01, **kw)
    assert measure.backscattering_strength_db(level, r, 1e-3, psi, absorption_db_per_m=0.01, **kw) == pytest.approx(-30.0, abs=1e-12)
    no_abs = measure.reverberation_level_db(-30.0, r, 1e-3, psi, **kw)
    assert no_abs - level == pytest.approx(2 * 0.01 * r, abs=1e-12)
    assert no_abs[1] - no_abs[0] == pytest.approx(-30 * math.log10(2.0), abs=1e-12)
    area = measure.insonified_area(r, 1e-3, psi, C)
    assert no_abs == pytest.approx(210.0 - 180.0 - 30.0 - 1.5 + 10 * np.log10(area / r**4), abs=1e-12)
    # The flat-boundary cell area of the simulator (sim/reverb.k_cell_area) is the same r (c tau / 2) Psi.
    assert reverb.k_cell_area(2 * 59.625 / C, 2 * 60.375 / C, C, psi) == pytest.approx(float(area[1]), rel=1e-9)
    with pytest.warns(UserWarning, match="c tau / 2 << r_t"):
        measure.reverberation_level_db(-30.0, 5.0, 2e-3, psi, sound_speed=C)


def test_g28_g29_psi_for_gaussian_patterns():
    """(G.29): the product of Gaussian power patterns has 1/W^2 = 1/W_x^2 + 1/W_r^2. (G.28) against a
    numerical (G.12) for a pattern Gaussian across the TILTED beam: within 1 % for 10-degree beams at
    10-50 degrees grazing, tilt 25 degrees; on axis Psi = Psi_3 sqrt(pi / ln 2) / (2 cos theta)."""
    wx, wr = math.radians(10.0), math.radians(8.0)
    w3 = measure.psi_3(wx, wr)
    assert w3 == pytest.approx(1 / math.sqrt(1 / wx**2 + 1 / wr**2), rel=1e-15)
    tilt = math.radians(25.0)
    a = np.array([math.cos(tilt), 0.0, math.sin(tilt)])
    n = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])
    phi = np.radians(np.linspace(-40.0, 40.0, 16001))
    for gdeg in (10.0, 25.0, 50.0):
        g = math.radians(gdeg)
        d = np.stack([np.cos(g) * np.cos(phi), np.cos(g) * np.sin(phi), np.full_like(phi, np.sin(g))], axis=-1)
        beta = np.arctan2(d[:, 1], d @ a)
        eps = np.arctan2(d @ n, np.hypot(d @ a, d[:, 1]))
        pattern = np.exp(-4 * math.log(2) * beta**2 / w3**2) * np.exp(-4 * math.log(2) * eps**2 / math.radians(30.0) ** 2)
        numeric = np.trapezoid(pattern, phi) / pattern[len(phi) // 2]
        assert measure.psi_gaussian(g, wx, wr, tilt=tilt) == pytest.approx(numeric, rel=0.01), gdeg
    on_axis = measure.psi_gaussian(math.radians(30.0), w3)
    assert on_axis == pytest.approx(w3 * math.sqrt(math.pi / math.log(2)) / (2 * math.cos(math.radians(30.0))), rel=1e-15)
    with pytest.warns(UserWarning, match="about 20 degrees"):
        measure.psi_gaussian(0.3, math.radians(25.0))


def test_g13_inverts_our_simulated_reverberation_to_the_strength_within_0p3_db():
    """Our simulator's mean seafloor reverberation (`sim/reverb.reverberation_power`, free field, NumPy
    engine) for a Lambert bottom (mu -27 dB) 10 m below a sonar with Gaussian transmit and receive
    patterns (10 and 8 degrees across the beam, 30 in elevation, tilted 25 degrees down), a 1 ms pulse:
    (G.13) with Psi from (G.28)-(G.29) recovers 10 log10(mu sin^2 g) within 0.3 dB at 10-50 degrees
    grazing (measured: +0.08, +0.09, -0.02, +0.02, -0.02, -0.01, -0.20 dB at 10, 15, 20, 25, 30, 40,
    50 degrees; the largest at 50 degrees, where c tau / 2 is 6 % of the range). With Psi taken as
    Psi_3 (the approximation the book mentions) the error grows to over 1 dB at 50 degrees."""
    h, tilt, tau, dt, mu_db = 10.0, math.radians(25.0), 1e-3, 50e-6, -27.0
    wx, wr, we = math.radians(10.0), math.radians(8.0), math.radians(30.0)
    a = np.array([math.cos(tilt), 0.0, math.sin(tilt)])
    n = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])

    def beam(width):
        def gain(d):
            x, y, z = d @ a, d[..., 1], d @ n
            return np.exp(-4 * math.log(2) * (np.arctan2(y, x) ** 2 / width**2 + np.arctan2(z, np.hypot(x, y)) ** 2 / we**2))
        return gain

    tx, rx = beam(wx), beam(wr)
    az = np.radians(np.arange(-30.0, 30.01, 0.25))
    times = np.arange(2 * h / C + dt, 0.09, dt)
    power = reverb.reverberation_power(FreeFieldProvider(C, engine="numpy"), np.zeros(3), h, times, mu_db, pattern=(az, tx, rx))
    pulse = np.convolve(power.sum(axis=0) * (az[1] - az[0]), np.ones(int(round(tau / dt))), mode="same")
    errors, crude = [], []
    for gdeg in (10, 15, 20, 25, 30, 40, 50):
        g = math.radians(gdeg)
        r = h / math.sin(g)
        level = 10 * np.log10(pulse[int(round((2 * r / C - times[0]) / dt))])
        axis = np.array([math.cos(g), 0.0, math.sin(g)])
        d = dict(tx_directivity_db=10 * np.log10(tx(axis)), rx_directivity_db=10 * np.log10(rx(axis)), sound_speed=C)
        truth = mu_db + 20 * math.log10(math.sin(g))
        errors.append(float(measure.backscattering_strength_db(level, r, tau, measure.psi_gaussian(g, wx, wr, tilt=tilt), **d)) - truth)
        crude.append(float(measure.backscattering_strength_db(level, r, tau, measure.psi_3(wx, wr), **d)) - truth)
    assert np.max(np.abs(errors)) < 0.3, errors
    assert abs(crude[-1]) > 1.0, crude
