"""Formulas and constants checked against values printed in open documents (docs/REFERENCES.md),
and the models added from them: the bubble-layer surface backscattering model, the Cox-Munk
slope components, the Francois-Garrison absorption against a published table, the ambient
noise fit against a third print.

  [G01] Gallaudet, dissertation (2001), https://hdl.handle.net/10945/11015, pp. 124-125, eqs. (7)-(12).
  [H10] Col, "Fine-scale variability in temperature, salinity, and pH in the upper-ocean and the effects
        on acoustic transmission loss in the Western Arctic Ocean", MS thesis, Naval Postgraduate
        School (2010), https://hdl.handle.net/10945/5385, Tables 1-2, eq. (2.3).
  [H05] Hurt, MS thesis, Naval Postgraduate School (2005), https://hdl.handle.net/10945/1936.
  [G22] Guerin, Capelle and Hartmann, arXiv:2210.05456v2, eq. (12).
"""

import math
import warnings

import numpy as np
import pytest

from gpusonar import acoustics
from gpusonar.sim import noise, surface

# [H10] Table 2 (depth m, T deg C, S, pH) and Table 1, case 1 (dB/km at 0.1, 1 and 10 kHz).
ARCTIC_TABLE = ((1, -1.43, 26.33, 7.88, 0.00144, 0.04946, 1.02269), (30, -1.34, 26.62, 7.89, 0.00145, 0.05032, 1.02746),
                (70, -0.89, 30.72, 7.95, 0.00149, 0.05742, 1.16074), (180, -1.53, 32.99, 7.63, 0.00089, 0.03840, 1.21621),
                (450, 0.74, 34.83, 7.86, 0.00114, 0.05152, 1.17101), (1000, -0.05, 34.91, 7.85, 0.00114, 0.04985, 1.11174),
                (3500, -0.28, 34.95, 7.87, 0.00112, 0.04660, 0.77979))


def test_francois_garrison_against_a_published_table():
    """21 values "calculated using the Francois-Garrison equation" for measured Arctic profiles,
    surface to 3500 m, pH 7.63 to 7.95, salinity 26 to 35: ours are within 0.6 % (the table has
    three digits at 0.1 kHz, hence the wider bound there)."""
    f = np.array([100.0, 1e3, 1e4])
    for z, t, s, ph, *printed in ARCTIC_TABLE:
        ours = acoustics.absorption_francois_garrison(f, t, s, z, ph) * 1e3          # dB/km
        assert ours[1:] == pytest.approx(printed[1:], rel=0.007), z
        assert ours[0] == pytest.approx(printed[0], rel=0.007, abs=6e-6), z
        # The simpler Ainslie-McColm fit stays within 3 % of the same table over this range.
        other = acoustics.absorption_ainslie_mccolm(f, t, s, z, ph) * 1e3
        assert other == pytest.approx(printed, rel=0.031), z


def test_ainslie_mccolm_relaxation_frequencies_as_printed():
    """[H10] after eq. (2.3): boric acid 0.769 kHz and magnesium sulphate 41.372 kHz at
    T = -0.256 deg C, S = 34.67, with the formula's 0.78 sqrt(S / 35) exp(T / 26) and 42 exp(T / 17)."""
    t, s = -0.256, 34.67
    assert 0.78 * math.sqrt(s / 35.0) * math.exp(t / 26.0) == pytest.approx(0.769, abs=5e-4)
    assert 42.0 * math.exp(t / 17.0) == pytest.approx(41.372, abs=5e-4)
    # The implemented formula has exactly those two relaxations: at f = f2 >> f1 the magnesium
    # sulphate term is half its high-frequency plateau 0.52 (1 + T / 43) (S / 35) f2.
    f2 = 42.0 * math.exp(t / 17.0)
    total = float(acoustics.absorption_ainslie_mccolm(f2 * 1e3, t, s, 0.0, 8.0)) * 1e3
    mgso4 = 0.5 * 0.52 * (1.0 + t / 43.0) * (s / 35.0) * f2
    water = 0.00049 * f2**2 * math.exp(-t / 27.0)
    f1 = 0.78 * math.sqrt(s / 35.0) * math.exp(t / 26.0)
    boric = 0.106 * f1 * f2**2 / (f1**2 + f2**2)
    assert total == pytest.approx(mgso4 + water + boric, rel=1e-9)


def test_ambient_noise_fit_against_a_third_print():
    """[H05] eqs. (2.7)-(2.9) and its listing, typed here: f in kHz, w in m/s, D from 0 to 1."""
    for f_khz, w, d in ((0.1, 5.0, 0.5), (1.0, 2.0, 0.0), (10.0, 10.0, 1.0), (50.0, 5.0, 0.5), (300.0, 15.0, 0.2)):
        want = {"wind": 50 + 7.5 * w**0.5 + 20 * math.log10(f_khz) - 40 * math.log10(f_khz + 0.4),
                "thermal": -15 + 20 * math.log10(f_khz),
                "shipping": 40 + 20 * (d - 0.5) + 26 * math.log10(f_khz) - 60 * math.log10(f_khz + 0.03),
                "turbulence": 17 - 30 * math.log10(f_khz)}
        total, parts = noise.wenz_spectrum_db(f_khz * 1e3, wind_speed=w, shipping=d, components=True)
        for key, value in want.items():
            assert float(parts[key]) == pytest.approx(value, abs=1e-9), (key, f_khz)
        assert float(total) == pytest.approx(10 * math.log10(sum(10 ** (v / 10) for v in want.values())), abs=1e-9)
    # A hand value: wind noise at 10 kHz in 5 m/s is 50 + 16.77 + 20 - 40.68 = 46.09 dB.
    assert float(noise.wenz_spectrum_db(10e3, 5.0, components=True)[1]["wind"]) == pytest.approx(46.09, abs=0.01)


def test_cox_munk_components():
    """[G22] eq. (12): m_u = 3.16e-3 U, m_c = 3e-3 + 1.92e-3 U (U at 12.5 m, 1 to 14 m/s)."""
    up, cross = acoustics.mean_square_slope_components(10.0)
    assert up == pytest.approx(0.0316) and cross == pytest.approx(0.0222)
    for u in (1.0, 5.0, 10.0, 14.0):
        up, cross = acoustics.mean_square_slope_components(u)
        # The total carried by mean_square_slope is the separate regression 0.003 + 0.00512 U.
        assert up + cross == pytest.approx(acoustics.mean_square_slope(u), rel=0.009)
        assert 0.3 < up / cross < 1.5
    assert acoustics.mean_square_slope_components(-2.0) == (0.0, 0.003)


def test_bubble_constants_agree_with_the_independent_bubble_dynamics():
    """[G01] prints delta = 2.55e-2 f^(1/3) and delta_r = 0.0136 without the unit of f. The
    module already has a resonant bubble from another source (Commander-Prosperetti damping,
    eqs. (7)-(9) of arXiv:2410.14415): its total damping constant at resonance, 2 b / w, equals
    the printed law with f in kHz within 2 % from 10 to 100 kHz, and k a at resonance is the
    printed delta_r. So the unit is kHz and neither number is a free constant."""
    for f in (10e3, 30e3, 68e3, 100e3):
        w = 2.0 * np.pi * f
        radii = np.geomspace(0.5, 2.0, 8001) * float(surface.resonant_radius(f))
        w0_sq, b = surface.bubble_oscillator(radii, f)
        i = int(np.argmin(np.abs(w0_sq - w * w)))
        assert 2.0 * b[i] / w == pytest.approx(surface.bubble_damping_constant(f), rel=0.02), f
        assert surface.bubble_damping_constant(f) == pytest.approx(2.55e-2 * (f / 1e3) ** (1 / 3))
        assert w * float(surface.resonant_radius(f)) / 1500.0 == pytest.approx(surface.BUBBLE_RERADIATION_DAMPING, rel=0.01)
        assert w * radii[i] / 1500.0 == pytest.approx(surface.BUBBLE_RERADIATION_DAMPING, rel=0.06)
    # In Hz the law would give a damping constant above 0.5: not a lightly damped resonance.
    assert 2.55e-2 * 30e3 ** (1 / 3) > 0.5


def test_bubble_backscatter_limits_as_printed():
    """[G01] eq. (10) and its two printed limits, (11) saturation and (12) sparse layer."""
    f, t = 30e3, np.radians([5.0, 10.0, 20.0, 45.0])
    delta, delta_r = surface.bubble_damping_constant(f), 0.0136
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Sparse: 1 m/s (the law's exponential branch below 6 m/s makes beta_I tiny).
        beta = surface.bubble_layer_optical_depth(1.0, f)
        assert beta < 1e-3
        assert surface.bubble_backscatter(t, 1.0, f) == pytest.approx(1.5 * delta_r * beta / (np.pi * delta), rel=0.02)
        # Saturated: the Hall-Novarini layer at 15 m/s is optically thick (beta_I above 2).
        thick = surface.bubble_layer_optical_depth(15.0, f, extinction="hall_novarini")
        assert thick > 2.0
        low = t[:3]                                          # up to 20 degrees: beta_I / sin(t) above 6.9
        got = surface.bubble_backscatter(low, 15.0, f, extinction="hall_novarini")
        assert got == pytest.approx(delta_r * np.sin(low) / (8.0 * np.pi * delta), rel=0.02)
        # In between, eq. (10) itself by hand at 10 m/s and 10 degrees with each extinction law:
        # the default `dahl_1997` (beta_I 0.257, -27.2 dB) and `apl_1994` (the default before A1, -25.5 dB).
        for name, level in (("dahl_1997", -27.2), ("apl_1994", -25.5)):
            beta = surface.bubble_layer_optical_depth(10.0, f, extinction=name)
            b = beta / math.sin(math.radians(10.0))
            want = delta_r * math.sin(math.radians(10.0)) / (8 * math.pi * delta) * (1 + 8 * b * math.exp(-2 * b) - math.exp(-4 * b))
            assert float(surface.bubble_backscatter(math.radians(10.0), 10.0, f, extinction=name)) == pytest.approx(want, rel=1e-12)
            assert 10 * math.log10(want) == pytest.approx(level, abs=0.1), name
        assert surface.bubble_backscatter(math.radians(10.0), 10.0, f) == surface.bubble_backscatter(
            math.radians(10.0), 10.0, f, extinction="dahl_1997")
    # The level never exceeds 2.36 times the saturation level (the maximum of 1 + 8 b e^-2b - e^-4b).
    b = np.linspace(0.0, 20.0, 20001)
    assert np.max(1 + 8 * b * np.exp(-2 * b) - np.exp(-4 * b)) == pytest.approx(2.357, abs=0.002)


def test_optical_depth_is_the_one_of_the_loss_law():
    """beta_I is one vertical crossing: a bounce at grazing angle t loses 2 beta_I / sin(t) nepers
    of intensity, which is the per-bounce loss of the law it is taken from (`apl_1994`, the default
    until A1, and `hall_novarini`; the default `dahl_1997` is a fit of beta_I itself)."""
    for u, f in ((6.0, 12e3), (10.0, 30e3), (15.0, 70e3)):
        beta = surface.bubble_layer_optical_depth(u, f, extinction="apl_1994")
        for deg in (5.0, 20.0, 60.0):
            t = math.radians(deg)
            assert (10 / math.log(10)) * 2 * beta / math.sin(t) == pytest.approx(
                float(surface.apl_1994_bubble_loss_db(t, u, f)), rel=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hn = surface.bubble_layer_optical_depth(u, f, extinction="hall_novarini")
            assert (10 / math.log(10)) * 2 * hn / math.sin(0.3) == pytest.approx(float(surface.bubble_loss_db(0.3, u, f)), rel=1e-9)
    with pytest.raises(ValueError, match="unknown extinction"):
        surface.bubble_layer_optical_depth(10.0, 30e3, extinction="other")


def test_facet_term_and_the_whole_model():
    """[G01] eq. (8): sec^4(ti) / (4 pi s^2) exp(-tan^2(ti) / s^2). At normal incidence it is
    1 / (4 pi s^2). It is the geometric-optics cross section (sec^4(ti) / 4) x (density of
    slopes at the specular slope tan(ti)) for the isotropic Gaussian density
    p = exp(-slope^2 / s^2) / (pi s^2), whose integral over the slope plane is one: so s^2 is
    the total mean square slope, the quantity `acoustics.mean_square_slope` gives."""
    s2 = acoustics.mean_square_slope(10.0)
    assert float(surface.facet_backscatter(0.5 * np.pi, 10.0)) == pytest.approx(1.0 / (4 * np.pi * s2), rel=1e-9)
    slope = np.linspace(0.0, 3.0, 300001)
    density = np.exp(-slope**2 / s2) / (np.pi * s2)
    assert np.trapezoid(density * 2 * np.pi * slope, slope) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(density * 2 * np.pi * slope**3, slope) == pytest.approx(s2, rel=1e-5)     # its mean square
    for deg in (60.0, 75.0, 85.0):
        t = math.radians(deg)
        tan_i = 1.0 / math.tan(t)
        geometric = 0.25 * (1 + tan_i**2) ** 2 * math.exp(-tan_i**2 / s2) / (math.pi * s2)
        assert float(surface.facet_backscatter(t, 10.0)) == pytest.approx(geometric, rel=1e-9)
    g = np.radians([10.0, 30.0, 60.0, 85.0, 90.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # inside the stated range: no warning
        total = surface.bubble_layer_surface_db(g, 10.0, 30e3)
    # Bubbles at low angles (the saturation level is -29.3 dB at 10 degrees and the bracket of eq. (10)
    # adds at most 3.7 dB: -27.2 dB with the default dahl_1997, -25.5 with apl_1994), facets near the normal.
    assert -28.0 < total[0] < -24.0 and total[-1] == pytest.approx(10 * math.log10(1 / (4 * np.pi * s2)), abs=0.05)
    assert np.all(np.diff(total) > 0.0)
    with pytest.warns(UserWarning, match="outside the range"):
        surface.bubble_layer_surface_db(g, 10.0, 300e3)
    with pytest.warns(UserWarning, match="outside the range"):
        surface.bubble_layer_surface_db(g, 20.0, 30e3)
    fn = surface.bubble_layer_surface_strength(10.0, 30e3)
    assert float(fn(g[1], g[1])) == pytest.approx(10 ** (total[1] / 10), rel=1e-12)


def test_scenario_key_for_the_bubble_layer_surface_model():
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    cfg = load_scenario_config("point_grid_fls_300k")
    cfg["n_pings"] = 1
    cfg["reverberation"] = [{"model": "bubble_layer_surface", "boundary_z": 0.0, "wind_speed": 8.0}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                      # 300 kHz is outside the model's stated range
        with_surface = run_scenario(cfg).records
        cfg["reverberation"] = []
        without = run_scenario(cfg).records
    assert len(with_surface) == 1 and np.all(np.isfinite(with_surface[0].samples.array))
    # The surface reverberation is there: the record differs from the one without the item.
    assert not np.array_equal(with_surface[0].samples.array, without[0].samples.array)


def test_pierson_moskowitz_constants_and_the_two_wave_statistics():
    """S(w) = a g^2 / w^5 exp(-b (w0 / w)^4), a = 8.1e-3, b = 0.74, w0 = g / U with U at 19.5 m:
    printed as eq. (5.1) of Ead, "Predicting the effects of sea surface scatter on broad band
    pulse propagation with an ocean acoustic parabolic equation model", MS thesis, Naval
    Postgraduate School (2004), https://hdl.handle.net/10945/1160. Its zeroth moment is the mean
    square height and its second moment the mean square vertical speed: the two closed forms the
    time-spreading model takes from the wind, integrated here numerically."""
    a, b, g = 8.1e-3, 0.74, 9.81
    for u in (4.0, 10.0, 15.0):
        w = np.geomspace(0.05 * g / u, 400.0 * g / u, 400001)
        s = a * g**2 / w**5 * np.exp(-b * (g / (u * w)) ** 4)
        m0, m2 = np.trapezoid(s, w), np.trapezoid(w**2 * s, w)
        assert math.sqrt(m0) == pytest.approx(u**2 / g * math.sqrt(a / (4 * b)), rel=1e-4)
        # `rms_wave_height` carries the rounded form: significant height 0.21 U^2 / g (exactly 0.2092).
        assert acoustics.rms_wave_height(u) == pytest.approx(math.sqrt(m0), rel=0.005)
        assert 4 * math.sqrt(m0) * g / u**2 == pytest.approx(0.2092, abs=2e-4)
        assert acoustics.rms_surface_vertical_speed(u) == pytest.approx(math.sqrt(m2), rel=2e-4)


def test_surface_noise_coherence_against_the_printed_closed_form():
    """Cron-Sherman vertical coherence, eq. (4) of Shajahan, Barclay and Lin (2022), and the
    isotropic sinc of their eq. (3); the horizontal pair is 2 J1(x) / x."""
    from scipy.special import j1
    f, c = 400.0, 1500.0
    k = 2 * np.pi * f / c
    for d in (0.3, 1.0, 2.2, 5.0):
        x = k * d
        vertical = noise.surface_noise_coherence([[0, 0, 10.0], [0, 0, 10.0 + d]], f, c)
        want = 2 * (math.sin(x) / x + (math.cos(x) - 1) / x**2) + 2j * (math.sin(x) / x**2 - math.cos(x) / x)
        assert vertical[0, 1] == pytest.approx(want, abs=1e-10)
        assert vertical[1, 0] == pytest.approx(np.conj(want), abs=1e-10) and vertical[0, 0] == pytest.approx(1.0)
        horizontal = noise.surface_noise_coherence([[0, 0, 10.0], [d, 0, 10.0]], f, c)
        assert horizontal[0, 1] == pytest.approx(2 * j1(x) / x, abs=1e-10)
        assert noise.isotropic_coherence([[0, 0, 0.0], [0, 0, d]], f, c)[0, 1] == pytest.approx(math.sin(x) / x, abs=1e-12)
    # The isotropic coherence first vanishes at half a wavelength; the real part of the
    # surface-noise one for a vertical pair already at 0.371 wavelengths (tan(x / 2) = x) and is
    # -4 / pi^2 at half a wavelength: the difference the paper maps.
    half = 0.5 * c / f
    assert abs(noise.isotropic_coherence([[0, 0, 0.0], [0, 0, half]], f, c)[0, 1]) < 1e-12
    assert noise.surface_noise_coherence([[0, 0, 0.0], [0, 0, half]], f, c)[0, 1].real == pytest.approx(-4 / np.pi**2, abs=1e-9)
    grid = np.linspace(0.3, 1.0, 7001) * c / f
    real = np.array([2 * (math.sin(k * d) / (k * d) + (math.cos(k * d) - 1) / (k * d) ** 2) for d in grid])
    first_zero = grid[np.argmax(real < 0.0)] * f / c
    assert first_zero == pytest.approx(0.371, abs=0.001)
