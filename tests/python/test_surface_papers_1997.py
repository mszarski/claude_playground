"""Our sea-surface code against the formulas and printed numbers of two surface papers
(docs/tasks/SST_GAPS_JACKSON.md G-2; issue sonar-64sm). Record only: these tests hold equalities
that are already true; no behaviour was changed to make them pass.

  [McD93] S. T. McDaniel, "Sea surface reverberation: A review", J. Acoust. Soc. Am. 94 (4),
          1905-1922 (1993).
  [D97]   P. H. Dahl, W. J. Plant, B. Nuetzel, A. Schmidt, H. Herwig and E. A. Terray,
          "Simultaneous acoustic and microwave backscattering from the sea surface",
          J. Acoust. Soc. Am. 101 (5), 2583-2595 (1997).
  [DP97]  P. H. Dahl and W. J. Plant, "The variability of high-frequency acoustic backscatter
          from the region near the sea surface", J. Acoust. Soc. Am. 101 (5), 2596-2602 (1997).

Every formula below is typed again here from docs/references/surface_papers_mcdaniel_dahl.md
(the note that read the page images; the PDFs are not in the repository), with its equation
number, and evaluated independently of the code under test. Grazing angles throughout.
"""

import math
import warnings

import numpy as np
import pytest

from gpusonar import acoustics
from gpusonar.sim import surface

DELTA_R = 0.0136                       # [D97] p. 2587 and [McD93] eq. (10): re-radiation damping at resonance
KNOTS_PER_MS_EXACT = 3600.0 / 1852.0   # the international knot is 1852 m per hour exactly


def d97_damping(f_hz):
    """[D97] p. 2587 (Devin 1959): delta = 2.55e-3 f^(1/3), f in Hz."""
    return 2.55e-3 * f_hz ** (1.0 / 3.0)


def d97_eq17(theta, beta_i, f_hz):
    """[D97] eq. (17): sigma_0^b = (sin t / 8 pi) (delta_r / delta) (1 + 8 b e^-2b - e^-4b), b = beta_I / sin t."""
    s = np.sin(theta)
    b = beta_i / s
    return s / (8.0 * math.pi) * (DELTA_R / d97_damping(f_hz)) * (1.0 + 8.0 * b * np.exp(-2.0 * b) - np.exp(-4.0 * b))


def d97_eq18(theta, f_hz):
    """[D97] eq. (18), saturation: sin t delta_r / (8 pi delta)."""
    return np.sin(theta) * DELTA_R / (8.0 * math.pi * d97_damping(f_hz))


def d97_eq19(beta_i, f_hz):
    """[D97] eq. (19), weak layer (beta_I << 1): 1.5 beta_I delta_r / (pi delta)."""
    return 1.5 * beta_i * DELTA_R / (math.pi * d97_damping(f_hz))


def mcd93_eq9(theta, n_density, f_hz, c_factor=1.0):
    """[McD93] eqs. (9)-(10) (Crowther 1980), with the [D97] damping in place of McDaniel's own
    (so that the comparison is of the form, not of the damping law):
    SS = delta_r sin t [1 + 8q e^-2q - e^-4q] / (8 pi delta), q = 2 C pi^2 a_r^3 N / (delta_r sin t),
    a_r = 3.2 / f (m, f in Hz)."""
    a_r = 3.2 / f_hz
    q = 2.0 * c_factor * math.pi**2 * a_r**3 * n_density / (DELTA_R * np.sin(theta))
    return DELTA_R * np.sin(theta) * (1.0 + 8.0 * q * np.exp(-2.0 * q) - np.exp(-4.0 * q)) / (8.0 * math.pi * d97_damping(f_hz))


def mcd93_eq7(theta, slope_var_x):
    """[McD93] eq. (7): SS = exp[-(2 tan^2 t sigma_x'^2)^-1] / (8 pi sin^4 t sigma_x'^2), as a
    cross section (10^(SS/10)); sigma_x'^2 the x component of the mean-square slope."""
    return np.exp(-1.0 / (2.0 * np.tan(theta) ** 2 * slope_var_x)) / (8.0 * math.pi * np.sin(theta) ** 4 * slope_var_x)


def mcd93_eq27(theta_deg, wind_knots, f_hz):
    """[McD93] eq. (27) (Chapman and Harris 1962, via secondary): SS = 3.3 beta log(theta/30)
    - 42.4 log(beta) + 2.6, beta = 158 (v f^(1/3))^-0.58, theta in degrees, v in knots, f in Hz."""
    beta = 158.0 * (wind_knots * f_hz ** (1.0 / 3.0)) ** -0.58
    return 3.3 * beta * np.log10(theta_deg / 30.0) - 42.4 * np.log10(beta) + 2.6


@pytest.fixture
def set_optical_depth(monkeypatch):
    """Make `bubble_backscatter` see a chosen beta_I (the laws only reach some values at some winds)."""
    def apply(value):
        monkeypatch.setattr(surface, "bubble_layer_optical_depth", lambda *args, **kwargs: value)
    return apply


def test_damping_constant_is_d97_in_hz():
    """[D97] p. 2587 prints delta = 2.55e-3 f^(1/3) with f in Hz. Ours is written 2.55e-2 (f/kHz)^(1/3)
    (the unit was "ours" before: Gallaudet's summary did not print it); 2.55e-2 x 1000^(-1/3) = 2.55e-3,
    so the two are the same law."""
    for f in (8e3, 12e3, 26.5e3, 50e3, 68e3, 70e3, 120e3, 330e3):
        assert surface.bubble_damping_constant(f) == pytest.approx(d97_damping(f), rel=1e-12)
    assert surface.BUBBLE_RERADIATION_DAMPING == DELTA_R


def test_resonant_radius_against_both_papers():
    """[McD93] eq. (10): a_r = 3.2 / f; [D97] p. 2587: a_R about 3.25 / f (m, f in Hz). Our Minnaert radius at
    the surface is 47.7 micrometres at 68 kHz against 47.1 and 47.8: within 1.5 % of both. [D97] Table II
    prints the resonant radii at the surface: 121, 64, 46 micrometres at 26.5, 50, 70 kHz."""
    for f in (12e3, 26.5e3, 50e3, 68e3, 70e3):
        ours = float(surface.resonant_radius(f))
        assert ours == pytest.approx(3.2 / f, rel=0.015)
        assert ours == pytest.approx(3.25 / f, rel=0.015)
    for f, printed_um in ((26.5e3, 121.0), (50e3, 64.0), (70e3, 46.0)):
        assert float(surface.resonant_radius(f)) * 1e6 == pytest.approx(printed_um, rel=0.015)     # ours 122.4, 64.9, 46.3


def test_saturation_level_against_d97_table_i():
    """[D97] Table I (p. 2589): the levels the data were forced to (eq. (18) at 17 degrees, 11-13 m/s):
    -26.9 / -27.8 / -28.3 dB at 26.5 / 50 / 70 kHz. Ours: -26.82 / -27.74 / -28.22 dB, 0.06 to 0.08 dB
    higher; the note measured "within 0.1 dB". Table I is printed to 0.1 dB, so 0.1 dB is the bound."""
    t = math.radians(17.0)
    for f, printed in ((26.5e3, -26.9), (50e3, -27.8), (70e3, -28.3)):
        ours = float(surface.bubble_saturation_db(t, f))
        assert ours == pytest.approx(10.0 * math.log10(d97_eq18(t, f)), abs=1e-9)
        assert abs(ours - printed) < 0.1, (f, ours, printed)
        assert ours > printed                                     # measured +0.08 / +0.06 / +0.08 dB


def test_bubble_term_is_d97_eq17_and_mcdaniel_eq9(set_optical_depth):
    """[D97] eq. (17) and [McD93] eq. (9) are one formula (q = beta_I / sin t with
    beta_I = 2 C pi^2 a_r^3 N / delta_r, [McD93] eq. (10)); `bubble_backscatter` is that formula,
    written with expm1 so that small b keeps its leading term. Checked against both, typed here."""
    theta = np.radians([2.0, 5.0, 17.0, 30.0, 45.0, 60.0, 80.0, 90.0])
    for f in (26.5e3, 68e3):
        for beta_i in (1e-6, 1e-3, 0.05, 0.6, 3.0, 40.0):
            set_optical_depth(beta_i)
            ours = surface.bubble_backscatter(theta, 5.0, f)
            np.testing.assert_allclose(ours, d97_eq17(theta, beta_i, f), rtol=1e-9)
            n_density = beta_i * DELTA_R / (2.0 * math.pi**2 * (3.2 / f) ** 3)      # eq. (10) solved for N, C = 1
            np.testing.assert_allclose(ours, mcd93_eq9(theta, n_density, f), rtol=1e-9)


def test_bubble_term_limits_d97_eq18_eq19(set_optical_depth):
    """[D97] eq. (18) (beta_I >~ 1, theta <~ 30 degrees: saturation) and eq. (19) (beta_I << 1: weak
    layer, independent of the angle) are the two limits of our bubble term."""
    f = 50e3
    low = np.radians([5.0, 10.0, 17.0, 30.0])
    set_optical_depth(20.0)
    np.testing.assert_allclose(surface.bubble_backscatter(low, 5.0, f), d97_eq18(low, f), rtol=1e-9)
    every = np.radians([5.0, 17.0, 45.0, 80.0])
    set_optical_depth(1e-7)
    np.testing.assert_allclose(surface.bubble_backscatter(every, 5.0, f), d97_eq19(1e-7, f), rtol=1e-5)
    # [D97] eq. (20) inverts the same weak-layer law to the resonant density N_I = beta_I delta_r / (2 pi^2 a_R^3);
    # with McDaniel's eq. (11), SS = 3 C pi a_r^3 N / delta, it gives eq. (19) again (C = 1).
    a_r = 3.2 / f
    n_i = 1e-7 * DELTA_R / (2.0 * math.pi**2 * a_r**3)
    assert 3.0 * math.pi * a_r**3 * n_i / d97_damping(f) == pytest.approx(d97_eq19(1e-7, f), rel=1e-12)


def test_overshoot_before_saturation_is_mcdaniels_3_to_4_db(set_optical_depth):
    """[McD93] Fig. 6 (p. 1908): the level overshoots saturation by 3-4 dB on the way there. Our bracket
    peaks at 2.36 times the saturation level (+3.73 dB) at an optical depth along the ray of about 0.6."""
    f, t = 30e3, math.radians(17.0)
    betas = np.geomspace(1e-3, 30.0, 4001) * math.sin(t)
    ratio = []
    for beta_i in betas:
        set_optical_depth(float(beta_i))
        ratio.append(float(surface.bubble_backscatter(t, 5.0, f)) / float(d97_eq18(t, f)))
    peak_db = 10.0 * math.log10(max(ratio))
    assert 3.0 < peak_db < 4.0
    assert peak_db == pytest.approx(3.73, abs=0.01)
    assert float(betas[int(np.argmax(ratio))]) / math.sin(t) == pytest.approx(0.6, abs=0.05)


def test_mcdaniels_damping_law_against_ours():
    """Characterisation. [McD93] eq. (10) uses delta = delta_r + 3.8e-4 f^(1/2) (f in Hz) instead of
    [D97]'s 2.55e-3 f^(1/3). The saturation level with hers minus ours: +0.30, +0.24, +0.05, -0.01, -0.21,
    -0.35, -0.62, -1.19 dB at 8, 12, 25, 30, 50, 68, 120, 330 kHz (the note's table, reproduced to 0.01 dB).
    Ours is [D97]'s; the two damping laws differ by less than 0.4 dB in 8-70 kHz, the band of the data."""
    for f_khz, want in ((8, 0.30), (12, 0.24), (25, 0.05), (30, -0.01), (50, -0.21), (68, -0.35), (120, -0.62), (330, -1.19)):
        f = f_khz * 1e3
        mcd = DELTA_R + 3.8e-4 * math.sqrt(f)
        assert 10.0 * math.log10(surface.bubble_damping_constant(f) / mcd) == pytest.approx(want, abs=0.01), f_khz


def test_optical_depth_convention_is_d97_section_vi():
    """[D97] Sec. VI: the two-way travel through the layer reduces a term by 20 log10(e) beta_I / sin t dB,
    i.e. beta_I is the optical depth (nepers of intensity) of one vertical crossing. Our
    `bubble_layer_optical_depth` is taken from a per-bounce loss law by the same convention, for both laws."""
    for u, f in ((6.0, 26.5e3), (10.0, 50e3), (15.0, 70e3)):
        for name, loss in (("apl_1994", surface.apl_1994_bubble_loss_db), ("hall_novarini", surface.bubble_loss_db)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")              # hall_novarini warns above 10 kHz (an upper estimate)
                beta_i = surface.bubble_layer_optical_depth(u, f, extinction=name)
                for deg in (5.0, 17.0, 45.0):
                    t = math.radians(deg)
                    assert 20.0 * math.log10(math.e) * beta_i / math.sin(t) == pytest.approx(float(loss(t, u, f)), rel=1e-9)


def test_facet_term_is_mcdaniel_eq7():
    """[McD93] eq. (7) (Kirchhoff, stationary phase, isotropic surface) with sigma_x'^2 = mss / 2, the
    x component of our total mean-square slope (`acoustics.mean_square_slope`), equals our facet term
    (Gallaudet's eq. (8) form) to 1e-9 relative at 2-15 m/s and 5-89 degrees (where both underflow
    to zero at the lowest angles and winds, both are zero)."""
    theta = np.radians(np.arange(5.0, 89.5, 1.0))
    for u in (2.0, 3.0, 5.0, 10.0, 15.0):
        mss = acoustics.mean_square_slope(u)
        np.testing.assert_allclose(surface.facet_backscatter(theta, u), mcd93_eq7(theta, 0.5 * mss), rtol=1e-9, atol=0.0)


def test_chapman_harris_is_mcdaniel_eq27():
    """`acoustics.chapman_harris_db` is [McD93] eq. (27) (Chapman and Harris 1962 as McDaniel reprints it;
    via secondary). With our knot constant (1.943844) the difference is below 1e-9 dB; with the exact knot
    (3600/1852 = 1.9438445) below 1e-6 relative (the constant is rounded at the seventh figure; 2e-5 dB at
    the extreme 2 degrees, 1 m/s, 400 Hz, where the law gives -193 dB)."""
    for th, u, f in ((2.0, 1.0, 400.0), (10.0, 5.0, 1000.0), (20.0, 2.0, 400.0), (30.0, 10.0, 3000.0),
                     (5.0, 15.0, 6400.0), (40.0, 8.0, 2000.0)):
        ours = float(acoustics.chapman_harris_db(math.radians(th), u, f))
        assert ours == pytest.approx(mcd93_eq27(th, u * acoustics.KNOTS_PER_MS, f), abs=1e-9)
        assert ours == pytest.approx(mcd93_eq27(th, u * KNOTS_PER_MS_EXACT, f), rel=1e-6)


def test_chapman_harris_exponents_at_30_degrees():
    """[McD93] p. 1918: at 30 degrees Chapman-Harris goes as wind^2.5 and frequency^0.8. Exactly, at 30
    degrees the angle term vanishes and SS = -42.4 log(beta) + 2.6, so intensity ~ v^(42.4 x 0.58 / 10)
    = v^2.459 and f^(2.459 / 3) = f^0.820."""
    t = math.radians(30.0)
    n_wind = (float(acoustics.chapman_harris_db(t, 10.0, 2000.0)) - float(acoustics.chapman_harris_db(t, 5.0, 2000.0))) / (10 * math.log10(2.0))
    n_freq = (float(acoustics.chapman_harris_db(t, 8.0, 3200.0)) - float(acoustics.chapman_harris_db(t, 8.0, 800.0))) / (10 * math.log10(4.0))
    assert n_wind == pytest.approx(42.4 * 0.58 / 10.0, abs=1e-9) and n_wind == pytest.approx(2.46, abs=0.005)
    assert n_freq == pytest.approx(42.4 * 0.58 / 30.0, abs=1e-9) and n_freq == pytest.approx(0.82, abs=0.005)


def test_dp97_mean_of_db_against_db_of_mean():
    """Characterisation of [DP97] eqs. (5)-(7). Within a 10 s window the intensity is exponential, so the
    mean of s = 10 log10(sigma) lies C_E k = 2.507 dB below 10 log10 of the mean intensity
    (k = 10 log10 e, C_E Euler's constant: eq. (6)) and the standard deviation of s is
    k pi / sqrt(6) = 5.57 dB (eq. (7) with sigma_T = 0). This is the size of the bias if a band mean
    is a mean of dB values rather than of intensities (docs/references, note Sec. 3.4)."""
    k = 10.0 * math.log10(math.e)
    assert 0.5772156649 * k == pytest.approx(2.507, abs=0.001)
    assert k * math.pi / math.sqrt(6.0) == pytest.approx(5.57, abs=0.005)
    rng = np.random.default_rng(1997)
    s = 10.0 * np.log10(rng.exponential(1.0, 200_000))
    assert float(np.mean(s)) == pytest.approx(-0.5772156649 * k, abs=0.05)
    assert float(np.std(s)) == pytest.approx(k * math.pi / math.sqrt(6.0), abs=0.05)
