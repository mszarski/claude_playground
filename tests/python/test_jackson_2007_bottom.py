"""The fluid-sediment bottom model (sim/bottom.py) against Jackson and Richardson, "High-Frequency
Seafloor Acoustics", Springer (2007) [J&R], as the repository's notes give its formulas
(docs/references/jackson_richardson_2007/01 to 04, cited [N1] to [N4] with their sections).

Every book formula is typed here again, in the book's notation and with its equation number,
independently of bottom.py, and our function is compared with it at the bounds the notes
measured. Numbers the book PRINTS are marked "printed"; numbers the notes' scripts computed and
that our code reproduces are marked "the note's computation"; values read off a figure by eye
are marked "figure, by eye". Record only (docs/tasks/SST_GAPS_JACKSON.md G-1): no formula of
bottom.py is changed by this file.

Conventions [N4 2.0], [N3 2.1]: exp(-i w t), Im k_p > 0; grazing angles theta from the
horizontal; a_rho = rho / rho_w; a_p = c_p / c_w = nu_p / (1 + i delta_p) (8.25); our azimuth is
the book's phi_s with phi_i = 0 (pi = backscatter).
"""
import math

import numpy as np
import pytest
from scipy.special import j0

from gpusonar import acoustics
from gpusonar.sim import bottom

# Fig. 13.2 (p. 337), "typical of a medium sand seafloor", printed [N4 2.5, 3.1]: a_rho, nu_p,
# delta_p, w2 (m^0.75), gamma2. The figure is at 30 kHz; c_w = 1500 m/s reproduces the notes'
# K_c = 20.93 rad/m.
FIG_13_2 = dict(rho=1.845, nu=1.1782, delta=0.01624, w2=0.000141, gamma2=3.25)
C_W = 1500.0


def wavenumber(f, c=C_W):
    return 2.0 * math.pi * f / c


# --- The book's formulas, retyped ---------------------------------------------------------------

def sin_theta_p(a_p, theta):
    """(8.46): sin theta_p = sqrt(1 - a_p^2 cos^2 theta_w), the root with Im >= 0 (p. 250)."""
    s = np.sqrt(1.0 - a_p**2 * np.cos(theta) ** 2 + 0j)
    return np.where(s.imag < 0.0, -s, s)


def v_ww(theta, a_rho, nu_p, delta_p):
    """(8.49)-(8.53): V_ww = (z_wp - 1) / (z_wp + 1), z_wp = a_rho a_p sin theta_w / sin theta_p."""
    a_p = nu_p / (1.0 + 1j * delta_p)                                        # (8.25)
    z_wp = a_rho * a_p * np.sin(theta) / sin_theta_p(a_p, theta)             # (8.53)
    return (z_wp - 1.0) / (z_wp + 1.0)                                       # (8.49)


def a_ww(theta_i, theta_s, phi_s, a_rho, nu_p, delta_p):
    """(13.21)-(13.25), p. 340: the fluid perturbation factor A_ww."""
    a_p = nu_p / (1.0 + 1j * delta_p)
    b = sin_theta_p(a_p, theta_i) * sin_theta_p(a_p, theta_s) / (a_p**2 * a_rho)                       # (13.23)
    g = (1.0 - 1.0 / a_rho) * (np.cos(theta_i) * np.cos(theta_s) * np.cos(phi_s) - b) - 1.0 + 1.0 / (a_p**2 * a_rho)  # (13.22)
    return 0.5 * (1.0 + v_ww(theta_i, a_rho, nu_p, delta_p)) * (1.0 + v_ww(theta_s, a_rho, nu_p, delta_p)) * g  # (13.21)


def differences(theta_i, theta_s, phi_s, k_w):
    """(13.5)-(13.8): |Delta K| (horizontal), Delta k_z, Delta k^2."""
    d_big_k = k_w * math.sqrt(max(math.cos(theta_i) ** 2 + math.cos(theta_s) ** 2
                                  - 2.0 * math.cos(theta_i) * math.cos(theta_s) * math.cos(phi_s), 0.0))
    d_kz = k_w * (math.sin(theta_i) + math.sin(theta_s))
    return d_big_k, d_kz, d_big_k**2 + d_kz**2


def c_h2(w2, gamma2):
    """(D.11)-(D.13), p. 477: C_h^2 = 2 pi w2 Gamma(2 - alpha) 2^(-2 alpha) / (alpha (1 - alpha) Gamma(1 + alpha)),
    alpha = gamma2 / 2 - 1."""
    alpha = 0.5 * gamma2 - 1.0
    return 2.0 * math.pi * w2 * math.gamma(2.0 - alpha) * 2.0 ** (-2.0 * alpha) / (alpha * (1.0 - alpha) * math.gamma(1.0 + alpha)), alpha


def i_k(q, alpha):
    """(13.16)-(13.17): I_K = integral_0^inf J0(u) exp(-q u^(2 alpha)) u du, by Gauss-Legendre panels of
    length pi on the real axis out to exp(-46) (an independent quadrature, not ours)."""
    end = (46.0 / q) ** (1.0 / (2.0 * alpha))
    n = int(math.ceil(end / math.pi)) + 1
    x, w = np.polynomial.legendre.leggauss(20)
    mid = (np.arange(n) + 0.5) * math.pi
    u = (mid[:, None] + 0.5 * math.pi * x).ravel()
    return float((np.tile(0.5 * math.pi * w, n) * j0(u) * np.exp(-q * u ** (2.0 * alpha)) * u).sum())


def book_cross_sections(theta_i, theta_s, phi_s, f, par):
    """Perturbation (13.9), Kirchhoff (13.10) with (13.12)-(13.13), (13.16)-(13.17), small-slope (13.11)."""
    k_w = wavenumber(f)
    d_big_k, d_kz, d_k2 = differences(theta_i, theta_s, phi_s, k_w)
    amp2 = abs(a_ww(theta_i, theta_s, phi_s, par["rho"], par["nu"], par["delta"])) ** 2
    pert = k_w**4 * amp2 * par["w2"] / d_big_k ** par["gamma2"]                                        # (13.9), (6.8)
    ch2, alpha = c_h2(par["w2"], par["gamma2"])
    ik = i_k(0.5 * ch2 * d_kz**2 * d_big_k ** (-2.0 * alpha), alpha)                                    # (13.17)
    theta_is = math.asin(min(math.sqrt(d_k2) / (2.0 * k_w), 1.0))                                      # (13.13)
    kirch = abs(v_ww(theta_is, par["rho"], par["nu"], par["delta"])) ** 2 / (8.0 * math.pi) * (d_k2 / (d_big_k * d_kz)) ** 2 * ik  # (13.10)
    ssa = k_w**4 * amp2 * ik / (2.0 * math.pi * d_big_k**2 * d_kz**2)                                 # (13.11)
    return pert, kirch, ssa


def book_volume(theta_i, theta_s, f, a_rho, nu_p, delta_p, sigma_2):
    """(14.8), (14.7), (14.9), (8.26): sigma = |V_wp(i)|^2 |V_wp(s)|^2 sigma_v / (2 k_w |a_rho|^2 Im[P(i) + P(s)]),
    P = sqrt(a_p^-2 - cos^2 theta), V_wp = 1 + V_ww (8.47), sigma_v = sigma_2 alpha_p,
    alpha_p = 40 pi f delta_p / (nu_p c_w ln 10) dB/m."""
    a_p = nu_p / (1.0 + 1j * delta_p)

    def p(theta):
        root = np.sqrt(a_p**-2 - np.cos(theta) ** 2 + 0j)
        return np.where(root.imag < 0.0, -root, root)
    alpha_p = 40.0 * math.pi * f * delta_p / (nu_p * C_W * math.log(10.0))
    sigma_v = sigma_2 * alpha_p
    v_wp_i, v_wp_s = 1.0 + v_ww(theta_i, a_rho, nu_p, delta_p), 1.0 + v_ww(theta_s, a_rho, nu_p, delta_p)
    return np.abs(v_wp_i) ** 2 * np.abs(v_wp_s) ** 2 * sigma_v / (2.0 * wavenumber(f) * a_rho**2 * (p(theta_i) + p(theta_s)).imag)


# The (11.1)-(11.3) relations as J&R pp. 313-314 print them [N1 2.4], [N3 2.5], with the two
# constants the notes judge misprinted given both ways (docs/tasks/SST_GAPS_JACKSON.md 0.4).
NU_LAST_PRINTED, NU_LAST_CONTINUOUS = 1.0091, 1.0019          # (11.2), Mz >= 5.3
K_FIRST_PRINTED, K_FIRST_CONTINUOUS = 0.4566, 0.4556          # (11.3), -1 <= Mz < 0


def a_rho_11_1(mz):
    if mz < 1.0:
        return 0.007797 * mz**2 - 0.17057 * mz + 2.3139
    if mz < 5.3:
        return -0.0165406 * mz**3 + 0.2290201 * mz**2 - 1.1069031 * mz + 3.0455
    return -0.0012973 * mz + 1.1565


def nu_p_11_2(mz, last=NU_LAST_CONTINUOUS):
    if mz < 1.0:
        return 0.002709 * mz**2 - 0.056452 * mz + 1.2778
    if mz < 5.3:
        return -0.0014881 * mz**3 + 0.0213937 * mz**2 - 0.1382798 * mz + 1.3425
    return -0.0024324 * mz + last


def k_11_3(mz, first=K_FIRST_CONTINUOUS):
    """dB m^-1 kHz^-1."""
    if mz < 0.0:
        return first
    if mz < 2.6:
        return 0.0245 * mz + 0.4556
    if mz < 4.5:
        return 0.1245 * mz + 0.1978
    if mz < 6.0:
        return 0.20098 * mz**2 - 2.5228 * mz + 8.0399
    if mz < 9.5:
        return 0.0117 * mz**2 - 0.2041 * mz + 0.9431
    return 0.0601


def delta_from_k(k, nu_p, c_w=1528.0):
    """(2.6) with alpha_p = k f_kHz: delta_p = k nu_p c_w ln 10 / (40 000 pi). c_w = 1528 m/s is the
    water the table was built with ([N1 3.1]: it reproduces 0.01624 at 1.5 phi)."""
    return k * nu_p * c_w * math.log(10.0) / (40000.0 * math.pi)


# --- Conventions and conversions ----------------------------------------------------------------

def test_loss_parameter_conversions_are_the_books():
    """(2.6)/(8.26) dB/m <-> delta_p and (8.27) dB per wavelength = 40 pi delta_p / ln 10 [N1 2.1], [N3 3.1].
    The note's computation: 10 dB/m at 20 kHz in 1767 m/s sand is delta 0.016189 and 0.8835 dB per wavelength."""
    assert bottom.DB_PER_WAVELENGTH == pytest.approx(40.0 * math.pi / math.log(10.0), rel=1e-15)
    assert bottom.DB_PER_WAVELENGTH == pytest.approx(54.58, abs=0.005)                      # printed: 54.58 delta_p
    par = bottom.parameters_from_physical(density=1845.0, sound_speed=1767.0, attenuation_db_per_m=10.0,
                                          attenuation_frequency=20e3, w2=1.41e-4, gamma2=3.25, volume_strength_db=-20.0,
                                          water_density=1000.0, water_sound_speed=1500.0)
    # (2.6): delta_p = alpha_p nu_p c_w ln(10) / (40 pi f), nu_p c_w = the sediment's phase speed.
    assert par["delta"] == pytest.approx(10.0 * 1767.0 * math.log(10.0) / (40.0 * math.pi * 20e3), rel=1e-13)
    assert par["delta"] == pytest.approx(0.016189, abs=5e-7)
    assert par["delta"] * bottom.DB_PER_WAVELENGTH == pytest.approx(0.8835, abs=5e-5)
    # (8.26) back again: alpha_p = 40 pi f delta_p / (nu_p c_w ln 10) = 10 dB/m.
    assert 40.0 * math.pi * 20e3 * par["delta"] / (par["nu"] * 1500.0 * math.log(10.0)) == pytest.approx(10.0, rel=1e-13)
    # (14.9): sigma_2 = sigma_v / alpha_p, alpha_p in dB/m: -20 dB/m over 10 dB/m.
    assert par["sigma2"] == pytest.approx(10.0 ** (-2.0) / 10.0, rel=1e-13)
    # (2.23), p. 26: Lambert mu < 1/pi, -4.97 dB (a third print of the bound).
    assert bottom.LAMBERT_MAX_MU_DB == pytest.approx(10.0 * math.log10(1.0 / math.pi), rel=1e-15)
    assert bottom.LAMBERT_MAX_MU_DB == pytest.approx(-4.97, abs=0.005)


def test_roughness_spectrum_conventions_structure_constant_and_unit_conversion():
    """(D.12) as printed against `structure_constant` ([N2 3.1]: equal to 2e-16; [N4 2.0]: 1.5757e-3 for
    w2 1.41e-4, gamma2 3.25), and D.1.3 (p. 479) against `w2_from_cm4`: in the reference-length form
    W = w2 / (h0 K)^gamma2 a cm^4 strength is divided by 1e8 for m^4; the plain form's cm^(4 - gamma2)
    strength is divided by 100^(4 - gamma2)."""
    for gamma2 in (2.5, 3.0, 3.25, 3.5, 3.9):
        book, alpha = c_h2(1.0, gamma2)
        ours, a = bottom.structure_constant(1.0, gamma2)
        assert a == alpha and ours == pytest.approx(book, rel=1e-14), gamma2
    assert bottom.structure_constant(0.000141, 3.25)[0] == pytest.approx(1.5757e-3, abs=5e-8)       # the note's computation
    for gamma2 in (3.0, 3.25, 3.5):
        beta = 0.0035
        plain_cm = beta * 1.0 ** (-gamma2)                     # W = beta (h0 K)^-gamma2 with h0 = 1 cm: w2 = beta h0^-gamma2 cm^(4 - gamma2)
        assert bottom.w2_from_cm4(beta, gamma2) == pytest.approx(plain_cm / 100.0 ** (4.0 - gamma2), rel=1e-13)
        h0 = 0.01                                              # the same in metres: beta 1e-8 m^4, h0 = 0.01 m
        assert bottom.w2_from_cm4(beta, gamma2) == pytest.approx(beta * 1e-8 * h0 ** (-gamma2), rel=1e-13)


# --- Flat-interface reflection (Ch. 8) ----------------------------------------------------------

def test_flat_interface_reflection_is_8_49_to_8_53():
    """[N3 3.1]: ours equal to the book's V_ww to 1.8e-15 over 0.5-90 deg for four sediments."""
    theta = np.radians(np.linspace(0.5, 90.0, 400))
    for a_rho, nu_p, delta_p in ((1.845, 1.1782, 0.01624), (2.0, 1.1, 0.01), (1.5, 0.98, 0.01), (2.5, 2.5, 0.01374)):
        assert np.max(np.abs(bottom.reflection(theta, a_rho, nu_p, delta_p) - v_ww(theta, a_rho, nu_p, delta_p))) < 1e-13


def test_fig_8_3_critical_and_intromission_angles():
    """Fig. 8.3 (lossless): sand nu_p 1.1, a_rho 2.0: critical angle 24.6 deg, printed (8.56); mud nu_p 0.98,
    a_rho 1.5: intromission angle about 10.5 deg, printed (8.57); the note's computation 10.46 deg and
    |V| 0.0009 at 10.5 deg [N3 2.2.3]. Also Fig. 14.2's critical angles about 18 and 25 deg (printed),
    17.8 and 24.6 deg computed (nu_p 1.05, 1.1)."""
    crit = math.degrees(math.acos(1.0 / 1.1))
    assert crit == pytest.approx(24.6, abs=0.05)
    below, above = np.radians([crit - 0.5, crit - 5.0, 5.0]), np.radians([crit + 0.5, crit + 5.0])
    assert np.allclose(np.abs(bottom.reflection(below, 2.0, 1.1, 0.0)), 1.0, atol=1e-14)
    assert np.all(np.abs(bottom.reflection(above, 2.0, 1.1, 0.0)) < 0.99)
    intromission = math.degrees(math.acos(math.sqrt((1.5**2 - 1.0 / 0.98**2) / (1.5**2 - 1.0))))        # (8.57)
    assert intromission == pytest.approx(10.46, abs=0.005)
    assert abs(bottom.reflection(math.radians(intromission), 1.5, 0.98, 0.0)) < 1e-12
    assert abs(bottom.reflection(math.radians(10.5), 1.5, 0.98, 0.0)) == pytest.approx(0.0009, abs=0.0001)
    assert [round(math.degrees(math.acos(1.0 / nu)), 1) for nu in (1.05, 1.1)] == [17.8, 24.6]


def test_coherent_reflection_is_13_14():
    """(13.14), p. 335: V_wwc = V_ww exp(-2 k_w^2 h^2 sin^2 theta) (Eckart); the note's computation 0.24569 at
    30 deg, 40 kHz, h 1 cm [N3 3.1]."""
    g = np.radians([5.0, 30.0, 60.0, 90.0])
    k_w = wavenumber(40e3)
    assert np.allclose(acoustics.rough_coherent_reflection(g, 0.01, 40e3, C_W), np.exp(-2.0 * k_w**2 * 0.01**2 * np.sin(g) ** 2), rtol=1e-13)
    assert acoustics.rough_coherent_reflection(math.radians(30.0), 0.01, 40e3, C_W) == pytest.approx(0.24569, abs=5e-6)


# --- Roughness scattering (Ch. 13) ---------------------------------------------------------------

GEOMETRIES = ([(t, t, 180.0) for t in (5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 70.0, 80.0, 85.0)]
              + [(20.0, 20.0, 90.0), (20.0, 20.0, 30.0), (20.0, 40.0, 0.0), (20.0, 10.0, 0.0), (20.0, 60.0, 120.0),
                 (20.0, 20.0, 1.0)])
"""Backscatter 5-85 deg and the notes' bistatic geometries at theta_i = 20 deg [N4 3.3] (azimuth in deg)."""


@pytest.mark.parametrize("f", [30e3, 100e3])
def test_perturbation_kirchhoff_and_small_slope_are_the_books(f):
    """(13.9) with (13.21)-(13.25); (13.10) with (13.12)-(13.13), (13.16)-(13.17); (13.11), at the Fig. 13.2
    inputs [N4 3.1-3.3]: the notes found < 0.001 dB. Measured here: perturbation to 1e-13, Kirchhoff and
    small-slope to 2e-7 (our I_K is a spline of the integral along a turned path)."""
    for deg_i, deg_s, deg_phi in GEOMETRIES:
        ti, ts, phi = math.radians(deg_i), math.radians(deg_s), math.radians(deg_phi)
        pert, kirch, ssa = book_cross_sections(ti, ts, phi, f, FIG_13_2)
        args = (ti, ts, f, C_W)
        assert float(bottom.perturbation_strength(*args, azimuth=phi, **FIG_13_2)) == pytest.approx(pert, rel=1e-10)
        assert float(bottom.kirchhoff_strength(*args, azimuth=phi, **FIG_13_2)) == pytest.approx(kirch, rel=2e-6)
        assert float(bottom.small_slope_strength(*args, azimuth=phi, **FIG_13_2)) == pytest.approx(ssa, rel=2e-6)


def test_perturbation_factor_identity_k_5_and_the_frequency_law_13_61():
    """(K.5) = (L.9), p. 530: A_ww(K_i, K_i) = -2 beta_w^2(K_i) V_ww(K_i), "a useful test of numerical
    algorithms" (the identity behind small-slope = Kirchhoff in the specular direction). (13.61), p. 351:
    with nu_p, delta_p independent of frequency sigma is proportional to f^(4 - gamma2) (7.5 dB per decade
    at gamma2 3.25)."""
    theta = np.radians([5.0, 20.0, 45.0, 80.0])
    a_spec = a_ww(theta, theta, 0.0, FIG_13_2["rho"], FIG_13_2["nu"], FIG_13_2["delta"])
    assert np.allclose(a_spec, -2.0 * np.sin(theta) ** 2 * v_ww(theta, FIG_13_2["rho"], FIG_13_2["nu"], FIG_13_2["delta"]), rtol=1e-12)
    ti, ts, phi = np.radians([10.0, 30.0, 20.0]), np.radians([10.0, 30.0, 50.0]), np.radians([180.0, 180.0, 70.0])
    for gamma2 in (3.0, 3.25, 3.6):
        par = dict(FIG_13_2, gamma2=gamma2)
        lo = bottom.perturbation_strength(ti, ts, 20e3, C_W, azimuth=phi, **par)
        hi = bottom.perturbation_strength(ti, ts, 200e3, C_W, azimuth=phi, **par)
        assert np.allclose(hi / lo, 10.0 ** (4.0 - gamma2), rtol=1e-12)
    assert 10.0 * (4.0 - 3.25) == pytest.approx(7.5)


def test_composite_cut_off_and_large_scale_slope_13_72_to_13_77():
    """(13.72) k_w h_s = 1/2 [D. Jackson et al. 1986a], (13.75) K_c = (8 pi w2 k_w^2 / (gamma2 - 2))^(1/(gamma2 - 2)),
    (13.77) sigma_sL^2 = pi w2 K_c^(4 - gamma2) / (4 - gamma2) per axis. Printed: Fig. 13.19 (p. 368), medium
    sand at 20 kHz, w2 = 0.0000422 / 0.0001406 / 0.0004218: rms slopes 0.0227 / 0.0595 / 0.1432. The note's
    computation: K_c 20.93 rad/m and sigma_sL 0.0760 at 30 kHz (Fig. 13.2 inputs), 143.7 rad/m and 0.157 at
    100 kHz [N4 3.2]."""
    for f in (20e3, 30e3, 100e3):
        for w2 in (0.0000422, 0.000141, 0.0004218):
            k_w = wavenumber(f)
            k_c = (8.0 * math.pi * w2 * k_w**2 / (3.25 - 2.0)) ** (1.0 / (3.25 - 2.0))                    # (13.75)
            slope = math.sqrt(math.pi * w2 * k_c ** (4.0 - 3.25) / (4.0 - 3.25))                          # (13.77)
            h_s2 = 2.0 * math.pi * w2 / ((3.25 - 2.0) * k_c ** (3.25 - 2.0))                              # (13.74)
            assert k_w * math.sqrt(h_s2) == pytest.approx(0.5, rel=1e-12)                                 # (13.72)
            ours = bottom.large_scale_slope(f, C_W, w2=w2, gamma2=3.25)
            assert ours[0] == pytest.approx(slope, rel=1e-12) and ours[1] == pytest.approx(k_c, rel=1e-12)
    printed = [round(bottom.large_scale_slope(20e3, C_W, w2=w2, gamma2=3.25)[0], 4) for w2 in (0.0000422, 0.0001406, 0.0004218)]
    assert printed == [0.0227, 0.0595, 0.1432]
    slope, k_c = bottom.large_scale_slope(30e3, C_W, **FIG_13_2)
    assert (round(k_c, 2), round(slope, 4)) == (20.93, 0.0760)
    slope, k_c = bottom.large_scale_slope(100e3, C_W, **FIG_13_2)
    assert (round(k_c, 1), round(slope, 3)) == (143.7, 0.157)


# --- Volume scattering (Ch. 14) -----------------------------------------------------------------

def test_volume_term_is_14_8_with_14_9():
    """(14.8), (14.9), (8.26): equal to the note's 1e-13 dB at the Fig. 14.2 case [N2 3.3], and at the
    Fig. 13.2 sand, backscatter and bistatic (our volume term has no azimuth: isotropic sigma_v, [N4 3.6])."""
    g = np.radians([2.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0])
    for f in (30e3, 100e3):
        for a_rho, nu_p, delta_p, sigma_2 in ((1.5, 1.05, 0.015, 0.002), (1.845, 1.1782, 0.01624, 0.001), (1.147, 0.984, 0.0019, 0.001)):
            ours = bottom.volume_strength(g, g[::-1], f, C_W, rho=a_rho, nu=nu_p, delta=delta_p, sigma2=sigma_2)
            assert np.allclose(ours, book_volume(g, g[::-1], f, a_rho, nu_p, delta_p, sigma_2), rtol=1e-11)
    # The note's computation at the Fig. 14.2 case (nu_p 1.05): -35.318 / -26.293 / -24.227 dB at 10 / 30 / 60 deg.
    db = 10.0 * np.log10(bottom.volume_strength(np.radians([10.0, 30.0, 60.0]), np.radians([10.0, 30.0, 60.0]), 30e3, C_W,
                                                rho=1.5, nu=1.05, delta=0.015, sigma2=0.002))
    assert np.allclose(db, [-35.318, -26.293, -24.227], atol=6e-4)


def test_fig_14_2_volume_curves_by_eye():
    """Fig. 14.2 (p. 382): 30 kHz, c_w 1500, a_rho 1.5, delta_p 0.015, sigma_2 0.002, nu_p 0.98 / 1.05 / 1.1.
    The note's computation: -24.1 / -23.6 / -23.3 dB at 90 deg, -27.3 / -26.3 / -25.7 at 30 deg,
    -33.4 / -35.3 / -40.2 at 10 deg; the figure, by eye [N1 3.1], [N2 3.3]: about -24 to -23.5 at 90 deg,
    -27 to -26 at 30 deg, -33 / -36 / -40 at 10 deg. Held within 1 dB of the readings (figure accuracy,
    docs/tasks/SST_GAPS_JACKSON.md section 6), and to 0.05 dB of the note's numbers."""
    g = np.radians([10.0, 30.0, 90.0])
    got = {nu: 10.0 * np.log10(bottom.volume_strength(g, g, 30e3, C_W, rho=1.5, nu=nu, delta=0.015, sigma2=0.002))
           for nu in (0.98, 1.05, 1.1)}
    note = {0.98: (-33.4, -27.3, -24.1), 1.05: (-35.3, -26.3, -23.6), 1.1: (-40.2, -25.7, -23.3)}
    read_10 = {0.98: -33.0, 1.05: -36.0, 1.1: -40.0}
    for nu, db in got.items():
        assert np.allclose(db, note[nu], atol=0.05), nu
        assert abs(db[0] - read_10[nu]) <= 1.0 and -27.0 - 1.0 <= db[1] <= -26.0 + 1.0 and -24.0 - 1.0 <= db[2] <= -23.5 + 1.0, nu
    # Suppressed below the critical angle: at 10 deg the order is mud > 1.05 > 1.1 (critical 17.8 / 24.6 deg).
    assert got[0.98][0] > got[1.05][0] > got[1.1][0]


# --- The default model as a whole, and its general properties ---------------------------------

# The note's computation of the book's small-slope + volume (14.8) model [N4 3.2], dB, at 5, 10, 20,
# 30, 40, 60, 70, 80, 85 deg (medium sand, our row's sigma_2 0.001) and 5 ... 85 deg (very fine silt, our
# row): our default (`small_slope` + `volume_strength`) equal to < 0.001 dB; printed there to 0.01 dB.
BOOK_SSA_VOL_MEDIUM_SAND = {
    30e3: ((5, 10, 20, 30, 40, 60, 70, 80, 85), (-47.48, -36.97, -28.85, -25.43, -25.15, -22.25, -18.29, -10.45, -5.57)),
    100e3: ((5, 10, 15, 20, 30, 40, 60, 70, 85), (-43.84, -33.33, -28.20, -25.19, -21.91, -22.57, -19.12, -15.46, -9.94))}
BOOK_SSA_VOL_VERY_FINE_SILT = {
    30e3: ((5, 10, 20, 30, 40, 60, 80, 85), (-42.03, -36.08, -31.95, -30.05, -28.86, -27.49, -26.34, -22.98)),
    100e3: ((5, 10, 20, 30, 40, 60, 80, 85), (-42.01, -36.06, -31.94, -30.04, -28.85, -27.45, -25.51, -20.33))}


def test_default_model_equals_the_books_small_slope_plus_volume():
    for table, par in ((BOOK_SSA_VOL_MEDIUM_SAND, dict(FIG_13_2, sigma2=0.001)),
                       (BOOK_SSA_VOL_VERY_FINE_SILT, bottom.sediment("very_fine_silt"))):
        for f, (deg, want) in table.items():
            got = bottom.backscattering_strength_db(np.radians(deg), f, C_W, **par)
            assert np.allclose(got, want, atol=0.0051), f                         # the note's two decimals


def test_medium_sand_row_is_fig_13_2():
    """`medium_sand` (configs/sediments.yaml) against Fig. 13.2's printed inputs [N4 3.1]: equal to the
    printed digits (our row's delta comes from a rounded 10 dB/m: 0.016189 against 0.01624)."""
    row = bottom.sediment("medium_sand")
    assert row["rho"] == pytest.approx(1.845, abs=5e-4) and row["nu"] == pytest.approx(1.1782, abs=5e-4)
    assert row["delta"] == pytest.approx(0.01624, rel=0.005) and row["w2"] == pytest.approx(0.000141, rel=1e-3)
    assert row["gamma2"] == 3.25


def test_small_slope_reciprocity_and_energy_at_the_books_sand():
    """(2.18), p. 24: sigma(i, s) = sigma(s, i). (2.20), p. 24: integral of sigma over the hemisphere +
    |V_wwc|^2 sin theta_i < sin theta_i, with V_wwc = 0 for a power-law relief (infinite h). Near specular
    (L.23), (L.26): (1 / sin theta_i) integral sigma dOmega = |V_wwi|^2 = |V_ww|^2 (1 - exp(-Delta k_z^2 h^2))
    -> |V_ww|^2. Lowest-order small-slope is not exactly energy conserving: measured 0.902 / 1.031 / 1.005
    times |V_ww|^2 at 20 / 40 / 70 deg (Fig. 13.2 inputs, 30 kHz)."""
    for ti, ts, phi in ((15.0, 40.0, 120.0), (5.0, 70.0, 20.0), (30.0, 30.0, 180.0), (50.0, 65.0, 5.0)):
        ti, ts, phi = np.radians([ti, ts, phi])
        a = bottom.small_slope_strength(ti, ts, 30e3, C_W, azimuth=phi, **FIG_13_2)
        b = bottom.small_slope_strength(ts, ti, 30e3, C_W, azimuth=phi, **FIG_13_2)
        assert a == pytest.approx(b, rel=1e-12) and a > 0.0
    n_t, n_p = 300, 180
    t = (np.arange(n_t) + 0.5) / n_t * (np.pi / 2)
    ph = (np.arange(n_p) + 0.5) / n_p * np.pi
    t, ph = np.meshgrid(t, ph, indexing="ij")
    for deg, lo, hi in ((20.0, 0.88, 0.92), (40.0, 1.01, 1.05), (70.0, 0.99, 1.02)):
        ti = math.radians(deg)
        s = bottom.small_slope_strength(ti, t, 30e3, C_W, azimuth=ph, **FIG_13_2)
        integral = 2.0 * (s * np.cos(t)).sum() * (np.pi / 2 / n_t) * (np.pi / n_p)       # dOmega = cos(theta_s) dtheta_s dphi_s
        assert integral < math.sin(ti)                                                   # (2.20)
        v2 = abs(v_ww(ti, FIG_13_2["rho"], FIG_13_2["nu"], FIG_13_2["delta"])) ** 2
        assert lo < integral / math.sin(ti) / v2 < hi, deg


# --- The grain-size table at its tabulated points (Ch. 11) ----------------------------------------

def test_grain_size_table_is_11_1_to_11_3_at_its_points():
    """`GRAIN_SIZE_TABLE` (Cintron 2001's print of the TR 9407 table) against (11.1)-(11.3) as J&R prints
    them, with delta_p by (2.6) and c_w 1528 m/s [N1 3.2], [N3 3.2]: rho within 4.5e-4, nu within 4.9e-5 and
    delta within 0.27 % (the notes), held at 5e-4, 1e-4 and 0.5 %. With the printed constants instead the
    table is missed: 1.0091 in (11.2) puts nu 0.0072 off at all seven rows from 5.5 to 9 phi; 0.4566 in
    (11.3) puts delta 0.2 % off at -1 and -0.5 phi, where 0.4556 is within 3e-4 (contradiction C3:
    the rows follow 0.4556)."""
    for mz, rho, nu, delta in bottom.GRAIN_SIZE_TABLE:
        assert a_rho_11_1(mz) == pytest.approx(rho, abs=5e-4), mz
        assert nu_p_11_2(mz) == pytest.approx(nu, abs=1e-4), mz
        assert delta_from_k(k_11_3(mz), nu_p_11_2(mz)) == pytest.approx(delta, rel=5e-3), mz
        if mz >= 5.5:
            assert abs(nu_p_11_2(mz, NU_LAST_PRINTED) - nu) == pytest.approx(0.0072, abs=1e-4), mz
        if mz < 0.0:
            assert delta_from_k(k_11_3(mz), nu_p_11_2(mz)) == pytest.approx(delta, rel=3e-4), mz
            assert abs(delta_from_k(k_11_3(mz, K_FIRST_PRINTED), nu_p_11_2(mz)) / delta - 1.0) > 2e-3, mz
    # The printed example the notes quote: 0.01709 with 0.4566 against the table's 0.01705 at -1 phi.
    assert round(delta_from_k(k_11_3(-1.0, K_FIRST_PRINTED), nu_p_11_2(-1.0)), 5) == 0.01709
    assert dict((r[0], r[3]) for r in bottom.GRAIN_SIZE_TABLE)[-1.0] == 0.01705
