"""Sub-critical penetration by roughness scattering (`penetration: rough`, G-13 of docs/tasks/SST_GAPS_JACKSON.md).

Jackson and Richardson (2007) Ch. 15 as retyped in
docs/references/jackson_richardson_2007/03_wave_theories_reflection_penetration.md ([N3]) Sec. 2.7: the zeroth-order
field (15.1), the first-order field (15.2)-(15.4), the sinusoidal ripple (15.5)-(15.10) and the averaged field
(15.11)-(15.16); the SAX99 inputs of Table 13.1 from 04_scattering_models.md ([N4]) Sec. 2.3 and the von Karman
spectrum (D.7) from 02_roughness_heterogeneity_statistics.md ([N2]) Sec. 2.1. Values read off the book's figures
are labelled `figure` and carry the reading's own width plus the note's 0.5-1.5 dB reading error, never less;
values we computed are labelled `ours`.
"""

import warnings

import numpy as np
import pytest

from gpusonar.config import load_scenario_config
from gpusonar.sim import SedimentProvider, engine, make_provider, run_scenario
from gpusonar.sim import bottom as bottom_model
from gpusonar.sim import penetration as pen
from gpusonar.sim.propagation import fluid_transmission, two_way_paths

LN10 = np.log(10.0)
# SAX99, Table 13.1 ([N4] Sec. 2.3): a_rho 2.00, nu 1.16, delta 0.010, gamma2 3.04, w2 0.0000433 m^(4 - gamma2)
# (units inferred, [N4] Sec. 6); K0 = 10 1/m and c_w = 1530 m/s are Fig. 15.8's ([N3] Sec. 2.7).
SAX99 = dict(a_rho=2.00, nu=1.16, delta=0.010)
SAX99_ROUGH = dict(w2=0.0000433, gamma2=3.04, K0=10.0)
C_SAX = 1530.0


def sax99_bottom(c=C_SAX):
    """SAX99 as the provider's bottom: speed nu c, density a_rho, attenuation in dB per wavelength ((8.27))."""
    return {"speed": SAX99["nu"] * c, "density": SAX99["a_rho"], "attenuation": SAX99["delta"] * 40 * np.pi / LN10}


def rough(c=C_SAX, f=20e3, rough_par=None, **kw):
    b = sax99_bottom(c)
    return SedimentProvider(c, b["speed"], 0.0, penetration="rough", bottom=b, frequency=f, engine="numpy",
                            roughness=dict(SAX99_ROUGH if rough_par is None else rough_par), **kw)


def fluid(c=C_SAX, f=20e3, **kw):
    b = sax99_bottom(c)
    return SedimentProvider(c, b["speed"], 0.0, penetration="fluid", bottom=b, frequency=f, **kw)


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **kw)


# --- the amplitude ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rho,nu,delta", [(2.0, 1.16, 0.010), (1.845, 1.178, 0.01624), (1.5, 0.98, 0.002)])
def test_our_perturbation_sediment_amplitude_is_a_wp(rho, nu, delta):
    """[N3] Sec. 3.5 item 4 / WP2 hint: the sediment-side amplitude B of the first-order solution in sim/bottom.py's
    docstring (boundary conditions moved to z = 0, solved here from the docstring's two equations) equals the book's
    T_wp^(1) / F = i k_w A_wp / beta_w(K_s) of (15.3) with A_wp of (15.4), to rounding: the same physics."""
    k = 1.7
    rng = np.random.default_rng(3)
    for _ in range(20):
        ti, ts, phi = rng.uniform(0.02, 1.55), rng.uniform(0.02, 1.55), rng.uniform(0, 2 * np.pi)
        k2sq = (k * (1 + 1j * delta) / nu) ** 2
        ki, ks = k * np.cos(ti), k * np.cos(ts)
        b1i, b1s = k * np.sin(ti), k * np.sin(ts)
        b2i, b2s = (bottom_model._vertical(k2sq, x) for x in (ki, ks))
        v = (rho * b1i - b2i) / (rho * b1i + b2i)
        t, g = 1 + v, b1i * (1 - v)
        q_dot_ki = ki * ks * np.cos(phi) - ki**2
        rhs1 = 1j * g * (1 - rho)                                           # A - B = rhs1
        rhs2 = t * (b1i**2 - b2i**2 / rho - (1 - 1 / rho) * q_dot_ki)       # i b1s A + (i b2s / rho) B = rhs2
        a = (rhs2 + (1j * b2s / rho) * rhs1) / (1j * b1s + 1j * b2s / rho)
        b = a - rhs1
        K_i, K_s = np.array([ki, 0.0]), ks * np.array([np.cos(phi), np.sin(phi)])
        expected = 1j * k * pen.a_wp(K_s, K_i, k, rho, nu, delta) / np.sin(ts)
        np.testing.assert_allclose(b, expected, rtol=1e-12)


def test_sigma_wp_forms():
    """The retyped (15.11) is (k_w^4 / nu^2) |A_wp|^2 W; ours carries |beta_p(K_s) / beta_w(K_s)|^2 more (module
    docstring; ours). They agree for a vertical sediment ray (K_s = 0) and ours vanishes as the sediment ray turns
    horizontal (K_s -> k_p)."""
    k = 2 * np.pi * 20e3 / C_SAX
    K_i = np.array([k * np.cos(np.radians(12)), 0.0])
    args = (k, SAX99["a_rho"], SAX99["nu"], 0.0, SAX99_ROUGH["w2"], SAX99_ROUGH["gamma2"], SAX99_ROUGH["K0"])
    for K_s, ratio in ((np.array([0.0, 0.0]), 1.0), (np.array([0.0, k / SAX99["nu"] * (1 - 1e-12)]), 0.0)):
        ours = pen.sigma_wp(K_s, K_i, *args)
        book = pen.sigma_wp(K_s, K_i, *args, obliquity=False)
        assert ours == pytest.approx(ratio * book, abs=1e-5 * book)


# --- the sinusoidal ripple (15.5)-(15.10) ------------------------------------------------------------------------

def test_diffraction_angle_and_cutoff():
    """(15.5): cos theta_ps = nu (cos theta_wi - lambda_w / lambda_0), i.e. Snell's law in the sediment for
    K_s = K_i - K0; (15.6): a propagating order needs lambda_w > lambda_0 (cos theta_wi - 1/nu), a high-frequency
    cutoff. SAX99's nu = 1.16 (critical 30.5 deg, (8.56)), lambda_0 0.48 m (Fig. 15.6). Cutoff at 15 deg: 30.7 kHz
    (ours, from (15.6)); the diffracted order is lossless-propagating below it and evanescent above it."""
    nu, l0, g = SAX99["nu"], 0.48, np.radians(15.0)
    for f in (10e3, 20e3, 30e3):
        lam = C_SAX / f
        k_w = 2 * np.pi / lam
        snell = (k_w * np.cos(g) - 2 * np.pi / l0) / (k_w / nu)                # K_s / k_p'
        assert pen.ripple_diffraction_cos(g, nu, lam, l0) == pytest.approx(snell, rel=1e-13)
    cut = pen.ripple_cutoff_wavelength(g, nu, l0)
    f_cut = C_SAX / cut
    assert f_cut == pytest.approx(30.69e3, rel=0.001)                         # ours
    for f, propagates in ((0.99 * f_cut, True), (1.01 * f_cut, False)):
        c = pen.ripple_diffraction_cos(g, nu, C_SAX / f, l0)
        assert (abs(c) <= 1.0) == propagates
    # Past the cutoff the field at 17 cm collapses (the order decays with depth): over 30 dB between 30 and 40 kHz.
    def level(f):
        p = pen.ripple_field(np.linspace(0, l0, 48), 0.17, g, f, C_SAX, h0=0.01, ripple_wavelength=l0, **SAX99)
        return 10 * np.log10(np.mean(np.abs(p) ** 2))
    assert level(30e3) - level(40e3) > 30.0


def test_zero_relief_is_the_zeroth_order_field():
    """h0 -> 0 gives the flat-interface field (15.1), V_wp exp(i K_i x + i k_p beta_p z) exactly, and P_+- grow as h0."""
    g, f, d, x = np.radians(12.0), 20e3, np.array([0.05, 0.17, 0.3]), 0.1
    p0 = pen.ripple_field(x, d, g, f, C_SAX, h0=0.0, ripple_wavelength=0.48, **SAX99)
    b = sax99_bottom()
    k_w = 2 * np.pi * f / C_SAX
    a_p = SAX99["nu"] / (1 + 1j * SAX99["delta"])
    root = np.sqrt(1 / a_p**2 - np.cos(g) ** 2 + 0j)
    root = np.where(root.imag < 0, -root, root)
    expected = fluid_transmission(g, C_SAX, **b) * np.exp(1j * k_w * np.cos(g) * x + 1j * k_w * root * d)
    np.testing.assert_allclose(p0, expected, rtol=1e-12)
    p1 = pen.ripple_field(x, d, g, f, C_SAX, h0=1e-4, ripple_wavelength=0.48, **SAX99) - p0
    p2 = pen.ripple_field(x, d, g, f, C_SAX, h0=2e-4, ripple_wavelength=0.48, **SAX99) - p0
    np.testing.assert_allclose(p2, 2 * p1, rtol=1e-12)


def test_fig_15_6_ripple_levels():
    """figure: Fig. 15.6 ([N3] Sec. 2.7): h0 0.01 m, lambda0 0.48 m, 10-40 kHz, 17 cm depth: one-way loss "about
    10-25 dB below the critical angle, depending on position along the ripple". The note gives neither the sediment
    nor the angle: SAX99 (Table 13.1) and 15 and 20 deg (below the 30.5 deg critical angle) are ours. Where the
    diffracted order propagates ((15.6)), the loss averaged over the ripple is within 10-25 dB widened by 3 dB
    (the reading is "about"); where the zeroth order is not negligible (10 kHz) it varies along the ripple."""
    x = np.linspace(0.0, 0.48, 97)
    for deg, freqs in ((15.0, (10e3, 20e3, 30e3)), (20.0, (10e3, 20e3, 30e3, 40e3))):
        g = np.radians(deg)
        for f in freqs:
            assert C_SAX / f > pen.ripple_cutoff_wavelength(g, SAX99["nu"], 0.48)       # the order propagates
            p = pen.ripple_field(x, 0.17, g, f, C_SAX, h0=0.01, ripple_wavelength=0.48, **SAX99)
            loss = -10 * np.log10(np.mean(np.abs(p) ** 2))
            assert 7.0 < loss < 28.0, (deg, f, loss)
        p = pen.ripple_field(x, 0.17, g, 10e3, C_SAX, h0=0.01, ripple_wavelength=0.48, **SAX99)
        spread = 20 * np.log10(np.abs(p).max() / np.abs(p).min())
        assert spread > 5.0                                                      # position dependence
    # Without the relief the same points are far quieter (the evanescent field alone): at 20 kHz, 15 deg, 49 dB (ours).
    p0 = pen.ripple_field(x, 0.17, np.radians(15.0), 20e3, C_SAX, h0=0.0, ripple_wavelength=0.48, **SAX99)
    assert -20 * np.log10(np.abs(p0[0])) > 45.0


# --- the averaged field (15.11)-(15.16) --------------------------------------------------------------------------

def _fig_15_8_peak(depth, obliquity=True):
    src, pt = np.array([0.0, 0.0, -4.0]), np.array([20.0, 0.0, depth])
    t0 = (np.hypot(20.0, 4.0) + depth / SAX99["nu"]) / C_SAX
    times = t0 + np.linspace(-0.3e-3, 1.0e-3, 261)
    _, pulsed = pen.mean_square_field(pt, 20e3, C_SAX, 0.0, source=src, radius=2.0, n=301, times=times,
                                      pulse_ts=100e-6, obliquity=obliquity, **SAX99, **SAX99_ROUGH)
    s0 = 10 ** (190.0 / 20)                                                     # SL 190 dB re 1 uPa at 1 m
    return 10 * np.log10(s0**2 * pulsed.max())


def test_fig_15_8_mean_square_field():
    """figure: Fig. 15.8 ([N3] Sec. 2.7): SAX99 (Table 13.1), (D.7) with K0 = 10 1/m, c_w 1530 m/s, source 4 m up,
    20 m off, u(t) = exp(-t^2 / t_s^2), t_s 100 us, SL 190 dB, (15.15): mean-square pressure peaks "about 133-141 dB
    at 5-40 cm" (20 kHz, [N3] Sec. 3.5). Ours: 141.6 dB at 5 cm falling to 133.5 dB at 40 cm; bounds widened by the
    note's 1.5 dB reading error. The retyped (15.11) without the obliquity factor gives 146.7 dB at 5 cm: outside."""
    peaks = [_fig_15_8_peak(d) for d in (0.05, 0.1, 0.2, 0.4)]
    assert all(131.5 < p < 142.5 for p in peaks), peaks
    assert np.all(np.diff(peaks) < 0)                                           # falls with depth
    assert peaks[0] - peaks[-1] > 5.0                                            # 141 -> 133: about 8 dB
    assert _fig_15_8_peak(0.05, obliquity=False) > 142.5


def _plane_scattered(deg, f, depth):
    """<|P_s|^2> for a unit plane wave at `deg` grazing, SAX99 (ours: the CW (15.14) with |P_i| = 1)."""
    return pen.mean_square_field(np.array([0.0, 0.0, depth]), f, C_SAX, 0.0, grazing=np.radians(deg), radius=2.5,
                                 n=401, **SAX99, **SAX99_ROUGH)


def _zeroth(deg, f, depth):
    b = sax99_bottom()
    g = np.radians(deg)
    k_w = 2 * np.pi * f / C_SAX
    a_p = SAX99["nu"] / (1 + 1j * SAX99["delta"])
    root = np.sqrt(1 / a_p**2 - np.cos(g) ** 2 + 0j)
    return abs(fluid_transmission(g, C_SAX, **b) * np.exp(1j * k_w * root * depth)) ** 2


def test_above_critical_the_zeroth_order_dominates_and_below_it_the_scattered_field():
    """[N3] WP2 (v): above the critical angle (30.5 deg) the zeroth order (15.1) is at least 20 dB above the mean
    scattered intensity (SAX99, 20 kHz, 17 and 30 cm; ours); below it (15 deg) the order is reversed by more than
    20 dB, and the scattered field sits 10-35 dB under the incident one (Figs. 15.3, 15.8, 15.11 read 5-31 dB;
    widened, figure)."""
    for depth in (0.17, 0.30):
        for deg in (40.0, 60.0, 90.0):
            assert 10 * np.log10(_zeroth(deg, 20e3, depth) / _plane_scattered(deg, 20e3, depth)) > 20.0
        s = _plane_scattered(15.0, 20e3, depth)
        assert 10 * np.log10(s / _zeroth(15.0, 20e3, depth)) > 20.0
        assert -35.0 < 10 * np.log10(s) < -10.0


def test_mean_field_vanishes_with_the_relief():
    """[N3] WP2 (iv): h -> 0 gives zero scattered field: <|P_s|^2> is linear in w2."""
    a = pen.mean_square_field(np.array([0.0, 0.0, 0.2]), 20e3, C_SAX, 0.0, grazing=np.radians(15), radius=1.0, n=101,
                              **SAX99, w2=1e-5, gamma2=3.04, K0=10.0)
    b = pen.mean_square_field(np.array([0.0, 0.0, 0.2]), 20e3, C_SAX, 0.0, grazing=np.radians(15), radius=1.0, n=101,
                              **SAX99, w2=0.0, gamma2=3.04, K0=10.0)
    c = pen.mean_square_field(np.array([0.0, 0.0, 0.2]), 20e3, C_SAX, 0.0, grazing=np.radians(15), radius=1.0, n=101,
                              **SAX99, w2=3e-5, gamma2=3.04, K0=10.0)
    assert b == 0.0
    assert c == pytest.approx(3 * a, rel=1e-12)


# --- the provider (one realisation) --------------------------------------------------------------------------------

SRC = np.array([0.0, 0.0, -10.0])


def _sub_point(deg=15.0, depth=0.3):
    return np.array([(10.0 + depth) / np.tan(np.radians(deg)), 0.0, depth])


def _cw(e, f, cols=slice(None)):
    """The field at the band centre of the rays of one point: sum amp exp(-i w t) (the synthesis's convention)."""
    return np.sum(np.where(e.valid[:, cols], e.amplitude[:, cols] * np.exp(-2j * np.pi * f * e.delay[:, cols]), 0), axis=1)


def test_points_above_critical_get_exactly_the_fluid_rays():
    pts = np.array([[0.0, 0.0, 0.3], [3.0, 1.0, 0.2], [0.0, 0.0, -2.0]])      # steep, steep, in the water
    a, b = rough().eigenrays(SRC, pts), fluid(engine="numpy").eigenrays(SRC, pts)
    for name in ("delay", "amplitude", "launch", "arrival", "n_surface", "n_bottom", "valid"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_sub_critical_points_keep_the_fluid_ray_and_gain_the_scattered_rays():
    pts = np.array([[0.0, 0.0, 0.3], _sub_point()])
    with pytest.warns(RuntimeWarning, match="roughness-scattered"):
        a = rough().eigenrays(SRC, pts)
    b = quiet(fluid(engine="numpy").eigenrays, SRC, pts)
    for name in ("delay", "amplitude", "launch", "arrival"):
        np.testing.assert_array_equal(getattr(a, name)[:, 0], getattr(b, name)[:, 0])
    assert not a.valid[0, 1:].any() and a.valid[1, 1:].sum() > 5
    # The scattered rays leave the source above the target's patch, steeper than the least-time ray's critical-angle
    # entry, and arrive at the delays of (15.16): later than the least-time ray.
    assert np.all(a.delay[1, a.valid[1]] >= a.delay[1, 0] - 1e-12)
    # One realisation at 20 kHz, 15 deg, 0.3 m: the scattered field is far above the evanescent one (ours: > 40 dB).
    assert 20 * np.log10(abs(_cw(a, 20e3, slice(1, None))[1]) / abs(a.amplitude[1, 0])) > 40.0


def test_zero_relief_is_exactly_fluid():
    """[N3] WP2 (iv): w2 = 0 gives zero scattered amplitude: the echo is `fluid`'s, and the scattered amplitude grows
    as sqrt(w2) for a fixed interface realisation."""
    pts = np.array([_sub_point(), [0.0, 0.0, 0.2]])
    zero = dict(SAX99_ROUGH, w2=0.0)
    a = quiet(rough(rough_par=zero).eigenrays, SRC, pts)
    assert np.all(a.amplitude[:, 1:] == 0)
    rx = SRC + np.array([0.3, 0.0, 0.0])
    amps = np.array([1.0, 0.5])
    q1, p1, m1, _, _ = quiet(two_way_paths, rough(rough_par=zero), SRC, rx, pts, amps, engine="numpy")
    q2, p2, m2, _, _ = quiet(two_way_paths, fluid(engine="numpy"), SRC, rx, pts, amps, engine="numpy")
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(p1, p2)
    e1 = quiet(rough(rough_par=dict(SAX99_ROUGH, w2=1e-5)).eigenrays, SRC, pts)
    e4 = quiet(rough(rough_par=dict(SAX99_ROUGH, w2=4e-5)).eigenrays, SRC, pts)
    np.testing.assert_allclose(e4.amplitude[0, 1:], 2 * e1.amplitude[0, 1:], rtol=1e-12)
    np.testing.assert_array_equal(e4.delay, e1.delay)


def test_realisations_average_to_the_mean_square_field():
    """Averaged over interface realisations (seeds) the CW intensity of the scattered rays at the band centre is the
    mean-square field (15.14) by quadrature, times 1 / a_rho (the sediment tier's per-leg half of the reciprocity
    factor). 160 seeds: the sample mean of exponential intensities is within 1 dB at about 3 standard deviations."""
    pt = _sub_point(15.0, 0.3)
    vals = []
    for seed in range(160):
        e = quiet(rough(rough_par=dict(SAX99_ROUGH, seed=seed)).eigenrays, SRC, pt[None, :])
        vals.append(abs(_cw(e, 20e3, slice(1, None))[0]) ** 2)
    mean = pen.mean_square_field(pt, 20e3, C_SAX, 0.0, source=SRC, radius=3.0, n=601, **SAX99, **SAX99_ROUGH)
    assert 10 * np.log10(np.mean(vals) / (mean / SAX99["a_rho"])) == pytest.approx(0.0, abs=1.0)
    # one realisation is one draw: the intensities spread like an exponential distribution (std / mean about 1)
    assert 0.6 < np.std(vals) / np.mean(vals) < 1.4


def test_frozen_interface():
    """The same seed gives the same interface for every ping and point: cells are global (level, i, j) squares and
    their deviates a hash of (seed, cell); neighbouring points share cells; another seed is another interface."""
    a = pen.quadtree_cells(10.0, 0.0, 1.5, 0.04, 3.0)
    b = pen.quadtree_cells(10.05, 0.0, 1.5, 0.04, 3.0)
    keys_a, keys_b = set(zip(*(x.tolist() for x in a[:3]))), set(zip(*(x.tolist() for x in b[:3])))
    assert len(keys_a & keys_b) > 0.5 * len(keys_a)
    # the cells tile the area without overlap: their areas add up to the covered area to within the edge cells
    area = float(np.sum(a[5] ** 2))
    assert np.pi * 1.5**2 < area < (2 * 1.5 + 2 * a[5].max()) ** 2
    c1 = pen.patch_coefficients(7, a[0], a[1], a[2])
    np.testing.assert_array_equal(c1, pen.patch_coefficients(7, a[0], a[1], a[2]))
    assert not np.allclose(c1, pen.patch_coefficients(8, a[0], a[1], a[2]))
    big = pen.patch_coefficients(1, np.zeros(200000, dtype=int), np.arange(200000), -np.arange(200000))
    assert np.mean(np.abs(big) ** 2) == pytest.approx(1.0, abs=0.01)
    assert abs(np.mean(big)) < 0.01
    p = rough()
    pt = _sub_point()
    e1, e2 = quiet(p.eigenrays, SRC, pt[None, :]), quiet(p.eigenrays, SRC, pt[None, :])
    np.testing.assert_array_equal(e1.amplitude, e2.amplitude)


def test_reciprocal_under_swapping_transmitter_and_receiver():
    """The echo of a buried point is the same with transmitter and receiver swapped (both legs use the same
    frozen interface and the same per-leg formula; (8.68) split evenly as in the fluid tier)."""
    pts = np.array([_sub_point(15.0, 0.25), _sub_point(18.0, 0.2) + np.array([0.0, 0.4, 0.0])])
    a, b = SRC, SRC + np.array([-1.5, 0.7, 0.2])
    amps = np.array([1.0, 0.7 + 0.2j])

    def h(tx, rx):
        q, p, m, _, _ = quiet(two_way_paths, rough(), tx, rx, pts, amps, engine="numpy")
        tau = (np.linalg.norm(q - tx, axis=1) + np.linalg.norm(p - rx, axis=1)) / C_SAX
        return np.sum(m * np.exp(-2j * np.pi * 20e3 * tau))
    np.testing.assert_allclose(h(a, b), h(b, a), rtol=1e-12)


def test_refusals_and_keys():
    b = sax99_bottom()
    base = dict(penetration="rough", bottom=b, frequency=2e4, engine="numpy")
    if engine.HAVE_CPP:
        with pytest.raises(ValueError, match="not implemented in the C\\+\\+ engine"):
            SedimentProvider(C_SAX, b["speed"], 0.0, **dict(base, engine="cpp"), roughness=dict(SAX99_ROUGH))
    with pytest.raises(ValueError, match="needs `roughness`"):
        SedimentProvider(C_SAX, b["speed"], 0.0, **base)
    with pytest.raises(ValueError, match="roughness.K0 is needed"):
        SedimentProvider(C_SAX, b["speed"], 0.0, **base, roughness={"w2": 1e-5, "gamma2": 3.0})
    with pytest.raises(ValueError, match="did you mean 'gamma2'"):
        SedimentProvider(C_SAX, b["speed"], 0.0, **base, roughness={"w2": 1e-5, "gama2": 3.0, "K0": 10.0})
    with pytest.raises(ValueError, match="subcritical"):
        SedimentProvider(C_SAX, b["speed"], 0.0, **base, roughness=dict(SAX99_ROUGH), subcritical="ray")
    with pytest.raises(ValueError, match="only with penetration: rough"):
        SedimentProvider(C_SAX, b["speed"], 0.0, penetration="fluid", bottom=b, frequency=2e4, roughness=dict(SAX99_ROUGH))
    with pytest.raises(ValueError, match="gamma2"):
        SedimentProvider(C_SAX, b["speed"], 0.0, **base, roughness={"w2": 1e-5, "gamma2": 4.5, "K0": 10.0})
    with pytest.raises(ValueError, match="penetration"):
        make_provider("sediment", C_SAX, {"interface_z": 0.0, "penetration": "ruff"})
    # roughness from a named sediment: w2 and gamma2 of its row in configs/sediments.yaml (w2 in m^(4 - gamma2))
    p = make_provider("sediment", C_SAX, {"interface_z": 0.0, "penetration": "rough", "engine": "numpy", "frequency": 2e4,
                                          "bottom": {"sediment": "medium_sand"},
                                          "roughness": {"sediment": "medium_sand", "K0": 10.0, "seed": 3}})
    row = bottom_model.sediment("medium_sand")
    assert (p.roughness["w2"], p.roughness["gamma2"], p.roughness["seed"]) == (row["w2"], row["gamma2"], 3)


# --- the scenario path ---------------------------------------------------------------------------------------------

def _small(name, **prop):
    cfg = load_scenario_config(name)
    cfg["platform"]["raster"] = {"x": [0.0, 0.0001, 0.1], "y": [0.0, 0.0001, 0.1]}
    cfg["propagation"].update(prop)
    return cfg


def test_scenario_rough_equals_fluid_where_nothing_is_sub_critical():
    """buried_cylinder at nadir: every buried point is above the critical angle, so `rough` gives `fluid`'s record
    bit for bit (both on the NumPy engine); the C++ engine refuses `rough`."""
    medium = {"speed": 1700.0, "density": 1.845, "attenuation": bottom_model.fluid_halfspace("medium_sand", 1500.0)["attenuation"]}
    on = run_scenario(_small("buried_cylinder", penetration="fluid", bottom=medium, engine="numpy")).records[0].samples.array
    r = run_scenario(_small("buried_cylinder", penetration="rough", bottom=medium, engine="numpy",
                            roughness={"sediment": "medium_sand", "K0": 10.0})).records[0].samples.array
    np.testing.assert_array_equal(np.asarray(on), np.asarray(r))
    if engine.HAVE_CPP:
        with pytest.raises(Exception, match="C\\+\\+ engine"):
            run_scenario(_small("buried_cylinder", penetration="rough", bottom=medium, engine="cpp",
                                roughness={"sediment": "medium_sand", "K0": 10.0}))
