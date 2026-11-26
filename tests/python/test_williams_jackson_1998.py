"""Williams and Jackson (1998), "Bistatic bottom scattering: Model, experiments, and model/data comparison",
J. Acoust. Soc. Am. 103 (1), 169-181 [WJ], as extracted in docs/references/bottom_bistatic_williams_jackson_1998.md
(the paper read on the page images; status words there: confirmed = printed in [WJ]; figure = read off a printed
curve, about +-0.5 dB, never a bound tighter than +-1 dB; ours = computed; one site = one site's values).

[WJ] 5.1: the published bistatic roughness model as an opt-in, `roughness_model="williams_jackson_1998"`: the
Kirchhoff cross section (11)-(15) and the first-order perturbation one (16)-(19) joined by (10) with eta = -2,
sigma = (sigma_k^-2 + sigma_p^-2)^(-1/2). Its two terms are our `kirchhoff_strength` and `perturbation_strength`
(equal to the print to 1e-13 dB, the note Sec. 4 items 3-4), reused, not typed again in bottom.py.

The test vectors are the note's (ours, `vectors.py`), recomputed for this package from a second re-typing with the
Kirchhoff integral by direct quadrature (scratch `a-wj/recompute.py`): all ten agree to 1e-4 dB.
"""
import cmath
import math
import warnings

import numpy as np
import pytest
from scipy import integrate, optimize, special

from gpusonar.sim import bottom

WJ = "williams_jackson_1998"
C = 1500.0                     # [WJ] prints no water sound speed; the note uses 1500 m/s (2 % in c: about 0.1 dB)
F = 40e3                       # the BAMS frequency of both sites ([WJ] Sec. II)
D = np.radians

# [WJ] Table I (p. 171), as printed (confirmed); w2 in cm^4 at K = 1 rad/cm, converted by `w2_from_cm4` (the note
# Sec. 2: the paper's convention is ours, h0 = 1 cm).
PANAMA_CITY = dict(rho=1.97, nu=1.126, delta=0.0166, w2=bottom.w2_from_cm4(0.00849, 3.12), gamma2=3.12)
ECKERNFOERDE = dict(rho=1.18, nu=0.991, delta=0.00186, w2=bottom.w2_from_cm4(0.00231, 3.42), gamma2=3.42)

# The note's 5.1 test vectors: Panama City, 40 kHz, c 1500 m/s, roughness only, (theta_i, theta_s, phi) in degrees,
# phi 180 = backscatter, 0 = forward (the note Sec. 2; `reverb_bistatic.bistatic_azimuth`). Ours (`vectors.py`);
# recomputed here independently (module docstring). Tolerance 0.01 dB (the note's).
VECTORS = [((5, 5, 180), -40.157), ((10, 10, 180), -31.870), ((20, 20, 180), -29.799), ((35, 35, 180), -28.154),
           ((60, 60, 180), -22.240), ((20, 20, 90), -29.141), ((20, 20, 30), -20.860), ((20, 20, 5), -0.799),
           ((10, 40, 0), -12.292), ((10, 40, 90), -29.179)]


def db(x):
    return 10.0 * np.log10(x)


def wj_retyped(ti, ts, phi, k, rho, nu, delta, w2, gamma2, integral=None):
    """[WJ] (2)-(4), (10)-(19) typed again here from the note, independent of bottom.py except for the special
    function F(q, alpha) = int_0^inf exp(-q u^(2 alpha)) J0(u) u du (`integral`; default `bottom.kirchhoff_integral`,
    which `test_the_kirchhoff_integral_at_the_vectors_against_direct_quadrature` holds against quadrature).
    Returns (sigma_br, sigma_kr, sigma_pr)."""
    integral = bottom.kirchhoff_integral if integral is None else integral
    kappa = (1 + 1j * delta) / nu                                                              # (18)

    def p_(th):                                                                                # (19), Im P > 0
        p = cmath.sqrt(kappa * kappa - math.cos(th) ** 2)
        return -p if p.imag < 0 else p

    def r_(th):                                                                                # flat interface
        y = rho * math.sin(th) / p_(th)
        return (y - 1) / (y + 1)

    dt = 0.5 * math.sqrt(math.cos(ti) ** 2 - 2 * math.cos(ti) * math.cos(ts) * math.cos(phi) + math.cos(ts) ** 2)  # (2)
    dz = 0.5 * (math.sin(ti) + math.sin(ts))                                                   # (3)
    d = math.hypot(dt, dz)                                                                     # (4)
    alpha = gamma2 / 2 - 1                                                                     # (13)
    ch2 = 2 * math.pi * w2 * math.gamma(2 - alpha) * 2 ** (-2 * alpha) / (alpha * (1 - alpha) * math.gamma(1 + alpha))  # (14)
    q = 2 * k * k * dz * dz * ch2 * (2 * k * dt) ** (-2 * alpha)                               # (12)
    sk = abs(r_(math.asin(d))) ** 2 / (8 * math.pi) * (d * d / (dz * dt)) ** 2 * float(integral(q, alpha))  # (11), (15)
    g = (1 / rho - 1) * (math.cos(ti) * math.cos(ts) * math.cos(phi) - p_(ti) * p_(ts) / rho) + 1 - kappa * kappa / rho  # (17)
    sp = 0.25 * k ** 4 * abs(1 + r_(ti)) ** 2 * abs(1 + r_(ts)) ** 2 * abs(g) ** 2 * w2 / (2 * k * dt) ** gamma2   # (16), (5)
    return (sk ** -2 + sp ** -2) ** -0.5, sk, sp                                               # (10), eta = -2


def integral_by_quadrature(q, alpha):
    """F(q, alpha) by direct quadrature between the zeros of J0 (no spline, no contour)."""
    f = lambda u: math.exp(-q * u ** (2 * alpha)) * special.j0(u) * u  # noqa: E731
    total, u0 = 0.0, 0.0
    for z in special.jn_zeros(0, 20000):
        total += integrate.quad(f, u0, z, limit=200, epsabs=0.0, epsrel=1e-12)[0]
        u0 = z
        if math.exp(-q * z ** (2 * alpha)) * z < 1e-16:
            break
    return total


def wj(ti, ts, phi, f=F, par=PANAMA_CITY, **kw):
    return bottom.roughness_strength(ti, ts, f, C, azimuth=phi, roughness_model=WJ, **par, **kw)


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """The model's once-per-process warnings start afresh in every test."""
    bottom._warned_wj98_range = False
    yield
    bottom._warned_wj98_range = False


# --- [WJ] 5.1: the published roughness model ------------------------------------------------------------------

def test_the_published_model_gives_the_notes_test_vectors():
    """Panama City (Table I), 40 kHz: the note's ten values within 0.01 dB (backscatter at 5-60 deg, out of plane,
    near forward, unequal angles)."""
    for (ti, ts, phi), want in VECTORS:
        got = float(db(wj(D(ti), D(ts), D(phi))))
        assert got == pytest.approx(want, abs=0.01), ((ti, ts, phi), got, want)


def test_the_model_equals_an_independent_re_typing_of_eqs_10_to_19():
    """bottom.py's model (our Kirchhoff and perturbation functions joined by (10)) against the re-typing above, to
    1e-9 dB, at the vectors and at Eckernfoerde (Table I) in and out of plane."""
    k = 2 * math.pi * F / C
    cases = [(PANAMA_CITY, g) for g, _ in VECTORS] + [(ECKERNFOERDE, g) for g in
                                                      ((5, 5, 180), (15, 15, 180), (20, 20, 60), (10, 30, 120), (30, 10, 20))]
    for par, (ti, ts, phi) in cases:
        want = wj_retyped(D(ti), D(ts), D(phi), k, **par)[0]
        got = float(wj(D(ti), D(ts), D(phi), par=par))
        assert db(got) == pytest.approx(db(want), abs=1e-9), (par, (ti, ts, phi))


def test_the_kirchhoff_integral_at_the_vectors_against_direct_quadrature():
    """What the re-typing shares with bottom.py, the special function F, against quadrature between the zeros of J0
    at the vectors' arguments: within 1e-6 relative (4e-6 dB), so the vectors do not rest on our spline."""
    k = 2 * math.pi * F / C
    for (ti, ts, phi), want in VECTORS:
        a, b = wj_retyped(D(ti), D(ts), D(phi), k, **PANAMA_CITY), wj_retyped(D(ti), D(ts), D(phi), k, **PANAMA_CITY,
                                                                              integral=integral_by_quadrature)
        assert b[1] == pytest.approx(a[1], rel=1e-6)
        assert float(db(b[0])) == pytest.approx(want, abs=0.01)


GRID_SITES = {"panama_city": PANAMA_CITY, "eckernfoerde": ECKERNFOERDE,
              **{n: {k: v for k, v in bottom.sediment(n).items() if k != "sigma2"}
                 for n in ("medium_sand", "sandy_gravel", "very_fine_silt")}}


def _grid():
    ti, ts, phi = np.meshgrid(D([3.0, 10.0, 20.0, 35.0, 60.0, 85.0]), D([5.0, 20.0, 45.0, 80.0]),
                              D([0.0, 5.0, 30.0, 90.0, 150.0, 180.0]), indexing="ij")
    return ti.ravel(), ts.ravel(), phi.ravel()


@pytest.mark.parametrize("name", list(GRID_SITES))
def test_never_above_the_smaller_term_and_at_most_1p5_db_below_it(name):
    """(10) with eta = -2: sigma = min / (1 + (min/max)^2)^(1/2), so it lies between min / sqrt(2) and min: never above
    the smaller of the two terms and at most 10 log10(2) / 2 = 1.505 dB below it (four sediments and the two sites,
    20, 40 and 100 kHz, 144 geometries each)."""
    par = GRID_SITES[name]
    ti, ts, phi = _grid()
    for f in (20e3, 40e3, 100e3):
        got = wj(ti, ts, phi, f=f, par=par)
        sk = bottom.kirchhoff_strength(ti, ts, f, C, azimuth=phi, **par)
        sp = bottom.perturbation_strength(ti, ts, f, C, azimuth=phi, **par)
        sp = np.where(sp > 0.0, sp, np.inf)            # exact specular (20/20 deg, phi 0): W(0) is infinite
        low = np.minimum(sk, sp)
        assert np.all(got <= low * (1 + 1e-12)), name
        assert np.all(db(got / low) >= -10 * math.log10(2) / 2 - 1e-9), name
        assert np.allclose(got, low / np.sqrt(1 + (low / np.maximum(sk, sp)) ** 2), rtol=1e-12, atol=0.0)


def test_where_kirchhoff_and_perturbation_are_equal_it_lies_1p5_db_below_both_by_construction():
    """Do not "fix" this: it is (10) as printed. Where the two terms are equal the smooth minimum with eta = -2 is
    2^(-1/2) times either, 1.505 dB below both (the note Sec. 0 item 1). At Panama City, 40 kHz, backscatter, the
    two cross between 5 deg (Kirchhoff 9.6 dB above) and 20 deg (5.9 dB below); at 10 deg they are within 0.07 dB
    and the model lies 1.47 dB below the smaller, the perturbation term (ours)."""
    def gap(deg):
        g = D(deg)
        return float(db(bottom.kirchhoff_strength(g, g, F, C, **PANAMA_CITY) / bottom.perturbation_strength(g, g, F, C, **PANAMA_CITY)))
    assert gap(5.0) > 9.0 and gap(20.0) < -5.0
    cross = optimize.brentq(gap, 5.0, 20.0, xtol=1e-12)
    g = D(cross)
    below = float(db(wj(g, g, np.pi) / bottom.kirchhoff_strength(g, g, F, C, **PANAMA_CITY)))
    assert below == pytest.approx(-10 * math.log10(2) / 2, abs=1e-6)
    ten = float(db(wj(D(10.0), D(10.0), np.pi) / bottom.perturbation_strength(D(10.0), D(10.0), F, C, **PANAMA_CITY)))
    assert ten == pytest.approx(-1.47, abs=0.01)


def test_far_from_specular_it_is_perturbation_and_at_specular_kirchhoff():
    """Far from the specular direction the Kirchhoff term is much larger and (10) gives the perturbation result;
    towards specular the perturbation term grows without bound (W(K) at K -> 0) and (10) gives Kirchhoff; in the
    exact specular direction (theta_i = theta_s, phi = 0) it IS the Kirchhoff result (the perturbation term is
    infinite there, our `perturbation_strength` returns 0 at K = 0 and is not used)."""
    for par in (PANAMA_CITY, GRID_SITES["medium_sand"]):
        g = D(5.0)                                                                   # backscatter at low grazing
        sk, sp = bottom.kirchhoff_strength(g, g, F, C, **par), bottom.perturbation_strength(g, g, F, C, **par)
        assert sk / sp > 5.0
        assert float(db(wj(g, g, np.pi, par=par) / sp)) == pytest.approx(0.0, abs=0.1)
        for th in (10.0, 30.0, 60.0):
            g = D(th)
            near = wj(g, g, D(0.2), par=par)
            assert float(db(near / bottom.kirchhoff_strength(g, g, F, C, azimuth=D(0.2), **par))) == pytest.approx(0.0, abs=0.01)
            exact = wj(g, g, 0.0, par=par)
            assert float(exact) == float(bottom.kirchhoff_strength(g, g, F, C, azimuth=0.0, **par)) and np.isfinite(exact)
    assert float(wj(0.0, D(20.0), np.pi)) == 0.0 and float(wj(0.0, 0.0, np.pi)) == 0.0   # grazing incidence: nothing


def test_reciprocity():
    """Exchanging the incident and scattered directions leaves the cross section unchanged (both terms are
    reciprocal: (11)-(19) are symmetric in theta_i and theta_s)."""
    ti, ts, phi = _grid()
    for par in (PANAMA_CITY, GRID_SITES["very_fine_silt"]):
        assert np.allclose(wj(ti, ts, phi, par=par), wj(ts, ti, phi, par=par), rtol=1e-11, atol=0.0)


def test_it_is_opt_in_and_the_default_and_the_model_spread_are_unchanged():
    """The default stays small-slope (D-86; [WJ] 5.4 is the owner's). The new model is kept out of
    `ROUGHNESS_MODELS`, which the fit's model spread and tools/boundary_models_study.py iterate over: their numbers
    stay what they were. Without the key every float is the small-slope one."""
    assert bottom.DEFAULT_ROUGHNESS_MODEL == "small_slope"
    assert WJ not in bottom.ROUGHNESS_MODELS and WJ in bottom.OPT_IN_ROUGHNESS_MODELS
    ti, ts, phi = _grid()
    par = GRID_SITES["medium_sand"]
    assert np.array_equal(bottom.roughness_strength(ti, ts, 30e3, C, azimuth=phi, **par),
                          bottom.small_slope_strength(ti, ts, 30e3, C, azimuth=phi, **par))


def test_validity_warnings_and_refusals():
    """The paper's stated scope (p. 169): 10-100 kHz (a warning outside, once per process), a power-law relief with
    a structure function, 2 < gamma2 < 4 ((6) prints <=; the end points have none: an error), no shadowing (the
    model has none: naming "wagner" is an error), unknown names refused with the nearest."""
    g = D([10.0, 20.0])
    with pytest.warns(UserWarning, match="outside the paper's 10-100 kHz"):
        wj(g, g, np.pi, f=300e3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wj(g, g, np.pi, f=5e3)                                                       # once per process
        bottom._warned_wj98_range = False
        wj(g, g, np.pi, f=50e3)                                                      # inside: silent
        wj(g, g, D(40.0), f=10e3)
        wj(g, g, D(40.0), f=100e3)
    for gamma2 in (2.0, 4.0, 4.5):
        with pytest.raises(ValueError, match="2 < gamma2 < 4"):
            wj(g, g, np.pi, par=dict(PANAMA_CITY, gamma2=gamma2))
    with pytest.raises(ValueError, match="has no shadowing"):
        wj(g, g, np.pi, shadowing="wagner")
    assert np.array_equal(wj(g, g, np.pi, shadowing="none"), wj(g, g, np.pi))
    with pytest.raises(ValueError, match="williams_jackson_1998"):
        bottom.roughness_strength(g, g, F, C, roughness_model="williams_jackson_1999", **PANAMA_CITY)


def test_the_fit_offers_the_model_and_recovers_its_parameters():
    """The model is defined for backscatter as well as bistatic geometry, with the same six parameters, so the fit
    (`sediment_fit.fit`, tools/fit_sediment.py `--roughness-model`) offers it: data made with it at Panama City's
    Table I inputs (backscatter 5-40 deg, 40 and 80 kHz, our default volume term with sigma2 0.0001, noise free)
    give back w2, gamma2 and sigma2."""
    from gpusonar.sim import sediment_fit
    truth = dict(PANAMA_CITY, sigma2=1e-4)
    deg = np.arange(5.0, 41.0, 5.0)
    g, f = np.concatenate([deg, deg]), np.repeat([40e3, 80e3], len(deg))
    y = np.concatenate([bottom.backscattering_strength_db(D(deg), fr, C, roughness_model=WJ, **truth) for fr in (40e3, 80e3)])
    fit = sediment_fit.fit(g, f, y, sound_speed=C, rho=truth["rho"], nu=truth["nu"], delta=truth["delta"], roughness_model=WJ,
                           n_starts=4)
    assert fit.roughness_model == WJ and fit.rms_residual_db < 1e-3
    for key in ("w2", "gamma2", "sigma2"):
        assert fit.params[key] == pytest.approx(truth[key], rel=2e-3), key
    # The tool offers it by name (its --help lists the choices).
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    run = subprocess.run([sys.executable, str(root / "tools" / "fit_sediment.py"), "--help"], capture_output=True, text=True,
                         env={"PYTHONPATH": str(root / "python"), "PATH": "/usr/bin:/bin"})
    assert run.returncode == 0 and WJ in run.stdout, run.stderr


def test_the_simulators_point_scatterer_seabed_takes_the_model_at_bistatic_geometry():
    """A `seafloor` of point scatterers with `model: fluid_sediment` and `roughness_model: williams_jackson_1998`
    (`sim/targets.py`): with projector and receiver apart each scatterer's intensity changes from the default
    model's by the two models' ratio at its own grazing angles and azimuth (computed here from the geometry)."""
    from gpusonar.sim import build_scene
    sand = bottom.sediment("medium_sand")
    item = dict({"type": "seafloor", "x_range": [2.0, 30.0], "y_range": [-8.0, 8.0], "density": 2.0, "model": "fluid_sediment"},
                **sand)
    tx, rx = np.array([0.0, 0.0, -6.0]), np.array([12.0, 5.0, -3.0])

    def amplitudes(**extra):
        scene = build_scene([dict(item, **extra)], np.random.default_rng(4))
        pts, amp, _ = scene.evaluate(0.0, tx, rx, frequency=F, sound_speed=C)
        return pts, amp
    pts, default = amplitudes()
    _, published = amplitudes(roughness_model=WJ)
    g_in = np.arcsin(-tx[2] / np.linalg.norm(pts - tx, axis=1))
    g_out = np.arcsin(-rx[2] / np.linalg.norm(pts - rx, axis=1))
    t_in, t_out = (pts - tx)[:, :2], (rx - pts)[:, :2]
    az = np.arccos(np.clip((t_in * t_out).sum(axis=1) / np.linalg.norm(t_in, axis=1) / np.linalg.norm(t_out, axis=1), -1, 1))
    want = (bottom.strength(g_in, g_out, F, C, azimuth=az, roughness_model=WJ, **sand)
            / bottom.strength(g_in, g_out, F, C, azimuth=az, **sand))
    assert np.allclose(np.abs(published) ** 2 / np.abs(default) ** 2, want, rtol=1e-6)
    assert want.min() < 0.5 and np.ptp(np.degrees(az)) > 90.0                  # a real bistatic spread, a real change


# --- [WJ] 5.3 / plan G-15: the spectral, non-isotropic volume term, opt-in ------------------------------------

# [WJ] Table I volume spectra (confirmed): w3 in cm^3 at k = 1 rad/cm (h0 = 1 cm), gamma3, mu (signed: the paper's
# mu is negative, density and compressibility fluctuations anti-correlated; the note Sec. 2).
ECKERNFOERDE_VOLUME = dict(w3=bottom.w3_from_cm3(0.00013, 4.0), gamma3=4.0, mu=-0.69)
PANAMA_CITY_VOLUME = dict(w3=bottom.w3_from_cm3(0.0000161, 4.0), gamma3=4.0, mu=-2.44)


def volume_retyped(ti, ts, phi, k, rho, nu, delta, w3, gamma3, mu, **_):
    """[WJ] (20)-(22) typed again here from the note, independent of bottom.py. Returns (sigma_bv, sigma_v)."""
    kappa = (1 + 1j * delta) / nu

    def p_(th):
        p = cmath.sqrt(kappa * kappa - math.cos(th) ** 2)
        return -p if p.imag < 0 else p

    def r_(th):
        y = rho * math.sin(th) / p_(th)
        return (y - 1) / (y + 1)
    dt2 = 0.25 * (math.cos(ti) ** 2 - 2 * math.cos(ti) * math.cos(ts) * math.cos(phi) + math.cos(ts) ** 2)      # (2)^2
    dk = k * abs(4 * dt2 + ((p_(ti) + p_(ts)).real) ** 2) ** 0.5                                                 # (22)
    bracket = mu * kappa ** 2 + math.cos(ti) * math.cos(ts) * math.cos(phi) - p_(ti) * p_(ts)
    sigma_v = math.pi / 2 * k ** 4 * abs(bracket) ** 2 * w3 / dk ** gamma3                                       # (21), (7)
    return (abs(1 + r_(ti)) ** 2 * abs(1 + r_(ts)) ** 2 * sigma_v / (2 * k * rho ** 2 * (p_(ti) + p_(ts)).imag),  # (20)
            sigma_v)


def wjv(ti, ts, phi, f=F, par=ECKERNFOERDE, vol=None, **kw):
    vol = ECKERNFOERDE_VOLUME if vol is None else vol
    return bottom.volume_strength(ti, ts, f, C, azimuth=phi, volume_model=WJ, **par, **vol, **kw)


def test_the_volume_units_follow_the_spectrum_convention():
    """W_rr(k) = w3 / (h0 k)^gamma3 with h0 = 1 cm, w3 in cm^3 (Table I): w3 [m^(3 - gamma3)] = w3 [cm^3] x
    10^(2 gamma3 - 6) (ours, as `w2_from_cm4`): Panama City 1.61e-5 cm^3 -> 1.61e-3 m^-1, Eckernfoerde 1.3e-4 -> 0.013."""
    assert bottom.w3_from_cm3(0.0000161, 4.0) == pytest.approx(1.61e-3, rel=1e-12)
    assert bottom.w3_from_cm3(0.00013, 4.0) == pytest.approx(0.013, rel=1e-12)
    k_cm = 3.0                                                                          # rad/cm: the same W either way
    w_cm3 = 0.00013 / k_cm ** 3.3
    assert bottom.w3_from_cm3(0.00013, 3.3) / (100.0 * k_cm) ** 3.3 == pytest.approx(w_cm3 * 1e-6, rel=1e-12)


def test_the_spectral_volume_term_gives_the_notes_eckernfoerde_vectors():
    """Eckernfoerde (Table I), 40 kHz, c 1500 m/s, the volume term alone: -39.139 / -34.068 / -30.412 dB at 5 / 10 /
    20 deg backscatter, -35.653 at (10, 10, phi 90), -34.038 at (20, 20, phi 60) (ours, `vectors.py`; recomputed in
    scratch `a-wj/recompute_vol.py`; 0.01 dB), and equal to the re-typing above to 1e-9 dB."""
    k = 2 * math.pi * F / C
    for (ti, ts, phi), want in (((5, 5, 180), -39.139), ((10, 10, 180), -34.068), ((20, 20, 180), -30.412),
                                ((10, 10, 90), -35.653), ((20, 20, 60), -34.038)):
        got = float(wjv(D(ti), D(ts), D(phi)))
        assert db(got) == pytest.approx(want, abs=0.01), (ti, ts, phi)
        assert db(got) == pytest.approx(db(volume_retyped(D(ti), D(ts), D(phi), k, **ECKERNFOERDE, **ECKERNFOERDE_VOLUME)[0]), abs=1e-9)
    for (ti, ts, phi) in ((7, 33, 20), (33, 7, 140), (60, 15, 0)):
        for par, vol in ((ECKERNFOERDE, ECKERNFOERDE_VOLUME), (PANAMA_CITY, PANAMA_CITY_VOLUME)):
            assert db(float(wjv(D(ti), D(ts), D(phi), par=par, vol=vol))) == pytest.approx(
                db(volume_retyped(D(ti), D(ts), D(phi), k, **par, **vol)[0]), abs=1e-9)


def test_the_volume_null_lies_at_the_notes_azimuths():
    """The bracket of (21) (dipole + monopole) vanishes near one azimuth: at Eckernfoerde 42.8 / 39.3 / 32.4 / 18.4
    deg for theta_i = theta_s = 5 / 10 / 15 / 20 deg (ours, `vectors.py`; within 0.5 deg). The isotropic sigma2 term
    has no null."""
    for theta, want in ((5.0, 42.8), (10.0, 39.3), (15.0, 32.4), (20.0, 18.4)):
        g = D(theta)
        r = optimize.minimize_scalar(lambda ph: float(wjv(g, g, D(ph))), bounds=(1.0, 90.0), method="bounded",
                                     options={"xatol": 1e-6})
        assert r.x == pytest.approx(want, abs=0.5), (theta, r.x)
        assert db(r.fun / float(wjv(g, g, np.pi))) < -30.0
    phis = D(np.linspace(1.0, 180.0, 50))
    ten = np.full(phis.shape, D(10.0))
    iso = bottom.volume_strength(ten, ten, F, C, azimuth=phis, sigma2=0.001, **ECKERNFOERDE)
    assert iso.shape == phis.shape and np.all(iso == iso[0])


def test_it_is_j_and_r_14_10_with_rho_rk_minus_one_and_mu_prime_minus_mu():
    """J&R (2007) (14.10), as the note Sec. 2 types it: sigma_v = (pi/2)[mu'^2 |k_p|^4 + 2 rho_rk mu'
    Re{(k_ps . k_pi) conj(k_p^2)} + |k_ps . k_pi|^2] W_rr(Delta k), with k_ps . k_pi = k^2 (cos cos cos phi - P P)
    and k_p^2 = k^2 kappa^2. With rho_rk = -1 and mu' = -mu it is (21) (ours, algebra; to 1e-12 relative)."""
    k = 2 * math.pi * F / C
    for par, vol in ((ECKERNFOERDE, ECKERNFOERDE_VOLUME), (PANAMA_CITY, PANAMA_CITY_VOLUME)):
        kappa = (1 + 1j * par["delta"]) / par["nu"]
        for ti, ts, phi in ((5, 5, 180), (10, 30, 90), (20, 20, 39), (45, 12, 10)):
            a, b, c = D(ti), D(ts), D(phi)
            pi_, ps = (bottom._vertical(kappa ** 2, np.cos(x)) for x in (a, b))
            dot = k * k * (math.cos(a) * math.cos(b) * math.cos(c) - pi_ * ps)
            kp2 = k * k * kappa ** 2
            mup, rho_rk = -vol["mu"], -1.0
            dt2 = 0.25 * (math.cos(a) ** 2 - 2 * math.cos(a) * math.cos(b) * math.cos(c) + math.cos(b) ** 2)
            dk = k * abs(4 * dt2 + ((pi_ + ps).real) ** 2) ** 0.5
            jr = math.pi / 2 * (mup ** 2 * abs(kp2) ** 2 + 2 * rho_rk * mup * (dot * np.conj(kp2)).real + abs(dot) ** 2) \
                * vol["w3"] / dk ** vol["gamma3"]
            ours = float(bottom.williams_jackson_1998_volume_cross_section(a, b, F, C, azimuth=c, nu=par["nu"], delta=par["delta"], **vol))
            assert ours == pytest.approx(float(jr), rel=1e-12)
            # In backscatter the bracket is kappa^2 (mu - 1): |...|^2 = |kappa|^4 (mu' + 1)^2.
            if (ti, ts, phi) == (5, 5, 180):
                assert ours == pytest.approx(math.pi / 2 * k ** 4 * abs(kappa) ** 4 * (mup + 1) ** 2 * vol["w3"] / dk ** vol["gamma3"],
                                             rel=1e-12)


def test_the_transmission_is_our_flat_interface_term_and_reciprocal():
    """(20) is our existing transmission form ([WJ] Sec. 4 item 5, 1.1e-14 dB): with sigma2 = sigma_v / (amplitude
    attenuation in dB/m), point by point, the default `volume_strength` gives the same number. That is the limit in
    which the spectral term reduces to ours: a sigma_v that does not depend on angle. Reciprocal in the two angles."""
    ti, ts, phi = _grid()
    for par, vol in ((ECKERNFOERDE, ECKERNFOERDE_VOLUME), (PANAMA_CITY, PANAMA_CITY_VOLUME)):
        k = 2 * math.pi * F / C
        sv = bottom.williams_jackson_1998_volume_cross_section(ti, ts, F, C, azimuth=phi, nu=par["nu"], delta=par["delta"], **vol)
        alpha_db = 20 / math.log(10) * k * par["delta"] / par["nu"]
        spectral = wjv(ti, ts, phi, par=par, vol=vol)
        flat = np.array([float(bottom.volume_strength(a, b, F, C, sigma2=s / alpha_db, **par)) for a, b, s in zip(ti, ts, sv)])
        assert np.allclose(spectral, flat, rtol=1e-13, atol=0.0)
        assert np.allclose(spectral, wjv(ts, ti, phi, par=par, vol=vol), rtol=1e-12, atol=0.0)


def test_the_volume_model_needs_its_parameters_and_class_rows_have_none():
    """No class value exists for w3, gamma3, mu ([WJ] prints two sites; J&R Table 14.1 sites only): without them the
    model is refused with that reason; a site row's `volume_spectral:` block gives them (`bottom.volume_spectral`),
    and `named_backscattering_strength_db` uses it. The default volume term is untouched."""
    g = D([10.0, 20.0])
    with pytest.raises(ValueError, match="needs w3, gamma3 and mu.*no class values"):
        bottom.volume_strength(g, g, F, C, volume_model=WJ, sigma2=0.001, **ECKERNFOERDE)
    with pytest.raises(ValueError, match="needs w3, gamma3 and mu"):
        bottom.strength(g, g, F, C, volume_model=WJ, sigma2=0.001, **ECKERNFOERDE, w3=0.01, gamma3=4.0)
    block = bottom.volume_spectral("williams_jackson_1998_panama_city")
    assert block == pytest.approx(PANAMA_CITY_VOLUME, rel=1e-12) and set(block) == {"w3", "gamma3", "mu"}
    for name in ("medium_sand", "very_fine_silt", "jackson_briggs_1992_arafura", "placeholder_sand"):
        with pytest.raises(ValueError, match="no class values"):
            bottom.volume_spectral(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                                  # the row's sigma2 is unconfirmed (ours)
        named = bottom.named_backscattering_strength_db("williams_jackson_1998_panama_city", g, F, C, volume_model=WJ)
        row = bottom.sediment("williams_jackson_1998_panama_city")
    assert np.array_equal(named, bottom.backscattering_strength_db(g, F, C, volume_model=WJ, **row, **PANAMA_CITY_VOLUME))
    with pytest.raises(ValueError, match="no class values"):
        bottom.named_backscattering_strength_db("medium_sand", g, F, C, volume_model=WJ)
    with pytest.raises(ValueError, match="did you mean 'williams_jackson_1998'"):
        bottom.volume_strength(g, g, F, C, volume_model="williams_jackson", sigma2=0.001, **ECKERNFOERDE)
    par = bottom.sediment("medium_sand")
    assert np.array_equal(bottom.volume_strength(g, g, F, C, **par), bottom.volume_strength(g, g, F, C, volume_model="stockhausen", **par))


# --- [WJ] 5.2: the full model (both terms) against the paper's Figs. 7-8 and Table AII ------------------------

# The note Sec. 3: readings of the printed model curves (figure, +-0.5 dB on a 5-dB grid; a test bound never tighter
# than +-1 dB) for theta_i = theta_s, and the note's re-typed model (ours, 0.1 dB as printed, so +-0.051 dB).
# (label, site, theta, phi, reading, ours)
FIGURE_READINGS = [
    ("7a 10 deg", "panama_city", 10, 180, -31.6, -31.9), ("7a 10 deg", "panama_city", 10, 100, -31.9, -32.1),
    ("7a 10 deg", "panama_city", 10, 50, -31.2, -31.2),
    ("7a 5 deg", "panama_city", 5, 180, -40.0, -40.1), ("7a 5 deg", "panama_city", 5, 100, -40.8, -40.8),
    ("7a 5 deg", "panama_city", 5, 50, -41.2, -41.3),
    ("7b 20 deg", "panama_city", 20, 180, -29.5, -29.7), ("7b 20 deg", "panama_city", 20, 100, -29.1, -29.3),
    ("7b 20 deg", "panama_city", 20, 50, -25.3, -25.6),
    ("7b 10 deg", "panama_city", 10, 180, -31.8, -31.9), ("7b 10 deg", "panama_city", 10, 100, -32.0, -32.1),
    ("7b 10 deg", "panama_city", 10, 50, -31.0, -31.2),
    ("7c 35 deg", "panama_city", 35, 180, -27.6, -27.9), ("7c 35 deg", "panama_city", 35, 100, -26.7, -26.6),
    ("7d 60 deg", "panama_city", 60, 180, -21.9, -22.1), ("7d 60 deg", "panama_city", 60, 100, -20.1, -19.6),
    ("8a 10 deg", "eckernfoerde", 10, 180, -34.0, -34.0), ("8a 5 deg", "eckernfoerde", 5, 180, -39.0, -39.1),
    ("8b 15 deg", "eckernfoerde", 15, 180, -31.7, -31.8), ("8c 20 deg", "eckernfoerde", 20, 180, -30.6, -30.4),
    ("8d 20 deg", "eckernfoerde", 20, 180, -30.3, -30.4), ("8d 10 deg", "eckernfoerde", 10, 180, -33.9, -34.0),
]


def full_model(ti, ts, phi, site, f=F):
    par, vol = {"panama_city": (PANAMA_CITY, PANAMA_CITY_VOLUME), "eckernfoerde": (ECKERNFOERDE, ECKERNFOERDE_VOLUME)}[site]
    return bottom.strength(ti, ts, f, C, azimuth=phi, roughness_model=WJ, volume_model=WJ, **par, **vol)


def test_the_full_model_reproduces_the_papers_figures_7_and_8():
    """[WJ] 5.2: Table I inputs, both terms of the paper: within +-1 dB of every reading of the note Sec. 3 (figure)
    and equal to the note's re-typed values (ours, 0.051 dB): e.g. backscatter at Panama City 5 / 10 / 20 / 35 / 60
    deg against -40.0 / -31.6 / -29.5 / -27.6 / -21.9 read, Eckernfoerde 5 / 10 / 15 / 20 deg against -39.0 / -34.0 /
    -31.7 / -30.6. Figure readings: label, never tighten."""
    for label, site, theta, phi, reading, ours in FIGURE_READINGS:
        got = float(db(full_model(D(theta), D(theta), D(phi), site)))
        assert got == pytest.approx(ours, abs=0.051), (label, phi, got)
        assert abs(got - reading) <= 1.0, (label, phi, got, reading)


def test_the_eckernfoerde_curves_dip_where_the_figure_shows_the_volume_null():
    """Fig. 8a (figure): the 10-deg model curve falls below -45 dB between about phi 35 and 50 deg, the 5-deg one
    between about 35 and 60; the 20-deg curve (8c) dips near 20 deg. The full model's minimum over phi lies there
    and below -45 dB at 5 and 10 deg (the roughness term fills the null; ours)."""
    for theta, lo, hi in ((10.0, 35.0, 50.0), (5.0, 35.0, 60.0), (20.0, 15.0, 25.0)):
        g = D(theta)
        phis = np.linspace(1.0, 90.0, 891)
        total = db(full_model(g, g, D(phis), "eckernfoerde"))
        assert lo <= phis[np.argmin(total)] <= hi, (theta, phis[np.argmin(total)])
        if theta < 15.0:
            assert total.min() < -45.0 and np.all(total[(phis >= lo + 2.0) & (phis <= lo + 6.0)] < -40.0), theta


# Table AII (Panama City, 40 kHz), the part the note prints (Sec. 2.2; the full 142 values are on p. 180 and not in
# the repository): the backscatter means per angle (ours, mean of intensity of the repeats) and ten bistatic
# entries as printed (confirmed). (theta_i, theta_s, phi, dB)
TABLE_AII_BACKSCATTER = {5: -39.1, 6: -37.4, 7: -36.0, 8: -34.5, 9: -33.5, 10: -32.2, 11: -31.6, 12: -30.2, 13: -28.9,
                         14: -28.2, 15: -28.3}
TABLE_AII_BISTATIC = [(16.8, 15.3, 138.3, -30.7), (21.5, 17.6, 121.7, -31.2), (25.0, 21.5, 95.0, -29.3),
                      (17.8, 12.7, 93.1, -32.0), (13.8, 22.2, 58.0, -31.4), (23.3, 33.1, 58.9, -21.7),
                      (57.6, 35.6, 63.3, -20.4), (20.5, 16.8, 26.0, -20.5), (13.7, 20.6, 21.0, -20.7), (9.9, 16.7, 14.3, -36.3)]


def test_against_the_panama_city_data_the_published_model_is_closer_than_our_default():
    """[WJ] Sec. 4 item 2, a recorded characterisation (one site; not a reason to change the default, [WJ] 5.4: the
    fluid model is known to overpredict at Panama City, J&R p. 351). Model minus data, with the paper's volume term
    in both, on the part of Table AII the note prints (ours; bounds +-0.05 dB around the values measured here):
      backscatter means 5-15 deg (11): published mean -0.46 / rms 1.21 dB, small-slope +1.31 / 1.62 dB;
      ten bistatic entries:             published mean +2.33 / rms 5.73 dB, small-slope +5.49 / 7.30 dB
    (the rms of the ten is dominated by the entry nearest forward, (9.9, 16.7, 14.3), -36.3 dB, which both models put
    17-18 dB higher; without it the published model is +0.69 / 2.00 dB, small-slope +4.10 / 4.80).
    The note's full-table numbers (142 values, scratch `ref-b/wj_data.py`): -0.45 / 1.31 and +3.34 / 4.85 against
    +1.36 / 1.73 and +6.08 / 7.11 dB. Either way the published model is closer, in backscatter and bistatic."""
    def stats(model):
        back = np.array([float(db(bottom.strength(D(t), D(t), F, C, roughness_model=model, volume_model=WJ, **PANAMA_CITY,
                                                  **PANAMA_CITY_VOLUME))) - s for t, s in TABLE_AII_BACKSCATTER.items()])
        bist = np.array([float(db(bottom.strength(D(a), D(b), F, C, azimuth=D(c), roughness_model=model, volume_model=WJ,
                                                  **PANAMA_CITY, **PANAMA_CITY_VOLUME))) - s for a, b, c, s in TABLE_AII_BISTATIC])
        return [(float(x.mean()), float(np.sqrt((x ** 2).mean()))) for x in (back, bist, bist[:-1])]
    (pb, pb_rms), (pi_, pi_rms), without_p = stats(WJ)
    (sb, sb_rms), (si, si_rms), without_s = stats("small_slope")
    assert without_p == pytest.approx((0.69, 2.00), abs=0.05) and without_s == pytest.approx((4.10, 4.80), abs=0.05)
    measured = {"published back": (pb, pb_rms), "published bistatic": (pi_, pi_rms), "default back": (sb, sb_rms),
                "default bistatic": (si, si_rms)}
    want = {"published back": (-0.46, 1.21), "published bistatic": (2.33, 5.73), "default back": (1.31, 1.62),
            "default bistatic": (5.49, 7.30)}
    for key, (mean, rms) in want.items():
        assert measured[key][0] == pytest.approx(mean, abs=0.05) and measured[key][1] == pytest.approx(rms, abs=0.05), (key, measured[key])
    assert abs(pb) < abs(sb) and pb_rms < sb_rms and abs(pi_) < abs(si) and pi_rms < si_rms


def test_the_point_scatterer_seabed_takes_the_spectral_volume_from_a_site_row_and_refuses_a_class_row():
    """`seafloor` `model: fluid_sediment` with `volume_model: williams_jackson_1998` (`sim/targets.py`): the site row's
    `volume_spectral:` block is used (the same amplitudes as with the three given); a class row is refused."""
    from gpusonar.sim import build_scene
    item = {"type": "seafloor", "x_range": [2.0, 30.0], "y_range": [-8.0, 8.0], "density": 2.0, "model": "fluid_sediment",
            "volume_model": WJ}
    tx, rx = np.array([0.0, 0.0, -6.0]), np.array([12.0, 5.0, -3.0])

    def amplitudes(**extra):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                               # the row's equivalent sigma2 is unconfirmed
            scene = build_scene([dict(item, **extra)], np.random.default_rng(4))
            return scene.evaluate(0.0, tx, rx, frequency=F, sound_speed=C)[1]
    named = amplitudes(sediment="williams_jackson_1998_panama_city")
    assert np.all(np.isfinite(named)) and np.abs(named).max() > 0.0
    assert np.array_equal(named, amplitudes(sediment="williams_jackson_1998_panama_city", **PANAMA_CITY_VOLUME))
    assert not np.array_equal(named, amplitudes(sediment="williams_jackson_1998_panama_city", volume_model="stockhausen"))
    with pytest.raises(ValueError, match="no class values"):
        amplitudes(sediment="medium_sand")
