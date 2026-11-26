"""Mourad and Jackson (1989), "High Frequency Sonar Equation Models for Bottom Backscatter and Forward
Loss", OCEANS '89 Proceedings, pp. 1168-1175 [MJ89], as extracted in
docs/references/bottom_mourad_jackson_1989.md (the paper read on the page images; status words there:
confirmed = printed in [MJ89]; figure = read off a printed curve, with its reading error; ours = computed).

C3: our volume term against the paper's eq. (39) with the misprint corrected.
C4: the paper's model as an opt-in (`roughness_model` / `volume_model` "mourad_jackson_1989"), backscatter only.
"""
import math
import warnings

import numpy as np
import pytest

from gpusonar.acoustics import NOMINAL_SOUND_SPEED as NOMINAL
from gpusonar.sim import bottom

C = 1500.0


def mj89_volume_39(theta, rho, nu, delta, sigma2, printed=False):
    """[MJ89] (39), typed again here from the note (Sec. 2.4), independent of bottom.py:
        sigma_pv = 5 delta sigma2 |1 - R^2(theta)|^2 sin^2 theta / (nu ln(10) |P|^2 Im P),
    P = sqrt(kappa^2 - cos^2 theta) (11), kappa = (1 + i delta) / nu (12), R = (y - 1) / (y + 1),
    y = rho sin theta / P (9)-(10). `printed=True` uses the factor |1 - R(theta)|^2 the page prints."""
    kappa = (1.0 + 1j * delta) / nu
    p = np.sqrt(kappa**2 - np.cos(theta) ** 2 + 0j)
    p = np.where(p.imag < 0.0, -p, p)
    y = rho * np.sin(theta) / p
    r = (y - 1.0) / (y + 1.0)
    factor = np.abs(1.0 - r) ** 2 if printed else np.abs(1.0 - r**2) ** 2
    return 5.0 * delta * sigma2 * factor * np.sin(theta) ** 2 / (nu * math.log(10.0) * np.abs(p) ** 2 * p.imag)


def test_volume_term_is_the_corrected_eq_39_and_not_its_print():
    """[MJ89] (39) PRINTS |1 - R(theta)|^2. That is a misprint for |1 - R^2(theta)|^2 (note Sec. 0 item 1):
    - the paper's own figures: with the printed factor its model lies 1.5-3 dB below its own solid curves
      at 20-60 deg (Figs. 6, 9, 10); with the squared R it reproduces all seven printed model curves read
      (Figs. 6-10) to rms 0.3-0.8 dB (note Sec. 3.2, figure readings +-1.5 dB; also
      test_the_paper_figures_reproduce_only_with_the_corrected_volume_term below);
    - Jackson and Richardson (2007) (14.8) and our own derivation (bottom.py module docstring) have the
      squared form: |1 - R^2|^2 sin^2 / |P|^2 = |1 + R|^4 / rho^2 (algebra, ours).
    So our `volume_strength` stays as it is: do NOT "fix" it to the printed form. Unaveraged, ours equals
    the corrected (39) to 1e-12 relative; the printed form is ours times |1 + R|^-2 exactly."""
    theta = np.radians(np.arange(5.0, 86.0, 5.0))
    for name in ("medium_sand", "very_fine_silt"):
        par = bottom.sediment(name)
        for f in (20e3, 100e3):
            ours = bottom.volume_strength(theta, theta, f, C, **par)
            corrected = mj89_volume_39(theta, par["rho"], par["nu"], par["delta"], par["sigma2"])
            assert np.allclose(ours, corrected, rtol=1e-12, atol=0.0), (name, f)
            printed = mj89_volume_39(theta, par["rho"], par["nu"], par["delta"], par["sigma2"], printed=True)
            r = bottom.reflection(theta, par["rho"], par["nu"], par["delta"])
            assert np.allclose(printed, ours / np.abs(1.0 + r) ** 2, rtol=1e-12, atol=0.0), (name, f)
            above = theta > np.radians(45.0)                                 # above the critical angle |1 + R| > 1 here,
            assert np.all(printed[above] < ours[above])                      # so the printed form is lower (by 2-5 dB)


# --- C4: the 1989 model as an opt-in (roughness_model / volume_model "mourad_jackson_1989") -------------

MJ = "mourad_jackson_1989"
C1_MS = 1528.0                                   # [MJ89] c1 = 1.528 m/ms (text after (3)); used for k in the figure checks


def mj89_total(theta, f, c, par, **kw):
    return bottom.strength(theta, theta, f, c, roughness_model=MJ, volume_model=MJ, **dict(par, **kw))


# The inset ("surface" and two "bulk") parameter sets of the paper's backscatter figures (confirmed: printed
# numbers in the figures; note Sec. 2.5) and the readings of the paper's model curves (figure: +-1.5 dB, +-2 dB
# at 90 deg; note Sec. 3.2): (f Hz, rho, nu, delta, w2 cm^4, gamma2, sigma2, {grazing deg: reading dB}).
FIGURES = {
    "Fig. 6 Quinault surface": (35e3, 1.339, 1.080, 0.01727, 0.002070, 3.25, 0.002,
                                {10: -36, 20: -28.5, 30: -25.9, 45: -24.6, 60: -23.5, 70: -22.5, 80: -17, 90: -5}),
    "Fig. 7 Puget Sound surface": (25e3, 1.178, 1.006, 0.01531, 0.000599, 3.25, 0.002,
                                   {10: -32.5, 20: -27.5, 30: -26, 45: -24.8, 60: -24, 80: -22.5, 90: -2}),
    "Fig. 8 Arafura surface": (20e3, 1.150, 0.989, 0.00871, 0.000391, 3.25, 0.002,
                               {5: -38, 10: -31, 20: -28, 30: -26.5, 45: -25, 60: -24, 80: -23.5, 90: -0.5}),
    "Fig. 9 Stanic 1988 surface": (40e3, 1.428, 1.1020, 0.01601, 0.002589, 3.25, 0.002,
                                   {10: -36, 20: -28.5, 30: -25, 45: -24, 60: -22.7, 70: -21, 80: -14, 90: -5}),
    "Fig. 10 Stanic 1989 surface": (20e3, 2.176, 1.2323, 0.01643, 0.005999, 3.25, 0.002,
                                    {10: -34.5, 20: -27.5, 30: -24.5, 45: -22.5, 60: -20, 70: -15.5, 80: -9, 90: -1}),
    "Fig. 6 Quinault bulk": (35e3, 1.93, 1.080, 0.02601, 0.004225, 3.67, 0.001,
                             {10: -33, 20: -29.5, 30: -28, 45: -25.5, 60: -21, 70: -15.5, 80: -10, 90: -9.5}),
    "Fig. 8 Arafura bulk": (20e3, 1.49, 0.989, 0.01327, 0.002166, 3.18, 0.005,
                            {10: -28, 20: -25.5, 30: -23.5, 45: -21.5, 60: -20.5, 80: -18, 90: 0}),
}


def figure_case(name):
    f, rho, nu, delta, w2_cm4, gamma2, sigma2, readings = FIGURES[name]
    par = dict(rho=rho, nu=nu, delta=delta, w2=bottom.w2_from_cm4(w2_cm4, gamma2), gamma2=gamma2, sigma2=sigma2)
    deg = np.array(sorted(readings), dtype=float)
    return f, par, deg, np.array([readings[d] for d in sorted(readings)], dtype=float)


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_the_paper_figures_are_reproduced(name):
    """[MJ89] Figs. 6-10: the model with each figure's inset parameters and c = 1528 m/s lies within the reading
    error of every reading of the paper's own model curve (figure readings: +-1.5 dB, +-2 dB at 90 deg; one
    exception allowed by the note: Arafura surface at 10 deg, 1.65 dB off, held at 2.0 dB), rms <= 1.0 dB per
    curve (measured by the extraction: rms 0.34-0.77 dB). These tolerances are the readings' and are not tightened."""
    f, par, deg, read = figure_case(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                     # Fig. 10 is at 0.84 phi, outside the relations' range: it warns
        got = 10 * np.log10(mj89_total(np.radians(deg), f, C1_MS, par))
    err = got - read
    tol = np.where(deg == 90, 2.0, 1.5)
    if name == "Fig. 8 Arafura surface":
        tol = np.where(deg == 10, 2.0, tol)
    assert np.all(np.abs(err) <= tol), dict(zip(deg, np.round(err, 2)))
    assert np.sqrt(np.mean(err**2)) <= 1.0, np.sqrt(np.mean(err**2))


def test_the_paper_figures_reproduce_only_with_the_corrected_volume_term():
    """The evidence for C3: with (39) as printed (|1 - R|^2) the surface curves of Figs. 6 and 9 miss readings by
    more than the reading error (the printed form is 1.5-3 dB low at 20-60 deg), and every surface curve fits worse."""
    worse = 0
    for name in [n for n in FIGURES if "surface" in n]:
        f, par, deg, read = figure_case(name)
        theta = np.radians(deg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rough = bottom.roughness_strength(theta, theta, f, C1_MS, roughness_model=MJ, **par)
            s = bottom.mourad_jackson_1989_slope(f, C1_MS, **par)
            printed = bottom.wagner_backscatter_shadowing(theta, s, cutoff=2.0) * bottom._mj89_average(
                lambda x: mj89_volume_39(x, par["rho"], par["nu"], par["delta"], par["sigma2"], printed=True), theta, s)
            good = mj89_total(theta, f, C1_MS, par)
        e_printed, e_good = 10 * np.log10(rough + printed) - read, 10 * np.log10(good) - read
        assert np.sqrt(np.mean(e_printed**2)) > np.sqrt(np.mean(e_good**2)), name
        if name.startswith(("Fig. 6", "Fig. 9")):
            assert np.max(np.abs(e_printed)) > 1.5, name
            worse += 1
    assert worse == 2


def test_printed_volume_form_after_the_slope_average_is_2_7_to_4_9_db_low():
    """Note C3 (ours, `ours.py` Sec. 7): after the paper's slope average the printed form is 2.7-4.9 dB below the
    corrected one for medium sand at 20 kHz, 20-85 deg (c = 1500 m/s)."""
    par = bottom.sediment("medium_sand")
    theta = np.radians([20.0, 30.0, 45.0, 60.0, 75.0, 85.0])
    s = bottom.mourad_jackson_1989_slope(20e3, C, **par)
    printed = bottom.wagner_backscatter_shadowing(theta, s, cutoff=2.0) * bottom._mj89_average(
        lambda x: mj89_volume_39(x, par["rho"], par["nu"], par["delta"], par["sigma2"], printed=True), theta, s)
    diff = 10 * np.log10(printed / bottom.volume_strength(theta, theta, 20e3, C, volume_model=MJ, **par))
    assert np.all((diff > -4.95) & (diff < -2.65)), np.round(diff, 2)


def test_exact_identities_of_the_1989_model():
    """The note's identities (C4 (b)-(d), (f); Sec. 2.4 checks)."""
    sand = bottom.sediment("medium_sand")
    theta = np.radians([10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0])
    # (b) gamma2 = 3 (alpha = 1/2): the closed-form Kirchhoff (19)-(22) equals our exact integral (to 1e-6 dB;
    # measured 7e-8 dB, our spline); at 90 deg it equals it for every gamma2 (algebra), here to 1e-12 relative.
    for f in (20e3, 100e3):
        par = dict(sand, gamma2=3.0)
        mj = bottom.mourad_jackson_1989_kirchhoff(theta, f, C, **par)
        assert np.max(np.abs(10 * np.log10(mj / bottom.kirchhoff_strength(theta, theta, f, C, **par)))) < 1e-6
    for g2 in (2.5, 3.25, 3.67, 3.9):
        par = dict(sand, gamma2=g2)
        assert float(bottom.mourad_jackson_1989_kirchhoff(0.5 * np.pi, 50e3, C, **par)) == pytest.approx(
            float(bottom.kirchhoff_strength(0.5 * np.pi, 0.5 * np.pi, 50e3, C, **par)), rel=1e-12)
    # (c) the perturbation term (30)-(31) without the (k/10)^2 of (32) is our perturbation result (backscatter).
    for name in ("medium_sand", "very_fine_silt"):
        par = bottom.sediment(name)
        a = bottom.mourad_jackson_1989_perturbation(theta[:-1], 50e3, C, low_wavenumber=False, **par)
        b = bottom.perturbation_strength(theta[:-1], theta[:-1], 50e3, C, **par)
        assert np.max(np.abs(10 * np.log10(a / b))) < 1e-10, name
        # and the (k/10)^2 term changes it by -0.02 dB at 5-45 deg, -2.0 dB at 85 deg (note Sec. 2.4)
        g = np.radians([5.0, 45.0, 85.0])
        effect = 10 * np.log10(bottom.mourad_jackson_1989_perturbation(g, 50e3, C, **par)
                               / bottom.mourad_jackson_1989_perturbation(g, 50e3, C, low_wavenumber=False, **par))
        assert np.allclose(effect, [-0.018, -0.035, -2.008], atol=0.002), effect
    # (d) the printed weights and nodes (28)-(29) are the three-point Gauss-Hermite ones. The note asks 1e-6; the
    # printed a_(+-1) = 0.295410 is 1.02e-6 above sqrt(pi)/6 = 0.29540898 (a slip in the sixth decimal), so it
    # holds at 1.1e-6 only; the model uses the exact quadrature (the difference is 5e-6 dB).
    x, w = np.polynomial.hermite.hermgauss(3)
    printed = sorted(bottom.MJ89_GAUSS_HERMITE_PRINTED, key=lambda an: an[1])
    assert np.allclose([an for an, _ in printed], w, rtol=0.0, atol=1.1e-6)
    assert abs(printed[0][0] - w[0]) > 1e-6                                              # the slip, recorded
    assert np.allclose([xn for _, xn in printed], x, rtol=0.0, atol=1e-6)
    assert w.sum() == pytest.approx(math.sqrt(math.pi), rel=1e-15)
    # (f) the slope (18) is the TOTAL slope of the relief below the K where k h_s = 1/sqrt(2): sqrt(2) x ours.
    for name in ("medium_sand", "very_fine_silt", "sandy_gravel"):
        par = bottom.sediment(name)
        for f in (20e3, 50e3, 100e3):
            per_axis, _ = bottom.large_scale_slope(f, C, small_scale_kh=2**-0.5, **par)
            assert bottom.mourad_jackson_1989_slope(f, C, **par) == pytest.approx(math.sqrt(2.0) * per_axis, rel=1e-12)
    # The slope average of a constant is that constant (the weights sum to sqrt(pi)); of the volume term with no
    # slope it is the unaveraged term; the volume term unaveraged is ours (C3).
    ones = bottom._mj89_average(lambda x: np.ones_like(x), theta, 0.3)
    assert np.allclose(ones, 1.0, rtol=1e-14)
    vol = lambda x: bottom.mourad_jackson_1989_volume_unaveraged(x, **sand)            # noqa: E731
    assert np.allclose(bottom._mj89_average(vol, theta[:-1], 0.0), vol(theta[:-1]), rtol=1e-14)
    assert np.allclose(vol(theta[:-1]), bottom.volume_strength(theta[:-1], theta[:-1], 50e3, C, **sand), rtol=1e-12)
    # Wagner's S in the 1989 model is R with the printed rule S = 1 for t > 2.
    s = bottom.mourad_jackson_1989_slope(100e3, C, **sand)
    no_shade = bottom.volume_strength(theta, theta, 100e3, C, volume_model=MJ, shadowing="none", **sand)
    assert np.allclose(bottom.volume_strength(theta, theta, 100e3, C, volume_model=MJ, **sand),
                       no_shade * bottom.wagner_backscatter_shadowing(theta, s, cutoff=2.0), rtol=1e-14)


def test_the_interpolation_angle_is_the_15_db_point_where_the_asymptotics_hold():
    """(36)-(37): theta_kdB is where the Kirchhoff approximation has fallen 15 dB below its 90-deg value, exactly so
    as theta_kdB -> 90 deg; the "+4" keeps it at or above 45 deg. The note's test (e) asks 0.5 dB "where it is above
    45 deg"; that holds only where theta_kdB is near 90 deg (75-87 deg here: -15.28 to -15.38 dB). At 55-57 deg the
    drop is -13.2 (medium sand, 100 kHz) and -13.8 dB (sandy gravel, 20 kHz) (the note's own Sec. 2.4 numbers);
    at the 45-deg floor (sandy gravel, 100 kHz) -4.4 dB. Held as measured."""
    def drop(name, f):
        par = bottom.sediment(name)
        t = bottom.mourad_jackson_1989_kirchhoff_angle(f, C, **par)
        k = lambda x: bottom.mourad_jackson_1989_kirchhoff(x, f, C, **par)               # noqa: E731
        return math.degrees(t), float(10 * np.log10(k(t) / k(0.5 * np.pi)))
    for name, f in (("medium_sand", 20e3), ("very_fine_silt", 20e3), ("very_fine_silt", 100e3)):
        deg, db = drop(name, f)
        assert deg > 75.0 and db == pytest.approx(-15.0, abs=0.5), (name, f, deg, db)
    for name, f, want_deg, want_db in (("medium_sand", 100e3, 54.8, -13.2), ("sandy_gravel", 20e3, 57.5, -13.8),
                                       ("sandy_gravel", 100e3, 45.6, -4.4)):
        deg, db = drop(name, f)
        assert deg == pytest.approx(want_deg, abs=0.05) and db == pytest.approx(want_db, abs=0.05), (name, f, deg, db)
        assert deg >= 45.0


# The model against our default (the note Sec. 4.3, computed by the extraction with c = 1500 m/s; one decimal):
# [MJ89] backscattering strength and its difference from the default (small_slope + volume) at 5 / 10 / 20 / 30 /
# 45 / 60 / 75 / 85 deg.
NOTE_TABLE = {
    ("medium_sand", 20e3): ((-44.8, -37.3, -30.0, -26.7, -25.5, -23.3, -16.1, -4.9), (3.8, 0.8, 0.0, -0.2, -0.1, -0.1, 0.0, -0.2)),
    ("medium_sand", 100e3): ((-34.5, -30.3, -25.2, -22.7, -22.1, -20.0, -13.8, -9.8), (9.3, 3.0, -0.1, -0.8, 0.1, -0.8, -0.4, 0.1)),
    ("muddy_sand", 20e3): ((-47.0, -37.4, -28.0, -25.5, -24.4, -23.5, -21.7, -11.3), (1.6, 0.3, 0.0, 0.0, 0.0, 0.0, -0.1, -0.3)),
    ("very_fine_silt", 100e3): ((-41.7, -36.1, -31.9, -30.0, -28.4, -27.5, -26.6, -20.6), (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3)),
    ("sandy_gravel", 100e3): ((-24.6, -21.5, -18.3, -15.7, -12.9, -15.1, -13.5, -13.3), (11.5, 4.4, 0.1, -0.5, 1.0, -1.5, -0.2, 0.0)),
}


def test_against_our_default_as_the_note_tabulates():
    theta = np.radians([5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 85.0])
    for (name, f), (level, diff) in NOTE_TABLE.items():
        par = bottom.sediment(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                 # sandy gravel lies outside the paper's range: it warns
            mj = 10 * np.log10(mj89_total(theta, f, C, par))
        default = bottom.backscattering_strength_db(theta, f, C, **par)
        assert np.allclose(mj, level, atol=0.06), (name, f, np.round(mj, 2))
        assert np.allclose(mj - default, diff, atol=0.11), (name, f, np.round(mj - default, 2))   # two roundings


def test_the_default_does_not_change():
    """Neither key is a default: with no volume_model (or "stockhausen") the volume term is the one before, and
    `strength` with neither key is small_slope + that volume term, every float identical."""
    theta_i, theta_s, phi = np.radians([3.0, 10.0, 30.0, 60.0, 85.0]), np.radians([3.0, 25.0, 30.0, 10.0, 85.0]), np.radians([180.0, 40.0, 180.0, 0.0, 180.0])
    for name in ("medium_sand", "very_fine_silt", "sandy_gravel"):
        par = bottom.sediment(name)
        for f in (20e3, 100e3):
            vol = bottom.volume_strength(theta_i, theta_s, f, C, **par)
            assert np.array_equal(bottom.volume_strength(theta_i, theta_s, f, C, volume_model="stockhausen", **par), vol)
            assert np.array_equal(bottom.volume_strength(theta_i, theta_s, f, C, azimuth=phi, **par), vol)
            total = bottom.strength(theta_i, theta_s, f, C, azimuth=phi, **par)
            assert np.array_equal(total, bottom.small_slope_strength(theta_i, theta_s, f, C, azimuth=phi, **par) + vol)
    assert bottom.DEFAULT_ROUGHNESS_MODEL == "small_slope" and MJ not in bottom.ROUGHNESS_MODELS
    import inspect
    assert inspect.signature(bottom.volume_strength).parameters["volume_model"].default is None


def test_refusals_and_warnings_outside_what_the_model_covers():
    par = bottom.sediment("medium_sand")
    g = np.radians([10.0, 20.0])
    # Backscatter only: the paper is monostatic.
    for kw in ({"roughness_model": MJ}, {"volume_model": MJ}):
        with pytest.raises(ValueError, match="backscatter model"):
            bottom.strength(g, g + 0.1, 50e3, C, **kw, **par)
        with pytest.raises(ValueError, match="backscatter model"):
            bottom.strength(g, g, 50e3, C, azimuth=2.0, **kw, **par)
    # Outside 10-100 kHz and outside the parameters of 1-9 phi: a warning, not silence.
    with pytest.warns(UserWarning, match="outside the paper's 10-100 kHz"):
        bottom.strength(g, g, 300e3, C, roughness_model=MJ, **par)
    with pytest.warns(UserWarning, match="outside what the paper's grain-size relations span"):
        bottom.strength(g, g, 50e3, C, roughness_model=MJ, **bottom.sediment("sandy_gravel"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # medium sand at 50 kHz: inside, silent
        bottom.strength(g, g, 50e3, C, roughness_model=MJ, volume_model=MJ, **par)
    # Grain sizes outside 1-9 phi are refused by the paper's relations.
    for phi in (0.5, 9.5, -1.0):
        with pytest.raises(ValueError, match="1 <= Mz <= 9"):
            bottom.mourad_jackson_1989_sediment(phi)
    # Unknown values name the nearest; no shadowing with a model that has none.
    with pytest.raises(ValueError, match="did you mean 'mourad_jackson_1989'"):
        bottom.volume_strength(g, g, 50e3, C, volume_model="mourad_jackson", **par)
    with pytest.raises(ValueError, match="unknown roughness_model"):
        bottom.roughness_strength(g, g, 50e3, C, roughness_model="mourad_jackson_1990", **par)


def test_the_fit_refuses_the_1989_model_and_takes_the_composite_models_with_shadowing():
    from gpusonar.sim import sediment_fit
    par = bottom.sediment("medium_sand")
    deg = np.array([5.0, 8.0, 12.0, 20.0, 30.0, 40.0])
    with pytest.raises(ValueError, match="not fitted here.*volume term"):
        sediment_fit.fit(deg, 30e3, np.full(deg.shape, -30.0), rho=par["rho"], nu=par["nu"], delta=par["delta"], roughness_model=MJ)
    # composite (with Wagner's factor, its default now) is fitted as before: data made by it give back its w2.
    y = bottom.backscattering_strength_db(np.radians(deg), 30e3, NOMINAL, roughness_model="composite", **par)
    fit = sediment_fit.fit(deg, 30e3, y, rho=par["rho"], nu=par["nu"], delta=par["delta"], roughness_model="composite",
                           hold={"gamma2": par["gamma2"]}, n_starts=2)
    assert fit.params["w2"] == pytest.approx(par["w2"], rel=1e-3) and fit.params["sigma2"] == pytest.approx(par["sigma2"], rel=1e-3)


def test_the_fit_tool_does_not_offer_the_1989_model():
    import subprocess
    import sys
    from pathlib import Path
    tool = Path(__file__).resolve().parents[2] / "tools" / "fit_sediment.py"
    out = subprocess.run([sys.executable, str(tool), "--roughness-model", MJ], capture_output=True, text=True)
    assert out.returncode == 2 and "invalid choice" in out.stderr


def test_the_paper_surface_parameters_from_grain_size():
    """[MJ89] (1)-(7) with delta by (3) (Hamilton via J&R (11.3), c1 = 1528 m/s) give the printed inset values
    (note Sec. 2.5): to the printed digits, except Quinault's delta (printed 0.01727, ours 0.01728: one unit of
    the last digit) and Arafura, whose printed sigma2 0.002 and w2 0.000391 contradict (4) and (6) (0.001 and
    0.000518): an internal inconsistency of the paper, recorded."""
    cases = {3.0: (1.3394, 1.0800, 0.01727, 0.002070), 4.83: (1.1782, 1.006, 0.01531, 0.000599),
             2.59: (1.428, 1.1020, 0.01601, 0.002589), 5.3: (1.150, 0.989, 0.00871, None)}
    for mz, (rho, nu, delta, w2_cm4) in cases.items():
        got = bottom.mourad_jackson_1989_sediment(mz)
        rel = bottom.mourad_jackson_1989_relations(mz)
        digits = lambda x: 10.0 ** -(len(repr(x).split(".")[1]))                       # noqa: E731
        assert got["rho"] == pytest.approx(rho, abs=0.5 * digits(rho) + 1e-12), mz
        assert got["nu"] == pytest.approx(nu, abs=0.5 * digits(nu) + 1e-12), mz
        assert got["delta"] == pytest.approx(delta, abs=(1.5 if mz in (3.0, 2.59) else 0.5) * digits(delta)), mz
        if w2_cm4 is not None:
            assert rel["w2_cm4"] == pytest.approx(w2_cm4, abs=0.5 * digits(w2_cm4)), mz
            assert got["w2"] == pytest.approx(bottom.w2_from_cm4(w2_cm4, 3.25), rel=1e-3)
    arafura = bottom.mourad_jackson_1989_relations(5.3)
    assert arafura["sigma2"] == 0.001 and arafura["w2_cm4"] == pytest.approx(0.000518, abs=6e-7)    # 0.0005175
    # The printed 0.000391 is (6)'s rational form evaluated at 5.3 phi, not its 0.5 cm step.
    h = (2.03846 - 0.26923 * 5.3) / (1.0 + 0.076923 * 5.3)
    assert 0.00207 * h * h == pytest.approx(0.000391, abs=5e-7)
    # The seven-digit constants (C5): rho(3.0) = 1.339399, nu(3.0) = 1.079999.
    assert (round(bottom.mourad_jackson_1989_relations(3.0)["rho"], 6), round(bottom.mourad_jackson_1989_relations(3.0)["nu"], 6)) == (1.339399, 1.079999)
