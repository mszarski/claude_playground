"""The boundary models against the numbers printed in one open measurement at 68 kHz
(issue sonar-5wa). Characterisations: the bounds are what our models give plus a small margin,
and each docstring says how far off we are and in which direction. Nothing here was tuned.

  [G01] Gallaudet, "Shallow water acoustic backscatter and reverberation measurements using a
        68-kHz cylindrical array", dissertation (2001), https://hdl.handle.net/10945/11015
        (public release). Page numbers are the printed ones. One site, northeastern Gulf of
        Mexico, 190-205 m deep; the TVSS towed at 78 m; 68 kHz CW, 200 us; one wind reading,
        "6 kts (3m/s)", sea state 1.5, taken 4.5 hours before the acoustic runs.

What the source prints as numbers, and what it shows only in figures:
  - Ch. 3, Table I(e) (p. 55): porosity 66 %, fine fraction 78 %, sand 21 %, gravel 1 %, bulk
    grain size Mz 6.36 ("estimated from porosity data"). The text calls the bottom a "silt-sand
    mixture" (pp. 180, 182) and "silt and sand" (p. 46); it never calls it silty clay.
  - Ch. 3, Table VI (p. 60): the input parameters of the composite-roughness model fitted to the
    backscatter (BOTTOM_FIT below). Text pp. 47-48: rho, nu, delta from grain-size regressions,
    w2 "the APL-UW model default", "the best fit to the data with gamma = 3.00 and sigma2 = 0.003".
    The fitted curve and the data it was fitted to are only in Fig. 15a (p. 76): not used.
  - Ch. 6, Tables I and II (pp. 196-198): mean backscattering strength in fixed bands of grazing
    angle, averaged over three runs, "computed using eqn (17)" (p. 178: SL 216.8 dB, a fixed
    DI of 13 dB, spherical spreading plus 0.024 dB/m). BANDS below.
  - Ch. 5, Table III (p. 158): the maximum surface backscattering strength of four bubble clouds
    from breaking ship waves; the text (p. 140) says these "are within 1dB of those predicted by
    eqn (11) for saturation conditions". The angles of the clouds are only in figures.
  - Ch. 5, the mean surface backscatter against angle with the APL-UW model for 60 kHz in 3 m/s
    is only in Fig. 10 (p. 168): not used. The text (p. 140) says that for incidence angles of
    30-60 degrees it "is well within the +-5dB uncertainty estimated for the APL-UW model".

Caveats on the band means, all from the source: they are raw eqn (17) values; the transmit
pattern varied by up to -9 dB from the calibrated source level (pp. 49, 130), a bias that makes
the printed level LOW; SF2 and SF3 include surface reverberation and SF3 and SS3 boundary
reverberation (Table I), which makes them HIGH; the source does not say whether a band mean is
a mean of dB values or of intensities (for Rayleigh speckle, as at the seafloor where the
scintillation index is 1.0-1.2, the two differ by 2.5 dB; for SS2 and SS3, indices 3.7 and 14,
by more); SS1 is inside the towship's wake and is not used.
"""

import math
import warnings

import numpy as np
import pytest

from gpusonar.sim import bottom, surface

FREQUENCY = 68e3            # [G01] Ch. 3 Table I(a), p. 55
WIND_SPEED = 3.0            # [G01] Ch. 3 Table I(d), p. 55; Ch. 5 p. 126
MEAN_GRAIN_SIZE_PHI = 6.36  # [G01] Ch. 3 Table I(e), p. 55

# [G01] Ch. 3 Table VI, p. 60, as printed: density ratio, sound speed ratio, loss parameter,
# volume parameter, spectral exponent, spectral strength (cm^4 at wavenumber 2 pi / lambda = 1 cm^-1).
BOTTOM_FIT_PRINTED = {"rho": 1.1483, "nu": 0.9864, "delta": 0.0033, "sigma2": 0.003, "gamma2": 3.00, "w2_cm4": 0.000518}
BOTTOM_FIT = dict({k: v for k, v in BOTTOM_FIT_PRINTED.items() if k != "w2_cm4"},
                  w2=bottom.w2_from_cm4(BOTTOM_FIT_PRINTED["w2_cm4"], BOTTOM_FIT_PRINTED["gamma2"]))

# [G01] Ch. 6 Table I (pp. 196-197, grazing angles in degrees) and Table II (p. 198, mean
# backscattering strength in dB): region -> (lowest grazing, highest grazing, printed dB).
BANDS = {"SF1": (72.0, 90.0, -19.4), "SF2": (48.0, 66.0, -22.6), "SF3": (29.0, 37.0, -28.7),
         "SS1": (68.0, 90.0, -12.9), "SS2": (44.0, 66.0, -41.5), "SS3": (27.0, 38.0, -51.3)}

# [G01] Ch. 5 Table III, p. 158: maximum S_S of bubble clouds A-D, dB.
CLOUD_PEAKS_DB = (-24.0, -25.2, -23.5, -23.1)


def _band_db(fn, band, n=45):
    """10 log10 of the mean cross section over the band (angles uniform in grazing)."""
    lo, hi, _ = BANDS[band]
    g = np.radians(np.linspace(lo, hi, n))
    return float(10.0 * np.log10(np.mean(fn(g))))


def _bottom_band_db(params, band, model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _band_db(lambda g: bottom.strength(g, g, FREQUENCY, bottom.NOMINAL_SOUND_SPEED,
                                                  roughness_model=model, **params), band)


def test_printed_values_as_data():
    """[G01] the values typed above, and the unit conversion of the spectral strength: 0.000518
    cm^4 with gamma 3.00 is 5.18e-6 m^(4 - gamma2) with wavenumbers in rad/m."""
    assert BOTTOM_FIT["w2"] == pytest.approx(5.18e-6, rel=1e-12)
    for lo, hi, level in BANDS.values():
        assert 0.0 < lo < hi <= 90.0 and -60.0 < level < 0.0
    # The surface band means fall away from the zenith, the seafloor ones away from the nadir (p. 182).
    assert BANDS["SS1"][2] > BANDS["SS2"][2] > BANDS["SS3"][2]
    assert BANDS["SF1"][2] > BANDS["SF2"][2] > BANDS["SF3"][2]
    # Eq. (11) at 68 kHz, delta_r sin(t) / (8 pi delta) with delta = 2.55e-2 f^(1/3), f in kHz:
    # -22.84 dB at the normal, our `bubble_saturation_db`; every printed cloud peak is below it.
    sat_normal = 10 * math.log10(0.0136 / (8 * math.pi * 2.55e-2 * 68.0 ** (1 / 3)))
    assert float(surface.bubble_saturation_db(0.5 * math.pi, FREQUENCY)) == pytest.approx(sat_normal, abs=1e-9)
    assert sat_normal == pytest.approx(-22.84, abs=0.01)
    assert max(CLOUD_PEAKS_DB) < sat_normal


def test_table_vi_geoacoustic_ratios_are_our_grain_size_interpolation():
    """[G01] Table VI's rho, nu and delta came from grain-size regressions at Mz 6.36 (pp. 47-48);
    our linear interpolation of the Cintron (2001) class table at 6.36 phi (`method="table_linear"`)
    gives the same to within 4e-5 (rho 1.14828, nu 0.98644, delta 0.003284). The spectral strength is
    our grain-size rule too (0.0005175 cm^4, printed 0.000518). The two fitted values differ from our
    class defaults: gamma2 3.00 against 3.25, sigma2 0.003 against 0.001 (three times ours).

    Since A5 (docs/references/CONFORMANCE.md Sec. 1.1) the default method is the printed relations,
    J&R (11.1)-(11.3): rho 1.148249, nu 0.986430, delta 0.003267 at 6.36 phi. Their rho is 5.08e-5
    below the printed 1.1483, just outside that print's own rounding (half a unit of the fourth
    decimal, 5e-5) and inside it once the rounding of the printed Mz 6.36 is added (+-0.005 phi times
    the slope 0.0013 of (11.1) above 5.3 phi: 6.5e-6); their delta is 3.3e-5 from the printed 0.0033
    (the print's rounding 5e-5), outside the 2e-5 held for the interpolation. So the strict check
    below names `table_linear` and keeps its tolerances unchanged; the relations get their own check
    at the print's rounding (5e-5, plus 1e-5 on rho for the rounding of Mz). Neither tolerance was
    widened."""
    ours = bottom.sediment_from_grain_size(MEAN_GRAIN_SIZE_PHI, method="table_linear")
    assert ours["rho"] == pytest.approx(BOTTOM_FIT["rho"], abs=5e-5)
    assert ours["nu"] == pytest.approx(BOTTOM_FIT["nu"], abs=5e-5)
    assert ours["delta"] == pytest.approx(BOTTOM_FIT["delta"], abs=2e-5)
    rel = bottom.sediment_from_grain_size(MEAN_GRAIN_SIZE_PHI)                     # the default: the relations
    assert rel["rho"] == pytest.approx(BOTTOM_FIT["rho"], abs=6e-5) and abs(rel["rho"] - BOTTOM_FIT["rho"]) > 5e-5
    assert rel["nu"] == pytest.approx(BOTTOM_FIT["nu"], abs=5e-5)
    assert rel["delta"] == pytest.approx(BOTTOM_FIT["delta"], abs=5e-5)
    assert {k: rel[k] for k in ("w2", "gamma2", "sigma2")} == {k: ours[k] for k in ("w2", "gamma2", "sigma2")}
    assert bottom.grain_size_w2_cm4(MEAN_GRAIN_SIZE_PHI) == pytest.approx(BOTTOM_FIT_PRINTED["w2_cm4"], abs=6e-7)  # 0.0005175 printed to 6 places
    assert ours["gamma2"] == 3.25 and ours["sigma2"] == 0.001


@pytest.mark.parametrize("model", bottom.ROUGHNESS_MODELS)
def test_bottom_model_with_the_site_fit_against_the_printed_band_means(model):
    """`fluid_sediment` with Table VI as its six inputs, every roughness model, band means:
      SF2 (48-66 deg): ours -22.8 dB, printed -22.6 dB: 0.2 dB low.
      SF3 (29-37 deg): ours -24.8 dB, printed -28.7 dB: 3.9 dB HIGH.
    Both are inside the source's own systematic uncertainty (up to -9 dB from the transmit
    pattern, plus reverberation in both bands). The five roughness models give the same numbers
    (within 0.05 dB) because at this site the roughness term is 20 to 30 dB below the volume term
    off the specular direction: for a sediment slower than water (nu = 0.986) with density
    ratio 1.15 the interface barely scatters, so these data cannot tell the roughness models
    apart. Not compared: SF1 (72-90 deg) contains the specular peak, whose band mean depends on
    beam width and averaging more than on the model (ours: -14.5 dB as a mean of intensity,
    -19.3 dB as a mean of dB, small_slope; printed -19.4 dB). The fitted model curve itself is
    only in Fig. 15a."""
    sf2 = _bottom_band_db(BOTTOM_FIT, "SF2", model) - BANDS["SF2"][2]
    sf3 = _bottom_band_db(BOTTOM_FIT, "SF3", model) - BANDS["SF3"][2]
    assert -0.5 < sf2 < 0.1, sf2          # measured -0.2
    assert 3.4 < sf3 < 4.4, sf3           # measured +3.9
    g = np.radians(np.linspace(29.0, 66.0, 38))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rough = bottom.roughness_strength(g, g, FREQUENCY, bottom.NOMINAL_SOUND_SPEED, roughness_model=model, **BOTTOM_FIT)
    vol = bottom.volume_strength(g, g, FREQUENCY, bottom.NOMINAL_SOUND_SPEED, **BOTTOM_FIT)
    assert np.all(10 * np.log10(rough / vol) < -20.0)


def test_class_default_against_the_site():
    """Our nearest named row is `fine_silt` (6 to 7 phi, the class that holds Mz 6.36; Cintron's
    "Fine Silt"); `sandy_mud` (6 phi) and `silty_clay` (8 phi) give the same within 0.2 dB here.
    With the class defaults (default small_slope) the band means are
      SF2: -27.6 dB against printed -22.6: 5.0 dB low;  SF3: -29.5 against -28.7: 0.8 dB low,
    and 4.8 dB below the site fit in both bands: the volume term, whose parameter is 0.001 in
    the class and 0.003 in the fit (10 log10(3) = 4.8 dB). The class default is good to about
    5 dB at this site, on the low side, as REFERENCES Sec. 11 already says (+-3 to 6 dB)."""
    for name in ("fine_silt", "sandy_mud", "silty_clay"):
        params = bottom.sediment(name)
        sf2 = _bottom_band_db(params, "SF2", None) - BANDS["SF2"][2]
        sf3 = _bottom_band_db(params, "SF3", None) - BANDS["SF3"][2]
        assert -5.4 < sf2 < -4.6, (name, sf2)     # measured -5.0 / -4.9 / -5.0
        assert -1.2 < sf3 < -0.4, (name, sf3)     # measured -0.8 / -0.8 / -0.9
    params = bottom.sediment("fine_silt")
    assert params["sigma2"] == 0.001
    for band in ("SF2", "SF3"):
        gap = _bottom_band_db(params, band, None) - _bottom_band_db(BOTTOM_FIT, band, None)
        assert gap == pytest.approx(-10 * math.log10(3.0), abs=0.3), band


def test_surface_models_lie_above_the_printed_band_means_at_3_m_s():
    """FINDING. At 68 kHz and 3 m/s the bubble-layer model (bubbles plus facets, no small-scale
    roughness term) is ABOVE the printed band means, not below them, with the two extinction laws
    taken from loss laws (the default since A1, `dahl_1997`, is not: next test):
      extinction apl_1994 (the default before A1):
                                      SS2 (44-66 deg) -38.1 dB, 3.4 dB above -41.5;
                                      SS3 (27-38 deg) -38.3 dB, 13.0 dB above -51.3.
      extinction hall_novarini:       -28.9 and -29.1 dB, 12.6 and 22.2 dB above.
    At this wind the bubble term is in its sparse limit (eq. (12)), proportional to the optical
    depth beta_I, which comes from our loss law (the source's law for it is not printed): beta_I
    = 0.0024 (apl_1994) or 0.021 (hall_novarini), hence the 9.5 dB between the two. The model
    is flat from 27 to 66 degrees; the printed means fall 10 dB over that range. D-87 / D-132
    call the model a lower bound between 30 and 70 degrees because a positive term is omitted;
    these printed means do not support that at 3 m/s. They do not establish a defect either:
    the printed levels may be biased low by up to 9 dB (transmit pattern), the averaging is not
    stated and the SS2/SS3 statistics are heavy-tailed (scintillation 3.7 and 14), SS3 holds
    bottom reverberation, the site was dominated by the towship's wake, and the one wind reading
    was 4.5 hours old. The empirical laws are further above: chapman_harris (fitted at 0.4-6.4
    kHz) 15.1 and 19.9 dB, kraus_surface 10.7 and 15.8 dB."""
    def law(model, ext=None):
        if model == "bubble_layer":
            return lambda g: 10 ** (surface.bubble_layer_surface_db(g, WIND_SPEED, FREQUENCY, extinction=ext) / 10)
        return lambda g: 10 ** (surface._empirical_surface_db(model, g, WIND_SPEED, FREQUENCY) / 10)

    expected = {("bubble_layer", "apl_1994"): (3.4, 13.0), ("bubble_layer", "hall_novarini"): (12.6, 22.2),
                ("chapman_harris", None): (15.1, 19.9), ("kraus_surface", None): (10.7, 15.8)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # 68 kHz and 3 m/s are inside the stated range
        _band_db(law("bubble_layer", "apl_1994"), "SS2")
    for (model, ext), (ss2, ss3) in expected.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got2 = _band_db(law(model, ext), "SS2") - BANDS["SS2"][2]
            got3 = _band_db(law(model, ext), "SS3") - BANDS["SS3"][2]
        assert got2 == pytest.approx(ss2, abs=0.3), (model, ext, got2)
        assert got3 == pytest.approx(ss3, abs=0.3), (model, ext, got3)
        assert got2 > 0.0 and got3 > 0.0                    # every model above the printed means
    assert surface.bubble_layer_optical_depth(WIND_SPEED, FREQUENCY, extinction="apl_1994") == pytest.approx(0.0024, abs=1e-4)


def test_surface_with_the_published_extinction_law_at_3_m_s():
    """`extinction: dahl_1997` (the default since A1; Dahl et al. 1997, Table II, the beta_I fit at 17 degrees,
    North Sea; docs/references/surface_papers_mcdaniel_dahl.md Sec. 2.8, 3.4). At 68 kHz, 3 m/s:
      beta_I 5.8e-4 (log10 f interpolation between the 50 and 70 kHz fits; ours), 6.2 dB below
      apl_1994's 2.4e-3;
      SS2 (44-66 deg) -43.7 dB, 2.2 dB BELOW -41.5 (apl_1994: 3.4 dB above);
      SS3 (27-38 deg) -44.4 dB, 6.9 dB above -51.3 (apl_1994: 13.0 dB above).
    The note computed -43.4 and -44.1 dB (-1.9 / +7.2) with the 70 kHz fit scaled to 68 kHz as f^0.8
    (beta_I 6.2e-4) instead; that construction is reproduced here too: the two differ by 0.3 dB. So the
    published law removes 6.1 of the 13.0 dB at 27-38 degrees and all of the 3.4 dB at 44-66
    degrees. What remains at 27-38 degrees (about 7 dB) is not explained by any law in hand; the
    averaging of the printed means (dB or intensity: 3.4-4.5 dB under Dahl and Plant's statistics)
    and the omitted roughness term are candidates, neither proven (the note Sec. 3.4)."""
    def model(beta_i):
        def fn(g):
            b = beta_i / np.sin(g)
            level = surface.BUBBLE_RERADIATION_DAMPING * np.sin(g) / (8 * np.pi * surface.bubble_damping_constant(FREQUENCY))
            # the facets attenuated by the layer, exp(-2 b) (A2; 0.001 dB in these bands at 3 m/s)
            return level * (1 + 8 * b * np.exp(-2 * b) - np.exp(-4 * b)) + surface.facet_backscatter(g, WIND_SPEED) * np.exp(-2 * b)
        return fn

    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # 68 kHz and 3 m/s are inside both stated ranges
        beta_i = surface.bubble_layer_optical_depth(WIND_SPEED, FREQUENCY, extinction="dahl_1997")
        ours = lambda g: 10 ** (surface.bubble_layer_surface_db(g, WIND_SPEED, FREQUENCY, extinction="dahl_1997") / 10)  # noqa: E731
        ss2 = _band_db(ours, "SS2") - BANDS["SS2"][2]
        ss3 = _band_db(ours, "SS3") - BANDS["SS3"][2]
    assert beta_i == pytest.approx(5.81e-4, rel=0.005)
    assert 10 * math.log10(surface.bubble_layer_optical_depth(WIND_SPEED, FREQUENCY, extinction="apl_1994") / beta_i) == pytest.approx(6.2, abs=0.1)
    assert surface.bubble_layer_optical_depth(WIND_SPEED, FREQUENCY) == beta_i               # the default (A1)
    assert _band_db(model(beta_i), "SS2") - BANDS["SS2"][2] == pytest.approx(ss2, abs=1e-9)   # the formula typed here
    assert ss2 == pytest.approx(-2.2, abs=0.1), ss2          # measured -2.15 (band mean -43.65 dB)
    assert ss3 == pytest.approx(+6.9, abs=0.1), ss3          # measured +6.88 (band mean -44.42 dB)
    note_beta = 10 ** (-5.75 + 5.35 * math.log10(WIND_SPEED)) * (68.0 / 70.0) ** 0.8         # the note's construction
    assert note_beta == pytest.approx(6.2e-4, rel=0.005)
    assert _band_db(model(note_beta), "SS2") == pytest.approx(-43.4, abs=0.05)                 # measured -43.41
    assert _band_db(model(note_beta), "SS3") == pytest.approx(-44.1, abs=0.05)                 # measured -44.13
    assert ss2 < 0.0 < ss3                                   # below the printed mean at 44-66 deg, still above at 27-38


def test_bubble_cloud_peaks_and_the_saturation_level():
    """[G01] p. 140: the cloud peaks of Table III "are within 1dB of those predicted by eqn (11)
    for saturation". Our eq. (11) at 68 kHz reaches each printed peak to within 1 dB at some
    grazing angle (from 27.5 deg for -25.2 dB to 48.4 deg for -23.1 dB; the clouds lie outside
    the wake, below 72 deg of grazing), so the statement holds for our saturation level as it
    is written. Our model with the ambient 3 m/s wind stays 19 to 21 dB below the clouds with the
    default extinction `dahl_1997` (13 to 15 dB with `apl_1994`, the default before A1): clouds
    from breaking ship waves are not in it, by design."""
    g = np.radians(np.linspace(1.0, 72.0, 7101))
    sat = surface.bubble_saturation_db(g, FREQUENCY)
    first = []
    for peak in CLOUD_PEAKS_DB:
        near = np.abs(sat - peak) <= 1.0
        assert np.any(near), peak
        first.append(float(np.degrees(g[near][0])))
    assert first == pytest.approx([37.5, 27.5, 43.0, 48.4], abs=0.1)
    ambient = surface.bubble_layer_surface_db(np.radians([30.0, 45.0, 60.0]), WIND_SPEED, FREQUENCY)
    gap = min(CLOUD_PEAKS_DB) - float(np.max(ambient))
    assert 18.7 < gap < 19.7, gap                            # measured 19.2 (to the lowest peak)
    ambient = surface.bubble_layer_surface_db(np.radians([30.0, 45.0, 60.0]), WIND_SPEED, FREQUENCY, extinction="apl_1994")
    gap = min(CLOUD_PEAKS_DB) - float(np.max(ambient))
    assert 12.5 < gap < 13.5, gap                            # measured 13.1 (to the lowest peak)
