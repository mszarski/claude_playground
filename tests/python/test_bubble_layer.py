"""The bubble layer as an effective medium (sim/surface.py: `hall_novarini_layer`,
`hall_novarini_screened`) and independent checks of the population behind every bubble law.

Outside numbers used here, each from a document opened for this work:
  - Nennouche et al., arXiv:2603.23727v2, eq. (24) and Figs. 9-10: the number of bubbles per
    m^3 and the optical scattering coefficient of the Hall-Novarini population (fixes the
    unit of n: m^-4);
  - Flatau et al., "Remote sensing of bubble clouds in seawater", arXiv:physics/0006060,
    Sec. 2.2, quoting Walsh and Mulhearn (1987): air volume fraction 1e-7 at 2 m depth in a
    10.5 m/s wind and 1e-8 at 6.2 m/s;
  - Beelen et al., arXiv:2410.14415v1, eqs. (6)-(9): the damped bubble response.
"""
import warnings

import numpy as np
import pytest

from gpusonar.sim import surface
from gpusonar.sim.timespread import boundary_parameters

C = 1500.0
RADII = np.geomspace(surface.R_MIN, surface.R_MAX, 20001)


def test_unit_of_the_population_against_the_sources_own_count():
    """Eq. (24) of the source: N_b(0) = 1.6e10 r_ref^4 / (3 r_min^3) (U/13)^3, and its Fig. 9
    shows between 1e7 and 1e8 bubbles per m^3 at the surface in a 13 m/s wind. Only n in
    m^-4 (per m^3 per metre of radius) gives that; per m^3 per micrometre would be 1e6 times it."""
    count = np.trapezoid(surface.hall_novarini_density(RADII, 0.0, 13.0), RADII)
    printed = 1.6e10 * 54.4e-6**4 / (3.0 * 10e-6**3)
    assert printed == pytest.approx(4.67e7, rel=0.01)
    assert count == pytest.approx(printed, rel=0.02)                 # the source's "approximately": radii above r_ref add little
    # Its Fig. 10: scattering coefficient for light, 2 x mean geometric cross section x count,
    # just under 0.09 per metre at the surface (eq. 25 approximates it as 2 * 3 pi r_min^2 N_b = 0.088).
    optical = np.trapezoid(surface.hall_novarini_density(RADII, 0.0, 13.0) * 2.0 * np.pi * RADII**2, RADII)
    assert 0.07 < optical < 0.09


def test_void_fraction_against_an_independent_quote():
    """Air volume fraction at 2 m: 1e-7 at 10.5 m/s and 1e-8 at 6.2 m/s (Walsh and Mulhearn as
    quoted in arXiv:physics/0006060). The population gives 0.76e-7 and 0.15e-8: it agrees at
    the higher wind and is 6.5 times lower at the lower one, where its layer is 0.4 m deep."""
    void = lambda u: float(np.trapezoid(surface.hall_novarini_density(RADII, 2.0, u) * 4.0 / 3.0 * np.pi * RADII**3, RADII))   # noqa: E731
    assert void(10.5) == pytest.approx(1e-7, rel=0.35)
    assert 0.1e-8 < void(6.2) < 1e-8


def test_heat_function_limits_and_resonance():
    """Phi -> 3 gamma for a large bubble (adiabatic) and -> 3 for a small one (isothermal);
    the natural frequency of a large bubble is Minnaert's; damping at resonance is a few
    percent to ten percent, with radiation k a = 0.0136 its smallest part."""
    w = 2 * np.pi * 1e3
    for a, want in ((5e-2, 3 * 1.4), (2e-7, 3.0)):
        w0_sq, _ = surface.bubble_oscillator(a, 1e3)
        p0 = surface.ATMOSPHERE + 2 * surface.SURFACE_TENSION / a
        re_phi = w0_sq * surface.WATER_DENSITY * a * a / p0 + 2 * surface.SURFACE_TENSION / (a * p0)
        assert re_phi == pytest.approx(want, rel=0.01)
    a = float(surface.resonant_radius(1e3))                            # 3 mm: adiabatic, surface tension negligible
    w0_sq, b = surface.bubble_oscillator(a, 1e3)
    assert np.sqrt(w0_sq) / w == pytest.approx(1.0, abs=0.01)
    for f, low, high in ((10e3, 0.04, 0.07), (30e3, 0.06, 0.09), (100e3, 0.09, 0.13)):
        a = float(surface.resonant_radius(f))
        w0_sq, b = surface.bubble_oscillator(a, f)
        delta = 2 * b / (2 * np.pi * f)
        assert low < delta < high
        assert 2 * np.pi * f * a / C == pytest.approx(0.0136, rel=0.01) and 2 * np.pi * f * a / C < 0.35 * delta
        assert 0.94 < np.sqrt(w0_sq) / (2 * np.pi * f) < 0.99           # below Minnaert: the gas is not quite adiabatic


def test_extinction_obeys_the_optical_theorem_and_exceeds_the_resonant_estimate():
    """The medium's extinction is the sum of 4 pi Im f / k over the population
    (dilute limit of eq. 6), and the scattering part of it is the radiation share of the
    damping. Against the resonant approximation of `bubble_extinction` it is 20 to 45 % higher:
    7 % from the lower resonance frequency, the rest from bubbles smaller than resonance,
    which absorb through viscous and thermal damping."""
    for f, low, high in ((10e3, 1.3, 1.5), (30e3, 1.25, 1.45), (100e3, 1.15, 1.35)):
        w, k = 2 * np.pi * f, 2 * np.pi * f / C
        w0_sq, b = surface.bubble_oscillator(RADII, f)
        n = surface.hall_novarini_density(RADII, 0.0, 10.0)
        amp = RADII * w * w / (w0_sq - w * w + 2j * b * w)
        direct = np.trapezoid(4 * np.pi / k * np.abs(amp.imag) * n, RADII)
        medium = surface.bubble_layer_medium(f, 0.0, 10.0, C)
        assert float(medium["extinction"][0]) == pytest.approx(direct, rel=0.02)
        assert low < float(medium["extinction"][0]) / float(surface.bubble_extinction(f, 0.0, 10.0, C)) < high
        # Scattering / extinction of one bubble = radiation damping / total damping.
        ratio = 4 * np.pi * np.abs(amp) ** 2 / (4 * np.pi / k * np.abs(amp.imag))
        assert np.allclose(ratio, (w * w * RADII / (2 * C)) / b, rtol=1e-9)
        assert 0.05 < float(medium["scattering"][0] / medium["extinction"][0]) < 0.3
        assert float(medium["phase_speed"][0]) < C                      # most of the air is in bubbles below resonance: a slow layer


def test_layer_loss_limits():
    f, u = 30e3, 6.0
    z, medium = surface._layer_profile(f, u, C)
    vertical = 10 / np.log(10) * 2 * np.trapezoid(medium["extinction"], z)
    g = np.radians([0.0, 1.0, 2.0, 5.0, 10.0, 30.0, 90.0])
    loss = surface.bubble_layer_loss_db(g, u, f, C)
    assert loss[-1] == pytest.approx(vertical, rel=1e-6)                # straight down: no refraction
    assert np.all(np.diff(loss) < 0) and np.isfinite(loss[0])           # finite at zero grazing without a floor
    # Refraction shortens the path: below the straight-path value, by more at small angles.
    straight = vertical / np.sin(g[1:])
    assert np.all(loss[1:] <= straight * (1 + 1e-9))
    assert loss[4] / straight[3] > 0.85 and loss[1] / straight[0] < 0.6
    # Snell at the surface for the zero-grazing ray: the angle there is acos(c_m / c).
    assert np.degrees(np.arccos(medium["phase_speed"][0] / C)) > 1.0
    assert surface.bubble_layer_loss_db(g, 0.0, f, C) == pytest.approx(0.0, abs=1e-9)
    assert isinstance(surface.bubble_layer_loss_db(0.2, u, f, C), float)
    # Wind cubed at fixed layer depth, while the layer is thin (little refraction).
    assert surface.bubble_layer_loss_db(np.pi / 2, 3.0, f, C) == pytest.approx(vertical / 8.0, rel=0.01)


def test_screened_layer_saturates_at_the_scattered_return():
    """An opaque layer returns the single-scattering albedo of a half-space,
    (w/2)(1 - m ln(1 + 1/m)): about 13 to 15 dB down at 10 degrees, whatever the wind,
    where pure extinction gives hundreds of dB. A thin layer is unchanged."""
    g = np.radians(10.0)
    m = np.sin(g)
    for f, u in ((100e3, 10.0), (100e3, 15.0), (30e3, 15.0)):
        z, medium = surface._layer_profile(f, u, C)
        albedo = np.trapezoid(medium["scattering"], z) / np.trapezoid(medium["extinction"], z)
        floor = -10 * np.log10(0.5 * albedo * (1 - m * np.log1p(1 / m)))
        assert surface.bubble_layer_loss_db(g, u, f, C) > 100.0
        assert surface.bubble_screened_loss_db(g, u, f, C) == pytest.approx(floor, abs=0.01)
        assert 12.0 < floor < 16.0
    thin = surface.bubble_layer_loss_db(g, 3.0, 10e3, C)
    assert surface.bubble_screened_loss_db(g, 3.0, 10e3, C) == pytest.approx(thin, rel=0.1)
    assert surface.bubble_screened_loss_db(g, 3.0, 10e3, C) <= thin


def test_adopted_empirical_law_against_hand_values_and_arlpy():
    """`apl_1994`: 1.26e-3 / sin(t) U^1.57 f_kHz^0.85, and below 6 m/s the 6 m/s value times
    exp(1.2 (U - 6)). Hand values: 10^1.57 = 37.1535, 30^0.85 = 18.0124, 6^1.57 = 16.6610,
    10^0.85 = 7.0795, sin 10 deg = 0.173648."""
    g = np.radians(10.0)
    assert float(surface.apl_1994_bubble_loss_db(g, 10.0, 30e3)) == pytest.approx(1.26e-3 / 0.173648 * 37.1535 * 18.0124, rel=1e-4)
    assert float(surface.apl_1994_bubble_loss_db(g, 10.0, 30e3)) == pytest.approx(4.856, abs=0.002)
    assert float(surface.apl_1994_bubble_loss_db(g, 3.0, 10e3)) == pytest.approx(1.26e-3 / 0.173648 * 16.6610 * 7.0795 * np.exp(-3.6), rel=1e-4)
    assert float(surface.apl_1994_bubble_loss_db(g, 6.0, 10e3)) == pytest.approx(float(surface.apl_1994_bubble_loss_db(g, 5.999999, 10e3)), rel=1e-5)   # continuous at 6 m/s
    assert surface.apl_1994_bubble_loss_db(0.0, 10.0, 30e3) == surface.apl_1994_bubble_loss_db(surface.MIN_GRAZING, 10.0, 30e3)
    assert surface.apl_1994_bubble_loss_db(np.pi / 2, 0.0, 30e3) < 1e-3
    with pytest.warns(UserWarning, match="above"):
        surface.apl_1994_bubble_loss_db(g, 25.0, 30e3)
    par = boundary_parameters({"wind_speed": 10.0, "rms_height": 0.0, "bubble_loss_db": {"model": "apl_1994"}}, surface=True, frequency=30e3)
    assert np.allclose(par["bubble_loss"][1], surface.apl_1994_bubble_loss_db(par["bubble_loss"][0], 10.0, 30e3))
    oracle = pytest.importorskip("arlpy.uwa")              # optional: arlpy is not a dependency
    for u in (2.0, 5.0, 6.0, 10.0, 15.0):
        for f in (3e3, 10e3, 30e3, 100e3):
            for deg in (3.0, 10.0, 45.0, 90.0):
                want = -20.0 * np.log10(float(oracle.bubble_surface_loss(u, f, np.pi / 2 - np.radians(deg))))
                assert float(surface.apl_1994_bubble_loss_db(np.radians(deg), u, f)) == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_where_the_population_law_departs_from_the_adopted_one():
    """The ratio of the two in dB (the same at every angle: both go as 1 / sin): they cross
    near 6 to 8 m/s (0.4 to 2 between 3.5 and 100 kHz at 6 m/s), and ours grows faster with
    frequency (about f^1.4 against f^0.85) and with wind above 7.5 m/s (U^3 times a layer
    depth that grows with U, against U^1.57). Below 6 m/s the adopted law falls
    exponentially and ours as U^3, so ours is the larger again."""
    g = np.radians(10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio = lambda u, f: float(surface.bubble_loss_db(g, u, f)) / float(surface.apl_1994_bubble_loss_db(g, u, f))   # noqa: E731
        assert 0.3 < ratio(6.0, 3.5e3) < 0.5 and 1.1 < ratio(6.0, 30e3) < 1.5 and 1.8 < ratio(6.0, 100e3) < 2.3
        assert ratio(3.0, 10e3) > 3.0
        assert ratio(10.0, 30e3, ) == pytest.approx(float(surface.bubble_loss_db(0.9, 10.0, 30e3)) / float(surface.apl_1994_bubble_loss_db(0.9, 10.0, 30e3)))
        assert 2.0 < ratio(10.0, 10e3) < 2.6 and 4.0 < ratio(10.0, 30e3) < 5.0 and 7.0 < ratio(10.0, 100e3) < 8.0
        assert ratio(15.0, 30e3) > 2.5 * ratio(10.0, 30e3)
        slope = np.log(float(surface.bubble_loss_db(g, 10.0, 100e3)) / float(surface.bubble_loss_db(g, 10.0, 10e3))) / np.log(10.0)
    assert 1.3 < slope < 1.45


def test_time_spread_takes_the_new_models_and_the_old_one_is_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in (("hall_novarini_layer", surface.bubble_layer_loss_db), ("hall_novarini_screened", surface.bubble_screened_loss_db)):
            par = boundary_parameters({"wind_speed": 6.0, "rms_height": 0.0, "bubble_loss_db": {"model": name}}, surface=True, frequency=30e3)
            grid, table = par["bubble_loss"]
            assert np.allclose(table, fn(grid, 6.0, 30e3))
        # The numbers of the first law, as reported before this work (10 m/s, 10 degrees).
        old = [float(surface.bubble_loss_db(np.radians(10.0), 10.0, f)) for f in (10e3, 30e3, 100e3)]
    assert old == pytest.approx([4.35, 21.59, 101.72], abs=0.01)
