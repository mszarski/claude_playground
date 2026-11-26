"""Forward scatter from the sea surface against Dahl's papers: the large-scale slope law
(docs/references/CONFORMANCE.md A4).

Sources (docs/references/surface_forward_scatter_dahl_1996_2004.md, "[NF]"; and
surface_bistatic_dahl_1999_mcdaniel_1986.md, "[NB]"):
- Dahl 1996, JASA 100, 748-758: eq. (A7) s_L^2 = 0.0046 ln(2.1 U^2) and Table I (printed s_L at the
  measured wind; measured kd*/(2 pi)), [NF] Secs. 2.2 and 2.4 (confirmed);
- Dahl 2004, JASA 115, 589-599: a low-pass slope S_L = 0.152 at 7 m/s and Dahl 2001's 0.146,
  [NF] Sec. 2.6 (confirmed);
- Dahl 1999, JASA 105, 2155-2169: s_L^2 = 0.017 at about 4 m/s, [NB] Sec. 2.2 (confirmed).

Time spreading takes this slope from the wind by default; the Cox-Munk total it used before is
`slope_law: cox_munk_total`.
"""

import warnings

import numpy as np
import pytest

from gpusonar import acoustics
from gpusonar.sim import ImageSourceProvider
from gpusonar.sim.timespread import boundary_parameters, wrap

# Dahl 1996 Table I (p. 752), printed: set -> (SD m, range m, grazing deg, U m/s, s_L, kd*/(2 pi)). RD = 66 m.
TABLE_I = {"1": (147.0, 1000.0, 12.0, 7.6, 0.149, 5.0), "2": (57.0, 1000.0, 7.0, 5.3, 0.137, 14.5),
           "3": (147.0, 576.0, 20.0, 3.2, 0.119, 4.9), "4": (147.0, 724.0, 16.0, 1.5, 0.085, 17.7),
           "5A": (147.0, 714.0, 17.0, 5.0, 0.135, 5.0), "5B": (57.0, 714.0, 10.0, 5.0, 0.135, 12.5)}
RD = 66.0


def test_the_law_is_dahl_1996_eq_a7_and_gives_the_printed_slopes_of_table_i():
    """Table I's s_L column is eq. (A7) at the measured wind (printed to 3 decimals)."""
    for sd, rng, g, u, s_l, _ in TABLE_I.values():
        assert acoustics.large_scale_mean_square_slope(u) == pytest.approx(0.0046 * np.log(2.1 * u * u), rel=1e-12)
        assert np.sqrt(acoustics.large_scale_mean_square_slope(u)) == pytest.approx(s_l, abs=5e-4)
    # [NF] WP2's values.
    for u, want in ((1.5, 0.0071), (3.2, 0.0141), (5.0, 0.0182), (7.6, 0.0221)):
        assert acoustics.large_scale_mean_square_slope(u) == pytest.approx(want, abs=5e-5)


def test_the_law_against_the_other_two_papers():
    # Dahl 2004: "a simple formula" of Dahl 2001 gives S_L = 0.146 at 7 m/s; the measured low-pass surface 0.152.
    s7 = np.sqrt(acoustics.large_scale_mean_square_slope(7.0))
    assert s7 == pytest.approx(0.146, abs=5e-4)
    assert s7 == pytest.approx(0.152, rel=0.05)
    # Dahl 1999 prints s_L^2 = 0.017 for runs 2-3 at about 4 m/s: (A7) 0.0162, 5 % lower.
    assert acoustics.large_scale_mean_square_slope(4.0) == pytest.approx(0.017, rel=0.06)
    # Smaller than the Cox-Munk total by 1.37 to 2.2 over 3.2 to 10 m/s ([NF] Sec. 3.1).
    ratio = [acoustics.mean_square_slope(u) / acoustics.large_scale_mean_square_slope(u) for u in (1.5, 3.2, 5.0, 7.0, 7.6, 10.0)]
    assert ratio == pytest.approx([1.50, 1.37, 1.57, 1.82, 1.90, 2.20], abs=0.01)


def test_the_law_is_held_at_its_1_5_m_s_value_below_that_with_a_warning():
    """Negative below 0.69 m/s; held at Dahl's lowest wind (Table I set 4) below 1.5 m/s."""
    at = acoustics.large_scale_mean_square_slope(1.5)
    assert at == pytest.approx(0.00714, abs=1e-5)
    for u in (0.0, 0.5, 1.0, 1.49):
        with pytest.warns(RuntimeWarning, match="1.5 m/s"):
            assert acoustics.large_scale_mean_square_slope(u) == at
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        acoustics.large_scale_mean_square_slope(1.5)
        assert acoustics.large_scale_mean_square_slope(0.3, warn=False) == at
        # No wind, no rough surface: time spreading does not warn about a slope it will not use.
        assert boundary_parameters({"wind_speed": 0.0}, surface=True) is None


def test_time_spreading_takes_the_large_scale_slope_from_the_wind_by_default():
    for u in (3.0, 6.0, 7.0, 9.14):
        par = boundary_parameters({"wind_speed": u}, surface=True)
        assert par["mean_square_slope"] == acoustics.large_scale_mean_square_slope(u)
        # The old law stays selectable, and changes nothing else.
        old = boundary_parameters({"wind_speed": u, "slope_law": "cox_munk_total"}, surface=True)
        assert old["mean_square_slope"] == acoustics.mean_square_slope(u)
        assert {k: v for k, v in old.items() if k != "mean_square_slope"} == {k: v for k, v in par.items() if k != "mean_square_slope"}
        assert boundary_parameters({"wind_speed": u, "slope_law": "large_scale"}, surface=True) == par
        # A slope given explicitly wins over either law.
        for law in ("large_scale", "cox_munk_total"):
            assert boundary_parameters({"wind_speed": u, "slope_law": law, "mean_square_slope": 0.01}, surface=True)["mean_square_slope"] == 0.01
    # CONFORMANCE.md A4's before -> after at the shipped scenarios' winds.
    assert boundary_parameters({"wind_speed": 6.0}, surface=True)["mean_square_slope"] == pytest.approx(0.0199, abs=5e-5)    # was 0.0337
    assert boundary_parameters({"wind_speed": 3.0}, surface=True)["mean_square_slope"] == pytest.approx(0.0135, abs=5e-5)    # was 0.0184
    with pytest.raises(ValueError, match="cox_munk_total"):
        boundary_parameters({"wind_speed": 6.0, "slope_law": "cox_munk"}, surface=True)
    with pytest.raises(ValueError, match="slope_law"):
        boundary_parameters({"rms_height": 0.01, "mean_square_slope": 0.01, "slope_law": "large_scale"}, surface=False)


def _across_kd_over_2pi(mss, sd, g_deg):
    """Our facet model's rms angle across the path at the receiver, sqrt(2 mss) sin g / (1 + RD/SD)
    (Dahl 1996 eq. (15) with q = sqrt 2; [NF] Sec. 3 item 2), as kd*/(2 pi) = 1 / (2 pi sigma)."""
    sigma = np.sqrt(2.0 * mss) * np.sin(np.radians(g_deg)) / (1.0 + RD / sd)
    return 1.0 / (2.0 * np.pi * sigma)


def test_with_the_default_slope_the_across_path_coherence_matches_table_i():
    """[NF] Sec. 3.2: with the wind's slope the closed form is 21-39 % short of Table I at five sets;
    with eq. (A7) it is within 20 % (set 4, 1.5 m/s, is Dahl's own outlier: s_LA is half of s_L)."""
    for name, (sd, rng, g, u, s_l, measured) in TABLE_I.items():
        new = _across_kd_over_2pi(boundary_parameters({"wind_speed": u}, surface=True)["mean_square_slope"], sd, g)
        old = _across_kd_over_2pi(boundary_parameters({"wind_speed": u, "slope_law": "cox_munk_total"}, surface=True)["mean_square_slope"], sd, g)
        assert abs(new / measured - 1.0) < abs(old / measured - 1.0)                 # closer at every set
        if name != "4":
            assert new == pytest.approx(measured, rel=0.20)
            assert old < 0.80 * measured


def _surface_tail(**surface):
    """One surface bounce of the flank scenarios: sonar 30 m, target 50 m, 1000 m, 8 kHz, 6 m/s."""
    inner = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=None, max_order=1)
    prov = wrap(inner, {"surface": dict({"wind_speed": 6.0}, **surface), "taps": 8, "seed": 3}, frequency=8000.0)
    e = prov.eigenrays(np.array([0.0, 0.0, 30.0]), np.array([[1000.0, 0.0, 50.0]]))
    base = inner.eigenrays(np.array([0.0, 0.0, 30.0]), np.array([[1000.0, 0.0, 50.0]]))
    k = base.delay.shape[1]
    specular = float(base.delay[0][base.n_surface[0] == 1][0])
    taps = e.valid[0, k:]
    w = np.abs(e.amplitude[0, k:][taps]) ** 2
    return float(np.sum(w * (e.delay[0, k:][taps] - specular)) / np.sum(w)), prov


def test_the_facet_tail_of_the_flank_scenarios_shortens_by_the_slope_ratio():
    """CONFORMANCE.md A4 before -> after at 6 m/s: facet mean delay 5.32 -> 3.14 ms (x0.590), the
    closed form mss (1 + sin^2 g) t1 t2 / (t1 + t2) at 4.57 degrees."""
    t1, t2 = (np.hypot(1000.0, 80.0) / 1500.0) * np.array([30.0, 50.0]) / 80.0
    g = np.arctan2(80.0, 1000.0)
    closed = lambda mss: mss * (1.0 + np.sin(g) ** 2) * t1 * t2 / (t1 + t2)        # noqa: E731
    new_mss = boundary_parameters({"wind_speed": 6.0}, surface=True)["mean_square_slope"]
    old_mss = boundary_parameters({"wind_speed": 6.0, "slope_law": "cox_munk_total"}, surface=True)["mean_square_slope"]
    assert closed(old_mss) * 1e3 == pytest.approx(5.32, abs=0.01) and closed(new_mss) * 1e3 == pytest.approx(3.14, abs=0.01)
    new, prov = _surface_tail()
    old, _ = _surface_tail(slope_law="cox_munk_total")
    assert prov.surface["mean_square_slope"] == new_mss                         # what `exclude_specular: true` reads too
    assert new / old == pytest.approx(new_mss / old_mss, rel=1e-6)                 # the whole tail scales with the slope
    assert new_mss / old_mss == pytest.approx(0.590, abs=1e-3)


@pytest.mark.xfail(strict=True, reason="Dahl 1999 (runs 2-4, about 4 m/s) measured horizontal angle spreads that imply a "
                   "mean-square slope of 0.008-0.011 ([NB] Sec. 3.3, our computation), half of eq. (A7)'s 0.0162. Recorded, "
                   "not followed (CONFORMANCE.md A4): Dahl 1999 puts the difference down to Bragg-scale waves a facet "
                   "model does not have")
def test_the_default_slope_gives_dahl_1999s_measured_angle_spreads():
    for mss_implied in (0.0081, 0.0106, 0.0103):                                  # [NB] Sec. 3.3, runs 2, 3, 4
        assert boundary_parameters({"wind_speed": 4.0}, surface=True)["mean_square_slope"] == pytest.approx(mss_implied, rel=0.25)
