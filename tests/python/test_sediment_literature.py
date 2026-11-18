"""The scattering parameters of the named sediments against the open documents they were read
from (docs/REFERENCES.md). Numbers typed here are the sources' own printed values, typed
independently of configs/sediments.yaml and of bottom.py where a check needs that.

  [S11] Snellen, Siemes and Simons, J. Acoust. Soc. Am. 129, 2878 (2011), Table II.
  [S13] Snellen et al., J. Acoust. Soc. Am. 134, 959 (2013), eqs. (3), (5).
  [C01] Cintron, MS thesis, Naval Postgraduate School (2001), Table 1.
  [B18] Brown, Johnson and Brownstead, arXiv:1809.08459, Table 1.
  [O16] Olson, Lyons and Saebo, arXiv:1601.06078.
  [F25] Fezzani, Berger, Le Bouffant and Lurton, Front. Remote Sens. (2025), doi:10.3389/frsen.2025.1574996.
"""

import math
import warnings

import numpy as np
import pytest

from gpusonar.sim import bottom

# [S11] Table II as printed: Mz, rho, nu, delta, sigma2, w2 (cm^4).
SNELLEN_2011_TABLE_II = (
    (-1.0, 2.492, 1.337, 0.0171, 0.002, 0.0129), (0.0, 2.314, 1.278, 0.0163, 0.002, 0.0086),
    (1.0, 2.151, 1.224, 0.0165, 0.002, 0.0056), (2.0, 1.615, 1.140, 0.0161, 0.002, 0.0035),
    (3.0, 1.339, 1.080, 0.0173, 0.002, 0.0021), (4.0, 1.223, 1.036, 0.0202, 0.002, 0.0011),
    (5.0, 1.169, 1.000, 0.0126, 0.002, 0.0005), (6.0, 1.149, 0.987, 0.0039, 0.001, 0.0005),
    (7.0, 1.147, 0.985, 0.0024, 0.001, 0.0005), (8.0, 1.146, 0.982, 0.0016, 0.001, 0.0005),
    (9.0, 1.145, 0.980, 0.0015, 0.001, 0.0005))

# [C01] Table 1: the name -> bulk grain size index of every fluid row of the table here.
CINTRON_NAMES = {
    "sandy_gravel": -1.0, "very_coarse_sand": -0.5, "muddy_sandy_gravel": 0.0, "coarse_sand": 0.5,
    "gravelly_muddy_sand": 1.0, "medium_sand": 1.5, "muddy_gravel": 2.0, "fine_sand": 2.5, "muddy_sand": 3.0,
    "very_fine_sand": 3.5, "clayey_sand": 4.0, "coarse_silt": 4.5, "sandy_silt": 5.0, "medium_silt": 5.5,
    "sandy_mud": 6.0, "fine_silt": 6.5, "sandy_clay": 7.0, "very_fine_silt": 7.5, "silty_clay": 8.0, "clay": 9.0}
FROM_ARXIV = ("medium_sand", "very_fine_silt")


def test_regression_reproduces_the_printed_table_to_its_digits():
    """[S13] eq. (5) against [S11] Table II: the same quantity from two papers."""
    for mz, _, _, _, sigma2, w2_cm4 in SNELLEN_2011_TABLE_II:
        assert round(bottom.grain_size_w2_cm4(mz), 4) == w2_cm4, mz
        assert bottom.grain_size_sigma2(mz) == sigma2, mz
        assert bottom.GRAIN_SIZE_W2_TABLE_CM4[mz] == w2_cm4
    # The printed example of the constant branch: h = 0.5 gives 0.00207 / 4.
    assert bottom.grain_size_w2_cm4(7.0) == pytest.approx(0.0005175, rel=1e-12)
    # The two branches meet at 5 phi (the rational form gives h = 0.5 there to the printed digits).
    assert bottom.grain_size_w2_cm4(5.0 - 1e-9) == pytest.approx(bottom.grain_size_w2_cm4(5.0), rel=1e-4)
    # Coarser is rougher, and the volume parameter steps at 5.5 phi.
    w = [bottom.grain_size_w2_cm4(x) for x in np.arange(-1.0, 9.01, 0.25)]
    assert all(a >= b for a, b in zip(w, w[1:]))
    assert bottom.grain_size_sigma2(5.49) == 0.002 and bottom.grain_size_sigma2(5.5) == 0.001
    for bad in (-1.01, 9.5):
        with pytest.raises(ValueError, match="outside the range"):
            bottom.grain_size_w2_cm4(bad)


def test_unit_conversion_against_a_source_that_prints_metres():
    """[B18] prints the spectral strength of two classes in m^(4 - gamma); [S13] gives cm^4 with a
    1 cm reference length. The regression, converted by w2_from_cm4, lands on [B18]'s numbers:
    an independent check of the conversion w2 [m] = w2 [cm^4] 10^(2 gamma - 8) and of the grain
    sizes 1.5 and 7.5 phi for medium sand and very fine silt."""
    for phi, printed in ((1.5, 1.410e-4), (7.5, 1.638e-5)):
        got = bottom.grain_size_scattering(phi)
        assert got["w2"] == pytest.approx(printed, rel=4e-3) and got["gamma2"] == 3.25
    # By hand for 1.5 phi: h = 1.634615 / 1.115385, w2 = 0.00207 h^2 cm^4, times 10^-1.5.
    h = (2.03846 - 0.26923 * 1.5) / (1.0 + 0.076923 * 1.5)
    assert bottom.grain_size_scattering(1.5)["w2"] == pytest.approx(0.00207 * h * h * 10.0**-1.5, rel=1e-12)
    # The wrong reading (a spectrum in cycles per cm, or no reference length) would be far off.
    assert abs(math.log10(bottom.grain_size_w2_cm4(1.5) * 1e-8 / 1.410e-4)) > 3.0


def test_every_row_has_its_grain_size_and_agrees_with_the_thesis_table():
    """rho, nu, delta of each row (read from UnderwaterAcoustics.jl or arXiv:1809.08459) against
    [C01] Table 1 as typed in bottom.GRAIN_SIZE_TABLE, at the grain size [C01] gives the name."""
    table = bottom.sediment_table()
    cintron = {row[0]: row[1:] for row in bottom.GRAIN_SIZE_TABLE}
    assert sorted(cintron) == sorted(CINTRON_NAMES.values())
    for name, phi in CINTRON_NAMES.items():
        entry = table[name]
        assert entry["mean_grain_size_phi"] == phi and "Cintron" in entry["provenance"]["mean_grain_size_phi"]
        par = bottom.sediment(name)
        rho, nu, delta = cintron[phi]
        assert par["rho"] == pytest.approx(rho, abs=1.1e-3), name          # very coarse sand: 2.4000 against 2.401
        assert par["nu"] == pytest.approx(nu, abs=5e-4 if name in FROM_ARXIV else 6e-5), name
        assert par["delta"] == pytest.approx(delta, rel=0.03 if name in FROM_ARXIV else 1e-9), name
        if "grain_size_phi" in entry:                                    # the GeoHab range, where the report has the class
            low, high = entry["grain_size_phi"]
            assert phi == (9.0 if high is None else 0.5 * (low + high)), name
    # [S11] Table II agrees too at whole phi (three or four printed digits).
    for mz, rho, nu, delta, _, _ in SNELLEN_2011_TABLE_II:
        c_rho, c_nu, c_delta = cintron[mz]
        assert c_rho == pytest.approx(rho, abs=1.1e-3) and c_nu == pytest.approx(nu, abs=6e-4)
        assert c_delta == pytest.approx(delta, abs=1.1e-4)
    # The speed ratio of sandy gravel: three prints of 1.337.
    assert cintron[-1.0][1] == 1.337 == SNELLEN_2011_TABLE_II[0][2]
    assert bottom.sediment("sandy_gravel")["nu"] == pytest.approx(1.337, abs=5e-6)


def test_table_scattering_parameters_are_the_published_relations_at_the_rows_grain_size():
    table = bottom.sediment_table()
    for name, phi in CINTRON_NAMES.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error")                 # complete, confirmed rows: no warning
            par = bottom.sediment(name)
        want = bottom.grain_size_scattering(phi)
        if name in FROM_ARXIV:
            assert par["w2"] == pytest.approx(want["w2"], rel=4e-3) and par["gamma2"] == 3.25
            continue
        assert par["w2"] == pytest.approx(want["w2"], rel=2e-5), name     # the file holds six digits
        assert par["gamma2"] == 3.25 and par["sigma2"] == want["sigma2"], name
        assert set(table[name]["provenance"]) == {"rho_nu_delta", "mean_grain_size_phi", "w2", "gamma2", "sigma2"}
        assert not bottom.is_reflection_only(name)
    # The one disagreement between sources, kept visible: medium sand's volume parameter.
    assert bottom.sediment("medium_sand")["sigma2"] == pytest.approx(0.001, rel=1e-9)
    assert bottom.grain_size_sigma2(1.5) == 0.002 and "0.002" in table["medium_sand"]["note"]
    assert bottom.sediment("very_fine_silt")["sigma2"] == pytest.approx(bottom.grain_size_sigma2(7.5), rel=0.02)


def test_volume_term_equals_the_published_expression():
    """[S13] eq. (3): sigma_pv = 5 delta sigma2 |1 - R^2|^2 sin^2(t) / (nu ln10 |P|^2 Im P),
    P = sqrt(kappa^2 - cos^2 t), kappa = (1 + i delta) / nu, R the reflection coefficient. Our
    volume term was derived independently (transmission in, attenuated scatterers, transmission
    out); it is the same function, which also fixes the meaning of sigma2 (cross section per
    unit volume over attenuation in dB/m)."""
    for name in ("coarse_sand", "medium_sand", "very_fine_sand", "sandy_silt", "clay"):
        par = bottom.sediment(name)
        rho, nu, delta, sigma2 = par["rho"], par["nu"], par["delta"], par["sigma2"]
        for deg in (5.0, 15.0, 30.0, 45.0, 60.0, 85.0):
            t = math.radians(deg)
            kappa = (1.0 + 1j * delta) / nu
            p = np.sqrt(kappa**2 - math.cos(t) ** 2 + 0j)
            p = -p if p.imag < 0 else p
            r = (rho * math.sin(t) - p) / (rho * math.sin(t) + p)
            published = 5.0 * delta * sigma2 * abs(1.0 - r * r) ** 2 * math.sin(t) ** 2 / (nu * math.log(10.0) * abs(p) ** 2 * p.imag)
            ours = float(bottom.volume_strength(t, t, 100e3, 1500.0, **par))
            assert ours == pytest.approx(published, rel=1e-10), (name, deg)


def test_sediment_from_grain_size_hits_the_classes_and_stays_between_them():
    for name, phi in CINTRON_NAMES.items():
        if name in FROM_ARXIV:
            continue
        got, row = bottom.sediment_from_grain_size(phi), bottom.sediment(name)
        assert got["rho"] == pytest.approx(row["rho"], abs=1.1e-3) and got["nu"] == pytest.approx(row["nu"], abs=6e-5)
        assert got["w2"] == pytest.approx(row["w2"], rel=2e-5) and got["sigma2"] == row["sigma2"]
    mid, lo, hi = (bottom.sediment_from_grain_size(x) for x in (2.2, 2.0, 2.5))
    for key in ("rho", "nu", "w2"):
        assert min(lo[key], hi[key]) < mid[key] < max(lo[key], hi[key])
    with pytest.raises(ValueError, match="outside the range"):
        bottom.sediment_from_grain_size(-3.0)


def test_family_trend_at_high_frequency():
    """With the published defaults the strength falls from sandy gravel to very fine sand at 100
    and 300 kHz (roughness scattering, coarser is rougher and harder), and the fine sediments
    sit on a plateau set by the volume term, which steps down at 5.5 phi with sigma2."""
    g = math.radians(20.0)
    sands = ("sandy_gravel", "very_coarse_sand", "muddy_sandy_gravel", "coarse_sand", "gravelly_muddy_sand",
             "medium_sand", "muddy_gravel", "fine_sand")
    for f in (100e3, 300e3):
        level = [float(bottom.named_backscattering_strength_db(n, g, f)) for n in sands]
        assert level == sorted(level, reverse=True) and level[0] - level[-1] > 8.0
        fine = [float(bottom.named_backscattering_strength_db(n, g, f)) for n in ("medium_silt", "fine_silt", "clay")]
        assert max(fine) - min(fine) < 0.5
        silt = float(bottom.named_backscattering_strength_db("sandy_silt", g, f))
        assert silt - fine[0] == pytest.approx(10 * math.log10(2.0), abs=1.0)     # the factor two in sigma2
    # A measured level for orientation, not a fit: medium sand at 45 degrees is "around -25 dB
    # for frequencies above 70 kHz" with a calibrated echosounder ([F25] Sec. 3.3). The default
    # parameters give -22.5 dB at 100 kHz and -19.7 dB at 300 kHz: 2.5 to 5.5 dB high, and rising
    # with frequency where the measurement is flat (recorded in docs/REFERENCES.md).
    for f, low, high in ((100e3, 1.5, 3.5), (300e3, 4.5, 6.0)):
        excess = float(bottom.named_backscattering_strength_db("medium_sand", math.radians(45.0), f)) + 25.0
        assert low < excess < high


def test_rock_and_pebbles_carry_a_lambert_law_inside_the_published_range():
    assert bottom.LAMBERT_MAX_MU_DB == pytest.approx(-4.97, abs=0.01)
    # Energy check of the bound: mu sin(ti) sin(ts) integrated over the hemisphere of scattered
    # directions equals the incident flux sin(ti) for mu = 1 / pi.
    ts = np.linspace(0.0, 0.5 * np.pi, 20001)
    assert np.trapezoid(np.sin(ts) * np.cos(ts), ts) * 2.0 * np.pi / np.pi == pytest.approx(1.0, abs=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rock = bottom.empirical_backscatter("rock")
    low, high = rock["mu_db_range"]
    assert rock["model"] == "lambert" and low <= rock["mu_db"] <= high < bottom.LAMBERT_MAX_MU_DB
    for cited in ("1601.06078", "10945/64176", "frsen.2025.1574996"):
        assert cited in rock["basis"]
    # The published values the range spans ([O16]: -13 and -18 dB; Gruber 2019: -23 dB; [F25]:
    # -13 dB at 45 degrees, i.e. mu = -13 - 20 log10(sin 45) = -10 dB).
    assert (low, high) == (-23.0, -10.0) and -13.0 - 20.0 * math.log10(math.sin(math.radians(45.0))) == pytest.approx(-10.0, abs=0.02)
    assert float(bottom.lambert_backscatter_db(math.radians(45.0), -10.0)) == pytest.approx(-13.0, abs=0.02)
    assert float(bottom.named_backscattering_strength_db("rock", math.radians(20.0), 100e3)) == pytest.approx(
        rock["mu_db"] + 20.0 * math.log10(math.sin(math.radians(20.0))))
    # [O16] quotes earlier rock measurements as -15 to -22 dB at 20 degrees: the law is inside
    # that band within 4 dB, and far below the handbook's -8 dB that [O16] shows to break the bound.
    at20 = float(bottom.named_backscattering_strength_db("rock", math.radians(20.0), 100e3))
    assert -27.0 < at20 < -15.0
    with pytest.warns(UserWarning, match=r"pebbles.*empirical_backscatter"):
        pebbles = bottom.empirical_backscatter("pebbles")
    assert pebbles["mu_db_range"] == [-10.0, -8.0] and pebbles["mu_db"] < bottom.LAMBERT_MAX_MU_DB
    with pytest.raises(ValueError, match="no empirical backscattering law"):
        bottom.empirical_backscatter("fine_sand")
    for name in ("rock", "pebbles"):
        assert bottom.is_reflection_only(name)


def test_composite_split_is_the_published_rayleigh_parameter_criterion():
    """Fonseca, Lurton, Fezzani and Roche (2025), eqs. (6)-(8): the cut-off wavenumber is where
    the Rayleigh parameter of the roughness above it is one, 4 k^2 zeta^2 = 1, with
    zeta^2 = 2 pi integral_Kc^inf W K dK, and the facet slope variance is
    2 pi integral_0^Kc W K^3 dK. The default `small_scale_kh` = 0.5 is that criterion, and
    `large_scale_slope` returns that slope variance per axis (half of it)."""
    for name, f in (("medium_sand", 100e3), ("coarse_sand", 30e3), ("fine_sand", 300e3)):
        par = bottom.sediment(name)
        k = 2.0 * np.pi * f / 1500.0
        slope, kc = bottom.large_scale_slope(f, 1500.0, **par)
        big_k = np.geomspace(kc, kc * 1e7, 400001)
        zeta2 = 2.0 * np.pi * np.trapezoid(par["w2"] * big_k ** (1.0 - par["gamma2"]), big_k)
        assert 4.0 * k * k * zeta2 == pytest.approx(1.0, rel=2e-3), name
        small_k = np.geomspace(kc * 1e-12, kc, 400001)        # the integrand goes as K^(3 - gamma2): integrable at 0
        slope2_total = 2.0 * np.pi * np.trapezoid(par["w2"] * small_k ** (3.0 - par["gamma2"]), small_k)
        assert 2.0 * slope**2 == pytest.approx(slope2_total, rel=2e-3), name
        # Olson and Lyons (arXiv:2106.05909, Sec. VIII) require k x small-scale rms height below one.
        assert k * math.sqrt(zeta2) == pytest.approx(0.5, rel=1e-3) and k * math.sqrt(zeta2) < 1.0
