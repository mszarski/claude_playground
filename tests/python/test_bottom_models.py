"""Bottom roughness scattering beyond first-order perturbation theory (sim/bottom.py): the
Kirchhoff and small-slope approximations, composite roughness, and the sediment table."""
import math
import warnings

import numpy as np
import pytest
from scipy.special import j0

from gpusonar.sim import bottom

C, F = 1500.0, 100e3
K = 2 * np.pi * F / C
SAND = bottom.PLACEHOLDER_SAND
SMOOTH = dict(SAND, w2=1e-12)                 # relief so small that every approximation is in its small-roughness limit
MODELS = ("perturbation", "kirchhoff", "small_slope", "composite", "combined")


def rough(model, ti, ts, phi=np.pi, par=SAND, f=F, **kw):
    return bottom.roughness_strength(ti, ts, f, C, azimuth=phi, roughness_model=model, **dict(par, **kw))


def old_perturbation(ti, ts, phi, rho, nu, delta, w2, gamma2, **_):
    """The roughness term exactly as the module computed it before the other models were added."""
    k2sq = (K * (1.0 + 1j * delta) / nu) ** 2
    ki, ks = K * np.cos(ti), K * np.cos(ts)
    b1i, b1s = K * np.sin(ti), K * np.sin(ts)
    b2i, b2s = bottom._vertical(k2sq, ki), bottom._vertical(k2sq, ks)
    v = (rho * b1i - b2i) / (rho * b1i + b2i)
    t, g = 1.0 + v, b1i * (1.0 - v)
    q_dot_ki = ki * ks * np.cos(phi) - ki**2
    q = np.sqrt(np.maximum(ki**2 + ks**2 - 2.0 * ki * ks * np.cos(phi), 0.0))
    rhs1 = 1j * g * (1.0 - rho)
    rhs2 = t * (b1i**2 - b2i**2 / rho - (1.0 - 1.0 / rho) * q_dot_ki)
    a = (rhs2 + (1j * b2s / rho) * rhs1) / (1j * b1s + 1j * b2s / rho)
    return b1s**2 * np.abs(a) ** 2 * (w2 / q**gamma2)


def test_default_is_the_perturbation_model_unchanged():
    ti, ts, phi = np.radians([5.0, 20.0, 35.0, 60.0]), np.radians([15.0, 20.0, 50.0, 10.0]), np.radians([180.0, 180.0, 40.0, 110.0])
    want = old_perturbation(ti, ts, phi, **SAND)
    assert np.array_equal(bottom.roughness_strength(ti, ts, F, C, azimuth=phi, **SAND), want)
    assert np.array_equal(rough("perturbation", ti, ts, phi), want)
    assert np.array_equal(bottom.strength(ti, ts, F, C, azimuth=phi, **SAND), want + bottom.volume_strength(ti, ts, F, C, **SAND))
    with pytest.raises(ValueError, match="roughness_model"):
        rough("lambert", 0.3, 0.3)


def test_perturbation_factor_is_the_published_fluid_sediment_form():
    """Jackson's form as printed in Olson and Jackson, arXiv:2009.12420, eqs. (13)-(16) and (20),
    for water over a half-space: sigma = k^4 |A|^2 W, A = (a (1 + Vi)(1 + Vs) - b (1 - Vi)(1 - Vs)) / 2,
    a = (1 - 1/rho) Ks.Ki / k^2 - 1 + 1 / (ac^2 rho), b = (rho - 1) sin(ti) sin(ts), ac the complex speed ratio."""
    rho, nu, delta = SAND["rho"], SAND["nu"], SAND["delta"]
    ti, ts, phi = np.radians([8.0, 20.0, 33.0, 70.0]), np.radians([8.0, 45.0, 12.0, 70.0]), np.radians([180.0, 60.0, 130.0, 10.0])
    vi, vs = bottom.reflection(ti, rho, nu, delta), bottom.reflection(ts, rho, nu, delta)
    ac = nu / (1 + 1j * delta)
    a = (1 - 1 / rho) * np.cos(ti) * np.cos(ts) * np.cos(phi) - 1 + 1 / (ac**2 * rho)
    b = (rho - 1) * np.sin(ti) * np.sin(ts)
    amp = 0.5 * (a * (1 + vi) * (1 + vs) - b * (1 - vi) * (1 - vs))
    factor, _ = bottom._perturbation_factor(ti, ts, phi, K, rho, nu, delta)
    assert np.allclose(factor, K**4 * np.abs(amp) ** 2, rtol=1e-10)


def test_structure_constant_is_the_integral_it_stands_for():
    """D(R) = 2 integral W (1 - cos K.R) d2K = 4 pi w2 R^(g - 2) integral x^(1 - g) (1 - J0(x)) dx."""
    x = np.geomspace(1e-8, 1e6, 2_000_001)
    for g2 in (2.6, 3.0, 3.25):
        ch2, a = bottom.structure_constant(1.0, g2)
        tail = 4 * np.pi * x[-1] ** (2 - g2) / (g2 - 2)               # the part beyond the grid, where J0 averages out
        assert a == pytest.approx(g2 / 2 - 1)
        assert ch2 == pytest.approx(4 * np.pi * np.trapezoid(x ** (1 - g2) * (1 - j0(x)), x) + tail, rel=2e-4)
    with pytest.raises(ValueError):
        bottom.structure_constant(1.0, 4.2)


def test_kirchhoff_integral_against_direct_quadrature_and_its_limits():
    for a in (0.5, 0.625, 0.9):
        for b in (0.05, 1.0, 30.0):
            x = np.linspace(0.0, (45 / b) ** (1 / (2 * a)), 2_000_001)
            direct = np.trapezoid(j0(x) * x * np.exp(-b * x ** (2 * a)), x)
            assert float(bottom.kirchhoff_integral(b, a)) == pytest.approx(direct, rel=1e-3)
        # a = 1/2 has a closed form: the Hankel transform of exp(-b x) is b / (b^2 + 1)^(3/2).
        small = 1e-4
        assert float(bottom.kirchhoff_integral(small, a)) == pytest.approx(
            small * 2 ** (2 * a + 1) * a * math.gamma(1 + a) / math.gamma(1 - a), rel=2e-3)
        big = 0.5 * bottom._large_b(a)
        assert float(bottom.kirchhoff_integral(big, a)) == pytest.approx(math.gamma(1 / a) / (2 * a) * big ** (-1 / a), rel=2e-3)
    b = np.geomspace(1e-9, 1e6, 200)
    assert np.allclose(bottom.kirchhoff_integral(b, 0.5), b / (b**2 + 1) ** 1.5, rtol=2e-3)


def test_kirchhoff_and_perturbation_agree_where_both_are_valid():
    """Small relief, specular direction: both give |V|^2 q^4 W(Q) / 4."""
    for theta in np.radians([15.0, 40.0, 70.0]):
        for phi in np.radians([0.5, 2.0]):                       # just off specular, so that Q > 0
            kir, per = rough("kirchhoff", theta, theta, phi, SMOOTH), rough("perturbation", theta, theta, phi, SMOOTH)
            assert kir == pytest.approx(per, rel=3e-3 if phi < 0.02 else 3e-2)
            q_h = 2 * K * np.cos(theta) * np.sin(phi / 2)
            q2 = q_h**2 + (2 * K * np.sin(theta)) ** 2
            v = bottom.reflection(np.arcsin(np.sqrt(q2) / (2 * K)), SAND["rho"], SAND["nu"], SAND["delta"])
            assert kir == pytest.approx(abs(v) ** 2 * q2**2 / 4 * SMOOTH["w2"] / q_h ** SMOOTH["gamma2"], rel=1e-3)
    # Away from the specular direction they differ even for small relief: Kirchhoff is not valid there.
    g = np.radians(2.0)
    assert rough("kirchhoff", g, g, np.pi, SMOOTH) > 10.0 * rough("perturbation", g, g, np.pi, SMOOTH)


def test_small_slope_joins_the_two():
    ti, ts, phi = np.radians([5.0, 20.0, 35.0, 60.0, 80.0]), np.radians([15.0, 20.0, 50.0, 10.0, 70.0]), np.radians([180.0, 180.0, 40.0, 110.0, 30.0])
    assert np.allclose(rough("small_slope", ti, ts, phi, SMOOTH), rough("perturbation", ti, ts, phi, SMOOTH), rtol=1e-3)
    # Rough bottom, exactly specular: equal to Kirchhoff, and finite where the power-law perturbation result is not.
    theta = np.radians([10.0, 30.0, 60.0, 90.0])
    ss, kir = rough("small_slope", theta, theta, 0.0), rough("kirchhoff", theta, theta, 0.0)
    assert np.all(np.isfinite(ss)) and np.allclose(ss, kir, rtol=1e-6)
    ch2, a = bottom.structure_constant(SAND["w2"], SAND["gamma2"])
    v = bottom.reflection(theta, SAND["rho"], SAND["nu"], SAND["delta"])
    qz = 2 * K * np.sin(theta)
    assert np.allclose(kir, np.abs(v) ** 2 / (8 * np.pi) * qz**2 * math.gamma(1 / a) / (2 * a) * (0.5 * qz**2 * ch2) ** (-1 / a), rtol=1e-9)
    # Backscatter at low and middle angles on the rough bottom: still perturbation theory.
    g = np.radians([5.0, 15.0, 30.0])
    assert np.allclose(rough("small_slope", g, g), rough("perturbation", g, g), rtol=0.03)


def test_composite_roughness():
    slope, kc = bottom.large_scale_slope(F, C, **SAND)
    g2, w2 = SAND["gamma2"], SAND["w2"]
    kk = np.geomspace(kc, kc * 1e8, 400_001)                      # small-scale height: integral of W over K > Kc
    assert K**2 * np.trapezoid(w2 * kk**-g2 * 2 * np.pi * kk, kk) == pytest.approx(0.25, rel=1e-4)       # (k h_s)^2 = 0.5^2
    kk = np.linspace(0.0, kc, 400_001)[1:]                        # large-scale slope along one axis: half the total
    assert np.trapezoid(0.5 * kk**2 * w2 * kk**-g2 * 2 * np.pi * kk, kk) == pytest.approx(slope**2, rel=1e-3)
    # No large-scale slope (the split pushed to very long scales): perturbation theory again.
    g = np.radians([3.0, 10.0, 30.0])
    assert np.allclose(rough("composite", g, g, small_scale_kh=200.0), rough("perturbation", g, g), rtol=1e-3)
    # The correction raises the level at small grazing angles and leaves the middle angles alone.
    low, mid = np.radians(2.0), np.radians(25.0)
    assert rough("composite", low, low) > 3.0 * rough("perturbation", low, low)
    assert rough("composite", mid, mid) == pytest.approx(rough("perturbation", mid, mid), rel=0.05)
    # It is the plain average of the perturbation result over the in-plane slope (backscatter).
    x, w = np.polynomial.hermite_e.hermegauss(40)
    local = low + np.arctan(slope * x)
    want = (w / w.sum() * np.where(local > 0, old_perturbation(np.clip(local, 1e-6, None), np.clip(local, 1e-6, None), np.pi, **SAND), 0.0)).sum()
    assert rough("composite", low, low) == pytest.approx(want, rel=2e-2)


def test_combined_model_covers_all_three_regimes():
    low, mid = np.radians(2.0), np.radians([15.0, 30.0, 45.0])
    assert rough("combined", low, low) == pytest.approx(rough("composite", low, low), rel=0.02)
    assert np.allclose(rough("combined", mid, mid), rough("perturbation", mid, mid), rtol=0.05)
    steep = np.radians([80.0, 90.0])
    assert np.allclose(10 * np.log10(rough("combined", steep, steep) / rough("kirchhoff", steep, steep)), 0.0, atol=2.5)
    assert np.all(np.isfinite(rough("combined", steep, steep)))


@pytest.mark.parametrize("model", MODELS)
def test_reciprocity(model):
    for ti, ts, phi in ((15.0, 40.0, 120.0), (5.0, 70.0, 20.0), (30.0, 30.0, 180.0), (50.0, 65.0, 5.0)):
        ti, ts, phi = np.radians([ti, ts, phi])
        a, b = rough(model, ti, ts, phi), rough(model, ts, ti, phi)
        # The slope average is a 12 x 12 point quadrature, which is not exactly invariant under
        # the rotation of the slope axes that the exchange amounts to.
        assert a == pytest.approx(b, rel=1e-4 if model in ("composite", "combined") else 1e-9) and a >= 0.0
    a = bottom.strength(np.radians(15.0), np.radians(40.0), F, C, azimuth=2.0, roughness_model=model, **SAND)
    b = bottom.strength(np.radians(40.0), np.radians(15.0), F, C, azimuth=2.0, roughness_model=model, **SAND)
    assert a == pytest.approx(b, rel=1e-4)


def scattered_fraction(model, ti, n):
    """Scattered power over the upper hemisphere per unit incident power on the bottom."""
    ts = (np.arange(n[0]) + 0.5) / n[0] * (np.pi / 2)
    ph = (np.arange(n[1]) + 0.5) / n[1] * np.pi                    # the pattern is symmetric in azimuth
    t, p = np.meshgrid(ts, ph, indexing="ij")
    if model == "combined":                                         # fewer slope nodes: 36 small-slope evaluations per direction
        s = bottom.combined_strength(ti, t, F, C, azimuth=p, slope_nodes=6, **SAND)
    else:
        s = rough(model, ti, t, p)
    return 2 * (s * np.cos(t)).sum() * (np.pi / 2 / n[0]) * (np.pi / n[1]) / np.sin(ti)


@pytest.mark.parametrize("model,grid", [("kirchhoff", (600, 360)), ("small_slope", (600, 360)), ("combined", (300, 180))])
def test_scattered_energy_stays_within_what_the_flat_bottom_reflects(model, grid):
    """A power-law relief has no coherent reflection left, so all of the reflected energy
    |V|^2 is in the scattered field, and no more than that."""
    for deg in (20.0, 40.0, 70.0):
        ti = np.radians(deg)
        budget = abs(bottom.reflection(ti, SAND["rho"], SAND["nu"], SAND["delta"])) ** 2
        assert 0.75 * budget < scattered_fraction(model, ti, grid) < 1.06 * budget


def test_sediment_table(tmp_path):
    with pytest.warns(UserWarning, match="unconfirmed"):
        assert bottom.sediment("placeholder_sand") == SAND
    with pytest.raises(ValueError, match="unknown sediment"):
        bottom.sediment("no_such_sediment")
    mine = tmp_path / "sediments.yaml"
    mine.write_text("site: {rho: 1.8, nu: 1.12, delta: 0.01, w2: 1.0e-5, gamma2: 3.0, sigma2: 0.001, status: confirmed, source: test}\n"
                    "half: {rho: 1.8, nu: 1.12}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bottom.sediment("site", mine)["nu"] == 1.12
    with pytest.raises(ValueError, match="lacks"):
        bottom.sediment("half", mine)


def test_published_sediment_rows_and_their_unit_conversions():
    """arXiv:1809.08459 Table 1 as printed (density kg/m3, sound speed m/s, attenuation dB/m at
    20 kHz, spectral strength m^(4 - gamma), exponent, volume scattering strength dB/m),
    converted to the model's six parameters by formulas written out again here."""
    printed = {"medium_sand": (1845.0, 1767.0, 10.0, 1.410e-4, 3.25, -20.0),
               "very_fine_silt": (1147.0, 1476.0, 1.4, 1.638e-5, 3.25, -28.6)}
    for name, (density, speed, att, w2, g2, sv) in printed.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error")                              # confirmed rows do not warn
            par = bottom.sediment(name)
        assert par["rho"] == pytest.approx(density / 1000.0) and par["nu"] == pytest.approx(speed / 1500.0)
        assert par["w2"] == w2 and par["gamma2"] == g2
        # The loss parameter gives back the printed attenuation at 20 kHz: the amplitude decays as
        # exp(-Im(k2) x), Im(k2) = k delta / nu, and 20 log10(e) dB per neper.
        k = 2 * np.pi * 20e3 / 1500.0
        assert 20 * np.log10(np.e) * k * par["delta"] / par["nu"] == pytest.approx(att, rel=1e-12)
        # The volume term's cross section per unit volume, as the model forms it at 20 kHz, is the printed level.
        sigma_vol = par["sigma2"] * (20 / np.log(10)) * k * par["delta"] / par["nu"]
        assert 10 * np.log10(sigma_vol) == pytest.approx(sv, abs=1e-9)
        bs = bottom.backscattering_strength_db(np.radians([10.0, 30.0]), 20e3, C, **par)
        assert np.all((bs > -50.0) & (bs < -15.0))
    sand, silt = bottom.sediment("medium_sand"), bottom.sediment("very_fine_silt")
    assert sand["nu"] > 1.0 > silt["nu"] and sand["delta"] > silt["delta"]
    g = np.radians(20.0)
    assert bottom.roughness_strength(g, g, F, C, **sand) > 100 * bottom.roughness_strength(g, g, F, C, **silt)   # sand: roughness
    assert bottom.volume_strength(g, g, F, C, **silt) > 100 * bottom.roughness_strength(g, g, F, C, **silt)      # silt: volume


def test_spectral_strength_from_the_centimetre_form():
    """W = beta (h K)^-gamma, h = 1 cm, beta in cm^4 (arXiv:physics/0607146 eq. (7)) against
    W = w2 K^-gamma in metres: evaluate both at one wavenumber."""
    beta, g2, k_per_m = 3.0e-3, 3.25, 700.0
    w_cm4 = beta * (1.0 * k_per_m / 100.0) ** -g2                        # h K with K in 1/cm
    assert bottom.w2_from_cm4(beta, g2) * k_per_m**-g2 == pytest.approx(w_cm4 * 1e-8, rel=1e-12)
    assert bottom.w2_from_cm4(1.0, 4.0) == pytest.approx(1.0)            # gamma = 4: beta is dimensionless in both


def test_scenario_item_takes_a_named_sediment_and_a_roughness_model():
    from test_reverberation import HEIGHT, empty_baseband_ping
    from gpusonar.sim import Scene, make_pose, simulate_ping
    sonar, _ = empty_baseband_ping()

    def ping(item):
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                            reverberation=[dict({"model": "fluid_sediment", "boundary_z": 0.0}, **item)], rng=np.random.default_rng(2))
        return rec.samples.array.copy()
    explicit = ping(SAND)
    with pytest.warns(UserWarning, match="unconfirmed"):
        named = ping({"sediment": "placeholder_sand"})
    assert np.array_equal(named, explicit)                                        # the table gives the same ping
    assert np.array_equal(ping(dict(SAND, roughness_model="perturbation")), explicit)
    with pytest.warns(UserWarning, match="unconfirmed"):
        louder = ping({"sediment": "placeholder_sand", "w2": 4 * SAND["w2"], "sigma2": 0.0})   # the item's own values win
    assert (np.abs(louder) ** 2).mean() / (np.abs(ping(dict(SAND, sigma2=0.0))) ** 2).mean() == pytest.approx(4.0, rel=1e-6)
    combined = ping(dict(SAND, roughness_model="combined"))
    ratio = (np.abs(combined) ** 2).mean() / (np.abs(explicit) ** 2).mean()
    assert 0.8 < ratio < 2.0 and not np.array_equal(combined, explicit)           # same order, not the same model
