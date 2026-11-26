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


def test_the_perturbation_model_is_unchanged_and_the_default_is_small_slope():
    """`perturbation` was the default until D-86 and still gives the earlier numbers exactly when
    named; with no model named the result is the small-slope one."""
    ti, ts, phi = np.radians([5.0, 20.0, 35.0, 60.0]), np.radians([15.0, 20.0, 50.0, 10.0]), np.radians([180.0, 180.0, 40.0, 110.0])
    want = old_perturbation(ti, ts, phi, **SAND)
    assert np.array_equal(bottom.roughness_strength(ti, ts, F, C, azimuth=phi, roughness_model="perturbation", **SAND), want)
    assert np.array_equal(rough("perturbation", ti, ts, phi), want)
    assert np.array_equal(bottom.strength(ti, ts, F, C, azimuth=phi, roughness_model="perturbation", **SAND),
                          want + bottom.volume_strength(ti, ts, F, C, **SAND))
    assert bottom.DEFAULT_ROUGHNESS_MODEL == "small_slope"
    default = bottom.roughness_strength(ti, ts, F, C, azimuth=phi, **SAND)
    assert np.array_equal(default, rough("small_slope", ti, ts, phi))
    assert np.array_equal(bottom.strength(ti, ts, F, C, azimuth=phi, **SAND), default + bottom.volume_strength(ti, ts, F, C, **SAND))
    assert np.allclose(default, want, rtol=0.1) and not np.array_equal(default, want)       # these angles: within 6 % (most towards forward scatter)


def test_the_default_is_finite_and_kirchhoff_at_normal_incidence_where_perturbation_diverges():
    """The reason for D-86, as numbers: backscatter at normal incidence (the specular direction)."""
    g = np.array([0.5 * np.pi])
    default = bottom.roughness_strength(g, g, F, C, **SAND)
    assert np.isfinite(default).all() and np.allclose(default, rough("kirchhoff", g, g), rtol=1e-9)
    assert rough("perturbation", g, g)[0] > 1e20 * default[0]


def test_outside_the_structure_function_range_the_default_falls_back_to_perturbation_and_says_so():
    g = np.radians([10.0, 30.0])
    par = dict(SAND, gamma2=4.5)
    bottom._warned_no_small_slope = False
    with pytest.warns(UserWarning, match="perturbation theory is used instead"):
        got = bottom.roughness_strength(g, g, F, C, **par)
    assert np.array_equal(got, bottom.roughness_strength(g, g, F, C, roughness_model="perturbation", **par))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # once
        bottom.roughness_strength(g, g, F, C, **par)
    with pytest.raises(ValueError, match="2 < gamma2 < 4"):
        bottom.roughness_strength(g, g, F, C, roughness_model="small_slope", **par)
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
    # It is the plain average of the perturbation result over the in-plane slope (backscatter), times
    # Wagner's backscatter shadow function R since C1 of docs/references/bottom_mourad_jackson_1989.md
    # (before C1 the first comparison was made with the default; `shadowing="none"` is that model).
    x, w = np.polynomial.hermite_e.hermegauss(40)
    local = low + np.arctan(slope * x)
    want = (w / w.sum() * np.where(local > 0, old_perturbation(np.clip(local, 1e-6, None), np.clip(local, 1e-6, None), np.pi, **SAND), 0.0)).sum()
    assert rough("composite", low, low, shadowing="none") == pytest.approx(want, rel=2e-2)
    shade = float(bottom.wagner_backscatter_shadowing(low, math.sqrt(2.0) * slope))
    assert rough("composite", low, low) == pytest.approx(want * shade, rel=2e-2) and shade < 0.98


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
    assert np.array_equal(ping(dict(SAND, roughness_model="small_slope")), explicit)          # the default (D-86)
    old = ping(dict(SAND, roughness_model="perturbation"))                        # the default before it, still there by name
    assert not np.array_equal(old, explicit) and (np.abs(old) ** 2).mean() == pytest.approx((np.abs(explicit) ** 2).mean(), rel=0.02)
    with pytest.warns(UserWarning, match="unconfirmed"):
        louder = ping({"sediment": "placeholder_sand", "w2": 4 * SAND["w2"], "sigma2": 0.0,
                       "roughness_model": "perturbation"})                        # the item's own values win; perturbation is linear in w2
    assert ((np.abs(louder) ** 2).mean() / (np.abs(ping(dict(SAND, sigma2=0.0, roughness_model="perturbation"))) ** 2).mean()
            == pytest.approx(4.0, rel=1e-6))
    combined = ping(dict(SAND, roughness_model="combined"))
    ratio = (np.abs(combined) ** 2).mean() / (np.abs(explicit) ** 2).mean()
    assert 0.8 < ratio < 2.0 and not np.array_equal(combined, explicit)           # same order, not the same model


# --- Wagner (1967) shadowing in the composite models -----------------------------------------------
# docs/references/shadowing_wagner_1967.md [W67] C-W1, C-W2 and docs/references/bottom_mourad_jackson_1989.md
# [MJ89] C1. The function is Wagner's backscatter shadow function R (his eq. (49) with (20)-(22)):
# R = (1 - exp(-2B)) / (2B), B = [pi^(-1/2) exp(-v^2) - v erfc v] / (4v), v = tan(grazing) / s, s the
# TOTAL (both-axes) rms slope = sqrt(2) x the per-axis slope `large_scale_slope` returns. pi^(-1/2) is
# printed in [W67] (22) and [MJ89] (25) (confirmed); Jackson and Richardson (2007) (13.82) print pi^(+1/2),
# a misprint.


def wagner_r(v):
    """R typed again here from [W67] (22) and (49) with (20), independent of bottom.py."""
    b = (math.exp(-v * v) - math.sqrt(math.pi) * v * math.erfc(v)) / (4.0 * math.sqrt(math.pi) * v)
    return (1.0 - math.exp(-2.0 * b)) / (2.0 * b)


def test_wagner_backscatter_shadowing_values_and_limits():
    """The note's values (ours, [W67] Sec. 2.3), recomputed from the retyped formula: R = 0.385059,
    0.719744, 0.906503, 0.987541, 0.998564, 0.999878 at v = 0.1, 0.25, 0.5, 1, 1.5, 2."""
    v = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    want = np.array([0.385059, 0.719744, 0.906503, 0.987541, 0.998564, 0.999878])
    assert np.allclose([wagner_r(x) for x in v], want, atol=1e-6)                       # the retyped formula gives the note's values
    got = bottom.wagner_backscatter_shadowing(np.arctan(v * 0.3), 0.3)                     # v = tan(grazing) / s
    assert np.allclose(got, want, atol=1e-6)
    assert np.allclose(got, [wagner_r(x) for x in v], rtol=1e-12)
    # Limits ([W67] Sec. 2.3, ours): R -> 2 sqrt(pi) v as v -> 0 (2 % at v = 0.01); R = 1 at normal incidence; 0 at grazing.
    small = float(bottom.wagner_backscatter_shadowing(np.arctan(0.01), 1.0))
    assert small / (2.0 * math.sqrt(math.pi) * 0.01) == pytest.approx(1.0, abs=0.02)
    assert float(bottom.wagner_backscatter_shadowing(0.5 * np.pi, 0.4)) == 1.0               # no shadowing at normal incidence
    assert float(bottom.wagner_backscatter_shadowing(0.0, 0.4)) == 0.0
    g = np.radians(np.linspace(0.1, 90.0, 900))
    r = bottom.wagner_backscatter_shadowing(g, 0.25)
    assert np.all(np.diff(r) >= 0.0) and np.all((r > 0.0) & (r <= 1.0))                     # monotonic in grazing
    # Large v: no cancellation (erfcx), and the [MJ89] rule "S = 1 for t > 2" (cutoff=2) is within 1.3e-4.
    big = bottom.wagner_backscatter_shadowing(np.arctan(np.array([2.5, 5.0, 30.0, 1e6])), 1.0)
    assert np.all(big <= 1.0) and np.all(1.0 - big < 1.3e-4) and big[-1] == 1.0
    cut = bottom.wagner_backscatter_shadowing(np.arctan([1.9, 2.1]), 1.0, cutoff=2.0)
    assert cut[1] == 1.0 and cut[0] == pytest.approx(wagner_r(1.9), rel=1e-12)


def test_jackson_richardson_shadowing_table_with_the_corrected_sign():
    """J&R's own table of 10 log R (our note 04, with pi^(-1/2)): -0.44 / -1.45 / -3.37 / -5.59 dB at 2 deg for
    per-axis slopes 0.05 / 0.1 / 0.2 / 0.35, and -0.26 dB at 5 deg for 0.1 (+-0.01 dB)."""
    for slope, want in ((0.05, -0.44), (0.1, -1.45), (0.2, -3.37), (0.35, -5.59)):
        got = 10 * np.log10(bottom.wagner_backscatter_shadowing(np.radians(2.0), math.sqrt(2.0) * slope))
        assert got == pytest.approx(want, abs=0.01), slope
    assert 10 * np.log10(bottom.wagner_backscatter_shadowing(np.radians(5.0), math.sqrt(2.0) * 0.1)) == pytest.approx(-0.26, abs=0.01)


def old_composite(ti, ts, phi, par=SAND, f=F, small_scale_kh=0.5, slope_nodes=12):
    """`composite_strength` exactly as it computed before shadowing was added."""
    k = 2.0 * np.pi * f / C
    slope, kc = bottom.large_scale_slope(f, C, w2=par["w2"], gamma2=par["gamma2"], small_scale_kh=small_scale_kh)
    ti, ts, phi = bottom._angles(ti, ts, phi)

    def small_scale(li, ls, ph):
        factor, q = bottom._perturbation_factor(li, ls, ph, k, par["rho"], par["nu"], par["delta"])
        return factor * np.where(q >= kc, bottom._power_law(q, par["w2"], par["gamma2"]), 0.0)
    return bottom._slope_average(small_scale, ti, ts, phi, slope, slope_nodes)


def test_shadowing_none_is_the_composite_and_combined_before_bit_for_bit():
    ti, ts, phi = np.radians([2.0, 5.0, 20.0, 60.0, 85.0]), np.radians([2.0, 15.0, 3.0, 60.0, 40.0]), np.radians([180.0, 60.0, 120.0, 0.0, 180.0])
    assert np.array_equal(rough("composite", ti, ts, phi, shadowing="none"), old_composite(ti, ts, phi))
    slope, _ = bottom.large_scale_slope(F, C, **SAND)
    par = {k: SAND[k] for k in ("rho", "nu", "delta", "w2", "gamma2")}
    before = bottom._slope_average(lambda li, ls, ph: bottom.small_slope_strength(li, ls, F, C, azimuth=ph, **par),
                                   *bottom._angles(ti, ts, phi), slope, 12)
    assert np.array_equal(rough("combined", ti, ts, phi, shadowing="none"), before)
    # With Wagner's factor (the default inside these two models) the result is the same average times it.
    shade = bottom.wagner_bistatic_shadowing(ti, ts, phi, math.sqrt(2.0) * slope)
    assert np.allclose(rough("composite", ti, ts, phi), old_composite(ti, ts, phi) * shade, rtol=1e-14)
    assert np.allclose(rough("combined", ti, ts, phi), before * shade, rtol=1e-14)
    # The other models carry no shadowing: "none" is accepted, "wagner" refused; unknown values name the nearest.
    for model in ("perturbation", "kirchhoff", "small_slope"):
        assert np.array_equal(rough(model, ti, ts, phi, shadowing="none"), rough(model, ti, ts, phi))
        with pytest.raises(ValueError, match="shadowing"):
            rough(model, ti, ts, phi, shadowing="wagner")
    with pytest.raises(ValueError, match="did you mean 'wagner'"):
        rough("composite", ti, ts, phi, shadowing="wagnr")


def test_bistatic_shadowing_is_symmetric_and_reduces_to_r_in_backscatter():
    """[W67] C-W2 (ours, derived from [W67] (33), (35)-(36), (46)-(47); out of plane not printed): backward half
    (cos phi < 0) R(min(ti, ts)); forward half R_b = (1 - exp[-2(B_i + B_s)]) / [2(B_i + B_s)]."""
    s = 0.3
    g = np.radians([1.0, 3.0, 7.0, 15.0, 40.0, 90.0])
    assert np.array_equal(bottom.wagner_bistatic_shadowing(g, g, np.pi, s), bottom.wagner_backscatter_shadowing(g, s))   # exactly R
    ti, ts = np.meshgrid(g, g, indexing="ij")
    for phi in np.radians([0.0, 45.0, 89.0, 91.0, 150.0, 180.0]):
        a, b = bottom.wagner_bistatic_shadowing(ti, ts, phi, s), bottom.wagner_bistatic_shadowing(ts, ti, phi, s)
        assert np.array_equal(a, b)                                                       # reciprocity
        r_min = np.minimum(bottom.wagner_backscatter_shadowing(ti, s), bottom.wagner_backscatter_shadowing(ts, s))
        if np.cos(phi) < 0.0:
            assert np.array_equal(a, bottom.wagner_backscatter_shadowing(np.minimum(ti, ts), s))
        else:
            assert np.all(a <= r_min + 1e-15)                                             # both rays must see the facet
    assert float(bottom.wagner_bistatic_shadowing(0.5 * np.pi, 0.5 * np.pi, 0.0, s)) == 1.0


def test_composite_shadowing_moves_the_opt_in_models_as_the_note_computes():
    """[MJ89] C1 "Moves" (computed by the extraction with c = 1500 m/s; backscatter roughness term, dB):
    medium sand 30 kHz 5 deg -41.58 -> -41.69; 100 kHz 5 deg -32.88 -> -33.56, 10 deg -29.70 -> -29.82;
    sandy gravel 30 kHz 5 deg -29.03 -> -29.89; 100 kHz 5 / 10 / 20 deg -21.37 -> -23.75 / -19.85 -> -20.73 /
    -18.00 -> -18.18; `combined` moves by the same amounts; very fine silt 0.00. These replace G-11's earlier
    numbers (-41.6 -> -42.5, -32.9 -> -35.9, -21.4 -> -27.8), which used the misprinted pi^(+1/2)."""
    cases = (("medium_sand", 30e3, 5.0, -41.58, -41.69), ("medium_sand", 100e3, 5.0, -32.88, -33.56),
             ("medium_sand", 100e3, 10.0, -29.70, -29.82), ("sandy_gravel", 30e3, 5.0, -29.03, -29.89),
             ("sandy_gravel", 100e3, 5.0, -21.37, -23.75), ("sandy_gravel", 100e3, 10.0, -19.85, -20.73),
             ("sandy_gravel", 100e3, 20.0, -18.00, -18.18))
    db = lambda x: 10 * np.log10(float(x))                                                # noqa: E731
    for name, f, deg, before, after in cases:
        par = bottom.sediment(name)
        g = np.radians(deg)
        off, on = rough("composite", g, g, par=par, f=f, shadowing="none"), rough("composite", g, g, par=par, f=f)
        assert db(off) == pytest.approx(before, abs=0.006) and db(on) == pytest.approx(after, abs=0.006), (name, f, deg)
        c_off, c_on = rough("combined", g, g, par=par, f=f, shadowing="none"), rough("combined", g, g, par=par, f=f)
        assert db(c_on) - db(c_off) == pytest.approx(after - before, abs=0.011), (name, f, deg)      # two 2-decimal roundings
    silt = bottom.sediment("very_fine_silt")
    g = np.radians([5.0, 10.0, 20.0])
    assert np.allclose(10 * np.log10(rough("composite", g, g, par=silt, f=100e3) / rough("composite", g, g, par=silt, f=100e3, shadowing="none")),
                       0.0, atol=0.005)


def test_wagner_shadowing_would_not_change_the_surface_facet_term():
    """[W67] C-W3: the sea-surface facet term (surface.facet_backscatter) carries no shadowing, because R with the
    total Cox-Munk slope is 1 to 1e-6 dB wherever that term is within 30 dB of its peak (3-20 m/s)."""
    from gpusonar.acoustics import mean_square_slope
    from gpusonar.sim import surface
    g = np.radians(np.linspace(1.0, 90.0, 8901))
    for wind in (3.0, 5.0, 10.0, 15.0, 20.0):
        facet = surface.facet_backscatter(g, wind)
        near = 10 * np.log10(np.maximum(facet, 1e-300) / facet.max()) > -30.0
        shade = bottom.wagner_backscatter_shadowing(g[near], math.sqrt(mean_square_slope(wind)))
        assert np.all(np.abs(10 * np.log10(shade)) < 1e-6), wind
