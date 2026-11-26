"""Wind-dependent sea-surface models (sim/surface.py): bubble loss per bounce from the
Hall-Novarini population, and the empirical high-frequency backscattering laws."""
import warnings

import numpy as np
import pytest

from gpusonar.sim import surface
from gpusonar.sim.timespread import TimeSpreadProvider, boundary_parameters

C = 1500.0


def test_hall_novarini_population_as_printed():
    """arXiv:2603.23727 eqs. (20)-(23): at the reference radius G = 1."""
    for u, scale in ((5.0, 0.4), (7.5, 0.4), (13.0, 0.4 + 0.115 * 5.5), (20.0, 0.4 + 0.115 * 12.5)):
        for z in (0.0, 1.0, 3.0):
            r_ref = 54.4e-6 + 1.984e-6 * z
            assert float(surface.hall_novarini_density(r_ref, z, u)) == pytest.approx(1.6e10 * (u / 13.0) ** 3 * np.exp(-z / scale))
    n = lambda r, z=0.0: float(surface.hall_novarini_density(r, z, 13.0))      # noqa: E731
    assert n(27.2e-6) == pytest.approx(1.6e10 * 2.0**4)                        # below the reference radius: r^-4
    assert n(108.8e-6) == pytest.approx(1.6e10 * 0.5**4.37)                    # above: r^-kappa, kappa = 4.37 at the surface
    assert n(108.8e-6 + 2 * 1.984e-6 * 2.55, 2.55) == pytest.approx(1.6e10 * 0.5**5.37 * np.exp(-2.55 / (0.4 + 0.115 * 5.5)))
    assert n(5e-6) == 0.0 and n(2e-3) == 0.0 and float(surface.hall_novarini_density(1e-4, 1.0, 0.0)) == 0.0


def test_resonant_radius_is_minnaert():
    a = surface.resonant_radius(30e3)
    assert 2 * np.pi * 30e3 * a == pytest.approx(np.sqrt(3 * 1.4 * 101325.0 / 1025.0))
    assert 100e-6 < a < 120e-6                                                  # about 0.1 mm at 30 kHz
    assert surface.resonant_radius(30e3, 10.0) > a                             # deeper: stiffer, so a larger bubble


@pytest.mark.parametrize("damping", [0.03, 0.08])
def test_extinction_is_the_integral_over_the_resonance(damping):
    """Sum the extinction cross sections of damped bubbles of all radii: the resonance
    approximation must come out, whatever the damping."""
    f, u, z = 20e3, 10.0, 0.5
    k = 2 * np.pi * f / C
    a_r = float(surface.resonant_radius(f, z))
    a = np.linspace(surface.R_MIN, surface.R_MAX, 2_000_001)
    scatter = 4 * np.pi * a**2 / (((a_r / a) ** 2 - 1) ** 2 + damping**2)
    extinction = scatter * damping / (k * a)
    near = np.abs(a - a_r) < 12 * damping * a_r                               # the resonance; off-resonant bubbles are not in the model
    direct = np.trapezoid((extinction * surface.hall_novarini_density(a, z, u))[near], a[near])
    assert float(surface.bubble_extinction(f, z, u, C)) == pytest.approx(direct, rel=0.1)


def test_bubble_loss_scaling_and_range():
    f = 8e3
    g = np.radians([5.0, 10.0, 30.0, 90.0])
    loss = surface.bubble_loss_db(g, 6.0, f)
    assert np.all(loss > 0) and np.allclose(loss * np.sin(g), loss[-1])       # slant path through the layer, twice
    assert loss[-1] == pytest.approx(10 / np.log(10) * 2 * surface.bubble_layer_depth_integral(f, 6.0))
    # Below 7.5 m/s the layer depth is fixed, so the loss is exactly cubic in wind speed.
    assert surface.bubble_loss_db(g, 3.0, f) == pytest.approx(loss / 8.0)
    assert surface.bubble_loss_db(g, 12.0, f)[0] > 8.0 * loss[0]              # above, the layer also deepens
    assert np.all(surface.bubble_loss_db(g, 0.0, f) == 0.0)
    assert surface.bubble_loss_db(0.0, 6.0, f) == surface.bubble_loss_db(surface.MIN_GRAZING, 6.0, f)   # floored, finite
    # The depth integral, done by hand for one case with a coarse sum.
    z = np.linspace(0.0, 12.0, 240_001)
    a = surface.resonant_radius(f, z)
    beta = 2 * np.pi**2 * a**2 * surface.hall_novarini_density(a, z, 6.0) / (2 * np.pi * f / C)
    assert surface.bubble_layer_depth_integral(f, 6.0) == pytest.approx(np.trapezoid(beta, z), rel=1e-4)
    with pytest.warns(UserWarning, match="outside the Hall-Novarini population"):
        assert np.all(surface.bubble_loss_db(g, 6.0, 1e3) == 0.0)             # resonant radius 3 mm: not in the population
    with pytest.warns(UserWarning, match="above 20"):
        surface.bubble_loss_db(g, 25.0, f)
    with pytest.warns(UserWarning, match="probably too high"):
        surface.bubble_loss_db(g, 6.0, 30e3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        surface.bubble_loss_db(g, 6.0, f)                                     # inside every range: silent


def test_time_spread_takes_the_bubble_loss_model():
    from test_timespread import SOURCE, F, one_bounce
    inner, pts, _ = one_bounce()
    base = inner.eigenrays(SOURCE, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    par = boundary_parameters({"wind_speed": 6.0, "rms_height": 0.0, "bubble_loss_db": {"model": "hall_novarini"}}, surface=True, frequency=F)
    grid, table = par["bubble_loss"]
    assert np.allclose(table[1:], surface.bubble_loss_db(grid[1:], 6.0, F))
    got = TimeSpreadProvider(inner, frequency=F, surface=par).eigenrays(SOURCE, pts)
    grazing = np.arcsin(abs(base.launch[0, j, 2]))
    want = float(surface.bubble_loss_db(grazing, 6.0, F))
    assert want > 0.3
    assert -20 * np.log10(abs(got.amplitude[0, j]) / abs(base.amplitude[0, j])) == pytest.approx(want, rel=0.02)   # table interpolation
    # Its own wind speed, and what it refuses.
    other = boundary_parameters({"rms_height": 0.1, "bubble_loss_db": {"model": "hall_novarini", "wind_speed": 3.0}}, surface=True, frequency=F)
    assert np.allclose(other["bubble_loss"][1], table / 8.0)
    with pytest.raises(ValueError, match="unknown bubble loss model"):
        boundary_parameters({"wind_speed": 6.0, "bubble_loss_db": {"model": "apl"}}, surface=True, frequency=F)
    with pytest.raises(ValueError, match="needs"):
        boundary_parameters({"rms_height": 0.1, "bubble_loss_db": {"model": "hall_novarini"}}, surface=True, frequency=F)
    # Numbers and tables still work as before, with or without the new argument.
    assert np.array_equal(boundary_parameters({"bubble_loss_db": 3.0}, surface=True)["bubble_loss"][1], [3.0, 3.0])
    assert "bubble_loss" not in boundary_parameters({"wind_speed": 6.0}, surface=True, frequency=F)


def test_kraus_surface_formula_as_printed():
    """arXiv:2210.06535 Sec. II-F, worked by hand: v = 10 knots, f = 30 kHz, 20 degrees."""
    v, f, t = 10.0, 30.0, np.radians(20.0)
    b = 4 * (v + 2) / (v + 1) + (2.5 * (f + 0.1) ** (-1 / 3) - 4) * np.cos(t) ** (1 / 8)
    want = 10 * np.log10(10**-5.05 * (1 + v) ** 2 * (f + 0.1) ** (v / 150) * np.tan(t) ** b)
    assert float(surface.kraus_surface_db(t, v / 1.943844, 30e3)) == pytest.approx(want, abs=1e-9)
    assert -40.0 < want < -20.0
    g = np.radians([5.0, 15.0, 30.0, 50.0])
    assert np.all(np.diff(surface.kraus_surface_db(g, 8.0, 30e3)) > 0)                                   # rises with grazing angle
    assert np.all(surface.kraus_surface_db(g, 12.0, 30e3) > surface.kraus_surface_db(g, 4.0, 30e3))     # and with wind
    assert surface.kraus_surface_db(np.radians(85.0), 8.0, 30e3) == surface.kraus_surface_db(np.radians(60.0), 8.0, 30e3)   # held
    with pytest.warns(UserWarning, match="below 10000 Hz"):
        surface.kraus_surface_db(g, 8.0, 3e3)
    # A pair of angles: the geometric mean of the two strengths (A3; TR 0702 Sec. 6.4.2); the mean angle before.
    strength = surface.kraus_surface_strength(8.0, 30e3)
    assert strength(g[0], g[2]) == pytest.approx(10 ** ((float(surface.kraus_surface_db(g[0], 8.0, 30e3))
                                                         + float(surface.kraus_surface_db(g[2], 8.0, 30e3))) / 20))
    before = surface.kraus_surface_strength(8.0, 30e3, bistatic_rule="mean_angle")
    assert before(g[0], g[2]) == pytest.approx(10 ** (float(surface.kraus_surface_db(0.5 * (g[0] + g[2]), 8.0, 30e3)) / 10))


def test_searay_bottom_formula_as_printed():
    """Same source, Sec. II-E, worked by hand for sand (bt = 2) at 100 kHz and 20 degrees."""
    bt, f, t = 2.0, 100.0, np.radians(20.0)
    gam = 1 + 125 * np.exp(-2.64 * (bt - 1.75) ** 2 - 50 / bt / np.tan(t) ** 2)
    beta = gam * (np.sin(t) + 0.19) ** (bt * np.cos(t) ** 16)
    want = 10 * np.log10(3.03 * beta * f ** (3.2 - 0.8 * bt) * 10 ** (2.8 * bt - 12) + 10**-4.42)
    assert float(surface.searay_bottom_db(t, 100e3, "sand")) == pytest.approx(want, abs=1e-9)
    assert float(surface.searay_bottom_db(t, 100e3, 2)) == pytest.approx(want, abs=1e-9)
    levels = [float(surface.searay_bottom_db(t, 100e3, name)) for name in ("mud", "sand", "gravel", "rock")]
    assert levels == sorted(levels) and -45.0 < levels[0] and levels[-1] < 0.0
    with pytest.raises(ValueError, match="bottom type"):
        surface.searay_bottom_db(t, 100e3, 7)


def test_scenario_reverberation_items_for_the_empirical_laws():
    from test_reverberation import HEIGHT, empty_baseband_ping
    from gpusonar.sim import Scene, make_pose, simulate_ping
    sonar, rec0 = empty_baseband_ping()
    fc = rec0.samples.center_frequency

    def power(item):
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                            reverberation=[dict(item, boundary_z=0.0)], rng=np.random.default_rng(2))
        return float((np.abs(rec.samples.array) ** 2).mean())
    calm, windy = power({"model": "kraus_surface", "wind_speed": 3.0}), power({"model": "kraus_surface", "wind_speed": 12.0})
    assert 0.0 < calm < windy
    # The same random draw scaled by the strength: two bottom types differ by a ratio between the
    # smallest and largest ratio of their strengths over the angles in the record.
    mud, rock = power({"model": "searay_bottom", "bottom_type": "mud"}), power({"model": "searay_bottom", "bottom_type": "rock"})
    g = np.radians(np.linspace(3.0, 60.0, 200))
    ratios = 10 ** ((surface.searay_bottom_db(g, fc, "rock") - surface.searay_bottom_db(g, fc, "mud")) / 10)
    assert ratios.min() <= rock / mud <= ratios.max()
    with pytest.raises(ValueError, match="unknown reverberation model"):
        power({"model": "apl_surface"})


# --- The empirical laws against the bubble layer's saturation level (D-87) --------------------------

def test_saturation_level_and_where_the_empirical_laws_cross_it():
    """The level is eq. (11), delta_r sin(t) / (8 pi delta): the limit of the bubble term for an
    opaque layer. The crossing frequencies are the table in sim/surface.py and CONFIGURATION."""
    from gpusonar import acoustics
    t = np.radians(10.0)
    for f, want in ((12e3, -27.9), (30e3, -29.3), (68e3, -30.4)):
        level = float(surface.bubble_saturation_db(t, f))
        assert level == pytest.approx(want, abs=0.05)
        assert level == pytest.approx(10 * np.log10(0.0136 * np.sin(t) / (8 * np.pi * 2.55e-2 * (f / 1e3) ** (1 / 3))), abs=1e-12)
        # the bubble term itself, with a very thick layer (hall_novarini at 40 m/s is opaque)
    thick = 10 * np.log10(surface.bubble_backscatter(t, 15.0, 68e3))
    assert thick == pytest.approx(float(surface.bubble_saturation_db(t, 68e3)), abs=1.0)
    table = {"chapman_harris": {3.0: 416, 5.0: 127, 6.0: 84, 8.0: 43, 10.0: 26, 15.0: 10},
             "kraus_surface": {3.0: None, 5.0: 511, 6.0: 223, 8.0: 74, 10.0: 36, 15.0: 12}}
    for model, row in table.items():
        for wind, khz in row.items():
            f = surface.saturation_crossing_frequency(model, wind)
            if khz is None:
                assert f is None
                continue
            assert f / 1e3 == pytest.approx(khz, abs=0.6)
            assert surface.excess_over_saturation_db(model, wind, 0.98 * f) < 0.0 < surface.excess_over_saturation_db(model, wind, 1.02 * f)
            assert surface.excess_over_saturation_db(model, wind, 3.0 * f) > 0.0          # and it stays above
    # The case quoted in REFERENCES: 68 kHz, 15 m/s, 10 degrees: -19.1 dB against -30.4 dB.
    assert float(acoustics.chapman_harris_db(t, 15.0, 68e3)) == pytest.approx(-19.1, abs=0.05)
    assert surface.excess_over_saturation_db("chapman_harris", 15.0, 68e3) == pytest.approx(11.4, abs=0.1)


def test_empirical_surface_strength_is_the_law_unchanged_warns_once_and_caps_on_request():
    import warnings
    from gpusonar.sim import reverb
    from gpusonar import acoustics
    g_in, g_out = np.radians([[3.0, 10.0, 25.0, 50.0]]), np.radians([[5.0], [20.0]])
    # Below the crossing: the law, bit for bit, and no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = surface.empirical_surface_strength("chapman_harris", 5.0, 8e3)
        assert np.array_equal(quiet(g_in, g_out), reverb.chapman_harris_strength(5.0, 8e3)(g_in, g_out))
        kraus = surface.empirical_surface_strength("kraus_surface", 5.0, 100e3)
        assert np.array_equal(kraus(g_in, g_out), surface.kraus_surface_strength(5.0, 100e3)(g_in, g_out))
        # 15 m/s crosses at 10 kHz, below the band the bubble model was compared with data in: no warning under 12 kHz.
        surface.empirical_surface_strength("chapman_harris", 15.0, 11e3)
    # Above it: still the law, with one warning per law for the process.
    surface._warned_above_saturation.clear()
    with pytest.warns(UserWarning, match="above the level a saturated bubble layer returns") as caught:
        loud = surface.empirical_surface_strength("chapman_harris", 5.0, 170e3)
    assert "no model here is validated" in str(caught[0].message) and "cap: bubble_saturation" in str(caught[0].message)
    assert np.array_equal(loud(g_in, g_out), reverb.chapman_harris_strength(5.0, 170e3)(g_in, g_out))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        surface.empirical_surface_strength("chapman_harris", 5.0, 330e3)                  # once
    with pytest.warns(UserWarning, match="bubble_layer_surface. is the model compared with data"):
        surface.empirical_surface_strength("kraus_surface", 10.0, 50e3)                   # the other law, inside 12-70 kHz
    # The cap: the smaller of the law and the saturation level at each angle, the pair by the geometric
    # mean (A3; at the mean angle before, `bistatic_rule="mean_angle"`); no warning.
    surface._warned_above_saturation.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        capped = surface.empirical_surface_strength("chapman_harris", 5.0, 330e3, cap="bubble_saturation")
    mean = 0.5 * (g_in + g_out)
    law = reverb.chapman_harris_strength(5.0, 330e3)(g_in, g_out)
    level = np.sqrt(10 ** (surface.bubble_saturation_db(g_in, 330e3) / 10) * 10 ** (surface.bubble_saturation_db(g_out, 330e3) / 10))
    one = lambda g: np.minimum(10 ** (acoustics.chapman_harris_db(g, 5.0, 330e3) / 10), 10 ** (surface.bubble_saturation_db(g, 330e3) / 10))  # noqa: E731
    assert np.allclose(capped(g_in, g_out), np.sqrt(one(g_in) * one(g_out)), rtol=1e-12)
    assert np.any(law > level) and np.all(capped(g_in, g_out) <= level * (1 + 1e-12))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        capped_before = surface.empirical_surface_strength("chapman_harris", 5.0, 330e3, cap="bubble_saturation", bistatic_rule="mean_angle")
    law_before = reverb.chapman_harris_strength(5.0, 330e3, bistatic_rule="mean_angle")(g_in, g_out)
    assert np.allclose(capped_before(g_in, g_out), np.minimum(law_before, 10 ** (surface.bubble_saturation_db(mean, 330e3) / 10)), rtol=1e-12)
    low = surface.empirical_surface_strength("chapman_harris", 5.0, 8e3, cap="bubble_saturation")
    assert np.array_equal(low(g_in, g_out), quiet(g_in, g_out))                           # nothing to cap at 8 kHz
    with pytest.raises(ValueError, match="unknown cap"):
        surface.empirical_surface_strength("chapman_harris", 5.0, 8e3, cap="lambert")


def test_scenario_item_cap_and_warning():
    import warnings
    from test_reverberation import HEIGHT, empty_baseband_ping
    from gpusonar.sim import Scene, make_pose, simulate_ping
    sonar, rec0 = empty_baseband_ping()
    fc = rec0.samples.center_frequency

    def power(item):
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                            reverberation=[dict(item, boundary_z=0.0)], rng=np.random.default_rng(2))
        return float((np.abs(rec.samples.array) ** 2).mean())
    wind = 15.0
    assert surface.excess_over_saturation_db("chapman_harris", wind, fc) > 3.0            # this sonar and wind: well above
    surface._warned_above_saturation.clear()
    with pytest.warns(UserWarning, match="saturated bubble layer"):
        plain = power({"model": "chapman_harris", "wind_speed": wind})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        capped = power({"model": "chapman_harris", "wind_speed": wind, "cap": "bubble_saturation"})
    g = np.radians(np.linspace(3.0, 60.0, 200))
    from gpusonar import acoustics
    drop = np.minimum(surface.bubble_saturation_db(g, fc) - acoustics.chapman_harris_db(g, wind, fc), 0.0)
    assert 10 ** (drop.min() / 10) <= capped / plain <= 10 ** (drop.max() / 10) and capped < plain


# --- The rock type law against the Lambert energy bound, and the measured value by name (D-88) -----

def test_searay_rock_exceeds_the_lambert_bound_warns_once_and_is_unchanged():
    import warnings
    from gpusonar.sim import bottom
    g = np.radians(20.0)
    bound = bottom.lambert_backscatter_db(g, bottom.LAMBERT_MAX_MU_DB)                    # -14.3 dB: all power, diffusely
    assert float(surface.searay_bottom_db(g, 100e3, "rock")) == pytest.approx(-7.2, abs=0.05) and bound == pytest.approx(-14.3, abs=0.05)
    for f in (6e3, 100e3, 300e3):                                                         # the rock law has no frequency dependence
        excess, angle = surface.searay_excess_over_lambert_bound_db(f, "rock")
        assert excess == pytest.approx(7.5, abs=0.1) and np.degrees(angle) == pytest.approx(25.0)
        assert all(surface.searay_excess_over_lambert_bound_db(f, bt)[0] < 0.0 for bt in ("mud", "sand", "gravel"))
    surface._warned_above_lambert_bound.clear()
    with pytest.warns(UserWarning, match="sediment: rock") as caught:
        fn = surface.searay_bottom_strength(100e3, "rock")
    assert "7.5 dB above the largest Lambert" in str(caught[0].message)
    assert fn(g, g) == pytest.approx(10 ** (surface.searay_bottom_db(g, 100e3, "rock") / 10), rel=1e-12)   # the law itself is as printed
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        surface.searay_bottom_strength(100e3, "rock")                                     # once
        surface.searay_bottom_strength(100e3, "sand")                                     # below the bound: silent


def test_reverberation_item_with_a_named_rock_is_lamberts_law_at_the_tabulated_value():
    """{boundary_z, sediment: rock}: mu = -16 dB from configs/sediments.yaml (measured range -23 to -10)."""
    from test_reverberation import HEIGHT, empty_baseband_ping
    from gpusonar.sim import Scene, bottom, make_pose, simulate_ping
    sonar, _ = empty_baseband_ping()

    def ping(item):
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                            reverberation=[dict(item, boundary_z=0.0)], rng=np.random.default_rng(2))
        return rec.samples.array.copy()
    law = bottom.empirical_backscatter("rock")
    assert law["mu_db"] == -16.0 and law["mu_db_range"] == [-23.0, -10.0] and law["mu_db"] < bottom.LAMBERT_MAX_MU_DB
    assert bottom.named_lambert_mu_db("rock") == -16.0
    named = ping({"sediment": "rock"})
    assert np.array_equal(named, ping({"mu_db": -16.0})) and np.array_equal(named, ping({"model": "lambert", "sediment": "rock"}))
    assert np.array_equal(ping({"sediment": "rock", "mu_db": -13.0}), ping({"mu_db": -13.0}))          # the item's own value wins
    # Against the type law at the angles of this record: 10 to 18 dB lower.
    surface._warned_above_lambert_bound.add("rock")
    ratio = (np.abs(named) ** 2).mean() / (np.abs(ping({"model": "searay_bottom", "bottom_type": "rock"})) ** 2).mean()
    g = np.radians(np.linspace(3.0, 60.0, 200))
    change = bottom.lambert_backscatter_db(g, -16.0) - surface.searay_bottom_db(g, sonar.center_frequency, "rock")
    assert 10 ** (change.min() / 10) <= ratio <= 10 ** (change.max() / 10) and change.max() < -10.0
    with pytest.raises(ValueError, match="fluid sediment and has no Lambert parameter"):
        ping({"sediment": "medium_sand"})


# --- Bistatic pairs: the geometric mean of the two backscattering strengths (A3) -------------------

def _strength_laws(rule=None):
    kw = {} if rule is None else {"bistatic_rule": rule}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"chapman_harris": surface.empirical_surface_strength("chapman_harris", 8.0, 9.9e3, **kw),
                "chapman_harris capped": surface.empirical_surface_strength("chapman_harris", 15.0, 68e3, cap="bubble_saturation", **kw),
                "kraus_surface": surface.empirical_surface_strength("kraus_surface", 8.0, 50e3, **kw),
                "kraus_surface_strength": surface.kraus_surface_strength(8.0, 50e3, **kw),
                "bubble_layer_surface": surface.bubble_layer_surface_strength(10.0, 30e3, **kw),
                "searay_bottom": surface.searay_bottom_strength(30e3, "sand", **kw)}


def test_bistatic_rule_is_the_geometric_mean_of_the_backscattering_strengths():
    """A3 (docs/references/CONFORMANCE.md Sec. 1.1): TR 0702 Sec. 6.4.2 (printed) makes a monostatic
    model bistatic by the geometric mean of the two backscattering strengths, sigma(g_in, g_out) =
    sqrt(sigma(g_in) sigma(g_out)), where ours took sigma((g_in + g_out) / 2). Every surface law here
    (chapman_harris and kraus_surface, capped or not, bubble_layer_surface) and searay_bottom follow
    it; `bistatic_rule="mean_angle"` (scenario key `bistatic_rule: mean_angle`) is the rule before A3.
    Equal angles: the two rules agree bit for bit. Chapman-Harris at 9.9 kHz, 8 m/s (`sst_pursuit`'s
    band and wind), pairs 5/5, 5/20, 5/40, 10/30, 5/80, 2/30 degrees: after minus before 0, -1.7, -3.6,
    -1.1, -5.9, -5.7 dB (the CONFORMANCE item, computed there with `slope_fls.py`; reproduced)."""
    from gpusonar.sim import reverb
    g_in, g_out = np.radians([[2.0, 5.0, 10.0, 30.0, 60.0]]), np.radians([[5.0], [20.0], [40.0], [80.0]])
    after, before = _strength_laws(), _strength_laws("mean_angle")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in after.items():
            np.testing.assert_allclose(fn(g_in, g_out), np.sqrt(fn(g_in, g_in) * fn(g_out, g_out)), rtol=1e-12, err_msg=name)
            for g in (g_in, g_out):                                              # equal angles: bit for bit
                assert np.array_equal(fn(g, g), before[name](g, g)), name
            assert np.array_equal(_strength_laws("geometric_mean")[name](g_in, g_out), fn(g_in, g_out)), name
        # The rule before, as it was written: the law at the mean angle.
        from gpusonar import acoustics
        mean = 0.5 * (g_in + g_out)
        assert np.array_equal(before["chapman_harris"](g_in, g_out), 10.0 ** (acoustics.chapman_harris_db(mean, 8.0, 9.9e3) / 10.0))
        assert np.array_equal(before["searay_bottom"](g_in, g_out), 10.0 ** (surface.searay_bottom_db(mean, 30e3, "sand") / 10.0))
        assert np.array_equal(reverb.chapman_harris_strength(8.0, 9.9e3)(g_in, g_out), after["chapman_harris"](g_in, g_out))
        assert np.array_equal(reverb.chapman_harris_strength(8.0, 9.9e3, bistatic_rule="mean_angle")(g_in, g_out),
                              before["chapman_harris"](g_in, g_out))
    pairs = np.radians([(5, 5), (5, 20), (5, 40), (10, 30), (5, 80), (2, 30)])
    moved = 10 * np.log10(after["chapman_harris"](pairs[:, 0], pairs[:, 1]) / before["chapman_harris"](pairs[:, 0], pairs[:, 1]))
    np.testing.assert_allclose(moved, [0.0, -1.7, -3.6, -1.1, -5.9, -5.7], atol=0.06)
    # The facet term's bubble attenuation becomes Dahl (1999) eq. (12) for the pair:
    # sqrt(exp(-2 b / sin t_i) exp(-2 b / sin t_s)) = exp(-(b / sin t_i + b / sin t_s)).
    with pytest.raises(ValueError, match="unknown bistatic_rule 'mean'.*geometric_mean.*mean_angle"):
        surface.empirical_surface_strength("chapman_harris", 8.0, 9.9e3, bistatic_rule="mean")


def test_scenario_key_bistatic_rule():
    """`bistatic_rule` reaches the strength through the simulator: absent is `geometric_mean` (A3);
    `mean_angle` is the rule before. Image-source paths with a bottom bounce give pairs of unequal
    angles at the surface even for one transmitter and receiver (direct out, bottom-bounced back):
    those are the pairs that move, so a monostatic sonar's surface reverberation falls too."""
    from test_reverberation import HEIGHT, empty_baseband_ping
    from gpusonar.sim import Scene, make_pose, simulate_ping
    from gpusonar.sim.propagation import ImageSourceProvider
    sonar, _ = empty_baseband_ping()
    provider = ImageSourceProvider(C, surface_z=0.0, bottom_z=60.0, bottom_coefficient=0.5, max_order=1)

    def power(**key):
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                            provider=provider, rng=np.random.default_rng(2),
                            reverberation=[dict({"model": "chapman_harris", "wind_speed": 8.0, "boundary_z": 0.0}, **key)])
        return rec.samples.array.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default, geometric, mean = power(), power(bistatic_rule="geometric_mean"), power(bistatic_rule="mean_angle")
        assert np.array_equal(default, geometric)
        ratio = float((np.abs(default) ** 2).mean() / (np.abs(mean) ** 2).mean())
        assert ratio < 1.0, ratio                                 # Chapman-Harris here: the cross pairs fall
        with pytest.raises(ValueError, match="unknown bistatic_rule"):
            power(bistatic_rule="arithmetic")
