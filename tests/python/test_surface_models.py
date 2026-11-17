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
    strength = surface.kraus_surface_strength(8.0, 30e3)
    assert strength(g[0], g[2]) == pytest.approx(10 ** (float(surface.kraus_surface_db(0.5 * (g[0] + g[2]), 8.0, 30e3)) / 10))


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
