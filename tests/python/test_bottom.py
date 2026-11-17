"""The rough fluid sediment scattering model (sim/bottom.py): limits with known answers."""
import numpy as np
import pytest

from gpusonar.sim import bottom, reverb

C, F = 1500.0, 100e3
K = 2 * np.pi * F / C
SAND = bottom.PLACEHOLDER_SAND
ROUGH = dict(w2=SAND["w2"], gamma2=SAND["gamma2"])
G = np.radians(np.array([5.0, 10.0, 20.0, 35.0, 50.0]))


def spectrum(q):
    return ROUGH["w2"] / q ** ROUGH["gamma2"]


def test_pressure_release_limit_is_the_classical_bragg_result():
    got = bottom.roughness_strength(G, G, F, C, rho=1e-9, nu=1.0, delta=0.0, **ROUGH)
    assert np.allclose(got, 4 * K**4 * np.sin(G) ** 4 * spectrum(2 * K * np.cos(G)), rtol=1e-6)
    # Bistatic, in the plane of incidence and out of it.
    ti, ts, phi = np.radians(20.0), np.radians(40.0), np.radians(70.0)
    q = K * np.sqrt(np.cos(ti) ** 2 + np.cos(ts) ** 2 - 2 * np.cos(ti) * np.cos(ts) * np.cos(phi))
    got = bottom.roughness_strength(ti, ts, F, C, rho=1e-9, nu=1.0, delta=0.0, azimuth=phi, **ROUGH)
    assert got == pytest.approx(4 * K**4 * np.sin(ti) ** 2 * np.sin(ts) ** 2 * spectrum(q), rel=1e-6)


def test_rigid_limit():
    got = bottom.roughness_strength(G, G, F, C, rho=1e9, nu=1.0, delta=0.0, **ROUGH)
    assert np.allclose(got, 4 * K**4 * (1 + np.cos(G) ** 2) ** 2 * spectrum(2 * K * np.cos(G)), rtol=1e-6)


def test_no_contrast_scatters_nothing_and_reciprocity_holds():
    assert np.all(bottom.roughness_strength(G, G, F, C, rho=1.0, nu=1.0, delta=0.0, **ROUGH) < 1e-20)
    a = bottom.strength(np.radians(15.0), np.radians(40.0), F, C, azimuth=np.radians(120.0), **SAND)
    b = bottom.strength(np.radians(40.0), np.radians(15.0), F, C, azimuth=np.radians(120.0), **SAND)
    assert a == pytest.approx(b, rel=1e-10)


def test_reflection_coefficient_and_critical_angle():
    v = bottom.reflection(G, SAND["rho"], SAND["nu"], 0.0)
    critical = np.arccos(1.0 / SAND["nu"])                             # 24.6 degrees
    assert np.allclose(np.abs(v[G < critical]), 1.0)                   # total reflection below it, without loss
    assert np.all(np.abs(v[G > critical]) < 1.0)
    normal = bottom.reflection(np.pi / 2, SAND["rho"], SAND["nu"], 0.0)
    z = SAND["rho"] * SAND["nu"]
    assert normal.real == pytest.approx((z - 1) / (z + 1)) and abs(normal.imag) < 1e-12
    assert abs(bottom.reflection(np.radians(10.0), SAND["rho"], SAND["nu"], SAND["delta"])) < 1.0   # loss leaks energy


def test_volume_term_is_the_depth_integral_of_what_it_says():
    """Sum the scatterers layer by layer: transmitted intensity in, reciprocal transmission out."""
    g = np.radians(40.0)
    kappa = (1 + 1j * SAND["delta"]) / SAND["nu"]
    b2 = K * np.sqrt(kappa**2 - np.cos(g) ** 2 + 0j)
    t = 1 + bottom.reflection(g, SAND["rho"], SAND["nu"], SAND["delta"])
    sigma_vol = SAND["sigma2"] * (20 / np.log(10)) * K * SAND["delta"] / SAND["nu"]
    z = np.linspace(0.0, 40.0 / b2.imag, 200_001)
    layers = sigma_vol * abs(t) ** 4 / SAND["rho"] ** 2 * np.exp(-4 * b2.imag * z)
    assert bottom.volume_strength(g, g, F, C, **SAND) == pytest.approx(np.trapezoid(layers, z), rel=1e-5)
    assert bottom.volume_strength(g, g, F, C, **dict(SAND, sigma2=0.0)) == 0.0
    # Below the critical angle the field in the sediment is evanescent: far less volume scattering.
    assert bottom.volume_strength(np.radians(10.0), np.radians(10.0), F, C, **SAND) < 0.2 * bottom.volume_strength(g, g, F, C, **SAND)


def test_frequency_dependence_of_the_roughness_term():
    """sigma_r ~ k^4 W(2 k cos) ~ f^(4 - gamma2) at a fixed angle (lossless sediment)."""
    p = dict(SAND, delta=0.0)
    a = bottom.roughness_strength(G, G, 50e3, C, **p)
    b = bottom.roughness_strength(G, G, 200e3, C, **p)
    assert np.allclose(b / a, 4.0 ** (4.0 - SAND["gamma2"]), rtol=1e-9)


def test_placeholder_sand_is_what_its_note_says_and_lambert_equivalent_matches():
    bs = float(bottom.backscattering_strength_db(np.radians(20.0), 100e3, C, **SAND))
    assert -33.0 < bs < -27.0
    mu = bottom.equivalent_lambert_mu_db(np.radians(20.0), 100e3, C, **SAND)
    assert mu + 20 * np.log10(np.sin(np.radians(20.0))) == pytest.approx(bs)


def test_reverberation_with_the_model_differs_from_lambert_by_the_ratio_of_strengths():
    from test_reverberation import HEIGHT, MU_DB, PULSE, empty_baseband_ping
    from gpusonar.sim import FreeFieldProvider
    sonar, rec = empty_baseband_ping()
    fc = rec.samples.center_frequency
    s = rec.samples
    times = s.start_time - PULSE + np.arange(s.n_samples) / s.sample_rate
    lam = reverb.reverberation_power(FreeFieldProvider(C), np.array(rec.tx_position()), 0.0, times, MU_DB)
    mod = reverb.reverberation_power(FreeFieldProvider(C), np.array(rec.tx_position()), 0.0, times, 0.0,
                                     strength=bottom.reverberation_strength(fc, C, **SAND))
    slant = 0.5 * C * times
    ok = (slant > 2.0 * HEIGHT) & (lam > 0) & (mod > 0)
    grazing = np.arcsin(HEIGHT / slant[ok])
    want = bottom.strength(grazing, grazing, fc, C, **SAND) / (10 ** (MU_DB / 10) * np.sin(grazing) ** 2)
    assert np.allclose(mod[ok] / lam[ok], want, rtol=0.05)


def test_scenario_reverberation_item_uses_the_model():
    from test_reverberation import HEIGHT, empty_baseband_ping
    from gpusonar.sim import Scene, make_pose, simulate_ping
    sonar, _ = empty_baseband_ping()

    def power(**extra):
        item = dict({"model": "fluid_sediment", "boundary_z": 0.0}, **dict(SAND, **extra))
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                            reverberation=[item], rng=np.random.default_rng(2))
        return float((np.abs(rec.samples.array) ** 2).mean())
    assert power() > 0.0
    assert power(w2=4 * SAND["w2"], sigma2=0.0) / power(sigma2=0.0) == pytest.approx(4.0, rel=1e-6)   # roughness term is linear in w2
    with pytest.raises(ValueError, match="needs"):
        simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                      reverberation=[{"model": "fluid_sediment", "boundary_z": 0.0, "rho": 1.9}])
