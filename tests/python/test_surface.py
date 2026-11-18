"""Milestone 12: sea-surface backscatter (Chapman-Harris) and rough-surface reflection loss."""

import numpy as np
import pytest

from gpusonar import acoustics
from gpusonar.sim import FreeFieldProvider, Scene, find_bellhop, make_pose, reverb, simulate_ping
from gpusonar.sonar import SonarModel

C = 1500.0


def test_chapman_harris_values():
    # 30 degrees: the angle term vanishes and S = -42.4 log10(beta) + 2.6.
    v_knots, f = 20.0, 1000.0
    beta = 158.0 * (v_knots * f ** (1 / 3)) ** -0.58
    s30 = acoustics.chapman_harris_db(np.radians(30.0), v_knots / acoustics.KNOTS_PER_MS, f)
    assert s30 == pytest.approx(-42.4 * np.log10(beta) + 2.6, abs=1e-9)
    assert s30 == pytest.approx(-34.0, abs=0.3)
    # The same formula as published with the wind in m/s uses 107 in place of 158.
    for wind in (3.0, 8.0, 15.0):
        for graze in (5.0, 20.0):
            beta_ms = 107.0 * (wind * 3000.0 ** (1 / 3)) ** -0.58
            alt = 3.3 * beta_ms * np.log10(graze / 30.0) - 42.4 * np.log10(beta_ms) + 2.6
            assert acoustics.chapman_harris_db(np.radians(graze), wind, 3000.0) == pytest.approx(alt, abs=0.3)
    # Stronger with wind and with grazing angle.
    g = np.radians([2.0, 5.0, 10.0, 30.0])
    assert np.all(np.diff(acoustics.chapman_harris_db(g, 8.0, 3000.0)) > 0)
    assert acoustics.chapman_harris_db(g[1], 12.0, 3000.0) > acoustics.chapman_harris_db(g[1], 4.0, 3000.0)


def test_rough_surface_reflection():
    assert acoustics.rms_wave_height(10.0) == pytest.approx(0.535, abs=0.01)     # Hs about 2.1 m
    assert acoustics.surface_coherent_reflection(0.2, 0.0, 8000.0) == 1.0
    k, h, g = 2 * np.pi * 8000.0 / C, acoustics.rms_wave_height(5.0), np.radians(4.0)
    assert acoustics.surface_coherent_reflection(g, 5.0, 8000.0, C) == pytest.approx(
        np.exp(-2 * (k * h * np.sin(g)) ** 2))
    # Steeper and windier both lose more.
    assert acoustics.surface_coherent_reflection(0.2, 5.0, 8000.0) < acoustics.surface_coherent_reflection(0.1, 5.0, 8000.0)


def test_surface_reverberation_level():
    """Sonar 20 m below the surface, Chapman-Harris at 8 m/s: the generated level follows
    S(theta) x insonified area / r^4."""
    depth, pulse, wind, fc = 20.0, 2e-3, 8.0, 27.5e3
    intensity = 0.0
    for seed in range(6):
        sonar = SonarModel.load("popoto_8", {
            "receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": [25.0, 230.0]},
                        "sensitivity_db": None},
            "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": fc, "f_stop": fc,
                                                              "duration": pulse, "window_alpha": 0.0,
                                                              "source_level_db_rms": None, "amplitude": 1.0}}})
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, depth)), Scene(), environment={"sound_speed": C},
                            adc=False, absorption_model="none", rng=np.random.default_rng(seed),
                            reverberation=[{"boundary_z": 0.0, "model": "chapman_harris", "wind_speed": wind}])
        intensity = intensity + (np.abs(rec.samples.array) ** 2).mean(axis=0) / 6
    s = rec.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    slant = 0.5 * C * (t - 0.5 * pulse)
    ok = (slant > 1.5 * depth) & (slant < 200.0)
    strength = 10 ** (acoustics.chapman_harris_db(np.arcsin(depth / np.maximum(slant, depth)), wind, fc) / 10)
    predicted = strength * np.pi * C * pulse / slant**3
    edges = np.linspace(np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1], 13).astype(int)
    diffs = np.array([10 * np.log10(intensity[a:b].mean() / predicted[a:b].mean())
                      for a, b in zip(edges[:-1], edges[1:])])
    assert np.abs(diffs).max() < 2.0, diffs


def test_unknown_reverberation_model_is_refused():
    sonar = SonarModel.load("popoto_8", {"receive": {"window": {"range": [25.0, 40.0]}}})
    with pytest.raises(ValueError, match="reverberation model"):
        simulate_ping(sonar, make_pose(position=(0, 0, 20.0)), Scene(), adc=False,
                      reverberation=[{"boundary_z": 0.0, "model": "nope"}])


@pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_wind_removes_surface_bounce_energy_at_long_range():
    from gpusonar.sim.bellhop import BellhopProvider, OceanEnvironment
    env = OceanEnvironment(depth=100.0, sound_speed=C, bottom_speed=1600.0, bottom_density=1.8,
                           bottom_attenuation=0.5)
    points = np.array([[2000.0, 0.0, 50.0], [8000.0, 0.0, 50.0]])
    source = np.array([0.0, 0.0, 30.0])

    def rays(wind):
        p = BellhopProvider(env, 8000.0, max_range=8200.0, range_step=50.0, depth_step=5.0, max_arrivals=80,
                            wind_speed=wind)
        return p.eigenrays(source, points)

    calm, windy = rays(0.0), rays(8.0)
    e_calm = np.sum(np.abs(calm.amplitude) ** 2, axis=1)
    e_windy = np.sum(np.abs(windy.amplitude) ** 2, axis=1)
    loss_db = 10 * np.log10(e_calm / e_windy)
    assert np.all(loss_db > 1.0)
    # What survives is dominated by paths with few surface bounces.
    strongest = np.argmax(np.abs(windy.amplitude), axis=1)
    assert np.all(windy.n_surface[np.arange(2), strongest] <= 1)
    # An arrival with no surface bounce is untouched.
    direct = (calm.n_surface == 0) & calm.valid
    assert direct.any()
    for i in range(2):
        a = np.sort(np.abs(calm.amplitude[i][direct[i]]))
        b = np.sort(np.abs(windy.amplitude[i][(windy.n_surface[i] == 0) & windy.valid[i]]))
        np.testing.assert_allclose(a, b, rtol=1e-12)
