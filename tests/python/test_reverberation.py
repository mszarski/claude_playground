"""Milestone 12 (part): bottom reverberation by the point-scatterer method.

Validation: reverberation level against time matches the analytic sonar-equation
prediction within 2 dB for a flat, uniform case.
"""

import numpy as np
import pytest

from gpusonar.sim import Scene, lambert_rms_amplitude, make_pose, seafloor_patch, simulate_ping
from gpusonar.sonar import SonarModel

C = 1500.0
MU_DB = -27.0          # Lambert parameter (Mackenzie's classic value for the seabed)
HEIGHT = 20.0
PULSE = 2e-3


def reverberation_ping(seed, density=1.5, radius=260.0):
    """An omnidirectional sonar 20 m above a flat, uniformly rough seabed; CW pulse."""
    sonar = SonarModel.load("popoto_8", {
        "receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": [25.0, 230.0]},
                    "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3,
                                                          "duration": PULSE, "window_alpha": 0.0,
                                                          "source_level_db": None, "amplitude": 1.0}}})
    rng = np.random.default_rng(seed)
    pts, amp = seafloor_patch((-radius, radius), (-radius, radius), density, depth=0.0,
                              rms_amplitude=lambert_rms_amplitude(MU_DB, density), rng=rng)
    scene = Scene().add_points(pts, amp, lambert=True)
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, -HEIGHT)), scene,
                        environment={"sound_speed": C}, absorption_model="none", adc=False)
    return rec


def test_reverberation_level_follows_the_sonar_equation():
    """Mean intensity against time, averaged over channels and realisations, against

        I(t) = mu sin^2(g) * A(t) / R^4,   A(t) = 2 pi R_ground * (c T / 2) / cos(g)

    for unit source amplitude at 1 m: the insonified annulus times Lambert's strength,
    two-way spherical spreading. With R_ground = R cos(g) this is I = mu sin^2(g) pi c T / R^3.
    """
    recs = [reverberation_ping(seed) for seed in (1, 2, 3)]
    x = np.concatenate([r.samples.array for r in recs], axis=0)          # 24 channel-realisations
    s = recs[0].samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    intensity = (np.abs(x) ** 2).mean(axis=0)
    # Analytic level at the middle of the pulse footprint.
    slant = 0.5 * C * (t - 0.5 * PULSE)
    ok = (slant > 1.5 * HEIGHT) & (slant < 200.0)
    sin_g = HEIGHT / slant
    predicted = 10 ** (MU_DB / 10) * sin_g**2 * np.pi * C * PULSE / slant**3
    # Compare in range bins (speckle averages down within each).
    edges = np.linspace(np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1], 13).astype(int)
    diffs = []
    for a, b in zip(edges[:-1], edges[1:]):
        diffs.append(10 * np.log10(intensity[a:b].mean() / predicted[a:b].mean()))
    diffs = np.array(diffs)
    assert np.abs(diffs).max() < 2.0, diffs
    assert abs(diffs.mean()) < 1.0
    # The level really does fall steeply over this span: about 30 log R + 20 log(sin g).
    assert 10 * np.log10(predicted[ok][0] / predicted[ok][-1]) > 30.0


def test_reverberation_is_rayleigh_distributed():
    """Many scatterers per resolution cell: Gaussian field, Rayleigh envelope."""
    rec = reverberation_ping(7)
    x = rec.samples.array
    s = rec.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    slant = 0.5 * C * t
    sel = (slant > 60.0) & (slant < 180.0)
    # Remove the deterministic range trend, then pool.
    env = np.abs(x[:, sel])
    env = env / np.sqrt((env**2).mean(axis=0, keepdims=True))
    env = env.ravel()
    # Rayleigh with unit mean square: mean = sqrt(pi)/2, scintillation index of intensity = 1.
    assert env.mean() == pytest.approx(np.sqrt(np.pi) / 2, rel=0.03)
    inten = env**2
    assert inten.var() / inten.mean() ** 2 == pytest.approx(1.0, rel=0.25)


def test_lambert_flag_only_changes_boundary_patches():
    sonar = SonarModel.load("popoto_8", {"receive": {"kind": "baseband", "sample_rate": 20e3,
                                                      "window": {"range": [40.0, 60.0]}}})
    p = [[0.0, 40.0, 30.0]]                                       # grazing angle 36.9 degrees from 30 m up
    kw = dict(environment={"sound_speed": C}, absorption_model="none", adc=False)
    pose = make_pose()
    plain = simulate_ping(sonar, pose, Scene().add_points(p), **kw).samples.array
    patch = simulate_ping(sonar, pose, Scene().add_points(p, lambert=True), **kw).samples.array
    ratio = np.abs(patch).max() / np.abs(plain).max()
    assert ratio == pytest.approx(30.0 / 50.0, rel=0.02)          # sqrt(sin g * sin g) = sin g = 0.6
