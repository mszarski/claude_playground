"""Seabed point scatterers that follow the bottom scattering model (seafloor `model: fluid_sediment`)."""

import numpy as np
import pytest

from gpusonar.sim import FreeFieldProvider, bottom, build_scene, make_pose, reverb, simulate_ping, targets
from gpusonar.sim.propagation import two_way_paths

C = 1500.0
F = 100e3
SAND = dict(bottom.PLACEHOLDER_SAND)
HEIGHT_ABOVE = 5.0


def seabed(density=150.0, seed=0, **extra):
    item = dict({"type": "seafloor", "x_range": [2.0, 62.0], "y_range": [-6.0, 6.0], "density": density,
                 "model": "fluid_sediment"}, **SAND, **extra)
    return build_scene([item], np.random.default_rng(seed))


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_scatterer_field_has_the_model_backscattering_strength(engine):
    """Mean backscattered intensity per unit area against grazing angle, from the amplitudes
    the path pairing hands to synthesis, against bottom.backscattering_strength_db.

    The amplitudes are complex Gaussian, so the mean of n intensities has a relative standard
    error of 1 / sqrt(n): each angle bin must be within 4 standard errors (and within 0.25 dB),
    and the whole field (108,000 scatterers, standard error 0.013 dB) within 0.06 dB."""
    density = 150.0
    scene = seabed(density)
    sonar = np.array([0.0, 0.0, -HEIGHT_ABOVE])
    pts, amp, vel = scene.evaluate(0.0, sonar, sonar, frequency=F, sound_speed=C)
    assert np.array_equal(pts, scene.positions) and scene.lambert.all()
    q, p, a, _, _ = two_way_paths(FreeFieldProvider(C), sonar, sonar, pts, amp, lambert=scene.lambert,
                                  normals=scene.normals, engine=engine)
    assert len(a) == len(scene)                                   # spreading is taken out: |a|^2 = cross section
    grazing = np.arcsin(HEIGHT_ABOVE / np.linalg.norm(pts - sonar, axis=1))
    model = 10.0 ** (bottom.backscattering_strength_db(grazing, F, C, **SAND) / 10.0)
    per_area = np.abs(a) ** 2 * density                           # each scatterer stands for 1 / density m^2
    edges = np.radians([4.7, 6, 8, 11, 15, 20, 27, 36, 48, 68])
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (grazing >= lo) & (grazing < hi)
        n = int(sel.sum())
        assert n > 250
        ratio = per_area[sel].mean() / model[sel].mean()
        assert abs(ratio - 1.0) < 4.0 / np.sqrt(n), (np.degrees(lo), n, ratio)
        worst = max(worst, abs(10 * np.log10(ratio)))
    assert worst < 0.25
    assert abs(10 * np.log10(per_area.mean() / model.mean())) < 0.06
    # Scatterer by scatterer the variance is exactly the model's: nothing but the draw is random.
    unit = np.abs(scene.amplitudes) ** 2 * density
    assert np.allclose(per_area, unit * model, rtol=1e-9)


def test_model_angles_come_from_the_ping_and_from_the_normals():
    scene = seabed(20.0)
    here, there = np.array([0.0, 0.0, -5.0]), np.array([30.0, 0.0, -2.0])
    a_here = scene.evaluate(0.0, here, here, frequency=F, sound_speed=C)[1]
    a_there = scene.evaluate(0.0, there, there, frequency=F, sound_speed=C)[1]
    assert not np.allclose(np.abs(a_here), np.abs(a_there))        # per ping, not fixed at build time
    assert np.array_equal(scene.evaluate(0.0, here, here, frequency=F, sound_speed=C)[1], a_here)
    with pytest.raises(ValueError, match="frequency"):
        scene.evaluate(0.0, here, here)
    # Projector and receiver apart: the model's bistatic strength, with its azimuth, and reciprocal.
    mod = scene.modifiers[0][2]
    one = np.array([[20.0, 0.0, 0.0]])
    tx, rx = np.array([0.0, 0.0, -5.0]), np.array([20.0, 15.0, -8.0])
    ctx = targets.PingContext(0.0, tx, rx, F, C, one, np.ones(1, dtype=complex), np.zeros((1, 3)), np.full((1, 3), np.nan))
    sigma, sin_in, sin_out = mod.cross_section(ctx)
    g_in, g_out = np.arcsin(5.0 / np.hypot(20.0, 5.0)), np.arcsin(8.0 / np.hypot(15.0, 8.0))
    assert sin_in[0] == pytest.approx(np.sin(g_in)) and sin_out[0] == pytest.approx(np.sin(g_out))
    # Incident sound travels along +x, scattered sound along +y: 90 degrees between them.
    assert sigma[0] == pytest.approx(float(bottom.strength(grazing_in=g_in, grazing_out=g_out, frequency=F, sound_speed=C,
                                                           azimuth=np.pi / 2, **SAND)), rel=1e-9)
    swapped = targets.PingContext(0.0, rx, tx, F, C, one, np.ones(1, dtype=complex), np.zeros((1, 3)), np.full((1, 3), np.nan))
    assert mod.cross_section(swapped)[0][0] == pytest.approx(sigma[0], rel=1e-9)
    # offset_db moves the strength, as for a reverberation boundary.
    louder = seabed(20.0, offset_db=6.0)
    assert np.allclose(louder.evaluate(0.0, here, here, frequency=F, sound_speed=C)[1], a_here * 10 ** 0.3)
    # A sloping floor: the grazing angle is taken from each scatterer's normal.
    slope = np.tan(np.radians(10.0))
    tilted = build_scene([dict({"type": "seafloor", "x_range": [19.9, 20.1], "y_range": [-0.1, 0.1], "density": 500.0,
                                "model": "fluid_sediment", "slope": [-slope, 0.0]}, **SAND)], np.random.default_rng(1))
    pts, amp, _ = tilted.evaluate(0.0, here, here, frequency=F, sound_speed=C)
    local = np.arcsin(-(((pts - here) / np.linalg.norm(pts - here, axis=1, keepdims=True)) * tilted.normals).sum(axis=1))
    # The floor has risen 20 tan(10 deg) = 3.5 m there, so the sonar is 1.5 m above it; the tilt adds 10 degrees.
    assert np.degrees(local).mean() == pytest.approx(np.degrees(np.arctan2(5.0 - 20.0 * slope, 20.0)) + 10.0, abs=0.1)
    _, _, a, _, _ = two_way_paths(FreeFieldProvider(C), here, here, pts, amp, lambert=tilted.lambert, normals=tilted.normals)
    want = 10.0 ** (bottom.backscattering_strength_db(local, F, C, **SAND) / 10.0)
    assert np.allclose(np.abs(a) ** 2 * 500.0, np.abs(tilted.amplitudes) ** 2 * 500.0 * want, rtol=1e-9)


def test_lambert_seafloor_is_the_default_and_unchanged():
    item = {"type": "seafloor", "x_range": [2.0, 12.0], "y_range": [-3.0, 3.0], "density": 30.0, "lambert_mu_db": -27.0}
    scene = build_scene([item], np.random.default_rng(7))
    same = build_scene([dict(item, model="lambert")], np.random.default_rng(7))
    assert not scene.modifiers and not same.modifiers
    assert np.array_equal(scene.positions, same.positions) and np.array_equal(scene.amplitudes, same.amplitudes)
    # The amplitudes are what they were before the option existed: sqrt(mu / density) x a unit draw
    # from the same random stream, whatever ping asks.
    rng = np.random.default_rng(7)
    rng.uniform(size=2 * len(scene))
    draw = (rng.standard_normal(len(scene)) + 1j * rng.standard_normal(len(scene))) / np.sqrt(2.0)
    assert np.allclose(scene.amplitudes, np.sqrt(10 ** -2.7 / 30.0) * draw)
    sonar = np.array([0.0, 0.0, -3.0])
    pts, amp, vel = scene.evaluate(3.0, sonar, sonar, frequency=F)
    assert amp is scene.amplitudes and vel is scene.velocities and np.array_equal(pts, scene.positions)
    # The model's seabed uses the same positions and the same draws; only the strength differs.
    model = build_scene([dict({k: v for k, v in item.items() if k != "lambert_mu_db"}, model="fluid_sediment", **SAND)],
                        np.random.default_rng(7))
    assert np.array_equal(model.positions, scene.positions)
    assert np.allclose(model.amplitudes * np.sqrt(10 ** -2.7), scene.amplitudes)
    with pytest.raises(ValueError, match="not lambert_mu_db"):
        build_scene([dict(item, model="fluid_sediment", **SAND)], np.random.default_rng(7))


def test_scatterer_seabed_agrees_with_scattering_function_reverberation():
    """The same sediment as a field of scatterers in the scene and as a `reverberation` boundary:
    two realisations of one model, so their mean levels against time agree (as the Lambert
    pair does in test_reverberation.py, with the same 2 dB allowance for the speckle of a few pings)."""
    from test_reverberation import HEIGHT, empty_baseband_ping
    sonar, _ = empty_baseband_ping()
    pose = make_pose(position=(0.0, 0.0, -HEIGHT))
    point, sf = [], []
    for seed in (11, 12):
        scene = build_scene([dict({"type": "seafloor", "x_range": [-260.0, 260.0], "y_range": [-260.0, 260.0],
                                   "density": 1.5, "model": "fluid_sediment"}, **SAND)], np.random.default_rng(seed))
        point.append(simulate_ping(sonar, pose, scene, environment={"sound_speed": C}, absorption_model="none",
                                   adc=False).samples.array)
    for seed in (21, 22):
        _, rec = empty_baseband_ping()
        fc = rec.samples.center_frequency
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, 0.0, np.random.default_rng(seed),
                                 strength=bottom.reverberation_strength(frequency=fc, sound_speed=C, **SAND))
        sf.append(rec.samples.array)
    a = (np.abs(np.concatenate(point, axis=0)) ** 2).mean(axis=0)
    b = (np.abs(np.concatenate(sf, axis=0)) ** 2).mean(axis=0)
    blocks = slice(300, 3900)
    ratio = 10 * np.log10(a[blocks].reshape(12, -1).mean(axis=1) / b[blocks].reshape(12, -1).mean(axis=1))
    assert np.abs(ratio).max() < 2.0, ratio
    assert abs(ratio.mean()) < 1.0, ratio


def test_sediment_seafloor_takes_a_roughness_model_and_defaults_to_small_slope():
    """D-86: the seabed scatterers use the model's default (small_slope); `roughness_model:
    perturbation` gives the amplitudes of before, at any angle the field has."""
    from gpusonar.sim import bottom
    item = dict({"type": "seafloor", "x_range": [0.2, 6.0], "y_range": [-2.0, 2.0], "density": 20.0, "model": "fluid_sediment"}, **SAND)
    sonar = np.array([0.0, 0.0, -3.0])                                  # grazing from 27 to 86 degrees

    def amplitudes(**extra):
        scene = build_scene([dict(item, **extra)], np.random.default_rng(4))
        pts, amp, _ = scene.evaluate(0.0, sonar, sonar, frequency=F, sound_speed=C)
        return pts, amp
    pts, default = amplitudes()
    _, named = amplitudes(roughness_model="small_slope")
    _, old = amplitudes(roughness_model="perturbation")
    assert np.array_equal(default, named) and not np.array_equal(default, old)
    d = pts - sonar
    g = np.arcsin(np.abs(d[:, 2]) / np.linalg.norm(d, axis=1))
    for amp, model in ((default, "small_slope"), (old, "perturbation")):
        want = bottom.strength(g, g, F, C, roughness_model=model, **SAND)
        ratio = np.abs(amp) ** 2 / np.abs(old) ** 2
        assert np.allclose(ratio, want / bottom.strength(g, g, F, C, roughness_model="perturbation", **SAND), rtol=1e-6)
    with pytest.raises(ValueError, match="roughness_model"):
        amplitudes(roughness_model="lambert")
