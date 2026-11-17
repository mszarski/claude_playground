"""Time spreading of boundary-bounce paths (sim/timespread.py), after SST TR 0702 Sec. 12.4."""
import numpy as np
import pytest

from gpusonar import acoustics
from gpusonar.sim import ImageSourceProvider, Scene, make_pose, simulate_ping
from gpusonar.sim.timespread import TimeSpreadProvider, boundary_parameters, wrap
from gpusonar.sonar import SonarModel

C, F = 1500.0, 8000.0
SOURCE = np.array([0.0, 0.0, 30.0])


def one_bounce(depth_point=30.0, rng_m=600.0, **kw):
    """Surface only, order 1: a direct ray and one surface-bounce ray to a point."""
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1, surface_coefficient=-1.0)
    pts = np.array([[rng_m, 0.0, depth_point]])
    return inner, pts, kw


def spread(inner, surface, **kw):
    return TimeSpreadProvider(inner, frequency=F, surface=surface, surface_z=0.0, **kw)


def test_flat_boundary_and_direct_ray_are_untouched():
    inner, pts, _ = one_bounce()
    base = inner.eigenrays(SOURCE, pts)
    same = spread(inner, None).eigenrays(SOURCE, pts)
    assert same.delay.shape == base.delay.shape and np.array_equal(same.amplitude, base.amplitude)
    rough = spread(inner, {"rms_height": 0.05, "mean_square_slope": 0.03, "rms_vertical_speed": 0.2}).eigenrays(SOURCE, pts)
    direct = base.n_surface[0] == 0
    assert np.array_equal(rough.amplitude[0, :2][direct], base.amplitude[0][direct])
    assert np.array_equal(rough.delay[0, :2], base.delay[0])
    # Taps exist only for the ray that bounced.
    taps_valid = rough.valid[0, 2:].reshape(2, -1)
    assert not taps_valid[direct].any() and taps_valid[~direct].all()


def test_coherent_part_is_the_rayleigh_parameter_factor():
    inner, pts, _ = one_bounce()
    base = inner.eigenrays(SOURCE, pts)
    h = 0.15
    out = spread(inner, {"rms_height": h, "mean_square_slope": 0.03, "rms_vertical_speed": 0.0}).eigenrays(SOURCE, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    grazing = np.arcsin(abs(base.launch[0, j, 2]))
    gamma = 2.0 * (2.0 * np.pi * F / C) * h * np.sin(grazing)
    assert 0.3 < gamma < 3.0                                            # neither flat nor fully rough
    assert abs(out.amplitude[0, j]) / abs(base.amplitude[0, j]) == pytest.approx(np.exp(-0.5 * gamma**2), rel=1e-12)


def test_energy_is_conserved_on_average():
    """Coherent plus incoherent intensity equals the flat-boundary intensity, over realisations."""
    inner, pts, _ = one_bounce()
    base = inner.eigenrays(SOURCE, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    total = []
    for seed in range(400):
        out = spread(inner, {"rms_height": 0.2, "mean_square_slope": 0.03, "rms_vertical_speed": 0.0}, seed=seed).eigenrays(SOURCE, pts)
        taps = out.amplitude[0, 2:].reshape(2, -1)[j]
        total.append(abs(out.amplitude[0, j]) ** 2 + np.sum(np.abs(taps) ** 2))
    assert np.mean(total) == pytest.approx(abs(base.amplitude[0, j]) ** 2, rel=0.06)
    assert np.std(total) > 0.1 * np.mean(total)                         # and it does fluctuate


@pytest.mark.parametrize("point_range, source_depth", [(600.0, 30.0), (600.0, 3.0)])
def test_decay_time_follows_the_reports_rule_of_thumb(point_range, source_depth):
    """Mid-path bounce: mss x half the one-way time. Bounce near one end: mss x twice the short leg."""
    inner, pts, _ = one_bounce(depth_point=30.0, rng_m=point_range)
    src = np.array([0.0, 0.0, source_depth])
    mss = 0.04
    base = inner.eigenrays(src, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    out = spread(inner, {"rms_height": 0.05, "mean_square_slope": mss, "rms_vertical_speed": 0.0}, taps=12).eigenrays(src, pts)
    taps_a = np.abs(out.amplitude[0, 2:].reshape(2, -1)[j]) ** 2
    taps_t = out.delay[0, 2:].reshape(2, -1)[j] - base.delay[0, j]
    assert np.all(np.diff(taps_t) > 0) and taps_t[0] > 0
    t_total = base.delay[0, j]
    t_short = t_total * source_depth / (source_depth + 30.0)            # the leg before the surface
    expected = 2.0 * mss * t_short * (t_total - t_short) / t_total
    if source_depth == 30.0:
        assert expected == pytest.approx(mss * 0.5 * t_total, rel=1e-9)             # the report's mid-path rule
    else:
        assert expected == pytest.approx(mss * 2.0 * t_short, rel=0.1)              # its near-end rule
    # The taps' mean intensities are the model's exactly; one realisation's are random about them.
    means = []
    for seed in range(300):
        o = spread(inner, {"rms_height": 0.05, "mean_square_slope": mss, "rms_vertical_speed": 0.0}, taps=12, seed=seed).eigenrays(src, pts)
        means.append(np.abs(o.amplitude[0, 2:].reshape(2, -1)[j]) ** 2)
    w = np.mean(means, axis=0)
    assert np.sum(w * taps_t) / np.sum(w) == pytest.approx(expected, rel=0.08)
    assert w[0] > w[3] > w[8]                                           # decaying


def test_amplitudes_decorrelate_at_the_doppler_rate_and_are_frozen_without_motion():
    inner, pts, _ = one_bounce()
    base = inner.eigenrays(SOURCE, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    grazing = np.arcsin(abs(base.launch[0, j, 2]))
    v = 0.3
    width = 2.0 * F * v * np.sin(grazing) / C                           # Hz
    lag = 1.0 / (2.0 * np.pi * width)                                   # correlation exp(-1) here

    def taps_at(t, seed, speed):
        p = spread(inner, {"rms_height": 0.05, "mean_square_slope": 0.03, "rms_vertical_speed": speed}, seed=seed)
        p.set_ping(t)
        return p.eigenrays(SOURCE, pts).amplitude[0, 2:].reshape(2, -1)[j]

    a = np.array([taps_at(0.0, s, v) for s in range(300)])
    b = np.array([taps_at(lag, s, v) for s in range(300)])
    far = np.array([taps_at(40.0 * lag, s, v) for s in range(300)])
    rho = lambda x, y: abs(np.sum(x * np.conj(y))) / np.sqrt(np.sum(abs(x) ** 2) * np.sum(abs(y) ** 2))   # noqa: E731
    assert rho(a, b) == pytest.approx(np.exp(-1.0), abs=0.08)
    assert rho(a, far) < 0.12
    assert np.array_equal(taps_at(0.0, 1, 0.0), taps_at(100.0, 1, 0.0))  # nothing moves: nothing changes
    # A moving sonar adds (speed x sqrt(mss)) to the boundary's rms speed.
    p = spread(inner, {"rms_height": 0.05, "mean_square_slope": 0.03, "rms_vertical_speed": 0.0}, seed=1)
    p.set_ping(0.0, platform_speed=2.0)
    first = p.eigenrays(SOURCE, pts).amplitude[0, 2:]
    p.set_ping(50.0, platform_speed=2.0)
    assert not np.allclose(first, p.eigenrays(SOURCE, pts).amplitude[0, 2:])


def test_nearby_scatterers_fade_together_and_distant_ones_do_not():
    inner, _, _ = one_bounce()
    pts = np.array([[600.0, 0.0, 30.0], [600.5, 0.3, 30.0], [900.0, 200.0, 30.0]])
    par = {"rms_height": 0.2, "mean_square_slope": 0.03, "rms_vertical_speed": 0.0}      # fully rough: incoherent only
    ratios_near, ratios_far = [], []
    for seed in range(200):
        out = spread(inner, par, seed=seed, coherence_length=10.0, taps=1).eigenrays(SOURCE, pts)
        tap = out.amplitude[:, 2:].reshape(3, 2, 1)[:, 1, 0]            # the bounce ray's one tap, per point
        ratios_near.append(tap[1] / tap[0])
        ratios_far.append(tap[2] / tap[0])
    assert np.std(np.abs(ratios_near)) < 1e-3                           # the same realisation
    assert np.std(np.angle(ratios_far)) > 1.0                           # independent


def test_wind_speed_gives_the_surface_statistics():
    p = boundary_parameters({"wind_speed": 8.0}, surface=True)
    assert p["rms_height"] == pytest.approx(0.25 * 0.21 * 64.0 / 9.81)
    assert p["mean_square_slope"] == pytest.approx(0.003 + 0.00512 * 8.0)
    # Second moment of the Pierson-Moskowitz spectrum, by quadrature.
    w = np.linspace(0.05, 60.0, 400_000)
    spec = 0.0081 * 9.81**2 * w**-5 * np.exp(-0.74 * (9.81 / (8.0 * w)) ** 4)
    assert p["rms_vertical_speed"] == pytest.approx(np.sqrt(np.trapezoid(w**2 * spec, w)), rel=2e-3)
    # The height uses the conventional rounded 0.21 U^2 / g; the spectrum's own zeroth moment is 0.4% lower.
    assert acoustics.rms_wave_height(8.0) == pytest.approx(np.sqrt(np.trapezoid(spec, w)), rel=1e-2)
    assert boundary_parameters({"rms_height": 0.0}, surface=False) is None
    with pytest.raises(ValueError):
        boundary_parameters({"wind_speed": 5.0}, surface=False)


def test_bottom_and_surface_bounces_each_use_their_own_boundary():
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=80.0, max_order=2)
    pts = np.array([[500.0, 0.0, 40.0]])
    base = inner.eigenrays(SOURCE, pts)
    only_bottom = TimeSpreadProvider(inner, frequency=F, bottom={"rms_height": 0.03, "mean_square_slope": 0.045, "rms_vertical_speed": 0.0},
                                     surface_z=0.0, bottom_z=80.0).eigenrays(SOURCE, pts)
    k = base.delay.shape[1]
    ratio = np.abs(only_bottom.amplitude[0, :k]) / np.abs(base.amplitude[0])
    assert np.allclose(ratio[base.n_bottom[0] == 0], 1.0)               # surface-only and direct rays: as before
    assert np.all(ratio[base.n_bottom[0] > 0] < 1.0)
    two = base.n_bottom[0] == 2
    one = (base.n_bottom[0] == 1) & (base.n_surface[0] == 0)
    if two.any() and one.any():
        assert ratio[two].max() < ratio[one].max()                      # two bounces lose more coherence than one


def test_a_ping_with_time_spreading_changes_only_the_bounce_arrival():
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [550.0, 700.0]}}})
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1, surface_coefficient=-1.0)
    scene = Scene().add_points([[0.0, 600.0, 60.0]], target_strength_db=10.0)
    pose = make_pose(position=(0.0, 0.0, 60.0))
    kw = dict(environment={"sound_speed": C}, adc=False, noise_rms=0.0)
    flat = simulate_ping(sonar, pose, scene, provider=inner, **kw).samples.array
    prov = wrap(inner, {"surface": {"wind_speed": 6.0}, "seed": 3}, frequency=sonar.center_frequency)
    rough = simulate_ping(sonar, pose, scene, provider=prov, scenario_time=0.0, **kw).samples.array
    later = simulate_ping(sonar, pose, scene, provider=prov, scenario_time=30.0, **kw).samples.array
    fs = sonar.config["receive"]["sample_rate"]
    start = 2.0 * 550.0 / C
    direct_end = int((2.0 * 600.0 / C - start + sonar.waveform_config["duration"]) * fs)
    first_bounce = int((600.0 + np.hypot(600.0, 120.0)) / C * fs - start * fs)
    assert first_bounce > direct_end - int(sonar.waveform_config["duration"] * fs)
    head = slice(0, first_bounce - 2)
    assert np.allclose(rough[:, head], flat[:, head], atol=1e-9 * np.abs(flat).max())      # the direct echo alone
    tail = slice(first_bounce, None)
    assert not np.allclose(rough[:, tail], flat[:, tail], atol=1e-3 * np.abs(flat).max())  # the bounce arrivals differ
    assert not np.allclose(rough[:, tail], later[:, tail], atol=1e-3 * np.abs(flat).max()) # and change with time
    assert np.allclose(rough[:, head], later[:, head], atol=1e-9 * np.abs(flat).max())


def test_bubble_loss_removes_energy_from_both_parts_of_a_surface_bounce():
    inner, pts, _ = one_bounce()
    base = inner.eigenrays(SOURCE, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    par = {"rms_height": 0.2, "mean_square_slope": 0.03, "rms_vertical_speed": 0.0}
    dry = TimeSpreadProvider(inner, frequency=F, surface=par, seed=5).eigenrays(SOURCE, pts)
    wet = TimeSpreadProvider(inner, frequency=F, surface=boundary_parameters(dict(par, bubble_loss_db=3.0), surface=True), seed=5).eigenrays(SOURCE, pts)
    assert np.allclose(np.abs(wet.amplitude[0, 2:]) ** 2, np.abs(dry.amplitude[0, 2:]) ** 2 * 10 ** -0.3)     # the tail
    assert abs(wet.amplitude[0, j]) ** 2 == pytest.approx(abs(dry.amplitude[0, j]) ** 2 * 10 ** -0.3)        # the coherent part
    assert wet.amplitude[0, 1 - j] == dry.amplitude[0, 1 - j]                                                  # the direct ray
    table = boundary_parameters({"bubble_loss_db": {"grazing_deg": [0.0, 90.0], "loss_db": [6.0, 0.0]}}, surface=True)
    only = TimeSpreadProvider(inner, frequency=F, surface=table).eigenrays(SOURCE, pts)                        # bubbles, flat surface
    grazing = np.degrees(np.arcsin(abs(base.launch[0, j, 2])))
    assert 20 * np.log10(abs(only.amplitude[0, j]) / abs(base.amplitude[0, j])) == pytest.approx(-6.0 * (1 - grazing / 90.0), abs=1e-9)


def test_the_sst_style_scenario_runs_and_its_bounce_arrivals_fade():
    from gpusonar.sim import run_scenario
    from gpusonar.config import load_scenario_config
    cfg = load_scenario_config("flank_sst_shallow")
    cfg["pings"]["count"] = 2
    quiet = dict(cfg, reverberation=None, receiver={"adc": False})
    a, b = (r.samples.array for r in run_scenario(quiet).records)
    assert np.isfinite(a).all() and np.abs(a).max() > 0
    assert not np.allclose(a, b, atol=1e-3 * np.abs(a).max())            # the surface changed between pings
    full = run_scenario(cfg).records[0].samples.array
    assert np.abs(full - a).max() > 0                                    # reverberation and noise were added
    again = run_scenario(quiet).records[0].samples.array
    assert np.array_equal(again, a)                                      # the seed reproduces the ping
