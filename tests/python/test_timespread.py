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
    """Mid-path bounce: mss x half the one-way time. Bounce near one end: mss x twice the short leg.
    (`decay_time: rule_of_thumb`, the default until D-85; named here since `facet` became it.)"""
    inner, pts, _ = one_bounce(depth_point=30.0, rng_m=point_range)
    src = np.array([0.0, 0.0, source_depth])
    mss = 0.04
    base = inner.eigenrays(src, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    out = spread(inner, {"rms_height": 0.05, "mean_square_slope": mss, "rms_vertical_speed": 0.0}, taps=12,
                 decay_time="rule_of_thumb").eigenrays(src, pts)
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
        o = spread(inner, {"rms_height": 0.05, "mean_square_slope": mss, "rms_vertical_speed": 0.0}, taps=12, seed=seed,
                   decay_time="rule_of_thumb").eigenrays(src, pts)
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

    def taps_at(t, seed, speed, spectrum="lorentzian"):
        p = spread(inner, {"rms_height": 0.05, "mean_square_slope": 0.03, "rms_vertical_speed": speed}, seed=seed,
                   doppler_spectrum=spectrum)
        p.set_ping(t)
        return p.eigenrays(SOURCE, pts).amplitude[0, 2:].reshape(2, -1)[j]

    a = np.array([taps_at(0.0, s, v) for s in range(300)])
    b = np.array([taps_at(lag, s, v) for s in range(300)])
    far = np.array([taps_at(40.0 * lag, s, v) for s in range(300)])
    rho = lambda x, y: abs(np.sum(x * np.conj(y))) / np.sqrt(np.sum(abs(x) ** 2) * np.sum(abs(y) ** 2))   # noqa: E731
    assert rho(a, b) == pytest.approx(np.exp(-1.0), abs=0.08)
    assert rho(a, far) < 0.12
    # The default (Gaussian) spectrum has rms width `width / sqrt(1 - exp(-Gamma^2))`, so the
    # correlation at lag t is exp(-(2 pi sigma t)^2 / 2): exp(-1/2) at 1 / (2 pi sigma).
    gamma2 = (2.0 * (2.0 * np.pi * F / C) * 0.05 * np.sin(grazing)) ** 2
    lag_g = np.sqrt(1.0 - np.exp(-gamma2)) / (2.0 * np.pi * width)
    ag = np.array([taps_at(0.0, s, v, "gaussian") for s in range(300)])
    bg = np.array([taps_at(lag_g, s, v, "gaussian") for s in range(300)])
    assert rho(ag, bg) == pytest.approx(np.exp(-0.5), abs=0.08)
    default = spread(inner, {"rms_height": 0.05, "mean_square_slope": 0.03, "rms_vertical_speed": v}, seed=3)
    assert default.doppler_spectrum == "gaussian"
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


# ---- Angle and frequency spreading, convolved bounces (the options beyond SST) ----------------

ROUGH = {"rms_height": 0.3, "mean_square_slope": 4e-4, "rms_vertical_speed": 0.3}     # Gamma >> 1: no coherent part


def bounce_taps(e, base, j, taps, lines=1):
    """The taps of ray j of the one field point: (amplitude, delay, launch, arrival), [J(, M)]."""
    k = base.delay.shape[1]
    shape = (k, taps) if lines == 1 else (k, taps, lines)
    part = lambda a: a[0, k:].reshape(shape + a.shape[2:])[j]          # noqa: E731
    return part(e.amplitude), part(e.delay), part(e.launch), part(e.arrival)


def steep_bounce():
    """A surface bounce at 39 degrees grazing, 3/8 of the way along its path."""
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1)
    pts = np.array([[100.0, 0.0, 50.0]])
    base = inner.eigenrays(SOURCE, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    return inner, pts, base, j, np.arcsin(abs(base.launch[0, j, 2])), 30.0 / 80.0


def test_the_options_off_give_the_output_of_the_version_before_them():
    """Sums over the eigenrays of a two-boundary case, from sim/timespread.py as it was before
    angle spreading, within-ping Doppler, the facet decay time and convolved bounces."""
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=80.0, max_order=2)
    pts = np.array([[500.0, 40.0, 40.0], [300.0, -20.0, 10.0]])
    kw = dict(frequency=F, surface=boundary_parameters({"wind_speed": 6.0}, surface=True),
              bottom={"rms_height": 0.03, "mean_square_slope": 0.02, "rms_vertical_speed": 0.0},
              surface_z=0.0, bottom_z=80.0, taps=4, seed=5, multi_bounce="sum", doppler_spectrum="lorentzian",
              decay_time="rule_of_thumb")
    p = TimeSpreadProvider(inner, **kw)
    p.set_ping(7.0, 1.5)
    e = p.eigenrays(SOURCE, pts)
    a = e.amplitude[e.valid]
    assert e.delay.shape == (2, 25) and int(e.valid.sum()) == 42 and e.range_rate is None
    assert complex(a.sum()) == pytest.approx(0.0071019667758292085 + 8.887691937657543e-05j, rel=1e-9)
    assert float(np.abs(a).sum()) == pytest.approx(0.028625828421519003, rel=1e-9)
    assert float(e.delay[e.valid].sum()) == pytest.approx(12.002601952374846, rel=1e-12)
    # Naming the defaults changes nothing, and the taps keep the specular directions.
    q = TimeSpreadProvider(inner, angle_spread=False, doppler=False, doppler_lines=8, **kw)
    q.set_ping(7.0, 1.5)
    same = q.eigenrays(SOURCE, pts)
    assert np.array_equal(same.amplitude, e.amplitude) and np.array_equal(same.delay, e.delay)
    base = inner.eigenrays(SOURCE, pts)
    k = base.delay.shape[1]
    assert np.array_equal(e.launch[:, k:].reshape(2, k, 4, 3), np.repeat(base.launch[:, :, None], 4, axis=2))


def test_a_taps_direction_is_that_of_the_facet_with_its_delay():
    """Exact geometry: follow each tap's launch direction to the surface. The path through that
    point is longer than the specular one by the tap's excess delay, and the tap arrives from
    that point. (Second-order geometry: the error is of the order of the turn over the grazing angle.)"""
    inner, pts, base, j, grazing, _ = steep_bounce()
    for mss, tolerance in ((4e-4, 0.11), (2.5e-5, 0.03)):
        par = dict(ROUGH, mean_square_slope=mss)
        for seed in range(20):
            e = spread(inner, par, taps=8, seed=seed, angle_spread=True).eigenrays(SOURCE, pts)
            off = spread(inner, par, taps=8, seed=seed).eigenrays(SOURCE, pts)
            assert np.array_equal(e.amplitude, off.amplitude) and np.array_equal(e.delay, off.delay)
            assert np.array_equal(e.launch[0, :2], base.launch[0]) and np.array_equal(e.arrival[0, :2], base.arrival[0])   # the coherent rays
            _, delay, launch, arrival = bounce_taps(e, base, j, 8)
            facet = SOURCE + (-SOURCE[2] / launch[:, 2])[:, None] * launch
            to_point = pts[0] - facet
            excess = (np.linalg.norm(facet - SOURCE, axis=1) + np.linalg.norm(to_point, axis=1)) / C - base.delay[0, j]
            assert np.allclose(excess, delay - base.delay[0, j], rtol=tolerance)
            exact = to_point / np.linalg.norm(to_point, axis=1)[:, None]
            turn = np.linalg.norm(arrival - base.arrival[0, j], axis=1)
            assert np.all(np.linalg.norm(arrival - exact, axis=1) < tolerance * turn)
            assert np.allclose(np.linalg.norm(launch, axis=1), 1.0) and np.all(turn > 0)


def angle_statistics(decay_time, n_seeds=300, taps=24):
    """Intensity-weighted mean-square turn of the taps at the source, in the vertical plane and across it."""
    inner, pts, base, j, grazing, f = steep_bounce()
    u = base.launch[0, j]
    in_plane = np.array([-u[2] * u[0], -u[2] * u[1], u[0] ** 2 + u[1] ** 2])
    in_plane /= np.linalg.norm(in_plane)
    sums = np.zeros(3)
    for seed in range(n_seeds):
        e = spread(inner, ROUGH, taps=taps, seed=seed, angle_spread=True, decay_time=decay_time).eigenrays(SOURCE, pts)
        amp, _, launch, _ = bounce_taps(e, base, j, taps)
        d = launch - u
        sums += [np.sum(abs(amp) ** 2 * (d @ in_plane) ** 2), np.sum(abs(amp) ** 2 * d[:, 1] ** 2), np.sum(abs(amp) ** 2)]
    return sums[0] / sums[2], sums[1] / sums[2], grazing, f


def test_rms_angular_spread_matches_the_closed_forms():
    mss = ROUGH["mean_square_slope"]
    # TR 0702's decay time: T = 2 mss t1 t2 / (t1 + t2), so <angle^2> = 2 c T r2 / (r1 (r1 + r2)) = 4 mss (r2 / (r1 + r2))^2.
    v2, h2, grazing, f = angle_statistics("rule_of_thumb")
    assert np.sqrt(v2 + h2) == pytest.approx(2.0 * np.sqrt(mss) * (1.0 - f), rel=0.05)
    # Facet decay time: Gaussian in each plane, 2 mss (r2 / (r1 + r2))^2 in the vertical plane and sin^2(g) of it across.
    v2, h2, grazing, f = angle_statistics("facet")
    assert np.sqrt(v2) == pytest.approx(np.sqrt(2.0 * mss) * (1.0 - f), rel=0.06)
    assert np.sqrt(h2) == pytest.approx(np.sqrt(2.0 * mss) * np.sin(grazing) * (1.0 - f), rel=0.06)
    assert h2 < v2                                                       # the glitter is longer along the path


def test_facet_decay_time_and_shape():
    """T = mss (1 + sin^2 g) t1 t2 / (t1 + t2), and the shape of a sum of two squared Gaussians
    (checked by drawing slopes and using tau = 2 (sx^2 + sy^2 sin^2 g) t1 t2 / (t1 + t2))."""
    from gpusonar.sim.timespread import facet_delay_density
    inner, pts, base, j, grazing, f = steep_bounce()
    mss, taps, lines = ROUGH["mean_square_slope"], 30, 2
    e = spread(inner, ROUGH, taps=taps, decay_time="facet", doppler=True, doppler_lines=lines).eigenrays(SOURCE, pts)
    amp, delay, _, _ = bounce_taps(e, base, j, taps, lines)
    w = (abs(amp) ** 2).sum(axis=1)                                     # with `doppler` the lines' magnitudes are not random
    tau = delay[:, 0] - base.delay[0, j]
    lever = f * (1.0 - f) * base.delay[0, j]
    expected = mss * (1.0 + np.sin(grazing) ** 2) * lever
    assert np.sum(w * tau) / np.sum(w) == pytest.approx(expected, rel=1e-9)
    default = spread(inner, ROUGH, taps=taps, doppler=True, doppler_lines=lines)                       # the default is this option (D-85)
    same = default.eigenrays(SOURCE, pts)
    assert default.decay_time == "facet" and np.array_equal(same.amplitude, e.amplitude) and np.array_equal(same.delay, e.delay)
    rule = spread(inner, ROUGH, taps=taps, decay_time="rule_of_thumb", doppler=True, doppler_lines=lines).eigenrays(SOURCE, pts)
    w_rule = (abs(bounce_taps(rule, base, j, taps, lines)[0]) ** 2).sum(axis=1)
    tau_rule = bounce_taps(rule, base, j, taps, lines)[1][:, 0] - base.delay[0, j]
    assert np.sum(w_rule * tau_rule) / np.sum(w_rule) == pytest.approx(2.0 * mss * lever, rel=1e-9)
    assert np.sum(w) == pytest.approx(np.sum(w_rule), rel=1e-12)        # the same energy
    rng = np.random.default_rng(1)
    sx, sy = rng.normal(0.0, np.sqrt(mss / 2.0), (2, 400_000))
    drawn = 2.0 * (sx**2 + sy**2 * np.sin(grazing) ** 2) * lever
    assert drawn.mean() == pytest.approx(expected, rel=0.01)
    hist, edges = np.histogram(drawn, bins=np.linspace(0.0, 3.0 * expected, 31), density=True)
    mid = 0.5 * (edges[1:] + edges[:-1])
    density = facet_delay_density(mid, mss * lever, mss * lever * np.sin(grazing) ** 2)
    assert np.allclose(hist[1:] * np.mean(drawn < 3.0 * expected), density[1:], rtol=0.06)
    assert facet_delay_density(0.3, 0.5, 0.5) == pytest.approx(np.exp(-0.3))            # equal axes: the exponential
    # The taps sample that density on a grid of three mean delays (their delays are then scaled to give the mean).
    on_grid = facet_delay_density(3.0 * expected * np.arange(1, taps + 1) / taps, mss * lever, mss * lever * np.sin(grazing) ** 2)
    assert np.allclose(w / w.sum(), on_grid / on_grid.sum(), rtol=1e-9)


def test_array_coherence_of_the_tail_drops_as_the_angular_spread_predicts():
    """Two hydrophones d apart see the tail with coherence exp(-d^2 / 2 sigma_d^2),
    sigma_d = lambda / (2 pi sigma_S), sigma_S the rms spread of the direction cosine along
    the line joining them (TR 0702 eqs. 100-103). Vertical aperture: sigma_S = sigma_v cos(g);
    horizontal, across the path: sigma_S = sigma_v sin(g)."""
    inner, pts, base, j, grazing, f = steep_bounce()
    mss, taps = ROUGH["mean_square_slope"], 24
    k = 2.0 * np.pi * F / C
    sigma_v = np.sqrt(2.0 * mss) * (1.0 - f)
    sigma = {"vertical": sigma_v * np.cos(grazing), "horizontal": sigma_v * np.sin(grazing)}
    axis = {"vertical": np.array([0.0, 0.0, 1.0]), "horizontal": np.array([0.0, 1.0, 0.0])}
    fields = {name: [] for name in sigma}
    fixed = {name: [] for name in sigma}
    for seed in range(400):
        e = spread(inner, ROUGH, taps=taps, seed=seed, angle_spread=True, decay_time="facet").eigenrays(SOURCE, pts)
        o = spread(inner, ROUGH, taps=taps, seed=seed, decay_time="facet").eigenrays(SOURCE, pts)
        amp, _, launch, _ = bounce_taps(e, base, j, taps)
        _, _, launch0, _ = bounce_taps(o, base, j, taps)
        for name in sigma:
            sigma_d = 1.0 / (k * sigma[name])                           # lambda / (2 pi sigma_S)
            row, row0 = [], []
            for d in (0.5 * sigma_d, sigma_d, 1.5 * sigma_d):
                # A plane wave from direction `launch` at the two hydrophones, +-d/2 along the axis.
                # The taps' gains are independent with zero mean, so the cross terms between taps
                # average to nothing and are left out: y+ conj(y-) -> sum |a|^2 exp(i k d s).
                row.append((np.sum(abs(amp) ** 2 * np.exp(1j * k * d * (launch @ axis[name]))), np.sum(abs(amp) ** 2)))
                phase0 = 0.5 * k * d * (launch0 @ axis[name])
                row0.append((np.sum(amp * np.exp(1j * phase0)), np.sum(amp * np.exp(-1j * phase0))))
            fields[name].append(row)
            fixed[name].append(row0)
    for name in sigma:
        y = np.array(fields[name])                                      # [seed, distance, (cross, power)]
        coherence = np.abs(np.sum(y[..., 0], axis=0)) / np.sum(y[..., 1], axis=0).real
        # (Beyond about 1.5 sigma_d the taps' cut at three mean delays shows: the coherence falls faster than the Gaussian.)
        assert coherence == pytest.approx(np.exp(-0.5 * np.array([0.25, 1.0, 2.25])), abs=0.08), name
        y0 = np.array(fixed[name])                                      # without angle spreading: one direction, every realisation
        same = np.abs(y0[..., 0] * np.conj(y0[..., 1])) / abs(y0[..., 0]) ** 2
        assert same == pytest.approx(1.0, abs=1e-9), name
    assert sigma["horizontal"] < sigma["vertical"]                      # so the coherence length across the path is the longer


def test_without_the_doppler_option_the_tail_is_frozen_during_a_ping():
    """What the process gives by default: one complex gain per tap per ping. It changes from
    ping to ping (tested above) and not within one: no ray has a range rate, so a tone
    through a bounce comes back as a tone. With `doppler` the sinusoids the gain is made of
    become rays of their own, and their sum at the ping time is that gain."""
    inner, pts, base, j, _, _ = steep_bounce()
    for spectrum in ("lorentzian", "gaussian"):
        frozen = spread(inner, ROUGH, taps=5, seed=2, doppler_spectrum=spectrum)
        moving = spread(inner, ROUGH, taps=5, seed=2, doppler_spectrum=spectrum, doppler=True)
        for p in (frozen, moving):
            p.set_ping(3.7)
        a, b = frozen.eigenrays(SOURCE, pts), moving.eigenrays(SOURCE, pts)
        assert a.range_rate is None and b.range_rate.shape == b.delay.shape == (1, 2 + 2 * 5 * 8)
        assert np.all(b.range_rate[0, :2] == 0.0) and np.any(b.range_rate[0, 2:] != 0.0)
        assert np.allclose(bounce_taps(b, base, j, 5, 8)[0].sum(axis=1), bounce_taps(a, base, j, 5)[0], rtol=1e-12)
        # Over a time t the lines turn by 2 pi nu t: the gain at a later ping is the lines' sum then.
        later = spread(inner, ROUGH, taps=5, seed=2, doppler_spectrum=spectrum)
        later.set_ping(3.7 + 0.8)
        amp = bounce_taps(b, base, j, 5, 8)[0]
        nu = -F * b.range_rate[0, 2:].reshape(2, 5, 8)[j] / C
        assert np.allclose((amp * np.exp(2j * np.pi * nu * 0.8)).sum(axis=1), bounce_taps(later.eigenrays(SOURCE, pts), base, j, 5)[0], rtol=1e-6)


@pytest.mark.parametrize("height", [0.3, 0.012])
def test_doppler_width_of_a_tone_through_one_surface_bounce(height):
    """rms width sigma_nu = 2 f v sin(g) / c / sqrt(1 - exp(-Gamma^2)); the tone's coherence
    falls as exp(-t^2 / 2 sigma_t^2), sigma_t = 1 / (2 pi sigma_nu) (TR 0702 eqs. 95-96 with
    sigma_gamma = sigma_nu / f). Mean power is that of the frozen tail."""
    inner, pts, base, j, grazing, _ = steep_bounce()
    v = 0.3
    par = dict(ROUGH, rms_height=height, rms_vertical_speed=v)
    gamma2 = (2.0 * (2.0 * np.pi * F / C) * height * np.sin(grazing)) ** 2
    assert gamma2 > 20.0 if height == 0.3 else 0.1 < gamma2 < 1.0
    sigma_nu = 2.0 * F * v * np.sin(grazing) / C / np.sqrt(1.0 - np.exp(-gamma2))
    assert sigma_nu < 10.0                                              # TR 0702, p. 98: "less than 10 Hz" at 8 kHz
    taps, lines = 4, 8
    t = np.array([0.0, 0.5, 1.0, 2.0]) / (2.0 * np.pi * sigma_nu)       # lags in units of sigma_t
    nus, powers, tones, frozen_power = [], [], [], []
    for seed in range(500):
        e = spread(inner, par, taps=taps, seed=seed, doppler=True, doppler_spectrum="gaussian", doppler_lines=lines).eigenrays(SOURCE, pts)
        amp = bounce_taps(e, base, j, taps, lines)[0].reshape(-1)
        nu = -F * e.range_rate[0, 2:].reshape(2, -1)[j] / C
        nus.append(nu)
        powers.append(abs(amp) ** 2)
        tones.append((amp[:, None] * np.exp(2j * np.pi * nu[:, None] * t)).sum(axis=0))    # the received tone at the lags
        o = spread(inner, par, taps=taps, seed=seed, doppler_spectrum="gaussian", doppler_lines=lines).eigenrays(SOURCE, pts)
        frozen_power.append(np.sum(abs(bounce_taps(o, base, j, taps)[0]) ** 2))
    nus, powers, tones = np.array(nus), np.array(powers), np.array(tones)
    assert np.sqrt(np.sum(powers * nus**2) / np.sum(powers)) == pytest.approx(sigma_nu, rel=0.03)
    assert abs(np.sum(powers * nus) / np.sum(powers)) < 0.05 * sigma_nu                   # no mean shift
    coherence = np.abs(np.mean(tones * np.conj(tones[:, :1]), axis=0)) / np.mean(abs(tones[:, 0]) ** 2)
    assert coherence == pytest.approx(np.exp(-0.5 * np.array([0.0, 0.25, 1.0, 4.0])), abs=0.06)
    incoherent = abs(base.amplitude[0, j]) ** 2 * (1.0 - np.exp(-gamma2))
    assert powers.sum(axis=1) == pytest.approx(incoherent, rel=1e-9)                     # every realisation: the lines' powers add up
    assert np.mean(abs(tones) ** 2, axis=0) == pytest.approx(incoherent, rel=0.1)        # and the tone's mean power at any time
    assert np.mean(frozen_power) == pytest.approx(incoherent, rel=0.1)


def test_no_boundary_motion_no_doppler_and_a_bottom_spreads_only_with_the_sonars_motion():
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=80.0, max_order=1)
    pts = np.array([[100.0, 0.0, 50.0]])
    still = dict(ROUGH, rms_vertical_speed=0.0)
    kw = dict(frequency=F, surface_z=0.0, bottom_z=80.0, taps=3, doppler=True, doppler_spectrum="gaussian")
    calm = TimeSpreadProvider(inner, surface=still, bottom=still, **kw)
    assert not np.any(calm.eigenrays(SOURCE, pts).range_rate)           # zero wind speed (no vertical speed): no spread
    assert spread(inner, None, doppler=True).eigenrays(SOURCE, pts).range_rate is None   # flat: the rays as they were
    windy = TimeSpreadProvider(inner, surface=ROUGH, bottom=still, **kw).eigenrays(SOURCE, pts)
    base = inner.eigenrays(SOURCE, pts)
    rate = windy.range_rate[0, 3:].reshape(3, 3, 8)
    assert np.any(rate[base.n_surface[0] == 1] != 0.0) and not np.any(rate[base.n_bottom[0] == 1])
    under_way = TimeSpreadProvider(inner, surface=None, bottom=still, **kw)
    under_way.set_ping(0.0, platform_speed=2.0)
    rate = under_way.eigenrays(SOURCE, pts).range_rate[0, 3:].reshape(3, 3, 8)[base.n_bottom[0] == 1]
    grazing = np.arcsin(abs(base.launch[0, base.n_bottom[0] == 1, 2]))[0]
    expected = 2.0 * 2.0 * np.sqrt(still["mean_square_slope"]) * np.sin(grazing)          # 2 (V sqrt(mss)) sin g, as a range rate
    assert np.sqrt(np.mean(rate**2)) == pytest.approx(expected, rel=0.5)                   # 24 draws
    # With angle spreading the sonar's motion acts through the taps' directions instead.
    both = TimeSpreadProvider(inner, surface=None, bottom=still, angle_spread=True, **kw)
    both.set_ping(0.0, platform_speed=2.0)
    assert not np.any(both.eigenrays(SOURCE, pts).range_rate)


def two_rough_bounces(**kw):
    """Source and point at mid-depth, surface and bottom both rough: the two rays with one
    bounce from each boundary. Returns the taps' mean intensities and excess delays per ray."""
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=60.0, max_order=2, bottom_coefficient=1.0)
    src, pts = np.array([0.0, 0.0, 30.0]), np.array([[400.0, 0.0, 30.0]])
    par = {"rms_height": 0.3, "mean_square_slope": 0.01, "rms_vertical_speed": 0.0}
    base = inner.eigenrays(src, pts)
    two = np.flatnonzero((base.n_surface[0] == 1) & (base.n_bottom[0] == 1))
    one = np.flatnonzero(base.n_surface[0] + base.n_bottom[0] == 1)
    # The closed forms these tests assert (exponential tails of mean 2 mss t1 t2 / (t1 + t2), their
    # sum and their convolution) are those of TR 0702's rule, so the rule is named: it was the
    # default when they were written (`facet` is since D-85).
    kw.setdefault("decay_time", "rule_of_thumb")
    p = TimeSpreadProvider(inner, frequency=F, surface=kw.pop("surface", par), bottom=kw.pop("bottom", par), surface_z=0.0, bottom_z=60.0,
                           doppler=True, doppler_lines=2, **kw)
    e = p.eigenrays(src, pts)
    k, taps = base.delay.shape[1], p.taps
    weight = (abs(e.amplitude[0, k:].reshape(k, taps, 2)) ** 2).sum(axis=2)               # mean tap intensities, exactly
    tau = e.delay[0, k:].reshape(k, taps, 2)[:, :, 0] - base.delay[0][:, None]
    return base, e, weight, tau, two, one


def test_convolved_bounces_start_low_and_last_longer():
    base, _, w_sum, tau_sum, two, one = two_rough_bounces(taps=40, multi_bounce="sum")
    _, _, w_con, tau_con, _, _ = two_rough_bounces(taps=40, multi_bounce="convolve")
    assert len(two) == 2 and len(one) == 2
    assert np.array_equal(w_sum[one], w_con[one]) and np.array_equal(tau_sum[one], tau_con[one])     # one bounce: nothing to convolve
    r = int(two[0])
    t_total = base.delay[0, r]
    t_one = 2.0 * 0.01 * (0.25 * t_total) * (0.75 * t_total) / t_total                     # each bounce: a quarter of the way from one end
    mean = lambda w, tau: np.sum(w * tau) / np.sum(w)                                      # noqa: E731
    rms = lambda w, tau: np.sqrt(np.sum(w * tau**2) / np.sum(w) - mean(w, tau) ** 2)       # noqa: E731
    assert mean(w_sum[r], tau_sum[r]) == pytest.approx(2.0 * t_one, rel=1e-9)
    assert mean(w_con[r], tau_con[r]) == pytest.approx(2.0 * t_one, rel=1e-9)
    assert np.sum(w_con[r]) == pytest.approx(np.sum(w_sum[r]), rel=1e-12)                  # the same energy
    # One exponential of mean 2T (what TR 0702's author calls too high at first and too low later)
    # against the convolution of two of mean T, tau exp(-tau / T) / T^2.
    assert np.argmax(w_sum[r]) == 0 and np.argmax(w_con[r]) > 0
    erlang = tau_con[r] * np.exp(-tau_con[r] / t_one)
    assert np.allclose(w_con[r] / w_con[r].sum(), erlang / erlang.sum(), atol=0.004)
    early = lambda w, tau: w[tau < 0.3 * t_one].sum() / w.sum()                            # noqa: E731
    assert early(w_con[r], tau_con[r]) < 0.5 * early(w_sum[r], tau_sum[r])
    # Spread about the mean: sqrt(2) T for the convolution, 2 T for the single exponential (both cut at three means).
    assert rms(w_con[r], tau_con[r]) < 0.85 * rms(w_sum[r], tau_sum[r])


def test_convolved_bounces_with_partly_coherent_boundaries():
    """Tail energy 1 - c1 c2 of the total and mean delay ((1 - c1) T1 + (1 - c2) T2) / (1 - c1 c2):
    a bounce delays the tail only when it scatters."""
    smooth = {"rms_height": 0.02, "mean_square_slope": 0.01, "rms_vertical_speed": 0.0}
    rough = {"rms_height": 0.05, "mean_square_slope": 0.03, "rms_vertical_speed": 0.0}
    base, e, w, tau, two, _ = two_rough_bounces(taps=30, multi_bounce="convolve", surface=rough, bottom=smooth)
    for r in two:
        g = np.arcsin(abs(base.launch[0, r, 2]))
        gam = lambda h: np.exp(-(2.0 * (2.0 * np.pi * F / C) * h * np.sin(g)) ** 2)        # noqa: E731
        c_s, c_b = gam(0.05), gam(0.02)
        assert 0.05 < c_s < c_b < 0.95
        lever = 0.25 * 0.75 * base.delay[0, r]
        t_s, t_b = 2.0 * 0.03 * lever, 2.0 * 0.01 * lever
        total = abs(base.amplitude[0, r]) ** 2
        assert np.sum(w[r]) == pytest.approx(total * (1.0 - c_s * c_b), rel=1e-9)
        assert abs(e.amplitude[0, r]) ** 2 == pytest.approx(total * c_s * c_b, rel=1e-9)
        assert np.sum(w[r] * tau[r]) / np.sum(w[r]) == pytest.approx(((1 - c_s) * t_s + (1 - c_b) * t_b) / (1 - c_s * c_b), rel=1e-9)


@pytest.mark.parametrize("mode", ["sum", "convolve"])
def test_with_two_rough_bounces_the_first_turns_the_launch_and_the_last_the_arrival(mode):
    """Each end's mean-square turn is that of the bounce nearest to it with its share of the
    excess delay: 2 c (T / 2 of 2 T) r_far / (r_near (r_near + r_far)) = 4 mss (3/4)^2 here."""
    mss = 1e-3
    par = {"rms_height": 0.3, "mean_square_slope": mss, "rms_vertical_speed": 0.0}
    base, e, w, _, two, one = two_rough_bounces(taps=12, multi_bounce=mode, angle_spread=True, surface=par, bottom=par)
    k = base.delay.shape[1]
    launch = e.launch[0, k:].reshape(k, 12, 2, 3)[:, :, 0]
    arrival = e.arrival[0, k:].reshape(k, 12, 2, 3)[:, :, 0]
    for r in two:
        for end, specular in ((launch, base.launch), (arrival, base.arrival)):
            turn2 = np.sum((end[r] - specular[0, r]) ** 2, axis=1)
            assert np.sum(w[r] * turn2) / np.sum(w[r]) == pytest.approx(4.0 * mss * 0.75**2, rel=0.02)
    r = int(one[0])                                                      # one bounce, half way: both ends 4 mss (1/2)^2
    assert np.sum(w[r] * np.sum((launch[r] - base.launch[0, r]) ** 2, axis=1)) / np.sum(w[r]) == pytest.approx(mss, rel=0.02)


def test_the_scenario_with_every_option_runs_on_both_engines():
    import warnings
    from gpusonar.sim import run_scenario
    from gpusonar.config import load_scenario_config
    cfg = load_scenario_config("flank_sst_spread")
    cfg["pings"]["count"] = 1
    assert all(cfg["time_spread"][key] for key in ("angle_spread", "doppler")) and cfg["time_spread"]["multi_bounce"] == "convolve"
    cfg["time_spread"] = dict(cfg["time_spread"], taps=2, doppler_lines=2)                 # (a quicker run)
    quiet = dict(cfg, reverberation=None, receiver={"adc": False}, sonar_overrides={"receive": {"window": {"range": [990, 1090]}}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # the C++ pairing carries the range rates: no fallback
        a = run_scenario(quiet).records[0].samples.array
        b = run_scenario(quiet, engine="numpy").records[0].samples.array
    assert np.isfinite(a).all() and np.abs(a).max() > 0
    # The paths of the two pairings are equal to rounding (test below), and one kernel
    # synthesises both, so the records differ by float32 rounding of a few samples at most.
    assert np.allclose(a, b, atol=1e-6 * np.abs(a).max())
    print(f"flank_sst_spread, C++ vs NumPy pairing: worst sample {np.abs(a - b).max() / np.abs(a).max():.1e} of the peak")
    # The Doppler lines are in the record: without the range rates it differs.
    # Angle spreading alone: the C++ pairing carries the directions.
    quiet["time_spread"] = dict(cfg["time_spread"], doppler=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = run_scenario(quiet).records[0].samples.array
    d = run_scenario(quiet, engine="numpy").records[0].samples.array
    assert np.allclose(c, d, atol=1e-6 * np.abs(c).max()) and not np.allclose(c, a, atol=1e-3 * np.abs(a).max())


def test_spread_rays_through_the_pairing_and_both_engines():
    """Angle spreading rides in the eigenrays' directions: the C++ and NumPy pairings give the
    same paths. A range rate becomes the velocity of the path's apparent position along the
    ray, on both engines (the C++ pairing carries it per ray; there is no fallback)."""
    import warnings
    from gpusonar.sim import propagation
    from gpusonar.sim.propagation import two_way_paths
    inner, pts, base, j, _, _ = steep_bounce()
    tx, rx = SOURCE, SOURCE + np.array([0.0, 0.0, 0.5])
    amp = np.array([1.0 + 0.0j])
    angled = spread(inner, ROUGH, taps=3, seed=4, angle_spread=True)
    a, b = (two_way_paths(angled, tx, rx, pts, amp, engine=engine) for engine in ("cpp", "numpy"))
    assert a[0].shape == ((1 + 1 + 3) ** 2, 3) and a[3] is None
    for x, y in zip(a[:3], b[:3]):
        assert np.allclose(x, y, rtol=1e-12, atol=1e-12)
    e = angled.eigenrays(tx, pts)
    keep = e.valid[0]
    q = tx + C * e.delay[0][keep][:, None] * e.launch[0][keep]           # apparent positions: along each tap's own direction
    assert np.allclose(a[0][::5], q, atol=1e-9)
    moving = spread(inner, ROUGH, taps=3, seed=4, doppler=True, doppler_lines=2)
    assert not hasattr(propagation, "_warned_doppler_fallback")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q, p, w, vq, vp = two_way_paths(moving, tx, rx, pts, amp, engine="cpp")
    for x, y in zip((q, p, w, vq, vp), two_way_paths(moving, tx, rx, pts, amp, engine="numpy")):
        assert x.shape == y.shape and np.allclose(x, y, rtol=1e-13, atol=1e-13)
    e, back = moving.eigenrays(tx, pts), moving.eigenrays(rx, pts)
    keep = e.valid[0]
    n = int(keep.sum())
    assert n == 1 + 1 + 3 * 2 and len(q) == n * n and vq.shape == vp.shape == (n * n, 3)
    assert np.allclose(vq.reshape(n, n, 3), (e.range_rate[0][keep][:, None] * e.launch[0][keep])[:, None, :])
    assert np.allclose(vp.reshape(n, n, 3), (back.range_rate[0][keep][:, None] * back.launch[0][keep])[None, :, :])
    assert two_way_paths(spread(inner, ROUGH, taps=3, seed=4), tx, rx, pts, amp, engine="numpy")[3] is None


def test_a_cw_ping_shows_the_lines_of_the_doppler_spread_tail():
    """End to end through the synthesis kernel: a 0.5 s tone, one surface bounce with one tap of
    one line each way (projector and array in different coherence cells, so the two differ).
    The direct echo stays at the carrier; the echoes with one bounce are shifted by the
    frequency of the ray that bounced and the echo with a bounce each way by the sum."""
    f0 = 8000.0
    sonar = SonarModel.load("flank_32", {"transmit": {"waveform": {"type": "cw", "f_start": f0, "f_stop": f0, "duration": 0.5, "window_alpha": 0.0}},
                                         "receive": {"window": {"range": [140.0, 600.0]}}})
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1)
    scene = Scene().add_points([[3.0, 154.0, 50.0]], target_strength_db=10.0)
    pose = make_pose(position=(3.0, 4.0, 32.0))
    par = dict(ROUGH, rms_vertical_speed=2.5)                            # faster than any sea: lines several Hz apart
    kw = dict(environment={"sound_speed": C}, adc=False, noise_rms=0.0)
    flat = simulate_ping(sonar, pose, scene, provider=inner, **kw)
    tx, rx = np.array(flat.tx_position()), flat.rx_positions().mean(axis=0)
    assert np.linalg.norm(tx - rx) > 0.4
    for seed in range(200):                                              # a realisation whose four lines stand apart
        prov = TimeSpreadProvider(inner, frequency=f0, surface=par, taps=1, seed=seed, coherence_length=0.25, doppler=True,
                                  doppler_spectrum="gaussian", doppler_lines=1)
        nu_out, nu_back = (-f0 * e.range_rate[0, 2:][e.valid[0, 2:]][0] / C for e in (prov.eigenrays(tx, scene.positions), prov.eigenrays(rx, scene.positions)))
        lines = np.array([0.0, nu_out, nu_back, nu_out + nu_back])
        if np.min(np.diff(np.sort(lines))) > 8.0:
            break
    else:
        raise AssertionError("no realisation with separated lines")
    rec = simulate_ping(sonar, pose, scene, provider=prov, **kw)       # (the default engine: the pairing falls back to NumPy)
    x = rec.samples.array[16]
    n = 1 << 21
    spectrum = abs(np.fft.rfft(x * np.hanning(len(x)), n))
    freq = np.fft.rfftfreq(n, 1.0 / rec.samples.sample_rate)
    near = (freq > f0 - 150.0) & (freq < f0 + 150.0)
    found = []
    work = np.where(near, spectrum, 0.0)
    for _ in range(4):                                                   # the four strongest lines
        i = int(np.argmax(work))
        found.append(freq[i])
        work[abs(freq - freq[i]) < 4.0] = 0.0
    assert np.sort(found) == pytest.approx(np.sort(f0 + lines), abs=0.3)
    assert work.max() < 0.3 * spectrum[near].max()                       # and nothing else of that size
    # A point in free water returns the tone alone.
    y = simulate_ping(sonar, pose, scene, **kw).samples.array[16]
    tone = abs(np.fft.rfft(y * np.hanning(len(y)), n))
    inside = abs(freq - f0) < 5.0
    assert np.sum(tone[near & inside] ** 2) > 0.9 * np.sum(tone[near] ** 2)
    assert np.sum(spectrum[near & inside] ** 2) < 0.6 * np.sum(spectrum[near] ** 2)


def _random_rays(rng, s, k, c=1500.0):
    """An eigenray set with nothing special about it: some rays invalid, some rates zero."""
    from gpusonar.sim.propagation import EigenraySet
    d = rng.normal(size=(s, k, 3))
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    a = rng.normal(size=(s, k, 3))
    a /= np.linalg.norm(a, axis=2, keepdims=True)
    delay = rng.uniform(0.2, 0.9, (s, k))
    return EigenraySet(delay=delay, amplitude=(rng.normal(size=(s, k)) + 1j * rng.normal(size=(s, k))) / (c * delay),
                       launch=d, arrival=a, n_surface=np.zeros((s, k), int), n_bottom=np.zeros((s, k), int),
                       valid=rng.uniform(size=(s, k)) > 0.2, sound_speed=c,
                       range_rate=np.where(rng.uniform(size=(s, k)) > 0.3, rng.normal(0.0, 0.4, (s, k)), 0.0))


class _Fixed:
    """A provider that hands back prepared eigenray sets (transmitter first, then receiver)."""

    def __init__(self, out, back):
        self.sets = [out, back]

    def eigenrays(self, source, points):
        return self.sets.pop(0)


@pytest.mark.parametrize("case", ["both legs", "out only", "back only", "moving scatterers", "lambert and normals"])
def test_cpp_pairing_carries_per_ray_range_rates_like_numpy(case):
    """The C++ pairing with per-ray range rates against the NumPy branch, on random rays
    (different ray counts on the two legs, invalid rays, zero rates): each pair's ends get
    v_q = v_scatterer + rate_out * launch_out and v_p = v_scatterer + rate_back * launch_back.
    The same arithmetic in the same order, so equality is to the last bit or two."""
    from gpusonar.sim.propagation import two_way_paths
    rng = np.random.default_rng(11)
    s, ko, kb = 7, 5, 3
    out, back = _random_rays(rng, s, ko), _random_rays(rng, s, kb)
    amp = rng.normal(size=s) + 1j * rng.normal(size=s)
    amp[2] = 0.0                                                         # a scatterer that returns nothing
    kw = {}
    if case == "out only":
        back.range_rate = None
    if case == "back only":
        out.range_rate = np.zeros((s, ko))                               # all zero counts as none
    if case in ("moving scatterers", "lambert and normals"):
        kw["velocities"] = rng.normal(0.0, 2.0, (s, 3))
    if case == "lambert and normals":
        kw["lambert"] = rng.uniform(size=s) > 0.5
        n = rng.normal(size=(s, 3))
        n[::2] = np.nan
        kw["normals"] = n
    tx, rx = np.array([0.0, 0.0, 30.0]), np.array([0.5, 0.0, 30.0])
    pts = rng.normal(size=(s, 3)) * 100.0
    copies = lambda: _Fixed(*(type(e)(**{f: getattr(e, f) for f in e.__dataclass_fields__}) for e in (out, back)))
    a = two_way_paths(copies(), tx, rx, pts, amp, engine="cpp", **kw)
    b = two_way_paths(copies(), tx, rx, pts, amp, engine="numpy", **kw)
    assert len(a) == len(b) == 5 and len(a[0]) > 20
    for x, y in zip(a, b):
        assert x.shape == y.shape
        np.testing.assert_allclose(x, y, rtol=1e-14, atol=1e-14)
    q, p, w, vq, vp = a
    # Independent of both: rebuild the velocities of the kept pairs from the definition.
    v = kw.get("velocities", np.zeros((s, 3)))
    w_all = amp[:, None, None] * (out.amplitude * 1500.0 * out.delay)[:, :, None] * (back.amplitude * 1500.0 * back.delay)[:, None, :]
    keep = out.valid[:, :, None] & back.valid[:, None, :] & (w_all != 0)
    if case == "lambert and normals":
        keep = np.zeros_like(keep)                                       # (weights can zero a pair; skip the rebuild)
    else:
        r_out = np.zeros((s, ko)) if out.range_rate is None else out.range_rate
        r_back = np.zeros((s, kb)) if back.range_rate is None else back.range_rate
        vq_all = v[:, None, None, :] + (r_out[..., None] * out.launch)[:, :, None, :] + np.zeros((s, ko, kb, 3))
        vp_all = v[:, None, None, :] + (r_back[..., None] * back.launch)[:, None, :, :] + np.zeros((s, ko, kb, 3))
        np.testing.assert_allclose(vq, vq_all[keep], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(vp, vp_all[keep], rtol=1e-14, atol=1e-14)
        assert not np.allclose(vq, vp)


def test_per_ray_doppler_reaches_the_record_through_the_cpp_pairing():
    """End to end on the C++ engine alone: a long tone through a moving surface comes back as
    lines at f0 - f0 (rate_out + rate_back) / c for the pairs of rays, which are in the
    paths' two velocities. Zeroing the rates (same rays, same amplitudes) leaves one line."""
    import warnings
    from gpusonar.sim.propagation import two_way_paths
    from gpusonar import core
    f0, c, dur = 8000.0, 1500.0, 0.5
    sonar = SonarModel.load("flank_32", {
        "transmit": {"waveform": {"type": "cw", "f_start": f0, "f_stop": f0, "duration": dur, "window_alpha": 1.0},
                     "lever_arm": [0.0, 0.0, 0.0], "beamwidth_az": None, "beamwidth_el": None,
                     "aperture_width": None, "aperture_height": None},
        "array": {"count": 1, "beamwidth_az": None, "beamwidth_el": None, "element_width": None, "element_height": None},
        "fov": None, "receive": {"window": {"range": [900.0, 1500.0]}, "kind": "baseband"}})
    rng = np.random.default_rng(3)
    out = _random_rays(rng, 1, 2)
    out.launch[0] = [[0.0, 1.0, 0.0], [0.0, 0.8, -0.6]]
    out.delay[0], out.valid[0] = [0.70, 0.72], True
    out.amplitude[0] = 1.0 / (c * out.delay[0])
    out.range_rate = np.array([[0.0, 0.9]])                              # 4.8 Hz on the bounce ray
    back = type(out)(**{f: np.copy(getattr(out, f)) if f != "sound_speed" else c for f in out.__dataclass_fields__})
    pose = make_pose()
    rec0 = sonar.make_record(pose, environment={"sound_speed": c})
    tx, rx = np.array(rec0.tx_position()), rec0.rx_positions().mean(axis=0)
    spectra = {}
    for label, rate in (("moving", out.range_rate), ("still", None)):
        o, b = (type(out)(**{f: getattr(e, f) for f in e.__dataclass_fields__}) for e in (out, back))
        o.range_rate = b.range_rate = rate
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            q, p, w, vq, vp = two_way_paths(_Fixed(o, b), tx, rx, np.zeros((1, 3)), np.array([1.0 + 0j]), engine="cpp")
        assert (vq is None) == (rate is None)
        rec = sonar.make_record(pose, environment={"sound_speed": c})
        core.synthesize_into(rec, q, p, w, vq, vp, spreading=False, arrival_passes=(8, 8))
        s = rec.samples
        x = s.array[0]
        n = 1 << 20
        f = np.fft.fftfreq(n, 1.0 / s.sample_rate) + s.center_frequency - f0
        spectrum = np.abs(np.fft.fft(x, n))
        near = np.abs(f) < 15.0
        order = np.argsort(f[near])
        spectra[label] = (f[near][order], spectrum[near][order])
    f, still = spectra["still"]
    assert abs(f[np.argmax(still)]) < 0.3
    f, moving = spectra["moving"]
    shift = -f0 * 0.9 / c                                               # one leg on the bounce ray: -4.8 Hz
    # Hann-weighted 0.5 s tone: lines 4 Hz wide between nulls, at 0, shift (two pairs) and 2 shift.
    peaks = [f[i] for i in range(1, len(f) - 1) if moving[i] > moving[i - 1] and moving[i] >= moving[i + 1]
             and moving[i] > 0.3 * moving.max()]
    assert len(peaks) == 3
    assert np.allclose(peaks, [2.0 * shift, shift, 0.0], atol=0.3)


def _cuda():
    from gpusonar import core
    return core.cuda_available()


@pytest.mark.skipif(not _cuda(), reason="no CUDA device")
def test_per_ray_doppler_on_the_cuda_kernel_matches_the_host():
    """The Doppler lines ride in each path's two velocities, which the device kernel reads
    from the same path struct as the host kernel: flank_sst_spread (every spreading option
    on) gives the same record on both, to the order of the float32 additions."""
    from gpusonar import core
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    cfg = load_scenario_config("flank_sst_spread")
    cfg["pings"]["count"] = 1
    cfg["time_spread"] = dict(cfg["time_spread"], taps=2, doppler_lines=2)
    quiet = dict(cfg, reverberation=None, receiver={"adc": False}, sonar_overrides={"receive": {"window": {"range": [990, 1090]}}})
    host = run_scenario(dict(quiet), backend=core.Backend.Host).records[0].samples.array
    cuda = run_scenario(dict(quiet), backend=core.Backend.Cuda).records[0].samples.array
    assert np.abs(host).max() > 0
    assert np.allclose(cuda, host, atol=2e-6 * np.abs(host).max())
    still = dict(quiet, time_spread=dict(quiet["time_spread"], doppler=False))
    flat = run_scenario(still, backend=core.Backend.Cuda).records[0].samples.array
    assert not np.allclose(cuda, flat, atol=1e-3 * np.abs(host).max())      # the range rates are in the device record


def test_the_default_decay_time_is_the_facet_one_also_through_a_scenario_section():
    """D-85: `decay_time` defaults to `facet`. One bounce, default options: the taps' mean excess
    delay is mss (1 + sin^2 g) t1 t2 / (t1 + t2), not TR 0702's 2 mss t1 t2 / (t1 + t2); the
    scenario section without the key gives the same provider, and `rule_of_thumb` still gives
    the rule."""
    inner, pts, base, j, grazing, f = steep_bounce()
    mss, taps = ROUGH["mean_square_slope"], 30
    lever = f * (1.0 - f) * base.delay[0, j]

    def mean_excess(n_seeds=200, **kw):
        w = np.zeros(taps)
        for seed in range(n_seeds):
            amp, delay, _, _ = bounce_taps(spread(inner, ROUGH, taps=taps, seed=seed, **kw).eigenrays(SOURCE, pts), base, j, taps)
            w += abs(amp) ** 2
        return np.sum(w * (delay - base.delay[0, j])) / np.sum(w)

    default = spread(inner, ROUGH, taps=taps)
    assert default.decay_time == "facet"
    assert mean_excess() == pytest.approx(mss * (1.0 + np.sin(grazing) ** 2) * lever, rel=0.05)
    assert mean_excess(decay_time="rule_of_thumb") == pytest.approx(2.0 * mss * lever, rel=0.05)
    section = {"surface": dict(ROUGH), "taps": taps}
    wrapped = wrap(inner, section, frequency=F)
    assert wrapped.decay_time == "facet"
    assert wrap(inner, dict(section, decay_time="rule_of_thumb"), frequency=F).decay_time == "rule_of_thumb"
    a, b = wrapped.eigenrays(SOURCE, pts), default.eigenrays(SOURCE, pts)
    assert np.array_equal(a.delay, b.delay) and np.array_equal(a.amplitude, b.amplitude)


# -- level fluctuations beyond Rayleigh, and the surface height apart from the wind -------------------

def _one_bounce(**options):
    from gpusonar.sim import ImageSourceProvider
    from gpusonar.sim.timespread import wrap
    inner = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=None, max_order=1)
    cfg = dict({"surface": {"wind_speed": 6.0}, "taps": 4, "coherence_length": 1e-3, "seed": 3, "doppler_lines": 64}, **options)
    return wrap(inner, cfg, frequency=12500.0)


def _line_of_points(n, range_m=60.0):
    p = np.zeros((n, 3))
    p[:, 0], p[:, 1], p[:, 2] = range_m, np.arange(n) * 2e-3, 11.0        # a new cell each, the same geometry to 2 %
    return p


def test_tap_shape_absent_or_infinite_changes_nothing():
    pts = _line_of_points(50)
    base = _one_bounce().eigenrays([0.0, 0.0, 11.0], pts)
    for shape in (None, float("inf")):
        same = _one_bounce(tap_shape=shape).eigenrays([0.0, 0.0, 11.0], pts)
        assert np.array_equal(same.amplitude, base.amplitude) and np.array_equal(same.delay, base.delay)


@pytest.mark.parametrize("shape", [0.7, 2.0, 6.0])
def test_tap_shape_gives_k_distributed_tap_intensity(shape):
    """The texture is a gamma variate of mean 1 and variance 1 / nu, common to a ray's taps: the
    mean tap intensity is unchanged, its second moment is (1 + 1/nu) times the Gaussian one, and
    the scintillation index is (1 + 1/nu)(2 - 1/M) - 1 for taps made of M sinusoids (1 + 2/nu as
    M grows). The direct path and the coherent arrival are untouched."""
    pts = _line_of_points(6000)
    plain = _one_bounce().eigenrays([0.0, 0.0, 11.0], pts)
    tex = _one_bounce(tap_shape=shape).eigenrays([0.0, 0.0, 11.0], pts)
    n_base = 2                                                # direct, surface (coherent part); then 4 taps per ray
    assert np.array_equal(tex.amplitude[:, :n_base], plain.amplitude[:, :n_base])
    i0, i1 = np.abs(plain.amplitude[:, -4:]) ** 2, np.abs(tex.amplitude[:, -4:]) ** 2          # the surface ray's four taps
    tau = i1 / i0
    assert np.allclose(tau, tau[:, :1], rtol=1e-9)             # one texture per ray
    tau = tau[:, 0]
    assert tau.mean() == pytest.approx(1.0, abs=4.0 / np.sqrt(shape * len(tau)))
    assert tau.var() == pytest.approx(1.0 / shape, rel=0.12)
    assert np.mean(tau**2) == pytest.approx(1.0 + 1.0 / shape, rel=0.08)
    si = lambda x: x.var(axis=0) / x.mean(axis=0) ** 2  # noqa: E731
    assert si(i0).mean() == pytest.approx(1.0 - 1.0 / 64, abs=0.06)
    assert si(i1).mean() == pytest.approx((1.0 + 1.0 / shape) * (2.0 - 1.0 / 64) - 1.0, rel=0.12)
    assert i1.mean() == pytest.approx(i0.mean(), rel=0.06)


def test_tap_texture_changes_slowly_in_time_and_follows_a_grazing_table():
    pts = _line_of_points(3000)
    plain = _one_bounce()

    def tau_at(t, p):
        p.set_ping(t)
        plain.set_ping(t)
        return (np.abs(p.eigenrays([0.0, 0.0, 11.0], pts).amplitude[:, -1]) / np.abs(plain.eigenrays([0.0, 0.0, 11.0], pts).amplitude[:, -1])) ** 2

    prov = _one_bounce(tap_shape=1.0, texture_rate=0.25)
    g = np.arctan2(22.0, 60.0)
    width = 2.0 * 12500.0 * acoustics.rms_surface_vertical_speed(6.0) * np.sin(g) / 1500.0      # Hz; the texture's is a quarter of it
    t0 = tau_at(0.0, prov)
    assert np.corrcoef(t0, tau_at(0.02 / width, prov))[0, 1] > 0.95
    assert abs(np.corrcoef(t0, tau_at(40.0 / width, prov))[0, 1]) < 0.15
    # A table: Gaussian where the table says so (here the bounce is at 20 degrees), textured where steep.
    steep = _one_bounce(tap_shape={"grazing_deg": [0.0, 10.0, 20.1], "shape": [1e6, 1e6, 0.5]})
    assert tau_at(0.0, steep).var() == pytest.approx(2.0, rel=0.35)
    gentle = _one_bounce(tap_shape={"grazing_deg": [0.0, 30.0, 60.0], "shape": [1e6, 1e6, 0.5]})
    assert tau_at(0.0, gentle).var() < 1e-4
    with pytest.raises(ValueError, match="tap_shape"):
        _one_bounce(tap_shape=-1.0)


def test_rms_height_apart_from_the_wind_and_height_scale():
    """`rms_height` beside `wind_speed` replaces the wind law's height and leaves its slope and
    vertical speed; `height_scale` multiplies the height in force. The coherent reflection
    follows: exp(-Gamma^2 / 2) in amplitude with Gamma = 2 k h sin(grazing)."""
    from gpusonar.sim.timespread import boundary_parameters
    wind = boundary_parameters({"wind_speed": 4.0}, surface=True)
    given = boundary_parameters({"wind_speed": 4.0, "rms_height": 0.12}, surface=True)
    scaled = boundary_parameters({"wind_speed": 4.0, "height_scale": 1.4}, surface=True)
    both = boundary_parameters({"wind_speed": 4.0, "rms_height": 0.1, "height_scale": 0.5}, surface=True)
    assert wind["rms_height"] == pytest.approx(acoustics.rms_wave_height(4.0))
    assert given["rms_height"] == 0.12 and scaled["rms_height"] == pytest.approx(1.4 * wind["rms_height"]) and both["rms_height"] == pytest.approx(0.05)
    for other in (given, scaled, both):
        assert other["mean_square_slope"] == wind["mean_square_slope"] and other["rms_vertical_speed"] == wind["rms_vertical_speed"]
    assert boundary_parameters({"wind_speed": 4.0, "height_scale": 1.0}, surface=True) == wind
    with pytest.raises(ValueError, match="height_scale"):
        boundary_parameters({"wind_speed": 4.0, "height_scale": -1.0}, surface=True)
    pts = _line_of_points(1, range_m=400.0)
    g = np.arctan2(22.0, 400.0)
    flat = _one_bounce(surface=None).eigenrays([0.0, 0.0, 11.0], pts)
    for cfg, h in (({"wind_speed": 4.0}, wind["rms_height"]), ({"wind_speed": 4.0, "height_scale": 1.4}, 1.4 * wind["rms_height"]),
                   ({"wind_speed": 4.0, "rms_height": 0.03}, 0.03)):
        rays = _one_bounce(surface=cfg).eigenrays([0.0, 0.0, 11.0], pts)
        gamma = 2.0 * (2.0 * np.pi * 12500.0 / 1500.0) * h * np.sin(g)
        assert abs(rays.amplitude[0, 1]) / abs(flat.amplitude[0, 1]) == pytest.approx(np.exp(-0.5 * gamma**2), rel=1e-6)
