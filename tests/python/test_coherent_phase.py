"""Slow wander of the coherent surface arrival, and the taps' reach (sim/timespread.py items 7, 8).

Closed forms used: the phase of a mirror heaving by zeta is 2 k zeta sin g, so its variance is
(2 k h sin g)^2 and it is proportional to frequency; a sum of sinusoids of a stated spectrum
has that spectrum; first-order reflection from a sinusoidal mirror turns the phase change into
an amplitude change by the angle Theta = rho K^2 (cos^2 a / sin^2 g + sin^2 a) / (2 k); a
chi-square variate of one degree of freedom has 99 % below 6.635, an exponential below 4.605.
"""
import pathlib
import sys

import numpy as np
import pytest

from gpusonar.sim import ImageSourceProvider, Scene, make_pose, simulate_ping
from gpusonar.sim.timespread import CoherentPhase, TimeSpreadProvider, facet_delay_density, facet_intervals, tail_span, wrap
from gpusonar.sonar import SonarModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
C, F = 1500.0, 6000.0
SOURCE = np.array([0.0, 0.0, 20.0])
POINT = np.array([[274.0, 0.0, 43.0]])                     # the measured geometry: 14.5 degrees at the surface
SURFACE = {"rms_height": 0.014, "mean_square_slope": 0.0105, "rms_vertical_speed": 0.045}
SWELL = {"rms_height": 0.008, "period_s": 5.0, "direction_deg": 90.0}          # crests along the path


def provider(coherent_phase=None, surface=SURFACE, frequency=F, **kw):
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1, surface_coefficient=-1.0)
    return TimeSpreadProvider(inner, frequency=frequency, surface=surface, surface_z=0.0, coherent_phase=coherent_phase, **kw), inner


def bounce(inner, source=SOURCE, point=POINT):
    base = inner.eigenrays(source, point)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    sin_g = abs(base.launch[0, j, 2])
    f = source[2] / (source[2] + point[0, 2])
    return base, j, sin_g, C * f * (1.0 - f) * base.delay[0, j]


# --- the taps' reach (item 7) --------------------------------------------------------------------------

def test_tail_span_is_the_quantile_of_the_shape():
    assert float(tail_span(1.0, 2.0, 0.99)) == pytest.approx(6.6349, rel=1e-4)            # b -> 0: chi-square, one degree of freedom
    assert float(tail_span(2.0, 4.0, 0.99)) == pytest.approx(-2.0 * np.log(0.01), rel=1e-9)    # a = b: the exponential of mean 2 a
    # In between: the share of the facet shape inside the span (integrated in sqrt(tau), where it is smooth).
    worst = 0.0
    for ratio in (1e-3, 0.03, 0.1, 0.3, 0.6, 1.0):
        for q in (0.9, 0.99):
            span = float(tail_span(1.0 + ratio, 2.0 * (1.0 + ratio**2), q))
            u = np.linspace(0.0, np.sqrt(span), 200001)
            inside = np.trapezoid(facet_delay_density(u * u, 1.0, ratio) * 2.0 * u, u)
            worst = max(worst, abs(inside - q))
    assert worst < 0.006


def test_facet_intervals_hold_the_energy_and_the_mean_exactly():
    a, b = 0.7, 0.05
    edges = np.linspace(0.0, 5.0, 11)
    w, t = facet_intervals(edges, a, b)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert (w * t).sum() == pytest.approx(a + b, rel=1e-12)
    u = np.linspace(0.0, np.sqrt(edges[-1]), 400001)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (facet_delay_density(u[1:] ** 2, a, b) * 2 * u[1:] + facet_delay_density(u[:-1] ** 2, a, b) * 2 * u[:-1]) * np.diff(u))])
    exact = np.diff(np.interp(np.sqrt(edges), u, cdf))
    assert np.allclose(w[:-1], exact[:-1], rtol=0, atol=2e-7)
    assert np.all((t[:-1] > edges[:-2]) & (t[:-1] < edges[1:-1])) and t[-1] > edges[-2]
    # the exponential (a = b): intervals in closed form
    w, t = facet_intervals(edges, 0.5, 0.5)
    assert np.allclose(w[:-1], np.exp(-edges[:-2]) - np.exp(-edges[1:-1]), rtol=0, atol=1e-10)


@pytest.mark.parametrize("rule", ["facet", "rule_of_thumb"])
def test_taps_reach_the_quantile_with_exact_energy_and_mean(rule):
    base, j, sin_g, rho = bounce(provider()[1])
    mss, taps = SURFACE["mean_square_slope"], 12
    lever = rho / C
    a, b = mss * lever, mss * lever * sin_g**2
    mean = a + b if rule == "facet" else 2.0 * a
    variance = 2.0 * (a * a + b * b) if rule == "facet" else mean**2

    def taps_of(**kw):
        p, _ = provider(taps=taps, decay_time=rule, doppler=True, doppler_lines=2, **kw)
        e = p.eigenrays(SOURCE, POINT)
        amp = e.amplitude[0, 2:].reshape(2, taps, 2)[j]
        return (np.abs(amp) ** 2).sum(axis=1), e.delay[0, 2:].reshape(2, taps, 2)[j][:, 0] - base.delay[0, j], e

    w0, t0, e0 = taps_of()
    w, t, e = taps_of(tail_quantile=0.99)
    span = float(tail_span(mean, variance, 0.99))
    assert t0[-1] < 3.3 * mean and t[-2] < span * (taps - 1) / taps                       # before: three means; now: the 99 % point
    assert span / (3.0 * mean) == pytest.approx(2.1 if rule == "facet" else 1.535, rel=0.03)      # 2.21 as the grazing angle goes to zero
    assert w.sum() == pytest.approx(w0.sum(), rel=1e-9)                                  # the same energy
    assert (w * t).sum() / w.sum() == pytest.approx(mean, rel=1e-9)                      # the shape's own mean, no rescaling
    if rule == "facet":
        want, _ = facet_intervals(span * np.arange(taps + 1) / taps, a, b)
        assert np.allclose(w / w.sum(), want, rtol=1e-9)
    # the coherent arrival and every other ray are untouched
    assert np.array_equal(e.amplitude[0, :2], e0.amplitude[0, :2]) and np.array_equal(e.delay[0, :2], e0.delay[0, :2])
    with pytest.raises(ValueError, match="tail_quantile"):
        provider(tail_quantile=1.0)


def test_tail_quantile_absent_is_the_earlier_output_and_two_bounces_keep_their_rule():
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=60.0, bottom_coefficient=0.5, max_order=2, surface_coefficient=-1.0)      # the constant, stated (no default since D-104)
    rough = {"rms_height": 0.05, "mean_square_slope": 0.02, "rms_vertical_speed": 0.1}
    kw = dict(frequency=F, surface=rough, bottom=rough, surface_z=0.0, bottom_z=60.0, taps=8, doppler=True, doppler_lines=2)
    plain = TimeSpreadProvider(inner, **kw).eigenrays(SOURCE, POINT)
    none = TimeSpreadProvider(inner, tail_quantile=None, coherent_phase=None, **kw).eigenrays(SOURCE, POINT)
    assert np.array_equal(plain.amplitude, none.amplitude) and np.array_equal(plain.delay, none.delay)
    far = TimeSpreadProvider(inner, tail_quantile=0.99, **kw).eigenrays(SOURCE, POINT)
    k = plain.amplitude.shape[1] - 8 * 2 * inner.eigenrays(SOURCE, POINT).delay.shape[1]
    power = lambda e: (np.abs(e.amplitude[0, k:]) ** 2).reshape(-1, 8, 2).sum(axis=(1, 2))      # noqa: E731
    assert np.allclose(power(far), power(plain), rtol=1e-9)                              # energy per ray unchanged
    reach = lambda e: (e.delay[0, k:].reshape(-1, 8, 2)[:, -1, 0] - e.delay[0, k:].reshape(-1, 8, 2)[:, 0, 0])   # noqa: E731
    many = (inner.eigenrays(SOURCE, POINT).n_surface[0] + inner.eigenrays(SOURCE, POINT).n_bottom[0]) >= 1
    assert np.all(reach(far)[many] > 1.2 * reach(plain)[many])


# --- the wander of the coherent arrival (item 8) -------------------------------------------------------

def test_phase_variance_is_that_of_a_heaving_mirror_and_follows_the_stated_spectrum():
    base, j, sin_g, rho = bounce(provider()[1])
    k = 2.0 * np.pi * F / C
    cp = CoherentPhase(dict(SWELL, waves=32, focusing=False, fresnel_limit=1e9), frequency=F)
    t = np.arange(0.0, 4000.0, 0.25)
    m = cp.modulation(t, np.zeros((len(t), 2)), np.tile([1.0, 0.0], (len(t), 1)), np.full(len(t), sin_g), np.zeros(len(t)), C, SURFACE["rms_height"])
    phase = -2.0 * np.pi * F * m["delay"]
    assert phase.var() == pytest.approx((2.0 * k * SWELL["rms_height"] * sin_g) ** 2, rel=0.02)
    assert np.all(m["height2"] == pytest.approx(SWELL["rms_height"] ** 2))
    # the rate is the derivative of the delay
    assert np.allclose(np.gradient(m["delay"], t)[5:-5], m["delay_rate"][5:-5], atol=0.02 * np.abs(m["delay_rate"]).max())
    spec = np.abs(np.fft.rfft(phase)) ** 2
    f = np.fft.rfftfreq(len(t), 0.25)
    line = (f > 0.2 * 0.7) & (f < 0.2 * 1.3)
    assert spec[line].sum() / spec[1:].sum() > 0.99                                      # a line at 1 / period, 10 % wide
    assert np.sqrt((spec[1:] * f[1:] ** 2).sum() / spec[1:].sum()) == pytest.approx(0.2 * np.sqrt(1.0 + 0.1**2), rel=0.03)
    # a table: three quarters of the power between 0.1 and 0.2 Hz, a quarter between 0.3 and 0.5 Hz
    table = {"frequency_hz": [0.1, 0.2, 0.2001, 0.2999, 0.3, 0.5], "density": [7.5, 7.5, 0.0, 0.0, 1.25, 1.25]}
    cp = CoherentPhase({"share": 0.5, "spectrum": table, "direction_deg": 0.0, "waves": 64, "fresnel_limit": 1e9}, frequency=F)
    d = cp.modulation(t, np.zeros((len(t), 2)), np.tile([0.0, 1.0], (len(t), 1)), np.full(len(t), sin_g), np.zeros(len(t)), C, 0.02)["delay"]
    spec = np.abs(np.fft.rfft(d)) ** 2
    low, high = spec[(f >= 0.1) & (f <= 0.2)].sum(), spec[(f >= 0.3) & (f <= 0.5)].sum()
    assert low / high == pytest.approx(3.0, rel=0.1) and (low + high) / spec[1:].sum() > 0.99
    assert (2.0 * np.pi * F * d).var() == pytest.approx(0.5 * (2.0 * k * 0.02 * sin_g) ** 2, rel=0.03)


def test_fresnel_angle_turns_phase_into_amplitude_and_large_zones_do_not_wander():
    base, j, sin_g, rho = bounce(provider()[1])
    k = 2.0 * np.pi * F / C
    t = np.arange(0.0, 2000.0, 0.25)
    args = (t, np.zeros((len(t), 2)), np.tile([1.0, 0.0], (len(t), 1)), np.full(len(t), sin_g), np.full(len(t), rho), C, SURFACE["rms_height"])
    z2 = (2.0 * k * SWELL["rms_height"] * sin_g) ** 2
    for direction, limit in ((90.0, 1.0), (0.0, 1.0), (0.0, 0.3)):
        cp = CoherentPhase(dict(SWELL, direction_deg=direction, relative_width=1e-6, waves=1, fresnel_limit=limit), frequency=F)
        wave_k = (2.0 * np.pi / 5.0) ** 2 / 9.81
        theta = rho * wave_k**2 * ((1.0 / sin_g**2) if direction == 0.0 else 1.0) / (2.0 * k)
        m = cp.modulation(*args)
        phase = -2.0 * np.pi * F * m["delay"]
        if theta < limit:
            assert phase.var() == pytest.approx(z2 * np.cos(theta) ** 2, rel=0.01)
            assert m["amplitude"].var() == pytest.approx(z2 * np.sin(theta) ** 2, rel=0.01)
            assert np.all(m["amplitude_variance"] == pytest.approx(z2 * np.sin(theta) ** 2))
            # in step: where the path is shortest (phase most advanced) the wave defocuses
            assert np.corrcoef(phase, m["amplitude"])[0, 1] == pytest.approx(-1.0, abs=1e-6)
        else:
            assert not np.any(m["delay"]) and not np.any(m["amplitude"]) and not np.any(m["height2"])
    # Crests along the path at this geometry: 0.03 rad. Waves of 0.2 Hz running along it: 0.63 rad, an amplitude variance
    # tan^2 = 0.52 of the phase variance (measured: 0.0037 / 0.0084 = 0.44; the direction of the waves is not known).
    assert rho * ((2.0 * np.pi / 5.0) ** 2 / 9.81) ** 2 / (2.0 * k) == pytest.approx(0.031, abs=0.003)
    along = rho * ((2.0 * np.pi / 5.0) ** 2 / 9.81) ** 2 / sin_g**2 / (2.0 * k)
    assert along == pytest.approx(0.63, abs=0.03) and np.tan(along) ** 2 == pytest.approx(0.52, abs=0.06)


def coherent_ray(p, j, time):
    p.set_ping(time)
    e = p.eigenrays(SOURCE, POINT)
    return e.delay[0, j], e.amplitude[0, j], (None if e.range_rate is None else e.range_rate[0, j]), e


def test_the_coherent_ray_moves_as_one_path_and_the_energy_is_unchanged():
    p0, inner = provider(doppler=True, doppler_lines=2)
    base, j, sin_g, rho = bounce(inner)
    k = 2.0 * np.pi * F / C
    p, _ = provider(dict(SWELL, focusing=False), doppler=True, doppler_lines=2)
    times = np.arange(0.0, 600.0, 0.37)
    rows = [coherent_ray(p, j, t) for t in times]
    delay = np.array([r[0] for r in rows]) - np.array([r[2] for r in rows]) * (-base.delay[0, j]) / C - base.delay[0, j]
    # (the ray's delay is set back by its rate x the time at which the kernel takes it: put that back)
    phase = -2.0 * np.pi * F * delay
    assert phase.var() == pytest.approx((2.0 * k * SWELL["rms_height"] * sin_g) ** 2, rel=0.05)
    assert np.sqrt(phase.var()) == pytest.approx(0.1, abs=0.01)                          # measured: 0.0084 rad^2
    # Energy: the coherent part is that of the small-scale height alone; with the tail it is what it was.
    e0, e1 = coherent_ray(p0, j, 0.0)[3], rows[0][3]
    h_small = np.sqrt(SURFACE["rms_height"] ** 2 - SWELL["rms_height"] ** 2)
    assert abs(e1.amplitude[0, j]) ** 2 == pytest.approx(abs(base.amplitude[0, j]) ** 2 * np.exp(-(2.0 * k * h_small * sin_g) ** 2), rel=1e-9)
    total = lambda e: (np.abs(e.amplitude[0]) ** 2).sum()                                # noqa: E731
    assert total(e1) == pytest.approx(total(e0), rel=1e-9)
    assert np.array_equal(e1.delay[0, 1 - j], e0.delay[0, 1 - j]) and e1.range_rate[0, 1 - j] == 0.0        # the direct ray is untouched
    # The tail keeps its own rate: rms_vertical_speed over the small-scale height (the wave-frequency limit).
    nu = -e1.range_rate[0, 2:] * F / C
    assert np.sqrt(np.mean(nu[e1.valid[0, 2:]] ** 2)) == pytest.approx(SURFACE["rms_vertical_speed"] / (2.0 * np.pi * h_small), rel=0.8)
    # Absent: the earlier output; a share of the surface's height: the same thing said another way.
    assert np.array_equal(provider(None, doppler=True, doppler_lines=2)[0].eigenrays(SOURCE, POINT).amplitude, e0.amplitude)
    share = (SWELL["rms_height"] / SURFACE["rms_height"]) ** 2
    q, _ = provider({"share": share, "period_s": 5.0, "direction_deg": 90.0, "focusing": False}, doppler=True, doppler_lines=2)
    assert coherent_ray(q, j, 3.0)[0] == pytest.approx(coherent_ray(p, j, 3.0)[0], rel=1e-12)
    with pytest.raises(ValueError, match="larger than the surface"):
        provider(dict(SWELL, rms_height=0.05))[0].eigenrays(SOURCE, POINT)
    for bad in ({"period_s": 5.0, "direction_deg": 0.0}, {"rms_height": 0.01, "direction_deg": 0.0}, {"rms_height": 0.01, "period_s": 5.0}):
        with pytest.raises(ValueError):
            provider(bad)


def test_long_range_low_grazing_paths_do_not_wander():
    far = np.array([[12000.0, 0.0, 30.0]])
    sea = {"rms_height": 0.3, "mean_square_slope": 0.02, "rms_vertical_speed": 0.4}
    plain, _ = provider(None, surface=sea, frequency=5000.0)
    with_swell, _ = provider({"share": 0.5, "period_s": 6.0, "direction_deg": 0.0, "spread_deg": 20.0}, surface=sea, frequency=5000.0)
    a, b = plain.eigenrays(SOURCE, far), with_swell.eigenrays(SOURCE, far)
    assert np.array_equal(a.amplitude, b.amplitude) and np.array_equal(a.delay, b.delay)
    assert b.range_rate is None or not np.any(b.range_rate)


def test_half_band_variance_ratio_reads_what_a_change_of_path_length_reads():
    """The measurement: 1.57 +- 0.07 where the estimator reads 1.42 for a pure change of path length
    and 1 for a phase that does not depend on frequency. The model's coherent arrival through the
    same estimator (tools/channel_smartocean.py)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import channel_smartocean as so
    from gpusonar.sim import channel_replay as cr
    from gpusonar.sim import channel_sounding as cs
    assert so.half_band_ratio_for_path_change() == pytest.approx(1.42, abs=0.02)
    cfg = {"surface": dict(SURFACE), "coherent_phase": dict(SWELL, focusing=False, legs=1), "taps": 4, "seed": 3, "incoherent": False}
    prov = wrap(ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1), cfg, frequency=F)
    n = 256
    f = np.fft.fftfreq(n, 1.0 / so.FS)
    win = cs.band_window(n, so.FS, so.BAND, "hann")
    t = np.arange(1600) / 4.0
    h = np.zeros((len(t), n), dtype=complex)
    first = None
    for i, when in enumerate(t):
        prov.set_ping(when)
        e = prov.eigenrays(SOURCE, POINT)
        ok = e.valid[0]
        first = e.delay[0][ok].min() - so.LEAD if first is None else first
        # every ray at its own delay at every frequency of the band: a change of delay is a phase proportional to frequency
        spectrum = (e.amplitude[0][ok][:, None] * np.exp(-2j * np.pi * (so.FC + f)[None, :] * (e.delay[0][ok] - first)[:, None])).sum(axis=0)
        h[i] = np.fft.ifft(spectrum * win)
    b1, b2 = so.find_pair((np.abs(h) ** 2).mean(axis=0), "R1")
    lower, upper = so.half_band_phase_variances(h, b1, b2, t, 4.0, so.FS)
    assert upper / lower == pytest.approx(1.42, abs=0.06)
    k = 2.0 * np.pi * F / C
    sin_g = bounce(ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1))[2]
    assert 0.5 * (lower + upper) == pytest.approx((2.0 * k * SWELL["rms_height"] * sin_g) ** 2, rel=0.25)


class BounceOnly(ImageSourceProvider):
    """Surface image paths without the direct ray: one echo path (surface both ways) to look at."""

    def eigenrays(self, source, points):
        e = super().eigenrays(source, points)
        e.valid = e.valid & (e.n_surface == 1)
        e.amplitude = np.where(e.valid, e.amplitude, 0.0)
        return e


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_the_echo_phase_follows_the_surface_through_a_long_pulse_and_from_ping_to_ping(engine):
    """A 1 s tone by way of the surface both ways, against a 5 s swell of 5 cm rms with its crests
    along the path: the received phase is twice 2 k zeta sin g at the time the sound was at the
    target, within the pulse (to the second order the linear law leaves) and from ping to ping."""
    f0, duration = 5000.0, 1.0
    sonar = SonarModel.load("flank_32", {"array": {"count": 1}, "transmit": {"waveform": {"type": "cw", "f_start": f0, "f_stop": f0, "duration": duration,
                                                                                         "window_alpha": 0.0, "source_level_db": 190.0}},
                                         "receive": {"kind": "baseband", "sample_rate": 2000.0, "window": {"range": [280.0, 1180.0]}}})
    swell = {"rms_height": 0.05, "period_s": 5.0, "relative_width": 1e-6, "waves": 1, "direction_deg": 0.0, "focusing": False, "span_s": duration}
    sea = {"rms_height": 0.05, "mean_square_slope": 0.0, "rms_vertical_speed": 0.0}
    point = np.array([[0.0, 274.0, 43.0]])                                               # the path runs along y: the swell (towards +x) has its crests along it
    inner = BounceOnly(C, surface_z=0.0, bottom_z=None, max_order=1, surface_coefficient=-1.0)
    flat = TimeSpreadProvider(inner, frequency=f0, surface=None, surface_z=0.0)
    wavy = TimeSpreadProvider(inner, frequency=f0, surface=sea, surface_z=0.0, coherent_phase=swell)
    scene = Scene().add_points(point.tolist(), target_strength_db=0.0)
    pose = make_pose(tuple(SOURCE))
    kw = dict(environment={"sound_speed": C}, adc=False, absorption_model="none", engine=engine)
    reference = np.array(simulate_ping(sonar, pose, scene, provider=flat, **kw).samples.array)[0]
    rec = simulate_ping(sonar, pose, scene, provider=flat, **kw)
    t = rec.samples.start_time + np.arange(rec.samples.n_samples) / rec.samples.sample_rate
    on = np.abs(reference) > 0.5 * np.abs(reference).max()
    one_way = float(inner.eigenrays(SOURCE, point).delay[0][inner.eigenrays(SOURCE, point).valid[0]][0])
    assert t[on][0] == pytest.approx(2.0 * one_way, abs=0.01) and t[on][-1] - t[on][0] == pytest.approx(duration, abs=0.01)
    cp = wavy.coherent_phase
    sin_g = abs(inner.eigenrays(SOURCE, point).launch[0][inner.eigenrays(SOURCE, point).valid[0]][0][2])
    f_frac = SOURCE[2] / (SOURCE[2] + point[0, 2])
    xy = SOURCE[:2] + f_frac * (point[0, :2] - SOURCE[:2])
    errors, phases = [], []
    for ping_time in (0.0, 0.8, 1.9, 3.1, 4.4, 7.7):
        wavy.set_ping(ping_time)
        got = np.array(simulate_ping(sonar, pose, scene, provider=wavy, scenario_time=ping_time, **kw).samples.array)[0]
        measured = np.unwrap(np.angle(got[on] / reference[on]))
        when = ping_time + t[on] - one_way                                               # when the sound of each sample was at the target
        m = cp.modulation(when, np.tile(xy, (on.sum(), 1)), np.tile([0.0, 1.0], (on.sum(), 1)), np.full(on.sum(), sin_g), np.zeros(on.sum()), C, 0.05)
        truth = -2.0 * np.pi * f0 * 2.0 * m["delay"]
        inner_part = slice(20, -20)
        errors.append(np.abs(measured - truth)[inner_part].max())
        phases.append((measured[inner_part].mean(), truth[inner_part].mean()))
        assert np.abs(got[on]).mean() == pytest.approx(np.abs(reference[on]).mean(), rel=1e-3)        # a mirror: the level is the flat surface's
    z = 2.0 * (2.0 * np.pi * f0 / C) * 0.05 * np.sqrt(2.0) * sin_g * 2.0                 # peak two-way phase, rad
    assert z > 1.0
    # Within a pulse the law is linear in time: what is left is the curvature, (2 pi / 5 s)^2 z (0.5 s)^2 / 2 at most.
    assert max(errors) < 1.15 * (2.0 * np.pi / 5.0) ** 2 * z * 0.5**2 / 2.0 + 0.02
    got, want = np.array(phases).T
    assert np.abs(got - want).max() < 0.08 * z + 0.02                                    # ping to ping: the surface's own phase
    assert np.ptp(want) > 0.8 * z
