"""The exact Kirchhoff (geometric-optics) forward-scatter tail, `time_spread.decay_time: kirchhoff_go` (opt-in).

Sources (docs/references/surface_forward_scatter_dahl_1996_2004.md, "[NF]"):
- Dahl 1996, JASA 100, 748-758: eqs. (4), (7)-(10) (the bistatic cross section sigma_geo and the intensity of a
  patch), eq. (13) (the sum over the surface equals a flat mirror "within 0.25 dB numerically"), Table I and Fig. 2
  ([NF] Secs. 2.2, 2.3, 2.7; equations and table confirmed, Fig. 2 read from figure);
- Dahl 2004, JASA 115, 589-599: Fig. 3c, the 52 m receiver at ASIAEX ([NF] Secs. 2.6, 2.7; read from figure).
The tolerances are [NF] WP3's: 2 dB rms (Fig. 3c), 3.5 dB rms (Fig. 2), 20 % (Table I kd*), 0.25 dB (eq. 13).
The figure readings carry about +-1 dB ([NF] header) and are labelled `figure`; the slopes are the papers' own
(Table I's s_LA, Dahl 2004's S_L about 0.15, or Dahl 1996 eq. (A7) at the measured wind), never fitted here.
"""

import numpy as np
import pytest

from gpusonar.sim import ImageSourceProvider
from gpusonar.sim.timespread import (TimeSpreadProvider, boundary_parameters, kirchhoff_go_cdf, kirchhoff_go_closure,
                                     kirchhoff_go_density, kirchhoff_go_intervals, kirchhoff_go_iso_delay, kirchhoff_go_tail,
                                     wrap)

E12 = np.exp(-0.5)
A7 = lambda u: 0.0046 * np.log(2.1 * u * u)                                   # noqa: E731  Dahl 1996 eq. (A7)

# Dahl 1996 Table I (p. 752), printed: set -> (SD m, range m, U m/s, s_LA, kd*/(2 pi)). RD = 66 m ([NF] Sec. 2.2).
TABLE_I = {"1": (147.0, 1000.0, 7.6, 0.155, 5.0), "2": (57.0, 1000.0, 5.3, 0.120, 14.5), "3": (147.0, 576.0, 3.2, 0.110, 4.9),
           "4": (147.0, 724.0, 1.5, 0.040, 17.7), "5A": (147.0, 714.0, 5.0, 0.114, 5.0), "5B": (57.0, 714.0, 5.0, 0.114, 12.5)}
RD96 = 66.0

# figure: Dahl 1996 Fig. 2, set 1 run 1, 4 ms pulse, ending at about 5.6 ms: (ms, dB re the plateau) ([NF] Sec. 2.7).
FIG2 = [(6.0, -1.5), (6.5, -4.5), (7.0, -7.0), (7.5, -8.0), (8.0, -9.0), (9.0, -10.2), (10.0, -11.5), (11.0, -12.5), (12.0, -13.8),
        (13.0, -14.6), (14.0, -15.5), (15.0, -17.0), (16.0, -17.8), (17.0, -18.5), (18.0, -18.3), (19.0, -19.5), (20.0, -21.2)]
# figure: Dahl 2004 Fig. 3c, lower VLA (52 m), 20 kHz, 3 ms pulse, peak at about 6.5 ms ([NF] Sec. 2.7).
FIG3C = [(7.5, -5.0), (8.0, -7.0), (9.0, -11.0), (10.0, -13.5), (11.0, -15.0), (12.0, -18.0), (13.0, -21.0)]


def geometry(sd, rd, rng, c):
    """(grazing, t1, t2) of the specular surface bounce between depths sd and rd at horizontal range rng."""
    g = np.arctan((sd + rd) / rng)
    total = np.hypot(rng, sd + rd) / c
    return g, total * sd / (sd + rd), total * rd / (sd + rd)


def pulse_levels_db(tail, pulse, after_end):
    """[NF] Sec. 3.3's comparison: the impulse response convolved with a rectangular pulse of length `pulse`,
    normalised to its peak, read at `after_end` (s) after the pulse's end. For a rectangle the convolution is
    CDF(t) - CDF(t - pulse), t from the specular arrival."""
    t = np.linspace(0.0, pulse + 0.05, 20001)
    rec = kirchhoff_go_cdf(t, tail) - kirchhoff_go_cdf(np.maximum(t - pulse, 0.0), tail) * (t > pulse)
    return 10.0 * np.log10(np.interp(pulse + np.asarray(after_end), t, rec) / rec.max())


# -- the model itself --------------------------------------------------------------------------------------------

def test_the_quadrature_equals_a_plain_sum_over_the_surface():
    """kirchhoff_go_density integrates [D96] eqs. (4), (9), (10) around each curve of equal delay. Against a plain
    sum over a 1 m x 0.25 m grid of the surface (independent code, as in [NF]'s scratch model): the same energy
    and mean delay, for set 1 of Table I."""
    sd, rng, _, s_la, _ = TABLE_I["1"]
    mss = s_la**2
    x = np.linspace(-200.0, 1200.0, 1401)
    y = np.linspace(-120.0, 120.0, 961)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vi = np.stack([X, Y, np.full_like(X, sd)], -1)
    vs = np.stack([rng - X, -Y, np.full_like(X, -RD96)], -1)
    ri, rs = np.linalg.norm(vi, axis=-1), np.linalg.norm(vs, axis=-1)
    q = vi / ri[..., None] - vs / rs[..., None]
    tan2 = (q[..., 0] ** 2 + q[..., 1] ** 2) / q[..., 2] ** 2                  # eq. (10)
    sigma = (1.0 + tan2) ** 2 / (4.0 * np.pi * mss) * np.exp(-tan2 / mss)       # eq. (9)
    intensity = sigma / (ri**2 * rs**2) * (x[1] - x[0]) * (y[1] - y[0])        # eq. (4), B = 1
    d = np.hypot(rng, sd + RD96)
    grid_energy = intensity.sum() * d**2                                       # re the flat mirror 1 / d^2, eq. (13)
    grid_mean = np.sum(intensity * (ri + rs - d)) / intensity.sum() / d         # excess path / (r1 + r2)
    g, f = np.arctan((sd + RD96) / rng), sd / (sd + RD96)
    assert kirchhoff_go_closure(g, f, mss) == pytest.approx(grid_energy, rel=0.003)
    delta = np.linspace(0.0, 0.3, 60001)[1:]
    w = kirchhoff_go_density(delta, g, f, mss) * (delta[1] - delta[0])
    assert np.sum(w) == pytest.approx(grid_energy, rel=0.003)
    assert np.sum(w * delta) / np.sum(w) == pytest.approx(grid_mean, rel=0.01)


def test_dahl_1996_eq_13_the_sum_is_a_flat_mirror_at_moderate_and_steep_grazing():
    """[D96] eq. (13): sum sigma_geo A / (R_i^2 R_s^2) = A_0 / (R_i + R_s)^2 within 0.25 dB. Holds for our
    quadrature from 20 degrees up at slopes to 0.03 (at 20 degrees: -0.21 dB at 0.03, -0.18 dB at 0.0221; at 15
    degrees -0.34 dB at 0.03), and tends to 0 dB as the slope vanishes. The Table I geometries: the xfail below."""
    for g_deg in (20.0, 30.0, 45.0, 60.0, 80.0, 90.0):
        for f in (0.1, 0.3, 0.5):
            for mss in (0.005, 0.0221, 0.03):
                assert abs(10.0 * np.log10(kirchhoff_go_closure(np.radians(g_deg), f, mss))) < 0.25, (g_deg, f, mss)
            assert abs(10.0 * np.log10(kirchhoff_go_closure(np.radians(g_deg), f, 1e-4))) < 0.01


@pytest.mark.xfail(strict=True, reason="[D96] eq. (13) 'within 0.25 dB' does not hold for our GO quadrature at Table I's "
                   "low-grazing sets: with s_LA the sum is -0.35 dB (set 1, 12 deg), -0.62 dB (set 2, 7 deg) and -0.40 dB "
                   "(set 5B, 10 deg) re the flat mirror (sets 3, 4, 5A: -0.07, -0.02, -0.13 dB) (computed; an independent grid sum agrees within 0.01 dB). Dahl's "
                   "geometry for the 0.25 dB is not printed. The model keeps the flat mirror's energy (the shape only is GO's)")
def test_dahl_1996_eq_13_within_025_db_at_every_table_i_geometry():
    for sd, rng, _, s_la, _ in TABLE_I.values():
        g, f = np.arctan((sd + RD96) / rng), sd / (sd + RD96)
        assert abs(10.0 * np.log10(kirchhoff_go_closure(g, f, s_la**2))) < 0.25


def test_the_go_mean_delay_converges_to_the_facet_form_at_steep_grazing():
    """The facet model's mean mss (1 + sin^2 g) t1 t2 / (t1 + t2) is the second-order limit of GO: equal at steep
    grazing and small slopes; at 7-12 degrees with Dahl's slopes GO is 1.4-1.6 times shorter ([NF] Sec. 3.3)."""
    for g_deg in (45.0, 60.0, 75.0, 89.0):
        for mss in (0.005, 0.02):
            g = np.radians(g_deg)
            tail = kirchhoff_go_tail(g, 0.3, 0.7, mss)
            facet = mss * (1.0 + np.sin(g) ** 2) * 0.3 * 0.7
            assert tail["mean"] == pytest.approx(facet, rel=0.01), (g_deg, mss)
    g = np.radians(20.0)
    assert kirchhoff_go_tail(g, 0.3, 0.7, 1e-4)["mean"] == pytest.approx(1e-4 * (1 + np.sin(g) ** 2) * 0.21, rel=0.005)
    g, t1, t2 = geometry(25.0, 52.0, 467.0, 1525.0)                            # [D04] 52 m: GO 1.00 ms, facet 1.57 ms
    facet = 0.0225 * (1.0 + np.sin(g) ** 2) * t1 * t2 / (t1 + t2)
    assert facet / kirchhoff_go_tail(g, t1, t2, 0.0225)["mean"] == pytest.approx(1.57, abs=0.03)


def test_the_tail_table_follows_the_direct_quadrature_between_its_nodes():
    """kirchhoff_go_tail interpolates a table in grazing and r1 / (r1 + r2) (exact in the slope): between nodes
    its mean and quantiles are within 2 % of the quadrature at that geometry."""
    for g_deg, f, mss in ((11.3, 0.27, 0.02), (3.7, 0.43, 0.013), (27.0, 0.08, 0.03), (6.2, 0.33, 0.0225)):
        g = np.radians(g_deg)
        tail = kirchhoff_go_tail(g, f, 1.0 - f, mss)
        delta = np.concatenate([np.linspace(0.0, 1e-3, 200001)[1:], np.linspace(1e-3, 0.2, 200001)[1:]])
        p = kirchhoff_go_density(delta, g, f, mss)
        w = p * np.gradient(delta)
        assert tail["mean"] == pytest.approx(np.sum(w * delta) / np.sum(w), rel=0.02), (g_deg, f, mss)
        cdf = np.cumsum(w) / np.sum(w)
        for q in (0.5, 0.9, 0.99):
            assert np.interp(q, tail["levels"], tail["delay"]) == pytest.approx(np.interp(q, cdf, delta), rel=0.02), (g_deg, f, q)


# -- against the papers' measurements (figure readings) -----------------------------------------------------------

@pytest.mark.parametrize("mss, label", [(0.15**2, "S_L 0.15 ([D04] Fig. 6, figure)"), (A7(7.0), "eq. (A7) at 7 m/s")])
def test_dahl_2004_fig_3c_52_m_within_2_db_rms(mss, label):
    """figure: [D04] Fig. 3c, the 52 m receiver (SD 25 m, 467 m, 9.4 degrees, 20 kHz, 3 ms): within 2 dB rms
    ([NF] WP3; computed there 1.0 with S_L^2 = 0.0225). Times after the pulse's end, taken at the 6.5 ms peak."""
    g, t1, t2 = geometry(25.0, 52.0, 467.0, 1525.0)
    tail = kirchhoff_go_tail(g, t1, t2, mss)
    t, read = np.array(FIG3C).T
    model = pulse_levels_db(tail, 3e-3, (t - 6.5) * 1e-3)
    rms = float(np.sqrt(np.mean((model - read) ** 2)))
    print(f"[D04] Fig. 3c 52 m, {label}: model - read {np.round(model - read, 1)} dB, rms {rms:.2f} dB")
    assert rms < 2.0


@pytest.mark.parametrize("mss, label", [(0.155**2, "s_LA of Table I"), (A7(7.6), "eq. (A7) at 7.6 m/s")])
def test_dahl_1996_fig_2_within_3_5_db_rms(mss, label):
    """figure: [D96] Fig. 2, set 1 (SD 147 m, RD 66 m, 1000 m, 12 degrees, 4 ms): within 3.5 dB rms ([NF] WP3;
    computed there 2.9 with s_LA^2). Times after the pulse's end at about 5.6 ms."""
    g, t1, t2 = geometry(147.0, 66.0, 1000.0, 1500.0)
    tail = kirchhoff_go_tail(g, t1, t2, mss)
    t, read = np.array(FIG2).T
    model = pulse_levels_db(tail, 4e-3, (t - 5.6) * 1e-3)
    rms = float(np.sqrt(np.mean((model - read) ** 2)))
    print(f"[D96] Fig. 2, {label}: model - read {np.round(model - read, 1)} dB, rms {rms:.2f} dB")
    assert rms < 3.5


def _go_kd_star(sd, rng, mss, taps=48):
    """kd*/(2 pi) across the path at the receiver from the provider's own distribution: taps of the GO tail's
    intervals (span: 0.999 of the energy), each tap's directions spread over its curve of equal delay as
    kirchhoff_go_iso_delay weights them; coherence sum w exp(i k d u_y) / sum w ([NF] Sec. 3.2's estimator)."""
    g, t1, t2 = geometry(sd, RD96, rng, 1500.0)
    f, total = t1 / (t1 + t2), t1 + t2
    tail = kirchhoff_go_tail(g, t1, t2, mss)
    span = np.interp(0.999, tail["levels"], tail["delay"])
    w, tau = kirchhoff_go_intervals(span * np.arange(taps + 1) / taps, tail)
    pts, pw = kirchhoff_go_iso_delay(tau / total, g, f, mss)                   # [J, N, 2], [J, N] (r1 + r2 = 1)
    rx = np.array([(1.0 - f) * np.cos(g), 0.0, -(1.0 - f) * np.sin(g)])
    to_rx = rx - np.concatenate([pts, np.zeros(pts.shape[:-1] + (1,))], axis=-1)
    u_y = to_rx[..., 1] / np.linalg.norm(to_rx, axis=-1)
    kd = np.linspace(0.0, 2.0 * np.pi * 30.0, 3001)
    gamma = np.abs(np.einsum("j,jn,jnk->k", w, pw, np.exp(1j * u_y[..., None] * kd)))
    i = int(np.argmax(gamma < E12))
    return (kd[i - 1] + (kd[i] - kd[i - 1]) * (gamma[i - 1] - E12) / (gamma[i - 1] - gamma[i])) / (2.0 * np.pi)


def test_table_i_coherence_across_the_path_within_20_percent():
    """[D96] Table I kd*/(2 pi) within 20 % ([NF] WP3) with Table I's s_LA, all six sets. Measured here: -7.2 /
    -7.9 / -14.3 / -19.6 / -2.9 / -12.8 % ([NF] Sec. 3.2's GO with s_LA: 4.7 / 13.6 / 4.2 / 14.4 / 4.9 / 11.0)."""
    for name, (sd, rng, u, s_la, measured) in TABLE_I.items():
        assert _go_kd_star(sd, rng, s_la**2) == pytest.approx(measured, rel=0.20), name


@pytest.mark.xfail(strict=True, reason="With the wind's slope, Dahl 1996 eq. (A7) at the measured wind (the default slope law), "
                   "GO gives kd*/(2 pi) -2.7 / -21.5 / -20.8 / -18.6 / -28.3 % of Table I at sets 1 / 2 / 3 / 5A / 5B "
                   "(computed): wider than eq. (15) with q = 1.348 for the same slope. The facet closed form, q = sqrt 2, is "
                   "within 20 % at these five sets (test_forward_scatter_dahl.py). Set 4 (1.5 m/s, s_L = 2 s_LA) left out")
def test_table_i_coherence_within_20_percent_with_the_winds_slope():
    for name, (sd, rng, u, s_la, measured) in TABLE_I.items():
        if name != "4":
            assert _go_kd_star(sd, rng, A7(u)) == pytest.approx(measured, rel=0.20), name


# -- through the provider -------------------------------------------------------------------------------------------

C, F = 1500.0, 20000.0
SOURCE = np.array([0.0, 0.0, 30.0])
ROUGH = {"rms_height": 0.5, "mean_square_slope": 0.02, "rms_vertical_speed": 0.3}       # Gamma >> 1: no coherent part


def _one_bounce(rng_m=600.0, depth=50.0, **kw):
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1)
    pts = np.array([[rng_m, 0.0, depth]])
    base = inner.eigenrays(SOURCE, pts)
    j = int(np.flatnonzero(base.n_surface[0] == 1)[0])
    prov = TimeSpreadProvider(inner, frequency=F, surface=dict(ROUGH, **kw.pop("surface", {})), surface_z=0.0, **kw)
    e = prov.eigenrays(SOURCE, pts)
    k = base.delay.shape[1]
    taps = prov.taps
    part = lambda a: a[0, k:].reshape((k, taps) + a.shape[2:])[j]                 # noqa: E731
    return base, j, e, part(e.amplitude), part(e.delay) - base.delay[0, j], part(e.launch), part(e.arrival), pts


def test_one_bounce_taps_carry_the_go_tail_exactly():
    """Each tap is an interval of the GO tail's delay (the last one everything later), at that interval's mean
    delay: the taps' mean is the GO mean, their energy the incoherent part, with and without tail_quantile."""
    for kw in ({}, {"tail_quantile": 0.99}):
        base, j, e, amp, tau, _, _, pts = _one_bounce(taps=24, decay_time="kirchhoff_go", **kw)
        g = np.arcsin(abs(base.launch[0, j, 2]))
        t1, t2 = 30.0 / 80.0 * base.delay[0, j], 50.0 / 80.0 * base.delay[0, j]
        tail = kirchhoff_go_tail(g, t1, t2, ROUGH["mean_square_slope"])
        span = np.interp(0.99, tail["levels"], tail["delay"]) if kw else 3.0 * tail["mean"]
        weight, delay = kirchhoff_go_intervals(span * np.arange(25) / 24, tail)
        assert np.allclose(tau, delay, rtol=1e-12)
        assert np.sum(weight) == pytest.approx(1.0, rel=1e-12)
        # At 7.6 degrees and this slope GO is much shorter than the facet tail.
        assert tail["mean"] < 0.8 * ROUGH["mean_square_slope"] * (1 + np.sin(g) ** 2) * t1 * t2 / (t1 + t2)
        assert np.sum(weight * delay) == pytest.approx(tail["mean"], rel=1e-9)            # the GO mean, no rescaling
        # The same random process as the facet tail (the same complex factor per tap): amplitude^2 = incoherent x weight x |g|^2.
        k = base.delay.shape[1]
        inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1)
        facet = TimeSpreadProvider(inner, frequency=F, surface=ROUGH, surface_z=0.0, taps=24, **kw).eigenrays(SOURCE, pts)
        amp_facet = facet.amplitude[0, k:].reshape(k, 24)[j]
        assert np.allclose(np.angle(amp), np.angle(amp_facet), atol=1e-12)
        incoherent = np.abs(base.amplitude[0, j]) ** 2 * (1.0 - np.abs(e.amplitude[0, j]) ** 2 / np.abs(base.amplitude[0, j]) ** 2)
        assert incoherent == pytest.approx(np.abs(base.amplitude[0, j]) ** 2, rel=1e-9)                # Gamma >> 1
        g2 = np.abs(amp) ** 2 / (incoherent * weight)
        assert np.mean(g2) == pytest.approx(1.0, abs=0.6)                                              # unit-variance process


def test_each_tap_comes_from_one_facet_seen_from_both_ends():
    """Exact geometry, the joint distribution of launch and arrival: follow each tap's launch direction to the
    surface; the path through that point is longer than the specular one by the tap's delay, and the tap arrives
    from that same point. (The facet option meets this only to second order: test_timespread.py.)"""
    for seed in range(12):
        base, j, e, amp, tau, launch, arrival, pts = _one_bounce(taps=10, seed=seed, angle_spread=True, decay_time="kirchhoff_go")
        point = SOURCE + (-SOURCE[2] / launch[:, 2])[:, None] * launch
        excess = (np.linalg.norm(point - SOURCE, axis=1) + np.linalg.norm(pts[0] - point, axis=1)) / C - base.delay[0, j]
        assert np.allclose(excess, tau, rtol=1e-6, atol=1e-12)
        exact = (pts[0] - point) / np.linalg.norm(pts[0] - point, axis=1)[:, None]
        assert np.allclose(arrival, exact, atol=1e-9)
        assert np.allclose(np.linalg.norm(launch, axis=1), 1.0) and np.all(np.abs(point[:, 1]) >= 0.0)


def test_the_drawn_points_follow_the_iso_delay_weights():
    """Over many seeds the taps' across-path positions on the surface have the second moment that
    kirchhoff_go_iso_delay's weights give at each tap's delay."""
    taps, seeds = 6, 400
    sums, want = np.zeros(taps), np.zeros(taps)
    for seed in range(seeds):
        base, j, e, amp, tau, launch, _, _ = _one_bounce(taps=taps, seed=seed, angle_spread=True, decay_time="kirchhoff_go")
        point = SOURCE + (-SOURCE[2] / launch[:, 2])[:, None] * launch
        sums += point[:, 1] ** 2
    d = base.delay[0, j] * C
    g = np.arcsin(abs(base.launch[0, j, 2]))
    pts_, pw = kirchhoff_go_iso_delay(tau / base.delay[0, j], g, 30.0 / 80.0, ROUGH["mean_square_slope"])
    want = np.sum(pw * (pts_[..., 1] * d) ** 2, axis=-1)
    assert np.allclose(sums / seeds, want, rtol=0.15)


def test_the_bottom_keeps_the_facet_tail():
    """Only surface bounces take the GO tail: a bottom-only path is bit-identical to `facet`; a surface-bottom path
    convolves the surface's GO tail with the bottom's facet tail (mean sum (1 - c_i) T_i / (1 - prod c_i))."""
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=80.0, bottom_coefficient=0.5, max_order=2)
    pts = np.array([[500.0, 40.0, 40.0], [300.0, -20.0, 10.0]])
    bottom = {"rms_height": 0.5, "mean_square_slope": 0.01, "rms_vertical_speed": 0.0}
    # Doppler lines: their magnitudes are not random, so the taps' energies are the model's mean intensities.
    kw = dict(frequency=F, surface=ROUGH, bottom=bottom, surface_z=0.0, bottom_z=80.0, taps=8, seed=5, angle_spread=True,
              doppler=True, doppler_lines=2)
    go = TimeSpreadProvider(inner, decay_time="kirchhoff_go", **kw).eigenrays(SOURCE, pts)
    facet = TimeSpreadProvider(inner, decay_time="facet", **kw).eigenrays(SOURCE, pts)
    base = inner.eigenrays(SOURCE, pts)
    s, k = base.delay.shape
    only_bottom = base.valid & (base.n_surface == 0) & (base.n_bottom > 0)
    surface_bottom = base.valid & (base.n_surface == 1) & (base.n_bottom == 1)
    assert only_bottom.any() and surface_bottom.any()
    tap = lambda e, a: a[:, k:].reshape((s, k, 8, 2) + a.shape[2:])         # noqa: E731
    for name in ("delay", "amplitude", "launch", "arrival"):
        assert np.array_equal(tap(go, getattr(go, name))[only_bottom], tap(facet, getattr(facet, name))[only_bottom])
        assert np.array_equal(getattr(go, name)[:, :k], getattr(facet, name)[:, :k])
    assert not np.allclose(tap(go, go.delay)[surface_bottom], tap(facet, facet.delay)[surface_bottom])
    prov = TimeSpreadProvider(inner, decay_time="kirchhoff_go", **kw)
    frac, is_surface = prov._bounce_fractions(SOURCE[2], pts[:, 2][:, None], base.launch[..., 2] < 0.0, base.n_surface, base.n_bottom)
    for i, r in zip(*np.nonzero(surface_bottom)):
        w = (np.abs(tap(go, go.amplitude)[i, r]) ** 2).sum(axis=-1)
        tau = tap(go, go.delay)[i, r, :, 0] - base.delay[i, r]
        g = np.arccos(np.hypot(*base.launch[i, r, :2]))
        total = base.delay[i, r]
        f_s, f_b = frac[i, r, int(np.argmax(is_surface[i, r, :2]))], frac[i, r, int(np.argmin(is_surface[i, r, :2]))]
        go_mean = kirchhoff_go_tail(g, f_s * total, (1.0 - f_s) * total, ROUGH["mean_square_slope"])["mean"]
        facet_mean = bottom["mean_square_slope"] * (1.0 + np.sin(g) ** 2) * f_b * (1.0 - f_b) * total
        # Both boundaries scatter everything here (Gamma >> 1): the tail's mean is the sum of the two means.
        assert np.sum(w * tau) / np.sum(w) == pytest.approx(go_mean + facet_mean, rel=1e-9)


def test_unknown_decay_time_is_refused_with_the_nearest_name():
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1)
    with pytest.raises(ValueError, match="kirchhoff_go"):
        TimeSpreadProvider(inner, frequency=F, surface=ROUGH, decay_time="kirchoff_go")
    with pytest.raises(ValueError, match="did you mean 'facet'"):
        wrap(inner, {"surface": ROUGH, "decay_time": "facets"}, frequency=F)
    assert wrap(inner, {"surface": ROUGH, "decay_time": "kirchhoff_go"}, frequency=F).decay_time == "kirchhoff_go"


def test_the_default_tail_is_unchanged():
    """Sums over the eigenrays of a three-boundary-order case with the default (facet) tail, computed from
    sim/timespread.py before `kirchhoff_go` existed (commit e0f97aa): bit for bit to the printed digits."""
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=80.0, bottom_coefficient=0.5, max_order=3)
    pts = np.array([[500.0, 40.0, 40.0], [300.0, -20.0, 10.0], [1000.0, 5.0, 50.0]])
    stored = {(): ((0.009391274014763123 - 0.002043562249644572j), 0.04563100523174423, 54.604238361902034, 131.25923394905064, 360.82392125078303),
              (("angle_spread", True), ("tail_quantile", 0.99)): ((0.010077759849200582 - 0.002563172752072863j), 0.04093884607482539,
                                                                  54.83372011700494, 125.75681295345952, 356.85598148007944)}
    for options, (total, magnitude, delay, launch, arrival) in stored.items():
        p = TimeSpreadProvider(inner, frequency=8000.0, surface=boundary_parameters({"wind_speed": 6.0}, surface=True),
                               bottom={"rms_height": 0.03, "mean_square_slope": 0.02, "rms_vertical_speed": 0.0},
                               surface_z=0.0, bottom_z=80.0, taps=6, seed=5, **dict(options))
        p.set_ping(3.0, 1.0)
        e = p.eigenrays(SOURCE, pts)
        a = e.amplitude[e.valid]
        assert e.delay.shape == (3, 49) and int(e.valid.sum()) == 129
        assert complex(a.sum()) == pytest.approx(total, rel=1e-12)
        assert float(np.abs(a).sum()) == pytest.approx(magnitude, rel=1e-12)
        assert float(e.delay[e.valid].sum()) == pytest.approx(delay, rel=1e-14)
        assert float((e.launch[e.valid] * [1, 2, 3]).sum()) == pytest.approx(launch, rel=1e-12)
        assert float((e.arrival[e.valid] * [3, 2, 1]).sum()) == pytest.approx(arrival, rel=1e-12)


def test_the_scenario_with_every_option_and_the_go_tail_runs_on_both_engines():
    """The taps are computed in Python (sim/timespread.py) and handed to either pairing and kernel: with
    `kirchhoff_go`, angle spreading and Doppler lines, the C++ and NumPy engines give the same record."""
    import warnings
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    cfg = load_scenario_config("flank_sst_spread")
    cfg["pings"]["count"] = 1
    cfg["time_spread"] = dict(cfg["time_spread"], taps=2, doppler_lines=2, decay_time="kirchhoff_go")
    quiet = dict(cfg, reverberation=None, receiver={"adc": False}, sonar_overrides={"receive": {"window": {"range": [990, 1090]}}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = run_scenario(quiet).records[0].samples.array
        b = run_scenario(quiet, engine="numpy").records[0].samples.array
    assert np.isfinite(a).all() and np.abs(a).max() > 0
    assert np.allclose(a, b, atol=1e-6 * np.abs(a).max())
    facet = run_scenario(dict(quiet, time_spread=dict(cfg["time_spread"], decay_time="facet"))).records[0].samples.array
    assert not np.allclose(a, facet, atol=1e-3 * np.abs(a).max())
