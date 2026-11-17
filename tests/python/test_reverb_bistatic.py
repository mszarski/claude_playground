"""Scattering-function reverberation with separate, moving transmitter and receiver
(sim/reverb_bistatic.py, `geometry="bistatic"`), after SST's formulation (TR 0702 Sec. 10-11;
docs/PHYSICS_NOTES.md Sec. 5 and its last section).

Each piece is checked against something independent of the code under test: an integral
over a Cartesian grid, reciprocity, the travel time through a known patch (the iso-time
ellipse), the closed-form Doppler factor of TR 0702 eq. (71), the launch elevations of an
image-source geometry, the point-scatterer method, and the NumPy engine against the C++ one.
"""

import numpy as np
import pytest

from gpusonar import _gpusonar as core
from gpusonar.sim import (FreeFieldProvider, ImageSourceProvider, Scene, lambert_rms_amplitude, make_pose, reverb,
                          reverb_bistatic, seafloor_patch, simulate_ping)
from gpusonar.sim import engine as sim_engine
from gpusonar.sonar import SonarModel

C = 1500.0
MU_DB = -27.0
F = 27.5e3
RX = np.array([0.0, 0.0, -20.0])          # 20 m above a bottom at z = 0
TX = np.array([150.0, 40.0, -30.0])       # 155 m away, 30 m above it
TIMES = np.arange(0.03, 0.4, 1e-3)


def sectors(n):
    return (np.arange(n) + 0.5) * 2.0 * np.pi / n


def blocks(x, n=10):
    return x[: len(x) // n * n].reshape(-1, n).sum(axis=1)


def patch_map(centre, radius):
    """A boundary that scatters only within `radius` of `centre` (a smooth bump)."""
    return lambda x, y: np.exp(-0.5 * ((x - centre[0]) ** 2 + (y - centre[1]) ** 2) / radius**2)


# --- geometry and level ------------------------------------------------------------------------

@pytest.mark.parametrize("provider", [
    FreeFieldProvider(C),
    ImageSourceProvider(C, surface_z=-60.0, bottom_z=None, max_order=1),
], ids=["free_field", "surface_bounce"])
def test_coincident_platforms_reproduce_the_monostatic_power(provider):
    """Monostatic is the special case: with the transmitter at the receiver the bistatic
    calculation gives the power of `reverberation_power`, in every azimuth. (Compared in
    blocks of ten bins: the monostatic code puts each ring into its nearest bin, which
    leaves a ripple of up to 2 dB from bin to bin that the new code does not have.)"""
    legacy = reverb.reverberation_power(provider, RX, 0.0, TIMES, MU_DB)
    got = reverb_bistatic.directional_power(provider, RX, RX, 0.0, TIMES, sectors(6), frequency=F, mu_db=MU_DB,
                                            n_elevation=4)
    assert got["doppler"].max() == 0.0 and got["doppler"].min() == 0.0
    per_sector = got["power"].sum(axis=1)                         # elevation bins add up
    for m in range(6):
        ratio = 10 * np.log10(blocks(per_sector[m])[1:-1] / blocks(legacy)[1:-1])
        assert np.abs(ratio).max() < 0.25, ratio
        assert abs(ratio.mean()) < 0.03, ratio


def test_bistatic_power_matches_an_integral_over_a_cartesian_grid():
    """I(t) = integral over the boundary of mu sin(g1) sin(g2) / (R1^2 R2^2), binned by
    (R1 + R2) / c: evaluated here on a square grid, which shares nothing with the polar
    grid, the range table or the time binning of the code."""
    got = reverb_bistatic.directional_power(FreeFieldProvider(C), TX, RX, 0.0, TIMES, sectors(180), frequency=F,
                                            mu_db=MU_DB)
    total = got["power"].sum(axis=(0, 1)) * (2 * np.pi / 180)
    h = 0.5
    axis = np.arange(-500.0, 650.0, h) + 0.5 * h
    x, y = np.meshgrid(axis, axis, indexing="ij")
    r1 = np.sqrt((x - TX[0]) ** 2 + (y - TX[1]) ** 2 + TX[2] ** 2)
    r2 = np.sqrt((x - RX[0]) ** 2 + (y - RX[1]) ** 2 + RX[2] ** 2)
    p = 10 ** (MU_DB / 10) * (-TX[2] / r1) * (-RX[2] / r2) / (r1 * r2) ** 2 * h * h
    idx = np.rint(((r1 + r2) / C - TIMES[0]) / 1e-3).astype(int).ravel()
    ok = (idx >= 0) & (idx < len(TIMES))
    reference = np.bincount(idx[ok], weights=p.ravel()[ok], minlength=len(TIMES))
    # Nothing arrives before the shortest path by way of the boundary (the specular point).
    first = np.hypot(np.hypot(TX[0], TX[1]), TX[2] + RX[2]) / C
    assert total[TIMES < first - 1e-3].max() == 0.0
    assert total[TIMES > first + 1e-3].min() > 0.0
    sel = TIMES > first + 5e-3
    ratio = 10 * np.log10(total[sel] / reference[sel])
    assert np.abs(ratio[:-2]).max() < 0.15, (ratio.min(), ratio.max())


def test_bistatic_power_is_reciprocal():
    """Omnidirectional platforms over a boundary with a symmetric strength: exchanging
    transmitter and receiver leaves the reverberation power unchanged. The cells are laid
    out about the receiver, so the two calculations use different grids."""
    kw = dict(frequency=F, mu_db=MU_DB)
    a = reverb_bistatic.directional_power(FreeFieldProvider(C), TX, RX, 0.0, TIMES, sectors(180), **kw)
    b = reverb_bistatic.directional_power(FreeFieldProvider(C), RX, TX, 0.0, TIMES, sectors(180), **kw)
    pa, pb = a["power"].sum(axis=(0, 1)), b["power"].sum(axis=(0, 1))
    sel = pa > 0
    assert sel.sum() > 250
    ratio = 10 * np.log10(blocks(pa[sel], 5) / blocks(pb[sel], 5))
    assert np.abs(ratio[2:-1]).max() < 0.1, ratio


def test_the_iso_time_locus_is_an_ellipse():
    """A small scattering patch returns at (|S - P| + |P - R|) / c, wherever it is: the
    cells of one two-way time lie on an ellipse about the two platforms, not on a circle
    about the receiver."""
    for centre in ([60.0, -80.0], [180.0, 90.0], [-120.0, 10.0]):
        p = np.array([centre[0], centre[1], 0.0])
        expected = (np.linalg.norm(p - TX) + np.linalg.norm(p - RX)) / C
        monostatic = 2.0 * np.linalg.norm(p - RX) / C
        assert abs(expected - monostatic) > 0.01
        got = reverb_bistatic.directional_power(FreeFieldProvider(C), TX, RX, 0.0, TIMES, sectors(360), frequency=F,
                                                mu_db=MU_DB, strength_map=patch_map(centre, 2.0))
        power = got["power"].sum(axis=(0, 1))
        assert (power * TIMES).sum() / power.sum() == pytest.approx(expected, abs=5e-4)     # half a bin
        # And the level: mu sin(g1) sin(g2) x area / (R1^2 R2^2), the patch's area 2 pi radius^2.
        r1, r2 = np.linalg.norm(p - TX), np.linalg.norm(p - RX)
        level = 10 ** (MU_DB / 10) * (-TX[2] / r1) * (-RX[2] / r2) * 2 * np.pi * 2.0**2 / (r1 * r2) ** 2
        assert power.sum() * (2 * np.pi / 360) == pytest.approx(level, rel=0.02)


def test_strength_is_given_the_true_angles_and_bistatic_azimuth():
    """The scattering strength is evaluated at the grazing angle of each leg and at the
    angle between the horizontal directions of the incident and scattered waves."""
    seen = {}

    def strength(g_in, g_out, azimuth):
        seen["args"] = np.broadcast_arrays(g_in, g_out, azimuth)
        return np.ones(np.broadcast(g_in, g_out, azimuth).shape)

    centre = np.array([60.0, -80.0, 0.0])
    reverb_bistatic.directional_power(FreeFieldProvider(C), TX, RX, 0.0, TIMES, sectors(720), frequency=F,
                                      strength=strength, min_range=99.5, max_range=100.5, ring_width=1.0)
    g_in, g_out, az = (a.copy() for a in seen["args"])
    # The cell nearest the test point (range 100 m from the receiver).
    m = int(np.argmin(np.abs(np.angle(np.exp(1j * (sectors(720) - np.arctan2(-80.0, 60.0)))))))
    cell = np.array([100.0 * np.cos(sectors(720)[m]), 100.0 * np.sin(sectors(720)[m]), 0.0])
    assert np.linalg.norm(cell - centre) < 0.5
    to_cell, back = cell - TX, RX - cell
    assert g_in[m, 0] == pytest.approx(np.arcsin(-TX[2] / np.linalg.norm(to_cell)), abs=2e-3)
    assert g_out[m, 0] == pytest.approx(np.arcsin(-RX[2] / np.linalg.norm(back)), abs=1e-9)
    cos_az = (to_cell[:2] @ back[:2]) / (np.linalg.norm(to_cell[:2]) * np.linalg.norm(back[:2]))
    assert az[m, 0] == pytest.approx(np.arccos(cos_az), abs=1e-9)
    # Monostatic: backscatter in every cell.
    reverb_bistatic.directional_power(FreeFieldProvider(C), RX, RX, 0.0, TIMES, sectors(12), frequency=F,
                                      strength=strength)
    assert np.allclose(seen["args"][2], np.pi)
    assert reverb_bistatic.takes_azimuth(strength) and not reverb_bistatic.takes_azimuth(lambda a, b: a)


def test_fluid_sediment_model_is_used_at_the_bistatic_azimuth():
    """sim/bottom.py's strength takes the bistatic azimuth: a patch's reverberation changes
    by the model's own ratio between that azimuth and backscatter."""
    from gpusonar.sim import bottom
    params = dict(rho=1.8, nu=1.1, delta=0.01, w2=2e-5, gamma2=3.0, sigma2=0.002)
    fn = lambda g_in, g_out, azimuth=np.pi: bottom.strength(g_in, g_out, F, C, azimuth=azimuth, **params)  # noqa: E731
    back_only = lambda g_in, g_out: bottom.strength(g_in, g_out, F, C, **params)                             # noqa: E731
    changes = []
    for where in ([75.0, 100.0], [100.0, -40.0], [40.0, 60.0]):
        kw = dict(frequency=F, strength_map=patch_map(where, 1.0))
        a = reverb_bistatic.directional_power(FreeFieldProvider(C), TX, RX, 0.0, TIMES, sectors(720), strength=fn, **kw)
        b = reverb_bistatic.directional_power(FreeFieldProvider(C), TX, RX, 0.0, TIMES, sectors(720),
                                              strength=back_only, **kw)
        centre = np.array([where[0], where[1], 0.0])
        g1 = np.arcsin(-TX[2] / np.linalg.norm(centre - TX))
        g2 = np.arcsin(-RX[2] / np.linalg.norm(centre - RX))
        to_cell, back = (centre - TX)[:2], (RX - centre)[:2]
        azimuth = np.arccos(to_cell @ back / (np.linalg.norm(to_cell) * np.linalg.norm(back)))
        expected = bottom.strength(g1, g2, F, C, azimuth=azimuth, **params) / bottom.strength(g1, g2, F, C, **params)
        changes.append(10 * np.log10(expected))
        assert a["power"].sum() / b["power"].sum() == pytest.approx(float(expected), rel=0.005)
    assert min(np.abs(changes)) > 0.3                                  # 7 % or more: the azimuth is what is tested


# --- Doppler -----------------------------------------------------------------------------------

def exact_doppler_factor(tx, rx, point, v_tx, v_rx):
    """TR 0702 eq. (71): Gamma = (1 - V_R . S_R) / (1 + V_S . S_S), S the gradient of the
    travel time with respect to that platform's position (straight rays: the unit vector
    from the scatterer to the platform over c)."""
    s_r = (rx - point) / np.linalg.norm(rx - point) / C
    s_s = (tx - point) / np.linalg.norm(tx - point) / C
    return (1.0 - v_rx @ s_r) / (1.0 + v_tx @ s_s)


def test_doppler_shift_is_the_closed_form_factor_to_first_order():
    rng = np.random.default_rng(0)
    for _ in range(20):
        tx, rx, point = rng.uniform(-200, 200, 3), rng.uniform(-200, 200, 3), rng.uniform(-200, 200, 3)
        v_tx, v_rx = rng.uniform(-10, 10, 3), rng.uniform(-10, 10, 3)
        u_s, u_r = (point - tx) / np.linalg.norm(point - tx), (point - rx) / np.linalg.norm(point - rx)
        shift = reverb_bistatic.doppler_shift(F, C, v_tx, u_s, v_rx, u_r)
        exact = (exact_doppler_factor(tx, rx, point, v_tx, v_rx) - 1.0) * F
        # Second order: (v / c)^2 f, 1.2 Hz at 10 m/s and 27.5 kHz, against shifts of hundreds of Hz.
        assert abs(shift - exact) < 2.0 * (10.0 * np.sqrt(3) / C) ** 2 * F
    # One platform at a time, closing head-on at 5 m/s: f v / c each.
    assert reverb_bistatic.doppler_shift(F, C, [5, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(F * 5 / C)
    assert reverb_bistatic.doppler_shift(F, C, [0, 0, 0], [1, 0, 0], [0, -5, 0], [0, -1, 0]) == pytest.approx(F * 5 / C)


def test_each_cell_takes_the_doppler_of_both_platforms():
    """The Doppler of a patch's reverberation is the closed-form factor at the patch, with
    the transmitter's velocity along the outgoing ray and the receiver's along the incoming
    one (elevation included: the platforms are well above the patch)."""
    v_tx, v_rx = np.array([4.0, -3.0, 0.5]), np.array([-2.0, 5.0, -1.0])
    for centre in ([60.0, -80.0], [180.0, 90.0], [10.0, 15.0]):
        point = np.array([centre[0], centre[1], 0.0])
        got = reverb_bistatic.directional_power(FreeFieldProvider(C), TX, RX, 0.0, np.arange(0.01, 0.4, 1e-3),
                                                sectors(360), frequency=F, mu_db=MU_DB, tx_velocity=v_tx,
                                                rx_velocity=v_rx, strength_map=patch_map(centre, 1.0), n_elevation=9)
        w = got["power"].sum(axis=2)
        doppler = (w * got["doppler"]).sum() / w.sum()
        exact = (exact_doppler_factor(TX, RX, point, v_tx, v_rx) - 1.0) * F
        assert doppler == pytest.approx(exact, abs=1.0), centre
        # The arrival direction of that power is the direction of the patch from the receiver.
        direction = (w[..., None] * got["direction"]).sum(axis=(0, 1)) / w.sum()
        assert direction / np.linalg.norm(direction) == pytest.approx((point - RX) / np.linalg.norm(point - RX), abs=0.02)


def bistatic_ping(tx=TX, rx=RX, v_tx=(0.0, 0.0, 0.0), v_rx=(0.0, 0.0, 0.0), name="popoto_8", ranges=(25.0, 230.0),
                  fs=20e3, duration=2e-3, waveform=None, scene=None, kind="baseband"):
    wf = {"type": "cw", "f_start": F, "f_stop": F, "duration": duration, "window_alpha": 0.0,
          "source_level_db": None, "amplitude": 1.0}
    wf.update(waveform or {})
    sonar = SonarModel.load(name, {
        "receive": {"kind": kind, "sample_rate": fs, "window": {"range": list(ranges)}, "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": wf}})
    return simulate_ping(sonar, make_pose(position=rx, velocity=v_rx), scene or Scene(),
                         environment={"sound_speed": C}, absorption_model="none", adc=False,
                         tx_pose=make_pose(position=tx, velocity=v_tx))


@pytest.mark.parametrize("engine", ["numpy", "cpp"])
def test_generated_series_has_the_doppler_of_both_platforms(engine):
    """In the time series: the reverberation of a patch, from a long tone, sits at the
    closed-form frequency shift. A stationary receiver with a moving transmitter is shifted
    too (the monostatic code takes the receiver's velocity for both legs)."""
    centre = [60.0, -80.0]
    point = np.array([centre[0], centre[1], 0.0])
    for v_tx, v_rx in (((-4.0, -6.0, 0.0), (0.0, 0.0, 0.0)), ((-4.0, -6.0, 0.0), (3.0, -5.0, 1.0)),
                       ((4.0, 6.0, 0.0), (3.0, -5.0, 1.0))):
        rec = bistatic_ping(v_tx=v_tx, v_rx=v_rx, duration=0.05, ranges=(60.0, 190.0))
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(1), n_azimuth=360,
                                 geometry="bistatic", strength_map=patch_map(centre, 1.0), engine=engine)
        x = rec.samples.array
        fs = rec.samples.sample_rate
        spectrum = (np.abs(np.fft.fft(x * np.hanning(x.shape[1]), 1 << 16, axis=1)) ** 2).sum(axis=0)
        f = np.fft.fftfreq(1 << 16, 1 / fs)
        exact = (exact_doppler_factor(TX, RX, point, np.array(v_tx), np.array(v_rx)) - 1.0) * F
        # Centroid of the main lobe (the burst is 50 ms long: its spectrum has sidelobes).
        near = np.abs(f - f[np.argmax(spectrum)]) < 15.0
        centroid = (f[near] * spectrum[near]).sum() / spectrum[near].sum()
        assert abs(exact) > 15.0
        assert centroid == pytest.approx(exact, abs=3.0)              # the tone is 20 Hz wide


# --- beam patterns -----------------------------------------------------------------------------

def test_beam_patterns_use_the_elevation_at_the_platform_for_bounce_paths():
    """A path that leaves the sonar upwards, bounces off the surface and reaches the bottom
    travelling down is weighted by the pattern towards its launch direction (up), not by
    its direction at the bottom. A transmit pattern that passes only downward launches must
    remove exactly the pairs whose outgoing leg is the surface bounce, in both codes."""
    provider = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1, surface_coefficient=-1.0)
    source = np.array([0.0, 0.0, 50.0])
    times = np.arange(0.15, 0.5, 1e-3)
    down = lambda d: (d[..., 2] > 0.0).astype(float)             # noqa: E731
    one = lambda d: np.ones(d.shape[:-1])                         # noqa: E731
    # Independent sum over pairs from the eigenrays themselves.
    dr = 0.25 * C * 1e-3
    rings = np.arange(0.5 * dr, 0.5 * C * (times[-1] + 1e-3), dr)
    es = provider.eigenrays(source, np.column_stack([rings, np.zeros_like(rings), np.full_like(rings, 100.0)]))
    assert es.delay.shape[1] == 2 and np.all(es.launch[:, 0, 2] > 0) and np.all(es.launch[:, 1, 2] < 0)
    assert np.all(es.arrival[:, :, 2] > 0)                      # both reach the bottom travelling down
    a2, sin_g = np.abs(es.amplitude) ** 2, np.abs(es.arrival[:, :, 2])
    expected = {"all": np.zeros(len(times)), "down": np.zeros(len(times))}
    for i in range(2):
        for j in range(2):
            idx = np.rint((es.delay[:, i] + es.delay[:, j] - times[0]) / 1e-3).astype(int)
            ok = (idx >= 0) & (idx < len(times))
            p = 10 ** (MU_DB / 10) * a2[:, i] * a2[:, j] * sin_g[:, i] * sin_g[:, j] * rings * dr
            np.add.at(expected["all"], idx[ok], p[ok])
            if i == 0:
                np.add.at(expected["down"], idx[ok], p[ok])
    assert 0.15 < expected["down"][100:].sum() / expected["all"][100:].sum() < 0.6     # the pattern removes a good part
    az = sectors(4)
    legacy = reverb.reverberation_power(provider, source, 100.0, times, MU_DB, pattern=(az, down, one))
    new = reverb_bistatic.directional_power(provider, source, source, 100.0, times, az, frequency=F, mu_db=MU_DB,
                                            tx_gain=down)["power"].sum(axis=1)
    for m in range(4):
        np.testing.assert_allclose(legacy[m], expected["down"], rtol=1e-9)
        assert np.abs(10 * np.log10(blocks(new[m])[1:-1] / blocks(expected["down"])[1:-1])).max() < 0.25
    # The same pattern on the receive leg removes the pairs that return by the surface.
    legacy_rx = reverb.reverberation_power(provider, source, 100.0, times, MU_DB, pattern=(az, one, down))
    np.testing.assert_allclose(legacy_rx[0], expected["down"], rtol=1e-9)       # symmetric pairs


def test_components_carry_the_arrival_elevation():
    """Near the sonar the bottom return arrives from below: each component's direction has
    the elevation of its rays at the receiver and the azimuth of its sector."""
    az = sectors(8)
    got = reverb_bistatic.directional_power(FreeFieldProvider(C), RX, RX, 0.0, np.arange(0.02, 0.4, 1e-3), az,
                                            frequency=F, mu_db=MU_DB, n_elevation=18)
    w = got["power"].sum(axis=2)
    assert (w[:, :9] == 0).all() and (w[:, 9:] > 0).all()        # nothing from above the horizontal
    for e in range(9, 18):
        d = got["direction"][3, e]
        elevation = np.degrees(np.arcsin(d[2]))
        assert (e - 9) * 10.0 <= elevation <= (e - 8) * 10.0
        assert np.arctan2(d[1], d[0]) == pytest.approx(az[3], abs=1e-9)
        assert np.linalg.norm(d) == pytest.approx(1.0)


# --- the time series ---------------------------------------------------------------------------

def test_bistatic_series_level_follows_the_cartesian_integral():
    """The generated reverberation of a bistatic ping has the level of the integral over
    the boundary (unit CW pulse of length T: intensity = power per second x T)."""
    intensity = 0.0
    for seed in range(6):
        rec = bistatic_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(seed),
                                 geometry="bistatic")
        intensity = intensity + (np.abs(rec.samples.array) ** 2).mean(axis=0) / 6
    s = rec.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    h = 1.0
    axis = np.arange(-400.0, 550.0, h) + 0.5 * h
    x, y = np.meshgrid(axis, axis, indexing="ij")
    r1 = np.sqrt((x - TX[0]) ** 2 + (y - TX[1]) ** 2 + TX[2] ** 2)
    r2 = np.sqrt(x**2 + y**2 + RX[2] ** 2)
    p = 10 ** (MU_DB / 10) * (-TX[2] / r1) * (-RX[2] / r2) / (r1 * r2) ** 2 * h * h
    # A scatterer at two-way time u fills the samples from u to u + T.
    edges = np.linspace(0.12, 0.30, 13)
    diffs = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t < b)
        tau = ((r1 + r2) / C).ravel()
        overlap = np.clip(np.minimum(b, tau + 2e-3) - np.maximum(a, tau), 0.0, None)
        predicted = (p.ravel() * overlap).sum() / (b - a)
        diffs.append(10 * np.log10(intensity[sel].mean() / predicted))
    diffs = np.array(diffs)
    assert np.abs(diffs).max() < 2.0, diffs
    assert abs(diffs.mean()) < 0.7, diffs


def test_bistatic_scattering_function_agrees_with_point_scatterers():
    """The point-scatterer method (Lambert patches, two-way eigenray pairs from the
    transmitter's and the receiver's own positions) is a second realisation of the same
    bistatic model."""
    density, radius = 1.0, 330.0
    point = []
    for seed in (11, 12, 13):
        rng = np.random.default_rng(seed)
        pts, amp = seafloor_patch((-radius + 75, radius + 75), (-radius, radius), density, depth=0.0,
                                  rms_amplitude=lambert_rms_amplitude(MU_DB, density), rng=rng)
        point.append(bistatic_ping(scene=Scene().add_points(pts, amp, lambert=True)).samples.array)
    sf = []
    for seed in (21, 22, 23):
        rec = bistatic_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(seed),
                                 geometry="bistatic")
        sf.append(rec.samples.array)
    a = (np.abs(np.concatenate(point)) ** 2).mean(axis=0)
    b = (np.abs(np.concatenate(sf)) ** 2).mean(axis=0)
    span = slice(1800, 4200)                                     # 0.12 to 0.24 s: after the specular onset
    ratio = 10 * np.log10(a[span].reshape(8, -1).mean(axis=1) / b[span].reshape(8, -1).mean(axis=1))
    assert np.abs(ratio).max() < 2.0, ratio
    assert abs(ratio.mean()) < 1.0, ratio


def relative_difference(a, b):
    d = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
    return d.max() / np.abs(b).max(), np.sqrt(np.mean(d**2) / np.mean(np.abs(b) ** 2))


ENGINE_CASES = {
    "moving_both": (dict(v_tx=(6.0, -4.0, 0.5), v_rx=(-3.0, 5.0, -0.5)), dict(n_azimuth=60)),
    "uniform_line_decimated": (dict(name="flank_32", ranges=(60.0, 500.0), fs=16e3, duration=0.02, v_rx=(3.0, 1.0, 0.0)),
                               dict(n_azimuth=90, n_elevation=3)),
    "patchy_boundary": (dict(), dict(n_azimuth=48, strength_map=patch_map([60.0, -80.0], 30.0), azimuth_subsamples=2)),
    "k_distributed": (dict(v_tx=(2.0, 0.0, 0.0)), dict(n_azimuth=40, statistics={"model": "k", "shape": 0.7})),
    "real_passband": (dict(kind="real", fs=100e3, ranges=(60.0, 120.0)), dict(n_azimuth=30)),
}


@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
@pytest.mark.parametrize("name", list(ENGINE_CASES))
def test_bistatic_engines_match_sample_for_sample(name):
    """Components (a direction with elevation and a Doppler shift each) on the C++ generator
    against the NumPy one, from the same seed."""
    assert reverb.cpp_has_components()
    ping_kw, reverb_kw = ENGINE_CASES[name]
    out = {}
    for engine in ("cpp", "numpy"):
        rec = bistatic_ping(**ping_kw)
        rng = np.random.default_rng(3)
        power = reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, rng, engine=engine,
                                         geometry="bistatic", **reverb_kw)
        out[engine] = (rec.samples.array, power, rng.bit_generator.state)
    assert np.abs(out["numpy"][0]).max() > 0.0
    worst, rms = relative_difference(out["cpp"][0], out["numpy"][0])
    assert worst < 5e-6 and rms < 5e-6, (worst, rms)
    np.testing.assert_array_equal(out["cpp"][1], out["numpy"][1])
    assert out["cpp"][2] == out["numpy"][2]


@pytest.mark.skipif(not core.cuda_available(), reason="no CUDA device")
@pytest.mark.parametrize("name", list(ENGINE_CASES))
def test_cuda_bistatic_reverberation_matches_the_host(name):
    """Components reach the device as the same per-sector spectra and leads as azimuth
    sectors do (the steering kernel is unchanged), so the device must agree with the host
    as in test_sim_cpp.py."""
    ping_kw, reverb_kw = ENGINE_CASES[name]
    out = {}
    for backend in (core.Backend.Host, core.Backend.Cuda):
        rec = bistatic_ping(**ping_kw)
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(3), engine="cpp",
                                 backend=backend, geometry="bistatic", **reverb_kw)
        out[backend] = rec.samples.array
    worst, rms = relative_difference(out[core.Backend.Cuda], out[core.Backend.Host])
    assert worst < 1e-4 and rms < 1e-4, (worst, rms)


def test_an_older_extension_falls_back_to_numpy_visibly(monkeypatch):
    """A build of the C++ generator without components must not be used silently."""
    monkeypatch.setattr(reverb, "cpp_has_components", lambda: False)
    monkeypatch.setattr(reverb, "_series_cpp", lambda *a, **k: pytest.fail("the C++ generator was used"))
    rec = bistatic_ping()
    if sim_engine.HAVE_CPP:
        with pytest.warns(RuntimeWarning, match="falling back to the NumPy engine"):
            reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(0), n_azimuth=12,
                                     geometry="bistatic", engine="cpp")
    else:
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(0), n_azimuth=12,
                                 geometry="bistatic")
    assert np.abs(rec.samples.array).max() > 0.0


# --- scenario keys and defaults ----------------------------------------------------------------

def test_default_geometry_is_unchanged_and_new_keys_need_bistatic():
    """Without the new keys nothing changes: the transmitter's own position and velocity
    are not used (the monostatic limits), and the new options are refused."""
    a, b = bistatic_ping(), bistatic_ping(tx=RX, v_tx=(5.0, 0.0, 0.0))
    for rec in (a, b):
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(4), n_azimuth=24)
    np.testing.assert_array_equal(a.samples.array, b.samples.array)
    with pytest.raises(ValueError, match="bistatic"):
        reverb.add_reverberation(a, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(4),
                                 strength_map=patch_map([0, 0], 1.0))
    with pytest.raises(ValueError, match="geometry"):
        reverb.add_reverberation(a, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(4), geometry="circular")


def test_scenario_item_keys_reach_the_generator():
    """`geometry`, `strength_map` (a table) and `statistics` in a `reverberation` item."""
    sonar = SonarModel.load("popoto_8", {
        "receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": [25.0, 230.0]}, "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": F, "f_stop": F, "duration": 2e-3,
                                                          "window_alpha": 0.0, "source_level_db": None,
                                                          "amplitude": 1.0}}})

    def run(item, tx=TX):
        rec = simulate_ping(sonar, make_pose(position=RX), Scene(), environment={"sound_speed": C}, adc=False,
                            absorption_model="none", tx_pose=make_pose(position=tx),
                            reverberation=[dict({"boundary_z": 0.0, "mu_db": MU_DB, "n_azimuth": 36}, **item)],
                            rng=np.random.default_rng(6))
        return rec.samples.array

    plain = run({})
    assert np.array_equal(plain, run({}, tx=RX))                  # default: the transmitter's place is not used
    bistatic = run({"geometry": "bistatic"})
    assert not np.array_equal(plain, bistatic)
    # A table: 20 dB up everywhere, by range only and by range and azimuth.
    for table in ({"range": [0.0, 1000.0], "offset_db": [20.0, 20.0]},
                  {"range": [0.0, 1000.0], "azimuth_deg": [0.0, 90.0, 180.0, 270.0], "offset_db": [[20.0, 20.0]] * 4,
                   "origin": [30.0, -10.0]}):
        raised = run({"geometry": "bistatic", "strength_map": table})
        assert np.abs(raised - 10.0 * bistatic).max() < 1e-4 * np.abs(raised).max()
    k = run({"geometry": "bistatic", "statistics": {"model": "k", "shape": 0.5}})
    assert not np.array_equal(k, bistatic)
    assert np.array_equal(run({"geometry": "bistatic", "statistics": {"model": "rayleigh"}}), bistatic)
    with pytest.raises(ValueError, match="shape"):
        run({"statistics": {"model": "k"}})


def test_strength_table_interpolates_in_range_and_azimuth():
    f = reverb_bistatic.table_strength_map([100.0, 200.0], [[0.0, 10.0], [20.0, 30.0]], azimuth_deg=[0.0, 180.0],
                                           origin=(5.0, 5.0))
    db = lambda x, y: 10 * np.log10(f(np.array(x + 5.0), np.array(y + 5.0)))    # noqa: E731
    assert db(100.0, 0.0) == pytest.approx(0.0) and db(200.0, 0.0) == pytest.approx(10.0)
    assert db(150.0, 0.0) == pytest.approx(5.0)
    assert db(-150.0, 0.0) == pytest.approx(25.0, abs=1e-6)
    assert db(0.0, 150.0) == pytest.approx(15.0)                 # 90 degrees: halfway round
    assert db(0.0, -150.0) == pytest.approx(15.0)                # 270 degrees: halfway round the other way
    assert db(50.0, 0.0) == pytest.approx(0.0) and db(900.0, 0.0) == pytest.approx(10.0)   # held beyond the ends
    by_range = reverb_bistatic.table_strength_map([0.0, 10.0], [0.0, -10.0])
    assert 10 * np.log10(by_range(np.array(3.0), np.array(4.0))) == pytest.approx(-5.0)
