"""Reverberation from scatterers that move (`scatterer_motion`, sim/reverb_bistatic.py):
a mean velocity (TR 0702 eq. 72) and a Doppler spread from random motion, against closed
forms. Static platform with moving scatterers, moving platform with static scatterers, and
both; the coherence time of eq. (96); unchanged power; zero motion changes nothing.
"""

import numpy as np
import pytest

from gpusonar import acoustics
from gpusonar.sim import FreeFieldProvider, reverb, reverb_bistatic
from gpusonar.sim import engine as sim_engine

from test_reverb_bistatic import C, F, MU_DB, RX, TX, bistatic_ping, exact_doppler_factor, patch_map, sectors
from test_reverb_statistics import ENGINES, generate

TIMES = np.arange(0.01, 0.4, 1e-3)
FF = FreeFieldProvider(C)


def exact_scatterer_factor(tx, rx, point, v):
    """TR 0702 eq. (72): gamma = (1 - V_T . S_Tq) / (1 + V_T . S_Tp), the slowness vectors at
    the scatterer being the gradients of each leg's travel time with respect to its position."""
    s_q = (point - tx) / np.linalg.norm(point - tx) / C
    s_p = (point - rx) / np.linalg.norm(point - rx) / C
    return (1.0 - v @ s_q) / (1.0 + v @ s_p)


def test_scatterer_doppler_is_eq_72_to_first_order():
    rng = np.random.default_rng(0)
    for _ in range(20):
        tx, rx, point = rng.uniform(-200, 200, 3), rng.uniform(-200, 200, 3), rng.uniform(-200, 200, 3)
        v = rng.uniform(-5, 5, 3)
        a_q, a_p = (point - tx) / np.linalg.norm(point - tx), (point - rx) / np.linalg.norm(point - rx)
        shift = reverb_bistatic.scatterer_doppler(F, C, v, a_q, a_p)
        exact = (exact_scatterer_factor(tx, rx, point, v) - 1.0) * F
        assert abs(shift - exact) < 2.0 * (5.0 * np.sqrt(3) / C) ** 2 * F        # second order: 1.2 Hz
    # Backscatter from a scatterer approaching at 1 m/s: +2 f v / c.
    assert reverb_bistatic.scatterer_doppler(F, C, [-1, 0, 0], [1, 0, 0], [1, 0, 0]) == pytest.approx(2 * F / C)


def patch_components(centre, **kw):
    got = reverb_bistatic.directional_power(FF, TX, RX, 0.0, TIMES, sectors(360), frequency=F, mu_db=MU_DB,
                                            strength_map=patch_map(centre, 1.0), n_elevation=9, **kw)
    w = got["power"].sum(axis=2)
    return (w * got["doppler"]).sum() / w.sum(), np.sqrt((w * got["spread"] ** 2).sum() / w.sum()), got


@pytest.mark.parametrize("centre", [[60.0, -80.0], [180.0, 90.0], [10.0, 15.0]])
def test_cell_doppler_with_moving_scatterers_and_platforms(centre):
    """The three cases: scatterers only, platforms only, both. The total factor is the
    product Gamma_pq gamma of eqs. (71) and (72)."""
    point = np.array([centre[0], centre[1], 0.0])
    v_s, v_tx, v_rx = np.array([1.5, -0.8, 0.3]), np.array([4.0, -3.0, 0.5]), np.array([-2.0, 5.0, -1.0])
    zero = np.zeros(3)
    gamma = exact_scatterer_factor(TX, RX, point, v_s)
    platform = exact_doppler_factor(TX, RX, point, v_tx, v_rx)
    for kw, factor in ((dict(scatterer_velocity=v_s), gamma),
                       (dict(tx_velocity=v_tx, rx_velocity=v_rx), platform),
                       (dict(scatterer_velocity=v_s, tx_velocity=v_tx, rx_velocity=v_rx), gamma * platform)):
        doppler, spread, _ = patch_components(centre, **kw)
        assert abs((factor - 1.0) * F) > 1.0
        assert doppler == pytest.approx((factor - 1.0) * F, abs=1.0)
        assert spread == 0.0
    assert patch_components(centre, scatterer_velocity=zero)[0] == 0.0


@pytest.mark.parametrize("centre", [[60.0, -80.0], [10.0, 15.0]])
def test_doppler_spread_of_a_cell_follows_the_geometry(centre):
    """Random scatterer velocities v give each scatterer the shift -f / c v . k,
    k = a_in + a_back. Isotropic velocities (sigma on each axis): sigma_f = f sigma |k| / c.
    A heaving boundary (vertical only): sigma_f = f sigma_w |k_z| / c, which is
    f sigma_w (sin g_in + sin g_out) / c."""
    point = np.array([centre[0], centre[1], 0.0])
    a_q, a_p = (point - TX) / np.linalg.norm(point - TX), (point - RX) / np.linalg.norm(point - RX)
    k = a_q + a_p
    _, iso, _ = patch_components(centre, scatterer_rms_speed=0.2)
    assert iso == pytest.approx(F * 0.2 * np.linalg.norm(k) / C, rel=0.01)
    _, heave, _ = patch_components(centre, scatterer_rms_vertical_speed=0.3)
    sin_in, sin_out = -TX[2] / np.linalg.norm(point - TX), -RX[2] / np.linalg.norm(point - RX)
    assert heave == pytest.approx(F * 0.3 * (sin_in + sin_out) / C, rel=0.01)
    _, both, _ = patch_components(centre, scatterer_rms_speed=0.2, scatterer_rms_vertical_speed=0.3)
    assert both == pytest.approx(np.hypot(iso, heave), rel=1e-6)
    # Monostatic backscatter: |k| = 2 whatever the grazing angle.
    got = reverb_bistatic.directional_power(FF, RX, RX, 0.0, TIMES, sectors(12), frequency=F, mu_db=MU_DB,
                                            scatterer_rms_speed=0.2, n_elevation=6)
    assert got["spread"][got["power"].sum(axis=2) > 0] == pytest.approx(2.0 * F * 0.2 / C, rel=1e-5)


def test_doppler_lines_are_a_gaussian_of_the_right_width():
    for spread, pulse in ((25.0, 0.05), (3.0, 0.4), (200.0, 0.02)):
        offsets, weights = reverb_bistatic.doppler_lines(spread, pulse)
        assert weights.sum() == pytest.approx(1.0) and (weights * offsets).sum() == pytest.approx(0.0, abs=1e-9)
        assert (weights * offsets**2).sum() == pytest.approx(spread**2)
        assert np.diff(offsets).max() <= 0.55 / pulse or len(offsets) == 41
        assert (weights * offsets**4).sum() == pytest.approx(3.0 * spread**4, rel=0.12)       # Gaussian kurtosis
    # Nothing to resolve: one line, and no spread at all is one line at zero.
    for spread in (0.0, 0.5):
        offsets, weights = reverb_bistatic.doppler_lines(spread, 0.05)
        assert len(offsets) == 1 and offsets[0] == 0.0 and weights[0] == 1.0


@pytest.mark.parametrize("engine", ENGINES)
def test_scatterer_doppler_spread_gives_the_coherence_time(engine):
    """Eq. (96) with the spread from scatterer motion: backscatter from scatterers of rms
    speed sigma_v has sigma_gamma = 2 sigma_v / c, so a tone decorrelates over
    sigma_t = 1 / (2 pi f sigma_gamma) = c / (4 pi f sigma_v). The tone's own finite length
    multiplies the coherence by 1 - |s| / T."""
    fs, n, fc, sigma_v, tone = 400.0, 4096, 10e3, 0.03, 0.5
    sigma_f = 2.0 * fc * sigma_v / C                                 # 0.4 Hz
    offsets, weights = reverb_bistatic.doppler_lines(sigma_f, tone)
    assert len(offsets) >= 5
    density = weights[:, None] * np.ones((1, 2))
    direction = np.tile([1.0, 0.0, 0.0], (len(offsets), 1))
    lags = np.arange(0, 81)
    rng = np.random.default_rng(3)
    cov = 0.0
    for _ in range(12):
        x = generate(engine, rng, density, np.array([-1.0, 1e6]), n_low=n, fs=fs, n_azimuth=1, fc=fc,
                     pulse=np.ones(int(tone * fs)), components=(direction, offsets))[0]
        cov = cov + np.array([np.mean(x * np.conj(np.roll(x, -k))) for k in lags])
    coherence = cov / cov[0].real
    s = lags / fs
    sigma_time = C / (4.0 * np.pi * fc * sigma_v)                    # 398 ms
    assert sigma_time == pytest.approx(1.0 / (2.0 * np.pi * sigma_f))
    expected = np.exp(-0.5 * (s / sigma_time) ** 2) * (1.0 - s / tone)
    assert np.abs(coherence - expected).max() < 0.08, np.abs(coherence - expected).max()
    # A faster sea: the width itself.
    sigma_v = 0.3
    sigma_f = 2.0 * fc * sigma_v / C
    offsets, weights = reverb_bistatic.doppler_lines(sigma_f, tone)
    density = weights[:, None] * np.ones((1, 2))
    direction = np.tile([1.0, 0.0, 0.0], (len(offsets), 1))
    cov = 0.0
    for _ in range(12):
        x = generate(engine, rng, density, np.array([-1.0, 1e6]), n_low=n, fs=fs, n_azimuth=1, fc=fc,
                     pulse=np.ones(int(tone * fs)), components=(direction, offsets))[0]
        cov = cov + np.array([np.mean(x * np.conj(np.roll(x, -k))) for k in lags])
    coherence = np.abs(cov / cov[0].real) / (1.0 - s / tone)
    measured = np.interp(np.exp(-0.5), coherence[:40][::-1], s[:40][::-1])
    assert measured == pytest.approx(C / (4.0 * np.pi * fc * sigma_v), rel=0.1)      # 39.8 ms


def spectrum_of(rec):
    x = rec.samples.array
    spectrum = (np.abs(np.fft.fft(x, 1 << 17, axis=1)) ** 2).sum(axis=0)
    return np.fft.fftfreq(1 << 17, 1 / rec.samples.sample_rate), spectrum


def centroid_and_width(f, spectrum, about, half_width):
    near = np.abs(f - about) < half_width
    mean = (f[near] * spectrum[near]).sum() / spectrum[near].sum()
    return mean, np.sqrt(((f[near] - mean) ** 2 * spectrum[near]).sum() / spectrum[near].sum())


@pytest.mark.parametrize("engine", ENGINES)
def test_series_centroid_and_width_in_the_three_cases(engine):
    """CW reverberation of a patch, 50 ms pulse with a cosine taper (its spectrum has a
    finite width, which adds in quadrature to the Doppler spread):

      static platforms, moving scatterers   centroid (gamma - 1) f, width sigma_f
      moving platforms, static scatterers   centroid (Gamma - 1) f, no extra width
      both                                  centroid (Gamma gamma - 1) f, width sigma_f
    """
    centre = [60.0, -80.0]
    point = np.array([centre[0], centre[1], 0.0])
    a_q, a_p = (point - TX) / np.linalg.norm(point - TX), (point - RX) / np.linalg.norm(point - RX)
    v_s, sigma_v = np.array([-2.0, 1.0, 0.0]), 1.2
    v_tx, v_rx = (-4.0, -6.0, 0.0), (3.0, -5.0, 1.0)
    sigma_f = F * sigma_v * np.linalg.norm(a_q + a_p) / C
    gamma = exact_scatterer_factor(TX, RX, point, v_s)
    platform = exact_doppler_factor(TX, RX, point, np.array(v_tx), np.array(v_rx))
    wf = {"window_alpha": 1.0}

    def run(seed, motion=None, **ping_kw):
        rec = bistatic_ping(duration=0.05, ranges=(60.0, 190.0), waveform=wf, **ping_kw)
        power = reverb.add_reverberation(rec, FF, 0.0, MU_DB, np.random.default_rng(seed), n_azimuth=360,
                                         geometry="bistatic", strength_map=patch_map(centre, 1.0), engine=engine,
                                         scatterer_motion=motion)
        return rec, power

    def averaged(motion=None, **ping_kw):
        total, energy = 0.0, 0.0
        for seed in range(48):
            rec, power = run(seed, motion, **ping_kw)
            f, sp = spectrum_of(rec)
            total, energy = total + sp, energy + float(np.sum(np.abs(rec.samples.array) ** 2))
        return f, total, energy / 48, power

    f, still, energy_still, power_still = averaged()
    _, own_width = centroid_and_width(f, still, 0.0, 150.0)
    assert sigma_f > 2.0 * own_width                                 # the spread is what is measured
    cases = {
        "scatterers": (dict(motion={"velocity": list(v_s), "rms_speed": sigma_v}), (gamma - 1.0) * F, sigma_f),
        "platforms": (dict(v_tx=v_tx, v_rx=v_rx), (platform - 1.0) * F, 0.0),
        "both": (dict(motion={"velocity": list(v_s), "rms_speed": sigma_v}, v_tx=v_tx, v_rx=v_rx),
                 (gamma * platform - 1.0) * F, sigma_f),
    }
    for name, (kw, shift, spread) in cases.items():
        f, sp, energy, power = averaged(**kw)
        mean, width = centroid_and_width(f, sp, shift, 150.0)
        assert abs(shift) > 20.0
        print(f"{engine} {name}: centroid {mean:+.2f} Hz (closed form {shift:+.2f}), width {width:.2f} Hz (closed form {np.hypot(own_width, spread):.2f}; pulse alone {own_width:.2f}), energy ratio {energy / energy_still:.3f}")
        assert mean == pytest.approx(shift, abs=3.0 if spread == 0.0 else 6.0), name
        assert width == pytest.approx(np.hypot(own_width, spread), rel=0.12, abs=0.5), name
        # The expected power is not changed by the motion. (The realised energy of a single
        # speckle cell scatters too much to test here: measured over 300 draws the ratio
        # with and without motion is 1.03 +- 0.06.)
        np.testing.assert_array_equal(power, power_still)


def test_zero_motion_changes_nothing_and_the_keys_are_checked():
    def run(**kw):
        rec = bistatic_ping(v_rx=(2.0, 0.0, 0.0))
        reverb.add_reverberation(rec, FF, 0.0, MU_DB, np.random.default_rng(4), n_azimuth=24, geometry="bistatic", **kw)
        return rec.samples.array

    plain = run()
    for motion in ({}, {"velocity": [0.0, 0.0, 0.0]}, {"velocity": [0, 0, 0], "rms_speed": 0.0, "rms_vertical_speed": 0.0},
                   {"wind_speed": 0.0}):
        assert np.array_equal(run(scatterer_motion=motion), plain)
    assert not np.array_equal(run(scatterer_motion={"velocity": [1.0, 0.0, 0.0]}), plain)
    assert not np.array_equal(run(scatterer_motion={"rms_speed": 20.0}), plain)       # resolvable by a 2 ms pulse
    assert np.array_equal(run(scatterer_motion={"rms_speed": 0.5}), plain)            # 18 Hz: far below 1 / T
    with pytest.raises(ValueError, match="scatterer_motion"):
        run(scatterer_motion={"speed": 1.0})
    with pytest.raises(ValueError, match="bistatic"):
        rec = bistatic_ping()
        reverb.add_reverberation(rec, FF, 0.0, MU_DB, np.random.default_rng(4), scatterer_motion={"rms_speed": 0.5})


def test_wind_speed_gives_the_surface_its_vertical_speed():
    """`wind_speed` is the rms vertical speed of the wind sea (acoustics.py); a surface
    patch seen in backscatter at grazing angle g spreads by 2 f sigma_w sin(g) / c."""
    seen = {}
    original = reverb_bistatic.directional_power

    def spy(*a, **kw):
        seen.update(kw)
        seen["out"] = original(*a, **kw)
        return seen["out"]

    rec = bistatic_ping(tx=np.array([0.0, 0.0, 30.0]), rx=np.array([0.0, 0.0, 30.0]))
    try:
        reverb_bistatic.directional_power = spy
        reverb.add_reverberation(rec, FF, 0.0, MU_DB, np.random.default_rng(1), n_azimuth=12, geometry="bistatic",
                                 scatterer_motion={"wind_speed": 8.0}, n_elevation=18)
    finally:
        reverb_bistatic.directional_power = original
    sigma_w = acoustics.rms_surface_vertical_speed(8.0)
    assert seen["scatterer_rms_vertical_speed"] == sigma_w and sigma_w > 0.3
    out = seen["out"]
    lit = out["power"].sum(axis=2) > 0
    sin_g = -out["direction"][..., 2]                              # the surface is above: rays go up
    expected = 2.0 * F * sigma_w * sin_g / C
    assert np.all(sin_g[lit] > 0)
    assert out["spread"][lit] == pytest.approx(expected[lit], rel=0.08)       # within a 10 degree elevation bin


def test_heave_and_vertical_rays_in_the_component_path():
    """A platform heaving at w above a patch straight below it: shift 2 f w / c (both legs
    vertical). The component path takes each platform's full velocity along its own ray;
    the default (monostatic) path uses horizontal velocity along a horizontal path and
    gives nothing here, which is its documented shortcut."""
    w = 0.8
    got = reverb_bistatic.directional_power(FF, RX, RX, 0.0, TIMES, sectors(36), frequency=F, mu_db=MU_DB,
                                            tx_velocity=(0, 0, w), rx_velocity=(0, 0, w), n_elevation=18,
                                            strength_map=patch_map([0.0, 0.0], 1.5))
    weight = got["power"].sum(axis=2)
    doppler = (weight * got["doppler"]).sum() / weight.sum()
    assert doppler == pytest.approx(2.0 * F * w / C, rel=0.01)      # 29.3 Hz, moving down towards the bottom
    # Horizontal speed towards a patch at 30 degrees grazing: 2 f v cos(30) / c.
    slant = 40.0
    centre = [np.sqrt(slant**2 - 20.0**2), 0.0]
    got = reverb_bistatic.directional_power(FF, RX, RX, 0.0, TIMES, sectors(360), frequency=F, mu_db=MU_DB,
                                            tx_velocity=(3, 0, 0), rx_velocity=(3, 0, 0), n_elevation=18,
                                            strength_map=patch_map(centre, 1.0))
    weight = got["power"].sum(axis=2)
    doppler = (weight * got["doppler"]).sum() / weight.sum()
    assert doppler == pytest.approx(2.0 * F * 3.0 * np.cos(np.radians(30.0)) / C, rel=0.01)


@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
def test_moving_scatterers_on_both_engines():
    out = []
    for engine in ("cpp", "numpy"):
        rec = bistatic_ping(v_tx=(2.0, 0.0, 0.0), duration=0.02, ranges=(60.0, 300.0))
        rng = np.random.default_rng(3)
        reverb.add_reverberation(rec, FF, 0.0, MU_DB, rng, n_azimuth=30, geometry="bistatic", engine=engine,
                                 n_elevation=3, scatterer_motion={"velocity": [0.5, -0.5, 0.0], "rms_speed": 0.4,
                                                                  "rms_vertical_speed": 0.2})
        out.append((rec.samples.array, rng.bit_generator.state))
    assert np.abs(out[0][0] - out[1][0]).max() < 5e-6 * np.abs(out[1][0]).max()
    assert out[0][1] == out[1][1]
