"""Milestone 12 (part): bottom reverberation by the point-scatterer method.

Validation: reverberation level against time matches the analytic sonar-equation
prediction within 2 dB for a flat, uniform case.
"""

import numpy as np
import pytest

from gpusonar.sim import Scene, lambert_rms_amplitude, make_pose, seafloor_patch, simulate_ping
from gpusonar.sonar import SonarModel

C = 1500.0
MU_DB = -27.0          # Lambert parameter, given explicitly throughout (the default until D-89; the default is now -29 dB)
HEIGHT = 20.0
PULSE = 2e-3


def reverberation_ping(seed, density=1.5, radius=260.0):
    """An omnidirectional sonar 20 m above a flat, uniformly rough seabed; CW pulse."""
    sonar = SonarModel.load("popoto_8", {
        "receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": [25.0, 230.0]},
                    "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3,
                                                          "duration": PULSE, "window_alpha": 0.0,
                                                          "source_level_db_rms": None, "amplitude": 1.0}}})
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


# --- the scattering-function method ----------------------------------------------------------

from gpusonar import flank  # noqa: E402
from gpusonar.sim import FreeFieldProvider, reverb  # noqa: E402


def empty_baseband_ping(name="popoto_8", ranges=(25.0, 230.0), fs=20e3, duration=PULSE):
    sonar = SonarModel.load(name, {
        "receive": {"kind": "baseband", "sample_rate": fs, "window": {"range": list(ranges)},
                    "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3,
                                                          "duration": duration, "window_alpha": 0.0,
                                                          "source_level_db_rms": None, "amplitude": 1.0}}})
    return sonar, simulate_ping(sonar, make_pose(position=(0.0, 0.0, -HEIGHT)), Scene(),
                                environment={"sound_speed": C}, adc=False)


def test_scattering_function_level_follows_the_sonar_equation():
    """Same flat, uniform case and the same analytic prediction as the point-scatterer test."""
    intensity = 0.0
    for seed in range(6):
        sonar, rec = empty_baseband_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), boundary_z=0.0, mu_db=MU_DB,
                                 rng=np.random.default_rng(seed))
        intensity = intensity + (np.abs(rec.samples.array) ** 2).mean(axis=0) / 6
    s = rec.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    slant = 0.5 * C * (t - 0.5 * PULSE)
    ok = (slant > 1.5 * HEIGHT) & (slant < 200.0)
    predicted = 10 ** (MU_DB / 10) * (HEIGHT / slant) ** 2 * np.pi * C * PULSE / slant**3
    edges = np.linspace(np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1], 13).astype(int)
    diffs = np.array([10 * np.log10(intensity[a:b].mean() / predicted[a:b].mean())
                      for a, b in zip(edges[:-1], edges[1:])])
    assert np.abs(diffs).max() < 2.0, diffs
    assert abs(diffs.mean()) < 1.0


def test_the_two_reverberation_methods_agree():
    """Point scatterers and the scattering function are two realisations of one model."""
    point = np.concatenate([reverberation_ping(seed).samples.array for seed in (11, 12)], axis=0)
    sf = []
    for seed in (21, 22):
        sonar, rec = empty_baseband_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(seed))
        sf.append(rec.samples.array)
    sf = np.concatenate(sf, axis=0)
    a = (np.abs(point) ** 2).mean(axis=0)
    b = (np.abs(sf) ** 2).mean(axis=0)
    blocks = slice(300, 3900)
    ratio = 10 * np.log10(a[blocks].reshape(12, -1).mean(axis=1) / b[blocks].reshape(12, -1).mean(axis=1))
    assert np.abs(ratio).max() < 2.0, ratio


def test_scattering_function_reverberation_has_array_structure():
    """Across the array the generated reverberation behaves like sound arriving from all
    azimuths: neighbouring hydrophones are partly coherent, and a beam collects only the
    part of it that arrives in the beam."""
    sonar, rec = empty_baseband_ping("flank_32", ranges=(60.0, 900.0), fs=16e3, duration=0.02)
    wf = rec.transmit.waveform
    assert wf.duration == 0.02
    reverb.add_reverberation(rec, FreeFieldProvider(C), boundary_z=0.0, mu_db=MU_DB,
                             rng=np.random.default_rng(3), n_azimuth=180)
    x = rec.samples.array
    mid = slice(2000, 9000)
    # Coherence between hydrophones d apart, for horizontally isotropic arrivals at 27.5 kHz
    # (2-D isotropy: Bessel J0(k d)).
    from scipy.special import j0
    k = 2 * np.pi * 27.5e3 / C
    for sep in (1, 2):
        num = np.mean(x[:-sep, mid] * np.conj(x[sep:, mid]))
        den = np.mean(np.abs(x[:, mid]) ** 2)
        assert np.real(num / den) == pytest.approx(j0(k * 0.094 * sep), abs=0.1)
    # Beamforming: per-channel reverberation power P gives a beam power well below the
    # N^2 P of a coherent arrival.
    bm = flank.farfield_beamform(rec, np.radians([0.0, 20.0, -35.0]))
    single = np.mean(np.abs(rec.samples.array[:, 3000:8000]) ** 2)     # after compression, per channel
    beam = np.mean(np.abs(bm.data[:, 3000:8000]) ** 2, axis=1)
    gain_db = 10 * np.log10(beam / single)
    assert np.all(gain_db < 20 * np.log10(32) - 8.0)       # far from coherent addition
    assert np.ptp(gain_db) < 4.0                           # about the same in every direction


def test_scattering_function_on_a_real_passband_record():
    """A real record gets the same reverberation level as a baseband one."""
    levels = {}
    for kind, fs in (("baseband", 20e3), ("real", 100e3)):
        sonar = SonarModel.load("popoto_8", {
            "receive": {"kind": kind, "sample_rate": fs, "window": {"range": [60.0, 120.0]}, "sensitivity_db": None},
            "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3,
                                                              "duration": PULSE, "window_alpha": 0.0,
                                                              "source_level_db_rms": None, "amplitude": 1.0}}})
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, -HEIGHT)), Scene(),
                            environment={"sound_speed": C}, adc=False,
                            reverberation=[{"boundary_z": 0.0, "mu_db": MU_DB}], absorption_model="none",
                            rng=np.random.default_rng(5))
        x = rec.samples.array
        levels[kind] = np.mean(np.abs(x) ** 2) * (2.0 if kind == "real" else 1.0)
    assert 10 * np.log10(levels["real"] / levels["baseband"]) == pytest.approx(0.0, abs=1.0)


def test_volume_layer_level_follows_the_sonar_equation():
    """A volume-scattering layer collapsed onto a sheet at its centre (SST's ReverbLayer): the
    sheet's strength per unit area is Sv x thickness at every angle, so against Lambert's law
    the prediction loses its sin^2(grazing) and keeps everything else."""
    sv_db, thickness = -70.0, 20.0
    intensity = 0.0
    for seed in range(6):
        sonar, rec = empty_baseband_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), boundary_z=0.0, mu_db=0.0, rng=np.random.default_rng(seed),
                                 strength=reverb.volume_layer_strength(sv_db, thickness))
        intensity = intensity + (np.abs(rec.samples.array) ** 2).mean(axis=0) / 6
    s = rec.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    slant = 0.5 * C * (t - 0.5 * PULSE)
    ok = (slant > 1.5 * HEIGHT) & (slant < 200.0)
    predicted = 10 ** (sv_db / 10) * thickness * np.pi * C * PULSE / slant**3
    edges = np.linspace(np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1], 13).astype(int)
    diffs = np.array([10 * np.log10(intensity[a:b].mean() / predicted[a:b].mean())
                      for a, b in zip(edges[:-1], edges[1:])])
    assert np.abs(diffs).max() < 2.0, diffs
    assert abs(diffs.mean()) < 1.0


def test_scenario_volume_layer_item_scales_with_its_strength():
    from gpusonar.sim import Scene, make_pose, simulate_ping
    sonar, _ = empty_baseband_ping()
    def power(sv_db):
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                            reverberation=[{"model": "volume", "layer": [40.0, 60.0], "sv_db": sv_db}], rng=np.random.default_rng(2))
        return float((np.abs(rec.samples.array) ** 2).mean())
    assert power(-60.0) > 0.0
    assert 10 * np.log10(power(-60.0) / power(-70.0)) == pytest.approx(10.0, abs=0.01)      # the same random numbers, ten times the power


def test_a_lambert_item_without_mu_db_uses_minus_29_db_and_warns_once_at_sonar_frequencies():
    """D-89: the bare default is -29 dB (Mackenzie 1961 as quoted by Muggleworth 1994, p. 34: deep
    water, 530 and 1030 Hz), 2 dB below the -27 dB it was; an explicit mu_db is untouched."""
    import warnings
    from gpusonar.sim import bottom
    sonar, rec0 = empty_baseband_ping()
    assert rec0.samples.center_frequency >= 10e3

    def ping(item):
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                            reverberation=[dict(item, boundary_z=0.0)], rng=np.random.default_rng(2))
        return rec.samples.array.copy()
    assert bottom.DEFAULT_LAMBERT_MU_DB == -29.0
    bottom._warned_default_lambert = False
    with pytest.warns(UserWarning, match="deep-water value measured at 0.5 and 1 kHz"):
        bare = ping({})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(ping({}), bare)                                # once
        named, old = ping({"mu_db": -29.0}), ping({"mu_db": MU_DB})          # an explicit value never warns
        assert bottom.default_lambert_mu_db(8e3) == -29.0
        bottom._warned_default_lambert = False
        assert bottom.default_lambert_mu_db(9.9e3) == -29.0 and bottom.default_lambert_mu_db() == -29.0   # below 10 kHz: silent
    assert np.array_equal(bare, named)
    assert 10 * np.log10((np.abs(bare) ** 2).mean() / (np.abs(old) ** 2).mean()) == pytest.approx(-2.0, abs=1e-4)
    # The functions that take mu_db directly have the same default.
    import inspect
    from gpusonar.sim import reverb_bistatic, reverb_relief
    for fn in (reverb_bistatic.directional_power, reverb_relief.directional_power):
        assert inspect.signature(fn).parameters["mu_db"].default == bottom.DEFAULT_LAMBERT_MU_DB


def _fluid_sediment_ping(item, seed=3):
    from gpusonar.sim import bottom
    sonar, _ = empty_baseband_ping()
    par = dict(bottom.sediment("medium_sand"), **item)
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, -HEIGHT)), Scene(), environment={"sound_speed": C}, adc=False,
                        reverberation=[dict({"model": "fluid_sediment", "boundary_z": 0.0}, **par)], rng=np.random.default_rng(seed))
    return rec.samples.array.copy()


def test_composite_item_carries_wagner_shadowing_and_shadowing_none_turns_it_off():
    """C1 of docs/references/bottom_mourad_jackson_1989.md: a `fluid_sediment` item with `roughness_model:
    composite` has Wagner's shadow function by default; `shadowing: none` is the composite before C1. The
    grazing angles here run from about 5 deg (230 m) to 53 deg, where 10 log R of medium sand at 27.5 kHz
    lies between -0.2 dB and 0 (bottom.wagner_backscatter_shadowing): the shadowed ping is weaker by no more."""
    from gpusonar.sim import bottom
    on = _fluid_sediment_ping({"roughness_model": "composite"})
    off = _fluid_sediment_ping({"roughness_model": "composite", "shadowing": "none"})
    assert not np.array_equal(on, off)
    ratio_db = 10 * np.log10((np.abs(on) ** 2).mean() / (np.abs(off) ** 2).mean())
    slope, _ = bottom.large_scale_slope(27.5e3, C, **bottom.sediment("medium_sand"))
    floor = 10 * np.log10(bottom.wagner_backscatter_shadowing(np.arcsin(HEIGHT / 230.0), np.sqrt(2.0) * slope))
    assert floor < ratio_db < 0.0, (floor, ratio_db)
    # The default (small_slope) carries none: naming it unchanged is the default ping, naming "wagner" is refused.
    assert np.array_equal(_fluid_sediment_ping({"shadowing": "none"}), _fluid_sediment_ping({}))
    with pytest.raises(ValueError, match="has no shadowing"):
        _fluid_sediment_ping({"shadowing": "wagner"})
    with pytest.raises(ValueError, match="did you mean 'none'"):
        _fluid_sediment_ping({"roughness_model": "composite", "shadowing": "non"})
