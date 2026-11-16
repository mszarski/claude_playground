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


# --- the scattering-function method ----------------------------------------------------------

from gpusonar import flank  # noqa: E402
from gpusonar.sim import FreeFieldProvider, reverb  # noqa: E402


def empty_baseband_ping(name="popoto_8", ranges=(25.0, 230.0), fs=20e3, duration=PULSE):
    sonar = SonarModel.load(name, {
        "receive": {"kind": "baseband", "sample_rate": fs, "window": {"range": list(ranges)},
                    "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3,
                                                          "duration": duration, "window_alpha": 0.0,
                                                          "source_level_db": None, "amplitude": 1.0}}})
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
                                                              "source_level_db": None, "amplitude": 1.0}}})
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, -HEIGHT)), Scene(),
                            environment={"sound_speed": C}, adc=False,
                            reverberation=[{"boundary_z": 0.0, "mu_db": MU_DB}], absorption_model="none",
                            rng=np.random.default_rng(5))
        x = rec.samples.array
        levels[kind] = np.mean(np.abs(x) ** 2) * (2.0 if kind == "real" else 1.0)
    assert 10 * np.log10(levels["real"] / levels["baseband"]) == pytest.approx(0.0, abs=1.0)
