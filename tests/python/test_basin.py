"""Late reverberation of a basin (sim/basin.py; scenario key `basin_reverberation:`).

Closed forms used: for the exponential law the share of the tail's energy later than t is
exp(-(t - onset) / T), its mean delay onset + T and its rms spread T; the two-way tail carries
2 E + E^2 of the echo's energy, E of it with the echo's own wavefront; a field of plane waves
from every azimuth has coherence J0(k d) between hydrophones d apart, one from a Gaussian sector
of rms sqrt(2) / (k L) has exp(-(d / L)^2) across it; a frozen field seen from a place d away
is correlated J0(k d) with itself.
"""
import numpy as np
import pytest
from scipy.special import j0

import gpusonar as gs
from gpusonar.sim import Scene, basin, make_pose, make_provider, run_scenario, simulate_ping
from gpusonar.sim import sources as src
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
ENV = {"sound_speed": C, "temperature": 12.0, "salinity": 35.0}
CFG = {"energy_db": -3.0, "onset_s": 0.010, "decay_s": 0.05, "coherence_m": "isotropic", "repeatable": True, "elevation_spread_deg": 0.0}


def model(kind="baseband", fs=8000.0, window=(0.0, 900.0), count=8, pitch=0.05, duration=0.004):
    w = dict(type="lfm", f_start=4000.0, f_stop=6000.0, duration=duration, window_alpha=0.1, source_level_db=190.0)
    return SonarModel.load("flank_32", {"array": {"count": count, "pitch": pitch}, "transmit": {"waveform": w},
                                        "receive": {"kind": kind, "sample_rate": fs, "window": {"range": list(window)}}})


def blank(m, pose=None):
    return m.make_record(pose or make_pose((0.0, 0.0, 20.0)), environment=ENV)


def impulse(rec, at=0.05):
    """A click on every hydrophone at the same instant (a broadside arrival), [C, N]."""
    s = rec.samples
    x = np.zeros((s.n_channels, s.n_samples), dtype=complex if s.kind == core.SampleKind.ComplexBaseband else float)
    x[:, int(round(at * s.sample_rate))] = 1.0
    return x


def times(rec):
    return np.arange(rec.samples.n_samples) / rec.samples.sample_rate


# --- the law ------------------------------------------------------------------------------------

def test_closed_forms_of_the_exponential_law_agree_with_its_density():
    onset, decay, e_db = 0.01, 0.145, -4.3
    t = np.linspace(0.0, 4.0, 400001)
    p = np.where(t >= onset, np.exp(-(t - onset) / decay) / decay, 0.0)
    dt = t[1] - t[0]
    later = np.cumsum(p[::-1])[::-1] * dt
    for at in (0.0, 0.05, 0.2, 0.6):
        assert later[np.searchsorted(t, at)] == pytest.approx(float(basin.energy_later_than(at, onset, decay)), abs=2e-4)
    mean = (p * t).sum() * dt
    assert mean == pytest.approx(onset + decay, rel=1e-4)
    assert np.sqrt((p * (t - mean) ** 2).sum() * dt) == pytest.approx(basin.rms_delay_spread(onset, decay), rel=1e-4)
    # The whole response: a unit arrival at delay 0 and the tail with 10^(e_db / 10) of its energy.
    e = 10.0 ** (e_db / 10.0)
    whole_later = e * later / (1.0 + e)
    for share in (0.1, 0.01, 0.001):
        assert t[np.argmax(whole_later < share)] == pytest.approx(float(basin.delay_of_share(share, onset, decay, e_db)), abs=2e-4)
    m1 = (e * p * t).sum() * dt / (1.0 + e)
    m2 = (e * p * t**2).sum() * dt / (1.0 + e)
    assert np.sqrt(m2 - m1**2) == pytest.approx(basin.rms_delay_spread(onset, decay, e_db), rel=1e-4)
    # Two-way: the diffuse part integrates to E + E^2.
    assert (basin.two_way_density(t, onset, decay, e) * dt).sum() == pytest.approx(e + e * e, rel=1e-3)


def test_the_measured_sets_reproduce_the_measured_times_within_the_fit():
    """99 % and 90 % of the energy: measured 465 / 165 ms at 274 m and 636 / 275 ms at 1.02 km
    (docs/figures/channel_validation_smartocean.md). The exponential with the documented numbers."""
    for name, (t99, t90) in (("smartocean_274m", (0.465, 0.165)), ("smartocean_1020m", (0.636, 0.275))):
        m = basin.MEASURED[name]
        assert float(basin.delay_of_share(0.01, m["onset_s"], m["decay_s"], m["energy_db"])) == pytest.approx(t99, rel=0.08)
        assert float(basin.delay_of_share(0.10, m["onset_s"], m["decay_s"], m["energy_db"])) == pytest.approx(t90, rel=0.08)


def test_configuration_has_no_defaults_and_refuses_what_it_does_not_know():
    for key in ("energy_db", "onset_s", "decay_s", "coherence_m", "repeatable"):
        with pytest.raises(ValueError, match="must be given"):
            basin.BasinReverberation({k: v for k, v in CFG.items() if k != key})
    with pytest.raises(ValueError, match="unknown keys"):
        basin.BasinReverberation(dict(CFG, decay=0.1))
    with pytest.raises(ValueError, match="exponential"):
        basin.BasinReverberation(dict(CFG, law="power"))
    with pytest.raises(ValueError, match="bearing_deg"):
        basin.BasinReverberation(dict(CFG, coherence_m=1.0))
    assert not hasattr(basin.attach(make_provider("free_field", C), None), "basin")


# --- the generated one-way tail -------------------------------------------------------------------

@pytest.mark.parametrize("kind,fs", [("baseband", 8000.0), ("real", 32000.0)])
def test_one_way_tail_has_the_energy_the_delays_and_the_spread_of_the_law(kind, fs):
    rec = blank(model(kind, fs, count=8))
    b = basin.BasinReverberation(CFG, seed=3)
    x = impulse(rec)
    if kind == "real":
        # A click has energy at every frequency; a real record's tail is made for the band the signal has.
        x[:, 1 + int(round(0.05 * fs))] = -1.0
    tail = b.one_way(rec, x.copy()) - x
    assert tail.dtype == x.dtype
    e_x, e_tail = (np.abs(x) ** 2).sum(), (np.abs(tail) ** 2).sum()
    n_independent = b.directions * CFG["decay_s"] * fs
    assert e_tail / e_x == pytest.approx(10.0 ** (CFG["energy_db"] / 10.0), rel=6.0 / np.sqrt(n_independent) + 0.01)
    power = (np.abs(tail) ** 2).sum(axis=0)
    t = times(rec) - 0.05
    # Nothing before the onset (less the array's own size in travel time and the band's ringing).
    assert power[t < CFG["onset_s"] - 0.001].sum() < 1e-3 * e_tail
    for at in (0.03, 0.06, 0.11, 0.16):
        assert power[t >= at].sum() / e_tail == pytest.approx(float(basin.energy_later_than(at, CFG["onset_s"], CFG["decay_s"])), abs=0.02)
    mean = (power * t).sum() / e_tail
    assert mean == pytest.approx(CFG["onset_s"] + CFG["decay_s"], rel=0.03)
    assert np.sqrt((power * (t - mean) ** 2).sum() / e_tail) == pytest.approx(basin.rms_delay_spread(CFG["onset_s"], CFG["decay_s"]), rel=0.04)


def coherence(tail, fs, f_lo, f_hi, fc):
    """Coherence of each channel with the first over the band [f_lo, f_hi] (absolute Hz)."""
    spec = np.fft.fft(tail, axis=1)
    f = np.fft.fftfreq(tail.shape[1], 1.0 / fs) + fc
    sel = (f >= f_lo) & (f <= f_hi)
    cross = (spec[:, sel] * np.conj(spec[0, sel])[None, :]).sum(axis=1)
    return cross / np.sqrt((np.abs(spec[:, sel]) ** 2).sum(axis=1) * (np.abs(spec[0, sel]) ** 2).sum())


def test_isotropic_tail_has_the_coherence_of_a_horizontally_diffuse_field():
    m = model(count=12, pitch=0.03, window=(0.0, 1200.0))
    rec = blank(m)
    b = basin.BasinReverberation(dict(CFG, decay_s=0.2, directions=256), seed=5)
    tail = b.one_way(rec, impulse(rec)) - impulse(rec)
    d = 0.03 * np.arange(12)
    got = coherence(tail, 8000.0, 4700.0, 5300.0, 5000.0)
    f = np.linspace(4700.0, 5300.0, 61)
    want = j0(2.0 * np.pi * f[None, :] * d[:, None] / C).mean(axis=1)           # passes through zero near 0.115 m, -0.40 at 0.18 m
    assert np.abs(got.imag).max() < 0.12
    assert np.abs(got.real - want).max() < 0.12
    assert got.real[6] < -0.2 and abs(got.real[4]) < 0.15


def test_sector_tail_decorrelates_over_the_stated_coherence_length():
    length = 0.6
    m = model(count=10, pitch=0.15, window=(0.0, 1200.0))
    rec = blank(m)
    b = basin.BasinReverberation(dict(CFG, decay_s=0.2, coherence_m=length, bearing_deg=90.0, directions=256), seed=6)     # from broadside
    tail = b.one_way(rec, impulse(rec)) - impulse(rec)
    d = 0.15 * np.arange(10)
    got = coherence(tail, 8000.0, 4500.0, 5500.0, 5000.0)
    # The sector's width is set at the centre frequency: exp(-(d / L)^2 (f / fc)^2), averaged over the band used.
    f = np.linspace(4500.0, 5500.0, 41)
    want = np.exp(-((d[:, None] / length) * (f[None, :] / 5000.0)) ** 2).mean(axis=1)
    assert np.abs(got - want).max() < 0.1
    assert abs(got[4]) == pytest.approx(np.exp(-1.0), abs=0.1)                  # 1/e at d = L


def test_vertical_coherence_follows_the_elevation_spread():
    spread = 6.0
    b = basin.BasinReverberation(dict(CFG, elevation_spread_deg=spread), seed=2)
    u = basin.plane_waves(4000, "isotropic", 20.0, elevation_spread_deg=spread, rng=np.random.default_rng(1))
    k, dz = 2.0 * np.pi * 5000.0 / C, 0.5
    got = np.mean(np.exp(1j * k * dz * u[:, 2]))
    assert abs(got) == pytest.approx(np.exp(-0.5 * (k * dz * np.radians(spread)) ** 2), abs=0.03)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0) and b.elevation_spread == spread


# --- echoes: the two-way tail -----------------------------------------------------------------------

def ping(m, cfg, *, t=0.0, engine=None, pose=None, seed=0, target=(0.0, 300.0, 20.0)):
    provider = basin.attach(make_provider("free_field", C, engine=engine), cfg, seed=seed)   # one engine per ping (a-engarg)
    scene = Scene().add_points([list(target)], target_strength_db=10.0)
    rec = simulate_ping(m, pose or make_pose((0.0, 0.0, 20.0)), scene, environment=ENV, provider=provider, adc=False, absorption_model="none",
                        scenario_time=t, engine=engine)
    return np.array(rec.samples.array), getattr(provider, "basin", None), rec


def test_echo_tail_carries_2e_plus_e2_and_a_share_e_keeps_the_echo_wavefront():
    m = model(count=8, pitch=0.15, window=(200.0, 1100.0))
    plain, _, rec = ping(m, None)
    with_tail, b, _ = ping(m, CFG, seed=4)
    e = 10.0 ** (CFG["energy_db"] / 10.0)
    tail = with_tail - plain
    echo = (np.abs(plain) ** 2).sum()
    assert b.last["echo_energy"] == pytest.approx(echo, rel=1e-6)
    assert b.last["common_energy"] / echo == pytest.approx(e, rel=0.15)                  # one kernel: 0.05 s x 2 kHz of independent values
    assert b.last["diffuse_energy"] / echo == pytest.approx(e + e * e, rel=0.05)
    assert (np.abs(tail) ** 2).sum() / echo == pytest.approx(2.0 * e + e * e, rel=0.1)
    # The modelled echo itself is untouched: nothing is added before its onset.
    t_echo = 2.0 * 300.0 / C - rec.samples.start_time
    early = times(rec) < t_echo + CFG["onset_s"] - 0.002
    assert (np.abs(tail[:, early]) ** 2).sum() < 1e-4 * (np.abs(tail) ** 2).sum()
    # Diffuse and common parts apart: with echoes off and own ping absent nothing is added at all.
    silent, _, _ = ping(m, dict(CFG, echoes=False), seed=4)
    assert np.array_equal(silent, plain)


def test_the_part_that_reached_the_target_late_comes_from_the_target_bearing():
    m = model(count=8, pitch=0.15, window=(200.0, 1100.0))
    target = (150.0, 260.0, 20.0)                                                        # 30 degrees off broadside
    plain, _, rec = ping(m, None, target=target)
    b = basin.BasinReverberation(CFG, seed=8)
    arr = b._array(rec)
    common = b._common(plain, arr, 1, 0)
    diffuse = b._diffuse(rec, arr, plain[arr["ref"]], arr["start"], lambda t: basin.two_way_density(t, b.onset, b.decay, b.energy), 2, 0)

    def beam_share(x):
        spec = np.fft.fft(x, axis=1)
        f = np.fft.fftfreq(x.shape[1], 1.0 / 8000.0) + 5000.0
        sel = np.abs(np.fft.fftfreq(x.shape[1], 1.0 / 8000.0)) < 900.0
        pos = np.asarray(rec.rx_positions(0.0))
        direction = np.array(target) - pos.mean(axis=0)
        tau = -(pos - pos.mean(axis=0)) @ (direction / np.linalg.norm(direction)) / C
        steered = (spec[:, sel] * np.exp(2j * np.pi * f[sel][None, :] * tau[:, None])).sum(axis=0)
        return (np.abs(steered) ** 2).sum() / (x.shape[0] * (np.abs(spec[:, sel]) ** 2).sum())

    assert beam_share(plain) == pytest.approx(1.0, abs=0.02)
    assert beam_share(common) == pytest.approx(1.0, abs=0.02)            # the full array gain, as the echo
    assert beam_share(diffuse) < 0.35                                    # 1 / 8 for an incoherent field; a few beams wide here


def test_tail_follows_the_echo_on_both_engines():
    m = model(count=4, pitch=0.15, window=(200.0, 700.0))
    out = {}
    for engine in ("cpp", "numpy"):
        plain, _, _ = ping(m, None, engine=engine)
        with_tail, _, _ = ping(m, CFG, engine=engine, seed=4)
        out[engine] = (plain, with_tail - plain)
    scale = np.abs(out["cpp"][1]).max()
    assert scale > 0.0
    assert np.abs(out["cpp"][1] - out["numpy"][1]).max() < 2e-3 * scale + 2.0 * np.abs(out["cpp"][0] - out["numpy"][0]).max()


# --- ping to ping -----------------------------------------------------------------------------------

def correlation(a, b):
    return float(np.real(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real))


@pytest.mark.parametrize("share", [1.0, 0.6, 0.0])
def test_repeatable_is_the_share_of_the_tail_that_is_the_same_on_the_next_ping(share):
    rec = blank(model(count=6))
    b = basin.BasinReverberation(dict(CFG, repeatable=share), seed=9)
    x = impulse(rec)
    first, second = b.one_way(rec, x.copy(), 0.0) - x, b.one_way(rec, x.copy(), 3.0) - x
    again = b.one_way(rec, x.copy(), 0.0) - x
    assert np.array_equal(first, again)                                   # a ping is reproducible whatever the share
    if share == 1.0:
        assert np.allclose(first, second, rtol=0, atol=1e-12 * np.abs(first).max())
    else:
        assert correlation(first, second) == pytest.approx(share, abs=0.04)


def test_a_frozen_tail_decorrelates_as_j0_when_the_array_moves():
    m = model(count=4)
    b = basin.BasinReverberation(dict(CFG, directions=128), seed=10)
    k = 2.0 * np.pi * 5000.0 / C
    rec0 = blank(m)
    x = impulse(rec0)
    here = b.one_way(rec0, x.copy()) - x
    for shift in (0.03, 0.1, 0.25):
        there = b.one_way(blank(m, make_pose((shift, 0.0, 20.0))), x.copy()) - x
        assert correlation(here, there) == pytest.approx(float(j0(k * shift)), abs=0.05)


# --- the sonar's own ping --------------------------------------------------------------------------

def test_own_ping_term_has_the_stated_energy_of_the_pulse_at_one_metre():
    own_db = -40.0
    m = model(count=6, pitch=0.5, window=(15.0, 700.0), duration=0.02)
    cfg = dict(CFG, own_ping_db=own_db, echoes=False, repeatable=False)                 # a new tail every ping: an average over pings
    provider = basin.attach(make_provider("free_field", C), cfg, seed=12)
    tails = []
    for i in range(8):
        rec = simulate_ping(m, make_pose((0.0, 0.0, 20.0)), Scene(), environment=ENV, provider=provider, adc=False, absorption_model="none",
                            scenario_time=2.0 * i)
        tails.append(np.array(rec.samples.array))
    tail = np.array(tails)                                                              # [pings, C, N]
    pulse, t0 = basin.transmitted_pulse(rec, m, environment=ENV, absorption_model="none")
    assert provider.basin.last["pulse_energy"] == pytest.approx((np.abs(pulse) ** 2).sum())
    # The pulse at 1 m is the source level: 190 dB re 1 uPa at 1 m as the simulator takes it (a peak pressure), in sample units.
    peak = 10.0 ** (190.0 / 20.0) * 10.0 ** ((m.config["receive"].get("sensitivity_db") or 0.0) / 20.0 + m.config["receive"].get("gain_db", 0.0) / 20.0)
    # (as a peak today; sqrt(2) more if the simulator comes to read source levels as rms: either is the simulator's own pulse)
    assert min(abs(np.abs(pulse).max() / peak - 1.0), abs(np.abs(pulse).max() / peak - np.sqrt(2.0))) < 0.02
    fs, start, n = rec.samples.sample_rate, rec.samples.start_time, rec.samples.n_samples
    tp = t0 + np.arange(len(pulse)) / fs
    w = np.abs(pulse) ** 2
    later = lambda t: basin.energy_later_than(t, CFG["onset_s"], CFG["decay_s"])        # noqa: E731
    expected = 10.0 ** (own_db / 10.0) * (w * (later(start - tp) - later(start + n / fs - tp))).sum()
    assert (np.abs(tail) ** 2).sum(axis=2).mean() == pytest.approx(expected, rel=0.05)
    # It decays with the law: the second 50 ms of the window against the first.
    k = int(0.05 * fs)
    ratio = (np.abs(tail[..., k:2 * k]) ** 2).sum() / (np.abs(tail[..., :k]) ** 2).sum()
    assert ratio == pytest.approx(np.exp(-0.05 / CFG["decay_s"]), rel=0.08)


# --- one-way sources and the scenario key ---------------------------------------------------------

SCENARIO = {
    "name": "basin_test", "sonar": "flank_32", "fidelity": "free_field",
    "sonar_overrides": {"array": {"count": 4, "pitch": 0.15},
                        "transmit": {"waveform": {"type": "lfm", "f_start": 4000.0, "f_stop": 6000.0, "duration": 0.004, "window_alpha": 0.1, "source_level_db": 190.0}},
                        "receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [100.0, 700.0]}}},
    "environment": {"sound_speed": C, "temperature": 12.0, "salinity": 35.0},
    "platform": {"position": [0.0, 0.0, 20.0]}, "pings": {"count": 2, "interval": 2.0},
    "scene": [{"type": "points", "label": "target", "positions": [[0.0, 300.0, 20.0]], "target_strength_db": 10.0}],
    "absorption": "none", "receiver": {"adc": False}, "seed": 7,
}


def arrays(result):
    return [np.array(r.samples.array) for r in result.records]


def test_scenario_without_the_key_is_bit_identical_and_with_it_gains_the_tail():
    import copy
    plain = arrays(run_scenario(copy.deepcopy(SCENARIO)))
    again = arrays(run_scenario(copy.deepcopy(SCENARIO)))
    assert all(np.array_equal(a, b) for a, b in zip(plain, again))
    off = arrays(run_scenario(dict(copy.deepcopy(SCENARIO), basin_reverberation=dict(CFG, echoes=False, sources=False))))
    assert all(np.array_equal(a, b) for a, b in zip(plain, off))          # the hook itself changes nothing
    with_tail = arrays(run_scenario(dict(copy.deepcopy(SCENARIO), basin_reverberation=dict(CFG))))
    e = 10.0 ** (CFG["energy_db"] / 10.0)
    for a, b in zip(plain, with_tail):
        assert (np.abs(b - a) ** 2).sum() / (np.abs(a) ** 2).sum() == pytest.approx(2.0 * e + e * e, rel=0.15)
    # repeatable: true, a sonar that has not moved: the same tail on both pings
    assert np.allclose(with_tail[0] - plain[0], with_tail[1] - plain[1], atol=1e-5 * np.abs(with_tail[0]).max())


def test_a_radiating_source_gets_a_one_way_tail_of_energy_e():
    m = model(count=4, pitch=0.15, window=(0.0, 900.0))
    item = [{"label": "pinger", "position": [0.0, 250.0, 20.0], "signal": {"type": "waveform", "ping_offset": 0.0}}]

    def listen(cfg):
        provider = basin.attach(make_provider("free_field", C), cfg, seed=13)
        rec = simulate_ping(m, make_pose((0.0, 0.0, 20.0)), Scene(), environment=ENV, provider=provider, absorption_model="none", adc=False,
                            sources=src.build_sources(item, np.random.default_rng(1)), transmit=False, rng=np.random.default_rng(1))
        return np.array(rec.samples.array), rec

    plain, rec = listen(None)
    with_tail, _ = listen(CFG)
    tail = with_tail - plain
    assert (np.abs(plain) ** 2).sum() > 0.0
    assert (np.abs(tail) ** 2).sum() / (np.abs(plain) ** 2).sum() == pytest.approx(10.0 ** (CFG["energy_db"] / 10.0), rel=0.06)
    t = times(rec) + rec.samples.start_time - 250.0 / C
    power = (np.abs(tail) ** 2).sum(axis=0)
    assert power[t >= 0.06].sum() / power.sum() == pytest.approx(float(basin.energy_later_than(0.06, CFG["onset_s"], CFG["decay_s"])), abs=0.05)
    off, _ = listen(dict(CFG, sources=False))
    assert np.array_equal(off, plain)


def test_raw_records_with_board_clocks_get_a_real_tail_of_the_right_energy():
    m = SonarModel.load("popoto_testauv_1row_clocks", {"receive": {"window": {"range": [250.0, 420.0]}}})
    scene = Scene().add_points([[0.0, 280.0, 10.0]], target_strength_db=10.0)

    def run(cfg):
        provider = basin.attach(make_provider("free_field", C), cfg, seed=14)
        rec = simulate_ping(m, make_pose((0.0, 0.0, 10.0)), scene, environment=ENV, provider=provider, adc=False, rng=np.random.default_rng(2))
        return np.array(rec.samples.array), provider

    plain, _ = run(None)
    with_tail, provider = run(dict(CFG, decay_s=0.03))
    assert with_tail.dtype == plain.dtype and not np.iscomplexobj(with_tail)
    last = provider.basin.last
    e = 10.0 ** (CFG["energy_db"] / 10.0)
    assert last["common_energy"] / last["echo_energy"] == pytest.approx(e, rel=0.25)
    assert last["diffuse_energy"] / last["echo_energy"] == pytest.approx(e + e * e, rel=0.08)
    # The synchronisation burst on the boards' first inputs is electrical: it has no tail (it is added after).
    assert np.isfinite(with_tail).all() and (np.abs(with_tail - plain) ** 2).sum() > 0.0


# --- the rehearsal scenarios -------------------------------------------------------------------------

FJORD = (("fjord_testauv_trial_1row", 1000.0, "smartocean_274m"), ("fjord_static_trial_2row", 3000.0, "smartocean_1020m"), ("fjord_static_trial_3row", 500.0, "smartocean_274m"))


def test_illustrative_profile_has_the_shape_of_the_casts_and_the_scenarios_carry_it():
    from gpusonar.config import load_scenario_config
    z, c = basin.stratified_profile(90.0, step=5.0)
    assert c[0] == pytest.approx(1480.5) and z[np.argmax(c)] == 19.0 and c.max() == pytest.approx(1495.0)
    assert float(np.interp(54.0, z, c)) == pytest.approx(1492.0) and np.all(np.diff(c[z <= 19.0]) > 0) and np.all(np.diff(c[(z >= 19.0) & (z <= 54.0)]) < 0)
    assert np.diff(c[z >= 54.0]) / np.diff(z[z >= 54.0]) == pytest.approx(0.017, abs=1e-6)          # below the casts: the pressure gradient (an assumption)
    section = np.array(basin.basin_section(3000.0, step=100.0))
    assert section[0, 1] == 50.0 and section[:, 1].max() <= 90.0 and section[:, 1].min() >= 50.0
    assert section[7, 1] == pytest.approx(53.0) and section[3, 1] > 80.0          # a basin, then a sill
    for name, length, measured in FJORD:
        cfg = load_scenario_config(name)
        env = cfg["propagation"]["environment"]
        assert np.allclose(env["ssp_depth"], z) and np.allclose(env["ssp_speed"], np.round(c, 1))
        assert np.allclose(env["bathymetry"], basin.basin_section(length, step=100.0))
        b = cfg["basin_reverberation"]
        m = basin.MEASURED[measured]
        assert all(b[k] == m[k] for k in ("energy_db", "onset_s", "decay_s", "repeatable")) and "own_ping_db" not in b
        basin.BasinReverberation(b)


def test_fjord_scenario_runs_through_bellhop_and_gains_the_tail():
    import copy
    from gpusonar.config import load_scenario_config
    from gpusonar.sim.bellhop import find_bellhop
    if find_bellhop() is None:
        pytest.skip("Bellhop is not installed (GPUSONAR_BELLHOP)")
    cfg = load_scenario_config("fjord_static_trial_3row")
    cfg["pings"] = dict(cfg["pings"], count=1)
    cfg["receiver"] = {"adc": False}
    cfg["sonar"] = "popoto_static_3row"                                    # ideal clocks: only the water is under test
    with_tail = arrays(run_scenario(copy.deepcopy(cfg)))[0]
    plain = arrays(run_scenario({k: v for k, v in copy.deepcopy(cfg).items() if k != "basin_reverberation"}))[0]
    e = 10.0 ** (cfg["basin_reverberation"]["energy_db"] / 10.0)
    ratio = (np.abs(with_tail - plain) ** 2).sum() / (np.abs(plain) ** 2).sum()
    # The window ends 0.33 s after the echo: most of the tail (decay 0.145 s) is in it.
    assert 0.6 * (2.0 * e + e * e) < ratio < 1.15 * (2.0 * e + e * e)


def test_every_variant_of_the_fjord_rehearsal_tool_loads_and_names_what_it_overrides():
    """tools/fjord_rehearsal.py puts the open-water rehearsal's noise (wind 5 m/s, shipping 0.5) into the fjord scenario
    for three of its six variants. The fjord's environment has a wind of 1.9 m/s and no shipping, and the scenario loader
    refuses a typed value that contradicts the environment unless it is named with its reason: the tool failed on those
    three. It names them; without the names the loader still refuses."""
    import importlib.util
    import pathlib
    import warnings
    from gpusonar import environment
    path = pathlib.Path(__file__).resolve().parents[2] / "tools" / "fjord_rehearsal.py"
    spec = importlib.util.spec_from_file_location("fjord_rehearsal", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    copied = ("profile, section", "+ late tail", "+ own ping, what-if")
    for build in tool.BUILDS.values():
        for name in tool.VARIANTS:
            cfg = tool.variant(build, name)
            with warnings.catch_warnings():
                warnings.simplefilter("error")                      # a name that overrides nothing is a warning: none here
                out = environment.expand_scenario(dict(cfg))
            noise = out["receiver"]["ambient_noise"]
            named = {k: v for k, v in (cfg.get("environment_overrides") or {}).items() if k.startswith("receiver.ambient_noise.")}
            if name in copied:
                assert (noise["wind_speed"], noise["shipping"]) == (5.0, 0.5) and "statistics" not in noise      # the open-water run's noise
                assert set(named) == {"receiver.ambient_noise.wind_speed", "receiver.ambient_noise.shipping"}
                assert all("open-water rehearsal's noise" in reason for reason in named.values())
                bare = dict(cfg, environment_overrides={k: v for k, v in cfg["environment_overrides"].items() if k not in named})
                with pytest.raises(environment.EnvironmentError, match="receiver.ambient_noise.wind_speed: 5.0 here, 1.9"):
                    environment.expand_scenario(bare)
            else:
                assert not named
            if name == "fjord as shipped":
                assert (noise["wind_speed"], noise["shipping"], noise["statistics"]["model"]) == (1.9, 0.0, "k")


def test_the_fjord_rehearsal_scores_a_detection_by_the_ranges_it_covers():
    """The chain's detector reports an echo and the basin's tail behind it as ONE detection at their energy-weighted
    range. For the two-row static array at 2.6 km that range is 63 m behind the target: a window of 30 m before to 60 m
    behind about the detection's range scored a 44 dB detection of the target as "behind" it and the target as not
    detected on four pings of four. The detection's first arrival and extent decide; a detection that begins 100 m behind
    the target is still not the target's."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "tools" / "fjord_rehearsal.py"
    spec = importlib.util.spec_from_file_location("fjord_rehearsal", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    cfg = {"scene": [{"positions": [[300.0, 2583.0, 20.0]]}]}                      # 2600.5 m from the origin, 6.62 degrees
    rng, brg = float(np.hypot(300.0, 2583.0)), float(np.degrees(np.arctan2(300.0, 2583.0)))

    def report(dets, tracks=()):
        return {"ownship_xyz": [0.0, 0.0, 20.0], "detections": dets, "tracks": list(tracks)}

    merged = {"range_m": rng + 63.3, "bearing_deg": brg - 0.17, "snr_db": 44.4, "first_range_m": rng + 1.5, "extent_m": 118.0}
    later = {"range_m": rng + 160.0, "bearing_deg": brg, "snr_db": 20.0, "first_range_m": rng + 100.0, "extent_m": 90.0}
    point = {"range_m": rng + 75.0, "bearing_deg": brg, "snr_db": 15.0}            # no extent: a point, 75 m behind
    away = {"range_m": rng + 20.0, "bearing_deg": brg + 40.0, "snr_db": 30.0, "first_range_m": rng + 19.0, "extent_m": 2.0}
    out = tool.score(cfg, [report([merged, later, point, away], [{"range_m": rng + 63.0, "bearing_deg": brg}]), report([later, away])], rows=1)
    first, second = out["per_ping"]
    assert first["detected"] and first["snr_db"] == 44.4 and first["range_error_m"] == pytest.approx(63.3) and first["first_range_error_m"] == pytest.approx(1.5)
    assert (first["behind_target"], first["elsewhere"]) == (2, 1) and first["tracks_off_target"] == 0      # the track at the merged detection's range is the target's
    assert not second["detected"] and (second["behind_target"], second["elsewhere"]) == (1, 1)
    assert out["pings_detected"] == 1 and out["track_confirmed_at_ping"] == 1
