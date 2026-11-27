"""The Doppler of boundary reverberation seen from a moving platform: each frequency's own shift
(`doppler: per_frequency`, the default since 2026-11-27, the owner: "FIx it") against one shift at
the centre frequency (`doppler: centre`, the law before, kept to reproduce earlier records;
sim/reverb.py, libs/sim reverb, the CUDA sum).

The law (TR 0702 eqs. (63)-(64), Sec. 8.3, confirmed; docs/PHYSICS_NOTES.md Sec. 3.4 and 14.1):
the echo of a still scatterer is the pulse time-scaled by Gamma = 1 - dT/dt, so a line sent at f
comes back at Gamma f: shifted by 2 (v . u) f / c (first order, monostatic). The default
generator shifted a whole sector by 2 (v . u) fc / c at the record's centre frequency fc, which is
right at fc only (`centre` still does). On a wide pulse the difference is (f - fc) 2 (v . u) / c.

The measurement here is independent of the generator's code path: one sector of boundary
(a known bearing) lit for a long record, a pulse of two Hann tones 2 kHz apart, the spectrum of
the output, and the power centroid about each tone, against that closed form.

Measured on the case below (ours, 3 seeds; run with -s):
  centre law:        both tones shifted by the centre's 20.33 Hz, 3.70 Hz off the physical shift each
  per_frequency law: 4.5 kHz at 16.63 Hz and 6.5 kHz at 24.02 Hz within 0.1 Hz, both engines.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.sim import FreeFieldProvider, reverb
from gpusonar.sim import engine as sim_engine

from test_reverb_bistatic import ENGINE_CASES, bistatic_ping
from test_sim_cpp import C, MU_DB, REVERB_CASES, empty_ping, relative_difference

core = gs.core
needs_cpp = pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
needs_cuda = pytest.mark.skipif(not core.cuda_available(), reason="no CUDA device")

# The case: a line pair 4.5 / 6.5 kHz about a 5.5 kHz carrier, 1 s Hann; complex baseband at 8 kHz;
# an 8 s record of one lit sector of eight (azimuth 22.5 deg); the platform at 3 m/s along x.
FC, FS, T_PULSE, T_RECORD, START = 5500.0, 8000.0, 1.0, 8.0, 1.0
TONES = (4500.0, 6500.0)
VELOCITY = np.array([3.0, 0.0, 0.0])
N_AZIMUTH, SECTOR = 8, 0
TOLERANCE_HZ = 0.5          # against a law difference of 3.70 Hz per tone

PHYSICAL = "per_frequency"


def sector_direction():
    a = (SECTOR + 0.5) * 2.0 * np.pi / N_AZIMUTH
    return np.array([np.cos(a), np.sin(a), 0.0])


def expected_shift(f, law):
    """The closed form: 2 (v . u) f / c (each frequency its own), or 2 (v . u) fc / c (one shift)."""
    return 2.0 * float(VELOCITY @ sector_direction()) * (f if law == PHYSICAL else FC) / C


def one_sector(engine, seed, doppler=None, batched=False):
    """One channel at the array centre; the reverberation of one lit sector; [n] complex baseband."""
    n_pulse = int(T_PULSE * FS)
    u = np.arange(n_pulse) / FS
    pulse = np.hanning(n_pulse) * sum(np.exp(2j * np.pi * (f - FC) * u) for f in TONES)
    n = int(T_RECORD * FS)
    times = START - T_PULSE + np.arange(n + n_pulse) / FS
    n_low = len(times)
    coarse = np.linspace(times[0], times[-1], 40)
    density = np.zeros((N_AZIMUTH, len(coarse)))
    density[SECTOR] = 1.0
    positions = np.zeros((1, 3))
    rng = np.random.default_rng(seed)
    kw = {} if doppler is None else dict(doppler=doppler)
    if engine == "numpy":
        out = reverb._series_numpy(rng, coarse, density, pulse, positions, positions[0], FS, n, START, times, FC, C,
                                   N_AZIMUTH, 1, n_low, VELOCITY, False, False, **kw)
    else:
        reverb.BATCHED_ON_HOST = batched
        try:
            out = reverb._series_cpp(rng, coarse, density, False, pulse, positions, FS, n, START, times[0], FC, C,
                                     N_AZIMUTH, 1, n_low, VELOCITY, False, False, core.Backend.Host, **kw)
        finally:
            reverb.BATCHED_ON_HOST = False
    return np.asarray(out)[0]


def tone_shifts(series_of_seeds):
    """Power centroid of each tone's line, less the tone, Hz: the spectrum summed over seeds, over a
    window that holds both laws' predictions with 6 Hz to spare."""
    n = len(series_of_seeds[0])
    power = sum(np.abs(np.fft.fft(x * np.hanning(n))) ** 2 for x in series_of_seeds)
    f = FC + np.fft.fftfreq(n, 1.0 / FS)
    shifts = []
    for tone in TONES:
        a, b = sorted((expected_shift(tone, "centre"), expected_shift(tone, PHYSICAL)))
        inside = (f >= tone + a - 6.0) & (f <= tone + b + 6.0)
        shifts.append(float((power[inside] * (f[inside] - tone)).sum() / power[inside].sum()))
    return np.array(shifts)


ENGINES = ["numpy", pytest.param("cpp", marks=needs_cpp), pytest.param("batched", marks=needs_cpp)]


def _engine(name):
    return ("cpp", True) if name == "batched" else (name, False)


@pytest.mark.parametrize("engine", ENGINES)
def test_each_tone_of_a_wide_pulse_takes_its_own_doppler_by_default(engine):
    """The physical law, the default and `per_frequency` alike. Failed before the fix by 3.65 Hz a
    tone (the generator then gave 20.28 / 20.41 Hz: the next test's numbers)."""
    eng, batched = _engine(engine)
    want = np.array([expected_shift(f, PHYSICAL) for f in TONES])
    for doppler in (None, "per_frequency"):
        got = tone_shifts([one_sector(eng, seed, doppler=doppler, batched=batched) for seed in range(3)])
        print(f"[revdop] {engine} {doppler or 'default'}: tone shifts {np.round(got, 3)} Hz, closed form {np.round(want, 3)}")
        assert np.all(np.abs(got - want) < TOLERANCE_HZ), (got, want)


@pytest.mark.parametrize("engine", ENGINES)
def test_centre_shifts_every_tone_by_the_centre_frequencys_doppler(engine):
    """The law before, kept as `centre`: every tone at the centre's shift, so each tone is off its
    physical shift by 2 (v . u) (f - fc) / c (here -3.70 and +3.70 Hz)."""
    eng, batched = _engine(engine)
    for doppler in ("centre",):
        got = tone_shifts([one_sector(eng, seed, doppler=doppler, batched=batched) for seed in range(3)])
        centre = expected_shift(FC, "centre")
        print(f"[revdop] {engine} {doppler or 'default'}: tone shifts {np.round(got, 3)} Hz, centre law {centre:.3f}")
        assert np.all(np.abs(got - centre) < TOLERANCE_HZ), got
        physical = np.array([expected_shift(f, PHYSICAL) for f in TONES])
        assert np.all(np.abs(got - physical) > 3.0), (got, physical)


@pytest.mark.parametrize("engine", ["numpy", pytest.param("cpp", marks=needs_cpp)])
def test_per_frequency_is_the_default_bit_for_bit(engine):
    rec_a, rec_b = empty_ping(**REVERB_CASES["moving_uniform"][0]), empty_ping(**REVERB_CASES["moving_uniform"][0])
    reverb.add_reverberation(rec_a, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(3), engine=engine, n_azimuth=90)
    reverb.add_reverberation(rec_b, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(3), engine=engine, n_azimuth=90,
                             doppler="per_frequency")
    np.testing.assert_array_equal(rec_a.samples.array, rec_b.samples.array)


@pytest.mark.parametrize("engine", ["numpy", pytest.param("cpp", marks=needs_cpp)])
def test_a_still_platform_is_the_same_under_either_law(engine):
    """No Doppler, nothing to scale: the two laws give the same record bit for bit."""
    out = []
    for law in ("centre", "per_frequency"):
        rec = empty_ping(**REVERB_CASES["flank_32_uniform_decimated"][0])
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(3), engine=engine, n_azimuth=90,
                                 doppler=law)
        out.append(rec.samples.array)
    np.testing.assert_array_equal(out[0], out[1])


def test_an_unknown_doppler_law_is_refused():
    with pytest.raises(ValueError, match="doppler"):
        reverb.add_reverberation(empty_ping(), FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(0), n_azimuth=8,
                                 doppler="exact")


@needs_cpp
def test_a_build_without_the_per_frequency_law_refuses_it(monkeypatch):
    """Never the old law in silence: an extension that does not know the option raises."""
    monkeypatch.setattr(core, "SIM_REVERB_PER_FREQUENCY", False, raising=False)
    with pytest.raises(RuntimeError, match="per_frequency"):
        reverb.add_reverberation(empty_ping(velocity=(2.0, 0.0, 0.0)), FreeFieldProvider(C), 0.0, MU_DB,
                                 np.random.default_rng(0), n_azimuth=8, engine="cpp", doppler="per_frequency")


# --- the two engines (and the batched formulation) with the option on ------------------------

MOVING_CASES = ["moving_uniform", "moving_displaced_elements", "lfm_tapered_moving", "flank_32_lfm_moving"]
EXTRA_CASES = {
    # the tapered pulse fills the period (n_taps = n_low): the scaled pulse wraps round it
    "lfm_tapered_moving": (dict(waveform={"type": "lfm", "f_start": 25e3, "f_stop": 30e3, "window_alpha": 0.2},
                                velocity=(2.0, -1.0, 0.0)), dict(n_azimuth=40, band_taper=True)),
    # a wide sweep on a long line: where the law matters
    "flank_32_lfm_moving": (dict(name="flank_32", ranges=(60.0, 500.0), fs=16e3, duration=0.05, velocity=(4.0, 1.0, 0.0),
                                 waveform={"type": "lfm", "f_start": 6e3, "f_stop": 10e3, "window_alpha": 0.2}),
                            dict(n_azimuth=90)),
}


def moving_case(name):
    return REVERB_CASES[name] if name in REVERB_CASES else EXTRA_CASES[name]


def run_moving(name, engine, doppler="per_frequency", backend=core.Backend.Host, batched=False):
    ping_kw, reverb_kw = moving_case(name)
    rec = empty_ping(**ping_kw)
    rng = np.random.default_rng(3)
    reverb.BATCHED_ON_HOST = batched
    try:
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, rng, engine=engine, backend=backend,
                                 doppler=doppler, **reverb_kw)
    finally:
        reverb.BATCHED_ON_HOST = False
    return rec.samples.array, rng.bit_generator.state


@needs_cpp
@pytest.mark.parametrize("name", MOVING_CASES)
def test_engines_agree_with_per_frequency(name):
    """At the tolerance of test_sim_cpp.py (REVERB_TOL 5e-6)."""
    (x_cpp, s_cpp), (x_np, s_np) = run_moving(name, "cpp"), run_moving(name, "numpy")
    assert s_cpp == s_np
    worst, rms = relative_difference(x_cpp, x_np)
    print(f"[revdop] engines {name}: worst {worst:.2e}, rms {rms:.2e}")
    assert worst < 5e-6 and rms < 5e-6, (worst, rms)
    # and the law did something on these moving cases
    x_centre, _ = run_moving(name, "numpy", doppler="centre")
    assert relative_difference(x_np, x_centre)[1] > 1e-4


@needs_cpp
@pytest.mark.parametrize("name", MOVING_CASES)
def test_batched_formulation_agrees_with_per_frequency(name):
    """What the CUDA backend runs, on the host (HostExec): at test_batched.py's BATCHED_TOL 5e-6."""
    (ref, s_ref), (got, s_got) = run_moving(name, "cpp"), run_moving(name, "cpp", batched=True)
    assert s_ref == s_got
    worst, rms = relative_difference(got, ref)
    print(f"[revdop] batched {name}: worst {worst:.2e}, rms {rms:.2e}")
    assert worst < 5e-6 and rms < 5e-6, (worst, rms)


@needs_cpp
@pytest.mark.parametrize("name", ["moving_both", "uniform_line_decimated", "k_distributed"])
@pytest.mark.parametrize("batched", [False, True])
def test_bistatic_components_agree_with_per_frequency(name, batched):
    """Components: each row's own Doppler, scaled the same way (Gamma = 1 + doppler / fc)."""
    ping_kw, reverb_kw = ENGINE_CASES[name]
    out = {}
    for engine in ("cpp", "numpy"):
        rec = bistatic_ping(**ping_kw)
        reverb.BATCHED_ON_HOST = batched and engine == "cpp"
        try:
            reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(3), engine=engine,
                                     geometry="bistatic", doppler="per_frequency", **reverb_kw)
        finally:
            reverb.BATCHED_ON_HOST = False
        out[engine] = rec.samples.array
    worst, rms = relative_difference(out["cpp"], out["numpy"])
    assert worst < 5e-6 and rms < 5e-6, (worst, rms)
    rec = bistatic_ping(**ping_kw)
    reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(3), engine="numpy",
                             geometry="bistatic", doppler="centre", **reverb_kw)
    assert relative_difference(out["numpy"], rec.samples.array)[1] > 1e-4        # the law did something


@pytest.mark.parametrize("name", ["flank_32_lfm_moving", "moving_uniform"])
def test_per_frequency_keeps_the_level_against_time(name):
    """The same seeds under either law: the power in 10 blocks along the record, summed over 6 seeds,
    agrees within 0.1 dB, and over the whole record within 0.03 dB (the scaled pulse carries 1 / Gamma
    of the energy: 0.01 dB at 4 m/s). Measured (ours): 6-10 kHz sweep at 4 m/s, blocks -0.025 ..
    +0.047 dB, whole record -0.005 dB, the two laws' records correlated 0.47 (the per-frequency
    shifts reach 10.7 Hz at the band's edges against a resolution of 20 Hz); the 27.5 kHz CW: within
    0.002 dB, correlated 0.9997. Block to block the spread is the sampling of a partly new realisation."""
    totals = {}
    for law in ("centre", "per_frequency"):
        totals[law] = 0.0
        for seed in range(6):
            ping_kw, reverb_kw = moving_case(name)
            rec = empty_ping(**ping_kw)
            reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(seed), engine="numpy",
                                     doppler=law, **reverb_kw)
            totals[law] = totals[law] + np.array([np.mean(np.abs(x) ** 2) for x in np.array_split(rec.samples.array, 10, axis=1)])
    d = 10 * np.log10(totals["per_frequency"] / totals["centre"])
    whole = 10 * np.log10(totals["per_frequency"].sum() / totals["centre"].sum())
    print(f"[revdop] level {name}: per_frequency - centre, 10 blocks {d.min():+.4f} .. {d.max():+.4f} dB, whole {whole:+.4f} dB")
    assert np.abs(d).max() < 0.1, d
    assert abs(whole) < 0.03, whole


# --- the scenario key ----------------------------------------------------------------------

def test_the_scenario_item_key_reaches_the_generator(monkeypatch):
    from test_sim_cpp import SCENARIO
    from gpusonar.sim.simulator import run_scenario
    seen = []
    real = reverb.add_reverberation
    monkeypatch.setattr(reverb, "add_reverberation", lambda *a, **k: seen.append(k.get("doppler")) or real(*a, **k))
    cfg = dict(SCENARIO, pings={"count": 1},
               reverberation=[dict(SCENARIO["reverberation"][0], doppler="per_frequency"), SCENARIO["reverberation"][1]])
    run_scenario(cfg, engine="numpy")
    assert seen == ["per_frequency", None]


def test_the_scenario_item_key_centre_reproduces_the_old_law(monkeypatch):
    from test_sim_cpp import SCENARIO
    from gpusonar.sim.simulator import run_scenario
    cfg = dict(SCENARIO, pings={"count": 1})
    old = dict(cfg, reverberation=[dict(i, doppler="centre") for i in SCENARIO["reverberation"]])
    a, b = run_scenario(old, engine="numpy").records[0], run_scenario(cfg, engine="numpy").records[0]
    assert relative_difference(b.samples.array, a.samples.array)[1] > 1e-4    # the platform moves at 1.5 m/s: the laws differ


# --- CUDA ------------------------------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize("name", MOVING_CASES)
def test_cuda_per_frequency_matches_the_batched_host(name):
    """The device's ReverbPulseKernel with the scaled pulse against the same kernel on HostExec."""
    (ref, s_ref), (got, s_got) = run_moving(name, "cpp", batched=True), run_moving(name, "cpp", backend=core.Backend.Cuda)
    assert s_ref == s_got
    worst, rms = relative_difference(got, ref)
    print(f"[revdop cuda] {name}: worst {worst:.2e}, rms {rms:.2e}")
    assert worst < 1e-4 and rms < 1e-4, (worst, rms)


@needs_cuda
def test_cuda_per_frequency_puts_each_tone_at_its_own_doppler():
    def on_device(seed):
        # one_sector's case on the device
        n_pulse = int(T_PULSE * FS)
        u = np.arange(n_pulse) / FS
        pulse = np.hanning(n_pulse) * sum(np.exp(2j * np.pi * (f - FC) * u) for f in TONES)
        n = int(T_RECORD * FS)
        times = START - T_PULSE + np.arange(n + n_pulse) / FS
        coarse = np.linspace(times[0], times[-1], 40)
        density = np.zeros((N_AZIMUTH, len(coarse)))
        density[SECTOR] = 1.0
        return np.asarray(reverb._series_cpp(np.random.default_rng(seed), coarse, density, False, pulse, np.zeros((1, 3)), FS,
                                             n, START, times[0], FC, C, N_AZIMUTH, 1, len(times), VELOCITY, False, False,
                                             core.Backend.Cuda, doppler="per_frequency"))[0]
    got = tone_shifts([on_device(seed) for seed in range(3)])
    want = np.array([expected_shift(f, PHYSICAL) for f in TONES])
    print(f"[revdop cuda] tone shifts {np.round(got, 3)} Hz, closed form {np.round(want, 3)}")
    assert np.all(np.abs(got - want) < TOLERANCE_HZ)
