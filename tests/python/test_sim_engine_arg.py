"""a-engarg: one engine choice per run reaches every stage it governs (the owner, 2026-11-26: "fix the engine argument
gap too").

Before this package the `engine` argument did not reach three eigenray providers the simulator builds for itself:
- `simulate_ping(engine=...)` without a provider built `make_provider("free_field", c)` with no engine, so an
  explicit-NumPy ping ran its eigenrays on the default engine (C++ when built; the record said `eigenrays: cpp`
  beside `pairing: numpy`), and `engine="cpp"` on a module without the C++ simulator first warned that the default
  fell back to NumPy, then was refused;
- `sim/basin.py` `transmitted_pulse` (the basin's answer to the ping itself, `own_ping_db`) built its free-field
  provider the same way (the record then read `eigenrays: numpy+cpp`);
- `sim/sources.py` `direct_arrival` with `transmit.direct_path: {boundaries: false}` likewise.
Now an explicit `engine=` (or GPUSONAR_SIM_ENGINE, which those providers read at each call, as before) applies to
every provider the simulator builds itself. A provider the caller passes in keeps its own engine ("a provider has
its own engine"); when that differs from an explicit `engine=` argument, `simulate_ping` says so once, naming both.

The C++-engine assertions need the C++ simulator in the module (needs_cpp); "module without it" is simulated by
setting `engine.HAVE_CPP` false. The default path (no engine given) is unchanged: sha256 of one ping of shipped
scenarios before and after is in the package's report."""

import json
import warnings

import numpy as np
import pytest

from gpusonar.sim import FreeFieldProvider, ImageSourceProvider, Scene, engine, make_pose, run_scenario, simulate_ping
from gpusonar.sim import timespread
from gpusonar.sim import waveform as wfm

from test_sampled_waveform import C, IMAGE, TIERS, _five_targets, _scenario, sonar_with

needs_cpp = pytest.mark.skipif(not engine.HAVE_CPP, reason="needs the C++ simulator in the extension module")
LFM = {"type": "lfm"}
SAMPLED = {"type": "sampled", "source": "lfm", "duration": 0.002, "window_alpha": 0.2}
# The basin's answer to the ping itself switched on (own_ping_db: a what-if value, as in sim/basin.py's example).
BASIN = {"energy_db": -4.2, "onset_s": 0.010, "decay_s": 0.145, "coherence_m": "isotropic", "repeatable": 0.7,
         "own_ping_db": -43.0}
EIGENRAY_BINDINGS = ("sim_free_field_eigenrays", "sim_image_source_eigenrays", "sim_plane_image_eigenrays",
                     "sim_sediment_eigenrays", "sim_snell_crossing")


def note(record):
    return json.loads(record.metadata.extra_json or "{}").get("gpusonar", {}).get("engine")


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)
    monkeypatch.setattr(engine, "_warned_default", False, raising=False)
    monkeypatch.setattr(wfm, "_warned", False)


@pytest.fixture
def no_cpp(clean_env, monkeypatch):
    monkeypatch.setattr(engine, "HAVE_CPP", False)


@pytest.fixture
def cpp_calls(monkeypatch):
    """Every C++ simulator binding called (sim/engine.py `call`), by name."""
    seen = []
    real = engine.call

    def counting(name, *args, **kwargs):
        seen.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(engine, "call", counting)
    return seen


def sampled_ping(**kwargs):
    sonar = sonar_with("sas_auv", SAMPLED, 18.0, 22.0, omni=True)
    return simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False,
                         **kwargs)


def basin_scenario():
    return _scenario("image_source", LFM, propagation=IMAGE, basin_reverberation=dict(BASIN))


def direct_only_ping(**kwargs):
    """A ping with nothing in the scene and the projector heard straight (no boundaries) inside the window."""
    sonar = sonar_with("sas_auv", {"type": "lfm", "duration": 0.002}, 0.5, 6.0, omni=True,
                       transmit={"direct_path": {"boundaries": False}})
    return simulate_ping(sonar, make_pose(), Scene(), environment={"sound_speed": C}, adc=False, **kwargs)


# --- an explicit NumPy run is NumPy in every stage it has ----------------------------------------------------------

@needs_cpp
def test_explicit_numpy_without_a_provider_runs_every_stage_on_numpy(clean_env, cpp_calls):
    """Before: {"eigenrays": "cpp", "pairing": "numpy", "synthesis": "numpy"} and two sim_free_field_eigenrays calls."""
    rec = sampled_ping(engine="numpy")
    assert note(rec) == {"eigenrays": "numpy", "pairing": "numpy", "synthesis": "numpy"}
    assert cpp_calls == []


@needs_cpp
def test_explicit_numpy_with_a_basin_tail_runs_its_pulse_on_numpy(clean_env, cpp_calls):
    """The own-ping pulse of the basin tail (sim/basin.py transmitted_pulse). Before: eigenrays "numpy+cpp"."""
    rec = run_scenario(basin_scenario(), engine="numpy").records[0]
    assert note(rec)["eigenrays"] == "numpy" and note(rec)["pairing"] == "numpy"
    assert not [n for n in cpp_calls if n in EIGENRAY_BINDINGS]


@needs_cpp
def test_explicit_numpy_direct_arrival_without_boundaries_runs_on_numpy(clean_env, cpp_calls):
    """transmit.direct_path {boundaries: false} (sim/sources.py direct_arrival). Before: eigenrays "cpp"."""
    rec = direct_only_ping(engine="numpy")
    assert note(rec) == {"eigenrays": "numpy"}         # one-way synthesis is not a recorded stage (sim/engine.py)
    assert cpp_calls == [] and np.any(rec.samples.array)


def test_explicit_numpy_without_the_cpp_simulator_says_nothing(no_cpp):
    """NumPy asked for: no default-fallback warning (before: one, from the free-field provider the ping built)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec = sampled_ping(engine="numpy")
        basin = run_scenario(basin_scenario(), engine="numpy").records[0]
        direct = direct_only_ping(engine="numpy")
    assert note(rec) == {"eigenrays": "numpy", "pairing": "numpy", "synthesis": "numpy"}
    assert note(basin)["eigenrays"] == "numpy" and note(direct)["eigenrays"] == "numpy"


# --- an explicit C++ run without the C++ simulator is refused, without a word first ---------------------------------

@pytest.mark.parametrize("run", ["ping", "basin", "direct"])
def test_explicit_cpp_without_the_cpp_simulator_raises_with_no_warning(no_cpp, run):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(engine.EngineUnavailable, match="engine argument"):
            if run == "ping":
                sampled_ping(engine="cpp")
            elif run == "basin":
                run_scenario(basin_scenario(), engine="cpp")
            else:
                direct_only_ping(engine="cpp")
    assert [str(w.message) for w in caught] == []


# --- the environment variable behaves as the argument does ---------------------------------------------------------

@needs_cpp
def test_the_environment_variable_numpy_reaches_the_same_stages(clean_env, monkeypatch, cpp_calls):
    """The variable reached those providers before as now (they read GPUSONAR_SIM_ENGINE at each call); the argument
    did not, so this failed before on the argument's side (C++ eigenray calls). Both now give the same record."""
    by_arg = [sampled_ping(engine="numpy"), run_scenario(basin_scenario(), engine="numpy").records[0],
              direct_only_ping(engine="numpy")]
    monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpy")
    by_env = [sampled_ping(), run_scenario(basin_scenario()).records[0], direct_only_ping()]
    for a, b in zip(by_arg, by_env):
        assert note(a) == note(b)
        assert a.samples.array.tobytes() == b.samples.array.tobytes()
    assert cpp_calls == []


# --- a provider passed in keeps its engine; a conflict with the argument is said once ------------------------------

def _conflicts(caught):
    return [str(w.message) for w in caught if "the provider passed in" in str(w.message)]


@needs_cpp
@pytest.mark.parametrize("own, asked", [("numpy", "cpp"), ("cpp", "numpy"), (None, "numpy")])
def test_a_provider_whose_engine_conflicts_warns_once_and_keeps_its_engine(clean_env, own, asked):
    """`own` None: the provider leaves it to the default (C++ here), which an explicit "numpy" contradicts."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = sampled_ping(provider=FreeFieldProvider(C, engine=own), engine=asked)
    msgs = _conflicts(caught)
    assert len(msgs) == 1
    ran = own or "cpp"
    assert f"'{ran}'" in msgs[0] and f"'{asked}'" in msgs[0]
    assert note(rec)["eigenrays"] == ran                                    # not overridden
    assert note(rec)["pairing"] == asked


@needs_cpp
def test_a_wrapped_provider_is_looked_through(clean_env):
    """A time-spread provider (sim/timespread.py) has no engine of its own: its inner provider's counts."""
    inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1, engine="cpp")
    wrapped = timespread.TimeSpreadProvider(inner, frequency=200e3, surface={"rms_height": 0.1, "mean_square_slope": 0.01,
                                                                            "rms_vertical_speed": 0.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = sampled_ping(provider=wrapped, engine="numpy")
    assert len(_conflicts(caught)) == 1
    assert inner.engine == "cpp" and note(rec)["eigenrays"] == "cpp"


@pytest.mark.parametrize("own, asked", [("numpy", "numpy"), ("numpy", None), ("cpp", None), (None, None)])
def test_no_warning_when_the_provider_agrees_or_nothing_is_asked(clean_env, own, asked):
    if "cpp" in (own, asked) and not engine.HAVE_CPP:
        pytest.skip("needs the C++ simulator in the extension module")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampled_ping(provider=FreeFieldProvider(C, engine=own), engine=asked)


def test_the_environment_variable_does_not_warn_against_a_providers_own_engine(clean_env, monkeypatch):
    """GPUSONAR_SIM_ENGINE is a default: a provider's own engine wins over it without a word, as in `resolve`."""
    if not engine.HAVE_CPP:
        pytest.skip("needs the C++ simulator in the extension module")
    monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec = sampled_ping(provider=FreeFieldProvider(C, engine="cpp"))
    assert note(rec)["eigenrays"] == "cpp"


# --- no mixed stage in a single-engine run -------------------------------------------------------------------------

RUNS = {
    "free_field": lambda e: run_scenario(_scenario("free_field", LFM, **TIERS["free_field"]), engine=e).records[0],
    "image_source + reverberation": lambda e: run_scenario(
        _scenario("image_source", LFM, **TIERS["image_source + reverberation"]), engine=e).records[0],
    "image_source + time spread": lambda e: run_scenario(
        _scenario("image_source", LFM, **TIERS["image_source + time spread"]), engine=e).records[0],
    "basin tail": lambda e: run_scenario(basin_scenario(), engine=e).records[0],
    "ping without a provider": lambda e: sampled_ping(engine=e),
    "direct arrival": lambda e: direct_only_ping(engine=e),
}


@pytest.mark.parametrize("eng", [None, "numpy", pytest.param("cpp", marks=needs_cpp)])
@pytest.mark.parametrize("run", list(RUNS))
def test_no_stage_runs_on_two_engines(clean_env, run, eng):
    stages = note(RUNS[run](eng))
    assert stages and not [v for v in stages.values() if "+" in v], stages
    if eng == "numpy":
        assert all(v == "numpy" for k, v in stages.items() if k != "synthesis"), stages
