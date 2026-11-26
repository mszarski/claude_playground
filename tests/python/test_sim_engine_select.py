"""a-engine: what `sim/engine.py` does when the C++ simulator is missing from the extension module, or the module is
older than the sources.

Before, `resolve("cpp")` (argument or GPUSONAR_SIM_ENGINE) returned "numpy" without a word when the module lacked
the simulator, and a module that had the simulator but lacked a newer binding failed later with a bare
AttributeError or TypeError. Now: an explicit `cpp` is refused (EngineUnavailable), the default falls back to NumPy
with one warning per process, an unknown name is refused with the nearest valid one, and a stale binding fails with
StaleModuleError naming it. "Module absent" is simulated by setting `engine.HAVE_CPP` false, "stale" by removing or
replacing a binding on the module."""

import json
import warnings

import numpy as np
import pytest

from gpusonar import _gpusonar as core
from gpusonar.sim import FreeFieldProvider, SedimentProvider, engine, make_pose, simulate_ping
from gpusonar.sim.propagation import two_way_paths

from test_sampled_waveform import C, _five_targets, sonar_with

POINTS = np.array([[10.0, 20.0, 30.0], [5.0, -3.0, 40.0]])


@pytest.fixture
def no_cpp(monkeypatch):
    """The module as if built without the C++ simulator; the once-per-process warning re-armed."""
    monkeypatch.setattr(engine, "HAVE_CPP", False)
    monkeypatch.setattr(engine, "_warned_default", False, raising=False)
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)
    monkeypatch.setattr(engine, "_warned_default", False, raising=False)


# --- rule 1: an explicit "cpp" without the C++ simulator is refused --------------------------------------------------

@pytest.mark.parametrize("arg", ["cpp", "CPP"])
def test_explicit_cpp_argument_without_the_module_is_refused(no_cpp, arg):
    with pytest.raises(engine.EngineUnavailable, match=r"engine argument.*built without the C\+\+ simulator") as e:
        engine.resolve(arg)
    msg = str(e.value)
    assert "SONAR_BUILD_PYTHON=ON" in msg and "numpy" in msg


def test_explicit_cpp_environment_variable_without_the_module_is_refused(no_cpp, monkeypatch):
    monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "cpp")
    with pytest.raises(engine.EngineUnavailable, match="GPUSONAR_SIM_ENGINE"):
        engine.resolve()
    assert engine.resolve("numpy") == "numpy"            # the argument still wins over the environment


def test_the_error_names_the_missing_bindings(no_cpp, monkeypatch):
    monkeypatch.delattr(core, "sim_two_way_paths")
    with pytest.raises(engine.EngineUnavailable, match="missing: sim_two_way_paths"):
        engine.resolve("cpp")


def test_a_provider_and_the_path_pairing_asked_for_cpp_do_not_run_numpy(no_cpp):
    with pytest.raises(engine.EngineUnavailable):
        FreeFieldProvider(C, engine="cpp").eigenrays(np.zeros(3), POINTS)
    with pytest.raises(engine.EngineUnavailable):
        SedimentProvider(C, 1650.0, 35.0, engine="cpp").eigenrays(np.zeros(3), POINTS)
    with pytest.raises(engine.EngineUnavailable):
        two_way_paths(FreeFieldProvider(C, engine="numpy"), np.zeros(3), np.zeros(3), POINTS, np.ones(2), engine="cpp")


@pytest.mark.filterwarnings("ignore:the extension module")   # simulate_ping's own provider takes the default engine
def test_simulate_ping_asked_for_cpp_is_refused(no_cpp):
    sonar = sonar_with("sas_auv", {"type": "lfm", "duration": 0.002}, 18.0, 22.0, omni=True)
    with pytest.raises(engine.EngineUnavailable):
        simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False,
                      engine="cpp")


# --- rule 2: the default falls back, once, aloud ---------------------------------------------------------------------

def test_the_default_without_the_module_runs_numpy_with_one_warning(no_cpp):
    with pytest.warns(RuntimeWarning, match=r"built without the C\+\+ simulator.*NumPy engine") as w:
        assert engine.resolve() == "numpy"
    assert len(w) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert engine.resolve() == "numpy"                # once per process
        assert engine.resolve(None) == "numpy"


def test_numpy_asked_for_without_the_module_is_silent(no_cpp, monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert engine.resolve("numpy") == "numpy"
        monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpy")
        assert engine.resolve() == "numpy"


def test_the_default_fallback_is_recorded_where_a_record_names_its_engine(no_cpp):
    """The only place a record names the simulator engine is the sampled pulse's note (waveform.py); the default
    fallback writes it, as a NumPy run asked for does."""
    wave = {"type": "sampled", "source": "lfm", "duration": 0.002, "window_alpha": 0.2}
    sonar = sonar_with("sas_auv", wave, 18.0, 22.0, omni=True)
    with pytest.warns(RuntimeWarning, match="NumPy engine"):
        rec = simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False)
    assert json.loads(rec.metadata.extra_json)["gpusonar"]["echo_synthesis"] == "numpy_sampled"


@pytest.mark.skipif(not engine.HAVE_CPP, reason="needs the C++ simulator in the extension module")
def test_the_default_with_the_module_is_cpp_and_silent(clean_env):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert engine.resolve() == "cpp"
        assert engine.resolve("cpp") == "cpp"
        assert engine.resolve("numpy") == "numpy"


# --- rule 3: an unknown name is refused with the nearest valid one ---------------------------------------------------

@pytest.mark.parametrize("name, near", [("cp", "cpp"), ("cppp", "cpp"), ("nunpy", "numpy"), ("num", "numpy")])
def test_an_unknown_engine_argument_names_the_nearest(clean_env, name, near):
    with pytest.raises(ValueError, match=rf"unknown simulator engine.*did you mean '{near}'"):
        engine.resolve(name)


def test_an_unknown_environment_value_names_the_nearest(clean_env, monkeypatch):
    monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpi")
    with pytest.raises(ValueError, match=r"GPUSONAR_SIM_ENGINE.*did you mean 'numpy'"):
        engine.resolve()


def test_an_unknown_name_is_refused_without_the_module_too(no_cpp):
    with pytest.raises(ValueError, match="did you mean 'cpp'"):
        engine.resolve("cpq")
    with pytest.raises(ValueError, match="use one of"):
        engine.resolve("fortran")


# --- rule 4: a module older than the sources fails naming the binding ------------------------------------------------

def test_a_missing_binding_fails_naming_it(clean_env, monkeypatch):
    monkeypatch.delattr(core, "sim_free_field_eigenrays")
    with pytest.raises(engine.StaleModuleError, match="has no sim_free_field_eigenrays: it is older than the sources"):
        FreeFieldProvider(C, engine="cpp").eigenrays(np.zeros(3), POINTS)
    with pytest.raises(engine.StaleModuleError, match="numpy"):
        engine.call("sim_free_field_eigenrays", C, (0.0, 0.0, 0.0), POINTS)


def test_a_binding_without_a_keyword_fails_naming_it(monkeypatch):
    """A binding built before a keyword was added refuses it with pybind11's 'incompatible function arguments'
    TypeError; the helper names the binding and the keyword. Simulated by a function with pybind11's message and
    signature line."""
    def old(*args, **kwargs):
        raise TypeError("sim_test_binding(): incompatible function arguments. The following argument types are "
                        "supported:\n    1. (a: float, b: float = 1.0) -> float")
    old.__doc__ = "sim_test_binding(a: float, b: float = 1.0) -> float\n\nA binding before `roughness`."
    monkeypatch.setattr(core, "sim_test_binding", old, raising=False)
    with pytest.raises(engine.StaleModuleError, match=r"sim_test_binding without the argument\(s\) roughness.*older"):
        engine.call("sim_test_binding", 1.0, b=2.0, roughness=None)


def test_a_wrong_argument_to_a_current_binding_stays_a_type_error():
    """Not every TypeError is a stale module: a wrong argument type to a binding that has every keyword stays a
    TypeError (with a note on rebuilding)."""
    with pytest.raises(TypeError, match="incompatible function arguments") as e:
        engine.call("sim_free_field_eigenrays", "not a speed", (0.0, 0.0, 0.0), POINTS)
    assert not isinstance(e.value, engine.StaleModuleError)
    assert any("rebuild" in n for n in getattr(e.value, "__notes__", []))
