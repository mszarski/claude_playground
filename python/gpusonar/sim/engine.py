"""Which implementation the simulator's ported parts run on.

The eigenray providers without a ray tracer (propagation.py), the path pairing of
`two_way_paths` (with per-ray range rates), the per-ping generation of
scattering-function reverberation (reverb.py) and echo synthesis for a sampled transmit
pulse (waveform.py) each exist twice: in NumPy, and in C++ (libs/sim: paths.cpp,
reverb.cpp, synth_kernel.hpp). Echo synthesis for analytic pulses is C++ only.
The C++ one is used when the extension module has it. To select the NumPy one, pass
`engine="numpy"` to the provider or function, or set the environment variable
`GPUSONAR_SIM_ENGINE=numpy` (read at each call).

What happens when the C++ simulator is not in the loaded module (a-engine; before, every
case below ran NumPy without a word):
- `cpp` asked for by argument or by GPUSONAR_SIM_ENGINE: `EngineUnavailable`, naming the
  missing functions; build the module with SONAR_BUILD_PYTHON=ON or select `numpy`.
- nothing asked for: NumPy, with one RuntimeWarning per process. Of the records, only a
  sampled pulse's says which engine ran (extra_json {"gpusonar": {"echo_synthesis":
  "numpy_sampled"}}, waveform.py); nothing else in a record names the engine.
- an unknown name: ValueError naming the nearest engine.
A module that has the C++ simulator but is older than the sources (a binding missing, or one
without a keyword the Python passes) fails at the call with `StaleModuleError` naming the
binding (`call`), not with a bare AttributeError or TypeError.

The two give the same results: the C++ follows the NumPy arithmetic step by step, and the
random numbers of reverberation are drawn by the NumPy generator on both.
"""

from __future__ import annotations

import difflib
import os
import warnings

from .. import _gpusonar as core

ENGINES = ("cpp", "numpy")
# The bindings whose presence says the module was built with the C++ simulator (python/bindings/bind_simcpp.cpp).
# Bindings added later are checked where they are called (`call`).
REQUIRED = ("sim_reverb_accumulate", "sim_two_way_paths")
HAVE_CPP = all(hasattr(core, name) for name in REQUIRED)

_HOW = ("build the extension module with SONAR_BUILD_PYTHON=ON (cmake --build <build dir>; it lands in "
        "python/gpusonar/), or select the NumPy engine (engine=\"numpy\" or GPUSONAR_SIM_ENGINE=numpy)")
_warned_default = False


class EngineUnavailable(RuntimeError):
    """The C++ simulator engine was asked for and the loaded extension module does not have it."""


class StaleModuleError(RuntimeError):
    """The loaded extension module is older than the Python sources: a binding they call is missing or
    does not take an argument they pass."""


def _module_file() -> str:
    return getattr(core, "__file__", None) or "gpusonar._gpusonar"


def _check_name(name: str, where: str) -> str:
    low = name.lower()
    if low not in ENGINES:
        near = difflib.get_close_matches(low, ENGINES, n=1, cutoff=0.4)
        hint = f"did you mean {near[0]!r}? " if near else ""
        raise ValueError(f"unknown simulator engine {name!r} ({where}): {hint}use one of {ENGINES}")
    return low


def resolve(engine: str | None = None) -> str:
    """'cpp' or 'numpy': the argument, else GPUSONAR_SIM_ENGINE, else C++ when it is built.

    An explicit 'cpp' (argument or environment variable) without the C++ simulator in the module raises
    EngineUnavailable; with nothing asked for the default falls back to NumPy with one warning per process.
    """
    global _warned_default
    if engine:
        name, where = _check_name(str(engine), "engine argument"), "the engine argument"
    elif os.environ.get("GPUSONAR_SIM_ENGINE"):
        name, where = _check_name(os.environ["GPUSONAR_SIM_ENGINE"], "GPUSONAR_SIM_ENGINE"), "GPUSONAR_SIM_ENGINE"
    else:
        name, where = "cpp", None
    if name == "numpy" or HAVE_CPP:
        return name
    missing = ", ".join(n for n in REQUIRED if not hasattr(core, n))
    missing = f" (missing: {missing})" if missing else ""
    if where is not None:
        raise EngineUnavailable(f"simulator engine 'cpp' asked for by {where}, but the extension module "
                                f"{_module_file()} was built without the C++ simulator{missing}: {_HOW}")
    if not _warned_default:
        _warned_default = True
        warnings.warn(f"the extension module {_module_file()} was built without the C++ simulator{missing}: the "
                      f"simulator runs on its NumPy engine (same results, slower). To silence this, {_HOW}",
                      RuntimeWarning, stacklevel=3)
    return "numpy"


def call(name: str, *args, **kwargs):
    """Call the extension module's binding `name`; a module older than the sources fails with
    StaleModuleError naming the binding (and the keyword it lacks), not a bare AttributeError/TypeError.

    Every call of a `sim_*` binding in gpusonar/sim (propagation.py, reverb.py) goes through here, so a binding
    or keyword added to python/bindings later is covered by calling it this way. Looked up at each call (tests
    replace bindings on the module)."""
    fn = getattr(core, name, None)
    if fn is None:
        raise StaleModuleError(f"the extension module {_module_file()} has no {name}: it is older than the "
                               f"sources (python/bindings); rebuild it (cmake --build <build dir>, "
                               f"SONAR_BUILD_PYTHON=ON), or select the NumPy engine (engine=\"numpy\" or "
                               f"GPUSONAR_SIM_ENGINE=numpy)")
    try:
        return fn(*args, **kwargs)
    except TypeError as e:
        if not str(e).startswith(f"{name}(): incompatible function arguments"):
            raise
        signature = (getattr(fn, "__doc__", None) or "").split("\n", 1)[0]
        lacking = [k for k in kwargs if f"({k}:" not in signature and f" {k}:" not in signature]
        if lacking:
            raise StaleModuleError(f"the extension module {_module_file()} has a {name} without the argument(s) "
                                   f"{', '.join(lacking)}: it is older than the sources (python/bindings); rebuild "
                                   f"it (cmake --build <build dir>, SONAR_BUILD_PYTHON=ON), or select the NumPy "
                                   f"engine (engine=\"numpy\" or GPUSONAR_SIM_ENGINE=numpy)") from e
        e.add_note(f"if {_module_file()} was built from older sources than python/gpusonar, rebuild it")
        raise
