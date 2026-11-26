"""Which implementation the simulator's ported parts run on.

The eigenray providers without a ray tracer (propagation.py), the path pairing of
`two_way_paths` (with per-ray range rates), the per-ping generation of
scattering-function reverberation (reverb.py) and echo synthesis for a sampled transmit
pulse (waveform.py) each exist twice: in NumPy, and in C++ (libs/sim: paths.cpp,
reverb.cpp, synth_kernel.hpp). Echo synthesis for analytic pulses is C++ only.
The C++ one is used when the extension module has it. To select the NumPy one, pass
`engine="numpy"` to the provider or function, or set the environment variable
`GPUSONAR_SIM_ENGINE=numpy` (read at each call).

One engine choice per run (a-engarg; the owner, 2026-11-26): `engine=` given to `run_scenario` or
`simulate_ping` reaches every stage it governs, the eigenray providers they build for themselves included
(the scenario's tier, the free field of a ping given no provider, the straight direct arrival, a basin's
own-ping pulse). Before, those last three ran on the default engine whatever was asked, so an explicit
NumPy ping recorded `eigenrays: cpp` (or `numpy+cpp`), and an explicit `cpp` on a module without the C++
simulator warned about the default's fallback before it was refused. A provider the caller passes in keeps
its own `engine`; when that differs from an explicit `engine=` argument, `check_provider` says so once,
naming both. GPUSONAR_SIM_ENGINE is a default: a provider's own engine wins over it without a word.

What happens when the C++ simulator is not in the loaded module (a-engine; before, every
case below ran NumPy without a word):
- `cpp` asked for by argument or by GPUSONAR_SIM_ENGINE: `EngineUnavailable`, naming the
  missing functions; build the module with SONAR_BUILD_PYTHON=ON or select `numpy`.
- nothing asked for: NumPy, with one RuntimeWarning per process; the record says so (below).
- an unknown name: ValueError naming the nearest engine.

Every record simulate_ping makes says what ran (a-engrec; the owner, 2026-11-26), in its free-form
metadata, not in the schema (docs/RECORD_FORMAT.md):
    extra_json {"gpusonar": {"engine": {"eigenrays": "cpp", "pairing": "cpp",
                                        "synthesis": "cpp_host", "reverberation": "cpp_host"}}}
- eigenrays: "cpp" | "numpy" (free_field, image_source, sediment); "bellhop" | "bellhop3d" (the
  eigenray and eigenray3d tiers, one implementation each). Every provider call in the ping counts
  (the echoes', reverberation's, one-way sources');
- pairing (`two_way_paths`): "cpp" | "numpy";
- synthesis (the two-way echoes): "cpp_host" | "cpp_cuda" | "numpy" (a sampled pulse on the NumPy
  engine, asked for or fallen back to; analytic pulses have the C++ kernel only);
- reverberation (reverb.py): "cpp_host" | "cpp_cuda" | "numpy".
A stage that did not run in the ping is absent (a passive capture's note is {}); one that ran on both
engines in one ping says "cpp+numpy" (first one first). Stages with one implementation shared by both
engines (time-spread taps, basin tail, noise, front end, ADC, board clocks) are not listed. The older
note of a sampled pulse on NumPy, {"gpusonar": {"echo_synthesis": "numpy_sampled"}} (waveform.py),
is kept as it was.
A module that has the C++ simulator but is older than the sources (a binding missing, or one
without a keyword the Python passes) fails at the call with `StaleModuleError` naming the
binding (`call`), not with a bare AttributeError or TypeError.

The two give the same results: the C++ follows the NumPy arithmetic step by step, and the
random numbers of reverberation are drawn by the NumPy generator on both.
"""

from __future__ import annotations

import contextvars
import difflib
import json
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


def provider_engine(provider) -> str | None:
    """The engine a provider's eigenrays will run on by its own setting: its `engine`, else GPUSONAR_SIM_ENGINE,
    else the default ('cpp' when built). A wrapper without an engine of its own (sim/timespread.py) is looked
    through to its `inner`. None for a provider with no engine choice (Bellhop). Raises nothing for a missing C++
    simulator (the provider's call does)."""
    seen = 0
    while provider is not None and not hasattr(provider, "engine") and hasattr(provider, "inner") and seen < 16:
        provider, seen = provider.inner, seen + 1
    if provider is None or not hasattr(provider, "engine"):
        return None
    if provider.engine:
        return _check_name(str(provider.engine), "the provider's engine")
    if os.environ.get("GPUSONAR_SIM_ENGINE"):
        return _check_name(os.environ["GPUSONAR_SIM_ENGINE"], "GPUSONAR_SIM_ENGINE")
    return "cpp" if HAVE_CPP else "numpy"


def check_provider(provider, engine: str | None, where: str) -> None:
    """A provider passed in by the caller keeps its own engine; when an explicit `engine` argument names another,
    say so (one RuntimeWarning naming both). Nothing when no argument is given or the provider has no engine
    choice."""
    if not engine:
        return
    asked = _check_name(str(engine), "engine argument")
    own = provider_engine(provider)
    if own is not None and own != asked:
        warnings.warn(f"{where}: the provider passed in runs its eigenrays on the {own!r} engine (its own `engine`, "
                      f"which it keeps), but the engine argument asks for {asked!r}: the other stages run on {asked!r}. "
                      f"To run one engine, give the provider engine={asked!r}, or leave the argument out",
                      RuntimeWarning, stacklevel=3)


# --- what ran, into the record (a-engrec) ---------------------------------------------------------------------------

STAGES = ("eigenrays", "pairing", "synthesis", "reverberation")
# The stages that ran in the ping being simulated: {stage: [what, ...]}; None outside simulate_ping (then `ran` keeps
# nothing; after a ping that raised, its unread collection until the next ping). A context variable, so that pings
# simulated in other threads keep their own.
_ran: contextvars.ContextVar = contextvars.ContextVar("gpusonar_sim_ran", default=None)


def ran(stage: str, what: str) -> None:
    """Say that `stage` (one of STAGES) ran on `what` in the ping being simulated; nothing outside a ping."""
    seen = _ran.get()
    if seen is not None:
        got = seen.setdefault(stage, [])
        if what not in got:
            got.append(what)


def on_backend(backend) -> str:
    """'cpp_host' or 'cpp_cuda': the C++ code on the backend it was given (it has no silent fallback: a CUDA
    backend on a build without CUDA raises)."""
    return "cpp_cuda" if backend == core.Backend.Cuda else "cpp_host"


def begin_ping() -> contextvars.Token:
    """Start collecting what runs in one ping (simulate_ping). A ping that raises leaves its collection behind
    until the next ping replaces it; nothing reads it."""
    return _ran.set({})


def end_ping(record: "core.PingRecord", token: contextvars.Token) -> None:
    """Write what ran in this ping into record.metadata.extra_json under gpusonar.engine (a few short strings; the
    other keys are kept as they were) and stop collecting."""
    seen = _ran.get() or {}
    _ran.reset(token)
    extra = json.loads(record.metadata.extra_json or "{}")
    extra.setdefault("gpusonar", {})["engine"] = {k: "+".join(seen[k]) for k in STAGES if k in seen}
    meta = record.metadata
    meta.extra_json = json.dumps(extra)
    record.metadata = meta
