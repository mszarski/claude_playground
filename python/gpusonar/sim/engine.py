"""Which implementation the simulator's ported parts run on.

The eigenray providers without a ray tracer (propagation.py), the path pairing of
`two_way_paths` (with per-ray range rates), the per-ping generation of
scattering-function reverberation (reverb.py) and echo synthesis for a sampled transmit
pulse (waveform.py) each exist twice: in NumPy, and in C++ (libs/sim: paths.cpp,
reverb.cpp, synth_kernel.hpp). Echo synthesis for analytic pulses is C++ only.
The C++ one is used when the extension module has it. To select the NumPy one, pass
`engine="numpy"` to the provider or function, or set the environment variable
`GPUSONAR_SIM_ENGINE=numpy` (read at each call).

The two give the same results: the C++ follows the NumPy arithmetic step by step, and the
random numbers of reverberation are drawn by the NumPy generator on both.
"""

from __future__ import annotations

import os

from .. import _gpusonar as core

ENGINES = ("cpp", "numpy")
HAVE_CPP = hasattr(core, "sim_reverb_accumulate") and hasattr(core, "sim_two_way_paths")


def resolve(engine: str | None = None) -> str:
    """'cpp' or 'numpy': the argument, else GPUSONAR_SIM_ENGINE, else C++ when it is built."""
    name = (engine or os.environ.get("GPUSONAR_SIM_ENGINE") or "cpp").lower()
    if name not in ENGINES:
        raise ValueError(f"unknown simulator engine {name!r}: use one of {ENGINES}")
    return name if HAVE_CPP else "numpy"
