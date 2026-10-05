"""Render Reachy Mini moves in MuJoCo: videos, contact sheets, grids (``render`` extra).

Headless: set ``MUJOCO_GL=egl`` on a machine with a GPU, or ``MUJOCO_GL=osmesa`` (``apt install libosmesa6``)
on CPU only.

Import this package before ``mujoco``. With OSMesa, MuJoCo loads Mesa's system LLVM; if torch later loads Triton
(which links its own LLVM statically), Triton's static initialisers bind to Mesa's LLVM symbols and the process
segfaults the next time torch trains. Loading Triton first avoids it.
"""
import importlib.util
import sys

if "mujoco" not in sys.modules and importlib.util.find_spec("triton") is not None:
    try:
        import triton._C.libtriton  # noqa: F401
    except ImportError:
        pass
