import importlib.util
import shutil
import sys

import pytest

sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from circuit_gen import config


def _has_module(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


HAS_KICAD_LIBS = config.KICAD_SYMBOL_DIR.exists() and config.KICAD_FOOTPRINT_DIR.exists()
HAS_SKIDL = importlib.util.find_spec("skidl") is not None  # importing skidl writes log files
HAS_PCBNEW = _has_module("pcbnew")
HAS_FREEROUTING = config.FREEROUTING_JAR.exists() and shutil.which("java") is not None

needs_libs = pytest.mark.skipif(not HAS_KICAD_LIBS, reason="KiCad libraries not installed")
needs_skidl = pytest.mark.skipif(not (HAS_SKIDL and HAS_KICAD_LIBS), reason="SKiDL or KiCad libraries missing")
needs_pcbnew = pytest.mark.skipif(not HAS_PCBNEW, reason="KiCad pcbnew module not installed")
needs_toolchain = pytest.mark.skipif(
    not (HAS_SKIDL and HAS_KICAD_LIBS and HAS_PCBNEW and HAS_FREEROUTING),
    reason="full toolchain (SKiDL, KiCad, FreeRouting) not installed")
