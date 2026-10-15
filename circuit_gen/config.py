"""
Tool locations. Each can be overridden with an environment variable.
"""

import os
from pathlib import Path

KICAD_SYMBOL_DIR = Path(os.environ.get("CIRCUIT_GEN_SYMBOL_DIR", "/usr/share/kicad/symbols"))
KICAD_FOOTPRINT_DIR = Path(os.environ.get("CIRCUIT_GEN_FOOTPRINT_DIR", "/usr/share/kicad/footprints"))
FREEROUTING_JAR = Path(os.environ.get("CIRCUIT_GEN_FREEROUTING_JAR", "/opt/freerouting/freerouting.jar"))
KICAD_CLI = os.environ.get("CIRCUIT_GEN_KICAD_CLI", "kicad-cli")
PYTHON = os.environ.get("CIRCUIT_GEN_PYTHON", "python3")

MODEL = os.environ.get("CIRCUIT_GEN_MODEL", "claude-opus-5-5")
