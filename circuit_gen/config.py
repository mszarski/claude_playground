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

# Which model writes the circuit: "hf" (open models via Hugging Face Inference
# Providers, needs HF_TOKEN) or "claude" (Anthropic API).
BACKEND = os.environ.get("CIRCUIT_GEN_BACKEND", "hf")
HF_MODEL = os.environ.get("CIRCUIT_GEN_HF_MODEL", "Qwen/Qwen3.8-27B")
HF_PROVIDER = os.environ.get("CIRCUIT_GEN_HF_PROVIDER", "auto")
HF_MAX_TOKENS = int(os.environ.get("CIRCUIT_GEN_HF_MAX_TOKENS", "65536"))
CLAUDE_MODEL = os.environ.get("CIRCUIT_GEN_CLAUDE_MODEL", "claude-opus-5-5")
