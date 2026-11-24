"""Every YAML file in the repository writes its numbers so that PyYAML's plain loader reads them as numbers (sonar-iwq).

PyYAML follows YAML 1.1: a float there needs a dot in the mantissa and a SIGNED exponent, so `38.0e3`, `1e3` and
`2.5e6` load as STRINGS. python/gpusonar/config.py's loader reads them as floats, but other readers do not (plain
`yaml.safe_load`, the sediment table, the WAV and navigation adapters, a user's own script), and the C++ dictionary
readers then refuse the string with a cast error. Write `38.0e+3` or `38000.0`.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

# A string a person means as a number: a decimal with an exponent (signed or not), or a spelled-out infinity or
# NaN. YAML 1.1's own forms (`1.0e+3`, `.inf`, `.nan`) never come out as strings, so whatever matches here is
# one PyYAML did not read as a number.
NUMBERISH = re.compile(r"^[-+]?(?:\d[\d_]*\.?\d*|\.\d+)[eE][-+]?\d+$|^[-+]?(?:inf|infinity|nan)$", re.IGNORECASE)


def yaml_files() -> list[Path]:
    """The YAML files git tracks (configs, docs/examples, test fixtures, ...), or every one under the root without git."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.yaml", "*.yml"], capture_output=True, text=True, check=True).stdout
        files = [ROOT / f for f in out.split()]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in ROOT.rglob("*.y*ml") if not any(part.startswith(".") or part == "build" for part in p.relative_to(ROOT).parts)]
    return sorted(p for p in files if p.suffix in (".yaml", ".yml") and p.exists())


def numberish_strings(value, where=""):
    """(path, text) of every string value or key under `value` that NUMBERISH matches."""
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and NUMBERISH.match(k.strip()):
                yield f"{where}.{k} (key)", k
            yield from numberish_strings(v, f"{where}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from numberish_strings(v, f"{where}[{i}]")
    elif isinstance(value, str) and NUMBERISH.match(value.strip()):
        yield where, value


def test_the_pattern_catches_what_pyyaml_reads_as_a_string():
    for text in ("38.0e3", "1e3", "2.5E6", "-1.5e3", "1e-05", "inf", "NaN"):
        assert isinstance(yaml.safe_load(f"v: {text}")["v"], str) and NUMBERISH.match(text), text
    for text in ("38.0e+3", "1.0e-05", "-1.5e-3", "38000.0", ".inf", "-.inf", ".nan", "12"):
        assert not isinstance(yaml.safe_load(f"v: {text}")["v"], str), text
    for text in ("e3", "1.2.3", "v1e3", "kHz", "0x1e3"):
        assert not NUMBERISH.match(text), text


def test_the_repository_has_yaml_files():
    files = [p.relative_to(ROOT).as_posix() for p in yaml_files()]
    assert len(files) > 100 and "configs/deploy/popoto_elevation_rehearsal.yaml" in files and "configs/sediments.yaml" in files


@pytest.mark.parametrize("path", yaml_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_number_is_read_as_a_string(path):
    found = [f"{where}: {text!r}" for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) for where, text in numberish_strings(doc)]
    assert not found, f"{path.relative_to(ROOT)}: PyYAML reads these as strings (write e.g. 38.0e+3 or 38000.0): {found}"
