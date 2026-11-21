"""The oracle gate (oracle_gate.py): an oracle test skips without its program, and fails instead
when GPUSONAR_REQUIRE_ORACLES is set; other skips are left alone (audit B5)."""

import os
import pathlib
import re
import subprocess
import sys

import pytest

from oracle_gate import oracle_variable

HERE = pathlib.Path(__file__).resolve().parent

SAMPLE = '''
import pytest

@pytest.mark.skipif(True, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_marked_oracle():
    pass

def test_imperative_oracle():
    pytest.skip("needs Bellhop and KRAKEN (GPUSONAR_BELLHOP, GPUSONAR_KRAKEN_DIR)")

def test_other_skip():
    pytest.skip("arlpy is not installed")

def test_passes():
    pass
'''


def _run(tmp_path, require):
    (tmp_path / "test_sample.py").write_text(SAMPLE)
    env = {k: v for k, v in os.environ.items() if k not in ("GPUSONAR_REQUIRE_ORACLES", "PYTEST_ADDOPTS")}
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), env.get("PYTHONPATH", "")])
    if require is not None:
        env["GPUSONAR_REQUIRE_ORACLES"] = require
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "-p", "oracle_gate",
                          "--rootdir", str(tmp_path), str(tmp_path / "test_sample.py")],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    return out.returncode, out.stdout + out.stderr


def test_reasons_are_recognised_by_the_variable_they_name():
    assert oracle_variable("Bellhop not installed (GPUSONAR_BELLHOP)") == "GPUSONAR_BELLHOP"
    assert oracle_variable("needs KRAKEN (GPUSONAR_KRAKEN_DIR)") == "GPUSONAR_KRAKEN_DIR"
    assert oracle_variable("arlpy is not installed") is None
    # Bellhop3D's variable contains Bellhop's: names match as whole words, the first of ORACLE_VARIABLES named wins.
    assert oracle_variable("needs Bellhop3D (GPUSONAR_BELLHOP3D)") == "GPUSONAR_BELLHOP3D"
    assert oracle_variable("needs Bellhop (GPUSONAR_BELLHOP) and Bellhop3D (GPUSONAR_BELLHOP3D)") == "GPUSONAR_BELLHOP"
    # An oracle that is a Python module is named by its module: the reasons the PyRAM cross-check gives.
    import test_ram_crosscheck as ram
    assert oracle_variable(ram.PYRAM_MISSING) == "pyram"
    assert oracle_variable(ram.pyram_wrong_version("1.2.0")) == "pyram"
    assert oracle_variable("pyramid is not installed") is None
    # The skip marks of the seamount test (Bellhop3D's reason must not be counted as Bellhop's).
    import test_seamount_3d as sm
    assert oracle_variable(sm.needs_bellhop3d.kwargs["reason"]) == "GPUSONAR_BELLHOP3D"
    assert oracle_variable(sm.needs_bellhop.kwargs["reason"]) == "GPUSONAR_BELLHOP"


@pytest.mark.parametrize("require", [None, "0"])
def test_without_the_switch_oracle_tests_skip_and_are_counted(tmp_path, require):
    code, text = _run(tmp_path, require)
    assert code == 0, text
    assert re.search(r"1 passed, 3 skipped", text), text
    assert "oracle tests skipped: 2 for want of GPUSONAR_BELLHOP" in text, text


def test_with_the_switch_oracle_skips_fail_and_other_skips_stay(tmp_path):
    code, text = _run(tmp_path, "1")
    assert code == 1, text
    assert re.search(r"1 failed, 1 passed, 1 skipped, 1 error", text), text      # the skipif mark acts at setup: an error
    assert text.count("GPUSONAR_REQUIRE_ORACLES is set") >= 2, text
    assert "oracle tests skipped" not in text


def test_every_oracle_skip_in_this_directory_names_its_variable():
    """A skip about Bellhop, KRAKEN or PyRAM that does not name GPUSONAR_BELLHOP(3D) / GPUSONAR_KRAKEN_DIR / pyram would pass
    the gate."""
    bad = []
    for path in sorted(HERE.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"(?:pytest\.skip\(|reason=)\s*(\"[^\"]*\"|'[^']*')", text):
            reason = m.group(1)
            if re.search(r"bellhop|kraken|pyram", reason, re.I) and oracle_variable(reason) is None:
                bad.append(f"{path.name}: {reason}")
    assert not bad, bad
