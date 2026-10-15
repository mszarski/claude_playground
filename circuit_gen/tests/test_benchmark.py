"""
The benchmark's own consistency: every reference design passes its board's
checks and hidden tests, so a model failure means the model, not the spec.
"""

import json
from pathlib import Path

import pytest

from circuit_gen.checks import check_circuit
from circuit_gen.runner import run_circuit_code
from circuit_gen.spice import simulate

from conftest import needs_skidl

BENCH = Path(__file__).parent.parent / "benchmarks"
CASES = {c["id"]: c for c in json.loads((BENCH / "boards.json").read_text())}
REFERENCES = sorted(p for p in (BENCH / "reference").glob("*.py") if not p.name.startswith("_"))


def test_boards_are_well_formed():
    for case in CASES.values():
        assert case["description"] and isinstance(case.get("tests", []), list)
        for t in case.get("tests", []):
            lo, hi = t["expect"]
            assert lo <= hi, case["id"]


def test_every_reference_has_a_board():
    assert {p.stem for p in REFERENCES} <= set(CASES)


@needs_skidl
@pytest.mark.parametrize("ref", REFERENCES, ids=lambda p: p.stem)
def test_reference_passes(ref, tmp_path):
    helpers = (BENCH / "reference" / "_parts.py").read_text()
    run = run_circuit_code(helpers + "\n" + ref.read_text(), tmp_path)
    assert run.ok, run.error
    assert not run.erc_errors
    report = check_circuit(run.circuit)
    assert report.ok, report.summary()
    sim = simulate(run.circuit, CASES[ref.stem]["tests"])
    assert sim.ok, sim.summary()
