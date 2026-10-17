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


def test_every_board_has_a_reference_and_two_tests():
    assert {p.stem for p in REFERENCES} == set(CASES)
    for case in CASES.values():
        assert len(case.get("tests", [])) >= 2, case["id"]


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


# Plausible wrong designs: each reference with one mistake. The hidden tests must
# reject every one, or they are too weak to tell a working board from a broken one.
MUTANTS = [
    ("reg_12v_5v", 'v5 & R("1k") & la', "v5 += la"),                     # LED without resistor
    ("usb_c_power", 'vbus & R("1k") & la', "vbus += la"),                 # LED without resistor
    ("mic_preamp", 'vcc & R("2.2k") & mic\n', ""),                        # no mic bias
    ("mic_preamp", 'mic & C("1u") & ninp', 'mic & C("1n") & ninp'),       # input corner too high
    ("rc_lowpass", 'R("1.6k", tolerance=0.01)', 'R("16k", tolerance=0.01)'),  # cutoff 10x low
    ("buffered_reference", "u[3] += div; u[2] += out; u[1] += out", "div += out"),  # unbuffered
    ("inverting_amp", 'inp & C("10u") & cin', 'inp & C("100n") & cin'),   # input corner 160 Hz
    ("i2c_sensor_breakout", 'vcc & R("4.7k") & sda\n', ""),               # missing pull-up
    ("led_resistor", 'R("330")', 'R("100")'),                             # LED overdriven
]


@needs_skidl
@pytest.mark.parametrize("board,old,new", MUTANTS, ids=lambda x: x if "(" not in x else "")
def test_hidden_tests_reject_mutant(board, old, new, tmp_path):
    helpers = (BENCH / "reference" / "_parts.py").read_text()
    code = (BENCH / "reference" / f"{board}.py").read_text()
    assert old in code
    run = run_circuit_code(helpers + "\n" + code.replace(old, new), tmp_path)
    assert run.ok, run.error
    assert not simulate(run.circuit, CASES[board]["tests"]).ok
