"""
Tests for the KiCad schematic export (circuit-synth plus our relabelling).

The fixture is circuit-synth 0.12.1's raw output for the inverting-amp
reference: an LM358 with three units, which it labels wrongly.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from circuit_gen import schematic
from circuit_gen.schematic import (compare, expected_nets, kicad_nets, relabel, schematic_nets,
                                   to_circuit_synth)

FIXTURES = Path(__file__).parent / "fixtures"
RAW = FIXTURES / "inverting_amp_circuit_synth.kicad_sch"
CIRCUIT = json.loads((FIXTURES / "inverting_amp_circuit.json").read_text())


@pytest.fixture
def raw_copy(tmp_path):
    path = tmp_path / "inverting_amp.kicad_sch"
    shutil.copy(RAW, path)
    return path


def test_generated_code_connects_every_pin():
    code = to_circuit_synth(CIRCUIT, "inverting amp")
    compile(code, "gen.py", "exec")
    connected = sum(1 for part in CIRCUIT["parts"] for pin in part["pins"] if pin.get("net"))
    assert code.count("] += nets[") == connected
    assert "def inverting_amp():" in code


def test_raw_circuit_synth_output_shorts_multi_unit_pins():
    # Labels for unit B's pins sit on unit A's pins, so nets merge or go missing.
    problems = compare(expected_nets(CIRCUIT), schematic_nets(RAW))
    assert any(p.startswith("U1 ") for p in problems)


def test_relabel_fixes_connections_and_layout(raw_copy):
    relabel(raw_copy, CIRCUIT)
    assert compare(expected_nets(CIRCUIT), schematic_nets(raw_copy)) == []
    text = raw_copy.read_text()
    assert "(hierarchical_label" not in text
    assert '(reference "#PWR' not in text
    # Every unit's instance record names its own unit.
    tree = schematic._parse_sheet(text)
    units = [(child_unit(s), inst_unit(s)) for s in schematic.children(tree, "symbol")
             if schematic.child(s, "lib_id")[1].endswith("LM358")]
    assert sorted(units) == [(1, 1), (2, 2), (3, 3)]
    # No two placed units overlap any more: their pins are at distinct points.
    points = [pt for ref, _, pt, _ in schematic._pins(tree)]
    assert len(points) == len(set(points))


def child_unit(sym):
    return int(schematic.child(sym, "unit")[1])


def inst_unit(sym):
    inst = schematic.child(sym, "instances")
    path = schematic.child(schematic.child(inst, "project"), "path")
    return int(schematic.child(path, "unit")[1])


@pytest.mark.skipif(not shutil.which("kicad-cli"), reason="kicad-cli not installed")
def test_kicad_agrees_after_relabel(raw_copy):
    relabel(raw_copy, CIRCUIT)
    nets = kicad_nets(raw_copy)
    assert nets is not None, "kicad-cli could not load the schematic"
    assert compare(expected_nets(CIRCUIT), nets) == []


@pytest.mark.skipif(not os.environ.get("CIRCUIT_GEN_CS_PYTHON"),
                    reason="circuit-synth not configured (CIRCUIT_GEN_CS_PYTHON)")
def test_export_end_to_end(tmp_path):
    path, problems = schematic.export_schematic(CIRCUIT, tmp_path, name="inverting_amp")
    assert path.exists() and problems == []
