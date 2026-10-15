"""
Tests for the circuit stage: S-expressions, library search, design checks and the SKiDL runner.
"""

from pathlib import Path

import pytest

from circuit_gen import checks, library, sexp
from circuit_gen.runner import run_circuit_code

from conftest import needs_libs, needs_skidl

EXAMPLES = Path(__file__).parent.parent / "examples"


class TestSexp:
    def test_nested_lists_and_strings(self):
        expr = sexp.parse('(a "b c" (d 1 2) (e "x\\"y"))')[0]
        assert expr == ["a", "b c", ["d", "1", "2"], ["e", 'x"y']]

    def test_child_lookup(self):
        expr = sexp.parse("(net GND (wire 1) (wire 2) (via v 0 0))")[0]
        assert len(sexp.children(expr, "wire")) == 2
        assert sexp.child(expr, "via") == ["via", "v", "0", "0"]
        assert sexp.child(expr, "missing") is None

    def test_unbalanced_raises(self):
        with pytest.raises(ValueError):
            sexp.parse("(a (b)")


def _part(ref, pins, footprint="Fake:FP"):
    return {"ref": ref, "symbol": f"Lib:{ref}", "value": "", "footprint": footprint,
            "pins": [{"num": str(i + 1), "name": n, "type": t, "net": net}
                     for i, (n, t, net) in enumerate(pins)]}


@pytest.fixture
def fake_footprints(monkeypatch):
    monkeypatch.setattr(library, "footprint_exists", lambda fp: fp.startswith("Fake:"))
    monkeypatch.setattr(library, "footprint_pads", lambda fp: {"1", "2", "3", "4"})


class TestChecks:
    def _ic(self, vcc_net="VCC"):
        return _part("U1", [("VCC", "PWRIN", vcc_net), ("GND", "PWRIN", "GND"),
                            ("OUT", "OUTPUT", "OUT")])

    def test_decoupled_ic_passes(self, fake_footprints):
        circuit = {"parts": [
            self._ic(),
            _part("C1", [("~", "PASSIVE", "VCC"), ("~", "PASSIVE", "GND")]),
            _part("J1", [("1", "PASSIVE", "VCC"), ("2", "PASSIVE", "GND"), ("3", "PASSIVE", "OUT")]),
        ]}
        report = checks.check_circuit(circuit)
        assert report.ok, report.summary()

    def test_missing_decoupling_cap(self, fake_footprints):
        circuit = {"parts": [
            self._ic(),
            _part("J1", [("1", "PASSIVE", "VCC"), ("2", "PASSIVE", "GND"), ("3", "PASSIVE", "OUT")]),
        ]}
        report = checks.check_circuit(circuit)
        assert any("decoupling" in e for e in report.errors)

    def test_dangling_net_and_missing_footprint(self, fake_footprints):
        circuit = {"parts": [
            _part("R1", [("~", "PASSIVE", "A"), ("~", "PASSIVE", "GND")], footprint=""),
            _part("R2", [("~", "PASSIVE", "GND"), ("~", "PASSIVE", None)]),
        ]}
        report = checks.check_circuit(circuit)
        assert any("no footprint" in e for e in report.errors)
        assert any("net 'A' only connects" in e for e in report.errors)
        assert any("unconnected" in w for w in report.warnings)

    def test_footprint_pad_mismatch(self, monkeypatch):
        monkeypatch.setattr(library, "footprint_exists", lambda fp: True)
        monkeypatch.setattr(library, "footprint_pads", lambda fp: {"1", "2"})
        circuit = {"parts": [
            _part("U1", [("A", "PASSIVE", "N1"), ("B", "PASSIVE", "N1"), ("C", "PASSIVE", "GND")]),
            _part("R1", [("~", "PASSIVE", "GND"), ("~", "PASSIVE", "N1")]),
        ]}
        report = checks.check_circuit(circuit)
        assert any("no pad(s) 3" in e for e in report.errors)

    def test_ground_names(self):
        for name in ["GND", "AGND", "VSS", "GND_PWR"]:
            assert checks.is_ground(name)
        for name in ["VCC", "+3V3", "SIGNAL_GND_SENSE"]:
            assert not checks.is_ground(name)


@needs_libs
class TestLibrary:
    def test_search_symbols_finds_555(self):
        names = [e.name for e in library.search_symbols("555 timer")]
        assert "NE555P" in names

    def test_get_symbol_follows_extends(self):
        # AMS1117-3.3 is defined by extending another symbol; pins come from the parent.
        pins = {p["name"]: p["num"] for p in library.get_symbol("Regulator_Linear", "AMS1117-3.3")["pins"]}
        assert pins == {"GND": "1", "VO": "2", "VI": "3"}

    def test_footprint_search_and_pads(self):
        assert "Package_TO_SOT_SMD:SOT-223-3_TabPin2" in library.search_footprints("SOT-223-3")
        assert "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm" in library.search_footprints("SOIC-8_3.9x4.9*")
        # This footprint file writes pad numbers unquoted.
        assert library.footprint_pads("Package_SO:SOIC-8_3.9x4.9mm_P1.27mm") == {str(i) for i in range(1, 9)}

    def test_unknown_symbol(self):
        with pytest.raises(KeyError):
            library.get_symbol("Device", "NoSuchPart")


@needs_skidl
class TestRunner:
    @pytest.mark.parametrize("example", ["ldo_3v3.py", "blinker_555.py"])
    def test_examples_pass_erc_and_checks(self, example, tmp_path):
        result = run_circuit_code((EXAMPLES / example).read_text(), tmp_path)
        assert result.ok, result.error
        assert not result.erc_errors
        report = checks.check_circuit(result.circuit)
        assert report.ok, report.summary()

    def test_python_error_is_reported(self, tmp_path):
        result = run_circuit_code('Part("Device", "NoSuchPart")', tmp_path)
        assert not result.ok
        assert "NoSuchPart" in result.error

    def test_circuit_dump_has_nets(self, tmp_path):
        result = run_circuit_code((EXAMPLES / "ldo_3v3.py").read_text(), tmp_path)
        nets = {pin["net"] for part in result.circuit["parts"] for pin in part["pins"]}
        assert {"VIN", "+3V3", "GND"} <= nets


@needs_skidl
class TestRequire:
    CODE = '''
vcc, gnd = Net("VCC"), Net("GND")
j = Part("Connector", "Conn_01x02_Pin", footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical")
r = Part("Device", "R", value="1k", footprint="Resistor_SMD:R_0805_2012Metric", tolerance=0.01)
j[1, 2] += vcc, gnd
r[1, 2] += vcc, gnd
'''

    def test_requirements_and_tolerance_are_dumped(self, tmp_path):
        code = self.CODE + 'require("i(VCC)", 0.004, 0.006, sources={"VCC": 5}, name="load")\n'
        result = run_circuit_code(code, tmp_path)
        assert result.ok, result.error
        assert result.circuit["requirements"][0]["name"] == "load"
        assert result.circuit["requirements"][0]["max"] == 0.006
        assert next(p for p in result.circuit["parts"] if p["ref"] == "R1")["tolerance"] == 0.01

    def test_bad_requirement_is_a_code_error(self, tmp_path):
        result = run_circuit_code(self.CODE + 'require("i(VCC)", 1, 0)\n', tmp_path)
        assert not result.ok and "above high" in result.error
