"""
Tests for the functional checks: DC-path and output-on-rail rules, and ngspice simulation.
"""

import shutil

import pytest

from circuit_gen import checks, library, spice

from conftest import needs_libs

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not installed")


def part(ref, symbol, value, pins):
    return {"ref": ref, "symbol": symbol, "value": value, "footprint": "Fake:FP",
            "pins": [{"num": num, "name": name, "type": typ, "net": net}
                     for num, name, typ, net in pins]}


def R(ref, a, b, value="10k"):
    return part(ref, "Device:R", value, [("1", "~", "PASSIVE", a), ("2", "~", "PASSIVE", b)])


def C(ref, a, b, value="100n"):
    return part(ref, "Device:C", value, [("1", "~", "PASSIVE", a), ("2", "~", "PASSIVE", b)])


def header(ref, *nets):
    return part(ref, "Connector:Conn_01x02_Pin", "", [(str(i + 1), f"Pin_{i + 1}", "PASSIVE", n)
                                                     for i, n in enumerate(nets)])


def lm358(ref, out_a, inn_a, inp_a, out_b, inn_b, inp_b, vcc="VCC", gnd="GND"):
    return part(ref, "Amplifier_Operational:LM358", "LM358", [
        ("1", "~", "OUTPUT", out_a), ("2", "-", "INPUT", inn_a), ("3", "+", "INPUT", inp_a),
        ("5", "+", "INPUT", inp_b), ("6", "-", "INPUT", inn_b), ("7", "~", "OUTPUT", out_b),
        ("4", "V-", "PWRIN", gnd), ("8", "V+", "PWRIN", vcc)])


def preamp(bias_input=True, unused_as_follower=True):
    """Non-inverting x101 amplifier around an LM358, AC-coupled in and out."""
    parts = [
        header("J1", "VCC", "GND"), header("J2", "MIC", "GND"), header("J3", "AUDIO_OUT", "GND"),
        R("R1", "VCC", "VREF"), R("R2", "VREF", "GND"), C("C1", "VREF", "GND", "10u"),
        C("C2", "MIC", "INP", "10u"),
        R("R4", "OUT", "FB", "100k"), R("R5", "FB", "FBC", "1k"), C("C3", "FBC", "VREF", "100u"),
        C("C4", "OUT", "AUDIO_OUT", "10u"), C("C5", "VCC", "GND"),
    ]
    if bias_input:
        parts.append(R("R3", "INP", "VREF", "100k"))
    if unused_as_follower:
        parts.append(lm358("U1", "OUT", "FB", "INP", "OUT_B", "OUT_B", "VREF"))
    else:
        parts.append(lm358("U1", "OUT", "FB", "INP", "GND", "VREF", "VREF"))
    return {"parts": parts}


@pytest.fixture
def fake_footprints(monkeypatch):
    monkeypatch.setattr(library, "footprint_exists", lambda fp: True)
    monkeypatch.setattr(library, "footprint_pads", lambda fp: {str(i) for i in range(1, 9)})


class TestStructuralRules:
    def test_good_preamp_passes(self, fake_footprints):
        report = checks.check_circuit(preamp())
        assert report.ok, report.summary()

    def test_capacitor_only_input_floats(self, fake_footprints):
        report = checks.check_circuit(preamp(bias_input=False))
        assert any("net 'INP' has no DC path" in e for e in report.errors)

    def test_output_tied_to_ground(self, fake_footprints):
        report = checks.check_circuit(preamp(unused_as_follower=False))
        assert any("U1 pin 7" in e and "tied directly to rail 'GND'" in e for e in report.errors)

    def test_555_control_voltage_is_exempt(self, fake_footprints):
        circuit = {"parts": [
            header("J1", "VCC", "GND"),
            part("U1", "Timer:NE555P", "NE555P", [
                ("1", "GND", "PWRIN", "GND"), ("8", "VCC", "PWRIN", "VCC"),
                ("5", "CV", "INPUT", "CV"), ("4", "R", "INPUT", "VCC")]),
            C("C1", "CV", "GND", "10n"), C("C2", "VCC", "GND"),
        ]}
        assert checks.check_circuit(circuit).ok


class TestValues:
    @pytest.mark.parametrize("text,value", [
        ("10k", 1e4), ("4k7", 4700), ("100n", 1e-7), ("10uF", 1e-5), ("1R5", 1.5),
        ("330", 330), ("2.2k", 2200), ("1M", 1e6), ("0.1u", 1e-7), ("120 Ω", 120)])
    def test_parse_value(self, text, value):
        assert spice.parse_value(text) == pytest.approx(value)

    def test_unparseable(self):
        with pytest.raises(ValueError):
            spice.parse_value("red")


@needs_ngspice
@needs_libs
class TestSimulation:
    def test_preamp_gain(self):
        test = {"name": "gain", "analysis": "ac", "freq": 1000, "sources": {"VCC": 5, "MIC": "ac"},
                "loads": {"AUDIO_OUT": 10000}, "measure": "gain(AUDIO_OUT,MIC)", "expect": [80, 120]}
        report = spice.simulate(preamp(), [test])
        assert report.ok, report.summary()
        assert report.unmodeled == []

    def test_broken_preamp_fails_gain(self):
        test = {"name": "gain", "analysis": "ac", "freq": 1000, "sources": {"VCC": 5, "MIC": "ac"},
                "loads": {"AUDIO_OUT": 10000}, "measure": "gain(AUDIO_OUT,MIC)", "expect": [80, 120]}
        # Gain resistor to ground instead of to the reference: output saturates.
        circuit = preamp()
        for p in circuit["parts"]:
            if p["ref"] == "C3":
                p["pins"][1]["net"] = "GND"
                p["ref"] = "R6"
                p["symbol"], p["value"] = "Device:R", "0"
        report = spice.simulate(circuit, [test])
        assert not report.ok

    def test_led_current_and_supply_current_sign(self):
        circuit = {"parts": [
            header("J1", "VCC", "GND"), R("R1", "VCC", "LED_A", "300"),
            part("D1", "Device:LED", "red", [("1", "K", "PASSIVE", "GND"), ("2", "A", "PASSIVE", "LED_A")]),
        ]}
        test = {"sources": {"VCC": 5}, "measure": "i(VCC)", "expect": [0.009, 0.0115]}
        report = spice.simulate(circuit, [test])
        assert report.ok, report.summary()

    def test_regulator_and_load_between_nets(self):
        circuit = {"parts": [
            part("U1", "Regulator_Linear:L7805", "L7805", [
                ("1", "IN", "PWRIN", "VIN"), ("2", "GND", "PWRIN", "GND"), ("3", "OUT", "PWROUT", "+5V")]),
            C("C1", "VIN", "GND", "330n"), C("C2", "+5V", "GND", "100n"),
        ]}
        ok = {"sources": {"VIN": 12}, "loads": {"+5V": 50}, "measure": "v(+5V)", "expect": [4.9, 5.1]}
        dropout = {"sources": {"VIN": 6}, "loads": {"+5V": 50}, "measure": "v(+5V)", "expect": [3.5, 4.5]}
        # A load between two named nets (here input to ground): 12 V / 1k plus the
        # regulator's quiescent current.
        between = {"sources": {"VIN": 12}, "loads": {"VIN": {"to": "GND", "r": 1000}},
                   "measure": "i(VIN)", "expect": [0.0118, 0.0125]}
        report = spice.simulate(circuit, [ok, dropout, between])
        assert report.ok, report.summary()

    def test_unknown_net_and_unmodeled_part(self):
        circuit = {"parts": [
            header("J1", "VCC", "GND"), R("R1", "VCC", "GND"),
            part("U1", "MCU_Microchip_ATtiny:ATtiny85-20PU", "ATtiny85", [
                ("8", "VCC", "PWRIN", "VCC"), ("4", "GND", "PWRIN", "GND")]),
        ]}
        report = spice.simulate(circuit, [{"sources": {"VDD": 5}, "measure": "v(VDD)", "expect": [0, 1]}])
        assert not report.ok
        assert "VDD" in report.results[0].error
        assert report.unmodeled == ["U1 (MCU_Microchip_ATtiny:ATtiny85-20PU)"]


@needs_ngspice
@needs_libs
class TestToleranceCorners:
    def _divider(self, tol=None):
        r1 = R("R1", "VCC", "OUT", "10k")
        r2 = R("R2", "OUT", "GND", "10k")
        if tol is not None:
            r1["tolerance"] = r2["tolerance"] = tol
        return {"parts": [header("J1", "VCC", "GND"), r1, r2]}

    def test_spread_covers_tolerance(self):
        test = {"sources": {"VCC": 10}, "measure": "v(OUT)", "expect": [4.5, 5.5]}
        result = spice.simulate(self._divider(), [test], corners=True).results[0]
        assert result.passed
        assert result.value == pytest.approx(5.0)
        # 5% resistors in opposite corners: 10*10.5/(9.5+10.5) = 5.25
        assert result.spread[0] == pytest.approx(4.75, abs=0.01)
        assert result.spread[1] == pytest.approx(5.25, abs=0.01)

    def test_corner_failure_and_part_tolerance(self):
        tight = {"sources": {"VCC": 10}, "measure": "v(OUT)", "expect": [4.9, 5.1]}
        assert not spice.simulate(self._divider(), [tight], corners=True).ok
        assert spice.simulate(self._divider(), [tight]).ok          # nominal only
        assert spice.simulate(self._divider(tol=0.01), [tight], corners=True).ok

    def test_regulator_input_current_includes_load(self):
        circuit = {"parts": [
            part("U1", "Regulator_Linear:L7805", "L7805", [
                ("1", "IN", "PWRIN", "VIN"), ("2", "GND", "PWRIN", "GND"), ("3", "OUT", "PWROUT", "+5V")]),
        ]}
        test = {"sources": {"VIN": 12}, "loads": {"+5V": 50}, "measure": "i(VIN)", "expect": [0.099, 0.102]}
        assert spice.simulate(circuit, [test]).ok


@needs_ngspice
@needs_libs
class TestRequirementsInChecks:
    def _circuit(self, low, high):
        c = {"parts": [header("J1", "VCC", "GND"), R("R1", "VCC", "OUT", "10k"), R("R2", "OUT", "GND", "10k")]}
        c["requirements"] = [{"measure": "v(OUT)", "min": low, "max": high, "sources": {"VCC": 10},
                              "loads": {}, "analysis": "op", "freq": 1000, "name": "half"}]
        return c

    def test_met(self, fake_footprints):
        report = checks.check_circuit(self._circuit(4.5, 5.5))
        assert report.ok and "REQUIREMENT OK: half" in report.summary()

    def test_not_met_across_tolerances(self, fake_footprints):
        report = checks.check_circuit(self._circuit(4.9, 5.1))
        assert any("requirement not met: half" in e for e in report.errors)

    def test_missing_requirements_warn(self, fake_footprints):
        c = self._circuit(0, 1)
        c["requirements"] = []
        report = checks.check_circuit(c)
        assert report.ok and any("no require(...)" in w for w in report.warnings)
