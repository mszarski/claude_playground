"""
Tests for spec-first generation: parsing frozen requirements and appending them.
"""

import pytest

from circuit_gen import generate
from circuit_gen.backends import Turn
from circuit_gen.spec import SpecError, parse_requirements, to_code

from test_generate import ScriptedBackend, calls, say

DESC = "An LED board with a filter. Name the nets VCC, IN, OUT and GND."

SPEC_REPLY = '''Here are the tests:
```python
require("i(VCC)", 0.008, 0.012, sources={"VCC": 5}, name="LED current")
require("gain(OUT,IN)", 0.6, 0.8, sources={"IN": "ac"}, analysis="ac", freq=1000)
```'''


class TestParse:
    def test_parses_code_block(self):
        reqs = parse_requirements(SPEC_REPLY)
        assert [r["measure"] for r in reqs] == ["i(VCC)", "gain(OUT,IN)"]
        assert reqs[0]["sources"] == {"VCC": 5} and reqs[0]["analysis"] == "op"
        assert reqs[1]["freq"] == 1000 and reqs[1]["name"] is None

    def test_round_trip(self):
        reqs = parse_requirements(SPEC_REPLY)
        assert parse_requirements(to_code(reqs)) == reqs

    @pytest.mark.parametrize("text,message", [
        ('import os\nrequire("v(A)", 0, 1)', "only require"),
        ('require("v(A)", 0, x)', "literals"),
        ('require("v(A)", 0)', "missing high"),
        ('require("v(A)", 2, 1)', "low above high"),
        ("no code here", "not valid Python"),
        ("", "no require"),
    ])
    def test_rejects(self, text, message):
        with pytest.raises(SpecError, match=message):
            parse_requirements(text)


def test_spec_first_freezes_requirements(tmp_path):
    backend = ScriptedBackend([say(SPEC_REPLY), say("done"), say("done"), say("done")])
    starts = []
    original_start = backend.start
    backend.start = lambda *a: (starts.append(a), original_start(*a))
    result = generate.generate_circuit(DESC, tmp_path, backend=backend, spec_first=True)

    assert len(starts) == 2                                  # spec call, then design call
    assert starts[0][2] == []                                # spec phase has no tools
    assert "require('i(VCC)', 0.008, 0.012" in starts[1][1]  # frozen spec shown to the designer
    assert (tmp_path / "spec.py").exists()
    assert result.code is None                               # designer never submitted a circuit


def test_bad_spec_retries_then_fails(tmp_path):
    backend = ScriptedBackend([say("import os"), say("still not"), say("nope")])
    result = generate.generate_circuit("x", tmp_path, backend=backend, spec_first=True)
    assert result.error.startswith("spec phase failed")
    assert len(backend.user_messages) == 3


def test_negative_gain_is_rejected_with_explanation():
    with pytest.raises(SpecError, match="magnitude"):
        parse_requirements('require("gain(OUT,IN)", -11, -9, analysis="ac", sources={"IN": "ac"})')


def test_unquoted_measure_is_accepted():
    reqs = parse_requirements("require(v(VCC), 4.75, 5.25, sources={'VCC': 5})\n"
                              "require(gain(OUT, IN), 9, 11, analysis='ac', sources={'IN': 'ac'})")
    assert [r["measure"] for r in reqs] == ["v(VCC)", "gain(OUT,IN)"]


@pytest.mark.parametrize("text,measure", [
    ("require(v(+5V), 4.75, 5.25, sources={'VIN': 12})", "v(+5V)"),
    ("require(freq('OUT'), 0.7, 1.3, analysis='tran', duration=6)", "freq(OUT)"),
    ("require(freq(OUT), 0.7, 1.3, analysis='tran')", "freq(OUT)"),
    ("require( gain( OUT , IN ), 9, 11, analysis='ac')", "gain(OUT,IN)"),
    ('require("v(OUT)", 1, 2)', "v(OUT)"),
])
def test_bare_measures(text, measure):
    assert parse_requirements(text)[0]["measure"] == measure


def test_tran_round_trip():
    reqs = parse_requirements("require('freq(OUT)', 0.7, 1.3, sources={'VCC': 5}, analysis='tran', duration=8)")
    again = parse_requirements(to_code(reqs))
    assert again == reqs and again[0]["duration"] == 8


def _dispute(name="LED current", reason="value is in mA, not A"):
    from circuit_gen.backends import ToolCall
    return Turn(text="", stop="tool_use", tool_calls=[ToolCall(
        id="d1", name="report_spec_problem", input={"requirement": name, "reason": reason},
        raw="{}")])


REVISED = 'require("i(VCC)", 0.009, 0.011, sources={"VCC": 5}, name="LED current")'


def test_dispute_gets_one_revision_then_ends(tmp_path):
    # spec, dispute, revised spec (the scripted backend plays both roles), dispute again
    backend = ScriptedBackend([say(SPEC_REPLY), _dispute(), say(REVISED), _dispute("again", "still wrong")])
    result = generate.generate_circuit(DESC, tmp_path, backend=backend, spec_first=True)
    assert result.spec_revisions == 1
    assert result.error == "spec disputed: again: still wrong"
    assert (tmp_path / "spec_revised.py").read_text().count("require(") == 1
    note = backend.tool_results[0][0].content
    assert "now replace the previous ones" in note and "0.009" in note


def test_dispute_without_spec_first_ends(tmp_path):
    backend = ScriptedBackend([_dispute()])
    result = generate.generate_circuit("x", tmp_path, backend=backend)
    assert result.error.startswith("spec disputed")


@pytest.mark.parametrize("text,message", [
    ('require("v(OUT)", 4, 5, loads={"OUT": 0.01})', "resistances in ohms"),
    ('require("v(OUT)", 4, 5, loads={"OUT": {"to": "VCC", "r": 0.5}})', "resistances in ohms"),
    ('require("i(VCC)", -0.02, -0.01, sources={"VCC": 5})', "magnitude"),
])
def test_spec_sanity_rejections(text, message):
    with pytest.raises(SpecError, match=message):
        parse_requirements(text)


def test_requirements_must_use_described_nets():
    from circuit_gen.spec import check_nets
    desc = "White LED driver from 9 V with a +5V rail. Name the nets VIN, +5V and GND."
    ok = parse_requirements("require('i(VIN)', 0.01, 0.03, sources={'VIN': 9, 'GND': 0})\n"
                            "require('v(+5V)', 4.8, 5.2, sources={'VIN': 9})")
    check_nets(ok, desc)
    bad = parse_requirements("require('i(VIN)', 0.019, 0.021, sources={'VIN': 9}, loads={'LED': 3.2})")
    with pytest.raises(SpecError, match="'LED' is not named"):
        check_nets(bad, desc)


def test_named_nets():
    from circuit_gen.spec import named_nets
    assert named_nets("Board. Name the nets IN_P, IN_N, OUT, VPOS (+12 V), VNEG (-12 V) and GND.") \
        >= {"IN_P", "IN_N", "OUT", "VPOS", "VNEG", "GND"}
    assert named_nets("Name the supply nets VCC and GND, and the 555 output net OUT.") >= {"VCC", "GND", "OUT"}
    assert named_nets("no clause here") is None
