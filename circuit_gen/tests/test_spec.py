"""
Tests for spec-first generation: parsing frozen requirements and appending them.
"""

import pytest

from circuit_gen import generate
from circuit_gen.backends import Turn
from circuit_gen.spec import SpecError, parse_requirements, to_code

from test_generate import ScriptedBackend, calls, say

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
    result = generate.generate_circuit("an LED board", tmp_path, backend=backend, spec_first=True)

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


def test_designer_can_dispute_a_frozen_requirement(tmp_path):
    from circuit_gen.backends import ToolCall
    dispute = Turn(text="", stop="tool_use", tool_calls=[ToolCall(
        id="d1", name="report_spec_problem",
        input={"requirement": "LED current", "reason": "value is in mA, not A"}, raw="{}")])
    backend = ScriptedBackend([say(SPEC_REPLY), dispute])
    result = generate.generate_circuit("an LED board", tmp_path, backend=backend, spec_first=True)
    assert not result.passed
    assert result.error == "spec disputed: LED current: value is in mA, not A"
