"""
Tests for programmatic training data.
"""

import pytest

from circuit_gen import procgen

from conftest import needs_skidl


def test_e_series_and_formatting():
    assert procgen.nearest(4800) == pytest.approx(4700)
    assert procgen.nearest(0.000102, procgen.E6) == pytest.approx(0.0001)
    assert procgen.fmt(4700) == "4.7k" and procgen.fmt(1e-7) == "100n" and procgen.fmt(330) == "330"


def test_generate_is_deterministic_and_unique():
    a = procgen.generate(36, seed=3)
    b = procgen.generate(36, seed=3)
    assert [x["description"] for x in a] == [x["description"] for x in b]
    assert len({x["description"] for x in a}) == 36
    assert {x["family"] for x in a} == set(procgen.FAMILIES)


@needs_skidl
@pytest.mark.parametrize("family", sorted(procgen.FAMILIES))
def test_every_family_verifies(family, tmp_path):
    board = procgen.generate(1, seed=7, families=[family])[0]
    passed, feedback, circuit = procgen.verify(board, tmp_path)
    assert passed, feedback
    session = procgen.tool_session(board, circuit, feedback)
    roles = [m["role"] for m in session["messages"]]
    assert roles[:2] == ["system", "user"] and roles[-1] == "assistant"
    assert any(m.get("tool_calls", [{}])[0].get("function", {}).get("name") == "check_circuit"
               for m in session["messages"] if m["role"] == "assistant" and m.get("tool_calls"))


@needs_skidl
def test_repair_session_shows_real_failure_then_fix(tmp_path):
    import random
    board = procgen.generate(1, seed=7, families=["voltage_divider"])[0]
    passed, feedback, circuit = procgen.verify(board, tmp_path / "ok")
    assert passed
    repair = procgen.plant_bug(board, str(tmp_path / "bug"), random.Random(0))
    assert repair and repair["code"] != board["code"]
    assert "requirement not met" in repair["feedback"]
    session = procgen.tool_session(board, circuit, feedback, repair)
    checks = [m for m in session["messages"] if m.get("tool_calls")
              and m["tool_calls"][0]["function"]["name"] == "check_circuit"]
    assert len(checks) == 2 and checks[1]["content"] == repair["diagnosis"]
    results = [m["content"] for m in session["messages"] if m["role"] == "tool"]
    assert results[-2].startswith("FAILED") and results[-1].startswith("PASSED")
