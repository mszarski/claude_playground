"""
Tests for the model-driven generator, using a scripted stand-in for the API client.
"""

from pathlib import Path
from types import SimpleNamespace

from circuit_gen import generate

from conftest import needs_skidl

EXAMPLES = Path(__file__).parent.parent / "examples"


class _Block(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


def text(t):
    return _Block(type="text", text=t)


def tool_use(id_, name, **args):
    return _Block(type="tool_use", id=id_, name=name, input=args)


def message(*content, stop_reason=None):
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
    return SimpleNamespace(content=list(content), stop_reason=stop_reason)


class FakeClient:
    """Plays back a fixed list of responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        # Copy the message list: the generator keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        response = self.responses.pop(0)

        class _Stream:
            def __enter__(self):
                return SimpleNamespace(get_final_message=lambda: response)

            def __exit__(self, *exc):
                return False
        return _Stream()


def test_tool_dispatch_without_skidl(tmp_path, monkeypatch):
    monkeypatch.setattr(generate.library, "search_footprints", lambda q: ["Lib:FP_1"])
    tools = generate.Toolbox(tmp_path)
    content, is_error = tools.execute("search_footprints", {"query": "x"})
    assert content == "Lib:FP_1" and not is_error
    content, is_error = tools.execute("nope", {})
    assert is_error


def test_invalid_tool_input_is_returned_as_error(tmp_path):
    client = FakeClient([
        message(tool_use("t1", "check_circuit")),      # missing the required `code`
        message(text("giving up")),
    ])
    result = generate.generate_circuit("anything", tmp_path, client=client)
    second_request = client.requests[1]["messages"]
    tool_result = second_request[-1]["content"][0]
    assert tool_result["is_error"] and "INVALID_JSON" in tool_result["content"]
    assert result.code is None and not result.passed


def test_refusal_stops_without_running_tools(tmp_path):
    client = FakeClient([message(tool_use("t1", "check_circuit", code="x"), stop_reason="refusal")])
    result = generate.generate_circuit("anything", tmp_path, client=client)
    assert result.error == "model declined the request"
    assert result.attempts == []


def test_request_shape(tmp_path):
    client = FakeClient([message(text("done"))])
    generate.generate_circuit("a board", tmp_path, client=client, model="claude-opus-5-5")
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["fallbacks"] == "default"
    assert all(t["eager_input_streaming"] for t in req["tools"])
    assert "ldo_3v3.py" in req["system"]          # few-shot examples are included


@needs_skidl
def test_repair_loop_keeps_last_passing_code(tmp_path):
    good = (EXAMPLES / "ldo_3v3.py").read_text()
    # First attempt leaves the input capacitor's ground side open, so the regulator
    # input is not decoupled.
    bad = good.replace("c_in[1, 2] += vin, gnd", "c_in[1] += vin")
    assert bad != good
    client = FakeClient([
        message(text("Looking up the regulator."),
                tool_use("t1", "search_symbols", query="AMS1117")),
        message(tool_use("t2", "check_circuit", code=bad)),
        message(tool_use("t3", "check_circuit", code=good)),
        message(text("Done: AMS1117 regulator board.")),
    ])
    result = generate.generate_circuit("5V to 3.3V regulator", tmp_path, client=client)

    assert result.passed
    assert result.code == good
    assert [a.passed for a in result.attempts] == [False, True]
    assert "FAILED" in result.attempts[0].feedback
    assert result.summary.startswith("Done")
    # The search result for AMS1117 went back to the model.
    first_results = client.requests[1]["messages"][-1]["content"][0]
    assert "AMS1117-3.3" in first_results["content"]
    assert (tmp_path / "transcript.jsonl").exists()
