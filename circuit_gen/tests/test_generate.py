"""
Tests for the model-driven generator and its backends, using scripted stand-ins
for the model APIs.
"""

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from circuit_gen import generate
from circuit_gen.backends import (ClaudeBackend, HuggingFaceBackend, ToolCall, ToolResult, Turn,
                                  make_backend)

from conftest import needs_skidl

EXAMPLES = Path(__file__).parent.parent / "examples"


class ScriptedBackend:
    """Plays back fixed turns and records what the loop sends back."""

    name = "scripted"
    model = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.tool_results = []
        self.user_messages = []
        self.started = None

    def start(self, system, user_message, tools):
        self.started = (system, user_message, tools)

    def step(self):
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.tool_results.append(results)

    def add_user_message(self, text):
        self.user_messages.append(text)


def calls(*specs):
    out = [ToolCall(id=f"t{i}", name=name, input=args, raw=json.dumps(args))
           for i, (name, args) in enumerate(specs)]
    return Turn(text="", tool_calls=out, stop="tool_use")


def say(text):
    return Turn(text=text)


class TestLoop:
    def test_tool_dispatch_without_skidl(self, tmp_path, monkeypatch):
        monkeypatch.setattr(generate.library, "search_footprints", lambda q: ["Lib:FP_1"])
        tools = generate.Toolbox(tmp_path)
        assert tools.execute("search_footprints", {"query": "x"}) == ("Lib:FP_1", False)
        assert tools.execute("nope", {})[1]

    def test_invalid_tool_input_is_returned_as_error(self, tmp_path):
        bad = Turn(text="", stop="tool_use",
                   tool_calls=[ToolCall(id="t1", name="check_circuit", input=None, raw='{"code": ')])
        backend = ScriptedBackend([bad, say("giving up"), say("still giving up"), say("done")])
        result = generate.generate_circuit("anything", tmp_path, backend=backend)
        first = backend.tool_results[0][0]
        assert first.is_error and "INVALID_JSON" in first.content
        assert result.code is None and not result.passed

    def test_refusal_stops_without_running_tools(self, tmp_path):
        backend = ScriptedBackend([Turn(text="", stop="refusal")])
        result = generate.generate_circuit("anything", tmp_path, backend=backend)
        assert result.error == "model declined the request"
        assert result.attempts == []

    def test_retries_turn_after_max_tokens(self, tmp_path):
        cut = Turn(text="", stop="max_tokens")
        backend = ScriptedBackend([cut, cut, say("done"), say("done"), say("done")])
        result = generate.generate_circuit("anything", tmp_path, backend=backend)
        assert result.error is None and result.summary == "done"

        backend = ScriptedBackend([cut, cut, cut])
        result = generate.generate_circuit("anything", tmp_path, backend=backend)
        assert result.error == "response hit max_tokens"

    def test_nudges_when_model_stops_without_passing(self, tmp_path):
        backend = ScriptedBackend([say("here is a design"), say("ok"), say("bye")])
        result = generate.generate_circuit("anything", tmp_path, backend=backend, max_nudges=2)
        assert len(backend.user_messages) == 2
        assert result.summary == "bye" and not result.passed

    def test_system_prompt_has_examples(self, tmp_path):
        backend = ScriptedBackend([say("done"), say("done"), say("done")])
        generate.generate_circuit("a board", tmp_path, backend=backend)
        system, user, tools = backend.started
        assert "ldo_3v3.py" in system and "a board" in user
        assert {t["name"] for t in tools} == {"search_symbols", "get_symbol", "search_footprints",
                                              "check_circuit", "simulate"}

    @needs_skidl
    def test_repair_loop_keeps_last_passing_code(self, tmp_path):
        good = (EXAMPLES / "ldo_3v3.py").read_text()
        # First attempt leaves the input capacitor's ground side open, so the regulator
        # input is not decoupled.
        bad = good.replace("c_in[1, 2] += vin, gnd", "c_in[1] += vin")
        assert bad != good
        backend = ScriptedBackend([
            calls(("search_symbols", {"query": "AMS1117"})),
            calls(("check_circuit", {"code": bad})),
            calls(("check_circuit", {"code": good})),
            say("Done: AMS1117 regulator board."),
        ])
        result = generate.generate_circuit("5V to 3.3V regulator", tmp_path, backend=backend)

        assert result.passed
        assert result.code == good
        assert [a.passed for a in result.attempts] == [False, True]
        assert "FAILED" in result.attempts[0].feedback
        assert result.summary.startswith("Done")
        assert "AMS1117-3.3" in backend.tool_results[0][0].content
        assert (tmp_path / "transcript.jsonl").exists()

    @needs_skidl
    def test_simulate_tool(self, tmp_path):
        tools = generate.Toolbox(tmp_path)
        code = (EXAMPLES / "ldo_3v3.py").read_text()
        content, is_error = tools.execute("simulate", {
            "code": code, "sources": {"VIN": 5}, "loads": {"+3V3": 33}, "measure": "v(+3V3)"})
        assert not is_error, content
        assert content.startswith("v(+3V3) = 3.2")
        content, is_error = tools.execute("simulate", {"code": code, "measure": "v(NOPE)"})
        assert is_error and "NOPE" in content


# ---- Hugging Face backend ----

def _hf_response(content="", tool_calls=(), finish_reason="stop"):
    tcs = [SimpleNamespace(id=i, function=SimpleNamespace(name=n, arguments=a))
           for i, n, a in tool_calls]
    msg = SimpleNamespace(content=content, tool_calls=tcs or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def _as_stream(response):
    """Split a whole response into streamed chunks, as the router sends them."""
    choice, msg = response.choices[0], response.choices[0].message
    chunk = lambda delta, finish=None: SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)
    text = msg.content or ""
    for part in (text[:len(text) // 2], text[len(text) // 2:]):
        yield chunk(SimpleNamespace(content=part, tool_calls=None))
    for i, tc in enumerate(msg.tool_calls or []):
        args = tc.function.arguments
        yield chunk(SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
            index=i, id=tc.id, function=SimpleNamespace(name=tc.function.name, arguments=None))]))
        for piece in (args[:3], args[3:]):
            yield chunk(SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                index=i, id=None, function=SimpleNamespace(name=None, arguments=piece))]))
    yield chunk(SimpleNamespace(content=None, tool_calls=None), choice.finish_reason)
    yield SimpleNamespace(choices=[], usage=response.usage)


class ReadError(Exception):
    """Stands in for httpx.ReadError (a dropped connection)."""


class FakeInferenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def chat_completion(self, **kwargs):
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert kwargs.get("stream") is True
        return _as_stream(response)


class TestHuggingFaceBackend:
    def test_tool_round_trip(self):
        client = FakeInferenceClient([
            _hf_response(tool_calls=[("c1", "search_symbols", '{"query": "555"}'),
                                     ("c2", "check_circuit", '{"code": "x = ')],
                         finish_reason="tool_calls"),
            _hf_response(content="done"),
        ])
        b = HuggingFaceBackend(model="Qwen/Qwen3.8-27B", client=client)
        b.start("sys", "design it", generate.TOOLS)
        turn = b.step()
        assert turn.stop == "tool_use"
        assert turn.tool_calls[0].input == {"query": "555"}
        assert turn.tool_calls[1].input is None          # unparseable arguments

        b.add_tool_results([ToolResult("c1", "NE555P"), ToolResult("c2", "bad json", True)])
        assert b.step().stop == "end"

        req = client.requests[1]
        assert req["model"] == "Qwen/Qwen3.8-27B"
        assert req["tools"][0]["type"] == "function"
        assert req["tools"][0]["function"]["parameters"]["required"] == ["query"]
        roles = [m["role"] for m in req["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "tool"]
        assert req["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"query": "555"}'
        assert req["messages"][4] == {"role": "tool", "tool_call_id": "c2", "content": "ERROR: bad json"}

    def test_length_finish_is_max_tokens_and_not_kept(self):
        client = FakeInferenceClient([_hf_response(content="...", finish_reason="length")])
        b = HuggingFaceBackend(model="m", client=client)
        b.start("sys", "u", generate.TOOLS)
        assert b.step().stop == "max_tokens"
        assert [m["role"] for m in b.messages] == ["system", "user"]

    def test_retries_dropped_connection(self, monkeypatch):
        monkeypatch.setattr("circuit_gen.backends.time.sleep", lambda s: None)
        client = FakeInferenceClient([ReadError("Connection reset by peer"),
                                      _hf_response(content="done")])
        b = HuggingFaceBackend(model="m", client=client)
        b.start("sys", "u", generate.TOOLS)
        turn = b.step()
        assert turn.text == "done" and len(client.requests) == 2

    def test_other_errors_are_not_retried(self):
        client = FakeInferenceClient([ValueError("bad request shape")])
        b = HuggingFaceBackend(model="m", client=client)
        b.start("sys", "u", generate.TOOLS)
        with pytest.raises(ValueError):
            b.step()

    def test_default_backend_is_hf(self, monkeypatch):
        monkeypatch.setattr("huggingface_hub.InferenceClient", lambda **kw: object())
        assert make_backend().name == "hf"


# ---- Claude backend ----

class _Block(SimpleNamespace):
    def model_dump(self):
        return dict(self.__dict__)


class FakeAnthropicClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        response = self.responses.pop(0)

        class _Stream:
            def __enter__(self):
                return SimpleNamespace(get_final_message=lambda: response)

            def __exit__(self, *exc):
                return False
        return _Stream()


class TestClaudeBackend:
    def test_request_shape_and_tool_results(self):
        client = FakeAnthropicClient([
            SimpleNamespace(stop_reason="tool_use", content=[
                _Block(type="text", text="Searching."),
                _Block(type="tool_use", id="tu1", name="search_symbols", input={"query": "555"})]),
            SimpleNamespace(stop_reason="end_turn", content=[_Block(type="text", text="done")]),
        ])
        b = ClaudeBackend(model="claude-opus-5-5", client=client)
        b.start("sys", "design it", generate.TOOLS)
        turn = b.step()
        assert turn.tool_calls[0].input == {"query": "555"}
        b.add_tool_results([ToolResult("tu1", "NE555P")])
        assert b.step().text == "done"

        req = client.requests[1]
        assert req["model"] == "claude-opus-5-5"
        assert req["thinking"] == {"type": "adaptive"}
        assert req["fallbacks"] == "default"
        assert all(t["eager_input_streaming"] for t in req["tools"])
        assert req["messages"][-1]["content"][0] == {
            "type": "tool_result", "tool_use_id": "tu1", "content": "NE555P", "is_error": False}

    def test_refusal(self):
        client = FakeAnthropicClient([SimpleNamespace(stop_reason="refusal", content=[])])
        b = ClaudeBackend(model="claude-opus-5-5", client=client)
        b.start("sys", "u", generate.TOOLS)
        assert b.step().stop == "refusal"
