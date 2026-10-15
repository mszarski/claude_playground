"""
Model backends for the circuit generator.

Each backend keeps the conversation in its provider's own message format and
exposes the same three calls to the generation loop:

    backend.start(system, user_message, tools)
    turn = backend.step()                       # one model response
    backend.add_tool_results([...]) / backend.add_user_message(text)

Tools are given in a neutral form: {"name", "description", "input_schema"}.

- HuggingFaceBackend: open-weight models through Hugging Face Inference
  Providers (needs HF_TOKEN). This is the default.
- ClaudeBackend: Claude through the Anthropic API (needs Anthropic credentials).
"""

import json
import time
from dataclasses import dataclass, field

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict | None      # None when the model's arguments were not valid JSON
    raw: str = ""


@dataclass
class Turn:
    text: str
    tool_calls: list = field(default_factory=list)
    stop: str = "end"        # "end", "tool_use", "max_tokens" or "refusal"
    record: dict = field(default_factory=dict)   # provider response, for the transcript


@dataclass
class ToolResult:
    id: str
    content: str
    is_error: bool = False


def _retry_delay(error, attempt, cap=60.0):
    """Seconds to wait before retrying: the server's Retry-After if given, else backoff."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        return min(float(headers.get("retry-after")), cap)
    except (TypeError, ValueError):
        return min(2.0 ** (attempt + 1), cap)


class HuggingFaceBackend:
    """Open models served by Hugging Face Inference Providers (OpenAI-style chat API)."""

    name = "hf"

    def __init__(self, model=None, client=None, provider=None, max_tokens=16384,
                 temperature=0.2, retries=7):
        if client is None:
            from huggingface_hub import InferenceClient
            client = InferenceClient(provider=provider or config.HF_PROVIDER)
        self.client = client
        self.model = model or config.HF_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.messages = []
        self.tools = []

    def start(self, system, user_message, tools):
        self.messages = [{"role": "system", "content": system},
                         {"role": "user", "content": user_message}]
        self.tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]

    def _complete(self):
        from huggingface_hub.errors import HfHubHTTPError
        for attempt in range(self.retries + 1):
            try:
                return self.client.chat_completion(
                    model=self.model, messages=self.messages, tools=self.tools,
                    max_tokens=self.max_tokens, temperature=self.temperature)
            except HfHubHTTPError as e:
                status = getattr(e.response, "status_code", None)
                if attempt == self.retries or (status is not None and status < 500 and status != 429):
                    raise
                time.sleep(_retry_delay(e, attempt))

    def step(self):
        response = self._complete()
        choice = response.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments
            if isinstance(raw, dict):    # some providers return parsed arguments
                args, raw = raw, json.dumps(raw)
            else:
                try:
                    args = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    args = None
            calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args, raw=raw or ""))

        assistant = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": c.raw}}
                                       for c in calls]
        stop = {"length": "max_tokens"}.get(choice.finish_reason, "tool_use" if calls else "end")
        if stop != "max_tokens":
            # A cut-off reply is not kept, so the turn can be retried from the same history.
            self.messages.append(assistant)

        record = {"finish_reason": choice.finish_reason, "message": assistant}
        usage = getattr(response, "usage", None)
        if usage is not None:
            record["usage"] = {"prompt_tokens": usage.prompt_tokens,
                               "completion_tokens": usage.completion_tokens}
        return Turn(text=msg.content or "", tool_calls=calls, stop=stop, record=record)

    def add_tool_results(self, results):
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.id, "content": content})

    def add_user_message(self, text):
        self.messages.append({"role": "user", "content": text})


class ClaudeBackend:
    """Claude through the Anthropic API, with adaptive thinking and streamed tool input."""

    name = "claude"

    def __init__(self, model=None, client=None, effort="high", max_tokens=64000, json_retries=2):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or config.CLAUDE_MODEL
        self.effort = effort
        self.max_tokens = max_tokens
        self.json_retries = json_retries
        self.system = ""
        self.messages = []
        self.tools = []

    def start(self, system, user_message, tools):
        self.system = system
        self.messages = [{"role": "user", "content": user_message}]
        self.tools = [{**t, "eager_input_streaming": True} for t in tools]

    def _request(self):
        for attempt in range(self.json_retries + 1):
            try:
                with self.client.beta.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system,
                    tools=self.tools,
                    messages=self.messages,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.effort},
                    cache_control={"type": "ephemeral"},
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                ) as stream:
                    return stream.get_final_message()
            except ValueError:
                # Raised by the SDK for streamed tool input that is not valid JSON.
                if attempt == self.json_retries:
                    raise

    def step(self):
        response = self._request()
        record = {"stop_reason": response.stop_reason,
                  "content": [b.model_dump() for b in response.content]}
        text = "\n".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason in ("refusal", "max_tokens"):
            return Turn(text=text, stop=response.stop_reason, record=record)

        self.messages.append({"role": "assistant", "content": response.content})
        calls = [ToolCall(id=b.id, name=b.name, input=b.input if isinstance(b.input, dict) else None,
                          raw=json.dumps(b.input, default=str))
                 for b in response.content if b.type == "tool_use"]
        return Turn(text=text, tool_calls=calls, stop="tool_use" if calls else "end", record=record)

    def add_tool_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.id, "content": r.content, "is_error": r.is_error}
            for r in results]})

    def add_user_message(self, text):
        self.messages.append({"role": "user", "content": text})


BACKENDS = {"hf": HuggingFaceBackend, "claude": ClaudeBackend}


def make_backend(name=None, model=None, **kwargs):
    name = name or config.BACKEND
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[name](model=model, **kwargs)
