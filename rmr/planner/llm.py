"""OpenAI-compatible Chat Completions client for the zero-shot planner (stdlib only).

Defaults to Hugging Face Inference Providers (``https://router.huggingface.co/v1``, authenticated with
``HF_TOKEN``). Any OpenAI-compatible endpoint works: set ``PLANNER_BASE_URL`` and ``PLANNER_API_KEY``, e.g. a local
Ollama (``http://localhost:11434/v1``, model ``qwen3.5:4b``) or llama.cpp server. ``PLANNER_BASE_URL=local`` runs the
model in-process with transformers instead (fully offline, no server; slow on CPU). ``PLANNER_MODEL`` overrides
the default model.

Reference: ``planner/gateway.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0), which calls the
Vercel AI Gateway with the ``openai`` package instead.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("PLANNER_BASE_URL", "https://router.huggingface.co/v1")
DEFAULT_MODEL = os.environ.get("PLANNER_MODEL", "moonshotai/Kimi-K3")


_LOCAL = {}


def _post_local(payload):
    """In-process chat completion with transformers, returning an OpenAI-shaped reply (no schema enforcement:
    the prompt asks for JSON and ``parse_json`` extracts it)."""
    import torch
    from transformers import AutoTokenizer

    from .finetune import lm_class

    name = payload["model"]
    if name not in _LOCAL:
        tok = AutoTokenizer.from_pretrained(name)
        _LOCAL[name] = (tok, lm_class(name).from_pretrained(name, dtype=torch.bfloat16).eval())
    tok, m = _LOCAL[name]
    text = tok.apply_chat_template(payload["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt")
    t = payload.get("temperature", 0.7)
    kw = dict(do_sample=True, temperature=t, top_p=0.95) if t else dict(do_sample=False)
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=payload.get("max_tokens", 600), pad_token_id=tok.eos_token_id, **kw)
    return {"choices": [{"message": {"content": tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)}}]}


def _key():
    key = os.environ.get("PLANNER_API_KEY") or os.environ.get("HF_TOKEN")
    if not key:
        raise SystemExit("Set HF_TOKEN (or PLANNER_API_KEY for another OpenAI-compatible endpoint).")
    return key


def _post(payload, timeout=180):
    if os.environ.get("PLANNER_BASE_URL") == "local":
        payload = {k: v for k, v in payload.items() if k != "response_format"}
        return _post_local(payload)
    req = urllib.request.Request(BASE_URL.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def parse_json(text):
    """The JSON object in a model reply, ignoring any ``<think>`` block and surrounding prose or fences."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def chat_json(messages, schema, name, model=None, temperature=0.7, retries=3, post=_post):
    """Chat completion constrained to a JSON schema (structured outputs); falls back to
    'reply with JSON' parsing for models or providers that reject ``response_format``."""
    model, structured = model or DEFAULT_MODEL, True
    for attempt in range(retries):
        try:
            kw = dict(model=model, messages=messages, temperature=temperature)
            if structured:
                kw["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": name, "schema": schema, "strict": True}}
            return parse_json(post(kw)["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and structured:
                structured = False      # no structured outputs here: retry as plain JSON
                continue
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (ValueError, KeyError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"no valid reply from {model} after {retries} attempts")
