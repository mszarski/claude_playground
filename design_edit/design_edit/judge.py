"""LLM-judge helpers usable from training jobs (OpenAI-compatible endpoint, HF router by default)."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

ROUTER = "https://router.huggingface.co/v1"

GENERIC_RUBRIC = """- The change clearly accomplishes what the request asks for.
- It changes nothing unrelated to the request.
- All text stays readable (no low-contrast or same-color text)."""

JUDGE = """You are a senior product designer grading an edit to a UI design.

Request: {request}

Rubric (all points must hold):
{rubric}

Design before (one node per line, indented by depth):
{before}

Change applied (patch):
{patch}

Answer with one word: PASS or FAIL."""


def chat(model: str, messages: list[dict], max_tokens: int = 16, temperature: float = 0.0,
         base_url: str = ROUTER, token: str | None = None, retries: int = 4) -> str:
    token = token or os.environ["HF_TOKEN"]
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(f"{base_url}/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            if attempt == retries - 1:
                return ""
            time.sleep(2 ** (attempt + 1))
    return ""


def judge_pass(request: str, before_text: str, patch: str, model: str,
               rubric: str = GENERIC_RUBRIC) -> bool | None:
    """True/False verdict, or None when the judge could not be reached (caller decides)."""
    msg = JUDGE.format(request=request, rubric=rubric, before=before_text, patch=patch)
    out = chat(model, [{"role": "user", "content": msg}], max_tokens=8)
    m = re.search(r"\b(PASS|FAIL)\b", out.upper())
    return None if m is None else m.group(1) == "PASS"
