"""Visual judge: a vision-language model grades before/after screenshots on separate dimensions.

Mirrors the criteria in Figma's post ("the requested element is present, legible, and contained;
unrelated content is preserved"): each dimension is judged separately, not as one overall score.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

ROUTER = "https://router.huggingface.co/v1"
DIMENSIONS = ("fulfilled", "preserved", "legible", "contained")

PROMPT = """You are a senior product designer reviewing an edit to a UI design.
Image 1 is BEFORE the edit, image 2 is AFTER.

Request: {request}
Patch that was applied (for reference): {patch}

Judge each dimension independently, true or false:
- fulfilled: the AFTER image clearly does what the request asks.
- preserved: nothing unrelated to the request visibly changed.
- legible: all text in AFTER is readable (no low-contrast, hidden or same-color text).
- contained: nothing in AFTER overlaps, is clipped, overflows its container or runs off the page.

Reply with only a JSON object: {{"fulfilled": bool, "preserved": bool, "legible": bool, "contained": bool, "reason": "<one sentence>"}}"""


def _img(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def judge(before_png: bytes, after_png: bytes, request: str, patch: str, model: str,
          token: str | None = None, base_url: str = ROUTER, retries: int = 4) -> dict | None:
    """Per-dimension verdicts plus 'reason', or None if the judge gave no parseable answer."""
    token = token or os.environ["HF_TOKEN"]
    content = [{"type": "text", "text": PROMPT.format(request=request, patch=patch)}, _img(before_png), _img(after_png)]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": 4000, "temperature": 0}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(f"{base_url}/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                text = json.load(resp)["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                out = json.loads(m.group(0))
                if all(isinstance(out.get(d), bool) for d in DIMENSIONS):
                    return out
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            pass
        time.sleep(2 ** (attempt + 1))
    return None


def passed(verdict: dict | None) -> bool:
    return bool(verdict) and all(verdict[d] for d in DIMENSIONS)
