"""Visual judge: a vision-language model grades before/after screenshots on separate dimensions.

Mirrors the criteria in Figma's post ("the requested element is present, legible, and contained;
unrelated content is preserved"): each dimension is judged separately, not as one overall score.
"minimal" was added after RL learned to over-edit (adding bold to "bump up the contrast"): the other
dimensions only noticed changes to *other* elements.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request

ROUTER = "https://router.huggingface.co/v1"
DIMENSIONS = ("fulfilled", "preserved", "legible", "contained", "minimal")

PROMPT = """You are a senior product designer reviewing an edit to a UI design.
Image 1 is BEFORE the edit, image 2 is AFTER.

Request: {request}
Patch that was applied (for reference): {patch}

Judge each dimension independently, true or false:
- fulfilled: the AFTER image clearly does what the request asks, including putting things exactly where it says
  (above / below / next to the named element).
- preserved: nothing unrelated to the request visibly changed.
- legible: all text in AFTER is readable (no low-contrast, hidden or same-color text).
- contained: nothing in AFTER overlaps, is clipped, overflows its container or runs off the page.
- minimal: the patch changes ONLY what the request needs. Extra changes to the edited element itself count as
  NOT minimal, e.g. making text bold or larger when only its color was asked for. Changes needed to keep the
  result readable or consistent (e.g. whitening a label on a newly dark button) are fine. Use the patch to check
  this, not just the image.

Reply with only a JSON object: {{"fulfilled": bool, "preserved": bool, "legible": bool, "contained": bool, "minimal": bool, "reason": "<one sentence>"}}"""


def _img(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


CLOSEUP_NOTE = """

Images 3 and 4 are 2x close-ups of the edited area, BEFORE and AFTER.
Some correct edits are visually subtle (a slightly darker gray, a heavier font weight, a 1px border, spacing).
For "fulfilled", rely on the change list above together with the close-ups: if the requested property changed in
the requested direction, the request is fulfilled even when the full-page screenshots look almost identical."""


def judge(before_png: bytes, after_png: bytes, request: str, patch: str, model: str,
          token: str | None = None, base_url: str = ROUTER, retries: int = 4,
          closeups: tuple[bytes, bytes] | None = None) -> dict | None:
    """Per-dimension verdicts plus 'reason', or None if the judge gave no parseable answer."""
    token = token or os.environ["HF_TOKEN"]
    text = PROMPT.format(request=request, patch=patch)
    content = [{"type": "text", "text": text}, _img(before_png), _img(after_png)]
    if closeups:
        head, tail = text.rsplit("\n\nReply with only a JSON object", 1)
        content = [{"type": "text", "text": head + CLOSEUP_NOTE + "\n\nReply with only a JSON object" + tail},
                   _img(before_png), _img(after_png), _img(closeups[0]), _img(closeups[1])]
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
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, KeyError):
            pass
        time.sleep(2 ** (attempt + 1))
    return None


def passed(verdict: dict | None) -> bool:
    return bool(verdict) and all(verdict[d] for d in DIMENSIONS)
