"""Deterministic layout validators from real rendered geometry (Playwright + Chromium).

The lint in rewards.py checks values (contrast, spacing scale). This module renders the design and
checks what designers actually see go wrong: elements clipped by their container, siblings
overlapping, text overflowing its box, content pushed off the page.

    with Renderer() as r:
        issues = r.issues(doc)              # set of strings like "clip:n12", "overlap:n4|n5"
        new = r.new_issues(before, after)   # issues the edit introduced
        png = r.screenshot(doc)             # bytes, for a visual judge
"""

from __future__ import annotations

import glob
import os

from .render import to_html
from .schema import Node

_JS_BOXES = """() => {
  const out = {};
  for (const el of document.querySelectorAll('[data-id]')) {
    const r = el.getBoundingClientRect();
    const p = el.parentElement && el.parentElement.closest('[data-id]');
    out[el.dataset.id] = {x: r.x, y: r.y, w: r.width, h: r.height,
      sw: el.scrollWidth, cw: el.clientWidth, sh: el.scrollHeight, ch: el.clientHeight,
      parent: p ? p.dataset.id : null, abs: getComputedStyle(el).position === 'absolute',
      text: el.childElementCount === 0 && el.textContent.trim().length > 0};
  }
  return out;
}"""

TOL = 1.0  # px


def _chromium() -> str | None:
    if os.environ.get("CHROMIUM_PATH"):
        return os.environ["CHROMIUM_PATH"]
    hits = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux64/chrome"))
    return hits[-1] if hits else None


class Renderer:
    def __init__(self, width: int = 1280) -> None:
        self.width = width
        self._pw = self._browser = self._page = None

    def __enter__(self) -> "Renderer":
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        exe = _chromium()
        self._browser = self._pw.chromium.launch(**({"executable_path": exe} if exe else {}),
                                                 args=["--no-sandbox"])
        self._page = self._browser.new_page(viewport={"width": self.width, "height": 900})
        return self

    def __exit__(self, *exc) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def boxes(self, doc: Node) -> dict:
        self._page.set_content(to_html(doc))
        return self._page.evaluate(_JS_BOXES)

    def issues(self, doc: Node) -> set[str]:
        b = self.boxes(doc)
        root = b.get(doc["id"])
        out: set[str] = set()
        kids: dict[str | None, list[str]] = {}
        for nid, r in b.items():
            kids.setdefault(r["parent"], []).append(nid)
            if r["w"] <= 0 or r["h"] <= 0:
                continue
            p = b.get(r["parent"]) if r["parent"] else None
            if p and (r["x"] < p["x"] - TOL or r["y"] < p["y"] - TOL or
                      r["x"] + r["w"] > p["x"] + p["w"] + TOL or r["y"] + r["h"] > p["y"] + p["h"] + TOL):
                out.add(f"clip:{nid}")
            if r["text"] and (r["sw"] > r["cw"] + TOL and r["cw"] > 0):
                out.add(f"text_overflow:{nid}")
            if root and nid != doc["id"] and (r["x"] + r["w"] > root["x"] + root["w"] + TOL or r["x"] < root["x"] - TOL):
                out.add(f"offscreen:{nid}")
        for sibs in kids.values():
            vis = [s for s in sibs if b[s]["w"] > 0 and b[s]["h"] > 0]
            for i, a in enumerate(vis):
                for c in vis[i + 1:]:
                    A, C = b[a], b[c]
                    ix = min(A["x"] + A["w"], C["x"] + C["w"]) - max(A["x"], C["x"])
                    iy = min(A["y"] + A["h"], C["y"] + C["h"]) - max(A["y"], C["y"])
                    if ix > TOL and iy > TOL:
                        out.add(f"overlap:{min(a, c)}|{max(a, c)}")
        return out

    def new_issues(self, before: Node, after: Node) -> set[str]:
        return self.issues(after) - self.issues(before)

    def screenshot(self, doc: Node) -> bytes:
        self._page.set_content(to_html(doc))
        el = self._page.query_selector(f'[data-id="{doc["id"]}"]')
        return (el or self._page).screenshot(type="png")
