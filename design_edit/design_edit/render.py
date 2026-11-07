"""Render a design to HTML/PNG, for eyeballing data and for vision-model judges."""

from __future__ import annotations

import html

from .schema import Node

_JUSTIFY = {"start": "flex-start", "end": "flex-end", "center": "center", "between": "space-between"}
_ALIGN = {"start": "flex-start", "end": "flex-end", "center": "center", "stretch": "stretch"}


def _style(node: Node) -> str:
    p = node["props"]
    s: list[str] = ["box-sizing:border-box"]
    if node["type"] == "frame":
        layout = p.get("layout")
        if layout in ("row", "col"):
            s += ["display:flex", f"flex-direction:{'row' if layout == 'row' else 'column'}"]
            if "gap" in p:
                s.append(f"gap:{p['gap']}px")
            if "justify" in p:
                s.append(f"justify-content:{_JUSTIFY.get(p['justify'], p['justify'])}")
            s.append(f"align-items:{_ALIGN.get(p.get('align', 'stretch'), 'stretch')}")
        else:
            s.append("position:relative")
        if "pad" in p:
            s.append(f"padding:{p['pad']}px")
    if node["type"] == "text":
        s += [f"font-size:{p.get('size', 14)}px", f"font-weight:{p.get('weight', 400)}",
              f"color:{p.get('color', '#111827')}", "line-height:1.35"]
    if "x" in p or "y" in p:
        s += ["position:absolute", f"left:{p.get('x', 0)}px", f"top:{p.get('y', 0)}px"]
    for key, css in (("w", "width"), ("h", "height")):
        if key in p:
            s.append(f"{css}:{p[key]}px")
            s.append(f"flex-shrink:0")  # fixed sizes are fixed, as in Figma: overflow is visible, not hidden
    if "fill" in p and node["type"] != "text":
        s.append(f"background:{p['fill']}")
    if "stroke" in p:
        s.append(f"border:1px solid {p['stroke']}")
    if "radius" in p:
        s.append(f"border-radius:{p['radius']}px")
    if "opacity" in p:
        s.append(f"opacity:{p['opacity']}")
    return ";".join(s)


def to_html_fragment(node: Node) -> str:
    inner = html.escape(str(node["props"].get("text", ""))) if node["type"] == "text" else ""
    inner += "".join(to_html_fragment(c) for c in node["children"])
    title = html.escape(f"{node['id']} {node['props'].get('name', '')}")
    return f'<div data-id="{node["id"]}" title="{title}" style="{_style(node)}">{inner}</div>'


def to_html(node: Node) -> str:
    return ("<!doctype html><html><head><meta charset='utf-8'><style>body{margin:0;font-family:"
            "Inter,system-ui,sans-serif}</style></head><body>" + to_html_fragment(node) + "</body></html>")


def screenshot(node: Node, path: str, width: int = 1280) -> str:
    """Render to PNG with Playwright (optional dependency)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 800})
        page.set_content(to_html(node))
        page.screenshot(path=path, full_page=True)
        browser.close()
    return path
