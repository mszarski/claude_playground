"""An index page linking every scenario's page, by family (docs/tasks/VISUALIZATION.md, V2). Standard library and
Pillow (already a dependency, for the thumbnails) only. The page is self-contained but for the pages it links."""

from __future__ import annotations

import html
import json
import os

from .catalogue import FAMILIES


def thumbnail(png: str, out: str, width: int = 220) -> bool:
    """A small JPEG of the top of a page (its section), for the index."""
    try:
        from PIL import Image
        im = Image.open(png).convert("RGB")
        w, h = im.size
        crop = im.crop((0, 0, w, min(h, int(0.33 * h))))
        crop.thumbnail((width, width))
        crop.save(out, "JPEG", quality=70, optimize=True)
        return True
    except Exception:                                         # noqa: BLE001
        return False


def _summary(r: dict) -> str:
    if r.get("error"):
        return "FAILED: " + r["error"]
    bits = [f"{r.get('mode', '?')}", f"{r.get('targets', 0)} target(s)",
            f"{r.get('eigenrays_drawn', 0)} of {r.get('eigenrays', 0)} eigenrays drawn"]
    if r.get("oneway_sources"):
        bits.append(f"{r['oneway_sources']} one-way source(s)")
    if r.get("direct"):
        bits.append("direct sound")
    return "; ".join(bits) + ". " + (r.get("note") or "")


def write_index(rows: list, out: str, total_s: float | None = None, thumbs: bool = True,
                title: str = "Scenario pictures") -> str:
    """Write `out`/index.html and `out`/index.json from the rows `tools/scenario_view.py` prints (one per scenario)."""
    by_name = {r["scenario"]: r for r in rows}
    order = [n for _, _, names in FAMILIES for n in names if n in by_name]
    order += sorted(n for n in by_name if n not in order)
    if thumbs:
        os.makedirs(os.path.join(out, "thumbs"), exist_ok=True)
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
             f"<title>{html.escape(title)}</title><style>"
             ":root{--bg:#fafaf8;--ink:#1d1d1b;--muted:#5d5d58;--line:#d6d6d0}"
             "@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#15171a;--ink:#e8e8e4;--muted:#a3a39d;--line:#3a3d42}}"
             "body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}"
             "table{border-collapse:collapse;width:100%}td,th{border-top:1px solid var(--line);padding:6px 8px;vertical-align:top;text-align:left}"
             ".m{color:var(--muted)} img{max-width:220px;height:auto;display:block} h2{margin-top:28px}"
             ".wrap{overflow-x:auto}</style></head><body>",
             f"<h1>{html.escape(title)}</h1><p class='m'>One 2-D page per scenario and, where made, a 3-D page (docs/tasks/VISUALIZATION.md). Every picture is "
             "the simulator's own model, not a measurement. Made by tools/scenario_view.py"
             + (f" in {total_s / 60:.1f} min" if total_s else "") + f"; {len(rows)} scenario(s).</p>"]
    fam_of = {n: (t, s) for t, s, names in FAMILIES for n in names}
    current = None
    for n in order:
        r = by_name[n]
        fam = fam_of.get(n, ("Other", "other"))[0]
        if fam != current:
            if current is not None:
                parts.append("</table></div>")
            parts.append(f"<h2>{html.escape(fam)}</h2><div class='wrap'><table><tr><th>scenario</th><th>tier</th><th>page</th><th>3-D</th><th>summary</th></tr>")
            current = fam
        png = os.path.join(out, f"{n}.png")
        img = ""
        if thumbs and os.path.exists(png) and thumbnail(png, os.path.join(out, "thumbs", f"{n}.jpg")):
            img = f"<a href='{html.escape(n)}.png'><img src='thumbs/{html.escape(n)}.jpg' alt='{html.escape(n)} section'></a>"
        elif os.path.exists(png):
            img = f"<a href='{html.escape(n)}.png'>page</a>"
        missing = r.get("missing") or []
        page3 = os.path.join(out, f"{n}.html")
        three = (f"<a href='{html.escape(n)}.html'>3-D page</a><br><span class='m'>{os.path.getsize(page3) / 1e3:.0f} kB"
                 + (f"; {r['html'].get('three_d_rays')} Bellhop3D rays" if (r.get("html") or {}).get("three_d_rays") else "")
                 + "</span>") if os.path.exists(page3) else "<span class='m'>none</span>"
        snd = r.get("audio") or {}
        if snd.get("method"):
            three += (f"<br><span class='m'>sound: {html.escape(snd['method'].replace('_', ' '))}, {snd.get('files')} WAV, "
                      f"{(snd.get('bytes') or 0) / 1e6:.1f} MB</span>")
        elif snd.get("skipped") or snd.get("error"):
            three += f"<br><span class='m'>no sound: {html.escape(str(snd.get('skipped') or snd.get('error'))[:120])}</span>"
        parts.append(f"<tr><td><a href='{html.escape(n)}.png'>{html.escape(n)}</a></td><td>{html.escape(str(r.get('tier', '?')))}</td>"
                     f"<td>{img}</td><td>{three}</td><td>{html.escape(_summary(r))}"
                     + (f"<br><span class='m'>not shown: {html.escape('; '.join(missing))}</span>" if missing else "")
                     + "</td></tr>")
    if current is not None:
        parts.append("</table></div>")
    parts.append("</body></html>")
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return path
