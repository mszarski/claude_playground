"""One self-contained HTML page per scenario, drawn in 3-D by three.js in the browser (docs/tasks/VISUALIZATION.md
section 5; package V4).

No Python dependency: the page is written from a scene trace (`viz/trace.py`, with `viz/scene3d.py`'s additions when
present) with the standard library. three.js (MIT, release 0.170.0, one ES module file, `three.module.min.js`,
691,648 bytes) is loaded from a pinned public CDN (`three=None` or "cdn": a small page that needs the network) or
embedded (`three="embed"`: the copy vendored under third_party/three/, or `three=<path>`: any local copy): the page
then works offline. The camera controls are our own (no three.js add-on).

What a page draws: the sea surface and the bed as distinct, labelled, semi-transparent surfaces (the surface with its
rms height where a model sets one; the bed flat, tilted, by range, a 3-D grid or a sediment interface, as the tier
has it; walls as planes); the projector, the hydrophones (glyphs of constant screen size, coloured by row, with the
array's axis and look), the targets and their motion over the pings; the processing's receive beams (`viz/beams.py`)
and the projector's pattern; the eigenrays of each leg with their bounces; one-way sources and the projector's
direct sound; the sound-speed profile; the models in force and what is not shown. A click on a ray gives its delay,
level and bounce list (boundary, grazing angle, loss); a hydrophone selector lists that hydrophone's arrivals. A time
slider moves a dot along every ray and lights the hydrophones as the strongest pairs arrive. Sound is package V5;
the live chain's taps (`trace["chain"]`) are package V6: the page keeps a `chain` member (null) for them.

Size: the page holds a thinned trace. `page_data` keeps the `max_rays` strongest eigenrays a leg and the `max_pairs`
strongest pairs, simplifies each polyline (Douglas-Peucker, tolerance 1/1000 of the scene's width across and 1/500 of
its depth, bounce points kept) and thins further until the data fit `DATA_BUDGET` bytes, so that a CDN page stays
under 400 kB and an embedded page under 1.2 MB (`PAGE_LIMIT_CDN`, `PAGE_LIMIT_EMBED`; ours).
"""

from __future__ import annotations

import html
import json
import math
import os

THREE_VERSION = "0.170.0"
THREE_CDN = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.min.js"
VENDORED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "third_party", "three",
                                        "three.module.min.js"))
DATA_BUDGET = 330_000          # bytes of page data (JSON), after thinning
PAGE_LIMIT_CDN = 400_000       # a page that loads three.js from the CDN
PAGE_LIMIT_EMBED = 1_200_000   # a page with three.js embedded (0.69 MB of it the library)


def three_source(arg: str | None) -> str | None:
    """`--three cdn|embed|PATH` to what `write_page` takes: None (CDN) or a local file to embed. "embed" is the
    vendored copy (an error if it is missing); None means embed when the vendored copy exists, else the CDN."""
    if arg == "cdn":
        return None
    if arg == "embed":
        if not os.path.exists(VENDORED):
            raise FileNotFoundError(f"no vendored three.js at {VENDORED}")
        return VENDORED
    if arg is None:
        return VENDORED if os.path.exists(VENDORED) else None
    return arg


# --- thinning -------------------------------------------------------------------------------------------------------

def _rdp(points, scale, keep):
    """Douglas-Peucker on `points` [[x, y, z]] in coordinates divided by `scale` (tolerance 1 there); indices in
    `keep` are always kept. Returns the kept indices."""
    n = len(points)
    if n <= 2:
        return list(range(n))
    q = [[p[0] / scale[0], p[1] / scale[1], p[2] / scale[2]] for p in points]
    marks = [False] * n
    marks[0] = marks[-1] = True
    for i in keep:
        if 0 <= i < n:
            marks[i] = True
    anchors = [i for i in range(n) if marks[i]]
    stack = list(zip(anchors[:-1], anchors[1:]))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        pa, pb = q[a], q[b]
        d = [pb[k] - pa[k] for k in range(3)]
        ll = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        worst, wi = -1.0, -1
        for i in range(a + 1, b):
            v = [q[i][k] - pa[k] for k in range(3)]
            if ll > 0:
                t = max(0.0, min(1.0, (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]) / ll))
                w = [v[k] - t * d[k] for k in range(3)]
            else:
                w = v
            e = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
            if e > worst:
                worst, wi = e, i
        if worst > 1.0:
            marks[wi] = True
            stack += [(a, wi), (wi, b)]
    return [i for i in range(n) if marks[i]]


def simplify(line, bounces, scale):
    """A polyline with fewer points (`_rdp`), its bounce points kept (the vertex nearest each bounce)."""
    keep = []
    for b in bounces:
        bx = b["xyz"]
        keep.append(min(range(len(line)), key=lambda i: sum((line[i][k] - bx[k]) ** 2 for k in range(3))))
    return [line[i] for i in _rdp(line, scale, keep)]


def _r(v, nd):
    if isinstance(v, float):
        return round(v, nd) if math.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_r(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _r(x, nd) for k, x in v.items()}
    return v


def _db(a):
    return round(20.0 * math.log10(max(abs(a), 1e-30)), 1)


def _ray(r, scale, nd, prefer3d=True):
    """One eigenray as the page holds it, or None when it has no path to draw."""
    b3 = r.get("bellhop3d") if prefer3d else None
    line = b3["polyline"] if b3 else r.get("polyline")
    if line is None:
        return None
    bounces = b3["bounces"] if b3 else r.get("bounces") or []
    out = {"k": r["k"], "delay": round(r["delay_s"], 8), "ns": r["n_surface"], "nb": r["n_bottom"],
           "amp": r["amplitude"], "db": _db(r["amplitude"]), "launch": _r(r.get("launch_deg"), 2),
           "line": _r(simplify(line, bounces, scale), nd),
           "bounces": [{"xyz": _r(b["xyz"], nd), "b": b["boundary"], "g": _r(b.get("grazing_deg"), 1),
                        "loss": None if b.get("loss_db") is None else _r(float(b["loss_db"]), 2)} for b in bounces]}
    if b3:
        out["path"] = (f"Bellhop3D's own 3-D ray (run type R, a separate process), shot about the eigenray's launch "
                       f"angles and matched by its bounce counts: passes {b3['miss_m']:.1f} m from the target; travel "
                       f"time along it {b3['travel_time_s']:.5f} s against the provider's delay {r['delay_s']:.5f} s")
        out["miss"] = round(b3["miss_m"], 2)
        out["tt"] = round(b3["travel_time_s"], 7)
        out["three_d"] = True
    else:
        out["path"] = r.get("path")
        if r.get("polyline_delay_s") is not None:
            out["tt"] = round(r["polyline_delay_s"], 7)
    return out


def page_data(trace: dict, max_rays: int = 8, max_pairs: int = 40, tol: float = 1.0, nd: int = 2) -> dict:
    """What the page draws: the trace thinned (the `max_rays` strongest drawable eigenrays a leg with their polylines
    simplified at `tol` times the module's tolerance and their bounces, the `max_pairs` strongest two-way pairs with
    their per-hydrophone delays) and rounded to 10**-nd m and 10 ns."""
    s3 = trace.get("scene3d") or {}
    pts = [trace["sonar"]["projector"]] + trace["sonar"]["hydrophones"] + [t["position"] for t in trace["targets"]]
    span = max(max(p[0] for p in pts) - min(p[0] for p in pts), max(p[1] for p in pts) - min(p[1] for p in pts), 1.0)
    env = trace["environment"]
    depth = env.get("bottom_z") or env.get("bottom_z_site") or max([abs(p[2]) for p in pts] + [1.0])
    scale = (1e-3 * tol * span, 1e-3 * tol * span, 2e-3 * tol * max(depth, 0.1))
    legs, undrawn = [], 0
    for leg in trace["legs"]:
        out = {}
        for which in ("out", "back"):
            drawable = [r for r in leg[which] if r.get("polyline") is not None or r.get("bellhop3d")]
            undrawn += len(leg[which]) - len(drawable)
            rays = sorted(drawable, key=lambda r: -r["amplitude"])[:max_rays]
            out[which] = [x for x in (_ray(r, scale, nd) for r in rays) if x is not None]
            out[which + "_total"] = len(leg[which])
        legs.append({"target": leg["target"], **out})
    pairs = sorted(trace["pairs"], key=lambda p: -p["level_db"])[:max_pairs]
    oneway = []
    for o in trace.get("oneway") or []:
        rays = [x for x in (_ray(r, scale, nd) for r in sorted(o["rays"], key=lambda r: -r["amplitude"])[:max_rays]) if x]
        arr = sorted(o["arrivals"], key=lambda a: -a["level_db"])[:max_pairs]
        oneway.append({"label": o["label"], "position": _r(o["position"], nd), "rays": rays, "signals": o.get("signals"),
                       "arrivals": [{"t": round(a["delay_s"], 8), "ch": [round(v, 8) for v in a["channel_delay_s"]],
                                     "ns": a["n_surface"], "nb": a["n_bottom"], "db": round(a["level_db"], 1), "k": a["k"]}
                                    for a in arr]})
    direct = None
    if trace.get("direct"):
        d = trace["direct"]
        direct = {"boundaries": d["boundaries"],
                  "rays": [x for x in (_ray(r, scale, nd) for r in d["rays"][:max_rays]) if x],
                  "arrivals": [{"t": round(a["delay_s"], 8), "ch": [round(v, 8) for v in a["channel_delay_s"]],
                                "ns": a["n_surface"], "nb": a["n_bottom"], "k": a["k"]} for a in d["arrivals"][:max_pairs]]}
    beams = trace.get("beams")
    if beams:
        beams = {"chain": beams["chain"], "broadside": beams["broadside"], "axis": beams["axis"],
                 "aperture_m": beams.get("aperture_m"), "n_elements": beams.get("n_elements"),
                 "center_frequency": beams.get("center_frequency"), "wavelength": beams.get("wavelength"),
                 "element_beamwidth_az_deg": beams.get("element_beamwidth_az_deg"),
                 "steered": [{k: s[k] for k in ("steer_deg", "peak_deg", "edges_deg")} for s in beams.get("steered") or []],
                 "projector": None if not beams.get("projector") else
                 {k: beams["projector"].get(k) for k in ("boresight", "beamwidth_az_deg", "beamwidth_el_deg", "boresight_rel_deg")},
                 "projector_text": beams.get("projector_text")}
    e = {k: env.get(k) for k in ("sound_speed", "surface_z", "bottom_z", "bottom_z_site", "bottom", "wind_speed", "ssp",
                                 "interface", "critical_angle_deg", "bathymetry_note", "absorption",
                                 "absorption_db_per_km", "planes", "bottom_slope", "wind_speed_propagation")}
    if env.get("surface"):
        e["surface"] = {k: env["surface"].get(k) for k in ("rms_height", "mean_square_slope", "wind_speed", "given")}
        bl = env["surface"].get("bubble_loss")
        if bl:
            e["surface"]["bubble_loss_db_at_10deg"] = None
            try:
                g, l = bl["grazing_deg"], bl["loss_db"]
                i = min(range(len(g)), key=lambda j: abs(g[j] - 10.0))
                e["surface"]["bubble_loss_db_at_10deg"] = l[i]
            except (KeyError, TypeError, ValueError):
                pass
    if env.get("bathymetry") is not None and "bed_grid" not in s3:
        e["bathymetry"] = env["bathymetry"]                       # [[range, depth]] from the projector
    if env.get("ssp_section"):
        sec = env["ssp_section"]
        step = max(1, len(sec["r"]) // 5)
        # c[i] is the profile at range r[i] (BellhopEnv.section)
        e["ssp_section"] = {"r": sec["r"][::step], "z": sec["z"], "c": sec["c"][::step]}
    return {
        "scenario": trace["scenario"], "tier": trace["tier"], "ping": trace.get("ping", 0), "time_s": trace.get("time_s"),
        "sonar": {**{k: trace["sonar"].get(k) for k in ("name", "mode", "band_hz", "center_frequency", "sample_kind",
                                                       "sample_rate", "n_channels", "source_level_db_rms")},
                  "projector": _r(trace["sonar"]["projector"], 4), "hydrophones": _r(trace["sonar"]["hydrophones"], 4)},
        "environment": _r(e, 3), "targets": _r(trace["targets"], 3), "legs": legs, "undrawn": undrawn,
        "pairs": [{"t": round(p["delay_s"], 8), "ch": [round(v, 8) for v in p["channel_delay_s"]], "ns": p["n_surface"],
                   "nb": p["n_bottom"], "db": round(p["level_db"], 1), "target": p.get("target", 0), "out": p.get("out"),
                   "back": p.get("back")} for p in pairs],
        "n_pairs": len(trace["pairs"]), "oneway": oneway, "direct": direct, "beams": beams,
        "models": [{k: m.get(k) for k in ("area", "model", "what", "key", "status", "chosen", "source")} for m in trace["models"]],
        "missing": trace.get("missing") or [], "view": {k: v for k, v in (trace.get("view") or {}).items() if k != "track"},
        "time_spread_taps": trace.get("time_spread_taps", False),
        "scene3d": {"box": s3.get("box"), "bed_grid": s3.get("bed_grid"), "motion": s3.get("motion"),
                    "bellhop3d": s3.get("bellhop3d"), "notes": s3.get("notes") or []},
        "chain": trace.get("chain"),            # reserved for the live chain's taps (V6)
        "limits": {"max_rays": max_rays, "max_pairs": max_pairs, "tolerance": tol},
    }


def fit_data(trace: dict, budget: int = DATA_BUDGET, max_rays: int = 8, max_pairs: int = 40) -> tuple[dict, str]:
    """`page_data` thinned until its JSON is at most `budget` bytes: coarser polylines first, then fewer rays and
    pairs. Returns (data, its JSON)."""
    tol = 1.0
    data = None
    for _ in range(12):
        data = page_data(trace, max_rays, max_pairs, tol)
        blob = json.dumps(data, separators=(",", ":"))
        if len(blob) <= budget:
            return data, blob
        if tol < 8.0:
            tol *= 2.0
        elif max_rays > 2:
            max_rays -= 2
        elif max_pairs > 10:
            max_pairs //= 2
        else:
            break
    data["limits"]["over_budget"] = True
    return data, json.dumps(data, separators=(",", ":"))


def write_page(trace: dict, path: str, three: str | None = None, max_rays: int = 8, max_pairs: int = 40,
               png: str | None = None, index: str | None = "index.html") -> dict:
    """Write the page. `three`: a local three.module.min.js to embed (offline; "embed" for the vendored copy), or None
    / "cdn" to load it from the CDN. `png`: the 2-D page's file name to link (default NAME.png beside the page).
    Returns {"bytes", "embedded", "rays", "pairs", "data_bytes", "limits"}."""
    three = None if three in (None, "cdn") else (VENDORED if three == "embed" else three)
    data, blob = fit_data(trace, DATA_BUDGET, max_rays, max_pairs)
    data["links"] = {"png": png if png is not None else f"{trace['scenario']}.png", "index": index}
    blob = json.dumps(data, separators=(",", ":"))
    if three:
        with open(three, "r", encoding="utf-8") as f:
            lib = f.read()
        loader = ("const THREE = await import(URL.createObjectURL(new Blob([document.getElementById('three-src').textContent],"
                  "{type:'text/javascript'})));")
        lib_tag = f'<script type="text/plain" id="three-src">{lib.replace("</script", "<\\/script")}</script>'
    else:
        loader = f"const THREE = await import('{THREE_CDN}');"
        lib_tag = ""
    blob = blob.replace("</", "<\\/")
    page = (_TEMPLATE.replace("%TITLE%", html.escape(f"{trace['scenario']} 3-D")).replace("%LIB%", lib_tag)
            .replace("%LOADER%", loader).replace("%DATA%", blob))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    n_rays = sum(len(leg["out"]) + len(leg["back"]) for leg in data["legs"])
    return {"bytes": len(page.encode("utf-8")), "embedded": bool(three), "rays": n_rays, "pairs": len(data["pairs"]),
            "data_bytes": len(blob), "limits": data["limits"],
            "three_d_rays": sum(bool(r.get("three_d")) for leg in data["legs"] for w in ("out", "back") for r in leg[w])}


_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "page3d_template.html")
with open(_TEMPLATE_PATH, encoding="utf-8") as _f:
    _TEMPLATE = _f.read()
