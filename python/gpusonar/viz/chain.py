"""The live chain seen in the scene viewer (docs/tasks/VISUALIZATION.md section 15; package V6).

The owner (2026-11-26): "ideally it uses taps from the real thing as well so we can visualize how the tracker is
working". This module runs the live chain (`gpusonar.live.LiveChain`, either engine) over a scene's pings and puts what
it gives into the scene trace's `chain` member, per ping, for `viz/page3d.py` and a 2-D chain page:

- from the report line alone (both engines, nothing changed): detections (range, bearing, snr, extent, arrivals, the
  optional range rate and Doppler path), tracks (id, kind, parent, position, velocity, speed, mean snr, seen), own-ship,
  `pings_without_nav`; and the existing `beamform` tap's power map (LiveChain.last: the cell map the detector reads);
- from the opt-in chain taps (`live.CHAIN_TAPS`, both engines, read only): the tracker's gates, associations,
  covariances, births, confirmations and drops (`detect.TRACKER_TAP`), the classifier's evidence per track
  (`classify.CLASS_EVIDENCE`), the detector's background per cell with its plan (thresholds by statistic);
- the truth to compare with: each labelled scene object's centroid at every ping (the scenario's scene, `Scene.at_time`;
  seafloor, sea-surface and volume items and areas wider than 30 m left out, as tools/gallery.py `scene_truth`), and
  per track the nearest object and the horizontal distance to it at the report's time (the association error).

The map is downsampled for the page (block maxima of the cell map in dB over its median; the detector's background by
block means) to keep an embedded page under `PAGE_LIMIT_CHAIN_EMBED` (1.5 MB, of which 0.69 MB three.js; ours).
Coordinates are the report's: world x north, y east (m); the tracker's frame (taps) is (east, north) offsets from the
first ping's array origin (`origin0`), converted here to world north/east for drawing.

Chain configuration: the deployment file under configs/deploy whose source names the scene's sonar (its `chain`
block), else the live chain's defaults (`live.DEFAULTS`); `chain_config` says which, and the page prints it.
"""

from __future__ import annotations

import base64
import copy
import glob
import json
import math
import os
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

CHAIN_MAP_BUDGET = 360_000       # bytes of base64 map data in a page (power and background, all pings; ours)
CHAIN_BUDGET = 560_000           # bytes of the chain member's JSON in a page (ours)
MAX_MAP_BEARINGS = 96            # rows of the downsampled map (ours)
MIN_MAP_RANGES = 48
TRUTH_MAX_WIDTH_M = 30.0         # an object wider than this is an area (seabed, waves), not a target (tools/gallery.py)
BOUNDARY_TYPES = ("seafloor", "sea_surface", "volume")
# The tolerances tests/python/test_live_cpp.py holds the two engines' report lines to (test_cpp_chain_gives_the_python_chains_report).
ENGINE_TOL = {"range_m": 0.05, "bearing_deg": 0.05, "x": 0.05, "y": 0.05, "vx": 0.05, "vy": 0.05}


# --- configuration -----------------------------------------------------------------------------------------------------

def chain_config(sonar_name: str) -> tuple[dict, str]:
    """(the chain block, where it comes from): the deployment file whose source names `sonar_name`, else {} (the live
    chain's defaults)."""
    from ..config import load_yaml
    for path in sorted(glob.glob(os.path.join(_ROOT, "configs", "deploy", "*.yaml"))):
        try:
            cfg = load_yaml(path)
        except Exception:                                    # noqa: BLE001  (a picture never stops on a file)
            continue
        if (cfg.get("source") or {}).get("sonar") == sonar_name:
            return copy.deepcopy(cfg.get("chain") or {}), f"the chain block of {os.path.relpath(path, _ROOT)}"
    return {}, "the live chain's defaults (gpusonar/live.py DEFAULTS): no deployment file under configs/deploy names this sonar"


def look_toward_boresight(config: dict, sonar_name: str, record) -> tuple[dict, str | None]:
    """The chain's `look` (which side of the line array its bearings are on) set so that its broadside is the sonar's
    boresight, when the configuration leaves `look` to the default and the default looks the other way: a
    forward-looking sonar's array is across the bow, and the chain's default (starboard, for a flank array) would place
    every echo astern, mirrored. Returns (config, a sentence for the page, or None when nothing changed)."""
    from .. import _gpusonar as core
    from ..live import DEFAULTS
    from ..sonar import SonarModel
    if "look" in config:
        return config, None
    rx = np.asarray(record.rx_positions(), float)
    axis = rx[-1, :2] - rx[0, :2]
    if float(np.linalg.norm(axis)) < 1e-9:
        return config, None
    axis = axis / np.linalg.norm(axis)
    sonar = SonarModel.load(sonar_name)
    bore = np.asarray(core.rotation_ned(*record.geometry.rx_pose.rpy), float) @ np.asarray(sonar.boresight, float)
    side = 1.0 if DEFAULTS["look"] == "starboard" else -1.0
    broadside = side * np.array([-axis[1], axis[0]])
    if float(broadside @ bore[:2]) >= 0.0:
        return config, None
    other = "port" if DEFAULTS["look"] == "starboard" else "starboard"
    return dict(config, look=other), (f"look: {other}, not the default {DEFAULTS['look']}: chosen so that the chain's "
                                      "broadside is the sonar's boresight (the default would place every echo on the "
                                      "other side of the array, mirrored)")


# --- truth ---------------------------------------------------------------------------------------------------------------

def truth_objects(cfg: dict, scene, times) -> list[dict]:
    """[{label, positions [[x, y, z] per ping]}]: each labelled scene object's centroid (all its scatterers, every part
    with its label) at each ping's scenario time (`Scene.at_time`). Left out, as tools/gallery.py `scene_truth` does:
    the seafloor, sea-surface and volume items, and items given as an area (`x_range`, `y_range`) wider than
    TRUTH_MAX_WIDTH_M (a seabed, waves, a boulder field)."""
    skip = set()
    for it in cfg.get("scene") or []:
        if it.get("type") in BOUNDARY_TYPES:
            skip.add(it.get("label"))
        elif "x_range" in it and max(np.ptp(it["x_range"]), np.ptp(it.get("y_range", [0.0, 0.0]))) > TRUTH_MAX_WIDTH_M:
            skip.add(it.get("label"))
    groups: dict = {}
    first = 0
    for label, count in zip(scene.labels, scene.counts):
        if count and label not in skip:
            groups.setdefault(str(label), []).extend(range(first, first + count))
        first += count
    per_time = [scene.at_time(float(t)) for t in times]
    return [{"label": label, "positions": [[float(v) for v in P[idx].mean(axis=0)] for P in per_time]}
            for label, idx in groups.items()]


def associate(tracks: list[dict], truth: list[dict], ping: int) -> list[dict]:
    """Per report track: the nearest truth object at this ping and the horizontal distance to it (m), or None when the
    scene has no object. The association error is that distance (ours: nearest object, no gate)."""
    out = []
    for t in tracks:
        best = None
        for o in truth:
            p = o["positions"][ping]
            d = float(math.hypot(t["x"] - p[0], t["y"] - p[1]))
            if best is None or d < best[1]:
                best = (o["label"], d)
        out.append({"id": t["id"], "truth": best[0] if best else None, "error_m": best[1] if best else None})
    return out


# --- simulate once, cache ---------------------------------------------------------------------------------------------

def gallery_quick(name: str):
    """tools/gallery.py's `--quick` thinning of a scene (a function of the config), or None; loaded from the file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_gallery_for_chain", os.path.join(_ROOT, "tools", "gallery.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.QUICK.get(name), sorted(mod.RUNNERS)


def simulate_and_cache(name: str, out_dir: str, quick: bool = False, engine: str | None = "cpp") -> dict:
    """Run the scenario (with the gallery's `--quick` thinning when `quick` and it has one) and write
    `out_dir`/NAME.records.h5 (every ping) and NAME.truth.json (truth objects, scenario times, platform positions,
    what was run). Returns the truth file's content."""
    from .. import _gpusonar as core
    from ..config import load_scenario_config
    from ..sim import run_scenario
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_scenario_config(name)
    thin = None
    if quick:
        fn, _ = gallery_quick(name)
        if fn is not None:
            fn(cfg)
            thin = "tools/gallery.py --quick thinning (QUICK[name]): fewer pings and scatterers, the same geometry"
    t0 = time.perf_counter()
    res = run_scenario(cfg, engine=engine)
    sim_s = time.perf_counter() - t0
    writer = core.RecordWriter(os.path.join(out_dir, f"{name}.records.h5"))
    for r in res.records:
        writer.write(r)
    writer.close()
    times = [float(t) for t in (res.times if res.times is not None else np.arange(len(res.records)))]
    info = {"scenario": name, "pings": len(res.records), "times_s": times, "quick": thin, "engine": engine or "default",
            "simulate_s": round(sim_s, 1), "sonar": cfg["sonar"],
            "platform": None if res.true_positions is None else [[float(v) for v in p] for p in res.true_positions],
            "truth": truth_objects(cfg, res.scene, times),
            "truth_rule": ("each labelled scene object's centroid at the ping's scenario time (Scene.at_time); seafloor, "
                           f"sea surface and volume items and areas wider than {TRUTH_MAX_WIDTH_M:g} m left out")}
    with open(os.path.join(out_dir, f"{name}.truth.json"), "w", encoding="utf-8") as f:
        json.dump(info, f)
    return info


def refresh_truth(name: str, out_dir: str) -> dict:
    """NAME.truth.json's truth made again from the scenario's scene (built as run_scenario builds it: the same seed;
    no simulation) at the cached pings' times, with the thinning the cache says it had."""
    from .. import acoustics
    from ..config import load_scenario_config
    from ..sim.scene import build_scene
    path = os.path.join(out_dir, f"{name}.truth.json")
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    cfg = load_scenario_config(name)
    if info.get("quick"):
        fn, _ = gallery_quick(name)
        if fn is not None:
            fn(cfg)
    cfg = acoustics.resolve_sea_state(cfg)
    scene = build_scene(cfg.get("scene", []), np.random.default_rng(int(cfg.get("seed", 0))))
    info["truth"] = truth_objects(cfg, scene, info["times_s"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f)
    return info


def read_records(path: str) -> list:
    from ..sources.container import ContainerSource
    with ContainerSource(path) as src:
        return list(src)


# --- one run of the chain ----------------------------------------------------------------------------------------------

def run_chain(records, config: dict, engine: str = "cpp", taps=("tracker", "classifier", "detector")) -> list[dict]:
    """The live chain over copies of `records`: per ping {report, map: {power [B, R], ranges, bearings} or None, taps,
    geometry: {origin, axis, broadside} of the array as the chain places detections, tap_seconds}. The taps are opt-in
    (`taps=()` runs the chain as it is)."""
    from ..live import LiveChain
    cfg = dict(copy.deepcopy(config), engine=engine)
    cfg.setdefault("backend", "host")
    out = []
    with LiveChain(cfg, taps={"chain": list(taps)} if taps else None) as chain:
        side = 1.0 if chain.cfg["look"] == "starboard" else -1.0
        for r in records:
            rec = r.copy()
            rx = np.asarray(rec.rx_positions(), float)
            axis = rx[-1, :2] - rx[0, :2]
            axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
            t0 = time.perf_counter()
            rep = chain.process(rec)
            dt = time.perf_counter() - t0
            last = chain.last or {}
            power = last.get("power")
            m = None
            if power is not None and last.get("ranges") is not None and np.size(power):
                m = {"power": np.asarray(power, dtype=np.float32), "ranges": np.asarray(last["ranges"], float),
                     "bearings": np.asarray(chain.bearings, float)}
            out.append({"report": rep, "map": m, "taps": chain.chain_taps, "seconds": dt,
                        "geometry": {"origin": [float(v) for v in rep["ownship_xyz"]], "axis": [float(axis[0]), float(axis[1])],
                                     "broadside": [float(side * -axis[1]), float(side * axis[0])], "look": chain.cfg["look"]},
                        "bearings_deg": [float(v) for v in np.degrees(chain.bearings)]})
    return out


def compare_engines(a: list[dict], b: list[dict]) -> list[str]:
    """Differences between two runs' report lines beyond ENGINE_TOL (the tolerances test_live_cpp.py uses): counts,
    ids, kinds and parents exactly; ranges, bearings, positions and velocities to 0.05. [] when there are none."""
    diffs = []
    for i, (x, y) in enumerate(zip(a, b)):
        rx, ry = x["report"], y["report"]
        if len(rx["detections"]) != len(ry["detections"]):
            diffs.append(f"ping {i}: {len(rx['detections'])} against {len(ry['detections'])} detections")
        else:
            for j, (d, e) in enumerate(zip(rx["detections"], ry["detections"])):
                for k in ("range_m", "bearing_deg"):
                    if abs(d[k] - e[k]) > ENGINE_TOL[k]:
                        diffs.append(f"ping {i} detection {j}: {k} {d[k]:.4f} against {e[k]:.4f}")
        ta = [(t["id"], t["kind"], t["parent"]) for t in rx["tracks"]]
        tb = [(t["id"], t["kind"], t["parent"]) for t in ry["tracks"]]
        if ta != tb:
            diffs.append(f"ping {i}: tracks (id, kind, parent) {ta} against {tb}")
        else:
            for t, u in zip(rx["tracks"], ry["tracks"]):
                for k in ("x", "y", "vx", "vy"):
                    if abs(t[k] - u[k]) > ENGINE_TOL[k]:
                        diffs.append(f"ping {i} track {t['id']}: {k} {t[k]:.3f} against {u[k]:.3f}")
    if len(a) != len(b):
        diffs.append(f"{len(a)} against {len(b)} pings")
    return diffs


# --- the trace's chain member ------------------------------------------------------------------------------------------

def _pool(a: np.ndarray, nb: int, nr: int, how: str) -> np.ndarray:
    """Block maxima (`max`) or means (`mean`) of [B, R] to at most [nb, nr] (edge blocks padded)."""
    B, R = a.shape
    kb, kr = max(1, math.ceil(B / nb)), max(1, math.ceil(R / nr))
    pb, pr = math.ceil(B / kb) * kb, math.ceil(R / kr) * kr
    fill = -np.inf if how == "max" else np.nan
    p = np.full((pb, pr), fill)
    p[:B, :R] = a
    p = p.reshape(pb // kb, kb, pr // kr, kr)
    return (np.max(p, axis=(1, 3)) if how == "max" else np.nanmean(p, axis=(1, 3))), kb, kr


def _u8(db, lo, hi):
    return np.clip(np.rint((np.asarray(db) - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def _rnd(v, nd=3):
    if isinstance(v, float):
        return round(v, nd) if math.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_rnd(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _rnd(x, nd) for k, x in v.items()}
    return v


def map_grid(n_pings: int, n_bearings: int, n_ranges: int, budget: int = CHAIN_MAP_BUDGET, layers: int = 2) -> tuple[int, int]:
    """(rows, columns) of the downsampled map: every ping's `layers` u8 grids, as base64, within `budget` bytes."""
    nb = min(n_bearings, MAX_MAP_BEARINGS)
    cells = budget * 3 // 4 // max(n_pings * layers, 1)
    nr = max(MIN_MAP_RANGES, min(n_ranges, cells // max(nb, 1)))
    while nb * nr * n_pings * layers * 4 // 3 > budget and nb > 16:
        nb = max(16, nb * 3 // 4)
    return nb, nr


def chain_member(runs: list[dict], truth: dict | None, config_source: str, engine: str, config: dict | None = None,
                 other: dict | None = None, audio: dict | None = None, note: str | None = None) -> dict:
    """The trace's `chain` member from `run_chain`'s runs (docs/tasks/VISUALIZATION.md section 15): per ping the report
    line (detections, tracks, own-ship, pings_without_nav), the downsampled power map and detector background, the
    tracker's tap (gates, associations, covariances in world north/east) and the classifier's evidence, the truth
    objects and the association error. `other`: {engine, differences} of a second engine's run (`--chain-engine both`)."""
    n = len(runs)
    maps = [r["map"] for r in runs if r["map"] is not None]
    nbm = max((m["power"].shape[0] for m in maps), default=1)
    nrm = max((m["power"].shape[1] for m in maps), default=1)
    nb, nr = map_grid(max(n, 1), nbm, nrm)
    origin0 = runs[0]["report"]["ownship_xyz"] if runs else [0.0, 0.0, 0.0]
    tr_truth = (truth or {}).get("truth") or []
    pings = []
    for i, run in enumerate(runs):
        rep = run["report"]
        p = {"ping": rep.get("ping", i), "time_ns": rep.get("time_ns"), "ownship": _rnd(rep["ownship_xyz"], 3),
             "pings_without_nav": rep.get("pings_without_nav"), "geometry": _rnd(run["geometry"], 6),
             "detections": [_rnd(d, 4) for d in rep["detections"]],
             "tracks": [_rnd(t, 4) for t in rep["tracks"]], "seconds": round(run["seconds"], 4),
             "report_counts": {"detections": len(rep["detections"]), "tracks": len(rep["tracks"])}}
        if rep.get("blanking"):
            p["blanking"] = rep["blanking"]
        m = run["map"]
        if m is not None:
            pw = np.asarray(m["power"], float)
            med = float(np.median(pw[pw > 0])) if np.any(pw > 0) else 1.0
            db = 10.0 * np.log10(np.maximum(pw, 1e-30) / med)
            pooled, kb, kr = _pool(db, nb, nr, "max")
            lo, hi = -3.0, 40.0
            ranges, bearings = m["ranges"], np.degrees(m["bearings"])
            mp = {"rows": int(pooled.shape[0]), "cols": int(pooled.shape[1]), "lo_db": lo, "hi_db": hi,
                  "u8": _b64(_u8(pooled, lo, hi)), "median": med,
                  "r0": float(ranges[0]), "dr": float((ranges[1] - ranges[0]) * kr if len(ranges) > 1 else 1.0),
                  "b0": float(bearings[0]), "db": float((bearings[1] - bearings[0]) * kb if len(bearings) > 1 else 1.0),
                  "block": [kb, kr], "native": [int(pw.shape[0]), int(pw.shape[1])],
                  "what": "the cell map the detector reads (LiveChain.last power, the beamform tap), dB over its median, "
                          f"block maxima of {kb} bearings x {kr} range cells"}
            det = (run.get("taps") or {}).get("detector")
            if det is not None and np.shape(det["background"]) == pw.shape:
                bg = 10.0 * np.log10(np.maximum(np.asarray(det["background"], float), 1e-30) / med)
                lin = 10.0 ** (bg / 10.0)
                bpool, _, _ = _pool(lin, nb, nr, "mean")
                mp["background_u8"] = _b64(_u8(10.0 * np.log10(np.maximum(bpool, 1e-30)), lo, hi))
                plan = det.get("plan") or {}
                mp["thresholds"] = [{"cells": int(s.get("cells", 0)), "pfa": s.get("pfa"),
                                     "factor_db": float(s["threshold_db"]) if "threshold_db" in s else
                                     10.0 * math.log10(max(float(s["factor"]), 1e-30))}
                                    for s in plan.get("scales") or [] if "threshold_db" in s or "factor" in s]
            p["map"] = mp
        tp = run.get("taps") or {}
        trk = tp.get("tracker")
        if trk is not None:
            p["tracker"] = _tracker_world(trk, origin0)
        if tp.get("classifier") is not None:
            p["evidence"] = [_rnd(e, 4) for e in tp["classifier"]]
        if tr_truth:
            p["association"] = [_rnd(a, 3) for a in associate(rep["tracks"], tr_truth, min(i, len(tr_truth[0]["positions"]) - 1))]
        pings.append(p)
    member = {"kind": "gpusonar.chain_view", "version": 1, "engine": engine, "config_source": config_source,
              "config": _rnd(config or {}, 6), "origin0": _rnd(origin0, 4), "n_pings": n, "pings": pings,
              "truth": [{"label": o["label"], "positions": _rnd(o["positions"], 3)} for o in tr_truth],
              "truth_rule": (truth or {}).get("truth_rule"), "times_s": (truth or {}).get("times_s"),
              "quick": (truth or {}).get("quick"), "taps": sorted({k for r in runs for k in (r.get("taps") or {}) if (r.get("taps") or {}).get(k) is not None}),
              "tap_seconds_per_ping": None, "other_engine": other, "audio": audio, "note": note,
              "map_limits": {"budget_bytes": CHAIN_MAP_BUDGET, "rows": nb, "cols": nr}}
    return member


def _tracker_world(tap: dict, origin0) -> dict:
    """The tracker's tap with positions, velocities and covariances turned from the tracker's frame ((east, north)
    offsets from `origin0`, the first ping's array) into world (north, east) for drawing."""
    n0, e0 = float(origin0[0]), float(origin0[1])

    def pos(x):          # (east, north) offset -> world (north, east)
        return [round(float(x[1]) + n0, 3), round(float(x[0]) + e0, 3)]

    def cov2(c):         # [[ee, en], [ne, nn]] -> (nn, ne, ee)
        return [round(float(c[1][1]), 4), round(float(c[0][1]), 4), round(float(c[0][0]), 4)]

    def track(t):
        c = np.asarray(t["cov"], float).reshape(4, 4)
        return {"id": t["id"], "pos": pos(t["state"][:2]), "vel": [round(float(t["state"][3]), 4), round(float(t["state"][2]), 4)],
                "cov": cov2(c[:2, :2]), "hits": t["hits"], "misses": t["misses"], "confirmed": t["confirmed"],
                "time": _rnd(t["time"], 4), **({"reason": t["reason"]} if "reason" in t else {})}
    pairs = []
    for q in tap["pairs"]:
        s = q["s"]                   # [ee, en, nn] in the tracker's frame
        pairs.append({"track": q["track"], "detection": q["detection"], "d2": _rnd(q["d2"], 4), "cost": _rnd(q["cost"], 4),
                      "gated": q["gated"], "nearest": q["nearest"], "pos": pos(q["x"]),
                      "s": [round(float(s[2]), 4), round(float(s[1]), 4), round(float(s[0]), 4)]})
    return {"time_s": _rnd(tap["time_s"], 4), "gate": tap["gate"], "association": tap["association"],
            "before": tap["before"], "pairs": pairs, "took": tap["took"], "missed": tap["missed"],
            "dropped": [track(t) for t in tap["dropped"]], "born": tap["born"], "confirmed_now": tap["confirmed_now"],
            "after": [track(t) for t in tap["after"]]}


def chain_json_bytes(member: dict) -> int:
    return len(json.dumps(member, separators=(",", ":")))


# --- sound per ping -----------------------------------------------------------------------------------------------------

def write_chain_audio(out_dir: str, name: str, records, channels=(0,), band=None) -> dict:
    """One WAV per ping for each hydrophone in `channels` (viz/audio.py: the default method of the record, one gain for
    every ping and hydrophone so that levels between pings are preserved) under `out_dir`/NAME.chain_audio/. Returns
    {dir, method_words, files [{ping, channel, file, bytes}], time_scale, record_start_s [per ping]}."""
    from . import audio
    adir = os.path.join(out_dir, f"{name}.chain_audio")
    os.makedirs(adir, exist_ok=True)
    s0 = records[0].samples
    if band is None:
        w = records[0].transmit.waveform
        band = sorted([float(w.f_start), float(w.f_stop)])
    kind = str(s0.kind).split(".")[-1]
    plan = audio.listen_plan(band, kind, float(s0.sample_rate), float(s0.center_frequency), s0.n_samples / float(s0.sample_rate))
    chans = [c for c in channels if 0 <= c < s0.n_channels]
    ys = [audio.to_audio(np.asarray(r.samples.array)[chans], plan) for r in records]
    g = audio.gain_for(ys)
    files = []
    words = audio._method_words(plan)
    for i, y in enumerate(ys):
        for j, c in enumerate(chans):
            fn = f"p{i:03d}.h{c:03d}.{plan['tag']}.wav"
            w = audio.write_pcm16(os.path.join(adir, fn), plan["rate_hz"], y[j], g,
                                  {"INAM": f"{name} ping {i} hydrophone {c}", "ICMT": f"{plan['method']}: {words}; one gain for "
                                   "every ping and hydrophone of the chain view (levels between pings preserved)",
                                   "ISFT": "gpusonar viz/chain.py"})
            files.append({"ping": i, "channel": c, "file": fn, "bytes": w["bytes"]})
    return {"dir": f"{name}.chain_audio", "method": plan["method"], "method_words": words, "files": files,
            "time_scale": plan["time_scale"], "record_start_s": [float(r.samples.start_time) for r in records],
            "gain_db": round(20.0 * math.log10(g), 3), "bytes": sum(f["bytes"] for f in files)}


# --- the 2-D chain page --------------------------------------------------------------------------------------------------

def draw_chain_png(member: dict, path: str, ping: int | None = None, title: str = "") -> dict:
    """A 2-D chain page (matplotlib): the bearing-range map of one ping (default the last) with its detections, the
    tracks' history in bearing and range, the truth; the plan view of every ping's tracks against the truth; a track
    table (id, kind, rule, position, speed, mean snr, seen, nearest truth, error)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pings = member["pings"]
    if not pings:
        raise ValueError("no pings in the chain member")
    i = len(pings) - 1 if ping is None else ping
    P = pings[i]
    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.62], width_ratios=[1.25, 1.0, 0.9])
    ax = fig.add_subplot(gs[0, 0])
    g = P["geometry"]
    org = np.asarray(P["ownship"][:2])
    ax_v, bs_v = np.asarray(g["axis"]), np.asarray(g["broadside"])

    def rb(xy):           # world (north, east) -> (range, bearing deg) about this ping's array
        d = np.asarray(xy) - org
        return float(np.hypot(*d)), float(np.degrees(np.arctan2(d @ ax_v, d @ bs_v)))
    m = P.get("map")
    if m:
        img = np.frombuffer(base64.b64decode(m["u8"]), dtype=np.uint8).reshape(m["rows"], m["cols"]).astype(float)
        img = m["lo_db"] + img / 255.0 * (m["hi_db"] - m["lo_db"])
        ext = [m["r0"], m["r0"] + m["dr"] * m["cols"], m["b0"], m["b0"] + m["db"] * m["rows"]]
        im = ax.imshow(img, extent=ext, origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=30)
        fig.colorbar(im, ax=ax, label="dB over the map's median", shrink=0.8)
    for d in P["detections"]:
        ax.plot(d["range_m"], d["bearing_deg"], "x", color="#eb6834", ms=7, mew=1.6)
    for t in P["tracks"]:
        hist = [rb([q["x"], q["y"]]) for pp in pings[:i + 1] for q in pp["tracks"] if q["id"] == t["id"]]
        if hist:
            h = np.array(hist)
            ax.plot(h[:, 0], h[:, 1], "-", color="#2a78d6", lw=1.4)
            ax.text(h[-1, 0], h[-1, 1], f" {t['id']} {t['kind']}", color="#2a78d6", fontsize=7)
    for o in member["truth"]:
        q = o["positions"][min(i, len(o["positions"]) - 1)]
        r_, b_ = rb(q[:2])
        ax.plot(r_, b_, "o", mfc="none", mec="#1baf7a", ms=12, mew=1.5)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("bearing from broadside (deg)")
    ax.set_title(f"ping {i}: map, detections (x), tracks' history (blue), truth (green circles)", fontsize=9)
    ax2 = fig.add_subplot(gs[0, 1])
    own = np.array([pp["ownship"][:2] for pp in pings])
    ax2.plot(own[:, 1], own[:, 0], "k.-", lw=0.8, ms=3, label="own ship")
    ids = sorted({t["id"] for pp in pings for t in pp["tracks"]})
    for tid in ids:
        h = np.array([[q["x"], q["y"]] for pp in pings for q in pp["tracks"] if q["id"] == tid])
        ax2.plot(h[:, 1], h[:, 0], "-", color="#2a78d6", lw=1.2)
        ax2.plot(h[-1, 1], h[-1, 0], ".", color="#2a78d6")
    trk = P.get("tracker")
    if trk:
        for t in trk["after"]:
            nn, ne, ee = t["cov"]
            w, v = np.linalg.eigh(np.array([[ee, ne], [ne, nn]]))
            a = np.linspace(0, 2 * np.pi, 48)
            e = (v @ (2.0 * np.sqrt(np.maximum(w, 0))[:, None] * np.array([np.cos(a), np.sin(a)])))
            ax2.plot(t["pos"][1] + e[0], t["pos"][0] + e[1], "-", color="#8e5cc4" if t["confirmed"] else "#bbbbbb", lw=0.8)
    for o in member["truth"]:
        q = np.array(o["positions"])
        ax2.plot(q[:, 1], q[:, 0], "--", color="#1baf7a", lw=1)
        ax2.annotate(o["label"], (q[-1, 1], q[-1, 0]), fontsize=7, color="#1baf7a")
    ax2.set_xlabel("east (m)")
    ax2.set_ylabel("north (m)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title("plan: tracks over the pings (blue), 2-sigma covariance at this ping (purple: confirmed), truth (green)", fontsize=8)
    ax3 = fig.add_subplot(gs[0, 2])
    if m and m.get("background_u8"):
        row = int(np.argmin([abs(m["b0"] + (k + 0.5) * m["db"]) for k in range(m["rows"])]))
        dets = P["detections"]
        if dets:
            best = max(dets, key=lambda d: d["snr_db"])
            row = min(max(int((best["bearing_deg"] - m["b0"]) / m["db"]), 0), m["rows"] - 1)
        img = np.frombuffer(base64.b64decode(m["u8"]), dtype=np.uint8).reshape(m["rows"], m["cols"]).astype(float)
        bg = np.frombuffer(base64.b64decode(m["background_u8"]), dtype=np.uint8).reshape(m["rows"], m["cols"]).astype(float)
        sc = lambda u: m["lo_db"] + u / 255.0 * (m["hi_db"] - m["lo_db"])
        r = m["r0"] + m["dr"] * (np.arange(m["cols"]) + 0.5)
        ax3.plot(r, sc(img[row]), color="0.3", lw=0.8, label="map (block max)")
        ax3.plot(r, sc(bg[row]), color="#2a78d6", lw=1.2, label="detector background")
        for th in m.get("thresholds") or []:
            ax3.plot(r, sc(bg[row]) + th["factor_db"], lw=0.8, ls="--",
                     label=f"background x factor, {'one sample' if th['cells'] == 0 else str(th['cells']) + ' cells'}")
        ax3.legend(fontsize=6, loc="upper right")
        ax3.set_title(f"detector along bearing {m['b0'] + (row + 0.5) * m['db']:.1f} deg (the strongest detection's)", fontsize=8)
        ax3.set_xlabel("range (m)")
        ax3.set_ylabel("dB over the map's median")
    else:
        ax3.text(0.05, 0.5, "no detector tap on this ping", transform=ax3.transAxes)
    axt = fig.add_subplot(gs[1, :])
    axt.axis("off")
    ev = {e["id"]: e for e in P.get("evidence") or []}
    asc = {a["id"]: a for a in P.get("association") or []}
    rows = [["id", "kind", "rule (later)", "x north", "y east", "speed m/s", "need x sigma", "seen", "mean snr", "truth", "error m"]]
    for t in P["tracks"][:18]:
        e, a = ev.get(t["id"], {}), asc.get(t["id"], {})
        rows.append([t["id"], t["kind"], (e.get("rule", "") + (" (" + ", ".join(e["later"]) + ")" if e.get("later") else "")),
                     f"{t['x']:.1f}", f"{t['y']:.1f}", f"{t['speed_m_s']:.2f}",
                     "" if not e else f"{e['need'] * e['speed_sigma']:.2f}", f"{t['seen']:.2f}",
                     "" if t.get("mean_snr_db") is None else f"{t['mean_snr_db']:.1f}", a.get("truth") or "",
                     "" if a.get("error_m") is None else f"{a['error_m']:.1f}"])
    tb = axt.table(cellText=rows, loc="upper left", cellLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1.0, 1.05)
    more = len(P["tracks"]) - 18
    import textwrap
    head = ((title or "") + f" — live chain ({member['engine']}), ping {i} of {len(pings)}: {len(P['detections'])} detections, "
            f"{len(P['tracks'])} tracks" + (f" (table: first 18 of {len(P['tracks'])})" if more > 0 else ""))
    conf = f"configuration: {member['config_source']}" + (f"; {member['quick']}" if member.get("quick") else "")
    fig.suptitle(head + "\n" + "\n".join(textwrap.wrap(conf, 190)), fontsize=8.5)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return {"bytes": os.path.getsize(path), "ping": i}
