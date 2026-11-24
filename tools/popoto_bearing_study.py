#!/usr/bin/env python3
"""The Popoto gallery scenes' bearing estimate, offline from captures that recorded the detector's bearing profiles
(sonar-4s2): tools/classify_rules_eval.py --profiles (on a GPU: GPUSONAR_JOB=clseval with that flag).

Each capture holds, per ping, every hit of detect.cluster with the bearing profile of the map summed over its range
cells (`cluster_profiles`) and the array's pose for gallery.echo_position. Here each hit's bearing is estimated again by

  centroid   detect.cluster's own (the power-weighted mean over its flagged cells): the gallery's default
  parabola   the vertex of the parabola through the profile in dB about its strongest flagged row
             (gallery.parabola_vertex; what gallery.subgrid_bearings computes from the map itself)
  live       the live chain's multiscale rule (detect.multiscale): the mean over the flagged rows weighted by the energy
             above max(half the strongest row's, the outermost rows'); on one or two rows the plain energy-weighted mean
  grid       the strongest flagged row's grid bearing (a reference: no sub-grid estimate at all)

and the gallery's pipeline after the detector is run again: the SNR floor (9 dB), echo_position, the tracker with the
capture's own settings, classify and the gallery's scoring (classify_rules_eval.evaluate_capture). With `centroid` this
reproduces the capture's tracks (to 1e-11 m). Two tables: per hit near the true object (within 200 m in range and 3 deg
in bearing of its centre, any arrival), the bearing error from the centre by SNR and by the hit's flagged rows; and per
scene the scored track's report error over the seeds.

Usage: tools/popoto_bearing_study.py --captures DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("centroid", "parabola", "live", "grid")


def _tools():
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "tools"))
    import gallery
    spec = importlib.util.spec_from_file_location("classify_rules_eval_study", ROOT / "tools" / "classify_rules_eval.py")
    cre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cre)
    return gallery, cre


def estimate(hit: dict, first: float, step: float, method: str) -> float:
    """A hit's bearing (rad) by `method` from its recorded profile (classify_rules_eval.cluster_profiles); the map's
    bearings are first + i * step."""
    if method == "centroid":
        return float(hit["bearing"])
    prof = np.asarray(hit["sum"], dtype=float)                     # dB, rows lo, lo + 1, ...
    lo, rows = int(hit["lo"]), np.asarray(hit["rows"], dtype=int)
    if method == "live":
        e = 10.0 ** (prof[rows - lo] / 10.0)
        soft = np.maximum(e - max(0.5 * e.max(), e[0], e[-1]), 0.0)
        if soft.sum() <= 0.0:
            soft = e
        return float(first + (soft * rows).sum() / soft.sum() * step)
    j = int(rows[np.argmax(prof[rows - lo])]) - lo
    if method == "grid":
        return float(first + (lo + j) * step)
    if method != "parabola":
        raise ValueError(f"method {method!r}: one of {METHODS}")
    if j == 0 or j == len(prof) - 1:
        return float(hit["bearing"])                               # at the edge of the recorded rows: as the gallery
    gallery, _ = _tools()
    return float(first + (lo + j + gallery.parabola_vertex(*prof[j - 1:j + 2])) * step)


def replay(cap: dict, method: str, snr_floor: float = 9.0) -> dict:
    """The capture with its tracks made again from its recorded hits, each at `method`'s bearing (gallery
    run_popoto_far after the detector)."""
    gallery, cre = _tools()
    from gpusonar import detect
    tracker, tracks = detect.Tracker(**cap["tracker_kw"]), []
    for p in cap["profiles"]:
        first, step, _ = p["bearings"]
        dets = []
        for h in p["hits"]:
            if not h["snr_db"] > snr_floor:
                continue
            e = p["echo"]
            x, y = gallery.echo_position(e["origin"], e["velocity"], h["range"], estimate(h, first, step, method), e["c"])
            dets.append(detect.Detection(float(np.hypot(x, y)), float(np.arctan2(y, x)), h["snr_db"], int(h["cells"]),
                                         time=p["time"] + h["range"] / e["c"]))
        tracks = tracker.update(p["time"], dets)
    return {**cap, "tracks": [cre.track_to_json(t) for t in tracks]}


def hit_errors(cap: dict, methods=METHODS) -> list[tuple]:
    """(snr_db, flagged rows, [error deg per method]) for each hit within 200 m in range and 3 deg in bearing of the
    true object's centre, seen from the array where it is when the echo arrives (gallery.echo_position)."""
    out = []
    for label, obj in cap["truth"].items():
        cen, v = np.asarray(obj["position"][:2], dtype=float), np.asarray(obj["velocity"][:2], dtype=float)
        for p in cap["profiles"]:
            e = p["echo"]
            if e is None:
                continue
            first, step, _ = p["bearings"]
            for h in p["hits"]:
                if not h["snr_db"] > 9.0:
                    continue
                rx = np.asarray(e["origin"]) + np.asarray(e["velocity"]) * 2.0 * h["range"] / e["c"]
                q = cen + v * (p["time"] + h["range"] / e["c"])
                b = np.arctan2(q[0] - rx[0], q[1] - rx[1])
                if abs(h["range"] - np.hypot(*(q - rx))) > 200.0 or abs(h["bearing"] - b) > np.radians(3.0):
                    continue
                out.append((h["snr_db"], len(h["rows"]), [float(np.degrees(estimate(h, first, step, m) - b)) for m in methods]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True, help="directory of classify_rules_eval --profiles captures")
    ap.add_argument("--json", default=None, help="write the per-capture scores of every method here")
    args = ap.parse_args(argv)
    _, cre = _tools()
    caps = [json.loads(f.read_text()) for f in sorted(Path(args.captures).glob("*.json"))]
    caps = [c for c in caps if c.get("profiles")]
    if not caps:
        raise SystemExit(f"no --profiles captures in {args.captures}")
    scores, hits = {}, {}
    for cap in caps:
        hits.setdefault(cap["scene"], []).extend(hit_errors(cap))
        for m in METHODS:
            scores.setdefault(m, {}).setdefault(cap["scene"], {})[str(cap["seed"])] = cre.compact(cre.evaluate_capture(replay(cap, m)))
    print("Per hit near the object, bearing error from its centre (deg): RMS by SNR (dB) and by flagged bearing rows")
    for scene, rows in sorted(hits.items()):
        print(f"{scene} ({len(rows)} hits)")
        for name, key, bins in (("snr", 0, [(9, 11.5), (11.5, 12.5), (12.5, 14), (14, 17), (17, 99)]),
                                ("rows", 1, [(1, 3), (3, 5), (5, 999)])):
            for lo, hi in bins:
                sel = np.array([r[2] for r in rows if lo <= r[key] < hi])
                if len(sel):
                    print(f"  {name} {lo}-{hi}: n {len(sel):4d}  " +
                          "  ".join(f"{m} {np.sqrt(np.mean(sel[:, i] ** 2)):.3f}" for i, m in enumerate(METHODS)))
    print("\nPer scene, the scored track's report at the last ping over the seeds (rules off): distance to the hull "
          "median / mean, cross-range from the centre mean / mean |.| / worst, speed error mean (m, m/s)")
    for scene in sorted(scores["centroid"]):
        print(scene)
        for m in METHODS:
            e, sp = [], []
            for cap in (c for c in caps if c["scene"] == scene):
                for label, x in scores[m][scene][str(cap["seed"])]["cells=off,row=off"].get("errors", {}).items():
                    e.append(x)
                    sp.append(abs(x["speed_m_s"] - float(np.hypot(*cap["truth"][label]["velocity"][:2]))))
            if not e:
                print(f"  {m:9s} no target tracked")
                continue
            hull = np.array([x["report_error_m"] for x in e])
            xr = np.array([x["report_centroid_cross_range_error_m"] for x in e])
            sp = np.array(sp)
            print(f"  {m:9s} hull {np.median(hull):5.1f} / {hull.mean():5.1f}  xr {xr.mean():+6.1f} / {np.abs(xr).mean():5.1f} / "
                  f"{xr[np.argmax(np.abs(xr))]:+6.1f}  speed {sp.mean():.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"tool": "tools/popoto_bearing_study.py", "methods": list(METHODS),
                                               "per_capture": scores}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
