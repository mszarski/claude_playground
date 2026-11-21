#!/usr/bin/env python3
"""Multi-seed evaluation of the opt-in classification rules from the detections (DECISIONS D-152):
classify.detections.min_mover_cells and row_snr_db, against each gallery scene's truth.

For every scene and seed the scene is simulated and tracked once, exactly as tools/gallery.py runs it, except that
the scenario's `seed` (the one generator of its scene draws, clutter and receiver noise: sim/simulator.py
run_scenario) is the scene's own seed plus k * --seed-step for k = 0 .. --seeds - 1. k = 0 is the gallery's own run.
The tracks are kept with each detection's real cell count: the gallery's `world_tracks` builds its detections with
cells = 1, so here the detector's own hits (`detect.cluster`) are recorded and each tracked detection takes the cell
count of its hit (matched on the ping, in order, by SNR). Then the tracks are classified offline over the grid

    min_mover_cells {off, 3, 4, 5, 6, 8} x row_snr_db {off, 3, 6}

with every other classify option the gallery's own for that scene. row_snr_db acts only where the gallery runs the row
test (the FLS transit scenes); in the Popoto scenes its columns equal "off".

Scoring (the gallery's own, gallery.classify_and_score): a true object is the scene item's centre moved by its
velocity to the last ping; its track is the confirmed track nearest to that point, if within the gallery's gate for
the scene (15 m in the FLS transit scenes, 200 m in the Popoto scenes; set in tools/gallery.py, not in the scene
files). A target is correctly classed when its track is `mover` if the object moves and `static` if not. A false
mover is a track classed `mover` that is no closer than the gate to every moving object. Against the grid's (off, off)
cell of the same capture: "lost" counts targets correct there and not correct with the rule; "gained" the reverse;
"false movers removed" is the baseline's false movers minus the rule's.

Usage:
  tools/classify_rules_eval.py [--backend cuda|host] [--scenes a,b] [--seeds 5] [--seed-step 1]
                               [--captures DIR] [--json OUT] [--table OUT] [--quick] [--commit HASH]
  tools/classify_rules_eval.py --evaluate-only --captures DIR --json OUT     re-score saved captures, no simulation

Captures are one JSON file per (scene, seed) in --captures (default build/classify_rules_eval/captures under the
repository); one already there is reused unless --recapture. On a GPU machine: tools/gpu_job.sh with
GPUSONAR_JOB=clseval (and GPUSONAR_ORACLES=1 for the Popoto scenes).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Gallery scenes with a moving object or a false mover in the committed gallery (docs/figures/gallery.json), or with
# a moving object in the scene file (popoto_env_summer_mud_60m: no confirmed track at its own seed).
SCENES = ("fls_120k_cluttered_seabed", "fls_120k_rough_surface", "fls_170k_traffic", "fls_120k_transit",
          "fls_170k_transit", "fls_330k_transit", "popoto_12km", "popoto_far_mover", "popoto_env_winter_sand_200m",
          "popoto_env_gravel_30m", "popoto_env_summer_mud_60m", "popoto_env_deep_silt_1000m")
MIN_MOVER_CELLS = (None, 3, 4, 5, 6, 8)
ROW_SNR_DB = (None, 3, 6)


def grid():
    """The rule settings, (off, off) first: [(key, detections option)]."""
    out = []
    for m in MIN_MOVER_CELLS:
        for r in ROW_SNR_DB:
            key = f"cells={'off' if m is None else m},row={'off' if r is None else r}"
            out.append((key, {"min_mover_cells": m, "row_snr_db": r}))
    return out


# --- tracks to and from JSON -------------------------------------------------------------------

def track_to_json(tr) -> dict:
    return {"id": int(tr.id), "state": [float(v) for v in tr.state], "hits": int(tr.hits),
            "history": [[float(a) for a in h] for h in tr.history], "snr": [float(s) for s in tr.snr],
            "cells": [int(c) for c in tr.cells]}


def track_from_json(d: dict):
    """What classify_tracks and the scoring read of a track."""
    return SimpleNamespace(id=d["id"], state=np.asarray(d["state"], dtype=float), hits=d["hits"],
                           history=[tuple(h) for h in d["history"]], snr=list(d["snr"]), cells=list(d["cells"]))


def truth_to_json(truth: dict) -> dict:
    return {k: {"position": [float(v) for v in pos], "velocity": [float(v) for v in vel]} for k, (pos, vel) in truth.items()}


def truth_from_json(d: dict) -> dict:
    return {k: (np.asarray(v["position"], dtype=float), np.asarray(v["velocity"], dtype=float)) for k, v in d.items()}


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# --- scoring -----------------------------------------------------------------------------------

def score(classes, tracks, truth: dict, t_last: float, gate: float) -> dict:
    """The gallery's scoring (gallery.classify_and_score) of one classification: per true object whether a track is
    within `gate` of where it is at the last ping, the kind of that track and whether it is the right one; and the
    number of movers no closer than `gate` to any moving object."""
    by_id = {c.id: c for c in classes}
    targets, moving_ends = {}, []
    for label, (pos, vel) in truth.items():
        end = np.asarray(pos)[:2] + np.asarray(vel)[:2] * t_last
        moving = bool(np.hypot(*np.asarray(vel)[:2]) > 0.0)
        if moving:
            moving_ends.append(end)
        best = min(tracks, key=lambda tr: np.hypot(tr.state[1] - end[0], tr.state[0] - end[1]), default=None)
        if best is not None and np.hypot(best.state[1] - end[0], best.state[0] - end[1]) < gate:
            kind = by_id[best.id].kind
            targets[label] = {"tracked": True, "kind": kind, "moving": moving,
                              "correct": kind == ("mover" if moving else "static")}
        else:
            targets[label] = {"tracked": False, "kind": None, "moving": moving, "correct": False}
    false = int(sum(c.kind == "mover" and all(np.hypot(c.position[1] - e[0], c.position[0] - e[1]) >= gate
                                              for e in moving_ends) for c in classes))
    return {"targets": targets, "false_movers": false}


def evaluate_capture(cap: dict) -> dict:
    """Every grid cell's classification of one capture, scored, with lost / gained / removed against (off, off)."""
    from gpusonar import classify
    tracks = [track_from_json(t) for t in cap["tracks"]]
    truth = truth_from_json(cap["truth"])
    out, base = {}, None
    for key, rule in grid():
        classes = classify.classify_tracks(tracks, **cap["classify_kw"], detections=rule)
        s = score(classes, tracks, truth, cap["t_last"], cap["gate"])
        s["counts"] = classify.summary(classes)
        if base is None:
            base = s
        s["lost"] = sorted(k for k, t in s["targets"].items() if base["targets"][k]["correct"] and not t["correct"])
        s["gained"] = sorted(k for k, t in s["targets"].items() if t["correct"] and not base["targets"][k]["correct"])
        s["false_movers_removed"] = base["false_movers"] - s["false_movers"]
        out[key] = s
    return out


def compact(evaluated: dict) -> dict:
    """One capture's evaluate_capture result for the JSON: per grid cell the kind of each true object's track (None:
    not tracked), the false movers, lost, gained and the counts by kind."""
    return {key: {"kinds": {lab: t["kind"] for lab, t in s["targets"].items()}, "false_movers": s["false_movers"],
                  "lost": s["lost"], "gained": s["gained"], "counts": s["counts"]} for key, s in evaluated.items()}


def _spread(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def aggregate(per_capture: dict) -> dict:
    """per_capture: {scene: {seed: evaluate_capture(...)}} -> per scene and pooled, per grid cell, over seeds."""
    keys = [k for k, _ in grid()]
    scenes = {}
    for scene, by_seed in per_capture.items():
        seeds = sorted(by_seed)
        cells = {}
        for key in keys:
            rows = [by_seed[s][key] for s in seeds]
            cells[key] = {
                "false_movers": _spread([r["false_movers"] for r in rows]),
                "false_movers_removed": _spread([r["false_movers_removed"] for r in rows]),
                "false_movers_removed_total": int(sum(r["false_movers_removed"] for r in rows)),
                "targets_lost_total": int(sum(len(r["lost"]) for r in rows)),
                "targets_gained_total": int(sum(len(r["gained"]) for r in rows)),
                "seeds_with_a_loss": int(sum(bool(r["lost"]) for r in rows)),
                "lost": {str(s): by_seed[s][key]["lost"] for s in seeds if by_seed[s][key]["lost"]},
                "gained": {str(s): by_seed[s][key]["gained"] for s in seeds if by_seed[s][key]["gained"]},
            }
        labels = sorted(by_seed[seeds[0]][keys[0]]["targets"])
        baseline = {lab: {"tracked_seeds": int(sum(by_seed[s][keys[0]]["targets"][lab]["tracked"] for s in seeds)),
                          "correct_seeds": int(sum(by_seed[s][keys[0]]["targets"][lab]["correct"] for s in seeds)),
                          "moving": by_seed[seeds[0]][keys[0]]["targets"][lab]["moving"],
                          "kinds": [by_seed[s][keys[0]]["targets"][lab]["kind"] for s in seeds]} for lab in labels}
        scenes[scene] = {"seeds": seeds, "baseline_targets": baseline, "cells": cells}
    pooled = {}
    n_seeds = min(len(v["seeds"]) for v in scenes.values()) if scenes else 0
    for key in keys:
        # Pooled per seed index (the k-th seed of every scene): the sum over scenes, then its spread over k.
        per_k_false = [sum(per_capture[sc][scenes[sc]["seeds"][k]][key]["false_movers"] for sc in scenes) for k in range(n_seeds)]
        per_k_removed = [sum(per_capture[sc][scenes[sc]["seeds"][k]][key]["false_movers_removed"] for sc in scenes) for k in range(n_seeds)]
        pooled[key] = {
            "false_movers": _spread(per_k_false) if n_seeds else None,
            "false_movers_removed": _spread(per_k_removed) if n_seeds else None,
            "false_movers_removed_total": int(sum(c["cells"][key]["false_movers_removed_total"] for c in scenes.values())),
            "targets_lost_total": int(sum(c["cells"][key]["targets_lost_total"] for c in scenes.values())),
            "targets_gained_total": int(sum(c["cells"][key]["targets_gained_total"] for c in scenes.values())),
            "captures_with_a_loss": int(sum(c["cells"][key]["seeds_with_a_loss"] for c in scenes.values())),
            "scenes_with_a_loss": sorted(sc for sc, c in scenes.items() if c["cells"][key]["targets_lost_total"]),
        }
    safe = [k for k in keys[1:] if pooled[k]["targets_lost_total"] == 0]
    verdict = {"cells_losing_no_target_anywhere": safe,
               "of_those_removing_false_movers": [k for k in safe if pooled[k]["false_movers_removed_total"] > 0],
               "baseline_false_movers_total": int(sum(r[keys[0]]["false_movers"] for d in per_capture.values() for r in d.values()))}
    return {"scenes": scenes, "pooled": pooled, "verdict": verdict}


def table(agg: dict) -> str:
    """Readable summary: pooled grid, then per scene the cells that change anything."""
    keys = [k for k, _ in grid()]
    lines = ["Pooled over scenes (false movers: per seed index the sum over scenes, mean +- sd over seeds).", "",
             "| rule | targets lost (captures) | targets gained | false movers | false movers removed (total) |",
             "| --- | --- | --- | --- | --- |"]
    for k in keys:
        p = agg["pooled"][k]
        fm, rm = p["false_movers"], p["false_movers_removed"]
        lines.append(f"| {k} | {p['targets_lost_total']} ({p['captures_with_a_loss']}) | {p['targets_gained_total']} | "
                     f"{fm['mean']:.1f} +- {fm['sd']:.1f} [{fm['min']:.0f}, {fm['max']:.0f}] | "
                     f"{rm['mean']:.1f} +- {rm['sd']:.1f} ({p['false_movers_removed_total']}) |")
    for scene, s in agg["scenes"].items():
        lines += ["", f"{scene} (seeds {s['seeds'][0]}..{s['seeds'][-1]}, {len(s['seeds'])}): baseline targets "
                  + "; ".join(f"{lab} {'moving' if b['moving'] else 'static'} tracked {b['tracked_seeds']} correct "
                              f"{b['correct_seeds']}" for lab, b in s["baseline_targets"].items()), ""]
        lines += ["| rule | lost | gained | false movers mean +- sd [min, max] | removed |", "| --- | --- | --- | --- | --- |"]
        for k in keys:
            c = s["cells"][k]
            if k != keys[0] and not (c["targets_lost_total"] or c["targets_gained_total"] or c["false_movers_removed_total"]
                                     or c["false_movers"]["max"] != s["cells"][keys[0]]["false_movers"]["max"]):
                continue
            fm = c["false_movers"]
            lost = "; ".join(f"seed {sd}: {','.join(v)}" for sd, v in c["lost"].items()) or "0"
            gained = "; ".join(f"seed {sd}: {','.join(v)}" for sd, v in c["gained"].items()) or "0"
            lines.append(f"| {k} | {lost} | {gained} | {fm['mean']:.1f} +- {fm['sd']:.1f} [{fm['min']:.0f}, {fm['max']:.0f}] | "
                         f"{c['false_movers_removed_total']} |")
    v = agg["verdict"]
    lines += ["", f"Cells losing no target in any capture: {', '.join(v['cells_losing_no_target_anywhere']) or 'none'}",
              f"Of those, removing false movers: {', '.join(v['of_those_removing_false_movers']) or 'none'}"]
    return "\n".join(lines) + "\n"


# --- capture (simulation) ------------------------------------------------------------------------

def capture(name: str, seed_offset: int, backend: str, quick: bool) -> dict:
    """Run one gallery scene with its seed moved by `seed_offset`; return its confirmed tracks (with real cell
    counts), truth, last ping time, gate and classify options, as the gallery's classify_and_score receives them."""
    sys.path.insert(0, str(ROOT / "tools"))
    import gallery
    import matplotlib.pyplot as plt
    from gpusonar import core, detect

    calls, got = [], {}

    class DetectProxy:                      # gallery calls detect.cluster through its module global
        def __getattr__(self, attr):
            return getattr(detect, attr)

        @staticmethod
        def cluster(*a, **k):
            hits = detect.cluster(*a, **k)
            calls.append(hits)
            return hits

    def world_tracks(per_ping, times, **kw):
        # The gallery's world_tracks, with each detection's cell count from the hit it came from.
        if len(calls) != len(per_ping):
            raise RuntimeError(f"{name}: {len(calls)} detector calls for {len(per_ping)} pings")
        tracker, tracks = detect.Tracker(**kw), []
        for t, hits, raw in zip(times, per_ping, calls):
            dets, j = [], 0
            for x, y, snr in hits:
                while j < len(raw) and raw[j].snr_db != snr:
                    j += 1
                if j == len(raw):
                    raise RuntimeError(f"{name}: a tracked detection matches no hit of its ping")
                dets.append(detect.Detection(float(np.hypot(x, y)), float(np.arctan2(y, x)), snr, int(raw[j].cells)))
                j += 1
            tracks = tracker.update(float(t), dets)
        return tracks

    orig_score = gallery.classify_and_score

    def classify_and_score(tracks, truth, t_last, gate, **kw):
        got.update(tracks=tracks, truth=truth, t_last=float(t_last), gate=float(gate), kw=kw)
        return orig_score(tracks, truth, t_last, gate, **kw)

    saved = gallery.detect, gallery.world_tracks, gallery.classify_and_score
    gallery.detect, gallery.world_tracks, gallery.classify_and_score = DetectProxy(), world_tracks, classify_and_score
    try:
        cfg = gallery.load_scenario_config(name)
        own_seed = int(cfg.get("seed", 0))
        cfg["seed"] = own_seed + int(seed_offset)
        if quick and name in gallery.QUICK:
            gallery.QUICK[name](cfg)
        t0 = time.perf_counter()
        be = core.Backend.Cuda if backend == "cuda" else core.Backend.Host
        out = gallery.RUNNERS[name](name, cfg, be, quick)
        plt.close(out[0])
        seconds = time.perf_counter() - t0
    finally:
        gallery.detect, gallery.world_tracks, gallery.classify_and_score = saved
    if not got:
        raise RuntimeError(f"{name}: the gallery runner did not classify")
    return {"scene": name, "scene_seed": own_seed, "seed": own_seed + int(seed_offset), "seed_offset": int(seed_offset),
            "backend": backend, "quick": bool(quick), "seconds": round(seconds, 1),
            "tracks": [track_to_json(t) for t in got["tracks"]], "truth": truth_to_json(got["truth"]),
            "t_last": got["t_last"], "gate": got["gate"], "classify_kw": _jsonable(got["kw"]),
            "gallery_classification": _jsonable(out[1].get("classification"))}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GPUSONAR_COMMIT", "unknown")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=None, help="cuda or host (default: cuda when available)")
    ap.add_argument("--scenes", default=",".join(SCENES), help="comma-separated gallery scenes (default: %(default)s)")
    ap.add_argument("--seeds", type=int, default=5, help="seeds per scene (default 5)")
    ap.add_argument("--seed-step", type=int, default=1, help="seed k is the scene's own seed + k * step (default 1)")
    ap.add_argument("--captures", default=str(ROOT / "build" / "classify_rules_eval" / "captures"),
                    help="directory of the per-(scene, seed) captures (default: %(default)s)")
    ap.add_argument("--recapture", action="store_true", help="simulate again even where a capture exists")
    ap.add_argument("--evaluate-only", action="store_true", help="no simulation: score the captures already there")
    ap.add_argument("--quick", action="store_true", help="the gallery's thinned scenes (a CPU smoke run, not the evaluation)")
    ap.add_argument("--json", default=None, help="write the result JSON here")
    ap.add_argument("--table", default=None, help="write the readable table (Markdown) here")
    ap.add_argument("--commit", default=None, help="the tree's commit, recorded in the JSON (default: git, else $GPUSONAR_COMMIT)")
    args = ap.parse_args(argv)
    names = [n for n in args.scenes.split(",") if n]
    cap_dir = Path(args.captures)
    commit = args.commit or _commit()
    per_capture, info = {}, {}
    if not args.evaluate_only:
        import gpusonar as gs
        from gpusonar.sim import find_bellhop
        backend = args.backend or ("cuda" if gs.cuda_available() else "host")
        cap_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            if name.startswith("popoto_") and find_bellhop() is None:
                raise SystemExit(f"{name} needs Bellhop (GPUSONAR_BELLHOP)")
            for k in range(args.seeds):
                path = cap_dir / f"{name}_k{k * args.seed_step}{'_quick' if args.quick else ''}.json"
                if path.exists() and not args.recapture:
                    continue
                cap = capture(name, k * args.seed_step, backend, args.quick)
                cap["commit"] = commit
                path.write_text(json.dumps(cap))
                print(f"[{name} seed {cap['seed']}] {len(cap['tracks'])} tracks, {cap['seconds']} s", flush=True)
    for name in names:
        files = sorted(cap_dir.glob(f"{name}_k*.json"))
        files = [f for f in files if f.stem.endswith("_quick") == args.quick]
        if not args.evaluate_only:
            want = {f"{name}_k{k * args.seed_step}{'_quick' if args.quick else ''}.json" for k in range(args.seeds)}
            files = [f for f in files if f.name in want]
        for f in files:
            cap = json.loads(f.read_text())
            per_capture.setdefault(name, {})[cap["seed"]] = evaluate_capture(cap)
            info.setdefault(name, {})[str(cap["seed"])] = {"tracks": len(cap["tracks"]), "seconds": cap["seconds"],
                                                           "backend": cap["backend"], "quick": cap["quick"],
                                                           "commit": cap.get("commit"), "gate_m": cap["gate"],
                                                           "classify_kw": cap["classify_kw"],
                                                           "gallery_classification": cap["gallery_classification"]}
    if not per_capture:
        raise SystemExit(f"no captures in {cap_dir}")
    agg = aggregate(per_capture)
    result = {"tool": "tools/classify_rules_eval.py", "command": "tools/classify_rules_eval.py " + " ".join(argv if argv is not None else sys.argv[1:]),
              "commit": commit, "grid": {k: r for k, r in grid()},
              "association": "a true object (scene item centre moved by its velocity to the last ping) takes the "
                             "confirmed track nearest to it if within the gallery's gate (gate_m); a false mover is a "
                             "mover no closer than the gate to every moving object (gallery.classify_and_score)",
              "captures": info, **agg,
              "per_capture": {sc: {str(sd): compact(v) for sd, v in d.items()} for sc, d in per_capture.items()}}
    text = table(agg)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, separators=(",", ":")) + "\n")
    if args.table:
        Path(args.table).parent.mkdir(parents=True, exist_ok=True)
        Path(args.table).write_text(text)


if __name__ == "__main__":
    main()
