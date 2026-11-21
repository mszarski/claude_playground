"""tools/classify_rules_eval.py scoring on a made-up case (no simulation): known tracks, known truth.

The scene: a boat moving at 4 m/s along +x whose detections are 6 cells, an anchored boat (static, 20 cells), and a
glint that the tracker joined into a straight line far from every moving object (2 cells, a false mover). With the
rules off the boat is a mover and the anchored boat static: both correct, one false mover. min_mover_cells 3 to 6
removes the glint and keeps the boat; 8 also makes the boat uncertain, which is a target lost."""
import importlib.util
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("classify_rules_eval", ROOT / "tools" / "classify_rules_eval.py")
cre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cre)

T = np.arange(8, dtype=float)              # eight pings, 1 s apart


def _track(i, x0, y0, vx, vy, cells, snr=20.0):
    # The tracker's frame (classify_tracks and the gallery's scoring): history (t, a, b) with a = world y, b = world x,
    # state [a, b, ...] (tools/gallery.py world_tracks). Tiny alternating jitter so the fit has a scatter.
    jit = 0.05 * (-1.0) ** np.arange(len(T))
    xs, ys = x0 + vx * T + jit, y0 + vy * T - jit
    hist = [[float(t), float(y), float(x)] for t, x, y in zip(T, xs, ys)]
    return {"id": i, "state": [float(ys[-1]), float(xs[-1]), vy, vx], "hits": len(T), "history": hist,
            "snr": [snr] * len(T), "cells": [cells] * len(T)}


def _capture():
    truth = {"boat": {"position": [100.0, 0.0, 0.0], "velocity": [4.0, 0.0, 0.0]},
             "anchored": {"position": [150.0, 50.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
             "far_buoy": {"position": [300.0, -200.0, 0.0], "velocity": [0.0, 0.0, 0.0]}}   # never tracked
    tracks = [_track(1, 100.0, 0.0, 4.0, 0.0, cells=6), _track(2, 150.0, 50.0, 0.0, 0.0, cells=20),
              _track(3, 200.0, -80.0, 0.0, 3.0, cells=2)]
    return {"tracks": tracks, "truth": truth, "t_last": float(T[-1]), "gate": 15.0,
            "classify_kw": {"min_speed": 0.5, "max_fit_rms": 2.5}, "seed": 1}


def test_scoring_of_known_tracks():
    out = cre.evaluate_capture(_capture())
    base = out["cells=off,row=off"]
    assert base["targets"]["boat"] == {"tracked": True, "kind": "mover", "moving": True, "correct": True}
    assert base["targets"]["anchored"]["kind"] == "static" and base["targets"]["anchored"]["correct"]
    assert base["targets"]["far_buoy"] == {"tracked": False, "kind": None, "moving": False, "correct": False}
    assert base["false_movers"] == 1 and base["lost"] == [] and base["false_movers_removed"] == 0
    for m in (3, 4, 5, 6):
        s = out[f"cells={m},row=off"]
        assert s["false_movers"] == 0 and s["false_movers_removed"] == 1 and s["lost"] == [], m
    s = out["cells=8,row=off"]
    assert s["lost"] == ["boat"] and s["targets"]["boat"]["kind"] == "uncertain" and s["false_movers_removed"] == 1
    # No row test in this capture (no `linear`): row_snr_db changes nothing.
    for m in ("off", 3, 8):
        assert out[f"cells={m},row=6"]["targets"] == out[f"cells={m},row=off"]["targets"]


def test_aggregate_over_seeds():
    one = cre.evaluate_capture(_capture())
    cap2 = _capture()
    cap2["tracks"][0]["cells"] = [3] * len(T)          # on this seed the boat is small: lost from 4 cells
    two = cre.evaluate_capture(cap2)
    agg = cre.aggregate({"made_up": {1: one, 2: two}})
    c5 = agg["scenes"]["made_up"]["cells"]["cells=5,row=off"]
    assert c5["targets_lost_total"] == 1 and c5["seeds_with_a_loss"] == 1 and c5["lost"] == {"2": ["boat"]}
    assert c5["false_movers_removed_total"] == 2 and c5["false_movers"]["mean"] == 0.0
    c3 = agg["pooled"]["cells=3,row=off"]
    assert c3["targets_lost_total"] == 0 and c3["false_movers_removed_total"] == 2
    assert agg["pooled"]["cells=off,row=off"]["false_movers"] == {"mean": 1.0, "sd": 0.0, "min": 1.0, "max": 1.0}
    assert agg["verdict"]["baseline_false_movers_total"] == 2
    assert "cells=3,row=off" in agg["verdict"]["of_those_removing_false_movers"]
    assert "cells=4,row=off" not in agg["verdict"]["cells_losing_no_target_anywhere"]
    assert "made_up" in cre.table(agg)


def test_tracks_round_trip_through_json():
    from gpusonar import detect
    tr = detect.Track(id=7, state=np.array([1.0, 2.0, 0.1, 0.2]), cov=np.eye(4), hits=3,
                      history=[(0.0, 1.0, 2.0)], snr=[15.5], cells=[4])
    back = cre.track_from_json(cre.track_to_json(tr))
    assert back.id == 7 and back.hits == 3 and back.cells == [4] and back.snr == [15.5]
    assert back.history == [(0.0, 1.0, 2.0)] and np.array_equal(back.state, tr.state)


def test_compact_keeps_the_scores():
    out = cre.evaluate_capture(_capture())
    c = cre.compact(out)
    assert c["cells=8,row=off"]["kinds"] == {"boat": "uncertain", "anchored": "static", "far_buoy": None}
    assert c["cells=8,row=off"]["lost"] == ["boat"] and c["cells=off,row=off"]["false_movers"] == 1
