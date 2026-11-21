"""tools/classify_rules_eval.py scoring on a made-up case (no simulation): known tracks, known truth.

The scene: a boat moving at 4 m/s along +x whose detections are 6 cells, an anchored boat (static, 20 cells), and a
glint that the tracker joined into a straight line far from every moving object (2 cells, a false mover). With the
rules off the boat is a mover and the anchored boat static: both correct, one false mover. min_mover_cells 3 to 6
removes the glint and keeps the boat; 8 also makes the boat uncertain, which is a target lost."""
import importlib.util
import pathlib

import numpy as np
import pytest

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
    boat = base["targets"]["boat"]
    assert {k: boat[k] for k in ("tracked", "kind", "moving", "correct")} == {"tracked": True, "kind": "mover", "moving": True, "correct": True}
    # Its errors (D-156): the state against the boat at the state's time (the last point, t = 7), the classified
    # position at the last ping against the boat there; no ownship in this capture, so no range / cross-range split.
    assert boat["state_time_s"] == 7.0 and boat["state_error_m"] < 0.1 and boat["report_error_m"] < 0.1
    assert boat["speed_m_s"] == pytest.approx(4.0, abs=0.05) and "state_range_error_m" not in boat
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


def test_errors_are_split_along_and_across_the_line_of_sight_at_the_states_time():
    """D-156: a track whose last point is its detection's time (8 s after the last ping) is compared with the object
    then; its classified position (at the last ping, as the live report gives it) with the object at the last ping.
    The ownship (due west of the ship at the state's time) comes from the capture's echo-grouping origin (tracker
    order: y, x). Cross range + is counter-clockwise
    seen from above (x north, y east)."""
    cap = _capture()
    cap["truth"] = {"ship": {"position": [800.0, 12000.0, 0.0], "velocity": [-6.0, 0.0, 0.0]}}
    hist = [[float(t + 8.0), 12000.0, float(800.0 - 6.0 * (t + 8.0) + 0.05 * (-1) ** i)] for i, t in enumerate(T)]
    # The state 10 m south of the ship (ahead along its course, to its right seen from the ownship looking east) and
    # 3 m beyond it.
    last = hist[-1]
    cap["tracks"] = [{"id": 1, "state": [last[1] + 3.0, last[2] - 10.0, 0.0, -6.0], "hits": len(T), "history": hist,
                      "snr": [20.0] * len(T), "cells": [6] * len(T)}]
    cap["classify_kw"] = {"min_speed": 0.5, "max_fit_rms": 2.5, "multipath": {"max_excess_range": 800.0, "bearing": 0.05,
                                                                              "velocity": 2.5, "origin": (0.0, 710.0)}}
    cap["gate"] = 200.0
    ship = cre.evaluate_capture(cap)["cells=off,row=off"]["targets"]["ship"]
    assert ship["kind"] == "mover" and ship["state_time_s"] == T[-1] + 8.0
    assert ship["state_range_error_m"] == pytest.approx(3.0, abs=0.5) and ship["state_cross_range_error_m"] == pytest.approx(-10.0, abs=0.5)
    # The fitted line read at the last ping is where the ship is then, within the jitter.
    assert ship["report_error_m"] < 0.2 and ship["speed_m_s"] == pytest.approx(6.0, abs=0.05)


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


def test_a_capture_that_carries_the_gallerys_own_rules_is_scored_from_off():
    """Since D-154 the gallery classifies with detections={"row_snr_db": 6}; a capture that recorded those options
    must still be scored over the grid starting from the rules off, the same as one that did not."""
    plain = cre.evaluate_capture(_capture())
    cap = _capture()
    cap["classify_kw"] = dict(cap["classify_kw"], detections={"row_snr_db": 6.0})
    assert cre.evaluate_capture(cap) == plain


def test_an_established_track_is_scored_before_a_nearer_uncertain_one():
    """D-155: a true object's track is the nearest track of an established kind within the gate, else the nearest of
    any kind. Before, the nearest of any kind: here a three-point track 2 m from the boat (uncertain), not the mover 8 m
    from it. The tool scores exactly as the gallery does (it calls gallery.object_track and count_false_movers)."""
    import sys
    from gpusonar import classify
    sys.path.insert(0, str(ROOT / "tools"))
    import gallery
    truth = cre.truth_from_json({"boat": {"position": [100.0, 0.0, 0.0], "velocity": [4.0, 0.0, 0.0]}})
    mover = _track(1, 100.0, 8.0, 4.0, 0.0, cells=6)
    short = _track(2, 100.0, 2.0, 4.0, 0.0, cells=6)
    short.update(history=short["history"][-3:], hits=3, snr=short["snr"][-3:], cells=short["cells"][-3:])
    kw = {"min_speed": 0.5, "max_fit_rms": 2.5}
    for tracks, want in (([mover, short], "mover"), ([short], "uncertain")):
        tr = [cre.track_from_json(t) for t in tracks]
        classes = classify.classify_tracks(tr, **kw)
        assert {c.id: c.kind for c in classes}.get(2) == "uncertain"
        s = cre.score(classes, tr, truth, float(T[-1]), 15.0)
        assert s["targets"]["boat"]["kind"] == want and s["targets"]["boat"]["correct"] == (want == "mover")
        _, g = gallery.classify_and_score(tr, truth, float(T[-1]), 15.0, **kw)
        assert g["targets"]["boat"]["kind"] == want and g["movers_not_on_a_moving_object"] == s["false_movers"] == 0
        # The gallery's position and velocity per target come from the same track as its class (score_tracks).
        pos = gallery.score_tracks(tr, truth, float(T[-1]), 15.0, kinds={c.id: c.kind for c in classes})["boat"]
        assert pos["tracked"] and abs(pos["position_error_m"] - (8.0 if want == "mover" else 2.0)) < 0.2
    # Nothing within the gate: not tracked, in both.
    tr = [cre.track_from_json(_track(1, 100.0, 40.0, 4.0, 0.0, cells=6))]
    classes = classify.classify_tracks(tr, **kw)
    assert cre.score(classes, tr, truth, float(T[-1]), 15.0)["targets"]["boat"]["tracked"] is False
    assert gallery.classify_and_score(tr, truth, float(T[-1]), 15.0, **kw)[1]["targets"] == {}
