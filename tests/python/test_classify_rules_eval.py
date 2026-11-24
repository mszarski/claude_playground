"""tools/classify_rules_eval.py scoring on a made-up case (no simulation): known tracks, known truth.

The scene: a boat moving at 4 m/s along +x whose detections are 6 cells, an anchored boat (static, 20 cells), and a
glint that the tracker joined into a straight line far from every moving object (2 cells, a false mover). With the
rules off the boat is a mover and the anchored boat static: both correct, one false mover. min_mover_cells 3 to 6
removes the glint and keeps the boat; 8 also makes the boat uncertain, which is a target lost."""
import importlib.util
import json
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
    # D-157: the measured rows go along, NaN (a miss) as null; a capture without them reads back with none.
    tr.history.append((1.0, 1.5, 2.5))
    tr.measured = [(1.0, 2.0), (float("nan"), float("nan"))]
    j = cre.track_to_json(tr)
    assert j["measured"] == [[1.0, 2.0], [None, None]]
    back = cre.track_from_json(json.loads(json.dumps(j)))
    assert back.measured[0] == (1.0, 2.0) and np.isnan(back.measured[1]).all()
    j.pop("measured")
    assert cre.track_from_json(j).measured == []


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


# --- D-157: an extended object is scored from its outline --------------------------------------------------------

SHIP = {"position": [800.0, 12000.0, 4.0], "velocity": [-6.0, 0.0, 0.0],
        "footprint": [[740.0, 12000.0], [860.0, 12000.0]]}           # popoto_12km's five points, 120 m along x


def _ship_track(i, dx, dy, vx=-6.0, n=6, dt=19.9):
    """A track of the ship moving with it, `dx` along and `dy` across its course from its centre (world x, y)."""
    t = np.arange(n) * dt
    xs, ys = 800.0 + dx + vx * t, 12000.0 + dy + 0.0 * t
    hist = [[float(a), float(y), float(x)] for a, x, y in zip(t, xs, ys)]
    return {"id": i, "state": [float(ys[-1]), float(xs[-1]), 0.0, vx], "hits": n, "history": hist,
            "snr": [20.0] * n, "cells": [6] * n}


def test_an_extended_object_is_scored_from_its_outline():
    """D-157: a track on the ship's bow, 55 m ahead of its centre and 5 m off its line, is 5 m from the ship (its
    outline, the segment through its five points), not 55 m; the centre's error is kept as centroid_error_m. Before,
    error_m was 55 m from the centre."""
    cap = {"tracks": [_ship_track(1, 55.0, 5.0)], "truth": {"ship": SHIP}, "t_last": 99.5, "gate": 200.0,
           "classify_kw": {"min_speed": 1.0, "static_speed": 1.0, "max_fit_rms": 60.0}, "seed": 1}
    s = cre.evaluate_capture(cap)["cells=off,row=off"]["targets"]["ship"]
    assert s["kind"] == "mover" and s["correct"]
    assert s["state_error_m"] == pytest.approx(5.0, abs=1e-6) and s["report_error_m"] == pytest.approx(5.0, abs=1e-6)
    assert s["state_centroid_error_m"] == pytest.approx(np.hypot(55.0, 5.0), abs=1e-6)
    # Without a footprint (a capture saved before D-157) the ship is its centre, as before.
    old = dict(cap, truth={"ship": {k: v for k, v in SHIP.items() if k != "footprint"}})
    s = cre.evaluate_capture(old)["cells=off,row=off"]["targets"]["ship"]
    assert s["state_error_m"] == pytest.approx(np.hypot(55.0, 5.0), abs=1e-6) == s["state_centroid_error_m"]


def test_range_and_cross_range_are_split_from_the_outlines_nearest_point():
    """With the ownship: the hull error split along and across the line of sight to the ship's centre, from the
    outline's nearest point; the centre's split kept as centroid_*."""
    from gpusonar import classify
    truth = cre.truth_from_json({"ship": SHIP})
    tr = [cre.track_from_json(_ship_track(1, 30.0, 40.0))]                   # 40 m beyond the line, on the hull's span
    classes = classify.classify_tracks(tr, time=99.5)
    s = cre.score(classes, tr, truth, 99.5, 200.0, origin=(0.0, 0.0))["targets"]["ship"]
    assert s["state_error_m"] == pytest.approx(40.0, abs=1e-6)
    assert s["state_range_error_m"] == pytest.approx(40.0 * np.cos(np.arctan2(800.0 - 6.0 * 99.5, 12000.0)), abs=0.1)
    assert s["state_centroid_error_m"] == pytest.approx(50.0, abs=1e-6)


def test_the_gate_and_false_movers_use_the_outline():
    """The gate is measured from the outline too: a mover 250 m ahead of the ship's centre is 190 m from its bow, within
    the 200 m gate (tracked; before, not), and a mover on the ship's stern 50 m from its centre is on a moving object,
    not a false mover, at the FLS scenes' 15 m gate (before: a false mover). A point object is unchanged."""
    import sys
    from gpusonar import classify
    sys.path.insert(0, str(ROOT / "tools"))
    import gallery
    truth = cre.truth_from_json({"ship": SHIP})
    tr = [cre.track_from_json(_ship_track(1, 250.0, 0.0))]
    classes = classify.classify_tracks(tr, time=99.5)
    s = cre.score(classes, tr, truth, 99.5, 200.0)
    assert s["targets"]["ship"]["tracked"] and s["targets"]["ship"]["state_error_m"] == pytest.approx(190.0, abs=1e-6)
    got = gallery.score_tracks(tr, truth, 99.5, 200.0)["ship"]
    assert got["tracked"] and got["position_error_m"] == pytest.approx(190.0, abs=1e-6) and got["centroid_error_m"] == pytest.approx(250.0, abs=1e-6)
    point = cre.truth_from_json({"ship": {k: v for k, v in SHIP.items() if k != "footprint"}})
    assert not cre.score(classes, tr, point, 99.5, 200.0)["targets"]["ship"]["tracked"]
    stern = [cre.track_from_json(_ship_track(2, -50.0, 3.0))]
    classes = classify.classify_tracks(stern, time=99.5)
    assert classes[0].kind == "mover"
    assert cre.score(classes, stern, truth, 99.5, 15.0)["false_movers"] == 0
    assert cre.score(classes, stern, point, 99.5, 15.0)["false_movers"] == 1
    assert gallery.classify_and_score(stern, truth, 99.5, 15.0)[1]["movers_not_on_a_moving_object"] == 0


def test_footprints_of_the_scene_items():
    """gallery.item_footprint: the convex hull of several points (popoto_12km's ship: the segment from 740 to 860 m),
    the plan of a hull or boat (length along the heading, the beam across), the rectangle of a small box; a sphere and
    a single point have none. The footprints go through the capture's JSON."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import gallery
    from gpusonar.config import load_scenario_config
    ship = gallery.scene_truth(load_scenario_config("popoto_12km"))["ship"]
    assert np.allclose(ship.footprint, [[740.0, 12000.0], [860.0, 12000.0]]) and np.allclose(ship[0], [800.0, 12000.0, 4.0])
    hull = gallery.item_footprint({"type": "hull", "origin": [230.0, 60.0, 0.0], "length": 12.0, "beam": 3.5, "heading_deg": -90.0})
    assert np.ptp(hull[:, 1]) == pytest.approx(12.0) and np.ptp(hull[:, 0]) == pytest.approx(3.5, abs=1e-3)
    assert gallery.outline_distance((230.0, 60.0), hull)[0] == 0.0 and gallery.outline_distance((240.0, 60.0), hull)[0] == pytest.approx(8.25, abs=1e-3)
    box = gallery.item_footprint({"type": "rocks", "x_range": [245.0, 255.0], "y_range": [-40.0, -30.0]})
    assert gallery.outline_distance((250.0, -20.0), box)[0] == pytest.approx(10.0)
    assert gallery.item_footprint({"type": "sphere", "centre": [1.0, 2.0, 3.0], "radius": 0.6}) is None
    assert gallery.item_footprint({"type": "points", "positions": [[1.0, 2.0, 0.0]]}) is None
    back = cre.truth_from_json(json.loads(json.dumps(cre.truth_to_json({"ship": ship}))))
    assert np.allclose(back["ship"].footprint, ship.footprint) and np.allclose(back["ship"][1], ship[1])


def test_cluster_profiles_record_each_hits_bearing_profiles():
    # --profiles (sonar-4s2): two separate blobs on a [bearing, range] map; each hit's rows, the window of rows about
    # them, and the summed and peak-column profiles in dB, unflagged neighbours included.
    from gpusonar import detect
    b = np.radians(np.arange(-10.0, 10.01, 0.5))
    r = np.arange(100.0, 200.0, 2.0)
    power = np.ones((len(b), len(r)))
    B, Rr = np.meshgrid(np.arange(len(b)), np.arange(len(r)), indexing="ij")
    power += 1e3 * np.exp(-0.5 * ((B - 12.3) / 1.5) ** 2 - 0.5 * ((Rr - 10) / 0.8) ** 2)
    power += 1e2 * np.exp(-0.5 * ((B - 30.0) / 1.5) ** 2 - 0.5 * ((Rr - 40) / 0.8) ** 2)
    mask = power > 30.0
    hits = detect.cluster(mask, power, r, b, guard=2, train=8)
    prof = cre.cluster_profiles(mask, power, r, b, hits, half=4)
    assert len(prof) == len(hits) == 2
    for h, p in zip(hits, prof):
        ib, ir = np.nonzero(mask & (np.abs(np.arange(len(b))[:, None] - (12 if p["peak_row"] < 20 else 30)) < 8))
        assert p["rows"] == sorted(set(ib.tolist()))
        assert p["lo"] == max(min(p["rows"]) - 4, 0)
        assert len(p["sum"]) == min(max(p["rows"]) + 4, len(b) - 1) - p["lo"] + 1
        cols = sorted(set(ir.tolist()))
        want = 10 * np.log10(power[p["lo"]:p["lo"] + len(p["sum"])][:, cols].sum(axis=1))
        np.testing.assert_allclose(p["sum"], want, atol=1e-3)
        j = int(np.argmax(power[ib, ir]))
        assert p["peak_row"] == ib[j]
        np.testing.assert_allclose(p["peak"], 10 * np.log10(power[p["lo"]:p["lo"] + len(p["peak"]), ir[j]]), atol=1e-3)
        assert p["bearing"] == h.bearing and p["cells"] == h.cells
        assert p["range"] == h.range and p["first_range"] == h.first_range and p["first_range"] < p["range"]
    assert prof[0]["peak_row"] == 12 and prof[1]["peak_row"] == 30


def test_flank_patrol_tracks_are_scored_in_the_order_the_scoring_reads():
    """sonar-a98: gallery.run_flank's tracker takes bearings from broadside (+y) towards +x, so its states are world
    (x, y); the scoring (object_track, classify_and_score, track_error) reads (y, x). A submarine crossing at 3 m/s at
    about 2.96 km, tracked by run_flank's tracker from the detections it would make: as tracked, no track is within the
    gate of it (its x and y swapped are 3.5 km away); seen through gallery.flank_tracks_yx it is the mover, scored with
    the range / cross-range split about the array (classify_and_score's `origin`, which classify_rules_eval reads)."""
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import gallery
    from gpusonar import detect
    interval, c = 12.0, 1500.0
    p0, v = np.array([-600.0, 2900.0]), np.array([3.0, 0.0])
    tracker = detect.Tracker(range_sigma=5.0, bearing_sigma=np.radians(0.7), confirm_hits=3, max_misses=2,
                             initial_speed_sigma=5.0)
    for i in range(6):
        t = i * interval
        r = 3000.0
        for _ in range(20):                                 # where the submarine is when the sound reaches it (D-156)
            q = p0 + v * (t + r / c)
            q = q + 4.0 * q / np.hypot(*q)                  # 4 m beyond its centre in range
            r = float(np.hypot(*q))
        h = detect.Detection(r, float(np.arctan2(q[0], q[1])), 20.0, 3)   # run_flank: bearing from +y towards +x
        h.time = t + r / c
        tracks = tracker.update(t, [h])
    truth = {"submarine": gallery.TrueObject(np.array([-600.0, 2900.0, 60.0]), np.array([3.0, 0.0, 0.0]),
                                             gallery.convex_hull([[-628.0, 2900.0], [-572.0, 2900.0]]))}
    t_last = 5 * interval
    kinds = {t.id: "mover" for t in tracks}
    assert gallery.object_track(tracks, kinds, truth["submarine"], 60.0) is None          # as tracked: (x, y) read as (y, x)
    yx = gallery.flank_tracks_yx(tracks)
    assert np.allclose(yx[0].state[:2], tracks[0].state[1::-1]) and np.allclose(yx[0].history[-1][1:], tracks[0].history[-1][:0:-1])
    classes, out = gallery.classify_and_score(yx, truth, t_last, 60.0, origin=(0.0, 0.0))
    assert out["targets"]["submarine"]["kind"] == "mover" and out["movers_not_on_a_moving_object"] == 0
    assert out["targets"]["submarine"]["speed_m_s"] == pytest.approx(3.0, abs=0.1)
    # Through the capture's JSON and classify_rules_eval's scoring: the 4 m beyond the centre is range, not cross-range.
    cap = {"tracks": [cre.track_to_json(t) for t in yx], "truth": cre.truth_to_json(truth), "t_last": t_last, "gate": 60.0,
           "classify_kw": {"origin": (0.0, 0.0)}, "seed": 17}
    sub = cre.evaluate_capture(cap)["cells=off,row=off"]["targets"]["submarine"]
    assert sub["kind"] == "mover" and sub["correct"]
    assert sub["report_centroid_range_error_m"] == pytest.approx(4.0, abs=0.5) and abs(sub["report_centroid_cross_range_error_m"]) < 0.5


def test_a_capture_without_a_recorded_range_estimate_is_the_centroid(tmp_path):
    """The leading edge is the default (sonar-ca3, sonar-a98); captures saved before the estimate was recorded were made
    with the centroid and are read so, and a run with the default refuses to reuse them (one estimate per directory)."""
    assert cre.capture_range_estimate({}) == "centroid"
    assert cre.capture_range_estimate({"range_estimate": "first"}) == "first"
    assert cre.capture_range_estimate({"range_estimate": "centroid"}) == "centroid"
    cap = _capture()
    (tmp_path / "fls_120k_transit_k0.json").write_text(json.dumps(cap))       # an old capture: no range_estimate
    with pytest.raises(SystemExit, match="'centroid', not 'first'"):
        cre.main(["--scenes", "fls_120k_transit", "--seeds", "1", "--captures", str(tmp_path), "--backend", "host"])


def test_a_capture_with_the_targets_doppler_part_is_scored_with_it(tmp_path):
    """sonar-fyj: a capture made with --range-doppler full carries `target_doppler` ({factor, origin}); evaluate_capture
    moves each class along the line of sight from the origin by factor x its closing speed before scoring, as the
    gallery's classify_and_score did. The boat (4 m/s along +x, seen from 100 m behind it on its course) opens at 4 m/s,
    which an up-chirp reads farther: with a factor of 2.5 s its report moves 10 m nearer the ownship (its truth was
    under it, so the error grows by 10 m); the static anchored boat does not move. A directory of
    captures made without it is refused for a run with it."""
    cap = _capture()
    base = cre.evaluate_capture(cap)["cells=off,row=off"]["targets"]
    cap["target_doppler"] = {"factor": 2.5, "origin": [0.0, 0.0]}        # tracker order (y, x): the ownship at x = 0
    cap["range_doppler"] = "full"
    got = cre.evaluate_capture(cap)["cells=off,row=off"]["targets"]
    assert got["boat"]["report_error_m"] == pytest.approx(base["boat"]["report_error_m"] + 10.0, abs=0.05)
    assert got["anchored"]["report_error_m"] == pytest.approx(base["anchored"]["report_error_m"], abs=0.05)
    old = _capture()
    old["range_estimate"] = "first"
    (tmp_path / "fls_120k_transit_k0.json").write_text(json.dumps(old))
    with pytest.raises(SystemExit, match="range_doppler 'off', not 'full'"):
        cre.main(["--scenes", "fls_120k_transit", "--seeds", "1", "--captures", str(tmp_path), "--backend", "host",
                  "--range-doppler", "full"])
