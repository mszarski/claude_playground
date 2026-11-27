"""The live chain seen in the scene viewer (python/gpusonar/viz/chain.py, page3d.py; docs/tasks/VISUALIZATION.md
section 15, package V6).

What is held: the trace's `chain` member is the report lines (detections, tracks, own-ship, counts) of the run it was
made from, in both engines; its tracker tap is the tracker's own, turned into world north/east; the truth is the
scenario's scene at each ping and the association error is the distance to the nearest object; two engines' runs are
compared at the tolerances test_live_cpp.py uses; a page with the chain view stays under its size limits and, in
headless Chromium (opt-in: Node and Playwright), draws as many detections and tracks at each ping as the report lines
hold. The taps' read-only proof (report lines bit-identical with them on and off) is in test_live_cpp.py.
"""

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest

from gpusonar.sim import Scene
from gpusonar.viz import chain as vchain
from gpusonar.viz import page3d
from gpusonar.viz import trace as tr
from test_live import CHAIN, copies, pings  # noqa: F401  (pings is a fixture)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
TRUTH = {"truth": [{"label": "crossing", "positions": [[250.0, 40.0 - 6.0 * i, 0.0] for i in range(6)]},
                   {"label": "still", "positions": [[320.0, -60.0, 0.0]] * 6}],
         "times_s": [2.0 * i for i in range(6)], "truth_rule": "the test's two points", "quick": None}


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_traces_chain_member_equals_the_report_lines(pings, engine):
    runs = vchain.run_chain(copies(pings), {k: v for k, v in CHAIN.items() if k != "backend"}, engine)
    m = vchain.chain_member(runs, TRUTH, "the test's chain", engine, CHAIN)
    assert m["n_pings"] == len(pings) == len(m["pings"]) and m["engine"] == engine
    assert set(m["taps"]) == {"tracker", "classifier", "detector"}
    for p, run in zip(m["pings"], runs):
        rep = run["report"]
        assert p["report_counts"] == {"detections": len(rep["detections"]), "tracks": len(rep["tracks"])}
        assert len(p["detections"]) == len(rep["detections"]) and len(p["tracks"]) == len(rep["tracks"])
        for a, b in zip(p["detections"], rep["detections"]):
            assert set(a) == set(b)
            for k in ("range_m", "bearing_deg", "snr_db"):
                assert a[k] == pytest.approx(b[k], abs=1e-4)
        for a, b in zip(p["tracks"], rep["tracks"]):
            assert (a["id"], a["kind"], a["parent"]) == (b["id"], b["kind"], b["parent"])
            for k in ("x", "y", "vx", "vy", "speed_m_s", "range_m", "seen"):
                assert a[k] == pytest.approx(b[k], abs=1e-4)
        np.testing.assert_allclose(p["ownship"], rep["ownship_xyz"], atol=1e-3)
        assert p["time_ns"] == rep["time_ns"] and p["pings_without_nav"] == rep["pings_without_nav"]
        # The tracker's tap in world (north, east): the state's (east, north) offsets from the first ping's origin.
        o0 = runs[0]["report"]["ownship_xyz"]
        for t, raw in zip(p["tracker"]["after"], run["taps"]["tracker"]["after"]):
            assert t["id"] == raw["id"]
            assert t["pos"] == pytest.approx([raw["state"][1] + o0[0], raw["state"][0] + o0[1]], abs=1e-3)
            assert t["cov"] == pytest.approx([raw["cov"][5], raw["cov"][1], raw["cov"][0]], rel=1e-3, abs=1e-4)
        # Every confirmed track the tap holds after the update is a reported track, and the report's are confirmed ones.
        assert [t["id"] for t in p["tracker"]["after"] if t["confirmed"]] == [t["id"] for t in rep["tracks"]]
        assert [e["id"] for e in p["evidence"]] == [t["id"] for t in rep["tracks"]]
        if p.get("map"):
            mp = p["map"]
            assert mp["rows"] * mp["cols"] == len(__import__("base64").b64decode(mp["u8"]))
            assert mp["native"] == list(run["map"]["power"].shape) and "background_u8" in mp
    assert any(p["tracks"] for p in m["pings"])


def test_truth_and_the_association_error_are_the_scenarios():
    """Two objects of a scene, one moving: their centroids at each ping's time; a track 3 m from the moving one and 4
    m from nothing else is associated with it at 3 m; the seafloor and an area wider than 30 m are not truth."""
    scene = Scene().add_points([[100.0, 10.0, 5.0], [102.0, 12.0, 5.0]], velocity=(1.0, -2.0, 0.0), label="boat")
    scene.add_points([[50.0, -20.0, 9.0]], label="buoy")
    scene.add_points([[0.0, 0.0, 20.0], [100.0, 100.0, 20.0]], label="bed")
    scene.add_points([[0.0, 0.0, 19.0], [80.0, 0.0, 19.0]], label="boulders")
    cfg = {"scene": [{"type": "points", "label": "boat"}, {"type": "points", "label": "buoy"},
                     {"type": "seafloor", "label": "bed"}, {"type": "boulder_field", "label": "boulders", "x_range": [0.0, 80.0],
                                                              "y_range": [0.0, 5.0]}]}
    truth = vchain.truth_objects(cfg, scene, [0.0, 10.0])
    assert [o["label"] for o in truth] == ["boat", "buoy"]
    assert truth[0]["positions"][0] == pytest.approx([101.0, 11.0, 5.0]) and truth[0]["positions"][1] == pytest.approx([111.0, -9.0, 5.0])
    tracks = [{"id": 7, "x": 111.0 + 3.0, "y": -9.0, "kind": "mover"}, {"id": 8, "x": 50.0, "y": -16.0, "kind": "static"}]
    a = vchain.associate(tracks, truth, 1)
    assert a == [{"id": 7, "truth": "boat", "error_m": pytest.approx(3.0)}, {"id": 8, "truth": "buoy", "error_m": pytest.approx(4.0)}]
    assert vchain.associate(tracks, [], 0)[0]["truth"] is None


def test_the_engines_are_compared_at_the_tests_tolerances(pings):
    runs = vchain.run_chain(copies(pings), CHAIN, "cpp", taps=())
    assert vchain.compare_engines(runs, runs) == []
    py = vchain.run_chain(copies(pings), CHAIN, "python", taps=())
    assert vchain.compare_engines(runs, py) == []                          # the two engines agree here (test_live_cpp)
    moved = json.loads(json.dumps([{"report": r["report"]} for r in runs]))
    moved[-1]["report"]["tracks"][0]["x"] += 0.06
    d = vchain.compare_engines(runs, moved)
    assert len(d) == 1 and "track" in d[0] and " x " in d[0]


def test_a_page_with_the_chain_view_stays_under_its_limits(pings, tmp_path):
    """The chain member's own budget (viz/chain.py CHAIN_BUDGET) and the page limits with it (page3d.PAGE_LIMIT_CHAIN_*),
    on 6 pings of a map 81 bearings x 299 cells; and 40 pings of a map ten times as long, downsampled to the budget."""
    runs = vchain.run_chain(copies(pings), CHAIN, "cpp")
    m = vchain.chain_member(runs, TRUTH, "the test's chain", "cpp", CHAIN)
    assert vchain.chain_json_bytes(m) < vchain.CHAIN_BUDGET
    big = [dict(r, map=dict(r["map"], power=np.tile(r["map"]["power"], (1, 10)), ranges=np.linspace(100, 3100, 2990))) for r in runs] * 7
    for r in big:
        r["taps"] = dict(r["taps"], detector=None)
    mb = vchain.chain_member(big[:40], TRUTH, "the test's chain", "cpp", CHAIN)
    assert sum(len(p["map"]["u8"]) for p in mb["pings"]) <= vchain.CHAIN_MAP_BUDGET
    assert vchain.chain_json_bytes(mb) < vchain.CHAIN_BUDGET
    trace = tr.static_trace("flank_target_1km")
    trace["chain"] = m
    cdn = page3d.write_page(trace, str(tmp_path / "c.html"), three="cdn")
    assert cdn["bytes"] < page3d.PAGE_LIMIT_CHAIN_CDN and cdn["chain"]["pings"] == len(pings)
    if os.path.exists(page3d.VENDORED):
        emb = page3d.write_page(trace, str(tmp_path / "e.html"), three="embed")
        assert emb["bytes"] < page3d.PAGE_LIMIT_CHAIN_EMBED
    # A page without a chain says so, with the command.
    t2 = tr.static_trace("flank_target_1km")
    t2["chain_missing"] = "make it with `tools/scenario_view.py flank_target_1km --chain`"
    page3d.write_page(t2, str(tmp_path / "n.html"), three="cdn")
    assert "--chain" in open(tmp_path / "n.html").read() and "no chain run for this scene" in open(tmp_path / "n.html").read()


def test_the_sound_per_ping_keeps_levels_between_pings(pings, tmp_path):
    from gpusonar.viz import audio
    snd = vchain.write_chain_audio(str(tmp_path), "t", copies(pings)[:3], channels=(0, 31))
    assert len(snd["files"]) == 6 and snd["method"] == "as_recorded"
    peaks = [np.max(np.abs(audio.read_pcm16(str(tmp_path / snd["dir"] / f["file"]))["samples"])) for f in snd["files"]]
    assert max(peaks) == pytest.approx(10 ** (-1 / 20), abs=2e-4)          # one gain for every ping and hydrophone
    assert all(p > 0 for p in peaks)


def _browser():
    node = shutil.which("node")
    if node is None:
        return None
    probe = subprocess.run([node, "-e", "require('playwright').chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader',"
                            "'--enable-unsafe-swiftshader']}).then(b=>b.close()).then(()=>process.exit(0),()=>process.exit(3))"],
                           capture_output=True, text=True, timeout=120)
    return node if probe.returncode == 0 else None


def test_the_chain_layer_draws_what_the_report_lines_hold(pings, tmp_path):
    node = _browser()
    if node is None:
        pytest.skip("Node with Playwright's Chromium is not available")
    if not os.path.exists(page3d.VENDORED):
        pytest.skip("third_party/three/three.module.min.js is not present (pages would need the network)")
    runs = vchain.run_chain(copies(pings), CHAIN, "cpp")
    m = vchain.chain_member(runs, TRUTH, "the test's chain", "cpp", CHAIN)
    trace = tr.static_trace("flank_target_1km")
    trace["chain"] = m
    page3d.write_page(trace, str(tmp_path / "chain.html"), three="embed")
    out = subprocess.run([node, os.path.join(ROOT, "tools", "page3d_check.js"), str(tmp_path / "chain.html")], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["ok"] and res["errors"] == [] and res["info"]["chain"] == len(pings)
    got = [(c["drawn_detections"], c["drawn_tracks"]) for c in res["chain"]]
    assert got == [(len(r["report"]["detections"]), len(r["report"]["tracks"])) for r in runs]
    assert any(c["ellipses"] > 0 and c["gates"] > 0 and c["map"] for c in res["chain"])
