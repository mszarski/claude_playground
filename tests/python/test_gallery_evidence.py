"""tools/gallery.py `--evidence` (opt-in, issue sonar-4pon): the classifier's evidence per track in gallery.json.

The gallery's `movers_not_on_a_moving_object` counts false movers but says nothing of which track or why. With
`--evidence` (gallery.EVIDENCE) classify_and_score adds `classification.evidence`: one row per track with the
classifier tap's numbers (classify.CLASS_EVIDENCE), the track's position and velocity in world (x, y) at the last
ping, whether it is one of the false movers counted, and its nearest true object. Off by default: the output is then
exactly what it was. No simulation here: known tracks, known truth (the made-up case of test_classify_rules_eval)."""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
T = np.arange(8, dtype=float)


@pytest.fixture()
def gallery():
    spec = importlib.util.spec_from_file_location("gallery_evidence_under_test", ROOT / "tools" / "gallery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _track(i, x0, y0, vx, vy, snr=20.0):
    from gpusonar.detect import Track
    jit = 0.05 * (-1.0) ** np.arange(len(T))
    xs, ys = x0 + vx * T + jit, y0 + vy * T - jit
    tr = Track(id=i, state=np.array([ys[-1], xs[-1], vy, vx]), cov=np.eye(4), hits=len(T))
    tr.history = [(float(t), float(y), float(x)) for t, x, y in zip(T, xs, ys)]
    tr.snr = [snr] * len(T)
    return tr


def _case(gallery):
    truth = {"boat": gallery.TrueObject((100.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
             "anchored": gallery.TrueObject((150.0, 50.0, 0.0), (0.0, 0.0, 0.0))}
    # A boat under way, an anchored boat and a glint joined into a line far from every moving object (a false mover).
    tracks = [_track(1, 100.0, 0.0, 4.0, 0.0), _track(2, 150.0, 50.0, 0.0, 0.0), _track(3, 200.0, -80.0, 0.0, 3.0)]
    return tracks, truth


def test_off_by_default_and_the_output_unchanged(gallery):
    tracks, truth = _case(gallery)
    assert gallery.EVIDENCE is False
    _, out = gallery.classify_and_score(tracks, truth, float(T[-1]), 15.0, min_speed=0.5, max_fit_rms=2.5)
    assert "evidence" not in out and out["movers_not_on_a_moving_object"] == 1


def test_evidence_names_the_false_mover_and_why(gallery, monkeypatch):
    tracks, truth = _case(gallery)
    _, plain = gallery.classify_and_score(tracks, truth, float(T[-1]), 15.0, min_speed=0.5, max_fit_rms=2.5)
    monkeypatch.setattr(gallery, "EVIDENCE", True)
    classes, out = gallery.classify_and_score(tracks, truth, float(T[-1]), 15.0, min_speed=0.5, max_fit_rms=2.5)
    ev = out.pop("evidence")
    assert out == plain                                          # read only: everything else as without it
    assert [e["id"] for e in ev] == [1, 2, 3]
    by = {e["id"]: e for e in ev}
    for e in ev:
        assert set(e) >= set(gallery.classify.CLASS_EVIDENCE) | {"position_xy", "velocity_xy", "false_mover", "nearest"}
    assert [e["id"] for e in ev if e["false_mover"]] == [3] and sum(e["false_mover"] for e in ev) == plain["movers_not_on_a_moving_object"]
    g = by[3]
    assert g["kind"] == "mover" and g["rule"] == "mover" and g["speed"] >= g["need"] * g["speed_sigma"]
    assert g["position_xy"] == pytest.approx([200.0, -80.0 + 3.0 * T[-1]], abs=0.1)     # world (x, y) at the last ping
    assert g["velocity_xy"] == pytest.approx([0.0, 3.0], abs=0.05)
    assert by[1]["nearest"]["label"] == "boat" and by[1]["nearest"]["distance_m"] < 0.2 and not by[1]["false_mover"]
    assert by[2]["kind"] == "static" and by[2]["nearest"]["label"] == "anchored"
