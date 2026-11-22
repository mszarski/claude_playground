"""Write tests/python/data/classes_d157.json: the default classes (fit through the measured positions, D-157) of the
tracker scenario of test_detect, both engines, both associations, at no time and at the last ping's time."""
import json, pathlib, sys
import numpy as np
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root / "python"), str(root / "tests" / "python")]
from gpusonar import classify, detect
from test_detect import SCENARIO_TRACKER, tracker_scenario
from test_live_cpp import core
out = {}
for association in ("likelihood", "in_order"):
    py = detect.Tracker(**SCENARIO_TRACKER, association=association)
    cpp = core.LiveTracker(**SCENARIO_TRACKER, association=association)
    for t, dets in tracker_scenario():
        a = py.update(t, [detect.Detection(*d) for d in dets])
        cpp.update(t, [d[:4] for d in dets], None)
    rec = {}
    for name, at in (("none", None), ("last_ping", t)):
        pc = classify.classify_tracks(a, min_speed=0.5, max_fit_rms=2.5, time=at)
        cc = cpp.classify(min_speed=0.5, max_fit_rms=2.5, time=at)
        rec[name] = {"python": [dict(id=c.id, kind=c.kind, position=list(c.position), velocity=list(c.velocity), fit_rms=c.fit_rms) for c in pc],
                     "cpp": [dict(id=c["id"], kind=c["kind"], position=list(c["position"]), velocity=list(c["velocity"]), fit_rms=c["fit_rms"]) for c in cc]}
    out[association] = rec
(root / "tests/python/data/classes_d157.json").write_text(json.dumps(out, indent=1) + "\n")
for a, r in out.items():
    print(a, [(c["id"], c["kind"]) for c in r["last_ping"]["python"]], [(c["id"], c["kind"]) for c in r["last_ping"]["cpp"]])
