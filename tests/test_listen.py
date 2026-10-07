import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rmr.listen import Listener, listening_traj
from rmr.motion import FPS

ROOT = Path(__file__).resolve().parents[1]


def _speech(phrases=((0.5, 2.0), (2.6, 4.4), (5.0, 5.6)), loud_at=3.4, seconds=8.0, sr=16000):
    """Noise floor with bursts of 'speech' (amplitude-modulated noise) and one louder emphasis."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    x = 0.002 * rng.standard_normal(len(t))
    for a, b in phrases:
        m = (t >= a) & (t < b)
        x[m] += 0.05 * rng.standard_normal(m.sum()) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t[m]))
    m = (t >= loud_at) & (t < loud_at + 0.3)
    x[m] *= 8
    return x.astype(np.float32), sr


def test_nods_at_pauses_and_perks_on_emphasis():
    x, sr = _speech()
    A, events = listening_traj(x, sr)
    kinds = [e for _, e in events]
    nods = [t for t, e in events if e.startswith("nod")]
    assert "perk" in kinds and len(nods) == 2            # phrase 1 and 2 earn nods; the 0.6 s phrase 3 is too short
    assert 2.2 <= nods[0] < 2.6 and 4.6 <= nods[1] < 5.0   # ~0.3 s into each pause
    assert dict((e, t) for t, e in events)["perk"] == pytest.approx(3.4, abs=0.1)
    # small, reachable-sized motion: |pitch| < 12 deg, antennas within [-10, 20] deg droop
    assert np.abs(np.degrees(A[:, 4])).max() < 12
    ears = -np.degrees(A[:, 6])
    assert ears.min() > -25 and ears.max() <= 15.01
    assert np.allclose(-A[:, 6], A[:, 7])                 # both antennas move together


def test_relaxes_after_silence():
    lis = Listener()
    for _ in range(3 * FPS):
        lis.step(-30.0)                                    # loud, steady
    for _ in range(12 * FPS):
        last = lis.step(-90.0)
    assert abs(np.degrees(last[3])) < 0.1 and abs(np.degrees(-last[6]) - 15) < 0.1   # back to neutral


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_port_matches_python():
    x, sr = _speech()
    from rmr.listen import loudness
    db = loudness(x, sr).tolist() + [-100.0] * 30
    lis = Listener()
    py = [lis.step(v) for v in db]
    js_src = (f"import {{ Listener }} from '{ROOT}/visualizer/src/Listen.js';"
              f"const l = new Listener(); const db = {json.dumps(db)};"
              f"const out = db.map((v) => {{ const s = l.step(v); return [s.head_pose[11], s.antennas_position[1],"
              f" Math.atan2(s.head_pose[2], s.head_pose[10])]; }});"
              f"console.log(JSON.stringify({{out, events: l.events}}));")
    r = subprocess.run(["node", "--input-type=module", "-e", js_src], capture_output=True, text=True, check=True)
    js = json.loads(r.stdout)
    py_cols = np.array([[p[2], p[7], p[4]] for p in py])
    # pitch from the pose matrix is exact for R = Ry(pitch) Rx(roll): atan2(sp*cr, cp*cr)
    assert np.allclose(np.array(js["out"]), py_cols, atol=1e-9)
    assert [(round(t, 2), e) for t, e in js["events"]] == lis.events
