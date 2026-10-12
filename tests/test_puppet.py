"""visualizer/src/Puppet.js: face pose and brows from the camera -> Reachy's head and antennas."""
import json
import math
import os
import shutil
import subprocess

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def _run(body):
    src = (f"import {{ Retarget, toFrame }} from '{ROOT}/visualizer/src/Puppet.js';"
           "const rot = (ax, d) => { const a = d * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);"
           " return ax === 'y' ? [[c,0,s],[0,1,0],[-s,0,c]] : ax === 'x' ? [[1,0,0],[0,c,-s],[0,s,c]] : [[c,-s,0],[s,c,0],[0,0,1]]; };"
           "const face = (ax, d, ty = 0) => ({ R: rot(ax, d), t: [0, ty, -40] });" + body)
    return json.loads(subprocess.run(["node", "--input-type=module", "-e", src], capture_output=True, text=True,
                                     check=True).stdout)


def test_head_follows_the_face_like_a_mirror():
    out = _run("const go = (m, s = {}) => { const r = new Retarget({ smooth: 0 }); r.calibrate(face('y', 0)); return r.frame(m, s); };"
               "console.log(JSON.stringify({ left: go(face('y', 20)), down: go(face('x', 15)), tilt: go(face('z', 10)),"
               " up: go(face('y', 0, 1.5)), copy: (() => { const r = new Retarget({ smooth: 0, mirror: false });"
               " r.calibrate(face('y', 0)); return r.frame(face('y', 20)); })(),"
               " brows: go(face('y', 0), { browOuterUpLeft: 1, browInnerUp: 0.5 }), frown: go(face('y', 0), { browDownRight: 1 }) }));")
    # turning toward the camera's +x (your left) turns Reachy to its right in mirror mode, to its left when copying
    assert math.isclose(out["left"]["yaw"], -20, abs_tol=1e-6) and math.isclose(out["copy"]["yaw"], 20, abs_tol=1e-6)
    assert math.isclose(out["down"]["pitch"], 15, abs_tol=1e-6)          # face pitched toward the floor = head down
    assert abs(abs(out["tilt"]["roll"]) - 10) < 1e-6
    assert out["up"]["z"] > 3 + 10                                      # rising 1.5 cm lifts the head
    assert out["brows"]["earR"] < 0 < out["brows"]["earL"] < 15         # your left brow -> its right antenna (mirror)
    assert out["frown"]["earL"] > 100 and round(out["frown"]["earR"]) == 15


def test_limits_smoothing_and_unreachable_poses_hold():
    out = _run("const r = new Retarget({ smooth: 0.5 }); r.calibrate(face('y', 0));"
               "const a = r.frame(face('y', 80)), b = r.frame(face('y', 80));"
               "const ok = toFrame({ pitch: 0, roll: 0, yaw: 0, z: 3, earR: 15, earL: 15 });"
               "const held = toFrame({ pitch: 0, roll: -25, yaw: 0, z: 25, earR: 15, earL: 15 }, ok);"
               "console.log(JSON.stringify({ a: a.yaw, b: b.yaw, held: held === ok, ant: ok.move.antennas }));")
    assert out["a"] == -50 and out["b"] == -50                          # clamped to the recipe limit
    assert out["held"]                                                  # out of reach -> the last good frame
    assert out["ant"][0] < 0 < out["ant"][1]                            # right droops negative, left positive
