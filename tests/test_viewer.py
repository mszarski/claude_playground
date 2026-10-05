import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from rmr.motion import rpy_to_mat, traj_to_move
from rmr.reach import Reach
from rmr.viewer import entries

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_browser_ik_matches_the_sdk_ik():
    rng = np.random.default_rng(0)
    R, cases = Reach(), []
    while len(cases) < 200:
        H = np.eye(4)
        H[:3, :3] = rpy_to_mat(np.radians(rng.uniform([-20, -25, -40], [20, 25, 40]))[None])[0]
        H[:3, 3] = rng.uniform([-0.01, -0.01, -0.015], [0.01, 0.01, 0.02])
        yaw = float(np.radians(rng.uniform(-60, 60)))
        q = R.ik(H, yaw)
        if np.all(np.isfinite(q)):
            cases.append((H.tolist(), yaw, q.tolist()))
    script = ("import { headJoints } from './visualizer/src/StewartIK.js';"
              "const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify(cases.map(([H, y]) => headJoints(H, y))));")
    script = script.replace("require('fs')", "(await import('fs'))")
    out = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(cases), cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    js = np.array(json.loads(out))
    py = np.array([c[2] for c in cases])
    assert js.shape == py.shape == (200, 7)
    assert np.abs(js - py).max() < 1e-9


def test_gallery_groups_variants_and_keeps_recipes(tmp_path):
    os.makedirs(tmp_path / "motions")
    A = np.zeros((5, 9))
    for i in range(2):
        json.dump(traj_to_move(A + 0.1234567 * i, "sneezing. Ah choo."), open(tmp_path / "motions" / f"s__{i}.json", "w"))
    json.dump(traj_to_move(A, "brave."), open(tmp_path / "motions" / "b__0.json", "w"))
    json.dump({"sneezing. Ah choo.": "hold 1"}, open(tmp_path / "recipes.json", "w"))
    e = entries(str(tmp_path), "demo")
    assert [(x["prompt"], len(x["moves"]), x["recipe"], x["source"]) for x in e] == \
        [("brave.", 1, "", "demo"), ("sneezing. Ah choo.", 2, "hold 1", "demo")]
    assert e[1]["moves"][1]["set_target_data"][0]["antennas"][0] == 0.12346
