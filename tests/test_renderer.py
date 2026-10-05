import os

import numpy as np
import pytest

import rmr.renderer  # noqa: F401  (before mujoco: see rmr/renderer/__init__.py)

mujoco = pytest.importorskip("mujoco")
from rmr.motion import FPS, move_to_traj, traj_to_move  # noqa: E402
from rmr.renderer.sim import Sim, scene_path  # noqa: E402

if not os.path.exists(scene_path()):
    pytest.skip("Reachy Mini model files not installed", allow_module_level=True)


def _move(pitch_deg=0.0, z=0.0, antennas=(0.0, 0.0), seconds=1.2):
    T = int(seconds * FPS)
    A = np.zeros((T, 9))
    ramp = np.clip(np.arange(T) / (0.5 * FPS), 0, 1)
    A[:, 2], A[:, 4] = z * ramp, np.radians(pitch_deg) * ramp
    A[:, 6], A[:, 7] = antennas[0] * ramp, antennas[1] * ramp
    return traj_to_move(A)


@pytest.fixture(scope="module")
def sim():
    try:
        return Sim(32, 32)
    except Exception as e:          # no OpenGL backend (set MUJOCO_GL=osmesa or egl)
        pytest.skip(f"no MuJoCo rendering backend: {e}")


def _head(sim, move):
    sim.play(move, render=False)
    sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "head")
    return sim.data.site_xmat[sid].reshape(3, 3).copy(), sim.data.site_xpos[sid].copy()


def test_pitch_and_height_move_the_simulated_head_as_commanded(sim):
    R0, p0 = _head(sim, _move())
    R1, _ = _head(sim, _move(pitch_deg=20))
    _, p2 = _head(sim, _move(z=0.015))
    rel = R1 @ R0.T
    angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    assert abs(angle - 20) < 2.5
    assert abs(rel[1, 0]) < 0.1 and abs(rel[0, 1]) < 0.1          # pure pitch about the world y axis
    assert rel[0, 2] > 0.25                                         # + pitch tips the head's front (+x) down
    assert abs((p2 - p0)[2] * 1000 - 15) < 2 and sim.ik_fail == 0


def test_play_renders_one_frame_per_step(sim):
    move = _move(antennas=(-0.5, 0.5), seconds=0.4)
    frames = sim.play(move)
    assert len(frames) == len(move_to_traj(move)) and frames[0].shape == (32, 32, 3)
    assert frames[0].dtype == np.uint8 and frames[-1].std() > 0
