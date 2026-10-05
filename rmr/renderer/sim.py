"""Faithful MuJoCo playback of a move, mirroring the SDK's MuJoCo daemon backend.

- Loads the official model (``descriptions/reachy_mini/mjcf/scenes/empty.xml`` from the ``reachy-mini``
  package, or from ``$REACHY_MINI_ROOT``). Only the model files are needed: ``pip install --no-deps reachy-mini``.
- Converts each head pose to joint targets with the SDK's analytical IK (``rmr.reach.Reach.ik``), drives
  ``data.ctrl`` and steps physics at 500 Hz so the Stewart platform's passive joints settle.
- Antenna targets are negated, as in the backend.
- Resets through a settle-and-ramp: snapping the motors straight to a target can close the linkage in a
  mirrored assembly mode (head facing ~165 deg the wrong way).
- Unreachable poses hold the last valid command, like the daemon.

Reference: ``renderer/sim.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import importlib.util
import os

import numpy as np

from ..motion import FPS, load_move, move_to_traj, rpy_to_mat
from ..reach import Reach


def scene_path(scene="empty"):
    root = os.environ.get("REACHY_MINI_ROOT")
    if root is None:
        spec = importlib.util.find_spec("reachy_mini")      # locate without importing the SDK
        if spec is None:
            raise SystemExit("Rendering needs the Reachy Mini model files: `pip install --no-deps reachy-mini` "
                             "or set REACHY_MINI_ROOT=<path to reachy_mini/src/reachy_mini>.")
        root = spec.submodule_search_locations[0]
    return os.path.join(root, "descriptions", "reachy_mini", "mjcf", "scenes", f"{scene}.xml")


def move_arrays(move, fps=FPS):
    """Move -> (head poses (T,4,4), antennas (T,2), body yaw (T,)) resampled to ``fps``."""
    A = move_to_traj(move, fps)
    H = np.tile(np.eye(4), (len(A), 1, 1))
    H[:, :3, :3] = rpy_to_mat(A[:, 3:6])
    H[:, :3, 3] = A[:, :3]
    return H, A[:, 6:8], A[:, 8]


class Sim:
    def __init__(self, width=520, height=420, scene="empty", distance=0.52, elevation=-6, azimuth=158):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene_path(scene))
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.R = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = distance, elevation, azimuth
        self.cam.lookat[:] = [0, 0, 0.16]
        self.ik = Reach().ik
        self.ik_fail = 0

    @staticmethod
    def _ctrl(q7, ant):
        return np.concatenate([q7, [-ant[0], -ant[1]]])

    def reset(self, tgt, neutral, ramp=300, settle=100):
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        con = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        q0 = self._ctrl(neutral, [0.0, 0.0])
        d.ctrl[:] = q0
        mj.mj_forward(m, d)
        for _ in range(settle):
            mj.mj_step(m, d)
        for k in range(1, ramp + 1):
            d.ctrl[:] = q0 + (np.asarray(tgt) - q0) * (k / ramp)
            mj.mj_step(m, d)
        for _ in range(settle):
            mj.mj_step(m, d)
        m.geom_contype[:], m.geom_conaffinity[:] = con

    def play(self, move, fps=FPS, render=True):
        """Move dict -> list of RGB frames at ``fps`` (``render=False``: physics only, returns ``[]``)."""
        H, ant, yaw = move_arrays(move, fps)
        n_sub = max(1, int(round((1.0 / fps) / self.model.opt.timestep)))
        q0 = self.ik(H[0], yaw[0])
        last = q0 if np.all(np.isfinite(q0)) else np.zeros(7)
        self.reset(self._ctrl(last, ant[0]), neutral=self.ik(np.eye(4), 0.0))
        frames, self.ik_fail = [], 0
        for i in range(len(H)):
            q = self.ik(H[i], yaw[i])
            if np.all(np.isfinite(q)):
                last = q
            else:
                self.ik_fail += 1
            self.data.ctrl[:] = self._ctrl(last, ant[i])
            for _ in range(n_sub):
                self.mj.mj_step(self.model, self.data)
            if render:
                self.R.update_scene(self.data, self.cam)
                frames.append(self.R.render())
        return frames

    def play_file(self, path, fps=FPS):
        return self.play(load_move(path), fps)
