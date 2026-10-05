"""Reachy Mini move format <-> 9-DoF trajectory arrays.

A *move* is the JSON format of Pollen's emotion/dance libraries and the SDK's recorded-move player::

    {"description": str, "time": [s, ...],
     "set_target_data": [{"head": 4x4, "antennas": [right, left], "body_yaw": rad}, ...]}

A *trajectory* is a ``(T, 9)`` float array sampled at ``FPS``, columns ``DOF``:
head position in metres, head roll/pitch/yaw in radians (extrinsic xyz), antennas in radians
(the right antenna droops with negative angles, the left with positive), body yaw in radians.

Reference: ``common/motion.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import json

import numpy as np

FPS = 25
DOF = ["x", "y", "z", "roll", "pitch", "yaw", "antenna_right", "antenna_left", "body_yaw"]


def mat_to_rpy(R):
    """Rotation matrices ``(..., 3, 3)`` -> extrinsic xyz roll/pitch/yaw ``(..., 3)``."""
    sy = np.hypot(R[..., 0, 0], R[..., 1, 0])
    return np.stack([np.arctan2(R[..., 2, 1], R[..., 2, 2]),
                     np.arctan2(-R[..., 2, 0], sy),
                     np.arctan2(R[..., 1, 0], R[..., 0, 0])], -1)


def rpy_to_mat(rpy):
    """``(T, 3)`` roll/pitch/yaw -> ``(T, 3, 3)`` rotation matrices ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, cp, cy = np.cos(rpy).T
    sr, sp, sy = np.sin(rpy).T
    R = np.empty((len(rpy), 3, 3))
    R[:, 0] = np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1)
    R[:, 1] = np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1)
    R[:, 2] = np.stack([-sp, cp * sr, cp * cr], -1)
    return R


def load_move(path):
    with open(path) as f:
        return json.load(f)


def move_to_traj(move, fps=FPS):
    """Move dict -> ``(T, 9)`` trajectory, resampled to ``fps`` on the move's own timestamps.

    Library clips are recorded at ~100 Hz; playing their raw frames at 25 fps would be 4x too slow.
    """
    frames = move["set_target_data"]
    t = np.asarray(move["time"], float)
    t = t - t[0]
    H = np.asarray([f["head"] for f in frames], float)
    A = np.concatenate([H[:, :3, 3], mat_to_rpy(H[:, :3, :3]),
                        np.asarray([f["antennas"] for f in frames], float),
                        np.asarray([[f.get("body_yaw", 0.0)] for f in frames], float)], -1)
    duration = float(t[-1]) if t[-1] > 0 else len(frames) / fps
    u = np.linspace(0, duration, max(2, int(round(duration * fps))))
    return np.stack([np.interp(u, t, A[:, j]) for j in range(len(DOF))], -1)


def traj_to_move(A, description="", fps=FPS):
    """``(T, 9)`` trajectory -> move dict."""
    T = len(A)
    H = np.tile(np.eye(4), (T, 1, 1))
    H[:, :3, :3] = rpy_to_mat(A[:, 3:6])
    H[:, :3, 3] = A[:, :3]
    return {"description": description,
            "time": (np.arange(T) / fps).tolist(),
            "set_target_data": [{"head": H[i].tolist(), "antennas": A[i, 6:8].tolist(),
                                 "body_yaw": float(A[i, 8]), "check_collision": False} for i in range(T)]}


def mirror(A):
    """Left-right (sagittal) mirror: negate y, roll, yaw and body yaw; swap and negate the antennas."""
    B = A.copy()
    B[:, [1, 3, 5, 8]] *= -1
    B[:, 6], B[:, 7] = -A[:, 7], -A[:, 6]
    return B


def stretch(A, factor):
    """Uniform time-stretch (``factor > 1`` = slower and longer)."""
    T = len(A)
    n = max(8, int(round(T * factor)))
    u = np.linspace(0, T - 1, n)
    return np.stack([np.interp(u, np.arange(T), A[:, j]) for j in range(A.shape[1])], -1)
