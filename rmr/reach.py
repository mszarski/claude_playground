"""Project a move onto the Stewart platform's reachable set, frame by frame.

The head's reachable set is coupled (a tilt reachable at one height is not at another), so clipping
each channel cannot guarantee a valid pose, and the robot freezes on an unreachable target. Instead,
each unreachable head pose is line-searched (bisection) from the last reachable pose toward the target,
with SE(3) interpolation and body yaw interpolated alongside. Antennas and timing are untouched.

Needs the SDK's analytical IK: ``pip install reachy-mini-rust-kinematics`` (the ``ik`` extra).
Reference: ``common/reach.py`` in pham-tuan-binh/reachy-motion-generator (Apache-2.0).
"""
import json
from pathlib import Path

import numpy as np

# From pollen-robotics/reachy_mini (Apache-2.0).
KINEMATICS = Path(__file__).parent / "assets" / "kinematics_data.json"


def se3_interp(H0, H1, a):
    """Interpolate 4x4 poses: linear in position, geodesic (axis-angle) in rotation."""
    R0, R1 = H0[:3, :3], H1[:3, :3]
    Rrel = R0.T @ R1
    angle = np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
    if angle < 1e-8:
        Ra = R0
    else:
        w = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
        w /= 2 * np.sin(angle)
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        t = a * angle
        Ra = R0 @ (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K)
    H = np.eye(4)
    H[:3, :3] = Ra
    H[:3, 3] = (1 - a) * H0[:3, 3] + a * H1[:3, 3]
    return H


class Reach:
    def __init__(self, kinematics_path=KINEMATICS):
        from reachy_mini_rust_kinematics import ReachyMiniRustKinematics

        with open(kinematics_path) as f:
            d = json.load(f)
        self.z_offset = d["head_z_offset"]
        self.kin = ReachyMiniRustKinematics(d["motor_arm_length"], d["rod_length"])
        for m in d["motors"]:
            self.kin.add_branch(m["branch_position"], np.linalg.inv(m["T_motor_world"]), 1 if m["solution"] else -1)

    def ik(self, H, body_yaw):
        """Head pose (4x4) + body yaw -> 7 joint targets ``[body_yaw, stewart_1..6]`` (NaN if unreachable).
        Uses the same yaw limits as the robot daemon."""
        p = np.array(H, float)
        p[2, 3] += self.z_offset
        return np.array(self.kin.inverse_kinematics_safe(p.tolist(), body_yaw=float(body_yaw),
                                                         max_relative_yaw=np.deg2rad(65),
                                                         max_body_yaw=np.deg2rad(160)))

    def reachable(self, H, body_yaw):
        return bool(np.all(np.isfinite(self.ik(H, body_yaw))))

    def project(self, move, iters=12):
        """Make every frame of ``move`` reachable, in place.
        Returns ``(move, fraction of frames that had to be projected)``."""
        frames = move["set_target_data"]
        last = (np.eye(4), 0.0)
        fixed = 0
        for f in frames:
            H, yaw = np.array(f["head"], float), float(f.get("body_yaw", 0.0))
            if self.reachable(H, yaw):
                last = (H, yaw)
                continue
            fixed += 1
            H0, y0 = last
            lo, hi = 0.0, 1.0
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if self.reachable(se3_interp(H0, H, mid), (1 - mid) * y0 + mid * yaw):
                    lo = mid
                else:
                    hi = mid
            Hp, yp = se3_interp(H0, H, lo), (1 - lo) * y0 + lo * yaw
            f["head"], f["body_yaw"] = Hp.tolist(), yp
            last = (Hp, yp)
        return move, fixed / max(1, len(frames))
