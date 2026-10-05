import numpy as np
import pytest

pytest.importorskip("reachy_mini_rust_kinematics")

from rmr.motion import traj_to_move
from rmr.reach import Reach, se3_interp


@pytest.fixture(scope="module")
def reach():
    return Reach()


def test_neutral_pose_is_reachable(reach):
    assert reach.reachable(np.eye(4), 0.0)


def test_extreme_pose_is_projected_to_reachable(reach):
    A = np.zeros((10, 9))
    A[5:, 2] = 0.08                       # 80 mm up: beyond the Stewart platform
    A[5:, 4] = np.radians(40)
    move, frac = reach.project(traj_to_move(A))
    assert frac == pytest.approx(0.5)
    assert all(reach.reachable(f["head"], f["body_yaw"]) for f in move["set_target_data"])


def test_se3_interp_endpoints():
    H1 = np.eye(4); H1[:3, 3] = [0, 0, 0.01]
    H1[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    np.testing.assert_allclose(se3_interp(np.eye(4), H1, 0), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(se3_interp(np.eye(4), H1, 1), H1, atol=1e-12)
