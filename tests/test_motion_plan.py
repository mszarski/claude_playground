import numpy as np
import pytest

from rmr.motion import FPS, mirror, move_to_traj, stretch, traj_to_move
from rmr.plan import extract, frames


def _clip(seconds=4.0):
    t = np.arange(int(seconds * FPS)) / FPS
    A = np.zeros((len(t), 9))
    A[:, 2] = 0.01 * np.sin(2 * np.pi * 0.3 * t)          # slow head bob, 10 mm
    A[:, 4] = np.radians(10) * np.sin(2 * np.pi * 0.25 * t)
    A[:, 3] = np.radians(3) * np.sin(2 * np.pi * 4 * t)    # fast roll wiggle -> energy
    A[:, 6], A[:, 7] = -0.5, 0.5
    A[:, 8] = 0.2
    return A


def test_move_roundtrip():
    A = _clip()
    B = move_to_traj(traj_to_move(A))
    assert abs(len(B) - len(A)) <= 1              # resampling spans time[0]..time[-1]
    u = np.linspace(0, (len(A) - 1) / FPS, len(B))
    expected = np.stack([np.interp(u, np.arange(len(A)) / FPS, A[:, j]) for j in range(9)], -1)
    np.testing.assert_allclose(B, expected, atol=1e-9)


def test_resamples_100hz_clip_to_25hz():
    A = _clip(2.0)
    move = traj_to_move(stretch(A, 4.0), fps=100)
    assert abs(len(move_to_traj(move)) - len(A)) <= 1


def test_mirror_is_an_involution():
    A = _clip()
    np.testing.assert_allclose(mirror(mirror(A)), A)


def test_extract_separates_posture_from_energy():
    plan = extract(_clip())
    keys = plan["keys"]
    assert plan["duration"] == 4.0 and len(keys) == 9
    assert keys[0]["earR"] == pytest.approx(np.degrees(0.5), abs=0.5)
    assert keys[0]["body"] == pytest.approx(np.degrees(0.2), abs=0.5)
    assert all(abs(k["roll"]) < 1.0 for k in keys[:-1])     # the 4 Hz wiggle is filtered out of posture...
    # ...and shows up as energy: RMS over 5 channels of a 3 deg sine on one of them = 3 / sqrt(2) / sqrt(5)
    assert np.mean([k["energy"] for k in keys]) == pytest.approx(3 / np.sqrt(10), abs=0.15)


def test_frames_interpolates_and_holds_missing_channels():
    plan = {"duration": 1.0, "keys": [{"t": 0, "pitch": 0, "z": 5}, {"t": 1.0, "pitch": 10}]}
    F = frames(plan)
    assert F.shape == (25, 8)
    assert F[12, 2] == pytest.approx(10 * 12 / 25)
    assert np.all(F[:, 5] == 5)


def test_procedural_dances_load():
    import importlib.util
    if importlib.util.find_spec("reachy_mini_dances_library") is None:
        pytest.skip("pip install --no-deps reachy-mini-dances-library")
    from rmr.library import dances
    moves = dances(bpms=(114,), beats=8, project=False)
    assert len(moves) >= 19
    A = move_to_traj(moves[0][1])
    assert A.shape[1] == 9 and len(A) >= 100 and np.ptp(A, 0).max() > 0
