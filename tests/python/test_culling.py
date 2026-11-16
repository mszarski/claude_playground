"""Milestone 5: coarse-to-fine ray culling (ASASIN Sec. II-C, Fig. 5).

Acceptance: image unchanged, speedup reported. The speedup figure printed here is for
the CPU backend; the paper's "factor of two" is a GPU figure for its own data.
"""

import time

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

C = 1500.0
TILT = np.radians(20.0)


def stripmap(n_pings=48, spacing=0.32, y0=30.0, y1=60.0):
    """A short stripmap collection: side-looking array flying along x past a few targets."""
    sonar = SonarModel.load("sas_auv", {"receive": {"window": {"range": [y0 - 2, y1 + 6]}}})
    track = (np.arange(n_pings) - n_pings / 2) * spacing
    targets = np.array([[0.0, 40.0, 0.0], [3.0, 50.0, 0.0], [-4.0, 35.0, 0.0]])
    targets[:, 2] = targets[:, 1] * np.tan(TILT)            # on the boresight plane
    scene = Scene().add_points(targets)
    records = [simulate_ping(sonar, make_pose(position=(x, 0.0, 0.0), velocity=(1.6, 0.0, 0.0)), scene,
                             environment={"sound_speed": C}, absorption_model="none", adc=False,
                             sequence=i)
               for i, x in enumerate(track)]
    recon.prepare(records)
    return sonar, records, track


def seafloor_grid(track, rows=160, cols=240, y0=32.0, y1=58.0):
    x = np.linspace(track[0] - 1.0, track[-1] + 1.0, rows)
    y = np.linspace(y0, y1, cols)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), (Y * np.tan(TILT)).ravel()]), (rows, cols)


@pytest.fixture(scope="module")
def collection():
    return stripmap()


@pytest.mark.parametrize("block", [1, 5, 16, 64])
def test_culled_image_is_identical(collection, block):
    _, records, track = collection
    pts, shape = seafloor_grid(track)
    kw = dict(interp="linear", fov_factor=1.5)
    plain = gs.core.backproject(records, pts, **kw)
    stats = {}
    culled = gs.core.backproject(records, pts, cull_block=block, grid_shape=shape, stats=stats, **kw)
    # Not "close": identical. A culled block contributes exactly nothing.
    np.testing.assert_array_equal(culled, plain)
    assert np.abs(plain).max() > 0
    assert 0 < stats["ping_blocks_kept"] < stats["ping_blocks_total"]


def test_narrow_beam_between_block_corners_is_not_lost():
    """The case the paper's corner test misses: a beam narrower than a block.

    Near the sonar a 9.5 degree beam is under a metre wide; with huge blocks the beam
    passes between corners. The widened corner test must still keep the block.
    """
    sonar = SonarModel.load("sas_auv", {"receive": {"window": {"range": [2.0, 12.0]}}})
    target = np.array([0.0, 5.0, 5.0 * np.tan(TILT)])
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target]),
                        environment={"sound_speed": C}, absorption_model="none", adc=False)
    recon.prepare([rec])
    x = np.linspace(-20.0, 20.0, 81)          # 40 m of along-track for a 0.8 m wide beam
    y = np.linspace(3.0, 9.0, 61)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), (Y * np.tan(TILT)).ravel()])
    kw = dict(interp="linear", fov_factor=1.5)
    plain = gs.core.backproject([rec], pts, **kw)
    assert np.abs(plain).max() > 0
    for block in (30, 81):                     # blocks 15 m and 40 m long
        culled = gs.core.backproject([rec], pts, cull_block=block, grid_shape=X.shape, **kw)
        np.testing.assert_array_equal(culled, plain)


def test_culling_requires_a_field_of_view_and_a_grid(collection):
    _, records, track = collection
    pts, shape = seafloor_grid(track, 20, 20)
    with pytest.raises(ValueError, match="fov_factor"):
        gs.core.backproject(records[:1], pts, cull_block=4, grid_shape=shape)
    with pytest.raises(ValueError, match="grid_rows"):
        gs.core.backproject(records[:1], pts, cull_block=4, grid_shape=(7, 7), fov_factor=1.5)


def test_bistatic_culling_tests_transmitter_and_receiver_separately():
    """Projector far from the array: a block must be in both beams to be kept."""
    sonar = SonarModel.load("sas_auv", {"receive": {"window": {"range": [20.0, 50.0]}}})
    target = np.array([10.0, 30.0, 30.0 * np.tan(TILT)])
    # Transmitter 20 m ahead of the receiver; both look to starboard with 9.5 degree beams.
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target]),
                        tx_pose=make_pose(position=(20.0, 0.0, 0.0)),
                        environment={"sound_speed": C}, absorption_model="none", adc=False)
    recon.prepare([rec])
    x = np.linspace(-10.0, 30.0, 161)
    y = np.linspace(25.0, 40.0, 61)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), (Y * np.tan(TILT)).ravel()])
    kw = dict(interp="linear", fov_factor=1.5)
    plain, wsum = gs.core.backproject([rec], pts, return_weight_sum=True, **kw)
    stats = {}
    culled = gs.core.backproject([rec], pts, cull_block=8, grid_shape=X.shape, stats=stats, **kw)
    np.testing.assert_array_equal(culled, plain)
    # The two beams are 20 m apart and each about 7 m wide at this range: they do not
    # overlap, so no point is in both and (nearly) every block is culled.
    assert wsum.max() == 0
    assert stats["ping_blocks_kept"] <= 0.05 * stats["ping_blocks_total"]


def test_report_cpu_speedup(capsys):
    """Measure and print the culling speedup on this machine's CPU backend.

    The gain depends on how much of the image each ping can see. This collection is
    64 m long with a beam footprint of 8-14 m, so most (ping, block) pairs are out of beam,
    as in a real stripmap run.
    """
    _, records, track = stripmap(n_pings=200)
    pts, shape = seafloor_grid(track, 400, 240)
    kw = dict(interp="linear", fov_factor=1.5)

    def timed(**extra):
        best = np.inf
        for _ in range(3):          # ASASIN's protocol: run three times, keep the last/best
            t = time.perf_counter()
            gs.core.backproject(records, pts, **kw, **extra)
            best = min(best, time.perf_counter() - t)
        return best

    t_plain = timed()
    rows = []
    for block in (4, 8, 16, 32):
        stats = {}
        gs.core.backproject(records, pts, cull_block=block, grid_shape=shape, stats=stats, **kw)
        t = timed(cull_block=block, grid_shape=shape)
        rows.append((block, t_plain / t, stats["ping_blocks_kept"] / stats["ping_blocks_total"]))
    with capsys.disabled():
        print(f"\n[culling, CPU backend] {len(records)} pings x {len(pts)} points, no culling: {t_plain:.2f} s")
        for block, speedup, kept in rows:
            print(f"[culling, CPU backend] b={block:2d}: speedup {speedup:.2f}x, blocks kept {100 * kept:.0f}%")
    # The kernel now rejects out-of-beam (point, ping) pairs with a cheap float32 test
    # before any float64 work, which took most of what block culling used to gain on a
    # CPU. Culling must still never cost time.
    assert max(r[1] for r in rows) > 0.95
