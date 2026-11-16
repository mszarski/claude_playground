"""Milestone 7: SAS with known navigation.

Acceptance: rail scenario and stripmap scenario; along-track resolution within 10% of D/2.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import recon
from gpusonar.metrics import width_at_level
from gpusonar.sim import run_scenario


@pytest.fixture(scope="module")
def stripmap():
    result = run_scenario("stripmap_points")
    recon.prepare(result.records)
    return result


def test_grid_follows_the_paper_geometry(stripmap):
    records = stripmap.records
    bw = records[0].geometry.rx[0].beamwidth_az
    grid = recon.stripmap_grid(records, (36.0, 48.0), 0.05)
    track = records[-1].rx_positions().mean(axis=0)[0] - records[0].rx_positions().mean(axis=0)[0]
    margin = 48.0 * np.tan(bw / 2)
    # The grid does not start at the first ping: it starts where the far corner has the
    # whole beam (ASASIN Fig. 4), and ends symmetrically.
    assert grid.x[0] == pytest.approx(margin)
    assert grid.x[-1] == pytest.approx(track - margin, abs=0.05)
    assert grid.points.shape == (len(grid.x), len(grid.y), 3)
    np.testing.assert_allclose(grid.points[..., 2], 0.0)            # on the seafloor, z = 0
    assert records[0].rx_positions()[:, 2].max() < 0                # the sonar is above it: negative z
    # Without the margin the grid spans the whole track.
    loose = recon.stripmap_grid(records, (36.0, 48.0), 0.05, full_support=False)
    assert loose.x[0] == 0.0 and loose.x[-1] == pytest.approx(track, abs=0.05)
    with pytest.raises(ValueError, match="full angular support"):
        recon.stripmap_grid(records[:10], (36.0, 48.0), 0.05)


def test_stripmap_targets_focus_where_they_are(stripmap):
    grid = recon.stripmap_grid(stripmap.records, (36.0, 47.0), (0.02, 0.04))
    stats = {}
    img = np.abs(recon.sas_image(stripmap.records, grid.points, interp="linear", stats=stats))
    assert img.shape == grid.shape
    origin_x = stripmap.records[0].rx_positions().mean(axis=0)[0]
    for tx, ty, _ in stripmap.scene.positions:
        ix = np.argmin(np.abs(grid.x + origin_x - tx))
        iy = np.argmin(np.abs(grid.y - ty))
        window = img[max(ix - 30, 0):ix + 31, max(iy - 15, 0):iy + 16]
        jx, jy = np.unravel_index(np.argmax(window), window.shape)
        assert abs((jx - min(ix, 30)) * 0.02) <= 0.04          # along-track, within two pixels
        assert abs((jy - min(iy, 15)) * 0.04) <= 0.04          # ground range, within one pixel
        assert window.max() > 0.5 * img.max()
    # Culling did real work on this geometry.
    assert stats["ping_blocks_kept"] < 0.8 * stats["ping_blocks_total"]


def along_track_cut(records, target, half=0.25, step=0.001, **kw):
    x = target[0] + np.arange(-half, half + step / 2, step)
    pts = np.column_stack([x, np.full_like(x, target[1]), np.full_like(x, target[2])])[:, None, :]
    img = recon.sas_image(records, pts, cull_block=0, **kw)[:, 0]
    return x, img


def test_along_track_resolution_is_half_the_element_length(stripmap):
    """The stripmap result: resolution D/2, independent of range and wavelength."""
    D = 0.04                                         # element length along track (sas_auv)
    target = stripmap.scene.positions[0]
    x, img = along_track_cut(stripmap.records, target)
    w3 = width_at_level(x, np.abs(img), -3.0)
    assert abs(x[np.argmax(np.abs(img))] - target[0]) < 0.003
    assert w3 == pytest.approx(D / 2, rel=0.10), w3


def test_resolution_does_not_depend_on_range(stripmap):
    widths = []
    for target in stripmap.scene.positions:
        x, img = along_track_cut(stripmap.records, target)
        widths.append(width_at_level(x, np.abs(img), -3.0))
    assert max(widths) / min(widths) < 1.08
    # A real aperture of the same 0.64 m array would give about lambda R / L = 10 cm here.
    assert max(widths) < 0.03


def test_processed_beam_sets_the_resolution(stripmap):
    """D/2 belongs to a processed extent of lambda/D. Less beam is coarser, more is finer."""
    target = stripmap.scene.positions[0]
    widths = {}
    for factor in (0.6, 1.13, 2.26):
        x, img = along_track_cut(stripmap.records, target, fov_factor=factor)
        widths[factor] = width_at_level(x, np.abs(img), -3.0)
    assert widths[0.6] > 1.5 * widths[1.13]
    assert widths[2.26] < 0.85 * widths[1.13]


def test_cross_track_resolution_is_set_by_bandwidth(stripmap):
    target = stripmap.scene.positions[0]
    y = target[1] + np.arange(-0.15, 0.1501, 0.001)
    pts = np.column_stack([np.full_like(y, target[0]), y, np.zeros_like(y)])[:, None, :]
    img = np.abs(recon.sas_image(stripmap.records, pts, cull_block=0)[:, 0])
    w3 = width_at_level(y, img, -3.0)
    slant = np.linalg.norm(target - np.array([target[0], 0.0, -15.0]))
    ground_factor = slant / target[1]               # ground range resolution = slant / cos(grazing)
    theory = 0.886 * 1500.0 / (2 * 40e3) * ground_factor   # flat 40 kHz sweep (10% Tukey taper)
    assert w3 == pytest.approx(theory, rel=0.10)


# --- rail ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rail():
    result = run_scenario("rail_points")
    recon.prepare(result.records)
    return result


def test_rail_resolution_is_aperture_limited(rail):
    """On a 0.3 m rail the element beam (about 110 degrees) is far wider than the angle
    the rail subtends, so resolution is set by the rail length: 0.886 lambda R / (2 L)."""
    records = rail.records
    target = rail.scene.positions[0]
    assert all(np.allclose(r.geometry.rx_pose.velocity, 0.0) for r in records)   # stop and go
    centres = np.array([r.rx_positions().mean(axis=0) for r in records])
    assert np.diff(centres[:, 0]) == pytest.approx(0.002)
    x, img = along_track_cut(records, target, half=0.01, step=2e-5, fov_factor=0.0)
    w3 = width_at_level(x, np.abs(img), -3.0)
    wavelength = 1480.0 / 2.72e6
    array_length = 63 * 0.3e-3
    # Transmitter fixed at the array centre, receivers spread over the array: the two-way
    # phase centres span the rail travel plus half the array length.
    L = (centres[-1, 0] - centres[0, 0]) + array_length / 2
    R = np.linalg.norm(target - centres[len(centres) // 2])
    theory = 0.886 * wavelength * R / (2 * L)
    assert abs(x[np.argmax(np.abs(img))] - target[0]) < 1e-4
    assert w3 == pytest.approx(theory, rel=0.10), (w3, theory)


def test_rail_resolves_two_targets(rail):
    records = rail.records
    a, b = rail.scene.positions
    x = np.arange(0.12, 0.16, 2e-4)
    y = np.arange(1.82, 1.90, 5e-4)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    img = np.abs(recon.sas_image(records, pts, interp="linear", fov_factor=0.0, cull_block=0))
    for t in (a, b):
        ix, iy = np.argmin(np.abs(x - t[0])), np.argmin(np.abs(y - t[1]))
        local = img[ix - 5:ix + 6, iy - 5:iy + 6]
        assert local.max() > 0.5 * img.max()
    # Between them the image falls well below the peaks.
    mid_x, mid_y = np.argmin(np.abs(x - 0.14)), np.argmin(np.abs(y - 1.86))
    assert img[mid_x, mid_y] < 0.1 * img.max()
