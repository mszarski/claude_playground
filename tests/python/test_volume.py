"""Milestone 9: 3D near-field imaging of a buried object (ASASIN Sec. III-B).

Acceptance: the buried cylinder appears at the correct depth with refraction on, and
defocuses with refraction off.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import gpusonar as gs
from gpusonar import recon
from gpusonar.sim import Scene, SedimentProvider, make_pose, simulate_ping, snell_crossing
from gpusonar.sonar import SonarModel

core = gs.core
C_WATER, C_SED = 1500.0, 1700.0
HEIGHT = 1.5          # sonar height above the seabed
DEPTH = 0.30          # burial depth of the cylinder axis


# --- the refraction model itself -----------------------------------------------------------

@pytest.mark.parametrize("sensor, point", [
    ((0.0, 0.0, -1.5), (0.8, 0.3, 0.4)),
    ((0.2, -0.1, -2.5), (-1.5, 1.0, 0.05)),
    ((0.0, 0.0, -1.0), (0.0, 0.0, 0.5)),          # straight down: no bending
    ((0.0, 0.0, -1.0), (3.0, 0.0, 0.2)),          # shallow grazing
])
def test_refracted_time_is_the_least_time_path(sensor, point):
    s, p = np.array(sensor), np.array(point)
    dist = np.hypot(*(p[:2] - s[:2]))
    a, b = -s[2], p[2]

    def travel(u):
        return np.hypot(u, a) / C_WATER + np.hypot(dist - u, b) / C_SED

    best = minimize_scalar(travel, bounds=(0.0, max(dist, 1e-9)), method="bounded", options={"xatol": 1e-13})
    t_kernel = core.refracted_time(sensor, point, C_WATER, C_SED, 0.0)
    assert t_kernel == pytest.approx(best.fun, rel=1e-10)
    t_sim, cross = snell_crossing(s, p[None, :], C_WATER, C_SED, 0.0)
    assert t_sim[0] == pytest.approx(t_kernel, rel=1e-10)        # simulator and kernel agree
    # Snell's law at the crossing point: sin(theta_1) / c_1 = sin(theta_2) / c_2.
    if dist > 0:
        u = np.hypot(*(cross[0, :2] - s[:2]))
        sin1 = u / np.hypot(u, a)
        sin2 = (dist - u) / np.hypot(dist - u, b)
        assert sin1 / C_WATER == pytest.approx(sin2 / C_SED, rel=1e-7)


def test_points_in_the_water_take_the_straight_path():
    t = core.refracted_time((0.0, 0.0, -1.5), (1.0, 2.0, -0.2), C_WATER, C_SED, 0.0)
    assert t == pytest.approx(np.linalg.norm([1.0, 2.0, 1.3]) / C_WATER, rel=1e-14)


def test_refraction_flag_changes_nothing_above_the_interface():
    sonar = SonarModel.load("subbottom")
    target = np.array([[0.1, 0.0, -0.3]])            # in the water column
    rec = simulate_ping(sonar, make_pose(position=(0, 0, -HEIGHT)), Scene().add_points(target),
                        environment={"sound_speed": C_WATER}, absorption_model="none", adc=False)
    recon.prepare([rec])
    plain = core.backproject([rec], target, stop_and_hop=True)
    refr = core.backproject([rec], target, stop_and_hop=True, refraction=(0.0, C_SED))
    np.testing.assert_allclose(refr, plain, rtol=2e-4)
    assert np.abs(plain[0]) > 0


# --- buried cylinder -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def survey():
    """A 2D synthetic aperture: the sonar rastered over a 1.8 m square, 1.5 m above the bed."""
    sonar = SonarModel.load("subbottom")
    # A cylinder 0.5 m long lying along x, as a line of scatterers on its axis.
    cx = np.arange(-0.25, 0.2501, 0.02)
    cylinder = np.column_stack([cx, np.zeros_like(cx), np.full_like(cx, DEPTH)])
    scene = Scene().add_points(cylinder)
    provider = SedimentProvider(C_WATER, C_SED, interface_z=0.0)
    grid = np.arange(-0.9, 0.9001, 0.075)
    records = []
    for i, px in enumerate(grid):
        for py in grid:
            records.append(simulate_ping(
                sonar, make_pose(position=(px, py, -HEIGHT)), scene, provider=provider,
                environment={"sound_speed": C_WATER}, absorption_model="none", adc=False,
                sequence=len(records)))
    recon.prepare(records)
    x = np.arange(-0.5, 0.5001, 0.025)
    y = np.arange(-0.3, 0.3001, 0.025)
    depths = np.arange(0.10, 0.5001, 0.01)
    on = recon.image_volume(records, x, y, depths, interface_z=0.0, sediment_speed=C_SED)
    off = recon.image_volume(records, x, y, depths)
    return records, on, off


def test_cylinder_appears_at_the_correct_depth_with_refraction(survey):
    _, on, _ = survey
    x, y, z = on.peak()
    assert z == pytest.approx(DEPTH, abs=0.015)
    assert abs(y) <= 0.025 and abs(x) <= 0.26         # on the cylinder's axis, within its length
    # The whole cylinder is there: along its axis the depth slice stays bright.
    layer = on.depth_slice(DEPTH)
    iy = int(np.argmin(np.abs(on.y)))
    along = layer[:, iy]
    inside = np.abs(on.x) <= 0.2
    assert along[inside].min() > 0.5 * along.max()
    assert along[np.abs(on.x) >= 0.4].max() < 0.3 * along.max()


def test_cylinder_is_misplaced_and_defocused_without_refraction(survey):
    _, on, off = survey
    _, _, z_off = off.peak()
    # Treating the sediment as water puts the object too shallow: sound is faster down
    # there, so the same echo time is mapped to less depth.
    expected_shift = DEPTH * (1.0 - C_WATER / C_SED)          # 3.5 cm at nadir
    assert DEPTH - z_off == pytest.approx(expected_shift, abs=0.015)
    assert DEPTH - z_off > 0.02
    # And it does not focus as well: the peak is lower than with the right model.
    assert np.abs(off.data).max() < 0.97 * np.abs(on.data).max()
    # At the true depth the uncorrected image has lost a good part of the response.
    assert off.depth_slice(DEPTH).max() < 0.8 * on.depth_slice(DEPTH).max()


def test_slices_and_maximum_intensity_projections(survey):
    _, on, _ = survey
    assert on.data.shape == (len(on.depths), len(on.x), len(on.y))
    top = on.mip("depth")                 # plan view
    side = on.mip("y")                    # depth against x
    assert top.shape == (len(on.x), len(on.y)) and side.shape == (len(on.depths), len(on.x))
    assert top.max() == pytest.approx(np.abs(on.data).max())
    # The side view shows the object at its depth, the plan view shows its length along x.
    iz, ix = np.unravel_index(np.argmax(side), side.shape)
    assert on.depths[iz] == pytest.approx(DEPTH, abs=0.015)
    bright = top.max(axis=1) > 0.5 * top.max()
    length = on.x[bright].max() - on.x[bright].min()
    assert length == pytest.approx(0.5, abs=0.12)
    np.testing.assert_array_equal(on.depth_slice(0.2), np.abs(on.data[10]))


def test_bistatic_culling_leaves_a_volume_layer_unchanged(survey):
    records, _, _ = survey
    # Give the elements a finite beam so there is something to cull, then compare.
    recs = [r.copy() for r in records[::9]]
    for r in recs:
        geo = r.geometry
        elements = geo.rx
        for e in elements:
            e.beamwidth_az = e.beamwidth_el = 0.5
        geo.rx = elements
        tx = geo.tx
        tx.beamwidth_az = tx.beamwidth_el = 0.5
        geo.tx = tx
        r.geometry = geo
    x = np.arange(-1.0, 1.0001, 0.05)
    plain = recon.image_volume(recs, x, x, [DEPTH], interface_z=0.0, sediment_speed=C_SED, fov_factor=1.5)
    culled = recon.image_volume(recs, x, x, [DEPTH], interface_z=0.0, sediment_speed=C_SED, fov_factor=1.5,
                                cull_block=4)
    np.testing.assert_array_equal(culled.data, plain.data)
    assert np.abs(plain.data).max() > 0
