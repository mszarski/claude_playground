"""sonar-0nsi: image sources off tilted planes and walls given as {point, normal} (opt-in), against closed forms.

`ImageSourceProvider(bottom_slope=[dz/dx, dz/dy])` tilts the bottom; a wall {point, normal, coefficient[, bounds]}
is any plane. Either turns on the general construction (`propagation._PlaneImages`), which checks every image
path for validity. Without them nothing changes (the last tests here)."""

import warnings

import numpy as np
import pytest

from gpusonar.sim.propagation import ImageSourceProvider, rayleigh_reflection, two_way_paths

C = 1500.0
SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}


def pytest_generate_tests(metafunc):
    """sonar-k9v7: both engines have the construction, so every test that does not choose its engine runs on each
    (the default engine set through GPUSONAR_SIM_ENGINE)."""
    if "default_engine" in metafunc.fixturenames and "engine" not in metafunc.fixturenames:
        metafunc.parametrize("default_engine", ["cpp", "numpy"], indirect=True)


@pytest.fixture(autouse=True)
def default_engine(request, monkeypatch):
    name = getattr(request, "param", None)
    if name is not None:
        monkeypatch.setenv("GPUSONAR_SIM_ENGINE", name)
    return name


def bed_normal(slope):
    n = np.array([slope[0], slope[1], -1.0])
    return n / np.linalg.norm(n)


def mirror(x, point, normal):
    return x - 2.0 * ((x - point) @ normal)[..., None] * normal


def column(rays, i, n_surface, n_bottom):
    cols = np.flatnonzero((rays.n_surface[i] == n_surface) & (rays.n_bottom[i] == n_bottom) & rays.valid[i])
    assert len(cols) == 1, cols
    return int(cols[0])


def sorted_rays(rays, i):
    """The valid rays of point i ordered by delay (then launch, bounce counts): (delay, amplitude, launch, arrival, n_surface, n_bottom)."""
    keep = np.flatnonzero(rays.valid[i])
    o = keep[np.lexsort((rays.n_bottom[i, keep], rays.n_surface[i, keep], np.round(rays.launch[i, keep, 2], 9),
                         np.round(rays.delay[i, keep], 15)))]
    return (rays.delay[i, o], rays.amplitude[i, o], rays.launch[i, o], rays.arrival[i, o], rays.n_surface[i, o],
            rays.n_bottom[i, o])


# --- (i) one tilted plane --------------------------------------------------------------------------------------

@pytest.mark.parametrize("alpha_deg, grazing_deg, table_deg", [(5.0, 20.0, 3.6), (5.0, 45.0, 9.9), (1.0, 5.0, 0.17),
                                                               (12.0, 30.0, 13.2), (2.0, 10.0, 0.71)])
def test_one_bounce_off_a_cross_slope_turns_the_ray_by_the_formula(alpha_deg, grazing_deg, table_deg):
    """docs/tasks/PROPAGATION_3D.md section 3 (ours): a ray along x at grazing g meets a bed whose depth grows across
    the bearing with slope tan(alpha); it leaves its vertical plane at the azimuth atan(tan g sin 2 alpha), and the
    image is the mirror of the field point in the tilted plane. Here the image is placed first (so the ray leaves at
    exactly g in the x-z plane) and the field point is its mirror."""
    a, g = np.radians(alpha_deg), np.radians(grazing_deg)
    zb = 40.0
    p = ImageSourceProvider(C, surface_z=None, bottom_z=zb, bottom=SAND, max_order=1, bottom_slope=[0.0, np.tan(a)])
    source = np.array([0.0, 0.0, 10.0])
    n = bed_normal([0.0, np.tan(a)])
    np.testing.assert_allclose(n, [0.0, np.sin(a), -np.cos(a)], atol=1e-15)
    length = 1.5 * (zb - source[2]) / np.sin(g)                  # the image lies below the bed
    image = source + length * np.array([np.cos(g), 0.0, np.sin(g)])
    point = mirror(image, np.array([0.0, 0.0, zb]), n)
    assert (point - [0.0, 0.0, zb]) @ n > 0.0                               # in the water
    rays = p.eigenrays(source, point[None])
    j = column(rays, 0, 0, 1)
    assert rays.delay[0, j] == pytest.approx(length / C, rel=1e-13)
    np.testing.assert_allclose(rays.launch[0, j], [np.cos(g), 0.0, np.sin(g)], atol=1e-13)
    np.testing.assert_allclose(rays.apparent_positions(source)[0, j], image, atol=1e-10)
    # Direction after the bounce: (cos g, sin g sin 2a, -sin g cos 2a) (section 3), and the same from the geometry:
    # the specular point is where the line to the image meets the bed, and the last leg runs from there to the point.
    expected = np.array([np.cos(g), np.sin(g) * np.sin(2 * a), -np.sin(g) * np.cos(2 * a)])
    np.testing.assert_allclose(rays.arrival[0, j], expected, atol=1e-13)
    t = (zb - source[2] - source[1] * np.tan(a)) / ((image - source)[2] - (image - source)[1] * np.tan(a))
    specular = source + t * (image - source)
    assert specular[2] == pytest.approx(zb + specular[1] * np.tan(a), abs=1e-10)
    leg = (point - specular) / np.linalg.norm(point - specular)
    np.testing.assert_allclose(rays.arrival[0, j], leg, atol=1e-12)
    phi = np.degrees(np.arctan2(rays.arrival[0, j, 1], rays.arrival[0, j, 0]))
    assert phi == pytest.approx(np.degrees(np.arctan(np.tan(g) * np.sin(2 * a))), abs=1e-10)
    assert phi == pytest.approx(table_deg, abs=0.05)                         # the table of section 3
    # The coefficient is the Rayleigh coefficient at the grazing angle to the TILTED bed, not to the horizontal.
    local = np.arcsin(abs(np.array([np.cos(g), 0.0, np.sin(g)]) @ n))
    assert rays.amplitude[0, j] * length == pytest.approx(complex(rayleigh_reflection(local, C, **SAND)), rel=1e-12)
    assert np.sin(local) == pytest.approx(np.sin(g) * np.cos(a), rel=1e-12)    # the cross-slope lowers it by cos(alpha)


def test_a_tilted_bed_moves_the_ghost_across_the_bearing():
    """The bed-bounce image of a point h above a bed tilted by alpha across x lies 2 h sin(alpha) across from the
    point: the bearing the array sees for that path (the direction of the image) differs from the object's."""
    a = np.radians(5.0)
    p = ImageSourceProvider(C, surface_z=0.0, bottom_z=12.0, bottom=SAND, max_order=1, bottom_slope=[0.0, np.tan(a)])
    source = np.array([0.0, 0.0, 3.0])
    point = np.array([45.0, 8.0, 6.0])
    h = (point - [0.0, 0.0, 12.0]) @ bed_normal([0.0, np.tan(a)])
    rays = p.eigenrays(source, point[None])
    image = rays.apparent_positions(source)[0, column(rays, 0, 0, 1)]
    assert image[1] - point[1] == pytest.approx(-2.0 * h * np.sin(a), rel=1e-12)
    flat = ImageSourceProvider(C, surface_z=0.0, bottom_z=12.0, bottom=SAND, max_order=1).eigenrays(source, point[None])
    assert flat.apparent_positions(source)[0, column(flat, 0, 0, 1)][1] == pytest.approx(point[1], abs=1e-12)


# --- (ii) zero tilt is the horizontal bottom -------------------------------------------------------------------

def water_points(seed, n=40, depth=60.0, x=(20.0, 400.0)):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(*x, n), rng.uniform(-80.0, 80.0, n), rng.uniform(0.5, depth - 0.5, n)])


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
@pytest.mark.parametrize("options", [dict(bottom=SAND, max_order=3), dict(bottom_coefficient=0.5 - 0.2j, max_order=2),
                                     dict(bottom=SAND), dict(bottom=SAND, surface_z=None, max_order=2)])
def test_zero_tilt_reproduces_the_flat_bottom(engine, options):
    """Every ray of the flat-bottom series is found, and no other: delays, launch and arrival directions and bounce
    counts to the bit; amplitudes to rounding (the coefficient is a product over the bounces in path order, where the
    flat series raises each coefficient to its count). Points on the bed are included."""
    options = dict(dict(surface_z=0.0, bottom_z=60.0), **options)
    source = np.array([0.0, 0.0, 20.0])
    pts = np.vstack([water_points(3), [[150.0, 10.0, 60.0], [80.0, -5.0, 0.0]]])
    flat = ImageSourceProvider(C, engine=engine, **options)
    tilted = ImageSourceProvider(C, engine=engine, bottom_slope=[0.0, 0.0], **options)
    a, b = flat.eigenrays(source, pts), tilted.eigenrays(source, pts)
    assert tilted.general and not flat.general
    assert flat.order_used == tilted.order_used                              # the energy rule picks the same order
    assert a.valid.sum() == b.valid.sum() == a.delay.size
    worst = 0.0
    for i in range(len(pts)):
        ra, rb = sorted_rays(a, i), sorted_rays(b, i)
        for k in (0, 2, 3, 4, 5):
            np.testing.assert_array_equal(ra[k], rb[k])
        np.testing.assert_allclose(rb[1], ra[1], rtol=1e-14, atol=0.0)
        worst = max(worst, float(np.max(np.abs(rb[1] - ra[1]) / np.abs(ra[1]))))
    assert worst < 1e-14


# --- (iii) a wall turned with the world -----------------------------------------------------------------------

def rotation_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("deg", [0.0, 30.0, 117.0])
def test_a_rotated_wall_is_the_same_wall_in_rotated_coordinates(deg):
    rot = rotation_z(deg)
    wall = {"point": [40.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0], "coefficient": 0.7 + 0.1j}
    turned = {"point": list(rot @ wall["point"]), "normal": list(rot @ wall["normal"]), "coefficient": wall["coefficient"]}
    kw = dict(surface_z=0.0, bottom_z=30.0, bottom=SAND, max_order=2, wall_order=1)
    source = np.array([0.0, 3.0, 8.0])
    pts = water_points(5, depth=30.0, x=(-30.0, 39.0))
    a = ImageSourceProvider(C, walls=[wall], **kw).eigenrays(source, pts)
    b = ImageSourceProvider(C, walls=[turned], **kw).eigenrays(rot @ source, pts @ rot.T)
    np.testing.assert_array_equal(a.valid, b.valid)
    np.testing.assert_allclose(b.delay, a.delay, rtol=1e-12)
    np.testing.assert_allclose(b.amplitude, a.amplitude, rtol=1e-11, atol=1e-18)
    np.testing.assert_allclose(b.launch, a.launch @ rot.T, atol=1e-11)
    np.testing.assert_allclose(b.arrival, a.arrival @ rot.T, atol=1e-11)


# Axis-aligned walls of a box -30 < x < 40, -30 < y < 35 (water inside), each in both forms.
AXIS_WALLS = {"x40": ({"axis": 0, "position": 40.0, "coefficient": 0.7},
                      {"point": [40.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0], "coefficient": 0.7}),
              "y-30": ({"axis": 1, "position": -30.0, "coefficient": 0.6 + 0.2j},
                       {"point": [0.0, -30.0, 0.0], "normal": [0.0, 1.0, 0.0], "coefficient": 0.6 + 0.2j}),
              "x-30": ({"axis": 0, "position": -30.0, "coefficient": -0.5j},
                       {"point": [-30.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0], "coefficient": -0.5j})}


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
@pytest.mark.parametrize("walls", [("x40",), ("x40", "y-30"), ("x40", "x-30")])
@pytest.mark.parametrize("options", [dict(surface_z=0.0, bottom_z=30.0, bottom=SAND, max_order=2),
                                     dict(surface_z=0.0, bottom_z=30.0, bottom_coefficient=0.5 - 0.2j, max_order=3),
                                     dict(surface_z=0.0, bottom_z=30.0, bottom=SAND),
                                     dict(surface_z=None, bottom_z=30.0, bottom_coefficient=0.4, max_order=1),
                                     dict(surface_z=None)])
def test_an_axis_wall_gives_the_rays_of_the_same_wall_as_a_plane(engine, walls, options):
    """sonar-8qn9 (D-201's finding): the same axis-aligned walls in both forms. The {axis, position} form adds each
    wall once to every image of the surface and bottom series without checks; the general form finds the one order of
    bounces that is valid for each. In a box with the points and the source inside, every ray of the axis form is a
    valid ray of the general form and the general form has no other valid ray: delays, launch and ARRIVAL directions
    and bounce counts to the bit, amplitudes to rounding (a product over the bounces in path order where the axis form
    raises each coefficient to its count). Before sonar-8qn9 the axis form mirrored only the vertical part of the
    arrival direction (both engines), so a wall path arrived with the wrong horizontal direction. Wall-only paths
    (`surface_z: None` and no bottom), paths mixing walls with surface and bottom bounces, one and two walls (at a
    right angle, and parallel)."""
    source = np.array([0.0, 3.0, 8.0])
    pts = water_points(6, depth=30.0, x=(-29.0, 39.0))
    pts[:, 1] = np.clip(pts[:, 1], -29.0, 34.0)
    old = ImageSourceProvider(C, engine=engine, walls=[AXIS_WALLS[w][0] for w in walls], **options)
    new = ImageSourceProvider(C, engine=engine, walls=[AXIS_WALLS[w][1] for w in walls], wall_order=1, **options)
    a, b = old.eigenrays(source, pts), new.eigenrays(source, pts)
    assert new.general and not old.general and old.order_used == new.order_used
    assert a.valid.all() and b.valid.sum() == a.valid.size
    n_walled = 0
    for i in range(len(pts)):
        ro, rn = sorted_rays(a, i), sorted_rays(b, i)
        for k in (0, 2, 3, 4, 5):
            np.testing.assert_array_equal(rn[k], ro[k])
        np.testing.assert_allclose(rn[1], ro[1], rtol=1e-14, atol=0.0)
        n_walled += int(np.sum(ro[3][:, 0] != ro[2][:, 0]) + np.sum(ro[3][:, 1] != ro[2][:, 1]))
    assert n_walled >= len(pts) * len(walls)                       # wall paths are among them, reversed horizontally


# --- (iv) a corner and a wedge ---------------------------------------------------------------------------------

@pytest.mark.parametrize("wall_order", [2, 3, 5])
def test_a_right_angle_corner_gives_the_corner_reflector_image(wall_order):
    """Two perpendicular walls: three images (one off each wall, one off both: the point turned 180 degrees about
    the corner's edge) whatever the order allowed; a monostatic path off both returns along its own line, reversed
    in the horizontal."""
    walls = [{"point": [40.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0], "coefficient": 0.8},
             {"point": [0.0, 30.0, 0.0], "normal": [0.0, -1.0, 0.0], "coefficient": 0.6j}]
    p = ImageSourceProvider(C, surface_z=None, walls=walls, wall_order=wall_order)
    source = np.array([2.0, -4.0, 5.0])
    pts = np.array([[20.0, 10.0, 7.0], [35.0, 25.0, 2.0], [-10.0, -20.0, 5.0]])
    rays = p.eigenrays(source, pts)
    assert (rays.valid.sum(axis=1) == 4).all()
    for i, x in enumerate(pts):
        corner = np.array([80.0 - x[0], 60.0 - x[1], x[2]])
        d = np.linalg.norm(corner - source)
        j = int(np.flatnonzero(rays.valid[i] & np.isclose(rays.delay[i], d / C, rtol=1e-13))[0])
        assert rays.amplitude[i, j] == pytest.approx(0.8 * 0.6j / d, rel=1e-13)
        u = (corner - source) / d
        np.testing.assert_allclose(rays.arrival[i, j], [-u[0], -u[1], u[2]], atol=1e-14)
    mono = p.eigenrays(pts[0], pts[0][None] + [[0.0, 0.0, 1e-9]])
    k = int(np.argmax(np.where(mono.valid[0], mono.delay[0], 0.0)))
    np.testing.assert_allclose(mono.arrival[0, k, :2], -mono.launch[0, k, :2], atol=1e-6)


@pytest.mark.parametrize("wedge_deg", [30.0, 45.0, 20.0])
def test_a_wedge_has_finitely_many_images_on_the_circle_about_its_edge(wedge_deg):
    """Surface and a bed rising towards -x make a wedge of angle theta with its edge along y. Every image lies at the
    point's distance from the edge (a mirror keeps it), and however many bounces are allowed, at most 360/theta rays
    (direct included) are valid: the higher orders all fail the checks."""
    theta = np.radians(wedge_deg)
    zb = 50.0
    slope = np.tan(theta)                                 # z = zb + slope x meets the surface at x0 = -zb / slope
    x0 = -zb / slope
    p = ImageSourceProvider(C, surface_z=0.0, bottom_z=zb, bottom_coefficient=0.9, max_order=24, bottom_slope=[slope, 0.0])
    rng = np.random.default_rng(7)
    xs = rng.uniform(x0 + 20.0, x0 + 150.0, 12)
    pts = np.column_stack([xs, rng.uniform(-20, 20, 12), rng.uniform(0.05, 0.95, 12) * (zb + slope * xs)])
    source = np.array([x0 + 90.0, 0.0, 0.3 * (zb + slope * (x0 + 90.0))])
    rays = p.eigenrays(source, pts)
    n_valid = rays.valid.sum(axis=1)
    assert (n_valid <= int(np.floor(360.0 / wedge_deg))).all() and (n_valid >= 3).all()
    images = rays.apparent_positions(source)
    for i, x in enumerate(pts):
        radius = np.hypot(x[0] - x0, x[2])
        r_img = np.hypot(images[i, rays.valid[i], 0] - x0, images[i, rays.valid[i], 2])
        np.testing.assert_allclose(r_img, radius, rtol=1e-9)
        np.testing.assert_allclose(images[i, rays.valid[i], 1], x[1], atol=1e-9)


# --- (v) reciprocity -------------------------------------------------------------------------------------------

def test_reciprocity_with_a_tilted_bed_and_walls():
    walls = [{"point": [60.0, 0.0, 0.0], "normal": [-np.cos(0.3), -np.sin(0.3), 0.0], "coefficient": 0.7},
             {"point": [0.0, -25.0, 0.0], "normal": [0.0, 1.0, 0.0], "coefficient": 0.5, "bounds": {"x": [-10.0, 40.0]}}]
    p = ImageSourceProvider(C, surface_z=0.0, bottom_z=20.0, bottom=SAND, max_order=2, wall_order=2,
                            bottom_slope=[0.02, -0.06], walls=walls)
    a = np.array([5.0, 3.0, 6.0])
    rng = np.random.default_rng(9)
    for b in np.column_stack([rng.uniform(-5, 50, 8), rng.uniform(-20, 15, 8), rng.uniform(1.0, 17.0, 8)]):
        ab, ba = p.eigenrays(a, b[None]), p.eigenrays(b, a[None])
        r1, r2 = sorted_rays(ab, 0), sorted_rays(ba, 0)
        assert len(r1[0]) == len(r2[0]) >= 4
        np.testing.assert_allclose(r1[0], r2[0], rtol=1e-12)
        np.testing.assert_allclose(r1[1], r2[1], rtol=1e-10)
        np.testing.assert_allclose(r1[3], -r2[2], atol=1e-10)                # arrival one way = minus launch the other
        np.testing.assert_array_equal(r1[4], r2[4])
        np.testing.assert_array_equal(r1[5], r2[5])


# --- validity: bounded walls, points out of the water ----------------------------------------------------------

def test_a_bounded_wall_reflects_only_on_its_face_and_blocks_only_there():
    wall = {"point": [30.0, 0.0, 0.0], "normal": [-1.0, 0.0, 0.0], "coefficient": 0.9, "bounds": {"y": [-5.0, 5.0]}}
    p = ImageSourceProvider(C, surface_z=None, walls=[wall])
    source = np.array([0.0, 0.0, 5.0])
    pts = np.array([[10.0, 1.0, 5.0],        # specular point at y = 0.67: on the face
                    [10.0, 12.0, 5.0],       # specular point at y = 9: off the face
                    [40.0, 1.0, 5.0],        # behind the wall: direct path blocked
                    [40.0, 30.0, 5.0]])      # behind the wall's line but beside its face: direct path clear
    rays = p.eigenrays(source, pts)
    assert rays.valid.tolist() == [[True, True], [True, False], [False, False], [True, False]]
    assert (rays.amplitude[~rays.valid] == 0).all()


def test_points_outside_the_water_get_no_paths():
    p = ImageSourceProvider(C, surface_z=0.0, bottom_z=20.0, bottom=SAND, max_order=1, bottom_slope=[0.0, 0.1])
    rays = p.eigenrays([0.0, 0.0, 5.0], [[30.0, 0.0, 19.0], [30.0, -50.0, 19.0], [30.0, 0.0, -1.0]])
    assert rays.valid[0].all() and not rays.valid[1].any() and not rays.valid[2].any()   # bed at y = -50 is 15 m deep


# --- (vii) the C++ construction against the NumPy one (sonar-k9v7) ---------------------------------------------

# The C++ construction (libs/sim paths.cpp `plane_image_eigenrays`) follows the NumPy arithmetic operation for
# operation except the dot products, which NumPy hands to BLAS (another summation order, fused multiply-adds on
# some machines). Bounds, with the largest differences measured over the random cases below on the development
# machine (OpenBLAS 0.3.34, Haswell kernels) in brackets: delays 1e-14 relative (2.7e-15), directions 1e-14 absolute
# (2.3e-15), amplitudes 1e-12 relative (3.4e-15; the bound is wider for a Rayleigh coefficient near its zero, where
# the error is relative to the terms); validity and bounce counts exactly (a point within rounding of a validity
# threshold would differ: none here). 86 % of the delays and 47 % of the arrival directions are bit-identical.
PARITY = dict(delay=1e-14, direction=1e-14, amplitude=1e-12)


def assert_engines_agree(kw, source, pts, c=C):
    """The same provider on both engines: every field, rays in the same order."""
    a = ImageSourceProvider(c, engine="cpp", **kw).eigenrays(source, pts)
    b = ImageSourceProvider(c, engine="numpy", **kw).eigenrays(source, pts)
    assert a.delay.shape == b.delay.shape
    for f in ("valid", "n_surface", "n_bottom"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f), err_msg=f)
    np.testing.assert_allclose(a.delay, b.delay, rtol=PARITY["delay"], atol=0.0)
    np.testing.assert_allclose(a.launch, b.launch, rtol=0.0, atol=PARITY["direction"])
    np.testing.assert_allclose(a.arrival, b.arrival, rtol=0.0, atol=PARITY["direction"])
    np.testing.assert_allclose(a.amplitude, b.amplitude, rtol=PARITY["amplitude"], atol=0.0)
    assert (a.amplitude[~a.valid] == 0).all()
    return a, b


def random_geometry(rng):
    """A random configuration of the general construction: a surface or none, a bed tilted in x and y, up to three
    walls facing the origin (some leaning, some bounded), field points in and out of the water."""
    kw = dict(surface_z=0.0 if rng.uniform() < 0.8 else None, bottom_z=float(rng.uniform(15.0, 60.0)),
              bottom_slope=[float(v) for v in rng.uniform(-0.15, 0.15, 2)] if rng.uniform() < 0.8 else [0.0, 0.0],
              max_order=int(rng.integers(0, 4)), wall_order=int(rng.integers(0, 3)))
    if rng.uniform() < 0.5:
        kw["bottom"] = SAND
    else:
        kw["bottom_coefficient"] = complex(rng.uniform(0.2, 0.9), rng.uniform(-0.3, 0.3))
    walls = []
    for _ in range(int(rng.integers(0, 4))):
        az = rng.uniform(0.0, 2.0 * np.pi)
        lean = float(rng.normal(0.0, 0.1)) if rng.uniform() < 0.3 else 0.0
        dist = rng.uniform(20.0, 80.0)
        wall = {"point": [-dist * np.cos(az), -dist * np.sin(az), 0.0], "normal": [np.cos(az), np.sin(az), lean],
                "coefficient": complex(rng.uniform(0.3, 0.95), rng.uniform(-0.2, 0.2))}
        if rng.uniform() < 0.4:
            axis = "xyz"[int(rng.integers(0, 3))]
            lo = rng.uniform(-40.0, 10.0) if axis != "z" else rng.uniform(0.0, 10.0)
            wall["bounds"] = {axis: [float(lo), float(lo + rng.uniform(5.0, 50.0))]}
        walls.append(wall)
    kw["walls"] = walls
    pts = np.column_stack([rng.uniform(-70.0, 70.0, 50), rng.uniform(-70.0, 70.0, 50),
                           rng.uniform(-2.0, kw["bottom_z"] + 8.0, 50)])
    source = np.array([rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0), rng.uniform(1.0, 10.0)])
    return kw, source, pts


def test_the_engines_agree_on_random_geometries():
    """sonar-k9v7: 200 random configurations (fixed seed) of planes, bounds, orders and points; the C++ construction
    equals the NumPy one to PARITY in every field, invalid rays included. Some rays are valid and some not."""
    rng = np.random.default_rng(20261126)
    n_valid = n_invalid = n_walls = n_bounded = 0
    for _ in range(200):
        kw, source, pts = random_geometry(rng)
        a, _ = assert_engines_agree(kw, source, pts)
        n_valid += int(a.valid.sum())
        n_invalid += int((~a.valid).sum())
        n_walls += len(kw["walls"])
        n_bounded += sum("bounds" in w for w in kw["walls"])
    assert n_valid > 10000 and n_invalid > 10000 and n_walls > 200 and n_bounded > 50


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_the_cpp_engine_computes_the_construction_in_cpp(engine, monkeypatch):
    """sonar-k9v7: on the C++ engine the provider calls the C++ construction; on the NumPy engine it does not."""
    from gpusonar import _gpusonar as core
    used = []
    real = core.sim_plane_image_eigenrays
    monkeypatch.setattr(core, "sim_plane_image_eigenrays", lambda *a, **k: used.append(1) or real(*a, **k))
    p = ImageSourceProvider(C, engine=engine, surface_z=0.0, bottom_z=30.0, bottom=SAND, max_order=1,
                            bottom_slope=[0.0, 0.05])
    p.eigenrays([0.0, 0.0, 5.0], water_points(1, depth=30.0))
    assert len(used) == (1 if engine == "cpp" else 0)


@pytest.mark.parametrize("name", ["fls_120k_quay_multipath", "pool_fls_p4_2v_walls"])
def test_the_engines_agree_on_the_scenes_with_planes(name):
    """The scattering points of the two shipped scenes that use the general construction, from the sonar's
    position, on both engines."""
    from gpusonar.config import load_scenario_config
    from gpusonar.sim.scene import build_scene
    cfg = load_scenario_config(name)
    pts = build_scene(cfg["scene"], np.random.default_rng(0)).at_time(0.0)
    kw, c = dict(cfg["propagation"]), cfg["environment"]["sound_speed"]
    assert ImageSourceProvider(c, **kw).general
    a, _ = assert_engines_agree(kw, np.asarray(cfg["platform"]["position"], dtype=float), pts, c)
    assert a.valid.any() and len(pts) > 1000


# --- (vi) the energy rule and the defaults ---------------------------------------------------------------------

def test_the_energy_rule_applies_to_a_tilted_bed():
    """D-105: orders kept while an arrival is within 20 dB of the strongest, up to the cap, with its warning."""
    kw = dict(surface_z=0.0, bottom_z=100.0, bottom_coefficient=0.5)
    src, pts = np.array([0.0, 0.0, 30.0]), np.array([[3000.0, 0.0, 50.0], [1000.0, 20.0, 60.0]])
    flat = ImageSourceProvider(C, **kw)
    level = ImageSourceProvider(C, bottom_slope=[0.0, 0.0], **kw)
    tilted = ImageSourceProvider(C, bottom_slope=[0.0, 0.004], **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for p in (flat, level, tilted):
            p.eigenrays(src, pts)
    assert flat.order_used == level.order_used == tilted.order_used == 7          # below the cap of 8: no warning
    capped = ImageSourceProvider(C, bottom_slope=[0.0, 0.004], max_order_cap=2, **kw)
    with pytest.warns(RuntimeWarning, match="cut at the cap of 2"):
        capped.eigenrays(src, pts)
    assert capped.order_used == 2 and capped.truncation["left_out_db"] > -20.0


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_the_engines_agree_on_pairing_paths_with_masked_rays(engine):
    """The construction's rays on each engine (to PARITY since sonar-k9v7; to the bit before, when both engines ran
    the NumPy construction), and the pairing of them, invalid ones included, Lambert normals and all."""
    walls = [{"point": [30.0, 0.0, 0.0], "normal": [-1.0, -0.2, 0.0], "coefficient": 0.7, "bounds": {"z": [0.0, 15.0]}}]
    kw = dict(surface_z=0.0, bottom_z=18.0, bottom=SAND, max_order=1, bottom_slope=[0.0, 0.05], walls=walls)
    p_cpp, p_np = ImageSourceProvider(C, engine="cpp", **kw), ImageSourceProvider(C, engine="numpy", **kw)
    rng = np.random.default_rng(4)
    pts = np.column_stack([rng.uniform(5, 40, 60), rng.uniform(-10, 10, 60), rng.uniform(1, 17, 60)])
    tx, rx = np.array([0.0, 0.0, 4.0]), np.array([0.0, 0.1, 4.0])
    a, b = assert_engines_agree(kw, tx, pts)
    assert not a.valid.all() and a.valid.any()
    amps = rng.standard_normal(60) + 1j * rng.standard_normal(60)
    normals = rng.standard_normal((60, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    lam = rng.uniform(size=60) > 0.5
    q1, p1, a1, *_ = two_way_paths(p_cpp, tx, rx, pts, amps, lambert=lam, normals=normals, engine=engine)
    q2, p2, a2, *_ = two_way_paths(p_np, tx, rx, pts, amps, lambert=lam, normals=normals, engine="numpy")
    np.testing.assert_allclose(q1, q2, rtol=1e-12)
    np.testing.assert_allclose(p1, p2, rtol=1e-12)
    np.testing.assert_allclose(a1, a2, rtol=1e-10)


def test_the_new_keys_are_opt_in_and_checked():
    plain = ImageSourceProvider(C, surface_z=0.0, bottom_z=60.0, bottom=SAND, max_order=1,
                                walls=[{"axis": 1, "position": 30.0, "coefficient": 0.5}])
    assert not plain.general and plain._planes is None and plain.wall_order is None
    with pytest.raises(ValueError, match=r"walls\[0\]\.positon: unknown key; did you mean 'position'\?"):
        ImageSourceProvider(C, max_order=1, walls=[{"axis": 1, "positon": 30.0, "coefficient": 0.5}])
    with pytest.raises(ValueError, match=r"walls\[0\]\.normals: unknown key; did you mean 'normal'\?"):
        ImageSourceProvider(C, max_order=1, walls=[{"point": [0, 0, 0], "normal": [1, 0, 0], "normals": 1, "coefficient": 0.5}])
    with pytest.raises(ValueError, match=r"walls\[0\]\.bounds\.yy: unknown key; did you mean 'y'\?"):
        ImageSourceProvider(C, max_order=1, walls=[{"point": [0, 0, 0], "normal": [1, 0, 0], "coefficient": 0.5, "bounds": {"yy": [0, 1]}}])
    with pytest.raises(ValueError, match="wall needs"):
        ImageSourceProvider(C, max_order=1, walls=[{"point": [0, 0, 0], "normal": [1, 0, 0]}])
    with pytest.raises(ValueError, match="does not say which side is water"):
        ImageSourceProvider(C, bottom_z=60.0, bottom=SAND, max_order=1, bottom_slope=[0.0, 0.1],
                            walls=[{"axis": 1, "position": 30.0, "coefficient": 0.5}])
    with pytest.raises(ValueError, match="wall_order applies"):
        ImageSourceProvider(C, max_order=1, wall_order=2, walls=[{"axis": 1, "position": 30.0, "coefficient": 0.5}])
    with pytest.raises(ValueError, match="there is no bottom"):
        ImageSourceProvider(C, max_order=1, bottom_slope=[0.0, 0.1])
    with pytest.raises(ValueError, match="a zero vector"):
        ImageSourceProvider(C, max_order=1, walls=[{"point": [0, 0, 0], "normal": [0, 0, 0], "coefficient": 0.5}])


def test_time_spreading_and_bottom_reverberation_refuse_a_tilted_bed():
    from gpusonar.sim import timespread
    from gpusonar.sim.simulator import run_scenario
    p = ImageSourceProvider(C, bottom_z=60.0, bottom=SAND, max_order=1, bottom_slope=[0.0, 0.1])
    with pytest.raises(ValueError, match="time_spread: not with"):
        timespread.wrap(p, {"surface": {"wind_speed": 5.0}}, frequency=1e5)
    cfg = {"name": "t", "sonar": "fls_120k", "fidelity": "image_source",
           "environment": {"base": "coastal_12c_sand", "water_depth_m": 60.0},
           "propagation": {"max_order": 1, "bottom_slope": [0.0, 0.05]},
           "sonar_overrides": {"receive": {"kind": "baseband", "sample_rate": 40000.0, "window": {"range": [20.0, 60.0]}}},
           "platform": {"position": [0, 0, 20.0]}, "pings": {"count": 1},
           "scene": [{"type": "points", "positions": [[40.0, 0.0, 30.0]], "target_strength_db": 0.0}],
           "reverberation": [{"mu_db": -27.0}], "receiver": {"adc": False}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="reverberation on the bottom with propagation.bottom_slope"):
            run_scenario(cfg)
