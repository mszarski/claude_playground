"""The simulator parts ported to C++ (libs/sim: paths.cpp, reverb.cpp) against the NumPy
implementations they replace.

Paths: delays, amplitudes and directions of the free-field, image-source and sediment
providers and the pairing of `two_way_paths`, to a few units in the last place.
Reverberation: both engines take their random numbers from the same NumPy generator, in
the same order, so a ping is compared sample for sample; the C++ engine works in single
precision where NumPy uses double for the first transform, which bounds the agreement.
The CUDA tests run where a device is present and compare its reverberation with the host's.
"""

import numpy as np
import pytest

from gpusonar import _gpusonar as core
from gpusonar.sim import (FreeFieldProvider, ImageSourceProvider, Scene, SedimentProvider, engine, make_pose,
                          reverb, run_scenario, simulate_ping, snell_crossing)
from gpusonar.sim.propagation import EigenrayProvider, EigenraySet, rayleigh_reflection, two_way_paths
from gpusonar.sonar import SonarModel

pytestmark = pytest.mark.skipif(not engine.HAVE_CPP, reason="extension module built without the C++ simulator parts")

C = 1500.0
# Doubles computed in the same order by both engines: a few units in the last place.
RTOL = 1e-13


def field_points(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-300, 300, n), rng.uniform(-300, 300, n), rng.uniform(1.0, 79.0, n)])


def assert_same_eigenrays(a: EigenraySet, b: EigenraySet):
    assert a.delay.shape == b.delay.shape and a.launch.shape == b.launch.shape
    np.testing.assert_allclose(a.delay, b.delay, rtol=RTOL, atol=0.0)
    # A Rayleigh coefficient near its zero (the angle of intromission of a slow bottom) is a
    # difference of nearly equal terms: its error is relative to the terms, not to itself.
    np.testing.assert_allclose(a.amplitude, b.amplitude, rtol=RTOL, atol=1e-15 * np.abs(b.amplitude).max())
    np.testing.assert_allclose(a.launch, b.launch, rtol=RTOL, atol=1e-15)
    np.testing.assert_allclose(a.arrival, b.arrival, rtol=RTOL, atol=1e-15)
    np.testing.assert_array_equal(a.n_surface, b.n_surface)
    np.testing.assert_array_equal(a.n_bottom, b.n_bottom)
    np.testing.assert_array_equal(a.valid, b.valid)
    assert a.valid.dtype == b.valid.dtype == bool
    assert a.sound_speed == b.sound_speed
    assert a.amplitude.dtype == b.amplitude.dtype == complex


# --- engine selection -----------------------------------------------------------------------

def test_engine_selection(monkeypatch):
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)
    assert engine.resolve() == "cpp"
    assert engine.resolve("numpy") == "numpy"
    monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpy")
    assert engine.resolve() == "numpy"
    assert engine.resolve("cpp") == "cpp"              # the argument wins over the environment
    with pytest.raises(ValueError):
        engine.resolve("fortran")


def test_environment_variable_selects_the_numpy_provider(monkeypatch):
    calls = []
    real = core.sim_free_field_eigenrays
    monkeypatch.setattr(core, "sim_free_field_eigenrays", lambda *a, **k: calls.append(1) or real(*a, **k))
    provider = FreeFieldProvider(C)
    provider.eigenrays(np.zeros(3), field_points(5))
    assert len(calls) == 1
    monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpy")
    provider.eigenrays(np.zeros(3), field_points(5))
    assert len(calls) == 1


# --- paths ----------------------------------------------------------------------------------

def test_free_field_paths_match():
    source = np.array([3.0, -7.0, 12.0])
    pts = field_points()
    a = FreeFieldProvider(C, engine="cpp").eigenrays(source, pts)
    b = FreeFieldProvider(C, engine="numpy").eigenrays(source, pts)
    assert_same_eigenrays(a, b)
    np.testing.assert_allclose(a.delay[:, 0], np.linalg.norm(pts - source, axis=1) / C, rtol=1e-14)
    # A single point given as [3].
    assert_same_eigenrays(FreeFieldProvider(C, engine="cpp").eigenrays(source, pts[0]),
                          FreeFieldProvider(C, engine="numpy").eigenrays(source, pts[0]))


@pytest.mark.parametrize("options", [
    dict(surface_z=0.0, bottom_z=None, max_order=1),
    dict(surface_z=0.0, bottom_z=80.0, bottom_coefficient=0.5, max_order=2),
    dict(surface_z=0.0, bottom_z=80.0, max_order=5, surface_coefficient=-0.9 + 0.1j, bottom_coefficient=0.4 - 0.3j),
    dict(surface_z=None, bottom_z=80.0, bottom_coefficient=0.5, max_order=3),
    dict(surface_z=None, bottom_z=None, max_order=3),
    dict(surface_z=0.0, bottom_z=80.0, max_order=3, bottom=dict(speed=1650.0, density=1.9, attenuation=0.8)),
    dict(surface_z=0.0, bottom_z=80.0, max_order=4, bottom=dict(speed=1450.0, density=1.4)),     # slow mud, no loss
    dict(surface_z=0.0, bottom_z=80.0, bottom_coefficient=0.5, max_order=2,
         walls=[dict(axis=0, position=400.0, coefficient=0.8), dict(axis=1, position=-350.0, coefficient=0.6 + 0.2j)]),
    dict(surface_z=0.0, bottom_z=80.0, max_order=2, bottom=dict(speed=1700.0, density=2.0, attenuation=0.5),
         walls=[dict(axis=1, position=350.0, coefficient=0.7)]),
    # an elastic bottom (rock, A8), and the same over a tilted bed (the plane construction)
    dict(surface_z=0.0, bottom_z=80.0, max_order=4, bottom=dict(speed=3820.0, density=2.5, attenuation=0.75,
                                                                shear_speed=1900.0, shear_attenuation=4.64)),
    dict(surface_z=0.0, bottom_z=80.0, max_order=3, bottom_slope=[0.05, 0.0],
         bottom=dict(speed=3820.0, density=2.5, attenuation=0.75, shear_speed=1900.0, shear_attenuation=4.64)),
])
def test_image_source_paths_match(options):
    source = np.array([0.0, 5.0, 20.0])
    pts = field_points(seed=1)
    a = ImageSourceProvider(C, engine="cpp", **options).eigenrays(source, pts)
    b = ImageSourceProvider(C, engine="numpy", **options).eigenrays(source, pts)
    assert_same_eigenrays(a, b)


AXIS_WALL_OPTIONS = dict(surface_z=0.0, bottom_z=80.0, max_order=2, bottom=dict(speed=1700.0, density=2.0, attenuation=0.5))


@pytest.mark.parametrize("axis", [0, 1, np.int64(0), np.int64(1)])
def test_each_wall_axis_runs_in_cpp_and_matches_numpy(axis, monkeypatch):
    """An {axis, position} wall of each valid axis (0: x = position, 1: y = position) runs in C++ when C++ is
    selected, never in NumPy (the NumPy series is made to raise), and agrees with NumPy to the tolerance of the
    neighbouring tests. NumPy integers are axes too."""
    source = np.array([0.0, 5.0, 20.0])
    pts = field_points(seed=2)
    walls = [dict(axis=axis, position=-120.0, coefficient=0.7 - 0.1j)]
    b = ImageSourceProvider(C, engine="numpy", walls=walls, **AXIS_WALL_OPTIONS).eigenrays(source, pts)
    calls = []
    real = core.sim_image_source_eigenrays
    monkeypatch.setattr(core, "sim_image_source_eigenrays", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(ImageSourceProvider, "_eigenrays_numpy", lambda *a, **k: pytest.fail("the NumPy series ran"))
    a = ImageSourceProvider(C, engine="cpp", walls=walls, **AXIS_WALL_OPTIONS).eigenrays(source, pts)
    assert calls == [1]
    assert_same_eigenrays(a, b)
    assert np.any(a.arrival[..., int(axis)] != a.launch[..., int(axis)])      # the wall rays are there


# Not a wall axis. 2 would be a horizontal plane laid once over every image of the surface and bottom series:
# it duplicates the surface or bottom image where it lies on one, and is no valid path elsewhere (before a-wallaxis
# NumPy mirrored in z and the C++ engine silently ran NumPy for it). Negative values were NumPy indices (-1 = z,
# -2 = y, -3 = x), 3 and -4 an IndexError, a float was cut to an integer (0.7 -> 0), a string was read ("1").
@pytest.mark.parametrize("axis", [2, -1, -2, -3, 3, 0.7, 1.5, 1.0, True, "1", None])
@pytest.mark.parametrize("eng", ["cpp", "numpy"])
def test_a_wall_axis_that_is_not_0_or_1_is_refused_in_both_engines(axis, eng):
    with pytest.raises(ValueError, match=r"walls\[0\]\.axis: .* give 0 \(a wall x = position\) or 1 \(a wall y = position\)"):
        ImageSourceProvider(C, engine=eng, walls=[dict(axis=axis, position=5.0, coefficient=0.8)], **AXIS_WALL_OPTIONS)


@pytest.mark.parametrize("wall, missing", [(dict(position=5.0, coefficient=0.8), "axis"),
                                           (dict(axis=0, coefficient=0.8), "position"),
                                           (dict(axis=0, position=float("nan"), coefficient=0.8), "position")])
@pytest.mark.parametrize("eng", ["cpp", "numpy"])
def test_an_axis_wall_needs_its_axis_and_a_finite_position(wall, missing, eng):
    with pytest.raises(ValueError, match=rf"walls\[0\]\.{missing}: "):
        ImageSourceProvider(C, engine=eng, walls=[wall], **AXIS_WALL_OPTIONS)


@pytest.mark.parametrize("shape", [(4, 2), (2, 2, 3)])
@pytest.mark.parametrize("eng", ["cpp", "numpy"])
@pytest.mark.parametrize("walls", [[], [dict(point=[0.0, 50.0, 0.0], normal=[0.0, -1.0, 0.0], coefficient=0.8)]])
def test_image_source_field_points_must_be_s_by_3(shape, eng, walls):
    """Field points other than [S, 3] (or one point [3]) are refused in both engines: before a-wallaxis the C++
    engine silently ran NumPy for them, which then failed with an IndexError."""
    provider = ImageSourceProvider(C, engine=eng, walls=walls, **AXIS_WALL_OPTIONS)
    with pytest.raises(ValueError, match=r"field points \[S, 3\]"):
        provider.eigenrays(np.zeros(3), np.ones(shape))


def test_image_path_lengths_are_those_of_the_images():
    source = np.array([0.0, 0.0, 20.0])
    e = ImageSourceProvider(C, surface_z=0.0, bottom_z=100.0, bottom_coefficient=0.5, max_order=2, engine="cpp").eigenrays(
        source, [[300.0, 0.0, 60.0]])
    image_z = np.array([60.0, -60.0, 260.0, 140.0, -140.0])      # direct, s, sb, b, bs
    np.testing.assert_allclose(e.delay[0], np.hypot(300.0, image_z - 20.0) / C, rtol=1e-15)
    np.testing.assert_allclose(e.amplitude[0], np.array([1, -1, -0.5, 0.5, -0.5]) / np.hypot(300.0, image_z - 20.0),
                               rtol=1e-15)


def test_rayleigh_coefficient_matches():
    g = np.linspace(0.0, np.pi / 2, 181)
    for kw in (dict(speed=1650.0, density=1.9, attenuation=0.8), dict(speed=1450.0, density=1.4),
               dict(speed=1700.0, density=2.0, attenuation=0.0, water_density=1.03),
               # elastic (A8): rock with shear, lossless rock, and a slow shear (sand of J&R Fig. 9.3)
               dict(speed=3820.0, density=2.5, attenuation=0.75, shear_speed=1900.0, shear_attenuation=4.64),
               dict(speed=3750.0, density=2.5, shear_speed=1950.0),
               dict(speed=1650.0, density=2.0, attenuation=0.55, water_density=1.0, shear_speed=250.0, shear_attenuation=0.55)):
        np.testing.assert_allclose(core.sim_rayleigh_reflection(g, C, **kw), rayleigh_reflection(g, C, **kw),
                                   rtol=1e-12, atol=1e-15)


def test_sediment_refraction_matches():
    source = np.array([2.0, -1.0, -12.0])
    rng = np.random.default_rng(2)
    # Points on both sides of the interface, one on it and one straight below the source.
    pts = np.column_stack([rng.uniform(-40, 40, 300), rng.uniform(-40, 40, 300), rng.uniform(-8.0, 6.0, 300)])
    pts[0] = [10.0, 3.0, 0.0]
    pts[1] = [2.0, -1.0, 3.0]
    for c_sed in (1700.0, 1450.0):
        t_a, x_a = snell_crossing(source, pts, C, c_sed, 0.0, engine="cpp")
        t_b, x_b = snell_crossing(source, pts, C, c_sed, 0.0, engine="numpy")
        np.testing.assert_allclose(t_a, t_b, rtol=RTOL)
        np.testing.assert_allclose(x_a, x_b, rtol=1e-12, atol=1e-12)
        a = SedimentProvider(C, c_sed, interface_z=0.0, transmission=0.8 - 0.1j, engine="cpp").eigenrays(source, pts)
        b = SedimentProvider(C, c_sed, interface_z=0.0, transmission=0.8 - 0.1j, engine="numpy").eigenrays(source, pts)
        np.testing.assert_allclose(a.delay, b.delay, rtol=RTOL)
        np.testing.assert_allclose(a.amplitude, b.amplitude, rtol=RTOL)
        # Directions follow the crossing point, which Newton's method leaves with a few
        # more units of rounding than the travel time (the time is stationary in it).
        np.testing.assert_allclose(a.launch, b.launch, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(a.arrival, b.arrival, rtol=0.0, atol=1e-10)
    # A source below the interface: straight paths on both engines.
    below = np.array([0.0, 0.0, 4.0])
    for eng in ("cpp", "numpy"):
        t, x = snell_crossing(below, pts, C, 1700.0, 0.0, engine=eng)
        np.testing.assert_allclose(t, np.linalg.norm(pts - below, axis=1) / C, rtol=1e-15)
        np.testing.assert_array_equal(x, pts)


def test_sediment_fluid_penetration_matches():
    """The opt-in `penetration: fluid` (G-3): transmission coefficient, reciprocity factor and attenuation (and
    the evanescent field of the sub-critical points, most of these 300), on both engines, to the tolerance of the
    default tier above; and every point above the interface as before."""
    source = np.array([2.0, -1.0, -12.0])
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(-40, 40, 300), rng.uniform(-40, 40, 300), rng.uniform(-8.0, 6.0, 300)])
    pts[0] = [10.0, 3.0, 0.0]
    pts[1] = [2.0, -1.0, 3.0]
    for bottom in ({"speed": 1700.0, "density": 1.845, "attenuation": 0.886}, {"speed": 1450.0, "density": 1.4,
                                                                                "attenuation": 0.2, "water_density": 1.03}):
        for f, policy in ((12e3, "evanescent"), (30e3, "evanescent"), (30e3, "ray")):
            kw = dict(penetration="fluid", bottom=bottom, frequency=f, subcritical=policy)
            with pytest.warns(RuntimeWarning, match="below the critical angle") if bottom["speed"] > C else _nothing():
                a = SedimentProvider(C, bottom["speed"], 0.0, engine="cpp", **kw).eigenrays(source, pts)
                b = SedimentProvider(C, bottom["speed"], 0.0, engine="numpy", **kw).eigenrays(source, pts)
            np.testing.assert_allclose(a.delay, b.delay, rtol=RTOL)
            # The evanescent field is exp(-x) with x up to about 60 here: its relative rounding is x times that of
            # x (measured 1.6e-13 at 2e-24 of the 1/(c t) level), so these points get 1e-12.
            np.testing.assert_allclose(a.amplitude, b.amplitude, rtol=RTOL if policy == "ray" else 1e-12)
            np.testing.assert_allclose(a.launch, b.launch, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(a.arrival, b.arrival, rtol=0.0, atol=1e-10)
            above = pts[:, 2] <= 0.0
            plain = SedimentProvider(C, bottom["speed"], 0.0, engine="cpp").eigenrays(source, pts)
            np.testing.assert_array_equal(a.amplitude[above], plain.amplitude[above])


def _nothing():
    import contextlib
    return contextlib.nullcontext()


# --- penetration: rough (G-13) --------------------------------------------------------------------------------
# NumPy's exp, log, pow, hypot, |z| and complex product are vectorised (its own polynomials, fused multiply-adds)
# and differ from libm and plain products by about an ulp, so the two engines do not agree bit for bit here. The
# integers (cells, hash, delay bins) agree exactly, so the shapes and validity masks are equal.

ROUGH_LN10 = np.log(10.0)
SAX99_BOTTOM = {"speed": 1.16 * 1530.0, "density": 2.0, "attenuation": 0.010 * 40 * np.pi / ROUGH_LN10}
MEDIUM_SAND = {"speed": 1767.0, "density": 1.845, "attenuation": 0.886}


def _rough_pair(c, bottom, f, roughness, source, pts):
    import warnings
    kw = dict(penetration="rough", bottom=bottom, frequency=f, roughness=dict(roughness))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = SedimentProvider(c, bottom["speed"], 0.0, engine="cpp", **kw).eigenrays(source, pts)
        b = SedimentProvider(c, bottom["speed"], 0.0, engine="numpy", **kw).eigenrays(source, pts)
    return a, b


def _rough_points(source, rng, n_sub=6):
    """Points below the critical angle on the straight line from `source` (10-20 deg, 0.05-0.4 m deep, any azimuth),
    two steep ones above it and one in the water."""
    deg = rng.uniform(10.0, 20.0, n_sub)
    depth = rng.uniform(0.05, 0.4, n_sub)
    az = rng.uniform(0.0, 2 * np.pi, n_sub)
    r = (depth - source[2]) / np.tan(np.radians(deg))
    sub = np.column_stack([source[0] + r * np.cos(az), source[1] + r * np.sin(az), depth])
    return np.vstack([sub, [source[0] + 0.5, source[1] - 0.3, 0.3], [source[0], source[1], 0.1],
                      [source[0] + 3.0, source[1], -2.0]])


# Measured over the 20 cases of test_rough_penetration_matches (2026-11-26; `rough_difference`): delays 6.4e-16
# relative at worst (a bin's delay is the intensity-weighted mean of its cells', and the weights differ by an ulp);
# amplitudes 2.6e-12 of the point's largest scattered ray, nearly all of it the phase w0 (t - t_b) of the merge
# taking that delay's rounding (w0 x 1e-17 s = 1.5e-12 rad at 20 kHz); directions 7.2e-15. Column 0 (`fluid`'s
# ray) 2.5e-16. The tolerances are two to three times the worst measured.
ROUGH_DELAY_RTOL = 2e-15
ROUGH_AMP_ATOL = 6e-12      # of the largest scattered amplitude of the same point
ROUGH_DIR_ATOL = 2e-14


def _assert_rough_equal(a, b):
    assert a.delay.shape == b.delay.shape, (a.delay.shape, b.delay.shape)
    np.testing.assert_array_equal(a.valid, b.valid)
    np.testing.assert_array_equal(a.n_surface, b.n_surface)
    np.testing.assert_array_equal(a.n_bottom, b.n_bottom)
    # column 0 is `fluid`'s ray: the tolerance of test_sediment_fluid_penetration_matches
    np.testing.assert_allclose(a.delay[:, 0], b.delay[:, 0], rtol=RTOL)
    np.testing.assert_allclose(a.amplitude[:, 0], b.amplitude[:, 0], rtol=1e-12)
    np.testing.assert_allclose(a.launch[:, 0], b.launch[:, 0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(a.arrival[:, 0], b.arrival[:, 0], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(a.delay[:, 1:], b.delay[:, 1:], rtol=ROUGH_DELAY_RTOL, atol=0.0)
    scale = np.abs(b.amplitude[:, 1:]).max(axis=1, keepdims=True) if b.delay.shape[1] > 1 else 1.0
    assert np.all(np.abs(a.amplitude[:, 1:] - b.amplitude[:, 1:]) <= ROUGH_AMP_ATOL * scale)
    np.testing.assert_allclose(a.launch[:, 1:], b.launch[:, 1:], rtol=0.0, atol=ROUGH_DIR_ATOL)
    np.testing.assert_allclose(a.arrival[:, 1:], b.arrival[:, 1:], rtol=0.0, atol=ROUGH_DIR_ATOL)


def rough_difference(a, b):
    """(largest relative delay difference, largest amplitude difference re the point's largest ray, largest direction
    difference) over the scattered columns: what the tolerances above are set from."""
    if a.delay.shape[1] == 1:
        return 0.0, 0.0, 0.0
    v = b.valid[:, 1:]
    dd = np.abs(a.delay[:, 1:] - b.delay[:, 1:])[v] / b.delay[:, 1:][v]
    scale = np.abs(b.amplitude[:, 1:]).max(axis=1, keepdims=True)
    da = (np.abs(a.amplitude[:, 1:] - b.amplitude[:, 1:]) / np.where(scale > 0, scale, 1.0))[v]
    dl = max(np.abs(a.launch[:, 1:] - b.launch[:, 1:]).max(), np.abs(a.arrival[:, 1:] - b.arrival[:, 1:]).max())
    return float(dd.max()), float(da.max()), float(dl)


@pytest.mark.parametrize("seed", [0, 1, 7, 2**63 + 5, -3])
@pytest.mark.parametrize("spectrum", [dict(w2=0.0000433, gamma2=3.04, K0=10.0),     # SAX99, von Karman (D.7)
                                      dict(w2=0.00002, gamma2=3.25, K0=0.0)])        # a pure power law
@pytest.mark.parametrize("site", ["sax99", "medium_sand"])
def test_rough_penetration_matches(seed, spectrum, site):
    """`penetration: rough` on both engines: the same cells, deviates and delay bins (shapes, validity equal), the
    `fluid` ray in column 0, the scattered rays to the measured rounding above; points below and above the critical
    angle and in the water, a source off the origin, several seeds (one beyond 2^63 and one negative: the hash takes
    the 64-bit pattern), two spectra and two sediments."""
    c, bottom, f = (1530.0, SAX99_BOTTOM, 20e3) if site == "sax99" else (C, MEDIUM_SAND, 30e3)
    source = np.array([1.5, -2.0, -10.0])
    pts = _rough_points(source, np.random.default_rng(abs(seed) % 1000 + (0 if site == "sax99" else 50)))
    a, b = _rough_pair(c, bottom, f, dict(spectrum, seed=seed), source, pts)
    assert b.valid[:6, 1:].sum(axis=1).min() > 5 and not b.valid[6:, 1:].any()
    _assert_rough_equal(a, b)


def test_rough_numerical_knobs_and_refinement_match():
    """Coarser and finer cells, other bins and reach, on both engines: the knobs reach the C++ engine."""
    source = np.array([0.0, 0.0, -6.0])
    pts = _rough_points(source, np.random.default_rng(11), n_sub=3)
    for knobs in (dict(patch=0.02, refine=2.0, floor_db=20.0, bin=2e-5), dict(patch=0.1, refine=5.0, floor_db=40.0),
                  dict(bin=1e-6)):
        a, b = _rough_pair(1530.0, SAX99_BOTTOM, 20e3, dict(w2=0.0000433, gamma2=3.04, K0=10.0, seed=4, **knobs),
                           source, pts)
        _assert_rough_equal(a, b)


def test_rough_parts_match():
    """The parts one by one: the quadtree's cells exactly (integers and centres), the radius exactly, the deviates and
    the cross section to their rounding, so that a disagreement above can be traced."""
    from gpusonar.sim import penetration as pen
    for args in ((10.0, 0.0, 1.5, 0.04, 3.0), (-3.3, 7.1, 6.0, 0.0375, 3.0), (0.01, -0.02, 0.3, 0.05, 2.0)):
        got, want = core.sim_rough_quadtree_cells(*args), pen.quadtree_cells(*args)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)
    for depth, k_loss, floor in ((0.3, 0.8, 30.0), (0.05, 3.0, 20.0), (0.4, 0.0, 40.0)):
        assert core.sim_rough_sediment_radius(depth, k_loss, floor) == pen._sediment_radius(np.float64(depth), k_loss, floor)
    rng = np.random.default_rng(5)
    lv, i, j = rng.integers(0, 8, 5000), rng.integers(-10**6, 10**6, 5000), rng.integers(-10**6, 10**6, 5000)
    for seed in (0, 7, 2**64 - 1):
        np.testing.assert_allclose(core.sim_rough_patch_coefficients(seed, lv, i, j), pen.patch_coefficients(seed, lv, i, j),
                                   rtol=5e-16, atol=0.0)          # measured 2.3e-16 (NumPy's log, 14 of 5000)
    k = 2 * np.pi * 20e3 / 1530.0
    K_i = np.column_stack([k * np.cos(rng.uniform(0.1, 0.5, 2000)), np.zeros(2000)])
    phi, ks = rng.uniform(0, 2 * np.pi, 2000), rng.uniform(0.0, 0.999, 2000) * k / 1.16
    K_s = np.column_stack([ks * np.cos(phi), ks * np.sin(phi)])
    for K0 in (10.0, 0.0):
        args = (k, 2.0, 1.16, 0.010, 0.0000433, 3.04, K0)
        np.testing.assert_allclose(core.sim_rough_sigma_wp(K_s, K_i, *args), pen.sigma_wp(K_s, K_i, *args), rtol=1e-14)  # measured 3.1e-15


def test_rough_zero_relief_is_fluid_on_the_cpp_engine():
    """w2 = 0 on the C++ engine: every scattered amplitude exactly 0, and column 0 `fluid`'s ray bit for bit."""
    import warnings
    source = np.array([0.0, 0.0, -10.0])
    pts = _rough_points(source, np.random.default_rng(2), n_sub=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = SedimentProvider(1530.0, SAX99_BOTTOM["speed"], 0.0, engine="cpp", penetration="rough", bottom=SAX99_BOTTOM,
                             frequency=20e3, roughness=dict(w2=0.0, gamma2=3.04, K0=10.0)).eigenrays(source, pts)
        b = SedimentProvider(1530.0, SAX99_BOTTOM["speed"], 0.0, engine="cpp", penetration="fluid", bottom=SAX99_BOTTOM,
                             frequency=20e3).eigenrays(source, pts)
    assert np.all(a.amplitude[:, 1:] == 0) and a.valid[:3, 1:].any()
    for name in ("delay", "amplitude", "launch", "arrival", "valid"):
        np.testing.assert_array_equal(getattr(a, name)[:, :1], getattr(b, name))


class MaskedProvider(EigenrayProvider):
    """Image-source eigenrays with some rays marked invalid, as a ray tracer returns them."""

    def __init__(self):
        self.inner = ImageSourceProvider(C, surface_z=0.0, bottom_z=80.0, bottom_coefficient=0.5, max_order=2, engine="numpy")
        self.sound_speed = C

    def eigenrays(self, source, points):
        e = self.inner.eigenrays(source, points)
        e.valid[::3, 1] = False
        e.valid[5] = False
        return e


@pytest.mark.parametrize("provider", [
    FreeFieldProvider(C), ImageSourceProvider(C, surface_z=0.0, bottom_z=80.0, bottom_coefficient=0.5, max_order=2),
    SedimentProvider(C, 1700.0, interface_z=60.0), MaskedProvider()], ids=lambda p: type(p).__name__)
@pytest.mark.parametrize("case", ["plain", "lambert", "normals", "moving", "all"])
def test_two_way_paths_match(provider, case):
    rng = np.random.default_rng(3)
    n = 200
    pts = field_points(n, seed=4)
    tx, rx = np.array([0.0, 0.0, 20.0]), np.array([0.3, -0.2, 20.1])
    amp = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    amp[::7] = 0.0                                             # scatterers that return nothing
    kw = {}
    if case in ("lambert", "normals", "all"):
        kw["lambert"] = rng.random(n) < 0.6
    if case in ("normals", "all"):
        normals = rng.standard_normal((n, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        normals[::4] = np.nan                                  # patches without a normal of their own
        kw["normals"] = normals
    if case in ("moving", "all"):
        kw["velocities"] = rng.standard_normal((n, 3))
    a = two_way_paths(provider, tx, rx, pts, amp, engine="cpp", **kw)
    b = two_way_paths(provider, tx, rx, pts, amp, engine="numpy", **kw)
    assert len(a[2]) == len(b[2]) > 0
    np.testing.assert_allclose(a[0], b[0], rtol=RTOL, atol=1e-11)        # q, metres
    np.testing.assert_allclose(a[1], b[1], rtol=RTOL, atol=1e-11)        # p
    np.testing.assert_allclose(a[2], b[2], rtol=1e-12, atol=0.0)         # amplitude
    assert a[2].dtype == b[2].dtype
    for va, vb in zip(a[3:], b[3:]):
        assert (va is None) == (vb is None)
        if va is not None:
            np.testing.assert_array_equal(va, vb)
    assert (a[3] is None) == (case not in ("moving", "all"))


def test_two_way_paths_without_motion_or_lambert_flags():
    """All-zero velocities and all-false flags are the same as none, on both engines."""
    pts = field_points(20, seed=5)
    for eng in ("cpp", "numpy"):
        q, p, amp, vq, vp = two_way_paths(FreeFieldProvider(C), np.zeros(3), np.zeros(3), pts, np.ones(20),
                                          velocities=np.zeros((20, 3)), lambert=np.zeros(20, dtype=bool), engine=eng)
        assert vq is None and vp is None
        np.testing.assert_allclose(q, pts, rtol=1e-14)
        np.testing.assert_allclose(amp, 1.0, rtol=1e-14)


# --- reverberation --------------------------------------------------------------------------

MU_DB = -27.0


def empty_ping(name="popoto_8", kind="baseband", fs=20e3, ranges=(25.0, 230.0), duration=2e-3, waveform=None,
               velocity=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), depth=-20.0, position_jitter=0.0):
    wf = {"type": "cw", "f_start": 27.5e3, "f_stop": 27.5e3, "duration": duration, "window_alpha": 0.0,
          "source_level_db_rms": None, "amplitude": 1.0}
    wf.update(waveform or {})
    cfg = {"receive": {"kind": kind, "sample_rate": fs, "window": {"range": list(ranges)}, "sensitivity_db": None},
           "transmit": {"lever_arm": [0, 0, 0], "waveform": wf}}
    sonar = SonarModel.load(name, cfg)
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, depth), velocity=velocity, rpy=rpy), Scene(),
                        environment={"sound_speed": C}, adc=False)
    if position_jitter > 0.0:
        # Elements off their nominal places: no longer a uniform line.
        cal = rec.calibration
        offsets = np.random.default_rng(99).normal(0.0, position_jitter, (rec.samples.n_channels, 3))
        cal.position_offset = [tuple(p) for p in offsets]
        rec.calibration = cal
    return rec


def relative_difference(a, b):
    """Largest sample difference over the largest sample, and RMS difference over RMS."""
    d = np.abs(a.astype(np.complex128) - b.astype(np.complex128))
    return d.max() / np.abs(b).max(), np.sqrt(np.mean(d**2) / np.mean(np.abs(b) ** 2))


# Single precision against NumPy's double first transform. Measured on the cases below:
# largest difference 1.6e-7 to 3.0e-7 of the largest sample, RMS difference 3e-7 to 7e-7 of
# the RMS.
REVERB_TOL = 5e-6

REVERB_CASES = {
    "popoto_8_baseband": (dict(), dict()),
    # elements off a uniform line: one complex exponential per channel and bin
    "popoto_8_displaced_elements": (dict(position_jitter=0.01), dict(n_azimuth=60)),
    "planar_4x7": (dict(name="popoto_4x7"), dict(n_azimuth=60)),
    # uniform line of 32: the per-sector stride path; long pulse, decimated generation
    "flank_32_uniform_decimated": (dict(name="flank_32", ranges=(60.0, 900.0), fs=16e3, duration=0.02),
                                   dict(n_azimuth=180)),
    "real_passband": (dict(kind="real", fs=100e3, ranges=(60.0, 120.0)), dict()),
    "lfm_tapered": (dict(waveform={"type": "lfm", "f_start": 25e3, "f_stop": 30e3, "window_alpha": 0.2}), dict()),
    # own speed: each sector's sequence and pulse are Doppler shifted
    "moving_uniform": (dict(name="flank_32", ranges=(60.0, 500.0), fs=16e3, duration=0.02,
                            velocity=(3.0, 1.0, 0.0)), dict(n_azimuth=90)),
    "moving_displaced_elements": (dict(velocity=(-2.0, 4.0, 0.5), position_jitter=0.01), dict(n_azimuth=40)),
    # beam patterns: power differs from sector to sector, and sectors behind the array have none
    "fls_patterns": (dict(name="fls_120k", fs=40e3, ranges=(20.0, 60.0), depth=20.0,
                          waveform={"f_start": 120e3, "f_stop": 120e3, "duration": 1e-3}),
                     dict(n_azimuth=48, boundary_z=60.0)),
    "absorption_and_range_limit": (dict(), dict(absorption_db_per_m=0.01, max_range=150.0, n_azimuth=30)),
}


def run_case(name, **cpp_options):
    """One case on the two engines from the same seed: (samples, returned power, generator state) of each."""
    ping_kw, reverb_kw = REVERB_CASES[name]
    reverb_kw = dict(reverb_kw)
    boundary = reverb_kw.pop("boundary_z", 0.0)
    out = {}
    for eng in ("cpp", "numpy"):
        rec = empty_ping(**ping_kw)
        rng = np.random.default_rng(3)
        power = reverb.add_reverberation(rec, FreeFieldProvider(C), boundary, MU_DB, rng, engine=eng, **reverb_kw,
                                         **(cpp_options if eng == "cpp" else {}))
        out[eng] = (rec.samples.array, power, rng.bit_generator.state)
    return out["cpp"], out["numpy"]


@pytest.mark.parametrize("name", list(REVERB_CASES))
def test_reverberation_matches_sample_for_sample(name):
    (x_cpp, p_cpp, state_cpp), (x_np, p_np, state_np) = run_case(name)
    assert x_cpp.shape == x_np.shape and x_cpp.dtype == x_np.dtype
    assert np.abs(x_np).max() > 0.0
    worst, rms = relative_difference(x_cpp, x_np)
    assert worst < REVERB_TOL, (worst, rms)
    assert rms < REVERB_TOL, (worst, rms)
    np.testing.assert_array_equal(p_cpp, p_np)              # the returned power is computed before the engines part
    # Both engines consumed the generator identically: what is drawn next is the same.
    assert state_cpp == state_np


def test_reverberation_cases_cover_both_steering_paths():
    """The cases above reach the uniform-line recurrence and the per-channel exponentials."""
    def uniform(rec):
        offsets = rec.rx_positions() - rec.rx_positions().mean(axis=0)
        along = offsets - offsets[0]
        return len(offsets) > 2 and np.allclose(along[1:], np.arange(1, len(offsets))[:, None] * along[1], atol=1e-9)
    kinds = {name: bool(uniform(empty_ping(**REVERB_CASES[name][0]))) for name in REVERB_CASES}
    assert kinds["popoto_8_baseband"] and kinds["flank_32_uniform_decimated"] and kinds["moving_uniform"]
    assert not kinds["popoto_8_displaced_elements"] and not kinds["planar_4x7"]
    assert not kinds["moving_displaced_elements"]


def test_reverberation_in_runs_of_sectors_is_the_same(monkeypatch):
    """Random numbers are handed over a run of sectors at a time; the run length changes nothing."""
    rec = empty_ping()
    reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(8), n_azimuth=20,
                             engine="cpp")
    whole = rec.samples.array
    calls = []
    real = core.sim_reverb_accumulate
    monkeypatch.setattr(core, "sim_reverb_accumulate", lambda *a, **k: calls.append(a[4].shape[0]) or real(*a, **k))
    monkeypatch.setattr(reverb, "SECTOR_RUN_BYTES", 1)        # one sector per call
    rec = empty_ping()
    rng = np.random.default_rng(8)
    reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, rng, n_azimuth=20, engine="cpp")
    assert calls == [1] * 20
    # Sectors are added in the same order, so the sums are the same to the last bit.
    np.testing.assert_array_equal(rec.samples.array, whole)


def test_cpp_reverberation_level_follows_the_sonar_equation():
    """The analytic check of test_reverberation.py, on the C++ engine explicitly."""
    height, pulse = 20.0, 2e-3
    intensity = 0.0
    for seed in range(6):
        rec = empty_ping()
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(seed), engine="cpp")
        intensity = intensity + (np.abs(rec.samples.array) ** 2).mean(axis=0) / 6
    s = rec.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    slant = 0.5 * C * (t - 0.5 * pulse)
    ok = (slant > 1.5 * height) & (slant < 200.0)
    predicted = 10 ** (MU_DB / 10) * (height / slant) ** 2 * np.pi * C * pulse / slant**3
    edges = np.linspace(np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1], 13).astype(int)
    diffs = np.array([10 * np.log10(intensity[a:b].mean() / predicted[a:b].mean())
                      for a, b in zip(edges[:-1], edges[1:])])
    assert np.abs(diffs).max() < 2.0, diffs
    assert abs(diffs.mean()) < 1.0


# --- the simulator with either engine --------------------------------------------------------

SCENARIO = {
    "name": "engine_check", "sonar": "popoto_8", "fidelity": "image_source",
    "sonar_overrides": {"receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": [30.0, 150.0]}}},
    "environment": {"sound_speed": C},
    "propagation": {"surface_z": 0.0, "bottom_z": 60.0, "max_order": 2,
                    "bottom": {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}},
    "platform": {"position": [0.0, 0.0, 20.0], "velocity": [1.5, 0.0, 0.0]},
    "pings": {"count": 2, "interval": 0.5},
    "scene": [{"type": "points", "positions": [[60.0, 10.0, 30.0], [90.0, -20.0, 45.0]], "target_strength_db": 0.0,
               "velocity": [0.0, 2.0, 0.0]},
              {"type": "points", "positions": [[110.0, 5.0, 60.0]], "target_strength_db": 5.0}],
    "reverberation": [{"boundary_z": 60.0, "mu_db": -27.0, "n_azimuth": 36},
                      {"boundary_z": 0.0, "model": "chapman_harris", "wind_speed": 5.0, "n_azimuth": 36}],
    "receiver": {"adc": False, "noise_rms": 1e-7},
    "seed": 5,
}


def test_scenario_is_the_same_on_both_engines(monkeypatch):
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)
    a = run_scenario(SCENARIO)                                  # the default: C++
    b = run_scenario(SCENARIO, engine="numpy")
    monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpy")
    c = run_scenario(SCENARIO)                                  # NumPy through the environment
    assert len(a.records) == len(b.records) == 2
    for ra, rb, rc in zip(a.records, b.records, c.records):
        xa, xb, xc = ra.samples.array, rb.samples.array, rc.samples.array
        np.testing.assert_array_equal(xb, xc)
        worst, rms = relative_difference(xa, xb)
        # Echoes, reverberation and receiver noise together. The noise is drawn after the
        # reverberation from the same generator, so it is identical only because both
        # engines leave the generator in the same state.
        assert worst < REVERB_TOL and rms < REVERB_TOL, (worst, rms)


def test_scenario_calls_the_cpp_engine_by_default(monkeypatch):
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)
    used = set()
    for name in ("sim_image_source_eigenrays", "sim_two_way_paths", "sim_reverb_accumulate", "sim_reverb_finish"):
        real = getattr(core, name)
        monkeypatch.setattr(core, name, lambda *a, _n=name, _f=real, **k: used.add(_n) or _f(*a, **k))
    run_scenario(SCENARIO, {"pings": {"count": 1}})
    assert used == {"sim_image_source_eigenrays", "sim_two_way_paths", "sim_reverb_accumulate", "sim_reverb_finish"}
    used.clear()
    run_scenario(SCENARIO, {"pings": {"count": 1}}, engine="numpy")
    assert used == set()


# --- CUDA -----------------------------------------------------------------------------------

def test_cuda_backend_is_refused_when_not_built():
    if core.cuda_compiled():
        pytest.skip("built with the CUDA backend")
    rec = empty_ping()
    with pytest.raises(RuntimeError, match="CUDA"):
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(0), n_azimuth=8,
                                 engine="cpp", backend=core.Backend.Cuda)


needs_cuda = pytest.mark.skipif(not core.cuda_available(), reason="no CUDA device")


@needs_cuda
@pytest.mark.parametrize("name", list(REVERB_CASES))
def test_cuda_reverberation_matches_the_host(name):
    """The device runs the same steering kernel as the host on the same sector spectra; the
    transforms run on the host for both. Differences come from the device's single-precision
    sine and cosine and from fused multiply-adds."""
    ping_kw, reverb_kw = REVERB_CASES[name]
    reverb_kw = dict(reverb_kw)
    boundary = reverb_kw.pop("boundary_z", 0.0)
    out = {}
    for backend in (core.Backend.Host, core.Backend.Cuda):
        rec = empty_ping(**ping_kw)
        rng = np.random.default_rng(3)
        reverb.add_reverberation(rec, FreeFieldProvider(C), boundary, MU_DB, rng, engine="cpp", backend=backend,
                                 **reverb_kw)
        out[backend] = (rec.samples.array, rng.bit_generator.state)
    (host, state_host), (cuda, state_cuda) = out[core.Backend.Host], out[core.Backend.Cuda]
    assert state_host == state_cuda
    assert np.abs(host).max() > 0.0
    worst, rms = relative_difference(cuda, host)
    assert worst < 1e-4 and rms < 1e-4, (worst, rms)


@needs_cuda
def test_cuda_reverberation_matches_numpy():
    (x_cpp, _, state_cpp), (x_np, _, state_np) = run_case("flank_32_uniform_decimated", backend=core.Backend.Cuda)
    assert state_cpp == state_np
    worst, rms = relative_difference(x_cpp, x_np)
    assert worst < 1e-4 and rms < 1e-4, (worst, rms)


def _rough_ping(backend, penetration="rough"):
    """One subbottom ping at 1.5 m over medium sand with five buried points 10-20 deg grazing from the sonar (below
    the 30.9 deg critical angle): `rough` gives each 1 + about 100 eigenrays a leg, so about 10^4 pairs a point."""
    import warnings
    sonar = SonarModel.load("subbottom", {"receive": {"window": {"range": [4.0, 9.5]}}})
    rng = np.random.default_rng(8)
    deg, depth, az = rng.uniform(10.0, 20.0, 5), rng.uniform(0.1, 0.3, 5), rng.uniform(0, 2 * np.pi, 5)
    r = (1.5 + depth) / np.tan(np.radians(deg))
    scene = Scene().add_points(np.column_stack([r * np.cos(az), r * np.sin(az), depth]), amplitude=np.ones(5))
    kw = dict(penetration=penetration, bottom=MEDIUM_SAND, frequency=20e3)
    if penetration == "rough":
        kw["roughness"] = {"sediment": "medium_sand", "K0": 10.0, "seed": 2}
    provider = SedimentProvider(C, MEDIUM_SAND["speed"], 0.0, engine="cpp", **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return simulate_ping(sonar, make_pose(position=(0.0, 0.0, -1.5)), scene, environment={"sound_speed": C},
                             provider=provider, adc=False, backend=backend).samples.array


def test_rough_echo_reaches_the_synthesis():
    """The scattered rays are ordinary eigenrays to the pairing and the synthesis: the ping is not `fluid`'s and is
    far louder: measured 32.0 dB in peak (ours, seed 2); the bound is 20 dB."""
    rough, fluid = _rough_ping(core.Backend.Host), _rough_ping(core.Backend.Host, "fluid")
    assert np.abs(rough).max() > 10.0 * np.abs(fluid).max() > 0


@needs_cuda
def test_cuda_synthesis_takes_the_rough_rays():
    """Eigenrays, `rough`'s included, are computed on the host for every backend (libs/sim/src/paths.cpp,
    penetration.cpp) and paired on the host (two_way_paths); the device synthesises the flat list of pairs
    (libs/sim/src/synth_cuda.cu: channels x paths, no number of rays per point anywhere), so the ping is the host's
    to the order of the float32 atomic additions (tolerance of test_sim.py's test_gpu_forward_model_matches_cpu)."""
    host, cuda = _rough_ping(core.Backend.Host), _rough_ping(core.Backend.Cuda)
    assert np.abs(host).max() > 0
    assert np.abs(cuda - host).max() < 1e-5 * np.abs(host).max()


@needs_cuda
def test_cuda_scenario_matches_the_host():
    a = run_scenario(SCENARIO, {"pings": {"count": 1}}, backend=core.Backend.Host)
    b = run_scenario(SCENARIO, {"pings": {"count": 1}}, backend=core.Backend.Cuda)
    worst, rms = relative_difference(b.records[0].samples.array, a.records[0].samples.array)
    # Echo synthesis also runs on the device here (its own tolerance: tests/python/test_sim.py).
    assert worst < 1e-3 and rms < 1e-3, (worst, rms)


@pytest.mark.parametrize("eng", ["cpp", "numpy"])
def test_wall_coefficient_applies_to_bottom_bounce_paths_too(eng):
    """A path that meets a fluid bottom and then a wall loses to both. (It used to keep the bottom's loss only.)"""
    bottom = dict(speed=1650.0, density=1.9, attenuation=0.8)
    pts = np.array([[150.0, 20.0, 12.0], [300.0, -40.0, 30.0]])
    source = np.array([0.0, 0.0, 10.0])
    kw = dict(surface_z=0.0, bottom_z=60.0, max_order=2, bottom=bottom, engine=eng)
    full = ImageSourceProvider(C, walls=[dict(axis=1, position=200.0, coefficient=1.0)], **kw).eigenrays(source, pts)
    lossy = ImageSourceProvider(C, walls=[dict(axis=1, position=200.0, coefficient=0.5)], **kw).eigenrays(source, pts)
    ratio = np.abs(lossy.amplitude) / np.abs(full.amplitude)
    # Every ray is either untouched by the wall (ratio 1) or halved, and some halved rays met the bottom.
    assert np.all(np.isclose(ratio, 1.0) | np.isclose(ratio, 0.5))
    assert np.any(np.isclose(ratio, 0.5) & (lossy.n_bottom > 0))
    assert np.isclose(ratio, 0.5).sum() == np.isclose(ratio, 1.0).sum()                # one walled copy of each ray
