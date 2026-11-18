"""Reverberation from a seabed with relief (`bathymetry`, sim/bathymetry.py and
sim/reverb_relief.py).

Oracles: a uniformly sloping plane seen from a platform is a level plane seen from the
platform's perpendicular height (the flat code, which is tested on its own); reciprocity;
a ridge's shadow from straight-line geometry; a patch on a slope from the sonar equation
with the facet's own angles and area; the point-scatterer seabed on the same bathymetry.
"""

import numpy as np
import pytest

from gpusonar.sim import (BellhopProvider, FreeFieldProvider, ImageSourceProvider, OceanEnvironment, Scene, bathymetry,
                          build_scene, find_bellhop, make_pose, reverb, reverb_bistatic, reverb_relief, simulate_ping)
from gpusonar.sim import engine as sim_engine
from gpusonar.sonar import SonarModel

from test_reverb_bistatic import blocks, patch_map, relative_difference, sectors

C, F, MU_DB = 1500.0, 27.5e3, -27.0
FF = FreeFieldProvider(C)
TIMES = np.arange(0.02, 0.4, 1e-3)
TX, RX = np.array([40.0, 25.0, 15.0]), np.array([0.0, 0.0, 20.0])      # z down: 15 and 20 m deep


# --- the description ---------------------------------------------------------------------------

def test_bathymetry_descriptions():
    p = bathymetry.make_bathymetry({"plane": {"depth": 50.0, "slope": [0.2, -0.1], "point": [10.0, 5.0]}})
    assert p.depth(10.0, 5.0) == pytest.approx(50.0) and p.depth(20.0, 5.0) == pytest.approx(52.0)
    assert p.depth(10.0, 15.0) == pytest.approx(49.0)
    n = p.normal(np.array([3.0]), np.array([4.0]))[0]
    assert n == pytest.approx(np.array([0.2, -0.1, -1.0]) / np.sqrt(1.05))
    assert p.area_factor(0.0, 0.0) == pytest.approx(np.sqrt(1.05))
    assert p.flat_depth is None and bathymetry.plane(30.0).flat_depth == 30.0
    # A grid of the same plane: bilinear interpolation is exact on it, the gradient numerical.
    x, y = np.linspace(-100, 100, 21), np.linspace(-50, 50, 11)
    g = bathymetry.make_bathymetry({"x": x, "y": y, "depth": 50.0 + 0.2 * (x[:, None] - 10.0) - 0.1 * (y[None, :] - 5.0)})
    xs, ys = np.array([-33.3, 12.0, 71.7]), np.array([4.4, -41.0, 17.0])
    assert g.depth(xs, ys) == pytest.approx(p.depth(xs, ys))
    assert g.normal(xs, ys) == pytest.approx(p.normal(xs, ys), abs=1e-9)
    assert g.depth(1e4, 0.0) == pytest.approx(g.depth(100.0, 0.0))            # held beyond the edge
    # Depth against range about an origin: a cone; with azimuth: deeper to the east.
    cone = bathymetry.make_bathymetry({"range": [0.0, 100.0], "depth": [20.0, 40.0], "origin": [5.0, 5.0]})
    assert cone.depth(5.0 + 30.0, 5.0 + 40.0) == pytest.approx(30.0)
    assert np.linalg.norm(cone.gradient(35.0, 45.0)) == pytest.approx(0.2, rel=1e-3)
    fan = bathymetry.make_bathymetry({"range": [0.0, 100.0], "azimuth_deg": [0.0, 90.0, 180.0, 270.0],
                                      "depth": [[30, 30], [30, 50], [30, 30], [30, 10]]})
    assert fan.depth(0.0, 100.0) == pytest.approx(50.0) and fan.depth(0.0, -50.0) == pytest.approx(20.0)
    assert fan.depth(70.7107, 70.7107) == pytest.approx(40.0, abs=1e-3)       # 45 degrees: halfway
    fn = bathymetry.make_bathymetry(lambda x, y: 30.0 + 0.1 * x)
    assert fn.gradient(1.0, 2.0)[0] == pytest.approx(0.1) and fn.gradient(1.0, 2.0)[1] == pytest.approx(0.0, abs=1e-12)
    assert bathymetry.make_bathymetry({"x": [0, 1], "y": [0, 1], "depth": [[7, 7], [7, 7]]}).flat_depth == 7.0
    with pytest.raises(ValueError):
        bathymetry.make_bathymetry({"depth": 3.0})


# --- a sloping plane is a level plane seen from somewhere else ---------------------------------

def rotated(slope, depth0, platform):
    """Perpendicular height of a platform over the plane, and its foot on the plane."""
    s = np.asarray(slope, dtype=float)
    n = np.array([s[0], s[1], -1.0]) / np.sqrt(1.0 + s @ s)
    h = (depth0 + s @ platform[:2] - platform[2]) / np.sqrt(1.0 + s @ s)
    return h, platform - h * n


@pytest.mark.parametrize("slope", [(0.2, 0.0), (0.15, -0.25), (-0.4, 0.3)])
def test_sloping_plane_equals_the_flat_case_rotated(slope):
    """Free field, omnidirectional, Lambert: the reverberation of a plane depends only on
    the platforms' perpendicular heights over it and the distance between their feet. The
    flat code at those heights is the oracle, for the level against time and for the time
    of the first return (the specular path)."""
    bed = bathymetry.plane(50.0, slope)
    (h_t, f_t), (h_r, f_r) = rotated(slope, 50.0, TX), rotated(slope, 50.0, RX)
    sep = np.linalg.norm(f_t - f_r)
    got = reverb_relief.directional_power(FF, TX, RX, bed, TIMES, sectors(180), frequency=F, mu_db=MU_DB)
    ref = reverb_bistatic.directional_power(FF, np.array([sep, 0.0, -h_t]), np.array([0.0, 0.0, -h_r]), 0.0, TIMES,
                                            sectors(180), frequency=F, mu_db=MU_DB)
    a, b = got["power"].sum(axis=(0, 1)), ref["power"].sum(axis=(0, 1))
    first = np.hypot(sep, h_t + h_r) / C
    assert a[TIMES < first - 1e-3].max() == 0.0 and a[TIMES > first + 1e-3].min() > 0.0
    lit = TIMES > first + 5e-3
    ratio = 10 * np.log10(blocks(a[lit])[:-1] / blocks(b[lit])[:-1])
    assert np.abs(ratio).max() < 0.01, ratio


def test_first_return_of_a_slope_is_the_perpendicular():
    """Monostatic: the first return comes from the foot of the perpendicular, at 2 h / c,
    from the azimuth of the up-slope direction, not from straight below."""
    slope = (0.5, 0.0)                                             # deepens to the north: nearest point is south
    bed = bathymetry.plane(50.0, slope)
    h, foot = rotated(slope, 50.0, RX)
    times = np.arange(0.02, 0.2, 2.5e-4)
    got = reverb_relief.directional_power(FF, RX, RX, bed, times, sectors(72), frequency=F, mu_db=MU_DB)
    total = got["power"].sum(axis=(0, 1))
    onset = times[np.flatnonzero(total > 0)[0]]
    assert onset == pytest.approx(2.0 * h / C, abs=3e-4) and 2.0 * h / C < 2.0 * 30.0 / C - 3e-3
    early = got["power"][:, :, times < 2.0 * h / C + 2e-3].sum(axis=(1, 2))
    azimuth = np.angle(np.sum(early * np.exp(1j * sectors(72))))
    assert np.cos(azimuth - np.arctan2(foot[1] - RX[1], foot[0] - RX[0])) > 0.99


def mound(x, y):
    return 50.0 - 12.0 * np.exp(-0.5 * ((x - 60.0) ** 2 + (y + 20.0) ** 2) / 25.0**2)


def test_reciprocity_over_relief():
    """Exchanging transmitter and receiver leaves the power unchanged over a slope and over
    a smooth bank (the cells are laid out about the receiver: two different grids)."""
    for bed in (bathymetry.plane(50.0, (0.15, -0.25)), bathymetry.make_bathymetry(mound)):
        a = reverb_relief.directional_power(FF, TX, RX, bed, TIMES, sectors(180), frequency=F, mu_db=MU_DB)
        b = reverb_relief.directional_power(FF, RX, TX, bed, TIMES, sectors(180), frequency=F, mu_db=MU_DB)
        pa, pb = a["power"].sum(axis=(0, 1)), b["power"].sum(axis=(0, 1))
        lit = pa > 0
        ratio = 10 * np.log10(blocks(pa[lit], 5)[2:-1] / blocks(pb[lit], 5)[2:-1])
        assert np.abs(ratio).max() < 0.1, ratio


# --- facets: area, angles, shadow --------------------------------------------------------------

def test_patch_on_a_slope_has_the_facet_angles_and_area():
    """mu sin(g1) sin(g2) A / (R1^2 R2^2) with g the angles to the facet and A its true
    area (the patch covers 2 pi sigma^2 of horizontal area)."""
    slope = (0.3, -0.2)
    bed = bathymetry.plane(50.0, slope)
    n = np.array([slope[0], slope[1], -1.0]) / np.sqrt(1.0 + 0.13)
    for centre in ([60.0, -30.0], [-40.0, 50.0]):
        point = np.array([centre[0], centre[1], float(bed.depth(*centre))])
        got = reverb_relief.directional_power(FF, TX, RX, bed, TIMES, sectors(720), frequency=F, mu_db=MU_DB,
                                              strength_map=patch_map(centre, 1.0), n_elevation=9,
                                              tx_velocity=(4.0, -3.0, 0.5), rx_velocity=(-2.0, 5.0, -1.0))
        r1, r2 = np.linalg.norm(point - TX), np.linalg.norm(point - RX)
        sin1, sin2 = -(point - TX) @ n / r1, -(point - RX) @ n / r2
        level = 10 ** (MU_DB / 10) * sin1 * sin2 * 2 * np.pi * np.sqrt(1.13) / (r1 * r2) ** 2
        power = got["power"].sum(axis=(0, 1))
        assert power.sum() * (2 * np.pi / 720) == pytest.approx(level, rel=0.02)
        assert (power * TIMES).sum() / power.sum() == pytest.approx((r1 + r2) / C, abs=5e-4)
        # Arrival direction and Doppler follow the cell's true position (its depth included).
        w = got["power"].sum(axis=2)
        direction = (w[..., None] * got["direction"]).sum(axis=(0, 1)) / w.sum()
        assert direction / np.linalg.norm(direction) == pytest.approx((point - RX) / r2, abs=0.02)
        shift = F / C * (np.array([4.0, -3.0, 0.5]) @ (point - TX) / r1 + np.array([-2.0, 5.0, -1.0]) @ (point - RX) / r2)
        assert (w * got["doppler"]).sum() / w.sum() == pytest.approx(shift, abs=1.0)


def test_strength_gets_facet_grazing_angles_and_facet_azimuth():
    slope = (0.3, -0.2)
    bed = bathymetry.plane(50.0, slope)
    n = np.array([slope[0], slope[1], -1.0]) / np.sqrt(1.13)
    seen = {}

    def strength(g_in, g_out, azimuth):
        seen["args"] = (g_in, g_out, azimuth)
        return np.ones_like(g_in)

    reverb_relief.directional_power(FF, TX, RX, bed, TIMES, sectors(360), frequency=F, strength=strength,
                                    min_range=79.5, max_range=80.5, ring_width=1.0)
    g_in, g_out, az = seen["args"]
    m = 57
    phi = sectors(360)[m]
    point = np.array([80.0 * np.cos(phi), 80.0 * np.sin(phi), 0.0])
    point[2] = bed.depth(point[0], point[1])
    a_q, a_p = (point - TX) / np.linalg.norm(point - TX), (point - RX) / np.linalg.norm(point - RX)
    assert g_in[m] == pytest.approx(np.arcsin(-a_q @ n), abs=1e-6)
    assert g_out[m] == pytest.approx(np.arcsin(-a_p @ n), abs=1e-6)
    t_in, t_out = a_q - (a_q @ n) * n, -a_p + (a_p @ n) * n
    assert az[m] == pytest.approx(np.arccos(t_in @ t_out / np.linalg.norm(t_in) / np.linalg.norm(t_out)), abs=1e-6)
    # Monostatic: backscatter in the facet's frame too.
    reverb_relief.directional_power(FF, RX, RX, bed, TIMES, sectors(36), frequency=F, strength=strength)
    assert np.allclose(seen["args"][2], np.pi, atol=1e-5)


def ridge():
    """A bank 20 m high across the x axis between x = 100 and 110 m on a bed 50 m deep."""
    x = np.array([-3000.0, 99.0, 100.0, 110.0, 111.0, 3000.0])
    return bathymetry.grid(x, [-3000.0, 3000.0], np.column_stack([[50.0, 50.0, 30.0, 30.0, 50.0, 50.0]] * 2))


def test_a_ridge_casts_a_shadow_of_the_right_extent():
    """Sonar 30 m above the bed, ridge top 10 m below the sonar from x = 100 to 110 m. The
    ray grazing the far edge of the top reaches the bed at x = 110 x 30 / 10 = 330 m: along
    azimuth phi nothing returns between the far edge (110 / cos phi) and 330 / cos phi, and
    beyond the shadow the level is that of the flat bed. Away from the ridge nothing changes."""
    bed = ridge()
    times = np.arange(0.02, 0.8, 1e-3)
    az = sectors(360)
    got = reverb_relief.directional_power(FF, RX, RX, bed, times, az, frequency=F, mu_db=MU_DB)
    flat = reverb_bistatic.directional_power(FF, RX, RX, 50.0, times, az, frequency=F, mu_db=MU_DB)
    open_water = reverb_relief.directional_power(FF, RX, RX, bed, times, az, frequency=F, mu_db=MU_DB, shadowing=False)
    for m in (0, 40, 325):
        phi = az[m]
        p, q = got["power"][m].sum(axis=0), flat["power"][m].sum(axis=0)
        dark = np.flatnonzero((p < 1e-3 * q) & (q > 0))
        start = 2.0 * np.hypot(110.0 / np.cos(phi), 10.0) / C          # the far edge of the top
        end = 2.0 * np.hypot(330.0 / np.cos(phi), 30.0) / C
        assert times[dark[0]] == pytest.approx(start, abs=3e-3)
        assert times[dark[-1]] == pytest.approx(end, abs=6e-3)
        assert np.all(np.diff(dark) == 1)                              # one shadow, no holes
        beyond = slice(dark[-1] + 8, dark[-1] + 60)
        assert p[beyond].sum() / q[beyond].sum() == pytest.approx(1.0, abs=0.02)
        # Without the shadow test only the facets turned away are dark: the back of the bank.
        unshadowed = open_water["power"][m].sum(axis=0)
        assert unshadowed[dark[len(dark) // 2]] > 0.5 * q[dark[len(dark) // 2]]
    for m in (120, 180, 240):                                          # looking away from the ridge
        p, q = got["power"][m].sum(axis=0), flat["power"][m].sum(axis=0)
        assert np.abs(blocks(p)[3:-1] / blocks(q)[3:-1] - 1.0).max() < 0.01


def test_shadow_is_cast_for_the_transmitter_too():
    """Bistatic: a cell hidden from either platform returns nothing. With the transmitter
    beyond the ridge the bed between the platforms' two shadows is all that is lit near the ridge."""
    bed = ridge()
    times = np.arange(0.02, 0.8, 1e-3)
    tx = np.array([400.0, 0.0, 20.0])
    a = reverb_relief.directional_power(FF, tx, RX, bed, times, sectors(360), frequency=F, mu_db=MU_DB)
    b = reverb_relief.directional_power(FF, RX, tx, bed, times, sectors(360), frequency=F, mu_db=MU_DB)
    pa, pb = a["power"].sum(axis=(0, 1)), b["power"].sum(axis=(0, 1))
    lit = pa > 0
    assert np.abs(10 * np.log10(blocks(pa[lit], 10)[2:-1] / blocks(pb[lit], 10)[2:-1])).max() < 0.5     # reciprocal
    # Along the x axis: the receiver's shadow covers x in (110, 330); the transmitter, 300 m
    # beyond the near edge of the top and 10 m above it, shadows the bed from x = 100 back to -500.
    strength_at = lambda x0: reverb_relief.directional_power(                              # noqa: E731
        FF, tx, RX, bed, times, sectors(360), frequency=F, mu_db=MU_DB, strength_map=patch_map([x0, 0.0], 2.0))["power"].sum()
    lit_level = strength_at(360.0)                                     # beyond the receiver's shadow, in view of both
    assert lit_level > 0.0 and strength_at(105.0) > 0.0                # and the top of the bank
    assert strength_at(200.0) < 1e-6 * lit_level                       # hidden from the receiver
    assert strength_at(60.0) < 1e-6 * lit_level                        # hidden from the transmitter (its shadow reaches x = -500)


# --- eigenrays over relief ---------------------------------------------------------------------

def test_surface_reflected_paths_are_exact_and_bottom_bounces_are_left_out():
    """Over relief a leg may be direct or surface-reflected (a cell is a point in the water:
    the image-source paths to it are exact). A patch on a slope under a surface: the four
    pairs of the two paths, by hand. Paths with a bottom bounce before the cell are not
    used: a provider with a bottom gives the same answer as one without."""
    slope = (0.2, 0.1)
    bed = bathymetry.plane(60.0, slope)
    n = np.array([slope[0], slope[1], -1.0]) / np.sqrt(1.05)
    surface = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1, surface_coefficient=-1.0)
    both = ImageSourceProvider(C, surface_z=0.0, bottom_z=60.0, max_order=3, surface_coefficient=-1.0,
                               bottom_coefficient=0.5)
    centre = [70.0, -30.0]
    point = np.array([centre[0], centre[1], float(bed.depth(*centre))])
    times = np.arange(0.02, 0.5, 1e-3)
    kw = dict(frequency=F, mu_db=MU_DB, strength_map=patch_map(centre, 1.0))
    got = reverb_relief.directional_power(surface, TX, RX, bed, times, sectors(720), **kw)["power"].sum(axis=(0, 1))
    expected = np.zeros(len(times))
    for src_t in (TX, TX * np.array([1, 1, -1])):                      # the platform and its image in the surface
        for src_r in (RX, RX * np.array([1, 1, -1])):
            r1, r2 = np.linalg.norm(point - src_t), np.linalg.norm(point - src_r)
            sin1, sin2 = -(point - src_t) @ n / r1, -(point - src_r) @ n / r2
            level = 10 ** (MU_DB / 10) * sin1 * sin2 * 2 * np.pi * np.sqrt(1.05) / (r1 * r2) ** 2
            expected[int(np.rint(((r1 + r2) / C - times[0]) / 1e-3))] += level
    assert got.sum() * (2 * np.pi / 720) == pytest.approx(expected.sum(), rel=0.02)
    assert len(np.flatnonzero(expected)) >= 3                          # the arrivals are apart in time
    # In time: direct-direct first, the two mixed pairs 3 ms apart (the patch is 6 ms long:
    # they run together), image-image last.
    for lo, hi in ((0.110, 0.125), (0.125, 0.142), (0.142, 0.158)):
        window = (times >= lo) & (times < hi)
        assert expected[window].sum() > 0.0
        assert got[window].sum() * (2 * np.pi / 720) == pytest.approx(expected[window].sum(), rel=0.02)
    with_bottom = reverb_relief.directional_power(both, TX, RX, bed, times, sectors(720), **kw)["power"].sum(axis=(0, 1))
    np.testing.assert_allclose(with_bottom, got, rtol=1e-9)


def test_what_leaving_out_bottom_bounces_costs_and_the_flat_approximation():
    """The size of what the default leaves out, where it can be computed: a level bed 60 m
    deep under a surface, bottom reflection coefficient 0.5, sonar 40 m above the bed. The
    paths with a bottom bounce before the cell carry under 0.2 dB of the bottom
    reverberation out to 2 water depths of range, and up to 5.7 dB by 6 water depths.
    `bottom_bounces="flat"` keeps them (earlier bounces off the provider's flat bottom): on
    a level bed that is the full answer."""
    surface = ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1, surface_coefficient=-1.0)
    both = ImageSourceProvider(C, surface_z=0.0, bottom_z=60.0, max_order=4, surface_coefficient=-1.0,
                               bottom_coefficient=0.5)
    times = np.arange(0.06, 0.5, 1e-3)
    az = sectors(12)
    full = reverb_bistatic.directional_power(both, RX, RX, 60.0, times, az, frequency=F, mu_db=MU_DB)["power"].sum(axis=(0, 1))
    kept = reverb_bistatic.directional_power(surface, RX, RX, 60.0, times, az, frequency=F, mu_db=MU_DB)["power"].sum(axis=(0, 1))
    loss = 10 * np.log10(blocks(full, 40) / blocks(kept, 40))
    assert np.all(loss > -1e-9) and loss.max() < 5.7, loss
    assert loss[:2].max() < 0.2, loss                                  # to 0.14 s: 105 m of slant range
    # The relief code on a bed that happens to be level (a function, so not recognised as flat).
    level = bathymetry.make_bathymetry(lambda x, y: 60.0 + 0.0 * x)
    excluded = reverb_relief.directional_power(both, RX, RX, level, times, az, frequency=F, mu_db=MU_DB)["power"].sum(axis=(0, 1))
    flat = reverb_relief.directional_power(both, RX, RX, level, times, az, frequency=F, mu_db=MU_DB,
                                           bottom_bounces="flat")["power"].sum(axis=(0, 1))
    assert np.abs(10 * np.log10(blocks(excluded, 40)[1:] / blocks(kept, 40)[1:])).max() < 0.02
    assert np.abs(10 * np.log10(blocks(flat, 40)[1:] / blocks(full, 40)[1:])).max() < 0.02


# --- through add_reverberation and the simulator -----------------------------------------------

def ping(tx=TX, rx=RX, ranges=(25.0, 230.0), scene=None, **kw):
    sonar = SonarModel.load("popoto_8", {
        "receive": {"kind": "baseband", "sample_rate": 20e3, "window": {"range": list(ranges)}, "sensitivity_db": None},
        "transmit": {"lever_arm": [0, 0, 0], "waveform": {"type": "cw", "f_start": F, "f_stop": F, "duration": 2e-3,
                                                          "window_alpha": 0.0, "source_level_db_rms": None,
                                                          "amplitude": 1.0}}})
    return simulate_ping(sonar, make_pose(position=rx), scene or Scene(), environment={"sound_speed": C},
                         absorption_model="none", adc=False, tx_pose=make_pose(position=tx), **kw)


def test_level_bathymetry_is_the_flat_code_bit_for_bit():
    def run(**kw):
        rec = ping()
        reverb.add_reverberation(rec, FF, 50.0, MU_DB, np.random.default_rng(2), n_azimuth=24, geometry="bistatic", **kw)
        return rec.samples.array

    plain = run()
    assert np.array_equal(run(bathymetry={"plane": {"depth": 50.0}}), plain)
    assert np.array_equal(run(bathymetry={"x": [-1e4, 1e4], "y": [-1e4, 1e4], "depth": [[50.0, 50.0], [50.0, 50.0]]}), plain)
    assert np.array_equal(run(bathymetry={"range": [0.0, 1e4], "depth": [50.0, 50.0]}), plain)
    assert not np.array_equal(run(bathymetry={"plane": {"depth": 50.0, "slope": [0.05, 0.0]}}), plain)
    with pytest.raises(ValueError, match="bistatic"):
        rec = ping()
        reverb.add_reverberation(rec, FF, 50.0, MU_DB, np.random.default_rng(2), bathymetry={"plane": {"depth": 50.0}})
    with pytest.raises(ValueError, match="below the seabed"):
        run(bathymetry={"plane": {"depth": 10.0, "slope": [0.01, 0.0]}})


@pytest.mark.skipif(not sim_engine.HAVE_CPP, reason="needs the C++ simulator engine")
@pytest.mark.parametrize("bed", [{"plane": {"depth": 50.0, "slope": [0.15, -0.25]}}, "ridge", "mound"])
def test_relief_on_both_engines(bed):
    spec = {"ridge": ridge(), "mound": mound}.get(bed, bed) if isinstance(bed, str) else bed
    out = []
    for engine in ("cpp", "numpy"):
        rec = ping()
        rng = np.random.default_rng(3)
        power = reverb.add_reverberation(rec, FreeFieldProvider(C, engine=engine), 0.0, MU_DB, rng, n_azimuth=48,
                                         geometry="bistatic", bathymetry=spec, engine=engine)
        out.append((rec.samples.array, power, rng.bit_generator.state))
    assert np.abs(out[1][0]).max() > 0.0
    worst, rms = relative_difference(out[0][0], out[1][0])
    assert worst < 2e-5 and rms < 2e-5, (worst, rms)
    assert out[0][2] == out[1][2]


def test_scatterer_seabed_and_scattering_function_on_the_same_slope():
    """One `bathymetry` description for both methods: the point-scatterer seabed of a scene
    and the scattering-function reverberation give the same level on a slope."""
    spec = {"plane": {"depth": 50.0, "slope": [0.25, -0.15]}}
    point = []
    for seed in (11, 12, 13):
        scene = build_scene([{"type": "seafloor", "x_range": [-260.0, 300.0], "y_range": [-260.0, 285.0], "density": 1.0,
                              "lambert_mu_db": MU_DB, "bathymetry": spec}], np.random.default_rng(seed))
        bed = bathymetry.make_bathymetry(spec)
        assert scene.at_time(0.0)[:, 2] == pytest.approx(bed.depth(scene.at_time(0.0)[:, 0], scene.at_time(0.0)[:, 1]))
        point.append(ping(scene=scene).samples.array)
    sf = []
    for seed in (21, 22, 23):
        rec = ping(reverberation=[{"mu_db": MU_DB, "geometry": "bistatic", "bathymetry": spec}],
                   rng=np.random.default_rng(seed))
        sf.append(rec.samples.array)
    a = (np.abs(np.concatenate(point)) ** 2).mean(axis=0)
    b = (np.abs(np.concatenate(sf)) ** 2).mean(axis=0)
    span = slice(800, 4000)                                           # 0.07 to 0.23 s
    ratio = 10 * np.log10(a[span].reshape(8, -1).mean(axis=1) / b[span].reshape(8, -1).mean(axis=1))
    assert np.abs(ratio).max() < 2.0, ratio
    assert abs(ratio.mean()) < 1.0, ratio


# --- Bellhop over relief: N x 2-D --------------------------------------------------------------

needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")


def shallow_provider():
    env = OceanEnvironment(depth=30.0, sound_speed=C, bottom_speed=1600.0, bottom_density=1.8, bottom_attenuation=0.5)
    return BellhopProvider(env, F, max_range=420.0, range_step=5.0, depth_step=0.5, max_arrivals=16)


@needs_bellhop
def test_bellhop_over_a_level_bed_by_bearing_equals_the_flat_run():
    """One Bellhop run per azimuth sector with that bearing's depth profile. Over a level
    bed (given as a function, so the relief code runs) every bearing's profile is flat and
    the result is the flat code's with the plain provider, bottom bounces included."""
    provider = shallow_provider()
    rx = np.array([0.0, 0.0, 5.0])
    times = np.arange(0.04, 0.5, 2e-3)
    flat = reverb_bistatic.directional_power(provider, rx, rx, 30.0, times, sectors(4), frequency=F, mu_db=MU_DB)
    level = reverb_relief.directional_power(provider, rx, rx, bathymetry.make_bathymetry(lambda x, y: 30.0 + 0.0 * x),
                                            times, sectors(4), frequency=F, mu_db=MU_DB)
    a, b = level["power"].sum(axis=1), flat["power"].sum(axis=1)
    for m in range(4):
        assert np.abs(10 * np.log10(blocks(a[m], 20) / blocks(b[m], 20))).max() < 0.1
    # The provider's bottom-bounce paths are in: more than the direct and surface paths alone.
    direct = reverb_bistatic.directional_power(ImageSourceProvider(C, surface_z=0.0, bottom_z=None, max_order=1), rx, rx,
                                               30.0, times, sectors(4), frequency=F, mu_db=MU_DB)["power"].sum(axis=1)
    assert blocks(a[0], 20)[-1] > 1.5 * blocks(direct[0], 20)[-1]


@needs_bellhop
def test_bellhop_bank_on_one_bearing_changes_that_bearing_only():
    """A bank in the north-east quadrant, shoaling from 30 m to 15 m around 200 m range: the
    slope facing the sonar returns more, the bearings without the bank are untouched."""
    def bank(x, y):
        r = np.hypot(x, y)
        return 30.0 - np.where((x > 0) & (y > 0), 15.0 * np.clip(1.0 - np.abs(r - 200.0) / 50.0, 0.0, 1.0), 0.0)

    provider = shallow_provider()
    rx = np.array([0.0, 0.0, 5.0])
    times = np.arange(0.04, 0.5, 2e-3)
    flat = reverb_bistatic.directional_power(provider, rx, rx, 30.0, times, sectors(4), frequency=F, mu_db=MU_DB)["power"].sum(axis=1)
    got = reverb_relief.directional_power(provider, rx, rx, bathymetry.make_bathymetry(bank), times, sectors(4),
                                          frequency=F, mu_db=MU_DB)["power"].sum(axis=1)
    ratio = [10 * np.log10(blocks(got[m], 20) / blocks(flat[m], 20)) for m in range(4)]
    for m in (1, 2, 3):
        assert np.abs(ratio[m]).max() < 0.1
    facing = (times[::20][: len(ratio[0])] > 2 * 150.0 / C - 0.02) & (times[::20][: len(ratio[0])] < 2 * 200.0 / C)
    assert ratio[0][facing].max() > 3.0                                # the up-slope, 150 to 200 m
    assert np.abs(ratio[0][:3]).max() < 0.1                            # before the bank
    with pytest.raises(NotImplementedError, match="N x 2-D"):
        reverb_relief.directional_power(provider, np.array([50.0, 0.0, 5.0]), rx, bathymetry.make_bathymetry(bank),
                                        times, sectors(4), frequency=F, mu_db=MU_DB)


def test_differing_element_patterns_are_reported():
    """The generator shares one sequence per direction among the channels: one receive
    pattern for all. A record whose elements differ says so."""
    rec = ping()
    geo = rec.geometry
    rx = list(geo.rx)
    rx[3].beamwidth_az = 0.5
    geo.rx = rx
    rec.geometry = geo
    if len({reverb._pattern_key(e) for e in rec.geometry.rx}) > 1:
        with pytest.warns(RuntimeWarning, match="different beam patterns"):
            reverb.add_reverberation(rec, FF, 50.0, MU_DB, np.random.default_rng(0), n_azimuth=12)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reverb.add_reverberation(ping(), FF, 50.0, MU_DB, np.random.default_rng(0), n_azimuth=12)
