"""Milestone 12 (part): propagation providers against Bellhop.

Bellhop (GPL-3) is an external oracle, run as a separate process. These tests are
skipped when it is not installed (set GPUSONAR_BELLHOP to bellhop.exe).
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.sim import (BellhopProvider, ImageSourceProvider, OceanEnvironment, Scene, find_bellhop,
                          make_pose, munk_profile, rayleigh_reflection, run_bellhop, simulate_ping)
from gpusonar.sonar import SonarModel

needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")

PEKERIS = dict(depth=100.0, sound_speed=1500.0, bottom_speed=1600.0, bottom_density=1.8, bottom_attenuation=0.5)
BOTTOM = dict(speed=1600.0, density=1.8, attenuation=0.5)


# --- Rayleigh reflection ---------------------------------------------------------------------

def test_rayleigh_reflection_limits():
    # Normal incidence: the impedance contrast.
    r = rayleigh_reflection(np.pi / 2, 1500.0, speed=1600.0, density=1.8)
    assert r == pytest.approx((1.8 * 1600 - 1500) / (1.8 * 1600 + 1500))
    # Below the critical angle acos(c_w / c_b) a lossless bottom reflects everything.
    critical = np.arccos(1500.0 / 1600.0)
    g = np.radians([1.0, 5.0, 0.9 * np.degrees(critical)])
    assert np.allclose(np.abs(rayleigh_reflection(g, 1500.0, speed=1600.0, density=1.8)), 1.0)
    # Above it, energy leaks into the sediment.
    assert abs(rayleigh_reflection(1.2 * critical, 1500.0, speed=1600.0, density=1.8)) < 0.6
    # Attenuation in the sediment takes a little off even below critical.
    lossy = abs(rayleigh_reflection(np.radians(5.0), 1500.0, **BOTTOM))
    assert 0.9 < lossy < 1.0
    # Grazing incidence: R -> -1.
    assert rayleigh_reflection(1e-6, 1500.0, **BOTTOM) == pytest.approx(-1.0, abs=1e-4)


# --- image source against Bellhop: Pekeris waveguide ---------------------------------------------

@needs_bellhop
@pytest.mark.parametrize("frequency", [1000.0, 8000.0])
def test_image_source_matches_bellhop_in_a_pekeris_waveguide(frequency):
    """Isovelocity water over a fluid half-space: the case the image-source provider claims."""
    env = OceanEnvironment(**PEKERIS)
    source_depth, receiver_depth = 30.0, 50.0
    ranges = np.arange(100.0, 3001.0, 50.0)
    arr = run_bellhop(env, frequency, source_depth, [receiver_depth], ranges)
    provider = ImageSourceProvider(1500.0, surface_z=0.0, bottom_z=100.0, max_order=40, bottom=BOTTOM)
    pts = np.column_stack([ranges, np.zeros_like(ranges), np.full_like(ranges, receiver_depth)])
    es = provider.eigenrays(np.array([0.0, 0.0, source_depth]), pts)

    # Arrival delays of the first four paths (direct, surface, bottom, surface-bottom).
    # Bellhop can list one eigenray twice (neighbouring beams), so match by nearest delay.
    for j in (0, 18, 40, 58):
        for d in np.sort(es.delay[j])[:4]:
            assert np.abs(arr.delay[0][j] - d).min() < 2e-6

    # Transmission loss, incoherent (energy) and coherent (with every bounce phase).
    tl_img_inc = -10 * np.log10((np.abs(es.amplitude) ** 2).sum(axis=1))
    tl_bh_inc = arr.transmission_loss(coherent=False)[0]
    assert np.abs(tl_img_inc - tl_bh_inc).max() < 1.0
    assert np.median(np.abs(tl_img_inc - tl_bh_inc)) < 0.2
    if frequency <= 1000.0:
        # Coherent comparison where the interference pattern is resolved by the 50 m range
        # sampling and insensitive to microsecond delay differences.
        p_img = (es.amplitude * np.exp(-2j * np.pi * frequency * es.delay)).sum(axis=1)
        tl_img = -20 * np.log10(np.abs(p_img))
        tl_bh = arr.transmission_loss()[0]
        assert np.median(np.abs(tl_img - tl_bh)) < 0.3
        assert np.abs(tl_img - tl_bh).max() < 1.0
    # Shallow-water ducting: far less loss than spherical spreading at 3 km.
    assert tl_bh_inc[-1] < 20 * np.log10(3000.0) - 5.0


@needs_bellhop
def test_bounce_phases_use_this_suites_time_convention():
    env = OceanEnvironment(**PEKERIS)
    arr = run_bellhop(env, 1000.0, 30.0, [50.0], [1000.0])
    ns, nb = arr.n_surface[0][0], arr.n_bottom[0][0]
    amp = arr.amplitude[0][0]

    def first(n_surface, n_bottom):
        return int(np.flatnonzero((ns == n_surface) & (nb == n_bottom))[0])

    assert np.angle(amp[first(0, 0)]) == pytest.approx(0.0, abs=1e-6)             # direct
    assert abs(np.angle(amp[first(1, 0)])) == pytest.approx(np.pi, abs=1e-6)      # pressure release
    i = first(0, 1)                                                              # one bottom bounce
    ours = rayleigh_reflection(abs(arr.arrival_angle[0][0][i]), 1500.0, **BOTTOM)
    assert np.angle(amp[i]) == pytest.approx(np.angle(ours), abs=np.radians(3.0))
    assert abs(np.angle(ours)) > 1.0      # a real phase shift, so the sign convention is tested


# --- Munk profile: the Bellhop provider itself -----------------------------------------------------

@needs_bellhop
def test_munk_profile_shows_refraction_the_straight_ray_model_cannot():
    z, c = munk_profile()
    env = OceanEnvironment(depth=5000.0, ssp_depth=z, ssp_speed=c, bottom_speed=1600.0,
                           bottom_density=1.8, bottom_attenuation=0.8)
    ranges = np.arange(1000.0, 100001.0, 1000.0)
    arr = run_bellhop(env, 50.0, 1000.0, [1000.0], ranges, angles=(-20.3, 20.3))
    tl = arr.transmission_loss(coherent=False)[0]
    spherical = 20 * np.log10(ranges)
    # Sound-channel propagation: tens of dB better than spherical spreading at 100 km...
    assert tl[-1] < spherical[-1] - 10.0
    # ...and every path at long range is a refracted one that never touches the bottom.
    far = len(ranges) - 1
    strongest = np.argsort(-np.abs(arr.amplitude[0][far]))[:5]
    assert arr.n_bottom[0][far][strongest[0]] == 0
    assert (arr.n_bottom[0][far][strongest] == 0).sum() >= 3
    # Travel time exceeds range / c_min by a little: rays cycle about the channel axis.
    assert arr.delay[0][far].min() == pytest.approx(100000.0 / 1500.0, rel=0.01)


@needs_bellhop
def test_provider_lookup_reproduces_a_direct_bellhop_run_off_grid():
    """Arrivals are cached on a grid; between nodes the delay is corrected with the arrival
    angle. Check against Bellhop run exactly at the off-grid points."""
    env = OceanEnvironment(**PEKERIS)
    provider = BellhopProvider(env, 8000.0, max_range=2000.0, range_step=20.0, depth_step=2.0, max_arrivals=6)
    source = np.array([0.0, 0.0, 30.0])
    pts = np.array([[1003.0, 0.0, 40.7], [0.0, 1507.0, 61.2], [-611.0, 611.0, 22.3]])
    es = provider.eigenrays(source, pts)
    assert es.valid.any(axis=1).all()
    for i, p in enumerate(pts):
        rng = np.hypot(p[0], p[1])
        exact = run_bellhop(env, 8000.0, 30.0, [p[2]], [rng])
        d_exact = np.sort(exact.delay[0][0])[:3]
        d_lookup = np.sort(es.delay[i][es.valid[i]])[:3]
        # Half a grid cell away at most; the first-order correction leaves microseconds.
        np.testing.assert_allclose(d_lookup, d_exact, atol=20e-6)
        # Launch directions point from the source towards the point, in the horizontal.
        horizontal = es.launch[i][es.valid[i]][:, :2]
        bearing = horizontal / np.linalg.norm(horizontal, axis=1)[:, None]
        np.testing.assert_allclose(bearing, np.tile(p[:2] / rng, (len(bearing), 1)), atol=1e-9)
    # One Bellhop run served all three points.
    assert len(provider._cache) == 1


# --- echoes through the eigenray tier ----------------------------------------------------------------

@needs_bellhop
def test_echo_at_the_eigenray_tier_matches_the_image_source_tier():
    """Same shallow isovelocity water, two providers: the multipath echo of a target at 1 km
    must have the same arrival structure."""
    c = 1500.0
    over = {"receive": {"window": {"range": [960.0, 1100.0]}, "kind": "baseband"},
            "transmit": {"waveform": {"type": "gaussian", "f_start": 6e3, "f_stop": 10e3,
                                      "duration": 2e-3, "window_alpha": 0.0}, "lever_arm": [0, 0, 0]}}
    sonar = SonarModel.load("flank_32", over)
    pose = make_pose(position=(0.0, 0.0, 30.0))
    target = np.array([0.0, 1000.0, 50.0])
    scene = Scene().add_points([target], target_strength_db=10.0)
    kw = dict(environment={"sound_speed": c}, absorption_model="none", adc=False)
    image = ImageSourceProvider(c, surface_z=0.0, bottom_z=100.0, max_order=6, bottom=BOTTOM)
    bellhop = BellhopProvider(OceanEnvironment(**PEKERIS), 8000.0, max_range=1200.0, range_step=10.0,
                              depth_step=2.0, max_arrivals=12)
    a = simulate_ping(sonar, pose, scene, provider=image, **kw).samples.array[16]
    b = simulate_ping(sonar, pose, scene, provider=bellhop, **kw).samples.array[16]
    direct = simulate_ping(sonar, pose, scene, **kw).samples.array[16]
    # Multipath is strong here: the echo lasts far longer than the direct-path one.
    assert (np.abs(a) > 0.05 * np.abs(a).max()).sum() > 3 * (np.abs(direct) > 0.05 * np.abs(direct).max()).sum()
    # Envelopes from the two providers agree (carrier phase is sensitive to microseconds
    # of delay at 8 kHz, so compare smoothed energy over time).
    def envelope(x):
        e = np.abs(x) ** 2
        k = np.ones(16) / 16
        return np.convolve(e, k, mode="same")
    ea, eb = envelope(a), envelope(b)
    corr = np.sum(ea * eb) / np.sqrt(np.sum(ea**2) * np.sum(eb**2))
    assert corr > 0.9, corr
    assert np.sum(eb) == pytest.approx(np.sum(ea), rel=0.25)       # total echo energy within 1 dB


def test_make_provider_builds_the_eigenray_tier():
    if find_bellhop() is None:
        pytest.skip("Bellhop not installed (GPUSONAR_BELLHOP)")
    p = gs.sim.make_provider("eigenray", 1500.0, {"environment": PEKERIS, "frequency": 8000.0,
                                                  "max_range": 500.0})
    assert isinstance(p, BellhopProvider) and p.tier == "eigenray"



def test_eigenray_tier_assumes_no_seabed():
    """Audit A8 (D-104 for Bellhop): the ocean used to fall back to 1600 m/s, 1.8 g/cm^3, 0.5 dB per wavelength."""
    with pytest.raises(ValueError, match="no seabed is assumed"):
        OceanEnvironment(depth=100.0, sound_speed=1500.0)
    with pytest.raises(ValueError, match=r"missing \['bottom_attenuation'\]"):
        OceanEnvironment(depth=100.0, sound_speed=1500.0, bottom_speed=1600.0, bottom_density=1.8)
    with pytest.raises(ValueError, match="no seabed is assumed"):
        gs.sim.make_provider("eigenray", 1500.0, {"environment": {"depth": 100.0, "sound_speed": 1500.0}, "frequency": 8000.0,
                                                  "max_range": 500.0})
    env = OceanEnvironment(**PEKERIS)
    assert (env.bottom_speed, env.bottom_density, env.bottom_attenuation) == (1600.0, 1.8, 0.5)
    sand = OceanEnvironment(depth=100.0, sound_speed=1500.0, sediment="medium_sand")
    assert sand.sediment_name == "medium_sand" and sand.bottom_speed > 1500.0
    with pytest.raises(ValueError, match="not both"):
        OceanEnvironment(depth=100.0, sound_speed=1500.0, sediment="medium_sand", bottom_speed=1600.0)

# --- range-dependent bathymetry: a ridge between source and receiver -------------------------

@needs_bellhop
def test_ridge_shadows_and_strips_steep_paths():
    """A ridge rising to 60 m in 200 m of water. Receivers inside the ridge get nothing;
    beyond it the loss is higher than over a flat bottom (up-slope reflection steepens
    rays past the critical angle); before it nothing changes."""
    base = dict(depth=200.0, sound_speed=1500.0, bottom_speed=1600.0, bottom_density=1.8, bottom_attenuation=0.5)
    ridge = [[0.0, 200.0], [3000.0, 200.0], [4000.0, 60.0], [5000.0, 200.0], [8000.0, 200.0]]
    ranges = np.array([1000.0, 2500.0, 4000.0, 6000.0, 8000.0])
    flat = run_bellhop(OceanEnvironment(**base), 8000.0, 30.0, [150.0], ranges)
    env = OceanEnvironment(**base, bathymetry=ridge)
    assert env.depth_at(4000.0) == pytest.approx(60.0) and env.depth_at(3500.0) == pytest.approx(130.0)
    with_ridge = run_bellhop(env, 8000.0, 30.0, [150.0], ranges)
    tl_flat = flat.transmission_loss(coherent=False)[0]
    tl_ridge = with_ridge.transmission_loss(coherent=False)[0]
    np.testing.assert_allclose(tl_ridge[:2], tl_flat[:2], atol=0.3)        # before the ridge
    assert len(with_ridge.amplitude[0][2]) == 0                            # 150 m deep under a 60 m crest
    assert np.all(tl_ridge[3:] > tl_flat[3:] + 1.5)                        # behind it
    # The provider carries it through: no eigenrays to a point inside the seabed.
    provider = BellhopProvider(env, 8000.0, max_range=8200.0, range_step=100.0, depth_step=10.0, max_arrivals=40)
    es = provider.eigenrays(np.array([0.0, 0.0, 30.0]), np.array([[4000.0, 0.0, 150.0], [6000.0, 0.0, 150.0]]))
    assert not es.valid[0].any() and es.valid[1].any()


@needs_bellhop
def test_bathymetry_is_validated():
    env = OceanEnvironment(**PEKERIS, bathymetry=[[0.0, 100.0], [1000.0, 150.0]])
    from gpusonar.sim.bellhop import BellhopError
    with pytest.raises(BellhopError, match="bathymetry"):
        run_bellhop(env, 8000.0, 30.0, [50.0], [500.0])


def test_every_provider_names_its_own_tier():
    """The sediment provider used to call itself `image_source`."""
    for tier, options in (("free_field", {}), ("image_source", {"bottom_z": None}),
                          ("sediment", {"sediment_speed": 1650.0, "interface_z": 50.0})):
        assert gs.sim.make_provider(tier, 1500.0, options).tier == tier
