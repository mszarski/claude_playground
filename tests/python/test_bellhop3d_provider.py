"""P3 of docs/tasks/PROPAGATION_3D.md: `Bellhop3DProvider`, the opt-in `eigenray3d` tier (sim/bellhop3d.py).

Without Bellhop3D: the tier's keys are refused by name with the nearest one, what it cannot take is refused, and a
request whose arrivals file would be too large is refused with the estimate before anything runs. With Bellhop3D
(GPUSONAR_BELLHOP3D): (a) over a level bed it equals the image-source closed form and our 2-D tier; (b) a seamount
centred on the x axis gives mirrored levels and arrival azimuths on mirrored bearings; (c) launch and arrival are unit
3-D vectors and rays over the flanks arrive out of their bearing's plane; (d) the node's delay is carried to an
off-node point within 20 us (the 2-D lookup's bound); and the paths of both pairing engines agree for its eigenrays.
"""

import numpy as np
import pytest

from gpusonar.sim import BellhopProvider, find_bellhop, two_way_paths
from gpusonar.sim.bellhop import OceanEnvironment
from gpusonar.sim.bellhop3d import Bellhop3DError, Bellhop3DProvider, find_bellhop3d
from gpusonar.sim.propagation import ImageSourceProvider, make_provider

needs_bellhop3d = pytest.mark.skipif(find_bellhop3d() is None,
                                     reason="Bellhop3D not installed (GPUSONAR_BELLHOP3D; bellhop3d.exe beside the Bellhop executable is found too)")
needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")

OCEAN = dict(depth=200.0, sound_speed=1500.0, bottom_speed=1700.0, bottom_density=1.9, bottom_attenuation=0.5)
ENV = OceanEnvironment(**OCEAN)
F = 8000.0
SOURCE = np.array([10.0, -20.0, 50.0])                    # not at the origin: the run's frame is the source's
RANGES = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
BEARING = np.radians(30.0)
POINTS = np.column_stack([SOURCE[0] + RANGES * np.cos(BEARING), SOURCE[1] + RANGES * np.sin(BEARING), np.full(5, 90.0)])


def loss(es):
    return -10.0 * np.log10(np.sum(np.where(es.valid, np.abs(es.amplitude) ** 2, 0.0), axis=1))


def test_keys_and_refusals():
    opts = dict(environment=OCEAN, frequency=F, max_range=6000.0)
    assert make_provider("eigenray3d", 1500.0, opts).tier == "eigenray3d"
    with pytest.raises(ValueError, match=r"propagation\.bathymetri: unknown key for fidelity eigenray3d; did you mean 'bathymetry'\?"):
        make_provider("eigenray3d", 1500.0, dict(opts, bathymetri={"plane": {"depth": 200.0}}))
    with pytest.raises(ValueError, match="n_alpha"):
        Bellhop3DProvider(ENV, F, 6000.0, n_alpha=121)
    with pytest.raises(ValueError, match="range profile"):
        Bellhop3DProvider(OceanEnvironment(**dict(OCEAN, bathymetry=[[0.0, 200.0], [1000.0, 150.0]])), F, 6000.0)
    with pytest.raises(ValueError, match="ssp_range"):
        Bellhop3DProvider(OceanEnvironment(**{k: v for k, v in OCEAN.items() if k != "sound_speed"}, ssp_depth=[0.0, 200.0],
                                           ssp_speed=[1500.0, 1500.0], ssp_range=[0.0, 1000.0],
                                           ssp_speed_by_range=[[1500.0, 1500.0], [1500.0, 1500.0]]), F, 6000.0)


def test_a_request_that_would_fill_the_disk_is_refused_before_anything_runs():
    """A dense field: 72 bearings x 50 depths x 61 ranges = 219,600 receivers, about 2.2 GB at 10 kB a receiver."""
    p = Bellhop3DProvider(ENV, F, 6000.0, range_step=100.0, max_file_mb=300.0)
    b = np.radians(np.arange(0.0, 360.0, 5.0))
    r = p.ranges[::1][:61]
    pts = np.array([[rr * np.cos(a), rr * np.sin(a), z] for a in b for rr in r for z in p.depths])
    with pytest.raises(Bellhop3DError, match=r"219600 receivers .* about 2196 MB .* over max_file_mb = 300"):
        p.eigenrays(np.array([0.0, 0.0, 50.0]), pts)
    assert p.runs == []


@pytest.fixture(scope="module")
def level():
    p = Bellhop3DProvider(ENV, F, max_range=5500.0, range_step=100.0, depth_step=20.0, n_beta=61, azimuth_span=20.0)
    return p, p.eigenrays(SOURCE, POINTS)


@needs_bellhop3d
@needs_bellhop
def test_level_bed_equals_the_closed_form_and_the_2d_tier(level, capsys):
    """Five receivers on one bearing, 90 m: one pair of runs (about 3 s each, 18 kB of arrivals). Measured: within 0.16
    dB of the image sources (0.155 at 1 km, under 0.05 from 2 km) and of our 2-D tier (0.16 at 5 km); the TL scaling
    of the merged arrivals is 0.01 dB. Bounds: 0.3 dB, 0.3 dB, 0.1 dB."""
    p, es = level
    img = ImageSourceProvider(1500.0, 0.0, 200.0, bottom=dict(speed=1700.0, density=1.9, attenuation=0.5), max_order=400,
                              engine="numpy").eigenrays(SOURCE, POINTS)
    two = BellhopProvider(ENV, F, max_range=5500.0, range_step=100.0, depth_step=20.0, angles=(-30.0, 30.0)).eigenrays(SOURCE, POINTS)
    to_img, to_two = loss(es) - loss(img), loss(es) - loss(two)
    with capsys.disabled():
        print(f"\n[eigenray3d] level bed: minus image sources {np.round(to_img, 3)}, minus the 2-D tier {np.round(to_two, 3)} dB; "
              f"TL scaling {p.energy_correction_db:.3f} dB; runs {p.runs}")
    assert np.abs(to_img).max() < 0.3 and np.abs(to_two).max() < 0.3
    assert p.energy_correction_db < 0.1
    assert len(p.runs) == 1 and p.runs[0]["receivers"] == 5 and p.runs[0]["bytes"] < 100_000


@needs_bellhop3d
def test_delay_of_the_node_is_carried_to_the_point(level):
    """The direct path's delay against |point - source| / c: on the nodes within 4 us (Bellhop3D's own interpolation
    along the ray); 3, 7 and 4 m off the node within 5 us (measured 4.8). Bound: 20 us, the 2-D lookup's."""
    p, es = level
    exact = np.linalg.norm(POINTS - SOURCE, axis=1) / 1500.0
    direct = es.valid & (es.n_surface == 0) & (es.n_bottom == 0)
    assert np.all(direct.sum(axis=1) == 1)
    assert np.abs(es.delay[direct] - exact).max() < 20e-6
    off = POINTS[2] + np.array([3.0, 7.0, 4.0])
    one = p.eigenrays(SOURCE, off[None])
    assert len(p.runs) == 1                                          # the same node: no new run
    d = one.valid[0] & (one.n_surface[0] == 0) & (one.n_bottom[0] == 0)
    assert abs(one.delay[0][d][0] - np.linalg.norm(off - SOURCE) / 1500.0) < 20e-6


@needs_bellhop3d
def test_both_pairing_engines_take_its_eigenrays(level):
    """two_way_paths pairs any EigenraySet; its 3-D launch and arrival vectors go through the NumPy and the C++ pairing
    alike (host). The synthesis kernels take only the paths."""
    p, _ = level
    amps = np.ones(len(POINTS))
    out = {e: two_way_paths(p, SOURCE, SOURCE, POINTS, amps, engine=e) for e in ("numpy", "cpp")}
    for a, b in zip(out["numpy"][:3], out["cpp"][:3]):
        assert a.shape == b.shape and a.shape[0] > 0
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12)


SEAMOUNT = {"seamount": dict(depth=200.0, height=180.0, center=[2500.0, 0.0], sigma=500.0)}


@needs_bellhop3d
def test_mirrored_bearings_over_a_centred_seamount(capsys):
    """A seamount centred on the x axis (summit 20 m deep at 2.5 km), bearings -8 and +8 degrees (over its flanks),
    3 and 4 km, 50 m; 241 x 61 hat beams over +-38 degrees: one pair of runs, about 19 s each. Measured: levels on the
    mirrored bearings within 0.32 and 0.83 dB (one receiver each; Bellhop3D's TL is not exactly symmetric there), the
    energy-weighted arrival azimuths mirrored within 0.6 degrees, rays arriving up to 13 to 15 degrees out of the
    bearing's plane at 3 km (the formula of section 3 gives 0.8 to 4 degrees a bounce at 2 to 10 degrees grazing over
    the 12 degree flank), unit vectors. Bounds: 1.0 dB, 1.5 degrees, out-of-plane above 0.5 degrees on every point."""
    p = Bellhop3DProvider(ENV, F, max_range=4200.0, bathymetry=SEAMOUNT, range_step=100.0, depth_step=20.0, n_beta=61,
                          azimuth_span=60.0)
    r = np.array([3000.0, 4000.0])
    pts = np.concatenate([np.column_stack([r * np.cos(np.radians(b)), r * np.sin(np.radians(b)), np.full(2, 50.0)]) for b in (-8.0, 8.0)])
    es = p.eigenrays(np.array([0.0, 0.0, 50.0]), pts)
    tl = loss(es)
    v = es.valid
    assert np.allclose(np.linalg.norm(es.launch[v], axis=1), 1.0) and np.allclose(np.linalg.norm(es.arrival[v], axis=1), 1.0)
    az = np.degrees(np.arctan2(es.arrival[..., 1], es.arrival[..., 0]))
    w = np.where(v, np.abs(es.amplitude) ** 2, 0.0)
    mean_az = np.sum(w * az, axis=1) / w.sum(axis=1)
    bearing = np.array([-8.0, -8.0, 8.0, 8.0])
    strong = v & (w > 0.01 * w.max(axis=1, keepdims=True))
    out_of_plane = np.array([np.abs(az[i][strong[i]] - bearing[i]).max() for i in range(4)])
    with capsys.disabled():
        print(f"\n[eigenray3d] centred seamount: TL -8 deg {np.round(tl[:2], 2)}, +8 deg {np.round(tl[2:], 2)} dB; mean arrival "
              f"azimuth {np.round(mean_az, 2)} deg; largest out-of-plane {np.round(out_of_plane, 1)} deg; TL scaling up to "
              f"{p.energy_correction_db:.2f} dB; runs {p.runs}")
    assert np.abs(tl[:2] - tl[2:]).max() < 1.0
    assert np.abs(mean_az[:2] + mean_az[2:]).max() < 1.5
    assert np.all(out_of_plane > 0.5)
