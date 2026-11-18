"""Two defaults that were silently wrong (audit S-1, S-3/M-2): a bottom now needs a description (no made-up reflection
coefficient), and the number of bounces or arrivals left to the provider follows an energy criterion with a cap that
is reported when it cuts (D-95, D-96)."""

import warnings

import numpy as np
import pytest

from gpusonar.sim.bellhop import Arrivals, BellhopProvider, OceanEnvironment
from gpusonar.sim.propagation import ImageSourceProvider, rayleigh_reflection
from gpusonar.sim.simulator import run_scenario

C = 1500.0
SRC = np.array([0.0, 0.0, 30.0])
PTS = np.array([[0.0, 1000.0, 50.0], [0.0, 3000.0, 20.0], [0.0, 300.0, 80.0]])
SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}


def test_a_bottom_needs_a_description():
    with pytest.raises(ValueError, match="no description"):
        ImageSourceProvider(C, bottom_z=100.0, max_order=1)
    ImageSourceProvider(C, bottom_z=100.0, max_order=1, bottom_coefficient=0.5)          # an explicit constant still works
    ImageSourceProvider(C, bottom_z=100.0, max_order=1, bottom=SAND)
    ImageSourceProvider(C, max_order=1)                                                  # no bottom: nothing to describe
    with pytest.raises(ValueError, match="wall needs"):
        ImageSourceProvider(C, max_order=1, walls=[{"axis": 0, "position": 400.0}])
    base = {"name": "t", "sonar": "flank_32", "fidelity": "image_source", "sonar_overrides": {"receive": {"window": {"range": [950.0, 1050.0]}}},
            "platform": {"position": [0, 0, 30.0]}, "pings": {"count": 1},
            "scene": [{"type": "points", "positions": [[0.0, 1000.0, 50.0]], "target_strength_db": 10.0}], "receiver": {"adc": False}}
    with pytest.raises(ValueError, match="no description"):
        run_scenario({**base, "environment": {"water_depth_m": 100.0}, "propagation": {"max_order": 1}})


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_the_default_bottom_is_the_rayleigh_reflection_of_the_environments_sediment(engine):
    from gpusonar import environment
    from gpusonar.sim import bottom as bottom_model
    from gpusonar.sim.propagation import make_provider
    cfg = environment.apply_to_scenario({"fidelity": "image_source", "propagation": {"max_order": 1}},
                                        environment.resolve({"water_depth_m": 100.0, "bottom": "medium_sand", "sound_speed_m_s": C}, strict=True))
    assert cfg["propagation"] == {"max_order": 1, "bottom_z": 100.0, "bottom": {"sediment": "medium_sand"}}
    provider = make_provider("image_source", C, cfg["propagation"])
    provider.engine = engine
    rays = provider.eigenrays(SRC, PTS)
    half = bottom_model.fluid_halfspace("medium_sand", C)
    one_bottom = (rays.n_bottom == 1) & (rays.n_surface == 0)
    for i in range(len(PTS)):
        j = int(np.flatnonzero(one_bottom[i])[0])
        r = C * rays.delay[i, j]
        grazing = np.arcsin(abs(rays.launch[i, j, 2]))
        assert rays.amplitude[i, j] * r == pytest.approx(complex(rayleigh_reflection(grazing, C, **half)), rel=1e-6)
    assert abs(rays.amplitude[1, int(np.flatnonzero(one_bottom[1])[0])]) * C * rays.delay[1, int(np.flatnonzero(one_bottom[1])[0])] > 0.9      # below the critical angle


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_bounce_order_follows_the_energy_criterion(engine):
    # 14 dB a bottom bounce: orders are kept while an arrival within 20 dB of the strongest exists
    lossy = ImageSourceProvider(C, bottom_z=100.0, bottom_coefficient=0.2, engine=engine)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rays = lossy.eigenrays(SRC, PTS)
    assert lossy.truncation is None
    full = ImageSourceProvider(C, bottom_z=100.0, bottom_coefficient=0.2, max_order=9, engine="numpy").eigenrays(SRC, PTS)
    rel = np.abs(full.amplitude) / np.abs(full.amplitude).max(axis=1, keepdims=True)
    bounces = (full.n_surface + full.n_bottom)[0]
    want = max(int(n) for n in np.unique(bounces) if rel[:, bounces == n].max() >= 0.1)
    assert lossy.order_used == want and 2 <= want <= 4
    assert (rays.n_surface + rays.n_bottom).max() == want
    # everything left out is below the floor; everything kept is the same ray as in the long series
    left = np.abs(full.amplitude)[:, bounces > want] / np.abs(full.amplitude).max(axis=1, keepdims=True)
    assert left.max() < 0.1
    energy = lambda e: np.sum(np.abs(e.amplitude) ** 2, axis=1)                  # noqa: E731
    assert np.all(energy(rays) / energy(full) > 0.985)
    # a stated order is honoured, silently
    two = ImageSourceProvider(C, bottom_z=100.0, bottom_coefficient=0.2, max_order=1, engine=engine)
    assert (two.eigenrays(SRC, PTS).n_surface + two.eigenrays(SRC, PTS).n_bottom).max() == 1 and two.order_used == 1
    # a tighter floor keeps fewer, a looser one more
    orders = [ImageSourceProvider(C, bottom_z=100.0, bottom_coefficient=0.2, engine=engine, arrival_floor_db=f) for f in (6.0, 40.0)]
    for p in orders:
        p.eigenrays(SRC, PTS)
    assert orders[0].order_used < want < orders[1].order_used


def test_a_lossless_duct_hits_the_cap_and_says_so():
    duct = ImageSourceProvider(C, bottom_z=100.0, bottom_coefficient=1.0, max_order_cap=5)
    with pytest.warns(RuntimeWarning, match="cut at the cap of 5 bounces"):
        rays = duct.eigenrays(SRC, PTS)
    assert duct.order_used == 5 and (rays.n_surface + rays.n_bottom).max() == 5
    assert duct.truncation["order"] == 5 and -20.0 < duct.truncation["left_out_db"] < 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        duct.eigenrays(SRC, PTS)                                                 # once per provider
    both = [ImageSourceProvider(C, bottom_z=100.0, bottom=SAND, engine=e) for e in ("cpp", "numpy")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = (p.eigenrays(SRC, PTS) for p in both)
    assert both[0].order_used == both[1].order_used and a.delay.shape == b.delay.shape and np.allclose(a.amplitude, b.amplitude, rtol=1e-9, atol=1e-15)


def _fake(provider, amplitudes):
    """Arrivals of the given amplitudes at every grid node (no Bellhop needed)."""
    arr = Arrivals(provider.frequency, provider.depths, provider.ranges)
    n = len(amplitudes)
    cell = lambda v: [[np.array(v, dtype=float if not np.iscomplexobj(v) else complex) for _ in provider.ranges] for _ in provider.depths]     # noqa: E731
    arr.amplitude = cell(np.asarray(amplitudes, dtype=complex))
    arr.delay = cell(0.7 + 0.01 * np.arange(n))
    arr.launch_angle = cell(np.linspace(-0.2, 0.2, n))
    arr.arrival_angle = cell(np.linspace(-0.2, 0.2, n))
    arr.n_surface = [[np.arange(n) % 3 for _ in provider.ranges] for _ in provider.depths]
    arr.n_bottom = [[np.arange(n) % 2 for _ in provider.ranges] for _ in provider.depths]
    provider._cache[round(float(SRC[2]), 2)] = arr
    return provider


def test_arrivals_kept_follow_the_energy_criterion():
    ocean = OceanEnvironment(depth=100.0, bottom_speed=1650.0, bottom_density=1.9, bottom_attenuation=0.8)
    amps = 1e-3 * 10.0 ** (-np.arange(30) * 2.0 / 20.0)                      # 2 dB apart: 11 lie within 20 dB of the strongest
    auto = _fake(BellhopProvider(ocean, 5000.0, 3200.0, range_step=100.0, depth_step=10.0), amps)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rays = auto.eigenrays(SRC, PTS)
    assert auto.arrivals_used == 11 and rays.valid.sum(axis=1).tolist() == [11, 11, 11] and auto.truncation is None
    assert auto.dropped_energy == pytest.approx(np.sum(amps[11:] ** 2) / np.sum(amps ** 2)) and auto.dropped_energy < 0.01
    eight = _fake(BellhopProvider(ocean, 5000.0, 3200.0, range_step=100.0, depth_step=10.0, max_arrivals=8), amps)       # stated: honoured
    assert eight.eigenrays(SRC, PTS).valid.sum(axis=1).tolist() == [8, 8, 8] and eight.dropped_energy == pytest.approx(np.sum(amps[8:] ** 2) / np.sum(amps ** 2))
    capped = _fake(BellhopProvider(ocean, 5000.0, 3200.0, range_step=100.0, depth_step=10.0, max_arrivals_cap=6), amps)
    with pytest.warns(RuntimeWarning, match="cut at the cap of 6 per point"):
        rays = capped.eigenrays(SRC, PTS)
    assert rays.valid.sum(axis=1).tolist() == [6, 6, 6] and capped.truncation["dropped_energy"] == pytest.approx(np.sum(amps[6:] ** 2) / np.sum(amps ** 2))
