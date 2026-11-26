"""The live chain's depth and multipath models over an elastic bottom (rock; the live-chain port of A8).

A8 (e6020b5) made a `rock` bottom reflect as an elastic solid (Jackson and Richardson 2007, (9.34)-(9.40)) in the
simulator. Here the models the live chain and its planning take a bottom reflection from do the same, through the
one coefficient (Python: `propagation.rayleigh_reflection`; C++: `sonar::elastic_reflection` in libs/core
reflection.hpp, which libs/sim and libs/live both call):

    gpusonar.lloyd           Layer.halfspace / one_way (the depth planner's isovelocity image series)
    gpusonar.coverage        the C++ coverage picture's bottom (resolution of `chain.coverage.bottom`)
    gpusonar.multipath_depth the predicted pattern (NumPy reference and the C++ stage both engines run)
    gpusonar.raytrace        eigenrays (Python reference and the C++ ray code)
    gpusonar.depthplan       the isovelocity branch (through lloyd) and the Bellhop branch (no longer pinned fluid)

Each over rock: the row's per-bounce losses of A8 (1.56 / 2.90 / 5.45 / 14.57 / 8.78 / 6.25 / 4.54 / 2.80 dB at 5 /
10 / 20 / 30 / 40 / 45 / 60 / 90 degrees; tests/python/test_elastic_bottom.py) where the model shows a single bounce;
zero shear (`shear_speed: 0`) is the fluid bit for bit; the row's unconfirmed shear is warned about once.
Engine equality through the chain over rock is in tests/python/test_live_cpp.py.
"""
import math
import warnings

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import coverage, depthplan, lloyd, raytrace
from gpusonar import multipath_depth as mp
from gpusonar.sim import bottom as bottom_model
from gpusonar.sim.propagation import rayleigh_reflection

core = gs.core
ANGLES = [5.0, 10.0, 20.0, 30.0, 40.0, 45.0, 60.0, 90.0]
ELASTIC_DB = [1.56, 2.90, 5.45, 14.57, 8.78, 6.25, 4.54, 2.80]       # A8's table for the row (test_elastic_bottom.py)
FLUID_DB = [0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.06, 2.80]
C = 1500.0


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


ROCK = quiet(bottom_model.halfspace, "rock", C)                       # {speed, density, attenuation, shear_speed, shear_attenuation}
ROCK_FLUID = quiet(bottom_model.fluid_halfspace, "rock", C)


def loss_db(v):
    return -20.0 * np.log10(np.abs(v))


def fluid_rows():
    out = []
    for name in sorted(bottom_model.sediment_table()):
        try:
            par = quiet(bottom_model.sediment_reflection, name)
        except (KeyError, ValueError):
            continue
        if "nu_t" not in par:
            out.append(name)
    return out


# --- the coefficient: one in Python, one in C++ ---------------------------------------------------------------------

def test_live_coefficient_is_the_simulators_on_rock():
    """libs/live's rayleigh_reflection with a shear speed is the simulator's (the same C++ function, bit for bit) and
    propagation.rayleigh_reflection's to 1e-13; the row's per-bounce losses; shear 0 is the five-argument fluid call."""
    g = np.radians(np.linspace(0.05, 90.0, 721))
    live = np.array([core.live_rayleigh_reflection(x, C, ROCK["speed"], ROCK["density"], ROCK["attenuation"], ROCK["shear_speed"],
                                                   ROCK["shear_attenuation"]) for x in g])
    sim = core.sim_rayleigh_reflection(g, C, **ROCK)
    assert np.array_equal(live, sim)
    assert np.max(np.abs(live - rayleigh_reflection(g, C, **ROCK))) < 1e-13
    at = np.radians(ANGLES)
    np.testing.assert_allclose(loss_db([core.live_rayleigh_reflection(x, C, ROCK["speed"], ROCK["density"], ROCK["attenuation"],
                                                                      ROCK["shear_speed"], ROCK["shear_attenuation"]) for x in at]), ELASTIC_DB, atol=0.005)
    fluid = [core.live_rayleigh_reflection(x, C, ROCK["speed"], ROCK["density"], ROCK["attenuation"]) for x in g]
    zero = [core.live_rayleigh_reflection(x, C, ROCK["speed"], ROCK["density"], ROCK["attenuation"], 0.0, ROCK["shear_attenuation"]) for x in g]
    assert fluid == zero
    np.testing.assert_allclose(loss_db(fluid)[[int(round((a - 0.05) / (89.95 / 720))) for a in ANGLES]], FLUID_DB, atol=0.02)


# --- lloyd: the planner's image series --------------------------------------------------------------------------------

def single_bounce(layer, own, target, grazing_deg):
    """V of the one bottom image (the surface's coefficient 0 removes the surface images): one frequency, max_order 1,
    from H = 1 + (r_d / r_b) V exp(-i k (r_b - r_d))."""
    lift = 2.0 * layer.depth - target - own
    r = lift / math.tan(math.radians(grazing_deg))
    f = 8000.0
    h, _ = lloyd.one_way([f], own, target, r, lloyd.Surface(wind_speed=0.0, rms_height=0.0, coefficient=0.0), layer)
    r_d, r_b = math.hypot(r, own - target), math.hypot(r, lift)
    k = 2.0 * math.pi * f / layer.sound_speed
    return complex((h.ravel()[0] - 1.0) * (r_b / r_d) * np.exp(1j * k * (r_b - r_d)))


def test_lloyd_layer_over_rock_is_elastic_and_zero_shear_is_the_fluid():
    named = lloyd.Layer(depth=100.0, bottom={"sediment": "rock"}, sound_speed=C, max_order=1)
    assert quiet(named.halfspace) == ROCK
    got = [loss_db(single_bounce(named, 30.0, 50.0, a)) for a in ANGLES[:-1]]
    np.testing.assert_allclose(got, ELASTIC_DB[:-1], atol=0.005)
    fluid = lloyd.Layer(depth=100.0, bottom={"sediment": "rock", "shear_speed": 0.0}, sound_speed=C, max_order=1)
    np.testing.assert_allclose([loss_db(single_bounce(fluid, 30.0, 50.0, a)) for a in ANGLES[:-1]], FLUID_DB[:-1], atol=0.005)
    f = lloyd.band(7000.0, 9000.0, 33)
    zt, rr = np.array([20.0, 60.0])[:, None], np.array([400.0, 2000.0])[None, :]
    a = lloyd.one_way(f, 30.0, zt, rr, layer=lloyd.Layer(depth=100.0, bottom={"sediment": "rock", "shear_speed": 0.0}, sound_speed=C, max_order=4))
    b = lloyd.one_way(f, 30.0, zt, rr, layer=lloyd.Layer(depth=100.0, bottom=dict(ROCK_FLUID), sound_speed=C, max_order=4))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # an explicit elastic mapping is taken as it is; a negative shear speed is refused
    e = lloyd.one_way(f, 30.0, zt, rr, layer=lloyd.Layer(depth=100.0, bottom=dict(ROCK), sound_speed=C, max_order=4))
    n = lloyd.one_way(f, 30.0, zt, rr, layer=lloyd.Layer(depth=100.0, bottom={"sediment": "rock"}, sound_speed=C, max_order=4))
    assert np.array_equal(e[0], n[0]) and not np.allclose(e[0], a[0])
    with pytest.raises(ValueError, match="shear"):
        lloyd.Layer(depth=100.0, bottom=dict(ROCK, shear_speed=-1.0)).halfspace()


def test_fluid_rows_resolve_as_before():
    """Every row without shear: Layer.halfspace is fluid_halfspace exactly, and neither the coverage nor the multipath
    options carry a shear key (their option sets are those of before the port)."""
    rows = fluid_rows()
    assert len(rows) >= 10 and "rock" not in rows
    for name in rows:
        half = quiet(lloyd.Layer(depth=50.0, bottom={"sediment": name}, sound_speed=C).halfspace)
        assert half == quiet(bottom_model.fluid_halfspace, name, C), name
        o = quiet(coverage.cpp_options, {"bottom": {"sediment": name}})
        assert not any("shear" in k for k in o), name
        o = quiet(mp.cpp_options, {"bottom": {"sediment": name}, "water_depth": 50.0})
        assert not any("shear" in k for k in o) and o["bottom_speed"] == half["speed"], name


# --- the multipath pattern --------------------------------------------------------------------------------------------

def test_multipath_pattern_over_rock_is_elastic_in_both_and_equal():
    """The bottom-reflected one-leg arrival of the pattern over rock is the row's elastic coefficient (per bounce, the
    table), the NumPy reference and the C++ stage agree (1e-12, as for sand in test_multipath_depth.py), and the
    pattern with `shear_speed: 0` is the fluid one bit for bit."""
    cfg = quiet(mp.options, {"bottom": {"sediment": "rock"}, "bottom_order": 1, "surface": {"coefficient": 0.0}})
    assert cfg["bottom"] == ROCK
    own, zt, depth = 30.0, 50.0, 100.0
    lift = 2.0 * depth - zt - own
    for a, want in zip(ANGLES[:-1], ELASTIC_DB[:-1]):
        r_h = lift / math.tan(math.radians(a))
        g = mp.Geometry(range_m=r_h, array_depth=own, projector_depth=own, sound_speed=C, f_centre=8000.0, bandwidth=2000.0, water_depth=depth)
        t0 = 2.0 * math.hypot(r_h, zt - own) / C
        _, amp = mp.pattern(zt, t0, g, cfg)
        _, a2 = core.live_multipath_pattern(zt, t0, mp.geometry_dict(g), quiet(mp.cpp_options, {"bottom": {"sediment": "rock"}, "bottom_order": 1,
                                                                                                "surface": {"coefficient": 0.0}}))
        a2 = np.asarray(a2)
        assert np.allclose(a2, amp[:, 0], atol=1e-12)
        spread = math.hypot(r_h, zt - own) / math.hypot(r_h, lift)
        assert loss_db(amp[6, 0] / spread) == pytest.approx(want, abs=0.005)         # (bottom image out, direct back)
        assert loss_db(a2[6] / spread) == pytest.approx(want, abs=0.005)
    g = mp.Geometry(range_m=2000.0, array_depth=own, projector_depth=31.5, sound_speed=C, f_centre=8000.0, bandwidth=2000.0, water_depth=depth)
    zero = quiet(mp.options, {"bottom": {"sediment": "rock", "shear_speed": 0.0}, "bottom_order": 3})
    plain = mp.options({"bottom": dict(ROCK_FLUID), "bottom_order": 3})
    d0, a0 = mp.pattern(zt, np.array([2.67, 2.68]), g, zero)
    d1, a1 = mp.pattern(zt, np.array([2.67, 2.68]), g, plain)
    assert np.array_equal(d0, d1) and np.array_equal(a0, a1)
    z0 = core.live_multipath_pattern(zt, 2.67, mp.geometry_dict(g), quiet(mp.cpp_options, {"bottom": {"sediment": "rock", "shear_speed": 0.0}, "bottom_order": 3}))
    z1 = core.live_multipath_pattern(zt, 2.67, mp.geometry_dict(g), mp.cpp_options({"bottom": dict(ROCK_FLUID), "bottom_order": 3}))
    assert z0[0] == z1[0] and z0[1] == z1[1]
    with pytest.raises(ValueError, match="shear"):
        mp.cpp_options({"bottom": dict(ROCK, shear_attenuation=-0.5)})


# --- the ray code -----------------------------------------------------------------------------------------------------

def test_eigenrays_over_rock_are_elastic_in_both_and_equal():
    """Isovelocity water over rock: an eigenray with one bottom bounce carries the elastic coefficient at its grazing
    angle (the launch angle here), in the Python reference and the C++ ray code, which agree as for sand
    (test_raytrace_cpp.py: amplitudes rel 1e-6)."""
    prof = raytrace.Profile(np.array([0.0, 100.0]), np.array([C, C]), 100.0)
    ref = raytrace.eigenrays(prof, 30.0, 50.0, 400.0, n_fan=801, max_bounces=2, bottom=dict(ROCK))
    got = core.ray_eigenrays([0.0, 100.0], [C, C], 100.0, 30.0, 50.0, 400.0, n_fan=801, max_bounces=2, bottom=dict(ROCK))
    flu = raytrace.eigenrays(prof, 30.0, 50.0, 400.0, n_fan=801, max_bounces=2, bottom=dict(ROCK_FLUID))
    assert len(ref) == len(got) == len(flu) >= 4
    for e, f, q in zip(ref, got, flu):
        assert (e.n_surface, e.n_bottom) == (f["n_surface"], f["n_bottom"]) == (q.n_surface, q.n_bottom)
        assert f["amplitude"] == pytest.approx(e.amplitude, rel=1e-6)
        if e.n_bottom == 1 and e.n_surface == 0:
            g = abs(e.launch)
            want = rayleigh_reflection(g, C, **ROCK) / rayleigh_reflection(g, C, **ROCK_FLUID)
            assert e.amplitude / q.amplitude == pytest.approx(complex(want), rel=1e-9)
            assert loss_db(rayleigh_reflection(g, C, **ROCK)) > 3.0           # at about 17 degrees: 4.5 dB
    zero = raytrace.eigenrays(prof, 30.0, 50.0, 400.0, n_fan=801, max_bounces=2, bottom=dict(ROCK, shear_speed=0.0))
    assert [z.amplitude for z in zero] == [q.amplitude for q in flu]
    z2 = core.ray_eigenrays([0.0, 100.0], [C, C], 100.0, 30.0, 50.0, 400.0, n_fan=801, max_bounces=2, bottom=dict(ROCK, shear_speed=0.0))
    f2 = core.ray_eigenrays([0.0, 100.0], [C, C], 100.0, 30.0, 50.0, 400.0, n_fan=801, max_bounces=2, bottom=dict(ROCK_FLUID))
    assert [z["amplitude"] for z in z2] == [f["amplitude"] for f in f2]


# --- the coverage picture ---------------------------------------------------------------------------------------------

def test_coverage_over_rock_is_the_elastic_image_series():
    """100 m of isovelocity water over rock: the coverage picture's one-way sum of powers is the image series with the
    elastic coefficient (as test_coverage.py holds it for sand, 0.3 dB), not with the fluid one; `shear_speed: 0` gives
    the fluid row's options exactly."""
    sonar = {"f_centre": 8000.0, "bandwidth": 2000.0, "pulse": 0.1}
    w = coverage.water([0.0, 100.0], [C, C])
    cfg = {"bottom": {"sediment": "rock"}, "rays": {"max_bounces": 2, "max_arrivals": 12, "fan": 1440}}
    o = quiet(coverage.cpp_options, cfg)
    assert o["bottom_shear_speed"] == ROCK["shear_speed"] and o["bottom_shear_attenuation"] == ROCK["shear_attenuation"]
    ranges, zt, own = np.linspace(200.0, 3000.0, 15), np.array([30.0, 70.0]), 50.0
    got = quiet(coverage.field, w, 100.0, own, zt, ranges, cfg, sonar)["incoherent_db"]
    fluid = quiet(coverage.field, w, 100.0, own, zt, ranges, {**cfg, "bottom": {"sediment": "rock", "shear_speed": 0.0}}, sonar)["incoherent_db"]

    def series(half):
        parts = []
        for img, _ns, nb in lloyd._images(zt[:, None], 100.0, 2):
            r_d, r_j = np.hypot(ranges[None, :], own - zt[:, None]), np.hypot(ranges[None, :], own - img)
            parts.append((r_d / r_j) ** 2 * np.abs(rayleigh_reflection(np.arcsin(np.abs(own - img) / r_j), C, **half)) ** (2 * nb))
        return 10.0 * np.log10(sum(parts))
    assert np.max(np.abs(got - series(ROCK))) < 0.3
    assert np.max(np.abs(fluid - series(ROCK_FLUID))) < 0.3
    assert np.max(fluid - got) > 1.0                                  # the elastic bottom returns less
    zero = quiet(coverage.cpp_options, {"bottom": {"sediment": "rock", "shear_speed": 0.0}})
    plain = coverage.cpp_options({"bottom": dict(ROCK_FLUID)})
    assert zero == plain


# --- the depth planner ------------------------------------------------------------------------------------------------

def test_depthplan_isovelocity_branch_over_rock_is_lloyds_elastic_series():
    env = {"name": "rock_100m", "water_depth_m": 100.0, "bottom": "rock", "sound_speed_m_s": C, "wind_speed_m_s": 5.0}
    got = quiet(depthplan.factor, env, 7000.0, 9000.0, [30.0], [50.0], [500.0, 2000.0])
    surf = depthplan.surface_of(env)
    f = lloyd.band(7000.0, 9000.0, 513)
    layer = lloyd.Layer(depth=100.0, bottom=dict(ROCK), sound_speed=C, max_grazing_deg=30.0)
    want = lloyd.summarise(*lloyd.one_way(f, 30.0, np.array([50.0])[:, None], np.array([500.0, 2000.0])[None, :], surf, layer))
    assert np.array_equal(got["peak_db"][0], want["peak_db"])
    fluid_env = dict(env, bottom=dict(ROCK_FLUID))
    fluid = quiet(depthplan.factor, fluid_env, 7000.0, 9000.0, [30.0], [50.0], [500.0, 2000.0])
    assert not np.allclose(fluid["energy_db"], got["energy_db"])


class _Stop(Exception):
    pass


def test_depthplan_bellhop_branch_takes_rock_as_elastic(monkeypatch):
    """The Bellhop branch is no longer pinned to a fluid: a named rock goes into Bellhop's shear column, a mapping with
    shear_speed too; a fluid mapping writes no shear (as before)."""
    from gpusonar.sim import bellhop
    seen = []

    def fake(ocean, *a, **k):
        seen.append(ocean)
        raise _Stop

    monkeypatch.setattr(bellhop, "find_bellhop", lambda: "bellhop.exe")
    monkeypatch.setattr(bellhop, "run_bellhop", fake)
    prof = {"depth_m": [0.0, 100.0], "speed_m_s": [1500.0, 1495.0]}
    for bottom in ("rock", dict(ROCK), dict(ROCK_FLUID), "medium_sand"):
        env = {"name": "b", "water_depth_m": 100.0, "bottom": bottom, "profile": prof, "wind_speed_m_s": 5.0}
        with pytest.raises(_Stop):
            quiet(depthplan.factor, env, 7000.0, 9000.0, [30.0], [50.0], [500.0])
    rock, rock_dict, fluid, sand = seen
    assert rock.bottom_shear_speed > 0.0 and rock_dict.bottom_shear_speed == ROCK["shear_speed"]
    assert rock_dict.bottom_shear_attenuation == ROCK["shear_attenuation"]
    assert not (fluid.bottom_shear_speed or 0.0) and not (sand.bottom_shear_speed or 0.0)


# --- the warning ------------------------------------------------------------------------------------------------------

def test_unconfirmed_shear_is_warned_once():
    """The row's shear numbers are unconfirmed: resolving `rock` in the live models says so once, however often they
    resolve it (a chain resolves its options once; the planner's image series once per call)."""
    lloyd.reset_shear_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            lloyd.Layer(depth=100.0, bottom={"sediment": "rock"}, sound_speed=C).halfspace()
            coverage.cpp_options({"bottom": {"sediment": "rock"}})
            mp.cpp_options({"bottom": {"sediment": "rock"}, "water_depth": 100.0})
        lloyd.Layer(depth=100.0, bottom={"sediment": "rock", "shear_speed": 0.0}, sound_speed=C).halfspace()
    shear = [w for w in caught if "shear" in str(w.message)]
    assert len(shear) == 1 and "rock" in str(shear[0].message) and "could not be confirmed" in str(shear[0].message)
    lloyd.reset_shear_warnings()
    with warnings.catch_warnings(record=True) as caught:                 # the fluid rock does not use the shear: no word on it
        warnings.simplefilter("always")
        lloyd.Layer(depth=100.0, bottom={"sediment": "rock", "shear_speed": 0.0}, sound_speed=C).halfspace()
        lloyd.Layer(depth=100.0, bottom={"sediment": "medium_sand"}, sound_speed=C).halfspace()
    assert not [w for w in caught if "shear" in str(w.message)]
