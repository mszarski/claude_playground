"""Sound-speed profiles that change with range in the Bellhop provider (`environment.profiles_by_range`,
`OceanEnvironment.ssp_range`): the files written, the checks, and Bellhop's arrivals against closed forms.

Bellhop (GPL-3) is an external oracle, run as a separate process. The tests that need it are skipped when it is not
installed (set GPUSONAR_BELLHOP to bellhop.exe).
"""

import os
import warnings

import numpy as np
import pytest

from gpusonar import depthplan, raytrace
from gpusonar import environment as site
from gpusonar.config import load_scenario_config
from gpusonar.sim import BellhopProvider, OceanEnvironment, find_bellhop, run_bellhop
from gpusonar.sim.bellhop import write_env

needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")

BOTTOM = dict(bottom_speed=1600.0, bottom_density=1.8, bottom_attenuation=0.5)
SCENARIOS = ("ssp_front_static_trial_2row", "ssp_thermocline_static_trial_3row", "ssp_front_testauv_trial_1row")


def _sections(**change):
    env = {"water_depth_m": 80.0, "bottom": {"speed": 1600.0, "density": 1.8, "attenuation": 0.5},
           "profiles_by_range": [{"range_m": 0.0, "depth_m": [0, 20, 80], "speed_m_s": [1500.0, 1490.0, 1491.0]},
                                 {"range_m": 1000.0, "depth_m": [0, 20, 80], "speed_m_s": [1480.0, 1481.0, 1482.5]}]}
    env.update(change)
    return env


# --- the files Bellhop reads -----------------------------------------------------------------

def test_range_independent_env_file_is_unchanged(tmp_path):
    """One profile: the `.env` is, byte for byte, what it was before profiles by range existed, and no `.ssp` is written."""
    env = OceanEnvironment(depth=80.0, ssp_depth=[0.0, 20.0, 80.0], ssp_speed=[1500.0, 1490.0, 1491.0], **BOTTOM)
    write_env(str(tmp_path / "a.env"), env, 6000.0, 12.5, [10.0, 40.0], [500.0, 1500.0])
    assert (tmp_path / "a.env").read_text() == (
        "'gpusonar'\n6000.000000\n1\n'CVW'\n3  0.0  80.0000\n  0.0000  1500.0000  /\n  20.0000  1490.0000  /\n  80.0000  1491.0000  /\n'A' 0.0\n"
        " 80.0000  1600.0000 0.0 1.8000 0.5000 /\n1\n12.500000 /\n2\n10.000000 40.000000 /\n2\n0.500000000 1.500000000 /\n'A'\n0\n-80.000 80.000 /\n"
        "0.0  84.000  1.675000\n")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.env"]


def test_ssp_file_is_bellhops_format(tmp_path):
    """Acoustics Toolbox doc/RangeDepSSPFile.htm and sspMod.f90 (Quad): the number of profiles, their ranges in km, then one
    row per depth of the `.env` profile with one speed per range. The `.env` differs from the single-profile one in the SSP
    option only ('Q'). The end profiles are repeated outside the given ranges, beyond Bellhop's ray box on both sides."""
    one = OceanEnvironment(depth=80.0, ssp_depth=[0.0, 20.0, 80.0], ssp_speed=[1500.0, 1490.0, 1491.0], **BOTTOM)
    two = OceanEnvironment(depth=80.0, ssp_depth=[0.0, 20.0, 80.0], ssp_range=[0.0, 1000.0],
                           ssp_speed_by_range=[[1500.0, 1490.0, 1491.0], [1480.0, 1481.0, 1482.5]], **BOTTOM)
    for name, env in (("a", one), ("b", two)):
        write_env(str(tmp_path / f"{name}.env"), env, 6000.0, 12.5, [10.0, 40.0], [500.0, 1500.0])
    assert (tmp_path / "b.env").read_text() == (tmp_path / "a.env").read_text().replace("'CVW'", "'QVW'")
    rows = [line.split() for line in (tmp_path / "b.ssp").read_text().splitlines()]
    assert rows[0] == ["4"]
    ranges_km = [float(v) for v in rows[1]]
    box_km = float((tmp_path / "b.env").read_text().split()[-1])
    assert ranges_km[1:3] == [0.0, 1.0] and ranges_km[0] < -box_km and ranges_km[3] > box_km
    assert np.array_equal(np.array(rows[2:], dtype=float), [[1500.0, 1500.0, 1480.0, 1480.0], [1490.0, 1490.0, 1481.0, 1481.0], [1491.0, 1491.0, 1482.5, 1482.5]])


def test_speed_at_range_and_local_profile():
    env = OceanEnvironment(depth=80.0, ssp_depth=[0.0, 20.0, 80.0], ssp_range=[0.0, 1000.0, 3000.0],
                           ssp_speed_by_range=[[1500.0, 1490.0, 1491.0], [1480.0, 1481.0, 1482.5], [1480.0, 1481.0, 1482.5]], **BOTTOM)
    assert env.speed_at(10.0) == pytest.approx(1495.0)                      # no range: the local profile
    assert np.array_equal(env.profile()[1], [1500.0, 1490.0, 1491.0])
    assert env.speed_at(10.0, 0.0) == pytest.approx(1495.0)
    assert env.speed_at(10.0, 250.0) == pytest.approx(0.75 * 1495.0 + 0.25 * 1480.5)      # linear in depth, then in range
    assert env.speed_at(50.0, 2000.0) == pytest.approx(1481.75)
    assert env.speed_at(50.0, 9000.0) == pytest.approx(1481.75)             # the last profile holds beyond its range
    # a grid deeper than the water is cut at the bottom, by the interpolation Bellhop uses
    deep = OceanEnvironment(depth=50.0, ssp_depth=[0.0, 20.0, 80.0], ssp_range=[0.0, 1000.0], ssp_speed_by_range=[[1500.0, 1490.0, 1491.0], [1480.0, 1481.0, 1482.5]], **BOTTOM)
    r, z, c = deep.section()
    assert z.tolist() == [0.0, 20.0, 50.0] and c[:, -1] == pytest.approx([1490.5, 1481.75])


# --- checks ------------------------------------------------------------------------------------

@pytest.mark.parametrize("change, words", [
    (dict(ssp_range=[0.0]), "two ranges or more"),
    (dict(ssp_range=[100.0, 1000.0]), "must start at 0"),
    (dict(ssp_range=[0.0, 1000.0, 1000.0], ssp_speed_by_range=[[1500.0, 1490.0, 1491.0]] * 3), "increase"),
    (dict(ssp_speed_by_range=[[1500.0, 1490.0, 1491.0], [1480.0, 1481.0]]), "same depth grid"),
    (dict(ssp_speed_by_range=[[1500.0, 1490.0, 1491.0]]), "same depth grid"),
    (dict(ssp_depth=[5.0, 20.0, 80.0]), "start at 0"),
    (dict(depth=120.0), "must reach the bottom"),
    (dict(ssp_speed=[1499.0, 1490.0, 1491.0]), "local profile"),
    (dict(ssp_speed_by_range=None), "all three"),
    (dict(ssp_depth=None), "all three"),
    (dict(ssp_speed_by_range=[[1500.0, 1490.0, -1.0], [1480.0, 1481.0, 1482.5]]), "positive"),
])
def test_ocean_environment_refuses_inconsistent_sections(change, words):
    kw = dict(depth=80.0, ssp_depth=[0.0, 20.0, 80.0], ssp_range=[0.0, 1000.0], ssp_speed_by_range=[[1500.0, 1490.0, 1491.0], [1480.0, 1481.0, 1482.5]], **BOTTOM)
    kw.update(change)
    with pytest.raises(ValueError, match=words):
        OceanEnvironment(**kw)


def test_relief_deeper_than_the_profiles_is_refused():
    """`along_profile` (reverberation over relief) deepens the ocean: the profiles must still reach the bottom."""
    env = OceanEnvironment(depth=80.0, ssp_depth=[0.0, 20.0, 80.0], ssp_range=[0.0, 1000.0], ssp_speed_by_range=[[1500.0, 1490.0, 1491.0], [1480.0, 1481.0, 1482.5]], **BOTTOM)
    provider = BellhopProvider(env, 6000.0, max_range=1000.0)
    assert provider.along_profile([[0.0, 80.0], [1000.0, 70.0]]).env.ssp_range == [0.0, 1000.0]
    with pytest.raises(ValueError, match="must reach the bottom"):
        provider.along_profile([[0.0, 80.0], [1000.0, 95.0]])


@pytest.mark.parametrize("change, words", [
    (dict(profile={"depth_m": [0, 80], "speed_m_s": [1500.0, 1500.0]}), "not both"),
    (dict(profiles_by_range=[{"range_m": 0.0, "depth_m": [0, 80], "speed_m_s": [1500.0, 1500.0]}]), "two profiles or more"),
    (dict(profiles_by_range=[{"range_m": 50.0, "depth_m": [0, 80], "speed_m_s": [1500.0, 1500.0]}, {"range_m": 900.0, "depth_m": [0, 80], "speed_m_s": [1490.0, 1490.0]}]), "must be at 0"),
    (dict(profiles_by_range=[{"range_m": 0.0, "depth_m": [0, 80], "speed_m_s": [1500.0, 1500.0]}, {"range_m": 900.0, "depth_m": [0, 80], "speed_m_s": [1490.0, 1490.0]},
                             {"range_m": 600.0, "depth_m": [0, 80], "speed_m_s": [1490.0, 1490.0]}]), "ranges must increase"),
    (dict(profiles_by_range=[{"range_m": 0.0, "depth_m": [0, 20, 80], "speed_m_s": [1500.0, 1490.0, 1491.0]}, {"range_m": 900.0, "depth_m": [0, 30, 80], "speed_m_s": [1490.0, 1490.0, 1490.0]}]), "another depth grid"),
    (dict(profiles_by_range=[{"range_m": 0.0, "depth_m": [0, 60], "speed_m_s": [1500.0, 1500.0]}, {"range_m": 900.0, "depth_m": [0, 60], "speed_m_s": [1490.0, 1490.0]}]), "must reach the bottom"),
    (dict(profiles_by_range=[{"range_m": 0.0, "depth_m": [5, 80], "speed_m_s": [1500.0, 1500.0]}, {"range_m": 900.0, "depth_m": [5, 80], "speed_m_s": [1490.0, 1490.0]}]), "start at 0"),
    (dict(profiles_by_range=[{"range_m": 0.0, "depth_m": [0, 80], "speed_m_s": [1500.0]}, {"range_m": 900.0, "depth_m": [0, 80], "speed_m_s": [1490.0, 1490.0]}]), "2 depths and 1 speeds"),
    (dict(profiles_by_range=[{"range_m": 0.0, "depth_m": [0, 80], "speed_m_s": [1500.0, 150.0]}, {"range_m": 900.0, "depth_m": [0, 80], "speed_m_s": [1490.0, 1490.0]}]), "not a speed of sound"),
    (dict(profiles_by_range=[{"range_m": 0.0, "depth_m": [0, 80], "speed_m_s": [1500.0, 1500.0], "temperature_c": 4.0}, {"range_m": 900.0, "depth_m": [0, 80], "speed_m_s": [1490.0, 1490.0]}]), "unknown keys"),
    (dict(water_depth_m=None), "need water_depth_m"),
    (dict(sound_speed_m_s=1470.0), "outside its own profile"),            # the record's speed is a speed of the LOCAL profile
])
def test_environment_refuses_inconsistent_profiles_by_range(change, words):
    with pytest.raises(site.EnvironmentError, match=words):
        site.resolve(_sections(**change), strict=True)


def test_environment_gives_the_local_profile_and_the_sections():
    env = site.resolve(_sections(reference_depth_m=10.0), strict=True)
    assert env["profile"] == {"depth": [0.0, 20.0, 80.0], "speed": [1500.0, 1490.0, 1491.0]}        # the local one
    assert env["profile_by_range"] == {"range": [0.0, 1000.0], "speed": [[1500.0, 1490.0, 1491.0], [1480.0, 1481.0, 1482.5]]}
    assert env["sound_speed"] == pytest.approx(1495.0)               # not given: the LOCAL profile at the reference depth
    assert site.resolve(_sections(sound_speed_m_s=1492.0), strict=True)["sound_speed"] == 1492.0
    cfg = site.apply_to_scenario({"name": "t", "fidelity": "eigenray", "propagation": {"max_range": 1500.0, "frequency": 6000.0}}, env)
    ocean = cfg["propagation"]["environment"]
    assert ocean["ssp_range"] == [0.0, 1000.0] and ocean["ssp_speed"] == [1500.0, 1490.0, 1491.0] and ocean["depth"] == 80.0
    made = OceanEnvironment(**ocean)
    assert made.speed_at(10.0, 1000.0) == pytest.approx(1480.5) and made.speed_at(10.0) == pytest.approx(1495.0)
    # a child environment's water replaces its parent's, whichever way each was written
    child = site.load({"base": "coastal_12c", "water_depth_m": 80.0, "profiles_by_range": _sections()["profiles_by_range"]})
    assert "profile" not in child and "sound_speed_m_s" not in child


def test_other_tiers_say_that_the_change_with_range_is_not_simulated():
    env = site.resolve(_sections(), strict=True)
    with pytest.warns(UserWarning, match="NOT simulated"):
        cfg = site.apply_to_scenario({"name": "t", "fidelity": "image_source", "propagation": {"max_order": 1}}, env)
    assert cfg["propagation"]["bottom_z"] == 80.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        site.apply_to_scenario({"name": "t", "fidelity": "eigenray", "propagation": {"max_range": 1500.0, "frequency": 6000.0}}, env)


def test_depth_planner_says_it_plans_with_the_local_profile():
    env, ranged = depthplan._local_water(_sections(name="x"))
    assert ranged and env["profile"] == {"depth_m": [0.0, 20.0, 80.0], "speed_m_s": [1500.0, 1490.0, 1491.0]}
    assert depthplan._local_water({"water_depth_m": 80.0}) == ({"water_depth_m": 80.0}, False)


@pytest.mark.parametrize("name", SCENARIOS)
def test_shipped_scenarios_carry_the_local_speed_at_the_sonar(name):
    """The record's one sound speed is what the vehicle would measure: the local profile at the platform's depth."""
    cfg = load_scenario_config(name)
    env, ocean = cfg["environment"], cfg["propagation"]["environment"]
    z = float(cfg["platform"]["position"][2])
    assert env["sound_speed"] == pytest.approx(np.interp(z, env["profile"]["depth"], env["profile"]["speed"]), abs=0.01)
    assert ocean["ssp_range"][0] == 0.0 and ocean["ssp_speed"] == ocean["ssp_speed_by_range"][0]
    assert any(row != ocean["ssp_speed"] for row in ocean["ssp_speed_by_range"])          # the water does change
    OceanEnvironment(**ocean)


# --- Bellhop's arrivals ------------------------------------------------------------------------

@needs_bellhop
def test_identical_profiles_at_two_ranges_reproduce_the_single_profile():
    """The same thermocline given once, and given at two ranges: Bellhop's six strongest arrivals at every receiver agree
    in bounce counts, to 2 microseconds and to 1 % in amplitude ('C' and 'Q' are the same linear interpolation in depth;
    'Q' adds steps at the profile ranges)."""
    z, c = [0.0, 15.0, 35.0, 100.0], [1500.0, 1500.0, 1485.0, 1486.1]
    bottom = dict(bottom_speed=1650.0, bottom_density=1.9, bottom_attenuation=0.8)
    one = OceanEnvironment(depth=100.0, ssp_depth=z, ssp_speed=c, **bottom)
    two = OceanEnvironment(depth=100.0, ssp_depth=z, ssp_range=[0.0, 1500.0], ssp_speed_by_range=[c, c], **bottom)
    depths, ranges = [20.0, 60.0, 90.0], [300.0, 1000.0, 2600.0]
    a, b = (run_bellhop(env, 6000.0, 30.0, depths, ranges) for env in (one, two))
    for i in range(len(depths)):
        for j in range(len(ranges)):
            ka, kb = np.argsort(-np.abs(a.amplitude[i][j]))[:6], np.argsort(-np.abs(b.amplitude[i][j]))[:6]
            assert np.array_equal(a.n_surface[i][j][ka], b.n_surface[i][j][kb]) and np.array_equal(a.n_bottom[i][j][ka], b.n_bottom[i][j][kb])
            assert np.allclose(a.delay[i][j][ka], b.delay[i][j][kb], rtol=0.0, atol=2e-6)
            assert np.allclose(a.amplitude[i][j][ka], b.amplitude[i][j][kb], rtol=1e-2, atol=0.0)
    # and through the provider: the same eigenrays for a field point between grid nodes
    kw = dict(frequency=6000.0, max_range=2700.0, range_step=50.0, depth_step=5.0, max_arrivals=4)
    points = np.array([[2583.0, 300.0, 61.0], [700.0, -90.0, 22.0]])
    ea, eb = (BellhopProvider(env, **kw).eigenrays([0.0, 0.0, 30.0], points) for env in (one, two))
    assert np.array_equal(ea.valid, eb.valid) and np.allclose(ea.delay, eb.delay, rtol=0.0, atol=2e-6)
    assert np.allclose(ea.amplitude, eb.amplitude, rtol=1e-2, atol=0.0) and ea.sound_speed == eb.sound_speed


@needs_bellhop
def test_constant_gradient_in_range_against_closed_form_and_own_ray_tracer():
    """Two isovelocity profiles, 1480 m/s at the source and 1520 m/s at 2 km: the speed is linear in RANGE (g = 0.02 /s) and
    the same at every depth. In a medium with a constant gradient every ray is an arc of a circle centred where the speed
    would be zero, here the vertical line r = -c0 / g, and the travel time between two points a distance d apart with
    speeds c_a and c_b is acosh(1 + g^2 d^2 / (2 c_a c_b)) / g. Bellhop's direct arrival has that delay to 20 microseconds
    (0.3 microseconds along the gradient) and the launch angle of that circle to 0.05 degrees. The same delays come from
    the project's own layered ray tracer for the problem turned by 90 degrees (the gradient in depth). The straight line
    at the local speed, which is all a chain without this knowledge has, is 0.7 to 16 ms later."""
    c0, c1, r_end = 1480.0, 1520.0, 2000.0
    g = (c1 - c0) / r_end
    env = OceanEnvironment(depth=1000.0, ssp_depth=[0.0, 1000.0], ssp_range=[0.0, r_end], ssp_speed_by_range=[[c0, c0], [c1, c1]], **BOTTOM)
    z_s, depths, ranges = 500.0, [300.0, 500.0, 650.0], [400.0, 1000.0, 1900.0]
    arr = run_bellhop(env, 5000.0, z_s, depths, ranges, angles=(-60.0, 60.0))
    turned = raytrace.Profile([0.0, 3000.0], [c0 - g * 100.0, c0 + g * 2900.0])           # c0 at 100 m depth, the same gradient downwards
    for i, z in enumerate(depths):
        for j, r in enumerate(ranges):
            direct = np.flatnonzero((arr.n_surface[i][j] == 0) & (arr.n_bottom[i][j] == 0))
            assert len(direct) == 1
            k = direct[0]
            d = np.hypot(r, z - z_s)
            t = np.arccosh(1.0 + g * g * d * d / (2.0 * c0 * (c0 + g * r))) / g
            assert arr.delay[i][j][k] == pytest.approx(t, abs=2e-5 if z != z_s else 5e-7)
            assert d / c0 - t > 7e-4                                               # what the local speed alone would say
            if z == z_s:
                assert arr.launch_angle[i][j][k] == pytest.approx(0.0, abs=1e-4)   # along the gradient: a straight ray
                assert t == pytest.approx(np.log((c0 + g * r) / c0) / g, abs=1e-12)
            else:
                z_centre = 0.5 * (z_s + z) - (r * r + 2.0 * r * c0 / g) / (2.0 * (z_s - z))      # the circle through both points
                assert np.degrees(arr.launch_angle[i][j][k]) == pytest.approx(np.degrees(np.arctan((c0 / g) / (z_centre - z_s))), abs=0.05)
                own = [e for e in raytrace.eigenrays(turned, 100.0, 100.0 + r, abs(z - z_s), max_angle_deg=89.0, n_fan=7201, max_bounces=0) if e.turns == 0]
                assert len(own) == 1 and own[0].time == pytest.approx(t, abs=1e-7)
                assert arr.delay[i][j][k] == pytest.approx(own[0].time, abs=2e-5)


@needs_bellhop
def test_travel_time_through_two_isovelocity_halves():
    """A front: 1500 m/s out to 900 m, 1470 m/s from 1100 m on, linear in range between. Along a horizontal ray (source and
    receiver at one depth: normal incidence on the front, no bending) the delay is 900 / c1 + 200 ln(c1 / c2) / (c1 - c2) +
    (r - 1100) / c2. The provider's eigenray to a point off the grid nodes has it too, with the speed at the point's own
    range in the correction from the node (the local speed there would be 2 % out). The water is 100 m deep so that
    Bellhop's ray step (a tenth of the depth) is 10 m: it interpolates the delay linearly between steps, which inside the
    front is out by up to step^2 |dc/dr| / (8 c^2) (27 microseconds were seen with 60 m steps)."""
    c1, c2 = 1500.0, 1470.0
    env = OceanEnvironment(depth=100.0, ssp_depth=[0.0, 100.0], ssp_range=[0.0, 900.0, 1100.0], ssp_speed_by_range=[[c1, c1], [c1, c1], [c2, c2]], **BOTTOM)
    through = 200.0 * np.log(c1 / c2) / (c1 - c2)
    arr = run_bellhop(env, 5000.0, 50.0, [50.0], [600.0, 1000.0, 2000.0, 3000.0], angles=(-45.0, 45.0))
    expected = [600.0 / c1, 900.0 / c1 + 100.0 * np.log(c1 / (0.5 * (c1 + c2))) / (0.5 * (c1 - c2)), 900.0 / c1 + through + 900.0 / c2, 900.0 / c1 + through + 1900.0 / c2]
    for j, t in enumerate(expected):
        k = np.flatnonzero((arr.n_surface[0][j] == 0) & (arr.n_bottom[0][j] == 0))
        assert len(k) == 1 and arr.delay[0][j][k[0]] == pytest.approx(t, abs=2e-6)
    assert 3000.0 / c1 - expected[3] < -0.025                                        # the local speed alone: 27 ms early
    provider = BellhopProvider(env, 5000.0, max_range=3000.0, min_range=2000.0, range_step=100.0, depth_step=5.0, max_arrivals=3)
    rays = provider.eigenrays([0.0, 0.0, 50.0], [[2040.0, 0.0, 46.0]])             # its grid node: 2000 m, 47.5 m
    k = np.flatnonzero(rays.valid[0] & (rays.n_surface[0] == 0) & (rays.n_bottom[0] == 0))
    assert len(k) == 1 and rays.sound_speed == c1
    assert rays.delay[0][k[0]] == pytest.approx(900.0 / c1 + through + 940.0 / c2, abs=5e-5)
    assert abs(rays.delay[0][k[0]] - (900.0 / c1 + through + 900.0 / c2 + 40.0 / c1)) > 4e-4


@needs_bellhop
def test_shipped_scenario_record_carries_the_local_speed():
    """One ping of the test-AUV scenario: the record's sound speed is the local profile's at the vehicle, the water
    depth the site's, and the echo is there."""
    from gpusonar.sim import run_scenario
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    res = run_scenario(load_scenario_config("ssp_front_testauv_trial_1row", {"pings": {"count": 1}}, expand=False))
    rec = res.records[0]
    assert rec.environment.sound_speed == pytest.approx(1497.5) and rec.environment.water_depth == pytest.approx(30.0)
    assert np.abs(rec.samples.array).max() > 0.0
