"""A site's boundary scattering by ONE rule (gpusonar.environment.boundary_scattering), shared by the simulator
(`reverberation: site`), the link budget and depth planner (reverb_budget.site_term, depthplan.plan) and the coverage
picture (coverage.site_reverberation, of_build): what it derives from a bottom and a wind, what it refuses to derive and
says so, that a stated value wins, that every shipped environment still loads and no shipped scenario's records take
anything from the new fields, and that the sites the builds plan in agree with the scenes that simulate the same water."""

import warnings

import numpy as np
import pytest

from gpusonar import coverage, depthplan, environment
from gpusonar import reverb_budget as rb
from gpusonar.config import list_configs, load_scenario_config
from gpusonar.sim.simulator import run_scenario

SAND = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}


def rule(env, f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return environment.boundary_scattering(env, f)


def test_the_rule_derives_from_the_bottom_and_the_wind_and_never_defaults_a_strength():
    # a complete fluid-sediment row inside 10-100 kHz: the fluid-sediment model of that row
    got = rule({"water_depth_m": 40.0, "bottom": "medium_sand", "wind_speed_m_s": 5.0}, 30.0e3)
    assert got["items"][0] == {"model": "fluid_sediment", "sediment": "medium_sand", "boundary": "bottom"}
    assert got["items"][1] == {"model": "chapman_harris", "wind_speed": 5.0, "boundary": "surface"}
    assert got["notes"] == {"bottom": None, "surface": None}
    # the same row at 5 kHz: outside the model's range, so the bottom is NOT included, and the reason says so
    got = rule({"water_depth_m": 40.0, "bottom": "medium_sand", "wind_speed_m_s": 5.0}, 5.0e3)
    assert [i["boundary"] for i in got["items"]] == ["surface"] and "10-100 kHz" in got["notes"]["bottom"]
    # rock: the row's Lambert parameter, at any frequency
    got = rule({"water_depth_m": 40.0, "bottom": "rock"}, 5.0e3)
    assert got["items"] == [{"sediment": "rock", "boundary": "bottom"}] and "no wind" in got["notes"]["surface"]
    # a bottom as numbers: nothing follows from them
    got = rule({"water_depth_m": 100.0, "bottom": SAND, "sea_state": 3}, 5.0e3)
    assert "as numbers" in got["notes"]["bottom"] and got["items"][0]["wind_speed"] == pytest.approx(6.9, abs=0.05)
    # nothing stated: nothing included
    got = rule({}, 5.0e3)
    assert got["items"] == [] and got["notes"]["bottom"].startswith("not included") and got["notes"]["surface"].startswith("not included")
    # stated wins, false is none, the wind is the site's
    got = rule({"water_depth_m": 100.0, "bottom": SAND, "wind_speed_m_s": 7.0, "bottom_scattering": {"model": "lambert", "mu_db": -30.0},
                "surface_scattering": {"model": "bubble_layer_surface"}}, 5.0e3)
    assert got["items"] == [{"model": "lambert", "mu_db": -30.0, "boundary": "bottom"}, {"model": "bubble_layer_surface", "wind_speed": 7.0, "boundary": "surface"}]
    got = rule({"water_depth_m": 100.0, "bottom": "medium_sand", "wind_speed_m_s": 7.0, "bottom_scattering": False, "surface_scattering": False}, 30.0e3)
    assert got["items"] == [] and "false" in got["notes"]["bottom"] and "false" in got["notes"]["surface"]
    # Chapman-Harris outside its fitted band: included, with a warning and the word in the source
    environment._SCATTER_TOLD.clear()
    with pytest.warns(UserWarning, match="extrapolation"):
        got = environment.boundary_scattering({"wind_speed_m_s": 5.0}, 120.0e3)
    assert "extrapolation" in got["sources"]["surface"]


def test_the_new_fields_are_checked_and_a_new_bed_is_derived_anew():
    bad = environment.problems([{"name": "x", "water_depth_m": 50.0, "bottom": SAND, "bottom_scattering": {"model": "lambert"}}])
    assert any("no default strength" in m for _, m in bad)
    bad = environment.problems([{"name": "x", "wind_speed_m_s": 5.0, "surface_scattering": {"model": "chapman_harris", "wind_speed": 3.0}}])
    assert any("wind" in m for _, m in bad)
    bad = environment.problems([{"name": "x", "water_depth_m": 50.0, "bottom": SAND, "bottom_scattering": {"model": "chapman_harris"}}])
    assert any("on the bottom" in m for _, m in bad)
    # a child that changes the bed does not inherit the parent's stated bed scattering
    child = environment.load({"base": "shelf_100m_sand", "bottom": "medium_sand"})
    assert "bottom_scattering" not in child
    assert environment.load({"base": "shelf_100m_sand"})["bottom_scattering"] == {"model": "lambert", "mu_db": -27.0}


def test_every_shipped_environment_loads_and_no_scenario_takes_reverberation_from_it():
    """The new fields change no simulated record: a scenario's reverberation comes from its own `reverberation:` list
    unless it says `reverberation: site` (none of the shipped ones does)."""
    for name in environment.known():
        environment.load_named(name)
    assert environment.problems(depthplan.load_environments("illustrative")) == []
    for name in list_configs("scenarios"):
        cfg = load_scenario_config(name)
        assert "reverberation" not in cfg.get("_environment_filled", []), name


SITE_SCENES = {   # a scene that simulates the water of a planning site: its typed boundaries are the rule's for that site
    "popoto_12km": "shelf_100m_sand_12c", "popoto_12km_2row": "shelf_100m_sand_12c", "popoto_12km_3row": "shelf_100m_sand_12c",
    "popoto_12km_two_ships": "shelf_100m_sand_12c", "popoto_env_deep_silt_1000m": "slope_1000m_silt_profile",
    "popoto_env_winter_sand_200m": "winter_200m_sand_profile", "popoto_env_summer_mud_60m": "summer_60m_mud_profile",
    "flank_patrol": None, "flank_target_1km": None, "flank_target_5km": None, "flank_target_10km": None, "popoto_far_mover": None,
    "popoto_trial_100m": None, "popoto_trial_1000m": None, "popoto_trial_2000m": None}


@pytest.mark.parametrize("name", sorted(SITE_SCENES))
def test_the_sites_scattering_is_the_scenes_that_simulate_the_same_water(name):
    """For every shipped scene on a site that now states its scattering, the rule's items give the scene's own laws on the
    same grid (to 1e-12): the budget, the planner and the coverage picture use what the simulator uses."""
    parts = rb.scenario_parts(name)
    env, fc = parts["environment"], parts["frequency"]
    got = rule(env, fc)
    mine = rb.strength_tables(got["items"], fc, env["sound_speed"])
    theirs = rb.strength_tables(parts["items"], fc, env["sound_speed"])
    for side in ("bottom", "surface"):
        assert (mine[side] is None) == (theirs[side] is None), (name, side, got["notes"])
        if mine[side] is not None:
            assert np.allclose(mine[side], theirs[side], rtol=1e-12, atol=0.0), (name, side)


def test_reverberation_site_in_a_scenario_gives_the_records_of_the_typed_items():
    """`reverberation: site` expands to the rule's items; the records equal those of the same items typed (both
    simulator engines take the expanded list)."""
    base = {"pings": {"count": 1}, "sonar_overrides": {"receive": {"window": {"range": [5.0, 25.0]}}},
            "environment": {"base": "coastal_12c", "water_depth_m": 40.0, "wind_speed_m_s": 5.0, "bottom": SAND,
                            "bottom_scattering": {"model": "lambert", "mu_db": -27.0}}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        site = load_scenario_config("fls_120k_cluttered_seabed", dict(base, reverberation="site"))
        typed = load_scenario_config("fls_120k_cluttered_seabed", dict(base, reverberation=[{"mu_db": -27.0}, {"model": "chapman_harris"}]))
        assert "reverberation" in site["_environment_filled"]
        a = run_scenario(site).records[0].samples.array
        b = run_scenario(typed).records[0].samples.array
    assert np.allclose(a, b, rtol=1e-6, atol=1e-12 * float(np.abs(b).max()))
    with pytest.raises(ValueError, match="site"):
        load_scenario_config("fls_120k_cluttered_seabed", dict(base, reverberation="sité"))


def test_a_builds_coverage_block_carries_the_rule_and_says_what_is_not_included():
    from gpusonar import buildspec
    spec = buildspec.load("popoto_face_1x32")
    block = coverage.of_build(spec, {"environment": "shelf_100m_sand"})
    rev = block["reverberation"]
    assert rev["items"] == [{"model": "lambert", "mu_db": -27.0, "boundary": "bottom"}, {"model": "chapman_harris", "wind_speed": 5.0, "boundary": "surface"}]
    assert rev["frequency_hz"] == pytest.approx(5000.0) and "stated (bottom_scattering)" in rev["note"]
    deep = coverage.site_reverberation({"wind_speed_m_s": 5.0}, 5000.0)
    assert deep["items"][0]["boundary"] == "surface" and "bottom: not included" in deep["note"]
    none = coverage.site_reverberation({"water_depth_m": 50.0, "bottom": SAND}, 5000.0)
    assert set(none) == {"note"} and none["note"].startswith("not included: bottom:")
    o = coverage.cpp_options({"reverberation": none})
    assert o["reverb"] == 2 and o["reverb_note"].startswith("not included")

