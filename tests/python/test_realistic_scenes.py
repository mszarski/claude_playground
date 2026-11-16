"""Every mode has a realistic scenario that simulates and images end to end.

These run `tools/gallery.py` in its quick mode (thinned scenes, fewer pings, same
geometry and processing) and check that each object shows where it should. The full-size
scenes run on a GPU through the same tool; their figures are in docs/figures and their
numbers in docs/SCENARIOS.md.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.config import load_scenario_config
from gpusonar.sim import find_bellhop

pytest.importorskip("matplotlib", reason="the gallery tool draws its figures with matplotlib")
_spec = importlib.util.spec_from_file_location(
    "gallery", pathlib.Path(__file__).resolve().parents[2] / "tools" / "gallery.py")
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)

HOST = gs.core.Backend.Host


def quick(name):
    cfg = load_scenario_config(name)
    if name in gallery.QUICK:
        gallery.QUICK[name](cfg)
    out = gallery.RUNNERS[name](name, cfg, HOST, True)
    gallery.plt.close(out[0])
    return out[1]


def test_every_mode_has_a_realistic_scenario():
    modes = {name: load_scenario_config(name)["sonar"] for name in gallery.RUNNERS}
    sonars = set(modes.values())
    assert {"l12_5d", "p4_2v", "fls_300k", "fls_750k"} <= sonars            # forward-looking, phases 1 to 4
    assert {"fls_120k", "fls_170k", "fls_330k", "sas_minsas120", "popoto_array"} <= sonars   # the owner's designs
    assert {"sas_auv", "sas_rail", "subbottom"} <= sonars                   # SAS, rail SAS, 3D
    assert {"flank_32", "popoto_8"} <= sonars                               # long-range arrays
    for name in gallery.RUNNERS:
        scene = load_scenario_config(name)["scene"]
        if name.startswith("popoto"):
            continue                                                        # a point target in reverberation and noise
        assert len(scene) >= 2 and {item["type"] for item in scene} - {"points"} or name == "flank_patrol", name


def test_pool_phase_1():
    m = quick("pool_fls_l12_5d")
    lv = m["object_over_median_db"]
    assert lv["cinder_block"] > 30 and lv["pipe"] > 35 and lv["far_wall"] > 20
    assert lv["wire_1m"] > 50 and lv["wire_2m"] > 40
    assert lv["rope"] < 10                                   # the soft target is barely there
    assert 0.25 < m["hidden_fraction"] < 0.6                 # back faces and what the block and pipe shadow


def test_pool_phase_2():
    lv = quick("pool_fls_p4_2v")["object_over_median_db"]
    assert lv["far_wall"] > 35
    assert min(lv[f"step_{i}"] for i in (1, 2, 3)) > 25
    assert min(v for k, v in lv.items() if k.startswith("ladder")) > 40
    assert min(v for k, v in lv.items() if k.startswith("float")) > 35


def test_seafloor_objects_and_harbour():
    lv = quick("seafloor_objects_fls")["object_over_median_db"]
    assert lv["sphere"] > 35 and lv["box"] > 35 and lv["rocks"] > lv["seafloor"]
    m = quick("harbour_fls_750k")
    lv = m["object_over_median_db"]
    assert min(lv[f"piling_{i}"] for i in range(1, 7)) > 50
    assert lv["quay_wall"] > 50 and lv["mooring_block"] > 35
    assert lv["seafloor"] < lv["quay_wall"] - 10
    assert m["hidden_fraction"] > 0.1                        # shadows behind pilings, block and rocks


def test_rail_sas_over_a_floor_with_objects():
    m = quick("rail_objects")
    lv = m["object_over_median_db"]
    assert lv["pipe"] > 30 and lv["brick"] > 28 and lv["ball"] > 25 and lv["pebbles"] > 25


@pytest.mark.skipif(not gs.core.HAVE_MOTION, reason="built without Ceres")
@pytest.mark.parametrize("name", ["wreck_sas", "minefield_sas"])
def test_sas_scenes_with_unknown_motion(name):
    m = quick(name)
    assert m["sway_error_rms_wavelengths"] < 0.2 and m["heave_error_rms_wavelengths"] < 0.25
    s = m["sharpness"]
    assert s["Estimated motion"] > 0.8 * s["True motion"]
    assert s["Estimated motion"] > 2.0 * s["Navigation only"]


def test_buried_objects_volume():
    m = quick("buried_objects")
    cyl, sph = m["cylinder_top_depth_m"], m["sphere_top_depth_m"]
    assert sph["refraction_on"] == pytest.approx(0.14, abs=0.02)
    assert 0.24 <= cyl["refraction_on"] <= 0.31              # between its top and its axis
    assert cyl["refraction_off"] < cyl["refraction_on"]      # without refraction it images too shallow
    assert m["cylinder_over_clutter_db"] > 12


@pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_flank_patrol_detects_and_tracks():
    m = quick("flank_patrol")
    sub, wreck = m["targets"]["submarine"], m["targets"]["wreck"]
    assert sub["tracked"] and sub["position_error_m"] < 60
    assert sub["track_velocity"][0] == pytest.approx(3.0, abs=1.0)
    assert abs(sub["track_velocity"][1]) < 1.0
    assert wreck["tracked"] and np.hypot(*wreck["track_velocity"]) < 0.6     # it does not move
    assert max(m["detections_per_ping"]) <= 6                               # no flood of false alarms


# --- the owner's designs: moving ownship, world-fixed boundaries -----------------------------

def test_fls_120k_transit_tracks_static_and_moving_targets_from_a_moving_auv():
    m = quick("fls_120k_transit")
    t = m["targets"]
    assert t["boat_under_way"]["tracked"] and t["boat_under_way"]["track_speed_m_s"] == pytest.approx(4.0, abs=1.0)
    assert t["boat_under_way"]["track_velocity_xy"][1] < -2.5                # crossing to port
    for name, limit in (("boat_at_anchor", 0.6), ("mooring_buoy", 0.6), ("pinnacle", 1.2)):
        assert t[name]["tracked"], name
        assert t[name]["track_speed_m_s"] < limit, name                       # fixed in the world, though ownship moves
    assert max(m["detections_per_ping"]) <= 12


@pytest.mark.parametrize("name", ["fls_170k_transit", "fls_330k_transit"])
def test_high_frequency_fls_sees_the_surface_targets_in_its_tilted_beam(name):
    m = quick(name)
    t = m["targets"]
    assert t["boat_under_way"]["tracked"] and t["boat_under_way"]["track_speed_m_s"] == pytest.approx(4.0, abs=1.2)
    assert t["mooring_buoy"]["tracked"] and t["mooring_buoy"]["track_speed_m_s"] < 0.6
    assert not t["pinnacle"]["tracked"]                                      # the seabed is below the 5 degree beam


def test_sas_survey_images_the_seabed_from_both_sides():
    m = quick("sas_minsas_survey")
    one, two = m["legs"]["leg_1"], m["legs"]["leg_2"]
    assert one["sonar_side"] != two["sonar_side"]
    for leg in (one, two):
        assert leg["covered_fraction"] > 0.4
        assert leg["far_strip_over_near_strip_db"] < -10.0                   # the wreck's shadow falls away from the sonar
        lv = leg["object_over_median_db"]
        assert lv["wreck_hull"] > 20 and lv["calibration_cube"] > 15 and lv["cable"] > 15


@pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_popoto_array_tracks_a_distant_mover_and_gains_with_size():
    m = quick("popoto_far_mover")
    ship = m["targets"]["ship"]
    assert ship["tracked"] and ship["position_error_m"] < 200.0
    assert ship["track_velocity_xy"][0] < -2.0                               # moving towards -x
    gain = m["peak_over_median_db_by_hydrophone_count"]
    assert gain["8"] < gain["16"] < gain["32"]
    assert gain["32"] - gain["8"] == pytest.approx(6.0, abs=2.5)             # 10 log10(4)


# --- more examples: realistic boats, clutter, waves, rocks, other oceans -----------------------

def test_fls_over_a_cluttered_seabed_still_tracks_the_boats():
    m = quick("fls_120k_cluttered_seabed")
    t = m["targets"]
    assert set(t) == {"boat_under_way", "boat_at_anchor"}                    # areas of seabed are not targets
    assert t["boat_under_way"]["tracked"] and t["boat_under_way"]["position_error_m"] < 4.0
    assert t["boat_under_way"]["track_velocity_xy"][1] < -2.5                # crossing to port at 5 m/s
    assert t["boat_at_anchor"]["tracked"] and t["boat_at_anchor"]["track_speed_m_s"] < 1.2
    # Boulders, debris and the pipeline are real echoes: many detections and tracks that are not boats.
    assert min(m["detections_per_ping"]) > 20 and m["confirmed_tracks"] > 10


def test_fls_170k_separates_closing_crossing_and_anchored_boats():
    t = quick("fls_170k_traffic")["targets"]
    assert t["boat_closing"]["tracked"] and t["boat_closing"]["track_velocity_xy"][0] < -4.5
    assert t["boat_crossing"]["tracked"] and t["boat_crossing"]["track_velocity_xy"][1] > 3.0
    assert t["boat_at_anchor"]["tracked"] and t["boat_at_anchor"]["track_speed_m_s"] < 0.6


@pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_popoto_long_range_design_holds_a_ship_at_12_km():
    m = quick("popoto_12km")
    ship = m["targets"]["ship"]
    assert ship["tracked"] and ship["position_error_m"] < 150.0
    assert ship["track_velocity_xy"][0] < -2.0
    gain = m["peak_over_median_db_by_hydrophone_count"]
    assert gain["8"] < gain["16"] < gain["32"] and gain["32"] > 18.0


@pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_the_ocean_decides_what_the_popoto_array_sees():
    # Same sonar, same ship. Winter water over sand at 8 km: tracked. Summer thermocline over
    # mud at 4 km, half the range: the downward-bent sound is absorbed by the bottom and the
    # ship near the surface is not detected.
    winter, summer = quick("popoto_env_winter_sand_200m"), quick("popoto_env_summer_mud_60m")
    assert winter["targets"]["ship"]["tracked"] and winter["targets"]["ship"]["position_error_m"] < 150.0
    assert not summer["targets"]["ship"]["tracked"]
    assert (winter["peak_over_median_db_by_hydrophone_count"]["32"]
            - summer["peak_over_median_db_by_hydrophone_count"]["32"]) > 10.0


@pytest.mark.parametrize("name, rocks, floor", [("sas_minsas_boulder_field", ["boulders"], "sand"),
                                                ("sas_minsas_rocky_ripples", ["outcrop", "stones", "cable"], "rippled_sand")])
def test_minsas_images_rocks_above_the_seabed(name, rocks, floor):
    leg = quick(name)["legs"]["leg_1"]
    assert leg["covered_fraction"] > 0.6
    lv = leg["object_over_median_db"]
    for label in rocks:
        assert lv[label] > lv[floor] + 4.0, label
