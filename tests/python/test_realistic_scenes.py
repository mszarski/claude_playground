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

_spec = importlib.util.spec_from_file_location(
    "gallery", pathlib.Path(__file__).resolve().parents[2] / "tools" / "gallery.py")
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)

HOST = gs.core.Backend.Host


def quick(name):
    cfg = load_scenario_config(name)
    if name in gallery.QUICK:
        gallery.QUICK[name](cfg)
    fig, metrics = gallery.RUNNERS[name](name, cfg, HOST, True)
    gallery.plt.close(fig)
    return metrics


def test_every_mode_has_a_realistic_scenario():
    modes = {name: load_scenario_config(name)["sonar"] for name in gallery.RUNNERS}
    sonars = set(modes.values())
    assert {"l12_5d", "p4_2v", "fls_300k", "fls_750k"} <= sonars            # forward-looking, phases 1 to 4
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
