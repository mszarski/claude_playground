"""sonar-1rf: every options dictionary of the live library refuses a key it does not read, by name and with the nearest
key it does read (as LiveChainCpp since D-166), and so do their Python counterparts; the key=value options of
build/bin/sonar_live name the nearest key too. Before, a misspelt key left its setting at the default without a word."""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import live
from gpusonar.sonar import SonarModel

core = gs.core


@pytest.fixture(scope="module")
def record():
    return SonarModel.load("flank_32").make_record(core.Pose())


@pytest.fixture(scope="module")
def reference():
    """The shared reference signal as a sonar file describes it (mode, level and channels beside what the sync reads)."""
    return dict(SonarModel.load("popoto_static_3row_clocks").config["receive"]["reference"])


def refused(where, key, nearest):
    """The message every binding gives: where, the key, and the nearest key read."""
    return (where + key + ": unknown key; did you mean '" + nearest + "'?").replace(".", r"\.").replace("?", r"\?")


def test_ping_assembler_refuses_unknown_decimate_and_reference_keys(record, reference):
    core.PingAssemblerCpp(record, {"stopband_db": 60.0})
    clocks = SonarModel.load("popoto_static_3row_clocks").make_record(core.Pose())
    core.PingAssemblerCpp(clocks, {}, reference)                    # the sonar's description of the burst: taken as it is
    with pytest.raises(ValueError, match=refused("PingAssemblerCpp: decimate.", "f_lo", "f_low")):
        core.PingAssemblerCpp(record, {"f_lo": 30e3})
    with pytest.raises(ValueError, match=refused("PingAssemblerCpp: reference.", "serch_time", "search_time")):
        core.PingAssemblerCpp(record, {}, dict(reference, serch_time=5e-3))


def test_the_other_decimate_and_reference_bindings_refuse_alike(record, reference):
    with pytest.raises(ValueError, match=refused("live_decimate_plan: decimate.", "stopband", "stopband_db")):
        core.live_decimate_plan(record, {"stopband": 60.0})
    with pytest.raises(ValueError, match=refused("PingDecimator: decimate.", "taps_", "taps")):
        core.PingDecimator(record, {"taps_": 99})
    with pytest.raises(ValueError, match=refused("StreamAssemblerCpp: decimate.", "output_rat", "output_rate")):
        core.StreamAssemblerCpp(record, 128, {"output_rat": 4000.0})
    with pytest.raises(ValueError, match=refused("live_synchronise: reference.", "tolerance", "tolerance_us")):
        core.live_synchronise(record, dict(reference, tolerance=2.0))
    with pytest.raises(ValueError, match=refused("live_decimate_record: reference.", "clip_levl", "clip_level")):
        core.live_decimate_record(record, {}, dict(reference, clip_levl=0.9))


def test_sound_speed_ladder_and_family_profile_refuse_unknown_keys():
    core.SoundSpeedLadder({"sample_sigma": 0.1})
    with pytest.raises(ValueError, match=refused("SoundSpeedLadder: ", "sample_sigm", "sample_sigma")):
        core.SoundSpeedLadder({"sample_sigm": 0.1})
    with pytest.raises(ValueError, match=refused("live_family_profile: ", "layer_dept", "layer_depth")):
        core.live_family_profile({"family": "mixed_layer", "layer_dept": 30.0}, 0.0, 1500.0, 300.0)
    with pytest.raises(ValueError, match=refused("live_water_profile: profile.", "bin", "bin_m")):
        core.live_water_profile({"bin": 2.0}, None, [], [], False, 0.0, 0.05, 10.0, 1500.0)


def test_navigation_and_water_depth_refuse_unknown_keys():
    core.NavigationCpp({"keep_s": 30.0, "water_depth": {"from_altitude": True, "hold_s": 10.0}})
    with pytest.raises(ValueError, match=refused("NavigationCpp: ", "keep", "keep_s")):
        core.NavigationCpp({"keep": 30.0})
    with pytest.raises(ValueError, match=refused("NavigationCpp: water_depth.", "hold", "hold_s")):
        core.NavigationCpp({"water_depth": {"from_altitude": True, "hold": 10.0}})
    with pytest.raises(ValueError, match=refused("WaterDepthTracker: ", "max_altitude", "max_altitude_m")):
        core.WaterDepthTracker({"from_altitude": True, "max_altitude": 200.0})
    # `source`, the deployment file's word for it (passed beside from_altitude by test_nav_adapter), is read: "altitude"
    # is from_altitude, and the two must agree.
    assert core.WaterDepthTracker({"source": "altitude"}).update(0, 0.0, 0.0, 10.0, 50.0)[0] == 60.0
    assert np.isnan(core.WaterDepthTracker({"source": "datagram"}).update(0, 0.0, 0.0, 10.0, 50.0)[0])
    with pytest.raises(ValueError, match="disagree"):
        core.WaterDepthTracker({"source": "datagram", "from_altitude": True})
    with pytest.raises(ValueError, match="altitude or datagram"):
        core.WaterDepthTracker({"source": "sonar"})


def test_derive_and_detector_plan_refuse_unknown_keys(record):
    assert core.live_derive(record, {"range_cell_m": 2.0})["range_cell_m"] == 2.0
    with pytest.raises(ValueError, match=refused("live_derive: ", "range_cel_m", "range_cell_m")):
        core.live_derive(record, {"range_cel_m": 2.0})
    args = (record, 100, 2, 0.5, [0.0, 0.01], 1, 10, 20, True)
    core.live_detector_plan(*args, {"merge_m": 3.0})
    with pytest.raises(ValueError, match=refused("live_detector_plan: ", "merge", "merge_m")):
        core.live_detector_plan(*args, {"merge": 3.0})


def test_a_key_far_from_every_key_read_is_named_without_a_guess():
    with pytest.raises(ValueError, match=r"SoundSpeedLadder: zzzzzzzzzzzz: unknown key \(.*\)$"):
        core.SoundSpeedLadder({"zzzzzzzzzzzz": 1.0})


# -- the Python counterparts --------------------------------------------------------------------------------------------

def test_python_decimate_nav_and_profile_sections_name_the_nearest_key():
    from gpusonar.elevation import profile_options
    with pytest.raises(ValueError, match=refused("decimate.", "f_lo", "f_low")):
        live.decimate_options({"f_lo": 30e3})
    with pytest.raises(ValueError, match=refused("nav.", "keep", "keep_s")):
        live.NavListener.from_config({"static": {"x": 0.0}, "keep": 30.0})          # before: passed over
    with pytest.raises(ValueError, match=refused("nav.water_depth.", "hold", "hold_s")):
        live.NavListener(water_depth={"source": "altitude", "hold": 10.0})
    with pytest.raises(ValueError, match=refused("nav.water_depth.", "hold", "hold_s")):
        live.NavListener(water_depth={"source": "datagram", "hold": 10.0})           # before: passed over unless altitude
    with pytest.raises(ValueError, match=refused("elevation.profile.", "bilt", "built")):
        profile_options({"bilt": {}})
    with pytest.raises(ValueError, match=refused("elevation.profile.built.", "sample_sigma", "sample_sigma_m_s")):
        profile_options({"built": {"sample_sigma": 0.1}})                             # before: passed over
    with pytest.raises(ValueError, match=refused("elevation.profile.supplied.", "sigma", "sigma_m_s")):
        profile_options({"supplied": {"sigma": 0.1}})                                 # before: passed over


def test_python_detector_plan_and_budget_refuse_an_unknown_chain_key(record):
    """live.detector_plan and live.budget_threshold_db take a chain configuration (the Python side of live_detector_plan
    and live_derive): merged over the defaults, a misspelt key was passed over."""
    with pytest.raises(ValueError, match=refused("chain.cfar.", "merge", "merge_m")):
        live.detector_plan(record, {"cell_statistic": "multiscale", "cfar": {"merge": 3.0}})
    with pytest.raises(ValueError, match=refused("chain.", "range_cell", "range_cell_m")):
        live.budget_threshold_db(record, {"range_cell": 2.0})
    assert np.isfinite(live.budget_threshold_db(record, {"range_cell_m": 2.0}))
