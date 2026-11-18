"""One environment per site (gpusonar/environment.py): the schema both layers share, what a scenario's blocks take
from it, contradictions, and that the single description gives the records the spelled-out one gave."""

import warnings

import numpy as np
import pytest

from gpusonar import acoustics, depthplan, environment
from gpusonar.config import list_configs, load_scenario_config
from gpusonar.sim.simulator import run_scenario
from gpusonar.sonar import SonarModel, resolve_environment


def test_one_schema_serves_the_build_layer_and_the_scenarios():
    assert set(depthplan.ENVIRONMENT_FIELDS) <= set(environment.FIELDS)
    assert all(depthplan.ENVIRONMENT_FIELDS[k] is environment.FIELDS[k] for k in depthplan.ENVIRONMENT_FIELDS)
    envs = depthplan.load_environments("illustrative")
    assert environment.problems(envs) == [] and depthplan.environment_problems(envs) == []
    # a scenario may name an entry of the build layer's list, and change fields of it
    shelf = environment.resolve({"base": "shelf_100m_sand", "temperature_c": 12.0}, strict=True)
    assert shelf["site_water_depth"] == 100.0 and shelf["bottom"] == {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}
    assert shelf["wind_speed"] == 5.0 and shelf["temperature"] == 12.0
    assert "shelf_100m_sand" in environment.known() and "coastal_12c" in environment.known()
    with pytest.raises(FileNotFoundError):
        environment.resolve("no_such_site")


def test_spellings_and_unknown_fields():
    a = environment.resolve({"sound_speed": 1480.0, "water_depth": 30.0, "wind_speed": 4.0})
    b = environment.resolve({"sound_speed_m_s": 1480.0, "water_depth_m": 30.0, "wind_speed_m_s": 4.0})
    assert a == b and a["water_depth"] == 30.0 and a["site_water_depth"] == 30.0
    with pytest.raises(ValueError, match="given twice"):
        environment.resolve({"water_depth": 30.0, "water_depth_m": 40.0}, strict=True)
    with pytest.raises(ValueError, match="unknown field"):
        environment.resolve({"watr_depth_m": 30.0}, strict=True)
    assert environment.resolve({"sea_state": 3})["wind_speed"] == acoustics.sea_state_wind_speed(3)
    hidden = environment.resolve({"water_depth_m": 80.0, "water_depth_known": False}, strict=True)
    assert "water_depth" not in hidden and hidden["site_water_depth"] == 80.0


def test_sound_speed_comes_from_one_rule():
    assert environment.resolve(None)["sound_speed"] == acoustics.NOMINAL_SOUND_SPEED == 1500.0
    assert "sound_speed" in environment.resolve(None)["defaulted"]
    assert environment.resolve({"temperature_c": 12.0, "salinity_psu": 35.0})["sound_speed"] == pytest.approx(acoustics.sound_speed(12.0, 35.0, 0.0))
    prof = {"depth_m": [0.0, 100.0], "speed_m_s": [1480.0, 1490.0]}
    assert environment.resolve({"water_depth_m": 100.0, "profile": prof, "temperature_c": 12.0})["sound_speed"] == 1480.0
    assert environment.resolve({"water_depth_m": 100.0, "profile": prof, "reference_depth_m": 50.0})["sound_speed"] == pytest.approx(1485.0)
    assert resolve_environment({"temperature": 12.0}, depth=40.0)["sound_speed"] == pytest.approx(acoustics.sound_speed(12.0, 35.0, 40.0))
    # a typed speed beside a temperature that implies another: a warning, unless it is named as deliberate
    with pytest.warns(UserWarning, match="geometry uses the first"):
        environment.resolve({"sound_speed_m_s": 1500.0, "temperature_c": 12.0}, strict=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        environment.resolve({"sound_speed_m_s": 1500.0, "temperature_c": 12.0, "overrides": {"sound_speed_m_s": "nominal"}}, strict=True)
        environment.resolve({"sound_speed_m_s": 1497.0, "temperature_c": 12.0}, strict=True)       # within 1 m/s
    with pytest.raises(ValueError, match="outside its own profile"):
        environment.resolve({"water_depth_m": 100.0, "profile": prof, "sound_speed_m_s": 1500.0}, strict=True)


def test_profile_families():
    lin = environment.resolve({"water_depth_m": 200.0, "profile": {"family": "linear", "surface_speed_m_s": 1480.0, "gradient_per_s": 0.017}}, strict=True)
    assert lin["profile"] == {"depth": [0.0, 200.0], "speed": [1480.0, pytest.approx(1483.4)]}
    munk = environment.resolve({"water_depth_m": 5000.0, "profile": {"family": "munk"}}, strict=True)
    z, c = np.asarray(munk["profile"]["depth"]), np.asarray(munk["profile"]["speed"])
    assert z[0] == 0.0 and z[-1] == 5000.0 and abs(z[np.argmin(c)] - 1300.0) <= 100.0 and c.min() == pytest.approx(1500.0, abs=0.5)
    assert [w for w, _ in environment.problems([{"name": "x", "water_depth_m": 10.0, "profile": {"family": "exotic"}}])] == ["environments.x.profile"]


BASE = {"name": "t", "sonar": "flank_32", "fidelity": "image_source",
        "sonar_overrides": {"receive": {"window": {"range": [150.0, 260.0]}}},
        "platform": {"position": [0, 0, 20.0]}, "pings": {"count": 1},
        "scene": [{"type": "points", "positions": [[0.0, 200.0, 30.0]], "target_strength_db": 10.0}], "seed": 3}
SPELLED = {**BASE, "environment": {"sound_speed": 1500.0, "temperature": 12.0, "salinity": 35.0},
           "propagation": {"bottom_z": 60.0, "max_order": 1, "bottom": {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}},
           "reverberation": [{"boundary_z": 60.0, "mu_db": -27.0}, {"boundary_z": 0.0, "model": "chapman_harris", "wind_speed": 7.0}],
           "absorption": "ainslie_mccolm", "receiver": {"adc": False, "ambient_noise": {"wind_speed": 7.0, "shipping": 0.3}}}
ONCE = {**BASE, "environment": {"base": "coastal_12c_sand", "water_depth_m": 60.0, "wind_speed_m_s": 7.0, "shipping": 0.3, "absorption": "ainslie_mccolm",
                                "water_depth_known": False},
        "propagation": {"max_order": 1}, "reverberation": [{"mu_db": -27.0}, {"model": "chapman_harris"}],
        "receiver": {"adc": False, "ambient_noise": {"model": "wenz"}}}


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_the_single_description_gives_the_same_record(engine):
    a = run_scenario(SPELLED, engine=engine).records[0]
    b = run_scenario(ONCE, engine=engine).records[0]
    assert np.array_equal(a.samples.array, b.samples.array) and np.abs(a.samples.array).max() > 0
    assert a.environment.sound_speed == b.environment.sound_speed == 1500.0 and a.environment.temperature == b.environment.temperature == 12.0
    assert np.isnan(a.environment.water_depth) and np.isnan(b.environment.water_depth)       # the vehicle is not told
    known = run_scenario({**ONCE, "environment": {**ONCE["environment"], "water_depth_known": True}}, engine=engine).records[0]
    assert known.environment.water_depth == 60.0 and np.array_equal(known.samples.array, a.samples.array)


def test_blocks_take_their_site_values_and_contradictions_are_errors():
    sonar = SonarModel.load("flank_32", BASE["sonar_overrides"])
    env = environment.resolve(ONCE["environment"], strict=True)
    out = environment.apply_to_scenario(ONCE, env, sonar)
    assert out["propagation"] == {"max_order": 1, "bottom_z": 60.0, "bottom": {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}}
    assert out["reverberation"] == [{"mu_db": -27.0, "boundary_z": 60.0}, {"model": "chapman_harris", "boundary_z": 0.0, "wind_speed": 7.0}]
    assert out["receiver"]["ambient_noise"] == {"model": "wenz", "wind_speed": 7.0, "shipping": 0.3} and out["absorption"] == "ainslie_mccolm"
    assert "reverberation[1].wind_speed" in out["_environment_filled"]
    # applying twice changes nothing (an expanded scenario can be run)
    again = environment.apply_to_scenario({k: v for k, v in out.items() if k != "_environment_filled"}, env, sonar)
    assert {k: v for k, v in again.items() if k != "_environment_filled"} == {k: v for k, v in out.items() if k != "_environment_filled"}

    wrong = {**ONCE, "reverberation": [{"mu_db": -27.0, "boundary_z": 55.0}, {"model": "chapman_harris", "wind_speed": 5.0}]}
    with pytest.raises(ValueError) as err:
        environment.apply_to_scenario(wrong, env, sonar)
    assert "reverberation[0].boundary_z" in str(err.value) and "reverberation[1].wind_speed" in str(err.value)
    with pytest.raises(ValueError, match="contradict the environment"):
        run_scenario(wrong)
    marked = {**wrong, "environment_overrides": {"reverberation[0].boundary_z": "a ledge", "reverberation[1].wind_speed": "a lull"}}
    out = environment.apply_to_scenario(marked, env, sonar)
    assert out["reverberation"][0]["boundary_z"] == 55.0 and out["reverberation"][1]["wind_speed"] == 5.0
    with pytest.warns(UserWarning, match="no longer differ"):
        environment.apply_to_scenario({**ONCE, "environment_overrides": {"reverberation[1].wind_speed": "stale"}}, env, sonar)
    for block, key in (({"propagation": {"max_order": 1, "bottom_z": 70.0}}, "propagation.bottom_z"),
                       ({"propagation": {"max_order": 1, "bottom": {"speed": 1700.0, "density": 1.9, "attenuation": 0.8}}}, "propagation.bottom"),
                       ({"receiver": {"ambient_noise": {"wind_speed": 3.0}}}, "receiver.ambient_noise.wind_speed"),
                       ({"time_spread": {"surface": {"wind_speed": 9.0}}}, "time_spread.surface.wind_speed")):
        with pytest.raises(ValueError, match=key.replace("[", r"\[").replace(".", r"\.")):
            environment.apply_to_scenario({**ONCE, **block}, env, sonar)
    assert environment.apply_to_scenario({**ONCE, "absorption": "thorp"}, env, sonar)["absorption"] == "thorp"       # a model, not a site value
    spread = environment.apply_to_scenario({**ONCE, "time_spread": {"surface": True, "taps": 3}}, environment.resolve({**ONCE["environment"], "rms_wave_height_m": 0.4}, strict=True), sonar)
    assert spread["time_spread"]["surface"] == {"wind_speed": 7.0, "rms_height": 0.4}


def test_the_ray_tracers_frequency_follows_the_sonar():
    cfg = {**BASE, "fidelity": "eigenray", "environment": {"base": "shelf_100m_sand_12c"}, "propagation": {"range_step": 20.0}}
    sonar = SonarModel.load("flank_32", BASE["sonar_overrides"])
    out = environment.apply_to_scenario(cfg, environment.resolve(cfg["environment"], strict=True), sonar)
    prop = out["propagation"]
    assert prop["frequency"] == sonar.center_frequency == 8000.0 and prop["wind_speed"] == 5.0
    assert prop["environment"] == {"depth": 100.0, "sound_speed": 1500.0, "bottom_speed": 1650.0, "bottom_density": 1.9, "bottom_attenuation": 0.8}
    assert prop["max_range"] == pytest.approx(1.1 * (260.0 + 0.5 * 1500.0 * sonar.waveform_config["duration"]), rel=0.01)
    low = SonarModel.load("flank_32", {"transmit": {"waveform": {"f_start": 4000.0, "f_stop": 6000.0}}})
    assert environment.apply_to_scenario(cfg, environment.resolve(cfg["environment"], strict=True), low)["propagation"]["frequency"] == 5000.0
    with pytest.raises(ValueError, match=r"propagation\.frequency"):                     # the typed copy that no longer follows
        environment.apply_to_scenario({**cfg, "propagation": {"frequency": 8000.0}}, environment.resolve(cfg["environment"], strict=True), low)
    with pytest.raises(ValueError, match=r"propagation\.environment\.depth"):
        environment.apply_to_scenario({**cfg, "propagation": {"environment": {"depth": 90.0}}}, environment.resolve(cfg["environment"], strict=True), sonar)
    named = environment.apply_to_scenario({**cfg, "environment": {"base": "shelf_100m_sand_12c", "bottom": "medium_sand"}},
                                          environment.resolve({"base": "shelf_100m_sand_12c", "bottom": "medium_sand"}, strict=True), sonar)
    assert named["propagation"]["environment"]["sediment"] == "medium_sand" and "bottom_speed" not in named["propagation"]["environment"]


def test_every_shipped_scenario_states_its_site_once():
    site_keys = {"propagation": ("bottom_z", "bottom", "frequency"), "receiver.ambient_noise": ("wind_speed", "shipping")}
    for name in list_configs("scenarios"):
        typed = load_scenario_config(name, expand=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)                       # no stale override, no unexplained sound speed
            warnings.filterwarnings("ignore", message=".*unconfirmed.*")
            warnings.filterwarnings("ignore", message=".*placeholder.*")
            cfg = load_scenario_config(name)
        env = cfg["environment"]
        assert env["_resolved"] and env["sound_speed"] > 1400.0
        if "base" in typed:
            continue                                                          # a variant: its parent is checked
        for key in site_keys["propagation"]:
            assert key not in (typed.get("propagation") or {}), f"{name}: propagation.{key} is typed; it belongs to the environment"
        assert "environment" not in (typed.get("propagation") or {}), name
        noise = (typed.get("receiver") or {}).get("ambient_noise") or {}
        assert not any(k in noise for k in site_keys["receiver.ambient_noise"]), name
        for item in typed.get("reverberation") or []:
            assert "boundary_z" not in item and "wind_speed" not in item, name
        assert "absorption" not in typed, name
        for key, reason in (typed.get("environment_overrides") or {}).items():
            assert isinstance(reason, str) and len(reason) > 20, f"{name}: override {key} needs its reason"
