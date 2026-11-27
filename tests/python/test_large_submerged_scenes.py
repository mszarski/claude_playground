"""The 12 km scenes for the class the owner set the requirement for first (2026-11-26: "Large submerged mostly but also
Large Ship is what we care about"; D-235; docs/tasks/DOPPLER.md, "Large submerged at 12 km"):

    popoto_12km_large_submerged        popoto_12km's water, array, platform and seed; three large_submerged targets
                                       (closing, oblique, crossing) in place of the ship; the shipped sweep only
    popoto_12km_large_submerged_tone   the same with a 1 s Hann tone at 6.5 kHz (0.5 kHz above the 4-6 kHz sweep, the
                                       rule of flank_patrol_doppler_tone) then the sweep, in one 2 s ping

- each file is its base with only what it states changed (the targets; the waveform);
- the targets carry the class file's values: depth inside its range, target strength at its low end, speed inside its
  range, the hull its illustrative length; the radial speeds the header states, against each beam's own-motion ridge;
- numbers are written so that PyYAML reads them as numbers (test_yaml_numbers.py's rule, here before git tracks them);
- one thinned ping of each runs on both synthesis engines, equal to the tolerance of test_sampled_engine.py (Bellhop).
"""

import copy

import numpy as np
import pytest
import yaml

from gpusonar import doppler, target_classes
from gpusonar.config import config_dir, load_scenario_config
from gpusonar.sim import run_scenario
from gpusonar.sim.bellhop import find_bellhop
from gpusonar.sonar import SonarModel

from test_doppler_scenes import _thinned
from test_sampled_engine import assert_same
from test_yaml_numbers import numberish_strings

SWEEP, TONE = "popoto_12km_large_submerged", "popoto_12km_large_submerged_tone"
LABELS = ("closing", "oblique", "crossing")
needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")


def _sonar(cfg):
    return SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))


def _class(name="large_submerged"):
    (c,) = [c for c in target_classes.load("reference_classes") if c["name"] == name]
    return c


@pytest.mark.parametrize("name", [SWEEP, TONE])
def test_the_scene_loads_and_writes_its_numbers_as_numbers(name):
    path = config_dir() / "scenarios" / f"{name}.yaml"
    assert list(numberish_strings(yaml.safe_load(path.read_text()))) == []
    cfg = load_scenario_config(name)
    assert cfg["name"] == name and cfg["fidelity"] == "eigenray"
    assert [it["label"] for it in cfg["scene"]] == list(LABELS)
    _sonar(load_scenario_config(name, expand=False))


def test_the_sweep_scene_is_popoto_12km_with_the_targets_replaced():
    a, b = load_scenario_config(SWEEP, expand=False), load_scenario_config("popoto_12km", expand=False)
    for c in (a, b):
        c.pop("name"), c.pop("scene"), c.pop("environment_overrides")
    assert a == b


def test_the_tone_scene_is_the_sweep_scene_with_a_tone_above_the_band():
    """Only the waveform differs; the tone sits 0.5 kHz above the sweep's top as flank_patrol_doppler_tone's does
    (10.5 kHz over 6-10 kHz); the train keeps popoto_long_range's sweep energy, half to each segment (as
    popoto_12km_doppler); the chain's tone keys come from the segment table."""
    a, b = copy.deepcopy(load_scenario_config(TONE, expand=False)), copy.deepcopy(load_scenario_config(SWEEP, expand=False))
    wave = a["sonar_overrides"].pop("transmit")["waveform"]
    for c in (a, b):
        c.pop("name"), c.pop("environment_overrides")
    assert a == b
    flank = load_scenario_config("flank_patrol_doppler_tone", expand=False)["sonar_overrides"]["transmit"]["waveform"]
    above = flank["segments"][0]["frequency"] - flank["segments"][2]["f_stop"]
    tone, gap, sweep = wave["segments"]
    assert above == 0.5e+3 and tone["frequency"] - sweep["f_stop"] == above
    assert wave["f_stop"] > tone["frequency"] and wave["f_start"] == sweep["f_start"]
    sonar, old = _sonar(load_scenario_config(TONE, expand=False)), _sonar(load_scenario_config(SWEEP, expand=False))
    owf = old.waveform_config
    seg = sonar.sampled_waveform.segments
    assert [s.kind for s in seg] == ["cw", "gap", "lfm"]
    assert (seg[2].f_start, seg[2].f_stop, seg[2].duration, seg[2].window_alpha) == (owf["f_start"], owf["f_stop"], owf["duration"], owf["window_alpha"])
    assert sonar.source_energy_db == pytest.approx(old.source_energy_db, abs=0.01)
    e = sonar.segment_energies_db()
    assert e[0] == pytest.approx(old.source_energy_db - 10.0 * np.log10(2.0), abs=0.01) and e[0] == pytest.approx(e[2], abs=1e-9)
    assert 0.9 < sonar.source_level_db_rms - old.source_level_db_rms < 1.2       # the Hann tone sets the peak (header: 1.0 dB)
    assert doppler.tone_from_sonar(sonar) == {"tone": "bank", "tone_frequency_hz": 6.5e+3, "tone_start_s": 0.0, "tone_duration_s": 1.0,
                                              "tone_window_alpha": 1.0}


@pytest.mark.parametrize("name", [SWEEP, TONE])
def test_the_targets_carry_the_class_files_values(name):
    """Depth inside the class's range, the five highlights summing to its LOW target strength, speed inside its range,
    the highlights spanning its illustrative hull length along the velocity."""
    cls = _class()
    lo, hi = cls["depth_m"]
    vlo, vhi = cls["speed_m_s"]
    for it in load_scenario_config(name, expand=False)["scene"]:
        p, v = np.asarray(it["positions"], float), np.asarray(it["velocity"], float)
        assert np.all((p[:, 2] >= lo) & (p[:, 2] <= hi)), it["label"]
        total = it["target_strength_db"] + 10.0 * np.log10(len(p))
        assert total == pytest.approx(cls["target_strength_db"][0], abs=0.01), it["label"]
        assert vlo <= np.linalg.norm(v) <= vhi, it["label"]
        axis = p[-1] - p[0]
        assert np.linalg.norm(axis) == pytest.approx(cls["cylinder"]["length_m"], abs=0.5), it["label"]
        assert abs(axis @ v) / (np.linalg.norm(axis) * np.linalg.norm(v)) > 0.9999, it["label"]       # the hull along its course


def test_the_radial_speeds_are_the_headers_against_each_beams_ridge():
    """Range rate (opening positive) and the beam's own-motion ridge (-v_platform . u) at the first ping, as the file's
    header states them: closing 6.0 m/s from its ridge, oblique 0.50 m/s, crossing on it."""
    cfg = load_scenario_config(SWEEP, expand=False)
    p0, vp = np.asarray(cfg["platform"]["position"], float), np.asarray(cfg["platform"]["velocity"], float)
    want = {"closing": (-15.0, -5.61, 6.00), "oblique": (0.0, -0.50, 0.50), "crossing": (15.0, -0.39, 0.00)}
    for it in cfg["scene"]:
        c, v = np.mean(np.asarray(it["positions"], float), axis=0), np.asarray(it["velocity"], float)
        d = c - p0
        u = d / np.linalg.norm(d)
        bearing, rate, ridge = np.degrees(np.arctan2(d[0], d[1])), (v - vp) @ u, -vp @ u
        b, r, off = want[it["label"]]
        assert (bearing, rate, abs(rate - ridge)) == (pytest.approx(b, abs=0.05), pytest.approx(r, abs=0.01), pytest.approx(off, abs=0.01)), it["label"]


@needs_bellhop
@pytest.mark.parametrize("name, label", [(SWEEP, "oblique"), (TONE, "crossing")])
def test_a_large_submerged_scene_runs_a_ping_on_both_engines(name, label):
    """Two hydrophones, one target, four eigenrays a path, a window of +-60 m about it, the scene's reverberation and
    noise (test_doppler_scenes._thinned): the C++ and NumPy engines agree to the tolerance of test_sampled_engine.py."""
    cfg = _thinned(name, label)
    a = run_scenario(copy.deepcopy(cfg), engine="cpp").records[0]
    b = run_scenario(copy.deepcopy(cfg), engine="numpy").records[0]
    assert np.abs(a.samples.array).max() > 0
    if name == TONE:
        assert int(a.transmit.waveform.type) == 3 and a.transmit.waveform.duration == pytest.approx(2.0)
    assert_same(a.samples.array, b.samples.array, f"{name}: one ping, C++ vs NumPy")
