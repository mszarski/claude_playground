"""The Doppler scenes of docs/tasks/DOPPLER.md package D2 (`flank_patrol_doppler`, `flank_patrol_doppler_underway`,
`popoto_far_mover_doppler`, `popoto_12km_doppler`): each is the scene it was made from with the ping replaced by a
train (a 1 s Hann tone at the band centre, then the scene's own sweep), the total energy kept and split in half.

- the files differ from their base scenes in the transmit waveform only (and the under-way one in the platform's
  speed), and the energies they state are the base sweep's;
- each runs one (thinned) ping on both synthesis engines, equal to float32 rounding (needs Bellhop);
- with GPUSONAR_DOPPLER_STUDY=1 (5 to 11 minutes a scene on one CPU core: not part of the suite),
  `tools/doppler_study.py` on the train against the baseline's margins (docs/tasks/doppler_baseline.json): each
  segment has half the energy, so a margin that noise sets falls by 3.0 dB and one that reverberation sets keeps its
  value. That holds for the sweep everywhere and for the tone where noise is the background (Popoto). It does NOT
  hold for the tone in flank_patrol: there the sweep's own reverberation fills the tone's Doppler cells (found by
  this package; the test holds it, and a control with the sweep muted gives the tone its ceiling back).
"""

import copy
import json
import os
import pathlib
import sys

import numpy as np
import pytest

from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario
from gpusonar.sim.bellhop import find_bellhop
from gpusonar.sonar import SonarModel

from test_sampled_engine import assert_same

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE = json.loads((ROOT / "docs" / "tasks" / "doppler_baseline.json").read_text())
SCENES = {"flank_patrol_doppler": "flank_patrol", "flank_patrol_doppler_underway": "flank_patrol",
          "popoto_far_mover_doppler": "popoto_far_mover", "popoto_12km_doppler": "popoto_12km"}
needs_bellhop = pytest.mark.skipif(find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
slow_study = pytest.mark.skipif(os.environ.get("GPUSONAR_DOPPLER_STUDY", "") in ("", "0"),
                                reason="the margin study takes minutes a scene on a CPU: GPUSONAR_DOPPLER_STUDY=1")


def _sonar(cfg):
    return SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides"))


@pytest.mark.parametrize("name", list(SCENES))
def test_a_doppler_scene_is_its_base_scene_with_a_train_of_the_same_energy(name):
    cfg, base = load_scenario_config(name, expand=False), load_scenario_config(SCENES[name], expand=False)
    a, b = copy.deepcopy(cfg), copy.deepcopy(base)
    for c in (a, b):
        c.pop("name")
        c.pop("environment_overrides", None)
        c.get("sonar_overrides", {}).get("transmit", {}).pop("waveform", None)
        if not c.get("sonar_overrides", {}).get("transmit", True):
            c["sonar_overrides"].pop("transmit")
    if name.endswith("_underway"):
        assert a["platform"].pop("velocity") == [1.5, 0, 0] and b["platform"].pop("velocity") == [0, 0, 0]
    assert a == b
    sonar, old = _sonar(cfg), _sonar(base)
    w = sonar.sampled_waveform
    kinds = [s.kind for s in w.segments]
    assert kinds == ["cw", "gap", "lfm"]
    tone, gap, sweep = w.segments
    owf = old.waveform_config
    assert (sweep.f_start, sweep.f_stop, sweep.duration, sweep.window_alpha) == \
        (owf["f_start"], owf["f_stop"], owf["duration"], owf["window_alpha"])
    assert tone.frequencies == (0.5 * (owf["f_start"] + owf["f_stop"]),) and tone.duration == 1.0 and tone.window_alpha == 1.0
    assert gap.duration == 0.0
    # The total energy is the base sweep's (to the 0.01 dB the file writes), split in half.
    assert sonar.source_energy_db == pytest.approx(old.source_energy_db, abs=0.01)
    seg = sonar.segment_energies_db()
    assert seg[0] == pytest.approx(old.source_energy_db - 10.0 * np.log10(2.0), abs=0.01) and seg[0] == pytest.approx(seg[2], abs=1e-9)
    # The peak the Hann tone needs: 1.0 to 1.3 dB over the base's source level (stated in each file's header).
    rise = sonar.source_level_db_rms - old.source_level_db_rms
    print(f"{name}: SL {sonar.source_level_db_rms:.2f} dB (+{rise:.2f}), energy {sonar.source_energy_db:.2f} dB re 1 uPa^2 s")
    assert 0.9 < rise < 1.2


def _thinned(name, label):
    """One ping of the scene with one target, two hydrophones, the receive window about the target and at most four
    eigenrays a path (the scene keeps 32, or 64 by energy): enough to synthesise the train through the scene's
    propagation (Bellhop) and its reverberation on both engines in a few seconds each instead of a minute."""
    cfg = load_scenario_config(name, expand=False)
    cfg["pings"] = dict(cfg["pings"], count=1)
    cfg["scene"] = [it for it in cfg["scene"] if it.get("label") == label]
    item = cfg["scene"][0]
    p = np.mean(np.asarray(item["positions"], dtype=float), axis=0)
    r = float(np.hypot(p[0], p[1]))
    over = cfg.setdefault("sonar_overrides", {})
    over.setdefault("array", {})["count"] = 2
    cfg["propagation"] = dict(cfg.get("propagation", {}), max_arrivals=4)
    over.setdefault("receive", {})["window"] = {"range": [r - 60.0, r + 60.0]}
    return cfg


@needs_bellhop
@pytest.mark.parametrize("name, label", [("flank_patrol_doppler", "small_vehicle"), ("flank_patrol_doppler_underway", "small_vehicle"),
                                         ("popoto_far_mover_doppler", "ship"), ("popoto_12km_doppler", "ship")])
def test_a_doppler_scene_runs_a_ping_on_both_engines(name, label):
    """Two hydrophones, one target, four eigenrays a path, a window of +-60 m about it, the scene's reverberation and noise: the C++ and
    NumPy engines agree to the tolerance of test_sampled_engine.py; the record carries the train."""
    cfg = _thinned(name, label)
    a = run_scenario(copy.deepcopy(cfg), engine="cpp").records[0]
    b = run_scenario(copy.deepcopy(cfg), engine="numpy").records[0]
    w = a.transmit.waveform
    assert int(w.type) == 3 and w.duration == pytest.approx(2.0)
    assert np.abs(a.samples.array).max() > 0
    assert_same(a.samples.array, b.samples.array, f"{name}: one ping, C++ vs NumPy")


# --- the baseline reproduced on the trains (slow) -------------------------------------------------------

def _baseline(run, label, waveform, range_rate=None):
    for t in BASELINE["runs"][run]["result"]["targets"]:
        if t["label"] == label and t["waveform"] == waveform and (range_rate is None or abs(t["range_rate_m_s"] - range_rate) < 0.02):
            return t
    raise KeyError((run, label, waveform, range_rate))


def _study(argv):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import doppler_study
    finally:
        sys.path.pop(0)
    return doppler_study.main(argv)


# (scene, target, its bearing from the baseline, baseline run for the tone, run for the sweep, range rate, options)
CASES = [
    ("flank_patrol_doppler", "small_vehicle", 23.4, "fp_cwhann", "fp_scene", None, ["--vmax", "3"]),
    ("flank_patrol_doppler_underway", "small_vehicle", 23.5, "fp_v15", "fp_v15_lfm", 1.38, ["--vmax", "4"]),
    ("popoto_far_mover_doppler", "ship", 6.9, "fm_cwhann", "fm_scene", None, ["--vmax", "3"]),
    ("popoto_12km_doppler", "ship", 3.4, "k12_cwhann", "k12_scene", None, ["--vmax", "3"]),
]
HALF = 10.0 * np.log10(2.0)
# The tone's background is a mean over a 1 s window of a filter about 1 Hz wide: about two independent values a ping,
# and the study takes two pings (noise: two). Its scatter is +-3 dB; measured against the expected -3.0 dB: ceilings
# +0.8, +1.5, +3.2, -2.3 dB, noise-limited bank margins +1.5, +1.5 dB. The sweep's background has hundreds of cells.
TONE_TOL, SWEEP_TOL = 3.5, 1.0


@needs_bellhop
@slow_study
@pytest.mark.parametrize("scene, label, bearing, tone_run, sweep_run, rate, extra", CASES)
def test_the_study_reproduces_the_baseline_on_the_train(scene, label, bearing, tone_run, sweep_run, rate, extra, tmp_path):
    """What half the energy in each segment predicts, and what it does not.

    - Ceilings (margins against noise alone) fall by 3.0 dB on both segments.
    - The sweep at zero Doppler: where noise sets its margin (the Popoto scenes) it falls 3.0 dB; where reverberation
      sets it (flank_patrol) it keeps its value within 2.0 dB (measured -0.6 and -1.5 dB: the tone's reverberation adds
      a little to the sweep filter's background).
    - The tone with the bank: where noise sets it (Popoto), baseline - 3.0 dB. Where the baseline's tone alone reached
      its ceiling by leaving the bottom's reverberation behind in Doppler (flank_patrol), the TRAIN'S tone does not:
      it is 25 to 29 dB under its ceiling (measured 10.9 / 36.3 dB still, 7.8 / 36.8 dB under way), because the
      sweep's reverberation, from ranges c T / 2 nearer, is broadband and fills every Doppler cell of the tone at the
      same delay (the control below mutes the sweep and the tone is back at its ceiling). Held here as found."""
    out = _study(["margin", scene, "--target", label, "--pings", "2", "--bearing", str(bearing), "--json",
                  str(tmp_path / "m.json"), *extra] + (["--radial", "2"] if rate is not None else []))
    got = {m["segment"].split(":")[1]: m for m in out["targets"]}
    tone_b = _baseline(tone_run, label, "cw_hann", rate)
    sweep_b = _baseline(sweep_run, label, "scene", rate)
    tone, sweep = got["cw"], got["lfm"]
    print(f"{scene}: tone bank {tone['bank_margin_db']:.1f} / {tone['noise_only_bank_margin_db']:.1f} dB "
          f"(baseline {tone_b['bank_margin_db']:.1f} / {tone_b['noise_only_bank_margin_db']:.1f}); sweep zero-Doppler "
          f"{sweep['zero_doppler_margin_db']:.1f} / {sweep['noise_only_zero_doppler_margin_db']:.1f} "
          f"(baseline {sweep_b['zero_doppler_margin_db']:.1f} / {sweep_b['noise_only_zero_doppler_margin_db']:.1f})")
    assert tone["noise_only_bank_margin_db"] == pytest.approx(tone_b["noise_only_bank_margin_db"] - HALF, abs=TONE_TOL)
    assert sweep["noise_only_zero_doppler_margin_db"] == pytest.approx(sweep_b["noise_only_zero_doppler_margin_db"] - HALF, abs=SWEEP_TOL)
    sweep_reverb_limited = sweep_b["noise_only_zero_doppler_margin_db"] - sweep_b["zero_doppler_margin_db"] > 6.0
    if sweep_reverb_limited:
        assert sweep["zero_doppler_margin_db"] == pytest.approx(sweep_b["zero_doppler_margin_db"], abs=2.0)
    else:
        assert sweep["zero_doppler_margin_db"] == pytest.approx(sweep_b["zero_doppler_margin_db"] - HALF, abs=SWEEP_TOL)
    if scene.startswith("flank_patrol"):
        # The sweep's reverberation in the tone's cells: the train's tone sits far under its ceiling.
        penalty = tone["noise_only_bank_margin_db"] - tone["bank_margin_db"]
        assert 20.0 < penalty < 35.0                       # measured 25.4 (still), 29.0 (under way)
        assert tone["bank_margin_db"] > sweep["zero_doppler_margin_db"] - 1.0   # still not worse than the sweep
    else:
        assert tone["bank_margin_db"] == pytest.approx(tone_b["bank_margin_db"] - HALF, abs=TONE_TOL)


@needs_bellhop
@slow_study
def test_muting_the_sweep_gives_the_tone_back_its_ceiling(tmp_path):
    """The control for the finding above: flank_patrol_doppler with the sweep replaced by silence (`--mute 2`; the tone
    keeps its half of the energy). The tone's bank margin returns to its ceiling, as the baseline's tone alone did."""
    out = _study(["margin", "flank_patrol_doppler", "--target", "small_vehicle", "--pings", "2", "--bearing", "23.4",
                  "--vmax", "3", "--mute", "2", "--json", str(tmp_path / "m.json")])
    (tone,) = out["targets"]
    print(f"tone alone: bank {tone['bank_margin_db']:.1f} / {tone['noise_only_bank_margin_db']:.1f} dB")
    assert tone["noise_only_bank_margin_db"] - tone["bank_margin_db"] < 4.0     # measured 33.9 / 36.3 dB: 2.4 (25.4 with the sweep)
