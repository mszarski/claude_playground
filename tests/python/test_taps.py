"""Taps: records captured after any stage can be inspected, exported, and fed back in."""

import json

import numpy as np
import pytest

from gpusonar import pipeline, taps
from gpusonar.sim import run_scenario

STEPS = [{"stage": "prepare"}, {"stage": "whiten", "gamma": 0.1},
         {"stage": "sas_image", "ground_range": [38.0, 44.0], "pixel": [0.04, 0.08], "interp": "linear"},
         {"stage": "drc", "target_median": 0.2}]


@pytest.fixture(scope="module")
def tapped(tmp_path_factory):
    d = tmp_path_factory.mktemp("taps")
    cfg = {"source": {"type": "scenario", "name": "stripmap_points", "sim_taps": str(d / "sim")}, "steps": STEPS,
           "backend": "host", "output": {"taps": {"dir": str(d / "stages")}}}
    return d, pipeline.run(cfg)


def test_every_stage_leaves_a_container_file_and_a_manifest(tapped):
    d, result = tapped
    m = taps.manifest(d / "stages")
    assert [e["stage"] for e in m] == ["source", "prepare", "whiten", "sas_image", "drc"]
    assert all(e["pings"] == len(result.records) for e in m)
    assert m[0]["kind"] != m[1]["kind"]                           # raw samples, then baseband
    assert m[3]["array_names"] == ["image"] and m[4]["array_names"] == ["display", "image"]
    source, prepared = taps.read_tap(d / "stages", "source"), taps.read_tap(d / "stages", "prepare")
    assert len(source) == len(result.records)
    assert not np.iscomplexobj(source[0].samples.array) and np.iscomplexobj(prepared[0].samples.array)
    assert np.allclose(taps.read_arrays(d / "stages", "sas_image")["image"], result.image)


def test_a_chain_restarted_from_a_tap_gives_the_same_image(tapped):
    d, result = tapped
    again = pipeline.run({"source": {"type": "tap", "dir": str(d / "stages"), "stage": "whiten"}, "steps": STEPS[2:],
                          "backend": "host"})
    assert np.allclose(again.image, result.image, rtol=1e-5, atol=1e-6 * np.abs(result.image).max())
    assert np.allclose(again.display, result.display, atol=1e-5)
    # From the raw source tap the whole chain runs again.
    full = pipeline.run({"source": {"type": "tap", "dir": str(d / "stages"), "stage": "source"}, "steps": STEPS,
                         "backend": "host"})
    assert np.allclose(full.image, result.image, rtol=1e-4, atol=1e-5 * np.abs(result.image).max())


def test_simulator_stages_are_captured_in_order(tapped):
    d, result = tapped
    stages = taps.read_sim_taps(d / "sim", ping=0)
    assert list(stages) == list(taps.SIM_STAGES)
    assert np.array_equal(stages["adc"], taps.read_tap(d / "stages", "source")[0].samples.array)
    meta = json.loads((d / "sim" / "sim_taps.json").read_text())
    assert len(meta) == len(taps.SIM_STAGES) * len(result.records)
    assert stages["echoes"].shape == stages["adc"].shape and np.abs(stages["echoes"]).max() > 0


def test_noise_shows_up_at_its_own_stage(tmp_path):
    with taps.sim_capture(tmp_path):
        run_scenario("fls_120k_transit", {"pings": {"count": 1}, "scene": [
            {"type": "points", "label": "p", "positions": [[100.0, 0.0, 20.0]], "target_strength_db": 0.0}]})
    s = taps.read_sim_taps(tmp_path)
    early = slice(0, 400)                                          # before any echo or reverberation arrives
    assert np.abs(s["echoes"][:, early]).max() == 0.0
    assert np.abs(s["noise"][:, early]).std() > 0.0                # ambient noise is added here
    assert np.abs(s["reverberation"] - s["echoes"]).max() > 0.0    # and the boundaries here


def test_wav_export_and_stats(tapped, tmp_path):
    d, _ = tapped
    raw, bb = taps.read_tap(d / "stages", "source")[0], taps.read_tap(d / "stages", "prepare")[0]
    info = taps.export_wav(raw, tmp_path / "raw.wav", channels=[0, 3])
    assert info["layout"] == ["ch0", "ch3"]
    data = np.frombuffer((tmp_path / "raw.wav").read_bytes()[44:], dtype="<f4").reshape(-1, 2)
    assert np.allclose(data[:, 1] / info["scale"], raw.samples.array[3], rtol=1e-5, atol=1e-6 * np.abs(raw.samples.array).max())
    assert taps.export_wav(bb, tmp_path / "bb.wav", channels=[0])["layout"] == ["ch0_I", "ch0_Q"]
    st = taps.stats(raw)
    assert len(st) == raw.samples.n_channels and all(c["rms"] > 0 and c["peak"] >= c["rms"] for c in st)
    f0, f1 = raw.transmit.waveform.f_start, raw.transmit.waveform.f_stop
    assert min(f0, f1) * 0.9 <= st[0]["strongest_line_hz"] <= max(f0, f1) * 1.1
