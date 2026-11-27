"""The GPU job's Doppler mode (tools/gpu_job.sh GPUSONAR_JOB=doppler, tools/doppler_gpu_runs.sh) and the switches it uses:
`--backend` / `--sim-backend` of tools/doppler_tone_eval.py, `--at-default` of tools/doppler_bank_threshold.py, and
`--doppler-combine`, `--doppler-tone`, `--scene`, `--backend`, `--json` of tools/live_chain_timing.py. Each switch is
opt-in: without it a tool does what it did. Also `--set`, `--backend`, `--sim-backend` and `--off-only` of
tools/doppler_bank_eval.py and the opt-in `hfm12` sub-run (docs/tasks/DOPPLER.md "HFM"), whose quick host form runs here. The whole mode in its quick host form (GPUSONAR_DOPPLER_QUICK=1,
GPUSONAR_DOPPLER_BACKEND=host: the same command lines with small sizes, the scenes in free field without boundary
reverberation) takes about ten minutes on a CPU: GPUSONAR_DOPPLER_JOB_QUICK=1 runs it."""
import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from gpusonar import core

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import doppler_bank_eval  # noqa: E402
import doppler_bank_threshold  # noqa: E402
import doppler_tone_eval  # noqa: E402
import live_chain_timing  # noqa: E402


def _timing_args(argv):
    """live_chain_timing's parsed arguments (its own parser, run up to the parse)."""
    seen = {}

    class Stop(Exception):
        pass

    real = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        seen["a"] = real(self, args, namespace)
        raise Stop

    argparse.ArgumentParser.parse_args = grab
    try:
        live_chain_timing.main(argv)
    except Stop:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real
    return seen["a"]


def test_timing_without_the_new_switches_runs_what_it_ran():
    assert live_chain_timing.variants(_timing_args([]), None) == [("plain", None)]
    # --doppler alone: the bank as before (no `combine` key: the chain's default, max).
    assert live_chain_timing.variants(_timing_args(["--doppler"]), None) == [
        ("max", {"mode": "sweep_bank", "max_range_rate_m_s": 3.0})]


def test_timing_rows_for_each_combine_and_the_tone():
    a = _timing_args(["--doppler", "--doppler-combine", "off", "max", "cfar", "--doppler-max-rate", "2"])
    assert live_chain_timing.variants(a, None) == [
        ("off", None), ("max", {"mode": "sweep_bank", "max_range_rate_m_s": 2.0}),
        ("cfar", {"mode": "sweep_bank", "max_range_rate_m_s": 2.0, "combine": "cfar"})]
    tone = {"tone": "bank", "tone_frequency_hz": 5000.0}
    got = live_chain_timing.variants(_timing_args(["--doppler-tone"]), tone)
    assert got == [("off", None), ("tone", dict(tone, report=True, max_range_rate_m_s=3.0))]
    with pytest.raises(SystemExit, match="needs --scene"):
        live_chain_timing.variants(_timing_args(["--doppler-tone"]), None)


def test_tone_eval_takes_the_chain_and_simulator_backends():
    assert doppler_tone_eval.sim_backend("host") == core.Backend.Host
    assert doppler_tone_eval.sim_backend("cuda") == core.Backend.Cuda
    with pytest.raises(ValueError):
        doppler_tone_eval.sim_backend("auto")
    old = doppler_tone_eval.BACKEND
    try:
        assert doppler_tone_eval.chain_config("cpp", None, 3.0)["backend"] == "host"          # the default, as before
        doppler_tone_eval.BACKEND = "cuda"
        assert doppler_tone_eval.chain_config("cpp", {"tone": "bank"}, 3.0)["backend"] == "cuda"
    finally:
        doppler_tone_eval.BACKEND = old


def test_the_count_at_the_default_budget_on_the_host(tmp_path):
    """Two noise pings, bank off and cfar at 1 m/s: the rows, the budget as shipped and the JSON written as it goes."""
    out = tmp_path / "fa.json"
    assert doppler_bank_threshold.main(["--at-default", "--combines", "off,cfar", "--max-rates", "1", "--pings", "2", "--backend", "host",
                                        "--workers", "1", "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["budget"] == 1e-3 and res["asked"] == 2 and set(res["rows"]) == {"off", "cfar/1"}
    off, cfar = res["rows"]["off"], res["rows"]["cfar/1"]
    assert off["pings"] == cfar["pings"] == 2 and off["complete"] and cfar["complete"]
    assert off["per_beam"] == 1 and cfar["per_beam"] == 7          # 1 m/s on the long-range array: 7 a beam (D3.2)
    assert cfar["threshold_db"] > off["threshold_db"]              # the union bound's correction
    assert off["backend"].lower().endswith("host")
    assert res["resolves"]["zero_count_upper_95_per_ping"] == pytest.approx(3.689 / 2, rel=1e-3)


def test_set_changes_the_scene_and_the_cache_name(tmp_path):
    cfg = doppler_tone_eval.apply_sets({"name": "x", "seed": 3, "pings": {"count": 2}, "receiver": {"adc": False}},
                                       ["fidelity=free_field", "reverberation=[]", "receiver.ambient_noise.level_db=50.0e+0"])
    assert cfg["fidelity"] == "free_field" and cfg["reverberation"] == [] and cfg["receiver"] == {"adc": False, "ambient_noise": {"level_db": 50.0}}
    assert doppler_tone_eval.apply_sets({"a": 1}, []) == {"a": 1}                 # nothing asked: nothing changed, no marker
    with pytest.raises(ValueError):
        doppler_tone_eval.apply_sets({}, ["fidelity"])


def test_the_gpu_job_has_the_mode_and_the_runner_refuses_no_output_directory(tmp_path):
    job = (ROOT / "tools" / "gpu_job.sh").read_text()
    assert 'GPUSONAR_JOB:-tests}" = "doppler"' in job and "tools/doppler_gpu_runs.sh /work/doppler" in job
    got = subprocess.run(["bash", str(ROOT / "tools" / "doppler_gpu_runs.sh")], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert got.returncode == 2 and "usage:" in got.stderr


@pytest.mark.skipif(os.environ.get("GPUSONAR_DOPPLER_JOB_QUICK", "") in ("", "0"),
                    reason="the mode's quick host form takes about ten minutes: GPUSONAR_DOPPLER_JOB_QUICK=1")
def test_the_mode_in_its_quick_host_form(tmp_path):
    out = tmp_path / "doppler"
    env = dict(os.environ, GPUSONAR_DOPPLER_QUICK="1", GPUSONAR_DOPPLER_BACKEND="host", OMP_NUM_THREADS="1", PYTHON=sys.executable)
    env.pop("GPUSONAR_UPLOAD_REPO", None)
    got = subprocess.run(["bash", str(ROOT / "tools" / "doppler_gpu_runs.sh"), str(out)], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=3600)
    assert got.returncode == 0, got.stdout[-4000:] + got.stderr[-2000:]
    tone = json.loads((out / "doppler_tone_eval.json").read_text())
    assert tone["backend"] == "host" and tone["sim_backend"] == "host" and tone["engines_equal_on"]
    assert set(tone["engines"]) == {"python", "cpp"} and tone["background"]["default"]["pings"] == 1
    assert tone["set"] == ["fidelity=free_field", "reverberation=[]"]
    fa = json.loads((out / "doppler_bank_at_default.json").read_text())
    assert set(fa["rows"]) == {"off", "max/1", "cfar/1"} and all(r["pings"] == 3 for r in fa["rows"].values())
    trial = json.loads((out / "timing_trial.json").read_text())
    assert sorted((r["variant"], r["engine"]) for r in trial) == sorted((v, e) for v in ("off", "max", "cfar") for e in ("cpp", "python"))
    scene = json.loads((out / "timing_popoto_12km_cpp.json").read_text())
    assert [r["variant"] for r in scene] == ["off", "max", "cfar"] and all(r["scene"] == "popoto_12km" for r in scene)
    tone_t = json.loads((out / "timing_popoto_12km_doppler.json").read_text())
    assert [r["variant"] for r in tone_t] == ["off", "tone"]
    for name in ("tone", "falsealarm", "timing"):
        assert "exit 0" in (out / f"{name}.log").read_text()
    # D5: the second pass on the same cached pings, the tracker taking the tone's rate; the classes ping by ping in both.
    rate = json.loads((out / "doppler_tone_eval_rate.json").read_text())
    assert rate["tracker_range_rate"] == "update" and rate["background"] == {} and rate["pings"] == tone["pings"]
    for res in (tone, rate):
        for s in res["engines"]["cpp"]["on"]["targets"].values():
            assert len(s["kinds"]) == res["pings"] and "first_mover_ping" in s and "velocity_error_m_s" in s


# --- the tone path's false detections attributed to the grating lobes (tools/doppler_tone_eval.py) ----------------------

def test_tone_eval_passes_the_lobe_notch_switch_and_the_ls12_run_has_both_passes():
    old = doppler_tone_eval.LOBE_NOTCH
    try:
        assert "tone_lobe_notch" not in doppler_tone_eval.chain_config("cpp", {"tone": "bank"}, 3.0)["doppler"]     # as before
        doppler_tone_eval.LOBE_NOTCH = "off"
        assert doppler_tone_eval.chain_config("cpp", {"tone": "bank"}, 3.0)["doppler"]["tone_lobe_notch"] == "off"
        assert "doppler" not in doppler_tone_eval.chain_config("cpp", None, 3.0)
    finally:
        doppler_tone_eval.LOBE_NOTCH = old
    runs = (ROOT / "tools" / "doppler_gpu_runs.sh").read_text()
    assert "--tone-lobe-notch off" in runs and "ls12_tone_train_lobe_off.json" in runs and "ls12_tone_train.json" in runs


def test_tone_eval_passes_the_tracker_range_rate_switch_and_the_runs_have_the_second_pass():
    """D5: --tracker-range-rate reaches the chain with the tone path on only; not given, the configuration is as before; the
    tone and ls12 sub-runs each take a second pass on the same cache with it (before -> after of the classes)."""
    old = doppler_tone_eval.TRACKER_RATE
    try:
        assert "tracker" not in doppler_tone_eval.chain_config("cpp", {"tone": "bank"}, 3.0)          # as before
        doppler_tone_eval.TRACKER_RATE = "update"
        assert doppler_tone_eval.chain_config("cpp", {"tone": "bank"}, 3.0)["tracker"] == {"range_rate": "update"}
        assert "tracker" not in doppler_tone_eval.chain_config("cpp", None, 3.0)
    finally:
        doppler_tone_eval.TRACKER_RATE = old
    runs = (ROOT / "tools" / "doppler_gpu_runs.sh").read_text()
    for name in ("doppler_tone_eval_rate.json", "ls12_tone_train_rate.json", "ls12_tone_alternate12_rate.json"):
        assert name in runs
    assert runs.count("--tracker-range-rate update") == 4          # the header's line and the three calls


def test_the_attribution_kinds_on_a_made_geometry():
    """A beam at -36 degrees with its own ridge +0.88 (notch 0.38) and a lobe's ridge -1.43 (notch 0.38), one at 0 with
    none; step 0.06 m/s. With the lobe notch off a rate on the lobe's ridge is `lobe_ridge`; with it applied, inside by
    more than half a step, `DEFECT` (and the summary says so first); a rate just inside the edge (a cell outside, its
    parabola inward) is `lobe_edge`; at the own notch's edge `notch_edge`; neither `other`."""
    lobe = {"beam": 0, "direction": 1.25, "ownship": -1.43, "notch": 0.38, "endfire": False, "response_db": 0.0}
    geo = {"bearings_deg": np.array([-36.0, 0.0]), "ownship": [0.88, 0.0], "notch": [0.38, 0.38], "lobes": {0: [lobe]},
           "step": 0.06, "lobes_applied": False, "sim_scale": 0.815}
    det = lambda b, r: {"bearing_deg": b, "range_m": 900.0, "range_rate_m_s": r, "snr_db": 20.0, "doppler_path": "tone"}  # noqa: E731
    on = doppler_tone_eval.attribute(det(-35.8, -1.40), geo)
    assert on["kind"] == "lobe_ridge" and not on["defect"] and on["visible_lobe"] and not on["endfire_notch"] and on["lobe_ridge_widths"] < 0.1
    scaled = doppler_tone_eval.attribute(det(-36.0, -1.43 * 0.815), geo)
    assert scaled["sim_scaled_lobe_widths"] == pytest.approx(0.0, abs=1e-9)
    edge = doppler_tone_eval.attribute(det(0.2, 0.55), geo)
    assert edge["kind"] == "notch_edge" and edge["visible_lobe"] is False and edge["own_ridge_widths"] == pytest.approx(0.55 / 0.38)
    other = doppler_tone_eval.attribute(det(0.0, 1.5), geo)
    assert other["kind"] == "other" and other["lobe_ridge_widths"] is None
    sweep = doppler_tone_eval.attribute(dict(det(-36.0, None), doppler_path="sweep"), geo)
    assert sweep["kind"] == "no_rate" and sweep["visible_lobe"] is True
    line = doppler_tone_eval.attribution_summary([on, edge, other, sweep], "x")
    assert line.startswith("x: 2 of 4 beyond the lobe onset; 1 on a lobe's ridge; 0 at a lobe notch's edge; 1 at the own notch's edge")
    assert "the rest 1: 0.0 deg +1.50 m/s" in line
    applied = dict(geo, lobes_applied=True)
    bad = doppler_tone_eval.attribute(det(-35.8, -1.40), applied)
    assert bad["defect"] and bad["kind"] == "DEFECT"
    rim = doppler_tone_eval.attribute(det(-35.8, -1.43 + 0.38 - 0.02), applied)     # 0.02 inside: under half a step
    assert not rim["defect"] and rim["kind"] == "lobe_edge"
    own_bad = doppler_tone_eval.attribute(det(-36.0, 0.88 + 0.1), geo)               # the own notch is always applied
    assert own_bad["kind"] == "DEFECT"
    assert doppler_tone_eval.attribution_summary([bad, rim], "y").startswith("y: DEFECT: 1 inside an applied notch; ")


def test_the_attribution_on_the_lobe_record_explains_every_false_detection_with_the_notch_off():
    """test_doppler_tone_lobes.py's still field (6.5 kHz, under way): with the lobe notch off every false tone detection is
    on a lobe's ridge, on a beam beyond the onset; with it there is none to attribute. The geometry is the auto plan's
    either way."""
    import test_doppler_tone_lobes as lobes
    rec = lobes.ping(field=2000)
    rows = {}
    old = doppler_tone_eval.LOBE_NOTCH
    try:
        for notch in ("off", "auto"):
            doppler_tone_eval.LOBE_NOTCH = notch
            with LiveChainFor(notch) as chain:
                rep = chain.process(rec.copy())
            geo = doppler_tone_eval.lobe_geometry(rec, {"tone": "bank"}, 3.0)
            assert geo["lobes_applied"] == (notch == "auto")
            rows[notch] = [doppler_tone_eval.attribute(d, geo) for d in rep["detections"]]
    finally:
        doppler_tone_eval.LOBE_NOTCH = old
    assert len(rows["off"]) >= 1 and all(r["kind"] == "lobe_ridge" and (r["visible_lobe"] or r["endfire_notch"]) for r in rows["off"]), rows["off"]
    assert rows["auto"] == []
    assert doppler_tone_eval.lobe_geometry(rec, None, 3.0) is None


def LiveChainFor(notch):
    from gpusonar.live import LiveChain
    return LiveChain({"engine": "cpp", "backend": "host", "doppler": {"tone": "bank", "report": True, "tone_lobe_notch": notch}})


# --- tools/doppler_bank_eval.py's opt-in switches and the hfm12 sub-run (DOPPLER.md "HFM") --------------------------------

QUICK = ["--set", "fidelity=free_field", "--set", "reverberation=[]"]
HFM = ["--set", "sonar_overrides.transmit.waveform.type=sampled", "--set", "sonar_overrides.transmit.waveform.source=hfm"]
OLD_KEYS = {"scene", "pings", "channels", "max_range_rate_m_s", "combine", "truth_at", "gate_range_m", "engines", "background"}


def test_bank_eval_takes_the_backends():
    old = doppler_bank_eval.BACKEND
    try:
        assert doppler_bank_eval.chain_config("cpp", True, 3.0)["backend"] == "host"           # the default, as before
        doppler_bank_eval.BACKEND = "cuda"
        assert doppler_bank_eval.chain_config("cpp", False, 3.0)["backend"] == "cuda"
    finally:
        doppler_bank_eval.BACKEND = old


def test_bank_eval_without_the_new_switches_writes_what_it_wrote(tmp_path, monkeypatch):
    """No new switch: the scene as loaded (here made free field by the loader, so that a CPU runs it in seconds), simulated by
    run_scenario as before, and the result's keys the old ones."""
    real = doppler_bank_eval.load_scenario_config

    def free_field(name, expand=False):
        cfg = real(name, expand=expand)
        cfg.update(fidelity="free_field", reverberation=[])
        return cfg
    monkeypatch.setattr(doppler_bank_eval, "load_scenario_config", free_field)
    out = tmp_path / "plain.json"
    assert doppler_bank_eval.main(["popoto_12km_large_submerged", "--pings", "1", "--background", "0", "--engines", "cpp", "--max-rate", "0.3",
                                   "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert set(res) == OLD_KEYS and set(res["engines"]["cpp"]) == {"off", "on", "summary"}


def test_bank_eval_changes_the_transmit_with_set_and_runs_an_hfm_without_the_bank(tmp_path):
    """--set makes the transmit an HFM (a sampled pulse: type Arbitrary in the record) and --off-only runs the chain without
    the bank (which refuses it); the result names the changes, the backends and the transmit; the cache names carry the
    changes (the LFM's records are not reused for the HFM)."""
    out, cache = tmp_path / "hfm.json", tmp_path / "cache"
    argv = ["popoto_12km_large_submerged", "--pings", "1", "--background", "0", "--engines", "cpp", "--truth-at", "reflection",
            "--gate-range-m", "250", "--cache", str(cache), "--json", str(out)]
    assert doppler_bank_eval.main(argv + QUICK + HFM + ["--off-only"]) == 0
    res = json.loads(out.read_text())
    assert res["set"] == ["fidelity=free_field", "reverberation=[]", "sonar_overrides.transmit.waveform.type=sampled",
                          "sonar_overrides.transmit.waveform.source=hfm"]
    assert res["off_only"] and res["backend"] == "host" and res["sim_backend"] == "host"
    assert res["transmit"]["type"] == "Arbitrary" and res["transmit"]["duration"] == 1.0
    assert set(res["engines"]["cpp"]) == {"off", "summary"} and set(res["engines"]["cpp"]["summary"]) == {"closing", "oblique", "crossing"}
    assert doppler_bank_eval.main(argv + QUICK + ["--set", "sonar_overrides.transmit.waveform.duration=4.0", "--off-only"]) == 0
    lfm = json.loads(out.read_text())
    assert lfm["transmit"]["type"] == "Lfm" and lfm["transmit"]["duration"] == 4.0
    assert len(list(cache.glob("*.h5"))) == 2
    with pytest.raises(ValueError, match="linear sweep"):           # the bank refuses an HFM: --off-only is needed
        doppler_bank_eval.main(argv + QUICK + HFM + ["--max-rate", "0.3"])


def test_the_hfm12_sub_run_in_its_quick_host_form(tmp_path):
    """GPUSONAR_DOPPLER_RUNS=hfm12 with GPUSONAR_DOPPLER_QUICK=1 on the host (one free-field ping each, the bank to 0.3 m/s): the
    six calls in their order, each JSON with its transmit (1 s and 4 s; LFM and HFM; the `eq` runs at 190.98 dB rms, the
    1 s pulse's energy). About a minute on this CPU."""
    out = tmp_path / "doppler"
    env = dict(os.environ, GPUSONAR_DOPPLER_RUNS="hfm12", GPUSONAR_DOPPLER_QUICK="1", GPUSONAR_DOPPLER_BACKEND="host", OMP_NUM_THREADS="1",
               PYTHON=sys.executable)
    env.pop("GPUSONAR_UPLOAD_REPO", None)
    got = subprocess.run(["bash", str(ROOT / "tools" / "doppler_gpu_runs.sh"), str(out)], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=1200)
    assert got.returncode == 0, got.stdout[-4000:] + got.stderr[-2000:]
    assert "exit 0" in (out / "hfm12.log").read_text()
    want = {"hfm1_plain": ("Arbitrary", 1.0, True), "hfm4_plain": ("Arbitrary", 4.0, True), "hfm4eq_plain": ("Arbitrary", 4.0, True),
            "lfm4eq_plain": ("Lfm", 4.0, True), "lfm1_cfar": ("Lfm", 1.0, False), "lfm4_cfar": ("Lfm", 4.0, False)}
    peak = {}
    for name, (kind, dur, off_only) in want.items():
        res = json.loads((out / f"hfm12_{name}.json").read_text())
        assert res["transmit"]["type"] == kind and res["transmit"]["duration"] == dur and res["off_only"] == off_only, name
        assert set(res["engines"]["cpp"]) == ({"off", "summary"} if off_only else {"off", "on", "summary"}), name
        assert res["combine"] == ("max" if off_only else "cfar") and res["truth_at"] == "reflection" and res["gate_range_m"] == 250.0
        peak[name] = res["transmit"]["source_level_db_peak"]
    # The energy bookkeeping: the eq runs 10 log10 4 under the shipped level (same energy as 1 s), the others at it.
    assert peak["hfm4eq_plain"] == pytest.approx(peak["hfm1_plain"] - 10 * np.log10(4.0), abs=0.01)
    assert peak["lfm4eq_plain"] == pytest.approx(peak["lfm4_cfar"] - 10 * np.log10(4.0), abs=0.01) and peak["hfm4_plain"] == peak["lfm1_cfar"]
