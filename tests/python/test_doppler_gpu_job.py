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
