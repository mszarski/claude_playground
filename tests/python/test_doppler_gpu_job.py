"""The GPU job's Doppler mode (tools/gpu_job.sh GPUSONAR_JOB=doppler, tools/doppler_gpu_runs.sh) and the switches it uses:
`--backend` / `--sim-backend` of tools/doppler_tone_eval.py, `--at-default` of tools/doppler_bank_threshold.py, and
`--doppler-combine`, `--doppler-tone`, `--scene`, `--backend`, `--json` of tools/live_chain_timing.py. Each switch is
opt-in: without it a tool does what it did. The whole mode in its quick host form (GPUSONAR_DOPPLER_QUICK=1,
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


def test_the_attribution_kinds_on_a_made_geometry():
    """A beam at -36 degrees with its own ridge +0.88 (notch 0.38) and a lobe's ridge -1.43 (notch 0.38), one at 0 with none."""
    geo = {"bearings_deg": np.array([-36.0, 0.0]), "ownship": [0.88, 0.0], "notch": [0.38, 0.38],
           "lobes": {0: [{"beam": 0, "direction": 1.25, "ownship": -1.43, "notch": 0.38, "endfire": False, "response_db": 0.0}]}}
    det = lambda b, r: {"bearing_deg": b, "range_m": 900.0, "range_rate_m_s": r, "snr_db": 20.0, "doppler_path": "tone"}  # noqa: E731
    on = doppler_tone_eval.attribute(det(-35.8, -1.40), geo)
    assert on["kind"] == "lobe_ridge" and on["visible_lobe"] and not on["endfire_notch"] and on["lobe_ridge_widths"] < 0.1
    edge = doppler_tone_eval.attribute(det(0.2, 0.55), geo)
    assert edge["kind"] == "notch_edge" and edge["visible_lobe"] is False and edge["own_ridge_widths"] == pytest.approx(0.55 / 0.38)
    other = doppler_tone_eval.attribute(det(0.0, 1.5), geo)
    assert other["kind"] == "other" and other["lobe_ridge_widths"] is None
    sweep = doppler_tone_eval.attribute(dict(det(-36.0, None), doppler_path="sweep"), geo)
    assert sweep["kind"] == "no_rate" and sweep["visible_lobe"] is True
    line = doppler_tone_eval.attribution_summary([on, edge, other, sweep], "x")
    assert line.startswith("x: 2 of 4 beyond the lobe onset; 1 on a lobe's ridge; 1 at the own notch's edge") and "the rest 1: 0.0 deg +1.50 m/s" in line


def test_the_attribution_on_the_lobe_record_explains_every_false_detection_with_the_notch_off():
    """test_doppler_tone_lobes.py's still field (6.5 kHz, under way): with the lobe notch off every false tone detection is
    on a lobe's ridge, on a beam beyond the onset; with it there is none to attribute. The geometry is the auto plan's
    either way."""
    import test_doppler_tone_lobes as lobes
    rec = lobes.ping(field=2000)
    rows = {}
    for notch in ("off", "auto"):
        with LiveChainFor(notch) as chain:
            rep = chain.process(rec.copy())
        geo = doppler_tone_eval.lobe_geometry(rec, {"tone": "bank"}, 3.0)
        rows[notch] = [doppler_tone_eval.attribute(d, geo) for d in rep["detections"]]
    assert len(rows["off"]) >= 1 and all(r["kind"] == "lobe_ridge" and (r["visible_lobe"] or r["endfire_notch"]) for r in rows["off"]), rows["off"]
    assert rows["auto"] == []
    assert doppler_tone_eval.lobe_geometry(rec, None, 3.0) is None


def LiveChainFor(notch):
    from gpusonar.live import LiveChain
    return LiveChain({"engine": "cpp", "backend": "host", "doppler": {"tone": "bank", "report": True, "tone_lobe_notch": notch}})
