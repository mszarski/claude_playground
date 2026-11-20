"""The resumable studies (tools/ssp_range_study.py, fjord_rehearsal.py, decay_time_sensitivity.py) reuse what their JSON
holds, recompute everything with `--fresh`, and say so in their help; decay_time_sensitivity's stored runs are keyed by the
settings they ran with. The measuring step is replaced by a stub that counts its calls: nothing is simulated, and every
file goes to pytest's temporary directory (audit A12). Also: tools/gallery.py --quick does not default to docs/figures
(A13), and tools/popoto_truth_check.py reads the ship's track from the scenario (G5)."""
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Counter:
    def __init__(self, result):
        self.calls, self.result = [], result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result(len(self.calls))


def _measure(n):
    return {"pings": 1, "pings_detected": 1, "median_snr_db": float(n), "per_ping": [{"detected": True, "snr_db": float(n)}],
            "track_confirmed_at_ping": None, "simulate_seconds": 0.0}


@pytest.mark.parametrize("name", ["ssp_range_study", "fjord_rehearsal", "decay_time_sensitivity"])
def test_help_says_it_resumes_and_offers_fresh(name):
    out = subprocess.run([sys.executable, str(ROOT / "tools" / f"{name}.py"), "--help"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "RESUMES FROM THE STORED JSON" in out.stdout and "is NOT recomputed" in out.stdout and "--fresh" in out.stdout


def test_decay_time_sensitivity_resumes_keys_by_settings_and_recomputes_with_fresh(tmp_path):
    dts = tool("decay_time_sensitivity")
    store = tmp_path / "decay.json"
    stub = Counter(_measure)
    args = dict(options=["facet"], log=lambda *_: None, measure=stub)
    res = dts.run(["popoto_12km:image"], 1, 1, store, taps=2, **args)
    assert list(res) == ["popoto_12km, image-source paths (1 pings, 2 taps)"]
    assert set(res["popoto_12km, image-source paths (1 pings, 2 taps)"]) == {"as shipped", "facet / seed 0"}
    assert len(stub.calls) == 2 and json.loads(store.read_text()) == res
    assert stub.calls[1][0][0]["time_spread"]["taps"] == 2 and stub.calls[1][0][0]["pings"]["count"] == 1

    dts.run(["popoto_12km:image"], 1, 1, store, taps=2, **args)                # the same settings: nothing is recomputed
    assert len(stub.calls) == 2
    for changed in (dict(taps=3), dict(taps=2, pings=2)):                       # another setting is another case, computed
        kw = dict(dict(taps=2, pings=1), **changed)
        before = len(stub.calls)
        res = dts.run(["popoto_12km:image"], 1, kw["pings"], store, taps=kw["taps"], **args)
        assert len(stub.calls) == before + 2
    assert len(res) == 3

    # --max-arrivals is in the key of a Bellhop case (the stub stands in for the simulation, so no Bellhop is needed)
    a = dts.case_label("popoto_env_gravel_30m", dts.scenario("popoto_env_gravel_30m", 3), 12, 3)
    b = dts.case_label("popoto_env_gravel_30m", dts.scenario("popoto_env_gravel_30m", 3), 8, 3)
    assert a != b and "12 strongest arrivals" in a

    before = len(stub.calls)
    res = dts.run(["popoto_12km:image"], 1, 1, store, taps=2, fresh=True, **args)   # --fresh: everything again, the rest dropped
    assert len(stub.calls) == before + 2 and list(res) == ["popoto_12km, image-source paths (1 pings, 2 taps)"]
    assert json.loads(store.read_text()) == res


def test_decay_time_sensitivity_keys_reproduce_the_committed_labels():
    """The committed JSON's cases name the settings they ran with; the generated key is the same text, so a rerun with
    those settings resumes them and any other setting does not."""
    dts = tool("decay_time_sensitivity")
    stored = json.loads((ROOT / "docs" / "figures" / "decay_time_sensitivity.json").read_text())
    assert dts.case_label("popoto_12km:image", dts.scenario("popoto_12km:image", 6), None, 6) in stored
    for case in ("popoto_12km", "popoto_env_gravel_30m", "popoto_env_summer_mud_60m", "popoto_env_winter_sand_200m"):
        assert dts.case_label(case, dts.scenario(case, 3), 12, 3) in stored, case


def test_ssp_range_study_resumes_and_fresh_recomputes(tmp_path):
    srs = tool("ssp_range_study")
    store = tmp_path / "ssp.json"
    stub = Counter(lambda n: {"n": n, "simulate_seconds": 0.0, "as deployed": {"per_ping": [{"detected": False}]}})
    name = next(iter(srs.SCENARIOS))
    res = srs.run([name], store, log=lambda *_: None, run_one=stub)
    assert len(stub.calls) == len(srs.WATERS) and set(res[name]) == set(srs.WATERS)
    srs.run([name], store, log=lambda *_: None, run_one=stub)
    assert len(stub.calls) == len(srs.WATERS)
    res = srs.run([name], store, log=lambda *_: None, run_one=stub, fresh=True)
    assert len(stub.calls) == 2 * len(srs.WATERS) and res[name][srs.WATERS[0]]["n"] == len(srs.WATERS) + 1
    assert json.loads(store.read_text()) == res


def test_fjord_rehearsal_resumes_and_fresh_recomputes(tmp_path):
    fr = tool("fjord_rehearsal")
    store = tmp_path / "fjord.json"
    stub = Counter(lambda n: {"error": f"stub {n}"})
    res = fr.run(["testauv_1row"], ["open water"], [], store, log=lambda *_: None, run_one=stub)
    assert len(stub.calls) == 1 and res == {"testauv_1row": {"open water": {"error": "stub 1"}}}
    fr.run(["testauv_1row"], ["open water"], [], store, log=lambda *_: None, run_one=stub)
    assert len(stub.calls) == 1
    store.write_text(json.dumps({"static_2row": {"open water": {"error": "old"}}, "testauv_1row": {"open water": {"error": "old"}}}))
    res = fr.run(["testauv_1row"], ["open water"], [], store, log=lambda *_: None, run_one=stub, fresh=True)
    assert len(stub.calls) == 2 and res == {"testauv_1row": {"open water": {"error": "stub 2"}}}
    assert json.loads(store.read_text()) == res


def test_gallery_quick_does_not_default_to_the_committed_gallery():
    pytest.importorskip("matplotlib", reason="the gallery tool draws its figures with matplotlib")
    gallery = tool("gallery")
    quick = gallery.output_dir(None, True).resolve()
    assert quick == ROOT / "build" / "gallery_quick" and not quick.is_relative_to(ROOT / "docs")
    assert gallery.output_dir(None, False) == pathlib.Path("docs/figures")
    assert gallery.output_dir("x", True) == pathlib.Path("x")
    readme = (ROOT / "README.md").read_text()
    assert "tools/gallery.py --backend host --quick " in readme and "build/gallery_quick" in readme


def test_truth_check_reads_the_ship_track_from_the_scenario():
    """What the script hard-coded before: the ship's five points at x 740..860 m, y 12 km, moving at -6 m/s, pings 19.9 s apart."""
    from gpusonar.config import load_scenario_config
    tc = tool("popoto_truth_check")
    cfg = load_scenario_config("popoto_12km")
    for k in range(6):
        ship = tc.ship_at(cfg, k)
        np.testing.assert_allclose(ship[:, 0], np.array([740.0, 770.0, 800.0, 830.0, 860.0]) - 6.0 * 19.9 * k)
        np.testing.assert_allclose(ship[:, 1], 12000.0)
