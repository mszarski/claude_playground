"""tools/gpu_benchmark_all.py: one cheap group in quick mode on whatever backend the build has.
The table and the JSON must have their shape on a host-only build (CUDA column "unavailable"),
and a case that raises must be recorded, not end the run."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import gpu_benchmark_all as gba  # noqa: E402

import gpusonar as gs  # noqa: E402


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("benchall")
    md, js = out / "table.md", out / "result.json"
    assert gba.main(["--quick", "--only", "detect", "--backend", "host", "--out", str(md), "--json", str(js)]) == 0
    return md.read_text(), json.loads(js.read_text())


def test_table_and_json_shape(quick_run):
    text, result = quick_run
    head = result["header"]
    assert head["groups"] == ["detect"] and head["quick"] and head["repeats"] == 1
    assert head["host_threads"] >= 1 and head["cuda_used"] is False and head["gpus"] == []
    assert head["cuda_available"] == bool(gs.cuda_available())
    cases = result["cases"]
    assert {c["case"] for c in cases} == {"CFAR", "doa_gate (sbl)", "doa_passive (sbl)"}
    assert [h["case"] for h in result["host_only"]] == ["doa_gate (conventional)"] and result["host_only"][0]["host_s"] > 0.0
    for c in cases:
        assert c["group"] == "detect" and c["host_error"] is None, c
        assert c["host_s"] > 0.0 and len(c["host_runs"]) == 1
        assert c["cuda_s"] is None and c["cuda_error"] == "unavailable" and c["speedup"] is None and c["agreement"] is None
        assert c["sizes"] and all(isinstance(v, (int, float, str)) for v in c["sizes"].values())
    assert "| Group | Case | What | Sizes | Host, s | CUDA, s | Speed-up | Host-CUDA rel. rms | Note |" in text
    rows = [line for line in text.splitlines() if line.startswith("| detect |")]
    assert len(rows) == len(cases) and all(line.count("|") == 10 and "| unavailable |" in line for line in rows)
    assert "## Host only: stages without a CUDA build" in text and "GPU: none" in text


def test_a_failing_case_is_recorded_and_the_run_goes_on():
    args = type("A", (), {"quick": True, "scale": 1.0, "repeats": 1, "budget": 5.0, "backend": "host", "traceback": False,
                          "groups": ["detect"]})()
    b = gba.Bench(args)
    b.group = "detect"

    def broken(_, backend):
        raise RuntimeError("kernel | fell over")

    bad = b.case("broken", "raises", {"n": 1}, broken)
    good = b.case("fine", "runs", {"n": 3}, lambda _, backend: np.ones(3))
    b.guard("set-up", lambda: 1 / 0)
    assert bad["host_s"] is None and "RuntimeError: kernel | fell over" in bad["host_error"]
    assert good["host_s"] > 0.0 and good["host_error"] is None
    assert b.cases[2]["note"] == "set-up failed" and "ZeroDivisionError" in b.cases[2]["host_error"]
    text = gba.markdown(b, gba.header(b, 0.0))
    assert "error: RuntimeError: kernel \\| fell over" in text and "3 cases with a CUDA backend" in text and "2 with an error" in text


def test_two_backends_give_a_speed_up_and_an_agreement_number():
    """The comparison path, with a stand-in for the device: the same NumPy on both `backends`,
    and one case whose `device` result is wrong."""
    args = type("A", (), {"quick": True, "scale": 1.0, "repeats": 2, "budget": 5.0, "backend": "host", "traceback": False,
                          "groups": ["imaging"]})()
    b = gba.Bench(args)
    b.cuda, b.group = True, "imaging"
    x = np.arange(1.0, 9.0)
    same = b.case("same", "identical on both", {"n": 8}, lambda _, backend: x.copy())
    wrong = b.case("wrong", "device result 1 % off", {"n": 8}, lambda _, backend: x * (1.01 if backend == gba.CUDA else 1.0))
    assert same["agreement"] == 0.0 and same["speedup"] > 0.0 and same["cuda_error"] is None and len(same["cuda_runs"]) == 2
    assert wrong["agreement"] == pytest.approx(0.01)
    text = gba.markdown(b, gba.header(b, 0.0))
    assert "| 1.0e-02 |" in text and "unavailable" not in text


def test_agreement_number_and_timing_rules():
    a = [np.ones(4), np.zeros((2, 2), dtype=np.complex64)]
    assert gba.relative_rms(a, a) == 0.0
    assert gba.relative_rms([np.ones(4)], [np.ones(4) * 1.1]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        gba.relative_rms([np.ones(4)], [np.ones(5)])
    calls = []
    m, out = gba.measure(lambda _: calls.append(1) or len(calls), None, repeats=3, budget=60.0)
    assert len(calls) == 4 and len(m["runs"]) == 3 and m["warmup_seconds"] is not None      # warm-up + three counted
    m, _ = gba.measure(lambda _: gba.Timed(2.5, "x"), None, repeats=2, budget=60.0)
    assert m["seconds"] == 2.5                                                              # a case's own per-ping time
    timer = gba.StageTimer()
    holder = type("H", (), {"inner": staticmethod(lambda: sum(range(20000))), "outer": staticmethod(lambda: None)})
    holder.outer = staticmethod(lambda: holder.inner())
    timer.patch(holder, "inner", "inner")
    timer.patch(holder, "outer", "outer")
    holder.outer()
    timer.restore()
    assert timer.seconds["inner"] > 0.0 and timer.seconds["outer"] >= 0.0 and timer.seconds["outer"] < timer.seconds["inner"] + 1.0
