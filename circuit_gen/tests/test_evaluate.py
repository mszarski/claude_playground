"""
Tests for multi-sample benchmark bookkeeping.
"""

from circuit_gen.evaluate import first_failure, format_table, summarize


def report(success, completed=(), generation=True, tested=False):
    stages = {s: {"ok": True} for s in completed}
    return {"success": success, "generation": generation, "functionally_tested": tested,
            "stages": stages}


class TestFirstFailure:
    def test_success(self):
        assert first_failure(report(True)) is None

    def test_generation(self):
        assert first_failure(report(False, generation=False)) == "generation"

    def test_first_missing_stage(self):
        r = report(False, completed=["circuit", "checks", "placement"])
        assert first_failure(r) == "routing"

    def test_simulation_counts_only_when_tested(self):
        layout = ["circuit", "checks", "placement", "routing", "drc", "export"]
        assert first_failure(report(False, completed=layout, tested=True)) == "simulation"


def test_infra_errors_are_excluded():
    cases = [{"id": "a"}]
    results = [{"id": "a", "sample": 1, "success": True, "failed_at": None},
               {"id": "a", "sample": 2, "success": False, "failed_at": "infra"}]
    s = summarize(cases, results, samples=2)
    assert s["pass_rate"] == 1.0 and s["infra_errors"] == 1
    assert s["boards"][0]["samples"] == 1
    assert "1 sample(s) excluded" in format_table(s)


def test_summary_rates_and_table():
    cases = [{"id": "a", "tests": [{}]}, {"id": "b"}]
    results = [
        {"id": "a", "sample": 1, "success": True, "failed_at": None},
        {"id": "a", "sample": 2, "success": False, "failed_at": "simulation"},
        {"id": "b", "sample": 1, "success": True, "failed_at": None},
        {"id": "b", "sample": 2, "success": True, "failed_at": None},
    ]
    s = summarize(cases, results, samples=2)
    assert s["pass_rate"] == 0.75
    assert s["pass_rate_tested"] == 0.5
    assert (s["solved_at_least_once"], s["solved_every_time"]) == (2, 1)
    assert s["boards"][0]["failures"] == {"simulation": 1}
    table = format_table(s)
    assert "| a | yes | 1/2 | simulation ×1 |" in table
    assert "Mean pass rate: 75%" in table


def test_connection_errors_are_infra():
    from circuit_gen.evaluate import _is_infra_error

    class ReadError(Exception):
        pass

    assert _is_infra_error(ReadError("Connection reset by peer"))
    assert not _is_infra_error(ValueError("bad circuit"))


def test_compare(tmp_path):
    import json
    from circuit_gen.evaluate import compare
    for name, model, passed in (("a", "org/m1", 2), ("b", "org/m2", 1)):
        d = tmp_path / name
        d.mkdir()
        d.joinpath("summary.json").write_text(json.dumps({
            "options": {"model": model},
            "boards": [{"id": "led", "passed": passed, "samples": 2}]}))
    table = compare([tmp_path / "a", tmp_path / "b"])
    assert "| board | org/m1 | org/m2 |" in table
    assert "| led | 2/2 | 1/2 |" in table and "**2/2**" in table


def test_compare_same_model_uses_run_names(tmp_path):
    import json
    from circuit_gen.evaluate import compare
    for name in ("base", "base_nothink"):
        d = tmp_path / name
        d.mkdir()
        d.joinpath("summary.json").write_text(json.dumps({
            "options": {"model": "org/m1"},
            "boards": [{"id": "led", "passed": 1, "samples": 1}]}))
    assert "| board | base | base_nothink |" in compare([tmp_path / "base", tmp_path / "base_nothink"])
