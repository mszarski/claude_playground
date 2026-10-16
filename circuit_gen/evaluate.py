"""
Multi-sample benchmark runs.

Each board is designed `samples` times. Model outputs vary between samples,
so a single run says little; the per-board pass rate and the stage where
failing samples stopped are what to compare between models and prompts.

Samples run in separate processes (KiCad's pcbnew is not thread-safe), each
writing to <out>/<board>/sNN/ with its own log.txt.
"""

import contextlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .pipeline import SIM_STAGE, STAGES, build


def _is_infra_error(e):
    """API or network failures: the sample says nothing about the model's design ability."""
    names = {t.__name__ for t in type(e).__mro__}
    return bool(names & {"HfHubHTTPError", "APIConnectionError", "APIStatusError",
                         "ConnectionError", "Timeout", "TimeoutError", "ReadError",
                         "RemoteProtocolError", "ConnectError", "TimeoutException",
                         "NetworkError"})


def first_failure(report):
    """The stage where a run stopped, or None if it succeeded."""
    if report.get("success"):
        return None
    if report.get("infra"):
        return "infra"
    if report.get("generation") is False:
        return "generation"
    required = STAGES + ([SIM_STAGE] if report.get("functionally_tested") else [])
    for stage in required:
        if not report.get("stages", {}).get(stage, {}).get("ok"):
            return stage
    return "unknown"


def _run_one(case, sample, out_dir, opts):
    """Design (or build) one sample. Runs in a worker process."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(out_dir / "log.txt", "w") as log, contextlib.redirect_stdout(log):
        try:
            if "code" in case:
                report = build(case["code"], out_dir, max_passes=opts["passes"], tests=case.get("tests"))
            else:
                report = design_and_build(case["description"], out_dir, opts, tests=case.get("tests"))
        except Exception as e:  # one broken sample must not stop the benchmark
            print(f"crashed: {type(e).__name__}: {e}")
            report = {"success": False, "generation": False, "stages": {},
                      "error": f"{type(e).__name__}: {e}", "infra": _is_infra_error(e)}
    # The model said its design was done (its checks passed) but the hidden tests failed.
    sim = report.get("stages", {}).get(SIM_STAGE)
    false_accept = bool(report.get("generation")) and sim is not None and not sim.get("ok")
    result = {"id": case["id"], "sample": sample, "success": bool(report.get("success")),
            "false_accept": false_accept,
            "failed_at": first_failure(report), "tested": bool(case.get("tests")),
            "seconds": round(time.time() - start, 1), "dir": str(out_dir)}
    (out_dir / "result.json").write_text(json.dumps(result))
    return result


def design_and_build(description, out_dir, opts, tests=None):
    """Generate a circuit with the model, then run the build pipeline on it."""
    from . import config
    from .backends import make_backend
    from .generate import generate_circuit

    out_dir = Path(out_dir)
    name = opts.get("backend") or config.BACKEND
    extra = {"effort": opts["effort"]} if name == "claude" and opts.get("effort") else {}
    spec_extra = dict(extra)
    if opts.get("spec_provider"):
        # Spec writer on Inference Providers, e.g. while the designer runs on a local server.
        spec_extra.update(provider=opts["spec_provider"], base_url="", max_tokens=16384)
    spec_backend = (make_backend(name, model=opts["spec_model"], **spec_extra)
                    if opts.get("spec_model") else None)
    if name == "hf" and opts.get("thinking") is not None:
        extra["thinking"] = opts["thinking"]
    backend = make_backend(name, model=opts.get("model"), **extra)
    gen = generate_circuit(description, out_dir, backend=backend, max_turns=opts["max_turns"],
                           spec_first=opts.get("spec_first", False), spec_backend=spec_backend)
    (out_dir / "generation.json").write_text(json.dumps({
        "description": description, "backend": backend.name, "model": backend.model,
        "spec_model": spec_backend.model if spec_backend else None,
        "passed_checks": gen.passed, "turns": gen.turns,
        "attempts": len(gen.attempts), "error": gen.error, "summary": gen.summary,
        "spec_revisions": gen.spec_revisions,
    }, indent=2))
    print(f"generation: passed_checks={gen.passed} attempts={len(gen.attempts)} "
          f"turns={gen.turns} error={gen.error}")
    if gen.code is None:
        return {"success": False, "stages": {}, "generation": False}
    report = build(gen.code, out_dir, max_passes=opts["passes"], tests=tests)
    report["generation"] = gen.passed
    report["success"] = report["success"] and gen.passed
    return report


def summarize(cases, results, samples):
    boards = []
    for case in cases:
        all_runs = sorted((r for r in results if r["id"] == case["id"]), key=lambda r: r["sample"])
        infra = sum(r["failed_at"] == "infra" for r in all_runs)
        runs = [r for r in all_runs if r["failed_at"] != "infra"]
        passed = sum(r["success"] for r in runs)
        failures = {}
        for r in runs:
            if r["failed_at"]:
                failures[r["failed_at"]] = failures.get(r["failed_at"], 0) + 1
        false_accepts = sum(r.get("false_accept", False) for r in runs)
        boards.append({"id": case["id"], "tested": bool(case.get("tests")), "passed": passed,
                       "false_accepts": false_accepts,
                       "samples": len(runs), "pass_rate": passed / len(runs) if runs else 0.0,
                       "failures": failures, "infra_errors": infra})
    tested = [b for b in boards if b["tested"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "samples_per_board": samples,
        "pass_rate": mean([b["pass_rate"] for b in boards]),
        "pass_rate_tested": mean([b["pass_rate"] for b in tested]),
        "solved_at_least_once": sum(b["passed"] > 0 for b in boards),
        "solved_every_time": sum(b["passed"] == b["samples"] for b in boards),
        "infra_errors": sum(b["infra_errors"] for b in boards),
        "false_accepts": sum(b.get("false_accepts", 0) for b in boards),
        "boards": boards,
    }


def format_table(summary):
    lines = ["| board | sim tests | passed | where failures stopped |", "|---|---|---|---|"]
    for b in summary["boards"]:
        fails = ", ".join(f"{k} ×{v}" for k, v in sorted(b["failures"].items())) or "—"
        lines.append(f"| {b['id']} | {'yes' if b['tested'] else 'no'} | "
                     f"{b['passed']}/{b['samples']} | {fails} |")
    n = len(summary["boards"])
    lines += ["",
              f"Mean pass rate: {summary['pass_rate']:.0%} over all boards, "
              f"{summary['pass_rate_tested']:.0%} over boards with simulation tests.",
              f"Solved at least once: {summary['solved_at_least_once']}/{n}; "
              f"every time: {summary['solved_every_time']}/{n}."]
    lines.append(f"False accepts (model declared done, hidden tests failed): "
                 f"{summary.get('false_accepts', 0)}.")
    if summary.get("infra_errors"):
        lines.append(f"{summary['infra_errors']} sample(s) excluded: API or network errors.")
    return "\n".join(lines)


def run_eval(cases, out, samples=1, jobs=4, resume=False, **opts):
    """Run every case `samples` times with `jobs` worker processes. Returns the summary.

    With `resume`, samples that already finished (have a result.json) are reused.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    opts = {"passes": 30, "max_turns": 40, **opts}
    work = [(case, s) for case in cases for s in range(1, samples + 1)]
    results = []
    if resume:
        done = []
        for case, s in work:
            f = out / case["id"] / f"s{s:02d}" / "result.json"
            if f.exists():
                results.append(json.loads(f.read_text()))
                done.append((case["id"], s))
        work = [(c, s) for c, s in work if (c["id"], s) not in done]
        print(f"resuming: {len(done)} samples already done, {len(work)} to run", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = {pool.submit(_run_one, case, s, out / case["id"] / f"s{s:02d}", opts): (case, s)
                   for case, s in work}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            status = "pass" if r["success"] else f"FAIL at {r['failed_at']}"
            print(f"[{i}/{len(work)}] {r['id']} s{r['sample']:02d}: {status} ({r['seconds']}s)",
                  flush=True)

    summary = summarize(cases, results, samples)
    summary["runs"] = sorted(results, key=lambda r: (r["id"], r["sample"]))
    summary["options"] = {k: v for k, v in opts.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(format_table(summary) + "\n")
    return summary


def compare(run_dirs):
    """Markdown table of per-board pass rates across several eval runs (e.g. models)."""
    summaries = []
    for d in run_dirs:
        s = json.loads((Path(d) / "summary.json").read_text())
        label = (s.get("options") or {}).get("model") or Path(d).name
        summaries.append((label, s))
    labels = [label for label, _ in summaries]
    if len(set(labels)) < len(labels):  # e.g. one model run with different options
        summaries = [(Path(d).name, s) for d, (_, s) in zip(run_dirs, summaries)]
    boards = []
    for _, s in summaries:
        for b in s["boards"]:
            if b["id"] not in boards:
                boards.append(b["id"])
    head = "| board | " + " | ".join(label for label, _ in summaries) + " |"
    lines = [head, "|---" * (len(summaries) + 1) + "|"]
    for bid in boards:
        cells = []
        for _, s in summaries:
            b = next((b for b in s["boards"] if b["id"] == bid), None)
            cells.append(f"{b['passed']}/{b['samples']}" if b else "—")
        lines.append(f"| {bid} | " + " | ".join(cells) + " |")
    totals = []
    for _, s in summaries:
        passed = sum(b["passed"] for b in s["boards"])
        total = sum(b["samples"] for b in s["boards"])
        totals.append(f"**{passed}/{total}**" if total else "—")
    lines.append("| **all** | " + " | ".join(totals) + " |")
    return "\n".join(lines)
