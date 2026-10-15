"""
Command line interface.

    python -m circuit_gen design "5V to 3.3V regulator board" -o runs/ldo
    python -m circuit_gen design "..." --backend hf --model Qwen/Qwen3.8-27B -o runs/x
    python -m circuit_gen build circuit_gen/examples/blinker_555.py -o runs/blinker
    python -m circuit_gen eval circuit_gen/benchmarks/boards.json -o runs/eval
    python -m circuit_gen eval --examples -o runs/examples
"""

import argparse
import json
import sys
from pathlib import Path

from .pipeline import build


def _print_report(report):
    stages = report["stages"]
    for name, stage in stages.items():
        status = "ok" if stage.get("ok") else "FAILED"
        detail = {k: v for k, v in stage.items() if k != "ok" and v not in (None, [], "")}
        print(f"  {name:10s} {status:6s} {json.dumps(detail)[:300]}")
    print(f"  success: {report['success']}  ({report['seconds']}s)")


def cmd_build(args):
    report = build(Path(args.circuit).read_text(), args.out, max_passes=args.passes)
    _print_report(report)
    return 0 if report["success"] else 1


def _design(description, out_dir, args):
    from . import config
    from .backends import make_backend
    from .generate import generate_circuit
    out_dir = Path(out_dir)
    name = args.backend or config.BACKEND
    extra = {"effort": args.effort} if name == "claude" else {}
    backend = make_backend(name, model=args.model, **extra)
    gen = generate_circuit(description, out_dir, backend=backend, max_turns=args.max_turns)
    (out_dir / "generation.json").write_text(json.dumps({
        "description": description, "backend": backend.name, "model": backend.model,
        "passed_checks": gen.passed, "turns": gen.turns,
        "attempts": len(gen.attempts), "error": gen.error, "summary": gen.summary,
    }, indent=2))
    print(f"generation: passed_checks={gen.passed} attempts={len(gen.attempts)} "
          f"turns={gen.turns} error={gen.error}")
    if gen.code is None:
        return {"success": False, "stages": {}, "generation": False}
    report = build(gen.code, out_dir, max_passes=args.passes)
    report["generation"] = gen.passed
    report["success"] = report["success"] and gen.passed
    _print_report(report)
    return report


def cmd_design(args):
    report = _design(args.description, args.out, args)
    return 0 if report["success"] else 1


def cmd_eval(args):
    out = Path(args.out)
    if args.examples:
        examples = sorted((Path(__file__).parent / "examples").glob("*.py"))
        cases = [{"id": p.stem, "code": p.read_text()} for p in examples]
    else:
        cases = json.loads(Path(args.benchmark).read_text())
    results = []
    for case in cases:
        print(f"== {case['id']}")
        case_dir = out / case["id"]
        if "code" in case:
            report = build(case["code"], case_dir, max_passes=args.passes)
            _print_report(report)
        else:
            report = _design(case["description"], case_dir, args)
        results.append({"id": case["id"], "success": report["success"],
                        "completed": report.get("completed", [])})
    passed = sum(r["success"] for r in results)
    summary = {"passed": passed, "total": len(results), "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{passed}/{len(results)} boards made it to DRC-clean Gerbers")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="circuit_gen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="lay out and route an existing SKiDL circuit file")
    b.add_argument("circuit")

    d = sub.add_parser("design", help="generate a circuit from a description, then build it")
    d.add_argument("description")

    e = sub.add_parser("eval", help="run a benchmark of descriptions (or the examples)")
    e.add_argument("benchmark", nargs="?", default=str(Path(__file__).parent / "benchmarks" / "boards.json"))
    e.add_argument("--examples", action="store_true", help="build the hand-written examples instead")

    for s in (b, d, e):
        s.add_argument("-o", "--out", required=True, help="output directory")
        s.add_argument("--passes", type=int, default=30, help="max FreeRouting passes")
    for s in (d, e):
        s.add_argument("--backend", default=None, choices=["hf", "claude"],
                       help="hf: open model via Hugging Face Inference Providers (default); claude: Anthropic API")
        s.add_argument("--model", default=None,
                       help="model id, e.g. Qwen/Qwen3.8-27B (hf) or claude-opus-5-5 (claude)")
        s.add_argument("--max-turns", type=int, default=40)
        s.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                       help="Claude backend only")

    args = p.parse_args(argv)
    return {"build": cmd_build, "design": cmd_design, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
