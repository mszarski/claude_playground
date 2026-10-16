"""
Command line interface.

    python -m circuit_gen design "5V to 3.3V regulator board" -o runs/ldo
    python -m circuit_gen design "..." --backend hf --model Qwen/Qwen3.8-27B -o runs/x
    python -m circuit_gen build circuit_gen/examples/blinker_555.py -o runs/blinker
    python -m circuit_gen eval circuit_gen/benchmarks/boards.json -o runs/eval --samples 5
    python -m circuit_gen eval --examples -o runs/examples
    python -m circuit_gen compare runs/qwen runs/glm
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


def _opts(args):
    return {"backend": args.backend, "model": args.model, "effort": args.effort,
            "max_turns": args.max_turns, "passes": args.passes, "spec_first": args.spec_first,
            "spec_model": args.spec_model, "thinking": False if args.no_thinking else None}


def cmd_design(args):
    from .evaluate import design_and_build
    report = design_and_build(args.description, args.out, _opts(args))
    if report.get("stages"):
        _print_report(report)
    return 0 if report["success"] else 1


def cmd_eval(args):
    from .evaluate import format_table, run_eval
    if args.examples:
        examples = sorted((Path(__file__).parent / "examples").glob("*.py"))
        cases = [{"id": p.stem, "code": p.read_text()} for p in examples]
    else:
        cases = json.loads(Path(args.benchmark).read_text())
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    summary = run_eval(cases, args.out, samples=args.samples, jobs=args.jobs,
                       resume=args.resume, **_opts(args))
    print()
    print(format_table(summary))
    return 0


def cmd_datagen(args):
    from .backends import make_backend
    from .datagen import generate_descriptions
    backend = make_backend(args.backend, model=args.model)
    out = Path(args.out)
    previous = json.loads(out.read_text()) if args.append and out.exists() else []
    boards = generate_descriptions(n_per_topic=args.per_topic, backend=backend,
                                   also_exclude=previous)
    out.write_text(json.dumps(previous + boards, indent=1))
    print(f"wrote {len(boards)} new board descriptions to {args.out} ({len(previous) + len(boards)} total)")
    return 0


def cmd_export(args):
    from .datagen import export_dataset
    stats = export_dataset(args.runs, args.out, include_benchmark=args.include_benchmark)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_procgen(args):
    from .procgen import build_dataset
    stats = build_dataset(args.n, args.out, seed=args.seed, jobs=args.jobs)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_compare(args):
    from .evaluate import compare
    print(compare(args.runs))
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
    e.add_argument("--samples", type=int, default=1, help="designs per board (model outputs vary)")
    e.add_argument("--jobs", type=int, default=4, help="samples run in parallel")
    e.add_argument("--only", default=None, help="comma-separated board ids to run")
    e.add_argument("--resume", action="store_true", help="reuse samples already finished in --out")

    g = sub.add_parser("datagen", help="write new (non-benchmark) board descriptions for training")
    g.add_argument("-o", "--out", required=True, help="output JSON file of boards")
    g.add_argument("--per-topic", type=int, default=4)
    g.add_argument("--append", action="store_true", help="add to the boards already in --out")
    g.add_argument("--backend", default=None, choices=["hf", "claude"])
    g.add_argument("--model", default=None)

    x = sub.add_parser("export", help="SFT dataset (JSONL) from successful eval runs")
    x.add_argument("runs", nargs="+", help="eval output directories")
    x.add_argument("-o", "--out", required=True, help="output directory")
    x.add_argument("--include-benchmark", action="store_true",
                   help="also export benchmark boards (they are held out by default)")

    pg = sub.add_parser("procgen", help="programmatic training set: parametric boards, verified, no LLM")
    pg.add_argument("-n", type=int, default=600, help="boards to generate")
    pg.add_argument("-o", "--out", required=True)
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--jobs", type=int, default=4)

    c = sub.add_parser("compare", help="side-by-side pass rates of several eval runs")
    c.add_argument("runs", nargs="+", help="eval output directories")

    for s in (b, d, e):
        s.add_argument("-o", "--out", required=True, help="output directory")
        s.add_argument("--passes", type=int, default=30, help="max FreeRouting passes")
    for s in (d, e):
        s.add_argument("--backend", default=None, choices=["hf", "claude"],
                       help="hf: open model via Hugging Face Inference Providers (default); claude: Anthropic API")
        s.add_argument("--model", default=None,
                       help="model id, e.g. Qwen/Qwen3.8-27B (hf) or claude-opus-5-5 (claude)")
        s.add_argument("--max-turns", type=int, default=40)
        s.add_argument("--spec-first", action="store_true",
                       help="write frozen requirements from the description before designing")
        s.add_argument("--spec-model", default=None,
                       help="model that writes the spec (default: the designer's model)")
        s.add_argument("--no-thinking", action="store_true",
                       help="hf backend: turn off the designer's thinking mode (Qwen3-style models)")
        s.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                       help="Claude backend only")

    args = p.parse_args(argv)
    return {"build": cmd_build, "design": cmd_design, "eval": cmd_eval,
            "compare": cmd_compare, "datagen": cmd_datagen, "export": cmd_export,
            "procgen": cmd_procgen}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
