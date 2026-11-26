#!/usr/bin/env python3
"""Split tests/python into shards by test file, so the GPU suite can run as several jobs at once.

    tools/test_shards.py I/N          print shard I of N (1-based): its test files, one a line
    tools/test_shards.py --table N    every shard's weight and file count

Every test file is in exactly one shard, whatever N. Files are weighed by their measured seconds in
tests/python/shard_weights.json (written from a suite's `--durations` lines by `--weights LOG`), or, for a file not
listed there, by the median listed weight (no file: by its count of `def test_`). The heaviest file goes first, each
to the lightest shard so far; ties go by name, so the split depends only on the tree. One shard of one is every file.
"""
import argparse
import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests" / "python"
WEIGHTS = TESTS / "shard_weights.json"


def files(tests=TESTS):
    return sorted(p.name for p in tests.glob("test_*.py"))


def weights(names, tests=TESTS, table=None):
    if table is None:
        table = json.loads(WEIGHTS.read_text())["seconds"] if WEIGHTS.exists() else {}
    known = [table[n] for n in names if n in table]
    if known:
        fallback = statistics.median(known)
        return {n: float(table.get(n, fallback)) for n in names}
    return {n: float(max(1, len(re.findall(r"^\s*def test_", (tests / n).read_text(), re.M)))) for n in names}


def split(n, names=None, w=None):
    """The N shards: lists of file names. Heaviest first, each to the lightest shard so far."""
    if n < 1:
        raise ValueError(f"the number of shards must be at least 1, not {n}")
    names = files() if names is None else list(names)
    w = weights(names) if w is None else w
    shards, load = [[] for _ in range(n)], [0.0] * n
    for name in sorted(names, key=lambda x: (-w[x], x)):
        k = min(range(n), key=lambda i: (load[i], i))
        shards[k].append(name)
        load[k] += w[name]
    return [sorted(s) for s in shards]


def parse(spec):
    m = re.fullmatch(r"(\d+)/(\d+)", spec)
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise ValueError(f"a shard is I/N with 1 <= I <= N, not {spec!r}")
    return int(m.group(1)), int(m.group(2))


def seconds_from_log(text):
    """Seconds a test file from pytest's `--durations` lines ('12.34s call tests/python/test_x.py::name')."""
    out = {}
    for sec, name in re.findall(r"^\s*([0-9.]+)s (?:call|setup|teardown)\s+tests/python/(test_\w+\.py)::", text, re.M):
        out[name] = out.get(name, 0.0) + float(sec)
    return {k: round(v, 1) for k, v in sorted(out.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("shard", nargs="?", help="I/N")
    ap.add_argument("--table", type=int, metavar="N", help="every shard's weight and file count")
    ap.add_argument("--weights", metavar="LOG", action="append", help="write shard_weights.json from these suite logs' --durations lines")
    a = ap.parse_args(argv)
    if a.weights:
        sec = {}
        for log in a.weights:
            for k, v in seconds_from_log(pathlib.Path(log).read_text(errors="replace")).items():
                sec[k] = round(sec.get(k, 0.0) + v, 1)
        if not sec:
            ap.error("no --durations lines in the logs")
        WEIGHTS.write_text(json.dumps({"what": "seconds a test file in the GPU suite (pytest --durations=0 --durations-min=0.5; tools/test_shards.py --weights)",
                                       "seconds": dict(sorted(sec.items()))}, indent=1) + "\n")
        print(f"{len(sec)} files, {sum(sec.values()):.0f} s")
        return 0
    if a.table:
        w = weights(files())
        for i, s in enumerate(split(a.table), 1):
            print(f"{i}/{a.table}: {len(s)} files, weight {sum(w[x] for x in s):.0f}")
        return 0
    if not a.shard:
        ap.error("give I/N, --table N or --weights LOG")
    try:
        i, n = parse(a.shard)
    except ValueError as e:
        ap.error(str(e))
    print("\n".join(f"tests/python/{x}" for x in split(n)[i - 1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
