#!/usr/bin/env python3
"""Validate and save one eval-v2.2 repair decision.

Usage: python3 save_item.py <id>     (reads proposals/<id>.json, writes results/<id>.json)
       python3 save_item.py <id> --try "<patch>"   (grade any patch against the proposal's checks)
Proposal JSON: {"verdict": "keep"|"fix"|"drop", "reason": "...", "criteria": [...], "checks": [...], "allowed": [...],
                "alternatives": ["<another valid patch>", ...]}
For keep/fix: the reference and every alternative must pass all checks with a perfect score, the unedited
design must fail at least one check, and every alternative must apply cleanly. "checks" may be [] only if every
criterion is genuinely subjective (then the item is judge-only)."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from design_edit.checks import grade_checks, eval_check
from design_edit.schema import parse, apply_patch, strip_fences

HERE = Path.cwd()  # run from the work directory holding items/, proposals/, results/
i = sys.argv[1]
item = json.load(open(HERE / "items" / f"{i}.json"))
p = json.load(open(HERE / "proposals" / f"{i}.json"))
if "--try" in sys.argv:
    patch = sys.argv[sys.argv.index("--try") + 1]
    print(grade_checks(item["before"], patch, p["checks"], p["allowed"])); sys.exit()
v = p.get("verdict")
assert v in ("keep", "fix", "drop"), "verdict must be keep/fix/drop"
assert p.get("reason"), "reason required"
if v != "drop":
    crit, checks, allowed = p.get("criteria"), p.get("checks"), p.get("allowed")
    assert isinstance(crit, list) and crit and all(isinstance(c, str) for c in crit), "criteria: non-empty list of str"
    assert isinstance(checks, list) and isinstance(allowed, list), "checks and allowed must be lists"
    if checks:
        errs = []
        for name, patch in [("reference", item["reference_patch"])] + [(f"alternative {k}", a) for k, a in enumerate(p.get("alternatives", []))]:
            try:
                apply_patch(parse(item["before"]), strip_fences(patch))
            except Exception as e:
                errs.append(f"{name} does not apply: {e}"); continue
            g = grade_checks(item["before"], patch, checks, allowed)
            if not g["pass"]:  # lint is not part of the eval pass; it ignores the item's design system
                errs.append(f"{name} fails checks: {g['why']}")
        before = parse(item["before"])
        if not any(eval_check(c, before, before) is False for c in checks):
            errs.append("the unedited design passes every check (do-nothing would pass)")
        if errs:
            print("REJECTED:\n" + "\n".join(errs)); sys.exit(1)
json.dump({"id": int(i), **p}, open(HERE / "results" / f"{i}.json", "w"), indent=1)
print(f"saved {i}: {v}")
