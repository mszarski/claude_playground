#!/usr/bin/env python3
"""Authoring and review tool for eval v3 (hard items, written and reviewed by agents).

Run from the work directory (it holds screens/, proposals/, approved/, rejected/).

    tool.py ds <seed>                  print the design system for this seed
    tool.py screen <sid>               validate screens/<sid>.dsl (+ screens/<sid>.json {"screen", "ds_seed"})
    tool.py save <iid>                 validate proposals/<iid>.json (see GATE below); prints OK or REJECTED
    tool.py try <iid> '<patch>'        grade any patch against the proposal's checks
    tool.py approve <iid> '<note>'     reviewer: accept a validated item
    tool.py reject <iid> '<reason>'    reviewer: reject an item
    tool.py status                     counts

Proposal: {"screen_id", "kind", "request", "reference_patch", "criteria": [...], "checks": [...], "allowed": [...],
           "alternatives": ["<valid patch>", ...], "traps": [{"patch": "...", "why_wrong": "...",
           "caught_by": "checks"|"judge"}], "notes": "..."}
GATE: the reference and every alternative apply, stay in the design language, change something and pass the checks;
the unedited design fails a check; at least one trap; every trap applies; traps marked "checks" fail the checks;
at least one trap is caught by the checks; >= 1 alternative unless notes start with "SINGLE:"; the last criterion is
"Nothing unrelated to the request changes."
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from design_edit.checks import eval_check, grade_checks  # noqa: E402
from design_edit.design_system import render_design_system, sample_design_system  # noqa: E402
from design_edit.leak import dsl_violations  # noqa: E402
from design_edit.schema import apply_patch, parse, serialize, strip_fences, walk  # noqa: E402

W = Path.cwd()
for d in ("screens", "proposals", "approved", "rejected"):
    (W / d).mkdir(exist_ok=True)


def load_screen(sid):
    meta = json.loads((W / "screens" / f"{sid}.json").read_text())
    text = (W / "screens" / f"{sid}.dsl").read_text().strip()
    return meta, text


def check_screen(sid):
    meta, text = load_screen(sid)
    doc = parse(text)
    nodes = [n for n, _ in walk(doc)]
    errs = []
    if not 35 <= len(nodes) <= 120:
        errs.append(f"{len(nodes)} nodes (need 35-120)")
    if len({n["id"] for n in nodes}) != len(nodes):
        errs.append("duplicate ids")
    bad = dsl_violations({"id": "n0", "type": "frame", "props": {}, "children": []}, doc)
    if bad:
        errs.append(f"not in the design language: {bad[:3]}")
    if not {"screen", "ds_seed"} <= set(meta):
        errs.append("meta needs screen and ds_seed")
    return doc, errs, len(nodes)


def applies(before, patch):
    try:
        after = apply_patch(parse(before), strip_fences(patch))
    except Exception as e:  # noqa: BLE001
        return None, f"does not apply: {e}"
    if dsl_violations(parse(before), after):
        return None, "not in the design language"
    if serialize(after) == serialize(parse(before)):
        return None, "changes nothing"
    return after, ""


def gate(iid):
    p = json.loads((W / "proposals" / f"{iid}.json").read_text())
    errs = []
    for k in ("screen_id", "kind", "request", "reference_patch", "criteria", "checks", "allowed", "traps"):
        if k not in p:
            errs.append(f"missing {k}")
    if errs:
        return p, errs
    doc, serr, _ = check_screen(p["screen_id"])
    errs += [f"screen: {e}" for e in serr]
    before = serialize(doc)
    crit, checks, allowed = p["criteria"], p["checks"], p["allowed"]
    if not crit or "nothing unrelated" not in crit[-1].lower():
        errs.append('last criterion must be "Nothing unrelated to the request changes."')
    if not checks:
        errs.append("checks must not be empty")
    alts = p.get("alternatives") or []
    if not alts and not str(p.get("notes", "")).startswith("SINGLE:"):
        errs.append("give at least one alternative valid answer (or notes 'SINGLE: <why only one answer>')")
    for name, patch in [("reference", p["reference_patch"])] + [(f"alternative {k}", a) for k, a in enumerate(alts)]:
        after, why = applies(before, patch)
        if after is None:
            errs.append(f"{name} {why}")
            continue
        g = grade_checks(before, patch, checks, allowed)
        if not g["pass"]:
            errs.append(f"{name} fails checks: {g['why']}")
    if checks and not any(eval_check(c, doc, doc) is False for c in checks):
        errs.append("the unedited design passes every check")
    traps = p["traps"] or []
    if not traps:
        errs.append("give at least one trap (a plausible wrong answer)")
    caught = 0
    for k, t in enumerate(traps):
        after, why = applies(before, t.get("patch", ""))
        if after is None:
            errs.append(f"trap {k} {why}")
            continue
        g = grade_checks(before, t["patch"], checks, allowed)
        if t.get("caught_by", "checks") == "checks":
            if g["pass"]:
                errs.append(f"trap {k} passes the checks but is marked caught_by=checks")
            else:
                caught += 1
    if traps and not caught:
        errs.append("at least one trap must be caught by the checks")
    return p, errs


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "ds":
        print(render_design_system(sample_design_system(int(args[0]))))
    elif cmd == "screen":
        _, errs, n = check_screen(args[0])
        print("REJECTED: " + "; ".join(errs) if errs else f"OK ({n} nodes)")
    elif cmd == "try":
        p = json.loads((W / "proposals" / f"{args[0]}.json").read_text())
        _, text = load_screen(p["screen_id"])
        print(grade_checks(serialize(parse(text)), args[1], p["checks"], p["allowed"]))
    elif cmd == "save":
        _, errs = gate(args[0])
        print("REJECTED:\n" + "\n".join(errs) if errs else f"OK {args[0]}")
    elif cmd in ("approve", "reject"):
        if cmd == "approve":
            p, errs = gate(args[0])
            if errs:
                print("REJECTED (fix first):\n" + "\n".join(errs))
                return
            (W / "approved" / f"{args[0]}.json").write_text(json.dumps({**p, "review": args[1]}, indent=1))
            (W / "rejected" / f"{args[0]}.json").unlink(missing_ok=True)
        else:
            (W / "rejected" / f"{args[0]}.json").write_text(json.dumps({"reason": args[1]}))
            (W / "approved" / f"{args[0]}.json").unlink(missing_ok=True)
        print(f"{cmd}d {args[0]}")
    elif cmd == "status":
        c = {d: len(list((W / d).glob("*.json"))) for d in ("proposals", "approved", "rejected")}
        c["screens"] = len(list((W / "screens").glob("*.dsl")))
        print(c)


if __name__ == "__main__":
    main()
