#!/usr/bin/env python3
"""Regenerate the committed outputs D-251 made stale (issue sonar-rar0), on the device, under the same file names.

    tools/regen_stale.py run --parts bank,d3,ls,baseline,p12km --out DIR [--backend cuda] [--cache DIR] [--quick]
    tools/regen_stale.py merge --out DIR [--tree HASH]      # DIR/<the committed file names>, from the committed files and DIR/regen/*.json
    tools/regen_stale.py --list                             # what is stale, what is not, and why (nothing run)

D-251 (2026-11-27) made the boundary reverberation's Doppler per frequency by default. A record moves only where the platform
moves AND there is boundary reverberation (D-251: scenes with a still platform or without boundary reverberation are
bit-identical under the new default, eight scenes, both engines). So of each file only the parts from such scenes are stale;
`STALE` lists them with the command that made them (from the file itself or docs/tasks/DOPPLER.md) and `KEPT` the parts that
are not stale or cannot be rerun, with the reason. `run` reruns the stale parts with the simulator and the chain on `--backend`
(the tools' `--backend` / `--sim-backend`, D-241 and D-247; `--sim-backend` for tools/doppler_study.py and both for
tools/popoto_12km_study.py were added for this, opt-in) into DIR/regen/, one JSON a part; `merge` writes each committed file's
name into DIR with the stale parts replaced and a `regenerated` block (date, tree, backend, the commands, what was replaced and
what was kept and why). Nothing under docs/ is written by either: the files are copied in by hand after they are read.

`--quick`: the plumbing on a CPU (free field, no reverberation, one ping, the smallest of each part; it measures nothing).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICK_SET = ["--set", "fidelity=free_field", "--set", "reverberation=[]"]

# file -> [(part key in the file, regen json name, command (the committed one), quick variant)]
#   the part key: "scenes.popoto_12km" is d["scenes"]["popoto_12km"]; "runs.k12_scene" is d["runs"]["k12_scene"] ({command, result})
STALE = {
    "docs/tasks/doppler_bank_eval.json": [
        ("scenes.popoto_far_mover", "bank_popoto_far_mover", "tools/doppler_bank_eval.py popoto_far_mover --background 4", None),
        ("scenes.popoto_12km", "bank_popoto_12km", "tools/doppler_bank_eval.py popoto_12km --background 4",
         "tools/doppler_bank_eval.py popoto_12km --pings 1 --background 1 --engines cpp --max-rate 0.3 " + " ".join(QUICK_SET)),
    ],
    "docs/tasks/doppler_d3_eval.json": [
        ("cfar_scenes.popoto_far_mover", "d3_cfar_popoto_far_mover", "tools/doppler_bank_eval.py popoto_far_mover --combine cfar --background 4", None),
        ("cfar_scenes.popoto_12km", "d3_cfar_popoto_12km", "tools/doppler_bank_eval.py popoto_12km --combine cfar --background 4",
         "tools/doppler_bank_eval.py popoto_12km --combine cfar --pings 1 --background 1 --engines cpp --max-rate 0.3 " + " ".join(QUICK_SET)),
        ("tone_p12", "d3_tone_p12", "tools/doppler_tone_eval.py popoto_12km_doppler --background 4", None),
    ],
    "docs/tasks/doppler_large_submerged_eval.json": [
        ("bank", "ls_bank", "tools/doppler_bank_eval.py popoto_12km_large_submerged --combine cfar --max-rate 8 --truth-at reflection --gate-range-m 100",
         "tools/doppler_bank_eval.py popoto_12km_large_submerged --combine cfar --max-rate 0.3 --pings 1 --background 0 --engines cpp --truth-at reflection "
         "--gate-range-m 100 " + " ".join(QUICK_SET)),
        ("alternate", "ls_alternate", "tools/doppler_tone_eval.py popoto_12km_large_submerged --alternate 6.5e+3 --max-rate 8 --truth-at reflection --gate-range-m 100", None),
        ("train_one_ping", "ls_train_one_ping", "tools/doppler_tone_eval.py popoto_12km_large_submerged_tone --pings 1 --background 0 --max-rate 8 --truth-at reflection "
         "--gate-range-m 100", None),
    ],
    "docs/tasks/doppler_baseline.json": [
        ("runs.fp_v15", "baseline_fp_v15", "tools/doppler_study.py margin flank_patrol --waveform cw_hann --platform-speed 1.5 --target small_vehicle --radial -2 0 0.25 0.5 1 2 --pings 6 --vmax 4", None),
        ("runs.fp_v15_lfm", "baseline_fp_v15_lfm", "tools/doppler_study.py margin flank_patrol --waveform scene --platform-speed 1.5 --target small_vehicle --radial -2 0.5 2 --pings 2 --vmax 4", None),
        ("runs.fm_scene", "baseline_fm_scene", "tools/doppler_study.py margin popoto_far_mover --waveform scene --pings 2 --vmax 3", None),
        ("runs.fm_cwhann", "baseline_fm_cwhann", "tools/doppler_study.py margin popoto_far_mover --waveform cw_hann --pings 6 --vmax 3", None),
        ("runs.k12_scene", "baseline_k12_scene", "tools/doppler_study.py margin popoto_12km --waveform scene --pings 2 --vmax 3", None),
        ("runs.k12_cwhann", "baseline_k12_cwhann", "tools/doppler_study.py margin popoto_12km --waveform cw_hann --pings 6 --vmax 3", None),
        ("runs.sp_scene", "baseline_sp_scene", "tools/doppler_study.py margin sst_pursuit --waveform scene --pings 3 --vmax 25", None),
        ("runs.rs_mono", "baseline_rs_mono", "tools/doppler_study.py reverb-spectrum --speed 3", "tools/doppler_study.py reverb-spectrum --speed 3"),
        ("runs.rs_bi", "baseline_rs_bi", "tools/doppler_study.py reverb-spectrum --speed 3 --geometry bistatic", None),
    ],
    "docs/design/popoto_12km.json": [
        ("*", "popoto_12km", "tools/popoto_12km_study.py && tools/popoto_12km_study.py --classes (the file is the tool's whole output)",
         "tools/popoto_12km_study.py --quick --rows 1"),
    ],
}
PARTS = {"bank": "docs/tasks/doppler_bank_eval.json", "d3": "docs/tasks/doppler_d3_eval.json", "ls": "docs/tasks/doppler_large_submerged_eval.json",
         "baseline": "docs/tasks/doppler_baseline.json", "p12km": "docs/design/popoto_12km.json"}
KEPT = {
    "docs/tasks/doppler_bank_eval.json": {"scenes.flank_patrol": "platform still: not stale (D-251: bit-identical)"},
    "docs/tasks/doppler_d3_eval.json": {"threshold_noise": "noise only: no reverberation, not stale",
                                        "cfar_scenes.flank_patrol": "platform still: not stale",
                                        "tone_alt": "flank_patrol, platform still: not stale",
                                        "tone_p12_wholetrain": "made before 51d2b12 and 80df69d (the chain as it was then): not reproducible at this tree; historical, kept as it was",
                                        "default_off": "hashes between two trees, not a simulation result"},
    "docs/tasks/doppler_large_submerged_eval.json": {"bank_first_run_ping_truth_summary": "the tool's old truth (LS.1 item 7): historical, kept",
                                                     "tone_false_detections_beyond_32_deg": "read from the CPU runs' alternate and train JSON: superseded by GL.8, GL.9 and D-252 (the device runs), kept as it was",
                                                     "planner": "the depth planner, no simulation: not stale", "tone_notch": "the chain's plan, no simulation: not stale"},
    "docs/tasks/doppler_baseline.json": {"runs.fp_scene": "flank_patrol, platform still: not stale", "runs.fp_cwhann": "platform still: not stale",
                                         "runs.fp_cw": "platform still: not stale", "runs.fp_sweep": "platform still: not stale",
                                         "runs.rs_surf": "--speed 0: platform still: not stale"},
    "docs/design/popoto_12km.json": {"cases.the same without reverberation (noise only)": "no reverberation: not stale (rerun with the rest: the tool writes the whole file)"},
}
GALLERY_QUICK = ("docs/handoff/examples/gallery_quick_fls_120k_transit.json: CPU-cheap, regenerated in the tree (tools/gallery.py --quick ...; "
                 "docs/handoff/EXAMPLES.md), not here")


def _get(d: dict, key: str):
    for k in key.split("."):
        d = d[k]
    return d


def _put(d: dict, key: str, value) -> None:
    *path, last = key.split(".")
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def commands(part: str, out: Path, backend: str, cache: Path, quick: bool) -> list:
    """(regen json name, argv) of one part."""
    py = os.environ.get("PYTHON", sys.executable)
    res = []
    for key, name, cmd, qcmd in STALE[PARTS[part]]:
        if quick and qcmd is None:
            continue
        base = shlex.split(qcmd if quick else cmd.split(" && ")[0] if key == "*" else cmd)
        dest = str(out / "regen" / f"{name}.json")
        tool = base[0]
        if tool.endswith(("doppler_bank_eval.py", "doppler_tone_eval.py")):
            argv = [py, *base, "--backend", backend, "--sim-backend", backend, "--cache", str(cache), "--json", dest]
            res.append((name, argv))
        elif tool.endswith("doppler_study.py"):
            res.append((name, [py, *base, "--sim-backend", "cuda" if backend == "cuda" else "host", "--json", dest]))
        elif tool.endswith("popoto_12km_study.py"):
            out_file = out / "regen" / "popoto_12km.json"
            first = [py, *base, "--backend", backend, "--sim-backend", "cuda" if backend == "cuda" else "host", "--out", str(out_file)]
            res.append((name, first))
            if not quick:
                res.append((name + "_classes", [py, base[0], "--classes", "--backend", backend, "--sim-backend", "cuda" if backend == "cuda" else "host", "--out", str(out_file)]))
    return res


def cmd_run(a) -> int:
    out = Path(a.out)
    (out / "regen").mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache) if a.cache else out / "cache"
    if "p12km" in a.parts and not (out / "regen" / "popoto_12km.json").exists():
        # the study keeps the half of the file it does not rerun: start from the committed one
        (out / "regen" / "popoto_12km.json").write_text((ROOT / PARTS["p12km"]).read_text())
    rc = 0
    for part in a.parts:
        for name, argv in commands(part, out, a.backend, cache, a.quick):
            print(f"=== regen {part} {name}: {' '.join(shlex.quote(x) for x in argv)}", flush=True)
            got = subprocess.run(argv, cwd=ROOT).returncode
            print(f"=== regen {part} {name}: exit {got}", flush=True)
            rc = rc or got
    return rc


def cmd_merge(a) -> int:
    out = Path(a.out)
    regen = out / "regen"
    for path, items in STALE.items():
        committed = json.loads((ROOT / path).read_text())
        replaced, missing = [], []
        for key, name, cmd, _ in items:
            f = regen / f"{name}.json"
            if not f.exists():
                missing.append(key)
                continue
            new = json.loads(f.read_text())
            if key == "*":
                committed = new
            elif key.startswith("runs."):
                _put(committed, key, {"command": f"python3 {cmd} --sim-backend {a.backend}", "result": new})
            elif key.startswith("cfar_scenes."):
                _put(committed, key, {k: new[k] for k in ("engines", "background", "pings")})
            else:
                _put(committed, key, new)
            replaced.append(key)
        if not replaced:
            continue
        committed["regenerated"] = {"date": datetime.date.today().isoformat(), "why": "D-251: the boundary reverberation's Doppler is per frequency by default (issue sonar-rar0)",
                                    "tree": a.tree, "backend": a.backend, "replaced": replaced, "not_regenerated": missing,
                                    "commands": {k: c for k, _, c, _ in items}, "kept": KEPT.get(path, {})}
        (out / Path(path).name).write_text(json.dumps(committed, indent=1, default=float) + "\n")
        print(f"{out / Path(path).name}: replaced {replaced}" + (f"; not regenerated {missing}" if missing else ""))
    return 0


def cmd_list(_a) -> int:
    for path, items in STALE.items():
        print(path)
        for key, _, cmd, _ in items:
            print(f"  stale  {key}: {cmd}")
        for key, why in KEPT.get(path, {}).items():
            print(f"  kept   {key}: {why}")
    print(GALLERY_QUICK)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("run", "merge"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--parts", default=",".join(PARTS), type=lambda s: [p for p in s.split(",") if p])
    ap.add_argument("--out")
    ap.add_argument("--backend", default="cuda", choices=("host", "cuda"))
    ap.add_argument("--cache")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tree", help="the tree the run was made on (merge)")
    a = ap.parse_args(argv)
    if a.list:
        return cmd_list(a)
    if not a.cmd or not a.out:
        ap.error("give run or merge, and --out DIR (or --list)")
    bad = [p for p in a.parts if p not in PARTS]
    if bad:
        ap.error(f"unknown parts {bad}: {', '.join(PARTS)}")
    return cmd_run(a) if a.cmd == "run" else cmd_merge(a)


if __name__ == "__main__":
    sys.exit(main())
