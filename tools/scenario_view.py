#!/usr/bin/env python3
"""Pictures of a scenario: its eigenrays and bounces, the models in force, and what each hydrophone receives
(docs/tasks/VISUALIZATION.md; prototype of packages V1 and V4). Opt-in: nothing else runs it.

    PYTHONPATH=python tools/scenario_view.py sst_echo_level flank_target_1km --out DIR [--wav 0] [--html]
    tools/scenario_view.py --all --out DIR            # every scenario (slow: each is simulated once)

Per scenario it writes DIR/<name>.trace.json (the scene trace), DIR/<name>.png (the 2-D page), with `--wav C`
DIR/<name>.ch<C>.wav (hydrophone C as sound, viz/audio.py) and with `--html` DIR/<name>.html (the 3-D page;
`--three PATH` embeds a local three.module.min.js so that the page works offline). The first ping only; a scenario
is simulated with its own settings (one ping) and the simulator's own stage taps (taps.sim_capture) are kept in a
temporary directory for the page. The `eigenray` tier needs Bellhop (GPUSONAR_BELLHOP), run as a separate process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))


def view(name: str, out: str, wav: int | None, html_page: bool, three: str | None, engine: str | None) -> dict:
    from gpusonar import taps
    from gpusonar.sim.simulator import run_scenario
    from gpusonar.viz import audio, page2d, page3d
    from gpusonar.viz import trace as tr

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gpusonar_view_") as tmp:
        with taps.sim_capture(tmp):
            result = run_scenario(name, overrides={"pings": {"count": 1}}, engine=engine)
        stages = taps.read_sim_taps(tmp, 0)
    t_sim = time.perf_counter() - t0
    trace = tr.build_trace(result, engine=engine)
    tr.write_trace(trace, os.path.join(out, f"{name}.trace.json"))
    t_trace = time.perf_counter() - t0 - t_sim
    drawn = page2d.draw_page(trace, result.records[0], os.path.join(out, f"{name}.png"), stages=stages)
    facts = {"scenario": name, "tier": trace["tier"], "simulate_s": round(t_sim, 2), "trace_s": round(t_trace, 2),
             "page_s": round(time.perf_counter() - t0 - t_sim - t_trace, 2), **drawn}
    if wav is not None:
        fs, y, how = audio.hydrophone_audio(result.records[0], wav)
        facts["wav"] = dict(audio.write_wav(os.path.join(out, f"{name}.ch{wav}.wav"), fs, y), **how)
    if html_page:
        facts["html"] = page3d.write_page(trace, os.path.join(out, f"{name}.html"), three=three)
    return facts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--all", action="store_true", help="every file under configs/scenarios")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav", type=int, default=None, help="write this hydrophone as a WAV file")
    ap.add_argument("--html", action="store_true", help="write the 3-D page")
    ap.add_argument("--three", default=None, help="a local three.module.min.js to embed in the 3-D page (offline)")
    ap.add_argument("--engine", default=None, choices=("cpp", "numpy"))
    args = ap.parse_args(argv)
    names = list(args.scenarios)
    if args.all:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "scenarios")
        names = sorted(f[:-5] for f in os.listdir(root) if f.endswith(".yaml"))
    if not names:
        ap.error("name a scenario or give --all")
    os.makedirs(args.out, exist_ok=True)
    for name in names:
        try:
            facts = view(name, args.out, args.wav, args.html, args.three, args.engine)
        except Exception as exc:                      # one scenario failing does not stop --all; it is reported
            facts = {"scenario": name, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(facts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
