#!/usr/bin/env python3
"""Pictures of a scenario: its eigenrays and bounces, the boundary and medium models in force, the receive beams,
and what each hydrophone receives (docs/tasks/VISUALIZATION.md, packages V1 and V2). Opt-in: nothing else runs it.

    PYTHONPATH=python tools/scenario_view.py sst_echo_level flank_target_1km --out DIR [--wav 0] [--html]
    tools/scenario_view.py --all --out DIR [--quick]      # every scenario and DIR/index.html

Per scenario it writes DIR/<name>.png (the 2-D page), DIR/<name>.trace.json (the scene trace, its per-channel
delays rounded), with `--wav C` DIR/<name>.ch<C>.wav (hydrophone C as sound, viz/audio.py) and with `--html`
DIR/<name>.html (the 3-D page; `--three PATH` embeds a local three.module.min.js so that the page works offline).
One ping is simulated with the scenario's own settings: the first, or the middle of a synthetic-aperture track
(`viz/catalogue.py` `view_plan`; the page says which). The simulator's own stage taps (taps.sim_capture) are kept in
a temporary directory for the page. `--quick` thins the densest scenes as tools/gallery.py --quick does (the page
says so). `--all` also writes DIR/index.html and DIR/index.json (family, tier, size and time of each page); one
scenario failing does not stop it: its page says what failed. The `eigenray` tiers need Bellhop (GPUSONAR_BELLHOP;
Bellhop3D for `eigenray3d`), run as separate processes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _run_keeping_provider(name, overrides, engine):
    """run_scenario, keeping the provider `make_provider` returned for the run (read only: the function is wrapped
    to remember its result, nothing it does changes), so that the trace uses the run's provider and its caches."""
    from gpusonar.sim import simulator
    from gpusonar.sim.simulator import run_scenario
    kept = []
    original = simulator.make_provider

    def remember(*args, **kwargs):
        p = original(*args, **kwargs)
        kept.append(p)
        return p
    simulator.make_provider = remember
    try:
        result = run_scenario(name, overrides=overrides or None, engine=engine)
    finally:
        simulator.make_provider = original
    return result, (kept[0] if kept else None)


def _rounded(trace: dict) -> dict:
    """The trace with per-channel delays rounded to 10 ns and polylines to 1 mm (a smaller file; nothing drawn moves)."""
    t = dict(trace)
    t["pairs"] = [dict(p, channel_delay_s=[round(v, 8) for v in p["channel_delay_s"]]) for p in trace["pairs"]]
    legs = []
    for leg in trace["legs"]:
        legs.append({k: ([dict(r, polyline=None if r["polyline"] is None else [[round(v, 3) for v in q] for q in r["polyline"]])
                          for r in v] if k in ("out", "back") else v) for k, v in leg.items()})
    t["legs"] = legs
    return t


def view(name: str, out: str, wav: int | None, html_page: bool, three: str | None, engine: str | None,
         quick: bool = False, all_rays: bool = False, truecolour: bool = False) -> dict:
    from gpusonar import taps
    from gpusonar.config import load_scenario_config
    from gpusonar.viz import audio, catalogue, page2d, page3d
    from gpusonar.viz import trace as tr

    t0 = time.perf_counter()
    raw = load_scenario_config(name, None, expand=False)
    plan = catalogue.view_plan(raw, quick=quick)
    with tempfile.TemporaryDirectory(prefix="gpusonar_view_") as tmp:
        with taps.sim_capture(tmp):
            result, base = _run_keeping_provider(name, plan["overrides"], engine)
        stages = taps.read_sim_taps(tmp, 0)
    t_sim = time.perf_counter() - t0
    trace = tr.build_trace(result, engine=engine, base_provider=base, view=plan)
    path_json = os.path.join(out, f"{name}.trace.json")
    tr.write_trace(_rounded(trace), path_json)
    t_trace = time.perf_counter() - t0 - t_sim
    path_png = os.path.join(out, f"{name}.png")
    drawn = page2d.draw_page(trace, result.records[0], path_png, stages=stages, all_rays=all_rays,
                               truecolour=truecolour)
    family, short = catalogue.scenario_family(name)
    rays = [r for leg in trace["legs"] for w in ("out", "back") for r in leg[w]]
    facts = {"scenario": name, "family": family, "family_short": short, "tier": trace["tier"],
             "mode": trace["sonar"]["mode"], "note": plan["note"], "targets": len(trace["targets"]),
             "eigenrays": len(rays), "eigenrays_drawn": sum(r["polyline"] is not None for r in rays),
             "oneway_sources": len(trace["oneway"]), "direct": trace["direct"] is not None,
             "missing": trace["missing"], "simulate_s": round(t_sim, 2), "trace_s": round(t_trace, 2),
             "page_s": round(time.perf_counter() - t0 - t_sim - t_trace, 2),
             "png_bytes": os.path.getsize(path_png), "trace_bytes": os.path.getsize(path_json), **drawn}
    if wav is not None:
        fs, y, how = audio.hydrophone_audio(result.records[0], wav)
        facts["wav"] = dict(audio.write_wav(os.path.join(out, f"{name}.ch{wav}.wav"), fs, y), **how)
    if html_page:
        facts["html"] = page3d.write_page(trace, os.path.join(out, f"{name}.html"), three=three)
    return facts


def _child(conn, call):
    try:
        conn.send(("ok", view(*call)))
    except Exception as exc:                          # sent back to the parent, which reports it
        conn.send(("error", f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=3)))
    conn.close()


def _with_timeout(call, seconds: float) -> dict:
    """view(*call) in a child process, stopped after `seconds` (TimeoutError): a scene too slow for this CPU."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, call))
    proc.start()
    child.close()
    if not parent.poll(seconds):
        proc.kill()
        proc.join()
        raise TimeoutError(f"no page after {seconds:g} s on this CPU (simulation, trace and page of one ping)")
    status, value = parent.recv()
    proc.join()
    if status == "error":
        raise RuntimeError(value)
    return value


def failure_page(name: str, out: str, error: str) -> None:
    """A page that says exactly what failed, in place of a silent omission."""
    import matplotlib
    matplotlib.use("Agg")
    import textwrap

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4))
    lines = [w for line in error.splitlines() for w in (textwrap.wrap(line, 150) or [""])]
    fig.text(0.02, 0.95, f"{name}: no picture.\n\n" + "\n".join(lines[:24]), va="top", fontsize=9, family="monospace")
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=72)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios", nargs="*")
    ap.add_argument("--all", action="store_true", help="every file under configs/scenarios, and DIR/index.html")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav", type=int, default=None, help="write this hydrophone as a WAV file")
    ap.add_argument("--html", action="store_true", help="write the 3-D page")
    ap.add_argument("--three", default=None, help="a local three.module.min.js to embed in the 3-D page (offline)")
    ap.add_argument("--engine", default=None, choices=("cpp", "numpy"))
    ap.add_argument("--quick", action="store_true", help="thin the densest scenes as tools/gallery.py --quick does")
    ap.add_argument("--all-rays", action="store_true", help="colour every eigenray, not only the strongest of each leg")
    ap.add_argument("--truecolour", action="store_true", help="true-colour PNG (about 3 times larger) instead of the fixed palette")
    ap.add_argument("--timeout", type=float, default=None, help="give up on a scenario after this many seconds (its page says so)")
    ap.add_argument("--skip-existing", action="store_true", help="reuse DIR/<name>.facts.json of scenarios already done")
    ap.add_argument("--index", action="store_true", help="write DIR/index.html for the pages named (always with --all)")
    args = ap.parse_args(argv)
    names = list(args.scenarios)
    if args.all:
        root = os.path.join(ROOT, "configs", "scenarios")
        names = sorted(f[:-5] for f in os.listdir(root) if f.endswith(".yaml"))
    if not names:
        ap.error("name a scenario or give --all")
    os.makedirs(args.out, exist_ok=True)
    rows = []
    t_all = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        facts_path = os.path.join(args.out, f"{name}.facts.json")
        if args.skip_existing and os.path.exists(facts_path):
            with open(facts_path, encoding="utf-8") as f:
                facts = json.load(f)
            rows.append(facts)
            continue
        try:
            call = (name, args.out, args.wav, args.html, args.three, args.engine, args.quick, args.all_rays, args.truecolour)
            facts = _with_timeout(call, args.timeout) if args.timeout else view(*call)
        except Exception as exc:                      # one scenario failing does not stop --all; it is reported
            from gpusonar.viz import catalogue
            err = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TimeoutError):
                err += (f". Listed for the GPU job (CUDA synthesis): `PYTHONPATH=python python3 tools/scenario_view.py {name} "
                        "--out DIR`, or run it here without --timeout")
            failure_page(name, args.out, err + ("" if isinstance(exc, TimeoutError) else "\n" + traceback.format_exc(limit=3)))
            fam, short = catalogue.scenario_family(name)
            facts = {"scenario": name, "family": fam, "family_short": short, "error": err,
                     "seconds": round(time.perf_counter() - t0, 2)}
        with open(facts_path, "w", encoding="utf-8") as f:
            json.dump(facts, f)
        rows.append(facts)
        print(json.dumps(facts), flush=True)
    if args.all or args.index:
        from gpusonar.viz import index
        index.write_index(rows, args.out, total_s=time.perf_counter() - t_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
