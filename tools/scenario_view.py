#!/usr/bin/env python3
"""Pictures of a scenario: its eigenrays and bounces, the boundary and medium models in force, the receive beams,
and what each hydrophone receives (docs/tasks/VISUALIZATION.md, packages V1 and V2). Opt-in: nothing else runs it.

    PYTHONPATH=python tools/scenario_view.py sst_echo_level flank_target_1km --out DIR [--wav 0] [--html]
    tools/scenario_view.py --all --out DIR [--quick]      # every scenario and DIR/index.html
    tools/scenario_view.py --all --3d --traces DIR --out DIR3   # every 3-D page from the traces of a previous run

Per scenario it writes DIR/<name>.png (the 2-D page), DIR/<name>.trace.json (the scene trace, its per-channel
delays rounded), with `--wav C` DIR/<name>.ch<C>.wav (hydrophone C as sound, viz/audio.py) and with `--html`
DIR/<name>.html (`--html` or `--3d`: the 3-D page, viz/page3d.py; `--three embed`, the default when
third_party/three holds the vendored copy, embeds three.js so that the page works offline, `--three cdn` loads it
from the pinned CDN; `eigenray3d` pages carry Bellhop3D's own 3-D rays, viz/scene3d.py, unless --no-bellhop3d).
One ping is simulated with the scenario's own settings: the first, or the middle of a synthetic-aperture track
(`viz/catalogue.py` `view_plan`; the page says which). The simulator's own stage taps (taps.sim_capture) are kept in
a temporary directory for the page. `--quick` thins the densest scenes as tools/gallery.py --quick does (the page
says so). `--all` also writes DIR/index.html and DIR/index.json (family, tier, size and time of each page); one
scenario failing does not stop it: its page says what failed. The `eigenray` tiers need Bellhop (GPUSONAR_BELLHOP;
Bellhop3D for `eigenray3d`), run as separate processes.

Sound (package V3/V5, viz/audio.py, viz/arrivals.py): with `--3d` the default `--audio side` writes DIR/<name>.audio/
beside the page: the whole record of every hydrophone as a 16-bit WAV, made audible by the method its band needs (as
recorded at or under 16 kHz, time-stretched for a record shorter than 0.5 s (FLS, SAS), else heterodyned to 2 kHz;
`--audio-method` and `--stretch K` choose otherwise), the parts
(echoes, each reverberation item through the opt-in per-item taps, noise, the direct sound, one-way sources) and the
`--arrivals 8` strongest single arrivals (by the synthesis's own amplitude, over the targets) for the hydrophones `--audio-parts` (default the first
and the last), a stereo pair of those two, and the page loads them. `--audio embed` puts them in the page (capped,
viz/page3d.py), `--audio none` writes no sound. With `--traces` the ping is simulated again for its record (the trace is
not): scenes whose previous run took longer than `--sim-limit` s (default 330) get a page without sound that says so and
names the GPU-job command.
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


AUDIO_DEFAULTS = {"mode": "none", "method": "auto", "factor": None, "center": 2000.0, "parts": None, "arrivals": 8,
                  "channels": None}


def _simulate(name: str, plan: dict, engine: str | None, sound: bool):
    """One ping under the simulator's stage taps (with the opt-in per-item reverberation taps and the ping's capture
    for single arrivals when `sound`). Returns (result, base provider, stages with "__meta__", capture box or None)."""
    from gpusonar import taps
    from gpusonar.viz import arrivals
    import contextlib
    with tempfile.TemporaryDirectory(prefix="gpusonar_view_") as tmp:
        with taps.sim_capture(tmp, per_item=sound), (arrivals.capture_ping() if sound else contextlib.nullcontext()) as box:
            result, base = _run_keeping_provider(name, plan["overrides"], engine)
        stages = taps.read_sim_taps(tmp, 0)
        with open(os.path.join(tmp, "sim_taps.json"), encoding="utf-8") as f:
            stages["__meta__"] = [m for m in json.load(f) if m["ping"] == 0]
    return result, base, stages, box


def _sound(name: str, out: str, trace: dict, record, stages: dict, box, audio_opts: dict, engine: str | None) -> dict:
    """The ping's sound beside its page (viz/audio.py write_ping_audio): the whole record of every hydrophone, the
    parts and the single arrivals for the parts' hydrophones. Returns the manifest with timings."""
    import numpy as np
    from gpusonar.viz import arrivals, audio
    t0 = time.perf_counter()
    n_ch = int(np.asarray(record.samples.array).shape[0])
    pc = audio_opts.get("parts")
    pc = sorted({0, n_ch - 1}) if pc in (None, "") else (list(range(n_ch)) if pc == "all" else
                                                         [int(c) for c in str(pc).split(",") if int(c) < n_ch])
    meta = stages.get("__meta__")
    parts = audio.decompose({k: v for k, v in stages.items() if k != "__meta__"}, meta, record)
    residual = audio.residual_db(parts, record.samples.array) if parts else None
    parts = {k: (np.asarray(v)[pc], w) for k, (v, w) in parts.items()}
    arr, notes = [], []
    n_arr = int(audio_opts.get("arrivals") or 0)
    if box is not None and n_arr > 0 and "out" in box and trace.get("pairs"):
        # With the kernel's paths captured: the strongest pairs of every target by the synthesis's own amplitude (none
        # silent); else the strongest target's by the trace's level, as before.
        for p in arrivals.strongest_pairs(trace, box, None if box.get("kernel") is not None else 0, n_arr):
            if not p["mapped"]:
                notes.append(f"pair out {p['out']} back {p['back']}: not found in the run's eigenrays")
                continue
            e = arrivals.single_pair_echoes(box, p["scatterer"], p["k_captured"], p["j_captured"], engine)
            arr.append({"samples": np.asarray(e)[pc], "what": f"one arrival of {trace['targets'][p.get('target') or 0]['label']}: outbound ray "
                        f"{p['out']}, return ray {p['back']} ({p['n_surface']} surface, {p['n_bottom']} bottom bounces), "
                        "synthesised alone by the simulator (viz/arrivals.py)",
                        "out": p["out"], "back": p["back"], "delay_s": p["delay_s"], "level_db": p["level_db"],
                        "synthesis_level_db": p.get("synthesis_level_db"), "ranked_by": p.get("ranked_by"),
                        "n_surface": p["n_surface"], "n_bottom": p["n_bottom"],
                        "channel_delay_s": {str(c): p["channel_delay_s"][c] for c in pc}})
        if len(arr) < n_arr and box.get("kernel") is not None:
            notes.append(f"single arrivals: {len(arr)} listed, not {n_arr}: only {len(arr)} of the targets' pairs reach "
                         "the synthesis with a non-zero amplitude (after Lambert's law, occlusion and the beam floor)")
    elif box is not None and n_arr > 0:
        notes.append("single arrivals: none (no target with a two-way pair at this ping)")
    wc = audio_opts.get("channels")
    wc = None if wc in (None, "", "all") else [int(c) for c in str(wc).split(",") if int(c) < n_ch]
    m = audio.write_ping_audio(out, name, record, trace, parts, arr, method=audio_opts.get("method", "auto"),
                               factor=audio_opts.get("factor"), parts_channels=pc,
                               center_hz=float(audio_opts.get("center") or 2000.0), whole_channels=wc)
    m["parts_residual_db"] = residual
    overload = any(e.get("model") == "overload" for e in trace.get("models") or [])
    m["front_end"] = audio.front_end_note(residual, bool(float(record.metadata.adc_full_scale or 0.0) > 0.0), overload)
    m["notes"] = notes
    m["arrivals_ranked_by"] = ("the synthesis's own amplitude over spherical spreading (viz/arrivals.py)" if box is not None
                               and box.get("kernel") is not None else "the trace's level (spreading and reflection only)")
    if any(n.startswith("single arrivals:") for n in notes):
        m["warning_words"] = next(n for n in notes if n.startswith("single arrivals:"))
    m["seconds"] = round(time.perf_counter() - t0, 2)
    adir = os.path.join(out, f"{name}.audio")
    with open(os.path.join(adir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    with open(os.path.join(adir, "manifest.js"), "w", encoding="utf-8") as f:
        f.write(audio.manifest_js(m))
    return m


def view(name: str, out: str, wav: int | None, html_page: bool, three: str | None, engine: str | None,
         quick: bool = False, all_rays: bool = False, truecolour: bool = False, bellhop3d: bool = True,
         audio_opts: dict | None = None) -> dict:
    from gpusonar.config import load_scenario_config
    from gpusonar.viz import audio, catalogue, page2d
    from gpusonar.viz import trace as tr

    audio_opts = dict(AUDIO_DEFAULTS, **(audio_opts or {}))
    sound = audio_opts["mode"] != "none"
    t0 = time.perf_counter()
    raw = load_scenario_config(name, None, expand=False)
    plan = catalogue.view_plan(raw, quick=quick)
    result, base, stages, box = _simulate(name, plan, engine, sound)
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
    manifest = None
    if sound:
        manifest = _sound(name, out, trace, result.records[0], stages, box, audio_opts, engine)
        facts["audio"] = _audio_facts(manifest)
    if html_page:
        facts["html"] = _page3d(trace, name, out, three, bellhop3d, manifest, audio_opts["mode"])
    return facts


def _audio_facts(m: dict) -> dict:
    return {"method": m["plan"]["method"], "tag": m["plan"]["tag"], "rate_hz": m["plan"]["rate_hz"],
            "files": len(m["files"]), "bytes": m["bytes"], "arrivals": len(m["arrivals"]),
            "kinds": sorted(m["kinds"]), "parts_residual_db": m.get("parts_residual_db"), "seconds": m.get("seconds"),
            "notes": m.get("notes"), "clipped": sum(f["clipped"] for f in m["files"])}


def _page3d(trace: dict, name: str, out: str, three: str | None, bellhop3d: bool = True, manifest: dict | None = None,
            audio_mode: str = "side") -> dict:
    """The 3-D page of a trace: `scene3d.enrich` (the bed grid, motion over the pings, Bellhop3D's own rays for
    `eigenray3d`), then `page3d.write_page` (with the ping's sound when `manifest` is given). Returns the page's facts
    with the time it took."""
    from gpusonar.viz import page3d, scene3d
    t0 = time.perf_counter()
    scene3d.enrich(trace, name, bellhop3d=bellhop3d)
    info = page3d.write_page(trace, os.path.join(out, f"{name}.html"), three=three, audio=manifest,
                             audio_mode="embed" if audio_mode == "embed" else "side",
                             audio_dir=os.path.join(out, f"{name}.audio"))
    info.update(seconds=round(time.perf_counter() - t0, 2), bellhop3d=(trace.get("scene3d") or {}).get("bellhop3d"),
                notes=(trace.get("scene3d") or {}).get("notes"))
    return info


def view3d(name: str, out: str, three: str | None, traces: str, bellhop3d: bool = True, audio_opts: dict | None = None,
           engine: str | None = None, sim_limit: float = 330.0) -> dict:
    """The 3-D page from a trace a previous run wrote (`traces`/NAME.trace.json): the trace is not made again. The
    2-D page and the facts beside it are copied into `out` (for the index). A scenario with no trace there (its run
    failed or timed out) gets a page from `static_trace` (geometry, boundaries, beams and models; no eigenrays) that
    says what is missing and how to make it. With sound (`audio_opts` mode side or embed) the ping is simulated again
    for its record (scenes whose previous run took at most `sim_limit` s): the sound and a new 2-D page (with the
    spectrogram and the reverberation by item) are made from it."""
    import shutil
    src = os.path.join(traces, f"{name}.trace.json")
    facts_src = os.path.join(traces, f"{name}.facts.json")
    facts = {}
    if os.path.exists(facts_src):
        with open(facts_src, encoding="utf-8") as f:
            facts = json.load(f)
    if os.path.abspath(traces) != os.path.abspath(out):
        for ext in (".png",):
            if os.path.exists(os.path.join(traces, name + ext)):
                shutil.copyfile(os.path.join(traces, name + ext), os.path.join(out, name + ext))
    audio_opts = dict(AUDIO_DEFAULTS, **(audio_opts or {}))
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f:
            trace = json.load(f)
        manifest = None
        if audio_opts["mode"] != "none":
            prev = facts.get("simulate_s")
            if prev is not None and prev > sim_limit:
                trace.setdefault("missing", []).insert(0, (
                    f"NO SOUND on this page: the ping took {prev:.0f} s to simulate on this CPU (limit {sim_limit:.0f} s for "
                    f"the sound run). Make it on the GPU job: `PYTHONPATH=python python3 tools/scenario_view.py {name} "
                    "--out DIR --3d`"))
                facts["audio"] = {"skipped": f"simulate {prev:.0f} s > {sim_limit:.0f} s"}
            else:
                try:
                    manifest, facts["audio"] = _sound_from_trace(name, out, trace, audio_opts, engine)
                except Exception as exc:                  # noqa: BLE001  (the page says so; never stops --all)
                    err = f"{type(exc).__name__}: {exc}"
                    trace.setdefault("missing", []).insert(0, f"NO SOUND on this page: the sound run failed ({err})")
                    facts["audio"] = {"error": err, "trace": traceback.format_exc(limit=4)}
        facts["html"] = _page3d(trace, name, out, three, bellhop3d, manifest, audio_opts["mode"])
        facts["html"]["from"] = "the trace of a previous run (" + os.path.basename(src) + ")"
    else:
        facts.setdefault("scenario", name)
        facts["html"] = static_page3d(name, out, three, facts.get("error") or "no trace from a previous run", bellhop3d)
    return facts


def _sound_from_trace(name: str, out: str, trace: dict, audio_opts: dict, engine: str | None):
    """Simulate the traced ping again (the same view plan, so the same ping and seed) for its record, check that it
    is the trace's (sample rate, samples, channels, start time), redraw the 2-D page with the spectrogram and the
    reverberation by item, and write the sound. Returns (manifest, facts)."""
    from gpusonar.config import load_scenario_config
    from gpusonar.viz import catalogue, page2d
    t0 = time.perf_counter()
    raw = load_scenario_config(name, None, expand=False)
    plan = catalogue.view_plan(raw, quick=False)
    result, _, stages, box = _simulate(name, plan, engine, True)
    t_sim = time.perf_counter() - t0
    rec = result.records[0]
    s, ts = rec.samples, trace["sonar"]
    same = (abs(float(s.sample_rate) - ts["sample_rate"]) < 1e-6 and int(s.n_samples) == ts["n_samples"]
            and int(s.n_channels) == ts["n_channels"] and abs(float(s.start_time) - ts["start_time_s"]) < 1e-9)
    if not same:
        raise RuntimeError(f"{name}: the record simulated again is not the trace's (rate, samples, channels or start)")
    t1 = time.perf_counter()
    drawn = page2d.draw_page(trace, rec, os.path.join(out, f"{name}.png"), stages=stages)
    t_png = time.perf_counter() - t1
    m = _sound(name, out, trace, rec, stages, box, audio_opts, engine)
    facts = dict(_audio_facts(m), simulate_s=round(t_sim, 2), png_s=round(t_png, 2), items=drawn.get("items"),
                 png_bytes=os.path.getsize(os.path.join(out, f"{name}.png")))
    return m, facts


def static_page3d(name: str, out: str, three: str | None, why: str, bellhop3d: bool = True) -> dict:
    """A 3-D page without eigenrays (`trace.static_trace`: no simulation, no provider call) that says why and how to
    make the full one."""
    from gpusonar.viz import trace as tr
    trace = tr.static_trace(name)
    trace["missing"].insert(0, f"NO EIGENRAYS: the one-ping run did not finish ({why.split('. Listed')[0]}). This page "
                               f"shows the geometry, boundaries, beams and models only. Make the trace on the GPU job "
                               f"(CUDA synthesis): `PYTHONPATH=python python3 tools/scenario_view.py {name} --out DIR --3d`")
    info = _page3d(trace, name, out, three, bellhop3d)
    info["from"] = "static_trace (no simulation)"
    return info


def _child(conn, call, fn=None):
    try:
        conn.send(("ok", (fn or view)(*call)))
    except Exception as exc:                          # sent back to the parent, which reports it
        conn.send(("error", f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=3)))
    conn.close()


def _with_timeout(call, seconds: float, fn=None) -> dict:
    """view(*call) (or fn(*call)) in a child process, stopped after `seconds` (TimeoutError): a scene too slow for
    this CPU."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, call, fn))
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
    ap.add_argument("--html", "--3d", dest="html", action="store_true",
                    help="write the 3-D page DIR/<name>.html (viz/page3d.py; Bellhop3D's own rays for eigenray3d)")
    ap.add_argument("--three", default=None, help="cdn | embed | PATH: load three.js from the pinned CDN, embed the "
                    "vendored copy (third_party/three; the default when it exists) or embed PATH")
    ap.add_argument("--traces", default=None, help="with --3d: build the 3-D pages from DIR2/<name>.trace.json of a "
                    "previous run (nothing simulated); a scenario without one gets a page that says what is missing")
    ap.add_argument("--no-bellhop3d", action="store_true", help="eigenray3d pages keep the 2-D projection (no Bellhop3D run)")
    ap.add_argument("--engine", default=None, choices=("cpp", "numpy"))
    ap.add_argument("--quick", action="store_true", help="thin the densest scenes as tools/gallery.py --quick does")
    ap.add_argument("--all-rays", action="store_true", help="colour every eigenray, not only the strongest of each leg")
    ap.add_argument("--truecolour", action="store_true", help="true-colour PNG (about 3 times larger) instead of the fixed palette")
    ap.add_argument("--timeout", type=float, default=None, help="give up on a scenario after this many seconds (its page says so)")
    ap.add_argument("--skip-existing", action="store_true", help="reuse DIR/<name>.facts.json of scenarios already done")
    ap.add_argument("--index", action="store_true", help="write DIR/index.html for the pages named (always with --all)")
    ap.add_argument("--audio", default=None, choices=("side", "embed", "none"),
                    help="with --3d: the ping's sound beside the page (side, the default with --3d), in it (embed, capped) or none")
    ap.add_argument("--audio-method", default="auto", choices=("auto", "as_recorded", "heterodyne", "time_stretch"),
                    help="auto: as recorded when the band's top is at or under 16 kHz, time stretch for a record "
                    "shorter than 0.5 s, else heterodyne to --audio-center")
    ap.add_argument("--stretch", type=float, default=None, help="time-stretch factor (default: the smallest whole number "
                    "that brings the band's top under 16 kHz)")
    ap.add_argument("--audio-center", type=float, default=2000.0, help="heterodyne: the band's centre after the shift (Hz)")
    ap.add_argument("--audio-parts", default=None, help="hydrophones for the parts and single arrivals: C,C,... or all "
                    "(default the first and the last)")
    ap.add_argument("--audio-channels", default=None, help="hydrophones whose whole record is written: C,C,... or all "
                    "(the default)")
    ap.add_argument("--arrivals", type=int, default=8, help="single arrivals of the strongest target (default 8; 0: none)")
    ap.add_argument("--sim-limit", type=float, default=330.0, help="with --traces and sound: simulate again only scenes "
                    "whose previous run took at most this many seconds")
    args = ap.parse_args(argv)
    audio_mode = args.audio or ("side" if args.html else "none")
    audio_opts = {"mode": audio_mode, "method": args.audio_method, "factor": args.stretch, "center": args.audio_center,
                  "parts": args.audio_parts, "arrivals": args.arrivals, "channels": args.audio_channels}
    names = list(args.scenarios)
    if args.all:
        root = os.path.join(ROOT, "configs", "scenarios")
        names = sorted(f[:-5] for f in os.listdir(root) if f.endswith(".yaml"))
    if not names:
        ap.error("name a scenario or give --all")
    os.makedirs(args.out, exist_ok=True)
    from gpusonar.viz import page3d
    three = page3d.three_source(args.three)
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
            if args.traces and args.html:
                call3 = (name, args.out, three, args.traces, not args.no_bellhop3d, audio_opts, args.engine, args.sim_limit)
                facts = _with_timeout(call3, args.timeout, view3d) if args.timeout else view3d(*call3)
                with open(facts_path, "w", encoding="utf-8") as f:
                    json.dump(facts, f)
                rows.append(facts)
                print(json.dumps({"scenario": name, "html": facts["html"]}), flush=True)
                continue
            call = (name, args.out, args.wav, args.html, three, args.engine, args.quick, args.all_rays, args.truecolour,
                    not args.no_bellhop3d, audio_opts)
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
            if args.html:
                try:
                    facts["html"] = static_page3d(name, args.out, three, err, not args.no_bellhop3d)
                except Exception as exc3:            # noqa: BLE001  (reported, never stops --all)
                    facts["html_error"] = f"{type(exc3).__name__}: {exc3}"
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
