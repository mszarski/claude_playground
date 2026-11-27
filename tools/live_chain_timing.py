#!/usr/bin/env python3
"""Where a live chain spends its time on a Popoto build's trial, per engine (sonar-drc).

    tools/live_chain_timing.py                                  # 1, 2 and 3 rows, both engines, 3 repeats
    tools/live_chain_timing.py --rows 3 --engine cpp --omp-threads 4 --repeats 5
    tools/live_chain_timing.py --rows 2 3 --decimate                 # the 2- and 3-row trials decimated first
    tools/live_chain_timing.py --doppler --doppler-combine off max cfar --backend cuda --json FILE
    tools/live_chain_timing.py --scene flank_patrol --scene-pings 3 --doppler --doppler-combine off max cfar --sim-backend cuda
    tools/live_chain_timing.py --scene popoto_12km_doppler --doppler-tone --sim-backend cuda --backend cuda

The pings are the ones tests/python/test_hsb.py's direct chain processes: the trial (tools/hsb_frame_timing.py
`popoto_stream`, the test's POPOTO table) framed as `serve` frames it and assembled again, then run through the
deployment file's chain (`LiveChain`, either engine, on this host's backend: CUDA where there is a GPU). Per repeat a
new chain takes the trial's pings in order. Measured per ping:
- `chain_s`: `LiveChain.process`, the whole chain;
- `sync_s`: the board synchronisation alone on a copy of the same ping (`core.live_synchronise` for the C++ engine,
  `sync.synchronise` for the Python engine), when the deployment synchronises at the full rate (no decimate-first
  stage); the rest of the chain is `chain_s - sync_s`;
- `stage_s`: the decimate-first stage alone on a copy of the same ping (`core.live_decimate_record` with the options
  and reference the chain gives it: filter, board clocks from the bursts and the baseband correction; the same C++
  code in both engines), when the deployment decimates first.
`--decimate` switches the decimate-first stage on (`decimate: {}`) for every deployment (the one-row file has it).
`--doppler` switches the sweep's Doppler bank on (`doppler: {mode: sweep_bank}`, D4; `--doppler-max-rate` its span, m/s):
the cost of the bank against the chain without it.
`--doppler-combine` (with `--doppler`; default max, the bank as above): one row per value, `off` (no bank), `max` (D4's
maximum over hypotheses) and `cfar` (D3's per-(beam, hypothesis) normalisation, `combine: cfar`).
`--omp-threads` sets OpenMP's thread count for this thread (`receive --chain-omp-threads`). Prints one JSON line per
(rows, engine): TIMING. Simulating the 3-row trial takes about 10 s; nothing is written to disk unless --json or --cache.

Scenes instead of the trial (DOPPLER.md table 9.5's scene rows; the tone path, D3 part B):
`--scene NAME` simulates `configs/scenarios/NAME.yaml` (its own seed; `--scene-pings` pings, default 3; the eigenray scenes
need GPUSONAR_BELLHOP; `--sim-backend` host or cuda, default host; `--cache DIR` keeps the records, with the file names of
tools/doppler_tone_eval.py --cache, so that a run of that tool on the same scene, pings and simulator backend is read back)
and times the chain with the deployment defaults the evaluation tools use (`{engine, backend}` and nothing else; bearings
-40 to 40 every 0.5, backprojection, multiscale). `--doppler-tone` adds the row `tone`: `doppler: {tone: bank, report: true,
max_range_rate_m_s: --doppler-max-rate}` with the tone's segment from the scene's train (gpusonar.doppler.tone_from_sonar;
a CW record its own tone), and an `off` row if none was asked. The sweep bank is refused on a train record: use
`--doppler` on the sweep scenes and `--doppler-tone` on the train scenes.
`--backend` sets the chain's backend (host, cuda or auto; default: the deployment file's, which is auto, and auto for a scene).
`--set KEY=VALUE` (with --scene) changes the scene as tools/doppler_tone_eval.py --set does (a plumbing check).
`--json FILE` writes every row so far after each row (a run stopped by a time limit keeps what it had).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

FRAME = 400
LEAD = 1.0


def assembled_pings(rows: int):
    """(pings as the direct chain of the loopback test gets them, deployment dict)."""
    from hsb_frame_timing import popoto_stream
    from gpusonar.sources import hsb
    records, cfg = popoto_stream(rows)
    src = cfg["source"]
    pings = hsb.records_from_frames(hsb.frames_from_records(records, FRAME, lead_seconds=LEAD), src["sonar"],
                                    src.get("sonar_overrides"), sound_speed=src["sound_speed"])
    return pings, cfg


def scene_pings(name: str, pings: int, sim_backend: str, cache: str | None, sets=None):
    """(records of the scene, chain dict of the evaluation tools, the tone's keys or None), simulated through
    tools/doppler_tone_eval.py's cache."""
    import doppler_tone_eval as tone_eval
    from gpusonar.config import load_scenario_config
    cfg = tone_eval.apply_sets(load_scenario_config(name, expand=False), sets)
    cfg["pings"] = dict(cfg.get("pings", {}), count=int(pings))
    tone_eval.CACHE, tone_eval.SIM_BACKEND = cache, sim_backend
    records = tone_eval.simulate(cfg)
    return records, {"chain": {"backend": "auto"}}, tone_eval.tone_keys(cfg)


def variants(args, tone: dict | None) -> list:
    """(name, the chain's `doppler` value or None) for each row asked."""
    if not args.doppler and not args.doppler_tone:
        return [("plain", None)]
    out = []
    if args.doppler:
        for c in args.doppler_combine:
            if c == "off":
                out.append(("off", None))
            else:
                d = {"mode": "sweep_bank", "max_range_rate_m_s": args.doppler_max_rate}
                if c != "max":
                    d["combine"] = c
                out.append((c, d))
    if args.doppler_tone:
        if tone is None:
            raise SystemExit("--doppler-tone needs --scene (the trial's pings are sweeps: no tone path)")
        if not any(n == "off" for n, _ in out):
            out.insert(0, ("off", None))
        out.append(("tone", dict(tone, report=True, max_range_rate_m_s=args.doppler_max_rate)))
    return out


def _spread(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"min": round(float(x.min()), 4), "median": round(float(np.median(x)), 4), "max": round(float(x.max()), 4)}


def time_engine(pings, cfg: dict, engine: str, repeats: int) -> dict:
    from gpusonar import core
    from gpusonar import sync as pysync
    from gpusonar.live import LiveChain
    chain_s, sync_s, stage_s = [], [], []
    full_rate_sync = cfg["chain"].get("sync") is not None and cfg["chain"].get("decimate") is None
    decimating = cfg["chain"].get("decimate") is not None
    backend = None
    for _ in range(repeats):
        with LiveChain(dict(cfg["chain"], engine=engine)) as chain:
            backend = str(chain.backend)
            for p in pings:
                rec = p.copy()
                t = time.perf_counter()
                chain.process(rec)
                chain_s.append(time.perf_counter() - t)
            ref, dec, bl = chain._reference, chain._decimate, chain._blanking
        if decimating:
            for p in pings:
                rec = p.copy()
                blank = ({"blank_until_s": rec.transmit.waveform.duration + bl["recovery_s"], "blank_taper_s": bl["taper_s"]}
                         if bl and bl["mode"] == "zero" else {})
                t = time.perf_counter()
                core.live_decimate_record(rec, dict(dec, **blank), ref)
                stage_s.append(time.perf_counter() - t)
        if full_rate_sync:
            for p in pings:
                rec = p.copy()
                t = time.perf_counter()
                if engine == "cpp":
                    core.live_synchronise(rec, ref)
                else:
                    pysync.synchronise(rec, ref, enforce=False)
                sync_s.append(time.perf_counter() - t)
    n = len(pings)
    after_first = [c for i, c in enumerate(chain_s) if i % n]
    out = {"engine": engine, "backend": backend, "pings": n, "repeats": repeats,
           "chain_s_first_ping": _spread(chain_s[::n]), "chain_s_after_first": _spread(after_first)}
    if sync_s:
        out["sync_s"] = _spread(sync_s)
        out["rest_s_after_first"] = round(float(np.median(after_first)) - float(np.median(sync_s)), 4)
    if stage_s:
        out["stage_s"] = _spread(stage_s)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    ap.add_argument("--engine", nargs="+", choices=["python", "cpp"], default=["cpp", "python"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decimate", action="store_true", help="switch the decimate-first stage on (decimate: {}) for every deployment")
    ap.add_argument("--doppler", action="store_true", help="switch the sweep's Doppler bank on (doppler: {mode: sweep_bank}; D4)")
    ap.add_argument("--doppler-max-rate", type=float, default=3.0, help="the bank's span either side of each beam's own-ship rate, m/s")
    ap.add_argument("--omp-threads", type=int, help="OpenMP threads for the chain (receive --chain-omp-threads); unset: as it is")
    ap.add_argument("--doppler-combine", nargs="+", choices=["off", "max", "cfar"], default=["max"],
                    help="with --doppler: one row per value (off: no bank; max: D4; cfar: D3's combine: cfar)")
    ap.add_argument("--doppler-tone", action="store_true", help="with --scene: the tone path (doppler: {tone: bank}, D3 part B) and an off row")
    ap.add_argument("--backend", choices=["host", "cuda", "auto"], help="the chain's backend (default: the deployment file's; auto for a scene)")
    ap.add_argument("--scene", help="time a scene's simulated pings instead of the trial (configs/scenarios/NAME.yaml)")
    ap.add_argument("--scene-pings", type=int, default=3, help="--scene: pings simulated (the first is left out of the median)")
    ap.add_argument("--sim-backend", choices=["host", "cuda"], default="host", help="--scene: the simulator's backend")
    ap.add_argument("--cache", help="--scene: keep the simulated records here (tools/doppler_tone_eval.py --cache names)")
    ap.add_argument("--json", help="write every row so far to this file after each row")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="--scene: change the scene (tools/doppler_tone_eval.py --set; a plumbing check, not a measurement)")
    args = ap.parse_args(argv)
    from gpusonar.sources import hsb
    applied = hsb.set_chain_omp_threads(args.omp_threads) if args.omp_threads else None
    rows_out = []
    for rows in ([None] if args.scene else args.rows):
        t = time.perf_counter()
        tone = None
        if args.scene:
            pings, cfg, tone = scene_pings(args.scene, args.scene_pings, args.sim_backend, args.cache, args.set)
        else:
            pings, cfg = assembled_pings(rows)
        if args.decimate and cfg["chain"].get("decimate") is None:
            cfg["chain"] = dict(cfg["chain"], decimate={})
        if args.backend:
            cfg["chain"] = dict(cfg["chain"], backend=args.backend)
        made = time.perf_counter() - t
        base = cfg["chain"]
        for name, dop in variants(args, tone):
            cfg["chain"] = dict(base, doppler=dop) if dop is not None else dict(base)
            for engine in args.engine:
                row = time_engine(pings, cfg, engine, args.repeats)
                row.update(rows=rows, channels=int(pings[0].samples.n_channels), n_samples=int(pings[0].samples.n_samples),
                           omp_threads=applied, simulated_s=round(made, 1), doppler=dop is not None and dop.get("mode") == "sweep_bank")
                if args.doppler or args.doppler_tone:
                    row.update(variant=name, doppler_config=dop)
                if args.scene:
                    row.update(scene=args.scene, sim_backend=args.sim_backend, set=args.set)
                print("TIMING " + json.dumps(row), flush=True)
                rows_out.append(row)
                if args.json:
                    Path(args.json).write_text(json.dumps(rows_out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
