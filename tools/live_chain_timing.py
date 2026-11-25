#!/usr/bin/env python3
"""Where a live chain spends its time on a Popoto build's trial, per engine (sonar-drc).

    tools/live_chain_timing.py                                  # 1, 2 and 3 rows, both engines, 3 repeats
    tools/live_chain_timing.py --rows 3 --engine cpp --omp-threads 4 --repeats 5

The pings are the ones tests/python/test_hsb.py's direct chain processes: the trial (tools/hsb_frame_timing.py
`popoto_stream`, the test's POPOTO table) framed as `serve` frames it and assembled again, then run through the
deployment file's chain (`LiveChain`, either engine, on this host's backend: CUDA where there is a GPU). Per repeat a
new chain takes the trial's pings in order. Measured per ping:
- `chain_s`: `LiveChain.process`, the whole chain;
- `sync_s`: the board synchronisation alone on a copy of the same ping (`core.live_synchronise` for the C++ engine,
  `sync.synchronise` for the Python engine), when the deployment synchronises at the full rate (no decimate-first
  stage); the rest of the chain is `chain_s - sync_s`.
`--omp-threads` sets OpenMP's thread count for this thread (`receive --chain-omp-threads`). Prints one JSON line per
(rows, engine): TIMING. Simulating the 3-row trial takes about 10 s; nothing is written to disk.
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


def _spread(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"min": round(float(x.min()), 4), "median": round(float(np.median(x)), 4), "max": round(float(x.max()), 4)}


def time_engine(pings, cfg: dict, engine: str, repeats: int) -> dict:
    from gpusonar import core
    from gpusonar import sync as pysync
    from gpusonar.live import LiveChain
    chain_s, sync_s = [], []
    full_rate_sync = cfg["chain"].get("sync") is not None and cfg["chain"].get("decimate") is None
    backend = None
    for _ in range(repeats):
        with LiveChain(dict(cfg["chain"], engine=engine)) as chain:
            backend = str(chain.backend)
            for p in pings:
                rec = p.copy()
                t = time.perf_counter()
                chain.process(rec)
                chain_s.append(time.perf_counter() - t)
            ref = chain._reference
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
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    ap.add_argument("--engine", nargs="+", choices=["python", "cpp"], default=["cpp", "python"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--omp-threads", type=int, help="OpenMP threads for the chain (receive --chain-omp-threads); unset: as it is")
    args = ap.parse_args(argv)
    from gpusonar.sources import hsb
    applied = hsb.set_chain_omp_threads(args.omp_threads) if args.omp_threads else None
    for rows in args.rows:
        t = time.perf_counter()
        pings, cfg = assembled_pings(rows)
        made = time.perf_counter() - t
        for engine in args.engine:
            row = time_engine(pings, cfg, engine, args.repeats)
            row.update(rows=rows, channels=int(pings[0].samples.n_channels), n_samples=int(pings[0].samples.n_samples),
                       omp_threads=applied, simulated_s=round(made, 1))
            print("TIMING " + json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
