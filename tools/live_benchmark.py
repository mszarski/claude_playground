#!/usr/bin/env python3
"""Time the long-range chain per ping on a recording: host and CUDA, bearings from the map and by SBL.

    tools/live_benchmark.py recording.h5 [--config configs/deploy/popoto_vehicle.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

import gpusonar as gs  # noqa: E402
from gpusonar.config import load_yaml  # noqa: E402
from gpusonar.live import cpp_chain_options  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--config", default=str(ROOT / "configs/deploy/popoto_vehicle.yaml"))
    args = ap.parse_args(argv)
    core = gs.core
    reader = gs.RecordReader(args.recording)
    records = [reader.read(i) for i in range(len(reader))]
    chain = load_yaml(args.config).get("chain") or {}
    backends = [("host", core.Backend.Host)] + ([("cuda", core.Backend.Cuda)] if core.cuda_available() else [])
    out = {}
    for doa in ("map", "sbl"):
        for name, backend in backends:
            cpp = core.LiveChainCpp(dict(cpp_chain_options(dict(chain, doa=doa)), backend=backend))
            reports = [json.loads(cpp.process(r.copy())) for r in records]
            times = [r["processing_seconds"] for r in reports]
            out[(doa, name)] = reports
            print(f"doa={doa:3s} backend={name:4s} s/ping: {' '.join(f'{t:.3f}' for t in times)}  median {np.median(times[1:]):.3f}  "
                  f"detections {[len(r['detections']) for r in reports]}  kinds {sorted(t['kind'] for t in reports[-1]['tracks'])}")
    if len(backends) == 2:
        for doa in ("map", "sbl"):
            a, b = out[(doa, "host")][-1], out[(doa, "cuda")][-1]
            same = len(a["detections"]) == len(b["detections"]) and all(
                abs(x["bearing_deg"] - y["bearing_deg"]) < 0.02 and abs(x["range_m"] - y["range_m"]) < 0.5
                for x, y in zip(a["detections"], b["detections"]))
            print(f"doa={doa}: host and cuda detections of the last ping agree: {same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
