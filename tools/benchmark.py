#!/usr/bin/env python3
"""Benchmark harness (ASASIN Sec. III-C): fixed datasets, each processed three times in a
row, the last run reported. Prints a table and writes JSON.

Datasets are simulated here with fixed seeds and sizes, so every machine times the same
work. Three modes, all through the one backprojection kernel:

  sas    stripmap SAS: `sas_auv`, N pings, seafloor image at a stated pixel size
  fls    forward-looking fan: `l12_5d`, both mux halves, range x bearing image
  flank  flank array: `flank_32`, one ping, range x bearing map to 10 km

Real-time factor = time the sonar took to collect the data / time to image it.
Above 1 is faster than real time.

Usage: tools/benchmark.py [--backend cuda|host] [--scale 1.0] [--out result.json]
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar import recon  # noqa: E402
from gpusonar.recon.multigpu import backproject_tiles  # noqa: E402
from gpusonar.sim import Scene, make_pose, simulate_ping  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

core = gs.core


def timed3(fn):
    """Run three times in a row, report the last (the paper's protocol)."""
    times = []
    out = None
    for _ in range(3):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return times[-1], times, out


def sas_dataset(scale, pixel=0.05):
    n_pings = max(int(round(200 * scale)), 20)
    advance, interval, tilt = 0.32, 0.2, np.radians(20.0)
    sonar = SonarModel.load("sas_auv", {
        "receive": {"kind": "baseband", "sample_rate": 100e3, "window": {"range": [30.0, 100.0]}}})
    targets = np.array([[n_pings * advance / 2, y, y * np.tan(tilt)] for y in (40.0, 60.0, 85.0)])
    scene = Scene().add_points(targets)
    records = [simulate_ping(sonar, make_pose(position=(i * advance, 0.0, 0.0), velocity=(advance / interval, 0, 0)),
                             scene, environment={"sound_speed": 1500.0}, absorption_model="none", adc=False,
                             sequence=i) for i in range(n_pings)]
    recon.prepare(records)
    x = np.arange(0.0, n_pings * advance, pixel)
    y = np.arange(32.0, 92.0, pixel)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([X, Y, Y * np.tan(tilt)], axis=-1)
    return {"records": records, "points": pts, "collection_time": n_pings * interval,
            "kwargs": dict(interp="linear", fov_factor=1.13),
            "describe": f"{n_pings} pings x 16 channels x {records[0].samples.n_samples} samples; "
                        f"image {pts.shape[0]} x {pts.shape[1]} at {pixel * 100:.0f} cm"}


def fls_dataset(scale):
    sonar = SonarModel.load("l12_5d")
    scene = Scene().add_points([[1.5, 0.1, 0.0], [2.2, -0.2, 0.0]])
    records = [simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1480.0},
                             absorption_model="none", sub_sequence=i, sequence=i) for i in range(2)]
    recon.prepare(records)
    ranges = np.linspace(0.8, 2.9, int(2000 * scale))
    bearings = np.radians(np.linspace(-20.0, 20.0, int(801 * scale)))
    origin, forward, across, _ = recon.array_frame(records[0])
    pts = recon.polar_points(np.vstack([r.rx_positions() for r in records]).mean(axis=0), forward, across,
                             ranges, bearings)
    # Two transmits of a 4 ms round trip each; the reference rig's ping interval is not
    # fixed yet, so the collection time is the acoustic minimum.
    return {"records": records, "points": pts, "collection_time": 2 * 2 * 2.95 / 1480.0,
            "kwargs": dict(interp="linear"),
            "describe": f"2 pings x 64 channels x {records[0].samples.n_samples} baseband samples; "
                        f"fan {pts.shape[0]} ranges x {pts.shape[1]} bearings"}


def flank_dataset(scale):
    sonar = SonarModel.load("flank_32")
    scene = Scene().add_points([[0.0, 5000.0, 0.0]], target_strength_db=10.0)
    rec = simulate_ping(sonar, make_pose(), scene, environment={"sound_speed": 1500.0}, adc=False)
    recon.prepare([rec])
    ranges = np.linspace(150.0, 10000.0, int(8000 * scale))
    bearings = np.radians(np.linspace(-60.0, 60.0, int(241 * scale)))
    origin, forward, across, _ = recon.array_frame(rec)
    pts = recon.polar_points(origin, forward, across, ranges, bearings)
    return {"records": [rec], "points": pts, "collection_time": 2 * 10000.0 / 1500.0 + 1.0,
            "kwargs": dict(interp="linear"),
            "describe": f"1 ping x 32 channels x {rec.samples.n_samples} baseband samples; "
                        f"map {pts.shape[0]} ranges x {pts.shape[1]} bearings"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cuda" if gs.cuda_available() else "host")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--modes", default="sas,fls,flank")
    ap.add_argument("--pixel", type=float, default=0.05, help="SAS pixel size, m")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    backend = core.Backend.Cuda if args.backend == "cuda" else core.Backend.Host
    n_dev = core.cuda_device_count() if args.backend == "cuda" else 0
    result = {"backend": args.backend, "scale": args.scale, "machine": platform.machine(),
              "devices": [core.cuda_device_name(i) for i in range(n_dev)], "modes": {}}
    print(f"backend {args.backend}; devices {result['devices'] or 'none'}; scale {args.scale}")
    builders = {"sas": sas_dataset, "fls": fls_dataset, "flank": flank_dataset}
    for mode in args.modes.split(","):
        d = builders[mode](args.scale, args.pixel) if mode == "sas" else builders[mode](args.scale)
        pts, kw = d["points"], d["kwargs"]
        flat = pts.reshape(-1, 3)
        n_records = sum(r.samples.n_channels for r in d["records"])
        entry = {"dataset": d["describe"], "pixels": int(flat.shape[0]), "records": int(n_records),
                 "collection_time_s": d["collection_time"]}

        def plain():
            return core.backproject(d["records"], flat, backend=backend, **kw)

        t, all_t, img = timed3(plain)
        entry["time_s"], entry["runs_s"] = t, all_t
        entry["real_time_factor"] = d["collection_time"] / t
        entry["pixel_records_per_s"] = flat.shape[0] * n_records / t
        # The same work through a session: inputs built once, samples resident on the device.
        t0 = time.perf_counter()
        session = core.BackprojectSession(d["records"])
        entry["session_prepare_s"] = time.perf_counter() - t0
        ts, all_ts, img_s = timed3(lambda: session.backproject(flat, backend=backend, **kw))
        entry["session_time_s"], entry["session_runs_s"] = ts, all_ts
        entry["session_identical"] = bool(np.array_equal(img, img_s))
        entry["session_resident_mb"] = session.resident_bytes / 1e6
        entry["session_real_time_factor"] = d["collection_time"] / ts
        entry["session_pixel_records_per_s"] = flat.shape[0] * n_records / ts
        if mode == "sas":
            stats = {}

            def culled():
                return core.backproject(d["records"], flat, backend=backend, cull_block=16,
                                        grid_shape=pts.shape[:2], stats=stats, **kw)

            tc, _, img_c = timed3(culled)
            entry["time_culled_s"] = tc
            entry["culling_speedup"] = t / tc
            entry["culled_identical"] = bool(np.array_equal(img, img_c))
            entry["ping_blocks_kept"] = stats["ping_blocks_kept"] / stats["ping_blocks_total"]
            entry["real_time_factor_culled"] = d["collection_time"] / tc
            tsc, _, img_sc = timed3(lambda: session.backproject(flat, backend=backend, cull_block=16,
                                                                grid_shape=pts.shape[:2], **kw))
            entry["session_time_culled_s"] = tsc
            entry["session_culled_identical"] = bool(np.array_equal(img, img_sc))
            entry["session_real_time_factor_culled"] = d["collection_time"] / tsc
            if n_dev > 1:
                tm, _, img_m = timed3(lambda: backproject_tiles(d["records"], pts, cull_block=16, **kw))
                entry["time_multi_gpu_s"] = tm
                entry["multi_gpu_devices"] = n_dev
                entry["multi_gpu_scaling"] = tc / tm
                entry["multi_gpu_identical"] = bool(np.array_equal(img_c.reshape(pts.shape[:2]), img_m))
                tms, _, img_ms = timed3(lambda: backproject_tiles(d["records"], pts, cull_block=16,
                                                                  session=session, **kw))
                entry["session_time_multi_gpu_s"] = tms
                entry["session_multi_gpu_scaling"] = tsc / tms
                entry["session_multi_gpu_identical"] = bool(np.array_equal(img_c.reshape(pts.shape[:2]), img_ms))
        result["modes"][mode] = entry
        print(f"[{mode}] {d['describe']}")
        for k, v in entry.items():
            if k not in ("dataset",):
                print(f"    {k}: {v}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    print("BENCHMARK_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
