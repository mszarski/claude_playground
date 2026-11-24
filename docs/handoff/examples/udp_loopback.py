#!/usr/bin/env python3
"""Example 7 of docs/handoff/EXAMPLES.md: the flank UDP path on one host, and a lost block.

    PYTHONPATH=python python3 docs/handoff/examples/udp_loopback.py --out /tmp/udp \
        [--program BUILD/bin/sonar_live] [--drop 1200]

Six pings of the 32-hydrophone flank array are simulated (the scene of
tests/python/test_live.py: ownship at 1.5 m/s, one target crossing at 3 m/s, one at rest,
receive window 100 to 400 m), cut into board blocks by `FlankBoardEmulator` and sent to a UDP
port on this host, paced to a quarter of real time. They are received twice:

  1. by the Python source `flank_udp` feeding `LiveChain` (python/gpusonar/live.py);
  2. with --program, by the standalone C++ program, which also takes navigation datagrams.

--drop N leaves block N out of the stream (block 1200 lies inside the second ping's receive
window: 1000 blocks of 128 samples per 2 s ping, window at blocks 1066 to 1266).
"""
import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

import gpusonar as gs
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator, make_source

OVERRIDES = {"receive": {"window": {"range": [100.0, 400.0]}}}
INTERVAL, N = 2.0, 6
CHAIN = {"bearings_deg": [-40.0, 40.0, 1.0], "range_cell_m": 1.0, "cfar": {"pfa": 1e-6, "guard_m": 15.0, "train_m": 50.0},
         "min_snr_db": 12.0, "backend": "host",
         "tracker": {"range_sigma": 3.0, "bearing_sigma_deg": 1.0, "confirm_hits": 3, "max_misses": 2},
         "classify": {"min_speed": 1.0, "static_speed": 0.6, "max_fit_rms": 10.0, "multipath": None}}
# The same settings as key=value for the C++ program (tests/python/test_live_cpp.py EXE_OPTIONS).
PROGRAM_OPTIONS = ["bearing_first_deg=-40", "bearing_last_deg=40", "bearing_step_deg=1", "range_cell_m=1", "cfar_pfa=1e-6",
                   "cfar_guard_m=15", "cfar_train_m=50", "min_snr_db=12", "tracker_range_sigma=3", "tracker_bearing_sigma_deg=1",
                   "tracker_confirm_hits=3", "tracker_max_misses=2", "classify_max_fit_rms=10", "multipath=0"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--program", help="path of the built sonar_live; left out: the Python source only")
ap.add_argument("--drop", type=int, nargs="*", default=[], help="block numbers to leave out of the stream")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

sonar = SonarModel.load("flank_32", OVERRIDES)
rng = np.random.default_rng(5)
pings = []
for i in range(N):
    t = i * INTERVAL
    scene = Scene().add_points([[40.0 - 3.0 * t, 250.0, 0.0], [-60.0, 320.0, 0.0]], target_strength_db=10.0)
    pings.append(simulate_ping(sonar, make_pose(position=(1.5 * t, 0.0, 0.0), velocity=(1.5, 0.0, 0.0)), scene,
                               environment={"sound_speed": 1500.0}, sequence=i, time_ns=int(t * 1e9), noise_rms=1e-5, rng=rng))
emulator = FlankBoardEmulator(pings, INTERVAL, block_samples=128)
print(f"stream: {N} pings, {sum(1 for _ in emulator.blocks())} blocks of 128 samples x 32 channels, dropping {args.drop or 'none'}")


def line(r):
    kinds = sorted(t["kind"] for t in r["tracks"])
    return f"  ping {r['ping']} sequence {r['sequence']} detections {len(r['detections'])} tracks {kinds}"


# 1. The Python source and chain.
source = make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": OVERRIDES, "max_pings": N, "timeout": 5.0})
try:
    sender = emulator.send(source.port, drop=args.drop, pace=0.25)
    with LiveChain(CHAIN) as chain:
        reports = chain.run(source)
    sender.join(timeout=60.0)
finally:
    source.close()
print("python source flank_udp -> LiveChain:")
for r in reports:
    print(line(r))
a = source.assembler
print(f"  assembler: dropped_pings {a.dropped_pings}, lost_samples {a.lost_samples}, late_blocks {a.late_blocks}; "
      f"source: bad_blocks {source.bad_blocks}, dropped_blocks {source.dropped_blocks}, overrun_pings {source.overrun_pings}")

# 2. The C++ program: a template record tells it the array; navigation arrives as JSON datagrams.
if args.program:
    w = gs.RecordWriter(str(out / "template.h5"))
    w.write(pings[0].copy())
    w.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    cmd = [args.program, "--udp", str(port), "--template", str(out / "template.h5"), "--nav-port", str(port + 1),
           "--report", str(out / "report.jsonl"), "--pings", str(N - (1 if args.drop else 0)), "idle_seconds=5", *PROGRAM_OPTIONS]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    time.sleep(1.0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(N + 1):                                         # navigation first: 1.5 m/s north
        t = i * INTERVAL
        tx.sendto(json.dumps({"time_ns": int(t * 1e9), "x": 1.5 * t, "y": 0.0, "z": 0.0, "yaw": 0.0,
                              "vx": 1.5, "vy": 0.0, "vz": 0.0}).encode(), ("127.0.0.1", port + 1))
    emulator.send(port, drop=args.drop, pace=0.25).join(timeout=120.0)
    _, err = proc.communicate(timeout=120.0)
    print(f"C++ program (exit status {proc.returncode}): {' '.join(cmd[:1] + ['--udp', 'PORT', '--template', 'template.h5', '--nav-port', 'PORT+1', '--report', 'report.jsonl'] + cmd[9:])}")
    for text in (out / "report.jsonl").read_text().splitlines():
        print(line(json.loads(text)))
    print("  end-of-run summary (stderr): " + err.strip().splitlines()[-1])
