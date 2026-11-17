#!/usr/bin/env python3
"""Stacked rows on the long-range array: does the data path carry 64 and 96 channels, and
what does the chain cost per ping with one, two and three rows?

  stream   one ping at the boards' rate (102.4 kHz, real, 24 bit) with 32, 64 and 96 channels
           through each transport: board blocks over UDP (sources/flank_stream.py), the same
           blocks through the C++ assembler (libs/live), fixed-size Sensor Bridge frames
           (sources/hsb.py), and the container file. Every transport must give the samples
           back within half a count of the 24-bit quantisation.
  scenarios  the three stacked-row scenarios through the C++ chain: depth against truth in free
           field with the vehicle rolled, in a shallow channel with bounces, and in the rehearsal
           with boards on their own clocks.
  timing   the C++ chain (host) per ping on pings of the `popoto_12km` scenario's size (complex
           baseband at 8 kHz, 11.3-12.7 km, bearings -60 to 60 degrees in 0.5 degree steps),
           and at the boards' full rate on a short window, for one, two and three rows.
           Without --full the full-length ping (0.5 to 13 km at 102.4 kHz) is not run: with
           96 channels it is 0.7 GB of samples before processing.

Writes docs/design/popoto_elevation_bench.json (read by tools/design_report.py). CPU only.
Usage: tools/popoto_elevation_bench.py [--no-write] [--full]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar.config import load_yaml  # noqa: E402
from gpusonar.live import LiveChain, cpp_chain_options  # noqa: E402
from gpusonar.sim import run_scenario  # noqa: E402
from gpusonar.sim import Scene, make_pose, simulate_ping  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402
from gpusonar.sources import FlankBoardEmulator, PingAssembler, hsb, make_source  # noqa: E402
from gpusonar.sources.flank_stream import decode_block  # noqa: E402

core = gs.core
SONARS = {1: "popoto_long_range", 2: "popoto_long_range_2row", 3: "popoto_long_range_3row"}
TARGETS = [[12000.0 * np.sin(np.radians(b)) + 0.0, (11600.0 + 80.0 * i) * np.cos(np.radians(b)), 4.0 + 20.0 * (i % 4)]
           for i, b in enumerate(np.linspace(-50.0, 50.0, 12))]            # twelve targets over the sector, 11.6 to 12.5 km


def pings(rows, overrides, targets, n, position=(0.0, 0.0, 10.0), noise=2e-6, ts=30.0):
    sonar = SonarModel.load(SONARS[rows], overrides)
    rng = np.random.default_rng(3)
    out = []
    for i in range(n):
        out.append(simulate_ping(sonar, make_pose(position=(position[0] + 30.0 * i, position[1], position[2]), velocity=(1.5, 0.0, 0.0)),
                                 Scene().add_points(targets, target_strength_db=ts), environment={"sound_speed": 1500.0},
                                 sequence=i, time_ns=int(20e9 * i), noise_rms=noise, rng=rng, adc=False))
    return sonar, out


def time_chain(records, rows):
    cfg = load_yaml(ROOT / "configs" / "deploy" / ("popoto_vehicle.yaml" if rows == 1 else "popoto_vehicle_elevation.yaml"))["chain"]
    cfg.pop("sync", None)                                  # these pings have ideal clocks
    chain = core.LiveChainCpp(dict(cpp_chain_options(cfg), backend=core.Backend.Host))
    reports = [json.loads(chain.process(r.copy())) for r in records]
    times = [r["processing_seconds"] for r in reports]
    last = reports[-1]
    with_el = sum(1 for d in last["detections"] if d.get("elevation_deg") is not None)
    return {"seconds_per_ping": float(np.median(times[1:]) if len(times) > 1 else times[0]), "all": times,
            "detections": len(last["detections"]), "with_elevation": with_el}


def stream_check(rows):
    """One ping at the boards' rate through every transport."""
    over = {"receive": {"window": {"range": [900.0, 1000.0]}}}
    sonar, recs = pings(rows, over, [[100.0, 950.0, 40.0]], 2, noise=1e-5, ts=10.0)
    n_ch = recs[0].samples.n_channels
    lsb = recs[0].metadata.adc_full_scale / 2**23
    interval = 3.0
    out = {"channels": n_ch, "boards": len(sonar.boards), "samples_per_ping": int(recs[0].samples.n_samples)}

    def same(got):
        return bool(len(got) == len(recs) and all(
            g.samples.array.shape == w.samples.array.shape and np.abs(g.samples.array - w.samples.array).max() <= 0.501 * lsb
            for g, w in zip(got, recs)))

    # Blocks as the boards would send them: a datagram stays under 16 kB whatever the channel count.
    block = max(8, (128 * 32 // n_ch) // 2 * 2)
    emulator = FlankBoardEmulator(recs, interval, block_samples=block)
    blocks = list(emulator.blocks())
    out["udp_block_samples"], out["udp_block_bytes"] = block, len(blocks[0])
    out["udp_blocks_per_second"] = float(recs[0].samples.sample_rate / block)
    assembler = PingAssembler(sonar)
    out["python_assembler_ok"] = same([r for b in blocks for r in assembler.feed(b)])
    cpp = core.PingAssemblerCpp(sonar.make_record(make_pose()))
    out["cpp_assembler_ok"] = same([r for b in blocks for r in cpp.feed(b)])
    source = make_source({"type": "flank_udp", "sonar": SONARS[rows], "sonar_overrides": over, "host": "127.0.0.1", "port": 0,
                          "max_pings": len(recs), "timeout": 10.0})
    try:
        sender = emulator.send(source.port, pace=0.0)
        got = list(source)
        sender.join(timeout=60.0)
        out["udp_loopback_ok"] = same(got)
        out["udp_lost_samples"] = int(source.assembler.lost_samples)
    finally:
        source.close()
    frame = 1600                                             # 15.6 ms of stream per Sensor Bridge frame
    frames = list(hsb.FrameStream(recs, frame, ping_interval=interval, lead_seconds=1.0))
    out["hsb_frame_bytes"] = len(frames[0])
    receiver = hsb.FrameReceiver(SONARS[rows], over, frame_samples=frame)
    out["hsb_frames_ok"] = same([r for f in frames for r in receiver.take(f)]) and decode_block(frames[0])["n_channels"] == n_ch
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ping.h5")
        w = gs.RecordWriter(path)
        for r in recs:
            w.write(r)
        w.close()
        out["recording_bytes_per_ping"] = int(os.path.getsize(path) / len(recs))
        back = gs.RecordReader(path)
        out["recording_ok"] = bool(len(back) == len(recs) and np.array_equal(back.read(0).samples.array, recs[0].samples.array)
                                   and np.allclose(back.read(0).rx_positions(), recs[0].rx_positions()))
    t = time_chain(recs, rows)
    out["chain_seconds_per_ping_full_rate_short_window"] = t["seconds_per_ping"]
    out["chain_found_the_target"] = t["detections"] >= 1
    return out


def scenario_check():
    """Truth checks on every stage's scenarios, C++ chain."""
    out = {}

    def chain_of(deploy, sync=True, **el):
        chain = load_yaml(ROOT / "configs" / "deploy" / f"{deploy}.yaml")["chain"]
        chain["tracker"] = dict(chain["tracker"], confirm_hits=2)
        if not sync:
            chain.pop("sync", None)
        if chain.get("elevation"):
            chain["elevation"] = dict(chain["elevation"], gradient_per_s=None, **el)      # the simulated water is isovelocity
        return chain

    def run(name, chain):
        with LiveChain(dict(chain, engine="cpp", backend="host")) as live:
            return live.run(run_scenario(name).records)

    def near(report, target):
        own = np.asarray(report["ownship_xyz"])
        strong = [d for d in report["detections"] if d["snr_db"] is not None and d["snr_db"] > 25.0]
        return min(strong, key=lambda d: abs(d["range_m"] - np.linalg.norm(np.asarray(target) - own))), own

    # Stage 1 and 2: the prototype's trials, raw samples, board clocks, the vehicle rolling.
    reps = run("popoto_testauv_trial_1row", chain_of("popoto_testauv_1row"))
    target = [120.0, 485.0, 15.0]
    rows = [near(r, target) for r in reps]
    out["testauv_1row"] = {"sync_residual_us": max(r["sync"]["worst_residual_us"] for r in reps),
                           "range_error_m": [d["range_m"] - float(np.linalg.norm(np.asarray(target) - own)) for d, own in rows],
                           "bearing_error_deg": [d["bearing_deg"] - float(np.degrees(np.arctan2(target[0] - own[0], target[1] - own[1]))) for d, own in rows]}
    reps = run("popoto_testauv_trial_2row", chain_of("popoto_testauv_2row", water_depth=30.0))
    target = [60.0, 294.0, 22.0]
    rows = [near(r, target)[0] for r in reps]
    out["testauv_2row"] = {"sync_residual_us": max(r["sync"]["worst_residual_us"] for r in reps), "true_depth_m": 22.0, "range_m": 300.0,
                           "depth_m": [d.get("depth_m") for d in rows], "ambiguous": [bool(d.get("elevation_ambiguous")) for d in rows]}
    # Stage 3: two rows on the face, shallow water at 3 km, with and without the water-depth bound on the sector.
    bounded = run("popoto_face_2row_shallow", chain_of("popoto_face_2row", sync=False, water_depth=100.0))
    wide = chain_of("popoto_face_2row", sync=False, water_depth=100.0)
    wide["elevation"].pop("sector_from_depth")
    wide = run("popoto_face_2row_shallow", wide)
    pick = lambda reps_: [max(r["detections"], key=lambda d: d["snr_db"]) for r in reps_]      # noqa: E731
    out["face_2row"] = {"true_depth_m": 60.0, "range_m": 3000.0, "true_elevation_deg": float(np.degrees(np.arctan2(-30.0, 2997.0))),
                        "depth_m": [d.get("depth_m") for d in pick(bounded)], "elevation_deg": [d["elevation_deg"] for d in pick(bounded)],
                        "ambiguous": [bool(d["elevation_ambiguous"]) for d in pick(bounded)],
                        "ambiguous_without_the_bound": [bool(d["elevation_ambiguous"]) for d in pick(wide)],
                        "candidates_without_the_bound": [len(d.get("elevation_alternatives_deg", [])) + 1 for d in pick(wide)]}
    # Stage 4: three rows. Free field, a shallow channel, and the full rehearsal.
    free = chain_of("popoto_vehicle_elevation", sync=False, gradient_sigma_per_s=0.0)
    last = run("popoto_elevation_free", free)[-1]
    truth = {(-300.0, 900.0): 10.0, (100.0, 1000.0): 60.0, (500.0, 1100.0): 160.0}
    errors = []
    for t in last["tracks"]:
        (_, depth) = min(truth.items(), key=lambda kv: np.hypot(kv[0][0] - t["x"], kv[0][1] - t["y"]))
        errors.append(t["depth_m"] - depth)
    out["free_field"] = {"tracks": len(last["tracks"]), "depth_errors_m": errors, "depth_sigma_m": [t["depth_sigma_m"] for t in last["tracks"]],
                         "roll_deg": 3.0, "pings": 5}
    best = pick(run("popoto_elevation_shallow", chain_of("popoto_vehicle_elevation", sync=False, water_depth=100.0)))
    out["shallow"] = {"true_depth_m": 50.0, "depth_m": [d["depth_m"] for d in best], "elevation_deg": [d["elevation_deg"] for d in best],
                      "true_elevation_deg": float(np.degrees(np.arctan2(-20.0, 1000.0))), "path": [d["path"] for d in best]}
    reps = run("popoto_elevation_rehearsal", load_yaml(ROOT / "configs" / "deploy" / "popoto_elevation_rehearsal.yaml")["chain"])
    deep = [near(r, [150.0, 1000.0, 60.0])[0] for r in reps]
    shallow_t = [near(r, [-250.0, 950.0, 8.0])[0] for r in reps]
    transmit = load_yaml(ROOT / "configs" / "deploy" / "popoto_elevation_rehearsal.yaml")["chain"]
    transmit["elevation"]["attitude_time"] = "transmit"
    deep_tx = [near(r, [150.0, 1000.0, 60.0])[0] for r in run("popoto_elevation_rehearsal", transmit)]
    out["rehearsal"] = {"sync_ok": all(r["sync"]["ok"] for r in reps), "worst_residual_us": max(r["sync"]["worst_residual_us"] for r in reps),
                        "deep_true_m": 60.0, "deep_depth_m": [d.get("depth_m") for d in deep],
                        "deep_depth_with_transmit_attitude_m": [d.get("image_depth_m") for d in deep_tx],
                        "shallow_true_m": 8.0, "shallow_depth_m": [d.get("depth_m") for d in shallow_t],
                        "roll_deg": 4.0, "pitch_deg": 1.0, "roll_rate_deg_s": 1.5, "boards": 12, "sound_speed": 1490.0}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--full", action="store_true", help="also one full-length ping (0.5-13 km at 102.4 kHz) per row count")
    args = ap.parse_args()
    R = {"cpu_threads": os.cpu_count(), "stream": {}, "timing_scenario_size": {}, "timing_full_length": {}}
    print("SCENARIOS: depth against truth, C++ chain")
    sc = scenario_check()
    R["scenarios"] = sc
    fmt = lambda v: ", ".join("none" if x is None else f"{x:+.1f}" if abs(x) < 10 else f"{x:.0f}" for x in v)      # noqa: E731
    a1, a2, f2, f, sh, re = sc["testauv_1row"], sc["testauv_2row"], sc["face_2row"], sc["free_field"], sc["shallow"], sc["rehearsal"]
    print(f"  stage 1, popoto_testauv_trial_1row (500 m, 4 boards, rolling): sync {a1['sync_residual_us']:.2f} us; range errors {fmt(a1['range_error_m'])} m; "
          f"bearing errors {fmt(a1['bearing_error_deg'])} deg")
    print(f"  stage 2, popoto_testauv_trial_2row (300 m, 8 boards, rolling, rows 0.30 m apart): sync {a2['sync_residual_us']:.2f} us; target at {a2['true_depth_m']:.0f} m read {fmt(a2['depth_m'])} m")
    print(f"  stage 3, popoto_face_2row_shallow (3 km, 100 m of water, rows 1.8 m apart): target at {f2['true_depth_m']:.0f} m read {fmt(f2['depth_m'])} m "
          f"(elevation {fmt(f2['elevation_deg'])} against {f2['true_elevation_deg']:+.2f} deg); ambiguous {f2['ambiguous']}; without the water-depth bound "
          f"{f2['ambiguous_without_the_bound']} with {f2['candidates_without_the_bound']} candidates")
    print(f"  stage 4, popoto_elevation_free (three rows, rolled {f['roll_deg']:.0f} deg, {f['pings']} pings): track depth errors {fmt(f['depth_errors_m'])} m; sigma from the angle {fmt(f['depth_sigma_m'])} m")
    print(f"  stage 4, popoto_elevation_shallow (target at {sh['true_depth_m']:.0f} m, direct path {sh['true_elevation_deg']:+.2f} deg): depth {fmt(sh['depth_m'])} m; "
          f"elevation {fmt(sh['elevation_deg'])} deg; path {', '.join(sh['path'])}")
    print(f"  stage 4, popoto_elevation_rehearsal ({re['boards']} boards, rolled {re['roll_deg']:.0f} deg and rolling {re['roll_rate_deg_s']} deg/s, {re['sound_speed']:.0f} m/s): sync "
          f"{re['worst_residual_us']:.2f} us; target at {re['deep_true_m']:.0f} m read {fmt(re['deep_depth_m'])} m (with the attitude of the transmit: {fmt(re['deep_depth_with_transmit_attitude_m'])} m); "
          f"target at {re['shallow_true_m']:.0f} m (direct and surface echoes merged) read {fmt(re['shallow_depth_m'])} m")
    print("STREAM: two pings at 102.4 kHz, real, 24 bit, 900-1000 m, through every transport")
    for rows in (1, 2, 3):
        s = stream_check(rows)
        R["stream"][str(rows)] = s
        print(f"  {rows} row(s), {s['channels']:2d} channels on {s['boards']:2d} boards: UDP blocks of {s['udp_block_samples']} samples "
              f"({s['udp_block_bytes']} bytes, {s['udp_blocks_per_second']:.0f} per second): assembler python {s['python_assembler_ok']}, "
              f"C++ {s['cpp_assembler_ok']}, loopback {s['udp_loopback_ok']} (lost {s['udp_lost_samples']}); Sensor Bridge frames of "
              f"{s['hsb_frame_bytes']} bytes {s['hsb_frames_ok']}; recording {s['recording_ok']}; chain {s['chain_seconds_per_ping_full_rate_short_window']:.2f} s "
              f"per ping, target found {s['chain_found_the_target']}")
    print("TIMING: C++ chain, host, per ping (median of pings 2-4), scenario size: baseband 8 kHz, 11.3-12.7 km, 241 bearings")
    over = {"receive": {"kind": "baseband", "sample_rate": 8.0e3, "window": {"range": [11300.0, 12700.0]}}}
    for rows in (1, 2, 3):
        _, recs = pings(rows, over, TARGETS, 4)
        t = time_chain(recs, rows)
        t["samples_per_channel"] = int(recs[0].samples.n_samples)
        R["timing_scenario_size"][str(rows)] = t
        print(f"  {rows} row(s): {t['seconds_per_ping']:.3f} s per ping; {t['detections']} detections, {t['with_elevation']} with an elevation")
    if args.full:
        print("TIMING: one full-length ping, 0.5-13 km at 102.4 kHz real (the deployment file's window)")
        for rows in (1, 2, 3):
            t0 = time.perf_counter()
            _, recs = pings(rows, {"receive": {"window": {"range": [500.0, 13000.0]}}}, [[740.0, 6000.0, 30.0]], 1, ts=40.0)
            t = time_chain(recs, rows)
            t.update(samples_per_channel=int(recs[0].samples.n_samples), raw_bytes=int(recs[0].samples.array.nbytes),
                     simulate_seconds=time.perf_counter() - t0 - t["seconds_per_ping"])
            R["timing_full_length"][str(rows)] = t
            print(f"  {rows} row(s): {t['seconds_per_ping']:.1f} s per ping, {t['raw_bytes'] / 1e6:.0f} MB of samples; {t['detections']} detections")
    if not args.no_write:
        path = ROOT / "docs" / "design" / "popoto_elevation_bench.json"
        if path.exists() and not args.full:
            R["timing_full_length"] = json.loads(path.read_text()).get("timing_full_length", {})   # keep an earlier --full run
        path.write_text(json.dumps(R, indent=2))
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
