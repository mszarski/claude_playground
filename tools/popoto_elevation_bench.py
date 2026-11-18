#!/usr/bin/env python3
"""Stacked rows on the long-range array: does the data path carry 64 and 96 channels, and
what does the chain cost per ping with one, two and three rows?

  stream   one ping at the boards' rate (102.4 kHz, real, 24 bit) with 32, 64 and 96 channels
           through each transport: board blocks over UDP (sources/flank_stream.py), the same
           blocks through the C++ assembler (libs/live), fixed-size Sensor Bridge frames
           (sources/hsb.py), and the container file. Every transport must give the samples
           back within half a count of the 24-bit quantisation.
  scenarios  every stage's scenarios through the C++ chain: the test AUV's row; the static drop-in
           array (two rows, three rows, free field, a passing target, listening); the face (two
           rows in shallow water; three rows in free field with the vehicle rolled, in a shallow
           channel with bounces, and in the rehearsal with boards on their own clocks).
  timing   the C++ chain (host) per ping on pings of the `popoto_12km` scenario's size (complex
           baseband at 8 kHz, 11.3-12.7 km, bearings -60 to 60 degrees in 0.5 degree steps),
           and at the boards' full rate on a short window, for one, two and three rows.
           Without --full the full-length ping (0.5 to 13 km at 102.4 kHz) is not run: with
           96 channels it is 0.7 GB of samples before processing.

  --full-rate  only this: one full-length ping at the boards' rate (0.5 to 13 km at 102.4 kHz, real) per row
           count through the C++ chain on the chosen backend, three ways: as it is (the chain converts
           to baseband itself), through the decimate-first stage (libs/live decimate.hpp) at its
           default rate, and at a 25th of the board rate. Also the stage alone, the streaming
           receiver, and the sizes of a raw and a decimated recording. Writes the `decimation.timing`
           section of docs/design/popoto_stage2.json under the backend's name. On a GPU:
               PYTHONPATH=python python3 tools/popoto_elevation_bench.py --full-rate --backend cuda

Writes docs/design/popoto_elevation_bench.json (read by tools/design_report.py). CPU only unless --backend says otherwise.
Usage: tools/popoto_elevation_bench.py [--no-write] [--full] [--full-rate [--backend host|cuda|auto] [--rows 1,2,3] [--clocks]]
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


CLOCKED = {2: "popoto_long_range_2row_clocks", 3: "popoto_long_range_3row_clocks"}


def machine_load():
    try:
        return [float(v) for v in os.getloadavg()]
    except OSError:
        return None


def full_rate_case(rows, backend, clocks=False, window=(500.0, 13000.0)):
    """One full-length raw ping through the chain as it is and through the decimate-first stage."""
    from gpusonar.live import sync_reference
    cfg = load_yaml(ROOT / "configs" / "deploy" / ("popoto_vehicle.yaml" if rows == 1 else "popoto_vehicle_elevation.yaml"))["chain"]
    over = {"receive": {"window": {"range": list(window)}}}
    t0 = time.perf_counter()
    if clocks and rows in CLOCKED:
        sonar = SonarModel.load(CLOCKED[rows], over)
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 10.0), velocity=(1.5, 0.0, 0.0)), Scene().add_points([[740.0, 6000.0, 30.0]], target_strength_db=40.0),
                            environment={"sound_speed": 1500.0}, time_ns=0, noise_rms=2e-6, rng=np.random.default_rng(3), adc=False)
        reference = sync_reference(cfg)
    else:
        clocks = False
        cfg.pop("sync", None)                                  # ideal clocks
        sonar, recs = pings(rows, over, [[740.0, 6000.0, 30.0]], 1, ts=40.0)
        rec, reference = recs[0], None
    simulate_s = time.perf_counter() - t0
    s = rec.samples
    out = {"channels": int(s.n_channels), "boards": len(sonar.boards), "board_clocks": bool(clocks), "samples_per_channel": int(s.n_samples),
           "sample_rate_hz": float(s.sample_rate), "window_m": list(window), "simulate_seconds": simulate_s, "load_average_before": machine_load()}
    options = cpp_chain_options(cfg, reference)
    want = core.Backend.Cuda if backend == "cuda" or (backend == "auto" and core.cuda_available()) else core.Backend.Host
    out["backend"] = "cuda" if want == core.Backend.Cuda else "host"

    def chain_time(decimate=None, record=None):
        o = dict(options, backend=want)
        if decimate is not None:
            o["decimate"] = decimate
        seconds = []
        for _ in range(2):                                     # the shorter of two runs: the machine is shared
            chain = core.LiveChainCpp(o)
            r = (record or rec).copy()
            report = json.loads(chain.process(r))
            seconds.append(report["processing_seconds"])
            del r, chain
        best = max(report["detections"], key=lambda d: d["snr_db"]) if report["detections"] else None
        return {"seconds": min(seconds), "all_seconds": seconds, "detections": len(report["detections"]),
                "strongest": {k: best[k] for k in ("range_m", "bearing_deg", "snr_db", "depth_m") if k in best} if best else None}

    out["chain_full_rate"] = chain_time()
    out["chain_decimate_first"] = chain_time({})
    out["chain_decimate_first_factor_25"] = chain_time({"output_rate": s.sample_rate / 25.0})
    # The stage on its own: all channels at once (a recording replayed), and streaming, block by block, as a receiver does it.
    plan = core.live_decimate_plan(rec, {})
    r = rec.copy()
    t0 = time.perf_counter()
    core.live_decimate_record(r, {}, reference)
    out["stage_whole_ping_seconds"] = time.perf_counter() - t0
    out["chain_on_decimated_records"] = chain_time(record=r)          # what a decimating receiver leaves the chain to do
    out.update(factor=int(plan["factor"]), output_rate_hz=plan["output_rate"], taps=len(plan["taps"]), latency_ms=1e3 * plan["delay_s"],
               decimated_samples_per_channel=int(r.samples.n_samples), raw_bytes_in_memory=int(np.asarray(rec.samples.array).nbytes),
               decimated_bytes_in_memory=int(np.asarray(r.samples.array).nbytes))
    dec = core.PingDecimator(rec, {}, reference)
    block = max(8, (128 * 32 // s.n_channels) // 2 * 2)
    n_try = min(s.n_samples, int(2.0 * s.sample_rate))                # two seconds of stream
    rows_ = np.ascontiguousarray(np.asarray(rec.samples.array)[:, :n_try].T)
    t0 = time.perf_counter()
    for a in range(0, n_try, block):
        dec.feed(rows_[a:a + block])
    spent = time.perf_counter() - t0
    out["stage_streaming_seconds_per_second_of_stream"] = spent / (n_try / s.sample_rate)
    out["stage_streaming_block_samples"] = block
    del rows_, dec
    with tempfile.TemporaryDirectory() as tmp:
        for label, record in (("raw", rec), ("decimated", r)):
            path = os.path.join(tmp, f"{label}.h5")
            w = gs.RecordWriter(path)
            t0 = time.perf_counter()
            w.write(record)
            w.close()
            out[f"recording_{label}_bytes"] = int(os.path.getsize(path))
            out[f"recording_{label}_write_seconds"] = time.perf_counter() - t0
    out["load_average_after"] = machine_load()
    return out


def full_rate(args):
    path = ROOT / "docs" / "design" / "popoto_stage2.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    timing = R.setdefault("decimation", {}).setdefault("timing", {})
    print(f"FULL RATE: one ping, 0.5-13 km at the boards' rate, C++ chain; load average {machine_load()}, {os.cpu_count()} threads")
    for rows in [int(v) for v in args.rows.split(",")]:
        t = full_rate_case(rows, args.backend, clocks=args.clocks)
        key = f"{t['backend']}{'_clocks' if t['board_clocks'] else ''}"
        timing.setdefault(key, {})[str(rows)] = t
        a, b, c, d = (t[k]["seconds"] for k in ("chain_full_rate", "chain_decimate_first", "chain_decimate_first_factor_25", "chain_on_decimated_records"))
        print(f"  {rows} row(s), {t['channels']} ch{', board clocks' if t['board_clocks'] else ''}, {t['backend']}: chain {a:.1f} s; decimate first {b:.1f} s "
              f"({a / b:.1f}x); at a 25th {c:.1f} s ({a / c:.1f}x); stage alone {t['stage_whole_ping_seconds']:.1f} s, chain after it {d:.1f} s; "
              f"streaming stage {t['stage_streaming_seconds_per_second_of_stream']:.2f} s per second of stream; recording "
              f"{t['recording_raw_bytes'] / 1e6:.0f} -> {t['recording_decimated_bytes'] / 1e6:.0f} MB; strongest "
              f"{t['chain_full_rate']['strongest']} / {t['chain_decimate_first']['strongest']}", flush=True)
        timing[key]["cpu_threads"] = os.cpu_count()
        if not args.no_write:
            path.write_text(json.dumps(R, indent=2))
    if not args.no_write:
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


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
        chain["tracker"] = dict(chain.get("tracker") or {}, confirm_hits=2)
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
    # The static drop-in array: its two trials (raw samples, board clocks), free field, a passing target, listening.
    from gpusonar.config import load_scenario_config
    for key, scenario, deploy in (("static_2row", "popoto_static_trial_2row", "popoto_static_2row"),
                                  ("static_3row", "popoto_static_trial_3row", "popoto_static_3row")):
        sc_ = load_scenario_config(scenario)
        target = sc_["scene"][0]["positions"][0]
        reps = run(scenario, chain_of(deploy, water_depth=float(sc_["environment"]["water_depth"])))
        rows = [near(r, target)[0] for r in reps]
        own = np.asarray(reps[0]["ownship_xyz"])
        out[key] = {"sync_residual_us": max(r["sync"]["worst_residual_us"] for r in reps), "true_depth_m": float(target[2]),
                    "range_m": float(np.linalg.norm(np.asarray(target) - own)), "water_depth_m": float(sc_["environment"]["water_depth"]),
                    "boards": len(reps[0]["sync"]["offsets_us"]),
                    "depth_m": [d.get("depth_m") for d in rows], "depth_sigma_m": [d.get("depth_sigma_m") for d in rows],
                    "ambiguous": [bool(d.get("elevation_ambiguous")) for d in rows], "path": [d.get("path") for d in rows]}
    # The same target depth further out in that shallow water: the bounces arrive inside the pulse's range sidelobes.
    sc_ = load_scenario_config("popoto_static_trial_3row")
    sc_["scene"][0]["positions"] = [[60.0, 294.0, 22.0]]
    sc_["sonar_overrides"]["receive"]["window"]["range"] = [100, 700]
    with LiveChain(dict(chain_of("popoto_static_3row", water_depth=30.0), engine="cpp", backend="host")) as live:
        reps = live.run(run_scenario(sc_).records)
    rows = [near(r, [60.0, 294.0, 22.0])[0] for r in reps]
    out["static_3row_300m_shallow"] = {"true_depth_m": 22.0, "range_m": 300.0, "water_depth_m": 30.0, "depth_m": [d.get("depth_m") for d in rows],
                                       "path": [d.get("path") for d in rows]}
    static = chain_of("popoto_static_3row", sync=False, gradient_sigma_per_s=0.0)
    last = run("popoto_static_free", static)[-1]
    truth = {tuple(o["positions"][0][:2]): float(o["positions"][0][2]) for o in load_scenario_config("popoto_static_free")["scene"]}
    errors = []
    for t in last["tracks"]:
        (_, depth) = min(truth.items(), key=lambda kv: np.hypot(kv[0][0] - t["x"], kv[0][1] - t["y"]))
        errors.append(t["depth_m"] - depth)
    out["static_free"] = {"tracks": len(last["tracks"]), "true_depths_m": sorted(truth.values()), "depth_errors_m": errors,
                          "depth_sigma_m": [t["depth_sigma_m"] for t in last["tracks"]], "tilt_deg": 2.0}
    sc_ = load_scenario_config("popoto_static_passing")
    mover = sc_["scene"][0]
    last = run("popoto_static_passing", chain_of("popoto_static_3row", sync=False, water_depth=float(sc_["environment"]["water_depth"])))[-1]
    movers = [t for t in last["tracks"] if t["kind"] == "mover"]
    out["static_passing"] = {"true_speed_m_s": float(np.linalg.norm(mover["velocity"])), "true_depth_m": float(mover["positions"][0][2]),
                             "pings": int(sc_["pings"]["count"]), "tracks": len(last["tracks"]), "kinds": sorted(t["kind"] for t in last["tracks"]),
                             "mover_speed_m_s": [t["speed_m_s"] for t in movers], "mover_vx_m_s": [t["vx"] for t in movers],
                             "mover_depth_m": [t.get("depth_m") for t in movers]}
    # Listening: two noise sources heard by the three-row static array (sim/passive.py: free field, no ping).
    from gpusonar.sim.passive import passive_record
    sonar = SonarModel.load("popoto_static_3row", {"receive": {"kind": "baseband", "sample_rate": 8.0e3, "window": {"range": [0.0, 1500.0]}}})
    pose = make_pose(position=(0.0, 0.0, 40.0), rpy=(np.radians(2.0), 0.0, 0.0))
    sources = [[-200.0, 500.0, 10.0], [150.0, 450.0, 110.0]]
    rec = passive_record(sonar, pose, [{"position": sources[0], "rms": 1.0}, {"position": sources[1], "rms": 0.7}], noise_rms=1.0,
                         rng=np.random.default_rng(2))
    frames, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=256, hop=128, n_average=12, first_deg=-60.0, last_deg=60.0,
                                 max_sources=2, min_snr_db=3.0, elevation=True, elevation_step_deg=0.25)
    centre = rec.rx_positions().mean(axis=0)
    true_el = [float(np.degrees(np.arcsin(-(np.asarray(q) - centre)[2] / np.linalg.norm(np.asarray(q) - centre)))) for q in sources]
    got = [sorted(float(np.degrees(e)) for e in f["elevations"]) for f in frames if len(f["peaks"]) == 2]
    out["static_passive"] = {"frames": len(frames), "frames_with_both": len(got), "true_elevation_deg": sorted(true_el),
                             "worst_elevation_error_deg": float(max(max(abs(a - b) for a, b in zip(g, sorted(true_el))) for g in got)) if got else None,
                             "source_rms_over_noise": [1.0, 0.7], "tilt_deg": 2.0}
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
    ap.add_argument("--full-rate", action="store_true", help="only the full-length ping at the boards' rate, with and without the decimate-first stage")
    ap.add_argument("--backend", default="host", choices=["host", "cuda", "auto"], help="for --full-rate")
    ap.add_argument("--rows", default="1,2,3", help="for --full-rate: row counts, comma separated")
    ap.add_argument("--clocks", action="store_true", help="for --full-rate: boards on their own clocks, synchronised (two and three rows)")
    args = ap.parse_args()
    if args.full_rate:
        return full_rate(args)
    R = {"cpu_threads": os.cpu_count(), "stream": {}, "timing_scenario_size": {}, "timing_full_length": {}}
    print("SCENARIOS: depth against truth, C++ chain")
    sc = scenario_check()
    R["scenarios"] = sc
    fmt = lambda v: ", ".join("none" if x is None else f"{x:+.1f}" if abs(x) < 10 else f"{x:.0f}" for x in v)      # noqa: E731
    a1, f2, f, sh, re = sc["testauv_1row"], sc["face_2row"], sc["free_field"], sc["shallow"], sc["rehearsal"]
    print(f"  test AUV, popoto_testauv_trial_1row (500 m, 4 boards, rolling): sync {a1['sync_residual_us']:.2f} us; range errors {fmt(a1['range_error_m'])} m; "
          f"bearing errors {fmt(a1['bearing_error_deg'])} deg")
    for key in ("static_2row", "static_3row"):
        a2 = sc[key]
        print(f"  static array, popoto_static_trial_{key[-4:]} ({a2['range_m']:.0f} m, {a2['water_depth_m']:.0f} m of water, {a2['boards']} boards, tilted 2 deg): sync {a2['sync_residual_us']:.2f} us; "
              f"target at {a2['true_depth_m']:.0f} m read {fmt(a2['depth_m'])} m (sigma {fmt(a2['depth_sigma_m'])} m); ambiguous {a2['ambiguous']}; path {a2['path']}")
    a3 = sc["static_3row_300m_shallow"]
    print(f"  static array, three rows, the target at {a3['true_depth_m']:.0f} m depth moved out to {a3['range_m']:.0f} m in {a3['water_depth_m']:.0f} m of water: read {fmt(a3['depth_m'])} m, path {a3['path']}")
    sf, sp, sl_ = sc["static_free"], sc["static_passing"], sc["static_passive"]
    print(f"  static array, popoto_static_free (targets at {sf['true_depths_m']} m): track depth errors {fmt(sf['depth_errors_m'])} m; sigma from the angle {fmt(sf['depth_sigma_m'])} m")
    print(f"  static array, popoto_static_passing ({sp['true_speed_m_s']:.1f} m/s at {sp['true_depth_m']:.0f} m, {sp['pings']} pings): tracks {sp['kinds']}; mover speed {fmt(sp['mover_speed_m_s'])} m/s, "
          f"along the array {fmt(sp['mover_vx_m_s'])} m/s, depth {fmt(sp['mover_depth_m'])} m")
    print(f"  static array, listening (two noise sources, elevations {fmt(sl_['true_elevation_deg'])} deg): both found in {sl_['frames_with_both']} of {sl_['frames']} frames; "
          f"worst elevation error {sl_['worst_elevation_error_deg']:.2f} deg")
    print(f"  face, popoto_face_2row_shallow (3 km, 100 m of water, rows 1.5 m apart): target at {f2['true_depth_m']:.0f} m read {fmt(f2['depth_m'])} m "
          f"(elevation {fmt(f2['elevation_deg'])} against {f2['true_elevation_deg']:+.2f} deg); ambiguous {f2['ambiguous']}; without the water-depth bound "
          f"{f2['ambiguous_without_the_bound']} with {f2['candidates_without_the_bound']} candidates")
    print(f"  face, popoto_elevation_free (three rows, rolled {f['roll_deg']:.0f} deg, {f['pings']} pings): track depth errors {fmt(f['depth_errors_m'])} m; sigma from the angle {fmt(f['depth_sigma_m'])} m")
    print(f"  face, popoto_elevation_shallow (target at {sh['true_depth_m']:.0f} m, direct path {sh['true_elevation_deg']:+.2f} deg): depth {fmt(sh['depth_m'])} m; "
          f"elevation {fmt(sh['elevation_deg'])} deg; path {', '.join(sh['path'])}")
    print(f"  face, popoto_elevation_rehearsal ({re['boards']} boards, rolled {re['roll_deg']:.0f} deg and rolling {re['roll_rate_deg_s']} deg/s, {re['sound_speed']:.0f} m/s): sync "
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
