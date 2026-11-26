#!/usr/bin/env python3
"""What the Sensor Bridge receive path costs per frame and per ping, without Sensor Bridge or Holoscan (sonar-h1f).

    tools/hsb_frame_timing.py                                  # 96 channels: per-frame cost of the frame path, 3 repeats
    tools/hsb_frame_timing.py --profile                        # ... and where it goes (cProfile, top functions)
    tools/hsb_frame_timing.py --schedule serial --engine python                 # HsbLiveApp's one-thread schedule, emulated
    tools/hsb_frame_timing.py --schedule serial --chain-seconds 2.64,2.91       # ... with a chain time given, not run
    tools/hsb_frame_timing.py --schedule threads --engine python                # what-if: the chain on a thread of its own
    tools/hsb_frame_timing.py --per-board                      # one stream per capture board, merged (sonar-bwu)

The stream is a Popoto build's shortest trial (`--rows` 1, 2 or 3: 32, 64 or 96 channels at 102.4 kHz, 4 pings 3 s
apart, boards on their own clocks), framed as `serve` frames it (`hsb.FrameStream`, 1 s lead) and received with the
build's deployment file, as tests/python/test_hsb.py's Popoto loopback does (`POPOTO` there; the same table here).
`--per-board`: every board's own stream instead (`serve --per-board`: frames of one board each, board k's capture started
k x 1031 samples before the first's), received with `source.boards: {}`, so the frame path merges them; a frame then
carries one board's channels and the frames per second are the boards' together.

What it measures, and what it cannot:
- The frame path: `hsb.FrameReceiver.take` (the frame-number bookkeeping and the ping assembler) on every frame, with
  the metadata `HsbFramesToPingsOp` passes. Not measured: Sensor Bridge's `LinuxReceiverOp.compute`, the copy of the
  frame from GPU memory (`frame_to_host`), reading the frame's Holoscan metadata and Holoscan's own per-operator cost;
  `--extra-frame-ms` adds a guessed amount for them to the schedules.
- `--schedule serial` emulates `HsbLiveApp` as it runs: Holoscan's default greedy scheduler runs every operator on one
  thread, so while the chain works on a ping no frame is taken and Sensor Bridge's receiver thread (C++, not Python)
  queues them, dropping the oldest beyond `--queue-s` seconds of stream (`receive --queue`; 4 s by default, D-187). The
  frames arrive on a virtual clock at the stream's rate; the receiver and the chain (`--engine`, run for real on this
  host's backend: CUDA where there is a GPU) take their measured time. Frames the queue dropped never reach the
  receiver, which counts them as it would. The socket's own losses are not emulated.
- `--schedule threads` is not what the application does: frames taken at real time on one thread while the chain runs
  on another, to show what a scheduler with two worker threads would face (the two share Python's GIL).

Prints one JSON line per result (FRAME_TIMING, PROFILE, SCHEDULE, THREADS). Simulating the 3-row trial takes about
10 s; nothing is written to disk.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

# rows -> (scenario, sonar it is received as or None, deployment file): tests/python/test_hsb.py POPOTO.
POPOTO = {1: ("popoto_testauv_trial_1row", None, "popoto_testauv_1row"),
          2: ("popoto_static_trial_3row", "popoto_static_2row_clocks", "popoto_static_2row"),
          3: ("popoto_static_trial_3row", None, "popoto_static_3row")}
LEAD = 1.0
DEFAULT_QUEUE_S = 4.0          # hsb.QUEUE_SECONDS


def popoto_stream(rows: int):
    """(records, deployment dict with the trial's sonar and sound speed) as test_hsb.popoto_trial makes them."""
    from gpusonar.config import load_scenario_config, load_yaml
    from gpusonar.sim import run_scenario
    scenario, sonar, deploy = POPOTO[rows]
    sc = load_scenario_config(scenario)
    if sonar:
        sc = dict(sc, sonar=sonar, name=f"{sc['name']}_as_{sonar}")
    records = run_scenario(sc).records
    cfg = load_yaml(str(ROOT / "configs" / "deploy" / f"{deploy}.yaml"))
    cfg["source"] = dict(cfg["source"], sonar=sc["sonar"], sonar_overrides=sc.get("sonar_overrides"),
                         sound_speed=float(records[0].environment.sound_speed))
    if (cfg.get("nav") or {}).get("port"):
        cfg.pop("nav")
    cfg.pop("output", None)
    return records, cfg


def frames_of(records, frame_samples: int, per_board: bool = False):
    """(frames as host arrays of bytes, as frame_to_host gives them; frames a second; seconds between transmits).
    `per_board`: one stream per capture board, as `sonar_live_hsb.py serve --per-board` sends them."""
    from gpusonar.sources import hsb
    options = {}
    if per_board:
        sys.path.insert(0, str(ROOT / "tools"))
        from sonar_live import PER_BOARD_START_STEP
        from sonar_live_hsb import board_map
        boards = board_map(records[0])
        options = {"boards": boards, "start_offsets": [-PER_BOARD_START_STEP * k for k in range(len(boards))]}
    st = hsb.FrameStream(records, frame_samples, lead_seconds=LEAD, **options)
    frames = [np.frombuffer(f, dtype=np.uint8).copy() for f in st]
    interval = (records[1].transmit.time_ns - records[0].transmit.time_ns) * 1e-9 if len(records) > 1 else float("nan")
    return frames, 1.0 / st.frame_seconds, interval


def receiver_for(cfg: dict, frame_samples: int):
    from gpusonar.sources import hsb
    src = cfg["source"]
    return hsb.FrameReceiver(src["sonar"], src.get("sonar_overrides"), sound_speed=src["sound_speed"],
                             frame_samples=frame_samples, boards=src.get("boards"))


def _metadata(k: int, frame) -> dict:
    return {"frame_number": k, "frame_bytes_received": len(frame), "packets_dropped": 0}


def summary_ms(seconds) -> dict:
    x = np.asarray(seconds, dtype=float) * 1e3
    if not len(x):
        return {}
    return {"mean": round(float(x.mean()), 4), "p50": round(float(np.median(x)), 4), "p99": round(float(np.percentile(x, 99)), 4),
            "max": round(float(x.max()), 3), "n": int(len(x))}


def frame_path_timing(frames, cfg: dict, frame_samples: int, fps: float, interval: float, repeats: int = 3) -> list[dict]:
    """Per repeat: `take` on every frame in order, timed one by one; the frames that completed a ping apart."""
    out = []
    for rep in range(repeats):
        rx = receiver_for(cfg, frame_samples)
        took, ping_frames = np.empty(len(frames)), []
        for k, frame in enumerate(frames):
            t0 = time.perf_counter()
            got = rx.take(frame, _metadata(k, frame))
            took[k] = time.perf_counter() - t0
            if got:
                ping_frames.append(k)
        rx.finish()
        other = np.delete(took, ping_frames)
        per_interval = float(took.sum()) / len(frames) * fps * interval
        out.append({"repeat": rep, "frames": len(frames), "pings": rx.pings, "frame_bytes": int(len(frames[0])),
                    "frames_per_s": fps, "all_ms": summary_ms(took), "without_ping_frames_ms": summary_ms(other),
                    "ping_frames_ms": summary_ms(took[ping_frames]), "total_s": round(float(took.sum()), 4),
                    "seconds_a_ping_interval": round(per_interval, 4), "budget_ms_a_frame": round(1e3 / fps, 3)})
    return out


def profile_frame_path(frames, cfg: dict, frame_samples: int, top: int = 15) -> list:
    import cProfile
    import pstats
    rx = receiver_for(cfg, frame_samples)
    pr = cProfile.Profile()
    pr.enable()
    for k, frame in enumerate(frames):
        rx.take(frame, _metadata(k, frame))
    rx.finish()
    pr.disable()
    st = pstats.Stats(pr)
    rows = sorted(((v[2], v[3], v[0], f"{Path(k[0]).name}:{k[1]}({k[2]})") for k, v in st.stats.items()), reverse=True)
    return [{"function": name, "tottime_s": round(tt, 4), "cumtime_s": round(ct, 4), "calls": n}
            for tt, ct, n, name in rows[:top]]


def make_chain(cfg: dict, engine: str):
    from gpusonar.live import LiveChain, NavListener
    nav = NavListener.from_config(cfg["nav"]) if cfg.get("nav") else None
    return LiveChain(dict(cfg["chain"], engine=engine), nav=nav)


def serial_schedule(frames, cfg: dict, frame_samples: int, fps: float, queue_frames: int, chain=None,
                    chain_seconds: list[float] | None = None, extra_frame_s: float = 0.0) -> dict:
    """HsbLiveApp's one-thread schedule on a virtual clock. Frame k arrives at k / fps; the one thread takes the frames in
    order (`take`, measured, plus `extra_frame_s`) and runs the chain on each ping as it completes (`chain`, measured,
    or `chain_seconds` in turn). The receiver's queue holds the newest `queue_frames`: older frames not yet taken are
    dropped and never reach the receiver."""
    rx = receiver_for(cfg, frame_samples)
    t, k, n = 0.0, 0, len(frames)
    dropped, peak, took, chain_s, wait = 0, 0, [], [], 0.0
    while k < n:
        if t < k / fps:
            wait += k / fps - t
            t = k / fps
        newest = min(int(t * fps + 1e-9), n - 1)
        if k < newest - queue_frames + 1:
            dropped += newest - queue_frames + 1 - k
            k = newest - queue_frames + 1
        peak = max(peak, newest - k + 1)
        t0 = time.perf_counter()
        records = rx.take(frames[k], _metadata(k, frames[k]))
        d = time.perf_counter() - t0 + extra_frame_s
        took.append(d)
        t += d
        for record in records:
            if chain_seconds:
                c = float(chain_seconds[min(len(chain_s), len(chain_seconds) - 1)])
            else:
                c0 = time.perf_counter()
                chain.process(record)
                c = time.perf_counter() - c0
            chain_s.append(round(c, 3))
            t += c
        k += 1
    rx.finish()
    st = rx.stats()
    return {"queue_frames": queue_frames, "frames": n, "frames_dropped_by_the_queue": dropped, "peak_backlog_frames": peak,
            "frames_lost_by_the_receiver": st["frames_lost"], "pings": st["pings"], "dropped_pings": st["dropped_pings"],
            "chain_s": chain_s, "frame_ms": summary_ms(took), "extra_frame_ms": 1e3 * extra_frame_s,
            "busy_s": round(t - wait, 3), "stream_s": round(n / fps, 3), "idle_s": round(wait, 3)}


def threaded_schedule(frames, cfg: dict, frame_samples: int, fps: float, chain) -> dict:
    """Not what HsbLiveApp does: frames taken at real time on this thread while `chain` runs each completed ping on a
    thread of its own (the GIL shared). Reports how late the frames were taken and what `take` cost meanwhile."""
    import queue
    rx = receiver_for(cfg, frame_samples)
    pings: queue.Queue = queue.Queue()
    chain_s, busy = [], threading.Event()

    def work():
        while True:
            record = pings.get()
            if record is None:
                return
            busy.set()
            c0 = time.perf_counter()
            chain.process(record)
            chain_s.append(round(time.perf_counter() - c0, 3))
            busy.clear()

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    late, took_busy, took_idle = [], [], []
    t0 = time.perf_counter()
    for k, frame in enumerate(frames):
        due = t0 + k / fps
        now = time.perf_counter()
        if now < due:
            time.sleep(due - now)
        start = time.perf_counter()
        late.append(start - due)
        during = busy.is_set()
        for record in rx.take(frame, _metadata(k, frame)):
            pings.put(record)
        (took_busy if during else took_idle).append(time.perf_counter() - start)
    pings.put(None)
    worker.join()
    rx.finish()
    return {"frames": len(frames), "late_ms": summary_ms(late), "frame_ms_while_the_chain_ran": summary_ms(took_busy),
            "frame_ms_otherwise": summary_ms(took_idle), "chain_s": chain_s, "pings": rx.pings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, choices=sorted(POPOTO), default=3)
    ap.add_argument("--frame-samples", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--per-board", action="store_true", help="one stream per capture board, merged by the frame path")
    ap.add_argument("--profile", action="store_true", help="cProfile of one pass of the frame path")
    ap.add_argument("--schedule", choices=["serial", "threads"], help="emulate the application's schedule (serial) or a "
                    "two-thread one (threads)")
    ap.add_argument("--engine", choices=["python", "cpp"], default="python", help="the chain the schedules run")
    ap.add_argument("--chain-seconds", help="serial: chain seconds per ping, comma-separated (the last repeats), instead "
                    "of running the chain")
    ap.add_argument("--queue-s", type=float, default=DEFAULT_QUEUE_S, help="seconds of stream the receiver queue holds")
    ap.add_argument("--extra-frame-ms", type=float, default=0.0, help="serial: ms added to every frame for what is not "
                    "measured here (receiver operator, copy from the GPU, metadata, Holoscan)")
    args = ap.parse_args(argv)
    t = time.perf_counter()
    records, cfg = popoto_stream(args.rows)
    if args.per_board:
        cfg["source"]["boards"] = dict(cfg["source"].get("boards") or {})
    frames, fps, interval = frames_of(records, args.frame_samples, args.per_board)
    print("STREAM " + json.dumps({"rows": args.rows, "channels": records[0].samples.n_channels, "per_board": args.per_board, "frames": len(frames),
                                  "frame_bytes": int(len(frames[0])), "frames_per_s": fps, "interval_s": interval,
                                  "simulated_s": round(time.perf_counter() - t, 1)}), flush=True)
    for row in frame_path_timing(frames, cfg, args.frame_samples, fps, interval, args.repeats):
        print("FRAME_TIMING " + json.dumps(row), flush=True)
    if args.profile:
        print("PROFILE " + json.dumps(profile_frame_path(frames, cfg, args.frame_samples)), flush=True)
    if args.schedule == "serial":
        import math
        queue = int(math.ceil(args.queue_s * fps))
        given = [float(x) for x in args.chain_seconds.split(",")] if args.chain_seconds else None
        chain = None if given else make_chain(cfg, args.engine)
        try:
            out = serial_schedule(frames, cfg, args.frame_samples, fps, queue, chain, given, args.extra_frame_ms * 1e-3)
        finally:
            if chain is not None:
                chain.close()
        out.update(engine=None if given else args.engine, chain_given=bool(given))
        print("SCHEDULE " + json.dumps(out), flush=True)
    elif args.schedule == "threads":
        with make_chain(cfg, args.engine) as chain:
            out = threaded_schedule(frames, cfg, args.frame_samples, fps, chain)
        print("THREADS " + json.dumps(dict(out, engine=args.engine)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
