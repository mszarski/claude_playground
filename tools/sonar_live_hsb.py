#!/usr/bin/env python3
"""The long-range chain fed through NVIDIA Holoscan Sensor Bridge, as two processes.

    tools/sonar_live_hsb.py serve --scenario flank_patrol          # the board: simulated pings sent through
    tools/sonar_live_hsb.py serve --replay flight.h5               # Sensor Bridge's emulator (or a recording)
    tools/sonar_live_hsb.py receive CONFIG --pings 20 --report r.jsonl      # the host: Sensor Bridge's Linux
    tools/sonar_live_hsb.py receive --sonar flank_32 --chain '{...}' ...    # receiver -> ping records -> chain

`serve` announces a board at --ip (127.0.0.1: both on one machine) and streams when `receive`
connects; it needs root (the emulator's data socket is a raw socket). `receive` needs a CUDA
device. CONFIG is a deployment file like configs/deploy/popoto_vehicle.yaml: its source names the
sonar, its chain, nav and output sections are used as by tools/sonar_live.py. Both sides must use
the same --frame-samples. Sensor Bridge itself: tools/build_hsb.sh.
`serve --scenario` simulates every ping before it announces the board; `receive` looks for the board
for --find-timeout seconds (default 30), so start it after serve says it is waiting for a receiver,
or give it longer.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from gpusonar.config import load_yaml  # noqa: E402
from gpusonar.sources import hsb, make_source  # noqa: E402
from sonar_live import ping_line  # noqa: E402


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text}: at least 1")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not value > 0.0:
        raise argparse.ArgumentTypeError(f"{text}: more than 0")
    return value


STOPPED_SHORT = 3   # receive's exit status when it stopped before reporting --pings pings (the statistics are printed)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="role", required=True)
    s = sub.add_parser("serve", help="send pings as the board, through Sensor Bridge's emulator")
    what = s.add_mutually_exclusive_group(required=True)
    what.add_argument("--scenario", help="a scenario to simulate and send")
    what.add_argument("--replay", help="a container file of ping records to send")
    s.add_argument("--ip", default="127.0.0.1", help="address of the interface the emulated board lives on")
    s.add_argument("--pace", type=float, default=1.0, help="multiple of real time")
    s.add_argument("--lead", type=float, default=1.0, help="seconds of stream before the first transmit")
    s.add_argument("--linger", type=float, default=10.0, help="seconds to keep streaming after the last ping")
    s.add_argument("--hold", type=float, default=0.0,
                   help="seconds the board stays up, quiet, after the linger when the receiver is still connected "
                        "(a real board that stops streaming stays on the network)")
    s.add_argument("--interval", type=float, help="seconds between transmits (default: the records' transmit times)")
    s.add_argument("--connect-timeout", type=float, default=120.0)
    s.add_argument("--catch-up", choices=hsb.CATCH_UP, default="spread",
                   help="after the sender falls behind: send the frames it owes at most two per frame period (spread, "
                        "the default) or back to back (burst, the behaviour before)")
    s.add_argument("--send-log", help="JSON file of every frame's send time")
    r = sub.add_parser("receive", help="receive with Sensor Bridge's LinuxReceiverOp and run the chain")
    r.add_argument("config", nargs="?", help="deployment YAML (source.sonar, chain, nav, output)")
    r.add_argument("--hololink", default="127.0.0.1", help="the board's address")
    r.add_argument("--sonar", help="sonar configuration of the stream (overrides the file's)")
    r.add_argument("--sonar-overrides", help="JSON")
    r.add_argument("--chain", help="chain settings, JSON or a YAML file; merged over the file's chain section")
    r.add_argument("--engine", choices=["python", "cpp"], help="override chain.engine")
    r.add_argument("--pings", type=int, help="stop once this many pings are accounted for, reported or dropped (default: "
                   "when the stream goes quiet); fewer reported exits with status 3 after printing the statistics")
    r.add_argument("--idle", type=float, default=15.0, help="seconds without a frame before stopping, whatever was lost")
    r.add_argument("--find-timeout", type=positive_float, default=30.0, metavar="S",
                   help="seconds to look for the board before giving up (default 30). A board that comes up later, "
                        "such as `serve --scenario` still simulating its pings, needs more")
    r.add_argument("--report", help="JSON lines file, one report per ping")
    r.add_argument("--recording", help="container file of the raw pings")
    r.add_argument("--queue", type=int, default=256, help="frames the receiver holds while the chain is busy")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--receiver-core", type=int, help="core for Sensor Bridge's receiver thread (its default is core 2)")
    r.add_argument("--chain-omp-threads", type=positive_int, metavar="N",
                   help="OpenMP threads for the chain's parallel stages (default: left as OpenMP has it, one per core "
                        "unless OMP_NUM_THREADS says otherwise). Applied with omp_set_num_threads on the thread that runs "
                        "the chain, before its first ping, so before the chain's first parallel region; it sets the count "
                        "for parallel regions that thread starts, other threads are not changed. Below the core count it "
                        "leaves cores to Sensor Bridge's receiver thread (DEPLOYMENT.md)")
    r.add_argument("--socket-buffer", type=int, help="receive buffer of the data socket in bytes; the kernel doubles it, up to twice rmem_max (DEPLOYMENT.md advises at least rmem_max)")
    r.add_argument("--arrival-log", help="JSON file of every frame's arrival (frame number, receive time, packets, bytes)")
    for p in (s, r):
        p.add_argument("--frame-samples", type=int, default=hsb.FRAME_SAMPLES, help="samples per frame, the same on both sides")
    return ap.parse_args(argv)


def serve_records(args):
    """(records, seconds between transmits or None to take the records' times)."""
    if args.replay:
        return list(make_source({"type": "container", "path": args.replay})), args.interval
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    sc = load_scenario_config(args.scenario)
    records = run_scenario(sc).records
    times = [r.transmit.time_ns for r in records]
    timed = all(b > a for a, b in zip(times, times[1:]))
    return records, args.interval or (None if timed else float(sc["pings"]["interval"]))


def receive_plan(args) -> dict:
    """What `receive` will run: the sonar of the stream, chain settings, navigation and outputs."""
    cfg = load_yaml(args.config) if args.config else {}
    source = cfg.get("source") or {}
    chain = dict(cfg.get("chain") or {})
    if args.chain:
        chain.update(json.loads(args.chain) if args.chain.lstrip().startswith("{") else load_yaml(args.chain))
    if args.engine:
        chain["engine"] = args.engine
    sonar = args.sonar or source.get("sonar")
    if not sonar:
        sys.exit("receive: name the stream's sonar with --sonar or a deployment file")
    if source.get("boards") is not None:
        # Sensor Bridge frames are one whole-array stream; per-board streams are merged before it (sonar-6f9).
        sys.exit("receive: source.boards: Sensor Bridge carries one whole-array stream; merge per-board streams with "
                 "tools/board_merge.py before the board side")
    if cfg.get("nav"):
        from gpusonar.live import check_nav_keys
        check_nav_keys(cfg["nav"])    # a key NavListener does not take is refused before the board is looked for (D-177)
    out = cfg.get("output") or {}
    stamp = time.strftime
    return {"sonar": sonar, "chain": chain, "nav": cfg.get("nav"), "sound_speed": source.get("sound_speed"),
            "sonar_overrides": json.loads(args.sonar_overrides) if args.sonar_overrides else source.get("sonar_overrides"),
            "report": args.report or (stamp(out["report"]) if out.get("report") else None),
            "recording": args.recording or (stamp(out["recording"]) if out.get("recording") else None),
            "report_udp": out.get("report_udp"), "taps": out.get("taps")}


def main(argv=None):
    args = parse(argv)
    if args.role == "serve":
        t0 = time.perf_counter()
        records, interval = serve_records(args)
        if args.scenario:     # the board is announced only after this: a receive started with it may need --find-timeout
            print(f"simulated {len(records)} pings of {args.scenario} in {time.perf_counter() - t0:.1f} s", flush=True)
        board = hsb.HsbBoardEmulator(records, args.frame_samples, ip=args.ip, pace=args.pace, lead_seconds=args.lead,
                                     ping_interval=interval, linger=args.linger, catch_up=args.catch_up, hold=args.hold)
        print("hsb: " + json.dumps(board.run(connect_timeout=args.connect_timeout, log=lambda t: print(t, flush=True))))
        if args.send_log:
            board.timing.dump(args.send_log)
        return 0
    from gpusonar.live import LiveChain, NavListener
    plan = receive_plan(args)
    # Every key of the `nav` section, as tools/sonar_live.py takes it (sonar-plk: water_depth, keep_s, surface_z and the
    # rest were dropped here before).
    nav = NavListener.from_config(plan["nav"]) if plan["nav"] else None
    with LiveChain(plan["chain"], nav=nav, recording=plan["recording"], report_path=plan["report"],
                   report_udp=plan["report_udp"], taps=plan["taps"]) as chain:
        _, stats = hsb.run_receiver(chain, plan["sonar"], plan["sonar_overrides"], hololink_ip=args.hololink,
                                    frame_samples=args.frame_samples, n_pings=args.pings, sound_speed=plan["sound_speed"],
                                    find_timeout=args.find_timeout, idle_timeout=args.idle, queue_frames=args.queue,
                                    receiver_affinity=args.receiver_core,
                                    socket_buffer=args.socket_buffer, log_arrivals=bool(args.arrival_log),
                                    chain_omp_threads=args.chain_omp_threads,
                                    on_report=None if args.quiet else (lambda r: print(ping_line(r), flush=True)))
    arrivals, chain_ns = stats.pop("arrivals", None), stats.pop("chain_wall_ns", None)
    if args.arrival_log:
        with open(args.arrival_log, "w", encoding="utf-8") as f:
            json.dump({"columns": ["frame", "received_ns", "packets", "bytes"], "rows": arrivals or [],
                       "chain_wall_ns": chain_ns or []}, f)
    print("hsb: " + json.dumps(stats), flush=True)
    short = hsb.stopped_short(stats, args.pings)
    if short:                     # the statistics are out; say what was missing and fail (sonar-dkb)
        print(f"receive: {short}", file=sys.stderr, flush=True)
    if stats.get("shutdown_error"):
        # The board stopped answering while the receiver shut down: Sensor Bridge's receiver thread is
        # still running and cannot be joined, so end the process here (the outputs are closed).
        print(f"receive: the board did not answer at shutdown ({stats['shutdown_error']})", file=sys.stderr, flush=True)
        sys.stdout.flush()
        os._exit(STOPPED_SHORT if short else 0)
    return STOPPED_SHORT if short else 0


if __name__ == "__main__":
    sys.exit(main())
