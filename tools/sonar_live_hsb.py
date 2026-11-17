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
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from gpusonar.config import load_yaml  # noqa: E402
from gpusonar.sources import hsb, make_source  # noqa: E402
from sonar_live import ping_line  # noqa: E402


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
    s.add_argument("--interval", type=float, help="seconds between transmits (default: the records' transmit times)")
    s.add_argument("--connect-timeout", type=float, default=120.0)
    r = sub.add_parser("receive", help="receive with Sensor Bridge's LinuxReceiverOp and run the chain")
    r.add_argument("config", nargs="?", help="deployment YAML (source.sonar, chain, nav, output)")
    r.add_argument("--hololink", default="127.0.0.1", help="the board's address")
    r.add_argument("--sonar", help="sonar configuration of the stream (overrides the file's)")
    r.add_argument("--sonar-overrides", help="JSON")
    r.add_argument("--chain", help="chain settings, JSON or a YAML file; merged over the file's chain section")
    r.add_argument("--engine", choices=["python", "cpp"], help="override chain.engine")
    r.add_argument("--pings", type=int, help="stop after this many (default: when the stream goes quiet)")
    r.add_argument("--idle", type=float, default=15.0, help="seconds without a frame before stopping")
    r.add_argument("--report", help="JSON lines file, one report per ping")
    r.add_argument("--recording", help="container file of the raw pings")
    r.add_argument("--queue", type=int, default=128, help="frames the receiver holds while the chain is busy")
    r.add_argument("--quiet", action="store_true")
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
        records, interval = serve_records(args)
        board = hsb.HsbBoardEmulator(records, args.frame_samples, ip=args.ip, pace=args.pace, lead_seconds=args.lead,
                                     ping_interval=interval, linger=args.linger)
        print("hsb: " + json.dumps(board.run(connect_timeout=args.connect_timeout, log=lambda t: print(t, flush=True))))
        return 0
    from gpusonar.live import LiveChain, NavListener
    plan = receive_plan(args)
    n = plan["nav"]
    nav = NavListener(port=n.get("port"), host=n.get("host", "0.0.0.0"), static=n.get("static")) if n else None
    with LiveChain(plan["chain"], nav=nav, recording=plan["recording"], report_path=plan["report"],
                   report_udp=plan["report_udp"], taps=plan["taps"]) as chain:
        _, stats = hsb.run_receiver(chain, plan["sonar"], plan["sonar_overrides"], hololink_ip=args.hololink,
                                    frame_samples=args.frame_samples, n_pings=args.pings, sound_speed=plan["sound_speed"],
                                    idle_timeout=args.idle, queue_frames=args.queue,
                                    on_report=None if args.quiet else (lambda r: print(ping_line(r), flush=True)))
    print("hsb: " + json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
