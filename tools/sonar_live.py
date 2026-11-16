#!/usr/bin/env python3
"""Run the long-range array chain on a live stream, a recording or a simulation.

    tools/sonar_live.py configs/deploy/popoto_vehicle.yaml             # the boards, over UDP
    tools/sonar_live.py CONFIG --replay flight.h5                      # a recording, same chain
    tools/sonar_live.py CONFIG --scenario popoto_12km                  # the simulator, same chain
    tools/sonar_live.py CONFIG --emulate flank_target_1km --port 0     # simulated pings sent as
                                                                       # board blocks over UDP

Paths under `output` may hold strftime fields (%Y%m%d_%H%M%S). --out DIR puts every output
in DIR instead. Ends when the source does, or after --pings.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar.config import load_yaml  # noqa: E402
from gpusonar.live import LiveChain, NavListener  # noqa: E402
from gpusonar.sources import make_source  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--replay", help="a container file to process instead of the configured source")
    ap.add_argument("--scenario", help="a simulated scenario to process instead")
    ap.add_argument("--emulate", help="a scenario to simulate and send to the configured UDP source as board blocks")
    ap.add_argument("--port", type=int, help="override the UDP port (0: any free port)")
    ap.add_argument("--out", help="directory for the recording, report and taps")
    ap.add_argument("--pings", type=int)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    out = dict(cfg.get("output") or {})
    stamp = time.strftime
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        out = {"recording": f"{args.out}/recording.h5", "report": f"{args.out}/report.jsonl",
               "report_udp": out.get("report_udp"), "taps": dict(out.get("taps") or {"pings": 1}, dir=f"{args.out}/taps")}
    sender = None
    if args.scenario:
        from gpusonar.sim import run_scenario
        source = run_scenario(args.scenario).records
    elif args.replay:
        source = make_source({"type": "container", "path": args.replay})
    else:
        src_cfg = dict(cfg["source"])
        if args.port is not None:
            src_cfg["port"] = args.port
        if args.emulate:
            from gpusonar.config import load_scenario_config
            from gpusonar.sim import run_scenario
            from gpusonar.sources import FlankBoardEmulator
            sc = load_scenario_config(args.emulate)
            records = run_scenario(sc).records
            src_cfg.update(sonar=sc["sonar"], sonar_overrides=sc.get("sonar_overrides"), host="127.0.0.1",
                           max_pings=len(records), timeout=10.0)
            source = make_source(src_cfg)
            sender = FlankBoardEmulator(records, float(sc["pings"]["interval"])).send(source.port, pace=0.25)
        else:
            source = make_source(src_cfg)
    nav_cfg = cfg.get("nav") if not (args.scenario or args.replay or args.emulate) else None
    nav = NavListener(port=nav_cfg.get("port"), host=nav_cfg.get("host", "0.0.0.0"), static=nav_cfg.get("static")) if nav_cfg else None
    with LiveChain(cfg.get("chain"), nav=nav, recording=stamp(out["recording"]) if out.get("recording") else None,
                   report_path=stamp(out["report"]) if out.get("report") else None,
                   report_udp=out.get("report_udp"), taps=out.get("taps")) as chain:
        for record in source:
            r = chain.process(record)
            if not args.quiet:
                kinds = {}
                for t in r["tracks"]:
                    kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
                print(f"ping {r['ping']:4d}  detections {len(r['detections']):3d}  tracks {json.dumps(kinds)}  "
                      f"{r['processing_seconds'] * 1e3:.0f} ms", flush=True)
            if args.pings and chain.pings >= args.pings:
                break
    if hasattr(source, "close"):
        source.close()
    if sender:
        sender.join(timeout=30.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
