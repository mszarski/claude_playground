#!/usr/bin/env python3
"""Turn a vendor's navigation stream into the chain's navigation datagrams.

    tools/nav_adapter.py configs/nav/generic_csv.yaml --replay nav.csv --out-udp 127.0.0.1:5702
    tools/nav_adapter.py MAPPING --replay nav.csv --speed 1 --out-udp 127.0.0.1:5702    # paced by its time stamps
    tools/nav_adapter.py MAPPING --replay nav.csv --out-file nav.jsonl                  # the datagrams, one per line
    tools/nav_adapter.py MAPPING --in-udp 5710 --out-udp 127.0.0.1:5702                 # live: vendor lines on a UDP port
    some_reader | tools/nav_adapter.py MAPPING --stdin --out-udp 127.0.0.1:5702         # live: vendor lines on stdin

The mapping (a YAML file) describes the vendor's side: field names or columns, units, sign and
frame conventions, angle order, time base and offset (python/gpusonar/navadapter.py). The same
datagrams feed tools/sonar_live.py (`nav.port`) and build/bin/sonar_live (`--nav-port`).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import navadapter  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mapping")
    ap.add_argument("--replay", help="a file of vendor lines")
    ap.add_argument("--speed", type=float, default=0.0, help="replay pace: 1 = as recorded, 0 = as fast as possible")
    ap.add_argument("--in-udp", type=int, help="live: vendor lines arrive on this UDP port")
    ap.add_argument("--stdin", action="store_true", help="live: vendor lines arrive on standard input")
    ap.add_argument("--out-udp", help="HOST:PORT of the chain's navigation port")
    ap.add_argument("--out-file", help="write the datagrams here, one JSON object per line")
    args = ap.parse_args(argv)
    mapping = navadapter.Mapping.load(args.mapping)
    sinks = []
    out_file = open(args.out_file, "w", encoding="utf-8") if args.out_file else None
    if out_file:
        sinks.append(lambda fix: out_file.write(json.dumps(fix) + "\n"))
    if args.out_udp:
        host, port = args.out_udp.rsplit(":", 1)
        sinks.append(navadapter.udp_sender(host, int(port)))
    if not sinks:
        sinks.append(lambda fix: print(json.dumps(fix), flush=True))

    def send(fix):
        for sink in sinks:
            sink(fix)

    n = -1
    try:
        if args.replay:
            with open(args.replay, encoding="utf-8", errors="replace") as f:
                n = navadapter.replay(mapping, f, send, speed=args.speed)
        elif args.in_udp is not None:
            n = navadapter.run_udp(mapping, args.in_udp, send)
        elif args.stdin:
            n = navadapter.replay(mapping, sys.stdin, send)
        else:
            ap.error("give --replay FILE, --in-udp PORT or --stdin")
    except KeyboardInterrupt:
        pass
    finally:
        if out_file:
            out_file.close()
    print(f"nav_adapter: {n} datagrams", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
