#!/usr/bin/env python3
"""Merge per-board GSFB streams into one whole-array stream, as a UDP relay (sonar-6f9).

    tools/board_merge.py popoto_testauv_1row_clocks --port 5710 --to 127.0.0.1:5700
    tools/board_merge.py SONAR --port 5710 --to HOST:PORT --boards boards.yaml --idle 30

Every capture board sends its own stream (its channels, its own sample counter and capture start, its board id in
the block header) to --port; the relay sends the whole-array stream (every channel on the reference board's counter)
to --to, where any whole-array receiver takes it: `flank_udp` without `source.boards`, `build/bin/sonar_live --udp`
without `stream_boards`, or a receiver elsewhere. The rules (board map, alignment by capture start, waiting, missing
boards, restarts) are those of python/gpusonar/sources/board_merge.py; `tools/sonar_live.py`, `build/bin/sonar_live`
(with and without --listen), the Holoscan application and `tools/sonar_live_hsb.py receive` run the same merge
in-process (`source.boards`, `stream_boards=1`; sonar-bwu), and need no relay.

--boards: a YAML file holding the `source.boards` mapping of a deployment file (channels, reference, missing,
wait_s); without it, the sonar file's `receive.boards` and every default. The statistics per board are printed as one
JSON line on standard error when the relay stops (--idle seconds without a datagram; 0: never).
"""
import argparse
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main(argv=None) -> int:
    from gpusonar.config import load_yaml
    from gpusonar.sonar import SonarModel
    from gpusonar.sources.board_merge import BoardMerger
    from gpusonar.sources.flank_stream import BlockError, stream_buffer_bytes

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sonar", help="the sonar file of the array (configs/sonars/NAME.yaml)")
    ap.add_argument("--port", type=int, required=True, help="UDP port the boards send to")
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on")
    ap.add_argument("--to", required=True, help="HOST:PORT of the whole-array receiver")
    ap.add_argument("--boards", help="YAML file with the source.boards mapping")
    ap.add_argument("--sample-bytes", type=int, default=3, help="the boards' sample word (source.sample_bytes)")
    ap.add_argument("--idle", type=float, default=30.0, help="stop after this many seconds without a datagram (0: never)")
    args = ap.parse_args(argv)
    sonar = SonarModel.load(args.sonar)
    options = load_yaml(args.boards) if args.boards else {}
    merger = BoardMerger.from_options(options, sonar, args.sample_bytes, where="boards.")
    host, port = args.to.rsplit(":", 1)
    to = (host, int(port))
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, stream_buffer_bytes(sonar.n_elements, sonar.sample_rate, args.sample_bytes))
    rx.bind((args.host, args.port))
    rx.settimeout(args.idle if args.idle > 0 else None)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    try:
        while True:
            try:
                data = rx.recv(65535)
            except socket.timeout:
                break
            try:
                blocks = merger.feed(data)
            except BlockError:
                continue                       # counted in the statistics; the last reason is in them
            for b in blocks:
                tx.sendto(b, to)
                sent += 1
    finally:
        for b in merger.finish():
            tx.sendto(b, to)
            sent += 1
        rx.close()
        tx.close()
        stats = dict(merger.stats(), sent_datagrams=sent, last_error=merger.last_error)
        print(f"board_merge: {json.dumps(stats)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
