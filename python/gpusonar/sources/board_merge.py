"""One stream per capture board in, the whole-array stream out (sonar-6f9, D-184).

The array's 4 to 12 capture boards each have their own sample counter and capture start, so none of them can send the
whole-array `GSFB` stream (every channel on one counter) that `flank_stream.PingAssembler` takes. Each board (its
forwarder) sends a per-board stream instead: `GSFB` version 1 blocks with flag bit 1 set and its board id at header
offset 44 (flank_stream.py), carrying that board's channels on its own counter and capture start. `BoardMerger` puts
the boards' streams back together as the whole-array stream, block for block, and the assembler downstream is the
unchanged one: its rules for order, holes, lost pings and restarts (D-161, D-177) apply to the merged stream as they
did. libs/live board_merge.hpp is the C++ twin (`BoardMergerCpp` in Python); the engine-parity tests in
tests/python/test_board_streams.py hold both to the same merged bytes and counts.

The rules (ours, all of them):

* The map. Board id -> its channels of the array (`channels`: [{id, first, count}]; default the sonar file's
  `receive.boards`, board i carrying boards[i]). Every channel is on exactly one board. Which board is wired to which
  hydrophones is unconfirmed until the owner says (docs/tasks/VEHICLE_ADAPTERS.md, question 3): it is configuration.
  A block of a board not in the map, of another channel count, or without a board id is refused by name.
* The time base. The merged stream is the `reference` board's (default the first in the map): its sample counter, its
  capture start, its transmit marks, ping index and gain code. The other boards' transmit fields are not used.
* Coarse alignment. Board b's sample i and the reference's sample j are the same instant when
      j = i + off_b,   off_b = -floor(d + 0.5),   d = (start_ref - start_b) [ns] * 1e-9 * fs
  (fs the nominal rate): the capture starts' difference in whole samples. That is all the merge aligns. What is left -
  the time stamps' error, clock offsets below a sample, drifts within a ping - is the synchronisation stage's to
  measure from the reference bursts (sync.py), which looks for each burst within `sync.search_time` (2 ms) of where it
  is expected. So the boards' capture starts must agree with true time to well inside 2 ms, less the drift between two
  boards over a receive window (25 ppm each, 17.9 s: up to 0.9 ms). Boards on free-running clocks drift apart by
  (ppm_b - ppm_ref) x time since capture start, so without a correction a capture must restart (or the clocks be
  disciplined) before that sum leaves the search: 40 s at 50 ppm apart. `clock_ppm` {board id: ppm} (sonar-ubg; each
  board's rate error, positive fast; default 0, unconfirmed until measured) takes the rates into the offset, worked
  out at each reference transmit (`offset`) and held for that ping: a rate known to 0.1 ppm keeps two boards within
  2 ms for 5.6 hours. The merged stream is stamped at the reference board's rate (flank_udp, sonar_live).
* Waiting. A reference block is the request for one merged span. It is emitted once every board has delivered its
  part, in the order of the spans (one waiting holds up those after it). A board's part is given up as missing once
  the reference has moved `wait_s` (0.05 s) past the span's end, or the board itself has moved `wait_s` past its part
  without delivering it, or the stream ends (finish()). Boards may therefore lag each other by up to `wait_s`; a block
  arriving later is counted late and not used.
* Missing. `missing: drop` (default): a span lacking any part of any board is not emitted; the assembler then sees a
  hole and drops the ping it falls in (D-161: no loss inside a ping), as for a lost whole-array block.
  `missing: zero`: the span is emitted with zeros in the missing board's channels (counted `zero_filled_blocks`); the
  ping is kept with those hydrophones silent for that stretch. Both count the samples each board did not deliver.
* Restarts. A block with another capture start than its board's is a restart of that board: its new counter is
  aligned by its new capture start (a capture start left is never returned to: its blocks are repeats). A restart of the
  reference first decides every waiting span, then the merged stream follows the new capture: the assembler sees a
  whole-array restart. A board must stamp its capture start; one whose counter goes back under the same capture start
  is not aligned again (its blocks are taken where they say they are).
* Repeats. A block whose first sample its board already sent is a repeat (counted, ignored).
* Output. Each merged span goes out as whole-array blocks of at most `block_samples` (default: what fits the stream's
  datagram, `block_samples_for`), each naming the latest reference transmit at or before its own end.

`ClockEstimator` estimates a board's rate from its blocks' arrival times (reported; used for the stamps only with
`clock.discipline`).

Counted per board (`stats()`): blocks, used_blocks, late_blocks, repeated_blocks, restarts, missing_samples,
bad_blocks, and its capture start; for the merge: merged_blocks (spans emitted), dropped_blocks, zero_filled_blocks,
bad_blocks (every refused block), pending.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from typing import Iterator

import numpy as np

from .flank_stream import (FLAG_OVERFLOW, clock_rate, MAX_DATAGRAM, MAX_SAMPLE_INDEX, SAMPLE_BYTES, HEADER_BYTES, BlockError,
                           FlankBoardEmulator, block_samples_for, decode_block, encode_block,
                           full_scale_counts)

WAIT_S = 0.05            # how long (in samples of the stream) a board may lag the others before its part is missing; ours
MAX_LEFT = 8             # capture starts remembered as left, per board (as flank_stream's restarts)
MAX_DONE = 64            # reference spans remembered as decided, for telling a repeat
MAX_NAMED = 8            # reference transmits remembered, for naming the blocks of a cut span
OPTION_KEYS = ("channels", "reference", "missing", "wait_s", "clock_ppm")
CLOCK_KEYS = ("bin_s", "window_s", "discipline", "min_span_s", "max_sigma_ppm")
CHANNEL_KEYS = ("id", "first", "count")
MISSING = ("drop", "zero")


def half_up(x: float) -> int:
    """floor(x + 0.5): the rounding both engines use (C++ std::floor), whatever the sign."""
    return int(math.floor(x + 0.5))


class _Board:
    def __init__(self, bid: int, first: int, count: int):
        self.id, self.first, self.count = bid, first, count
        self.ppm = 0.0                    # its clock's configured rate error (clock_ppm; sonar-ubg)
        self.start_ns = None              # the capture start of its latest capture
        self.left = []
        self.blocks = {}                  # (capture start, own first sample) -> [n, counts [n, count] int32, flags, used]
        self.frontier = None              # own sample after the furthest block of the latest capture
        self.n_blocks = self.used_blocks = self.late_blocks = self.repeated_blocks = 0
        self.restarts = self.missing_samples = self.bad_blocks = 0

    def stats(self) -> dict:
        return {"blocks": self.n_blocks, "used_blocks": self.used_blocks, "late_blocks": self.late_blocks,
                "repeated_blocks": self.repeated_blocks, "restarts": self.restarts, "missing_samples": self.missing_samples,
                "bad_blocks": self.bad_blocks, "start_ns": self.start_ns}


class BoardMerger:
    """Per-board blocks in (`feed`), whole-array blocks out. The rules are in this module's docstring."""

    def __init__(self, boards: list, n_channels: int, sample_rate: float, reference: int | None = None, missing: str = "drop",
                 wait_s: float = WAIT_S, sample_bytes: int = SAMPLE_BYTES, block_samples: int | None = None, where: str = "boards.",
                 clock_ppm: dict | None = None):
        self.n_channels = int(n_channels)
        self.sample_rate = float(sample_rate)
        self.sample_bytes = int(sample_bytes)
        full_scale_counts(self.sample_bytes)             # 2, 3 or 4, or an error
        if not boards:
            raise ValueError(f"{where}channels: no board")
        self.boards = []
        on = np.zeros(self.n_channels, dtype=np.int64)
        for b in boards:
            bid, first, count = int(b["id"]), int(b["first"]), int(b["count"])
            if not 0 <= bid < (1 << 16):
                raise ValueError(f"{where}channels: board id {bid} does not fit the block header (0 to 65535)")
            if any(x.id == bid for x in self.boards):
                raise ValueError(f"{where}channels: board {bid} given twice")
            if count < 1 or first < 0 or first + count > self.n_channels:
                raise ValueError(f"{where}channels: board {bid}: channels {first} to {first + count - 1} are not channels of the "
                                 f"array's {self.n_channels}")
            on[first:first + count] += 1
            self.boards.append(_Board(bid, first, count))
        for c in range(self.n_channels):
            if on[c] != 1:
                raise ValueError(f"{where}channels: channel {c} is on {'no' if on[c] == 0 else 'more than one'} board")
        self._by_id = {b.id: b for b in self.boards}
        self.reference = self.boards[0].id if reference is None else int(reference)
        if self.reference not in self._by_id:
            raise ValueError(f"{where}reference: board {self.reference} is not in the board map")
        if missing not in MISSING:
            raise ValueError(f"{where}missing: {missing!r}: drop or zero")
        self.missing = missing
        wait_s = float(wait_s)
        if not math.isfinite(wait_s) or wait_s < 0.0:
            raise ValueError(f"{where}wait_s: {wait_s!r}: a number of seconds, 0 or more")
        self.wait = half_up(wait_s * self.sample_rate)
        self.block_samples = int(block_samples) if block_samples else block_samples_for(self.n_channels, self.sample_bytes)
        for bid, ppm in (clock_ppm or {}).items():
            if isinstance(bid, bool) or not isinstance(bid, (int, np.integer)) or int(bid) not in self._by_id:
                raise ValueError(f"{where}clock_ppm: board {bid} is not in the board map")
            self._by_id[int(bid)].ppm = clock_rate(ppm, f"{where}clock_ppm")
        self._ref = self._by_id[self.reference]
        self._others = [b for b in self.boards if b is not self._ref]
        self.merged_blocks = self.dropped_blocks = self.zero_filled_blocks = self.bad_blocks = 0
        self.last_error = ""
        self._pending = []                # reference spans waiting, by first sample: dicts
        self._done = []                   # first samples of reference spans decided (latest MAX_DONE)
        self._named = []                  # (tx, index) of reference transmits, sorted (latest MAX_NAMED)
        self._ref_frontier = None
        self._decided_to = None           # merged sample after the furthest span decided
        self._anchor = 0                  # anchor of the latest reference block (its transmit, else its first sample)

    @classmethod
    def from_options(cls, options: dict | None, sonar, sample_bytes: int = SAMPLE_BYTES, where: str = "boards.") -> "BoardMerger":
        """From a deployment file's `source.boards` ({} for every default); an unknown key is refused by name."""
        from ..live import _refuse_unknown
        options = dict(options or {})
        _refuse_unknown(options, OPTION_KEYS, where)
        channels = options.get("channels")
        if channels is None:
            channels = [{"id": i, "first": int(b.get("first", 0)), "count": int(b.get("count", sonar.n_elements))}
                        for i, b in enumerate(sonar.boards)]
        for k, c in enumerate(channels):
            if not isinstance(c, dict):
                raise ValueError(f"{where}channels[{k}]: {{id, first, count}}, not {c!r}")
            _refuse_unknown(c, CHANNEL_KEYS, f"{where}channels[{k}].")
            for key in CHANNEL_KEYS:
                if key not in c:
                    raise ValueError(f"{where}channels[{k}].{key}: needed")
        return cls(channels, sonar.n_elements, sonar.sample_rate, reference=options.get("reference"),
                   missing=options.get("missing", "drop"), wait_s=options.get("wait_s", WAIT_S), sample_bytes=sample_bytes,
                   where=where, clock_ppm=options.get("clock_ppm"))

    def clock_ppm(self, bid: int) -> float:
        """The configured rate error of a board's clock (ppm; 0 when not given)."""
        return self._by_id[bid].ppm

    # -- alignment ---------------------------------------------------------------------------------------------

    def offset(self, board: _Board, start_ns: int, anchor: int) -> int:
        """off_b: the reference's sample index minus that of a board whose capture started at `start_ns`, at the same
        instant near the reference's sample `anchor` (module docstring). With the boards' clock rates e (clock_ppm 1e-6):
            d = x (1 + e_b) + anchor (e_b - e_ref) / (1 + e_ref),   x = (start_ref - start_b) 1e-9 fs
        (ours: board b counts (t - start_b) fs (1 + e_b) at true time t, the reference (t - start_ref) fs (1 + e_ref);
        the difference at the reference's count `anchor`). All rates 0: d = x."""
        eb, er = board.ppm * 1e-6, self._ref.ppm * 1e-6
        x = float(self._ref.start_ns - start_ns) * 1e-9 * self.sample_rate
        d = x * (1.0 + eb) + float(anchor) * (eb - er) / (1.0 + er)
        return -half_up(d)

    def _spans(self, board: _Board, anchor: int) -> list:
        """The board's blocks as [merged first, merged end, key], sorted by merged first."""
        out = []
        for key, blk in board.blocks.items():
            f = key[1] + self.offset(board, key[0], anchor)
            out.append([f, f + blk[0], key])
        out.sort(key=lambda x: (x[0], x[1], x[2]))
        return out

    def _frontier(self, board: _Board, anchor: int):
        """Merged sample after the board's furthest block of its latest capture, or None."""
        return None if board.frontier is None else board.frontier + self.offset(board, board.start_ns, anchor)

    # -- blocks ------------------------------------------------------------------------------------------------

    def _refuse(self, message: str, board: _Board | None = None):
        self.bad_blocks += 1
        if board is not None:
            board.bad_blocks += 1
        self.last_error = message
        raise BlockError(message)

    def feed(self, data: bytes, arrival_ns: int | None = None) -> list:
        """Take one per-board block; return the whole-array blocks it completes (bytes, in order)."""
        try:
            block = decode_block(data, self.sample_bytes)
        except BlockError as e:
            self._refuse(str(e))
        bid = block["board"]
        if bid is None:
            self._refuse("a whole-array block (no board id) where per-board streams are merged")
        board = self._by_id.get(bid)
        if board is None:
            self._refuse(f"board {bid} is not in the board map (boards {', '.join(str(b.id) for b in self.boards)})")
        if block["n_channels"] != board.count:
            self._refuse(f"board {bid} sent {block['n_channels']} channels, the board map gives it {board.count}", board)
        first, n, tx = block["sample_index"], block["n_samples"], block["tx_sample"]
        if n < 1 or not 0 <= first < MAX_SAMPLE_INDEX or not -1 <= tx < MAX_SAMPLE_INDEX or tx >= first + n:
            self._refuse(f"board {bid}: a block without samples, or its sample index or transmit out of range", board)
        out = []
        ns = block["start_ns"]
        if board.start_ns is None:
            board.start_ns = ns
        elif ns != board.start_ns:
            if ns in board.left:                          # a capture this board has left: stale
                board.repeated_blocks += 1
                return out
            board.restarts += 1
            board.left = (board.left + [board.start_ns])[-MAX_LEFT:]
            if board is self._ref:
                out += self._decide(force=True)           # the old capture's spans, with what the boards gave
                self._pending, self._done, self._named = [], [], []
                self._ref_frontier = self._decided_to = None
            board.start_ns, board.frontier = ns, None
        board.n_blocks += 1
        if board is self._ref:
            if first in self._done or any(p["first"] == first for p in self._pending):
                board.repeated_blocks += 1
                return out
            p = {"first": first, "n": n, "tx": tx, "index": block["ping_index"], "gain": block["gain_code"],
                 "flags": block["flags"] & FLAG_OVERFLOW, "start_ns": ns, "counts": block["counts"],
                 "anchor": tx if tx >= 0 else first}
            at = len(self._pending)
            while at and self._pending[at - 1]["first"] > first:
                at -= 1
            self._pending.insert(at, p)
            self._anchor = p["anchor"]
            if tx >= 0 and (tx, block["ping_index"]) not in self._named:
                self._named = sorted(self._named + [(tx, block["ping_index"])])[-MAX_NAMED:]
            self._ref_frontier = first + n if self._ref_frontier is None else max(self._ref_frontier, first + n)
        else:
            if (ns, first) in board.blocks:
                board.repeated_blocks += 1
                return out
            if not self._wanted(board, ns, first, n):
                board.late_blocks += 1
                return out
            board.blocks[(ns, first)] = [n, block["counts"], block["flags"] & FLAG_OVERFLOW, False]
            board.frontier = first + n if board.frontier is None else max(board.frontier, first + n)
        out += self._decide(force=False)
        self._prune()
        return out

    def finish(self) -> list:
        """The streams have ended: decide every waiting span with what the boards gave."""
        out = self._decide(force=True)
        self._prune()
        return out

    # -- deciding ----------------------------------------------------------------------------------------------

    def _wanted(self, board: _Board, ns: int, first: int, n: int) -> bool:
        """Whether a board's block may still be used: wanted by a waiting span, or neither behind its board's stream by
        more than the wait nor in a stretch already decided. Before the reference is heard every block is kept."""
        if self._ref.start_ns is None:
            return True
        f = first + self.offset(board, ns, self._anchor)
        e = f + n
        if any(min(e, p["first"] + p["n"]) > max(f, p["first"]) for p in self._pending):
            return True
        front = self._frontier(board, self._anchor)
        if front is not None and e <= front - self.wait:
            return False
        return not (self._decided_to is not None and e <= self._decided_to)

    def _cover(self, board: _Board, a: int, b: int, anchor: int, dest=None) -> tuple:
        """(samples of merged [a, b) the board's blocks hold, their overflow flags); copied into `dest` [b - a, count]."""
        pos, covered, flags = a, 0, 0
        for f, e, key in self._spans(board, anchor):
            if f >= b:
                break
            x0, x1 = max(f, pos), min(e, b)
            if x1 <= x0:
                continue
            blk = board.blocks[key]
            if dest is not None:
                dest[x0 - a:x1 - a] = blk[1][x0 - f:x1 - f]
                if not blk[3]:
                    blk[3] = True
                    board.used_blocks += 1
            covered += x1 - x0
            flags |= blk[2]
            pos = x1
        return covered, flags

    def _status(self, p: dict, board: _Board, force: bool) -> str:
        end = p["first"] + p["n"]
        if board.start_ns is not None and self._cover(board, p["first"], end, p["anchor"])[0] == p["n"]:
            return "have"
        front = self._frontier(board, p["anchor"]) if board.start_ns is not None else None
        if force or self._ref_frontier >= end + self.wait or (front is not None and front >= end + self.wait):
            return "missing"
        return "wait"

    def _decide(self, force: bool) -> list:
        out = []
        while self._pending:
            p = self._pending[0]
            if any(self._status(p, b, force) == "wait" for b in self._others):
                break
            self._pending.pop(0)
            self._done = (self._done + [p["first"]])[-MAX_DONE:]
            end = p["first"] + p["n"]
            self._decided_to = end if self._decided_to is None else max(self._decided_to, end)
            short = False
            for b in self._others:
                miss = p["n"] - (self._cover(b, p["first"], end, p["anchor"])[0] if b.start_ns is not None else 0)
                if miss:
                    b.missing_samples += miss
                    short = True
            if short and self.missing == "drop":
                self.dropped_blocks += 1
                continue
            if short:
                self.zero_filled_blocks += 1
            self.merged_blocks += 1
            self._ref.used_blocks += 1
            out += self._emit(p)
        return out

    def _emit(self, p: dict) -> list:
        out = []
        for k0 in range(0, p["n"], self.block_samples):
            k1 = min(k0 + self.block_samples, p["n"])
            p0, p1 = p["first"] + k0, p["first"] + k1
            counts = np.zeros((k1 - k0, self.n_channels), dtype=np.int32)
            ref = self._ref
            counts[:, ref.first:ref.first + ref.count] = p["counts"][k0:k1]
            flags = p["flags"]
            for b in self._others:
                part = np.zeros((k1 - k0, b.count), dtype=np.int32)
                flags |= self._cover(b, p0, p1, p["anchor"], part)[1]
                counts[:, b.first:b.first + b.count] = part
            if 0 <= p["tx"] < p1:
                tx, index = p["tx"], p["index"]
            else:
                before = [t for t in self._named if t[0] < p1]
                tx, index = before[-1] if before else (-1, p["index"])
            out.append(encode_block(counts, p0, tx, index, p["gain"], flags, p["start_ns"], self.sample_bytes))
        return out

    def _prune(self) -> None:
        """Forget board blocks no waiting span needs that are behind their board by more than the wait, or in a stretch
        decided (measured at the latest reference anchor)."""
        if self._ref.start_ns is None:
            return
        for b in self._others:
            if not b.blocks:
                continue
            front = self._frontier(b, self._anchor)
            for f, e, key in self._spans(b, self._anchor):
                if any(min(e, p["first"] + p["n"]) > max(f, p["first"]) for p in self._pending):
                    continue
                if (front is not None and e <= front - self.wait) or (self._decided_to is not None and e <= self._decided_to):
                    del b.blocks[key]

    def stats(self) -> dict:
        return {"boards": {str(b.id): b.stats() for b in self.boards}, "merged_blocks": self.merged_blocks,
                "dropped_blocks": self.dropped_blocks, "zero_filled_blocks": self.zero_filled_blocks,
                "bad_blocks": self.bad_blocks, "pending": len(self._pending)}


class ClockEstimator:
    """A board clock's rate against the host clock, from the arrival times of its blocks (sonar-ubg; ours).

    Each block's last sample `sample_end` was taken, on the board's counter, at nominal time x = sample_end / fs after
    capture start, and arrived at `arrival_ns` on the host clock: y = (arrival - capture start) - x = latency - e x / (1 + e)
    for a board counting fs (1 + e) a true second. Arrival delays are one-sided (queues only add), so per bin of `bin_s`
    seconds of stream the earliest arrival is kept (its x and y), the bins of the last `window_s` seconds are fitted with
    a least-squares line y = a + s x, and e = -s / (1 + s). `sigma_ppm` is the slope's standard error from the bins'
    scatter about the line. Nothing is assumed about the delays but that their least value is steady over the window: a
    host clock being slewed (NTP corrections up to 500 ppm) or a latency that drifts goes into the estimate whole.
    A new capture start starts the estimate again. `discipline`: the stamps use the estimate (stamp_ppm) once it spans
    `min_span_s` with `sigma_ppm` at most `max_sigma_ppm`; else the configured rate. libs/live board_merge.hpp
    ClockEstimator is the same, sums taken in the same order."""

    def __init__(self, sample_rate: float, bin_s: float = 1.0, window_s: float = 600.0, discipline: bool = False,
                 min_span_s: float = 300.0, max_sigma_ppm: float = 0.5, where: str = "clock."):
        for key, v in (("bin_s", bin_s), ("window_s", window_s), ("min_span_s", min_span_s), ("max_sigma_ppm", max_sigma_ppm)):
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(float(v)) or float(v) < 0.0:
                raise ValueError(f"{where}{key}: {v!r}: a number, 0 or more")
        if not isinstance(discipline, (bool, np.bool_)):
            raise ValueError(f"{where}discipline: {discipline!r}: true or false")
        self.sample_rate = float(sample_rate)
        self.bin = max(half_up(float(bin_s) * self.sample_rate), 1)
        self.window_bins = max(half_up(float(window_s) / max(float(bin_s), 1e-12)), 1)
        self.discipline, self.min_span_s, self.max_sigma_ppm = bool(discipline), float(min_span_s), float(max_sigma_ppm)
        self.start_ns = None
        self.restarts = 0
        self._bins = {}                   # bin -> (x, y) of its earliest arrival, s
        self._latest = None

    @classmethod
    def from_options(cls, options: dict | None, sample_rate: float, where: str = "clock.") -> "ClockEstimator":
        from ..live import _refuse_unknown
        options = dict(options or {})
        _refuse_unknown(options, CLOCK_KEYS, where)
        return cls(sample_rate, where=where, **options)

    def feed(self, sample_end: int, start_ns: int, arrival_ns: int) -> bool:
        """One block: the counter after its last sample, its capture start, its arrival (ns, host clock). True when it
        opens a new bin (a new estimate is then worth working out)."""
        if self.start_ns is None:
            self.start_ns = start_ns
        elif start_ns != self.start_ns:
            self.restarts += 1
            self.start_ns, self._bins, self._latest = start_ns, {}, None
        x = float(sample_end) / self.sample_rate
        y = float(arrival_ns - start_ns) * 1e-9 - x
        k = sample_end // self.bin
        old = self._bins.get(k)
        if old is None or y < old[1]:
            self._bins[k] = (x, y)
        if self._latest is None or k > self._latest:
            self._latest = k
            for j in [j for j in self._bins if j <= k - self.window_bins]:
                del self._bins[j]
            return True
        return False

    def estimate(self) -> dict:
        """{ppm, sigma_ppm (None with fewer than 3 bins), span_s, bins}."""
        keys = sorted(self._bins)
        m = len(keys)
        pts = [self._bins[k] for k in keys]
        span = (max(p[0] for p in pts) - min(p[0] for p in pts)) if m else 0.0
        if m < 3:
            return {"ppm": None, "sigma_ppm": None, "span_s": span, "bins": m}
        sx = sy = 0.0
        for x, y in pts:
            sx += x
            sy += y
        mx, my = sx / m, sy / m
        sxx = sxy = 0.0
        for x, y in pts:
            sxx += (x - mx) * (x - mx)
            sxy += (x - mx) * (y - my)
        slope = sxy / sxx
        rr = 0.0
        for x, y in pts:
            r = y - my - slope * (x - mx)
            rr += r * r
        sigma = math.sqrt(rr / (m - 2) / sxx)
        return {"ppm": -slope / (1.0 + slope) * 1e6, "sigma_ppm": sigma / ((1.0 + slope) * (1.0 + slope)) * 1e6, "span_s": span, "bins": m}

    def stamp_ppm(self, configured: float) -> float:
        """The rate the stamps use: the estimate when disciplining and it is good enough, else `configured`."""
        if not self.discipline:
            return configured
        e = self.estimate()
        if e["ppm"] is None or e["span_s"] < self.min_span_s or e["sigma_ppm"] > self.max_sigma_ppm:
            return configured
        return e["ppm"]


class BoardStreamEmulator:
    """Plays simulated whole-array pings as the capture boards would send them, one stream per board: board k carries
    its channels (`boards`: [{id, first, count}]) on its own counter, which starts at sample `start_offsets[k]` of the
    whole-array stream (negative: before the first transmit; its capture start is stamped to match), cut into its own
    blocks (`block_samples`: one size, or one per board; default what fits a datagram). `restarts` {board id: sample of
    the whole-array stream}: that board starts a new capture there (counter from 0, a new capture start). Every board
    marks every transmit on its own counter. `blocks()` interleaves the boards' blocks by the instant their last sample
    was taken, as they would arrive.

    `ping_times` (true seconds after `start_ns`, one per record) with `clock_ppm` (one rate per board): the boards'
    clocks run at their own rates (sonar-ubg). Board k counts (t - its capture start) fs (1 + ppm_k 1e-6) at true time t;
    the first board in `boards` is taken as the merge's reference, and every other board's stretch of each ping is placed
    on its counter by BoardMerger's offset at that ping's transmit (the same model). Only each ping's window, with a few
    blocks either side, is sent: an hour of a stream needs no hour of samples."""

    def __init__(self, records, ping_interval: float, boards: list, start_offsets=None, start_ns: int = 0, block_samples=None,
                 gain_code: int = 0, sample_bytes: int = SAMPLE_BYTES, restarts: dict | None = None, clock_ppm=None,
                 ping_times=None):
        self.whole = FlankBoardEmulator(records, ping_interval, start_ns=start_ns, gain_code=gain_code, sample_bytes=sample_bytes)
        self.boards = [(int(b["id"]), int(b["first"]), int(b["count"])) for b in boards]
        self.offsets = [int(o) for o in (start_offsets or [0] * len(self.boards))]
        if block_samples is None or isinstance(block_samples, (int, np.integer)):
            self.block_samples = [int(block_samples) if block_samples else block_samples_for(c, sample_bytes) for _, _, c in self.boards]
        else:
            self.block_samples = [int(x) for x in block_samples]
        self.restarts = dict(restarts or {})
        self.start_ns = int(start_ns)
        self.sample_rate = self.whole.sample_rate
        self.sample_bytes = int(sample_bytes)
        self.records = list(records)
        self.ping_times = None if ping_times is None else [float(t) for t in ping_times]
        self.clock_ppm = [float(p) for p in (clock_ppm or [0.0] * len(self.boards))]
        if self.ping_times is not None and len(self.ping_times) != len(self.records):
            raise ValueError("ping_times: one time per record")
        if clock_ppm is not None and self.ping_times is None:
            raise ValueError("clock_ppm: give ping_times too (the boards' rates are emulated on pings at true times)")

    def stamp(self, origin: int) -> int:
        """The capture start of a counter whose sample 0 is sample `origin` of the whole-array stream."""
        return self.start_ns + half_up(origin * 1e9 / self.sample_rate)

    def _board_blocks(self, k: int, stream: np.ndarray) -> Iterator[tuple]:
        bid, first, count = self.boards[k]
        period, total, bs = self.whole.period, len(stream), self.block_samples[k]
        segments = [(self.offsets[k], total)]
        if bid in self.restarts:
            r = int(self.restarts[bid])
            segments = [(self.offsets[k], r), (r, total)]
        for origin, to in segments:
            ns = self.stamp(origin)
            for i0 in range(0, to - origin, bs):
                i1 = min(i0 + bs, to - origin)
                j0, j1 = i0 + origin, i1 + origin
                counts = np.zeros((i1 - i0, count), dtype=np.int32)
                a, b = max(j0, 0), min(j1, total)
                if b > a:
                    counts[a - j0:b - j0] = stream[a:b, first:first + count]
                tx, index = -1, 0
                if j1 - 1 >= 0:
                    ping = (j1 - 1) // period
                    if ping * period - origin >= 0:
                        tx, index = ping * period - origin, ping
                yield j1, k, bid, encode_block(counts, i0, tx, index, self.whole.gain_code, 0, ns, self.sample_bytes, board=bid)

    def _timed_blocks(self) -> list:
        """The blocks of the `ping_times` mode, as (ping, merged end, board order, id, bytes)."""
        fs_ = self.sample_rate
        w0, full = self.whole.window_start, self.whole.full_counts
        stamps = [self.stamp(o) for o in self.offsets]
        merger = BoardMerger([{"id": b, "first": f, "count": c} for b, f, c in self.boards], sum(c for _, _, c in self.boards), fs_,
                             reference=self.boards[0][0], clock_ppm={b: p for (b, _, _), p in zip(self.boards, self.clock_ppm)})
        merger._ref.start_ns = stamps[0]
        bmax = max(self.block_samples)
        out = []
        txs = []
        for k, t in enumerate(self.ping_times):
            txs.append(half_up((t - (stamps[0] - self.start_ns) * 1e-9) * fs_ * (1.0 + self.clock_ppm[0] * 1e-6)))
        offs = [[merger.offset(merger._by_id[b], stamps[q], txs[k]) for q, (b, _, _) in enumerate(self.boards)] for k in range(len(txs))]
        for k, r in enumerate(self.records):
            x = np.clip(np.rint(r.samples.array.T * self.whole.counts_per_volt), -full, full - 1).astype(np.int32)
            a, n = txs[k] + w0, len(x)
            bs0 = self.block_samples[0]
            r0, r1 = ((a - 2 * bmax) // bs0) * bs0, -(-(a + n + 2 * bmax) // bs0) * bs0       # the reference's stretch
            for q, (bid, first, count) in enumerate(self.boards):
                off, bs = offs[k][q], self.block_samples[q]
                lo = r0 if q == 0 else ((r0 - off - 2 * bs) // bs) * bs
                hi = r1 if q == 0 else r1 - off + 2 * bs
                own_txs = [(txs[j] - offs[j][q], j) for j in range(len(txs))]
                for i0 in range(lo, hi, bs):
                    i1 = i0 + bs
                    counts = np.zeros((bs, count), dtype=np.int32)
                    j0, j1 = max(i0 + off, a), min(i1 + off, a + n)
                    if j1 > j0:
                        counts[j0 - off - i0:j1 - off - i0] = x[j0 - a:j1 - a, first:first + count]
                    named = [p for p in own_txs if 0 <= p[0] < i1]
                    tx, index = max(named) if named else (-1, 0)
                    out.append((k, i1 + off, q, bid, encode_block(counts, i0, tx, index, self.whole.gain_code, 0, stamps[q],
                                                                  self.sample_bytes, board=bid)))
        return out

    def blocks(self) -> list:
        """[(board id, block bytes)] of every board, in the order they would arrive."""
        if self.ping_times is not None:
            every = self._timed_blocks()
            every.sort(key=lambda x: (x[0], x[1], x[2]))
            return [(bid, data) for _, _, _, bid, data in every]
        stream = self.whole.stream()
        every = [x for k in range(len(self.boards)) for x in self._board_blocks(k, stream)]
        every.sort(key=lambda x: (x[0], x[1]))
        return [(bid, data) for _, _, bid, data in every]

    def send(self, port: int, host: str = "127.0.0.1", drop=(), pace: float = 0.0) -> threading.Thread:
        """Send every board's blocks to one UDP port from a background thread (`drop`: indices into blocks() to leave
        out; `pace` > 0: that fraction of real time)."""
        drop = set(drop)
        for (_, _, c), bs in zip(self.boards, self.block_samples):
            if HEADER_BYTES + bs * c * self.sample_bytes > MAX_DATAGRAM:
                raise ValueError(f"a block of {bs} samples of {c} channels does not fit a UDP datagram")
        blocks = self.blocks()

        def run():
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                for i, (_, block) in enumerate(blocks):
                    if i in drop:
                        continue
                    s.sendto(block, (host, port))
                    if pace > 0.0:
                        time.sleep(pace * decode_block(block, self.sample_bytes)["n_samples"] / self.sample_rate / len(self.boards))
                    elif i % 4 == 3:
                        time.sleep(0.001)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread
