"""Low-rate capture stream for a flank array (first drawn for 32 channels, 24-bit, 64 kSPS): the host
side between a transport that delivers opaque blocks of bytes and ping records.

The capture board is not designed yet, so the block layout below is ours and provisional
(docs/HOLOSCAN_NOTES.md section 8: the Sensor Bridge data plane carries opaque frames and
has no user metadata, so ping index, sample counter and gain state must travel inside the
payload). Everything about it is in this file and in the sonar config; nothing else
depends on it.

Block, little-endian, 128-byte header then samples:

    0   4s   magic "GSFB"
    4   u16  version (1)
    6   u16  n_channels
    8   u32  n_samples in this block
    12  u64  index of the block's first sample since capture start
    20  i64  sample index at which the most recent transmit started (-1: none yet)
    28  u32  ping index of that transmit
    32  u16  gain code            (carried into the record's metadata)
    34  u16  flags                (bit 0: ADC overflow in this block)
    36  u64  capture start, ns since the Unix epoch (PTP time when the board has it)
    44  ...  reserved, zero
    128 n_samples x n_channels x sample_bytes: signed samples, sample by sample

The header does not say how many bytes a sample has: `sample_bytes` (3: 24-bit, the default; 2 or
4 for boards with another word) is a parameter of the stream, set alike on the sender and on the
receiver (`source.sample_bytes` in a deployment file, `stream_sample_bytes` for build/bin/sonar_live).
Full scale is 2^(8 sample_bytes - 1) counts. Putting it in the header (there are reserved bytes
from offset 44) would be a change of the format, which is a contract with the boards.

With 6400 samples per block a block of 32 channels is 614,528 bytes, a multiple of 128 as the
Sensor Bridge data plane requires. The plain-UDP source here needs blocks that fit a datagram:
`block_samples_for` sizes them from the channel count (128 samples for 32 channels of 24 bits).

`PingAssembler` cuts the continuous stream into ping records: each transmit opens the
receive window of the sonar config, and a ping is emitted when its window is complete.
Blocks are placed by their sample counter, so blocks that arrive out of order still make
their ping; a block that never arrives discards the ping it falls in and is counted, and
the stream recovers at the next transmit. The rules are in `PingAssembler`'s docstring.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Iterator

import numpy as np

from .. import _gpusonar as core
from ..sonar import SonarModel
from .base import Source, register

MAGIC = b"GSFB"
VERSION = 1
HEADER_BYTES = 128
_HEADER = struct.Struct("<4sHHIQqIHHQ")
FLAG_OVERFLOW = 1
SAMPLE_BYTES = 3                       # the default word: 24 bits
FULL_SCALE_COUNTS = float(1 << 23)     # of the default word (full_scale_counts for another)
# Payload bytes of one block of the plain-UDP stream: what 128 samples of 32 channels of 24 bits take. Well inside
# a datagram (65,507 bytes) and inside what a socket buffer absorbs in bursts; the emulator and the tools size
# their blocks from it and the channel count, so an array of any size fits.
STREAM_PAYLOAD_BYTES = 128 * 32 * 3
MAX_DATAGRAM = 65507
STREAM_BUFFER_BYTES = 8 << 20          # the least receive buffer asked for
STREAM_BUFFER_SECONDS = 0.5            # ... or this much of the board stream, when that is more
# Blocks the stream may move past a hole before it is given up (PingAssembler): 2 lets the last block of a window
# and the one after it swap places (ours; libs/live udp_source.hpp kReorderBlocks is the same).
REORDER_BLOCKS = 2
# Raw rows a ping may hold behind holes on the decimating path before it is given up (ours: 64 MiB bounds the memory of
# a stalled ping; a whole raw window of the default flank_32 config is larger, see PingAssembler). libs/live kMaxHeldBytes.
MAX_HELD_BYTES = 64 << 20
# Pings open at once: the reorder grace keeps an earlier ping open while a later one fills (ours; libs/live kMaxOpenPings).
MAX_OPEN_PINGS = 2
# Blocks held as a possible restart; also how many in a row, consistent, confirm one (ours; libs/live kMaxHeldBlocks).
MAX_HELD_BLOCKS = 8
# Sample indices and transmit samples at or above this are refused, in both engines (they would overflow int64 sums in C++).
MAX_SAMPLE_INDEX = 1 << 62


def full_scale_counts(sample_bytes: int = SAMPLE_BYTES) -> float:
    """Counts at full scale of a signed sample of `sample_bytes` bytes."""
    if sample_bytes not in (2, 3, 4):
        raise BlockError("sample_bytes must be 2, 3 or 4")
    return float(1 << (8 * sample_bytes - 1))


def block_samples_for(n_channels: int, sample_bytes: int = SAMPLE_BYTES) -> int:
    """Samples per block so that a block of this many channels fits the stream's payload budget."""
    return max(STREAM_PAYLOAD_BYTES // (int(n_channels) * int(sample_bytes)), 1)


def stream_buffer_bytes(n_channels: int, sample_rate: float, sample_bytes: int = SAMPLE_BYTES, given: int | None = None) -> int:
    """The socket receive buffer to ask for: `given`, else the larger of 8 MB and half a second of the stream."""
    if given:
        return int(given)
    return int(max(STREAM_BUFFER_BYTES, STREAM_BUFFER_SECONDS * n_channels * sample_rate * sample_bytes))


class BlockError(ValueError):
    """A block that is not what the header says it is."""


def encode_block(counts: np.ndarray, sample_index: int, tx_sample: int = -1, ping_index: int = 0,
                 gain_code: int = 0, flags: int = 0, start_ns: int = 0, sample_bytes: int = SAMPLE_BYTES) -> bytes:
    """`counts` [n_samples, n_channels] signed integers within the word (24 bits by default) -> block bytes."""
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise BlockError("counts must be [n_samples, n_channels]")
    full = int(full_scale_counts(sample_bytes))
    if counts.size and (counts.max() >= full or counts.min() < -full):
        raise BlockError(f"sample outside the {8 * sample_bytes}-bit range")
    n, ch = counts.shape
    header = _HEADER.pack(MAGIC, VERSION, ch, n, sample_index, tx_sample, ping_index, gain_code, flags, start_ns)
    u = np.ascontiguousarray(counts, dtype="<i4").view(np.uint8).reshape(n, ch, 4)[:, :, :sample_bytes]
    return header.ljust(HEADER_BYTES, b"\0") + u.tobytes()


def decode_block(data: bytes, sample_bytes: int = SAMPLE_BYTES) -> dict:
    """Block bytes -> header fields and `counts` [n_samples, n_channels] int32."""
    if len(data) < HEADER_BYTES:
        raise BlockError("block shorter than its header")
    magic, version, ch, n, sample_index, tx_sample, ping_index, gain_code, flags, start_ns = \
        _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise BlockError("bad magic")
    if version != VERSION:
        raise BlockError(f"unsupported block version {version}")
    if len(data) != HEADER_BYTES + sample_bytes * n * ch:
        raise BlockError(f"block length does not match its header for samples of {sample_bytes} bytes")
    if sample_bytes == 3:
        raw = np.frombuffer(data, dtype=np.uint8, offset=HEADER_BYTES).reshape(n, ch, 3).astype(np.int32)
        counts = raw[:, :, 0] | (raw[:, :, 1] << 8) | (raw[:, :, 2] << 16)
        counts = np.where(counts >= (1 << 23), counts - (1 << 24), counts)
    else:
        full_scale_counts(sample_bytes)        # 2 or 4, or an error
        counts = np.frombuffer(data, dtype="<i2" if sample_bytes == 2 else "<i4", offset=HEADER_BYTES).reshape(n, ch).astype(np.int32)
    return {"n_channels": ch, "n_samples": n, "sample_index": sample_index, "tx_sample": tx_sample,
            "ping_index": ping_index, "gain_code": gain_code, "flags": flags, "start_ns": start_ns,
            "counts": counts}


class PingAssembler:
    """Continuous blocks in, ping records out.

    With `decimate` ({} for every default; keys in libs/live decimate.hpp and docs/CONFIGURATION.md)
    each ping is band-selected and decimated to complex baseband as its blocks arrive, and with
    `reference` (the shared reference signal of the sonar's boards) synchronised at its end from
    the raw stretches kept around the two reference bursts: the records that come out are complex
    baseband, and the full-rate ping is never held (but for rows waiting behind a hole, below).

    The rules (the C++ PingAssembler in libs/live udp_source.cpp follows the same ones; the
    engine-parity tests in tests/python/test_hsb_receive.py hold both to them):

    * Order. A block is placed by its `sample_index`, not by when it arrives. A jump forward leaves a
      hole; a block arriving later inside a hole fills it (`late_blocks`) and its samples go to the ping
      whose window they fall in. An earlier ping stays open beside a later one only for the reorder grace
      below, and at most MAX_OPEN_PINGS (2) are open at once (a third gives up the oldest, counted); records
      come out in transmit order.
      A ping is emitted the moment its window is complete unless an earlier ping is still open or may
      still be named (an index jump before it with a hole still open there): then it waits for that
      decision, at most `reorder_blocks` blocks. In order, nothing waits.
    * Giving up. A hole is given up whole (its samples in `lost_samples`) once the stream is
      `reorder_blocks` blocks past its end, unless it reaches into an open ping's window. An open ping is
      given up (`dropped_pings`) once the stream is `reorder_blocks` blocks past its window's end; then the
      holes up to its window's end are given up, cut at the block boundary after it (blocks assumed of the
      current block's length): a hole running on into the next ping's window keeps the rest. A later ping
      being named does not give up an open one, unless it was transmitted before that one's window closed
      (overlapping pings): then the earlier one is given up at once, as it always was.
    * Repeats and late blocks. A block behind the furthest one seen that fills no hole is a repeat
      (`repeated_blocks`, ignored) or, if it falls in a hole given up, too late (`too_late_blocks`, also in
      `late_blocks`; not placed; its samples stay in `lost_samples`; a copy of it later is a repeat).
    * Pings never named. Every block names the latest transmit (`tx_sample`, `ping_index`), so a ping all of
      whose blocks are lost shows only as a jump of the ping index between two pings in transmit order.
      The jump d (modulo 2^32: a wrap continues) counts d - 1 dropped pings when the transmits are d ping
      intervals apart, the interval being the median of the last 5 transmit gaps between consecutive indices
      and "d apart" meaning within half an interval of d intervals and longer than d - 1 of the longest of
      those 5 by more than half an interval, and only when samples between the two transmits were lost (given
      up or still awaited): the blocks of a ping lost whole name it, so they are missing from the sample
      counter, while a sender that skipped an index leaves none missing (sonar-2ht). Otherwise it is a
      renumbering and counts nothing. Failure modes: a sender that skips indices across a gap near d
      intervals that also lost samples there is counted as d - 1 lost; a real loss with jitter beyond half an
      interval is not counted. Before an interval is known (fewer than two consecutive pings heard) a jump counts nothing:
      losses are then under-counted, never invented. Pings before the first one heard are not counted.
    * A ping named late (a block behind the stream naming a transmit between two heard ones, its index in
      between) is placed in order. A block naming a transmit before the last ping already emitted or dropped
      is too late for it.
    * A ping first named after its window has opened is dropped ("joined too late") unless every sample of the
      window before that block is still in an open hole.
    * Restart. A block that does not fit the stream - another capture start (`start_ns`), behind the stream
      by more than a receive window and `reorder_blocks` + 1 blocks, or naming a transmit this stream cannot
      have named there - is held (`last_block` "held", at most MAX_HELD_BLOCKS = 8). It starts the stream
      afresh (`restarts`) when (a) held blocks of one capture make a contiguous run of two, either of another
      capture start, or of this one (a board without a clock) and ending before the first sample this stream
      heard by more than the restart span (a window and `reorder_blocks` + 1 blocks) - samples this stream
      never had, so no copy of its own blocks, and too far back to be blocks from before the join arriving
      late - or (b) 8 blocks are held in a row, no block fitting the stream in between, of one capture start
      and consistent with each other (distinct samples, transmits not going back): the old stream has gone
      quiet. A stale copy of this stream's blocks meets (a) never and (b) only in a burst of 8 with the live
      stream silent. A restart without a clock whose counter lands where this stream already was is caught
      only by (b), 8 blocks late (the held blocks are then taken, not lost). The old stream is closed (its
      open pings dropped, its holes lost, its complete pings emitted), the run taken in sample order, and the
      old capture start is never returned to: its blocks are repeats. A held block not confirmed once
      `reorder_blocks` + 2 blocks have fitted the stream after it is stale, counted as a repeat. A block up to
      `reorder_blocks` blocks before the first one heard, fitting it, is a late block.
    * Malformed: a sample index or transmit sample at or above 2^62, a transmit sample below -1, or one at or
      beyond the block's own end (a block names a transmit at or before its samples), is refused (BlockError;
      the C++ engine counts it in bad_blocks).
    * `finish()` ends the stream: holes not yet given up are counted, pings with a hole are dropped, complete
      pings waiting for an earlier one are returned; a ping cut short by the end alone is not counted.
    * Memory. Raw path: one window buffer per open ping, so at most 2 (base: 1). Decimating path: rows behind
      holes are held raw; when all pings together hold more than `max_held_bytes` (64 MiB), the earliest open
      ping is given up (counted) until under it. In order nothing is held, so it never fires on a clean stream. The decimator takes one ping at a time, in order: a ping named late in front of the
      one it is on is too late; a ping is not started while an earlier one may still be named.
    * In order, overlapping streams included, the records and counts are those of the assembler before
      2026-11 (b6816a4).

    `last_block` says what the last block was: "in_order", "gap", "late", "too_late", "repeated", "held" or
    "restart"; `last_first`, `last_samples` are its first sample and length; `used_blocks` counts blocks whose
    samples were taken. `holds(a, b)` says whether samples a..b-1 may still arrive in time."""

    def __init__(self, sonar: SonarModel, pose=None, sound_speed: float | None = None, decimate: dict | None = None,
                 reference: dict | None = None, sample_bytes: int = SAMPLE_BYTES, reorder_blocks: int = REORDER_BLOCKS,
                 max_held_bytes: int = MAX_HELD_BYTES):
        self.sonar = sonar
        self.sample_bytes = int(sample_bytes)
        if int(reorder_blocks) < 0:
            raise ValueError("reorder_blocks must be 0 or more")
        self.reorder_blocks = int(reorder_blocks)
        self.max_held_bytes = int(max_held_bytes)
        self.pose = pose if pose is not None else core.Pose()
        self.sound_speed = float(sound_speed or sonar.nominal_sound_speed)
        template = sonar.make_record(self.pose, environment={"sound_speed": self.sound_speed})
        s = template.samples
        self.sample_rate = s.sample_rate
        self.n_channels = s.n_channels
        self.window_start = int(round(s.start_time * s.sample_rate))     # samples after the transmit
        self.window_length = s.n_samples
        full_scale = template.metadata.adc_full_scale
        self.volts_per_count = (full_scale if np.isfinite(full_scale) else 1.0) / full_scale_counts(self.sample_bytes)
        self.dropped_pings = 0
        self.lost_samples = 0
        self.overflow_blocks = 0
        self.late_blocks = 0              # arrived behind a later block (placed, or too late)
        self.too_late_blocks = 0          # ... in a hole already given up: not placed
        self.repeated_blocks = 0          # samples already received (or stale): ignored
        self.restarts = 0
        self.used_blocks = 0
        self.held_peak_bytes = 0          # most raw bytes held behind holes at once (decimating path)
        self.open_peak = 0                # most pings open at once
        self.last_block = None
        self.last_first = self.last_samples = None
        self._decimator = None
        if decimate is not None:
            from ..live import decimate_options
            self._decimator = core.PingDecimator(template, decimate_options(decimate) or {}, reference)
            self.decimation = self._decimator.plan()
        self._suspects = []               # (fitting blocks seen when held, block): maybe a restart
        self._fitting = 0                 # blocks that fitted the stream
        self._held_run = 0                # blocks held one after the other, none fitting in between
        self._left = []                   # capture starts (start_ns) left by a restart: never returned to
        self._reset()

    def _reset(self) -> None:
        self._started = False
        self._frontier = self._origin = self._stream_ns = None
        self._holes = []                  # [a, b): samples not received, not yet given up (sorted, disjoint)
        self._expired = []                # [a, b): given up, the recent ones
        self._lost = []                   # [a, b): given up, ending after the last ping released (an index jump needs one, sonar-2ht)
        self._pings = []                  # open or undecided pings, in transmit order
        self._owner = None                # the ping the decimator is open for
        self._released = []               # (tx, index) of the pings emitted or dropped, the latest 8
        self._deltas = []                 # transmit gaps between consecutive indices, the latest 5
        self._first_tx = None
        self._max_named = None            # (tx, index) of the latest transmit named

    # -- holes ---------------------------------------------------------------------------------------------

    def _take_holes(self, a: int, b: int) -> list:
        """The parts of [a, b) in open holes, removed from them."""
        got, keep = [], []
        for h0, h1 in self._holes:
            x, y = max(a, h0), min(b, h1)
            if y <= x:
                keep.append([h0, h1])
                continue
            got.append((x, y))
            if h0 < x:
                keep.append([h0, x])
            if y < h1:
                keep.append([y, h1])
        self._holes = keep
        return got

    def _in_holes(self, a: int, b: int) -> bool:
        """Whether every sample of [a, b) is in an open hole."""
        for h0, h1 in self._holes:
            if h0 <= a < h1:
                a = h1
                if a >= b:
                    return True
        return a >= b

    def _lose(self, h0: int, h1: int) -> None:
        self.lost_samples += h1 - h0
        self._expired.append([h0, h1])
        self._lost = (self._lost + [[h0, h1]])[-64:]

    def _give_up(self, limit: int) -> None:
        """Give up the holes ending at or before `limit` that reach into no open ping's window."""
        keep = []
        for h0, h1 in self._holes:
            held = any(p["state"] == "open" and h0 < p["w0"] + self.window_length and h1 > p["w0"] for p in self._pings)
            if h1 <= limit and not held:
                self._lose(h0, h1)
            else:
                keep.append([h0, h1])
        self._holes = keep

    def _give_up_to(self, end: int, n: int) -> None:
        """Give up every hole below `end`, a hole across it cut at the block boundary after it."""
        keep = []
        for h0, h1 in self._holes:
            if h0 >= end:
                keep.append([h0, h1])
                continue
            cut = min(h1, h0 + -(-(end - h0) // n) * n) if n > 0 else h1
            self._lose(h0, cut)
            if cut < h1:
                keep.append([cut, h1])
        self._holes = keep

    def holds(self, a: int, b: int) -> bool:
        """Whether some of samples a..b-1 may still arrive in time: not yet reached, or in a hole not given up."""
        if self._frontier is None or b > self._frontier:
            return True
        return any(min(b, h1) > max(a, h0) for h0, h1 in self._holes)

    # -- pings ---------------------------------------------------------------------------------------------

    def _known(self, tx: int, index: int) -> bool:
        return any(p["tx"] == tx and p["index"] == index for p in self._pings) or (tx, index) in self._released

    def _neighbours(self, tx: int):
        """(tx, index) of the nearest heard transmits before and after `tx`."""
        heard = self._released + [(p["tx"], p["index"]) for p in self._pings]
        before = [h for h in heard if h[0] < tx]
        after = [h for h in heard if h[0] > tx]
        return (max(before) if before else None), (min(after) if after else None)

    def _fits(self, block: dict) -> bool:
        """Whether a block belongs to this stream where it says it does."""
        first, n, tx, index = block["sample_index"], block["n_samples"], block["tx_sample"], block["ping_index"]
        if block["start_ns"] != self._stream_ns:
            return False
        if first >= self._frontier:                       # ahead: names the latest transmit, or a later one
            if tx < 0:
                return self._max_named is None
            return self._max_named is None or tx > self._max_named[0] or (tx, index) == self._max_named
        in_hole = any(min(first + n, h1) > max(first, h0) for h0, h1 in self._holes)
        if (first < self._frontier - self._span(n) and not in_hole) or first < self._origin - self.reorder_blocks * n:
            return False                                  # (a block in a hole still open always may come)
        if tx < 0:
            return self._first_tx is None or first < self._first_tx
        if self._known(tx, index):
            return True
        if tx > first + n or self._max_named is None or tx > self._max_named[0]:
            return False
        if any(p["tx"] == tx for p in self._pings) or any(r[0] == tx for r in self._released):
            return False                                  # a heard transmit under another index
        before, after = self._neighbours(tx)              # a transmit not heard yet: its index must fit between
        if before is not None and not 0 < (index - before[1]) & 0xFFFFFFFF < (1 << 31):
            return False
        if after is not None and not 0 < (after[1] - index) & 0xFFFFFFFF < (1 << 31):
            return False
        return True

    def _span(self, n: int) -> int:
        """How far behind the furthest block a block may arrive and still belong to the stream."""
        return self.window_start + self.window_length + (self.reorder_blocks + 1) * n

    def _name(self, block: dict) -> None:
        """A block names a transmit not heard before; open its ping, in transmit order."""
        tx, index, first = block["tx_sample"], block["ping_index"], block["sample_index"]
        if self._released and tx <= self._released[-1][0]:
            return                                        # too late: a later ping was already emitted or dropped
        if self._owner is not None and self._owner["tx"] > tx:
            return                                        # too late for the decimator, already on a later ping
        if self._first_tx is None:
            self._first_tx = tx
        if self._max_named is None or tx > self._max_named[0]:
            self._max_named = (tx, index)
        w0 = tx + self.window_start
        state = "open" if first <= w0 or self._in_holes(w0, first) else "dropped"     # else joined too late
        p = {"tx": tx, "index": index, "gain": block["gain_code"], "start_ns": block["start_ns"], "overflow": False,
             "w0": w0, "covered": 0, "fed": 0, "pending": {}, "held": 0, "record": None, "state": state,
             "buffer": np.zeros((self.window_length, self.n_channels), dtype=np.int32)
             if state == "open" and not self._decimator else None}
        at = len(self._pings)
        while at and self._pings[at - 1]["tx"] > tx:
            at -= 1
        self._pings.insert(at, p)
        # Transmitted before the previous window closed: the previous ping is given up, as it always was (until
        # 2026-11 every ping open when a later one was named was). Likewise this one, if the next was transmitted
        # inside its window.
        if at and self._pings[at - 1]["state"] == "open" and tx < self._pings[at - 1]["w0"] + self.window_length:
            self._drop(self._pings[at - 1])
        if at + 1 < len(self._pings) and p["state"] == "open" and self._pings[at + 1]["tx"] < w0 + self.window_length:
            self._drop(p)
        opened = [q for q in self._pings if q["state"] == "open" and q["covered"] < self.window_length]
        for q in opened[:max(len(opened) - MAX_OPEN_PINGS, 0)]:
            self._drop(q)                                 # more open than the reorder grace allows: the oldest goes

    def _drop(self, p: dict) -> None:
        p["state"], p["buffer"], p["pending"], p["held"] = "dropped", None, {}, 0
        if p is self._owner:
            self._owner = None

    def _place(self, p: dict, block: dict, a: int, b: int) -> None:
        w0 = p["w0"]
        a, b = max(a, w0), min(b, w0 + self.window_length)
        if b <= a:
            return
        rows = block["counts"][a - block["sample_index"]:b - block["sample_index"]]
        if self._decimator:                    # filtered and decimated as it arrives, in order
            piece = (rows * self.volts_per_count).astype(np.float32)
            p["pending"][a - w0] = piece
            p["held"] += piece.nbytes
            self._drain(p)
        else:
            p["buffer"][a - w0:b - w0] = rows
        p["covered"] += b - a
        p["overflow"] |= bool(block["flags"] & FLAG_OVERFLOW)

    def _drain(self, p: dict) -> None:
        if p is not self._owner:
            return
        while p["fed"] in p["pending"]:
            piece = p["pending"].pop(p["fed"])
            p["held"] -= piece.nbytes
            self._decimator.feed(piece)
            p["fed"] += len(piece)

    def _settle_pings(self) -> None:
        """Build the records of complete pings (the decimator takes one ping at a time, in order)."""
        for p in self._pings:
            if p["state"] == "dropped" or p["state"] == "complete":
                continue
            if not self._decimator:
                if p["covered"] == self.window_length:
                    p["record"], p["buffer"], p["state"] = self._record(p), None, "complete"
                continue
            if self._owner is not p:
                if self._owner is not None or self._gap_open(p):
                    break                          # an earlier ping has the decimator, or may still be named
                self._owner = p
                self._decimator.open()
                self._drain(p)
            if p["covered"] < self.window_length:
                break
            p["record"], p["state"], self._owner = self._record(p), "complete", None
        self.held_peak_bytes = max(self.held_peak_bytes, sum(p["held"] for p in self._pings))
        self.open_peak = max(self.open_peak, sum(1 for p in self._pings if p["state"] == "open"))

    def _gap_open(self, p: dict) -> bool:
        """Whether a ping not yet heard may still be named before `p`."""
        if not self._released:
            return False
        tx, index = self._released[-1]
        if (p["index"] - index) & 0xFFFFFFFF < 2:
            return False
        return any(h0 < p["tx"] and h1 > tx for h0, h1 in self._holes)

    def _release(self, out: list, wait: bool = True) -> None:
        """Emit or count the decided pings at the head, in transmit order."""
        while self._pings and self._pings[0]["state"] in ("complete", "dropped"):
            p = self._pings[0]
            if wait and self._gap_open(p):
                break
            self._pings.pop(0)
            if self._released:
                tx, index = self._released[-1]
                d = (p["index"] - index) & 0xFFFFFFFF
                gap = p["tx"] - tx
                if d == 1:
                    self._deltas = (self._deltas + [gap])[-5:]
                elif d >= 2 and self._deltas:
                    interval = sorted(self._deltas)[len(self._deltas) // 2]
                    # d intervals within half an interval, and longer than d - 1 of the longest recent intervals
                    # by half an interval (with jitter seen, a skip of the index needs a longer gap to count)
                    # and samples lost between the two transmits: the blocks of a ping lost whole name it, so they are
                    # missing from the counter; a sender that skipped an index leaves none missing (sonar-2ht)
                    if abs(gap - d * interval) * 2 < interval and (gap - (d - 1) * max(self._deltas)) * 2 > interval and \
                            any(h0 < p["tx"] and h1 > tx for h0, h1 in self._lost + self._holes):
                        self.dropped_pings += d - 1               # pings none of whose blocks arrived
            self._released = (self._released + [(p["tx"], p["index"])])[-8:]
            self._lost = [h for h in self._lost if h[1] > p["tx"]]   # later pings are transmitted after this one
            if p["state"] == "complete":
                out.append(p["record"])
            else:
                self.dropped_pings += 1

    def _close(self, out: list, drop_open: bool) -> None:
        """End of a stream (a restart, or finish()): count what is open, return what is complete."""
        for p in self._pings:
            if p["state"] == "open" and p["covered"] < self.window_length and (
                    drop_open or any(h0 < p["w0"] + self.window_length and h1 > p["w0"] for h0, h1 in self._holes)):
                self._drop(p)
        for h0, h1 in self._holes:
            self._lose(h0, h1)
        self._holes = []
        self._settle_pings()                              # complete pings that waited for the decimator
        self._pings = [p for p in self._pings if p["state"] != "open"]
        self._owner = None
        self._release(out, wait=False)

    def finish(self) -> list:
        """The stream has ended: give up what is still awaited (counted) and return the pings that were waiting."""
        out = []
        self.repeated_blocks += len(self._suspects)
        self._suspects, self._held_run = [], 0
        if self._started:
            self._close(out, drop_open=False)
            self._reset()
        return out

    # -- blocks --------------------------------------------------------------------------------------------

    def feed(self, data: bytes) -> list:
        """Take one block; return the ping records it releases (usually none or one)."""
        block = decode_block(data, self.sample_bytes)
        if block["n_channels"] != self.n_channels:
            raise BlockError(f"block has {block['n_channels']} channels, the sonar config {self.n_channels}")
        if not 0 <= block["sample_index"] < MAX_SAMPLE_INDEX or not -1 <= block["tx_sample"] < MAX_SAMPLE_INDEX:
            raise BlockError("sample index or transmit sample out of range (at most 2^62)")
        if block["tx_sample"] >= block["sample_index"] + block["n_samples"]:
            raise BlockError("transmit sample beyond the block's own end")
        self.last_first, self.last_samples = block["sample_index"], block["n_samples"]
        out = []
        if not self._started or self._fits(block):
            self._fitting += 1
            self._held_run = 0
            stale = [h for h in self._suspects if self._fitting - h[0] > self.reorder_blocks + 2]
            if stale:
                self.repeated_blocks += len(stale)
                self._suspects = [h for h in self._suspects if all(h is not x for x in stale)]
            self.last_block = self._take(block, out)
            return out
        if block["start_ns"] in self._left:               # a capture already left: stale
            self.repeated_blocks += 1
            self.last_block = "repeated"
            return out
        self._held_run += 1
        run = self._restart_run(block)
        if run is None:
            self._suspects.append((self._fitting, block))
            if len(self._suspects) > MAX_HELD_BLOCKS:
                self._suspects.pop(0)
                self.repeated_blocks += 1
            self.last_block = "held"
            return out
        self.restarts += 1                                # a restart: the old stream ends here
        self._close(out, drop_open=True)
        if self._stream_ns != block["start_ns"]:
            self._left = (self._left + [self._stream_ns])[-8:]
        self._reset()
        self.repeated_blocks += sum(1 for _, h in self._suspects if all(h is not x for x in run))
        self._suspects, self._held_run = [], 0
        for b in sorted(run, key=lambda b: b["sample_index"]):
            if not self._started or self._fits(b):
                self._take(b, out)
            else:
                self.repeated_blocks += 1
        self.last_block = "restart"
        return out

    def _restart_run(self, block: dict):
        """The held blocks, this one among them, that confirm a restart, or None.

        (a) A contiguous run of one capture: two blocks of another capture start, or, of this stream's capture
            start (a board without a clock, its sample counter gone back), two that end before the first sample
            this stream heard (by more than the late-block allowance): samples this stream never had, so not a
            stale copy of its own blocks. A restart from a counter that lands where this stream already was is
            not caught by (a).
        (b) MAX_HELD_BLOCKS blocks held one after the other, no block fitting the stream in between, of one
            capture start and consistent with each other (distinct samples, transmits not going back): the old
            stream has gone quiet and these make a stream of their own. Stale copies would need that many in a
            row with the live stream silent."""
        ns = block["start_ns"]
        held = [h for _, h in self._suspects if h["start_ns"] == ns] + [block]
        chain, grown = [block], True
        while grown:
            grown = False
            for h in held:
                if all(h is not c for c in chain) and any(
                        h["sample_index"] + h["n_samples"] == c["sample_index"] or c["sample_index"] + c["n_samples"] == h["sample_index"]
                        for c in chain):
                    chain.append(h)
                    grown = True
        if len(chain) >= 2:
            if ns != self._stream_ns:
                return chain
            n = block["n_samples"]
            if all(c["sample_index"] + c["n_samples"] <= self._origin - self._span(n) for c in chain):
                return chain
        if self._held_run >= MAX_HELD_BLOCKS and len(held) >= MAX_HELD_BLOCKS:
            run = sorted(held, key=lambda h: h["sample_index"])
            if all(x["sample_index"] + x["n_samples"] <= y["sample_index"] and x["tx_sample"] <= y["tx_sample"]
                   for x, y in zip(run, run[1:])):
                return run
        return None

    def _take(self, block: dict, out: list) -> str:
        """A block that fits the stream: place it; return what it was."""
        first, n = block["sample_index"], block["n_samples"]
        end = first + n
        if not self._started:
            self._started = True
            self._frontier = self._origin = first
            self._stream_ns = block["start_ns"]
        if first < self._origin:                          # just before the first block heard: a late block
            self._holes.insert(0, [first, self._origin])
            self._origin = first
        frontier = self._frontier
        kind = None
        fresh = []                                        # [a, b) received for the first time
        if first < frontier:
            fresh = self._take_holes(first, min(end, frontier))
        if end > frontier:
            if first > frontier:
                self._holes.append([frontier, first])
                kind = "gap"
            fresh.append((max(first, frontier), end))
        if not fresh:
            late = self._take_expired(first, end)
            if late:
                self.late_blocks += 1
                self.too_late_blocks += 1
                return "too_late"
            self.repeated_blocks += 1
            return "repeated"
        if first < frontier and fresh[0][0] < frontier:
            self.late_blocks += 1
            kind = kind or "late"
        self.used_blocks += 1
        self._frontier = max(frontier, end)
        if block["flags"] & FLAG_OVERFLOW:
            self.overflow_blocks += 1
        if block["tx_sample"] >= 0 and not self._known(block["tx_sample"], block["ping_index"]):
            self._name(block)
        for p in self._pings:
            if p["state"] == "open":
                for a, b in fresh:
                    self._place(p, block, a, b)
        self._settle_pings()
        horizon = self.reorder_blocks * n
        for p in self._pings:
            if p["state"] == "open" and self._frontier >= p["w0"] + self.window_length + horizon:
                self._drop(p)                             # given up: its holes with it
                self._give_up_to(p["w0"] + self.window_length, n)
        self._settle_pings()
        while sum(p["held"] for p in self._pings) > self.max_held_bytes:
            first_open = next(p for p in self._pings if p["state"] == "open")
            self._drop(first_open)                        # rows held behind holes, all pings together, past the bound
            self._give_up_to(first_open["w0"] + self.window_length, n)
            self._settle_pings()
        self._give_up(self._frontier - horizon)
        self._release(out)
        low = self._frontier - self._span(n)
        self._expired = [h for h in self._expired if h[1] > low][-64:]
        return kind or "in_order"

    def _take_expired(self, a: int, b: int) -> bool:
        """Whether [a, b) falls in a hole given up; that part is then taken out (a copy later is a repeat)."""
        hit, keep = False, []
        for h0, h1 in self._expired:
            x, y = max(a, h0), min(b, h1)
            if y <= x:
                keep.append([h0, h1])
                continue
            hit = True
            if h0 < x:
                keep.append([h0, x])
            if y < h1:
                keep.append([y, h1])
        self._expired = keep
        return hit

    def _record(self, p: dict) -> "core.PingRecord":
        t_tx_ns = p["start_ns"] + int(round(p["tx"] / self.sample_rate * 1e9))
        record = self.sonar.make_record(self.pose, sequence=int(p["index"]), time_ns=t_tx_ns,
                                        environment={"sound_speed": self.sound_speed})
        if self._decimator:
            self._decimator.finish(record)         # complex baseband in place, boards synchronised
        else:
            s = record.samples
            kind = s.kind
            s.array = (p["buffer"].T * self.volts_per_count).astype(np.float32)
            s.kind = kind
            s.start_time = self.window_start / self.sample_rate
            s.storage = core.StorageType.Int32
            s.storage_scale = self.volts_per_count
            record.samples = s
        m = record.metadata
        extra = json.loads(m.extra_json or "{}")
        extra["flank_stream"] = {"gain_code": int(p["gain"]), "adc_overflow": bool(p["overflow"]),
                                 "tx_sample": int(p["tx"])}
        m.extra_json = json.dumps(extra)
        record.metadata = m
        return record


def window_sound_speed(value) -> float | None:
    """`source.sound_speed` of a deployment file as a number, or None for the sonar file's nominal value (absent, None
    or 0). Anything else that is not a positive, finite number of m/s is refused: the receive window is laid out by it."""
    if value is None:
        return None
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"source.sound_speed: a number of m/s, not {value!r}")
    c = float(value)
    if c == 0.0:
        return None
    if not np.isfinite(c) or c < 0.0:
        raise ValueError(f"source.sound_speed: a positive number of m/s, not {value!r}")
    return c


@register
class FlankUdpSource(Source):
    """Ping records from blocks arriving as UDP datagrams: a stand-in for the Sensor
    Bridge socket receiver, and the receive side of the emulator test.

    One thread drains the socket and a second assembles pings, so slow processing
    downstream delays records but does not lose samples (up to `queue_pings` waiting
    records; beyond that the oldest waiting ping is dropped and counted in `overrun_pings`).
    Blocks waiting to be assembled may hold `queue_mb` megabytes; beyond that arriving blocks
    are dropped and counted in `dropped_blocks` (0: no bound).
    Iteration ends after `max_pings` records or `timeout` seconds without one (0 or None: never).

    `sonar` has no default: the array the blocks come from must be named. `sample_bytes` is the
    boards' word (see the top of this file). `sound_speed` (m/s; None or 0: the sonar file's
    `nominal_sound_speed`) is the speed at which the sonar file's `receive.window.range` becomes
    the window's first sample after the transmit and its length, and the speed the records carry
    until navigation or `chain.sound_speed` replaces it. It is a statement of the layout the
    boards' timing was set up for (the synchronisation bursts are looked for from the window's
    two ends), not a measurement: the same value in `tools/sonar_live.py --export-template`
    (the C++ program's template) and `tools/sonar_live_hsb.py receive`.
    The socket's receive buffer is `buffer_bytes`, or
    the larger of 8 MB and half a second of this sonar's stream; `buffer_granted` is what the
    system gave, and a warning says so when that is less.
    """

    type_name = "flank_udp"

    def __init__(self, sonar: str, port: int = 0, host: str = "127.0.0.1",
                 sonar_overrides: dict | None = None, max_pings: int | None = None, timeout: float | None = 2.0,
                 queue_pings: int = 8, decimate: dict | None = None, reference: dict | None = None,
                 sample_bytes: int = SAMPLE_BYTES, buffer_bytes: int | None = None, queue_mb: float = 1024.0,
                 sound_speed: float | None = None):
        import queue
        import sys
        import warnings
        model = SonarModel.load(sonar, sonar_overrides)
        sound_speed = window_sound_speed(sound_speed)
        if decimate is not None and reference is None and len(model.boards) > 1:
            reference = (decimate.get("reference") if isinstance(decimate, dict) else None) or model.config["receive"].get("reference")
        self.assembler = PingAssembler(model, sound_speed=sound_speed, decimate=decimate, reference=reference, sample_bytes=sample_bytes)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stream_bytes_per_second = self.assembler.n_channels * self.assembler.sample_rate * int(sample_bytes)
        self.buffer_asked = stream_buffer_bytes(self.assembler.n_channels, self.assembler.sample_rate, int(sample_bytes), buffer_bytes)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, min(self.buffer_asked, 0x7fffffff))
        got = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.buffer_granted = got // 2 if sys.platform.startswith("linux") else got    # Linux reports twice what is usable
        if self.buffer_granted < self.buffer_asked:
            warnings.warn(f"flank_udp: asked for a {self.buffer_asked}-byte socket buffer, the system granted {self.buffer_granted} "
                          f"({self.buffer_granted / max(self.stream_bytes_per_second, 1.0):.2f} s of the stream): raise net.core.rmem_max, "
                          "or blocks may be lost while a ping is processed", RuntimeWarning, stacklevel=2)
        self.socket.bind((host, port))
        self.socket.settimeout(0.2)
        self.port = self.socket.getsockname()[1]
        self.max_pings = max_pings
        self.timeout = float(timeout) if timeout else None
        self.bad_blocks = 0
        self.overrun_pings = 0
        self.pings_discarded_at_stop = 0   # complete pings the assembler still held at close()
        self.dropped_blocks = 0
        self._queue_limit = int(queue_mb * 1024 * 1024)
        self._queued_bytes = 0
        self._queue = queue.Queue(maxsize=queue_pings)
        self._blocks = queue.Queue()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._receive, daemon=True),
                         threading.Thread(target=self._assemble, daemon=True)]
        for thread in self._threads:
            thread.start()

    def _receive(self) -> None:
        """Drain the socket and nothing else: assembling a finished ping takes tens of
        milliseconds, long enough to overflow a small socket buffer."""
        while not self._stop.is_set():
            try:
                data = self.socket.recv(65535)
                if self._queue_limit and self._queued_bytes + len(data) > self._queue_limit:
                    self.dropped_blocks += 1               # the assembler sees the gap and drops the ping it falls in
                    continue
                self._queued_bytes += len(data)
                self._blocks.put(data)
            except socket.timeout:
                continue
            except OSError:
                return

    def _assemble(self) -> None:
        import queue
        while not self._stop.is_set():
            try:
                data = self._blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            self._queued_bytes -= len(data)
            try:
                records = self.assembler.feed(data)
            except BlockError:
                self.bad_blocks += 1
                continue
            for r in records:
                try:
                    self._queue.put_nowait(r)
                except queue.Full:
                    self._queue.get_nowait()
                    self._queue.put_nowait(r)
                    self.overrun_pings += 1

    def __iter__(self) -> Iterator["core.PingRecord"]:
        import queue
        count = 0
        while self.max_pings is None or count < self.max_pings:
            try:
                record = self._queue.get(timeout=self.timeout)
            except queue.Empty:
                return
            count += 1
            yield record

    def close(self) -> None:
        self._stop.set()
        self.socket.close()
        for thread in self._threads:
            thread.join(timeout=2.0)
        if not any(thread.is_alive() for thread in self._threads):
            # The end of the stream: what is still awaited is counted; complete pings still waiting are not used.
            self.pings_discarded_at_stop += len(self.assembler.finish())


class FlankBoardEmulator:
    """Plays simulated ping records as the capture board would send them: one continuous
    24-bit stream, zeros outside the receive windows, cut into blocks."""

    def __init__(self, records, ping_interval: float, block_samples: int | None = None, start_ns: int = 0,
                 gain_code: int = 0, sample_bytes: int = SAMPLE_BYTES):
        self.records = list(records)
        first = self.records[0].samples
        self.sample_rate = first.sample_rate
        self.sample_bytes = int(sample_bytes)
        # Blocks sized from the channel count, so that any array's block fits a datagram (128 samples for 32 channels).
        self.block_samples = int(block_samples) if block_samples else block_samples_for(first.n_channels, self.sample_bytes)
        self.n_channels = first.n_channels
        self.period = int(round(ping_interval * self.sample_rate))
        self.window_start = int(round(first.start_time * self.sample_rate))
        full_scale = self.records[0].metadata.adc_full_scale
        self.full_counts = int(full_scale_counts(self.sample_bytes))
        self.counts_per_volt = self.full_counts / (full_scale if np.isfinite(full_scale) else 1.0)
        self.start_ns = int(start_ns)
        self.gain_code = int(gain_code)
        if self.window_start + first.n_samples > self.period:
            raise ValueError("the receive window is longer than the ping interval")

    def blocks(self) -> Iterator[bytes]:
        """Every block of the run, in order."""
        n_ch = self.records[0].samples.n_channels
        total = self.period * len(self.records)
        stream = np.zeros((total, n_ch), dtype=np.int32)
        for i, r in enumerate(self.records):
            x = np.rint(r.samples.array.T * self.counts_per_volt)
            a = i * self.period + self.window_start
            stream[a:a + len(x)] = np.clip(x, -self.full_counts, self.full_counts - 1).astype(np.int32)
        for first in range(0, total, self.block_samples):
            ping = first // self.period                     # the transmit at or before this block
            last_tx = ping * self.period
            chunk = stream[first:first + self.block_samples]
            yield encode_block(chunk, first, last_tx, ping, self.gain_code, 0, self.start_ns, self.sample_bytes)

    def send(self, port: int, host: str = "127.0.0.1", drop=(), pace: float = 0.0) -> threading.Thread:
        """Send every block to a UDP port from a background thread. `drop` lists block
        numbers to leave out; `pace` > 0 sleeps to that fraction of real time."""
        drop = set(drop)
        # Blocks of any size can be handed to an assembler (the Sensor Bridge carries 6400 samples); over UDP one must fit a datagram.
        if HEADER_BYTES + self.block_samples * self.n_channels * self.sample_bytes > MAX_DATAGRAM:
            raise ValueError(f"a block of {self.block_samples} samples of {self.n_channels} channels does not fit a UDP datagram: "
                             f"at most {(MAX_DATAGRAM - HEADER_BYTES) // (self.n_channels * self.sample_bytes)} (leave block_samples out)")

        def run():
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                for i, block in enumerate(self.blocks()):
                    if i in drop:
                        continue
                    s.sendto(block, (host, port))
                    if pace > 0.0:
                        time.sleep(pace * self.block_samples / self.sample_rate)
                    elif i % 4 == 3:
                        time.sleep(0.001)                   # let the receiver drain its buffer

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread
