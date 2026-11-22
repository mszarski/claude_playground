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
    baseband, and the full-rate ping is never held (but for blocks waiting behind a gap, below).

    The rules (the C++ PingAssembler in libs/live udp_source.cpp follows the same ones, held by the
    engine-parity tests):

    * Order. A block is placed by its `sample_index`, not by when it arrives. A jump forward leaves a
      hole; a block that later arrives inside a hole fills it (`late_blocks`) and its samples go into the
      ping being assembled. A ping is emitted the moment its window is complete, so reordering adds no
      output latency. With `decimate`, samples behind a hole wait (raw) until the hole is filled.
    * Giving up. A hole is given up whole (its samples counted in `lost_samples`) once the stream has moved
      `reorder_blocks` blocks past its end; a hole reaching into the open ping's window is held as long as the
      ping is open, and given up with it. The open ping is given up (`dropped_pings`) when a later ping is named, when the stream has
      moved `reorder_blocks` blocks past the window's end, or at a restart. So a dropped ping is counted up to
      `reorder_blocks` blocks after its window ends, not at the gap; complete pings are never delayed.
    * A block arriving behind the furthest one seen that fills no hole is a repeat (`repeated_blocks`,
      ignored) or, when it falls in a hole already given up, too late (`too_late_blocks`, also in
      `late_blocks`; not placed, its samples stay in `lost_samples`).
    * Restart: a block with another capture start (`start_ns`), or behind the stream by more than
      a receive window and `reorder_blocks` + 1 blocks, or before the first block heard, or naming a transmit
      this stream has not named recently (the header's `tx_sample`, `ping_index`), starts the stream afresh
      (`restarts`): the open ping is dropped, open holes are counted lost, and pings before the first one
      named afterwards are not counted (as when joining). A board without a clock (`start_ns` constant) that
      restarts within that span of its own start cannot be told from a repeat: its blocks count as repeats
      until one names a transmit not heard before.
    * Pings never named. Every block names the latest transmit, so a ping all of whose blocks are lost
      shows only as a jump of the header's `ping_index`: a jump of d (modulo 2^32, so a wrap counts as
      continuing) counts d - 1 dropped pings, if the transmits are far enough apart for d windows (d x
      (window_start + window_length) samples); otherwise the numbering restarted and nothing is counted.
      Pings before the first one named (joining late, or after a restart) are not counted.
    * A ping first named after its window has opened is dropped ("joined too late") unless every sample of
      the window before that block is still in an open hole (its blocks may yet arrive).

    `last_block` says what the last block was: "in_order", "gap" (it opened a hole), "late", "too_late",
    "repeated" or "restart"; `last_first`, `last_samples` are its first sample and length. `holds(a, b)` says
    whether samples a..b-1 may still arrive in time (FrameReceiver maps frame numbers onto it)."""

    def __init__(self, sonar: SonarModel, pose=None, sound_speed: float | None = None, decimate: dict | None = None,
                 reference: dict | None = None, sample_bytes: int = SAMPLE_BYTES, reorder_blocks: int = REORDER_BLOCKS):
        self.sonar = sonar
        self.sample_bytes = int(sample_bytes)
        if int(reorder_blocks) < 0:
            raise ValueError("reorder_blocks must be 0 or more")
        self.reorder_blocks = int(reorder_blocks)
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
        self.too_late_blocks = 0          # ... after their hole was given up: not placed
        self.repeated_blocks = 0          # samples already received: ignored
        self.restarts = 0
        self.last_block = None
        self.last_first = self.last_samples = None
        self._decimator = None
        if decimate is not None:
            from ..live import decimate_options
            self._decimator = core.PingDecimator(template, decimate_options(decimate) or {}, reference)
            self.decimation = self._decimator.plan()
        self._reset()

    def _reset(self) -> None:
        self._frontier = None             # end of the furthest block seen
        self._origin = None               # first sample heard (since the start or the last restart)
        self._stream_ns = None            # the stream's capture start
        self._holes = []                  # [a, b): samples not received, not yet given up (sorted, disjoint)
        self._expired = []                # [a, b): holes given up, the recent ones
        self._named = []                  # (tx_sample, ping_index) of the latest transmits named
        self._first_tx = None
        self._ping = None                 # the ping being filled

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

    def _give_up(self, limit: int, window: tuple | None = None) -> None:
        """Give up the holes that end at or before `limit`, but those reaching into the open ping's window, and
        every hole reaching into `window` (a ping given up)."""
        keep = []
        for h0, h1 in self._holes:
            held = self._ping is not None and h0 < self._ping["w0"] + self.window_length and h1 > self._ping["w0"]
            if (window is not None and h0 < window[1] and h1 > window[0]) or (h1 <= limit and not held):
                self.lost_samples += h1 - h0
                self._expired.append([h0, h1])
            else:
                keep.append([h0, h1])
        self._holes = keep

    def holds(self, a: int, b: int) -> bool:
        """Whether some of samples a..b-1 may still arrive in time: not yet reached, or in a hole not given up."""
        if self._frontier is None or b > self._frontier:
            return True
        return any(min(b, h1) > max(a, h0) for h0, h1 in self._holes)

    def _span(self, n: int) -> int:
        """How far behind the furthest block a block may arrive and still be told apart from a restart."""
        return self.window_start + self.window_length + (self.reorder_blocks + 1) * n

    def _consistent(self, block: dict) -> bool:
        """A block behind the stream whose header names what this stream named there."""
        if block["tx_sample"] < 0:
            return self._first_tx is None or block["sample_index"] < self._first_tx
        return (block["tx_sample"], block["ping_index"]) in self._named

    # -- pings ---------------------------------------------------------------------------------------------

    def _abandon(self) -> None:
        """The open ping cannot complete: count it, and give up the holes in its window."""
        p = self._ping
        self._ping = None
        self.dropped_pings += 1
        self._give_up(self._frontier - self.reorder_blocks * (self.last_samples or 0), (p["w0"], p["w0"] + self.window_length))

    def _name(self, block: dict) -> None:
        """A block names a transmit later than any named so far."""
        tx, index, first = block["tx_sample"], block["ping_index"], block["sample_index"]
        if self._named:
            last_tx, last_index = self._named[-1]
            d = (index - last_index) & 0xFFFFFFFF
            if d >= 2 and d * (self.window_start + self.window_length) <= tx - last_tx:
                self.dropped_pings += d - 1               # pings none of whose blocks arrived
        else:
            self._first_tx = tx
        self._named = (self._named + [(tx, index)])[-4:]
        if self._ping is not None:                       # the previous window never completed
            self._abandon()
        w0 = tx + self.window_start
        if first > w0 and not self._in_holes(w0, first):
            self.dropped_pings += 1                       # joined too late for this ping
            return
        self._ping = {"tx": tx, "index": index, "gain": block["gain_code"], "start_ns": block["start_ns"],
                      "overflow": False, "w0": w0, "covered": 0, "fed": 0, "pending": {},
                      "buffer": None if self._decimator else np.zeros((self.window_length, self.n_channels), dtype=np.int32)}
        if self._decimator:
            self._decimator.open()

    def _place(self, block: dict, a: int, b: int) -> None:
        p = self._ping
        w0 = p["w0"]
        a, b = max(a, w0), min(b, w0 + self.window_length)
        if b <= a:
            return
        rows = block["counts"][a - block["sample_index"]:b - block["sample_index"]]
        if self._decimator:                    # filtered and decimated as it arrives, in order
            p["pending"][a - w0] = (rows * self.volts_per_count).astype(np.float32)
            while p["fed"] in p["pending"]:
                piece = p["pending"].pop(p["fed"])
                self._decimator.feed(piece)
                p["fed"] += len(piece)
        else:
            p["buffer"][a - w0:b - w0] = rows
        p["covered"] += b - a
        p["overflow"] |= bool(block["flags"] & FLAG_OVERFLOW)

    def feed(self, data: bytes) -> list:
        """Take one block; return the ping records it completes (none or one)."""
        block = decode_block(data, self.sample_bytes)
        if block["n_channels"] != self.n_channels:
            raise BlockError(f"block has {block['n_channels']} channels, the sonar config {self.n_channels}")
        first, n = block["sample_index"], block["n_samples"]
        end = first + n
        self.last_first, self.last_samples = first, n
        kind = None
        if self._frontier is not None and (
                block["start_ns"] != self._stream_ns or
                (first < self._frontier and (first < max(self._origin, self._frontier - self._span(n)) or
                                             not self._consistent(block)))):
            self.restarts += 1                            # a restart: whatever was open is lost
            if self._ping is not None:
                self._ping = None
                self.dropped_pings += 1
            self.lost_samples += sum(h1 - h0 for h0, h1 in self._holes)
            self._reset()
            kind = "restart"
        if self._frontier is None:
            self._frontier = self._origin = first
            self._stream_ns = block["start_ns"]
        frontier = self._frontier
        fresh = []                                        # [a, b) received for the first time
        if first < frontier:
            fresh = self._take_holes(first, min(end, frontier))
        if end > frontier:
            if first > frontier:
                self._holes.append([frontier, first])
                kind = kind or "gap"
            fresh.append((max(first, frontier), end))
        out = []
        if not fresh:
            if any(min(end, h1) > max(first, h0) for h0, h1 in self._expired):
                self.late_blocks += 1
                self.too_late_blocks += 1
                self.last_block = "too_late"
            else:
                self.repeated_blocks += 1
                self.last_block = "repeated"
            return out
        if first < frontier and fresh[0][0] < frontier:
            self.late_blocks += 1
            kind = kind or "late"
        self.last_block = kind or "in_order"
        self._frontier = max(frontier, end)
        if block["flags"] & FLAG_OVERFLOW:
            self.overflow_blocks += 1
        if first >= frontier and block["tx_sample"] >= 0 and (not self._named or block["tx_sample"] > self._named[-1][0]):
            self._name(block)
        p = self._ping
        if p is not None:
            for a, b in fresh:
                self._place(block, a, b)
            if p["covered"] == self.window_length:
                out.append(self._record(p))
                self._ping = None
        horizon = self.reorder_blocks * n
        if self._ping is not None and self._frontier >= self._ping["w0"] + self.window_length + horizon:
            self._abandon()
        self._give_up(self._frontier - horizon)
        low = self._frontier - self._span(n)
        self._expired = [h for h in self._expired if h[1] > low][-64:]
        return out

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
    boards' word (see the top of this file). The socket's receive buffer is `buffer_bytes`, or
    the larger of 8 MB and half a second of this sonar's stream; `buffer_granted` is what the
    system gave, and a warning says so when that is less.
    """

    type_name = "flank_udp"

    def __init__(self, sonar: str, port: int = 0, host: str = "127.0.0.1",
                 sonar_overrides: dict | None = None, max_pings: int | None = None, timeout: float | None = 2.0,
                 queue_pings: int = 8, decimate: dict | None = None, reference: dict | None = None,
                 sample_bytes: int = SAMPLE_BYTES, buffer_bytes: int | None = None, queue_mb: float = 1024.0):
        import queue
        import sys
        import warnings
        model = SonarModel.load(sonar, sonar_overrides)
        if decimate is not None and reference is None and len(model.boards) > 1:
            reference = (decimate.get("reference") if isinstance(decimate, dict) else None) or model.config["receive"].get("reference")
        self.assembler = PingAssembler(model, decimate=decimate, reference=reference, sample_bytes=sample_bytes)
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
