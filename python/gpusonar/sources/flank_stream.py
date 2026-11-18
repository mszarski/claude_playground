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
A gap in the sample counter (a lost block) discards the ping it falls in and is counted;
the stream recovers at the next transmit.
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
    baseband, and the full-rate ping is never held."""

    def __init__(self, sonar: SonarModel, pose=None, sound_speed: float | None = None, decimate: dict | None = None,
                 reference: dict | None = None, sample_bytes: int = SAMPLE_BYTES):
        self.sonar = sonar
        self.sample_bytes = int(sample_bytes)
        self.pose = pose if pose is not None else core.Pose()
        self.sound_speed = float(sound_speed or sonar.config.get("nominal_sound_speed", 1500.0))
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
        self._decimator = None
        if decimate is not None:
            from ..live import decimate_options
            self._decimator = core.PingDecimator(template, decimate_options(decimate) or {}, reference)
            self.decimation = self._decimator.plan()
        self._next_sample = None          # the sample index the next block should start at
        self._ping = None                 # the ping being filled: dict(tx, index, buffer, filled, ...)
        self._done_ping = None            # ping index already emitted or discarded

    def _open(self, block: dict) -> None:
        self._ping = {"tx": block["tx_sample"], "index": block["ping_index"], "gain": block["gain_code"],
                      "start_ns": block["start_ns"], "overflow": False, "filled": 0,
                      "buffer": None if self._decimator else np.zeros((self.window_length, self.n_channels), dtype=np.int32)}
        if self._decimator:
            self._decimator.open()

    def feed(self, data: bytes) -> list:
        """Take one block; return the ping records it completes (usually none or one)."""
        block = decode_block(data, self.sample_bytes)
        if block["n_channels"] != self.n_channels:
            raise BlockError(f"block has {block['n_channels']} channels, the sonar config {self.n_channels}")
        first, n = block["sample_index"], block["n_samples"]
        if self._next_sample is not None and first != self._next_sample:
            # A gap (or a restart). Whatever ping it cuts through is lost.
            self.lost_samples += max(first - self._next_sample, 0)
            if self._ping is not None:
                w0 = self._ping["tx"] + self.window_start
                if first > w0 + self._ping["filled"] or first < self._next_sample:
                    self.dropped_pings += 1
                    self._done_ping = self._ping["index"]
                    self._ping = None
        self._next_sample = first + n
        if block["flags"] & FLAG_OVERFLOW:
            self.overflow_blocks += 1
        out = []
        tx = block["tx_sample"]
        if tx >= 0 and block["ping_index"] != self._done_ping and (
                self._ping is None or self._ping["index"] != block["ping_index"]):
            if self._ping is not None:                       # the previous window never completed
                self.dropped_pings += 1
            if first > tx + self.window_start:               # joined too late for this ping
                self.dropped_pings += 1
                self._done_ping = block["ping_index"]
                self._ping = None
            else:
                self._open(block)
        if self._ping is not None:
            p = self._ping
            w0 = p["tx"] + self.window_start
            a = max(first, w0 + p["filled"])
            b = min(first + n, w0 + self.window_length)
            if b > a:
                if a != w0 + p["filled"]:
                    self.dropped_pings += 1
                    self._done_ping = p["index"]
                    self._ping = None
                    return out
                if self._decimator:                # filtered and decimated as it arrives
                    self._decimator.feed((block["counts"][a - first:b - first] * self.volts_per_count).astype(np.float32))
                else:
                    p["buffer"][a - w0:b - w0] = block["counts"][a - first:b - first]
                p["filled"] = b - w0
                p["overflow"] |= bool(block["flags"] & FLAG_OVERFLOW)
            if p["filled"] == self.window_length:
                out.append(self._record(p))
                self._done_ping = p["index"]
                self._ping = None
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
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_asked)
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
        if HEADER_BYTES + self.block_samples * first.n_channels * self.sample_bytes > MAX_DATAGRAM:
            raise ValueError(f"a block of {self.block_samples} samples of {first.n_channels} channels does not fit a UDP datagram: "
                             f"at most {(MAX_DATAGRAM - HEADER_BYTES) // (first.n_channels * self.sample_bytes)}")
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
