"""The flank capture stream carried by NVIDIA Holoscan Sensor Bridge: the board side played
by Sensor Bridge's own emulator, the host side received by its `LinuxReceiverOp`.

    ping records -> fixed-size frames -> hololink.emulation (HSBEmulator, LinuxDataPlane)
      -> UDP on an ordinary network interface (127.0.0.1 for a test on one machine)
      -> hololink.operators.LinuxReceiverOp -> HsbFramesToPingsOp -> LiveChainOp -> reports

A frame is one block of sources/flank_stream.py with a fixed number of samples, so every
frame has the same size, as the Sensor Bridge data plane needs (the receiver is told one
frame size when it starts). The stream is continuous: zeros outside the receive windows,
and it starts `lead_seconds` before the first transmit because a receiver joins a running
stream and Sensor Bridge's receiver keeps only the newest frame until the application
takes its first one.

Nothing here needs Sensor Bridge or Holoscan to import: the framing and the frame handling
(`frames_from_records`, `FrameReceiver`) are plain Python, `hololink` is imported inside the
functions that use it (tools/build_hsb.sh builds it), and the operators exist only where
`holoscan` imports. Sensor Bridge is Apache-2.0 and is called, not copied.

What the receiver operator gives the next operator (Sensor Bridge 2.7.0, read from its
sources; the operators here are tested in the Holoscan runtime behind a stand-in that emits
the same): one message per frame holding a single tensor
named "" - uint8, one-dimensional, `bytes_written` long, in GPU memory, always at the same
address (the next frame overwrites it, so it is copied to the host here before anything
else) - and, as Holoscan metadata, among others `frame_number` (counted by the sender),
`frame_bytes_received`, `packets_dropped` (cumulative, from the packet sequence numbers),
`bytes_written` and `timestamp_s`/`timestamp_ns`.
"""

from __future__ import annotations

import ctypes
import math
import time
from typing import Iterable, Iterator

import numpy as np

from ..sonar import SonarModel
from .flank_stream import FULL_SCALE_COUNTS, HEADER_BYTES, REORDER_BLOCKS, BlockError, PingAssembler, encode_block

FRAME_SAMPLES = 400            # 6.25 ms of the 64 kSPS stream: 38,528 bytes with 32 channels, 160 frames a second.
# A frame leaves the sender as one burst of 29 datagrams (1408-byte payloads) into the receiver's
# socket buffer, which Sensor Bridge's receiver leaves at the system default unless a frame is
# larger: 212,992 bytes where net.core.rmem_max is that (Hugging Face containers; it cannot be
# raised there), which on loopback is 92 datagrams of 2304 bytes of kernel accounting, about 3
# frames or 20 ms of the stream at real time. Measured (job 6b010e824fdc5ab3c55c549d, L4): every
# datagram lost at real time was refused by that full buffer (the socket's drop count equals the
# packet-number gaps and RcvbufErrors), while the receiving process ran the chain on a ping, which
# delays both Sensor Bridge's receiver thread and the sending process; a sender that had fallen
# behind then sent the frames it owed back to back. Hence catch_up="spread" (below) by default.
# On loopback 1600-sample frames (110 datagrams) lost 5 frames in 502 and 800-sample frames (55) 1 in 1003.
FRAME_ALIGN = 128              # the Sensor Bridge data plane moves frames in 128-byte pages


def frame_bytes(n_channels: int, frame_samples: int = FRAME_SAMPLES) -> int:
    """Size of every frame; a multiple of 128 bytes or the configuration is refused."""
    size = HEADER_BYTES + 3 * int(frame_samples) * int(n_channels)
    if size % FRAME_ALIGN:
        raise ValueError(f"{frame_samples} samples of {n_channels} channels make a {size}-byte frame, "
                         f"not a multiple of {FRAME_ALIGN}")
    return size


class FrameStream:
    """Ping records as the board's continuous stream, cut into frames of `frame_samples`.

    Each ping's transmit sits at its record's transmit time after the first one (or every
    `ping_interval` seconds when that is given), `lead_seconds` after the stream starts.
    The stream's start time is the first transmit time less the lead, so assembled records
    carry the original times; records that start at time zero come out `lead_seconds` later.
    The tail is padded with zeros to a whole frame. `idle()` continues the stream with
    frames of zeros, as a board with nothing in the water would.
    """

    def __init__(self, records, frame_samples: int = FRAME_SAMPLES, ping_interval: float | None = None,
                 lead_seconds: float = 0.0, start_ns: int | None = None, gain_code: int = 0):
        self.records = list(records)
        if not self.records:
            raise ValueError("no ping records to send")
        first = self.records[0].samples
        self.sample_rate = float(first.sample_rate)
        self.n_channels = int(first.n_channels)
        self.frame_samples = int(frame_samples)
        self.frame_bytes = frame_bytes(self.n_channels, self.frame_samples)
        self.window_start = int(round(first.start_time * self.sample_rate))
        self.window_length = int(first.n_samples)
        full_scale = self.records[0].metadata.adc_full_scale
        self.counts_per_volt = FULL_SCALE_COUNTS / (full_scale if np.isfinite(full_scale) else 1.0)
        self.gain_code = int(gain_code)
        lead = int(round(lead_seconds * self.sample_rate))
        t0 = int(self.records[0].transmit.time_ns)
        if ping_interval is not None:
            period = int(round(ping_interval * self.sample_rate))
            self.tx = [lead + i * period for i in range(len(self.records))]
        else:
            self.tx = [lead + int(round((int(r.transmit.time_ns) - t0) * 1e-9 * self.sample_rate)) for r in self.records]
        self.ping_index = [int(r.metadata.sequence) & 0xFFFFFFFF for r in self.records]
        if len(set(self.ping_index)) != len(self.ping_index):
            self.ping_index = list(range(len(self.records)))
        self.start_ns = int(start_ns) if start_ns is not None else max(t0 - int(round(lead / self.sample_rate * 1e9)), 0)
        for i, r in enumerate(self.records):
            s = r.samples
            if (s.n_channels, s.n_samples, int(round(s.start_time * s.sample_rate))) != (self.n_channels, self.window_length, self.window_start):
                raise ValueError(f"ping {i} has a different receive window or channel count from the first")
            if i and self.tx[i] < self.tx[i - 1] + self.window_start + self.window_length:
                raise ValueError(f"ping {i} is transmitted before the receive window of ping {i - 1} has closed "
                                 "(give ping_interval if the records carry no transmit times)")
            # A frame's header names the transmit at or before its first sample, and the assembler
            # must hear of a transmit before that ping's window opens.
            if -self.tx[i] % self.frame_samples > self.window_start:
                raise ValueError(f"a frame of {self.frame_samples} samples is longer than the {self.window_start} samples "
                                 "between a transmit and its receive window: use shorter frames")
        end = self.tx[-1] + self.window_start + self.window_length
        self.n_frames = -(-end // self.frame_samples)
        self.frame_seconds = self.frame_samples / self.sample_rate

    def _counts(self, i: int) -> np.ndarray:
        x = np.rint(self.records[i].samples.array.T * self.counts_per_volt)
        return np.clip(x, -(1 << 23), (1 << 23) - 1).astype(np.int32)

    def _encode(self, chunk: np.ndarray, first: int, last_tx: int) -> bytes:
        if last_tx < 0:
            return encode_block(chunk, first, -1, 0, self.gain_code, 0, self.start_ns)
        return encode_block(chunk, first, self.tx[last_tx], self.ping_index[last_tx], self.gain_code, 0, self.start_ns)

    def __iter__(self) -> Iterator[bytes]:
        n, w0, wl = self.frame_samples, self.window_start, self.window_length
        last_tx, cached = -1, (None, None)                  # one ping's samples as counts at a time
        for k in range(self.n_frames):
            first = k * n
            while last_tx + 1 < len(self.tx) and self.tx[last_tx + 1] <= first:
                last_tx += 1
            chunk = np.zeros((n, self.n_channels), dtype=np.int32)
            for i in (last_tx, last_tx + 1):                # windows of the last transmit and of one inside this frame
                if not 0 <= i < len(self.tx):
                    continue
                a, b = max(first, self.tx[i] + w0), min(first + n, self.tx[i] + w0 + wl)
                if b > a:
                    if cached[0] != i:
                        cached = (i, self._counts(i))
                    chunk[a - first:b - first] = cached[1][a - self.tx[i] - w0:b - self.tx[i] - w0]
            yield self._encode(chunk, first, last_tx)

    def idle(self) -> Iterator[bytes]:
        """Frames of zeros that continue the stream after the last ping, without end."""
        chunk = np.zeros((self.frame_samples, self.n_channels), dtype=np.int32)
        k = self.n_frames
        while True:
            yield self._encode(chunk, k * self.frame_samples, len(self.tx) - 1)
            k += 1


def frames_from_records(records, frame_samples: int = FRAME_SAMPLES, **options) -> Iterator[bytes]:
    """Every frame of the run, in order, all the same size. Options as `FrameStream`."""
    return iter(FrameStream(records, frame_samples, **options))


MAX_AWAITED = 1024             # frame numbers of one gap awaited at most (the rest are lost at once): ours, far above a window


class FrameReceiver:
    """Frames in, ping records out, with the bookkeeping of what the transport lost.

    `take(frame, metadata)` accepts the bytes of one frame (anything with the buffer
    protocol) and the receiver operator's metadata for it when there is any. A frame that
    arrived short (`frame_bytes_received` below its size: packets missing, their bytes left
    at 0xFF) is not used; the gap it leaves, like the gap of a frame that never arrived,
    makes the assembler discard the ping it cuts through and count it, exactly as a lost
    block does on the UDP path. `finish()` ends the stream (what is still awaited is counted).

    Every frame used is placed by its sample counter (flank_stream.PingAssembler, whose rules
    decide what a frame is); the sender's `frame_number`, when the metadata has it, is used only
    to count frames lost. Counters (`stats()`):

    * `frames_lost` (only with frame numbers; without them it stays 0 and `lost_samples` says what
      was lost): frame numbers skipped and never received by the time their place was given up -
      their ping emitted or dropped, or, between pings, the stream `reorder_blocks` frames past them.
      A frame still awaited is in no counter until then (or `finish()`). Frames before the first
      one seen are not counted.
    * `frames_late`: frames that arrived behind a later one. Those in time are used, their ping
      kept; those whose place was already given up (`frames_too_late`, a part of `frames_late`, each
      frame counted once) are not used, and stay in `frames_lost` and `lost_samples`. With frame numbers a
      frame is too late only if its number was counted lost (so frames_lost >= frames_too_late), however
      late it comes; without them, only within the assembler's reach (later, it counts as a repeat).
    * `frames_repeated`: duplicates - samples already received - ignored; also a frame that did not
      fit the stream and was not confirmed as a restart (it cannot be told from a stale copy), and a
      damaged frame whose number was already received (its metadata is not taken).
    * `pings_discarded_at_stop`: complete pings `finish(keep=False)` returned to a caller that had stopped
      (`run_receiver`). Pings emitted (`pings`) + `dropped_pings` + these = pings heard.
    * `frames`: frames used. Arrivals = frames + frames_repeated + frames_too_late + frames held
      (at most 8, waiting to be confirmed as a restart; none after `finish()`), not counting damaged
      and bad frames.
    * `trouble_frames`: the first 16 frame numbers at which a loss or a damaged frame was noticed:
      for a loss, the number of the frame that arrived after the gap, entered when a frame of the
      gap is given up as lost (a gap filled by late frames is not trouble).
    * `damaged_frames`: the frame numbers of the first 16 frames counted in `frames_damaged` (arrived with
      packets missing, `frame_bytes_received` below the frame's size; not used). A damaged frame without a
      frame number is counted but not listed, so fewer listed than `frames_damaged` means some are not known.

    Frame numbers are taken as the receiver's extended counter (Sensor Bridge extends the 16-bit
    wire counter on the host, HOLOSCAN_NOTES.md section 8; the metadata does not say its width, so
    that is an assumption for LinuxReceiverOp): they are not expected to wrap. A number that goes
    back is judged by the samples: a restart the assembler confirms (another capture start, or the
    samples starting again) restarts the numbering; a frame the assembler takes as continuing the
    stream renumbers it (a wrap or a sender's reset), counting nothing lost by number.
    """

    def __init__(self, sonar: str | SonarModel = "flank_32", sonar_overrides: dict | None = None, pose=None,
                 sound_speed: float | None = None, frame_samples: int | None = None, reorder_blocks: int = REORDER_BLOCKS):
        model = sonar if isinstance(sonar, SonarModel) else SonarModel.load(sonar, sonar_overrides)
        self.assembler = PingAssembler(model, pose, sound_speed, reorder_blocks=reorder_blocks)
        self.frame_samples = frame_samples
        self.frame_bytes = frame_bytes(self.assembler.n_channels, frame_samples) if frame_samples else None
        self.frames = 0                 # frames used
        self.frames_lost = 0            # never received by the time their place was given up (frame numbers only)
        self.frames_damaged = 0         # arrived with packets missing
        self.frames_repeated = 0        # duplicates (or unconfirmed restarts)
        self.frames_late = 0            # arrived out of order
        self.frames_too_late = 0        # ... after their place was given up (also in frames_late and frames_lost)
        self.bad_frames = 0             # not a block of this stream (wrong size, magic, channel count, range)
        self.packets_dropped = 0        # the receiver's cumulative count
        self.first_frame_number = None
        self.trouble_frames = []        # frame numbers at which a loss or a damaged frame was noticed (the first 16)
        self.damaged_frames = []        # frame numbers of damaged frames (the first 16; sonar-h0e)
        self._last_frame_number = None
        self._awaited = []              # [lo, hi, first sample of lo, samples per frame, noticed at]: numbers skipped
        self._anchor = None             # (number, first sample, samples) of the last frame used
        self._gaps_reported = set()
        self._lost_numbers = []         # [lo, hi]: numbers counted in frames_lost, the latest 256 ranges
        self._seen = self._counts()
        self.pings = 0
        self.pings_discarded_at_stop = 0   # complete pings finish() returned to a caller that had stopped

    def _counts(self) -> tuple:
        a = self.assembler
        return a.used_blocks, a.repeated_blocks, a.late_blocks, a.too_late_blocks, a.restarts

    def _account(self) -> bool:
        """Take the assembler's counts since the last call; whether it restarted."""
        now = self._counts()
        used, repeated, late, too_late, restarts = (x - y for x, y in zip(now, self._seen))
        self._seen = now
        self.frames += used
        self.frames_repeated += repeated
        self.frames_late += late
        self.frames_too_late += too_late
        return restarts > 0

    def _lose(self, lo: int, hi: int, noticed: int) -> None:
        self.frames_lost += hi - lo
        self._lost_numbers = (self._lost_numbers + [[lo, hi]])[-256:]
        if len(self.trouble_frames) < 16 and noticed not in self._gaps_reported:
            self._gaps_reported.add(noticed)
            self.trouble_frames.append(noticed)

    def _settle(self, everything: bool = False) -> None:
        """Numbers still awaited whose samples the assembler no longer holds open are lost."""
        a = self.assembler
        keep = []
        for lo, hi, start, n, noticed in self._awaited:
            k = 0
            while lo + k < hi and (everything or not a.holds(start + k * n, start + (k + 1) * n)):
                k += 1
            if k:
                self._lose(lo, lo + k, noticed)
            if lo + k < hi:
                keep.append([lo + k, hi, start + k * n, n, noticed])
        self._awaited = keep

    def _arrived(self, number: int) -> bool:
        """Whether `number` was awaited (and is no longer)."""
        for i, (lo, hi, start, n, noticed) in enumerate(self._awaited):
            if lo <= number < hi:
                parts = [[lo, number, start, n, noticed]] if lo < number else []
                if number + 1 < hi:
                    parts.append([number + 1, hi, start + (number + 1 - lo) * n, n, noticed])
                self._awaited[i:i + 1] = parts
                return True
        return False

    def _number(self, number: int, verdict: str) -> None:
        """Frame-number bookkeeping once the assembler has judged the frame (`verdict`: its last_block, or
        "unused" for a frame not given to it)."""
        if verdict == "held":
            return
        if verdict == "restart":                          # the old run's awaited numbers can no longer come
            self._settle(everything=True)
            self._last_frame_number = number
            return
        last = self._last_frame_number
        if last is None:
            self.first_frame_number = number
        elif number > last + 1:
            lo = last + 1
            if self._anchor is None:                      # nowhere to place them: lost at once
                self._lose(lo, number, number)
                lo = number
            elif number - lo > MAX_AWAITED:               # far more than a window behind: lost at once
                self._lose(lo, number - MAX_AWAITED, number)
                lo = number - MAX_AWAITED
            if lo < number:
                a_number, a_first, a_n = self._anchor
                self._awaited.append([lo, number, a_first + (lo - a_number) * a_n, a_n, number])
        elif number <= last:
            if self.first_frame_number is not None and number < self.first_frame_number and verdict == "late":
                # Placed before the first frame seen (it was late): the numbers between are awaited too.
                a = self.assembler
                if number + 1 < self.first_frame_number:
                    self._awaited.insert(0, [number + 1, self.first_frame_number, a.last_first + a.last_samples,
                                             a.last_samples, self.first_frame_number])
                self.first_frame_number = number
                return
            if self._arrived(number) or verdict not in ("in_order", "gap"):
                return
            self._awaited = []                            # the samples go on, the numbers went back: renumbered
        self._last_frame_number = number

    def take(self, frame, metadata=None) -> list:
        get = metadata.get if metadata is not None else (lambda key, default=None: default)
        number = get("frame_number", None)
        number = int(number) if number is not None else None
        if number is not None:
            for i, (lo, hi) in enumerate(self._lost_numbers):
                if lo <= number < hi:                     # counted lost, here after all: too late, once
                    self._lost_numbers[i:i + 1] = [r for r in ([lo, number], [number + 1, hi]) if r[0] < r[1]]
                    self.frames_late += 1
                    self.frames_too_late += 1
                    return []
        # A number at or below the last one and not awaited: a repeat, unless the samples say the sender restarted.
        # Its metadata is not taken, and a damaged one is a repeat outright (it cannot restart anything).
        old = (number is not None and self._last_frame_number is not None and number <= self._last_frame_number and
               not any(lo <= number < hi for lo, hi, *_ in self._awaited))
        if not old:
            self.packets_dropped = max(self.packets_dropped, int(get("packets_dropped", 0) or 0))
        size = len(frame)
        received = get("frame_bytes_received", None)
        records, verdict = [], "unused"
        if old and received is not None and int(received) < size:
            self.frames_repeated += 1
            return []
        if received is not None and int(received) < size:
            self.frames_damaged += 1
            if number is not None and len(self.trouble_frames) < 16:
                self.trouble_frames.append(number)
            if number is not None and len(self.damaged_frames) < 16:
                self.damaged_frames.append(number)
        elif self.frame_bytes is not None and size != self.frame_bytes:
            self.bad_frames += 1
        else:
            a = self.assembler
            try:
                records = a.feed(frame)
                verdict = a.last_block
            except BlockError:
                self.bad_frames += 1
            self._account()
            if number is not None and verdict == "too_late":
                # A numbered frame not counted lost (a resend of a damaged one, say): a repeat, so that every
                # frame too late is one counted lost.
                self.frames_late -= 1
                self.frames_too_late -= 1
                self.frames_repeated += 1
        if number is not None:
            self._number(number, verdict)
            if verdict in ("in_order", "gap", "late", "restart"):
                a = self.assembler
                self._anchor = (number, a.last_first, a.last_samples)
        elif verdict == "restart":
            self._awaited = []
        self._settle()
        self.pings += len(records)
        return records

    def finish(self, keep: bool = True) -> list:
        """The stream has ended: what is still awaited is lost; the pings that were waiting are returned. With
        `keep` False the caller has stopped and will not use them: they are counted in `pings_discarded_at_stop`."""
        records = self.assembler.finish()
        self._account()
        self._settle(everything=True)
        if keep:
            self.pings += len(records)
        else:
            self.pings_discarded_at_stop += len(records)
        return records

    def stats(self) -> dict:
        a = self.assembler
        return {"frames": self.frames, "frames_lost": self.frames_lost, "frames_damaged": self.frames_damaged,
                "frames_repeated": self.frames_repeated, "frames_late": self.frames_late, "frames_too_late": self.frames_too_late,
                "bad_frames": self.bad_frames, "packets_dropped": self.packets_dropped, "first_frame_number": self.first_frame_number,
                "trouble_frames": list(self.trouble_frames), "damaged_frames": list(self.damaged_frames), "pings": self.pings,
                "pings_discarded_at_stop": self.pings_discarded_at_stop, "dropped_pings": a.dropped_pings,
                "lost_samples": a.lost_samples}


def records_from_frames(frames: Iterable, sonar: str | SonarModel = "flank_32", sonar_overrides: dict | None = None,
                        **options) -> list:
    """The ping records a run of frames holds (the host side without any transport)."""
    receiver = FrameReceiver(sonar, sonar_overrides, **options)
    return [record for frame in frames for record in receiver.take(frame)] + receiver.finish()


def raw_sockets_permitted() -> bool:
    """Whether this process may open the raw IP socket Sensor Bridge's emulator sends its data
    through (CAP_NET_RAW: root on an ordinary machine; not granted in every container)."""
    import os
    import socket
    if os.environ.get("HSB_EMULATOR_UDP_SOCKET") == "1":    # an emulator built with HSB_UDP_PATCH=1 needs none
        return True
    try:
        socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW).close()
    except OSError:
        return False
    return True


CATCH_UP = ("burst", "spread")


QUEUE_SECONDS = 4.0             # of stream the receiver operator holds while the chain works on a ping (D-187; owner's)


def queue_frames_for(sample_rate: float, frame_samples: int, seconds: float = QUEUE_SECONDS) -> int:
    """Frames of `frame_samples` samples that hold `seconds` of a stream at `sample_rate`: the receiver operator's
    queue (it drops the oldest frames beyond it). 640 for flank_32's 160 frames a second, 1024 for a Popoto build's
    256 (400-sample frames at 102.4 kHz); before D-187 a fixed 256 frames, 1.6 s and 1.0 s."""
    if not (sample_rate > 0 and frame_samples > 0 and seconds > 0):
        raise ValueError("queue_frames_for: sample_rate, frame_samples and seconds must be positive")
    return int(math.ceil(seconds * sample_rate / frame_samples))


LOG_FRAMES = 100_000            # per-frame logs keep the newest this many frames (10.4 min at 160 frames/s)


class SendTiming:
    """When each frame left the sender against when it was due, to tell a sender that fell
    behind (and then sent the frames it owed back to back) from losses in the receiver.
    Times are seconds of `time.perf_counter()`; `wall_ns` maps them to the system clock that
    the receiver's metadata (`received_s`, `received_ns`) uses. Python's garbage collector
    pauses are timed alongside. The summary covers every frame; the per-frame rows for
    `dump` keep the newest LOG_FRAMES. Frame 0 is not recorded: it is the send retried until
    a receiver has connected, and the moment it goes through defines t0 (it is on time by
    definition)."""

    def __init__(self):
        from collections import deque
        self.t0 = self.period = None
        self.last_start = self.last_end = None
        self.records = deque(maxlen=LOG_FRAMES)  # (frame, start - due, start - previous start, send duration)
        self.wall_offset_ns = None
        self.n = 0
        self.max_late = self.max_gap = self.max_send = 0.0
        self.behind, self.close = [], []          # first 16 frame numbers of each
        self.n_behind = self.n_close = 0
        self.gc_pauses, self.max_gc = 0, 0.0
        self._gc_t = None

    def start(self, t0: float, period: float) -> None:
        import gc
        self.t0, self.period = t0, period
        self.last_start = self.last_end = t0
        self.wall_offset_ns = time.time_ns() - int(time.perf_counter() * 1e9)
        gc.callbacks.append(self._on_gc)

    def stop(self) -> None:
        """Take the garbage-collector callback out again (idempotent; called from a finally)."""
        import gc
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)

    def _on_gc(self, phase, info):
        if phase == "start":
            self._gc_t = time.perf_counter()
        elif self._gc_t is not None:
            self.gc_pauses += 1
            self.max_gc = max(self.max_gc, time.perf_counter() - self._gc_t)

    def sent(self, k: int, t_start: float, t_end: float) -> None:
        late, gap, took = t_start - (self.t0 + k * self.period), t_start - self.last_start, t_end - t_start
        self.records.append((k, late, gap, took))
        self.n += 1
        self.max_late, self.max_gap, self.max_send = max(self.max_late, late), max(self.max_gap, gap), max(self.max_send, took)
        if late > self.period:
            self.n_behind += 1
            if len(self.behind) < 16:
                self.behind.append(k)
        if gap < 0.25 * self.period:
            self.n_close += 1
            if len(self.close) < 16:
                self.close.append(k)
        self.last_start, self.last_end = t_start, t_end

    def wall_ns(self, t: float) -> int:
        return int(t * 1e9) + self.wall_offset_ns

    def dump(self, path: str) -> None:
        """The kept frames' send times as JSON: frame, system-clock ns at the send's start, ms late, ms since the previous send, ms the send call took."""
        import json
        rows = [[k, self.wall_ns(self.t0 + k * self.period + late), round(1e3 * late, 3), round(1e3 * gap, 3), round(1e3 * took, 3)]
                for k, late, gap, took in self.records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"period_ms": 1e3 * self.period, "columns": ["frame", "wall_ns", "late_ms", "gap_ms", "send_ms"], "rows": rows}, f)

    def summary(self) -> dict:
        self.stop()
        if not self.n:
            return {}
        return {"period_ms": 1e3 * self.period, "max_late_ms": 1e3 * self.max_late,
                "frames_late_by_a_period": self.n_behind, "first_late_frames": list(self.behind),
                "back_to_back_frames": self.n_close, "first_back_to_back_frames": list(self.close),
                "max_gap_ms": 1e3 * self.max_gap, "max_send_ms": 1e3 * self.max_send,
                "gc_pauses": self.gc_pauses, "max_gc_ms": 1e3 * self.max_gc}


class HsbBoardEmulator:
    """Sends ping records as the board would, through Sensor Bridge's emulator
    (`hololink.emulation`), from the interface that owns `ip`.

    `run()` blocks: it announces the board (the emulator's own enumeration broadcasts), waits
    for a receiver to configure the data channel, then sends every frame paced to `pace` times
    real time, and keeps the stream going with idle frames until the receiver disconnects or
    `linger` seconds pass (a receiver still has to talk to the board to shut down cleanly).
    With `hold`, a board whose receiver is still connected after the linger stays up, quiet, that
    many seconds more, as a real board that stops streaming stays on the network: a receiver that
    stops for want of frames (its idle timeout) can still switch the data channel off.
    The emulator's data socket is a raw IP socket, so the process needs CAP_NET_RAW (root).
    """

    def __init__(self, records, frame_samples: int = FRAME_SAMPLES, ip: str = "127.0.0.1", pace: float = 1.0,
                 lead_seconds: float = 1.0, ping_interval: float | None = None, linger: float = 10.0,
                 catch_up: str = "spread", hold: float = 0.0, **stream_options):
        if pace <= 0.0:
            raise ValueError("pace is a multiple of real time and must be positive")
        if catch_up not in CATCH_UP:
            raise ValueError(f"catch_up is one of {CATCH_UP}")
        self.stream = FrameStream(records, frame_samples, ping_interval=ping_interval, lead_seconds=lead_seconds, **stream_options)
        self.ip, self.pace, self.linger, self.catch_up, self.hold = ip, float(pace), float(linger), catch_up, float(hold)
        self.sent = self.idle_sent = 0
        self.timing = SendTiming()

    def frames(self) -> Iterator[bytes]:
        return iter(self.stream)

    def _wait(self, t0: float, k: int, period: float) -> None:
        """Sleep until frame k is due (t0 + k periods). With catch_up "spread" (the default) a
        frame never leaves sooner than half a period after the one before, so the sender makes
        up a stall of its own at twice the frame rate, as a board's stream never comes faster;
        with "burst" (the behaviour before) a late frame goes at once, so after a stall the
        frames owed leave back to back and can overflow the receiver's socket buffer."""
        due = t0 + k * period
        if self.catch_up == "spread" and self.timing.last_end is not None:
            due = max(due, self.timing.last_start + 0.5 * period)
        wait = due - time.perf_counter()
        if wait > 0.0:
            time.sleep(wait)

    def run(self, connect_timeout: float = 120.0, log=None) -> dict:
        import hololink.emulation as hemu
        log = log or (lambda text: None)
        if not raw_sockets_permitted():
            raise PermissionError("Sensor Bridge's emulator sends through a raw IP socket, which this process may not open "
                                  "(it needs CAP_NET_RAW: run as root, give the container that capability, or build with "
                                  "HSB_UDP_PATCH=1 and set HSB_EMULATOR_UDP_SOCKET=1)")
        hsb = hemu.HSBEmulator()
        data_plane = hemu.LinuxDataPlane(hsb, hemu.IPAddress(self.ip), 0, 0)   # data plane 0, sensor 0
        hsb.start()
        period = self.stream.frame_seconds / self.pace
        try:
            log(f"board emulator at {self.ip}: {self.stream.n_frames} frames of {self.stream.frame_bytes} bytes, "
                f"{1.0 / period:.1f} frames/s; waiting for a receiver")
            t_wait = time.perf_counter()
            t0 = None
            for k, frame in enumerate(self.stream):
                array = np.frombuffer(frame, dtype=np.uint8).copy()       # writable: the emulator takes it by DLPack
                if t0 is None:
                    while True:                                           # send() returns 0 until a receiver has connected
                        n = data_plane.send(array)
                        if n:
                            break
                        if time.perf_counter() - t_wait > connect_timeout:
                            raise TimeoutError(f"no receiver configured the data channel within {connect_timeout:.0f} s")
                        time.sleep(0.02)
                    t0 = time.perf_counter()
                    self.timing.start(t0, period)
                    log("receiver connected")
                else:
                    self._wait(t0, k, period)
                    t_send = time.perf_counter()
                    n = data_plane.send(array)
                    self.timing.sent(k, t_send, time.perf_counter())
                if n < 0:
                    raise RuntimeError(f"the emulator failed to send frame {k}")
                if n == 0:
                    log(f"the receiver disconnected at frame {k}")
                    break
                self.sent += 1
            else:
                k = self.stream.n_frames
                connected = True
                for frame in self.stream.idle():
                    if (k - self.stream.n_frames) * period > self.linger:
                        break
                    self._wait(t0, k, period)
                    t_send = time.perf_counter()
                    if data_plane.send(np.frombuffer(frame, dtype=np.uint8).copy()) <= 0:
                        connected = False
                        break
                    self.timing.sent(k, t_send, time.perf_counter())
                    self.idle_sent += 1
                    k += 1
                if connected and self.hold > 0:
                    log(f"lingered {self.linger:.0f} s; the board stays up, quiet, {self.hold:.0f} s more")
                    time.sleep(self.hold)
        finally:
            self.timing.stop()
            hsb.stop()
        out = {"frames_sent": self.sent, "idle_frames_sent": self.idle_sent, "frame_bytes": self.stream.frame_bytes,
               "frames_per_second": 1.0 / period, "catch_up": self.catch_up, "timing": self.timing.summary()}
        log(f"sent {self.sent} frames and {self.idle_sent} idle frames")
        return out


# -- the host side: CUDA driver calls, operators, application -------------------------------------


class CudaDriver:
    """The few CUDA driver calls the host side needs, through libcuda directly: the device's
    primary context for the receiver operator, and a copy of a frame from GPU memory."""

    def __init__(self, device: int = 0):
        lib = ctypes.CDLL("libcuda.so.1")
        self._lib = lib
        self._check(lib.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self._check(lib.cuDeviceGet(ctypes.byref(dev), int(device)), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        self._check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self._device = dev
        self.context = int(ctx.value)                       # what LinuxReceiverOp takes as frame_context
        lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _check(code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what} failed with CUDA error {code}")

    @staticmethod
    def available() -> bool:
        try:
            lib = ctypes.CDLL("libcuda.so.1")
            count = ctypes.c_int()
            return lib.cuInit(0) == 0 and lib.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
        except OSError:
            return False

    def to_host(self, pointer: int, size: int) -> np.ndarray:
        """Wait for the work queued in the context (the receiver's copy of the frame onto the
        GPU is asynchronous) and copy `size` bytes at a device address into a new array."""
        lib = self._lib
        out = np.empty(int(size), dtype=np.uint8)
        self._check(lib.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")
        self._check(lib.cuCtxSynchronize(), "cuCtxSynchronize")
        self._check(lib.cuMemcpyDtoH_v2(out.ctypes.data, int(pointer), out.size), "cuMemcpyDtoH")
        return out

    def release(self) -> None:
        self._lib.cuDevicePrimaryCtxRelease(self._device)


def frame_to_host(message, cuda: CudaDriver | None = None) -> np.ndarray:
    """The bytes of the frame in a receiver message: a tensor map with one tensor (named ""),
    or the tensor itself. GPU tensors are copied through `cuda`; host ones are used as they are."""
    tensor = message
    if hasattr(message, "keys") and not hasattr(message, "shape"):
        tensor = message[""] if "" in message else next(iter(message.values()))
    gpu = getattr(tensor, "__cuda_array_interface__", None)
    if gpu is not None:
        if cuda is None:
            raise RuntimeError("the frame is in GPU memory and no CUDA driver was given to copy it")
        return cuda.to_host(gpu["data"][0], int(np.prod(gpu["shape"])) * np.dtype(gpu["typestr"]).itemsize)
    return np.ascontiguousarray(np.asarray(tensor)).view(np.uint8).reshape(-1)


class Board:
    """The `device` the receiver operator starts and stops. The capture board streams as soon
    as the data channel is configured, so there is nothing to switch on yet; a real board's
    gain and trigger registers would be written here."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


METADATA_KEYS = ("frame_number", "frame_bytes_received", "packets_dropped", "bytes_written", "received_frame_number",
                 "timestamp_s", "timestamp_ns", "received_s", "received_ns", "frame_packets_received")


def _proc_udp_sockets() -> dict:
    """inode -> (rx_queue bytes, drops) for every UDP socket of this network namespace (/proc/net/udp)."""
    out = {}
    try:
        with open("/proc/net/udp", encoding="ascii") as f:
            next(f)
            for line in f:
                p = line.split()
                out[int(p[9])] = (int(p[4].split(":")[1], 16), int(p[-1]))
    except (OSError, ValueError, IndexError, StopIteration):
        pass
    return out


def set_chain_omp_threads(n: int) -> int | None:
    """omp_set_num_threads(n) in the OpenMP runtime this process has loaded (libgomp or libomp): the thread
    count of the parallel regions that the *calling* thread starts from now on (OpenMP's nthreads-var is per
    thread; other threads keep theirs). `LiveChainOp` calls it from the thread that runs the chain, before
    the chain's first ping, when `receive --chain-omp-threads` asks for it (sonar-ihd: with one OpenMP
    thread per vCPU, flank.backprojection_map kept Sensor Bridge's receiver thread waiting 10-13 ms on an
    8-vCPU L4 host and the socket overflowed). Returns n, or None when no OpenMP runtime is loaded."""
    if int(n) < 1:
        raise ValueError(f"OpenMP threads: {n} (at least 1)")
    with open("/proc/self/maps") as f:
        libs = {line.split()[-1] for line in f
                if ("libgomp" in line or "libomp" in line or "libiomp" in line) and line.split()[-1].startswith("/")}
    for path in sorted(libs):
        try:
            ctypes.CDLL(path).omp_set_num_threads(ctypes.c_int(int(n)))
            return int(n)
        except (OSError, AttributeError):
            continue
    return None


def udp_counters() -> dict:
    """The namespace's UDP counters (/proc/net/snmp): RcvbufErrors counts datagrams a full receive
    buffer refused, SndbufErrors sends refused for want of buffer."""
    try:
        with open("/proc/net/snmp", encoding="ascii") as f:
            rows = [line.split() for line in f if line.startswith("Udp:")]
        return {k: int(v) for k, v in zip(rows[0][1:], rows[1][1:])}
    except (OSError, ValueError, IndexError):
        return {}


def cpu_throttling() -> dict:
    """This container's CPU quota and how often it was throttled (cgroup v2 cpu.max, cpu.stat), when readable."""
    out = {}
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as f:
            out["cpu_max"] = f.read().strip()
        with open("/sys/fs/cgroup/cpu.stat", encoding="ascii") as f:
            for line in f:
                k, v = line.split()
                if k in ("nr_periods", "nr_throttled", "throttled_usec"):
                    out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


class DataSocketWatch:
    """The receiver operator's data socket, found from outside Sensor Bridge: the UDP socket of
    this process whose receive timeout is LinuxReceiver's 100 ms. Reports its receive buffer and
    the kernel's count of datagrams it dropped because that buffer was full; can set the buffer
    (up to net.core.rmem_max) to measure what it does."""

    def __init__(self, receive_buffer: int | None = None):
        self.receive_buffer = receive_buffer
        self.fd = self.inode = None
        self.rcvbuf = None
        self.drops = 0
        self.max_rx_queue = 0

    def locate(self) -> bool:
        # A heuristic, not an interface: the first datagram socket of this process whose receive
        # timeout is exactly 100 ms (what LinuxReceiver sets on its data socket, Sensor Bridge 2.7.0
        # linux_receiver.cpp). Another such socket in the process (another receiver, or a library
        # that happens to use 100 ms) could be taken instead; then the drop count is that socket's,
        # and with a receive_buffer (receive --socket-buffer) that socket's buffer is the one changed.
        # `stats()["found"]` and the drop count agreeing with the packet-number gaps are the check.
        import os
        import socket
        import struct
        if self.inode is not None:
            return True
        for name in os.listdir("/proc/self/fd"):
            fd = int(name)
            try:
                if not os.readlink(f"/proc/self/fd/{fd}").startswith("socket:"):
                    continue
                sock = socket.socket(fileno=os.dup(fd))
            except OSError:
                continue
            try:
                if sock.type != socket.SOCK_DGRAM:
                    continue
                sec, usec = struct.unpack("ll", sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, 16))
                if (sec, usec) != (0, 100000):
                    continue
                if self.receive_buffer:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(self.receive_buffer))
                self.rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                self.fd, self.inode = fd, os.fstat(sock.fileno()).st_ino
                return True
            finally:
                sock.close()
        return False

    def sample(self) -> None:
        if self.inode is None and not self.locate():
            return
        entry = _proc_udp_sockets().get(self.inode)
        if entry is not None:
            self.max_rx_queue = max(self.max_rx_queue, entry[0])
            self.drops = max(self.drops, entry[1])

    def stats(self) -> dict:
        return {"found": self.inode is not None, "rcvbuf": self.rcvbuf, "drops": self.drops, "max_rx_queue": self.max_rx_queue}

class ReceiveStop:
    """When `HsbLiveApp` stops (issue sonar-dkb). `check` is called about four times a second on the
    scheduler's thread with what has happened so far and returns why to stop, or None:

    * "pings": `n_pings` pings are accounted for, reported by the chain or counted as dropped by the
      receiver (`FrameReceiver.dropped_pings`, pings lost whole). A ping that is still waiting when the
      receive stops is counted by `FrameReceiver.finish` in `pings_discarded_at_stop`.
    * "idle": no frame (good, damaged or not of this stream) for `idle_timeout` seconds, whatever was
      lost; the timer starts with the first check, so a stream that never starts stops too.

    Without `n_pings` only the idle rule stops; without `idle_timeout` only the count does."""

    def __init__(self, n_pings: int | None = None, idle_timeout: float | None = 15.0, clock=time.monotonic):
        self.n_pings, self.idle_timeout, self._clock = n_pings, idle_timeout, clock
        self.reason = None
        self._seen = self._t_seen = None

    def check(self, reported: int, dropped: int, frames_heard: int) -> str | None:
        if self.reason is not None:
            return self.reason
        now = self._clock()
        if self._seen is None or frames_heard != self._seen:
            self._seen, self._t_seen = frames_heard, now
        if self.n_pings is not None and reported + dropped >= self.n_pings:
            self.reason = "pings"
        elif self.idle_timeout is not None and now - self._t_seen > self.idle_timeout:
            self.reason = "idle"
        return self.reason


def stopped_short(stats: dict, n_pings: int | None) -> str | None:
    """What a receive that ended before reporting `n_pings` pings tells the caller, or None when it did not."""
    reported = stats.get("reports", 0)
    if n_pings is None or reported >= n_pings:
        return None
    return (f"stopped short ({stats.get('stopped')}): {reported} of {n_pings} pings reported; "
            f"dropped {stats.get('dropped_pings')}, discarded at stop {stats.get('pings_discarded_at_stop')}, "
            f"frames lost {stats.get('frames_lost')}, damaged {stats.get('frames_damaged')}, "
            f"samples lost {stats.get('lost_samples')}, frames heard {stats.get('frames')}")


try:
    from holoscan.core import Application, Operator
except Exception:  # noqa: BLE001  (the SDK needs a CUDA runtime even to import)
    Application = Operator = None

if Operator is not None:

    class HsbFramesToPingsOp(Operator):
        """Receiver frames in ("input"), ping records out ("ping", `{"record": record}` as PingSourceOp)."""

        def __init__(self, fragment, *args, receiver: FrameReceiver, cuda: CudaDriver | None = None,
                     socket_watch: DataSocketWatch | None = None, arrivals=None, **kwargs):
            self.receiver = receiver
            self._cuda = cuda
            self._ready = []
            self.last_metadata = {}
            self.socket_watch = socket_watch
            self.arrivals = arrivals            # (frame_number, received_ns, packets, bytes) per frame, when a list or deque is given
            self._count = 0
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec):
            spec.input("input")
            spec.output("ping")

        def compute(self, op_input, op_output, context):
            message = op_input.receive("input")
            if message is None:
                return
            frame = frame_to_host(message, self._cuda)       # first: the next frame reuses the GPU buffer
            metadata = {}
            try:
                for key in METADATA_KEYS:
                    if key in self.metadata:
                        metadata[key] = self.metadata[key]
            except Exception:  # noqa: BLE001  (metadata switched off in the application)
                metadata = {}
            self.last_metadata = metadata
            if self.socket_watch is not None and self._count % 16 == 0:
                self.socket_watch.sample()
            self._count += 1
            if self.arrivals is not None and "frame_number" in metadata:
                self.arrivals.append((int(metadata["frame_number"]),
                                      int(metadata.get("received_s", 0)) * 1_000_000_000 + int(metadata.get("received_ns", 0)),
                                      int(metadata.get("frame_packets_received", -1)), int(metadata.get("frame_bytes_received", -1))))
            self._ready += self.receiver.take(frame, metadata or None)
            if self._ready:                                   # at most one ping completes per frame
                op_output.emit({"record": self._ready.pop(0), "t_source": time.perf_counter()}, "ping")

    class LiveChainOp(Operator):
        """Runs `gpusonar.live.LiveChain` on each record: recording, the chain (Python or C++
        engine), report file and report datagrams are the chain's. Keeps the reports, calls
        `on_report(report)` and `on_done()` after `n_pings`."""

        def __init__(self, fragment, *args, chain=None, n_pings: int | None = None, on_report=None, on_done=None,
                     omp_threads: int | None = None, **kwargs):
            self.chain = chain
            self._omp_threads = omp_threads    # opt-in: set_chain_omp_threads on this operator's thread before the first ping
            self.omp_applied = None
            self.reports = []
            from collections import deque
            self.wall_ns = deque(maxlen=4096)  # (system-clock ns at the start, at the end) of the newest 4096 chain runs
            self._n = n_pings
            self._on_report, self._on_done = on_report, on_done
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec):
            spec.input("ping")

        def compute(self, op_input, op_output, context):
            message = op_input.receive("ping")
            if self._n is not None and len(self.reports) >= self._n:
                return
            if self._omp_threads is not None and self.omp_applied is None:
                self.omp_applied = set_chain_omp_threads(self._omp_threads) or "no OpenMP runtime"
            t0 = time.time_ns()
            report = self.chain.process(message["record"])
            self.wall_ns.append((t0, time.time_ns()))
            self.reports.append(report)
            if self._on_report:
                self._on_report(report)
            if self._n is not None and len(self.reports) >= self._n and self._on_done:
                self._on_done()

    class HsbStopOp(Operator):
        """Applies `ReceiveStop` on the scheduler's thread, ticked by a periodic condition (sonar-dkb), and
        switches the receiver's boolean condition and its own off when it says stop."""

        def __init__(self, fragment, *args, rule: ReceiveStop, receiver: FrameReceiver, chain_op, conditions=(),
                     on_stop=None, **kwargs):
            self.rule, self.receiver, self.chain_op = rule, receiver, chain_op
            self._conditions, self._on_stop = tuple(conditions), on_stop
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec):
            pass

        def compute(self, op_input, op_output, context):
            r = self.receiver
            reason = self.rule.check(len(self.chain_op.reports), r.assembler.dropped_pings,
                                     r.frames + r.frames_damaged + r.bad_frames)
            if reason is not None:
                if self._on_stop:
                    self._on_stop()
                for c in self._conditions:
                    c.disable_tick()

    class HsbLiveApp(Application):
        """LinuxReceiverOp -> HsbFramesToPingsOp -> LiveChainOp.

        The receiver keeps up to `queue_frames` frames (default: QUEUE_SECONDS of stream) while the chain works on a ping (it
        drops the oldest beyond that, which `FrameReceiver` counts as lost frames). Ends
        after `n_pings` pings are accounted for (reported, or dropped by the receiver), or when
        no frame has arrived for `idle_timeout` seconds (`ReceiveStop`; `stop_reason` says which).
        """

        def __init__(self, chain, receiver: FrameReceiver, hololink_channel, cuda: CudaDriver, frame_size: int,
                     n_pings: int | None = None, queue_frames: int | None = None, idle_timeout: float | None = 15.0,
                     on_report=None, receiver_affinity=None, socket_buffer: int | None = None, log_arrivals: bool = False,
                     chain_omp_threads: int | None = None):
            self.chain, self.receiver = chain, receiver
            self._omp_threads = chain_omp_threads
            self._channel, self._cuda, self._frame_size = hololink_channel, cuda, int(frame_size)
            if queue_frames is None:        # QUEUE_SECONDS of this stream
                queue_frames = queue_frames_for(receiver.assembler.sample_rate, receiver.frame_samples)
            if int(queue_frames) < 1:
                raise ValueError("queue_frames must be at least 1")
            self._n, self._queue, self._idle = n_pings, int(queue_frames), idle_timeout
            self.queue_frames = self._queue
            self._on_report, self._affinity = on_report, receiver_affinity
            self.socket_watch = DataSocketWatch(socket_buffer)
            from collections import deque
            self.arrivals = deque(maxlen=LOG_FRAMES) if log_arrivals else None   # the newest LOG_FRAMES frames
            self.frames_op = self.chain_op = None
            self._running = None
            self.rule = ReceiveStop(n_pings, idle_timeout or None)
            super().__init__()

        @property
        def stop_reason(self) -> str | None:
            return self.rule.reason

        def _finish(self):
            """The chain has reported `n_pings` pings (on the scheduler's thread)."""
            if self.rule.reason is None:
                self.rule.reason = "pings"
            self.socket_watch.sample()
            self._running.disable_tick()
            self._stop_running.disable_tick()

        def compose(self):
            import datetime

            import hololink
            from holoscan.conditions import BooleanCondition, PeriodicCondition
            self._running = BooleanCondition(self, name="running", enable_tick=True)
            self._stop_running = BooleanCondition(self, name="stop_running", enable_tick=True)   # one per operator
            # Sensor Bridge pins its receiver thread to core 2 (HOLOLINK_AFFINITY names another); its Python
            # binding takes one core number here, not a set (a set fails to convert in 2.7.0).
            options = {} if self._affinity is None else {"receiver_affinity": int(self._affinity)}
            # use_frame_ready_condition=False (sonar-dkb): with Sensor Bridge's default the receiver operator
            # waits on an asynchronous condition that only a new frame sets, and the greedy scheduler looks at
            # it again only then; once frames stop, switching its boolean condition off is never seen and the
            # application never ends. Without it, the receiver's compute waits for a frame for at most 1 s and
            # returns, so the scheduler sees the switch within a second.
            rx = hololink.operators.LinuxReceiverOp(
                self, self._running, name="receiver", frame_size=self._frame_size, frame_context=self._cuda.context,
                hololink_channel=self._channel, device=Board(), pages=self._queue, queue_size=self._queue,
                use_frame_ready_condition=False, **options)
            self.frames_op = HsbFramesToPingsOp(self, receiver=self.receiver, cuda=self._cuda, socket_watch=self.socket_watch,
                                                arrivals=self.arrivals, name="frames_to_pings")
            self.chain_op = LiveChainOp(self, chain=self.chain, n_pings=self._n, on_report=self._on_report,
                                        on_done=self._finish, omp_threads=self._omp_threads, name="chain")
            self.add_flow(rx, self.frames_op, {("output", "input")})
            self.add_flow(self.frames_op, self.chain_op, {("ping", "ping")})
            # The stop rule, ticked four times a second on the scheduler's thread; it switches off the
            # receiver's boolean condition and its own (a condition object serves one operator).
            self.add_operator(HsbStopOp(self, self._stop_running,
                                        PeriodicCondition(self, recess_period=datetime.timedelta(milliseconds=250), name="stop_tick"),
                                        rule=self.rule, receiver=self.receiver, chain_op=self.chain_op,
                                        conditions=(self._running, self._stop_running), on_stop=self.socket_watch.sample,
                                        name="stop"))


_KEEP = []       # applications whose stop failed (run_receiver), kept to the end of the process


def run_receiver(chain, sonar: str = "flank_32", sonar_overrides: dict | None = None, hololink_ip: str = "127.0.0.1",
                 frame_samples: int = FRAME_SAMPLES, n_pings: int | None = None, sound_speed: float | None = None,
                 find_timeout: float = 30.0, on_report=None, **app_options) -> tuple[list, dict]:
    """Find the board at `hololink_ip`, receive its frames with Sensor Bridge's Linux receiver
    and run `chain` (a `gpusonar.live.LiveChain`) on the pings. Returns (reports, statistics).
    Needs a CUDA device (the receiver puts each frame in GPU memory)."""
    if Application is None:
        raise RuntimeError("the Holoscan SDK is not usable here")
    import hololink
    receiver = FrameReceiver(sonar, sonar_overrides, sound_speed=sound_speed, frame_samples=frame_samples)
    cuda = CudaDriver()
    udp0, cpu0 = udp_counters(), cpu_throttling()
    shutdown_error = None
    try:
        metadata = hololink.Enumerator.find_channel(channel_ip=hololink_ip, timeout=hololink.Timeout(float(find_timeout)))
        channel = hololink.DataChannel(metadata)
        app = HsbLiveApp(chain, receiver, channel, cuda, receiver.frame_bytes, n_pings=n_pings, on_report=on_report, **app_options)
        board = channel.hololink()
        board.start()
        try:
            board.reset()
            app.run()
        except Exception as e:  # noqa: BLE001
            # After the stop rule fired, an error from the operators' stop is the board no longer answering
            # (Sensor Bridge's receiver switches the data channel off over the control plane, 2.7.0
            # data_channel.cpp unconfigure): the run is over, so it is reported, not raised. Its receiver
            # thread may then still be running; the caller should end the process without unwinding it
            # (`receive` does, os._exit after its statistics).
            if app.stop_reason is None:
                raise
            shutdown_error = f"{type(e).__name__}: {e}"
        finally:
            board.stop()
    finally:
        cuda.release()
    receiver.finish(keep=False)  # the end of the stream: what is still awaited is counted, waiting pings in pings_discarded_at_stop
    stats = receiver.stats()
    stats["reports"] = len(app.chain_op.reports)
    stats["queue_frames"] = getattr(app, "queue_frames", None)   # frames the receiver operator held (default: QUEUE_SECONDS)
    stats["stopped"] = app.stop_reason or "ended"        # pings, idle, or the application ended by itself
    if shutdown_error:
        stats["shutdown_error"] = shutdown_error
        # Not destroyed before the process ends: after a failed stop its receiver thread may still be running, and
        # destroying the application would join it. Only those are kept (sonar-qnk): one that stopped cleanly goes.
        _KEEP.append(app)
    udp1, cpu1 = udp_counters(), cpu_throttling()
    stats["socket"] = app.socket_watch.stats()
    stats["udp_counters"] = {k: udp1[k] - udp0.get(k, 0) for k in ("InDatagrams", "InErrors", "RcvbufErrors", "SndbufErrors")
                             if k in udp1}
    stats["cpu"] = dict(cpu1, **{k + "_during": cpu1[k] - cpu0[k] for k in ("nr_throttled", "throttled_usec")
                                 if k in cpu1 and k in cpu0})
    if app.chain_op.omp_applied is not None:
        stats["chain_omp_threads"] = app.chain_op.omp_applied
    stats["chain_wall_ns"] = list(app.chain_op.wall_ns)
    if app.arrivals is not None:
        stats["arrivals"] = list(app.arrivals)
    return app.chain_op.reports, stats
