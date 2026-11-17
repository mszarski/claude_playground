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
import time
from typing import Iterable, Iterator

import numpy as np

from ..sonar import SonarModel
from .flank_stream import FULL_SCALE_COUNTS, HEADER_BYTES, BlockError, PingAssembler, encode_block

FRAME_SAMPLES = 400            # 6.25 ms of the 64 kSPS stream: 38,528 bytes with 32 channels, 160 frames a second.
# A frame leaves the sender as one burst of datagrams into the receiver's socket buffer, which
# Sensor Bridge's receiver leaves at the system default (208 KB of kernel accounting, about 48
# datagrams) unless a frame is larger, and which a container may not be able to raise. On loopback
# 1600-sample frames (110 datagrams) lost 5 frames in 502 and 800-sample frames (55) 1 in 1003.
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


class FrameReceiver:
    """Frames in, ping records out, with the bookkeeping of what the transport lost.

    `take(frame, metadata)` accepts the bytes of one frame (anything with the buffer
    protocol) and the receiver operator's metadata for it when there is any. A frame that
    arrived short (`frame_bytes_received` below its size: packets missing, their bytes left
    at 0xFF) is not used; the gap it leaves, like the gap of a frame that never arrived,
    makes the assembler discard the ping it cuts through and count it, exactly as a lost
    block does on the UDP path. A frame seen twice (same `frame_number`) is ignored.
    """

    def __init__(self, sonar: str | SonarModel = "flank_32", sonar_overrides: dict | None = None, pose=None,
                 sound_speed: float | None = None, frame_samples: int | None = None):
        model = sonar if isinstance(sonar, SonarModel) else SonarModel.load(sonar, sonar_overrides)
        self.assembler = PingAssembler(model, pose, sound_speed)
        self.frame_samples = frame_samples
        self.frame_bytes = frame_bytes(self.assembler.n_channels, frame_samples) if frame_samples else None
        self.frames = 0                 # frames used
        self.frames_lost = 0            # missing between the first frame seen and the last (sender's frame counter)
        self.frames_damaged = 0         # arrived with packets missing
        self.frames_repeated = 0
        self.bad_frames = 0             # not a block of this stream (wrong size, magic, channel count)
        self.packets_dropped = 0        # the receiver's cumulative count
        self.first_frame_number = None
        self.trouble_frames = []        # frame numbers at which a loss or a damaged frame was noticed (the first 16)
        self._last_frame_number = None
        self.pings = 0

    def take(self, frame, metadata=None) -> list:
        get = metadata.get if metadata is not None else (lambda key, default=None: default)
        number = get("frame_number", None)
        if number is not None:
            number = int(number)
            if self._last_frame_number is not None and number <= self._last_frame_number:
                self.frames_repeated += 1
                return []
            if self._last_frame_number is None:
                self.first_frame_number = number
            else:
                self.frames_lost += number - self._last_frame_number - 1
                if number - self._last_frame_number > 1 and len(self.trouble_frames) < 16:
                    self.trouble_frames.append(number)
            self._last_frame_number = number
        self.packets_dropped = max(self.packets_dropped, int(get("packets_dropped", 0) or 0))
        size = len(frame)
        received = get("frame_bytes_received", None)
        if received is not None and int(received) < size:
            self.frames_damaged += 1
            if number is not None and len(self.trouble_frames) < 16:
                self.trouble_frames.append(number)
            return []
        if self.frame_bytes is not None and size != self.frame_bytes:
            self.bad_frames += 1
            return []
        try:
            records = self.assembler.feed(frame)
        except BlockError:
            self.bad_frames += 1
            return []
        self.frames += 1
        self.pings += len(records)
        return records

    def stats(self) -> dict:
        a = self.assembler
        return {"frames": self.frames, "frames_lost": self.frames_lost, "frames_damaged": self.frames_damaged,
                "frames_repeated": self.frames_repeated, "bad_frames": self.bad_frames,
                "packets_dropped": self.packets_dropped, "first_frame_number": self.first_frame_number, "trouble_frames": list(self.trouble_frames),
                "pings": self.pings, "dropped_pings": a.dropped_pings, "lost_samples": a.lost_samples}


def records_from_frames(frames: Iterable, sonar: str | SonarModel = "flank_32", sonar_overrides: dict | None = None,
                        **options) -> list:
    """The ping records a run of frames holds (the host side without any transport)."""
    receiver = FrameReceiver(sonar, sonar_overrides, **options)
    return [record for frame in frames for record in receiver.take(frame)]


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


class HsbBoardEmulator:
    """Sends ping records as the board would, through Sensor Bridge's emulator
    (`hololink.emulation`), from the interface that owns `ip`.

    `run()` blocks: it announces the board (the emulator's own enumeration broadcasts), waits
    for a receiver to configure the data channel, then sends every frame paced to `pace` times
    real time, and keeps the stream going with idle frames until the receiver disconnects or
    `linger` seconds pass (a receiver still has to talk to the board to shut down cleanly).
    The emulator's data socket is a raw IP socket, so the process needs CAP_NET_RAW (root).
    """

    def __init__(self, records, frame_samples: int = FRAME_SAMPLES, ip: str = "127.0.0.1", pace: float = 1.0,
                 lead_seconds: float = 1.0, ping_interval: float | None = None, linger: float = 10.0, **stream_options):
        if pace <= 0.0:
            raise ValueError("pace is a multiple of real time and must be positive")
        self.stream = FrameStream(records, frame_samples, ping_interval=ping_interval, lead_seconds=lead_seconds, **stream_options)
        self.ip, self.pace, self.linger = ip, float(pace), float(linger)
        self.sent = self.idle_sent = 0

    def frames(self) -> Iterator[bytes]:
        return iter(self.stream)

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
                    log("receiver connected")
                else:
                    wait = t0 + k * period - time.perf_counter()
                    if wait > 0.0:
                        time.sleep(wait)
                    n = data_plane.send(array)
                if n < 0:
                    raise RuntimeError(f"the emulator failed to send frame {k}")
                if n == 0:
                    log(f"the receiver disconnected at frame {k}")
                    break
                self.sent += 1
            else:
                k = self.stream.n_frames
                for frame in self.stream.idle():
                    wait = t0 + k * period - time.perf_counter()
                    if wait > 0.0:
                        time.sleep(wait)
                    if (k - self.stream.n_frames) * period > self.linger:
                        break
                    if data_plane.send(np.frombuffer(frame, dtype=np.uint8).copy()) <= 0:
                        break
                    self.idle_sent += 1
                    k += 1
        finally:
            hsb.stop()
        out = {"frames_sent": self.sent, "idle_frames_sent": self.idle_sent, "frame_bytes": self.stream.frame_bytes,
               "frames_per_second": 1.0 / period}
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
                 "timestamp_s", "timestamp_ns")

try:
    from holoscan.core import Application, Operator
except Exception:  # noqa: BLE001  (the SDK needs a CUDA runtime even to import)
    Application = Operator = None

if Operator is not None:

    class HsbFramesToPingsOp(Operator):
        """Receiver frames in ("input"), ping records out ("ping", `{"record": record}` as PingSourceOp)."""

        def __init__(self, fragment, *args, receiver: FrameReceiver, cuda: CudaDriver | None = None, **kwargs):
            self.receiver = receiver
            self._cuda = cuda
            self._ready = []
            self.last_metadata = {}
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
            self._ready += self.receiver.take(frame, metadata or None)
            if self._ready:                                   # at most one ping completes per frame
                op_output.emit({"record": self._ready.pop(0), "t_source": time.perf_counter()}, "ping")

    class LiveChainOp(Operator):
        """Runs `gpusonar.live.LiveChain` on each record: recording, the chain (Python or C++
        engine), report file and report datagrams are the chain's. Keeps the reports, calls
        `on_report(report)` and `on_done()` after `n_pings`."""

        def __init__(self, fragment, *args, chain=None, n_pings: int | None = None, on_report=None, on_done=None, **kwargs):
            self.chain = chain
            self.reports = []
            self._n = n_pings
            self._on_report, self._on_done = on_report, on_done
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec):
            spec.input("ping")

        def compute(self, op_input, op_output, context):
            message = op_input.receive("ping")
            if self._n is not None and len(self.reports) >= self._n:
                return
            report = self.chain.process(message["record"])
            self.reports.append(report)
            if self._on_report:
                self._on_report(report)
            if self._n is not None and len(self.reports) >= self._n and self._on_done:
                self._on_done()

    class HsbLiveApp(Application):
        """LinuxReceiverOp -> HsbFramesToPingsOp -> LiveChainOp.

        The receiver keeps up to `queue_frames` frames while the chain works on a ping (it
        drops the oldest beyond that, which `FrameReceiver` counts as lost frames). Ends
        after `n_pings` reports, or when no frame has arrived for `idle_timeout` seconds.
        """

        def __init__(self, chain, receiver: FrameReceiver, hololink_channel, cuda: CudaDriver, frame_size: int,
                     n_pings: int | None = None, queue_frames: int = 256, idle_timeout: float | None = 15.0,
                     on_report=None, receiver_affinity=None):
            self.chain, self.receiver = chain, receiver
            self._channel, self._cuda, self._frame_size = hololink_channel, cuda, int(frame_size)
            self._n, self._queue, self._idle = n_pings, int(queue_frames), idle_timeout
            self._on_report, self._affinity = on_report, receiver_affinity
            self.frames_op = self.chain_op = None
            self._running = None
            super().__init__()

        def _finish(self):
            self._running.disable_tick()

        def _watch(self):
            """Stop the application when the stream has gone quiet."""
            seen, t_seen = -1, time.perf_counter()
            while self._running.check_tick_enabled():
                time.sleep(0.25)
                count = self.receiver.frames + self.receiver.frames_damaged + self.receiver.bad_frames
                if count != seen:
                    seen, t_seen = count, time.perf_counter()
                elif time.perf_counter() - t_seen > self._idle:
                    self._running.disable_tick()
                    try:
                        self.stop_execution()               # wakes a scheduler that is waiting for a frame
                    except AttributeError:                  # an SDK without it: the receiver's next tick ends the run
                        pass
                    return

        def compose(self):
            import threading

            import hololink
            from holoscan.conditions import BooleanCondition
            self._running = BooleanCondition(self, name="running", enable_tick=True)
            # Sensor Bridge pins its receiver thread to core 2 (HOLOLINK_AFFINITY names another); its Python
            # binding takes one core number here, not a set (a set fails to convert in 2.7.0).
            options = {} if self._affinity is None else {"receiver_affinity": int(self._affinity)}
            rx = hololink.operators.LinuxReceiverOp(
                self, self._running, name="receiver", frame_size=self._frame_size, frame_context=self._cuda.context,
                hololink_channel=self._channel, device=Board(), pages=self._queue, queue_size=self._queue, **options)
            self.frames_op = HsbFramesToPingsOp(self, receiver=self.receiver, cuda=self._cuda, name="frames_to_pings")
            self.chain_op = LiveChainOp(self, chain=self.chain, n_pings=self._n, on_report=self._on_report,
                                        on_done=self._finish, name="chain")
            self.add_flow(rx, self.frames_op, {("output", "input")})
            self.add_flow(self.frames_op, self.chain_op, {("ping", "ping")})
            if self._idle:
                threading.Thread(target=self._watch, daemon=True).start()


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
    try:
        metadata = hololink.Enumerator.find_channel(channel_ip=hololink_ip, timeout=hololink.Timeout(float(find_timeout)))
        channel = hololink.DataChannel(metadata)
        app = HsbLiveApp(chain, receiver, channel, cuda, receiver.frame_bytes, n_pings=n_pings, on_report=on_report, **app_options)
        board = channel.hololink()
        board.start()
        try:
            board.reset()
            app.run()
        finally:
            board.stop()
    finally:
        cuda.release()
    return app.chain_op.reports, receiver.stats()
