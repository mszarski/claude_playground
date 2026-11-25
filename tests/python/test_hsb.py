"""The flank stream through NVIDIA Holoscan Sensor Bridge (gpusonar/sources/hsb.py): framing and
frame handling everywhere; the loopback through Sensor Bridge's emulator and its Linux receiver
operator where Sensor Bridge is built (tools/build_hsb.sh) and a CUDA device exists."""

import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.live import LiveChain
from gpusonar.sources import hsb
from gpusonar.sources.flank_stream import decode_block
from test_live import CHAIN, INTERVAL, N, OVERRIDES, copies, pings  # noqa: F401  (pings is a fixture)

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "sonar_live_hsb.py"
FRAME = 400                        # samples per frame: 6.25 ms, 38,528 bytes
LEAD = 1.0                         # s of stream before the first transmit


def assemble(frames, **options):
    receiver = hsb.FrameReceiver("flank_32", OVERRIDES, **options)
    return [r for f in frames for r in receiver.take(f)], receiver


def test_frames_are_all_one_size_and_hold_the_pings(pings):
    stream = hsb.FrameStream(pings, FRAME, lead_seconds=LEAD)
    frames = list(stream)
    assert len(frames) == stream.n_frames and {len(f) for f in frames} == {hsb.frame_bytes(32, FRAME)} == {38528}
    assert len(frames[0]) % 128 == 0
    heads = [decode_block(f) for f in frames]
    assert [h["sample_index"] for h in heads] == [k * FRAME for k in range(len(frames))]      # continuous, tail padded
    assert heads[0]["tx_sample"] == -1 and heads[-1]["ping_index"] == N - 1
    assert heads[-1]["n_samples"] == FRAME and not heads[-1]["counts"][-1].any()
    records, receiver = assemble(frames, frame_samples=FRAME)
    assert len(records) == N and receiver.stats()["dropped_pings"] == 0 and receiver.stats()["lost_samples"] == 0
    lsb = pings[0].metadata.adc_full_scale / 2**23
    for got, want in zip(records, pings):
        assert got.samples.array.shape == want.samples.array.shape
        assert np.abs(got.samples.array - want.samples.array).max() <= 0.5 * lsb * (1 + 1e-6)   # the 24-bit quantisation
        assert got.metadata.sequence == want.metadata.sequence
        # Records that start at time zero come out later by the lead; the spacing is the pings'.
        assert got.transmit.time_ns - records[0].transmit.time_ns == want.transmit.time_ns - pings[0].transmit.time_ns
    assert records[0].transmit.time_ns == int(LEAD * 1e9)
    # With an epoch to count from, the times are the records' own.
    late = copies(pings)
    for r in late:
        t = r.transmit
        t.time_ns += 1_700_000_000_000_000_000
        r.transmit = t
    again, _ = assemble(hsb.frames_from_records(late, FRAME, lead_seconds=LEAD))
    assert [r.transmit.time_ns for r in again] == [r.transmit.time_ns for r in late]
    # A second pass over the same stream gives the same bytes; idle frames continue it.
    assert list(stream)[7] == frames[7]
    idle = decode_block(next(stream.idle()))
    assert idle["sample_index"] == len(frames) * FRAME and not idle["counts"].any()


def test_frame_sizes_the_data_plane_cannot_carry_are_refused(pings):
    with pytest.raises(ValueError, match="multiple of 128"):
        hsb.frame_bytes(32, 1601)
    with pytest.raises(ValueError, match="shorter frames"):          # the window opens 8533 samples after the transmit
        hsb.FrameStream(pings, 12800, lead_seconds=0.05)
    with pytest.raises(ValueError, match="before the receive window"):
        hsb.FrameStream(pings, FRAME, ping_interval=0.2)
    assert hsb.FrameStream(pings, 6400).frame_bytes == 614528        # the size the stream's notes name


def test_a_dropped_frame_drops_exactly_the_ping_it_cuts(pings):
    stream = hsb.FrameStream(pings, FRAME, lead_seconds=LEAD)
    frames = list(stream)
    window = stream.tx[2] + stream.window_start + stream.window_length // 2
    cut = window // FRAME                                            # a frame in the middle of ping 2's window
    meta = [{"frame_number": k, "frame_bytes_received": len(f), "packets_dropped": 0} for k, f in enumerate(frames)]
    # Lost outright (the receiver's queue overflowed, or its last packet was lost).
    receiver = hsb.FrameReceiver("flank_32", OVERRIDES, frame_samples=FRAME)
    got = [r for k, f in enumerate(frames) if k != cut for r in receiver.take(f, meta[k])]
    s = receiver.stats()
    assert [r.metadata.sequence for r in got] == [0, 1, 3, 4, 5]
    assert (s["frames_lost"], s["dropped_pings"], s["lost_samples"], s["frames"]) == (1, 1, FRAME, len(frames) - 1)
    lsb = pings[0].metadata.adc_full_scale / 2**23
    assert np.abs(got[2].samples.array - pings[3].samples.array).max() <= 0.5 * lsb * (1 + 1e-6)
    # Arrived with packets missing: not used, same result.
    receiver = hsb.FrameReceiver("flank_32", OVERRIDES, frame_samples=FRAME)
    got = []
    for k, f in enumerate(frames):
        m = dict(meta[k], packets_dropped=3 if k >= cut else 0)
        if k == cut:
            f = f[:4096] + b"\xff" * 4096 + f[8192:]
            m["frame_bytes_received"] = len(f) - 4096
        got += receiver.take(f, m)
    s = receiver.stats()
    assert [r.metadata.sequence for r in got] == [0, 1, 3, 4, 5]
    assert (s["frames_damaged"], s["frames_lost"], s["dropped_pings"], s["packets_dropped"]) == (1, 0, 1, 3)
    # A frame between pings costs nothing but is counted; a repeated frame is ignored.
    receiver = hsb.FrameReceiver("flank_32", OVERRIDES, frame_samples=FRAME)
    quiet = (stream.tx[1] - 100) // FRAME
    got = []
    for k, f in enumerate(frames):
        if k != quiet:
            got += receiver.take(f, meta[k])
            if k == cut:
                got += receiver.take(f, meta[k])
    s = receiver.stats()
    assert len(got) == N and (s["frames_lost"], s["frames_repeated"], s["dropped_pings"]) == (1, 1, 0)
    # A receiver that joins late starts counting at the frame it first sees.
    receiver = hsb.FrameReceiver("flank_32", OVERRIDES, frame_samples=FRAME)
    got = [r for k, f in enumerate(frames) if k >= 20 for r in receiver.take(f, meta[k])]
    assert len(got) == N and receiver.stats()["first_frame_number"] == 20 and receiver.stats()["frames_lost"] == 0
    # Something that is not a frame of this stream.
    assert receiver.take(b"\0" * len(frames[0]), None) == [] and receiver.take(frames[0][:-128], None) == []
    assert receiver.stats()["bad_frames"] == 2


def test_receive_stops_when_its_pings_are_accounted_for_or_the_stream_is_quiet():
    """ReceiveStop (sonar-dkb): the stop rule of the receive application, with a stand-in clock."""
    now = [0.0]
    clock = lambda: now[0]                                                   # noqa: E731
    rule = hsb.ReceiveStop(n_pings=6, idle_timeout=15.0, clock=clock)
    assert rule.check(0, 0, 0) is None
    now[0] = 14.0
    assert rule.check(0, 0, 0) is None                                       # quiet, but not for 15 s yet
    now[0] = 15.5
    assert rule.check(0, 0, 0) == "idle"                                     # a stream that never started
    # All six reported: stop at once.
    rule = hsb.ReceiveStop(6, 15.0, clock=clock)
    assert rule.check(5, 0, 100) is None and rule.check(6, 0, 101) == "pings" and rule.reason == "pings"
    # One ping lost whole, the other five reported: accounted for, stop without waiting for the idle timeout.
    rule = hsb.ReceiveStop(6, 15.0, clock=clock)
    assert rule.check(5, 1, 102) == "pings"
    # The last ping lost (so never counted as dropped) and the stream stopped early: the idle timeout stops it,
    # counted from the last frame heard, not from the start.
    now[0] = 100.0
    rule = hsb.ReceiveStop(6, 15.0, clock=clock)
    for k in range(50):
        now[0] += 0.25
        assert rule.check(min(5, k // 8), 0, 10 * k) is None                # frames keep coming
    t_last = now[0]
    while rule.check(5, 0, 490) is None:
        now[0] += 0.25
    assert rule.reason == "idle" and 15.0 < now[0] - t_last <= 15.25
    # Without a count only the idle rule stops; without an idle timeout only the count does.
    rule = hsb.ReceiveStop(None, 15.0, clock=clock)
    assert rule.check(10**6, 0, 1) is None
    rule = hsb.ReceiveStop(6, None, clock=clock)
    now[0] += 10**6
    assert rule.check(0, 0, 1) is None and rule.check(6, 0, 1) == "pings"
    # What the tool says when it stopped short (and exits with status 3); nothing when it did not.
    stats = {"reports": 5, "stopped": "idle", "dropped_pings": 0, "pings_discarded_at_stop": 0, "frames_lost": 16,
             "frames_damaged": 0, "lost_samples": 6400, "frames": 1980}
    assert hsb.stopped_short(dict(stats, reports=6), 6) is None and hsb.stopped_short(stats, None) is None
    said = hsb.stopped_short(stats, 6)
    assert said.startswith("stopped short (idle): 5 of 6 pings reported") and "frames lost 16" in said


STAND_IN_RECEIVER = r"""
# A HsbLiveApp with Sensor Bridge's receiver operator replaced by a stand-in that behaves as LinuxReceiverOp
# does in its two modes (use_frame_ready_condition, Sensor Bridge 2.7.0 base_receiver_op.cpp): an asynchronous
# condition set by a feeding thread per frame, or a compute that waits for a frame for at most 1 s.
import json, queue, sys, threading, time, types
import numpy as np
import holoscan
from holoscan.conditions import AsynchronousCondition, AsynchronousEventState
sys.path.insert(0, sys.argv[1])
from test_hsb import CHAIN, FRAME, LEAD, OVERRIDES, copies
from test_live import pings
from gpusonar.live import LiveChain
from gpusonar.sources import hsb

records = pings.__wrapped__()
stream = hsb.FrameStream(copies(records), FRAME, lead_seconds=LEAD)
frames = list(stream)
last = int(sys.argv[2])                    # frames from this one on are never sent: the stream stops early
period = 0.0005

class StandInReceiverOp(hsb.Operator):
    def __init__(self, fragment, *args, ready=None, **kwargs):
        self.ready, self.q, self.k = ready, queue.Queue(), 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.output("output")

    def start(self):
        def feed():
            for k in range(min(last, len(frames))):
                time.sleep(period)
                self.q.put(k)
                if self.ready is not None:
                    self.ready.event_state = AsynchronousEventState.EVENT_DONE
        threading.Thread(target=feed, daemon=True).start()

    def compute(self, op_input, op_output, context):
        try:
            k = self.q.get(timeout=0 if self.ready is not None else 1.0)
        except queue.Empty:
            return
        if self.ready is not None:
            self.ready.event_state = AsynchronousEventState.EVENT_WAITING
            if not self.q.empty():
                self.ready.event_state = AsynchronousEventState.EVENT_DONE
        self.metadata["frame_number"] = k
        self.metadata["frame_bytes_received"] = len(frames[k])
        self.metadata["packets_dropped"] = 0
        op_output.emit({"": holoscan.as_tensor(np.frombuffer(frames[k], dtype=np.uint8).copy())}, "output")

def linux_receiver_op(fragment, *conditions, name, use_frame_ready_condition=True, **kwargs):
    ready = AsynchronousCondition(fragment, name="frame_ready") if use_frame_ready_condition else None
    return StandInReceiverOp(fragment, *conditions, *([ready] if ready is not None else []), ready=ready, name=name)

sys.modules["hololink"] = types.SimpleNamespace(operators=types.SimpleNamespace(LinuxReceiverOp=linux_receiver_op))
receiver = hsb.FrameReceiver("flank_32", OVERRIDES, frame_samples=FRAME)
t0 = time.monotonic()
with LiveChain(CHAIN) as chain:
    app = hsb.HsbLiveApp(chain, receiver, None, types.SimpleNamespace(context=0), receiver.frame_bytes, n_pings=len(records),
                         idle_timeout=1.0)
    app.run()
receiver.finish(keep=False)
stats = dict(receiver.stats(), reports=len(app.chain_op.reports), stopped=app.stop_reason or "ended")
print("RESULT " + json.dumps({"seconds": time.monotonic() - t0, "stats": stats,
                              "short": hsb.stopped_short(stats, len(records)), "tx": [int(t) for t in stream.tx]}))
"""


@pytest.mark.parametrize("cut_ping", [None, 5, 3])
def test_receive_application_stops_when_the_stream_ends_early(pings, cut_ping):
    """sonar-dkb: the receive application ends, and says what it lost, when fewer than n_pings pings arrive.
    Holoscan's runtime and our operators behind a stand-in for Sensor Bridge's receiver operator (no
    Sensor Bridge, no GPU); the stream stops before ping `cut_ping`'s window (None: nothing is cut). Before
    the fix the application waited for ever once frames stopped: the receiver operator's asynchronous
    condition is looked at again only when a frame sets it."""
    if hsb.Operator is None:
        pytest.skip("Holoscan SDK not usable here")
    stream = hsb.FrameStream(copies(pings), FRAME, lead_seconds=LEAD)
    n_frames = len(list(stream))
    last = n_frames if cut_ping is None else (stream.tx[cut_ping] + stream.window_start) // FRAME
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "python"), os.environ.get("PYTHONPATH", "")]))
    try:
        out = subprocess.run([sys.executable, "-c", STAND_IN_RECEIVER, str(ROOT / "tests" / "python"), str(last)],
                             capture_output=True, text=True, env=env, timeout=240)
    except subprocess.TimeoutExpired:
        pytest.fail("the receive application did not end within 240 s after its stream stopped")
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads([ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")][-1][7:])
    stats = got["stats"]
    print("stand-in receive:", got)
    if cut_ping is None:
        assert stats["stopped"] == "pings" and stats["reports"] == N and got["short"] is None
    else:
        assert stats["stopped"] == "idle" and stats["reports"] == cut_ping
        # The pings never heard are not counted anywhere (nothing after them names them); what was heard is.
        assert stats["pings"] + stats["dropped_pings"] + stats["pings_discarded_at_stop"] == cut_ping
        assert got["short"].startswith(f"stopped short (idle): {cut_ping} of {N} pings reported")


class FakeTensor:
    """What a host-memory tensor looks like to numpy."""

    def __init__(self, data):
        self._a = np.frombuffer(data, dtype=np.uint8)
        self.shape = self._a.shape

    @property
    def __array_interface__(self):
        return self._a.__array_interface__


class FakeCuda:
    """Stands in for the driver: 'device addresses' are indices into a list of buffers."""

    def __init__(self):
        self.buffers = []
        self.copies = 0

    def tensor(self, data):
        self.buffers.append(np.frombuffer(data, dtype=np.uint8))
        handle = len(self.buffers) - 1
        return type("GpuTensor", (), {"__cuda_array_interface__": {"data": (handle, False), "shape": (len(data),),
                                                                   "typestr": "|u1", "version": 3}})()

    def to_host(self, pointer, size):
        self.copies += 1
        return self.buffers[pointer][:size].copy()


def test_receiver_messages_become_ping_records(pings):
    """The operator's frame handling, on messages shaped like the receiver operator's: a map with
    one unnamed uint8 tensor (GPU memory on the real path), and metadata beside it."""
    frames = list(hsb.frames_from_records(pings[:2], FRAME, lead_seconds=0.1))
    cuda = FakeCuda()
    receiver = hsb.FrameReceiver("flank_32", OVERRIDES, frame_samples=FRAME)
    records = []
    for k, f in enumerate(frames):
        message = {"": cuda.tensor(f)} if k % 2 else {"": FakeTensor(f)}
        host = hsb.frame_to_host(message, cuda)
        assert host.dtype == np.uint8 and host.shape == (len(f),)
        records += receiver.take(host, {"frame_number": 100 + k, "frame_bytes_received": len(f), "packets_dropped": 0})
    assert cuda.copies == len(frames) // 2 and len(records) == 2 and receiver.stats()["frames_lost"] == 0
    lsb = pings[0].metadata.adc_full_scale / 2**23
    assert np.abs(records[1].samples.array - pings[1].samples.array).max() <= 0.5 * lsb * (1 + 1e-6)
    assert np.array_equal(hsb.frame_to_host(FakeTensor(frames[3])), np.frombuffer(frames[3], dtype=np.uint8))
    with pytest.raises(RuntimeError, match="GPU memory"):
        hsb.frame_to_host({"": cuda.tensor(frames[0])}, None)
    # The assembled records run through the chain like any others.
    with LiveChain(CHAIN) as chain:
        report = chain.process(records[0])
    assert len(report["detections"]) >= 2 and any(abs(d["range_m"] - 250.0) < 15.0 for d in report["detections"])


def test_cuda_driver_copies_a_frame_from_device_memory():
    if not hsb.CudaDriver.available():
        pytest.skip("no CUDA device")
    import ctypes
    cuda = hsb.CudaDriver()
    lib = cuda._lib
    data = np.random.default_rng(4).integers(0, 256, hsb.frame_bytes(32, FRAME), dtype=np.uint8)
    lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    pointer = ctypes.c_uint64()
    assert lib.cuCtxSetCurrent(cuda.context) == 0 and lib.cuMemAlloc_v2(ctypes.byref(pointer), data.size) == 0
    try:
        assert lib.cuMemcpyHtoD_v2(pointer.value, data.ctypes.data, data.size) == 0
        tensor = type("GpuTensor", (), {"__cuda_array_interface__": {"data": (pointer.value, False), "shape": (data.size,),
                                                                     "typestr": "|u1", "version": 3}})()
        assert np.array_equal(hsb.frame_to_host({"": tensor}, cuda), data)
    finally:
        lib.cuMemFree_v2(pointer.value)
        cuda.release()


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_operators_in_a_holoscan_application_behind_a_stand_in_receiver(pings, engine):
    """HsbFramesToPingsOp and LiveChainOp inside the Holoscan runtime, fed by an operator that emits
    what the receiver operator does (a map with one unnamed uint8 tensor, metadata beside it) from
    host memory. Everything of the Sensor Bridge path except Sensor Bridge; the chain in either engine
    (engine cpp: sonar-8rj)."""
    if hsb.Operator is None:
        pytest.skip("Holoscan SDK not usable here")
    import holoscan
    from holoscan.conditions import CountCondition
    frames = list(hsb.frames_from_records(copies(pings), FRAME, lead_seconds=LEAD))
    chain_cfg = dict(CHAIN, engine=engine)
    cut = (2 * int(INTERVAL * 64000) + int(LEAD * 64000) + 30000) // FRAME        # inside ping 2's window

    class StandInReceiverOp(hsb.Operator):
        def __init__(self, fragment, *args, drop=(), **kwargs):
            self.k, self.drop = 0, set(drop)
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec):
            spec.output("output")

        def compute(self, op_input, op_output, context):
            while self.k in self.drop:
                self.k += 1
            frame = np.frombuffer(frames[self.k], dtype=np.uint8).copy()
            self.metadata["frame_number"] = self.k
            self.metadata["frame_bytes_received"] = frame.size
            self.metadata["packets_dropped"] = 0
            self.k += 1
            op_output.emit({"": holoscan.as_tensor(frame)}, "output")

    class App(hsb.Application):
        def __init__(self, chain, drop=()):
            self.chain, self.drop = chain, drop
            self.receiver = hsb.FrameReceiver("flank_32", OVERRIDES, frame_samples=FRAME)
            super().__init__()

        def compose(self):
            rx = StandInReceiverOp(self, CountCondition(self, len(frames) - len(self.drop)), drop=self.drop, name="receiver")
            self.frames_op = hsb.HsbFramesToPingsOp(self, receiver=self.receiver, name="frames_to_pings")
            self.chain_op = hsb.LiveChainOp(self, chain=self.chain, name="chain")
            self.add_flow(rx, self.frames_op, {("output", "input")})
            self.add_flow(self.frames_op, self.chain_op, {("ping", "ping")})

    with LiveChain(chain_cfg) as chain:
        direct = chain.run(hsb.records_from_frames(frames, "flank_32", OVERRIDES))
    with LiveChain(chain_cfg) as chain:
        app = App(chain)
        app.run()
    got, stats = app.chain_op.reports, app.receiver.stats()
    print("stand-in receiver:", stats, "metadata seen:", app.frames_op.last_metadata)
    assert stats["frames"] == len(frames) and stats["pings"] == N and stats["frames_lost"] == 0
    assert app.frames_op.last_metadata.get("frame_number") == len(frames) - 1     # the metadata travels with the frame
    strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}   # noqa: E731
    assert [strip(r) for r in got] == [strip(r) for r in direct]                  # the same records, so the same reports
    # A frame that never arrives costs the ping it cuts and nothing else.
    with LiveChain(chain_cfg) as chain:
        app = App(chain, drop=(cut,))
        app.run()
    stats = app.receiver.stats()
    assert [r["sequence"] for r in app.chain_op.reports] == [0, 1, 3, 4, 5]
    assert (stats["frames_lost"], stats["dropped_pings"]) == (1, 1)


def test_tool_arguments(tmp_path, pings):
    """The tool's two roles parse, and `receive` resolves the sonar, chain and outputs without Sensor Bridge."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import sonar_live_hsb as tool
    finally:
        sys.path.pop(0)
    a = tool.parse(["serve", "--scenario", "flank_patrol", "--pace", "2", "--frame-samples", "800"])
    assert (a.role, a.scenario, a.ip, a.pace, a.frame_samples) == ("serve", "flank_patrol", "127.0.0.1", 2.0, 800)
    a = tool.parse(["receive", str(ROOT / "configs/deploy/popoto_vehicle.yaml"), "--pings", "4", "--report", "r.jsonl",
                    "--chain", '{"min_snr_db": 11.0}', "--engine", "cpp"])
    plan = tool.receive_plan(a)
    assert plan["sonar"] == "popoto_long_range" and plan["chain"]["min_snr_db"] == 11.0 and plan["chain"]["engine"] == "cpp"
    assert plan["chain"]["bearings_deg"] == [-60.0, 60.0, 0.5] and plan["report"] == "r.jsonl" and plan["nav"] == {"port": 5702}
    assert tool.parse(["serve", "--scenario", "flank_patrol"]).hold == 0.0                  # off by default
    assert tool.parse(["serve", "--scenario", "flank_patrol", "--hold", "30"]).hold == 30.0
    assert tool.parse(["receive", "--sonar", "flank_32"]).chain_omp_threads is None      # off by default: OpenMP as it is
    assert tool.parse(["receive", "--sonar", "flank_32", "--chain-omp-threads", "4"]).chain_omp_threads == 4
    for bad in ("0", "-2", "x"):
        with pytest.raises(SystemExit):
            tool.parse(["receive", "--sonar", "flank_32", "--chain-omp-threads", bad])
    # How long receive looks for the board (sonar-8rj): 30 s by default, as before; longer for a board that comes up
    # late, such as serve --scenario still simulating its pings.
    assert tool.parse(["receive", "--sonar", "flank_32"]).find_timeout == 30.0
    assert tool.parse(["receive", "--sonar", "flank_32", "--find-timeout", "600"]).find_timeout == 600.0
    for bad in ("0", "-1", "x"):
        with pytest.raises(SystemExit):
            tool.parse(["receive", "--sonar", "flank_32", "--find-timeout", bad])
    a = tool.parse(["receive", "--sonar", "flank_32", "--sonar-overrides", json.dumps(OVERRIDES), "--chain", json.dumps(CHAIN),
                    "--recording", str(tmp_path / "rec.h5")])
    plan = tool.receive_plan(a)
    assert plan["sonar_overrides"] == OVERRIDES and plan["chain"] == CHAIN and plan["nav"] is None and plan["report"] is None
    with pytest.raises(SystemExit):
        tool.parse(["serve"])                                         # nothing to send
    with pytest.raises(SystemExit):
        tool.receive_plan(tool.parse(["receive"]))                    # no sonar named
    w = gs.RecordWriter(str(tmp_path / "pings.h5"))
    for r in copies(pings[:2]):
        w.write(r)
    w.close()
    records, interval = tool.serve_records(tool.parse(["serve", "--replay", str(tmp_path / "pings.h5")]))
    assert len(records) == 2 and interval is None and hsb.FrameStream(records, FRAME).tx[1] == int(INTERVAL * 64000)


def test_receive_passes_its_options_to_the_receiver(monkeypatch, capsys):
    """`receive` hands --find-timeout (sonar-8rj) and the chain's engine to run_receiver, without Sensor Bridge."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import sonar_live_hsb as tool
    finally:
        sys.path.pop(0)
    seen = {}

    def fake_run_receiver(chain, sonar, sonar_overrides, **options):
        seen.update(options, sonar=sonar, engine=chain.cfg["engine"])
        return [], {"reports": 0, "stopped": "idle"}
    monkeypatch.setattr(hsb, "run_receiver", fake_run_receiver)
    for argv, find, engine in ((["--find-timeout", "600", "--engine", "cpp"], 600.0, "cpp"), ([], 30.0, "python")):
        seen.clear()
        assert tool.main(["receive", "--sonar", "flank_32", "--chain", json.dumps(CHAIN), "--quiet", *argv]) == 0
        assert (seen["find_timeout"], seen["engine"], seen["sonar"]) == (find, engine, "flank_32")
    assert '"stopped": "idle"' in capsys.readouterr().out


def test_receive_builds_its_navigation_as_sonar_live_does(monkeypatch, tmp_path):
    """`receive` takes every key of the deployment file's `nav` section, as tools/sonar_live.py does
    (NavListener.from_config), and refuses one it does not know before it looks for the board (sonar-plk). Before, it
    passed on port, host, static, lever arm and mount only: water_depth, keep_s, surface_z and the rest were dropped
    without a word."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import sonar_live_hsb as tool
    finally:
        sys.path.pop(0)
    seen = {}

    def fake_run_receiver(chain, sonar, sonar_overrides, **options):
        seen["nav"] = chain.nav
        return [], {"reports": 0, "stopped": "idle"}
    monkeypatch.setattr(hsb, "run_receiver", fake_run_receiver)
    nav = {"static": {"z": 20.0, "altitude": 30.0}, "water_depth": {"source": "altitude"}, "keep_s": 30.0, "surface_z": 1.5,
           "profile_step_m": 2.0, "history_margin_s": 3.0}
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(json.dumps({"source": {"sonar": "flank_32", "sonar_overrides": OVERRIDES}, "chain": CHAIN, "nav": nav}))
    assert tool.main(["receive", str(deploy), "--quiet"]) == 0
    got = seen["nav"]
    assert (got.keep_ns, got.surface_z, got.profile_step_m, got.history_margin_ns) == (30_000_000_000, 1.5, 2.0, 3_000_000_000)
    assert got._water is not None and got.socket is None             # the depth from the log's altitude; no port, no socket
    seen.clear()
    deploy.write_text(json.dumps({"source": {"sonar": "flank_32"}, "chain": CHAIN, "nav": dict(nav, keep=30.0)}))
    with pytest.raises(ValueError, match=r"nav\.keep: unknown key; did you mean 'keep_s'\?"):
        tool.receive_plan(tool.parse(["receive", str(deploy)]))       # refused before the board is looked for
    with pytest.raises(ValueError, match=r"nav\.keep: unknown key"):
        tool.main(["receive", str(deploy), "--quiet"])
    assert seen == {}
    deploy.write_text(json.dumps({"source": {"sonar": "flank_32"}, "chain": CHAIN, "nav": dict(nav, water_depth={"sorce": "altitude"})}))
    with pytest.raises(ValueError, match=r"nav\.water_depth\.sorce: unknown key"):
        tool.receive_plan(tool.parse(["receive", str(deploy)]))


OMP_CHILD = r"""
import ctypes, json, sys, threading
import gpusonar  # noqa: F401  (loads the OpenMP runtime the library is built with)
from gpusonar.sources import hsb
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if ("libgomp" in l or "libomp" in l) and l.split()[-1].startswith("/")})
omp = ctypes.CDLL(libs[0]) if libs else None
before = omp.omp_get_max_threads() if omp else None
got = {}
def chain_thread():                     # as the chain's operator thread: applied there, before its first parallel region
    got["applied"] = hsb.set_chain_omp_threads(int(sys.argv[1]))
    got["after"] = omp.omp_get_max_threads() if omp else None
t = threading.Thread(target=chain_thread)
t.start(); t.join()
print(json.dumps({"runtime": libs, "before": before, "main_after": omp.omp_get_max_threads() if omp else None, **got}))
"""


def test_run_receiver_keeps_only_the_applications_whose_stop_failed(monkeypatch):
    """sonar-qnk: run_receiver keeps an application alive to the end of the process only when its stop failed
    (shutdown_error: its receiver thread may still be running); before, every call added its application to _KEEP.
    Sensor Bridge, Holoscan and the CUDA driver are stand-ins (no GPU)."""
    import types

    class StandInApp:
        made = []

        def __init__(self, chain, receiver, channel, cuda, frame_size, n_pings=None, on_report=None, fail_stop=False):
            self.stop_reason, self.fail_stop = None, fail_stop
            self.chain_op = types.SimpleNamespace(reports=[], omp_applied=None, wall_ns=[])
            self.socket_watch = types.SimpleNamespace(stats=lambda: {})
            self.arrivals = None
            StandInApp.made.append(self)

        def run(self):
            self.stop_reason = "idle"
            if self.fail_stop:          # the board gone: Sensor Bridge's unconfigure() times out in the operators' stop
                raise RuntimeError("control plane timeout")

    board = types.SimpleNamespace(start=lambda: None, reset=lambda: None, stop=lambda: None)
    hololink = types.SimpleNamespace(Enumerator=types.SimpleNamespace(find_channel=lambda **kw: None), Timeout=float,
                                     DataChannel=lambda metadata: types.SimpleNamespace(hololink=lambda: board))
    monkeypatch.setitem(sys.modules, "hololink", hololink)
    monkeypatch.setattr(hsb, "Application", object)
    monkeypatch.setattr(hsb, "HsbLiveApp", StandInApp, raising=False)     # absent without the Holoscan SDK
    monkeypatch.setattr(hsb, "CudaDriver", lambda: types.SimpleNamespace(release=lambda: None))
    monkeypatch.setattr(hsb, "_KEEP", [])
    for _ in range(3):
        stats = hsb.run_receiver(None, "flank_32", OVERRIDES, frame_samples=FRAME)[1]
        assert stats["stopped"] == "idle" and "shutdown_error" not in stats
    assert hsb._KEEP == []                                   # clean stops: nothing kept (before: all three)
    stats = hsb.run_receiver(None, "flank_32", OVERRIDES, frame_samples=FRAME, fail_stop=True)[1]
    assert stats["shutdown_error"] == "RuntimeError: control plane timeout"
    assert hsb._KEEP == [StandInApp.made[-1]]                # the one whose stop failed is kept


def test_chain_omp_threads_sets_openmps_thread_count_for_the_chains_thread():
    """receive --chain-omp-threads N (sonar-ihd): OpenMP's thread count for parallel regions the chain's thread
    starts is N once applied; other threads keep theirs. In a child process with OMP_NUM_THREADS unset."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMP_")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), os.environ.get("PYTHONPATH", "")])
    out = subprocess.run([sys.executable, "-c", OMP_CHILD, "3"], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    if not got["runtime"]:
        pytest.skip("the library is built without OpenMP")
    assert got["applied"] == 3 and got["after"] == 3
    assert got["main_after"] == got["before"]            # the setting is the calling thread's, not the process's


class FakeEmulation:
    """Stands in for `hololink.emulation`: a data plane that takes every frame at once, except
    that the send of frame `stall_at` takes `stall` seconds (a sender that falls behind)."""

    def __init__(self, stall_at=None, stall=0.0):
        outer = self

        class DataPlane:
            def __init__(self, *args):
                self.k = 0

            def send(self, array):
                if self.k == outer.stall_at:
                    time.sleep(outer.stall)
                self.k += 1
                return array.size

        self.HSBEmulator = type("HSBEmulator", (), {"start": lambda self: None, "stop": lambda self: None})
        self.LinuxDataPlane = DataPlane
        self.IPAddress = lambda ip: ip
        self.stall_at, self.stall = stall_at, stall


def test_a_sender_that_falls_behind_bursts_unless_told_to_spread(pings, monkeypatch):
    """The emulated board's timing record (for STATUS open item 34): after a 40 ms stall at 160 frames
    a second the frames owed leave back to back ("burst", as before) or at most two per period ("spread")."""
    import types
    monkeypatch.setenv("HSB_EMULATOR_UDP_SOCKET", "1")             # no raw socket needed by the stand-in
    package = types.ModuleType("hololink")
    for catch_up in hsb.CATCH_UP:
        fake = FakeEmulation(stall_at=40, stall=0.040)
        package.emulation = fake
        monkeypatch.setitem(sys.modules, "hololink", package)
        monkeypatch.setitem(sys.modules, "hololink.emulation", fake)
        board = hsb.HsbBoardEmulator(copies(pings[:2]), FRAME, pace=1.0, lead_seconds=0.1, ping_interval=1.6, linger=0.0,
                                     catch_up=catch_up)
        out = board.run(connect_timeout=5.0)
        t = out["timing"]
        assert out["frames_sent"] == board.stream.n_frames and out["catch_up"] == catch_up
        assert t["period_ms"] == pytest.approx(6.25) and t["max_late_ms"] > 30.0 and t["frames_late_by_a_period"] >= 1
        assert 40 in t["first_late_frames"] or 41 in t["first_late_frames"]
        if catch_up == "burst":
            assert t["back_to_back_frames"] >= 4                    # the six frames owed, sent at once
        else:
            assert t["back_to_back_frames"] == 0                    # never two within a quarter period
        print(catch_up, t)


def test_the_emulated_board_stays_up_quiet_for_hold_seconds_after_its_linger(pings, monkeypatch):
    """serve --hold (sonar-dkb): a board whose receiver is still connected after the linger stays up, sending
    nothing, so a receiver that stops on its idle timeout can still switch the data channel off; a receiver
    that disconnected during the linger ends it at once, as before."""
    import types
    monkeypatch.setenv("HSB_EMULATOR_UDP_SOCKET", "1")
    package = types.ModuleType("hololink")
    n_frames = hsb.FrameStream(copies(pings[:1]), FRAME, lead_seconds=0.05).n_frames
    for hold, disconnect_at in ((0.0, None), (0.6, None), (0.6, 3)):
        fake = FakeEmulation()
        stopped = []
        fake.HSBEmulator = type("HSBEmulator", (), {"start": lambda self: None, "stop": lambda self: stopped.append(time.perf_counter())})
        if disconnect_at is not None:                   # the receiver disconnects a few frames into the linger
            last = n_frames + disconnect_at

            class Plane:
                def __init__(self, *args):
                    self.k = 0

                def send(self, array):
                    self.k += 1
                    return 0 if self.k > last else array.size
            fake.LinuxDataPlane = Plane
        package.emulation = fake
        monkeypatch.setitem(sys.modules, "hololink", package)
        monkeypatch.setitem(sys.modules, "hololink.emulation", fake)
        board = hsb.HsbBoardEmulator(copies(pings[:1]), FRAME, pace=1.0, lead_seconds=0.05, linger=0.05, hold=hold)
        t0 = time.perf_counter()
        out = board.run(connect_timeout=5.0)
        t_end = stopped[-1] - t0
        sending = board.stream.n_frames * board.stream.frame_seconds
        print(hold, disconnect_at, round(t_end, 3), out["idle_frames_sent"])
        if disconnect_at is None:
            assert hold <= t_end - sending < hold + 0.4         # the linger (50 ms) and the hold, then the emulator stops
        else:
            assert t_end - sending < 0.3 and out["idle_frames_sent"] == disconnect_at   # disconnected in the linger: no hold


def test_the_data_socket_watch_finds_the_receivers_socket_and_counts_its_drops():
    """DataSocketWatch picks the UDP socket with LinuxReceiver's 100 ms receive timeout and reads the
    kernel's drop count for it (/proc/net/udp); a datagram refused by a full buffer is counted."""
    import socket
    import struct
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    other = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx.bind(("127.0.0.1", 0))
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, struct.pack("ll", 0, 100000))
        watch = hsb.DataSocketWatch(receive_buffer=4096)
        assert watch.locate() and watch.stats()["found"]
        assert watch.stats()["rcvbuf"] == rx.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)   # set through the dup
        before = hsb.udp_counters().get("RcvbufErrors", 0)
        for _ in range(64):                                         # far beyond a 4 KB buffer, nobody reading
            tx.sendto(b"x" * 1400, rx.getsockname())
        watch.sample()
        s = watch.stats()
        assert s["drops"] > 0 and s["max_rx_queue"] > 0
        if hsb.udp_counters():
            assert hsb.udp_counters()["RcvbufErrors"] - before >= s["drops"]
        print(s, hsb.cpu_throttling())
    finally:
        for sock in (rx, other, tx):
            sock.close()


def sensor_bridge_reason():
    try:
        import hololink  # noqa: F401
        import hololink.emulation  # noqa: F401
        hololink.operators.LinuxReceiverOp
    except Exception as exc:  # noqa: BLE001
        return f"Sensor Bridge not built here (tools/build_hsb.sh): {type(exc).__name__}"
    if not hsb.raw_sockets_permitted():
        return "the emulator's raw data socket is not permitted here (no CAP_NET_RAW)"
    if not hsb.CudaDriver.available():
        return "no CUDA device (the Sensor Bridge receiver puts frames in GPU memory)"
    return None


def lines(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.startswith("{")]


def run_loopback(tmp_path, pings, pace, serve_args=(), receive_args=(), serve_prefix=(), tag="run", receive_env=None,
                 scenario=None, stream=None, frame=FRAME, n_pings=N, check=True):
    """One loopback: `serve` (the board, through Sensor Bridge's emulator) and `receive` (Sensor
    Bridge's LinuxReceiverOp -> frames to pings -> chain) as separate processes, as deployed.
    `serve` replays `pings` from a file, or with `scenario` (a scenario file) simulates its pings itself.
    `stream`: receive's arguments naming the stream's sonar and chain (default: flank_32 with test_live's window and
    chain); `frame`: samples per frame on both sides; `n_pings`: receive's --pings. With `check` False a failed
    receive is returned, not asserted (its statistics are {} when it printed none).
    Returns (sender's statistics, receiver's statistics, the receive process, its report path)."""
    if scenario is None:
        w = gs.RecordWriter(str(tmp_path / "pings.h5"))
        for r in copies(pings):
            w.write(r)
        w.close()
    source = ("--scenario", str(scenario)) if scenario is not None else ("--replay", str(tmp_path / "pings.h5"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "python"), os.environ.get("PYTHONPATH", "")]))
    if stream is None:
        stream = ("--sonar", "flank_32", "--sonar-overrides", json.dumps(OVERRIDES), "--chain", json.dumps(CHAIN))
    serve = subprocess.Popen([*serve_prefix, sys.executable, str(TOOL), "serve", *source,
                              "--frame-samples", str(frame), "--lead", str(LEAD), "--pace", str(pace), "--linger", "20", "--hold", "30",
                              "--send-log", str(tmp_path / f"{tag}_send.json"), *serve_args],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        receive = subprocess.run([sys.executable, str(TOOL), "receive", *stream, "--frame-samples", str(frame), "--pings", str(n_pings),
                                  "--report", str(tmp_path / f"{tag}.jsonl"), "--recording", str(tmp_path / f"{tag}.h5"),
                                  "--arrival-log", str(tmp_path / f"{tag}_arrivals.json"), *receive_args],
                                 capture_output=True, text=True, timeout=600, env=dict(env, **(receive_env or {})))
        sent, _ = serve.communicate(timeout=120)
    finally:
        serve.kill()
    print(sent[-1500:])
    print(receive.stdout[-3000:])
    if check:
        assert receive.returncode == 0, receive.stderr[-3000:]
        assert serve.returncode == 0, sent[-3000:]
    said = [ln for ln in sent.splitlines() if ln.startswith("hsb: ")]
    got = [ln for ln in receive.stdout.splitlines() if ln.startswith("hsb: ")]
    board = json.loads(said[-1][5:]) if said else {}
    stats = json.loads(got[-1][5:]) if got else {}
    return board, stats, receive, tmp_path / f"{tag}.jsonl"


def receive_delays(tmp_path, tag):
    """The receive delay in ms over the frames in both logs of a loopback: when
    Sensor Bridge's receiver thread had a frame's last datagram (`received_ns`, its clock_gettime(CLOCK_REALTIME)
    right after recv) minus when the board started sending that frame (`wall_ns` of the send log). Both
    processes run on this host, so both are the same system clock: the sender takes its send times on
    time.perf_counter() and maps them to CLOCK_REALTIME with one offset read at its first sent frame
    (SendTiming.start), so there is no base to subtract; what the two clocks can drift apart over a run
    (NTP slewing, at most 0.05 %, so up to 10 ms over a 20 s run in the worst case, usually microseconds) or
    a step of the system clock would move the least and the median delay off the loopback's floor of a few
    tenths of a millisecond (0.21-0.22 ms in 20 L4 runs), which the caller checks. Frames are matched by the
    sender's frame number (the receiver's `frame_number` metadata); frame 0 is not in the send log.
    Returns (worst, least, median, frames)."""
    sent = {row[0]: row[1] for row in json.loads((tmp_path / f"{tag}_send.json").read_text())["rows"]}
    got = json.loads((tmp_path / f"{tag}_arrivals.json").read_text())["rows"]
    delays = [(row[1] - sent[row[0]]) / 1e6 for row in got if row[0] in sent]
    assert len(delays) > 0.9 * len(got)
    return max(delays), min(delays), float(np.median(delays)), len(delays)


def usable_cpus(cgroup_root="/sys/fs/cgroup") -> int:
    """CPUs this process may use: its affinity, capped by a cgroup CPU quota rounded up (cgroup v2 cpu.max, or v1
    cpu.cfs_quota_us / cpu.cfs_period_us). A container's vCPUs are often a quota: a Hugging Face a10g-small job
    (4 vCPUs) lists the host's 96 CPUs in its affinity (job 6b04b897c97b8ed4fcd689e1), sonar-8rj."""
    import math
    n = len(os.sched_getaffinity(0))
    root = pathlib.Path(cgroup_root)
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]
    except (OSError, ValueError):
        try:
            quota = (root / "cpu" / "cpu.cfs_quota_us").read_text().strip()
            period = (root / "cpu" / "cpu.cfs_period_us").read_text().strip()
        except OSError:
            return n
    if quota in ("max", "-1") or float(period) <= 0:
        return n
    return max(1, min(n, math.ceil(float(quota) / float(period))))


def test_usable_cpus_honours_a_cgroup_quota(tmp_path):
    n = len(os.sched_getaffinity(0))
    assert usable_cpus(tmp_path / "none") == n                       # no cgroup files: the affinity
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cpu.max").write_text("max 100000\n")
    assert usable_cpus(v2) == n                                       # no quota
    (v2 / "cpu.max").write_text("150000 100000\n")
    assert usable_cpus(v2) == min(n, 2)                               # 1.5 CPUs of quota: 2, rounded up
    v1 = tmp_path / "v1" / "cpu"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("100000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    assert usable_cpus(tmp_path / "v1") == 1
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    assert usable_cpus(tmp_path / "v1") == n


def loopback_against_direct(tmp_path, pings, engine, scenario=None, tag="hsb"):
    """A real-time loopback (`serve` replaying `pings`, or with `scenario` simulating them itself) received with
    the chain's `engine`, held to: nothing lost, the worst receive delay, and the reports of the same pings
    framed and assembled here and run straight through a chain of the same engine. Returns (got, direct)."""
    # Real time (owner's ruling 2026-11-21, D-153; GPUSONAR_HSB_PACE overrides). The margin is the receiver's
    # socket buffer (212,992 bytes, Hugging Face's rmem_max: about 20 ms of the stream): it overflowed while
    # the chain ran on a ping when the board sent its owed frames back to back; with the board's spread
    # catch-up (the default) 20 real-time runs lost nothing, and a halved buffer still loses
    # (test_loopback_loss_study; DEPLOYMENT.md says what a deployed host should set).
    # The chain's OpenMP runs on half the CPUs this process may use (sonar-ihd): with one OpenMP thread per vCPU,
    # flank.backprojection_map kept Sensor Bridge's receiver thread waiting 10-13 ms on an 8-vCPU L4 host and the
    # socket overflowed; with half, its worst delay was 1-3 ms in 16 runs. The socket buffer stays the default.
    # Half of the CPUs the container is given (usable_cpus: a CPU quota caps the affinity; sonar-8rj).
    pace = float(os.environ.get("GPUSONAR_HSB_PACE", "1.0"))
    omp = max(1, usable_cpus() // 2)
    receive_args = ("--chain-omp-threads", str(omp)) + (("--engine", engine) if engine != "python" else ())
    board, stats, receive, report = run_loopback(tmp_path, pings, pace, receive_args=receive_args, tag=tag, scenario=scenario)
    print("sender:", board)
    print("transport:", stats)
    assert stats.get("chain_omp_threads") in (omp, "no OpenMP runtime")
    assert stats["pings"] == N and stats["frames_lost"] == 0 and stats["frames_damaged"] == 0 and stats["dropped_pings"] == 0
    assert stats["bad_frames"] == 0 and stats["lost_samples"] == 0
    # Headroom, so that a receiver stall the buffer still absorbed fails too: the worst receive delay.
    worst, floor, median, n = receive_delays(tmp_path, tag)
    print(f"receive delay over {n} frames: worst {worst:.2f} ms, median {median:.2f} ms, least {floor:.2f} ms "
          f"(chain OpenMP threads {omp}, engine {engine})")
    assert -1.0 < floor and median < 2.0          # the two clocks agree (see receive_delays)
    assert worst < 5.0
    # The same pings straight through the chain: framed and assembled here, no transport.
    direct_records = hsb.records_from_frames(hsb.frames_from_records(copies(pings), FRAME, lead_seconds=LEAD), "flank_32", OVERRIDES)
    sent_samples = direct_records[0].samples.array.copy()            # the chain conditions records in place
    with LiveChain(dict(CHAIN, engine=engine)) as chain:
        direct = chain.run(direct_records)
    got = lines(report)
    assert len(got) == len(direct) == N
    assert_same_reports(got, direct)
    # What was recorded on the way in is what the board sent.
    raw = list(gs.sources.make_source({"type": "container", "path": str(tmp_path / f"{tag}.h5")}))
    assert len(raw) == N and np.array_equal(raw[0].samples.array, sent_samples)
    return got, direct


def assert_same_reports(got, direct, tolerance=0.05):
    """The same pings, tracks and detection counts; positions, ranges and bearings within `tolerance` (m, deg)."""
    worst = {"track_m": 0.0, "range_m": 0.0, "bearing_deg": 0.0}
    for a, b in zip(got, direct):
        assert (a["sequence"], a["time_ns"]) == (b["sequence"], b["time_ns"])
        assert [(t["id"], t["kind"]) for t in a["tracks"]] == [(t["id"], t["kind"]) for t in b["tracks"]]
        assert len(a["detections"]) == len(b["detections"])
        for ta, tb in zip(a["tracks"], b["tracks"]):
            worst["track_m"] = max(worst["track_m"], float(np.hypot(ta["x"] - tb["x"], ta["y"] - tb["y"])))
        for da, db in zip(a["detections"], b["detections"]):
            worst["range_m"] = max(worst["range_m"], abs(da["range_m"] - db["range_m"]))
            worst["bearing_deg"] = max(worst["bearing_deg"], abs(da["bearing_deg"] - db["bearing_deg"]))
    print("largest differences from the direct chain:", worst)
    assert worst["track_m"] < tolerance and worst["range_m"] < tolerance and worst["bearing_deg"] < tolerance


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_loopback_through_sensor_bridge_gives_the_direct_chains_reports(pings, tmp_path, engine):
    """The board replays recorded pings; `receive` runs the chain in either engine (sonar-8rj: engine cpp)."""
    reason = sensor_bridge_reason()
    if reason:
        pytest.skip(reason)
    got, _ = loopback_against_direct(tmp_path, pings, engine)
    # Two confirmed tracks. The stream carries no navigation, so the records sit at the origin and the
    # standing target takes ownship's speed (test_live.py shows the same without navigation).
    assert [t["kind"] for t in got[-1]["tracks"]] == ["mover", "mover"]


# The pings of test_live.pings as a scenario the board simulates itself (serve --scenario, sonar-8rj): ownship at
# 1.5 m/s along its array, a target crossing at 3 m/s and one standing, free field, the same receive window.
SCENARIO = {
    "name": "hsb_loopback", "sonar": "flank_32", "fidelity": "free_field", "sonar_overrides": OVERRIDES,
    "platform": {"position": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0], "velocity": [1.5, 0.0, 0.0]},
    "pings": {"count": N, "interval": INTERVAL},
    "scene": [{"type": "points", "label": "crossing", "positions": [[40.0, 250.0, 0.0]], "velocity": [-3.0, 0.0, 0.0],
               "target_strength_db": 10.0},
              {"type": "points", "label": "standing", "positions": [[-60.0, 320.0, 0.0]], "target_strength_db": 10.0}],
    "receiver": {"noise_rms": 1e-5, "adc": True},
    "seed": 5,
}


def scenario_file(tmp_path):
    import yaml
    path = tmp_path / "hsb_loopback.yaml"
    path.write_text(yaml.safe_dump(SCENARIO, sort_keys=False))
    return path


def test_the_scenario_the_board_simulates_is_framed_like_recorded_pings(tmp_path):
    """serve --scenario (sonar-8rj) without Sensor Bridge: the tool's own simulation of the scenario file is
    deterministic, its pings fit the stream's frames, and the chain finds both targets in them."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import sonar_live_hsb as tool
    finally:
        sys.path.pop(0)
    path = scenario_file(tmp_path)
    records, interval = tool.serve_records(tool.parse(["serve", "--scenario", str(path)]))
    again, _ = tool.serve_records(tool.parse(["serve", "--scenario", str(path)]))
    assert len(records) == N and interval is None                  # timed records: their own transmit times
    assert all(np.array_equal(a.samples.array, b.samples.array) for a, b in zip(records, again))
    stream = hsb.FrameStream(records, FRAME, lead_seconds=LEAD)
    assert stream.tx[1] - stream.tx[0] == int(INTERVAL * 64000)
    assembled = hsb.records_from_frames(iter(stream), "flank_32", OVERRIDES, frame_samples=FRAME)
    assert len(assembled) == N
    assert [r.transmit.time_ns for r in assembled] == [r.transmit.time_ns for r in records]
    for engine in ("python", "cpp"):
        with LiveChain(dict(CHAIN, engine=engine)) as chain:
            reports = chain.run(hsb.records_from_frames(hsb.frames_from_records(records, FRAME, lead_seconds=LEAD),
                                                        "flank_32", OVERRIDES))
        assert len(reports) == N
        assert any(abs(d["range_m"] - 250.0) < 15.0 for d in reports[0]["detections"])
        assert any(abs(d["range_m"] - 325.0) < 15.0 for d in reports[0]["detections"])
        assert len(reports[-1]["tracks"]) == 2, (engine, reports[-1]["tracks"])


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_loopback_of_a_scenario_the_board_simulates_gives_the_direct_chains_reports(tmp_path, engine):
    """serve --scenario (sonar-8rj): the emulated board simulates its pings itself; `receive` in either engine
    reports what the same pings, simulated here, framed and assembled, give straight through the chain."""
    reason = sensor_bridge_reason()
    if reason:
        pytest.skip(reason)
    from gpusonar.sim import run_scenario
    path = scenario_file(tmp_path)
    pings = run_scenario(str(path)).records                           # as serve_records simulates them
    got, _ = loopback_against_direct(tmp_path, pings, engine, scenario=path, tag="hsb_scenario")
    assert len(got[-1]["tracks"]) == 2


# --- A Popoto build's channel count and rate through Sensor Bridge (sonar-x8v) ---------------------------------------
# Every loopback above runs flank_32: 32 channels at 64 kHz, 49 Mbit/s. The Popoto builds are 32, 64 and 96 channels
# at 102.4 kHz (docs/handoff/hardware_builds.json: 78.6, 157.3 and 235.9 Mbit/s of samples). The trial for each row
# count is its build's shortest simulated one, received with its build's deployment file (chain, sync, elevation):
#   1 row:  popoto_testauv_trial_1row (4 boards on their own clocks, 150-900 m window, 4 pings 3 s apart);
#   2 rows: popoto_static_trial_3row's scene with the two-row sonar (popoto_static_trial_2row looks 2.45-2.75 km out,
#           5 s apart: 21 s of stream and about 100 s of simulation on one CPU, against 11 s and 5 s; the stream's rate is
#           the channel count's whatever the window);
#   3 rows: popoto_static_trial_3row (12 boards on their own clocks, 90-400 m window, 4 pings 3 s apart).
# The stream is continuous (zeros between windows), so the transport carries the build's full rate all the time; what
# a longer receive window (12.7 km: 17 s a ping) adds is chain time and memory per ping, not stream rate.
POPOTO = {1: ("popoto_testauv_trial_1row", None, "popoto_testauv_1row"),
          2: ("popoto_static_trial_3row", "popoto_static_2row_clocks", "popoto_static_2row"),
          3: ("popoto_static_trial_3row", None, "popoto_static_3row")}
# Seconds of stream the receiver operator holds while the chain works on a ping (`receive --queue`, in frames): the
# tool's default was 256 frames when these were measured (1.6 s of flank_32's 160 frames a second, 1.0 s of a Popoto
# build's 256); since D-187 it is these 4 s of whatever stream. The queue drops the oldest frames beyond it.
POPOTO_QUEUE_S = 4.0
_popoto_cache = {}


def popoto_trial(rows):
    """(records, deployment dict for `receive`) of a row count's trial. The deployment file is the build's with the
    trial's sonar (its boards on their own clocks), receive window and sound speed (the window's layout, D-183); a
    vehicle's navigation port and the file's outputs are left out (the stream carries no navigation; outputs are the
    test's). Simulated once per process."""
    if rows not in _popoto_cache:
        from gpusonar.config import load_scenario_config, load_yaml
        from gpusonar.sim import run_scenario
        scenario, sonar, deploy = POPOTO[rows]
        sc = load_scenario_config(scenario)
        if sonar:
            sc = dict(sc, sonar=sonar, name=f"{sc['name']}_as_{sonar}")
        records = run_scenario(sc).records
        cfg = load_yaml(str(ROOT / "configs" / "deploy" / f"{deploy}.yaml"))
        cfg["source"] = dict(cfg["source"], sonar=sc["sonar"], sonar_overrides=sc.get("sonar_overrides"),
                             sound_speed=float(records[0].environment.sound_speed))
        if (cfg.get("nav") or {}).get("port"):
            cfg.pop("nav")
        cfg.pop("output", None)
        _popoto_cache[rows] = (records, cfg)
    records, cfg = _popoto_cache[rows]
    return [r.copy() for r in records], json.loads(json.dumps(cfg))


def popoto_direct(records, cfg, engine, frame=FRAME, sequences=None, recording=None):
    """The same pings framed and assembled here, without transport, straight through the deployment's chain; with
    `sequences`, only the pings of those sequence numbers (the ones a lossy loopback reported), in order. Returns
    (reports, the first of those pings' samples). With `recording`, the chain records them there as `receive` does."""
    from gpusonar.live import NavListener
    src = cfg["source"]
    assembled = hsb.records_from_frames(hsb.frames_from_records(records, frame, lead_seconds=LEAD), src["sonar"],
                                        src.get("sonar_overrides"), sound_speed=src["sound_speed"])
    if sequences is not None:
        assembled = [r for r in assembled if int(r.metadata.sequence) in set(sequences)]
    sent = assembled[0].samples.array.copy()                          # the chain conditions records in place
    nav = NavListener.from_config(cfg["nav"]) if cfg.get("nav") else None
    with LiveChain(dict(cfg["chain"], engine=engine), nav=nav, recording=recording) as chain:
        return chain.run(assembled), sent


def stream_numbers(records, frame=FRAME):
    st = hsb.FrameStream(records, frame, lead_seconds=LEAD)
    fps = 1.0 / st.frame_seconds
    return {"channels": st.n_channels, "sample_rate": st.sample_rate, "frame_samples": frame, "frame_bytes": st.frame_bytes,
            "frames_per_s": fps, "mbit_s": st.frame_bytes * 8 * fps / 1e6, "frames": st.n_frames,
            "stream_s": st.n_frames * st.frame_seconds}


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_a_popoto_trial_is_framed_at_its_builds_rate_and_its_pings_come_back(rows):
    """Without Sensor Bridge: the stream of a Popoto build's trial in the loopback's frames carries the build's rate,
    and assembled again at the trial's sound speed it gives back every ping at its transmit time, quantised to the
    stream's 24 bits."""
    records, cfg = popoto_trial(rows)
    builds = {pathlib.Path(b["sonar_file"]).stem: b for b in json.loads((ROOT / "docs" / "handoff" / "hardware_builds.json").read_text())["builds"]}
    build = builds[cfg["source"]["sonar"].removesuffix("_clocks")]
    n = stream_numbers(records)
    assert n["channels"] == build["channels"] == 32 * rows and n["sample_rate"] == build["sample_rate_hz"] == 102400.0
    # 400-sample frames: 256 a second whatever the channel count; the rate is the samples' plus a 128-byte header a frame.
    assert n["frames_per_s"] == pytest.approx(256.0)
    assert n["frame_bytes"] * n["frames_per_s"] == pytest.approx(build["stream"]["payload_bytes_per_s"] + 128 * 256.0)
    src = cfg["source"]
    back = hsb.records_from_frames(hsb.frames_from_records(records, FRAME, lead_seconds=LEAD), src["sonar"], src.get("sonar_overrides"),
                                   sound_speed=src["sound_speed"])
    assert [r.transmit.time_ns for r in back] == [r.transmit.time_ns for r in records]
    lsb = records[0].metadata.adc_full_scale / hsb.FULL_SCALE_COUNTS
    for a, b in zip(back, records):
        assert a.samples.array.shape == b.samples.array.shape
        clipped = np.abs(b.samples.array) >= records[0].metadata.adc_full_scale
        # Half a step of the 24-bit word, plus the float32 rounding of the values (0.53 step measured at 2 and 3 rows).
        assert np.all(np.abs(a.samples.array - b.samples.array)[~clipped] <= 0.55 * lsb)


def popoto_loopback(tmp_path, rows, engine, tag, frame=FRAME, socket_buffer=None, queue_s=POPOTO_QUEUE_S, check=True, omp=None,
                    serve_args=()):
    """One real-time loopback of a row count's trial: `serve --replay` of its pings, `receive` with the build's
    deployment file, the chain's OpenMP on half the usable CPUs (D-162, D-173). Returns (board, stats, receive process,
    report path, measurements)."""
    import math

    import yaml
    records, cfg = popoto_trial(rows)
    path = tmp_path / f"{tag}_deploy.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    numbers = stream_numbers(records, frame)
    queue = math.ceil(queue_s * numbers["frames_per_s"])
    omp = omp or max(1, usable_cpus() // 2)
    pace = float(os.environ.get("GPUSONAR_HSB_PACE", "1.0"))
    receive_args = ["--chain-omp-threads", str(omp), "--queue", str(queue), "--engine", engine, "--quiet"]
    if socket_buffer:
        receive_args += ["--socket-buffer", str(socket_buffer)]
    board, stats, receive, report = run_loopback(tmp_path, records, pace, serve_args=serve_args, receive_args=receive_args, tag=tag,
                                                 stream=(str(path),), frame=frame, n_pings=len(records), check=check)
    m = dict(numbers, rows=rows, engine=engine, queue_frames=queue, omp=omp, pace=pace, exit=receive.returncode,
             interval_s=(records[1].transmit.time_ns - records[0].transmit.time_ns) * 1e-9)
    try:
        worst, floor, median, n = receive_delays(tmp_path, tag)
        m.update(worst_ms=worst, median_ms=median, least_ms=floor, delays=n)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        m["delay_error"] = f"{type(exc).__name__}: {exc}"
    try:
        log = json.loads((tmp_path / f"{tag}_arrivals.json").read_text())
        m["chain_s"] = [round((b - a) * 1e-9, 3) for a, b in log.get("chain_wall_ns", [])]
    except (OSError, ValueError):
        m["chain_s"] = []
    timing = board.get("timing") or {}
    m.update(sender_max_late_ms=timing.get("max_late_ms"), sender_late_by_a_period=timing.get("frames_late_by_a_period"),
             sender_back_to_back=timing.get("back_to_back_frames"))
    for key in ("reports", "pings", "frames", "frames_lost", "frames_damaged", "frames_late", "dropped_pings", "lost_samples",
                "bad_frames", "packets_dropped", "stopped", "chain_omp_threads"):
        m[key] = stats.get(key)
    m["socket"] = stats.get("socket")
    m["udp"] = stats.get("udp_counters")
    m["cpu_throttled_during"] = (stats.get("cpu") or {}).get("nr_throttled_during")
    if receive.returncode != 0:
        m["stderr"] = receive.stderr[-600:]
    return board, stats, receive, report, m


# What a Popoto-rate loopback is held to, and what it is not (D-186). Measured on Hugging Face a10g-large jobs (12-CPU
# quota, rmem_max 212,992 B; no L4 was available), 210 real-time runs with the C++ and Python engines
# (6b063a21d0582cfbbefd60a3, 6b0641ce1460ffe1d5e98956): frames were lost in 5 to 8 runs of 10 at 32 channels and in
# every run at 64 and 96, with the default socket buffer, with the buffer doubled to 425,984 B (--socket-buffer
# 212992, what the container's rmem_max allows), with 200-sample frames, and with the chain's OpenMP on 2 threads as on
# 6. Every datagram lost was refused by the receiver's full socket buffer (the socket's kernel drop count equals
# RcvbufErrors in all 210 runs); the chain kept up (C++: 0.13, 1.4 and 2.2 s a ping against 3 s); the worst receive
# delay of a run was 1.2 to 99 ms, against 0.8 to 3.5 ms for flank_32 on the same machines. So what holds here is asserted:
# losses only as counted refusals of the socket buffer, every ping accounted for, nothing malformed, the chain in real
# time, and the reports of the pings that came through equal to the same pings straight through the chain. A run that
# loses nothing passes; one that loses frames ends as an expected failure (xfail) with its numbers, so the loss stays
# visible in the summary. A deployed host needs what Sensor Bridge's own setup guide asks (DEPLOYMENT.md).
POPOTO_START_S = 0.25          # of stream after the receiver connects in which start-up loss is expected (D-187); LEAD is 1 s


def popoto_startup_loss(tmp_path, tag, start_frames):
    """(frame numbers that never arrived, worst receive delay in ms over the frames from `start_frames` on) of a
    loopback, from its send and arrival logs."""
    send = {row[0]: row[1] for row in json.loads((tmp_path / f"{tag}_send.json").read_text())["rows"]}
    arrived = {row[0]: row[1] for row in json.loads((tmp_path / f"{tag}_arrivals.json").read_text())["rows"]}
    missing = [k for k in range(0, max(arrived) + 1) if k not in arrived]
    after = [(arrived[k] - send[k]) * 1e-6 for k in arrived if k in send and k >= start_frames]
    return missing, (max(after) if after else float("nan"))


POPOTO_ENGINES = [(1, "cpp"), (2, "cpp"), (3, "cpp"), (1, "python"), (2, "python")]
# Not (3, "python"): the Python engine's 2.5-2.9 s a ping and its per-frame work at 256 frames a second of 115 kB
# together fall behind the stream, and the receiver's queue drops frames beyond the socket's losses (493 to 721 frames
# in a run against about 11 refused at the socket; job 6b063a21d0582cfbbefd60a3). The vehicle runs the C++ engine.


@pytest.mark.parametrize("rows,engine", POPOTO_ENGINES)
def test_loopback_at_a_popoto_builds_channel_count_and_rate_gives_the_direct_chains_reports(tmp_path, rows, engine):
    """sonar-x8v: a Popoto build's trial through Sensor Bridge's emulator and Linux receiver at real time, at its
    build's 32, 64 or 96 channels and 102.4 kHz, received with its deployment file: the chain keeps up, every ping is
    accounted for, any loss is the socket buffer's and counted, and the reported pings give the direct chain's reports.
    Frames lost make the test an expected failure that says how many (see above: so far every 64- and 96-channel run)."""
    reason = sensor_bridge_reason()
    if reason:
        pytest.skip(reason)
    records, cfg = popoto_trial(rows)
    n = len(records)
    _, stats, receive, report, m = popoto_loopback(tmp_path, rows, engine, tag=f"popoto{rows}", check=False)
    print("POPOTO", json.dumps(m))
    assert receive.returncode in (0, 3), receive.stderr[-3000:]     # 3: fewer pings reported than sent, said on stderr
    assert m["chain_omp_threads"] in (m["omp"], "no OpenMP runtime")
    assert stats["stopped"] == "pings" and stats["bad_frames"] == 0
    assert stats["pings"] + stats["dropped_pings"] + stats["pings_discarded_at_stop"] == n
    assert (receive.returncode == 3) == (stats["reports"] < n) and stats["reports"] == stats["pings"]
    # Every datagram lost was refused by the receiver's full socket buffer (the kernel's count for that socket).
    assert stats["socket"]["found"] and stats["socket"]["drops"] == stats["udp_counters"]["RcvbufErrors"]
    lossless = stats["frames_lost"] == 0 and stats["frames_damaged"] == 0 and stats["socket"]["drops"] == 0
    assert lossless or stats["socket"]["drops"] > 0                  # a loss with no refused datagram is another fault
    assert -1.0 < m["least_ms"] and m["median_ms"] < 2.0               # the two clocks agree (receive_delays)
    # Real time: the chain takes less than a ping interval on every ping after the first (which pays for set-up).
    assert max(m["chain_s"][1:]) < m["interval_s"], m["chain_s"]
    got = lines(report)
    assert len(got) == stats["reports"] and got, "no ping came through"
    direct, _ = popoto_direct(records, cfg, engine, sequences=[r["sequence"] for r in got], recording=str(tmp_path / "direct.h5"))
    assert len(direct) == len(got)
    assert all(r["sync"]["ok"] for r in got if "sync" in r)
    assert_same_reports(got, direct)
    # What `receive` recorded is what the same chain records of the same pings without transport. (Not the assembled
    # samples themselves: with a Popoto deployment the recording differs from them by float32 rounding, 2e-9 to 7e-9 V
    # measured, transport or not.)
    raw = list(gs.sources.make_source({"type": "container", "path": str(tmp_path / f"popoto{rows}.h5")}))
    ref = list(gs.sources.make_source({"type": "container", "path": str(tmp_path / "direct.h5")}))
    assert len(raw) == len(ref) == len(got)
    assert all(np.array_equal(a.samples.array, b.samples.array) for a, b in zip(raw, ref))
    # D-187: where the loss is. In every run located (L4 and a10g-large, 52 runs) the frames lost were the first ones
    # after the receiver connected, before the first ping's window: Sensor Bridge's receiver thread is not yet reading
    # while it allocates its queue. Such a run passes if, after that start, nothing is lost and the delay bound holds.
    missing, worst_after = popoto_startup_loss(tmp_path, f"popoto{rows}", POPOTO_START_S * m["frames_per_s"])
    print("POPOTO_START", json.dumps({"rows": rows, "engine": engine, "frames_lost": stats["frames_lost"], "missing": missing[:32],
                                      "worst_ms_after_start": worst_after, "worst_ms": m["worst_ms"]}))
    only_at_start = bool(missing) and max(missing) < POPOTO_START_S * m["frames_per_s"] and stats["frames_damaged"] == 0 \
        and len(missing) == stats["frames_lost"] and stats["reports"] == n
    if only_at_start and worst_after < 5.0:
        return
    if not lossless or m["worst_ms"] >= 5.0:
        pytest.xfail(f"measured limit (D-186): {rows * 32} channels at 102.4 kHz lost {stats['frames_lost']} frames "
                     f"({stats['frames_damaged']} damaged, {stats['socket']['drops']} datagrams refused by a "
                     f"{stats['socket']['rcvbuf']}-byte socket buffer), {stats['dropped_pings']} of {n} pings; worst receive "
                     f"delay {m['worst_ms']:.1f} ms (flank_32's bound 5 ms)")


def popoto_loss_location(tmp_path, tag, rows, frame=FRAME, width=4):
    """sonar-bep: where in a ping's cycle a Popoto-rate loopback loses frames, from the sender's and the receiver's
    per-frame logs and the chain's run times. A frame's `phase_s` is its place in the stream after the ping's
    transmit (the window is `window_s`). For every frame number missing from the arrival log: the phase, whether the
    board sent it while the chain ran on a ping (and how long into that run), and the sender's lateness, gap and send
    time over the frames around it. Over the whole run, per second of phase: frames missing, and the worst sender
    lateness, sender gap, send-call time and receive delay."""
    records, _ = popoto_trial(rows)
    n = stream_numbers(records, frame)
    period = 1.0 / n["frames_per_s"]
    interval = (records[1].transmit.time_ns - records[0].transmit.time_ns) * 1e-9
    r0 = records[0]
    window = [float(r0.samples.start_time), float(r0.samples.start_time) + r0.samples.array.shape[0] / n["sample_rate"]]
    send_log = json.loads((tmp_path / f"{tag}_send.json").read_text())
    send = {row[0]: row for row in send_log["rows"]}
    log = json.loads((tmp_path / f"{tag}_arrivals.json").read_text())
    arrived = {row[0]: row for row in log["rows"]}
    chain = log["chain_wall_ns"]
    first, last = min(arrived), max(arrived)
    missing = [k for k in range(first, last + 1) if k not in arrived]
    phase = lambda k: (k * period - LEAD) % interval                  # noqa: E731
    bins = {}
    for k in range(first, last + 1):
        b = bins.setdefault(int(phase(k)), {"frames": 0, "missing": 0, "late_ms": 0.0, "gap_ms": 0.0, "send_ms": 0.0, "delay_ms": 0.0})
        b["frames"] += 1
        b["missing"] += k not in arrived
        if k in send:
            b["late_ms"], b["gap_ms"] = max(b["late_ms"], send[k][2]), max(b["gap_ms"], send[k][3])
            b["send_ms"] = max(b["send_ms"], send[k][4])
            if k in arrived:
                b["delay_ms"] = max(b["delay_ms"], round((arrived[k][1] - send[k][1]) * 1e-6, 3))
    detail = []
    for k in missing[:24]:
        t = send[k][1] if k in send else None
        into = [(i, round((t - a) * 1e-6, 1)) for i, (a, b) in enumerate(chain) if t is not None and a <= t <= b]
        since = [round((t - b) * 1e-6, 1) for a, b in chain if t is not None and t > b]
        around = range(k - width, k + width + 1)
        detail.append({"frame": k, "phase_s": round(phase(k), 4), "chain_run_and_ms_into_it": into[0] if into else None,
                       "ms_after_last_chain_run": since[-1] if since and not into else None,
                       "late_ms": [send[j][2] if j in send else None for j in around],
                       "gap_ms": [send[j][3] if j in send else None for j in around],
                       "send_ms": [send[j][4] if j in send else None for j in around],
                       "delay_ms": [round((arrived[j][1] - send[j][1]) * 1e-6, 2) if j in arrived and j in send else None for j in around]})
    starts = [round(((a - send[first + 1][1]) * 1e-9 + (first + 1) * period - LEAD) % interval, 3) for a, _ in chain] if first + 1 in send else []
    return {"period_ms": 1e3 * period, "interval_s": interval, "window_s": window, "frames": last - first + 1,
            "missing": len(missing), "missing_phase_s": [round(phase(k), 3) for k in missing[:64]],
            "chain_start_phase_s": starts, "chain_s": [round((b - a) * 1e-9, 3) for a, b in chain],
            "missing_frames": missing[:64], "by_second_of_phase": {str(s): bins[s] for s in sorted(bins)}, "detail": detail[:6]}


def test_the_loss_location_places_missing_frames_in_the_pings_cycle(tmp_path):
    """Without Sensor Bridge: from made-up logs of a one-row run in which frames 300 and 301 never arrived and the
    board sent them 0.2 s into the chain's run on ping 0, the study's reading says so."""
    records, _ = popoto_trial(1)
    period = 1.0 / stream_numbers(records)["frames_per_s"]
    t0 = 1_700_000_000_000_000_000
    wall = lambda k: t0 + int(round(k * period * 1e9))                # noqa: E731
    rows = [[k, wall(k), 0.01, 1e3 * period, 0.3] for k in range(1, 1000)]
    rows[299][2:] = [9.0, 12.9, 0.4]                                   # frame 300: 9 ms late
    (tmp_path / "t_send.json").write_text(json.dumps({"rows": rows}))
    got = [[k, wall(k) + 300_000, 28, 38528] for k in range(0, 1000) if k not in (300, 301)]
    chain = [[wall(300) - 200_000_000, wall(300) + 100_000_000]]
    (tmp_path / "t_arrivals.json").write_text(json.dumps({"rows": got, "chain_wall_ns": chain}))
    w = popoto_loss_location(tmp_path, "t", 1)
    assert w["missing"] == 2 and [d["frame"] for d in w["detail"]] == [300, 301]
    assert w["detail"][0]["phase_s"] == pytest.approx((300 * period - LEAD) % w["interval_s"], abs=1e-4)
    assert w["detail"][0]["chain_run_and_ms_into_it"] == [0, 200.0] or w["detail"][0]["chain_run_and_ms_into_it"] == (0, 200.0)
    assert w["detail"][0]["late_ms"][4] == 9.0 and w["detail"][0]["delay_ms"][3] == pytest.approx(0.3)
    assert sum(b["missing"] for b in w["by_second_of_phase"].values()) == 2
    assert sum(b["frames"] for b in w["by_second_of_phase"].values()) == 1000
    assert popoto_startup_loss(tmp_path, "t", 64) == ([300, 301], pytest.approx(0.3))
    assert w["chain_s"] == [0.3] and w["interval_s"] == pytest.approx(3.0)


@pytest.mark.skipif(os.environ.get("GPUSONAR_HSB_STUDY") != "1", reason="a measurement, run with GPUSONAR_HSB_STUDY=1")
def test_popoto_rate_loopback_study(tmp_path):
    """sonar-x8v: Popoto-rate loopbacks repeated, one line per run (POPOTO_STUDY {...}). Measures; asserts nothing
    about the transport. GPUSONAR_HSB_STUDY_RUNS (10), _ROWS ("1,2,3"), _ENGINES ("cpp"), _CONFIGS (space-separated
    frame:buffer:queue_s[:omp[:catch_up]], buffer "def" or a --socket-buffer in bytes, omp "half" or a thread count,
    catch_up "spread" or "burst"; default "400:def:4"), _BUDGET_S (seconds after which no run starts; 5400)."""
    reason = sensor_bridge_reason()
    if reason:
        pytest.skip(reason)
    runs = int(os.environ.get("GPUSONAR_HSB_STUDY_RUNS", "10"))
    rows_list = [int(x) for x in os.environ.get("GPUSONAR_HSB_STUDY_ROWS", "1,2,3").split(",")]
    engines = os.environ.get("GPUSONAR_HSB_STUDY_ENGINES", "cpp").split(",")
    configs = os.environ.get("GPUSONAR_HSB_STUDY_CONFIGS", "400:def:4").split()
    budget = float(os.environ.get("GPUSONAR_HSB_STUDY_BUDGET_S", "5400"))
    print("cpus: affinity", len(os.sched_getaffinity(0)), "usable", usable_cpus(), "rmem",
          {k: pathlib.Path(f"/proc/sys/net/core/{k}").read_text().strip() for k in ("rmem_default", "rmem_max")}, flush=True)
    for rows in rows_list:
        popoto_trial(rows)                                             # simulated before the clock starts
    t0 = time.time()
    for k in range(runs):
        for cfg_text in configs:
            frame, buffer, queue_s, *more = cfg_text.split(":")
            omp = None if not more or more[0] == "half" else int(more[0])
            catch = ("--catch-up", more[1]) if len(more) > 1 else ()
            for rows in rows_list:
                for engine in engines:
                    if time.time() - t0 > budget:
                        print("POPOTO_STUDY budget reached", flush=True)
                        return
                    tag = f"s{k}_{cfg_text.replace(':', '_')}_{rows}_{engine}"
                    _, _, _, _, m = popoto_loopback(tmp_path, rows, engine, tag, frame=int(frame),
                                                    socket_buffer=None if buffer == "def" else int(buffer), queue_s=float(queue_s),
                                                    check=False, omp=omp, serve_args=catch)
                    m.update(run=k, config=cfg_text)
                    print("POPOTO_STUDY " + json.dumps(m), flush=True)
                    try:
                        where = popoto_loss_location(tmp_path, tag, rows, int(frame))
                        print("POPOTO_LOSS " + json.dumps(dict(where, run=k, config=cfg_text, rows=rows, engine=engine)), flush=True)
                    except (OSError, ValueError, KeyError, IndexError) as exc:
                        print(f"POPOTO_LOSS failed {type(exc).__name__}: {exc}", flush=True)
                    for f in tmp_path.glob(f"{tag}*.h5"):
                        f.unlink()                                     # the disk: recordings are not needed here


def loss_around(tmp_path, tag, stats, board, width=3):
    """For each frame the receiver missed or got damaged: the sender's lateness and gap before it, and the
    arrival times of the frames around it (ms after the frame before)."""
    send = {row[0]: row for row in json.loads((tmp_path / f"{tag}_send.json").read_text())["rows"]}
    log = json.loads((tmp_path / f"{tag}_arrivals.json").read_text())
    arrived = {row[0]: row for row in log["rows"]}
    chain = log["chain_wall_ns"]
    out = []
    for k in stats["trouble_frames"]:
        frames = range(k - width, k + width + 1)
        t = send[k][1] if k in send else None
        during = [i for i, (a, b) in enumerate(chain) if t is not None and a <= t <= b]
        after = [round((t - b) * 1e-6, 1) for a, b in chain if t is not None and t > b]
        out.append({"frame": k, "sent_during_chain_run_of_ping": during, "ms_after_last_chain_run": after[-1] if after else None,
                    "chain_runs_ms": [round((b - a) * 1e-6, 1) for a, b in chain],
                    "sender_late_ms": [send[j][2] if j in send else None for j in frames],
                    "sender_gap_ms": [send[j][3] if j in send else None for j in frames],
                    "arrived": [j in arrived for j in frames],
                    "packets": [arrived[j][2] if j in arrived else None for j in frames]})
    return out


@pytest.mark.skipif(os.environ.get("GPUSONAR_HSB_STUDY") != "1", reason="a measurement, run with GPUSONAR_HSB_STUDY=1")
def test_loopback_loss_study(pings, tmp_path):
    """Why real time loses datagrams (STATUS open item 34). Each run prints what the sender, the
    receiver's socket and the kernel saw. Measures, asserts nothing about the transport."""
    reason = sensor_bridge_reason()
    if reason:
        pytest.skip(reason)
    import shutil
    cores = sorted(os.sched_getaffinity(0))
    receiver_core = 2 if 2 in cores else cores[-1]
    away = ",".join(str(c) for c in cores if c != receiver_core)
    print("cores:", cores, "cpu_count:", os.cpu_count(), "cpu:", hsb.cpu_throttling(), "rmem:",
          {k: pathlib.Path(f"/proc/sys/net/core/{k}").read_text().strip() for k in ("rmem_default", "rmem_max")})
    one = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    burst = ("--catch-up", "burst")
    # The kernel doubles a buffer it is asked for (socket(7)): 53248 asked is 106496, half the default.
    half = ("--socket-buffer", "53248")
    runs = [*[("pace 1.0", 1.0, (), (), ())] * 6,
            ("pace 1.0, catch-up burst (before)", 1.0, burst, (), ()),
            ("pace 1.0, catch-up burst (before)", 1.0, burst, (), ()),
            ("pace 1.0, catch-up burst (before)", 1.0, burst, (), ()),
            ("pace 1.0, receive buffer halved", 1.0, (), half, ()),
            ("pace 1.0, receive buffer halved", 1.0, (), half, ()),
            ("pace 1.0, chain on one thread, catch-up burst", 1.0, burst, (), (), one),
            ("pace 1.0, chain on one thread, catch-up burst", 1.0, burst, (), (), one),
            ("pace 1.0, sender off the receiver's core", 1.0, (), ("--receiver-core", str(receiver_core)),
             ("taskset", "-c", away) if shutil.which("taskset") and away else ())]
    table = []
    for i, (name, pace, serve_args, receive_args, prefix, *receive_env) in enumerate(runs):
        tag = f"study{i}"
        board, stats, _, _ = run_loopback(tmp_path, pings, pace, serve_args, receive_args, prefix, tag=tag,
                                          receive_env=receive_env[0] if receive_env else None)
        row = {"run": name, "frames": stats["frames"], "lost": stats["frames_lost"], "damaged": stats["frames_damaged"],
               "datagrams_dropped": stats["packets_dropped"], "pings": stats["pings"], "socket": stats["socket"],
               "udp": stats["udp_counters"], "cpu_throttled_during": stats["cpu"].get("nr_throttled_during"),
               "sender": board["timing"], "around_losses": loss_around(tmp_path, tag, stats, board)}
        print("STUDY", json.dumps(row))
        table.append(row)
    print("STUDY SUMMARY")
    for r in table:
        t = r["sender"]
        print(f"  {r['run']:42s} lost {r['lost']} damaged {r['damaged']} datagrams {r['datagrams_dropped']:3d} "
              f"socket drops {r['socket']['drops']} rcvbuf {r['socket']['rcvbuf']} RcvbufErrors {r['udp'].get('RcvbufErrors')} "
              f"sender late>period {t.get('frames_late_by_a_period')} back-to-back {t.get('back_to_back_frames')} "
              f"max late {t.get('max_late_ms', 0):.1f} ms max gc {t.get('max_gc_ms', 0):.1f} ms")
