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


def test_operators_in_a_holoscan_application_behind_a_stand_in_receiver(pings):
    """HsbFramesToPingsOp and LiveChainOp inside the Holoscan runtime, fed by an operator that emits
    what the receiver operator does (a map with one unnamed uint8 tensor, metadata beside it) from
    host memory. Everything of the Sensor Bridge path except Sensor Bridge."""
    if hsb.Operator is None:
        pytest.skip("Holoscan SDK not usable here")
    import holoscan
    from holoscan.conditions import CountCondition
    frames = list(hsb.frames_from_records(copies(pings), FRAME, lead_seconds=LEAD))
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

    with LiveChain(CHAIN) as chain:
        direct = chain.run(hsb.records_from_frames(frames, "flank_32", OVERRIDES))
    with LiveChain(CHAIN) as chain:
        app = App(chain)
        app.run()
    got, stats = app.chain_op.reports, app.receiver.stats()
    print("stand-in receiver:", stats, "metadata seen:", app.frames_op.last_metadata)
    assert stats["frames"] == len(frames) and stats["pings"] == N and stats["frames_lost"] == 0
    assert app.frames_op.last_metadata.get("frame_number") == len(frames) - 1     # the metadata travels with the frame
    strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}   # noqa: E731
    assert [strip(r) for r in got] == [strip(r) for r in direct]                  # the same records, so the same reports
    # A frame that never arrives costs the ping it cuts and nothing else.
    with LiveChain(CHAIN) as chain:
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


def run_loopback(tmp_path, pings, pace, serve_args=(), receive_args=(), serve_prefix=(), tag="run", receive_env=None):
    """One loopback: `serve` (the board, through Sensor Bridge's emulator) and `receive` (Sensor
    Bridge's LinuxReceiverOp -> frames to pings -> chain) as separate processes, as deployed.
    Returns (sender's statistics, receiver's statistics, the receive process, its report path)."""
    w = gs.RecordWriter(str(tmp_path / "pings.h5"))
    for r in copies(pings):
        w.write(r)
    w.close()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "python"), os.environ.get("PYTHONPATH", "")]))
    serve = subprocess.Popen([*serve_prefix, sys.executable, str(TOOL), "serve", "--replay", str(tmp_path / "pings.h5"),
                              "--frame-samples", str(FRAME), "--lead", str(LEAD), "--pace", str(pace), "--linger", "20",
                              "--send-log", str(tmp_path / f"{tag}_send.json"), *serve_args],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        receive = subprocess.run([sys.executable, str(TOOL), "receive", "--sonar", "flank_32", "--sonar-overrides", json.dumps(OVERRIDES),
                                  "--chain", json.dumps(CHAIN), "--frame-samples", str(FRAME), "--pings", str(N),
                                  "--report", str(tmp_path / f"{tag}.jsonl"), "--recording", str(tmp_path / f"{tag}.h5"),
                                  "--arrival-log", str(tmp_path / f"{tag}_arrivals.json"), *receive_args],
                                 capture_output=True, text=True, timeout=600, env=dict(env, **(receive_env or {})))
        sent, _ = serve.communicate(timeout=120)
    finally:
        serve.kill()
    print(sent[-1500:])
    print(receive.stdout[-3000:])
    assert receive.returncode == 0, receive.stderr[-3000:]
    assert serve.returncode == 0, sent[-3000:]
    board = json.loads([ln for ln in sent.splitlines() if ln.startswith("hsb: ")][-1][5:])
    stats = json.loads([ln for ln in receive.stdout.splitlines() if ln.startswith("hsb: ")][-1][5:])
    return board, stats, receive, tmp_path / f"{tag}.jsonl"


def test_loopback_through_sensor_bridge_gives_the_direct_chains_reports(pings, tmp_path):
    reason = sensor_bridge_reason()
    if reason:
        pytest.skip(reason)
    # Half of real time (GPUSONAR_HSB_PACE overrides). At real time the receiver's socket buffer (212,992
    # bytes, Hugging Face's rmem_max: about 20 ms of the stream) overflowed while the chain ran on a ping
    # (test_loopback_loss_study; hsb.py FRAME_SAMPLES says what was measured).
    pace = float(os.environ.get("GPUSONAR_HSB_PACE", "0.5"))
    board, stats, receive, report = run_loopback(tmp_path, pings, pace, tag="hsb")
    print("sender:", board)
    print("transport:", stats)
    assert stats["pings"] == N and stats["frames_lost"] == 0 and stats["frames_damaged"] == 0 and stats["dropped_pings"] == 0
    assert stats["bad_frames"] == 0 and stats["lost_samples"] == 0
    # The same pings straight through the chain: framed and assembled here, no transport.
    direct_records = hsb.records_from_frames(hsb.frames_from_records(copies(pings), FRAME, lead_seconds=LEAD), "flank_32", OVERRIDES)
    sent_samples = direct_records[0].samples.array.copy()            # the chain conditions records in place
    with LiveChain(CHAIN) as chain:
        direct = chain.run(direct_records)
    got = lines(report)
    assert len(got) == len(direct) == N
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
    assert worst["track_m"] < 0.05 and worst["range_m"] < 0.05 and worst["bearing_deg"] < 0.05
    # Two confirmed tracks. The stream carries no navigation, so the records sit at the origin and the
    # standing target takes ownship's speed (test_live.py shows the same without navigation).
    assert [t["kind"] for t in got[-1]["tracks"]] == ["mover", "mover"]
    # What was recorded on the way in is what the board sent.
    raw = list(gs.sources.make_source({"type": "container", "path": str(tmp_path / "hsb.h5")}))
    assert len(raw) == N and np.array_equal(raw[0].samples.array, sent_samples)


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
