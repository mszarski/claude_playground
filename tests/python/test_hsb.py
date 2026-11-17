"""The flank stream through NVIDIA Holoscan Sensor Bridge (gpusonar/sources/hsb.py): framing and
frame handling everywhere; the loopback through Sensor Bridge's emulator and its Linux receiver
operator where Sensor Bridge is built (tools/build_hsb.sh) and a CUDA device exists."""

import json
import os
import pathlib
import subprocess
import sys

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


def test_loopback_through_sensor_bridge_gives_the_direct_chains_reports(pings, tmp_path):
    reason = sensor_bridge_reason()
    if reason:
        pytest.skip(reason)
    # Half of real time: the receiver's socket buffer (not ours to size, and not raisable in a container)
    # holds about 10 ms of the stream at full rate, and one late wake-up of its thread then costs packets.
    pace = float(os.environ.get("GPUSONAR_HSB_PACE", "0.5"))
    w = gs.RecordWriter(str(tmp_path / "pings.h5"))
    for r in copies(pings):
        w.write(r)
    w.close()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "python"), os.environ.get("PYTHONPATH", "")]))
    serve = subprocess.Popen([sys.executable, str(TOOL), "serve", "--replay", str(tmp_path / "pings.h5"), "--frame-samples", str(FRAME),
                              "--lead", str(LEAD), "--pace", str(pace), "--linger", "20"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        receive = subprocess.run([sys.executable, str(TOOL), "receive", "--sonar", "flank_32", "--sonar-overrides", json.dumps(OVERRIDES),
                                  "--chain", json.dumps(CHAIN), "--frame-samples", str(FRAME), "--pings", str(N),
                                  "--report", str(tmp_path / "hsb.jsonl"), "--recording", str(tmp_path / "hsb.h5")],
                                 capture_output=True, text=True, timeout=600, env=env)
        sent, _ = serve.communicate(timeout=120)
    finally:
        serve.kill()
    print(sent[-1500:])
    print(receive.stdout[-3000:])
    assert receive.returncode == 0, receive.stderr[-3000:]
    assert serve.returncode == 0, sent[-3000:]
    stats = json.loads([ln for ln in receive.stdout.splitlines() if ln.startswith("hsb: ")][-1][5:])
    print("transport:", stats)
    assert stats["pings"] == N and stats["frames_lost"] == 0 and stats["frames_damaged"] == 0 and stats["dropped_pings"] == 0
    assert stats["bad_frames"] == 0 and stats["lost_samples"] == 0
    # The same pings straight through the chain: framed and assembled here, no transport.
    direct_records = hsb.records_from_frames(hsb.frames_from_records(copies(pings), FRAME, lead_seconds=LEAD), "flank_32", OVERRIDES)
    with LiveChain(CHAIN) as chain:
        direct = chain.run(direct_records)
    got = lines(tmp_path / "hsb.jsonl")
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
    assert {t["kind"] for t in got[-1]["tracks"]} >= {"mover", "static"}
    # What was recorded on the way in is what the board sent.
    raw = list(gs.sources.make_source({"type": "container", "path": str(tmp_path / "hsb.h5")}))
    assert len(raw) == N and np.array_equal(raw[0].samples.array, direct_records[0].samples.array)
