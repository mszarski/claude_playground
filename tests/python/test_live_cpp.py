"""The C++ live library against the Python stages it replaces, and the standalone program."""

import json
import pathlib
import subprocess

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import classify, detect
from gpusonar.live import LiveChain
from gpusonar.sources import FlankBoardEmulator, PingAssembler, make_source
from test_live import CHAIN, INTERVAL, N, OVERRIDES, copies, pings  # noqa: F401  (pings is a fixture)

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "bin" / "sonar_live"
CPP_CHAIN = {"bearing_first_deg": -40.0, "bearing_last_deg": 40.0, "bearing_step_deg": 1.0, "range_cell_m": 1.0, "cfar_pfa": 1e-6,
             "cfar_guard_m": 15.0, "cfar_train_m": 50.0, "min_snr_db": 12.0, "tracker_range_sigma": 3.0,
             "tracker_bearing_sigma": np.radians(1.0), "tracker_confirm_hits": 3, "tracker_max_misses": 2,
             "classify_min_speed": 1.0, "classify_static_speed": 0.6, "classify_max_fit_rms": 10.0, "multipath": False}
EXE_OPTIONS = ["bearing_first_deg=-40", "bearing_last_deg=40", "bearing_step_deg=1", "range_cell_m=1", "cfar_pfa=1e-6",
               "cfar_guard_m=15", "cfar_train_m=50", "min_snr_db=12", "tracker_range_sigma=3", "tracker_bearing_sigma_deg=1",
               "tracker_confirm_hits=3", "tracker_max_misses=2", "classify_max_fit_rms=10", "multipath=0"]


def test_cluster_matches_python():
    rng = np.random.default_rng(3)
    power = rng.exponential(1.0, (40, 300)).astype(np.float32)
    power[10:13, 100:104] += 60.0
    power[30, 250] += 80.0
    ranges, bearings = np.linspace(100.0, 400.0, 300), np.radians(np.linspace(-40.0, 40.0, 40))
    det, ratio = core.cfar(power, pfa=1e-3, guard=4, train=20, return_ratio=True)
    want = detect.cluster(det, power, ranges, bearings, guard=4, train=20)
    got = core.live_cluster(det, power, ratio, list(ranges), list(bearings))
    assert len(got) == len(want) > 5
    for g, w in zip(got, want):
        assert g[0] == pytest.approx(w.range, abs=1e-4) and g[1] == pytest.approx(w.bearing, abs=1e-7)   # float32 sums
        assert g[3] == w.cells and g[2] == pytest.approx(w.snr_db, abs=0.05)


def test_tracker_and_classifier_match_python():
    rng = np.random.default_rng(8)
    opts = dict(process_noise=0.3, range_sigma=1.5, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0)
    py_tracker, cpp_tracker = detect.Tracker(**opts), core.LiveTracker(**opts)
    # Three movers, a row of five sliding along a line, clutter that comes and goes.
    for i in range(9):
        t = 0.5 * i
        truth = [(200.0 + 4.0 * t, 30.0), (150.0, -40.0 + 3.0 * t), (260.0 - 2.0 * t, 80.0 - 2.0 * t)]
        truth += [(120.0 + 12.0 * k + 2.0 * t, -90.0) for k in range(5)]
        truth += [(300.0, -20.0), (180.0, 100.0)]
        dets = []
        for x, y in truth:
            x, y = x + 0.3 * rng.standard_normal(), y + 0.3 * rng.standard_normal()
            dets.append((float(np.hypot(x, y)), float(np.arctan2(y, x)), float(15.0 + 5.0 * rng.random()), 2))
        for _ in range(3):
            if rng.random() < 0.5:
                dets.append((float(rng.uniform(100, 350)), float(rng.uniform(-0.6, 0.6)), 13.0, 1))
        a = py_tracker.update(t, [detect.Detection(*d) for d in dets])
        b = cpp_tracker.update(t, dets)
        assert [tr.id for tr in a] == [tr["id"] for tr in b]
        for tr, cr in zip(a, b):
            assert np.allclose(tr.state, cr["state"], atol=1e-8) and tr.hits == cr["hits"] and tr.misses == cr["misses"]
            assert np.allclose(np.array(tr.history), np.array(cr["history"]), atol=1e-8) and np.allclose(tr.snr, cr["snr"])
    kw = dict(min_speed=0.5, max_fit_rms=2.5)
    want = classify.classify_tracks(a, linear={"length": 40.0, "width": 2.0}, **kw)
    got = cpp_tracker.classify(linear=True, linear_length=40.0, linear_width=2.0, **kw)
    assert [c.kind for c in want] == [c["kind"] for c in got]
    assert {"mover", "static", "line"} <= {c.kind for c in want}          # the scene exercises each rule
    for w, g in zip(want, got):
        assert np.allclose(w.position, g["position"], atol=1e-8) and np.allclose(w.velocity, g["velocity"], atol=1e-8)
        assert w.speed_sigma == pytest.approx(g["speed_sigma"], rel=1e-8) and w.seen == pytest.approx(g["seen"])
    mp = {"max_excess_range": 60.0, "bearing": 0.2, "velocity": 3.0, "origin": (0.0, 0.0)}
    want = classify.classify_tracks(a, multipath=mp, **kw)
    got = cpp_tracker.classify(multipath=True, max_excess_range=60.0, multipath_bearing=0.2, multipath_velocity=3.0, **kw)
    assert [(c.kind, c.parent) for c in want] == [(c["kind"], c["parent"]) for c in got]


def test_cpp_chain_gives_the_python_chains_report(pings):
    with LiveChain(CHAIN) as chain:
        want = chain.run(copies(pings))
    cpp = core.LiveChainCpp(CPP_CHAIN)
    got = [json.loads(cpp.process(r)) for r in copies(pings)]
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"])
        for a, b in zip(w["detections"], g["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
        assert [(t["id"], t["kind"], t["parent"]) for t in w["tracks"]] == [(t["id"], t["kind"], t["parent"]) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "range_m"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key
    assert any(t["kind"] == "mover" for t in got[-1]["tracks"]) and any(t["kind"] == "static" for t in got[-1]["tracks"])
    power, ranges, bearings = cpp.last_map()
    assert power.shape == (len(bearings), len(ranges)) and power.max() > 0


def test_cpp_assembler_matches_python(pings):
    emulator = FlankBoardEmulator(pings[:3], INTERVAL, block_samples=128, start_ns=10**18, gain_code=3)
    blocks = list(emulator.blocks())
    template = pings[0].copy()
    py_asm = PingAssembler(gs.sonar.SonarModel.load("flank_32", OVERRIDES))
    cpp_asm = core.PingAssemblerCpp(template)
    a, b = [], []
    for i, block in enumerate(blocks):
        if i == len(blocks) // 2:
            continue                                               # one lost block: both must drop the same ping
        a += py_asm.feed(block)
        b += cpp_asm.feed(block)
    assert len(a) == len(b) == 2 and cpp_asm.dropped_pings == py_asm.dropped_pings == 1
    assert cpp_asm.lost_samples == py_asm.lost_samples == 128
    for x, y in zip(a, b):
        assert np.array_equal(x.samples.array, y.samples.array)
        assert x.transmit.time_ns == y.transmit.time_ns and x.metadata.sequence == y.metadata.sequence
        assert x.samples.start_time == pytest.approx(y.samples.start_time)
    assert cpp_asm.feed(b"not a block") == [] and cpp_asm.bad_blocks == 1


# --- the standalone program ------------------------------------------------------------------

needs_exe = pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")


@needs_exe
def test_program_replays_a_recording_like_the_python_chain(pings, tmp_path):
    with LiveChain(CHAIN, recording=str(tmp_path / "rec.h5")) as chain:
        want = chain.run(copies(pings))
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), *EXE_OPTIONS,
                          "classify_min_speed=1"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    got = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    assert len(got) == N
    movers = [t for t in got[-1]["tracks"] if t["kind"] == "mover"]
    want_movers = [t for t in want[-1]["tracks"] if t["kind"] == "mover"]
    assert len(movers) == len(want_movers) == 1
    assert movers[0]["x"] == pytest.approx(want_movers[0]["x"], abs=0.1) and movers[0]["vx"] == pytest.approx(-3.0, abs=0.6)


@needs_exe
def test_program_takes_blocks_over_udp_with_navigation(pings, tmp_path):
    import socket
    import time
    template = pings[0].copy()
    w = gs.RecordWriter(str(tmp_path / "template.h5"))
    w.write(template)
    w.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    nav_port = port + 1
    proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", str(tmp_path / "template.h5"), "--nav-port", str(nav_port),
                             "--recording", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), "--pings", str(N),
                             *EXE_OPTIONS], stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(N + 1):                                     # navigation first: 1.5 m/s north
            t = i * INTERVAL
            tx.sendto(json.dumps({"time_ns": int(t * 1e9), "x": 1.5 * t, "y": 0.0, "z": 0.0, "yaw": 0.0, "vx": 1.5}).encode(),
                      ("127.0.0.1", nav_port))
        sender = FlankBoardEmulator(pings, INTERVAL, block_samples=128).send(port, pace=0.25)
        sender.join(timeout=120.0)
        _, err = proc.communicate(timeout=120.0)
    finally:
        proc.kill()
    reports = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    if "0 dropped, 0 samples lost" not in err or len(reports) < N:
        pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    assert "0 pings without navigation" in err
    assert reports[-1]["ownship_xyz"][0] == pytest.approx(1.5 * (N - 1) * INTERVAL, abs=0.6)   # array centre, from navigation
    kinds = {t["kind"] for t in reports[-1]["tracks"]}
    assert "mover" in kinds and "static" in kinds
    raw = list(make_source({"type": "container", "path": str(tmp_path / "rec.h5")}))
    assert len(raw) == N and not np.iscomplexobj(raw[0].samples.array)   # recorded before conditioning


def test_cpp_sync_matches_python_and_focuses_the_four_board_array():
    from gpusonar import sync
    from test_calibration_sync import four_board_ping
    sonar, rec, target, truth = four_board_ping(bearing_deg=10.0)
    x = rec.samples.array
    s_ = rec.samples
    s_.array = x + 2e-7 * np.random.default_rng(1).standard_normal(x.shape)
    rec.samples = s_
    ref = sonar.config["receive"]["reference"]
    want = sync.synchronise(rec.copy(), ref)
    opts = {"bearing_first_deg": -30.0, "bearing_last_deg": 30.0, "bearing_step_deg": 0.5, "range_cell_m": 0.5, "cfar_pfa": 1e-6,
            "cfar_guard_m": 5.0, "cfar_train_m": 20.0, "min_snr_db": 12.0, "multipath": False,
            "sync": {k: ref[k] for k in ("f_start", "f_stop", "duration", "lead", "tail")}}
    chain = core.LiveChainCpp(opts)
    rep = json.loads(chain.process(rec.copy()))
    assert rep["sync"]["ok"] and rep["sync"]["worst_residual_us"] < 1.0
    assert np.allclose(rep["sync"]["offsets_us"], want.estimate.offset * 1e6, atol=0.05)       # 50 ns
    assert np.allclose(rep["sync"]["drift_ppm"], want.estimate.drift_ppm, atol=0.2)
    assert max(abs(v) for v in rep["sync"]["offsets_us"]) > 100.0
    power, ranges, bearings = chain.last_map()
    ranges = np.array(ranges)
    inside = (ranges > 520.0) & (ranges < 680.0)
    ib, ir = np.unravel_index(np.argmax(power[:, inside]), power[:, inside].shape)
    assert ranges[inside][ir] == pytest.approx(600.0, abs=1.5) and np.degrees(bearings[ib]) == pytest.approx(10.0, abs=1.0)
    assert all(d["range_m"] > 450.0 + 0.5 * 1500.0 * (ref["lead"] + ref["duration"]) for d in rep["detections"])
    unsynced = core.LiveChainCpp({k: v for k, v in opts.items() if k != "sync"})
    unsynced.process(rec.copy())
    assert unsynced.last_map()[0][:, inside].max() < 0.5 * power[:, inside].max()


@needs_exe
def test_program_resolves_bearings_with_sbl_and_listens_passively(tmp_path):
    from test_doa import R, active_ping, passive_record
    w = gs.RecordWriter(str(tmp_path / "active.h5"))
    w.write(active_ping([2.0, 4.0]))
    w.close()
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "active.h5"), "doa=sbl", "bearing_first_deg=-30", "bearing_last_deg=30",
                          "cfar_pfa=1e-6", "cfar_guard_m=20", "cfar_train_m=60", "min_snr_db=12", "sbl_min_snr_db=10",
                          "sbl_dynamic_range_db=15"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    rep = json.loads(out.stdout.splitlines()[0])
    got = sorted(d["bearing_deg"] for d in rep["detections"] if abs(d["range_m"] - R) < 10.0)
    assert len(got) == 2 and got[0] == pytest.approx(2.0, abs=0.4) and got[1] == pytest.approx(4.0, abs=0.4)
    w = gs.RecordWriter(str(tmp_path / "passive.h5"))
    w.write(passive_record([(-20.0, 1.0), (15.0, 0.5)], seconds=2.0))
    w.close()
    out = subprocess.run([str(EXE), "--passive", str(tmp_path / "passive.h5"), "--band", "4000", "6000", "n_fft=256", "hop=128",
                          "n_average=12", "bearing_first_deg=-60", "bearing_last_deg=60", "sbl_min_snr_db=3"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    frames = [json.loads(line) for line in out.stdout.splitlines()]
    assert len(frames) >= 7 and frames[1]["time_s"] > frames[0]["time_s"]
    good = sum(1 for f in frames if len(f["bearings_deg"]) == 2 and abs(min(f["bearings_deg"]) + 20.0) < 0.6
               and abs(max(f["bearings_deg"]) - 15.0) < 0.6)
    assert good >= 0.8 * len(frames)


def test_one_deployment_file_runs_either_engine(pings, tmp_path):
    """engine: cpp puts libs/live behind the same LiveChain: same report fields, same tracks, taps and recording."""
    with LiveChain(CHAIN) as chain:
        want = chain.run(copies(pings))
    with LiveChain(dict(CHAIN, engine="cpp"), recording=str(tmp_path / "rec.h5"), report_path=str(tmp_path / "rep.jsonl"),
                   taps={"dir": str(tmp_path / "taps"), "pings": 1}) as chain:
        got = chain.run(copies(pings))
        assert chain.last["power"].shape[0] == len(chain.bearings)
    assert set(want[-1]) == set(got[-1]) and set(want[-1]["tracks"][0]) == set(got[-1]["tracks"][0])
    for w, g in zip(want, got):
        assert g["ping"] == w["ping"]
        assert [(t["id"], t["kind"]) for t in w["tracks"]] == [(t["id"], t["kind"]) for t in g["tracks"]]
    assert len((tmp_path / "rep.jsonl").read_text().splitlines()) == N
    stages = {p.name.split(".")[0].split("_", 1)[-1] for p in (tmp_path / "taps" / "ping0000").iterdir()}
    assert stages >= {"raw", "prepare", "beamform", "report"}
    with pytest.raises(ValueError, match="backprojection"):
        LiveChain(dict(CHAIN, engine="cpp", beamformer="mvdr"))


def test_python_chain_takes_sbl_bearings():
    from test_doa import R, active_ping
    cfg = {"bearings_deg": [-30.0, 30.0, 0.5], "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 60.0}, "min_snr_db": 12.0,
           "backend": "host", "doa": "sbl", "sbl": {"min_snr_db": 10.0, "dynamic_range_db": 15.0}}
    with LiveChain(cfg) as chain:
        rep = chain.process(active_ping([2.0, 4.0]))
    got = sorted(d["bearing_deg"] for d in rep["detections"] if abs(d["range_m"] - R) < 10.0)
    assert len(got) == 2 and got[0] == pytest.approx(2.0, abs=0.4) and got[1] == pytest.approx(4.0, abs=0.4)


@needs_exe
def test_exported_template_and_command_run_the_program(pings, tmp_path):
    """tools/sonar_live.py --export-template writes what build/bin/sonar_live needs and prints its command."""
    import shlex
    import sys
    import yaml
    sys.path.insert(0, str(ROOT / "tools"))
    import sonar_live as tool
    cfg = {"source": {"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": OVERRIDES, "port": 5799},
           "chain": {k: v for k, v in CHAIN.items() if k != "backend"},
           "output": {"report": str(tmp_path / "rep.jsonl")}}
    (tmp_path / "deploy.yaml").write_text(yaml.safe_dump(cfg))
    assert tool.main([str(tmp_path / "deploy.yaml"), "--export-template", str(tmp_path / "template.h5")]) == 0
    command = shlex.split(tool.export_template(cfg, str(tmp_path / "template.h5")))
    assert command[:5] == ["build/bin/sonar_live", "--udp", "5799", "--template", str(tmp_path / "template.h5")]
    t = gs.RecordReader(str(tmp_path / "template.h5")).read(0)
    assert t.samples.array.shape == pings[0].samples.array.shape and t.samples.sample_rate == pings[0].samples.sample_rate
    assert np.allclose(t.rx_positions() - t.rx_positions().mean(axis=0), pings[0].rx_positions() - pings[0].rx_positions().mean(axis=0))
    # Every option the tool prints is one the program accepts, and gives the Python chain's tracks on a replay.
    w = gs.RecordWriter(str(tmp_path / "pings.h5"))
    for r in copies(pings):
        w.write(r)
    w.close()
    options = [c for c in command if "=" in c]
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "pings.h5"), *options], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    with LiveChain(CHAIN) as chain:
        want = chain.run(copies(pings))
    got = [json.loads(line) for line in out.stdout.splitlines()]
    assert [(t["id"], t["kind"]) for t in want[-1]["tracks"]] == [(t["id"], t["kind"]) for t in got[-1]["tracks"]]


def test_sbl_mode_keeps_the_maps_detection_where_it_resolves_nothing():
    """A gate SBL is not run on (here: contrast threshold out of reach, or no gates allowed) keeps the map's detections."""
    from test_doa import active_ping
    base = {"bearings_deg": [-30.0, 30.0, 0.5], "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 60.0}, "min_snr_db": 12.0,
            "backend": "host", "engine": "cpp"}
    with LiveChain(base) as chain:
        want = chain.process(active_ping([2.0, 4.0]))["detections"]
    assert want
    for sbl in ({"min_contrast_db": 200.0}, {"max_gates": 0}):
        with LiveChain(dict(base, doa="sbl", sbl=sbl)) as chain:
            got = chain.process(active_ping([2.0, 4.0]))["detections"]
        key = lambda d: (round(d["range_m"], 3), round(d["bearing_deg"], 3))  # noqa: E731
        assert sorted(map(key, got)) == sorted(map(key, want))


@needs_exe
def test_program_listens_to_an_untriggered_board_stream(tmp_path):
    """--listen: blocks with no transmit mark, over UDP, become passive bearing frames as they arrive."""
    import socket
    import time
    from gpusonar.sources.flank_stream import FULL_SCALE_COUNTS, encode_block
    from test_doa import passive_record
    base = passive_record([(-20.0, 1.0), (15.0, 0.5)], seconds=2.0)
    # The boards send real samples: put the baseband record on its carrier at four times the rate.
    xb, fs_b, fc = base.samples.array, base.samples.sample_rate, base.samples.center_frequency
    up, n = 4, base.samples.array.shape[1]
    spec = np.zeros((xb.shape[0], up * n), dtype=complex)
    spec[:, :n // 2], spec[:, -(n // 2):] = np.fft.fft(xb, axis=1)[:, :n // 2], np.fft.fft(xb, axis=1)[:, -(n // 2):]
    t = np.arange(up * n) / (up * fs_b)
    real = np.real(up * np.fft.ifft(spec, axis=1) * np.exp(2j * np.pi * fc * t)[None, :]).astype(np.float32)
    rec = base.copy()
    s_ = rec.samples
    s_.array = real
    s_.kind = core.SampleKind.RealPassband
    s_.sample_rate = up * fs_b
    s_.start_time = 0.0
    rec.samples = s_
    w = gs.RecordWriter(str(tmp_path / "template.h5"))
    w.write(rec)
    w.close()
    full_scale = rec.metadata.adc_full_scale
    scale = FULL_SCALE_COUNTS / (full_scale if np.isfinite(full_scale) else 1.0)
    x = real.T                                                     # [samples, channels]
    counts = np.rint(x * min(scale, 0.2 * FULL_SCALE_COUNTS / np.abs(x).max())).astype(np.int32)
    N_FFT, HOP = 1024, 512                                         # the same 32 ms blocks at the higher rate
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    n_frames = len(counts) // (N_FFT + 11 * HOP)
    proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", str(tmp_path / "template.h5"), "--listen", "--band", "4000", "6000",
                             "--recording", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), "--pings", str(n_frames),
                             f"n_fft={N_FFT}", f"hop={HOP}", "n_average=12", "bearing_first_deg=-60", "bearing_last_deg=60", "sbl_min_snr_db=3"],
                            stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i, first in enumerate(range(0, len(counts), 128)):
            tx.sendto(encode_block(counts[first:first + 128], first, -1, 0, 0, 0, 5_000_000_000), ("127.0.0.1", port))
            if i % 4 == 3:
                time.sleep(0.002)
        _, err = proc.communicate(timeout=120.0)
    finally:
        proc.kill()
    frames = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    if "0 dropped, 0 samples lost" not in err or len(frames) < n_frames:
        pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    assert n_frames >= 5 and frames[0]["time_ns"] >= 5_000_000_000 and frames[1]["time_ns"] > frames[0]["time_ns"]
    good = sum(1 for f in frames if len(f["bearings_deg"]) == 2 and abs(min(f["bearings_deg"]) + 20.0) < 0.6
               and abs(max(f["bearings_deg"]) - 15.0) < 0.6)
    assert good >= 0.8 * len(frames)
    raw = list(make_source({"type": "container", "path": str(tmp_path / "rec.h5")}))
    assert len(raw) == n_frames and raw[0].samples.array.shape[1] == N_FFT + 11 * HOP
