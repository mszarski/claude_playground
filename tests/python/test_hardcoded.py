"""Another sonar, band, board or platform is a configuration change for the live chain and its programs.

  - the sizes the chain used to type (range cell, CFAR guard and training, tracker sigmas, multipath gate, fit limit,
    row tolerance, passive block) come from one function, libs/live derive.hpp: every shipped flank sonar gets exactly
    the numbers typed until now, a deliberately different sonar gets numbers that follow from it, and the chain
    still finds a target where the truth says, on both engines;
  - capacity limits say what happened: SBL on more channels than its kernels take, passive on stacked rows without
    the elevation option, a report line too long for a datagram, a block queue that fills, a stream that stops;
  - the stream's word length, socket buffer, block size and idle time are parameters.
tests/cpp/test_derive.cpp holds the closed-form checks of the same code."""

import json
import pathlib
import socket
import subprocess
import time

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import elevation, live, recon, sync, tonal
from gpusonar.config import load_yaml
from gpusonar.live import LiveChain, join_report_parts, split_report
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator, PingAssembler, flank_stream, make_source

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "bin" / "sonar_live"
needs_exe = pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")
STANDING = {"range_cell_m": 2.0, "cfar_guard_m": 100.0, "cfar_train_m": 300.0, "tracker_range_sigma": 15.0,
            "tracker_bearing_sigma": 0.017453292519943295, "multipath_bearing": 0.05235987755982988, "classify_max_fit_rms": 60.0}
FLANK_SONARS = ["flank_32", "popoto_8", "popoto_8_low", "popoto_array", "popoto_4x7", "popoto_4x8", "popoto_long_range",
                "popoto_long_range_2row_clocks", "popoto_long_range_3row_clocks", "popoto_face_1row_clocks", "popoto_static_2row_clocks",
                "popoto_static_3row_clocks", "popoto_testauv_1row_clocks"]
# A sonar that is none of the shipped ones in any respect the chain's sizes depend on: 900-1100 Hz instead of kHz,
# a 0.2 s pulse, eight hydrophones 0.6 m apart, 16 kHz sampling, fresh water.
OTHER = {"array": {"count": 8, "pitch": 0.6},
         "transmit": {"waveform": {"f_start": 900.0, "f_stop": 1100.0, "duration": 0.2}},
         "receive": {"sample_rate": 16000.0, "bandpass_hz": [200.0, 6000.0], "window": {"range": [500.0, 1500.0]}}}
OTHER_C = 1480.0
TARGET = (180.0, 900.0, 0.0)          # along the array, broadside, down


# --- one source for the chain's sizes ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", FLANK_SONARS)
def test_every_shipped_flank_sonar_gets_exactly_the_numbers_typed_until_now(name):
    """To the last bit, at the nominal sound speed and 60 m/s either side of it."""
    sonar = SonarModel.load(name)
    for c in (1440.0, 1500.0, 1560.0):
        rec = sonar.make_record(core.Pose(), environment={"sound_speed": c})
        d = core.live_derive(rec, {})
        assert {k: d[k] for k in STANDING} == STANDING, (name, c)
        assert core.live_passive_sizes(rec) == (1024, 512, 4608), (name, c)
        # Rows closer than 2 cm cannot be at these wavelengths; the tolerance is 2 cm, or tighter where a quarter wavelength is.
        top = max(rec.transmit.waveform.f_start, rec.transmit.waveform.f_stop)
        assert d["row_tolerance_m"] == min(0.02, 0.25 * c / top)
        facts = json.loads(d["json"])["sonar"]
        assert facts["rows"] == len(elevation.find_rows(rec))


def test_the_python_defaults_type_none_of_the_derived_numbers():
    """live.DEFAULTS leaves them to the derivation, and the options passed to the C++ chain do not carry them."""
    for path, name, _ in live.DERIVED:
        node = live.DEFAULTS
        for key in path:
            node = node[key]
        assert node is None, path
    o = live.cpp_chain_options({})
    assert not set(STANDING) & set(o) and "elevation" not in o
    typed = live.cpp_chain_options({"range_cell_m": 1.0, "tracker": {"bearing_sigma_deg": 2.0}, "elevation": {"row_tolerance_m": 0.005}})
    assert typed["range_cell_m"] == 1.0 and typed["tracker_bearing_sigma"] == pytest.approx(np.radians(2.0))
    assert typed["elevation"]["row_tolerance_m"] == 0.005 and typed["elevation"]["min_row_elements"] == 3
    assert "cfar_guard_m" not in typed
    assert elevation.DEFAULTS["row_tolerance_m"] is None and tonal.DEFAULTS["row_tolerance_m"] is None


@pytest.fixture(scope="module")
def other_pings():
    sonar = SonarModel.load("flank_32", OTHER)
    rng = np.random.default_rng(11)
    scene = Scene().add_points([list(TARGET)], target_strength_db=15.0)
    return [simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, environment={"sound_speed": OTHER_C}, sequence=i,
                          time_ns=int(i * 3e9), noise_rms=1e-5, rng=rng) for i in range(3)]


def test_a_different_sonar_gets_sizes_that_follow_from_it_and_the_chain_still_detects_with_truth(other_pings):
    """Nothing about the sonar is typed in the chain's configuration: bearings and the detection threshold only."""
    resolution = OTHER_C / (2.0 * 200.0)                     # 3.7 m
    beam = (OTHER_C / 1000.0) / (8 * 0.6)                    # 17.7 degrees
    got = {}
    for engine in ("python", "cpp"):
        with LiveChain({"bearings_deg": [-40.0, 40.0, 1.0], "engine": engine, "backend": "host",
                        "tracker": {"confirm_hits": 2}}) as chain:
            assert chain.resolved is None
            reports = chain.run([p.copy() for p in other_pings])
            got[engine] = (reports, chain.resolved)
        r = chain.resolved
        assert r["range_cell_m"] == pytest.approx(resolution, rel=1e-9) and r["range_cell_m"] > 2.0
        assert r["cfar_guard_m"] == 100.0 and r["cfar_train_m"] == 300.0          # 15 m and 44 m of floor: the standing values hold
        assert r["tracker_range_sigma"] == 15.0 and r["classify_max_fit_rms"] == 60.0
        assert r["tracker_bearing_sigma_deg"] == pytest.approx(np.degrees(beam / 16.0), rel=1e-9) and r["tracker_bearing_sigma_deg"] > 1.0
        assert r["multipath_bearing_deg"] == pytest.approx(3.0 * np.degrees(beam / 16.0), rel=1e-9)
        assert r["scaled"] == ["range_cell_m", "tracker.bearing_sigma_deg", "classify.multipath.bearing_deg"] and r["given"] == []
        assert r["sonar"]["row_elements"] == 8 and r["sonar"]["aperture_m"] == pytest.approx(4.8)
        # Truth: the target's slant range and its bearing from broadside.
        want_range, want_bearing = float(np.hypot(TARGET[0], TARGET[1])), float(np.degrees(np.arctan2(TARGET[0], TARGET[1])))
        for rep in reports:
            best = max(rep["detections"], key=lambda d: d["snr_db"])
            assert abs(best["range_m"] - want_range) < 2.0 * resolution, engine
            assert abs(best["bearing_deg"] - want_bearing) < 2.0, engine          # a ninth of the beam
        track = min(reports[-1]["tracks"], key=lambda t: np.hypot(t["x"] - TARGET[0], t["y"] - TARGET[1]))
        assert np.hypot(track["x"] - TARGET[0], track["y"] - TARGET[1]) < 40.0, engine
    # The two engines resolved the same numbers from the same function, and report the same detections.
    assert got["python"][1] == got["cpp"][1]
    for a, b in zip(got["python"][0], got["cpp"][0]):
        assert len(a["detections"]) == len(b["detections"])
        for x, y in zip(a["detections"], b["detections"]):
            assert x["range_m"] == pytest.approx(y["range_m"], abs=0.05) and x["bearing_deg"] == pytest.approx(y["bearing_deg"], abs=0.05)


def test_a_given_value_wins_on_both_engines(other_pings):
    for engine in ("python", "cpp"):
        with LiveChain({"bearings_deg": [-40.0, 40.0, 1.0], "engine": engine, "backend": "host", "range_cell_m": 8.0,
                        "cfar": {"guard_m": 60.0}, "tracker": {"bearing_sigma_deg": 3.0}}) as chain:
            chain.process(other_pings[0].copy())
        r = chain.resolved
        assert (r["range_cell_m"], r["cfar_guard_m"], r["cfar_train_m"]) == (8.0, 60.0, 300.0)
        assert r["tracker_bearing_sigma_deg"] == pytest.approx(3.0) and r["given"] == ["range_cell_m", "cfar.guard_m", "tracker.bearing_sigma_deg"]
        assert np.diff(chain.last["ranges"])[0] == pytest.approx(8.0)


def test_the_generated_example_build_runs_with_its_thresholds_left_to_the_chain(tmp_path):
    """docs/examples/reference_design/example_static_2x24.yaml (two rows of 24, 9-13 kHz, six-input boards, 64 kHz) through
    the generator, then its deployment file with every typed threshold taken out: the chain derives them (this array
    is inside the range the standing values suit) and reports the same as with them typed."""
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("build_from_spec", ROOT / "tools" / "build_from_spec.py")
    gen = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(gen)
    from gpusonar import buildspec as bs
    from gpusonar import config
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    gen.generate(bs.load(ROOT / "docs" / "examples" / "reference_design" / "example_static_2x24.yaml"), tmp_path, diagrams=False)
    config.add_config_dir(tmp_path / "configs")
    try:
        cfg = load_scenario_config("example_static_2x24_free")
        cfg["pings"]["count"] = 2
        recs = run_scenario(cfg).records
    finally:
        config.remove_config_dir(tmp_path / "configs")
    chain = load_yaml(tmp_path / "configs" / "deploy" / "example_static_2x24.yaml")["chain"]
    if recs[0].samples.is_complex():
        chain.pop("sync", None)
    stripped = {k: v for k, v in chain.items() if k not in ("range_cell_m", "cfar", "min_snr_db", "tracker", "classify")}
    for engine in ("python", "cpp"):
        out = {}
        for label, c in (("typed", chain), ("derived", stripped)):
            with LiveChain(dict(c, engine=engine, backend="host")) as lc:
                out[label] = lc.run([r.copy() for r in recs])
            if label == "derived":
                assert lc.resolved["given"] == [] and lc.resolved["scaled"] == []
                assert lc.resolved["sonar"]["rows"] == 2 and lc.resolved["sonar"]["row_elements"] == 24
        assert sum(len(r["detections"]) for r in out["derived"]) >= 2
        for a, b in zip(out["typed"], out["derived"]):
            a, b = dict(a, processing_seconds=0), dict(b, processing_seconds=0)
            assert json.dumps(a) == json.dumps(b), engine


# --- capacity limits say what happened --------------------------------------------------------------------------------

def test_sbl_on_more_channels_than_its_kernels_take_falls_back_and_the_report_says_so():
    """A single row of 72 with `doa: sbl`: this raised "the kernel path takes at most 64 channels"."""
    assert core.SBL_KERNEL_MAX_CHANNELS == 64
    over = {"array": {"count": 72, "pitch": 0.094}, "receive": {"window": {"range": [200.0, 420.0]}}}
    sonar = SonarModel.load("flank_32", over)
    scene = Scene().add_points([[60.0, 300.0, 0.0]], target_strength_db=10.0)
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, environment={"sound_speed": 1500.0}, noise_rms=1e-5,
                        rng=np.random.default_rng(2))
    cfg = {"bearings_deg": [-40.0, 40.0, 1.0], "range_cell_m": 1.0, "cfar": {"pfa": 1e-6, "guard_m": 15.0, "train_m": 50.0},
           "min_snr_db": 12.0, "backend": "host", "doa": "sbl"}
    with LiveChain(cfg) as chain:
        rep = chain.process(rec.copy())
    assert rep["sbl_fallback"] is True and chain.sbl_fallbacks == 1
    best = max(rep["detections"], key=lambda d: d["snr_db"])
    assert best["bearing_deg"] == pytest.approx(np.degrees(np.arctan2(60.0, 300.0)), abs=0.5)
    assert best["range_m"] == pytest.approx(np.hypot(60.0, 300.0), abs=2.0)
    # 32 channels: no flag in the line at all.
    small = simulate_ping(SonarModel.load("flank_32", {"receive": {"window": {"range": [200.0, 420.0]}}}), make_pose(position=(0.0, 0.0, 0.0)),
                          scene, environment={"sound_speed": 1500.0}, noise_rms=1e-5, rng=np.random.default_rng(2))
    with LiveChain(cfg) as chain:
        assert "sbl_fallback" not in chain.process(small)
    # The gate solver itself: the flag, and the host reference's answer.
    two_way = 2.0 * float(np.hypot(60.0, 300.0)) / 1500.0
    grid = list(np.radians(np.arange(-40.0, 40.25, 0.25)))
    conditioned = rec.copy()
    recon.prepare([conditioned], backend=core.Backend.Host)
    asked = core.doa_gate(conditioned, two_way, 256, grid)
    eigen = core.doa_gate(conditioned, two_way, 256, grid, kernels=False)
    assert asked["fallback"] is True and eigen["fallback"] is False and asked["peaks"] == eigen["peaks"] and asked["peaks"]


def _far_source_record(sonar_name, towards, seed=3):
    """A raw record of a sonar with a far noise source in its band: 60 tones of random phase, each hydrophone hearing
    them earlier by (its position . towards) / c."""
    sonar = SonarModel.load(sonar_name)
    rec = sonar.make_record(core.Pose(), environment={"sound_speed": 1500.0})
    w = rec.transmit.waveform
    rng = np.random.default_rng(seed)
    n = 1024 + 7 * 512
    fs = rec.samples.sample_rate
    lead = np.array([e.lever_arm for e in rec.geometry.rx]) @ np.asarray(towards) / 1500.0
    t = np.arange(n) / fs
    x = np.zeros((len(lead), n))
    for f, ph in zip(rng.uniform(min(w.f_start, w.f_stop), max(w.f_start, w.f_stop), 60), rng.uniform(0.0, 2.0 * np.pi, 60)):
        x += np.cos(2.0 * np.pi * f * (t[None, :] + lead[:, None]) + ph)
    x += 0.5 * rng.standard_normal(x.shape)
    s = rec.samples
    kind = s.kind
    s.array = x.astype(np.float32)
    s.kind = kind
    s.start_time = 0.0
    rec.samples = s
    return rec


def test_passive_on_stacked_rows_without_the_elevation_option_gives_the_cone_bearing_and_says_how():
    """Three rows of 32 (96 channels) without `elevation`: this raised on the kernel path and returned a wrong or missing
    source on the reference path, because all 96 hydrophones were taken as one line."""
    cone, up = np.radians(15.0), np.radians(20.0)
    towards = (np.sin(cone), np.sqrt(1.0 - np.sin(cone) ** 2 - np.sin(up) ** 2), -np.sin(up))
    rec = _far_source_record("popoto_static_3row", towards)
    assert rec.samples.n_channels == 96 and len(elevation.find_rows(rec)) == 3
    frames, _ = core.doa_passive(rec, first_deg=-60.0, last_deg=60.0)
    assert len(frames) == 1 and frames[0]["rows_as_snapshots"] == 3 and "sbl_fallback" not in frames[0]
    assert np.degrees(frames[0]["peaks"][0][0]) == pytest.approx(15.0, abs=0.5)
    with_el, _ = core.doa_passive(rec, first_deg=-60.0, last_deg=60.0, elevation=True)
    assert "rows_as_snapshots" not in with_el[0] and with_el[0]["peaks"][0][0] == frames[0]["peaks"][0][0]
    assert np.degrees(with_el[0]["elevations"][0]) == pytest.approx(20.0, abs=1.5)
    # One row of the same hydrophones: as before, no flag.
    line, _ = core.doa_passive(_far_source_record("popoto_long_range", towards), first_deg=-60.0, last_deg=60.0)
    assert "rows_as_snapshots" not in line[0] and np.degrees(line[0]["peaks"][0][0]) == pytest.approx(15.0, abs=0.5)


def _long_report(n=300):
    return {"ping": 3, "sequence": 9, "time_ns": 123, "ownship_xyz": [0.0, 0.0, 10.0], "pings_without_nav": 0, "sync": None,
            "detections": [{"range_m": 1000.0 + i, "bearing_deg": 0.1 * i, "snr_db": 12.0, "elevation_deg": 1.0, "depth_m": 50.0,
                            "path": "direct"} for i in range(n)],
            "tracks": [{"id": i, "kind": "mover", "parent": None, "x": 1.0 * i, "y": 2.0, "vx": 0.0, "vy": 0.0, "speed_m_s": 0.0,
                        "range_m": 5.0, "mean_snr_db": 14.0, "seen": 1.0} for i in range(40)],
            "processing_seconds": 0.01, "rows": 3}


def test_a_report_too_long_for_a_datagram_is_split_into_complete_lines_never_cut():
    report = _long_report()
    assert split_report(report, 10**6) == [json.dumps(report)]             # it fits: the line, no part field
    parts = [json.loads(p) for p in split_report(report, 3000)]
    assert len(parts) > 5 and all(len(json.dumps(p).encode()) <= 3000 for p in parts)
    # A reader that knows nothing of parts: every datagram is a report of this ping with every field, and fewer detections.
    for k, p in enumerate(parts):
        assert set(p) == set(report) | {"part"} and p["part"] == [k + 1, len(parts)]
        assert (p["ping"], p["sequence"], p["time_ns"], p["rows"]) == (3, 9, 123, 3)
    # A reader that does: the report again.
    assert join_report_parts(parts[::-1]) == report
    with pytest.raises(ValueError):
        join_report_parts(parts[1:])


def test_the_chain_sends_a_long_report_as_parts_over_udp(other_pings):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(5.0)
    for engine in ("python", "cpp"):
        # A threshold low enough for hundreds of detections, and a datagram limit they do not fit.
        with LiveChain({"bearings_deg": [-40.0, 40.0, 1.0], "engine": engine, "backend": "host", "min_snr_db": -50.0,
                        "cell_statistic": "mean", "cfar": {"pfa": 0.2}}, report_udp=("127.0.0.1", rx.getsockname()[1]), report_udp_max_bytes=1000) as chain:
            report = chain.process(other_pings[0].copy())
        assert len(json.dumps(report)) > 2000 and chain.reports_split == 1, engine
        parts = [json.loads(rx.recv(65535))]
        while len(parts) < parts[0]["part"][1]:
            data = rx.recv(65535)
            assert len(data) <= 1000
            parts.append(json.loads(data))                         # each one parses: nothing was cut
        assert join_report_parts(parts) == json.loads(json.dumps(report)), engine
    rx.close()


# --- the stream: word length, block size, socket buffer, queue ----------------------------------------------------------

@pytest.mark.parametrize("sample_bytes", [2, 3, 4])
def test_blocks_of_any_word_length_on_both_ends_and_both_assemblers(sample_bytes):
    over = {"receive": {"window": {"range": [100.0, 130.0]}}}
    sonar = SonarModel.load("flank_32", over)
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), Scene().add_points([[0.0, 115.0, 0.0]], target_strength_db=10.0),
                        environment={"sound_speed": 1500.0}, noise_rms=1e-5, rng=np.random.default_rng(1))
    full = flank_stream.full_scale_counts(sample_bytes)
    assert full == 2.0 ** (8 * sample_bytes - 1)
    counts = np.array([[-full, full - 1, 1, -1]], dtype=np.int64).T.repeat(2, axis=1)
    back = flank_stream.decode_block(flank_stream.encode_block(counts, 5, sample_bytes=sample_bytes), sample_bytes)
    assert np.array_equal(back["counts"], counts) and back["sample_index"] == 5
    with pytest.raises(flank_stream.BlockError):
        flank_stream.encode_block(np.array([[full]]), 0, sample_bytes=sample_bytes)
    emulator = FlankBoardEmulator([rec], 3.0, sample_bytes=sample_bytes)
    assert emulator.block_samples == flank_stream.block_samples_for(32, sample_bytes) == 128 * 3 // sample_bytes
    py_asm = PingAssembler(sonar, sample_bytes=sample_bytes)
    cpp_asm = core.PingAssemblerCpp(sonar.make_record(core.Pose()), sample_bytes=sample_bytes)
    wrong = PingAssembler(sonar, sample_bytes=2 if sample_bytes != 2 else 3)
    py_out, cpp_out = [], []
    for block in emulator.blocks():
        py_out += py_asm.feed(block)
        cpp_out += cpp_asm.feed(block)
        with pytest.raises(flank_stream.BlockError):
            wrong.feed(block)                                      # another word length is an error, never samples
    assert len(py_out) == len(cpp_out) == 1
    step = 2.5 / full                                              # volts per count at this word length
    assert np.abs(py_out[0].samples.array - rec.samples.array).max() <= 0.5001 * step + 1e-9
    np.testing.assert_allclose(cpp_out[0].samples.array, py_out[0].samples.array, rtol=0, atol=1e-9)


def test_blocks_are_sized_from_the_channel_count_and_the_buffer_from_the_data_rate():
    assert flank_stream.block_samples_for(32) == 128 and flank_stream.block_samples_for(96) == 42 and flank_stream.block_samples_for(8) == 512
    assert flank_stream.block_samples_for(10000) == 1
    for n in (8, 32, 96, 200, 400):
        assert flank_stream.HEADER_BYTES + flank_stream.block_samples_for(n) * n * 3 <= flank_stream.MAX_DATAGRAM
    assert flank_stream.stream_buffer_bytes(32, 102400.0) == 8 << 20                          # half a second is 4.9 MB
    assert flank_stream.stream_buffer_bytes(96, 102400.0) == 96 * 102400 * 3 // 2            # 14.7 MB
    assert flank_stream.stream_buffer_bytes(96, 102400.0, given=1 << 20) == 1 << 20
    three = SonarModel.load("popoto_static_3row").make_record(core.Pose())
    assert core.live_stream_buffer_bytes(three) == flank_stream.stream_buffer_bytes(96, 102400.0)
    assert core.live_stream_bytes_per_second(three, 3) == 96 * 102400 * 3
    # A block that cannot go in a datagram is refused before anything is sent (it used to fail in the sending thread, unseen).
    rec = SonarModel.load("popoto_static_3row", {"receive": {"window": {"range": [100.0, 101.0]}}}).make_record(core.Pose())
    assert FlankBoardEmulator([rec], 3.0).block_samples == 42
    with pytest.raises(ValueError, match="does not fit a UDP datagram"):
        FlankBoardEmulator([rec], 3.0, block_samples=512).send(9)
    assert len(next(FlankBoardEmulator([rec], 3.0, block_samples=512).blocks())) == 128 + 512 * 96 * 3     # handed over directly: any size


def test_the_udp_source_names_its_sonar_reports_its_buffer_and_bounds_its_queue():
    with pytest.raises(TypeError):
        make_source({"type": "flank_udp", "port": 0})              # no default sonar
    over = {"receive": {"window": {"range": [100.0, 130.0]}}}
    with pytest.warns(RuntimeWarning, match="socket buffer"):      # more than any system grants: said, not hidden
        greedy = make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": over, "buffer_bytes": 1 << 40, "timeout": 0.5})
    assert 0 < greedy.buffer_granted < greedy.buffer_asked
    greedy.close()
    source = make_source({"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": over, "buffer_bytes": 1 << 16, "timeout": 1.0,
                          "queue_mb": 0.02})
    assert source.buffer_asked == 1 << 16 and source.buffer_granted >= 1 << 16
    assert source.stream_bytes_per_second == 32 * 64000 * 3
    # The assembler is held up while blocks arrive: the queue fills at 20 kB and the rest are counted, not kept.
    feed = source.assembler.feed
    source.assembler.feed = lambda data: (time.sleep(0.05), feed(data))[1]
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    block = flank_stream.encode_block(np.zeros((32, 32), dtype=np.int32), 0)
    for _ in range(60):
        tx.sendto(block, ("127.0.0.1", source.port))
        time.sleep(0.001)
    time.sleep(0.3)
    tx.close()
    assert source.dropped_blocks > 10
    assert list(source) == []                                      # and it ends after `timeout` without a ping
    source.close()


# --- sync and navigation keys -------------------------------------------------------------------------------------------

def test_sync_tolerance_and_search_time_are_keys():
    rec = SonarModel.load("popoto_long_range").make_record(core.Pose())
    assert sync.alignment_tolerance(rec) == 1.0 / (20.0 * 6000.0) == core.live_sync_tolerance(0.0, 6000.0)
    assert sync.alignment_tolerance(rec, {"tolerance_us": 0.87}) == pytest.approx(0.87e-6) == core.live_sync_tolerance(0.87, 6000.0)
    chain = {"sync": {"reference": {"f_start": 38e3, "f_stop": 48e3, "duration": 4e-3, "lead": 5e-3, "tail": 10e-3}, "search_time": 5e-3,
                      "tolerance_us": 0.87, "min_quality": 0.5}}
    ref = live.sync_reference(chain)
    assert (ref["search_time"], ref["tolerance_us"], ref["min_quality"]) == (5e-3, 0.87, 0.5)
    o = live.cpp_chain_options(chain, ref)["sync"]
    assert (o["search_time"], o["tolerance_us"]) == (5e-3, 0.87)
    core.LiveChainCpp(dict(live.cpp_chain_options(chain, ref), backend=core.Backend.Host))      # the C++ chain takes them


def test_boards_further_apart_than_the_default_search_need_the_key():
    """Four boards whose clocks start up to 3 ms apart: the default 2 ms search does not find the bursts, 5 ms does."""
    from test_calibration_sync import four_board_ping
    sonar, rec, _, truth = four_board_ping()
    ref = sonar.config["receive"]["reference"]
    s = rec.samples                                                # the fixture's own clocks are close; shift one board's samples
    x = s.array.copy()
    shift = int(3e-3 * s.sample_rate)
    rows = [r for r, b in enumerate(rec.metadata.channel_board) if b == 2]
    x[rows] = np.roll(x[rows], shift, axis=1)                      # board 2 started 3 ms late
    kind = s.kind
    s.array = x
    s.kind = kind
    rec.samples = s
    far = sync.estimate_clocks_reference(rec, dict(ref, search_time=5e-3))
    assert far.offset[2] - (truth["offset"][2] - truth["offset"][0]) == pytest.approx(-shift / s.sample_rate, abs=2e-6)
    near = None
    try:
        near = sync.estimate_clocks_reference(rec, ref)            # 2 ms either side: the burst is outside it
    except sync.SyncError:
        pass
    assert near is None or abs(near.offset[2] - far.offset[2]) > 0.5e-3
    c = core.live_synchronise(rec.copy(), {k: ref[k] for k in ("f_start", "f_stop", "duration", "lead", "tail")} | {"search_time": 5e-3})
    assert c["offset"][2] == pytest.approx(far.offset[2], abs=0.5e-6)


def test_navigation_history_margin_is_a_key():
    nav = live.NavListener.from_config({"static": {"x": 0.0, "y": 0.0, "z": 5.0}, "history_margin_s": 0.5})
    for i in range(1, 60):
        nav.push({"time_ns": int(i * 1e8), "x": 0.0, "y": 0.0, "z": 5.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0})
    got = [t for t, _ in nav.history(int(2e9), int(3e9))]
    assert got[0] == int(1.5e9) and got[-1] == int(3.5e9)
    wide = live.NavListener(static={"x": 0.0, "y": 0.0, "z": 5.0})
    assert wide.history_margin_ns == int(2e9)
    nav.close()
    wide.close()
    # A navigation datagram longer than 4 kB is taken whole (it was cut at 4096 bytes and dropped as bad JSON).
    listener = live.NavListener(port=0, host="127.0.0.1")
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fix = {"time_ns": 77, "x": 1.0, "y": 2.0, "z": 3.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "padding": "x" * 6000}
    tx.sendto(json.dumps(fix).encode(), ("127.0.0.1", listener.port))
    for _ in range(50):
        if listener.at(77) is not None:
            break
        time.sleep(0.05)
    assert listener.at(77) is not None and listener.at(77)[0][2] == 3.0
    tx.close()
    listener.close()


# --- the standalone program ---------------------------------------------------------------------------------------------

def _template(tmp_path, over=None):
    sonar = SonarModel.load("flank_32", over or {"receive": {"window": {"range": [100.0, 130.0]}}})
    path = str(tmp_path / "template.h5")
    w = core.RecordWriter(path)
    w.write(sonar.make_record(core.Pose()))
    w.close()
    return path


@needs_exe
def test_program_idle_time_is_a_key_and_its_summary_counts_everything(tmp_path):
    """No block ever arrives: with idle_seconds=1 the program ends by itself (it waited a fixed 30 s), returns 0 and
    prints the socket buffer it was granted and every counter."""
    t0 = time.monotonic()
    out = subprocess.run([str(EXE), "--udp", "0", "--template", _template(tmp_path), "idle_seconds=1", "bind=127.0.0.1"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert time.monotonic() - t0 < 25.0
    assert "socket buffer asked 8388608 bytes, granted" in out.stderr and "6.1 MB/s" in out.stderr
    last = out.stderr.strip().splitlines()[-1]
    assert "0 pings" in last and "0 blocks dropped at the full queue" in last and "0 reports split" in last and "0 report datagrams refused" in last
    # A word length that is not one: an error, not a guess.
    bad = subprocess.run([str(EXE), "--udp", "0", "--template", _template(tmp_path), "stream_sample_bytes=5", "idle_seconds=1"],
                         capture_output=True, text=True, timeout=120)
    assert bad.returncode == 1 and "sample_bytes must be 2, 3 or 4" in bad.stderr


@needs_exe
def test_program_prints_what_it_resolved_once_and_stamps_a_long_output_path(tmp_path, other_pings):
    path = str(tmp_path / "pings.h5")
    w = core.RecordWriter(path)
    for p in other_pings[:2]:
        w.write(p)
    w.close()
    # An output path longer than the 1 kB the stamp's buffer used to be (then the unstamped path was used, and a restart wrote over it).
    deep = tmp_path
    while len(str(deep)) < 1100:
        deep = deep / ("d" * 100)
    deep.mkdir(parents=True)
    out = subprocess.run([str(EXE), "--replay", path, "--report", str(deep / "rep_%Y%m%d.jsonl"), "bearing_first_deg=-40",
                          "bearing_last_deg=40", "bearing_step_deg=1"], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    line = [text for text in out.stderr.splitlines() if text.startswith("sonar_live: chain settings ")]
    assert len(line) == 1                                           # once, at the first ping
    resolved = json.loads(line[0][len("sonar_live: chain settings "):])
    with LiveChain({"bearings_deg": [-40.0, 40.0, 1.0], "engine": "cpp", "backend": "host"}) as chain:
        chain.process(other_pings[0].copy())
    assert resolved == chain.resolved and resolved["range_cell_m"] == pytest.approx(OTHER_C / 400.0)
    stamped = list(deep.glob("rep_2*.jsonl"))
    assert len(stamped) == 1 and len(stamped[0].read_text().splitlines()) == 2 and not list(deep.glob("rep_%*"))


@needs_exe
def test_program_block_queue_is_bounded_and_counts_what_it_drops(tmp_path):
    """Blocks arrive faster than a queue of 20 kB lets through while the chain is busy: dropped and counted in the
    summary, the program ends normally."""
    over = {"receive": {"window": {"range": [100.0, 400.0]}}}
    sonar = SonarModel.load("flank_32", over)
    rng = np.random.default_rng(4)
    scene = Scene().add_points([[20.0, 250.0, 0.0]], target_strength_db=10.0)
    recs = [simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, environment={"sound_speed": 1500.0}, sequence=i,
                          time_ns=int(i * 1e9), noise_rms=1e-5, rng=rng) for i in range(3)]
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", _template(tmp_path, over), "stream_queue_mb=0.02", "idle_seconds=3",
                             "bind=127.0.0.1", "bearing_step_deg=1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.0)
    FlankBoardEmulator(recs, 3.0).send(port, pace=0.0).join(timeout=120.0)
    _, err = proc.communicate(timeout=300)
    assert proc.returncode == 0, f"sonar_live ended abnormally ({proc.returncode}): {err.strip()[-400:]}"   # a crash is not lost data
    last = err.strip().splitlines()[-1]
    dropped = int(last.split(" blocks dropped at the full queue")[0].split(", ")[-1])
    assert dropped > 0, last


# --- tracks do not depend on depth, an elevation pick or an unproven neighbour -------------------------------------------

def _twelve_km_two_rows(pings=6):
    """popoto_12km_2row by straight rays (runs anywhere), the ship's points 4 dB up, no reverberation: at this level and
    without a water depth the two rows' elevation is ambiguous on every ping, and its pick changes from ping to ping."""
    import importlib.util
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    spec_ = importlib.util.spec_from_file_location("popoto_12km_study", ROOT / "tools" / "popoto_12km_study.py")
    study = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(study)
    cfg = load_scenario_config("popoto_12km_2row")
    cfg.pop("truth", None)
    cfg = study.image_source(cfg)
    cfg.pop("reverberation", None)
    cfg["scene"][0]["target_strength_db"] += 4.0
    cfg["pings"]["count"] = pings
    return run_scenario(cfg).records


def test_an_ambiguous_elevation_does_not_move_the_detection_and_the_mover_keeps_one_track():
    """Two rows, a mover at 12 km, the elevation pick jumping between +10, -12 and -30 degrees: placed by the pick, the
    detection moved by hundreds of metres from ping to ping, the track appeared two pings late 300 m off and stayed
    `uncertain`. Now the correction is applied only to an unambiguous elevation, and the report says when it was not."""
    recs = _twelve_km_two_rows()
    chain = load_yaml(ROOT / "configs" / "deploy" / "popoto_face_2row.yaml")["chain"]
    chain.pop("sync", None)
    chain["elevation"] = dict(chain["elevation"], correct_bearing=True, water_depth=None)
    chain["elevation"].pop("sector_from_rays", None)                     # the shipped file now limits the search by rays (D-118), which removes some of the
                                                                         # ambiguity this test is about: it is taken out so the rule for an ambiguous pick is still tested
    got = {}
    for engine in ("python", "cpp"):
        with LiveChain(dict(chain, engine=engine, backend="host")) as c:
            reps = got[engine] = c.run([r.copy() for r in recs])
        picks = []
        for i, rep in enumerate(reps):
            assert len(rep["detections"]) == 1, engine
            d = rep["detections"][0]
            assert d["elevation_ambiguous"] is True and d["bearing_corrected"] is False and "cone_bearing_deg" not in d, engine
            # Where the ship is: 12 km off, crossing at 6 m/s against the vehicle's 1.5 m/s, 19.9 s a ping.
            assert abs(d["range_m"] - 12000.0) < 30.0, engine
            assert d["bearing_deg"] == pytest.approx(np.degrees(np.arctan2(800.0 - 7.5 * 19.9 * i, 12000.0)), abs=2.0), engine
            picks.append(d["elevation_deg"])
        assert max(picks) - min(picks) > 8.0, engine                     # the pick really does jump (by 10.9 deg with the noise of D-103; 40 deg with the earlier draw)
        # One track, there from the third ping, at the ship, a mover by the last ping: no depth was needed for it.
        assert [len(r["tracks"]) for r in reps] == [0, 0, 1, 1, 1, 1], engine
        last = reps[-1]["tracks"][0]
        assert last["kind"] == "mover" and abs(last["range_m"] - 12000.0) < 60.0 and 3.0 < last["speed_m_s"] < 8.0, engine
        assert last.get("depth_m") is None, engine                       # ambiguous throughout: no depth, and the track all the same
        assert {t["id"] for r in reps for t in r["tracks"]} == {last["id"]}, engine
    for a, b in zip(got["python"], got["cpp"]):
        assert a["detections"][0]["bearing_deg"] == pytest.approx(b["detections"][0]["bearing_deg"], abs=0.05)
        assert [t["kind"] for t in a["tracks"]] == [t["kind"] for t in b["tracks"]]


def test_a_track_is_called_multipath_only_of_a_track_classed_as_an_object():
    """A weaker track 40 m behind a stronger one on its bearing, with its velocity. When the stronger is a clean mover the
    weaker is its echo. When the stronger is itself `uncertain` (seen on too few pings to be classed) the weaker is
    not explained away by it: it keeps its own class. Both engines."""
    from gpusonar import classify, detect

    def run(late):
        opts = dict(process_noise=0.3, range_sigma=6.0, bearing_sigma=np.radians(0.4), confirm_hits=3, max_misses=2, initial_speed_sigma=6.0)
        py_tracker, cpp_tracker = detect.Tracker(**opts), core.LiveTracker(**opts)
        for i in range(8):
            t = 2.0 * i
            dets = [(1040.0 + 3.0 * t, 0.2, 15.0, 2)]
            if not late or i >= 5:                 # the stronger one seen on the last three pings only: too short to class
                dets.insert(0, (1000.0 + 3.0 * t, 0.2, 30.0, 2))
            a = py_tracker.update(t, [detect.Detection(*d) for d in dets])
            cpp_tracker.update(t, dets)
        kw = dict(min_speed=1.0, static_speed=1.0, max_fit_rms=3.0)
        mp = {"max_excess_range": 800.0, "bearing": np.radians(3.0), "velocity": 2.5, "origin": (0.0, 0.0)}
        want = classify.classify_tracks(a, multipath=mp, **kw)
        got = cpp_tracker.classify(multipath=True, max_excess_range=800.0, multipath_bearing=np.radians(3.0), multipath_velocity=2.5, **kw)
        assert [(c.kind, c.parent) for c in want] == [(c["kind"], c["parent"]) for c in got]
        return sorted(want, key=lambda c: -c.mean_snr_db)

    strong, weak = run(late=False)
    assert (strong.kind, weak.kind, weak.parent) == ("mover", "multipath", strong.id)
    strong, weak = run(late=True)
    assert strong.kind == "uncertain" and weak.kind == "mover" and weak.parent is None


# --- what snr_db is ---------------------------------------------------------------------------------------------------------

def test_snr_db_against_the_beam_output_snr_of_a_link_budget_is_what_the_helper_says():
    """A point target in free field. The reference is what a link budget states: the peak of the noiseless, unweighted
    beam output at the target over the mean noise power of the same beamformer. The chain's `snr_db` is below it by
    the aperture window's loss and by what the range cell does to an echo one resolution long: 9.4 dB with the
    2 m cell of mean power and the Hann window (1.8 + 7.7; this was the "unexplained" part of the chain reading
    12 dB under the budget, the rest being the source level's peak-or-rms convention), 6.6 dB with `cell_statistic: peak`."""
    from gpusonar import flank
    sonar = SonarModel.load("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1500.0]}}})
    was = {"cell_statistic": "mean"}             # the detector this accounting is of; the default since is `multiscale` (test_detector.py)
    cases = {"mean": was, "peak": {"cell_statistic": "peak"}, "rect": dict(was, window="rect"), "half metre": dict(was, range_cell_m=0.5)}
    read = {k: [] for k in cases}
    ideal = []
    for trial in range(3):
        rng = np.random.default_rng(100 + trial)
        r_t, b_t = 1200.0 + rng.uniform(-3.0, 3.0), np.radians(rng.uniform(-10.0, 10.0))
        scene = Scene().add_points([[r_t * np.sin(b_t), r_t * np.cos(b_t), 0.0]], target_strength_db=0.0)
        kw = dict(environment={"sound_speed": 1500.0}, adc=False, absorption_model="none")
        rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=2e-4, rng=rng, **kw)
        quiet = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=0.0, rng=rng, **kw)
        only = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), Scene().add_points([[0.0, 1200.0, 0.0]], target_strength_db=-300.0),
                             noise_rms=2e-4, rng=rng, **kw)
        recon.prepare([quiet, only], backend=core.Backend.Host)
        peak = np.abs(flank.backprojection_map(quiet, r_t + np.arange(-2.0, 2.0, 0.01), np.array([b_t]), window="rect", interp="linear",
                                               backend=core.Backend.Host).image) ** 2
        noise = np.abs(flank.backprojection_map(only, np.arange(1000.0, 1400.0, 0.3), np.radians(np.arange(-30.0, 30.0, 2.0)), window="rect",
                                                interp="linear", backend=core.Backend.Host).image) ** 2
        ideal.append(10.0 * np.log10(peak.max() / noise.mean()))
        for name, cfg in cases.items():
            for engine in (("cpp", "python") if name in ("mean", "peak") and trial == 0 else ("cpp",)):
                with LiveChain(dict(cfg, bearings_deg=[-40.0, 40.0, 0.5], backend="host", engine=engine)) as chain:
                    rep = chain.process(rec.copy())
                best = max(rep["detections"], key=lambda d: d["snr_db"])
                assert abs(best["range_m"] - r_t) < 2.5
                if engine == "cpp":
                    read[name].append(best["snr_db"])
                else:
                    assert best["snr_db"] == pytest.approx(read[name][-1], abs=0.05)     # the engines read the same
    template = sonar.make_record(core.Pose(), environment={"sound_speed": 1500.0})
    for name, cfg in cases.items():
        loss = live.snr_loss_db(template, cfg.get("range_cell_m", 2.0), cfg.get("window", "hann"), cfg.get("cell_statistic", "mean"))
        measured = float(np.mean(ideal) - np.mean(read[name]))
        assert measured == pytest.approx(loss["total_db"], abs=1.0), (name, measured, loss)
    mean = live.snr_loss_db(template, 2.0)
    assert mean["samples_per_cell"] == 8 and mean["window_db"] == pytest.approx(1.76, abs=0.02) and mean["cell_db"] == pytest.approx(7.7, abs=0.15)
    assert live.snr_loss_db(template, 2.0, cell_statistic="peak")["total_db"] == pytest.approx(6.6, abs=0.15)
    # In a budget's units the chain as shipped reports a point echo from 20 dB, not 13; with the peak statistic from 17.
    assert live.detection_threshold_db(was) == pytest.approx(10.7, abs=0.05)
    assert live.budget_threshold_db(template, was) == pytest.approx(20.1, abs=0.2)
    assert live.budget_threshold_db(template, {"cell_statistic": "peak"}) == pytest.approx(17.3, abs=0.2)
    with pytest.raises(ValueError):
        LiveChain({"cell_statistic": "median"})
