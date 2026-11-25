"""Per-board streams (sonar-6f9, D-184): one GSFB stream per capture board, each with its own sample counter and capture
start, merged into the whole-array stream the existing assemblers take (`BoardMerger`, Python; `BoardMergerCpp`, the
C++ twin in libs/live board_merge.hpp). The merge is a stage in front of the unchanged `PingAssembler` of either
engine, so the D-161 rules for the whole-array stream hold as they were; the merge counts per board what it lost.

The board's block names itself with flag bit 1 and a board id at header offset 44 (reserved and zero in a whole-array
block, which therefore reads bit for bit as before)."""

import json
import pathlib
import struct
import sys

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.config import load_scenario_config
from gpusonar.sim import run_scenario
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator, PingAssembler, make_source
from gpusonar.sources import flank_stream as fs
from gpusonar.sources import board_merge as bm

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

TRIAL = "popoto_testauv_trial_1row"             # four boards of 8 on their own clocks, 4 pings 3 s apart
DEPLOY = ROOT / "configs" / "deploy" / "popoto_testauv_1row.yaml"
EXE = ROOT / "build" / "bin" / "sonar_live"
# Each board's capture starts at its own sample of the whole-array stream (negative: before the first transmit) and
# cuts its own blocks: what the merge must undo. Board 0 is the reference (its counter is the merged stream's).
OFFSETS = [0, -1234, 517, -40960]
BLOCKS = [512, 300, 512, 1000]


@pytest.fixture(scope="module")
def trial():
    sc = load_scenario_config(TRIAL)
    records = run_scenario(sc).records
    sonar = SonarModel.load(sc["sonar"], sc.get("sonar_overrides"))
    return sc, sonar, records


def _boards(sonar):
    return [{"id": i, "first": int(b["first"]), "count": int(b["count"])} for i, b in enumerate(sonar.boards)]


def _emulator(sc, records, sonar, **kw):
    kw.setdefault("start_offsets", OFFSETS)
    kw.setdefault("block_samples", BLOCKS)
    return bm.BoardStreamEmulator(records, float(sc["pings"]["interval"]), _boards(sonar), start_ns=1_700_000_000_000_000_000, **kw)


def _assembler(sonar, records, cpp=False):
    c = float(records[0].environment.sound_speed)
    if cpp:
        template = sonar.make_record(core.Pose(), environment={"sound_speed": c})
        return core.PingAssemblerCpp(template)
    return PingAssembler(sonar, sound_speed=c)


def _merger(sonar, options=None, cpp=False):
    options = dict({"channels": _boards(sonar)}, **(options or {}))
    if cpp:
        return core.BoardMergerCpp(options, sonar.n_elements, sonar.sample_rate)
    return bm.BoardMerger.from_options(options, sonar)


def _whole(sc, records, sonar, cpp=False):
    a = _assembler(sonar, records, cpp)
    blocks = FlankBoardEmulator(records, float(sc["pings"]["interval"]), start_ns=1_700_000_000_000_000_000).blocks()
    return [r for b in blocks for r in a.feed(b)] + list(a.finish())


def _merged(blocks, merger, assembler):
    out, merged = [], []
    for _, b in blocks:
        for m in merger.feed(b):
            merged.append(m)
            out += assembler.feed(m)
    for m in merger.finish():
        merged.append(m)
        out += assembler.feed(m)
    return out + list(assembler.finish()), merged


def _same_records(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.transmit.time_ns == w.transmit.time_ns and g.metadata.sequence == w.metadata.sequence
        assert g.samples.start_time == w.samples.start_time
        np.testing.assert_array_equal(g.samples.array, w.samples.array)


# --- the block -------------------------------------------------------------------------------------------------

def test_a_whole_array_block_is_unchanged_and_a_board_block_names_its_board():
    counts = np.arange(4 * 8, dtype=np.int32).reshape(4, 8) - 16
    whole = fs.encode_block(counts, 1000, 990, 7, 2, fs.FLAG_OVERFLOW, 123456789)
    # The version 1 header as it always was: fields to offset 44, then zeros to 128.
    head = struct.pack("<4sHHIQqIHHQ", b"GSFB", 1, 8, 4, 1000, 990, 7, 2, 1, 123456789)
    assert whole[:128] == head.ljust(128, b"\0")
    assert fs.decode_block(whole)["board"] is None
    board = fs.encode_block(counts, 1000, 990, 7, 2, fs.FLAG_OVERFLOW, 123456789, board=3)
    assert board[128:] == whole[128:] and board[:34] == whole[:34] and board[36:44] == whole[36:44]
    assert struct.unpack_from("<H", board, 34)[0] == fs.FLAG_OVERFLOW | fs.FLAG_BOARD
    assert struct.unpack_from("<H", board, 44)[0] == 3 and board[46:128] == b"\0" * 82
    b = fs.decode_block(board)
    assert b["board"] == 3 and b["flags"] & fs.FLAG_OVERFLOW
    with pytest.raises(fs.BlockError, match="board"):
        fs.encode_block(counts, 0, board=1 << 16)


def test_the_whole_array_receivers_refuse_a_board_block(trial):
    """A per-board block given to a receiver that takes one whole-array stream is refused, by name in Python and
    counted in bad_blocks in C++, even when its channel count happens to be the array's."""
    sc, sonar, records = trial
    counts = np.zeros((16, sonar.n_elements), dtype=np.int32)
    block = fs.encode_block(counts, 0, 0, 0, board=0)
    with pytest.raises(fs.BlockError, match="per-board block.*source.boards"):
        _assembler(sonar, records).feed(block)
    cpp = _assembler(sonar, records, cpp=True)
    assert cpp.feed(block) == [] and cpp.bad_blocks == 1


# --- the merge -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cpp", [False, True])
def test_per_board_streams_merge_into_the_whole_array_pings(trial, cpp):
    """Four boards, each with its own counter origin (one starts 0.4 s before the first transmit, one 5 ms after it)
    and block size, interleaved as they would arrive: the pings assembled from the merged stream are the pings of
    the whole-array stream, sample for sample, time stamp for time stamp."""
    sc, sonar, records = trial
    want = _whole(sc, records, sonar, cpp)
    assert len(want) == 4
    merger = _merger(sonar, cpp=cpp)
    got, _ = _merged(_emulator(sc, records, sonar).blocks(), merger, _assembler(sonar, records, cpp))
    _same_records(got, want)
    stats = json.loads(merger.stats_json()) if cpp else merger.stats()
    # Board 2 starts 517 samples after the reference: the stream's first two spans (512 samples each) lack it and are
    # left out (`missing: drop`), before any window opens. Nothing else is missing.
    assert stats["dropped_blocks"] == 2 and stats["zero_filled_blocks"] == 0 and stats["bad_blocks"] == 0
    for k, b in stats["boards"].items():
        assert b["missing_samples"] == (OFFSETS[2] if k == "2" else 0)
        assert b["late_blocks"] == 0 and b["restarts"] == 0 and b["repeated_blocks"] == 0


def test_both_engines_merge_to_the_same_bytes_and_counts(trial):
    """The same board blocks, reordered, duplicated and some lost, give the same merged blocks and the same counts in
    both engines (seeded shuffles; both rules)."""
    sc, sonar, records = trial
    base = _emulator(sc, records, sonar).blocks()
    for seed in range(6):
        rng = np.random.default_rng(seed)
        blocks = list(base)
        for _ in range(len(blocks) // 40):                         # swap neighbours: reordering within a few blocks
            i = int(rng.integers(0, len(blocks) - 3))
            j = i + int(rng.integers(1, 3))
            blocks[i], blocks[j] = blocks[j], blocks[i]
        lose = set(rng.choice(len(blocks), size=12, replace=False).tolist())
        twice = set(rng.choice(len(blocks), size=6, replace=False).tolist())
        stream = []
        for k, b in enumerate(blocks):
            if k in lose:
                continue
            stream.append(b)
            if k in twice:
                stream.append(b)
        for missing in ("drop", "zero"):
            py, cc = _merger(sonar, {"missing": missing}), _merger(sonar, {"missing": missing}, cpp=True)
            a = [m for _, b in stream for m in py.feed(b)] + py.finish()
            c = [m for _, b in stream for m in cc.feed(b)] + cc.finish()
            assert len(a) == len(c) and all(x == y for x, y in zip(a, c)), (seed, missing)
            assert py.stats() == json.loads(cc.stats_json()), (seed, missing)


@pytest.mark.parametrize("cpp", [False, True])
def test_a_block_lost_on_one_board_drops_its_ping_or_is_zero_filled(trial, cpp):
    """Board 2 loses one block inside ping 1's window. `missing: drop` (the default): the merged stream leaves that
    span out, the assembler drops ping 1 as it drops any ping with a hole (D-161), the other pings are whole; the
    board's lost samples are counted. `missing: zero`: every ping comes out, ping 1 with zeros where board 2 lost."""
    sc, sonar, records = trial
    want = _whole(sc, records, sonar, cpp)
    emu = _emulator(sc, records, sonar)
    period = int(round(float(sc["pings"]["interval"]) * sonar.sample_rate))
    victim = None
    for k, (board, block) in enumerate(emu.blocks()):
        b = fs.decode_block(block)
        merged_first = b["sample_index"] + OFFSETS[board]
        if board == 2 and merged_first > period + int(0.25 * sonar.sample_rate):
            victim, lost = k, b["n_samples"]
            break
    blocks = [x for k, x in enumerate(emu.blocks()) if k != victim]
    for missing in ("drop", "zero"):
        merger = _merger(sonar, {"missing": missing}, cpp)
        got, _ = _merged(blocks, merger, _assembler(sonar, records, cpp))
        stats = json.loads(merger.stats_json()) if cpp else merger.stats()
        assert stats["boards"]["2"]["missing_samples"] == lost + OFFSETS[2]      # and the 517 before it started
        assert all(b["missing_samples"] == 0 for k, b in stats["boards"].items() if k != "2")
        if missing == "drop":
            assert stats["dropped_blocks"] >= 3 and stats["zero_filled_blocks"] == 0
            _same_records(got, [want[0], want[2], want[3]])
        else:
            assert stats["dropped_blocks"] == 0 and stats["zero_filled_blocks"] >= 3
            _same_records([got[0], got[2], got[3]], [want[0], want[2], want[3]])
            diff = got[1].samples.array != want[1].samples.array
            assert diff.any() and not diff[[c for c in range(32) if not 16 <= c < 24]].any()
            assert np.count_nonzero(got[1].samples.array[16:24][diff[16:24]]) == 0


@pytest.mark.parametrize("cpp", [False, True])
def test_a_board_that_restarts_is_aligned_again_by_its_new_capture_start(trial, cpp):
    """Board 1 restarts its capture between pings 1 and 2 (counter from 0, a new capture start): the merge counts the
    restart and aligns its new counter by the new start; every ping is whole. The reference board restarting starts the
    merged stream afresh: the assembler sees a restart of the whole-array stream (D-161) and the pings after it."""
    sc, sonar, records = trial
    want = _whole(sc, records, sonar, cpp)
    period = int(round(float(sc["pings"]["interval"]) * sonar.sample_rate))
    restart_at = 2 * period - int(0.5 * sonar.sample_rate)      # merged sample, between the windows of pings 1 and 2
    merger = _merger(sonar, cpp=cpp)
    got, _ = _merged(_emulator(sc, records, sonar, restarts={1: restart_at}).blocks(), merger, _assembler(sonar, records, cpp))
    _same_records(got, want)
    stats = json.loads(merger.stats_json()) if cpp else merger.stats()
    assert stats["boards"]["1"]["restarts"] == 1 and sum(b["restarts"] for b in stats["boards"].values()) == 1
    assert stats["dropped_blocks"] == 2 and stats["boards"]["1"]["missing_samples"] == 0      # the 2 of board 2's late start
    # The reference restarts: the time stamps of the later pings follow its new capture start, as a whole-array
    # stream's restart would; the samples are the same.
    merger = _merger(sonar, cpp=cpp)
    a = _assembler(sonar, records, cpp)
    got, _ = _merged(_emulator(sc, records, sonar, restarts={0: restart_at}).blocks(), merger, a)
    assert a.restarts == 1 and len(got) == 4
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g.samples.array, w.samples.array)
    assert all(abs(g.transmit.time_ns - w.transmit.time_ns) <= 1 for g, w in zip(got, want))     # ns rounding of the new start


@pytest.mark.parametrize("cpp", [False, True])
def test_a_board_that_falls_silent_costs_the_pings_after_it(trial, cpp):
    """Board 3 stops sending after ping 1: with `missing: drop` the merge gives up waiting `wait_s` after each span it
    lacks, pings 2 and 3 are dropped and the samples it never sent are counted against board 3; `zero` keeps them."""
    sc, sonar, records = trial
    want = _whole(sc, records, sonar, cpp)
    period = int(round(float(sc["pings"]["interval"]) * sonar.sample_rate))
    silent = 2 * period - int(0.5 * sonar.sample_rate)
    blocks = [(b, x) for b, x in _emulator(sc, records, sonar).blocks()
              if not (b == 3 and fs.decode_block(x)["sample_index"] + OFFSETS[3] >= silent)]
    merger = _merger(sonar, cpp=cpp)
    got, _ = _merged(blocks, merger, _assembler(sonar, records, cpp))
    _same_records(got, want[:2])
    stats = json.loads(merger.stats_json()) if cpp else merger.stats()
    assert stats["boards"]["3"]["missing_samples"] > 2 * 0.9 * period - 0.5 * sonar.sample_rate
    merger = _merger(sonar, {"missing": "zero"}, cpp=cpp)
    got, _ = _merged(blocks, merger, _assembler(sonar, records, cpp))
    _same_records(got[:2], want[:2])
    assert len(got) == 4 and not got[3].samples.array[24:32].any()


def test_a_board_that_starts_after_the_first_window_opens_costs_that_ping(trial):
    sc, sonar, records = trial
    want = _whole(sc, records, sonar)
    late = int(0.5 * sonar.sample_rate)                         # after ping 0's window has opened (0.2 s)
    offsets = [0, -1234, late, -40960]
    got, _ = _merged(_emulator(sc, records, sonar, start_offsets=offsets).blocks(), _merger(sonar), _assembler(sonar, records))
    _same_records(got, want[1:])


@pytest.mark.parametrize("cpp", [False, True])
def test_a_wrong_board_or_channel_count_is_refused_by_name(trial, cpp):
    sc, sonar, records = trial
    merger = _merger(sonar, cpp=cpp)
    cases = [(fs.encode_block(np.zeros((8, 8), dtype=np.int32), 0, board=7), "board 7 is not in the board map"),
             (fs.encode_block(np.zeros((8, 9), dtype=np.int32), 0, board=1), "board 1 sent 9 channels, the board map gives it 8"),
             (fs.encode_block(np.zeros((8, 8), dtype=np.int32), 0), "no board id")]
    for block, message in cases:
        if cpp:
            assert merger.feed(block) == []
            assert message in merger.last_error
        else:
            with pytest.raises(fs.BlockError, match=message):
                merger.feed(block)
    stats = json.loads(merger.stats_json()) if cpp else merger.stats()
    assert stats["bad_blocks"] == 3 and stats["boards"]["1"]["bad_blocks"] == 1


@pytest.mark.parametrize("cpp", [False, True])
def test_the_board_options_refuse_unknown_keys_and_a_map_that_is_not_the_array(trial, cpp):
    sc, sonar, records = trial
    make = (lambda o: core.BoardMergerCpp(o, sonar.n_elements, sonar.sample_rate)) if cpp else (lambda o: bm.BoardMerger.from_options(o, sonar))
    with pytest.raises(ValueError, match="wiat_s.*wait_s"):
        make({"wiat_s": 0.1})
    with pytest.raises(ValueError, match="missing"):
        make({"missing": "repeat"})
    with pytest.raises(ValueError, match="channel 31"):
        make({"channels": [{"id": 0, "first": 0, "count": 31}]})
    with pytest.raises(ValueError, match="twice"):
        make({"channels": [{"id": 0, "first": 0, "count": 16}, {"id": 0, "first": 16, "count": 16}]})
    with pytest.raises(ValueError, match="reference"):
        make({"reference": 9})
    # The default map is the sonar file's receive.boards, board i carrying boards[i]'s channels.
    m = bm.BoardMerger.from_options({}, sonar)
    assert [(b.id, b.first, b.count) for b in m.boards] == [(i, 8 * i, 8) for i in range(4)] and m.reference == 0
    assert m.missing == "drop" and m.wait == int(np.floor(0.05 * sonar.sample_rate + 0.5))


# --- the receivers -----------------------------------------------------------------------------------------------

def test_the_flank_udp_source_merges_per_board_streams(trial):
    """`source.boards` in a deployment file: the Python source merges the boards' datagrams before its assembler; the
    statistics say what each board lost. Without the key it is the whole-array source as before."""
    sc, sonar, records = trial
    src = {"type": "flank_udp", "sonar": sc["sonar"], "sonar_overrides": sc.get("sonar_overrides"), "port": 0,
           "sound_speed": float(records[0].environment.sound_speed), "max_pings": 4, "timeout": 10.0, "boards": {}}
    source = make_source(src)
    try:
        _emulator(sc, records, sonar).send(source.port, pace=0.25).join(timeout=120.0)
        got = list(source)
    finally:
        source.close()
    if source.assembler.lost_samples or len(got) < 4:
        pytest.skip(f"the loopback dropped data: {len(got)} pings, {source.assembler.lost_samples} samples lost")
    _same_records(got, _whole(sc, records, sonar))
    stats = source.board_stats()
    assert set(stats["boards"]) == {"0", "1", "2", "3"} and stats["boards"]["2"]["missing_samples"] == OFFSETS[2]
    with pytest.raises(ValueError, match="source.boards.chanels"):
        make_source(dict(src, boards={"chanels": []}))


def test_the_relay_tool_turns_board_streams_into_one_stream(trial):
    """tools/board_merge.py: per-board datagrams in, whole-array datagrams out, for a receiver that does not merge
    (the Sensor Bridge path, the Holoscan source operator)."""
    import socket
    import threading
    import board_merge
    sc, sonar, records = trial
    source = make_source({"type": "flank_udp", "sonar": sc["sonar"], "sonar_overrides": sc.get("sonar_overrides"), "port": 0,
                          "sound_speed": float(records[0].environment.sound_speed), "max_pings": 4, "timeout": 15.0})
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    relay = threading.Thread(target=board_merge.main, args=([sc["sonar"], "--port", str(port), "--to", f"127.0.0.1:{source.port}",
                                                             "--idle", "3"],), daemon=True)
    relay.start()
    try:
        import time
        time.sleep(1.0)
        _emulator(sc, records, sonar).send(port, pace=0.25).join(timeout=120.0)
        got = list(source)
    finally:
        source.close()
        relay.join(timeout=20.0)
    if source.assembler.lost_samples or len(got) < 4:
        pytest.skip(f"the loopback dropped data: {len(got)} pings, {source.assembler.lost_samples} samples lost")
    _same_records(got, _whole(sc, records, sonar))


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_merged_trial_reports_what_the_whole_array_stream_reports(trial, engine):
    """The chain on the pings of the merged per-board streams reports exactly what it reports on the whole-array
    stream's pings (both engines; the records are the same, so the reports are)."""
    from gpusonar.config import load_yaml
    from gpusonar.live import LiveChain
    sc, sonar, records = trial
    cfg = load_yaml(str(DEPLOY))["chain"]
    cfg = dict(cfg, engine=engine)
    got, _ = _merged(_emulator(sc, records, sonar).blocks(), _merger(sonar), _assembler(sonar, records))
    want = _whole(sc, records, sonar)

    def reports(recs):
        with LiveChain(cfg) as chain:
            out = []
            for r in recs:
                rep = chain.process(r)
                rep.pop("processing_seconds", None)
                out.append(rep)
            return out
    a, b = reports(got), reports(want)
    assert a == b and all(r["sync"]["ok"] for r in a) and any(r["detections"] for r in a)


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live is not built")
def test_the_program_merges_per_board_streams_like_the_whole_array_stream(trial, tmp_path):
    """build/bin/sonar_live with the merge (`stream_boards=1`, from a deployment file's `source.boards` through
    --export-template) reports on the four boards' streams what it reports on the whole-array stream."""
    import shlex
    import socket
    import subprocess
    import time
    import sonar_live
    from gpusonar.config import load_yaml
    sc, sonar, records = trial
    outcome = {}
    for name in ("whole", "boards"):
        cfg = load_yaml(str(DEPLOY))
        cfg["source"] = dict(cfg["source"], sonar=sc["sonar"], sonar_overrides=sc.get("sonar_overrides"), timeout=15.0, host="127.0.0.1",
                             sound_speed=float(records[0].environment.sound_speed))
        if name == "boards":
            cfg["source"]["boards"] = {"missing": "drop"}
        cfg.pop("nav")
        cfg["output"] = {"report": str(tmp_path / f"{name}.jsonl")}
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        cfg["source"]["port"] = probe.getsockname()[1]
        probe.close()
        cmd = shlex.split(sonar_live.export_template(cfg, str(tmp_path / f"{name}.h5"), str(EXE)))
        assert ("stream_boards=1" in cmd) == (name == "boards")
        proc = subprocess.Popen([*cmd, "--pings", "4"], stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(1.0)
            if name == "boards":
                _emulator(sc, records, sonar).send(cfg["source"]["port"], pace=0.25).join(timeout=120.0)
            else:
                FlankBoardEmulator(records, float(sc["pings"]["interval"]), start_ns=1_700_000_000_000_000_000).send(
                    cfg["source"]["port"], pace=0.25).join(timeout=120.0)
            _, err = proc.communicate(timeout=180.0)
        finally:
            proc.kill()
        path = tmp_path / f"{name}.jsonl"
        reports = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        outcome[name] = (proc.returncode, reports, err)
    for name, (code, reports, err) in outcome.items():
        assert code == 0, err.strip()[-400:]
        if "0 dropped, 0 samples lost" not in err or len(reports) < 4:
            pytest.skip(f"the loopback dropped data ({name}): {err.strip().splitlines()[-1] if err.strip() else 'no summary'}")
    assert "sonar_live: boards {" in outcome["boards"][2]
    strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731
    assert [strip(r) for r in outcome["boards"][1]] == [strip(r) for r in outcome["whole"][1]]


def test_emulate_per_board_reports_what_the_simulator_gives_the_chain(tmp_path, capsys):
    """tools/sonar_live.py --emulate TRIAL --per-board: the four boards' streams, each starting at its own time, merged
    by the source, give the reports of the same pings given straight to the chain (--scenario), to the tolerances of
    test_emulate.py's whole-array case (the window opens on a whole sample of the stream)."""
    import sonar_live

    def run(argv, name):
        out = tmp_path / name
        assert sonar_live.main([str(DEPLOY), *argv, "--port", "0", "--out", str(out), "--quiet"]) == 0
        return [json.loads(line) for line in (out / "report.jsonl").read_text().splitlines()]
    direct = run(["--scenario", TRIAL], "direct")
    merged = run(["--emulate", TRIAL, "--per-board"], "boards")
    err = capsys.readouterr().err
    if len(merged) < len(direct):
        pytest.skip(f"the loopback dropped data: {len(merged)} of {len(direct)} pings")
    assert "sonar_live: boards {" in err
    assert len(merged) == len(direct) == 4
    for got, want in zip(merged, direct):
        assert got["sync"]["ok"] and got["sync"]["worst_residual_us"] < 1.0
        assert got["sync"]["offsets_us"] == pytest.approx(want["sync"]["offsets_us"], abs=0.001)
        a = min(got["detections"], key=lambda d: abs(d["range_m"] - 497.0))
        b = min(want["detections"], key=lambda d: abs(d["range_m"] - 497.0))
        assert a["range_m"] == pytest.approx(b["range_m"], abs=2.1) and abs(a["bearing_deg"] - b["bearing_deg"]) < 0.25


def test_source_boards_reach_the_program_as_key_value_options():
    """`source.boards` -> stream_boards and stream_board_* (live.program_options, as --export-template prints them), read
    back by the program's parser; a deployment without it gives none of them (the printed command as before)."""
    from gpusonar import live
    deploy = {"source": {"boards": {"channels": [{"id": 4, "first": 0, "count": 16}, {"id": 9, "first": 16, "count": 16}],
                                    "reference": 9, "missing": "zero", "wait_s": 0.02}}}
    flat = live.program_options({}, None, deploy)
    assert flat["stream_boards"] == 1 and flat["stream_board_channels"] == "4:0:16,9:16:16"
    got = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    assert got["stream_boards"] is True and got["stream_board_channels"] == "4:0:16,9:16:16"
    assert got["stream_board_reference"] == 9 and got["stream_board_missing"] == "zero" and got["stream_board_wait_s"] == 0.02
    assert not any(k.startswith("stream_board") for k in live.program_options({}, None, {"source": {}}))
    defaults = json.loads(core.live_options_from_text(""))
    assert defaults["stream_boards"] is False and defaults["stream_board_missing"] == "drop"
    with pytest.raises(ValueError, match="source.boards.wait: unknown key; did you mean 'wait_s'"):
        live.program_options({}, None, {"source": {"boards": {"wait": 0.1}}})
    with pytest.raises(ValueError, match="stream_board_missing"):
        core.live_options_from_text("stream_board_missing=repeat\n")
