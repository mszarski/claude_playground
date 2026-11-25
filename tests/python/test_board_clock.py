"""A board's sample clock against true time (sonar-ubg, D-185).

`transmit.time_ns` is the capture start plus the transmit's sample index over the NOMINAL sample rate, in both engines
(flank_stream.PingAssembler._record, libs/live udp_source.cpp make_record). A board clock fast by e (counting fs (1 + e)
samples a true second) puts the stamp e x (time since capture start) late: 25 ppm is 90 ms after an hour, and the pose
navigation gives a ping is taken at that wrong time. What is tested here:

* the error itself, on a stream whose counter is an hour in (blocks with large sample indices), and what it does to the
  pose of a vehicle moving at 2 m/s;
* `clock_ppm` (source.clock_ppm, stream_clock_ppm; per board in source.boards.clock_ppm, stream_board_ppm): a configured
  rate error, default 0 (the stamp as before, bit for bit), unconfirmed until measured;
* the merge of per-board streams aligning boards whose clocks run at different rates by their configured rates;
* an estimate of the rate from the blocks' arrival times on the host clock (`ClockEstimator`), reported, and used for the
  stamps only when asked (`clock.discipline`).
"""

import json
import math

import numpy as np
import pytest

import gpusonar as gs
import gpusonar.sync  # noqa: F401  (gs.sync)
from gpusonar.config import load_scenario_config
from gpusonar.live import NavListener
from gpusonar.sim import run_scenario
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator, PingAssembler, make_source
from gpusonar.sources import board_merge as bm
from gpusonar.sources import flank_stream as fs

core = gs.core
OVERRIDES = {"receive": {"window": {"range": [100.0, 400.0]}}}
START_NS = 1_700_000_000_000_000_000
HOUR = 3600.0
PPM = 25.0                                  # the build specs' clock tolerance (configs/builds: boards.clock_tolerance_ppm)


def _hour_stream(sonar, ppm, times=(0.0, HOUR), block=1000):
    """Blocks of a board whose clock runs fast by `ppm`: a transmit at each true time in `times` (s after capture start),
    the board's counter at fs (1 + ppm 1e-6) per true second; only each ping's window and a block either side are sent
    (an hour of samples is not needed). Samples are zero; each block names the latest transmit at or before its end."""
    fs_ = sonar.sample_rate
    a = PingAssembler(sonar)
    w0, n = a.window_start, a.window_length
    txs = [int(math.floor(t * fs_ * (1.0 + ppm * 1e-6) + 0.5)) for t in times]
    out = []
    for tx in txs:
        first = (tx // block) * block
        while first < tx + w0 + n + block:
            named = [(t, k) for k, t in enumerate(txs) if t < first + block]
            t, k = named[-1] if named else (-1, 0)
            out.append(fs.encode_block(np.zeros((block, sonar.n_elements), dtype=np.int32), first, t, k, 0, 0, START_NS))
            first += block
    return out


def _records(sonar, blocks, cpp=False, clock_ppm=None):
    if cpp:
        template = sonar.make_record(core.Pose())
        a = core.PingAssemblerCpp(template) if clock_ppm is None else core.PingAssemblerCpp(template, clock_ppm=clock_ppm)
        return [r for b in blocks for r in a.feed(b)] + list(a.finish())
    a = PingAssembler(sonar) if clock_ppm is None else PingAssembler(sonar, clock_ppm=clock_ppm)
    return [r for b in blocks for r in a.feed(b)] + a.finish()


def _moving_vehicle(t0_ns, seconds, speed=2.0, rate=10.0):
    """A vehicle going north at `speed`, fixes at `rate` Hz on the true clock, from t0 for `seconds`."""
    nav = NavListener()
    for k in range(int(seconds * rate) + 1):
        t = t0_ns + int(k * 1e9 / rate)
        nav.push({"time_ns": t, "x": speed * (t - START_NS) * 1e-9, "y": 0.0, "z": 10.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                  "vx": speed, "vy": 0.0, "vz": 0.0})
    return nav


@pytest.mark.parametrize("cpp", [False, True])
def test_a_fast_board_clock_moves_the_stamp_and_the_pose_and_clock_ppm_takes_it_out(cpp):
    """An hour in, at 25 ppm fast: the stamp is 90 ms late and the pose of a vehicle at 2 m/s 0.18 m ahead (before, and
    still the default). With clock_ppm 25 the stamp is true to half a sample (7.8 us) and the pose to a millimetre."""
    sonar = SonarModel.load("flank_32", OVERRIDES)
    blocks = _hour_stream(sonar, PPM)
    nav = _moving_vehicle(START_NS + int((HOUR - 2.0) * 1e9), 4.0)
    true_ns = START_NS + int(HOUR * 1e9)
    late = _records(sonar, blocks, cpp)
    assert len(late) == 2
    err = late[1].transmit.time_ns - true_ns
    assert err == pytest.approx(PPM * 1e-6 * HOUR * 1e9, rel=1e-4)                  # 90.0 ms
    assert nav.at(late[1].transmit.time_ns)[0][0] - nav.at(true_ns)[0][0] == pytest.approx(0.18, abs=1e-3)
    right = _records(sonar, blocks, cpp, clock_ppm=PPM)
    assert abs(right[1].transmit.time_ns - true_ns) <= 0.5e9 / sonar.sample_rate + 1   # half a sample: the transmit is a whole sample
    assert abs(nav.at(right[1].transmit.time_ns)[0][0] - nav.at(true_ns)[0][0]) < 1e-3
    assert right[0].transmit.time_ns == late[0].transmit.time_ns == START_NS


@pytest.mark.parametrize("cpp", [False, True])
def test_clock_ppm_zero_is_the_stamp_as_before(cpp):
    """The default: capture start + round(tx / fs * 1e9), the expression of both engines before, bit for bit."""
    sonar = SonarModel.load("flank_32", OVERRIDES)
    blocks = _hour_stream(sonar, PPM, times=(0.0, 1234.567891))
    a, b = _records(sonar, blocks, cpp), _records(sonar, blocks, cpp, clock_ppm=0.0)
    tx = json.loads(a[1].metadata.extra_json)["flank_stream"]["tx_sample"] if not cpp else int(math.floor(1234.567891 * sonar.sample_rate * (1 + PPM * 1e-6) + 0.5))
    assert a[1].transmit.time_ns == b[1].transmit.time_ns == START_NS + int(round(tx / sonar.sample_rate * 1e9))


def test_both_engines_stamp_alike_with_a_rate():
    sonar = SonarModel.load("flank_32", OVERRIDES)
    for ppm in (-31.7, 3.3, 25.0, 1000.0):
        blocks = _hour_stream(sonar, ppm, times=(0.0, 777.7, 3599.9))
        a, c = _records(sonar, blocks, False, ppm), _records(sonar, blocks, True, ppm)
        assert [r.transmit.time_ns for r in a] == [r.transmit.time_ns for r in c]


def test_the_sources_take_clock_ppm_and_refuse_what_is_not_a_rate():
    src = {"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": OVERRIDES, "port": 0, "timeout": 0.2}
    s = make_source(dict(src, clock_ppm=12.5))
    assert s.assembler.clock_ppm == 12.5
    s.close()
    s = make_source(src)
    assert s.assembler.clock_ppm == 0.0
    s.close()
    for bad in (float("nan"), "25", True, 2e6):
        with pytest.raises(ValueError, match="clock_ppm"):
            make_source(dict(src, clock_ppm=bad))


# --- several boards on clocks of their own ---------------------------------------------------------------------

TRIAL = "popoto_testauv_trial_1row"
TIMES = [0.0, 1200.0, 2400.0, HOUR]             # four pings an hour apart in all, true seconds after the reference's start
RATES = {0: 25.0, 1: -20.0, 2: 10.0, 3: -5.0}  # each board's counter against true time, ppm


@pytest.fixture(scope="module")
def trial():
    sc = load_scenario_config(TRIAL)
    records = run_scenario(sc).records
    sonar = SonarModel.load(sc["sonar"], sc.get("sonar_overrides"))
    return sc, sonar, records


def _boards(sonar):
    return [{"id": i, "first": int(b["first"]), "count": int(b["count"])} for i, b in enumerate(sonar.boards)]


def _assembler(sonar, records, cpp, clock_ppm=0.0):
    c = float(records[0].environment.sound_speed)
    if cpp:
        return core.PingAssemblerCpp(sonar.make_record(core.Pose(), environment={"sound_speed": c}), clock_ppm=clock_ppm)
    return PingAssembler(sonar, sound_speed=c, clock_ppm=clock_ppm)


@pytest.mark.parametrize("cpp", [False, True])
def test_boards_on_their_own_rates_are_aligned_by_their_configured_rates(trial, cpp):
    """Four boards whose counters run at +25, -20, +10 and -5 ppm, pings over an hour. Aligned by the capture starts
    alone (boards.clock_ppm not given) board 1 is 45 ppm x 3600 s = 0.16 s off the reference at the last ping: its part
    of each span is not where the merge looks (with `missing: zero` its channels are filled with zeros, with `drop` the
    pings are dropped) and the ping cannot be synchronised. With each board's rate
    configured the merged pings are the simulated pings (to half a count), stamped at their true times to half a sample."""
    sc, sonar, records = trial
    emu = bm.BoardStreamEmulator(records, float(sc["pings"]["interval"]), _boards(sonar), start_offsets=[0, -1234, 517, -40960],
                                 start_ns=START_NS, block_samples=[512, 300, 512, 1000], clock_ppm=[RATES[i] for i in range(4)],
                                 ping_times=TIMES)
    blocks = emu.blocks()
    want = list(records)
    for configured in (False, True):
        options = {"channels": _boards(sonar)}
        if configured:
            options["clock_ppm"] = dict(RATES)
        else:
            options["missing"] = "zero"           # with "drop" pings 1 to 3 are dropped whole: board 1's parts never come
        merger = core.BoardMergerCpp(options, sonar.n_elements, sonar.sample_rate) if cpp else bm.BoardMerger.from_options(options, sonar)
        a = _assembler(sonar, records, cpp, RATES[0] if configured else 0.0)
        got = [r for _, b in blocks for m in merger.feed(b) for r in a.feed(m)]
        got += [r for m in merger.finish() for r in a.feed(m)] + list(a.finish())
        assert len(got) == 4
        lsb = records[0].metadata.adc_full_scale / 2**23
        same = [float(np.abs(g.samples.array - w.samples.array).max()) <= 0.51 * lsb + 1e-12 for g, w in zip(got, want)]
        if configured:
            assert all(same)
            for g, t in zip(got, TIMES):
                assert abs(g.transmit.time_ns - (START_NS + int(round(t * 1e9)))) <= 0.5e9 / sonar.sample_rate + 1
            gs.sync.estimate_clocks_reference(got[3], sonar.config["receive"]["reference"])     # every burst found
        else:
            assert same[0] and not same[3]
            stats = json.loads(merger.stats_json()) if cpp else merger.stats()
            assert stats["boards"]["1"]["missing_samples"] > 0 and stats["zero_filled_blocks"] > 0
            assert got[3].transmit.time_ns - (START_NS + int(HOUR * 1e9)) == pytest.approx(PPM * 1e-6 * HOUR * 1e9, rel=1e-3)
            reference = sonar.config["receive"]["reference"]
            with pytest.raises(gs.sync.SyncError, match="not found"):
                gs.sync.estimate_clocks_reference(got[3], reference)


def test_both_engines_merge_boards_on_their_own_rates_alike(trial):
    sc, sonar, records = trial
    emu = bm.BoardStreamEmulator(records, float(sc["pings"]["interval"]), _boards(sonar), start_offsets=[0, -1234, 517, -40960],
                                 start_ns=START_NS, clock_ppm=[RATES[i] for i in range(4)], ping_times=TIMES)
    options = {"channels": _boards(sonar), "clock_ppm": {0: 25.0, 1: -20.0, 2: 10.0, 3: -5.0}}
    py, cc = bm.BoardMerger.from_options(options, sonar), core.BoardMergerCpp(options, sonar.n_elements, sonar.sample_rate)
    a = [m for _, b in emu.blocks() for m in py.feed(b)] + py.finish()
    c = [m for _, b in emu.blocks() for m in cc.feed(b)] + cc.finish()
    assert len(a) == len(c) and all(x == y for x, y in zip(a, c))
    assert py.stats() == json.loads(cc.stats_json())


@pytest.mark.parametrize("cpp", [False, True])
def test_a_rate_for_a_board_not_in_the_map_or_not_a_number_is_refused(trial, cpp):
    sc, sonar, records = trial
    make = (lambda o: core.BoardMergerCpp(o, sonar.n_elements, sonar.sample_rate)) if cpp else (lambda o: bm.BoardMerger.from_options(o, sonar))
    with pytest.raises(ValueError, match="clock_ppm.*board 7"):
        make({"channels": _boards(sonar), "clock_ppm": {7: 1.0}})
    with pytest.raises(ValueError, match="clock_ppm"):
        make({"channels": _boards(sonar), "clock_ppm": {1: float("nan")}})


# --- the rate from the blocks' arrival times -----------------------------------------------------------------------

def _arrivals(ppm, seconds, fs_=102400.0, block=1024, seed=0, floor_ms=0.5, mean_ms=0.3, spikes=0.01, spike_ms=(5.0, 15.0)):
    """(sample after each block, arrival ns): a board at `ppm`, its block sent when its last sample is taken and
    arriving after a one-sided delay: a floor, an exponential part and, in a fraction `spikes` of blocks, a stall
    (ours: the shape of the receive delays measured on L4 loopback, D-162, worst 1-2 ms with the chain's threads capped
    and 10-13 ms without; never measured on a vehicle's network)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs_ / block)
    ends = (np.arange(n, dtype=np.int64) + 1) * block
    true_s = ends / (fs_ * (1.0 + ppm * 1e-6))
    delay = floor_ms + rng.exponential(mean_ms, n) + np.where(rng.random(n) < spikes, rng.uniform(*spike_ms, n), 0.0)
    return ends, (START_NS + np.rint((true_s + delay * 1e-3) * 1e9)).astype(np.int64)


@pytest.mark.parametrize("ppm", [25.0, -21.0, 0.0])
def test_the_rate_is_estimated_from_arrival_times_alike_in_both_engines(ppm):
    """Ten minutes of blocks with delays up to 15 ms: the slope of each second's earliest arrival gives the rate to a
    few hundredths of a ppm (synthetic delays; the real network's are not known), and both engines give the same
    numbers."""
    ends, arrive = _arrivals(ppm, 600.0)
    py = bm.ClockEstimator(102400.0)
    cc = core.ClockEstimatorCpp(102400.0)
    for e, t in zip(ends.tolist(), arrive.tolist()):
        py.feed(e, START_NS, t)
        cc.feed(e, START_NS, t)
    a, c = py.estimate(), json.loads(cc.estimate_json())
    assert a == c
    assert a["ppm"] == pytest.approx(ppm, abs=0.05) and a["sigma_ppm"] < 0.05 and a["span_s"] > 590.0 and a["bins"] >= 590


def test_the_estimate_needs_a_long_enough_span_and_starts_again_with_a_capture():
    est = bm.ClockEstimator(102400.0)
    assert est.estimate()["ppm"] is None
    ends, arrive = _arrivals(25.0, 2.5)
    for e, t in zip(ends.tolist(), arrive.tolist()):
        est.feed(e, START_NS, t)
    assert est.estimate()["bins"] == 3 and est.estimate()["ppm"] is not None
    est.feed(1024, START_NS + 5, int(arrive[-1]) + 10)            # another capture start: the old bins are of another counter
    assert est.estimate()["bins"] == 1 and est.estimate()["ppm"] is None and est.restarts == 1


@pytest.mark.parametrize("cpp", [False, True])
def test_discipline_stamps_with_the_estimate_only_when_asked_and_when_it_is_good(cpp):
    """`clock.discipline`: once the estimate spans `min_span_s` with a sigma under `max_sigma_ppm`, the stamps use it
    in place of clock_ppm. Off (the default), the estimate is only reported."""
    sonar = SonarModel.load("flank_32", OVERRIDES)
    ends, arrive = _arrivals(PPM, 600.0, fs_=sonar.sample_rate)
    blocks = _hour_stream(sonar, PPM)
    true_ns = START_NS + int(HOUR * 1e9)
    for discipline, want_ms in ((False, 90.0), (True, 0.0)):
        options = {"discipline": discipline}
        if cpp:
            clock = core.ClockEstimatorCpp(sonar.sample_rate, options)
        else:
            clock = bm.ClockEstimator.from_options(options, sonar.sample_rate)
        for e, t in zip(ends.tolist(), arrive.tolist()):
            clock.feed(e, START_NS, t)
        ppm = clock.stamp_ppm(0.0)                                    # what the assembler is given
        recs = _records(sonar, blocks, cpp, clock_ppm=ppm)
        assert (recs[1].transmit.time_ns - true_ns) * 1e-6 == pytest.approx(want_ms, abs=0.2)


def test_the_flank_udp_source_reports_the_estimate():
    src = {"type": "flank_udp", "sonar": "flank_32", "sonar_overrides": OVERRIDES, "port": 0, "timeout": 0.2}
    s = make_source(dict(src, clock={}))
    try:
        assert s.clock_stats()["ppm"] is None
    finally:
        s.close()
    s = make_source(src)
    assert s.clock_stats() is None
    s.close()
    with pytest.raises(ValueError, match="source.clock.dicipline"):
        make_source(dict(src, clock={"dicipline": True}))


def test_source_clock_keys_reach_the_program():
    from gpusonar import live
    flat = live.program_options({}, None, {"source": {"clock_ppm": 12.5, "clock": {"discipline": True, "window_s": 300.0},
                                                      "boards": {"clock_ppm": {0: 1.5, 3: -2.25}}}})
    got = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    assert got["stream_clock_ppm"] == 12.5 and got["stream_board_ppm"] == "0:1.5,3:-2.25"
    assert got["stream_clock_estimate"] is True and got["stream_clock_discipline"] is True and got["stream_clock_window_s"] == 300.0
    assert not any(k.startswith("stream_clock") for k in live.program_options({}, None, {"source": {}}))
    with pytest.raises(ValueError, match="source.clock.windows_s"):
        live.program_options({}, None, {"source": {"clock": {"windows_s": 1.0}}})


EXE = __import__("pathlib").Path(__file__).resolve().parents[2] / "build" / "bin" / "sonar_live"


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live is not built")
def test_the_program_stamps_a_fast_clock_by_stream_clock_ppm(tmp_path):
    """build/bin/sonar_live on the hour-in stream of a board 25 ppm fast: the report's time_ns is 90 ms late without
    the key (as before) and true to half a sample with stream_clock_ppm=25; stream_clock_estimate=1 prints the estimate."""
    import socket
    import subprocess
    import time
    sonar = SonarModel.load("flank_32", OVERRIDES)
    template = tmp_path / "t.h5"
    w = core.RecordWriter(str(template))
    w.write(sonar.make_record(core.Pose()))
    w.close()
    blocks = _hour_stream(sonar, PPM, block=500)
    true_ns = START_NS + int(HOUR * 1e9)
    times = {}
    for name, extra in (("default", []), ("ppm", ["stream_clock_ppm=25", "stream_clock_estimate=1"])):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        report = tmp_path / f"{name}.jsonl"
        proc = subprocess.Popen([str(EXE), "--udp", str(port), "--template", str(template), "--report", str(report), "--pings", "2",
                                 "idle_seconds=5", "bearing_step_deg=2", *extra], stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(1.0)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                for k, b in enumerate(blocks):
                    s.sendto(b, ("127.0.0.1", port))
                    if k % 4 == 3:
                        time.sleep(0.002)
            _, err = proc.communicate(timeout=120.0)
        finally:
            proc.kill()
        assert proc.returncode == 0, err[-400:]
        lines = [json.loads(x) for x in report.read_text().splitlines()]
        if len(lines) < 2:
            pytest.skip(f"the loopback dropped data: {err.strip().splitlines()[-1]}")
        times[name] = (lines[1]["time_ns"], err)
    assert (times["default"][0] - true_ns) * 1e-6 == pytest.approx(90.0, abs=0.01)
    assert abs(times["ppm"][0] - true_ns) <= 0.5e9 / sonar.sample_rate + 1
    assert "sonar_live: clock {" in times["ppm"][1] and "sonar_live: clock" not in times["default"][1]
