"""Decimate first: the boards' real stream band-selected and decimated to complex baseband
before the chain (libs/live decimate.hpp).

The filter against what was asked of it and against a direct NumPy evaluation of the same
definition; tones in and out of the band (closed form); streaming against whole-ping; the
board clocks estimated from the raw stretches kept around the reference bursts against the
full-rate synchronisation, and the baseband correction against synchronising first; the
chain's results through the stage against the full-rate path on the one-, two- and three-row
stage scenarios with board clocks; both receivers (Python and C++ assemblers); listening;
recordings raw and decimated; and the latency the stage adds.
"""

import json
import pathlib
import subprocess

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import live, recon, sync
from gpusonar.config import load_scenario_config, load_yaml
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, run_scenario, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator, PingAssembler

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "bin" / "sonar_live"


def raw_ping(sonar="popoto_testauv_1row_clocks", window=(300.0, 420.0), target=(40.0, 350.0, 5.0), noise=0.0, seed=0, **over):
    model = SonarModel.load(sonar, dict({"receive": {"window": {"range": list(window)}}}, **over))
    rec = simulate_ping(model, make_pose(position=(0.0, 0.0, 10.0)), Scene().add_points([list(target)], target_strength_db=10.0),
                        environment={"sound_speed": 1500.0}, noise_rms=noise, rng=np.random.default_rng(seed), adc=False)
    return model, rec


def response_db(taps, fs, freqs):
    n = np.arange(len(taps))
    return 20.0 * np.log10(np.abs(np.array([np.sum(taps * np.exp(-2j * np.pi * f / fs * n)) for f in freqs])) + 1e-300)


# --- the filter -----------------------------------------------------------------------------------

def test_plan_follows_the_sonars_band_and_rate_and_the_filter_meets_what_was_asked():
    model, rec = raw_ping()
    w, fs = rec.transmit.waveform, rec.samples.sample_rate
    band = abs(w.f_stop - w.f_start)
    plan = core.live_decimate_plan(rec)
    # Defaults: the centre of the transmit band, 1.5 x the band kept, at least twice that as the output rate.
    assert plan["carrier"] == pytest.approx(0.5 * (w.f_start + w.f_stop)) and plan["passband"] == pytest.approx(0.75 * band)
    assert plan["factor"] == int(fs // (2.0 * 1.5 * band)) and plan["output_rate"] == pytest.approx(fs / plan["factor"])
    assert plan["output_rate"] >= 3.0 * band and plan["stopband"] == pytest.approx(plan["output_rate"] - plan["passband"])
    taps = np.array(plan["taps"])
    assert len(taps) % 2 == 1 and np.allclose(taps, taps[::-1]) and taps.sum() == pytest.approx(1.0)
    assert plan["delay_s"] == pytest.approx(0.5 * (len(taps) - 1) / fs)
    # Measured: ripple in the kept band and rejection of everything that would alias onto it.
    in_band = response_db(taps, fs, np.linspace(0.0, plan["passband"], 200))
    stop = response_db(taps, fs, np.linspace(plan["stopband"], 0.5 * fs, 4000))
    assert in_band.max() - in_band.min() <= 0.1 and stop.max() <= -79.0
    # Asked for something else, it gives that: a rate (never below it), a rejection, a length.
    low = core.live_decimate_plan(rec, {"output_rate": 4000.0})
    assert low["factor"] == 25 and low["output_rate"] == pytest.approx(4096.0) and len(low["taps"]) > len(taps)      # a narrower transition
    assert response_db(np.array(low["taps"]), fs, np.linspace(low["stopband"], 0.5 * fs, 4000)).max() <= -79.0
    weak = core.live_decimate_plan(rec, {"stopband_db": 50.0, "ripple_db": 0.5})
    assert len(weak["taps"]) < len(taps)
    assert response_db(np.array(weak["taps"]), fs, np.linspace(weak["stopband"], 0.5 * fs, 4000)).max() <= -49.0
    assert len(core.live_decimate_plan(rec, {"taps": 99})["taps"]) == 99
    # Another band than the transmit's (listening), and another sonar altogether: nothing is built in.
    other = core.live_decimate_plan(rec, {"f_low": 8000.0, "f_high": 9000.0})
    assert other["carrier"] == 8500.0 and other["factor"] == int(fs // 3000.0)
    flank = SonarModel.load("flank_32").make_record(make_pose())
    p2 = core.live_decimate_plan(flank)
    wf = flank.transmit.waveform
    assert p2["input_rate"] == flank.samples.sample_rate and p2["carrier"] == pytest.approx(wf.center())
    assert p2["factor"] == int(flank.samples.sample_rate // (3.0 * wf.bandwidth()))
    with pytest.raises(ValueError):
        core.live_decimate_plan(rec, {"f_low": 60000.0, "f_high": 70000.0})      # above half the board rate
    with pytest.raises(ValueError):
        core.live_decimate_plan(rec, {"output_rate": 1000.0})                    # below the band kept


def test_output_is_the_definition_evaluated_directly_and_streaming_equals_whole_ping():
    _, rec = raw_ping(noise=1e-6)
    x = np.asarray(rec.samples.array, dtype=np.float64)
    fs, t0 = rec.samples.sample_rate, rec.samples.start_time
    plan = core.live_decimate_plan(rec)
    taps, d, fc = np.array(plan["taps"]), plan["factor"], plan["carrier"]
    whole = rec.copy()
    core.live_decimate_record(whole)
    s = whole.samples
    assert s.is_complex() and s.sample_rate == pytest.approx(fs / d) and s.center_frequency == fc and s.start_time == t0
    assert s.n_samples == (x.shape[1] - 1) // d + 1
    # Mix to the band's centre (phase from the transmit instant), low-pass, keep every d-th sample; zeros outside the window.
    t = t0 + np.arange(x.shape[1]) / fs
    mixed = 2.0 * x[5] * np.exp(-2j * np.pi * fc * t)
    want = np.convolve(mixed, taps, mode="same")[::d]
    got = np.asarray(s.array)[5]
    assert np.abs(got - want).max() <= 2e-5 * np.abs(want).max()
    # Fed in pieces of any size, as blocks arrive: the same samples, exactly.
    dec = core.PingDecimator(rec)
    rows = np.ascontiguousarray(x.T.astype(np.float32))
    at = 0
    for n in (1, 7, 128, 1000, 3, 4096, 10**9):
        dec.feed(rows[at:at + n])
        at += n
    like = rec.copy()
    dec.finish(like)
    assert np.array_equal(np.asarray(like.samples.array), np.asarray(s.array))
    dec.open()                                                                    # and again for the next ping
    dec.feed(rows)
    again = rec.copy()
    dec.finish(again)
    assert np.array_equal(np.asarray(again.samples.array), np.asarray(s.array))
    # A complex record is left as it is (the stage runs once).
    before = np.asarray(whole.samples.array).copy()
    core.live_decimate_record(whole)
    assert np.array_equal(np.asarray(whole.samples.array), before)


def test_tones_in_the_band_come_through_and_tones_outside_it_do_not():
    model, rec = raw_ping()
    fs, n, t0 = rec.samples.sample_rate, rec.samples.n_samples, rec.samples.start_time
    plan = core.live_decimate_plan(rec)
    t = t0 + np.arange(n) / fs

    def through(f, phase=0.3):
        r = rec.copy()
        s = r.samples
        kind = s.kind
        x = np.zeros((s.n_channels, n), dtype=np.float32)
        x[:] = np.cos(2.0 * np.pi * f * t + phase)
        s.array = x
        s.kind = kind
        r.samples = s
        core.live_decimate_record(r)
        y = np.asarray(r.samples.array)[0]
        tm = t0 + np.arange(len(y)) / r.samples.sample_rate
        return y[40:-40], np.exp(1j * (2.0 * np.pi * (f - plan["carrier"]) * tm + phase))[40:-40]

    for f in (plan["carrier"], plan["carrier"] - 1000.0, plan["carrier"] + 1000.0, plan["carrier"] + 0.95 * plan["passband"]):
        y, want = through(f)
        assert np.abs(y - want).max() < 0.01                                      # unit amplitude, the phase of the transmit clock
    # The synchronisation reference (above the acoustic band) and a tone that would alias onto the band: gone.
    ref = model.config["receive"]["reference"]
    for f in (0.5 * (ref["f_start"] + ref["f_stop"]), plan["carrier"] + plan["output_rate"], plan["carrier"] - plan["stopband"] - 50.0,
              plan["carrier"] + plan["stopband"] + 50.0):
        assert np.abs(through(f)[0]).max() < 10.0 ** (-73.0 / 20.0)        # a real tone is two lines, each 80 dB down


def test_latency_is_half_the_filter():
    _, rec = raw_ping()
    plan = core.live_decimate_plan(rec)
    half = (len(plan["taps"]) - 1) // 2
    dec = core.PingDecimator(rec)
    rows = np.zeros((half, rec.samples.n_channels), dtype=np.float32)
    dec.feed(rows)                                                                # one sample short of the first output
    like = rec.copy()
    # An output sample needs `half` input samples after its own: 0.83 ms at 102.4 kHz with the default filter.
    assert plan["delay_s"] == pytest.approx(half / rec.samples.sample_rate) and plan["delay_s"] < 1.5e-3
    dec.finish(like)
    assert like.samples.n_samples == (half - 1) // plan["factor"] + 1


# --- board clocks ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clocked():
    """One ping of four boards on their own clocks (offsets to 0.3 ms, drifts to 25 ppm), raw, with the shared reference."""
    model, rec = raw_ping(noise=2e-6, window=(300.0, 420.0))
    assert len(rec.metadata.boards) == 4 and not rec.samples.is_complex()
    return model, rec


def test_clocks_from_the_kept_stretches_are_the_full_rate_estimate_and_the_baseband_correction_matches(clocked):
    model, rec = clocked
    reference = model.config["receive"]["reference"]
    truth_offset = np.array([b["clock_offset"] for b in model.config["receive"]["boards"]])
    truth_drift = np.array([b["clock_drift_ppm"] for b in model.config["receive"]["boards"]])
    # Full rate, as the chain has done it so far (both engines).
    a = rec.copy()
    rep = sync.synchronise(a, reference)
    b = rec.copy()
    cpp = core.live_synchronise(b, reference)
    # The stage: estimate from the raw stretches around the two bursts, correct the baseband samples.
    c = rec.copy()
    got = core.live_decimate_record(c, {}, reference)
    assert got["ok"] and rep.ok and cpp["ok"]
    assert np.allclose(got["offset"], cpp["offset"], atol=1e-10) and np.allclose(got["drift_ppm"], cpp["drift_ppm"], atol=1e-4)
    assert np.allclose(got["offset"], rep.estimate.offset, atol=2e-8) and np.allclose(got["residual"], cpp["residual"], atol=2e-8)
    assert np.allclose(got["offset"], truth_offset - truth_offset[0], atol=1e-6)
    assert np.allclose(got["drift_ppm"], truth_drift - truth_drift[0], atol=1.0)
    assert [bd.synchronised for bd in c.metadata.boards] == [1, 1, 1, 1]
    assert c.metadata.boards[1].clock_offset == pytest.approx(got["offset"][1])
    facts = live.decimated_sync(c)
    assert facts["ok"] and facts["worst_residual_us"] == pytest.approx(max(got["residual"]) * 1e6, rel=1e-4) and "flagged" not in facts
    # Correcting after decimation gives what decimating the corrected ping gives: the echo's samples to a part in a thousand.
    core.live_decimate_record(b)
    y_stage, y_first = np.asarray(c.samples.array), np.asarray(b.samples.array)
    recon.prepare([c])
    recon.prepare([b])
    z_stage, z_first = np.asarray(c.samples.array), np.asarray(b.samples.array)
    peak = np.abs(z_first).max()
    assert np.abs(z_stage - z_first).max() < 2e-3 * peak
    # Without the correction the boards do not line up: that is what the stage is there to prevent.
    d = rec.copy()
    core.live_decimate_record(d)
    recon.prepare([d])
    assert np.abs(np.asarray(d.samples.array) - z_first).max() > 0.3 * peak
    # And after decimation there is nothing left to synchronise on: the reference is above the band and the record complex.
    assert y_stage.shape == y_first.shape
    with pytest.raises(sync.SyncError):
        sync.synchronise(d, reference)


def strong(report, floor=30.0):   # well clear of the detector's threshold: a sidelobe at 25 dB in one path can be 24.9 in the other
    return sorted((d for d in report["detections"] if d["snr_db"] is not None and d["snr_db"] > floor), key=lambda d: -d["snr_db"])


TOLERANCE = {"range_m": 0.5, "bearing_deg": 0.05, "snr_db": 0.3, "elevation_deg": 0.05, "depth_m": 0.3, "depth_sigma_m": 0.05,
             "elevation_sigma_deg": 0.01, "cone_bearing_deg": 0.05, "image_depth_m": 0.3}


def assert_same_results(full, dec, n_boards):
    """Detections that stand clear of the noise, the tracks they make, and the clocks: the same through both paths."""
    assert len(full) == len(dec)
    compared = 0
    for a, b in zip(full, dec):
        assert a["sync"]["ok"] and b["sync"]["ok"] and len(b["sync"]["offsets_us"]) == n_boards
        assert np.allclose(a["sync"]["offsets_us"], b["sync"]["offsets_us"], atol=0.02)
        assert np.allclose(a["sync"]["drift_ppm"], b["sync"]["drift_ppm"], atol=0.01)
        assert a["sync"]["worst_residual_us"] == pytest.approx(b["sync"]["worst_residual_us"], abs=0.02)
        sa, sb = strong(a), strong(b)
        assert len(sa) == len(sb) >= 1
        for da in sa:
            db = min(sb, key=lambda q: abs(q["range_m"] - da["range_m"]) + 20.0 * abs(q["bearing_deg"] - da["bearing_deg"]))
            for key, tol in TOLERANCE.items():
                if key in da and da[key] is not None:
                    assert db[key] == pytest.approx(da[key], abs=tol), key
            for key in ("path", "elevation_ambiguous"):
                assert db.get(key) == da.get(key)
            compared += 1
        ta = sorted((t for t in a["tracks"] if t["mean_snr_db"] > 30.0), key=lambda t: t["range_m"])
        tb = sorted((t for t in b["tracks"] if t["mean_snr_db"] > 30.0), key=lambda t: t["range_m"])
        assert [t["kind"] for t in ta] == [t["kind"] for t in tb]
        for x, y in zip(ta, tb):
            assert np.hypot(x["x"] - y["x"], x["y"] - y["y"]) < max(1.0, 1e-3 * x["range_m"])     # 0.06 degrees of bearing
            if x.get("depth_m") is not None:
                assert y["depth_m"] == pytest.approx(x["depth_m"], abs=0.3)
    return compared


STAGES = [("popoto_testauv_trial_1row", "popoto_testauv_1row", 4), ("popoto_static_trial_2row", "popoto_static_2row", 8),
          ("popoto_static_trial_3row", "popoto_static_3row", 12)]
_records = {}


def stage_records(scenario):
    if scenario not in _records:
        _records[scenario] = run_scenario(scenario).records[:3]
    return _records[scenario]


@pytest.mark.parametrize("scenario,deploy,n_boards", STAGES)
def test_chain_results_through_the_stage_equal_the_full_rate_path(scenario, deploy, n_boards):
    """One, two and three rows, raw samples at the boards' rate, every board on its own clock: detections, bearings,
    elevations, depths, tracks and the clock estimates, within the tolerances above. C++ engine."""
    recs = stage_records(scenario)
    chain = load_yaml(ROOT / "configs" / "deploy" / f"{deploy}.yaml")["chain"]
    chain["tracker"] = dict(chain["tracker"], confirm_hits=2)
    if chain.get("elevation"):
        water = load_scenario_config(scenario)["environment"].get("water_depth")
        chain["elevation"] = dict(chain["elevation"], gradient_per_s=None, water_depth=water)
    out = {}
    for label, extra in (("full", {}), ("decimated", {"decimate": {}})):
        with LiveChain(dict(chain, engine="cpp", backend="host", **extra)) as c:
            out[label] = c.run([r.copy() for r in recs])
    assert assert_same_results(out["full"], out["decimated"], n_boards) >= len(recs)
    assert all(not r.samples.is_complex() for r in recs)                         # the raw pings are what was fed, both times
    if n_boards == 4:
        # The Python engine through the same stage, and a coarser output rate (a 25th of the board rate).
        with LiveChain(dict(chain, engine="python", backend="host", decimate={})) as c:
            py = c.run([r.copy() for r in recs[:2]])
        assert assert_same_results(out["full"][:2], py, n_boards) >= 2
        with LiveChain(dict(chain, engine="cpp", backend="host", decimate={"output_rate": recs[0].samples.sample_rate / 25.0})) as c:
            coarse = c.run([r.copy() for r in recs[:2]])
        for a, b in zip(out["full"][:2], coarse):                                # the same detections; looser on level and bearing
            da, db = strong(a)[0], strong(b)[0]
            assert db["range_m"] == pytest.approx(da["range_m"], abs=0.5) and db["bearing_deg"] == pytest.approx(da["bearing_deg"], abs=0.25)
            assert db["snr_db"] == pytest.approx(da["snr_db"], abs=1.0)


# --- the receivers --------------------------------------------------------------------------------

def test_both_receivers_decimate_as_blocks_arrive_and_the_chain_takes_their_records(clocked, tmp_path):
    model, rec = clocked
    reference = model.config["receive"]["reference"]
    second = rec.copy()
    m = second.metadata
    m.sequence = 1
    second.metadata = m
    blocks = list(FlankBoardEmulator([rec, second], ping_interval=1.0, block_samples=96).blocks())
    raw_asm = PingAssembler(model)
    raw = [r for b in blocks for r in raw_asm.feed(b)]
    py_asm = PingAssembler(model, decimate={}, reference=reference)
    py = [r for b in blocks for r in py_asm.feed(b)]
    cpp_asm = core.PingAssemblerCpp(model.make_record(make_pose()), {}, reference)
    cpp = [r for b in blocks for r in cpp_asm.feed(b)]
    assert len(raw) == len(py) == len(cpp) == 2 and py_asm.decimation["factor"] == 17
    for r_raw, r_py, r_cpp in zip(raw, py, cpp):
        batch = r_raw.copy()
        core.live_decimate_record(batch, {}, reference)                          # the raw ping, assembled first, then the stage
        want = np.asarray(batch.samples.array)
        for got in (r_py, r_cpp):
            assert got.samples.is_complex() and got.samples.sample_rate == batch.samples.sample_rate
            assert np.abs(np.asarray(got.samples.array) - want).max() <= 1e-6 * np.abs(want).max()
            assert live.decimated_sync(got)["ok"]
        assert r_py.metadata.sequence == r_raw.metadata.sequence and r_py.transmit.time_ns == r_raw.transmit.time_ns
        assert json.loads(r_py.metadata.extra_json)["flank_stream"]["tx_sample"] == json.loads(r_raw.metadata.extra_json)["flank_stream"]["tx_sample"]
    # The chain on the receivers' records (already complex, already synchronised) against the chain on the raw pings.
    chain = {"sync": {"reference": reference}, "bearings_deg": [-50.0, 50.0, 0.5], "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 60.0},
             "min_snr_db": 15.0, "backend": "host", "tracker": {"confirm_hits": 2}}
    for engine in ("cpp", "python"):
        with LiveChain(dict(chain, engine=engine)) as c:
            full = c.run([r.copy() for r in raw])
        with LiveChain(dict(chain, engine=engine)) as c:
            dec = c.run([r.copy() for r in (cpp if engine == "cpp" else py)])
        assert assert_same_results(full, dec, 4) >= 2
    # Recordings: raw (what the chain decimates itself) and decimated (what a decimating receiver hands on) both replay.
    sizes = {}
    for label, records in (("raw", raw), ("decimated", py)):
        path = tmp_path / f"{label}.h5"
        w = gs.RecordWriter(str(path))
        for r in records:
            w.write(r)
        w.close()
        sizes[label] = path.stat().st_size
        back = gs.RecordReader(str(path))
        assert np.allclose(np.asarray(back.read(0).samples.array), np.asarray(records[0].samples.array), rtol=1e-6, atol=1e-12)
        with LiveChain(dict(chain, engine="cpp", decimate={})) as c:
            replayed = c.run([back.read(i) for i in range(2)])
        assert assert_same_results(full, replayed, 4) >= 2
    # 4 bytes a raw sample against 8 bytes for one complex sample in 17: 8.5 times less, less the headers.
    assert 6.0 < sizes["raw"] / sizes["decimated"] < 8.6


def test_listening_on_the_decimated_stream_gives_the_bearings_of_the_raw_one():
    from gpusonar.sim.passive import passive_record
    sonar = SonarModel.load("popoto_long_range", {"receive": {"window": {"range": [0.0, 400.0]}}})
    pose = make_pose(position=(0.0, 0.0, 30.0))
    sources = [[-300.0, 900.0, 30.0], [400.0, 800.0, 30.0]]
    rec = passive_record(sonar, pose, [{"position": sources[0], "rms": 1.0e-4}, {"position": sources[1], "rms": 0.7e-4}], noise_rms=1.0e-4,
                         rng=np.random.default_rng(5))
    assert not rec.samples.is_complex() and rec.samples.sample_rate == 102400.0
    centre = rec.rx_positions().mean(axis=0)
    want = sorted(float(np.degrees(np.arctan2(q[0] - centre[0], q[1] - centre[1]))) for q in sources)
    kw = dict(first_deg=-60.0, last_deg=60.0, max_sources=2, min_snr_db=3.0, n_average=8)
    raw_frames, _ = core.doa_passive(rec, 4000.0, 6000.0, n_fft=4096, hop=2048, **kw)
    # The same stretch through the stream assembler, decimating as it goes (frames of 256 decimated samples).
    plan = core.live_decimate_plan(rec, {"f_low": 4000.0, "f_high": 6000.0})
    chunk = 256 + 7 * 128
    blocks = list(FlankBoardEmulator([rec], ping_interval=rec.samples.n_samples / rec.samples.sample_rate, block_samples=128).blocks())
    asm = core.StreamAssemblerCpp(sonar.make_record(pose), chunk, {"f_low": 4000.0, "f_high": 6000.0})
    chunks = [r for b in blocks for r in asm.feed(b)]
    assert len(chunks) == (rec.samples.n_samples - len(plan["taps"]) // 2) // plan["factor"] // chunk
    assert chunks[0].samples.is_complex() and chunks[0].samples.n_samples == chunk and chunks[0].samples.center_frequency == 5000.0
    step_ns = chunks[1].transmit.time_ns - chunks[0].transmit.time_ns
    assert step_ns == pytest.approx(chunk * plan["factor"] / 102400.0 * 1e9, abs=2)
    found = 0
    for c in chunks:
        frames, _ = core.doa_passive(c, 4000.0, 6000.0, n_fft=256, hop=128, **kw)
        for f in frames:
            got = sorted(float(np.degrees(p[0])) for p in f["peaks"])
            if len(got) == 2:
                assert np.allclose(got, want, atol=0.5)
                found += 1
    both_raw = [sorted(float(np.degrees(p[0])) for p in f["peaks"]) for f in raw_frames if len(f["peaks"]) == 2]
    assert found >= len(chunks) - 1 and len(both_raw) >= 1 and np.allclose(both_raw[0], want, atol=0.5)
    # A gap in the stream restarts the filter and drops what was waiting, as a gap drops a raw chunk.
    asm2 = core.StreamAssemblerCpp(sonar.make_record(pose), chunk, {"f_low": 4000.0, "f_high": 6000.0})
    n_gap = sum(len(asm2.feed(b)) for i, b in enumerate(blocks) if i != 40)
    assert n_gap < len(chunks) and asm2.dropped_chunks == 1


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")
def test_program_decimates_a_replayed_raw_recording(clocked, tmp_path):
    model, rec = clocked
    ref = model.config["receive"]["reference"]
    path = tmp_path / "raw.h5"
    w = gs.RecordWriter(str(path))
    w.write(rec)
    w.close()
    args = [str(EXE), "--replay", str(path), "bearing_first_deg=-50", "bearing_last_deg=50", "cfar_pfa=1e-6", "cfar_guard_m=20", "cfar_train_m=60",
            "min_snr_db=15", "sync=1", f"sync_f_start={ref['f_start']}", f"sync_f_stop={ref['f_stop']}", f"sync_duration={ref['duration']}",
            f"sync_lead={ref['lead']}", f"sync_tail={ref['tail']}"]
    full = [json.loads(line) for line in subprocess.run(args, check=True, capture_output=True, text=True).stdout.splitlines()]
    dec = [json.loads(line) for line in subprocess.run(args + ["decimate=1"], check=True, capture_output=True, text=True).stdout.splitlines()]
    assert assert_same_results(full, dec, 4) == 1
    coarse = subprocess.run(args + ["decimate=1", "decimate_output_rate=4096", "decimate_stopband_db=70"], check=True, capture_output=True, text=True)
    assert len(json.loads(coarse.stdout.splitlines()[0])["detections"]) >= 1


def test_deployment_keys_reach_both_engines_and_the_exported_command(tmp_path):
    cfg = {"source": {"type": "flank_udp", "sonar": "popoto_4x8", "port": 5700, "decimate": {"output_rate": 4096.0}},
           "nav": {"port": 5702, "water_depth": {"source": "altitude", "hold_s": 30.0}, "keep_s": 60.0},
           "chain": {"sync": {"sonar": "popoto_4x8"}, "blanking": {"recovery_s": 0.02, "taper_s": 0.005},
                     "elevation": {"profile": {"built": {"sample_sigma_m_s": 0.1}, "family": {"kind": "isothermal", "gradient_per_s": 0.017}}}}}
    o = live.cpp_chain_options(dict(cfg["chain"], decimate={"stopband_db": 70.0}), live.sync_reference(cfg["chain"]))
    assert o["decimate"] == {"stopband_db": 70.0} and o["blanking"] == {"recovery_s": 0.02, "taper_s": 0.005, "mode": "zero"}
    core.LiveChainCpp(o)                                                          # the C++ chain takes them
    with pytest.raises(ValueError):
        live.cpp_chain_options({"decimate": {"rate": 1.0}})
    import sys
    import yaml
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    cmd = subprocess.run([sys.executable, str(ROOT / "tools" / "sonar_live.py"), str(path), "--export-template", str(tmp_path / "t.h5")],
                         check=True, capture_output=True, text=True).stdout
    for want in ("decimate=1", "decimate_streaming=1", "decimate_output_rate=4096", "blanking=1", "blanking_recovery_s=0.02",
                 "nav_water_depth_from_altitude=1", "nav_hold_s=30", "nav_keep_s=60", "elevation_profile=1",
                 "elevation_profile_sample_sigma=0.1", "elevation_profile_family=isothermal", "elevation_profile_ladder=supplied,built,family,local,prior"):
        assert want in cmd, want
    if EXE.exists():                                                             # and the program parses every one of them
        args = [a for a in cmd.split()[5:] if "=" in a and not a.startswith("--")]
        bad = subprocess.run([str(EXE), "--replay", "/nonexistent.h5"] + args, capture_output=True, text=True)
        assert "unknown option" not in bad.stderr
