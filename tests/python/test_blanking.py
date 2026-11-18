"""Blanking the projector's own sound (chain.blanking), and board synchronisation when that
sound has clipped a reference burst (chain.sync min_quality).

The projector is a metre or two from the hydrophones: its direct sound is far above the
inputs' full scale for the whole pulse, and the inputs take a while to recover. The simulator
has no overload model yet, so these tests clip the samples themselves: the transmit chirp at
24 V arriving a millisecond after the transmit, the inputs then stuck at the rail and
decaying, everything clipped at the ADC's full scale.
"""

import json
import pathlib

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import sync
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sources import FlankBoardEmulator

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
C = 1500.0
CHAIN = {"bearings_deg": [-50.0, 50.0, 0.5], "cfar": {"pfa": 1e-6, "guard_m": 20.0, "train_m": 60.0}, "min_snr_db": 12.0, "backend": "host",
         "tracker": {"confirm_hits": 2}}
BLANK = {"recovery_s": 0.03, "taper_s": 0.005}
TARGET = (60.0, 400.0, 5.0)


def ping(sonar, window=(50.0, 600.0), noise=2e-6, seed=0, **over):
    model = SonarModel.load(sonar, dict({"receive": {"window": {"range": list(window)}}}, **over))
    rec = simulate_ping(model, make_pose(position=(0.0, 0.0, 10.0)), Scene().add_points([list(TARGET)], target_strength_db=10.0),
                        environment={"sound_speed": C}, noise_rms=noise, rng=np.random.default_rng(seed), adc=False)
    return model, rec


def overload(rec, level=24.0, delay=1e-3, stuck=0.01, tau=3e-3):
    """The record with the projector's own sound at the inputs: `level` V of the transmit chirp from `delay` s after the
    transmit, the inputs then stuck at the rail for `stuck` s and decaying with `tau`; clipped at the ADC's full scale."""
    out = rec.copy()
    s = out.samples
    kind = s.kind
    w = rec.transmit.waveform
    rail = rec.metadata.adc_full_scale
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate - delay
    k = (w.f_stop - w.f_start) / w.duration
    blast = np.where((t >= 0) & (t <= w.duration), level * np.cos(2 * np.pi * (w.f_start * t + 0.5 * k * t**2)), 0.0)
    after = t - w.duration
    blast += np.where(after > 0, rail * np.where(after < stuck, 1.0, np.exp(-(after - stuck) / tau)), 0.0)
    s.array = np.clip(np.asarray(s.array, dtype=np.float64) + blast, -rail, rail).astype(np.float32)
    s.kind = kind
    out.samples = s
    return out


@pytest.fixture(scope="module")
def loud():
    """One ping of the test vehicle's row (0.1 s pulse, window from 50 m, so it opens while the projector is still sending),
    clean and with the overload."""
    model, clean = ping("popoto_testauv_1row")
    return model, clean, overload(clean)


def run(rec, engine, **extra):
    with LiveChain(dict(CHAIN, engine=engine, **extra)) as chain:
        report = chain.run([rec.copy()])[0]
        power, ranges = np.asarray(chain.last["power"]), np.asarray(chain.last["ranges"])
    return report, power, ranges


def test_blank_interval_zeroes_with_raised_cosine_edges_real_and_complex():
    _, rec = ping("popoto_testauv_1row", window=(50.0, 200.0), noise=0.0)
    for complex_ in (False, True):
        r = rec.copy()
        if complex_:
            core.live_decimate_record(r)
        s = r.samples
        kind = s.kind
        s.array = np.ones_like(np.asarray(s.array))
        s.kind = kind
        r.samples = s
        core.live_blank_interval(r, 0.10, 0.15, 0.01)
        x = np.abs(np.asarray(r.samples.array))
        t = r.samples.start_time + np.arange(r.samples.n_samples) / r.samples.sample_rate
        out = np.where(t < 0.10, 0.10 - t, np.where(t > 0.15, t - 0.15, 0.0))
        want = np.where(out <= 0.0, 0.0, np.where(out >= 0.01, 1.0, 0.5 - 0.5 * np.cos(np.pi * out / 0.01)))
        assert np.allclose(x, want[None, :], atol=1e-6)
        assert x[:, (t > 0.101) & (t < 0.149)].max() == 0.0 and np.all(np.diff(x[0][t > 0.15]) >= -1e-7)      # no step: it rises smoothly


@pytest.mark.parametrize("engine", ["cpp", "python"])
def test_clipped_direct_sound_makes_false_detections_and_blanking_removes_them(loud, engine):
    model, clean, noisy = loud
    pulse = clean.transmit.waveform.duration
    want_range = float(np.linalg.norm(np.array(TARGET) - np.array([0.0, 0.0, 10.0])))
    r_clean, p_clean, ranges = run(clean, engine)
    assert len(r_clean["detections"]) == 1 and "blanking" not in r_clean
    target = r_clean["detections"][0]
    assert target["range_m"] == pytest.approx(want_range, abs=5.0)
    far = (ranges > 450.0) & (ranges < 590.0)
    floor = float(np.median(p_clean[:, far]))

    # Not blanked: the clipped pulse correlates with the replica. It is "detected" where it sits, and the matched filter
    # carries it to the far end of the window too (the correlation is circular over the block), 40 dB above the noise there.
    r_loud, p_loud, _ = run(noisy, engine)
    false = [d for d in r_loud["detections"] if abs(d["range_m"] - target["range_m"]) > 20.0]
    assert any(d["range_m"] < 100.0 and d["snr_db"] > 60.0 for d in false)
    assert any(d["range_m"] > 500.0 and d["snr_db"] > 20.0 for d in false)
    assert float(np.max(p_loud[:, ranges > 590.0])) > 1e3 * float(np.max(p_clean[:, ranges > 590.0]))

    # Blanked (zeroed before conditioning, tapered back): the report says from where it can see, nothing is reported nearer,
    # nothing false is left further out, and the target and the noise floor are what they were without the overload.
    r_blank, p_blank, ranges_b = run(noisy, engine, blanking=BLANK)
    min_range = 0.5 * C * (pulse + BLANK["recovery_s"] + BLANK["taper_s"])
    assert r_blank["blanking"] == {"min_range_m": pytest.approx(min_range), "mode": "zero"}
    assert ranges_b[0] >= min_range and ranges_b[0] < min_range + 2.5          # the map itself starts there
    assert len(r_blank["detections"]) == 1
    got = r_blank["detections"][0]
    assert got["range_m"] == pytest.approx(target["range_m"], abs=0.5) and got["bearing_deg"] == pytest.approx(target["bearing_deg"], abs=0.05)
    assert got["snr_db"] == pytest.approx(target["snr_db"], abs=0.5)
    keep = (ranges_b > 450.0) & (ranges_b < 590.0)
    assert 10.0 * np.log10(float(np.median(p_blank[:, keep])) / floor) == pytest.approx(0.0, abs=0.3)
    assert float(np.max(p_blank[:, ranges_b > 590.0])) < 3.0 * float(np.max(p_clean[:, ranges > 590.0]))

    # Mode "ignore" leaves the samples alone and only does not report inside the blanked range: the leak further out stays.
    r_ignore, _, _ = run(noisy, engine, blanking=dict(BLANK, mode="ignore"))
    assert r_ignore["blanking"]["mode"] == "ignore" and all(d["range_m"] >= min_range for d in r_ignore["detections"])
    assert any(d["range_m"] > 500.0 and abs(d["range_m"] - target["range_m"]) > 20.0 for d in r_ignore["detections"])

    # Through the decimate-first stage the blanking is done on the raw samples inside it, before the filter; the minimum
    # range grows by the filter's half length (0.83 ms: 0.6 m).
    r_dec, _, _ = run(noisy, engine, blanking=BLANK, decimate={})
    assert len(r_dec["detections"]) == 1 and r_dec["detections"][0]["snr_db"] == pytest.approx(target["snr_db"], abs=0.5)
    assert min_range < r_dec["blanking"]["min_range_m"] < min_range + 1.0
    # A clean ping is not hurt by blanking beyond losing what is nearer than the minimum range.
    r_same, _, _ = run(clean, engine, blanking=BLANK)
    assert len(r_same["detections"]) == 1 and r_same["detections"][0]["snr_db"] == pytest.approx(target["snr_db"], abs=0.2)


def test_minimum_range_follows_pulse_recovery_and_taper(loud):
    _, clean, _ = loud
    pulse = clean.transmit.waveform.duration
    for rec_s, taper in ((0.0, 0.0), (0.02, 0.0), (0.05, 0.01), (0.2, 0.005)):
        report, _, _ = run(clean, "cpp", blanking={"recovery_s": rec_s, "taper_s": taper})
        assert report["blanking"]["min_range_m"] == pytest.approx(0.5 * C * (pulse + rec_s + taper))
    table = json.loads((ROOT / "docs" / "design" / "popoto_stage2.json").read_text())["blanking"]
    assert table["pulses"]["0.1"]["0.05"] == pytest.approx(0.5 * C * (0.1 + 0.05 + table["taper_s"]))
    assert table["pulses"]["1.0"]["0.0"] == pytest.approx(0.5 * C * (1.0 + table["taper_s"]))
    with pytest.raises(ValueError):
        LiveChain(dict(CHAIN, blanking={"mode": "mute"}))


def test_a_clipped_reference_burst_is_not_used_and_the_board_is_flagged():
    model, rec = ping("popoto_testauv_1row_clocks")
    ref = model.config["receive"]["reference"]
    boards = model.config["receive"]["boards"]
    truth = np.array([b["clock_offset"] for b in boards]) - boards[0]["clock_offset"]
    drift = np.array([b["clock_drift_ppm"] for b in boards]) - boards[0]["clock_drift_ppm"]
    noisy = overload(rec)                                    # the first burst, 5 ms after the window opens, is under the pulse
    # As it was: the burst is searched for in clipped samples. Here the search fails loudly (it could as well have
    # locked onto a harmonic of the clipped pulse).
    with pytest.raises(sync.SyncError):
        sync.synchronise(noisy.copy(), ref, enforce=False)
    with pytest.raises(RuntimeError):
        core.live_synchronise(noisy.copy(), ref)
    # With min_quality: every board is timed from its second burst alone, flagged, and the ping is not called ok,
    # because a drift cannot be had from one burst (the offsets are right at the end of the window and off by the
    # drift times the window's length at its start).
    checked = dict(ref, min_quality=0.5)
    span = rec.samples.n_samples / rec.samples.sample_rate
    results = {"python": sync.synchronise(noisy.copy(), checked, enforce=False)}
    py = results["python"]
    cpp = core.live_synchronise(noisy.copy(), checked)
    dec = core.live_decimate_record(noisy.copy(), {}, checked)
    assert not py.ok and not cpp["ok"] and not dec["ok"]
    assert list(py.estimate.flagged) == cpp["flagged"] == dec["flagged"] == [1, 1, 1, 1]
    assert np.allclose(py.estimate.offset, cpp["offset"], atol=2e-8) and np.allclose(cpp["offset"], dec["offset"], atol=1e-10)
    assert np.all(py.estimate.drift_ppm == 0.0) and cpp["drift_ppm"] == [0.0] * 4
    assert np.all(np.abs(np.array(cpp["offset"]) - truth) <= np.abs(drift) * 1e-6 * span + 2e-6)
    assert np.abs(np.array(cpp["offset"]) - truth).max() > 5e-6                 # which is why it is not ok
    # A clean ping passes the same check untouched.
    clean = core.live_synchronise(rec.copy(), checked)
    plain = core.live_synchronise(rec.copy(), ref)
    assert clean["ok"] and clean["flagged"] == [0, 0, 0, 0] and clean["offset"] == plain["offset"] and clean["drift_ppm"] == plain["drift_ppm"]
    # Clipping alone decides too: a burst that still correlates well but sits in railed samples is not trusted.
    partly = rec.copy()
    s = partly.samples
    kind = s.kind
    x = np.asarray(s.array).copy()
    n0 = int(ref["lead"] * s.sample_rate)
    row = rec.metadata.boards[2].reference_channel
    x[row, n0:n0 + 60] = rec.metadata.adc_full_scale                             # 15% of the burst's samples at the rail
    s.array = x
    s.kind = kind
    partly.samples = s
    got = core.live_synchronise(partly, checked)
    assert got["flagged"] == [0, 0, 1, 0] and not got["ok"]
    # In the chain, both engines, with and without the decimate-first stage: the ping is processed, flagged and counted.
    for engine in ("cpp", "python"):
        for extra in ({}, {"decimate": {}}):
            with LiveChain(dict(CHAIN, engine=engine, sync={"reference": ref, "min_quality": 0.5}, blanking=BLANK, **extra)) as chain:
                report = chain.run([noisy.copy()])[0]
                assert chain.boards_out_of_sync == 1
            assert report["sync"]["ok"] is False and report["sync"]["flagged"] == [0, 1, 2, 3]
            assert any(abs(d["range_m"] - 404.8) < 5.0 for d in report["detections"])


def test_a_reference_sent_after_the_blanked_stretch_synchronises_as_usual():
    """The remedy: time the first burst after the pulse and the inputs' recovery (the reference's `lead`)."""
    model, rec = ping("popoto_testauv_1row_clocks", receive={"window": {"range": [50.0, 600.0]}, "reference": {"lead": 0.08}})
    ref = model.config["receive"]["reference"]
    assert rec.samples.start_time + ref["lead"] > rec.transmit.waveform.duration + BLANK["recovery_s"] + BLANK["taper_s"]
    boards = model.config["receive"]["boards"]
    truth = np.array([b["clock_offset"] for b in boards]) - boards[0]["clock_offset"]
    noisy = overload(rec)
    for engine in ("cpp", "python"):
        for extra in ({}, {"decimate": {}}):
            with LiveChain(dict(CHAIN, engine=engine, sync={"reference": ref, "min_quality": 0.5}, blanking=BLANK, **extra)) as chain:
                report = chain.run([noisy.copy()])[0]
            assert report["sync"]["ok"] is True and "flagged" not in report["sync"]
            assert np.allclose(np.array(report["sync"]["offsets_us"]) * 1e-6, truth, atol=1e-6)
            strong = [d for d in report["detections"] if d["snr_db"] > 40.0]
            assert len(strong) == 1 and abs(strong[0]["range_m"] - 404.8) < 5.0


def test_listening_blanks_around_the_transmits_the_blocks_mark(loud):
    model, clean, noisy = loud
    fs = noisy.samples.sample_rate
    period = 1.0
    blocks = list(FlankBoardEmulator([noisy, noisy], ping_interval=period, block_samples=128).blocks())
    until, taper = clean.transmit.waveform.duration + BLANK["recovery_s"], BLANK["taper_s"]
    chunk = 8192
    for decimate in (None, {}):
        plain = core.StreamAssemblerCpp(model.make_record(make_pose()), chunk if decimate is None else chunk // 16, decimate)
        blanked = core.StreamAssemblerCpp(model.make_record(make_pose()), chunk if decimate is None else chunk // 16, decimate)
        blanked.set_blanking(until, taper)
        a = [r for b in blocks for r in plain.feed(b)]
        b_ = [r for b in blocks for r in blanked.feed(b)]
        assert len(a) == len(b_) > 10 and blanked.blanked_samples > 0 and plain.blanked_samples == 0
        rate = a[0].samples.sample_rate
        xa = np.concatenate([np.abs(np.asarray(r.samples.array)[0]) for r in a])
        xb = np.concatenate([np.abs(np.asarray(r.samples.array)[0]) for r in b_])
        t = np.arange(len(xa)) / rate
        since = t % period                                           # s after the last transmit
        inside = (since > 0.07) & (since < until - 0.002)            # the overload, from where the window opens
        assert xa[inside].max() > 1.0 and xb[inside].max() < 1e-3    # volts at the rail without; nothing with
        later = since > until + taper + 0.005
        assert np.allclose(xa[later], xb[later], atol=1e-6 if decimate is None else 1e-4)


def test_blanking_on_the_simulators_own_overload_model():
    """The same with the simulator doing the clipping: the projector's direct sound (transmit.direct_path) into a
    front end that clips at the ADC's full scale, stays on the rail and recovers (receive.overload; the values are
    illustrative, as everywhere in this file)."""
    model = SonarModel.load("popoto_testauv_1row", {"receive": {"window": {"range": [50.0, 600.0]},
                                                               "overload": {"adc": {"dead_time": 0.01, "recovery_tau": 0.003}}},
                                                   "transmit": {"direct_path": True}})
    rec = simulate_ping(model, make_pose(position=(0.0, 0.0, 10.0)), Scene().add_points([list(TARGET)], target_strength_db=10.0),
                        environment={"sound_speed": C}, noise_rms=2e-6, rng=np.random.default_rng(0))
    x = np.abs(np.asarray(rec.samples.array))
    rail = rec.metadata.adc_full_scale
    assert not rec.samples.is_complex() and x.max() >= 0.99 * rail                # the inputs did clip
    for engine in ("cpp", "python"):
        plain, _, _ = run(rec, engine)
        blanked, _, _ = run(rec, engine, blanking=BLANK)
        min_range = blanked["blanking"]["min_range_m"]
        assert any(d["range_m"] < min_range for d in plain["detections"])         # the pulse itself, "detected"
        real = [d for d in blanked["detections"] if abs(d["range_m"] - 404.8) < 5.0]
        assert len(real) == 1 and all(d["range_m"] >= min_range for d in blanked["detections"])
        assert len(blanked["detections"]) <= len(plain["detections"]) - 1
