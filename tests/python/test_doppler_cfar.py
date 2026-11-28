"""D3 part A: the sweep bank's second combiner, `chain.doppler: {mode: sweep_bank, combine: cfar}` (libs/live doppler.hpp
DopplerCombine::Cfar; gpusonar/doppler.py NormalisedFold), opt-in. The engine-parity cases are in test_live_cpp.py.

D4's combiner (`max`) keeps the largest power over each beam's hypotheses and lets the detector estimate its background
from that maximum: the background rises (4.8 to 5.6 dB on noise for 17 to 33 hypotheses, D4) and every target loses
that much margin. `cfar` normalises each (beam, hypothesis) map by its own background first (the detector's estimate,
detector_background, on that hypothesis' own cell means), keeps the largest NORMALISED value, and tests it against
thresholds planned for the budget divided by the hypotheses per beam. Why that holds the rate (ours): each normalised
map is a statistic the detector's thresholds were planned for, and P(max over N > x) <= N P(one > x) (the union bound).
On the safe side, because neighbouring hypotheses are correlated: their noise is nearly the same.

So, for a target whose zero hypothesis is its best (a still one), the detection's snr_db is the one without the bank (its
own hypothesis' background is the one the detector estimates without the bank) and its margin falls by the threshold
correction only: the factor of the budget / N over that of the budget, which in an exponential tail is about
10 log10((x + ln N) / x) at a threshold x (ours: 0.6 to 1.0 dB for x of 13 to 14 dB and N = 17 to 35).
"""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import detect, doppler, live
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
WINDOW = {"receive": {"window": {"range": [100.0, 400.0]}}}
CHAIN = {"bearings_deg": [-50.0, 50.0, 1.0], "backend": "host"}
MAX = {"mode": "sweep_bank", "max_range_rate_m_s": 2.5, "report": True}
CFAR = dict(MAX, combine="cfar")
strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731


def ping(target=(0.0, 250.0, 0.0), velocity=(0.0, 0.0, 0.0), own=(0.0, 0.0, 0.0), noise_rms=1e-4, seed=1, ts_db=0.0, sequence=0):
    """flank_32 (6-10 kHz LFM, 1 s) with one point target; the array along x, broadside +y (test_doppler_bank.py's)."""
    sonar = SonarModel.load("flank_32", WINDOW)
    scene = Scene().add_points([list(target)], velocity=velocity, target_strength_db=ts_db)
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0), velocity=own), scene, environment={"sound_speed": C},
                         noise_rms=noise_rms, adc=False, absorption_model="none", rng=np.random.default_rng(seed), sequence=sequence,
                         time_ns=sequence * 2_000_000_000)


def run(cfg, rec):
    with LiveChain(cfg) as chain:
        rep = chain.process(rec.copy())
        return rep, chain.detector


def on_target(rep, bearing_deg=0.0):
    near = [d for d in rep["detections"] if abs(d["bearing_deg"] - bearing_deg) < 2.0]
    return max(near, key=lambda d: d["snr_db"]) if near else None


# --- the fold, against a reference written here -------------------------------------------------------------------------

def test_the_normalised_fold_is_the_largest_ratio_of_each_hypothesis_to_its_own_background():
    """Three hypotheses on two beams (beam 0 takes 0..1, beam 1 takes 1..2), maps of different levels: the fold keeps,
    per sample, the largest sample / (the detector's background of its cell in ITS hypothesis); the reference is each
    beam's centre hypothesis' background; ties keep the earlier hypothesis; skipped cells are 0."""
    rng = np.random.default_rng(3)
    n_b, n_c, k, skip, guard, train = 2, 60, 4, 2, 1, 8
    p = {"first": [0, 1], "per_beam": 2}
    maps = [rng.exponential(1.0, (n_b, n_c * k)).astype(np.float32) * s for s in (1.0, 10.0, 0.1)]
    fold = doppler.NormalisedFold(n_b, n_c, k, skip)
    for h, m in enumerate(maps):
        fold.fold(p, h, m, guard, train)
    fine, cells, level = fold.maps()
    want = np.zeros((n_b, n_c * k))
    want_which = np.full((n_b, n_c * k), -1)
    want_ref = np.zeros((n_b, n_c))
    for b in range(n_b):
        for h in range(p["first"][b], p["first"][b] + p["per_beam"]):
            m = maps[h][b:b + 1, skip * k:].reshape(1, n_c - skip, k)
            cm = (m.astype(np.float64).sum(axis=2) / k).astype(np.float32)
            bg, _ = detect.background(cm, guard, train)
            if h == p["first"][b] + (p["per_beam"] - 1) // 2:
                want_ref[b, skip:] = bg[0]
            r = (m[0].astype(np.float64) / np.maximum(bg[0], 1e-300)[:, None]).astype(np.float32).ravel()
            better = r > want[b, skip * k:]
            want[b, skip * k:][better] = r[better]
            want_which[b, skip * k:][better] = h
    assert np.array_equal(level, want_ref)
    assert np.array_equal(fold.which, want_which)
    assert np.allclose(fine, (want * np.repeat(want_ref, k, axis=1)).astype(np.float32), rtol=1e-6, atol=0)
    assert np.all(fine[:, :skip * k] == 0.0) and np.all(cells[:, :skip] == 0.0)
    # Each beam's maps were scaled by 1, 10 and 0.1: normalised they are alike, the scale no longer picks the winner.
    assert 0.2 < np.mean(fold.which[0, skip * k:] == 0) < 0.8 and 0.2 < np.mean(fold.which[1, skip * k:] == 1) < 0.8


def test_the_detector_given_its_own_background_is_the_detector():
    """detect.multiscale with `level` = the background it would estimate gives the same detections and background (the
    path combine cfar uses, with another background, is the detector's own test)."""
    rng = np.random.default_rng(5)
    rec = SonarModel.load("flank_32", WINDOW).make_record(core.Pose(), environment={"sound_speed": C})
    n_b, n_c, k = 21, 200, 8
    bearings = np.radians(np.linspace(-10.0, 10.0, n_b))
    fine = rng.exponential(1.0, (n_b, n_c * k)).astype(np.float32)
    fine[10, 800:804] *= 3000.0
    cells = fine.reshape(n_b, n_c, k).mean(axis=2)
    plan = core.live_detector_plan(rec, n_c, k, 0.25, [float(b) for b in bearings], 1, 2, 30, True, {"merge_m": 5.0})
    fr, cr = 0.25 * np.arange(n_c * k), 2.0 * np.arange(n_c) + 0.875
    a, bg_a = detect.multiscale(fine, cells, fr, cr, bearings, 2, 30, plan)
    own, _ = detect.background(cells, 2, 30)
    b, bg_b = detect.multiscale(fine, cells, fr, cr, bearings, 2, 30, plan, level=own)
    assert np.array_equal(bg_a, bg_b) and len(a) == len(b) >= 1
    assert [(d.range, d.bearing, d.snr_db) for d in a] == [(d.range, d.bearing, d.snr_db) for d in b]


# --- the plan -----------------------------------------------------------------------------------------------------------

def test_the_plan_divides_the_budget_by_the_hypotheses_per_beam_and_moves_nothing_else():
    rec = SonarModel.load("flank_32", WINDOW).make_record(core.Pose(), environment={"sound_speed": C})
    bearings = np.radians(np.arange(-50.0, 50.5, 1.0))
    m, c = doppler.plan(rec, bearings, MAX), doppler.plan(rec, bearings, CFAR)
    assert m["combine"] == "max" and m["budget_divisor"] == 1.0
    assert c["combine"] == "cfar" and c["budget_divisor"] == float(c["per_beam"]) == 29.0
    assert c["threshold_db"] == 0.0 and json.loads(c["json"])["combine"] == "cfar"
    for key in ("rates", "scales", "shifts", "first", "per_beam", "step", "pedestal_m", "baseband_max_range_rate"):
        assert c[key] == m[key], key
    assert doppler.plan(rec, bearings, dict(CFAR, threshold_db=0.5))["threshold_db"] == 0.5


def test_the_option_is_refused_when_unknown_and_said_only_when_not_max():
    assert live.DEFAULTS["doppler"]["combine"] == "max"
    assert "doppler_combine" not in live.cpp_chain_options({"doppler": MAX})
    assert live.cpp_chain_options({"doppler": CFAR})["doppler_combine"] == "cfar"
    assert "doppler_combine" not in live.cpp_chain_options({"doppler": dict(CFAR, mode="off")})
    with pytest.raises(ValueError, match="combine"):
        doppler.merged(dict(CFAR, combine="sum"))
    with pytest.raises(ValueError, match="doppler combine"):
        core.LiveChainCpp({"doppler_mode": "sweep_bank", "doppler_combine": "sum"})
    with pytest.raises(ValueError, match=r"chain\.doppler\.combin: unknown key; did you mean 'combine'\?"):
        live.check_keys(dict(live.DEFAULTS, doppler={"mode": "sweep_bank", "combin": "cfar"}))
    text = json.loads(core.live_options_from_text("doppler_mode=sweep_bank\ndoppler_combine=cfar\n"))
    assert text["doppler_combine"] == "cfar" and json.loads(core.live_options_from_text(""))["doppler_combine"] == "max"


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_one_hypothesis_with_cfar_is_the_chain_without_the_bank_byte_for_byte(engine):
    recs = [ping((30.0, 250.0, 0.0), velocity=(0.0, 1.0, 0.0), noise_rms=1e-5, seed=s, sequence=s) for s in range(2)]
    with LiveChain(dict(CHAIN, engine=engine)) as off:
        want = [strip(off.process(r.copy())) for r in recs]
    with LiveChain(dict(CHAIN, engine=engine, doppler={"mode": "sweep_bank", "combine": "cfar", "max_range_rate_m_s": 0.0})) as one:
        got = [strip(one.process(r.copy())) for r in recs]
    assert json.dumps(got) == json.dumps(want)


# --- what it does to a target -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_still_target_keeps_its_snr_and_loses_only_the_threshold_correction(engine):
    """A still point at broadside, the platform still: its best hypothesis is the zero one, whose normalised value is the
    detection without the bank. snr_db is the same (the other hypotheses' normalised values are no larger at its
    peak); the single-sample threshold rises by the union bound's correction (29 hypotheses), 0.5 to 1.2 dB; D4's max
    lost about 6 dB of snr here (its background rise) and moved no threshold."""
    rec = ping((0.0, 250.0, 0.0), ts_db=-40.0, noise_rms=1e-4)
    off, plan_off = run(dict(CHAIN, engine=engine), rec)
    on, plan_on = run(dict(CHAIN, engine=engine, doppler=CFAR), rec)
    mx, _ = run(dict(CHAIN, engine=engine, doppler=MAX), rec)
    a, b, m = on_target(off), on_target(on), on_target(mx)
    assert a is not None and b is not None and m is not None
    correction = plan_on["scales"][0]["threshold_db"] - plan_off["scales"][0]["threshold_db"]
    assert 0.5 < correction < 1.2, correction
    assert b["snr_db"] == pytest.approx(a["snr_db"], abs=0.05)
    assert abs(b["range_rate_m_s"]) <= b["range_rate_step_m_s"] + 1e-9
    assert a["snr_db"] - m["snr_db"] > 4.0          # D4's max: the background of a maximum
    print(f"\n{engine}: snr off {a['snr_db']:.2f}, cfar {b['snr_db']:.2f}, max {m['snr_db']:.2f} dB; threshold +{correction:.2f} dB")


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_point_at_2_m_s_gains_the_mismatch_less_the_correction(engine):
    """The point opening at 2 m/s (a 1 s, 4 kHz sweep loses 11.7 dB to it, closed form; the simulator's pulse 11.1 dB on
    the map, test_doppler_bank.py): with cfar the detection's snr_db gains about that (its own hypothesis' background is
    the noise's), D4's max gains about 6 dB less; the margin over the single-sample threshold gains it less the
    correction. About 20 dB over the noise without the bank: weaker, the noise on the smeared echo lifts the off chain's
    strongest sample and the gain reads less (7.7 dB at 14 dB off, measured while writing this test)."""
    rec = ping((0.0, 250.0, 0.0), velocity=(0.0, 2.0, 0.0), ts_db=-30.0, noise_rms=1e-4)
    off, plan_off = run(dict(CHAIN, engine=engine), rec)
    on, plan_on = run(dict(CHAIN, engine=engine, doppler=CFAR), rec)
    mx, plan_mx = run(dict(CHAIN, engine=engine, doppler=MAX), rec)
    a, b, m = on_target(off), on_target(on), on_target(mx)
    assert a is not None and b is not None and m is not None
    gain = b["snr_db"] - a["snr_db"]
    assert 9.5 < gain < 12.2, gain
    assert b["snr_db"] - m["snr_db"] > 4.0, (b["snr_db"], m["snr_db"])
    correction = plan_on["scales"][0]["threshold_db"] - plan_off["scales"][0]["threshold_db"]
    margin_gain = (b["snr_db"] - plan_on["scales"][0]["threshold_db"]) - (a["snr_db"] - plan_off["scales"][0]["threshold_db"])
    assert margin_gain == pytest.approx(gain - correction, abs=1e-9) and margin_gain > 8.0
    assert b["range_rate_m_s"] == pytest.approx(2.0, abs=b["range_rate_step_m_s"] + 1e-9)
    print(f"\n{engine}: snr off {a['snr_db']:.2f} -> cfar {b['snr_db']:.2f} (max {m['snr_db']:.2f}) dB; margin +{margin_gain:.2f} dB")


# --- the false-alarm rate -----------------------------------------------------------------------------------------------

def _noise_pings(n, seed):
    # D-256: the shipped popoto_long_range sends a 2 s hyperbolic sweep, which the bank refuses; this test is of the bank: its 1 s linear sweep is stated
    sonar = SonarModel.load("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1300.0]}}, "transmit": {"waveform": {"type": "lfm", "source": None, "duration": 1.0}}})
    scene = Scene().add_points([[0.0, 1100.0, 0.0]], target_strength_db=-300.0)
    return [simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=2e-4, rng=np.random.default_rng(seed + i), sequence=i,
                          time_ns=i * 10_000_000_000, environment={"sound_speed": C}, adc=False, absorption_model="none") for i in range(n)]


def test_on_noise_the_cfar_combiner_gives_no_more_false_detections_than_the_chain_without_the_bank():
    """Noise-only pings of the long-range array (4-6 kHz, 1 s: 17 hypotheses for 3 m/s), the budget loosened to 3 false
    detections a ping so that a few dozen pings count: the union bound keeps the count at or under the chain's without
    the bank (within three Poisson sigmas of the difference), and above a fifth of it (the bound is not so loose that the
    combiner is blind; measured on more pings by tools/doppler_bank_threshold.py --combine cfar)."""
    pings = _noise_pings(24, 9100)
    base = dict(CHAIN, bearings_deg=[-60.0, 60.0, 0.5], engine="cpp", cfar={"false_alarms_per_ping": 3.0})
    counts = {}
    for name, extra in (("off", {}), ("cfar", {"doppler": {"mode": "sweep_bank", "combine": "cfar"}})):
        with LiveChain(dict(base, **extra)) as chain:
            counts[name] = sum(len(chain.process(p.copy())["detections"]) for p in pings)
            if extra:
                assert json.loads(chain._cpp.doppler())["per_beam"] == 17
    off, on = counts["off"], counts["cfar"]
    print(f"\nnoise only, 24 pings at 3 a ping: off {off}, cfar {on}")
    assert off > 30
    assert on <= off + 3.0 * np.sqrt(on + off), counts
    assert on > 0.2 * off, counts
