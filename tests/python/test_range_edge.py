"""cfar.range_edge_correction (opt-in, sonar-8h09; libs/live detector.hpp RangeEdge, gpusonar.detect._pulse_edge_range).

The live chain's default range is the leading edge (D-160): the first sample of a detection over the single-sample
threshold F. A point echo S over its background crosses F where S |rho(x)|^2 = F on its compressed pulse's skirt and
range sidelobes, which for a flat band B lies about (c / (2 pi B)) sqrt(S / F) before its peak: the stronger the echo,
the earlier (ours; docs/CONFIGURATION.md, chain.cfar.range_edge_correction). A 2 s sweep has 3 dB more energy than a 1 s one, so on the same
target it reads earlier: that is the offset of the face builds' free-field self-test, not a fault of the simulator or of
the replica's time origin (the cross-correlation of the record with its replica peaks at the truth to millimetres).

With `pulse` the leading edge is referenced to the compressed pulse: a local peak no more than kEdgeNoise times what a
stronger peak's pulse puts there is that pulse's skirt; the range is the nearest remaining arrival's, at its peak.
Held here, both engines: a still point target reads within a few centimetres for every pulse (0.1 to 4 s, linear and
hyperbolic, tapers, a 1 kHz band, 16 kHz sampling); an extended echo still reads its nearest highlight (D-178's intent);
off is the default bit for bit; the engines agree; the option is checked and reaches the program.
"""
import json

import numpy as np
import pytest

from gpusonar import core, live
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

C = 1500.0
KW = dict(environment={"sound_speed": C}, adc=False, absorption_model="none")
R = 4000.0
NOISE = 2e-4


def sonar_for(T=2.0, shape="hfm", alpha=0.0, f0=4000.0, f1=6000.0, fs=8000.0, r=R):
    wf = {"duration": float(T), "f_start": f0, "f_stop": f1, "window_alpha": float(alpha)}
    wf.update({"type": "lfm", "source": None} if shape == "lfm" else {"type": "sampled", "source": "hfm"})
    return SonarModel.load("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": fs, "window": {"range": [r - 250.0, r + 250.0]}},
                                                 "transmit": {"waveform": wf}})


def ping(sonar, points, ts_db, seed=1, noise=NOISE):
    scene = Scene()
    for p, ts in zip(points, np.broadcast_to(ts_db, len(points))):
        scene.add_points([list(p)], target_strength_db=float(ts))
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=noise, rng=np.random.default_rng(seed), **KW)


def two_way_range(rec, p):
    """Half the two-way path, mean over the hydrophones (transmitter at the record's lever arm)."""
    g = rec.geometry
    tx = np.asarray(g.tx.lever_arm, dtype=float)
    return float(np.mean([0.5 * (np.linalg.norm(p - tx) + np.linalg.norm(p - np.asarray(e.lever_arm, dtype=float))) for e in g.rx]))


def detect(rec, engine, edge=None, **cfar):
    cfg = {"cell_statistic": "multiscale", "backend": "host", "engine": engine, "cfar": dict(cfar)}
    if edge is not None:
        cfg["cfar"]["range_edge_correction"] = edge
    with LiveChain(cfg) as chain:
        return chain.process(rec.copy())["detections"]


POINT_CASES = [dict(T=T, shape=s) for s in ("lfm", "hfm") for T in (0.1, 0.5, 1.0, 2.0, 4.0)] + [
    dict(T=2.0, shape="lfm", alpha=0.1), dict(T=2.0, shape="hfm", alpha=1.0), dict(T=2.0, shape="hfm", f0=4500.0, f1=5500.0),
    dict(T=2.0, shape="lfm", fs=16000.0)]


@pytest.mark.parametrize("case", POINT_CASES, ids=lambda c: "-".join(f"{k}{v}" for k, v in c.items()))
def test_a_still_point_echo_reads_at_its_peak_with_the_pulse_reference(case):
    """A target 20 dB TS at 4 km broadside (S 24 to 40 dB over the background): off, the leading edge reads 0.25 to 2.5 m
    short, more for the longer pulses (their SNR); pulse, within 4 cm of the truth on both engines, the same range."""
    sonar = sonar_for(**case)
    p = np.array([0.0, R, 0.0])
    rec = ping(sonar, [p], 20.0, seed=3)
    truth = two_way_range(rec, p)
    got = {}
    for engine in ("cpp", "python"):
        off = detect(rec, engine)
        on = detect(rec, engine, "pulse")
        assert len(off) == len(on) == 1, (engine, off, on)
        got[engine] = on[0]["range_m"]
        assert off[0]["range_m"] - truth <= -0.2, (engine, off[0]["range_m"] - truth)     # the offset the option removes
        assert abs(on[0]["range_m"] - truth) < 0.04, (engine, on[0]["range_m"] - truth, off[0]["range_m"] - truth)
        # Only the range moves.
        assert {k: v for k, v in on[0].items() if k != "range_m"} == {k: v for k, v in off[0].items() if k != "range_m"}
    assert got["python"] == pytest.approx(got["cpp"], abs=1e-6)


@pytest.mark.parametrize("ts", [0.0, 10.0, 30.0])
def test_the_offset_grows_with_the_snr_and_the_reference_removes_it_at_every_snr(ts):
    """The leading edge's offset is a function of the echo's SNR (the 2 s hyperbolic sweep, 4 km): the plain edge reads
    earlier the stronger the echo; with pulse it is within 4 cm at 10 and 30 dB TS (S about 27, 47 dB), and at 0 dB TS
    (S about 17 dB, 4 dB over the threshold) within 15 cm, what noise leaves of a peak's position there (6 cm on this
    seed; the centroid of the same echo reads 12 cm off on average, docs/CONFIGURATION.md)."""
    sonar = sonar_for()
    p = np.array([0.0, R, 0.0])
    rec = ping(sonar, [p], ts, seed=5)
    truth = two_way_range(rec, p)
    for engine in ("cpp", "python"):
        on = detect(rec, engine, "pulse")
        assert len(on) == 1 and abs(on[0]["range_m"] - truth) < (0.04 if ts >= 10.0 else 0.15), (engine, on, truth)
    off = detect(rec, "cpp")[0]["range_m"] - truth
    assert off < (-0.3 if ts >= 10.0 else 0.05)
    if ts >= 30.0:
        assert off < -1.9      # the skirt reaches 6 m (S 47 dB); the edge stops at the start of the cell before the peak's


def test_an_extended_echo_still_reads_its_nearest_highlight():
    """D-178's property kept: a target whose first highlight is the weakest (8 dB) with five 12 dB stronger 6 to 30 m
    behind it on the same bearing (a channel's later arrivals, a hull's far parts) reads at the first highlight with
    pulse (within 4 cm), as the leading edge reads near it (its own skirt, within a few decimetres) and the centroid
    tens of metres behind; both engines alike."""
    sonar = sonar_for()
    p0 = np.array([0.0, R, 0.0])
    pts = [p0 + np.array([0.0, dr, 0.0]) for dr in (0.0, 6.0, 12.0, 18.0, 24.0, 30.0)]
    rec = ping(sonar, pts, [8.0, 20.0, 20.0, 20.0, 20.0, 20.0], seed=7)
    truth = two_way_range(rec, p0)
    got = {}
    for engine in ("cpp", "python"):
        on = detect(rec, engine, "pulse")
        off = detect(rec, engine)
        cen = detect(rec, engine, range_estimate="centroid")
        assert len(on) == len(off) == len(cen) == 1, (engine, on, off, cen)
        assert abs(on[0]["range_m"] - truth) < 0.04, (engine, on[0]["range_m"] - truth)
        assert abs(off[0]["range_m"] - truth) < 1.0
        assert cen[0]["range_m"] - truth > 10.0
        got[engine] = on[0]["range_m"]
    assert got["python"] == pytest.approx(got["cpp"], abs=1e-6)


def test_a_strong_highlight_close_behind_a_weak_one_is_reported_at_the_weak_one():
    """An arrival is not taken for the skirt: a 10 dB highlight 2 m in front of a 25 dB one (in the next range cell, so
    the sidelobe rule keeps its cell; two cells apart it drops it, as before) stands 15 dB under it, 3.6 dB above what
    the strong one's pulse may put 2 m before it (-18.6 dB with the noise and scalloping allowance), so pulse reads the
    weak one within 4 cm, not the strong one 2 m behind; the plain leading edge reads before it (its skirt)."""
    sonar = sonar_for()
    p0 = np.array([0.0, R + 0.9, 0.0])
    rec = ping(sonar, [p0, p0 + np.array([0.0, 2.0, 0.0])], [10.0, 25.0], seed=9)
    truth = two_way_range(rec, p0)
    for engine in ("cpp", "python"):
        on = detect(rec, engine, "pulse")
        off = detect(rec, engine)
        assert len(on) == len(off) == 1 and abs(on[0]["range_m"] - truth) < 0.04, (engine, on, truth)
        assert off[0]["range_m"] - truth < -0.1, (engine, off, truth)


def test_a_direct_arrival_ahead_of_stronger_paths_close_behind_is_still_the_range():
    """Shallow water's case (popoto_face_2row_shallow, 100 m of water: surface and bottom paths 0.6 to 1.9 m behind the
    direct one): a 10 dB direct arrival with four 2 to 4 dB stronger paths 0.6 to 1.9 m behind, which add to a peak about
    a metre behind it. The direct arrival's samples stand above what that peak's pulse explains, so pulse reads the
    direct arrival (within 10 cm; the plain edge 0.65 m before it), not the composite peak; both engines alike. On that
    scene's own records (3 pings, with `elevation`) pulse reads what the plain edge reads, to the sample: there the
    direct arrival is a shoulder on the rise, no local peak, and the rule for two adjacent unexplained samples takes it."""
    sonar = sonar_for()
    p0 = np.array([0.0, R + 0.9, 0.0])
    pts = [p0] + [p0 + np.array([0.0, dr, 0.0]) for dr in (0.6, 0.9, 1.2, 1.9)]
    rec = ping(sonar, pts, [10.0, 12.0, 12.0, 14.0, 12.0], seed=13)
    truth = two_way_range(rec, p0)
    got = {}
    for engine in ("cpp", "python"):
        on = detect(rec, engine, "pulse")
        off = detect(rec, engine)
        assert len(on) == len(off) == 1, (engine, on, off)
        assert abs(on[0]["range_m"] - truth) < 0.1, (engine, on[0]["range_m"] - truth, off[0]["range_m"] - truth)
        assert off[0]["range_m"] - truth < -0.3
        got[engine] = on[0]["range_m"]
    assert got["python"] == pytest.approx(got["cpp"], abs=1e-6)


def test_off_is_the_default_bit_for_bit_and_the_plan_says_the_option_only_when_on():
    sonar = sonar_for()
    p = np.array([0.0, R, 0.0])
    rec = ping(sonar, [p, p + np.array([300.0, 40.0, 0.0])], [20.0, 5.0], seed=11)
    for engine in ("cpp", "python"):
        assert detect(rec, engine, "off") == detect(rec, engine)
    with LiveChain({"cell_statistic": "multiscale", "backend": "host"}) as chain:
        chain.process(rec.copy())
        plain = chain.detector
    with LiveChain({"cell_statistic": "multiscale", "backend": "host", "cfar": {"range_edge_correction": "pulse"}}) as chain:
        chain.process(rec.copy())
        edged = chain.detector
    assert "range_edge_correction" not in plain
    assert edged.pop("range_edge_correction") == "pulse" and edged == plain


def test_the_pulse_envelope_is_the_replicas_compressed_pulse():
    """pulse_power: a rectangular 2 s linear sweep of 2 kHz compresses to sinc^2(2 B x / c) (TB 4000: the Fresnel ripple
    is small) within 0.01 of the peak over the mainlobe and first sidelobes; both sides alike; the hyperbolic sweep's
    pulse is as narrow (the same band) and symmetric too (an autocorrelation)."""
    lfm = sonar_for(shape="lfm").make_record(core.Pose(), environment={"sound_speed": C})
    hfm = sonar_for(shape="hfm").make_record(core.Pose(), environment={"sound_speed": C})
    step, n = 0.01, 400
    x = step * np.arange(n)
    near, far = np.asarray(core.live_pulse_power(lfm, step, n, True)), np.asarray(core.live_pulse_power(lfm, step, n, False))
    assert near == pytest.approx(far, abs=1e-6)
    assert near == pytest.approx(np.sinc(2.0 * 2000.0 * x / C) ** 2, abs=0.01)
    h_near, h_far = np.asarray(core.live_pulse_power(hfm, step, n, True)), np.asarray(core.live_pulse_power(hfm, step, n, False))
    assert h_near == pytest.approx(h_far, abs=1e-6)
    first_null = lambda p: x[np.argmax(np.diff(p) > 0)]      # noqa: E731
    assert first_null(h_near) == pytest.approx(first_null(near), abs=0.03)


def test_the_option_is_checked_alike_and_reaches_the_program():
    with pytest.raises(ValueError, match="range_edge_correction"):
        LiveChain({"cell_statistic": "multiscale", "backend": "host", "cfar": {"range_edge_correction": "peak"}})
    for cfg in ({"cell_statistic": "multiscale", "cfar": {"range_estimate": "centroid", "range_edge_correction": "pulse"}},
                {"cell_statistic": "mean", "cfar": {"range_edge_correction": "pulse"}},
                {"cell_statistic": "multiscale", "cfar": {"range_edge_correction": "pulse"}, "doppler": {"mode": "sweep_bank"}}):
        with pytest.raises(ValueError, match="range_edge_correction"):
            LiveChain(dict(cfg, backend="host"))
    with pytest.raises(ValueError, match="range_edge_correction"):
        core.live_options_from_text("cfar_range_edge_correction=on\n")
    with pytest.raises(ValueError, match="range_edge_correction"):
        core.LiveChainCpp({"cfar_range_edge_correction": "pulse", "cfar_range_estimate": "centroid", "cell_multiscale": True})
    with pytest.raises(ValueError, match="range_edge_correction"):
        core.LiveChainCpp({"cfar_range_edge_correction": "pulse", "cell_multiscale": True, "doppler_mode": "sweep_bank"})
    flat = live.program_options({"cell_statistic": "multiscale", "cfar": {"range_edge_correction": "pulse"}})
    assert flat["cfar_range_edge_correction"] == "pulse"
    assert json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))["cfar_range_edge_correction"] == "pulse"
    assert "cfar_range_edge_correction" not in live.program_options({"cell_statistic": "multiscale"})
    assert "cfar_range_edge_correction" not in json.loads(core.live_options_from_text(""))
