"""The C++ far-field beamformers and array calibration (libs/live beamform, calibrate) against
the NumPy stages they replace (gpusonar.flank, gpusonar.calibrate), on simulated pings, and
the C++ chain with `beamformer: farfield | mvdr` against the Python chain."""

import json
import subprocess

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import calibrate, flank, recon
from gpusonar.live import LiveChain, cpp_chain_options
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel
from gpusonar.sonar_equation import noise_rms_volts
from test_calibration_sync import calibration_pings
from test_detect import scene_ping, target_snr_db
from test_live import CHAIN, N, copies, pings  # noqa: F401  (pings is a fixture)
from test_live_cpp import CPP_CHAIN, EXE, EXE_OPTIONS

core = gs.core
C = 1500.0
BEARINGS = np.radians(np.arange(-40.0, 40.01, 1.0))


def beam_map(result: dict) -> flank.BeamMap:
    return flank.BeamMap(**result)


def worst(a, b) -> float:
    """Largest difference between two maps, as a fraction of the reference map's peak."""
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.fixture(scope="module")
def jammed():
    """A target at 2 km, 6 degrees, in ambient noise, and a broadband noise source at -20 degrees,
    30 dB above the ambient noise (the scene of test_detect's MVDR tests)."""
    sonar, rec = scene_ping([(2000.0, 6.0)], [0.0], noise_db=50.0, seed=2)
    ambient = noise_rms_volts(50.0, -180.0, sonar.sample_rate)
    flank.add_jammer(rec, np.radians(-20.0), ambient * 10 ** (30 / 20), np.random.default_rng(9))
    recon.prepare([rec])
    return rec


# --- far field -------------------------------------------------------------------------------

@pytest.mark.parametrize("window", ["rect", "hann", "hamming"])
def test_farfield_matches_numpy(jammed, window):
    want = flank.farfield_beamform(jammed.copy(), BEARINGS, window=window, engine="numpy")
    got = beam_map(core.farfield_beamform_cpp(jammed.copy(), list(BEARINGS), window))
    assert got.data.shape == want.data.shape == (81, jammed.samples.n_samples) and got.data.dtype == np.complex64
    # Both work in single precision; they differ in the order of the sums and in the FFT.
    assert worst(got.data, want.data) < 1e-5
    assert got.n_valid == want.n_valid == jammed.samples.n_samples - 1200      # a 0.1 s pulse at 12 kHz
    assert np.array_equal(got.bearings, want.bearings)
    assert np.allclose(got.times, want.times, rtol=0, atol=1e-12) and np.allclose(got.ranges, want.ranges, rtol=0, atol=1e-9)
    assert np.array_equal(got.valid().data, got.data[:, :got.n_valid])
    assert np.abs(core.steering_delays_cpp(jammed, list(BEARINGS)) - flank.steering_delays(jammed, BEARINGS)).max() < 1e-15


def test_farfield_conditions_a_raw_record_like_numpy_does():
    """Given a raw real-passband ping, both stages convert and compress it in place first."""
    _, rec = scene_ping([(2040.0, 12.0)], [10.0], noise_db=50.0, seed=3)
    a, b = rec.copy(), rec.copy()
    assert not a.samples.is_complex()
    want = flank.farfield_beamform(a, BEARINGS, engine="numpy")
    got = beam_map(core.farfield_beamform_cpp(b, list(BEARINGS)))
    assert b.samples.is_complex() and json.loads(b.metadata.extra_json)["gpusonar"]["compressed"]
    assert worst(got.data, want.data) < 1e-5
    # The target is where it was put: 2040 m, 12 degrees.
    ib, it = np.unravel_index(np.argmax(np.abs(got.valid().data)), got.valid().data.shape)
    assert np.degrees(BEARINGS[ib]) == pytest.approx(12.0, abs=0.5) and got.ranges[it] == pytest.approx(2040.0, abs=1.0)
    # A second call finds the record already conditioned and gives the same map.
    again = beam_map(core.farfield_beamform_cpp(b, list(BEARINGS)))
    assert np.array_equal(again.data, got.data)


@pytest.mark.parametrize("layout", ["dense", "sparse"])
def test_farfield_matches_numpy_on_another_array(layout):
    """Eight elements, 20 to 35 kHz, filled or thinned to 3 m: nothing is specific to flank_32."""
    sonar = SonarModel.load("popoto_8", {"array": {"layout_choice": layout}, "receive": {"window": {"range": [1440.0, 1560.0]}},
                                         "transmit": {"waveform": {"duration": 0.05}}})
    b = np.radians(-23.0)
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([1500.0 * np.array([np.sin(b), np.cos(b), 0.0])]),
                        environment={"sound_speed": C}, absorption_model="none", adc=False)
    bearings = np.radians(np.linspace(-60.0, 60.0, 241))
    want = flank.farfield_beamform(rec.copy(), bearings, window="hann", engine="numpy")
    got = beam_map(core.farfield_beamform_cpp(rec.copy(), list(bearings), "hann"))
    assert worst(got.data, want.data) < 1e-5
    ib, _ = np.unravel_index(np.argmax(np.abs(got.data)), got.data.shape)
    assert np.degrees(bearings[ib]) == pytest.approx(-23.0, abs=0.5)


# --- MVDR ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{}, {"n_subbands": 12, "loading": 0.1, "block": 200, "guard_blocks": 2}])
def test_mvdr_matches_numpy(jammed, kw):
    """Default settings, and ones where the last block is short and the guard is wider."""
    want = flank.mvdr_beamform(jammed.copy(), BEARINGS, engine="numpy", **kw)
    got = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), **kw))
    assert got.data.shape == want.data.shape
    # Looser than the conventional beamformer: the weights come from inverting covariances that
    # NumPy accumulates in single precision.
    assert worst(got.data, want.data) < 1e-4
    assert got.n_valid == want.n_valid


def test_mvdr_cpp_finds_the_target_under_a_jammer_that_hides_it_from_the_conventional_beamformer(jammed):
    conv = beam_map(core.farfield_beamform_cpp(jammed.copy(), list(BEARINGS)))
    mvdr = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS)))
    i_t = int(np.argmin(np.abs(BEARINGS - np.radians(6.0))))
    i_j = int(np.argmin(np.abs(BEARINGS - np.radians(-20.0))))
    snr_c, peak_c = target_snr_db(conv, i_t, 2000.0)
    snr_m, peak_m = target_snr_db(mvdr, i_t, 2000.0)
    assert snr_m > snr_c + 10.0, (snr_c, snr_m)
    assert peak_m == pytest.approx(peak_c, rel=0.25)          # the target itself is untouched
    # Looking straight at the jammer, nothing is rejected: that is the look direction.
    jam_c = np.mean(np.abs(conv.valid().data[i_j]) ** 2)
    jam_m = np.mean(np.abs(mvdr.valid().data[i_j]) ** 2)
    assert 10 * np.log10(jam_m / jam_c) == pytest.approx(0.0, abs=1.5)


def test_bad_input_is_refused(jammed):
    with pytest.raises(ValueError, match="no training data"):
        core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), block=jammed.samples.n_samples // 2)
    with pytest.raises(ValueError):
        flank.mvdr_beamform(jammed.copy(), BEARINGS, block=jammed.samples.n_samples // 2)
    with pytest.raises(ValueError, match="window"):
        core.farfield_beamform_cpp(jammed.copy(), list(BEARINGS), "kaiser")
    with pytest.raises(ValueError, match="n_subbands"):
        core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), n_subbands=0)


# --- calibration -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def injected():
    rng = np.random.default_rng(11)
    n = 32
    position = np.zeros((n, 3))
    position[:, :2] = rng.normal(0.0, 5e-3, (n, 2))             # 5 mm RMS in the array plane
    return {"gain": 10 ** (rng.normal(0.0, 1.0, n) / 20),       # 1 dB RMS
            "phase": np.radians(rng.normal(0.0, 20.0, n)),      # 20 degrees RMS
            "position": position}


@pytest.fixture(scope="module")
def calibration(injected):
    sonar, records, sources = calibration_pings(injected)
    recon.prepare(records)
    return sonar, records, sources


def test_channel_responses_match_numpy(calibration):
    _, records, sources = calibration
    for rec, src in zip(records, sources):
        want = calibrate.channel_responses(rec.copy(), src)
        got = core.calibrate_channel_responses_cpp(rec.copy(), src)
        assert got.shape == want.shape == (32,)
        assert np.abs(got - want).max() < 1e-5 * np.abs(want).max()


@pytest.mark.parametrize("planar", [True, False])
def test_calibration_estimate_matches_numpy_and_recovers_the_injected_errors(calibration, injected, planar):
    _, records, sources = calibration
    want = calibrate.estimate_channel_errors(copies(records), sources, planar=planar)
    got = calibrate.ChannelErrors(**core.calibrate_estimate_cpp(copies(records), sources, planar))
    assert np.abs(20 * np.log10(got.gain / want.gain)).max() < 1e-4            # dB
    assert np.degrees(np.abs(got.phase - want.phase)).max() < 1e-3
    assert np.abs(got.position - want.position).max() < 1e-7                  # m
    assert got.residual_rms_deg == pytest.approx(want.residual_rms_deg, abs=1e-3)
    # And against the truth, in the gauge the estimator reports: mean gain 1, mean phase 0, mean offset 0.
    gain = injected["gain"] / injected["gain"].mean()
    phase = injected["phase"] - injected["phase"].mean()
    position = injected["position"] - injected["position"].mean(axis=0)
    assert np.abs(20 * np.log10(got.gain / gain)).max() < 0.02                 # dB, against 1 dB RMS injected
    assert np.degrees(np.abs(got.phase - phase)).max() < 1.0
    assert np.abs(got.position[:, :2] - position[:, :2]).max() < 0.3e-3        # 0.3 mm
    assert np.abs(got.position[:, 2]).max() < 1e-9                             # sources in the plane see nothing out of it
    assert got.residual_rms_deg < 0.5


def test_calibration_needs_enough_sources(calibration):
    _, records, sources = calibration
    with pytest.raises(ValueError, match="at least 3 sources"):
        core.calibrate_estimate_cpp(copies(records[:2]), sources[:2])
    with pytest.raises(ValueError, match="one source position per"):
        core.calibrate_estimate_cpp(copies(records), sources[:3])


def test_applying_the_calibration_in_cpp_restores_the_beam(calibration, injected):
    sonar, records, sources = calibration
    est = calibrate.ChannelErrors(**core.calibrate_estimate_cpp(copies(records), sources))
    # A fresh ping on a target that was not one of the calibration sources.
    b = np.radians(11.0)
    target = 1000.0 * np.array([np.sin(b), np.cos(b), 0.0])

    def ping(errors):
        return simulate_ping(sonar, make_pose(), Scene().add_points([target]), environment={"sound_speed": C},
                             absorption_model="none", adc=False, channel_errors=errors)

    bearings = np.radians(np.linspace(-20.0, 40.0, 241))

    def pattern(rec):
        bm = beam_map(core.farfield_beamform_cpp(rec, list(bearings)))
        return np.abs(bm.data).max(axis=1)

    ideal, raw = pattern(ping(None)), pattern(ping(injected))
    # C++ application against the Python one: the same samples and the same tables afterwards.
    a, b_ = ping(injected), ping(injected)
    recon.prepare([a])
    calibrate.store_calibration(a, est)
    core.apply_calibration(a)
    core.calibrate_apply_cpp(b_, est.gain, est.phase, est.position)
    assert np.abs(a.samples.array - b_.samples.array).max() < 1e-6 * np.abs(a.samples.array).max()
    assert b_.calibration.gain == [1.0] * 32 and b_.calibration.phase == [0.0] * 32
    assert np.allclose(np.array(b_.calibration.position_offset), est.position, atol=1e-12)
    c = ping(injected)
    core.calibrate_store_cpp(c, est.gain, est.phase, est.position)
    assert np.allclose(c.calibration.gain, est.gain) and np.allclose(c.calibration.phase, est.phase)
    fixed = pattern(b_)
    assert raw.max() < 0.985 * ideal.max()                        # errors cost coherent gain
    # Calibration gives it back, up to the one thing it cannot see: the array's mean gain.
    common = injected["gain"].mean()
    assert fixed.max() == pytest.approx(ideal.max() * common, rel=0.01)
    far = np.abs(np.degrees(bearings) - 11.0) > 15.0
    assert raw[far].mean() > 1.5 * ideal[far].mean()
    assert fixed[far].mean() < 0.75 * raw[far].mean()


# --- the chain -------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["farfield", "mvdr"])
def test_cpp_chain_with_a_farfield_beamformer_gives_the_python_chains_report(pings, name):
    with LiveChain(dict(CHAIN, beamformer=name)) as chain:
        want = chain.run(copies(pings))
        want_power, want_ranges = chain.last["power"], chain.last["ranges"]
    cpp = core.LiveChainCpp(dict(CPP_CHAIN, beamformer=name))
    got = [json.loads(cpp.process(r)) for r in copies(pings)]
    power, ranges, bearings = cpp.last_map()
    # The map the detector sees: the same cells, the same power.
    assert power.shape == want_power.shape and np.allclose(ranges, want_ranges, rtol=0, atol=1e-9)
    assert worst(power, want_power) < 1e-4
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"]) >= 2
        for a, b in zip(w["detections"], g["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
        assert [(t["id"], t["kind"], t["parent"]) for t in w["tracks"]] == [(t["id"], t["kind"], t["parent"]) for t in g["tracks"]]
        for a, b in zip(w["tracks"], g["tracks"]):
            for key in ("x", "y", "vx", "vy", "range_m"):
                assert a[key] == pytest.approx(b[key], abs=0.05), key
    # Both targets are tracked and the crossing one is called a mover. (With MVDR the standing one
    # is left "uncertain" by this scene's six pings, in the Python chain as well.)
    kinds = [t["kind"] for t in got[-1]["tracks"]]
    assert len(kinds) >= 2 and "mover" in kinds and (name == "mvdr" or "static" in kinds)


def test_python_chain_passes_the_beamformer_to_the_cpp_engine(pings):
    assert cpp_chain_options(dict(CHAIN, beamformer="mvdr"))["beamformer"] == "mvdr"
    assert cpp_chain_options(CHAIN)["beamformer"] == "backprojection"
    with pytest.raises(ValueError, match="beamformer"):
        cpp_chain_options(dict(CHAIN, beamformer="capon"))
    with pytest.raises(ValueError, match="beamformer"):
        core.LiveChainCpp(dict(CPP_CHAIN, beamformer="capon"))
    out = {}
    for engine in ("python", "cpp"):
        with LiveChain(dict(CHAIN, beamformer="farfield", engine=engine)) as chain:
            out[engine] = chain.run(copies(pings))[-1]
    t_end = (N - 1) * 2.0
    for x, y in ((40.0 - 3.0 * t_end, 250.0), (-60.0, 320.0)):
        a = min(out["python"]["tracks"], key=lambda t: np.hypot(t["x"] - x, t["y"] - y))
        b = min(out["cpp"]["tracks"], key=lambda t: np.hypot(t["x"] - x, t["y"] - y))
        assert a["kind"] == b["kind"] and np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 0.1
        assert np.hypot(b["x"] - x, b["y"] - y) < 6.0


@pytest.mark.skipif(not EXE.exists(), reason="build/bin/sonar_live not built")
def test_program_takes_the_beamformer_option(pings, tmp_path):
    with LiveChain(dict(CHAIN, beamformer="farfield"), recording=str(tmp_path / "rec.h5")) as chain:
        want = chain.run(copies(pings))
    out = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), "--report", str(tmp_path / "rep.jsonl"), *EXE_OPTIONS,
                          "classify_min_speed=1", "beamformer=farfield"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    got = [json.loads(line) for line in (tmp_path / "rep.jsonl").read_text().splitlines()]
    assert len(got) == N
    for w, g in zip(want, got):
        assert len(w["detections"]) == len(g["detections"]) >= 2
        for a, b in zip(w["detections"], g["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
    bad = subprocess.run([str(EXE), "--replay", str(tmp_path / "rec.h5"), "beamformer=capon"], capture_output=True, text=True, timeout=300)
    assert bad.returncode != 0 and "beamformer" in bad.stderr
