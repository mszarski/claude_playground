"""The chain's detector at the range resolution (`cell_statistic: multiscale`; libs/live detector.hpp, libs/detect
statistic.hpp, gpusonar.detect.multiscale) against closed forms, Monte Carlo and simulated pings, on both engines:

  - the thresholds of its statistics (one sample of several rows, the mean of correlated samples) against the
    chi-square closed form and against band-limited noise drawn here; what the detector until now really asked for;
  - on noise-only pings: the false-alarm budget is kept (measured at a budget large enough to count), nothing at the
    default budget, nothing from the detector as it was;
  - a point echo reads its budget SNR less the window and the straddling (2.3 dB, where the mean of a 2 m cell read
    9.4 dB less), is found where `detection_probability` says and the detector as it was is not;
  - two targets further apart than the stated resolution are two detections and two tracks, nearer are one;
  - `cell_statistic: mean` is the detector as it was: the chain's detections recomputed here from the map with the
    cell-averaging detector alone;
  - both engines give the same detections, with stacked rows too, and the same background at the ends of a map.
"""
import json

import numpy as np
import pytest
from scipy import stats

from gpusonar import core, detect, flank, live, recon
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

WINDOW = {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1700.0]}}}
KW = dict(environment={"sound_speed": 1500.0}, adc=False, absorption_model="none")
POSE = dict(position=(0.0, 0.0, 0.0))
NOISE = 2e-4
PING_INTERVAL_NS = 2_000_000_000   # pings 2 s apart: a track's points need time between them for a fit (D-156)


@pytest.fixture(scope="module")
def sonar():
    return SonarModel.load("popoto_long_range", WINDOW)


@pytest.fixture(scope="module")
def template(sonar):
    return sonar.make_record(core.Pose(), environment={"sound_speed": 1500.0})


def at(r, bearing_deg, z=0.0):
    b = np.radians(bearing_deg)
    return [r * np.sin(b), r * np.cos(b), z]


def ping(sonar, points, ts_db, seed, noise=NOISE, sequence=0):
    scene = Scene()
    for p, ts in zip(points, np.broadcast_to(ts_db, len(points))):
        scene.add_points([p], target_strength_db=float(ts))
    return simulate_ping(sonar, make_pose(**POSE), scene, noise_rms=noise, rng=np.random.default_rng(seed), sequence=sequence,
                         time_ns=sequence * PING_INTERVAL_NS, **KW)


def budget_snr_db(sonar, r, bearing_deg, ts_db=0.0, noise=NOISE, seed=1):
    """What a link budget states for this target: the peak of the noiseless, unweighted beam output over the mean noise
    power of the same beamformer."""
    quiet = ping(sonar, [at(r, bearing_deg)], ts_db, seed, noise=0.0)
    only = ping(sonar, [at(r, 0.0)], -300.0, seed + 1, noise=noise)
    recon.prepare([quiet, only], backend=core.Backend.Host)
    peak = np.abs(flank.backprojection_map(quiet, r + np.arange(-2.0, 2.0, 0.01), np.radians([bearing_deg]), window="rect", interp="linear",
                                           backend=core.Backend.Host).image) ** 2
    noise_map = np.abs(flank.backprojection_map(only, np.arange(1000.0, 1600.0, 0.3), np.radians(np.arange(-30.0, 30.0, 2.0)), window="rect",
                                                interp="linear", backend=core.Backend.Host).image) ** 2
    return float(10.0 * np.log10(peak.max() / noise_map.mean()))


# --- the statistics --------------------------------------------------------------------------------------------------

def band_noise(rng, n_trials, n, step_over_resolution):
    """Complex noise of unit power and a flat band, sampled `step_over_resolution` of a range resolution apart."""
    m = int(2 ** np.ceil(np.log2(n + 64)))
    z = (rng.standard_normal((n_trials, m)) + 1j * rng.standard_normal((n_trials, m))) / np.sqrt(2.0)
    spectrum = np.fft.fft(z, axis=1) * (np.abs(np.fft.fftfreq(m)) <= 0.5 * step_over_resolution)
    return np.fft.ifft(spectrum, axis=1)[:, 32:32 + n] / np.sqrt(step_over_resolution)


def test_thresholds_against_the_chi_square_closed_form_and_band_limited_noise(template):
    # One sample of `rows` maps added in power: Erlang, 2 x rows degrees of freedom.
    for rows in (1, 2, 3):
        for pfa in (1e-3, 1e-6, 1e-9):
            want = stats.gamma.isf(pfa, rows) / rows
            assert core.weighted_exponential_threshold([1.0], rows, pfa) == pytest.approx(want, rel=1e-9)
    # The mean of the 8 samples of a 2 m cell (a quarter of a metre apart, 0.375 m resolution) and of four cells, in noise
    # of a flat band: its samples correlate by the sinc.
    corr = np.sinc(np.arange(32) * 0.25 / 0.375).tolist()
    rng = np.random.default_rng(3)
    power = np.abs(band_noise(rng, 1500, 4096, 0.25 / 0.375)) ** 2
    assert power.mean() == pytest.approx(1.0, abs=0.01)
    for n, pfas in ((8, (1e-2, 1e-3, 1e-4)), (32, (1e-2, 1e-3))):
        w = core.mean_power_weights(corr, n)
        assert sum(w) == pytest.approx(1.0)
        stat = power.reshape(1500, -1, n).mean(axis=2)
        assert stat.var() == pytest.approx(1.0 / core.equivalent_looks(w), rel=0.05)
        for pfa in pfas:
            t = core.weighted_exponential_threshold(w, 1, pfa)
            assert np.mean(stat > t) == pytest.approx(pfa, rel=0.25), (n, pfa)       # 77 to 7700 exceedances expected
        # ... and of three rows of it.
        three = (stat + np.abs(band_noise(rng, 1500, 4096, 0.25 / 0.375) ** 2).reshape(1500, -1, n).mean(axis=2)
                 + np.abs(band_noise(rng, 1500, 4096, 0.25 / 0.375) ** 2).reshape(1500, -1, n).mean(axis=2)) / 3.0
        assert np.mean(three > core.weighted_exponential_threshold(w, 3, 1e-3)) == pytest.approx(1e-3, rel=0.3)
    # What the detector until now asked of that mean: the one-sample factor for 1e-5 (10.7 dB), which the mean of a cell
    # exceeds in noise about once in 1e21 cells, not once in 1e5: 6 dB of threshold nobody had chosen.
    w8 = core.mean_power_weights(corr, 8)
    factor = detect.cfar_threshold_factor(1e-5, 300)
    assert 10.0 * np.log10(factor) == pytest.approx(10.7, abs=0.05)
    assert core.weighted_exponential_tail(w8, 1, factor) < 1e-18
    assert 10.0 * np.log10(core.weighted_exponential_threshold(w8, 1, 1e-5)) == pytest.approx(5.9, abs=0.1)
    # The background estimated from n looks: the cell-averaging factor is the one-sample case.
    assert -np.log(1e-5) * core.training_correction(1e-5, 300) == pytest.approx(factor)


def test_noise_through_the_matched_filter_correlates_as_the_replica_and_a_cell_mean_holds_its_looks(template):
    """White noise through the matched filter has the replica's power spectrum, so its samples correlate by the replica's
    autocorrelation: for the long-range sweep under its 10 % Tukey taper that is narrower in band than the flat band's
    sinc, and the mean of a 2 m cell holds 5.48 looks where the sinc gives 5.80 (the plan assumed the sinc; the chain's
    map measured 5.42). The replica is built here in NumPy, not taken from the library; then noise is put through it
    and the variance of the cell means is counted."""
    from scipy.signal import fftconvolve
    from scipy.signal.windows import tukey
    w = template.transmit.waveform
    band, c, step = abs(w.f_stop - w.f_start), template.environment.sound_speed, 0.25
    assert (w.duration, band, w.window_alpha) == (1.0, 2000.0, 0.1)
    fs = 3.0 * c / (2.0 * step)                                   # three replica samples per map sample: 9 kHz
    t = (np.arange(int(round(w.duration * fs))) + 0.5) / fs
    replica = tukey(len(t), w.window_alpha) * np.exp(1j * np.pi * band / w.duration * (t - 0.5 * w.duration) ** 2)
    auto = np.array([np.vdot(replica[:len(replica) - 3 * d], replica[3 * d:]).real for d in range(32)]) / np.vdot(replica, replica).real
    corr = np.asarray(core.live_sample_correlation(template, step, 32))
    assert corr == pytest.approx(auto, abs=2e-3)
    sinc = np.sinc(np.arange(32) * step / 0.375)
    assert abs(corr[1] - sinc[1]) > 0.01 and corr[1] > sinc[1]    # a narrower band: neighbouring samples are more alike
    looks = core.equivalent_looks(core.mean_power_weights(corr.tolist(), 8))
    assert looks == pytest.approx(5.48, abs=0.03) and core.equivalent_looks(core.mean_power_weights(sinc.tolist(), 8)) == pytest.approx(5.80, abs=0.03)
    untapered = template.copy()
    untapered.transmit.waveform.window_alpha = 0.0                # without the taper the sweep's band is flat: the sinc's looks
    flat = core.equivalent_looks(core.mean_power_weights(core.live_sample_correlation(untapered, step, 32), 8))
    assert flat == pytest.approx(5.80, abs=0.05)
    rng = np.random.default_rng(5)
    n = 3 * 8 * 6000                                              # 6000 cells of 8 map samples a trial
    stat = []
    for _ in range(4):
        z = (rng.standard_normal(n + len(replica)) + 1j * rng.standard_normal(n + len(replica))) / np.sqrt(2.0)
        y = fftconvolve(z, np.conj(replica[::-1]), mode="valid")[:n:3]          # the matched filter, at the map's samples
        power = np.abs(y) ** 2
        stat.append((power / power.mean()).reshape(-1, 8).mean(axis=1))
    stat = np.concatenate(stat)
    assert 1.0 / stat.var() == pytest.approx(looks, rel=0.04)     # 24000 cell means: 1 % of statistical error on the variance
    assert abs(1.0 / stat.var() - 5.80) > 0.15
    assert live.detector_plan(template, {"bearings_deg": [-60.0, 60.0, 0.5]})["scales"][1]["looks"] == pytest.approx(looks)


def test_the_plan_states_its_tests_and_keeps_the_budget_between_them(template):
    plan = live.detector_plan(template, {"bearings_deg": [-60.0, 60.0, 0.5]})
    sample, cell = plan["scales"]
    assert (sample["cells"], cell["cells"]) == (0, 1) and plan["samples_per_cell"] == 8 and plan["cell_m"] == 2.0
    assert sample["tests"] * sample["pfa"] + cell["tests"] * cell["pfa"] == pytest.approx(1e-3)
    assert sample["factor"] == pytest.approx(-np.log(sample["pfa"]))
    assert cell["looks"] == pytest.approx(5.48, abs=0.05) and plan["merge_cells"] == 7
    assert live.detection_threshold_db({"cell_statistic": "multiscale"}, record=template) == pytest.approx(12.9, abs=0.3)
    assert live.detection_threshold_db({"cell_statistic": "multiscale"}, record=template, rows=3) == pytest.approx(9.4, abs=0.3)
    # How the tests are counted (detector.hpp): places = beams x extent / resolution (or / cell); the sample's tests grow
    # as u = rows x threshold, the cell mean's as sqrt(u) x the statistic's hazard, u = rows x looks x threshold; the
    # constants are the measured ones for the number of rows.
    extent = plan["cell_m"] * 399                                  # the 0.8 km window in whole 2 m cells
    for rows, c_sample, c_mean in ((1, 0.191, 0.515), (2, 0.208, 0.532), (3, 0.192, 0.598), (4, 0.192, 0.598)):
        p = live.detector_plan(template, {"bearings_deg": [-60.0, 60.0, 0.5]}, rows=rows)
        sm, cl = p["scales"]
        assert (sm["search"], cl["search"]) == (c_sample, c_mean)                       # four rows: the three-row values (not measured)
        assert sm["tests"] == pytest.approx(c_sample * p["beams"] * extent / p["resolution_m"] * rows * sm["factor"], rel=1e-9)
        x, a = cl["factor"], rows * cl["looks"]
        h = 1e-4 * x
        hazard = (np.log(core.weighted_exponential_tail(cl["weights"], rows, x - h)) - np.log(core.weighted_exponential_tail(cl["weights"], rows, x + h))) / (2 * h * a)
        assert cl["tests"] == pytest.approx(c_mean * p["beams"] * extent / p["cell_m"] * np.sqrt(a * x) * hazard, rel=1e-6)
        assert 0.4 < hazard < 1.0
        for sc in (sm, cl):                                         # each test's threshold is where tail x tests is its share
            assert core.weighted_exponential_tail(sc["weights"], rows, sc["factor"]) * sc["tests"] == pytest.approx(0.5e-3, rel=1e-6)
            assert sc["pfa"] == pytest.approx(core.weighted_exponential_tail(sc["weights"], rows, sc["factor"]))
    # A typed factor is the detector as it was: that many tests per place whatever the threshold and the rows.
    for rows in (1, 3):
        old = live.detector_plan(template, {"bearings_deg": [-60.0, 60.0, 0.5], "cfar": {"search_factor": 2.5}}, rows=rows)
        new = live.detector_plan(template, {"bearings_deg": [-60.0, 60.0, 0.5]}, rows=rows)
        assert old["scales"][0]["tests"] == pytest.approx(2.5 * old["beams"] * extent / old["resolution_m"]) and old["scales"][0]["search"] == 2.5
        assert old["scales"][1]["tests"] == pytest.approx(2.5 * old["beams"] * extent / old["cell_m"])
        moved = [10.0 * np.log10(n_["factor"] / o_["factor"]) for n_, o_ in zip(new["scales"], old["scales"])]
        assert 0.05 < moved[0] < 0.2 and abs(moved[1]) < 0.1        # the sample's threshold rises by about a tenth of a decibel
    with pytest.raises(ValueError):
        live.detection_threshold_db({"cell_statistic": "multiscale"})                 # it follows from the map: no record, no number
    longer = live.detector_plan(template, {"cfar": {"integrate_cells": [1, 4, 16], "false_alarms_per_ping": 1e-2}})
    assert [s["cells"] for s in longer["scales"]] == [0, 1, 4, 16]
    assert sum(s["tests"] * s["pfa"] for s in longer["scales"]) == pytest.approx(1e-2)
    assert [s["factor"] for s in longer["scales"]] == sorted((s["factor"] for s in longer["scales"]), reverse=True)


def test_background_follows_a_slope_at_the_ends_and_both_engines_compute_the_same():
    rng = np.random.default_rng(5)
    n_b, n, guard, train = 6, 900, 50, 150
    level = np.exp(-np.log(2.0) / 300.0 * np.arange(n))                               # 3 dB per 300 cells, as reverberation falls
    power = (level[None, :] * rng.gamma(40.0, 1.0 / 40.0, (n_b, n))).astype(np.float32)
    got, worth = detect.background(power, guard, train)
    plain, _ = detect._window_mean(power.astype(float), guard, train)
    assert np.abs(10.0 * np.log10(got / level)).max() < 0.35
    assert 10.0 * np.log10(plain[:, 0] / level[0]).mean() == pytest.approx(-1.25, abs=0.15)   # what the plain mean reads at the near end
    assert np.array_equal(got[:, 200:700], plain[:, 200:700])                         # balanced windows: the plain mean
    cpp, cpp_worth = core.live_detector_background(power, guard, train)
    assert np.allclose(cpp, got, rtol=1e-9) and np.allclose(cpp_worth, worth, rtol=1e-12)
    assert worth[450] == 300.0 and worth[0] < 40.0


# --- noise alone -----------------------------------------------------------------------------------------------------

def test_noise_only_pings_keep_the_false_alarm_budget_and_give_nothing_at_the_default(sonar, monkeypatch):
    """Six pings of noise. At a budget of 3 false detections a ping (so that they can be counted) the detector reports
    1.5 to 6 a ping (3.3 measured on 60 pings of the 12 km window); at the default (one in a thousand pings) nothing, and
    the detector as it was nothing either."""
    seen = {}
    plain = detect.multiscale

    def spy(*args):
        seen["args"] = args
        return plain(*args)

    monkeypatch.setattr(detect, "multiscale", spy)
    base = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host"}
    counted, default, was = 0, 0, 0
    with LiveChain(dict(base, engine="python", cell_statistic="multiscale", cfar={"false_alarms_per_ping": 3.0})) as loose, \
            LiveChain(dict(base, engine="cpp", cell_statistic="multiscale")) as chain, LiveChain(dict(base, engine="cpp", cell_statistic="mean")) as old:
        for i in range(6):
            rec = ping(sonar, [at(1300.0, 0.0)], -300.0, 700 + i, sequence=i)
            counted += len(loose.process(rec.copy())["detections"])
            default += len(chain.process(rec.copy())["detections"])
            was += len(old.process(rec.copy())["detections"])
        assert loose.detector["scales"][0]["threshold_db"] < chain.detector["scales"][0]["threshold_db"] - 1.0
    assert 9 <= counted <= 36, counted
    assert default == 0 and was == 0
    # The same map through the NumPy detector at the default budget: nothing.
    args = seen["args"]
    default_plan = live.detector_plan(rec, base)
    assert len(plain(*args[:-1], default_plan)[0]) == 0
    assert args[-1]["scales"][0]["pfa"] * args[-1]["scales"][0]["tests"] == pytest.approx(3e3 * default_plan["scales"][0]["pfa"] * default_plan["scales"][0]["tests"])
    assert args[-1]["scales"][0]["pfa"] > 1e3 * default_plan["scales"][0]["pfa"]       # fewer tests are counted at the lower threshold


def test_one_row_and_three_rows_keep_the_same_false_alarm_budget():
    """tools/detector_search_factor.py --verify, short: 40 noise-only pings of one row and of three rows through the
    chain's own map and the detector as shipped, at a budget of 3 false detections a ping (120 expected each, 9 % of
    statistical error). The tool's 300 pings a row count read 0.96 and 0.87 of the budget (0.03); with the one flat factor
    the detector had, three rows read 1.4 times what one row did. Here: both between 0.65 and 1.25 of the budget, and
    three rows no more than 1.25 times one row."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "tools" / "detector_search_factor.py"
    spec = importlib.util.spec_from_file_location("detector_search_factor", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    got = tool.verify([1, 3], 40, [3.0], 9000, log=lambda *_: None)
    one, three = got["1"]["3.0"], got["3"]["3.0"]
    assert 0.65 < one["ratio"] < 1.25 and 0.65 < three["ratio"] < 1.25, (one, three)
    assert three["found"] < 1.25 * one["found"]


# --- a point echo ----------------------------------------------------------------------------------------------------

def test_a_point_echo_reads_its_budget_snr_less_window_and_straddling_on_both_engines(sonar, template):
    """The reference is the link budget's number (peak of the noiseless unweighted beam over mean noise). The multiscale
    detector's `snr_db` is 2.3 dB below it (Hann window 1.76 dB, the echo between samples 0.5 dB); the mean of a 2 m cell
    was 9.4 dB below. The range is held to the centroid's 1 m, so range_estimate centroid is given (D-158: the default
    leading edge of a 0 dB point echo lies 1.4 m before it, as D-157 measured about 1 m)."""
    loss = live.snr_loss_db(template, 2.0, "hann", "multiscale")
    assert loss["window_db"] == pytest.approx(1.76, abs=0.02) and loss["cell_db"] == pytest.approx(0.55, abs=0.15)
    assert live.snr_loss_db(template, 2.0)["total_db"] == pytest.approx(9.4, abs=0.2)             # as before
    read, want = [], []
    for trial in range(3):
        rng = np.random.default_rng(40 + trial)
        r_t, b_t = 1300.0 + rng.uniform(-3.0, 3.0), rng.uniform(-10.0, 10.0)
        want.append(budget_snr_db(sonar, r_t, b_t, ts_db=0.0))
        rec = ping(sonar, [at(r_t, b_t)], 0.0, 60 + trial)
        got = {}
        for engine in ("cpp", "python"):
            with LiveChain({"cell_statistic": "multiscale", "backend": "host", "engine": engine,
                            "cfar": {"range_estimate": "centroid"}}) as chain:
                rep = chain.process(rec.copy())
            assert len(rep["detections"]) == 1, engine
            got[engine] = d = rep["detections"][0]
            assert abs(d["range_m"] - r_t) < 1.0 and abs(d["bearing_deg"] - b_t) < 0.5 and d["extent_m"] <= 6.0, (engine, d)
        assert got["python"]["snr_db"] == pytest.approx(got["cpp"]["snr_db"], abs=0.02)
        assert got["python"]["range_m"] == pytest.approx(got["cpp"]["range_m"], abs=1e-3)
        read.append(got["cpp"]["snr_db"])
    assert float(np.mean(want) - np.mean(read)) == pytest.approx(loss["total_db"], abs=0.8)


def test_detection_probability_is_what_the_chain_does(sonar, template):
    """At a budget SNR of 17 dB a point echo is detected nine times in ten by the multiscale detector and almost never
    by the detector as it was; the closed form says so and twelve simulated pings agree. The thresholds a range
    prediction should use follow."""
    new, old = {"cell_statistic": "multiscale"}, {"cell_statistic": "mean"}
    assert live.detection_probability(17.0, template, new) == pytest.approx(0.92, abs=0.05)
    assert live.detection_probability(17.0, template, old) < 0.01
    curve = live.detection_probability(np.arange(8.0, 26.0, 1.0), template, new)
    assert np.all(np.diff(curve) >= 0.0) and curve[0] < 1e-3 and curve[-1] > 0.999
    # Pd one half: 15.0 dB of budget SNR where the detector as it was needs 19.7 (its own formula says 20.1: a threshold, not a probability).
    assert live.budget_threshold_db(template, new) == pytest.approx(15.0, abs=0.3)
    assert live.budget_threshold_db(template, old, pd=0.5) == pytest.approx(19.7, abs=0.3)
    assert live.budget_threshold_db(template, old) == pytest.approx(20.1, abs=0.2)
    assert live.budget_threshold_db(template, new, pd=0.9) == pytest.approx(16.7, abs=0.3)
    # Rows added in power need less of each row (the detector as it was gained nothing from them: its threshold on the
    # mean did not move); a fluctuating echo needs more for a high probability.
    assert live.budget_threshold_db(template, new, rows=3) == pytest.approx(11.1, abs=0.3)
    assert live.budget_threshold_db(template, new, rows=2) == pytest.approx(12.5, abs=0.3)
    assert live.budget_threshold_db(template, old, rows=3, pd=0.5) == pytest.approx(19.7, abs=0.3)
    assert live.budget_threshold_db(template, new, pd=0.9, fluctuation="swerling1") > live.budget_threshold_db(template, new, pd=0.9) + 5.0
    with pytest.raises(ValueError):
        live.detection_probability(17.0, template, new, fluctuation="rician")
    r_t, b_t = 1300.0, 4.0
    snr0 = budget_snr_db(sonar, r_t, b_t)
    noise = NOISE * 10.0 ** ((snr0 - 17.0) / 20.0)
    found = {"multiscale": 0, "mean": 0}
    for trial in range(12):
        rng = np.random.default_rng(900 + trial)
        r, b = r_t + rng.uniform(-3.0, 3.0), b_t + rng.uniform(-3.0, 3.0)
        rec = ping(sonar, [at(r, b)], 0.0, 300 + trial, noise=noise)
        for name in found:
            with LiveChain({"cell_statistic": name, "backend": "host", "engine": "cpp"}) as chain:
                rep = chain.process(rec.copy())
            near = [d for d in rep["detections"] if abs(d["range_m"] - r) < 6.0 and abs(d["bearing_deg"] - b) < 2.5]
            found[name] += bool(near)
            assert len(rep["detections"]) == len(near)                 # and nothing else
    assert found["multiscale"] >= 9 and found["mean"] <= 1, found


# --- one target one detection, two targets two -----------------------------------------------------------------------

@pytest.mark.parametrize("second, detections", [
    ((1330.0, 0.0), 2),       # 30 m behind on the same bearing: beyond the tracker's range sigma (15 m, merged up to 16 m)
    ((1306.0, 0.0), 1),       # 6 m behind: one detection, 8 m long
    ((1300.0, 10.0), 2),      # the same range, 10 degrees apart: more than one and a half half-power beamwidths (7.9 degrees)
    ((1300.0, 1.5), 1),       # 1.5 degrees apart: one beam, one detection
    ((1322.0, 5.0), 2),       # both at once
])
def test_two_targets_beyond_the_stated_resolution_are_two_and_nearer_are_one(sonar, second, detections):
    """Two echoes are one detection or two. A merged pair's range lies between them by the centroid, so that is given
    (D-158: the default, the leading edge, is the nearer echo's front)."""
    rec = ping(sonar, [at(1300.0, 0.0), at(*second)], [0.0, -3.0], 81)
    got = {}
    for engine in ("cpp", "python"):
        with LiveChain({"cell_statistic": "multiscale", "backend": "host", "engine": engine, "cfar": {"range_estimate": "centroid"}}) as chain:
            got[engine] = chain.process(rec.copy())["detections"]
            assert chain.detector["merge_cells"] == 7 and chain.detector["merge_deg"] == pytest.approx(2.6, abs=0.2)
        assert len(got[engine]) == detections, (engine, got[engine])
    for a, b in zip(got["cpp"], got["python"]):
        assert a["range_m"] == pytest.approx(b["range_m"], abs=1e-3) and a["snr_db"] == pytest.approx(b["snr_db"], abs=0.02)
        assert a["extent_m"] == b["extent_m"]
    if detections == 1 and second[1] == 0.0:
        assert got["cpp"][0]["extent_m"] >= 8.0 and 1300.0 < got["cpp"][0]["range_m"] < 1306.0


def test_two_targets_thirty_metres_apart_are_two_tracks_and_a_strong_one_is_one(sonar):
    """Four pings: two targets 30 m apart in range on one bearing keep two tracks, on both engines. A target 30 dB up
    (62 dB of budget SNR: its range and bearing sidelobes far above the threshold) is one detection and one track.
    The chain's default range is the leading edge (D-158): the tracks are at 1299 and 1329 m, about 1 m before the two
    0 dB point echoes (1300 and 1330 m with the centroid, before D-158)."""
    pair = [ping(sonar, [at(1300.0, 3.0), at(1330.0, 3.0)], [0.0, 0.0], 200 + i, sequence=i) for i in range(4)]
    strong = [ping(sonar, [at(1300.0, -8.0)], 30.0, 240 + i, sequence=i) for i in range(4)]
    for engine in ("cpp", "python"):
        with LiveChain({"cell_statistic": "multiscale", "backend": "host", "engine": engine}) as chain:
            reps = chain.run([r.copy() for r in pair])
        assert [len(r["detections"]) for r in reps] == [2, 2, 2, 2], engine
        assert len(reps[-1]["tracks"]) == 2 and sorted(round(t["range_m"]) for t in reps[-1]["tracks"]) == [1299, 1329], engine
        for estimate in ("centroid", None):
            cfg = {"cell_statistic": "multiscale", "backend": "host", "engine": engine}
            if estimate is not None:
                cfg["cfar"] = {"range_estimate": estimate}
            with LiveChain(cfg) as chain:
                reps = chain.run([r.copy() for r in strong])
            assert [len(r["detections"]) for r in reps] == [1, 1, 1, 1], engine
            d = reps[0]["detections"][0]
            assert d["snr_db"] > 57.0 and abs(d["bearing_deg"] + 8.0) < 0.3 and d["extent_m"] <= 6.0, d
            if estimate == "centroid":
                assert abs(d["range_m"] - 1300.0) < 1.0, d
            else:
                # The default leading edge of an echo 62 dB over the noise: its onset crosses the single-sample threshold
                # 2.0 m before the peak (D-158; the centroid is within 0.02 m).
                assert d["range_m"] == pytest.approx(1298.0, abs=0.25), d
            assert len(reps[-1]["tracks"]) == 1


# --- the detector as it was, and the engines ---------------------------------------------------------------------------

def test_cell_statistic_mean_is_the_detector_as_it_was(sonar):
    """`cell_statistic: mean` with the 2 m cell: the chain's detections are those of the cell-averaging detector on the
    map's cell means, recomputed here with nothing of the new detector: mean of 8 samples, the one-sample factor for
    `cfar.pfa` over 150 cells either side beyond 50, touching cells grouped at their power-weighted centroid."""
    rec = ping(sonar, [at(1250.0, -12.0), at(1400.0, 9.0), at(1404.0, 9.0)], [5.0, 10.0, 6.0], 77)
    cfg = {"cell_statistic": "mean", "range_cell_m": 2.0, "cfar": {"pfa": 1e-5, "guard_m": 100.0, "train_m": 300.0}, "min_snr_db": 9.0, "backend": "host"}
    reports = {}
    for engine in ("python", "cpp"):
        with LiveChain(dict(cfg, engine=engine)) as chain:
            reports[engine] = chain.process(rec.copy())
            assert chain.detector is None
    prepared = rec.copy()
    recon.prepare([prepared], backend=core.Backend.Host)
    bearings = np.radians(np.arange(-40.0, 40.25, 0.5))
    s = prepared.samples
    r0, r1 = 0.5 * 1500.0 * s.start_time, 0.5 * 1500.0 * (s.start_time + (s.n_samples - 1) / s.sample_rate - prepared.transmit.waveform.duration)
    fine_r = r0 + 0.25 * np.arange(int((r1 - r0) / 0.25) // 8 * 8)
    fan = flank.backprojection_map(prepared, fine_r, bearings, window="hann", interp="linear", backend=core.Backend.Host)
    power = (np.abs(fan.image) ** 2).T.reshape(len(bearings), -1, 8).mean(axis=2)
    ranges = fine_r.reshape(-1, 8).mean(axis=1)
    want = [h for h in detect.cluster(detect.cfar(power, pfa=1e-5, guard=50, train=150), power, ranges, bearings, guard=50, train=150) if h.snr_db > 9.0]
    assert len(want) == 2
    for engine, tolerance in (("python", 1e-6), ("cpp", 2e-3)):
        got = reports[engine]["detections"]
        assert len(got) == len(want), engine
        for g, w in zip(got, want):
            assert "extent_m" not in g
            assert g["range_m"] == pytest.approx(w.range, abs=tolerance) and g["snr_db"] == pytest.approx(w.snr_db, abs=10 * tolerance)
            assert g["bearing_deg"] == pytest.approx(np.degrees(w.bearing), abs=tolerance)
    with pytest.raises(ValueError):
        LiveChain({"cell_statistic": "median"})
    with pytest.raises(ValueError):
        live.cpp_chain_options({"cell_statistic": "median"})


def test_both_engines_give_the_same_detections_with_stacked_rows_and_the_program_takes_the_keys():
    """Three rows added in power (the single-sample statistic then has six degrees of freedom and a 9.4 dB threshold):
    the same detections, elevations and plan from both engines; and the keys reach the C++ options by their names."""
    sonar = SonarModel.load("popoto_long_range_3row", WINDOW)
    rec = simulate_ping(sonar, make_pose(position=(0.0, 0.0, 50.0)), Scene().add_points([at(1250.0, -15.0, 50.0), at(1400.0, 10.0, 80.0)], target_strength_db=0.0),
                        noise_rms=NOISE, rng=np.random.default_rng(12), **KW)
    cfg = {"cell_statistic": "multiscale", "backend": "host", "elevation": {}}
    got = {}
    for engine in ("cpp", "python"):
        with LiveChain(dict(cfg, engine=engine)) as chain:
            got[engine] = chain.process(rec.copy())
            plan = chain.detector
        assert plan["rows"] == 3 and plan["scales"][0]["threshold_db"] == pytest.approx(9.3, abs=0.3), engine
        assert len(got[engine]["detections"]) == 2, engine
    for a, b in zip(got["cpp"]["detections"], got["python"]["detections"]):
        assert a["range_m"] == pytest.approx(b["range_m"], abs=2e-3) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=1e-3)
        assert a["snr_db"] == pytest.approx(b["snr_db"], abs=0.02) and a["extent_m"] == b["extent_m"]
        assert a["elevation_deg"] == pytest.approx(b["elevation_deg"], abs=0.05)
    o = live.cpp_chain_options({"cell_statistic": "multiscale", "cfar": {"false_alarms_per_ping": 0.01, "integrate_cells": [1, 4], "merge_m": 8.0}})
    assert o["cell_multiscale"] is True and o["cfar_false_alarms_per_ping"] == 0.01 and o["cfar_integrate_cells"] == [1, 4] and o["cfar_merge_m"] == 8.0
    assert live.cpp_chain_options({"cell_statistic": "mean"})["cell_multiscale"] is False


@pytest.mark.parametrize("range_m, ts_db", [(6000.0, 15.0), (12000.0, 36.0)])
def test_a_submerged_target_with_a_resolved_surface_echo_is_one_detection_and_one_track(range_m, ts_db):
    """A target at 150 m seen from 225 m, the surface above (image-source paths, no bottom): its echo arrives by the
    direct path and, 2 x 225 x 150 / range later in path, by the surface (5.6 m at 12 km, 11 m at 6 km; in range half of
    that for one leg by the surface, all of it for both). It is one target: one detection on every ping and one track,
    on both engines, with the arrivals listed inside the detection for whoever needs their delays. (The two paths with
    one leg by the surface add: that arrival is 6 dB above the direct one, which the levels here keep above the threshold.)"""
    from gpusonar.config import load_scenario_config
    from gpusonar.sim import run_scenario
    cfg = load_scenario_config("popoto_12km")
    cfg.pop("truth", None)
    cfg.pop("reverberation", None)
    cfg["fidelity"] = "image_source"
    cfg["propagation"] = {"surface_z": 0.0, "bottom_z": None, "max_order": 1}
    cfg.pop("environment_overrides", None)          # the rays of this case are image paths: the scenario's Bellhop override does not apply
    cfg["sonar_overrides"]["receive"]["window"] = {"range": [range_m - 500.0, range_m + 500.0]}
    cfg["platform"] = {"position": [0.0, 0.0, 225.0], "rpy": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]}
    cfg["pings"] = {"count": 4, "interval": 18.0}
    cfg["scene"] = [{"type": "points", "label": "submerged", "positions": [[300.0, range_m, 150.0]], "target_strength_db": ts_db}]
    records = run_scenario(cfg).records
    slant = float(np.hypot(np.hypot(300.0, range_m), 75.0))
    later = 2.0 * 225.0 * 150.0 / range_m                     # extra path of a leg by the surface, m
    got = {}
    for engine in ("cpp", "python"):
        with LiveChain({"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "engine": engine}) as chain:
            reps = got[engine] = chain.run([r.copy() for r in records])
            assert chain.detector["merge_cells"] == 7          # cells up to 16 m apart in range are one detection
        for rep in reps:
            assert len(rep["detections"]) == 1, (engine, rep["detections"])
            d = rep["detections"][0]
            assert d["first_range_m"] - 1.0 <= slant + 0.5 and slant + later <= d["first_range_m"] + d["extent_m"] + 1.0, (engine, d)
            # The direct echo and the echoes by the surface are inside it, apart by the surface's delay.
            assert len(d["arrivals"]) >= 2, (engine, d)
            ranges = np.array([a[0] for a in d["arrivals"]])
            assert np.abs(ranges - slant).min() < 0.5, (engine, d)
            assert min(np.abs(ranges - slant - 0.5 * later).min(), np.abs(ranges - slant - later).min()) < 0.5, (engine, d)
        assert len(reps[-1]["tracks"]) == 1 and {t["id"] for r in reps for t in r["tracks"]} == {reps[-1]["tracks"][0]["id"]}, engine
    for a, b in zip(got["cpp"], got["python"]):
        assert len(a["detections"][0]["arrivals"]) == len(b["detections"][0]["arrivals"])
        assert np.allclose(a["detections"][0]["arrivals"], b["detections"][0]["arrivals"], atol=0.05)


def test_beam_samples_are_what_the_map_is_the_power_of(sonar):
    """The complex samples behind a detection (libs/live chain.hpp `beam_samples`, `core.live_beam_samples`): over its
    range window at its bearing their strongest squared magnitude over the background is its `snr_db`, at its range
    when that is the centroid (given: the default leading edge lies before the peak, D-158)."""
    rec = ping(sonar, [at(1300.0, 6.0)], 0.0, 91)
    with LiveChain({"backend": "host", "engine": "cpp", "cfar": {"range_estimate": "centroid"}}) as chain:
        d = chain.process(rec)["detections"][0]               # the record is left prepared (baseband, compressed)
        power, ranges, _ = chain._cpp.last_map()
    grid = np.radians(np.arange(-40.0, 40.25, 0.5))
    bearing = grid[int(np.argmin(np.abs(grid - np.radians(d["bearing_deg"]))))]
    window = d["first_range_m"] - 1.0 + 0.25 * np.arange(int(d["extent_m"] / 0.25) + 8)
    z = core.live_beam_samples(rec, [float(r) for r in window], float(bearing))
    assert z.dtype == np.complex64 and len(z) == len(window)
    peak = int(np.argmax(np.abs(z)))
    assert abs(window[peak] - d["range_m"]) < 0.5
    background = np.median(np.asarray(power))                 # the map's cell means in noise: their median is near their mean
    assert 10.0 * np.log10(np.abs(z[peak]) ** 2 / background) == pytest.approx(d["snr_db"], abs=1.0)
    same = flank.backprojection_map(rec, window, np.array([bearing]), window="hann", interp="linear", backend=core.Backend.Host).image.ravel()
    assert np.allclose(z, same, rtol=1e-4, atol=1e-6 * np.abs(same).max())


# --- D-157: the leading edge as a detection's range (cfar.range_estimate: first) -----------------------------------

def _late_arrivals_ping(sonar, r0=1300.0, bearing_deg=5.0, seed=21):
    """A weak first arrival at `r0` and five arrivals 6 dB stronger 10 to 50 m behind it on the same bearing (a target
    in a channel: the direct path first, the surface and bottom paths after it), close enough to be one detection."""
    points = [at(r0 + dr, bearing_deg) for dr in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)]
    return ping(sonar, points, [0.0, 6.0, 6.0, 6.0, 6.0, 6.0], seed)


def test_the_leading_edge_is_the_first_arrival_and_the_centroid_lies_behind_it(sonar):
    """D-157: with cfar.range_estimate first the detection's range is its first sample above the single-sample threshold:
    the weak first arrival, within one range cell; centroid is the energy-weighted mean, about 25 m behind it. Both
    engines give the same; the chain's default (first since D-158) is `first` given explicitly bit for bit; the
    report's first_range_m (the nearest cell's centre) is coarser than the leading edge and within a cell of it."""
    r0 = 1300.0
    rec = _late_arrivals_ping(sonar, r0)
    got = {}
    for estimate in (None, "centroid", "first"):
        for engine in ("cpp", "python"):
            cfg = {"cell_statistic": "multiscale", "backend": "host", "engine": engine}
            if estimate is not None:
                cfg["cfar"] = {"range_estimate": estimate}
            with LiveChain(cfg) as chain:
                got[estimate, engine] = chain.process(rec.copy())
                cell = chain.detector["cell_m"]
            assert len(got[estimate, engine]["detections"]) == 1, (estimate, engine, got[estimate, engine]["detections"])
    slant = float(np.hypot(*at(r0, 5.0)[:2]))
    for engine in ("cpp", "python"):
        first = got["first", engine]["detections"][0]
        centroid = got["centroid", engine]["detections"][0]
        assert got["first", engine]["detections"] == got[None, engine]["detections"]      # D-158: the default
        assert abs(first["range_m"] - slant) < cell, (engine, first)
        assert 15.0 < centroid["range_m"] - slant < 35.0, (engine, centroid)
        assert abs(first["first_range_m"] - first["range_m"]) <= cell
        # The same group either way: only its range differs.
        assert {k: v for k, v in first.items() if k != "range_m"} == {k: v for k, v in centroid.items() if k != "range_m"}
    assert got["first", "python"]["detections"][0]["range_m"] == pytest.approx(got["first", "cpp"]["detections"][0]["range_m"], abs=1e-6)


def test_the_range_estimate_is_checked_and_reaches_the_program():
    """D-145: a word that is not centroid or first is refused by both engines, and first given outside cell_statistic
    multiscale too; the key goes through program_options to the program's options, which print the value used
    (D-158: first with multiscale when not given, centroid with mean or peak)."""
    with pytest.raises(ValueError, match="range_estimate"):
        live.cpp_chain_options({"cell_statistic": "multiscale", "cfar": {"range_estimate": "peak"}})
    with pytest.raises(ValueError, match="range_estimate"):
        live.cpp_chain_options({"cell_statistic": "mean", "cfar": {"range_estimate": "first"}})
    with pytest.raises(ValueError, match="range_estimate"):
        LiveChain({"cell_statistic": "peak", "cfar": {"range_estimate": "first"}, "backend": "host"})
    with pytest.raises(ValueError, match="range_estimate"):
        core.live_options_from_text("cfar_range_estimate=leading\n")
    with pytest.raises(ValueError, match="range_estimate"):
        core.LiveChainCpp({"cfar_range_estimate": "first", "cell_multiscale": False})
    flat = live.program_options({"cell_statistic": "multiscale", "cfar": {"range_estimate": "first"}})
    assert flat["cfar_range_estimate"] == "first"
    text = json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))
    assert text["cfar_range_estimate"] == "first"
    assert json.loads(core.live_options_from_text(""))["cfar_range_estimate"] == "first"
    assert live.program_options({"cell_statistic": "multiscale"})["cfar_range_estimate"] == "first"
    flat = live.program_options({"cell_statistic": "multiscale", "cfar": {"range_estimate": "centroid"}})
    assert flat["cfar_range_estimate"] == "centroid"
    assert json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))["cfar_range_estimate"] == "centroid"


# --- D-158: the leading edge is the live chain's default (cell_statistic multiscale) -------------------------------

@pytest.mark.parametrize("statistic, want", [("multiscale", "first"), ("mean", "centroid"), ("peak", "centroid")])
def test_the_chain_default_range_estimate_is_the_same_in_both_engines(statistic, want):
    """D-158: left out, the range estimate is first with cell_statistic multiscale and the centroid with mean or peak, the
    same in both engines: Python's rule (chain_range_estimate, the plan's options), the C++ parser's printed value
    (options_json), and the C++ chain built from the Python options. The detector's own default stays centroid."""
    merged = live._merge(live.DEFAULTS, {"cell_statistic": statistic})
    assert merged["cfar"]["range_estimate"] is None
    assert live.chain_range_estimate(merged) == want
    assert json.loads(core.live_options_from_text(f"cell_statistic={statistic}\n"))["cfar_range_estimate"] == want
    flat = live.program_options({"cell_statistic": statistic})
    assert json.loads(core.live_options_from_text("".join(f"{k}={v}\n" for k, v in flat.items())))["cfar_range_estimate"] == want
    if statistic == "multiscale":
        assert live.detector_options(merged)["range_estimate"] == "first"
        assert flat["cfar_range_estimate"] == "first"
    # The explicit centroid is accepted with every statistic, in both engines.
    for engine in ("python", "cpp"):
        LiveChain({"cell_statistic": statistic, "backend": "host", "engine": engine, "cfar": {"range_estimate": "centroid"}}).close()
        LiveChain({"cell_statistic": statistic, "backend": "host", "engine": engine}).close()
    assert json.loads(core.live_options_from_text(f"cell_statistic={statistic}\ncfar_range_estimate=centroid\n"))["cfar_range_estimate"] == "centroid"


@pytest.mark.parametrize("statistic", ["mean", "peak"])
def test_first_given_with_mean_or_peak_is_refused_by_both_engines(statistic):
    """D-158: an explicit first outside multiscale is refused in Python (check_choices) and by the C++ chain, whichever
    order the program's keys come in; left out, the same configuration is accepted (the centroid)."""
    for engine in ("python", "cpp"):
        with pytest.raises(ValueError, match="range_estimate"):
            LiveChain({"cell_statistic": statistic, "backend": "host", "engine": engine, "cfar": {"range_estimate": "first"}})
    for text in (f"cell_statistic={statistic}\ncfar_range_estimate=first\n", f"cfar_range_estimate=first\ncell_statistic={statistic}\n"):
        o = json.loads(core.live_options_from_text(text))
        assert o["cfar_range_estimate"] == "first"
    with pytest.raises(ValueError, match="range_estimate"):
        core.LiveChainCpp({"cfar_range_estimate": "first", "cell_multiscale": False, "cell_peak": statistic == "peak"})
    core.LiveChainCpp({"cell_multiscale": False, "cell_peak": statistic == "peak"})


def test_mean_without_the_key_gives_the_centroid_on_both_engines(sonar):
    """D-158: cell_statistic mean with no range_estimate runs as before on both engines: the same detections as with
    centroid given explicitly (the mean detector has no leading edge)."""
    rec = _late_arrivals_ping(sonar)
    got = {}
    for engine in ("cpp", "python"):
        for estimate in (None, "centroid"):
            cfg = {"cell_statistic": "mean", "backend": "host", "engine": engine}
            if estimate is not None:
                cfg["cfar"] = {"range_estimate": estimate}
            with LiveChain(cfg) as chain:
                got[engine, estimate] = chain.process(rec.copy())["detections"]
        assert got[engine, None] == got[engine, "centroid"] and len(got[engine, None]) >= 1, engine


# --- sonar-d99: the leading edge and the arrivals are held to the single-sample test, whatever integrate_cells holds ---

def test_the_single_sample_test_is_always_the_first_scale_and_the_leading_edge_holds_to_it(sonar, template):
    """sonar-d99: the leading edge (range_estimate first) and the arrivals take the threshold of scale 0. The plan puts
    the single-sample test there whatever `integrate_cells` holds ([2, 4] leaves out the cell mean, not the sample):
    so with [2, 4] the leading edge is still the first arrival, the arrivals all stand above the single-sample threshold,
    and both engines give the same detection."""
    for cells in ([2, 4], [4], [1, 4, 16], None):
        plan = live.detector_plan(template, {"cfar": {"integrate_cells": cells}})
        assert plan["scales"][0]["cells"] == 0 and [s["cells"] for s in plan["scales"][1:]] == (cells or [1]), cells
    r0 = 1300.0
    rec = _late_arrivals_ping(sonar, r0)
    slant = float(np.hypot(*at(r0, 5.0)[:2]))
    got = {}
    for engine in ("cpp", "python"):
        cfg = {"cell_statistic": "multiscale", "backend": "host", "engine": engine, "cfar": {"range_estimate": "first", "integrate_cells": [2, 4]}}
        with LiveChain(cfg) as chain:
            got[engine] = chain.process(rec.copy())["detections"]
            plan, cell = chain.detector, chain.detector["cell_m"]
        assert [s["cells"] for s in plan["scales"]] == [0, 2, 4]
        assert len(got[engine]) == 1, (engine, got[engine])
        d = got[engine][0]
        assert abs(d["range_m"] - slant) < cell, (engine, d)
        assert len(d["arrivals"]) >= 2 and all(a[1] > plan["scales"][0]["threshold_db"] - 1e-4 for a in d["arrivals"]), (engine, d)
    p, c = got["python"][0], got["cpp"][0]
    assert p["range_m"] == pytest.approx(c["range_m"], abs=1e-6)
    assert np.allclose(p["arrivals"], c["arrivals"], atol=1e-4)


def test_a_plan_whose_first_scale_is_not_the_single_sample_is_refused_by_both_engines(template):
    """sonar-d99: a plan built by hand whose first scale is a mean (the single sample left out) would have the flag test,
    the arrivals and the leading edge take a mean's threshold for the sample's: both detectors refuse it."""
    plan = live.detector_plan(template, {"cfar": {"integrate_cells": [2, 4]}})
    n_b, n_c, k = 3, 30, int(plan["samples_per_cell"])
    rng = np.random.default_rng(0)
    fine = rng.exponential(size=(n_b, n_c * k)).astype(np.float32)
    mean = fine.reshape(n_b, n_c, k).mean(axis=2).astype(np.float32)
    fr, cr, b = [float(v) for v in np.arange(n_c * k)], [float(v) for v in np.arange(n_c) * k + 3.5], [0.0, 0.01, 0.02]
    detect.multiscale(fine, mean, fr, cr, b, 2, 4, plan)                              # as planned: accepted
    core.live_multiscale_detect(fine, mean, fr, cr, b, 2, 4, plan, backend=core.Backend.Host)
    for bad in (dict(plan, scales=plan["scales"][1:]), dict(plan, scales=[plan["scales"][0], dict(plan["scales"][1], cells=0)])):
        with pytest.raises(ValueError, match="detector plan"):
            detect.multiscale(fine, mean, fr, cr, b, 2, 4, bad)
        with pytest.raises(ValueError, match="detector plan"):
            core.live_multiscale_detect(fine, mean, fr, cr, b, 2, 4, bad, backend=core.Backend.Host)


def test_range_estimate_centroid_given_is_the_output_from_before_d158():
    """D-158: `cfar.range_estimate: centroid` given restores the chain's output from before the default became first,
    byte for byte: the report lines of both engines on two short scenarios (a flank array with a crossing and a still
    target; a long-range array with late arrivals) against tests/python/data/centroid_d158.json, which
    data/make_centroid_d158.py wrote with the code of 42d4539 and no range_estimate given. The default (first) moves
    the ranges of the multiscale detections, so it is not that output."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "data"))
    try:
        import make_centroid_d158 as golden
    finally:
        sys.path.pop(0)
    want = json.loads(golden.GOLDEN.read_text())
    got = golden.reports({"range_estimate": "centroid"})
    assert got.keys() == want.keys()
    for name in want:
        for engine in ("cpp", "python"):
            assert got[name][engine] == want[name][engine], (name, engine)
    default = golden.reports()
    assert any(default[name][engine] != want[name][engine] for name in want for engine in ("cpp", "python"))
