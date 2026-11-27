"""The received signal at each hydrophone as sound, its parts, single arrivals and the page's clock
(python/gpusonar/viz/audio.py, arrivals.py, page3d.py; docs/tasks/VISUALIZATION.md packages V3 and V5, section 14).

What is held: the WAV's rate, duration and channel count, and its INFO chunk names the method; a heterodyne puts a known
tone where the plan says and nothing folds through 0 Hz; a time stretch divides frequencies and multiplies durations by
its factor; levels between hydrophones are preserved; the delay between the two channels of a stereo pair is
sample-exact; the opt-in per-item reverberation taps leave the record bit-identical and the parts sum to the whole
(both engines), each item starting at its boundary's two-way time; single arrivals sum to the echoes (both engines) and
peak at the trace's per-hydrophone delay after the matched filter; the 3-D page's sound and its lit hydrophones (headless
Chromium, skipped without Node and Playwright); page sizes stay under their limits.
"""

import hashlib
import math
import json
import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import pytest
from gpusonar import taps
from gpusonar.sim import engine as sim_engine
from gpusonar.sim.simulator import run_scenario
from gpusonar.viz import arrivals, audio, page2d, page3d, scene3d
from gpusonar.viz import trace as tr

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _run(name, engine="numpy", **overrides):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_scenario(name, overrides=overrides or None, engine=engine)
    except sim_engine.EngineUnavailable as exc:
        pytest.skip(str(exc))


def _peak_freq(y, rate):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1.0 / rate)[np.argmax(spec)], spec, np.fft.rfftfreq(len(y), 1.0 / rate)


# --- the plan and the operator ----------------------------------------------------------------------------------------

def test_plans_name_their_method_band_and_what_they_misrepresent():
    flank = audio.listen_plan([6000.0, 10000.0], "RealPassband", 64000.0, 8000.0, 2.0)
    assert flank["method"] == "as_recorded" and flank["rate_hz"] == 24000 and flank["tag"] == "rec"
    assert flank["kept_hz"] == [0.0, 12000.0] and flank["lost_hz"] == [[12000.0, 32000.0]]
    # A heterodyne asked for by name (since V6 a record this short is stretched by default:
    # test_the_default_method_per_family).
    fls = audio.listen_plan([3.75e6, 6.25e6], "RealPassband", 50e6, 5e6, 0.003, method="heterodyne")
    assert fls["method"] == "heterodyne" and fls["tag"] == "het2000"
    assert fls["kept_hz"][1] - fls["kept_hz"][0] == pytest.approx(3600.0)              # 2 x (2000 - 200) Hz
    assert fls["kept_audio_hz"][0] >= audio.HET_FLOOR_HZ - 1.0 / 0.003 and "Doppler" in fls["misrepresents"]
    # A 3 ms record holds 11 bins of the kept 3.6 kHz, 0.27 ms (a rail scene) under one: said, not silently.
    assert fls["warning"] is None
    short = audio.listen_plan([1.7136e6, 3.7264e6], "RealPassband", 50e6, 2.72e6, 0.00027, method="heterodyne")
    assert "click or silence" in short["warning"] and audio._method_words(short).startswith("WARNING")
    assert flank["warning"] is None
    st = audio.listen_plan([3.75e6, 6.25e6], "RealPassband", 50e6, 5e6, 0.003, method="time_stretch")
    assert st["factor"] == 391 and st["audio_duration_s"] == pytest.approx(0.003 * 391)
    assert st["kept_audio_hz"][1] <= 0.5 * st["rate_hz"] and "slower" in st["misrepresents"]
    # The Popoto 4-6 kHz baseband boards: as recorded, the carrier put back.
    popoto = audio.listen_plan([4000.0, 6000.0], "ComplexBaseband", 8000.0, 5000.0, 2.871)
    assert popoto["method"] == "as_recorded" and popoto["kept_hz"] == [1000.0, 9000.0]


# The default method per family (the owner's rules of 2026-11-27): (band Hz, sample kind, rate Hz, centre Hz, record s,
# the method auto must take). Bands and rates: the sonar configs as docs/tasks/VISUALIZATION.md section 4.1 counts them.
FAMILY_RECORDS = {
    "flank 6-10 kHz, real": ([6e3, 10e3], "RealPassband", 64e3, 8e3, 2.0, "as_recorded"),
    "Popoto 4-6 kHz, baseband": ([4e3, 6e3], "ComplexBaseband", 8e3, 5e3, 2.871, "as_recorded"),
    "Popoto 27.5 kHz board, 1 s": ([20e3, 35e3], "RealPassband", 102.4e3, 27.5e3, 1.0, "heterodyne"),
    "FLS 3.75-6.25 MHz, 3 ms": ([3.75e6, 6.25e6], "RealPassband", 50e6, 5e6, 0.003, "time_stretch"),
    "FLS 110-130 kHz, baseband 40 ms": ([110e3, 130e3], "ComplexBaseband", 40e3, 120e3, 0.04, "time_stretch"),
    "SAS rail 1.7-3.7 MHz, 0.27 ms": ([1.7136e6, 3.7264e6], "RealPassband", 50e6, 2.72e6, 0.00027, "time_stretch"),
    "SAS 60-120 kHz, baseband 0.3 s": ([60e3, 120e3], "ComplexBaseband", 100e3, 90e3, 0.3, "time_stretch"),
}


@pytest.mark.parametrize("family", list(FAMILY_RECORDS))
def test_the_default_method_per_family(family):
    band, kind, fs, fc, T, want = FAMILY_RECORDS[family]
    plan = audio.listen_plan(band, kind, fs, fc, T)
    assert plan["method"] == want and plan["chosen_by"].startswith("auto")
    words = audio._method_words(plan)
    assert "misrepresents" in words and plan["warning"] is None
    if want == "time_stretch":
        # The rule: the band's top under 16 kHz, then at least 0.5 s of sound while the band's centre stays >= 500 Hz.
        k = plan["factor"]
        assert band[1] / k <= audio.STRETCH_TOP_HZ
        assert T * k >= audio.STRETCH_MIN_SOUND_S - 1e-9 or 0.5 * (band[0] + band[1]) / (k + 1) < audio.STRETCH_MIN_CENTRE_HZ
        assert 0.5 * (band[0] + band[1]) / k >= audio.STRETCH_MIN_CENTRE_HZ or k == math.ceil(band[1] / audio.STRETCH_TOP_HZ)
        assert f"x{k:g}" in words and "factor:" in words and plan["tag"] == f"x{k:g}"
    # Asked for by name, time stretch keeps its old factor (the band's top under 16 kHz only).
    st = audio.listen_plan(band, kind, fs, fc, T, method="time_stretch")
    assert st["factor"] == max(1, math.ceil(band[1] / audio.STRETCH_TOP_HZ))


def test_the_rail_record_is_stretched_to_half_a_second():
    plan = audio.listen_plan([1.7136e6, 3.7264e6], "RealPassband", 50e6, 2.72e6, 0.00027)
    assert plan["factor"] == 1852 and plan["audio_duration_s"] == pytest.approx(0.5, abs=1e-3)   # ceil(0.5 / 0.27 ms)
    fls = audio.listen_plan([3.75e6, 6.25e6], "RealPassband", 50e6, 5e6, 0.003)
    assert fls["factor"] == 391                         # the pitch rule already gives 1.17 s


@pytest.mark.parametrize("family", list(FAMILY_RECORDS))
def test_no_default_output_is_all_zeros(family):
    """A pulse in the band, made audible by the default plan, is heard: not all zeros (the heterodyned rail records were
    silent before, docs/tasks/VISUALIZATION.md section 14.3 (1))."""
    band, kind, fs, fc, T, _ = FAMILY_RECORDS[family]
    n = int(round(fs * T))
    t = np.arange(n) / fs
    f0 = 0.5 * (band[0] + band[1])
    burst = np.exp(-0.5 * ((t - 0.5 * T) / (0.1 * T)) ** 2)
    x = burst * (np.exp(2j * np.pi * (f0 - fc) * t) if "omplex" in kind else np.cos(2 * np.pi * f0 * t))
    plan = audio.listen_plan(band, kind, fs, fc, T)
    y = audio.to_audio(x[None, :], plan)[0]
    assert np.max(np.abs(y)) > 1e-3 * np.max(np.abs(x))


@pytest.mark.parametrize("name", ["rail_points", "rail_position_errors", "rail_cylinder_glint", "rail_objects"])
def test_the_rail_scenes_default_sound_is_not_silent(name, tmp_path):
    from gpusonar.config import load_scenario_config
    from gpusonar.viz import catalogue
    plan = catalogue.view_plan(load_scenario_config(name, None, expand=False))
    rec = _run(name, "numpy", **plan["overrides"]).records[0]
    s = rec.samples
    p = audio.listen_plan(_band(rec), str(s.kind).split(".")[-1],
                          float(s.sample_rate), float(s.center_frequency), s.n_samples / float(s.sample_rate))
    assert p["method"] == "time_stretch" and p["audio_duration_s"] >= 0.45
    y = audio.to_audio(np.asarray(s.array), p)
    assert np.all(np.max(np.abs(y), axis=1) > 0.0)                       # every hydrophone heard


def _band(rec):
    w = rec.transmit.waveform
    return sorted([float(w.f_start), float(w.f_stop)])


def test_heterodyne_puts_a_tone_where_stated_and_nothing_folds_through_zero():
    fs, fc, T = 100e3, 300e3, 0.5
    n = int(fs * T)
    t = np.arange(n) / fs
    # complex baseband about 300 kHz: tones at 301 kHz (kept), 299.2 kHz (kept), 305 kHz and 297.0 kHz (outside the
    # kept 3.6 kHz about the band centre 300 kHz: dropped, never folded)
    x = sum(a * np.exp(2j * np.pi * (f - fc) * t) for a, f in ((1.0, 301e3), (0.5, 299.2e3), (1.0, 305e3), (1.0, 297.0e3)))
    plan = audio.listen_plan([280e3, 320e3], "ComplexBaseband", fs, fc, T)
    assert plan["shift_hz"] == pytest.approx(298e3)
    y = audio.to_audio(x[None, :], plan)[0]
    assert len(y) == int(round(T * plan["rate_hz"]))
    f0, spec, f = _peak_freq(y, plan["rate_hz"])
    assert f0 == pytest.approx(3000.0, abs=1.0 / T)                                     # 301 kHz - 298 kHz
    p = spec ** 2
    near = lambda g: p[np.abs(f - g) < 30.0].sum()
    assert near(1200.0) > 0.1 * near(3000.0)                                           # 299.2 kHz kept at 1.2 kHz
    # 305 kHz would be 7 kHz, 297 kHz would fold to 1 kHz through 0 Hz: neither is there, nor anything under 200 Hz.
    assert near(7000.0) < 1e-6 * p.sum() and near(1000.0) < 1e-6 * p.sum() and p[f < 190.0].sum() < 1e-6 * p.sum()
    # Real passband at 2 MHz with a 610 kHz tone in a 580-620 kHz band: at 610 - 598 = 12 kHz? No: only 3.6 kHz kept.
    fs2 = 2e6
    t2 = np.arange(int(fs2 * 0.05)) / fs2
    plan2 = audio.listen_plan([580e3, 620e3], "RealPassband", fs2, 0.0, 0.05, method="heterodyne")
    y2 = audio.to_audio(np.cos(2 * np.pi * 601e3 * t2), plan2)
    assert _peak_freq(y2, plan2["rate_hz"])[0] == pytest.approx(601e3 - plan2["shift_hz"], abs=20.0)
    y3 = audio.to_audio(np.cos(2 * np.pi * 610e3 * t2), plan2)
    assert np.max(np.abs(y3)) < 1e-9                                                     # outside the kept band


def test_time_stretch_divides_frequency_and_multiplies_duration_by_its_factor():
    fs, T = 1e6, 0.02
    t = np.arange(int(fs * T)) / fs
    x = np.cos(2 * np.pi * 150e3 * t) * (t > 0.01)
    plan = audio.listen_plan([100e3, 200e3], "RealPassband", fs, 0.0, T, method="time_stretch", factor=20)
    y = audio.to_audio(x, plan)
    assert plan["factor"] == 20 and len(y) == int(round(T * 20 * plan["rate_hz"]))
    assert _peak_freq(y, plan["rate_hz"])[0] == pytest.approx(150e3 / 20, abs=1.0 / (20 * T))
    # the onset at 10 ms is heard at 200 ms
    env = np.abs(y)
    onset = np.argmax(env > 0.5 * env.max()) / plan["rate_hz"]
    assert onset == pytest.approx(0.01 * 20, abs=0.002)


def test_wav_rate_duration_channels_and_method(tmp_path):
    plan = audio.listen_plan([6000.0, 10000.0], "RealPassband", 64000.0, 8000.0, 1.0)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 64000))
    y = audio.to_audio(x, plan)
    g = audio.gain_for([y])
    info = {"ICMT": f"{plan['method']}: {audio._method_words(plan)}"}
    w1 = audio.write_pcm16(str(tmp_path / "m.wav"), plan["rate_hz"], y[0], g, info)
    w2 = audio.write_pcm16(str(tmp_path / "s.wav"), plan["rate_hz"], y, g, info)
    m, s = audio.read_pcm16(str(tmp_path / "m.wav")), audio.read_pcm16(str(tmp_path / "s.wav"))
    assert (m["rate"], m["channels"], m["bits"], m["format"]) == (24000, 1, 16, 1)
    assert (s["rate"], s["channels"]) == (24000, 2) and w2["bytes"] > w1["bytes"]
    assert s["samples"].shape[1] / s["rate"] == pytest.approx(1.0) and m["samples"].shape[1] == 24000
    assert m["info"]["ICMT"].startswith("as_recorded: as recorded") and "misrepresents" in m["info"]["ICMT"]
    assert np.max(np.abs(s["samples"])) == pytest.approx(10 ** (-1 / 20), abs=1e-4)      # the loudest at -1 dBFS


def test_relative_levels_between_hydrophones_are_preserved():
    plan = audio.listen_plan([6000.0, 10000.0], "RealPassband", 64000.0, 8000.0, 0.5)
    t = np.arange(32000) / 64000.0
    x = np.stack([a * np.sin(2 * np.pi * 8000.0 * t) for a in (1.0, 0.5, 0.1)])
    y = audio.to_audio(x, plan) * audio.gain_for([audio.to_audio(x, plan)])
    rms = np.sqrt(np.mean(y ** 2, axis=1))
    np.testing.assert_allclose(rms / rms[0], [1.0, 0.5, 0.1], rtol=1e-9)


def test_inter_channel_delay_is_sample_exact(tmp_path):
    # As recorded, 64 kHz to 24 kHz: 8 input samples are 3 output samples; the right channel is the left delayed.
    plan = audio.listen_plan([6000.0, 10000.0], "RealPassband", 64000.0, 8000.0, 1.0)
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(64000)
    x = np.stack([x0, np.roll(x0, 8)])
    y = audio.to_audio(x, plan)
    np.testing.assert_allclose(y[1], np.roll(y[0], 3), atol=1e-9 * np.max(np.abs(y)))
    g = audio.gain_for([y])
    audio.write_pcm16(str(tmp_path / "pair.wav"), plan["rate_hz"], y, g)
    back = audio.read_pcm16(str(tmp_path / "pair.wav"))["samples"]
    assert np.max(np.abs(back[1] - np.roll(back[0], 3))) <= 1.0 / 32767 + 1e-12         # one LSB of rounding
    # Time stretch x4 from 1 MHz: 125 input samples (125 us) are 500 us after the stretch, 24 samples at 48 kHz.
    plan2 = audio.listen_plan([100e3, 200e3], "RealPassband", 1e6, 0.0, 0.01, method="time_stretch", factor=4)
    x1 = rng.standard_normal(10000)
    y2 = audio.to_audio(np.stack([x1, np.roll(x1, 125)]), plan2)
    assert plan2["rate_hz"] == 48000
    np.testing.assert_allclose(y2[1], np.roll(y2[0], 24), atol=1e-9 * np.max(np.abs(y2)))
    # Heterodyne: the envelope's delay is kept (the carrier's phase is the shifted frequency's, see audio.py).
    plan3 = audio.listen_plan([280e3, 320e3], "ComplexBaseband", 100e3, 300e3, 0.2, method="heterodyne")
    z = np.zeros(20000, complex)
    z[5000:5200] = np.hanning(200)
    y3 = audio.to_audio(np.stack([z, np.roll(z, 250)]), plan3)        # 2.5 ms
    e = np.abs(np.fft.ifft(np.fft.fft(y3, axis=1) * (np.fft.fftfreq(y3.shape[1]) > 0), axis=1))
    d = (np.argmax(e[1]) - np.argmax(e[0])) / plan3["rate_hz"]
    assert d == pytest.approx(0.0025, abs=1.5 / plan3["rate_hz"])


# --- the parts: per-item reverberation taps, both engines -------------------------------------------------------------

ITEMS = {"reverberation": [{"model": "fluid_sediment"}, {"model": "chapman_harris", "wind_speed": 8.0},
                           {"model": "volume", "layer": [8.0, 12.0], "sv_db": -50.0}],
         "pings": {"count": 1}, "platform": {"position": [0, 0, 3.0]},
         "sonar_overrides": {"receive": {"window": {"range": [1.0, 60.0]}}}}


@pytest.mark.parametrize("engine", ["numpy", "cpp"])
def test_per_item_taps_leave_the_record_unchanged_and_the_parts_sum_to_the_whole(engine, tmp_path):
    plain = _run("sst_sediment_reverb", engine, **ITEMS).records[0]
    with taps.sim_capture(str(tmp_path / "t"), per_item=True):
        result = _run("sst_sediment_reverb", engine, **ITEMS)
    rec = result.records[0]
    h = lambda r: hashlib.sha256(np.ascontiguousarray(r.samples.array).tobytes()).hexdigest()
    assert h(plain) == h(rec)                                                   # the taps only read the record
    stages = taps.read_sim_taps(str(tmp_path / "t"), 0)
    meta = json.load(open(tmp_path / "t" / "sim_taps.json"))
    parts = audio.decompose(stages, meta, rec)
    names = [k for k in parts if k.startswith("reverberation_")]
    assert names == ["reverberation_0_bottom", "reverberation_1_surface", "reverberation_2_volume"]
    total = sum(v for v, _ in parts.values())
    whole = np.asarray(stages["noise"])
    assert np.max(np.abs(total - whole)) <= 1e-6 * np.max(np.abs(whole))
    items = sum(parts[k][0] for k in names)
    np.testing.assert_allclose(items, np.asarray(stages["reverberation"]) - np.asarray(stages["echoes"]), rtol=0,
                               atol=1e-6 * np.max(np.abs(whole)))
    # Each item starts at its boundary's two-way time (sonar at 3 m, surface at 0, the volume sheet at its layer's
    # centre 10 m, the bed at 20 m; a 2 ms pulse; the onset at 1 % of the item's peak power).
    s = rec.samples
    c = result.config["environment"]["sound_speed"]
    expect = {"reverberation_1_surface": 2 * 3.0 / c, "reverberation_2_volume": 2 * 7.0 / c, "reverberation_0_bottom": 2 * 17.0 / c}
    for k, t_exp in expect.items():
        pw = np.mean(np.abs(parts[k][0]) ** 2, axis=0)
        onset = s.start_time + np.argmax(pw > 1e-2 * pw.max()) / s.sample_rate
        assert t_exp - 1.5e-3 <= onset <= t_exp + 0.5e-3, (k, onset, t_exp)


def test_per_item_taps_are_off_by_default(tmp_path):
    with taps.sim_capture(str(tmp_path / "t")):
        _run("sst_sediment_reverb", "numpy", **ITEMS)
    assert not any(k.startswith("reverberation_item") for k in taps.read_sim_taps(str(tmp_path / "t"), 0))


# --- single arrivals --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["numpy", "cpp"])
def test_single_arrivals_sum_to_the_echoes_and_peak_at_the_traces_delay(engine, tmp_path):
    with taps.sim_capture(str(tmp_path / "t")), arrivals.capture_ping() as box:
        result = _run("sst_echo_level", engine)
    echoes = np.asarray(taps.read_sim_taps(str(tmp_path / "t"), 0)["echoes"])
    trace = tr.build_trace(result, engine=engine)
    pairs = arrivals.strongest_pairs(trace, box, 0, 100)
    assert len(pairs) == len(trace["pairs"]) == 9 and all(p["mapped"] for p in pairs)
    one = [arrivals.single_pair_echoes(box, p["scatterer"], p["k_captured"], p["j_captured"]) for p in pairs]
    total = sum(e.astype(complex) for e in one)
    # One target, one scatterer: the pairs of the target are all of the scene's echoes (linearity of synthesis).
    assert np.max(np.abs(total - echoes)) <= 1e-5 * np.max(np.abs(echoes))
    # Each arrival alone, matched-filtered, peaks at the trace's delay at every hydrophone (1.5 samples).
    rec = result.records[0]
    dt = 1.0 / rec.samples.sample_rate
    for p, e in zip(pairs[:4], one[:4]):
        t, env = page2d.compressed(page2d._with_samples(rec, e))
        for ch in range(env.shape[0]):
            assert abs(t[np.argmax(env[ch])] - p["channel_delay_s"][ch]) <= 1.5 * dt, (p["out"], p["back"], ch)
    # The capture changed nothing: the record is the one a plain run gives.
    plain = _run("sst_echo_level", engine).records[0]
    assert np.array_equal(np.asarray(plain.samples.array), np.asarray(rec.samples.array))


# --- one ping's files, and the pages ----------------------------------------------------------------------------------

def _ping_audio(tmp_path, name="sst_echo_level", **kw):
    with taps.sim_capture(str(tmp_path / "t"), per_item=True), arrivals.capture_ping() as box:
        result = _run(name, "numpy", **kw)
    stages = taps.read_sim_taps(str(tmp_path / "t"), 0)
    meta = json.load(open(tmp_path / "t" / "sim_taps.json"))
    rec = result.records[0]
    trace = tr.build_trace(result, engine="numpy")
    parts = audio.decompose(stages, meta, rec)
    arr = []
    for p in arrivals.strongest_pairs(trace, box, 0, 8):
        e = arrivals.single_pair_echoes(box, p["scatterer"], p["k_captured"], p["j_captured"])
        arr.append({"samples": e, "what": "one arrival", "out": p["out"], "back": p["back"], "delay_s": p["delay_s"],
                    "level_db": p["level_db"], "n_surface": p["n_surface"], "n_bottom": p["n_bottom"]})
    m = audio.write_ping_audio(str(tmp_path), name, rec, trace, parts, arr)
    return m, trace, rec, stages


def test_ping_audio_files_say_their_method_and_keep_levels(tmp_path):
    m, trace, rec, _ = _ping_audio(tmp_path)
    n_ch = rec.samples.n_channels
    assert m["plan"]["method"] == "as_recorded" and m["method_words"].startswith("as recorded")
    whole = [f for f in m["files"] if f["kind"] == "whole"]
    assert len(whole) == n_ch and len(m["arrivals"]) == 8 and m["parts_channels"] == [0, n_ch - 1]
    adir = tmp_path / "sst_echo_level.audio"
    waves = [audio.read_pcm16(str(adir / f["file"])) for f in whole]
    for w in waves:
        assert w["rate"] == m["plan"]["rate_hz"] and w["channels"] == 1
        assert w["samples"].shape[1] / w["rate"] == pytest.approx(rec.samples.n_samples / rec.samples.sample_rate, abs=1e-3)
        assert "as_recorded" in w["info"]["ICMT"] and "levels preserved" in w["info"]["ICMT"]
    # Levels between hydrophones: the WAVs' rms ratios are the record's (one gain for the kind).
    x = np.asarray(rec.samples.array)
    r_rec = np.sqrt(np.mean(np.abs(x) ** 2, axis=1))
    r_wav = np.array([np.sqrt(np.mean(w["samples"] ** 2)) for w in waves])
    np.testing.assert_allclose(r_wav / r_wav[0], r_rec / r_rec[0], rtol=2e-3)
    assert max(np.max(np.abs(w["samples"])) for w in waves) == pytest.approx(10 ** (-1 / 20), abs=2e-4)
    # The stereo pair: the two hydrophones' own files, one gain, sample for sample (one LSB).
    st = audio.read_pcm16(str(adir / m["stereo"]["file"]))
    a, b = m["stereo"]["channels"]
    assert st["channels"] == 2 and np.max(np.abs(st["samples"][0] - waves[a]["samples"][0])) <= 1.0 / 32767 + 1e-12
    assert np.max(np.abs(st["samples"][1] - waves[b]["samples"][0])) <= 1.0 / 32767 + 1e-12
    assert os.path.exists(adir / "manifest.js") and os.path.exists(adir / "h000.js")
    # A single arrival's file, matched-filtered in its own band, peaks at the trace's delay at that hydrophone.
    arr0 = m["arrivals"][0]
    f = next(x for x in m["files"] if x["kind"] == arr0["kind"] and x["channel"] == 0)
    y = audio.read_pcm16(str(adir / f["file"]))["samples"][0]
    pair = max(trace["pairs"], key=lambda p: p["level_db"])
    env = np.abs(np.fft.ifft(np.fft.fft(y) * (np.fft.fftfreq(len(y)) > 0)))
    t_peak = m["record_start_s"] + np.argmax(env) / m["plan"]["rate_hz"]
    assert abs(t_peak - pair["channel_delay_s"][0]) < 0.03                              # within the 50 ms pulse


def test_pages_with_sound_stay_under_their_size_limits(tmp_path):
    m, trace, rec, stages = _ping_audio(tmp_path)
    scene3d.enrich(trace)
    side = page3d.write_page(trace, str(tmp_path / "s.html"), three="cdn", audio=m)
    assert side["bytes"] < page3d.PAGE_LIMIT_CDN and side["audio"] == "side"
    assert '<script src="sst_echo_level.audio/manifest.js"></script>' in open(tmp_path / "s.html").read()
    emb = page3d.write_page(trace, str(tmp_path / "e.html"), three="cdn", audio=m, audio_mode="embed",
                            audio_dir=str(tmp_path / "sst_echo_level.audio"))
    assert emb["audio"] == "embed" and emb["audio_bytes"] <= page3d.AUDIO_EMBED_CAP
    assert emb["bytes"] < page3d.PAGE_LIMIT_AUDIO_EMBED
    # The 2-D page with the spectrogram and the reverberation by item stays under 300 kB.
    stages["__meta__"] = json.load(open(tmp_path / "t" / "sim_taps.json"))
    drawn = page2d.draw_page(trace, rec, str(tmp_path / "p.png"), stages=stages)
    assert os.path.getsize(tmp_path / "p.png") < 300_000 and drawn["spectrogram_channel"] == 0


# --- the page in a browser (opt-in: Node and Playwright with Chromium) ------------------------------------------------

def _browser():
    node = shutil.which("node")
    if node is None:
        return None, "Node is not installed"
    probe = subprocess.run([node, "-e", "require('playwright').chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader',"
                            "'--enable-unsafe-swiftshader']}).then(b=>b.close()).then(()=>process.exit(0),()=>process.exit(3))"],
                           capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        return None, "Playwright with Chromium is not available to Node"
    return node, None


PROBE = r"""
const { chromium } = require('playwright'); const path = require('path');
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
  const p = await (await b.newContext({ viewport: { width: 1100, height: 760 } })).newPage();
  const errors = []; p.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  p.on('pageerror', e => errors.push(String(e))); p.on('requestfailed', r => errors.push('failed ' + r.url()));
  await p.goto('file://' + path.resolve(process.argv[2])); await p.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
  const times = JSON.parse(process.argv[3]);
  const lit = []; for (const t of times) lit.push(await p.evaluate(t => window.__viz.setTime(t), t));
  const sound = await p.evaluate(() => [window.__viz.sound(), window.__viz.setSound(1, 'whole'), window.__viz.win, window.__page]);
  console.log(JSON.stringify({ lit, sound, errors })); await b.close();
})().catch(e => { console.error(e); process.exit(2); });
"""


def test_page_lights_the_hydrophones_at_the_traces_delays_and_plays_their_sound(tmp_path):
    node, why = _browser()
    if node is None:
        pytest.skip(why)
    if not os.path.exists(page3d.VENDORED):
        pytest.skip("third_party/three/three.module.min.js is not present (pages would need the network)")
    m, trace, rec, _ = _ping_audio(tmp_path)
    scene3d.enrich(trace)
    page3d.write_page(trace, str(tmp_path / "sst_echo_level.html"), three="embed", audio=m)
    data = page3d.page_data(trace)
    t_max = max([p["t"] for p in data["pairs"]] + [r["delay"] for leg in data["legs"] for w in ("out", "back") for r in leg[w]]
                + [m["record_start_s"] + m["record_duration_s"]]) * 1.02
    win = 0.004 * t_max
    # From the trace: at a pair's delay at hydrophone c, c is lit, and the lit set is every hydrophone with an arrival
    # within the window; long before any arrival nothing is.
    pick = sorted(trace["pairs"], key=lambda p: -p["level_db"])
    times = [pick[0]["channel_delay_s"][0], pick[-1]["channel_delay_s"][3], 0.2]
    expected = [sorted(c for c in range(len(trace["sonar"]["hydrophones"]))
                       if any(abs(p["channel_delay_s"][c] - t) < win for p in trace["pairs"])) for t in times]
    path = tmp_path / "probe.js"
    path.write_text(PROBE)
    out = subprocess.run([node, str(path), str(tmp_path / "sst_echo_level.html"), json.dumps(times)], capture_output=True,
                         text=True, timeout=300, cwd=str(tmp_path))
    assert out.returncode == 0, out.stdout + out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["errors"] == []
    assert res["lit"] == expected and 0 in res["lit"][0] and 3 in res["lit"][1] and res["lit"][2] == []
    first, switched, page_win, info = res["sound"]
    assert first["kind"] == "whole" and first["src"] == "sst_echo_level.audio/h000.whole.rec.wav"
    assert switched == {"hyd": 1, "kind": "whole"} and page_win == pytest.approx(win, rel=1e-9)
    assert info["audio"] and info["audioFiles"] == len(m["files"])


# --- V6 part A: single arrivals by the synthesis's amplitude; the parts and the front end --------------------------

def _scenario_view():
    import importlib.util
    spec = importlib.util.spec_from_file_location("scenario_view", os.path.join(ROOT, "tools", "scenario_view.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_single_arrivals_ranked_by_the_synthesis_are_never_silent(tmp_path):
    """seafloor_objects_fls: the trace's strongest target (the box, its strongest patch turned away from the sonar) has
    one pair, which the synthesis weights to zero by Lambert's law with the patch's normal: ranked by the trace, its one
    single arrival is silent (docs/tasks/VISUALIZATION.md section 14.3 (2)). Ranked by the synthesis's own amplitude
    over every target, each listed arrival synthesised alone is heard, and the levels come in descending order."""
    from gpusonar.config import load_scenario_config
    from gpusonar.viz import catalogue
    plan = catalogue.view_plan(load_scenario_config("seafloor_objects_fls", None, expand=False))
    with taps.sim_capture(str(tmp_path / "t")), arrivals.capture_ping() as box:
        result = _run("seafloor_objects_fls", "numpy", **plan["overrides"])
    assert box.get("kernel") is not None and len(box["kernel"]["q"]) > 0
    trace = tr.build_trace(result, engine="numpy")
    old = dict(box)
    old.pop("kernel")
    by_trace = arrivals.strongest_pairs(trace, old, 0, 8)
    silent = [p for p in by_trace if p["mapped"] and not np.any(
        arrivals.single_pair_echoes(old, p["scatterer"], p["k_captured"], p["j_captured"]))]
    assert silent                                                       # the old ranking gives a silent arrival here
    pairs = arrivals.strongest_pairs(trace, box, None, 8)
    assert 0 < len(pairs) <= 8 and all(p["ranked_by"] == "synthesis" for p in pairs)
    levels = [p["synthesis_level_db"] for p in pairs]
    assert levels == sorted(levels, reverse=True)
    for p in pairs:
        e = arrivals.single_pair_echoes(box, p["scatterer"], p["k_captured"], p["j_captured"])
        assert np.max(np.abs(e)) > 0.0, p
    # Every pair the kernel holds a path for is a candidate: fewer than 8 here means fewer exist, and the page says so.
    n_all = sum(1 for t in range(len(trace["targets"])) for _ in arrivals.strongest_pairs(trace, box, t, 10 ** 6))
    assert len(pairs) == min(8, n_all)
    sv = _scenario_view()
    out = tmp_path / "page"
    out.mkdir()
    facts = sv.view("seafloor_objects_fls", str(out), None, False, None, "numpy", audio_opts={"mode": "side"})
    m = json.load(open(out / "seafloor_objects_fls.audio" / "manifest.json"))
    assert len(m["arrivals"]) == len(pairs) and facts["audio"]["arrivals"] == len(pairs)
    assert m["arrivals_ranked_by"].startswith("the synthesis")
    if len(pairs) < 8:
        assert m["warning_words"].startswith(f"single arrivals: {len(pairs)} listed, not 8")
    adir = out / "seafloor_objects_fls.audio"
    for a in m["arrivals"]:
        for f in (x for x in m["files"] if x["kind"] == a["kind"]):
            assert np.max(np.abs(audio.read_pcm16(str(adir / f["file"]))["samples"])) > 0.0, f["file"]


def test_the_overload_scenes_page_says_the_parts_do_not_sum(tmp_path):
    """popoto_testauv_direct_overload: the parts are taken before the front end, which clips; the page says so, with
    the residual, instead of showing a silent mismatch. A scene without an overload front end (sst_echo_level) says its
    parts are before the front end and sum to the record."""
    sv = _scenario_view()
    facts = sv.view("popoto_testauv_direct_overload", str(tmp_path), None, True, "cdn", "numpy", audio_opts={"mode": "side"})
    m = json.load(open(tmp_path / "popoto_testauv_direct_overload.audio" / "manifest.json"))
    fe = m["front_end"]
    assert fe["overload"] and fe["mismatch"] and fe["residual_db"] > audio.PARTS_MISMATCH_DB
    assert "before the front end" in fe["statement"] and "do NOT sum" in fe["statement"] and "clipped" in fe["statement"]
    js = open(tmp_path / "popoto_testauv_direct_overload.audio" / "manifest.js").read()
    assert "do NOT sum" in js and facts["html"]["audio"] == "side"
    page = open(tmp_path / "popoto_testauv_direct_overload.html").read()
    assert "A.front_end.mismatch" in page and "parts ≠ record" in page          # the page's warning line reads it
    sv.view("sst_echo_level", str(tmp_path), None, False, None, "numpy", audio_opts={"mode": "side"})
    fe2 = json.load(open(tmp_path / "sst_echo_level.audio" / "manifest.json"))["front_end"]
    assert not fe2["overload"] and not fe2["mismatch"] and "before the front end" in fe2["statement"]
