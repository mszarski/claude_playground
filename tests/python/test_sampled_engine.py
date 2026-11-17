"""Echo synthesis of sampled transmit pulses in the C++ kernel (libs/sim synth_kernel.hpp,
`pair_sampled`) against the NumPy reference engine (python/gpusonar/sim/waveform.py
`synthesize_into`), on the cases of test_sampled_waveform.py; and the pass count of the
kernel's arrival-time solve.

The two engines do the same arithmetic in double precision: the same emission-time
quadratic, the same fine table read with the same cubic, the same gain at the same knot
frequencies. They differ in rounding only:

- the kernel adds each path's contribution to the float32 record as it goes (one rounding
  of 6e-8 relative per path and sample), NumPy sums in double and rounds once;
- the kernel advances the carrier phasor by a recurrence re-seeded every 16 samples (exact
  for a phase quadratic in the sample index, which the carrier's is), NumPy takes the
  exponential at every sample;
- element velocities: the kernel takes them from the pose, NumPy by a central difference.

So the tolerance is a few float32 roundings: relative rms below 2e-6 (measured 3e-8 to 4e-7)
and no sample further off than 5e-6 of the record's peak.

Measured numbers are printed (run with -s).
"""

import json
import warnings

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.sim import Scene, make_pose, run_scenario, simulate_ping
from gpusonar.sim import waveform as wfm
from gpusonar.sim.bellhop import find_bellhop
from gpusonar.sonar import SonarModel

from test_sampled_waveform import C, IMAGE, TIERS, _five_targets, _scenario, rel_rms, sonar_with

core = gs.core
RMS_TOL = 2e-6
PEAK_TOL = 5e-6
needs_cuda = pytest.mark.skipif(not core.cuda_available(), reason="no CUDA device")


def assert_same(a, b, label):
    """Two records that should differ by float32 rounding only."""
    a, b = np.asarray(a), np.asarray(b)
    peak = np.abs(b).max()
    assert peak > 0
    err, worst = rel_rms(b, a), float(np.abs(a - b).max() / peak)
    print(f"{label}: relative rms {err:.2e}, worst sample {worst:.2e} of the peak")
    assert err < RMS_TOL and worst < PEAK_TOL
    return err


def both(sonar, pose, scene, **kw):
    kw = {"environment": {"sound_speed": C}, "adc": False, **kw}
    return [simulate_ping(sonar, pose, scene, engine=e, **kw) for e in ("cpp", "numpy")]


# --- C++ against NumPy on the cases of test_sampled_waveform.py --------------------------------

@pytest.mark.parametrize("kind", ["baseband", "real"])
@pytest.mark.parametrize("oversample", [2, 8])
def test_engines_agree_on_a_sampled_lfm_with_a_moving_sonar(kind, oversample):
    wave = {"type": "sampled", "source": "lfm", "window_alpha": 0.2, "oversample": oversample}
    sonar = sonar_with("sas_auv", wave, 17.0, 23.0, kind, omni=True)
    a, b = both(sonar, make_pose(velocity=(1.5, 0.0, 0.0)), _five_targets(sonar, 20.0), absorption_model="none")
    assert_same(a.samples.array, b.samples.array, f"sas_auv omni, oversample {oversample} ({kind})")


@pytest.mark.parametrize("kind", ["baseband", "real"])
@pytest.mark.parametrize("name, r0", [("fls_300k", 20.0), ("flank_32", 300.0)])
@pytest.mark.parametrize("knots", [8, 0, 3])
def test_engines_agree_with_beam_patterns_and_absorption(name, r0, kind, knots):
    """Loss that varies across the band: both engines weight the same `gain_knots + 1` parts
    of the envelope by the gain at the same frequencies (the kernel per sample, NumPy on the
    whole table), so nothing but rounding separates them. `gain_knots: 0` is one gain at the
    band centre."""
    dur = min(SonarModel.load(name).waveform_config["duration"], 0.02)
    wave = {"type": "sampled", "source": "lfm", "duration": dur, "window_alpha": 0.2, "gain_knots": knots}
    sonar = sonar_with(name, wave, r0 - 3.0, r0 + 3.0, kind)
    a, b = both(sonar, make_pose(velocity=(1.0, 0.5, 0.0)), _five_targets(sonar, r0),
                absorption_model="francois_garrison", acceleration=(0.3, -0.2, 0.1))
    assert_same(a.samples.array, b.samples.array, f"{name} beams and absorption, {knots} knots ({kind})")


@pytest.mark.parametrize("label", list(TIERS))
@pytest.mark.parametrize("kind", ["baseband", "real"])
def test_engines_agree_in_every_propagation_tier(label, kind):
    """Every tier, real and baseband, moving sonar and target. The whole record is compared:
    path pairing and reverberation also switch engine with `engine`, and are tested to agree
    on their own (test_sim_cpp.py)."""
    tier = label.split(" ")[0]
    if tier == "eigenray" and find_bellhop() is None:
        pytest.skip("Bellhop not installed (GPUSONAR_BELLHOP)")
    cfg = _scenario(tier, {"type": "sampled", "source": "lfm"}, kind=kind, **dict(TIERS[label]))
    a = run_scenario(dict(cfg), engine="cpp").records[0]
    b = run_scenario(dict(cfg), engine="numpy").records[0]
    assert a.transmit.waveform.type == core.WaveformType.Arbitrary
    assert_same(a.samples.array, b.samples.array, f"{label} ({kind})")


@pytest.mark.parametrize("kind", ["baseband", "real"])
@pytest.mark.parametrize("source, extra", [("barker", {"code": 13}), ("mseq", {"degree": 7}),
                                           ("hfm", {"duration": 0.004}), ("stepped_fm", {"duration": 0.004, "steps": 8})])
def test_engines_agree_on_phase_codes_and_other_sources(kind, source, extra):
    wave = {"type": "sampled", "source": source, **extra}
    if source in ("barker", "mseq"):
        n_chips = 13 if source == "barker" else 127
        wave["duration"] = n_chips / 7500.0
    sonar = sonar_with("popoto_8", wave, 280.0, 330.0, kind)
    a, b = both(sonar, make_pose(velocity=(0.0, 1.0, 0.0)), _five_targets(sonar, 300.0),
                absorption_model="francois_garrison")
    assert_same(a.samples.array, b.samples.array, f"popoto_8 {source} ({kind})")


@pytest.mark.parametrize("kind", ["baseband", "real"])
@pytest.mark.parametrize("mover", ["target", "sonar"])
def test_engines_agree_on_a_fast_closing_target(kind, mover):
    """Doppler as time scaling, 12 m/s closing at 400 m: the kernel runs the arrival-time
    solve for sampled pulses to the same 8 passes as NumPy. With the analytic pulses' 3 and
    4 passes the two would differ by 7e-3 (see the pass-count test below)."""
    r0, v = 400.0, 12.0
    wave = {"type": "sampled", "source": "mseq", "degree": 7, "duration": 0.05, "f_start": 6e3, "f_stop": 10e3}
    sonar = sonar_with("flank_32", wave, r0 - 40.0, r0 + 60.0, kind, omni=True)
    target = sonar.boresight * r0
    if mover == "target":
        scene, pose = Scene().add_points([target], velocity=-v * sonar.boresight), make_pose()
    else:
        scene, pose = Scene().add_points([target]), make_pose(velocity=v * sonar.boresight)
    a, b = both(sonar, pose, scene, absorption_model="none")
    assert_same(a.samples.array, b.samples.array, f"{mover} closing at 12 m/s ({kind})")
    few = simulate_ping(sonar, pose, scene, environment={"sound_speed": C}, adc=False, absorption_model="none")
    np.testing.assert_array_equal(few.samples.array, a.samples.array)          # C++ is the default engine


def test_board_clock_errors_reach_both_engines_alike():
    """Each channel's sample clock (offset and drift), with the mixer on the nominal clock:
    3 us and 40 ppm on the second capture board."""
    wave = {"type": "sampled", "source": "lfm", "duration": 0.01, "window_alpha": 0.2}
    boards = [{"first": 0, "count": 3}, {"first": 3, "count": 3, "clock_offset": 3e-6, "clock_drift_ppm": 40.0}]
    recs = {}
    for label, cfg in (("clean", None), ("clocks", boards)):
        extra = {"array": {"count": 6}} if cfg is None else {"array": {"count": 6}, "receive": {"boards": cfg}}
        sonar = sonar_with("flank_32", wave, 290.0, 320.0, **extra)
        truth = {}
        recs[label] = [simulate_ping(sonar, make_pose(velocity=(2.0, 0.0, 0.0)), _five_targets(sonar, 300.0),
                                     environment={"sound_speed": C}, adc=False, engine=engine,
                                     clock_truth=truth).samples.array for engine in ("cpp", "numpy")]
    assert truth["offset"] == [0.0, 3e-6] and truth["drift_ppm"] == [0.0, 40.0]
    assert_same(*recs["clocks"], "board clocks, 3 us and 40 ppm")
    np.testing.assert_array_equal(recs["clocks"][0][:3], recs["clean"][0][:3])
    assert rel_rms(recs["clean"][0][3:], recs["clocks"][0][3:]) > 0.05      # 3 us is 0.15 rad at 8 kHz


# --- which engine ran is visible -------------------------------------------------------------------

def test_cpp_is_the_default_for_sampled_pulses_and_leaves_no_note(monkeypatch):
    """C++ synthesises a sampled pulse by default, silently and without a note in the record.
    The NumPy reference runs when it is selected, and says so in the record's extra_json."""
    wave = {"type": "sampled", "source": "hfm", "duration": 0.02, "window_alpha": 0.2}
    cfg = _scenario("image_source", wave, propagation=IMAGE, reverberation=[{"boundary_z": 100.0, "mu_db": -27.0}])
    monkeypatch.setattr(wfm, "_warned", False)
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = run_scenario(dict(cfg), engine="cpp").records[0]
        c = run_scenario(dict(cfg)).records[0]
        b = run_scenario(dict(cfg), engine="numpy").records[0]
        monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpy")
        d = run_scenario(dict(cfg)).records[0]
    for r in (a, c):
        assert "echo_synthesis" not in json.loads(r.metadata.extra_json or "{}").get("gpusonar", {})
    for r in (b, d):
        assert json.loads(r.metadata.extra_json)["gpusonar"]["echo_synthesis"] == "numpy_sampled"
    np.testing.assert_array_equal(a.samples.array, c.samples.array)
    np.testing.assert_array_equal(b.samples.array, d.samples.array)
    assert_same(a.samples.array, b.samples.array, "image source with reverberation")


def test_a_waveform_without_a_fine_table_runs_in_numpy_and_says_so(monkeypatch):
    """`table_factor: 0` asks for the windowed sinc at every output sample, which the kernel
    does not do (it reads the fine table): NumPy runs, with a warning and the note."""
    wave = {"type": "sampled", "source": "lfm", "duration": 0.002, "window_alpha": 0.2, "table_factor": 0}
    sonar = sonar_with("sas_auv", wave, 18.0, 22.0, omni=True)
    monkeypatch.setattr(wfm, "_warned", False)
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)
    with pytest.warns(RuntimeWarning, match="NumPy engine"):
        rec = simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False)
    assert json.loads(rec.metadata.extra_json)["gpusonar"]["echo_synthesis"] == "numpy_sampled"
    assert not wfm.kernel_supports(sonar.sampled_waveform)
    many = sonar_with("sas_auv", dict(wave, table_factor=8, gain_knots=core.MAX_ENVELOPE_BANDS), 18.0, 22.0)
    assert not wfm.kernel_supports(many.sampled_waveform)
    assert wfm.kernel_supports(sonar_with("sas_auv", dict(wave, table_factor=8, gain_knots=core.MAX_ENVELOPE_BANDS - 1),
                                          18.0, 22.0).sampled_waveform)


def test_the_kernel_refuses_a_sampled_pulse_without_its_table():
    sonar = sonar_with("sas_auv", {"type": "sampled", "source": "lfm"}, 18.0, 22.0, omni=True)
    rec = sonar.make_record(make_pose(), environment={"sound_speed": C})
    q = np.array([sonar.boresight * 20.0])
    with pytest.raises(ValueError, match="envelope"):
        core.synthesize_into(rec, q, q, np.array([1.0 + 0j]))
    table, rate, offset, carrier, knots = sonar.sampled_waveform.kernel_table()
    with pytest.raises(ValueError, match="shape"):
        core.synthesize_into(rec, q, q, np.array([1.0 + 0j]), envelope=(table[:, :-1], rate, offset, carrier, knots))
    assert table.shape == (len(sonar.sampled_waveform.envelope) * 8 + 6, len(knots) + 1)
    np.testing.assert_allclose(table[:, -1], table[:, :-1].sum(axis=1), atol=1e-15)


# --- a constant envelope is a tone: the sampled branch against the analytic one ------------------

def _as_arbitrary(sonar, pose):
    rec = sonar.make_record(pose, environment={"sound_speed": C})
    tx = rec.transmit
    w = tx.waveform
    w.type = core.WaveformType.Arbitrary
    tx.waveform = w
    rec.transmit = tx
    assert rec.transmit.waveform.type == core.WaveformType.Arbitrary
    return rec


@pytest.mark.parametrize("kind", ["baseband", "real"])
def test_a_table_of_ones_gives_the_analytic_tone(kind):
    """A check of the kernel's sampled branch that does not go through the NumPy engine: with
    b(u) = 1 the echo is exp(2 pi i F u(t)), the analytic kernel's rectangular CW pulse (moving
    sonar and targets, platform acceleration, beam patterns, absorption at F). With the same
    pass count the two differ by rounding only."""
    f0, dur, n_fine = 8000.0, 0.02, 4000
    sonar = sonar_with("flank_32", {"type": "cw", "f_start": f0, "f_stop": f0, "duration": dur, "window_alpha": 0.0},
                       380.0, 430.0, kind, array={"count": 3})
    pose = make_pose(velocity=(2.0, 1.0, 0.0))
    q = np.array([sonar.boresight * 400.0, sonar.boresight * 405.0 + [30.0, 0.0, 10.0]])
    v = np.array([-12.0 * sonar.boresight, [0.5, 0.0, 0.0]])
    amp = np.array([1.0 + 0.5j, -0.7j])
    absorption = (6e3, 10e3, list(np.linspace(0.002, 0.012, core.ABSORPTION_TABLE_SIZE)))
    kw = dict(absorption=absorption, platform_acceleration=(0.2, 0.0, 0.1))
    ref = sonar.make_record(pose, environment={"sound_speed": C})
    core.synthesize_into(ref, q, q, amp, v, v, arrival_passes=(8, 8), **kw)
    out = _as_arbitrary(sonar, pose)
    table = np.ones((n_fine, 2), dtype=complex)
    core.synthesize_into(out, q, q, amp, v, v, envelope=(table, (n_fine - 6) / dur, 2.0, f0, [f0]), **kw)
    err = rel_rms(ref.samples.array, out.samples.array)
    print(f"table of ones ({kind}) vs analytic CW: {err:.2e}")
    assert err < 2e-7                                                   # measured 0 to 3e-8


def test_gain_across_the_band_is_the_weighted_sum_of_the_knot_gains():
    """Three parts of the envelope with weights (0.2, 0.3, 0.5) about 6, 8 and 10 kHz, and
    absorption rising across the band: the echo of a still target at range R is the tone
    times sum_i w_i 10^(-alpha(f_i) 2 R / 20). Checked against the analytic kernel's tone
    without absorption, scaled by that sum (omnidirectional elements)."""
    f0, dur, n_fine, r0 = 8000.0, 0.02, 4000, 400.0
    sonar = sonar_with("flank_32", {"type": "cw", "f_start": f0, "f_stop": f0, "duration": dur, "window_alpha": 0.0},
                       380.0, 430.0, omni=True, transmit={"lever_arm": [0.0, 0.0, 0.0]}, array={"count": 1})
    q = np.array([sonar.boresight * r0])
    one = np.array([1.0 + 0j])
    alpha = np.linspace(0.002, 0.012, core.ABSORPTION_TABLE_SIZE)               # dB/m over 6-10 kHz
    weights, knots = np.array([0.2, 0.3, 0.5]), [6e3, 8e3, 10e3]
    ref = sonar.make_record(make_pose(), environment={"sound_speed": C})
    core.synthesize_into(ref, q, q, one, arrival_passes=(8, 8))
    out = _as_arbitrary(sonar, make_pose())
    table = np.ones((n_fine, 4), dtype=complex)
    table[:, :3] = weights
    core.synthesize_into(out, q, q, one, absorption=(6e3, 10e3, list(alpha)),
                         envelope=(table, (n_fine - 6) / dur, 2.0, f0, knots))
    length = 2.0 * np.linalg.norm(q[0] - np.asarray(ref.rx_positions()[0]))
    expected = float(weights @ 10.0 ** (-np.interp(knots, np.linspace(6e3, 10e3, len(alpha)), alpha) * length / 20.0))
    assert 0.3 < expected < 0.7 and abs(expected - 10.0 ** (-0.007 * length / 20.0)) > 0.02
    err = rel_rms(ref.samples.array * expected, out.samples.array)
    print(f"three weighted bands vs the weighted sum of knot gains ({expected:.4f}): {err:.2e}")
    assert err < 1e-5


# --- the pass count of the arrival-time solve ------------------------------------------------------

def _closing_tone(passes, v=12.0, r0=400.0, f0=8000.0):
    """A long rectangular CW ping off a target closing at `v` from `r0`, and its closed form:
    the echo is a tone at f0 (c + v) / (c - v) starting at 2 r0 / (c + v)."""
    sonar = sonar_with("flank_32", {"type": "cw", "f_start": f0, "f_stop": f0, "duration": 0.05, "window_alpha": 0.0},
                       r0 - 20.0, r0 + 30.0, omni=True, transmit={"lever_arm": [0.0, 0.0, 0.0]}, array={"count": 1})
    rec = sonar.make_record(make_pose(), environment={"sound_speed": C})
    q = np.array([sonar.boresight * r0])
    v_t = np.array([-v * sonar.boresight])
    kw = {} if passes is None else {"arrival_passes": passes}
    core.synthesize_into(rec, q, q, np.array([1.0 + 0j]), v_t, v_t, spreading=False, **kw)
    s = rec.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    gamma = (C + v) / (C - v)
    t0 = 2.0 * r0 / (C + v)
    on = (t >= t0) & (gamma * (t - t0) <= 0.05)
    exact = np.where(on, np.exp(2j * np.pi * (f0 * gamma * (t - t0) - s.center_frequency * t)), 0.0)
    x = s.array[0] / rec.transmit.waveform.amplitude
    inner = np.nonzero(on)[0][2:-2]
    return x, rel_rms(exact[inner], x[inner]), float(np.angle(np.vdot(exact[inner], x[inner])))


def test_arrival_passes_default_is_unchanged_and_more_passes_reach_the_closed_form():
    """The analytic kernel stops the arrival-time solve after 3 passes outbound and 4 on
    return. Each pass gains a factor v / c; what 3 passes leave outbound is about
    (R / c)(v / c)^3 = 0.14 us for 12 m/s at 400 m, a phase of 0.007 rad at 8 kHz. Measured
    against the closed form (relative rms, which for a pure phase error is the phase in rad):

        passes (out, back)   (3, 4) default   (4, 4)    (4, 5)    (5, 5)    (8, 8)
        error                6.9e-3           5.6e-5    5.6e-5    5e-7      1e-7 (float32)

    The default is left as it was (existing records stay bit-identical); `arrival_passes`
    sets it per call, and sampled pulses use (8, 8) like the NumPy reference."""
    default, e_default, ph_default = _closing_tone(None)
    explicit, e34, _ = _closing_tone((3, 4))
    np.testing.assert_array_equal(default, explicit)
    errors = {p: _closing_tone(p)[1] for p in ((4, 4), (4, 5), (5, 5), (8, 8))}
    print(f"arrival passes vs closed form, 12 m/s at 400 m, 8 kHz: (3, 4) {e34:.2e} (phase {ph_default:+.2e} rad), "
          + ", ".join(f"{p} {e:.2e}" for p, e in errors.items()))
    predicted = 2 * np.pi * 8000.0 * (400.0 / C) * (12.0 / C) ** 3
    assert e34 == pytest.approx(predicted, rel=0.15)                  # 6.9e-3 measured, 6.9e-3 predicted
    assert errors[(4, 4)] == pytest.approx(predicted * 12.0 / C, rel=0.3)
    assert errors[(5, 5)] < 2e-6 and errors[(8, 8)] < 5e-7
    with pytest.raises(ValueError, match="passes"):
        _closing_tone((0, 4))


# --- CUDA against host (skipped without a device) -------------------------------------------------

@needs_cuda
@pytest.mark.parametrize("kind", ["baseband", "real"])
@pytest.mark.parametrize("name, r0, knots", [("fls_300k", 20.0, 8), ("flank_32", 300.0, 8), ("sas_auv", 20.0, 0)])
def test_cuda_and_host_kernels_agree_on_a_sampled_pulse(name, r0, kind, knots):
    """The device runs the same kernel body (synth_kernel.hpp) on a device copy of the
    envelope table; only the order of the float32 additions into a sample differs."""
    dur = min(SonarModel.load(name).waveform_config["duration"], 0.02)
    wave = {"type": "sampled", "source": "lfm", "duration": dur, "window_alpha": 0.2, "gain_knots": knots}
    sonar = sonar_with(name, wave, r0 - 3.0, r0 + 3.0, kind)
    kw = dict(environment={"sound_speed": C}, adc=False, absorption_model="francois_garrison", acceleration=(0.3, -0.2, 0.1))
    pose, scene = make_pose(velocity=(1.0, 0.5, 0.0)), _five_targets(sonar, r0)
    host = simulate_ping(sonar, pose, scene, backend=core.Backend.Host, **kw)
    cuda = simulate_ping(sonar, pose, scene, backend=core.Backend.Cuda, **kw)
    ref = simulate_ping(sonar, pose, scene, engine="numpy", **kw)
    assert "echo_synthesis" not in json.loads(cuda.metadata.extra_json or "{}").get("gpusonar", {})
    assert_same(cuda.samples.array, host.samples.array, f"{name} CUDA vs host ({kind})")
    assert_same(cuda.samples.array, ref.samples.array, f"{name} CUDA vs NumPy ({kind})")


@needs_cuda
@pytest.mark.parametrize("label", ["image_source + time spread", "image_source + reverberation"])
def test_cuda_and_host_agree_on_a_sampled_pulse_through_a_scenario(label):
    cfg = _scenario("image_source", {"type": "sampled", "source": "mseq", "degree": 7, "duration": 0.02},
                    **dict(TIERS[label]))
    host = run_scenario(dict(cfg), backend=core.Backend.Host).records[0]
    cuda = run_scenario(dict(cfg), backend=core.Backend.Cuda).records[0]
    assert_same(cuda.samples.array, host.samples.array, f"{label}: CUDA vs host")


@needs_cuda
def test_cuda_kernel_takes_the_arrival_pass_count():
    """`arrival_passes` reaches the device kernel (it is in the parameter block both backends
    share): the analytic tone is the same on both at each setting."""
    f0 = 8000.0
    sonar = sonar_with("flank_32", {"type": "cw", "f_start": f0, "f_stop": f0, "duration": 0.02, "window_alpha": 0.0},
                       380.0, 430.0, omni=True, array={"count": 2})
    q = np.array([sonar.boresight * 400.0])
    v = np.array([-12.0 * sonar.boresight])
    out = {}
    for passes in ((3, 4), (8, 8)):
        for backend in (core.Backend.Host, core.Backend.Cuda):
            rec = sonar.make_record(make_pose(), environment={"sound_speed": C})
            core.synthesize_into(rec, q, q, np.array([1.0 + 0j]), v, v, arrival_passes=passes, backend=backend)
            out[passes, backend] = rec.samples.array
        assert_same(out[passes, core.Backend.Cuda], out[passes, core.Backend.Host], f"passes {passes}: CUDA vs host")
    assert rel_rms(out[(8, 8), core.Backend.Cuda], out[(3, 4), core.Backend.Cuda]) > 1e-3
