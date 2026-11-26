"""Pulse trains (`transmit.waveform: {type: sampled, source: train, segments: [...]}`, sim/waveform.py `build_train`;
docs/tasks/DOPPLER.md package D2) and the transmitted-energy accounting (sonar.py `source_energy_db`).

- a one-segment train IS today's sampled pulse (bit for bit: the same samples from the same expressions), and a
  one-tone train is the analytic CW to interpolation accuracy;
- the train's envelope equals its analytic pieces; its spectrum and energy per segment are the configured ones;
- the energy at 1 m of a free-field echo equals `source_energy_db` (and what `source_level_db_rms` has always meant:
  the level of the envelope's peak before the taper, so a Hann tone at the sweep's level carries 0.375 / 0.969 of the
  sweep's energy);
- both synthesis engines (C++ kernel and NumPy reference) agree on a train as on any sampled pulse; CUDA when present;
- a moving point target's echo is the whole train time-scaled by the closed-form two-way factor, so each segment
  carries its Doppler: the tone at Gamma f, the sweep's matched filter peaking at the true range rate;
- the record holds the train in its existing fields (type Arbitrary, the replica as sent): it validates, survives
  the container file, and both live chains read it.

Measured numbers are printed (run with -s) and quoted in the assertions' comments.
"""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar.live import LiveChain
from gpusonar.processing import doppler_scale
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sim import waveform as wfm
from gpusonar.sonar import SonarModel

from test_sampled_engine import assert_same, both, needs_cuda
from test_sampled_waveform import C, _five_targets, rel_rms, sonar_with

core = gs.core

F_CW, T_CW, T_SWEEP = 8000.0, 0.05, 0.05


def train(cw=None, sweep=None, gap=0.0, **top):
    """A tone (Hann) then a 6-10 kHz sweep (Tukey 0.2), short so that the tests run quickly."""
    cw = {"kind": "cw", "frequency": F_CW, "duration": T_CW, "window_alpha": 1.0, **(cw or {})}
    sweep = {"kind": "lfm", "f_start": 6e3, "f_stop": 10e3, "duration": T_SWEEP, "window_alpha": 0.2, **(sweep or {})}
    return {"type": "sampled", "source": "train", "segments": [cw, {"kind": "gap", "duration": gap}, sweep], **top}


def gamma_closing(v):
    """Two-way time-scale factor of a target closing at v on a still sonar (TR 0702 eq. 68, radial)."""
    return (C + v) / (C - v)


# --- the waveform itself ---------------------------------------------------------------------------------

@pytest.mark.parametrize("source", ["lfm", "hfm"])
def test_a_one_segment_train_is_todays_sampled_sweep_bit_for_bit(source):
    """The same grid, the same expressions in the same order: equal samples, equal record, both engines."""
    old = {"type": "sampled", "source": source, "f_start": 6e3, "f_stop": 10e3, "duration": 0.02, "window_alpha": 0.2}
    new = {"type": "sampled", "source": "train",
           "segments": [{"kind": source, "f_start": 6e3, "f_stop": 10e3, "duration": 0.02, "window_alpha": 0.2}]}
    a, b = wfm.build_waveform(old), wfm.build_waveform(new)
    np.testing.assert_array_equal(a.envelope, b.envelope)
    assert (a.sample_rate, a.carrier, a.f_start, a.f_stop, a.duration) == (b.sample_rate, b.carrier, b.f_start, b.f_stop, b.duration)
    recs = []
    for wave in (old, new):
        sonar = sonar_with("flank_32", wave, 297.0, 303.0)
        recs.append(both(sonar, make_pose(velocity=(1.0, 0.0, 0.0)), _five_targets(sonar, 300.0), absorption_model="francois_garrison"))
    for i in range(2):
        np.testing.assert_array_equal(recs[0][i].samples.array, recs[1][i].samples.array)
        np.testing.assert_array_equal(np.asarray(recs[0][i].transmit.waveform.replica), np.asarray(recs[1][i].transmit.waveform.replica))


@pytest.mark.parametrize("kind", ["baseband", "real"])
def test_a_one_tone_train_is_the_analytic_cw_to_interpolation_accuracy(kind):
    """What stopped a sampled CW: build_waveform takes the envelope's rate from the stated band, so f_start == f_stop
    (a tone's nominal band) is refused. A tone in a train has a band, f +- 16 / T, and is a sampled pulse. Against the
    analytic CW (type 0, the same Tukey taper) the record differs by interpolation only: measured 1.8e-6 (baseband
    and real), omnidirectional elements, no absorption (the sampled LFM's own figure at oversample 8 is 2.4e-6)."""
    with pytest.raises(ValueError, match="f_start and f_stop must differ"):
        wfm.build_waveform({"type": "sampled", "source": "lfm", "f_start": F_CW, "f_stop": F_CW, "duration": 0.02})
    tone = {"type": "sampled", "source": "train",
            "segments": [{"kind": "cw", "frequency": F_CW, "duration": 0.02, "window_alpha": 0.5}]}
    analytic = {"type": "cw", "f_start": F_CW, "f_stop": F_CW, "duration": 0.02, "window_alpha": 0.5}
    w = wfm.build_waveform(tone)
    assert (w.f_start, w.f_stop) == (F_CW - 800.0, F_CW + 800.0) and w.carrier == F_CW
    recs = []
    for wave in (analytic, tone):
        sonar = sonar_with("flank_32", wave, 297.0, 303.0, kind, omni=True)
        recs.append(simulate_ping(sonar, make_pose(velocity=(1.0, 0.0, 0.0)), _five_targets(sonar, 300.0),
                                  environment={"sound_speed": C}, absorption_model="none", adc=False).samples.array)
    err = rel_rms(recs[0], recs[1])
    print(f"one-tone train vs analytic CW ({kind}): {err:.2e}")
    assert err < 1e-5


def test_the_train_envelope_is_its_analytic_pieces():
    """On its own grid the envelope IS the closed form of each piece (to rounding: the plan's 1e-6 holds there with
    room). Between samples the default interpolator reproduces it to its own accuracy: measured worst 4.8e-6 of the
    peak (median 3e-7); a single tone or sweep built alone shows the same floor, 2.9e-6, at oversample 8 and 16 alike
    (the Kaiser window's table, linear between 1024 entries a sample), so it is the interpolator's, not the train's."""
    wave = train(gap=0.01)
    w = wfm.build_waveform(wave)
    cw, gap, sweep = w.segments
    assert [s.kind for s in w.segments] == ["cw", "gap", "lfm"]
    assert (cw.start, gap.start, sweep.start, w.duration) == (0.0, T_CW, T_CW + 0.01, pytest.approx(T_CW + 0.01 + T_SWEEP))

    def closed(u):
        out = np.zeros(len(u), dtype=complex)
        for s in (cw, sweep):
            ul = u - s.start
            x = np.clip(ul / s.duration, 0.0, 1.0)
            if s.kind == "cw":
                win = 0.5 * (1.0 - np.cos(2.0 * np.pi * x))                                   # Hann
                sig = np.exp(2j * np.pi * F_CW * ul)
            else:
                edge = 0.1
                win = np.where(x < edge, 0.5 * (1 - np.cos(np.pi * x / edge)), np.where(x > 1 - edge, 0.5 * (1 - np.cos(np.pi * (1 - x) / edge)), 1.0))
                sig = np.exp(2j * np.pi * (6e3 * ul + 0.5 * 4e3 / T_SWEEP * ul * ul))
            inside = (ul >= 0.0) & (ul <= s.duration)
            out += np.where(inside, s.scale * win * sig, 0.0)
        return out * np.exp(-2j * np.pi * w.carrier * u)

    u = np.arange(len(w.envelope)) / w.sample_rate
    on_grid = float(np.abs(w.envelope - closed(u)).max())
    rng = np.random.default_rng(2)
    v = np.sort(rng.uniform(0.0, w.duration, 4000))
    off_grid = float(np.abs(w.evaluate(v) - closed(v)).max())
    print(f"train envelope vs its pieces: on the grid {on_grid:.1e}, between samples {off_grid:.1e} of the peak")
    assert on_grid < 1e-10                     # measured 1.4e-12: the phase's rounding at 900 cycles
    assert off_grid < 1e-5
    # Inside the gap the pulse is silent; the record's band holds both segments.
    in_gap = (u > gap.start) & (u < gap.stop)
    assert in_gap.sum() > 300 and np.all(w.envelope[in_gap] == 0.0)
    assert np.abs(w.evaluate(np.linspace(gap.start + 1e-4, gap.stop - 1e-4, 50))).max() < 1e-6   # the sinc's tails
    assert (w.f_start, w.f_stop) == (6e3, 10e3)


def test_spectrum_and_energy_per_segment_follow_the_shares():
    """energy_share 0.3 / 0.7: the segments' energies (integral |b|^2) in that ratio, and in the spectrum of the record's
    replica the tone's energy sits within +-4 / T of its frequency while the sweep's spreads over 6-10 kHz."""
    wave = train(cw={"energy_share": 0.3}, sweep={"energy_share": 0.7})
    w = wfm.build_waveform(wave)
    cw, _, sweep = w.segments
    assert cw.energy / (cw.energy + sweep.energy) == pytest.approx(0.3, rel=1e-9)
    assert w.envelope_energy() == pytest.approx(cw.energy + sweep.energy, rel=1e-12)
    assert max(np.abs(w.envelope)) == pytest.approx(1.0)                   # normalised to the peak, as every sampled pulse
    rate = 64e3
    p = w.passband(rate)
    u = np.arange(len(p)) / rate
    spectra = {}
    for s in (cw, sweep):
        seg = np.where((u >= s.start) & (u <= s.stop), p, 0.0)
        spectra[s.kind] = np.abs(np.fft.rfft(seg, 1 << 18)) ** 2
    f = np.fft.rfftfreq(1 << 18, 1.0 / rate)
    tone = spectra["cw"][np.abs(f - F_CW) <= 4.0 / T_CW].sum() / spectra["cw"].sum()
    sweep_in = spectra["lfm"][(f >= 6e3) & (f <= 10e3)].sum() / spectra["lfm"].sum()
    energy_ratio = spectra["cw"].sum() / (spectra["cw"].sum() + spectra["lfm"].sum())
    print(f"tone within +-4/T: {tone:.6f} of its energy; sweep inside 6-10 kHz: {sweep_in:.4f}; tone share {energy_ratio:.4f}")
    assert tone > 0.9999 and sweep_in > 0.99
    assert energy_ratio == pytest.approx(0.3, abs=2e-3)                    # the replica's float32 samples, Riemann sums


def test_a_comb_of_tones_shares_its_segment_and_schroeder_phases_lower_its_peak():
    """Option E (docs/tasks/DOPPLER.md section 4): several tones at once in one segment. Equal energy per tone, and
    Schroeder's phases give a lower peak than all-zero phases (so more energy for the same peak)."""
    freqs = [6.5e3, 7.5e3, 8.5e3, 9.5e3]
    seg = {"kind": "cw", "frequencies": freqs, "duration": 0.05, "window_alpha": 1.0}
    out = {}
    for rule in ("schroeder", "zero"):
        w = wfm.build_waveform({"type": "sampled", "source": "train", "f_start": 6e3, "f_stop": 10e3,
                                "segments": [{**seg, "comb_phase": rule}]})
        out[rule] = w.envelope_energy()
        rate = 64e3
        p = w.passband(rate)
        spec = np.abs(np.fft.rfft(p, 1 << 18)) ** 2
        f = np.fft.rfftfreq(1 << 18, 1.0 / rate)
        per_tone = [spec[np.abs(f - fk) <= 100.0].sum() for fk in freqs]
        assert np.ptp(per_tone) / np.mean(per_tone) < 1e-3
    print(f"comb energy at unit peak: Schroeder {out['schroeder']:.4f} s, zero phases {out['zero']:.4f} s")
    assert out["schroeder"] > 1.5 * out["zero"]


def test_bad_trains_are_refused_by_name():
    with pytest.raises(ValueError, match=r"segments\[0\]\.frequncy: unknown key; did you mean 'frequency'\?"):
        wfm.build_waveform({"type": "sampled", "source": "train", "segments": [{"kind": "cw", "frequncy": 8e3, "duration": 0.1}]})
    with pytest.raises(ValueError, match=r"transmit\.waveform\.segmnts: unknown key; did you mean 'segments'\?"):
        wfm.build_waveform({"type": "sampled", "source": "train", "segmnts": []})
    with pytest.raises(ValueError, match="not a segment kind"):
        wfm.build_waveform(train(cw={"kind": "tone"}))
    with pytest.raises(ValueError, match="outside the stated band"):
        wfm.build_waveform({**train(), "f_start": 7e3, "f_stop": 10e3})
    with pytest.raises(ValueError, match="on every tone and sweep"):
        wfm.build_waveform(train(cw={"energy_share": 0.5}))
    with pytest.raises(ValueError, match="not both"):
        wfm.build_waveform(train(cw={"energy_share": 0.5}, sweep={"source_energy_db": 190.0}))
    with pytest.raises(ValueError, match="`frequency`"):
        wfm.build_waveform(train(cw={"frequencies": [8e3, 9e3]}))
    with pytest.raises(ValueError, match="window_alpha"):
        wfm.build_waveform(train(sweep={"window_alpha": 1.5}))


# --- energy accounting -----------------------------------------------------------------------------------

def _echo_energy_at_1m_db(wave, r0=300.0, overrides=None):
    """Energy (dB re 1 uPa^2 s) of the free-field echo of a 0 dB point target with spreading off, i.e. of the pulse at
    1 m: sensitivity and gain 0 dB, so the record is in uPa; a baseband record's envelope e has integral p^2 = 0.5
    integral |e|^2. Returns it and the sonar."""
    extra = {"receive": {"sensitivity_db": 0.0, "gain_db": 0.0}, **(overrides or {})}
    sonar = sonar_with("flank_32", wave, r0 - 5.0, r0 + 5.0, "baseband", omni=True, **extra)
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([sonar.boresight * r0]), environment={"sound_speed": C},
                        absorption_model="none", adc=False, spreading=False)
    e = np.asarray(rec.samples.array[0], dtype=complex)
    return 10.0 * np.log10(0.5 * np.sum(np.abs(e) ** 2) / rec.samples.sample_rate), sonar, rec


def test_the_source_level_key_means_what_it_meant_and_the_energy_is_accounted():
    """`source_level_db_rms` is unchanged: the RMS level of a constant-envelope pulse at the envelope's peak, applied
    before the taper. So the energy at 1 m is SL + 10 log10(T (1 - 5 alpha / 8)): measured on the echo within 1e-3 dB
    for an LFM (Tukey 0.05) and a Hann CW; at the same SL the Hann CW carries 4.12 dB less (the 4.1 dB of
    docs/tasks/DOPPLER.md section 0 item 4)."""
    lfm = {"type": "lfm", "f_start": 6e3, "f_stop": 10e3, "duration": 0.05, "window_alpha": 0.05, "source_level_db_rms": 200.0}
    cw = {"type": "cw", "f_start": F_CW, "f_stop": F_CW, "duration": 0.05, "window_alpha": 1.0, "source_level_db_rms": 200.0}
    got = {}
    for name, wave in (("lfm", lfm), ("cw", cw)):
        measured, sonar, rec = _echo_energy_at_1m_db(wave)
        assert sonar.source_level_db_rms == 200.0                        # the key is read as before
        assert rec.transmit.waveform.amplitude == pytest.approx(np.sqrt(2.0) * 10.0 ** 10.0)   # peak, before the taper
        assert sonar.source_energy_db == pytest.approx(200.0 + 10.0 * np.log10(0.05 * (1.0 - 5.0 * wave["window_alpha"] / 8.0)))
        print(f"{name}: echo energy at 1 m {measured:.4f} dB, accounted {sonar.source_energy_db:.4f} dB")
        assert measured == pytest.approx(sonar.source_energy_db, abs=1e-3)
        got[name] = measured
    assert got["lfm"] - got["cw"] == pytest.approx(10.0 * np.log10((1.0 - 5.0 * 0.05 / 8.0) / 0.375), abs=1e-3)   # 4.12 dB


def test_source_energy_states_the_energy_and_the_level_follows():
    """Opt-in `source_energy_db` (dB re 1 uPa^2 s at 1 m): the sonar derives SL from it, the echo carries that energy,
    and a Hann tone and a Tukey sweep given the same energy echo the same energy. Given with a different SL: refused."""
    for wave in ({"type": "lfm", "f_start": 6e3, "f_stop": 10e3, "duration": 0.05, "window_alpha": 0.05},
                 {"type": "cw", "f_start": F_CW, "f_stop": F_CW, "duration": 0.05, "window_alpha": 1.0},
                 train(cw={"energy_share": 0.5}, sweep={"energy_share": 0.5})):
        measured, sonar, _ = _echo_energy_at_1m_db({**wave, "source_level_db_rms": None, "source_energy_db": 185.0})
        print(f"{wave.get('source', wave['type'])}: SL derived {sonar.source_level_db_rms:.3f} dB; echo energy {measured:.4f} dB")
        assert measured == pytest.approx(185.0, abs=1e-3)
        assert sonar.source_energy_db == pytest.approx(185.0, abs=1e-9)
    with pytest.raises(ValueError, match="both given and differ"):
        sonar_with("flank_32", {"source_level_db_rms": 200.0, "source_energy_db": 185.0}, 290.0, 310.0)
    with pytest.raises(ValueError, match="only for CW, LFM and sampled"):
        sonar_with("flank_32", {"type": "gaussian", "source_level_db_rms": None, "source_energy_db": 185.0}, 290.0, 310.0)
    # Without the key nothing changes: the configured SL stands.
    assert sonar_with("flank_32", {}, 290.0, 310.0).source_level_db_rms == 200.0


def test_a_train_states_each_segments_energy_and_the_echo_carries_it():
    """`source_energy_db` on each segment: the train's total is their sum, the sonar's SL follows, and the echo's energy
    in each segment's time span is the stated one (within 0.01 dB: the spans' edges hold a little of the other)."""
    wave = train(cw={"source_energy_db": 180.0}, sweep={"source_energy_db": 186.0}, gap=0.01)
    wave = {**wave, "source_level_db_rms": None}
    measured, sonar, rec = _echo_energy_at_1m_db(wave)
    assert measured == pytest.approx(10.0 * np.log10(10.0 ** 18.0 + 10.0 ** 18.6), abs=1e-3)
    seg_db = sonar.segment_energies_db()
    assert seg_db[0] == pytest.approx(180.0, abs=1e-9) and seg_db[1] is None and seg_db[2] == pytest.approx(186.0, abs=1e-9)
    s = rec.samples
    e = np.asarray(s.array[0], dtype=complex)
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate - 2.0 * 300.0 / C
    for seg, want in ((sonar.sampled_waveform.segments[0], 180.0), (sonar.sampled_waveform.segments[2], 186.0)):
        inside = (t >= seg.start - 1e-3) & (t <= seg.stop + 1e-3)
        got = 10.0 * np.log10(0.5 * np.sum(np.abs(e[inside]) ** 2) / s.sample_rate)
        print(f"{seg.kind}: echo energy {got:.4f} dB, stated {want}")
        assert got == pytest.approx(want, abs=0.01)


# --- synthesis: engines, CUDA, Doppler -------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["baseband", "real"])
def test_engines_agree_on_a_train(kind):
    """C++ kernel against the NumPy reference, beam patterns and absorption across the band, a moving sonar and
    moving targets: the tolerance of test_sampled_engine.py (rms 2e-6, worst sample 5e-6 of the peak). Measured
    rms 6.0e-8, worst sample 1.5e-7 of the peak (baseband and real)."""
    sonar = sonar_with("flank_32", train(cw={"energy_share": 0.5}, sweep={"energy_share": 0.5}, gap=0.005), 295.0, 310.0, kind)
    rng = np.random.default_rng(1)
    scene = Scene().add_points(sonar.boresight * 300.0 + rng.uniform(-1, 1, (5, 3)), velocity=(0.5, -1.5, 0.0),
                               amplitude=np.exp(1j * rng.uniform(0, 6, 5)))
    a, b = both(sonar, make_pose(velocity=(1.0, 0.5, 0.0)), scene, absorption_model="francois_garrison",
                acceleration=(0.3, -0.2, 0.1))
    assert a.transmit.waveform.type == core.WaveformType.Arbitrary
    assert_same(a.samples.array, b.samples.array, f"train ({kind})")


@needs_cuda
@pytest.mark.parametrize("kind", ["baseband", "real"])
def test_cuda_and_host_agree_on_a_train(kind):
    sonar = sonar_with("flank_32", train(cw={"energy_share": 0.5}, sweep={"energy_share": 0.5}), 295.0, 310.0, kind)
    kw = dict(environment={"sound_speed": C}, adc=False, absorption_model="francois_garrison", acceleration=(0.3, -0.2, 0.1))
    pose, scene = make_pose(velocity=(1.0, 0.5, 0.0)), _five_targets(sonar, 300.0)
    host = simulate_ping(sonar, pose, scene, backend=core.Backend.Host, **kw)
    cuda = simulate_ping(sonar, pose, scene, backend=core.Backend.Cuda, **kw)
    ref = simulate_ping(sonar, pose, scene, engine="numpy", **kw)
    assert "echo_synthesis" not in json.loads(cuda.metadata.extra_json or "{}").get("gpusonar", {})
    assert_same(cuda.samples.array, host.samples.array, f"train CUDA vs host ({kind})")
    assert_same(cuda.samples.array, ref.samples.array, f"train CUDA vs NumPy ({kind})")


def _closing_echo(engine, wave, v=6.0, r0=400.0):
    sonar = sonar_with("flank_32", wave, r0 - 20.0, r0 + 40.0, "baseband", omni=True,
                       transmit={"lever_arm": [0.0, 0.0, 0.0]}, array={"count": 1})
    target = sonar.boresight * r0
    rec = simulate_ping(sonar, make_pose(), Scene().add_points([target], velocity=-v * sonar.boresight),
                        environment={"sound_speed": C}, absorption_model="none", adc=False, spreading=False, engine=engine)
    return sonar, rec


@pytest.mark.parametrize("engine", ["cpp", "numpy"])
def test_a_moving_targets_echo_carries_the_doppler_on_both_segments(engine):
    """A target closing at 6 m/s: the echo is the train time-scaled by Gamma = (c + v) / (c - v) (measured relative rms
    against the closed form 5.7e-8 on both engines); the tone's spectral peak sits at Gamma f (within 1 % of the 1/T resolution); the
    sweep's matched filter over a bank of range rates peaks at the true -6 m/s; and the echo is the sum of the echoes
    of its pieces sent alone at their own times (the same grid: measured 1.6e-16)."""
    v, r0 = 6.0, 400.0
    wave = train(gap=0.01, normalise="none")
    sonar, rec = _closing_echo(engine, wave, v, r0)
    s = rec.samples
    x = np.asarray(s.array[0], dtype=complex) / rec.transmit.waveform.amplitude
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    g = gamma_closing(v)
    t0 = 2.0 * r0 / (C + v)
    w = sonar.sampled_waveform
    arg = g * (t - t0)
    want = w.evaluate(arg) * np.exp(2j * np.pi * (w.carrier * arg - s.center_frequency * t))
    err = rel_rms(want, x)
    print(f"{engine}: train echo vs the time-scaled train {err:.2e}")
    assert err < 1e-6
    cw, _, sweep = w.segments
    # The tone: the spectral peak of the echo's first segment.
    tone = np.where((arg >= cw.start) & (arg <= cw.stop), x, 0.0)
    n_fft = 1 << 22
    spec = np.abs(np.fft.fft(tone, n_fft)) ** 2
    f = np.fft.fftfreq(n_fft, 1.0 / s.sample_rate) + s.center_frequency
    f_peak = f[int(np.argmax(spec))]
    print(f"{engine}: tone echo at {f_peak:.3f} Hz, Gamma f = {g * F_CW:.3f} Hz (still: {F_CW} Hz)")
    assert f_peak == pytest.approx(g * F_CW, abs=0.01 / T_CW)
    # The sweep: a bank of its own replicas time-scaled per range rate (processing.doppler_scale), peak at the truth.
    rates = np.arange(-12.0, 12.01, 0.5)
    peaks = []
    for vr in rates:
        eta = doppler_scale(vr, C)
        tr = np.arange(int(np.floor(sweep.duration / eta * s.sample_rate)) + 1) / s.sample_rate
        rep = sweep.signal(eta * tr) * np.exp(-2j * np.pi * s.center_frequency * tr)
        z = np.abs(np.correlate(x, rep, mode="valid")) / np.sqrt(np.sum(np.abs(rep) ** 2))
        peaks.append(z.max())
    best = rates[int(np.argmax(peaks))]
    print(f"{engine}: sweep segment's bank peaks at {best:+.1f} m/s (truth {-v:+.1f})")
    assert best == pytest.approx(-v, abs=0.25)
    # Linearity: the pieces sent alone (each with silence where the other was, on the same band and grid).
    pieces = []
    for keep in (0, 2):
        segs = [dict(sg) if i == keep else {"kind": "gap", "duration": sg["duration"]} for i, sg in enumerate(wave["segments"])]
        part = {**wave, "segments": segs, "f_start": 6e3, "f_stop": 10e3}
        pieces.append(np.asarray(_closing_echo(engine, part, v, r0)[1].samples.array[0], dtype=complex))
    n = min(len(pieces[0]), len(pieces[1]), len(x))
    summed = (pieces[0][:n] + pieces[1][:n]) / rec.transmit.waveform.amplitude
    lin = rel_rms(x[:n], summed)
    print(f"{engine}: train echo vs the sum of its pieces' echoes {lin:.2e}")
    assert lin < 1e-5


# --- the record, and the readers that take it -----------------------------------------------------------

def test_the_record_of_a_train_validates_round_trips_and_needs_no_new_field(tmp_path):
    """Type Arbitrary with its replica as sent, f_start / f_stop spanning the train's band, the duration its segments'
    sum; validate() accepts it; the container file returns it; the envelope recovered from the file is the train."""
    wave = train(cw={"energy_share": 0.5}, sweep={"energy_share": 0.5}, gap=0.005)
    sonar = sonar_with("flank_32", wave, 295.0, 310.0)
    rec = simulate_ping(sonar, make_pose(), _five_targets(sonar, 300.0), environment={"sound_speed": C}, adc=False)
    report = gs.validate(rec)
    assert report.ok, str(report)
    w = rec.transmit.waveform
    assert w.type == core.WaveformType.Arbitrary
    assert (w.f_start, w.f_stop) == (6e3, 10e3)
    assert w.duration == pytest.approx(T_CW + 0.005 + T_SWEEP)
    path = str(tmp_path / "train.h5")
    with core.RecordWriter(path, core.WriterOptions()) as writer:
        writer.write(rec)
    back = core.RecordReader(path).read(0)
    np.testing.assert_array_equal(np.asarray(back.transmit.waveform.replica), np.asarray(w.replica))
    assert back.transmit.waveform.duration == w.duration and back.transmit.waveform.f_start == 6e3
    np.testing.assert_array_equal(back.samples.array, rec.samples.array)
    recovered = wfm.SampledWaveform.from_record(back.transmit.waveform)
    u = np.linspace(0.0, w.duration, 3000)[5:-5]
    err = rel_rms(sonar.sampled_waveform.evaluate(u), recovered.evaluate(u))
    print(f"train envelope recovered from the file's replica: relative rms {err:.2e}")
    assert err < 1e-3                                                # measured 2.2e-6


def test_both_live_chains_read_train_records_alike():
    """The live chains (not changed by this package) take a train record as any sampled pulse: matched filter on the
    whole replica as sent. On the two-target ping set of test_live.py with a 0.1 s Hann tone and a 0.1 s sweep, both
    engines report the same detections, and both targets are among them."""
    from test_live_cpp import CPP_CHAIN
    from test_live import CHAIN, OVERRIDES
    wave = train(cw={"duration": 0.1, "energy_share": 0.5}, sweep={"duration": 0.1, "energy_share": 0.5})
    sonar = SonarModel.load("flank_32", gs.config.deep_merge(OVERRIDES, {"transmit": {"waveform": wave}}))
    rng = np.random.default_rng(5)
    pings = []
    for i in range(3):
        t = 2.0 * i
        scene = Scene().add_points([[40.0 - 3.0 * t, 250.0, 0.0], [-60.0, 320.0, 0.0]], target_strength_db=10.0)
        pings.append(simulate_ping(sonar, make_pose(position=(1.5 * t, 0.0, 0.0), velocity=(1.5, 0.0, 0.0)), scene,
                                   environment={"sound_speed": 1500.0}, sequence=i, time_ns=int(t * 1e9), noise_rms=1e-5, rng=rng))
    with LiveChain(CHAIN) as chain:
        want = chain.run([p.copy() for p in pings])
    cpp = core.LiveChainCpp(CPP_CHAIN)
    got = [json.loads(cpp.process(p.copy())) for p in pings]
    for w_, g_ in zip(want, got):
        assert len(w_["detections"]) == len(g_["detections"]) >= 2
        for a, b in zip(w_["detections"], g_["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
    last = want[-1]["detections"]
    for x, y in ((40.0 - 3.0 * 4.0 - 1.5 * 4.0, 250.0), (-60.0 - 1.5 * 4.0, 320.0)):
        r = np.hypot(x, y)
        assert min(abs(d["range_m"] - r) for d in last) < 3.0
