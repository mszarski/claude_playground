"""D4: the sweep's Doppler bank in the live chain (docs/tasks/DOPPLER.md section 6; libs/live doppler.hpp;
gpusonar/doppler.py), opt-in as `chain.doppler: {mode: sweep_bank}`. The engine-parity cases are in test_live_cpp.py.

The mismatch of a linear sweep, ours (derived here, held against an ideal sweep and the simulator). An echo of range rate
r (opening positive) is the pulse time-scaled by eta = (c - r) / (c + r): its sweep rate is k eta^2 against the
replica's k = B / T. The correlation's phase difference over the pulse is 2 pi [(f0 (eta - 1) + k tau) t + k (eta^2 - 1)
t^2 / 2]: the delay tau takes the linear term (the range-Doppler coupling), and a quadratic error is left whose value at
the ends of the pulse, about its centre, is

    phi = pi k |eta^2 - 1| T^2 / 4 = pi B T |r| / c          (eta^2 - 1 = -4 r / c to first order)

The compressed peak over the matched one is then M(phi) = max over x0 of |integral_{-1/2}^{1/2} exp(i 4 phi (u - x0)^2)
du|: the delay is free, so the residual chirp's stationary point x0 (in pulse lengths from the centre) moves to where
the Fresnel integral is largest (for phi above about 2 that is not the centre: at 2 m/s on a 1 s, 4 kHz sweep the
centred value would say 13.5 dB, the free delay 11.7 dB, an ideal sweep 11.7 dB). The design note's tolerance phi = pi / 2
(|r| = c / (2 B T)) costs 0.97 dB; hypotheses that far apart leave at most phi = pi / 4 between an echo and the nearest
one: 0.24 dB.
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
WINDOW = {"receive": {"window": {"range": [100.0, 400.0]}}}     # the chain's derived training windows need the room
CHAIN = {"bearings_deg": [-50.0, 50.0, 1.0], "backend": "host"}
BANK = {"mode": "sweep_bank", "max_range_rate_m_s": 2.5}


def closed_form_db(r, band, duration):
    """-20 log10 M(phi), phi = pi B T |r| / c: the peak a sweep loses on an echo of range rate r (ours, above)."""
    phi = np.pi * band * duration * abs(r) / C
    u = np.linspace(-0.5, 0.5, 20001)
    best = max(abs(np.trapezoid(np.exp(4j * phi * (u - x0) ** 2), u)) for x0 in np.linspace(0.0, 0.8, 321))
    return float(-20.0 * np.log10(best))


def ping(target=(0.0, 250.0, 0.0), velocity=(0.0, 0.0, 0.0), own=(0.0, 0.0, 0.0), noise_rms=1e-4, seed=1, ts_db=0.0, sequence=0):
    """flank_32 (6-10 kHz LFM, 1 s) with one point target; the array along x, broadside +y."""
    sonar = SonarModel.load("flank_32", WINDOW)
    scene = Scene().add_points([list(target)], velocity=velocity, target_strength_db=ts_db)
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0), velocity=own), scene, environment={"sound_speed": C},
                         noise_rms=noise_rms, adc=False, absorption_model="none", rng=np.random.default_rng(seed), sequence=sequence,
                         time_ns=sequence * 2_000_000_000)


def bank_maps(cfg, record):
    """Run the Python chain on one ping and catch its map of every hypothesis (as the chain folds them), the detector's
    map and the chain; returns ({h: fine}, fine handed to the detector, chain, report)."""
    folds, seen = {}, []
    plain_fold, plain_detect = doppler.fold, detect.multiscale

    def spy_fold(p, h, fine, best, which):
        folds[h] = np.array(fine)
        return plain_fold(p, h, fine, best, which)

    def spy_detect(*args):
        seen.append(np.array(args[0]))
        return plain_detect(*args)

    doppler.fold, detect.multiscale = spy_fold, spy_detect
    try:
        with LiveChain(cfg) as chain:
            report = chain.process(record)
    finally:
        doppler.fold, detect.multiscale = plain_fold, plain_detect
    return folds, seen[-1], chain, report


def beam_of(chain, target):
    return int(np.argmin(np.abs(chain.bearings - np.arctan2(target[0], target[1]))))


# --- the closed form ----------------------------------------------------------------------------------------------------

def test_the_mismatch_closed_form_is_the_librarys_and_its_reference_values():
    for phi in (0.1, np.pi / 4, np.pi / 2, 2.0, 4.19, 8.04, 16.76):
        u = np.linspace(-0.5, 0.5, 40001)
        mine = max(abs(np.trapezoid(np.exp(4j * phi * (u - x0) ** 2), u)) for x0 in np.linspace(0.0, 0.8, 801))
        assert core.live_sweep_mismatch(phi) == pytest.approx(mine, rel=2e-4), phi
    assert -20 * np.log10(core.live_sweep_mismatch(np.pi / 2)) == pytest.approx(0.967, abs=0.002)     # the design note's tolerance
    assert -20 * np.log10(core.live_sweep_mismatch(np.pi / 4)) == pytest.approx(0.239, abs=0.002)     # half of it: the bank's worst
    assert core.live_sweep_mismatch(0.0) == 1.0


@pytest.mark.parametrize("rate", [0.25, 0.5, 1.0, 2.0])
def test_an_ideal_sweep_loses_what_the_closed_form_says(rate):
    """A 6-10 kHz, 1 s LFM (no taper) against its echo time-scaled for `rate`, at complex baseband, 48 kHz: the matched
    filter's peak over the matched peak."""
    f0, band, T, fs, fc = 6000.0, 4000.0, 1.0, 48000.0, 8000.0
    eta = (C - rate) / (C + rate)
    t = np.arange(int(1.2 * T * fs)) / fs

    def lfm(x):
        return np.where((x >= 0) & (x <= T), np.exp(2j * np.pi * (f0 * x + 0.5 * band / T * x * x)), 0.0)

    echo, replica = lfm(eta * t) * np.exp(-2j * np.pi * fc * t), lfm(t) * np.exp(-2j * np.pi * fc * t)
    n = 1 << int(np.ceil(np.log2(2 * len(t))))
    out = np.fft.ifft(np.fft.fft(echo, n) * np.conj(np.fft.fft(replica, n)))
    loss = -20.0 * np.log10(np.abs(out).max() / np.sum(np.abs(replica) ** 2))
    assert loss == pytest.approx(closed_form_db(rate, band, T), abs=0.1)


# --- the plan: hypotheses, spacing, own ship ----------------------------------------------------------------------------

def test_hypotheses_are_spaced_by_the_sweeps_tolerance_and_none_loses_more_than_a_quarter_db():
    rec = SonarModel.load("flank_32", WINDOW).make_record(core.Pose(), environment={"sound_speed": C})
    bearings = np.radians(np.arange(-50.0, 50.5, 1.0))
    p = doppler.plan(rec, bearings, BANK)
    w = rec.transmit.waveform
    band, T = abs(w.f_stop - w.f_start), w.duration
    assert p["step"] == pytest.approx(C / (2.0 * band * T)) and p["step"] == pytest.approx(0.1875)
    K = int(np.ceil(2.5 / p["step"] - 1e-9))
    assert p["per_beam"] == 2 * K + 1 == 29
    assert p["rates"] == pytest.approx(list(p["step"] * np.arange(-K, K + 1)))        # still: every beam centred on 0
    z = p["zero"]
    assert p["rates"][z] == 0.0 and p["scales"][z] == 1.0 and p["shifts"][z] == 0.0      # exactly today's compression
    assert all(f == 0 for f in p["first"]) and all(o == 0.0 for o in p["ownship"])
    assert p["scales"] == pytest.approx([(C - r) / (C + r) for r in p["rates"]], rel=1e-15)
    factor = detect.lfm_range_doppler_factor(w)
    assert p["shifts"] == pytest.approx([2.0 * factor * r / C for r in p["rates"]], rel=1e-12)
    # Any target rate in the span is within half a step of a hypothesis: the closed form there is under 0.25 dB.
    truth = np.linspace(-2.5, 2.5, 2001)
    worst = max(np.min(np.abs(np.asarray(p["rates"]) - v)) for v in truth)
    assert worst <= 0.5 * p["step"] + 1e-12
    assert closed_form_db(worst, band, T) < 0.25
    assert p["baseband_max_range_rate"] == 0.0           # 2 x 2.6 m/s x 10 kHz / c = 35 Hz is inside the band's untapered room (400 Hz)
    assert p["threshold_db"] == pytest.approx(core.live_bank_threshold_db(29))
    assert p["pedestal_m"] == pytest.approx(29 * p["step"] * T)


def test_a_narrow_sweep_gets_the_baseband_allowance_and_hypotheses_may_be_given_as_a_count():
    rec = SonarModel.load("flank_32", WINDOW).make_record(core.Pose(), environment={"sound_speed": C})
    w = rec.transmit.waveform
    w.f_start, w.f_stop = 8000.0, 8100.0                  # 100 Hz: 10 Hz of untapered room, a 2.6 m/s shift is 35 Hz
    rec.transmit.waveform = w
    p = doppler.plan(rec, [0.0], dict(BANK, hypotheses=7, step_m_s=0.5))
    assert p["per_beam"] == 7 and p["step"] == 0.5 and p["rates"] == pytest.approx([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    assert p["baseband_max_range_rate"] == pytest.approx(1.5)        # 2 x 1.5 m/s x 8100 Hz / c = 16 Hz: more than the 10 Hz of room
    p = doppler.plan(rec, [0.0], BANK)                                # step c / (2 B T) = 7.5 m/s: one hypothesis each side
    assert p["per_beam"] == 3 and p["baseband_max_range_rate"] == pytest.approx(7.5)


def test_each_beam_is_centred_on_its_own_ship_rate_and_an_unknown_velocity_widens_the_span():
    rec = ping(own=(1.5, 0.0, 0.0), noise_rms=0.0)
    bearings = np.radians(np.arange(-50.0, 50.5, 1.0))
    p = doppler.plan(rec, bearings, BANK)
    step, K = p["step"], (p["per_beam"] - 1) // 2
    assert p["velocity_known"] and p["per_beam"] == 29
    for b, first, own in zip(bearings, p["first"], p["ownship"]):
        assert own == pytest.approx(-1.5 * np.sin(b), abs=1e-6)              # a still scatterer along the beam: closing ahead
        centre = p["rates"][first + K]
        assert abs(centre - own) <= 0.5 * step + 1e-9
    assert p["rates"][0] == pytest.approx(-(K + 6) * step) and p["rates"][-1] == pytest.approx((K + 6) * step)   # 1.5 sin 50 = 1.15 m/s: 6 steps
    s = rec.geometry
    for pose in ("tx_pose", "rx_pose"):
        q = getattr(s, pose)
        q.velocity = [float("nan"), 0.0, 0.0]
        setattr(s, pose, q)
    rec.geometry = s
    p = doppler.plan(rec, bearings, BANK)
    assert not p["velocity_known"] and p["per_beam"] == 2 * int(np.ceil((2.5 + 2.0) / step - 1e-9)) + 1 == 49
    assert all(f == 0 for f in p["first"]) and all(o == 0.0 for o in p["ownship"])


def test_the_bank_refuses_what_it_cannot_do():
    rec = SonarModel.load("flank_32", WINDOW).make_record(core.Pose(), environment={"sound_speed": C})
    w = rec.transmit.waveform
    w.type = core.WaveformType.Cw
    rec.transmit.waveform = w
    with pytest.raises(ValueError, match="not a linear sweep"):
        doppler.plan(rec, [0.0], BANK)
    with pytest.raises(ValueError, match="odd"):
        doppler.merged(dict(BANK, hypotheses=4))
    with pytest.raises(ValueError, match="mode"):
        doppler.merged({"mode": "bank"})
    for engine in ("python", "cpp"):
        with pytest.raises(ValueError, match="multiscale"):
            LiveChain(dict(CHAIN, engine=engine, cell_statistic="mean", doppler=BANK))
        with pytest.raises(ValueError, match="backprojection"):
            LiveChain(dict(CHAIN, engine=engine, beamformer="farfield", doppler=BANK))
        with pytest.raises(ValueError, match="maximum_range|max_range_rate_m_s"):
            LiveChain(dict(CHAIN, engine=engine, doppler={"mode": "sweep_bank", "max_range_rate": 2.0}))


# --- the gain ----------------------------------------------------------------------------------------------------------

def test_a_point_target_at_2_m_s_on_a_1_s_4_khz_sweep_gets_the_mismatch_back():
    """The plan's test (DOPPLER.md D4): a point target opening at 2 m/s at broadside, 6-10 kHz, 1 s. The bank's map at the
    target over the zero hypothesis' map: the closed form, 11.7 dB (the plan guessed 5 to 6 dB from the scene's study,
    whose small vehicle is not a point: its highlights and paths overlap inside the 4 m the mismatch smears each over).
    The simulator's pulse has a 5 % Tukey taper, which softens the ends where the phase error is largest: 0.3 to 0.6 dB
    less. On the map's samples (0.75 of a resolution apart) the matched peak is sampled less well than the smeared one."""
    target = (0.0, 250.0, 0.0)
    folds, fine, chain, rep = bank_maps(dict(CHAIN, engine="python", doppler=dict(BANK, report=True)), ping(target, velocity=(0.0, 2.0, 0.0)))
    b = beam_of(chain, target)
    z = chain._bank["zero"]
    zero = folds[z][b].max()
    best = max(f[b].max() for f in folds.values())
    gain = 10.0 * np.log10(best / zero)
    want = closed_form_db(2.0, 4000.0, 1.0)
    assert want == pytest.approx(11.70, abs=0.02)
    assert want - 1.2 < gain < want + 0.2, (gain, want)
    h = max(folds, key=lambda k: folds[k][b].max())
    # A coarse estimate: within one step of the truth (the grid's nearest is 2.06 m/s; the simulated echo's own scaling and
    # the map's sampling move the strongest sample by a hypothesis at most).
    assert chain._bank["rates"][h] == pytest.approx(2.0, abs=chain._bank["step"] + 1e-9)
    (d,) = [d for d in rep["detections"] if abs(d["bearing_deg"]) < 2.0]
    assert d["range_rate_m_s"] == pytest.approx(2.0, abs=chain._bank["step"] + 1e-9) and d["range_rate_step_m_s"] == chain._bank["step"]
    # The detection itself (the map's strongest sample over its background, the bank's background a maximum too): gross gain.
    with LiveChain(dict(CHAIN, engine="python")) as off:
        (d0,) = [d for d in off.process(ping(target, velocity=(0.0, 2.0, 0.0)))["detections"] if abs(d["bearing_deg"]) < 2.0]
    # Gross: the map's gain less the rise of the detector's background, a mean of maxima over 29 hypotheses (measured here
    # on the maps' noise): about 5.7 dB, which happens to be the plan's guess. The threshold is unchanged (0 dB shift).
    rise = 10.0 * np.log10(np.mean(np.maximum.reduce([f for f in folds.values()])) / np.mean(folds[z]))
    assert 4.0 < rise < 7.0
    assert d["snr_db"] - d0["snr_db"] == pytest.approx(gain - rise, abs=1.5)
    print(f"\nmap gain {gain:.2f} dB (closed form {want:.2f}); detection snr {d0['snr_db']:.2f} -> {d['snr_db']:.2f} dB, threshold shift "
          f"{chain._bank['threshold_db']:+.2f} dB, background rise {rise:.2f} dB")


def test_zero_speed_the_zero_hypothesis_is_todays_map_and_the_detection_stays_where_it_was():
    """With a still target and platform the zero hypothesis' map is the map without the bank, sample for sample, and
    every hypothesis is aligned to it (doppler.hypothesis_maps' shifts): the detection keeps its range and bearing."""
    target = (30.0, 250.0, 0.0)
    rec = ping(target)
    folds, _, chain, rep = bank_maps(dict(CHAIN, engine="python", doppler=dict(BANK, report=True)), rec.copy())
    _, fine_off, _, rep_off = bank_maps(dict(CHAIN, engine="python"), rec.copy())
    zero = folds[chain._bank["zero"]][:, :fine_off.shape[1]]
    assert np.array_equal(zero, fine_off)
    assert len(rep["detections"]) == len(rep_off["detections"]) == 1
    (a,), (b,) = rep["detections"], rep_off["detections"]
    assert a["range_m"] == pytest.approx(b["range_m"], abs=1e-9) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
    assert abs(a["range_rate_m_s"]) <= chain._bank["step"] + 1e-12          # coarse: the aligned neighbours peak nearly as high


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_one_hypothesis_is_the_chain_without_the_bank_byte_for_byte(engine):
    """max_range_rate 0: the zero hypothesis alone, a shift of 0 dB, no pedestal: the report is the one without the bank."""
    recs = [ping((30.0, 250.0, 0.0), velocity=(0.0, 1.0, 0.0), noise_rms=1e-5, seed=s, sequence=s) for s in range(2)]
    strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731
    with LiveChain(dict(CHAIN, engine=engine)) as off:
        want = [strip(off.process(r.copy())) for r in recs]
    with LiveChain(dict(CHAIN, engine=engine, doppler={"mode": "sweep_bank", "max_range_rate_m_s": 0.0})) as one:
        got = [strip(one.process(r.copy())) for r in recs]
    assert json.dumps(got) == json.dumps(want)


def test_own_ship_is_taken_out_per_beam():
    """A still target at 40 degrees seen from the platform making 1.5 m/s along the array: its echo closes at
    1.5 sin 40 = 0.96 m/s. Without the bank it loses the closed form's 8.55 dB (the simulator's tapered pulse 8.0 dB) against
    the same target seen from a still platform; with it, its beam's hypotheses are centred on that rate and it loses
    nothing but the quarter dB between hypotheses and the map's sampling of a narrower peak (under 1 dB)."""
    target = (250.0 * np.sin(np.radians(40.0)), 250.0 * np.cos(np.radians(40.0)), 0.0)
    cfg = dict(CHAIN, engine="python", doppler=dict(BANK, report=True))
    still, _, chain, _ = bank_maps(cfg, ping(target))
    moving, _, chain_m, rep = bank_maps(cfg, ping(target, own=(1.5, 0.0, 0.0)))
    b = beam_of(chain, target)
    z = chain._bank["zero"]
    reference = still[z][b].max()
    off_loss = 10.0 * np.log10(reference / moving[chain_m._bank["zero"]][b].max())
    on_loss = 10.0 * np.log10(reference / max(f[b].max() for f in moving.values()))
    want = closed_form_db(1.5 * np.sin(np.radians(40.0)), 4000.0, 1.0)
    assert want == pytest.approx(8.55, abs=0.02)
    assert want - 1.0 < off_loss < want + 0.3, (off_loss, want)
    assert abs(on_loss) < 1.0, on_loss
    (d,) = [d for d in rep["detections"] if abs(d["bearing_deg"] - 40.0) < 2.0]
    assert d["range_rate_m_s"] == pytest.approx(-1.5 * np.sin(np.radians(40.0)), abs=chain_m._bank["step"] + 1e-9)
    print(f"\nown ship 0.96 m/s along the beam: off loses {off_loss:.2f} dB (closed form {want:.2f}), the bank {on_loss:.2f} dB")


# --- the false-alarm rate ------------------------------------------------------------------------------------------------

def _noise_pings(n, seed):
    # D-256: the shipped popoto_long_range sends a 2 s hyperbolic sweep, which the bank refuses; this test is of the bank: its 1 s linear sweep is stated
    sonar = SonarModel.load("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1300.0]}}, "transmit": {"waveform": {"type": "lfm", "source": None, "duration": 1.0}}})
    scene = Scene().add_points([[0.0, 1100.0, 0.0]], target_strength_db=-300.0)
    return [simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=2e-4, rng=np.random.default_rng(seed + i), sequence=i,
                          time_ns=i * 10_000_000_000, environment={"sound_speed": C}, adc=False, absorption_model="none") for i in range(n)]


def test_the_false_alarm_rate_with_the_bank_is_held_and_the_measured_shift_restores_it():
    """Noise-only pings of the long-range array (4-6 kHz, 1 s: 17 hypotheses for 3 m/s), the detector's budget loosened to
    3 false detections a ping so that a few dozen pings count. With the chain's default shift (0 dB: libs/live
    bank_threshold_db) the bank gives no more false detections than the zero hypothesis alone (in fact far fewer: the
    maximum's background rises more than its tail), so the configured rate is held on the safe side; with the shift
    tools/doppler_bank_threshold.py measured for 17 hypotheses at this budget (-2.34 dB, docs/tasks/doppler_bank_threshold.json)
    the count is the zero hypothesis' within the counts' resolution (three Poisson sigmas of the difference)."""
    pings = _noise_pings(24, 9100)
    base = dict(CHAIN, bearings_deg=[-60.0, 60.0, 0.5], engine="cpp", cfar={"false_alarms_per_ping": 3.0})
    counts = {}
    for name, extra in (("off", {}), ("bank", {"doppler": {"mode": "sweep_bank"}}),
                        ("measured shift", {"doppler": {"mode": "sweep_bank", "threshold_db": -2.34}})):
        with LiveChain(dict(base, **extra)) as chain:
            counts[name] = sum(len(chain.process(p.copy())["detections"]) for p in pings)
            if extra:
                plan = json.loads(chain._cpp.doppler())
                assert plan["per_beam"] == 17
    off, on, shifted = counts["off"], counts["bank"], counts["measured shift"]
    print(f"\nnoise only, 24 pings at 3 a ping: off {off}, bank (0 dB) {on}, bank with the measured -2.34 dB {shifted}")
    assert off > 30
    assert on <= off + 3.0 * np.sqrt(on + off), counts
    assert abs(shifted - off) <= 3.0 * np.sqrt(shifted + off), counts
    assert core.live_bank_threshold_db(17) == 0.0 and core.live_bank_threshold_db(1) == 0.0


# --- the option ----------------------------------------------------------------------------------------------------------

def test_off_by_default_said_only_when_on_and_unknown_keys_refused():
    assert live.DEFAULTS["doppler"]["mode"] == "off"
    assert not any(k.startswith("doppler") for k in live.cpp_chain_options({}))
    assert not any(k.startswith("doppler") for k in live.cpp_chain_options({"doppler": {"mode": "off", "max_range_rate_m_s": 5.0}}))
    on = live.cpp_chain_options({"doppler": dict(BANK, report=True)})
    assert on["doppler_mode"] == "sweep_bank" and on["doppler_max_range_rate_m_s"] == 2.5 and on["doppler_report"] is True
    assert "doppler_step_m_s" not in on and "doppler_threshold_db" not in on
    with pytest.raises(ValueError, match=r"chain\.doppler\.max_rate: unknown key; did you mean 'max_range_rate_m_s'\?"):
        live.check_keys(dict(live.DEFAULTS, doppler={"mode": "sweep_bank", "max_rate": 2.0}))
    with pytest.raises(ValueError, match="doppler_mod"):
        core.LiveChainCpp({"doppler_mod": "sweep_bank"})
    with pytest.raises(ValueError, match="doppler mode"):
        core.LiveChainCpp({"doppler_mode": "bank"})
    defaults = json.loads(core.live_options_from_text(""))
    assert defaults["doppler_mode"] == "off" and defaults["doppler_report"] is False and defaults["doppler_threshold_db"] is None
    text = json.loads(core.live_options_from_text("doppler_mode=sweep_bank\ndoppler_max_range_rate_m_s=1.5\ndoppler_report=1\n"))
    assert text["doppler_mode"] == "sweep_bank" and text["doppler_max_range_rate_m_s"] == 1.5 and text["doppler_report"] is True


def test_the_detector_plan_is_as_before_without_an_offset_or_a_pedestal():
    rec = SonarModel.load("flank_32", WINDOW).make_record(core.Pose(), environment={"sound_speed": C})
    args = (rec, 400, 8, 0.125, [float(b) for b in np.radians(np.arange(-50.0, 50.5, 1.0))], 1, 120, 150, True)
    base = {"false_alarms_per_ping": 1e-3, "merge_m": 15.0}
    a = core.live_detector_plan(*args, base)
    b = core.live_detector_plan(*args, dict(base, threshold_offset_db=0.0, pedestal_m=0.0))
    c = core.live_detector_plan(*args, dict(base, threshold_offset_db=-1.5, pedestal_m=6.0))
    assert a["json"] == b["json"] and np.array_equal(a["range_envelope"], b["range_envelope"])
    for sa, sc in zip(a["scales"], c["scales"]):
        assert sc["factor"] == pytest.approx(sa["factor"] * 10 ** -0.15, rel=1e-12) and sc["pfa"] == sa["pfa"]
    ea, ec = np.asarray(a["range_envelope"]), np.asarray(c["range_envelope"])
    assert np.all(ec >= ea) and np.any(ec > ea) and np.array_equal(ec[8:], ea[8:])      # 6 m of 1 m cells: raised within 7 cells only


@pytest.mark.skipif(not core.cuda_available(), reason="no CUDA device")
@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_bank_on_a_gpu_gives_the_hosts_detections(engine):
    """Each hypothesis compressed and mapped on the device (the existing kernels in a loop): the host's detections and
    range rates, within single-precision rounding. Run by the GPU job (tools/gpu_job.sh); skipped without a device."""
    recs = [ping((0.0, 250.0, 0.0), velocity=(0.0, 2.0, 0.0), noise_rms=1e-5, seed=s, sequence=s) for s in range(2)]
    out = {}
    for backend in ("host", "cuda"):
        with LiveChain(dict(CHAIN, engine=engine, backend=backend, doppler=dict(BANK, report=True))) as chain:
            out[backend] = [chain.process(r.copy()) for r in recs]
    for h, d in zip(out["host"], out["cuda"]):
        assert len(h["detections"]) == len(d["detections"])
        for a, b in zip(h["detections"], d["detections"]):
            assert a["range_m"] == pytest.approx(b["range_m"], abs=0.05) and a["bearing_deg"] == pytest.approx(b["bearing_deg"], abs=0.05)
            assert a["snr_db"] == pytest.approx(b["snr_db"], abs=0.05) and a["range_rate_m_s"] == b["range_rate_m_s"]
