"""The tone path's notch about the own-motion ridge seen through each beam's grating lobes (libs/live doppler.hpp
plan_tone step 4b, `chain.doppler.tone_lobe_notch: auto | off`; docs/tasks/DOPPLER.md, "Grating-lobe notch"), both
engines (one C++ function, tone_stage, and the chain's merge written twice).

The defect it removes (DOPPLER.md "Large submerged at 12 km", LS.1 item 0, measured on popoto_12km_large_submerged):
with the tone above the sweep's band the long-range array's 0.15 m pitch is 0.65 of a wavelength at 6.5 kHz, a beam
steered beyond about 32 degrees has a grating lobe near endfire, the still water seen through it lies 2.31 m/s (1.5 m/s
along the line x lambda / d) from the beam's own-ship rate, and the tone path tested that ridge as a target.

What is held here, on records made here (the long-range array, a 1 s Hann tone at 6.5 kHz, complex baseband at 8 kHz,
the platform at 1.5 m/s along the line):
- the lobes' directions from the array's geometry and the tone's wavelength (the arithmetic itself: GoogleTest
  DopplerToneLobes);
- (a) a field of still scatterers all round: false tone detections beyond 32 degrees with the lobe notch off, none with
  it (the default with the tone path): fails on the code before it;
- (b) a mover outside every notch on a beam that has a lobe: reported with and without the lobe notch, at the same rate;
- (c) a mover whose rate lies in a lobe's notch: the tone does not report it with the lobe notch (the documented cost),
  the sweep pings of an alternating pair still do;
- (d) where no lobe adds a notch (no visible lobe; or a still platform, whose lobes' ridges are the beam's own) the
  stage and the chain are the code before it bit for bit;
- the option: auto by default, said only when off, checked in both engines.
The engines' equality on (a) is in test_live_cpp.py; the device against the host in test_doppler_device.py.
"""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import doppler, live
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
F = 6.5e+3
V = 1.5                                   # m/s along the line (+x), as popoto_12km's platform
PITCH = 0.15                              # popoto_long_range (configs/sonars/popoto_long_range.yaml)
CW = {"type": "cw", "f_start": F, "f_stop": F, "duration": 1.0, "window_alpha": 1.0}
LFM = {"type": "lfm", "f_start": 4.0e+3, "f_stop": 6.0e+3, "duration": 1.0, "window_alpha": 0.05}
BEARINGS_DEG = [-40.0, 40.0, 1.0]
TONE = {"tone": "bank", "report": True}
OFF = dict(TONE, tone_lobe_notch="off")
strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731


def sonar(wave, sonar_name="popoto_long_range", window=(300.0, 2300.0)):
    over = {"receive": {"kind": "baseband", "sample_rate": 8.0e+3, "window": {"range": list(window)}}, "transmit": {"waveform": wave}}
    if sonar_name == "flank_32":
        over["receive"] = {"window": {"range": list(window)}}
    return SonarModel.load(sonar_name, over)


def ping(wave=CW, field=0, movers=(), own=(V, 0.0, 0.0), seed=5, sequence=0, sonar_name="popoto_long_range", ts_field=-35.0):
    """A ping with `field` still point scatterers all round (400 to 1400 m, the horizontal plane: the water behind every
    lobe) and point movers [(bearing deg, range m, radial speed over ground m/s, opening positive)] at -20 dB.
    The field at -35 dB each: on 6 seeds (ours, while writing this) 15 false tone detections without the lobe notch and
    0 with it, 0 at 5 kHz. At -30 dB 15 -> 2 and at -25 dB (4 seeds) 11 -> 5, with 0 and 2 at 5 kHz where no lobe is:
    a stronger field reaches every beam through its far sidelobes (-40 to -60 dB) at every rate within +-|v|, which no
    notch about a ridge removes (DOPPLER.md "Grating-lobe notch", what is left)."""
    scene = Scene()
    rng = np.random.default_rng(seed)
    if field:
        r = np.sqrt(rng.uniform(400.0 ** 2, 1400.0 ** 2, field))
        a = rng.uniform(-np.pi, np.pi, field)
        scene.add_points(np.stack([r * np.sin(a), r * np.cos(a), np.zeros(field)], axis=1), velocity=(0.0, 0.0, 0.0), target_strength_db=ts_field)
    for bearing, rng_m, speed in movers:
        u = np.array([np.sin(np.radians(bearing)), np.cos(np.radians(bearing)), 0.0])
        scene.add_points([list(rng_m * u)], velocity=tuple(speed * u), target_strength_db=-20.0)
    rec = simulate_ping(sonar(wave, sonar_name), make_pose(position=(0.0, 0.0, 0.0), velocity=own), scene, environment={"sound_speed": C},
                        noise_rms=1e-5, adc=False, absorption_model="none", rng=np.random.default_rng(seed + 1), sequence=sequence,
                        time_ns=sequence * 2_000_000_000)
    if not rec.samples.is_complex():
        core.to_baseband(rec, max_range_rate=6.0)
    return rec


def bearings():
    first, last, step = BEARINGS_DEG
    return [float(b) for b in np.radians(np.arange(first, last + 0.5 * step, step))]


def stage(rec, cfg, keep_map=False):
    return core.live_tone_stage(rec, bearings(), doppler.tone_options(cfg), hann=True, keep_map=keep_map)


def run(cfg, recs):
    with LiveChain(cfg) as chain:
        return [strip(chain.process(r.copy())) for r in recs]


def chain_cfg(engine, tone):
    return {"engine": engine, "backend": "host", "bearings_deg": BEARINGS_DEG, "doppler": tone}


def ridge_offset():
    """The radial speed over ground of a mover on the first lobe's ridge of a beam at a negative bearing: -V lambda / d."""
    return -V * (C / F) / PITCH


@pytest.fixture(scope="module")
def field_ping():
    return ping(field=2000)


# --- the lobes ----------------------------------------------------------------------------------------------------------

def test_the_plan_takes_the_lobes_from_the_pitch_and_the_wavelength():
    """popoto_long_range at 6.5 kHz: pitch 0.15 m from the hydrophones; every beam's visible lobes are
    asin(sin b + m lambda / d) for the m with |sin| <= 1 (the arithmetic's edge cases are GoogleTest's): from 32.6
    degrees; each lobe's ridge is -v . u_g (V lambda / d = 2.31 m/s from the beam's) and its notch the beam's rule at g.
    The endfire rule: a beam whose response towards endfire (the Hann aperture's array factor, computed here) is at
    least -40 dB of its main lobe's also has the endfire ridge notched (from 24 degrees here: the skirt of a lobe centred
    beyond endfire). At 5 kHz (half a wavelength) nothing within +-40 degrees."""
    from gpusonar import flank
    rec = ping()
    p = stage(rec, TONE)["plan"]
    lam = C / F
    assert p["lobe_notch"] == "auto" and p["pitch"] == pytest.approx(PITCH, rel=1e-6)
    b = np.asarray(bearings())
    w = np.asarray(flank.aperture_weights(rec, "hann"), dtype=float)
    x = np.arange(len(w)) * PITCH
    resp = lambda s, s0: 20 * np.log10(abs(np.sum(w * np.exp(2j * np.pi * x * (s - s0) / lam))) / w.sum())  # noqa: E731
    by_beam = {}
    for g in p["lobes"]:
        by_beam.setdefault(g["beam"], []).append(g)
    for i, bi in enumerate(b):
        lobes = [g for g in by_beam.get(i, []) if not g["endfire"]]
        ends = [g for g in by_beam.get(i, []) if g["endfire"]]
        want = [np.arcsin(np.sin(bi) + m * lam / PITCH) for m in (-2, -1, 1, 2) if abs(np.sin(bi) + m * lam / PITCH) <= 1]
        assert [g["direction"] for g in lobes] == pytest.approx(want, abs=1e-12), np.degrees(bi)
        want_ends = [side * np.pi / 2 for side in (-1, 1) if resp(side, np.sin(bi)) >= -40.0]
        assert [g["direction"] for g in ends] == pytest.approx(want_ends, abs=1e-12), np.degrees(bi)
        for g in ends:
            assert g["response_db"] == pytest.approx(resp(np.sin(g["direction"]), np.sin(bi)), abs=1e-6)
        for g in lobes + ends:
            assert g["ownship"] == pytest.approx(-V * np.sin(g["direction"]), abs=1e-9)
            assert g["notch"] >= 3.0 * p["resolution"]
        for g in lobes:
            assert abs(g["ownship"] - p["ownship"][i]) == pytest.approx(V * lam / PITCH, abs=1e-9)
            assert abs(np.degrees(bi)) >= 32.5
    assert min(abs(np.degrees(b[list(by_beam)]))) == pytest.approx(24.0)
    assert p["lobe_beams"] == len(by_beam) and all(p["lobe_hidden"][i] > 0.0 for i in by_beam)
    assert all(h == 0.0 for i, h in enumerate(p["lobe_hidden"]) if i not in by_beam)
    assert '"lobe_beams": ' in p["json"]
    low = stage(ping(wave=dict(CW, f_start=5.0e+3, f_stop=5.0e+3)), TONE)["plan"]
    assert low["lobes"] == [] and low["lobe_beams"] == 0 and "lobe" not in low["json"]


# --- (a) the defect and the remedy --------------------------------------------------------------------------------------

def test_the_lobe_ridge_is_no_false_detection_with_the_lobe_notch(field_ping):
    """(a) The still field all round, the platform under way: without the lobe notch the tone stage reports the water
    behind the lobes as movers beyond 31 degrees at about the lobe's ridge (the defect: on the code before this package
    the default tone path did the same); with it (the default with the tone path) nothing. The bank is the same: only the
    cells tested and their backgrounds change."""
    off, on = stage(field_ping, OFF, keep_map=True), stage(field_ping, TONE, keep_map=True)
    assert np.array_equal(off["power"], on["power"])
    assert len(off["detections"]) >= 1, "the defect did not show: the test record no longer makes it"
    p = off["plan"]
    lam = C / F
    for d in off["detections"]:
        assert abs(np.degrees(d["bearing"])) > 31.0, d
        b = d["bearing"]
        lobe_rate = -V * np.clip(np.sin(b) - np.sign(b) * lam / PITCH, -1, 1)
        assert d["range_rate"] == pytest.approx(lobe_rate, abs=0.4), d       # the lobe's ridge, within its notch
        assert abs(d["range_rate"] - p["ownship"][d["beam"]]) > 1.5
    assert on["detections"] == []
    assert on["tests"] < off["tests"]


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_chain_reports_no_lobe_ridge_with_the_tone_path(engine, field_ping):
    (off,) = run(chain_cfg(engine, OFF), [field_ping])
    (on,) = run(chain_cfg(engine, TONE), [field_ping])
    assert len([d for d in off["detections"] if d["doppler_path"] == "tone"]) >= 1
    assert on["detections"] == []


# --- (b) and (c): what it keeps and what it costs -----------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_mover_outside_every_notch_on_a_lobed_beam_keeps_its_detection_and_rate(engine):
    """(b) A mover at -36 degrees (a beam with a lobe) opening at 1.0 m/s over ground: 1.0 m/s from its beam's ridge
    (notch +-0.38 m/s) and 3.3 m/s from its lobe's: reported by the tone with and without the lobe notch, at the same
    range rate (the bank and the parabola are the same; only backgrounds of cells near the lobe's notch move)."""
    rec = ping(field=0, movers=((-36.0, 900.0, 1.0),))
    (off,) = run(chain_cfg(engine, OFF), [rec])
    (on,) = run(chain_cfg(engine, TONE), [rec])
    near = lambda rep: [d for d in rep["detections"] if abs(d["bearing_deg"] + 36.0) < 2.0 and abs(d["range_m"] - 900.0) < 0.25 * C]  # noqa: E731
    a, b = near(off), near(on)
    assert len(a) == 1 and len(b) == 1, (off["detections"], on["detections"])
    own = V * np.sin(np.radians(36.0))                       # -v . u_b at -36 degrees
    assert b[0]["range_rate_m_s"] == pytest.approx(own + 1.0, abs=0.5 * b[0]["range_rate_step_m_s"])
    assert b[0]["range_rate_m_s"] == pytest.approx(a[0]["range_rate_m_s"], abs=1e-9)
    assert b[0]["bearing_deg"] == pytest.approx(a[0]["bearing_deg"], abs=1e-9) and b[0]["range_m"] == pytest.approx(a[0]["range_m"], abs=1e-9)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_a_mover_in_a_lobes_notch_is_left_to_the_sweep(engine):
    """(c) The cost: a mover at -36 degrees closing at V lambda / d = 2.31 m/s over ground sits on its beam's lobe ridge.
    With the lobe notch the tone does not report it (without, it does); alternating pings: the sweep ping (4-6 kHz, no
    lobe within +-40 degrees) reports it either way."""
    s = ridge_offset()
    recs = [ping(wave=CW if i == 0 else LFM, movers=((-36.0, 900.0 + s * 2.0 * i, s),), seed=7 + i, sequence=i) for i in range(2)]
    near = lambda rep: [d for d in rep["detections"] if abs(d["bearing_deg"] + 36.0) < 2.0 and abs(d["range_m"] - 900.0) < 0.25 * C]  # noqa: E731
    off = run(chain_cfg(engine, OFF), recs)
    on = run(chain_cfg(engine, TONE), recs)
    assert [d["doppler_path"] for d in near(off[0])] == ["tone"]
    assert near(on[0]) == []
    for rep in (off[1], on[1]):
        got = near(rep)
        assert len(got) == 1 and got[0]["doppler_path"] == "sweep", rep["detections"]


# --- (d) no lobe added: the code before it, bit for bit -----------------------------------------------------------------

@pytest.mark.parametrize("case", ["no_visible_lobe", "still_platform"])
def test_where_no_lobe_adds_a_notch_the_stage_and_the_chain_are_unchanged(case):
    """(d) The long-range array at 5 kHz under way (pitch half a wavelength: no lobe within +-40 degrees), and the flank
    array at 10.5 kHz on a still platform (lobes beyond 30 degrees, each ridge at 0, inside the beam's own notch): the
    lobe notch adds no cell, and auto gives off's power, ratio, detections and report lines bit for bit (off is the code
    before this package: tone_notch_mask is its loop when no lobe is listed)."""
    movers = ((0.0, 700.0, 1.0), (-35.0, 900.0, -1.2))
    if case == "no_visible_lobe":
        recs = [ping(wave=dict(CW, f_start=5.0e+3, f_stop=5.0e+3), field=300, movers=movers, seed=11 + i, sequence=i) for i in range(2)]
    else:
        recs = [ping(wave=dict(CW, f_start=10.5e+3, f_stop=10.5e+3, duration=0.5), field=300, movers=movers, own=(0.0, 0.0, 0.0),
                     seed=11 + i, sequence=i, sonar_name="flank_32") for i in range(2)]
    a, b = stage(recs[0], OFF, keep_map=True), stage(recs[0], TONE, keep_map=True)
    assert b["plan"]["lobe_beams"] == 0 and all(h == 0.0 for h in b["plan"]["lobe_hidden"])
    assert (b["plan"]["lobes"] == []) == (case == "no_visible_lobe")
    assert np.array_equal(a["power"], b["power"]) and np.array_equal(a["ratio"], b["ratio"])
    assert a["detections"] == b["detections"] and a["tests"] == b["tests"] and a["factor"] == b["factor"]
    for engine in ("python", "cpp"):
        assert run(chain_cfg(engine, TONE), recs) == run(chain_cfg(engine, OFF), recs)


# --- the option ---------------------------------------------------------------------------------------------------------

def test_the_option_is_auto_said_only_when_off_and_checked():
    assert doppler.DEFAULTS["tone_lobe_notch"] == "auto" and live.DEFAULTS["doppler"]["tone_lobe_notch"] == "auto"
    assert "doppler_tone_lobe_notch" not in doppler.tone_options(TONE)
    assert doppler.tone_options(OFF)["doppler_tone_lobe_notch"] == "off"
    assert not any(k.startswith("doppler") for k in live.cpp_chain_options({}))
    with pytest.raises(ValueError, match="tone_lobe_notch"):
        doppler.merged({"tone": "bank", "tone_lobe_notch": "on"})
    with pytest.raises(ValueError, match="tone_lobe_notch"):
        core.LiveChainCpp({"doppler_tone": "bank", "doppler_tone_lobe_notch": "on"})
    assert "doppler_tone_lobe_notch" not in json.loads(core.live_options_from_text("doppler_tone=bank\n"))
    assert json.loads(core.live_options_from_text("doppler_tone=bank\ndoppler_tone_lobe_notch=off\n"))["doppler_tone_lobe_notch"] == "off"
    p = stage(ping(), OFF)["plan"]
    assert p["lobe_notch"] == "off" and p["lobes"] == [] and p["lobe_beams"] == 0 and '"lobe_notch": "off"' in p["json"]


# --- the invariant: no reported tone detection inside a notch applied to its beam ---------------------------------------

TRAIN = {"type": "sampled", "source": "train", "f_start": 4.0e+3, "f_stop": 6.6e+3,
         "segments": [{"kind": "cw", "frequency": F, "duration": 1.0, "window_alpha": 1.0}, {"kind": "gap", "duration": 0.0},
                      {"kind": "lfm", "f_start": 4.0e+3, "f_stop": 6.0e+3, "duration": 1.0, "window_alpha": 0.1}]}
TONE_TRAIN = dict(TONE, tone_frequency_hz=F, tone_start_s=0.0, tone_duration_s=1.0, tone_window_alpha=1.0)


def reverb_train_ping():
    """popoto_12km_large_submerged_tone's train (a 6.5 kHz tone, then the 4-6 kHz sweep; the record's centre 5.3 kHz)
    over a flat bottom 50 m below, the simulator's boundary reverberation, under way at 1.5 m/s. That reverberation
    shifts each sector by its Doppler at the record's CENTRE frequency (sim/reverb.py), so the tone sees the still water
    at 5.3 / 6.5 of its physical ridge, where the lobe notch's edge cuts it: the case of the GPU run of 2026-11-27,
    many detections at the notch's edges (DOPPLER.md GL.8)."""
    s = SonarModel.load("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": 8.0e+3, "window": {"range": [300.0, 1500.0]}},
                                              "transmit": {"waveform": TRAIN}})
    return simulate_ping(s, make_pose(position=(0.0, 0.0, 50.0), velocity=(V, 0.0, 0.0)), Scene(), environment={"sound_speed": C}, adc=False,
                         noise_rms=1e-7, absorption_model="none", reverberation=[{"mu_db": -27.0, "boundary_z": 0.0}], rng=np.random.default_rng(3))


def applied_mask(plan, beam):
    """The (ridge, half-width) pairs of every notch the plan applies to `beam`: its own and its lobes'."""
    return [(plan["ownship"][beam], plan["notch"][beam])] + [(g["ownship"], g["notch"]) for g in plan["lobes"] if g["beam"] == beam]


def assert_outside_applied_notches(rec, tone, bearings_deg, engines=("python", "cpp"), batched=("off", "on"), backend="host"):
    first, last, step = bearings_deg
    b = [float(x) for x in np.radians(np.arange(first, last + 0.5 * step, step))]
    for form in batched:
        st = core.live_tone_stage(rec, b, doppler.tone_options(dict(tone, batched=form)), hann=True,
                                  **({} if backend == "host" else {"backend": core.Backend.Cuda}))
        p = st["plan"]
        for d in st["detections"]:                     # the strongest cell itself: never inside, exactly
            r = p["rates"][d["hypothesis"]]
            assert all(abs(r - ridge) > half for ridge, half in applied_mask(p, d["beam"])), (form, d)
            # Its refined rate: at most half a step further in (the parabola's reach).
            assert all(abs(d["range_rate"] - ridge) > half - 0.5 * p["step"] - 1e-9 for ridge, half in applied_mask(p, d["beam"])), (form, d)
        for engine in engines:
            cfg = {"engine": engine, "backend": backend, "bearings_deg": list(bearings_deg), "doppler": dict(tone, batched=form)}
            (rep,) = run(cfg, [rec])
            for d in rep["detections"]:
                if d["doppler_path"] not in ("tone", "both"):
                    continue
                # The beam: the nearest, or either of two when the refined bearing lies half a step between them.
                near = np.nonzero(np.abs(np.degrees(b) - d["bearing_deg"]) <= 0.5 * step + 1e-6)[0]
                assert any(all(abs(d["range_rate_m_s"] - ridge) > half - 0.5 * p["step"] - 1e-9 for ridge, half in applied_mask(p, int(k)))
                           for k in near), (engine, form, d)
    return st


@pytest.mark.parametrize("case", ["field", "reverb_train"])
def test_no_tone_detection_lies_inside_a_notch_applied_to_its_beam(case):
    """Both engines, the loop and the batched form on the host, the lobe notch auto: the strongest cell of every tone
    detection is outside every notch applied to its beam (its own and its lobes'), and its refined rate at most half a
    step inside. On the reverberation train the stage does report detections (at the notches' edges): the invariant is
    tested where it matters."""
    if case == "field":
        rec, tone = ping(field=2000, movers=((-36.0, 900.0, 1.0),)), TONE
    else:
        rec, tone = reverb_train_ping(), TONE_TRAIN
    st = assert_outside_applied_notches(rec, tone, [-40.0, 40.0, 0.5])
    assert len(st["detections"]) >= 1
