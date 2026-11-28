"""D-256 (the owner, 2026-11-28): the 12 km long-range baseline transmits a hyperbolic sweep (HFM) of 2 s at the shipped
197 dB rms across 4-6 kHz, plain matched filter, every Doppler option off. Held here: the build spec's `hfm` waveform
type and what it derives (blind range c T / 2, ping interval, energy a ping, the detector's threshold); the shipped
baseline itself, so it cannot drift back; the builds and scenes that deliberately keep a linear sweep, each by
explicit statement; the alternate build's sea-state-5 placeholder status. Ours."""

import copy
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import buildspec as bs
from gpusonar.config import load_scenario_config, load_sonar_config, load_yaml
from gpusonar.sonar import SonarModel

core = gs.core
ROOT = pathlib.Path(__file__).resolve().parents[2]
C = 1500.0                                   # the builds' nominal sound speed (buildspec, acoustics.NOMINAL_SOUND_SPEED)
HFM_2S = {"type": "sampled", "source": "hfm", "f_start": 4000.0, "f_stop": 6000.0, "duration": 2.0, "window_alpha": 0.1,
          "source_level_db_rms": 197.0}
FACES = ("popoto_face_1x32", "popoto_face_2x32", "popoto_face_3x32")      # the stages of the 12 km face: the baseline
LINEAR_BUILDS = {"popoto_static_2x32": 0.1, "popoto_static_3x32": 0.1, "popoto_testauv_1x32": 0.1, "popoto_face_3x32_ss5": 4.0}
# Scenes on a long-range sonar that keep the former 1 s linear sweep by explicit statement, and why (their files say so too).
LINEAR_SCENES = {
    "popoto_12km_large_submerged": "the sweep bank's scene (D4, D-243, HFM.8): the bank takes a linear sweep only",
    "popoto_elevation_free": "row and elevation geometry at about 1 km: inside the 2 s pulse's 1.5 km blind range",
    "popoto_elevation_shallow": "row and elevation geometry at 1 km: inside the 2 s pulse's 1.5 km blind range",
    "popoto_elevation_rehearsal": "the elevation rehearsal at about 1 km: inside the 2 s pulse's 1.5 km blind range",
}


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _with_waveform(name, waveform):
    spec = copy.deepcopy(bs.load(name))
    spec["projector"]["waveform"] = dict(waveform)
    spec["operation"]["range_m"] = [None, spec["operation"]["range_m"][1]]
    return spec


# --- the build spec's `hfm` type ------------------------------------------------------------------------------------

def test_spec_hfm_type_writes_the_sampled_hyperbolic_sweep_the_simulator_takes():
    spec = _with_waveform("popoto_face_3x32", {"type": "hfm", "duration_s": 2.0, "window_alpha": 0.1})
    assert not [i for i in bs.validate(spec) if i.level == "error"]
    w = bs.sonar_config(spec)["transmit"]["waveform"]
    assert {k: w[k] for k in HFM_2S} == HFM_2S
    model = SonarModel.from_config(bs.sonar_config(spec))
    rec = model.make_record(core.Pose(), environment={"sound_speed": C})
    assert rec.transmit.waveform.type == core.WaveformType.Arbitrary and rec.transmit.waveform.duration == pytest.approx(2.0)
    assert len(rec.transmit.waveform.replica) > 0                                         # the replica the chain compresses with
    assert bs.window_range(spec)[0] == pytest.approx(0.5 * C * 2.0) == 1500.0             # the window opens at the blind range
    bad = _with_waveform("popoto_face_3x32", {"type": "chirp", "duration_s": 2.0})
    assert any(i.level == "error" and i.where == "projector.waveform" and "hfm" in i.message for i in bs.validate(bad))
    with pytest.raises(ValueError, match="lfm, hfm, cw"):
        bs.waveform_type_keys("chirp")


def test_what_the_pulse_derives_follows_the_two_second_hyperbolic_sweep():
    """Blind range c T / 2, ping interval 2 r_max / c + T, duty, energy a ping SL + 10 log10 T, the detector's threshold;
    against the former 1 s linear sweep of the same face."""
    hfm = bs.derive(_with_waveform("popoto_face_3x32", {"type": "hfm", "duration_s": 2.0, "window_alpha": 0.1}))
    lfm = bs.derive(_with_waveform("popoto_face_3x32", {"type": "lfm", "duration_s": 1.0, "window_alpha": 0.1}))
    assert hfm["pulse_s"] == 2.0 and hfm["blind_range_m"] == pytest.approx(1500.0) and lfm["blind_range_m"] == pytest.approx(750.0)
    assert hfm["window_m"] == [1500.0, 12700.0] and lfm["window_m"] == [750.0, 12700.0]
    assert hfm["ping_interval_s"] == pytest.approx(2.0 * 12700.0 / C + 2.0)                # 18.93 s
    assert hfm["pulse_s"] / hfm["ping_interval_s"] == pytest.approx(0.1056, abs=1e-4)       # duty 10.6 % (5.6 % at 1 s)
    eq_h, g_h = bs.budget(hfm)
    eq_l, g_l = bs.budget(lfm)
    # Energy a ping: +10 log10(2) = 3.01 dB of SNR at every range, the level unchanged (197 dB rms).
    assert float(eq_h.snr(12000.0) - eq_l.snr(12000.0)) == pytest.approx(10.0 * np.log10(2.0), abs=1e-9)
    # The detector: the shorter window (1.5 to 12.7 km) needs a little less; the shape of the sweep does not enter.
    assert -0.1 < g_h["threshold_db"] - g_l["threshold_db"] <= 0.0
    same = bs.derive(_with_waveform("popoto_face_3x32", {"type": "lfm", "duration_s": 2.0, "window_alpha": 0.1}))
    assert bs.detection(hfm)["threshold_db"] == pytest.approx(bs.detection(same)["threshold_db"], abs=1e-3)    # 1.3e-4 dB apart (the template's records)


def test_a_linear_sweep_over_the_hyperbolic_base_is_stated_whole():
    """The 0.1 s builds and the alternate extend popoto_long_range, whose waveform is now sampled: their generated files
    state `type: lfm` and null the base's `source`, so they resolve to an analytic linear sweep and nothing of the HFM."""
    for name, pulse in LINEAR_BUILDS.items():
        spec = bs.load(name)
        assert spec["projector"]["waveform"]["type"] == "lfm" and spec["projector"]["waveform"]["duration_s"] == pulse, name
        sonar = bs.names(spec)["sonar"]
        text = (ROOT / "configs" / "sonars" / f"{sonar}.yaml").read_text()
        assert "type: lfm" in text and "source: null" in text, sonar
        w = load_sonar_config(sonar)["transmit"]["waveform"]
        assert w["type"] == "lfm" and w.get("source") is None and w["duration"] == pulse, sonar
        rec = SonarModel.load(sonar).make_record(core.Pose())
        assert rec.transmit.waveform.type == core.WaveformType.Lfm, sonar


# --- the shipped baseline -------------------------------------------------------------------------------------------

def test_the_shipped_long_range_sonar_is_a_two_second_hyperbolic_sweep_at_197_db():
    """D-256: so that it cannot drift back to the 1 s linear sweep without this test saying so."""
    w = load_sonar_config("popoto_long_range")["transmit"]["waveform"]
    assert {k: w[k] for k in HFM_2S} == HFM_2S
    for name in FACES:
        spec = bs.load(name)
        assert spec["projector"]["waveform"] == {"type": "hfm", "duration_s": 2.0, "window_alpha": 0.1}, name
        assert spec["projector"]["source_level_db_rms"] == 197.0 and spec["projector"]["band_hz"] == [4000.0, 6000.0], name
        sonar = bs.names(spec)["sonar"]
        for variant in (sonar, sonar + "_clocks"):
            got = load_sonar_config(variant)["transmit"]["waveform"]
            assert {k: got[k] for k in HFM_2S} == HFM_2S, variant
            assert load_sonar_config(variant)["receive"]["window"]["range"][0] == 1500.0, variant
        rec = SonarModel.load(sonar).make_record(core.Pose())
        assert rec.transmit.waveform.type == core.WaveformType.Arbitrary and rec.transmit.waveform.duration == pytest.approx(2.0)
    # Every Doppler option is off in the baseline's deployment (D-244, D-256): the plain matched filter.
    chain = load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle_elevation.yaml")["chain"]
    assert "doppler" not in chain and chain.get("range_doppler", "off") == "off"


def test_scenes_that_keep_the_linear_sweep_state_it_and_say_why():
    for name, why in LINEAR_SCENES.items():
        text = (ROOT / "configs" / "scenarios" / f"{name}.yaml").read_text()
        cfg = load_scenario_config(name)
        stated = (cfg.get("sonar_overrides") or {}).get("transmit", {}).get("waveform", {})
        assert stated.get("type") == "lfm" and stated.get("duration") == 1.0 and "source" in stated and stated["source"] is None, name
        assert "D-256" in text, name                                                     # the file says why
        w = load_sonar_config(cfg["sonar"], cfg["sonar_overrides"])["transmit"]["waveform"]
        assert w["type"] == "lfm" and w["duration"] == 1.0 and w["source_level_db_rms"] == 197.0, name


def test_no_long_range_scene_puts_a_target_inside_its_pulse():
    """A target inside c T / 2 of the platform is in the blind range of the pulse the scene transmits: listed (with the
    pulse it would need) rather than lost. Every scene on a 4-6 kHz long-range sonar is checked."""
    from gpusonar.config import list_configs
    inside = []
    for name in list_configs("scenarios"):
        cfg = load_scenario_config(name)
        if not isinstance(cfg.get("sonar"), str) or not cfg["sonar"].startswith(("popoto_long_range", "popoto_face_1row")):
            continue
        sonar_cfg = load_sonar_config(cfg["sonar"], cfg.get("sonar_overrides"))
        w = sonar_cfg["transmit"]["waveform"]
        blind = 0.5 * C * float(w["duration"])
        if sonar_cfg.get("short_range"):
            # The short-range mode (docs/design/short_range_mode.md): its short pings see inside the long pulse's blind
            # range, so a target is lost only inside the short pulse's (tests/python/test_short_range_mode.py shows both).
            short = (sonar_cfg["short_range"].get("waveform") or {}).get("duration", w["duration"])
            blind = 0.5 * C * min(float(w["duration"]), float(short))
        p = np.asarray(cfg.get("platform", {}).get("position", [0.0, 0.0, 0.0]), dtype=float)
        for item in cfg.get("scene") or []:
            for q in item.get("positions") or ([item["position"]] if item.get("position") else []):
                if np.linalg.norm(np.asarray(q, dtype=float)[:2] - p[:2]) < blind:
                    inside.append((name, item.get("label")))
    assert inside == []


def test_the_alternate_is_a_placeholder_that_needs_a_larger_transmitter():
    """D-256: sea state 5 needs a larger transmitter not yet identified; the alternate build is not claimed as met."""
    text = (ROOT / "configs" / "builds" / "popoto_face_3x32_ss5.yaml").read_text()
    assert "needs a larger transmitter, not yet identified" in text and "placeholder" in text
    assert "needs a larger transmitter, not yet identified" in (ROOT / "docs" / "DESIGN.md").read_text()


# --- the deep-site acceptance reads the baseline from its own config -------------------------------------------------

def test_deep12_baseline_case_is_the_shipped_sonar_and_the_former_baseline_a_named_comparison():
    d12 = tool("deep12_eval")
    k = d12.parse_case("ss3_baseline")
    assert k["wave"] == "hfm" and k["pulse_s"] == 2.0 and k["source_level_db"] == 197.0 and k["sets"] == [] and k["sea_state"] == 3
    assert d12.ACCEPTANCE["baseline"][0] == "ss3_baseline"
    assert d12.ACCEPTANCE["former_baseline"][0] == "ss3_lfm1"
    # A linear case now states its sweep whole (type and length), since the shipped sonar is no longer linear.
    assert d12.parse_case("ss3_lfm1")["sets"] == ["sonar_overrides.transmit.waveform.type=lfm", "sonar_overrides.transmit.waveform.source=null",
                                                  "sonar_overrides.transmit.waveform.duration=1.0"]


# --- the live chain on the hyperbolic record: both engines, and the range-Doppler correction refused -----------------

def _small_hfm_record():
    """An empty record of the shipped long-range sonar (the 2 s hyperbolic sweep), at baseband over a short window."""
    sonar = SonarModel.load("popoto_long_range", {"receive": {"kind": "baseband", "sample_rate": 8.0e+3, "window": {"range": [11300.0, 11600.0]}}})
    rec = sonar.make_record(core.Pose())
    assert rec.transmit.waveform.type == core.WaveformType.Arbitrary
    return rec


@pytest.mark.parametrize("engine", ["python", "cpp"])
@pytest.mark.parametrize("mode", ["ownship", "full"])
def test_both_engines_refuse_the_linear_sweeps_range_doppler_correction_on_a_sampled_pulse(engine, mode):
    """The correction is f0 T / B x the closing speed, a linear sweep's coupling; on the hyperbolic record it was 0, silently
    (detect.lfm_range_doppler_factor returns 0 for an Arbitrary pulse): a correction asked for and not made. Refused, both
    engines, the same words; off processes as before. Fails before D-256's package (no error was raised)."""
    from gpusonar.live import LiveChain
    rec = _small_hfm_record()
    with LiveChain({"engine": engine, "range_doppler": mode}) as chain:
        with pytest.raises(ValueError, match=f"range_doppler {mode}: the record's pulse is a sampled waveform"):
            chain.process(rec.copy())
    with LiveChain({"engine": engine}) as chain:
        chain.process(rec.copy())
    # A linear sweep is corrected as before (its factor f0 T / B = 2.5 s at 4-6 kHz, 1 s).
    lfm = SonarModel.load("popoto_long_range",
                          {"transmit": {"waveform": {"type": "lfm", "source": None, "duration": 1.0}},
                           "receive": {"kind": "baseband", "sample_rate": 8.0e+3, "window": {"range": [11300.0, 11600.0]}}}).make_record(core.Pose())
    from gpusonar import detect
    assert detect.lfm_range_doppler_factor(lfm.transmit.waveform) == pytest.approx(2.5)
    with LiveChain({"engine": engine, "range_doppler": mode}) as chain:
        chain.process(lfm.copy())


@pytest.mark.skipif(__import__("gpusonar.sim", fromlist=["find_bellhop"]).find_bellhop() is None, reason="Bellhop not installed (GPUSONAR_BELLHOP)")
def test_both_engines_compress_the_shipped_hyperbolic_record_alike():
    """popoto_12km with the shipped sonar (2 s hyperbolic sweep), its eigenrays thinned to 16 arrivals and 2 pings (the
    gallery's quick form; about 3 min on one CPU core): the two engines give the same detections and tracks, and the ship
    (five points over 120 m about 800 m across, 12 km out) is detected on every ping. Free field would not do: without the
    surface the ship at 4 m sits at the threshold (measured: 0 and 1 detections at 12.0 dB)."""
    import warnings
    from gpusonar.live import LiveChain
    from gpusonar.sim import run_scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_scenario("popoto_12km", {"pings": {"count": 2}, "propagation": {"max_arrivals": 16}})
    assert res.records[0].transmit.waveform.type == core.WaveformType.Arbitrary and res.records[0].transmit.waveform.duration == pytest.approx(2.0)
    out = {}
    for engine in ("python", "cpp"):
        with LiveChain({"engine": engine}) as chain:
            out[engine] = chain.run([r.copy() for r in res.records])
    for a, b in zip(out["python"], out["cpp"]):
        assert len(a["detections"]) == len(b["detections"]) >= 1
        for x, y in zip(a["detections"], b["detections"]):
            assert x["range_m"] == pytest.approx(y["range_m"], abs=0.05) and x["bearing_deg"] == pytest.approx(y["bearing_deg"], abs=0.05)
            assert x["snr_db"] == pytest.approx(y["snr_db"], abs=0.01)
        assert [(t["id"], t["kind"]) for t in a["tracks"]] == [(t["id"], t["kind"]) for t in b["tracks"]]
        assert any(11950.0 < d["range_m"] < 12100.0 for d in b["detections"])
