"""a-engrec: every simulated record says which simulator engine ran each stage that has an engine choice (the owner,
2026-11-26: "yes record the engine in every record"), in the record's free-form metadata, not in the schema:

    extra_json {"gpusonar": {"engine": {"eigenrays": ..., "pairing": ..., "synthesis": ..., "reverberation": ...}}}

- eigenrays: "cpp" | "numpy" (free_field, image_source, sediment), "bellhop" (eigenray), "bellhop3d" (eigenray3d);
- pairing (two_way_paths): "cpp" | "numpy";
- synthesis (the two-way echoes): "cpp_host" | "cpp_cuda" | "numpy" (a sampled pulse on the NumPy engine; analytic
  pulses have the C++ kernel only);
- reverberation (sim/reverb.py): "cpp_host" | "cpp_cuda" | "numpy".
A stage that did not run in the ping is absent; a stage that ran on two engines in one ping lists both ("cpp+numpy").
The value is what ran, a fallback included. The older key `echo_synthesis: numpy_sampled` stays as it was.

Before this package only the sampled-pulse note existed, so every test here asking for `engine` failed. Nothing else
in a record moves (the sha256 proof is in the package's report; `test_the_note_changes_nothing_else` holds it here)."""

import json
import warnings

import numpy as np
import pytest

from gpusonar import _gpusonar as core
from gpusonar.live import LiveChain
from gpusonar.sim import FreeFieldProvider, Scene, engine, make_pose, run_scenario, simulate_ping
from gpusonar.sim import waveform as wfm
from gpusonar.sim.bellhop import find_bellhop

from test_sampled_waveform import C, IMAGE, TIERS, _five_targets, _scenario, sonar_with

needs_cpp = pytest.mark.skipif(not engine.HAVE_CPP, reason="needs the C++ simulator in the extension module")
LFM = {"type": "lfm"}


def note(record):
    return json.loads(record.metadata.extra_json or "{}").get("gpusonar", {}).get("engine")


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("GPUSONAR_SIM_ENGINE", raising=False)
    monkeypatch.setattr(engine, "_warned_default", False, raising=False)
    monkeypatch.setattr(wfm, "_warned", False)


@pytest.fixture
def no_cpp(clean_env, monkeypatch):
    """The module as if built without the C++ simulator."""
    monkeypatch.setattr(engine, "HAVE_CPP", False)


def expected(eng, tier, reverb=False):
    want = {"eigenrays": {"eigenray": "bellhop"}.get(tier, eng), "pairing": eng, "synthesis": "cpp_host"}
    if reverb:
        want["reverberation"] = "cpp_host" if eng == "cpp" else "numpy"
    return want


# --- each tier, each engine ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("eng", [pytest.param("cpp", marks=needs_cpp), "numpy"])
@pytest.mark.parametrize("tier", ["free_field", "image_source", "image_source + reverberation", "sediment"])
def test_each_tier_names_the_engine_that_ran(clean_env, tier, eng):
    rec = run_scenario(_scenario(tier.split(" ")[0], LFM, **TIERS[tier]), engine=eng).records[0]
    assert note(rec) == expected(eng, tier.split(" ")[0], reverb="reverberation" in tier)
    assert "echo_synthesis" not in json.loads(rec.metadata.extra_json)["gpusonar"]      # analytic pulse: no old note


@pytest.mark.skipif(find_bellhop() is None, reason="needs Bellhop (GPUSONAR_BELLHOP)")
@pytest.mark.parametrize("eng", [pytest.param("cpp", marks=needs_cpp), "numpy"])
def test_the_eigenray_tier_names_bellhop(clean_env, eng):
    rec = run_scenario(_scenario("eigenray", LFM, **TIERS["eigenray"]), engine=eng).records[0]
    assert note(rec) == expected(eng, "eigenray")


# --- the default, the environment variable, and the fallback -------------------------------------------------------

@needs_cpp
def test_the_default_with_the_module_is_cpp(clean_env):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec = run_scenario(_scenario("image_source", LFM, **TIERS["image_source + reverberation"])).records[0]
    assert note(rec) == expected("cpp", "image_source", reverb=True)


def test_the_environment_variable_numpy_is_recorded(clean_env, monkeypatch):
    monkeypatch.setenv("GPUSONAR_SIM_ENGINE", "numpy")
    rec = run_scenario(_scenario("image_source", LFM, **TIERS["image_source + reverberation"])).records[0]
    assert note(rec) == expected("numpy", "image_source", reverb=True)


def test_the_default_without_the_cpp_simulator_records_numpy(no_cpp):
    with pytest.warns(RuntimeWarning, match=r"built without the C\+\+ simulator"):
        rec = run_scenario(_scenario("image_source", LFM, **TIERS["image_source + reverberation"])).records[0]
    assert note(rec) == expected("numpy", "image_source", reverb=True)


def test_a_sampled_pulse_without_the_cpp_simulator_says_numpy_twice(no_cpp):
    wave = {"type": "sampled", "source": "lfm", "duration": 0.002, "window_alpha": 0.2}
    sonar = sonar_with("sas_auv", wave, 18.0, 22.0, omni=True)
    with pytest.warns(RuntimeWarning, match="NumPy engine"):
        rec = simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False)
    extra = json.loads(rec.metadata.extra_json)["gpusonar"]
    assert extra["echo_synthesis"] == "numpy_sampled"                         # the old key, unchanged
    assert extra["engine"] == {"eigenrays": "numpy", "pairing": "numpy", "synthesis": "numpy"}


@needs_cpp
def test_a_sampled_pulse_the_kernel_refuses_records_the_numpy_fallback(clean_env):
    """`table_factor: 0` on the C++ engine: the pairing ran in C++, the synthesis fell back to NumPy."""
    wave = {"type": "sampled", "source": "lfm", "duration": 0.002, "window_alpha": 0.2, "table_factor": 0}
    sonar = sonar_with("sas_auv", wave, 18.0, 22.0, omni=True)
    with pytest.warns(RuntimeWarning, match="NumPy engine"):
        rec = simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False,
                            engine="cpp")
    extra = json.loads(rec.metadata.extra_json)["gpusonar"]
    assert extra["echo_synthesis"] == "numpy_sampled"
    assert extra["engine"] == {"eigenrays": "cpp", "pairing": "cpp", "synthesis": "numpy"}


@needs_cpp
def test_a_sampled_pulse_on_the_kernel_records_cpp(clean_env):
    wave = {"type": "sampled", "source": "lfm", "duration": 0.002, "window_alpha": 0.2}
    sonar = sonar_with("sas_auv", wave, 18.0, 22.0, omni=True)
    rec = simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False,
                        engine="cpp")
    assert note(rec) == {"eigenrays": "cpp", "pairing": "cpp", "synthesis": "cpp_host"}
    assert "echo_synthesis" not in json.loads(rec.metadata.extra_json)["gpusonar"]


@pytest.mark.skipif(not core.cuda_available(), reason="no CUDA device")
@needs_cpp
def test_the_cuda_backend_is_recorded(clean_env):
    rec = run_scenario(_scenario("image_source", LFM, **TIERS["image_source + reverberation"]), engine="cpp",
                       backend=core.Backend.Cuda).records[0]
    assert note(rec) == {"eigenrays": "cpp", "pairing": "cpp", "synthesis": "cpp_cuda", "reverberation": "cpp_cuda"}


@needs_cpp
def test_a_provider_on_another_engine_than_the_pairing_is_recorded_as_it_ran(clean_env):
    """The provider has its own `engine`: eigenrays in NumPy, pairing in C++ (and, since a-engarg, a warning naming
    both)."""
    sonar = sonar_with("sas_auv", {"type": "lfm", "duration": 0.002}, 18.0, 22.0, omni=True)
    with pytest.warns(RuntimeWarning, match="provider passed in"):
        rec = simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False,
                            provider=FreeFieldProvider(C, engine="numpy"), engine="cpp")
    assert note(rec) == {"eigenrays": "numpy", "pairing": "cpp", "synthesis": "cpp_host"}


def test_a_ping_where_no_stage_ran_carries_an_empty_note(clean_env):
    sonar = sonar_with("sas_auv", {"type": "lfm", "duration": 0.002}, 18.0, 22.0, omni=True)
    passive = simulate_ping(sonar, make_pose(), _five_targets(sonar, 20.0), environment={"sound_speed": C}, adc=False,
                            transmit=False)
    empty = simulate_ping(sonar, make_pose(), Scene(), environment={"sound_speed": C}, adc=False)
    assert note(passive) == {} and note(empty) == {}


def test_outside_a_ping_nothing_is_collected(clean_env):
    """A provider called on its own leaves no state behind for the next ping."""
    FreeFieldProvider(C, engine="numpy").eigenrays(np.zeros(3), np.array([[1.0, 2.0, 3.0]]))
    sonar = sonar_with("sas_auv", {"type": "lfm", "duration": 0.002}, 18.0, 22.0, omni=True)
    rec = simulate_ping(sonar, make_pose(), Scene(), environment={"sound_speed": C}, adc=False)
    assert note(rec) == {}


# --- nothing else moves; the file format carries it; consumers ignore it -------------------------------------------

def test_the_note_is_the_only_new_content_of_extra_json(clean_env):
    """An analytic-pulse record had an empty extra_json before; now it holds the note and nothing else. (That samples
    and every other field are bit-identical before and after was shown by sha256 on four shipped scenarios, both
    engines, in the package's report: it needs the tree before the change, so it is not a test.)"""
    rec = run_scenario(_scenario("image_source", LFM, **TIERS["image_source + reverberation"]), engine="numpy").records[0]
    assert json.loads(rec.metadata.extra_json) == {"gpusonar": {"engine": expected("numpy", "image_source", reverb=True)}}


def test_the_note_round_trips_through_the_record_file(clean_env, tmp_path):
    """Written by the C++ writer and read by the C++ reader (libs/io: extra_json is one string attribute, opaque to
    both and to the validator, which checks only that it is a JSON object)."""
    rec = run_scenario(_scenario("image_source", LFM, **TIERS["image_source + reverberation"]), engine="numpy").records[0]
    path = str(tmp_path / "r.h5")
    with core.RecordWriter(path) as w:
        w.write(rec)
    back = core.RecordReader(path).read(0)
    assert back.metadata.extra_json == rec.metadata.extra_json
    assert note(back) == expected("numpy", "image_source", reverb=True)
    assert core.validate(back).error_count == 0 and core.validate_file(path).error_count == 0
    np.testing.assert_array_equal(back.samples.array, rec.samples.array)


def _without_note(record):
    r = record.copy()
    extra = json.loads(r.metadata.extra_json)
    extra["gpusonar"].pop("engine")
    if not extra["gpusonar"]:
        extra.pop("gpusonar")
    m = r.metadata
    m.extra_json = json.dumps(extra)
    r.metadata = m
    return r


LIVE = {"bearings_deg": [-40.0, 40.0, 1.0], "range_cell_m": 1.0, "cfar": {"pfa": 1e-6, "guard_m": 15.0, "train_m": 50.0},
        "min_snr_db": 12.0, "backend": "host"}
LIVE_CPP = {"bearing_first_deg": -40.0, "bearing_last_deg": 40.0, "bearing_step_deg": 1.0, "range_cell_m": 1.0,
            "cfar_pfa": 1e-6, "cfar_guard_m": 15.0, "cfar_train_m": 50.0, "min_snr_db": 12.0}


def test_both_live_chains_ignore_the_note(clean_env):
    """The Python chain (live.py) and the C++ chain (libs/live, whose prepare() writes its own `compressed` mark into
    the same `gpusonar` object, inside its first brace) give the same detections with and without the note."""
    from gpusonar.sonar import SonarModel
    sonar = SonarModel.load("flank_32", {"receive": {"window": {"range": [100.0, 400.0]}}})
    rec = simulate_ping(sonar, make_pose(velocity=(1.5, 0.0, 0.0)),
                        Scene().add_points([[40.0, 250.0, 0.0], [-60.0, 320.0, 0.0]], target_strength_db=10.0),
                        environment={"sound_speed": 1500.0}, noise_rms=1e-3, rng=np.random.default_rng(5))
    assert note(rec)
    bare = _without_note(rec)
    with LiveChain(LIVE) as chain:
        with_py = chain.run([rec.copy()])
    with LiveChain(LIVE) as chain:
        without_py = chain.run([bare.copy()])
    assert with_py[0]["detections"] and with_py[0]["detections"] == without_py[0]["detections"]
    with_cpp = json.loads(core.LiveChainCpp(LIVE_CPP).process(rec.copy()))
    without_cpp = json.loads(core.LiveChainCpp(LIVE_CPP).process(bare.copy()))
    assert with_cpp["detections"] and with_cpp["detections"] == without_cpp["detections"]
