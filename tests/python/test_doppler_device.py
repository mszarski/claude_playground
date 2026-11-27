"""D8 (docs/tasks/DOPPLER.md): the sweep's Doppler bank and the tone's Doppler filter bank by their batched formulation
(libs/live doppler_batched.hpp), opt-in as `chain.doppler.batched` (auto: on the cuda backend; on: on either backend).

On the host the batched formulation runs the same kernels on CPU threads (dsp::HostExec) in the loop's order, so it must
give the loop's results BIT FOR BIT: the tests on the host hold that (report lines equal, maps equal), with one batch and
with several. On a device (the GPU job; skipped here without one) the transforms are cuFFT's and the maths library the
device's: the maps differ from the host's in the last bits. The tolerance, ours: a map sample is a backprojection sum of
compressed samples, each the product of two single-precision transforms (pocketfft against cuFFT, rms relative error
about 1e-7 x sqrt(log2 n) each) and the device's sin and cos against the host's (an ulp); the error of a sample's power,
relative to the strongest sample of its beam, is then of the order of 1e-6 to 1e-5. The tests allow 1e-4 of the beam's
strongest sample (10 to 100 times that) plus 1e-4 relative; detections as tests/python/test_live_cpp.py compares the
engines (0.05 m, 0.05 degrees, 0.05 dB, the same range rate).
"""

import json

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import doppler
from gpusonar.live import LiveChain
from gpusonar.sim import Scene, make_pose, simulate_ping
from gpusonar.sonar import SonarModel

core = gs.core
C = 1500.0
CHAIN = {"bearings_deg": [-40.0, 40.0, 2.0], "backend": "host"}
MAX = {"mode": "sweep_bank", "max_range_rate_m_s": 1.5, "report": True}
CFAR = dict(MAX, combine="cfar")
needs_cuda = pytest.mark.skipif(not core.cuda_available(), reason="no CUDA device")
strip = lambda r: {k: v for k, v in r.items() if k != "processing_seconds"}  # noqa: E731

# Two arrays: the flank array (6-10 kHz, 1 s sweep) and the long-range array (popoto_long_range's sweep), short windows.
SONARS = {"flank": ("flank_32", {"receive": {"window": {"range": [100.0, 400.0]}}}),
          "long_range": ("popoto_long_range", {"receive": {"window": {"range": [200.0, 700.0]}}})}
T_TONE = 0.5
CW = {"type": "cw", "f_start": 8000.0, "f_stop": 8000.0, "duration": T_TONE, "window_alpha": 1.0}


def ping(which="flank", target=(0.0, 250.0, 0.0), velocity=(0.0, 1.0, 0.0), seed=1, sequence=0, wave=None, window=None):
    name, over = SONARS[which]
    over = dict(over)
    if window is not None:
        over["receive"] = {"window": {"range": list(window)}}
    if wave is not None:
        over["transmit"] = {"waveform": wave}
    sonar = SonarModel.load(name, over)
    scene = Scene().add_points([list(target)], velocity=velocity, target_strength_db=0.0)
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)), scene, environment={"sound_speed": C},
                         noise_rms=1e-4, adc=False, absorption_model="none", rng=np.random.default_rng(seed), sequence=sequence,
                         time_ns=sequence * 2_000_000_000)


def reports(cfg, recs):
    with LiveChain(cfg) as chain:
        return [strip(chain.process(r.copy())) for r in recs]


def same_detections(a, b):
    """test_live_cpp.py's comparison of two engines' report lines."""
    assert len(a["detections"]) == len(b["detections"]), (a["detections"], b["detections"])
    for x, y in zip(a["detections"], b["detections"]):
        assert x["range_m"] == pytest.approx(y["range_m"], abs=0.05) and x["bearing_deg"] == pytest.approx(y["bearing_deg"], abs=0.05)
        assert x["snr_db"] == pytest.approx(y["snr_db"], abs=0.05)
        rx, ry = x.get("range_rate_m_s"), y.get("range_rate_m_s")
        # The sweep bank's rate is a hypothesis of its grid; the tone's is interpolated between filters from their
        # powers, so the device's differs from the host's by the powers' float difference (seen on an L4: 2.1e-8 m/s
        # on 1 m/s). 1e-5 m/s is far under one filter's step.
        assert (rx is None) == (ry is None) and (rx is None or rx == pytest.approx(ry, abs=1.0e-5)), (rx, ry)


def baseband_of(rec, bearings, cfg):
    """The baseband record the chain banks (live.py: to_baseband with the plan's allowance) and the plan."""
    r = rec.copy()
    plan = doppler.plan(r, bearings, cfg)
    core.to_baseband(r, max_range_rate=float(plan["baseband_max_range_rate"]))
    return r, plan


def bank(base, bearings, cfg, backend=core.Backend.Host, blank=None):
    return doppler.batched_maps(base, bearings, cfg, True, 1.0, blank, 0.0, 0.0, 1e-5, backend)


def assert_maps_close(dev, host):
    """The device's map against the host's (the module's docstring): 1e-4 of each beam's strongest sample plus 1e-4 relative."""
    dev, host = np.asarray(dev, dtype=float), np.asarray(host, dtype=float)
    scale = np.max(np.abs(host), axis=-1, keepdims=True)
    assert np.all(np.abs(dev - host) <= 1e-4 * scale + 1e-4 * np.abs(host)), float(np.max(np.abs(dev - host) / scale))


# --- the arithmetic of the batches (CPU) ------------------------------------------------------------------------------

def test_the_batch_is_what_the_memory_holds_and_the_runs_split_at_a_new_transform_length():
    lay = core.live_sweep_bank_layout(32, 50_000, 65_536, 35, 161, 2_400, 7, 0, 32, False, False, 0, 0, [65_536] * 35)
    per, fixed = lay["per_hypothesis"], lay["fixed"]
    # Per hypothesis: its filter, its products, its compressed samples and its map (8 bytes a complex float).
    assert per == 8 * 65_536 + 8 * 32 * 65_536 + 8 * 32 * 50_000 + 8 * 161 * 2_400 * 7
    for room, want in ((fixed + 10 * per, 10), (fixed + 10 * per - 1, 9), (fixed, 1), (0, 1), (fixed + 100 * per, 35)):
        assert core.live_sweep_bank_layout(32, 50_000, 65_536, 35, 161, 2_400, 7, 0, 32, False, False, room)["batch"] == want
    assert core.live_sweep_bank_layout(32, 50_000, 65_536, 35, 161, 2_400, 7, 0, 32, False, False, 0, 4)["batch"] == 4    # asked
    assert core.live_sweep_bank_layout(32, 50_000, 65_536, 35, 161, 2_400, 7, 0, 32, False, False, 0, 99)["batch"] == 35
    # cfar: the batch's backgrounds count per hypothesis too.
    cf = core.live_sweep_bank_layout(32, 50_000, 65_536, 35, 161, 2_400, 7, 100, 32, True, True, 0, 0)
    assert cf["per_hypothesis"] > per and cf["fixed"] > fixed
    # Batches never mix transform lengths and keep hypothesis order.
    runs = core.live_sweep_bank_layout(1, 1, 1, 7, 1, 1, 1, 0, 1, False, False, 0, 3, [8, 8, 8, 8, 9, 9, 8])["runs"]
    assert runs == [0, 3, 3, 1, 4, 2, 6, 1]


def test_the_option_is_auto_by_default_said_only_when_set_and_checked():
    assert doppler.cpp_options(MAX) == doppler.cpp_options(dict(MAX, batched="auto", batch=None))
    assert "doppler_batched" not in doppler.cpp_options(MAX) and "doppler_batch" not in doppler.cpp_options(MAX)
    assert doppler.cpp_options(dict(MAX, batched="on", batch=3))["doppler_batched"] == "on"
    assert doppler.cpp_options(dict(MAX, batched="on", batch=3))["doppler_batch"] == 3
    assert doppler.tone_options({"tone": "bank", "batched": "off"})["doppler_batched"] == "off"
    assert not doppler.batched(MAX, core.Backend.Host) and doppler.batched(MAX, core.Backend.Cuda)
    assert doppler.batched(dict(MAX, batched="on"), core.Backend.Host) and not doppler.batched(dict(MAX, batched="off"), core.Backend.Cuda)
    for bad in ({"batched": "yes"}, {"batch": 0}, {"batch": 1.5}):
        with pytest.raises(ValueError):
            doppler.merged(dict(MAX, **bad))
    with LiveChain(dict(CHAIN, engine="cpp", doppler=MAX)) as chain:
        assert "doppler_batched" not in chain._cpp.options()
    with LiveChain(dict(CHAIN, engine="cpp", doppler=dict(MAX, batched="on", batch=2))) as chain:
        o = chain._cpp.options()
        assert '"doppler_batched": "on"' in o and '"doppler_batch": 2' in o


# --- the batched formulation on the host is the loop, bit for bit (CPU) ------------------------------------------------

@pytest.mark.parametrize("which", ["flank", "long_range"])
@pytest.mark.parametrize("combine", ["max", "cfar"])
def test_the_batched_bank_on_the_host_gives_the_loops_report_bit_for_bit(which, combine):
    """The C++ chain: the loop (auto on the host) against the batched formulation on the host, unbatched (as memory allows)
    and in batches of 1 and 3 hypotheses: the same report lines, byte for byte."""
    cfg = MAX if combine == "max" else CFAR
    recs = [ping(which, seed=s, sequence=s) for s in range(2)]
    want = reports(dict(CHAIN, engine="cpp", doppler=cfg), recs)
    assert any(d.get("range_rate_m_s") is not None for d in want[0]["detections"])
    for batch in (None, 1, 3):
        got = reports(dict(CHAIN, engine="cpp", doppler=dict(cfg, batched="on", batch=batch)), recs)
        assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), batch


@pytest.mark.parametrize("combine", ["max", "cfar"])
def test_the_batched_maps_equal_the_loops_maps_bit_for_bit(combine):
    """The maps themselves: the cells the C++ chain's detector is handed (LiveChain.last_map) and the hypothesis of every
    fine sample (last_hypotheses), loop against the batched formulation in one batch and in batches of 2."""
    cfg = MAX if combine == "max" else CFAR
    rec = ping("long_range")

    def maps(d):
        with LiveChain(dict(CHAIN, engine="cpp", doppler=d)) as chain:
            chain.process(rec.copy())
            power, ranges, _ = chain._cpp.last_map()
            which, rates = chain._cpp.last_hypotheses()
            return np.array(power), np.array(ranges), np.array(which), list(rates)

    want = maps(cfg)
    assert want[2].size > 0 and len(set(want[2].ravel().tolist())) > 1
    for batch in (None, 2):
        got = maps(dict(cfg, batched="on", batch=batch))
        for a, b in zip(got, want):
            assert np.array_equal(a, b), batch


@pytest.mark.parametrize("combine", ["max", "cfar"])
def test_the_python_engine_with_the_batched_bank_reports_what_its_loop_does(combine):
    """The Python engine calls the same compiled stage; downstream of the map it is its own code, so its lines equal its
    loop's within the engines' tolerances (its loop folds in NumPy), and the C++ engine's batched lines likewise."""
    cfg = MAX if combine == "max" else CFAR
    recs = [ping("flank", seed=s, sequence=s) for s in range(2)]
    loop = reports(dict(CHAIN, engine="python", doppler=cfg), recs)
    on = reports(dict(CHAIN, engine="python", doppler=dict(cfg, batched="on")), recs)
    cpp = reports(dict(CHAIN, engine="cpp", doppler=dict(cfg, batched="on")), recs)
    for a, b, c in zip(loop, on, cpp):
        same_detections(a, b)
        same_detections(b, c)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_batched_tone_bank_on_the_host_is_the_loop_bit_for_bit(engine):
    """The tone path (one function for both engines, tone_stage): its bank, CFAR and leakage rule by the batched form on
    the host give the loop's power, ratio and detections bit for bit; the chain's report lines likewise."""
    rec = ping("flank", target=(0.0, 500.0, 0.0), velocity=(0.0, 1.0, 0.0), wave=CW, window=[100.0, 900.0])
    bearings = [float(b) for b in np.radians(np.arange(-40.0, 40.5, 2.0))]
    r = rec.copy()
    core.to_baseband(r, max_range_rate=6.0)
    tone = {"tone": "bank", "report": True}
    a = core.live_tone_stage(r, bearings, doppler.tone_options(tone), hann=True, keep_map=True)
    b = core.live_tone_stage(r, bearings, doppler.tone_options(dict(tone, batched="on")), hann=True, keep_map=True)
    assert np.array_equal(a["power"], b["power"]) and np.array_equal(a["ratio"], b["ratio"])
    assert a["detections"] == b["detections"] and len(a["detections"]) >= 1
    assert reports(dict(CHAIN, engine=engine, doppler=dict(tone, batched="on")), [rec]) == reports(dict(CHAIN, engine=engine, doppler=tone), [rec])


def test_the_profile_counts_the_stages():
    core.doppler_profile(reset=True)
    reports(dict(CHAIN, engine="cpp", doppler=dict(MAX, batched="on", batch=2)), [ping("flank")])
    p = core.doppler_profile(reset=True)
    assert p["batched_calls"] == 1 and p["pings"] == 1 and p["batches"] >= 2 and p["hypotheses"] >= 3
    assert all(p[k] >= 0.0 for k in ("prepare", "transfer", "compress", "map", "fold")) and p["map"] > 0.0
    reports(dict(CHAIN, engine="cpp", doppler=MAX), [ping("flank")])
    q = core.doppler_profile(reset=True)
    assert q["batched_calls"] == 0 and q["pings"] == 1 and q["map"] > 0.0 and q["compress"] > 0.0


# --- on a device (the GPU job) -----------------------------------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize("which", ["flank", "long_range"])
@pytest.mark.parametrize("combine", ["max", "cfar"])
@pytest.mark.parametrize("batch", [None, 2])
def test_the_device_bank_gives_the_hosts_maps(which, combine, batch):
    """sweep_bank_stage on the device against the same on the host: maps within the module's tolerance, unbatched and in
    batches of 2; the winning hypothesis the same nearly everywhere (it may differ only where two hypotheses' powers are
    within the tolerance of each other)."""
    cfg = dict(MAX if combine == "max" else CFAR, batch=batch)
    rec = ping(which)
    bearings = [float(b) for b in np.radians(np.arange(-40.0, 40.5, 2.0))]
    base, _ = baseband_of(rec, bearings, cfg)
    host = bank(base, bearings, cfg, core.Backend.Host)
    dev = bank(base, bearings, cfg, core.Backend.Cuda)
    assert dev["fine"].shape == host["fine"].shape and dev["k"] == host["k"]
    assert_maps_close(dev["fine"], host["fine"])
    assert_maps_close(dev["cells"], host["cells"])
    if combine == "cfar":
        np.testing.assert_allclose(dev["level"], host["level"], rtol=1e-4, atol=1e-4 * float(np.max(host["level"])))
    assert np.mean(dev["which"] != host["which"]) < 1e-3


@needs_cuda
@pytest.mark.parametrize("engine", ["python", "cpp"])
@pytest.mark.parametrize("which", ["flank", "long_range"])
@pytest.mark.parametrize("combine", ["max", "cfar"])
@pytest.mark.parametrize("batch", [None, 1])
def test_the_device_chain_reports_the_hosts_lines(engine, which, combine, batch):
    """The chain with backend cuda (the batched bank on the device: batched auto) against the host's loop: the same
    detections within the engines' tolerances and the same range rates; and the two engines equal on the device."""
    cfg = dict(MAX if combine == "max" else CFAR, batch=batch)
    recs = [ping(which, seed=s, sequence=s) for s in range(2)]
    host = reports(dict(CHAIN, engine=engine, doppler=cfg), recs)
    dev = reports(dict(CHAIN, engine=engine, backend="cuda", doppler=cfg), recs)
    other = reports(dict(CHAIN, engine="cpp" if engine == "python" else "python", backend="cuda", doppler=cfg), recs)
    for h, d, o in zip(host, dev, other):
        same_detections(h, d)
        same_detections(d, o)


@needs_cuda
@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_device_tone_bank_gives_the_hosts_power_and_detections(engine):
    """tone_stage on the device (the beams, the bank, the CFAR and the leakage rule there) against the host's loop: power
    and ratio within the module's tolerance, the same detections; the chain's lines within the engines' tolerances."""
    rec = ping("flank", target=(0.0, 500.0, 0.0), velocity=(0.0, 1.0, 0.0), wave=CW, window=[100.0, 900.0])
    bearings = [float(b) for b in np.radians(np.arange(-40.0, 40.5, 2.0))]
    r = rec.copy()
    core.to_baseband(r, max_range_rate=6.0)
    tone = {"tone": "bank", "report": True}
    host = core.live_tone_stage(r, bearings, doppler.tone_options(tone), hann=True, keep_map=True)
    dev = core.live_tone_stage(r, bearings, doppler.tone_options(tone), hann=True, keep_map=True, backend=core.Backend.Cuda)
    nb = host["power"].shape[0]
    assert_maps_close(dev["power"].reshape(nb, -1), host["power"].reshape(nb, -1))
    assert len(dev["detections"]) == len(host["detections"]) >= 1
    for a, b in zip(dev["detections"], host["detections"]):
        assert a["range"] == pytest.approx(b["range"], abs=0.05) and a["range_rate"] == pytest.approx(b["range_rate"], abs=1e-3)
        assert a["snr_db"] == pytest.approx(b["snr_db"], abs=0.05)
    (h,) = reports(dict(CHAIN, engine=engine, doppler=tone), [rec])
    (d,) = reports(dict(CHAIN, engine=engine, backend="cuda", doppler=tone), [rec])
    same_detections(h, d)


# --- one device body per kernel (batch5: the default chain differed between programs on cuda) --------------------------

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_no_kernel_template_is_instantiated_for_the_device_in_two_cuda_files():
    """A kernel template instantiated in two .cu files registers one host stub with two device bodies; which runs depends
    on the program's link and load order. The batched Doppler bank once instantiated the backprojection and far-field
    kernels again in doppler_cuda.cu (compiled with --fmad=false), and with the Doppler options OFF the Python module,
    sonar_live and the Holoscan program gave reports differing in late digits on an L4. Each built CUDA object's weak
    `parallel_for_kernel` stubs must be its own. Runs where a CUDA build exists (the GPU job's build/), skips elsewhere."""
    import shutil
    import subprocess
    objects = sorted((ROOT / "build").glob("**/*.cu.o")) if (ROOT / "build").exists() else []
    if not objects or shutil.which("nm") is None:
        pytest.skip("no CUDA objects built (build/**/*.cu.o) or no nm")
    owner = {}
    for o in objects:
        out = subprocess.run(["nm", str(o)], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[1] == "W" and "parallel_for_kernel" in parts[2]:
                owner.setdefault(parts[2], set()).add(o.name)
    twice = {k: sorted(v) for k, v in owner.items() if len(v) > 1}
    assert owner and not twice, twice


_DEFAULT_RUN = r"""
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
import test_doppler_device as t
from gpusonar.live import LiveChain
recs = [t.ping("flank", seed=s, sequence=s) for s in range(2)]
if sys.argv[2] == "bank_first":     # the batched Doppler bank's kernels loaded and run first in this process
    t.reports(dict(t.CHAIN, engine="cpp", backend="cuda", doppler=t.MAX), recs)
print(json.dumps(t.reports(dict(t.CHAIN, engine="cpp", backend="cuda"), recs), sort_keys=True))
"""


@needs_cuda
def test_the_default_chain_on_the_device_gives_the_same_lines_in_fresh_processes():
    """The default chain (Doppler options off) on the device in two fresh processes, one of which runs the batched bank
    first: bit-identical report lines (no default-path number may depend on what else a process has run)."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    here = str(ROOT / "tests" / "python")
    lines = [subprocess.run([sys.executable, "-c", _DEFAULT_RUN, here, mode], capture_output=True, text=True, check=True, env=env).stdout
             for mode in ("plain", "plain", "bank_first")]
    assert lines[0] and lines[0] == lines[1] == lines[2]


# --- the tone path's grating-lobe notch (doppler.hpp plan_tone step 4b) on either backend --------------------------------

def _lobe_ping():
    import test_doppler_tone_lobes as lobes
    return lobes, lobes.ping(field=1000, movers=((-36.0, 900.0, 1.0),))


def test_the_batched_tone_bank_with_the_lobe_notch_on_the_host_is_the_loop_bit_for_bit():
    """The lobe notch is a mask the batched form takes as it is (tone_notch_mask): on the host its power, ratio and
    detections are the loop's bit for bit, with the notch and without it."""
    lobes, rec = _lobe_ping()
    for tone in (lobes.TONE, lobes.OFF):
        a = lobes.stage(rec, tone, keep_map=True)
        b = lobes.stage(rec, dict(tone, batched="on"), keep_map=True)
        assert np.array_equal(a["power"], b["power"]) and np.array_equal(a["ratio"], b["ratio"])
        assert a["detections"] == b["detections"] and len(a["detections"]) >= 1
    assert lobes.stage(rec, lobes.TONE)["plan"]["lobe_beams"] > 0


@needs_cuda
@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_the_device_tone_bank_with_the_lobe_notch_gives_the_hosts_power_and_detections(engine):
    """On the device with the lobe notch (auto) and without (off): power within the module's tolerance, the same
    detections (0.05 m, 1e-3 m/s, 0.05 dB), and the chain's lines within the engines' tolerances."""
    lobes, rec = _lobe_ping()
    for tone in (lobes.TONE, lobes.OFF):
        opts = doppler.tone_options(tone)
        host = core.live_tone_stage(rec, lobes.bearings(), opts, hann=True, keep_map=True)
        dev = core.live_tone_stage(rec, lobes.bearings(), opts, hann=True, keep_map=True, backend=core.Backend.Cuda)
        nb = host["power"].shape[0]
        assert_maps_close(dev["power"].reshape(nb, -1), host["power"].reshape(nb, -1))
        assert len(dev["detections"]) == len(host["detections"]) >= 1
        for a, b in zip(dev["detections"], host["detections"]):
            assert a["range"] == pytest.approx(b["range"], abs=0.05) and a["range_rate"] == pytest.approx(b["range_rate"], abs=1e-3)
            assert a["snr_db"] == pytest.approx(b["snr_db"], abs=0.05)
        (h,) = reports(dict(lobes.chain_cfg(engine, tone)), [rec])
        (d,) = reports(dict(lobes.chain_cfg(engine, tone), backend="cuda"), [rec])
        same_detections(h, d)


@needs_cuda
@pytest.mark.parametrize("case", ["field", "reverb_train"])
def test_on_the_device_no_tone_detection_lies_inside_a_notch_applied_to_its_beam(case):
    """The invariant of test_doppler_tone_lobes.py on the device (backend cuda: the batched tone stage there, both
    engines): the mask reaches the device's CFAR and flags."""
    import test_doppler_tone_lobes as lobes
    if case == "field":
        rec, tone = lobes.ping(field=2000, movers=((-36.0, 900.0, 1.0),)), lobes.TONE
    else:
        rec, tone = lobes.reverb_train_ping(), lobes.TONE_TRAIN
    lobes.assert_outside_applied_notches(rec, tone, [-40.0, 40.0, 0.5], batched=("auto",), backend="cuda")


def test_the_batched_inputs_carry_the_plans_notch_mask():
    """The plumbing the device form depends on: with the lobe notch the stage tests fewer cells than without (the mask is
    the batched form's input, tone_notch_mask), identically in the loop and the batched form on the host."""
    lobes, rec = _lobe_ping()
    tests = {}
    for tone in (lobes.TONE, lobes.OFF):
        for form in ("off", "on"):
            tests[(tone["tone_lobe_notch"] if "tone_lobe_notch" in tone else "auto", form)] = lobes.stage(rec, dict(tone, batched=form))["tests"]
    assert tests[("auto", "off")] == tests[("auto", "on")] < tests[("off", "off")] == tests[("off", "on")]
