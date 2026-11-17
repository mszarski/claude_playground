"""The batched (device) formulations of the reverberation generator and of the far-field
beamformers against the reference host code.

The CUDA backend of these stages runs a sequence of whole-array steps (kernels and batched
transforms: libs/sim/include/sonar/sim/reverb_batched.hpp, libs/live/include/sonar/live/
beamform_batched.hpp). The same sequence runs on the host through HostExec (`batched=True`),
which is what most of this file tests, so the device code's arithmetic is checked on a machine
without a GPU; the CUDA tests at the end run where a device is present."""

import time

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import flank, processing
from gpusonar.sim import FreeFieldProvider, reverb
from test_beamform_cpp import BEARINGS, beam_map, jammed, worst  # noqa: F401  (jammed is a fixture)
from test_reverb_bistatic import ENGINE_CASES, bistatic_ping
from test_sim_cpp import C, MU_DB, REVERB_CASES, empty_ping, relative_difference

core = gs.core
needs_cuda = pytest.mark.skipif(not core.cuda_available(), reason="no CUDA device")

# Single precision throughout on both sides; the batched formulation takes its Doppler
# rotation and its transforms in single precision where the reference rotates in double.
# Measured on the host: 2e-7 to 9e-7 (RMS and largest sample).
BATCHED_TOL = 5e-6


def monostatic(name, backend=core.Backend.Host, batched=False):
    ping_kw, reverb_kw = REVERB_CASES[name]
    reverb_kw = dict(reverb_kw)
    boundary = reverb_kw.pop("boundary_z", 0.0)
    rec = empty_ping(**ping_kw)
    rng = np.random.default_rng(3)
    reverb.BATCHED_ON_HOST = batched
    try:
        reverb.add_reverberation(rec, FreeFieldProvider(C), boundary, MU_DB, rng, engine="cpp", backend=backend, **reverb_kw)
    finally:
        reverb.BATCHED_ON_HOST = False
    return rec.samples.array, rng.bit_generator.state


def bistatic(name, backend=core.Backend.Host, batched=False):
    ping_kw, reverb_kw = ENGINE_CASES[name]
    rec = bistatic_ping(**ping_kw)
    reverb.BATCHED_ON_HOST = batched
    try:
        reverb.add_reverberation(rec, FreeFieldProvider(C), 0.0, MU_DB, np.random.default_rng(3), engine="cpp",
                                 backend=backend, geometry="bistatic", **reverb_kw)
    finally:
        reverb.BATCHED_ON_HOST = False
    return rec.samples.array


# --- reverberation -------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(REVERB_CASES))
def test_batched_reverberation_matches_the_reference(name):
    (ref, state_ref), (got, state_got) = monostatic(name), monostatic(name, batched=True)
    assert state_ref == state_got                      # the same random numbers were drawn
    assert np.abs(ref).max() > 0.0
    worst_, rms = relative_difference(got, ref)
    assert worst_ < BATCHED_TOL and rms < BATCHED_TOL, (worst_, rms)


@pytest.mark.parametrize("name", list(ENGINE_CASES))
def test_batched_bistatic_reverberation_matches_the_reference(name):
    """Components: a direction with elevation and a Doppler shift of its own for every row."""
    ref, got = bistatic(name), bistatic(name, batched=True)
    assert np.abs(ref).max() > 0.0
    worst_, rms = relative_difference(got, ref)
    assert worst_ < BATCHED_TOL and rms < BATCHED_TOL, (worst_, rms)


def reverb_inputs(n_low=600, n_rows=40, n_channels=6, seed=0):
    rng = np.random.default_rng(seed)
    coarse = np.linspace(0.0, 0.3, 31)
    density = rng.uniform(0.5, 2.0, (n_rows, len(coarse)))
    density[::7] = 0.0                                              # rows without power draw nothing
    offsets = rng.normal(0.0, 0.3, (n_channels, 3))
    pulse = np.exp(1j * rng.uniform(0, 2 * np.pi, 40))
    direction = rng.normal(size=(n_rows, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    doppler = rng.uniform(-40.0, 40.0, n_rows)
    doppler[3] = 0.0
    return dict(coarse=coarse, density=density, offsets=offsets, pulse=pulse, direction=direction, doppler=doppler,
                n_low=n_low, fs=8000.0, time0=0.01, fc=20e3, c=1500.0, n_azimuth=n_rows, decim=4)


def accumulate(p, rows_given, batched=False, run=7, components=True):
    extra = dict(direction=p["direction"], doppler=p["doppler"]) if components else {}
    active = np.flatnonzero(core.sim_reverb_active(p["coarse"], p["density"], p["fs"], p["time0"], p["n_azimuth"], p["decim"],
                                                   p["n_low"], components=components))
    rng = np.random.default_rng(5)
    total = np.zeros((len(p["offsets"]), p["n_low"]), dtype=np.complex64)
    for first in range(0, len(active), run):
        rows = active[first:first + run]
        normals = rng.standard_normal((len(rows), 2, p["n_low"]))
        kw = dict(rows=rows) if rows_given else {}
        core.sim_reverb_accumulate(total, p["coarse"], p["density"], first, normals, p["pulse"], p["offsets"], p["fs"],
                                   p["time0"], p["fc"], p["c"], p["n_azimuth"], p["decim"], velocity=(1.5, -0.5, 0.0),
                                   batched=batched, **extra, **kw)
    return total


@pytest.mark.parametrize("components", [True, False])
def test_naming_the_rows_changes_nothing(components):
    """`rows` only spares the generator the search for the sectors with power in every call."""
    p = reverb_inputs()
    a, b = accumulate(p, False, components=components), accumulate(p, True, components=components)
    assert np.abs(a).max() > 0.0 and np.array_equal(a, b)


def test_rows_are_checked():
    p = reverb_inputs()
    total = np.zeros((len(p["offsets"]), p["n_low"]), dtype=np.complex64)
    normals = np.zeros((2, 2, p["n_low"]))
    args = (total, p["coarse"], p["density"], 0, normals, p["pulse"], p["offsets"], p["fs"], p["time0"], p["fc"], p["c"],
            p["n_azimuth"], p["decim"])
    with pytest.raises(ValueError, match="rows"):
        core.sim_reverb_accumulate(*args, direction=p["direction"], doppler=p["doppler"], rows=np.array([1, 400]))
    with pytest.raises(ValueError, match="rows"):
        core.sim_reverb_accumulate(*args, direction=p["direction"], doppler=p["doppler"], rows=np.array([1]))


@pytest.mark.parametrize("run", [1, 7, 1000])
def test_batched_accumulate_matches_the_reference_whatever_the_runs(run):
    """The components are added in their order whatever the size of a run (the reference host
    code has its own batches), so the result does not depend on how the caller cuts them."""
    p = reverb_inputs()
    ref = accumulate(p, True, run=1000)
    got = accumulate(p, True, batched=True, run=run)
    d = np.abs(got - ref)
    assert d.max() < BATCHED_TOL * np.abs(ref).max()
    assert np.sqrt((d**2).mean() / (np.abs(ref) ** 2).mean()) < BATCHED_TOL


@pytest.mark.parametrize("carrier,real", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("decim", [1, 3])
def test_batched_finish_matches_the_reference(carrier, real, decim):
    """Interpolation to the record's rate, the record's window (also beyond the series: zeros)
    and the carrier, whose phase reaches 1e5 turns here."""
    rng = np.random.default_rng(1)
    n_low, fs = 501, 9000.0
    total = (rng.normal(size=(5, n_low)) + 1j * rng.normal(size=(5, n_low))).astype(np.complex64)
    args = (total, fs, 1700, 2.0131, 2.0, 43e3, decim)
    ref = core.sim_reverb_finish(*args, carrier=carrier, real_output=real)
    got = core.sim_reverb_finish(*args, carrier=carrier, real_output=real, batched=True)
    assert ref.shape == got.shape == (5, 1700) and ref.dtype == got.dtype
    assert np.abs(ref).max() > 0.0
    if decim == 1:
        assert np.all(ref[:, n_low:] == 0) and np.all(got[:, n_low:] == 0)        # past the end of the series
    assert np.abs(got - ref).max() < BATCHED_TOL * np.abs(ref).max()


def test_the_extension_states_what_its_generator_can_do():
    assert core.SIM_REVERB_COMPONENTS is True and core.SIM_REVERB_ROWS is True and core.SIM_REVERB_BATCHED is True
    assert reverb.cpp_has_components()


def test_capability_does_not_depend_on_a_docstring(monkeypatch):
    """A profiler or a test that wraps the kernel must not make the generator fall back."""
    monkeypatch.setattr(core, "sim_reverb_accumulate", lambda *a, **k: None)
    assert reverb.cpp_has_components()
    monkeypatch.delattr(core, "SIM_REVERB_COMPONENTS")
    assert not reverb.cpp_has_components()             # an extension from before the attribute: its docstring decides


def test_profile_counts_the_work():
    core.sim_reverb_profile(reset=True)
    p = reverb_inputs()
    accumulate(p, True, batched=True, run=7)
    prof = core.sim_reverb_profile(reset=True)
    n_active = int(np.count_nonzero(p["density"].any(axis=1)))
    assert prof["components"] == n_active and prof["calls"] == -(-n_active // 7)
    assert prof["transform"] > 0.0 and prof["steer"] > 0.0 and prof["active"] == 0.0
    accumulate(p, False, run=7)
    assert core.sim_reverb_profile()["active"] > 0.0   # without `rows` every call searches
    assert core.sim_reverb_profile(reset=True)["components"] == n_active
    assert core.sim_reverb_profile()["components"] == 0


# --- beamformers ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", ["rect", "hann"])
def test_batched_farfield_matches_the_reference(jammed, window):
    ref = beam_map(core.farfield_beamform_cpp(jammed.copy(), list(BEARINGS), window))
    got = beam_map(core.farfield_beamform_cpp(jammed.copy(), list(BEARINGS), window, batched=True))
    assert got.data.shape == ref.data.shape and got.n_valid == ref.n_valid
    assert worst(got.data, ref.data) < 1e-5
    assert np.array_equal(got.times, ref.times) and np.array_equal(got.bearings, ref.bearings)


@pytest.mark.parametrize("kw", [{}, {"n_subbands": 12, "loading": 0.1, "block": 200, "guard_blocks": 2},
                                {"n_subbands": 5, "loading": 1e-3, "block": 300, "guard_blocks": 0}])
def test_batched_mvdr_matches_the_reference(jammed, kw):
    """Default settings; a short last block with a wider guard; little loading (a worse-conditioned
    covariance) and no guard."""
    ref = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), **kw))
    got = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), batched=True, **kw))
    assert got.data.shape == ref.data.shape and got.n_valid == ref.n_valid
    assert worst(got.data, ref.data) < 1e-4
    want = flank.mvdr_beamform(jammed.copy(), BEARINGS, engine="numpy", **kw)  # and the NumPy stage
    assert worst(got.data, want.data) < 1e-4


def test_batched_mvdr_with_a_bearing_count_that_is_not_a_multiple_of_the_group(jammed):
    bearings = list(BEARINGS[:13])
    ref = beam_map(core.mvdr_beamform_cpp(jammed.copy(), bearings))
    got = beam_map(core.mvdr_beamform_cpp(jammed.copy(), bearings, batched=True))
    assert worst(got.data, ref.data) < 1e-4


def test_batched_mvdr_without_loading_falls_back_or_agrees(jammed):
    """Without loading a difference of sample covariances need not be positive definite; the
    batched code then hands the call to the reference code. Either way the answer is the reference's."""
    kw = dict(loading=0.0, n_subbands=4)
    ref = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), **kw))
    got = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), batched=True, **kw))
    assert worst(got.data, ref.data) < 1e-3


def test_batched_mvdr_refuses_a_record_without_signal(jammed):
    rec = jammed.copy()
    rec.samples.array = np.zeros_like(rec.samples.array)
    for batched in (False, True):
        with pytest.raises(RuntimeError, match="singular|positive"):
            core.mvdr_beamform_cpp(rec.copy(), list(BEARINGS), loading=0.0, batched=batched)


def test_python_beamformers_use_the_cpp_ones_and_keep_numpy_as_reference(jammed):
    for name, kw in (("farfield_beamform", dict(window="hann")), ("mvdr_beamform", {})):
        want = getattr(flank, name)(jammed.copy(), BEARINGS, engine="numpy", **kw)
        got = getattr(flank, name)(jammed.copy(), BEARINGS, engine="cpp", **kw)
        auto = getattr(flank, name)(jammed.copy(), BEARINGS, **kw)
        assert isinstance(got, flank.BeamMap) and got.data.dtype == want.data.dtype == np.complex64
        assert worst(got.data, want.data) < 1e-4
        assert np.array_equal(auto.data, got.data)                 # the C++ one is the default when it is built
        assert got.n_valid == want.n_valid and np.allclose(got.times, want.times, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="engine"):
        flank.farfield_beamform(jammed.copy(), BEARINGS, engine="fortran")


def test_doppler_bank_takes_a_backend(jammed, monkeypatch):
    seen = []
    orig = core.pulse_compress

    def spy(records, *a, **k):
        seen.append(k.get("backend"))
        return orig(records, *a, **k)

    monkeypatch.setattr(core, "pulse_compress", spy)
    rec = jammed.copy()
    a = processing.doppler_bank(rec, [-2.0, 0.0, 2.0])
    b = processing.doppler_bank(rec, [-2.0, 0.0, 2.0], backend=core.Backend.Host)
    assert seen[:3] == [None] * 3 and seen[3:] == [core.Backend.Host] * 3
    assert np.array_equal(a[0].samples.array, b[0].samples.array) and a[1] == b[1] and np.array_equal(a[2], b[2])


# --- CUDA ----------------------------------------------------------------------------------

@needs_cuda
@pytest.mark.parametrize("name", list(REVERB_CASES))
def test_cuda_reverberation_matches_the_batched_host(name):
    """The device and HostExec run the same steps; they differ by cuFFT against pocketfft and by
    the device's single-precision sine and cosine."""
    (ref, state_ref), (got, state_got) = monostatic(name, batched=True), monostatic(name, core.Backend.Cuda)
    assert state_ref == state_got
    worst_, rms = relative_difference(got, ref)
    print(f"[reverb cuda] {name}: worst {worst_:.2e}, rms {rms:.2e}")
    assert worst_ < 1e-4 and rms < 1e-4, (worst_, rms)


@needs_cuda
@pytest.mark.parametrize("name", list(ENGINE_CASES))
def test_cuda_bistatic_reverberation_matches_the_batched_host(name):
    ref, got = bistatic(name, batched=True), bistatic(name, core.Backend.Cuda)
    worst_, rms = relative_difference(got, ref)
    print(f"[reverb cuda] bistatic {name}: worst {worst_:.2e}, rms {rms:.2e}")
    assert worst_ < 1e-4 and rms < 1e-4, (worst_, rms)


@needs_cuda
@pytest.mark.parametrize("carrier,real", [(False, False), (True, True)])
def test_cuda_finish_matches_the_host(carrier, real):
    rng = np.random.default_rng(1)
    total = (rng.normal(size=(5, 501)) + 1j * rng.normal(size=(5, 501))).astype(np.complex64)
    args = (total, 9000.0, 1700, 2.0131, 2.0, 43e3, 3)
    ref = core.sim_reverb_finish(*args, carrier=carrier, real_output=real)
    got = core.sim_reverb_finish(*args, carrier=carrier, real_output=real, backend=core.Backend.Cuda)
    assert np.abs(got - ref).max() < 1e-5 * np.abs(ref).max()


@needs_cuda
def test_cuda_farfield_matches_the_host(jammed):
    ref = beam_map(core.farfield_beamform_cpp(jammed.copy(), list(BEARINGS), "hann"))
    got = beam_map(core.farfield_beamform_cpp(jammed.copy(), list(BEARINGS), "hann", backend=core.Backend.Cuda))
    print(f"[beamform cuda] farfield: worst {worst(got.data, ref.data):.2e}")
    assert worst(got.data, ref.data) < 1e-4
    t = time.perf_counter()
    for _ in range(3):
        core.farfield_beamform_cpp(jammed.copy(), list(BEARINGS), "hann", backend=core.Backend.Cuda)
    print(f"[beamform cuda] farfield, 81 bearings x {jammed.samples.n_samples} samples: {(time.perf_counter() - t) / 3 * 1e3:.0f} ms")


@needs_cuda
@pytest.mark.parametrize("kw", [{}, {"n_subbands": 12, "loading": 0.1, "block": 200, "guard_blocks": 2}])
def test_cuda_mvdr_matches_the_host(jammed, kw):
    ref = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), **kw))
    got = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), backend=core.Backend.Cuda, **kw))
    print(f"[beamform cuda] mvdr {kw}: worst {worst(got.data, ref.data):.2e}")
    assert worst(got.data, ref.data) < 1e-4
    t = time.perf_counter()
    for _ in range(3):
        core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), backend=core.Backend.Cuda, **kw)
    print(f"[beamform cuda] mvdr, 81 bearings x {jammed.samples.n_samples} samples: {(time.perf_counter() - t) / 3 * 1e3:.0f} ms")


@needs_cuda
def test_cuda_mvdr_without_loading_is_the_reference(jammed):
    kw = dict(loading=0.0, n_subbands=4)
    ref = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), **kw))
    got = beam_map(core.mvdr_beamform_cpp(jammed.copy(), list(BEARINGS), backend=core.Backend.Cuda, **kw))
    assert worst(got.data, ref.data) < 1e-3
