"""Delay estimation in C++ (core.estimate_delays) against the NumPy reference
(gpusonar.motion.correlate_window), and the CUDA path against the host one.

Coarse delay: ASASIN Sec. II-D's (its ref. [11]), the quadratic-interpolated peak of the
correlation magnitude. Fine delay: the phase at the integer peak, which is ours (inherited
from correlate_window); ASASIN takes the phase at the interpolated point.
"""

import numpy as np
import pytest

import gpusonar as gs
from gpusonar import motion

core = gs.core
pytestmark = pytest.mark.skipif(not core.HAVE_MOTION, reason="built without Ceres")

FS, FC = 100e3, 200e3


def pair_of_pings(n_channels=8, n_samples=1200, seed=0):
    """Band-limited complex noise; the second ping is the first delayed by a different
    fraction of a sample per channel, with a phase rotation and independent noise."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n_samples)
    band = np.exp(-0.5 * (f / 0.12) ** 2)
    a = np.empty((n_channels, n_samples), np.complex64)
    b = np.empty_like(a)
    for c in range(n_channels):
        spec = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) * band
        d = -4.0 + c * 1.1                                   # samples, inside a +/-8 lag span
        a[c] = np.fft.ifft(spec)
        b[c] = np.fft.ifft(spec * np.exp(-2j * np.pi * f * d)) * np.exp(-1j * (0.3 + 0.4 * c))
        b[c] += 0.05 * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) * np.abs(a[c]).mean()
    return a, b


def reference(a, b, ca, cb, centres, half, max_lag, dtype):
    out = []
    for i, j in zip(ca, cb):
        for c in centres:
            out.append(motion.correlate_window(a[i].astype(dtype), b[j].astype(dtype), int(c), half, max_lag, FS, FC))
    return out


CASE = dict(ca=[2, 3, 4, 5, 6, 7], cb=[0, 1, 2, 3, 4, 5], centres=[140, 300, 457, 600, 820, 1050], half=64, max_lag=8)


def run_cpp(a, b, backend=core.Backend.Host, **kw):
    k = {**CASE, **kw}
    return core.estimate_delays(a, b, k["ca"], k["cb"], k["centres"], k["half"], k["max_lag"], FS, FC, backend)


def test_cpp_delays_are_bit_close_to_the_numpy_reference_in_double():
    """Same input in double: the only difference left is the order of the sums."""
    a, b = pair_of_pings()
    got = run_cpp(a, b)
    ref = reference(a, b, CASE["ca"], CASE["cb"], CASE["centres"], CASE["half"], CASE["max_lag"], np.complex128)
    assert len(ref) == got["coarse"].size == 36
    assert got["valid"].tolist() == [r is not None for r in ref]
    assert got["valid"].sum() >= 24                  # the case exercises the peak, not only the edge
    for j, r in enumerate(ref):
        if r is None:
            continue
        assert got["coarse"][j] == pytest.approx(r[0], rel=1e-12, abs=1e-12 / FS)
        assert got["fine"][j] == pytest.approx(r[1], rel=1e-12, abs=1e-9 / FC)
        assert got["coherence"][j] == pytest.approx(r[2], rel=1e-12)


def test_cpp_delays_match_the_default_single_precision_reference():
    """Against what estimate_motion's NumPy path computes (complex64 sums): within float
    rounding. Stated: coarse 1e-4 sample, fine 1e-5 period, coherence 1e-5."""
    a, b = pair_of_pings(seed=3)
    got = run_cpp(a, b)
    ref = reference(a, b, CASE["ca"], CASE["cb"], CASE["centres"], CASE["half"], CASE["max_lag"], np.complex64)
    assert got["valid"].tolist() == [r is not None for r in ref]
    for j, r in enumerate(ref):
        if r is None:
            continue
        assert abs(got["coarse"][j] - r[0]) * FS < 1e-4
        assert abs(got["fine"][j] - r[1]) * FC < 1e-5
        assert abs(got["coherence"][j] - r[2]) < 1e-5


def test_peak_at_the_lag_edge_is_invalid_in_both():
    a, b = pair_of_pings(n_channels=2)
    t = np.arange(a.shape[1])
    pulse = lambda t0: sum(np.exp(-0.5 * ((t - c - t0) / 10.0) ** 2) for c in (300, 600)).astype(np.complex64)
    a[0], b[1] = pulse(0.0), pulse(12.0)              # 12 samples: beyond a +/-8 span, peak at the edge
    got = run_cpp(a, b, ca=[0], cb=[1], centres=[300, 600])
    ref = reference(a, b, [0], [1], [300, 600], 64, 8, np.complex128)
    assert ref == [None, None] and not got["valid"].any()


def test_cpp_delays_reject_windows_outside_the_record():
    a, b = pair_of_pings(n_channels=2)
    with pytest.raises(ValueError):
        run_cpp(a, b, ca=[0], cb=[1], centres=[60])           # 60 - 64 - 8 < 0
    with pytest.raises(ValueError):
        run_cpp(a, b, ca=[0], cb=[2], centres=[300])          # no channel 2
    with pytest.raises(ValueError):
        motion._delay_backend(motion.MotionOptions(delay_backend="gpu"))


@pytest.mark.skipif(core.cuda_available(), reason="checks the refusal without a device")
def test_cuda_delay_backend_without_a_device_is_an_error():
    with pytest.raises(RuntimeError):
        motion._delay_backend(motion.MotionOptions(delay_backend="cuda"))


@pytest.mark.skipif(not core.cuda_available(), reason="needs a CUDA build and device")
def test_cuda_delays_match_the_host_port():
    a, b = pair_of_pings(n_channels=16, n_samples=4000, seed=5)
    kw = dict(ca=list(range(4, 16)), cb=list(range(12)), centres=list(np.linspace(200, 3800, 12).astype(int)),
              half=128, max_lag=12)
    host = run_cpp(a, b, core.Backend.Host, **kw)
    dev = run_cpp(a, b, core.Backend.Cuda, **kw)
    assert dev["valid"].tolist() == host["valid"].tolist()
    v = host["valid"]
    # Same kernels and summation order in double; the device may contract to FMA.
    np.testing.assert_allclose(dev["coarse"][v], host["coarse"][v], rtol=1e-11, atol=1e-11 / FS)
    np.testing.assert_allclose(dev["fine"][v], host["fine"][v], rtol=1e-11, atol=1e-9 / FC)
    np.testing.assert_allclose(dev["coherence"][v], host["coherence"][v], rtol=1e-11)


# --- end to end ---------------------------------------------------------------------------------

def test_estimate_motion_with_the_cpp_delays_matches_the_numpy_path():
    from test_motion import collect
    col, _ = collect(n_pings=12, with_target=False, seed=4)
    ref = motion.estimate_motion([r.copy() for r in col.records])
    got = motion.estimate_motion([r.copy() for r in col.records], motion.MotionOptions(delay_backend="host"))
    assert got.n_measurements == ref.n_measurements and got.n_rejected == ref.n_rejected
    assert got.unwrap_corrections == ref.unwrap_corrections
    # Single- against double-precision correlation sums: well under a micrometre.
    np.testing.assert_allclose(got.y, ref.y, atol=1e-7)
    np.testing.assert_allclose(got.z, ref.z, atol=1e-7)
    if core.cuda_available():
        dev = motion.estimate_motion([r.copy() for r in col.records], motion.MotionOptions(delay_backend="cuda"))
        np.testing.assert_allclose(dev.y, got.y, atol=1e-9)
        np.testing.assert_allclose(dev.z, got.z, atol=1e-9)
