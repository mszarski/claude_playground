"""Environment generators, each against a statistic with a closed form.

A random field cannot be checked value by value, so every generator here is
checked against the quantity it is *specified* by: the Pierson-Moskowitz surface
against its significant wave height and peak wavenumber, the fractal seabed
against its spectral slope recovered from the realisation, and the internal-wave
field against ``delta c = -(dc/dz) zeta`` depth by depth.
"""

from __future__ import annotations

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, MunkProfile, PiecewiseLinearProfile, Scene,
    fractal_bathymetry, gaussian_seamount, internal_wave_perturbation,
    pierson_moskowitz_surface, significant_wave_height_pm, spectral_field, trace,
    wave_number_peak_pm,
)
from hydropt.environment import G
from hydropt.launch import fan_2d


@pytest.fixture(autouse=True)
def _double():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _psd_slope(field: torch.Tensor, dx: float, dy: float,
               k_lo: float, k_hi: float) -> float:
    """Least-squares slope of log PSD against log k, over a wavenumber band."""
    ny, nx = field.shape
    kx = 2 * math.pi * torch.fft.fftfreq(nx, d=dx, dtype=field.dtype)
    ky = 2 * math.pi * torch.fft.fftfreq(ny, d=dy, dtype=field.dtype)
    k = (kx.reshape(1, -1) ** 2 + ky.reshape(-1, 1) ** 2).sqrt()
    power = torch.fft.fft2(field).abs() ** 2
    keep = (k > k_lo) & (k < k_hi)
    lk = k[keep].log()
    lp = power[keep].clamp_min(1e-300).log()
    a = torch.stack([lk, torch.ones_like(lk)], dim=1)
    return float(torch.linalg.lstsq(a, lp.unsqueeze(1)).solution[0])


# --------------------------------------------------------------------------- #
# Deterministic
# --------------------------------------------------------------------------- #
def test_seamount_relief_is_the_height_asked_for():
    """The default centre snaps to a node, so the summit is actually sampled."""
    field = gaussian_seamount((64, 64), (50.0, 50.0), base_depth=200.0,
                              height=80.0, width=400.0, learnable=False)
    h = field.heights
    assert float(h.max()) == pytest.approx(200.0, abs=1e-6)
    assert float(h.max() - h.min()) == pytest.approx(80.0, rel=1e-9)


def test_a_summit_between_nodes_loses_exactly_the_gaussian_factor():
    """Not a bug to be hidden: the relief lost is exp(-(dr/width)^2), and knowing
    that is the difference between a coarse grid and a wrong answer."""
    dx = 50.0
    width = 400.0
    on_node = gaussian_seamount((64, 64), (dx, dx), base_depth=200.0, height=80.0,
                                width=width, centre=(1600.0, 1600.0),
                                learnable=False)
    off = gaussian_seamount((64, 64), (dx, dx), base_depth=200.0, height=80.0,
                            width=width, centre=(1600.0 + dx / 2, 1600.0 + dx / 2),
                            learnable=False)
    predicted = 80.0 * math.exp(-2 * (dx / 2) ** 2 / width**2)
    assert float(on_node.heights.max() - on_node.heights.min()) == pytest.approx(
        80.0, rel=1e-9)
    assert float(off.heights.max() - off.heights.min()) == pytest.approx(
        predicted, rel=1e-9)


def test_seamount_is_a_rise_not_a_pit():
    """Depth is positive downward, so the summit must be the *smallest* depth.

    The far corner does not sit at exactly ``base_depth``: a Gaussian has
    infinite support, so the tail is still 0.04 m high 566 m out on a grid only
    775 m across.  That is the shape behaving correctly, and asserting the
    predicted tail is a better check than loosening a tolerance around it.
    """
    dx = 25.0
    n = 32
    width = 200.0
    field = gaussian_seamount((n, n), (dx, dx), base_depth=500.0, height=120.0,
                              width=width, learnable=False)
    assert float(field.heights.min()) == pytest.approx(380.0, rel=1e-9)
    # Summit snapped to node n // 2 on each axis; the far corner is node (0, 0).
    corner = math.hypot(dx * (n // 2), dx * (n // 2))
    tail = 120.0 * math.exp(-((corner / width) ** 2))
    assert float(field.heights.max()) == pytest.approx(500.0 - tail, rel=1e-9)
    assert tail < 0.05, "tail should be small, just not zero"


# --------------------------------------------------------------------------- #
# Spectral machinery
# --------------------------------------------------------------------------- #
def test_sample_normalisation_hits_the_requested_rms_exactly():
    f = spectral_field((128, 128), (10.0, 10.0), lambda k: k.clamp_min(1e-9) ** -3,
                       2.5, generator=_gen())
    assert float(f.std(unbiased=False)) == pytest.approx(2.5, rel=1e-12)
    assert float(f.mean()) == pytest.approx(0.0, abs=1e-10)


def test_ensemble_normalisation_is_right_on_average_and_varies():
    """The distinction the module docstring draws, asserted rather than asserted-to.

    Sample mode removes the variance of the variance; ensemble mode keeps it.
    """
    draws = torch.tensor([
        float(spectral_field((64, 64), (10.0, 10.0),
                             lambda k: k.clamp_min(1e-9) ** -3, 2.5,
                             normalise="ensemble", generator=_gen(s)).std(unbiased=False))
        for s in range(40)
    ])
    assert float(draws.mean()) == pytest.approx(2.5, rel=0.05)
    assert float(draws.std()) > 1e-3, "ensemble draws should not all have equal RMS"
    sample = torch.tensor([
        float(spectral_field((64, 64), (10.0, 10.0),
                             lambda k: k.clamp_min(1e-9) ** -3, 2.5,
                             normalise="sample", generator=_gen(s)).std(unbiased=False))
        for s in range(5)
    ])
    assert float(sample.std()) < 1e-12, "sample mode should pin the RMS"


def test_the_dc_component_is_removed():
    """A random mean offset is not roughness: a surface generator that produced
    one would silently shift the whole boundary."""
    f = spectral_field((64, 64), (5.0, 5.0), lambda k: torch.ones_like(k), 1.0,
                       generator=_gen(3))
    assert abs(float(f.mean())) < 1e-10


def test_an_all_zero_spectrum_is_an_error_not_a_flat_field():
    with pytest.raises(ValueError, match="zero everywhere"):
        spectral_field((32, 32), (1.0, 1.0), lambda k: torch.zeros_like(k), 1.0)


def test_unknown_normalisation_is_rejected():
    with pytest.raises(ValueError, match="unknown normalise"):
        spectral_field((32, 32), (1.0, 1.0), lambda k: torch.ones_like(k), 1.0,
                       normalise="quadratic")


@pytest.mark.parametrize("exponent", [2.5, 3.0, 3.5])
def test_the_realised_spectrum_has_the_slope_that_was_asked_for(exponent):
    """Recovered from the realisation itself, not from the generator's own psd.

    A consistent bias of about -0.02 shows up across every exponent: the fit is
    over discrete wavenumber nodes, whose density per log-k bin is not uniform.
    It is a property of the estimator, not of the field, which is why the
    tolerance is 0.05 and not 1e-6.
    """
    dx = 20.0
    n = 512
    field = fractal_bathymetry((n, n), (dx, dx), base_depth=3000.0, rms=60.0,
                               exponent=exponent, learnable=False, generator=_gen(1))
    relief = field.heights - 3000.0
    k_lo = 4 * 2 * math.pi / (n * dx)
    k_hi = math.pi / dx / 3
    assert _psd_slope(relief, dx, dx, k_lo, k_hi) == pytest.approx(-exponent, abs=0.05)
    assert float(relief.std(unbiased=False)) == pytest.approx(60.0, rel=1e-12)


def test_fractal_bathymetry_sits_on_its_base_depth():
    field = fractal_bathymetry((64, 64), (50.0, 50.0), base_depth=1200.0, rms=30.0,
                               learnable=False, generator=_gen(2))
    assert float(field.heights.mean()) == pytest.approx(1200.0, abs=1e-8)


# --------------------------------------------------------------------------- #
# Pierson-Moskowitz
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("wind", [5.0, 10.0, 15.0])
def test_the_surface_has_the_significant_wave_height_the_wind_implies(wind):
    """H_s = 0.22 U^2 / g, and RMS = H_s / 4."""
    hs = significant_wave_height_pm(wind)
    assert hs == pytest.approx(0.22 * wind**2 / G, rel=1e-12)
    kp = wave_number_peak_pm(wind)
    dx = 2 * math.pi / kp / 8  # eight nodes per peak wavelength
    surface = pierson_moskowitz_surface((256, 256), (dx, dx), wind, generator=_gen(4))
    # Depth positive downward, so elevation is the negative of the stored height.
    assert float(surface.heights.std(unbiased=False)) == pytest.approx(hs / 4, rel=1e-12)
    assert 4 * float(surface.heights.std(unbiased=False)) == pytest.approx(hs, rel=1e-12)


def test_the_wavenumber_peak_is_not_the_frequency_peak_mapped_over():
    r"""A Jacobian separates them, and conflating the two misplaces the dominant
    wavelength by 10%.

    ``S(k) = (alpha/2) k^-3 exp(-beta g^2 / U^4 k^2)`` peaks at
    ``k^2 = 2 beta g^2 / 3 U^4``; the ``S(omega)`` peak at ``0.877 g / U`` maps to
    ``0.769 g / U^2``.
    """
    u = 10.0
    beta = 0.74
    assert wave_number_peak_pm(u) == pytest.approx(
        math.sqrt(2 * beta / 3) * G / u**2, rel=1e-12)
    from_frequency = 0.877**2 * G / u**2
    assert wave_number_peak_pm(u) < from_frequency
    assert wave_number_peak_pm(u) / from_frequency == pytest.approx(0.914, abs=0.01)


def test_the_surface_spectrum_actually_peaks_where_it_says():
    """Measured off the realisation, by radial averaging."""
    u = 10.0
    kp = wave_number_peak_pm(u)
    dx = 2 * math.pi / kp / 6
    n = 512
    surface = pierson_moskowitz_surface((n, n), (dx, dx), u, generator=_gen(5))
    elevation = -surface.heights
    kx = 2 * math.pi * torch.fft.fftfreq(n, d=dx)
    k = (kx.reshape(1, -1) ** 2 + kx.reshape(-1, 1) ** 2).sqrt()
    power = torch.fft.fft2(elevation).abs() ** 2
    # Radially average into bins, then find the peak of k * S_2D(k), which is the
    # omnidirectional S(k) the spectrum was specified by.
    edges = torch.linspace(0.2 * kp, 3.0 * kp, 24)
    centres, omni = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (k >= lo) & (k < hi)
        if int(m.sum()) < 8:
            continue
        kc = float(k[m].mean())
        centres.append(kc)
        omni.append(float(power[m].mean()) * 2 * math.pi * kc)
    peak = centres[int(torch.tensor(omni).argmax())]
    assert peak == pytest.approx(kp, rel=0.25), f"peak at {peak:.4f}, expected {kp:.4f}"


def test_a_zero_or_negative_wind_is_rejected():
    with pytest.raises(ValueError, match="wind_speed must be positive"):
        pierson_moskowitz_surface((32, 32), (5.0, 5.0), 0.0)


# --------------------------------------------------------------------------- #
# Internal waves
# --------------------------------------------------------------------------- #
def _iw_setup(rms_disp: float = 8.0, seed: int = 6):
    base = PiecewiseLinearProfile([0.0, 50.0, 200.0, 1000.0],
                                  [1530.0, 1530.0, 1500.0, 1498.0], learnable=False)
    nz, ny, nx = 41, 32, 32
    dz = 25.0
    field = internal_wave_perturbation(base, (nz, ny, nx), (200.0, 200.0, dz),
                                       rms_displacement=rms_disp,
                                       correlation_length=800.0,
                                       water_depth=1000.0, learnable=False,
                                       generator=_gen(seed))
    z = torch.arange(nz, dtype=torch.get_default_dtype()) * dz
    return base, field, z


def test_the_perturbation_is_the_displacement_times_the_background_gradient():
    r"""delta c = -(dc/dz) zeta, depth by depth -- the whole physical content."""
    rms_disp = 8.0
    base, field, z = _iw_setup(rms_disp)
    c = base.c_of_z(z)
    dcdz = torch.gradient(c, spacing=(z,))[0]
    mode = torch.sin(math.pi * z / 1000.0)
    predicted = (dcdz * mode * rms_disp).abs()
    measured = field.values.reshape(z.shape[0], -1).std(dim=1, unbiased=False)
    live = predicted > 1e-9
    ratio = measured[live] / predicted[live]
    assert float(ratio.max()) == pytest.approx(1.0, rel=1e-9)
    assert float(ratio.min()) == pytest.approx(1.0, rel=1e-9)


def test_the_perturbation_vanishes_at_the_surface_and_the_seabed():
    """A trapped internal wave displaces nothing at either boundary."""
    _, field, z = _iw_setup()
    assert float(field.values[0].abs().max()) == 0.0
    assert float(field.values[-1].abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_the_perturbation_vanishes_in_an_isothermal_layer():
    """The reason for building this from a displacement rather than from noise.

    Between 0 and 50 m the profile is isothermal, so no vertical displacement can
    change the sound speed there -- and the field reflects that without being
    told.  A field of independent noise would put fluctuations there.
    """
    _, field, z = _iw_setup()
    inside = z < 45.0  # strictly inside the layer, away from the knot at 50 m
    assert float(field.values[inside].abs().max()) == pytest.approx(0.0, abs=1e-12)
    # And it is emphatically non-zero in the thermocline below.
    thermocline = (z > 100.0) & (z < 190.0)
    assert float(field.values[thermocline].abs().max()) > 0.1


def test_the_perturbation_scales_linearly_with_displacement():
    _, a, _ = _iw_setup(rms_disp=4.0)
    _, b, _ = _iw_setup(rms_disp=12.0)
    assert torch.allclose(b.values, 3.0 * a.values, rtol=1e-12)


def test_higher_modes_have_more_zero_crossings():
    base = PiecewiseLinearProfile([0.0, 1000.0], [1530.0, 1480.0], learnable=False)
    for mode in (1, 2, 3):
        field = internal_wave_perturbation(base, (81, 8, 8), (200.0, 200.0, 12.5),
                                           rms_displacement=5.0,
                                           correlation_length=600.0, mode=mode,
                                           water_depth=1000.0, learnable=False,
                                           generator=_gen(7))
        column = field.values[:, 4, 4]
        sign = torch.sign(column[column.abs() > 1e-12])
        crossings = int((sign[1:] * sign[:-1] < 0).sum())
        assert crossings == mode - 1, f"mode {mode} gave {crossings} sign changes"


# --------------------------------------------------------------------------- #
# The generators produce things the rest of hydropt can use
# --------------------------------------------------------------------------- #
def test_a_generated_seabed_traces_and_carries_gradients():
    bottom = fractal_bathymetry((16, 16), (100.0, 100.0), base_depth=200.0, rms=15.0,
                                learnable=True, generator=_gen(8))
    scene = Scene(field=IsoProfile(1500.0, learnable=False),
                  bottom=bottom, surface=FlatHeight(0.0),
                  source=(0.0, 0.0, 50.0),
                  surface_loss=ConstantLoss(1.0, learnable=False),
                  bottom_loss=ConstantLoss(6.0, learnable=False),
                  freqs_khz=torch.tensor([10.0]),
                  step_size=2.0, n_steps=300, max_bounces=6)
    result = trace(scene, fan_2d(24, elev_range_deg=(-25.0, 25.0)))
    assert int(result.n_bottom.sum()) > 0, "no ray reached the generated seabed"
    result.pos.sum().backward()
    assert bottom.heights.grad is not None
    assert float(bottom.heights.grad.abs().sum()) > 0.0


def test_a_generated_surface_and_field_build_a_scene_and_refract():
    surface = pierson_moskowitz_surface((16, 16), (20.0, 20.0), 8.0,
                                        generator=_gen(9))
    base = MunkProfile(learnable=False)
    field = internal_wave_perturbation(base, (9, 8, 8), (500.0, 500.0, 250.0),
                                       rms_displacement=10.0,
                                       correlation_length=1500.0, learnable=True,
                                       generator=_gen(10))
    scene = Scene(field=field, bottom=FlatHeight(2000.0), surface=surface,
                  source=(0.0, 0.0, 1000.0),
                  surface_loss=ConstantLoss(0.5, learnable=False),
                  bottom_loss=ConstantLoss(6.0, learnable=False),
                  freqs_khz=torch.tensor([1.0]),
                  step_size=20.0, n_steps=200, max_bounces=4)
    result = trace(scene, fan_2d(16, elev_range_deg=(-10.0, 10.0)))
    assert torch.isfinite(result.pos).all()
    result.tau.sum().backward()
    assert field.values.grad is not None
    assert float(field.values.grad.abs().sum()) > 0.0
