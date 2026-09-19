"""Gaussian beam summation: the fix for the spreading defect, and its limits.

A splat sums ``amplitude * exp(-n^2 / W^2)`` over rays, and for a dense fan the
number of rays landing within ``W`` of a point falls as ``1/s^2``.  So the sum is
``amplitude * W^2 / s^2``, and reproducing free-field spreading requires
``amplitude * W^2`` to be **constant**.  A fixed ``sigma_d`` breaks that, which is
the ``1/R^4`` defect.  Gaussian beams satisfy it identically: in a homogeneous
medium ``amplitude = 1/(s^2 + beta^2)`` and ``W^2 = c(s^2+beta^2)/(omega beta)``,
whose product is ``c/(omega beta)`` at every range.

The tests below check that identity, the absolute constant it implies, and --
the point of the whole construction -- that the answer does not depend on
``beta``, which parameterises the decomposition rather than the physics.
"""

from __future__ import annotations

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, Scene, beam_sum_kwargs, extract_arrivals,
    gaussian_beams, make_time_grid, splat_etc, structured_fan, trace,
)
from hydropt.launch import fan_2d

C = 1500.0
FREQ_KHZ = 0.2
OMEGA = 2 * math.pi * FREQ_KHZ * 1e3
Z_SRC = 500.0
NO_ABSORPTION = lambda f: torch.zeros_like(f)  # noqa: E731


@pytest.fixture(autouse=True)
def _double():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _free_scene(n_steps: int = 200, step: float = 25.0) -> Scene:
    return Scene(field=IsoProfile(C, learnable=False), surface=FlatHeight(-1e5),
                 bottom=FlatHeight(1e5), source=(0.0, 0.0, Z_SRC),
                 surface_loss=ConstantLoss(0.0, learnable=False),
                 bottom_loss=ConstantLoss(0.0, learnable=False),
                 freqs_khz=torch.tensor([FREQ_KHZ]), step_size=step,
                 n_steps=n_steps, max_bounces=0)


def _fan(n: int, half_deg: float):
    dirs, elev, azim = structured_fan(n, n, elev_range_deg=(-half_deg, half_deg),
                                      azim_range_deg=(-half_deg, half_deg))
    d = math.radians(2 * half_deg) / (n - 1)
    weights = (torch.cos(elev).reshape(-1, 1) * d * d).expand(n, n).reshape(-1)
    ev = elev.reshape(-1, 1).expand(n, n).reshape(-1).contiguous()
    az = azim.reshape(1, -1).expand(n, n).reshape(-1).contiguous()
    return dirs, weights.contiguous(), ev, az


# --------------------------------------------------------------------------- #
# The per-ray sigma_d that makes it possible
# --------------------------------------------------------------------------- #
def test_a_scalar_and_a_broadcast_sigma_d_give_identical_answers():
    """Backwards compatibility has to be exact, not approximate: every existing
    call passes a float, and none of them may move."""
    scene = _free_scene(n_steps=100, step=20.0)
    result = trace(scene, fan_2d(40, elev_range_deg=(-10.0, 10.0)))
    rx = torch.tensor([[500.0, 0.0, Z_SRC]])
    grid = make_time_grid(0.2, 0.6, 200)
    kw = dict(sigma_t=2e-3, absorption=NO_ABSORPTION)
    scalar = splat_etc(result, rx, grid, torch.tensor([FREQ_KHZ]), sigma_d=20.0, **kw)
    per_ray = splat_etc(result, rx, grid, torch.tensor([FREQ_KHZ]),
                        sigma_d=torch.full((40,), 20.0), **kw)
    per_vertex = splat_etc(result, rx, grid, torch.tensor([FREQ_KHZ]),
                           sigma_d=torch.full_like(result.arclen, 20.0), **kw)
    assert torch.allclose(per_ray, scalar, rtol=1e-14, atol=0.0)
    assert torch.allclose(per_vertex, scalar, rtol=1e-14, atol=0.0)

    point = torch.tensor([500.0, 0.0, Z_SRC])
    a = extract_arrivals(result, point, torch.tensor([FREQ_KHZ]), sigma_d=20.0)
    b = extract_arrivals(result, point, torch.tensor([FREQ_KHZ]),
                         sigma_d=torch.full((40,), 20.0))
    assert a.n_arrivals == b.n_arrivals > 0
    assert torch.allclose(b.amplitude, a.amplitude, rtol=1e-14, atol=0.0)


def test_a_mis_shaped_sigma_d_is_rejected():
    scene = _free_scene(n_steps=60, step=20.0)
    result = trace(scene, fan_2d(20, elev_range_deg=(-10.0, 10.0)))
    rx = torch.tensor([[400.0, 0.0, Z_SRC]])
    grid = make_time_grid(0.2, 0.5, 100)
    kw = dict(sigma_t=2e-3, absorption=NO_ABSORPTION)
    with pytest.raises(ValueError, match="5 entries for 20 rays"):
        splat_etc(result, rx, grid, torch.tensor([FREQ_KHZ]),
                  sigma_d=torch.ones(5), **kw)
    with pytest.raises(ValueError, match=r"expected \(20, 61\)"):
        splat_etc(result, rx, grid, torch.tensor([FREQ_KHZ]),
                  sigma_d=torch.ones(20, 9), **kw)


def test_beam_sum_kwargs_hands_over_the_width_and_the_amplitude():
    scene = _free_scene(n_steps=60, step=25.0)
    dirs, _, ev, az = _fan(9, 6.0)
    beams = gaussian_beams(scene, ev, az, beam_width=300.0, freq_khz=FREQ_KHZ)
    kw = beam_sum_kwargs(beams)
    assert torch.allclose(kw["spreading"], beams.spreading)
    assert torch.allclose(kw["sigma_d"] * math.sqrt(2.0), beams.width)


# --------------------------------------------------------------------------- #
# The identity the whole method rests on
# --------------------------------------------------------------------------- #
def test_amplitude_times_beam_area_is_constant_in_a_homogeneous_medium():
    """``1/(s^2+beta^2) * c(s^2+beta^2)/(omega beta) = c/(omega beta)``.

    Checked on the real beams, not on the algebra: if either quantity were off,
    their product would drift with range and the spreading law would come out
    wrong.
    """
    scene = _free_scene(n_steps=160, step=25.0)
    dirs, _, ev, az = _fan(9, 6.0)
    for beta in (150.0, 600.0):
        beams = gaussian_beams(scene, ev, az, beam_width=beta, freq_khz=FREQ_KHZ)
        live = beams.result.alive > 0
        product = (beams.spreading * beams.width ** 2)[live]
        expected = C / (OMEGA * beta)
        assert float(product.max() / product.min()) == pytest.approx(1.0, abs=1e-9)
        assert float(product.median()) == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# What it does to a rendered level
# --------------------------------------------------------------------------- #
def _energy(beams, weights, ranges, grid, **override):
    kw = beam_sum_kwargs(beams)
    kw.update(override)
    dt = float(grid[1] - grid[0])
    rx = torch.stack([torch.tensor(ranges), torch.zeros(len(ranges)),
                      torch.full((len(ranges),), Z_SRC)], dim=-1)
    etc = splat_etc(beams.result, rx, grid, torch.tensor([FREQ_KHZ]), sigma_t=6e-3,
                    ray_weights=weights, absorption=NO_ABSORPTION,
                    ray_chunk=4000, **kw)
    return etc[:, 0].sum(-1) * dt


@pytest.mark.parametrize("beta", [200.0, 600.0])
def test_the_beam_sum_reproduces_free_field_spreading(beta):
    ranges = [600.0, 1200.0, 2400.0]
    scene = _free_scene(n_steps=200, step=25.0)
    dirs, w, ev, az = _fan(51, 22.0)
    beams = gaussian_beams(scene, ev, az, beam_width=beta, freq_khz=FREQ_KHZ)
    grid = make_time_grid(0.2, 2.2, 1200)
    e = _energy(beams, w, ranges, grid)
    exponents = [math.log(float(e[i] / e[i + 1]))
                 / math.log(ranges[i + 1] / ranges[i]) for i in range(2)]
    assert all(abs(x - 2.0) < 0.02 for x in exponents), exponents


def test_a_fixed_width_kernel_is_what_breaks_it():
    """The control: same beams, same amplitudes, one fixed ``sigma_d`` -- and the
    spreading exponent doubles.  The defect is the width, not the amplitude."""
    ranges = [600.0, 1200.0, 2400.0]
    scene = _free_scene(n_steps=200, step=25.0)
    dirs, w, ev, az = _fan(51, 22.0)
    beams = gaussian_beams(scene, ev, az, beam_width=400.0, freq_khz=FREQ_KHZ)
    grid = make_time_grid(0.2, 2.2, 1200)
    e = _energy(beams, w, ranges, grid, sigma_d=60.0)
    exponents = [math.log(float(e[i] / e[i + 1]))
                 / math.log(ranges[i + 1] / ranges[i]) for i in range(2)]
    # The near interval reads 3.65 rather than 4: at 600 m a +/-22 deg fan spans
    # +/-242 m against a 6-sigma gate of 360 m, so the kernel is clipped by the
    # edge of the fan.  The far interval is clean, and both are nowhere near 2.
    assert all(x > 3.4 for x in exponents), exponents
    assert exponents[-1] > 3.8, exponents


def test_the_absolute_level_matches_a_closed_form_with_nothing_fitted():
    r"""``E s^2 = pi c / (omega beta)``, so ``E s^2 omega beta / (pi c) = 1``.

    No calibration constant: the beam sum is absolutely normalised, which the
    fixed-aperture estimator never was.  ``E`` is the ETC summed over bins *times
    dt*, because an ETC is an energy density in time -- omitting that is a factor
    of ``1/dt``, which is how this constant was first mis-measured as 531.
    """
    ranges = [600.0, 1200.0, 2400.0]
    scene = _free_scene(n_steps=200, step=25.0)
    dirs, w, ev, az = _fan(51, 22.0)
    grid = make_time_grid(0.2, 2.2, 1200)
    for beta in (200.0, 600.0):
        beams = gaussian_beams(scene, ev, az, beam_width=beta, freq_khz=FREQ_KHZ)
        e = _energy(beams, w, ranges, grid)
        norm = [float(e[i]) * ranges[i] ** 2 * OMEGA * beta / (math.pi * C)
                for i in range(3)]
        assert all(abs(v - 1.0) < 0.03 for v in norm), f"beta={beta}: {norm}"


def test_the_answer_does_not_depend_on_the_beam_width_parameter():
    """The strongest evidence that this reconstructs the field rather than
    depending on the decomposition: ``beta`` is a free parameter of the beam
    family, so a correct sum must be insensitive to it."""
    ranges = [1200.0]
    scene = _free_scene(n_steps=200, step=25.0)
    dirs, w, ev, az = _fan(51, 22.0)
    grid = make_time_grid(0.2, 2.2, 1200)
    levels = []
    for beta in (200.0, 500.0, 1200.0):
        beams = gaussian_beams(scene, ev, az, beam_width=beta, freq_khz=FREQ_KHZ)
        e = float(_energy(beams, w, ranges, grid)[0])
        levels.append(e * ranges[0] ** 2 * OMEGA * beta / (math.pi * C))
    assert max(levels) / min(levels) < 1.03, levels
