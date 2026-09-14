"""Gaussian beams, against the homogeneous closed form and at caustics.

A Gaussian beam in a homogeneous medium has an exact answer: with ``Q1 = s I``
and ``Q2 = I`` the beam determinant is ``(s + i beta)^2``, so the spreading is
``1 / (s^2 + beta^2)`` and the transverse width is
``W^2 = c (s^2 + beta^2) / (omega beta)``.  Both are checked to floating-point
equality, which is possible here because the dynamic quantities come from exact
forward-mode derivatives rather than from a finite difference.

The point of the method is what happens where geometry fails: at the source and
at every caustic the geometric tube has zero area and infinite intensity, and
the beam does not.
"""

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, MunkProfile, Scene, make_time_grid,
)
from hydropt.beams import gaussian_beams, suggest_beam_width
from hydropt.launch import fibonacci_sphere, structured_fan

C = 1500.0
FREQ_KHZ = 0.5


def _free_scene(**kw) -> Scene:
    base = dict(field=IsoProfile(C), bottom=FlatHeight(1e6), surface=FlatHeight(-1e6),
                source=(0.0, 0.0, 0.0), step_size=5.0, n_steps=200)
    base.update(kw)
    return Scene(**base)


def _angles(n: int = 5, half: float = 15.0):
    _, elev, azim = structured_fan(n, n, (-half, half), (-half, half))
    return elev.repeat_interleave(n), azim.repeat(n)


def _munk_scene(n_steps: int = 1000, **kw) -> Scene:
    return _free_scene(field=MunkProfile(learnable=False), source=(0.0, 0.0, 1000.0),
                       step_size=25.0, n_steps=n_steps, **kw)


# --------------------------------------------------------------------------- #
# The homogeneous closed form
# --------------------------------------------------------------------------- #
def test_homogeneous_spreading_is_exactly_one_over_s_squared_plus_beta_squared():
    beta = 9.0
    scene = _free_scene()
    beams = gaussian_beams(scene, *_angles(), beam_width=beta, freq_khz=FREQ_KHZ)
    s = beams.result.arclen
    expected = 1.0 / (s * s + beta**2)
    assert torch.allclose(beams.spreading, expected, rtol=1e-10)


def test_the_beam_is_finite_at_the_source_where_geometry_is_not():
    """``det Q1`` is exactly zero at a point source -- that is what "point
    source" means -- so the geometric tube is infinite there and the beam is
    exactly ``1/beta^2``."""
    beta = 9.0
    scene = _free_scene()
    beams = gaussian_beams(scene, *_angles(), beam_width=beta, freq_khz=FREQ_KHZ)
    assert torch.allclose(beams.spreading[:, 0],
                          torch.full_like(beams.spreading[:, 0], 1.0 / beta**2),
                          rtol=1e-10)
    # The geometric tube at the same vertex is only finite because of a clamp.
    assert (beams.geometric[:, 0] > 1e20).all()


def test_the_beam_converges_to_spherical_spreading_far_from_the_source():
    beta = 9.0
    scene = _free_scene()
    beams = gaussian_beams(scene, *_angles(), beam_width=beta, freq_khz=FREQ_KHZ)
    s = beams.result.arclen[:, -1]
    assert (s > 50 * beta).all()
    assert torch.allclose(beams.spreading[:, -1], 1.0 / (s * s), rtol=1e-3)


def test_beam_width_matches_the_closed_form():
    """``W^2 = c (s^2 + beta^2) / (omega beta)`` -- the standard Gaussian beam
    waist, recovered from ``P = (1/c) dQ/ds`` rather than assumed."""
    beta = 9.0
    scene = _free_scene()
    beams = gaussian_beams(scene, *_angles(), beam_width=beta, freq_khz=FREQ_KHZ)
    omega = 2.0 * math.pi * FREQ_KHZ * 1e3
    s = beams.result.arclen
    expected = (C * (s * s + beta**2) / (omega * beta)).sqrt()
    # Away from the ends, where dQ/ds is a one-sided difference.
    assert torch.allclose(beams.width[:, 5:-5], expected[:, 5:-5], rtol=2e-2)


@pytest.mark.parametrize("beta", [1.0, 9.0, 50.0])
def test_smaller_beams_track_geometry_more_closely(beta):
    scene = _free_scene()
    beams = gaussian_beams(scene, *_angles(), beam_width=beta, freq_khz=FREQ_KHZ)
    s = beams.result.arclen[:, 100]
    ratio = beams.spreading[:, 100] / (1.0 / (s * s))
    # 1/(s^2+b^2) vs 1/s^2 differ by exactly the factor below.
    assert torch.allclose(ratio, s * s / (s * s + beta**2), rtol=1e-10)


# --------------------------------------------------------------------------- #
# Caustics: the reason the method exists
# --------------------------------------------------------------------------- #
def test_spreading_is_bounded_by_one_over_beta_squared_everywhere():
    """``|det Q| >= beta^2`` cannot be violated: the imaginary part of the beam
    determinant does not vanish where the real part does."""
    beta = 24.0
    scene = _munk_scene()
    beams = gaussian_beams(scene, *_angles(9, 8.0), beam_width=beta,
                           freq_khz=FREQ_KHZ)
    live = beams.result.alive > 0
    assert torch.isfinite(beams.spreading).all()
    assert beams.spreading[live].max() <= 1.0 / beta**2 * (1 + 1e-9)


def test_a_caustic_rich_channel_stays_finite():
    beta = 24.0
    scene = _munk_scene(n_steps=1400)
    beams = gaussian_beams(scene, *_angles(9, 8.0), beam_width=beta,
                           freq_khz=FREQ_KHZ)
    assert int(beams.caustics.max()) > 0, "no caustics in a Munk channel"
    assert torch.isfinite(beams.spreading).all()
    assert torch.allclose(beams.kmah_phase, -0.5 * math.pi * beams.caustics)


def test_the_beam_determinant_cannot_vanish_where_the_geometric_one_does():
    """The guarantee the whole method rests on, stated directly.

    ``det Q1`` gets arbitrarily close to zero -- at the source it *is* zero, and
    at every caustic it passes through zero -- while ``|det Q| >= beta^2``
    always, because the imaginary part is still there when the real part goes.
    Testing the determinants rather than a sampled spike avoids depending on
    whether the step grid happened to land near a caustic.
    """
    beta = 24.0
    scene = _munk_scene(n_steps=1400)
    beams = gaussian_beams(scene, *_angles(9, 8.0), beam_width=beta,
                           freq_khz=FREQ_KHZ)
    live = beams.result.alive > 0

    det_geometric = 1.0 / beams.geometric[live]
    det_beam = 1.0 / beams.spreading[live]
    assert det_geometric.min() < beta**2 / 100.0, "geometry never came near a caustic"
    assert det_beam.min() >= beta**2 * (1 - 1e-9)


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def test_beams_work_on_an_unstructured_fan():
    """Unlike the neighbour-difference tube, this needs no grid: the dynamic
    quantities come from each ray's own derivatives."""
    dirs = fibonacci_sphere(24)
    dirs = dirs[dirs[:, 0] > 0.7]
    elev = torch.asin(dirs[:, 2].clamp(-1, 1))
    azim = torch.atan2(dirs[:, 1], dirs[:, 0])
    beams = gaussian_beams(_free_scene(), elev, azim, beam_width=9.0,
                           freq_khz=FREQ_KHZ)
    s = beams.result.arclen
    assert torch.allclose(beams.spreading, 1.0 / (s * s + 81.0), rtol=1e-10)


def test_structured_fan_axes_can_be_passed_broadcast():
    """`structured_fan` reports axes, not per-ray angles.  Passing them as
    `elev[:, None], azim[None, :]` must give the same ray order as its
    `directions`, so a tube and a beam comparison lines up ray for ray."""
    scene = _free_scene()
    dirs, elev, azim = structured_fan(4, 3, elev_range_deg=(-10.0, 10.0),
                                      azim_range_deg=(-6.0, 6.0))
    beams = gaussian_beams(scene, elev[:, None], azim[None, :],
                           beam_width=9.0, freq_khz=FREQ_KHZ)
    assert beams.spreading.shape[0] == dirs.shape[0]
    # The bundle the beams traced is the bundle `directions` describes.
    from hydropt.tracer import trace
    assert torch.allclose(beams.result.pos, trace(scene, dirs).pos, atol=1e-12)


def test_beam_spreading_reaches_the_renderer():
    from hydropt import splat_etc

    scene = _munk_scene()
    scene.receivers = torch.tensor([[12_000.0, 0.0, 1000.0]])
    scene.freqs_khz = torch.tensor([FREQ_KHZ])
    elev, azim = _angles(11, 8.0)
    beams = gaussian_beams(scene, elev, azim, beam_width=24.0, freq_khz=FREQ_KHZ)
    grid = make_time_grid(7.7, 8.4, 300)

    kw = dict(sigma_d=300.0, sigma_t=8e-3)
    plain = splat_etc(beams.result, scene.receivers, grid, scene.freqs_khz, **kw)
    beamed = splat_etc(beams.result, scene.receivers, grid, scene.freqs_khz,
                       spreading=beams.spreading, **kw)
    assert plain.sum() > 0 and beamed.sum() > 0
    assert not torch.allclose(plain, beamed)


def test_beam_spreading_is_differentiable_in_the_profile_shape():
    """Reverse-over-forward again: five of the six traces are jvps, so this is
    the check that an inversion can still use beam spreading."""
    scene = _free_scene(field=MunkProfile(), source=(0.0, 0.0, 1000.0),
                        step_size=25.0, n_steps=120)
    beams = gaussian_beams(scene, *_angles(3, 3.0), beam_width=24.0,
                           freq_khz=FREQ_KHZ)
    beams.spreading.sum().backward()
    assert scene.field.eps.grad is not None
    assert torch.isfinite(scene.field.eps.grad).all()
    assert scene.field.eps.grad.abs().item() > 0


def test_suggest_beam_width_is_a_few_wavelengths():
    assert suggest_beam_width(1.0, wavelengths=3.0) == pytest.approx(3 * 1.5)
    assert suggest_beam_width(10.0, wavelengths=1.0) == pytest.approx(0.15)
    # Higher frequency, narrower beam.
    assert suggest_beam_width(10.0) < suggest_beam_width(1.0)


def test_beams_survive_boundary_reflections():
    scene = _free_scene(bottom=FlatHeight(400.0), surface=FlatHeight(0.0),
                        source=(0.0, 0.0, 200.0),
                        bottom_loss=ConstantLoss(3.0, learnable=False),
                        step_size=10.0, n_steps=200)
    beams = gaussian_beams(scene, *_angles(5, 30.0), beam_width=9.0,
                           freq_khz=FREQ_KHZ)
    assert int(beams.result.n_bottom.sum() + beams.result.n_surface.sum()) > 0
    assert torch.isfinite(beams.spreading).all()
    assert (beams.spreading > 0).all()
