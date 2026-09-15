"""Reflection geometry off sloping boundaries, and energy loss per bounce."""

import math

import pytest
import torch

from hydropt import (
    BilinearHeightField, ConstantLoss, FlatHeight, IsoProfile,
    RayleighBottomLoss, Scene, directions_from_angles, find_crossing,
    grazing_angle, reflect,
)

C = 1500.0


def _sloping_bottom(depth_at_x0: float, slope: float, *, learnable: bool = False):
    """Bottom with constant slope dh/dx, as a 2x2 bilinear patch over 100 km."""
    span = 100_000.0
    h = torch.tensor([[depth_at_x0, depth_at_x0 + slope * span]] * 2)
    return BilinearHeightField(h, origin=(0.0, -span / 2), spacing=(span, span),
                               learnable=learnable)


# --------------------------------------------------------------------------- #
# Pure geometry
# --------------------------------------------------------------------------- #
def test_reflect_mirrors_about_the_normal():
    n = torch.tensor([[0.0, 0.0, 1.0]])
    d = torch.tensor([[0.6, 0.0, 0.8]])
    assert torch.allclose(reflect(d, n), torch.tensor([[0.6, 0.0, -0.8]]))


def test_reflect_preserves_magnitude_and_flips_the_normal_component():
    torch.manual_seed(1)
    d = torch.randn(64, 3)
    n = torch.randn(64, 3)
    n = n / n.norm(dim=-1, keepdim=True)
    r = reflect(d, n)
    assert torch.allclose(r.norm(dim=-1), d.norm(dim=-1), atol=1e-12)
    assert torch.allclose((r * n).sum(-1), -(d * n).sum(-1), atol=1e-12)
    # Reflecting twice is the identity.
    assert torch.allclose(reflect(r, n), d, atol=1e-12)


def test_reflect_preserves_slowness_magnitude():
    """Mirroring is linear, so |(xi, eta, zeta)| = 1/c survives a bounce."""
    n = torch.tensor([[-0.09950371902099893, 0.0, 0.9950371902099893]])
    slow = torch.tensor([[1.0, 0.3, -0.5]]) / C
    assert torch.allclose(reflect(slow, n).norm(dim=-1), slow.norm(dim=-1), atol=1e-15)


def test_sloping_bottom_normal_is_analytic():
    """For z = h(x), the surface normal is (-dh/dx, 0, 1) normalised."""
    slope = 0.1
    bottom = _sloping_bottom(100.0, slope)
    xy = torch.tensor([[30_000.0, 0.0]])
    h, dh = bottom.height_and_slope(xy)
    assert dh[0, 0].item() == pytest.approx(slope, rel=1e-12)
    assert h.item() == pytest.approx(100.0 + slope * 30_000.0, rel=1e-12)
    expected = torch.tensor([-slope, 0.0, 1.0])
    expected = expected / expected.norm()
    assert torch.allclose(bottom.normal(xy)[0], expected, atol=1e-12)


def test_horizontal_ray_reflects_off_a_slope_by_twice_the_slope_angle():
    """A ray skimming horizontally onto a slope of angle a leaves at 2a."""
    slope = 0.1
    a = math.atan(slope)
    n = _sloping_bottom(100.0, slope).normal(torch.tensor([[10.0, 0.0]]))
    out = reflect(torch.tensor([[1.0, 0.0, 0.0]]), n)[0]
    assert math.atan2(out[2].item(), out[0].item()) == pytest.approx(2 * a, rel=1e-12)


def test_grazing_angle_is_measured_from_the_boundary_plane():
    n = torch.tensor([[0.0, 0.0, 1.0]])
    assert grazing_angle(torch.tensor([[1.0, 0.0, 0.0]]), n).item() == pytest.approx(0.0)
    assert grazing_angle(torch.tensor([[0.0, 0.0, 1.0]]), n).item() == pytest.approx(math.pi / 2)
    assert grazing_angle(torch.tensor([[1.0, 0.0, 1.0]]), n).item() == pytest.approx(math.pi / 4)


def test_find_crossing_locates_a_flat_boundary_exactly():
    bottom = FlatHeight(200.0)
    p0 = torch.tensor([[0.0, 0.0, 100.0]])
    p1 = torch.tensor([[400.0, 0.0, 300.0]])
    # z goes 100 -> 300, so z = 200 at half way.
    assert find_crossing(p0, p1, bottom, n_newton=2)[0].item() == pytest.approx(0.5, abs=1e-12)


def test_find_crossing_is_differentiable_in_the_boundary_parameters():
    """The Newton refinement exists precisely so this gradient is not zero:
    raising a flat bottom by dh moves the crossing fraction by -dh / dz."""
    bottom = FlatHeight(200.0, learnable=True)
    p0 = torch.tensor([[0.0, 0.0, 100.0]])
    p1 = torch.tensor([[400.0, 0.0, 300.0]])
    t = find_crossing(p0, p1, bottom, n_newton=2)
    t.sum().backward()
    assert bottom.z0.grad.item() == pytest.approx(1.0 / 200.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Reflection inside the tracer
# --------------------------------------------------------------------------- #
def test_tracer_reflects_off_a_slope_with_the_right_outgoing_angle():
    slope = 0.05
    # Source 100 m deep, bottom 200 m at x=0 and sloping down at 0.05: a 10 deg
    # ray meets it near x = 790 m, leaving ~1 km of traced path afterwards to
    # measure the outgoing direction on.
    scene = Scene(field=IsoProfile(C), bottom=_sloping_bottom(200.0, slope),
                  surface=FlatHeight(-1e6), source=(0.0, 0.0, 100.0),
                  step_size=2.0, n_steps=900, max_bounces=1)
    elev = math.radians(10.0)
    res = scene.trace(directions_from_angles(torch.tensor([elev]), torch.tensor([0.0])))

    assert res.n_bottom.item() == 1
    # Outgoing direction, taken well clear of the bounce.
    out = res.pos[0, -1] - res.pos[0, -20]
    out = out / out.norm()
    a = math.atan(slope)
    # Incoming elevation +10 deg below horizontal; a slope tilts the mirror by a,
    # so the outgoing elevation is -(elev - 2a).
    assert math.asin(out[2].item()) == pytest.approx(-(elev - 2 * a), abs=1e-6)


def test_bounce_energy_loss_is_charged_once_per_bounce():
    loss_db = 2.5
    scene = Scene(field=IsoProfile(C), bottom=FlatHeight(200.0), surface=FlatHeight(0.0),
                  source=(0.0, 0.0, 100.0),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  bottom_loss=ConstantLoss(loss_db, learnable=False),
                  step_size=5.0, n_steps=1200)
    res = scene.trace(directions_from_angles(torch.tensor([math.radians(20.0)]),
                                             torch.tensor([0.0])))
    n_bottom = int(res.n_bottom.item())
    assert n_bottom >= 2
    assert res.refl_db[0, -1].item() == pytest.approx(n_bottom * loss_db, abs=1e-12)


def test_no_double_counting_of_a_single_bounce():
    """A ray hitting a boundary almost exactly at a step end must be charged
    once, not twice -- the min_advance nudge is what guarantees it."""
    scene = Scene(field=IsoProfile(C), bottom=FlatHeight(200.0), surface=FlatHeight(-1e6),
                  source=(0.0, 0.0, 100.0), step_size=100.0, n_steps=40,
                  bottom_loss=ConstantLoss(1.0, learnable=False), max_bounces=99)
    # Straight down: reaches z = 200 after exactly 100 m, i.e. at a step boundary.
    res = scene.trace(torch.tensor([[0.0, 0.0, 1.0]]))
    assert res.n_bottom.item() == 1


def test_grazing_dependent_rayleigh_loss_inside_the_tracer():
    """Steeper bounces must lose more energy than near-critical ones."""
    bottom_loss = RayleighBottomLoss(1800.0, 1700.0, 0.5, learnable=False)
    losses = []
    for elev_deg in (10.0, 45.0):
        scene = Scene(field=IsoProfile(C), bottom=FlatHeight(200.0),
                      surface=FlatHeight(-1e6), source=(0.0, 0.0, 100.0),
                      bottom_loss=bottom_loss, step_size=5.0, n_steps=200,
                      max_bounces=1)
        res = scene.trace(directions_from_angles(torch.tensor([math.radians(elev_deg)]),
                                                 torch.tensor([0.0])))
        assert res.n_bottom.item() == 1
        losses.append(res.refl_db[0, -1].item())
    # 10 deg is below the 28 deg critical angle -> near-total reflection.
    assert losses[0] < 0.5
    assert losses[1] > 4.0


def test_rayleigh_loss_vanishes_below_critical_for_a_lossless_sediment():
    critical = math.degrees(math.acos(1500.0 / 1700.0))
    model = RayleighBottomLoss(1800.0, 1700.0, 0.0, learnable=False)
    below = torch.tensor([1.0, 10.0, critical - 1.0]) * math.pi / 180.0
    assert model(below).abs().max().item() < 1e-12
    above = torch.tensor([critical + 5.0, 60.0]) * math.pi / 180.0
    assert (model(above) > 1.0).all()


def test_rayleigh_loss_is_monotone_above_critical_and_never_negative():
    model = RayleighBottomLoss(1800.0, 1700.0, 0.3, learnable=False)
    graze = torch.linspace(0.001, math.pi / 2, 400)
    loss = model(graze)
    assert (loss >= -1e-12).all()
    critical = math.acos(1500.0 / 1700.0)
    upper = loss[graze > critical + 0.05]
    assert (upper[1:] - upper[:-1] > -1e-9).all()


def test_surface_and_bottom_losses_are_charged_to_the_right_boundary():
    scene = Scene(field=IsoProfile(C), bottom=FlatHeight(200.0), surface=FlatHeight(0.0),
                  source=(0.0, 0.0, 100.0),
                  surface_loss=ConstantLoss(1.0, learnable=False),
                  bottom_loss=ConstantLoss(10.0, learnable=False),
                  step_size=5.0, n_steps=2000)
    res = scene.trace(directions_from_angles(torch.tensor([math.radians(25.0)]),
                                             torch.tensor([0.0])))
    expected = res.n_surface.item() * 1.0 + res.n_bottom.item() * 10.0
    assert res.n_surface.item() > 0 and res.n_bottom.item() > 0
    assert res.refl_db[0, -1].item() == pytest.approx(expected, abs=1e-12)
