"""``torch.autograd.gradcheck`` on tiny float64 scenes.

These are the tests that actually pin down "differentiable": every physical
parameter class the inverse problems optimise -- boundary losses, sound-speed
profile knots, bathymetry node heights, source position -- is checked against
central finite differences through the full forward model (RK4 integration,
boundary reflection, ETC splatting).

Scenes are deliberately tiny.  Each one is also kept away from the handful of
places where the forward model is genuinely non-smooth, and where a finite
difference would therefore straddle a kink rather than measure a derivative:

* the piecewise-linear profile's ``dc/dz`` jumps at a knot;
* the ``space_gate`` / ``time_gate`` sparsification cut-offs;
* the local-minimum selection along a ray, and the time-bin rounding;
* ``max_bounces`` and the ``min_advance`` clamp.

``n_bisect`` and ``n_newton`` are raised above the defaults so the boundary
intersection is converged to machine precision -- with a loose bracket, the
Newton residual itself jumps as the bracket does.
"""

import torch
from torch.autograd import gradcheck

from hydropt import (
    BilinearHeightField, ConstantLoss, FlatHeight, IsoProfile, MunkProfile,
    PiecewiseLinearProfile, RayleighBottomLoss, Scene, make_time_grid,
    spherical_fan, thorp_db_per_km,
)

GRADCHECK_KW = dict(eps=1e-6, atol=1e-6, rtol=1e-4, nondet_tol=0.0)


def _inject(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Swap a registered parameter or buffer for a plain tensor.

    gradcheck needs to drive the quantity as a *function input*, and nn.Module
    refuses to have a non-Parameter assigned over a Parameter.
    """
    module._parameters.pop(name, None)
    module._buffers.pop(name, None)
    object.__setattr__(module, name, tensor)


def _tiny_scene(**overrides) -> Scene:
    kw = dict(
        field=IsoProfile(1500.0),
        bottom=FlatHeight(200.0),
        surface=FlatHeight(0.0),
        source=(0.0, 0.0, 60.0),
        receivers=torch.tensor([[900.0, 0.0, 120.0]]),
        surface_loss=ConstantLoss(0.8),
        bottom_loss=ConstantLoss(3.0),
        freqs_khz=torch.tensor([0.3, 1.2]),
        step_size=20.0,
        n_steps=60,
        n_bisect=40,
        n_newton=3,
    )
    kw.update(overrides)
    return Scene(**kw)


def _render(scene: Scene, *, n_rays: int = 9, sigma_d: float = 120.0,
            sigma_t: float = 6e-3) -> torch.Tensor:
    dirs = spherical_fan(n_rays, 1, (-14.0, 14.0), (0.0, 0.0))
    grid = make_time_grid(0.55, 0.75, 24)
    return scene.render(dirs, grid, sigma_d=sigma_d, sigma_t=sigma_t)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def test_gradcheck_thorp_absorption():
    f = torch.tensor([0.2, 1.0, 8.0], dtype=torch.float64, requires_grad=True)
    assert gradcheck(thorp_db_per_km, (f,), **GRADCHECK_KW)


def test_gradcheck_rayleigh_bottom_loss_in_grazing_angle():
    """Includes angles either side of the critical angle, where the reflection
    coefficient goes from real to complex."""
    model = RayleighBottomLoss(1800.0, 1700.0, 0.5)
    graze = torch.tensor([0.1, 0.4, 0.6, 1.2], dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda g: model(g), (graze,), **GRADCHECK_KW)


def test_gradcheck_rayleigh_bottom_loss_in_sediment_parameters():
    graze = torch.tensor([0.15, 0.7, 1.3], dtype=torch.float64)

    def f(rho2, c2, alpha):
        model = RayleighBottomLoss(1800.0, 1700.0, 0.5, learnable=False)
        for nm, val in zip(("rho2", "c2", "alpha_lambda"), (rho2, c2, alpha)):
            _inject(model, nm, val)
        return model(graze)

    args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                 for v in (1800.0, 1700.0, 0.5))
    assert gradcheck(f, args, **GRADCHECK_KW)


def test_gradcheck_munk_profile_value_and_gradient():
    pts = torch.tensor([[0.0, 0.0, 300.0], [500.0, 200.0, 1800.0]], dtype=torch.float64)

    def f(c1, z1, B, eps):
        prof = MunkProfile(learnable=False)
        for nm, val in zip(("c1", "z1", "B", "eps"), (c1, z1, B, eps)):
            _inject(prof, nm, val)
        c, g = prof.c_and_grad(pts)
        return torch.cat((c, g[:, 2]))

    args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                 for v in (1500.0, 1300.0, 1300.0, 7.37e-3))
    assert gradcheck(f, args, **GRADCHECK_KW)


def test_gradcheck_gridded_field_values():
    from hydropt import GriddedField

    pts = torch.tensor([[250.0, 300.0, 120.0], [710.0, 90.0, 430.0]], dtype=torch.float64)

    def f(values):
        field = GriddedField(values, (0.0, 0.0, 0.0), (500.0, 500.0, 250.0), learnable=False)
        _inject(field, "values", values)
        c, g = field.c_and_grad(pts)
        return torch.cat((c, g.reshape(-1)))

    values = (1500.0 + 5.0 * torch.rand(3, 3, 3, dtype=torch.float64)).requires_grad_(True)
    assert gradcheck(f, (values,), **GRADCHECK_KW)


# --------------------------------------------------------------------------- #
# Through the full forward model
# --------------------------------------------------------------------------- #
def test_gradcheck_boundary_loss_parameters():
    """Boundary losses enter as a smooth 10^(-L/10) factor per bounce."""
    def f(l_surface, l_bottom):
        scene = _tiny_scene()
        _inject(scene.surface_loss, "loss_db", l_surface)
        _inject(scene.bottom_loss, "loss_db", l_bottom)
        return _render(scene)

    args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (0.8, 3.0))
    assert gradcheck(f, args, **GRADCHECK_KW)


def test_gradcheck_profile_knot_values():
    """Knot values bend every ray, so this exercises the gradient of the whole
    RK4 unroll, not just an energy factor."""
    depths = torch.tensor([0.0, 100.0, 220.0], dtype=torch.float64)

    def f(values):
        prof = PiecewiseLinearProfile(depths, [1500.0, 1500.0, 1500.0])
        _inject(prof, "values", values)
        return _render(_tiny_scene(field=prof))

    values = torch.tensor([1502.0, 1496.0, 1508.0], dtype=torch.float64, requires_grad=True)
    assert gradcheck(f, (values,), **GRADCHECK_KW)


def test_gradcheck_bathymetry_node_heights():
    """The test that justifies the Newton refinement in find_crossing: with plain
    bisection these gradients would all be exactly zero."""
    def f(heights):
        bottom = BilinearHeightField(heights, origin=(-200.0, -600.0),
                                     spacing=(700.0, 1200.0), learnable=False)
        _inject(bottom, "heights", heights)
        return _render(_tiny_scene(bottom=bottom))

    heights = torch.tensor([[195.0, 205.0, 210.0],
                            [198.0, 203.0, 207.0],
                            [201.0, 199.0, 204.0]], dtype=torch.float64, requires_grad=True)
    assert gradcheck(f, (heights,), **GRADCHECK_KW)


def test_gradcheck_source_position():
    def f(source):
        scene = _tiny_scene()
        _inject(scene, "source", source)
        return _render(scene)

    source = torch.tensor([0.0, 0.0, 60.0], dtype=torch.float64, requires_grad=True)
    assert gradcheck(f, (source,), **GRADCHECK_KW)


def test_gradcheck_receiver_positions():
    """Array geometry is a scene parameter too, so the ETC must be
    differentiable in the receiver coordinates."""
    def f(receivers):
        scene = _tiny_scene()
        _inject(scene, "receivers", receivers)
        return _render(scene)

    receivers = torch.tensor([[900.0, 0.0, 120.0], [900.0, 40.0, 80.0]],
                             dtype=torch.float64, requires_grad=True)
    assert gradcheck(f, (receivers,), **GRADCHECK_KW)


def test_gradcheck_survives_gradient_checkpointing():
    """Checkpointed and non-checkpointed graphs must give the same gradient --
    it is the same function, recomputed."""
    def make(checkpoint_every):
        def f(l_bottom):
            scene = _tiny_scene(checkpoint_every=checkpoint_every)
            _inject(scene.bottom_loss, "loss_db", l_bottom)
            return _render(scene)
        return f

    x = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    plain = torch.autograd.grad(make(0)(x).sum(), x)[0]
    ckpt = torch.autograd.grad(make(7)(x).sum(), x)[0]
    assert torch.allclose(plain, ckpt, rtol=1e-11, atol=1e-13)
    assert plain.abs() > 0


def test_gradients_are_finite_and_nonzero_for_every_parameter_class():
    """A smoke test with all four parameter classes live at once: a silently
    detached parameter would show up here as a None or zero gradient."""
    prof = PiecewiseLinearProfile([0.0, 100.0, 220.0], [1502.0, 1496.0, 1508.0])
    bottom = BilinearHeightField(torch.full((3, 3), 200.0, dtype=torch.float64),
                                 origin=(-200.0, -600.0), spacing=(700.0, 1200.0))
    scene = _tiny_scene(field=prof, bottom=bottom, learn_source=True)
    _render(scene).sum().backward()

    for name, p in scene.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name} has an identically zero gradient"


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #
def test_retired_rays_do_not_produce_nan_gradients():
    """Regression: rays retired by max_bounces freeze in place, leaving
    zero-length segments.  ``sqrt(0)`` has an infinite derivative, and although
    those segments are masked out of the splat, autograd still evaluated
    ``sqrt``'s backward on them as 0/0 -- which turned every parameter gradient,
    and then every parameter, into NaN a few Adam steps later."""
    scene = _tiny_scene(
        bottom=BilinearHeightField(
            torch.tensor([[198.0, 205.0, 202.0],
                          [201.0, 196.0, 208.0],
                          [199.0, 203.0, 197.0]], dtype=torch.float64),
            origin=(-200.0, -600.0), spacing=(700.0, 1200.0)),
        # Low enough that a good part of the fan is retired mid-flight.
        max_bounces=2,
        learn_source=True,
    )
    result = scene.trace(spherical_fan(24, 1, (-35.0, 35.0), (0.0, 0.0)))
    assert (result.alive[:, -1] == 0).any(), "no ray was retired; test is not exercising the bug"

    _render(scene, n_rays=24, sigma_d=150.0).sum().backward()
    for name, p in scene.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"


def test_ray_passing_exactly_through_a_receiver_has_finite_gradients():
    """Zero miss distance is the other place a norm's backward is 0/0."""
    scene = _tiny_scene(field=IsoProfile(1500.0), learn_source=True,
                        receivers=torch.tensor([[900.0, 0.0, 60.0]]))
    # Launched dead level from (0, 0, 60) straight at the receiver.
    etc = scene.render(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
                       make_time_grid(0.55, 0.75, 24), sigma_d=120.0, sigma_t=6e-3)
    assert etc.sum() > 0
    etc.sum().backward()
    # This ray never bounces, so the boundary losses legitimately have no
    # gradient at all; what matters is that nothing came back non-finite.
    graded = {n: p.grad for n, p in scene.named_parameters() if p.grad is not None}
    assert "source" in graded
    for name, grad in graded.items():
        assert torch.isfinite(grad).all(), f"{name} has a non-finite gradient"
