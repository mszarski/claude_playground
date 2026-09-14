"""Ray-tube spreading, against the one case with a closed-form answer.

In a homogeneous medium the tube Jacobian is exactly ``s^2 cos(e)`` and the
spreading is exactly ``1/s^2``, so that is the anchor.  The finite-difference
version should miss it by exactly its own truncation term and converge at second
order; the forward-mode version should hit it to machine precision.  Everything
after that is cross-validation between the two, plus caustic behaviour that only
a refracting medium produces.
"""

import math

import pytest
import torch

from hydropt import (
    ConstantLoss, FlatHeight, IsoProfile, MunkProfile, Scene, make_time_grid,
)
from hydropt.launch import structured_fan
from hydropt.spreading import ray_tube, ray_tube_jvp, spherical_spreading

C = 1500.0
VERTEX = 150


def _free_scene(field=None, **kw) -> Scene:
    base = dict(field=field or IsoProfile(C), bottom=FlatHeight(1e6),
                surface=FlatHeight(-1e6), source=(0.0, 0.0, 0.0),
                step_size=5.0, n_steps=200)
    base.update(kw)
    return Scene(**base)


def _fan(n: int = 21, half: float = 20.0):
    return structured_fan(n, n, (-half, half), (-half, half))


def _per_ray_angles(elev, azim):
    n = int(azim.shape[0])
    return elev.repeat_interleave(n), azim.repeat(int(elev.shape[0]))


# --------------------------------------------------------------------------- #
# The homogeneous anchor
# --------------------------------------------------------------------------- #
def test_homogeneous_jacobian_is_s_squared_cos_elevation():
    scene = _free_scene()
    dirs, elev, azim = _fan()
    result = scene.trace(dirs)
    tube = ray_tube(result, elev, azim)

    n = int(azim.shape[0])
    expected = result.arclen[:, VERTEX] ** 2 * torch.cos(elev.repeat_interleave(n))
    rel = (tube.jacobian[:, VERTEX].abs() - expected).abs() / expected
    assert rel.max() < 1e-3


def test_homogeneous_spreading_reduces_to_one_over_s_squared():
    scene = _free_scene()
    dirs, elev, azim = _fan()
    result = scene.trace(dirs)
    tube = ray_tube(result, elev, azim)
    reference = spherical_spreading(result)
    rel = ((tube.spreading[:, VERTEX] - reference[:, VERTEX]).abs()
           / reference[:, VERTEX])
    assert rel.max() < 1e-3
    assert tube.valid[:, VERTEX].all()


@pytest.mark.parametrize("n", [11, 21, 41])
def test_error_is_exactly_the_central_difference_truncation(n):
    """The residual is not slop: a central difference of ``s d(e, a)`` carries a
    factor ``sinc(de) sinc(da)``, so the relative error is ``(de^2 + da^2)/6``.
    Matching that pins down the implementation rather than merely bounding it."""
    scene = _free_scene()
    dirs, elev, azim = _fan(n)
    result = scene.trace(dirs)
    tube = ray_tube(result, elev, azim)
    reference = spherical_spreading(result)

    measured = ((tube.spreading[:, VERTEX] - reference[:, VERTEX]).abs()
                / reference[:, VERTEX]).median().item()
    de = float(elev[1] - elev[0])
    da = float(azim[1] - azim[0])
    assert measured == pytest.approx((de**2 + da**2) / 6.0, rel=0.02)


def test_neighbour_differences_converge_at_second_order():
    scene = _free_scene()
    errors = []
    for n in (11, 21, 41):
        dirs, elev, azim = _fan(n)
        result = scene.trace(dirs)
        reference = spherical_spreading(result)
        tube = ray_tube(result, elev, azim)
        errors.append(((tube.spreading[:, VERTEX] - reference[:, VERTEX]).abs()
                       / reference[:, VERTEX]).median().item())
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.05)


def test_forward_mode_is_exact_in_the_homogeneous_case():
    """jvp differentiates the trajectory itself, so there is no truncation term
    left to converge away."""
    scene = _free_scene()
    dirs, elev, azim = _fan(11)
    result = scene.trace(dirs)
    tube = ray_tube_jvp(scene, *_per_ray_angles(elev, azim))
    reference = spherical_spreading(result)
    rel = ((tube.spreading[:, VERTEX] - reference[:, VERTEX]).abs()
           / reference[:, VERTEX])
    assert rel.max() < 1e-10


def test_no_caustics_in_a_homogeneous_medium():
    scene = _free_scene()
    dirs, elev, azim = _fan()
    tube = ray_tube(scene.trace(dirs), elev, azim)
    assert int(tube.caustics.max()) == 0
    assert torch.equal(tube.kmah_phase, torch.zeros_like(tube.kmah_phase))


# --------------------------------------------------------------------------- #
# A refracting medium: the two methods must agree, and caustics must appear
# --------------------------------------------------------------------------- #
def _munk_scene(n_steps: int = 800) -> Scene:
    return _free_scene(field=MunkProfile(learnable=False),
                       source=(0.0, 0.0, 1000.0), step_size=20.0, n_steps=n_steps)


def test_the_two_methods_agree_in_a_refracting_channel():
    scene = _munk_scene()
    dirs, elev, azim = structured_fan(15, 15, (-7.0, 7.0), (-7.0, 7.0))
    fan = ray_tube(scene.trace(dirs), elev, azim)
    exact = ray_tube_jvp(scene, *_per_ray_angles(elev, azim))

    k = 600
    interior = fan.valid[:, k]
    rel = ((fan.jacobian[:, k] - exact.jacobian[:, k]).abs()
           / exact.jacobian[:, k].abs().clamp_min(1e-30))
    assert rel[interior].median() < 5e-3


def test_a_refracting_channel_focuses_and_produces_caustics():
    """The physics 1/s^2 cannot express: a sound channel concentrates energy at
    convergence zones, and the tube turns inside out getting there.

    Traced out to 24 km, because the first convergence zone of a Munk channel is
    tens of kilometres away -- at 12 km the focusing is already visible but no
    ray has passed a caustic yet."""
    scene = _munk_scene(n_steps=1400)
    dirs, elev, azim = structured_fan(21, 21, (-9.0, 9.0), (-9.0, 9.0))
    result = scene.trace(dirs)
    tube = ray_tube(result, elev, azim)
    reference = spherical_spreading(result)

    k = 1200
    ratio = tube.spreading[:, k] / reference[:, k]
    assert ratio.max() > 10.0, "no focusing found in a Munk channel"
    assert int(tube.caustics[:, k].sum()) > 0, "no caustics found"
    # KMAH phase is -pi/2 per caustic.
    assert torch.allclose(tube.kmah_phase, -0.5 * math.pi * tube.caustics)


def test_the_jacobian_floor_keeps_spreading_finite_at_a_caustic():
    scene = _munk_scene(n_steps=1400)
    dirs, elev, azim = structured_fan(21, 21, (-9.0, 9.0), (-9.0, 9.0))
    result = scene.trace(dirs)
    tube = ray_tube(result, elev, azim, min_jacobian=1e-3)
    assert torch.isfinite(tube.spreading).all()
    # The floor is relative to s^2, so the ratio to spherical spreading is
    # bounded by 1/min_jacobian whatever the geometry does.
    ratio = tube.spreading / spherical_spreading(result)
    assert ratio.max() <= 1.0 / 1e-3 / math.cos(math.radians(9.0)) + 1.0


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def test_tube_is_invalidated_where_neighbours_have_different_bounce_histories():
    """A difference taken across a reflection is a step, not a derivative."""
    scene = _free_scene(bottom=FlatHeight(300.0), surface=FlatHeight(0.0),
                        source=(0.0, 0.0, 150.0),
                        bottom_loss=ConstantLoss(3.0, learnable=False),
                        step_size=10.0, n_steps=300)
    dirs, elev, azim = structured_fan(21, 9, (-40.0, 40.0), (-5.0, 5.0))
    result = scene.trace(dirs)
    tube = ray_tube(result, elev, azim)
    assert int(result.n_bottom.sum()) > 0, "scene did not produce reflections"
    assert not bool(tube.valid.all()), "no vertex was invalidated near a reflection"
    # Invalidated vertices fall back to spherical spreading rather than nonsense.
    fallback = ~tube.valid
    assert torch.allclose(tube.spreading[fallback],
                          spherical_spreading(result)[fallback])


def test_ray_tube_rejects_a_fan_it_was_not_traced_from():
    scene = _free_scene()
    dirs, elev, azim = _fan(11)
    result = scene.trace(dirs)
    with pytest.raises(ValueError, match="structured fan"):
        ray_tube(result, elev, azim[:-1])


def test_spreading_changes_the_rendered_energy():
    """Plumbing check: passing a tube must actually reach the renderer."""
    from hydropt import splat_etc

    scene = _munk_scene()
    scene.receivers = torch.tensor([[10_000.0, 0.0, 1000.0]])
    scene.freqs_khz = torch.tensor([0.2])
    dirs, elev, azim = structured_fan(21, 21, (-9.0, 9.0), (-9.0, 9.0))
    result = scene.trace(dirs)
    tube = ray_tube(result, elev, azim)
    grid = make_time_grid(6.4, 7.0, 400)

    kw = dict(sigma_d=200.0, sigma_t=5e-3)
    plain = splat_etc(result, scene.receivers, grid, scene.freqs_khz, **kw)
    with_tube = splat_etc(result, scene.receivers, grid, scene.freqs_khz,
                          spreading=tube.spreading, **kw)
    assert plain.sum() > 0 and with_tube.sum() > 0
    assert not torch.allclose(plain, with_tube)


@pytest.mark.parametrize("name,expect_zero", [("eps", False), ("c1", True)])
def test_spreading_is_differentiable_in_the_profile_shape(name, expect_zero):
    """``c1`` scales the whole profile, and Snell's law is scale-invariant, so
    ray geometry -- and hence spreading -- genuinely does not depend on it.  A
    zero gradient there is the right answer; a zero gradient for ``eps``, which
    changes the channel's shape, would mean the graph was broken."""
    scene = _free_scene(field=MunkProfile(), source=(0.0, 0.0, 1000.0),
                        step_size=20.0, n_steps=150)
    dirs, elev, azim = structured_fan(5, 5, (-3.0, 3.0), (-3.0, 3.0))
    ray_tube(scene.trace(dirs), elev, azim).spreading.sum().backward()

    grad = getattr(scene.field, name).grad
    assert grad is not None and torch.isfinite(grad).all()
    if expect_zero:
        assert grad.abs().item() < 1e-15
    else:
        assert grad.abs().item() > 1e-6


def test_forward_mode_gradients_match_the_neighbour_version():
    """Reverse-over-forward has to work, or the exact method would be unusable
    for the inverse problems hydropt exists for."""
    def grad_of(method):
        scene = _free_scene(field=MunkProfile(), source=(0.0, 0.0, 1000.0),
                            step_size=20.0, n_steps=150)
        dirs, elev, azim = structured_fan(5, 5, (-3.0, 3.0), (-3.0, 3.0))
        if method == "fan":
            tube = ray_tube(scene.trace(dirs), elev, azim)
        else:
            tube = ray_tube_jvp(scene, *_per_ray_angles(elev, azim))
        tube.spreading.sum().backward()
        return float(scene.field.eps.grad)

    assert grad_of("fan") == pytest.approx(grad_of("jvp"), rel=1e-3)


def test_spherical_spreading_helper_matches_one_over_s_squared():
    scene = _free_scene()
    dirs, _, _ = _fan(5)
    result = scene.trace(dirs)
    s = result.arclen.clamp_min(1.0)
    assert torch.allclose(spherical_spreading(result), 1.0 / (s * s))
