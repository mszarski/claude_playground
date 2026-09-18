"""Eigenrays, against the answers a straight line gives for free.

In isovelocity water with no boundary in the way the path from a point to a
receiver is a straight line, so every quantity has a closed form: the direction
is the unit vector between them, the travel time is ``L / c``, and the ray
tube's cross-section is ``L^2``, giving spherical spreading.  Nothing in the
solver is told any of that -- it shoots a coarse fan and refines -- so these are
tests of the construction rather than restatements of it.

The point of the module is what the splat cannot do: put the arrival direction
where the path actually arrives from, rather than where a nearby ray happened
to be pointing.
"""

import math

import pytest
import torch

from hydropt import ConstantLoss, FlatHeight, IsoProfile, Scene
from hydropt.eigenray import eigenray_arrivals, find_eigenrays

C = 1500.0


def _scene(depth=200.0, surface=-1e5, **kw):
    return Scene(field=IsoProfile(C, learnable=False),
                 bottom=FlatHeight(depth), surface=FlatHeight(surface),
                 source=(0.0, 0.0, 50.0),
                 receivers=torch.zeros(1, 3),
                 bottom_loss=ConstantLoss(0.0, learnable=False),
                 surface_loss=ConstantLoss(0.0, learnable=False),
                 freqs_khz=torch.tensor([10.0]),
                 step_size=kw.pop("step_size", 2.0),
                 n_steps=kw.pop("n_steps", 400), max_bounces=kw.pop("bounces", 0))


def test_a_straight_path_is_found_exactly():
    scene = _scene()
    src = torch.tensor([0.0, 0.0, 50.0])
    rcv = torch.tensor([300.0, 40.0, 70.0])
    dirs, residual, _ = find_eigenrays(scene, src, rcv, bracket_rays=600,
                                       bracket_half_angle_deg=30.0)
    assert dirs.shape[0] == 1
    truth = (rcv - src) / (rcv - src).norm()
    # The refined launch direction is the geometric one, to a small fraction of
    # a degree -- the bracket fan's spacing is 2.5 degrees, so this is the
    # refinement working and not the fan being lucky.
    assert float(torch.rad2deg(torch.acos((dirs[0] * truth).sum()))) < 0.05
    assert float(residual[0]) < 0.5


def test_the_arrival_is_the_closed_form_one():
    scene = _scene()
    src = torch.tensor([0.0, 0.0, 50.0])
    rcv = torch.tensor([300.0, 40.0, 70.0])
    a = eigenray_arrivals(scene, src, rcv, torch.tensor([10.0]),
                          bracket_rays=600, bracket_half_angle_deg=30.0)
    assert a.n_arrivals == 1
    length = float((rcv - src).norm())
    assert float(a.time[0]) == pytest.approx(length / C, rel=2e-3)
    assert float(a.path_length[0]) == pytest.approx(length, rel=2e-3)
    truth = (rcv - src) / (rcv - src).norm()
    assert float(torch.rad2deg(torch.acos((a.direction[0] * truth).sum()))) < 0.2
    # Spreading from the tube's own divergence has to come out spherical --
    # times absorption, which is 1.19 dB/km at 10 kHz and not negligible over
    # 300 m.  Leaving it out of the expectation looks like an 8 percent error
    # in the solver and is not one.
    from hydropt.absorption import thorp_db_per_km
    alpha = float(thorp_db_per_km(torch.tensor([10.0])))
    expected = (1.0 / length ** 2) * 10.0 ** (-alpha * length / 1.0e4)
    assert float(a.amplitude[0, 0] ** 2) == pytest.approx(expected, rel=0.02)


def test_spreading_follows_one_over_r_squared_across_a_decade():
    """The tube Jacobian is measured, not assumed, so this can fail."""
    from hydropt.absorption import thorp_db_per_km

    scene = _scene(n_steps=1200)
    src = torch.tensor([0.0, 0.0, 50.0])
    alpha = float(thorp_db_per_km(torch.tensor([10.0])))
    got = []
    for r in (100.0, 300.0, 1000.0):
        a = eigenray_arrivals(scene, src, torch.tensor([r, 0.0, 50.0]),
                              torch.tensor([10.0]), bracket_rays=600,
                              bracket_half_angle_deg=20.0)
        assert a.n_arrivals == 1
        # Absorption out, so what is left is the geometry alone: 1.19 dB/km at
        # 10 kHz is 0.76 over a kilometre and would otherwise read as the
        # spreading law being wrong by a quarter.
        absorbed = 10.0 ** (-alpha * r / 1.0e4)
        got.append(float(a.amplitude[0, 0] ** 2) * r ** 2 / absorbed)
    assert max(got) / min(got) < 1.05


def test_multipath_comes_back_as_separate_paths_not_one_smeared_one():
    """A surface bounce is a different path, not a wider version of the direct.

    This is what a splat cannot express: it returns many arrivals around one
    another with a spread of directions, where the truth is a small number of
    discrete paths each with its own.
    """
    scene = _scene(depth=60.0, surface=0.0, bounces=2)
    src = torch.tensor([200.0, 0.0, 4.0])
    rcv = torch.tensor([0.0, 0.0, 20.0])
    a = eigenray_arrivals(scene, src, rcv, torch.tensor([10.0]),
                          bracket_rays=4000, bracket_half_angle_deg=45.0,
                          max_paths=4)
    assert a.n_arrivals >= 2
    direct = float((rcv - src).norm())
    assert float(a.path_length[0]) == pytest.approx(direct, rel=5e-3)
    # The surface image of the source sits at -4 m, so its path is longer by a
    # known amount -- and it must arrive from a different direction, not the
    # same one.
    image = math.hypot(200.0, 4.0 + 20.0)
    assert any(abs(float(p) - image) < 0.05 * image for p in a.path_length[1:])
    cosines = (a.direction[1:] * a.direction[0]).sum(-1)
    assert float(torch.rad2deg(torch.acos(cosines.clamp(-1, 1))).min()) > 1.0


def test_it_stays_differentiable_through_the_refinement():
    """The last Newton step carries the implicit derivative of the solution."""
    scene = _scene()
    src = torch.tensor([0.0, 0.0, 50.0], requires_grad=True)
    rcv = torch.tensor([300.0, 40.0, 70.0])
    a = eigenray_arrivals(scene, src, rcv, torch.tensor([10.0]),
                          bracket_rays=600, bracket_half_angle_deg=30.0)
    a.time.sum().backward()
    assert src.grad is not None and bool(torch.isfinite(src.grad).all())
    assert float(src.grad.abs().max()) > 0.0


def test_the_travel_time_gradient_is_the_one_fermat_predicts():
    """dT/d(source) = -(unit vector along the path) / c, exactly.

    Fermat again: the eigenray is stationary in travel time, so moving the
    source only changes the time through the path it already had.  A finite
    difference has to agree, and it is a real check of the refinement -- a
    solver that let the launch direction drag its own derivative in would be
    wrong here by whatever the drag was.
    """
    scene = _scene()
    rcv = torch.tensor([300.0, 40.0, 70.0])
    base = torch.tensor([0.0, 0.0, 50.0])
    src = base.clone().requires_grad_(True)
    a = eigenray_arrivals(scene, src, rcv, torch.tensor([10.0]),
                          bracket_rays=600, bracket_half_angle_deg=30.0)
    a.time.sum().backward()
    unit = (rcv - base) / (rcv - base).norm()
    assert torch.allclose(src.grad, -unit / C, rtol=0.05, atol=1e-8)
