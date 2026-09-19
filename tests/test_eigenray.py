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
from hydropt.absorption import thorp_db_per_km
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


def test_it_returns_one_arrival_s_worth_where_the_splat_returns_two_pi_per_leg():
    """The offset that makes this worth having, as a number.

    ``extract_arrivals`` sums ``exp(-d^2/2 sigma^2)/s^2`` over every ray inside
    its acceptance and never divides by the sum of those weights.  On a lattice
    of pitch ``p`` with ``sigma = factor * p`` that sum is ``2 pi factor^2``, so
    the splat reports about 2 pi times one arrival's worth of energy -- eight
    decibels, invariant to how the fan was sampled and therefore invisible to
    any convergence test.

    An eigenray has no acceptance and no weights: one path, one arrival, the
    energy the tube's divergence gives.  So the two differ by roughly 2 pi, and
    the eigenray is the one that reduces to ``1/L^2``.
    """
    import warnings as _w

    from hydropt import target_arrivals
    from hydropt.launch import fibonacci_cone
    from hydropt.targets import ExtendedTarget, IsotropicScattering

    elements = torch.stack([torch.zeros(16), (torch.arange(16.0) - 7.5) * 0.02,
                            torch.full((16,), 50.0)], dim=-1)
    scene = _scene()
    scene.receivers = elements
    target = ExtendedTarget(torch.tensor([[0.0, 0.0, 0.0]]),
                            [IsotropicScattering(1.0, learnable=False)],
                            position=(250.0, 0.0, 50.0), learnable=False)
    tx = fibonacci_cone(4000, torch.tensor([1.0, 0.0, 0.0]), 6.0)

    got = {}
    for leg in ("splat", "eigenray"):
        with _w.catch_warnings(), torch.no_grad():
            _w.simplefilter("ignore")
            a = target_arrivals(scene, target, tx, return_leg=leg,
                                n_rx_rays=1500, rx_half_angle_deg=20.0,
                                max_arrivals_per_leg=400)
        got[leg] = float((a.amplitude ** 2).sum())
        assert a.n_arrivals > 0
    ratio = got["splat"] / got["eigenray"]
    two_pi_squared = (2.0 * math.pi) ** 2
    assert 0.5 * two_pi_squared < ratio < 2.0 * two_pi_squared, (
        f"expected about (2 pi)^2 = {two_pi_squared:.1f} for two legs, "
        f"got {ratio:.2f}")

    # And the eigenray leg gives one arrival per path, where the splat gives
    # one per ray that happened to pass nearby.
    with _w.catch_warnings(), torch.no_grad():
        _w.simplefilter("ignore")
        n_splat = target_arrivals(scene, target, tx, return_leg="splat",
                                  n_rx_rays=1500, rx_half_angle_deg=20.0,
                                  max_arrivals_per_leg=400).n_arrivals
        n_eig = target_arrivals(scene, target, tx, return_leg="eigenray",
                                n_rx_rays=1500, rx_half_angle_deg=20.0,
                                max_arrivals_per_leg=400).n_arrivals
    assert n_splat > 50 * n_eig


def test_the_eigenray_leg_does_not_care_how_the_bracket_was_sampled():
    """No acceptance width means nothing to tie to the fan's spacing."""
    import warnings as _w

    from hydropt import target_arrivals
    from hydropt.launch import fibonacci_cone
    from hydropt.targets import ExtendedTarget, IsotropicScattering

    elements = torch.stack([torch.zeros(16), (torch.arange(16.0) - 7.5) * 0.02,
                            torch.full((16,), 50.0)], dim=-1)
    scene = _scene()
    scene.receivers = elements
    target = ExtendedTarget(torch.tensor([[0.0, 0.0, 0.0]]),
                            [IsotropicScattering(1.0, learnable=False)],
                            position=(250.0, 0.0, 50.0), learnable=False)
    tx = fibonacci_cone(4000, torch.tensor([1.0, 0.0, 0.0]), 6.0)
    seen = []
    for half, n in ((45.0, 420), (20.0, 1500), (5.0, 6000)):
        with _w.catch_warnings(), torch.no_grad():
            _w.simplefilter("ignore")
            a = target_arrivals(scene, target, tx, return_leg="eigenray",
                                n_rx_rays=n, rx_half_angle_deg=half,
                                max_arrivals_per_leg=400)
        seen.append(float((a.amplitude ** 2).sum()))
    assert max(seen) / min(seen) < 1.05


def test_a_direct_path_is_not_charged_for_what_the_ray_hits_afterwards():
    """The path ends at the receiver, and the trace does not.

    A trace runs a fixed number of steps.  A receiver reached early leaves the
    ray flying on, and whatever boundary it meets out there belongs to no path
    that arrived.  Charging the roughness of it is not a rounding error: one
    spurious surface bounce over a wind sea at 120 kHz annihilates the arrival,
    and the target vanishes from the image.  That is exactly what happened to
    the boat in ``examples/21`` at 90 m while the same scene at 300 m was fine,
    because there the overshoot was too short to reach the seabed.

    Built shallow and rough so the overshoot cannot miss the bottom, and with a
    clean line of sight so the true answer is known: one direct arrival.
    """
    scene = Scene(field=IsoProfile(C, learnable=False),
                  bottom=FlatHeight(30.0), surface=FlatHeight(0.0),
                  source=(0.0, 0.0, 12.0), receivers=torch.zeros(1, 3),
                  bottom_loss=ConstantLoss(0.0, learnable=False),
                  surface_loss=ConstantLoss(0.0, learnable=False),
                  freqs_khz=torch.tensor([120.0]),
                  step_size=2.0, n_steps=200, max_bounces=6)
    src = torch.tensor([61.6, -31.4, 1.9])       # a hull patch near the surface
    rcv = torch.tensor([0.0, 0.0, 12.0])         # the array, 70 m away
    arrivals = eigenray_arrivals(scene, src, rcv, scene.freqs_khz,
                                 bracket_rays=2000,
                                 bracket_half_angle_deg=45.0)
    # 70 m of the 400 m the trace covers, so the direct ray overshoots by 330 m
    # and reaches the seabed well beyond the receiver.
    assert arrivals.n_arrivals >= 1, "the direct path was annihilated by a bounce it never made"

    span = float((rcv - src).norm())
    direct = int(arrivals.time.detach().argmin())
    assert float(arrivals.time[direct]) == pytest.approx(span / C, rel=1e-3)
    # Unobstructed and unbounced, so the only things it pays are spherical
    # spreading and Thorp -- 2.7 dB over 70 m at 120 kHz, which is most of the
    # 27% this sits below 1/L.  A single spurious 120 kHz surface bounce would
    # put it at zero, not merely low.
    alpha = float(thorp_db_per_km(torch.tensor([120.0])))
    expected = (1.0 / span) * 10.0 ** (-alpha * span / 20000.0)
    amp = float(arrivals.amplitude[direct].max())
    assert amp == pytest.approx(expected, rel=0.05)


def test_batched_solve_matches_one_pair_at_a_time():
    """Several pairs in one solve are the same paths as one solve per pair.

    A bottom-bounce scene, so each pair has a direct and a reflected path;
    three receivers at different ranges, so the pairs converge at different
    iterations and the per-pair stopping has something to do.
    """
    from hydropt.eigenray import eigenray_arrivals_batched

    scene = _scene(depth=120.0, bounces=2)
    src = torch.tensor([0.0, 0.0, 50.0])
    rcvs = torch.tensor([[200.0, 10.0, 60.0], [350.0, -30.0, 40.0],
                         [500.0, 25.0, 80.0]])
    kw = dict(bracket_rays=800, bracket_half_angle_deg=40.0)
    together = eigenray_arrivals_batched(
        scene, src.reshape(1, 3).expand(3, 3), rcvs, torch.tensor([10.0]), **kw)
    for n in range(3):
        alone = eigenray_arrivals(scene, src, rcvs[n], torch.tensor([10.0]), **kw)
        got = together[n]
        assert got.n_arrivals == alone.n_arrivals >= 2
        assert torch.allclose(got.time, alone.time, rtol=0, atol=1e-7)
        assert torch.allclose(got.amplitude, alone.amplitude, rtol=1e-4, atol=0)
        assert torch.allclose(got.direction, alone.direction, atol=1e-4)
        assert torch.allclose(got.launch_direction, alone.launch_direction, atol=1e-4)


def test_a_bounce_path_spreads_as_its_unfolded_length_both_ways():
    """1/L^2 for the reflected path too, and the same from either end.

    The tube Jacobian used to be taken in the chord's tangent-plane offsets,
    which are angles only for a path along the chord: a flat-bottom bounce
    30 degrees off it read +2.7 dB over 1/L^2 one way and +3.4 dB the other.
    In the path's own frames both directions read the unfolded length.
    """
    from hydropt.absorption import thorp_db_per_km

    scene = _scene(depth=120.0, bounces=2)
    alpha = float(thorp_db_per_km(torch.tensor([10.0])))
    a = torch.tensor([0.0, 0.0, 50.0])
    b = torch.tensor([200.0, 3.0, 61.0])
    for src, rcv in ((a, b), (b, a)):
        arr = eigenray_arrivals(scene, src, rcv, torch.tensor([10.0]),
                                bracket_rays=1500, bracket_half_angle_deg=40.0)
        assert arr.n_arrivals == 2                      # direct and one bounce
        for k in range(2):
            L = float(arr.path_length[k])
            want = 10.0 ** (-alpha * L / 1e4) / L ** 2
            got = float(arr.amplitude[k, 0] ** 2)
            assert abs(10.0 * math.log10(got / want)) < 0.05
    # the bounce path's length is the image-source one
    unfolded = math.sqrt(200.0 ** 2 + 3.0 ** 2 + (2 * 120.0 - 50.0 - 61.0) ** 2)
    assert float(arr.path_length[1]) == pytest.approx(unfolded, abs=0.05)


def test_the_bracket_keeps_the_direct_path_in_a_busy_channel():
    """Every low-order path survives the bracket when many bounces are allowed.

    Shallow water, six bounces: more distinct signatures reach the receiver
    than an 8-path cap would keep, and which ones the cap dropped was decided
    by where the fan's rays fell.  On one leg it dropped the direct path.  Now
    the bracket keeps every path it finds, and the direct, single-surface and
    single-bottom paths are all there, from either end.
    """
    scene = _scene(depth=30.0, surface=0.0, bounces=6, n_steps=200)
    src = torch.tensor([0.0, 0.0, 12.0])
    rcv = torch.tensor([228.0, -85.0, 2.0])
    for s, r in ((src, rcv), (rcv, src)):
        _, residual, signature = find_eigenrays(scene, s, r, bracket_rays=2000,
                                                bracket_half_angle_deg=45.0)
        sig = set(int(v) for v in signature[residual < 2.5])
        assert {0, 1, 1000} <= sig, sig
        assert len(sig) > 8
