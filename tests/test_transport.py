"""The Sinkhorn divergence, against the distance it is supposed to be.

Two translated copies of the same blob are ``d`` apart in Wasserstein-2 by
``d^2``, exactly, whatever the blob looks like -- so the divergence has a closed
form to be held to rather than a plausibility check.  The tests also pin the
property the whole thing exists for: the value keeps growing with separation
where a squared-error loss saturates and stops saying anything.
"""

import math

import pytest
import torch

from hydropt.transport import (
    sinkhorn_divergence, sinkhorn_potentials, symmetric_potential,
)


def _grid(n=81, half=20.0):
    x = torch.linspace(-half, half, n)
    return x, x.clone()


def _blob(x, y, cx, cy, width=0.8):
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    return torch.exp(-0.5 * (((gx - cx) / width) ** 2 + ((gy - cy) / width) ** 2))


def test_a_distribution_is_no_distance_from_itself():
    """The debiasing exists for this, and it has to be exact.

    Plain entropic OT is minimised at a blurred version of ``a``, not at ``a``,
    so used as a loss it drags a fit towards a smeared answer.  The subtraction
    only cancels if the self-terms use the SYMMETRIC potential: with the
    ordinary alternating update the two potentials come back half an iteration
    apart and this returns 6 instead of 0.
    """
    x, y = _grid()
    a = _blob(x, y, 1.0, -2.0)
    assert float(sinkhorn_divergence(a, a, x=x, y=y, blur=1.0)) == pytest.approx(
        0.0, abs=1e-9)


@pytest.mark.parametrize("d", [0.5, 2.0, 5.0, 10.0])
def test_translated_blobs_give_the_squared_distance(d):
    """W2^2 between a distribution and its own translate is exactly d^2."""
    x, y = _grid()
    a = _blob(x, y, -6.0, 0.0)
    b = _blob(x, y, -6.0 + d, 0.0)
    got = float(sinkhorn_divergence(a, b, x=x, y=y, blur=0.8, n_iter=400))
    assert got == pytest.approx(d * d, rel=0.02)


def test_it_keeps_growing_where_squared_error_gives_up():
    """The property the loss is for.

    Once two blobs stop overlapping, squared error is the same whatever the
    separation -- it is comparing empty cells with empty cells -- so it has
    nothing left to descend on.  Transport keeps charging for the distance.
    """
    x, y = _grid(121, 40.0)
    a = _blob(x, y, -15.0, 0.0)
    far = [5.0, 10.0, 20.0, 30.0]
    ot = [float(sinkhorn_divergence(a, _blob(x, y, -15.0 + d, 0.0),
                                    x=x, y=y, blur=1.5, n_iter=400)) for d in far]
    an = a / a.sum()
    se = [float((((_blob(x, y, -15.0 + d, 0.0)) / _blob(x, y, -15.0 + d, 0.0).sum()
                  - an) ** 2).sum()) for d in far]
    assert all(ot[i + 1] > ot[i] * 1.5 for i in range(len(ot) - 1))
    assert max(se) / min(se) < 1.05          # squared error: flat, i.e. useless


def test_the_gradient_points_home_from_far_away():
    """A gradient that still knows which way the target is at 30 m.

    This is the whole claim.  The model's blob is 30 m from the measurement's
    and they share not one lit cell; the squared-error gradient there is
    whatever the numerical dust says, while the transport gradient points along
    the line between them.
    """
    x, y = _grid(121, 40.0)
    target = _blob(x, y, 12.0, 0.0)
    centre = torch.tensor([-18.0, 0.0], requires_grad=True)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    model = torch.exp(-0.5 * (((gx - centre[0]) / 0.8) ** 2
                              + ((gy - centre[1]) / 0.8) ** 2))
    sinkhorn_divergence(model, target, x=x, y=y, blur=1.5,
                        n_iter=400).backward()
    step = -centre.grad
    home = torch.tensor([12.0 - (-18.0), 0.0])
    cos = float((step @ home) / (step.norm() * home.norm()))
    assert cos > 0.99


def test_the_separable_kernel_matches_a_dense_one():
    """The factorisation is an optimisation, so it has to change nothing.

    On a grid the squared-Euclidean cost splits across the axes, which turns
    one enormous matrix multiply into two small ones -- 4.5e8 entries into 62k
    for the image this is built for.  Checked here against the dense form on a
    grid small enough to build it.
    """
    n, blur = 11, 1.2
    x, y = _grid(n, 5.0)
    a = _blob(x, y, -1.0, 0.5, width=1.0)
    b = _blob(x, y, 1.5, -1.0, width=1.0)
    a, b = a / a.sum(), b / b.sum()

    gx, gy = torch.meshgrid(x, y, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    cost = ((pts.unsqueeze(1) - pts.unsqueeze(0)) ** 2).sum(-1)
    dense = torch.exp(-cost / (blur * blur))

    u = torch.ones(n * n, dtype=a.dtype)
    v = torch.ones(n * n, dtype=a.dtype)
    af, bf = a.reshape(-1), b.reshape(-1)
    for _ in range(400):
        u = af / (dense @ v)
        v = bf / (dense.T @ u)
    f_dense = blur * blur * torch.log(u).reshape(n, n)

    kx = torch.exp(-((x.reshape(-1, 1) - x.reshape(1, -1)) ** 2) / (blur * blur))
    ky = torch.exp(-((y.reshape(-1, 1) - y.reshape(1, -1)) ** 2) / (blur * blur))
    f_sep, _ = sinkhorn_potentials(a, b, kx, ky, blur=blur, n_iter=400, tol=0.0)
    assert torch.allclose(f_sep, f_dense, atol=1e-8)


def test_it_refuses_rather_than_underflowing():
    """Past the kernel's reach the answer is not a distance, and looks like one.

    ``exp(-c/blur^2)`` underflows to zero, mass on either side of that zero
    cannot be transported, and the value comes back finite and wrong: two blobs
    30 m apart at blur=1 returned 1404 against a true 900 before this check
    existed.
    """
    x, y = _grid(161, 40.0)
    a = _blob(x, y, -25.0, 0.0)
    b = _blob(x, y, 25.0, 0.0)
    with pytest.raises(ValueError, match="reaches only"):
        sinkhorn_divergence(a, b, x=x, y=y, blur=1.0)
    # And with enough blur it is exact again.
    got = float(sinkhorn_divergence(a, b, x=x, y=y, blur=2.5, n_iter=600))
    assert got == pytest.approx(50.0 ** 2, rel=0.02)


def test_the_reach_is_three_times_shorter_in_float32():
    """Single precision underflows sooner, so the same blur reaches less far.

    sqrt(-log(tiny)) is 26.6 in float64 and 9.3 in float32.  A blur chosen for
    a float64 run can therefore refuse -- or worse, would have silently lied --
    in float32, which is exactly the kind of dtype-pinned constant this package
    has been bitten by before.
    """
    x, y = _grid(121, 40.0)
    a, b = _blob(x, y, -12.0, 0.0), _blob(x, y, 12.0, 0.0)
    assert float(sinkhorn_divergence(a, b, x=x, y=y, blur=1.2, n_iter=600)) \
        == pytest.approx(24.0 ** 2, rel=0.03)
    with pytest.raises(ValueError, match="float32"):
        sinkhorn_divergence(a.float(), b.float(), x=x.float(), y=y.float(),
                            blur=1.2)


def test_mass_is_normalised_away():
    """Brightness is not position: doubling an image must not move the answer."""
    x, y = _grid()
    a, b = _blob(x, y, -3.0, 0.0), _blob(x, y, 2.0, 0.0)
    one = float(sinkhorn_divergence(a, b, x=x, y=y, blur=1.0, n_iter=400))
    two = float(sinkhorn_divergence(a * 17.0, b, x=x, y=y, blur=1.0, n_iter=400))
    assert two == pytest.approx(one, rel=1e-6)


def test_bad_inputs_are_rejected():
    x, y = _grid(21, 5.0)
    a = _blob(x, y, 0.0, 0.0)
    with pytest.raises(ValueError, match="blur"):
        sinkhorn_divergence(a, a, x=x, y=y, blur=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        sinkhorn_divergence(a - 0.5, a, x=x, y=y, blur=1.0)
    with pytest.raises(ValueError, match="no mass"):
        sinkhorn_divergence(torch.zeros_like(a), a, x=x, y=y, blur=1.0)
    with pytest.raises(ValueError, match="but b is"):
        sinkhorn_divergence(a, a[:-1], x=x, y=y, blur=1.0)
    with pytest.raises(ValueError, match="coordinates"):
        sinkhorn_divergence(a, a, x=x[:-1], y=y, blur=1.0)


def test_the_symmetric_potential_is_actually_symmetric():
    x, y = _grid(41, 10.0)
    a = _blob(x, y, 0.5, -0.5) ; a = a / a.sum()
    blur = 1.0
    kx = torch.exp(-((x.reshape(-1, 1) - x.reshape(1, -1)) ** 2) / blur ** 2)
    ky = torch.exp(-((y.reshape(-1, 1) - y.reshape(1, -1)) ** 2) / blur ** 2)
    p = symmetric_potential(a, kx, ky, blur=blur, n_iter=400, tol=0.0)
    # It solves the symmetric problem: a = u * (K u) with u = exp(p / eps).
    u = torch.exp(p / blur ** 2)
    assert torch.allclose(u * (kx @ u @ ky.T), a, atol=1e-10)
