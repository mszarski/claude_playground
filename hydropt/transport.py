"""Optimal transport between images, for losses that reach.

A squared-error loss compares two images cell by cell, which means it has
nothing to say about a target it does not overlap.  Put the model's boat 50 m
from the real one and every cell holding the model is empty in the measurement
and vice versa: the mismatch is the same whichever way the model moves, and the
gradient that remains is speckle.  No learning rate or schedule repairs that --
it is a property of the loss, and it is why ``examples/16`` is a refiner rather
than a search.

Optimal transport asks a different question: not "does this cell match" but
"what would it cost to *move* this mass onto that mass".  That cost falls as
the two get closer, from any separation, so the gradient points at the target
across the whole image.

**Entropic regularisation, and what the blur means.**  Exact OT on an image is
a linear program.  Adding an entropy term makes it a matrix-scaling problem
solvable by Sinkhorn iterations, and the regularisation strength has a physical
reading: ``blur`` is the distance over which mass can be rearranged for free.
Below it the divergence stops distinguishing arrangements; above it the
geometry is intact.  Choose it as the scale you are willing to be wrong by.

**Debiased.**  Plain entropic OT is minimised not at ``a == b`` but at a
blurred version of it, so used as a loss it pulls the fit towards a smeared
answer.  The Sinkhorn divergence subtracts the self-terms,

    S(a, b) = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2

which restores the minimum to ``a == b`` while keeping the long reach.

**The blur sets a hard reach, and running past it fails silently.**  The
iterations multiply by ``exp(-c/blur^2)``, which underflows to zero once the
cost is large enough -- and mass separated by a zero kernel cannot be
transported at all, so the answer stops being a distance without saying so.
Two blobs 30 m apart at ``blur=1`` come back as 1404 where the truth is 900.
The limit is ``sqrt(-log(tiny)) * blur``: **26.6 blurs in float64 and 9.3 in
float32**, since single precision underflows so much sooner.
:func:`sinkhorn_divergence` checks the separation against it and refuses rather
than returning a plausible wrong number.

**Separable, or it does not fit.**  The squared-Euclidean cost on a grid splits
as ``(x_i - x_k)^2 + (y_j - y_l)^2``, so the Gibbs kernel is a product of a
kernel along each axis and applying it is two small matrix multiplies rather
than one enormous one.  For a 91 x 233 image the full cost matrix is 4.5e8
entries -- 3.6 GB in float64, per iteration -- against two matrices of 8k and
54k entries for the separable form.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["sinkhorn_divergence", "sinkhorn_potentials",
           "symmetric_potential"]


def _axis_kernels(x: Tensor, y: Tensor, blur: float) -> tuple[Tensor, Tensor]:
    """``exp(-c/eps)`` along each axis, with ``eps = blur^2``."""
    eps = blur * blur
    kx = torch.exp(-((x.reshape(-1, 1) - x.reshape(1, -1)) ** 2) / eps)
    ky = torch.exp(-((y.reshape(-1, 1) - y.reshape(1, -1)) ** 2) / eps)
    return kx, ky


def _apply(kx: Tensor, ky: Tensor, v: Tensor) -> Tensor:
    """``K v`` for the separable kernel: ``[nx, ny] -> [nx, ny]``."""
    return kx @ v @ ky.transpose(-1, -2)


def _normalise(w: Tensor, name: str) -> Tensor:
    if bool((w.detach() < 0).any()):
        raise ValueError(f"{name} must be non-negative")
    total = w.sum()
    if float(total.detach()) <= 0.0:
        raise ValueError(f"{name} has no mass")
    return w / total


def sinkhorn_potentials(a: Tensor, b: Tensor, kx: Tensor, ky: Tensor, *,
                        blur: float, n_iter: int, tol: float
                        ) -> tuple[Tensor, Tensor]:
    """Dual potentials ``(f, g)`` for the entropic problem, **detached**.

    Run without gradient on purpose.  By the envelope theorem the derivative of
    the optimal value with respect to a marginal is that marginal's potential,
    so differentiating ``<f, a> + <g, b>`` with the potentials held fixed gives
    the right gradient -- and costs one iteration's memory instead of all of
    them.  Unrolling a few hundred iterations to get the same number back is a
    way to run out of memory for nothing.
    """
    eps = blur * blur
    with torch.no_grad():
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        floor = torch.finfo(a.dtype).tiny
        for _ in range(n_iter):
            u_prev = u
            u = a / _apply(kx, ky, v).clamp_min(floor)
            v = b / _apply(kx.transpose(-1, -2), ky.transpose(-1, -2), u
                           ).clamp_min(floor)
            if float((u - u_prev).abs().max()) < tol:
                break
        f = eps * torch.log(u.clamp_min(floor))
        g = eps * torch.log(v.clamp_min(floor))
    return f, g


def symmetric_potential(a: Tensor, kx: Tensor, ky: Tensor, *, blur: float,
                        n_iter: int, tol: float) -> Tensor:
    """The single potential of the symmetric problem ``OT(a, a)``, detached.

    The self-terms in a Sinkhorn divergence are symmetric problems, whose two
    potentials are equal -- but the ordinary alternating update never quite
    says so: it refreshes one marginal after the other, so the two come back
    half an iteration apart and their difference is what is left over.  Fed
    into the debiasing that residue does not cancel, and ``S(a, a)`` comes out
    at 6 instead of 0, which is not a subtlety when the whole point of
    debiasing is that the minimum sits at ``a == b``.

    Averaging the update keeps the symmetry exactly.
    """
    eps = blur * blur
    with torch.no_grad():
        u = torch.ones_like(a)
        floor = torch.finfo(a.dtype).tiny
        for _ in range(n_iter):
            u_prev = u
            u = (u * a / _apply(kx, ky, u).clamp_min(floor)).sqrt()
            if float((u - u_prev).abs().max()) < tol:
                break
        return eps * torch.log(u.clamp_min(floor))


def sinkhorn_divergence(a: Tensor, b: Tensor, *, x: Tensor, y: Tensor,
                        blur: float, n_iter: int = 200, tol: float = 1e-9
                        ) -> Tensor:
    """Debiased entropic OT between two grids, ``S(a, b) >= 0``, ``S(a, a) = 0``.

    Args:
        a, b: ``[nx, ny]`` non-negative grids -- an image, or any density.
            Both are normalised to unit mass, so the divergence sees *where*
            the energy is and not how much there is.  For fitting a pose that
            is what you want: the target's level depends on aspect and range,
            and a loss that chased it would fit brightness rather than
            position.
        x, y: ``[nx]`` and ``[ny]`` coordinates, in metres.  Physical units,
            not indices -- a bearing cell and a range cell are different sizes,
            and transporting across one is not the same as across the other.
        blur: the distance mass moves for free, metres.  Sets the reach and the
            resolution: below it arrangements are indistinguishable.
        n_iter, tol: Sinkhorn iterations and the stopping tolerance.

    The gradient flows to ``a`` and ``b`` through the potentials, which is what
    makes this usable as an image loss inside a fit.
    """
    if a.shape != b.shape:
        raise ValueError(f"a is {tuple(a.shape)} but b is {tuple(b.shape)}")
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got {tuple(a.shape)}")
    if a.shape != (int(x.shape[0]), int(y.shape[0])):
        raise ValueError(f"grid is {tuple(a.shape)} but coordinates are "
                         f"{int(x.shape[0])} x {int(y.shape[0])}")
    if blur <= 0.0:
        raise ValueError(f"blur must be positive, got {blur}")

    an, bn = _normalise(a, "a"), _normalise(b, "b")

    # Refuse rather than underflow.  Past sqrt(-log(tiny)) * blur the kernel
    # between the two distributions is exactly zero, no mass can move, and what
    # comes back is not a distance -- it merely looks like one.
    reach = blur * math.sqrt(-math.log(float(torch.finfo(a.dtype).tiny)))
    xs, ys = x.to(a.dtype), y.to(a.dtype)
    ad, bd = an.detach(), bn.detach()
    sep = math.hypot(
        float((ad.sum(1) * xs).sum() - (bd.sum(1) * xs).sum()),
        float((ad.sum(0) * ys).sum() - (bd.sum(0) * ys).sum()))
    if sep > reach:
        raise ValueError(
            f"the two distributions are {sep:.1f} m apart but blur={blur:g} "
            f"reaches only {reach:.1f} m in {a.dtype} -- the kernel underflows "
            f"to zero between them and the result would not be a distance. "
            f"Use blur >= {sep / math.sqrt(-math.log(float(torch.finfo(a.dtype).tiny))):.2f}")
    kx, ky = _axis_kernels(x.to(a.dtype), y.to(a.dtype), blur)

    f_ab, g_ab = sinkhorn_potentials(an, bn, kx, ky, blur=blur, n_iter=n_iter,
                                     tol=tol)
    f_aa = symmetric_potential(an, kx, ky, blur=blur, n_iter=n_iter, tol=tol)
    g_bb = symmetric_potential(bn, kx, ky, blur=blur, n_iter=n_iter, tol=tol)
    # Feydy's form: the self-terms cancel the entropic bias, and with the
    # potentials detached this differentiates correctly in both marginals.
    return ((f_ab - f_aa) * an).sum() + ((g_ab - g_bb) * bn).sum()
