r"""Triangle-mesh targets: Kirchhoff scattering straight off the geometry.

A :class:`hydropt.targets.ScatteringPattern` answers one question -- given an
incident and a scattered direction, what is the bistatic cross-section? -- and a
mesh can answer it directly, by integrating the Kirchhoff (physical optics)
surface integral over every facet and summing coherently:

.. math::
    f(\hat{k}_i, \hat{k}_s) = \frac{1}{\lambda} \sum_{\text{facets}}
        (\hat{n}\cdot\hat{k}_i)_+ \int_{T} e^{i\mathbf{q}\cdot\mathbf{r}}\,dA,
    \qquad \mathbf{q} = k(\hat{k}_s - \hat{k}_i), \qquad \sigma = |f|^2

So a mesh needs **no new propagation machinery**.  It is a pattern like any
other: one highlight, one return fan, the same cost as a sphere.  Everything
`hydropt` already does -- two-way tracing, multipath, beamforming, gradients --
works unchanged.

**The facet integral has to be exact.**  The tempting shortcut is to treat a
facet as a point of amplitude ``A e^{i q.c}``, which is what most quick
implementations do.  It is valid only for ``|q| d << 1``, and at 100 kHz with
centimetre facets ``|q| d`` is about 25 radians: that approximation came out
**16x too high even at 82,000 facets**, converging only as fast as facet area.
With the exact integral the same sphere is right to 0.1 dB at 20,000.

The exact integral over a triangle with vertices :math:`p_j` is :math:`2A` times
the second divided difference of :math:`\exp` at the vertex phases
:math:`z_j = i\,\mathbf{q}\cdot p_j`.  Evaluating *that* stably is the whole
difficulty -- the naive form divides by vertex-phase differences that vanish
whenever a facet lies near a phase front, which is exactly where the specular
return comes from.  :func:`triangle_phase_integral` handles it.

**What this does not model.**  Facets are culled by their own normal, which is
right for a convex body and is all a hull needs from outside.  There is no
ray-casting between facets, so a mesh that shadows itself -- a superstructure
over a deck, a propeller behind a skeg -- will have the hidden parts still
contributing.  Physical optics also has no edge diffraction, so it understates
grazing returns and takes an edge-on facet to exactly zero.

Facets must be small enough to resolve the surface's *curvature*, not its phase:
about :math:`\sqrt{\lambda R}/3` for a surface of radius :math:`R`.  At 100 kHz
on a 1 m radius that is 4 cm.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from .targets import ExtendedTarget, ScatteringPattern

__all__ = [
    "load_obj",
    "facet_geometry",
    "triangle_phase_integral",
    "MeshScattering",
    "visible_facets",
    "mesh_target",
    "boat_hull_mesh",
    "icosphere",
]

_EPS = 1e-30


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def load_obj(path: str | Path) -> tuple[Tensor, Tensor]:
    """Read vertices and triangles from a Wavefront ``.obj`` file.

    Only ``v`` and ``f`` records are read; normals, texture coordinates,
    materials and groups are ignored, because the scattering integral derives
    the normal from the winding and needs nothing else.  Polygons with more than
    three vertices are fan-triangulated.  Indices may be negative (relative), as
    the format allows.

    Returns:
        ``(vertices [V, 3], faces [F, 3])`` with ``faces`` of dtype ``long``.
    """
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, "r") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for j in range(1, len(idx) - 1):  # fan-triangulate
                    faces.append((idx[0], idx[j], idx[j + 1]))
    if not verts or not faces:
        raise ValueError(f"{path}: found {len(verts)} vertices and {len(faces)} faces")
    return (torch.tensor(verts, dtype=torch.get_default_dtype()),
            torch.tensor(faces, dtype=torch.long))


def facet_geometry(vertices: Tensor, faces: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per-facet ``(centroid [F,3], unit normal [F,3], area [F])``.

    The normal follows the winding by the right-hand rule, so an outward-wound
    mesh gives outward normals -- which is what the culling in
    :class:`MeshScattering` assumes.  Differentiable in ``vertices``.
    """
    tri = vertices[faces]                       # [F, 3, 3]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = torch.linalg.cross(e1, e2, dim=-1)
    norm = cross.norm(dim=-1)
    area = 0.5 * norm
    normal = cross / norm.clamp_min(_EPS).unsqueeze(-1)
    return tri.mean(dim=1), normal, area


# --------------------------------------------------------------------------- #
# The exact facet integral
# --------------------------------------------------------------------------- #
def _sinhc(x: Tensor) -> Tensor:
    """``sinh(x)/x``, exact at ``x = 0``, stable near it."""
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    series = 1.0 + x * x / 6.0 + x ** 4 / 120.0
    return torch.where(small, series, torch.sinh(safe) / safe)


def _collapse_tol(dtype: torch.dtype) -> float:
    """Vertex-phase gap below which the divided difference goes to the series.

    Two errors race as the gap ``g`` closes.  The nested divided difference
    subtracts two nearly equal terms and divides by ``g``, so it loses
    ``eps / g``; the series truncates at the cubic term, so it loses ``~g^3``.
    They cross at ``g = eps^(1/4)``, and that is the tolerance -- which is
    ``1.2e-4`` in float64 but ``1.8e-2`` in float32.  A single hard-coded
    constant cannot serve both: at float64's ``1e-5``, float32 keeps only two
    digits through the subtraction, and the facet integral loses up to 1e-4
    relative right where the specular return lives.
    """
    return float(torch.finfo(dtype).eps) ** 0.25


def _dd1(a: Tensor, b: Tensor) -> Tensor:
    """First divided difference of ``exp``: ``(e^a - e^b)/(a - b)``.

    Written as ``e^{(a+b)/2} sinhc((a-b)/2)``, which has no cancellation and is
    exact in the limit ``a -> b``, where the quotient form is 0/0.
    """
    return torch.exp(0.5 * (a + b)) * _sinhc(0.5 * (a - b))


def triangle_phase_integral(tri: Tensor, q: Tensor, *, area: Tensor | None = None
                            ) -> Tensor:
    r"""``\int_T exp(i q.r) dA`` over triangles, exactly.

    Args:
        tri: triangle vertices, ``[F, 3, 3]``.
        q: wave-vector difference, broadcastable to ``[..., F, 3]``.
        area: per-facet areas ``[F]``, if already computed.

    Returns:
        complex ``[..., F]``.

    In barycentric coordinates the integrand is ``exp(sum_j z_j lambda_j)`` with
    ``z_j = i q.p_j``, whose integral over the unit simplex is the second divided
    difference of ``exp`` at the three ``z_j``.  The textbook form
    ``sum_j e^{z_j} / prod_{k != j}(z_j - z_k)`` is useless numerically: it
    divides by vertex-phase differences that go to zero exactly where the facet
    is near a phase front, which is where the specular return lives.

    This evaluates it as a nested divided difference instead, and -- since the
    result is symmetric in the three nodes -- always puts the **widest-separated
    pair** in the denominator, so the division is as well-conditioned as the
    facet allows.  Only when all three phases collapse together (a facet lying
    in a phase front, where the integral is just the area) does it fall back to
    a series.
    """
    if area is None:
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        area = 0.5 * torch.linalg.cross(e1, e2, dim=-1).norm(dim=-1)
    phase = (tri * q.unsqueeze(-2)).sum(-1)      # [..., F, 3] real
    z = torch.complex(torch.zeros_like(phase), phase)

    z0, z1, z2 = z[..., 0], z[..., 1], z[..., 2]
    gaps = torch.stack([(z0 - z1).abs(), (z1 - z2).abs(), (z0 - z2).abs()], dim=-1)
    widest = gaps.argmax(dim=-1)

    # Reorder to (a, c) = the widest-separated pair, b = the remaining node.
    # dd2 is symmetric, so this changes conditioning only, never the value.
    a = torch.where(widest == 1, z1, z0)
    c = torch.where(widest == 0, z1, z2)
    b = torch.where(widest == 0, z2, torch.where(widest == 1, z0, z1))

    # The widest gap IS |a - c| by construction, so one tolerance decides both
    # the branch and whether the denominator is safe to divide by.
    tol = _collapse_tol(phase.dtype)
    collapsed = gaps.max(dim=-1).values < tol
    denom = a - c
    safe = torch.where(collapsed, torch.ones_like(denom), denom)
    nested = (_dd1(a, b) - _dd1(b, c)) / safe

    # All three nodes together: the facet lies in a phase front.
    centre = z.mean(dim=-1)
    d = z - centre.unsqueeze(-1)
    # Expanding about the mean node: with lambda over the unit simplex,
    # int 1 = 1/2, int lambda_j = 1/6, int lambda_j^2 = 1/12 and
    # int lambda_j lambda_k = 1/24, which puts BOTH quadratic terms at 1/48.
    # (Both vanish analytically since d sums to zero; they are kept because it
    # does not numerically.)  The truncation is then O(d^3) rather than O(d^2),
    # which is what lets the tolerance above sit where float32 needs it.
    series = torch.exp(centre) * (0.5
                                  + d.sum(-1) / 6.0
                                  + (d * d).sum(-1) / 48.0
                                  + d.sum(-1) ** 2 / 48.0)
    dd2 = torch.where(collapsed, series, nested)
    return 2.0 * area.to(dd2.dtype) * dd2


# --------------------------------------------------------------------------- #
# The pattern
# --------------------------------------------------------------------------- #
def _view_basis(direction: Tensor) -> tuple[Tensor, Tensor]:
    """Two unit vectors spanning the plane perpendicular to ``direction``."""
    d = direction / direction.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    # Pick the axis least aligned with d, so the cross product is well conditioned.
    alt = torch.zeros_like(d)
    alt.scatter_(-1, d.abs().argmin(dim=-1, keepdim=True), 1.0)
    u = torch.linalg.cross(d, alt, dim=-1)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    v = torch.linalg.cross(d, u, dim=-1)
    return u, v


def visible_facets(centroid: Tensor, view: Tensor, cell: float,
                   *, tolerance: float, normal: Tensor | None = None,
                   grid: int = 512) -> Tensor:
    r"""Which facets are not hidden behind another, seen along ``view``.

    A depth buffer rather than ray-casting.  Casting every facet against every
    other is :math:`O(F^2)` per direction -- 225 million tests for a
    15,000-facet hull, and ``compose_arrivals`` asks for hundreds of directions
    -- whereas projecting the centroids onto the plane perpendicular to the line
    of sight, binning them, and keeping the nearest per bin is :math:`O(F)`.

    Args:
        centroid: ``[F, 3]`` facet centroids.
        view: ``[P, 3]`` directions **from the observer towards the surface**.
            For an incident direction that is the direction of travel; for a
            scattered one it is its negation, because the receiver is downstream.
        cell: bin size (m) on the view plane.  Should be about one facet across:
            much larger and facets on the same surface compete with each other,
            much smaller and nothing ever occludes anything.
        tolerance: depth margin (m).  A facet is culled only when something sits
            more than this in front of it.  **This is what keeps the answer
            honest**: with a strict nearest-per-bin rule, two facets of one
            continuous lit surface landing in the same bin would knock each
            other out and the surface would lose energy for no physical reason.
            Occlusion should only remove what is genuinely behind something
            else, which on a real body means a separation of many facets.
        normal: ``[F, 3]`` facet normals.  Given, the margin is widened where
            the surface is seen obliquely, by ``cell / |n.d|`` -- the depth a
            bin spans on a surface tilted that far.  Without it a convex body
            culls its own limb, where the surface runs nearly along the line of
            sight: on a sphere that cost 0.26% of the return, which is small but
            is not the zero a convex body is owed.
        grid: bins per axis; the projection is wrapped into this many, so a
            body far larger than ``grid * cell`` will alias.  It costs no
            memory -- the depth buffer is compacted onto the bins that facets
            actually fall in -- so it can be raised freely to avoid aliasing.

    Returns:
        ``[P, F]`` boolean, detached -- visibility is piecewise constant, so it
        carries no useful gradient, exactly as back-face culling does not.
    """
    c = centroid.detach()
    d = view.detach().reshape(-1, 3)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    u, v = _view_basis(d)

    depth = c @ d.T                                   # [F, P], along the view
    a = torch.round((c @ u.T) / cell).long()
    b = torch.round((c @ v.T) / cell).long()
    key = (a % grid) * grid + (b % grid)              # [F, P]

    p = d.shape[0]
    big = torch.finfo(depth.dtype).max
    flat = (torch.arange(p, device=c.device).reshape(1, -1) * (grid * grid)
            + key).reshape(-1)
    # Compact the keys before reducing.  A dense buffer over the whole index
    # space is `p * grid^2` cells to hold at most `p * F` occupied ones, and
    # that is not a constant factor: 9,216 directions at the default grid asked
    # for 19 GB and fell over, while the facets actually landing in it were
    # 23 million.  Reducing into the unique keys is the same answer with memory
    # set by the data instead of by the resolution of the bins.
    uniq, inverse = torch.unique(flat, return_inverse=True)
    nearest = torch.full((int(uniq.numel()),), big, dtype=depth.dtype,
                         device=c.device)
    nearest = nearest.scatter_reduce(0, inverse, depth.reshape(-1), reduce="amin",
                                     include_self=True)
    front = nearest[inverse].reshape(depth.shape)
    margin = tolerance
    if normal is not None:
        n = normal.detach()
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        obliquity = (n @ d.T).abs().clamp_min(1e-3)   # [F, P]
        margin = tolerance + cell / obliquity
    return (depth <= front + margin).T.contiguous()  # [P, F]


def _facet_block(tri: Tensor, normal: Tensor, area: Tensor, ki: Tensor,
                 q: Tensor, visible: Tensor | None = None) -> Tensor:
    """One block of facets' contribution to the scattering amplitude, ``[P, B]``.

    Split out so it can be wrapped in :func:`torch.utils.checkpoint.checkpoint`:
    the block's intermediates are the memory cost, and they are recomputed in
    the backward pass rather than kept.
    """
    # Lit facets only: a facet turned away from the source does not radiate.
    # Exactly zero behind, so no gradient there either -- the same convention
    # PlateScattering's obliquity factor follows.
    lit = (-(ki @ normal.T)).clamp_min(0.0)                      # [P, f]
    if visible is not None:
        lit = lit * visible.to(lit.dtype)
    integral = triangle_phase_integral(tri, q.unsqueeze(-2), area=area)
    return (lit.unsqueeze(1).to(integral.dtype) * integral).sum(-1)


def _diffuse_block(normal: Tensor, area: Tensor, ki: Tensor, ks: Tensor,
                   visible: Tensor | None = None) -> Tensor:
    """One block of facets' INCOHERENT contribution, ``[P]``.

    Lambert on each facet: ``mu A cos(theta_i) cos(theta_s)``, summed in power
    with no phase, so it survives where the coherent sum cancels.  The ``mu``
    is applied by the caller.  Frequency-flat, which is what "diffuse" means
    here -- the roughness responsible for it is assumed fine compared with
    every wavelength in the band.
    """
    ci = (-(ki @ normal.T)).clamp_min(0.0)                       # [P, f]
    cs = (ks @ normal.T).clamp_min(0.0)                          # [P, f]
    w = ci * cs * area.reshape(1, -1)
    if visible is not None:
        w = w * visible.to(w.dtype)
    return w.sum(-1)


class MeshScattering(ScatteringPattern):
    r"""Bistatic cross-section of a triangle mesh, by coherent physical optics.

    Args:
        vertices: ``[V, 3]`` in the **body frame**, metres.
        faces: ``[F, 3]`` triangle indices, wound so normals point outward.
        sound_speed: to turn frequency into wavelength (m/s).
        learnable: register ``vertices`` as a parameter, so a loss on the image
            can deform the shape.  The faces are fixed -- topology is not
            something a gradient can move.
        diffuse_db: Lambert scattering strength ``mu`` of the surface, in dB.
            ``None`` (the default) leaves the body a pure mirror, which is what
            physical optics on a smooth mesh describes and is wrong for
            anything built by people.  A smooth analytic hull measured +14.4 dB
            of target strength at beam aspect and -21.8 dB at 58 degrees off
            it: 36 dB, because away from specular there is nothing left to
            return.  A real vessel carries ribs, plating seams, a rudder, a
            prop and internal structure, loses perhaps 10 to 20 dB off beam
            aspect rather than 36, and stays detectable at every heading.  This
            term is what represents that: ``mu A cos(theta_i) cos(theta_s)``
            per facet, summed in POWER rather than amplitude, so it does not
            cancel where the coherent sum does.  The same Lambert form the
            seabed uses, and frequency-flat for the same reason.
        learnable_diffuse: register ``diffuse_db`` as a parameter, so a loss on
            the image can fit the surface's roughness.
        occlusion: cull facets hidden behind other facets, by depth buffer
            (:func:`visible_facets`).  Without it a mesh that shadows itself --
            a superstructure over a deck, a propeller behind a skeg, the far
            wall of anything concave -- keeps contributing from the hidden
            parts.  A convex body has nothing to occlude, so this is a no-op
            for a sphere or a faired hull and costs only the pass.
            **Bistatic needs both ends**: a facet must be visible from the
            source *and* from the receiver, which are the same test only when
            they coincide.
        occlusion_cell, occlusion_tolerance: bin size and depth margin (m),
            both defaulting from the mesh's own median edge length.  See
            :func:`visible_facets` for why the margin matters.
        facet_chunk: facets summed per block.  The working set is
            ``pairs x bands x chunk``, so this trades memory against Python
            iterations; lower it for a big mesh or many directions.
        checkpoint: recompute each block during the backward pass instead of
            keeping its intermediates.  Without this, memory grows with the
            *whole* mesh -- a 13,000-facet hull against a few hundred direction
            pairs exhausted several GB and was killed -- because every block's
            temporaries stay alive for backward.  With it, memory is set by
            ``facet_chunk`` alone and the mesh can be as fine as the physics
            needs, at approximately one extra forward evaluation.

    Validated against the closed forms the rest of `hydropt` uses: a faceted
    sphere reproduces :class:`hydropt.targets.CurvedSurfaceScattering`'s
    ``sigma = a^2/4``, and one flat facet at normal incidence reproduces
    :class:`hydropt.targets.PlateScattering`'s ``(A/lambda)^2`` exactly.
    """

    def __init__(self, vertices: Tensor, faces: Tensor, *,
                 sound_speed: float = 1500.0, learnable: bool = False,
                 diffuse_db: float | None = None,
                 learnable_diffuse: bool = True,
                 facet_chunk: int = 512, checkpoint: bool = True,
                 occlusion: bool = True, occlusion_cell: float | None = None,
                 occlusion_tolerance: float | None = None) -> None:
        super().__init__()
        v = torch.as_tensor(vertices, dtype=torch.get_default_dtype()).reshape(-1, 3)
        f = torch.as_tensor(faces, dtype=torch.long).reshape(-1, 3)
        if int(f.max()) >= v.shape[0] or int(f.min()) < 0:
            raise ValueError(f"face indices out of range for {v.shape[0]} vertices")
        if learnable:
            self.vertices = nn.Parameter(v)
        else:
            self.register_buffer("vertices", v)
        self.register_buffer("faces", f)
        self.register_buffer("sound_speed", torch.as_tensor(float(sound_speed)))
        if diffuse_db is None:
            self.diffuse_db = None
        else:
            mu = torch.as_tensor(float(diffuse_db))
            if learnable_diffuse:
                self.diffuse_db = nn.Parameter(mu)
            else:
                self.register_buffer("diffuse_db", mu)
        self.facet_chunk = int(facet_chunk)
        self.checkpoint = bool(checkpoint)
        self.occlusion = bool(occlusion)
        # Sized from the mesh itself: a bin about one facet across, and a depth
        # margin of a few bins so a continuous surface never shadows itself.
        tri0 = v[f]
        edge = float((tri0[:, 1] - tri0[:, 0]).norm(dim=-1).median())
        self.occlusion_cell = float(occlusion_cell) if occlusion_cell else edge
        self.occlusion_tolerance = (float(occlusion_tolerance)
                                    if occlusion_tolerance
                                    else 4.0 * self.occlusion_cell)

    @property
    def n_facets(self) -> int:
        return int(self.faces.shape[0])

    def centroid(self) -> Tensor:
        """Area-weighted centroid of the surface, ``[3]`` -- where to hang it."""
        _, _, area = facet_geometry(self.vertices, self.faces)
        tri = self.vertices[self.faces]
        return (tri.mean(1) * area.unsqueeze(-1)).sum(0) / area.sum().clamp_min(_EPS)

    def cross_section(self, incident: Tensor, scattered: Tensor,
                      freqs_khz: Tensor) -> Tensor:
        dtype = freqs_khz.dtype
        incident = incident.to(dtype)
        scattered = scattered.to(dtype)
        shape = torch.broadcast_shapes(incident.shape[:-1], scattered.shape[:-1])
        ki = incident.expand(*shape, 3).reshape(-1, 3)
        ks = scattered.expand(*shape, 3).reshape(-1, 3)

        lam = self.sound_speed.to(dtype) / (freqs_khz * 1.0e3)        # [B]
        k = 2.0 * math.pi / lam.clamp_min(_EPS)
        # q = k (ks - ki), one per (pair, band)
        q = (ks - ki).unsqueeze(1) * k.reshape(1, -1, 1)              # [P, B, 3]

        # Cast the geometry to the caller's dtype, as every other pattern does:
        # a mesh built under one default dtype must still work when asked a
        # question in another.
        verts = self.vertices.to(dtype)
        tri = verts[self.faces]                                       # [F, 3, 3]
        centroid, normal, area = facet_geometry(verts, self.faces)

        seen = None
        if self.occlusion:
            # From the source, and from the receiver.  `ki` travels towards the
            # surface so it is already a view direction; `ks` travels away, so
            # the receiver's line of sight is its negation.
            seen = (visible_facets(centroid, ki, self.occlusion_cell,
                                   tolerance=self.occlusion_tolerance,
                                   normal=normal)
                    & visible_facets(centroid, -ks, self.occlusion_cell,
                                     tolerance=self.occlusion_tolerance,
                                     normal=normal))

        complex_dtype = (torch.complex128 if dtype == torch.float64
                         else torch.complex64)
        total = torch.zeros(q.shape[0], q.shape[1], dtype=complex_dtype,
                            device=q.device)
        diffuse = torch.zeros(q.shape[0], dtype=dtype, device=q.device)
        for start in range(0, tri.shape[0], self.facet_chunk):
            stop = start + self.facet_chunk
            t, n, a = tri[start:stop], normal[start:stop], area[start:stop]
            vis = None if seen is None else seen[:, start:stop]
            if self.checkpoint and torch.is_grad_enabled() and q.requires_grad:
                part = torch.utils.checkpoint.checkpoint(
                    _facet_block, t, n, a, ki, q, vis, use_reentrant=False)
            else:
                part = _facet_block(t, n, a, ki, q, vis)
            total = total + part
            if self.diffuse_db is not None:
                diffuse = diffuse + _diffuse_block(n, a, ki, ks, vis)

        amp = total / lam.reshape(1, -1).to(total.dtype)
        sigma = amp.real ** 2 + amp.imag ** 2
        if self.diffuse_db is not None:
            mu = 10.0 ** (self.diffuse_db.to(dtype) / 10.0)
            sigma = sigma + (mu * diffuse).unsqueeze(-1)
        return sigma.reshape(*shape, -1)

    def extra_repr(self) -> str:
        return (f"{int(self.vertices.shape[0])} vertices, {self.n_facets} facets, "
                f"chunk={self.facet_chunk}, "
                f"occlusion={'on' if self.occlusion else 'off'}")


# --------------------------------------------------------------------------- #
# What the body hides: segments through a mesh
# --------------------------------------------------------------------------- #
def segment_mesh_transmission(starts: Tensor, ends: Tensor, vertices: Tensor,
                              faces: Tensor, *, facet_chunk: int = 2048
                              ) -> Tensor:
    """Which segments reach their far end without passing through the mesh.

    ``[N]``, ``1`` where the straight segment from ``starts[i]`` to ``ends[i]``
    misses the body and ``0`` where it crosses it.  This is what casts an
    acoustic shadow: a body on the seabed blocks the path to the patches behind
    it, and the dark patch that leaves is what an operator reads the body's
    height from.

    Args:
        starts, ends: ``[N, 3]`` endpoints, world frame, metres.
        vertices, faces: the occluding mesh, **world frame** -- place the body
            before calling; this knows nothing of a target's pose.
        facet_chunk: facets per block, to bound the ``[N, F]`` working set.

    Segments are tested with Moller-Trumbore, **without back-face culling**:
    the question is whether the path crosses the surface at all, and while a
    body seen from outside presents its front facets, a patch under an overhang
    or a source inside a hull presents its back ones.  Only segments whose
    closest approach to the mesh's bounding sphere falls inside it are tested;
    for a 12 m body at 85 m in a 120-degree fan that is a few percent of the
    rays, and the rest cost one distance each.

    **The mask is a step, so it has no gradient.**  The shadow's edge is
    precisely the observable that carries the body's height, and a hard mask
    puts a zero derivative on it, so a height cannot be fitted through this.
    Softening it is not a matter of blending each facet's edges: the facets
    along a silhouette are geometrically correlated, and treating them as
    independent over-counts them badly (a ray a hand's breadth outside a
    tessellated sphere comes back 99% occluded).  A differentiable edge needs a
    distance to the body's *silhouette*, which is a separate piece of work.
    """
    starts = starts.reshape(-1, 3)
    ends = ends.reshape(-1, 3)
    if starts.shape != ends.shape:
        raise ValueError(f"starts is {tuple(starts.shape)} but ends is "
                         f"{tuple(ends.shape)}")
    v = vertices.reshape(-1, 3)
    f = faces.reshape(-1, 3)
    out = torch.ones(starts.shape[0], dtype=starts.dtype, device=starts.device)
    if f.shape[0] == 0 or starts.shape[0] == 0:
        return out

    # Bounding sphere, then each segment's closest approach to its centre.
    lo, hi = v.min(dim=0).values, v.max(dim=0).values
    centre = 0.5 * (lo + hi)
    radius = (v - centre).norm(dim=-1).max()
    seg = ends - starts
    length2 = (seg * seg).sum(-1).clamp_min(_EPS)
    t_near = (((centre - starts) * seg).sum(-1) / length2).clamp(0.0, 1.0)
    miss = (starts + t_near.unsqueeze(-1) * seg - centre).norm(dim=-1)
    near = miss <= radius
    if not bool(near.any()):
        return out

    p0, d = starts[near], seg[near]
    blocked = torch.zeros(p0.shape[0], dtype=torch.bool, device=p0.device)
    for i in range(0, f.shape[0], facet_chunk):
        tri = v[f[i:i + facet_chunk]]                        # [C, 3, 3]
        n, c = p0.shape[0], tri.shape[0]
        v0 = tri[:, 0].unsqueeze(0).expand(n, c, 3)
        e1 = (tri[:, 1] - tri[:, 0]).unsqueeze(0).expand(n, c, 3)
        e2 = (tri[:, 2] - tri[:, 0]).unsqueeze(0).expand(n, c, 3)
        dd = d.unsqueeze(1).expand(n, c, 3)
        pv = torch.linalg.cross(dd, e2, dim=-1)
        det = (e1 * pv).sum(-1)                              # [N, C]
        # A segment parallel to a facet is the only one with nothing to say.
        parallel = det.abs() < _EPS
        safe = torch.where(parallel, torch.ones_like(det), det)
        sv = p0.unsqueeze(1) - v0
        u = (sv * pv).sum(-1) / safe
        qv = torch.linalg.cross(sv, e1, dim=-1)
        w = (dd * qv).sum(-1) / safe
        t = (e2 * qv).sum(-1) / safe
        hit = ((t > 0.0) & (t < 1.0) & ~parallel
               & (u >= 0.0) & (w >= 0.0) & (u + w <= 1.0))
        blocked = blocked | hit.any(dim=1)

    out = out.clone()
    out[near] = (~blocked).to(out.dtype)
    return out


# --------------------------------------------------------------------------- #
# Building a target from a mesh
# --------------------------------------------------------------------------- #
def mesh_target(vertices: Tensor, faces: Tensor, *,
                position: tuple[float, float, float] | Tensor = (0.0, 0.0, 0.0),
                yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                n_patches: int = 1, split_axis: int | None = None,
                sound_speed: float = 1500.0,
                learnable: bool = True, learnable_shape: bool = False,
                diffuse_db: float | None = None,
                facet_chunk: int = 512, checkpoint: bool = True,
                occlusion: bool = True) -> ExtendedTarget:
    """An :class:`~hydropt.targets.ExtendedTarget` whose scattering is a mesh.

    Args:
        vertices, faces: the mesh, body frame, metres.
        position, yaw, pitch, roll: where to put it, as for ``ExtendedTarget``.
        n_patches: how many highlights to split the mesh into.  **This is the
            one real choice.**  ``1`` treats the whole body as a point scatterer
            carrying the mesh's full aspect dependence -- cheapest, and right
            when the body is small against the range resolution.  Larger values
            partition the facets along the body's longest axis, giving each
            patch its own position, so the target acquires *extent* in range and
            bearing, at one return fan per patch.
        split_axis: which body axis to partition along (0=x, 1=y, 2=z).  The
            default picks the mesh's longest extent, which is what you want for
            one hull.  **It is the wrong choice for a mesh of separate bodies**
            -- a catamaran's longest extent is still its length, so every patch
            would straddle both hulls and sit midway between them, at a range
            that belongs to neither.  Pass the axis that separates the bodies
            (``split_axis=1`` for hulls set apart athwartships) and each patch
            then belongs to one hull.
        learnable: position and orientation are parameters.
        learnable_shape: the vertices are parameters too, so a loss on the
            image reaches the geometry.
        sound_speed, diffuse_db, facet_chunk, checkpoint, occlusion: passed to
            :class:`MeshScattering`.  ``diffuse_db`` is the one that decides
            whether the body is visible anywhere but beam-on.

    **Two bodies in one patch also interfere as though they were one.**  A
    patch's facet phases are referred to its own centroid under a plane-wave
    front, so a patch spanning bodies ``d`` apart at range ``R`` carries a
    curvature error ``d^2 / 8R``; across a catamaran's 5 m gap at 85 m that is
    2.4 wavelengths at 100 kHz, and the resulting cancellation is spurious.
    Splitting across the gap keeps each body's phases referred to its own
    centroid, where the same error is small.

    **Splitting limits occlusion to within a patch.**  Each patch is its own
    :class:`MeshScattering` and knows nothing of the others, so with
    ``n_patches > 1`` one part of the body can no longer hide another -- a
    superstructure in patch 3 will not shadow a deck in patch 4.  The split is
    along the body's longest axis, so what survives is occlusion between facets
    that are near each other, and what is lost is occlusion between distant
    parts.  Use ``n_patches=1`` when shadowing between parts of the body
    matters more than its extent in the image does.

    Each patch's highlight sits at its own facets' area-weighted centroid, and
    its pattern sees only its own facets, so the coherent sum within a patch and
    the coherent sum across patches (which ``compose_arrivals`` already does,
    with true path phase) together reproduce the whole body.
    """
    v = torch.as_tensor(vertices, dtype=torch.get_default_dtype()).reshape(-1, 3)
    f = torch.as_tensor(faces, dtype=torch.long).reshape(-1, 3)
    if n_patches < 1:
        raise ValueError(f"n_patches must be at least 1, got {n_patches}")
    n_patches = min(n_patches, int(f.shape[0]))

    centroid, _, _ = facet_geometry(v, f)
    if n_patches == 1:
        groups = [f]
    else:
        # Split along the body's longest extent: the axis that actually buys
        # resolvable separation in the image.
        if split_axis is None:
            extent = centroid.max(0).values - centroid.min(0).values
            axis = int(extent.argmax())
        elif split_axis not in (0, 1, 2):
            raise ValueError(f"split_axis must be 0, 1 or 2, got {split_axis}")
        else:
            axis = int(split_axis)
        order = centroid[:, axis].argsort()
        groups = [f[chunk] for chunk in torch.chunk(order, n_patches)
                  if chunk.numel() > 0]

    offsets, patterns = [], []
    for group in groups:
        pattern = MeshScattering(v, group, sound_speed=sound_speed,
                                 learnable=learnable_shape,
                                 diffuse_db=diffuse_db,
                                 facet_chunk=facet_chunk, checkpoint=checkpoint,
                                 occlusion=occlusion)
        offsets.append(pattern.centroid().detach().reshape(1, 3))
        patterns.append(pattern)
    return ExtendedTarget(torch.cat(offsets, dim=0), patterns, position=position,
                          yaw=yaw, pitch=pitch, roll=roll, learnable=learnable)


# --------------------------------------------------------------------------- #
# Meshes to try it on
# --------------------------------------------------------------------------- #
def icosphere(subdivisions: int = 3, radius: float = 1.0
              ) -> tuple[Tensor, Tensor]:
    """A geodesic sphere, outward-wound -- the reference body for validation.

    Its cross-section is known exactly (``sigma = a^2/4`` at every aspect), which
    is what makes it the right thing to check a facet integrator against.
    """
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = torch.tensor(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
         [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]],
        dtype=torch.get_default_dtype())
    faces = torch.tensor(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
        dtype=torch.long)
    verts = verts / verts.norm(dim=-1, keepdim=True)
    for _ in range(int(subdivisions)):
        cache: dict[tuple[int, int], int] = {}
        rows = list(verts)
        new_faces = []

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in cache:
                p = rows[i] + rows[j]
                rows.append(p / p.norm().clamp_min(_EPS))
                cache[key] = len(rows) - 1
            return cache[key]

        for a, b, c in faces.tolist():
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = torch.stack(rows)
        faces = torch.tensor(new_faces, dtype=torch.long)
    return verts * float(radius), faces


def cylinder_mesh(length: float = 2.0, radius: float = 0.5, *,
                  n_axial: int = 24, n_around: int = 32, caps: bool = True
                  ) -> tuple[Tensor, Tensor]:
    """A closed circular cylinder along x, outward-wound.

    The shape most bottom objects are modelled as -- a mine-like object, a
    length of pipeline, a piling -- and one with a closed-form cross-section to
    check against: broadside, physical optics gives ``sigma = a L^2 / 2 lambda``
    in the backscattering convention this package uses throughout (the radar
    ``2 pi a L^2 / lambda`` over ``4 pi``, the same factor that turns a sphere's
    ``pi a^2`` into ``a^2 / 4``).  It is the cylinder's counterpart to the
    plate's ``(A / lambda)^2``.

    Tessellate it for the wavelength, not for the picture: flat facets sample a
    curved surface, and the specular return is 11 percent low at a facet arc of
    one wavelength.  Half a wavelength is within a percent.

    Args:
        length, radius: metres.
        n_axial, n_around: rings along the axis, and points around each.
        caps: close the ends.  An open tube scatters from its inside as well,
            which is right for a pipe and wrong for a body.
    """
    dt = torch.get_default_dtype()
    if length <= 0.0 or radius <= 0.0:
        raise ValueError(f"length and radius must be positive, got "
                         f"{length} and {radius}")
    if n_axial < 2 or n_around < 3:
        raise ValueError(f"need at least 2 rings of at least 3 points, got "
                         f"{n_axial} and {n_around}")
    x = torch.linspace(-0.5 * length, 0.5 * length, n_axial, dtype=dt)
    theta = torch.arange(n_around, dtype=dt) * (2.0 * math.pi / n_around)
    ring = torch.stack([torch.zeros_like(theta), radius * theta.cos(),
                        radius * theta.sin()], dim=-1)
    verts = (ring.unsqueeze(0) + torch.stack(
        [x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1).unsqueeze(1)
    ).reshape(-1, 3)

    faces = []
    for i in range(n_axial - 1):
        for j in range(n_around):
            k = (j + 1) % n_around
            a, b = i * n_around + j, i * n_around + k
            c, d = a + n_around, b + n_around
            # theta counterclockwise in y-z and x increasing put the right-hand
            # normal radially outward for this winding.
            faces += [[a, b, c], [b, d, c]]

    if caps:
        hub_lo, hub_hi = verts.shape[0], verts.shape[0] + 1
        verts = torch.cat([verts,
                           torch.tensor([[-0.5 * length, 0.0, 0.0],
                                         [0.5 * length, 0.0, 0.0]], dtype=dt)])
        top = (n_axial - 1) * n_around
        for j in range(n_around):
            k = (j + 1) % n_around
            faces += [[hub_lo, k, j],                     # -x cap faces -x
                      [hub_hi, top + j, top + k]]         # +x cap faces +x

    tri = torch.tensor(faces, dtype=torch.long)
    _, _, area = facet_geometry(verts, tri)
    floor = float(torch.finfo(dt).eps) ** 0.5
    return verts, tri[area > floor * float(area.max())]


def _interp1d(xq: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """Linear interpolation of ``y(x)`` at ``xq``; ``x`` non-decreasing."""
    idx = torch.searchsorted(x, xq).clamp(1, x.shape[0] - 1)
    x0, x1 = x[idx - 1], x[idx]
    t = ((xq - x0) / (x1 - x0).clamp_min(1e-30)).clamp(0.0, 1.0)
    return y[idx - 1] + t * (y[idx] - y[idx - 1])


def boat_hull_mesh(length: float = 12.0, beam: float = 3.0, draft: float = 1.0,
                   *, n_long: int = 40, n_around: int = 16,
                   transom: float = 0.62, deadrise_stern: float = 4.5,
                   deadrise_bow: float = 1.25) -> tuple[Tensor, Tensor]:
    """A displacement hull's **wetted** surface: the part a sonar can see.

    Parametric rather than a real lines plan, but the right shape in the ways
    that matter acoustically, which a symmetric spindle is not:

    * **A transom.** A boat is not double-ended.  It carries most of its beam
      and draft to a flat stern, which is a large near-vertical plate and one of
      the strongest features on the whole body from astern.  ``transom`` is the
      fraction of full beam and draft carried aft; the transom face itself is
      closed with facets, because below the waterline it is wetted surface.
    * **Deadrise that varies.** Sections go from nearly flat-bottomed and boxy
      at the stern to a sharp V at the bow, which is what a hull does and what
      decides whether the bottom throws a specular return straight down.
      Controlled by ``deadrise_stern`` / ``deadrise_bow``, the superellipse
      exponents at each end: large is boxy, 2 is a half-ellipse, 1 is a V.
    * **A fine entry.** The waterline tapers to a stem at the bow, so there is
      no specular point facing forward -- which is exactly why a hull is weak
      bow-on to a forward-looking sonar.

    Args:
        length, beam, draft: overall dimensions (m).
        n_long, n_around: facets along the hull and around each section.  Facets
            must resolve curvature, about ``sqrt(lambda R)/3``; at 100 kHz on a
            1 m radius that is 4 cm.
        transom: fraction of full beam and draft carried to the stern.
        deadrise_stern, deadrise_bow: section superellipse exponents.

    Returns outward-wound ``(vertices, faces)`` centred on the hull, ``+x``
    forward to the bow, ``y`` to port, ``z`` **down** into the water to match
    the depth-positive-down frame.  Outward matters: facets are culled by their
    own normal, so an inward-wound hull is invisible from outside and returns
    the far side instead.  Degenerate triangles at the stem are dropped.
    """
    dt = torch.get_default_dtype()
    s = torch.linspace(0.0, 1.0, n_long, dtype=dt)        # 0 = stern, 1 = bow
    x = (s - 0.5) * length

    # Waterline and keel: taper to nothing at the bow, but stay full aft.
    bow_taper = (1.0 - s ** 2.6).clamp_min(0.0) ** 0.5
    keel_taper = (1.0 - s ** 3.2).clamp_min(0.0) ** 0.62
    # A smooth blend from the transom to full section, not a clamped power.
    # The clamp put a slope discontinuity at s = 0.3, and the half-beam peaks
    # EXACTLY there -- so the hull's widest point, where the beam-aspect
    # specular sits, was a knuckle running round the hull.  Physical optics on
    # a crease does not converge: the return jumped 25 dB between mesh
    # resolutions while the same kernel was exact on every smooth shape.  A
    # real waterline is smooth at its widest point.  The smoothstep has zero
    # slope at both ends, so the blend meets the full section tangentially.
    t = (s / 0.30).clamp(0.0, 1.0)
    fill = transom + (1.0 - transom) * (t * t * (3.0 - 2.0 * t))
    half_beam = 0.5 * beam * bow_taper * fill
    keel = draft * keel_taper * fill

    # Superellipse sections, boxy aft and V-shaped forward.
    n_exp = deadrise_stern + (deadrise_bow - deadrise_stern) * s
    # Stations by ARC LENGTH around each section, not by parameter angle.  A
    # superellipse sampled uniformly in theta puts z = draft * sin(theta)^(2/n)
    # at the waterline, and for n = 2.9 that exponent is 0.69: the first facet
    # below the waterline was 0.62 m tall at the resolution examples/21 used,
    # against the 0.075 m physical optics needs there at 120 kHz, and
    # quadrupling n_around shrank it by only 2.6x.  That is exactly where the
    # specular point sits for a sonar looking up at a hull, so the return there
    # never converged -- it jumped 25 dB between resolutions -- while the same
    # kernel was exact on a plate, a sphere and a cylinder.  Uniform arc length
    # makes every facet in a section the same size, so refinement reaches it.
    fine = torch.linspace(0.0, math.pi, 4096, dtype=dt)
    cf, sf = torch.cos(fine), torch.sin(fine)
    stations = torch.linspace(0.0, 1.0, n_around, dtype=dt)
    rows = []
    for i in range(n_long):
        e = 2.0 / float(n_exp[i])
        yf = half_beam[i] * cf.sign() * cf.abs().clamp_min(1e-12) ** e
        zf = keel[i] * sf.abs().clamp_min(1e-12) ** e
        seg = torch.hypot(yf[1:] - yf[:-1], zf[1:] - zf[:-1])
        arc = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
        if float(arc[-1]) > 0.0:
            arc = arc / arc[-1]
            y = _interp1d(stations, arc, yf)
            z = _interp1d(stations, arc, zf)
        else:  # a collapsed section at the stem: every station at the point
            y = yf[:1].expand(n_around)
            z = zf[:1].expand(n_around)
        rows.append(torch.stack([x[i].expand(n_around), y, z], dim=-1))
    verts = torch.cat(rows, dim=0)

    faces = []
    for i in range(n_long - 1):
        for j in range(n_around - 1):
            a = i * n_around + j
            b, c, d = a + 1, a + n_around, a + n_around + 1
            faces += [[a, b, c], [b, d, c]]

    # Close the transom: a fan over the stern section, facing aft (-x).
    stern_centre = verts[:n_around].mean(dim=0, keepdim=True)
    hub = verts.shape[0]
    verts = torch.cat([verts, stern_centre], dim=0)
    for j in range(n_around - 1):
        faces += [[hub, j + 1, j]]

    tri = torch.tensor(faces, dtype=torch.long)
    # The stem collapses to a point, so triangles there are degenerate: zero
    # area, and a normal that is whatever dividing by a clamped zero gives.
    # They contribute nothing, so drop them rather than carry a facet whose
    # orientation is undefined.
    e1 = verts[tri[:, 1]] - verts[tri[:, 0]]
    e2 = verts[tri[:, 2]] - verts[tri[:, 0]]
    area = 0.5 * torch.linalg.cross(e1, e2, dim=-1).norm(dim=-1)
    # Drop the facets the bow and keel tapers collapse to nothing.  The cutoff
    # has to scale with the dtype: a vertex carries an absolute error of about
    # eps times its own size, so a facet whose relative area falls to eps has a
    # normal made entirely of rounding, pointing anywhere.  In float64 those
    # collapsed facets come out at 1e-17 of the largest and a fixed 1e-9 buries
    # them; in float32 they come out at 1e-8 and a fixed 1e-9 KEEPS them, and
    # three of the fourteen then face inward, where back-face culling lets them
    # scatter as if they were the far side of the hull.  sqrt(eps) sits decades
    # above the rounding floor in both, and decades below the smallest facet
    # this generator means to make (2.8e-2 of the largest).
    floor = float(torch.finfo(dt).eps) ** 0.5
    return verts, tri[area > floor * float(area.max())]


def seawall_mesh(length: float = 100.0, height: float = 10.0, *,
                 slope_deg: float = 90.0, n_along: int = 40, n_up: int = 8
                 ) -> tuple[Tensor, Tensor]:
    """One face of a seawall or breakwater, wound to face ``+y``.

    Body frame: the wall runs along ``x``, its toe along ``z = 0`` and its
    crest ``height`` above it (``z`` is depth-down, so the crest is at
    ``-height``).  ``slope_deg`` is the face's angle from horizontal: 90 is a
    vertical caisson wall, 34 or so a rubble-mound armour slope, which leans
    away from the water as it rises.  The water is on the ``+y`` side.

    Place it with :func:`mesh_target` -- ``yaw`` along the wall's line, the
    toe on the seabed -- and give it the diffuse channel a rock or concrete
    face has (``diffuse_db`` of -5 to -10): seen along its length from a
    sonar in the harbour, a wall is at grazing incidence, where a flat face
    has no specular return at all and everything it shows is roughness.
    Hand the same mesh to :func:`hydropt.reverb.reverberation_arrivals` as
    an occluder and it shadows the water beyond it.
    """
    if length <= 0.0 or height <= 0.0:
        raise ValueError("a wall needs a positive length and height")
    if not 0.0 < slope_deg <= 90.0:
        raise ValueError(f"slope_deg must be in (0, 90], got {slope_deg}")
    run = height / math.tan(math.radians(slope_deg))      # horizontal set-back
    xs = torch.linspace(-length / 2.0, length / 2.0, n_along + 1)
    t = torch.linspace(0.0, 1.0, n_up + 1)
    X, T = torch.meshgrid(xs, t, indexing="ij")           # [n_along+1, n_up+1]
    Y = -run * T                                          # leans away from +y
    Z = -height * T                                       # rises (depth down)
    verts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    faces = []
    for i in range(n_along):
        for j in range(n_up):
            a = i * (n_up + 1) + j
            b = a + (n_up + 1)
            # wound so the normal points to +y (the water side): right-hand
            # rule on (a, b, a+1) gives x-along cross z-up ... = +y
            faces.append([a, b, a + 1])
            faces.append([a + 1, b, b + 1])
    faces_t = torch.tensor(faces, dtype=torch.long)
    _, normal, _ = facet_geometry(verts, faces_t)
    if float(normal[:, 1].mean()) < 0.0:                  # face the water
        faces_t = faces_t[:, [0, 2, 1]]
    return verts, faces_t
