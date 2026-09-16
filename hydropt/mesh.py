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

    denom = a - c
    safe = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    nested = (_dd1(a, b) - _dd1(b, c)) / safe

    # All three nodes together: the facet lies in a phase front.
    centre = z.mean(dim=-1)
    d = z - centre.unsqueeze(-1)
    series = torch.exp(centre) * (0.5
                                  + d.sum(-1) / 6.0
                                  + (d * d).sum(-1) / 24.0
                                  + d.sum(-1) ** 2 / 36.0)
    collapsed = (gaps.max(dim=-1).values < 1e-5).unsqueeze(-1).squeeze(-1)
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
            body far larger than ``grid * cell`` will alias.

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
    nearest = torch.full((p * grid * grid,), big, dtype=depth.dtype,
                         device=c.device)
    nearest = nearest.scatter_reduce(0, flat, depth.reshape(-1), reduce="amin",
                                     include_self=True)
    front = nearest[flat].reshape(depth.shape)
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


class MeshScattering(ScatteringPattern):
    r"""Bistatic cross-section of a triangle mesh, by coherent physical optics.

    Args:
        vertices: ``[V, 3]`` in the **body frame**, metres.
        faces: ``[F, 3]`` triangle indices, wound so normals point outward.
        sound_speed: to turn frequency into wavelength (m/s).
        learnable: register ``vertices`` as a parameter, so a loss on the image
            can deform the shape.  The faces are fixed -- topology is not
            something a gradient can move.
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

        amp = total / lam.reshape(1, -1).to(total.dtype)
        return (amp.real ** 2 + amp.imag ** 2).reshape(*shape, -1)

    def extra_repr(self) -> str:
        return (f"{int(self.vertices.shape[0])} vertices, {self.n_facets} facets, "
                f"chunk={self.facet_chunk}, "
                f"occlusion={'on' if self.occlusion else 'off'}")


# --------------------------------------------------------------------------- #
# Building a target from a mesh
# --------------------------------------------------------------------------- #
def mesh_target(vertices: Tensor, faces: Tensor, *,
                position: tuple[float, float, float] | Tensor = (0.0, 0.0, 0.0),
                yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                n_patches: int = 1, sound_speed: float = 1500.0,
                learnable: bool = True, learnable_shape: bool = False,
                facet_chunk: int = 512, occlusion: bool = True) -> ExtendedTarget:
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
        learnable: position and orientation are parameters.
        learnable_shape: the vertices are parameters too, so a loss on the
            image reaches the geometry.
        sound_speed, facet_chunk, occlusion: passed to :class:`MeshScattering`.

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
        extent = centroid.max(0).values - centroid.min(0).values
        axis = int(extent.argmax())
        order = centroid[:, axis].argsort()
        groups = [f[chunk] for chunk in torch.chunk(order, n_patches)
                  if chunk.numel() > 0]

    offsets, patterns = [], []
    for group in groups:
        pattern = MeshScattering(v, group, sound_speed=sound_speed,
                                 learnable=learnable_shape,
                                 facet_chunk=facet_chunk, occlusion=occlusion)
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
    fill = transom + (1.0 - transom) * (s / 0.30).clamp(0.0, 1.0) ** 1.5
    half_beam = 0.5 * beam * bow_taper * fill
    keel = draft * keel_taper * fill

    # Superellipse sections, boxy aft and V-shaped forward.
    n_exp = deadrise_stern + (deadrise_bow - deadrise_stern) * s
    theta = torch.linspace(0.0, math.pi, n_around, dtype=dt)
    ct, st = torch.cos(theta), torch.sin(theta)
    rows = []
    for i in range(n_long):
        e = 2.0 / float(n_exp[i])
        y = half_beam[i] * ct.sign() * ct.abs().clamp_min(1e-12) ** e
        z = keel[i] * st.abs().clamp_min(1e-12) ** e
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
    return verts, tri[area > 1e-9 * float(area.max())]
