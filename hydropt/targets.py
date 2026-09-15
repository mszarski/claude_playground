r"""Extended targets: several highlights, an orientation, and aspect dependence.

A point scatterer gives the right arrival time and the wrong image.  At 100 kHz
a 15 mm wavelength resolves a metre-scale object into *several* returns spread
over its length, and its echo strength swings by tens of dB between broadside
and end-on.  A single isotropic point reproduces neither.

Two separate things are needed, and they are kept separate here:

**Geometry** -- :class:`ExtendedTarget` is a rigid cloud of highlights with a
body-frame layout, a position and an orientation.  Each highlight is an
ordinary scatterer, so a multi-highlight target needs no new propagation code:
the legs compose per highlight and sum.  Aspect dependence in the *image* comes
out of this for free, because rotating the body changes the highlights' relative
ranges and therefore how the returns spread in time and bearing.

**Directivity** -- a :class:`ScatteringPattern` gives the bistatic
cross-section of one highlight as a function of the incident and scattered
directions in the body frame.  This is what produces the broadside spike.

Why the patterns return an absolute cross-section
-------------------------------------------------
``sigma`` in m^2, not a normalised lobe times a separate ``TS``.  Plate and
cylinder scattering both set their own absolute level *and* their angular width
from the same geometry and wavelength -- a 2 m plate at 100 kHz is not a 2 m
plate at 10 kHz in either respect -- so splitting level from shape would let the
two disagree.  :class:`IsotropicScattering` is the frequency-flat special case
and takes a ``TS`` directly.

Where aspect dependence is exact, and where it is not
----------------------------------------------------
On the **coherent** path it is exact.  :func:`hydropt.active.compose_arrivals`
already pairs every inbound arrival with every outbound one, and each pair
carries both directions the pattern needs: the inbound arrival's ``direction``
is where the energy was travelling when it reached the target, and the outbound
arrival's ``launch_direction`` is the direction it was scattered into.  No
approximation, and no extra traces.

On the **energy** path it is not available, and the reason is structural rather
than an omission.  :func:`hydropt.active.render_echo` is fast because the two
legs are *separable*, so the echo is one convolution; a ``sigma`` that depends
on both directions at once is exactly the thing that breaks that separability.
An energy render also throws away which incident direction fed which outbound
ray, which is the information a bistatic pattern consumes.  So:

* ``render_echo`` handles **multi-highlight** targets exactly, at one inbound
  render (the highlights are just several receive points) plus one outbound
  trace per highlight.
* An aspect-dependent ``sigma`` on the energy path would need a render per
  incident-direction bin.  That is not implemented; use the coherent path, which
  gets it exactly and is what a beamformer consumes anyway.

References
----------
Urick, *Principles of Underwater Sound*, 3rd ed., ch. 9 (target strength of
simple shapes).  The plate and cylinder forms below are the standard
physical-optics (Kirchhoff) results, and both are checked against their
textbook broadside values in the tests.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "ScatteringPattern",
    "IsotropicScattering",
    "PlateScattering",
    "CylinderScattering",
    "CurvedSurfaceScattering",
    "ExtendedTarget",
    "rotation_matrix",
    "sinc",
]

_EPS = 1e-30


def sinc(x: Tensor) -> Tensor:
    """``sin(x) / x``, with the removable singularity at zero filled in.

    ``torch.sinc`` is ``sin(pi x)/(pi x)``; the acoustics literature writes the
    unnormalised form, and mixing the two misplaces every null by a factor of
    pi.  This is the unnormalised one.
    """
    return torch.sinc(x / math.pi)


def rotation_matrix(yaw: Tensor, pitch: Tensor, roll: Tensor) -> Tensor:
    """Body-to-world rotation, ``[3, 3]``, from yaw-pitch-roll in radians.

    Applied in the order roll, then pitch, then yaw (``R = Rz Ry Rx``), the
    usual vehicle convention.  ``z`` is depth, positive downward, so a positive
    pitch puts the nose *down*.
    """
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    one = torch.ones_like(cy)
    zero = torch.zeros_like(cy)
    rz = torch.stack([torch.stack([cy, -sy, zero]),
                      torch.stack([sy, cy, zero]),
                      torch.stack([zero, zero, one])])
    ry = torch.stack([torch.stack([cp, zero, sp]),
                      torch.stack([zero, one, zero]),
                      torch.stack([-sp, zero, cp])])
    rx = torch.stack([torch.stack([one, zero, zero]),
                      torch.stack([zero, cr, -sr]),
                      torch.stack([zero, sr, cr])])
    return rz @ ry @ rx


# --------------------------------------------------------------------------- #
# Scattering patterns
# --------------------------------------------------------------------------- #
class ScatteringPattern(nn.Module):
    """Bistatic scattering cross-section of one highlight, in its body frame.

    Subclasses implement :meth:`cross_section`.  Directions are *propagation*
    directions and unit length: ``incident`` points the way the sound was
    travelling when it struck, ``scattered`` the way it left.  Backscatter is
    therefore ``scattered = -incident``, not ``scattered = incident``.
    """

    def cross_section(self, incident: Tensor, scattered: Tensor,
                      freqs_khz: Tensor) -> Tensor:
        """``sigma`` in m^2, broadcast to ``[..., B]``.

        Args:
            incident: ``[..., 3]`` unit propagation directions, body frame.
            scattered: ``[..., 3]`` unit propagation directions, body frame.
            freqs_khz: ``[B]`` band centres.
        """
        raise NotImplementedError

    def forward(self, incident: Tensor, scattered: Tensor,
                freqs_khz: Tensor) -> Tensor:
        return self.cross_section(incident, scattered, freqs_khz)

    def peak_target_strength_db(self, freqs_khz: Tensor) -> Tensor:
        """``TS`` at the pattern's strongest aspect, ``[B]`` -- for reporting."""
        axis = torch.zeros(3, dtype=freqs_khz.dtype, device=freqs_khz.device)
        axis[0] = 1.0
        sigma = self.cross_section(axis, -axis, freqs_khz)
        return 10.0 * torch.log10(sigma.clamp_min(_EPS))


class IsotropicScattering(ScatteringPattern):
    """One cross-section, no aspect and no frequency dependence.

    The behaviour of :class:`hydropt.active.PointTarget`, in pattern form, so a
    highlight can be isotropic without being a special case downstream.
    """

    def __init__(self, target_strength_db: float = -10.0, *,
                 learnable: bool = True) -> None:
        super().__init__()
        ts = torch.as_tensor(float(target_strength_db))
        if learnable:
            self.target_strength_db = nn.Parameter(ts)
        else:
            self.register_buffer("target_strength_db", ts)

    def cross_section(self, incident: Tensor, scattered: Tensor,
                      freqs_khz: Tensor) -> Tensor:
        sigma = 10.0 ** (self.target_strength_db.to(freqs_khz.dtype) / 10.0)
        shape = torch.broadcast_shapes(incident.shape[:-1], scattered.shape[:-1])
        return sigma.expand(*shape, int(freqs_khz.shape[0]))

    def extra_repr(self) -> str:
        return f"TS={float(self.target_strength_db):.1f} dB"


class PlateScattering(ScatteringPattern):
    r"""Rigid rectangular plate, physical optics (Kirchhoff).

    The Kirchhoff integral over a flat facet of area ``A = a b`` is

    .. math::
        f(\hat{k}_i, \hat{k}_s) = \frac{A}{\lambda}\,|\hat{n}\cdot\hat{k}_i|\,
        \mathrm{sinc}\!\left(\frac{q_a a}{2}\right)
        \mathrm{sinc}\!\left(\frac{q_b b}{2}\right),
        \qquad \mathbf{q} = k(\hat{k}_s - \hat{k}_i)

    and ``sigma = |f|^2``.  At normal-incidence backscatter both sincs are one
    and the obliquity factor is one, giving the textbook plate cross-section
    ``sigma = (A/lambda)^2`` -- so ``TS = 20 log10(A/lambda)``, which the tests
    check directly.  Off broadside by ``theta`` in the ``a`` direction,
    backscatter has ``q_a a / 2 = k a sin(theta)``, giving the familiar first
    null at ``sin(theta) = lambda / (2a)``.

    Args:
        length: plate extent along ``length_axis`` (m).
        width: extent along the remaining in-plane direction (m).
        normal: body-frame plate normal; defaults to ``+z``.
        length_axis: body-frame direction of the ``length`` side, defaulting to
            body ``x``.  It is orthogonalised against ``normal``, and if the two
            are parallel the next axis is used instead.  Given explicitly
            because a silently-chosen in-plane axis makes ``length`` and
            ``width`` mean whichever way round the implementation happened to
            pick, which is not something a caller can be expected to guess.
        sound_speed: used to turn frequency into wavelength (m/s).
        learnable: register ``length`` and ``width`` as parameters.

    Physical optics is a high-frequency approximation with two consequences
    worth knowing before trusting a number out of it.  It is good near
    broadside and it *understates* the cross-section at grazing aspects, where
    edge diffraction dominates and this model has none.  And the obliquity
    factor takes ``sigma`` to **exactly zero** for an edge-on plate, which is a
    modelling artefact rather than physics: a real plate seen edge-on returns
    its edge.  Exact zero also means no gradient, so do not try to fit a plate's
    orientation from an edge-on aspect.
    """

    def __init__(self, length: float, width: float, *,
                 normal: tuple[float, float, float] | Tensor = (0.0, 0.0, 1.0),
                 length_axis: tuple[float, float, float] | Tensor = (1.0, 0.0, 0.0),
                 sound_speed: float = 1500.0, learnable: bool = True) -> None:
        super().__init__()
        for name, v in {"length": length, "width": width}.items():
            t = torch.as_tensor(float(v))
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)
        dt = torch.get_default_dtype()
        n = torch.as_tensor(normal, dtype=dt).reshape(3)
        n = n / n.norm().clamp_min(_EPS)
        want = torch.as_tensor(length_axis, dtype=dt).reshape(3)
        a = want - (want @ n) * n  # project the requested axis into the plane
        if float(a.norm()) < 1e-6:  # requested axis was parallel to the normal
            alt = torch.zeros_like(n)
            alt[int(n.abs().argmin())] = 1.0
            a = alt - (alt @ n) * n
        a = a / a.norm().clamp_min(_EPS)
        self.register_buffer("normal", n)
        self.register_buffer("length_axis", a)
        self.register_buffer("width_axis", torch.cross(n, a, dim=0))
        self.register_buffer("sound_speed", torch.as_tensor(float(sound_speed)))

    def _axes(self, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """In-plane axes ``(length_axis, width_axis)``, both perpendicular to n."""
        return self.length_axis.to(dtype), self.width_axis.to(dtype)

    def cross_section(self, incident: Tensor, scattered: Tensor,
                      freqs_khz: Tensor) -> Tensor:
        dtype = freqs_khz.dtype
        incident = incident.to(dtype)
        scattered = scattered.to(dtype)
        lam = self.sound_speed.to(dtype) / (freqs_khz * 1.0e3)  # [B]
        k = 2.0 * math.pi / lam.clamp_min(_EPS)  # [B]

        a_hat, b_hat = self._axes(dtype)
        n = self.normal.to(dtype)
        a = self.length.to(dtype).abs()
        b = self.width.to(dtype).abs()

        d = scattered - incident  # [..., 3]
        qa = (d * a_hat).sum(-1).unsqueeze(-1) * k  # [..., B]
        qb = (d * b_hat).sum(-1).unsqueeze(-1) * k
        obliquity = (incident * n).sum(-1).abs().unsqueeze(-1)  # [..., 1]

        amp = (a * b / lam) * obliquity * sinc(qa * a / 2.0) * sinc(qb * b / 2.0)
        return amp * amp

    def extra_repr(self) -> str:
        return (f"length={float(self.length):.3f} m, width={float(self.width):.3f} m, "
                f"normal={self.normal.tolist()}")


class CylinderScattering(ScatteringPattern):
    r"""Finite rigid cylinder, physical optics -- the canonical hull-like target.

    Broadside, Urick's result for a cylinder of radius ``r`` and length ``L`` is

    .. math:: \sigma_{\text{broadside}} = \frac{r L^2}{2\lambda}

    and the aspect dependence is the same aperture factor as a line of length
    ``L``: off broadside by ``theta``, backscatter picks up
    ``sinc^2(k L sin(theta))``, with a ``cos^2(theta)`` obliquity factor from the
    foreshortened projected length.  Both are checked in the tests -- the
    broadside level against the closed form, the null positions against
    ``sin(theta) = lambda / (2L)``.

    The axis is body ``x``, so the strong return is at ``incident``
    perpendicular to the axis.  This is a *specular* model: it has the big
    broadside spike and the nulls, and it does not have the circumferential
    (Lamb) waves that give a real elastic shell its mid-frequency structure.

    End-on it returns **exactly zero**, because the projected length vanishes.
    A real cylinder end-on returns its end cap, of order ``(pi r^2 / lambda)^2``,
    so a target that must stay visible at every aspect wants the cap as its own
    highlight -- an :class:`ExtendedTarget` carrying a hull cylinder plus two
    :class:`PlateScattering` caps.  Exact zero also means no gradient there.

    Args:
        length: cylinder length along body ``x`` (m).
        radius: cylinder radius (m).
        sound_speed: for wavelength (m/s).
        learnable: register ``length`` and ``radius`` as parameters.
    """

    def __init__(self, length: float, radius: float, *,
                 sound_speed: float = 1500.0, learnable: bool = True) -> None:
        super().__init__()
        for name, v in {"length": length, "radius": radius}.items():
            t = torch.as_tensor(float(v))
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)
        self.register_buffer("sound_speed", torch.as_tensor(float(sound_speed)))

    def cross_section(self, incident: Tensor, scattered: Tensor,
                      freqs_khz: Tensor) -> Tensor:
        dtype = freqs_khz.dtype
        incident = incident.to(dtype)
        scattered = scattered.to(dtype)
        lam = self.sound_speed.to(dtype) / (freqs_khz * 1.0e3)
        k = 2.0 * math.pi / lam.clamp_min(_EPS)

        axis = torch.zeros(3, dtype=dtype, device=incident.device)
        axis[0] = 1.0
        length = self.length.to(dtype).abs()
        radius = self.radius.to(dtype).abs()

        # Aperture factor along the axis, from the same q = k(s - i) as the plate.
        d = scattered - incident
        qa = (d * axis).sum(-1).unsqueeze(-1) * k  # [..., B]
        aperture = sinc(qa * length / 2.0)

        # Obliquity: the projected length shortens as cos(theta), theta measured
        # from broadside, i.e. sin of the angle off the axis.
        cos_theta = (1.0 - (incident * axis).sum(-1) ** 2).clamp_min(0.0).sqrt()
        cos_theta = cos_theta.unsqueeze(-1)

        sigma_broadside = radius * length * length / (2.0 * lam)
        return sigma_broadside * (cos_theta * aperture) ** 2

    def extra_repr(self) -> str:
        return (f"length={float(self.length):.3f} m, "
                f"radius={float(self.radius):.3f} m")


class CurvedSurfaceScattering(ScatteringPattern):
    r"""A doubly-curved convex surface -- and the reason a real hull is visible.

    Geometric optics gives a convex surface with principal radii ``R1`` and
    ``R2`` a backscattering cross-section

    .. math:: \sigma = \frac{R_1 R_2}{4}

    which is **independent of aspect and of frequency**.  Setting ``R1 = R2 = a``
    recovers Urick's sphere, ``TS = 10 log10(a^2 / 4)``, and the tests check that
    limit directly.

    Why this exists.  :class:`CylinderScattering` is a *straight* cylinder, and a
    straight cylinder of length ``L`` returns only within about ``lambda / 2L`` of
    its own broadside -- 0.18 degrees for a 2.4 m section at 100 kHz.  Model a
    boat hull as straight sections and it glints from one point and vanishes
    elsewhere, which is not what sonars see.  A real hull is **curved in plan**:
    the waterline is a curve, so the surface is doubly curved and there is a
    specular point on it at *every* aspect.  With a section radius of 0.75 m and a
    plan radius of 30 m that is ``sigma`` = 5.6 m^2, ``TS`` = +7.5 dB, all round --
    an ordinary small-craft target strength, and visible.

    So: use :class:`CylinderScattering` for something genuinely straight and
    unfaired (a pipe, a mast, a torpedo body seen beam-on) and this for anything
    with fairing in two directions, which is most of a hull.

    What it leaves out, all of which make a real hull *more* visible rather than
    less: roughness at the 15 mm scale from fouling, plating seams and fittings,
    which scatters diffusely; the hull-air interface behind a thin shell, which
    reflects almost everything; and for GRP craft, returns from the engine,
    tanks and internal structure the sound reaches through the hull.

    Args:
        radius_1, radius_2: principal radii of curvature (m).  Learnable.
        normal: optional body-frame outward normal.  Given one, the patch is
            one-sided -- lit only from in front, with a smooth obliquity so the
            gradient survives.  Omitted, it scatters from every direction, which
            is what you want for a patch standing in for a whole convex body.
        learnable: register the radii as parameters.
    """

    def __init__(self, radius_1: float, radius_2: float, *,
                 normal: tuple[float, float, float] | Tensor | None = None,
                 learnable: bool = True) -> None:
        super().__init__()
        for name, v in {"radius_1": radius_1, "radius_2": radius_2}.items():
            t = torch.as_tensor(float(v))
            if learnable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)
        if normal is None:
            self.register_buffer("normal", None)
        else:
            n = torch.as_tensor(normal, dtype=torch.get_default_dtype()).reshape(3)
            self.register_buffer("normal", n / n.norm().clamp_min(_EPS))

    def cross_section(self, incident: Tensor, scattered: Tensor,
                      freqs_khz: Tensor) -> Tensor:
        dtype = freqs_khz.dtype
        incident = incident.to(dtype)
        scattered = scattered.to(dtype)
        sigma = (self.radius_1.to(dtype).abs() * self.radius_2.to(dtype).abs()
                 / 4.0)
        shape = torch.broadcast_shapes(incident.shape[:-1], scattered.shape[:-1])
        out = sigma.expand(*shape, 1)
        if self.normal is not None:
            n = self.normal.to(dtype)
            # Lit when the wave travels into the face, seen when it leaves it.
            lit = (-(incident * n).sum(-1)).clamp_min(0.0)
            seen = (scattered * n).sum(-1).clamp_min(0.0)
            out = out * (lit * seen).unsqueeze(-1)
        return out.expand(*shape, int(freqs_khz.shape[0]))

    def extra_repr(self) -> str:
        return (f"R1={float(self.radius_1):.3f} m, R2={float(self.radius_2):.3f} m"
                + ("" if self.normal is None
                   else f", normal={self.normal.tolist()}"))


# --------------------------------------------------------------------------- #
# Extended target
# --------------------------------------------------------------------------- #
class ExtendedTarget(nn.Module):
    """A rigid body of point highlights with an orientation.

    Args:
        highlights: ``[N, 3]`` body-frame offsets (m).  The body origin is
            whatever you choose it to be; ``position`` places that origin.
        patterns: one :class:`ScatteringPattern` shared by every highlight, or a
            sequence of ``N`` of them.  A bare ``float`` or a sequence of
            ``N`` floats is taken as isotropic ``TS`` in dB.
        position: ``(x, y, z)`` of the body origin (m).
        yaw, pitch, roll: orientation in **degrees**.  Yaw is the useful one for
            a target on a level course; all three are learnable.
        learnable: register position and orientation as parameters.

    The highlights are fixed in the body frame by default -- a real target's
    highlights do not move relative to one another -- but pass
    ``learnable_layout=True`` to fit them, which is what you want when
    recovering an unknown target's shape from its echo.
    """

    def __init__(
        self,
        highlights: Tensor,
        patterns: ScatteringPattern | float | list,
        position: tuple[float, float, float] | Tensor = (0.0, 0.0, 0.0),
        *,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        learnable: bool = True,
        learnable_layout: bool = False,
    ) -> None:
        super().__init__()
        h = torch.as_tensor(highlights, dtype=torch.get_default_dtype()).reshape(-1, 3)
        if h.shape[0] == 0:
            raise ValueError("an extended target needs at least one highlight")
        if learnable_layout:
            self.highlights = nn.Parameter(h)
        else:
            self.register_buffer("highlights", h)

        n = h.shape[0]
        if isinstance(patterns, ScatteringPattern):
            pats = [patterns] * n  # shared module: one set of parameters
        elif isinstance(patterns, (int, float)):
            pats = [IsotropicScattering(float(patterns), learnable=learnable)] * n
        else:
            pats = list(patterns)
            if len(pats) != n:
                raise ValueError(f"got {len(pats)} patterns for {n} highlights")
            pats = [IsotropicScattering(float(p), learnable=learnable)
                    if isinstance(p, (int, float)) else p for p in pats]
        # Deduplicated by identity, so a shared pattern is registered once and
        # its parameters are not multiply-counted by an optimiser.
        unique: list[ScatteringPattern] = []
        self._pattern_index: list[int] = []
        for pat in pats:
            for i, seen in enumerate(unique):
                if seen is pat:
                    self._pattern_index.append(i)
                    break
            else:
                unique.append(pat)
                self._pattern_index.append(len(unique) - 1)
        self.patterns = nn.ModuleList(unique)

        pos = torch.as_tensor(position, dtype=h.dtype).reshape(3)
        ang = torch.as_tensor([yaw, pitch, roll], dtype=h.dtype) * (math.pi / 180.0)
        if learnable:
            self.position = nn.Parameter(pos)
            self.orientation = nn.Parameter(ang)
        else:
            self.register_buffer("position", pos)
            self.register_buffer("orientation", ang)

    @property
    def n_highlights(self) -> int:
        return int(self.highlights.shape[0])

    def rotation(self) -> Tensor:
        """Body-to-world rotation matrix, ``[3, 3]``."""
        return rotation_matrix(*self.orientation)

    def world_positions(self) -> Tensor:
        """Highlight positions in world coordinates, ``[N, 3]``."""
        return self.position.reshape(1, 3) + self.highlights @ self.rotation().T

    def pattern_for(self, index: int) -> ScatteringPattern:
        """The pattern governing highlight ``index``."""
        return self.patterns[self._pattern_index[index]]

    def cross_section(self, index: int, incident_world: Tensor,
                      scattered_world: Tensor, freqs_khz: Tensor) -> Tensor:
        """Bistatic ``sigma`` for one highlight, from **world** directions.

        The rotation is applied here rather than in the pattern, so a pattern is
        written once in the body frame and works at any orientation.  Note both
        arguments are rotated by the same matrix, so a pattern that depends only
        on the angle *between* them is orientation-independent, as it should be.
        """
        r = self.rotation().to(incident_world.dtype)
        inc = incident_world @ r  # world -> body is R^T x, i.e. x @ R
        sca = scattered_world @ r
        return self.pattern_for(index).cross_section(inc, sca, freqs_khz)

    def extra_repr(self) -> str:
        yaw, pitch, roll = (float(a) * 180.0 / math.pi for a in self.orientation)
        return (f"{self.n_highlights} highlights, position={self.position.tolist()}, "
                f"yaw={yaw:.1f} deg, pitch={pitch:.1f} deg, roll={roll:.1f} deg")
