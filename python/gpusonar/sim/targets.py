"""Targets as sets of highlights, with aspect dependence, and seabed scatterers that follow
the bottom scattering model.

SST builds every target from highlights: point scatterers fixed to the target's trajectory
(TR 0702 Sec. 9.2; docs/PHYSICS_NOTES.md Sec. 4). `HighlightTarget` is that, plus what TR 0702
leaves to an external program: the amplitude of each highlight worked out once per ping from
the directions to the projector and to the receiver in the target's own frame. Two kinds of
aspect dependence are here:

  * `AspectTable`: a user table of target strength against aspect (and optionally elevation).
  * Geometric-acoustics shapes (`SphereGlint`, `CylinderGlint`, `PlateGlint`, `DiscGlint`):
    smooth rigid bodies in the Kirchhoff (physical optics) approximation. Their highlights
    are specular glints, and a glint is not a fixed point of the body: it sits where the
    surface normal bisects the directions to the source and to the receiver, so it moves over
    the body as the sonar moves. That is what a fixed point scatterer cannot do.

Everything is evaluated before the eigenray pairing, as per-ping positions, amplitudes and
velocities of ordinary scatterers, so both simulator engines (sim/engine.py) see it.

Source of the formulas. E. Lavia, J. D. Gonzalez and S. Blanc, "Modelling high-frequency
backscattering from a mesh of curved surfaces using Kirchhoff approximation" (2018),
arXiv:1805.06945v2, https://arxiv.org/abs/1805.06945 (opened for this work):

  Eq. (1)  f = (i k / 2 pi) Int_lit exp(2 i k khat.x) khat.n dS     backscattering amplitude;
           khat the direction the incident wave travels in, n the outward normal;
           TS = 20 log10 |f|  (their Sec. 5).
  Eq. (5)  sphere, radius r:  f = (i / 4k) exp(-2ikr) (exp(2ikr) - 2ikr - 1);
           for kr >> 1, |f| -> r / 2, i.e. TS = 10 log10(r^2 / 4).
  Eq. (6)  cylinder without end caps, radius a, length 2b, theta from the axis:
           f = -(a/2) tan(theta) sin(2kb cos(theta)) [2/pi - H1(2ka sin(theta)) - i J1(2ka sin(theta))]
           (H1 the Struve function; they cite Gaunaurd, IEEE J. Oceanic Eng. OE-10(3), 1985).
  Eq. (7)  flat end cap (disc of radius a):
           f = -i k a^2 cos(theta) exp(-2ikb cos(theta)) J1(2ka sin(theta)) / (2ka sin(theta)).

What this module adds to those (ours; each is checked in tests/python/test_scene_targets.py
against the equations above or against a direct numerical evaluation of Eq. (1)):

  * Large-argument form of Eq. (6). With theta_n = 90 deg - theta the angle from broadside
    and L = 2b:  |f|^2 = (a L^2 / 2 lambda) cos(theta_n) sinc^2(k L sin(theta_n)),
    sinc(x) = sin(x)/x: the sinc-type pattern with nulls at k L sin(theta_n) = m pi and the
    broadside peak a L^2 / (2 lambda). Note the factor is cos(theta_n), which is what the
    Kirchhoff integral gives. (Urick's table of simple forms is often quoted with cos^2; that
    table could not be opened for this work, so nothing is taken from it.)
  * Rectangle from Eq. (1): |f| = (A / lambda) |khat.n| |sinc(k khat.u) sinc(k khat.v)| for
    edge vectors u, v and area A.
  * Bistatic use. Eq. (1) is monostatic. For a source direction s and a receiver direction r
    (unit vectors from the surface point towards each) we use the symmetric Kirchhoff element
        dA  ->  (1 / lambda) (n.(s + r) / 2) dA,     lit when n.s > 0 and n.r > 0,
    with the phase from the true path lengths. It reduces to Eq. (1) when s = r and is
    reciprocal; away from the specular direction its sidelobes are an approximation, as the
    Kirchhoff sidelobes already are.
  * Phase. Eq. (1) is written for exp(-i w t); the simulator's signals are exp(+i w t) (a
    longer path turns the phase negative), so the conjugate is used: an element of flat
    surface carries +i (a quarter period early) and an element of cylinder glint line
    exp(+i pi / 4). With these, a large plate sums to a mirror image of the source, which a
    test checks.
  * `resolved` shapes. A cylinder's glint line and a plate's face are also offered as rows of
    coherent points (spacing at most half a wavelength). The far-field patterns above then come
    out of the sum by themselves, and so does the near field (range < L^2 / lambda), which is
    where imaging sonars work: there the closed forms are wrong and `resolved: true` is needed.

Validity. All of this is high-frequency: k a >> 1 for curved bodies (creeping waves and
resonances are not here), surfaces rigid unless `reflection` is set (a constant, real or
complex, multiplying the amplitude: -1 for pressure release). `KA_MIN` is the value below
which a warning is given; test_rigid_sphere_validity measures how far the exact rigid sphere
is from a^2/4 above it. Directions are straight lines from the sonar (as for occluders):
with a multipath provider every path pair gets the amplitude of the direct geometry.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .. import _gpusonar as core

KA_MIN = 10.0
"""Warn below this k a. The exact backscatter of a rigid sphere (partial-wave series, worked out
in tests/python/test_scene_targets.py) stays within 0.63 dB of a^2/4 for k a >= 10 and within
0.23 dB for k a >= 20; for k a >= 5 it is within 1.3 dB and for k a >= 2 within 2.4 dB."""

FLAT_PHASE = 1j                         # flat element: a quarter period early (exp(+i w t) signals)
LINE_PHASE = np.exp(0.25j * np.pi)      # cylinder glint line: an eighth of a period early


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def _directions(points, tx, rx):
    """Unit vectors from each point towards the source and towards the receiver."""
    return _unit(tx - points), _unit(rx - points)


# --- aspect table ---------------------------------------------------------------------------

class AspectTable:
    """Target strength (dB re 1 m^2) against aspect, interpolated linearly in dB.

    Aspect is the bearing of the sonar seen from the target, in the target's frame: 0 = from
    ahead (+x), 90 = from the starboard beam (+y), 180 = from astern. A table that stops at
    180 degrees is taken as the same on both sides; one that reaches 360 is periodic.
    With `elevation_deg` (the sonar's elevation above the target's deck plane, positive up)
    `ts_db` is [elevations, aspects]. For a projector and receiver apart, the bisector of the
    two directions is used (the usual monostatic-bistatic equivalence; an approximation).
    """

    def __init__(self, aspect_deg, ts_db, elevation_deg=None):
        self.aspect = np.asarray(aspect_deg, dtype=float)
        self.ts = np.asarray(ts_db, dtype=float)
        self.elevation = None if elevation_deg is None else np.asarray(elevation_deg, dtype=float)
        if np.any(np.diff(self.aspect) <= 0) or self.aspect[0] < 0.0 or self.aspect[-1] > 360.0:
            raise ValueError("aspect_deg must increase within 0..360")
        want = (len(self.aspect),) if self.elevation is None else (len(self.elevation), len(self.aspect))
        if self.ts.shape != want:
            raise ValueError(f"ts_db must have shape {want}")
        self.symmetric = self.aspect[-1] <= 180.0

    @classmethod
    def from_config(cls, cfg) -> "AspectTable":
        if isinstance(cfg, AspectTable):
            return cfg
        return cls(cfg["aspect_deg"], cfg["ts_db"], cfg.get("elevation_deg"))

    def _along_aspect(self, row, aspect):
        if self.symmetric:
            return np.interp(np.abs((aspect + 180.0) % 360.0 - 180.0), self.aspect, row)
        return np.interp(aspect % 360.0, self.aspect, row, period=360.0)

    def ts_db(self, direction):
        """Target strength for unit vectors [n, 3] (target frame) pointing at the sonar."""
        d = np.atleast_2d(np.asarray(direction, dtype=float))
        aspect = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        if self.elevation is None:
            return self._along_aspect(self.ts, aspect)
        elev = np.degrees(np.arcsin(np.clip(-d[:, 2], -1.0, 1.0)))      # frame z is down
        rows = np.stack([self._along_aspect(row, aspect) for row in self.ts])      # [E, n]
        e = np.clip(elev, self.elevation[0], self.elevation[-1])
        hi = np.clip(np.searchsorted(self.elevation, e), 1, len(self.elevation) - 1)
        w = (e - self.elevation[hi - 1]) / (self.elevation[hi] - self.elevation[hi - 1])
        cols = np.arange(len(e))
        return (1.0 - w) * rows[hi - 1, cols] + w * rows[hi, cols]

    def amplitude(self, to_tx, to_rx):
        """Pressure ratio at 1 m for unit vectors towards the source and the receiver."""
        return 10.0 ** (self.ts_db(_unit(np.asarray(to_tx) + np.asarray(to_rx))) / 20.0)


# --- geometric-acoustics shapes (target frame) ----------------------------------------------

class Shape:
    """A smooth body in the target frame. `evaluate(tx, rx, k)` returns the positions [n, 3]
    and complex amplitudes [n] (pressure ratio at 1 m, i.e. sqrt of the differential cross
    section) of its `n` highlights for a source at `tx` and a receiver at `rx`."""

    n = 1
    reflection = 1.0

    def evaluate(self, tx, rx, k):
        raise NotImplementedError

    def nominal(self):
        """Positions to list in the scene before any ping (ground truth, plots)."""
        raise NotImplementedError

    def checks(self, tx, rx, k) -> list[str]:
        """Messages for conditions of validity that do not hold."""
        return []

    def occluder(self):
        """A ("sphere" | "capsule", a, b) primitive in the target frame, or None."""
        return None


def _far_field_message(name, extent, tx, rx, centre, k):
    r = min(np.linalg.norm(tx - centre), np.linalg.norm(rx - centre))
    limit = extent**2 * k / (2.0 * np.pi)
    if r < limit:
        return [f"{name}: range {r:.1f} m is inside L^2/lambda = {limit:.1f} m, where the closed-form pattern "
                f"does not hold; use resolved: true"]
    return []


def _spacing_message(name, spacing, k):
    if spacing > np.pi / k * 1.0001:
        return [f"{name}: spacing {spacing * 1e3:.2f} mm is more than half a wavelength ({np.pi / k * 1e3:.2f} mm); "
                f"the row of points will have grating lobes"]
    return []


@dataclass
class SphereGlint(Shape):
    """Rigid sphere, k a >> 1: one highlight of amplitude a / 2 (TS = 10 log10(a^2 / 4), the
    limit of Eq. (5)) at the specular point, where the normal bisects the directions to the
    source and to the receiver. The same amplitude holds for a source and receiver apart (the
    reflected rays of a sphere spread evenly), short of the forward-scattering lobe.
    Far field of the sphere: at a range r from the centre the true echo is weaker by
    (r - a) / r (the curved mirror spreads the rays), which is not applied."""

    centre: np.ndarray
    radius: float
    reflection: complex = 1.0

    def __post_init__(self):
        self.centre = np.asarray(self.centre, dtype=float)

    def glint_normal(self, tx, rx):
        n = _unit(_unit(tx - self.centre) + _unit(rx - self.centre))
        for _ in range(8):                       # the specular point for sources at a finite range
            p = self.centre + self.radius * n
            n = _unit(_unit(tx - p) + _unit(rx - p))
        return n

    def evaluate(self, tx, rx, k):
        n = self.glint_normal(tx, rx)
        return (self.centre + self.radius * n)[None, :], np.array([self.reflection * 0.5 * self.radius], dtype=complex)

    def nominal(self):
        return self.centre[None, :]

    def checks(self, tx, rx, k):
        return [f"sphere: k a = {k * self.radius:.1f} is below {KA_MIN:g}"] if k * self.radius < KA_MIN else []

    def occluder(self):
        return ("sphere", self.centre, self.radius)


@dataclass
class CylinderGlint(Shape):
    """Side of a rigid circular cylinder between `start` and `end`, k a >> 1.

    The glint is a line along the cylinder at the azimuth where the outward normal lies in
    the plane of the axis and the bisector B = s + r. Per unit length it has the amplitude
    sqrt(a |B_perp| / (4 lambda)) (B_perp: the part of B across the axis), from the
    stationary point of Eq. (1) around the circumference.

    `resolved=False`: one highlight at the middle of the line carrying the whole pattern,
        L sqrt(a |B_perp| / (4 lambda)) sinc(k L B_par / 2),
    which for backscatter is (a L^2 / 2 lambda) cos(theta_n) sinc^2(k L sin(theta_n)) in power.
    Valid beyond L^2 / lambda.
    `resolved=True`: `spacing` apart along the line, one highlight each, each with the
    directions it sees itself. Valid at any range beyond a few radii.
    """

    start: np.ndarray
    end: np.ndarray
    radius: float
    resolved: bool = False
    spacing: float | None = None
    reflection: complex = 1.0

    def __post_init__(self):
        self.start, self.end = np.asarray(self.start, dtype=float), np.asarray(self.end, dtype=float)
        self.length = float(np.linalg.norm(self.end - self.start))
        self.axis = (self.end - self.start) / self.length
        if self.resolved:
            if not self.spacing:
                raise ValueError("a resolved cylinder needs `spacing` (at most half a wavelength)")
            self.n = max(int(np.ceil(self.length / float(self.spacing))), 1)
        else:
            self.n = 1
        self.step = self.length / self.n
        self.stations = self.start + (np.arange(self.n) + 0.5)[:, None] * self.step * self.axis

    def evaluate(self, tx, rx, k):
        points = self.stations
        for _ in range(3):                       # directions from the glint itself, not from the axis
            s, r = _directions(points, tx, rx)
            b = s + r
            b_par = b @ self.axis
            b_perp = b - b_par[:, None] * self.axis
            size = np.linalg.norm(b_perp, axis=1)
            points = self.stations + self.radius * b_perp / np.maximum(size, 1e-12)[:, None]
        # Each highlight stands for `step` of the line (all of it when not resolved); the sinc
        # is the exact sum of the linear phase along that piece.
        amp = (self.reflection * LINE_PHASE * np.sqrt(self.radius * size * k / (8.0 * np.pi)) * self.step
               * np.sinc(k * self.step * b_par / (2.0 * np.pi)))
        return points, amp.astype(complex)

    def nominal(self):
        return self.stations.copy()

    def checks(self, tx, rx, k):
        out = [f"cylinder: k a = {k * self.radius:.1f} is below {KA_MIN:g}"] if k * self.radius < KA_MIN else []
        if self.resolved:
            return out + _spacing_message("cylinder", self.step, k)
        return out + _far_field_message("cylinder", self.length, tx, rx, 0.5 * (self.start + self.end), k)

    def occluder(self):
        return ("capsule", (self.start, self.end), self.radius)


def _flat_amplitude(normal, s, r, k):
    """(1 / lambda) n.(s + r) / 2 with the flat-element phase, zero unless both sides are lit."""
    cs, cr = s @ normal, r @ normal
    return np.where((cs > 0.0) & (cr > 0.0), FLAT_PHASE * (k / (2.0 * np.pi)) * 0.5 * (cs + cr), 0.0)


@dataclass
class PlateGlint(Shape):
    """A rigid flat facet: the parallelogram origin + a u + b v (a, b in [0, 1]), scattering
    from the side its normal u x v points to (both sides with `two_sided`).

    `resolved=False`: one highlight at the centre with (A / lambda) (n.(s + r) / 2)
    sinc(k B.u / 2) sinc(k B.v / 2), B = s + r: the glint when the normal bisects the two
    directions, the sinc sidelobes elsewhere. Valid beyond (longest edge)^2 / lambda.
    `resolved=True`: a grid `spacing` apart, each point (1 / lambda) (n.(s + r) / 2) dA.
    """

    origin: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    resolved: bool = False
    spacing: float | None = None
    two_sided: bool = False
    reflection: complex = 1.0

    def __post_init__(self):
        self.origin, self.edge_u, self.edge_v = (np.asarray(x, dtype=float) for x in (self.origin, self.edge_u, self.edge_v))
        cross = np.cross(self.edge_u, self.edge_v)
        self.area = float(np.linalg.norm(cross))
        self.normal = cross / self.area
        if self.resolved:
            if not self.spacing:
                raise ValueError("a resolved plate needs `spacing` (at most half a wavelength)")
            nu = max(int(np.ceil(np.linalg.norm(self.edge_u) / float(self.spacing))), 1)
            nv = max(int(np.ceil(np.linalg.norm(self.edge_v) / float(self.spacing))), 1)
            a, b = np.meshgrid((np.arange(nu) + 0.5) / nu, (np.arange(nv) + 0.5) / nv, indexing="ij")
            self.points = self.origin + a.reshape(-1, 1) * self.edge_u + b.reshape(-1, 1) * self.edge_v
        else:
            nu = nv = 1
            self.points = (self.origin + 0.5 * (self.edge_u + self.edge_v))[None, :]
        self.cell = self.area / (nu * nv)
        self.cell_u, self.cell_v = self.edge_u / nu, self.edge_v / nv
        self.step = max(np.linalg.norm(self.cell_u), np.linalg.norm(self.cell_v))
        self.n = len(self.points)

    def evaluate(self, tx, rx, k):
        s, r = _directions(self.points, tx, rx)
        amp = _flat_amplitude(self.normal, s, r, k)
        if self.two_sided:
            amp = amp + _flat_amplitude(-self.normal, s, r, k)
        # Each highlight stands for one cell (the whole plate when not resolved); the sincs
        # are the exact sum of the linear phase over it.
        b = s + r
        amp = (self.reflection * amp * self.cell
               * np.sinc(k * (b @ self.cell_u) / (2.0 * np.pi)) * np.sinc(k * (b @ self.cell_v) / (2.0 * np.pi)))
        return self.points, amp.astype(complex)

    def nominal(self):
        return self.points.copy()

    def checks(self, tx, rx, k):
        if self.resolved:
            return _spacing_message("plate", self.step, k)
        extent = max(np.linalg.norm(self.edge_u), np.linalg.norm(self.edge_v))
        return _far_field_message("plate", extent, tx, rx, self.points[0], k)


@dataclass
class DiscGlint(Shape):
    """A rigid flat disc (a cylinder's end cap), scattering from the side `normal` points to.

    `resolved=False`: one highlight at the centre with (pi a^2 / lambda) (n.(s + r) / 2)
    2 J1(x) / x, x = k a |B_tangential|: Eq. (7) (x = 2 k a sin(theta) for backscatter).
    `resolved=True`: a grid `spacing` apart over the disc.
    """

    centre: np.ndarray
    normal: np.ndarray
    radius: float
    resolved: bool = False
    spacing: float | None = None
    reflection: complex = 1.0

    def __post_init__(self):
        self.centre, self.normal = np.asarray(self.centre, dtype=float), _unit(self.normal)
        self.area = np.pi * self.radius**2
        if self.resolved:
            if not self.spacing:
                raise ValueError("a resolved disc needs `spacing` (at most half a wavelength)")
            ref = np.array([0.0, 0.0, 1.0]) if abs(self.normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = _unit(np.cross(self.normal, ref))
            v = np.cross(self.normal, u)
            m = max(int(np.ceil(2.0 * self.radius / float(self.spacing))), 1)
            g = ((np.arange(m) + 0.5) / m - 0.5) * 2.0 * self.radius
            a, b = np.meshgrid(g, g, indexing="ij")
            inside = a**2 + b**2 <= self.radius**2
            self.points = self.centre + a[inside][:, None] * u + b[inside][:, None] * v
            self.cell = self.area / max(len(self.points), 1)          # the grid stands for the whole area
        else:
            self.points = self.centre[None, :]
            self.cell = self.area
        self.n = len(self.points)

    def evaluate(self, tx, rx, k):
        s, r = _directions(self.points, tx, rx)
        amp = self.reflection * _flat_amplitude(self.normal, s, r, k) * self.cell
        if not self.resolved:
            from scipy.special import j1
            b = s + r
            x = k * self.radius * np.linalg.norm(b - (b @ self.normal)[:, None] * self.normal, axis=1)
            amp = amp * np.where(x > 1e-6, 2.0 * j1(x) / np.maximum(x, 1e-6), 1.0)
        return self.points, amp.astype(complex)

    def nominal(self):
        return self.points.copy()

    def checks(self, tx, rx, k):
        if self.resolved:
            return _spacing_message("disc", 2.0 * self.radius / max(int(np.ceil(2.0 * self.radius / float(self.spacing))), 1), k)
        return _far_field_message("disc", 2.0 * self.radius, tx, rx, self.centre, k)


def shapes_from_config(item: dict) -> list[Shape]:
    """Shapes of one `shapes:` entry of a target: {shape: sphere | cylinder | plate | disc, ...}."""
    kind = item["shape"]
    common = {"reflection": complex(item.get("reflection", 1.0))}
    sampled = {"resolved": bool(item.get("resolved", False)), "spacing": item.get("spacing"), **common}
    if kind == "sphere":
        return [SphereGlint(item.get("centre", [0.0, 0.0, 0.0]), float(item["radius"]), **common)]
    if kind == "cylinder":
        out = [CylinderGlint(item["start"], item["end"], float(item["radius"]), **sampled)]
        if item.get("caps", False):
            axis = out[0].axis
            out += [DiscGlint(item["start"], -axis, float(item["radius"]), **sampled),
                    DiscGlint(item["end"], axis, float(item["radius"]), **sampled)]
        return out
    if kind == "plate":
        return [PlateGlint(item["origin"], item["edge_u"], item["edge_v"], two_sided=bool(item.get("two_sided", False)),
                           **sampled)]
    if kind == "disc":
        return [DiscGlint(item["centre"], item["normal"], float(item["radius"]), **sampled)]
    raise ValueError(f"unknown target shape {kind!r}")


# --- highlights and the target ----------------------------------------------------------------

class Response:
    """A highlight's complex frequency response H(f) (exp(+i w t): a delay tau is
    exp(-2 pi i f tau)), from a table {frequency, magnitude_db, phase_deg} or a callable.

    The synthesis kernel takes one complex amplitude and one delay per path, so the response
    is reduced to its first-order form about the centre frequency fc: the magnitude and phase
    at fc, and the group delay -d(phase)/d(omega) there as an extra delay. Curvature of the
    response across the band (a resonance narrower than the pulse bandwidth) is not carried.
    """

    def __init__(self, frequency=None, magnitude_db=None, phase_deg=None, fn=None):
        self.fn = fn
        if fn is None:
            self.f = np.asarray(frequency, dtype=float)
            self.mag = np.zeros(len(self.f)) if magnitude_db is None else np.asarray(magnitude_db, dtype=float)
            self.phase = np.zeros(len(self.f)) if phase_deg is None else np.unwrap(np.radians(np.asarray(phase_deg, dtype=float)))

    @classmethod
    def from_config(cls, cfg):
        if cfg is None or isinstance(cfg, Response):
            return cfg
        if callable(cfg):
            return cls(fn=cfg)
        return cls(cfg["frequency"], cfg.get("magnitude_db"), cfg.get("phase_deg"))

    def _value(self, f):
        if self.fn is not None:
            h = complex(self.fn(f))
            return abs(h), float(np.angle(h))
        return 10.0 ** (float(np.interp(f, self.f, self.mag)) / 20.0), float(np.interp(f, self.f, self.phase))

    def at(self, fc: float):
        """(complex factor, extra delay in s) that stand for H near fc."""
        mag, phase = self._value(fc)
        df = 1e-3 * fc
        lo, hi = self._value(fc - df)[1], self._value(fc + df)[1]
        slope = (np.angle(np.exp(1j * (hi - lo)))) / (2.0 * df)             # d(phase)/df, rad/Hz
        delay = -slope / (2.0 * np.pi)
        # H(f) ~ mag exp(i (phase + slope (f - fc))); the delay supplies exp(-2 pi i f delay)
        # = exp(i slope f), which leaves exp(i (phase - slope fc)) for the amplitude.
        return mag * np.exp(1j * (phase - slope * fc)), delay


@dataclass
class Highlight:
    """One highlight: a point of the target with a strength.

    position            in the target frame (x ahead, y to starboard, z down), m
    target_strength_db  dB re 1 m^2 (or `amplitude`, the pressure ratio at 1 m, complex)
    delay               s between reception and re-emission
    response            `Response` (or its config): complex factor and group delay at the ping's frequency
    aspect_table        `AspectTable` (or its config) added, in dB, to the strength
    """

    position: np.ndarray
    target_strength_db: float | None = None
    amplitude: complex = 1.0
    delay: float = 0.0
    response: Response | None = None
    aspect_table: AspectTable | None = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.target_strength_db is not None:
            self.amplitude = 10.0 ** (float(self.target_strength_db) / 20.0)
        self.response = Response.from_config(self.response)
        if self.aspect_table is not None:
            self.aspect_table = AspectTable.from_config(self.aspect_table)


@dataclass
class PingContext:
    """What a per-ping scene modifier is given: the ping, and its own slice of the scene."""

    time: float
    tx: np.ndarray
    rx: np.ndarray
    frequency: float | None
    sound_speed: float
    points: np.ndarray
    amplitudes: np.ndarray
    velocities: np.ndarray
    normals: np.ndarray

    def wavenumber(self, who: str) -> float:
        if self.frequency is None:
            raise ValueError(f"{who} needs the ping's frequency: call Scene.evaluate(..., frequency=...)")
        return 2.0 * np.pi * float(self.frequency) / float(self.sound_speed)


class HighlightTarget:
    """A rigid set of highlights and shapes on a trajectory.

    origin, velocity      the target frame's origin at t = 0 (NED, m) and its velocity (m/s)
    heading_deg, pitch_deg, roll_deg   attitude at t = 0 (heading 0 = +x, 90 = +y)
    turn_rate_deg         heading rate, deg/s (the body turns about its origin; the origin
                          keeps its straight line)
    trajectory            optional fn(t) -> (origin [3], rpy [3] in rad) replacing all of the above
    highlights            `Highlight`s
    shapes                `Shape`s (specular glints)
    aspect_table          `AspectTable` for the target as a whole: the total (incoherently
                          summed) strength of the highlights at each aspect; the highlights'
                          own strengths then only share it out
    common_center         SST's `commonCenter`: every highlight takes the direction and
                          spreading of the target's origin and differs only in arrival time
                          (highlights only; glints are always placed where they are)
    random_position       RMS (per axis, m) of a Gaussian jitter of the highlight positions,
                          drawn once: breaks up the grating lobes of regularly spaced highlights
    """

    def __init__(self, origin=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), heading_deg: float = 0.0,
                 pitch_deg: float = 0.0, roll_deg: float = 0.0, turn_rate_deg: float = 0.0, trajectory=None,
                 highlights=(), shapes=(), aspect_table=None, common_center: bool = False,
                 random_position: float = 0.0, rng: np.random.Generator | None = None):
        self.origin = np.asarray(origin, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.rpy = np.radians([roll_deg, pitch_deg, heading_deg])
        self.turn_rate = float(np.radians(turn_rate_deg))
        self.trajectory = trajectory
        self.highlights = [h if isinstance(h, Highlight) else Highlight(**h) for h in highlights]
        self.shapes = list(shapes)
        self.aspect_table = None if aspect_table is None else AspectTable.from_config(aspect_table)
        self.common_center = bool(common_center)
        self.local = np.array([h.position for h in self.highlights], dtype=float).reshape(-1, 3)
        if random_position > 0.0 and len(self.local):
            self.local = self.local + float(random_position) * (rng or np.random.default_rng()).standard_normal(self.local.shape)
        self.n = len(self.highlights) + sum(s.n for s in self.shapes)
        self._warned: set[str] = set()

    @classmethod
    def from_config(cls, item: dict, rng: np.random.Generator | None = None) -> "HighlightTarget":
        """From a scene item {type: target, origin, velocity, heading_deg, ..., highlights: [...],
        shapes: [...], aspect_table: {...}, common_center, random_position}."""
        shapes = [s for cfg in item.get("shapes", []) for s in shapes_from_config(cfg)]
        highlights = [Highlight(**{k: v for k, v in h.items()}) for h in item.get("highlights", [])]
        return cls(item.get("origin", (0.0, 0.0, 0.0)), item.get("velocity", (0.0, 0.0, 0.0)),
                   float(item.get("heading_deg", 0.0)), float(item.get("pitch_deg", 0.0)), float(item.get("roll_deg", 0.0)),
                   float(item.get("turn_rate_deg", 0.0)), None, highlights, shapes, item.get("aspect_table"),
                   bool(item.get("common_center", False)), float(item.get("random_position", 0.0)), rng)

    # -- frame --
    def pose(self, t: float):
        """(origin [3], body-to-NED rotation [3, 3], origin velocity [3], angular velocity [3])."""
        if self.trajectory is not None:
            dt = 1e-3
            (p0, a0), (p1, a1) = self.trajectory(t - dt), self.trajectory(t + dt)
            p, a = self.trajectory(t)
            rot = np.asarray(core.rotation_ned(*(float(x) for x in a)))
            yaw_rate = np.angle(np.exp(1j * (a1[2] - a0[2]))) / (2 * dt)
            return (np.asarray(p, dtype=float), rot, (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)) / (2 * dt),
                    np.array([0.0, 0.0, yaw_rate]))
        rot = np.asarray(core.rotation_ned(float(self.rpy[0]), float(self.rpy[1]), float(self.rpy[2] + self.turn_rate * t)))
        return self.origin + self.velocity * t, rot, self.velocity, np.array([0.0, 0.0, self.turn_rate])

    def is_static(self) -> bool:
        return self.trajectory is None and not np.any(self.velocity) and self.turn_rate == 0.0

    def occluders(self) -> list:
        """The shapes' occluders in the world frame (static targets only)."""
        origin, rot, _, _ = self.pose(0.0)
        out = []
        for shape in self.shapes:
            occ = shape.occluder()
            if occ is None:
                continue
            kind, a, b = occ
            if kind == "sphere":
                out.append((kind, list(origin + rot @ a), float(b)))
            else:
                out.append((kind, (list(origin + rot @ a[0]), list(origin + rot @ a[1])), float(b)))
        return out

    def nominal(self):
        """World positions and amplitudes to list in the scene before any ping."""
        origin, rot, _, _ = self.pose(0.0)
        local = [self.local] + [s.nominal() for s in self.shapes]
        amp = [np.array([h.amplitude for h in self.highlights], dtype=complex)] + [np.zeros(s.n, dtype=complex) for s in self.shapes]
        return origin + np.vstack(local) @ rot.T, np.concatenate(amp)

    # -- per ping --
    def _warn(self, messages):
        for m in messages:
            key = m[:m.find(":") + 8]                       # one warning per shape kind and condition
            if key not in self._warned:
                self._warned.add(key)
                warnings.warn("target " + m, stacklevel=3)

    def evaluate(self, ctx: PingContext):
        """Positions [n, 3], amplitudes [n] and velocities [n, 3] of the highlights for a ping."""
        origin, rot, v0, omega = self.pose(ctx.time)
        tx_w, rx_w = np.asarray(ctx.tx, dtype=float), np.asarray(ctx.rx, dtype=float)
        tx, rx = (tx_w - origin) @ rot, (rx_w - origin) @ rot           # the sonar in the target frame
        c = float(ctx.sound_speed)
        local, amps = [], []
        if self.highlights:
            pts = self.local
            s, r = _directions(pts, tx, rx)
            amp = np.array([h.amplitude for h in self.highlights], dtype=complex)
            delay = np.array([h.delay for h in self.highlights], dtype=float)
            if self.aspect_table is not None:
                # The table is the strength of the whole set: scale the highlights so their
                # cross sections add up to it at this aspect.
                total = np.sum(np.abs(amp) ** 2)
                centre_s, centre_r = _directions(np.zeros((1, 3)), tx, rx)
                amp = amp * self.aspect_table.amplitude(centre_s, centre_r)[0] / np.sqrt(max(total, 1e-300))
            for i, h in enumerate(self.highlights):
                if h.aspect_table is not None:
                    amp[i] *= h.aspect_table.amplitude(s[i], r[i])[0]
                if h.response is not None:
                    factor, extra = h.response.at(ctx.wavenumber("a highlight response") * c / (2.0 * np.pi))
                    amp[i] *= factor
                    delay[i] += extra
            if self.common_center:
                pts = _on_centre_line(pts, tx, rx)
            if np.any(delay):
                pts, amp = _delayed(pts, amp, delay * c, tx, rx)
            local.append(pts)
            amps.append(amp)
        if self.shapes:
            k = ctx.wavenumber("a target shape")
            for shape in self.shapes:
                p, a = shape.evaluate(tx, rx, k)
                self._warn(shape.checks(tx, rx, k))
                local.append(p)
                amps.append(a)
        local = np.vstack(local) if local else np.zeros((0, 3))
        arm = local @ rot.T
        return origin + arm, np.concatenate(amps) if amps else np.zeros(0, dtype=complex), v0 + np.cross(omega, arm)


def _delayed(points, amplitudes, extra_path, tx, rx):
    """Move each point away from the sonar along the bisector so that the two-way path grows
    by `extra_path` (m), and scale its amplitude so that the spreading loss stays that of the
    true position (exactly for the direct path)."""
    moved = points.copy()
    for _ in range(3):
        s, r = _directions(moved, tx, rx)
        b = s + r
        now = np.linalg.norm(tx - moved, axis=1) + np.linalg.norm(rx - moved, axis=1)
        want = np.linalg.norm(tx - points, axis=1) + np.linalg.norm(rx - points, axis=1) + extra_path
        moved = moved - _unit(b) * ((want - now) / np.maximum(np.linalg.norm(b, axis=1), 1e-6))[:, None]
    loss = (np.linalg.norm(tx - moved, axis=1) * np.linalg.norm(rx - moved, axis=1)
            / (np.linalg.norm(tx - points, axis=1) * np.linalg.norm(rx - points, axis=1)))
    return moved, amplitudes * loss


def _on_centre_line(points, tx, rx):
    """Put each point on the bisector through the target's origin at the place that has the
    point's own two-way path length (SST's commonCenter: direction and loss of the origin,
    arrival time of the highlight)."""
    want = np.linalg.norm(tx - points, axis=1) + np.linalg.norm(rx - points, axis=1)
    s, r = _directions(np.zeros((1, 3)), tx, rx)
    b = (s + r)[0]
    size = max(np.linalg.norm(b), 1e-6)
    along = np.zeros(len(points))
    for _ in range(4):
        moved = -(b / size)[None, :] * along[:, None]
        now = np.linalg.norm(tx - moved, axis=1) + np.linalg.norm(rx - moved, axis=1)
        along = along + (want - now) / size
    return -(b / size)[None, :] * along[:, None]


# --- seabed scatterers with the bottom model ---------------------------------------------------

SEDIMENT_KEYS = ("rho", "nu", "delta", "w2", "gamma2", "sigma2")


@dataclass
class SedimentScatterers:
    """Per-ping amplitudes of a field of seabed point scatterers so that its cross section
    per unit area is `bottom.strength(grazing_in, grazing_out, ...)` (sim/bottom.py: interface
    roughness plus sediment volume) at each scatterer's own angles, in place of Lambert's law.

    Each scatterer keeps one complex Gaussian draw `g / sqrt(density)` for the whole run
    (ping-to-ping coherence, as for the Lambert seabed). Per ping it is multiplied by
    sqrt(sigma(g_in, g_out, azimuth) / (sin g_in sin g_out)), the angles taken on the straight
    lines to the projector and to the receiver, from the scatterer's normal if it has one.
    The scatterers stay flagged as Lambert patches, so the pairing (either engine) multiplies
    by sqrt(sin g_in sin g_out) of each eigenray pair: for the direct pair the product is
    exactly sqrt(sigma); a pair with a boundary bounce gets sigma of the direct angles times
    Lambert's ratio for its own angles. The frequency is the ping's centre frequency.
    """

    params: dict
    offset_db: float = 0.0

    def cross_section(self, ctx: PingContext):
        """(sigma per unit area [n], sin g_in [n], sin g_out [n]) for the context's scatterers."""
        from . import bottom
        s, r = _directions(ctx.points, np.asarray(ctx.tx, dtype=float), np.asarray(ctx.rx, dtype=float))
        n = np.where(np.isfinite(ctx.normals).all(axis=1)[:, None], ctx.normals, np.array([0.0, 0.0, -1.0]))
        sin_in, sin_out = (s * n).sum(axis=1), (r * n).sum(axis=1)
        lit = (sin_in > 1e-6) & (sin_out > 1e-6)
        g_in = np.arcsin(np.clip(np.where(lit, sin_in, 1.0), 0.0, 1.0))
        g_out = np.arcsin(np.clip(np.where(lit, sin_out, 1.0), 0.0, 1.0))
        # Azimuth between the horizontal directions of travel of the incident and scattered
        # waves in the scatterer's tangent plane: pi when the sound goes back where it came from.
        t_in = -(s - sin_in[:, None] * n)
        t_out = r - sin_out[:, None] * n
        norm = np.linalg.norm(t_in, axis=1) * np.linalg.norm(t_out, axis=1)
        azimuth = np.where(norm > 1e-12, np.arccos(np.clip((t_in * t_out).sum(axis=1) / np.maximum(norm, 1e-12), -1.0, 1.0)), np.pi)
        k = ctx.wavenumber("a fluid_sediment seafloor")
        sigma = bottom.strength(grazing_in=g_in, grazing_out=g_out, frequency=k * float(ctx.sound_speed) / (2.0 * np.pi),
                                sound_speed=float(ctx.sound_speed), azimuth=azimuth, **self.params)
        sigma = np.where(lit, np.asarray(sigma, dtype=float) * 10.0 ** (self.offset_db / 10.0), 0.0)
        return sigma, np.where(lit, sin_in, 1.0), np.where(lit, sin_out, 1.0)

    def evaluate(self, ctx: PingContext):
        sigma, sin_in, sin_out = self.cross_section(ctx)
        return ctx.points, ctx.amplitudes * np.sqrt(sigma / (sin_in * sin_out)), ctx.velocities

    @classmethod
    def from_config(cls, item: dict) -> "SedimentScatterers":
        if "sediment" in item:
            # A name in configs/sediments.yaml (or `sediment_file`); the item's own values win.
            from . import bottom
            item = dict(bottom.sediment(item["sediment"], item.get("sediment_file"), supplied=item), **item)
        missing = [k for k in SEDIMENT_KEYS if k not in item]
        if missing:
            raise ValueError(f"a fluid_sediment seafloor needs {missing}")
        # `roughness_model`, `small_scale_kh` and `shadowing` as for the reverberation item (sim/bottom.py);
        # without them the model's default applies (small_slope since D-86; `roughness_model: perturbation`
        # gives the earlier amplitudes).
        extra = {k: (str(item[k]) if k in ("roughness_model", "shadowing") else float(item[k]))
                 for k in ("roughness_model", "small_scale_kh", "shadowing") if k in item}
        return cls({**{k: float(item[k]) for k in SEDIMENT_KEYS}, **extra}, float(item.get("offset_db", 0.0)))
