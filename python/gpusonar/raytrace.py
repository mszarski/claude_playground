"""Rays and eigenrays through a layered sound-speed profile: our own compact, range-independent calculation.

The profile is piecewise linear in depth, so in every layer a ray is an arc of a circle and everything is closed form.
z is depth (down positive), theta the grazing angle, s = sin(theta) signed (positive: going down), p = cos(theta) / c the
Snell constant of the ray, g = dc/dz of the layer. From Snell's law, d(cos theta) = p g dz and dz = tan(theta) dr:

    ds/dr = -p g            s is linear in range inside a layer, through a turning point (s = 0) as well
    c = sqrt(1 - s^2) / p   which gives the depth, z = z_1 + (c - c_1) / g
    dt = dr / (c cos theta) = p dr / (1 - s^2) = -ds / (g (1 - s^2))
    t_b - t_a = (atanh(s_a) - atanh(s_b)) / g

(g = 0: a straight segment.) A ray is stepped from layer boundary to layer boundary with these, mirrored at the surface
and at the bottom, each reflection counted. Nothing is integrated numerically.

Eigenrays between a source and a receiver are found by shooting a fan of launch angles, bracketing the receiver's depth
at the receiver's range between neighbouring rays with the same reflection history, and closing the bracket by
bisection on the launch angle. Their amplitude is geometric: the power leaving the source in the solid angle
cos(theta_0) d theta_0 d phi crosses, at range r, the area r d phi |dz / d theta_0| d theta_0 cos(theta_r), so

    I(r) / I(1 m) = cos(theta_0) / (r |dz / d theta_0| cos(theta_r)),     p^2 ~ rho c I

(isovelocity: z = r tan(theta_0) gives 1 / R^2). At a caustic dz / d theta_0 -> 0 and geometric acoustics fails: the
amplitude is clipped at `caustic_clip` times the spherical value and the eigenray is marked; the -pi/2 phase of a
caustic is not applied. Surface reflections multiply by `surface_coefficient` (-1), bottom ones by the Rayleigh
coefficient of the given fluid half-space at the ray's grazing angle there.

This is the core's own code (no Bellhop source was read or copied; Bellhop is run as a separate program in the tests,
as an oracle for the delays). `ray_depth` here answers the same question as `elevation.ray_depth` (which steps by
Snell's law in small steps) exactly; a test holds the two together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Profile:
    """c(z), linear between knots, from the surface (z = 0) to `water_depth` (None: no bottom, the last gradient continues)."""

    depth: np.ndarray
    speed: np.ndarray
    water_depth: float | None = None

    def __post_init__(self):
        d, c = np.asarray(self.depth, dtype=float), np.asarray(self.speed, dtype=float)
        if d.ndim != 1 or d.size != c.size or d.size < 1 or np.any(np.diff(d) <= 0.0):
            raise ValueError("profile: depths must increase and be as many as the speeds")
        if d.size == 1:
            d, c = np.array([d[0], d[0] + 1.0]), np.array([c[0], c[0]])
        if d[0] > 0.0:                                   # continued to the surface with the first speed
            d, c = np.concatenate([[0.0], d]), np.concatenate([[c[0]], c])
        bottom = self.water_depth if self.water_depth else max(d[-1], 0.0) + 2.0e4
        if d[-1] < bottom:                               # continued to the bottom with the last gradient
            g = (c[-1] - c[-2]) / (d[-1] - d[-2])
            d, c = np.concatenate([d, [bottom]]), np.concatenate([c, [c[-1] + g * (bottom - d[-1])]])
        elif d[-1] > bottom:
            keep = d < bottom
            d, c = np.concatenate([d[keep], [bottom]]), np.concatenate([c[keep], [np.interp(bottom, d, c)]])
        self.depth, self.speed = d, c
        self.gradient = np.diff(c) / np.diff(d)
        self.bottom = float(bottom)

    def c(self, z):
        return np.interp(z, self.depth, self.speed)

    def layer(self, z: float, down: bool) -> int:
        """Index of the layer a ray at depth z moving down (or up) is in."""
        i = int(np.searchsorted(self.depth, z, side="right" if down else "left")) - 1
        return min(max(i, 0), len(self.gradient) - 1)


@dataclass
class Ray:
    """A ray at the end of `trace`: where it is and how it got there."""

    z: float
    time: float
    s: float                 # sin of the grazing angle there, positive going down
    n_surface: int
    n_bottom: int
    turns: int
    ok: bool = True          # False: it never got there (a horizontal ray in an isovelocity layer)


def trace(profile: Profile, z0: float, launch: float, r_end: float, max_steps: int = 100000, path=None) -> Ray:
    """The ray leaving depth z0 at grazing angle `launch` (rad, positive DOWN), followed to horizontal range `r_end`.
    `path`: a list that receives (r, z, t) at every layer boundary, turning point and reflection."""
    c0 = float(profile.c(z0))
    p = np.cos(launch) / c0
    s = float(np.sin(launch))
    z, r, t = float(z0), 0.0, 0.0
    ns = nb = turns = 0
    d, c, g_all = profile.depth, profile.speed, profile.gradient
    if s == 0.0:                                         # horizontal: it bends towards slower water
        gi = g_all[profile.layer(z, True)]
        if gi == 0.0:
            return Ray(z, float("nan"), 0.0, 0, 0, 0, False)
        s = -1e-12 * np.sign(gi)
    for _ in range(max_steps):
        down = s > 0.0
        i = profile.layer(z, down)
        g, z_lo, z_hi, c_lo, c_hi = g_all[i], d[i], d[i + 1], c[i], c[i + 1]
        if g == 0.0:
            zb = z_hi if down else z_lo
            dr = abs(zb - z) * (p * c_lo) / abs(s)
            if r + dr >= r_end:
                frac = r_end - r
                z_end = z + np.sign(s) * frac * abs(s) / (p * c_lo)
                return Ray(float(z_end), t + frac / (p * c_lo * c_lo), s, ns, nb, turns)
            r, t, z, s_b = r + dr, t + dr / (p * c_lo * c_lo), zb, s
        else:
            # where it leaves the layer: the boundary ahead if it gets there before turning, else the one behind
            if down:
                reach = p * c_hi < 1.0
                s_b = np.sqrt(max(1.0 - (p * c_hi) ** 2, 0.0)) if reach else -np.sqrt(max(1.0 - (p * c_lo) ** 2, 0.0))
                zb = z_hi if reach else z_lo
            else:
                reach = p * c_lo < 1.0
                s_b = -np.sqrt(max(1.0 - (p * c_lo) ** 2, 0.0)) if reach else np.sqrt(max(1.0 - (p * c_hi) ** 2, 0.0))
                zb = z_lo if reach else z_hi
            dr = (s - s_b) / (p * g)
            if r + dr >= r_end:
                s_e = s - p * g * (r_end - r)
                z_end = z_lo + (np.sqrt(max(1.0 - s_e * s_e, 0.0)) / p - c_lo) / g
                if path is not None and s * s_e < 0.0:   # it turned on the way
                    path.append((r + s / (p * g), z_lo + (1.0 / p - c_lo) / g, t + np.arctanh(s) / g))
                return Ray(float(z_end), t + (np.arctanh(s) - np.arctanh(s_e)) / g, float(s_e), ns, nb, turns + int(s * s_e < 0.0))
            if not reach:
                turns += 1
                if path is not None:                     # the turning point itself
                    path.append((r + s / (p * g), z_lo + (1.0 / p - c_lo) / g, t + np.arctanh(s) / g))
            r, t, z = r + dr, t + (np.arctanh(s) - np.arctanh(s_b)) / g, zb
        s = float(s_b)
        if z <= 0.0 and s < 0.0:
            s, ns = -s, ns + 1
        elif z >= profile.bottom and s > 0.0:
            s, nb = -s, nb + 1
        if path is not None:
            path.append((r, z, t))
        if s == 0.0:                                     # grazing a boundary exactly: nudge back into the water
            s = 1e-12 if z <= 0.0 else -1e-12
    return Ray(z, float("nan"), s, ns, nb, turns, False)


def ray_depth(receiver_depth: float, elevation: float, slant_range: float, ssp_depth, ssp_speed, water_depth: float = 0.0) -> float:
    """Depth at which an arrival seen at `elevation` (rad, UP positive) at `receiver_depth` started, back along the ray for
    the one-way travel time slant_range / c(receiver). A thin call to the chain's own (libs/live `ray_depth_arcs`, what
    `elevation.ray_method: arcs` uses): one implementation. `ray_depth_reference` below is the same calculation with this
    module's Python `trace`, which a test holds it to."""
    from . import _gpusonar as core
    return float(core.ray_depth_arcs(float(receiver_depth), float(elevation), float(slant_range), [float(v) for v in ssp_depth],
                                     [float(v) for v in ssp_speed], float(water_depth or 0.0)))


def ray_depth_reference(receiver_depth: float, elevation: float, slant_range: float, ssp_depth, ssp_speed, water_depth: float = 0.0) -> float:
    """`ray_depth` in Python: closed form layer by layer, bisection on range for the travel time."""
    prof = Profile(ssp_depth, ssp_speed, water_depth or None)
    t_want = slant_range / float(prof.c(receiver_depth))
    lo, hi = 0.0, slant_range * 1.2 + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ray = trace(prof, receiver_depth, -elevation, mid)
        if not ray.ok or ray.time > t_want:
            hi = mid
        else:
            lo = mid
    return trace(prof, receiver_depth, -elevation, 0.5 * (lo + hi)).z


@dataclass
class Eigenray:
    launch: float            # rad at the source, positive down
    arrive: float            # rad at the receiver, positive down (the ray's own direction of travel)
    time: float              # s
    amplitude: complex       # pressure re 1 m from the source (spreading and reflection coefficients; no absorption)
    n_surface: int
    n_bottom: int
    turns: int
    caustic: bool = False    # the spreading was clipped


def _rayleigh(grazing: float, c_water: float, bottom: dict) -> complex:
    from .sim.propagation import rayleigh_reflection
    return complex(rayleigh_reflection(grazing, c_water, float(bottom["speed"]), float(bottom["density"]), float(bottom.get("attenuation", 0.0))))


def eigenrays(profile: Profile, z_source: float, z_receiver: float, range_m: float, max_angle_deg: float = 60.0, n_fan: int = 2401,
              max_bounces: int = 4, surface_coefficient: float = -1.0, bottom: dict | None = None, caustic_clip: float = 10.0) -> list[Eigenray]:
    """Every ray joining a source at depth `z_source` and a receiver at `z_receiver` a horizontal `range_m` apart with at
    most `max_bounces` boundary reflections, earliest first. `bottom`: {speed, density, attenuation} of a fluid half-space
    (None: bottom-reflected rays are left out, the non-ducting case)."""
    angles = np.radians(np.linspace(-max_angle_deg, max_angle_deg, int(n_fan) | 1))
    angles = angles[np.abs(angles) > 1e-9]
    rays = [trace(profile, z_source, a, range_m) for a in angles]
    c_s, c_r = float(profile.c(z_source)), float(profile.c(z_receiver))
    out = []

    def key(q: Ray):
        return (q.n_surface, q.n_bottom, q.turns, q.s > 0.0)

    def close(a_lo, ray_lo, a_hi, ray_hi, level):
        """Eigenrays with launch angles between two rays of the fan. Where the two have different histories (one has
        reflected or turned and the other not) the interval is halved until its parts have one history each."""
        if not (ray_lo.ok and ray_hi.ok):
            return
        if key(ray_lo) != key(ray_hi):
            if level < 30:
                mid = 0.5 * (a_lo + a_hi)
                ray_mid = trace(profile, z_source, mid, range_m)
                close(a_lo, ray_lo, mid, ray_mid, level + 1)
                close(mid, ray_mid, a_hi, ray_hi, level + 1)
            return
        if (ray_lo.z - z_receiver) * (ray_hi.z - z_receiver) > 0.0:
            return
        if ray_lo.n_surface + ray_lo.n_bottom > max_bounces or (ray_lo.n_bottom and bottom is None):
            return
        lo, hi, f_lo = a_lo, a_hi, ray_lo.z - z_receiver
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            q = trace(profile, z_source, mid, range_m)
            if not q.ok or key(q) != key(ray_lo):
                break
            if (q.z - z_receiver) * f_lo > 0.0:
                lo = mid
            else:
                hi = mid
        launch = 0.5 * (lo + hi)
        q = trace(profile, z_source, launch, range_m)
        if not q.ok or key(q) != key(ray_lo) or abs(q.z - z_receiver) > 0.05:
            return
        # geometric spreading from the two rays that bracket it
        dz_dth = abs(ray_hi.z - ray_lo.z) / (a_hi - a_lo)
        cos_r = max(np.sqrt(max(1.0 - q.s * q.s, 0.0)), 1e-6)
        slant = float(np.hypot(range_m, z_receiver - z_source))
        intensity = np.cos(launch) / (range_m * max(dz_dth, 1e-12) * cos_r)
        caustic = intensity > (caustic_clip / slant) ** 2
        amp = np.sqrt(min(intensity, (caustic_clip / slant) ** 2) * c_r / c_s)
        coeff = complex(surface_coefficient) ** q.n_surface
        if q.n_bottom:
            c_b = float(profile.speed[-1])
            coeff *= _rayleigh(float(np.arccos(min(np.cos(launch) / c_s * c_b, 1.0))), c_b, bottom) ** q.n_bottom
        out.append(Eigenray(float(launch), float(np.arcsin(np.clip(q.s, -1.0, 1.0))), q.time, amp * coeff, q.n_surface, q.n_bottom, q.turns, bool(caustic)))

    for i in range(len(angles) - 1):
        close(angles[i], rays[i], angles[i + 1], rays[i + 1], 0)
    out.sort(key=lambda e: e.time)
    # the same ray found in two neighbouring brackets
    kept = []
    for e in out:
        if not any(abs(e.time - k.time) < 1e-9 and e.n_surface == k.n_surface and e.n_bottom == k.n_bottom for k in kept):
            kept.append(e)
    return kept


def transmission_loss(rays: list[Eigenray], frequency: float | None = None) -> float:
    """dB re 1 m: incoherent (sum of powers) or, with `frequency`, coherent (sum of pressures with their delays' phases)."""
    if not rays:
        return float("inf")
    if frequency is None:
        return float(-10.0 * np.log10(sum(abs(e.amplitude) ** 2 for e in rays)))
    return float(-20.0 * np.log10(max(abs(sum(e.amplitude * np.exp(-2j * np.pi * frequency * e.time) for e in rays)), 1e-300)))


def surface_delay(profile: Profile, z_own: float, z_target: float, range_m: float, **kw):
    """One-way delay (s) of the once-surface-reflected arrival behind the first arrival that touches no boundary, through
    the profile, or None when one of the two does not exist (a shadow: no direct path, or no surface path)."""
    rays = eigenrays(profile, z_own, z_target, range_m, max_bounces=1, **kw)
    direct = [e for e in rays if e.n_surface == 0 and e.n_bottom == 0]
    surf = [e for e in rays if e.n_surface == 1 and e.n_bottom == 0]
    if not direct or not surf:
        return None
    return surf[0].time - direct[0].time


def flat_depth_from_delay(delay: float, z_own: float, range_m: float, sound_speed: float) -> float:
    """The target depth the flat, straight-ray formula gives for a one-way surface delay: the z_t that solves
    sqrt(r^2 + (z_t + z)^2) - sqrt(r^2 + (z_t - z)^2) = c delay (r: horizontal)."""
    lo, hi = 0.0, 1.0e4
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.hypot(range_m, mid + z_own) - np.hypot(range_m, mid - z_own) > sound_speed * delay:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
