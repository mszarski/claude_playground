"""Reverberation power from a seabed with relief (sim/bathymetry.py): the calculation of
reverb_bistatic.py with every cell at its own depth.

Per cell (azimuth sector x range ring about the receiver's ground point, at the horizontal
position (x, y)) and per pair of eigenrays (q from the transmitter, p from the receiver):

    position   R = (x, y, D(x, y))
    time       T_q(R) + T_p(R)                    eigenrays to the cell itself
    area       r dr dphi sqrt(1 + |grad D|^2)     the facet's true area
    grazing    sin g_q = -a_q . n,  sin g_p = -a_p . n     angles to the facet (n its unit
               normal into the water, a the rays' unit vectors of travel on arrival);
               a facet turned away from either ray (sin g <= 0) returns nothing
    azimuth    angle between the projections of a_q and of -a_p on the facet's plane
    shadow     a cell hidden from a platform by nearer relief returns nothing on that leg
    direction, Doppler   from the rays at the platforms towards the true cell, as before

What the eigenrays over relief are:

* Paths that do not touch the bottom before the cell (direct, and surface-reflected) are
  exact for the image-source and free-field providers: the cell is a point in the water.
* Paths with a bottom bounce before the cell are not image paths over a bed that is not
  flat. By default they are left out (`bottom_bounces="exclude"`: what remains is exact,
  and what is lost is small while the range is within a water depth or two and grows to
  several dB in a shallow waveguide with a reflective bed). `bottom_bounces="flat"` keeps
  them as the provider gives them, that is with the earlier bounces off the provider's own
  flat bottom and only the last incidence on the true facet: right in the limit of a
  level bed, an approximation over gentle relief, not shadowed.
* A provider that can trace rays over a depth profile of its own (`along_profile`: the
  Bellhop provider) is used N x 2-D instead: one profile, and one run, per azimuth sector
  along the radial from the platforms, with all its paths (bottom bounces and blocking by
  the relief are then Bellhop's). Both platforms must be over the same ground point, since
  a profile is laid out from the source. Out-of-plane reflection is not modelled.
* Shadowing is tested along the straight line from each path's apparent origin (the
  platform, or its image in the surface) to the cell: exact for straight rays; for a
  refracting provider it is skipped for the paths whose apparent origin is not one point.
"""

from __future__ import annotations

import numpy as np

from . import reverb_bistatic as rb
from .bathymetry import Bathymetry, cross_bearing_slope, out_of_plane_turn
from .propagation import EigenrayProvider

# Slope (rise over run) by which nearer relief must stand above the line of sight to shadow a
# cell: 0.2 mrad, 0.2 m at 1 km. Absorbs the interpolation of the horizon between azimuths.
SHADOW_TOLERANCE = 2e-4

# The cross-slope warning of an N x 2-D run (docs/tasks/PROPAGATION_3D.md, option B; ours). A ray grazing the bed at
# CROSS_SLOPE_GRAZING_DEG leaves its bearing's vertical plane by phi = atan(tan g sin 2 alpha) where the bed slopes
# by alpha across the bearing (bathymetry.out_of_plane_turn); a bearing is flagged where phi exceeds
# CROSS_SLOPE_TURN_DEG somewhere along it. 10 degrees: the energy that carries beyond a few water depths in the
# shipped eigenray waters travels below the bottom's critical angle (20 to 28 degrees for 1600 to 1700 m/s under
# 1500 m/s), and 10 degrees is the middle of that span. 1.8 degrees: half the 3.6 degree receive beam of the two
# long-range arrays (flank_32 at 8 kHz, popoto_long_range at 5 kHz: the arrays that use this tier). Over the
# seamount of test_seamount_3d.py this flags the bearings from -16 to +2 and +10 to +30 degrees (the flanks, where
# Bellhop3D's 3-D field is up to 3.6 dB below N x 2-D, D-149/D-197) and not -20, +6 (the summit's line) or +32.
CROSS_SLOPE_GRAZING_DEG = 10.0
CROSS_SLOPE_TURN_DEG = 1.8


def cross_slope_report(bathymetry: Bathymetry, origin, azimuths, max_range: float, step: float = 10.0,
                       grazing_deg: float = CROSS_SLOPE_GRAZING_DEG, turn_deg: float = CROSS_SLOPE_TURN_DEG) -> dict:
    """Where an N x 2-D run about the horizontal `origin` misses horizontal refraction by the bed: per bearing
    (`azimuths`, radians) the largest cross-bearing slope (degrees) within `max_range` and the out-of-plane turn
    (degrees) of a ray grazing it at `grazing_deg`. `flagged` marks the bearings whose turn exceeds `turn_deg`."""
    az = np.atleast_1d(np.asarray(azimuths, dtype=float))
    r = np.arange(0.0, float(max_range) + step, step)
    slope = np.array([np.abs(cross_bearing_slope(bathymetry, origin, a, r)).max() for a in az])
    turn = np.degrees(np.abs(out_of_plane_turn(slope, np.radians(grazing_deg))))
    return dict(azimuth_deg=np.degrees(az), slope_deg=np.degrees(np.arctan(slope)), turn_deg=turn,
                flagged=turn > turn_deg, grazing_deg=float(grazing_deg), threshold_deg=float(turn_deg))


class Horizon:
    """What a point at `origin` can see of the bed: for directions about it, the lowest
    line-of-sight slope (depth below the origin over horizontal distance) that clears all
    the relief nearer than each distance."""

    def __init__(self, origin, bathymetry: Bathymetry, max_distance: float, step: float, n_azimuth: int):
        self.origin = np.asarray(origin, dtype=float)
        self.step, self.n_az = float(step), int(n_azimuth)
        n_d = int(np.ceil(max_distance / step)) + 3
        self.distance = (np.arange(n_d) + 0.5) * step
        az = np.arange(self.n_az) * 2.0 * np.pi / self.n_az
        x = self.origin[0] + self.distance[None, :] * np.cos(az)[:, None]
        y = self.origin[1] + self.distance[None, :] * np.sin(az)[:, None]
        slope = (bathymetry.depth(x, y) - self.origin[2]) / self.distance[None, :]
        # clear[a, i]: the least slope among the samples at least two steps nearer than i.
        running = np.minimum.accumulate(slope, axis=1)
        self.clear = np.full_like(slope, np.inf)
        self.clear[:, 2:] = running[:, :-2]

    def visible(self, x, y, z) -> np.ndarray:
        dx, dy = x - self.origin[0], y - self.origin[1]
        d = np.hypot(dx, dy)
        slope = (z - self.origin[2]) / np.maximum(d, 1e-9)
        i = np.clip(np.floor(d / self.step).astype(int), 0, len(self.distance) - 1)
        a = (np.arctan2(dy, dx) % (2.0 * np.pi)) / (2.0 * np.pi) * self.n_az
        a0 = np.floor(a).astype(int) % self.n_az
        fa = a - np.floor(a)
        # The horizon between the two neighbouring directions, linearly.
        c0, c1 = self.clear[a0, i], self.clear[(a0 + 1) % self.n_az, i]
        both = np.isfinite(c0) & np.isfinite(c1)
        clear = np.where(both, np.where(both, c0, 0.0) * (1.0 - fa) + np.where(both, c1, 0.0) * fa, np.inf)
        return slope <= clear + SHADOW_TOLERANCE


def _gridded_eigenrays(provider, platform, pts):
    """Eigenrays to bed cells from a provider that looks arrivals up on a depth grid (Bellhop):
    asked for half a grid step above the bed so that the node used is in the water, with the
    delay carried down to the bed along each arrival."""
    dz = float(provider.depths[1] - provider.depths[0]) if len(provider.depths) > 1 else 0.0
    es = provider.eigenrays(platform, pts - np.array([0.0, 0.0, 0.5 * dz]))
    es.delay = es.delay + 0.5 * dz * es.arrival[:, :, 2] / es.sound_speed
    return es


def _leg(provider, platform, pts, normals, bathymetry, shadowing, horizon_args, cache, bottom_bounces):
    """One leg to the cells `pts` [N, 3]: for the usable paths, |amplitude|^2 (zero where the
    path does not light the facet), delay, sin(grazing) on the facet, the launch and arrival
    unit vectors."""
    if getattr(provider, "relief_aware", False):
        es = _gridded_eigenrays(provider, platform, pts)
        shadowing, bottom_bounces = False, "flat"                  # the provider traced the relief
    else:
        es = provider.eigenrays(platform, pts)
    use = es.valid
    if bottom_bounces == "exclude":
        use = use & (es.n_bottom == 0)
    sin_g = -np.einsum("nkc,nc->nk", es.arrival, normals)
    lit = use & (sin_g > 0.0)
    keep = lit.any(axis=0)
    lit, sin_g = lit[:, keep], sin_g[:, keep]
    arrival, launch, delay = es.arrival[:, keep], es.launch[:, keep], es.delay[:, keep]
    bounced = (es.n_bottom[:, keep] > 0).any(axis=0)
    if shadowing:
        for k in range(lit.shape[1]):
            rows = lit[:, k]
            if not rows.any() or bounced[k]:
                continue
            # Straight rays: every cell sees the path come from one point.
            apparent = pts[rows] - arrival[rows, k] * (es.sound_speed * delay[rows, k])[:, None]
            origin = np.median(apparent, axis=0)
            if np.abs(apparent - origin).max() > 0.5:
                continue                                   # a refracted path: no straight line to test
            key = tuple(np.round(origin, 3))
            if key not in cache:
                cache[key] = Horizon(origin, bathymetry, *horizon_args)
            lit[:, k] &= cache[key].visible(pts[:, 0], pts[:, 1], pts[:, 2])
    a2 = np.where(lit, np.abs(es.amplitude[:, keep]) ** 2, 0.0)
    return dict(a2=a2, delay=delay, sin_g=np.clip(sin_g, 0.0, 1.0), arrival=arrival, launch=launch)


def directional_power(provider: EigenrayProvider, tx, rx, bathymetry: Bathymetry, times: np.ndarray,
                      azimuths: np.ndarray, *, frequency: float, mu_db: float = -29.0, strength=None,   # bottom.DEFAULT_LAMBERT_MU_DB (D-89)
                      tx_velocity=(0.0, 0.0, 0.0), rx_velocity=(0.0, 0.0, 0.0), tx_gain=None, rx_gain=None,
                      absorption_db_per_m: float = 0.0, max_range: float | None = None, min_range: float = 0.0,
                      ring_width: float | None = None, n_elevation: int = 1, strength_map=None,
                      subsamples: int = 1, scatterer_velocity=None, scatterer_rms_speed: float = 0.0,
                      scatterer_rms_vertical_speed: float = 0.0, shadowing: bool = True,
                      bottom_bounces: str = "exclude", doppler_resolution: float | None = None,
                      max_doppler_bins: int = 128) -> dict:
    """As `reverb_bistatic.directional_power` (same arguments and result), for a bed given by
    `bathymetry` instead of a flat boundary at one depth. `bottom_bounces`: "exclude" or
    "flat" (see the header)."""
    if bottom_bounces not in ("exclude", "flat"):
        raise ValueError("bottom_bounces: use 'exclude' or 'flat'")
    moving = scatterer_velocity is not None and bool(np.any(np.asarray(scatterer_velocity, dtype=float)))
    random_motion = scatterer_rms_speed > 0.0 or scatterer_rms_vertical_speed > 0.0
    tx, rx = np.asarray(tx, dtype=float), np.asarray(rx, dtype=float)
    vt, vr = np.asarray(tx_velocity, dtype=float), np.asarray(rx_velocity, dtype=float)
    azimuths = np.asarray(azimuths, dtype=float)
    n_az, n_el, n_t = len(azimuths), int(n_elevation), len(times)
    if n_el < 1 or subsamples < 1:
        raise ValueError("n_elevation and subsamples must be at least 1")
    dphi = 2.0 * np.pi / n_az
    dt = times[1] - times[0]
    c = provider.sound_speed
    separation = float(np.hypot(tx[0] - rx[0], tx[1] - rx[1]))
    r_max = max_range or 0.5 * c * (times[-1] + dt) + 0.5 * separation
    dr = ring_width or 0.25 * c * dt
    rings = np.arange(min_range + 0.5 * dr, r_max, dr)
    out = dict(power=np.zeros((n_az, n_el, n_t)), direction=np.zeros((n_az, n_el, 3)), doppler=np.zeros((n_az, n_el)),
               spread=np.zeros((n_az, n_el)), elevation_bin=np.full((n_az, n_el), -1, dtype=np.int64))
    if len(rings) == 0:
        return out
    for name, platform in (("transmitter", tx), ("receiver", rx)):
        if float(bathymetry.depth(platform[0], platform[1])) <= platform[2]:
            raise ValueError(f"reverberation over relief: the {name} is below the seabed")
    if strength is None:
        mu = 10.0 ** (mu_db / 10.0)
        scatter = lambda s_in, s_out, az: mu * s_in * s_out                          # noqa: E731
    elif rb.takes_azimuth(strength):
        scatter = lambda s_in, s_out, az: strength(np.arcsin(s_in), np.arcsin(s_out), az())   # noqa: E731
    else:
        scatter = lambda s_in, s_out, az: strength(np.arcsin(s_in), np.arcsin(s_out))         # noqa: E731
    horizon_args = (rings[-1] + separation + 2.0 * dr, dr, max(720, 4 * n_az * subsamples))
    horizons: dict = {}
    v_s = np.asarray(scatterer_velocity, dtype=float) if moving else np.zeros(3)
    total = rb.ComponentSum(n_az, n_el, n_t, rb.doppler_bins(
        frequency, c, (np.linalg.norm(vt), np.linalg.norm(vr), np.linalg.norm(v_s)), doppler_resolution, max_doppler_bins))
    per_chunk = max(1, rb.CHUNK_CELLS // (4 * len(rings)))
    fan = hasattr(provider, "along_profile")
    if fan:
        if separation > 1.0:
            raise NotImplementedError("reverberation over relief with a ray-tracing provider is N x 2-D about one "
                                      "ground point: the transmitter must be above or below the receiver")
        per_chunk = 1                                              # one bearing, one depth profile, at a time
        _warn_cross_slope(provider, bathymetry, rx, azimuths, rings[-1])
        base_provider = provider
        profile_step = float(min(getattr(provider, "ranges", [0.0, dr])[1] - getattr(provider, "ranges", [0.0, dr])[0], 10.0))
        profile_ranges = np.arange(0.0, rings[-1] + 2.0 * profile_step, profile_step)
    for first in range(0, n_az, per_chunk):
        sectors = np.arange(first, min(first + per_chunk, n_az))
        for sub in range(subsamples):
            phi = azimuths[sectors] + ((sub + 0.5) / subsamples - 0.5) * dphi
            x = (rx[0] + rings[None, :] * np.cos(phi)[:, None]).ravel()
            y = (rx[1] + rings[None, :] * np.sin(phi)[:, None]).ravel()
            pts = np.column_stack([x, y, bathymetry.depth(x, y)])
            if fan:
                along = bathymetry.depth(rx[0] + profile_ranges * np.cos(phi[0]), rx[1] + profile_ranges * np.sin(phi[0]))
                provider = base_provider.along_profile(np.column_stack([profile_ranges, along]))
            normals = bathymetry.normal(x, y)
            cell = np.tile(rings * dr / subsamples, len(sectors)) * bathymetry.area_factor(x, y)
            if strength_map is not None:
                cell = cell * np.asarray(strength_map(x, y), dtype=float)
            forth = _leg(provider, tx, pts, normals, bathymetry, shadowing, horizon_args, horizons, bottom_bounces)
            back = _leg(provider, rx, pts, normals, bathymetry, shadowing, horizon_args, horizons, bottom_bounces)
            comp_base = np.repeat(sectors * n_el, len(rings))
            for q in range(forth["a2"].shape[1]):
                a2_q = forth["a2"][:, q]
                if not a2_q.any():
                    continue
                a_q, u_q = forth["arrival"][:, q], forth["launch"][:, q]
                w_q = a2_q * cell if tx_gain is None else a2_q * cell * tx_gain(u_q)
                for p in range(back["a2"].shape[1]):
                    a2_p = back["a2"][:, p]
                    if not a2_p.any():
                        continue
                    a_p, u_p = back["arrival"][:, p], back["launch"][:, p]

                    def azimuth(a_q=a_q, a_p=a_p):
                        # Incident and scattered directions of travel, projected on the facet.
                        t_in = a_q - np.einsum("nc,nc->n", a_q, normals)[:, None] * normals
                        t_out = -a_p + np.einsum("nc,nc->n", a_p, normals)[:, None] * normals
                        norm = np.linalg.norm(t_in, axis=1) * np.linalg.norm(t_out, axis=1)
                        cos = np.einsum("nc,nc->n", t_in, t_out) / np.where(norm > 1e-12, norm, 1.0)
                        return np.where(norm > 1e-12, np.arccos(np.clip(cos, -1.0, 1.0)), np.pi)

                    t = forth["delay"][:, q] + back["delay"][:, p]
                    pw = w_q * a2_p * scatter(forth["sin_g"][:, q], back["sin_g"][:, p], azimuth)
                    if rx_gain is not None:
                        pw = pw * rx_gain(u_p)
                    if absorption_db_per_m:
                        pw = pw * 10.0 ** (-absorption_db_per_m * c * t / 10.0)
                    elevation = np.arcsin(np.clip(u_p[:, 2], -1.0, 1.0))
                    base = comp_base + np.clip(np.floor((elevation + 0.5 * np.pi) / np.pi * n_el).astype(int), 0, n_el - 1)
                    f_d = (rb.doppler_factor(c, u_q @ vt, u_p @ vr, a_q @ v_s if moving else None,
                                             a_p @ v_s if moving else None) - 1.0) * frequency
                    var = (rb.scatterer_doppler_variance(frequency, c, a_q, a_p, scatterer_rms_speed,
                                                         scatterer_rms_vertical_speed) if random_motion else None)
                    total.add(base, f_d, (t - times[0]) / dt, pw, u_p, var)
    return total.result()


def _warn_cross_slope(provider, bathymetry, origin, azimuths, max_range) -> None:
    """Warn, once per call, where the N x 2-D fan crosses steep cross-bearing relief; the provider keeps the
    largest turn found so far as `cross_slope` (like `truncation`). Nothing else changes."""
    rep = cross_slope_report(bathymetry, origin, azimuths, max_range)
    if not rep["flagged"].any():
        return
    worst = int(np.argmax(rep["turn_deg"]))
    found = dict(turn_deg=float(rep["turn_deg"][worst]), slope_deg=float(rep["slope_deg"][worst]),
                 azimuth_deg=float(rep["azimuth_deg"][worst]), bearings_flagged=int(rep["flagged"].sum()),
                 bearings=len(rep["flagged"]))
    previous = getattr(provider, "cross_slope", None)
    if previous is None or found["turn_deg"] > previous["turn_deg"]:
        provider.cross_slope = found
    import warnings
    warnings.warn(f"reverberation over relief: N x 2-D on {found['bearings_flagged']} of {found['bearings']} bearings where the bed "
                  f"slopes across the bearing (up to {found['slope_deg']:.1f} deg at {found['azimuth_deg']:.0f} deg): a ray grazing it at "
                  f"{CROSS_SLOPE_GRAZING_DEG:g} deg leaves its vertical plane by {found['turn_deg']:.1f} deg (over {CROSS_SLOPE_TURN_DEG:g}). "
                  "Horizontal refraction by the bed is not modelled; over a 12 deg seamount flank it moved levels by 3 to 4 dB "
                  "(docs/tasks/PROPAGATION_3D.md)", RuntimeWarning, stacklevel=3)
