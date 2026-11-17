"""Reverberation power for a transmitter and a receiver at different places, each moving.

`reverb.reverberation_power` is monostatic: one set of eigenrays from the array centre
serves both legs, so its loci of constant two-way time are circles about the sonar and
every azimuth has the same paths. This module computes the same quantity without those
assumptions, after the directional form of SST's scattering function (TR 0702 Sec. 11.2-11.3;
docs/PHYSICS_NOTES.md Sec. 5.3-5.4 and the section on this module at its end):

* The boundary is cut into cells on a polar grid about the point below (or above) the
  receiver: azimuth sectors, which are also the arrival directions the generator works
  with, times range rings.
* Each cell has receive-leg eigenrays p (from the receiver, used backwards by reciprocity)
  and transmit-leg eigenrays q (from the transmitter). A pair contributes
  |a_q|^2 |a_p|^2 x S(grazing in, grazing out, bistatic azimuth) x cell area x both beam
  patterns at two-way time T_q + T_p. Binning the cells by that time is the integral over
  SST's iso-time locus: for straight rays the cells of one time bin lie between two
  ellipses with the platforms' ground points as foci (a circle when they coincide).
* Each contribution also has an arrival direction at the receiver (azimuth of its sector,
  elevation of its ray there) and a Doppler shift,

      f_d = f / c x (V_S . u_Sq + V_R . u_Rp),

  with u the unit vector along the ray leaving each platform towards the cell. This is
  the platform Doppler factor of TR 0702 eq. (71), Gamma_pq = (1 - V_R . S_Rp) /
  (1 + V_S . S_Sq) with slowness vectors S = -u / c, to first order in v / c.
* Contributions are collected into components: (azimuth sector, bin of arrival elevation).
  A component carries its power against time, the power-weighted mean of its arrival
  directions and the power-weighted mean of its Doppler shifts. The generator
  (reverb.py, libs/sim/src/reverb.cpp) draws one Gaussian sequence per component.

The environment is taken as horizontally stratified with a flat boundary, as in the
monostatic code: the eigenrays of a leg then depend only on the horizontal distance from
its platform, so each leg needs one provider call along a line of distances. The transmit
leg is looked up in a table half a ring apart, with its delay carried to the cell's own
distance by the ray's horizontal slowness (exact to first order for any stratified medium).

The boundary's strength may vary over the boundary through `strength_map`, a function of
world (x, y) returning a linear factor on the scattering strength.
"""

from __future__ import annotations

import inspect

import numpy as np

from .propagation import EigenrayProvider

# Cells handled at once (sectors x rings): bounds the temporaries of one pair of paths.
CHUNK_CELLS = 1 << 20
# Components with less than this fraction of the strongest component's power are dropped.
NEGLIGIBLE = 1e-12


def takes_azimuth(strength) -> bool:
    """True when a scattering-strength function takes the bistatic azimuth as a third
    argument (sim/bottom.py's does); the older ones take the two grazing angles only."""
    try:
        params = inspect.signature(strength).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 3 or any(p.kind == p.VAR_POSITIONAL for p in params)


def bistatic_azimuth(incident_azimuth, scattered_azimuth):
    """Angle between the horizontal directions of travel of the incident and scattered
    waves, in [0, pi]: pi for backscatter, 0 for forward scatter (the convention of
    sim/bottom.py)."""
    return np.abs(np.angle(np.exp(1j * (np.asarray(scattered_azimuth) - np.asarray(incident_azimuth)))))


def doppler_shift(frequency: float, sound_speed: float, tx_velocity, tx_direction, rx_velocity, rx_direction):
    """Frequency shift of an echo from a fixed scatterer, first order in v / c:
    f / c x (V_S . u_S + V_R . u_R), u the unit vectors along the rays leaving the
    transmitter and the receiver towards the scatterer ([..., 3])."""
    return frequency / sound_speed * (np.asarray(tx_direction) @ np.asarray(tx_velocity, dtype=float)
                                      + np.asarray(rx_direction) @ np.asarray(rx_velocity, dtype=float))


def _leg(es, side):
    """Per-path arrays of one leg, [points, paths], for the paths that are incident on the
    boundary somewhere: |amplitude|^2 (zero where not incident), delay, grazing angle at
    the boundary, horizontal slowness there, and the launch direction's horizontal part
    and vertical component at the platform."""
    incident = es.valid & (es.arrival[:, :, 2] * side > 0.0)
    keep = incident.any(axis=0)
    a2 = np.where(incident, np.abs(es.amplitude) ** 2, 0.0)[:, keep]
    arrival, launch = es.arrival[:, keep], es.launch[:, keep]
    graze = np.arcsin(np.clip(np.abs(arrival[:, :, 2]), 0.0, 1.0))
    slowness = np.hypot(arrival[:, :, 0], arrival[:, :, 1]) / es.sound_speed
    return dict(a2=a2, delay=np.where(a2 > 0.0, es.delay[:, keep], 0.0), graze=graze, slowness=slowness,
                horizontal=np.hypot(launch[:, :, 0], launch[:, :, 1]), vertical=launch[:, :, 2])


def directional_power(provider: EigenrayProvider, tx, rx, boundary_z: float, times: np.ndarray,
                      azimuths: np.ndarray, *, frequency: float, mu_db: float = -27.0, strength=None,
                      tx_velocity=(0.0, 0.0, 0.0), rx_velocity=(0.0, 0.0, 0.0), tx_gain=None, rx_gain=None,
                      absorption_db_per_m: float = 0.0, max_range: float | None = None, min_range: float = 0.0,
                      ring_width: float | None = None, n_elevation: int = 1, strength_map=None,
                      subsamples: int = 1) -> dict:
    """Reverberation power per radian of receiver azimuth in each time bin, for unit source
    amplitude at 1 m, from a flat boundary at depth `boundary_z`, by arrival direction.

    `tx`, `rx` are the platform positions and `azimuths` [M] the centres of uniform azimuth
    sectors about the receiver covering the circle. `times` are the centres of uniform bins
    of two-way time. `strength` is a function of (grazing in, grazing out) or of (grazing
    in, grazing out, bistatic azimuth) returning the linear scattering strength per unit
    area; without it, Lambert's law with `mu_db`. `tx_gain`, `rx_gain` are power patterns of
    world unit vectors [..., 3] (None: omnidirectional). `max_range`, `min_range` bound the
    rings by horizontal distance from the receiver. `subsamples` azimuths are evaluated
    within each sector (more when the transmitter is far from the receiver and close to
    the boundary, where a sector is wide compared with the transmitter's height).

    Returns a dict of
      power      [M, E, n_time]  E = n_elevation bins of the ray's elevation at the receiver
      direction  [M, E, 3]       unit vector from the receiver towards the arrival (power-weighted mean)
      doppler    [M, E]          Hz at `frequency` (power-weighted mean)
    """
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
    # The cells of two-way time t lie within an ellipse of semi-major axis c t / 2 about the
    # midpoint: no further than c t / 2 + separation / 2 from the receiver.
    r_max = max_range or 0.5 * c * (times[-1] + dt) + 0.5 * separation
    dr = ring_width or 0.25 * c * dt
    rings = np.arange(min_range + 0.5 * dr, r_max, dr)
    out = dict(power=np.zeros((n_az, n_el, n_t)), direction=np.zeros((n_az, n_el, 3)), doppler=np.zeros((n_az, n_el)))
    if len(rings) == 0:
        return out
    side = 1.0 if boundary_z >= rx[2] else -1.0
    if (boundary_z >= tx[2]) != (boundary_z >= rx[2]) and tx[2] != boundary_z and rx[2] != boundary_z:
        raise ValueError("bistatic reverberation: the boundary must be on the same side of both platforms")
    back = _leg(provider.eigenrays(rx, np.column_stack([rx[0] + rings, np.full_like(rings, rx[1]),
                                                        np.full_like(rings, boundary_z)])), side)
    # Transmit leg against horizontal distance from the transmitter, half a ring apart.
    dd = 0.5 * dr
    d_first = max(0.0, min_range - separation, separation - (rings[-1] + 0.5 * dr))
    table = d_first + (np.arange(int(np.ceil((rings[-1] + 0.5 * dr + separation - d_first) / dd)) + 1) + 0.5) * dd
    forth = _leg(provider.eigenrays(tx, np.column_stack([tx[0] + table, np.full_like(table, tx[1]),
                                                         np.full_like(table, boundary_z)])), side)
    k_t, k_r = forth["a2"].shape[1], back["a2"].shape[1]
    if k_t == 0 or k_r == 0:
        return out
    if strength is None:
        mu = 10.0 ** (mu_db / 10.0)
        scatter = lambda g_in, g_out, az: mu * np.sin(g_in) * np.sin(g_out)        # noqa: E731
    elif takes_azimuth(strength):
        scatter = strength
    else:
        scatter = lambda g_in, g_out, az: strength(g_in, g_out)                     # noqa: E731
    area = rings * dr                                              # per radian of receiver azimuth
    # Elevation bin of each receive path on each ring (the ray's vertical direction cosine
    # at the receiver: +1 straight down in NED).
    elevation = np.arcsin(np.clip(back["vertical"], -1.0, 1.0))
    e_bin = np.clip(np.floor((elevation + 0.5 * np.pi) / np.pi * n_el).astype(int), 0, n_el - 1)   # [rings, k_r]
    n_comp = n_az * n_el
    power = np.zeros(n_comp * n_t)
    weight = np.zeros(n_comp)
    moment = np.zeros((n_comp, 4))                                 # sum of power x (direction, Doppler)
    per_chunk = max(1, CHUNK_CELLS // len(rings))
    for first in range(0, n_az, per_chunk):
        sectors = np.arange(first, min(first + per_chunk, n_az))
        for sub in range(subsamples):
            phi = azimuths[sectors] + ((sub + 0.5) / subsamples - 0.5) * dphi
            cp, sp = np.cos(phi)[:, None], np.sin(phi)[:, None]
            x, y = rx[0] + rings[None, :] * cp, rx[1] + rings[None, :] * sp          # [sectors, rings]
            ox, oy = x - tx[0], y - tx[1]
            d = np.hypot(ox, oy)
            psi = np.arctan2(oy, ox)                                                 # azimuth transmitter -> cell
            cq, sq = np.cos(psi), np.sin(psi)
            j = np.clip(np.rint((d - table[0]) / dd).astype(int), 0, len(table) - 1)
            beyond = d - table[j]
            # The scattered wave travels from the cell back to the receiver: azimuth phi + pi.
            az_b = bistatic_azimuth(psi, phi[:, None] + np.pi)
            cell = np.broadcast_to(area[None, :] / subsamples, d.shape)
            if strength_map is not None:
                cell = cell * np.asarray(strength_map(x, y), dtype=float)
            comp_base = (sectors * n_el)[:, None]
            for q in range(k_t):
                a2_q = forth["a2"][j, q]
                if not a2_q.any():
                    continue
                t_q = forth["delay"][j, q] + beyond * forth["slowness"][j, q]
                g_q, h_q, v_q = forth["graze"][j, q], forth["horizontal"][j, q], forth["vertical"][j, q]
                u_q = np.stack([h_q * cq, h_q * sq, v_q], axis=-1)                   # leaving the transmitter
                w_q = a2_q * cell if tx_gain is None else a2_q * cell * tx_gain(u_q)
                f_q = u_q @ vt
                for p in range(k_r):
                    a2_p = back["a2"][:, p]
                    if not a2_p.any():
                        continue
                    h_p, v_p = back["horizontal"][None, :, p], back["vertical"][None, :, p]
                    u_p = np.stack([h_p * cp, h_p * sp, np.broadcast_to(v_p, d.shape)], axis=-1)   # leaving the receiver
                    t = t_q + back["delay"][None, :, p]
                    pw = w_q * a2_p[None, :] * scatter(g_q, back["graze"][None, :, p], az_b)
                    if rx_gain is not None:
                        pw = pw * rx_gain(u_p)
                    if absorption_db_per_m:
                        pw = pw * 10.0 ** (-absorption_db_per_m * c * t / 10.0)
                    # Shared between the two neighbouring time bins, linearly.
                    pos = (t - times[0]) / dt
                    i0 = np.floor(pos).astype(np.int64)
                    frac = pos - i0
                    ok = (pw > 0.0) & np.isfinite(pw)
                    lo, hi = ok & (i0 >= 0) & (i0 < n_t), ok & (i0 >= -1) & (i0 < n_t - 1)
                    comp = comp_base + e_bin[None, :, p]
                    w_lo, w_hi = np.where(lo, pw * (1.0 - frac), 0.0), np.where(hi, pw * frac, 0.0)
                    power += np.bincount((comp * n_t + i0)[lo], weights=w_lo[lo], minlength=n_comp * n_t)
                    power += np.bincount((comp * n_t + i0 + 1)[hi], weights=w_hi[hi], minlength=n_comp * n_t)
                    w = (w_lo + w_hi).ravel()
                    flat = comp.ravel()
                    weight += np.bincount(flat, weights=w, minlength=n_comp)
                    f_d = (frequency / c) * (f_q + u_p @ vr)
                    for axis in range(3):
                        moment[:, axis] += np.bincount(flat, weights=w * u_p[..., axis].ravel(), minlength=n_comp)
                    moment[:, 3] += np.bincount(flat, weights=w * f_d.ravel(), minlength=n_comp)
    # A component more than 120 dB below the strongest is dropped: it would cost a sequence
    # in the generator for nothing, and its mean direction is not defined in double precision.
    has = weight > NEGLIGIBLE * weight.max(initial=0.0)
    power = power.reshape(n_comp, n_t)
    power[~has] = 0.0
    direction = np.zeros((n_comp, 3))
    mean = moment[has, :3] / weight[has, None]
    direction[has] = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    doppler = np.zeros(n_comp)
    doppler[has] = moment[has, 3] / weight[has]
    out["power"] = power.reshape(n_az, n_el, n_t)
    out["direction"] = direction.reshape(n_az, n_el, 3)
    out["doppler"] = doppler.reshape(n_az, n_el)
    return out


def table_strength_map(ranges, offset_db, azimuth_deg=None, origin=(0.0, 0.0)):
    """A boundary whose strength varies with horizontal range, and optionally azimuth, about a
    point fixed in the world: the function of (x, y) that `directional_power` takes.

    `offset_db` [len(ranges)] or [len(azimuth_deg), len(ranges)] is added to the boundary's
    scattering strength (dB). It is interpolated linearly in range (held beyond the ends)
    and in azimuth (degrees from x = north towards y = east, around the circle).
    """
    ranges = np.asarray(ranges, dtype=float)
    table = np.asarray(offset_db, dtype=float)
    if ranges.ndim != 1 or len(ranges) < 1 or np.any(np.diff(ranges) <= 0.0):
        raise ValueError("strength_map: range must be increasing")
    if azimuth_deg is None:
        if table.shape != ranges.shape:
            raise ValueError("strength_map: offset_db must have one value per range")
        table, az = table[None, :], None
    else:
        az = np.radians(np.asarray(azimuth_deg, dtype=float)) % (2.0 * np.pi)
        if table.shape != (len(az), len(ranges)):
            raise ValueError("strength_map: offset_db must have shape [azimuths, ranges]")
        order = np.argsort(az)
        az, table = az[order], table[order]
        if np.any(np.diff(az) <= 0.0):
            raise ValueError("strength_map: azimuths must be distinct")
    origin = (float(origin[0]), float(origin[1]))

    def factor(x, y):
        ox, oy = np.asarray(x, dtype=float) - origin[0], np.asarray(y, dtype=float) - origin[1]
        r = np.clip(np.hypot(ox, oy), ranges[0], ranges[-1])
        hi = np.clip(np.searchsorted(ranges, r, side="right"), 1, max(len(ranges) - 1, 1))
        lo = hi - 1
        if len(ranges) == 1:
            hi = lo
            fr = np.zeros_like(r)
        else:
            fr = (r - ranges[lo]) / (ranges[hi] - ranges[lo])
        if az is None:
            db = table[0][lo] * (1.0 - fr) + table[0][hi] * fr
        else:
            a = np.arctan2(oy, ox) % (2.0 * np.pi)
            k1 = np.searchsorted(az, a, side="right") % len(az)
            k0 = (k1 - 1) % len(az)
            span = (az[k1] - az[k0]) % (2.0 * np.pi)
            fa = np.where(span > 0.0, ((a - az[k0]) % (2.0 * np.pi)) / np.where(span > 0.0, span, 1.0), 0.0)
            db = ((table[k0, lo] * (1.0 - fr) + table[k0, hi] * fr) * (1.0 - fa)
                  + (table[k1, lo] * (1.0 - fr) + table[k1, hi] * fr) * fa)
        return 10.0 ** (db / 10.0)

    return factor
