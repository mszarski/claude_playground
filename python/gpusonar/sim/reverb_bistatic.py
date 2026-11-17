"""Reverberation power for a transmitter and a receiver at different places, each moving.

`reverb.reverberation_power` is monostatic: one set of eigenrays from the array centre
serves both legs, so its loci of constant two-way time are circles about the sonar and
every azimuth has the same paths. This module computes the same quantity without those
assumptions, after the directional form of SST's scattering function (TR 0702 Sec. 11.2-11.3;
docs/PHYSICS_NOTES.md Sec. 5.3-5.4, 14 and 17):

* The boundary is cut into cells on a polar grid about the point below (or above) the
  receiver: azimuth sectors, which are also the arrival directions the generator works
  with, times range rings.
* Each cell has receive-leg eigenrays p (from the receiver, used backwards by reciprocity)
  and transmit-leg eigenrays q (from the transmitter). A pair contributes
  |a_q|^2 |a_p|^2 x S(grazing in, grazing out, bistatic azimuth) x cell area x both beam
  patterns at two-way time T_q + T_p. Binning the cells by that time is the integral over
  SST's iso-time locus: for straight rays the cells of one time bin lie between two
  ellipses (the boundary plane cut by spheroids with the platforms as foci; circles when
  the platforms coincide).
* Each contribution also has an arrival direction at the receiver (azimuth of its sector,
  elevation of its ray there) and a Doppler shift,

      f_d = f / c x (V_S . u_Sq + V_R . u_Rp),

  with u the unit vector along the ray leaving each platform towards the cell. This is
  the platform Doppler factor of TR 0702 eq. (71), Gamma_pq = (1 - V_R . S_Rp) /
  (1 + V_S . S_Sq) with slowness vectors S = -u / c, to first order in v / c; the code
  uses the factor itself (`doppler_factor`), f_d = (Gamma_pq - 1) f.
* Contributions are collected into components: (azimuth sector, bin of arrival elevation,
  bin of total Doppler shift), the cells of SST's "perception space" (eq. 129). A component
  carries its power against time, the power-weighted mean of its arrival directions and
  of its Doppler shifts. Because the shift is part of the key, pairs of eigenrays with
  different shifts are never merged, even when they share a receive ray, and the shift of
  the returns from one direction can change along the record. The generator (reverb.py,
  libs/sim/src/reverb.cpp) draws one Gaussian sequence per component.

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


def doppler_factor(sound_speed: float, tx_along, rx_along, scatterer_in=None, scatterer_back=None):
    """The total Doppler factor of an echo, exactly as TR 0702 eqs. (68), (71), (72) have it:

        Gamma gamma = (1 + V_R . u_R / c) / (1 - V_S . u_S / c) x (1 - V_T . a_q / c) / (1 + V_T . a_p / c)

    from the velocity components along the rays: `tx_along` = V_S . u_S and `rx_along` =
    V_R . u_R (u the unit vectors along the rays leaving the platforms towards the
    scatterer), `scatterer_in` = V_T . a_q and `scatterer_back` = V_T . a_p (a the rays'
    directions of travel on arrival at the scatterer). The frequency shift is
    (factor - 1) f."""
    factor = (1.0 + np.asarray(rx_along) / sound_speed) / (1.0 - np.asarray(tx_along) / sound_speed)
    if scatterer_in is not None:
        factor = factor * (1.0 - np.asarray(scatterer_in) / sound_speed) / (1.0 + np.asarray(scatterer_back) / sound_speed)
    return factor


def doppler_bins(frequency: float, sound_speed: float, speeds, resolution: float | None, max_bins: int):
    """Bins of total Doppler for the components (TR 0702 eq. 129): (lowest shift, bin width,
    number of bins). They span every shift the given speeds allow (each platform f v / c
    along its ray, scatterers 2 f v / c), no finer than `resolution` (half the pulse's own
    1 / T: finer cannot be told apart within a pulse) and no more than `max_bins` of them.
    One bin when nothing moves."""
    tx_speed, rx_speed, scatterer_speed = (float(v) for v in speeds)
    half = 1.05 * frequency / sound_speed * (tx_speed + rx_speed + 2.0 * scatterer_speed)
    if half <= 0.0 or max_bins <= 1:
        return 0.0, 1.0, 1
    width = max(2.0 * half / max_bins, float(resolution or 0.0))
    return -half, width, max(int(np.ceil(2.0 * half / width)), 1)


class ComponentSum:
    """Accumulates the contributions of cells into components.

    A component is (azimuth sector, bin of arrival elevation at the receiver, bin of total
    Doppler shift). Contributions with different Doppler shifts are therefore never given
    one shift, whatever their paths and arrival times: two pairs of eigenrays that share a
    receive ray but not a transmit ray fall into different components (unless their shifts
    are within one bin, which a pulse cannot resolve), and so do the early and the late
    returns of one direction when the shift changes along the record. Within a component
    the direction and the shift are power-weighted means.

    Only components that receive power are stored (most (direction, Doppler) cells are
    empty: a direction has a few shifts, one per pair of paths)."""

    def __init__(self, n_sectors: int, n_elevation: int, n_times: int, bins):
        self.n_az, self.n_el, self.n_t = int(n_sectors), int(n_elevation), int(n_times)
        self.f_lo, self.width, self.n_dop = bins
        n_full = self.n_az * self.n_el * self.n_dop
        self.slot = np.full(n_full, -1, dtype=np.int64)
        if self.n_dop == 1:                                        # nothing moves: every (sector, elevation), in order
            self.slot[:] = np.arange(n_full)
            self.ids, capacity = list(range(n_full)), n_full
        else:
            self.ids, capacity = [], 1024
        self._allocate(capacity)

    def _allocate(self, capacity):
        old = getattr(self, "capacity", 0)
        grown = [np.zeros(capacity * self.n_t), np.zeros(capacity), np.zeros((capacity, 4)), np.zeros(capacity)]
        if old:
            grown[0][: old * self.n_t] = self.power
            grown[1][:old], grown[2][:old], grown[3][:old] = self.weight, self.moment, self.variance
        self.power, self.weight, self.moment, self.variance = grown
        self.capacity = capacity

    def add(self, base, shift, time_position, power, direction, variance=None):
        """Contributions [N]: `base` = sector * n_elevation + elevation bin, `shift` their
        Doppler shift (Hz), `time_position` their two-way time in bins from the first bin's
        centre (shared linearly between the two neighbouring bins), `power`, `direction`
        [N, 3] towards the arrival at the receiver, `variance` of the shift from random
        scatterer motion (Hz^2) or None."""
        n_t = self.n_t
        i0 = np.floor(time_position).astype(np.int64)
        frac = time_position - i0
        ok = (power > 0.0) & np.isfinite(power)
        lo, hi = ok & (i0 >= 0) & (i0 < n_t), ok & (i0 >= -1) & (i0 < n_t - 1)
        w_lo, w_hi = np.where(lo, power * (1.0 - frac), 0.0), np.where(hi, power * frac, 0.0)
        w = w_lo + w_hi
        if self.n_dop == 1:
            comp = base
        else:
            used = lo | hi
            if not used.any():
                return
            base, shift, i0, w_lo, w_hi, w, lo, hi, direction = (v[used] for v in (base, shift, i0, w_lo, w_hi, w, lo, hi, direction))
            if variance is not None:
                variance = variance[used]
            full = base * self.n_dop + np.clip(np.floor((shift - self.f_lo) / self.width).astype(np.int64), 0, self.n_dop - 1)
            comp = self.slot[full]
            if (comp < 0).any():
                fresh = np.unique(full[comp < 0])
                self.slot[fresh] = len(self.ids) + np.arange(len(fresh))
                self.ids.extend(fresh.tolist())
                if len(self.ids) > self.capacity:
                    self._allocate(max(2 * self.capacity, len(self.ids)))
                comp = self.slot[full]
        size = self.capacity
        self.power += np.bincount((comp * n_t + i0)[lo], weights=w_lo[lo], minlength=size * n_t)
        self.power += np.bincount((comp * n_t + i0 + 1)[hi], weights=w_hi[hi], minlength=size * n_t)
        self.weight += np.bincount(comp, weights=w, minlength=size)
        if variance is not None:
            self.variance += np.bincount(comp, weights=w * variance, minlength=size)
        for axis in range(3):
            self.moment[:, axis] += np.bincount(comp, weights=w * direction[:, axis], minlength=size)
        self.moment[:, 3] += np.bincount(comp, weights=w * shift, minlength=size)

    def result(self) -> dict:
        """power [sectors, J, n_time], direction [sectors, J, 3], doppler, spread and
        elevation_bin [sectors, J]: the components of each sector along the second axis, in
        order of elevation bin then Doppler; J is the number of elevation bins when nothing
        moves (the position is then the elevation bin), else the largest number of
        components any sector has (the rest are empty, elevation_bin -1)."""
        n = len(self.ids)
        ids = np.asarray(self.ids, dtype=np.int64)
        weight = self.weight[:n]
        # A component more than 120 dB below the strongest is dropped: it would cost a sequence
        # in the generator for nothing, and its mean direction is not defined in double precision.
        has = weight > NEGLIGIBLE * weight.max(initial=0.0)
        power = self.power[: n * self.n_t].reshape(n, self.n_t)
        direction, doppler, spread = np.zeros((n, 3)), np.zeros(n), np.zeros(n)
        mean = self.moment[:n][has, :3] / weight[has, None]
        direction[has] = mean / np.linalg.norm(mean, axis=1, keepdims=True)
        doppler[has] = self.moment[:n][has, 3] / weight[has]
        spread[has] = np.sqrt(self.variance[:n][has] / weight[has])
        sector, elevation = ids // (self.n_el * self.n_dop), (ids // self.n_dop) % self.n_el
        if self.n_dop == 1:
            place, width = elevation, self.n_el
            keep = np.ones(n, dtype=bool)
            power[~has] = 0.0
        else:
            keep = has
            order = np.argsort(ids[keep], kind="stable")
            kept = np.flatnonzero(keep)[order]
            counts = np.bincount(sector[kept], minlength=self.n_az)
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            place = np.zeros(n, dtype=np.int64)
            place[kept] = np.arange(len(kept)) - starts[sector[kept]]
            width = max(int(counts.max(initial=0)), 1)
        out = dict(power=np.zeros((self.n_az, width, self.n_t)), direction=np.zeros((self.n_az, width, 3)),
                   doppler=np.zeros((self.n_az, width)), spread=np.zeros((self.n_az, width)),
                   elevation_bin=np.full((self.n_az, width), -1, dtype=np.int64))
        s, k = sector[keep], place[keep]
        out["power"][s, k], out["direction"][s, k] = power[keep], direction[keep]
        out["doppler"][s, k], out["spread"][s, k], out["elevation_bin"][s, k] = doppler[keep], spread[keep], elevation[keep]
        return out


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
                horizontal=np.hypot(launch[:, :, 0], launch[:, :, 1]), vertical=launch[:, :, 2],
                arrival_horizontal=np.hypot(arrival[:, :, 0], arrival[:, :, 1]), arrival_vertical=arrival[:, :, 2])


def scatterer_doppler(frequency: float, sound_speed: float, velocity, incident, scattered_back):
    """Frequency shift from the scatterers' own mean velocity, first order in v / c:

        -f / c x V_T . (a_q + a_p),

    `incident` = a_q the unit vector of travel of the transmit-leg ray on arrival at the
    scatterer and `scattered_back` = a_p that of the receive-leg ray traced from the
    receiver (the scattered wave travels along -a_p). This is TR 0702 eq. (72),
    gamma = (1 - V_T . S_Tq) / (1 + V_T . S_Tp), with the slowness vectors at the scatterer
    S_Tq = a_q / c and S_Tp = a_p / c (gradients of each leg's travel time with respect to
    the scatterer's position)."""
    k = np.asarray(incident, dtype=float) + np.asarray(scattered_back, dtype=float)
    return -frequency / sound_speed * (k @ np.asarray(velocity, dtype=float))


def scatterer_doppler_variance(frequency: float, sound_speed: float, incident, scattered_back,
                               rms_speed: float = 0.0, rms_vertical_speed: float = 0.0):
    """Variance (Hz^2) of the frequency shift from random scatterer velocities: independent
    Gaussian velocity components of standard deviation `rms_speed` on every axis (volume
    scatterers), plus `rms_vertical_speed` on the vertical only (a boundary that heaves, such
    as the sea surface). The shift of one scatterer is -f / c x v . k with k = a_q + a_p,
    so its variance is (f / c)^2 (rms_speed^2 |k|^2 + rms_vertical_speed^2 k_z^2). For
    backscatter at grazing angle g, |k| = 2 and k_z = 2 sin(g)."""
    k = np.asarray(incident, dtype=float) + np.asarray(scattered_back, dtype=float)
    return (frequency / sound_speed) ** 2 * (rms_speed**2 * np.sum(k * k, axis=-1) + rms_vertical_speed**2 * k[..., 2] ** 2)


def doppler_lines(spread: float, pulse_length: float, max_lines: int = 41):
    """A Gaussian Doppler spread of standard deviation `spread` (Hz) as a set of lines:
    (offsets [n] in Hz, weights [n] summing to 1), with mean 0 and variance spread^2 exactly.
    The lines are no further apart than half the frequency resolution 1 / pulse_length of
    the pulse, so that within a pulse they are indistinguishable from the continuum (the
    bins in total Doppler of TR 0702 eq. (129)); one line when the spread is below a
    twentieth of that resolution."""
    if not spread > 0.0 or spread * pulse_length < 0.05:
        return np.zeros(1), np.ones(1)
    n = int(min(max(np.ceil(6.0 * spread * pulse_length / 0.5) + 1, 5), max_lines))
    n += 1 - n % 2                                                 # odd: one line at the mean
    x = np.linspace(-3.0, 3.0, n)
    w = np.exp(-0.5 * x * x)
    w /= w.sum()
    return spread * x / np.sqrt(np.sum(w * x * x)), w


def directional_power(provider: EigenrayProvider, tx, rx, boundary_z: float, times: np.ndarray,
                      azimuths: np.ndarray, *, frequency: float, mu_db: float = -27.0, strength=None,
                      tx_velocity=(0.0, 0.0, 0.0), rx_velocity=(0.0, 0.0, 0.0), tx_gain=None, rx_gain=None,
                      absorption_db_per_m: float = 0.0, max_range: float | None = None, min_range: float = 0.0,
                      ring_width: float | None = None, n_elevation: int = 1, strength_map=None,
                      subsamples: int = 1, scatterer_velocity=None, scatterer_rms_speed: float = 0.0,
                      scatterer_rms_vertical_speed: float = 0.0, doppler_resolution: float | None = None,
                      max_doppler_bins: int = 128) -> dict:
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

    Returns a dict of the components of each sector (`ComponentSum`: a component is a bin of
    the ray's elevation at the receiver and a bin of total Doppler shift; `doppler_bins`
    with `doppler_resolution`, `max_doppler_bins`), J along the second axis:
      power          [M, J, n_time]
      direction      [M, J, 3]   unit vector from the receiver towards the arrival (power-weighted mean)
      doppler        [M, J]      Hz at `frequency`: (Gamma gamma - 1) f, exact (power-weighted mean)
      spread         [M, J]      Hz: rms Doppler spread from random scatterer motion (power-weighted
                                 mean of the variance), zero without it
      elevation_bin  [M, J]      -1 where there is no component
    With nothing moving J = n_elevation and the position is the elevation bin.

    `scatterer_velocity` (world vector) is the scatterers' mean velocity and
    `scatterer_rms_speed`, `scatterer_rms_vertical_speed` their random motion
    (`scatterer_doppler`, `scatterer_doppler_variance`).
    """
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
    # The cells of two-way time t lie within an ellipse of semi-major axis c t / 2 about the
    # midpoint: no further than c t / 2 + separation / 2 from the receiver.
    r_max = max_range or 0.5 * c * (times[-1] + dt) + 0.5 * separation
    dr = ring_width or 0.25 * c * dt
    rings = np.arange(min_range + 0.5 * dr, r_max, dr)
    out = dict(power=np.zeros((n_az, n_el, n_t)), direction=np.zeros((n_az, n_el, 3)), doppler=np.zeros((n_az, n_el)),
               spread=np.zeros((n_az, n_el)), elevation_bin=np.full((n_az, n_el), -1, dtype=np.int64))
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
    v_s = np.asarray(scatterer_velocity, dtype=float) if moving else np.zeros(3)
    total = ComponentSum(n_az, n_el, n_t, doppler_bins(frequency, c, (np.linalg.norm(vt), np.linalg.norm(vr), np.linalg.norm(v_s)),
                                                       doppler_resolution, max_doppler_bins))
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
                if moving or random_motion:
                    ah_q = forth["arrival_horizontal"][j, q]
                    a_q = np.stack([ah_q * cq, ah_q * sq, forth["arrival_vertical"][j, q]], axis=-1)
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
                    var = None
                    along_in = along_back = None
                    if moving or random_motion:
                        ah_p = back["arrival_horizontal"][None, :, p]
                        a_p = np.stack([ah_p * cp, ah_p * sp,
                                        np.broadcast_to(back["arrival_vertical"][None, :, p], d.shape)], axis=-1)
                        if moving:
                            along_in, along_back = a_q @ v_s, a_p @ v_s
                        if random_motion:
                            var = scatterer_doppler_variance(frequency, c, a_q, a_p, scatterer_rms_speed,
                                                             scatterer_rms_vertical_speed).ravel()
                    f_d = (doppler_factor(c, f_q, u_p @ vr, along_in, along_back) - 1.0) * frequency
                    total.add(np.broadcast_to(comp_base + e_bin[None, :, p], d.shape).ravel(), f_d.ravel(),
                              ((t - times[0]) / dt).ravel(), pw.ravel(), u_p.reshape(-1, 3), var)
    return total.result()


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
