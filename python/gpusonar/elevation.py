"""Elevation, and from it depth, with a flank array of two or more stacked rows.

One horizontal line of hydrophones measures the angle between an arrival and its own axis
(the cone angle) and nothing else: an arrival from above or below is reported at a bearing
that is too small, and its depth is unknown. A second line at a different height on the
vehicle (and a third, at an unequal spacing) measures the missing angle.

What the live chain does with rows (`chain.elevation` in a deployment file; off by default,
and a record with one row is processed exactly as before):

  map        each row is beamformed on its own over range x bearing and the rows' power maps are
             added. A row's bearing response does not depend on the elevation of the arrival,
             so nothing is lost on arrivals from above or below (a coherent sum of rows
             steered at the horizon would lose them).
  elevation  at each detection the whole array is steered over elevation at the detection's cone
             angle and range, by the same backprojection kernel as the map (true time delays, so
             exact for the 4-6 kHz band and for rows that are not in one plane). The peak is
             the maximum-likelihood elevation of one arrival in noise. With it come a quality
             (the coherent power over the rows' incoherent power: 1 for a single plane wave),
             a one-sigma from the Cramer-Rao bound at the measured signal-to-noise ratio, and
             a flag when a second lobe is within a margin of the first (ambiguous).
  bearing    cone angle and elevation give the direction; the horizontal bearing follows.
  depth      own depth + slant range x the vertical part of that direction, with the
             vehicle's roll and pitch from the record's pose. An arrival that came by the
             surface or the bottom lands at its mirror image above the surface or below the
             bottom; folded back into the water column it is the target's depth again, and
             where it landed says which path it took.

The same functions exist in libs/live (elevation.hpp); tests/python/test_elevation.py compares
them. Formulas for the bound are derived in docs/DESIGN.md Sec. 3.6 and checked by Monte
Carlo in tools/popoto_elevation_study.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _gpusonar as core

DEFAULTS = {
    "sector_deg": [-30.0, 30.0, 0.25],  # elevation searched, in the vehicle's frame: first, last, step (up positive)
    "sector_from_depth": None,          # {bounces: 1, margin_deg: 1.0}: search only where water depth and range let an arrival
                                        # come from (straight rays, this many boundary bounces); needs the water depth
    "range_window_m": None,             # half-width of the range search about a detection; None: 1.5 range cells
    "max_detections": 32,               # strongest detections given an elevation, per ping
    "arrival": "first",                 # "first": the earliest arrival in the range window that is within
    "first_arrival_db": 8.0,            #   this of the strongest and
    "first_arrival_snr_db": 13.0,       #   this far above the noise (the direct path arrives first);
                                        # "strongest": the strongest arrival
    "min_quality": 0.6,                 # below this (several arrivals in one cell, interference) no elevation is reported
    "ambiguity_db": 1.0,                # a second lobe within this of the first is never trusted, and
    "ambiguity_sigma": 3.0,             # one further down must lose to the first by this many sigma of the noise: else the
                                        # detection is flagged ambiguous and its candidates go to the track, which picks
    "alternatives_db": 6.0,             # lobes within this of the first are the candidates
    "correct_bearing": True,            # report and track the horizontal bearing, not the cone angle
    "depth": True,                      # report depth, and let tracks carry it
    "fold": True,                       # fold a depth outside the water column back in (needs the water depth)
    "classify": True,                   # a track whose depth lies outside the water column is multipath
    "classify_sigma": 2.0,              # ... by more than this many sigma
    "water_depth": None,                # m; None: the record's environment.water_depth (navigation's, else unknown)
    "water_depth_sigma_m": 0.0,         # how well it is known: into the bottom test and the depth of bottom-bounce arrivals
    "attitude_time": "transmit",        # "arrival": roll, pitch and yaw as they were when each echo arrived, from the attitude
                                        # history in the record (navigation.ins); "transmit": as the record's pose has them
    "attitude_rate_sigma_deg_s": 0.0,   # with "arrival": added uncertainty per second of gap to the nearest attitude sample
    "gradient_per_s": None,             # assumed mean vertical sound-speed gradient (1/s; 0.017 = isothermal water): depth_m
                                        # along the ray through c(z) = local speed + g (z - own depth). None: straight ray
    "gradient_sigma_per_s": 0.05,       # how well that gradient is known: depth sigma gains range^2 x this / 2c
    "surface_z": 0.0,                   # z of the sea surface in the frame the navigation uses
    "ssp": None,                        # {depth: [m...], speed: [m/s...]}: depth_m by a ray traced through this profile
                                        # instead of the straight line (which is wrong by s^2 g / 2c in a gradient g).
                                        # Also a table [[depth, speed], ...], or {file: profile.csv} (two columns)
    "ray_method": "steps",              # how depth_m is traced back through a profile: "steps" (Snell's law in small steps, as
                                        # always) or "arcs" (closed-form arcs layer by layer: the chain's one ray calculation,
                                        # libs/live raytrace.hpp, shared with multipath_depth.rays and chain.coverage)
    "profile": None,                    # the sound-speed ladder: {} or {ladder: [...], built: {...}, supplied: {...},
                                        # family: {...}} (PROFILE_DEFAULTS below). None: as before
    "roll_sigma_deg": 0.0,              # the navigation's roll accuracy, added to the elevation sigma
    "depth_process_noise": 0.5,         # m per sqrt(s): how fast a track's depth may wander
    "row_tolerance_m": None,            # elements within this of one line are one row; None: derived (2 cm, or a quarter of
                                        # the shortest wavelength when that is less: libs/live derive.hpp)
    "min_row_elements": 3,              # a line of fewer hydrophones than this (dead boards) is not a row
}


# The sound-speed ladder (libs/live environment.hpp explains it; the rungs are tried in this order
# and the one used is named with every depth as `depth_profile_source`).
PROFILE_DEFAULTS = {
    "ladder": ["supplied", "built", "family", "local", "prior"],
    "built": {                          # a profile the vehicle builds from its own (depth, sound speed) samples
        "bin_m": 2.0,                   # samples are binned by depth
        "sample_sigma_m_s": None,       # the sound-speed sensor's accuracy: no default, the built rung needs it (datasheet)
        "age_sigma_m_s_per_hour": 0.5,  # a bin loses this much certainty per hour since it was sampled (placeholder, unconfirmed)
        "distance_sigma_m_s_per_km": 0.5,   # and this per km from where it was sampled (placeholder, unconfirmed)
        "max_age_s": None,              # bins older than this are not used
        "max_distance_m": None,         # nor bins sampled further away than this
        "min_span_m": 5.0,              # used once the sampled depths span this much
        "extrapolation_sigma_per_s": None,  # outside the sampled depths sigma grows by this per metre; None: gradient_sigma_per_s
        "file": None,                   # the bins are kept in this file, so a restart keeps the profile
        "cast": None,                   # a cast to start from (a static array): text lines "depth speed"
    },
    "supplied": {"sigma_m_s": None},    # how well elevation.ssp holds along the path; None: no claim (the gradient prior)
    "ensemble": None,                   # {gradients_per_s: [...], extent_m: ..., sigma_members: true}: where the vehicle has not
                                        # sampled the water, depths (and the coverage picture) are also evaluated through a small
                                        # set of profiles: the unsampled water continued with each of these gradients for extent_m
                                        # (a length or a list of lengths; every pairing above and below), and the sampled part
                                        # moved by its sigma each way. An assumed family counts as sampled at the vehicle only.
                                        # The spread is reported (depth_ensemble_m, depth_multipath_ensemble_m,
                                        # max_range_ensemble_m). gradients_per_s and extent_m have no default
    "family": None,                     # an assumption: {kind: isothermal, gradient_per_s: ...} or {kind: mixed_layer,
                                        # gradient_per_s, layer_depth_m, thermocline_gradient_per_s, thermocline_thickness_m};
                                        # sigma_m_s: how well it holds (None: no claim). No parameter has a default
}


def normalise_ssp(ssp):
    """`elevation.ssp` in any of its forms as {depth: [...], speed: [...]} (increasing depth), or None.
    Forms: that dict; a table [[depth, speed], ...]; {file: path} or a path to a text file of two
    columns (depth m, speed m/s; commas or blanks; lines that do not start with a number are skipped)."""
    if not ssp:
        return None
    if isinstance(ssp, dict) and "file" not in ssp:
        depth, speed = ssp["depth"], ssp["speed"]
    else:
        if isinstance(ssp, (str, dict)) or hasattr(ssp, "__fspath__"):
            rows = []
            with open(ssp["file"] if isinstance(ssp, dict) else ssp, encoding="utf-8") as f:
                for line in f:
                    parts = line.split("#")[0].replace(",", " ").replace(";", " ").split()
                    try:
                        rows.append((float(parts[0]), float(parts[1])))
                    except (ValueError, IndexError):
                        continue
        else:
            rows = [(float(r[0]), float(r[1])) for r in ssp]
        rows.sort()
        depth, speed = [r[0] for r in rows], [r[1] for r in rows]
    depth, speed = [float(v) for v in depth], [float(v) for v in speed]
    if len(depth) != len(speed) or len(depth) < 2 or np.any(np.diff(depth) <= 0.0):
        raise ValueError("elevation.ssp: needs two or more (depth, speed) pairs at increasing depths")
    return {"depth": depth, "speed": speed}


def profile_options(profile: dict | None) -> dict | None:
    """`elevation.profile` as the options of core.SoundSpeedLadder (and of the C++ chain)."""
    if profile is None:
        return None
    unknown = set(profile) - set(PROFILE_DEFAULTS)
    if unknown:
        raise ValueError(f"elevation.profile: unknown key(s) {sorted(unknown)}")
    b = dict(PROFILE_DEFAULTS["built"], **(profile.get("built") or {}))
    s = dict(PROFILE_DEFAULTS["supplied"], **(profile.get("supplied") or {}))
    out = {"ladder": [str(r) for r in profile.get("ladder", PROFILE_DEFAULTS["ladder"])], "bin_m": float(b["bin_m"]),
           "sample_sigma": float(b["sample_sigma_m_s"] or 0.0), "age_sigma_per_hour": float(b["age_sigma_m_s_per_hour"]),
           "distance_sigma_per_km": float(b["distance_sigma_m_s_per_km"]), "max_age_s": float(b["max_age_s"] or 0.0),
           "max_distance_m": float(b["max_distance_m"] or 0.0), "min_span_m": float(b["min_span_m"]),
           "extrapolation_sigma_per_s": float(b["extrapolation_sigma_per_s"] or 0.0), "supplied_sigma": float(s["sigma_m_s"] or 0.0)}
    for key in ("file", "cast"):
        if b[key]:
            out[key] = str(b[key])
    ens = profile.get("ensemble")
    if ens is not None:
        unknown = set(ens) - {"gradients_per_s", "extent_m", "sigma_members"}
        if unknown or ens.get("gradients_per_s") is None or ens.get("extent_m") is None:
            raise ValueError("elevation.profile.ensemble: needs gradients_per_s (a list) and extent_m; sigma_members is the only other key")
        extents = ens["extent_m"] if isinstance(ens["extent_m"], (list, tuple)) else [ens["extent_m"]]
        out.update(ensemble=True, ensemble_gradients=[float(v) for v in ens["gradients_per_s"]], ensemble_extents=[float(v) for v in extents],
                   ensemble_sigma_members=bool(ens.get("sigma_members", True)))
    fam = profile.get("family")
    if fam:
        names = {"gradient_per_s": "family_gradient", "layer_depth_m": "layer_depth", "thermocline_gradient_per_s": "thermocline_gradient",
                 "thermocline_thickness_m": "thermocline_thickness", "sigma_m_s": "family_sigma"}
        unknown = set(fam) - set(names) - {"kind"}
        if unknown:
            raise ValueError(f"elevation.profile.family: unknown key(s) {sorted(unknown)}")
        out["family"] = str(fam["kind"])
        out.update({names[k]: float(v) for k, v in fam.items() if k in names and v is not None})
    return out


def tilted_profiles(own_depth: float, depth, speed, sigma, gradient_sigma: float):
    """The two profiles a depth's refraction sigma is the half difference of: (knots, c + e, c - e)
    with e(z) = sign(z - own depth) min(sigma(z), gradient_sigma |z - own depth|): the profile's
    error taken as one tilt about the vehicle, where the sound speed is measured, and never
    larger than the prior allows. NumPy counterpart of libs/live `tilted_profiles`."""
    depth, speed, sigma = (np.asarray(v, dtype=float) for v in (depth, speed, sigma))
    g = abs(float(gradient_sigma))
    excess = lambda z: np.interp(z, depth, sigma) - g * np.abs(z - own_depth)      # noqa: E731
    base = np.unique(np.append(depth, own_depth)) if depth[0] < own_depth < depth[-1] else depth
    knots = [base[0]]
    for a, b in zip(base[:-1], base[1:]):
        ea, eb = excess(a), excess(b)
        if (ea < 0.0) != (eb < 0.0) and ea != eb:      # both limits are straight between knots: where they cross
            z = a + (b - a) * ea / (ea - eb)
            if a < z < b:
                knots.append(z)
        knots.append(b)
    knots = np.array(knots)
    e = np.where(knots < own_depth, -1.0, 1.0) * np.minimum(np.interp(knots, depth, sigma), g * np.abs(knots - own_depth))
    c = np.interp(knots, depth, speed)
    return knots, c + e, c - e


def profile_depth_sigma(receiver_depth: float, elevation: float, slant_range: float, depth, speed, sigma, gradient_sigma: float,
                        water_depth: float | None = None) -> float:
    """One sigma of `ray_depth` through a profile whose error at its knots is `sigma` (m/s): the
    sensitivity of the depth to the tilt of `tilted_profiles`, measured with a twentieth of it each
    way and scaled back up (linear, and not folded at the boundaries: a sigma beyond the water depth
    says the depth is not known)."""
    knots, fast, slow = tilted_profiles(receiver_depth, depth, speed, sigma, gradient_sigma)
    mid, half = 0.5 * (fast + slow), 0.5 * (fast - slow)
    eps = 0.05
    return 0.5 * abs(ray_depth(receiver_depth, elevation, slant_range, knots, mid + eps * half, water_depth)
                     - ray_depth(receiver_depth, elevation, slant_range, knots, mid - eps * half, water_depth)) / eps


# --- rows and frame -------------------------------------------------------------------------------

def _levers(record) -> np.ndarray:
    return np.array([e.lever_arm for e in record.geometry.rx], dtype=float).reshape(-1, 3)


def row_axis(levers: np.ndarray) -> np.ndarray:
    """Unit vector along the rows, body frame: the median step between consecutive channels
    (channels run row by row, so most steps are along a row)."""
    d = np.diff(levers, axis=0)
    norm = np.linalg.norm(d, axis=1)
    d = d[norm > 0.0] / norm[norm > 0.0, None]
    if len(d) == 0:
        raise ValueError("row_axis: needs at least two elements at different positions")
    a = np.median(d, axis=0)
    return a / np.linalg.norm(a)


def find_rows(record, tolerance: float = 0.02, min_elements: int = 3) -> list[list[int]]:
    """Channel rows of each line of the receive array, in channel order. Elements whose
    positions across the row direction agree within `tolerance` (m) are one row; rows with
    fewer than `min_elements` channels left (dead boards) are not used."""
    levers = _levers(record)
    if len(levers) < 2:
        return [list(range(len(levers)))]
    a = row_axis(levers)
    q = levers - np.outer(levers @ a, a)
    rows: list[list[int]] = []
    anchors: list[np.ndarray] = []
    for i, p in enumerate(q):
        for k, anchor in enumerate(anchors):
            if np.linalg.norm(p - anchor) <= tolerance:
                rows[k].append(i)
                break
        else:
            rows.append([i])
            anchors.append(p)
    return [r for r in rows if len(r) >= min_elements]


@dataclass
class Frame:
    origin: np.ndarray     # world: mean position of the elements used
    along: np.ndarray      # unit, along the rows towards increasing channel number
    up: np.ndarray         # unit, the vehicle's up made perpendicular to `along`
    look: np.ndarray       # unit, perpendicular to both, on the side the array looks
    heights: np.ndarray    # [N] element positions along `up`, m about the origin (the elements used)


def array_frame(record, rows, look: str = "starboard", rpy=None) -> Frame:
    """The stacked array's own frame in the world, from the record's pose (or with another
    attitude `rpy`). Elevation is measured from the (along, look) plane towards `up`: the
    vehicle's deck plane, so roll and pitch are in it and the world direction that comes out
    is level-referenced."""
    used = [i for r in rows for i in r]
    levers = _levers(record)
    offsets = record.calibration.position_offset
    if len(offsets) == len(levers):
        body = (levers + np.asarray(offsets, dtype=float).reshape(-1, 3))[used]
    else:
        body = levers[used]
    pose = record.geometry.rx_pose
    rot = np.asarray(core.rotation_ned(*(pose.rpy if rpy is None else rpy)))
    pos = np.asarray(pose.position) + body @ rot.T
    origin = pos.mean(axis=0)
    along = rot @ row_axis(levers)
    up = rot @ np.array([0.0, 0.0, -1.0])
    up = up - along * (along @ up)
    up /= np.linalg.norm(up)
    side = 1.0 if look == "starboard" else -1.0
    return Frame(origin, along, up, side * np.cross(along, up), (pos - origin) @ up)


def attitude_at(record, t: float):
    """(rpy, gap) from the record's attitude history (navigation.ins: samples at seconds after
    the transmit) at time `t`: interpolated between the samples around it (gap 0), or the
    nearest sample and how far away in time it is. (None, None) without a history."""
    ins = record.navigation.ins
    if len(ins) == 0:
        return None, None
    times = np.array([a.t for a in ins])
    rpy = np.array([a.rpy for a in ins])
    i = int(np.searchsorted(times, t))
    if i == 0:
        return rpy[0], float(times[0] - t)
    if i == len(times):
        return rpy[-1], float(t - times[-1])
    w = (t - times[i - 1]) / max(times[i] - times[i - 1], 1e-12)
    return rpy[i - 1] + w * np.angle(np.exp(1j * (rpy[i] - rpy[i - 1]))), 0.0      # the short way round


def tx_position(record) -> np.ndarray:
    """World position of the projector at the transmit."""
    g = record.geometry
    return np.asarray(g.tx_pose.position) + np.asarray(core.rotation_ned(*g.tx_pose.rpy)) @ np.asarray(g.tx.lever_arm)


def direction(frame: Frame, sin_cone, sin_elevation) -> np.ndarray:
    """World unit vector(s) towards an arrival at a cone angle (from broadside, towards
    increasing channel number) and an elevation in the array's frame."""
    sb, se = np.broadcast_arrays(np.asarray(sin_cone, dtype=float), np.asarray(sin_elevation, dtype=float))
    out = np.sqrt(np.maximum(1.0 - sb**2 - se**2, 0.0))
    return sb[..., None] * frame.along + se[..., None] * frame.up + out[..., None] * frame.look


# --- the bound ------------------------------------------------------------------------------------

def rms_wavenumber(f_low: float, f_high: float, sound_speed: float) -> float:
    """Root-mean-square wavenumber of a flat band, rad/m: (2 pi / c) sqrt(fc^2 + B^2 / 12)."""
    fc, band = 0.5 * (f_low + f_high), abs(f_high - f_low)
    return 2.0 * np.pi / sound_speed * np.sqrt(fc**2 + band**2 / 12.0)


def sin_elevation_sigma(heights, element_snr: float, f_low: float, f_high: float, sound_speed: float = 1500.0) -> float:
    """Cramer-Rao bound on sin(elevation) for one arrival of unknown complex amplitude in
    white noise: 1 / (k_rms sqrt(2 rho sum_n (h_n - mean h)^2)), with rho the signal-to-noise
    power ratio at one hydrophone after pulse compression and h_n the heights of the elements."""
    h = np.asarray(heights, dtype=float)
    spread = float(((h - h.mean()) ** 2).sum())
    if spread <= 0.0 or element_snr <= 0.0:
        return float("inf")
    return 1.0 / (rms_wavenumber(f_low, f_high, sound_speed) * np.sqrt(2.0 * element_snr * spread))


def unambiguous_elevation(separation: float, frequency: float, sound_speed: float = 1500.0) -> float:
    """Half-width (rad) of the elevation sector in which two rows `separation` apart have no
    second lobe at `frequency`: asin(lambda / (2 d)); pi/2 when d <= lambda / 2."""
    x = sound_speed / frequency / (2.0 * separation)
    return float(np.pi / 2.0 if x >= 1.0 else np.arcsin(x))


def cone_bias(cone_bearing, elevation):
    """Horizontal bearing minus the cone angle a line array reports, rad, for an arrival at
    `elevation` from the array's plane: asin(sin(cone) / cos(elevation)) - cone."""
    cone_bearing, elevation = np.asarray(cone_bearing, dtype=float), np.asarray(elevation, dtype=float)
    return np.arcsin(np.clip(np.sin(cone_bearing) / np.cos(elevation), -1.0, 1.0)) - cone_bearing


# --- depth ----------------------------------------------------------------------------------------

def fold_depth(z: float, water_depth: float | None, surface_z: float = 0.0) -> tuple[float, str]:
    """A straight-line depth folded back into the water column, and the path that put it where
    it was: "direct" (inside), "surface" (the mirror image above the surface: one surface
    bounce), "bottom" (below the bottom: one bottom bounce) or "multiple". Without a water
    depth only the surface is known."""
    d = z - surface_z
    if not np.isfinite(d):
        return float("nan"), "unknown"
    if not water_depth or not np.isfinite(water_depth) or water_depth <= 0.0:
        return (surface_z + abs(d), "surface" if d < 0.0 else "direct")
    h = float(water_depth)
    m = d % (2.0 * h)
    folded = m if m <= h else 2.0 * h - m
    path = "direct" if 0.0 <= d <= h else "surface" if -h <= d < 0.0 else "bottom" if h < d <= 2.0 * h else "multiple"
    return surface_z + folded, path


def outside_water(z: float, sigma: float, water_depth: float | None, surface_z: float = 0.0, n_sigma: float = 2.0,
                  water_depth_sigma: float = 0.0) -> bool:
    """Whether a straight-line depth lies above the surface or below the bottom by more than
    `n_sigma` standard deviations (the bottom's own uncertainty included): an arrival that
    cannot have come straight from an object."""
    if not (np.isfinite(z) and np.isfinite(sigma)):
        return False
    d = z - surface_z
    if d < -n_sigma * sigma:
        return True
    return bool(water_depth and np.isfinite(water_depth) and water_depth > 0.0
                and d > water_depth + n_sigma * float(np.hypot(sigma, water_depth_sigma)))


def ray_depth(receiver_depth: float, elevation: float, slant_range: float, ssp_depth, ssp_speed,
              water_depth: float | None = None, step: float | None = None, method: str = "steps") -> float:
    """Depth at which an arrival seen at `elevation` (rad, up positive, level-referenced) at
    `receiver_depth` started, back along the ray for the one-way travel time slant_range / c
    (c: the sound speed at the receiver) through a sound-speed profile, linear between its
    knots. Snell's law in steps of `step` m along the ray (default: a two-thousandth of the
    range, between 0.5 and 10 m); the ray turns where it would have to enter water too fast
    for it, and is mirrored at the surface (depth 0) and, with `water_depth`, at the bottom.
    In constant sound speed this is the straight line, folded."""
    if method == "arcs":                # closed-form arcs: the chain's one ray calculation (raytrace.py, libs/live raytrace.hpp)
        from .raytrace import ray_depth as arcs
        return arcs(receiver_depth, elevation, slant_range, ssp_depth, ssp_speed, water_depth or 0.0)
    if method != "steps":
        raise ValueError(f"ray_method: {method!r} (steps or arcs)")
    zs, cs = np.asarray(ssp_depth, dtype=float), np.asarray(ssp_speed, dtype=float)
    if step is None:
        step = min(max(slant_range / 2000.0, 0.5), 10.0)
    speed = lambda z: float(np.interp(z, zs, cs))                       # noqa: E731
    c0 = speed(receiver_depth)
    snell = np.cos(elevation) / c0
    z, down = float(receiver_depth), elevation < 0.0
    remaining = slant_range / c0                                        # s
    while remaining > 0.0:
        c = speed(z)
        sin_t = np.sqrt(max(1.0 - (snell * c) ** 2, 0.0))
        ds = min(step, remaining * c)
        z_next = z + (1.0 if down else -1.0) * max(sin_t, 1e-4) * ds
        if z_next < 0.0:
            z_next, down = -z_next, True
        elif water_depth and z_next > water_depth:
            z_next, down = 2.0 * water_depth - z_next, False
        elif snell * speed(z_next) >= 1.0 and snell * c < 1.0:
            down = not down                                             # a turning point: back towards slower water
            z_next = z
        z = z_next
        remaining -= ds / c
    return z


# --- active: elevation at each detection ----------------------------------------------------------

@dataclass
class Elevation:
    valid: bool = False            # an elevation was found with enough quality
    sin_elevation: float = float("nan")   # in the array's frame, up positive
    sigma: float = float("nan")    # rad, one sigma of the elevation (Cramer-Rao at the measured SNR, plus roll and attitude timing)
    quality: float = 0.0           # coherent power over the rows' incoherent power, 0..1
    ambiguous: bool = False        # another lobe could be the true one, or the peak is at the sector's edge
    snr_db: float = float("nan")   # signal-to-noise ratio at one hydrophone, dB (what the bound uses)
    range: float = float("nan")    # slant range of the peak, m
    world: tuple = (float("nan"),) * 3   # unit vector towards the arrival, world (north, east, down)
    alternatives: tuple = ()       # unit vectors of the other candidate lobes, world
    attitude_gap: float = 0.0      # s between the echo's arrival and the attitude used for it


def row_records(record, rows):
    """One record per row (copies): each is an ordinary line array to every other stage."""
    from . import sync
    return [sync.select_channels(record, r) for r in rows]


def weight_power(n_elements: int, window: str) -> float:
    """Sum of the squared aperture weights of a row of `n_elements`, as the map applies them."""
    if window != "hann" or n_elements < 2:
        return float(n_elements)
    x = (np.arange(n_elements) - (n_elements - 1) / 2.0) / n_elements
    return float((np.cos(np.pi * x) ** 4).sum())


def estimate(record, rows, subs, hits, backgrounds, frame: Frame, cfg: dict, *, noise_weight: float,
             range_cell: float, backend=core.Backend.Host) -> list[Elevation]:
    """Elevation of each detection. `hits` are (slant range, cone bearing rad, range of the
    cluster's nearest cell or NaN); `backgrounds`
    the map's mean noise power at each; `noise_weight` the sum of squared aperture weights that
    made the map, so that background / noise_weight is the noise power at one hydrophone;
    `subs` the rows' records (conditioned)."""
    out = [Elevation() for _ in hits]
    if not hits:
        return out
    c = record.environment.sound_speed
    w = record.transmit.waveform
    f_low, f_high = min(w.f_start, w.f_stop), max(w.f_start, w.f_stop)
    band = (f_high - f_low) or 1.0 / w.duration
    first, last, step = cfg["sector_deg"]
    grid = np.radians(np.arange(first, last + 0.5 * step, step))
    se = np.sin(grid)
    half = cfg["range_window_m"] if cfg.get("range_window_m") else 1.5 * range_cell
    dr = 0.25 * c / band                                    # half a range resolution cell
    n_hi = int(np.ceil(half / dr))
    counts = np.array([len(r) for r in rows], dtype=float)
    n_total = counts.sum()
    coherent_gain = n_total**2 / (counts**2).sum()          # |sum of rows|^2 over sum of |row|^2, for one plane wave
    spread = float(((frame.heights - frame.heights.mean()) ** 2).sum())
    k_rms = rms_wavenumber(f_low, f_high, c)
    look = "starboard" if float(np.cross(frame.along, frame.up) @ frame.look) > 0.0 else "port"
    pose = record.geometry.rx_pose
    turning = bool(np.any(np.asarray(pose.rates) != 0.0))
    arrival = cfg.get("attitude_time", "transmit") == "arrival"
    prior = cfg.get("sector_from_depth")
    water = cfg.get("water_depth") or record.environment.water_depth
    water = float(water) if water and np.isfinite(water) and water > 0.0 else None
    for i, (rng, bearing, first) in enumerate(hits):
        el = out[i]
        # The array as the record's pose has it when this echo arrives (the kernel turns it with the
        # pose's rates), and as the attitude history says it then was: the angles are found in the
        # first frame and turned into a world direction with the second.
        t_arr = 2.0 * rng / c
        kf = array_frame(record, rows, look, np.asarray(pose.rpy) + np.asarray(pose.rates) * t_arr) if turning else frame
        wf, extra = kf, 0.0
        if arrival:
            # The compressed echo is the whole pulse, received from t_arr to t_arr + its length: its middle.
            t_att = t_arr + 0.5 * w.duration
            rpy, gap = attitude_at(record, t_att)
            if rpy is None:
                el.attitude_gap = float(t_att)                           # nothing but the pose at the transmit
            else:
                wf, el.attitude_gap = array_frame(record, rows, look, rpy), float(gap)
            extra = np.radians(cfg.get("attitude_rate_sigma_deg_s", 0.0)) * el.attitude_gap
        tx = tx_position(record) - kf.origin
        # The range searched: about the detection, and back to the leading edge of its cluster
        # (the direct arrival is the first), but no further than eight half-windows.
        lead = half
        if np.isfinite(first):
            lead = min(max(rng - first + range_cell, half), 8.0 * half)
        offsets = dr * np.arange(-int(np.ceil(lead / dr)), n_hi + 1)
        sb = float(np.sin(bearing))
        u = direction(kf, sb, se)                                           # [E, 3]
        # Points of constant two-way path (projector -> point -> array centre) along elevation: the
        # projector is not at the array's centre, and at constant range from the array the path
        # would change with elevation by far more than the rows' own differences.
        path = 2.0 * (rng + offsets)                                        # [R]
        rho = (path[None, :] ** 2 - tx @ tx) / (2.0 * (path[None, :] - (u @ tx)[:, None]))   # [E, R]
        pts = kf.origin + rho[:, :, None] * u[:, None, :]                   # [E, R, 3]
        z = np.array([core.backproject([s], pts.reshape(-1, 3), interp="sinc", backend=backend).reshape(pts.shape[:2])
                      for s in subs])                                       # [K, E, R]
        p = np.abs(z.sum(axis=0)).astype(float) ** 2
        if prior and water is not None:
            # Straight rays with so many bounces cannot arrive from steeper than this: do not look there.
            own = float(wf.origin[2] - cfg.get("surface_z", 0.0))
            n_b, margin = float(prior.get("bounces", 1)), np.radians(prior.get("margin_deg", 1.0))
            level = np.arcsin(np.clip(-direction(wf, sb, se)[:, 2], -1.0, 1.0))
            allowed = (level <= np.arctan((own + n_b * water) / rng) + margin) & (level >= -np.arctan((water - own + n_b * water) / rng) - margin)
            if allowed.sum() >= 3:
                p[~allowed] = 0.0
        noise = backgrounds[i] / noise_weight                                # at one hydrophone
        best = p.max(axis=0)                                                 # over elevation, at each range
        j = int(np.argmax(best))
        if cfg["arrival"] == "first":
            # Bounce paths are longer than the direct one: take the earliest arrival that stands
            # clear of the noise and is not far below the strongest.
            floor = max(best[j] * 10.0 ** (-cfg["first_arrival_db"] / 10.0),
                        n_total * noise * 10.0 ** (cfg["first_arrival_snr_db"] / 10.0))
            for jj in range(1, j):
                if best[jj] >= floor and best[jj] >= best[jj - 1] and best[jj] > best[jj + 1]:
                    j = jj
                    break
        e = int(np.argmax(p[:, j]))
        col = p[:, j]
        el.range = float(rho[e, j])

        def refined(k):
            """Sub-grid peak: a parabola through the three points about a maximum."""
            if k == 0 or k == len(grid) - 1:
                return grid[k]
            den = col[k - 1] - 2.0 * col[k] + col[k + 1]
            return grid[k] + (float(0.5 * (col[k - 1] - col[k + 1]) / den) if den < 0.0 else 0.0) * np.radians(step)

        edge = e == 0 or e == len(grid) - 1 or col[max(e - 1, 0)] == 0.0 or col[min(e + 1, len(grid) - 1)] == 0.0
        angle = refined(e)
        # The other lobes: maxima outside the main lobe (which runs down to its first minima).
        lo = e
        while lo > 0 and col[lo - 1] < col[lo]:
            lo -= 1
        hi = e
        while hi < len(col) - 1 and col[hi + 1] < col[hi]:
            hi += 1
        rest = np.concatenate([col[:lo], col[hi + 1:]])
        second = float(rest.max()) if len(rest) else 0.0
        # Would noise swap the first and the second? Their difference in amplitude is (1 - g) of the
        # peak, g the second's amplitude over the first's, and its noise sqrt(1 - g) of the sum's.
        g = np.sqrt(second / col[e]) if col[e] > 0.0 else 1.0
        margin_sigma = np.sqrt(col[e] / (n_total * noise) * max(1.0 - g, 0.0)) if noise > 0.0 else float("inf")
        el.ambiguous = bool(edge or second > col[e] * 10.0 ** (-cfg["ambiguity_db"] / 10.0) or margin_sigma < cfg.get("ambiguity_sigma", 3.0))
        others = []
        if el.ambiguous:
            floor_alt = col[e] * 10.0 ** (-cfg.get("alternatives_db", 6.0) / 10.0)
            for k in range(1, len(col) - 1):
                if (k < lo or k > hi) and col[k] >= floor_alt and col[k] >= col[k - 1] and col[k] > col[k + 1]:
                    others.append(tuple(float(v) for v in direction(wf, sb, float(np.sin(refined(k))))))
        el.alternatives = tuple(others)
        j0, j1 = max(j - 1, 0), min(j + 2, p.shape[1])
        incoherent = float((np.abs(z[:, e, j0:j1]) ** 2).sum())
        el.quality = float(p[e, j0:j1].sum() / (coherent_gain * incoherent)) if incoherent > 0.0 else 0.0
        snr = float(col[e]) / (n_total**2 * noise) if noise > 0.0 else float("inf")
        el.snr_db = float(10.0 * np.log10(max(snr, 1e-30)))
        el.sin_elevation = float(np.sin(angle))
        sigma_sin = 1.0 / (k_rms * np.sqrt(2.0 * snr * spread)) if spread > 0.0 and snr > 0.0 else float("inf")
        el.sigma = float(np.sqrt((sigma_sin / max(np.cos(angle), 1e-6)) ** 2 + np.radians(cfg["roll_sigma_deg"]) ** 2 + extra**2))
        el.world = tuple(float(v) for v in direction(wf, sb, el.sin_elevation))
        el.valid = bool(el.quality >= cfg["min_quality"] and np.isfinite(el.sigma))
    return out


def refraction_sigma(slant_range: float, sound_speed: float, gradient_sigma: float) -> float:
    """Depth uncertainty (m) from not knowing the mean vertical sound-speed gradient along the
    path to within `gradient_sigma` (1/s): a gradient g curves the ray with radius c / g, which
    after `slant_range` is range^2 g / 2c away from the straight line."""
    return float(slant_range**2 * abs(gradient_sigma) / (2.0 * sound_speed))


def gradient_profile(own_depth: float, sound_speed: float, gradient: float, water_depth: float | None):
    """The two-knot profile c(z) = sound_speed + gradient (z - own_depth), from the surface to
    the bottom (or 10 km): (depths, speeds) for `ray_depth`."""
    bottom = max(float(water_depth or 0.0), own_depth, 0.0) + 10000.0
    return [0.0, bottom], [sound_speed - gradient * own_depth, sound_speed + gradient * (bottom - own_depth)]


def profile(record, subs, frame: Frame, bearing: float, ranges, sector_deg=(-30.0, 30.0, 0.5), backend=core.Backend.Host):
    """For analysis: the arrivals along range at one cone bearing. At each of `ranges` (m, slant)
    the elevation of the peak (rad, array frame), its coherent power and its quality.
    Bounce arrivals of one target show up as later peaks at other elevations."""
    grid = np.radians(np.arange(sector_deg[0], sector_deg[1] + 0.5 * sector_deg[2], sector_deg[2]))
    sb = float(np.sin(bearing))
    u = direction(frame, sb, np.sin(grid))
    tx = tx_position(record) - frame.origin
    path = 2.0 * np.asarray(ranges, dtype=float)
    rho = (path[None, :] ** 2 - tx @ tx) / (2.0 * (path[None, :] - (u @ tx)[:, None]))
    pts = frame.origin + rho[:, :, None] * u[:, None, :]
    z = np.array([core.backproject([s], pts.reshape(-1, 3), interp="sinc", backend=backend).reshape(pts.shape[:2]) for s in subs])
    p = np.abs(z.sum(axis=0)) ** 2
    e = np.argmax(p, axis=0)
    cols = np.arange(p.shape[1])
    counts = np.array([s.samples.n_channels for s in subs], dtype=float)
    incoherent = (np.abs(z[:, e, cols]) ** 2).sum(axis=0)
    quality = p[e, cols] / np.maximum(counts.sum() ** 2 / (counts**2).sum() * incoherent, 1e-300)
    return grid[e], p[e, cols], quality


# --- passive: elevation of each bearing -----------------------------------------------------------

def passive_elevation(record, rows, bearing: float, f_low: float, f_high: float, *, n_fft: int = 1024, hop: int = 512,
                      first: int = 0, n_blocks: int = -1, sector_deg=(-30.0, 30.0, 0.5),
                      look: str = "starboard") -> tuple[float, float]:
    """Elevation (rad, array frame, up positive) of a noise source at cone `bearing`, and its
    quality (0..1), from raw or conditioned data: every row is steered at the bearing in each
    frequency bin of each block (n_fft samples stepped by hop from sample `first`), and the rows'
    beams are then steered over elevation and their power added over bins and blocks. NumPy
    reference of libs/live `passive_elevation` (core.live_passive_elevation)."""
    s = record.samples
    x = np.asarray(s.array)
    fs = s.sample_rate
    carrier = s.center_frequency if s.is_complex() else 0.0
    frame = array_frame(record, rows, look)
    c = record.environment.sound_speed
    pos = record.rx_positions() - frame.origin
    total = (x.shape[1] - first - n_fft) // hop + 1
    n_blocks = total if n_blocks < 0 else min(n_blocks, total)
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(n_fft) + 0.5) / n_fft)
    k = np.arange(n_fft)
    freqs = carrier + np.where(k <= n_fft // 2, k, k - n_fft) * fs / n_fft if s.is_complex() else np.fft.rfftfreq(n_fft, 1.0 / fs)
    bins = np.nonzero((freqs >= f_low) & (freqs <= f_high) & (freqs > 0.0))[0]
    f = freqs[bins]
    grid = np.radians(np.arange(sector_deg[0], sector_deg[1] + 0.5 * sector_deg[2], sector_deg[2]))
    sb = float(np.sin(bearing))
    u0 = direction(frame, sb, 0.0)
    u = direction(frame, sb, np.sin(grid))                                   # [E, 3]
    centres = np.array([pos[r].mean(axis=0) for r in rows])                  # [K, 3]
    beams = np.zeros((len(rows), n_blocks, len(bins)), dtype=complex)        # each row steered at the bearing, level
    for k, r in enumerate(rows):
        lead = (pos[r] - centres[k]) @ u0 / c                                # [n]
        steer = np.exp(-2j * np.pi * f[None, :] * lead[:, None])             # [n, F]
        for b in range(n_blocks):
            seg = x[r, first + b * hop:first + b * hop + n_fft] * win
            spec = (np.fft.fft(seg, axis=1) if s.is_complex() else np.fft.rfft(seg, axis=1))[:, bins]
            beams[k, b] = (spec * steer).sum(axis=0)
    lead_rows = centres @ u.T / c                                            # [K, E]
    power = np.zeros(len(grid))
    for e in range(len(grid)):
        steer = np.exp(-2j * np.pi * f[None, :] * lead_rows[:, e, None])     # [K, F]
        power[e] = (np.abs((beams * steer[:, None, :]).sum(axis=0)) ** 2).sum()
    e = int(np.argmax(power))
    frac = 0.0
    if 0 < e < len(grid) - 1:
        den = power[e - 1] - 2.0 * power[e] + power[e + 1]
        frac = float(0.5 * (power[e - 1] - power[e + 1]) / den) if den < 0.0 else 0.0
    counts = np.array([len(r) for r in rows], dtype=float)
    incoherent = float((np.abs(beams) ** 2).sum())
    quality = float(power[e] / (counts.sum() ** 2 / (counts**2).sum() * incoherent)) if incoherent > 0.0 else 0.0
    return float(grid[e] + frac * np.radians(sector_deg[2])), quality
