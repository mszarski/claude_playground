"""Platform motion estimation for SAS (ASASIN Sec. II-D).

Sway and heave come from time delays between overlapping phase centres of consecutive
pings (displaced phase centre micronavigation), solved jointly over all pings by the
Ceres problem in libs/motion (Eq. 8). Surge comes from the DVL and attitude from the INS,
both read from the records. Delay estimation follows the paper's two-step scheme:

1. coarse delay from the quadratic-interpolated peak of the correlation magnitude;
2. fine delay from the phase of the complex correlation, ambiguous by whole carrier
   periods;
3. solve with coarse delays, unwrap the fine delays around that solution, solve again.

The delay estimator is NumPy for now; the solver is C++ (Ceres).

Units and conventions: metres, seconds, world frame north-east-down (z positive down).
Delays are two-way, so a range extent of d metres is 2 d fs / c samples. `seafloor_z` is a
z in the world frame; 0.0 is the seabed only for a scene whose origin is on the seabed.

Window sizes can be given in samples (`window_samples`, `max_lag_samples`: the defaults,
tuned for one sample rate and bandwidth) or in metres of range (`window_m`, `max_lag_m`),
which are turned into samples from the record and so hold for another sonar
(docs/HARDCODED_AUDIT.md R-3). `max_lag` bounds the change of two-way range between pings
that can be measured: 12 samples is 9 mm at a prepared rate of 1 MHz and 0.9 m at 10 kHz.
A window whose correlation peaks at the edge of the lag span is dropped; the result's log
counts those, because many of them mean the span is too short, not the echoes incoherent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _gpusonar as core
from .recon.fls import prepare


@dataclass
class MotionOptions:
    n_windows: int = 8               # range windows per ping pair (k in Eq. 5)
    window_samples: int = 256        # correlation window, samples of the prepared record
    max_lag_samples: int = 12        # largest delay searched, samples either way
    window_m: float | None = None    # the window as metres of range; replaces window_samples
    max_lag_m: float | None = None   # the largest change of range between pings, m; replaces max_lag_samples
    min_coherence: float = 0.5       # measurements below this are dropped
    lambda_smooth: float = 1.0       # lambda1 in Eq. 8
    lambda_dvl: float = 1.0          # lambda2 in Eq. 8
    huber: bool = True               # Eq. 3; False = the squared loss of the Eq. 5 caption
    weight_by_coherence: bool = True
    residual_scale_periods: float = 1.0 / 16.0   # residual scale as a fraction of a carrier period
    smooth_scale: float = 2e-3       # m/s
    dvl_scale: float = 5e-3          # m
    seafloor_z: float | str = 0.0    # flat seafloor (the paper's default: origin on the seabed), or
                                     # "auto": from the records (DVL altitude, else water depth)
    bathymetry: object = None        # optional callable z(x, y) for a known seafloor


@dataclass
class MotionResult:
    y: np.ndarray                    # sway position per ping, m
    z: np.ndarray                    # heave position per ping, m
    vy: np.ndarray
    vz: np.ndarray
    coarse: object                   # solver output of the first pass
    fine: object                     # solver output of the second pass
    n_measurements: int
    n_rejected: int
    unwrap_corrections: int          # fine delays moved by a whole period in pass two
    log: list = field(default_factory=list)
    windows: dict = field(default_factory=dict)   # what was used: window and lag in samples and metres,
                                                  # seafloor_z, and how many windows peaked at the lag edge


def phase_centre_shift(record_a, record_b) -> int:
    """Channel index shift s such that channel k + s of the first ping and channel k of
    the second share a phase centre, from the along-track advance between the pings."""
    pa, pb = record_a.rx_positions(), record_b.rx_positions()
    axis = pa[-1] - pa[0]
    n = pa.shape[0]
    pitch = np.linalg.norm(axis) / (n - 1)
    axis = axis / np.linalg.norm(axis)
    advance = (pb.mean(axis=0) - pa.mean(axis=0)) @ axis
    # Phase centres sit at half the element spacing, so an advance moves them by 2 * advance / pitch.
    return int(round(2.0 * advance / pitch))


def correlate_window(s1: np.ndarray, s2: np.ndarray, centre: int, half: int, max_lag: int,
                     fs: float, fc: float):
    """Delay of s2 relative to s1 around sample `centre`.

    Returns (coarse delay, fine delay within half a period of zero, coherence). The fine
    delay is ambiguous by multiples of 1 / fc.
    """
    a = s1[centre - half:centre + half]
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.vdot(a, s2[centre - half + l:centre + half + l]) for l in lags])
    mag = np.abs(corr)
    i = int(np.argmax(mag))
    if i == 0 or i == len(lags) - 1:
        return None
    y0, y1, y2 = mag[i - 1], mag[i], mag[i + 1]
    frac = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
    coarse = (lags[i] + frac) / fs
    # Baseband echoes carry exp(-j 2 pi fc tau), so the correlation phase is -2 pi fc delta.
    fine = -np.angle(corr[i]) / (2.0 * np.pi * fc)
    b = s2[centre - half + lags[i]:centre + half + lags[i]]
    coherence = mag[i] / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real + 1e-300)
    return coarse, fine, float(coherence)


def _ping_nav(records) -> list:
    t0 = records[0].transmit.time_ns
    out = []
    for r in records:
        pose = r.geometry.rx_pose
        nav = core.PingNav()
        nav.time = (r.transmit.time_ns - t0) * 1e-9
        nav.x, nav.y, nav.z = pose.position
        dvl = r.navigation.dvl
        nav.surge = dvl[0].velocity_body[0] if dvl else pose.velocity[0]
        nav.rpy = pose.rpy
        nav.dvl_altitude = dvl[0].altitude if dvl else float("nan")
        out.append(nav)
    return out


def window_sizes(record, options: MotionOptions) -> tuple[int, int]:
    """(window, max lag) in samples of this (prepared) record. Metres of range, when given,
    become samples through the two-way delay: n = 2 d fs / c."""
    per_m = 2.0 * record.samples.sample_rate / record.environment.sound_speed
    window, lag = int(options.window_samples), int(options.max_lag_samples)
    if options.window_m is not None:
        window = max(2 * int(np.ceil(0.5 * float(options.window_m) * per_m)), 16)
    if options.max_lag_m is not None:
        lag = max(int(np.ceil(float(options.max_lag_m) * per_m)), 2)
    if window < 4 or lag < 1:
        raise ValueError("motion: the correlation window needs at least 4 samples and the lag span at least 1")
    if window + 2 * lag >= record.samples.n_samples:
        raise ValueError(f"motion: a window of {window} samples with {lag} of lag does not fit the record's "
                         f"{record.samples.n_samples} samples")
    return window, lag


def resolve_seafloor_z(records, options: MotionOptions) -> float:
    """`options.seafloor_z` as a number: itself, or "auto" from the records."""
    if isinstance(options.seafloor_z, str):
        if options.seafloor_z != "auto":
            raise ValueError("motion: seafloor_z must be a z in metres or 'auto'")
        from .recon.sas import seafloor_z
        return seafloor_z(records)[0]
    return float(options.seafloor_z)


def measure_delays(records, options: MotionOptions, stats: dict | None = None):
    """Delay measurements between every consecutive ping pair. Returns a list of dicts with
    the solver's DelayMeasurement (coarse delay filled in), the fine delay and coherence.
    `stats`, when given, receives the window and lag used (samples and metres) and the
    number of windows dropped because the correlation peaked at the edge of the lag span."""
    out, rejected, at_edge = [], 0, 0
    floor_z = resolve_seafloor_z(records, options)
    window, max_lag = window_sizes(records[0], options) if len(records) else (0, 0)
    for i in range(len(records) - 1):
        ra, rb = records[i], records[i + 1]
        shift = phase_centre_shift(ra, rb)
        n = ra.samples.n_channels
        if shift <= 0 or shift >= n:
            continue                                   # no overlapping phase centres
        xa, xb = ra.samples.array, rb.samples.array
        fs, fc, t0 = ra.samples.sample_rate, ra.samples.center_frequency, ra.samples.start_time
        c = ra.environment.sound_speed
        half = window // 2
        lo = half + max_lag
        hi = ra.samples.n_samples - half - max_lag
        centres = np.linspace(lo, hi, options.n_windows).astype(int)
        pose = ra.geometry.rx_pose
        rot = core.rotation_ned(*pose.rpy)
        look = rot @ np.asarray(ra.geometry.rx[0].boresight)
        look_h = np.array([look[0], look[1], 0.0])
        look_h /= np.linalg.norm(look_h)
        tx_lever = np.asarray(ra.geometry.tx.lever_arm)
        for k in range(n - shift):
            lever_a = np.asarray(ra.geometry.rx[k + shift].lever_arm)
            lever_b = np.asarray(rb.geometry.rx[k].lever_arm)
            pc = np.asarray(pose.position) + rot @ (0.5 * (tx_lever + lever_a))
            for centre in centres:
                est = correlate_window(xa[k + shift], xb[k], int(centre), half,
                                       max_lag, fs, fc)
                if est is None:
                    rejected += 1
                    at_edge += 1
                    continue
                coarse, fine, coh = est
                if coh < options.min_coherence:
                    rejected += 1
                    continue
                t_ref = t0 + centre / fs
                slant = 0.5 * c * t_ref
                # Footprint: on the seafloor, at the ground range this slant range reaches.
                z_floor = floor_z
                ground = np.sqrt(max(slant**2 - (z_floor - pc[2]) ** 2, 0.0))
                ps = np.array([pc[0], pc[1], z_floor]) + ground * look_h
                if options.bathymetry is not None:
                    for _ in range(3):                 # settle the footprint on the known seafloor
                        z_floor = float(options.bathymetry(ps[0], ps[1]))
                        ground = np.sqrt(max(slant**2 - (z_floor - pc[2]) ** 2, 0.0))
                        ps = np.array([pc[0], pc[1], z_floor]) + ground * look_h
                m = core.DelayMeasurement()
                m.ping = i
                m.tx_lever_1, m.rx_lever_1 = tuple(tx_lever), tuple(lever_a)
                m.tx_lever_2, m.rx_lever_2 = tuple(np.asarray(rb.geometry.tx.lever_arm)), tuple(lever_b)
                m.t_ref = float(t_ref)
                m.footprint = tuple(ps)
                m.delay = float(coarse)
                m.weight = coh if options.weight_by_coherence else 1.0
                out.append({"measurement": m, "coarse": coarse, "fine": fine, "coherence": coh, "fc": fc})
    if stats is not None and len(records):
        per_m = 2.0 * records[0].samples.sample_rate / records[0].environment.sound_speed
        stats.update(window_samples=window, max_lag_samples=max_lag, window_m=window / per_m,
                     max_lag_m=max_lag / per_m, seafloor_z=floor_z, rejected_at_lag_edge=at_edge)
    return out, rejected


def estimate_motion(records, options: MotionOptions | None = None) -> MotionResult:
    """Estimate sway and heave for a straight-line collection (the two-pass scheme).

    Records must hold navigation's poses and a DVL sample each. They are prepared
    (baseband, pulse compressed) in place if they are not already.
    """
    if not core.HAVE_MOTION:
        raise RuntimeError("gpusonar was built without motion estimation (Ceres not found)")
    options = options or MotionOptions()
    records = list(records)
    prepare(records)
    pings = _ping_nav(records)
    # Origin on the seafloor below the first ping: its height is its DVL altitude.
    floor_z = resolve_seafloor_z(records, options)
    if np.isfinite(pings[0].dvl_altitude):
        pings[0].z = floor_z - pings[0].dvl_altitude
    used: dict = {}
    found, rejected = measure_delays(records, options, used)
    if not found:
        hint = ""
        if used.get("rejected_at_lag_edge"):
            hint = (f" {used['rejected_at_lag_edge']} windows peaked at the edge of the lag span "
                    f"({used['max_lag_samples']} samples, {used['max_lag_m'] * 1e3:.1f} mm of range): raise max_lag_m")
        raise RuntimeError("no usable delay measurements: do consecutive pings share phase centres?" + hint)
    fc = found[0]["fc"]
    so = core.MotionSolveOptions()
    so.sound_speed = records[0].environment.sound_speed
    so.lambda_smooth, so.lambda_dvl = options.lambda_smooth, options.lambda_dvl
    so.huber = options.huber
    so.residual_scale = options.residual_scale_periods / fc
    so.smooth_scale, so.dvl_scale = options.smooth_scale, options.dvl_scale
    so.use_weights = options.weight_by_coherence

    # Pass 1: coarse delays only.
    measurements = [f["measurement"] for f in found]
    coarse = core.solve_motion(pings, measurements, so)
    # Unwrap each fine delay to the period nearest the delay the coarse solution predicts.
    corrections = 0
    for f in found:
        m = f["measurement"]
        predicted = core.predict_delay(pings, m, coarse.y, coarse.z, so.sound_speed)
        n_pred = np.round((predicted - f["fine"]) * fc)
        n_coarse = np.round((f["coarse"] - f["fine"]) * fc)
        corrections += int(n_pred != n_coarse)
        m.delay = float(f["fine"] + n_pred / fc)
    # Pass 2: unwrapped fine delays, starting from the coarse trajectory.
    for nav, y, z in zip(pings, coarse.y, coarse.z):
        nav.y, nav.z = y, z
    fine = core.solve_motion(pings, measurements, so)
    n = len(records)
    log = [f"pass 1 (coarse): {coarse.report}", f"pass 2 (fine): {fine.report}",
           f"windows: {used['window_samples']} samples ({used['window_m']:.3f} m), lag +/-{used['max_lag_samples']} "
           f"samples ({used['max_lag_m'] * 1e3:.1f} mm of range), seafloor z {used['seafloor_z']:.3f} m"]
    if used["rejected_at_lag_edge"]:
        log.append(f"{used['rejected_at_lag_edge']} of {len(found) + rejected} windows peaked at the edge of the lag "
                   "span and were dropped: if this is most of the rejections, the motion between pings exceeds "
                   "max_lag")
    return MotionResult(np.array(fine.y[:n]), np.array(fine.z[:n]), np.array(fine.vy), np.array(fine.vz),
                        coarse, fine, len(found), rejected, corrections, log, used)


def apply_motion(records, result: MotionResult) -> None:
    """Write the estimated sway and heave (positions and velocities) into the records' poses."""
    for r, y, z, vy, vz in zip(records, result.y, result.z, result.vy, result.vz):
        geo = r.geometry
        for name in ("rx_pose", "tx_pose"):
            pose = getattr(geo, name)
            p, v = pose.position, pose.velocity
            pose.position = (p[0], float(y), float(z))
            pose.velocity = (v[0], float(vy), float(vz))
            setattr(geo, name, pose)
        r.geometry = geo
