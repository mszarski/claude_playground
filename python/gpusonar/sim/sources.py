"""Radiating sources: sound that reaches the array one way.

The ping simulator makes echoes: projector -> scatterer -> receiver. A source here radiates a
signal of its own, which travels to the receiver once: SST's `Source` + `DirectSignal`
(APL-UW TR 0702 Secs. 7-8). Scenario key `sources:`, a list of

    - label: tanker
      origin: [x, y, z]              # NED, m; with velocity, heading_deg, turn_rate_deg as for a
      velocity: [vx, vy, vz]         # scene target (sim/targets.py HighlightTarget), or `position:`
      signal: {type: noise | tonals | waveform | file, ...}       # or a list of such mappings (summed)

Signals (levels re 1 uPa at 1 m; they become sample units through the sonar's
`receive.sensitivity_db + gain_db`, as ambient noise does):

  noise      Gaussian noise with a spectrum: `level_db` (dB re 1 uPa^2/Hz at 1 m, at
             `reference_frequency`, default 1 kHz), `slope_db_per_octave` (default 0), optional
             `band: [f_lo, f_hi]`; or a table `spectrum: {frequency: [...], level_db: [...]}`.
  tonals     lines: `lines: [{frequency, level_db, phase_deg}]` (level_db the rms source
             level; phase drawn once if not given), and/or a harmonic family as TR 0702 eq. 29,
             x(t) = sum A_n cos(2 pi n f1 t + phi_n): `family: {fundamental, harmonics:
             [[n, 20 log10 A_n, phase_deg], ...]}` (A_n the peak amplitude in uPa at 1 m, as SST's table).
  waveform   a pulse emitted at given times: `waveform: {...}` (any transmit waveform of the
             simulator, `type: sampled` included; default: the sonar's own), `source_level_db`
             (default: the waveform's, else the sonar's), and when: `times: [...]` (scenario
             seconds), `repeat: {first, interval, count}`, or `ping_offset: s` (once per ping,
             that long after the sonar's transmit instant; the default, 0: a separate
             transmitter firing with the sonar, SST's bistatic direct signal).
  file       samples from a .wav/.npy/.h5 file (`path`, `dataset`, `sample_rate`): `level_db`
             (the rms source level the file's rms is scaled to) or `scale`, `start` (scenario
             time of the first sample), `loop`.

Propagation is one way through the scenario's eigenray provider (free field, image source,
sediment, Bellhop, and the time-spread wrapper with its taps, angle and Doppler spreading):
eigenrays from the array's reference point to the source, travelled backwards (reciprocity),
exactly as the return leg of an echo. Each ray is an apparent source position with a
velocity; the delay to every element, the 1/r spreading, the receive beam pattern, absorption
and the board clocks follow from it. Doppler is the exact time scaling: the emission time u of
the sound received at t solves |P(u) - R(t)| = c (t - u), whose derivative is TR 0702 eq. 63.

What runs where:

  waveform   the C++ synthesis kernel (host or CUDA), unchanged. The kernel's path is
             projector -> q -> p -> element; here the projector is made omnidirectional and
             q rides 1 m from it (q = tx + r0 (e - v/c), moving with it), so the first leg is a
             constant 1 m whatever the motion, and p is the apparent source. The 1 m of
             delay is taken out through the record's start time (with the emission time), its
             spreading is 1, and its absorption is put back at the centre frequency. Exact:
             tested against the closed forms and against the streaming synthesiser.
             A sampled waveform the kernel cannot take falls back to NumPy as echoes do.
  noise, tonals, file
             a streaming synthesiser in NumPy (this file; no C++ or CUDA version): the
             retarded time in closed form on a coarse time grid per (ray, channel), linear in
             between; tonals evaluated analytically at their emission time; sampled signals
             read with the windowed-sinc interpolator of sim/waveform.py. Both `engine`
             settings use it (the eigenrays under it come from the selected engine).

The sonar's own projector heard directly (`transmit.direct_path`, `direct_arrival` at the end
of this file) is such a pulse from the projector's position; see docs/PHYSICS_NOTES.md Sec. 22.

Limits: positions and velocities are first order over a record (straight lines), as for
echoes; eigenrays are found once per ping for the source's position at the transmit
instant; the receive beam pattern is taken at the record's centre frequency for sampled
signals (at its own frequency for a tone); absorption of a sampled signal uses each ray's
mean path length to the array.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .. import _gpusonar as core
from .. import acoustics
from .. import environment as site
from . import waveform as wf_mod
from .targets import HighlightTarget

SIGNAL_TYPES = ("noise", "tonals", "waveform", "file")
KERNEL_LEG = 1.0                # m: the constant first leg of the kernel's path for a one-way pulse
# The streaming synthesiser solves the retarded time exactly every KNOT_INTERVAL seconds of
# the record and interpolates the delay linearly between those knots. With straight-line
# motion the delay's curvature is v_t^2 / (r c) (v_t the speed across the line of sight, r
# the range), so the interpolation is off by at most  h^2 v_t^2 / (8 r c)  seconds: at
# h = 4 ms that is 0.008 rad for 10 m/s at 10 m and 100 kHz, and 0.17 rad for 20 m/s at
# 2 m. Both numbers are defaults of a source's `stream:` block ({knot_interval, oversample});
# `knot_interval: auto` halves the spacing until the measured error is under
# KNOT_PHASE_TOLERANCE, and a fixed spacing that leaves more than KNOT_PHASE_LIMIT warns
# (docs/HARDCODED_AUDIT.md M-9).
KNOT_INTERVAL = 4e-3            # s: spacing of the exact retarded-time solutions in the streaming synthesiser
STREAM_OVERSAMPLE = 2.0         # emission-time series are held at this multiple of the record's rate
KNOT_PHASE_TOLERANCE = 0.01     # rad: what `knot_interval: auto` refines to
KNOT_PHASE_LIMIT = 0.1          # rad: a fixed spacing that leaves more than this warns


@dataclass
class Source:
    label: str
    track: HighlightTarget
    signals: list
    state: dict = field(default_factory=dict)
    stream: dict = field(default_factory=dict)   # {knot_interval: seconds or "auto", oversample}: see KNOT_INTERVAL

    def at(self, t: float):
        """(position [3], velocity [3]) at scenario time t."""
        origin, _, velocity, _ = self.track.pose(float(t))
        return np.asarray(origin, dtype=float), np.asarray(velocity, dtype=float)


def build_sources(cfg, rng: np.random.Generator | None = None) -> list:
    """`Source`s from a scenario's `sources:` list. Random phases of tonals are drawn here,
    once, so that a tone is one continuous signal over all the pings of a run."""
    rng = rng or np.random.default_rng(0)
    out = []
    for i, item in enumerate(cfg or []):
        if isinstance(item, Source):
            out.append(item)
            continue
        if "signal" not in item:
            raise ValueError("a source needs a `signal`")
        origin = item.get("origin", item.get("position", (0.0, 0.0, 0.0)))
        track = HighlightTarget(origin, item.get("velocity", (0.0, 0.0, 0.0)), float(item.get("heading_deg", 0.0)),
                                turn_rate_deg=float(item.get("turn_rate_deg", 0.0)), trajectory=item.get("trajectory"))
        signals = item["signal"] if isinstance(item["signal"], (list, tuple)) else [item["signal"]]
        prepared = []
        for sig in signals:
            sig = dict(sig)
            kind = str(sig.get("type", "")).lower()
            if kind not in SIGNAL_TYPES:
                raise ValueError(f"unknown source signal type {sig.get('type')!r}: one of {SIGNAL_TYPES}")
            sig["type"] = kind
            if kind == "tonals":
                sig["_lines"] = _tonal_lines(sig, rng)
            prepared.append(sig)
        stream = dict(item.get("stream") or {})
        unknown = set(stream) - {"knot_interval", "oversample"}
        if unknown:
            raise ValueError(f"source `stream`: unknown keys {sorted(unknown)} (knot_interval, oversample)")
        if "oversample" in stream and not float(stream["oversample"]) >= 1.0:
            raise ValueError("source `stream.oversample` must be at least 1")
        out.append(Source(str(item.get("label", f"source_{i}")), track, prepared, stream=stream))
    return out


def knot_phase_error(delay_knots: np.ndarray, frequency: float) -> float:
    """Largest phase error (rad) of reading delays [..., J] linearly between their knots, at
    `frequency`: a parabola through three knots departs from the chord by an eighth of the
    second difference."""
    if delay_knots.shape[-1] < 3:
        return 0.0
    return float(2.0 * np.pi * frequency * np.max(np.abs(np.diff(delay_knots, 2, axis=-1))) / 8.0)


def _tonal_lines(sig: dict, rng) -> list:
    """[(frequency Hz, peak amplitude in uPa at 1 m, phase rad)] of a tonal signal."""
    lines = []
    for line in sig.get("lines", []):
        phase = np.radians(float(line["phase_deg"])) if line.get("phase_deg") is not None else float(rng.uniform(0.0, 2.0 * np.pi))
        lines.append((float(line["frequency"]), np.sqrt(2.0) * 10.0 ** (float(line["level_db"]) / 20.0), phase))
    family = sig.get("family")
    if family:
        f1 = float(family["fundamental"])
        for row in family["harmonics"]:
            n, level = float(row[0]), float(row[1])
            phase = np.radians(float(row[2])) if len(row) > 2 and row[2] is not None else float(rng.uniform(0.0, 2.0 * np.pi))
            lines.append((n * f1, 10.0 ** (level / 20.0), phase))                # eq. 29: 20 log10 A_n, A_n the peak
    if not lines:
        raise ValueError("a tonal source needs `lines` or a `family`")
    return lines


# --- paths ----------------------------------------------------------------------------------

def one_way_paths(provider, rx_reference, position, velocity=None, bounces: bool = False):
    """Eigenrays from a source at `position` to the receiver, as apparent sources seen from
    `rx_reference`: (p [K, 3], excess amplitude [K] complex, velocity [K, 3]).

    The amplitude is relative to spherical spreading over |p - element|. An apparent source
    moves like the image it is (the vertical velocity mirrored once per bounce), corrected
    along the ray so that its range rate is exactly the source's speed along the ray where it
    leaves the source (which differs from the image rule only when rays bend), plus the
    ray's own path-length rate when a moving boundary gives it one (sim/timespread.py)."""
    rx_reference = np.asarray(rx_reference, dtype=float)
    position = np.asarray(position, dtype=float).reshape(1, 3)
    rays = provider.eigenrays(rx_reference, position)
    amp = rays.excess_amplitude()[0]
    valid = rays.valid[0] & (amp != 0)
    p = rays.apparent_positions(rx_reference)[0][valid]
    launch, arrival = rays.launch[0][valid], rays.arrival[0][valid]
    v = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)
    parity = ((np.asarray(rays.n_surface)[0] + np.asarray(rays.n_bottom)[0])[valid] % 2).astype(float)
    w = np.tile(v, (len(p), 1))
    w[:, 2] *= 1.0 - 2.0 * parity
    w = w + ((arrival @ v) - np.einsum("ij,ij->i", w, launch))[:, None] * launch
    if rays.range_rate is not None:
        w = w + rays.range_rate[0][valid][:, None] * launch
    if bounces:
        # Also the number of boundary bounces of each ray (the direct arrival's projector pattern).
        return p, amp[valid], w, (np.asarray(rays.n_surface)[0] + np.asarray(rays.n_bottom)[0])[valid]
    return p, amp[valid], w


def _channel_clocks(record):
    n_ch = record.samples.n_channels
    meta = record.metadata
    offset, drift = np.zeros(n_ch), np.zeros(n_ch)
    for i in range(min(n_ch, len(meta.channel_board))):
        b = meta.channel_board[i]
        if b < len(meta.boards):
            offset[i] = meta.boards[b].clock_offset
            drift[i] = meta.boards[b].clock_drift_ppm * 1e-6
    return offset, drift


def _elements(record):
    """Element positions [C, 3] at the transmit instant (true: `rx_positions` includes the
    position errors the simulator put in the record's calibration, as the kernel does) and
    their velocities."""
    h = 1e-4
    pos = np.asarray(record.rx_positions(0.0), dtype=float)
    vel = (np.asarray(record.rx_positions(h), dtype=float) - np.asarray(record.rx_positions(-h), dtype=float)) / (2.0 * h)
    return pos, vel


def retarded_delay(p, w, rx_p, rx_v, t, c: float):
    """Travel time s >= 0 of the sound received at time t by a receiver at rx_p + rx_v t from
    a source at p + w u (u = t - s the emission time): the root of |P(t - s) - R(t)| = c s.
    p, w: [3]; rx_p, rx_v: [C, 3]; t: [C, M]. Returns (s [C, M], unit vector from the receiver
    to the source at emission [C, M, 3])."""
    a = p[None, None, :] + w[None, None, :] * t[..., None] - rx_p[:, None, :] - rx_v[:, None, :] * t[..., None]
    aw = a @ w
    k = c * c - float(w @ w)
    s = (-aw + np.sqrt(aw * aw + k * np.einsum("cmi,cmi->cm", a, a))) / k
    d = a - w[None, None, :] * s[..., None]
    return s, d / np.maximum(c * s, 1e-12)[..., None]


def doppler_factor(source_velocity, receiver_velocity, direction, c: float) -> float:
    """TR 0702 eq. 63 for straight rays: received frequency / emitted frequency for a source
    and a receiver moving at the given velocities, `direction` the unit vector from the
    source to the receiver: (1 - V_R . n / c) / (1 - V_S . n / c)."""
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    return float((1.0 - np.dot(receiver_velocity, n) / c) / (1.0 - np.dot(source_velocity, n) / c))


# --- signals ---------------------------------------------------------------------------------

def _noise_psd(sig: dict):
    """f (Hz) -> source spectrum level, dB re 1 uPa^2/Hz at 1 m (-inf outside the band)."""
    band = sig.get("band")
    if sig.get("spectrum") is not None:
        fr = np.log(np.asarray(sig["spectrum"]["frequency"], dtype=float))
        lv = np.asarray(sig["spectrum"]["level_db"], dtype=float)
        base = lambda f: np.interp(np.log(np.maximum(f, 1e-3)), fr, lv)          # noqa: E731
    else:
        if sig.get("level_db") is None:
            raise ValueError("a noise source needs `level_db` (dB re 1 uPa^2/Hz at 1 m) or a `spectrum` table")
        level, slope, ref = float(sig["level_db"]), float(sig.get("slope_db_per_octave", 0.0)), float(sig.get("reference_frequency", 1e3))
        base = lambda f: level + slope * np.log2(np.maximum(f, 1e-3) / ref)       # noqa: E731

    def psd(f):
        f = np.asarray(f, dtype=float)
        out = base(f)
        if band is not None:
            out = np.where((f >= float(band[0])) & (f <= float(band[1])), out, -np.inf)
        return out
    return psd


def _noise_series(n: int, rate: float, psd_db, carrier: float, lo: float, hi: float, sens_db: float, rng) -> np.ndarray:
    """A realisation of Gaussian noise as emission-time samples at `rate`: real (carrier 0)
    or the complex envelope about `carrier`, with the one-sided spectrum `psd_db` kept
    between the absolute frequencies lo and hi (the receiver's band) and zero outside."""
    if carrier:
        f = np.fft.fftfreq(n, 1.0 / rate) + carrier
        keep = (f > lo) & (f < hi) & (f > 0.0)
        level = np.where(keep, 10.0 ** ((np.where(keep, psd_db(np.where(keep, f, 1.0)), -np.inf) + sens_db) / 10.0), 0.0)
        # Envelope of a real signal with one-sided density S: two-sided density 2 S about the carrier.
        g = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
        return np.fft.ifft(g * np.sqrt(2.0 * level * rate * n))
    f = np.fft.rfftfreq(n, 1.0 / rate)
    keep = (f > lo) & (f < hi)
    level = np.where(keep, 10.0 ** ((np.where(keep, psd_db(np.where(keep, f, 1.0)), -np.inf) + sens_db) / 10.0), 0.0)
    g = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    return np.fft.irfft(g * np.sqrt(level * rate * n / 4.0), n=n)


def _file_series(sig: dict, state: dict, carrier: float, lo: float, hi: float, sens_db: float):
    """(samples, rate) of a file signal: real, or the complex envelope about `carrier`,
    limited to the receiver's band; scaled to sample units. Cached on the source."""
    key = ("file", carrier, lo, hi)
    if key not in state:
        data, rate = wf_mod._read_file(sig["path"], sig.get("dataset"), float(sig["sample_rate"]) if sig.get("sample_rate") is not None else None)
        if rate is None:
            raise ValueError("a file source needs `sample_rate` (the file does not give one)")
        data = np.real(np.asarray(data)).astype(float)
        rms = float(np.sqrt(np.mean(data * data)))
        if sig.get("level_db") is not None:
            scale = 10.0 ** ((float(sig["level_db"]) + sens_db) / 20.0) / max(rms, 1e-300)
        else:
            scale = float(sig.get("scale", 1.0)) * 10.0 ** (sens_db / 20.0)
        spectrum = np.fft.fft(data * scale)
        f = np.fft.fftfreq(len(data), 1.0 / rate)
        if carrier:
            if hi > 0.5 * rate:
                raise ValueError("file source: the file's sample rate does not cover the receiver's band")
            # The analytic signal in the receiver's band, then its envelope about the carrier
            # in the file's own time (sample m at m / rate).
            analytic = np.fft.ifft(np.where((f > lo) & (f < hi), 2.0 * spectrum, 0.0))
            cyc = carrier * np.arange(len(data)) / rate
            series = analytic * np.exp(-2j * np.pi * (cyc - np.floor(cyc)))
        else:
            series = np.real(np.fft.ifft(np.where((np.abs(f) > lo) & (np.abs(f) < hi), spectrum, 0.0)))
        state[key] = (series, float(rate))
    return state[key]


# --- the streaming synthesiser ---------------------------------------------------------------

def _stream(record, paths, env_abs, c: float, *, tones=(), series=None, series_rate: float = 0.0,
            series_first: float = 0.0, series_carrier: float = 0.0, time_zero: float = 0.0,
            interp=(wf_mod.DEFAULT_TAPS, wf_mod.DEFAULT_BETA), spreading: bool = True,
            knot_interval=None, f_top: float | None = None) -> np.ndarray:
    """One-way arrivals of a continuous signal for every (ray, channel), [C, N] in the
    record's sample kind. `tones`: [(frequency, peak amplitude in sample units, phase)],
    evaluated at scenario time `time_zero` + emission time. `series`: emission-time samples
    (real, or a complex envelope about `series_carrier`), sample m at record time
    `series_first` + m / `series_rate`. `env_abs(f)` gives absorption in dB/m.
    `knot_interval`: seconds between exact retarded-time solutions (None: KNOT_INTERVAL), or
    "auto" to halve it until linear interpolation of the delay is good to
    KNOT_PHASE_TOLERANCE at `f_top` (Hz; None: the top of the record's band)."""
    p_all, amp_all, w_all = paths
    s = record.samples
    baseband = s.kind == core.SampleKind.ComplexBaseband
    fs, start, n, fc = s.sample_rate, s.start_time, s.n_samples, s.center_frequency
    rx_p, rx_v = _elements(record)
    n_ch = rx_p.shape[0]
    offset, drift = _channel_clocks(record)
    rot_rx = np.asarray(core.rotation_ned(*record.geometry.rx_pose.rpy))
    elements = list(record.geometry.rx)
    directional = any(e.beamwidth_az == e.beamwidth_az or e.beamwidth_el == e.beamwidth_el for e in elements)
    out = np.zeros((n_ch, n), dtype=complex if baseband else float)
    idx = np.arange(n, dtype=float)
    t_true = lambda k: start + offset[:, None] + k[None, :] / (fs * (1.0 + drift[:, None]))       # noqa: E731
    auto = isinstance(knot_interval, str)
    if auto and knot_interval != "auto":
        raise ValueError("stream.knot_interval must be seconds or 'auto'")
    interval = KNOT_INTERVAL if (knot_interval is None or auto) else float(knot_interval)
    if not interval > 0.0:
        raise ValueError("stream.knot_interval must be positive")
    if f_top is None:
        f_top = (fc + 0.5 * fs) if baseband else 0.5 * fs
    for _ in range(12):
        n_knots = max(int(np.ceil(n / fs / interval)) + 1, 2)
        knots = np.linspace(0.0, n - 1.0, n_knots)
        t_knots = t_true(knots)                                                                   # [C, J]
        worst = max((knot_phase_error(retarded_delay(p_all[k], w_all[k], rx_p, rx_v, t_knots, c)[0], f_top)
                     for k in range(len(p_all))), default=0.0)
        if not auto or worst <= KNOT_PHASE_TOLERANCE or n_knots >= n:
            break
        interval *= 0.5
    if worst > KNOT_PHASE_LIMIT:
        warnings.warn(f"one-way source: reading the delay linearly between knots {interval * 1e3:.3g} ms apart leaves "
                      f"up to {worst:.2g} rad at {f_top:.0f} Hz for this geometry (a fast pass close to the array); "
                      "set the source's `stream: {knot_interval: auto}` or a shorter interval", RuntimeWarning,
                      stacklevel=2)
    t_nominal = start + idx / fs
    spectrum = None
    if series is not None:
        spectrum = np.fft.fft(series)
        f_series = np.abs(np.fft.fftfreq(len(series), 1.0 / series_rate) + series_carrier)
    chunk = max(1, int(2_000_000 // max(n, 1)))
    for k in range(len(p_all)):
        delay_k, direction = retarded_delay(p_all[k], w_all[k], rx_p, rx_v, t_knots, c)            # [C, J]
        gain_k = np.ones_like(delay_k)
        if spreading:
            gain_k = gain_k / np.maximum(c * delay_k, 1e-9)
        pattern = np.stack([wf_mod._directivity(elements[i], rot_rx, direction[i], np.array([1.0]))[:, 0] for i in range(n_ch)])
        mean_range = float(np.mean(c * delay_k))
        shaped = None
        if series is not None:
            shaped = np.fft.ifft(spectrum * 10.0 ** (-env_abs(f_series) * mean_range / 20.0))
            if not np.iscomplexobj(series):
                shaped = shaped.real
        for c0 in range(0, n_ch, chunk):
            rows = slice(c0, min(c0 + chunk, n_ch))
            delay = np.stack([np.interp(idx, knots, delay_k[i]) for i in range(rows.start, rows.stop)])
            gain = np.stack([np.interp(idx, knots, (gain_k * pattern)[i]) for i in range(rows.start, rows.stop)])
            t = t_true(idx)[rows]
            u = t - delay                                                    # emission time
            total = np.zeros(u.shape, dtype=complex if (baseband or series_carrier) else float)
            for freq, peak, phase in tones:
                if baseband and abs(freq - fc) >= 0.5 * fs:
                    continue                                                 # outside the receiver's band
                if not baseband and freq >= 0.5 * fs:
                    continue
                loss = 10.0 ** (-float(env_abs(np.array([freq]))[0]) * mean_range / 20.0)
                g_tone = gain
                if directional and fc > 0.0:
                    # A directional element: its pattern at the tone's own frequency.
                    ratio = np.array([freq / fc])
                    pat = np.stack([wf_mod._directivity(elements[i], rot_rx, direction[i], ratio)[:, 0] for i in range(rows.start, rows.stop)])
                    g_tone = np.stack([np.interp(idx, knots, gain_k[i] * pat[j]) for j, i in enumerate(range(rows.start, rows.stop))])
                cycles = freq * (time_zero + u) + phase / (2.0 * np.pi)
                if baseband:
                    cycles = cycles - fc * t_nominal[None, :]
                    total += peak * loss * g_tone * np.exp(2j * np.pi * (cycles - np.floor(cycles)))
                else:
                    total += peak * loss * g_tone * np.cos(2.0 * np.pi * (cycles - np.floor(cycles)))
            if shaped is not None:
                value = wf_mod.interpolate(shaped, (u - series_first) * series_rate, interp[0], interp[1])
                if baseband:
                    cycles = series_carrier * u - fc * t_nominal[None, :]
                    value = value * np.exp(2j * np.pi * (cycles - np.floor(cycles)))
                elif series_carrier:
                    cycles = series_carrier * u
                    value = value * np.exp(2j * np.pi * (cycles - np.floor(cycles)))      # analytic: a ray's phase applies
                total += gain * value
            contribution = amp_all[k] * total
            out[rows] += contribution if baseband else np.real(contribution)
    return out


# --- one-way pulses through the kernel -------------------------------------------------------

def _emission_times(sig: dict, scenario_time: float, t_lo: float, t_hi: float) -> list:
    """Emission times relative to the ping's transmit instant that can reach the record:
    those in [t_lo, t_hi] (record seconds)."""
    if sig.get("times") is not None:
        times = np.asarray(sig["times"], dtype=float) - scenario_time
    elif sig.get("repeat") is not None:
        r = sig["repeat"]
        first, interval = float(r.get("first", 0.0)), float(r["interval"])
        k0 = int(np.ceil((scenario_time + t_lo - first) / interval))
        k1 = int(np.floor((scenario_time + t_hi - first) / interval))
        if r.get("count") is not None:
            k1 = min(k1, int(r["count"]) - 1)
        times = first + interval * np.arange(max(k0, 0), k1 + 1) - scenario_time
    else:
        times = np.array([float(sig.get("ping_offset", 0.0))])
    return [float(t) for t in times if t_lo <= t <= t_hi]


def _source_sonar(sonar, sig: dict):
    """A copy of the sonar whose transmit waveform is the source's: it gives the record's
    waveform fields, the envelope table of a sampled pulse and the absorption band."""
    from ..sonar import SonarModel
    import copy
    from ..sonar import SOURCE_LEVEL_ALIAS, SOURCE_LEVEL_KEY, source_level_db_rms
    level = source_level_db_rms(sig, "sources[].signal")          # RMS, dB re 1 uPa at 1 m; `source_level_db` is the deprecated name
    if sig.get("waveform") is None and level is None and sig.get("amplitude") is None:
        return sonar
    cfg = copy.deepcopy(sonar.config)
    w = dict(sig["waveform"]) if sig.get("waveform") is not None else dict(cfg["transmit"]["waveform"])
    if level is not None:
        w.pop(SOURCE_LEVEL_ALIAS, None)
        w[SOURCE_LEVEL_KEY] = level
    if sig.get("amplitude") is not None:
        w["amplitude"] = sig["amplitude"]
        w.pop(SOURCE_LEVEL_ALIAS, None)
        w.pop(SOURCE_LEVEL_KEY, None)
    cfg["transmit"] = dict(cfg["transmit"], waveform=w)
    return SonarModel.from_config(cfg)


def _pulse(record, sonar, source, sig, provider, env, absorption_model, scenario_time, c, backend, engine, spreading, acceleration,
           ray_gain=None):
    """The one-way arrivals of every emission of a pulsed source that reaches the record,
    through the echo synthesis kernel (module docstring): [C, N] to add, or None.
    `ray_gain(p, bounces) -> [C]` (the direct arrival's projector pattern): the kernel is then
    run once per ray and each ray's arrivals are scaled per channel."""
    from .simulator import absorption_table
    if sig.get("_sonar") is None:
        sig["_sonar"] = _source_sonar(sonar, sig)
    src_sonar = sig["_sonar"]
    s = record.samples
    kind, start, t_end = s.kind, s.start_time, s.start_time + s.n_samples / s.sample_rate
    baseband = kind == core.SampleKind.ComplexBaseband
    center = s.center_frequency
    wcfg = src_sonar.waveform_config
    duration = float(wcfg["duration"])
    rx_ref0 = np.asarray(record.rx_positions(), dtype=float).mean(axis=0)
    pos0, _ = source.at(scenario_time)
    # Emissions early enough to arrive before the record ends and late enough to still be heard.
    span = float(sig.get("max_delay", 3.0 * float(np.linalg.norm(pos0 - rx_ref0)) / c + 1.0))
    times = _emission_times(sig, scenario_time, start - span - duration, t_end)
    if not times:
        return None
    table = absorption_table(src_sonar, env, absorption_model)
    fc = 0.5 * (float(wcfg["f_start"]) + float(wcfg.get("f_stop", wcfg["f_start"])))
    alpha = float(wf_mod._absorption(table, np.array([fc]))[0])
    # A copy of the record takes the changes the kernel needs; its samples are what is added.
    work = record.copy()
    blank = work.samples
    blank.array = np.zeros_like(blank.array)
    blank.kind = kind
    work.samples = blank
    rx_pose0 = record.copy().geometry.rx_pose
    v_rx = np.asarray(rx_pose0.velocity, dtype=float)
    template = src_sonar.make_record(rx_pose0, environment={"sound_speed": c})
    sampled = template.transmit.waveform.type == core.WaveformType.Arbitrary
    t = work.transmit
    t.waveform = template.transmit.waveform
    work.transmit = t
    weighted = None
    for t_e in times:
        shift = t_e - KERNEL_LEG / c                       # kernel time = true time - shift
        pose = core.Pose()
        pose.position = tuple(np.asarray(rx_pose0.position, dtype=float) + v_rx * shift)
        pose.rpy = tuple(np.asarray(rx_pose0.rpy, dtype=float) + np.asarray(rx_pose0.rates, dtype=float) * shift)
        pose.velocity, pose.rates = rx_pose0.velocity, rx_pose0.rates
        tx_pose = core.Pose()
        tx_pose.position, tx_pose.velocity = pose.position, pose.velocity
        geo = work.geometry
        geo.rx_pose, geo.tx_pose = pose, tx_pose
        tx = geo.tx
        tx.lever_arm = (0.0, 0.0, 0.0)
        tx.beamwidth_az = tx.beamwidth_el = float("nan")
        geo.tx = tx
        work.geometry = geo
        smp = work.samples
        smp.start_time = start - shift
        work.samples = smp
        src_pos, src_vel = source.at(scenario_time + t_e)
        rx_ref = np.asarray(work.rx_positions(), dtype=float).mean(axis=0)
        if ray_gain is None:
            p, amp, w = one_way_paths(provider, rx_ref, src_pos, src_vel)
        else:
            p, amp, w, n_bounce = one_way_paths(provider, rx_ref, src_pos, src_vel, bounces=True)
        if len(p) == 0:
            continue
        tx_pos = np.asarray(pose.position, dtype=float)
        q = np.tile(tx_pos + KERNEL_LEG * (np.array([1.0, 0.0, 0.0]) - v_rx / c), (len(p), 1))
        vq = np.tile(v_rx, (len(p), 1))
        amp = amp * KERNEL_LEG * 10.0 ** (alpha * KERNEL_LEG / 20.0)
        if baseband:
            cyc = center * shift
            amp = amp * np.exp(-2j * np.pi * (cyc - np.floor(cyc)))
        p_kernel = p - w * (KERNEL_LEG / c)

        def run(rays):
            if sampled:
                wf_mod.synthesize(work, src_sonar.sampled_waveform, q[rays], p_kernel[rays], amp[rays], vq[rays], w[rays], absorption=table,
                                  spreading=spreading, platform_acceleration=tuple(float(a) for a in acceleration),
                                  backend=backend, engine=engine)
            else:
                core.synthesize_into(work, q[rays], p_kernel[rays], amp[rays], vq[rays], w[rays], absorption=table, spreading=spreading,
                                     backend=backend, platform_acceleration=tuple(float(a) for a in acceleration))

        if ray_gain is None:
            run(slice(None))
            continue
        for k in range(len(p)):
            # One ray at a time, so that each can be weighted per channel.
            smp = work.samples
            smp.array = np.zeros_like(smp.array)
            smp.kind = kind
            work.samples = smp
            run(slice(k, k + 1))
            part = np.asarray(ray_gain(p[k], int(n_bounce[k])))[:, None] * np.array(work.samples.array)
            weighted = part if weighted is None else weighted + part
    if ray_gain is not None:
        return weighted
    return np.array(work.samples.array)


# --- entry point -----------------------------------------------------------------------------

def add_sources(record, sonar, sources, provider, *, environment: dict, absorption_model: str = "francois_garrison",
                scenario_time: float = 0.0, rng: np.random.Generator | None = None, backend=None,
                engine: str | None = None, spreading: bool = True, acceleration=(0.0, 0.0, 0.0)) -> None:
    """Add the one-way sound of `sources` (from `build_sources`) into record.samples. The
    record's true board clocks and element position errors must already be set, as for echoes."""
    if not sources:
        return
    rng = rng or np.random.default_rng()
    backend = core.Backend.Host if backend is None else backend
    env = environment
    c = record.environment.sound_speed
    s = record.samples
    baseband = s.kind == core.SampleKind.ComplexBaseband
    fs, start, n, fc = s.sample_rate, s.start_time, s.n_samples, s.center_frequency
    rx_cfg = sonar.config["receive"]
    sens = float(rx_cfg.get("sensitivity_db") or 0.0) + float(rx_cfg.get("gain_db", 0.0))
    lo, hi = (fc - 0.5 * fs, fc + 0.5 * fs) if baseband else (0.0, 0.5 * fs)

    def env_abs(f):
        return np.asarray(acoustics.absorption(np.maximum(np.asarray(f, dtype=float), 1.0), *site.water(env), absorption_model), dtype=float) * np.ones(np.shape(f))

    rx_ref = np.asarray(record.rx_positions(), dtype=float).mean(axis=0)
    added = None
    for source in sources:
        position, velocity = source.at(scenario_time)
        paths = None
        for sig in source.signals:
            kind = sig["type"]
            if kind == "waveform":
                part = _pulse(record, sonar, source, sig, provider, env, absorption_model, scenario_time, c, backend, engine, spreading, acceleration)
                if part is not None:
                    added = part if added is None else added + part
                continue
            if paths is None:
                paths = one_way_paths(provider, rx_ref, position, velocity)
            if len(paths[0]) == 0:
                continue
            reach = float(np.max(np.linalg.norm(paths[0] - rx_ref, axis=1))) / c
            speed = float(np.max(np.linalg.norm(paths[2], axis=1))) + float(np.linalg.norm(record.geometry.rx_pose.velocity))
            margin = (wf_mod.DEFAULT_TAPS + 4) / fs + (reach + n / fs + abs(start)) * speed / c + 0.01
            u_first, u_last = start - reach - margin, start + n / fs + margin
            # A real record with complex ray amplitudes (reflection phases): carry the signal
            # as an analytic envelope so the phase can be applied.
            analytic = baseband or bool(np.any(np.abs(np.imag(paths[1])) > 1e-12 * np.abs(paths[1])))
            carrier = fc if baseband else (0.25 * fs if analytic else 0.0)
            if kind == "tonals":
                tones = [(f, a * 10.0 ** (sens / 20.0), ph) for f, a, ph in sig["_lines"]]
                in_band = [f for f, _, _ in tones if lo < f < hi]
                stream_kw = dict(knot_interval=source.stream.get("knot_interval"), f_top=max(in_band) if in_band else hi)
                if analytic and not baseband:
                    part = _stream_analytic_tones(record, paths, env_abs, c, tones, scenario_time, spreading, **stream_kw)
                else:
                    part = _stream(record, paths, env_abs, c, tones=tones, time_zero=scenario_time, spreading=spreading,
                                   **stream_kw)
                added = part if added is None else added + part
                continue
            else:
                rate = float(source.stream.get("oversample", STREAM_OVERSAMPLE)) * fs
                if kind == "noise":
                    m = int(np.ceil((u_last - u_first) * rate)) + 1
                    series = _noise_series(m, rate, _noise_psd(sig), carrier, lo, hi, sens, rng)
                    first = u_first
                else:
                    data, rate = _file_series(sig, source.state, carrier, lo, hi, sens)
                    t0 = float(sig.get("start", 0.0)) - scenario_time            # record time of the file's first sample
                    i0 = int(np.floor((u_first - t0) * rate))
                    i1 = int(np.ceil((u_last - t0) * rate)) + 1
                    take = np.arange(i0, i1)
                    if sig.get("loop"):
                        series = data[np.mod(take, len(data))]
                        if carrier:
                            # The envelope's carrier phase runs on with the file's own time.
                            wraps = np.floor_divide(take, len(data))
                            series = series * np.exp(-2j * np.pi * carrier * wraps * len(data) / rate)
                    else:
                        inside = (take >= 0) & (take < len(data))
                        series = np.where(inside, data[np.clip(take, 0, len(data) - 1)], 0.0)
                    first = t0 + i0 / rate
                    if carrier:
                        # The envelope is about the carrier in the file's time; `_stream` wants record time.
                        cyc = carrier * t0
                        series = series * np.exp(-2j * np.pi * (cyc - np.floor(cyc)))
                part = _stream(record, paths, env_abs, c, series=series, series_rate=rate, series_first=first,
                               series_carrier=carrier, spreading=spreading,
                               knot_interval=source.stream.get("knot_interval"), f_top=hi)
            added = part if added is None else added + part
    if added is not None:
        s = record.samples
        kind = s.kind
        s.array = (s.array + added.astype(s.array.dtype))
        s.kind = kind
        record.samples = s


def _stream_analytic_tones(record, paths, env_abs, c, tones, time_zero, spreading, **stream_kw):
    """Tones into a real record when ray amplitudes are complex: each ray's phase is added
    to the tone's phase (the exact effect of a frequency-independent reflection phase)."""
    p, amp, w = paths
    out = None
    for k in range(len(p)):
        phase = float(np.angle(amp[k]))
        shifted = [(f, a, ph + phase) for f, a, ph in tones]
        part = _stream(record, (p[k:k + 1], np.array([abs(amp[k])]), w[k:k + 1]), env_abs, c, tones=shifted,
                       time_zero=time_zero, spreading=spreading, **stream_kw)
        out = part if out is None else out + part
    return out


# --- the sonar's own projector heard directly (`transmit.direct_path`) -----------------------

class _Projector:
    """The sonar's projector as a one-way source: where it is at the transmit instant, moving
    with the vehicle."""

    def __init__(self, position, velocity, scenario_time: float):
        self.position, self.velocity, self.time = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float), float(scenario_time)
        self.signals, self.state = [{"type": "waveform", "ping_offset": 0.0}], {}

    def at(self, t: float):
        return self.position + self.velocity * (float(t) - self.time), self.velocity


def direct_path_config(sonar) -> dict | None:
    """`transmit.direct_path` of a sonar: None when absent or false (nothing changes), else
    {"boundaries": bool}. `true` is {boundaries: true}: the arrival straight from the projector
    and every other eigenray the scenario's provider finds between the projector and the array
    (surface and bottom images, to the provider's bounce order). `{boundaries: false}`: the
    straight arrival only."""
    cfg = sonar.config["transmit"].get("direct_path")
    if not cfg:
        return None
    cfg = {} if cfg is True else dict(cfg)
    unknown = set(cfg) - {"boundaries"}
    if unknown:
        raise ValueError(f"transmit.direct_path: unknown keys {sorted(unknown)}")
    return {"boundaries": bool(cfg.get("boundaries", True))}


def direct_arrival(record, sonar, provider, options: dict | None = None, *, environment: dict,
                   absorption_model: str = "francois_garrison", scenario_time: float = 0.0, backend=None,
                   engine: str | None = None, spreading: bool = True, acceleration=(0.0, 0.0, 0.0),
                   window: tuple | None = None):
    """The transmitted pulse as the array hears it straight from the projector (and off the
    boundaries): [C, N] in the record's sample units for the record's receive window, or for
    `window` = (start_time, n_samples); None when nothing arrives in it. Nothing is added to
    the record.

    This is a one-way source at the projector's true position (`record.tx_position()`), moving
    with the vehicle, emitting the sonar's own waveform at the transmit instant, through the
    one-way pulse path above: the same kernel as the echoes (C++, host or CUDA, for analytic
    pulses on both engines; C++ or NumPy by `engine` for a sampled pulse), with each
    element's own distance to the projector (and to its images) for the delay, the phase and
    the 1/r spreading, the receive pattern of each element in the arrival's own direction,
    absorption, the board clocks and the true element positions.

    The one-way path makes the projector omnidirectional. A projector with a beam pattern is
    weighted here, per ray and per element, by its pattern at the band centre in the direction
    the ray leaves it towards that element: the straight line from the apparent source to the
    element, mirrored in the horizontal once per boundary bounce (exact for flat horizontal
    boundaries and straight rays; with walls or bent rays, an approximation)."""
    from .propagation import make_provider
    options = options or {"boundaries": True}
    c = record.environment.sound_speed
    backend = core.Backend.Host if backend is None else backend
    if not options.get("boundaries", True):
        provider = make_provider("free_field", c)
    h = 1e-4
    tx_position = np.asarray(record.tx_position(0.0), dtype=float)
    tx_velocity = (np.asarray(record.tx_position(h), dtype=float) - np.asarray(record.tx_position(-h), dtype=float)) / (2.0 * h)
    projector = _Projector(tx_position, tx_velocity, scenario_time)
    work = record
    if window is not None:
        # Another stretch of time on the same sample clock (the interval before the window opens).
        work = record.copy()
        smp = work.samples
        kind = smp.kind
        smp.array = np.zeros((smp.n_channels, int(window[1])), dtype=smp.array.dtype)
        smp.kind = kind
        smp.start_time = float(window[0])
        work.samples = smp
    tx = record.geometry.tx
    ray_gain = None
    if tx.beamwidth_az == tx.beamwidth_az or tx.beamwidth_el == tx.beamwidth_el:
        rot_tx = np.asarray(core.rotation_ned(*record.geometry.tx_pose.rpy))
        elements, _ = _elements(record)

        def ray_gain(p, n_bounce):
            d = elements - np.asarray(p, dtype=float)[None, :]
            d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            if n_bounce % 2:
                d = d * np.array([1.0, 1.0, -1.0])
            return wf_mod._directivity(tx, rot_tx, d, np.array([1.0]))[:, 0]

    return _pulse(work, sonar, projector, projector.signals[0], provider, environment, absorption_model, scenario_time, c,
                  backend, engine, spreading, acceleration, ray_gain=ray_gain)
