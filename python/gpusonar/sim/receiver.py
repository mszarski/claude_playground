"""Receive chain effects applied after echo synthesis: noise, the ADC, and (opt-in,
`receive.overload`) a front end that clips and recovers."""

from __future__ import annotations

import numpy as np

from .. import _gpusonar as core


def add_noise(samples: np.ndarray, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise of the given RMS (per real component for complex data)."""
    if rms <= 0.0:
        return samples
    if np.iscomplexobj(samples):
        noise = rng.standard_normal(samples.shape) + 1j * rng.standard_normal(samples.shape)
        return (samples + rms * noise).astype(np.complex64)
    return (samples + rms * rng.standard_normal(samples.shape)).astype(np.float32)


def apply_hardware_tvg(record: "core.PingRecord", samples: np.ndarray) -> np.ndarray:
    """Apply the analogue gain ramp described by the record (piecewise linear in dB)."""
    t_knots, g_knots = record.metadata.tvg_time, record.metadata.tvg_gain_db
    if len(t_knots) == 0:
        return samples
    s = record.samples
    t = s.start_time + np.arange(s.n_samples) / s.sample_rate
    gain = 10.0 ** (np.interp(t, t_knots, g_knots) / 20.0)
    return (samples * gain).astype(samples.dtype)


def quantise(samples: np.ndarray, bits: int, full_scale: float) -> tuple[np.ndarray, float]:
    """Mid-tread ADC: round to 2^bits levels over [-full_scale, +full_scale) and clip.

    Returns the quantised samples (still in physical units) and the step size.
    """
    step = full_scale / 2.0 ** (bits - 1)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1

    def q(x):
        return np.clip(np.rint(x / step), lo, hi) * step

    if np.iscomplexobj(samples):
        return (q(samples.real) + 1j * q(samples.imag)).astype(np.complex64), step
    return q(samples).astype(np.float32), step


def apply_adc(record: "core.PingRecord", samples: np.ndarray) -> np.ndarray:
    """Quantise per the record's ADC description and set its integer storage to match."""
    bits = record.metadata.adc_bits
    full_scale = record.metadata.adc_full_scale
    if bits <= 0 or not np.isfinite(full_scale):
        return samples
    out, step = quantise(samples, bits, full_scale)
    s = record.samples
    s.storage = core.StorageType.Int16 if bits <= 16 else core.StorageType.Int32
    s.storage_scale = step
    record.samples = s
    return out


def sync_chirp(t, f_start: float, f_stop: float, duration: float) -> np.ndarray:
    """The shared reference burst at times `t` after its start: a Hann-windowed linear chirp."""
    t = np.asarray(t, dtype=float)
    inside = (t >= 0.0) & (t <= duration)
    k = (f_stop - f_start) / duration
    tt = np.where(inside, t, 0.0)
    return np.where(inside, np.sin(np.pi * tt / duration) ** 2
                    * np.cos(2.0 * np.pi * (f_start * tt + 0.5 * k * tt**2)), 0.0)


def reference_times(record: "core.PingRecord", ref: dict) -> tuple[float, float]:
    """True times (after transmit) of the two reference bursts for this record's window."""
    s = record.samples
    t_end = s.start_time + s.n_samples / s.sample_rate
    return s.start_time + float(ref["lead"]), t_end - float(ref["tail"]) - float(ref["duration"])


REFERENCE_MODES = {"sync_chirp"}


def add_reference_signal(record: "core.PingRecord", sonar, clocks: dict) -> None:
    """Sum the shared synchronisation reference into each board's reference input, as that
    board's own (wrong) sample clock sees it."""
    ref = sonar.config["receive"].get("reference")
    if not ref:
        return
    if ref.get("mode") not in REFERENCE_MODES:
        # A misspelt mode used to remove the reference silently (audit A15).
        raise ValueError(f"receive.reference.mode {ref.get('mode')!r}: the simulated modes are {sorted(REFERENCE_MODES)}")
    if record.samples.is_complex():
        return
    s = record.samples
    x = s.array
    t_a, t_b = reference_times(record, ref)
    n = np.arange(s.n_samples)
    for board, channel in enumerate(ref["channels"]):
        if channel not in record.metadata.channel_element:
            continue
        row = record.metadata.channel_element.index(channel)
        if ref.get("dedicated"):
            x[row] = 0.0                      # a reference input has no hydrophone on it
        offset = clocks["offset"][board]
        drift = clocks["drift_ppm"][board] * 1e-6
        t_true = s.start_time + offset + n / (s.sample_rate * (1.0 + drift))
        for t0 in (t_a, t_b):
            x[row] += float(ref["level"]) * sync_chirp(t_true - t0, ref["f_start"], ref["f_stop"],
                                                       ref["duration"]).astype(np.float32)
    s.array = x
    record.samples = s


def apply_channel_errors(record: "core.PingRecord", errors: dict) -> None:
    """Impose per-channel gain and phase errors on the samples (the truth a calibration has
    to find), then leave the record's calibration tables neutral: the processor starts
    without them. Position errors were already used by the synthesis."""
    s = record.samples
    x = s.array
    n_ch = x.shape[0]
    gain = np.asarray(errors.get("gain", np.ones(n_ch)), dtype=float)
    phase = np.asarray(errors.get("phase", np.zeros(n_ch)), dtype=float)
    factor = gain * np.exp(1j * phase)
    kind = s.kind
    if np.iscomplexobj(x):
        s.array = (x * factor[:, None]).astype(np.complex64)
    else:
        # A phase shift of a real signal: rotate its analytic signal and take the real part.
        spectrum = np.fft.rfft(x, axis=1)
        s.array = np.fft.irfft(spectrum * factor[:, None], n=x.shape[1], axis=1).astype(np.float32)
    s.kind = kind
    record.samples = s
    cal = record.calibration
    cal.position_offset = [(0.0, 0.0, 0.0)] * n_ch
    record.calibration = cal


# --- overload and recovery of the front end (`receive.overload`) --------------------------------
#
# A behavioural model: no claim about any particular board. Everything is carried in the
# record's own sample units, volts at the ADC with the whole of `receive.gain_db` applied
# (the simulator's echoes are made that way), so a limit given in volts at the input stage's
# output is first referred to the ADC through the gain that follows that stage.
#
#   x  -> gain switch a(t) -> input stage: limit, dead time, gain recovery
#      -> AC coupling (one-pole high-pass) -> `receive.tvg` ramp -> ADC: limit, dead time, recovery
#
# One limiting stage, for input u, limits -L_n and +L_p, dead time T_d, recovery time
# constant tau_g, with tau the time since the stage last clipped (its own channel, or any
# channel of its board when the stage is shared):
#
#   clipped (u >= L_p or u <= -L_n):   y = the rail
#   0 < tau <= T_d:                    y = the rail it last hit (`dead_output: rail`) or 0
#   tau > T_d:                         y = g u,   g = 1 - exp(-(tau - T_d) / tau_g)   (1 if tau_g = 0)
#
# The coupling capacitor between the stages: v_C' = (y1 - v_C) / tau_c, tau_c = 1 / (2 pi f_c),
# output y2 = y1 - v_C. The capacitor follows the mean of what the input stage puts out. An
# undistorted or symmetrically clipped burst has no mean and leaves no charge. A burst whose
# clipped output has mean m for a time T (unequal rails; the rail held through the dead time)
# leaves v_C = m (1 - exp(-T / tau_c)), and after it the output is the signal minus
# v_C exp(-t / tau_c): an offset that decays with the corner's time constant and, where it
# exceeds the ADC's full scale after the gain behind the capacitor, holds the ADC on its rail
# for tau_c ln(|v_C| / full scale). In samples it is the bilinear transform of s / (s + 1 / tau_c)
# with the corner pre-warped: unit gain and the right phase in the band, and a step decaying
# by (1 - k) / (1 + k) a sample, k = tan(pi f_c / fs), which is exp(-1 / (fs tau_c)) to k^2 / 3.

STAGE_KEYS = {"limit_v", "gain_db", "dead_time", "dead_output", "recovery_tau", "scope", "clip"}
OVERLOAD_KEYS = {"preamp", "adc", "highpass_corner_hz", "gain_switch"}
FLAG_PREAMP, FLAG_ADC, FLAG_HELD = 1, 2, 4
GROUP_ROWS = 4                  # channels passed through the model together when no stage is shared


def _stage_config(cfg: dict, name: str) -> dict:
    unknown = set(cfg) - STAGE_KEYS
    if unknown:
        raise ValueError(f"receive.overload.{name}: unknown keys {sorted(unknown)}")
    out = {"dead_time": float(cfg.get("dead_time", 0.0)), "recovery_tau": float(cfg.get("recovery_tau", 0.0)),
           "dead_output": str(cfg.get("dead_output", "rail")), "scope": str(cfg.get("scope", "channel"))}
    if out["dead_output"] not in ("rail", "zero"):
        raise ValueError(f"receive.overload.{name}.dead_output: rail or zero")
    if out["scope"] not in ("channel", "board"):
        raise ValueError(f"receive.overload.{name}.scope: channel or board")
    if out["dead_time"] < 0.0 or out["recovery_tau"] < 0.0:
        raise ValueError(f"receive.overload.{name}: dead_time and recovery_tau must not be negative")
    return out


def overload_config(sonar) -> dict | None:
    """The front-end model of a sonar's `receive.overload` block, limits referred to the ADC;
    None when the key is absent (the receiver is then exactly what it was).

        receive:
          overload:
            preamp: {limit_v, gain_db, dead_time, dead_output, recovery_tau, scope}
            highpass_corner_hz: f
            adc: {limit_v, clip, dead_time, dead_output, recovery_tau, scope}
            gain_switch: {gain_db, until, ramp, report}

    Nothing here has a hardware value as its default: `preamp` is modelled only when given
    (its `limit_v`, volts at its output, is required; `gain_db`, the gain up to that output,
    defaults to all of `receive.gain_db`); the coupling only when `highpass_corner_hz` is
    given; the ADC limit is `receive.adc_full_scale` (`adc.limit_v` replaces it, `adc.clip:
    false` removes it); dead times and recovery time constants are 0 unless given."""
    rx = sonar.config["receive"]
    if rx.get("overload") is None:
        return None
    cfg = dict(rx["overload"] or {})
    unknown = set(cfg) - OVERLOAD_KEYS
    if unknown:
        raise ValueError(f"receive.overload: unknown keys {sorted(unknown)}")
    total_gain = float(rx.get("gain_db", 0.0))

    def limits(value, scale):
        pair = [float(v) for v in value] if isinstance(value, (list, tuple)) else [float(value)] * 2
        if len(pair) != 2 or min(pair) <= 0.0:
            raise ValueError("receive.overload: limit_v is a positive number or [negative rail, positive rail] magnitudes")
        return pair[0] * scale, pair[1] * scale

    out = {"preamp": None, "adc": None, "highpass_corner_hz": None, "gain_switch": None}
    if cfg.get("preamp") is not None:
        stage = _stage_config(cfg["preamp"], "preamp")
        if cfg["preamp"].get("limit_v") is None:
            raise ValueError("receive.overload.preamp needs limit_v (V at the input stage's output; no default)")
        gain = float(cfg["preamp"].get("gain_db", total_gain))
        stage["gain_db"] = gain
        stage["limit"] = limits(cfg["preamp"]["limit_v"], 10.0 ** ((total_gain - gain) / 20.0))
        out["preamp"] = stage
    adc = dict(cfg.get("adc") or {})
    if adc.get("clip", True):
        level = adc.get("limit_v", rx.get("adc_full_scale"))
        if level is not None and np.all(np.isfinite(np.asarray(level, dtype=float))):
            stage = _stage_config(adc, "adc")
            stage["limit"] = limits(level, 1.0)
            out["adc"] = stage
    if cfg.get("highpass_corner_hz") is not None:
        out["highpass_corner_hz"] = float(cfg["highpass_corner_hz"])
        if out["highpass_corner_hz"] <= 0.0:
            raise ValueError("receive.overload.highpass_corner_hz must be positive")
    if cfg.get("gain_switch") is not None:
        g = dict(cfg["gain_switch"])
        unknown = set(g) - {"gain_db", "until", "ramp", "report"}
        if unknown:
            raise ValueError(f"receive.overload.gain_switch: unknown keys {sorted(unknown)}")
        if g.get("gain_db") is None:
            raise ValueError("receive.overload.gain_switch needs gain_db (the gain change while transmitting, dB; no default)")
        until = float(g["until"]) if g.get("until") is not None else float(sonar.waveform_config["duration"])
        ramp = max(float(g.get("ramp", 0.0)), 1.0 / sonar.sample_rate)
        out["gain_switch"] = {"gain_db": float(g["gain_db"]), "until": until, "ramp": ramp, "report": bool(g.get("report", True))}
    return out


def report_gain_switch(record: "core.PingRecord", cfg: dict | None) -> None:
    """Write a reported gain switch into the record's hardware gain ramp (existing fields:
    metadata.tvg_time, tvg_gain_db), so that a processor undoes it as it undoes any ramp."""
    g = (cfg or {}).get("gain_switch")
    if not g or not g["report"]:
        return
    meta = record.metadata
    if len(meta.tvg_time):
        raise ValueError("receive.overload.gain_switch with report: true cannot be combined with receive.tvg / "
                         "gain_ramp (one ramp in the record); set report: false")
    meta.tvg_time = [g["until"], g["until"] + g["ramp"]]
    meta.tvg_gain_db = [g["gain_db"], 0.0]
    record.metadata = meta


def gain_switch_db(g: dict, t: np.ndarray) -> np.ndarray:
    """Gain (dB) of the transmit-time gain switch at times t after transmit: `gain_db` up
    to `until`, then linear in dB to 0 over `ramp` (held flat outside, as a record's ramp)."""
    return np.interp(t, [g["until"], g["until"] + g["ramp"]], [g["gain_db"], 0.0])


def limit_stage(u: np.ndarray, sample_rate: float, limit, dead_time: float = 0.0, recovery_tau: float = 0.0,
                dead_output: str = "rail", shared: bool = False):
    """One limiting stage (the comment above) on u [G, N]: (y, clipped, dead, gain). With
    `shared` the G rows share the dead time and the gain recovery (a stage common to a board).
    `dead`: inside a dead time and not on a rail; `gain`: the recovery gain (0 while clipped or dead)."""
    lo, hi = float(limit[0]), float(limit[1])
    up, down = u >= hi, u <= -lo
    clipped = up | down
    if not clipped.any():
        return u, clipped, clipped, np.ones(u.shape)
    if dead_time == 0.0 and recovery_tau == 0.0 and not shared:
        return np.clip(u, -lo, hi), clipped, np.zeros_like(clipped), np.where(clipped, 0.0, 1.0)
    idx = np.arange(u.shape[1])
    last = np.maximum.accumulate(np.where(clipped, idx[None, :], -1), axis=1)
    rail = np.where(up, hi, np.where(down, -lo, 0.0))
    since = np.where(last >= 0, (idx[None, :] - last) / sample_rate, np.inf)
    if shared:
        group = last.max(axis=0)[None, :]
        since_group = np.broadcast_to(np.where(group >= 0, (idx[None, :] - group) / sample_rate, np.inf), u.shape)
    else:
        since_group = since
    if recovery_tau > 0.0:
        gain = -np.expm1(-np.maximum(since_group - dead_time, 0.0) / recovery_tau)       # 1 where never clipped (inf)
    else:
        gain = np.ones(u.shape)
    gain = np.where(since_group <= dead_time, 0.0, gain)
    held = (since > 0.0) & (since <= dead_time)
    y = gain * u
    if dead_output == "rail":
        y = np.where(held, np.take_along_axis(rail, np.maximum(last, 0), axis=1), y)
    y = np.where(clipped, rail, y)
    return y, clipped, (since_group <= dead_time) & ~clipped, np.where(clipped, 0.0, gain)


def coupling_highpass(y: np.ndarray, sample_rate: float, corner_hz: float) -> np.ndarray:
    """The AC coupling: the one-pole high-pass s / (s + 2 pi f_c) by the bilinear transform
    (corner pre-warped), the capacitor uncharged before the first sample."""
    from scipy.signal import lfilter
    k = float(np.tan(np.pi * corner_hz / sample_rate))
    return lfilter([1.0 / (1.0 + k), -1.0 / (1.0 + k)], [1.0, -(1.0 - k) / (1.0 + k)], y, axis=-1)


def apply_front_end(record: "core.PingRecord", samples: np.ndarray, cfg: dict, pre: np.ndarray | None = None,
                    truth: dict | None = None, arrays: bool = False) -> np.ndarray:
    """Pass the record's samples (after noise) through the overload model `cfg`
    (`overload_config`). `pre` [C, M]: what reached the inputs in the M sample periods before
    the window opened; the model runs over it first so that the window starts with the state
    the front end is really in, and only the window is returned. This replaces
    `apply_hardware_tvg`: the record's ramp is applied here, at the input stage when it is the
    reported gain switch and ahead of the ADC otherwise.

    `truth` (a dict) receives, per channel: `peak_db_re_preamp_limit`, `peak_db_re_adc_limit`
    (the strongest drive into each stage against its rail, pre-window included; nan without
    that stage), `clipped_fraction` (of the window), `last_overload_time` (s after transmit of
    the last sample on a rail or dead; nan if never), and with `arrays` also `time` [N],
    `flags` [C, N] (bit 1: input stage on its rail, 2: ADC on its rail, 4: inside a dead
    time), `gain` [C, N] (the stages' recovery gain, 0 while clipped or dead; the gain switch
    is not in it), `offset` [C, N] (what the input stage's stuck output leaves at the ADC
    input: after an overload, the decaying charge of the coupling capacitor) and `pre_samples`."""
    if np.iscomplexobj(samples):
        raise ValueError("receive.overload needs passband samples (receive.kind: real): a complex envelope cannot be clipped")
    s = record.samples
    fs, n = s.sample_rate, samples.shape[1]
    n_ch = samples.shape[0]
    m = 0 if pre is None else pre.shape[1]
    t = s.start_time + (np.arange(n + m) - m) / fs
    switch = cfg.get("gain_switch")
    t_knots, g_knots = record.metadata.tvg_time, record.metadata.tvg_gain_db
    ramp = 10.0 ** (np.interp(t, t_knots, g_knots) / 20.0) if len(t_knots) and not (switch and switch["report"]) else None
    switch_gain = 10.0 ** (gain_switch_db(switch, t) / 20.0) if switch else None
    pa, adc, corner = cfg.get("preamp"), cfg.get("adc"), cfg.get("highpass_corner_hz")
    boards = np.asarray(list(record.metadata.channel_board), dtype=int)
    if len(boards) != n_ch:
        boards = np.zeros(n_ch, dtype=int)
    if any(st is not None and st["scope"] == "board" for st in (pa, adc)):
        groups = [np.nonzero(boards == b)[0] for b in np.unique(boards)]
    else:
        groups = [np.arange(i, min(i + GROUP_ROWS, n_ch)) for i in range(0, n_ch, GROUP_ROWS)]
    out = np.empty(samples.shape, dtype=np.float32)
    want = truth is not None
    if want:
        peak = {"preamp": np.full(n_ch, np.nan), "adc": np.full(n_ch, np.nan)}
        last_time = np.full(n_ch, np.nan)
        clipped_fraction = np.zeros(n_ch)
        if arrays:
            flags = np.zeros((n_ch, n), dtype=np.uint8)
            gain_out = np.ones((n_ch, n), dtype=np.float32)
            offset_out = np.zeros((n_ch, n), dtype=np.float32)

    def drive(u, limit):
        return 20.0 * np.log10(np.maximum(np.max(np.maximum(u / limit[1], -u / limit[0]), axis=1), 1e-300))

    for rows in groups:
        u = np.empty((len(rows), n + m))
        if m:
            u[:, :m] = pre[rows]
        u[:, m:] = samples[rows]
        if switch_gain is not None:
            u *= switch_gain
        flag = np.zeros(u.shape, dtype=np.uint8)
        gain = None
        stuck = None                                     # what the input stage's overload leaves at the ADC input beside the signal
        y = u
        if pa is not None:
            if want:
                peak["preamp"][rows] = drive(u, pa["limit"])
            y, clipped, dead, gain = limit_stage(u, fs, pa["limit"], pa["dead_time"], pa["recovery_tau"], pa["dead_output"],
                                                 pa["scope"] == "board")
            if clipped.any():
                flag |= clipped * np.uint8(FLAG_PREAMP) | dead * np.uint8(FLAG_HELD)
                stuck = True
        if corner is not None:
            y = coupling_highpass(y, fs, corner)
        if stuck is not None:
            # The output less the signal at the recovery gain: the limiter's error while it lasts,
            # then the charge it left on the capacitor.
            stuck = (y - gain * (coupling_highpass(u, fs, corner) if corner is not None else u)) if want and arrays else None
        if ramp is not None:
            y = y * ramp
            if stuck is not None:
                stuck = stuck * ramp
        if adc is not None:
            if want:
                peak["adc"][rows] = drive(y, adc["limit"])
            y, clipped, dead, g = limit_stage(y, fs, adc["limit"], adc["dead_time"], adc["recovery_tau"], adc["dead_output"],
                                              adc["scope"] == "board")
            if clipped.any():
                flag |= clipped * np.uint8(FLAG_ADC) | dead * np.uint8(FLAG_HELD)
                gain = g if gain is None else gain * g
        out[rows] = y[:, m:]
        if want:
            busy = flag != 0
            any_busy = busy.any(axis=1)
            last_index = busy.shape[1] - 1 - np.argmax(busy[:, ::-1], axis=1)
            last_time[rows] = np.where(any_busy, t[last_index], np.nan)
            if n:
                clipped_fraction[rows] = np.mean((flag[:, m:] & (FLAG_PREAMP | FLAG_ADC)) != 0, axis=1)
            if arrays:
                flags[rows] = flag[:, m:]
                if gain is not None:
                    gain_out[rows] = gain[:, m:]
                if stuck is not None:
                    offset_out[rows] = stuck[:, m:]
    if want:
        truth.update(peak_db_re_preamp_limit=peak["preamp"], peak_db_re_adc_limit=peak["adc"],
                     clipped_fraction=clipped_fraction, last_overload_time=last_time,
                     pre_window_samples=int(m), start_time=float(s.start_time), sample_rate=float(fs))
        if arrays:
            truth.update(time=t[m:], flags=flags, gain=gain_out, offset=offset_out)
            if m:
                truth["pre_samples"] = np.asarray(pre, dtype=np.float32)
    return out


def overload_intervals(flags: np.ndarray, time: np.ndarray, merge_gap: float = 0.0, mask: int = 0xFF) -> list:
    """[(t_first, t_last), ...] of the samples of one channel where `flags & mask` is set,
    intervals closer than `merge_gap` seconds joined (a clipped carrier leaves its rail
    twice a cycle: give a period of the lowest frequency)."""
    on = np.nonzero((np.asarray(flags) & mask) != 0)[0]
    if len(on) == 0:
        return []
    dt = float(time[1] - time[0]) if len(time) > 1 else 0.0
    breaks = np.nonzero(np.diff(on) * dt > merge_gap + 1.5 * dt)[0]
    first = np.concatenate([[on[0]], on[breaks + 1]])
    last = np.concatenate([on[breaks], [on[-1]]])
    return [(float(time[a]), float(time[b])) for a, b in zip(first, last)]


def recovery_time(truth: dict, after: float, gain_tolerance_db: float, offset_tolerance: float) -> np.ndarray:
    """Per channel, the time (s after transmit, not before `after`) from which the front end
    stays within tolerance to the end of the window: no stage on its rail or dead, the
    recovery gain within `gain_tolerance_db` of 1, and the offset left at the ADC input below
    `offset_tolerance` (sample units). `truth` from `apply_front_end(..., arrays=True)`.
    `after` itself where a channel is never outside; inf where it has not come back by the end."""
    t = truth["time"]
    bad = (truth["flags"] != 0) | (truth["gain"] < 10.0 ** (-abs(gain_tolerance_db) / 20.0)) | (np.abs(truth["offset"]) > offset_tolerance)
    bad = bad & (t[None, :] >= after)
    out = np.full(bad.shape[0], float(after))
    any_bad = bad.any(axis=1)
    last = bad.shape[1] - 1 - np.argmax(bad[:, ::-1], axis=1)
    out[any_bad] = t[last[any_bad]] + 1.0 / truth["sample_rate"]
    out[any_bad & (last == bad.shape[1] - 1)] = np.inf
    return out


def apply_channel_factors(x: np.ndarray, errors: dict | None) -> np.ndarray:
    """The per-channel gain and phase errors of `apply_channel_errors` on a bare array of real samples."""
    if errors is None or (errors.get("gain") is None and errors.get("phase") is None):
        return x
    n_ch = x.shape[0]
    factor = np.asarray(errors.get("gain", np.ones(n_ch)), dtype=float) * np.exp(1j * np.asarray(errors.get("phase", np.zeros(n_ch)), dtype=float))
    return np.fft.irfft(np.fft.rfft(x, axis=1) * factor[:, None], n=x.shape[1], axis=1)
