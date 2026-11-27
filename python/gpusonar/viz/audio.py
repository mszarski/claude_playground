"""The received signal at each hydrophone as sound (docs/tasks/VISUALIZATION.md sections 4.3 and 14; packages V3, V5).

Every listenable file states how it was made; nothing is changed silently. Three methods (`listen_plan`), each named
in the WAV's file name, in its INFO chunk and in the manifest beside it, with what it misrepresents:

- `as_recorded`: the band is audible (its top at or under `AUDIBLE_TOP_HZ`, 16 kHz: ours, a listener's and a laptop
  speaker's practical limit, not a hearing threshold). Real passband samples are resampled to the smallest standard
  rate of at least 2.2 x the top of the band; complex baseband samples are put back on their own carrier at such a
  rate. Pitch, time, Doppler and the delays between hydrophones are as recorded; what lies above half the output rate
  (noise only, outside the sonar's band) is dropped and said so.
- `heterodyne` (the default for a band above 16 kHz in a record of 0.5 s or more, the owner's choice of 2026-11-27):
  every frequency moves down by
  the same number of hertz, so that the centre of the band sits at `center_hz` (2 kHz). Only a band `2 x (center_hz -
  floor_hz)` wide about the centre is kept (3.6 kHz: nothing below 200 Hz, nothing folds through 0 Hz); of a 20 to
  100 kHz wide FLS or SAS sweep the rest is lost, and with it the sweep's range resolution. Durations are kept; a
  Doppler shift keeps its size in hertz but not its ratio to the carrier; harmonic ratios are lost; the delay between
  hydrophones is kept in the envelope (exactly) and in the phase difference of each frequency, but the carrier's phase
  is that of the shifted frequency, so a delay heard as a phase (below about 1.5 kHz) is not the true delay.
- `time_stretch` (the default for a band above 16 kHz in a record shorter than 0.5 s: every FLS and most SAS records;
  the owner's agreement of 2026-11-27, "oh this includes fls, yes of course that makes sense"; else an option): the
  record is played `factor` times slower. Every frequency is divided by the factor and every duration multiplied by it;
  ratios, the sweep's shape and Doppler as a ratio are kept; the sound lasts `factor` times longer; delays between
  hydrophones are kept exactly, scaled by the factor. The factor (`stretch_factor`, rule ours): asked for by name
  without a factor, the smallest whole number that brings the top of the band under 16 kHz; chosen by `auto`, that
  number raised towards a sound of at least 0.5 s (`STRETCH_MIN_SOUND_S`) as far as keeps the band's centre at or
  above 500 Hz (`STRETCH_MIN_CENTRE_HZ`), so that a 0.27 ms rail record lasts 0.5 s instead of 63 ms.

Levels: one gain per kind of signal (the whole record, echoes alone, one reverberation item, ...) for all the
hydrophones of a ping, so that the levels between hydrophones are preserved; the loudest hydrophone peaks at -1 dBFS.
The gain is written (dB from the record's sample unit to full scale) so that two kinds can be compared. 16-bit PCM WAV
(standard library), mono per hydrophone or a stereo pair of two hydrophones with the same operator on both channels
(the inter-channel delay is the array's physics and is kept sample-exact where the method keeps time, see above).

The FFT resampling treats the record as periodic: a sound at its end can ring into its start (a few output samples).

The functions `hydrophone_audio`, `write_wav` and `read_wav` of the prototype (32-bit float) are kept as they were.
"""

from __future__ import annotations

import base64
import json
import math
import os
import struct

import numpy as np

AUDIBLE_TOP_HZ = 16000.0          # ours: the band is played as recorded when its top is at or under this
HET_CENTER_HZ = 2000.0            # the owner's default centre after a heterodyne (2026-11-27)
HET_FLOOR_HZ = 200.0              # ours: nothing kept below this after a heterodyne
STRETCH_TOP_HZ = 16000.0          # ours: the automatic stretch factor brings the band's top under this
STANDARD_RATES = (8000, 16000, 24000, 32000, 48000)
PEAK_DBFS = -1.0                  # the loudest hydrophone of a kind peaks here
METHOD_TAG = {"as_recorded": "rec", "heterodyne": "het", "time_stretch": "x"}
STRETCH_BELOW_S = 0.5             # the owner (2026-11-27): time stretch by default for records shorter than this
STRETCH_MIN_SOUND_S = 0.5         # ours: `auto` stretches a short record to at least this much sound ...
STRETCH_MIN_CENTRE_HZ = 500.0     # ours: ... as far as keeps the band's centre at or above this


def default_method(sonar_band, duration: float) -> str:
    """The method `auto` takes (the owner's rules of 2026-11-27): as recorded when the band's top is at or under 16 kHz;
    time stretch for a record shorter than 0.5 s; heterodyne otherwise (a long record above the audible band)."""
    hi = max(float(v) for v in sonar_band)
    if hi <= AUDIBLE_TOP_HZ:
        return "as_recorded"
    return "time_stretch" if float(duration) < STRETCH_BELOW_S else "heterodyne"


def stretch_factor(sonar_band, duration: float, auto: bool) -> tuple[float, str]:
    """(factor, the rule in words) of a time stretch with no factor given (rule ours, module docstring)."""
    lo, hi = sorted(float(v) for v in sonar_band)
    k_pitch = max(1, math.ceil(max(hi, 1.0) / STRETCH_TOP_HZ))
    rule = f"the smallest whole factor bringing the band's top ({hi / 1e3:.4g} kHz) under {STRETCH_TOP_HZ / 1e3:g} kHz: {k_pitch}"
    if not auto:
        return float(k_pitch), rule
    k_dur = math.ceil(STRETCH_MIN_SOUND_S / max(float(duration), 1e-12))
    k_low = max(1, math.floor(0.5 * (lo + hi) / STRETCH_MIN_CENTRE_HZ))
    k = max(k_pitch, min(k_dur, k_low))
    if k > k_pitch:
        rule += (f"; raised to {k} for at least {STRETCH_MIN_SOUND_S:g} s of sound (a {float(duration) * 1e3:.3g} ms record"
                 + (f", the band's centre kept at or above {STRETCH_MIN_CENTRE_HZ:g} Hz)" if k == k_low < k_dur else ")"))
    return float(k), rule


# --- the plan: which method, what is kept and lost ----------------------------------------------------------------

def record_band(sample_kind: str, sample_rate: float, center_frequency: float) -> tuple[float, float]:
    """The band the samples can hold: [fc - fs/2, fc + fs/2] for complex baseband, [0, fs/2] for real passband."""
    if "omplex" in str(sample_kind):
        return max(center_frequency - 0.5 * sample_rate, 0.0), center_frequency + 0.5 * sample_rate
    return 0.0, 0.5 * sample_rate


def _std_rate(at_least: float) -> int:
    for r in STANDARD_RATES:
        if r >= at_least:
            return r
    return int(math.ceil(at_least / 1000.0) * 1000)


def listen_plan(sonar_band, sample_kind: str, sample_rate: float, center_frequency: float, duration: float,
                method: str = "auto", center_hz: float = HET_CENTER_HZ, floor_hz: float = HET_FLOOR_HZ,
                factor: float | None = None) -> dict:
    """How one ping's samples are made audible: the method and every number it uses, with what it keeps, loses and
    misrepresents. `sonar_band` [lo, hi] Hz (the transmit band); `duration` s of the record."""
    lo, hi = sorted(float(v) for v in sonar_band)
    r_lo, r_hi = record_band(sample_kind, sample_rate, center_frequency)
    chosen = method == "auto"
    if chosen:
        method = default_method([lo, hi], duration)
    plan = {"method": method, "chosen_by": "auto (default_method)" if chosen else "asked for",
            "sonar_band_hz": [lo, hi], "record_band_hz": [r_lo, r_hi], "sample_kind": str(sample_kind),
            "record_rate_hz": float(sample_rate), "record_center_hz": float(center_frequency),
            "record_duration_s": float(duration), "audible_top_hz": AUDIBLE_TOP_HZ}
    if method == "as_recorded":
        top = hi if "omplex" not in str(sample_kind) else r_hi
        rate = _std_rate(2.2 * top)
        kept = [r_lo, min(r_hi, 0.5 * rate)]
        plan.update(rate_hz=rate, shift_hz=0.0, factor=1.0, kept_hz=kept, kept_audio_hz=kept,
                    audio_duration_s=duration, time_scale=1.0,
                    misrepresents="nothing in pitch, time, Doppler or the delay between hydrophones; only what lies "
                                  f"above {0.5 * rate / 1e3:g} kHz is dropped (outside the sonar's band)")
    elif method == "heterodyne":
        c = 0.5 * (lo + hi)
        width = min(hi - lo, 2.0 * (center_hz - floor_hz))
        # The shift is a whole number of the record's frequency bins (1/duration): the operator is then exact.
        shift = round((c - center_hz) * duration) / duration
        kept = [c - 0.5 * width, c + 0.5 * width]
        audio = [kept[0] - shift, kept[1] - shift]
        rate = _std_rate(2.5 * audio[1])
        plan.update(rate_hz=rate, shift_hz=shift, factor=1.0, kept_hz=kept, kept_audio_hz=audio, center_hz=center_hz,
                    audio_duration_s=duration, time_scale=1.0,
                    misrepresents=f"pitch: every frequency {shift / 1e3:.6g} kHz lower (Doppler keeps its size in Hz, "
                                  "not its ratio; harmonic ratios are lost); only "
                                  f"{width / 1e3:.3g} kHz of the {(hi - lo) / 1e3:.4g} kHz band is kept"
                                  + (" (the sweep's range resolution is lost)" if width < hi - lo else "")
                                  + "; time and the envelope delay between hydrophones are kept, the carrier phase is "
                                    "the shifted frequency's")
    elif method == "time_stretch":
        if factor:
            k, rule = float(factor), "given"
        else:
            k, rule = stretch_factor([lo, hi], duration, auto=chosen)
        plan["factor_rule"] = rule
        kept = [r_lo, r_hi]
        audio = [r_lo / k, r_hi / k]
        rate = _std_rate(min(2.2 * audio[1], 48000.0))
        if 0.5 * rate < audio[1]:
            kept[1] = 0.5 * rate * k
            audio[1] = 0.5 * rate
        plan.update(rate_hz=rate, shift_hz=0.0, factor=k, kept_hz=kept, kept_audio_hz=audio,
                    audio_duration_s=duration * k, time_scale=k,
                    misrepresents=f"time and pitch: {k:g} times slower and {k:g} times lower (ratios, the sweep's "
                                  "shape and Doppler as a ratio are kept); delays between hydrophones scaled by "
                                  f"{k:g}, exactly")
    else:
        raise ValueError(f"audio method {method!r}: auto, as_recorded, heterodyne or time_stretch")
    plan["warning"] = None
    if method == "heterodyne" and duration * (plan["kept_hz"][1] - plan["kept_hz"][0]) < 8.0:
        # ours: fewer than 8 frequency bins of 1 / duration in the kept band
        plan["warning"] = (f"the record lasts {duration * 1e3:.3g} ms: the kept band holds "
                           f"{duration * (plan['kept_hz'][1] - plan['kept_hz'][0]):.2g} frequency bins, so the sound is a "
                           "click or silence; time_stretch keeps the whole band and lengthens it")
    plan["lost_hz"] = [b for b in ([r_lo, plan["kept_hz"][0]], [plan["kept_hz"][1], r_hi]) if b[1] - b[0] > 1e-6]
    plan["tag"] = METHOD_TAG[method] + (f"{plan['factor']:g}" if method == "time_stretch" else
                                        f"{center_hz:g}" if method == "heterodyne" else "")
    plan["scene_to_audio"] = f"audio time = (time after transmit - record start) x {plan['time_scale']:g}"
    # A complex record's carrier is moved to a whole output bin: every frequency is then off by this much (< half a
    # bin of 1 / duration; zero for a real record).
    off = 0.0
    if "omplex" in str(sample_kind) and duration > 0:
        k = plan["factor"]
        g = (center_frequency - plan["shift_hz"]) / k
        off = (round(g * duration * k) - g * duration * k) / (duration * k)
    plan["bin_offset_hz"] = off
    return plan


# --- the operator ----------------------------------------------------------------------------------------------------

def to_audio(samples, plan: dict) -> np.ndarray:
    """[C, M] float audio of [C, N] (or [N]) samples by `plan` (real passband or complex baseband on the plan's record
    centre). One linear operator for every channel: the analytic spectrum's bins at absolute frequency F are moved to
    F (as recorded), F - shift (heterodyne) or F / factor (time stretch); bins outside the kept band are dropped."""
    x = np.asarray(samples)
    one = x.ndim == 1
    x = np.atleast_2d(x)
    n = x.shape[1]
    fs = plan["record_rate_hz"]
    T = n / fs
    rate = plan["rate_hz"]
    k = plan["factor"]
    m = int(round(T * k * rate))
    if np.iscomplexobj(x):
        X = np.fft.fft(x.astype(np.complex128), axis=1)
        F = np.fft.fftfreq(n, 1.0 / fs) + plan["record_center_hz"]
    else:
        X = np.fft.fft(x.astype(np.float64), axis=1)
        F = np.fft.fftfreq(n, 1.0 / fs)
        X = X * np.where(F > 0, 2.0, 0.0)[None, :]
    G = (F - plan["shift_hz"]) / k
    lo, hi = plan["kept_hz"]
    keep = (F > 0) & (F >= lo - 1e-9) & (F <= hi + 1e-9) & (G > 0) & (G < 0.5 * rate)
    if plan["method"] == "heterodyne":
        keep &= G >= plan["kept_audio_hz"][0] - 1e-9
    idx = np.round(G[keep] * T * k).astype(int)            # output bin: G / (output bin width 1 / (T k))
    Y = np.zeros((x.shape[0], m), dtype=np.complex128)
    Y[:, idx] = X[:, keep]
    y = np.real(np.fft.ifft(Y, axis=1)) * (m / n)
    return y[0] if one else y


def gain_for(signals, peak_dbfs: float = PEAK_DBFS) -> float:
    """One gain (linear, record unit to full scale 1.0) for a list of [C, M] audio arrays: the loudest sample of all
    of them at `peak_dbfs`."""
    peak = max((float(np.max(np.abs(s))) for s in signals if np.size(s)), default=0.0)
    return 10.0 ** (peak_dbfs / 20.0) / peak if peak > 0 else 1.0


# --- WAV: 16-bit PCM with an INFO chunk ------------------------------------------------------------------------------

def _info_chunk(info: dict) -> bytes:
    body = b"INFO"
    for key, text in info.items():
        raw = str(text).encode("utf-8") + b"\x00"
        if len(raw) % 2:
            raw += b"\x00"
        body += key.encode("ascii")[:4].ljust(4, b" ") + struct.pack("<I", len(raw)) + raw
    return b"LIST" + struct.pack("<I", len(body)) + body


def write_pcm16(path: str, rate: int, y, gain: float, info: dict | None = None) -> dict:
    """A 16-bit PCM WAV of `y` ([M] mono, or [2, M] a stereo pair) times `gain` (full scale 1.0), with a LIST/INFO
    chunk (INAM title, ICMT the method in words, ISFT the writer). Returns {bytes, clipped, peak_dbfs}."""
    y = np.atleast_2d(np.asarray(y, dtype=float)) * gain
    clipped = int(np.sum(np.abs(y) > 1.0))
    q = np.clip(np.rint(y * 32767.0), -32768, 32767).astype("<i2")
    ch = q.shape[0]
    data = q.T.reshape(-1).tobytes()                       # interleaved: sample-exact pairs
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, ch, int(rate), int(rate) * 2 * ch, 2 * ch, 16)
    extra = _info_chunk(info) if info else b""
    body = b"WAVE" + fmt + extra + b"data" + struct.pack("<I", len(data)) + data
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return {"bytes": 8 + len(body), "clipped": clipped, "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-12)), 2)}


def read_pcm16(path: str) -> dict:
    """{rate, channels, samples [C, M] float (full scale 1), info {key: text}} of a WAV `write_pcm16` wrote."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    i, out, info = 12, {}, {}
    while i + 8 <= len(raw):
        cid, size = raw[i:i + 4], struct.unpack("<I", raw[i + 4:i + 8])[0]
        body = raw[i + 8:i + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            out.update(format=tag, channels=ch, rate=rate, bits=bits)
        elif cid == b"LIST" and body[:4] == b"INFO":
            j = 4
            while j + 8 <= len(body):
                k, n = body[j:j + 4].decode("ascii"), struct.unpack("<I", body[j + 4:j + 8])[0]
                info[k.strip()] = body[j + 8:j + 8 + n].rstrip(b"\x00").decode("utf-8")
                j += 8 + n + (n % 2)
        elif cid == b"data":
            q = np.frombuffer(body, dtype="<i2").reshape(-1, out["channels"]).T
            out["samples"] = q.astype(float) / 32767.0
        i += 8 + size + (size % 2)
    out["info"] = info
    return out


# --- the parts of a ping (the simulator's taps) ----------------------------------------------------------------------

def _boundary_name(info: dict | None, surface_z: float | None = 0.0) -> str:
    if not info:
        return "item"
    if info.get("boundary"):
        return str(info["boundary"])
    if info.get("model") == "volume":
        return "volume"
    z = info.get("boundary_z")
    if z is not None and surface_z is not None and abs(float(z) - float(surface_z)) < 1e-6:
        return "surface"
    return "bottom" if z is not None else str(info.get("model"))


def decompose(stages: dict, meta: list | None = None, record=None) -> dict:
    """{name: (samples [C, N], what)} of one ping from the simulator's taps (taps.read_sim_taps; with
    sim_capture(per_item=True) the reverberation by item). The parts are differences of successive taps, so that their
    sum is the samples after the noise stage (before the hardware gain and the ADC) to float rounding; the record's
    time-varying gain, when it has one and no overload model, is applied to each part (a linear gain); the ADC's
    rounding and an overload front end are not (said in `what`). Names: echoes, basin_tail, reverberation_<i>_<kind>,
    reverberation (all items when not split), sources, direct, reference_and_errors, noise."""
    info = {m["stage"]: m.get("info") for m in (meta or [])}
    order = list(stages.keys())
    f64 = lambda a: np.asarray(a).astype(np.complex128 if np.iscomplexobj(a) else np.float64)
    parts = {}
    if "echoes" not in stages:
        return parts
    parts["echoes"] = (f64(stages["echoes"]), "echoes alone (the simulator's tap after synthesis)")
    prev = f64(stages["echoes"])
    items = [s for s in order if s.startswith("reverberation_item_")]
    direct = f64(stages["direct_arrival"]) if "direct_arrival" in stages else None
    if "reverberation_start" in stages:
        start = f64(stages["reverberation_start"])
        if np.any(start != prev):
            parts["basin_tail"] = (start - prev, "the basin's late tail (sim/basin.py)")
        prev = start
        for s in items:
            x = f64(stages[s])
            it = info.get(s) or {}
            name = f"reverberation_{it.get('index', items.index(s))}_{_boundary_name(it)}"
            parts[name] = (x - prev, f"reverberation item {it.get('index')}: {it.get('model')} ({_boundary_name(it)})")
            prev = x
    if "reverberation" in stages:
        x = f64(stages["reverberation"])
        rest = x - prev - (direct if direct is not None else 0.0)
        if "reverberation_start" not in stages:
            parts["reverberation"] = (rest, "reverberation, all items together (and one-way sources, a basin tail, "
                                            "if any)")
        elif np.any(np.abs(rest) > 0):
            parts["sources"] = (rest, "one-way sources (sim/sources.py)")
        if direct is not None:
            parts["direct"] = (direct, "the projector heard directly (transmit.direct_path)")
        prev = x
    if "reference_and_clocks" in stages:
        x = f64(stages["reference_and_clocks"])
        d = x - prev
        if np.any(np.abs(d) > 0):
            parts["reference_and_errors"] = (d, "the reference signal and the channel errors (not sound in the water)")
        prev = x
    if "noise" in stages:
        x = f64(stages["noise"])
        parts["noise"] = (x - prev, "ambient noise")
        prev = x
    if record is not None:
        md = record.metadata
        if len(md.tvg_time):
            s = record.samples
            t = s.start_time + np.arange(s.n_samples) / s.sample_rate
            g = 10.0 ** (np.interp(t, md.tvg_time, md.tvg_gain_db) / 20.0)
            parts = {k: (v * g[None, :], w + "; the record's gain ramp applied") for k, (v, w) in parts.items()}
    return parts


def residual_db(parts: dict, whole) -> float:
    """20 log10 of the rms of (whole - sum of the parts) over the rms of the whole: the ADC's rounding (and an
    overload front end) is what the parts do not hold."""
    total = sum(v for v, _ in parts.values())
    w = np.asarray(whole).astype(complex if np.iscomplexobj(whole) else float)
    r = np.sqrt(np.mean(np.abs(w - total) ** 2))
    return float(20.0 * np.log10(max(r, 1e-30) / max(np.sqrt(np.mean(np.abs(w) ** 2)), 1e-30)))


PARTS_MISMATCH_DB = -40.0         # ours: parts that miss the record by more than this (of its rms) are said not to sum


def front_end_note(residual_db: float | None, adc: bool, overload: bool) -> dict:
    """What the page says about the parts and the front end (docs/tasks/VISUALIZATION.md section 14, V6 part A): the
    parts are the simulator's taps before the front end (the ADC's rounding; an overload front end's clipping and
    recovery), and where they miss the record by more than PARTS_MISMATCH_DB of its rms the page says they do not sum
    to it, and why, instead of showing a silent mismatch."""
    where = "before the front end: before " + ("the overload front end's clipping and recovery and " if overload else "") + (
        "the ADC's rounding" if adc else "any ADC (this record has none)")
    mismatch = residual_db is not None and residual_db > PARTS_MISMATCH_DB
    text = f"Parts are the simulator's taps {where}."
    if residual_db is not None:
        text += f" The parts' sum differs from the record by {residual_db:.1f} dB of the record's rms."
    if mismatch:
        text += (" They do NOT sum to the record on this page: " + (
            "the front end clipped (receive.overload); the whole record is what the front end delivered, the parts are "
            "what reached it" if overload else "the ADC's step is near the record's level (its rounding is not in the parts)"))
    return {"adc": bool(adc), "overload": bool(overload), "residual_db": residual_db, "mismatch": bool(mismatch),
            "statement": text}


# --- what the page shows beside the sound -------------------------------------------------------------------------

def _u8(db, floor=-60.0):
    return np.clip(np.rint((np.asarray(db) - floor) / -floor * 255.0), 0, 255).astype(np.uint8)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def envelope_db(env: np.ndarray, n_out: int = 600, ref: float | None = None):
    """Max-pooled envelope [C, N] to [C, n_out] in dB re `ref` (default its peak), as uint8 over -60..0 dB."""
    env = np.atleast_2d(np.asarray(env, float))
    n = env.shape[1]
    step = max(1, int(math.ceil(n / n_out)))
    m = int(math.ceil(n / step))
    pad = np.zeros((env.shape[0], m * step))
    pad[:, :n] = env
    pooled = pad.reshape(env.shape[0], m, step).max(axis=2)
    ref = ref if ref is not None else max(float(pooled.max()), 1e-30)
    return _u8(20.0 * np.log10(np.maximum(pooled / ref, 1e-6))), step


def spectrogram_power(y: np.ndarray, rate: float, f_max: float, n_time: int = 240, n_freq: int = 96):
    """(power [n_freq, T], seconds per column, hertz per row, time of the first column's centre) of a mono audio
    signal up to `f_max`: Hann windows of one FFT size (frequency step at most f_max / n_freq), at most `n_time`
    columns, each row the largest bin it covers."""
    y = np.asarray(y, float)
    nfft = int(min(max(64, 1 << int(math.ceil(math.log2(rate * n_freq / max(f_max, 1.0))))), 1 << 14))
    if len(y) < nfft:
        y = np.pad(y, (0, nfft - len(y)))
    n = len(y)
    hop = max(nfft // 4, int(math.ceil((n - nfft) / max(n_time - 1, 1))), 1)
    starts = np.arange(0, n - nfft + 1, hop)
    w = np.hanning(nfft)
    S = np.abs(np.fft.rfft(np.stack([y[i:i + nfft] * w for i in starts]), axis=1)).T ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / rate)
    edges = np.linspace(0.0, f_max, n_freq + 1)
    rows = np.zeros((n_freq, S.shape[1]))
    band = np.clip(np.searchsorted(edges, f, side="right") - 1, -1, n_freq)
    for r in range(n_freq):
        sel = band == r
        if np.any(sel):
            rows[r] = S[sel].max(axis=0)
    return rows, hop / rate, f_max / n_freq, (nfft / 2) / rate


# --- one ping's sound: the files and the manifest a page reads ---------------------------------------------------------

def manifest_js(manifest: dict) -> str:
    return "window.__AUDIO = " + json.dumps(manifest, separators=(",", ":")) + ";\n"


def write_ping_audio(out_dir: str, name: str, record, trace: dict, parts: dict | None = None,
                     arrivals: list | None = None, method: str = "auto", factor: float | None = None,
                     parts_channels: list | None = None, stereo: tuple | None = None,
                     center_hz: float = HET_CENTER_HZ, whole_channels: list | None = None) -> dict:
    """Write one ping's sound beside its page, under `out_dir`/`name`.audio/: the whole record for every hydrophone
    (`h<C>.whole.<tag>.wav`), the parts (`decompose`) and the single arrivals (`arrivals`: [{pair, samples [C, N]}])
    for the hydrophones in `parts_channels` (default the first and the last), a stereo pair of two hydrophones
    (default the same two), per hydrophone a data file `h<C>.js` (matched-filter envelope and spectrogram of each
    kind) and `manifest.json` / `manifest.js` (the method, the numbers, the files, the mapping between scene time and
    audio time). Returns the manifest."""
    from .page2d import compressed, _with_samples
    s = record.samples
    x = np.asarray(s.array)
    n_ch, n = x.shape
    plan = listen_plan(trace["sonar"]["band_hz"], str(s.kind).split(".")[-1], float(s.sample_rate),
                       float(s.center_frequency), n / float(s.sample_rate), method=method, factor=factor,
                       center_hz=center_hz)
    adir = os.path.join(out_dir, f"{name}.audio")
    os.makedirs(adir, exist_ok=True)
    if parts_channels is None:
        parts_channels = sorted({0, n_ch - 1})
    parts_channels = [c for c in parts_channels if 0 <= c < n_ch]
    whole_channels = list(range(n_ch)) if whole_channels is None else [c for c in whole_channels if 0 <= c < n_ch]
    kinds = {"whole": (x, "the record as stored (after the hardware gain and the ADC)")}
    for k, (v, what) in (parts or {}).items():
        if k == "reference_and_errors" or not np.any(np.abs(v) > 0):
            continue
        kinds[k] = (v, what)
    arr = []
    for i, a in enumerate(arrivals or []):
        kinds[f"arrival_{i}"] = (np.asarray(a["samples"]), a["what"])
        arr.append({k: a[k] for k in a if k != "samples"} | {"kind": f"arrival_{i}"})
    t0 = float(s.start_time)
    files, gains, peaks = [], {}, {}
    data = {c: {"env": {}, "spec": {}} for c in range(n_ch)}
    arrival_gain = None
    def full(v):
        """[C, N] samples of a kind given for every channel or for `parts_channels` only (the rest zero)."""
        v = np.asarray(v)
        if v.shape[0] == n_ch:
            return v
        f = np.zeros((n_ch, v.shape[1]), dtype=v.dtype)
        f[parts_channels] = v
        return f

    for kind, (v, what) in kinds.items():
        chans = whole_channels if kind == "whole" else parts_channels
        v = full(v)
        y = to_audio(v[chans], plan)
        if kind.startswith("arrival_"):
            if arrival_gain is None:      # one gain for all the arrivals (their levels compare)
                arrival_gain = gain_for([to_audio(full(a["samples"])[parts_channels], plan) for a in arrivals])
            g = arrival_gain
        else:
            g = gain_for([y])
        gains[kind] = round(20.0 * math.log10(g), 3)
        t_env, env = compressed(_with_samples(record, v))
        env_ref = max(float(env[chans].max()), 1e-30)          # one reference for the kind: levels between hydrophones kept
        # the spectrogram shows the sonar's band as heard (with a margin), not the whole output band
        f_top = min(0.5 * plan["rate_hz"], plan["kept_audio_hz"][1] * 1.25,
                    1.3 * (plan["sonar_band_hz"][1] - plan["shift_hz"]) / plan["factor"])
        specs = [spectrogram_power(y[j], plan["rate_hz"], f_top) for j in range(len(chans))]
        spec_ref = max(max(float(sp[0].max()) for sp in specs), 1e-30)
        for j, c in enumerate(chans):
            fn = f"h{c:03d}.{kind}.{plan['tag']}.wav"
            info = {"INAM": f"{name} hydrophone {c}: {kind}",
                    "ICMT": f"{plan['method']}: {_method_words(plan)}; gain {gains[kind]:.2f} dB re record unit, one gain "
                            f"for every hydrophone of this kind (levels preserved); {what}",
                    "ISFT": "gpusonar viz/audio.py"}
            w = write_pcm16(os.path.join(adir, fn), plan["rate_hz"], y[j], g, info)
            files.append({"channel": c, "kind": kind, "file": fn, "bytes": w["bytes"], "clipped": w["clipped"],
                          "peak_dbfs": w["peak_dbfs"]})
            e8, step = envelope_db(env[c], ref=env_ref)
            data[c]["env"][kind] = {"u8": b64(e8[0]), "dt": step / float(s.sample_rate), "t0": float(t_env[0])}
            rows, dt, df, tc = specs[j]
            sp = _u8(10.0 * np.log10(np.maximum(rows / spec_ref, 1e-6)))
            data[c]["spec"][kind] = {"u8": b64(sp.reshape(-1)), "rows": int(sp.shape[0]), "cols": int(sp.shape[1]),
                                     "dt": dt, "df": df, "t0": tc}
    if stereo is None and n_ch > 1 and len(whole_channels) > 1:
        stereo = (parts_channels[0], parts_channels[-1]) if len(parts_channels) > 1 else (0, n_ch - 1)
    stereo_file = None
    if stereo is not None and n_ch > 1 and len(whole_channels) > 1:
        a, b = stereo
        y = to_audio(x[[a, b]], plan)
        fn = f"stereo_h{a:03d}_h{b:03d}.whole.{plan['tag']}.wav"
        g = 10.0 ** (gains["whole"] / 20.0)
        w = write_pcm16(os.path.join(adir, fn), plan["rate_hz"], y, g,
                        {"INAM": f"{name} hydrophones {a} (left) and {b} (right)",
                         "ICMT": f"{plan['method']}: {_method_words(plan)}; the same operator on both channels, the "
                                 "delay between them is the array's", "ISFT": "gpusonar viz/audio.py"})
        stereo_file = {"channels": [a, b], "file": fn, "bytes": w["bytes"]}
    for c in sorted(set(whole_channels) | set(parts_channels)):
        with open(os.path.join(adir, f"h{c:03d}.js"), "w", encoding="utf-8") as f:
            f.write(f"(window.__AUDIO_DATA = window.__AUDIO_DATA || {{}})[{c}] = " + json.dumps(data[c], separators=(",", ":")) + ";\n")
    manifest = {
        "kind": "gpusonar.scene_audio", "version": 1, "scenario": name, "dir": f"{name}.audio",
        "plan": plan, "method_words": _method_words(plan),
        "record_start_s": t0, "record_duration_s": n / float(s.sample_rate),
        "levels": "one gain per kind for all the hydrophones of this ping: levels between hydrophones are preserved; "
                  "the loudest hydrophone of each kind peaks at -1 dBFS; gains are dB from the record's sample unit to "
                  "full scale, so kinds compare by their gains",
        "gains_db": gains, "kinds": {k: w for k, (_, w) in kinds.items()}, "parts_channels": parts_channels,
        "n_channels": n_ch, "files": files, "stereo": stereo_file, "arrivals": arr,
        "time": {"scene_to_audio": "audio_s = (scene_s - record_start_s) * time_scale",
                 "time_scale": plan["time_scale"], "record_start_s": t0},
        "bytes": sum(f["bytes"] for f in files) + (stereo_file["bytes"] if stereo_file else 0),
    }
    with open(os.path.join(adir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(adir, "manifest.js"), "w", encoding="utf-8") as f:
        f.write(manifest_js(manifest))
    return manifest


def _method_words(plan: dict) -> str:
    m = plan["method"]
    kept = f"{plan['kept_hz'][0] / 1e3:.4g}-{plan['kept_hz'][1] / 1e3:.4g} kHz"
    lost = ", ".join(f"{a / 1e3:.4g}-{b / 1e3:.4g} kHz" for a, b in plan["lost_hz"]) or "nothing"
    head = {"as_recorded": "as recorded",
            "heterodyne": f"heterodyne: shifted down {plan['shift_hz'] / 1e3:.6g} kHz (band centre to {plan.get('center_hz', 0) / 1e3:g} kHz)",
            "time_stretch": f"time stretch x{plan['factor']:g} ({plan['factor']:g} times slower and lower; factor: "
                            f"{plan.get('factor_rule', 'given')})"}[m]
    return (("WARNING: " + plan["warning"] + ". " if plan.get("warning") else "")
            + f"{head}; {plan['rate_hz'] / 1e3:g} kHz output; kept {kept}, lost {lost}; "
            f"{plan['audio_duration_s']:.4g} s of sound; misrepresents: {plan['misrepresents']}")


# --- the prototype's functions (kept) --------------------------------------------------------------------------------

def _baseband(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """Complex baseband of a real passband signal about fc (FFT: the negative half dropped)."""
    n = len(x)
    X = np.fft.fft(x)
    f = np.fft.fftfreq(n, 1.0 / fs)
    X[f < 0] = 0.0
    a = np.fft.ifft(2.0 * X)
    return a * np.exp(-2j * np.pi * fc * np.arange(n) / fs)


def hydrophone_audio(record, channel: int = 0, mode: str = "auto", audio_center: float = 2000.0,
                     audio_bandwidth: float | None = None, audible_top: float = 20000.0):
    """The prototype's form (V1): (sample rate Hz, samples float [N], what was done) of one channel of a record as
    sound. `auto`: `as_is` for a real audible band, `remodulate` for an audible baseband record, `shift` otherwise.
    Superseded by `listen_plan` / `to_audio` / `write_ping_audio` (16-bit PCM, every hydrophone, stated levels)."""
    s = record.samples
    x = np.asarray(s.array)[channel]
    fs, fc = float(s.sample_rate), float(s.center_frequency)
    if mode == "auto":
        top = fc + 0.5 * fs if s.is_complex() else None
        if not s.is_complex():
            mode = "as_is" if fc < audible_top * 0.8 else "shift"
        else:
            mode = "remodulate" if top is not None and top <= audible_top else "shift"
    if mode == "as_is":
        if s.is_complex():
            raise ValueError("as_is needs real passband samples; use remodulate or shift")
        return fs, x.astype(float), {"mode": "as_is", "sample_rate": fs}
    a = x.astype(complex) if s.is_complex() else _baseband(x.astype(float), fs, fc)
    if mode == "shift" and audio_bandwidth is None:
        audio_bandwidth = min(fs, 1.8 * float(audio_center))
    if audio_bandwidth is not None:
        A = np.fft.fft(a)
        A[np.abs(np.fft.fftfreq(len(a), 1.0 / fs)) > 0.5 * audio_bandwidth] = 0.0
        a = np.fft.ifft(A)
    carrier = fc if mode == "remodulate" else float(audio_center)
    half = 0.5 * (audio_bandwidth or fs)
    out_fs = float(int(np.ceil(2.5 * (carrier + half) / 1000.0)) * 1000)
    n = len(a)
    m = int(round(n * out_fs / fs))
    A = np.fft.fft(a)
    f = np.fft.fftfreq(n, 1.0 / fs)
    B = np.zeros(m, dtype=complex)
    keep = np.abs(f) < 0.5 * min(fs, out_fs)
    B[np.round(f[keep] * m / out_fs).astype(int) % m] = A[keep]
    b = np.fft.ifft(B) * (m / n)
    y = np.real(b * np.exp(2j * np.pi * carrier * np.arange(m) / out_fs))
    return out_fs, y, {"mode": mode, "carrier_hz": carrier, "from_center_hz": fc, "sample_rate": out_fs,
                       "kept_bandwidth_hz": audio_bandwidth or fs, "shift_hz": (fc - carrier) if mode == "shift" else 0.0}


def write_wav(path: str, fs: float, y: np.ndarray, normalise: bool = True) -> dict:
    """A mono 32-bit float WAV (format 3) as taps.export_wav writes them. Returns {sample_rate, n, scale}."""
    y = np.asarray(y, dtype=float)
    scale = 1.0 / max(float(np.abs(y).max()), 1e-30) if normalise else 1.0
    data = (y * scale).astype("<f4").tobytes()
    rate = int(round(fs))
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 3, 1, rate, rate * 4, 4, 32,
                         b"data", len(data))
    with open(path, "wb") as f:
        f.write(header + data)
    return {"sample_rate": rate, "n": len(y), "scale": scale}


def read_wav(path: str):
    """(sample rate, samples) of a WAV written by `write_wav` (for the tests)."""
    with open(path, "rb") as f:
        raw = f.read()
    fields = struct.unpack("<4sI4s4sIHHIIHH4sI", raw[:44])
    return fields[7], np.frombuffer(raw[44:], dtype="<f4")
