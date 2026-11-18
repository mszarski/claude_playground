"""Passive tonal analysis of the flank array: Goertzel lines and DEMON.

Two things a listening array can say about a source besides its bearing:

- narrowband lines (machinery tonals): a small bank of Goertzel filters around each
  frequency asked for, on the array steered at a bearing (or on the hydrophones). For each
  line and block of data: frequency (interpolated between bins), amplitude, phase, signal
  to noise against the bins either side, the bearing the line is strongest at, and a
  track of the frequency from block to block (Doppler).
- DEMON, detection of envelope modulation on noise: the broadband noise of a propeller is
  modulated at the shaft and blade rates. Per bearing: the beam restricted to a band,
  its envelope, decimated, the envelope's spectrum averaged over time and normalised by
  its local background; lines above a threshold set from a false-alarm rate; and from the
  lines the shaft rate, the blade rate and the number of blades when the harmonics allow.
  The normalised spectra over bearing are the display product ("DEMONgram").

`TonalAnalyzer.push(record)` takes stretches of data as they arrive (records of any
length, raw real or complex baseband) and returns the results completed by each: cutting
one recording into pieces gives the same results as passing it whole. This module is the
NumPy reference of libs/live `TonalAnalyzer` (core.TonalAnalyzerCpp, and `tonal=1` of
build/bin/sonar_live --passive / --listen); tests/python/test_tonal.py holds one against
the other.

Formulation read in, re-implemented here (no code copied):
- Goertzel filter: arlpy `arlpy.signal.goertzel` (org-arl, BSD 3-clause) and
  SignalAnalysis.jl `goertzel` (org-arl, MIT): the second-order recurrence
  s[n] = x[n] + 2 cos(w) s[n-1] - s[n-2], output s[N-1] - exp(-jw) s[N-2].
- DEMON: SignalAnalysis.jl `demon` (org-arl, MIT): the Hilbert envelope reduced to its
  RMS over each decimation block (its default method `:rms`), then a high-pass at 1 Hz.
  Here the band-pass, the envelope and the decimation are one step in the frequency
  domain (the power of the band's bins in each short block, which is the mean square of
  the band-limited analytic signal under the block's window), because that is where the
  array is steered; the high-pass is the lower limit of the lines looked for (`f_min_hz`).
Everything after that (windowed-bin interpolation, noise estimate, background
normalisation, threshold, harmonic grouping, tracking) is derived in the comments below.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULTS = {
    "bearings_deg": [-80.0, 80.0, 1.0],   # first, last, step: the beams lines and modulation are looked for on
    "window": "hann",                     # aperture weighting of each row of hydrophones ("rect": none)
    "row_tolerance_m": 0.02,              # hydrophones within this of one line are one row (as chain.elevation)
    "attach_deg": 3.0,                    # a result within this of a passive bearing is reported as that bearing's
    "goertzel": None,                     # {lines: [...], ...}: see GOERTZEL
    "demon": None,                        # {} or {...}: see DEMON
}

GOERTZEL = {
    "lines": [],                          # Hz, absolute: the frequencies to watch (required)
    "on": "beams",                        # "beams": the array steered over bearings_deg; "hydrophones": channel powers added
    "block_s": 1.0,                       # s of data per estimate: the bins are 1 / block_s apart
    "hop_s": None,                        # s between estimates (default: block_s, no overlap)
    "search_bins": 4,                     # the line is looked for this many bins either side of where it is expected
    "guard_bins": 3,                      # bins left out between the search span and the noise bins
    "noise_bins": 8,                      # bins each side the noise is estimated from
    "min_snr_db": 10.0,                   # a line is detected (and tracked) above this
    "track": {"alpha": 0.5, "beta": 0.1,  # alpha-beta filter on the line's frequency, block to block
              "max_misses": 3,            # blocks without a detection before the search returns to the nominal frequency
              "max_shift_hz": 0.0},       # a track further than this from nominal is dropped (0: no limit)
}

DEMON = {
    "band": None,                         # [f_low, f_high] Hz absolute; default: the passive band, else the transmit band
    "envelope": "square",                 # "square": band power (square law); "hilbert": its root (RMS of the analytic signal)
    "envelope_rate_hz": 100.0,            # sample rate of the envelope: modulation up to half of this is seen
    "window_hops": 2,                     # each envelope sample is from a block this many envelope samples long
    "segment_s": 8.0,                     # s of envelope per spectrum: the modulation bins are 1 / segment_s apart
    "segment_overlap": 0.5,               # fraction of a segment shared with the next
    "n_average": 8,                       # spectra averaged (the most recent ones)
    "min_segments": 4,                    # no result before this many spectra are in the average
    "f_min_hz": 1.0,                      # modulation lines are looked for from here ...
    "f_max_hz": None,                     # ... to here (default: 0.8 of half the envelope rate)
    "background_bins": 16,                # bins each side the background of a bin is estimated from
    "guard_bins": 2,                      # bins left out either side of the bin itself
    "clip": 3.0,                          # bins this far above the first-pass background are left out of the second
    "pfa": 1e-6,                          # false-alarm probability per bearing and modulation bin: sets the threshold
    "max_lines": 12,                      # lines kept per source, strongest first
    "max_sources": 4,                     # sources (groups of lines by bearing) reported
    "min_separation_deg": 4.0,            # lines closer in bearing than this are one source
    "shaft_hz": [0.5, 30.0],              # shaft rates considered
    "max_harmonic": 12,                   # highest harmonic of the shaft rate a line may be
    "harmonic_tolerance_bins": 1.0,       # a line this far (in modulation bins) from a harmonic still belongs to it
    "score_fraction": 0.9,                # the highest shaft rate explaining this fraction of the best score is taken
    "blades": [2, 9],                     # blade counts considered
    "gram": False,                        # put the normalised spectra over bearing (dB) in every result
}


def _rint(x: float) -> int:
    """Nearest integer, halves up (the same in the C++ analyzer)."""
    return int(math.floor(x + 0.5))


def _merge(base: dict, over: dict | None) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise ValueError(f"tonal: unknown key {k!r}")
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def configure(config: dict | None) -> dict:
    """`passive.tonal` merged over the defaults. Unknown keys are errors."""
    cfg = _merge(DEFAULTS, config)
    if cfg["goertzel"] is not None:
        cfg["goertzel"] = _merge(GOERTZEL, cfg["goertzel"])
        if not cfg["goertzel"]["lines"]:
            raise ValueError("tonal.goertzel.lines: give the frequencies to watch")
        if cfg["goertzel"]["on"] not in ("beams", "hydrophones"):
            raise ValueError("tonal.goertzel.on: beams or hydrophones")
        if cfg["goertzel"]["guard_bins"] < 1 or cfg["goertzel"]["noise_bins"] < 1 or cfg["goertzel"]["search_bins"] < 0:
            raise ValueError("tonal.goertzel: guard_bins and noise_bins must be at least 1")
    if cfg["demon"] is not None:
        cfg["demon"] = _merge(DEMON, cfg["demon"])
        if cfg["demon"]["envelope"] not in ("square", "hilbert"):
            raise ValueError("tonal.demon.envelope: square or hilbert")
    if cfg["window"] not in ("hann", "rect"):
        raise ValueError("tonal.window: hann or rect")
    return cfg


def cpp_options(config: dict | None, f_low: float = 0.0, f_high: float = 0.0) -> dict:
    """`passive.tonal` as the options of the C++ analyzer (core.TonalAnalyzerCpp) and the
    `key=value` arguments of build/bin/sonar_live: {"tonal": "1", "tonal_goertzel_lines": "300,450", ...}."""
    cfg = configure(config)
    b = cfg["bearings_deg"]
    o = {"tonal": 1, "tonal_bearing_first_deg": b[0], "tonal_bearing_last_deg": b[1], "tonal_bearing_step_deg": b[2],
         "tonal_window": cfg["window"], "tonal_row_tolerance_m": cfg["row_tolerance_m"], "tonal_attach_deg": cfg["attach_deg"]}
    g = cfg["goertzel"]
    if g is not None:
        t = g["track"] or {}
        o.update(tonal_goertzel_lines=",".join(repr(float(f)) for f in g["lines"]), tonal_goertzel_on=g["on"],
                 tonal_goertzel_block_s=g["block_s"], tonal_goertzel_hop_s=g["hop_s"] or g["block_s"],
                 tonal_goertzel_search_bins=int(g["search_bins"]), tonal_goertzel_guard_bins=int(g["guard_bins"]),
                 tonal_goertzel_noise_bins=int(g["noise_bins"]), tonal_goertzel_min_snr_db=g["min_snr_db"],
                 tonal_goertzel_track=int(bool(g["track"])))
        if g["track"]:
            o.update(tonal_goertzel_alpha=t["alpha"], tonal_goertzel_beta=t["beta"], tonal_goertzel_max_misses=int(t["max_misses"]),
                     tonal_goertzel_max_shift_hz=t["max_shift_hz"])
    d = cfg["demon"]
    if d is not None:
        band = d["band"] or [f_low, f_high]
        o.update(tonal_demon=1, tonal_demon_f_low=band[0], tonal_demon_f_high=band[1], tonal_demon_envelope=d["envelope"],
                 tonal_demon_envelope_rate_hz=d["envelope_rate_hz"], tonal_demon_window_hops=int(d["window_hops"]),
                 tonal_demon_segment_s=d["segment_s"], tonal_demon_segment_overlap=d["segment_overlap"],
                 tonal_demon_n_average=int(d["n_average"]), tonal_demon_min_segments=int(d["min_segments"]),
                 tonal_demon_f_min_hz=d["f_min_hz"], tonal_demon_f_max_hz=d["f_max_hz"] or 0.0,
                 tonal_demon_background_bins=int(d["background_bins"]), tonal_demon_guard_bins=int(d["guard_bins"]),
                 tonal_demon_clip=d["clip"], tonal_demon_pfa=d["pfa"], tonal_demon_max_lines=int(d["max_lines"]),
                 tonal_demon_max_sources=int(d["max_sources"]), tonal_demon_min_separation_deg=d["min_separation_deg"],
                 tonal_demon_shaft_min_hz=d["shaft_hz"][0], tonal_demon_shaft_max_hz=d["shaft_hz"][1],
                 tonal_demon_max_harmonic=int(d["max_harmonic"]), tonal_demon_harmonic_tolerance_bins=d["harmonic_tolerance_bins"],
                 tonal_demon_score_fraction=d["score_fraction"], tonal_demon_blades_min=int(d["blades"][0]),
                 tonal_demon_blades_max=int(d["blades"][1]), tonal_demon_gram=int(bool(d["gram"])))
    return {k: (v if isinstance(v, str) else repr(v)) for k, v in o.items()}


# --- Goertzel ---------------------------------------------------------------------------------

def goertzel(x, frequency: float, fs: float) -> np.ndarray:
    """The Fourier sum of `x` (last axis; real or complex) at one frequency, by the Goertzel
    recurrence: X = sum_n x[n] exp(-2 pi j f n / fs), phase referred to the first sample.
    `frequency` need not be a multiple of fs / N. For N samples a real cosine of amplitude A
    on the frequency gives |X| = A N / 2, a complex exponential A N.

    s[n] = x[n] + 2 cos(w) s[n-1] - s[n-2];  y = s[N-1] - exp(-jw) s[N-2] = exp(jw(N-1)) X.
    (arlpy / SignalAnalysis.jl run one more step on a zero sample and return exp(jwN) X.)"""
    x = np.asarray(x)
    w = 2.0 * np.pi * frequency / fs
    coeff = 2.0 * np.cos(w)
    s1 = np.zeros(x.shape[:-1], dtype=complex if np.iscomplexobj(x) else float)
    s2 = np.zeros_like(s1)
    for n in range(x.shape[-1]):
        s1, s2 = x[..., n] + coeff * s1 - s2, s1
    return (s1 - np.exp(-1j * w) * s2) * np.exp(-1j * w * (x.shape[-1] - 1))


def fourier_sums(x, frequencies, fs: float, chunk: int = 4096) -> np.ndarray:
    """The same sums as `goertzel` at several frequencies, [..., F], by direct products
    (what the reference analyzer uses: one pass of NumPy instead of a loop over samples)."""
    x = np.asarray(x)
    f = np.asarray(frequencies, dtype=float) / fs
    out = np.zeros(x.shape[:-1] + (len(f),), dtype=complex)
    for i in range(0, x.shape[-1], chunk):
        n = np.arange(i, min(i + chunk, x.shape[-1]))
        out += x[..., n] @ np.exp(-2j * np.pi * np.mod(np.outer(n, f), 1.0))
    return out


def hann_offset(ratio: float) -> float:
    """Where a line sits between two bins of a Hann-weighted transform, from the ratio of the
    smaller of the two magnitudes to the larger: 0 on the larger bin, 0.5 half way.

    Derivation: the Hann window's transform, in bins x off the line, is
    W(x) = sinc(x) / (1 - x^2) (N large). A line d bins above bin k gives |H[k]| = W(d),
    |H[k+1]| = W(1 - d), whose ratio is (1 + d) / (2 - d); solved for d."""
    return float(np.clip((2.0 * ratio - 1.0) / (ratio + 1.0), 0.0, 1.0))


def hann_gain(offset: float) -> float:
    """W(x) above: the Hann-weighted bin's response to a line `offset` bins away (1 on the bin)."""
    if abs(1.0 - offset * offset) < 1e-9:
        return 0.5
    return float(np.sinc(offset) / (1.0 - offset * offset))


def crb_frequency(snr: float, n: int, fs: float, real: bool = False) -> float:
    """Cramer-Rao bound on the standard deviation (Hz) of the frequency of one tone in white
    noise from `n` samples. `snr` is the tone's mean square over the noise variance per sample.

    Derivation (complex tone A exp(j(wn + p)) in complex noise of variance s2): the Fisher
    information of (w, p) is (2 A^2 / s2) [[sum n^2, sum n], [sum n, N]], so
    var(w) = 6 s2 / (A^2 N (N^2 - 1)). A real cosine in real noise has half the information per
    unit of A^2 / s2 and A^2 / 2 mean square: the same expression in terms of `snr`, times 2."""
    var_w = (12.0 if real else 6.0) / (snr * n * (n * n - 1.0))
    return float(np.sqrt(var_w) * fs / (2.0 * np.pi))


# --- pieces shared with the C++ analyzer -------------------------------------------------------

def gamma_tail(n: int, x: float) -> float:
    """P(G > x) for G a sum of n unit exponentials: exp(-x) sum_{k<n} x^k / k!."""
    term = total = math.exp(-x)
    for k in range(1, n):
        term *= x / k
        total += term
    return total


def ratio_tail(n: int, m: float, t: float) -> float:
    """P(X / Y > t) for X the mean of n unit exponentials and Y, independent of it, a Gamma variable
    of shape m and mean 1 (the mean of m of them; m need not be whole). Averaging the Gamma tail
    exp(-n t y) sum_{k<n} (n t y)^k / k! over Y gives, with a = n t / m,
    sum_{k<n} a^k / k! x m (m + 1) ... (m + k - 1) x (1 + a)^-(m + k)."""
    a = n * t / m
    term = total = (1.0 + a) ** (-m)
    for k in range(1, n):
        term *= a * (m + k - 1) / (k * (1.0 + a))
        total += term
    return total


def demon_threshold(pfa: float, n_spectra: int, correlation: float, background: int = 0) -> tuple[float, float]:
    """Ratio to the background a bin of the averaged envelope spectrum exceeds with probability
    `pfa` when there is only noise, and the number of independent spectra it stands for.

    A bin of one spectrum of noise is exponentially distributed; the mean of n independent ones
    is a Gamma variable of shape n. Overlapping segments are not independent: with squared
    correlation r between neighbouring spectra the variance of the mean of K is
    (1 + 2 r (K - 1) / K) / K, so they stand for n = K / (1 + 2 r (K - 1) / K) independent ones
    (spectra further apart share nothing for the overlaps used). The background is the mean of
    `background` neighbouring bins. Under the Hann window bins next to each other correlate by
    (2/3)^2 and next but one by (1/6)^2 (the window's square has Fourier coefficients 3/8, -1/4,
    1/16), so they count as background / (1 + 8/9 + 1/18) independent ones, and the mean is a Gamma
    variable of shape n x background x 18/35: the threshold is on the ratio of the two
    (`ratio_tail`; `background` 0: the background taken as known). n is seldom whole: the threshold
    is interpolated between the whole numbers either side. Measured on noise (four 60 s records,
    32 hydrophones): 1.0e-3 to 1.5e-3 of the cells above the threshold for 1e-3 asked for, 0.4e-4 to
    2.1e-4 for 1e-4; at 1e-6 three false lines in all, where 0.1 to 0.5 would be expected."""
    k = max(int(n_spectra), 1)
    n = max(k / (1.0 + 2.0 * correlation * (k - 1) / k), 1.0)

    def solve(n_whole):
        lo, hi = 1.0, 1e3
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            tail = ratio_tail(n_whole, n_whole * background * 18.0 / 35.0, mid) if background > 0 else gamma_tail(n_whole, n_whole * mid)
            lo, hi = (mid, hi) if tail > pfa else (lo, mid)
        return 0.5 * (lo + hi)

    below = int(math.floor(n))
    t = solve(below)
    if n > below:
        t += (n - below) * (solve(below + 1) - t)
    return t, n


def split_mean(s: np.ndarray, guard: int, width: int, lo: int, hi: int) -> np.ndarray:
    """Mean of s[j] (first axis) over the bins j with guard < |j - k| <= guard + width that lie in
    [lo, hi], for every k; 0 where there are none."""
    n = s.shape[0]
    cs = np.concatenate([np.zeros((1,) + s.shape[1:]), np.cumsum(s, axis=0)])
    out = np.zeros_like(s, dtype=float)
    for k in range(n):
        a0, a1 = max(k - guard - width, lo), min(k - guard - 1, hi)
        b0, b1 = max(k + guard + 1, lo), min(k + guard + width, hi)
        count, total = 0, 0.0
        if a1 >= a0:
            total, count = total + cs[a1 + 1] - cs[a0], count + a1 - a0 + 1
        if b1 >= b0:
            total, count = total + cs[b1 + 1] - cs[b0], count + b1 - b0 + 1
        if count:
            out[k] = total / count
    return out


def harmonic_set(freqs, weights, df: float, cfg: dict) -> dict:
    """Shaft rate, blade rate and blade count from the modulation lines of one source.

    Every line over every harmonic number is a candidate shaft rate. A candidate's score is the
    summed weight (dB above the background) of the lines within tolerance of one of its
    harmonics. Half of any rate explains the same lines, so among the candidates within
    `score_fraction` of the best score the highest rate is taken; it is then refitted to its
    lines by least squares. The blade rate is taken to be the strongest line of the set (the
    usual appearance of a propeller: the blade-rate line stands highest); its harmonic number is
    the number of blades. With the strongest line the fundamental itself the blade count is not
    known, and the "shaft" rate may be the blade rate of a propeller whose shaft lines are in
    the noise. `confidence` = fraction of the lines' weight the set explains x (1 - 2^-(n - 1))
    for n lines in the set: 0 for a single line; a score for ranking, not a probability."""
    f = np.asarray(freqs, dtype=float)
    w = np.asarray(weights, dtype=float)
    out = {"shaft_hz": None, "blade_hz": None, "blades": None, "confidence": 0.0, "harmonics": [0] * len(f)}
    if len(f) == 0:
        return out
    tol = cfg["harmonic_tolerance_bins"] * df
    lo, hi = cfg["shaft_hz"]
    max_h = int(cfg["max_harmonic"])

    def match(f0, k):
        h = np.floor(f / f0 + 0.5)
        ok = (h >= 1) & (h <= max_h) & (np.abs(f - h * f0) <= np.minimum(tol * np.maximum(1.0, h / k), 0.25 * f0))
        return np.where(ok, h, 0).astype(int)

    best = None                                   # (score, f0, k)
    cands = []
    for i in range(len(f)):
        for k in range(1, max_h + 1):
            f0 = f[i] / k
            if lo <= f0 <= hi:
                score = float(w[match(f0, k) > 0].sum())
                cands.append((score, f0, k))
                if best is None or score > best[0]:
                    best = (score, f0, k)
    if best is None:
        return out
    pick = None
    for score, f0, k in cands:
        if score >= cfg["score_fraction"] * best[0] and (pick is None or f0 > pick[1]):
            pick = (score, f0, k)
    h = match(pick[1], pick[2])
    in_set = h > 0
    f0 = float((w[in_set] * h[in_set] * f[in_set]).sum() / (w[in_set] * h[in_set] ** 2).sum())
    n = int(in_set.sum())
    out["harmonics"] = [int(v) for v in h]
    if n < 2:
        return out                               # one line is a line, not a rate
    out["shaft_hz"] = f0
    out["confidence"] = float(w[in_set].sum() / w.sum() * (1.0 - 0.5 ** (n - 1)))
    strongest = int(h[int(np.argmax(np.where(in_set, w, -np.inf)))])
    if cfg["blades"][0] <= strongest <= cfg["blades"][1] and strongest >= 2:
        out["blades"] = strongest
        out["blade_hz"] = strongest * f0
    return out


class _Line:
    def __init__(self, nominal):
        self.nominal = float(nominal)
        self.frequency, self.rate, self.tracking, self.misses = float(nominal), 0.0, False, 0


class TonalAnalyzer:
    """`passive.tonal` on a stream of records. `push(record)` returns the results completed by
    that record's samples: dicts {"kind": "goertzel", "time_ns", "lines": [...]} and
    {"kind": "demon", "time_ns", "segments", "threshold_db", "sources": [...]}; `time_ns` is the
    time of the last sample used. Records that follow one another in time (transmit.time_ns +
    samples.start_time, to within half a sample) are one stream; after a gap the partly filled
    blocks are dropped and the averages and tracks carry on.

    A Goertzel line: nominal_hz; detected (signal to noise at least `min_snr_db`); frequency_hz
    (interpolated); amplitude and level_db = 20 log10(amplitude), in the samples' units: the
    cosine's amplitude at a hydrophone for real samples, the magnitude for complex baseband
    (the bin's noise is in it: subtract 10^(noise_db / 10) from amplitude^2 at low signal to
    noise); phase_rad of the tone at the middle of the block; snr_db = (bin power - noise) /
    noise; noise_db, the mean power of the noise bins; bearing_deg where the line is strongest
    (None on hydrophones); track_hz and rate_hz_s, the tracked frequency and its rate of
    change (None while not tracking).
    A DEMON source: bearing_deg; shaft_hz, blade_hz, blades (None where the lines do not say);
    confidence; lines, each frequency_hz, snr_db (dB above the background), modulation (the
    line's index in the envelope: for the square-law envelope twice the amplitude's, less the
    share of the beam that is not the source) and harmonic (of the shaft rate; 0: not one)."""

    def __init__(self, config: dict | None = None, f_low: float = 0.0, f_high: float = 0.0):
        self.cfg = configure(config)
        self._band = (float(f_low), float(f_high))
        b = self.cfg["bearings_deg"]
        self.bearings = np.radians(np.arange(b[0], b[1] + 0.5 * b[2], b[2]))
        g = self.cfg["goertzel"]
        self._lines = [_Line(f) for f in g["lines"]] if g else []
        self._key = None
        self._next_ns = None
        self.gram = None                       # the latest DEMONgram: {"bearings_deg", "frequencies_hz", "db"}

    # -- set-up from the first record -------------------------------------------------------

    def _setup(self, record):
        from . import elevation
        s = record.samples
        cfg = self.cfg
        self.fs = float(s.sample_rate)
        self.is_complex = bool(s.is_complex())
        self.carrier = float(s.center_frequency) if self.is_complex else 0.0
        self.c = float(record.environment.sound_speed)
        rows = elevation.find_rows(record, cfg["row_tolerance_m"])
        if not rows:
            rows = [list(range(s.n_channels))]
        pos = record.rx_positions()
        used = [i for r in rows for i in r]
        axis = pos[rows[0][-1]] - pos[rows[0][0]]
        axis = axis / np.linalg.norm(axis)
        self.rows = rows
        self.u = (pos - pos[used].mean(axis=0)) @ axis            # along the rows, m about the centre
        self.weights = np.zeros(s.n_channels)                     # each row's weights sum to 1
        for r in rows:
            w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(len(r)) + 0.5) / len(r)) if cfg["window"] == "hann" else np.ones(len(r))
            self.weights[r] = w / w.sum()
        self._buf = np.zeros((s.n_channels, 0), dtype=np.complex64 if self.is_complex else np.float32)
        self._first = 0                                           # stream index of the buffer's first sample
        self._run_ns = 0
        g = cfg["goertzel"]
        if g:
            self.g_n = max(_rint(g["block_s"] * self.fs), 8)
            self.g_hop = max(_rint((g["hop_s"] or g["block_s"]) * self.fs), 1)
            self.g_next = 0
            half = 0.5 * self.fs
            for line in self._lines:
                fb = line.nominal - self.carrier
                if not (0.0 < fb < half if not self.is_complex else abs(fb) < half):
                    raise ValueError(f"tonal.goertzel.lines: {line.nominal} Hz is outside the sampled band")
        d = cfg["demon"]
        if d:
            w = record.transmit.waveform
            band = d["band"] or (self._band if self._band[1] > 0.0 else (min(w.f_start, w.f_stop), max(w.f_start, w.f_stop)))
            self.d_hop = max(_rint(self.fs / d["envelope_rate_hz"]), 1)
            self.d_n = self.d_hop * int(d["window_hops"])
            self.d_rate = self.fs / self.d_hop
            self.d_next = 0
            k = np.arange(self.d_n)
            freqs = (self.carrier + np.where(k <= self.d_n // 2, k, k - self.d_n) * self.fs / self.d_n if self.is_complex
                     else np.fft.rfftfreq(self.d_n, 1.0 / self.fs))
            self.d_bins = np.nonzero((freqs >= band[0]) & (freqs <= band[1]) & (freqs > 0.0))[0]
            if len(self.d_bins) == 0:
                raise ValueError("tonal.demon: no bin of the envelope blocks inside the band asked for")
            self.d_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * (k + 0.5) / self.d_n)
            # [bearing, channel, bin]: weights and the phase that brings a plane wave from each bearing into line
            self.d_steer = self.weights[None, :, None] * np.exp(
                -2j * np.pi * freqs[self.d_bins][None, None, :] * self.u[None, :, None] * np.sin(self.bearings)[:, None, None] / self.c)
            self.s_n = max(_rint(d["segment_s"] * self.d_rate), 16)
            self.s_hop = max(_rint(self.s_n * (1.0 - d["segment_overlap"])), 1)
            self.s_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(self.s_n) + 0.5) / self.s_n)
            lag = self.s_window[self.s_hop:] @ self.s_window[:self.s_n - self.s_hop] if self.s_hop < self.s_n else 0.0
            self.s_correlation = float((lag / (self.s_window @ self.s_window)) ** 2)
            self.s_df = self.d_rate / self.s_n
            f_max = d["f_max_hz"] or 0.8 * 0.5 * self.d_rate
            self.k_lo = max(int(np.ceil(d["f_min_hz"] / self.s_df - 1e-9)), 2)
            self.k_hi = min(int(np.floor(f_max / self.s_df + 1e-9)), self.s_n // 2 - 1)
            self._env = np.zeros((0, len(self.bearings)))
            self._spectra = []

    # -- the stream ---------------------------------------------------------------------------

    def push(self, record) -> list[dict]:
        s = record.samples
        key = (float(s.sample_rate), int(s.n_channels), bool(s.is_complex()), float(s.center_frequency),
               float(record.environment.sound_speed))
        start_ns = int(record.transmit.time_ns) + _rint(s.start_time * 1e9)
        if key != self._key:
            self._key = key
            self._setup(record)
            self._run_ns = start_ns
        elif abs(start_ns - self._next_ns) > 0.5e9 / self.fs:
            # A gap: what is half filled is dropped; averages and tracks carry on.
            self._buf = self._buf[:, :0]
            self._first = 0
            self._run_ns = start_ns
            if self.cfg["goertzel"]:
                self.g_next = 0
            if self.cfg["demon"]:
                self.d_next = 0
                self._env = self._env[:0]
        x = np.asarray(s.array)
        self._buf = np.concatenate([self._buf, x.astype(self._buf.dtype)], axis=1)
        end = self._first + self._buf.shape[1]
        self._next_ns = self._run_ns + _rint(end / self.fs * 1e9)
        events = []
        keep = end
        if self.cfg["goertzel"]:
            while self.g_next + self.g_n <= end:
                i = self.g_next - self._first
                events.append(self._goertzel_block(self._buf[:, i:i + self.g_n], self.g_next + self.g_n))
                self.g_next += self.g_hop
            keep = min(keep, self.g_next)
        if self.cfg["demon"]:
            n_blocks = (end - self.d_next - self.d_n) // self.d_hop + 1 if end - self.d_next >= self.d_n else 0
            if n_blocks > 0:
                events.extend(self._demon_blocks(n_blocks))
            keep = min(keep, self.d_next)
        if keep > self._first:
            self._buf = self._buf[:, keep - self._first:]
            self._first = keep
        events.sort(key=lambda e: e["time_ns"])
        return events

    def _time_ns(self, index: int) -> int:
        return self._run_ns + _rint(index / self.fs * 1e9)

    # -- Goertzel -----------------------------------------------------------------------------

    def _beam_power(self, h, f_abs, sin_bearings):
        """Mean over the rows of |row steered at each bearing|^2, [B, M], and the first row's beam."""
        sb = np.asarray(sin_bearings, dtype=float)
        power, first = 0.0, None
        for r in self.rows:
            steer = np.exp(-2j * np.pi * f_abs[None, None, :] * self.u[r][None, :, None] * sb[:, None, None] / self.c)
            beam = (self.weights[r][None, :, None] * h[r][None, :, :] * steer).sum(axis=1)     # [B, M]
            power = power + np.abs(beam) ** 2
            if first is None:
                first = beam
        return power / len(self.rows), first

    def _goertzel_block(self, x, end_index: int) -> dict:
        g = self.cfg["goertzel"]
        n, fs = self.g_n, self.fs
        search, guard, noise = int(g["search_bins"]), int(g["guard_bins"]), int(g["noise_bins"])
        m_half = search + guard + noise
        m = np.arange(-m_half - 1, m_half + 2)
        dt = self.g_hop / fs
        tr = g["track"]
        # A cosine of amplitude A (real samples) or an exponential of magnitude A (complex) reads A.
        scale = (1.0 if self.is_complex else 2.0) / (0.5 * n)
        lines = []
        for line in self._lines:
            centre = line.frequency + line.rate * dt if (tr and line.tracking) else line.nominal
            f_abs = centre + m * fs / n
            rect = fourier_sums(x, f_abs - self.carrier, fs) * scale                  # [C, 2M + 3]
            # A Hann window is the rectangular one's bin less half its two neighbours.
            h = 0.5 * rect[:, 1:-1] - 0.25 * (rect[:, :-2] + rect[:, 2:])              # [C, 2M + 1]
            f_abs = f_abs[1:-1]
            lo, hi = m_half - search, m_half + search + 1
            bearing = None
            if g["on"] == "beams":
                power, _ = self._beam_power(h, f_abs, np.sin(self.bearings))
                ib, im = np.unravel_index(int(np.argmax(power[:, lo:hi])), power[:, lo:hi].shape)
                col = power[:, lo + im]
                frac = 0.0
                if 0 < ib < len(col) - 1:
                    den = col[ib - 1] - 2.0 * col[ib] + col[ib + 1]
                    frac = float(np.clip(0.5 * (col[ib - 1] - col[ib + 1]) / den, -0.5, 0.5)) if den < 0.0 else 0.0
                bearing = float(self.bearings[ib] + frac * (self.bearings[1] - self.bearings[0])) if len(self.bearings) > 1 else float(self.bearings[ib])
                power, first = self._beam_power(h, f_abs, [np.sin(bearing)])
                p, first = power[0], first[0]
            else:
                p = (np.abs(h[[i for r in self.rows for i in r]]) ** 2).mean(axis=0)
                first = h[self.rows[0][0]]
            k = lo + int(np.argmax(p[lo:hi]))
            side = 1 if p[k + 1] >= p[k - 1] else -1
            delta = hann_offset(float(np.sqrt(p[k + side] / p[k]))) if p[k] > 0.0 else 0.0
            delta = min(delta, 0.5)
            frequency = float(f_abs[k] + side * delta * fs / n)
            signal = float(p[k]) / hann_gain(delta) ** 2
            floor = float(np.concatenate([p[:noise], p[-noise:]]).mean())
            snr = max(signal / floor - 1.0, 1e-6) if floor > 0.0 else 1e-6
            snr_db = float(10.0 * np.log10(snr))
            detected = snr_db >= g["min_snr_db"]
            if tr:
                if detected:
                    if line.tracking:
                        residual = frequency - centre
                        line.frequency = centre + tr["alpha"] * residual
                        line.rate += tr["beta"] * residual / dt
                    else:
                        line.frequency, line.rate, line.tracking = frequency, 0.0, True
                    line.misses = 0
                elif line.tracking:
                    line.frequency = centre
                    line.misses += 1
                    if line.misses > tr["max_misses"]:
                        line.tracking, line.rate, line.frequency = False, 0.0, line.nominal
                if line.tracking and tr["max_shift_hz"] > 0.0 and abs(line.frequency - line.nominal) > tr["max_shift_hz"]:
                    line.tracking, line.rate, line.frequency = False, 0.0, line.nominal
            amplitude = float(np.sqrt(signal))
            lines.append({"nominal_hz": line.nominal, "detected": bool(detected), "frequency_hz": frequency,
                          "amplitude": amplitude, "level_db": float(20.0 * np.log10(max(amplitude, 1e-300))),
                          # of the tone at the middle of the block, where it does not depend on the frequency
                          # estimate (first row's beam, or the first hydrophone): the bin's phase is referred to
                          # the block's first sample, the Hann window is centred n / 2 samples later
                          "phase_rad": float(np.angle(first[k] * np.exp(1j * np.pi * np.mod((f_abs[k] - self.carrier) * n / fs, 2.0)))),
                          "snr_db": snr_db, "noise_db": float(10.0 * np.log10(max(floor, 1e-300))),
                          "bearing_deg": None if bearing is None else float(np.degrees(bearing)),
                          "track_hz": float(line.frequency) if (tr and line.tracking) else None,
                          "rate_hz_s": float(line.rate) if (tr and line.tracking) else None})
        return {"kind": "goertzel", "time_ns": self._time_ns(end_index), "block_s": n / fs, "lines": lines}

    # -- DEMON --------------------------------------------------------------------------------

    def _demon_blocks(self, n_blocks: int) -> list[dict]:
        d = self.cfg["demon"]
        events = []
        batch = 64
        for b0 in range(0, n_blocks, batch):
            nb = min(batch, n_blocks - b0)
            i0 = self.d_next - self._first
            idx = i0 + self.d_hop * np.arange(nb)[:, None] + np.arange(self.d_n)[None, :]        # [K, n]
            seg = self._buf[:, idx] * self.d_window                                                 # [C, K, n]
            spec = (np.fft.fft(seg, axis=2) if self.is_complex else np.fft.rfft(seg, axis=2))[:, :, self.d_bins]
            env = np.zeros((nb, len(self.bearings)))
            for r in self.rows:
                beam = np.einsum("bcf,ckf->bkf", self.d_steer[:, r, :], spec[r])
                env += (np.abs(beam) ** 2).sum(axis=2).T
            if d["envelope"] == "hilbert":
                env = np.sqrt(env)
            first_block = self.d_next
            self.d_next += nb * self.d_hop
            self._env = np.concatenate([self._env, env])
            # Envelope sample j of this batch ends at stream index first_block + j * hop + n.
            done = len(self._env) - nb
            while len(self._env) >= self.s_n:
                seg_e = self._env[:self.s_n]
                last = self.s_n - 1 - done                 # index, within this batch, of the segment's last envelope sample
                mean = seg_e.mean(axis=0)
                y = np.where(mean > 0.0, seg_e / np.where(mean > 0.0, mean, 1.0) - 1.0, 0.0) * self.s_window[:, None]
                # Scaled so that a modulation 1 + m cos(...) of the envelope reads m^2 on its bin.
                per = np.abs(np.fft.rfft(y, axis=0)) ** 2 * (2.0 / self.s_window.sum()) ** 2
                self._spectra.append(per)
                if len(self._spectra) > int(d["n_average"]):
                    self._spectra.pop(0)
                self._env = self._env[self.s_hop:]
                done -= self.s_hop
                if len(self._spectra) >= int(d["min_segments"]):
                    events.append(self._demon_result(first_block + last * self.d_hop + self.d_n))
        return events

    def _demon_result(self, end_index: int) -> dict:
        d = self.cfg["demon"]
        s = np.mean(self._spectra, axis=0)                                   # [F, B]
        n_f = s.shape[0]
        top = n_f - 1
        first = split_mean(s, int(d["guard_bins"]), int(d["background_bins"]), self.k_lo, top)
        clipped = np.where(s > d["clip"] * first, first, s)
        bg = split_mean(clipped, int(d["guard_bins"]), int(d["background_bins"]), self.k_lo, top)
        z = np.where(bg > 0.0, s / np.where(bg > 0.0, bg, 1.0), 0.0)
        threshold, n_eff = demon_threshold(d["pfa"], len(self._spectra), self.s_correlation, 2 * int(d["background_bins"]))
        freqs = np.arange(n_f) * self.s_df
        step = self.bearings[1] - self.bearings[0] if len(self.bearings) > 1 else 1.0
        sep = int(np.floor(np.radians(d["min_separation_deg"]) / step + 1e-9)) if len(self.bearings) > 1 else 0
        self.gram = {"bearings_deg": np.degrees(self.bearings), "frequencies_hz": freqs[self.k_lo:self.k_hi + 1],
                     "db": 10.0 * np.log10(np.maximum(z[self.k_lo:self.k_hi + 1].T, 1e-12)), "threshold_db": 10.0 * np.log10(threshold)}
        found = []
        n_b = len(self.bearings)
        for k in range(self.k_lo, self.k_hi + 1):
            for b in range(n_b):
                v = z[k, b]
                if v < threshold:
                    continue
                # A line is the largest value within a bin in frequency and `min_separation_deg` in bearing.
                is_peak = True
                for kk in (k - 1, k, k + 1):
                    for bb in range(max(b - sep, 0), min(b + sep, n_b - 1) + 1):
                        if (kk, bb) == (k, b):
                            continue
                        other = z[kk, bb]
                        if other > v or (other == v and (kk, bb) < (k, b)):
                            is_peak = False
                if not is_peak:
                    continue
                excess = np.maximum(s[k - 1:k + 2, b] - bg[k - 1:k + 2, b], 0.0)
                side = 1 if excess[2] >= excess[0] else -1
                delta = min(hann_offset(float(np.sqrt(excess[1 + side] / excess[1]))), 0.5) if excess[1] > 0.0 else 0.0
                frac = 0.0
                if 0 < b < n_b - 1:
                    den = z[k, b - 1] - 2.0 * v + z[k, b + 1]
                    frac = float(np.clip(0.5 * (z[k, b - 1] - z[k, b + 1]) / den, -0.5, 0.5)) if den < 0.0 else 0.0
                found.append({"frequency_hz": float((k + side * delta) * self.s_df), "snr_db": float(10.0 * np.log10(v)),
                              "modulation": float(np.sqrt(excess[1]) / hann_gain(delta)),
                              "bearing": float(self.bearings[b] + frac * step)})
        found.sort(key=lambda q: -q["snr_db"])
        groups = []                                                           # lines by bearing, strongest first
        for q in found:
            for grp in groups:
                if abs(q["bearing"] - grp[0]["bearing"]) < np.radians(d["min_separation_deg"]):
                    if len(grp) < int(d["max_lines"]):
                        grp.append(q)
                    break
            else:
                if len(groups) < int(d["max_sources"]):
                    groups.append([q])
        sources = []
        for grp in groups:
            w = np.array([q["snr_db"] for q in grp])
            hs = harmonic_set([q["frequency_hz"] for q in grp], w, self.s_df, d)
            sources.append({"bearing_deg": float(np.degrees((w * [q["bearing"] for q in grp]).sum() / w.sum())),
                            "shaft_hz": hs["shaft_hz"], "blade_hz": hs["blade_hz"], "blades": hs["blades"],
                            "confidence": hs["confidence"],
                            "lines": [{"frequency_hz": q["frequency_hz"], "snr_db": q["snr_db"], "modulation": q["modulation"],
                                       "harmonic": h} for q, h in zip(grp, hs["harmonics"])]})
        out = {"kind": "demon", "time_ns": self._time_ns(end_index), "segments": len(self._spectra), "independent": n_eff,
               "threshold_db": float(10.0 * np.log10(threshold)), "sources": sources}
        if d["gram"]:
            out["gram"] = {"bearings_deg": [float(v) for v in self.gram["bearings_deg"]],
                           "frequencies_hz": [float(v) for v in self.gram["frequencies_hz"]],
                           "db": [[round(float(v), 1) for v in row] for row in self.gram["db"]]}
        return out


# --- with the passive bearings -----------------------------------------------------------------

def attach(frames: list[dict], events: list[dict], record_time_ns: int, frame_end_s, attach_deg: float) -> list[dict]:
    """Put each result into the passive frame during which it was completed, as that frame's
    "tonal": {"goertzel": [...], "demon": [...]}, every line and source with "peak": the index of
    the frame's bearing it lies within `attach_deg` of (the nearest), or -1. `frames` are report
    lines of the passive chain ({"bearings_deg": [...], ...}); `frame_end_s[i]` is the end of frame
    i in seconds after `record_time_ns`. Results later than the last frame go to the last."""
    if not frames:
        return frames
    for ev in events:
        t = (ev["time_ns"] - record_time_ns) * 1e-9
        i = next((j for j, end in enumerate(frame_end_s) if t <= end + 1e-9), len(frames) - 1)
        fr = frames[i]
        peaks = fr.get("bearings_deg", [])

        def peak(bearing_deg):
            if bearing_deg is None or not peaks:
                return -1
            j = int(np.argmin(np.abs(np.asarray(peaks) - bearing_deg)))
            return j if abs(peaks[j] - bearing_deg) <= attach_deg else -1

        body = {k: v for k, v in ev.items() if k != "kind"}
        if ev["kind"] == "goertzel":
            body["lines"] = [dict(q, peak=peak(q["bearing_deg"])) for q in ev["lines"] if q["detected"]]
        else:
            body["sources"] = [dict(q, peak=peak(q["bearing_deg"])) for q in ev["sources"]]
        fr.setdefault("tonal", {}).setdefault(ev["kind"], []).append(body)
    return frames


def features(source: dict | None = None, lines: list[dict] | None = None) -> dict:
    """What a classifier can use of a bearing's tonal results, as plain numbers (NaN: not known):
    from a DEMON source its shaft and blade rates, blade count, confidence, number of modulation
    lines and the strongest one's level; from Goertzel lines the number detected, the strongest's
    signal to noise and each line's frequency offset from nominal (Doppler). Nothing in the chain
    uses them by default: the live classifier's decisions are unchanged."""
    nan = float("nan")
    out = {"demon_shaft_hz": nan, "demon_blade_hz": nan, "demon_blades": nan, "demon_confidence": 0.0, "demon_lines": 0,
           "demon_max_snr_db": nan, "tonal_lines": 0, "tonal_max_snr_db": nan, "tonal_mean_shift_hz": nan}
    if source:
        out.update(demon_shaft_hz=nan if source["shaft_hz"] is None else source["shaft_hz"],
                   demon_blade_hz=nan if source["blade_hz"] is None else source["blade_hz"],
                   demon_blades=nan if source["blades"] is None else float(source["blades"]),
                   demon_confidence=source["confidence"], demon_lines=len(source["lines"]),
                   demon_max_snr_db=max((q["snr_db"] for q in source["lines"]), default=nan))
    det = [q for q in (lines or []) if q.get("detected", True)]
    if det:
        out.update(tonal_lines=len(det), tonal_max_snr_db=max(q["snr_db"] for q in det),
                   tonal_mean_shift_hz=float(np.mean([q["frequency_hz"] - q["nominal_hz"] for q in det])))
    return out


def passive_report(records, passive: dict, engine: str = "python") -> list[dict]:
    """The passive chain's report lines for a sequence of records, with the tonal results when
    `passive["tonal"]` is set: what `build/bin/sonar_live --passive` prints. `passive` holds
    f_low, f_high, n_fft, hop, n_average and the bearing-grid and SBL settings of core.doa_passive
    (`method`, `first_deg`, ...), plus `tonal`. Bearings are always by libs/live (there is no
    Python passive bearing stage); `engine` chooses the tonal analyzer: "python" (this module) or "cpp"."""
    import json

    from . import _gpusonar as core
    p = dict(passive)
    tonal_cfg = p.pop("tonal", None)
    f_low, f_high = float(p.pop("f_low", 0.0)), float(p.pop("f_high", 0.0))
    n_fft, hop, n_average = int(p.get("n_fft", 1024)), int(p.get("hop", 512)), int(p.get("n_average", 8))
    analyzer = None
    if tonal_cfg is not None:
        analyzer = (TonalAnalyzer(tonal_cfg, f_low, f_high) if engine == "python"
                    else core.TonalAnalyzerCpp(cpp_options(tonal_cfg, f_low, f_high)))
    lines = []
    pending = []                               # results completed while a record had no frame to carry them
    for rec in records:
        frames, _ = core.doa_passive(rec, f_low, f_high, **p)
        s = rec.samples
        out = [{"time_ns": int(rec.transmit.time_ns) + _rint(fr["time"] * 1e9), "time_s": fr["time"],
                "noise_db": float(10.0 * np.log10(max(fr["noise"], 1e-300))),
                "bearings_deg": [float(np.degrees(q[0])) for q in fr["peaks"]],
                "power_db": [float(10.0 * np.log10(max(q[1], 1e-300))) for q in fr["peaks"]],
                "snr_db": [float(q[2]) for q in fr["peaks"]]} for fr in frames]
        if analyzer is not None:
            events = analyzer.push(rec) if engine == "python" else json.loads(analyzer.push(rec))
            pending.extend(events)
            if not out:
                continue
            events, pending = pending, []
            span = n_fft + (n_average - 1) * hop
            ends = [s.start_time + (i + 1) * span / s.sample_rate for i in range(len(out))]
            attach(out, events, int(rec.transmit.time_ns), ends, configure(tonal_cfg)["attach_deg"])
        lines.extend(out)
    return lines


# --- model figures: how far away a line or a propeller is detected ------------------------------

def line_snr_db(level_db: float, noise_spectrum_db: float, block_s: float, array_gain_db: float) -> float:
    """Signal to noise of a tonal in its Goertzel bin: the tone's level at a hydrophone (dB re 1 uPa)
    over the noise in the bin, which is the noise spectrum level (dB re 1 uPa^2/Hz) times the
    Hann-weighted block's noise bandwidth 1.5 / block_s, less the array gain. What `snr_db` of a
    line estimates (tests: a tone in white noise)."""
    return level_db - noise_spectrum_db - 10.0 * math.log10(1.5 / block_s) + array_gain_db


def demon_snr_db(hydrophone_snr_db: float, array_gain_db: float, depth: float, bandwidth: float, segment_s: float,
                 rows: int = 1) -> float:
    """Level of a DEMON line above the background of the normalised envelope spectrum (its `snr_db`),
    for a source whose noise is `hydrophone_snr_db` above the ambient noise in the band at a
    hydrophone and whose amplitude is modulated with index `depth` (small) at the line's rate.

    The beam holds source and noise in the ratio r = snr x array gain. Square law doubles the
    modulation index and the unmodulated noise dilutes it: m = 2 depth r / (1 + r). The power of
    Gaussian noise of bandwidth B fluctuates about its mean with one-sided spectral density 2 / B
    per Hz (relative), at rates far below B; a Hann-weighted segment of T seconds has noise
    bandwidth 1.5 / T, and a line of index m reads m^2 against 2 x (2 / B) x 1.5 / T = 6 / (B T).
    Rows steered separately and added have independent noise fluctuations: 6 / (B T rows). So the
    line stands 1 + m^2 B T rows / 6 above the background. The same to first order for the
    "hilbert" envelope (index and fluctuation both halve). Against the analyzer on a synthetic
    propeller (32 hydrophones, 2 kHz, 4 s segments, depth 0.4): within 0.6 dB for the source 12 to
    21 dB below the noise at a hydrophone (23.4 measured for 24.0 at -12 dB, 13.2 for 12.6 at
    -21 dB); 1.5 dB optimistic at -6 dB, where the beam is mostly the source and its own modulated
    noise raises the background. It is the weak-source end that sets a detection range."""
    r = 10.0 ** ((hydrophone_snr_db + array_gain_db) / 10.0)
    m = 2.0 * depth * r / (1.0 + r)
    return 10.0 * math.log10(1.0 + m * m * bandwidth * segment_s * rows / 6.0)


def detection_ranges(sonar, noise_spectrum_db: float, *, tonal_levels_db=(), tonal_hz: float | None = None,
                     broadband_levels_db=(), depth: float = 0.1, config: dict | None = None, band=None, rows: int = 1,
                     spreading: float = 20.0, **equation) -> dict:
    """Model figures, not measurements: the range at which a tonal of each source level (dB re 1 uPa
    at 1 m) reaches `goertzel.min_snr_db` in its bin, and at which a modulation line of index
    `depth` of a broadband source of each spectrum level (dB re 1 uPa^2/Hz at 1 m, flat over the
    band) reaches the DEMON threshold, for a sonar's array (`rows` rows of equal length) and the
    `passive.tonal` settings. One-way loss by gpusonar.sonar_equation (spreading law plus
    absorption at the frequency); noise uncorrelated between hydrophones (array gain 10 log10 of a
    row's hydrophones, less 1.76 dB with the Hann aperture weighting); the DEMON range is where the
    line's expected level equals the threshold (detected about half the time). No multipath, no
    directional noise, no source directivity, no fluctuation of the source."""
    from .sonar_equation import SonarEquation
    cfg = configure(config if config is not None else {"goertzel": {"lines": [tonal_hz or sonar.center_frequency]}, "demon": {}})
    g = cfg["goertzel"] or GOERTZEL
    d = cfg["demon"] or DEMON
    gain = 10.0 * math.log10(sonar.n_elements / rows / (1.5 if cfg["window"] == "hann" else 1.0))
    band = band or d["band"] or (sonar.center_frequency - 0.5 * sonar.bandwidth, sonar.center_frequency + 0.5 * sonar.bandwidth)
    width = band[1] - band[0]
    # Hann segments half overlapped: spectra a segment apart correlate by (1/6)^2 (the analyzer computes it for any overlap)
    correlation = 1.0 / 36.0 if abs(d["segment_overlap"] - 0.5) < 1e-9 else 0.0
    threshold_db = 10.0 * math.log10(demon_threshold(d["pfa"], int(d["n_average"]), correlation, 2 * int(d["background_bins"]))[0])

    class Line(SonarEquation):
        def snr(self, range_m):
            return line_snr_db(self.source_level_db - self.transmission_loss(range_m), self.noise_spectrum_db, g["block_s"], gain)

    class Modulated(SonarEquation):
        def snr(self, range_m):
            return demon_snr_db(self.source_level_db - float(self.transmission_loss(range_m)) - self.noise_spectrum_db, gain, depth,
                                width, d["segment_s"], rows)

    def make(cls, level, frequency):
        return cls(float(level), float(frequency), width, g["block_s"], sonar.n_elements, noise_spectrum_db, spreading=spreading,
                   array_gain_db=gain, **equation)

    f_line = tonal_hz or sonar.center_frequency
    f_mid = 0.5 * (band[0] + band[1])
    return {"array_gain_db": gain, "rows": rows, "band_hz": [float(band[0]), float(band[1])],
            "tonal": [{"source_level_db": float(sl), "frequency_hz": float(f_line), "threshold_db": float(g["min_snr_db"]),
                       "snr_at_1km_db": float(make(Line, sl, f_line).snr(1000.0)),
                       "range_m": make(Line, sl, f_line).detection_range(g["min_snr_db"])} for sl in tonal_levels_db],
            "demon": [{"spectrum_level_db": float(sl), "depth": depth, "threshold_db": threshold_db,
                       "snr_at_1km_db": float(make(Modulated, sl, f_mid).snr(1000.0)),
                       "range_m": make(Modulated, sl, f_mid).detection_range(threshold_db)} for sl in broadband_levels_db]}
