#!/usr/bin/env python3
"""The SFI Smart Ocean dataset (SODAC), 4-8 kHz band, against our propagation and time spreading.

    GPUSONAR_SMARTOCEAN=<dir> tools/channel_smartocean.py sound   --cache <dir>     (raw recordings -> soundings)
    GPUSONAR_SMARTOCEAN=<dir> tools/channel_smartocean.py analyse --cache <dir> [--figures docs/figures]
                                                                   [--out docs/figures/channel_validation_smartocean.md]
    tools/fetch_channels.py <dir> --smartocean --from-hf            (the owner's private copy; nothing else is contacted)

<dir> holds RX-LF.zip, TX-waveforms.zip, CTD.csv and (optional) bathymetry.json of the dataset
(van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh, IEEE DataPort 2025, doi:10.21227/3aa6-4k33,
CC BY 4.0; Descriptor: IEEE Data Descriptions 2 (2025) 218-223, read as hal-05793814). The
files need a login at IEEE DataPort; nothing of them is in this repository. bathymetry.json is
made by `bathymetry` below from the dataset's depth geodatabase (Kartverket "Sjokart - Dybdedata",
needs pyogrio; depth along the two paths only).

What the dataset states (Descriptor article): positions and depths of the sender (20 m) and of
two bottom-tripod hydrophones, R1 (43 m, 0.27 km) and R2 (14 m, 1.02 km); band 4-8 kHz about
6 kHz; source level; 94 Pa per full scale; two CTD casts "from the deep end of the platform
close to the transmitters", at the start and the end of the night; "weather conditions were
calm, with little wind and some drizzle". Not stated: wind speed, wave height, seabed type,
tripod heights, water depth at the instruments (the chart gives it), tide.

Probes (found in the recordings by correlation; one cycle every 10 min, 80 cycles):
  LFM-3000   0.745 s chirp every 3 s, 5 of them      -> intensity profile to 3 s      (all cycles)
  LFM-250    chirp train, period 250 ms, 16 s        -> 4 Hz snapshots, 250 ms window (cycles 14-80)
  LFM-25     chirp train, period 25 ms, 10 s         -> 40 Hz snapshots, 25 ms window (cycles 14-80)
  A-PRBS     aperiodic BPSK sequence, 60 s (16 s in cycles 1-13) -> least squares, 0.5 s blocks every 0.25 s, 50 ms window
  LFM-225, P-PRBS  period 225 / 227.3 ms, 16 s       -> 4.4 Hz snapshots              (cycles 1-13)
All soundings: band 5 kHz about 6 kHz with a raised-cosine weight (half-power width 0.29 ms,
first sidelobe -31 dB). No tracker: `sim/channel_sounding.py`.

    tools/channel_smartocean.py bathymetry --gdb <unpacked .gdb> --out <dir>/bathymetry.json     (once; needs pyogrio)

What `analyse` does, and what is stated, assumed and fitted in each part:

  arrivals     the measured profile against straight rays and Bellhop (isovelocity, each cast) over
               the charted seabed. Stated: geometry, casts. Assumed: the seabed half-space, a 1.5 m
               tripod at R1, that the casts hold along the paths. Fitted: nothing.
  surface      the surface reflection at R1 against the direct arrival, snapshot by snapshot:
               coherent share, phase and amplitude variance, spectrum, half-band ratio, the
               fluctuating profile behind it and its tail (`channel_sounding.tail_decay`), the
               spectra at the peak and behind it. Then our model with a surface fitted to the
               tail (two numbers: rms height from the tail's energy, rms vertical speed from its
               frequency; the slope is NOT fitted: Cox-Munk for the wind whose Pierson-Moskowitz
               height that is), passed through the same soundings and fits, under each decay rule.
  spread       energy against delay to 3 s, both ranges. Nothing assumed or fitted.
  levels       absolute level of the first arrival (94 Pa per full scale, stated) against the
               stated source levels, Bellhop and spherical spreading.
  noise        the stretch before each cycle's first transmission: spectrum level and slope in
               4-8 kHz, tail statistics, `noise_fit.fit`, coherence between the hydrophones; the
               wind speed whose Wenz-type curve (sim/noise.py) has the same level (one fitted number).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gpusonar.sim import channel_replay as cr  # noqa: E402
from gpusonar.sim import channel_sounding as cs  # noqa: E402

FC, FS, BAND = 6000.0, 12000.0, 5000.0
LEAD = 0.005
PERIODIC = {"LFM-3000-LF": 36000, "LFM-250-LF": 3000, "LFM-25-LF": 300, "LFM-225-LF": 2700, "P-PRBS-LF": 2728}   # period, samples at FS
EARLY = 720                                   # delay bins kept of the 4 Hz soundings: -5 .. +55 ms
ATTRIBUTION = ("Data: SFI Smart Ocean dataset (van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh, IEEE DataPort 2025, "
               "doi:10.21227/3aa6-4k33, CC BY 4.0)")


def data_dir() -> Path | None:
    d = os.environ.get("GPUSONAR_SMARTOCEAN")
    return Path(d) if d and Path(d).is_dir() else None


# -- sounding stage -------------------------------------------------------------------------------------

class Source:
    """Recordings and probes, from the dataset's zips or from a cache of complex envelopes
    (<cache>/bb/R1_14.npy, made by this class on first use; 15 MB per recording)."""

    def __init__(self, data: Path | None, cache: Path):
        self.data, self.cache = data, cache
        (cache / "bb").mkdir(parents=True, exist_ok=True)
        meta = cache / "bb" / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self._members = None
        self._probes: dict = {}

    def keys(self) -> list:
        if self.data is not None and (self.data / "RX-LF.zip").exists():
            return ["%s_%02d" % (rx, cyc) for cyc, rx, _ in cs.sodac_members(str(self.data / "RX-LF.zip"))]
        return sorted(self.meta, key=lambda k: (int(k[3:]), k[:2]))

    def recording(self, key: str) -> np.ndarray:
        path = self.cache / "bb" / (key + ".npy")
        if path.exists():
            return np.load(path)
        if self._members is None:
            self._members = {"%s_%02d" % (rx, cyc): name for cyc, rx, name in cs.sodac_members(str(self.data / "RX-LF.zip"))}
        b, info = cs.load_sodac_recording(str(self.data / "RX-LF.zip"), self._members[key], FC, FS)
        np.save(path, b)
        self.meta[key] = {"start_s": info["start_s"], "comment": info.get("ICMT"), "n": int(len(b))}
        (self.cache / "bb" / "meta.json").write_text(json.dumps(self.meta))
        return b

    def probe(self, name: str) -> np.ndarray:
        if name not in self._probes:
            path = self.cache / ("probe_" + name + ".npy")
            if not path.exists():
                np.save(path, cs.load_sodac_probe(str(self.data / "TX-waveforms.zip"), name, FC, FS).astype(np.complex64))
            self._probes[name] = np.load(path)
        return self._probes[name]


def sound_recording(src: Source, key: str) -> dict:
    """All soundings of one recording, as arrays to store (see the module docstring)."""
    from scipy.signal import resample_poly
    y = src.recording(key)
    cycle = int(key[3:])
    out: dict = {"start_s": np.float64(src.meta.get(key, {}).get("start_s") or np.nan)}
    first = None
    names = ["LFM-3000-LF", "LFM-250-LF", "LFM-25-LF"] if cycle >= 14 else ["LFM-3000-LF", "LFM-225-LF", "P-PRBS-LF"]
    for name in names:
        p, per = src.probe(name), PERIODIC[name]
        k, quality = cs.find_probe(y, p)
        tag = name[:-3].lower().replace("-", "")
        if name == "LFM-3000-LF":
            ch = cs.sound_periodic(y, p[:per], FS, FC, start=k, n_periods=5, band=BAND, lead=LEAD)
            out[tag + "_total"] = (np.abs(ch.h[:, 0]) ** 2).mean(axis=0).astype(np.float32)
            out[tag + "_coherent"] = (np.abs(ch.h[:, 0].mean(axis=0)) ** 2).astype(np.float32)
        else:
            ch = cs.sound_periodic(y, p[per:2 * per], FS, FC, start=k + per, n_periods=len(p) // per - 2, band=BAND, lead=LEAD)
            out[tag + "_h"] = ch.h[:, 0, :EARLY if per > EARLY else per].astype(np.complex64)
            if per > EARLY:
                out[tag + "_total"] = (np.abs(ch.h[:, 0]) ** 2).mean(axis=0).astype(np.float32)
        out[tag + "_start"] = np.float64(k / FS)
        out[tag + "_quality"] = np.float64(quality)
        first = k if first is None else min(first, k)
    name = "A-PRBS-LF" if cycle >= 14 else "A-PRBS-LF-16s"
    p = src.probe(name)
    k, quality = cs.find_probe(y, p, head=24000)
    y6 = resample_poly(y[k - 600:k + len(p) + 6000], 1, 2)
    ch = cs.sound_sliding(y6, resample_poly(p, 1, 2), FS / 2, FC, start=300, block=3000, hop=1500, n_delay=300, band=BAND, lead=LEAD)
    out["aprbs_h"] = ch.h[:, 0].astype(np.complex64)
    out["aprbs_start"], out["aprbs_quality"] = np.float64(k / FS), np.float64(quality)
    # Signal-free: from 0.2 s to 0.3 s before the first transmission of the cycle, found by its
    # power (the first 10 ms block 20 dB above the median of 0.2-1.5 s): 2.4 s in cycles 14-80, 3.5 s before.
    blocks = (np.abs(y[: len(y) // 120 * 120].reshape(-1, 120)) ** 2).mean(axis=1)
    loud = np.flatnonzero(blocks[20:] > 100.0 * np.median(blocks[20:150]))
    onset = (20 + int(loud[0])) * 120 if len(loud) else min(first, k)
    out["noise"] = y[int(0.2 * FS):onset - int(0.3 * FS)].astype(np.complex64)
    return out


def stage_sound(args) -> int:
    src = Source(data_dir(), Path(args.cache))
    out_dir = Path(args.cache) / "soundings"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = src.keys()
    if args.only:
        keys = [k for k in keys if k in args.only]
    for key in keys:
        dst = out_dir / (key + ".npz")
        if dst.exists() and not args.force:
            continue
        res = sound_recording(src, key)
        tmp = out_dir / (key + ".tmp.npz")
        np.savez(tmp, **res)
        os.replace(tmp, dst)
        print(key, "noise %.1f s" % (len(res["noise"]) / FS), " ".join("%s %.0f" % (k[:-8], res[k]) for k in sorted(res) if k.endswith("_quality")), flush=True)
    return 0


# -- per-recording analysis ----------------------------------------------------------------------------

LEAD_BINS = int(round(LEAD * FS))
# Where the second strong arrival is looked for, ms after the first (R1: the surface reflection;
# R2: the second of the doublet). Chosen from the data, not from a model.
SECOND = {"R1": (3.6, 5.4), "R2": (1.4, 2.8)}


def as_channel(h: np.ndarray, fs_time: float, fs_delay: float = FS) -> cr.TimeVaryingChannel:
    return cr.TimeVaryingChannel(h=np.asarray(h, dtype=np.complex128)[:, None, :], fs_time=fs_time, fs_delay=fs_delay, fc=FC)


def peak_delay(profile: np.ndarray, k: int) -> float:
    """Sub-bin position of a peak of an intensity profile (parabola through three points of its logarithm)."""
    a, b, c = np.log(np.maximum(profile[k - 1:k + 2], 1e-300))
    d = a - 2.0 * b + c
    return k + (0.5 * (a - c) / d if d < 0 else 0.0)


def find_pair(total: np.ndarray, rx: str, fs_delay: float = FS) -> tuple[int, int]:
    """Delay bins (earlier, later) of the pair of strong arrivals SECOND[rx] apart: the strongest
    arrival in the first 12 ms of the window and its strongest partner that far before or after it."""
    n = int(0.012 * fs_delay) + int(LEAD * fs_delay)
    top = int(np.argmax(total[:n]))
    lo, hi = (int(round(v * 1e-3 * fs_delay)) for v in SECOND[rx])
    after = top + lo + int(np.argmax(total[top + lo:top + hi + 1]))
    if top - hi < 1:
        return top, after
    before = top - hi + int(np.argmax(total[top - hi:top - lo + 1]))
    return (top, after) if total[after] >= total[before] else (before, top)


def pair_statistics(ch: cr.TimeVaryingChannel, rx: str, floor_bins: slice | None = None) -> dict:
    """The second arrival against the first, snapshot by snapshot: z = h2 / h1 (peak bins).

    phase_rms, amplitude_rms: of arg z (straight line removed) and |z| / mean|z|, each with the
    sounding's fluctuation floor taken out (a floor of power f per bin adds f / (2 |h|^2) to each
    quadrature of each arrival; f is measured in `floor_bins`, default the 2.5 ms before the first
    arrival). coherent: |<z>|^2 / <|z|^2> with the same correction: the second arrival's
    coherent share when the first is steady. Doppler widths of the fluctuating part of z."""
    h = ch.h[:, 0, :]
    total = (np.abs(h) ** 2).mean(axis=0)
    b1, b2 = find_pair(total, rx, ch.fs_delay)
    ref = h[:, b1] / np.abs(h[:, b1])
    hr = h * np.conj(ref)[:, None]
    fluct = total - np.abs(hr.mean(axis=0)) ** 2
    fb = floor_bins or slice(max(b1 - int(0.0035 * ch.fs_delay), 0), b1 - int(0.001 * ch.fs_delay))
    floor = float(np.median(fluct[fb]))
    z = h[:, b2] / h[:, b1]
    t = ch.time
    ph = np.unwrap(np.angle(z))
    ph = ph - np.polyval(np.polyfit(t, ph, 1), t)
    amp = np.abs(z) / np.abs(z).mean()
    p1, p2 = total[b1], total[b2]
    quad = 0.5 * floor * (1.0 / p1 + 1.0 / p2)                      # variance the floor adds to each quadrature of z / |z|
    out = {"bin1": b1, "bin2": b2, "delay_ms": float((peak_delay(total, b2) - peak_delay(total, b1)) / ch.fs_delay * 1e3),
           "level2_db": float(10 * np.log10(p2 / p1)), "level1_pa2": float(p1), "floor_db": float(10 * np.log10(max(floor, 1e-300) / p1)),
           "phase_rms_raw": float(ph.std()), "amplitude_rms_raw": float(amp.std()),
           "phase_var": float(ph.var() - quad), "amplitude_var": float(amp.var() - quad), "floor_quadrature_var": float(quad)}
    zc = z * np.exp(-1j * np.angle(z.mean()))
    out["coherent"] = float(min(np.abs(zc.mean()) ** 2 / max((np.abs(zc) ** 2).mean() - 2.0 * quad * np.abs(zc.mean()) ** 2, 1e-300), 1.5))
    nfft = int(min(len(z), 256))
    nu, s = cr.doppler_spectrum((z / np.abs(z).mean() - (z / np.abs(z).mean()).mean())[:, None], ch.fs_time, nfft=nfft)
    s_floor = 2.0 * quad / ch.fs_time
    w = cr.spectrum_widths(nu, s, floor=s_floor)
    out["doppler"] = {k: float(v) for k, v in w.items()}
    out["spectrum"] = (nu, s, s_floor)
    out["series"] = (ph, amp)
    # Level statistics of each arrival's energy (floor removed as in channel_replay.arrival_statistics).
    for name, b, p in (("si1", b1, p1), ("si2", b2, p2)):
        e = np.abs(h[:, b]) ** 2
        sig = max(e.mean() - floor, 1e-300)
        out[name] = float(max(e.var() - floor ** 2 - 2.0 * floor * sig, 0.0) / sig ** 2)
    # The fluctuating intensity profile around each arrival, relative to that arrival's steady peak.
    for name, b in (("profile1", b1), ("profile2", b2)):
        seg = slice(b - 12, b + 37)
        out[name] = (fluct[seg] - floor) / max(np.abs(hr[:, b].mean()) ** 2, 1e-300)
        out[name.replace("profile", "steady")] = np.abs(hr[:, seg].mean(axis=0)) ** 2 / max(np.abs(hr[:, b].mean()) ** 2, 1e-300)
    # Spectra over time of the fluctuation at the second arrival's peak, 0.4-0.9 ms behind it, and of the
    # floor (per delay bin, relative to the arrival's steady peak): does the tail fluctuate like the peak?
    nfft_d = int(min(len(z), 64 if ch.fs_time < 10.0 else 256))
    steady_peak = max(np.abs(hr[:, b2].mean()) ** 2, 1e-300)

    def spec(sl):
        nu_d, s_d = cr.doppler_spectrum(hr[:, sl] - hr[:, sl].mean(axis=0, keepdims=True), ch.fs_time, nfft=nfft_d)
        return nu_d, s_d / ((sl.stop - sl.start) * steady_peak)
    n_a, n_b = int(round(0.4e-3 * ch.fs_delay)), int(round(0.9e-3 * ch.fs_delay))
    nu_d, s_peak = spec(slice(b2, b2 + 1))
    out["delay_spectra"] = (nu_d, s_peak, spec(slice(b2 + n_a, b2 + n_b + 1))[1], spec(fb)[1])
    # Lower and upper half of the band: a change of path length gives a phase proportional to frequency.
    out["phase_var_lower"], out["phase_var_upper"] = half_band_phase_variances(h, b1, b2, t, ch.fs_time, ch.fs_delay)
    out["phase_var_slow"], out["amplitude_var_slow"] = slow_variance(ph, ch.fs_time), slow_variance(amp - amp.mean(), ch.fs_time)
    p1s = np.unwrap(np.angle(h[:, b1]))
    out["first_phase"] = p1s - np.polyval(np.polyfit(t, p1s, 1), t)
    return out


def half_band_phase_variances(h: np.ndarray, b1: int, b2: int, t: np.ndarray, fs_time: float, fs_delay: float) -> tuple[float, float]:
    """Slow phase variance of arrival b2 against arrival b1 in the lower and the upper half of the
    band: each arrival's nine bins (raised-cosine taper) are transformed along delay and the bins
    400 to 2000 Hz below and above the centre are summed. The taper smears the spectrum, so the
    two halves are closer in effective frequency than their nominal centres:
    `half_band_ratio_for_path_change` gives what a pure change of path length reads."""
    half = {}
    for name, b in (("1", b1), ("2", b2)):
        seg = h[:, b - 4:b + 5] * np.hanning(11)[1:-1][None, :]
        hf = np.fft.fft(seg, 64, axis=1)
        f = np.fft.fftfreq(64, 1.0 / fs_delay)
        half[name] = (hf[:, (f < -400.0) & (f > -2000.0)].sum(axis=1), hf[:, (f > 400.0) & (f < 2000.0)].sum(axis=1))
    var = []
    for i in (0, 1):
        zz = half["2"][i] / half["1"][i]
        p = np.unwrap(np.angle(zz))
        var.append(slow_variance(p - np.polyval(np.polyfit(t, p, 1), t), fs_time))
    return var[0], var[1]


def half_band_ratio_for_path_change(fs_delay: float = FS, spacing: float = 4.4e-3) -> float:
    """Upper over lower half-band phase variance that `half_band_phase_variances` reads for an
    arrival whose path length changes (phase proportional to frequency), computed by passing such
    an arrival through the same estimator. (Frequency squared of the nominal half-band centres
    would be 1.92; the estimator's own smearing makes it less.)"""
    k = 256
    t = np.arange(400) / 4.0
    f = np.fft.fftfreq(k, 1.0 / fs_delay)
    win = cs.band_window(k, fs_delay, BAND, "hann")
    dtau = 0.05 * np.sqrt(2.0) * np.sin(2.0 * np.pi * 0.25 * t) / (2.0 * np.pi * FC)
    b1 = 40
    b2 = b1 + int(round(spacing * fs_delay))
    hf = np.exp(-2j * np.pi * (f[None, :] + FC) * (b1 / fs_delay)) + np.exp(-2j * np.pi * (f[None, :] + FC) * (b2 / fs_delay + dtau[:, None]))
    h = np.fft.ifft(hf * win[None, :], axis=1)
    lo, up = half_band_phase_variances(h, b1, b2, t, 4.0, fs_delay)
    return up / lo


def slow_variance(x: np.ndarray, fs_time: float, edge: float = 1.0) -> float:
    """Variance of a real series below `edge` Hz, less the white level seen above it (to the
    Nyquist frequency): what a slow fluctuation contributes when the sounding's own noise is
    white from snapshot to snapshot. With fs_time <= 2 edge there is nothing above: plain variance."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    p = np.abs(np.fft.rfft(x)) ** 2
    p[1:] *= 2.0
    if len(x) % 2 == 0:
        p[-1] /= 2.0
    p /= len(x) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs_time)
    low, high = (f > 0) & (f <= edge), f > edge
    if high.sum() < 4:
        return float(p[1:].sum())
    return float(p[low].sum() - p[high].mean() * low.sum())


def schroeder(total: np.ndarray, fs: float = FS) -> dict:
    """Delay-spread numbers of an intensity profile whose first arrival is at LEAD: the times
    after it by which all but 10 %, 1 % and 0.1 % of the energy has arrived, the rms delay
    spread, and the energy in the first 10 ms. The profile's last 10 % sets a noise level that
    is subtracted."""
    p = np.asarray(total, dtype=float)
    p = p - np.median(p[int(0.9 * len(p)):])
    p = p[LEAD_BINS - 12:]
    t = (np.arange(len(p)) - 12) / fs
    c = np.cumsum(p[::-1])[::-1]
    c = c / c[0]
    out = {"t%d" % db: float(t[np.argmax(c < 10.0 ** (-db / 10.0))]) for db in (10, 20, 30)}
    w = np.maximum(p, 0.0)
    m = (w * t).sum() / w.sum()
    out["rms_spread_s"] = float(np.sqrt((w * (t - m) ** 2).sum() / w.sum()))
    out["first_10ms_share"] = float(p[: 12 + int(0.010 * fs)].sum() / p.sum())
    out["energy_pa2"] = float(p.sum())
    return out


def noise_statistics(x: np.ndarray, nfft: int = 256) -> dict:
    """One-sided pressure spectral density of a signal-free complex envelope (Pa^2/Hz against
    absolute frequency; Hann, 50 % overlap) and the tail of its 4-8 kHz part: var(log|e|)
    (pi^2/8 for a Gaussian, tools/fit_noise.py's statistic) and the q99.95 / q50 ratio of |e|."""
    w = np.hanning(nfft + 1)[:nfft]
    segs = np.lib.stride_tricks.sliding_window_view(x, nfft)[:: nfft // 2]
    spec = (np.abs(np.fft.fft(segs * w[None, :], axis=1)) ** 2).mean(axis=0) / (FS * (w ** 2).sum())
    # The envelope b has x = Re(b e^{i w t}): its two-sided density at offset f is twice the one-sided density of x at fc + f, over 2.
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / FS)) + FC
    psd = np.fft.fftshift(spec) / 2.0
    xf = np.fft.fft(x)
    ff = np.fft.fftfreq(len(x), 1.0 / FS)
    e = np.fft.ifft(np.where(np.abs(ff) < 2000.0, xf, 0.0))[200:-200]
    a = np.abs(e)
    return {"frequency": f, "psd": psd, "n_blocks": len(segs), "var_log": float(np.log(a).var()),
            "q_ratio": float(np.quantile(a, 0.9995) / np.median(a)), "band_pa2": float((a ** 2).mean() / 2.0), "seconds": len(x) / FS}


def analyse_recording(z, rx: str, cycle: int) -> dict:
    """Everything measured from one recording's soundings (the arrays of `sound_recording`)."""
    out: dict = {"rx": rx, "cycle": cycle, "start_s": float(z["start_s"])}
    slow = "lfm250" if cycle >= 14 else "lfm225"
    per = PERIODIC["LFM-250-LF" if cycle >= 14 else "LFM-225-LF"]
    soundings = {"chirp4": (z[slow + "_h"], FS / per, FS), "prbs": (z["aprbs_h"], 4.0, FS / 2)}
    if cycle >= 14:
        soundings["chirp40"] = (z["lfm25_h"], 40.0, FS)
    else:
        soundings["pprbs"] = (z["pprbs_h"], FS / PERIODIC["P-PRBS-LF"], FS)
    for name, (h, fs_time, fs_delay) in soundings.items():
        ch, drift = cs.remove_drift(as_channel(h, fs_time, fs_delay))
        try:
            st = pair_statistics(ch, rx)
        except (IndexError, ValueError) as err:
            out[name] = {"error": str(err)}
            continue
        st["drift_rate"], st["first_phase_residual"] = drift["rate"], drift["phase_residual_rad"]
        st["seconds"] = ch.duration
        out[name] = st
    c4 = out["chirp4"]
    # Absolute time (s since 14 Nov 00:00 UTC by the recorder's clock) at which the 4 Hz probe's first arrival came in.
    start = float(z[slow + "_start"])
    strongest = LEAD_BINS                                             # find_probe aligned the strongest arrival there
    out["first_arrival_time_s"] = out["start_s"] + start + float(peak_delay((np.abs(soundings["chirp4"][0]) ** 2).mean(axis=0).astype(np.float64), c4["bin1"]) - strongest) / FS
    out["profile4"] = (np.abs(soundings["chirp4"][0]) ** 2).mean(axis=0)
    out["spread"] = schroeder(z["lfm3000_total"])
    out["pdp"] = z["lfm3000_total"]
    if len(z["noise"]) > 4096:
        out["noise"] = noise_statistics(np.asarray(z["noise"], dtype=np.complex128))
        out["noise_series"] = z["noise"]
    return out


# -- the documented geometry through our propagation models ---------------------------------------------

STATED = {"source_depth": cs.SODAC_POSITIONS["TX"][2], "R1": {"depth": cs.SODAC_POSITIONS["R1"][2]}, "R2": {"depth": cs.SODAC_POSITIONS["R2"][2]}}
ASSUMED = {"tripod_height": 1.5,                # m, hydrophone above the seabed at R1 (not stated)
           "bottom": {"bottom_speed": 1600.0, "bottom_density": 1.8, "bottom_attenuation": 0.5}}   # the seabed is not named: the suite's default half-space


def path_profile(bathy: dict, rx: str) -> tuple[list, float]:
    """([[range, depth], ...] from the sender, horizontal range) of one path from bathymetry.json.
    At R1 the chart (10 m contours, soundings of 2002) puts the seabed above the stated hydrophone
    depth, so the last 60 m are lowered to hydrophone depth + ASSUMED tripod height; R2 is the chart."""
    r, d = np.asarray(bathy[rx]["range"], dtype=float), np.asarray([np.nan if v is None else v for v in bathy[rx]["depth"]], dtype=float)
    ok = np.isfinite(d) & (r >= 0.0)
    r, d = r[ok], d[ok]
    r[0] = 0.0
    length = float(bathy[rx]["length"])
    if rx == "R1":
        d = d + np.clip(1.0 - np.abs(r - length) / 60.0, 0.0, 1.0) * (STATED[rx]["depth"] + ASSUMED["tripod_height"] - np.interp(length, r, d))
    return [[float(a), float(b)] for a, b in zip(r, d)], length


def bellhop_arrivals(bathy: dict, rx: str, speed, depth=None, *, beams: int = 6001) -> list:
    """Bellhop arrivals at a hydrophone for a profile (`speed` a number: isovelocity), over the
    charted bathymetry of its path: [{delay s, level dB re 1 m, n_surface, n_bottom, launch deg
    (positive down), arrive deg, surface_grazing deg (Snell, from the launch angle)}] by delay."""
    from gpusonar.sim import bellhop as bh
    prof, length = path_profile(bathy, rx)
    zmax = max(p[1] for p in prof) + 1.0
    if depth is None:
        env = bh.OceanEnvironment(depth=zmax, sound_speed=float(speed), bathymetry=prof, **ASSUMED["bottom"])
    else:
        env = bh.OceanEnvironment(depth=zmax, ssp_depth=np.asarray(depth), ssp_speed=np.asarray(speed), bathymetry=prof, **ASSUMED["bottom"])
    arr = bh.run_bellhop(env, FC, STATED["source_depth"], [STATED[rx]["depth"]], [length], n_beams=beams, angles=(-60.0, 60.0))
    c_src, c_top = env.speed_at(STATED["source_depth"]), env.speed_at(0.0)
    rows = []
    for i in np.argsort(arr.delay[0][0]):
        la = float(arr.launch_angle[0][0][i])
        rows.append({"delay": float(arr.delay[0][0][i]), "level_db": float(20 * np.log10(max(abs(arr.amplitude[0][0][i]), 1e-30))),
                     "n_surface": int(arr.n_surface[0][0][i]), "n_bottom": int(arr.n_bottom[0][0][i]), "launch_deg": float(np.degrees(la)),
                     "arrive_deg": float(np.degrees(arr.arrival_angle[0][0][i])),
                     "surface_grazing_deg": float(np.degrees(np.arccos(np.clip(np.cos(la) * c_top / c_src, -1.0, 1.0))))})
    return rows


def straight_rays(rx: str, length: float, c: float) -> dict:
    """Direct and surface-reflected straight rays: delays (s), their difference, the grazing angle."""
    zs, zr = STATED["source_depth"], STATED[rx]["depth"]
    d, s = np.hypot(length, zr - zs), np.hypot(length, zr + zs)
    return {"direct": d / c, "surface": s / c, "difference_ms": (s - d) / c * 1e3, "grazing_deg": float(np.degrees(np.arctan2(zr + zs, length)))}


# -- everything, collected -------------------------------------------------------------------------------

def collect(cache: Path, log=print) -> dict:
    """analyse_recording over every sounding file of the cache, kept in <cache>/analysis.pkl."""
    import pickle
    store = cache / "analysis.pkl"
    files = sorted((cache / "soundings").glob("R?_??.npz"))
    if store.exists() and store.stat().st_mtime > max(f.stat().st_mtime for f in files) and store.stat().st_mtime > Path(__file__).stat().st_mtime:
        return pickle.loads(store.read_bytes())
    res: dict = {}
    for f in files:
        rx, cycle = f.stem[:2], int(f.stem[3:])
        with np.load(f) as z:
            res[(rx, cycle)] = analyse_recording(z, rx, cycle)
    log(f"analysed {len(res)} recordings")
    store.write_bytes(pickle.dumps(res))
    return res


def med(values, digits: int = 3) -> dict:
    """{median, lo, hi (the 16 % and 84 % points), n} of a list (NaN left out)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    return {"median": round(float(np.median(v)), digits + 2), "lo": round(float(np.quantile(v, 0.16)), digits + 2), "hi": round(float(np.quantile(v, 0.84)), digits + 2), "n": int(len(v))}


def theoretical_pulse(n_before: int = 12, n_after: int = 36, fs: float = FS) -> np.ndarray:
    """Intensity response of the soundings to one steady path, on the bins of `pair_statistics`'
    profiles (peak at index n_before), averaged over where within a bin the path falls."""
    k = 256
    w = cs.band_window(k, fs, BAND, "hann")
    f = np.fft.fftfreq(k)
    acc = np.zeros(n_before + n_after + 1)
    for off in np.linspace(-0.45, 0.45, 7):
        p = np.abs(np.fft.ifft(w * np.exp(-2j * np.pi * f * (64 + off)))) ** 2
        acc += p[64 - n_before:64 + n_after + 1] / p.max()
    return acc / acc.max()


def derive(res: dict, data: Path | None) -> dict:
    """The numbers of the report from the per-recording results (and Bellhop, when the CTD file,
    bathymetry.json and the executable are there)."""
    from gpusonar import acoustics
    from gpusonar.sim import timespread
    from gpusonar.sim.bellhop import find_bellhop
    out: dict = {"n_recordings": len(res)}
    cycles = sorted({c for _, c in res})
    late = [c for c in cycles if c >= 14]

    def get(rx, sounding, key, cyc=None):
        return [res[(rx, c)][sounding].get(key) if (rx, c) in res and sounding in res[(rx, c)] and "error" not in res[(rx, c)][sounding] else None
                for c in (cyc or cycles)]

    # ---- propagation: the profiles, straight rays and Bellhop -----------------------------------------
    casts = cs.load_ctd(str(data / "CTD.csv")) if data is not None and (data / "CTD.csv").exists() else []
    bathy = json.loads((data / "bathymetry.json").read_text()) if data is not None and (data / "bathymetry.json").exists() else None
    prop: dict = {"casts": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in casts]}
    if casts:
        c_src = float(np.mean([np.interp(STATED["source_depth"], c["depth"], c["speed"]) for c in casts]))
        prop["speed_at_source"] = c_src
        prop["speed_surface"] = [float(c["speed"][0]) for c in casts]
        prop["hysteresis_at_source"] = [float(np.interp(STATED["source_depth"], c["depth"], c["hysteresis"])) for c in casts]
    else:
        c_src = 1500.0
    for rx in ("R1", "R2"):
        length = float(bathy[rx]["length"]) if bathy else cs.SODAC_RANGE_M[rx]
        prop[rx] = {"range_m": length, "straight": straight_rays(rx, length, c_src)}
        if bathy and casts and find_bellhop():
            runs = {"isovelocity": bellhop_arrivals(bathy, rx, c_src)}
            for i, c in enumerate(casts):
                runs[f"cast{i + 1}"] = bellhop_arrivals(bathy, rx, c["speed"], c["depth"])
            prop[rx]["bellhop"] = runs
    out["propagation"] = prop

    # ---- what was measured, per receiver ----------------------------------------------------------------
    for rx in ("R1", "R2"):
        m: dict = {}
        for snd, cyc in (("chirp4", cycles), ("prbs", late), ("chirp40", late)):
            row = {"n": len([v for v in get(rx, snd, "delay_ms", cyc) if v is not None])}
            for key in ("delay_ms", "level2_db", "floor_db", "coherent", "phase_var", "amplitude_var", "phase_var_slow", "amplitude_var_slow", "si1", "si2",
                        "first_phase_residual", "seconds"):
                row[key] = med(get(rx, snd, key, cyc))
            dop = [d for d in get(rx, snd, "doppler", cyc) if d]
            for key in ("rms", "w50", "w90", "shape"):
                row["doppler_" + key] = med([d[key] for d in dop])
            lo, up = np.array([v for v in get(rx, snd, "phase_var_lower", cyc) if v is not None]), np.array([v for v in get(rx, snd, "phase_var_upper", cyc) if v is not None])
            if len(lo) > 4:
                rng = np.random.default_rng(0)
                boot = [up[i].sum() / lo[i].sum() for i in (rng.integers(0, len(lo), len(lo)) for _ in range(400))]
                row["band_ratio"] = {"value": float(up.sum() / lo.sum()), "se": float(np.std(boot))}
            m[snd] = row
        m["spread"] = {k: med([res[(rx, c)]["spread"][k] for c in cycles if (rx, c) in res], 4) for k in ("t10", "t20", "t30", "rms_spread_s", "first_10ms_share", "energy_pa2")}
        m["level1_pa2"] = med(get(rx, "chirp4", "level1_pa2"), 5)
        out[rx] = m
    # sub-band centres of pair_statistics' halves (raised-cosine band weight, 400-2000 Hz either side)
    f = np.linspace(400.0, 2000.0, 400)
    wgt = 0.5 * (1.0 + np.cos(2.0 * np.pi * f / BAND))
    off = float((f * wgt).sum() / wgt.sum())
    out["band_centres_hz"] = [FC - off, FC + off]
    out["band_ratio_nominal"] = ((FC + off) / (FC - off)) ** 2
    out["band_ratio_expected"] = {"chirp4": half_band_ratio_for_path_change(FS), "chirp40": half_band_ratio_for_path_change(FS), "prbs": half_band_ratio_for_path_change(FS / 2)}

    # ---- travel time between the receivers (both recorders' clocks taken as one) ---------------------------
    dt = [res[("R2", c)]["first_arrival_time_s"] - res[("R1", c)]["first_arrival_time_s"] for c in cycles if ("R1", c) in res and ("R2", c) in res]
    out["travel_time_difference_s"] = med(dt, 5)

    # ---- the surface arrival at R1: one surface, two fitted numbers -------------------------------------------
    r1 = out["R1"]
    bh_r1 = prop["R1"].get("bellhop", {})
    surf = None
    for name in ("cast1", "cast2", "isovelocity"):
        rows = [a for a in bh_r1.get(name, []) if a["n_surface"] == 1 and a["n_bottom"] == 0]
        direct = [a for a in bh_r1.get(name, []) if a["n_surface"] == 0 and a["n_bottom"] == 0]
        if rows and direct:
            s, d = max(rows, key=lambda a: a["level_db"]), max(direct, key=lambda a: a["level_db"])
            prop["R1"].setdefault("surface_minus_direct_ms", {})[name] = (s["delay"] - d["delay"]) * 1e3
            prop["R1"].setdefault("surface_grazing_deg", {})[name] = s["surface_grazing_deg"]
            prop["R1"].setdefault("direct_level_db", {})[name] = d["level_db"]
            prop["R1"].setdefault("surface_re_direct_db", {})[name] = s["level_db"] - d["level_db"]
            if surf is None:
                surf = (s, d)
    grazing = np.radians(np.mean([prop["R1"]["surface_grazing_deg"][k] for k in ("cast1", "cast2")])) if surf else np.radians(prop["R1"]["straight"]["grazing_deg"])
    c_top = float(np.mean(prop.get("speed_surface", [c_src])))
    k_top = 2.0 * np.pi * FC / c_top
    chi = 2.0 * k_top * np.sin(grazing)                               # rad of phase per metre of surface height
    best = r1["prbs"] if r1["prbs"]["n"] else r1["chirp4"]
    coh = best["coherent"]["median"]
    gamma2 = -np.log(min(coh, 1.0))
    h_coh = float(np.sqrt(max(gamma2, 0.0)) / chi)
    h_phase = float(np.sqrt(max(best["phase_var_slow"]["median"], 0.0)) / chi)
    f_wave = best["doppler_rms"]["median"]
    surface = {"grazing_deg": float(np.degrees(grazing)), "chi_rad_per_m": float(chi), "coherent": coh, "gamma2": float(gamma2),
               "rms_height_from_coherent_m": h_coh, "rms_height_from_phase_m": h_phase, "rms_frequency_hz": f_wave,
               "rms_vertical_speed_m_s": float(2.0 * np.pi * f_wave * h_coh),
               "source_heave_equivalent_m": float(np.sqrt(max(gamma2, 0.0)) / (k_top * (np.sin(grazing) + np.sin(np.radians(abs(surf[1]["launch_deg"])) if surf else 0.08))))}
    # t1 t2 / (t1 + t2) of the R1 surface bounce, from the straight unfolding of the path.
    st = prop["R1"]["straight"]
    t1 = st["surface"] * STATED["source_depth"] / (STATED["source_depth"] + STATED["R1"]["depth"])
    t2 = st["surface"] - t1
    t12 = t1 * t2 / (t1 + t2)
    surface["t12_ms"] = t12 * 1e3
    # The stacked fluctuating profile behind the surface arrival, and the tail fit with the pulse taken out.
    for rx, which in (("R1", "profile2"), ("R1", "profile1"), ("R2", "profile2"), ("R2", "profile1")):
        stack = np.array([p for p in get(rx, "chirp4", which) if p is not None])
        mean = stack.mean(axis=0)
        se = stack.std(axis=0) / np.sqrt(len(stack))
        pulse = theoretical_pulse()
        steady_all = np.array([p for p in get(rx, "chirp4", which.replace("profile", "steady")) if p is not None])
        steady = steady_all.mean(axis=0)
        sin2 = float(np.sin(grazing) ** 2)
        grid = np.geomspace(0.02e-3, 4e-3, 90)

        def facet_shape(lag, t):
            a = t / (1.0 + sin2)
            return timespread.facet_delay_density(lag, a, a * sin2)
        # Windows: to 3 ms behind the arrival, and to 0.9 ms (before the next steady echo at R1).
        fits = {}
        for wname, sl in (("3ms", slice(6, None)), ("0.9ms", slice(6, 24))):
            y = np.maximum(mean, 0.0)[sl]
            fits["exponential_" + wname] = cs.tail_decay(y, pulse[sl], FS, steady=steady[sl], decay_grid=grid)
            fits["facet_shape_" + wname] = cs.tail_decay(y, pulse[sl], FS, steady=steady[sl], decay_grid=grid, density=facet_shape)
            fits["exponential_pulse_only_" + wname] = cs.tail_decay(y, pulse[sl], FS, decay_grid=grid)
        q = len(stack) // 4
        parts = {name: [cs.tail_decay(np.maximum(stack[i * q:(i + 1) * q].mean(axis=0), 0.0)[6:24], pulse[6:24], FS, steady=steady_all[i * q:(i + 1) * q].mean(axis=0)[6:24],
                                      decay_grid=grid, **kw) for i in range(4)] for name, kw in (("exponential", {}), ("facet_shape", {"density": facet_shape}))}
        out[rx][which] = {"mean": mean.tolist(), "se": se.tolist(), "fits": fits, "quarters": parts, "pulse": pulse.tolist(), "steady": steady.tolist(),
                          "fluctuating_over_steady_energy": float(mean[6:24].sum() / steady[6:24].sum()), "n": int(len(stack))}
    # Does the tail fluctuate as the peak does? (spectra over time at the peak and 0.4-0.9 ms behind it, floor removed)
    for rx in ("R1", "R2"):
        for snd in ("chirp4", "chirp40", "prbs"):
            rows = [v for v in get(rx, snd, "delay_spectra", late) if v is not None]
            if not rows:
                continue
            nu = np.asarray(rows[0][0])
            peak, tail, floor = (np.mean([np.asarray(r[i]) for r in rows], axis=0) for i in (1, 2, 3))
            entry = {"nu": nu.tolist(), "peak": peak.tolist(), "tail": tail.tolist(), "floor": floor.tolist()}
            for name, s in (("peak", peak - floor), ("tail", tail - floor)):
                tot = s.sum()
                entry[name + "_rms_hz"] = float(np.sqrt(max((s * nu ** 2).sum(), 0.0) / tot)) if tot > 0 else float("nan")
                entry[name + "_share_above_0.5hz"] = float(s[np.abs(nu) > 0.5].sum() / tot) if tot > 0 else float("nan")
                entry[name + "_power"] = float(tot * (nu[1] - nu[0]))
            out[rx].setdefault("delay_spectra", {})[snd] = entry
    # The mean Doppler spectrum of the second arrival's fluctuation relative to the first (60 s soundings, 40 Hz soundings)
    for rx in ("R1", "R2"):
        for snd in ("prbs", "chirp40"):
            rows = [v for v in get(rx, snd, "spectrum", late) if v is not None]
            if rows:
                nu = np.asarray(rows[0][0])
                s = np.mean([np.asarray(r[1]) for r in rows], axis=0)
                fl = float(np.mean([r[2] for r in rows]))
                w = cr.spectrum_widths(nu, s, floor=fl)
                fold = s[nu > 0][: (nu < 0).sum() - 1] + s[nu < 0][::-1][: (nu > 0).sum()][: len(s[nu > 0][: (nu < 0).sum() - 1])]
                out[rx].setdefault("mean_spectrum", {})[snd] = {"nu": nu.tolist(), "s": s.tolist(), "floor": fl, "widths": {k: float(v) for k, v in w.items()},
                                                                 "peak_hz": float(nu[nu > 0][int(np.argmax(fold))])}
    # ---- two kinds of fluctuation behind the one coherent share ------------------------------------------------
    # The fluctuating energy around the arrival (0.5 ms before to 0.9 ms behind) is split by the fit into a part with
    # the steady profile's own shape (unspread: a mirror that heaves, or the sender) and a tail. Each is read as a
    # height through Gamma^2 = energy share; each has its own frequency (spectra at the peak and behind it).
    p2 = out["R1"]["profile2"]
    ef = p2["fluctuating_over_steady_energy"]
    q = p2["fits"]["exponential_0.9ms"]["tail_share"]
    ds4 = out["R1"].get("delay_spectra", {}).get("chirp4", {})
    surface["coherent_arrival_and_tail"] = float(1.0 / (1.0 + ef))
    f_long, f_short = float(ds4.get("peak_rms_hz", f_wave)), float(ds4.get("tail_rms_hz", f_wave))
    h_long, h_short = float(np.sqrt((1.0 - q) * ef) / chi), float(np.sqrt(q * ef) / chi)
    surface["unspread"] = {"energy_share": float((1.0 - q) * ef), "rms_height_m": h_long, "rms_frequency_hz": f_long,
                           "mean_square_slope_single_wave": float((h_long * (2.0 * np.pi * f_long) ** 2 / 9.81) ** 2)}
    surface["tail"] = {"energy_share": float(q * ef), "rms_height_m": h_short, "rms_frequency_hz": f_short, "rms_vertical_speed_m_s": float(2.0 * np.pi * f_short * h_short)}
    # The wind whose Pierson-Moskowitz height (acoustics.rms_wave_height) is the tail's, and what that wind's laws say of the rest.
    u = float(np.sqrt(h_short * 9.81 / (0.25 * 0.21)))
    surface["equivalent_wind_m_s"] = u
    mss = acoustics.mean_square_slope(u)
    surface["wind_law"] = {"mean_square_slope": mss, "rms_vertical_speed": acoustics.rms_surface_vertical_speed(u),
                           "rms_frequency_hz": acoustics.rms_surface_vertical_speed(u) / (2.0 * np.pi * max(acoustics.rms_wave_height(u), 1e-9))}
    surface["decay_ms_wind_law"] = {"facet": mss * (1.0 + np.sin(grazing) ** 2) * t12 * 1e3, "rule_of_thumb": 2.0 * mss * t12 * 1e3}
    surface["implied_mean_square_slope"] = {name: {"facet": f["decay_time_s"] / ((1.0 + np.sin(grazing) ** 2) * t12), "rule_of_thumb": f["decay_time_s"] / (2.0 * t12)}
                                            for name, f in p2["fits"].items() if "pulse_only" not in name}
    surface["model"] = model_through_estimators(surface, prop["R1"]["range_m"], c_src)
    out["surface"] = surface
    out["surface"] = surface

    # ---- do the two receivers see the same fluctuation? (a moving source would be common to both) ---------
    rho, rho_first, null = [], [], []
    series = {(rx, c): res[(rx, c)]["prbs"]["series"][0] for rx in ("R1", "R2") for c in late if (rx, c) in res and "series" in res[(rx, c)].get("prbs", {})}
    firsts = {(rx, c): res[(rx, c)]["prbs"]["first_phase"] for rx in ("R1", "R2") for c in late if (rx, c) in res and "first_phase" in res[(rx, c)].get("prbs", {})}

    def best_corr(a, b, lags=8):
        n = min(len(a), len(b))
        a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
        vals = [np.corrcoef(a[max(l, 0):n + min(l, 0)], b[max(-l, 0):n - max(l, 0)])[0, 1] for l in range(-lags, lags + 1)]
        return float(vals[lags]), float(vals[int(np.argmax(np.abs(vals)))])
    common = [c for c in late if ("R1", c) in series and ("R2", c) in series]
    for i, c in enumerate(common):
        rho.append(best_corr(series[("R1", c)], series[("R2", c)]))
        rho_first.append(best_corr(firsts[("R1", c)], firsts[("R2", c)]))
        other = common[(i + 7) % len(common)]
        null.append(best_corr(series[("R1", c)], series[("R2", other)]))
    out["cross_receiver"] = {"n": len(common), "pair_phase_zero_lag": med([r[0] for r in rho]), "pair_phase_best_lag_abs": med([abs(r[1]) for r in rho]),
                             "null_best_lag_abs": med([abs(r[1]) for r in null]), "first_phase_zero_lag": med([r[0] for r in rho_first]),
                             "first_phase_best_lag_abs": med([abs(r[1]) for r in rho_first])}

    # ---- noise ---------------------------------------------------------------------------------------------
    from gpusonar.sim import noise as noise_model
    from gpusonar.sim import noise_fit as nf
    noise: dict = {}
    for rx in ("R1", "R2"):
        rows = [res[(rx, c)]["noise"] for c in cycles if (rx, c) in res and "noise" in res[(rx, c)]]
        psd = np.array([r["psd"] for r in rows])
        f = rows[0]["frequency"]
        band = (f >= 4000.0) & (f <= 8000.0)
        level = 10.0 * np.log10(psd[:, band].mean(axis=1)) + 120.0
        medpsd = np.median(psd, axis=0)
        slope = np.polyfit(np.log2(f[band] / 1e3), 10.0 * np.log10(medpsd[band]), 1)[0]
        segs = [np.asarray(res[(rx, c)]["noise_series"], dtype=np.complex128)[None, :] for c in cycles if (rx, c) in res and "noise_series" in res[(rx, c)]]
        model = nf.fit(nf.Recording(segments=segs, sample_rate=FS, center_frequency=FC), nfft=256)
        noise[rx] = {"seconds": float(sum(r["seconds"] for r in rows)), "level_4_8_khz_db": med(level.tolist(), 1), "slope_db_per_octave": float(slope),
                     "frequency": f.tolist(), "psd_db": (10.0 * np.log10(medpsd) + 120.0).tolist(),
                     "psd_lo_db": (10.0 * np.log10(np.quantile(psd, 0.16, axis=0)) + 120.0).tolist(), "psd_hi_db": (10.0 * np.log10(np.quantile(psd, 0.84, axis=0)) + 120.0).tolist(),
                     "var_log": med([r["var_log"] for r in rows]), "q_ratio": med([r["q_ratio"] for r in rows]),
                     "fit": {"family": model.family, "tail_parameter": float(model.tail_parameter),
                             "excluded": {k: v for k, v in model.report.items() if k in ("blocks", "blocks_used", "transient_blocks", "periodic_blocks", "lines", "n_lines")}}}
        # the wind speed whose Wenz-type curve (sim/noise.py, no shipping) has the measured 4-8 kHz level: one fitted number
        winds = np.linspace(0.0, 15.0, 301)
        lv = [10.0 * np.log10(np.mean(10.0 ** (noise_model.wenz_spectrum_db(f[band], wind_speed=w, shipping=0.0) / 10.0))) for w in winds]
        noise[rx]["wenz_equivalent_wind_m_s"] = float(np.interp(noise[rx]["level_4_8_khz_db"]["median"], lv, winds)) if lv[0] < noise[rx]["level_4_8_khz_db"]["median"] < lv[-1] else None
        noise[rx]["wenz_floor_db"] = float(lv[0])
        noise[rx]["wenz_slope_db_per_octave"] = float(np.polyfit(np.log2(f[band] / 1e3), noise_model.wenz_spectrum_db(f[band], wind_speed=2.0, shipping=0.0), 1)[0])
    # coherence between the hydrophones (1.02 km and 0.27 km from the sender, about 1.2 km apart): the recordings start 0.526 s apart
    coh_rows = []
    for c in cycles:
        if ("R1", c) in res and ("R2", c) in res and "noise_series" in res[("R1", c)] and "noise_series" in res[("R2", c)]:
            lag = int(round((res[("R2", c)]["start_s"] - res[("R1", c)]["start_s"]) * FS))
            a, b = np.asarray(res[("R1", c)]["noise_series"])[max(lag, 0):], np.asarray(res[("R2", c)]["noise_series"])[max(-lag, 0):]
            n = min(len(a), len(b))
            if n >= 2048:
                coh_rows.append(np.stack([a[:n], b[:n]]).astype(np.complex128))
    if coh_rows:
        model = nf.fit(nf.Recording(segments=coh_rows, sample_rate=FS, center_frequency=FC), nfft=256, transient_threshold=None)
        g = np.abs(model.coherence())
        g01 = g[:, 0, 1] if g.ndim == 3 else g
        noise["coherence"] = {"mean_abs": float(np.mean(g01)), "max_abs": float(np.max(g01)), "floor": float(model.report.get("coherence_floor", np.nan)),
                              "seconds": float(sum(s.shape[1] for s in coh_rows) / FS)}
    out["noise"] = noise
    return out


# -- figures and the table ---------------------------------------------------------------------------------

def strong_arrivals(rows: list, within_db: float = 14.0, window: float = 0.025) -> list:
    """Bellhop arrivals within `within_db` of the strongest and `window` s of the first of those."""
    top = max(a["level_db"] for a in rows)
    keep = [a for a in rows if a["level_db"] > top - within_db]
    t0 = min(a["delay"] for a in keep)
    return [dict(a, after_ms=(a["delay"] - t0) * 1e3, rel_db=a["level_db"] - top) for a in keep if a["delay"] - t0 < window]


def figures(out: dict, res: dict, directory: Path, prefix: str = "channel_validation_smartocean") -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gpusonar.sim import noise as noise_model
    directory.mkdir(parents=True, exist_ok=True)
    made = []
    prop = out["propagation"]
    cycles = sorted({c for _, c in res})
    hours = {c: (res[("R1", c)]["start_s"] - 86400.0) / 3600.0 for c in cycles if ("R1", c) in res}     # hours from midnight UTC, 15 Nov
    credit = ATTRIBUTION

    # 1. the water, the paths, the arrivals -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax = axes[0, 0]
    for i, c in enumerate(prop["casts"]):
        ax.plot(c["speed"], c["depth"], color=f"C{i}", label=f"{c['date']} {c['start'][:5]} UTC")
        ax.plot(c["speed_down"], c["depth"], color=f"C{i}", lw=0.6, ls=":")
        ax.plot(c["speed_up"], c["depth"], color=f"C{i}", lw=0.6, ls=":")
    ax.axhline(STATED["source_depth"], color="k", lw=0.6)
    ax.text(ax.get_xlim()[0], STATED["source_depth"] - 0.8, " sender 20 m", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Sound speed (m/s), CTD.csv as given")
    ax.set_ylabel("Depth (m)")
    ax.set_title("The two casts (dotted: way down and way up)", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    if (Path(os.environ.get("GPUSONAR_SMARTOCEAN", "")) / "bathymetry.json").exists():
        bathy = json.loads((Path(os.environ["GPUSONAR_SMARTOCEAN"]) / "bathymetry.json").read_text())
        for rx, col in (("R1", "C0"), ("R2", "C1")):
            prof, length = path_profile(bathy, rx)
            p = np.array(prof)
            ax.plot(p[:, 0], p[:, 1], color=col, label=f"towards {rx}")
            ax.plot([length], [STATED[rx]["depth"]], marker="v", color=col)
        ax.plot([0], [STATED["source_depth"]], marker="o", color="k")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    ax.set_xlabel("Range from the sender (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Charted seabed along the two paths (10 m contours and soundings);\nsender and hydrophones at their stated depths", fontsize=9)
    for j, rx in enumerate(("R1", "R2")):
        ax = axes[0, 1 + j]
        prof = []
        for c in cycles:
            if (rx, c) in res and "bin1" in res[(rx, c)]["chirp4"]:
                p = np.concatenate([np.full(40, np.nan), res[(rx, c)]["profile4"]])
                b = res[(rx, c)]["chirp4"]["bin1"] + 40
                prof.append(p[b - 36:b + 400] / np.nanmax(p[b - 36:b + 400]))
        prof = np.array(prof)
        t = (np.arange(prof.shape[1]) - 36) / FS * 1e3
        ax.fill_between(t, 10 * np.log10(np.nanquantile(prof, 0.16, axis=0)), 10 * np.log10(np.nanquantile(prof, 0.84, axis=0)), color="0.8")
        ax.plot(t, 10 * np.log10(np.nanmedian(prof, axis=0)), color="k", lw=0.9, label=f"measured, {len(prof)} cycles (median, 16-84 %)")
        for name, col, off in (("isovelocity", "C2", -0.12), ("cast1", "C0", 0.0), ("cast2", "C1", 0.12)):
            rows = prop[rx].get("bellhop", {}).get(name)
            if rows:
                for a in strong_arrivals(rows):
                    ax.plot([a["after_ms"] + off] * 2, [-40, a["rel_db"]], color=col, lw=1.6, alpha=0.8)
                ax.plot([], [], color=col, lw=1.6, label=f"Bellhop, {name}")
        ax.set_xlim(-3, 25)
        ax.set_ylim(-40, 3)
        ax.set_xlabel("Delay after the first strong arrival (ms)")
        ax.set_ylabel("dB re the strongest")
        ax.set_title(f"{rx}, {prop[rx]['range_m']:.0f} m: 4 Hz chirp soundings; Bellhop over the charted\nseabed (assumed half-space; arrivals within 14 dB of the strongest)", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax = axes[1, 1 + j]
        d = [res[(rx, c)]["chirp4"].get("delay_ms") for c in cycles if (rx, c) in res]
        hh = [hours[c] for c in cycles if (rx, c) in res]
        ax.plot(hh, d, "k.", ms=4, label="measured")
        if rx == "R1":
            for name, col in (("isovelocity", "C2"), ("cast1", "C0"), ("cast2", "C1")):
                v = prop[rx].get("surface_minus_direct_ms", {}).get(name)
                if v:
                    ax.axhline(v, color=col, lw=1.0, label=f"Bellhop, {name}")
            ax.axhline(prop[rx]["straight"]["difference_ms"], color="C2", lw=1.0, ls="--", label="straight rays")
            ax.set_title("R1: surface reflection after the direct arrival", fontsize=9)
        else:
            ax.set_title("R2: spacing of the two strong arrivals", fontsize=9)
        ax.set_xlabel("Hours from midnight UTC, 15 November 2024")
        ax.set_ylabel("ms")
        ax.legend(fontsize=7)
    fig.suptitle(credit, fontsize=8)
    path = directory / f"{prefix}_arrivals.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    made.append(path)

    # 2. the surface reflection at R1 ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    late = [c for c in cycles if c >= 14 and ("R1", c) in res and "series" in res[("R1", c)].get("prbs", {})]
    ax = axes[0, 0]
    if late:
        var = [res[("R1", c)]["prbs"]["phase_var"] for c in late]
        c0 = late[int(np.argsort(var)[len(var) // 2])]
        ph, amp = res[("R1", c0)]["prbs"]["series"]
        t = np.arange(len(ph)) / 4.0
        ax.plot(t, ph, color="C0", lw=0.9, label="phase (rad)")
        ax.plot(t, amp - 1.0, color="C3", lw=0.9, label="amplitude - 1")
        ax.set_title(f"R1: surface reflection over direct arrival, cycle {c0}\n(the median one): 60 s of 0.5 s least-squares soundings", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
    ax = axes[0, 1]
    for snd, col, lab in (("prbs", "C0", "60 s, 4 Hz soundings"), ("chirp40", "C1", "10 s, 40 Hz soundings")):
        ms = out["R1"].get("mean_spectrum", {}).get(snd)
        if ms:
            nu, s = np.array(ms["nu"]), np.array(ms["s"])
            ax.semilogy(nu, s, color=col, lw=0.9, label=f"{lab}: rms {ms['widths']['rms']:.2f} Hz, w90/w50 {ms['widths']['shape']:.2f}")
            ax.axhline(ms["floor"], color=col, lw=0.6, ls=":")
    sf = out["surface"]
    ax.axvline(sf["wind_law"]["rms_frequency_hz"], color="0.4", lw=0.8, ls="--")
    ax.axvline(-sf["wind_law"]["rms_frequency_hz"], color="0.4", lw=0.8, ls="--")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("Hz")
    ax.set_title("Its spectrum, mean over cycles 14-80 (dotted: floors; dashed:\nrms frequency of a Pierson-Moskowitz sea of the same height)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[0, 2]
    p2 = out["R1"]["profile2"]
    t = (np.arange(len(p2["mean"])) - 12) / FS * 1e3
    ax.plot(t, 10 * np.log10(np.maximum(p2["steady"], 1e-9)), color="0.5", lw=0.9, label="steady part")
    ax.plot(t, 10 * np.log10(np.maximum(p2["mean"], 1e-9)), color="k", lw=1.2, label=f"fluctuating part, {p2['n']} cycles")
    ax.plot(t, 10 * np.log10(np.maximum(sf["unspread"]["energy_share"] / max(p2["fluctuating_over_steady_energy"], 1e-12) * np.array(p2["steady"])
                                        * np.sum(np.maximum(p2["mean"], 0.0)[6:24]) / np.sum(np.array(p2["steady"])[6:24]), 1e-9)),
            color="C2", lw=0.9, ls=":", label="unspread part of the fit")
    for name, col in (("facet", "C0"), ("rule_of_thumb", "C3")):
        ax.plot(t, 10 * np.log10(np.maximum(sf["model"][name]["profile"], 1e-9)), color=col, lw=1.0, ls="--",
                label=f"our model through the same soundings: {name.replace('_', ' ')}")
    ax.set_xlim(-1, 3)
    ax.set_ylim(-45, 3)
    ax.set_xlabel("Delay after the surface reflection (ms)")
    ax.set_ylabel("dB re its steady peak")
    ax.set_title(f"Behind the surface reflection (model: height and speed fitted to the\ntail, slope by Cox-Munk for the {sf['equivalent_wind_m_s']:.1f} m/s wind of that height)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    ds = out["R1"].get("delay_spectra", {}).get("chirp4")
    if ds:
        nu = np.array(ds["nu"])
        for key, col, lab in (("peak", "k", "at the peak"), ("tail", "C3", "0.4-0.9 ms behind it"), ("floor", "0.6", "before the direct arrival")):
            ax.semilogy(nu, ds[key], color=col, lw=1.0, label=lab + (f": rms {ds[key + '_rms_hz']:.2f} Hz" if key != "floor" else ""))
        ax.set_xlabel("Hz")
        ax.set_title("Spectra over time by delay (4 Hz chirp soundings, per delay bin,\nre the steady peak): the tail fluctuates faster than the peak", fontsize=9)
        ax.legend(fontsize=7)
    ax = axes[1, 1]
    for rx, col in (("R1", "C0"), ("R2", "C1")):
        cc = [c for c in cycles if (rx, c) in res and "coherent" in res[(rx, c)]["chirp4"]]
        ax.plot([hours[c] for c in cc], [1.0 - res[(rx, c)]["chirp4"]["coherent"] for c in cc], ".", color=col, ms=5,
                label=f"{rx}: " + ("surface reflection" if rx == "R1" else "second of the two arrivals"))
    ax.set_yscale("log")
    ax.set_xlabel("Hours from midnight UTC, 15 November 2024")
    ax.set_ylabel("1 - coherent share (16 s)")
    ax.set_title("Through the night", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1, 2]
    for rx, col in (("R1", "C0"), ("R2", "C1")):
        cc = [c for c in cycles if (rx, c) in res and "phase_var_slow" in res[(rx, c)]["chirp4"]]
        ax.plot([res[(rx, c)]["chirp4"]["phase_var_slow"] for c in cc], [res[(rx, c)]["chirp4"]["amplitude_var_slow"] for c in cc], ".", color=col, ms=5, label=rx)
    lim = [1e-4, 1e-1]
    ax.plot(lim, lim, color="k", lw=0.8, label="a complex Gaussian tap on a steady arrival")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Variance of the phase (rad^2)")
    ax.set_ylabel("Variance of the amplitude (relative)")
    ax.set_title("Phase against amplitude fluctuation of the second arrival", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle(credit, fontsize=8)
    path = directory / f"{prefix}_surface.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    made.append(path)

    # 3. delay spread and noise ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    ax = axes[0]
    for rx, col in (("R1", "C0"), ("R2", "C1")):
        pd = np.array([res[(rx, c)]["pdp"] / res[(rx, c)]["spread"]["energy_pa2"] for c in cycles if (rx, c) in res])
        m = np.median(pd, axis=0)
        sm = m[: len(m) // 24 * 24].reshape(-1, 24).mean(axis=1) * 24                       # energy per 2 ms
        t = (np.arange(len(sm)) * 24 + 12 - LEAD_BINS) / FS
        ax.semilogy(t, sm, color=col, lw=0.9, label=f"{rx}, {out['propagation'][rx]['range_m']:.0f} m: 90 % of the energy within {out[rx]['spread']['t10']['median'] * 1e3:.0f} ms, 99 % within {out[rx]['spread']['t20']['median'] * 1e3:.0f} ms")
        rows = out["propagation"][rx].get("bellhop", {}).get("cast1")
        if rows:
            t0 = min(a["delay"] for a in rows)
            e = np.array([10.0 ** (a["level_db"] / 10.0) for a in rows])
            ax.plot([a["delay"] - t0 for a in rows], e / e.sum(), "x", color=col, ms=4)
    ax.set_xlim(-0.05, 2.2)
    ax.set_ylim(1e-8, 1)
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Share of the received energy per 2 ms")
    ax.set_title("Whole response (0.745 s chirps 3 s apart); crosses: Bellhop arrivals\n(2-D, charted seabed, cast 1), each as its share of their sum", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    for rx, col in (("R1", "C0"), ("R2", "C1")):
        n = out["noise"][rx]
        f = np.array(n["frequency"])
        ok = (f > 2600.0) & (f < 11400.0)
        ax.fill_between(f[ok] / 1e3, np.array(n["psd_lo_db"])[ok], np.array(n["psd_hi_db"])[ok], color=col, alpha=0.25)
        ax.plot(f[ok] / 1e3, np.array(n["psd_db"])[ok], color=col, lw=1.0, label=f"{rx}: {n['level_4_8_khz_db']['median']:.1f} dB in 4-8 kHz, {n['seconds']:.0f} s")
    ff = np.linspace(2600.0, 11400.0, 100)
    for w, ls in ((0.0, ":"), (2.0, "--"), (5.0, "-")):
        ax.plot(ff / 1e3, noise_model.wenz_spectrum_db(ff, wind_speed=w, shipping=0.0), color="k", lw=0.8, ls=ls, label=f"simulator's Wenz-type curve, {w:.0f} m/s, no shipping")
    ax.set_xlabel("kHz")
    ax.set_ylabel("dB re 1 uPa^2/Hz")
    ax.set_title("Noise before each cycle's first transmission (median and 16-84 % over cycles;\n94 Pa per full scale, as stated)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    for rx, col in (("R1", "C0"), ("R2", "C1")):
        cc = [c for c in cycles if (rx, c) in res and "noise" in res[(rx, c)]]
        f = res[(rx, cc[0])]["noise"]["frequency"]
        band = (f >= 4000.0) & (f <= 8000.0)
        ax.plot([hours[c] for c in cc], [10 * np.log10(res[(rx, c)]["noise"]["psd"][band].mean()) + 120.0 for c in cc], ".", color=col, ms=5, label=rx)
    ax.set_xlabel("Hours from midnight UTC, 15 November 2024")
    ax.set_ylabel("4-8 kHz level, dB re 1 uPa^2/Hz")
    ax.set_title("Noise level through the night", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle(credit, fontsize=8)
    path = directory / f"{prefix}_spread_noise.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    made.append(path)
    return made


def fmt(m: dict, digits: int = 3, scale: float = 1.0) -> str:
    if not m or not m.get("n"):
        return "-"
    return f"{m['median'] * scale:.{digits}g} ({m['lo'] * scale:.{digits}g} to {m['hi'] * scale:.{digits}g})"


def markdown(out: dict) -> str:
    prop, sf, nz = out["propagation"], out["surface"], out["noise"]
    lines = ["# SFI Smart Ocean dataset, 4-8 kHz: measured against our propagation and time spreading", "",
             ATTRIBUTION + ". Made by `tools/channel_smartocean.py`; nothing of the dataset is in this repository.", "",
             f"{out['n_recordings']} recordings (80 cycles, two hydrophones). Values are medians over cycles with the 16 % and 84 % points in brackets.", "",
             "## Arrivals: straight rays, Bellhop with the measured profiles, and the measurement", "",
             "Stated: positions, depths, the two casts. From the chart: seabed along the paths. Assumed: the seabed half-space (1600 m/s, 1.8, 0.5 dB/wavelength), "
             "a 1.5 m tripod at R1. Fitted: nothing.", "",
             "| | straight rays | Bellhop, isovelocity | Bellhop, cast 1 (17:05) | Bellhop, cast 2 (07:07) | measured |", "| --- | --- | --- | --- | --- | --- |"]
    r1 = prop["R1"]
    smd = r1.get("surface_minus_direct_ms", {})
    gz = r1.get("surface_grazing_deg", {})
    lines.append(f"| R1 ({r1['range_m']:.0f} m): surface reflection after direct, ms | {r1['straight']['difference_ms']:.2f} | {smd.get('isovelocity', float('nan')):.2f} | {smd.get('cast1', float('nan')):.2f} | "
                 f"{smd.get('cast2', float('nan')):.2f} | {fmt(out['R1']['chirp4']['delay_ms'], 4)} |")
    lines.append(f"| R1: grazing angle at the surface, degrees | {r1['straight']['grazing_deg']:.1f} | {gz.get('isovelocity', float('nan')):.1f} | {gz.get('cast1', float('nan')):.1f} | {gz.get('cast2', float('nan')):.1f} | not measured |")
    srd = r1.get("surface_re_direct_db", {})
    lines.append(f"| R1: surface reflection re direct, dB | {-20 * np.log10(r1['straight']['surface'] / r1['straight']['direct']):.1f} | {srd.get('isovelocity', float('nan')):.1f} | "
                 f"{srd.get('cast1', float('nan')):.1f} | {srd.get('cast2', float('nan')):.1f} | {fmt(out['R1']['chirp4']['level2_db'], 2)} |")
    for name in ("isovelocity", "cast1", "cast2"):
        rows = prop["R2"].get("bellhop", {}).get(name)
        if rows:
            txt = "; ".join(f"+{a['after_ms']:.2f} ms {a['rel_db']:.0f} dB S{a['n_surface']}B{a['n_bottom']}" for a in strong_arrivals(rows, 8.0, 0.006))
            lines.append(f"| R2 ({prop['R2']['range_m']:.0f} m), Bellhop {name}: arrivals within 8 dB, first 6 ms | | | | | {txt} |")
    lines.append(f"| R2: the two strong arrivals: spacing ms; second re first dB | {prop['R2']['straight']['difference_ms']:.2f} (direct, surface) | | | | "
                 f"{fmt(out['R2']['chirp4']['delay_ms'], 3)}; {fmt(out['R2']['chirp4']['level2_db'], 2)} |")
    lines += ["", f"First arrival at R2 minus first arrival at R1 (recorder time stamps): {fmt(out['travel_time_difference_s'], 5)} s.", ""]
    lines += ["## The surface reflection at R1 (the one arrival that is a single surface bounce and stands alone)", "",
              "| Quantity | 4 Hz chirp soundings, 16 s, all cycles | least squares on the 60 s sequence, 4 Hz, cycles 14-80 | 40 Hz chirp soundings, 10 s, cycles 14-80 |", "| --- | --- | --- | --- |"]
    for label, key, digits, scale in (("soundings' fluctuation floor, dB re direct", "floor_db", 3, 1.0), ("coherent share", "coherent", 4, 1.0),
                                      ("phase variance, rad^2 (below 1 Hz, white part removed)", "phase_var_slow", 3, 1.0),
                                      ("relative amplitude variance (same)", "amplitude_var_slow", 3, 1.0), ("scintillation index of the arrival's energy", "si2", 3, 1.0),
                                      ("rms width of the fluctuation's spectrum, Hz", "doppler_rms", 3, 1.0), ("shape w90/w50 (Gaussian 2.44)", "doppler_shape", 3, 1.0)):
        lines.append(f"| {label} | " + " | ".join(fmt(out["R1"][s][key], digits, scale) for s in ("chirp4", "prbs", "chirp40")) + " |")
    lines.append("| phase variance, upper over lower half band; in brackets what this estimator reads for a pure change of path length (1 for none) | "
                 + " | ".join(f"{out['R1'][s]['band_ratio']['value']:.2f} +- {out['R1'][s]['band_ratio']['se']:.2f} ({out['band_ratio_expected'][s]:.2f})" if "band_ratio" in out["R1"][s] else "-"
                              for s in ("chirp4", "prbs", "chirp40")) + " |")
    un, tl = sf["unspread"], sf["tail"]
    lines += ["", f"Coherent share of the arrival and its tail together (0.5 ms before to 0.9 ms behind): {sf['coherent_arrival_and_tail']:.4f}. The fit below splits the fluctuating energy in two:", "",
              "| Part | share of the arrival's energy | rms frequency, Hz | as an rms surface height at %.1f degrees, cm |" % sf["grazing_deg"], "| --- | --- | --- | --- |",
              f"| unspread (the steady profile's own shape) | {un['energy_share']:.4f} | {un['rms_frequency_hz']:.2f} | {un['rms_height_m'] * 100:.2f} |",
              f"| tail | {tl['energy_share']:.4f} | {tl['rms_frequency_hz']:.2f} | {tl['rms_height_m'] * 100:.2f} |", "",
              f"The sender hangs from a floating platform: {sf['source_heave_equivalent_m'] * 100:.1f} cm rms of heave would do to the peak what the unspread part does. "
              f"A single wave of the unspread part's height and frequency has a mean-square slope of {un['mean_square_slope_single_wave']:.1e}: no tail from it under either rule.", "",
              f"A Pierson-Moskowitz sea of the tail's height is a {sf['equivalent_wind_m_s']:.1f} m/s wind; that wind's laws give an rms frequency of {sf['wind_law']['rms_frequency_hz']:.2f} Hz "
              f"(measured behind the arrival: {tl['rms_frequency_hz']:.2f}) and a Cox-Munk mean-square slope of {sf['wind_law']['mean_square_slope']:.4f}.", ""]
    p2 = out["R1"]["profile2"]
    lines += ["### The tail behind it", "",
              f"Fluctuating intensity around the arrival, mean of {p2['n']} cycles, fitted as (steady profile) x a + (pulse convolved with a tail of mean excess delay T) x b:", "",
              "| Tail shape | window | T, ms | tail's share of the fluctuating energy | misfit (without a tail) |", "| --- | --- | --- | --- | --- |"]
    for name, f in p2["fits"].items():
        if "pulse_only" in name:
            continue
        shape, win = name.rsplit("_", 1)
        lines.append(f"| {shape.replace('_', ' ')} | {win.replace('ms', ' ms')} behind | {f['decay_time_s'] * 1e3:.2f} | {f['tail_share']:.2f} | {f['misfit']:.2f} ({f['misfit_no_tail']:.1f}) |")
    q = p2["quarters"]
    lines += ["", "Four quarters of the cycles, 0.9 ms window: exponential T = " + ", ".join(f"{x['decay_time_s'] * 1e3:.2f}" for x in q["exponential"]) + " ms; facet shape T = "
              + ", ".join(f"{x['decay_time_s'] * 1e3:.2f}" for x in q["facet_shape"]) + " ms.", "",
              f"The two rules for this bounce (t1 t2 / (t1 + t2) = {sf['t12_ms']:.1f} ms) with that slope: facet {sf['decay_ms_wind_law']['facet']:.2f} ms, rule of thumb "
              f"{sf['decay_ms_wind_law']['rule_of_thumb']:.2f} ms. Our model with that surface (height and vertical speed of the tail part: two fitted numbers; slope from the wind law), "
              "sampled like the soundings and passed through the same fit:", "",
              "| | T read with an exponential, 0.9 ms window | with the facet shape, 0.9 ms | with an exponential, 3 ms | mean excess delay put in |", "| --- | --- | --- | --- | --- |"]
    for rule in ("facet", "rule_of_thumb"):
        m = sf["model"][rule]
        lines.append(f"| model, {rule.replace('_', ' ')} | {m['exponential_0.9ms']['decay_time_s'] * 1e3:.2f} | {m['facet_shape_0.9ms']['decay_time_s'] * 1e3:.2f} | {m['exponential_3ms']['decay_time_s'] * 1e3:.2f} | {m['closed_form_ms']:.2f} |")
    f = p2["fits"]
    lines.append(f"| measured | {f['exponential_0.9ms']['decay_time_s'] * 1e3:.2f} | {f['facet_shape_0.9ms']['decay_time_s'] * 1e3:.2f} | {f['exponential_3ms']['decay_time_s'] * 1e3:.2f} | |")
    ims = sf["implied_mean_square_slope"]
    lines += ["", "Read the other way, the measured T is a mean-square slope of " + "; ".join(
        f"{v['facet']:.4f} (facet) or {v['rule_of_thumb']:.4f} (rule of thumb) from the {k.replace('_', ' ')} fit" for k, v in ims.items() if k in ("exponential_0.9ms", "exponential_3ms")) + ".", ""]
    ds = out["R1"].get("delay_spectra", {})
    if ds:
        lines += ["| Fluctuation over time (floor removed) | rms frequency at the peak, Hz | 0.4-0.9 ms behind, Hz | share above 0.5 Hz at the peak | behind |", "| --- | --- | --- | --- | --- |"]
        for s, lab in (("chirp4", "4 Hz chirp soundings"), ("chirp40", "40 Hz chirp soundings"), ("prbs", "least squares, 60 s")):
            if s in ds:
                d = ds[s]
                lines.append(f"| {lab} | {d['peak_rms_hz']:.2f} | {d['tail_rms_hz']:.2f} | {d['peak_share_above_0.5hz']:.2f} | {d['tail_share_above_0.5hz']:.2f} |")
        lines.append("")
    lines += ["## R2 (1.02 km): the second of the two strong arrivals against the first", "",
              "| Quantity | 4 Hz chirp soundings | least squares, 60 s |", "| --- | --- | --- |"]
    for label, key, digits in (("coherent share", "coherent", 4), ("phase variance, rad^2", "phase_var_slow", 3), ("relative amplitude variance", "amplitude_var_slow", 3),
                               ("rms width, Hz", "doppler_rms", 3), ("soundings' floor, dB re the first", "floor_db", 3)):
        lines.append(f"| {label} | " + " | ".join(fmt(out["R2"][s][key], digits) for s in ("chirp4", "prbs")) + " |")
    cx = out["cross_receiver"]
    lines += ["", f"Same transmissions at both hydrophones ({cx['n']} cycles, 60 s each): correlation of the first arrivals' phase (line removed) between R1 and R2 "
              f"{fmt(cx['first_phase_zero_lag'], 2)}; of the second-against-first phase {fmt(cx['pair_phase_zero_lag'], 2)} (largest over +-2 s of lag, absolute: "
              f"{fmt(cx['pair_phase_best_lag_abs'], 2)}; the same between different cycles: {fmt(cx['null_best_lag_abs'], 2)}).", "",
              "## Delay spread against range", "", "| | R1, 274 m | R2, 1020 m |", "| --- | --- | --- |"]
    for label, key, scale, digits in (("90 % of the energy within, ms", "t10", 1e3, 3), ("99 % within, ms", "t20", 1e3, 3), ("99.9 % within, ms", "t30", 1e3, 3),
                                      ("rms delay spread, ms", "rms_spread_s", 1e3, 3), ("share in the first 10 ms", "first_10ms_share", 1.0, 2)):
        lines.append(f"| {label} | {fmt(out['R1']['spread'][key], digits, scale)} | {fmt(out['R2']['spread'][key], digits, scale)} |")
    lv = out["levels"]
    lines += ["", "## Levels", "", "| | R1 | R2 |", "| --- | --- | --- |",
              f"| first strong arrival, dB re 1 uPa rms | {lv['R1']['first_db']:.1f} | {lv['R2']['first_db']:.1f} |",
              f"| loss from the stated source level (183 measured, 181 expected), dB | {lv['R1']['loss_measured_sl']:.1f}, {lv['R1']['loss_expected_sl']:.1f} | {lv['R2']['loss_measured_sl']:.1f}, {lv['R2']['loss_expected_sl']:.1f} |",
              f"| Bellhop, strongest arrival, casts 1 and 2 (plus {lv['absorption_db_per_km']:.2f} dB/km), dB | {lv['R1'].get('bellhop', '-')} | {lv['R2'].get('bellhop', '-')} |",
              f"| spherical spreading to the slant range, dB | {lv['R1']['spherical']:.1f} | {lv['R2']['spherical']:.1f} |",
              f"| whole response re the first strong arrival, dB | {lv['R1']['total_re_first_db']:.1f} | {lv['R2']['total_re_first_db']:.1f} |", "",
              "## Noise (before each cycle's first transmission)", "", "| | R1 | R2 |", "| --- | --- | --- |"]
    lines.append(f"| signal-free time, s | {nz['R1']['seconds']:.0f} | {nz['R2']['seconds']:.0f} |")
    lines.append(f"| 4-8 kHz spectrum level, dB re 1 uPa^2/Hz | {fmt(nz['R1']['level_4_8_khz_db'], 3)} | {fmt(nz['R2']['level_4_8_khz_db'], 3)} |")
    lines.append(f"| slope over 4-8 kHz, dB/octave (simulator's wind noise: {nz['R1']['wenz_slope_db_per_octave']:.1f}) | {nz['R1']['slope_db_per_octave']:.1f} | {nz['R2']['slope_db_per_octave']:.1f} |")
    lines.append("| wind speed whose Wenz-type curve has that level, m/s (one fitted number) | "
                 + " | ".join("%.1f" % nz[r]["wenz_equivalent_wind_m_s"] if nz[r]["wenz_equivalent_wind_m_s"] is not None else "above 15" for r in ("R1", "R2")) + " |")
    lines.append(f"| var(log envelope) (complex Gaussian: 0.411) | {fmt(nz['R1']['var_log'], 3)} | {fmt(nz['R2']['var_log'], 3)} |")
    lines.append(f"| envelope q99.95 / q50 (complex Gaussian: 3.31) | {fmt(nz['R1']['q_ratio'], 3)} | {fmt(nz['R2']['q_ratio'], 3)} |")
    lines.append(f"| tools/fit_noise.py family | {nz['R1']['fit']['family']} {nz['R1']['fit']['tail_parameter']:.2f} | {nz['R2']['fit']['family']} {nz['R2']['fit']['tail_parameter']:.2f} |")
    if "coherence" in nz:
        c = nz["coherence"]
        lines += ["", f"Coherence between the two hydrophones (about 1.2 km apart, {c['seconds']:.0f} s in common): mean magnitude {c['mean_abs']:.3f}, largest {c['max_abs']:.3f}; "
                  f"the estimator's floor for independent channels is {c['floor']:.3f}."]
    return "\n".join(lines) + "\n"


def levels(out: dict) -> dict:
    """Absolute levels: the first strong arrival in dB re 1 uPa rms (the soundings are for a unit-rms
    probe envelope and the recording in pascals, so |h|^2 / 2 is the path's mean-square pressure),
    the loss from the stated source levels, Bellhop's strongest arrival with Ainslie-McColm
    absorption for the casts' water, and spherical spreading."""
    from gpusonar import acoustics
    prop = out["propagation"]
    lv: dict = {}
    try:
        alpha = 1e3 * float(acoustics.absorption_ainslie_mccolm(FC, temperature=11.5, salinity=32.0, depth=20.0))   # dB/km
    except (AttributeError, TypeError):
        alpha = float("nan")
    lv["absorption_db_per_km"] = alpha
    for rx in ("R1", "R2"):
        p = out[rx]["level1_pa2"]["median"]
        first = 10.0 * np.log10(p / 2.0) + 120.0
        slant = float(np.hypot(prop[rx]["range_m"], STATED[rx]["depth"] - STATED["source_depth"]))
        row = {"first_db": first, "loss_measured_sl": cs.SODAC_SOURCE_LEVEL_DB["measured"] - first, "loss_expected_sl": cs.SODAC_SOURCE_LEVEL_DB["expected"] - first,
               "spherical": 20.0 * np.log10(slant), "total_re_first_db": 10.0 * np.log10(out[rx]["spread"]["energy_pa2"]["median"] / p)}
        bh = prop[rx].get("bellhop", {})
        if bh:
            row["bellhop"] = ", ".join("%.1f" % (-max(a["level_db"] for a in bh[k]) + (alpha * slant / 1e3 if np.isfinite(alpha) else 0.0)) for k in ("cast1", "cast2") if k in bh)
        lv[rx] = row
    return lv


def model_through_estimators(surface: dict, range_m: float, sound_speed: float, n_time: int = 1600, seed: int = 3,
                             time_spread: dict | None = None, rules=("facet", "rule_of_thumb")) -> dict:
    """Our time spreading for the R1 geometry (direct and surface-reflected straight rays; surface
    of the rms height and vertical speed fitted to the tail, slope of the equivalent wind's Cox-Munk law),
    sampled like the 4 Hz soundings and passed through the same estimators, for each decay rule:
    what the tail fit reads when the model is the truth."""
    from gpusonar.sim import timespread
    from gpusonar.sim.propagation import ImageSourceProvider
    grid = np.geomspace(0.02e-3, 4e-3, 90)
    sin2 = float(np.sin(np.radians(surface["grazing_deg"])) ** 2)

    def facet_shape(lag, t):
        a = t / (1.0 + sin2)
        return timespread.facet_delay_density(lag, a, a * sin2)
    out = {}
    for rule in rules:
        cfg = {"surface": {"rms_height": surface["tail"]["rms_height_m"], "mean_square_slope": surface["wind_law"]["mean_square_slope"],
                           "rms_vertical_speed": surface["tail"]["rms_vertical_speed_m_s"]},
               "decay_time": rule, "taps": 24, "doppler_lines": 48, "seed": seed}
        cfg.update(time_spread or {})                  # further `time_spread` keys (tail_quantile, taps): the `tail` subcommand
        prov = timespread.wrap(ImageSourceProvider(sound_speed, surface_z=0.0, bottom_z=None, max_order=1), cfg, frequency=FC)
        k = 256
        ch = cr.model_channel(prov, [0.0, 0.0, STATED["source_depth"]], np.array([[range_m, 0.0, STATED["R1"]["depth"]]]), n_time=n_time, fs_time=4.0, fs_delay=FS,
                              n_delay=k, fc=FC, bandwidth=FS, lead=LEAD)
        h = np.fft.ifft(np.fft.fft(ch.h[:, 0, :], axis=1) * cs.band_window(k, FS, BAND, "hann")[None, :], axis=1)
        rng = np.random.default_rng(seed)
        h = h + 10.0 ** (-38.0 / 20.0) * np.abs(h).max() / np.sqrt(2.0) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))   # the soundings' floor
        st = pair_statistics(as_channel(h, 4.0), "R1")
        y, steady, pulse = np.maximum(st["profile2"], 0.0), st["steady2"], theoretical_pulse()
        fit_e = cs.tail_decay(y[6:24], pulse[6:24], FS, steady=steady[6:24], decay_grid=grid)
        fit_f = cs.tail_decay(y[6:24], pulse[6:24], FS, steady=steady[6:24], decay_grid=grid, density=facet_shape)
        fit_e3 = cs.tail_decay(y[6:], pulse[6:], FS, steady=steady[6:], decay_grid=grid)
        out[rule] = {"coherent": st["coherent"], "phase_var": st["phase_var_slow"], "amplitude_var": st["amplitude_var_slow"], "doppler_rms": st["doppler"]["rms"],
                     "exponential_0.9ms": fit_e, "facet_shape_0.9ms": fit_f, "exponential_3ms": fit_e3, "profile": y.tolist(),
                     "closed_form_ms": surface["decay_ms_wind_law"][rule]}
    return out


def stage_bathymetry(args) -> int:
    """bathymetry.json (depth every 10 m along the lines from the sender to R1 and R2) from the
    dataset's file geodatabase, unpacked at `args.gdb`. Needs pyogrio, pyproj, shapely and scipy
    (not dependencies of the suite: use a virtual environment). Layers read: dybdekurve (depth
    contours), dybdepunkt and grunne (soundings), kystkontur (coastline, depth 0)."""
    import pyproj
    import shapely
    from pyogrio.raw import read
    from scipy.interpolate import griddata
    tr = pyproj.Transformer.from_crs(4326, 25832, always_xy=True)
    xy = {k: tr.transform(v[1], v[0]) for k, v in cs.SODAC_POSITIONS.items()}
    xs, ys = [v[0] for v in xy.values()], [v[1] for v in xy.values()]
    bbox = (min(xs) - 400, min(ys) - 400, max(xs) + 400, max(ys) + 400)
    pts = []
    for layer in ("dybdekurve", "dybdepunkt", "grunne", "kystkontur"):
        meta, _, geom, fields = read(args.gdb, layer=layer, bbox=bbox)
        names = list(meta["fields"])
        for i in range(len(geom) if geom is not None else 0):
            depth = 0.0 if layer == "kystkontur" else float(fields[names.index("dybde")][i])
            if np.isfinite(depth):
                pts += [(p[0], p[1], depth) for p in shapely.get_coordinates(shapely.from_wkb(geom[i]))]
    pts = np.array(pts)
    out = {"source": "Kartverket Sjokart - Dybdedata (the dataset's Basisdata_46_Vestland_25832_Dybdedata_FGDB): contours, soundings and coastline within 400 m "
                     "of the instruments, linear interpolation every 10 m along the lines from the sender to each hydrophone"}
    for rx in ("R1", "R2"):
        a, b = np.array(xy["TX"]), np.array(xy[rx])
        length = float(np.hypot(*(b - a)))
        s = np.linspace(-30.0, length + 30.0, int((length + 60.0) / 10.0) + 1)
        d = griddata(pts[:, :2], pts[:, 2], a[None] + (s[:, None] / length) * (b - a)[None], method="linear")
        out[rx] = {"range": s.tolist(), "depth": [None if not np.isfinite(v) else round(float(v), 1) for v in d], "length": length}
        print(rx, f"{length:.1f} m; depth at the sender {np.interp(0.0, s, d):.1f} m, at the hydrophone {np.interp(length, s, d):.1f} m")
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)
    return 0


TAIL_OPTIONS = [("three mean decay times (as shipped), 24 taps", {}),
                ("tail_quantile 0.9, 24 taps", {"tail_quantile": 0.9}),
                ("tail_quantile 0.99, 24 taps", {"tail_quantile": 0.99}),
                ("tail_quantile 0.99, 48 taps", {"tail_quantile": 0.99, "taps": 48}),
                ("tail_quantile 0.999, 48 taps", {"tail_quantile": 0.999, "taps": 48})]


def stage_tail(args) -> int:
    """How far the model's taps reach against the two windows of the measured tail fit (R1's surface
    reflection): the model of `model_through_estimators` under each entry of TAIL_OPTIONS, from the
    numbers `analyse` left in the cache (no recordings needed). Two seeds of the scattered taps."""
    cache = Path(args.cache)
    out = json.loads((cache / "analysis_numbers.json").read_text())
    sf, fits = out["surface"], out["R1"]["profile2"]["fits"]
    c_src = out["propagation"]["speed_at_source"]
    measured = {k: 1e3 * fits[k]["decay_time_s"] for k in ("exponential_0.9ms", "facet_shape_0.9ms", "exponential_3ms")}
    lines = ["# The reach of the facet tail against the measured tail of R1's surface reflection", "", ATTRIBUTION + ".", "",
             "Made by `tools/channel_smartocean.py tail`: our time spreading for the R1 geometry with the surface fitted to the tail (two numbers: height, vertical speed; the slope is "
             f"Cox-Munk's for the wind whose height that is, {sf['wind_law']['mean_square_slope']:.4f}, not fitted), sampled like the 4 Hz soundings and passed through the same fits as the "
             "measurement, with the taps spanning three mean decay times (as shipped) or the stated share of the shape's energy. Mean excess delay T read by each fit, ms; two seeds of the taps.", "",
             "| Model | rule | T, exponential fit, first 0.9 ms | T, facet-shape fit, 0.9 ms | T, exponential fit, 3 ms | mean excess delay put in | last tap at, ms |", "| --- | --- | --- | --- | --- | --- | --- |",
             f"| measured | | {measured['exponential_0.9ms']:.2f} | {measured['facet_shape_0.9ms']:.2f} | {measured['exponential_3ms']:.2f} | | |"]
    res = {"measured_ms": measured}
    from gpusonar.sim import timespread
    sin2 = float(np.sin(np.radians(sf["grazing_deg"])) ** 2)
    for label, extra in TAIL_OPTIONS:
        for rule in (("facet", "rule_of_thumb") if extra.get("tail_quantile") in (None, 0.99) and extra.get("taps") is None else ("facet",)):
            runs = [model_through_estimators(sf, out["propagation"]["R1"]["range_m"], c_src, seed=seed, time_spread=extra, rules=(rule,))[rule] for seed in (3, 4)]
            put = sf["decay_ms_wind_law"][rule]
            a = put / (1.0 + sin2) if rule == "facet" else 0.5 * put
            b = a * sin2 if rule == "facet" else a
            q = extra.get("tail_quantile")
            last = 3.0 * put if q is None else float(timespread.tail_span(a + b, 2.0 * (a * a + b * b), q))
            cell = lambda key: " / ".join(f"{1e3 * r[key]['decay_time_s']:.2f}" for r in runs)     # noqa: E731
            lines.append(f"| {label} | {rule} | {cell('exponential_0.9ms')} | {cell('facet_shape_0.9ms')} | {cell('exponential_3ms')} | {put:.2f} | {last:.2f} |")
            res.setdefault(label, {})[rule] = {k: [1e3 * r[k]["decay_time_s"] for r in runs] for k in measured}
            print(lines[-1], flush=True)
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    print(text)
    return 0


def stage_analyse(args) -> int:
    cache = Path(args.cache)
    res = collect(cache)
    out = derive(res, data_dir())
    out["levels"] = levels(out)
    (cache / "analysis_numbers.json").write_text(json.dumps(out, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))
    text = markdown(out)
    if args.figures:
        for p in figures(out, res, Path(args.figures)):
            print("wrote", p)
        text += "\nFigures: `channel_validation_smartocean_arrivals.png`, `channel_validation_smartocean_surface.png`, `channel_validation_smartocean_spread_noise.png`.\n"
    if args.out:
        Path(args.out).write_text(text)
        print("wrote", args.out)
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("sound", help="recordings -> soundings (restartable; one recording in memory at a time)")
    a.add_argument("--cache", required=True)
    a.add_argument("--only", nargs="*")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=stage_sound)
    a = sub.add_parser("analyse", help="soundings -> numbers, table, figures")
    a.add_argument("--cache", required=True)
    a.add_argument("--figures", default=None)
    a.add_argument("--out", default=None)
    a.set_defaults(func=stage_analyse)
    a = sub.add_parser("tail", help="the model's tap reach (time_spread.tail_quantile) against the measured tail fits; needs only analysis_numbers.json")
    a.add_argument("--cache", required=True)
    a.add_argument("--out", default=None)
    a.add_argument("--json", default=None)
    a.set_defaults(func=stage_tail)
    a = sub.add_parser("bathymetry", help="the unpacked depth geodatabase -> bathymetry.json (needs pyogrio, pyproj, shapely)")
    a.add_argument("--gdb", required=True, help="the .gdb directory")
    a.add_argument("--out", required=True)
    a.set_defaults(func=stage_bathymetry)
    args = ap.parse_args(argv)
    if args.stage in ("bathymetry", "tail"):
        return args.func(args)
    if data_dir() is None and not (Path(args.cache) / "bb" / "meta.json").exists():
        print("GPUSONAR_SMARTOCEAN is not set to a directory with the dataset (and the cache has no recordings)", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
