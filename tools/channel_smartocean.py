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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("sound", help="recordings -> soundings (restartable; one recording in memory at a time)")
    a.add_argument("--cache", required=True)
    a.add_argument("--only", nargs="*")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=stage_sound)
    args = ap.parse_args(argv)
    if data_dir() is None and not (Path(args.cache) / "bb" / "meta.json").exists():
        print("GPUSONAR_SMARTOCEAN is not set to a directory with the dataset (and the cache has no recordings)", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
