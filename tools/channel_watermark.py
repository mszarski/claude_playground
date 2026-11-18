#!/usr/bin/env python3
"""The Watermark channels (FFI) against our time spreading: what can be tested without knowing
the sea, at 14 kHz and at 35 kHz.

    tools/fetch_channels.py <dir> --watermark          (770 MB zip from www.ffi.no, unpacked)
    tools/channel_watermark.py --data <dir> [--figures docs/figures] [--out docs/figures/channel_validation_watermark.md]

Watermark: P. A. van Walree, F.-X. Socheleau, R. Otnes, T. Jenserud, "The Watermark benchmark
for underwater acoustic modulation schemes", IEEE J. Ocean. Eng. 42 (2017) 1007-1018 (read:
the open copy hal-01630274); BCH1: F.-X. Socheleau, A. Pottier, C. Laot, "Watermark: BCH1
dataset description", hal-01404491 (read). "Free download" from FFI; the archive carries no
licence text, so nothing of it is redistributed here. Channels used:

  NOF1  Oslofjord, June; 750 m, 10 m of water; source and hydrophone bottom-mounted; 10-18 kHz;
        60 soundings of 32.9 s, 400 s apart; snapshots 7.8 Hz apart
  NCS1  Norwegian shelf, June; 540 m, 80 m; both bottom-mounted; 10-18 kHz; 60 soundings of
        32.6 s, 600 s apart; snapshots 31.4 Hz apart
  BCH1  Brest harbour, May; 800 m, 20 m; source at 2 m, hydrophones at 3, 4, 5, 6 m; 32.5-37.5
        kHz; one sounding of 59.4 s; snapshots 9.8 Hz apart
  (KAU1, KAU2: towed source off Kauai, 4-8 kHz; not used: moving source, not isovelocity.)

These are correlative soundings (a matched filter per probe period), not adaptive estimates:
there is no tracker to low-pass the fluctuations, which was what made the UACR "Green" file
useless. The price is aliasing: Doppler beyond half the snapshot rate folds back.

Not stated for any channel (paper, BCH1 report, readme; the user manual, FFI-rapport 16/01378,
is served from a host this session could not reach): wind, sea state, sound-speed profile,
seabed, the heights of the bottom-mounted units above the seabed. So no arrival can be given a
grazing angle with confidence, and nothing here uses one. What is tested needs no geometry:

1. The link between an arrival's coherent share c and the rms Doppler width s of its scattered
   part. sim/timespread.py has c = exp(-Gamma^2) and s = (chi v / 2 pi) / sqrt(1 - c) with
   Gamma = chi h, so
        s sqrt((1 - c) / ln(1 / c)) = v / (2 pi h)
   whatever the grazing angle, the number of bounces or the frequency: the rms frequency of the
   surface waves, one number for all the arrivals of a sounding. If the model is right the
   left side is the same for every partly coherent arrival of a sounding although c and s
   differ; `channel_replay.wave_frequency`.
2. The shape of the Doppler spectrum of scattered arrivals (Gaussian or Lorentzian).
3. The scintillation index of fully scattered arrivals (1 for complex Gaussian taps).
4. The Doppler width in the upper against the lower half of the band (proportional to
   frequency for fully scattered arrivals).
Nothing is fitted.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gpusonar.sim import channel_replay as cr  # noqa: E402

CHANNELS = {"NOF1": dict(fc=14e3, place="Oslofjord, 750 m, 10 m water, bottom-mounted"),
            "NCS1": dict(fc=14e3, place="Norwegian shelf, 540 m, 80 m water, bottom-mounted"),
            "BCH1": dict(fc=35e3, place="Brest harbour, 800 m, 20 m water, suspended")}


def files(data: Path, name: str) -> list:
    for root in (data / "Watermark" / "input" / "channels", data / "input" / "channels", data):
        found = sorted((root / name / "mat").glob(f"{name}_*.mat"))
        if found:
            return found
    return []


def analyse(ch: cr.TimeVaryingChannel, max_arrivals: int = 12, threshold_db: float = -20.0) -> list:
    """Every arrival within `threshold_db` of the strongest: coherent share, rms width of the
    scattered part, shape, scintillation, and the upper/lower band width ratio. No phase reference:
    the mean Doppler shift has been removed by the sounder and the platforms do not move."""
    p = (np.abs(ch.h[:, 0]) ** 2).mean(axis=0)
    floor = float(np.median(p))
    line = 2.5 * ch.fs_time / min(128, ch.n_time)                 # the coherent line: two and a half Doppler bins
    rows = []
    for b in sorted(cr.find_arrivals(p, n=max_arrivals, min_separation=2, threshold_db=threshold_db)):
        try:
            st = cr.arrival_statistics(ch, 0, b, floor=floor, tail_bins=4, line_half_width=line, nfft=128)
        except ValueError:
            continue
        sub = cr.subband_statistics(ch, 0, b, before=1, after=3, nfft=128, line_half_width=line)
        rows.append({"delay_ms": b / ch.fs_delay * 1e3, "level_db": 10.0 * np.log10(p[b] / p.max()), "coherent": st["coherent_fraction"],
                     "doppler_rms": st["doppler"]["rms"], "shape": st["doppler"]["shape"], "scintillation": st["scintillation_index"],
                     "band_ratio": sub["doppler_ratio"], "frequency_ratio": sub["frequency_ratio"],
                     "wave_frequency": cr.wave_frequency(st["coherent_fraction"], st["doppler"]["rms"])})
    return rows


def _gsd(x) -> float:
    x = np.asarray([v for v in x if np.isfinite(v) and v > 0])
    return float(np.exp(np.log(x).std())) if len(x) > 1 else float("nan")


def summarise(name: str, soundings: list) -> dict:
    """soundings: list of `analyse` results (one per file)."""
    partial = [[r for r in rows if 0.03 < r["coherent"] < 0.9 and r["doppler_rms"] > 0] for rows in soundings]
    partial = [rows for rows in partial if len(rows) >= 3]
    rough = [r for rows in soundings for r in rows if r["coherent"] < 0.05 and r["doppler_rms"] > 0]
    scattered = [r for rows in soundings for r in rows if r["coherent"] < 0.9 and r["doppler_rms"] > 0]
    out = {"name": name, "soundings": len(soundings), "arrivals": int(sum(len(r) for r in soundings)), "partial_soundings": len(partial),
           "partial_arrivals": int(sum(len(r) for r in partial)), "rough_arrivals": len(rough)}
    if partial:
        out["wave_frequency_median"] = float(np.median([np.median([r["wave_frequency"] for r in rows]) for rows in partial]))
        out["invariant_scatter"] = float(np.median([_gsd([r["wave_frequency"] for r in rows]) for rows in partial]))
        out["width_scatter"] = float(np.median([_gsd([r["doppler_rms"] for r in rows]) for rows in partial]))
        out["share_range"] = [float(min(r["coherent"] for rows in partial for r in rows)), float(max(r["coherent"] for rows in partial for r in rows))]
        # Does the width rise as the coherent share falls, as 1 / sqrt(1 - c) x sqrt(ln(1/c))? Rank correlation within soundings.
        from scipy.stats import spearmanr
        rho = [spearmanr([r["coherent"] for r in rows], [r["doppler_rms"] for r in rows])[0] for rows in partial if len(rows) >= 4]
        out["rank_correlation_width_share"] = float(np.nanmedian(rho)) if rho else float("nan")
    if scattered:
        sh = np.array([r["shape"] for r in scattered])
        out["shape"] = [float(np.median(sh)), float(np.quantile(sh, 0.1)), float(np.quantile(sh, 0.9))]
    if rough:
        si = np.array([r["scintillation"] for r in rough])
        out["scintillation"] = [float(np.median(si)), float(np.quantile(si, 0.1)), float(np.quantile(si, 0.9)), float(np.mean(si > 1.3))]
        br = np.array([r["band_ratio"] for r in rough if np.isfinite(r["band_ratio"]) and 0.3 < r["band_ratio"] < 3.0])
        out["band_ratio"] = [float(br.mean()), float(br.std() / np.sqrt(len(br))), float(np.mean([r["frequency_ratio"] for r in rough])), int(len(br))]
        out["rough_width"] = [float(np.median([r["doppler_rms"] for r in rough])), float(max(r["doppler_rms"] for r in rough))]
    return out


def report(results: dict, coverage: dict) -> str:
    lines = ["# Watermark channels: tests that need no geometry and no sea state", "",
             "Nothing is fitted. Wind, sea state, sound speed and seabed are not stated for any channel.", "",
             "| Channel | Carrier | Doppler coverage | Soundings, arrivals | rms wave frequency v / (2 pi h): median | scatter of it across the arrivals of a sounding | scatter of the raw width | coherent shares |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        if "wave_frequency_median" in r:
            lines.append(f"| {name} | {CHANNELS[name]['fc'] / 1e3:g} kHz | +-{coverage[name] / 2:.1f} Hz | {r['partial_soundings']}, {r['partial_arrivals']} partly coherent | "
                         f"{r['wave_frequency_median']:.2f} Hz | x{r['invariant_scatter']:.2f} | x{r['width_scatter']:.2f} | {r['share_range'][0]:.2f} to {r['share_range'][1]:.2f} |")
        else:
            lines.append(f"| {name} | {CHANNELS[name]['fc'] / 1e3:g} kHz | +-{coverage[name] / 2:.1f} Hz | {r['soundings']}, {r['arrivals']} | no partly coherent arrivals | | | |")
    for name, r in results.items():
        if "rank_correlation_width_share" in r:
            lines.append("")
            lines.append(f"{name}: rank correlation between the width and the coherent share across the arrivals of a sounding, median "
                         f"{r['rank_correlation_width_share']:+.2f} (the model: negative; the width should rise by up to 1.9 as the share falls from 0.9 to 0.03).")
    lines += ["", "| Channel | Scattered arrivals: Doppler shape w90/w50, median (10-90 %) | Fully scattered arrivals | their scintillation index, median (10-90 %) | share above 1.3 | "
              "their rms width, median (largest) | upper / lower band width ratio | frequency ratio |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        sh = f"{r['shape'][0]:.2f} ({r['shape'][1]:.2f}-{r['shape'][2]:.2f})" if "shape" in r else "-"
        if "scintillation" in r:
            s, b, w = r["scintillation"], r["band_ratio"], r["rough_width"]
            lines.append(f"| {name} | {sh} | {r['rough_arrivals']} | {s[0]:.2f} ({s[1]:.2f}-{s[2]:.2f}) | {100 * s[3]:.0f} % | {w[0]:.2f} Hz ({w[1]:.2f}) | "
                         f"{b[0]:.3f} +-{b[1]:.3f} (n={b[3]}) | {b[2]:.3f} |")
        else:
            lines.append(f"| {name} | {sh} | none | | | | | |")
    return "\n".join(lines) + "\n"


def figure(all_rows: dict, results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    colors = {"NOF1": "C0", "NCS1": "C1", "BCH1": "C2"}
    ax = axes[0]
    c = np.linspace(0.02, 0.95, 200)
    for name, soundings in all_rows.items():
        rows = [r for s in soundings for r in s if 0.02 < r["coherent"] < 0.95 and r["doppler_rms"] > 0]
        if not rows:
            continue
        ax.plot([r["coherent"] for r in rows], [r["doppler_rms"] for r in rows], marker=".", ls="none", color=colors[name], ms=4, alpha=0.5,
                label=f"{name}, {CHANNELS[name]['fc'] / 1e3:g} kHz")
        if "wave_frequency_median" in results[name]:
            ax.plot(c, results[name]["wave_frequency_median"] * np.sqrt(np.log(1.0 / c) / (1.0 - c)), color=colors[name], lw=1.2)
    ax.set_xlabel("Coherent share of the arrival")
    ax.set_ylabel("rms Doppler width of its scattered part (Hz)")
    ax.set_yscale("log")
    ax.set_title("Lines: v / (2 pi h) sqrt(ln(1/c) / (1 - c)), one v/h per channel (its median)", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    for name, soundings in all_rows.items():
        si = [r["scintillation"] for s in soundings for r in s if r["coherent"] < 0.05]
        if si:
            ax.hist(np.clip(si, 0, 3), bins=np.linspace(0, 3, 31), histtype="step", color=colors[name], label=f"{name} (n={len(si)})")
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_xlabel("Scintillation index of fully scattered arrivals (line: complex Gaussian)")
    ax.legend(fontsize=8)
    ax = axes[2]
    for name, soundings in all_rows.items():
        br = [r["band_ratio"] for s in soundings for r in s if r["coherent"] < 0.05 and 0.3 < r["band_ratio"] < 3.0]
        if br:
            ax.hist(br, bins=np.linspace(0.6, 1.9, 40), histtype="step", color=colors[name], label=f"{name}: mean {np.mean(br):.3f}")
            ax.axvline(results[name]["band_ratio"][2], color=colors[name], lw=0.8, ls="--")
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_xlabel("Doppler width, upper / lower half of the band (dashed: frequency ratio)")
    ax.legend(fontsize=8)
    fig.suptitle("Measured: Watermark (FFI; van Walree et al. 2017). No geometry, no sea state, nothing fitted", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("GPUSONAR_WATERMARK_DIR", ""))
    ap.add_argument("--figures", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    data = Path(args.data) if args.data else None
    if data is None or not any(files(data, n) for n in CHANNELS):
        print("No Watermark channels: run tools/fetch_channels.py <dir> --watermark and pass --data <dir> (or set GPUSONAR_WATERMARK_DIR).")
        return 2
    all_rows, results, coverage = {}, {}, {}
    for name in CHANNELS:
        soundings = []
        for f in files(data, name):
            ch = cr.load_watermark(str(f))
            coverage[name] = ch.fs_time
            soundings.append(analyse(ch))
        if soundings:
            all_rows[name], results[name] = soundings, summarise(name, soundings)
    text = report(results, coverage)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    if args.figures:
        path = Path(args.figures) / "channel_validation_watermark.png"
        figure(all_rows, results, path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
