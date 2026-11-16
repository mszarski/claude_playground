#!/usr/bin/env python3
"""Which frequency and source level take the Popoto-board array to a given range?

For a ship-sized target at `--range` (default 12 km) this works out, against frequency,
the source level that leaves `--snr` dB (default 13) at the beamformer output:

    SL = SNR + 2 TL - TS + NL - AG - 10 log10(T)

with absorption (Ainslie-McColm) and ambient noise (four-term fit) at each frequency.
Two array rules: a fixed number of hydrophones at half-wavelength spacing (the array gets
longer as the frequency drops), and a fixed aperture filled at half-wavelength spacing
(the count rises with frequency). Spherical spreading is the conservative curve; where
Bellhop is installed (GPUSONAR_BELLHOP) its transmission loss for shallow water is used as
well, at a few frequencies.

Writes docs/design/popoto_range_study.png and .json. CPU only.
Usage: tools/popoto_range_study.py [--range 12000] [--ts 25] [--pulse 1.0] [--depth 100]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gpusonar import acoustics  # noqa: E402
from gpusonar.sim import find_bellhop  # noqa: E402
from gpusonar.sim.noise import wenz_spectrum_db  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
C = 1500.0


def required_source_level(f, range_m, n, ts, pulse, snr, wind=5.0, tl_one_way=None):
    """Source level (dB re 1 uPa at 1 m) for `snr` dB out, at frequency f (Hz, array)."""
    f = np.asarray(f, dtype=float)
    alpha = acoustics.absorption(f, 12.0, 35.0, 50.0, model="ainslie_mccolm")
    tl = 20 * np.log10(range_m) + alpha * range_m if tl_one_way is None else tl_one_way
    noise = wenz_spectrum_db(f, wind, 0.5)
    return snr + 2 * tl - ts + noise - 10 * np.log10(n) - 10 * np.log10(pulse)


def acoustic_power_w(source_level_db, directivity_index_db=0.0):
    """Radiated acoustic power for a source level: SL = 170.8 + 10 log10(P) + DI."""
    return float(10.0 ** ((source_level_db - 170.8 - directivity_index_db) / 10.0))


def bellhop_loss(frequency, range_m, depth, bottom):
    """Incoherent one-way loss (dB) from Bellhop for isovelocity water over a fluid bottom,
    absorption added along the direct distance."""
    from gpusonar.sim.bellhop import OceanEnvironment, run_bellhop
    env = OceanEnvironment(depth=depth, sound_speed=C, **bottom)
    arr = run_bellhop(env, frequency, 10.0, [min(0.5 * depth, 50.0)], [range_m])
    tl = float(arr.transmission_loss(coherent=False)[0, 0])
    return tl + float(acoustics.absorption(frequency, 12.0, 35.0, 50.0, model="ainslie_mccolm")) * range_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", type=float, default=12000.0)
    ap.add_argument("--ts", type=float, default=25.0)
    ap.add_argument("--pulse", type=float, default=1.0)
    ap.add_argument("--snr", type=float, default=13.0)
    ap.add_argument("--depth", type=float, default=100.0)
    ap.add_argument("--aperture", type=float, default=3.0)
    args = ap.parse_args()
    f = np.geomspace(1e3, 40e3, 120)
    out = {"range_m": args.range, "target_strength_db": args.ts, "pulse_s": args.pulse, "snr_db": args.snr,
           "fixed_count": {}, "fixed_aperture": {}, "bellhop": []}
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True, sharey=True)
    for n, col in zip((8, 16, 32, 64), (BLUE, ORANGE, AQUA, YELLOW)):
        sl = required_source_level(f, args.range, n, args.ts, args.pulse, args.snr)
        i = int(np.argmin(sl))
        out["fixed_count"][str(n)] = {"best_frequency_hz": float(f[i]), "source_level_db": float(sl[i]),
                                      "array_length_m": float(n * C / (2 * f[i])),
                                      "acoustic_power_w_omni": acoustic_power_w(sl[i])}
        a1.plot(f / 1e3, sl, color=col, lw=2, label=f"{n} hydrophones")
        a1.plot(f[i] / 1e3, sl[i], "o", color=col, ms=8, mec=SURFACE, mew=2)
        a1.annotate(f"{sl[i]:.0f} dB at {f[i] / 1e3:.1f} kHz\n({n * C / (2 * f[i]):.1f} m long)", (f[i] / 1e3, sl[i]),
                    textcoords="offset points", xytext=(6, -22), fontsize=8)
    a1.set_title(f"Fixed hydrophone count, half-wavelength spacing: source level for {args.snr:.0f} dB at "
                 f"{args.range / 1e3:.0f} km", loc="left", fontsize=10)
    for length, col in zip((1.5, 3.0, 6.0), (BLUE, ORANGE, AQUA)):
        n = np.maximum(np.floor(2 * length * f / C), 1.0)
        sl = required_source_level(f, args.range, n, args.ts, args.pulse, args.snr)
        i = int(np.argmin(sl))
        out["fixed_aperture"][str(length)] = {"best_frequency_hz": float(f[i]), "source_level_db": float(sl[i]),
                                              "hydrophones": int(n[i]),
                                              "acoustic_power_w_omni": acoustic_power_w(sl[i])}
        a2.plot(f / 1e3, sl, color=col, lw=2, label=f"{length:.1f} m aperture")
        a2.plot(f[i] / 1e3, sl[i], "o", color=col, ms=8, mec=SURFACE, mew=2)
        a2.annotate(f"{sl[i]:.0f} dB at {f[i] / 1e3:.1f} kHz\n({int(n[i])} hydrophones)", (f[i] / 1e3, sl[i]),
                    textcoords="offset points", xytext=(6, -22), fontsize=8)
    # The shipped 27.5 kHz array (configs/sonars/popoto_array.yaml) for comparison: what 12 km would
    # take in its own band, and how far its own source level reaches.
    from gpusonar.sonar import SonarModel
    model = SonarModel.load("popoto_array")
    wf = model.config["transmit"]["waveform"]
    fc, n_cur = 0.5 * (float(wf["f_start"]) + float(wf["f_stop"])), len(model.positions)
    sl_cur = float(wf["source_level_db"])
    need = float(required_source_level(fc, args.range, n_cur, args.ts, float(wf["duration"]), args.snr))
    rr = np.geomspace(500.0, 30000.0, 400)
    ok = [float(required_source_level(fc, r, n_cur, args.ts, float(wf["duration"]), args.snr)) <= sl_cur for r in rr]
    out["current_array"] = {"frequency_hz": fc, "hydrophones": n_cur, "pulse_s": float(wf["duration"]), "source_level_db": sl_cur,
                            "source_level_needed_at_range_db": need,
                            "reach_at_its_source_level_m": float(rr[np.flatnonzero(ok)[-1]]) if any(ok) else 0.0}
    a2.set_title("Fixed aperture, filled at half-wavelength spacing", loc="left", fontsize=10)
    if find_bellhop() is not None:
        bottom = dict(bottom_speed=1600.0, bottom_density=1.8, bottom_attenuation=0.5)
        fb = np.array([2e3, 4e3, 8e3, 12e3])
        for fr in fb:
            tl = bellhop_loss(float(fr), args.range, args.depth, bottom)
            n = max(np.floor(2 * args.aperture * fr / C), 1.0)
            sl = float(required_source_level(fr, args.range, n, args.ts, args.pulse, args.snr, tl_one_way=tl))
            out["bellhop"].append({"frequency_hz": float(fr), "one_way_loss_db": tl, "hydrophones": int(n),
                                   "source_level_db": sl, "acoustic_power_w_omni": acoustic_power_w(sl),
                                   "spherical_one_way_loss_db": float(20 * np.log10(args.range) + acoustics.absorption(
                                       fr, 12.0, 35.0, 50.0, model="ainslie_mccolm") * args.range)})
        a2.plot(fb / 1e3, [b["source_level_db"] for b in out["bellhop"]], "s", color=INK, ms=7,
                label=f"{args.aperture:.1f} m aperture, Bellhop loss in {args.depth:.0f} m of water over sand")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("centre frequency (kHz)")
        ax.set_ylim(160, 240)
        ax.legend(frameon=False, fontsize=8, loc="upper center")
    a1.set_ylabel("source level needed (dB re 1 µPa at 1 m)")
    fig.suptitle(f"Target strength +{args.ts:.0f} dB, {args.pulse:.1f} s pulse, ambient noise at 5 m/s wind; "
                 "curves use spherical spreading (conservative), noise-limited", fontsize=10)
    dest = ROOT / "docs" / "design"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "popoto_range_study.png", dpi=120)
    (dest / "popoto_range_study.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
