#!/usr/bin/env python3
"""The decay law of a basin's late reverberation, fitted to the two measured energy-time curves.

    tools/basin_reverberation_fit.py --cache <dir> [--out docs/figures/basin_reverberation_fit.md] [--figure docs/figures/basin_reverberation_fit.png]

<dir> is the cache of `tools/channel_smartocean.py sound` (its `soundings/` and `analysis_numbers.json`): the
SFI Smart Ocean dataset, 4-8 kHz (van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh, IEEE DataPort 2025,
doi:10.21227/3aa6-4k33, CC BY 4.0). Nothing of the dataset is written anywhere: only the fitted numbers and a
figure of the median curves.

For each hydrophone (R1 at 274 m, R2 at 1020 m) and each of the 80 cycles: the intensity profile of the 0.745 s
chirp soundings to 3 s (`lfm3000_total`), the level of its last tenth subtracted as noise, summed from the end:
the share of the energy still to come at each delay. The median over the cycles is fitted, from an onset to where
it has fallen to a floor, in dB, every 5 ms, by

    exponential        share(t) = A exp(-(t - onset) / T)                      2 numbers
    power law          share(t) = A (1 + (t - onset) / t0)^(1 - n)            3 numbers
    two exponentials   share(t) = A (w exp(-u / T1) + (1 - w) exp(-u / T2))    4 numbers

and the result is what `basin_reverberation` (sim/basin.py) takes: energy_db = A / (1 - A) in dB (the tail over
everything else), onset_s, decay_s = T. Also: how much of the late energy is the same from chirp to chirp
(five chirps 3 s apart: with a part a that repeats, a part f that does not and noise n, the power of the mean is
|a|^2 + (F + N) / 5 and the mean power |a|^2 + F + N), and the tail's energy against the source's at 1 m.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

FS, LEAD = 12000.0, 60                       # the soundings' delay rate and the bin of the first arrival (tools/channel_smartocean.py)
ATTRIBUTION = ("Data: SFI Smart Ocean dataset (van Walree, Otnes, Tomasi, Henriksen, Danre, Bergh, IEEE DataPort 2025, doi:10.21227/3aa6-4k33, CC BY 4.0); "
               "soundings by `tools/channel_smartocean.py`. One fjord, one night, 4-8 kHz.")
RANGE = {"R1": 274, "R2": 1020}


def _two(u, p):
    w = np.clip(p[3], 1e-6, 1.0 - 1e-6)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return p[0] + 10.0 * np.log10(w * np.exp(-u / abs(p[1])) + (1.0 - w) * np.exp(-u / abs(p[2])))


FORMS = {"exponential": (lambda u, p: p[0] - 10.0 / np.log(10.0) * u / p[1], [-4.0, 0.15], 2),
         "power law": (lambda u, p: p[0] + 10.0 * (1.0 - p[2]) * np.log10(1.0 + u / abs(p[1])), [-4.0, 0.5, 4.0], 3),
         "two exponentials": (_two, [-4.0, 0.05, 0.17, 0.3], 4)}


def curves(cache: Path, rx: str) -> dict:
    """Energy still to come against delay after the first arrival (median and 16 / 84 % over the cycles), and
    the share of the energy in delay bands that repeats over the five chirps."""
    edges = [0.0, 0.010, 0.050, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    later, frozen = [], []
    for path in sorted(glob.glob(str(cache / "soundings" / (rx + "_*.npz")))):
        z = np.load(path)
        total, coherent = z["lfm3000_total"].astype(float), z["lfm3000_coherent"].astype(float)
        noise = np.median(total[int(0.9 * len(total)):])
        p = (total - noise)[LEAD - 12:]
        c = np.cumsum(p[::-1])[::-1]
        later.append(c / c[0])
        row = []
        for a, b in zip(edges[:-1], edges[1:]):
            sl = slice(LEAD + int(a * FS), LEAD + int(b * FS))
            row.append((5.0 * coherent[sl].sum() - total[sl].sum()) / 4.0 / (total[sl].sum() - noise * (sl.stop - sl.start)))
        frozen.append(row)
    later, frozen = np.array(later), np.array(frozen)
    t = (np.arange(later.shape[1]) - 12) / FS
    return {"t": t, "median": np.median(later, axis=0), "lo": np.quantile(later, 0.16, axis=0), "hi": np.quantile(later, 0.84, axis=0), "n": len(later),
            "frozen_edges": edges, "frozen": np.median(frozen, axis=0), "frozen_lo": np.quantile(frozen, 0.16, axis=0), "frozen_hi": np.quantile(frozen, 0.84, axis=0)}


def fit(cv: dict, onset: float, floor_db: float) -> dict:
    from scipy.optimize import least_squares
    db = 10.0 * np.log10(np.maximum(cv["median"], 1e-12))
    idx = np.flatnonzero((cv["t"] >= onset) & (db >= floor_db))[::60]
    u, y = cv["t"][idx] - onset, db[idx]
    out = {"to_s": float(cv["t"][idx[-1]]), "points": int(len(idx))}
    for name, (fn, p0, n_par) in FORMS.items():
        r = least_squares(lambda p: fn(u, p) - y, p0)
        out[name] = {"parameters": [float(v) for v in r.x], "n_parameters": n_par, "rms_db": float(np.sqrt(np.mean(r.fun**2))), "max_db": float(np.abs(r.fun).max())}
    return out


def run(cache: Path) -> dict:
    from gpusonar.sim import basin
    numbers = json.loads((cache / "analysis_numbers.json").read_text())
    res = {}
    for rx in ("R1", "R2"):
        cv = curves(cache, rx)
        fits = {f"{int(onset * 1e3)} ms, {int(floor)} dB": fit(cv, onset, floor) for onset in (0.010, 0.020, 0.050) for floor in (-30.0, -40.0)}
        main = fits["10 ms, -30 dB"]["exponential"]
        a = 10.0 ** (main["parameters"][0] / 10.0)
        energy_db, decay = 10.0 * np.log10(a / (1.0 - a)), main["parameters"][1]
        lv, sp = numbers["levels"][rx], numbers[rx]["spread"]
        tail_level = lv["first_db"] + lv["total_re_first_db"] + 10.0 * np.log10(a)                # dB re 1 uPa rms: the tail's energy as a level
        res[rx] = {"range_m": RANGE[rx], "cycles": cv["n"], "fits": fits,
                   "basin_reverberation": {"energy_db": float(energy_db), "onset_s": 0.010, "decay_s": float(decay)},
                   "tail_share_of_all": float(a),
                   "measured_share_later_than_10ms": float(cv["median"][np.searchsorted(cv["t"], 0.010)]),
                   "model_ms": {k: float(1e3 * basin.delay_of_share(s, 0.010, decay, energy_db)) for k, s in (("t90", 0.1), ("t99", 0.01), ("t99.9", 0.001))},
                   "measured_ms": {"t90": 1e3 * sp["t10"]["median"], "t99": 1e3 * sp["t20"]["median"], "t99.9": 1e3 * sp["t30"]["median"]},
                   "model_rms_spread_ms": 1e3 * basin.rms_delay_spread(0.010, decay, energy_db), "measured_rms_spread_ms": 1e3 * sp["rms_spread_s"]["median"],
                   "tail_re_source_at_1m_db": {"source level 183 dB (measured)": float(tail_level - 183.0), "source level 181 dB (expected)": float(tail_level - 181.0)},
                   "frozen_share": [{"from_ms": 1e3 * a_, "to_ms": 1e3 * b_, "median": float(m), "lo": float(lo), "hi": float(hi)}
                                    for a_, b_, m, lo, hi in zip(cv["frozen_edges"][:-1], cv["frozen_edges"][1:], cv["frozen"], cv["frozen_lo"], cv["frozen_hi"])],
                   "curve": {"t": cv["t"][::120].tolist(), "median_db": (10.0 * np.log10(np.maximum(cv["median"][::120], 1e-12))).tolist()}}
    return res


def markdown(res: dict) -> str:
    lines = ["# Late reverberation of a fjord: the decay law fitted to the measured energy-time curves", "", ATTRIBUTION, "",
             "Made by `tools/basin_reverberation_fit.py`. The curve fitted is the share of the received energy still to come, against delay after the first arrival "
             "(median of the cycles; 0.745 s chirps, 5 kHz of band). Fits in dB, every 5 ms, from the onset to the floor named.", "",
             "| Hydrophone | fitted from, down to | to delay, s | exponential (2 numbers): rms / max misfit, dB; T, s | power law (3): rms / max; t0, s; exponent | two exponentials (4): rms / max; T1, T2, s; weight |",
             "| --- | --- | --- | --- | --- | --- |"]
    for rx, r in res.items():
        for key, f in r["fits"].items():
            e, p, t = f["exponential"], f["power law"], f["two exponentials"]
            lines.append(f"| {rx}, {r['range_m']} m | {key} | {f['to_s']:.2f} | {e['rms_db']:.2f} / {e['max_db']:.2f}; {e['parameters'][1]:.3f} | "
                         f"{p['rms_db']:.2f} / {p['max_db']:.2f}; {abs(p['parameters'][1]):.3g}; {p['parameters'][2]:.3g} | "
                         f"{t['rms_db']:.2f} / {t['max_db']:.2f}; {abs(t['parameters'][1]):.3g}, {abs(t['parameters'][2]):.3g}; {np.clip(t['parameters'][3], 0, 1):.2f} |")
    lines += ["", "The exponential from 10 ms to -30 dB, as `basin_reverberation` takes it, and what it then reads against the measurement "
              "(the modelled arrivals as one arrival at delay 0):", "",
              "| | R1, 274 m | R2, 1020 m |", "| --- | --- | --- |"]
    r1, r2 = res["R1"], res["R2"]
    row = lambda label, fn: lines.append(f"| {label} | {fn(r1)} | {fn(r2)} |")      # noqa: E731
    row("`energy_db` (tail over the rest)", lambda r: f"{r['basin_reverberation']['energy_db']:+.1f}")
    row("`onset_s`", lambda r: f"{r['basin_reverberation']['onset_s']:.3f}")
    row("`decay_s`", lambda r: f"{r['basin_reverberation']['decay_s']:.3f}")
    row("tail's share of all the energy: fitted exponential (measured later than 10 ms)", lambda r: f"{r['tail_share_of_all']:.3f} ({r['measured_share_later_than_10ms']:.3f})")
    for k, label in (("t90", "90 %"), ("t99", "99 %"), ("t99.9", "99.9 %")):
        row(f"{label} of the energy within, ms: model (measured)", lambda r, k=k: f"{r['model_ms'][k]:.0f} ({r['measured_ms'][k]:.0f})")
    row("rms delay spread, ms: model (measured)", lambda r: f"{r['model_rms_spread_ms']:.0f} ({r['measured_rms_spread_ms']:.0f})")
    row("tail's energy re the source's at 1 m, dB (source level 183 measured / 181 expected)",
        lambda r: " / ".join(f"{v:.1f}" for v in r["tail_re_source_at_1m_db"].values()))
    lines += ["", "Share of the energy in a band of delays that is the same on five chirps 3 s apart (noise removed; median of the cycles, 16 to 84 %). "
              "The source hung from a floating platform; the hydrophones stood on the seabed.", "",
              "| Delay, ms | R1, 274 m | R2, 1020 m |", "| --- | --- | --- |"]
    for a, b in zip(r1["frozen_share"], r2["frozen_share"]):
        lines.append(f"| {a['from_ms']:.0f}-{a['to_ms']:.0f} | {a['median']:.2f} ({a['lo']:.2f} to {a['hi']:.2f}) | {b['median']:.2f} ({b['lo']:.2f} to {b['hi']:.2f}) |")
    return "\n".join(lines) + "\n"


def figure(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    for rx, colour in (("R1", "#2a78d6"), ("R2", "#eb6834")):
        r = res[rx]
        t = np.array(r["curve"]["t"])
        ax.plot(t, r["curve"]["median_db"], color=colour, lw=1.8, label=f"{rx}, {r['range_m']} m: measured (median of {r['cycles']})")
        b = r["basin_reverberation"]
        a = r["tail_share_of_all"]
        u = np.linspace(b["onset_s"], 1.6, 50)
        ax.plot(u, 10.0 * np.log10(a) - 10.0 / np.log(10.0) * (u - b["onset_s"]) / b["decay_s"], color=colour, lw=1.0, ls="--",
                label=f"exponential, T = {b['decay_s']:.3f} s, {b['energy_db']:+.1f} dB re the rest")
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(-50.0, 1.0)
    ax.set_xlabel("delay after the first arrival, s")
    ax.set_ylabel("energy still to come, dB re all")
    ax.grid(True, color="#d9d8d2", lw=0.6)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Late reverberation in a fjord, 4-8 kHz: SFI Smart Ocean dataset (CC BY 4.0), our soundings", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figure", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    cache = Path(args.cache)
    if not (cache / "soundings").is_dir() or not (cache / "analysis_numbers.json").exists():
        print("no soundings / analysis_numbers.json under --cache (make them with tools/channel_smartocean.py sound, analyse)", file=sys.stderr)
        return 2
    res = run(cache)
    text = markdown(res)
    if args.figure:
        figure(res, Path(args.figure))
        text += f"\nFigure: `{Path(args.figure).name}`.\n"
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
