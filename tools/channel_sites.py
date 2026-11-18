#!/usr/bin/env python3
"""Every site of the Underwater Acoustic Channel Repository against our time spreading: what
each site allows, what it does not, and the model's dependencies tested across arrivals.

    tools/channel_sites.py --data DIR [--purple-json numbers.json] [--figures docs/figures] [--out docs/figures/channel_validation_sites.md]
    (tools/channel_validation.py --sites all  runs the Purple comparison first and then this)

Three kinds of result, kept apart:

1. The full comparison (tools/channel_validation.py: geometry from the dataset, one fitted
   number per recording, predictions through the same estimators) needs, of a site: the
   geometry of each file (range, depths), straight-ray arrivals that can be told apart, and a
   channel estimator fast enough to follow the surface's fluctuations. `SITES` says for each of
   the eight which of these it has. Only Purple (SPACE'08) has all three.
2. Dependencies inside Purple, across 3 recordings x 3 ranges and all arrivals: Doppler width
   against the closed form (frequency x sin(grazing) x sqrt(bounces)), coherent share against
   exp(-Gamma^2), decay time against the facet rule and the rule of thumb, scintillation
   against the number of bounces; and the scaling with frequency INSIDE the band
   (`channel_replay.subband_statistics`: upper against lower half of 9.2-15.8 kHz).
3. Model-free numbers for the sites whose geometry is not given per file (Blue, Red, Yellow) or
   whose estimator is slow (Green) or whose window ends before the surface arrival (Black):
   the shape of the Doppler spectrum and the scintillation index of every arrival found. These
   do not test a prediction; they say whether what Purple shows is general.

No site states wind, wave height, sea state, a sound-speed profile or a seabed type (paper
arXiv:2609.03207 Sec. II and the site pages were searched for each). So no site allows a
comparison with nothing fitted, none allows a Bellhop run with a stated profile, and none
allows a test of a named sediment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import channel_validation as cv  # noqa: E402
from gpusonar import acoustics  # noqa: E402
from gpusonar.sim import channel_replay as cr  # noqa: E402

# What each site states (site pages and arXiv:2609.03207 Sec. II; estimator from the files' `meta`).
# tracker: RLS -> (1 - lambda) fs_delay / (2 pi), as documented; LMS -> mu fs_delay / (4 pi), which
# ASSUMES an input of mean power 1/2 (symbols alternating with zeros); the files do not give it.
SITES = {
    "purple": dict(place="Martha's Vineyard (SPACE'08), Oct 2008", fc=12.5e3, band="9.2-15.8 kHz", signal="BPSK PN 4095, 6.51 ksym/s",
                   geometry="moored; source 4 m above seabed, arrays' top element 3.3 m above it, water 15 m; 60 m, 200 m, 1 km",
                   files=[f"purple_{i}" for i in (1, 3, 5, 6, 8, 10, 11, 13)], protocol="full",
                   why="geometry stated per file; arrivals resolved at 60 and 200 m; RLS tracker 10.4 Hz"),
    "green": dict(place="Norwegian fjord (Norway'24), Nov 2024", fc=6e3, band="4.5 kHz about 6 kHz", signal="BPSK PN, 4.5 ksym/s, 60 s every 10 min (64 slots)",
                  geometry="moored; source 20 m, hydrophone 43 m, water 60 m, 270 m", files=["green"], protocol="none",
                  why="geometry stated, but the estimator (LMS, 3600 taps, step 1/14400) follows only about 0.05 Hz and snapshots are 4.5 Hz apart: "
                      "every arrival comes out steady. Nothing about spreading can be read from it"),
    "black": dict(place="Mariana Trench, Oct 2024", fc=18e3, band="12.5 kHz about 18 kHz", signal="QPSK PN 8192, 12.5 ksym/s",
                  geometry="vertical link: source 8718 m deep, 8-element array 6 m under a ship", files=["black"], protocol="none",
                  why="the delay window is 7 ms; the surface reflection above the array would come 8 ms after the direct path (2 x 6 m / c). "
                      "Only the direct path is in the file"),
    "red": dict(place="Singapore, Nov 2024", fc=25e3, band="20-30 kHz", signal="BPSK m-sequence 8191, 9.6 ksym/s",
                geometry="drifting boat; source about 6 m, top hydrophone 4.6 m, water 8-20 m and not uniform, range 100-400 m (not per file)",
                files=["red_1", "red_2"], protocol="model-free",
                why="range and water depth are not given per file: arrivals cannot be assigned without fitting the geometry"),
    "yellow": dict(place="Kauai (KAM'11), Jul 2011", fc=13e3, band="6.25 kHz about 13 kHz", signal="BPSK PN 4095, 6.25 ksym/s",
                   geometry="moored; source and array at mid depth in 100 m; 3 km and 7 km", files=["yellow_1", "yellow_4"], protocol="model-free",
                   why="geometry stated, but the arrivals are not where straight rays put them (no surface arrival 1.1 ms after the first at 3 km): "
                       "the water is not isovelocity and no profile is given"),
    "blue": dict(place="North Atlantic (MACE'10), Jun 2010", fc=13e3, band="10.5-15.5 kHz", signal="BPSK PN 2047, 4.9 ksym/s",
                 geometry="source towed at up to 1.5 m/s, 30-60 m deep, 3-7 km from a moored array at 50 m in 100 m (none of it per file)",
                 files=["blue_1"], protocol="model-free", why="range, source depth and speed are not given per file"),
    "pink": dict(place="off Kochi, Japan, Jul 2022", fc=6e3, band="4 kHz about 6 kHz", signal="QPSK PN 8192, 4 ksym/s",
                 geometry="source 176 m, array 146-172 m, about 200 m of water over flat and sloping seabed, 14 km", files=[], protocol="not run",
                 why="14 km with no sound-speed profile and no bathymetry; 1.5 GB per file. Not downloaded"),
    "brown": dict(place="North Pacific (ATOC), Nov 1994", fc=75.0, band="37.5 Hz about 75 Hz", signal="BPSK PN 1023, 37.5 sym/s",
                  geometry="source 652 m, array 900-1600 m, 3250 km", files=[], protocol="not applicable",
                  why="deep refracted paths that never touch the surface in the way our boundary model describes; nothing to compare"),
}


def tracker_bandwidth(ch: cr.TimeVaryingChannel) -> tuple[float, str]:
    b = ch.estimator_bandwidth()
    if b is not None:
        return b, "RLS, stated"
    mu = ch.meta.get("mu")
    if mu:
        return float(mu) * ch.fs_delay / (4.0 * np.pi), "LMS, input power assumed"
    return float("nan"), "unknown"


def survey(ch: cr.TimeVaryingChannel, max_receivers: int = 4, threshold_db: float = -18.0) -> list:
    """Model-free: every arrival found in the delay profile of up to `max_receivers` receivers."""
    ref = cr.reference_phase(ch)
    prof = cr.delay_profile(ch, ref)
    rows = []
    for m in range(min(ch.n_receivers, max_receivers)):
        p, floor = prof["total"][m], float(prof["floor"][m])
        pk = int(np.argmax(p))
        for b in sorted(cr.find_arrivals(p, n=8, min_separation=4, threshold_db=threshold_db)):
            try:
                st = cr.arrival_statistics(ch, m, b, reference=ref, floor=floor, tail_bins=8)
            except ValueError:
                continue
            rows.append({"receiver": m, "delay_ms": (b - pk) / ch.fs_delay * 1e3, "level_db": 10 * np.log10(p[b] / p[pk]), "coherent": st["coherent_fraction"],
                         "scintillation": st["scintillation_index"], "doppler_rms": st["doppler"]["rms"], "doppler_shape": st["doppler"]["shape"],
                         "is_reference": b == pk})
    return rows


def frequency_scaling(data: Path, files: list, n_receivers: int = 4) -> list:
    """Purple: Doppler width and coherent share of the surface arrivals in the lower and upper
    halves of the band (`subband_statistics`)."""
    rows = []
    for name in files:
        index = int(name.split("_")[1])
        if not (data / f"{name}.mat").exists() or (index - 1) % 5 + 1 == 5:
            continue
        ch, geo, rx = cv.load_measured(data, index, n_receivers)
        groups = cv.arrival_groups(geo, rx, cv.STATED["symbol_rate"], window=(ch.n_delay - 60) / ch.fs_delay)
        bins = cv.analyse(ch, groups, quarters=False)["_bins"]
        ref = cr.reference_phase(ch)
        for gi, g in enumerate(groups):
            if g["n_surface"] < 1 or gi + 1 >= len(groups):
                continue
            out = []
            for m in range(ch.n_receivers):
                b = bins[m][gi]
                nxt = min([x for x in bins[m][gi + 1:] if x > b] or [b + 12])
                if b < 0 or nxt - b < 5:
                    continue
                out.append(cr.subband_statistics(ch, m, b, after=int(min(nxt - b - 3, 8)), reference=ref))
            if out:
                rows.append({"file": name, "range": geo["range"], "arrival": g["name"], "frequency_ratio": float(np.mean([o["frequency_ratio"] for o in out])),
                             "doppler_ratio": float(np.mean([o["doppler_ratio"] for o in out])), "doppler_ratio_sd": float(np.std([o["doppler_ratio"] for o in out])),
                             "coherent_lower": float(np.mean([o["lower"]["coherent_fraction"] for o in out])),
                             "coherent_upper": float(np.mean([o["upper"]["coherent_fraction"] for o in out]))})
    return rows


def dependencies(purple: dict) -> list:
    """One row per Purple surface arrival well above the floor: measured values and the closed
    forms of sim/timespread.py for the recording's fitted wind (grazing angle, bounces)."""
    c = cv.ASSUMED["sound_speed"]
    rows = []
    for site in purple["sites"]:
        index = int(site["label"].split("_")[1])
        system = (index - 1) % 5 + 1
        geo = cv.geometry(system)
        rx = cv.receiver_positions(geo, [geo["n_elements"] // 2])
        groups = {g["name"]: g for g in cv.arrival_groups(geo, rx, cv.STATED["symbol_rate"])}
        u = site["wind_speed"]
        h, v = acoustics.rms_wave_height(u), acoustics.rms_surface_vertical_speed(u)
        for g in site["groups"]:
            meas = g["measured"]
            if g["n_surface"] < 1 or g["name"] not in groups or meas["level_db"][0] is None or meas["level_db"][0] < site["floor_db_re_direct"] + 8.0:
                continue
            mem = max(groups[g["name"]]["members"], key=lambda m: m["n_surface"])
            ns, graz = mem["n_surface"], mem["grazing"]
            gamma2 = ns * (4.0 * np.pi * cv.STATED["fc"] / c * h * np.sin(graz)) ** 2
            width = 2.0 * cv.STATED["fc"] * v * np.sin(graz) * np.sqrt(ns) / c / np.sqrt(max(1.0 - np.exp(-gamma2), 1e-12))
            val = lambda k, src=meas: (src[k][0] if src[k][0] is not None else float("nan"))  # noqa: E731
            rows.append({"file": site["label"], "range": site["range"], "arrival": g["name"], "n_surface": ns, "n_bounces": len(mem["name"]),
                         "grazing_deg": float(np.degrees(graz)), "gamma2": float(gamma2), "fitted_here": "wind_fit" in site and g["name"] == "S",
                         "doppler_measured": val("doppler_rms"), "doppler_closed_form": float(width),
                         "coherent_measured": val("coherent_arrival"), "coherent_closed_form": float(np.exp(-gamma2)),
                         "decay_measured": val("decay_ms"), "decay_facet": val("decay_ms", g["model"]["A"]),
                         "decay_rule": val("decay_ms", g["model"]["C"]) if "C" in g["model"] else float("nan"),
                         "scintillation": val("scintillation"), "scintillation_model": val("scintillation", g["model"]["A"]),
                         "shape": val("doppler_shape")})
    return rows


def _stats(ratio) -> str:
    r = np.asarray([x for x in ratio if np.isfinite(x) and x > 0], dtype=float)
    if not len(r):
        return "-"
    lg = np.log(r)
    return f"median {np.exp(np.median(lg)):.2f}, scatter x{np.exp(lg.std()):.2f} (n={len(r)})"


def report(dep: list, freq: list, surveys: dict, wind: dict) -> str:
    lines = ["# All UACR sites against our time spreading", "",
             "Data: Underwater Acoustic Channel Repository v1.0, doi:10.5281/zenodo.21287414 (CC BY 4.0); Li et al., arXiv:2609.03207.",
             "No site states wind, sea state, a sound-speed profile or a seabed type.", "",
             "## 1. What each site allows", "", "| Site | Where, when | Carrier, band | Geometry as stated | Estimator | Protocol | Why |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in SITES.items():
        est = surveys.get(name, {}).get("tracker", "")
        lines.append(f"| {name} | {s['place']} | {s['fc'] / 1e3:g} kHz, {s['band']} | {s['geometry']} | {est} | {s['protocol']} | {s['why']} |")
    lines += ["", "Fitted per Purple recording (one number each, the wind speed, from the Doppler width of the first surface arrival at 60 m): "
              + ", ".join(f"recording {k}: {v:.2f} m/s" for k, v in wind.items()) + ".", ""]
    if dep:
        free = [r for r in dep if not r["fitted_here"]] or dep
        lines += ["## 2. Dependencies across the Purple arrivals", "",
                  f"{len(free)} surface arrivals that were not used for a fit (grazing {min(r['grazing_deg'] for r in free):.1f} to "
                  f"{max(r['grazing_deg'] for r in free):.1f} degrees, one to two surface bounces).", "",
                  "| Ingredient | Measured / predicted | |", "| --- | --- | --- |",
                  f"| Doppler rms width: 2 f v sin(g) sqrt(bounces) / c / sqrt(1 - exp(-Gamma^2)) | {_stats([r['doppler_measured'] / r['doppler_closed_form'] for r in free])} | closed form, fitted wind |"]
        for lo, hi, tag in ((0.0, 12.0, "grazing under 12 deg"), (12.0, 90.0, "grazing over 12 deg")):
            sel = [r for r in free if lo <= r["grazing_deg"] < hi]
            lines.append(f"| ... {tag} | {_stats([r['doppler_measured'] / r['doppler_closed_form'] for r in sel])} | |")
        for ns in (1, 2):
            sel = [r for r in free if r["n_surface"] == ns]
            lines.append(f"| ... {ns} surface bounce(s) | {_stats([r['doppler_measured'] / r['doppler_closed_form'] for r in sel])} | |")
        part = [r for r in dep if 0.02 < r["coherent_measured"] < 0.95 and r["gamma2"] < 6.0]
        hs = [np.sqrt(np.log(1.0 / r["coherent_measured"]) / r["gamma2"]) for r in part]
        lines.append(f"| coherent share exp(-Gamma^2): rms height that the measured share implies / Pierson-Moskowitz height | {_stats(hs)} | arrivals with a measurable coherent part |")
        for rng in sorted({r["range"] for r in part}):
            sel = [np.sqrt(np.log(1.0 / r["coherent_measured"]) / r["gamma2"]) for r in part if r["range"] == rng]
            lines.append(f"| ... at {rng:.0f} m | {_stats(sel)} | |")
        lines.append(f"| decay time, facet rule (model A through the estimators) | {_stats([r['decay_measured'] / r['decay_facet'] for r in dep])} | |")
        lines.append(f"| decay time, rule of thumb (model C) | {_stats([r['decay_measured'] / r['decay_rule'] for r in dep])} | |")
        for lo, hi, tag in ((0.0, 12.0, "grazing under 12 deg (Gamma^2 of order 1)"), (12.0, 90.0, "grazing over 12 deg (Gamma^2 over 7)")):
            sel = [r for r in dep if lo <= r["grazing_deg"] < hi]
            lines.append(f"| ... {tag}: facet; rule of thumb | {_stats([r['decay_measured'] / r['decay_facet'] for r in sel])} | {_stats([r['decay_measured'] / r['decay_rule'] for r in sel])} |")
        for nb in sorted({r["n_bounces"] for r in dep}):
            sel = [r for r in dep if r["n_bounces"] == nb and r["gamma2"] > 6.0 and np.isfinite(r["scintillation"])]
            if sel:
                si = np.array([r["scintillation"] for r in sel])
                lines.append(f"| scintillation index, fully rough arrivals with {nb} bounce(s) in all | measured median {np.median(si):.2f}, range {si.min():.2f} to {si.max():.2f} (n={len(si)}) | "
                             f"model {np.median([r['scintillation_model'] for r in sel]):.2f}; K shape for the median: "
                             f"{('%.1f' % (2.0 / (np.median(si) - 1.0))) if np.median(si) > 1.05 else 'none needed'} |")
        sh = np.array([r["shape"] for r in dep if np.isfinite(r["shape"])])
        lines.append(f"| Doppler shape w90/w50 (Gaussian 2.44; Lorentzian about 4 on this axis) | median {np.median(sh):.2f}, range {sh.min():.2f} to {sh.max():.2f} (n={len(sh)}) | |")
        lines.append("")
    if freq:
        lines += ["## 3. Scaling with frequency inside the band (Purple)", "",
                  "Upper against lower half of the band; a width proportional to frequency gives the frequency ratio.", "",
                  "| File | Range (m) | Arrival | frequency ratio | Doppler width ratio | coherent share, lower | upper |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in freq:
            lines.append(f"| {r['file']} | {r['range']:.0f} | {r['arrival']} | {r['frequency_ratio']:.3f} | {r['doppler_ratio']:.3f} +-{r['doppler_ratio_sd']:.3f} | "
                         f"{r['coherent_lower']:.3f} | {r['coherent_upper']:.3f} |")
        dr = np.array([r["doppler_ratio"] for r in freq])
        lines += ["", f"Doppler width ratio over {len(dr)} arrivals: mean {dr.mean():.3f}, standard deviation {dr.std():.3f}, standard error {dr.std() / np.sqrt(len(dr)):.3f}; "
                  f"frequency ratio {np.mean([r['frequency_ratio'] for r in freq]):.3f}; no dependence on frequency would give 1.", ""]
    lines += ["## 4. Model-free numbers at the other sites", "",
              "Every arrival within 18 dB of the strongest, up to four receivers; the strongest arrival is the phase reference and is left out.",
              "These do not test a prediction (the geometry is not known); the tracker bandwidth limits the widths that can be seen.", "",
              "| Site | File | Carrier | Tracker (Hz) | Arrivals | Doppler rms (Hz), median and range | Shape w90/w50, median and range | Scintillation index, median and range | Coherent share, median |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, sv in surveys.items():
        for fname, rows in sv.get("files", {}).items():
            rows = [r for r in rows if not r["is_reference"]]
            if not rows:
                lines.append(f"| {name} | {fname} | {SITES[name]['fc'] / 1e3:g} kHz | {sv['tracker']} | none besides the reference | | | | |")
                continue
            q = lambda k: np.array([r[k] for r in rows])  # noqa: E731
            lines.append(f"| {name} | {fname} | {SITES[name]['fc'] / 1e3:g} kHz | {sv['tracker']} | {len(rows)} | {np.median(q('doppler_rms')):.2f} ({q('doppler_rms').min():.2f}-{q('doppler_rms').max():.2f}) | "
                         f"{np.median(q('doppler_shape')):.2f} ({q('doppler_shape').min():.2f}-{q('doppler_shape').max():.2f}) | "
                         f"{np.median(q('scintillation')):.2f} ({q('scintillation').min():.2f}-{q('scintillation').max():.2f}) | {np.median(q('coherent')):.3f} |")
    return "\n".join(lines) + "\n"


def figure(dep: list, freq: list, surveys: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    col = {60.0: "C0", 200.0: "C1", 1000.0: "C2"}
    ax = axes[0, 0]
    for r in dep:
        ax.plot(r["doppler_closed_form"], r["doppler_measured"], marker="o" if r["n_surface"] == 1 else "s", color=col[r["range"]],
                mfc="none" if r["fitted_here"] else col[r["range"]], ls="none", ms=6)
    lim = [0.0, 1.1 * max(max(r["doppler_closed_form"], r["doppler_measured"]) for r in dep)]
    ax.plot(lim, lim, color="k", lw=0.8)
    ax.set_xlabel("Closed form, fitted wind (Hz)")
    ax.set_ylabel("Measured rms Doppler width (Hz)")
    ax.set_title("Doppler width; open: the arrival the wind was fitted to;\nsquares: two surface bounces", fontsize=9)
    for rng, c in col.items():
        ax.plot([], [], marker="o", color=c, ls="none", label=f"{rng:.0f} m")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[0, 1]
    g2 = np.logspace(-1.5, 1.4, 100)
    ax.plot(g2, np.exp(-g2), color="k", lw=0.9, label="exp(-Gamma^2)")
    ax.plot(g2, np.exp(-1.9 * g2), color="k", lw=0.9, ls="--", label="rms height x 1.38")
    for r in dep:
        ax.plot(r["gamma2"], max(r["coherent_measured"], 1e-3), marker="o" if r["n_surface"] == 1 else "s", color=col[r["range"]], ls="none", ms=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 1.3)
    ax.set_xlabel("Gamma^2 for the fitted wind (Pierson-Moskowitz height)")
    ax.set_ylabel("Measured coherent share (arrival and tail)")
    ax.set_title("Coherent share", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[0, 2]
    for key, marker, label in (("decay_facet", "o", "facet"), ("decay_rule", "x", "rule of thumb")):
        ax.plot([r[key] for r in dep], [r["decay_measured"] for r in dep], marker=marker, ls="none", color="C0" if key == "decay_facet" else "C3", ms=6, label=label)
    ax.plot([0.05, 10], [0.05, 10], color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.1, 10)
    ax.set_xlabel("Model through the same estimators (ms)")
    ax.set_ylabel("Measured tail decay time (ms)")
    ax.set_title("Decay time", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for r in dep:
        ax.plot(r["grazing_deg"], r["scintillation"], marker="o" if r["n_bounces"] == 1 else ("s" if r["n_bounces"] == 2 else "^"), color=col[r["range"]], ls="none", ms=6)
        ax.plot(r["grazing_deg"], r["scintillation_model"], marker=".", color="0.6", ls="none")
    ax.axhline(1.0, color="k", lw=0.8)
    for marker, label in (("o", "1 bounce"), ("s", "2 bounces"), ("^", "3 bounces"), (".", "model")):
        ax.plot([], [], marker=marker, color="0.4", ls="none", label=label)
    ax.legend(fontsize=8)
    ax.set_xlabel("Grazing angle at the surface (degrees)")
    ax.set_ylabel("Scintillation index of the arrival's energy")
    ax.set_title("Level fluctuation; line: Rayleigh (complex Gaussian) = 1", fontsize=9)
    ax = axes[1, 1]
    if freq:
        x = np.arange(len(freq))
        ax.errorbar(x, [r["doppler_ratio"] for r in freq], yerr=[r["doppler_ratio_sd"] for r in freq], marker="o", ls="none", color="C0")
        ax.axhline(np.mean([r["frequency_ratio"] for r in freq]), color="k", lw=0.9, label="proportional to frequency")
        ax.axhline(1.0, color="k", lw=0.9, ls="--", label="independent of frequency")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['file'].split('_')[1]}:{r['arrival']}" for r in freq], rotation=90, fontsize=6)
        ax.legend(fontsize=8)
    ax.set_ylabel("Doppler width, upper / lower half of the band")
    ax.set_title("Scaling with frequency inside 9.2-15.8 kHz (file:arrival)", fontsize=9)
    ax = axes[1, 2]
    names = []
    for i, (name, sv) in enumerate([("purple", None)] + [(n, s) for n, s in surveys.items() if any(s.get("files", {}).values())]):
        if name == "purple":
            sh, si = [r["shape"] for r in dep], [r["scintillation"] for r in dep]
        else:
            rows = [r for rows in sv["files"].values() for r in rows if not r["is_reference"]]
            if not rows:
                continue
            sh, si = [r["doppler_shape"] for r in rows], [r["scintillation"] for r in rows]
        ax.plot(np.full(len(sh), len(names)) - 0.15, sh, marker="o", ls="none", color="C0", ms=4)
        ax.plot(np.full(len(si), len(names)) + 0.15, si, marker="s", ls="none", color="C3", ms=4)
        names.append(f"{name}\n{SITES[name]['fc'] / 1e3:g} kHz")
    ax.axhline(2.44, color="C0", lw=0.8)
    ax.axhline(1.0, color="C3", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0.0, 6.0)
    ax.plot([], [], marker="o", color="C0", ls="none", label="Doppler shape w90/w50 (line: Gaussian)")
    ax.plot([], [], marker="s", color="C3", ls="none", label="scintillation index (line: Rayleigh)")
    ax.legend(fontsize=7)
    ax.set_title("All sites, every arrival found (no geometry used outside Purple)", fontsize=9)
    fig.suptitle("Measured: UACR (CC BY 4.0). Model: sim/timespread.py, one fitted number (wind) per Purple recording", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("GPUSONAR_UACR_DIR", ""))
    ap.add_argument("--purple-json", default=None, help="numbers written by tools/channel_validation.py --json (all Purple recordings)")
    ap.add_argument("--figures", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    data = Path(args.data) if args.data else None
    if data is None or not data.is_dir():
        print("No measured channels: run tools/fetch_channels.py <dir> --set all-sites and pass --data <dir> (or set GPUSONAR_UACR_DIR).")
        return 2
    purple = json.loads(Path(args.purple_json).read_text()) if args.purple_json else {"sites": []}
    dep = dependencies(purple)
    wind = {}
    for s in purple["sites"]:
        wind.setdefault(s["recording"] + 1, s["wind_speed"])
    freq = frequency_scaling(data, SITES["purple"]["files"])
    surveys = {}
    for name, s in SITES.items():
        entry = {"tracker": "", "files": {}}
        for fname in s["files"]:
            path = data / f"{fname}.mat"
            if not path.exists():
                continue
            if name == "purple":
                ch = cr.load_uacr(str(path), receivers=[0])
            else:
                import h5py
                with h5py.File(path, "r") as f:
                    n_rx = int(f["h_hat"].shape[1])
                ch = cr.load_uacr(str(path), receivers=list(range(min(4, n_rx))))
                entry["files"][fname] = survey(ch)
            bw, how = tracker_bandwidth(ch)
            entry["tracker"] = f"{bw:.2g} ({how})"
        surveys[name] = entry
    text = report(dep, freq, surveys, wind)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    if args.figures and dep:
        path = Path(args.figures) / "channel_validation_dependencies.png"
        figure(dep, freq, surveys, path)
        print(f"wrote {path}")
    if args.json:
        Path(args.json).write_text(json.dumps({"dependencies": dep, "frequency_scaling": freq, "surveys": surveys}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
