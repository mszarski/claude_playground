#!/usr/bin/env python3
"""Our time spreading against measured channels: the suite's first comparison with real data.

Measured: the "Purple" channels of the Underwater Acoustic Channel Repository (SPACE'08, south
of Martha's Vineyard, October 2008; 12.5 kHz, 9.2-15.8 kHz band; one moored source, moored
arrays at 60 m, 200 m and 1 km; 15 m of water; 53 s per file). Z. Li et al., "Underwater
Acoustic Channel Library", arXiv:2609.03207; data doi:10.5281/zenodo.21287414 (CC BY 4.0).
Get them with tools/fetch_channels.py; nothing of the data is in this repository.

Model: straight-ray images in the stated geometry (`ImageSourceProvider`) wrapped by
sim/timespread.py, laid out as h(t, tau) on the measurement's grid, passed through a one-pole
model of the measurement's channel estimator, with white noise at the measurement's floor
(`channel_replay.model_channel`, `estimator_filter`). The SAME estimators
(`channel_replay.arrival_statistics`) then run on both.

What the dataset states, and what it does not (everything below marked ASSUMED is ours):

  stated    water depth 15 m; source 4 m above the seabed; top array element "approximately
            3.3 m" above it; ranges 60 m, 200 m, 1 km; element spacings; centre frequency,
            band and symbol rate; the estimator (RLS, forgetting factor, one update per delay
            sample) and its phase loop
  ASSUMED   sound speed 1500 m/s, no gradient; flat seabed; its reflection (a sand-like fluid
            half-space; only levels depend on it); omnidirectional source and hydrophones;
            that files k, k+5, k+10 of one "recording" (purple_1..5, 6..10, 11..15) were
            taken at the same time; a fully developed wind sea (Pierson-Moskowitz height and
            vertical speed, Cox-Munk slope) described by one number, the wind speed
  NOT GIVEN wind speed, wave height or sea state at the time; sound-speed profile; seabed type;
            tide; time of day

One free parameter is adjusted, per recording: the wind speed, fitted to ONE measured number,
the rms Doppler width of the first surface-bounce arrival at 60 m (closed form of
sim/timespread.py: width = 2 f v sin(grazing) / c for a fully rough bounce, v the rms vertical
speed of the surface). Everything else in the table is then a prediction: the Doppler widths
of the other arrivals and ranges, the shape of the Doppler spectrum, every delay spread,
every coherent share and scintillation index, the levels. Nothing else is tuned. The options
of the model are not tuned either: both values of each are run and reported.

    tools/channel_validation.py [--data DIR | --synthetic] [--out table.md] [--json numbers.json]
                                [--figures docs/figures] [--seeds 3] [--receivers 8]

--data defaults to $GPUSONAR_UACR_DIR. --synthetic runs the whole procedure on a "measurement"
made by the model itself with a known wind (5 m/s, Gaussian Doppler, facet decay): a check of
the procedure, and what the table looks like when the model is exactly right.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gpusonar import acoustics  # noqa: E402
from gpusonar.sim import ImageSourceProvider  # noqa: E402
from gpusonar.sim import channel_replay as cr  # noqa: E402
from gpusonar.sim.timespread import wrap  # noqa: E402

# -- the experiment ---------------------------------------------------------------------------------

STATED = {"water_depth": 15.0, "source_height": 4.0, "top_element_height": 3.3, "fc": 12500.0, "symbol_rate": 1e7 / 1536.0}
# system: (range m, vertical elements used, spacing m). Systems 1-2 are cross arrays whose first 16
# elements are the vertical leg ("numbered top to bottom, then left to right").
SYSTEMS = {1: (60.0, 16, 0.0375), 3: (200.0, 24, 0.05), 5: (1000.0, 12, 0.12)}
ASSUMED = {"sound_speed": 1500.0, "bottom": {"speed": 1700.0, "density": 1.9, "attenuation": 0.5}}
OPTIONS = {
    "A": {"doppler_spectrum": "gaussian", "decay_time": "facet", "multi_bounce": "convolve"},
    "B": {"doppler_spectrum": "lorentzian", "decay_time": "rule_of_thumb", "multi_bounce": "sum"},
    "C": {"doppler_spectrum": "gaussian", "decay_time": "rule_of_thumb", "multi_bounce": "convolve"},   # the suite's defaults
}
METRICS = ["delay_ms", "level_db", "coherent", "coherent_arrival", "scintillation", "doppler_rms", "doppler_w50", "doppler_w90",
           "doppler_shape", "coherence_time_ms", "decay_ms", "mean_excess_ms"]


def geometry(system: int) -> dict:
    rng, n, spacing = SYSTEMS[system]
    depth = STATED["water_depth"]
    return {"range": rng, "source_depth": depth - STATED["source_height"], "depth": depth, "n_elements": n, "spacing": spacing,
            "top_depth": depth - STATED["top_element_height"]}


def receiver_positions(geo: dict, indices) -> np.ndarray:
    return np.array([[geo["range"], 0.0, geo["top_depth"] + geo["spacing"] * int(i)] for i in indices])


def flat_provider(geo: dict, max_order: int = 3):
    return ImageSourceProvider(ASSUMED["sound_speed"], surface_z=0.0, bottom_z=geo["depth"], max_order=max_order, bottom=ASSUMED["bottom"])


def ray_name(n_surface: int, n_bottom: int, up_first: bool) -> str:
    n = int(n_surface + n_bottom)
    if n == 0:
        return "D"
    seq = ("SB" if (up_first or n_bottom == 0) and n_surface > 0 else "BS") * n
    return seq[:n]


def arrival_groups(geo: dict, rx: np.ndarray, bandwidth: float, merge: float = 0.3e-3, window: float = 0.02) -> list:
    """The flat-boundary eigenrays of the middle receiver, merged into groups the measurement
    cannot separate (closer than `merge` seconds): [{name, delay s after the direct path,
    grazing rad of its first member's surface bounce (or None), n_surface}]."""
    rays = flat_provider(geo).eigenrays([0.0, 0.0, geo["source_depth"]], rx[len(rx) // 2][None, :])
    ok = np.flatnonzero(rays.valid[0])
    d0 = rays.delay[0, ok].min()
    rows = sorted(({"name": ray_name(rays.n_surface[0, k], rays.n_bottom[0, k], rays.launch[0, k, 2] < 0.0), "delay": float(rays.delay[0, k] - d0),
                    "grazing": float(np.arcsin(abs(rays.launch[0, k, 2]))), "n_surface": int(rays.n_surface[0, k]),
                    "travel": float(rays.delay[0, k])} for k in ok), key=lambda r: r["delay"])
    groups: list = []
    for r in rows:
        if r["delay"] > window:
            break
        if groups and r["delay"] - groups[-1]["members"][-1]["delay"] < merge:
            groups[-1]["members"].append(r)
        else:
            groups.append({"members": [r]})
    for g in groups:
        g["name"] = "+".join(m["name"] for m in g["members"])
        g["delay"] = float(np.mean([m["delay"] for m in g["members"]]))
        g["n_surface"] = max(m["n_surface"] for m in g["members"])
    return groups


# -- one channel through the estimators ---------------------------------------------------------------

def analyse(ch: cr.TimeVaryingChannel, groups: list, *, search: int = 5, max_tail: int = 14, quarters: bool = True) -> dict:
    """Per arrival group: every metric as (value, standard error), the value the mean over
    receivers of the whole-record estimate, the standard error from the four quarters of the
    record (std of the quarter means / 2)."""
    def one(c: cr.TimeVaryingChannel) -> dict:
        ref = cr.reference_phase(c)
        prof = cr.delay_profile(c, ref)
        out = {g["name"]: {m: [] for m in METRICS} for g in groups}
        out["_bins"], out["_floor"], out["_direct_level"] = [], [], []
        for m in range(c.n_receivers):
            p = prof["total"][m]
            floor = float(prof["floor"][m])
            first = int(np.flatnonzero(p > 0.25 * p.max())[0])
            d = first + int(np.argmax(p[first:first + 4]))                       # the first strong peak: the direct group
            bins = []
            want = [d + int(round(g["delay"] * c.fs_delay)) for g in groups]
            for i, b in enumerate(want):
                # Look for the peak near the prediction, but not past half way to the neighbouring arrivals.
                gap = min([abs(b - x) for j, x in enumerate(want) if j != i] or [2 * search])
                reach = int(np.clip(gap // 2, 1, search))
                lo, hi = max(b - reach, 1), min(b + reach + 1, c.n_delay - 1)
                bins.append(d if i == 0 else (lo + int(np.argmax(p[lo:hi])) if lo < hi else -1))
            # Levels are relative to the whole channel's power above the floor: the direct arrival is no
            # reference where it overlaps the bottom bounce (their sum depends on a fraction of a wavelength).
            level0 = float((p - floor).sum())
            for i, g in enumerate(groups):
                b = bins[i]
                rec = out[g["name"]]
                nxt = min([x for x in bins[i + 1:] if x > b] or [c.n_delay])
                tail = int(np.clip(nxt - b - 3, 4, max_tail))
                try:
                    if b < 0 or b + tail >= c.n_delay:
                        raise ValueError
                    st = cr.arrival_statistics(c, m, b, reference=ref, floor=floor, tail_bins=tail)
                except ValueError:
                    for k in METRICS:
                        rec[k].append(np.nan)
                    continue
                w = st["doppler"]
                vals = {"delay_ms": (b - d) / c.fs_delay * 1e3, "level_db": 10.0 * np.log10(st["level"] / level0), "coherent": st["coherent_fraction"],
                        "coherent_arrival": st["coherent_fraction_arrival"], "scintillation": st["scintillation_index"], "doppler_rms": w["rms"],
                        "doppler_w50": w["w50"], "doppler_w90": w["w90"], "doppler_shape": w["shape"], "coherence_time_ms": st["coherence_time_s"] * 1e3,
                        "decay_ms": st["decay_time_s"] * 1e3, "mean_excess_ms": st["mean_excess_delay_s"] * 1e3}
                for k in METRICS:
                    rec[k].append(vals[k] if np.isfinite(vals[k]) else np.nan)
            out["_bins"].append(bins)
            out["_floor"].append(floor)
            out["_direct_level"].append(level0)
        return out

    whole = one(ch)
    result = {"_bins": whole["_bins"], "_floor": float(np.mean(whole["_floor"])), "_direct_level": float(np.mean(whole["_direct_level"]))}
    parts = []
    if quarters:
        n = ch.n_time // 4
        for q in range(4):
            sub = cr.TimeVaryingChannel(h=ch.h[q * n:(q + 1) * n], fs_time=ch.fs_time, fs_delay=ch.fs_delay, fc=ch.fc)
            parts.append(one(sub))
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        for g in groups:
            result[g["name"]] = {}
            for k in METRICS:
                value = float(np.nanmedian(whole[g["name"]][k])) if np.isfinite(whole[g["name"]][k]).any() else float("nan")
                se = float("nan")
                if parts:
                    qm = [np.nanmedian(p[g["name"]][k]) if np.isfinite(p[g["name"]][k]).any() else np.nan for p in parts]
                    se = float(np.nanstd(qm) / 2.0) if np.isfinite(qm).sum() >= 3 else float("nan")
                result[g["name"]][k] = (value, se)
    return result


# -- the model in the measurement's form ---------------------------------------------------------------

def simulate(geo: dict, rx: np.ndarray, *, wind: float, options: dict, seed: int, like: cr.TimeVaryingChannel, floor: float,
             direct_level: float, bubbles: bool = False, estimator: float | None = None, lead: float = 3e-3) -> cr.TimeVaryingChannel:
    import warnings
    surface = {"wind_speed": float(wind)}
    if bubbles:
        surface["bubble_loss_db"] = {"model": "hall_novarini"}
        warnings.filterwarnings("ignore", message="bubble_loss_db")    # the law's own caveat above 10 kHz: see the report
    # Many taps and many sinusoids per tap: a record of the model is then as ergodic as the sea is
    # (with the defaults, 6 taps of 8 lines, one seed's Doppler spectrum is a handful of lines).
    prov = wrap(flat_provider(geo), dict({"surface": surface, "taps": 24, "doppler_lines": 48, "seed": int(seed)}, **options), frequency=like.fc)
    ch = cr.model_channel(prov, [0.0, 0.0, geo["source_depth"]], rx, n_time=like.n_time, fs_time=like.fs_time, fs_delay=like.fs_delay,
                          n_delay=like.n_delay, fc=like.fc, bandwidth=STATED["symbol_rate"], lead=lead)
    if estimator:
        ch = cr.estimator_filter(ch, estimator)
    # Scale to the measurement's total power above its floor, then add the floor.
    p = (np.abs(ch.h) ** 2).mean(axis=0)
    scale = np.sqrt(direct_level / max(float(p.sum(axis=1).mean()), 1e-300))
    rng = np.random.default_rng(1000 + seed)
    ch.h = ch.h * scale + np.sqrt(floor / 2.0) * (rng.standard_normal(ch.h.shape) + 1j * rng.standard_normal(ch.h.shape))
    return ch


def closed_forms(group: dict, wind: float, fc: float, c: float) -> dict:
    """sim/timespread.py's closed forms for a group's first member with one surface bounce."""
    m = next((x for x in group["members"] if x["n_surface"] == 1), None)
    if m is None:
        return {}
    g = m["grazing"]
    h, mss, v = acoustics.rms_wave_height(wind), acoustics.mean_square_slope(wind), acoustics.rms_surface_vertical_speed(wind)
    gamma2 = (4.0 * np.pi * fc / c * h * np.sin(g)) ** 2
    return {"grazing_deg": float(np.degrees(g)), "rayleigh_gamma2": float(gamma2), "coherent": float(np.exp(-gamma2)),
            "doppler_gaussian_rms": float(2.0 * fc * v * np.sin(g) / c / np.sqrt(max(1.0 - np.exp(-gamma2), 1e-12))),
            "doppler_lorentzian_half_width": float(2.0 * fc * v * np.sin(g) / c)}


def fit_wind(doppler_rms: float, grazing: float, fc: float, c: float) -> float:
    """The wind speed whose Pierson-Moskowitz rms vertical speed gives a fully rough surface
    bounce the measured rms Doppler width: width = 2 f v sin(g) / c, v proportional to U."""
    return float(doppler_rms * c / (2.0 * fc * np.sin(grazing) * acoustics.rms_surface_vertical_speed(1.0)))


def average(results: list) -> dict:
    """Mean over seeds of analysed model channels, with the standard deviation over seeds: the
    spread of one record of this length about the model's mean, which is what one measured
    record is to be compared with."""
    out = {}
    for name in (k for k in results[0] if not k.startswith("_")):
        out[name] = {}
        for k in METRICS:
            v = np.array([r[name][k][0] for r in results], dtype=float)
            ok = np.isfinite(v)
            out[name][k] = (float(v[ok].mean()) if ok.any() else float("nan"),
                            float(v[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"))
    return out


# -- measurements ----------------------------------------------------------------------------------------

def load_measured(data: Path, index: int, n_receivers: int):
    system = (index - 1) % 5 + 1
    geo = geometry(system)
    idx = np.unique(np.round(np.linspace(0, geo["n_elements"] - 1, min(n_receivers, geo["n_elements"]))).astype(int))
    ch = cr.load_uacr(str(data / f"purple_{index}.mat"), receivers=[int(i) for i in idx])
    return ch, geo, receiver_positions(geo, idx)


def synthetic_measured(system: int, n_receivers: int, wind: float = 5.0, n_time: int = 3476, seed: int = 99):
    geo = geometry(system)
    idx = np.unique(np.round(np.linspace(0, geo["n_elements"] - 1, n_receivers)).astype(int))
    rx = receiver_positions(geo, idx)
    like = cr.TimeVaryingChannel(h=np.zeros((n_time, len(rx), 300), dtype=complex), fs_time=1e7 / 1536.0 / 100.0, fs_delay=1e7 / 768.0, fc=STATED["fc"],
                                 meta={"lambda": 0.995, "synthetic": True}, name=f"synthetic system {system}, wind {wind} m/s")
    ch = simulate(geo, rx, wind=wind, options=OPTIONS["A"], seed=seed, like=like, floor=2e-4, direct_level=0.44,
                  estimator=like.estimator_bandwidth())
    ch.meta, ch.name = like.meta, like.name
    return ch, geo, rx


def run(sites: list, seeds: int = 3, log=print) -> dict:
    """sites: [(label, recording, channel, geo, rx)]. The first site of each recording must be the
    60 m one: its first surface arrival fixes the wind."""
    c = ASSUMED["sound_speed"]
    out = {"stated": STATED, "assumed": ASSUMED, "free_parameters": 1, "sites": []}   # one per recording
    wind = {}
    for label, recording, ch, geo, rx in sites:
        groups = arrival_groups(geo, rx, STATED["symbol_rate"], window=(ch.n_delay - 60) / ch.fs_delay)
        meas = analyse(ch, groups)
        site = {"label": label, "recording": recording, "range": geo["range"], "groups": [], "estimator_bandwidth_hz": ch.estimator_bandwidth(),
                "floor_db_re_direct": 10.0 * np.log10(meas["_floor"] / meas["_direct_level"]), "duration_s": ch.duration, "receivers": int(ch.n_receivers)}
        if recording not in wind:
            s = next(g for g in groups if g["name"] == "S")
            wind[recording] = fit_wind(meas["S"]["doppler_rms"][0], s["members"][0]["grazing"], ch.fc, c)
            lo = fit_wind(meas["S"]["doppler_rms"][0] - 2 * meas["S"]["doppler_rms"][1], s["members"][0]["grazing"], ch.fc, c)
            site["wind_fit"] = {"wind_speed": wind[recording], "from": "rms Doppler width of S at this range", "two_sigma": wind[recording] - lo}
            log(f"{label}: wind fitted to the Doppler width of S: {wind[recording]:.2f} +- {wind[recording] - lo:.2f} m/s (2 sigma)")
        u = wind[recording]
        site["wind_speed"] = u
        site["surface"] = {"rms_height": acoustics.rms_wave_height(u), "mean_square_slope": acoustics.mean_square_slope(u),
                           "rms_vertical_speed": acoustics.rms_surface_vertical_speed(u)}
        sims = {}
        for key, opts, bub in (("A", OPTIONS["A"], False), ("C", OPTIONS["C"], False), ("B", OPTIONS["B"], False), ("A+bubbles", OPTIONS["A"], True)):
            runs = []
            for seed in range(1 if bub else seeds):                     # bubbles change levels only: one seed
                mc = simulate(geo, rx, wind=u, options=opts, seed=seed, like=ch, floor=meas["_floor"], direct_level=meas["_direct_level"],
                              bubbles=bub, estimator=ch.estimator_bandwidth())
                runs.append(analyse(mc, groups, quarters=False))
                if seed == 0 and key in ("A", "B"):
                    site.setdefault("_model", {})[key] = mc
            sims[key] = average(runs)
            log(f"{label}: model {key} done")
        site["_measured"], site["_bins"] = ch, meas["_bins"]
        for g in groups:
            row = {"name": g["name"], "predicted_delay_ms": g["delay"] * 1e3, "closed_form": closed_forms(g, u, ch.fc, c), "n_surface": g["n_surface"],
                   "measured": {k: meas[g["name"]][k] for k in METRICS}, "model": {key: {k: sims[key][g["name"]][k] for k in METRICS} for key in sims}}
            site["groups"].append(row)
        out["sites"].append(site)
    return out


# -- report ------------------------------------------------------------------------------------------------

# What the measurement cannot resolve, whatever the statistics: one delay bin is 0.077 ms, one
# Doppler bin 0.13 Hz. Added to the statistical tolerance; not adjusted to the outcome.
RESOLUTION = {"delay_ms": 0.08, "level_db": 0.5, "coherent": 0.02, "coherent_arrival": 0.02, "scintillation": 0.05, "doppler_rms": 0.07,
              "doppler_w50": 0.07, "doppler_w90": 0.13, "doppler_shape": 0.2, "coherence_time_ms": 8.0, "decay_ms": 0.04, "mean_excess_ms": 0.04}


def verdict(meas, model, metric: str = "") -> str:
    """`in` when the measured value is within two standard deviations of the model's: the
    larger of the measurement's standard error (quarters of the record) and the model's spread
    from record to record (seeds), plus the resolution."""
    (m, sm), (s, ss) = meas, model
    if not (np.isfinite(m) and np.isfinite(s)):
        return "-"
    tol = 2.0 * max(np.nan_to_num(sm), np.nan_to_num(ss)) + RESOLUTION.get(metric, 0.0)
    return "in" if abs(m - s) <= tol else "OUT"


def fmt(pair, digits=2) -> str:
    v, s = pair
    if not np.isfinite(v):
        return "-"
    return f"{v:.{digits}f}" + (f" +-{s:.{digits}f}" if np.isfinite(s) else "")


ROWS = [  # metric, heading, which model runs to show, digits
    ("delay_ms", "delay after direct (ms)", ("A",), 3),
    ("level_db", "level re whole channel (dB)", ("A",), 1),
    ("level_db", "the same with bubble loss", ("A+bubbles",), 1),
    ("coherent_arrival", "coherent share, arrival and tail", ("A", "C"), 3),
    ("coherent", "coherent share, at the peak", ("A", "C"), 3),
    ("scintillation", "scintillation index", ("A", "C"), 2),
    ("doppler_rms", "Doppler rms width (Hz)", ("A", "B"), 2),
    ("doppler_w50", "Doppler 50 % half width (Hz)", ("A", "B"), 2),
    ("doppler_shape", "Doppler shape w90/w50", ("A", "B"), 2),
    ("coherence_time_ms", "coherence time (ms)", ("A", "B"), 0),
    ("decay_ms", "tail decay time (ms)", ("A", "C", "B"), 3),
    ("mean_excess_ms", "tail mean excess delay (ms)", ("A", "C", "B"), 3),
]


def markdown(result: dict) -> str:
    lines = ["# Time spreading against measured channels (UACR Purple, SPACE'08)", "",
             "A: gaussian Doppler spectrum, facet decay time, bounces convolved. C: gaussian, rule_of_thumb, convolve (the suite's defaults).",
             "B: lorentzian, rule_of_thumb, sum (SST-style). One free parameter per recording (wind speed), fitted to the row marked FIT;",
             "every other row is a prediction. +- on the measured column: standard error from the four quarters of the record; on the model",
             "columns: spread of one record from seed to seed. `in`/`OUT`: within twice the larger of the two, plus the resolution.", ""]
    for site in result["sites"]:
        s = site["surface"]
        lines += [f"## {site['label']}: {site['range']:.0f} m, wind {site['wind_speed']:.2f} m/s"
                  + (" (fitted here)" if "wind_fit" in site else " (from the 60 m file of the same recording)"),
                  f"rms height {s['rms_height']:.3f} m, mean-square slope {s['mean_square_slope']:.4f}, rms vertical speed {s['rms_vertical_speed']:.3f} m/s; "
                  f"estimator bandwidth {site['estimator_bandwidth_hz']:.1f} Hz; floor {site['floor_db_re_direct']:.1f} dB per delay bin re the whole channel; "
                  f"{site['receivers']} receivers, {site['duration_s']:.0f} s", "",
                  "| arrival | quantity | measured | A | | C | | B | |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g in site["groups"]:
            if not np.isfinite(g["measured"]["level_db"][0]) or g["measured"]["level_db"][0] < site["floor_db_re_direct"] + 8.0:
                continue
            for metric, heading, runs, digits in ROWS:
                if g["n_surface"] == 0 and metric not in ("delay_ms", "level_db", "coherent", "scintillation"):
                    continue
                meas = g["measured"][metric]
                tag = " FIT" if ("wind_fit" in site and g["name"] == "S" and metric == "doppler_rms") else ""
                cells = []
                for slot in ("A", "C", "B"):
                    key = "A+bubbles" if (slot == "A" and runs == ("A+bubbles",)) else slot
                    if key in runs and key in g["model"]:
                        cells += [fmt(g["model"][key][metric], digits), "fit" if (tag and slot == "A") else verdict(meas, g["model"][key][metric], metric)]
                    else:
                        cells += ["", ""]
                lines.append(f"| {g['name']} | {heading}{tag} | {fmt(meas, digits)} | " + " | ".join(cells) + " |")
            cf = g["closed_form"]
            if cf:
                lines.append(f"| {g['name']} | closed forms | | grazing {cf['grazing_deg']:.1f} deg, Gamma^2 {cf['rayleigh_gamma2']:.2f}, coherent {cf['coherent']:.3f} | | "
                             f"Doppler rms {cf['doppler_gaussian_rms']:.2f} Hz | | Lorentzian half width {cf['doppler_lorentzian_half_width']:.2f} Hz | |")
        lines.append("")
    return "\n".join(lines)


def summary(result: dict) -> dict:
    """Counts of in/OUT per quantity over all surface arrivals well above the floor, for A and B."""
    out = {}
    for metric, heading, runs, _ in ROWS:
        for key in runs:
            n_in = n_out = 0
            ratios = []
            for site in result["sites"]:
                for g in site["groups"]:
                    if g["n_surface"] == 0 or not np.isfinite(g["measured"]["level_db"][0]) or g["measured"]["level_db"][0] < site["floor_db_re_direct"] + 8.0:
                        continue
                    if ("wind_fit" in site and g["name"] == "S" and metric == "doppler_rms") or key not in g["model"]:
                        continue
                    v = verdict(g["measured"][metric], g["model"][key][metric], metric)
                    n_in += v == "in"
                    n_out += v == "OUT"
                    if v != "-" and g["measured"][metric][0] != 0:
                        ratios.append(g["model"][key][metric][0] / g["measured"][metric][0])
            out[f"{metric}:{key}:{heading}" if key == "A+bubbles" else f"{metric}:{key}"] = {"in": n_in, "out": n_out, "median_model_over_measured": float(np.median(ratios)) if ratios else float("nan")}
    return out


def strip(result: dict) -> dict:
    return json.loads(json.dumps({k: v for k, v in result.items()}, default=lambda o: None if isinstance(o, cr.TimeVaryingChannel) else float(o)
                                 if isinstance(o, (np.floating, np.integer)) else (o.tolist() if isinstance(o, np.ndarray) else str(o))))


# -- figures (the style of tools/sst_validation.py) -------------------------------------------------------

def figures(result: dict, directory: Path, prefix: str = "channel_validation") -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    db = lambda x: 10.0 * np.log10(np.maximum(x, 1e-300))  # noqa: E731
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    sites = result["sites"]
    synthetic = any("synthetic" in s["label"] for s in sites)
    what = "synthetic 'measurement'" if synthetic else "measured (UACR Purple, SPACE'08, CC BY 4.0)"

    # 1. Delay profiles.
    fig, axes = plt.subplots(len(sites), 1, figsize=(9, 2.9 * len(sites)), constrained_layout=True, squeeze=False)
    for ax, site in zip(axes[:, 0], sites):
        ch = site["_measured"]
        d = site["_bins"][0][0]
        ms = (np.arange(ch.n_delay) - d) / ch.fs_delay * 1e3
        pm = (np.abs(ch.h[:, 0]) ** 2).mean(axis=0)
        ax.plot(ms, db(pm / pm.sum()), color="k", lw=1.1, label=what)
        for key, color in (("A", "C0"), ("B", "C1")):
            mc = site["_model"][key]
            p = (np.abs(mc.h[:, 0]) ** 2).mean(axis=0)
            dm = int(round(mc.meta["lead"] * mc.fs_delay))
            ax.plot((np.arange(mc.n_delay) - dm) / mc.fs_delay * 1e3, db(p / p.sum()), color=color, lw=0.9,
                    label=f"model {key} ({OPTIONS[key]['decay_time']}, {OPTIONS[key]['multi_bounce']})")
        for g in site["groups"]:
            if g["predicted_delay_ms"] < ms[-1] - 1:
                ax.annotate(g["name"], (g["predicted_delay_ms"], -4.0), fontsize=7, rotation=45, ha="left")
        ax.set_xlim(-2.0, min(ms[-1], 18.0))
        ax.set_ylim(-42.0, 4.0)
        ax.set_ylabel("Power per delay bin\n(dB re whole channel)")
        ax.set_title(f"{site['label']}: {site['range']:.0f} m, wind {site['wind_speed']:.1f} m/s"
                     + (" (fitted to the Doppler width of S here)" if "wind_fit" in site else " (not fitted here)") + "; top element", fontsize=9)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    axes[-1, 0].set_xlabel("Delay after the direct arrival (ms); labels: straight-ray images in the stated geometry (sound speed assumed)")
    paths.append(directory / f"{prefix}_profiles.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    # 2. Doppler spectra of the surface arrivals.
    panels = [(site, gi) for site in sites for gi, g in enumerate(site["groups"])
              if g["n_surface"] >= 1 and np.isfinite(g["measured"]["level_db"][0]) and g["measured"]["level_db"][0] > site["floor_db_re_direct"] + 10.0][:6]
    if panels:
        cols = min(3, len(panels))
        rows = int(np.ceil(len(panels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows), constrained_layout=True, squeeze=False)
        for ax, (site, gi) in zip(axes.ravel(), panels):
            g = site["groups"][gi]
            for chn, color, label, lw in ((site["_measured"], "k", what.split(" (")[0], 1.2), (site["_model"]["A"], "C0", "model A (gaussian)", 0.9),
                                          (site["_model"]["B"], "C1", "model B (lorentzian)", 0.9)):
                if chn is site["_measured"]:
                    b = site["_bins"][0][gi]
                else:
                    b = int(round(chn.meta["lead"] * chn.fs_delay)) + (site["_bins"][0][gi] - site["_bins"][0][0])
                ref = cr.reference_phase(chn)
                nu, s = cr.doppler_spectrum(chn.h[:, :, b - 1:b + 2] * np.conj(ref)[:, :, None], chn.fs_time, 512)
                ax.plot(nu, db(s / s[np.abs(nu) > 0.3].max()), color=color, lw=lw, label=label)
            ax.set_xlim(-15.0, 15.0)
            ax.set_ylim(-50.0, 5.0)
            ax.set_title(f"{site['label']}, {site['range']:.0f} m: {g['name']}", fontsize=9)
            ax.set_xlabel("Doppler (Hz)")
        for ax in axes.ravel()[len(panels):]:
            ax.set_visible(False)
        axes[0, 0].set_ylabel("Spectrum (dB re pedestal maximum)")
        axes[0, 0].legend(loc="upper right", fontsize=7)
        paths.append(directory / f"{prefix}_doppler.png")
        fig.savefig(paths[-1], dpi=110)
        plt.close(fig)

    # 3. Scattering function at the first site.
    site = sites[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, constrained_layout=True)
    for ax, chn, title in ((axes[0], site["_measured"], what), (axes[1], site["_model"]["A"], "model A"), (axes[2], site["_model"]["B"], "model B")):
        nu, tau, s = cr.scattering_function(chn, 0, 256, cr.reference_phase(chn))
        d = site["_bins"][0][0] if chn is site["_measured"] else int(round(chn.meta["lead"] * chn.fs_delay))
        img = db(s / s.max()).T
        m = ax.imshow(img, extent=[nu[0], nu[-1], (len(tau) - d) / chn.fs_delay * 1e3, -d / chn.fs_delay * 1e3], cmap="gray", vmin=-70.0, vmax=0.0, aspect="auto")
        ax.set_xlim(-12.0, 12.0)
        ax.set_ylim(12.0, -1.0)
        ax.set_xlabel("Doppler (Hz)")
        ax.set_title(f"{site['label']}, {site['range']:.0f} m: {title}", fontsize=9)
    axes[0].set_ylabel("Delay after the direct arrival (ms)")
    fig.colorbar(m, ax=axes, label="Scattering function (dB re maximum)")
    paths.append(directory / f"{prefix}_scattering.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("GPUSONAR_UACR_DIR", ""))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--prefix", default="channel_validation")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--receivers", type=int, default=8)
    ap.add_argument("--snapshots", type=int, default=3476, help="--synthetic only: length of the synthetic record")
    ap.add_argument("--sites", default="purple", choices=["purple", "all"],
                    help="all: after the Purple comparison, every other site and the cross-arrival tests (tools/channel_sites.py)")
    ap.add_argument("--recordings", default="0,1,2", help="which of the three Purple recordings (files 1-5, 6-10, 11-15) to use")
    args = ap.parse_args(argv)
    sites = []
    if args.synthetic:
        for system in (1, 3):
            ch, geo, rx = synthetic_measured(system, min(args.receivers, 4), n_time=args.snapshots)
            sites.append((f"synthetic_{system}", 0, ch, geo, rx))
    else:
        data = Path(args.data) if args.data else None
        if data is None or not (data / "purple_1.mat").exists():
            print("No measured channels: run tools/fetch_channels.py <dir> and pass --data <dir> (or set GPUSONAR_UACR_DIR), or use --synthetic.")
            return 2
        for recording in (int(r) for r in args.recordings.split(",")):
            if not (data / f"purple_{5 * recording + 1}.mat").exists():
                continue
            for system in (1, 3, 5):
                index = 5 * recording + system
                if (data / f"purple_{index}.mat").exists():
                    ch, geo, rx = load_measured(data, index, args.receivers)
                    sites.append((f"purple_{index}", recording, ch, geo, rx))
    result = run(sites, seeds=args.seeds)
    result["summary"] = summary(result)
    text = markdown(result)
    print(text)
    print("Summary over surface arrivals (rows other than the fitted one): quantity:model -> in / out, median model/measured")
    for k, v in result["summary"].items():
        print(f"  {k:28s} {v['in']:2d} in {v['out']:2d} out   {v['median_model_over_measured']:.2f}")
    if args.figures:
        for p in figures(result, Path(args.figures), args.prefix):
            print(f"wrote {p}")
    for site in result["sites"]:
        for k in [k for k in site if k.startswith("_")]:
            del site[k]
    if args.out:
        Path(args.out).write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(strip(result), indent=1))
    if args.sites == "all" and not args.synthetic:
        import tempfile
        import channel_sites
        with tempfile.TemporaryDirectory() as tmp:
            numbers = args.json or str(Path(tmp) / "purple.json")
            if not args.json:
                Path(numbers).write_text(json.dumps(strip(result), indent=1))
            extra = ["--data", str(args.data), "--purple-json", numbers]
            if args.figures:
                extra += ["--figures", args.figures, "--out", str(Path(args.figures) / "channel_validation_sites.md")]
            return channel_sites.main(extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
