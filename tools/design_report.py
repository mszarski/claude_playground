#!/usr/bin/env python3
"""Design report: every number of the sonar designs derived from the config files, each
owner requirement checked against them, and the diagrams.

The YAML files under configs/ are the single source of truth. This tool reads them,
derives what follows from them (wavelength, grating-lobe limit, beamwidth, resolution,
ping rate, data rate, link budget), checks the requirements in REQUIREMENTS below, and
writes

    docs/DESIGN.md                 the report (generated: edit this tool or the configs)
    docs/design/derived.json       the derived numbers
    docs/design/*.png              block diagram, arrays and beams, link budgets, scenario geometry

Usage: tools/design_report.py [--check]     --check fails if a requirement is not met
CPU only; needs matplotlib.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Wedge  # noqa: E402

from gpusonar import acoustics  # noqa: E402
from gpusonar.config import list_configs, load_scenario_config, load_sonar_config  # noqa: E402
from gpusonar.sim.noise import wenz_spectrum_db  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402
from gpusonar.sonar_equation import SonarEquation  # noqa: E402

# Palette: the first slots of a colour-blind-checked categorical set, on a near-white surface.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})

DESIGN_SONARS = ["fls_120k", "fls_170k", "fls_330k", "sas_minsas120", "popoto_array", "popoto_long_range"]
SCENARIO_OF = {"fls_120k": "fls_120k_transit", "fls_170k": "fls_170k_transit", "fls_330k": "fls_330k_transit",
               "sas_minsas120": "sas_minsas_survey", "popoto_array": "popoto_far_mover",
               "popoto_long_range": "popoto_12km"}
# Assumptions of the link budgets (not requirements): stated in the report.
# SAS: one resolution cell of seabed, 2.1 cm x 1.9 cm, Lambert -27 dB at 15 degrees grazing:
# -27 + 10 log10(sin^2 15) + 10 log10(0.033 x 0.019) = -70.7 dB.
SAS_CELL_TS = -27.0 + 10 * np.log10(np.sin(np.radians(15.0)) ** 2) + 10 * np.log10(0.021 * 0.01875)
TARGET_STRENGTH_DB = {"fls_120k": 5.0, "fls_170k": 5.0, "fls_330k": 5.0, "sas_minsas120": float(SAS_CELL_TS),
                      "popoto_array": 25.0, "popoto_long_range": 25.0}
TARGET_NOTE = {"fls_120k": "small boat hull", "fls_170k": "small boat hull", "fls_330k": "small boat hull",
               "sas_minsas120": "one resolution cell of seabed (2.1 cm x 1.9 cm) at -27 dB Lambert and 15 degrees grazing; "
                                "array gain includes the pings of the synthetic aperture at full range",
               "popoto_array": "ship-sized target", "popoto_long_range": "ship-sized target"}
WIND = 5.0


def sonar_for(name: str) -> SonarModel:
    """The sonar as its scenario uses it (the Popoto array is scaled there)."""
    if name.startswith("popoto"):
        over = load_scenario_config(SCENARIO_OF[name])["sonar_overrides"]
        over = {k: v for k, v in over.items() if k in ("array", "transmit")}
        return SonarModel.load(name, over)
    return SonarModel.load(name)


def derive(name: str) -> dict:
    s = sonar_for(name)
    cfg = s.config
    c = s.nominal_sound_speed
    w = cfg["transmit"]["waveform"]
    f0, f1 = float(w["f_start"]), float(w["f_stop"])
    fc, band = 0.5 * (f0 + f1), abs(f1 - f0)
    lam, lam_min = c / fc, c / max(f0, f1)
    arr = cfg["array"]
    pitch = float(np.linalg.norm(s.positions[1] - s.positions[0]))
    length = s.n_elements * pitch
    rx = cfg["receive"]
    r_max = float(rx["window"]["range"][1])
    if name.startswith("popoto"):
        r_max = float(load_scenario_config(SCENARIO_OF[name])["sonar_overrides"]["receive"]["window"]["range"][1])
    fs, bits = float(rx["sample_rate"]), int(rx.get("adc_bits", 16))
    duration = float(w["duration"])
    noise = float(wenz_spectrum_db(np.array([fc]), WIND, 0.5)[0])
    scenario = load_scenario_config(SCENARIO_OF[name])
    env = scenario.get("environment", {})
    gain = None
    if s.mode == "sas":
        # Synthetic aperture: every ping whose beam covers the cell adds coherently.
        tx_w = float(cfg["transmit"]["aperture_width"])
        pings_in_aperture = max(r_max * 0.886 * lam / tx_w / (length / 2), 1.0)
        gain = 10 * np.log10(s.n_elements * pings_in_aperture)
    eq = SonarEquation.from_sonar(s, noise, target_strength_db=TARGET_STRENGTH_DB[name],
                                  temperature=float(env.get("temperature", 10.0)),
                                  salinity=float(env.get("salinity", 35.0)), array_gain_db=gain)
    grating = lam_min / pitch - 1.0
    d = {
        "sonar": name, "mode": s.mode, "scenario": SCENARIO_OF[name],
        "centre_frequency_khz": fc / 1e3, "bandwidth_khz": band / 1e3, "wavelength_mm": lam * 1e3,
        "pulse_ms": duration * 1e3, "range_resolution_cm": 100 * c / (2 * band),
        "elements": s.n_elements, "pitch_mm": pitch * 1e3, "pitch_in_wavelengths_at_top_of_band": pitch / lam_min,
        "grating_lobe_free_steering_deg": float(np.degrees(np.arcsin(min(grating, 1.0)))) if grating > 0 else 0.0,
        "array_length_m": length, "beamwidth_deg": float(np.degrees(0.886 * lam / length)),
        "element_fov_az_deg": float(np.degrees(s.beamwidth_az)) if np.isfinite(s.beamwidth_az) else None,
        "element_fov_el_deg": float(np.degrees(s.beamwidth_el)) if np.isfinite(s.beamwidth_el) else None,
        "element_width_mm": (1e3 * float(arr["element_width"]) if arr.get("element_width") is not None else
                             (1e3 * 0.886 * lam / s.beamwidth_az if np.isfinite(s.beamwidth_az) else None)),
        "element_height_mm": (1e3 * float(arr["element_height"]) if arr.get("element_height") is not None else
                              (1e3 * 0.886 * lam / s.beamwidth_el if np.isfinite(s.beamwidth_el) else None)),
        "tilt_up_deg": float(np.degrees(np.arcsin(-s.boresight[2]))) + 0.0,
        "max_range_m": r_max, "min_ping_interval_s": 2 * r_max / c + duration,
        "cross_range_at_max_range_m": float(r_max * 0.886 * lam / length),
        "adc_rate_khz": fs / 1e3, "adc_bits": bits,
        "samples_per_ping_per_channel": int(round((2 * r_max / c + duration) * fs)),
        "raw_data_rate_mbit_s": s.n_elements * fs * bits / 1e6,
        "source_level_db": float(w["source_level_db"]),
        "absorption_db_per_km": float(eq.absorption_db_per_m() * 1e3),
        "noise_spectrum_db": noise, "target_strength_db": TARGET_STRENGTH_DB[name],
        "array_gain_db": float(eq.array_gain()), "processing_gain_db": float(eq.processing_gain()),
        "snr_at_max_range_db": float(eq.snr(r_max)),
        "range_for_13_db_m": float(eq.detection_range(13.0)),
        "boards": len(s.boards),
    }
    if s.mode == "sas":
        tx_width = float(cfg["transmit"]["aperture_width"])
        d["along_track_resolution_cm"] = 100 * tx_width / 2
        d["max_advance_per_ping_m"] = length / 2
        d["max_speed_at_max_range_m_s"] = (length / 2) / (2 * r_max / c + duration)
        d["transmit_beamwidth_deg"] = float(np.degrees(0.886 * lam / tx_width))
        d["synthetic_aperture_at_50_m"] = float(50.0 * 0.886 * lam / tx_width)
    d["_eq"] = eq
    return d


# --- requirements: what the owner asked for, where it lives, and a check -----------------------

def requirements(D: dict) -> list:
    """(id, requirement, where it is set, value found, met?)"""
    out = []

    def add(rid, text, where, value, ok):
        out.append({"id": rid, "requirement": text, "set_in": where, "value": value, "met": bool(ok)})

    for name, f, rng, vfov, tilted in (("fls_120k", 120.0, 300.0, 20.0, False), ("fls_170k", 170.0, 450.0, 5.0, True),
                                       ("fls_330k", 330.0, 300.0, 5.0, True)):
        d = D[name]
        tag = name.split("_")[1].upper()
        add(f"FLS-{tag}-1", f"Centre frequency {f:.0f} kHz", f"configs/sonars/{name}.yaml: transmit.waveform.f_start, f_stop",
            f"{d['centre_frequency_khz']:.0f} kHz", abs(d["centre_frequency_khz"] - f) < 0.5)
        add(f"FLS-{tag}-2", f"Range {rng:.0f} m", f"configs/sonars/{name}.yaml: receive.window.range",
            f"{d['max_range_m']:.0f} m; SNR there {d['snr_at_max_range_db']:.0f} dB (link budget)",
            abs(d["max_range_m"] - rng) < 0.5 and d["snr_at_max_range_db"] > 13.0)
        add(f"FLS-{tag}-3", "120 degrees horizontal field of view", f"configs/sonars/{name}.yaml: fov.horizontal_deg",
            f"{d['element_fov_az_deg']:.0f} degrees; no grating lobe within +/-{d['grating_lobe_free_steering_deg']:.0f} degrees",
            abs(d["element_fov_az_deg"] - 120.0) < 0.5 and d["grating_lobe_free_steering_deg"] >= 60.0)
        add(f"FLS-{tag}-4", f"Vertical field of view {vfov:.0f} degrees, adjustable" + (", steered up" if tilted else ""),
            f"configs/sonars/{name}.yaml: fov.vertical_deg, fov.tilt_up_deg",
            f"{d['element_fov_el_deg']:.0f} degrees, tilted up {d['tilt_up_deg']:.1f} degrees",
            abs(d["element_fov_el_deg"] - vfov) < 0.5 and (d["tilt_up_deg"] > 0.5) == tilted)
        sc = load_scenario_config(SCENARIO_OF[name])
        labels = {i["label"]: i for i in sc["scene"]}
        moving = [k for k, i in labels.items() if np.any(np.asarray(i.get("velocity", [0, 0, 0]), dtype=float) != 0)]
        add(f"FLS-{tag}-5", "Ownship (an AUV at about 20 m) under way against static and moving targets, one a boat hull; "
                            "seabed and sea surface fixed in the world",
            f"configs/scenarios/{SCENARIO_OF[name]}.yaml: platform, scene, reverberation",
            f"depth {sc['platform']['position'][2]:.0f} m, {np.linalg.norm(sc['platform']['velocity']):.0f} m/s; "
            f"moving: {', '.join(moving)}; static: {', '.join(k for k in labels if k not in moving)}; "
            f"{len(sc['reverberation'])} boundaries with ownship Doppler",
            abs(sc["platform"]["position"][2] - 20.0) < 0.5 and np.linalg.norm(sc["platform"]["velocity"]) > 0
            and moving and len(labels) > len(moving) and any(i["type"] == "hull" for i in labels.values())
            and len(sc["reverberation"]) == 2)
    d = D["sas_minsas120"]
    sc = load_scenario_config("sas_minsas_survey")
    types = {i["label"]: i["type"] for i in sc["scene"]}
    add("SAS-1", "SAS like the Kraken MINSAS 120 at 330 kHz", "configs/sonars/sas_minsas120.yaml: transmit.waveform, array",
        f"{d['centre_frequency_khz']:.0f} kHz, {d['bandwidth_khz']:.0f} kHz, {d['array_length_m']:.2f} m array, "
        f"{d['along_track_resolution_cm']:.1f} x {d['range_resolution_cm']:.1f} cm (published: 40 kHz, 1.09 m, 3.3 x 3.0 cm, or 2.1 x 1.9 cm post-processed)",
        abs(d["centre_frequency_khz"] - 330) < 0.5 and abs(d["bandwidth_khz"] - 40) < 0.5
        and abs(d["array_length_m"] - 1.09) < 0.01 and abs(d["along_track_resolution_cm"] - 2.1) < 0.05)
    add("SAS-2", "Survey paths", "configs/scenarios/sas_minsas_survey.yaml: platform.legs",
        f"{len(sc['platform']['legs'])} legs, headings " + ", ".join(f"{leg['heading_deg']:.0f}" for leg in sc["platform"]["legs"])
        + f" degrees; {sc['platform']['speed'] * sc['pings']['interval']:.2f} m per ping (limit {d['max_advance_per_ping_m']:.3f} m)",
        len(sc["platform"]["legs"]) >= 2 and sc["platform"]["speed"] * sc["pings"]["interval"] <= d["max_advance_per_ping_m"] + 1e-6)
    add("SAS-3", "Seabed with a shipwreck, a cable, rocks and a cube for calibration",
        "configs/scenarios/sas_minsas_survey.yaml: scene", ", ".join(types),
        {"wreck_hull", "cable", "rocks", "calibration_cube"} <= set(types))
    d = D["popoto_array"]
    sc = load_scenario_config("popoto_far_mover")
    ship = next(i for i in sc["scene"] if i["label"] == "ship")
    rng = float(np.hypot(*np.mean(ship["positions"], axis=0)[:2]))
    add("POP-1", "Popoto array with ownship moving and a big moving target many km away",
        "configs/scenarios/popoto_far_mover.yaml: platform.velocity, scene",
        f"ownship {np.linalg.norm(sc['platform']['velocity']):.1f} m/s; target at {rng / 1e3:.1f} km moving "
        f"{np.linalg.norm(ship['velocity']):.0f} m/s, +{d['target_strength_db']:.0f} dB",
        np.linalg.norm(sc["platform"]["velocity"]) > 0 and rng > 3000 and np.linalg.norm(ship["velocity"]) > 0)
    counts = [SonarModel.load("popoto_array", {"array": {"count": n}}) for n in (8, 16, 32, 64)]
    add("POP-2", "The Popoto array scales to more elements", "configs/sonars/popoto_array.yaml: array.count, receive.board_size",
        "; ".join(f"{s.n_elements} hydrophones on {len(s.boards)} boards" for s in counts),
        [s.n_elements for s in counts] == [8, 16, 32, 64] and [len(s.boards) for s in counts] == [1, 2, 4, 8])
    d = D["popoto_long_range"]
    study = json.loads((ROOT / "docs" / "design" / "popoto_range_study.json").read_text())
    best = study["fixed_count"][str(d["elements"])]
    add("POP-3", "Frequency and source level that take the Popoto array to 12 km",
        "configs/sonars/popoto_long_range.yaml: transmit.waveform, array (sized by tools/popoto_range_study.py)",
        f"{d['centre_frequency_khz']:.0f} kHz, {d['source_level_db']:.0f} dB, {d['elements']} hydrophones, {d['pulse_ms'] / 1e3:.0f} s pulse: "
        f"{d['_eq'].snr(12000.0):.0f} dB at 12 km with spherical spreading (study: {best['source_level_db']:.0f} dB at "
        f"{best['best_frequency_hz'] / 1e3:.1f} kHz for 13 dB)",
        d["_eq"].snr(12000.0) >= 13.0 and d["max_range_m"] >= 12000.0)
    envs = sorted(n for n in list_configs("scenarios") if n.startswith("popoto_env_"))
    seen = set()
    for n in envs:
        e = load_scenario_config(n)["propagation"]["environment"]
        seen.add((e["depth"], e["bottom_speed"], tuple(e["ssp_speed"])))
    add("POP-4", "Popoto examples with different ocean characteristics, sound-speed profiles, bottom types and depths",
        "configs/scenarios/popoto_env_summer_mud_60m.yaml: propagation.environment (and the other popoto_env_* files)", f"{len(envs)} environments: " + ", ".join(envs),
        len(envs) >= 4 and len(seen) == len(envs))
    extra = {n: {i["type"] for i in load_scenario_config(n)["scene"]}
             for n in ("fls_120k_cluttered_seabed", "fls_120k_rough_surface", "fls_170k_traffic")}
    add("FLS-X", "More FLS examples: a realistic boat, clutter on the seabed, waves on the sea surface",
        "configs/scenarios/fls_120k_cluttered_seabed.yaml: scene (also fls_120k_rough_surface, fls_170k_traffic)",
        "; ".join(f"{n}: {', '.join(sorted(t))}" for n, t in extra.items()),
        all("boat" in t for t in extra.values()) and "debris" in extra["fls_120k_cluttered_seabed"]
        and "sea_surface" in extra["fls_120k_rough_surface"])
    rocks = {n: sum(int(i.get("count", 0)) for i in load_scenario_config(n)["scene"] if i["type"] == "rocks")
             for n in ("sas_minsas_boulder_field", "sas_minsas_rocky_ripples")}
    add("SAS-4", "More SAS examples with rocks", "configs/scenarios/sas_minsas_boulder_field.yaml: scene (also sas_minsas_rocky_ripples)",
        "; ".join(f"{n}: {c} rocks" for n, c in rocks.items()), all(c >= 20 for c in rocks.values()))
    for name in DESIGN_SONARS:
        cfg = load_sonar_config(name)
        add(f"CFG-{name}", f"`{name}`: elements and array configurable",
            f"configs/sonars/{name}.yaml: array.count, array.pitch" + (", fov" if "fov" in cfg else ""),
            "overrides checked: count 2x gives 2x aperture",
            abs(SonarModel.load(name, {"array": {"count": 2 * D[name]["elements"] if name != "popoto_array" else 16}}).n_elements
                - (2 * D[name]["elements"] if name != "popoto_array" else 16)) == 0)
    return out


# --- diagrams ------------------------------------------------------------------------------------

def box(ax, xy, w, h, text, color=BLUE, text_color="white", fs=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", color=text_color, fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=MUTED, shrinkA=2, shrinkB=2))


def draw_system(path):
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.grid(False)
    srcs = ["Simulator\n(scenarios)", "Container file\n(.h5 replay)", "WAV + sidecar\n(Popoto boards)",
            "Ultrasound rig\n(NVIDIA reference)", "Flank / Popoto\nboard stream (UDP)"]
    for i, t in enumerate(srcs):
        box(ax, (0.2, 5.0 - i * 1.15), 2.3, 0.85, t, color=AQUA)
        arrow(ax, (2.5, 5.42 - i * 1.15), (3.3, 3.1))
    box(ax, (3.3, 2.55), 1.7, 1.1, "Ping record\n(one format)", color=INK)
    stages = [("Condition\nbaseband, pulse\ncompression (cuFFT)", 5.5), ("Synchronise\ncalibrate\n(multi-board)", 7.4),
              ("Motion\nestimation\n(SAS, Ceres)", 9.3)]
    arrow(ax, (5.0, 3.1), (5.5, 3.1))
    for (t, x), nxt in zip(stages, [7.4, 9.3, None]):
        box(ax, (x, 2.45), 1.6, 1.3, t, color=BLUE, fs=8.5)
        if nxt:
            arrow(ax, (x + 1.6, 3.1), (nxt, 3.1))
    box(ax, (11.2, 4.3), 2.1, 1.2, "Backprojection kernel\n(host / CUDA)\nfan, stripmap, volume,\nrange x bearing", color=ORANGE, fs=8.5)
    box(ax, (11.2, 2.5), 2.1, 1.1, "Far-field beamformer\nMVDR (flank, Popoto)", color=ORANGE, fs=8.5)
    box(ax, (11.2, 0.6), 2.1, 1.2, "Detect (CFAR)\ncluster, track\ndisplay, GeoTIFF", color=YELLOW, text_color=INK, fs=8.5)
    arrow(ax, (10.9, 3.4), (11.2, 4.6))
    arrow(ax, (10.9, 3.0), (11.2, 3.0))
    ax.add_patch(FancyArrowPatch((13.3, 4.9), (13.3, 1.2), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=MUTED,
                                 connectionstyle="arc3,rad=-0.55"))
    arrow(ax, (12.25, 2.5), (12.25, 1.8))
    ax.text(0.2, 6.0, "Sources", color=MUTED, fontsize=10)
    ax.text(5.5, 4.0, "Stages (each a library function and a Holoscan operator)", color=MUTED, fontsize=10)
    ax.text(3.3, 1.9, "configs/sonars/*.yaml describe the array,\ntransmit and sampling of every record", color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def draw_arrays(D, path):
    fig, axes = plt.subplots(len(DESIGN_SONARS), 3, figsize=(14, 2.9 * len(DESIGN_SONARS)),
                             gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    for row, name in zip(axes, DESIGN_SONARS):
        d, s = D[name], sonar_for(name)
        ax = row[0]
        along = (s.positions - s.positions.mean(axis=0)) @ (s.positions[-1] - s.positions[0]) / np.linalg.norm(s.positions[-1] - s.positions[0])
        w = 0.8 * d["pitch_mm"] / 1e3
        for a, b in zip(along, s.channel_board):
            ax.add_patch(plt.Rectangle((a - w / 2, -0.5), w, 1.0, fc=[BLUE, ORANGE, AQUA, YELLOW][int(b) % 4], ec="none"))
        ax.set_xlim(along[0] - 2 * w - 0.02 * d["array_length_m"], along[-1] + 2 * w + 0.02 * d["array_length_m"])
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel("position along the array (m)")
        ax.set_title(f"{name}: {d['elements']} elements, pitch {d['pitch_mm']:.1f} mm "
                     f"({d['pitch_in_wavelengths_at_top_of_band']:.2f} wavelengths at the top of the band),\n"
                     f"{d['array_length_m']:.2f} m long" + (f", {d['boards']} boards of 8 (colours)" if d["boards"] > 1 else ""),
                     fontsize=9, loc="left")
        # Plan view: sector and one beam.
        ax = row[1]
        ax.grid(False)
        fov = d["element_fov_az_deg"] or 360.0
        fwd = 90.0
        ax.add_patch(Wedge((0, 0), 1.0, fwd - fov / 2, fwd + fov / 2, fc=BLUE, alpha=0.18, ec="none"))
        bw = max(d["beamwidth_deg"], 1.2) if s.mode != "sas" else max(d["transmit_beamwidth_deg"], 1.2)
        ax.add_patch(Wedge((0, 0), 1.0, fwd - bw / 2, fwd + bw / 2, fc=ORANGE, ec="none"))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.15 if fov < 200 else -1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        label = (f"plan: sector {fov:.0f} deg,\nbeam {d['beamwidth_deg']:.2f} deg" if s.mode != "sas"
                 else f"plan: element {fov:.0f} deg,\ntransmit {d['transmit_beamwidth_deg']:.1f} deg")
        if d["element_fov_az_deg"] is None:
            label = f"plan: omnidirectional hydrophones,\nbeam {d['beamwidth_deg']:.1f} deg at broadside"
        ax.set_title(label, fontsize=9, loc="left")
        # Side view: vertical field of view and tilt.
        ax = row[2]
        ax.grid(False)
        vf = d["element_fov_el_deg"]
        if vf is None:
            ax.add_patch(Wedge((0, 0), 1.0, -90, 90, fc=BLUE, alpha=0.18, ec="none"))
            ax.set_title("side: omnidirectional\n", fontsize=9, loc="left")
        else:
            up = d["tilt_up_deg"]
            ax.add_patch(Wedge((0, 0), 1.0, up - vf / 2, up + vf / 2, fc=BLUE, alpha=0.35, ec="none"))
            ax.plot([0, 1.05 * np.cos(np.radians(up))], [0, 1.05 * np.sin(np.radians(up))], color=ORANGE, lw=1.5)
            ax.plot([0, 1.05], [0, 0], color=MUTED, lw=0.8, ls="--")
            ax.set_title(f"side: {vf:.0f} deg vertical,\nboresight {up + 0.0:+.1f} deg from horizontal", fontsize=9, loc="left")
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.55, 0.55)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_link_budgets(D, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    groups = [("Forward-looking sonars", ["fls_120k", "fls_170k", "fls_330k"], np.geomspace(20, 1500, 200)),
              ("SAS (one seabed resolution cell)", ["sas_minsas120"], np.geomspace(10, 400, 200)),
              ("Popoto arrays (6-10 kHz by count; 12 km design)", ["popoto_array"], np.geomspace(500, 30000, 200))]
    for ax, (title, names, r) in zip(axes, groups):
        if names == ["popoto_array"]:
            eq = D["popoto_array"]["_eq"]
            for n, col in zip((8, 16, 32, 64), (BLUE, ORANGE, AQUA, YELLOW)):
                e = SonarEquation(eq.source_level_db, eq.center_frequency, eq.bandwidth, eq.pulse_duration, n,
                                  eq.noise_spectrum_db, target_strength_db=eq.target_strength_db,
                                  temperature=eq.temperature, salinity=eq.salinity)
                ax.plot(r / 1e3, e.snr(r), color=col, lw=2, label=f"{n} hydrophones")
                ax.annotate(f"{n}", (r[-1] / 1e3, e.snr(r[-1])), textcoords="offset points", xytext=(4, 0), fontsize=8, va="center")
            lr = D["popoto_long_range"]
            ax.plot(r / 1e3, lr["_eq"].snr(r), color=INK, lw=2, ls="--",
                    label=f"popoto_long_range: {lr['elements']} at {lr['centre_frequency_khz']:.0f} kHz, {lr['source_level_db']:.0f} dB")
            ax.plot(12.0, lr["_eq"].snr(12000.0), "o", color=INK, ms=8, mec=SURFACE, mew=2)
            ax.annotate(f"{lr['_eq'].snr(12000.0):.0f} dB at 12 km", (12.0, lr["_eq"].snr(12000.0)), textcoords="offset points",
                        xytext=(8, 14), fontsize=8)
            ax.set_xlabel("range (km)")
            ax.axvline(6.0, color=MUTED, lw=0.8, ls=":")
            ax.axvline(12.0, color=MUTED, lw=0.8, ls=":")
        else:
            for name, col in zip(names, (BLUE, ORANGE, AQUA)):
                d = D[name]
                ax.plot(r, d["_eq"].snr(r), color=col, lw=2, label=name)
                ax.plot(d["max_range_m"], d["snr_at_max_range_db"], "o", color=col, ms=8, mec=SURFACE, mew=2)
                ax.annotate(f"{name}\n{d['snr_at_max_range_db']:.0f} dB at {d['max_range_m']:.0f} m",
                            (d["max_range_m"], d["snr_at_max_range_db"]), textcoords="offset points", xytext=(8, 8), fontsize=8)
            ax.set_xlabel("range (m)")
        ax.set_xscale("log")
        ax.axhline(13.0, color=MUTED, lw=0.9, ls="--")
        ax.text(ax.get_xlim()[0] * 1.05, 14.0, "13 dB", color=MUTED, fontsize=8)
        ax.set_ylabel("output SNR (dB)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(-20, 90)
        if len(names) > 1 or names == ["popoto_array"]:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_fls_geometry(name, path):
    sc = load_scenario_config(SCENARIO_OF[name])
    s = SonarModel.load(name)
    d_el, tilt = np.degrees(s.beamwidth_el), np.degrees(np.arcsin(-s.boresight[2]))
    r_max = float(sc["sonar_overrides"]["receive"]["window"]["range"][1])
    depth, own = float(sc["environment"]["water_depth"]), np.asarray(sc["platform"]["position"], dtype=float)
    vel = np.asarray(sc["platform"]["velocity"], dtype=float)
    t_end = (sc["pings"]["count"] - 1) * sc["pings"]["interval"]
    fig, (top, side) = plt.subplots(2, 1, figsize=(11, 8.2), gridspec_kw={"height_ratios": [2.2, 1.0]}, constrained_layout=True)
    top.add_patch(Wedge((own[1], own[0]), r_max, 90 - 60, 90 + 60, fc=BLUE, alpha=0.12, ec="none"))
    top.plot(own[1], own[0], "s", color=INK, ms=7)
    top.annotate(f"ownship, {np.linalg.norm(vel):.0f} m/s", (own[1], own[0]), textcoords="offset points", xytext=(8, -4), fontsize=9)
    top.add_patch(FancyArrowPatch((own[1], own[0]), (own[1] + vel[1] * 12, own[0] + vel[0] * 12), arrowstyle="-|>",
                                  mutation_scale=12, color=INK, lw=1.5))
    side.axhline(0, color=BLUE, lw=1.5)
    side.axhline(depth, color=YELLOW, lw=2)
    side.text(r_max * 0.02, -1.5, "sea surface (fixed in the world)", ha="left", va="bottom", fontsize=8, color=MUTED)
    side.text(r_max * 0.02, depth - 1.5, "seabed (fixed in the world)", ha="left", va="bottom", fontsize=8, color=MUTED)
    for sign in (-1, 1):
        ang = np.radians(tilt + sign * d_el / 2)
        side.plot([0, r_max * np.cos(ang)], [own[2], own[2] - r_max * np.sin(ang)], color=ORANGE, lw=1.2)
    side.fill([0, r_max, r_max], [own[2], own[2] - r_max * np.tan(np.radians(tilt + d_el / 2)),
                                  own[2] - r_max * np.tan(np.radians(tilt - d_el / 2))], color=ORANGE, alpha=0.12, lw=0)
    side.plot(0, own[2], "s", color=INK, ms=7)
    for item, col in zip(sc["scene"], (ORANGE, AQUA, YELLOW, MUTED)):
        pos = np.asarray(item.get("origin", item.get("centre", [np.mean(item.get("x_range", [0])), np.mean(item.get("y_range", [0])), item.get("depth", 0)])), dtype=float)
        v = np.asarray(item.get("velocity", (0, 0, 0)), dtype=float)
        top.plot(pos[1], pos[0], "o", color=col, ms=9, mec=SURFACE, mew=2)
        if np.any(v != 0):
            top.add_patch(FancyArrowPatch((pos[1], pos[0]), (pos[1] + v[1] * t_end * 2.5, pos[0] + v[0] * t_end * 2.5),
                                          arrowstyle="-|>", mutation_scale=12, color=col, lw=1.8))
        speed = f", {np.linalg.norm(v):.0f} m/s" if np.any(v != 0) else ", static"
        top.annotate(item["label"] + speed, (pos[1], pos[0]), textcoords="offset points", xytext=(9, 6), fontsize=9)
        side.plot(np.hypot(pos[0] - own[0], pos[1] - own[1]), pos[2], "o", color=col, ms=9, mec=SURFACE, mew=2)
        side.annotate(item["label"], (np.hypot(pos[0], pos[1]), pos[2]), textcoords="offset points", xytext=(7, 7), fontsize=8)
    top.set_aspect("equal")
    top.set_xlim(-r_max * 0.9, r_max * 0.9)
    top.set_ylim(-0.05 * r_max, 1.05 * r_max)
    top.set_xlabel("y, to starboard (m)")
    top.set_ylabel("x, ahead (m)")
    top.set_title(f"{SCENARIO_OF[name]}: plan view, 120 degree sector to {r_max:.0f} m", loc="left", fontsize=10)
    side.set_xlim(0, r_max)
    side.set_ylim(depth + 6, -8)
    side.set_xlabel("range (m)")
    side.set_ylabel("depth (m)")
    side.set_title(f"side view: vertical field of view {d_el:.0f} degrees, boresight {tilt:+.1f} degrees "
                   "(vertical scale exaggerated)", loc="left", fontsize=10)
    fig.savefig(path, dpi=115)
    plt.close(fig)


def draw_sas_geometry(path):
    sc = load_scenario_config("sas_minsas_survey")
    win = sc["sonar_overrides"]["receive"]["window"]["range"]
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    alt = -sc["platform"]["legs"][0]["start"][2]
    for leg, col in zip(sc["platform"]["legs"], (BLUE, ORANGE)):
        h = np.radians(leg["heading_deg"])
        start = np.asarray(leg["start"][:2], dtype=float)
        end = start + np.array([np.cos(h), np.sin(h)]) * sc["platform"]["speed"] * sc["pings"]["interval"] * (leg["pings"] - 1)
        ax.add_patch(FancyArrowPatch((start[1], start[0]), (end[1], end[0]), arrowstyle="-|>", mutation_scale=14, color=col, lw=2.5))
        stb = np.array([-np.sin(h), np.cos(h)])              # starboard in (x, y)
        g0, g1 = (np.sqrt(max(win[0] ** 2 - alt ** 2, 0)), np.sqrt(win[1] ** 2 - alt ** 2))
        corners = [start + stb * g0, end + stb * g0, end + stb * g1, start + stb * g1]
        ax.add_patch(Polygon([(c[1], c[0]) for c in corners], fc=col, alpha=0.12, ec=col, lw=0.8))
        ax.annotate(f"leg, heading {leg['heading_deg']:.0f} deg\nswath to starboard", (start[1], start[0]),
                    textcoords="offset points", xytext=(8, -18 if leg["heading_deg"] < 90 else 8), fontsize=9, color=col)
    for item in sc["scene"]:
        t = item["type"]
        if t == "hull":
            hd, L, o = np.radians(item["heading_deg"]), item["length"], np.asarray(item["origin"][:2], dtype=float)
            a, b = o - L / 2 * np.array([np.cos(hd), np.sin(hd)]), o + L / 2 * np.array([np.cos(hd), np.sin(hd)])
            ax.plot([a[1], b[1]], [a[0], b[0]], color=INK, lw=7, solid_capstyle="round")
            ax.annotate("wreck (hull, deckhouse, mast)", (o[1], o[0]), textcoords="offset points", xytext=(10, 10), fontsize=9)
        elif t == "cable":
            w = np.asarray(item["waypoints"], dtype=float)
            ax.plot(w[:, 1], w[:, 0], color=AQUA, lw=2)
            ax.annotate("cable", (w[1, 1], w[1, 0]), textcoords="offset points", xytext=(-34, 4), fontsize=9)
        elif t == "rocks":
            ax.add_patch(plt.Rectangle((item["y_range"][0], item["x_range"][0]), np.ptp(item["y_range"]), np.ptp(item["x_range"]),
                                       fc=YELLOW, alpha=0.35, ec="none"))
            ax.annotate("rocks", (np.mean(item["y_range"]), item["x_range"][1]), textcoords="offset points", xytext=(0, 4), fontsize=9, ha="center")
        elif t == "box" and item["label"] == "calibration_cube":
            ax.plot(item["min"][1], item["min"][0], "s", color=ORANGE, ms=8, mec=INK)
            ax.annotate("calibration cube", (item["min"][1], item["min"][0]), textcoords="offset points", xytext=(8, -3), fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-8, 108)
    ax.set_ylim(-6, 46)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title(f"sas_minsas_survey: two legs at {alt:.0f} m altitude looking at one patch of seabed from both sides", loc="left", fontsize=10)
    fig.savefig(path, dpi=115)
    plt.close(fig)


def draw_popoto_geometry(path):
    sc = load_scenario_config("popoto_far_mover")
    ship = next(i for i in sc["scene"] if i["label"] == "ship")
    pos, v = np.mean(ship["positions"], axis=0), np.asarray(ship["velocity"], dtype=float)
    own_v = np.asarray(sc["platform"]["velocity"], dtype=float)
    t_end = (sc["pings"]["count"] - 1) * sc["pings"]["interval"]
    win = sc["sonar_overrides"]["receive"]["window"]["range"]
    fig, ax = plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)
    ax.add_patch(Wedge((0, 0), win[1], 90 - 40, 90 + 40, width=win[1] - win[0], fc=BLUE, alpha=0.14, ec="none"))
    ax.plot(0, 0, "s", color=INK, ms=7)
    ax.add_patch(FancyArrowPatch((0, 0), (own_v[0] * t_end * 8, 0), arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.8))
    ax.annotate(f"ownship and array axis, {np.linalg.norm(own_v):.1f} m/s", (0, 0), textcoords="offset points", xytext=(8, -14), fontsize=9)
    ax.plot(pos[0], pos[1], "o", color=ORANGE, ms=10, mec=SURFACE, mew=2)
    ax.add_patch(FancyArrowPatch((pos[0], pos[1]), (pos[0] + v[0] * t_end * 3, pos[1] + v[1] * t_end * 3), arrowstyle="-|>",
                                 mutation_scale=14, color=ORANGE, lw=2))
    ax.annotate(f"ship, {np.linalg.norm(v):.0f} m/s, {np.hypot(pos[0], pos[1]) / 1e3:.1f} km", (pos[0], pos[1]),
                textcoords="offset points", xytext=(10, 8), fontsize=9)
    ax.text(0, win[0] - 250, f"receive window {win[0] / 1e3:.1f} to {win[1] / 1e3:.1f} km", ha="center", fontsize=8, color=MUTED)
    ax.set_aspect("equal")
    ax.set_xlim(-4800, 4800)
    ax.set_ylim(-600, 7200)
    ax.set_xlabel("x, along the array (m)")
    ax.set_ylabel("y, broadside (m)")
    ax.set_title("popoto_far_mover: plan view (ownship and target motion drawn to exaggerated length)", loc="left", fontsize=10)
    fig.savefig(path, dpi=115)
    plt.close(fig)


# --- the report --------------------------------------------------------------------------------

def fmt(v, digits=1):
    return "—" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def stacked_rows(D) -> list:
    """Section 3's subsection on the optional stacked rows of the long-range Popoto array. Every
    number comes from tools/popoto_elevation_study.py (analysis) or tools/popoto_elevation_bench.py
    (measured on simulated data), through their JSON files in docs/design/."""
    S = json.loads((ROOT / "docs" / "design" / "popoto_elevation_study.json").read_text())
    B = json.loads((ROOT / "docs" / "design" / "popoto_elevation_bench.json").read_text())
    lr = D["popoto_long_range"]
    two, three, wide = S["chosen"]["two_rows"], S["chosen"]["three_rows"], S["chosen"]["wide"]
    up = S["largest_row_offset_m"]
    d = 2.0 * up
    acc = S["accuracy"]
    L = ["### Option: stacked rows on the long-range Popoto array (elevation and depth)\n"]
    L.append("One horizontal line of hydrophones measures one angle: the angle between an arrival and the line itself (the cone\n"
             "angle). It cannot tell an echo from above from one from below, it reports the bearing of anything off the\n"
             "horizontal a little wrong, and it knows nothing of depth. A second line of 32 hydrophones at a different height on the\n"
             "vehicle (or a third) measures the missing angle, in the active mode and in the passive one.\n")
    L.append("This is an option. One row stays the default; the software processes a one-row array exactly as before (the report\n"
             "lines are identical to the last digit, checked). The numbers below are from `tools/popoto_elevation_study.py`\n"
             "(analysis, with the estimator checked by Monte Carlo) and `tools/popoto_elevation_bench.py` (the software run on\n"
             "simulated pings). Nothing here has been on a vehicle.\n")
    L.append("**What is offered**\n")
    L.append("| | One row (default) | Two rows | Three rows |\n| --- | --- | --- | --- |")
    L.append("| Sonar file | `popoto_long_range` | `popoto_long_range_2row` | `popoto_long_range_3row` |")
    L.append(f"| Row heights about the hull's centreline (m) | 0 | {-up:+.2f}, {up:+.2f} | {-up:+.2f}, 0, {up:+.2f} |")
    L.append("| Hydrophones | 32 | 64 | 96 |")
    L.append("| Capture boards (8 inputs each, own clock, one synchronisation reference input summed in) | "
             + " | ".join(str(B["stream"][k]["boards"]) for k in "123") + " |")
    L.append("| Cables to the boards: one per hydrophone, plus the reference to every board | 32 + 4 | 64 + 8 | 96 + 12 |")
    rate = lr["raw_data_rate_mbit_s"]
    L.append(f"| Raw data rate while receiving, {lr['adc_rate_khz']:.1f} kHz × {lr['adc_bits']} bit (Mbit/s) | {rate:.0f} | {2 * rate:.0f} | {3 * rate:.0f} |")
    n = lr["samples_per_ping_per_channel"]
    L.append(f"| One ping to {lr['max_range_m'] / 1e3:.1f} km as recorded now, {n:,} samples per channel × 4 bytes (MB) | "
             + " | ".join(f"{k * 32 * n * 4 / 1e6:.0f}" for k in (1, 2, 3)) + " |")
    L.append(f"| The same per hour at one ping every {lr['min_ping_interval_s']:.1f} s (GB) | "
             + " | ".join(f"{k * 32 * n * 4 / 1e9 * 3600.0 / lr['min_ping_interval_s']:.0f}" for k in (1, 2, 3)) + " |")
    t = B["timing_scenario_size"]
    L.append(f"| Processing per ping, C++ chain on {B['cpu_threads']} CPU threads of a shared machine, pings of the `popoto_12km` scenario's size (s) | "
             + " | ".join(f"{t[k]['seconds_per_ping']:.2f}" for k in "123") + " |")
    full = B.get("timing_full_length") or {}
    if full:
        L.append(f"| The same for one full-length ping at the boards' rate ({full['1']['samples_per_channel']:,} real samples per channel), "
                 f"against the {lr['min_ping_interval_s']:.1f} s between pings (s) | " + " | ".join(f"{full[k]['seconds_per_ping']:.1f}" for k in "123") + " |")
    L.append(f"| One-sigma elevation at 13 / 20 / 30 dB (deg) | none | "
             + " | ".join(" / ".join(f"{acc[g]['sigma_deg'][k]:.2f}" for k in ("13.0", "20.0", "30.0")) for g in (two, three)) + " |")
    half = S["unambiguous_half_sector_deg"][f"{d:.3f}"]
    amb = S["ambiguity"]
    def lobe(g, a):
        v = amb[g][a]["second_lobe_db"]
        return "none" if v <= -99.0 else f"{v:.0f} dB"
    L.append("| Second elevation lobe inside ±30°, arrival at 0° / 10° / 20° | n/a | "
             + " | ".join(" / ".join(lobe(g, a) for a in ("0.0", "10.0", "20.0")) for g in (two, three)) + " |")
    gain = S["unchanged"]["detection_gain_db"]
    L.append(f"| Detection against one row, same false-alarm rate (dB) | 0 | {gain['2']['rows_added_as_power_db']:.1f} | {gain['3']['rows_added_as_power_db']:.1f} |")
    L.append(f"\nThe heights assume a round hull of {S['hull_diameter_m']:.3f} m (21 inch) with the rows 45° below and above its widest point. "
             "**The hull, its fairing and where hydrophones can be mounted are not confirmed**; the heights are in the sonar files under\n"
             "`unconfirmed`, and the layout takes any heights, any number of rows, and an offset outwards for each (a curved hull).\n")
    L.append("**How accurate, and what that is in depth.** The bound on elevation for one arrival in noise is\n"
             "σ(sin e) = 1 / (k √(2 ρ Σ(h − h̄)²)), with ρ the signal-to-noise ratio at one hydrophone after pulse compression, h the\n"
             "heights of all hydrophones and k the band's RMS wavenumber. For two rows d apart and a row's SNR ρ_row after its own\n"
             "beamforming that is λ / (π d √(2 ρ_row)): accuracy is bought with row separation and with SNR, nothing else. The\n"
             "estimator in the chain reaches the bound (Monte Carlo over bound: "
             f"{acc[two]['monte_carlo_over_bound_at_20_db']:.2f} at 20 dB; on simulated pings within 30%).\n")
    L.append("| Rows | 13 dB | 20 dB | 30 dB | No second lobe within (at 6 kHz) | Fits the assumed hull |\n| --- | --- | --- | --- | --- | --- |")
    for name, a in acc.items():
        if not name.startswith("2 rows"):
            continue
        sep = a["heights_m"][1] - a["heights_m"][0]
        hs = S["unambiguous_half_sector_deg"].get(f"{sep:.3f}")
        L.append(f"| {name} | {a['sigma_deg']['13.0']:.2f}° | {a['sigma_deg']['20.0']:.2f}° | {a['sigma_deg']['30.0']:.2f}° | "
                 + (f"±{hs[2]:.0f}°" if hs else "±90°") + f" | {'yes' if a['fits_hull'] else 'no'} |")
    L.append(f"\nThree or four rows inside the same {d:.2f} m are no more accurate than two ({acc[three]['sigma_deg']['20.0']:.2f}° at 20 dB): "
             "the top-to-bottom span sets the accuracy. What the middle row buys is the removal of the second lobe.\n")
    ds = S["depth_sigma_m"]
    snr = S["link_snr_db"]
    def cell(v):
        return "not detected" if v is None else f"{v:.0f} m" if v >= 10 else f"{v:.1f} m"
    L.append("Depth error is range × elevation error. With the link budget of Sec. 5 for one row (spherical spreading, noise only):\n")
    L.append("| Target, rows | 1 km | 6 km | 12 km |\n| --- | --- | --- | --- |")
    for label in snr:
        L.append(f"| *{label}: SNR* | {snr[label]['1000.0']:.0f} dB | {snr[label]['6000.0']:.0f} dB | {snr[label]['12000.0']:.0f} dB |")
        for g in (two, wide):
            r = ds[f"{label}; {g}"]
            L.append(f"| {label}; {g} | {cell(r['1000.0'])} | {cell(r['6000.0'])} | {cell(r['12000.0'])} |")
    L.append("\n(SNR above 40 dB is counted as 40 dB.) Read plainly: on a hull of this size the option gives **depth to a few metres at 1 km, "
             f"to about {ds['ship, +25 dB; ' + two]['6000.0']:.0f} m at 6 km on a ship-sized target, and at 12 km only whether the echo comes from\n"
             f"above or below, after several pings** ({ds['ship, +25 dB; ' + two]['12000.0']:.0f} m one-sigma per ping; a track averages its pings). "
             "Rows a metre apart would be 2.6 times better and do not fit.\nAt short range the limit is not noise but roll, mounting and the channel (below).\n")
    L.append("**Which lobe.** Two rows d apart cannot tell elevations λ / d apart in sin(e). Half a wavelength apart (0.125 m at 6 kHz) there is\n"
             f"no second lobe at all; at {d:.2f} m the sector without one is ±{half[2]:.0f}° at 6 kHz (±{half[0]:.0f}° at 4 kHz). The band smears the second lobe\n"
             f"only a little: with two rows it stands {-amb[two]['10.0']['second_lobe_db']:.1f} dB below the true one for an arrival at 10° and "
             f"{-amb[two]['20.0']['second_lobe_db']:.1f} dB at 20°, and at 13 dB the\n"
             f"estimator picks it {100 * amb[two]['20.0']['wrong_lobe_fraction']['13.0']:.0f}% of the time at 20°. The chain flags such detections "
             "(`elevation_ambiguous`) and keeps them out of the tracks' depth.\n"
             f"A third row halfway (rows {d / 2:.2f} m apart, less than a wavelength) leaves no lobe above {amb[three]['20.0']['second_lobe_db']:.0f} dB anywhere in ±30°.\n"
             "On a hull this small equal spacing is the better choice; unequal spacings pay only when the span is several wavelengths.\n")
    arr = S["arrival_angles_deg"]
    L.append("How steep arrivals are decides between two rows and three (straight rays; direct path / one bounce / two bounces):\n")
    L.append("| Water depth | 1 km | 3 km | 6 km | 12 km |\n| --- | --- | --- | --- | --- |")
    for depth in arr:
        L.append(f"| {float(depth):.0f} m | " + " | ".join("/".join(f"{v:.1f}" for v in arr[depth][r]) + "°" for r in ("1000.0", "3000.0", "6000.0", "12000.0")) + " |")
    L.append(f"\nIn shallow water beyond 1 km everything arrives within about 10° of level and **two rows are enough**. In deep water, or inside\n"
             "1 km, arrivals are steeper and **three rows are needed** to trust the lobe. A sand bottom reflects fully only below "
             f"{S['critical_grazing_deg_sand']:.0f}° grazing, so steeper bounce paths die out.\n")
    cb = S["cone_bias_deg"]
    L.append("**Bearing.** A line reports sin(cone) = cos(elevation) × sin(bearing): the bearing of an arrival off the horizontal is too small.\n"
             "With the elevation known the chain reports the horizontal bearing (`bearing_deg`; `cone_bearing_deg` keeps the old value).\n"
             "Error removed, in degrees:\n")
    L.append("| Elevation | 15° off broadside | 30° | 45° | 60° | Cross-range at 45°, 1 km / 12 km |\n| --- | --- | --- | --- | --- | --- |")
    for e in cb:
        b45 = np.radians(cb[e]["45.0"])
        L.append(f"| {float(e):.0f}° | " + " | ".join(f"{cb[e][b]:.2f}" for b in ("15.0", "30.0", "45.0", "60.0")) + f" | {1000 * b45:.0f} m / {12000 * b45:.0f} m |")
    L.append("\nIn shallow water at long range this is small (arrivals within 2° of level: 0.03°). It matters in deep water and close in.\n")
    mp = S["multipath"]
    c1 = mp["cases"]["target 50 m, 1 km"]
    L.append("**In a channel.** One target sends back several echoes: straight, by the surface, by the bottom, and by mixtures. "
             f"In 100 m of water with the array at 30 m and a target at 50 m depth 1 km away they arrive from {c1['direct']['elevation_deg']:+.1f}°, "
             f"{c1['surface']['elevation_deg']:+.1f}° ({c1['surface']['excess_range_m']:.1f} m later in range) and {c1['bottom']['elevation_deg']:+.1f}° "
             f"({c1['bottom']['excess_range_m']:.1f} m later).\n")
    L.append(f"- The rows' own beam in elevation is {mp['elevation_beamwidth_deg']:.0f}° wide. Arrivals are **never separated by elevation**, only by range "
             f"(resolution {mp['range_resolution_m']:.3f} m); elevation is then measured on each arrival that range has separated.\n"
             "- The direct echo is the first. The chain searches back to the leading edge of each detection and takes the first arrival that stands\n"
             "  clear (`elevation.arrival: first`), so the depth it reports is the direct path's.\n"
             "- Followed straight back, a surface echo points at the target's mirror image above the surface and a bottom echo at its image below\n"
             "  the bottom. Folded back into the water column both give the target's depth again, and where the image lay says which path it\n"
             "  was (`path`: direct, surface, bottom). A track whose depth stays outside the water column is marked multipath without needing its\n"
             "  parent track.\n"
             "- Direct and surface echoes are 2 × (array depth) × (target depth) / range apart. For a target near the surface they are one\n"
             f"  detection at every useful range (beyond {mp['direct_and_surface_merge_beyond_m']['4'] / 1e3:.1f} km for a target at 4 m; "
             f"{mp['direct_and_surface_merge_beyond_m']['50'] / 1e3:.0f} km at 50 m). The merged pair reads as a source at the surface,\n"
             "  which for a ship is the right answer and for a shallow diver or mine is a depth error of its own depth.\n"
             "- Echoes that go out by one path and back by another arrive between these, two at a time from different elevations at the same\n"
             "  range. Their elevation is a mixture; the quality figure drops and below `min_quality` none is reported.\n"
             "- The unweighted pulse has range sidelobes 13 dB down. In a channel the neighbours' sidelobes pull each arrival's elevation by up to\n"
             "  about a degree (17 m at 1 km): in multipath, that and not noise is the short-range accuracy.\n")
    sc = B["scenarios"]
    f, sh, re = sc["free_field"], sc["shallow"], sc["rehearsal"]
    L.append("Measured on simulated pings (C++ chain; the Python chain agrees to 0.01° and 0.2 m):\n")
    L.append("| Scenario | Truth | Reported |\n| --- | --- | --- |")
    L.append(f"| `popoto_elevation_free`: free field, vehicle rolled {f['roll_deg']:.0f}°, three targets near 1 km, {f['pings']} pings | depths 10, 60, 160 m | "
             f"track depth errors {', '.join(f'{e:+.1f}' for e in f['depth_errors_m'])} m (reported sigma {max(f['depth_sigma_m']):.1f} m) |")
    L.append(f"| `popoto_elevation_shallow`: 100 m of water, bounces to second order, target at 1 km | depth {sh['true_depth_m']:.0f} m, direct path {sh['true_elevation_deg']:+.2f}° | "
             f"depth {min(sh['depth_m']):.0f} to {max(sh['depth_m']):.0f} m over {len(sh['depth_m'])} pings, path {sh['path'][0]} |")
    L.append(f"| `popoto_elevation_rehearsal`: raw 102.4 kHz, {re['boards']} boards on their own clocks, rolled {re['roll_deg']:.0f}°, pitched {re['pitch_deg']:.0f}°, 100 m of water | "
             f"targets at {re['deep_true_m']:.0f} m and {re['shallow_true_m']:.0f} m | boards synchronised to {re['worst_residual_us']:.2f} µs; "
             f"{min(re['deep_depth_m']):.0f} to {max(re['deep_depth_m']):.0f} m; the shallow one {min(re['shallow_depth_m']):.1f} to {max(re['shallow_depth_m']):.1f} m (its direct and surface echoes merged) |")
    L.append("\n**Passive.** Listening (`--passive`, `--listen` with `elevation=1`) gives every bearing an elevation the same way: each row is steered at\n"
             "the bearing in every frequency bin and block of a frame, and the rows' beams are steered over elevation. On simulated broadband\n"
             "sources 1 km away at 20 m and 250 m depth, heard from 100 m with the vehicle rolled 3°, each 0.2 s frame gives both elevations within\n"
             "1.5° (`tests/python/test_elevation.py`). The second-lobe and roll remarks above apply unchanged; there is no range, so no depth,\n"
             "from one position.")
    rf = S["refraction_error_m"]
    k0 = next(iter(rf))
    L.append("\n**Bent rays.** By default depth is range × the sine of the level elevation: a straight ray. In a sound-speed gradient g the ray curves\n"
             "with radius c / g and the straight line is wrong by about range² × g / 2c: "
             + "; ".join(f"{abs(rf[k0][r][0]):.0f} m at {float(r) / 1e3:.0f} km" for r in rf[k0]) + f" in {k0.split(',')[0]} ({k0.split(', ')[1]}),\n"
             "far more through a thermocline. Given a profile (`elevation.ssp`) the chain follows the ray through it instead, with turning\n"
             "points and reflections. **Beyond a few kilometres depth needs the profile**; the vehicle has to supply one (CTD cast or model).\n")
    se = S["sensitivity"]
    L.append("**What the vehicle and the mounting must provide.** Elevation is measured against the vehicle; depth needs it against the horizon.\n")
    L.append("| Error source | Elevation error | Depth error at 1 km / 6 km / 12 km |\n| --- | --- | --- |")
    for a, v in se["roll_depth_error_m"].items():
        L.append(f"| Roll known to {a}° | {a}° | {v[0]:.1f} / {v[1]:.0f} / {v[2]:.0f} m |")
    for label, key, unit in (("One row further out than the drawing by", "out_offset_error_deg", " mm"),
                             ("Phase mismatch between the rows' channels of", "phase_mismatch_error_deg", "° of phase"),
                             ("Clock error between boards on different rows of", "board_clock_error_deg", " µs")):
        for a, v in se[key].items():
            if key == "board_clock_error_deg" and a not in ("0.5", "1.0", "8.3"):
                continue
            if key != "board_clock_error_deg" and a not in ("1.0", "5.0"):
                continue
            L.append(f"| {label} {a}{unit} | {v:.2f}° | " + " / ".join(f"{r * np.radians(v):.0f}" for r in (1000.0, 6000.0, 12000.0)) + " m |")
    L.append(f"\n- **Roll.** Roll error is elevation error one for one. The attitude sensor's roll accuracy is the first number to ask the vehicle\n"
             "  team for; it is `elevation.roll_sigma_deg` in the deployment file (0.2° assumed, unconfirmed) and goes into every reported depth\n"
             "  sigma. **Not done yet:** the chain applies the roll navigation gives for the transmit time. An echo from 12 km arrives 16 s\n"
             "  later; if the vehicle rolls in between, the difference is a depth error. Until roll is applied at each echo's arrival time the\n"
             "  depths hold for a vehicle that keeps its roll over a ping.\n"
             "- **Mounting.** The rows' heights matter little (a 5 mm error in separation is "
             f"{se['separation_error_5mm_at_10deg_deg']:.2f}° at 10° elevation, nothing at 0°). Their offset outwards\n"
             "  matters a great deal: 1 mm is 0.15°. No drawing will be that good; the offset is calibrated in water (below).\n"
             "- **Synchronisation.** The present tolerance between boards (a twentieth of a period, 8.3 µs) is fine for bearing and far too\n"
             f"  loose for elevation. The reference scheme reaches {re['worst_residual_us']:.2f} µs in simulation with 300 µs offsets and 25 ppm drifts (0.1°). Wiring\n"
             "  each board to four hydrophones of each row at the same stations (`array.interleave: 4`) makes the rows of a pair share a\n"
             "  clock and removes the term for two rows; the price is that a dead board then takes four stations out of both rows.\n"
             "- **Calibration.** The existing calibration (`calibrate.py`) fits each channel's gain, phase and position from known sources. With\n"
             "  rows it must be run with sources at several elevations as well as bearings (a source lowered to three depths at three\n"
             "  bearings); in simulation it recovers a row mounted 5 mm out to 0.4 mm and removes the 0.75° it caused.\n"
             "- **Dead channels.** Rows are found from the hydrophone positions in the record, whatever the wiring. `chain.dead_channels` leaves\n"
             "  channels out; a dead hydrophone or a dead board shortens a row and the estimate carries on (tested with one hydrophone and\n"
             "  one board out: depths within 5 m at 1 km).\n")
    st = B["stream"]
    L.append("**Data path.** " + f"Checked at the boards' rate (102.4 kHz, 24 bit) with 64 and 96 channels: board blocks over UDP "
             f"({st['2']['udp_block_samples']} and {st['3']['udp_block_samples']} samples a datagram, {st['2']['udp_blocks_per_second']:.0f} and "
             f"{st['3']['udp_blocks_per_second']:.0f} datagrams a second), the C++ and Python assemblers,\n"
             f"Sensor Bridge frames ({st['2']['hsb_frame_bytes']:,} and {st['3']['hsb_frame_bytes']:,} bytes for 1600 samples), the container file and the C++ chain all carry them unchanged\n"
             "(samples back within half a count). The record format already stores every hydrophone's position: no format change.\n"
             "The elevation stage runs on the backprojection kernel (host or CUDA), so it needs no GPU code of its own; it has not been run on a GPU.\n")
    if full:
        L.append(f"**Compute.** Cost grows in proportion to the rows. On pings already at baseband (8 kHz complex, the scenario's size) the C++ chain takes "
                 f"{t['1']['seconds_per_ping']:.1f}, {t['2']['seconds_per_ping']:.1f} and {t['3']['seconds_per_ping']:.1f} s for one, two and three rows.\n"
                 f"On one full-length ping as the boards deliver it (102.4 kHz real, 0.5 to 13 km) it takes {full['1']['seconds_per_ping']:.0f}, "
                 f"{full['2']['seconds_per_ping']:.0f} and {full['3']['seconds_per_ping']:.0f} s on {B['cpu_threads']} CPU threads of a shared machine, "
                 f"against {lr['min_ping_interval_s']:.1f} s between pings:\n"
                 "**on a CPU of this class the chain does not keep up with two or three rows at the boards' full rate, and barely with one.** "
                 "The band is 2 kHz wide, so the\nsamples carry 13 times more than needed: converting to baseband and decimating before the chain "
                 "(on the boards or in a first stage on the\nvehicle computer), or a GPU (the one-row map was 17 times faster on an L4, `RESULTS.md`), "
                 "closes the gap. Neither is done for rows; it is a\ndecision to take with the vehicle computer.\n")
    L.append("**What it does not improve.** Bearing resolution and range resolution (the array is no longer and the band no wider). "
             "Left/right: rows in one vertical plane cannot tell port from starboard; the hull behind the array does that, as now. "
             f"Detection range, much: {gain['2']['rows_added_as_power_db']:.1f} dB with two rows, roughly a tenth more range under spherical spreading. "
             "Depth at 12 km in shallow water.\n")
    L.append("**Unconfirmed, and for the owner to decide** (all in config under `unconfirmed`, none in code): the hull diameter and where rows can be\n"
             "mounted, which sets the separation and with it every accuracy above; whether 8 or 12 boards, their power and their cabling fit the\n"
             "vehicle; row-by-row or interleaved wiring; the attitude sensor's roll accuracy and its time stamp; whether the vehicle can supply\n"
             "water depth and a sound-speed profile; and, as for one row, whether Popoto boards can be batched and synchronised at all.\n")
    return L


def write_report(D, reqs, out_md):
    fls = ["fls_120k", "fls_170k", "fls_330k"]
    L = []
    L.append("# Design\n")
    L.append("**Generated by `tools/design_report.py` from the files under `configs/`. Do not edit by hand:** change a\n"
             "config (or the tool) and run it again. `tests/python/test_design.py` fails if a requirement below is\n"
             "not met by the configs or if this file is out of date.\n")
    L.append("Nothing here has been built or measured. Every number is either a requirement from the owner, a value\n"
             "chosen in a config file, or arithmetic on those. Published figures used for comparison are marked.\n")
    L.append("## 1. How the pieces fit\n")
    L.append("![System block diagram](design/system.png)\n")
    L.append("- **One record format.** Every source produces ping records; every stage takes and returns them.\n"
             "- **One description per sonar.** `configs/sonars/<name>.yaml` holds the array (element count, pitch, axis,\n"
             "  boresight, field of view), the transmit (band, pulse, source level) and the sampling (rate, bits, window,\n"
             "  boards). The simulator, the processing and this report all read that file and nothing else.\n"
             "- **One description per scenario.** `configs/scenarios/<name>.yaml` names a sonar, the platform's motion,\n"
             "  the scene, the propagation model, reverberation and noise.\n"
             "- **Runs on a CPU.** Every stage has a host implementation; the CUDA backend is optional and gives the same\n"
             "  results. `SCENARIOS.md` lists measured CPU run times.\n")
    L.append("## 2. Requirements and where each is met\n")
    L.append("| Id | Requirement (owner) | Set in | Value found | Met |\n| --- | --- | --- | --- | --- |")
    for r in reqs:
        L.append(f"| {r['id']} | {r['requirement']} | `{r['set_in']}` | {r['value']} | {'yes' if r['met'] else '**NO**'} |")
    L.append("\nEach row is checked by `tests/python/test_design.py`; the scenarios are run by "
             "`tests/python/test_realistic_scenes.py` (thinned) and `tools/gallery.py` (full size).\n")
    L.append("## 3. Arrays and beams\n")
    L.append("![Arrays and beams](design/arrays.png)\n")
    L.append("| Quantity | " + " | ".join(f"`{n}`" for n in DESIGN_SONARS) + " | How it is derived |\n| --- |" + " --- |" * (len(DESIGN_SONARS) + 1))
    rows = [("Centre frequency (kHz)", "centre_frequency_khz", 0, "config: mean of `f_start`, `f_stop`"),
            ("Bandwidth (kHz)", "bandwidth_khz", 0, "config"),
            ("Wavelength (mm)", "wavelength_mm", 2, "c / f at 1500 m/s"),
            ("Elements", "elements", 0, "config: `array.count`"),
            ("Pitch (mm)", "pitch_mm", 1, "config: `array.pitch`"),
            ("Pitch at the top of the band (wavelengths)", "pitch_in_wavelengths_at_top_of_band", 3, "pitch / (c / f_stop)"),
            ("Steering with no grating lobe (deg, each side)", "grating_lobe_free_steering_deg", 0, "asin(wavelength / pitch − 1)"),
            ("Array length (m)", "array_length_m", 3, "elements × pitch"),
            ("Beamwidth at broadside (deg)", "beamwidth_deg", 2, "0.886 wavelength / length"),
            ("Cross-range cell at full range (m)", "cross_range_at_max_range_m", 2, "range × beamwidth"),
            ("Range resolution (cm)", "range_resolution_cm", 2, "c / 2B, before windowing"),
            ("Horizontal field of view (deg)", "element_fov_az_deg", 0, "config: `fov.horizontal_deg` or element width"),
            ("Vertical field of view (deg)", "element_fov_el_deg", 0, "config: `fov.vertical_deg` or element height"),
            ("Boresight above horizontal (deg)", "tilt_up_deg", 1, "config: `fov.tilt_up_deg` or `array.boresight`"),
            ("Element width along the array (mm)", "element_width_mm", 1, "config, or 0.886 wavelength / horizontal field of view"),
            ("Element height (mm)", "element_height_mm", 1, "config, or 0.886 wavelength / vertical field of view"),
            ("Capture boards", "boards", 0, "config: `receive.board_size`")]
    for label, key, dig, how in rows:
        L.append(f"| {label} | " + " | ".join(fmt(float(D[n][key]) if isinstance(D[n][key], (int, float)) and dig else D[n][key], dig) if D[n][key] is not None else "omni" for n in DESIGN_SONARS) + f" | {how} |")
    d = D["sas_minsas120"]
    L.append(f"\nSAS-specific (`sas_minsas120`): along-track resolution {d['along_track_resolution_cm']:.1f} cm (half the "
             f"{2 * d['along_track_resolution_cm']:.1f} cm transmit aperture); transmit beam {d['transmit_beamwidth_deg']:.1f} degrees, "
             f"so the synthetic aperture is {d['synthetic_aperture_at_50_m']:.1f} m long at 50 m; the platform may advance at most "
             f"{d['max_advance_per_ping_m']:.3f} m per ping (half the array), which at the full {d['max_range_m']:.0f} m range allows "
             f"{d['max_speed_at_max_range_m_s']:.2f} m/s. Kraken publishes 40 kHz, a 109 cm array, 3.3 cm × 3.0 cm and 2 to 6 knots for "
             "the MINSAS 120; the 330 kHz centre frequency is the owner's figure; element count, pitch and element sizes are choices made here.\n")
    L.append("The Popoto array is shown as its scenario uses it (32 hydrophones, 6–10 kHz). `array.count` sets the size; "
             "boards of eight inputs follow.\n")
    L.extend(stacked_rows(D))
    L.append("## 4. Timing and data\n")
    L.append("| Quantity | " + " | ".join(f"`{n}`" for n in DESIGN_SONARS) + " | How it is derived |\n| --- |" + " --- |" * (len(DESIGN_SONARS) + 1))
    for label, key, dig, how in [("Full range (m)", "max_range_m", 0, "config: `receive.window.range` (scenario window for the Popoto array)"),
                                 ("Pulse length (ms)", "pulse_ms", 1, "config"),
                                 ("Shortest ping interval (s)", "min_ping_interval_s", 3, "2 × range / c + pulse"),
                                 ("ADC rate (kHz)", "adc_rate_khz", 1, "config: `receive.sample_rate`"),
                                 ("ADC bits", "adc_bits", 0, "config"),
                                 ("Samples per ping per channel", "samples_per_ping_per_channel", 0, "interval × rate"),
                                 ("Raw data rate while receiving (Mbit/s)", "raw_data_rate_mbit_s", 1, "elements × rate × bits")]:
        L.append(f"| {label} | " + " | ".join(fmt(float(D[n][key]) if dig else D[n][key], dig) for n in DESIGN_SONARS) + f" | {how} |")
    L.append("\nThe simulations take the receive chain as already at complex baseband (twice the bandwidth), which is what a\n"
             "front end with digital down-conversion would deliver; the ADC rates above are for sampling the carrier directly.\n")
    L.append("## 5. Link budgets\n")
    L.append("![Link budgets](design/link_budgets.png)\n")
    L.append("Output SNR = SL − 2 TL + TS − NL + array gain + 10 log10(T), with spherical spreading plus Francois–Garrison\n"
             f"absorption and the four-term ambient-noise fit at {WIND:.0f} m/s wind (`gpusonar.sonar_equation`).\n")
    L.append("| Quantity | " + " | ".join(f"`{n}`" for n in DESIGN_SONARS) + " |\n| --- |" + " --- |" * len(DESIGN_SONARS))
    for label, key, dig in [("Source level (dB re 1 µPa at 1 m)", "source_level_db", 0), ("Absorption (dB/km)", "absorption_db_per_km", 1),
                            ("Noise spectrum level (dB re 1 µPa²/Hz)", "noise_spectrum_db", 1), ("Target strength assumed (dB)", "target_strength_db", 0),
                            ("Array gain (dB)", "array_gain_db", 1), ("Pulse-compression gain, 10 log10(BT) (dB)", "processing_gain_db", 1),
                            ("**SNR at full range (dB)**", "snr_at_max_range_db", 1), ("Range at which SNR falls to 13 dB (m)", "range_for_13_db_m", 0)]:
        L.append(f"| {label} | " + " | ".join(fmt(float(D[n][key]), dig) for n in DESIGN_SONARS) + " |")
    L.append("\nTarget assumed: " + "; ".join(f"`{n}`: {TARGET_NOTE[n]}" for n in DESIGN_SONARS) + ".\n")
    d = D["sas_minsas120"]
    L.append(f"- **SAS:** with a {d['source_level_db']:.0f} dB source a single cell of seabed stays 13 dB above noise out to "
             f"{d['range_for_13_db_m']:.0f} m, not the {d['max_range_m']:.0f} m of the receive window "
             f"({d['snr_at_max_range_db']:.0f} dB there). Objects are much stronger than one cell of sand. The array gain "
             "is evaluated at full range and is smaller closer in, so the curve is optimistic at short range.\n"
             "- **170 kHz:** specified to 450 m after the first budget showed 14 dB at 600 m.\n")
    study = json.loads((ROOT / "docs" / "design" / "popoto_range_study.json").read_text())
    cur, lr = study["current_array"], D["popoto_long_range"]
    L.append("### Popoto array to 12 km: which frequency and how much power\n")
    L.append("![Source level needed for 12 km against frequency](design/popoto_range_study.png)\n")
    L.append(f"From `tools/popoto_range_study.py`: target +{study['target_strength_db']:.0f} dB, {study['pulse_s']:.0f} s pulse, "
             f"{study['snr_db']:.0f} dB at the beamformer output, ambient noise at 5 m/s wind, spherical spreading plus absorption.\n")
    L.append(f"- **The present band cannot do it.** At {cur['frequency_hz'] / 1e3:.1f} kHz absorption alone costs about "
             f"{float(acoustics.absorption(cur['frequency_hz'], 12.0, 35.0, 50.0, model='ainslie_mccolm')) * 24000.0:.0f} dB over the 24 km round trip; 12 km would need "
             f"{cur['source_level_needed_at_range_db']:.0f} dB. With {cur['hydrophones']} hydrophones at {cur['source_level_db']:.0f} dB "
             f"that band reaches about {cur['reach_at_its_source_level_m'] / 1e3:.1f} km on this budget.\n"
             "- **The frequency has to come down to 4 to 6 kHz.** Below that, ambient noise rises and a half-wavelength array of a\n"
             "  given count gets long; above it, absorption takes over.\n")
    L.append("| Array (half-wavelength spacing) | Best frequency | Array length | Source level for 12 km | Acoustic power, omnidirectional projector |\n| --- | --- | --- | --- | --- |")
    for n, v in study["fixed_count"].items():
        L.append(f"| {n} hydrophones | {v['best_frequency_hz'] / 1e3:.1f} kHz | {v['array_length_m']:.1f} m | {v['source_level_db']:.0f} dB | {v['acoustic_power_w_omni']:.0f} W |")
    for a, v in study["fixed_aperture"].items():
        L.append(f"| {float(a):.1f} m aperture ({v['hydrophones']} hydrophones) | {v['best_frequency_hz'] / 1e3:.1f} kHz | {float(a):.1f} m | {v['source_level_db']:.0f} dB | {v['acoustic_power_w_omni']:.0f} W |")
    L.append(f"\n- **Chosen (`popoto_long_range`):** {lr['elements']} hydrophones at {lr['pitch_mm'] / 10:.0f} cm ({lr['array_length_m']:.1f} m), "
             f"{lr['centre_frequency_khz'] - lr['bandwidth_khz'] / 2:.0f} to {lr['centre_frequency_khz'] + lr['bandwidth_khz'] / 2:.0f} kHz, "
             f"{lr['pulse_ms'] / 1e3:.0f} s, {lr['source_level_db']:.0f} dB: {lr['_eq'].snr(12000.0):.0f} dB at 12 km on this budget. "
             "Acoustic power is 10^((SL - 170.8 - DI) / 10) W; a projector with directivity needs less, and electrical power is\n"
             "  higher by the transducer's efficiency (not known until a projector is chosen). Doubling the hydrophones saves 3 dB;\n"
             "  doubling the pulse length saves 3 dB.\n"
             "- **Spherical spreading is the cautious case.** In shallow water sound is trapped between surface and bottom:")
    if study["bellhop"]:
        L.append("  Bellhop for 100 m of isovelocity water over sand, 3 m aperture, gives\n\n"
                 "| Frequency | One-way loss, Bellhop | One-way loss, spherical | Source level for 12 km (noise only) |\n| --- | --- | --- | --- |")
        for b in study["bellhop"]:
            L.append(f"| {b['frequency_hz'] / 1e3:.0f} kHz | {b['one_way_loss_db']:.1f} dB | {b['spherical_one_way_loss_db']:.1f} dB | {b['source_level_db']:.0f} dB |")
        L.append("\n  That is 14 dB less loss each way, but it holds only for that depth, bottom and profile, and at those levels\n"
                 "  reverberation, not noise, decides detection. The `popoto_env_*` scenarios (`SCENARIOS.md`) show how much the\n"
                 "  environment moves the answer: a downward-refracting summer profile over mud loses far more than spherical spreading.\n")
    L.append("- The boards sample at 102.4 kHz and need no change for this band; the hydrophones, the projector and its power\n"
             "  amplifier do. None is chosen; the values are in `configs/sonars/popoto_long_range.yaml` under `unconfirmed`.\n")
    L.append("- These are noise-limited figures. Reverberation, which the scenarios include, usually sets the limit at short\n"
             "  and medium range.\n- Source levels are design values, not measured; the front-end sensitivities and gains in the configs are placeholders.\n"
             "- For the Popoto array the simulation uses Bellhop multipath, which in shallow water carries more energy to long\n"
             "  range than spherical spreading (see `RESULTS.md`).\n")
    L.append("## 6. Scenario geometry\n")
    for n in fls:
        L.append(f"![{SCENARIO_OF[n]}](design/geometry_{SCENARIO_OF[n]}.png)\n")
    L.append("![sas_minsas_survey](design/geometry_sas_minsas_survey.png)\n")
    L.append("![popoto_far_mover](design/geometry_popoto_far_mover.png)\n")
    L.append("Every scenario, including those not shown here, has a plan and side view drawn from its file by\n"
             "`tools/scenario_diagrams.py` in `docs/scenarios/`. Simulated images, animations, detections and tracks are in\n"
             "`SCENARIOS.md`.\n")
    return "\n".join(L)


def build(write=True):
    D = {n: derive(n) for n in DESIGN_SONARS}
    reqs = requirements(D)
    out = ROOT / "docs" / "design"
    text = write_report(D, reqs, ROOT / "docs" / "DESIGN.md")
    if write:
        out.mkdir(parents=True, exist_ok=True)
        draw_system(out / "system.png")
        draw_arrays(D, out / "arrays.png")
        draw_link_budgets(D, out / "link_budgets.png")
        for n in ("fls_120k", "fls_170k", "fls_330k"):
            draw_fls_geometry(n, out / f"geometry_{SCENARIO_OF[n]}.png")
        draw_sas_geometry(out / "geometry_sas_minsas_survey.png")
        draw_popoto_geometry(out / "geometry_popoto_far_mover.png")
        (out / "derived.json").write_text(json.dumps(
            {"sonars": {n: {k: v for k, v in d.items() if not k.startswith("_")} for n, d in D.items()},
             "requirements": reqs}, indent=2))
    return D, reqs, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    D, reqs, text = build()
    md = ROOT / "docs" / "DESIGN.md"
    tail = md.read_text().split("<!-- measured -->", 1)[1] if md.exists() and "<!-- measured -->" in md.read_text() else ""
    md.write_text(text + ("\n<!-- measured -->" + tail if tail else ""))
    bad = [r for r in reqs if not r["met"]]
    for r in reqs:
        print(("ok   " if r["met"] else "FAIL ") + r["id"] + ": " + r["value"])
    if args.check and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
