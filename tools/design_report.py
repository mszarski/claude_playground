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
        "source_level_db": float(w.get("source_level_db_rms", w.get("source_level_db"))),
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
        seen.add((e["depth"], e["bottom_speed"], tuple(e.get("ssp_speed") or [e["sound_speed"]])))
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


STAGE_FILES = {   # sonar file -> (deployment file, simulated trials)
    "popoto_testauv_1row": ("popoto_testauv_1row", "`popoto_testauv_trial_1row`"),
    "popoto_static_2row": ("popoto_static_2row", "`popoto_static_trial_2row`"),
    "popoto_static_3row": ("popoto_static_3row", "`popoto_static_trial_3row`, `popoto_static_free`, `popoto_static_passing`"),
    "popoto_face_1row": ("popoto_face_1row", "`popoto_12km` (the same row as `popoto_long_range`)"),
    "popoto_long_range_2row": ("popoto_face_2row", "`popoto_face_2row_shallow`, `popoto_12km_2row`"),
    "popoto_long_range_3row": ("popoto_vehicle_elevation", "`popoto_elevation_free`, `_shallow`, `_rehearsal`, `popoto_12km_3row`"),
}


def _compute_options_text() -> str:
    """The two compute options with the measured figures there are (docs/benchmarks, docs/design)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gpusonar import buildspec
    t = buildspec.measured_timings()
    gpu = t.get("gpu")
    out = ""
    if gpu:
        h = gpu["header"]
        out += (f"Measured on an {h['gpus'][0]} (`benchmarks/gpu_benchmark_all_l4.md`): the C++ chain takes {gpu['seconds']['1']:.3f}, {gpu['seconds']['2']:.3f} and {gpu['seconds']['3']:.3f} s a ping for one, two and three rows "
                f"against {gpu['host_seconds']['1']:.2f}, {gpu['host_seconds']['2']:.2f} and {gpu['host_seconds']['3']:.2f} s on that machine's {h['host_threads']} CPU threads ({gpu['host_seconds']['1'] / gpu['seconds']['1']:.0f} times for one row), on pings of scenario size "
                f"({gpu['samples']:,} baseband samples a channel) and with the row heights before the 1.5 m change. Raw pings at the boards' full rate have not been run on a GPU. ")
    out += ("Decimate-first timings: `design/popoto_stage2.json`." if t.get("stage2") else "The decimate-first stage is being built separately: **its timing is pending** (`design/popoto_stage2.json` does not exist yet).")
    return out


def _surface_line() -> str:
    """One line for the trade space from the depth plan (docs/design/popoto_depth_planner.json)."""
    path = ROOT / "docs" / "design" / "popoto_depth_planner.json"
    if not path.exists():
        return "pending (`tools/popoto_depth_planner.py --shipped`).\n"
    P = json.loads(path.read_text())
    out = []
    for name in ("popoto_face_1x32", "popoto_face_2x32", "popoto_face_3x32"):
        pl = P["builds"].get(name, {}).get("plans", {}).get("deep_no_bottom_return", {})
        if "classes" in pl:
            c = pl["classes"]["ship"]
            out.append(f"{pl['rows']} row{'s' if pl['rows'] > 1 else ''}: {c['at_own_depth']['10']['margin_db']:+.0f} dB at 10 m, {c['recommended']['margin_db']:+.1f} dB at {c['recommended']['own_depth_m']:g} m")
    return ("the ship class at 12 km, margin over 13 dB, " + "; ".join(out) + " (*Surface interference, own depth, and target classes*; every target figure assumed).\n") if out else "not planned.\n"


def stacked_rows(D) -> list:
    """Section 3's subsection on the Popoto long-range array as a staged build with stacked rows. Every number
    comes from tools/popoto_elevation_study.py (analysis) or tools/popoto_elevation_bench.py (the software on
    simulated data), through their JSON files in docs/design/, or from the config files."""
    S = json.loads((ROOT / "docs" / "design" / "popoto_elevation_study.json").read_text())
    B = json.loads((ROOT / "docs" / "design" / "popoto_elevation_bench.json").read_text())
    lr = D["popoto_long_range"]
    two, three = S["chosen"]["two_rows"], S["chosen"]["three_rows"]
    up, mid, d = S["outer_row_m"], S["three_rows"]["best_middle_m"], S["baseline_m"]
    f_lo, f_hi = S["band_hz"]
    acc, T, W3 = S["accuracy"], S["two_rows"], S["three_rows"]
    stages = S["stages"]
    sc = B["scenarios"]
    names = list(stages)
    face2, face3 = (next(v for v in stages.values() if v["sonar"] == n) for n in ("popoto_long_range_2row", "popoto_long_range_3row"))
    auv = next(v for v in stages.values() if v["sonar"] == "popoto_testauv_1row")
    static3 = next(v for v in stages.values() if v["sonar"] == "popoto_static_3row")
    face_cfg = load_sonar_config("popoto_long_range_3row")
    static_cfg = load_sonar_config("popoto_static_3row")
    TS = S["trade_space"]
    keys = TS["keys"]
    g2, g3 = (S["unchanged"]["detection_gain_db"][k]["rows_added_as_power_db"] for k in ("2", "3"))
    margin2 = T["margin_db"]
    se = S["sensitivity"]

    def km(v):
        return f"{v / 1e3:.1f}"

    def lst(values, digits=0):
        return ", ".join("none" if v is None else f"{v:.{digits}f}" for v in values)

    def trade_table():
        out = ["| Channels | As | Beam | Map gain | Reach ship / small (km) | Elevation at 20 dB | Boards |\n| --- | --- | --- | --- | --- | --- | --- |"]
        for label, v in TS["fixed_channels"].items():
            out.append(f"| {v['channels']} | {label} | {v['beamwidth_deg']:.1f}° | {v['gain_isotropic_noise_db'] + v['rows_detection_gain_db']:.1f} dB | {km(v['reach_ship_m'])} / {km(v['reach_small_m'])} | "
                       + (f"{v['elevation_sigma_deg_at_20_db']:.2f}° (repeat {v['highest_repeat_db']:.1f} dB)" if v["rows"] > 1 else "none") + f" | {v['boards']} |")
        return out

    plat = face_cfg["platform"]
    L = ["### The long-range Popoto array as a staged build: one row, then stacked rows (elevation and depth)\n"]
    L.append("One horizontal line of hydrophones measures one angle, the angle between an arrival and the line (the cone angle). It cannot tell an\n"
             "echo from above from one from below, reports the bearing of anything off the horizontal a little wrong, and knows nothing of depth.\n"
             "More lines at other heights measure the missing angle, in the active mode (the main one) and in the passive one.\n")
    L.append("**What the owner has said, and how it is read here.**\n\n"
             "| The owner's words | Read as | Where it is |\n| --- | --- | --- |\n"
             f"| \"I can have a vertical straight face 2m high and 9m long\" | the full vehicle offers a flat vertical face of {plat['height_m']:g} m × {plat['span_m']:g} m (stated) | `platform` in `popoto_long_range_2row/_3row.yaml` |\n"
             f"| \"Need way more margin. Projector etc, I think probably try like a 7m horizontal and the vertical spacing is maybe a 1.5\" | the rows keep inside {plat['usable_span_m']:g} m of the length and the outer rows are {plat['row_baseline_m']:g} m apart; the rest of the face is margin and room for the projector. Working values (\"probably\", \"maybe\"): both are parameters, and the cost of each is shown below | `platform.usable_span_m`, `row_baseline_m`; `--array-length`, `--baseline` of the study |\n"
             "| \"Let's eval both two and three\" | two rows and three rows are both evaluated and both shipped, on the static array and on the face | the stages below |\n"
             "| \"No space for second row on auv but static drop in array could have two or three rows. So let's consider both\" | the test AUV carries ONE row. Stacked rows go first on a static drop-in array: a frame lowered and left in place, with two or with three rows. The owner gave no dimensions for it: all parameters | stages 2 and 3; `popoto_static_*` |\n"
             "| \"Probably row by row\" | each board serves one row (default); four of each row per board stays an alternative | `array.interleave` |\n"
             "| \"Sure make it configurable\" (roll accuracy and its time stamp); \"good in general, sprintnav IMU and dvl\" | the vehicle's navigation is a SPRINT-Nav (inertial navigation with a Doppler velocity log). Its roll accuracy, output rate and time stamp are NOT known here and are not guessed: they are settings, with the depth error against roll accuracy tabulated below so the datasheet figure can be dropped in | `elevation.roll_sigma_deg`, `attitude_time` |\n"
             "| \"We wont have a long range SVP but we can measure auv local speed of sound and we will have some idea of water depth\"; of the isothermal assumption: \"Won't always we need to handle it\" | local sound speed and an approximate water depth come from the vehicle; the profile along the path is unknown, is not always isothermal, and is carried as an uncertainty unless something better is known | `sound_speed`, `water_depth`, `water_depth_sigma_m`, `gradient_per_s`, `gradient_sigma_per_s`, `ssp` |\n"
             "| \"the popoto at 9m x 2 is still for active as well right\" | yes: active pinging is the main mode on the face; listening is supported with the same rows | |\n"
             "| \"Does our array need to be 64? We can add just need to understand what we get\" | the number of hydrophones is open; the trade space says what each addition buys | *Trade space* |\n"
             "| (compute at the boards' full rate) \"Prepare both options\" | both decimating first and a GPU are to be prepared | *Data path and compute* |\n"
             "| \"Keep at 32, can start with a smaller array in the 7m zone. Probably makes sense to keep the one design for auv and static right? Except for elevation.\" then \"Yes treat auv special case and have static array the longer row.\" | 32 hydrophones a row, fixed. The 7 m is a zone, not a length to fill. ONE row design, the long module (32 at 0.15 m, 4.8 m), for the static array and the face; the test AUV's short row is the one exception. \"A smaller array\" on the face is fewer rows of that module: one, then two, then three | *Row modules*; `array.module` in `configs/builds/` |\n"
             "| \"What can we do about the projector? Just blank?\" | an options analysis of the projector's direct sound | *The projector's direct sound* |\n"
             "| \"What does it look like for popoto at 12km though. Need that design\" | the 12 km design point in one place | *The 12 km design point* |\n")
    L.append("**Trade space in one table** (the face, rows within {:.0f} m, outer rows {:.2f} m apart, {:.0f}–{:.0f} kHz; noise-limited reach on the budget of Sec. 5; details further down):\n".format(
        S["array_length_m"], d, f_lo / 1e3, f_hi / 1e3))
    L.extend(trade_table())
    L.append("\nFor a fixed number of channels the trade is bearing and range (one long row) against elevation (stacked rows).\n")
    L.append("**The stages.** One software path, both engines, takes every stage from its files alone; each stage has a sonar file, a deployment file\n"
             "and simulated trials that the tests run.\n")
    L.append("| | " + " | ".join(f"Stage {n}" for n in names) + " |\n| --- |" + " --- |" * len(names))

    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(fn(stages[n]) for n in names) + " |")

    row("Sonar file", lambda x: f"`{x['sonar']}`")
    row("Deployment file", lambda x: f"`{STAGE_FILES[x['sonar']][0]}`")
    row("Simulated trials", lambda x: STAGE_FILES[x["sonar"]][1])
    row("Mounting", lambda x: f"{x['platform'].get('mounting', '')}, {x['platform'].get('span_m', 0):g} m × {x['platform'].get('height_m', 0):g} m")
    row("Under way", lambda x: "no: fixed pose" if x["platform"].get("static") else "yes")
    row("Hydrophones (rows × per row)", lambda x: f"{x['channels']} ({x['rows']} × {x['hydrophones_per_row']})")
    row("Capture boards of 8 inputs", lambda x: str(x["boards"]))
    row("Row length, spacing", lambda x: f"{x['row_length_m']:.2f} m, {x['pitch_m'] * 100:.0f} cm ({x['pitch_wavelengths_at_6khz']:.2f} λ at {f_hi / 1e3:.0f} kHz)")
    row("Row heights about the centre (m)", lambda x: ", ".join(f"{h:+.2f}" for h in x["heights_m"]) if x["rows"] > 1 else "0")
    row("Beam at broadside", lambda x: f"{x['beamwidth_deg']:.1f}°")
    row("Row's gain against sea noise / against the boards' own noise (dB)", lambda x: f"{x['gain_isotropic_noise_db']:.1f} / {x['gain_uncorrelated_noise_db']:.1f}")
    row("Pulse, and the range it hides", lambda x: f"{x['pulse_s']:g} s, {x['blind_range_m']:.0f} m")
    row("Noise-limited reach, ship (+25 dB) / small target (+5 dB), km", lambda x: f"{km(x['reach_ship_m'])} / {km(x['reach_small_m'])}")
    row("Elevation, one sigma at 20 dB", lambda x: f"{x['elevation_sigma_deg_at_20_db']:.2f}°" if x["rows"] > 1 else "none")
    row("Highest repeat of the elevation response in the searched sector", lambda x: ("none" if x["highest_repeat_db"] <= -99 else f"{x['highest_repeat_db']:.1f} dB") if x["rows"] > 1 else "n/a")
    row("Raw data rate while receiving (Mbit/s)", lambda x: f"{x['data_rate_mbit_s']:.0f}")
    row("Cables: one per hydrophone + the sync reference to each board", lambda x: f"{x['channels']} + {x['boards']}")
    L.append(f"\nReach is where noise alone would end detection on the link budget of Sec. 5 with the source level in the file ({auv['source_level_db']:.0f} dB, "
             "**not chosen for the test AUV or the static array**); reverberation and the channel decide earlier or later.\n")
    P, PS = S["prototype_span"], S["baseline_sweep"]
    low = P["3 m, a 185 dB projector"]
    a1 = sc["testauv_1row"]
    L.append("**Stage 1, one row on the test AUV: what it is for.** It proves the chain on real boards and under way: capture, synchronisation of four boards, the\n"
             "stream, recording, conditioning, beamforming, detection, tracking with ownship's motion taken out, classification, the report to the vehicle. Bearing\n"
             "and range are the full system's; only the beam is wider. There is no room for a second row on the test AUV, so it shows no elevation.\n\n"
             "| Row of 32 over | Beam | Gain against sea noise | Reach, ship / small (km) |\n| --- | --- | --- | --- |")
    for key in ("2 m", "3 m", "4.8 m (the full row)"):
        L.append(f"| {key} | {P[key]['beamwidth_deg']:.1f}° | {P[key]['gain_isotropic_noise_db']:.1f} dB | {km(P[key]['reach_ship_m'])} / {km(P[key]['reach_small_m'])} |")
    L.append(f"\nCloser together than half a wavelength the hydrophones hear the same sea noise: 32 over 3 m gain {P['3 m']['gain_isotropic_noise_db']:.1f} dB, what 20 at half a wavelength would.\n"
             "The other twelve still count against the boards' own noise and against a dead channel, and they are the full system's count, boards and wiring.\n"
             f"A higher band would suit the short row ({auv['pitch_m']:.2f} m is half a wavelength at {S['prototype_half_wavelength_frequency_hz'] / 1e3:.1f} kHz) but would be another projector and another band than\n"
             f"the full face's, so the prototype keeps {f_lo / 1e3:.0f}–{f_hi / 1e3:.0f} kHz. Its pulse is {auv['pulse_s']:g} s: the array cannot listen while it transmits, a {face3['pulse_s']:g} s pulse hides everything\n"
             f"inside {face3['blind_range_m']:.0f} m and {auv['pulse_s']:g} s everything inside {auv['blind_range_m']:.0f} m, at {10 * np.log10(face3['pulse_s'] / auv['pulse_s']):.0f} dB less processing gain. "
             f"With a 185 dB projector instead, the 3 m row reaches {km(low['reach_ship_m'])} / {km(low['reach_small_m'])} km.\n"
             f"Simulated trial (raw 102.4 kHz, four boards on their own clocks, the vehicle rolling 2°/s, 30 m of water): the boards synchronise to {a1['sync_residual_us']:.2f} µs and the\n"
             f"target at 500 m is found within {max(abs(v) for v in a1['range_error_m']):.1f} m and {max(abs(v) for v in a1['bearing_error_deg']):.2f}°.\n")
    sp = static_cfg["platform"]
    s2, s3, s3far, sfree, spass, slis = (sc[k] for k in ("static_2row", "static_3row", "static_3row_300m_shallow", "static_free", "static_passing", "static_passive"))
    L.append("**Stages 2 and 3, the static drop-in array: where stacked rows first go in the water.** A frame carrying two or three rows, lowered from the surface (or\n"
             "stood on the bottom) and left in place. The owner: \"have static array the longer row\": its rows are the face's own row module (32 over\n"
             f"{static3['row_length_m']:.1f} m), the outer rows {static3['baseline_m']:.2f} m apart **as on the face**, so the frame is the face's array without the vehicle: the same beam, reach, elevation\n"
             f"behaviour, boards and harness. The frame's size follows ({sp['span_m']:g} m × {sp['height_m']:g} m with the margins in the spec, parameters); its hanging depth ({sp['depth_m']:g} m here), tilt sensing, projector position\n"
             "and cabling are open and flagged `unconfirmed`. What is different from a vehicle, and where it is set:\n\n"
             "| | Static array | Where |\n| --- | --- | --- |\n"
             "| Motion | none: no Doppler of its own, reverberation does not spread in Doppler, and everything that moves in the tracker's frame is a target | scenario `platform.velocity: [0, 0, 0]` |\n"
             "| Pose | a fixed position, depth and heading typed in, or datagrams from a tilt and compass sensor on the frame | `nav.static` or `nav.port` in the deployment file |\n"
             "| Tilt | a hanging frame's tilt is elevation error one for one, exactly as a vehicle's roll (table below). Without a sensor it is assumed plumb, with a stated uncertainty | `elevation.roll_sigma_deg` (1° shipped: a placeholder), `attitude_time: transmit` |\n"
             "| Projector | on the frame or hung apart: its position relative to the array is in the sonar file and in every record, and the elevation search follows the true two-way path | `transmit.lever_arm` |\n"
             "| Sound speed, water depth | measured once at deployment and typed in | `chain.sound_speed`, `elevation.water_depth` |\n"
             "| Cable, power, where the compute sits | not decided: at the surface end of the cable, or in a housing on the frame | open; *Parts list* |\n")
    L.append("The frame's height is chosen from this curve (two rows; three rows have the accuracy of their outer pair):\n\n"
             "| Rows apart | Elevation at 13 / 20 / 30 dB | No repeat within (at the top of the band) | Highest repeat inside ±30° | Depth at 300 m (30 dB) | Depth at 1 km (20 dB) | Per µs of clock error between rows |\n| --- | --- | --- | --- | --- | --- | --- |")
    for sep, v in PS.items():
        L.append(f"| {sep} m | " + " / ".join(f"{x:.2f}°" for x in v["sigma_deg"].values()) + f" | ±{v['repeat_free_half_sector_deg_at_6khz']:.0f}° | "
                 + ("none" if v["highest_repeat_db_in_sector"] <= -99 else f"{v['highest_repeat_db_in_sector']:.1f} dB")
                 + f" | {v['depth_sigma_m_at_300m_30db']:.1f} m | {v['depth_sigma_m_at_1km_20db']:.0f} m | {v['clock_error_deg_per_us']:.2f}° |")
    L.append("\nBelow about 0.15 m a second row gives the sign of the elevation and little more. Up to about a wavelength there is no repeat inside the searched\n"
             "sector and two rows give one answer; wider, accuracy keeps improving in proportion and the repeats come in, which is what the third row is for.\n")
    L.append("Simulated trials of the static array (C++ chain; the Python chain is held to it by the tests):\n\n| Scenario | Truth | Reported |\n| --- | --- | --- |\n"
             f"| `popoto_static_trial_2row`: two rows, raw 102.4 kHz, {s2['boards']} boards on their own clocks, tilted 2°, {s2['water_depth_m']:.0f} m of water, target {s2['range_m'] / 1e3:.1f} km off | depth {s2['true_depth_m']:.0f} m | boards synchronised to {s2['sync_residual_us']:.2f} µs; depth {lst(s2['depth_m'])} m, one answer with the water-depth bound |\n"
             f"| `popoto_static_trial_3row`: three rows, {s3['boards']} boards, tilted 2°, {s3['water_depth_m']:.0f} m of water, target {s3['range_m']:.0f} m off | depth {s3['true_depth_m']:.0f} m | {s3['sync_residual_us']:.2f} µs; depth {lst(s3['depth_m'], 1)} m |\n"
             f"| the same with the target at {s3far['range_m']:.0f} m | depth {s3far['true_depth_m']:.0f} m | {lst(s3far['depth_m'], 1)} m, path `{s3far['path'][0]}`: in {s3far['water_depth_m']:.0f} m of water the bounces arrive inside the pulse's range sidelobes and pull the estimate |\n"
             f"| `popoto_static_free`: three rows, free field, targets about 400 m off | depths {lst(sfree['true_depths_m'])} m | track depth errors {lst(sfree['depth_errors_m'], 1)} m |\n"
             f"| `popoto_static_passing`: a target crossing at {spass['true_speed_m_s']:.0f} m/s, {spass['true_depth_m']:.0f} m deep, {spass['pings']} pings | one mover | tracks: {', '.join(spass['kinds'])}; mover at {lst(spass['mover_speed_m_s'], 1)} m/s, depth {lst(spass['mover_depth_m'], 1)} m |\n"
             f"| listening: two noise sources (`sim/passive.py`), tilted {slis['tilt_deg']:.0f}° | elevations {lst(slis['true_elevation_deg'], 1)}° | both in {slis['frames_with_both']} of {slis['frames']} frames, elevation within {slis['worst_elevation_error_deg']:.2f}° |\n")
    L.append("What the static array cannot show: ownship's motion and the vehicle's attitude sensor. With the long row it does show the full row's beam and reach.\n")
    BC = S["baseline_comparison"]
    L.append(f"**Stages 4 and 5, the full face.** Outer rows at {-up:+.2f} and {up:+.2f} m, {d:.2f} m apart ({S['margin_m']:.2f} m free at the top and bottom of the {S['face_height_m']:g} m), which is\n"
             f"{d * f_lo / 1500:.1f} to {d * f_hi / 1500:.1f} wavelengths. Accuracy is bought with separation and SNR, nothing else: for one arrival in noise\n"
             "σ(sin e) = 1 / (k √(2 ρ Σ(h − h̄)²)), ρ the SNR at one hydrophone after pulse compression, h the hydrophones' heights, k the band's RMS wavenumber.\n"
             f"The estimator reaches it (Monte Carlo over bound {acc[two]['monte_carlo_over_bound_at_20_db']:.2f}; on simulated pings within 30%).\n")
    L.append("| Rows | 13 dB | 20 dB | 30 dB |\n| --- | --- | --- | --- |")
    for name, v in acc.items():
        L.append(f"| {name} | " + " | ".join(f"{v['sigma_deg'][k]:.2f}°" for k in ("13.0", "20.0", "30.0")) + " |")
    L.append("\nWhat the owner's margin costs (the earlier baseline against the working one):\n\n"
             "| Outer rows apart | Elevation at 13 / 20 / 30 dB | Repeats every | Two rows give one answer beyond (100 m of water, one bounce) | Best middle row | Per µs between rows | Per mm a row stands out |\n| --- | --- | --- | --- | --- | --- | --- |")
    for sep, v in BC.items():
        L.append(f"| {sep} m | " + " / ".join(f"{x:.2f}°" for x in v["sigma_deg"].values()) + f" | {v['repeat_spacing_deg_at_5khz']:.1f}° | {km(v['two_rows_unambiguous_beyond_m_100m_water_one_bounce'])} km | "
                 f"{v['best_middle_m']:+.2f} m ({v['best_middle_level_db']:.1f} dB) | {v['clock_error_deg_per_us']:.3f}° | {v['row_out_error_deg_per_mm']:.3f}° |")
    LL = TS["length_limit"]
    L.append(f"\nAccuracy and every mounting and clock tolerance scale with the baseline; the repeats move apart, which helps two rows a little. The length limit\n"
             f"({LL['face_length_m']:g} → {LL['array_length_m']:g} m) does not touch the 32-hydrophone row ({face3['row_length_m']:.1f} m); it lowers what a filled row could hold from {LL['half_wavelength_count'][1]} to\n"
             f"{LL['half_wavelength_count'][0]} hydrophones at half a wavelength ({LL['gain_db'][1]:.1f} → {LL['gain_db'][0]:.1f} dB, beam {LL['beam_deg'][1]:.1f}° → {LL['beam_deg'][0]:.1f}°, reach {LL['reach_ship_km'][1]:.1f} → {LL['reach_ship_km'][0]:.1f} km on a ship).\n")
    pj = S["projector"]
    L.append(f"**The projector on the face.** No position was given (\"Projector etc\" needs room). The sonar files put it in the forward end margin at mid-height,\n"
             f"{pj['lever_arm_m'][0]:g} m from the array's centre: a placeholder, {pj['nearest_hydrophone_m']:.2f} to {pj['farthest_hydrophone_m']:.2f} m from the hydrophones. Two things follow, both from the files' own numbers:\n\n"
             f"- **Its direct sound overloads the inputs.** {pj['direct_level_db_at_nearest']:.0f} dB re 1 µPa at the nearest hydrophone is {pj['direct_volts_at_adc']:.0f} V at the ADC with the file's sensitivity and gain,\n"
             f"  {pj['over_full_scale_db']:.0f} dB over full scale ({pj['adc_full_scale_v']:.2f} V), for the whole pulse (to {pj['direct_sound_ends_s']:.3f} s after the ping). The plan clamps the preamp inputs and the software uses\n"
             f"  nothing inside {pj['blind_range_m']:.0f} m. How quickly the inputs recover afterwards is **not known**, and the simulator models neither the direct sound nor the recovery.\n"
             f"- **Its offset costs no accuracy.** Every record carries the projector's lever arm, and the beamformer and the elevation search hold the true path projector →\n"
             f"  target → each hydrophone (tested with the projector off-centre). A row's far field begins at {pj['far_field_of_row_m']:.0f} m; the backprojection kernel is exact inside it too.\n")
    L.append(f"**Two rows repeat.** Rows {d:.2f} m apart cannot tell elevations λ/d apart: {T['repeat_spacing_deg'][0]:.1f}° at {f_lo / 1e3:.0f} kHz, {T['repeat_spacing_deg'][1]:.1f}° at {(f_lo + f_hi) / 2e3:.0f} kHz, {T['repeat_spacing_deg'][2]:.1f}° at {f_hi / 1e3:.0f} kHz.\n"
             f"The band smears the repeats, but with true time delays over {f_lo / 1e3:.0f}–{f_hi / 1e3:.0f} kHz the m-th repeat is still at sinc²(m B / 2 f) of the true peak whatever the\n"
             "separation: " + ", ".join(f"{v:.1f} dB" for v in T["repeat_level_db"]) + " for m = 1 to 4. How often the estimator lands on a repeat, in clean noise:\n")
    L.append("| Searching | 13 dB | 20 dB | 30 dB |\n| --- | --- | --- | --- |")
    for label, v in T["wrong_pick"].items():
        L.append(f"| two rows, {label} | " + " | ".join(f"{100 * x:.1f}%" for x in v.values()) + " |")
    L.append(f"\nSo in clean noise the band does resolve two rows from 20 dB up. But the margin is {margin2:.1f} dB, and a range sidelobe of another arrival or a few degrees of\n"
             "phase error between rows tips it. The chain therefore never trusts a repeat within 1 dB of the peak: with two rows this far apart every detection is\n"
             "flagged `elevation_ambiguous` and carries its candidates, unless something else decides:\n\n"
             f"- **Water depth and range.** Straight rays cannot arrive from steeper than atan(depth × (bounces + 1) / range). Where that is inside ±{T['unambiguous_half_sector_deg']:.1f}° (half a\n"
             "  repeat) there is one candidate; `elevation.sector_from_depth` searches only there. Two rows are unambiguous beyond (direct / one bounce / two):\n\n"
             "  | Water depth | " + " | ".join(f"{float(k):.0f} m" for k in T["sector_ranges_m"]) + " |\n  | --- |" + " --- |" * len(T["sector_ranges_m"]) + "\n"
             "  | Range | " + " | ".join(" / ".join(f"{x / 1e3:.1f}" for x in v) + " km" for v in T["sector_ranges_m"].values()) + " |\n\n"
             f"- **The track.** A track that has a depth takes the candidate that fits it (they are {T['candidate_depth_step_m_per_km']:.0f} m of depth apart per km of range); a track without one waits\n"
             "  for an unambiguous detection and is not started on a guess.\n"
             "- **A third row.**\n")
    srch = W3["search"]
    L.append(f"**Three rows: where the middle one goes.** With the outer rows fixed, the highest repeat inside ±30° over the band (worst of arrivals at 0°, 10°, 20°)\n"
             "against the middle row's height (the mirror image about the centre is the same):\n\n"
             "| Middle row at (m) | " + " | ".join(k for k in list(srch)[::2]) + " |\n| --- |" + " --- |" * len(list(srch)[::2]) + "\n"
             "| Highest repeat (dB) | " + " | ".join(f"{srch[k]:.1f}" for k in list(srch)[::2]) + " |\n\n"
             f"Lowest at {mid:+.2f} m ({up + mid:.2f} m above the bottom row): {W3['best_level_db']:.1f} dB, against {W3['equal_level_db']:.1f} dB with equal spacing and {T['repeat_level_db'][0]:.1f} dB with two rows.\n"
             "Wrong picks searching ±30°, arrival at 2° / at 15°:\n")
    snr_keys = list(next(iter(W3["wrong_pick"].values())))
    L.append("| Rows | " + " | ".join(f"{float(k):.0f} dB" for k in snr_keys) + " |\n| --- |" + " --- |" * len(snr_keys))
    for name, v in W3["wrong_pick"].items():
        L.append(f"| {name} | " + " | ".join(f"{100 * x[0]:.1f}% / {100 * x[1]:.1f}%" for x in v.values()) + " |")
    L.append(f"\nThree rows are no more accurate than two (the outer rows set that); the third buys one elevation angle instead of a list of possible ones, and {g3 - g2:.1f} dB of detection.\n")
    t = B["timing_scenario_size"]
    f2 = sc["face_2row"]
    L.append("**Two rows or three: side by side** (the face; the static array's rows are at the same heights).\n\n| | Two rows | Three rows (recommended) |\n| --- | --- | --- |\n"
             f"| Hydrophones, boards, cables | {face2['channels']}, {face2['boards']}, {face2['channels']} + {face2['boards']} | {face3['channels']}, {face3['boards']}, {face3['channels']} + {face3['boards']} |\n"
             f"| Elevation at 13 / 20 / 30 dB | " + " / ".join(f"{acc[two]['sigma_deg'][k]:.2f}°" for k in ("13.0", "20.0", "30.0")) + " | " + " / ".join(f"{acc[three]['sigma_deg'][k]:.2f}°" for k in ("13.0", "20.0", "30.0")) + " |\n"
             f"| Highest repeat | {T['repeat_level_db'][0]:.1f} dB: never trusted alone | {W3['best_level_db']:.1f} dB |\n"
             f"| Wrong pick at 13 dB, clean noise, ±30° | {100 * W3['wrong_pick'][two]['13.0'][0]:.0f}% | {100 * W3['wrong_pick'][three]['13.0'][0]:.1f}% |\n"
             f"| One answer without help | where water depth and range bound the arrivals to ±{T['unambiguous_half_sector_deg']:.1f}° (100 m of water: beyond {T['sector_ranges_m']['100.0'][1] / 1e3:.1f} km with one bounce) | everywhere above about 10 dB a row |\n"
             "| Deep water, short range | a list of possible angles, not one (bearing, range and the track are unaffected), until a track has a depth | works |\n"
             f"| Detection against one row | +{g2:.1f} dB | +{g3:.1f} dB |\n"
             "| A dead board | the row it serves loses 8 of 32; the estimate carries on | the same |\n"
             "| A dead row | no elevation | two rows left: the two-row behaviour |\n"
             f"| Processing per ping, scenario size, C++ on {B['cpu_threads']} shared CPU threads (one row: {t['1']['seconds_per_ping']:.2f} s) | {t['2']['seconds_per_ping']:.2f} s | {t['3']['seconds_per_ping']:.2f} s |\n"
             f"| Raw data rate | {face2['data_rate_mbit_s']:.0f} Mbit/s | {face3['data_rate_mbit_s']:.0f} Mbit/s |\n"
             f"| Simulated check | `popoto_face_2row_shallow`: target at {f2['true_depth_m']:.0f} m, 3 km: read {lst(f2['depth_m'])} m with the water-depth bound; without it {f2['candidates_without_the_bound'][0]} candidates | free field, channel and rehearsal below |\n")
    L.append("**Recommended: three rows** as the default of the option, with two rows a sound first step (the middle row can be added later: its boards and\n"
             "cables are separate) and the right choice only where the water is shallow against the ranges of interest. Both are shipped, for the frame and for the face.\n")
    ds, snr = S["depth_sigma_m"], S["link_snr_db"]
    RF = S["refraction_m"]

    def cell(v):
        return "not detected" if v is None else f"{v:.0f} m" if v >= 10 else f"{v:.1f} m"

    L.append("**What the angle is in depth, and what limits it.** Depth error is range × elevation error. From noise alone, on the link budget of Sec. 5 for one row:\n")
    L.append("| Target, rows | 1 km | 6 km | 12 km |\n| --- | --- | --- | --- |")
    for label in snr:
        L.append(f"| *{label}: SNR* | {snr[label]['1000.0']:.0f} dB | {snr[label]['6000.0']:.0f} dB | {snr[label]['12000.0']:.0f} dB |")
        for g in (three, "2 rows, 0.38 m apart"):
            r_ = ds[f"{label}; {g}"]
            L.append(f"| {label}; {g} | {cell(r_['1000.0'])} | {cell(r_['6000.0'])} | {cell(r_['12000.0'])} |")
    L.append(f"\n(SNR above 40 dB counted as 40 dB.) Rows {d:.1f} m apart are {d / 0.38:.0f} times better than the 0.38 m of a small hull. But noise is no longer what limits depth.\n"
             "**Roll (or a frame's tilt) is elevation error one for one**: drop the attitude sensor's datasheet figure into this table.\n\n"
             "| Roll known to | Depth error at 1 km | 6 km | 12 km |\n| --- | --- | --- | --- |\n"
             + "".join(f"| {a_}° | {v[0]:.1f} m | {v[1]:.1f} m | {v[2]:.0f} m |\n" for a_, v in se["roll_depth_error_m"].items())
             + "\nAnd the path's bending:\n\n| Mean sound-speed gradient along the path unknown by | 1 km | 6 km | 12 km |\n| --- | --- | --- | --- |\n"
             + "".join(f"| {k} /s" + (" (isothermal water: pressure alone)" if k == "0.017" else "") + " | "
                       + " | ".join(f"{RF[k][r]:.0f} m" for r in ("1000.0", "6000.0", "12000.0")) + " |\n" for k in RF))
    X = S["refraction_crossover_m"]
    L.append("\n**No long-range sound-speed profile will exist, and the water will not always be isothermal.** The vehicle measures the sound speed where it is; that fixes\n"
             "the angle at the array exactly (and is used for beamforming). It does not fix the path: a mean vertical gradient g curves the ray with radius c/g,\n"
             f"range² × g / 2c from the straight line. Beyond {X['0.05']['20.0']:.0f} m (gradient known to 0.05 /s, 20 dB) or {X['0.017']['20.0']:.0f} m (known to 0.017 /s) the unknown gradient, not the\n"
             "measurement, limits depth. What the chain does today: it assumes a gradient (`elevation.gradient_per_s`; 0.017 /s, isothermal water, in the vehicle's file:\n"
             "an assumption) and follows the ray through it; it adds range² × `gradient_sigma_per_s` / 2c to every reported depth sigma (0.05 /s by default, from the span\n"
             "between isothermal water and a thermocline averaged over the column: unconfirmed); a track's sigma keeps that term however many pings it averages; a\n"
             "measured profile goes in `elevation.ssp` and replaces the assumption. **In open water depth is good to metres inside a kilometre and to tens of metres\n"
             "at a few kilometres (a shallow channel adds its own error: below); beyond that the report still gives, reliably, the angle at the array, whether the echo comes from above or below, which arrival of a group came\n"
             "first, and how a track's elevation changes.** Not built yet: a profile the vehicle accumulates from its own sound-speed and depth readings as it changes\n"
             "depth, and named profile families (mixed layer over a thermocline); see *Open*.\n")
    cb = S["cone_bias_deg"]
    L.append("**Bearing.** A line reports sin(cone) = cos(elevation) × sin(bearing). With the elevation known the chain reports the horizontal bearing (`bearing_deg`;\n"
             "`cone_bearing_deg` keeps the old value). Error removed: " + "; ".join(f"{cb[e]['45.0']:.2f}° at {float(e):.0f}° elevation" for e in cb) + " (45° off broadside).\n")
    mp = S["multipath"]
    sh, f, re = sc["shallow"], sc["free_field"], sc["rehearsal"]
    L.append(f"**In a channel.** One target sends back several echoes. Range separates them (resolution {mp['range_resolution_m']:.3f} m) and elevation is measured on each; the direct one\n"
             "is first, and the chain takes the first arrival of each detection. Followed straight back, a surface echo points at the target's mirror image above the\n"
             "surface and a bottom echo at its image below the bottom: folded back both give the target's depth, and where the image lay says which path it was (`path`).\n"
             f"The three rows' own beam in elevation is {mp['elevation_beamwidth_deg']:.1f}° wide: direct and surface arrivals differ by more than that within "
             + ", ".join(f"{v:.0f} m for an array or target at {k} m" for k, v in mp["direct_surface_resolved_in_elevation_within_m"].items()) + ",\n"
             "so only at short range do two arrivals in one range cell separate in elevation; beyond it only range separates them. A target near the\n"
             f"surface merges with its own surface echo at every useful range (beyond {mp['direct_and_surface_merge_beyond_m']['4'] / 1e3:.1f} km at 4 m depth) and reads as a source at the surface.\n"
             f"The water depth the bottom fold needs is the vehicle's approximate one (`water_depth`, `water_depth_sigma_m`): its uncertainty goes, doubled, into the depth\n"
             "of bottom-bounce arrivals and into the test that marks a track below the bottom as multipath.\n")
    L.append("Measured on simulated pings of the face (C++ chain; the Python chain agrees to 0.02° and 0.3 m):\n\n| Scenario | Truth | Reported |\n| --- | --- | --- |\n"
             f"| `popoto_elevation_free`: three rows, free field, rolled {f['roll_deg']:.0f}°, three targets near 1 km | depths 10, 60, 160 m | track depth errors {lst(f['depth_errors_m'], 1)} m |\n"
             f"| `popoto_elevation_shallow`: 100 m of water, bounces to second order, 1 km | depth {sh['true_depth_m']:.0f} m, direct path {sh['true_elevation_deg']:+.2f}° | {lst(sh['depth_m'])} m, {lst(sh['elevation_deg'], 1)}°, path {sh['path'][0]} |\n"
             f"| `popoto_elevation_rehearsal`: raw 102.4 kHz, {re['boards']} boards on their own clocks, rolled {re['roll_deg']:.0f}° and rolling {re['roll_rate_deg_s']}°/s, local sound speed {re['sound_speed']:.0f} m/s | targets at {re['deep_true_m']:.0f} m and {re['shallow_true_m']:.0f} m, 1 km | boards synchronised to {re['worst_residual_us']:.2f} µs; {lst(re['deep_depth_m'])} m "
             f"(with the attitude of the ping instead of the arrival: {lst(re['deep_depth_with_transmit_attitude_m'])} m); the shallow one {lst(re['shallow_depth_m'], 1)} m, merged with its surface echo |\n")
    pull2 = max(abs(v - f2["true_depth_m"]) for v in f2["depth_m"])
    pull3 = max(abs(v - sh["true_depth_m"]) for v in sh["depth_m"])
    sc2 = load_scenario_config("popoto_face_2row_shallow")
    one_leg = float(sc2["platform"]["position"][2]) * f2["true_depth_m"] / f2["range_m"]      # range excess of a surface bounce on one leg
    L.append(f"**What a channel does to the depth, and the sigma does not say.** An echo that bounced once on one leg arrives just behind the direct echo ({one_leg:.1f} m in the\n"
             f"two-row check, against a range resolution of {mp['range_resolution_m']:.3f} m) and from another elevation; it is not cleanly separated, and it pulls the angle. In the two-row check at 3 km the target at {f2['true_depth_m']:.0f} m reads\n"
             f"{lst(f2['depth_m'])} m (up to {pull2:.0f} m off, {np.degrees(pull2 / f2['range_m']):.1f}°); in the three-row check at 1 km the one at {sh['true_depth_m']:.0f} m reads {lst(sh['depth_m'])} m ({pull3:.0f} m off); the static array at\n"
             f"{s3far['range_m']:.0f} m in {s3far['water_depth_m']:.0f} m of water reads {lst(s3far['depth_m'])} m for {s3far['true_depth_m']:.0f} m. In free field the same chain is within a metre. The size of the pull depends on the\n"
             "echoes' phases, so on the geometry to a fraction of a wavelength: with the projector 1.5 m below the array's centre instead of beside it, the two-row case read\n"
             "closer to the truth (an earlier run of this bench, not kept in its file). The reported `depth_sigma_m` holds noise, roll and the gradient prior, **not this**: in shallow water at long\n"
             "range treat depth as good to a few tenths of a degree of elevation, whatever the sigma says. Not modelled further here: a longer band or a window with\n"
             "lower range sidelobes would separate the arrivals better.\n")
    clock, need = se["board_clock_error_deg"], se["clock_us_for_deg"]
    L.append("**What the vehicle and the mounting must provide.**\n\n| Error source | Elevation error | Depth error at 1 / 6 / 12 km |\n| --- | --- | --- |")
    for label, key, unit in (("One row further out from the face than the other by", "out_offset_error_deg", " mm"),
                             ("Phase mismatch between rows of", "phase_mismatch_error_deg", "° of phase"),
                             ("Clock error between boards on different rows of", "board_clock_error_deg", " µs")):
        for a_, v in se[key].items():
            L.append(f"| {label} {a_}{unit} | {v:.3f}° | " + " / ".join(f"{r_ * np.radians(v):.1f}" for r_ in (1000.0, 6000.0, 12000.0)) + " m |")
    transmit_bias = abs(re["deep_true_m"] - float(np.mean(re["deep_depth_with_transmit_attitude_m"])))
    L.append(f"\n- **Roll is the first limit of the angle.** Everything that scales with the rows' separation shrank {d / 0.38:.1f} times against a small hull; roll did not.\n"
             "  `elevation.roll_sigma_deg` (0.2° assumed for the vehicle: **to be replaced by the SPRINT-Nav's figure**) goes into every sigma. **Attitude is taken at each echo's\n"
             f"  arrival** (`attitude_time: arrival`): an echo from 12 km arrives {2 * 12000 / 1500:.0f} s after the ping, and the chain uses the navigation samples around that time (they are\n"
             "  written into each record). Without samples it falls back to the ping's attitude and adds `attitude_rate_sigma_deg_s` × the time of flight. In the rehearsal\n"
             f"  above the difference is {transmit_bias:.0f} m at 1 km.\n"
             f"- **Synchronisation: still open.** Row by row (the owner's choice), boards of different rows must agree to {need['0.05']:.2f} µs for 0.05° and {need['0.1']:.2f} µs for 0.1°\n"
             f"  ({clock['1.0']:.3f}° per µs with rows {d:.2f} m apart); the tolerance one row needs for clean beams, {1e6 / (20 * f_hi):.1f} µs (a twentieth of a period at the top of the band, `sync.alignment_tolerance`), would be {clock['1.0'] * 1e6 / (20 * f_hi):.1f}°. The reference scheme reaches {re['worst_residual_us']:.2f} µs in simulation.\n"
             "  Whether real boards do is the open question. `array.interleave: 4` (four of each row per board) removes the term for two rows and stays available.\n"
             f"- **Mounting and calibration.** 1 mm of one row standing out from the face is {se['out_offset_error_deg']['1.0']:.2f}°. The existing calibration, run with sources at several depths\n"
             "  (`planar=False`), recovers a row's offset in simulation (`tests/python/test_elevation.py` holds the figures).\n"
             "- **Sound speed and water depth** from the vehicle: `sound_speed` and `water_depth` in the navigation datagrams, or as settings.\n")
    mg, ob = TS["marginal_32_to_64_per_row"], TS["one_by_64_vs_two_by_32"]
    r64, fill = TS["per_row"][keys["row_64"]], TS["per_row"][keys["fills"]]
    L.append("### Trade space: how many hydrophones, and where\n")
    L.append(f"The outline again (the face, rows within {S['array_length_m']:g} m, outer rows {d:.2f} m apart, {f_lo / 1e3:.0f}–{f_hi / 1e3:.0f} kHz, {face3['source_level_db']:.0f} dB, {face3['pulse_s']:g} s pulse; noise-limited reach on the budget of Sec. 5):\n")
    L.extend(trade_table())
    L.append(f"\n**Does it need to be 64?** For bearing, range and tracks, no: one row of 32 is the design of Sec. 5. Sixty-four buy one of two things. As one row of 64\n"
             f"inside the {S['array_length_m']:g} m: {mg['gain_db']:.1f} dB, a beam of {mg['beam_deg'][1]:.1f}° instead of {mg['beam_deg'][0]:.1f}°, {r64['reverberation_margin_vs_32_at_half_wavelength_db']:.1f} dB less reverberation in the beam, noise-limited reach {mg['reach_ship_km'][0]:.1f} → {mg['reach_ship_km'][1]:.1f} km (ship) and\n"
             f"{mg['reach_small_km'][0]:.1f} → {mg['reach_small_km'][1]:.1f} km (small target), no elevation; and since {fill['hydrophones_per_row']} hydrophones at half a wavelength already fill that length, 64 gain no more than {fill['hydrophones_per_row']} would.\n"
             f"As two rows of 32: {ob['map_gain_db'][1]:.1f} dB in the map against {ob['map_gain_db'][0]:.1f} dB, the old beam, and elevation as a list of possible angles (two rows cannot tell which of several an echo came from; the list becomes one angle where water depth and range rule the others out). Bearing and range against elevation is the trade for a\n"
             "fixed channel count; 96 as three rows make the elevation unambiguous.\n")
    L.append("| A row of | Length | Beam | Gain (sea noise) | Reach ship / small (km) | Reverberation in the beam against 32 | Boards a row |\n| --- | --- | --- | --- | --- | --- | --- |")
    for label, v in TS["per_row"].items():
        note = f" (repeats in bearing at {v['wideband_lobe_steered_30_deg_db']:.0f} dB when steered 30°)" if v["wideband_lobe_steered_30_deg_db"] > -12.5 else ""
        L.append(f"| {label}{note} | {v['row_length_m']:.2f} m | {v['beamwidth_deg']:.1f}° | {v['gain_isotropic_noise_db']:.1f} dB | {km(v['reach_ship_m'])} / {km(v['reach_small_m'])} | "
                 f"{-v['reverberation_margin_vs_32_at_half_wavelength_db']:+.1f} dB | {v['boards']} |")
    g4 = TS["rows"]["4"]["rows_detection_gain_db"] - TS["rows"]["3"]["rows_detection_gain_db"]
    L.append("\n- **Along the face.** Each doubling of a filled row is 3 dB against noise, half the beam and 3 dB less reverberation; in noise-limited range a few per cent.\n"
             f"  The knee is the usable length: {LL['half_wavelength_count'][0]} hydrophones at half a wavelength fill {LL['array_length_m']:g} m, and more in the same length buy nothing against sea noise. Sparser\n"
             "  than half a wavelength gives the narrow beam for fewer hydrophones but full repeats in bearing when steered.\n"
             f"- **Up the face.** The second row buys elevation and {g2:.1f} dB of detection; the third makes elevation unambiguous (+{g3 - g2:.1f} dB); a fourth adds {g4:.1f} dB.\n"
             "  Accuracy is set by the outer rows' separation, not the count.\n"
             "- **Noise or reverberation.** More source level buys noise-limited range (6 dB is worth four times the hydrophones in a row) and nothing against\n"
             "  reverberation; a longer row buys both. Where reverberation limits (short and medium range in the scenarios), length is the only one that helps.\n"
             "- **Own depth, and the surface.** None of the reach figures here contains the surface-reflected path. Against a target near the surface in water that does not duct, own depth is worth more than any row: "
             + _surface_line() +
             "- **Band.** Another band moves every row: spacing and row length scale with the wavelength, the reach with absorption and noise\n"
             "  (`tools/popoto_range_study.py`), and the rows' repeat spacing with it.\n"
             f"- **Cost** goes with the channel count: boards = channels / 8, one cable per hydrophone and one reference per board, {lr['raw_data_rate_mbit_s'] / lr['elements']:.2f} Mbit/s and "
             f"{lr['samples_per_ping_per_channel'] * 4 / 1e6:.1f} MB per ping per\n  channel (to {lr['max_range_m'] / 1e3:.1f} km, as recorded now), compute in proportion (measured at 32, 64 and 96 channels).\n"
             "- **Passive.** Listening gains with the row's length like the active map, and the rows give each bearing an elevation; the repeats and the roll apply unchanged.\n")
    st = B["stream"]
    full = B.get("timing_full_length") or {}
    L.append("**Data path and compute.** Checked at the boards' rate (102.4 kHz, 24 bit) with 32, 64 and 96 channels: board blocks over UDP "
             f"({st['1']['udp_block_samples']}, {st['2']['udp_block_samples']} and {st['3']['udp_block_samples']} samples a datagram), the C++ and Python\nassemblers, Sensor Bridge frames ({st['2']['hsb_frame_bytes']:,} and {st['3']['hsb_frame_bytes']:,} bytes for 1600 samples at 64 and 96 channels), "
             "the container file and the chain carry them\nunchanged. The record format already stores every hydrophone's position and the attitude samples: no format change. "
             f"On pings of the `popoto_12km`\nscenario's size (baseband) the C++ chain takes {t['1']['seconds_per_ping']:.2f}, {t['2']['seconds_per_ping']:.2f} and {t['3']['seconds_per_ping']:.2f} s for one, two and three rows on {B['cpu_threads']} shared CPU threads, against "
             f"{lr['min_ping_interval_s']:.1f} s between pings:\nreal time with room.\n"
             + (f"(Timings are from an otherwise idle machine with the earlier row heights; the work does not depend on them.) **At the boards' full rate it is not real time on such a CPU**: one full-length ping (102.4 kHz real, 0.5 to 13 km) took {full['1']['seconds_per_ping']:.0f}, {full['2']['seconds_per_ping']:.0f} and "
                f"{full['3']['seconds_per_ping']:.0f} s (measured with\nearlier row heights; the map dominates and does not depend on them). The band is {(f_hi - f_lo) / 1e3:.0f} kHz wide, so the samples carry {102.4e3 / (2 * (f_hi - f_lo)):.0f} times "
                "more than needed. The owner: \"Prepare both options\":\nconverting to baseband and decimating before the chain, and a GPU. "
                + _compute_options_text() + "\n" if full else ""))
    L.append("**What carries over from stage to stage, and what changes.** Unchanged: the boards and their count per row, the synchronisation scheme, the stream and record\n"
             f"formats, the software and its settings, the report. From the test AUV to the static array: the row changes from the short module ({auv['pitch_m']:.2f} m spacing) to the long one ({face3['pitch_m']:.2f} m),\n"
             "two or three of them, no motion, a typed-in pose. From the static array to the face: the row module, its harness, the rows' heights and the boards carry over unchanged; the platform changes: motion,\n"
             "the attitude source (the vehicle's navigation, at each echo's arrival), the projector's place and pulse, cabling, and where the compute sits.\n")
    L.append("**What stacked rows do not improve.** Bearing and range resolution; left/right (the face decides that); detection range, much; depth at long range without\n"
             "a profile.\n")
    L.append("**Open.**\n\n"
             "- The static array: the depth it hangs at, whether it has a tilt sensor, where its projector hangs, its cabling, the margins of its frame, the trial site's depth and ranges.\n"
             "- The SPRINT-Nav: attitude accuracy, output rate, time-stamp accuracy and message format (from its datasheet and interface document). The adapter from\n"
             "  its messages to the navigation datagram is not written; DVL altitude plus depth as the water-depth source is not wired in.\n"
             "- The projector: which, its source level, its place on the face, and how fast the inputs recover from its direct sound (*The projector's direct sound*).\n"
             f"- Board synchronisation between rows: {need['0.05']:.2f} µs for 0.05°. Not answered; whether Popoto boards can be batched at all is a question for Popoto.\n"
             "- The sound-speed profile: a profile built from the vehicle's own readings, and profile families, are not built; today's chain has the gradient prior,\n"
             "  the local sound speed and a supplied profile.\n"
             "- Compute at the boards' full rate: both options (decimate first; GPU) are to be prepared; neither has been timed at the full rate on the hardware that would be used.\n"
             "- The parts list's open questions (`design/popoto_parts.md`).\n")
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
    sys.path.insert(0, str(ROOT / "tools"))
    import reference_plan
    L.extend(reference_plan.staged_plan())
    L.extend(reference_plan.row_modules())
    L.extend(stacked_rows(D))
    L.extend(reference_plan.projector_options())
    L.extend(reference_plan.depth_and_targets())
    L.extend(reference_plan.twelve_km())
    L.extend(reference_plan.parts_section())
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
    if study.get("surface"):
        sf = study["surface"]
        L.append(f"- **The surface-reflected path is not in the curves above.** For a target at {sf['target_depth_m']:g} m and {sf['range_m'] / 1e3:g} km it changes the source level needed by "
                 + ", ".join(f"{-r['peak_db']:+.0f} dB with the array at {r['own_depth_m']:g} m" for r in sf["by_own_depth"])
                 + f" (flat surface, the whole band; `gpusonar.lloyd`). In ducting water the bottom's bounces replace this term. Source levels here and in every build spec are rms levels ({sf['source_level_convention'].split(':')[0]}).\n")
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
