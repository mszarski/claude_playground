#!/usr/bin/env python3
"""What the projector and its amplifier must do, computed from a build spec.

For each build under configs/builds/ this prints (markdown) and writes
docs/design/projector_requirements.json:

  link        what a source level buys: SNR at the map at 6 and 12 km and the reach, for
              190..203 dB and 1, 2, 3 rows, from gpusonar.buildspec.performance (the link
              budget of docs/DESIGN.md Sec. 5; nothing is retyped here)
  power       acoustic power for the level, electrical power against efficiency, energy
              per ping and per hour at the build's ping interval, heat left in the transducer
  cavitation  the source level a radiator of a given wet area can hold without cavitation
              against depth, and the least depth for the build's level
  direct      the direct sound at the hydrophones: what reduction towards the array is
              needed (peak and rms conventions), what distance does it, and what a short
              broadside line of small projectors leaves (exact near-field sum, over the band)
  pulse       the same energy with a longer pulse at a lower level: blind range, overload,
              Doppler tolerance of a linear sweep
  drive       volts and volt-amperes for a transmitting voltage response and an impedance
  band        what a single resonance costs across the band (flatness of the response)
  depth       the surface-image factor of docs/DESIGN.md ("a target near the surface")
              against own depth, beside the cavitation limit there

Conventions. Source level is RMS, dB re 1 uPa at 1 m (the owner's decision on finding P-5 of
docs/HARDCODED_AUDIT.md). The link budget is an energy equation and takes it as it is. The
overload figures of buildspec.describe treat the same number as a PEAK pressure, so with an
rms specification every overload figure is 3 dB higher: both are printed. Cavitation is set
by the peak pressure: rms + 3 dB for a sweep or a tone.

Cavitation relation (docs/design/projector_selection.md Sec. 1.4 has the sources and the
uncertainty): cavitation begins where the peak acoustic pressure at the radiating surface
reaches the static pressure there (the tensile strength of sea water at a few kHz and for
long pulses is taken as zero):

    p_c(h)  = p_atm + rho g h - p_vapour                      [Pa, peak]
    I_c(h)  = gamma p_c^2 / (2 rho c)                         [W/m^2; 0.33 W/cm^2 at the surface]
    P_c     = I_c A                                           [W radiated, A the wet radiating area]
    SL_c    = 10 log10(rho c P_c / (4 pi)) + 120 + DI         [dB re 1 uPa at 1 m, rms]

gamma <= 1 is the ratio of the mean to the largest intensity over the radiating face (1 for a
uniformly pulsating surface). For a sphere of radius a this is SL_c = 20 log10(p_c a / sqrt 2).

Usage: tools/projector_requirements.py [build ...] [--levels 190 194 197 200 203] [--ranges 6000 12000]
       [--depths 5 10 30 100 200] [--areas 0.01 0.03 0.1 0.3] [--no-json]
CPU only; pure Python apart from gpusonar.buildspec.
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import buildspec  # noqa: E402

RHO, G, P_ATM, P_VAPOUR = 1025.0, 9.81, 101325.0, 1700.0      # sea water kg/m3; m/s2; Pa; Pa (about 15 C)
DETECTION_DB = 13.0                                           # at the map: the threshold buildspec.performance uses


# --- power and energy ----------------------------------------------------------------------------------

def acoustic_power_w(source_level_db: float, di_db: float = 0.0, c: float = 1500.0) -> float:
    """Radiated power for an rms source level: SL = 10 log10(rho c P / 4 pi) + 120 + DI (170.8 + 10 log P + DI)."""
    return 4.0 * math.pi * 10.0 ** ((source_level_db - di_db - 120.0) / 10.0) / (RHO * c)


def source_level_db(power_w: float, di_db: float = 0.0, c: float = 1500.0) -> float:
    return 10.0 * math.log10(RHO * c * power_w / (4.0 * math.pi)) + 120.0 + di_db


def energy(source_level: float, pulse_s: float, interval_s: float, efficiency: float, amplifier_efficiency: float = 0.85,
           di_db: float = 0.0) -> dict:
    """Electrical power into the transducer and from the supply during the pulse, energy per ping and per hour."""
    pa = acoustic_power_w(source_level, di_db)
    pe = pa / efficiency
    ps = pe / amplifier_efficiency
    return {"acoustic_w": pa, "into_transducer_w": pe, "from_supply_w": ps, "joule_per_ping": ps * pulse_s,
            "wh_per_hour": ps * pulse_s / interval_s, "average_w": ps * pulse_s / interval_s,
            "heat_in_transducer_average_w": (pe - pa) * pulse_s / interval_s, "duty": pulse_s / interval_s}


# --- cavitation ----------------------------------------------------------------------------------------

def cavitation_pressure_pa(depth_m: float, tensile_pa: float = 0.0) -> float:
    """Peak acoustic pressure at which cavitation begins at this depth."""
    return P_ATM + RHO * G * depth_m - P_VAPOUR + tensile_pa


def cavitation_source_level_db(depth_m: float, area_m2: float, di_db: float = 0.0, gamma: float = 1.0, c: float = 1500.0) -> float:
    """Largest rms source level (dB re 1 uPa at 1 m) a radiator of wet area `area_m2` holds without cavitation."""
    intensity = gamma * cavitation_pressure_pa(depth_m) ** 2 / (2.0 * RHO * c)
    return source_level_db(intensity * area_m2, di_db, c)


def cavitation_depth_m(source_level: float, area_m2: float, di_db: float = 0.0, gamma: float = 1.0, c: float = 1500.0) -> float:
    """Least depth at which `source_level` (rms) is free of cavitation; 0 when it is free at the surface."""
    intensity = acoustic_power_w(source_level, di_db, c) / area_m2 / gamma
    p = math.sqrt(2.0 * RHO * c * intensity)
    return max((p - P_ATM + P_VAPOUR) / (RHO * G), 0.0)


# --- electrical drive ----------------------------------------------------------------------------------

def drive(source_level: float, tvr_db: float, impedance_ohm: float | None = None, phase_deg: float = 0.0) -> dict:
    """Volts (rms) for a source level from a transmitting voltage response (dB re 1 uPa/V at 1 m), and with the
    impedance magnitude and phase at that frequency the current, volt-amperes and real power."""
    v = 10.0 ** ((source_level - tvr_db) / 20.0)
    out = {"volts_rms": v, "volts_peak": v * math.sqrt(2.0)}
    if impedance_ohm:
        out.update({"amps_rms": v / impedance_ohm, "volt_amperes": v * v / impedance_ohm,
                    "real_power_w": v * v / impedance_ohm * math.cos(math.radians(phase_deg))})
    return out


def resonance_band_loss(f_lo: float, f_hi: float, edge_db: float) -> dict:
    """A single resonance centred (geometrically) in the band whose response is `edge_db` down at the band edges:
    its mechanical Q, and the energy lost across a linear sweep driven at constant voltage."""
    f0 = math.sqrt(f_lo * f_hi)
    x = f_hi / f0 - f0 / f_hi
    q = math.sqrt(10.0 ** (edge_db / 10.0) - 1.0) / x
    f = np.linspace(f_lo, f_hi, 2001)
    h = 1.0 / (1.0 + q * q * (f / f0 - f0 / f) ** 2)
    return {"q": q, "mean_loss_db": float(-10.0 * np.log10(h.mean())), "headroom_to_equalise_db": edge_db,
            "volts_factor_to_equalise": 10.0 ** (edge_db / 20.0)}


# --- the build -----------------------------------------------------------------------------------------

def link_table(d: dict, levels, rows=(1, 2, 3), ranges=(6000.0, 12000.0)) -> list[dict]:
    """SNR at the map and reach for each source level and row count: buildspec.performance on the build's
    own derived numbers with the level and the number of rows replaced."""
    out = []
    for k in rows:
        for sl in levels:
            dd = dict(d)
            dd.update({"source_level_db": float(sl), "rows": int(k), "elevation": None})
            p = buildspec.performance(dd, ranges=ranges)
            g = p["rows_detection_gain_db"]
            row = {"rows": k, "source_level_db": float(sl), "reach_ship_m": p["reach_ship_m"], "reach_small_m": p["reach_small_m"]}
            for r in ranges:
                for key in ("ship", "small"):
                    row[f"snr_{key}_{r / 1e3:g}km_db"] = p["row_snr_db"][key][str(r)] + g
            out.append(row)
    return out


def line_direct(d: dict, n: int, spacing_m: float, axis: int, n_f: int = 81) -> dict:
    """Direct sound from a line of n equal in-phase point projectors centred on the build's projector position,
    against ONE projector of the same far-field source level at broadside: exact spherical-wave sum at every
    hydrophone across the band. Returns the change (dB, negative is quieter) at the worst hydrophone: at the worst
    frequency of the sweep (what an input sees at some instant) and averaged over the band."""
    pos, tx, c = np.asarray(d["positions"], float), np.asarray(d["projector_m"], float), d["sound_speed"]
    f = np.linspace(d["band_hz"][0], d["band_hz"][1], n_f)
    off = (np.arange(n) - 0.5 * (n - 1)) * spacing_m
    src = np.repeat(tx[None, :], n, axis=0)
    src[:, axis] += off
    r = np.linalg.norm(pos[:, None, :] - src[None, :, :], axis=2)                       # [hydrophone, element]
    k = 2.0 * np.pi * f / c
    field = (np.exp(-1j * k[None, None, :] * r[:, :, None]) / r[:, :, None]).sum(axis=1) / n   # each element 1/n of the far-field amplitude
    single = 1.0 / np.linalg.norm(pos - tx, axis=1)
    rel = 20.0 * np.log10(np.abs(field) / single[:, None] + 1e-12)                      # [hydrophone, frequency]
    over_single = np.asarray(d["direct"]["over_full_scale_each_db"], float)
    over = over_single[:, None] + rel
    # directivity index of the line at the band centre (in-phase point sources): n^2 / sum sinc(2 d_ij f / c)
    x = off
    di = 10.0 * math.log10(n * n / np.sinc(2.0 * (x[:, None] - x[None, :]) * d["centre_hz"] / c).sum())
    half = None
    if n > 1:
        th = np.radians(np.linspace(0.0, 90.0, 9001))
        psi = np.pi * spacing_m * d["centre_hz"] / c * np.sin(th)
        with np.errstate(divide="ignore", invalid="ignore"):
            af = np.where(np.abs(np.sin(psi)) < 1e-12, 1.0, np.sin(n * psi) / (n * np.sin(psi)))
        below = np.flatnonzero(np.abs(af) < 10.0 ** (-3.0 / 20.0))
        half = float(np.degrees(th[below[0]])) if below.size else 90.0
    return {"elements": n, "spacing_m": spacing_m, "axis": "along" if axis == 0 else "vertical",
            "worst_overload_peak_db": float(over.max()), "worst_overload_band_mean_db": float((10.0 * np.log10((10.0 ** (over / 10.0)).mean(axis=1))).max()),
            "change_at_worst_frequency_db": float(over.max() - over_single.max()), "per_element_source_level_db": d["source_level_db"] - 20.0 * math.log10(n),
            "directivity_index_db": di, "half_beamwidth_3db_deg": half,
            "acoustic_power_w": acoustic_power_w(d["source_level_db"], di)}


def requirements(spec: dict, levels, depths, areas, ranges=(6000.0, 12000.0)) -> dict:
    d = buildspec.derive(spec)
    c, (f_lo, f_hi), fc, pulse, sl = d["sound_speed"], d["band_hz"], d["centre_hz"], d["pulse_s"], d["source_level_db"]
    interval = d["ping_interval_s"]
    dr = d["direct"]
    pj = spec.get("projector") or {}
    out = {"build": spec["build"], "source_level_db": sl, "source_level_convention": "rms, dB re 1 uPa at 1 m",
           "band_hz": [f_lo, f_hi], "fractional_bandwidth": (f_hi - f_lo) / fc, "wavelength_m": [c / f_hi, c / f_lo],
           "pulse_s": pulse, "ping_interval_s": interval, "duty": pulse / interval, "pings_per_hour": 3600.0 / interval,
           "projector_position_m": d["projector_m"], "projector_model": pj.get("model"), "amplifier": pj.get("amplifier")}
    out["link_ranges_m"] = [float(r) for r in ranges]
    out["link"] = link_table(d, levels, rows=sorted({1, 2, 3} | {d["rows"]}), ranges=tuple(float(r) for r in ranges))
    out["power"] = {f"{sl_:g}": {f"{eta:g}": energy(sl_, pulse, interval, eta) for eta in (0.3, 0.5, 0.7)} for sl_ in levels}
    out["cavitation"] = {
        "threshold_peak_pa": {f"{h:g}": cavitation_pressure_pa(h) for h in depths},
        "intensity_w_cm2": {f"{h:g}": cavitation_pressure_pa(h) ** 2 / (2.0 * RHO * c) / 1e4 for h in depths},
        "limit_db": {f"{a:g}": {f"{h:g}": cavitation_source_level_db(h, a, c=c) for h in depths} for a in areas},
        "least_depth_for_build_level_m": {f"{a:g}": cavitation_depth_m(sl, a, c=c) for a in areas},
        "least_depth_gamma_half_m": {f"{a:g}": cavitation_depth_m(sl, a, gamma=0.5, c=c) for a in areas},
        "area_free_at_surface_m2": acoustic_power_w(sl, c=c) / (cavitation_pressure_pa(0.0) ** 2 / (2.0 * RHO * c)),
        "sphere_diameter_free_at_surface_m": 2.0 * math.sqrt(acoustic_power_w(sl, c=c) / (cavitation_pressure_pa(0.0) ** 2 / (2.0 * RHO * c)) / (4.0 * math.pi)),
    }
    full_scale_db = d["input_full_scale_pressure_db"]
    out["direct"] = {
        "nearest_m": dr["nearest_m"], "farthest_m": dr["farthest_m"], "hydrophones": d["channels"], "overloaded": dr["hydrophones_overloaded"],
        "input_full_scale_db_re_upa": full_scale_db,
        "overload_nearest_db": {"level_taken_as_peak": dr["over_full_scale_db"], "level_taken_as_rms": dr["over_full_scale_db"] + 3.0},
        "overload_farthest_db": {"level_taken_as_peak": dr["over_full_scale_farthest_db"], "level_taken_as_rms": dr["over_full_scale_farthest_db"] + 3.0},
        "reduction_towards_array_needed_db": {"to_full_scale": dr["over_full_scale_db"] + 3.0, "with_6_db_of_margin": dr["over_full_scale_db"] + 9.0},
        "source_level_rms_for_full_scale_db": dr["source_level_for_full_scale_db"] - 3.0,
        "distance_for_full_scale_m": {"level_taken_as_peak": dr["distance_for_full_scale_m"], "level_taken_as_rms": dr["distance_for_full_scale_m"] * 10.0 ** (3.0 / 20.0)},
        "overload_against_distance_rms_db": {f"{x:g}": sl + 3.0 - 20.0 * math.log10(x) - full_scale_db for x in (0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)},
        "lines": [line_direct(d, n, 0.5 * c / fc, axis) for axis in (0, 2) for n in (2, 3, 4)],
    }
    # the same energy with another pulse length: SL falls 10 log10 of the ratio, the blind range grows with the pulse
    trade = []
    for t in sorted({0.1, 0.5, 1.0, 2.0, 4.0, 10.0, pulse}):
        s = sl - 10.0 * math.log10(t / pulse)
        iv = max(interval - pulse + t, t)
        e = energy(s, t, iv, 0.5)
        v_tol = c / (2.0 * (f_hi - f_lo) * t)
        trade.append({"pulse_s": t, "source_level_db": s, "time_bandwidth_db": 10.0 * math.log10((f_hi - f_lo) * t), "blind_range_m": 0.5 * c * t,
                      "ping_interval_s": iv, "duty": t / iv, "overload_nearest_rms_db": dr["over_full_scale_db"] + 3.0 + s - sl,
                      "acoustic_w": e["acoustic_w"], "from_supply_w": e["from_supply_w"], "joule_per_ping": e["joule_per_ping"], "average_w": e["average_w"],
                      "cavitation_depth_0p03m2_m": cavitation_depth_m(s, 0.03, c=c),
                      "lfm_speed_tolerance_m_s": v_tol, "doppler_channels_for_10_m_s": int(math.ceil(10.0 / v_tol))})
    out["pulse_trade"] = trade
    out["band"] = {f"{e:g}": resonance_band_loss(f_lo, f_hi, e) for e in (1.0, 3.0, 6.0, 10.0)}
    out["drive"] = {f"{tvr:g}": drive(sl, tvr) for tvr in (130.0, 135.0, 140.0, 145.0, 150.0)}
    # the surface image (docs/DESIGN.md, "A target near the surface is the hard case"): 20 log10(2 sin(2 pi z1 z2 / (lambda R))) on each leg
    lam, z2, rng = c / fc, 4.0, 12000.0
    out["depth"] = {"target_depth_m": z2, "range_m": rng, "own_depth_for_full_image_gain_m": lam * rng / (4.0 * z2),
                    "surface_image_one_way_db": {f"{h:g}": 20.0 * math.log10(max(2.0 * abs(math.sin(2.0 * math.pi * h * z2 / (lam * rng))), 1e-6)) for h in depths},
                    "pressure_bar_abs": {f"{h:g}": (P_ATM + RHO * G * h) / 1e5 for h in depths}}
    return out


def cavitation_warning(source_level: float, depth_m: float, area_m2: float) -> str | None:
    """The check proposed for the build validator (docs/design/projector_selection.md, Appendix A)."""
    limit = cavitation_source_level_db(depth_m, area_m2)
    if source_level > limit - 3.0:
        return (f"source level {source_level:.0f} dB (rms) is within 3 dB of, or above, the cavitation limit of a {area_m2:g} m2 radiator at {depth_m:g} m "
                f"({limit:.1f} dB): free of cavitation below {cavitation_depth_m(source_level + 3.0, area_m2):.0f} m with 3 dB of margin")
    return None


# --- printing ------------------------------------------------------------------------------------------

def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("| " + " | ".join("---" for _ in head) + " |")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


def report(q: dict, depths, areas):
    print(f"## {q['build']}\n")
    print(f"Source level in the file {q['source_level_db']:g} dB re 1 µPa at 1 m (rms), band {q['band_hz'][0] / 1e3:g}–{q['band_hz'][1] / 1e3:g} kHz "
          f"({100 * q['fractional_bandwidth']:.0f}% of the centre), pulse {q['pulse_s']:g} s every {q['ping_interval_s']:.1f} s "
          f"(duty {100 * q['duty']:.1f}%, {q['pings_per_hour']:.0f} pings an hour).\n")
    print("**What a source level buys** (SNR at the map; detection at 13 dB; ship +25 dB, small +5 dB):\n")
    rr = [f"{r / 1e3:g}km" for r in q["link_ranges_m"]]
    far = rr[-1]
    table(["Rows", "Source level (dB)"] + [f"Ship at {x[:-2]} km (dB)" for x in rr] + [f"margin at {far[:-2]} km (dB)"] + [f"Small at {x[:-2]} km (dB)" for x in rr]
          + ["Reach ship / small (km)"],
          [[r["rows"], f"{r['source_level_db']:g}"] + [f"{r[f'snr_ship_{x}_db']:.1f}" for x in rr] + [f"{r[f'snr_ship_{far}_db'] - DETECTION_DB:+.1f}"]
           + [f"{r[f'snr_small_{x}_db']:.1f}" for x in rr] + [f"{r['reach_ship_m'] / 1e3:.1f} / {r['reach_small_m'] / 1e3:.1f}"] for r in q["link"]])
    print("**Power and energy** (omnidirectional; amplifier 85% efficient; at the build's pulse and interval):\n")
    rows = []
    for sl, by in q["power"].items():
        e3, e5, e7 = by["0.3"], by["0.5"], by["0.7"]
        rows.append([sl, f"{e5['acoustic_w']:.0f}", f"{e3['into_transducer_w']:.0f} / {e5['into_transducer_w']:.0f} / {e7['into_transducer_w']:.0f}",
                     f"{e5['from_supply_w']:.0f}", f"{e5['joule_per_ping']:.0f}", f"{e5['wh_per_hour']:.1f}", f"{e5['heat_in_transducer_average_w']:.1f}"])
    table(["Source level (dB)", "Acoustic (W)", "Into the transducer at 30 / 50 / 70% (W)", "From the supply at 50% (W)", "Energy a ping at 50% (J)",
           "Average at 50% (W = Wh per hour)", "Heat in the transducer, average at 50% (W)"], rows)
    cav = q["cavitation"]
    print("**Cavitation limit** (rms source level a radiator of this wet area holds; uniform face, no tensile strength):\n")
    table(["Wet radiating area", "sphere of that area"] + [f"at {h:g} m" for h in depths] + [f"least depth for {q['source_level_db']:g} dB", "the same with half the face doing the work"],
          [[f"{a:g} m²", f"{2 * math.sqrt(a / (4 * math.pi)):.2f} m dia."] + [f"{cav['limit_db'][f'{a:g}'][f'{h:g}']:.1f} dB" for h in depths]
           + [f"{cav['least_depth_for_build_level_m'][f'{a:g}']:.0f} m", f"{cav['least_depth_gamma_half_m'][f'{a:g}']:.0f} m"] for a in areas])
    print(f"Threshold: " + ", ".join(f"{cav['intensity_w_cm2'][f'{h:g}']:.2f} W/cm² at {h:g} m" for h in depths)
          + f". {q['source_level_db']:g} dB is free of cavitation at the surface from {cav['area_free_at_surface_m2']:.3f} m² of wet radiating area "
          f"(a sphere {cav['sphere_diameter_free_at_surface_m']:.2f} m across).\n")
    dq = q["direct"]
    print(f"**Direct sound** (projector at {q['projector_position_m']} m; nearest hydrophone {dq['nearest_m']:.2f} m, farthest {dq['farthest_m']:.2f} m; "
          f"full scale of an input {dq['input_full_scale_db_re_upa']:.0f} dB re 1 µPa peak; {dq['overloaded']} of {dq['hydrophones']} overloaded):\n")
    table(["", "level taken as a peak (the files' tables)", "level taken as rms (the specification)"],
          [["over full scale at the nearest", f"{dq['overload_nearest_db']['level_taken_as_peak']:+.1f} dB", f"{dq['overload_nearest_db']['level_taken_as_rms']:+.1f} dB"],
           ["over full scale at the farthest", f"{dq['overload_farthest_db']['level_taken_as_peak']:+.1f} dB", f"{dq['overload_farthest_db']['level_taken_as_rms']:+.1f} dB"],
           ["distance at which it is at full scale", f"{dq['distance_for_full_scale_m']['level_taken_as_peak']:.1f} m", f"{dq['distance_for_full_scale_m']['level_taken_as_rms']:.1f} m"]])
    print(f"Reduction towards the array for full scale at the nearest hydrophone: {dq['reduction_towards_array_needed_db']['to_full_scale']:.1f} dB "
          f"({dq['reduction_towards_array_needed_db']['with_6_db_of_margin']:.1f} dB with 6 dB of margin); the rms source level that just reaches full scale there: "
          f"{dq['source_level_rms_for_full_scale_db']:.1f} dB. Against distance (rms level): "
          + ", ".join(f"{x} m: {v:+.0f} dB" for x, v in dq["overload_against_distance_rms_db"].items()) + ".\n")
    print("A broadside line of in-phase projectors at half a wavelength (centre of the band), centred on the same position, same far-field source level:\n")
    table(["Line", "each element (dB)", "directivity index (dB)", "acoustic power (W)", "half beam width at −3 dB", "worst hydrophone, worst frequency (dB over full scale, rms)", "change from one projector (dB)"],
          [[f"{ln['elements']} {ln['axis']}", f"{ln['per_element_source_level_db']:.1f}", f"{ln['directivity_index_db']:.1f}", f"{ln['acoustic_power_w']:.0f}",
            f"{ln['half_beamwidth_3db_deg']:.0f}°", f"{ln['worst_overload_peak_db'] + 3.0:+.1f}", f"{ln['change_at_worst_frequency_db']:+.1f}"] for ln in dq["lines"]])
    print("**The same energy with another pulse** (transducer 50% efficient; linear sweep over the band):\n")
    table(["Pulse (s)", "Source level (dB)", "10 log BT (dB)", "Blind inside (m)", "Ping interval (s)", "Duty", "Nearest hydrophone over full scale (dB, rms)", "Acoustic (W)",
           "From the supply (W)", "Average (W)", "Cavitation-free below (m), 0.03 m²", "Linear sweep tolerates (m/s)", "Doppler channels for ±10 m/s"],
          [[f"{t['pulse_s']:g}", f"{t['source_level_db']:.1f}", f"{t['time_bandwidth_db']:.1f}", f"{t['blind_range_m']:.0f}", f"{t['ping_interval_s']:.1f}", f"{100 * t['duty']:.1f}%",
            f"{t['overload_nearest_rms_db']:+.1f}", f"{t['acoustic_w']:.0f}", f"{t['from_supply_w']:.0f}", f"{t['average_w']:.0f}", f"{t['cavitation_depth_0p03m2_m']:.0f}",
            f"±{t['lfm_speed_tolerance_m_s']:.2f}", 2 * t["doppler_channels_for_10_m_s"] + 1] for t in q["pulse_trade"]])
    print("**Flatness across the band** (one resonance in the middle, constant drive voltage):\n")
    table(["Response at the band edges", "Mechanical Q", "Energy lost over the sweep", "Voltage to equalise"],
          [[f"−{e} dB", f"{b['q']:.1f}", f"{b['mean_loss_db']:.1f} dB", f"×{b['volts_factor_to_equalise']:.2f}"] for e, b in q["band"].items()])
    print(f"**Drive voltage for {q['source_level_db']:g} dB** (V rms = 10^((SL − TVR) / 20); volt-amperes = V² / |Z|; real power = V² cos φ / |Z|):\n")
    table(["TVR (dB re 1 µPa/V at 1 m)", "V rms", "V peak"], [[t, f"{v['volts_rms']:.0f}", f"{v['volts_peak']:.0f}"] for t, v in q["drive"].items()])
    dp = q["depth"]
    print(f"**Own depth** (target at {dp['target_depth_m']:g} m, {dp['range_m'] / 1e3:g} km; the surface-image factor of docs/DESIGN.md on each leg; "
          f"it reaches its maximum at {dp['own_depth_for_full_image_gain_m']:.0f} m):\n")
    table(["Own depth (m)"] + [f"{h:g}" for h in depths],
          [["surface image, one way (dB)"] + [f"{dp['surface_image_one_way_db'][f'{h:g}']:+.1f}" for h in depths],
           ["absolute pressure (bar)"] + [f"{dp['pressure_bar_abs'][f'{h:g}']:.1f}" for h in depths],
           [f"cavitation limit, 0.03 m² (dB)"] + [f"{cavitation_source_level_db(h, 0.03):.1f}" for h in depths]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("builds", nargs="*", help="build names or spec files (default: every shipped build)")
    ap.add_argument("--levels", type=float, nargs="+", default=[190.0, 194.0, 197.0, 200.0, 203.0])
    ap.add_argument("--depths", type=float, nargs="+", default=[5.0, 10.0, 30.0, 100.0, 200.0])
    ap.add_argument("--areas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--ranges", type=float, nargs="+", default=[6000.0, 12000.0], help="ranges (m) of the link table")
    ap.add_argument("--no-json", action="store_true")
    args = ap.parse_args()
    names = args.builds or buildspec.shipped()
    out = {}
    for name in names:
        path = Path(name) if Path(name).suffix else ROOT / "configs" / "builds" / f"{name}.yaml"
        spec = buildspec.load(path)
        q = requirements(spec, args.levels, args.depths, args.areas, args.ranges)
        report(q, args.depths, args.areas)
        out[q["build"]] = q
    if not args.no_json:
        dest = ROOT / "docs" / "design" / "projector_requirements.json"
        dest.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"wrote {dest.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
