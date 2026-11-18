#!/usr/bin/env python3
"""Listed projectors against the projector requirement of a build.

Every figure in CANDIDATES was read from the manufacturer's public data sheet or product page named in
`url` (opened 2026-11-18). Values marked `read` were read off a plotted curve: TVR to about +/-1 dB,
admittance and impedance to about +/-15 %, beam patterns to about +/-2 dB. Nothing is from memory.

For each candidate this prints, with the conventions of tools/projector_requirements.py (source level
rms, dB re 1 uPa at 1 m; SL = TVR + 20 log10 V_rms):

  the source level at 4, 5 and 6 kHz at the sheet's rated drive for a duty of at least the build's;
  the volts (and where the sheet gives admittance or impedance, the watts and volt-amperes) for the
  build's level and for 3 and 6 dB less; how many units driven in phase give the build's level across
  the band (20 log10 N on the weakest frequency, mutual coupling ignored);
  the cavitation limit at depth: the MAKER'S own statement scaled as 20 log10 of the static pressure, and
  the upper bound of tools/projector_requirements.py from the unit's wet radiating area;
  the depth rating against the owner's 500 m;
  the direct sound left at the build's hydrophones with the unit's own pattern.

Usage: tools/projector_candidates.py [build] [--depth-required 500] [--no-json]
Writes docs/design/projector_candidates.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import projector_requirements as pr  # noqa: E402
from gpusonar import buildspec  # noqa: E402

F = (4000.0, 5000.0, 6000.0)
NEP = "https://www.neptune-sonar.co.uk/downloads/"
GEO = "https://geospectrum.ca/wp-content/uploads/2022/05/"
BW = "https://www.benthowave.com/products/Specs/"

# tvr: dB re 1 uPa/V at 1 m at 4, 5, 6 kHz (read off the sheet's curve). drive: [(V rms, duty)] from the sheet, or
# power: [(W, duty)] with z_ohm: |Z| at 4, 5, 6 kHz. g_ms / b_ms: conductance / susceptance (mS) at 4, 5, 6 kHz (read).
# cav: (level dB or None, V rms or None, W or None, minimum depth m) as the maker states it. pattern: 'omni', 'toroid' or
# 'piston' with to_side_db: response at 90 deg from the axis at 4, 5, 6 kHz (read; None: not on the sheet).
CANDIDATES = [
    {"model": "Neptune T335", "type": "Tonpilz piston with internal tuning network, directional", "url": NEP + "3aced3db-b5b4-4f6e-8dd1-d0ed3e1bcbd6/T335.pdf",
     "where": "catalogue pp. B27-B28: specification table; Transmit Sensitivity, Admittance and Beam Pattern graphs",
     "resonance": "3.5 / 7.0 kHz", "band": "3-8 kHz", "tvr": (146.5, 146.5, 146.5), "tvr_sheet": "146 dB re 1 uPa/V at 1 m",
     "drive": [(750.0, 0.01), (400.0, 0.10), (100.0, 1.0)], "g_ms": (2.6, 2.2, 3.0), "b_ms": (-2.0, 0.5, 1.0), "impedance": "160 ohm nominal",
     "pattern": "piston", "beam": "conical; at 5 kHz about -3 dB at +/-45 deg", "to_side_db": (-9.5, -12.0, -14.0), "efficiency": None,
     "depth_m": 600.0, "depth_words": "Operating Depth 600 m", "size": "183 x 183 mm flange, 257 mm long, body dia. 120 mm", "mass": "11.4 kg air / 6.7 kg water",
     "area_m2": math.pi * 0.076 ** 2, "cav": None, "convention": "drive in V rms; level convention not stated", "pulse_max": "not stated",
     "note": "the product page prints '46 dB' for the transmit sensitivity (a typo); the data sheet says 146 dB"},
    {"model": "Massa TR-1075A", "type": "Tonpilz piston, parallel-tuned with transformer, directional", "url": "https://www.massa.com/wp-content/uploads/2023/03/TR-1075-Datasheet-230327.pdf",
     "where": "pp. 1-3: text, tuning chart, Nominal Response Curves (TVR and impedance magnitude)",
     "resonance": "3.5 kHz untuned; tuned for 0 deg phase at 4 kHz", "band": "2.5-10 kHz", "tvr": (152.5, 150.0, 149.5), "tvr_sheet": "curve only",
     "power": [(600.0, 0.30), (200.0, 1.0)], "z_ohm": (85.0, 190.0, 95.0), "impedance": "100 ohm at 4 kHz nominal",
     "pattern": "piston", "beam": "80 deg conical beam angle; piston half a wavelength across at 4 kHz", "to_side_db": (None, None, None), "efficiency": None,
     "depth_m": 600.0, "depth_words": "Maximum Operating Depth is 600 meters", "size": "7.25 in (184 mm) square flange, about 8.6 in (218 mm) long", "mass": "25 lb (11.3 kg) in air; in water not stated",
     "area_m2": math.pi * 0.094 ** 2, "cav": (None, None, 600.0, 30.0), "cav2": (None, None, 200.0, 9.0),
     "convention": "not stated (power in watts; curves 'dB re 1 uPa/1V @ 1m')", "pulse_max": "not stated",
     "note": "volts from watts assume a resistive load (V = sqrt(P |Z|)); the phase is given only at 4 kHz. Minimum depths 'approximately 30 and 100 feet' for 200 and 600 W"},
    {"model": "Massa TR-1075D", "type": "Tonpilz piston, series-tuned with shunt resistor (lower Q)", "url": "https://www.massa.com/wp-content/uploads/2023/03/TR-1075-Datasheet-230327.pdf",
     "where": "pp. 2-3", "resonance": "tuned at 3.5 kHz", "band": "2.5-10 kHz", "tvr": (147.5, 146.5, 148.5), "tvr_sheet": "curve only",
     "power": [(1000.0, 0.30), (335.0, 1.0)], "z_ohm": (280.0, 390.0, 210.0), "impedance": "200 ohm at 3.5 kHz nominal",
     "pattern": "piston", "beam": "80 deg conical beam angle", "to_side_db": (None, None, None), "efficiency": None,
     "depth_m": 600.0, "depth_words": "Maximum Operating Depth is 600 meters", "size": "as TR-1075A", "mass": "25 lb (11.3 kg) in air",
     "area_m2": math.pi * 0.094 ** 2, "cav": (None, None, 600.0, 30.0), "convention": "not stated", "pulse_max": "not stated",
     "note": "part of the input power is dissipated in the shunt resistor; the cavitation depths are quoted for the family at 200 and 600 W"},
    {"model": "Neptune T296", "type": "piston projector, directional", "url": NEP + "8682c070-8807-4d6c-bc6c-a37dcb3d849e/T296.pdf",
     "where": "catalogue pp. B19-B20", "resonance": "1.7 kHz", "band": "1-6 kHz", "tvr": (139.5, 142.0, 137.5), "tvr_sheet": "142 dB re 1 uPa/V at 1 m",
     "drive": [(750.0, 0.10), (350.0, 1.0)], "impedance": "400 ohm nominal",
     "pattern": "piston", "beam": "conical; patterns given at 1.7 and 3.5 kHz only", "to_side_db": (None, None, None), "efficiency": None,
     "depth_m": 600.0, "depth_words": "Operating Depth 600 m", "size": "260 x 260 mm flange, 418 mm long, face dia. 264 mm", "mass": "27.1 kg air / 12.2 kg water",
     "area_m2": math.pi * 0.132 ** 2, "cav": None, "convention": "drive in V rms; level convention not stated", "pulse_max": "not stated", "note": "the band ends at 6 kHz"},
    {"model": "Neptune T170", "type": "free-flooded ring", "url": NEP + "b76a71d6-dd89-48b1-8026-63b45c5761d3/T170.pdf",
     "where": "catalogue pp. B23-B24", "resonance": "6 kHz", "band": "4-8 kHz", "tvr": (135.0, 139.0, 140.0), "tvr_sheet": "140 dB re 1 uPa/V at 1 m",
     "drive": [(750.0, 0.10), (250.0, 1.0)], "g_ms": (0.7, 1.1, 1.2), "b_ms": (1.8, 1.65, 1.4), "impedance": "68 nF at 1 kHz with 10 m of cable",
     "pattern": "toroid", "beam": "omni +/-2 dB around the axis; toroidal, nulls on the axis (patterns at 4 and 6 kHz)", "efficiency": None,
     "depth_m": math.inf, "depth_words": "Operating Depth Unlimited", "size": "ring 191 mm outside, 121 mm inside diameter, 85 mm high", "mass": "4.7 kg air / 2.6 kg water with 10 m of cable",
     "area_m2": math.pi * 0.085 * (0.191 + 0.121) + 0.5 * math.pi * (0.191 ** 2 - 0.121 ** 2), "cav": (None, 750.0, None, 10.0),
     "convention": "drive in V rms; level convention not stated", "pulse_max": "not stated", "note": "'750 Vrms (Minimum 10m depth)' is on the PDF, not on the web page"},
    {"model": "GeoSpectrum M27-450", "type": "free-flooded ring", "url": GEO + "SP-M27-450-R0.pdf",
     "where": "pp. 1-2: characteristics table; Figures 1 (TVR), 2 (admittance), 3 (beam), 4 (dimensions)", "resonance": "4250 Hz", "band": "curve 2.5-6.5 kHz", "tvr": (136.0, 135.0, 134.0),
     "tvr_sheet": "138 dB re 1 uPa/V at 1 m at resonance", "drive": [(600.0, None)], "g_ms": (0.35, 0.12, 0.2), "b_ms": (1.8, 2.0, 2.5), "impedance": "60 nF",
     "pattern": "toroid", "beam": "toroidal (pattern at 4.25 kHz)", "efficiency": None,
     "depth_m": 6000.0, "depth_words": "Maximum Operating Depth 6000 Meters", "size": "dia. 6.5 in (165 mm) flange, 5.9 in body, 2.71 in bore, 6.7 in (170 mm) high", "mass": "5 kg in air; in water not stated",
     "area_m2": math.pi * 0.137 * (0.150 + 0.069), "cav": (193.0, None, None, 30.0), "max_spl": 193.0,
     "convention": "drive in V rms; level convention not stated", "pulse_max": "not stated; duty not stated", "note": "'Cavitation Depth at 193 dB re uPa @ 1 m: 30 Meters'"},
    {"model": "Neptune T160", "type": "free-flooded ring", "url": NEP + "8df0fff1-08ad-40c2-a577-90a755c087ca/T160.pdf",
     "where": "catalogue pp. B21-B22", "resonance": "2.8 kHz", "band": "2.5-6 kHz", "tvr": (133.5, 134.0, 130.0), "tvr_sheet": "134 dB re 1 uPa/V at 1 m",
     "drive": [(750.0, 0.10), (250.0, 1.0)], "impedance": "38 nF at 1 kHz with 10 m of cable", "pattern": "toroid", "beam": "toroidal; lobed at 6 kHz", "efficiency": None,
     "depth_m": math.inf, "depth_words": "Operating Depth Unlimited", "size": "ring 234 mm outside, 160 mm inside diameter, 106 mm high", "mass": "5.7 kg air / 2.6 kg water with 10 m of cable",
     "area_m2": math.pi * 0.106 * (0.234 + 0.160) + 0.5 * math.pi * (0.234 ** 2 - 0.160 ** 2), "cav": (None, 750.0, None, 10.0),
     "convention": "drive in V rms; level convention not stated", "pulse_max": "not stated", "note": ""},
    {"model": "GeoSpectrum M21-070-5000", "type": "flexural disc (bender)", "url": GEO + "SP-M21-070-5000-R1.pdf",
     "where": "pp. 1-2: characteristics table; Figures 1 (TVR), 2 (admittance), 3-4 (beams), 5 (dimensions)", "resonance": "5 kHz", "band": "1-10 kHz transmitting range", "tvr": (126.0, 136.5, 131.0),
     "tvr_sheet": "137 dB re 1 uPa/V at 1 m maximum", "drive": [(600.0, None)], "g_ms": (0.03, 0.43, 0.05), "b_ms": (0.9, 1.1, 1.2), "impedance": "35 nF",
     "pattern": "omni", "beam": "omnidirectional at resonance", "efficiency": None,
     "depth_m": 700.0, "depth_words": "Maximum Operating Depth 700 Meters", "size": "disc dia. 80 mm, 25 mm thick", "mass": "0.6 kg",
     "area_m2": 2.0 * math.pi * 0.04 ** 2, "cav": (190.0, None, None, 25.0), "max_spl": 190.0,
     "convention": "drive in V rms; level convention not stated", "pulse_max": "not stated; duty not stated", "note": "'Cavitation Depth at 190 dB re uPa @1m: 25 Meters'. Sharp: about 0.85 kHz wide at -3 dB"},
    {"model": "Neptune T420", "type": "small piston projector", "url": NEP + "fbeb4025-8243-4084-b976-7a59740bfbfa/T420.pdf",
     "where": "catalogue pp. B29-B30", "resonance": "5 kHz", "band": "4.5-14 kHz", "tvr": (115.0, 126.5, 124.0), "tvr_sheet": "126 dB re 1 uPa/V at 1 m",
     "drive": [(2000.0, 0.10), (600.0, 1.0)], "impedance": "20 kohm nominal", "pattern": "piston", "beam": "at 5 kHz about -7 dB at 90 deg, -5 dB behind", "to_side_db": (None, -7.0, None), "efficiency": None,
     "depth_m": 600.0, "depth_words": "Operating Depth 600 m", "size": "102 x 102 mm flange, 165 mm long", "mass": "3.8 kg air / 2.1 kg water",
     "area_m2": math.pi * 0.04 ** 2, "cav": (None, 2000.0, None, 20.0), "convention": "drive in V rms; level convention not stated", "pulse_max": "not stated", "note": "'2,000 Vrms (Minimum 20m depth)'"},
    {"model": "Benthowave BII7594", "type": "free-flooded ring, multi-resonance", "url": BW + "BII7590Datasheet.pdf",
     "where": "pp. 1-2 specification table; p. 6 TVR graph", "resonance": "4, 7.5, 11 kHz", "band": "1-30 kHz (series)", "tvr": (130.0, 117.0, 118.0), "tvr_sheet": "graph",
     "drive": [(400.0, 0.20)], "impedance": "115 nF; power factor 0.36 at fs", "pattern": "toroid", "beam": "omnidirectional to toroidal", "efficiency": "0.9 +/-5 % at fs",
     "depth_m": 500.0, "depth_words": "Operating Depth <= 500 m", "size": "dia. 114 x 90 mm", "mass": ">= 1.2 kg in air",
     "area_m2": math.pi * 0.09 * 0.114 * 1.6, "cav": None, "convention": "'SL = 20*logVi + TVR ... Vi in Vrms'", "pulse_max": "30 s at 480 W, duty <= 20 %",
     "note": "the TVR swings 13 dB between 4 and 5.5 kHz; the maker says the series 'is not designed for underwater acoustic detection'"},
    {"model": "Benthowave BII7596", "type": "free-flooded ring, multi-resonance", "url": BW + "BII7590Datasheet.pdf",
     "where": "pp. 1-2 specification table; p. 8 TVR and admittance graphs", "resonance": "3.5 and 6.5 kHz", "band": "1-30 kHz (series)", "tvr": (131.0, 124.0, 130.0), "tvr_sheet": "graph",
     "drive": [(600.0, 0.40)], "impedance": "69 nF; power factor 0.38 at fs", "pattern": "toroid", "beam": "omnidirectional to toroidal", "efficiency": "0.9 +/-5 % at fs",
     "depth_m": 300.0, "depth_words": "Operating Depth <= 300 m", "size": "dia. 168 x 115 mm", "mass": ">= 2.2 kg in air",
     "area_m2": math.pi * 0.115 * 0.168 * 1.6, "cav": None, "convention": "'SL = 20*logVi + TVR ... Vi in Vrms'", "pulse_max": "200 s at 600 W, duty <= 40 %", "note": "ripple of +/-4 dB across the band, cable-length dependent"},
]


def pressure_ratio_db(h: float, h_ref: float) -> float:
    return 20.0 * math.log10(pr.cavitation_pressure_pa(h) / pr.cavitation_pressure_pa(h_ref))


def rated_voltage(c: dict, duty: float):
    """Largest sheet voltage whose duty rating covers `duty`, per frequency (a tuple of three)."""
    if "drive" in c:
        ok = [v for v, d in c["drive"] if d is None or d >= duty]
        return (max(ok),) * 3 if ok else None
    ok = [p for p, d in c["power"] if d >= duty]
    return tuple(math.sqrt(max(ok) * z) for z in c["z_ohm"]) if ok else None


def evaluate(c: dict, d: dict, depth_required: float, depths) -> dict:
    sl, duty = d["source_level_db"], d["pulse_s"] / d["ping_interval_s"]
    v = rated_voltage(c, duty)
    at_rated = [t + 20.0 * math.log10(x) for t, x in zip(c["tvr"], v)]
    out = {"model": c["model"], "type": c["type"], "url": c["url"], "where": c["where"], "rated_volts_rms": list(v), "level_at_rated_db": at_rated,
           "meets_level_across_band": min(at_rated) >= sl, "short_by_db": max(sl - min(at_rated), 0.0),
           "units_for_level": int(math.ceil(10.0 ** (max(sl - min(at_rated), 0.0) / 20.0) - 1e-9)),
           "depth_rating_m": c["depth_m"], "depth_ok": c["depth_m"] >= depth_required, "depth_margin": c["depth_m"] / depth_required}
    need = {}
    for lv in (sl, sl - 3.0, sl - 6.0):
        volts = [10.0 ** ((lv - t) / 20.0) for t in c["tvr"]]
        row = {"volts_rms": volts, "within_rating": all(x <= y * 1.001 for x, y in zip(volts, v))}
        if "g_ms" in c:
            row["watts"] = [x * x * g * 1e-3 for x, g in zip(volts, c["g_ms"])]
            row["volt_amperes"] = [x * x * math.hypot(g, b) * 1e-3 for x, g, b in zip(volts, c["g_ms"], c["b_ms"])]
        elif "z_ohm" in c:
            row["volt_amperes"] = [x * x / z for x, z in zip(volts, c["z_ohm"])]
        need[f"{lv:g}"] = row
    out["drive_for"] = need
    # cavitation: the maker's statement scaled with static pressure; and the upper bound from the wet area
    cav = c.get("cav")
    anchor = None
    if cav:
        lv, volts, watts, h0 = cav
        if lv is None and volts is not None:
            lv = max(c["tvr"]) + 20.0 * math.log10(volts) if "tvr_peak" not in c else c["tvr_peak"] + 20.0 * math.log10(volts)
        if lv is None and watts is not None:
            lv = max(t + 10.0 * math.log10(watts * z) for t, z in zip(c["tvr"], c["z_ohm"]))
        anchor = {"level_db": lv, "depth_m": h0}
        out["cavitation_maker_db"] = {f"{h:g}": lv + pressure_ratio_db(h, h0) for h in depths}
        out["least_depth_for_level_maker_m"] = max((pr.cavitation_pressure_pa(h0) * 10.0 ** ((sl - lv) / 20.0) - pr.P_ATM + pr.P_VAPOUR) / (pr.RHO * pr.G), 0.0)
    out["cavitation_anchor"] = anchor
    out["cavitation_bound_db"] = {f"{h:g}": pr.cavitation_source_level_db(h, c["area_m2"]) for h in depths}
    if anchor:
        out["bound_minus_maker_db"] = pr.cavitation_source_level_db(anchor["depth_m"], c["area_m2"]) - anchor["level_db"]
    # direct sound with the unit's own pattern (rms convention: +3 dB on the files' figures)
    over = np.asarray(d["direct"]["over_full_scale_each_db"], float) + 3.0
    pos, tx = np.asarray(d["positions"], float), np.asarray(d["projector_m"], float)
    if c["pattern"] == "toroid":
        # ring axis along the face (towards the rows): response sin(angle from the axis), floor -20 dB (the plotted patterns' floor)
        rel = pos - tx
        ang = np.arccos(np.clip(np.abs(rel[:, 0]) / np.linalg.norm(rel, axis=1), 0.0, 1.0))
        red = np.maximum(20.0 * np.log10(np.maximum(np.sin(ang), 1e-3)), -20.0)
        out["direct_worst_db"] = float((over + red).max())
        out["direct_how"] = "ring axis along the face, pointing at the rows (sin pattern, floor -20 dB; far-field pattern applied in the near field)"
        out["direct_reduction_db"] = float((over + red).max() - over.max())
    elif c["pattern"] == "piston" and any(x is not None for x in c.get("to_side_db", ())):
        side = max(x for x in c["to_side_db"] if x is not None)
        out["direct_worst_db"] = float(over.max() + side)
        out["direct_reduction_db"] = float(side)
        out["direct_how"] = "piston facing out from the face; the rows are 90 deg off its axis (worst frequency read from the sheet's pattern)"
    else:
        out["direct_worst_db"] = float(over.max())
        out["direct_reduction_db"] = 0.0
        out["direct_how"] = "no reduction taken: " + ("omnidirectional" if c["pattern"] == "omni" else "the sheet gives no response at 90 deg in this band")
    return out


def f3(x, fmt="{:.1f}"):
    return " / ".join(fmt.format(v) for v in x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("build", nargs="?", default="popoto_face_3x32")
    ap.add_argument("--depth-required", type=float, default=500.0, help="the owner: 'Depth no further than 500m'")
    ap.add_argument("--depths", type=float, nargs="+", default=[5.0, 10.0, 30.0, 100.0, 200.0, 500.0])
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    spec = buildspec.load(ROOT / "configs" / "builds" / f"{a.build}.yaml")
    d = buildspec.derive(spec)
    sl, duty = d["source_level_db"], d["pulse_s"] / d["ping_interval_s"]
    res = [evaluate(c, d, a.depth_required, a.depths) for c in CANDIDATES]
    print(f"## Candidates against {a.build}: {sl:g} dB rms, {d['band_hz'][0] / 1e3:g}-{d['band_hz'][1] / 1e3:g} kHz, duty {100 * duty:.1f} %, depth {a.depth_required:g} m\n")
    print("**Data sheet values** (read off curves where the sheet gives a curve):\n")
    pr.table(["Model", "Type", "Resonance; band", "TVR at 4 / 5 / 6 kHz (dB re 1 µPa/V at 1 m)", "Rated drive (duty)", "Pattern", "Depth", "Size", "Mass", "Impedance", "Convention; pulse length"],
             [[c["model"], c["type"], f"{c['resonance']}; {c['band']}", f3(c["tvr"]) + f" (sheet: {c['tvr_sheet']})",
               "; ".join((f"{v:g} V rms" if "drive" in c else f"{v:g} W") + (f" at {100 * dd:g} %" if dd is not None else " (duty not stated)") for v, dd in c.get("drive", c.get("power"))),
               c["beam"], c["depth_words"], c["size"], c["mass"], c["impedance"], f"{c['convention']}; {c['pulse_max']}"] for c in CANDIDATES])
    print("**Computed** (level = TVR + 20 log V rms at the largest drive the sheet rates for this duty):\n")
    rows = []
    for c, r in zip(CANDIDATES, res):
        need = r["drive_for"][f"{sl:g}"]
        power = ""
        if "watts" in need:
            power = f"; {f3(need['watts'], '{:.0f}')} W, {f3(need['volt_amperes'], '{:.0f}')} VA"
        elif "volt_amperes" in need:
            power = f"; {f3(need['volt_amperes'], '{:.0f}')} VA"
        rows.append([c["model"], f3(r["level_at_rated_db"]), "**yes**" if r["meets_level_across_band"] else f"no, {r['short_by_db']:.1f} dB short", r["units_for_level"],
                     f3(need["volts_rms"], "{:.0f}") + " V" + power + ("" if need["within_rating"] else " (over the rating)"),
                     f3(r["drive_for"][f"{sl - 3:g}"]["volts_rms"], "{:.0f}") + " V" + ("" if r["drive_for"][f"{sl - 3:g}"]["within_rating"] else " (over)"),
                     f3(r["drive_for"][f"{sl - 6:g}"]["volts_rms"], "{:.0f}") + " V" + ("" if r["drive_for"][f"{sl - 6:g}"]["within_rating"] else " (over)"),
                     ("unlimited" if math.isinf(c["depth_m"]) else f"{c['depth_m']:g} m") + (": ok" if r["depth_ok"] else ": **fails**") + ("" if math.isinf(c["depth_m"]) else f" (×{r['depth_margin']:.1f})"),
                     f"{r['direct_worst_db']:+.1f} dB ({r['direct_reduction_db']:+.1f})"])
    pr.table(["Model", "Level at rated drive, 4 / 5 / 6 kHz (dB)", f"{sl:g} dB across the band from one unit", f"Units in phase for {sl:g} dB",
              f"For {sl:g} dB at 4 / 5 / 6 kHz", f"For {sl - 3:g} dB", f"For {sl - 6:g} dB", f"Depth rating against {a.depth_required:g} m", f"Direct sound at {sl:g} dB, worst hydrophone (change)"], rows)
    print("**Cavitation** (rms source level; the maker's statement scaled as 20 log of static pressure, and the uniform-face bound of projector_requirements.py for the unit's wet area):\n")
    rows = []
    for c, r in zip(CANDIDATES, res):
        an = r["cavitation_anchor"]
        rows.append([c["model"], f"{c['area_m2']:.3f} m²", f"{an['level_db']:.1f} dB at {an['depth_m']:g} m" if an else "none on the sheet"]
                    + [(f"{r['cavitation_maker_db'][f'{h:g}']:.1f}" if an else "—") + f" / {r['cavitation_bound_db'][f'{h:g}']:.1f}" for h in a.depths]
                    + [f"{r['least_depth_for_level_maker_m']:.0f} m" if an else "—", f"{r['bound_minus_maker_db']:.1f} dB" if an else "—"])
    pr.table(["Model", "Wet radiating area (from the drawing)", "Maker's statement"] + [f"at {h:g} m: maker / bound (dB)" for h in a.depths]
             + [f"Least depth for {sl:g} dB (maker)", "Bound above the maker"], rows)
    if not a.no_json:
        dest = ROOT / "docs" / "design" / "projector_candidates.json"
        dest.write_text(json.dumps({"build": a.build, "source_level_db": sl, "duty": duty, "depth_required_m": a.depth_required,
                                    "candidates": [{**{k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in c.items()}, "computed": r} for c, r in zip(CANDIDATES, res)]},
                                   indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"wrote {dest.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
