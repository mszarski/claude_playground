#!/usr/bin/env python3
"""The per-build numbers of docs/handoff/HARDWARE.md, computed from the shipped files.

    tools/handoff_hardware.py                       # -> docs/handoff/hardware_builds.json
    tools/handoff_hardware.py --markdown            # also print the tables HARDWARE.md carries
    tools/handoff_hardware.py --out FILE

Nothing is typed in here. For every build spec under configs/builds/ the numbers are
`gpusonar.buildspec.describe_shipped` (geometry, sampling, data rates, the projector's direct sound, clock
tolerances) and `gpusonar.buildspec.performance` (the link budget against the build's own detector) of its
generated sonar and deployment files, and the block sizes of the UDP stream (`gpusonar.sources.flank_stream`).
The "design range" of a build is the far end of its receive window (`receive.window.range` of its sonar file);
the face builds are also evaluated at 12 km, the design point of docs/DESIGN.md.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "handoff" / "hardware_builds.json"
DESIGN_POINT_M = 12000.0   # docs/DESIGN.md, "The 12 km design point"


def build_numbers(name: str) -> dict:
    from gpusonar import buildspec
    from gpusonar.sources import flank_stream as fs

    spec = buildspec.load(name)
    files = buildspec.names(spec)
    d = buildspec.describe_shipped(files["sonar"], files["deploy"])
    far = float(d["window_m"][1])
    ranges = sorted({1000.0, far} | ({DESIGN_POINT_M} if far >= DESIGN_POINT_M else set()))
    p = buildspec.performance(d, ranges=tuple(ranges))
    ch, rate = int(d["channels"]), float(d["sample_rate_hz"])
    block = fs.block_samples_for(ch)
    block_bytes = fs.HEADER_BYTES + block * ch * fs.SAMPLE_BYTES
    el = d.get("elevation") or {}
    at = lambda table, r: table[r] if r in table else table[str(r)]   # noqa: E731
    out = {
        "build": name, "stage": spec.get("stage"), "title": spec.get("title"),
        "sonar_file": f"configs/sonars/{files['sonar']}.yaml", "clocks_file": f"configs/sonars/{files['sonar']}_clocks.yaml",
        "deploy_file": f"configs/deploy/{files['deploy']}.yaml", "row_module": (spec.get("array") or {}).get("module"),
        "platform": d["platform"],
        "rows": d["rows"], "per_row": d["per_row"], "channels": ch, "pitch_m": round(d["pitch_m"], 4), "row_length_m": round(d["row_length_m"], 3),
        "pitch_wavelengths_top_of_band": round(d["pitch_wavelengths_top"], 3), "heights_m": d["heights_m"],
        "boards": d["n_boards"], "inputs_per_board": d["inputs_per_board"], "reference_channels": (d.get("reference") or {}).get("channels"),
        "band_hz": d["band_hz"], "pulse_s": d["pulse_s"], "source_level_db_rms": d["source_level_db"],
        "sample_rate_hz": rate, "adc_bits": d["adc_bits"], "adc_full_scale_v": d["adc_full_scale_v"], "gain_db": d["gain_db"],
        "sensitivity_db": d["sensitivity_db"], "input_full_scale_pressure_db": round(d["input_full_scale_pressure_db"], 1),
        "projector_m": d["projector_m"],
        "direct_sound": {k: round(float(d["direct"][k]), 4) for k in ("nearest_m", "farthest_m", "level_db", "volts", "over_full_scale_db", "over_full_scale_farthest_db",
                                                                    "ends_s", "min_range_memoryless_m", "source_level_for_full_scale_db", "distance_for_full_scale_m")},
        "blind_range_m": d["blind_range_m"], "window_m": d["window_m"], "ping_interval_s": round(d["ping_interval_s"], 3), "samples_per_ping": d["samples_per_ping"],
        "reference_times_s": d.get("reference_times_s"),
        "rate_board_bit_s": d["rate_board_bit_s"], "rate_total_bit_s": d["rate_total_bit_s"],
        "stream": {"sample_bytes": fs.SAMPLE_BYTES, "payload_bytes_per_s": int(ch * rate * fs.SAMPLE_BYTES), "block_samples": block, "block_bytes": block_bytes,
                   "blocks_per_s": round(rate / block, 1), "bytes_per_s_with_headers": int(round(rate / block * block_bytes)),
                   "socket_buffer_asked_bytes": fs.stream_buffer_bytes(ch, rate)},
        "record_bytes_per_ping": d["record_bytes_per_ping"], "record_bytes_per_hour_raw": d["record_bytes_per_hour"],
        "record_bytes_per_hour_decimated": d["record_bytes_per_hour_decimated"], "recording_mode": d["recording_mode"],
        "clock_tolerance_within_row_us": round(d["alignment_tolerance_us"], 2),
        "clock_tolerance_between_rows_us": round(el["clock_us_for_error"], 2) if el.get("clock_us_for_error") else None,
        "beamwidth_deg": round(p["beamwidth_deg"], 2), "range_resolution_m": p["range_resolution_m"],
        "grating_lobe_free_steering_deg": round(p["grating_lobe_free_steering_deg"], 1), "bearings_deg": ((d.get("deploy") or {}).get("chain") or {}).get("bearings_deg"),
        "row_gain_sea_noise_db": round(p["gain_isotropic_noise_db"], 1),
        "elevation": None if not el else {"baseline_m": el["baseline_m"], "repeat_spacing_deg": round(el["repeat_spacing_deg"], 1),
                                          "highest_repeat_db": round(el["highest_repeat_db"], 1), "sigma_deg_at_20_db": round(at(el["sigma_deg"], 20.0), 2),
                                          "deg_per_us": round(el["deg_per_us"], 3), "deg_per_mm_out": round(el["deg_per_mm_out"], 3)},
        "detector": {"needs_one_row_snr_db": round(p["detection"]["threshold_db"], 1), "at_probability": p["detection"]["probability"],
                     "needs_for_0p9_db": round(p["detection"]["threshold_high_db"], 1), "false_alarms_per_ping": p["detection"]["false_alarms_per_ping"]},
        "reach_ship_m": round(p["reach_ship_m"]), "reach_small_m": round(p["reach_small_m"]),
        "noise_spectrum_db": round(p["noise_spectrum_db"], 1), "assumptions": p["assumptions"],
        "at_range": {str(int(r)): {"row_snr_ship_db": round(at(p["row_snr_db"]["ship"], r), 1), "margin_ship_db": round(at(p["margin_db"]["ship"], r), 1),
                                   "pd_ship": round(at(p["detection_probability"]["ship"], r), 2), "margin_small_db": round(at(p["margin_db"]["small"], r), 1),
                                   "pd_small": round(at(p["detection_probability"]["small"], r), 2)} for r in ranges},
        "design_range_m": far,
        "unconfirmed": d.get("unconfirmed"),
    }
    return out


def markdown(builds: list[dict]) -> str:
    def table(head, rows):
        return "\n".join(["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])

    out = []
    for b in builds:
        s, ds = b["stream"], b["direct_sound"]
        far = str(int(b["design_range_m"]))
        rows = [
            ("Hydrophones", f"{b['rows']} x {b['per_row'][0]} = {b['channels']}, pitch {b['pitch_m']:.2f} m ({b['pitch_wavelengths_top_of_band']:.2f} wavelengths at the top of the band), row {b['row_length_m']:.2f} m"
                            + (f", rows at {', '.join(f'{h:+.2f}' for h in b['heights_m'])} m" if b["rows"] > 1 else "")),
            ("Boards", f"{b['boards']} of {b['inputs_per_board']} inputs; reference summed into inputs {b['reference_channels']}"),
            ("Transmit", f"{b['band_hz'][0] / 1e3:g}-{b['band_hz'][1] / 1e3:g} kHz, {b['pulse_s']:g} s, {b['source_level_db_rms']:g} dB re 1 uPa at 1 m rms; projector at {b['projector_m']} m"),
            ("Receive window, ping interval", f"{b['window_m'][0]:g}-{b['window_m'][1]:g} m; {b['ping_interval_s']:.2f} s at the shortest; {b['samples_per_ping']:,} samples a channel"),
            ("Beam, range cell", f"{b['beamwidth_deg']:.1f} deg at broadside; {b['range_resolution_m']} m before windowing"),
            ("Ambiguity", f"no grating lobe within +/-{b['grating_lobe_free_steering_deg']:.0f} deg (the deployment file steers {b['bearings_deg']})"
                          + ("" if not b["elevation"] else f"; elevation repeats every {b['elevation']['repeat_spacing_deg']} deg, highest {b['elevation']['highest_repeat_db']} dB")),
            ("Elevation", "none (one row)" if not b["elevation"] else f"{b['elevation']['sigma_deg_at_20_db']} deg at 20 dB a row; {b['elevation']['deg_per_us']} deg per us between rows; {b['elevation']['deg_per_mm_out']} deg per mm a row stands out"),
            ("Clock agreement", f"{b['clock_tolerance_within_row_us']} us within a row" + ("" if b["clock_tolerance_between_rows_us"] is None else f"; {b['clock_tolerance_between_rows_us']} us between rows")),
            ("Data rate", f"{b['rate_board_bit_s'] / 1e6:.1f} Mbit/s a board, {b['rate_total_bit_s'] / 1e6:.1f} Mbit/s in all = {s['payload_bytes_per_s']:,} bytes/s of samples"),
            ("UDP stream", f"blocks of {s['block_samples']} samples, {s['block_bytes']:,} bytes, {s['blocks_per_s']:g} a second: {s['bytes_per_s_with_headers']:,} bytes/s with headers; socket buffer asked {s['socket_buffer_asked_bytes']:,} bytes"),
            ("Recording", f"{b['record_bytes_per_ping'] / 1e6:.0f} MB a ping; {b['record_bytes_per_hour_raw'] / 1e9:.1f} GB an hour raw, {b['record_bytes_per_hour_decimated'] / 1e9:.1f} decimated (shipped: {b['recording_mode']})"),
            ("Direct sound", f"{ds['level_db']:.0f} dB re 1 uPa at the nearest hydrophone ({ds['nearest_m']:.2f} m): {ds['over_full_scale_db']:+.0f} dB against full scale, {ds['over_full_scale_farthest_db']:+.0f} dB at the farthest; ends {ds['ends_s']:.3f} s; first usable range {ds['min_range_memoryless_m']:.0f} m"),
            ("Detector needs (one row's SNR)", f"{b['detector']['needs_one_row_snr_db']} dB for a detection probability of {b['detector']['at_probability']} on one ping, {b['detector']['needs_for_0p9_db']} dB for 0.9"),
            ("Noise-limited reach", f"{b['reach_ship_m'] / 1e3:.1f} km (ship, +25 dB) / {b['reach_small_m'] / 1e3:.1f} km (small, +5 dB)"),
            (f"At its design range, {b['design_range_m'] / 1e3:g} km", f"ship: margin {b['at_range'][far]['margin_ship_db']:+.1f} dB, Pd {b['at_range'][far]['pd_ship']:.2f}; small: margin {b['at_range'][far]['margin_small_db']:+.1f} dB, Pd {b['at_range'][far]['pd_small']:.2f}"),
        ]
        k = str(int(DESIGN_POINT_M))
        if k in b["at_range"]:
            a = b["at_range"][k]
            rows.append(("At 12 km", f"ship: one row's SNR {a['row_snr_ship_db']} dB, margin {a['margin_ship_db']:+.1f} dB, Pd {a['pd_ship']:.2f}; small: margin {a['margin_small_db']:+.1f} dB, Pd {a['pd_small']:.2f}"))
        out.append(f"#### {b['build']}\n\n" + table(("Quantity", "Value"), rows))
    return "\n\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSON file to write (default: docs/handoff/hardware_builds.json)")
    ap.add_argument("--markdown", action="store_true", help="print one table per build")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT / "python"))
    from gpusonar import buildspec

    builds = sorted((build_numbers(n) for n in buildspec.shipped()), key=lambda b: str(b["stage"]))
    doc = {"generated_by": "tools/handoff_hardware.py", "sources": ["configs/builds/*.yaml", "configs/sonars/*.yaml", "configs/deploy/*.yaml",
                                                                    "python/gpusonar/buildspec.py (describe_shipped, performance)",
                                                                    "python/gpusonar/sources/flank_stream.py (block sizes)"],
           "status": "arithmetic on configuration values; nothing here is measured on hardware", "builds": builds}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    if args.markdown:
        print(markdown(builds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
