#!/usr/bin/env python3
"""Parts list of a build, generated from its config files; and the table for the staged Popoto plan.

    tools/parts_list.py              # the five stages -> docs/design/popoto_parts.md and .json

Each row is a part the design needs, with the quantity per stage, THE REQUIREMENT THE DESIGN PLACES ON IT (computed
from the sonar and deployment files and from the study and benchmark files), the config key or study number that
fixes that requirement, and a status:

    chosen                  named in the owner's plan (docs/plan/GPU-FLS-Phased-Plan.pdf) or by the owner; what the
                            plan says about it is QUOTED, not paraphrased
    specified, not chosen   the design says what it must do; no part is named
    unconfirmed             needed, but the design cannot yet say what it must be

No part number, price, weight, power draw or specification is invented here: a figure that is not in the plan or
computed from the files is written as "to be supplied". `parts(d)` is a function of `buildspec.describe(...)`, so
tools/build_from_spec.py makes the same table for any build.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from gpusonar import buildspec  # noqa: E402

PLAN = "docs/plan/GPU-FLS-Phased-Plan.pdf"
# Quoted from the owner's plan (read for this table; page numbers are the PDF's own footers).
QUOTES = {
    "PMM6081": [("p. 13, 'What Popoto documents', Receive inputs (source: PMM6081 ICD)", "8 single-ended hydrophone inputs, 43 dB preamp, 37 mVpp full scale, 100 Hz high-pass"),
                ("p. 13, Timing (PMM6081 ICD)", "1PPS input, battery-backed clock"),
                ("p. 13, Software (PMM6081 spec)", "Debian Linux on a quad-core Arm; Python, C++ and MATLAB APIs; Ethernet"),
                ("p. 13, Raw data (Popoto docs)", "Baseband and passband waveform playback and recording; recordstart writes .wav files"),
                ("p. 13, Price (Popoto store)", "$7,825 development kit; $6,150 OEM board"),
                ("p. 14, to confirm with Popoto", "Whether the hydrophone inputs can be used at 6–10 kHz (the board is specified for 10–100 kHz)"),
                ("p. 14, to confirm with Popoto", "ADC sample rate, and whether all eight inputs record together, sample-aligned, in one file"),
                ("p. 14, batching boards", "Each board runs on its own sample clock, so the recordings drift apart and must be re-aligned in software before beamforming."),
                ("p. 14, batching boards", "When to stop: past two boards, the custom 32-channel ADC board in Phase 6 is cheaper and removes the problem with one shared clock.")],
    "AS-1": [("p. 12, Hardware design, item 1", "Two calibrated reference hydrophones (for example AS-1, −208 dBV re 1 µPa) check sensitivity."),
             ("p. 19, costs", "the AS-1 hydrophone (€412 ex VAT)")],
    "hydrophones": [("p. 12, Phase 0", "Popoto doesn't list hydrophones of its own, so \"Popoto hydrophones\" here means ordinary passive hydrophones wired to the board's inputs."),
                    ("p. 12, Hardware design, item 1 (the integrated array, not the Popoto boards)",
                     "32 low-cost piezo elements potted in polyurethane along a 3 m strip, each with a low-noise preamp and band-pass filter (about 2–20 kHz) at the element, driving differential pairs.")],
    "PMT-28": [("p. 13, Transmit (PMM6081 spec)", "190 dB re 1 µPa at 1 m with the PMT-28 transducer, 20–35 kHz"),
               ("p. 14", "The PMT-28 can't transmit there, so this needs a separate low-frequency speaker, and Popoto should confirm the sample rate and filtering.")],
    "projector": [("p. 12, Hardware design, item 4", "The FPGA plays the chirp through a DAC to a class-D power amplifier, matching transformer and projector. Stage 1 uses an underwater speaker; "
                   "stage 2 uses a free-flooded ring or flextensional projector with a 1–2 kW amplifier and a capacitor bank.")],
    "blanking": [("p. 12, Hardware design, item 5", "The projector is separate, so no transmit/receive switch is needed; preamp inputs are clamped and the software ignores samples during transmit.")],
    "sync": [("p. 14, batching boards", "Best method: wire one shared reference signal into one input on every board (leaving 7 hydrophones per board, 28 in all)."),
             ("p. 14", "Alternatives: use the direct transmit pulse, which every board hears, or the 1PPS time-stamps."),
             ("p. 14", "Accuracy needed: about 1.5 µs at 35 kHz, or 5 µs at 10 kHz, for clean beams.")],
    "compute": [("p. 9, AUV compute", "Develop on DGX Spark, and plan a dedicated port-and-benchmark step before choosing the vehicle computer."),
                ("p. 9, table", "Jetson AGX Orin: Linux sockets, 10GbE, CPU-bound; 15–75 W; Module 100 × 87 mm; $3,499 dev kit"),
                ("p. 9, table", "Jetson AGX Thor: Accelerated Ethernet capture (CoE); GPU-direct not documented; 40–130 W; Module 100 × 87 mm; $5,499 dev kit"),
                ("p. 9, table", "DGX Spark: RoCE, ConnectX-7; 240 W supply; 150 × 150 × 50 mm; $4,699")],
    "mounting": [("p. 12, Hardware design, item 6", "The array sits in a faired strip decoupled from the hull, away from thrusters; flow and machinery noise set the real noise floor.")],
}


def _mbit(v):
    return f"{v / 1e6:.1f} Mbit/s"


def _size(v):
    return f"{v / 1e9:.2f} GB" if v >= 1e9 else f"{v / 1e6:.0f} MB"


def noise_requirement(d: dict, wind: float = 5.0) -> dict:
    """The noise a hydrophone and its input may add: below the sea's own at the band centre (the link budget's ambient
    level), as a pressure spectrum level and as a voltage density at the board's input with the config's sensitivity."""
    from gpusonar.sim.noise import wenz_spectrum_db
    level = float(wenz_spectrum_db(np.array([d["centre_hz"]]), wind, 0.5)[0])
    calm = float(wenz_spectrum_db(np.array([d["centre_hz"]]), 0.5, 0.5)[0])
    return {"ambient_db": level, "calm_db": calm, "wind": wind, "input_nv_rthz": 1e9 * 10.0 ** ((level + d["sensitivity_db"]) / 20.0),
            "input_nv_rthz_calm": 1e9 * 10.0 ** ((calm + d["sensitivity_db"]) / 20.0)}


def timing_text(d: dict, timings: dict) -> dict:
    """Processing time for this build from the measurement files, or why there is none. Nothing is extrapolated."""
    rows = str(d["rows"])
    standard = d["per_row"][0] == 32 and d["rows"] in (1, 2, 3)
    out = {}
    cpu, full, gpu = timings.get("cpu"), timings.get("cpu_full_rate"), timings.get("gpu")
    if not standard:
        why = f"not measured for {d['rows']} x {d['per_row'][0]} (the measurements are of 1, 2 and 3 rows of 32)"
        return {"cpu": why, "cpu_full_rate": why, "gpu": why, "stage2": "pending" if not timings.get("stage2") else "see docs/design/popoto_stage2.json"}
    out["cpu"] = (f"{cpu['seconds'][rows]:.2f} s a ping on {cpu['threads']} shared CPU threads ({cpu['what']}; earlier row heights, an otherwise idle machine)" if cpu else "not measured")
    out["cpu_full_rate"] = (f"{full['seconds'][rows]:.1f} s a ping on {full['threads']} shared CPU threads ({full['what']}): NOT real time without decimating first" if full else "not measured")
    if gpu and rows in gpu["seconds"]:
        h = gpu["header"]
        out["gpu"] = (f"{gpu['seconds'][rows]:.3f} s a ping on an {h['gpus'][0]} against {gpu['host_seconds'][rows]:.3f} s on its host ({h['processor']}, {h['host_threads']} threads): {gpu['what']}; "
                      f"{gpu['samples']:,} samples a channel (scenario size, baseband), the row heights before the 1.5 m change. Full-rate raw pings were not run on a GPU")
    else:
        out["gpu"] = "not measured"
    s2 = timings.get("stage2")
    if s2:
        flat = []

        def walk(prefix, v):
            if isinstance(v, dict):
                for k, x in v.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), x)
            elif isinstance(v, (int, float)) and "tim" in prefix.lower() or (isinstance(v, (int, float)) and "second" in prefix.lower()):
                flat.append(f"{prefix} = {v:.3g}")

        walk("", s2)
        out["stage2"] = "docs/design/popoto_stage2.json: " + ("; ".join(flat[:12]) if flat else "present, no timing entries found")
    else:
        out["stage2"] = "pending (docs/design/popoto_stage2.json does not exist: the decimate-first stage is being built separately)"
    return out


def parts(d: dict, timings: dict | None = None, static: bool | None = None) -> list[dict]:
    """The rows for one build: {key, item, qty, status, requirement, fixed_by, plan (quotes), open}."""
    timings = buildspec.measured_timings() if timings is None else timings
    hw, plat = d["hardware"], d["platform"]
    static = bool(plat.get("static")) if static is None else static
    el, dr, cab = d["elevation"], d["direct"], d["cables"]
    f_lo, f_hi = d["band_hz"]
    nz = noise_requirement(d)
    tt = timing_text(d, timings)
    popoto = "PMM6081" in str((hw.get("boards") or {}).get("model") or "")
    board_model = (hw.get("boards") or {}).get("model")
    spare_inputs = sum(b["spare_inputs"] for b in d["boards"])
    sync_need = (f"boards of different rows agree to {el['clock_us_for_error']:.2f} µs for {el['clock_error_deg']:g}° of elevation ({el['deg_per_us']:.3f}° per µs, rows {el['baseline_m']:g} m apart); "
                 if el else "") + f"boards of one row agree to {d['alignment_tolerance_us']:.1f} µs for clean beams (a twentieth of a period at {f_hi / 1e3:g} kHz)"
    over = dr["over_full_scale_db"]
    direct = (f"the projector's direct sound at the nearest hydrophone ({dr['nearest_m']:.2f} m) is {dr['level_db']:.0f} dB re 1 µPa = {dr['volts']:.1f} V at the ADC, "
              + (f"**{over:.0f} dB over** full scale" if over > 0 else f"{-over:.0f} dB under full scale") + f" ({d['adc_full_scale_v']:.2f} V; the input's full scale is {d['input_full_scale_pressure_db']:.0f} dB re 1 µPa), "
              f"from {dr['starts_s'] * 1e3:.1f} ms to {dr['ends_s']:.3f} s after the trigger")
    rows = []

    def add(key, item, qty, status, requirement, fixed_by, plan=(), open_=""):
        rows.append({"key": key, "item": item, "qty": qty, "status": status, "requirement": requirement, "fixed_by": fixed_by,
                     "plan": [{"where": w, "quote": q} for w, q in plan], "open": open_})

    hyd = (hw.get("hydrophone") or {})
    add("hydrophones", "Array hydrophones" + (f" ({hyd['model']})" if hyd.get("model") else ""), d["channels"], "specified, not chosen" if not hyd.get("model") else "chosen",
        f"usable over {f_lo / 1e3:g}–{f_hi / 1e3:g} kHz; sensitivity as assumed in the files, {d['sensitivity_db']:g} dB re 1 V/µPa (another value changes the full-scale pressure and the noise figure below, "
        f"not the processing); own noise plus the input's below the sea's at {d['centre_hz'] / 1e3:g} kHz: {nz['ambient_db']:.0f} dB re 1 µPa²/Hz at {nz['wind']:g} m/s wind "
        f"({nz['input_nv_rthz']:.0f} nV/√Hz at the board input with that sensitivity; {nz['calm_db']:.0f} dB, {nz['input_nv_rthz_calm']:.1f} nV/√Hz, in a calm sea); matched in phase between rows "
        "(5° of mismatch is 0.16° of elevation at a 1.5 m baseline: `sensitivity.phase_mismatch_error_deg` of the study) or calibrated in water; omnidirectional in the vertical plane over the searched sector",
        "`receive.sensitivity_db`; link budget of DESIGN.md Sec. 5 (ambient noise, `gpusonar.sim.noise.wenz_spectrum_db`); `sensitivity` of popoto_elevation_study.json",
        QUOTES["hydrophones"] if popoto else (), "which hydrophone; its noise and band; whether the board's own input noise is below the figure above (the plan gives none)")
    add("reference_hydrophones", "Reference hydrophones (calibration)", 2 if popoto else "to be decided", "chosen" if popoto else "unconfirmed",
        "a calibrated sensitivity in the band, to check every channel in water and to be the source's monitor in the row-offset calibration (`calibrate.estimate_channel_errors(..., planar=False)`)",
        "DEPLOYMENT.md, first time in water with rows", QUOTES["AS-1"] if popoto else (), "the plan names the AS-1 \"for example\"; its calibration band is not quoted there")
    add("boards", f"Capture boards ({board_model or 'not chosen'})", d["n_boards"], "chosen" if popoto else ("specified, not chosen" if not board_model else "chosen"),
        f"{d['inputs_per_board']} inputs each ({d['channels']} hydrophones" + (f", {spare_inputs} inputs spare" if spare_inputs else "") + f"); samples {f_lo / 1e3:g}–{f_hi / 1e3:g} kHz: the files assume "
        f"{d['sample_rate_hz'] / 1e3:g} kHz and {d['adc_bits']} bits; all inputs of a board sampled together; {_mbit(d['rate_board_bit_s'])} a board over the network; {sync_need}; input range: {direct}",
        "`receive.board_size`, `receive.sample_rate`, `receive.adc_bits`, `receive.adc_full_scale`, `receive.gain_db`; `sensitivity.clock_us_for_deg` and `projector` of popoto_elevation_study.json",
        QUOTES["PMM6081"] if popoto else (),
        ("sample rate, word length and simultaneous sampling are assumptions; the band is below the 10–100 kHz the plan says the board is specified for; "
         "whether boards can be batched; the plan advises against more than two" if popoto else "see the spec's unconfirmed list"))
    add("input_protection", "Input protection, blanking and recovery", f"{d['channels']} inputs", "unconfirmed",
        direct + (". The inputs must survive it and recover before the first range of interest; the recovery time is not known and the simulator models neither the direct sound nor the recovery. "
                  f"Three ways out, none chosen: clamp and accept the recovery; move the projector (full scale at {dr['distance_for_full_scale_m']:.0f} m from the nearest hydrophone); "
                  f"or lower the source level while close (full scale at {dr['source_level_for_full_scale_db']:.0f} dB)" if over > 0 else ". Inside the inputs' range"),
        "`transmit.lever_arm`, `transmit.waveform.source_level_db`, `receive.adc_full_scale`; `projector` of popoto_elevation_study.json", QUOTES["blanking"] if popoto else (),
        "recovery time of the inputs after overload; whether the clamp the plan describes exists on these boards")
    add("projector", "Projector" + (f" ({hw['projector']['model']})" if (hw.get("projector") or {}).get("model") else ""), 1, "specified, not chosen" if not (hw.get("projector") or {}).get("model") else "chosen",
        f"{d['source_level_db']:g} dB re 1 µPa at 1 m over {f_lo / 1e3:g}–{f_hi / 1e3:g} kHz for a {d['pulse_s']:g} s sweep (a duty of {100 * d['pulse_s'] / d['ping_interval_s']:.1f}% at the shortest ping interval, "
        f"{d['ping_interval_s']:.1f} s); {10 ** ((d['source_level_db'] - 170.8) / 10):.0f} W acoustic if omnidirectional (10^((SL − 170.8)/10)); its place relative to the array known to a few centimetres "
        "(it is in every record and the beamformer uses the true path)",
        "`transmit.waveform` (source level from the 12 km link budget, DESIGN.md Sec. 5); `transmit.lever_arm`", (QUOTES["PMT-28"] + QUOTES["projector"]) if popoto else (),
        "which projector; whether one exists at this level in this band at a size the platform takes; its beam pattern; its position")
    add("amplifier", "Power amplifier and matching", 1, "specified, not chosen" if not (hw.get("projector") or {}).get("amplifier") else "chosen",
        f"drives the projector to {d['source_level_db']:g} dB over {f_lo / 1e3:g}–{f_hi / 1e3:g} kHz for {d['pulse_s']:g} s; electrical power = acoustic power / the projector's efficiency (not known until a projector is chosen); "
        "a trigger whose instant is in the board stream",
        "`transmit.waveform`", QUOTES["projector"] if popoto else (), "power, energy storage for the pulse, how the transmit instant reaches the boards")
    if d["sync_scheme"] == "shared_reference":
        ref = d["reference"] or {}
        level_in = (hw.get("sync") or {}).get("reference_level_v")
        add("sync_reference", "Synchronisation reference generator and distribution", f"1 generator, {d['n_boards']} lines", "specified, not chosen",
            (f"a burst {ref['f_start'] / 1e3:g}–{ref['f_stop'] / 1e3:g} kHz, {ref['duration'] * 1e3:g} ms, twice a ping, " if ref else "a burst above the acoustic band, twice a ping, ")
            + (f"{level_in * 1e3:g} mV peak at the ADC ({level_in / 10 ** (d['gain_db'] / 20) * 1e6:.0f} µV at the input), " if level_in else "")
            + f"summed into the first input of every board without loading the hydrophone on it; the lines' delays equal to a fraction of the tolerance: {sync_need}",
            "`receive.reference`, `chain.sync`; `sensitivity.clock_us_for_deg` of popoto_elevation_study.json", QUOTES["sync"] if popoto else (),
            "the summing circuit is hardware to design: the plan's method gives the reference an input of its own (7 hydrophones a board), the files sum it into a hydrophone input (8 a board); "
            "0.4–0.5 µs is reached in simulation, real boards are unconfirmed")
    add("cables", "Hydrophone cables", d["channels"], "specified, not chosen",
        f"{cab['shortest_m']:.1f} to {cab['longest_m']:.1f} m, {cab['mean_m']:.1f} m on average, {cab['total_m']:.0f} m in all, ASSUMING the boards sit at "
        f"[{cab['assumed_board_position_m'][0]:g}, {cab['assumed_board_position_m'][1]:g}, {cab['assumed_board_position_m'][2]:g}] m from the array's centre (behind the face) and each run goes along the face, "
        f"then up or down, with {cab['slack_m']:g} m of slack; screened, single-ended into the boards as the plan describes them; capacitance loads a passive hydrophone (loss = C_h / (C_h + C_cable): to be checked "
        "against the chosen hydrophone)",
        "hydrophone positions (`array`), `hardware.boards.position_m`, `hardware.boards.cable_slack_m`", (), "where the boards really sit; connectors; penetrators")
    add("network", "Network: boards to the compute unit", f"{d['n_boards']} board ports + 1", "specified, not chosen",
        f"{_mbit(d['rate_board_bit_s'])} from each board, {_mbit(d['rate_total_bit_s'])} in all while receiving; UDP blocks of at most 16 kB (the block format of `sources/flank_stream.py`); "
        "no loss tolerated inside a ping (a lost block discards the ping)",
        "`receive.sample_rate` × `receive.adc_bits` × channels; `source.port` of the deployment file", QUOTES["PMM6081"][2:3] if popoto else (),
        "whether the boards can stream raw samples over Ethernet at all (the plan documents recording to .wav files)")
    comp = hw.get("compute") or {}
    add("compute", "Compute unit" + (f" ({comp['unit']})" if comp.get("unit") else ""), 1, "specified, not chosen" if not comp.get("unit") else "chosen",
        f"one ping every {d['ping_interval_s']:.1f} s at the shortest. Option A, decimate first on a CPU: {tt['cpu']}; at the boards' full rate without decimation: {tt['cpu_full_rate']}; "
        f"with the decimate-first stage: {tt['stage2']}. Option B, a GPU: {tt['gpu']}",
        "docs/design/popoto_elevation_bench.json (`timing_scenario_size`, `timing_full_length`); docs/benchmarks/gpu_benchmark_all_l4.json (group `live`)", QUOTES["compute"] if popoto else (),
        "which computer; neither option has been timed at the boards' full rate on the hardware that would be used")
    add("storage", "Storage", 1, "specified, not chosen",
        f"{_size(d['record_bytes_per_ping'])} a ping as recorded ({d['samples_per_ping']:,} samples × {d['channels']} channels × 4 bytes, window {d['window_m'][0]:g}–{d['window_m'][1]:g} m), "
        f"{_size(d['record_bytes_per_hour'])} an hour at {d['pings_per_hour']:.0f} pings; listening continuously {_size(d['stream_bytes_per_hour_continuous'])} an hour",
        "`receive.window`, `receive.sample_rate`; `output.recording` of the deployment file", (), "capacity and write rate of the volume; whether to store decimated pings instead")
    nav = hw.get("navigation") or {}
    if static:
        add("navigation", "Navigation and attitude; a static frame: tilt and compass sensor", "0 or 1", "unconfirmed",
            "without one the frame is taken as plumb and `elevation.roll_sigma_deg` states how plumb; with one, the same datagrams as a vehicle's. Tilt error is elevation error one for one",
            "`nav.static` / `nav.port`, `chain.elevation.roll_sigma_deg`; `sensitivity.roll_depth_error_m` of popoto_elevation_study.json", (), "whether the frame carries a sensor; how its depth and heading are measured")
    else:
        add("navigation", "Navigation and attitude; a static frame: tilt and compass sensor", 1, "chosen" if nav.get("unit") else "unconfirmed",
            "position, velocity, roll, pitch, yaw, time-stamped against the boards' time, through the ping and until its last echo "
            f"({d['window_s'][1]:.1f} s); roll accuracy goes one for one into elevation" + (" (0.2° is 3.5 m of depth at 1 km)" if el else " (one row: attitude only steers the report's frame)")
            + (f". {nav['unit']}: accuracy, output rate and time-stamp accuracy are from its datasheet — to be supplied" if nav.get("unit") else ""),
            "`nav.port`; `chain.elevation.roll_sigma_deg`, `attitude_time`; `sensitivity.roll_depth_error_m` of popoto_elevation_study.json", (),
            "the unit's figures and message format; the adapter from its messages to the navigation datagram")
    add("sound_speed", "Sound speed at the array, and water depth", "from the platform", "unconfirmed",
        "local sound speed (it sets beamforming and the angle at the array) and an approximate water depth with its uncertainty (folding bottom bounces; bounding the search of two rows)"
        + (f"; two rows give one elevation only beyond {el['one_answer_beyond_m_per_m_of_water']:.0f} m of range per m of water depth (one bounce)" if el and el.get("one_answer_beyond_m_per_m_of_water") else ""),
        "`sound_speed`, `water_depth` in the navigation datagram or `chain.sound_speed`, `chain.elevation.water_depth`, `water_depth_sigma_m`", (), "which sensors; DVL altitude plus depth as the water-depth source")
    add("mounting", "Mounting: face or frame, fairing", 1, "unconfirmed",
        (f"rows in one plane: 1 mm of one row standing out is {el['deg_per_mm_out']:.3f}° of elevation; " if el else "")
        + f"hydrophone positions along a row to a small fraction of {d['wavelength_m'] * 100:.0f} cm (a wavelength), or calibrated; clear of the projector's near field and of machinery",
        "`array` (positions), `platform`; `sensitivity.out_offset_error_deg` of popoto_elevation_study.json", QUOTES["mounting"] if popoto else (), "the mechanical design is not started")
    add("power", "Power", "to be supplied", "unconfirmed",
        f"consumers: {d['n_boards']} boards, the reference generator, the amplifier (pulsed: {d['pulse_s']:g} s every {d['ping_interval_s']:.1f} s at most), the compute unit, the network. No draw is known for any of them",
        "—", (), "every figure")
    spare_board = d["inputs_per_board"]
    add("spares", "Spares and failure tolerance", f"proposed: 1 board, {spare_board} hydrophones", "specified, not chosen",
        f"a dead hydrophone is named in `chain.dead_channels` and left out; a dead board takes {spare_board} of a row's {d['per_row'][0]} and the estimate carries on"
        + ("; a dead row leaves " + ("two rows: the two-row behaviour (candidates, not one elevation)" if d["rows"] == 3 else "no elevation") if d["rows"] > 1 else "")
        + ". The chain does not detect a dead channel by itself. One spare board and one board's worth of hydrophones is a proposal, not a requirement derived from anything",
        "`chain.dead_channels`; tests/python/test_elevation.py (dead hydrophone, dead board, dead row)", (), "spares policy")
    return rows


def stage_columns() -> list[tuple[str, dict]]:
    from build_diagrams import STAGES
    return [(label, buildspec.describe_shipped(sonar, deploy)) for label, sonar, deploy, _ in STAGES]


def findings(d: dict) -> list[str]:
    """Where the owner's plan and the files disagree (Popoto builds)."""
    f_lo, f_hi = d["band_hz"]
    return [
        f"**Band.** The plan: \"the board is specified for 10–100 kHz\" and asks Popoto only about 6–10 kHz. The files use the inputs at {f_lo / 1e3:g}–{f_hi / 1e3:g} kHz, below both. "
        "The files do not flag the band as unconfirmed on the receive side (only `transmit`).",
        f"**Full scale.** The plan's \"37 mVpp full scale\" with the \"43 dB preamp\" is read in the files as 18.5 mV peak at the input, {d['adc_full_scale_v']:.2f} V at the ADC "
        f"(18.5 mV × 10^(43/20) = 2.613 V, written 2.61). With {d['sensitivity_db']:g} dB re 1 V/µPa that is a full-scale pressure of {d['input_full_scale_pressure_db']:.0f} dB re 1 µPa at the hydrophone. "
        "Consistent, on the reading that 37 mVpp is at the connector; the plan does not say where it is referred to.",
        "**Sample rate and word length.** The files assume 102.4 kHz and 24 bits. The plan gives neither for the PMM6081 (\"ADC sample rate\" is one of its questions for Popoto). Both are flagged `unconfirmed`.",
        "**Synchronisation wiring.** The plan's best method gives the reference \"one input on every board (leaving 7 hydrophones per board, 28 in all)\". The stage files sum the reference into the first "
        "hydrophone input above the acoustic band and keep 8 hydrophones a board (32 a row on 4 boards); `popoto_4x7` is the plan's wiring. Summing needs a circuit the plan does not describe; "
        "with 7 a board a row of 32 needs 5 boards.",
        "**How many boards.** The plan: \"When to stop: past two boards, the custom 32-channel ADC board in Phase 6 is cheaper and removes the problem with one shared clock.\" The stages use 4, 8 and 12 Popoto boards.",
        "**Synchronisation accuracy.** The plan asks \"about 1.5 µs at 35 kHz, or 5 µs at 10 kHz, for clean beams\"; stacked rows 1.5 m apart ask 0.87 µs between rows for 0.05° of elevation, tighter than either.",
        "**Hydrophones.** The plan's −208 dBV re 1 µPa is the AS-1 reference hydrophone's figure; the files use it as the sensitivity of every array hydrophone. The plan's integrated array has a preamp and "
        "band-pass filter at each element \"driving differential pairs\"; the PMM6081's inputs are \"single-ended\" with their own 43 dB preamp, and the files assume passive hydrophones straight into them.",
        "**Projector.** The plan's Popoto transmit chain (PMT-28, \"20–35 kHz\", \"190 dB\") cannot serve this band; the files assume 197 dB at 4–6 kHz from a projector nobody has chosen, "
        "against the plan's \"underwater speaker\" first and \"free-flooded ring or flextensional projector with a 1–2 kW amplifier\" second.",
        "**Streaming.** The files take the boards' samples as UDP blocks in real time. The plan documents recording to .wav files (\"recordstart writes .wav files\") and says the trial "
        "\"doesn't prove ... live streaming\". The `wav_watch` source covers the file route.",
    ]


def markdown(columns: list[tuple[str, dict]], timings: dict | None = None, heading: str = "##") -> list[str]:
    """The parts tables for one or more builds side by side (quantities per build; requirements with each build's numbers)."""
    timings = buildspec.measured_timings() if timings is None else timings
    tables = [parts(d, timings) for _, d in columns]
    keys = []
    for t in tables:
        keys += [r["key"] for r in t if r["key"] not in keys]
    by = [{r["key"]: r for r in t} for t in tables]
    L = [f"{heading} Quantities and status\n", "| Item | Status | " + " | ".join(label for label, _ in columns) + " |", "| --- | --- |" + " --- |" * len(columns)]
    for k in keys:
        first = next(b[k] for b in by if k in b)
        statuses = sorted({b[k]["status"] for b in by if k in b})
        L.append(f"| {first['item']} | {' / '.join(statuses)} | " + " | ".join(str(b[k]["qty"]) if k in b else "—" for b in by) + " |")
    L.append(f"\n{heading} Numbers that size the parts\n")
    L.append("| | " + " | ".join(label for label, _ in columns) + " |\n| --- |" + " --- |" * len(columns))

    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(fn(d) for _, d in columns) + " |")

    row("Hydrophones, boards", lambda d: f"{d['channels']}, {d['n_boards']}")
    row("Data rate: a board / all", lambda d: f"{_mbit(d['rate_board_bit_s'])} / {_mbit(d['rate_total_bit_s'])}")
    row("Receive window (stated in the sonar file)", lambda d: f"{d['window_m'][0]:g}–{d['window_m'][1]:g} m")
    row("Shortest ping interval", lambda d: f"{d['ping_interval_s']:.2f} s")
    row("Recording: a ping / an hour", lambda d: f"{_size(d['record_bytes_per_ping'])} / {_size(d['record_bytes_per_hour'])}")
    row("Cables: shortest–longest, total (boards assumed at the array's centre)", lambda d: f"{d['cables']['shortest_m']:.1f}–{d['cables']['longest_m']:.1f} m, {d['cables']['total_m']:.0f} m")
    row("Projector to the nearest hydrophone", lambda d: f"{d['direct']['nearest_m']:.2f} m")
    row("Direct sound there, against the inputs' full scale", lambda d: f"{d['direct']['level_db']:.0f} dB re 1 µPa: {d['direct']['over_full_scale_db']:+.0f} dB")
    row("Direct sound ends", lambda d: f"{d['direct']['ends_s']:.3f} s (blind inside {d['blind_range_m']:g} m)")
    row("Source level that just reaches full scale", lambda d: f"{d['direct']['source_level_for_full_scale_db']:.0f} dB")
    row("Clock agreement between rows (0.05° of elevation)", lambda d: f"{d['elevation']['clock_us_for_error']:.2f} µs" if d["elevation"] else "one row: n/a")
    row("Clock agreement within a row (clean beams)", lambda d: f"{d['alignment_tolerance_us']:.1f} µs")
    tts = [timing_text(d, timings) for _, d in columns]
    L.append("| Compute, option A (CPU, decimated pings of scenario size) | " + " | ".join(t["cpu"].split(" (")[0] for t in tts) + " |")
    L.append("| Compute, CPU at the boards' full rate, no decimation | " + " | ".join(t["cpu_full_rate"].split(" (")[0] for t in tts) + " |")
    L.append("| Compute, option A with the decimate-first stage | " + " | ".join("pending" if t["stage2"].startswith("pending") else "see popoto_stage2.json" for t in tts) + " |")
    L.append("| Compute, option B (GPU, same decimated pings) | " + " | ".join(t["gpu"].split(":")[0] for t in tts) + " |")
    cpu, gpu = timings.get("cpu"), timings.get("gpu")
    caveats = []
    if cpu:
        caveats.append(f"CPU figures: {cpu['what']}, {cpu['threads']} shared threads, measured on the face's sonar files with 1, 2 and 3 rows of 32: the other stages have the same channel counts "
                       f"and shorter windows, so these are upper figures for them. {cpu['note']}")
    if timings.get("cpu_full_rate"):
        caveats.append(f"Full rate: {timings['cpu_full_rate']['what']}.")
    if gpu:
        h = gpu["header"]
        caveats.append(f"GPU figures: {gpu['what']}; {h['gpus'][0]} on {h['processor']} ({h['host_threads']} threads), {h['date_utc']} UTC, {gpu['samples']:,} samples a channel (scenario size, baseband), "
                       "the tree before the row heights changed to 1.5 m; raw full-rate pings have not been run on a GPU.")
    caveats.append("Decimate-first: " + tts[0]["stage2"] + ".")
    L.append("\n" + " ".join(caveats) + "\n")
    L.append(f"{heading} What the design asks of each part, and what the plan says\n")
    last = by[-1]
    for k in keys:
        r = last.get(k) or next(b[k] for b in by if k in b)
        label = columns[-1][0] if k in last else next(lab for (lab, _), b in zip(columns, by) if k in b)
        L.append(f"**{r['item']}** — {r['status']}.")
        L.append(f"- Requirement ({label}): {r['requirement']}.")
        others = [(lab, b[k]) for (lab, _), b in zip(columns, by) if k in b and b[k]["requirement"] != r["requirement"]]
        if k in ("input_protection", "cables") and others:
            for lab, o in others:
                L.append(f"  - {lab}: {o['requirement'].split('. The inputs must')[0].split(', ASSUMING')[0]}.")
        L.append(f"- Fixed by: {r['fixed_by']}.")
        for q in r["plan"]:
            L.append(f"- The plan ({q['where']}): \"{q['quote']}\"")
        if r["open"]:
            L.append(f"- Open: {r['open']}.")
        L.append("")
    return L


def write_stage_table() -> Path:
    cols = stage_columns()
    timings = buildspec.measured_timings()
    L = ["# Parts list: the long-range array, stage by stage\n",
         "**Generated by `tools/parts_list.py` from the sonar and deployment files of the five stages, the study and benchmark files, and the owner's plan. Do not edit by hand.**\n",
         f"Status words: `chosen` (named in the owner's plan, `{PLAN}`, or by the owner; the plan's words are quoted), `specified, not chosen` (the design says what the part must do), "
         "`unconfirmed` (needed; the design cannot yet say what it must be). No part number, price, weight, power draw or specification is invented: what is not in the plan or computed from the files is \"to be supplied\". "
         "Prices appear only inside quotations from the plan.\n"]
    L += markdown(cols, timings)
    L.append("## Where the plan and the files disagree\n")
    L += [f"- {x}" for x in findings(cols[-1][1])]
    L.append("\n## Open questions\n")
    seen = []
    table = parts(cols[-1][1], timings) + [r for r in parts(cols[1][1], timings) if r["key"] == "navigation"]
    for r in table:
        if r["open"] and r["open"] not in seen:
            seen.append(r["open"])
            L.append(f"- {r['item']}: {r['open']}.")
    out = ROOT / "docs" / "design" / "popoto_parts.md"
    out.write_text("\n".join(L) + "\n")
    data = {label: [{k: v for k, v in r.items()} for r in parts(d, timings)] for label, d in cols}
    (ROOT / "docs" / "design" / "popoto_parts.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    print("wrote", write_stage_table())
