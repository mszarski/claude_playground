#!/usr/bin/env python3
"""Generated sections of docs/DESIGN.md for the long-range array's reference design (called by tools/design_report.py):

    staged_plan()        the builds in order, what each demonstrates, what carries over, bring-up, what the platform supplies
    row_modules()        the short and the long row of 32, side by side on the face, with the 12 km point for each
    projector_options()  the projector's direct sound: numbers first, then the options and a recommended combination
    depth_and_targets()  the surface image in the budget, own depth as a lever, target classes (tools/popoto_depth_planner.py)
    twelve_km()          the 12 km design point: what the system that detects there is, and what it can tell you
    parts_section()      the parts list (tools/parts_list.py)

Every number is computed here from the build specs and config files (gpusonar.buildspec), or read from
docs/design/popoto_elevation_study.json, popoto_elevation_bench.json, popoto_range_study.json, popoto_12km.json
(tools/popoto_12km_study.py), docs/benchmarks/gpu_benchmark_all_l4.json, and popoto_stage2.json when it exists.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from gpusonar import buildspec as bs  # noqa: E402
from gpusonar import elevation as el  # noqa: E402

DESIGN = ROOT / "docs" / "design"
PLAN = [  # (build, what it is for, what it demonstrates)
    ("popoto_testauv_1x32", "the chain on real boards and under way (the short row: the one exception)", "capture and synchronisation of four boards, the stream, recording, bearing, range, tracks with ownship's motion taken out, the report to the vehicle"),
    ("popoto_static_2x32", "the long row and stacked rows in the water for the first time", "the row module the face will carry, its beam and reach; elevation as a list of possible angles (two rows cannot tell which of several angles an echo came from), and one angle where water depth rules the others out; row calibration; eight boards synchronised between rows"),
    ("popoto_static_3x32", "the recommended row arrangement, still off the vehicle", "one elevation and a depth; a passing target tracked with its depth; listening with elevations"),
    ("popoto_face_1x32", "the face to start with: one long row", "the row on the vehicle with the vehicle's navigation and the projector in its place (what the direct sound does); the 12 km detection"),
    ("popoto_face_2x32", "elevation on the vehicle where water is shallow against the range", "attitude at each echo's arrival; the list of possible elevation angles and the water-depth bound that reduces it to one, under way"),
    ("popoto_face_3x32", "the recommended end state", "everything, at range"),
]


def _json(name):
    p = DESIGN / name
    return json.loads(p.read_text()) if p.exists() else None


def _km(v):
    return f"{v / 1e3:.1f}"


def build(name: str, **array):
    """(spec, derived numbers, expected performance) of a shipped build, optionally with array fields changed."""
    spec = bs.load(name)
    if array:
        spec = copy.deepcopy(spec)
        spec["array"].update(array)
    d = bs.derive(spec)
    return spec, d, bs.performance(d)


def map_snr(perf, target, r=12000.0):
    """SNR of the summed map at range r for `ship` or `small`, dB."""
    return perf["row_snr_db"][target][str(r)] + perf["rows_detection_gain_db"]


def bearing_sigma_deg(d, perf, target, r=12000.0):
    """Cramer-Rao bound on bearing at broadside for one row at the SNR of the map (the same bound as elevation's, along the row)."""
    n = d["per_row"][0]
    x = (np.arange(n) - (n - 1) / 2.0) * d["pitch_m"]
    snr = 10.0 ** (min(map_snr(perf, target, r), 40.0) / 10.0)
    return float(np.degrees(el.sin_elevation_sigma(x, snr / n, d["band_hz"][0], d["band_hz"][1], d["sound_speed"])))


def elevation_sigma_deg(d, perf, target, r=12000.0):
    n = d["per_row"][0]
    snr = perf["row_snr_db"][target][str(r)]
    return float(np.degrees(el.sin_elevation_sigma(np.repeat(d["elevation"]["heights_m"], n), 10.0 ** (min(snr, 40.0) / 10.0) / n, d["band_hz"][0], d["band_hz"][1], d["sound_speed"])))


def staged_plan() -> list:
    L = ["### The staged plan in one place\n"]
    L.append("Six builds, one software path, one spec file each (`configs/builds/`); the sonar and deployment files of each are generated from its spec\n"
             "(`tools/build_from_spec.py`; `REFERENCE_DESIGN.md` is the guide). The owner: \"Keep at 32, can start with a smaller array in the 7m zone. Probably makes sense to keep the one\n"
             "design for auv and static right? Except for elevation.\" and then: \"Yes treat auv special case and have static array the longer row.\" So: **32 hydrophones a row is fixed; the 7 m is a zone the\n"
             "rows stay inside, not a length to fill; ONE row design, the long module, serves the static array and the face; the test AUV's short row exists only because of that vehicle's span; and the face\n"
             "\"can start with a smaller array\" by carrying fewer rows of the same module: one, then two, then three.**\n")
    L.append("| Build | Rows × module | For | Beam | Reach, ship / small (km) | Elevation at 20 dB | Boards | Direct sound over full scale |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = {}
    for name, why, _ in PLAN:
        spec, d, perf = build(name)
        rows[name] = (spec, d, perf)
        e = perf.get("elevation")
        L.append(f"| `{name}` | {d['rows']} × `{spec['array'].get('module', '—')}` | {why} | {perf['beamwidth_deg']:.1f}° | {_km(perf['reach_ship_m'])} / {_km(perf['reach_small_m'])} | "
                 + (f"{e['sigma_deg']['20.0']:.2f}°" + (" (one of several possible angles: see below)" if d["rows"] == 2 else "") if e else "none") + f" | {d['n_boards']} | {d['direct']['over_full_scale_db']:+.0f} dB |")
    L.append("\n(Two rows: " + possible_angles(rows["popoto_face_2x32"][1]) + ".)")
    L.append("\n(Reach: noise-limited, the link budget of Sec. 5, " f"{rows['popoto_face_3x32'][1]['source_level_db']:g} dB source level in every file: **no projector is chosen for any build**. "
             f"The static builds use a {rows['popoto_static_3x32'][1]['pulse_s']:g} s pulse for short-range trials and the face {rows['popoto_face_3x32'][1]['pulse_s']:g} s: that, not the row, is the difference in their reach.)\n")
    L.append("**What each build demonstrates, and what it cannot.**\n")
    for name, _, shows in PLAN:
        L.append(f"- `{name}`: {shows}.")
    auv, face = rows["popoto_testauv_1x32"][1], rows["popoto_face_3x32"][1]
    L.append(f"\n**What carries over.** The boards and their count per row ({auv['boards_per_row'][0] if auv['boards_per_row'] else auv['n_boards']} of {auv['inputs_per_board']} inputs), the synchronisation scheme, the stream and record formats, the software and its\n"
             "settings, the report, and from the static array on the row module itself with its harness. What changes between builds is in the spec and nowhere else: the number of rows and their heights,\n"
             f"the platform (pose typed in or from navigation), the pulse ({auv['pulse_s']:g} s on the short-range builds, {face['pulse_s']:g} s on the face), the projector's place, the receive window\n"
             f"({auv['window_m'][0]:g}–{auv['window_m'][1]:g} m against {face['window_m'][0]:g}–{face['window_m'][1]:g} m: each sonar file now states its own).\n")
    L.append("**Bring-up, every build** (the generator writes the list for a build, with its numbers, into `BUILD.md`): simulated self-tests → the rehearsal sent as board blocks over UDP → boards on the bench\n"
             "(stream, channel map, synchronisation residual against the build's tolerance) → projector at low level in water (clipping and recovery of the nearest inputs) → a known target (range, bearing,\n"
             "SNR against the budget) → with rows: each row alone, row offsets from a source at known depths, depth against truth level and tilted → every measured value back into the spec.\n")
    L.append("**What the platform must supply.**\n\n| | Test AUV | Static frame | Face |\n| --- | --- | --- | --- |\n"
             "| Pose and attitude | navigation datagrams (unit not named) | a typed-in pose, assumed plumb to 1° (a placeholder), or a tilt and compass sensor | SPRINT-Nav (the owner); accuracy, rate and time stamp from its datasheet — to be supplied |\n"
             "| Sound speed at the array | in the datagram or typed | measured once, typed | in the datagram (the owner: the AUV measures it) |\n"
             "| Water depth | not needed for one row | typed (two rows need it to choose among their possible elevation angles) | in the datagram, to about 5 m (assumed) |\n"
             f"| Mounting | 3 m of hull (2–3 m by the owner's words) | a frame {rows['popoto_static_3x32'][1]['platform']['span_m']:g} m × {rows['popoto_static_3x32'][1]['platform']['height_m']:g} m (the long row plus margins: parameters) | 7 m of the 9 m × 2 m face, rows 1.5 m apart |\n"
             "| Power, network, storage | to be supplied: see the parts list | the same, down a cable or in a housing | the same |\n")
    L.append("**Expected accuracy, plainly.** Bearing and range are the row's: a beam of the width in the table and a range cell of "
             f"{rows['popoto_face_3x32'][2]['range_resolution_m']:.3f} m before windowing. Elevation is bought with the rows' separation and SNR. Depth is elevation times range, and beyond a few hundred metres\n"
             "it is limited by the sound-speed profile nobody will have, not by the array (table in *The 12 km design point*). In a shallow channel a bounce arriving inside the pulse's range sidelobes\n"
             "pulls the elevation by tenths of a degree, and **no reported sigma contains that**.\n")
    return L


def row_modules() -> list:
    S = _json("popoto_elevation_study.json")
    mods = bs.modules()
    L = ["### Row modules: one row design, and the test AUV's exception\n"]
    L.append("A row module is a named row: 32 hydrophones at one spacing, their harness, four boards. A build names its module and says how many are stacked (`array.module`, `array.rows`,\n"
             "`array.heights_m`). Two are shipped (`configs/builds/modules/`); the owner's decision: \"treat auv special case and have static array the longer row\".\n")
    L.append("| Module | Hydrophones, spacing | Length | Used by | Spacing in wavelengths at the band centre |\n| --- | --- | --- | --- | --- |")
    lam = 1500.0 / 5000.0
    for name, m in mods.items():
        a = m["array"]
        users = [n for n in bs.shipped() if (bs.load_yaml(bs.config_dir() / "builds" / f"{n}.yaml").get("array") or {}).get("module") == name]
        L.append(f"| `{name}` | {a['count']} at {a['pitch_m'] * 100:g} cm | {(a['count'] - 1) * a['pitch_m']:.2f} m | " + ", ".join(f"`{u}`" for u in users) + f" | {a['pitch_m'] / lam:.2f} |")
    L.append("\n**Why the test AUV's prototype reaches less: the short row against the long one**, each as one row with the face's projector and pulse (noise-limited; 13 dB at the map for detection), and the long row stacked:\n")
    L.append("| Row | Beam | Row's gain against sea noise | Map gain with the rows | Reach, ship / small (km) | SNR at 12 km, ship | Margin at 12 km | Source level for 13 dB at 12 km | or pulse at the shipped source level |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    out = {}
    cases = (("1 × short (the test AUV's row)", build("popoto_face_1x32", pitch_m=mods["testauv_short"]["array"]["pitch_m"])), ("1 × long", build("popoto_face_1x32")),
             ("2 × long", build("popoto_face_2x32")), ("3 × long", build("popoto_face_3x32")))
    for label, (spec, d, perf) in cases:
        snr = map_snr(perf, "ship")
        out[label] = (d, perf, snr)
        gap = max(bs.BUDGET_THRESHOLD_DB - snr, 0.0)
        L.append(f"| {label} | {perf['beamwidth_deg']:.1f}° | {perf['gain_isotropic_noise_db']:.1f} dB | {perf['gain_isotropic_noise_db'] + perf['rows_detection_gain_db']:.1f} dB | {_km(perf['reach_ship_m'])} / {_km(perf['reach_small_m'])} | "
                 f"{snr:.1f} dB | {snr - bs.BUDGET_THRESHOLD_DB:+.1f} dB | {d['source_level_db'] + gap:.0f} dB" + (" (as shipped)" if not gap else "") + f" | {d['pulse_s'] * 10.0 ** (gap / 10.0):.1f} s |")
    (ds, ps, snr_s), (dl, pl, snr_l) = out["1 × short (the test AUV's row)"], out["1 × long"]
    L.append(f"\n- **Closer than half a wavelength the hydrophones share the sea's noise.** 32 at {ds['pitch_m'] * 100:g} cm gain {ps['gain_isotropic_noise_db']:.1f} dB against sea noise, not the {ps['gain_uncorrelated_noise_db']:.1f} dB\n"
             f"  that 32 give at half a wavelength; against the boards' own noise they still gain {ps['gain_uncorrelated_noise_db']:.1f} dB. The beam is {ps['beamwidth_deg']:.1f}° against {pl['beamwidth_deg']:.1f}°, so it also takes in more reverberation.\n"
             f"- **The short row costs {snr_l - snr_s:.1f} dB and {ps['beamwidth_deg'] / pl['beamwidth_deg']:.1f} times the beam width.** At 12 km on the ship-sized target it has {snr_s - 13.0:+.1f} dB of margin against {snr_l - 13.0:+.1f} dB for the long row"
             + (f"; where the margin is negative the table says what source level or pulse would restore it" if snr_s < bs.BUDGET_THRESHOLD_DB else "") + ".\n"
             "- **Static and face builds share everything in the module**: expected beam, gain and reach per row, the four-board harness, the calibration procedure. They differ in the platform: motion, the\n"
             "  attitude source, the projector's place, cabling and where the compute sits.\n"
             "- On the test AUV itself the prototype also uses a 0.1 s pulse (10 dB less processing gain) for short-range trials; its own reach is in the plan's table above.\n")
    if S:
        ts = S["trade_space"]["fixed_channels"]
        L.append("Kept as trade-space rows only, not as builds (the owner: \"Keep at 32\"): " + "; ".join(
            f"{k}: beam {v['beamwidth_deg']:.1f}°, map gain {v['gain_isotropic_noise_db'] + v['rows_detection_gain_db']:.1f} dB, reach {_km(v['reach_ship_m'])} / {_km(v['reach_small_m'])} km"
            for k, v in ts.items() if v["hydrophones_per_row"] != 32) + " (`trade_space.fixed_channels` of the study).\n")
    return L


def projector_options() -> list:
    from gpusonar import live
    B = _json("popoto_elevation_bench.json")
    L = ["### The projector's direct sound: \"Just blank?\"\n"]
    L.append("The owner: \"What can we do about the projector? Just blank?\" Blanking, meaning the chain ignores samples taken during the transmit, is necessary and is not enough. Numbers first, all from the\n"
             "build files (`gpusonar.buildspec.describe`: source level − 20 log₁₀(distance to the nearest hydrophone), through the files' sensitivity and gain):\n")
    L.append("| Build | Projector to nearest hydrophone | Direct level there | At the ADC | Over full scale | Direct sound ends | First sync burst after that | Distance for full scale | Source level for full scale |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    ds = {}
    for name in ("popoto_testauv_1x32", "popoto_static_2x32", "popoto_static_3x32", "popoto_face_1x32", "popoto_face_2x32", "popoto_face_3x32"):
        _, d, _ = build(name)
        ds[name] = d
        dr = d["direct"]
        gap = (d["reference_times_s"][0] - dr["ends_s"]) * 1e3 if d.get("reference_times_s") else None
        L.append(f"| `{name}` | {dr['nearest_m']:.2f} m | {dr['level_db']:.0f} dB re 1 µPa | {dr['volts']:.0f} V | **{dr['over_full_scale_db']:+.0f} dB** | {dr['ends_s']:.3f} s | "
                 + (f"{gap:.1f} ms" if gap is not None else "—") + f" | {dr['distance_for_full_scale_m']:.0f} m | {dr['source_level_for_full_scale_db']:.0f} dB |")
    d = ds["popoto_face_3x32"]
    dr = d["direct"]
    L.append(f"\nThe files' gain and full scale ARE the plan's figures for the capture board (\"43 dB preamp, 37 mVpp full scale\": {d['gain_db']:g} dB and 18.5 mV peak at the input, {d['adc_full_scale_v']:.2f} V after the gain), so the computation with the plan's\n"
             f"figures is the same computation: full scale is {d['input_full_scale_pressure_db']:.0f} dB re 1 µPa at the hydrophone. What is NOT the plan's is the hydrophone sensitivity ({d['sensitivity_db']:g} dB re 1 V/µPa, a placeholder): a hydrophone\n"
             "10 dB less sensitive would overload 10 dB less and need 10 dB more signal everywhere else. Every projector position in the files is a placeholder.\n")
    L.append(f"**What blanking does and does not do.** With a {d['pulse_s']:g} s pulse nothing inside {d['blind_range_m']:g} m is usable whatever the inputs do, because echoes from there arrive during the transmit. So the samples lost to\n"
             "blanking cost nothing by themselves. What matters is (a) that the inputs are not damaged; (b) how long after the pulse ends they take to recover, because that time adds to the blind range\n"
             f"({0.5 * d['sound_speed'] * 0.01:g} m per 10 ms); (c) that the synchronisation bursts and any calibration signal do not fall inside the transmit or the recovery. On (c) the shipped timing is a finding: the first burst is\n"
             f"{(d['reference_times_s'][0] - dr['ends_s']) * 1e3:.1f} ms after the direct sound ends at the farthest hydrophone on the face. The build validator now warns of it.\n")
    blank_key = next((k for k in live.DEFAULTS if "blank" in k), None)
    from gpusonar.sim import receiver
    overload_model = hasattr(receiver, "overload_config")
    L.append(f"**Every hydrophone is overloaded, not only the nearest.** On the face (three rows) {dr['hydrophones_overloaded']} of {d['channels']} are, from {dr['over_full_scale_farthest_db']:+.1f} dB at the farthest ({dr['farthest_m']:.2f} m) to "
             f"{dr['over_full_scale_db']:+.1f} dB at the nearest; the direct sound falls to full scale only {dr['distance_for_full_scale_m']:.1f} m from the projector, so **no position on a {d['platform']['span_m']:g} m face avoids it**: at least\n"
             f"{dr['gain_reduction_needed_db']:.1f} dB of gain reduction during the transmit is needed, or inputs that are built to be overloaded. The first usable range is {dr['min_range_memoryless_m']:.0f} m with an input that has no memory "
             f"({ds['popoto_testauv_1x32']['direct']['min_range_memoryless_m']:.0f} m with the 0.1 s pulse), plus {dr['range_per_ms_of_recovery_m']:.2f} m per millisecond of recovery. These are peak levels (add 3 dB if a source level is an rms figure) and the water path only: "
             "hull-borne and electrical coupling from the amplifier are not modelled and could exceed it.\n")
    DS = _json("direct_sound_study.json")
    if DS and "popoto_long_range_3row" in DS:
        sm = DS["popoto_long_range_3row"]["summary"]
        L.append(f"The simulator now models this (`transmit.direct_path`, `receive.overload`; `tools/direct_sound_study.py`): its per-hydrophone figures for the face are {sm['overload_db_farthest']:+.1f} to {sm['overload_db_nearest']:+.1f} dB over full scale, "
                 f"{sm['hydrophones_overloaded']} of {sm['hydrophones']} overloaded, full scale at {sm['distance_for_no_overload_m']:.1f} m (`design/direct_sound_study.json`; the same closed form as here, checked against the simulated samples). "
                 "Its recovery parameter sets are illustrative: no board has been measured. Blanking alone suffices only if the front end has no memory; a gain switch of more than the overload removes the problem at its source.\n")
    L.append("| Option | What it buys | What it costs or needs | Known? |\n| --- | --- | --- | --- |")
    step = [(k, 20.0 * np.log10(k)) for k in (2.0, 4.0, 8.0)]
    L.append("| Ignore window in the chain (blanking) | discards the samples of the transmit and of the recovery; " + (f"`chain.{blank_key}`" if blank_key else "the chain today starts at the receive window "
             f"(`receive.window`, from the blind range); a configurable blanking window is being added to the chain separately") + " | nothing; it protects nothing | yes |")
    L.append(f"| Input clamp or limiter (protection) | survival at {dr['over_full_scale_db']:+.0f} dB over full scale ({dr['volts']:.0f} V equivalent at the ADC, {dr['volts'] / 10 ** (d['gain_db'] / 20):.2f} V at the input) | a recovery time after the clamp releases: **unknown** | "
             "the plan says of its own integrated design \"preamp inputs are clamped\"; for the PMM6081 it gives no damage threshold and no clamp: unknown |")
    L.append(f"| Switchable or time-varying input gain | {dr['over_full_scale_db']:.0f} dB less gain during the transmit removes the overload; a ramp afterwards is a TVG (the record format carries one) | the board must offer it | "
             "unknown: the plan lists \"other gain options\" among its questions for Popoto |")
    L.append("| Move the projector | 6 dB per doubling of distance: " + ", ".join(f"×{k:g} ({dr['nearest_m'] * k:.1f} m): {dr['over_full_scale_db'] - g:+.0f} dB" for k, g in step)
             + f"; full scale at {dr['distance_for_full_scale_m']:.0f} m | room on the platform; beyond the face it means a pole, or below or behind the hull; the direct sound then ends later "
             f"({1e3 / d['sound_speed']:.2f} ms per metre) | geometry: yes |")
    L.append(f"| A projector with a null towards the array, or a baffle | must deliver {dr['over_full_scale_db']:.0f} dB towards the nearest hydrophone to reach full scale, more for margin | a directional source or a structure at 4–6 kHz "
             f"(a wavelength is {d['wavelength_m'] * 100:.0f} cm); no product is assumed | unknown |")
    L.append("| Lower source level, longer pulse, same energy | each 3 dB less level is twice the pulse: " + ", ".join(
        f"−{x:g} dB: {d['pulse_s'] * 10 ** (x / 10):.0f} s, blind inside {0.5 * d['sound_speed'] * d['pulse_s'] * 10 ** (x / 10) / 1e3:.1f} km" for x in (3.0, 6.0, 10.0))
             + f". The 12 km budget is unchanged (it depends on energy); full scale would need −{dr['over_full_scale_db']:.0f} dB, a {d['pulse_s'] * 10 ** (dr['over_full_scale_db'] / 10):.0f} s pulse: not a way out by itself | blind range and ping interval grow with the pulse | yes |")
    L.append("| Schedule the sync bursts clear of the transmit | the bursts survive | `sync.reference.lead_s` in the spec (after the recovery time), or open the window later | yes, once the recovery time is |")
    f2 = (B or {}).get("scenarios", {}).get("face_2row")
    L.append("\n**Position and the multipath depth bias.** The projector's place also moves the bias a shallow channel puts on depth: with the projector beside the rows in the end margin the two-row check at 3 km reads "
             + (f"{min(f2['depth_m']):.0f}–{max(f2['depth_m']):.0f} m for a target at {f2['true_depth_m']:.0f} m" if f2 else "low")
             + " (`popoto_elevation_bench.json`); an earlier run with it 1.5 m below the array's centre read closer to the truth. Below the array was better on that count; it is one geometry, not a rule.\n")
    L.append("**Recommended combination.** (1) Protection first: a clamp or limiter at every input, or confirmation from the board's documentation that the inputs survive the level in the table. (2) The projector as far\n"
             "from the nearest hydrophone as the platform allows; every doubling is 6 dB less to recover from. (3) An ignore window in the chain from the trigger to the end of the direct sound plus the measured\n"
             "recovery. (4) The first sync burst after that window. (5) Lower gain during the transmit if the board offers it. Not recommended as the fix: a longer, weaker pulse.\n")
    L.append("**To measure on the bench before hardware is frozen.** Recovery time of an input against overload level (0, +10, +20, +30 dB over full scale, a burst as long as the pulse); the damage threshold from the\n"
             "board's documentation or its maker; whether the reference burst summed into an input survives next to an overloaded hydrophone signal. The spec carries the results: `projector.blanking_after_s`,\n"
             "`boards.overload_damage_vpp`, `boards.gain_switchable` (all unknown today). "
             + ("Also on the bench: coupling from the amplifier into the inputs through the hull and the wiring, with the projector out of the water. "
                "The simulator's receiver has an overload and recovery model (`receive.overload`, with `transmit.direct_path`): a measured recovery time can be tried on the short-range scenarios "
                "(`popoto_testauv_direct_overload*`) before hardware is frozen.\n" if overload_model else
                "An input-overload and recovery model for the simulator's receiver (`receive.overload` in a sonar file) is being added separately; it is the way to try a recovery time against the short-range scenarios. "
                "It is not in this tree, so nothing here simulates the direct sound.\n"))
    return L


def twelve_km() -> list:
    S, B, R, K = (_json(n) for n in ("popoto_elevation_study.json", "popoto_elevation_bench.json", "popoto_range_study.json", "popoto_12km.json"))
    T = bs.measured_timings()
    from gpusonar.sim.noise import wenz_spectrum_db
    r = 12000.0
    options = [("1 × 32 long", build("popoto_face_1x32")), ("2 × 32 long", build("popoto_face_2x32")), ("3 × 32 long", build("popoto_face_3x32")),
               ("1 × 32 short (the test AUV's row, for comparison)", build("popoto_face_1x32", pitch_m=0.09)),
               ("1 × 46 (fills 7 m)", build("popoto_face_2x32", rows=1, heights_m=None, count=46, module=None)), ("1 × 64 in 7 m", build("popoto_face_2x32", rows=1, heights_m=None, count=64, pitch_m=0.109, module=None)),
               ("2 × 48 in 7 m", build("popoto_face_2x32", count=48, pitch_m=0.146, module=None))]
    main = {k: v for k, v in options[:3]}
    _, d3, p3 = main["3 × 32 long"]
    _, d2, p2 = main["2 × 32 long"]
    _, d1, p1 = main["1 × 32 long"]
    f_lo, f_hi = d3["band_hz"]
    c = d3["sound_speed"]
    dep = d3["deploy"]["chain"]
    roll = dep["elevation"]["roll_sigma_deg"]
    grad = dep["elevation"]["gradient_sigma_per_s"]
    sync_us = B["scenarios"]["rehearsal"]["worst_residual_us"] if B else None
    L = ["### The 12 km design point\n"]
    L.append("The owner: \"What does it look like for popoto at 12km though. Need that design\". This section is self-contained: what the system that detects at 12 km is, and what it can and cannot tell you there.\n"
             f"The targets are the two the study uses: a **ship-sized target of +25 dB** and a **small target of +5 dB**. Everything is for the full face ({d3['band_hz'][0] / 1e3:g}–{d3['band_hz'][1] / 1e3:g} kHz, a {d3['pulse_s']:g} s sweep, "
             f"{d3['source_level_db']:g} dB re 1 µPa at 1 m, rows of the long module, outer rows {d3['elevation']['baseline_m']:g} m apart).\n")
    # summary
    L.append("**At 12 km you get** (noise-limited budget; ship-sized target; one sigma):\n\n| Rows | Detection, ship (+25 dB) | Detection, small (+5 dB) | Bearing | Elevation | Depth with a measured profile | Depth without one |\n| --- | --- | --- | --- | --- | --- | --- |")
    for label, (_, d, perf) in main.items():
        snr_s, snr_m = map_snr(perf, "ship"), map_snr(perf, "small")
        b = bearing_sigma_deg(d, perf, "ship")
        if d["elevation"]:
            e = elevation_sigma_deg(d, perf, "ship")
            clock = d["elevation"]["deg_per_us"] * (sync_us or 0.0)
            known = float(np.sqrt((r * np.radians(e)) ** 2 + (r * np.radians(roll)) ** 2 + (r * np.radians(clock)) ** 2))
            e_txt = f"±{e:.2f}° from noise" + (", as a list of possible angles " + f"{d['elevation']['repeat_spacing_deg']:.1f}° apart unless water depth rules the others out (bearing, range and the track do not depend on it)" if d["rows"] == 2 else ", one answer")
            d_txt, n_txt = f"±{known:.0f} m (noise, roll to {roll:g}°, clocks to {sync_us:.2f} µs)", f"not in metres: ±{r * r * grad / (2 * c):.0f} m from a gradient unknown by {grad:g} /s"
        else:
            e_txt, d_txt, n_txt = "none", "none", "none"
        L.append(f"| {label.split(' long')[0]} | yes: {snr_s:.1f} dB, {snr_s - bs.BUDGET_THRESHOLD_DB:+.1f} dB of margin | no: {snr_m:.1f} dB, {bs.BUDGET_THRESHOLD_DB - snr_m:.0f} dB short | ±{b:.2f}° = ±{r * np.radians(b):.0f} m across "
                 f"(beam {perf['beamwidth_deg']:.1f}°, {r * np.radians(perf['beamwidth_deg']):.0f} m) | {e_txt} | {d_txt} | {n_txt} |")
    L.append("\nFor that to hold: a projector that delivers the source level in this band (none is chosen); hydrophones and inputs whose own noise is below the sea's (parts list); the inputs recovered from the direct sound\n"
             f"well inside the {d3['blind_range_m']:g} m blind range; noise, not reverberation, limiting at 12 km (below); for elevation, boards of different rows synchronised to {d3['elevation']['clock_us_for_error']:.2f} µs and the roll known as stated;\n"
             "for depth in metres, a sound-speed profile along the path. Spreading at least as kind as spherical (in shallow water it is kinder: below).\n")
    # link budget
    noise = p3["noise_spectrum_db"]
    L.append(f"**Link budget at 12 km.** {p3['assumptions']}; noise spectrum level {noise:.1f} dB re 1 µPa²/Hz at {d3['centre_hz'] / 1e3:g} kHz. Blind inside {d3['blind_range_m']:g} m (the pulse); a ping every {d3['ping_interval_s']:.1f} s "
             f"for the {d3['window_m'][0]:g}–{d3['window_m'][1]:g} m window ({60.0 / d3['ping_interval_s']:.1f} a minute).\n")
    L.append("| Face option | Beam | Row gain (sea noise) | Map gain | SNR at 12 km, ship | margin | SNR at 12 km, small | short by | Reached at: ship / small (km) |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for label, (_, d, perf) in options:
        s_, m_ = map_snr(perf, "ship"), map_snr(perf, "small")
        L.append(f"| {label} | {perf['beamwidth_deg']:.1f}° | {perf['gain_isotropic_noise_db']:.1f} dB | {perf['gain_isotropic_noise_db'] + perf['rows_detection_gain_db']:.1f} dB | {s_:.1f} dB | {s_ - 13.0:+.1f} dB | {m_:.1f} dB | "
                 f"{bs.BUDGET_THRESHOLD_DB - m_:.1f} dB | {_km(perf['reach_ship_m'])} / {_km(perf['reach_small_m'])} |")
    gap = bs.BUDGET_THRESHOLD_DB - map_snr(p3, "small")
    calm = float(wenz_spectrum_db(np.array([d3["centre_hz"]]), 0.5, 0.5)[0])
    g = {k: bs.rows_detection_gain_db(k) for k in (2, 3, 4)}
    fill = options[4][1][2]["gain_isotropic_noise_db"] - p1["gain_isotropic_noise_db"]
    slope = (p3["row_snr_db"]["ship"]["6000.0"] - p3["row_snr_db"]["ship"]["12000.0"]) / 6.0
    L.append(f"\n**What would close the {gap:.0f} dB for the small target at 12 km** (three rows; each alone):\n\n| Lever | To close it | What one step gives |\n| --- | --- | --- |\n"
             f"| Source level | {d3['source_level_db'] + gap:.0f} dB re 1 µPa at 1 m | dB for dB |\n"
             f"| Pulse length (same level) | {d3['pulse_s'] * 10 ** (gap / 10):.0f} s, blind inside {0.5 * c * d3['pulse_s'] * 10 ** (gap / 10) / 1e3:.0f} km | 3 dB per doubling; the blind range doubles with it |\n"
             f"| Hydrophones in a row | {32 * 10 ** (gap / 10):.0f} at half a wavelength, {32 * 10 ** (gap / 10) * 0.15:.0f} m long | 3 dB per doubling of a filled row; the 7 m zone holds 46: +{fill:.1f} dB (and the owner keeps 32) |\n"
             f"| Rows | not reachable | 2 rows +{g[2]:.1f} dB, 3 rows +{g[3]:.1f} dB, 4 rows +{g[4]:.1f} dB over one |\n"
             f"| Lower noise | {noise - gap:.0f} dB re 1 µPa²/Hz | a calm sea is {calm:.0f} dB: {noise - calm:.0f} dB; the platform's own noise is not in the budget at all |\n"
             f"| Come closer | {_km(p3['reach_small_m'])} km | about {slope:.1f} dB per km between 6 and 12 km |\n")
    L.append(f"The small target at 12 km is not a matter of one lever: it is reached at {_km(p3['reach_small_m'])} km. **These are noise-limited figures with spherical spreading, the cautious case for loss.**")
    if R and R.get("bellhop"):
        b5 = min(R["bellhop"], key=lambda x: abs(x["frequency_hz"] - d3["centre_hz"]))
        L.append(f"In shallow water sound is ducted: Bellhop for 100 m of isovelocity water over sand gives {b5['spherical_one_way_loss_db'] - b5['one_way_loss_db']:.0f} dB less loss each way at {b5['frequency_hz'] / 1e3:g} kHz "
                 f"({b5['one_way_loss_db']:.1f} against {b5['spherical_one_way_loss_db']:.1f} dB; `popoto_range_study.json`), which more than covers the small target's shortfall in that water, and at those levels reverberation, not noise, decides.\n")
    # reverberation, through the chain
    if K:
        L.append("**Through the chain, with reverberation** (`tools/popoto_12km_study.py`: the ship of `popoto_12km`, five points of +18 dB crossing at 6 m/s at 12 km, six pings; C++ chain with each row count's deployment file; SNR is over the local background the detector measures):\n")
        L.append("| Water | Rows | Ship detected on | Median SNR | Track confirmed at ping | Track on the ship | Later arrivals tracked / classed multipath | Other tracks |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
        for label, by in K["cases"].items():
            for rows_, v in by.items():
                if v.get("error"):
                    L.append(f"| {label} | {rows_} | failed: {v['error'][:80]} | | | | | |")
                    continue
                on = ", ".join(f"{t['kind']} at {t['speed_m_s']:.1f} m/s" for t in v["track_on_ship"]) or "none"
                L.append(f"| {label} | {rows_} | {v['pings_detected']} of {v['pings']} pings | " + (f"{v['median_snr_db']:.1f} dB" if v["median_snr_db"] is not None else "—")
                         + f" | {v['track_confirmed_at_ping'] or '—'} | {on} | {v['later_arrivals']['tracks']} / {v['later_arrivals']['classified_multipath']} | " + ", ".join(f"{n} {k}" for k, n in v["track_kinds"].items()) + " |")
        iso = K["cases"].get("popoto_12km: 100 m isovelocity over sand, image-source paths", {})
        quiet = K["cases"].get("the same without reverberation (noise only)", {})
        cmp_ = [(rows_, iso[rows_].get("median_snr_db"), quiet[rows_].get("median_snr_db")) for rows_ in iso if rows_ in quiet and not iso[rows_].get("error") and not quiet[rows_].get("error")]
        if cmp_:
            L.append("\nWhere reverberation rather than noise decides: in the image-source case the ship's SNR with and without the scenario's bottom and surface reverberation is "
                     + "; ".join(f"{rows_} row{'s' if rows_ != '1' else ''}: " + (f"{a:.1f} dB" if a is not None else "not detected") + " against " + (f"{b:.1f} dB" if b is not None else "not detected") for rows_, a, b in cmp_)
                     + ". The image-source paths spread spherically with two bounces and carry far less energy to 12 km than Bellhop's ducted field, so they sit near the noise-limited budget; the Bellhop rows are the shallow-water case."
                     + ("" if K.get("bellhop") else " **Bellhop was not available for this run: the environment rows are missing.**") + "\n")
        sc12 = bs.load_yaml(bs.config_dir() / "scenarios" / "popoto_12km.yaml")
        z_a, z_t = float(sc12["platform"]["position"][2]), float(sc12["scene"][0]["positions"][0][2])
        lloyd = 20.0 * np.log10(2.0 * np.sin(2.0 * np.pi * z_a * z_t / (d3["wavelength_m"] * r)))
        L.append(f"**A target near the surface is the hard case for straight rays.** The scenario's ship is at {z_t:g} m and the array at {z_a:g} m: at 12 km the direct path and the surface-reflected one (reflection −1) differ by "
                 f"{2e3 * z_a * z_t / r:.1f} mm, a small fraction of the {d3['wavelength_m'] * 100:.0f} cm wavelength, and nearly cancel: 20 log₁₀(2 sin(2π z₁ z₂ / λR)) = {lloyd:.0f} dB on each leg against the budget's free-field loss, before the bottom's bounces put energy back. "
                 "The noise-limited budget above does not contain this (it is the cautious case for spreading, not for the surface image); the image-source rows of the table do, which is why the ship is detected on few or no pings there, "
                 "and marginally enough that the projector's place on the face decides it. Bellhop's field for the same water holds the ship on every ping. Which is right for a given day is the environment's to say: "
                 "**the 12 km detection of a shallow target rests on the water ducting sound, or on the array's depth, not on the array's size.** The surface term is now in the planner's budget and own depth is planned "
                 "(*Surface interference, own depth, and target classes* above): in non-ducting water the ship is lost with the array at this depth and held with it deep.\n")
        if K.get("not_run"):
            L.append("Not run: " + "; ".join(K["not_run"]) + ".\n")
        L.append("The environments differ by far more than the row count does: the row count buys two to four decibels, the water tens. A downward-refracting profile over a soft bottom can leave nothing at 12 km whatever the array.\n")
    else:
        L.append("\n**Through the chain, with reverberation: pending** (`tools/popoto_12km_study.py` writes `docs/design/popoto_12km.json`; it has not been run in this tree).\n")
    # bearing
    b3 = bearing_sigma_deg(d3, p3, "ship")
    cone = S["cone_bias_deg"] if S else {}
    L.append(f"**Bearing at 12 km.** The beam is {p3['beamwidth_deg']:.1f}° wide: {r * np.radians(p3['beamwidth_deg']):.0f} m across at 12 km, so two targets closer than that in bearing and in the same range cell are one. A single target's bearing is good to\n"
             f"±{b3:.2f}° (±{r * np.radians(b3):.0f} m across) at the {map_snr(p3, 'ship'):.0f} dB available on the ship (the bound for a row of this length, at broadside; steered to 45° it is 1.4 times that). The cone-angle correction is\n"
             f"negligible there: an arrival within {np.degrees(np.arctan2(100.0, r)):.1f}° of level (100 m of water) moves the bearing by less than " + (f"{cone['2.0']['45.0']:.2f}° at 45° off broadside" if cone else "a hundredth of a degree") + ".\n")
    # elevation and depth
    L.append("**Elevation and depth at 12 km** (rows 1.5 m apart; the SNR is the one the budget gives at 12 km, not a nominal 20 dB):\n")
    L.append("| | Ship (+25 dB) | Small (+5 dB) |\n| --- | --- | --- |")
    snr_row = {t: p3["row_snr_db"][t][str(r)] for t in ("ship", "small")}
    L.append(f"| SNR of one row at 12 km | {snr_row['ship']:.1f} dB | {snr_row['small']:.1f} dB: not detected |")
    e2, e3 = elevation_sigma_deg(d2, p2, "ship"), elevation_sigma_deg(d3, p3, "ship")
    L.append(f"| Elevation sigma from noise, 2 rows / 3 rows | {e2:.2f}° / {e3:.2f}° | — |")
    L.append(f"| Depth from noise | {r * np.radians(e2):.0f} m / {r * np.radians(e3):.0f} m | — |")
    if S:
        se = S["sensitivity"]
        L.append("| Depth from roll known to " + " / ".join(f"{k}°" for k in se["roll_depth_error_m"]) + " | " + " / ".join(f"{v[2]:.0f}" for v in se["roll_depth_error_m"].values()) + " m | the same |")
        L.append(f"| Depth from clock error between rows: the budget ({d3['elevation']['clock_us_for_error']:.2f} µs) / reached in simulation ({sync_us:.2f} µs) / the within-row tolerance ({d3['alignment_tolerance_us']:.1f} µs) | "
                 + " / ".join(f"{r * np.radians(d3['elevation']['deg_per_us'] * us):.0f}" for us in (d3["elevation"]["clock_us_for_error"], sync_us, d3["alignment_tolerance_us"])) + " m | the same |")
        for k, v in S["refraction_m"].items():
            L.append(f"| Depth from the profile: mean gradient unknown by {k} /s" + (" (isothermal water against none)" if k == "0.017" else "") + f" | {v['12000.0']:.0f} m | the same |")
    if T.get("stage2"):
        L.append("| Depth from the profile, by rung of the profile ladder | see `docs/design/popoto_stage2.json` | |")
    else:
        L.append("| Depth from the profile, by rung of the profile ladder (local sensor; self-built profile; family; measured) | **pending**: `docs/design/popoto_stage2.json` does not exist yet; the gradient-prior rows above are what there is | |")
    if S:
        wp3, wp2 = S["three_rows"]["wrong_pick"], S["two_rows"]["wrong_pick"]
        key = max((k for k in next(iter(wp3.values())) if float(k) <= snr_row["ship"]), key=float)
        three = next(v for n, v in wp3.items() if n.startswith("3 rows") and "equal" not in n)[key]
        two = next(v for n, v in wp3.items() if n.startswith("2 rows"))[key] if any(n.startswith("2 rows") for n in wp3) else None
        wide2 = wp2.get("searching -30 to 30 deg", {})
        k2 = max((k for k in wide2 if float(k) <= snr_row["ship"]), key=float) if wide2 else None
        L.append(f"\n**Two rows against three at that SNR.** At {snr_row['ship']:.0f} dB a row (the study's nearest tabulated SNR below it: {float(key):.0f} dB), in clean noise searching ±30°: three rows pick a wrong lobe "
                 f"{100 * three[0]:.1f}% of the time" + (f", two rows {100 * two[0]:.0f}%" if two else "") + (f" (two rows at {float(k2):.0f} dB: {100 * wide2[k2]:.0f}%)" if k2 and not two else "")
                 + f". Two rows' repeats are {d2['elevation']['repeat_spacing_deg']:.1f}° apart and only {S['two_rows']['margin_db']:.1f} dB down, so the chain reports the list of possible angles, not one, unless water depth and range rule the others out: at 12 km in 100 m of water every arrival is within "
                 f"{np.degrees(np.arctan2(2 * 100.0, r)):.1f}° of level (one bounce), well inside the ±{d2['elevation']['unambiguous_half_sector_deg']:.1f}° that leaves one candidate. **So at 12 km in shallow water two rows are not ambiguous; they are in deep water.**\n"
                 "Multipath is the other limit: at 12 km in 100 m of water the direct, surface and bottom echoes are within a fraction of a range cell and of the rows' elevation beam of each other, they merge, and the elevation read is of the\n"
                 "mixture. The shallow-water checks at 1–3 km already show a pull of tenths of a degree that no sigma contains; at 12 km a tenth of a degree is "
                 f"{r * np.radians(0.1):.0f} m.\n")
    L.append(f"**What is knowable at 12 km, honestly.** The angle of arrival at the array to about {e3:.1f}° from noise on the ship-sized target, plus the roll accuracy ({roll:g}° assumed); whether the energy comes from above or below level;\n"
             "which arrival of a group came first; how a track's elevation changes over pings. **Depth in metres only with a sound-speed profile along the path**: without one the gradient prior alone is worth\n"
             + (f"{S['refraction_m']['0.05']['12000.0']:.0f} m at 12 km" if S else "kilometres") + ", more than any water depth, and the report says so in `depth_refraction_sigma_m`.\n")
    L += without_a_profile(d2, d3)
    # tracking
    from gpusonar import live as _live
    confirm = (dep.get("tracker") or {}).get("confirm_hits", _live.DEFAULTS["tracker"]["confirm_hits"])       # the chain's own unless the build says
    L.append(f"**Tracking at 12 km.** {60.0 / d3['ping_interval_s']:.1f} pings a minute on target (one every {d3['ping_interval_s']:.1f} s for the full window). The shipped tracker confirms on {confirm} detections "
             f"(`tracker.confirm_hits`), so a track exists {confirm * d3['ping_interval_s']:.0f} s after the first detection at the earliest, {(confirm - 1) * d3['ping_interval_s']:.0f} s if counted from it; a mover's class needs its speed, "
             f"which at {b3:.2f}° of bearing accuracy ({r * np.radians(b3):.0f} m across) takes several pings more for a crossing target. "
             + ("Measured: the table above gives the ping at which the track on the ship was confirmed in each water and how the later (bounced) arrivals were classed.\n" if K else "Measured figures: pending (`popoto_12km.json`).\n"))
    # hardware
    cpu, gpu = T.get("cpu"), T.get("gpu")
    L.append("**Hardware for the 12 km point** (details and the plan's own words: *Parts list*).\n")
    L.append(f"- **Projector:** {d3['source_level_db']:g} dB re 1 µPa at 1 m over {f_lo / 1e3:g}–{f_hi / 1e3:g} kHz for {d3['pulse_s']:g} s every {d3['ping_interval_s']:.1f} s (duty {100 * d3['pulse_s'] / d3['ping_interval_s']:.1f}%); "
             f"{10 ** ((d3['source_level_db'] - 170.8) / 10):.0f} W acoustic from an omnidirectional source. **None is chosen**; the plan's transducer covers 20–35 kHz.")
    L.append(f"- **Direct sound:** {d3['direct']['level_db']:.0f} dB re 1 µPa at the nearest hydrophone, {d3['direct']['over_full_scale_db']:.0f} dB over the inputs' full scale at the placeholder position: *The projector's direct sound* above.")
    L.append(f"- **Hydrophones, boards, synchronisation:** {d3['channels']} hydrophones on {d3['n_boards']} boards (three rows), {d2['channels']} on {d2['n_boards']} (two); rows agree to {d3['elevation']['clock_us_for_error']:.2f} µs for {d3['elevation']['clock_error_deg']:g}°.")
    L.append(f"- **Data:** {d3['rate_total_bit_s'] / 1e6:.0f} Mbit/s while receiving (three rows); {d3['samples_per_ping']:,} samples a channel and {d3['record_bytes_per_ping'] / 1e6:.0f} MB a ping for the {d3['window_m'][0]:g}–{d3['window_m'][1]:g} m window, "
             f"{d3['record_bytes_per_hour'] / 1e9:.0f} GB an hour recorded raw (4 bytes a sample), {d3['record_bytes_per_hour_decimated'] / 1e9:.0f} GB an hour decimated by {d3['decimation_factor']}. "
             f"**Recorded {d3['recording_mode']} by default** (the owner's decision; `storage.record`): a decimated recording has lost the synchronisation reference and everything outside the band for good, so raw is the safe default during trials "
             "and `decimated` the documented option for storage-limited runs. For one / two / three rows: "
             + " / ".join(f"{dd['record_bytes_per_hour'] / 1e9:.0f}" for dd in (d1, d2, d3)) + " GB an hour raw, " + " / ".join(f"{dd['record_bytes_per_hour_decimated'] / 1e9:.1f}" for dd in (d1, d2, d3)) + " decimated.")
    cpus = [bs.cpu_seconds_per_ping(build(n)[0]) for n in ("popoto_face_1x32", "popoto_face_2x32", "popoto_face_3x32")]
    if all(cpus):
        L.append("- **Compute: the plan** (the owner's decision on the stage-2 timings). Over the full window the decimate-first stage on a CPU takes " + " / ".join(f"{x['decimate_first']:.1f}" for x in cpus)
                 + f" s a ping for 1 / 2 / 3 rows against {d3['ping_interval_s']:.1f} s between pings: **one row may run on a CPU; two or three rows need a GPU** (`compute.option` defaults to `gpu` for stacked rows and the validator warns of a CPU "
                 "option slower than the ping interval). **The full-window GPU figure is still unmeasured**: the L4 numbers below are for scenario-size decimated pings.")
    if cpu and gpu:
        L.append(f"- **Compute, both options, measured:** on decimated pings of the `popoto_12km` scenario's size the C++ chain takes {cpu['seconds']['1']:.2f} / {cpu['seconds']['2']:.2f} / {cpu['seconds']['3']:.2f} s a ping for 1 / 2 / 3 rows on {cpu['threads']} shared CPU threads, "
                 f"and {gpu['seconds']['1']:.3f} / {gpu['seconds']['2']:.3f} / {gpu['seconds']['3']:.3f} s on an {gpu['header']['gpus'][0]} ({gpu['host_seconds']['1'] / gpu['seconds']['1']:.0f} times its own host's {gpu['header']['host_threads']} threads for one row; the tree before the 1.5 m row heights). "
                 + (f"A full-length raw ping at the boards' rate took {T['cpu_full_rate']['seconds']['1']:.0f} / {T['cpu_full_rate']['seconds']['2']:.0f} / {T['cpu_full_rate']['seconds']['3']:.0f} s on the CPU: not real time against {d3['ping_interval_s']:.1f} s without decimating first. " if T.get("cpu_full_rate") else "")
                 + ("Decimate-first timings: `docs/design/popoto_stage2.json`." if T.get("stage2") else "Timings with the decimate-first stage: **pending** (`popoto_stage2.json`). Raw full-rate pings have not been run on a GPU."))
    L.append("\n**What the earlier builds reach instead** (noise-limited, ship / small): " + "; ".join(
        f"`{n}`: {_km(build(n)[2]['reach_ship_m'])} / {_km(build(n)[2]['reach_small_m'])} km" for n in ("popoto_testauv_1x32", "popoto_static_2x32", "popoto_static_3x32"))
             + f". They use a {build('popoto_testauv_1x32')[1]['pulse_s']:g} s pulse (10 dB less processing gain than 1 s) for short-range trials; the static array with the face's 1 s pulse has the face's reach. No projector is chosen for them either.\n")
    L.append("Scenarios and files for this point: `popoto_12km` (one row), `popoto_12km_2row` with `configs/deploy/popoto_face_2row.yaml`, `popoto_12km_3row` with `configs/deploy/popoto_vehicle_elevation.yaml`; "
             "`tests/python/test_buildspec.py` runs the stacked-row ones through both engines (image-source paths always; Bellhop when present).\n")
    return L


def possible_angles(d) -> str:
    """Plain words for what two rows report (used wherever the plan used to say "candidates")."""
    e = d["elevation"]
    return (f"with two rows {e['baseline_m']:g} m apart the array cannot tell which of several elevation angles (about {e['repeat_spacing_deg']:.1f}° apart) an echo came from, so it reports the list of possible angles "
            "instead of one; bearing, range and the track are unaffected; the list collapses to one angle when the water depth and the range rule the others out, when a track already has a depth, or with a third row")


def _bands(b):
    return ", ".join(f"{lo:g}–{hi:g} m" if hi > lo else f"{lo:g} m" for lo, hi in b) or "none"


def depth_and_targets() -> list:
    """Surface interference, own depth as a lever, target classes: from popoto_depth_planner.json, surface_image_validation.json, popoto_12km.json."""
    P, V, K = (_json(n) for n in ("popoto_depth_planner.json", "surface_image_validation.json", "popoto_12km.json"))
    L = ["### Surface interference, own depth, and target classes\n"]
    if not P:
        return L + ["**Pending**: `tools/popoto_depth_planner.py --shipped` writes `docs/design/popoto_depth_planner.json`; it has not been run in this tree.\n"]
    L.append("The owner, on the finding that a ship at 4 m is lost at 12 km with the array at 10 m: \"Well we are an auv so we can change our depth, and in general I want to detect large subsurface movers as well like other xl auvs "
             "and larger. But yes let's add surface reflection loss so we can detect ships\". Three things follow: the surface-reflected path is now in every range figure; own depth is a planned quantity; and the builds are planned "
             "against named target classes. **Every target strength, target depth, speed, environment and operating depth limit below is an assumed placeholder** (`configs/builds/targets/`, `configs/builds/environments/`).\n")
    ph = P["physics"]
    lobe, rows = ph["first_lobe"], {r["own_depth_m"]: r for r in ph["by_own_depth"]}
    r_km, z_ship = ph["range_m"] / 1e3, ph["ship_depth_m"]
    L.append(f"**The surface image (Lloyd mirror).** A source and a target below the surface are joined by the direct path and by the path reflected at the surface, which arrives with the opposite sign. Their sum, each way, is "
             f"|2 sin(2π z₁ z₂ / λR)| of the direct path: nothing at the surface, a first maximum of twice the pressure where own depth × target depth = λR / 4 ({lobe['depth_product_m2']:.0f} m² at {r_km:g} km and a "
             f"{ph['wavelength_m'] * 100:.0f} cm wavelength). The model (`gpusonar.lloyd`) has the surface's roughness (coherent reflection falling with rms wave height, an input of its own), bubble loss at each bounce by a named wind law, "
             "the whole transmitted band, the part a rough surface scatters (kept as power), and, where there is a bottom, the layer's image series. Two-way factor on the echo for a target at "
             f"{z_ship:g} m and {r_km:g} km, dB against the free-field budget (single frequency / all the echo's energy over the band / the peak of the compressed echo, which is what detection uses):\n")
    labels = [k for k in next(iter(rows.values())) if k != "own_depth_m"]
    L.append("| Own depth | " + " | ".join(labels) + " |\n| --- |" + " --- |" * len(labels))
    for z, row in rows.items():
        L.append(f"| {z:g} m | " + " | ".join(f"{row[k]['narrowband_db']:+.1f} / {row[k]['energy_db']:+.1f} / {row[k]['peak_db']:+.1f}" for k in labels) + " |")
    null = rows[min(rows, key=lambda z: abs(z - lobe["first_null_m"]))]["flat"]
    r10 = ph["roughness"][0]
    hk = [k for k in r10 if k.startswith("coherent_reflection_rms_")]
    deep = {t["target_depth_m"]: t for t in ph["by_target_depth"]}
    L.append(f"\n- **Own depth is worth tens of decibels against a near-surface target.** At 10 m the echo is {-rows[10.0]['flat']['peak_db']:.0f} dB down; it crosses the free-field value at {lobe['zero_db_m']:.0f} m and peaks, "
             f"{rows[min(rows, key=lambda z: abs(z - lobe['peak_m']))]['flat']['peak_db']:+.1f} dB, at {lobe['peak_m']:.0f} m.\n"
             f"- **The band fills the nulls between lobes, not the one at the surface.** At the first null below the peak ({lobe['first_null_m']:.0f} m) a single frequency gives {null['narrowband_db']:.0f} dB; the 4–6 kHz sweep gives "
             f"{null['peak_db']:+.1f} dB at the peak and {null['energy_db']:+.1f} dB in energy. Close to the surface the three figures agree: a wider band does not help a shallow array see a shallow target. Beyond a depth product of about "
             f"{ph['band_fills_beyond_depth_product_m2']:.0f} m² the pattern is washed out altogether.\n"
             f"- **Roughness hardly matters at these angles.** At {r_km:g} km the surface path meets the surface at {r10['grazing_deg']:.2f}° (array at 10 m) to {ph['roughness'][-1]['grazing_deg']:.1f}° (at {ph['roughness'][-1]['own_depth_m']:g} m): the coherent reflection is "
             + ", ".join(f"{ph['roughness'][-1][k]:.2f} for {k.split('_')[-1][:-1]} m rms" for k in hk) + f" at the steeper end (the wind law gives {ph['wind_law_rms_height_m']['5']:.2f} m at 5 m/s and {ph['wind_law_rms_height_m']['9.8']:.2f} m at 9.8 m/s; "
             "measured heights did not follow it, so the height is its own input). The incoherent limit, where the surface path carries no fixed phase, leaves the peak at the free-field value and the energy 3 dB a leg above it.\n"
             f"- **Bubbles, by the `apl_1994` law, remove the surface path in a high wind** ({r10['bubble_loss_db_9p8_m_s']:.0f} dB a bounce at {r10['grazing_deg']:.2f}° and 9.8 m/s): the null fills and the lobe's gain goes with it (last column). "
             "That law grows without limit towards grazing and its sources state no range of validity; a target inside the bubble layer is not modelled. Treat that column as a bound, not a prediction.\n"
             "- **A deep target is spared.** Over own depths of " f"{deep[4.0]['own_depths_m'][0]:g}–{deep[4.0]['own_depths_m'][1]:g} m the peak factor ranges over "
             + "; ".join(f"{t['peak_db_lowest']:+.0f} to {t['peak_db_highest']:+.0f} dB for a target at {z:g} m" for z, t in deep.items())
             + ": once the two paths differ by more than the pulse resolves, the direct-then-surface and surface-then-direct echoes arrive together and add (+6 dB), whatever the depth.\n")
    if V:
        ch, big = V["through_chain"], V["largest_difference_db"]
        rows_c = [r for r in ch["rows"] if r["chain_factor_db"] is not None]
        L.append(f"**Checked.** Against the simulator's image-source paths through `run_scenario` on both engines (echo with the surface over echo without, own depth 10–600 m, target 4, 30, 150 m, at 12 and 3 km): within "
                 f"{max(v for k, v in big.items() if k.endswith('peak')):.1f} dB at the peak and {max(v for k, v in big.items() if k.endswith('energy')):.1f} dB in energy, the largest differences in the nulls, where the sweep's taper matters. "
                 f"Through the C++ chain with noise (a strong point at 4 m, 12 km): the detector's SNR against own depth follows the model within {max(abs(r['chain_factor_db'] - r['model_peak_db']) for r in rows_c):.1f} dB over "
                 f"{min(r['model_peak_db'] for r in rows_c):.0f} to {max(r['model_peak_db'] for r in rows_c):+.0f} dB."
                 + (" Against Bellhop, one way at 12 km: " + "; ".join(f"{k.split(' (')[0]}: energy within {v['one_way_energy_difference_db']['largest']:.1f} dB, two-way peak within {v['two_way_peak_difference_db']['largest']:.1f} dB" for k, v in V["bellhop"].items())
                    + " (the largest in the 10 m / 4 m null, where the arrivals file's digits limit Bellhop)." if V.get("bellhop") else " Bellhop was not available for the check.")
                 + f" The projector's depth counts in a null: 1.5 m below the array (the one-row file) is {20 * np.log10(11.5 / 10):.1f} dB there (`tools/surface_image_validation.py`).\n")
        if ch.get("chain_minus_budget_db") is not None:
            gap = -ch["chain_minus_budget_db"]
            need = bs.DETECTOR_THRESHOLD_DB
            L.append(f"> **THE BUDGET'S {bs.BUDGET_THRESHOLD_DB:g} dB THRESHOLD IS NOT THE SHIPPED DETECTOR'S.** Measured, same target, free field "
                     f"(`tools/surface_image_validation.py`): the chain's `snr_db` reads {ch['chain_snr_free_field_db']:.1f} dB where the budget says {ch['budget_snr_free_field_db']:.1f} dB, "
                     f"{gap:.1f} dB less. For a detection probability of {bs.DESIGN_DETECTION_PROBABILITY:g} the detector needs about {need[0.5]:g} dB on the budget's footing "
                     f"({need[0.9]:g} dB for 0.9; `buildspec.DETECTOR_THRESHOLD_DB`, from `live.budget_threshold_db`): {need[0.5] - bs.BUDGET_THRESHOLD_DB:.1f} dB more than the "
                     f"{bs.BUDGET_THRESHOLD_DB:g} dB used below. **Every margin below is therefore optimistic by about {need[0.5] - bs.BUDGET_THRESHOLD_DB:.1f} dB** until the budget tools "
                     "take the detector's own threshold and quote detection probabilities (`live.detection_probability`); that step is open. Source levels are RMS in the simulator and "
                     "in the budget alike (D-102).\n")
    # the plans
    face = P["builds"]
    b3 = face["popoto_face_3x32"]["plans"]
    L.append("**Which water is which.** *Non-ducting*: the direct path and the surface image only. It applies where the bottom returns little towards the target: water deep against the range (the bounce meets the bottom above its critical "
             "angle), a soft bottom (silt, clay, mud), or sound bent down onto such a bottom. *Ducted*: shallow against the range over sand or gravel, where many bounces arrive nearly whole; the image series (isovelocity) or Bellhop (with a profile) "
             "then replaces the two-path factor and is worth tens of decibels more than spherical spreading. The planner computes whichever the environment describes and says which it used.\n")
    L.append("**The depth plan** (`tools/popoto_depth_planner.py`; three rows on the face, 12 km; margin over 13 dB at the map, at the LOW end of each class's assumed target strength, worst over the class's depths; own depths "
             f"{b3[next(iter(b3))]['own_depths_m'][0]:g} m to the platform's limit, {b3[next(iter(b3))]['platform_max_depth_m']:g} m (the owner: \"Depth no further than 500m\"), and no closer to the bottom than the clearance, a placeholder):\n")
    L.append("| Water | Class | Free-field margin | At 10 m | Best own depth | Margin there | Own depths that hold it | Reach there |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
    for env_name, pl in b3.items():
        if "skipped" in pl:
            L.append(f"| {env_name} | not run: {pl['skipped']} | | | | | | |")
            continue
        for cname, c in pl["classes"].items():
            if "skipped" in c:
                continue
            r = c["recommended"]
            ten = next((v for k, v in c["at_own_depth"].items() if 9.0 <= float(k) <= 13.0), None)
            L.append(f"| {pl['environment'].get('title', env_name)} | {cname} ({c['ts_db'][0]:+.0f} dB, {c['depth_m'][0]:g}–{c['depth_m'][1]:g} m) | {c['free_field_margin_db']:+.1f} dB | "
                     + (f"{ten['margin_db']:+.0f} dB" if ten else "—") + f" | {r['own_depth_m']:g} m" + (" (the deepest searched)" if r["at_the_deepest_searched"] else "") + f" | {r['margin_db']:+.1f} dB | {_bands(r['held_band_m'])} | {_km(r['reach_m'])} km |")
    if (DESIGN / "popoto_depth_planner.png").exists():
        L.append("\n![Detection margin against range and own depth](design/popoto_depth_planner.png)")
    inside = [c["recommended"]["inside_platform_limit"] for pl in b3.values() if "classes" in pl for c in pl["classes"].values() if "recommended" in c]
    L.append(f"\nEvery best depth in the table is inside the platform's limit ({sum(1 for x in inside if x)} of {len(inside)}): the search goes no deeper. Where it says *the deepest searched*, the limit or the bottom clearance is what stopped it "
             "and a deeper array would do better. The limit is also the depth rating the parts list asks of every wet part.\n")
    deep_pl = b3.get("deep_no_bottom_return", {})
    if "classes" in deep_pl:
        ship, both = deep_pl["classes"]["ship"], deep_pl["together"].get("ship + large_submerged")
        sr = ship["recommended"]
        L.append(f"\n- **A ship in non-ducting water wants the array deep.** Best at {sr['own_depth_m']:g} m ({sr['margin_db']:+.1f} dB); held anywhere in {_bands(sr['held_band_m'])}; within 3 dB of the best over {_bands(sr['within_3db_band_m'])}. "
                 f"At 10 m it is {ship['at_own_depth']['10']['margin_db']:+.0f} dB and the reach is {_km(ship['at_own_depth']['10']['reach_m'])} km; the shallowest depth that holds it at 12 km is {ship['shallowest_own_depth_held_m']:g} m.")
        if both:
            L.append(f"- **One depth for the ship and a deep submerged target:** {_bands(both['held_band_m'])}; best {both['best_own_depth_m']:g} m ({both['margin_of_the_weaker_db']:+.1f} dB for the weaker).")
        dv = ship["depth_diversity"]
        L.append("- **Depth diversity.** Pinging at more than one depth (or as the vehicle changes depth) fills what one depth leaves: at the ship's target strength, every target depth from 1 to "
                 f"{dv['deepest_target_m']:g} m at 12 km is held by " + ", then ".join(f"{s['own_depth_m']:g} m ({100 * s['covered_share']:.0f}%)" for s in dv["steps"])
                 + f"; the best single depth covers {100 * dv['one_depth_best_share']:.0f}%. For weaker targets and other waters: the JSON's `depth_diversity`.")
        x = deep_pl["classes"].get("xl_auv")
        if x and "recommended" in x:
            L.append(f"- **The small submerged class is not reached at 12 km at its low figure** ({x['ts_db'][0]:+.0f} dB: {x['recommended']['margin_db']:+.1f} dB at best, reach {_km(x['recommended']['reach_m'])} km); at broadside "
                     f"({x['ts_db'][1]:+.0f} dB assumed) add {x['margin_high_ts_db']:.0f} dB.")
        L.append("- **What depth costs.** The nulls move with the target's depth: the table's last column of the planner (`target_depths_not_held_m`) lists the target depths a chosen own depth leaves uncovered; the bottom limits own depth in shallow "
                 "water (clearance is a parameter); and with a real profile the answer changes (the profile rows above: the same classes through Bellhop). **The plan rests on the water it is given**: isovelocity unless a profile is supplied, "
                 "and every shipped water is illustrative.\n")
    L.append("Rows only move the margin: " + "; ".join(
        f"{name.split('_')[-1]}: ship best {pl['plans']['deep_no_bottom_return']['classes']['ship']['recommended']['margin_db']:+.1f} dB at {pl['plans']['deep_no_bottom_return']['classes']['ship']['recommended']['own_depth_m']:g} m"
        for name, pl in face.items() if "deep_no_bottom_return" in pl.get("plans", {}) and "classes" in pl["plans"]["deep_no_bottom_return"] and name.startswith("popoto_face")) + " (non-ducting water).\n")
    stat = {n: v for n, v in face.items() if not n.startswith("popoto_face")}
    if stat:
        L.append("**The static array's hanging depth is the same choice.** At the end of each build's own receive window, non-ducting water / 100 m over sand, the ship class: " + "; ".join(
            f"`{n}` ({_km(v['range_of_interest_m'])} km): " + " / ".join(
                (f"{v['plans'][e]['classes']['ship']['recommended']['own_depth_m']:g} m ({v['plans'][e]['classes']['ship']['recommended']['margin_db']:+.0f} dB)" if "classes" in v["plans"].get(e, {}) else "—")
                for e in ("deep_no_bottom_return", "shelf_100m_sand")) for n, v in stat.items())
            + ". Their specs hang or run them at " + ", ".join(f"{v.get('spec_depth_m') or v.get('trial_own_depth_m'):g} m" for v in stat.values()) + " today (placeholders).\n")
    # target classes
    L.append("**Target classes** (`targets:` in a build spec). The owner wants large subsurface movers as well as ships. Three classes are shipped as examples, each marked `assumed`:\n")
    L.append("| Class | Target strength used: low to high | Depths | Speeds | Where the figure comes from |\n| --- | --- | --- | --- | --- |")
    for c in P["classes"]:
        L.append(f"| `{c['name']}`: {c.get('title', '')} | {c['ts_db'][0]:+.0f} to {c['ts_db'][1]:+.1f} dB (**{c['status']}**) | {c['depth_m'][0]:g}–{c['depth_m'][1]:g} m | "
                 + (f"{c['speed_m_s'][0]:g}–{c['speed_m_s'][1]:g} m/s" if c.get("speed_m_s") else "—") + f" | {c.get('note', '')} |")
    L.append("\nNo measured target strength of a ship, an unmanned vehicle or a submarine is cited in this repository; these are placeholders for the owner to replace. What can be computed is the geometric-acoustics figure of a bare rigid "
             "cylinder (the simulator's own glint model, band-averaged over 4–6 kHz, seen from 12 km):\n")
    asp = [a["aspect_deg"] for a in next(iter(P["cylinders"].values()))["by_aspect"]]
    L.append("| Diameter × length | " + " | ".join("broadside" if a == 0 else ("end-on" if a == 90 else f"{a:g}°") for a in asp) + " | Broadside lobe | Far field beyond |\n| --- |" + " --- |" * (len(asp) + 2))
    for label, cy in P["cylinders"].items():
        L.append(f"| {label} | " + " | ".join(f"{a['ts_db']:+.0f}" for a in cy["by_aspect"]) + f" | {cy['broadside_lobe_deg']:.2f}° | {_km(cy['far_field_beyond_m'])} km |")
    big = P["cylinders"][list(P["cylinders"])[-1]]
    L.append(f"\n(dB re 1 m.) A smooth cylinder is strong only within a fraction of a degree of broadside and end-on, and tens of decibels lower in between. A long hull is inside its own far-field distance at sonar ranges: the largest reads "
             f"{big['by_aspect'][0]['ts_db']:+.1f} dB at 12 km against {big['broadside_far_field_db']:+.1f} dB in the far field. **A real hull is not this**: a sail, fins and a tapered stern raise the figure off broadside, and a coating can lower "
             "it everywhere. The table sizes a study; it predicts no vehicle.\n")
    if K and K.get("class_cases"):
        C = K["class_cases"]
        L.append(f"**Through the chain** (`tools/popoto_12km_study.py --classes`; C++ chain, each row count's deployment file as shipped, so **no sound-speed profile knowledge**; six pings; planned own depths {C['planned_own_depths_m']['ship']:g} m "
                 f"for the ship and {C['planned_own_depths_m']['both']:g} m for both classes; the submerged mover is {C['mover']['target_strength_db']:g} dB as a whole at {C['mover']['depth_m']:g} m and {C['mover']['speed_m_s']:g} m/s, inside its class's assumed range):\n")
        L.append("| Case | Rows | Detected on | Median SNR | Track confirmed at ping | Class | Speed error | Heading error | Depth reported (truth) | Truth within two sigma | Pings with several possible elevations |\n"
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for label, by in C["cases"].items():
            for rows_ in sorted(by):
                v = by[rows_]
                if v.get("error"):
                    L.append(f"| {label} | {rows_} | failed: {v['error'][:70]} | | | | | | | | |")
                    continue
                t = v.get("track")
                dep = (f"{t['depth_m']:.0f} ± {t['depth_sigma_m']:.0f} m ({v['true_depth_m']:g} m)" if t and t.get("depth_m") is not None else "none")
                L.append(f"| {label} | {rows_} | {v['pings_detected']} of {v['pings']} | " + (f"{v['median_snr_db']:.1f} dB" if v["median_snr_db"] is not None else "—") + f" | {v['track_confirmed_at_ping'] or '—'} | "
                         + (f"{t['kind']} | {t['speed_error_m_s']:+.1f} m/s | {t['heading_error_deg']:+.1f}° | {dep} | " + ("—" if t["true_depth_within_two_sigma"] is None else ("yes" if t["true_depth_within_two_sigma"] else "**no**"))
                            if t else "no track | | | | ") + f" | {v['pings_with_several_possible_elevations']} |")
        if C.get("not_run"):
            L.append("\nNot run: " + "; ".join(C["not_run"]) + ".")
        L.append("\n(Detections are counted within 20 m of the straight-line range to the target; in the ducted case the strongest arrivals come later than that, so the count can read 0 while the track on the target exists. "
                 "With two rows the generated deployment file keeps each detection at the row's own range and bearing (`elevation.correct_bearing: false`): placed with a wrongly picked elevation angle, detections jumped by up to a kilometre "
                 "at 12 km and the track broke.)")
        L.append("\nThe ship is lost with the array at 10 m and held at the planned depth; the submerged mover is tracked and classed as a mover at 6 and 12 km with nothing known of the sound-speed profile; the depth it reports then carries a sigma of "
                 "kilometres at 12 km, which is the honest figure. The cylinder is seen only on the pings where it is abeam: aspect, not range, decides it.\n")
    return L


def without_a_profile(d2, d3) -> list:
    """What is available with no sound-speed profile at all, and what a dive adds: the owner's rule that a dive is never a requirement."""
    S2 = bs.measured_timings().get("stage2") or {}
    summ = (S2.get("profile_ladder") or {}).get("summary") or {}
    L = ["**With no sound-speed profile at all.** The owner: \"We can dive but it should [not] be a requirement to get a track of a subsea mover.\" It is not. With nothing but the gradient prior or the vehicle's local sound speed (the bottom "
         "rungs of the profile ladder) the system gives: range; bearing (with the cone correction); range rate, speed and heading from the track; mover against multipath; the elevation angle at the array with its sigma; whether the target is "
         "above or below the vehicle; which arrival of a group came first; how the elevation changes along the track; and a depth with an honest, wide sigma and the name of the rung it rests on (`depth_profile_source`). "
         "Track initiation, confirmation and the `mover` class use bearing, range and their history only: nothing in the tracker or the classifier waits for a depth, a profile or a single elevation "
         "(`detect.py`: a detection without a depth still starts and confirms a track; only the track's depth is withheld until a detection offers one). "
         f"That holds for two rows as well, where {possible_angles(d2)}. The generator's `submerged` self-test and the class cases above hold the software to it.\n",
         "**What a dive adds (optional).** Only \"depth in metres with a small sigma at several kilometres\" needs a profile covering the water between vehicle and target."]
    if summ:
        keys = list(summ)
        r6 = "6000"
        L[-1] += (" The stage-2 study (`docs/design/popoto_stage2.json`, illustrative profiles): median depth error at 6 km by rung: " + "; ".join(f"{k}: {summ[k][r6]['median_abs_error_m']:.1f} m" for k in keys if r6 in summ[k]) + ".")
    L[-1] += (" So a dive through the water column before a search, and again when the water changes, is a step in the operating checklist marked optional; for the static array, a cast at deployment. The depth planner says which it used: "
              "isovelocity water unless the environment carries a profile.\n")
    return L


def parts_section() -> list:
    import parts_list
    cols = parts_list.stage_columns()
    L = ["### Parts list\n",
         "Generated from the stage files (`tools/parts_list.py`; the full table with every requirement, what fixes it and the plan's own words: `design/popoto_parts.md`). Status: `chosen` = named in the owner's plan or by the owner;\n"
         "`specified, not chosen` = the design says what it must do; `unconfirmed` = needed, not yet specifiable. No part number, price, weight or power draw is invented.\n"]
    md = parts_list.markdown(cols, heading="####")
    cut = next(i for i, x in enumerate(md) if x.startswith("#### What the design asks"))
    L += md[:cut]
    L.append("**Where the plan and the files disagree** (findings, not changes):\n")
    L += [f"- {x}" for x in parts_list.findings(cols[-1][1])]
    L.append("\n![hardware, three rows on the face](design/build_face_3x32_hardware.png)\n\n![one ping, three rows on the face](design/build_face_3x32_timing.png)\n\n"
             "The same two diagrams for every stage: `design/build_<stage>_hardware.png`, `_timing.png` (`tools/build_diagrams.py`).\n")
    return L


if __name__ == "__main__":
    for fn in (staged_plan, row_modules, projector_options, twelve_km):
        print("\n".join(fn()))
