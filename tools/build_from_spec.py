#!/usr/bin/env python3
"""From a build spec to everything needed to simulate the build and to run it on hardware.

    tools/build_from_spec.py SPEC --validate             # check the spec: errors and warnings, nothing written
    tools/build_from_spec.py SPEC --out DIR              # generate (deterministic: the same spec gives the same files)
    tools/build_from_spec.py SPEC --out DIR --self-test  # ... then simulate, run the chain on both engines, check the truth
    tools/build_from_spec.py --write-shipped             # the shipped builds' sonar and deployment files into configs/
    tools/build_from_spec.py --check-shipped             # fail if those files are not what the specs generate

SPEC is a name under configs/builds/ or a path. Fields: docs/REFERENCE_DESIGN.md. In DIR:

    configs/sonars/<name>.yaml, <name>_clocks.yaml       the sonar (extends a shipped base where the spec names one)
    configs/deploy/<name>.yaml                           the deployment file for tools/sonar_live.py
    configs/scenarios/<build>_{free,shallow,rehearsal,passing,passive}.yaml    simulated trials, each with its truth
    chain.options, listen.options, COMMANDS.txt          the same settings for the standalone C++ program build/bin/sonar_live
    template.h5                                          with --template: the C++ program's template record (large for a long window)
    diagrams/build_<name>_{hardware,timing}.png          tools/build_diagrams.py
    parts.md, performance.json, BUILD.md                 parts list, expected performance, bring-up checklist and self-test list
    self_test.json                                       with --self-test: what was run and what it found

To use the generated files without copying them: GPUSONAR_CONFIG_PATH=DIR/configs tools/sonar_live.py DIR/configs/deploy/<name>.yaml ...

Self-tests (each simulated trial through the Python chain and the C++ chain, host backend):
    free       range and bearing of every target; with rows, its depth (two rows: the true elevation among the candidates)
    shallow    the same with surface and bottom echoes; depth within the multipath allowance
    rehearsal  raw samples at the boards' rate, every board on its own clock: the boards synchronise; range, bearing, depth
    passing    one mover, its speed
    passive    the radiating sources' bearings, the sonar silent
    stream     pings as board blocks through both assemblers and over UDP loopback at the build's channel count
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from gpusonar import buildspec as bs  # noqa: E402
from gpusonar import config  # noqa: E402

KINDS = ("free", "shallow", "rehearsal", "passing", "submerged", "passive", "stream")
CRITERIA = {
    "free": "every target: range within 3 m, bearing within 1 deg, in both engines; three or more rows: depth within 3 m + 0.3% of range; two rows: the true elevation within 0.5 deg of the reported one or of a listed candidate",
    "shallow": "the target: range within 6 m (a detection is a cluster of the direct echo and its first bounces), bearing within 1 deg; rows: depth within 3 m + range x sin(0.7 deg) (the multipath allowance) when the report gives one elevation, else the truth among the candidates",
    "rehearsal": "the boards synchronise (ok, worst residual under 1 us) when the build has board clocks; range within 6 m, bearing within 1 deg; rows: depth as in shallow, with the attitude the platform had",
    "passing": "at least one track of kind mover, speed within 1 m/s of the truth, in both engines",
    "submerged": "a submerged mover of the spec's first submerged target class, with NO sound-speed profile and no water depth given to the chain (the deployment file's gradient prior only): "
                 "a track of kind mover, speed within 1 m/s and heading within 15 degrees of the truth; with rows, any depth the track reports has the true depth inside two of its sigmas, "
                 "and the track exists even when every detection lists several possible elevations. Skipped when the spec names no submerged target class",
    "passive": "both sources within 1 deg of their bearings in at least 80% of the frames, on the Eigen path and on the kernel path (a row of more than 64: the kernels refuse, skipped and said so)",
    "stream": "both assemblers and the UDP loopback return every sample within half a count; no sample lost",
}


def generate(spec: dict, out: Path, template: bool = False, diagrams: bool = True) -> dict:
    """Write everything for one build under `out`; return {path: what}."""
    warnings = bs.check(spec)
    out = Path(out)
    written = {}
    for rel, text in bs.files(spec).items():
        p = out / "configs" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        written[str(p.relative_to(out))] = "config"
    d = bs.derive(spec)
    perf = bs.performance(d)
    n = bs.names(spec)
    # The standalone C++ program: the chain section as key=value options.
    import sonar_live
    from gpusonar.live import cpp_chain_options, sync_reference
    deploy = bs.deploy_config(spec)
    config.add_config_dir(out / "configs")
    try:
        opts = _options(deploy["chain"])
        (out / "chain.options").write_text("".join(f"{k}={v}\n" for k, v in opts.items()))
        lo = bs.listen_options(spec)
        (out / "listen.options").write_text("".join(f"{k}={v:g}\n" if isinstance(v, float) else f"{k}={v}\n" for k, v in lo.items()))
        net = deploy["source"]["port"]
        cmds = ["# Commands for this build. GPUSONAR_CONFIG_PATH lets the tools find the generated sonar and scenario files.",
                f"export GPUSONAR_CONFIG_PATH={out}/configs", "",
                "# 1. simulate a trial and run the chain on it (add --engine cpp for the C++ chain):",
                f"tools/sonar_live.py {out}/configs/deploy/{n['deploy']}.yaml --scenario {spec['build']}_rehearsal --out run/", "",
                "# 2. rehearse ashore: the same pings sent as board blocks over UDP to the configured source:",
                f"tools/sonar_live.py {out}/configs/deploy/{n['deploy']}.yaml --emulate {spec['build']}_rehearsal --port 0 --out run/", "",
                "# 3. on the platform, from the boards:",
                f"tools/sonar_live.py {out}/configs/deploy/{n['deploy']}.yaml", "",
                "# 4. the standalone C++ program with the same settings (write the template record first: --template here, or",
                f"#    tools/sonar_live.py {out}/configs/deploy/{n['deploy']}.yaml --export-template template.h5):",
                f"build/bin/sonar_live --udp {net} --template {out}/template.h5" + (f" --nav-port {deploy['nav']['port']}" if deploy["nav"].get("port") else "")
                + f" --recording '{deploy['output']['recording']}' --report '{deploy['output']['report']}' --report-udp {deploy['output']['report_udp'][0]}:{deploy['output']['report_udp'][1]}"
                + f" --options {out}/chain.options", "",
                "# 5. listening without transmitting (passive bearings" + (", with elevations" if d["rows"] > 1 else "") + "):",
                f"build/bin/sonar_live --udp {net} --template {out}/template.h5 --listen --band {lo['f_low']:g} {lo['f_high']:g} --recording listen.h5 --report listen.jsonl --options {out}/listen.options"]
        (out / "COMMANDS.txt").write_text("\n".join(cmds) + "\n")
        if template:
            sonar_live.export_template(deploy, str(out / "template.h5"))
            written["template.h5"] = "template record"
        assert cpp_chain_options(deploy["chain"], sync_reference(deploy["chain"]))      # every key is one the chain takes
    finally:
        config.remove_config_dir(out / "configs")
    if diagrams:
        import build_diagrams
        for p in build_diagrams.draw_build(d, out / "diagrams", n["sonar"], spec.get("title")):
            written[str(p.relative_to(out))] = "diagram"
    import parts_list
    timings = bs.measured_timings()
    (out / "parts.md").write_text("\n".join([f"# Parts list: {spec.get('title', spec['build'])}\n",
                                             "Generated by `tools/build_from_spec.py` (`tools/parts_list.py`). Status words and rules: `docs/design/popoto_parts.md`.\n"]
                                            + parts_list.markdown([(spec.get("stage", spec["build"]), d)], timings)) + "\n")
    (out / "performance.json").write_text(json.dumps({"build": spec["build"], "derived": _plain(d), "performance": perf}, indent=2) + "\n")
    (out / "BUILD.md").write_text(build_md(spec, d, perf, warnings, out))
    for name in ("chain.options", "listen.options", "COMMANDS.txt", "parts.md", "performance.json", "BUILD.md"):
        written[name] = "generated"
    return written


def _options(chain: dict) -> dict:
    """The chain section as the key=value settings of build/bin/sonar_live (the same mapping `--export-template` prints)."""
    from gpusonar.live import cpp_chain_options, sync_reference
    o = cpp_chain_options(chain, sync_reference(chain))
    sync = o.pop("sync", None)
    o["look"] = "starboard" if o.pop("starboard") else "port"
    o["window"] = "hann" if o.pop("hann") else "none"
    # Sizes the deployment file leaves out are not on the command line either: the program derives them (derive.hpp).
    if "tracker_bearing_sigma" in o:
        o["tracker_bearing_sigma_deg"] = float(np.degrees(o.pop("tracker_bearing_sigma")))
    if o.pop("cell_peak", False):
        o["cell_statistic"] = "peak"
    if "multipath_bearing" in o:
        o["multipath_bearing_deg"] = float(np.degrees(o.pop("multipath_bearing")))
    o["multipath"] = int(o["multipath"])
    if "dead_channels" in o:
        o["dead_channels"] = ",".join(str(v) for v in o["dead_channels"])
    el = o.pop("elevation", None)
    if el is not None:
        o["elevation"] = 1
        el["roll_sigma_deg"] = float(np.degrees(el.pop("roll_sigma")))
        el["sector_margin_deg"] = float(np.degrees(el.pop("sector_margin")))
        el["attitude_rate_sigma_deg_s"] = float(np.degrees(el.pop("attitude_rate_sigma")))
        for key in ("ssp_depth", "ssp_speed"):
            values = el.pop(key)
            if values:
                el[key] = ",".join(f"{v:.10g}" for v in values)
        o.update({f"elevation_{k}": (int(v) if isinstance(v, bool) else v) for k, v in el.items()})
    if sync:
        o["sync"] = 1
        o.update({f"sync_{k}": (int(v) if isinstance(v, bool) else v) for k, v in sync.items()})
    return {k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in o.items()}


def _plain(d: dict):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return conv({k: v for k, v in d.items() if k not in ("positions", "row_of", "channel_board")})


def build_md(spec: dict, d: dict, perf: dict, warnings: list, out: Path) -> str:
    """The build's own page: numbers, expected performance, what is unconfirmed, bring-up checklist, self-tests."""
    n = bs.names(spec)
    el = d["elevation"]
    static = spec["platform"]["kind"] == "static"
    L = [f"# {spec.get('title', spec['build'])}\n",
         f"Generated by `tools/build_from_spec.py` from the spec `{spec['build']}`. Every number is arithmetic on the spec (`gpusonar/buildspec.py`) or a measurement file named beside it.\n",
         "![hardware](diagrams/build_%s_hardware.png)\n\n![timing](diagrams/build_%s_timing.png)\n" % (n["sonar"], n["sonar"]),
         "## What it is\n", "| | |\n| --- | --- |",
         f"| Hydrophones | {d['rows']} × {d['per_row'][0]} = {d['channels']}, {d['pitch_m'] * 100:.1f} cm apart ({d['pitch_wavelengths_top']:.2f} wavelengths at {d['band_hz'][1] / 1e3:g} kHz), rows {d['row_length_m']:.2f} m long"
         + (f", at {', '.join(f'{h:+.2f}' for h in d['heights_m'])} m" if el else "") + " |",
         f"| Boards | {d['n_boards']} of {d['inputs_per_board']} inputs; {d['sample_rate_hz'] / 1e3:g} kHz, {d['adc_bits']} bit; {d['rate_total_bit_s'] / 1e6:.1f} Mbit/s in all |",
         f"| Transmit | {d['band_hz'][0] / 1e3:g}–{d['band_hz'][1] / 1e3:g} kHz, {d['pulse_s']:g} s, {d['source_level_db']:g} dB re 1 µPa at 1 m; nothing inside {d['blind_range_m']:g} m |",
         f"| Receive window | {d['window_m'][0]:g}–{d['window_m'][1]:g} m: a ping every {d['ping_interval_s']:.2f} s at the shortest, {d['record_bytes_per_ping'] / 1e6:.0f} MB recorded a ping |",
         f"| Direct sound | {d['direct']['level_db']:.0f} dB re 1 µPa at the nearest hydrophone ({d['direct']['nearest_m']:.2f} m): {d['direct']['over_full_scale_db']:+.0f} dB against the inputs' full scale |",
         "\n## Expected performance\n", f"Assumptions: {perf['assumptions']}.\n", "| | |\n| --- | --- |",
         f"| Beam at broadside | {perf['beamwidth_deg']:.1f}° |",
         f"| Row's gain against sea noise / against the boards' own noise | {perf['gain_isotropic_noise_db']:.1f} / {perf['gain_uncorrelated_noise_db']:.1f} dB |",
         f"| Rows added as power | +{perf['rows_detection_gain_db']:.1f} dB of detection |",
         f"| Noise-limited reach, ship (+25 dB) / small (+5 dB) | {perf['reach_ship_m'] / 1e3:.1f} / {perf['reach_small_m'] / 1e3:.1f} km |",
         f"| Range resolution | {perf['range_resolution_m']:.3f} m before windowing |",
         f"| Steering with no grating lobe | ±{perf['grating_lobe_free_steering_deg']:.0f}° |"]
    if el:
        e = perf["elevation"]
        L += [f"| Elevation, one sigma at 13 / 20 / 30 dB a row | " + " / ".join(f"{e['sigma_deg'][k]:.2f}°" for k in ("13.0", "20.0", "30.0")) + " |",
              "| Repeats of the elevation response | " + (f"every {e['repeat_spacing_deg']:.1f}° at the band centre; highest inside ±30°: {e['highest_repeat_db']:.1f} dB" if e["repeat_spacing_deg"] else "none") + " |",
              f"| Clock agreement between rows for {e['clock_error_deg']:g}° | {e['clock_us_for_error']:.2f} µs |",
              f"| A row 1 mm out of plane | {e['deg_per_mm_out']:.3f}° |"]
        L.append(f"\nDepth, one sigma, ship-sized target (roll known to {perf['depth_budget_inputs']['roll_sigma_deg']:g}°, gradient unknown by {perf['depth_budget_inputs']['gradient_sigma_per_s']:g} /s):\n\n"
                 "| Range | from noise | from roll | from the unknown gradient | total |\n| --- | --- | --- | --- | --- |")
        for r, v in perf["depth_sigma_ship_m"].items():
            L.append(f"| {float(r) / 1e3:g} km | " + ("not detected | | | |" if v is None else f"{v['from_noise_m']:.1f} m | {v['from_roll_m']:.1f} m | {v['from_gradient_m']:.0f} m | {v['total_m']:.0f} m |"))
        L.append(f"\nMultipath: {perf['multipath_note']}.")
    L.append("\n## Not confirmed\n")
    L.append("From the spec (every value not listed under a section's `confirmed`), as carried into the generated files:\n")
    flags = bs.sonar_config(spec)["unconfirmed"] + bs.deploy_config(spec)["unconfirmed"] + (["receive.boards", "receive.reference"] if bs.needs_sync(spec) else [])
    L.append(", ".join(f"`{f}`" for f in flags) + ".\n")
    if warnings:
        L.append("Validator warnings:\n")
        L += [f"- `{w.where}`: {w.message}" for w in warnings]
    L.append("\n## Bring-up checklist\n")
    steps = ["Ashore, simulated: `--self-test` passes (below). Read `self_test.json`: the measured errors, not just pass or fail.",
             f"Ashore, emulated stream: command 2 of `COMMANDS.txt` (the rehearsal's pings as board blocks over UDP into the deployment file). The report must match command 1's.",
             "Boards on the bench, no water: the stream arrives on the configured port, no lost blocks, every channel's RMS and spectrum as expected (`tools/inspect_tap.py` on the recording); a tone injected at one input appears on that channel only: this checks the channel-to-board map.",
             ]
    if bs.needs_sync(spec):
        need = f"{el['clock_us_for_error']:.2f} µs between rows" if el else f"{d['alignment_tolerance_us']:.1f} µs"
        steps.append(f"Synchronisation on the bench: with the reference wired to every board, the report's `sync.worst_residual_us` must be under {need}. Swap two reference lines: the residual must not change (it would be the lines, not the clocks).")
    steps += [f"Projector on, in water, at the lowest level first: record one ping and look at the raw samples of the nearest hydrophone for clipping during the pulse and for how long after it the channel is disturbed "
              f"(expected from the files: {d['direct']['over_full_scale_db']:+.0f} dB against full scale at full level). The first usable range follows from that, not from the pulse length alone.",
              "A known target (or a transponder) at a surveyed range and bearing: range, bearing, SNR against the link budget above. A shortfall of more than a few dB is sensitivity, source level or noise: measure each (below).",
              ]
    if d["rows"] > 1:
        steps += ["Each row alone (`chain.dead_channels` naming the others): the same detections from every row. A row that differs is miswired or misplaced.",
                  "Row offsets: a source at three known depths at two or three bearings; `calibrate.estimate_channel_errors(..., planar=False)`; store the result with the sonar.",
                  "Elevation: the same source; `depth_m` against its depth, level and " + ("with the frame tilted" if static else "rolled 5°") + ". An error that follows the roll is the attitude source or its time stamp.",
                  ]
        if d["rows"] == 2 and el["repeat_spacing_deg"]:
            steps.append(f"Two rows: the candidates (`elevation_alternatives_deg`) are {el['repeat_spacing_deg']:.1f}° apart; with the site's water depth typed in or sent by navigation, `sector_from_depth` must leave the right one.")
    steps.append("Operating depth: `tools/popoto_depth_planner.py " + spec["build"] + "` for the water and the targets of the day (water depth, bottom, sea state or wave height, a profile if there is one). "
                 + ("The frame's hanging depth is that choice." if static else "Own depth is a lever: against a target near the surface in water that does not duct, tens of decibels.")
                 + " The answer rests on the water it is given; say which.")
    steps.append("OPTIONAL, never a precondition: " + ("a sound-speed cast at deployment" if static else "a dive through the water column before a search, and again when the water changes")
                 + ", so that the chain has a profile covering the water between the array and a target. It improves only the depth in metres at range. Detection, bearing, range, the track, its speed and heading, "
                 "mover against multipath, the elevation angle and whether the target is above or below are all available without it (the `submerged` self-test checks exactly that), and the depth then comes with a wide, honest sigma "
                 "and the name of what it rests on (`depth_profile_source`).")
    steps.append("Feed back: every measured value replaces its assumption in the spec and moves to `confirmed`; generate again and re-run the self-tests (docs/REFERENCE_DESIGN.md, step 7).")
    L += [f"{i + 1}. {s}" for i, s in enumerate(steps)]
    L.append("\n## Self-tests and their pass criteria\n")
    L.append(f"`tools/build_from_spec.py {spec['build']} --out {out} --self-test`\n")
    L += [f"- **{k}**: {v}." for k, v in CRITERIA.items()]
    return "\n".join(L) + "\n"


# --- self-tests ---------------------------------------------------------------------------------------

def _run(records, chain, engine):
    from gpusonar.live import LiveChain
    with LiveChain(dict(chain, engine=engine, backend="host")) as live:
        return live.run([r.copy() for r in records])


def _nearest(report, target):
    own = np.asarray(report["ownship_xyz"])
    rng = float(np.linalg.norm(np.asarray(target) - own))
    hits = [x for x in report["detections"] if x["snr_db"] is not None]
    if not hits:
        return None, own, rng
    near = [x for x in hits if abs(x["range_m"] - rng) < 8.0]
    if near:
        # Several arrivals of one target (surface and bottom bounces) can lie within a few metres: the direct one is the
        # nearest to the true range among those within 10 dB of the strongest.
        top = max(x["snr_db"] for x in near)
        near = [x for x in near if x["snr_db"] >= top - 10.0]
    return (min(near or hits, key=lambda x: abs(x["range_m"] - rng))), own, rng


def _check_target(report, target, rows, allowance_deg, look_sign=1.0, range_tolerance=3.0):
    """Errors of the detection nearest one target, and whether they pass."""
    det, own, rng = _nearest(report, target)
    if det is None:
        return {"found": False, "ok": False}
    bearing = float(np.degrees(np.arctan2(target[0] - own[0], look_sign * (target[1] - own[1]))))
    out = {"found": True, "range_error_m": det["range_m"] - rng, "bearing_error_deg": det["bearing_deg"] - bearing, "snr_db": det["snr_db"]}
    ok = abs(out["range_error_m"]) <= range_tolerance and abs(out["bearing_error_deg"]) <= 1.0
    if rows > 1:
        true_el = float(np.degrees(np.arcsin((own[2] - target[2]) / rng)))
        out["true_depth_m"], out["true_elevation_deg"] = float(target[2]), true_el
        out["depth_m"], out["elevation_deg"] = det.get("depth_m"), det.get("elevation_deg")
        out["ambiguous"] = bool(det.get("elevation_ambiguous"))
        cands = [e for e in [det.get("elevation_deg")] + list(det.get("elevation_alternatives_deg") or []) if e is not None]
        out["candidates_deg"] = cands
        if not cands:
            ok = False
        elif out["ambiguous"]:
            out["candidate_error_deg"] = min(abs(e - true_el) for e in cands)
            ok = ok and out["candidate_error_deg"] <= max(0.5, allowance_deg)
        else:
            out["depth_error_m"] = det["depth_m"] - target[2] if det.get("depth_m") is not None else None
            ok = ok and out["depth_error_m"] is not None and abs(out["depth_error_m"]) <= 3.0 + (rng * np.sin(np.radians(allowance_deg)) if allowance_deg else 0.003 * rng)
    out["ok"] = bool(ok)
    return out


def self_test(spec: dict, out: Path, kinds=KINDS, engines=("python", "cpp"), log=print) -> dict:
    """Run the build's simulated trials through the chain and check them against their truth."""
    from gpusonar import core
    from gpusonar.sim import run_scenario
    out = Path(out)
    config.add_config_dir(out / "configs")
    results = {"build": spec["build"], "engines": list(engines), "tests": {}}
    rows = int(spec["array"]["rows"])
    sign = 1.0 if spec["array"].get("look", "starboard") == "starboard" else -1.0
    scenarios = bs.scenario_configs(spec)
    try:
        for kind in kinds:
            t0 = time.time()
            res: dict = {"criterion": CRITERIA[kind]}
            try:
                if kind == "stream":
                    res.update(_stream_test(spec))
                elif kind == "passive":
                    sc = scenarios["passive"]
                    truth = sc.pop("truth")
                    rec = run_scenario(sc).records[-1]
                    f_lo, f_hi = (float(x) for x in spec["projector"]["band_hz"])
                    own = np.asarray(sc["platform"]["position"], dtype=float)
                    want = sorted(float(np.degrees(np.arctan2(s[0] - own[0], sign * (s[1] - own[1])))) for s in truth["sources"])
                    res["true_bearings_deg"] = want
                    ok = True
                    for label, kernels in (("eigen", False), ("kernels", True)):
                        try:
                            # With stacked rows the elevation option must be on: the rows are then handled as rows (as `elevation=1` in listen.options).
                            frames, _ = core.doa_passive(rec, f_lo, f_hi, n_fft=256, hop=128, n_average=12, first_deg=-60.0, last_deg=60.0, max_sources=2, min_snr_db=3.0, kernels=kernels,
                                                         **({"elevation": True, "elevation_step_deg": 0.25} if rows > 1 else {}))
                        except ValueError as exc:
                            if kernels and "at most 64" in str(exc):
                                res[label] = {"skipped": f"{rec.samples.n_channels} channels: {exc} (the Eigen path has no limit)"}
                                continue
                            raise
                        good, worst = 0, 0.0
                        for f in frames:
                            got = sorted(float(np.degrees(p[0])) for p in f["peaks"])
                            if len(got) == 2 and all(abs(a - b) <= 1.0 for a, b in zip(got, want)):
                                good += 1
                                worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
                        res[label] = {"frames": len(frames), "frames_with_both_within_1_deg": good, "worst_error_deg": worst}
                        ok = ok and len(frames) > 0 and good >= 0.8 * len(frames)
                    res["ok"] = bool(ok)
                elif kind == "submerged" and kind not in scenarios:
                    res["skipped"] = "the spec names no target class deeper than 10 m (targets:)"
                    res["ok"] = True
                elif kind == "submerged":
                    sc = scenarios[kind]
                    truth = sc.pop("truth")
                    records = run_scenario(sc).records
                    chain = bs.chain_config(spec, sync=False)                            # as deployed: no profile; and here no water depth either
                    chain.pop("engine", None)
                    chain["tracker"] = dict(chain.get("tracker") or {}, confirm_hits=2)
                    if chain.get("elevation"):
                        chain["elevation"] = dict(chain["elevation"], water_depth=None)
                        chain["elevation"].pop("sector_from_depth", None)
                    ok = True
                    for engine in engines:
                        reports = _run(records, chain, engine)
                        last = reports[-1]
                        movers = [t for t in last["tracks"] if t["kind"] == "mover"]
                        e = {"tracks": sorted(t["kind"] for t in last["tracks"]), "mover_speed_m_s": [t["speed_m_s"] for t in movers],
                             "mover_heading_deg": [float(np.degrees(np.arctan2(t.get("vy", 0.0), t["vx"]))) for t in movers],
                             "depth_m": [t.get("depth_m") for t in movers], "depth_sigma_m": [t.get("depth_sigma_m") for t in movers],
                             "depth_profile_source": [t.get("depth_profile_source") for t in movers],
                             "detections_with_several_possible_elevations": sum(1 for r in reports for dd in r["detections"] if dd.get("elevation_alternatives_deg")),
                             "first_ping_with_a_track": next((i + 1 for i, r in enumerate(reports) if r["tracks"]), None)}
                        good = [t for t in movers if abs(t["speed_m_s"] - truth["speed_m_s"]) <= 1.0
                                and abs(np.degrees(np.angle(np.exp(1j * (np.arctan2(t.get("vy", 0.0), t["vx"]) - np.radians(truth["heading_deg"])))))) <= 15.0]
                        honest = all(t.get("depth_m") is None or not t.get("depth_sigma_m") or abs(t["depth_m"] - truth["depth_m"]) <= 2.0 * t["depth_sigma_m"] for t in good)
                        e["true_depth_within_two_sigma"] = bool(honest)
                        e["ok"] = bool(good) and honest
                        res[engine] = e
                        ok = ok and e["ok"]
                    res["ok"] = bool(ok)
                else:
                    sc = scenarios[kind]
                    truth = sc.pop("truth")
                    records = run_scenario(sc).records
                    chain = bs.chain_config(spec, sync=bool(truth.get("sync")))
                    chain.pop("engine", None)
                    chain["tracker"] = dict(chain.get("tracker") or {}, confirm_hits=2)
                    if chain.get("elevation"):
                        chain["elevation"] = dict(chain["elevation"], gradient_per_s=None, gradient_sigma_per_s=0.0, water_depth=truth.get("water_depth"))   # the simulated water is isovelocity
                        if truth.get("water_depth") is None:
                            chain["elevation"].pop("sector_from_depth", None)
                    ok = True
                    for engine in engines:
                        reports = _run(records, chain, engine)
                        last = reports[-1]
                        e: dict = {"detections": len(last["detections"]), "seconds_per_ping": float(np.median([r["processing_seconds"] for r in reports]))}
                        if kind == "passing":
                            movers = [t for t in last["tracks"] if t["kind"] == "mover"]
                            e["tracks"] = sorted(t["kind"] for t in last["tracks"])
                            e["mover_speed_m_s"] = [t["speed_m_s"] for t in movers]
                            e["ok"] = bool(movers) and any(abs(t["speed_m_s"] - truth["speed_m_s"]) <= 1.0 for t in movers)
                        else:
                            allowance = 0.7 if kind in ("shallow", "rehearsal") else 0.0
                            # With boundaries a detection is a cluster of the direct echo and its first bounces, a few metres deep.
                            e["targets"] = [_check_target(last, t, rows if truth.get("check_depth") else 1, allowance, sign, range_tolerance=6.0 if allowance else 3.0) for t in truth["targets"]]
                            e["ok"] = all(t["ok"] for t in e["targets"])
                            if truth.get("sync"):
                                e["sync_ok"], e["sync_worst_residual_us"] = bool(last["sync"]["ok"]), float(last["sync"]["worst_residual_us"])
                                e["ok"] = e["ok"] and e["sync_ok"] and e["sync_worst_residual_us"] < 1.0
                        res[engine] = e
                        ok = ok and e["ok"]
                    res["ok"] = bool(ok)
            except Exception as exc:  # noqa: BLE001  a self-test reports what broke and goes on to the next
                res["ok"], res["error"] = False, f"{type(exc).__name__}: {exc}"
            res["seconds"] = round(time.time() - t0, 1)
            results["tests"][kind] = res
            log(f"  {spec['build']:24s} {kind:10s} {'pass' if res['ok'] else 'FAIL'}  ({res['seconds']} s)" + (f"  {res['error']}" if res.get("error") else ""))
    finally:
        config.remove_config_dir(out / "configs")
    results["ok"] = all(r["ok"] for r in results["tests"].values())
    (out / "self_test.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    return results


def _stream_test(spec: dict) -> dict:
    """Two short pings at the boards' rate as board blocks: both assemblers, and UDP loopback."""
    from gpusonar import core
    from gpusonar.sim import Scene, make_pose, simulate_ping
    from gpusonar.sonar import SonarModel
    from gpusonar.sources import FlankBoardEmulator, PingAssembler, make_source
    n = bs.names(spec)
    r0 = float(bs.window_range(spec)[0]) + 100.0
    over = {"receive": {"window": {"range": [r0, r0 + 60.0]}}}
    sonar = SonarModel.load(n["sonar"], over)
    rng = np.random.default_rng(3)
    recs = [simulate_ping(sonar, make_pose(position=(0.0, 0.0, 10.0)), Scene().add_points([[20.0, r0 + 30.0, 12.0]], target_strength_db=10.0), environment={"sound_speed": 1500.0},
                          sequence=i, time_ns=int(3e9 * i), noise_rms=1e-5, rng=rng, adc=False) for i in range(2)]
    n_ch = recs[0].samples.n_channels
    lsb = recs[0].metadata.adc_full_scale / 2 ** 23

    def same(got):
        # half a count, plus what single precision cannot hold at the record's largest sample (the earlier fixed 0.001 count
        # was less than that once levels rose 3 dB: D-93)
        slack = 0.501 * lsb + 2.0 * float(np.spacing(np.float32(max(np.abs(w.samples.array).max() for w in recs))))
        return bool(len(got) == len(recs) and all(g.samples.array.shape == w.samples.array.shape and np.abs(g.samples.array - w.samples.array).max() <= slack for g, w in zip(got, recs)))

    from gpusonar.sources.flank_stream import block_samples_for
    block = block_samples_for(n_ch)                             # the stream's payload budget, whatever the channel count
    emulator = FlankBoardEmulator(recs, 3.0, block_samples=block)
    blocks = list(emulator.blocks())
    out = {"channels": n_ch, "boards": len(sonar.boards), "block_samples": block, "block_bytes": len(blocks[0])}
    assembler = PingAssembler(sonar)
    out["python_assembler_ok"] = same([r for b in blocks for r in assembler.feed(b)])
    cpp = core.PingAssemblerCpp(sonar.make_record(make_pose()))
    out["cpp_assembler_ok"] = same([r for b in blocks for r in cpp.feed(b)])
    source = make_source({"type": "flank_udp", "sonar": n["sonar"], "sonar_overrides": over, "host": "127.0.0.1", "port": 0, "max_pings": len(recs), "timeout": 20.0})
    try:
        sender = emulator.send(source.port, pace=0.0)
        got = list(source)
        sender.join(timeout=60.0)
        out["udp_loopback_ok"] = same(got)
        out["udp_lost_samples"] = int(source.assembler.lost_samples)
    finally:
        source.close()
    out["ok"] = bool(out["python_assembler_ok"] and out["cpp_assembler_ok"] and out["udp_loopback_ok"] and out["udp_lost_samples"] == 0)
    return out


def field_table() -> str:
    """The spec's fields as a Markdown table (docs/REFERENCE_DESIGN.md holds it; a test keeps the two together)."""
    L = ["| Field | Unit | Needed | Meaning | Where a spec sheet has it | If unknown |", "| --- | --- | --- | --- | --- | --- |"]
    for section, fields in bs.SCHEMA.items():
        for name, (kind, unit, required, meaning, where, unknown) in fields.items():
            if isinstance(kind, tuple):
                meaning = f"{meaning} (one of: {', '.join(kind)})"
            L.append(f"| `{section}.{name}` | {unit or '—'} | {'yes' if required else 'no'} | {meaning} | {where or '—'} | {unknown} |")
    return "\n".join(L)


def planning_table() -> str:
    """The fields of a target class and of an environment (`targets:`, `environments:` of a spec) as a Markdown table."""
    from gpusonar import depthplan, target_classes
    L = ["| Field | Unit | Needed | Meaning | Where to find it | If unknown |", "| --- | --- | --- | --- | --- | --- |"]
    for section, fields in (("targets[]", target_classes.FIELDS), ("environments[]", depthplan.ENVIRONMENT_FIELDS)):
        for name, (kind, unit, required, meaning, where, unknown) in fields.items():
            if isinstance(kind, tuple):
                meaning = f"{meaning} (one of: {', '.join(kind)})"
            L.append(f"| `{section}.{name}` | {unit or '—'} | {'yes' if required else 'no'} | {meaning} | {where or '—'} | {unknown} |")
    return "\n".join(L)


# --- the shipped builds -------------------------------------------------------------------------------

def shipped_files() -> dict:
    """{path under configs/: text} of the sonar and deployment files of every shipped build."""
    out = {}
    for name in bs.shipped():
        spec = bs.load(name)
        if spec.get("shipped_configs") is False:
            continue
        bs.check(spec)
        for rel, text in bs.files(spec).items():
            if not rel.startswith("scenarios/"):
                out[rel] = text
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--template", action="store_true", help="also write template.h5 (one empty record of the full receive window)")
    ap.add_argument("--no-diagrams", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--only", help="self-tests to run, comma separated: " + ", ".join(KINDS))
    ap.add_argument("--engines", default="python,cpp")
    ap.add_argument("--fields", action="store_true", help="print the table of the spec's fields")
    ap.add_argument("--write-shipped", action="store_true")
    ap.add_argument("--check-shipped", action="store_true")
    args = ap.parse_args(argv)
    if args.fields:
        print(field_table() + "\n\n" + planning_table())
        return 0
    if args.write_shipped or args.check_shipped:
        bad = []
        for rel, text in shipped_files().items():
            p = ROOT / "configs" / rel
            if args.write_shipped:
                p.write_text(text)
                print("wrote", p.relative_to(ROOT))
            elif not p.exists() or p.read_text() != text:
                bad.append(rel)
        if bad:
            print("not what the specs generate (run tools/build_from_spec.py --write-shipped):", ", ".join(bad))
            return 1
        return 0
    if not args.spec:
        ap.error("give a spec (a name under configs/builds or a path)")
    spec = bs.load(args.spec)
    issues = bs.validate(spec)
    for i in issues:
        print(i)
    if any(i.level == "error" for i in issues):
        return 1
    if args.validate:
        print(f"{spec['build']}: valid ({len(issues)} warning{'s' if len(issues) != 1 else ''})")
        return 0
    if not args.out:
        ap.error("give --out DIR (or --validate)")
    written = generate(spec, Path(args.out), template=args.template, diagrams=not args.no_diagrams)
    print(f"{spec['build']}: {len(written)} files under {args.out}")
    if args.self_test:
        kinds = tuple(k for k in (args.only.split(",") if args.only else KINDS))
        res = self_test(spec, Path(args.out), kinds=kinds, engines=tuple(args.engines.split(",")))
        print(f"{spec['build']}: self-tests {'PASS' if res['ok'] else 'FAIL'} ({args.out}/self_test.json)")
        return 0 if res["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
