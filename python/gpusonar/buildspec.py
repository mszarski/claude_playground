"""A build of a line-array sonar described in spec-sheet terms, and everything that follows from it.

One YAML file under configs/builds/ says what a build IS: the platform it is mounted on, the rows of
hydrophones, the hydrophone, the capture boards, how the boards are synchronised, the projector, the
navigation, what is known of the sound speed and the water depth, the network, storage, compute, and
the water the first trial is in. From it this module makes

  the sonar file            configs/sonars/<name>.yaml (and <name>_clocks.yaml: every board on its own clock)
  the deployment file       configs/deploy/<name>.yaml (and the options of the C++ program)
  scenarios                 free-field truth, shallow multipath, rehearsal, passing target, passive
  the numbers               geometry, rates, storage, timing, direct sound, synchronisation tolerance
  expected performance      beam, gain, reach, elevation accuracy, repeats, depth error budget
  the parts list            quantity, requirement, what fixes it, status

and checks the spec (`validate`). tools/build_from_spec.py is the command; docs/REFERENCE_DESIGN.md the guide.

Status of a value. Every section may carry `confirmed: [field, ...]`: the fields whose values are
measured, taken from a datasheet in hand, or stated by the owner. Every other field that has a value is
`assumed`; a field left out or null is `unknown`. Assumed and unknown fields end up in the `unconfirmed`
lists of the generated files, which accumulate through `base:` (config.merge_flags).

Nothing here is specific to one sonar: the shipped Popoto builds are specs like any other.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import acoustics
from .config import config_dir, deep_merge, load_sonar_config, load_yaml

ROOT = Path(__file__).resolve().parents[2]

# --- the schema ---------------------------------------------------------------------------------------
# section -> field -> (kind, unit, required, meaning, where a spec sheet has it, what to do if unknown)
# kinds: num, int, str, bool, vec2, vec3, list, map, or a tuple of allowed words.
F = "leave it out: it is listed as unknown"
SCHEMA = {
    "platform": {
        "kind": (("vehicle", "static"), "", True, "under way (navigation gives pose and attitude) or a frame left in place (a typed-in pose)", "the platform's general arrangement", "must be given"),
        "name": ("str", "", False, "short name used in the diagrams", "", F),
        "mounting": ("str", "", False, "what the array is fixed to, in words", "general arrangement drawing", F),
        "span_m": ("num", "m", True, "length of the face or frame along the array", "general arrangement drawing", "must be given: measure it"),
        "height_m": ("num", "m", True, "height of the face or frame", "general arrangement drawing", "must be given"),
        "usable_span_m": ("num", "m", False, "length the rows may take (the rest is margin, fairing, projector)", "integration notes", "defaults to span_m"),
        "usable_height_m": ("num", "m", False, "height the rows may take", "integration notes", "defaults to height_m"),
        "depth_m": ("num", "m", False, "static only: the depth the frame hangs or stands at", "trial plan", "static: must be given (it is the typed-in pose)"),
        "max_depth_m": ("num", "m", False, "deepest the platform operates: the depth planner searches own depths no deeper, and it is the depth rating required of hydrophones, cables, connectors, housings and the projector",
                        "the platform's depth rating / the owner", "leave it out: the planner then uses each environment's limits (300 m when it has none: a placeholder) and the parts list has no rating"),
    },
    "array": {
        "module": ("str", "", False, "a row module under configs/builds/modules: a named row design (count, pitch, hydrophone, boards a row) that several builds share; fields given here win", "the project's row design", "leave out and give count and pitch_m"),
        "rows": ("int", "", True, "number of horizontal rows (1: bearing only; 2 or more: elevation and depth)", "array drawing", "must be given"),
        "count": ("int", "", True, "hydrophones in each row", "array drawing", "must be given"),
        "pitch_m": ("num", "m", False, "spacing along a row (uniform rows)", "array drawing", "give positions_m instead, or half a wavelength at the top of the band"),
        "positions_m": ("list", "m", False, "positions along the row from its centre, for a non-uniform row (instead of pitch_m)", "array drawing", "use pitch_m"),
        "heights_m": ("list", "m", False, "height of each row above the centre of the face, lowest first (rows > 1)", "array drawing", "must be given for rows > 1"),
        "out_m": ("list", "m", False, "how far each row stands out from the face (a curved hull)", "array drawing / survey", "zero: flat face"),
        "look": (("starboard", "port"), "", False, "which side the face looks to", "general arrangement", "starboard"),
        "wiring": (("row_by_row", "interleave"), "", False, "row_by_row: every board serves one row. interleave: `interleave` hydrophones of each row in turn (rows share a board's clock)", "wiring list", "row_by_row"),
        "interleave": ("int", "", False, "hydrophones of each row per turn when wiring is interleave", "wiring list", "half a board's inputs for two rows"),
        "board_position_m": ("vec3", "m", False, "where the capture boards sit, body frame [along, out, down] from the array's centre: cable lengths are measured to it", "integration notes", "the array's centre, behind the face: an assumption the parts list states"),
        "cable_slack_m": ("num", "m", False, "added to every cable run", "wiring practice", "0.5 m assumed"),
    },
    "hydrophone": {
        "model": ("str", "", False, "part name", "datasheet title", F),
        "sensitivity_db": ("num", "dB re 1 V/uPa", True, "open-circuit receive sensitivity (with its own preamp if it has one)", "datasheet: receive sensitivity (OCV)", "must be given: -208 is a small passive element; flag it assumed"),
        "noise_db": ("num", "dB re 1 uPa^2/Hz", False, "equivalent input noise at the band centre (self-noise)", "datasheet: equivalent noise pressure", "leave out: the parts list states the limit it must meet"),
        "band_hz": ("vec2", "Hz", False, "usable band", "datasheet: frequency range", "leave out: the validator then cannot check it covers the transmit band"),
        "preamp_gain_db": ("num", "dB", False, "gain of a preamp at the element (added to the board's gain)", "datasheet", "0"),
    },
    "boards": {
        "model": ("str", "", False, "part name", "datasheet title", F),
        "inputs": ("int", "", True, "hydrophone inputs per board", "datasheet: channels", "must be given"),
        "sample_rate_hz": ("num", "Hz", True, "ADC sample rate per channel", "datasheet / ICD: sample rate", "must be given; flag it assumed until the maker confirms"),
        "adc_bits": ("int", "bit", True, "sample word length", "datasheet", "must be given"),
        "gain_db": ("num", "dB", False, "preamp gain of the board's inputs", "ICD: input gain", "0"),
        "input_full_scale_vpp": ("num", "V peak-to-peak", False, "largest signal at the INPUT connector (before the board's gain)", "ICD: full-scale input", "give adc_full_scale_v instead"),
        "adc_full_scale_v": ("num", "V peak", False, "full scale referred to the gain stage's output: input peak x gain. What the records carry", "derived from the ICD", "derived from input_full_scale_vpp and gain_db"),
        "highpass_hz": ("num", "Hz", False, "corner of the input's AC coupling", "ICD", F),
        "simultaneous": ("bool", "", False, "all inputs of one board sampled on the same instant", "ICD / ask the maker", "leave out: it is an open question in the parts list"),
        "clock_tolerance_ppm": ("num", "ppm", False, "how far a board's sample clock may be off: the drift the simulated boards are given", "oscillator datasheet", "25 assumed"),
        "clock_offset_max_s": ("num", "s", False, "largest start offset between boards the simulated boards are given", "how recording is started", "0.3 ms assumed"),
        "jitter_rms_s": ("num", "s", False, "sample-clock jitter of the simulated boards", "oscillator datasheet", "20 ns assumed"),
        "clocks": ("map", "", False, "the simulated boards' clock errors: {seed: n} draws them inside the tolerance; {explicit: [[offset_s, drift_ppm], ...]} gives them, first board first", "", "{seed: 1}"),
        "overload_damage_vpp": ("num", "V peak-to-peak", False, "largest input the board survives", "the board's absolute maximum ratings", "leave out: an open question in the parts list"),
        "gain_switchable": ("bool", "", False, "the input gain can be lowered during the transmit", "ICD / ask the maker", "leave out: unknown"),
        "shared_clock": ("bool", "", False, "true: all boards run from one sample clock (no synchronisation stage)", "ICD", "false: every board on its own clock"),
    },
    "sync": {
        "scheme": (("shared_reference", "none"), "", False, "shared_reference: one electrical burst wired to every board and summed into its first input. none: one clock for all", "system design", "shared_reference when boards have their own clocks"),
        "reference": ("map", "", False, "the burst: {f_start_hz, f_stop_hz, duration_s, level_v, lead_s, tail_s} (level: V peak at the ADC)", "system design", "a 4 ms sweep in the top fifth of the Nyquist band, clear of the acoustic band"),
        "elevation_error_deg": ("num", "deg", False, "elevation error allowed from clock error between rows: sets the required agreement", "error budget", "0.05"),
    },
    "projector": {
        "model": ("str", "", False, "part name", "datasheet title", F),
        "amplifier": ("str", "", False, "power amplifier part name", "datasheet title", F),
        "source_level_db_rms": ("num", "dB re 1 uPa at 1 m, rms", True, "source level in the band as an RMS pressure level (the usual datasheet convention; a peak figure is 3 dB higher for a tone)", "datasheet: TVR x rms drive voltage, or SL", "must be given: the link budget needs it; flag it assumed"),
        "band_hz": ("vec2", "Hz", True, "transmitted band", "datasheet: usable band; waveform design", "must be given"),
        "waveform": ("map", "", True, "{type: lfm | cw, duration_s, window_alpha}", "waveform design", "must be given"),
        "position_m": ("vec3", "m", True, "body frame [along, out, down] from the array's centre", "integration drawing", "must be given: a placeholder is allowed if flagged assumed"),
        "blanking_after_s": ("num", "s", False, "how long after the end of the direct sound the inputs are still unusable (recovery from overload): the first usable range follows from it", "bench measurement of the boards", "leave out: unknown, and the validator says so when the direct sound overloads the inputs"),
    },
    "navigation": {
        "unit": ("str", "", False, "part name of the navigation / attitude source", "platform documentation", F),
        "adapter": ("str", "", False, "what turns its messages into the chain's navigation datagram (docs/DEPLOYMENT.md)", "interface document", "json_udp: the datagram itself"),
        "roll_sigma_deg": ("num", "deg", False, "roll (or frame tilt) accuracy, one sigma: goes into every elevation and depth sigma", "datasheet: roll/pitch accuracy", "vehicle 0.2, frame without a tilt sensor 1.0: placeholders, flagged"),
        "rate_hz": ("num", "Hz", False, "output rate of attitude", "interface document", F),
        "timestamp_sigma_s": ("num", "s", False, "accuracy of its time stamps against the boards' time", "interface document", F),
        "attitude_time": (("arrival", "transmit"), "", False, "use attitude as it was when each echo arrived (needs samples through the ping) or at the transmit", "", "vehicle: arrival; typed-in pose: transmit"),
        "attitude_rate_sigma_deg_s": ("num", "deg/s", False, "sigma added per second an echo lies beyond the attitude samples", "", "0.5 assumed"),
        "static_pose": ("map", "", False, "static only: {x, y, z, roll, pitch, yaw} north, east, down in m; radians", "trial plan", "level at depth_m, heading north"),
        "tilt_sensor": ("bool", "", False, "static only: the frame sends navigation datagrams from a tilt and compass sensor", "frame design", "false: the typed-in pose"),
    },
    "sound_speed": {
        "nominal_m_s": ("num", "m/s", False, "value used when nothing better is known", "", "1500"),
        "rung": (("local_sensor", "typed", "gradient_prior", "profile"), "", False, "the best knowledge available: local_sensor (navigation sends it), typed (measured once), gradient_prior (a mean gradient and its sigma), profile (a measured profile)", "platform sensors / trial plan", "gradient_prior"),
        "value_m_s": ("num", "m/s", False, "typed: the measured local sound speed", "CTD / SVP cast", F),
        "gradient_per_s": ("num", "1/s", False, "assumed mean vertical gradient along the path (0.017: isothermal)", "climatology", "0.017, flagged"),
        "gradient_sigma_per_s": ("num", "1/s", False, "how wrong that may be: range^2 x this / 2c into every depth sigma", "climatology", "0.05, flagged"),
        "profile": ("map", "", False, "{depth: [...], speed: [...]} a measured profile", "SVP cast", F),
    },
    "water_depth": {
        "source": (("navigation", "typed", "unknown"), "", False, "where the chain gets it: the navigation datagram, a typed value, or nowhere", "platform sensors (altitude + depth)", "unknown: no folding of bottom bounces, no depth bound for two rows"),
        "value_m": ("num", "m", False, "typed: the water depth", "chart / echo sounder", F),
        "sigma_m": ("num", "m", False, "how well it is known", "", "5, flagged"),
    },
    "network": {
        "stream_port": ("int", "", False, "UDP port the boards' blocks arrive on", "ICD", "5700"),
        "nav_port": ("int", "", False, "UDP port of the navigation datagrams", "ICD", "5702"),
        "report_udp": ("list", "", False, "[host, port] the report lines go to", "ICD", "[127.0.0.1, 5701]"),
    },
    "storage": {
        "recording": ("str", "", False, "path of the raw recording (strftime fields allowed)", "", "/data/sonar/<build>_%Y%m%d_%H%M%S.h5"),
        "report": ("str", "", False, "path of the report", "", "the same with .jsonl"),
        "taps_dir": ("str", "", False, "directory for the stage taps of the first pings", "", "/data/sonar/taps"),
        "taps_pings": ("int", "", False, "pings tapped", "", "2"),
        "capacity_gb": ("num", "GB", False, "size of the recording volume", "computer's specification", F),
        "record": (("raw", "decimated"), "", False, "what the recording holds. raw: every sample of every board, the synchronisation reference included (everything can be redone; the chain decimates when compute.option says so). "
                   "decimated: the receiver decimates as blocks arrive (`source.decimate`) and only the band is kept: the reference and everything outside the band are gone for good; for storage-limited runs",
                   "the trial plan", "raw: the safe default during trials"),
    },
    "compute": {
        "option": (("decimate_first_cpu", "gpu", "cpu"), "", False, "how real time is reached at the boards' full rate. decimate_first_cpu: the chain converts to baseband and decimates first (`chain.decimate`), on a CPU. "
                   "gpu: the map and the elevation search on a GPU. cpu: the chain as it is", "", "two or more rows: gpu (decimate-first on a CPU is slower than the ping interval over the full window); one row: decimate_first_cpu"),
        "unit": ("str", "", False, "part name of the computer", "datasheet title", F),
        "engine": (("python", "cpp"), "", False, "which engine tools/sonar_live.py runs (the standalone C++ program's command is always generated)", "", "python"),
        "backend": (("auto", "host", "cuda"), "", False, "kernels on the CPU or the GPU", "", "auto"),
    },
    "operation": {
        "range_m": ("vec2", "m", True, "[first, last] range the receive window covers; first null: the blind range of the pulse", "requirement", "must be given"),
        "bearings_deg": ("list", "deg", False, "[first, last, step] from broadside", "requirement", "[-60, 60, 0.5]"),
        "chain": ("map", "", False, "settings of the chain laid over the defaults (docs/CONFIGURATION.md)", "", "the defaults"),
        "coverage": ("map", "", False, "{environment: one of the build's `environments` by name, or an environment mapping}: the deployment file gets `chain.coverage` (what the sonar can see "
                     "from where it is) for that site. The site must state its water depth, bottom and wind: none has a default", "the site the build is deployed at",
                     "leave it out: no coverage picture"),
        "detection_probability": ("num", "", False, "the probability of detecting a steady point echo on one ping that the link budget's thresholds, margins and reaches are stated for "
                                  "(the threshold follows from the chain's detector for it: buildspec.detection)", "requirement", "0.5 (buildspec.DESIGN_DETECTION_PROBABILITY)"),
    },
    "trial": {
        "water_depth_m": ("num", "m", True, "water depth of the simulated trial", "trial plan", "must be given"),
        "own_depth_m": ("num", "m", True, "depth of the array in the trial", "trial plan", "must be given"),
        "ranges_m": ("list", "m", True, "target ranges: the first for the free-field truth check, the last for the shallow-water one", "trial plan", "must be given, beyond the pulse's blind range"),
        "target_depths_m": ("list", "m", False, "target depths for the truth checks", "trial plan", "above, level with and below the array"),
        "target_strength_db": ("num", "dB", False, "strength of the simulated targets", "", "10"),
        "wind_speed_m_s": ("num", "m/s", False, "sea state as a wind speed, for the ambient noise", "trial plan", "5"),
        "sound_speed_m_s": ("num", "m/s", False, "sound speed of the simulated water (isovelocity)", "trial plan", "nominal"),
        "pulse_s": ("num", "s", False, "pulse length used in the long-range scenarios, when the sonar's is too short for them", "", "the projector's"),
        "roll_deg": ("num", "deg", False, "roll (frame: tilt) in the rehearsal", "", "vehicle 3, frame 2"),
        "roll_rate_deg_s": ("num", "deg/s", False, "vehicle only: roll rate during the ping in the rehearsal", "", "2"),
        "speed_m_s": ("num", "m/s", False, "vehicle only: speed", "", "1.5"),
        "bottom": ("map", "", False, "{speed, density, attenuation} of the sea bed for the shallow scenarios", "site survey", "sand: 1650 m/s, 1.9, 0.8 dB/wavelength: textbook values, flagged"),
        "seed": ("int", "", False, "seed of the scenarios' noise", "", "1"),
    },
}
TOP = {"build": "name of the build: file names follow it", "title": "one line for headings", "stage": "label of the stage of a staged plan",
       "base_sonar": "a shipped sonar file the generated one extends (only what differs is written)",
       "names": "{sonar, deploy}: file names when they must differ from the build's name",
       "targets": "target classes the build is planned against: a list ({name, target_strength_db or cylinder, depth_m, speed_m_s, status, note}) or the name of a file under configs/builds/targets/ "
                  "(gpusonar.target_classes.FIELDS)",
       "environments": "waters the build is planned in: a list ({name, water_depth_m, bottom, wind_speed_m_s or sea_state, rms_wave_height_m, bubble_model, profile, own_depth_limits_m, ...}) or the name of a file "
                       "under configs/builds/environments/ (gpusonar.depthplan.ENVIRONMENT_FIELDS)"}
ROW_WORDS = {2: "two_rows", 3: "three_rows", 4: "four_rows"}
# Thresholds of the chain typed here for the long-range array (2 m cells, 100 m guard, 300 m training, tracker and classifier
# gates). The chain is being changed to derive them from the sonar's band and pulse; when that is merged this dictionary should
# shrink to `bearings_deg` and `look`, and a build's own values go in `operation.chain` (overrides only). Used in one place:
# `chain_config`.
# Only what a build decides. The range cell, CFAR windows, SNR floor, tracker and classifier sizes are the chain's own:
# it derives or defaults them from the sonar (libs/live derive.hpp, gpusonar.live), and a second copy here would drift.
DEFAULT_CHAIN = {"bearings_deg": [-60.0, 60.0, 0.5], "look": "starboard"}
# What a deployment file's `chain.multipath_depth` holds wherever a build switches it on (`operation.chain.multipath_depth`;
# the build's own keys are laid over these). No shipped build switches it on.
MULTIPATH_DEPTH_DEPLOYED = {"rays": True, "withhold_flagged": True}
# The link budget has NO typed detection threshold. Its SNR is that of ONE row (peak signal over mean noise after the
# matched filter and an unweighted row beamformer, RMS source level); what that SNR must be for a detection is asked of
# the chain's own detector for the build's deployed settings: `detection(d)` (gpusonar.live.budget_threshold_db and
# detection_probability). The detector's figure already holds the aperture window's loss, where the echo falls between
# the map's samples, the false-alarm budget per ping over the build's own window, and the rows added in power; so the
# budget adds no row gain of its own. What is stated instead of a threshold is the detection probability on one ping
# that "detected" stands for: a build's `operation.detection_probability`, by default this.
DESIGN_DETECTION_PROBABILITY = 0.5
# The second probability every table quotes beside the design one (a threshold for "nine pings in ten").
HIGH_DETECTION_PROBABILITY = 0.9
DEFAULT_REFERENCE = {"f_start_hz": 38.0e3, "f_stop_hz": 48.0e3, "duration_s": 4.0e-3, "level_v": 2.0e-3, "lead_s": 5.0e-3, "tail_s": 10.0e-3}
C_ASSUMED_BOTTOM = {"speed": 1650.0, "density": 1.9, "attenuation": 0.8}


@dataclass
class Issue:
    level: str      # "error" or "warning"
    where: str      # section.field
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


class SpecError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("build spec is not valid:\n  " + "\n  ".join(str(i) for i in issues if i.level == "error"))


def load(path) -> dict:
    """A build spec from a file under configs/builds (a name) or a path."""
    p = Path(str(path))
    if not p.exists():
        p = config_dir() / "builds" / f"{path}.yaml"
    if not p.exists():
        known = sorted(q.stem for q in (config_dir() / "builds").glob("*.yaml"))
        raise FileNotFoundError(f"no build spec {path!r}; known: {', '.join(known)}")
    return resolve_module(load_yaml(p))


def resolve_module(spec: dict) -> dict:
    """Lay the spec over its row module (`array.module`): the module gives `array` and `hydrophone` fields (and
    `confirmed` lists), the spec's own fields win. The module's name stays in the spec."""
    name = (spec.get("array") or {}).get("module")
    if not name:
        return spec
    path = config_dir() / "builds" / "modules" / f"{name}.yaml"
    if not path.exists():
        known = sorted(q.stem for q in (config_dir() / "builds" / "modules").glob("*.yaml"))
        raise FileNotFoundError(f"no row module {name!r}; known: {', '.join(known)}")
    module = load_yaml(path)
    out = copy.deepcopy(spec)
    for section in ("array", "hydrophone"):
        mod, mine = module.get(section) or {}, spec.get(section) or {}
        merged = {**mod, **mine}
        confirmed = [k for k in (mod.get("confirmed") or []) if k not in mine or mine.get(k) == mod.get(k)] + [k for k in (mine.get("confirmed") or [])]
        merged.pop("confirmed", None)
        if confirmed:
            merged["confirmed"] = sorted(set(confirmed))
        if merged:
            out[section] = merged
    return out


def modules() -> dict:
    """The shipped row modules: {name: mapping}."""
    return {q.stem: load_yaml(q) for q in sorted((config_dir() / "builds" / "modules").glob("*.yaml"))}


def shipped() -> list[str]:
    return sorted(p.stem for p in (config_dir() / "builds").glob("*.yaml"))


def get(spec: dict, section: str, key: str, default=None):
    v = (spec.get(section) or {}).get(key)
    return default if v is None else v


def status(spec: dict, section: str, key: str) -> str:
    """`confirmed`, `assumed` (a value is there but nobody vouches for it) or `unknown` (no value)."""
    sec = spec.get(section) or {}
    if sec.get(key) is None:
        return "unknown"
    return "confirmed" if key in (sec.get("confirmed") or []) else "assumed"


# --- validation ---------------------------------------------------------------------------------------

def _kind_ok(kind, v) -> bool:
    num = isinstance(v, (int, float)) and not isinstance(v, bool)
    if isinstance(kind, tuple):
        return v in kind
    return {"num": num, "int": isinstance(v, int) and not isinstance(v, bool), "str": isinstance(v, str), "bool": isinstance(v, bool),
            "list": isinstance(v, list), "map": isinstance(v, dict),
            "vec2": isinstance(v, list) and len(v) == 2, "vec3": isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)}[kind]


def validate(spec: dict) -> list[Issue]:
    """Every problem found, errors and warnings. An empty list of errors means the generator will run."""
    out: list[Issue] = []

    def err(where, msg):
        out.append(Issue("error", where, msg))

    def warn(where, msg):
        out.append(Issue("warning", where, msg))

    if not isinstance(spec.get("build"), str) or not spec["build"].replace("_", "").isalnum():
        err("build", "give the build a name of letters, digits and underscores (file names follow it)")
    for key in spec:
        if key not in SCHEMA and key not in TOP:
            err(key, f"unknown section; sections are {', '.join(SCHEMA)}")
    for section, fields in SCHEMA.items():
        sec = spec.get(section)
        if sec is None:
            if any(f[2] for f in fields.values()):
                err(section, "section missing; it must give " + ", ".join(k for k, f in fields.items() if f[2]))
            continue
        if not isinstance(sec, dict):
            err(section, "must be a mapping")
            continue
        for key, v in sec.items():
            if key == "confirmed":
                for name in (v or []):
                    if name not in fields:
                        err(f"{section}.confirmed", f"{name!r} is not a field of {section}")
                    elif sec.get(name) is None:
                        err(f"{section}.confirmed", f"{name!r} is marked confirmed but has no value")
                continue
            if section == "projector" and key == "source_level_db":
                err("projector.source_level_db", "renamed projector.source_level_db_rms: source levels are RMS levels (dB re 1 uPa at 1 m). If the figure you hold is a peak level, subtract 3 dB")
            elif key not in fields:
                err(f"{section}.{key}", f"unknown field; fields are {', '.join(fields)}")
            elif v is not None and not _kind_ok(fields[key][0], v) and not (section == "operation" and key == "range_m" and isinstance(v, list) and len(v) == 2):
                kind = fields[key][0]
                want = "one of " + ", ".join(kind) if isinstance(kind, tuple) else {"num": "a number", "int": "a whole number", "vec2": "[a, b]", "vec3": "[x, y, z]"}.get(kind, f"a {kind}")
                err(f"{section}.{key}", f"is {v!r}; it must be {want}" + (f" ({fields[key][1]})" if fields[key][1] else ""))
        for key, f in fields.items():
            if f[2] and sec.get(key) is None:
                err(f"{section}.{key}", f"missing: {f[3]}" + (f" ({f[1]})" if f[1] else "") + f". If unknown: {f[5]}")
    # target classes and environments (planning inputs: they change no generated sonar or deployment file)
    from . import depthplan, target_classes
    for key, load_items, find_problems in (("targets", target_classes.load, target_classes.problems), ("environments", depthplan.load_environments, depthplan.environment_problems)):
        if spec.get(key) is None:
            continue
        try:
            items = load_items(spec[key]) if isinstance(spec[key], (str, list)) else spec[key]
        except FileNotFoundError as exc:
            err(key, str(exc))
            continue
        for where, msg in find_problems(items):
            err(where, msg)
    if get(spec, "operation", "coverage") is not None and not any(i.level == "error" for i in out):
        from . import coverage                              # the generator's own refusal (chain_config), said here so that --validate reports it
        try:
            coverage.of_build(spec, spec["operation"]["coverage"])
        except ValueError as exc:
            err("operation.coverage", str(exc))
    if any(i.level == "error" for i in out):
        return out

    # --- consistency: only reached when every field is of its kind ---
    p, a, h, b, pj, op, tr = (spec[k] for k in ("platform", "array", "hydrophone", "boards", "projector", "operation", "trial"))
    c = float(get(spec, "sound_speed", "nominal_m_s", acoustics.NOMINAL_SOUND_SPEED))
    rows, count = a["rows"], a["count"]
    if rows < 1 or count < 2:
        err("array.rows", "at least one row of at least two hydrophones")
        return out
    f_lo, f_hi = (float(x) for x in pj["band_hz"])
    if not 0 < f_lo < f_hi:
        err("projector.band_hz", f"[{f_lo:g}, {f_hi:g}] Hz: give [lowest, highest] with lowest > 0 (hertz, not kilohertz)")
        return out
    if f_hi < 100.0:
        warn("projector.band_hz", f"{f_hi:g} Hz at the top of the band: is this in kHz? The field is in Hz")
    lam_min = c / f_hi
    if (a.get("pitch_m") is None) == (a.get("positions_m") is None):
        err("array.pitch_m", "give exactly one of pitch_m (uniform rows) and positions_m")
        return out
    if a.get("positions_m") is not None and len(a["positions_m"]) != count:
        err("array.positions_m", f"{len(a['positions_m'])} positions for count {count}")
        return out
    along = np.asarray(a["positions_m"], dtype=float) if a.get("positions_m") is not None else (np.arange(count) - (count - 1) / 2.0) * float(a["pitch_m"])
    length = float(np.ptp(along))
    pitch = float(np.min(np.diff(np.sort(along))))
    if pitch <= 0:
        err("array.pitch_m", "hydrophones of a row must be at different positions")
        return out
    if a.get("pitch_m") is not None and float(a["pitch_m"]) > 5.0:
        warn("array.pitch_m", f"{a['pitch_m']} m between hydrophones: is this in mm or cm? The field is in metres")
    # pitch against the band: grating lobes
    steer = max(abs(float(x)) for x in get(spec, "operation", "bearings_deg", DEFAULT_CHAIN["bearings_deg"])[:2])
    if a.get("pitch_m") is not None:
        free = lam_min / float(a["pitch_m"]) - 1.0
        if free <= 0:
            err("array.pitch_m", f"{a['pitch_m']} m is {float(a['pitch_m']) / lam_min:.2f} wavelengths at {f_hi:g} Hz: a full repeat of the beam (grating lobe) is in view even at broadside. "
                                 f"Half a wavelength is {lam_min / 2:.3f} m")
        elif np.degrees(np.arcsin(min(free, 1.0))) < steer:
            warn("array.pitch_m", f"{a['pitch_m']} m is {float(a['pitch_m']) / lam_min:.2f} wavelengths at {f_hi:g} Hz: no grating lobe only within +/-{np.degrees(np.arcsin(min(free, 1.0))):.0f} deg of broadside, "
                                  f"but operation.bearings_deg steers to {steer:g} deg. Half a wavelength is {lam_min / 2:.3f} m")
    # rows inside the usable zone
    usable = float(p.get("usable_span_m") or p["span_m"])
    usable_h = float(p.get("usable_height_m") or p["height_m"])
    if usable > float(p["span_m"]) + 1e-9:
        err("platform.usable_span_m", f"{usable:g} m is longer than the platform's span, {p['span_m']:g} m")
    if length > usable + 1e-9:
        err("array.count", f"a row is {length:.2f} m long (centre to centre of the end hydrophones); the usable span is {usable:g} m. "
                           f"At this pitch {int(usable // pitch) + 1} hydrophones fit")
    heights = [float(x) for x in (a.get("heights_m") or [0.0])]
    if rows > 1 and len(a.get("heights_m") or []) != rows:
        err("array.heights_m", f"{rows} rows need {rows} heights (m above the centre of the face, lowest first); got {a.get('heights_m')}")
        return out
    if rows == 1 and len(heights) != 1:
        err("array.heights_m", "one row has one height (or leave heights_m out)")
        return out
    if sorted(heights) != heights or len(set(heights)) != len(heights):
        err("array.heights_m", f"{heights}: list the rows from the lowest up, each at its own height")
    if max(abs(x) for x in heights) > usable_h / 2 + 1e-9:
        err("array.heights_m", f"a row at {max(heights, key=abs):+g} m is outside the usable height ({usable_h:g} m: +/-{usable_h / 2:g} m about the centre)")
    if a.get("out_m") is not None and len(a["out_m"]) != rows:
        err("array.out_m", f"{rows} rows need {rows} values")
    # boards
    inputs = int(b["inputs"])
    if inputs < 1:
        err("boards.inputs", "at least one input per board")
        return out
    wiring = a.get("wiring", "row_by_row")
    if wiring == "interleave":
        n_int = a.get("interleave")
        if rows < 2:
            err("array.wiring", "interleave needs at least two rows")
        elif n_int is None or n_int < 1 or count % n_int or (rows * n_int) % inputs:
            err("array.interleave", f"give how many hydrophones of each row go in turn: it must divide the row ({count}) and {rows} x it must fill whole boards of {inputs} inputs")
    elif count % inputs:
        warn("boards.inputs", f"{count} hydrophones a row on boards of {inputs} inputs: {int(np.ceil(count / inputs))} boards a row with {int(np.ceil(count / inputs)) * inputs - count} input(s) spare "
                              "(each board still serves one row)")
    # sampling against the band
    fs = float(b["sample_rate_hz"])
    if fs < 2.0 * f_hi:
        err("boards.sample_rate_hz", f"{fs:g} Hz cannot sample a band reaching {f_hi:g} Hz (needs more than {2 * f_hi:g} Hz)")
    elif fs < 2.5 * f_hi:
        warn("boards.sample_rate_hz", f"{fs:g} Hz is only {fs / f_hi:.2f} times the top of the band ({f_hi:g} Hz): the anti-alias filter has little room")
    if not 8 <= int(b["adc_bits"]) <= 32:
        err("boards.adc_bits", f"{b['adc_bits']}: a sample word of 8 to 32 bits")
    elif int(b["adc_bits"]) != 24:
        warn("boards.adc_bits", f"{b['adc_bits']} bits: the board stream (sources/flank_stream.py) carries 24-bit samples; a board with another word length needs its own packing")
    if b.get("input_full_scale_vpp") is None and b.get("adc_full_scale_v") is None:
        err("boards.input_full_scale_vpp", "give the full-scale input (V peak-to-peak at the connector) or adc_full_scale_v")
        return out
    gain = float(b.get("gain_db") or 0.0)
    if b.get("input_full_scale_vpp") is not None and b.get("adc_full_scale_v") is not None:
        implied = 0.5 * float(b["input_full_scale_vpp"]) * 10.0 ** (gain / 20.0)
        if abs(implied / float(b["adc_full_scale_v"]) - 1.0) > 0.02:
            err("boards.adc_full_scale_v", f"{b['adc_full_scale_v']} V contradicts input_full_scale_vpp {b['input_full_scale_vpp']} V with {gain:g} dB of gain, which is {implied:.3f} V peak. Give one, or make them agree")
    if b.get("highpass_hz") is not None and float(b["highpass_hz"]) > 0.5 * f_lo:
        warn("boards.highpass_hz", f"the inputs' high-pass corner ({b['highpass_hz']:g} Hz) is close to the bottom of the band ({f_lo:g} Hz)")
    if h.get("band_hz") is not None:
        hb = [float(x) for x in h["band_hz"]]
        if hb[0] > f_lo or hb[1] < f_hi:
            err("hydrophone.band_hz", f"[{hb[0]:g}, {hb[1]:g}] Hz does not cover the transmitted band [{f_lo:g}, {f_hi:g}] Hz")
    if not -260.0 < float(h["sensitivity_db"]) < -100.0:
        warn("hydrophone.sensitivity_db", f"{h['sensitivity_db']} dB re 1 V/uPa is outside anything usual (-220 to -150): check the unit (dB re 1 V/uPa, a negative number)")
    # synchronisation
    own_clocks = not b.get("shared_clock", False)
    scheme = get(spec, "sync", "scheme", "shared_reference" if own_clocks else "none")
    n_boards = len(board_table(spec))
    if own_clocks and n_boards > 1 and scheme == "none":
        err("sync.scheme", f"{n_boards} boards on their own clocks need a synchronisation scheme (shared_reference), or set boards.shared_clock: true")
    if scheme == "shared_reference":
        ref = reference_of(spec)
        if ref["f_stop_hz"] >= 0.5 * fs or ref["f_start_hz"] >= ref["f_stop_hz"]:
            err("sync.reference", f"the reference burst ({ref['f_start_hz']:g}-{ref['f_stop_hz']:g} Hz) must lie below half the sample rate ({0.5 * fs:g} Hz)")
        elif ref["f_start_hz"] < f_hi * 1.2:
            err("sync.reference", f"the reference burst ({ref['f_start_hz']:g}-{ref['f_stop_hz']:g} Hz) must be clear of the acoustic band (top {f_hi:g} Hz): it is summed into a hydrophone input")
        if spec.get("sync", {}).get("reference") is None and "sync" in spec:
            pass
    clocks = b.get("clocks") or {}
    if "explicit" in clocks and len(clocks["explicit"]) != n_boards:
        err("boards.clocks", f"{len(clocks['explicit'])} explicit clock errors for {n_boards} boards")
    # projector and waveform
    wf = pj["waveform"]
    if wf.get("type") not in ("lfm", "cw") or wf.get("duration_s") is None:
        err("projector.waveform", "give {type: lfm | cw, duration_s: ...}")
        return out
    pulse = float(wf["duration_s"])
    if not 120.0 < float(pj["source_level_db_rms"]) < 240.0:
        warn("projector.source_level_db_rms", f"{pj['source_level_db_rms']} dB re 1 uPa at 1 m is outside anything usual (150 to 230)")
    pos = np.asarray(pj["position_m"], dtype=float)
    if abs(pos[0]) > float(p["span_m"]) / 2 + 1e-9 or abs(pos[2]) > float(p["height_m"]) / 2 + 1e-9:
        warn("projector.position_m", f"{[float(x) for x in pos]} is outside the platform's face ({p['span_m']:g} m x {p['height_m']:g} m about the array's centre): hung apart?")
    d = derive(spec)
    if d["direct"]["nearest_m"] < 0.05:
        err("projector.position_m", "the projector is on top of a hydrophone")
    elif d["direct"]["over_full_scale_db"] > 0.0:
        dr = d["direct"]
        warn("projector.position_m", f"the projector's direct sound is {dr['level_db']:.0f} dB re 1 uPa at the nearest hydrophone ({dr['nearest_m']:.2f} m away): "
                                     f"{dr['volts']:.1f} V at the ADC, {dr['over_full_scale_db']:.0f} dB over full scale ({d['adc_full_scale_v']:.2f} V) for the whole pulse; "
                                     f"{dr['hydrophones_overloaded']} of {d['channels']} hydrophones are overloaded ({dr['over_full_scale_farthest_db']:+.0f} dB at the farthest, {dr['farthest_m']:.2f} m). "
                                     f"Needed: {dr['gain_reduction_needed_db']:.0f} dB less gain during the transmit, or the projector {dr['distance_for_full_scale_m']:.0f} m from every hydrophone, "
                                     f"or a source level of {dr['source_level_for_full_scale_db']:.0f} dB; and protection and a known recovery time: the first usable range is "
                                     f"{dr['min_range_memoryless_m']:.0f} m with an input that has no memory, plus {dr['range_per_ms_of_recovery_m']:.2f} m per millisecond of recovery "
                                     "(the source level is an RMS level; the volts and the overload are the pulse's PEAK, 3 dB above it)")
    if d.get("reference_times_s") and d["direct"]["over_full_scale_db"] > 0.0:
        gap = d["reference_times_s"][0] - d["direct"]["ends_s"]
        rec = pj.get("blanking_after_s")
        if rec is None or gap < float(rec):
            warn("sync.reference", f"the first synchronisation burst starts {gap * 1e3:.1f} ms after the projector's direct sound ends at the farthest hydrophone, and that sound overloads the inputs: "
                                   + ("how long they take to recover is not given (projector.blanking_after_s)" if rec is None else f"they take {float(rec) * 1e3:g} ms to recover")
                                   + ". Open the window later (operation.range_m) or lengthen sync.reference.lead_s so the burst is clear of it")
    # operation and trial
    pd_design = op.get("detection_probability")
    if pd_design is not None and not 0.0 < float(pd_design) < 1.0:
        err("operation.detection_probability", f"{pd_design}: a probability between 0 and 1 (the detection probability on one ping the budget's margins are stated for)")
    r0, r1 = op["range_m"]
    blind = 0.5 * c * pulse
    if r1 is None or (r0 is not None and float(r0) >= float(r1)):
        err("operation.range_m", "give [first, last] with last beyond first (first null: the blind range)")
        return out
    if r0 is not None and float(r0) < blind - 1e-6:
        warn("operation.range_m", f"the window opens at {float(r0):g} m but the array cannot listen while it transmits: nothing inside {blind:g} m ({pulse:g} s pulse) is usable")
    if p.get("max_depth_m") is not None:
        if float(p["max_depth_m"]) <= 0:
            err("platform.max_depth_m", "a depth in metres below the surface")
        elif p.get("depth_m") is not None and float(p["depth_m"]) > float(p["max_depth_m"]):
            err("platform.depth_m", f"the frame hangs at {p['depth_m']:g} m, deeper than the platform's limit ({p['max_depth_m']:g} m)")
        elif float(tr["own_depth_m"]) > float(p["max_depth_m"]):
            err("trial.own_depth_m", f"the trial runs the array at {tr['own_depth_m']:g} m, deeper than the platform's limit ({p['max_depth_m']:g} m)")
    if p["kind"] == "static":
        if p.get("depth_m") is None and get(spec, "navigation", "static_pose") is None and not get(spec, "navigation", "tilt_sensor", False):
            err("platform.depth_m", "a static platform needs the depth it hangs at (or navigation.static_pose, or a tilt sensor)")
    if float(tr["own_depth_m"]) >= float(tr["water_depth_m"]):
        err("trial.own_depth_m", f"the array at {tr['own_depth_m']:g} m is below the bottom of the trial's water ({tr['water_depth_m']:g} m)")
    trial_pulse = float(tr.get("pulse_s") or pulse)
    for i, r in enumerate(tr["ranges_m"]):
        this_blind = 0.5 * c * (pulse if i == 0 else trial_pulse)
        if float(r) < this_blind + 60.0:
            err("trial.ranges_m", f"a target at {float(r):g} m is inside (or within 60 m of) the pulse's blind range, {this_blind:g} m: it would not be seen")
    for z in tr.get("target_depths_m") or []:
        if not 0.0 < float(z) < float(tr["water_depth_m"]):
            err("trial.target_depths_m", f"a target at {z:g} m is not in the trial's water (0 to {tr['water_depth_m']:g} m)")
    if rows == 2:
        e = d["elevation"]
        if e["repeat_spacing_deg"] is not None:
            need = d["elevation"]["two_rows_one_answer_beyond_m"]
            far = float(tr["ranges_m"][-1])
            if need is not None and far < need:
                warn("trial.ranges_m", f"two rows {e['baseline_m']:g} m apart repeat every {e['repeat_spacing_deg']:.1f} deg; in {tr['water_depth_m']:g} m of water they give one elevation only beyond "
                                       f"{need:.0f} m (one bounce). The shallow trial at {far:g} m will report candidates, not one depth")
    if rows > 1 and get(spec, "navigation", "roll_sigma_deg") is None:
        warn("navigation.roll_sigma_deg", "no attitude accuracy given: a placeholder goes into every depth sigma (vehicle 0.2 deg, frame 1.0 deg)")
    ws = get(spec, "water_depth", "source", "unknown")
    if rows == 2 and ws == "unknown":
        warn("water_depth.source", "two rows need the water depth to choose among their candidate elevations (elevation.sector_from_depth); without it every detection stays ambiguous")
    if ws == "typed" and get(spec, "water_depth", "value_m") is None and rows > 1:
        warn("water_depth.value_m", "source is typed but no value is given yet: chain.elevation.water_depth stays null until it is typed in at deployment")
    if get(spec, "sound_speed", "rung") == "typed" and get(spec, "sound_speed", "value_m_s") is None:
        err("sound_speed.value_m_s", "rung is typed but no value is given")
    if get(spec, "sound_speed", "rung") == "profile" and get(spec, "sound_speed", "profile") is None:
        err("sound_speed.profile", "rung is profile but no profile is given")
    from . import _gpusonar as core
    sbl_max = int(getattr(core, "SBL_KERNEL_MAX_CHANNELS", 64))
    if count > sbl_max:
        warn("array.count", f"{count} hydrophones in one row: the sparse Bayesian bearing solver's kernels (doa: sbl) take at most {sbl_max} channels in a row; with more the chain falls back to "
                            "the host path (slower, and flagged in its report); bearings from the map (doa: map, the default) are unaffected")
    cpu = cpu_seconds_per_ping(spec)
    if cpu and compute_option(spec) != "gpu":
        took = cpu["decimate_first"] if compute_option(spec) == "decimate_first_cpu" else cpu["as_it_is"]
        if took > d["ping_interval_s"]:
            warn("compute.option", f"{compute_option(spec)}: measured {took:.1f} s a ping for {rows} row(s) on a CPU (docs/design/popoto_stage2.json, window {cpu['window_m'][0]:g}-{cpu['window_m'][1]:g} m) against "
                                   f"{d['ping_interval_s']:.1f} s between pings: not real time. The plan for two or more rows over the full window is a GPU (compute.option: gpu); "
                                   "a GPU has not been timed on a full-length ping either")
    if recording_mode(spec) == "decimated":
        warn("storage.record", "decimated: the recording will not hold the synchronisation reference or anything outside the band; nothing recorded this way can be re-synchronised or re-filtered later. Raw is the default for trials")
    for cls in target_classes_of(spec, resolve=False):
        if cls.get("status", "assumed" if cls.get("target_strength_db") is not None else "model") != "confirmed":
            warn(f"targets.{cls['name']}", "its target strength is " + ("a placeholder nobody vouches for (status: assumed)" if cls.get("target_strength_db") is not None else
                                                                        "the model figure of a bare rigid cylinder (status: model)") + ": every reach and depth planned against it is conditional on it")
    for env in environments_of(spec):
        if env.get("illustrative", True):
            warn(f"environments.{env['name']}", "an illustrative environment, not a measured one: the planned depths hold for it only")
        if env.get("own_depth_limits_m") is None and p.get("max_depth_m") is None:
            warn(f"environments.{env['name']}.own_depth_limits_m", "no operating depth limits given: the planner uses 5 to 300 m, a placeholder")
        if p["kind"] == "static" and p.get("depth_m") is not None and env.get("water_depth_m") is not None and float(p["depth_m"]) >= float(env["water_depth_m"]):
            warn(f"environments.{env['name']}.water_depth_m", f"the frame hangs at {p['depth_m']:g} m, below this environment's bottom ({env['water_depth_m']:g} m)")
    if rows * count > 64:
        warn("array.rows", f"{rows * count} channels in {rows} rows: passive listening must run with the elevation option (elevation=1, as the generated listen.options has it), which solves on rows. "
                           "Without it every channel is taken as one line: the bearings are wrong for stacked rows and the sparse Bayesian kernels (a GPU backend, or kernels=True) raise an error above 64 channels")
    return out


def check(spec: dict) -> list[Issue]:
    """Validate; raise SpecError on errors; return the warnings."""
    issues = validate(spec)
    if any(i.level == "error" for i in issues):
        raise SpecError(issues)
    return issues


# --- planning inputs: target classes and environments ---------------------------------------------------

def target_classes_of(spec: dict, resolve: bool = True, range_m: float = 12000.0) -> list:
    """The spec's target classes (`targets:`), resolved (target strength range, the cylinder's figures) unless told not to."""
    from . import target_classes
    classes = target_classes.load(spec.get("targets"))
    if not resolve or not classes:
        return classes
    f_lo, f_hi = (float(x) for x in spec["projector"]["band_hz"])
    return target_classes.resolve(classes, f_lo, f_hi, range_m, float(get(spec, "sound_speed", "nominal_m_s", acoustics.NOMINAL_SOUND_SPEED)))


def environments_of(spec: dict) -> list:
    from . import depthplan
    return depthplan.load_environments(spec.get("environments"))


# --- geometry and wiring ------------------------------------------------------------------------------

def names(spec: dict) -> dict:
    n = spec.get("names") or {}
    return {"sonar": n.get("sonar", spec["build"]), "deploy": n.get("deploy", spec["build"])}


def reference_of(spec: dict) -> dict:
    return {**DEFAULT_REFERENCE, **(get(spec, "sync", "reference", {}) or {})}


def board_table(spec: dict) -> list[dict]:
    """The boards in channel order: {first, count, row}. Row by row every board serves one row (a row whose
    count is not a multiple of the inputs leaves inputs spare); interleaved wiring fills boards across rows."""
    a, inputs = spec["array"], int(spec["boards"]["inputs"])
    rows, count = int(a["rows"]), int(a["count"])
    out = []
    if a.get("wiring", "row_by_row") == "interleave":
        for first in range(0, rows * count, inputs):
            out.append({"first": first, "count": min(inputs, rows * count - first), "row": None})
        return out
    for r in range(rows):
        for first in range(0, count, inputs):
            out.append({"first": r * count + first, "count": min(inputs, count - first), "row": r})
    return out


def clock_errors(spec: dict) -> list[tuple[float, float]]:
    """(offset s, drift ppm) of each simulated board, the first being the reference of the others."""
    b = spec["boards"]
    n = len(board_table(spec))
    clocks = b.get("clocks") or {"seed": 1}
    if "explicit" in clocks:
        return [(float(o), float(dr)) for o, dr in clocks["explicit"]]
    rng = np.random.default_rng(int(clocks.get("seed", 1)))
    off_max, ppm = float(b.get("clock_offset_max_s") or 3.0e-4), float(b.get("clock_tolerance_ppm") or 25.0)
    out = [(0.0, 0.0)]
    for _ in range(n - 1):
        out.append((float(np.round(rng.uniform(-off_max, off_max), 6)), float(np.round(rng.uniform(-ppm, ppm), 1))))
    return out


def adc_full_scale(spec: dict) -> float:
    b = spec["boards"]
    if b.get("adc_full_scale_v") is not None:
        return float(b["adc_full_scale_v"])
    return 0.5 * float(b["input_full_scale_vpp"]) * 10.0 ** (float(b.get("gain_db") or 0.0) / 20.0)


def window_range(spec: dict) -> list[float]:
    c = float(get(spec, "sound_speed", "nominal_m_s", acoustics.NOMINAL_SOUND_SPEED))
    r0, r1 = spec["operation"]["range_m"]
    blind = 0.5 * c * float(spec["projector"]["waveform"]["duration_s"])
    return [float(blind if r0 is None else r0), float(r1)]


# --- the generated files ------------------------------------------------------------------------------

def _flag(spec, flags, section, key, target):
    if status(spec, section, key) != "confirmed" and target not in flags:
        flags.append(target)


def sonar_config(spec: dict) -> dict:
    """The complete sonar file of a build (no `base:`): array, transmit, receive, with the platform and a
    `hardware` block that the diagrams and the parts list read. `emit` writes it against a base."""
    p, a, h, b, pj = (spec[k] for k in ("platform", "array", "hydrophone", "boards", "projector"))
    rows, count = int(a["rows"]), int(a["count"])
    c = float(get(spec, "sound_speed", "nominal_m_s", acoustics.NOMINAL_SOUND_SPEED))
    starboard = a.get("look", "starboard") == "starboard"
    arr: dict = {"axis": [1, 0, 0], "boresight": [0, 1 if starboard else -1, 0]}
    row_keys = {"count": count}
    if a.get("positions_m") is not None:
        row_keys = {"positions": [float(x) for x in a["positions_m"]]}
    if rows == 1:
        arr.update({"layout": "line", **row_keys} if "positions" in row_keys else {"layout": "uniform_line", "count": count, "pitch": float(a["pitch_m"])})
        rows_key = "array.layout"
    else:
        word = ROW_WORDS.get(rows, f"rows_{rows}")
        outs = a.get("out_m") or [0.0] * rows
        layout = {"layout": "stacked_lines", "rows": [{"up": float(u), "out": float(o), **({"positions": row_keys["positions"]} if "positions" in row_keys else {})}
                                                       for u, o in zip(a["heights_m"], outs)]}
        if a.get("wiring") == "interleave":
            layout["interleave"] = int(a["interleave"])
        if "positions" not in row_keys:
            arr.update({"count": count, "pitch": float(a["pitch_m"])})
        arr.update({"layout_choice": word, "layouts": {word: layout}})
        rows_key = f"array.layouts.{word}.rows"
    wf = pj["waveform"]
    f_lo, f_hi = (float(x) for x in pj["band_hz"])
    transmit = {"kind": "point", "lever_arm": [float(x) if float(x) != int(float(x)) else int(float(x)) for x in pj["position_m"]],
                "waveform": {"type": wf.get("type", "lfm"), "f_start": f_lo, "f_stop": f_hi, "duration": float(wf["duration_s"]),
                             "window_alpha": float(wf.get("window_alpha", 0.1)), "source_level_db_rms": float(pj["source_level_db_rms"])}}
    boards = board_table(spec)
    receive: dict = {"kind": "real", "sample_rate": float(b["sample_rate_hz"]), "adc_bits": int(b["adc_bits"]), "adc_full_scale": adc_full_scale(spec),
                     "sensitivity_db": float(h["sensitivity_db"]), "gain_db": float(b.get("gain_db") or 0.0) + float(h.get("preamp_gain_db") or 0.0)}
    if b.get("highpass_hz") is not None:
        receive["highpass_corner_hz"] = float(b["highpass_hz"])
    receive["window"] = {"range": window_range(spec)}
    if all(x["count"] == int(b["inputs"]) for x in boards):
        receive["board_size"] = int(b["inputs"])
    else:
        receive["boards"] = [{"first": x["first"], "count": x["count"]} for x in boards]
    platform = {"name": p.get("name", spec["build"]), "kind": p["kind"], "span_m": float(p["span_m"]), "height_m": float(p["height_m"])}
    for key in ("usable_span_m", "usable_height_m", "max_depth_m"):
        if p.get(key) is not None:
            platform[key] = float(p[key])
    if rows > 1:
        platform["row_baseline_m"] = float(max(a["heights_m"]) - min(a["heights_m"]))
    if p.get("mounting"):
        platform["mounting"] = p["mounting"]
    if p["kind"] == "static":
        platform["static"] = True
        if p.get("depth_m") is not None:
            platform["depth_m"] = float(p["depth_m"])
    ref = reference_of(spec)
    scheme = get(spec, "sync", "scheme", "none" if b.get("shared_clock") else "shared_reference")
    hardware = {
        "hydrophone": {k: h.get(k) for k in ("model", "noise_db", "band_hz", "preamp_gain_db") if h.get(k) is not None},
        "boards": {"model": b.get("model"), "inputs": int(b["inputs"]), "gain_db": float(b.get("gain_db") or 0.0),
                   **{k: b[k] for k in ("input_full_scale_vpp", "simultaneous", "clock_tolerance_ppm", "shared_clock", "overload_damage_vpp", "gain_switchable") if b.get(k) is not None},
                   **({"row_module": a["module"]} if a.get("module") else {}),
                   "wiring": a.get("wiring", "row_by_row"),
                   "position_m": [float(x) for x in (a.get("board_position_m") or [0.0, 0.0, 0.0])], "cable_slack_m": float(a.get("cable_slack_m", 0.5))},
        "sync": {"scheme": scheme, "elevation_error_deg": float(get(spec, "sync", "elevation_error_deg", 0.05))},
        "projector": {k: pj.get(k) for k in ("model", "amplifier", "blanking_after_s") if pj.get(k) is not None},
        "navigation": {k: v for k, v in (spec.get("navigation") or {}).items() if k in ("unit", "adapter", "rate_hz", "timestamp_sigma_s", "tilt_sensor") and v is not None},
        "sound_speed": {"rung": get(spec, "sound_speed", "rung", "gradient_prior")},
        "water_depth": {"source": get(spec, "water_depth", "source", "unknown")},
        "compute": {"option": compute_option(spec), **{k: v for k, v in (spec.get("compute") or {}).items() if k == "unit" and v is not None}},
        "storage": {"record": recording_mode(spec), **{k: v for k, v in (spec.get("storage") or {}).items() if k == "capacity_gb" and v is not None}},
    }
    if scheme == "shared_reference":
        hardware["sync"]["reference_level_v"] = float(ref["level_v"])
    hardware = {k: v for k, v in hardware.items() if v}
    flags: list[str] = []
    for key in ("span_m", "height_m", "usable_span_m", "usable_height_m", "depth_m", "max_depth_m"):
        if p.get(key) is not None:
            _flag(spec, flags, "platform", key, f"platform.{key}")
    _flag(spec, flags, "array", "pitch_m" if a.get("pitch_m") is not None else "positions_m", "array.pitch")
    if rows > 1:
        _flag(spec, flags, "array", "heights_m", rows_key)
        if status(spec, "array", "heights_m") != "confirmed":
            flags.append("platform.row_baseline_m")
    _flag(spec, flags, "hydrophone", "sensitivity_db", "receive.sensitivity_db")
    for key, target in (("sample_rate_hz", "receive.sample_rate"), ("adc_bits", "receive.adc_bits"), ("gain_db", "receive.gain_db"),
                        ("highpass_hz", "receive.highpass_corner_hz"), ("simultaneous", "receive.simultaneous_sampling")):
        if key == "simultaneous" or b.get(key) is not None:
            _flag(spec, flags, "boards", key, target)
    if not (status(spec, "boards", "input_full_scale_vpp") == "confirmed" or status(spec, "boards", "adc_full_scale_v") == "confirmed"):
        flags.append("receive.adc_full_scale")
    _flag(spec, flags, "operation", "range_m", "receive.window")
    tx_keys = [("source_level_db_rms", "transmit.waveform.source_level_db_rms"), ("band_hz", "transmit.waveform.f_start"), ("waveform", "transmit.waveform.duration"),
               ("position_m", "transmit.lever_arm")]
    if all(status(spec, "projector", k) != "confirmed" for k, _ in tx_keys) and status(spec, "projector", "model") != "confirmed":
        flags.append("transmit")
    else:
        for key, target in tx_keys:
            _flag(spec, flags, "projector", key, target)
    for section, keys in (("hydrophone", ("model", "noise_db", "band_hz")), ("boards", ("model", "overload_damage_vpp", "gain_switchable")), ("projector", ("model", "amplifier", "blanking_after_s")),
                          ("navigation", ("unit", "rate_hz", "timestamp_sigma_s")), ("compute", ("unit",))):
        for key in keys:
            _flag(spec, flags, section, key, f"hardware.{section}.{key}")
    cfg = {"name": names(spec)["sonar"], "mode": "flank", "nominal_sound_speed": c, "build": spec["build"], "platform": platform, "array": arr,
           "transmit": transmit, "receive": receive, "hardware": hardware, "unconfirmed": flags}
    return cfg


def clocks_config(spec: dict) -> dict:
    """The overlay of the `<name>_clocks` sonar: every board with its own clock error, and the reference."""
    boards = board_table(spec)
    jitter = float(spec["boards"].get("jitter_rms_s") or 2.0e-8)
    ref = reference_of(spec)
    rows = [{"first": x["first"], "count": x["count"], "clock_offset": off, "clock_drift_ppm": ppm, "jitter_rms": jitter, "reference_channel": x["first"]}
            for x, (off, ppm) in zip(boards, clock_errors(spec))]
    return {"base": names(spec)["sonar"], "name": names(spec)["sonar"] + "_clocks",
            "receive": {"boards": rows,
                        "reference": {"mode": "sync_chirp", "f_start": float(ref["f_start_hz"]), "f_stop": float(ref["f_stop_hz"]), "duration": float(ref["duration_s"]),
                                      "level": float(ref["level_v"]), "lead": float(ref["lead_s"]), "tail": float(ref["tail_s"]), "channels": [x["first"] for x in boards]}},
            "unconfirmed": ["receive.boards", "receive.reference"]}


def compute_option(spec: dict) -> str:
    """`compute.option`, or its default: a GPU for two or more rows, decimate-first on a CPU for one (the owner's decision on the
    stage-2 timings: docs/design/popoto_stage2.json)."""
    return get(spec, "compute", "option", "gpu" if int(spec["array"]["rows"]) > 1 else "decimate_first_cpu")


def recording_mode(spec: dict) -> str:
    return get(spec, "storage", "record", "raw")


def cpu_seconds_per_ping(spec: dict) -> dict | None:
    """Measured seconds per ping of the chain on a CPU for this build's row count, from docs/design/popoto_stage2.json (full window,
    the boards' rate): {decimate_first, as_it_is, window_m}. None when the build is not one of the measured ones (rows of 32 at the
    measured sample rate): nothing is extrapolated."""
    t = (measured_timings().get("stage2") or {}).get("decimation", {}).get("timing", {}).get("host", {}).get(str(int(spec["array"]["rows"])))
    if not t or int(spec["array"]["count"]) * int(spec["array"]["rows"]) != t.get("channels") or float(spec["boards"]["sample_rate_hz"]) != t.get("sample_rate_hz"):
        return None
    return {"decimate_first": float(t["chain_decimate_first"]["seconds"]), "as_it_is": float(t["chain_full_rate"]["seconds"]), "window_m": t.get("window_m")}


def needs_sync(spec: dict) -> bool:
    return get(spec, "sync", "scheme", "none" if spec["boards"].get("shared_clock") else "shared_reference") == "shared_reference" and len(board_table(spec)) > 1


def chain_config(spec: dict, sync: bool | None = None) -> dict:
    """The `chain` section of the deployment file."""
    a, p = spec["array"], spec["platform"]
    rows = int(a["rows"])
    static = p["kind"] == "static"
    chain: dict = {}
    if needs_sync(spec) if sync is None else sync:
        ref = reference_of(spec)
        chain["sync"] = {"reference": {"f_start": float(ref["f_start_hz"]), "f_stop": float(ref["f_stop_hz"]), "duration": float(ref["duration_s"]),
                                       "lead": float(ref["lead_s"]), "tail": float(ref["tail_s"])}}
    base = copy.deepcopy(DEFAULT_CHAIN)
    base["look"] = a.get("look", "starboard")
    if get(spec, "operation", "bearings_deg") is not None:
        base["bearings_deg"] = [float(x) for x in spec["operation"]["bearings_deg"]]
    chain.update(base)
    for key in ("engine", "backend"):
        if get(spec, "compute", key) is not None:
            chain[key] = spec["compute"][key]
    if compute_option(spec) == "decimate_first_cpu" and recording_mode(spec) == "raw":
        chain["decimate"] = {}                              # the recording holds the raw ping; the chain decimates (docs/DEPLOYMENT.md, Stage 2)
    if get(spec, "sound_speed", "rung") == "typed":
        chain["sound_speed"] = float(spec["sound_speed"]["value_m_s"])
    if rows > 1:
        nav_tilt = bool(get(spec, "navigation", "tilt_sensor", False))
        el: dict = {"sector_deg": [-30.0, 30.0, 0.25]}
        if rows == 2:
            el["sector_from_depth"] = {"bounces": 1, "margin_deg": 1.0}
        el["sector_from_rays"] = {}                         # every stacked-row build (the owner): a far lobe of the rows is not taken for an arrival
        el["multipath_sigma"] = {}                          # every stacked-row build: the depth sigma says when echoes the rows cannot tell apart share the range cell
        el["attitude_time"] = get(spec, "navigation", "attitude_time", "arrival" if (not static or nav_tilt) else "transmit")
        el["attitude_rate_sigma_deg_s"] = float(get(spec, "navigation", "attitude_rate_sigma_deg_s", 0.5))
        el["roll_sigma_deg"] = float(get(spec, "navigation", "roll_sigma_deg", 1.0 if static else 0.2))
        el["water_depth"] = float(spec["water_depth"]["value_m"]) if get(spec, "water_depth", "value_m") is not None else None
        el["water_depth_sigma_m"] = float(get(spec, "water_depth", "sigma_m", 5.0))
        if get(spec, "sound_speed", "rung") == "profile":
            el["ssp"] = spec["sound_speed"]["profile"]
            el["gradient_per_s"] = None
        else:
            el["gradient_per_s"] = float(get(spec, "sound_speed", "gradient_per_s", 0.017))
        el["gradient_sigma_per_s"] = float(get(spec, "sound_speed", "gradient_sigma_per_s", 0.05))
        chain["elevation"] = el
    if get(spec, "operation", "coverage") is not None:
        from . import coverage
        chain["coverage"] = coverage.of_build(spec, spec["operation"]["coverage"])     # raises when the site lacks water depth, bottom or wind
    chain = deep_merge(chain, get(spec, "operation", "chain", {}) or {})
    if chain.get("multipath_depth") is not None:            # wherever it is on (the owner): through rays, and no flagged depth reported
        chain["multipath_depth"] = {**MULTIPATH_DEPTH_DEPLOYED, **chain["multipath_depth"]}
    return chain


def _idle_seconds(spec: dict) -> float:
    """How long without a block before the stream is taken to have ended: three ping intervals (the interval is the
    two-way time to the far edge of the receive window plus the pulse, as `describe` has it), at least 30 s."""
    c = float(get(spec, "sound_speed", "nominal_m_s", acoustics.NOMINAL_SOUND_SPEED))
    interval = 2.0 * float(window_range(spec)[1]) / c + float(sonar_config(spec)["transmit"]["waveform"]["duration"])
    return float(max(30.0, np.ceil(3.0 * interval)))


def deploy_config(spec: dict) -> dict:
    """The deployment file: where pings come from, navigation, the chain, where results go."""
    p = spec["platform"]
    rows = int(spec["array"]["rows"])
    static = p["kind"] == "static"
    net, sto = spec.get("network") or {}, spec.get("storage") or {}
    sonar = names(spec)["sonar"] + ("_clocks" if needs_sync(spec) else "")
    cfg: dict = {"source": {"type": "flank_udp", "sonar": sonar, "host": "0.0.0.0", "port": int(net.get("stream_port", 5700)),
                              # the stream has ended after this long without a block (the program's idle_seconds): three ping intervals, at least 30 s
                              "timeout": _idle_seconds(spec)}}
    if recording_mode(spec) == "decimated":
        cfg["source"]["decimate"] = {}                      # the receiver decimates as blocks arrive: the recording holds decimated pings only
    if static and not get(spec, "navigation", "tilt_sensor", False):
        pose = get(spec, "navigation", "static_pose") or {"x": 0.0, "y": 0.0, "z": float(p.get("depth_m") or 0.0), "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        cfg["nav"] = {"static": {k: float(v) for k, v in pose.items()}}
    else:
        cfg["nav"] = {"port": int(net.get("nav_port", 5702))}
    cfg["chain"] = chain_config(spec)
    stem = f"/data/sonar/{spec['build']}_%Y%m%d_%H%M%S"
    cfg["output"] = {"recording": sto.get("recording", stem + ".h5"), "report": sto.get("report", stem + ".jsonl"),
                     "report_udp": list(net.get("report_udp", ["127.0.0.1", 5701])),
                     "taps": {"dir": sto.get("taps_dir", "/data/sonar/taps"), "pings": int(sto.get("taps_pings", 2))}}
    flags: list[str] = []
    if "sync" in cfg["chain"]:
        flags.append("chain.sync")
    if "static" in cfg["nav"] and status(spec, "navigation", "static_pose") != "confirmed" and status(spec, "platform", "depth_m") != "confirmed":
        flags.append("nav.static")
    if rows > 1:
        for section, key, target in (("navigation", "roll_sigma_deg", "chain.elevation.roll_sigma_deg"),
                                     ("navigation", "attitude_rate_sigma_deg_s", "chain.elevation.attitude_rate_sigma_deg_s"),
                                     ("water_depth", "sigma_m", "chain.elevation.water_depth_sigma_m"),
                                     ("sound_speed", "gradient_per_s", "chain.elevation.gradient_per_s"),
                                     ("sound_speed", "gradient_sigma_per_s", "chain.elevation.gradient_sigma_per_s")):
            if target.split(".")[-1] in cfg["chain"]["elevation"] and cfg["chain"]["elevation"][target.split(".")[-1]] is not None:
                _flag(spec, flags, section, key, target)
    if get(spec, "operation", "coverage") is not None:     # the site, the classes and the levels: flagged unless somebody vouches for them
        site = spec["operation"]["coverage"]["environment"]
        env = next(e for e in environments_of(spec) if e["name"] == site) if isinstance(site, str) else site
        if env.get("illustrative", True):
            flags += ["chain.coverage.water_depth_m", "chain.coverage.bottom", "chain.coverage.surface"]
        if any(c.get("status") != "confirmed" for c in target_classes_of(spec, resolve=False)):
            flags.append("chain.coverage.classes")
        if env.get("illustrative", True) or status(spec, "projector", "source_level_db_rms") != "confirmed":
            flags.append("chain.coverage.budget")
    cfg["unconfirmed"] = flags
    return cfg


def listen_options(spec: dict) -> dict:
    """`key=value` settings of `build/bin/sonar_live --listen` (passive bearings from the untriggered stream) for
    this build: frames of about a quarter of a second at the boards' rate."""
    fs = float(spec["boards"]["sample_rate_hz"])
    f_lo, f_hi = (float(x) for x in spec["projector"]["band_hz"])
    n_fft = int(2 ** round(np.log2(fs * 0.04)))
    b = get(spec, "operation", "bearings_deg", DEFAULT_CHAIN["bearings_deg"])
    o = {"f_low": f_lo, "f_high": f_hi, "n_fft": n_fft, "hop": n_fft // 2, "n_average": 12, "bearing_first_deg": float(b[0]), "bearing_last_deg": float(b[1]),
         "bearing_step_deg": float(b[2]), "look": spec["array"].get("look", "starboard")}
    if int(spec["array"]["rows"]) > 1:
        o["elevation"] = 1
    return o


def _target(r, bearing_deg, depth):
    b = np.radians(bearing_deg)
    return [float(np.round(r * np.sin(b), 1)), float(np.round(r * np.cos(b), 1)), float(depth)]


def scenario_configs(spec: dict) -> dict:
    """The build's simulated trials, by kind: free (truth in free field), shallow (boundaries), rehearsal (raw
    samples, every board on its own clock, the attitude the platform would have), passing (a mover) and
    passive (radiating sources, no transmission). Each carries a `truth` block the self-tests check against."""
    p, a, b, pj, tr = (spec[k] for k in ("platform", "array", "boards", "projector", "trial"))
    rows = int(a["rows"])
    static = p["kind"] == "static"
    sonar = names(spec)["sonar"]
    c = float(tr.get("sound_speed_m_s") or get(spec, "sound_speed", "nominal_m_s", acoustics.NOMINAL_SOUND_SPEED))
    f_lo, f_hi = (float(x) for x in pj["band_hz"])
    band = f_hi - f_lo
    base_rate = 4.0 * band                                       # complex samples: twice the bandwidth, twice over
    own, depth = float(tr["own_depth_m"]), float(tr["water_depth_m"])
    ts = float(tr.get("target_strength_db", 10.0))
    seed = int(tr.get("seed", 1))
    sign = 1.0 if a.get("look", "starboard") == "starboard" else -1.0
    near, far = float(tr["ranges_m"][0]), float(tr["ranges_m"][-1])
    depths = [float(z) for z in (tr.get("target_depths_m") or [max(2.0, own - 0.3 * own), own, min(depth - 2.0, own + 0.5 * (depth - own))])]
    env = {"sound_speed": c, "temperature": 12.0, "salinity": 35.0}
    noise = {"adc": False, "ambient_noise": {"wind_speed": float(tr.get("wind_speed_m_s", 5.0)), "shipping": 0.5}}
    speed = 0.0 if static else float(tr.get("speed_m_s", 1.5))
    roll = np.radians(float(tr.get("roll_deg", 2.0 if static else 3.0)))
    rate = 0.0 if static else np.radians(float(tr.get("roll_rate_deg_s", 2.0)))
    bottom = tr.get("bottom") or C_ASSUMED_BOTTOM
    # No `max_order`: how many bounces are kept is the simulator's energy rule (D-105), as in the shipped trial scenarios.
    prop = {"surface_z": 0.0, "bottom_z": depth, "bottom": dict(bottom)}
    pulse_far = float(tr.get("pulse_s") or pj["waveform"]["duration_s"])
    far_tx = {"transmit": {"waveform": {"duration": pulse_far}}} if pulse_far != float(pj["waveform"]["duration_s"]) else {}

    def points(pos, label="target", **kw):
        return {"type": "points", "label": label, "positions": [pos], "target_strength_db": ts, **kw}

    blind = 0.5 * c * float(pj["waveform"]["duration_s"])
    near_window = [max(blind, near - 150.0), near + 250.0]
    out = {}
    # free field: the depths the chain reports must be the targets'.
    tg = [[float(np.round((near + 45.0 * i) * np.sin(np.radians(bb)), 1)), float(np.round(sign * (near + 45.0 * i) * np.cos(np.radians(bb)), 1)), z]
          for i, (bb, z) in enumerate(zip((-15.0, 5.0, 25.0), depths))]
    out["free"] = {"name": f"{spec['build']}_free", "sonar": sonar, "fidelity": "free_field",
                   "sonar_overrides": {"receive": {"kind": "baseband", "sample_rate": base_rate, "window": {"range": near_window}}},
                   "environment": dict(env), "platform": {"position": [0, 0, own], "rpy": [float(np.round(roll, 4)), 0, 0], "velocity": [speed, 0, 0]},
                   "pings": {"count": 4, "interval": 4.0},
                   "scene": [points(t, label=f"t{i}") for i, t in enumerate(tg)], "absorption": "ainslie_mccolm", "receiver": copy.deepcopy(noise), "seed": seed,
                   "truth": {"targets": tg, "check_depth": rows > 1}}
    # shallow water: boundaries, one target at the far range.
    t_far = [float(np.round(far * np.sin(np.radians(8.0)), 1)), float(np.round(sign * far * np.cos(np.radians(8.0)), 1)), depths[-1]]
    out["shallow"] = {"name": f"{spec['build']}_shallow", "sonar": sonar, "fidelity": "image_source",
                      "sonar_overrides": deep_merge({"receive": {"kind": "baseband", "sample_rate": base_rate, "window": {"range": [far - 200.0, far + 300.0]}}}, far_tx),
                      "environment": {**env, "water_depth": depth}, "propagation": copy.deepcopy(prop),
                      "platform": {"position": [0, 0, own], "rpy": [float(np.round(roll, 4)), 0, 0], "velocity": [speed, 0, 0]},
                      "pings": {"count": 4, "interval": 5.0}, "scene": [points(t_far)], "absorption": "ainslie_mccolm", "receiver": copy.deepcopy(noise), "seed": seed + 1,
                      "truth": {"targets": [t_far], "check_depth": rows > 1, "water_depth": depth}}
    # rehearsal: raw samples at the boards' rate, every board on its own clock, attitude as on the platform.
    r_reh = far if rows == 2 else near
    t_reh = [float(np.round(r_reh * np.sin(np.radians(10.0)), 1)), float(np.round(sign * r_reh * np.cos(np.radians(10.0)), 1)), depths[-1] if rows > 1 else depths[1]]
    plat = {"position": [0, 0, own], "rpy": [float(np.round(roll, 4)), 0.0 if static else 0.0175, 0], "velocity": [speed, 0, 0]}
    if rate:
        plat["rates"] = [float(np.round(rate, 4)), 0, 0]
    out["rehearsal"] = {"name": f"{spec['build']}_rehearsal", "sonar": sonar + ("_clocks" if needs_sync(spec) else ""), "fidelity": "image_source",
                        "sonar_overrides": deep_merge({"receive": {"window": {"range": [r_reh - 150.0, r_reh + 150.0]}}}, far_tx if rows == 2 else {}),
                        "environment": {**env, "water_depth": depth}, "propagation": copy.deepcopy(prop), "platform": plat,
                        "pings": {"count": 3, "interval": 4.0}, "scene": [points(t_reh)], "absorption": "ainslie_mccolm", "receiver": copy.deepcopy(noise), "seed": seed + 2,
                        "truth": {"targets": [t_reh], "check_depth": rows > 1, "water_depth": depth, "sync": needs_sync(spec)}}
    # a target crossing: the tracker must report one mover with its speed.
    t_pass = [-40.0, float(sign * near), depths[-1] if rows > 1 else depths[1]]
    out["passing"] = {"name": f"{spec['build']}_passing", "sonar": sonar, "fidelity": "image_source",
                      "sonar_overrides": {"receive": {"kind": "baseband", "sample_rate": base_rate, "window": {"range": near_window}}},
                      "environment": {**env, "water_depth": depth}, "propagation": copy.deepcopy(prop),
                      "platform": {"position": [0, 0, own], "rpy": [0, 0, 0], "velocity": [speed, 0, 0]},
                      "pings": {"count": 8, "interval": 4.0}, "scene": [points(t_pass, label="mover", velocity=[3.0, 0.0, 0.0])],
                      "absorption": "ainslie_mccolm", "receiver": copy.deepcopy(noise), "seed": seed + 3,
                      "truth": {"targets": [t_pass], "speed_m_s": 3.0, "check_depth": False}}
    # a submerged mover of the spec's first submerged target class, at speed: tracked with NO sound-speed profile knowledge.
    # A dive for a profile is optional: it must never be needed to detect, track or classify (the owner).
    sub = next((cl for cl in target_classes_of(spec, resolve=False) if float(cl["depth_m"][0]) >= 10.0), None)
    if sub is not None:
        z_sub = float(min(0.5 * (float(sub["depth_m"][0]) + float(sub["depth_m"][1])), depth - 5.0))
        v_sub = float(min((sub.get("speed_m_s") or [0.0, 3.0])[1], 4.0))             # the class's top speed, at most 4 m/s so that it stays inside the short window
        t_sub = [-40.0, float(sign * near), z_sub]
        out["submerged"] = copy.deepcopy(out["passing"])
        out["submerged"].update(name=f"{spec['build']}_submerged", scene=[points(t_sub, label=str(sub["name"]), velocity=[v_sub, 0.0, 0.0])], seed=seed + 5,
                                truth={"targets": [t_sub], "speed_m_s": v_sub, "heading_deg": 0.0, "depth_m": z_sub, "class": sub["name"], "check_depth": False,
                                       "profile_knowledge": "none"})
    # passive: two radiating sources, the sonar silent.
    src = [[float(np.round(near * np.sin(np.radians(bb)), 1)), float(np.round(sign * near * np.cos(np.radians(bb)), 1)), z] for bb, z in ((-20.0, depths[0]), (12.0, depths[-1]))]
    out["passive"] = {"name": f"{spec['build']}_passive", "sonar": sonar, "fidelity": "free_field",
                      "sonar_overrides": {"receive": {"kind": "baseband", "sample_rate": base_rate, "window": {"range": [0.0, 0.5 * c * 1.0]}}},
                      "environment": dict(env), "platform": {"position": [0, 0, own], "rpy": [0, 0, 0], "velocity": [0, 0, 0]},
                      "pings": {"count": 2, "interval": 2.0, "transmit": False}, "scene": [],
                      "sources": [{"label": f"source{i}", "origin": s, "velocity": [0.0, 0.0, 0.0],
                                   "signal": {"type": "noise", "level_db": 130.0, "reference_frequency": 0.5 * (f_lo + f_hi), "band": [f_lo, f_hi]}} for i, s in enumerate(src)],
                      "absorption": "ainslie_mccolm", "receiver": copy.deepcopy(noise), "seed": seed + 4,
                      "truth": {"sources": src}}
    return out


# --- writing files ------------------------------------------------------------------------------------

def diff_against(cfg: dict, base: dict) -> dict:
    """What a file extending `base` must hold so that it resolves to `cfg` (keys only the base has stay)."""
    out = {}
    for key, v in cfg.items():
        if key in ("unconfirmed", "confirmed"):
            continue
        if isinstance(v, dict) and isinstance(base.get(key), dict):
            sub = diff_against(v, base[key])
            if sub:
                out[key] = sub
        elif key not in base or base[key] != v:
            out[key] = copy.deepcopy(v)
    return out


def against_base(cfg: dict, base_name: str) -> dict:
    """`cfg` as a file with `base: base_name`: only what differs, the flags the base lacks, and under
    `confirmed` the base's flags this build has settled."""
    base = load_sonar_config(base_name)
    out = {"base": base_name, **diff_against(cfg, base)}
    mine, theirs = list(cfg.get("unconfirmed", [])), list(base.get("unconfirmed", []))

    def covered(flag, by):
        return any(flag == b or flag.startswith(b + ".") for b in by)

    confirmed = [f for f in theirs if not covered(f, mine) and not any(m.startswith(f + ".") for m in mine)]
    # A base flag that is broader than ours (`transmit` over `transmit.lever_arm`) is taken off and ours put on.
    broader = [f for f in theirs if any(m.startswith(f + ".") for m in mine) and f not in mine]
    kept = [f for f in theirs if f not in confirmed and f not in broader]
    out["unconfirmed"] = [f for f in mine if not covered(f, kept)]
    if confirmed or broader:
        out["confirmed"] = confirmed + broader
    return out


def dump(cfg: dict, header: str) -> str:
    """YAML text of a generated file: a comment header, then the mapping (short lists on one line)."""
    import yaml

    class Dumper(yaml.SafeDumper):
        pass

    def represent_float(dumper, value):
        if value != value or value in (float("inf"), float("-inf")):
            return yaml.SafeDumper.represent_float(dumper, value)
        text = repr(float(value))
        if "e" in text and "." not in text.split("e")[0]:
            text = text.replace("e", ".0e")
        return dumper.represent_scalar("tag:yaml.org,2002:float", text)

    Dumper.add_representer(float, represent_float)
    body = yaml.dump(cfg, Dumper=Dumper, sort_keys=False, default_flow_style=None, width=160, allow_unicode=True)
    head = "".join(f"# {line}\n" if line else "#\n" for line in header.strip("\n").split("\n"))
    return head + body


def files(spec: dict) -> dict:
    """Every config file of the build: {relative path: text}. Deterministic."""
    n = names(spec)
    title = spec.get("title", spec["build"])
    made = f"GENERATED by tools/build_from_spec.py from configs/builds/{spec['build']}.yaml. Do not edit: change the spec and generate again."
    sonar = sonar_config(spec)
    full = sonar if not spec.get("base_sonar") else against_base(sonar, spec["base_sonar"])
    rows, count = int(spec["array"]["rows"]), int(spec["array"]["count"])
    out = {f"sonars/{n['sonar']}.yaml": dump(full, f"{title}: {rows} x {count} hydrophones on {len(board_table(spec))} boards of {spec['boards']['inputs']} inputs.\n{made}\n"
                                                   "Keys: docs/CONFIGURATION.md. `platform` and `hardware` describe the build for the diagrams and the parts list\n"
                                                   "(docs/REFERENCE_DESIGN.md); the simulator and the chain read array, transmit and receive.\n"
                                                   "`unconfirmed`: values nobody has measured or stated; they add to the base file's (`confirmed` takes one off).")}
    if needs_sync(spec):
        out[f"sonars/{n['sonar']}_clocks.yaml"] = dump(clocks_config(spec), f"{title} with every board on its own clock, for rehearsals: clock offsets, drifts and jitter as the\n"
                                                       "simulator's truth, and the shared synchronisation reference (one electrical signal wired to every board and\n"
                                                       f"summed into its first input, above the acoustic band). The chain has to estimate the clocks.\n{made}")
    out[f"deploy/{n['deploy']}.yaml"] = dump(deploy_config(spec), f"{title}: the deployment file.\n  tools/sonar_live.py configs/deploy/{n['deploy']}.yaml\n{made}\n"
                                             "Keys of `chain`: docs/CONFIGURATION.md. How to run, rehearse and bring up: docs/DEPLOYMENT.md, docs/REFERENCE_DESIGN.md.\n"
                                             "`chain.elevation.water_depth: null` means: what navigation sends, else unknown.")
    for kind, sc in scenario_configs(spec).items():
        out[f"scenarios/{sc['name']}.yaml"] = dump(sc, f"{title}: simulated trial `{kind}` (python/gpusonar/buildspec.py: scenario_configs).\n{made}\n"
                                                   "`truth` is what the self-tests check the chain's reports against (tools/build_from_spec.py --self-test).")
    return out


# --- derived numbers ----------------------------------------------------------------------------------

def positions(spec: dict) -> np.ndarray:
    """Hydrophone lever arms [N, 3] in channel order, body frame (x along, y to starboard, z down)."""
    from .sonar import _positions
    arr = sonar_config(spec)["array"]
    if "layouts" in arr:
        top = {k: v for k, v in arr.items() if k not in ("layouts", "layout_choice")}
        arr = {**arr["layouts"][arr["layout_choice"]], **top}
    return _positions(arr)


def derive(spec: dict) -> dict:
    """Everything that is arithmetic on the spec: geometry, wiring, rates, timing, storage, the projector's
    direct sound, the synchronisation tolerance, the elevation response. Read by the validator, the diagrams,
    the parts list and the plan text."""
    d = describe(sonar_config(spec), clocks_config(spec) if needs_sync(spec) else None, deploy_config(spec))
    d["detection_probability"] = float(get(spec, "operation", "detection_probability", DESIGN_DETECTION_PROBABILITY))
    return d


def describe(sonar_cfg: dict, clocks_cfg: dict | None = None, deploy_cfg: dict | None = None) -> dict:
    """The same numbers from config files alone (a resolved sonar config, optionally its `_clocks` overlay and
    its deployment file): what the diagrams and the parts list are functions of."""
    from .sonar import SonarModel
    model = SonarModel.from_config(copy.deepcopy(sonar_cfg))
    cfg = model.config
    c = model.nominal_sound_speed
    pos = model.positions
    w, rx = cfg["transmit"]["waveform"], cfg["receive"]
    hw = cfg.get("hardware") or {}
    plat = cfg.get("platform") or {}
    f_lo, f_hi = sorted((float(w["f_start"]), float(w.get("f_stop", w["f_start"]))))
    fc = 0.5 * (f_lo + f_hi)
    lam = c / fc
    pulse = float(w["duration"])
    fs, bits = float(rx["sample_rate"]), int(rx.get("adc_bits", 24))
    # rows from the heights (z is down)
    up = np.round(-pos[:, 2], 4)
    heights = sorted(set(up.tolist()))
    row_of = np.array([heights.index(u) for u in up.tolist()])
    per_row = [int((row_of == i).sum()) for i in range(len(heights))]
    along = pos[row_of == 0, 0]
    pitch = float(np.median(np.diff(np.sort(along)))) if len(along) > 1 else 0.0
    row_length = float(np.ptp(along))
    inputs = int((hw.get("boards") or {}).get("inputs") or rx.get("board_size") or max(int(b.get("count", 0)) for b in model.boards if b) or len(pos))
    boards = []
    for i, b in enumerate(model.boards):
        first, count = (int(b["first"]), int(b["count"])) if b else (0, len(pos))
        rows_here = sorted(set(row_of[first:first + count].tolist()))
        boards.append({"index": i, "first": first, "count": count, "rows": rows_here, "spare_inputs": inputs - count})
    board_pos = np.asarray((hw.get("boards") or {}).get("position_m") or [0.0, 0.0, 0.0], dtype=float)
    slack = float((hw.get("boards") or {}).get("cable_slack_m", 0.5))
    runs = np.abs(pos - board_pos).sum(axis=1) + slack                      # along the face, then up or down, then out: no diagonals
    tx = np.asarray(cfg["transmit"].get("lever_arm", [0.0, 0.0, 0.0]), dtype=float)
    dist = np.linalg.norm(pos - tx, axis=1)
    from .sonar import PEAK_OVER_RMS_DB, source_level_db_rms
    sl = source_level_db_rms(w)                                             # RMS, dB re 1 uPa at 1 m
    sl = float("nan") if sl is None else sl
    sens, gain = float(rx.get("sensitivity_db", float("nan"))), float(rx.get("gain_db", 0.0))
    full_scale = float(rx.get("adc_full_scale", float("nan")))
    level = sl - 20.0 * np.log10(dist.min())                                # RMS level of the direct sound at the nearest hydrophone
    volts = 10.0 ** ((level + PEAK_OVER_RMS_DB + sens + gain) / 20.0)       # its PEAK at the ADC: what the full scale (a peak voltage) limits
    over = 20.0 * np.log10(volts / full_scale)
    over_each = sl + PEAK_OVER_RMS_DB - 20.0 * np.log10(dist) + sens + gain - 20.0 * np.log10(full_scale)          # every hydrophone, dB re ADC full scale
    r0, r1 = (float(x) for x in rx["window"]["range"])
    t_open, t_close = 2.0 * r0 / c, 2.0 * r1 / c + pulse
    n_samples = int(np.ceil((t_close - t_open) * fs))
    interval = t_close
    n = len(pos)
    ref = (clocks_cfg or {}).get("receive", {}).get("reference")
    baseline = float(heights[-1] - heights[0]) if len(heights) > 1 else 0.0
    eps = float((hw.get("sync") or {}).get("elevation_error_deg", 0.05))
    el = None
    if len(heights) > 1:
        from . import elevation as elev
        spacing = float(np.degrees(np.arcsin(c / (fc * baseline)))) if c / (fc * baseline) < 1.0 else None
        half = 0.5 * float(np.degrees(np.arcsin(min(c / (f_hi * baseline), 1.0)))) if spacing is not None else None
        water = (((deploy_cfg or {}).get("chain") or {}).get("elevation") or {}).get("water_depth")
        el = {"heights_m": heights, "baseline_m": baseline, "baseline_wavelengths": [baseline * f_lo / c, baseline * f_hi / c],
              "repeat_spacing_deg": spacing, "unambiguous_half_sector_deg": half,
              "highest_repeat_db": max(second_lobe_db(heights, a, f_lo, f_hi, c) for a in (0.0, 10.0, 20.0)),
              "sigma_deg": {str(s): float(np.degrees(elev.sin_elevation_sigma(np.repeat(heights, per_row[0]), 10.0 ** (s / 10.0) / per_row[0], f_lo, f_hi, c))) for s in (13.0, 20.0, 30.0)},
              "clock_us_for_error": 1e6 * baseline * np.sin(np.radians(eps)) / c, "clock_error_deg": eps,
              "deg_per_us": float(np.degrees(np.arcsin(min(1e-6 * c / baseline, 1.0)))), "deg_per_mm_out": float(np.degrees(1e-3 / baseline)),
              "two_rows_one_answer_beyond_m": None, "elevation_beamwidth_deg": float(np.degrees(0.886 * lam / baseline))}
        if len(heights) == 2 and half is not None:
            el["one_answer_beyond_m_per_m_of_water"] = 2.0 / np.tan(np.radians(half))        # one bounce: depth x 2 / tan(half sector)
            if water:
                el["two_rows_one_answer_beyond_m"] = float(water) * el["one_answer_beyond_m_per_m_of_water"]
    out = {
        "name": cfg.get("name"), "build": cfg.get("build"), "platform": plat, "hardware": hw,
        "sound_speed": c, "band_hz": [f_lo, f_hi], "centre_hz": fc, "wavelength_m": lam, "pulse_s": pulse, "source_level_db": sl,
        "channels": n, "rows": len(heights), "per_row": per_row, "heights_m": heights, "pitch_m": pitch, "row_length_m": row_length,
        "pitch_wavelengths_top": pitch * f_hi / c, "positions": pos, "row_of": row_of, "channel_board": model.channel_board,
        "inputs_per_board": inputs, "boards": boards, "n_boards": len(boards), "boards_per_row": [sum(1 for b in boards if b["rows"] == [i]) for i in range(len(heights))],
        "interleave": int(((cfg["array"].get("layouts") or {}).get(cfg["array"].get("layout_choice"), {}) or {}).get("interleave", 0)),
        "sample_rate_hz": fs, "adc_bits": bits, "adc_full_scale_v": full_scale, "sensitivity_db": sens, "gain_db": gain,
        "input_full_scale_pressure_db": 20.0 * np.log10(full_scale) - sens - gain,
        "projector_m": tx.tolist(),
        "direct": {"nearest_m": float(dist.min()), "farthest_m": float(dist.max()), "level_db": float(level), "volts": float(volts), "over_full_scale_db": float(over),
                   "starts_s": float(dist.min() / c), "ends_s": float(pulse + dist.max() / c),
                   "over_full_scale_farthest_db": float(over_each.min()), "hydrophones_overloaded": int((over_each > 0.0).sum()),
                   "gain_reduction_needed_db": float(max(over_each.max(), 0.0)), "over_full_scale_each_db": over_each,
                   # With a front end that has no memory the first usable echo ends after the direct sound has passed the farthest
                   # hydrophone; every millisecond of recovery after that adds c / 2 x 1 ms of range.
                   "min_range_memoryless_m": float(0.5 * c * (pulse + dist.max() / c)), "range_per_ms_of_recovery_m": float(0.5 * c * 1e-3),
                   "source_level_for_full_scale_db": float(sl - over), "distance_for_full_scale_m": float(dist.min() * 10.0 ** (over / 20.0))},
        "blind_range_m": 0.5 * c * pulse, "window_m": [r0, r1], "window_s": [t_open, t_close], "samples_per_ping": n_samples, "ping_interval_s": interval,
        "pings_per_hour": 3600.0 / interval,
        "rate_channel_bit_s": fs * bits, "rate_board_bit_s": inputs * fs * bits, "rate_total_bit_s": n * fs * bits,
        "stream_bytes_per_ping": n * n_samples * int(np.ceil(bits / 8)), "record_bytes_per_ping": n * n_samples * 4,
        "record_bytes_per_hour": n * n_samples * 4 * 3600.0 / interval,
        "recording_mode": (hw.get("storage") or {}).get("record", "raw"), "decimation_factor": _decimation_factor(fs, f_lo, f_hi),
        "record_bytes_per_hour_decimated": n * int(np.ceil(n_samples / _decimation_factor(fs, f_lo, f_hi))) * 8 * 3600.0 / interval, "stream_bytes_per_hour_continuous": n * fs * int(np.ceil(bits / 8)) * 3600.0,
        "cables": {"assumed_board_position_m": board_pos.tolist(), "slack_m": slack, "shortest_m": float(runs.min()), "longest_m": float(runs.max()),
                   "mean_m": float(runs.mean()), "total_m": float(runs.sum()), "reference_cables": len(boards) if ref or (hw.get("sync") or {}).get("scheme") == "shared_reference" else 0,
                   "per_row_total_m": [float(runs[row_of == i].sum()) for i in range(len(heights))]},
        "reference": ref, "sync_scheme": (hw.get("sync") or {}).get("scheme", "shared_reference" if ref else "none"),
        "alignment_tolerance_us": 1e6 / (20.0 * f_hi),             # sync.alignment_tolerance: a twentieth of a period at the top of the band (clean beams)
        "elevation": el, "deploy": deploy_cfg,
        # for `detection`: the resolved sonar file (the detector's thresholds follow from its array, pulse and window) and the design probability
        "sonar_config": cfg, "detection_probability": DESIGN_DETECTION_PROBABILITY,
        "beamwidth_deg": float(np.degrees(0.886 * lam / (per_row[0] * pitch))) if pitch else None,
        "far_field_m": 2.0 * (per_row[0] * pitch) ** 2 / lam,
        "unconfirmed": list(cfg.get("unconfirmed", [])) + [f for f in (clocks_cfg or {}).get("unconfirmed", [])],
    }
    out["record_bytes_per_hour_chosen"] = out["record_bytes_per_hour_decimated"] if out["recording_mode"] == "decimated" else out["record_bytes_per_hour"]
    if ref:
        out["reference_times_s"] = [t_open + float(ref["lead"]), t_close - float(ref["tail"]) - float(ref["duration"])]
    return out


def _decimation_factor(fs: float, f_lo: float, f_hi: float) -> int:
    """The decimate-first stage's default factor (libs/live decimate.hpp defaults: the band plus a quarter of its width each side,
    sampled twice over): 17 for 4 to 6 kHz at 102.4 kHz, as docs/design/popoto_stage2.json has it (a test holds the two together)."""
    return max(int(fs // (2.0 * 1.5 * (f_hi - f_lo))), 1)


def describe_shipped(sonar_name: str, deploy_name: str | None = None) -> dict:
    """`describe` for files under configs/: the sonar by name, its `_clocks` variant when there is one."""
    from .config import _resolve
    sonar = load_sonar_config(sonar_name)
    clocks = None
    try:
        _resolve("sonars", sonar_name + "_clocks")
        clocks = load_sonar_config(sonar_name + "_clocks")
    except FileNotFoundError:
        pass
    deploy = None
    if deploy_name:
        deploy = load_yaml(_resolve("deploy", deploy_name))
    return describe(sonar, clocks, deploy)


# --- expected performance -----------------------------------------------------------------------------

def elevation_pattern(heights, true_deg, f_lo, f_hi, c=acoustics.NOMINAL_SOUND_SPEED, sector=(-30.0, 30.0, 0.1), n_f=64):
    """Noise-free response of rows at `heights` steered over elevation with true time delays over the band
    (power, 1 at the arrival): the estimator's ambiguity function (tools/popoto_elevation_study.py: pattern)."""
    h = np.asarray(heights, dtype=float)
    f = np.linspace(f_lo, f_hi, n_f)
    grid = np.radians(np.arange(sector[0], sector[1] + 0.5 * sector[2], sector[2]))
    lag = h[:, None] * (np.sin(grid)[None, :] - np.sin(np.radians(true_deg))) / c
    p = np.abs(np.exp(-2j * np.pi * f[None, :, None] * lag[:, None, :]).sum(axis=(0, 1))) ** 2
    return np.degrees(grid), p / (len(h) * n_f) ** 2


def second_lobe_db(heights, true_deg, f_lo, f_hi, c=acoustics.NOMINAL_SOUND_SPEED, sector=(-30.0, 30.0, 0.1)) -> float:
    """Highest point of that response outside the main lobe, inside the sector, dB (-99: none)."""
    _, p = elevation_pattern(heights, true_deg, f_lo, f_hi, c, sector)
    e = int(np.argmax(p))
    lo = e
    while lo > 0 and p[lo - 1] < p[lo]:
        lo -= 1
    hi = e
    while hi < len(p) - 1 and p[hi + 1] < p[hi]:
        hi += 1
    rest = np.concatenate([p[:lo], p[hi + 1:]])
    return float(max(10.0 * np.log10(max(rest.max(), 1e-30)), -99.0)) if len(rest) else -99.0


def directivity_db(n: int, pitch: float, f: float, c: float = acoustics.NOMINAL_SOUND_SPEED) -> float:
    """Array gain of a line of n hydrophones against isotropic noise at broadside: N^2 / sum sinc(2 d_ij / lambda)
    (sonar_equation.isotropic_noise_gain_db, which SonarEquation.from_sonar uses for any array)."""
    from .sonar_equation import isotropic_noise_gain_db
    return isotropic_noise_gain_db(np.arange(n) * pitch, f, c)


# --- what the chain's detector needs of the budget's SNR -------------------------------------------------

_DETECTION_CACHE: dict = {}
_TEMPLATE_CACHE: dict = {}          # the last two templates (a three-row face's is 100 MB of zeros)


def detector_template(sonar_cfg: dict):
    """An empty record of a sonar for the detector's arithmetic in gpusonar.live (`budget_threshold_db`,
    `detection_probability`, `detector_plan`): its array, pulse, band and receive window, as complex baseband at four
    times the band so that a full-rate window is not allocated. The detector's thresholds do not depend on the rate the
    record was sampled at (the map is laid from the band and the window): tests/python/test_buildspec.py holds this
    template to the full-rate one."""
    from . import core
    from .sonar import SonarModel
    key = json.dumps({k: sonar_cfg.get(k) for k in ("array", "transmit", "nominal_sound_speed")} | {"window": sonar_cfg["receive"]["window"]}, sort_keys=True, default=str)
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]
    cfg = copy.deepcopy(sonar_cfg)
    w = cfg["transmit"]["waveform"]
    band = abs(float(w.get("f_stop", w["f_start"])) - float(w["f_start"])) or 1.0 / float(w["duration"])
    cfg["receive"] = {**cfg["receive"], "kind": "baseband", "sample_rate": 4.0 * band}
    for key in ("reference", "clock_errors", "highpass_corner_hz"):
        cfg["receive"].pop(key, None)
    model = SonarModel.from_config(cfg)
    while len(_TEMPLATE_CACHE) >= 2:
        _TEMPLATE_CACHE.pop(next(iter(_TEMPLATE_CACHE)))
    _TEMPLATE_CACHE[key] = record = model.make_record(core.Pose(), environment={"sound_speed": model.nominal_sound_speed})
    return record


def deployed_chain(d: dict) -> dict:
    """The chain settings the detector's figures are computed for: the build's deployment file's (`d["deploy"]["chain"]`),
    else what `chain_config` gives a build with nothing changed. Synchronisation is left out (it does not touch the detector)."""
    chain = copy.deepcopy(((d.get("deploy") or {}).get("chain")) or DEFAULT_CHAIN)
    chain.pop("sync", None)
    return chain


def _rows_added(d: dict, chain: dict) -> int:
    """Rows whose maps the deployed chain adds in power: it does so only when it is told to estimate elevation
    (live._rows_of); a build's `chain_config` says so for two rows or more."""
    if d.get("rows_added") is not None:                      # a what-if (tools/projector_requirements.py: the same build with fewer or more rows)
        return int(d["rows_added"])
    return int(d["rows"]) if (d.get("deploy") is None or chain.get("elevation") is not None) else 1


def detection(d: dict, pd: float | None = None, rows: int | None = None) -> dict:
    """What the chain's detector, as this build deploys it, needs of the link budget (`budget`, `performance`): the SNR of
    ONE row at which a steady point echo is detected on one ping with probability `pd` (default: the build's design
    value, `operation.detection_probability`), with `rows` rows' maps added in power (default: the build's rows).

    Computed by gpusonar.live.budget_threshold_db on the build's own template record and chain settings: nothing is typed.
    Returns {"probability", "threshold_db" (the build's rows), "threshold_one_row_db", "rows", "rows_gain_db" (what the
    rows are worth AT THIS DETECTOR: one row's threshold less theirs), "threshold_high_db" and "probability_high" (the same
    for HIGH_DETECTION_PROBABILITY), "statistic", "false_alarms_per_ping", "window_loss_db", "straddle_loss_db",
    "sample_threshold_db" (the reported snr_db a detection needs), "includes", "leaves_out"}.

    No gain is counted twice: the budget's SNR holds the pulse's matched-filter gain (10 log10 T against the noise
    spectrum level) and ONE row's gain against isotropic sea noise, unweighted; this threshold holds the aperture window,
    the straddling of the map's samples, the false-alarm budget and the rows' combination. Neither holds the other's."""
    from . import live
    pd = float(d.get("detection_probability", DESIGN_DETECTION_PROBABILITY) if pd is None else pd)
    if not 0.0 < pd < 1.0:
        raise ValueError(f"detection probability: {pd} (between 0 and 1)")
    chain = deployed_chain(d)
    rows = _rows_added(d, chain) if rows is None else int(rows)
    cfg = d["sonar_config"]
    key = json.dumps([cfg.get("array"), cfg["transmit"], cfg["receive"]["window"], cfg.get("nominal_sound_speed"), chain, pd, rows], sort_keys=True, default=str)
    if key in _DETECTION_CACHE:
        return copy.deepcopy(_DETECTION_CACHE[key])
    record = detector_template(cfg)
    n = d["per_row"][0]
    full = live._merge(live.DEFAULTS, chain)

    def need(p, r):
        return float(live.budget_threshold_db(record, chain, pd=p, rows=r, row_elements=n))

    one, mine = need(pd, 1), need(pd, rows)
    statistic = full["cell_statistic"]
    loss = live.snr_loss_db(record, float(live._map_geometry(record, full)["cell"]), full["window"], statistic, n)
    out = {"probability": pd, "threshold_db": mine, "threshold_one_row_db": one, "rows": rows, "rows_gain_db": one - mine,
           "probability_high": HIGH_DETECTION_PROBABILITY, "threshold_high_db": need(HIGH_DETECTION_PROBABILITY, rows),
           "statistic": statistic, "false_alarms_per_ping": float(full["cfar"]["false_alarms_per_ping"]) if statistic == "multiscale" else None,
           "window_loss_db": loss["window_db"], "straddle_loss_db": loss["cell_db"],
           "sample_threshold_db": float(live.detection_threshold_db(chain, record=record, rows=rows)) if statistic == "multiscale" else None,
           "includes": "the aperture window's loss, where the echo falls between the map's samples, the detector's thresholds for its false-alarm budget over this build's window, "
                       "the rows' maps added in power",
           "leaves_out": "an echo that fluctuates from ping to ping, an echo longer than the range resolution, reverberation, the sweep's taper, noise shared between rows"}
    _DETECTION_CACHE[key] = out
    return copy.deepcopy(out)


def detection_for_row(n: int, pitch: float, rows: int = 1, pulse_s: float | None = None, window_m=None, base_sonar: str = "popoto_long_range", pd: float | None = None) -> dict:
    """`detection` for a row that is not a build: `n` hydrophones `pitch` m apart, `rows` of them added in power, with the
    band of `base_sonar`, its pulse and receive window unless given, and the chain's default settings. For trade studies
    (tools/popoto_elevation_study.py), so that a hypothetical row is held to the same detector as a build."""
    from .sonar import SonarModel
    cfg = copy.deepcopy(SonarModel.load(base_sonar).config)
    cfg["array"] = {"layout": "uniform_line", "count": int(n), "pitch": float(pitch), "axis": [1, 0, 0], "boresight": [0, 1, 0]}
    for key in ("boards", "board_size", "clock_errors", "reference"):
        cfg["receive"].pop(key, None)
    if pulse_s is not None:
        cfg["transmit"]["waveform"]["duration"] = float(pulse_s)
    if window_m is not None:
        cfg["receive"]["window"] = {"range": [float(window_m[0]), float(window_m[1])]}
    d = {"rows": int(rows), "per_row": [int(n)], "sonar_config": cfg, "deploy": None}
    return detection(d, pd, rows)


def detection_probability(d: dict, row_snr_db, rows: int | None = None):
    """The probability that the build's chain detects, on one ping, a steady point echo whose link-budget SNR of one row
    is `row_snr_db` (a number or an array): gpusonar.live.detection_probability on the build's template and chain."""
    from . import live
    chain = deployed_chain(d)
    rows = _rows_added(d, chain) if rows is None else int(rows)
    return live.detection_probability(row_snr_db, detector_template(d["sonar_config"]), chain, rows=rows, row_elements=d["per_row"][0])


def budget(d: dict, wind: float = 5.0, target_strength_db: float = 0.0):
    """(SonarEquation, gains) of a build: the link budget of docs/DESIGN.md Sec. 5 for ONE of its rows (spherical
    spreading, Ainslie-McColm absorption at 12 C and 50 m, ambient noise at `wind` m/s, the row's gain against isotropic
    noise, an RMS source level). `eq.snr(r)` is the free-field SNR of one row: peak signal over mean noise after the
    matched filter and an unweighted row beamformer. `gains["threshold_db"]` is what the build's detector needs of it
    (`detection`): margin = eq.snr(r) - gains["threshold_db"]. No row gain is in the equation: the rows are in the threshold."""
    from .sim.noise import wenz_spectrum_db
    from .sonar_equation import SonarEquation
    c, (f_lo, f_hi), fc = d["sound_speed"], d["band_hz"], d["centre_hz"]
    n, pitch = d["per_row"][0], d["pitch_m"]
    noise = float(wenz_spectrum_db(np.array([fc]), wind, 0.5)[0])
    di = directivity_db(n, pitch, fc, c)
    det = detection(d)
    eq = SonarEquation(d["source_level_db"], fc, f_hi - f_lo, d["pulse_s"], n, noise, target_strength_db=target_strength_db, temperature=12.0, depth=50.0,
                       absorption_model="ainslie_mccolm", array_gain_db=di)
    return eq, {"noise_spectrum_db": noise, "gain_isotropic_noise_db": di, "rows_detection_gain_db": det["rows_gain_db"], "wind_speed_m_s": float(wind),
                "threshold_db": det["threshold_db"], "detection_probability": det["probability"]}


def performance(d: dict, wind: float = 5.0, ranges=(500.0, 1000.0, 3000.0, 6000.0, 12000.0), roll_sigma_deg: float | None = None,
                gradient_sigma: float | None = None, environment: dict | None = None, own_depth_m: float | None = None, classes: list | None = None) -> dict:
    """Expected performance of a build from its derived numbers `d` (describe / derive): the link budget of
    docs/DESIGN.md Sec. 5 (spherical spreading, Ainslie-McColm absorption at 12 C and 50 m, ambient noise at
    `wind` m/s) with the row's gain against isotropic noise, held against what the build's own detector needs
    (`detection`): thresholds, margins, detection probabilities and reaches; and the elevation bounds of the stacked rows.
    `row_snr_db` is the SNR of one row; `margin_db` = row_snr_db - detection.threshold_db;
    `rows_detection_gain_db` is what the rows are worth at the detector (already inside the threshold: add it to nothing)."""
    c, (f_lo, f_hi), fc = d["sound_speed"], d["band_hz"], d["centre_hz"]
    n, pitch, rows = d["per_row"][0], d["pitch_m"], d["rows"]
    _, g = budget(d, wind)
    det = detection(d)
    noise, di = g["noise_spectrum_db"], g["gain_isotropic_noise_db"]
    thr = det["threshold_db"]
    out = {"assumptions": f"spherical spreading, Ainslie-McColm absorption (12 C, 50 m), ambient noise at {wind:g} m/s wind, an RMS source level, a steady point echo; detection: "
                          f"probability {det['probability']:g} on one ping by the chain's own detector as deployed, which needs {thr:.1f} dB of one row's SNR"
                          + (f" with {det['rows']} rows added" if det["rows"] > 1 else "") + f" ({det['threshold_high_db']:.1f} dB for {det['probability_high']:g}); "
                          "target strengths +25 dB (ship) and +5 dB (small)",
           "noise_spectrum_db": noise, "beamwidth_deg": d["beamwidth_deg"], "gain_isotropic_noise_db": di, "gain_uncorrelated_noise_db": float(10 * np.log10(n)),
           "rows_detection_gain_db": det["rows_gain_db"], "detection": det, "range_resolution_m": c / (2.0 * (f_hi - f_lo)), "blind_range_m": d["blind_range_m"],
           "grating_lobe_free_steering_deg": float(np.degrees(np.arcsin(min(c / f_hi / pitch - 1.0, 1.0)))) if c / f_hi / pitch > 1.0 else 0.0}
    snr, margin, prob = {}, {}, {}
    for key, ts in (("ship", 25.0), ("small", 5.0)):
        eq, _ = budget(d, wind, ts)
        out[f"reach_{key}_m"] = float(eq.detection_range(thr))
        out[f"reach_{key}_high_m"] = float(eq.detection_range(det["threshold_high_db"]))
        snr[key] = {str(r): float(eq.snr(r)) for r in ranges}
        margin[key] = {k: v - thr for k, v in snr[key].items()}
        prob[key] = dict(zip(snr[key], np.atleast_1d(detection_probability(d, np.array(list(snr[key].values())), det["rows"])).tolist()))
    out["row_snr_db"], out["margin_db"], out["detection_probability"] = snr, margin, prob
    el = d["elevation"]
    if el:
        from . import elevation as elev
        dep = (((d.get("deploy") or {}).get("chain") or {}).get("elevation") or {})
        roll = float(dep.get("roll_sigma_deg", 0.2) if roll_sigma_deg is None else roll_sigma_deg)
        grad = float(dep.get("gradient_sigma_per_s", 0.05) if gradient_sigma is None else gradient_sigma)
        out["elevation"] = {k: el[k] for k in ("baseline_m", "repeat_spacing_deg", "unambiguous_half_sector_deg", "highest_repeat_db", "sigma_deg", "clock_us_for_error",
                                               "clock_error_deg", "deg_per_us", "deg_per_mm_out", "two_rows_one_answer_beyond_m")}
        depth_budget = {}
        for r in ranges:
            s = snr["ship"][str(r)]
            if s < 10.0:
                depth_budget[str(r)] = None
                continue
            sig = float(elev.sin_elevation_sigma(np.repeat(el["heights_m"], n), 10.0 ** (min(s, 40.0) / 10.0) / n, f_lo, f_hi, c))
            parts = {"from_noise_m": r * sig, "from_roll_m": r * np.radians(roll), "from_gradient_m": r * r * grad / (2.0 * c)}
            depth_budget[str(r)] = {**{k: float(v) for k, v in parts.items()}, "total_m": float(np.sqrt(sum(v * v for v in parts.values())))}
        out["depth_sigma_ship_m"] = depth_budget
        out["depth_budget_inputs"] = {"roll_sigma_deg": roll, "gradient_sigma_per_s": grad}
        out["multipath_note"] = ("not in any sigma: in a shallow channel an echo that bounced once arrives within the pulse's range sidelobes of the direct one and pulls the "
                                 "elevation by tenths of a degree (docs/DESIGN.md Sec. 3)")
    if environment is not None:
        out["surface"] = surface_term(d, environment, own_depth_m, classes, ranges)
    return out


def surface_term(d: dict, environment: dict, own_depth_m: float | None = None, classes: list | None = None, ranges=(500.0, 1000.0, 3000.0, 6000.0, 12000.0),
                 threshold_db: float | None = None) -> dict:
    """What the free-field figures of `performance` leave out, for one environment (depthplan.ENVIRONMENT_FIELDS) and one
    own depth (default: the shallowest allowed): the two-way propagation factor of gpusonar.lloyd (surface image, roughness,
    bubble loss, the band; the bottom's images when the environment has one) at each range, and the SNR and margin with it.
    `classes` (resolved target classes): one entry each, worst over the class's depths, at the low end of its target
    strength. SNRs are one row's; `threshold_db` (default: the build's detector's, `detection`) is what the margin is over. Without classes: the reference ship (+25 dB) at 4 m, the popoto_12km scenario's."""
    from . import depthplan
    own = float(depthplan.own_depth_grid(environment, max_depth=(d.get("platform") or {}).get("max_depth_m"))[0] if own_depth_m is None else own_depth_m)
    eq, gains = budget(d, depthplan.wind_of(environment))
    threshold_db = gains["threshold_db"] if threshold_db is None else float(threshold_db)       # of one row's SNR, the build's detector's (`detection`)
    rr = np.asarray([r for r in ranges if r > d["blind_range_m"]], dtype=float)
    f_lo, f_hi = d["band_hz"]
    items = [dict(c) for c in (classes or [])] or [{"name": "ship", "ts_db": [25.0, 25.0], "depth_m": [4.0, 4.0]}]
    out = {"environment": environment.get("name"), "own_depth_m": own, "wind_speed_m_s": depthplan.wind_of(environment), "rms_wave_height_m": depthplan.surface_of(environment).height(),
           "threshold_db": threshold_db, "ranges_m": rr.tolist(), "classes": {}}
    for c in items:
        lo, hi = c["depth_m"]
        if environment.get("water_depth_m") is not None:
            hi = min(hi, float(environment["water_depth_m"]) - 1.0)
            if lo > hi:
                continue
        zt = np.unique(np.linspace(lo, hi, 9))
        fac = depthplan.factor(environment, f_lo, f_hi, [own], zt, rr, float(d["projector_m"][2]), eq.absorption_db_per_m())
        out["method"] = fac["method"]
        worst = fac["peak_db"][0].min(axis=0)
        free = np.asarray(eq.snr(rr)) + c["ts_db"][0]
        def by_range(v):
            return {str(float(r)): float(x) for r, x in zip(rr, v)}
        out["classes"][c["name"]] = {"ts_db": c["ts_db"][0], "depth_m": [float(lo), float(hi)], "factor_db": by_range(worst),
                                     "factor_narrowband_db": by_range(fac["narrowband_db"][0].min(axis=0)), "factor_energy_db": by_range(fac["energy_db"][0].min(axis=0)),
                                     "snr_free_field_db": by_range(free), "snr_db": by_range(free + worst), "margin_db": by_range(free + worst - threshold_db)}
    return out


# --- measured timings ---------------------------------------------------------------------------------

def measured_timings() -> dict:
    """Processing time per ping by rows (1, 2, 3 of 32) from the measurement files in the repository, each with what
    it was measured on. Nothing is extrapolated: a build with another channel count gets no number from here."""
    out = {"cpu": None, "cpu_full_rate": None, "gpu": None, "stage2": None}
    bench = ROOT / "docs" / "design" / "popoto_elevation_bench.json"
    if bench.exists():
        b = json.loads(bench.read_text())
        out["cpu"] = {"seconds": {k: v["seconds_per_ping"] for k, v in b["timing_scenario_size"].items()}, "threads": b["cpu_threads"], "note": b.get("timing_note", ""),
                      "what": "C++ chain, pings of the popoto_12km scenario's size (baseband, decimated)"}
        if b.get("timing_full_length"):
            out["cpu_full_rate"] = {"seconds": {k: v["seconds_per_ping"] for k, v in b["timing_full_length"].items()}, "threads": b["cpu_threads"],
                                    "what": "C++ chain, one full-length ping at the boards' rate (raw, 0.5 to 13 km), earlier row heights"}
    gpu = ROOT / "docs" / "benchmarks" / "gpu_benchmark_all_l4.json"
    if gpu.exists():
        g = json.loads(gpu.read_text())
        sec, host, sizes = {}, {}, None
        for case in g["cases"]:
            if case["group"] == "live" and case["case"].endswith("engine cpp, doa map, backprojection"):
                r = str(case["sizes"]["rows"])
                sec[r], host[r], sizes = case["cuda_s"], case["host_s"], case["sizes"]
        head = g.get("header", {})
        out["gpu"] = {"seconds": sec, "host_seconds": host, "gpu": head.get("gpu", "NVIDIA L4") if isinstance(head, dict) else "NVIDIA L4",
                      "samples": sizes["samples"] if sizes else None, "header": head,
                      "what": "C++ chain, bearings from the map, backprojection; median seconds per ping after the first"}
    stage2 = ROOT / "docs" / "design" / "popoto_stage2.json"
    if stage2.exists():
        out["stage2"] = json.loads(stage2.read_text())
    return out
