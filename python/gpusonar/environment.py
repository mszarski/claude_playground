"""The site a sonar works in, described once: water depth, bottom, surface state, sound speed, water
properties, absorption model and ambient-noise inputs.

ONE schema (`FIELDS`) serves the two layers that need a site:

- a scenario (`environment:` in configs/scenarios/*.yaml): inline, or the name of an environment (a file under
  configs/environments/, or an entry of a list file there or under configs/builds/environments/), or
  `{base: <name>, ...}` to change some fields of a named one. `apply_to_scenario` then gives the site's values to
  everything that needs them: `propagation` (image-source boundaries, the Bellhop ocean and its frequency),
  `time_spread`, each `reverberation` item, `receiver.ambient_noise` and `absorption`;
- a build spec (`environments:` in configs/builds/*.yaml; planner in gpusonar/depthplan.py), a list of the same
  mappings with the planning fields (`own_depth_limits_m`, `bottom_clearance_m`).

This is the SIMULATED site: the truth the synthetic sound is made in. It is not what a running sonar knows about
the water. That is the live chain's business (libs/live/environment.hpp: the sound-speed ladder a processor
climbs from a nominal value to a measured profile) and nothing here feeds it, except the two numbers a record
has always carried (`sound_speed`, and `water_depth` unless `water_depth_known: false`).

Field names carry their unit (`water_depth_m`). The scenario layer's older spellings (`water_depth`, `sound_speed`,
`temperature`, `salinity`, `wind_speed`, `depth`) are accepted as aliases of the same fields (`ALIASES`); giving a
field under both spellings with different values is an error.

A value typed where the environment already gives it (a `wind_speed` in a reverberation item, a `frequency` in
`propagation`) is checked: equal is fine, different is an error unless the scenario names it under
`environment_overrides` with its reason:

    environment_overrides:
      propagation.wind_speed: "the eigenrays reflect from a flat surface; roughness is carried by time_spread"

Inside an environment, `overrides: {sound_speed_m_s: "..."}` does the same for fields of one environment that
do not agree with each other (a typed sound speed beside a temperature and salinity that imply another).
"""

from __future__ import annotations

import copy
import warnings

import numpy as np

from . import acoustics
from .acoustics import NOMINAL_SOUND_SPEED
from .config import config_dirs, load_yaml

# The water the formulas assume when an environment does not say (absorption, sound speed from temperature).
# One definition; `resolve` records which were used (`defaulted`).
DEFAULT_TEMPERATURE_C = 10.0
DEFAULT_SALINITY_PSU = 35.0
DEFAULT_REFERENCE_DEPTH_M = 0.0
DEFAULT_PH = 8.0
# The ambient-noise inputs used when a Wenz spectrum is asked for and the environment gives none.
DEFAULT_WIND_SPEED_M_S = 5.0
DEFAULT_SHIPPING = 0.5
DEFAULT_ABSORPTION = "francois_garrison"
SOUND_SPEED_TOLERANCE_M_S = 1.0         # a typed speed this far from the one its temperature implies is reported

# field -> (kind, unit, required, meaning, where to find it, what to do if unknown)
FIELDS = {
    "name": ("str", "", True, "short name (letters, digits, underscores)", "", "must be given"),
    "title": ("str", "", False, "one line for headings", "", "the name"),
    "water_depth_m": ("num", "m", False, "water depth. Left out or null: no bottom return at all (deep water, or a bottom that returns nothing): the surface image only", "chart / echo sounder", "leave it out: the non-ducting case"),
    "bottom": ("str or map", "", False, "the sea bed: a name in configs/sediments.yaml, or {speed m/s, density re water, attenuation dB/wavelength}. Needs water_depth_m", "site survey", "leave it out with water_depth_m: the bottom then only limits own depth"),
    "wind_speed_m_s": ("num", "m/s", False, "wind at 10 m: ambient noise, the wind law for the surface's rms height (when rms_wave_height_m is not given) and the bubble law", "forecast / trial plan", "5 (the link budget's)"),
    "sea_state": ("int", "", False, "0 to 9, in place of wind_speed_m_s: its median wind speed (acoustics.SEA_STATE_WIND_SPEED)", "forecast", "give wind_speed_m_s"),
    "rms_wave_height_m": ("num", "m", False, "rms displacement of the surface (a quarter of the significant wave height). An input of its own: measured heights did not follow the wind law (PHYSICS_NOTES Sec. 21)",
                          "wave buoy / forecast: significant height / 4", "leave it out: the Pierson-Moskowitz height for the wind speed"),
    "bubble_model": ("str", "", False, "wind law for the loss at each surface bounce: apl_1994 (sim/surface.py), or null for none", "", "null: no bubble loss"),
    "sound_speed_m_s": ("num", "m/s", False, "sound speed of isovelocity water", "CTD", "1500"),
    "profile": ("map", "", False, "{depth_m: [...], speed_m_s: [...]}: a sound-speed profile. The factor then comes from Bellhop (GPUSONAR_BELLHOP) and needs water_depth_m", "SVP cast / climatology", "leave it out: isovelocity"),
    "profiles_by_range": ("list of maps", "", False, "[{range_m, depth_m: [...], speed_m_s: [...]}, ...] in place of `profile`: water that changes with horizontal distance from the sonar, the same on every "
                          "bearing (Bellhop's 2-D sound-speed field). The first is at range_m 0: the LOCAL profile, the one the vehicle measures; ranges increase; every profile is on the same "
                          "depth grid, from 0 to the bottom or deeper. Linear in range between two profiles; the last holds beyond its range", "SVP casts along the line / an ocean model", "leave it out: the same water everywhere"),
    "temperature_c": ("num", "deg C", False, "water temperature: absorption, and the sound speed when none is given", "CTD", "10 for absorption (flagged in `defaulted`)"),
    "salinity_psu": ("num", "PSU", False, "salinity: absorption, sound speed; under 5 is fresh water", "CTD", "35"),
    "ph": ("num", "", False, "acidity, for the boric-acid term of absorption", "climatology", "8"),
    "reference_depth_m": ("num", "m", False, "depth at which absorption and the sound-speed formula are evaluated", "the sonar's working depth", "0"),
    "absorption": ("str", "", False, "absorption model: francois_garrison, ainslie_mccolm, thorp or none", "", "francois_garrison"),
    "shipping": ("num", "", False, "shipping activity for the ambient-noise spectrum, 0 (light) to 1 (heavy)", "site knowledge", "0.5"),
    "water_depth_known": ("bool", "", False, "false: the simulated vehicle is not told the water depth (the record's environment.water_depth stays unset; a processor has it from navigation or not at all)", "", "true"),
    "frame_origin_lla": ("[lat, lon, alt]", "deg, deg, m", False, "where the local frame's origin is on the globe (written into records)", "survey plan", "leave it out"),
    "own_depth_limits_m": ("[shallowest, deepest]", "m", False, "depths the platform may operate at in this water (a static frame: depths it can be hung at); never deeper than the build's platform.max_depth_m", "the operating rules",
                           "5 m to the build's platform.max_depth_m (300 m when that is not given: a placeholder, flagged)"),
    "bottom_clearance_m": ("num", "m", False, "how close to the bottom the platform may come", "operating rules", "10: a placeholder, flagged"),
    "illustrative": ("bool", "", False, "true: nobody measured this environment; it is an example", "", "true"),
    "note": ("str", "", False, "where the values come from, in words", "", "leave it out"),
    "overrides": ("map", "", False, "{field: reason}: fields of this environment stated on purpose although another field implies a different value (a nominal sound speed beside a temperature)", "", "leave it out"),
    "base": ("str", "", False, "the named environment this one changes (scenario layer)", "", "leave it out"),
}
# What a scenario may say beyond the table's words (docs/CONFIGURATION.md, "Environment"): `name` is optional inline;
# `sound_speed_m_s` left out is the profile (the local one of `profiles_by_range`) at reference_depth_m, else Mackenzie/Marczak from temperature_c and
# salinity_psu, else acoustics.NOMINAL_SOUND_SPEED, and beside a profile it is the speed at the sonar (what a record
# carries); `profile` may be a named family: {family: munk[, axis_depth_m, axis_speed_m_s, points]} or
# {family: linear, surface_speed_m_s, gradient_per_s}.
# The fields the depth planner reads (the build layer's guide lists these; gpusonar.depthplan.ENVIRONMENT_FIELDS).
PLANNING_FIELDS = ("name", "title", "water_depth_m", "bottom", "wind_speed_m_s", "sea_state", "rms_wave_height_m", "bubble_model", "sound_speed_m_s",
                   "profile", "own_depth_limits_m", "bottom_clearance_m", "illustrative", "note")
# Older spellings of the scenario layer, accepted everywhere an environment is read.
ALIASES = {"water_depth": "water_depth_m", "sound_speed": "sound_speed_m_s", "temperature": "temperature_c", "salinity": "salinity_psu",
           "wind_speed": "wind_speed_m_s", "rms_wave_height": "rms_wave_height_m", "depth": "reference_depth_m"}
PROFILE_FAMILIES = ("munk", "linear")
SURFACE_REVERB_MODELS = ("chapman_harris", "kraus_surface", "bubble_layer_surface")
BOTTOM_REVERB_MODELS = ("lambert", "searay_bottom", "fluid_sediment")


_TOLD: set = set()


class EnvironmentError(ValueError):       # noqa: A001 (a ValueError: existing callers catch that)
    """An environment, or a scenario's use of one, that does not hold together."""


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _same(a, b) -> bool:
    """Two configured values agree (numbers to a part in 1e9, mappings and lists item by item)."""
    if _num(a) and _num(b):
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def normalise(env: dict | None, where: str = "environment", strict: bool = True) -> dict:
    """A copy of an environment mapping under the canonical field names (`ALIASES` resolved). `strict`: unknown
    fields are an error (a scenario or a file); otherwise they pass through (a mapping built in code)."""
    out = {}
    for key, value in (env or {}).items():
        canon = ALIASES.get(key, key)
        if strict and canon not in FIELDS:
            raise EnvironmentError(f"{where}.{key}: unknown field; fields are {', '.join(FIELDS)} (older spellings: {', '.join(ALIASES)})")
        if canon in out and not _same(out[canon], value):
            raise EnvironmentError(f"{where}: {canon} is given twice ({key} and another spelling) with different values")
        out[canon] = copy.deepcopy(value)
    return out


# --- named environments -------------------------------------------------------------------------------

def _environment_files() -> list:
    out = []
    for root in config_dirs():
        for sub in (("environments",), ("builds", "environments")):
            d = root.joinpath(*sub)
            if d.is_dir():
                out += sorted(d.glob("*.yaml"))
    return out


def known() -> list:
    """Names a scenario can give: files of configs/environments/ and entries of the list files."""
    names = []
    for path in _environment_files():
        data = load_yaml(path)
        if isinstance(data.get("environments"), list):
            names += [str(e.get("name")) for e in data["environments"] if isinstance(e, dict)]
        else:
            names.append(path.stem)
    return sorted(set(names))


def load_named(name: str, _seen=()) -> dict:
    """The named environment, canonical field names, its own `base` resolved. A file configs/environments/<name>.yaml
    (one mapping) wins; else an entry of that name in a list file (`environments: [...]`) there or under
    configs/builds/environments/."""
    if name in _seen:
        raise EnvironmentError(f"environment inheritance loop at {name!r}")
    found = None
    files = _environment_files()
    for path in files:
        if path.stem == name and path.parent.name == "environments" and path.parent.parent.name != "builds":
            data = load_yaml(path)
            if not isinstance(data.get("environments"), list):
                found = dict(data)
                found.setdefault("name", name)
                break
    if found is None:
        for path in files:
            data = load_yaml(path)
            for e in data.get("environments") or [] if isinstance(data.get("environments"), list) else []:
                if isinstance(e, dict) and e.get("name") == name:
                    found = dict(e)
                    break
            if found is not None:
                break
    if found is None:
        raise FileNotFoundError(f"no environment {name!r}; known: {', '.join(known())}")
    return _with_base(normalise(found, f"environment {name!r}"), _seen + (name,))


def _with_base(env: dict, _seen=()) -> dict:
    base = env.pop("base", None)
    if base is None:
        return env
    parent = load_named(str(base), _seen)
    out = dict(parent)
    if "sea_state" in env or "wind_speed_m_s" in env:           # the two say the same thing: the child's replaces both
        out.pop("sea_state", None)
        out.pop("wind_speed_m_s", None)
    if "profile" in env and "sound_speed_m_s" not in env:
        out.pop("sound_speed_m_s", None)
    if "profiles_by_range" in env:                               # the child's water replaces the parent's, however the parent wrote it
        if "sound_speed_m_s" not in env:
            out.pop("sound_speed_m_s", None)
        out.pop("profile", None)
    elif "profile" in env:
        out.pop("profiles_by_range", None)
    overrides = {**(parent.get("overrides") or {}), **(env.get("overrides") or {})}
    out.update(env)
    if overrides:
        out["overrides"] = overrides
    return out


def load(value, where: str = "environment") -> dict:
    """An environment as a scenario gives it (None, a name, or a mapping with an optional `base`), canonical names."""
    if value is None:
        return {}
    if isinstance(value, str):
        return load_named(value)
    if not isinstance(value, dict):
        raise EnvironmentError(f"{where}: give a mapping or the name of an environment")
    return _with_base(normalise(value, where))


def load_list(value) -> list:
    """The `environments` value of a build spec as a list: a list as written, or the named list file's entries
    (configs/builds/environments/<name>.yaml or configs/environments/<name>.yaml)."""
    if value is None:
        return []
    if isinstance(value, str):
        for path in _environment_files():
            if path.stem == value:
                data = load_yaml(path)
                if isinstance(data.get("environments"), list):
                    return list(data["environments"])
        lists = sorted({p.stem for p in _environment_files() if isinstance(load_yaml(p).get("environments"), list)})
        raise FileNotFoundError(f"no environment file {value!r}; known: {', '.join(lists)}")
    return list(value)


# --- checking -----------------------------------------------------------------------------------------

def problems(envs, named: bool = True) -> list:
    """(field, message) for everything wrong with a list of environments. `named`: each needs a unique name (a
    build spec's list); a scenario's inline environment needs none."""
    if not isinstance(envs, list):
        return [("environments", "must be a list of environments ({name, water_depth_m, bottom, wind_speed_m_s, ...}) or the name of a file under configs/builds/environments/")]
    out, seen = [], set()
    for i, e in enumerate(envs):
        where = f"environments[{i}]"
        if not isinstance(e, dict):
            out.append((where, "must be a mapping"))
            continue
        name = e.get("name")
        where = f"environments.{name}" if isinstance(name, str) else where
        canon = {}
        for key, value in e.items():
            c = ALIASES.get(key, key)
            if c not in FIELDS:
                out.append((f"{where}.{key}", f"unknown field; fields are {', '.join(FIELDS)}"))
            elif c in canon and not _same(canon[c], value):
                out.append((f"{where}.{key}", f"given twice (as {c} too) with different values"))
            canon[c] = value
        e = canon
        if named or name is not None:
            if not isinstance(name, str) or not name.replace("_", "").isalnum():
                out.append((f"{where}.name", "give the environment a name of letters, digits and underscores"))
            elif name in seen:
                out.append((f"{where}.name", "two environments have this name"))
            seen.add(name)
        depth = e.get("water_depth_m")
        if depth is not None and not (_num(depth) and depth > 0):
            out.append((f"{where}.water_depth_m", f"is {depth!r}; a depth in metres, or leave it out for no bottom return"))
            depth = None
        bottom = e.get("bottom")
        if bottom is not None:
            if depth is None:
                out.append((f"{where}.bottom", "a bottom needs water_depth_m"))
            if isinstance(bottom, str):
                from .sim import bottom as bottom_model
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        bottom_model.sediment_reflection(bottom)
                except (KeyError, ValueError) as exc:
                    out.append((f"{where}.bottom", f"{bottom!r} is not a sediment of configs/sediments.yaml ({exc})"))
            elif not (isinstance(bottom, dict) and all(_num(bottom.get(k)) for k in ("speed", "density"))):
                out.append((f"{where}.bottom", "give a sediment name of configs/sediments.yaml or {speed (m/s), density (re water), attenuation (dB/wavelength)}"))
        if e.get("sea_state") is not None and e.get("wind_speed_m_s") is not None:
            out.append((f"{where}.sea_state", "give sea_state or wind_speed_m_s, not both"))
        if e.get("sea_state") is not None:
            try:
                acoustics.sea_state_wind_speed(e["sea_state"])
            except ValueError as exc:
                out.append((f"{where}.sea_state", str(exc)))
        for key in ("wind_speed_m_s", "rms_wave_height_m", "bottom_clearance_m", "salinity_psu", "reference_depth_m"):
            if e.get(key) is not None and not (_num(e[key]) and e[key] >= 0):
                out.append((f"{where}.{key}", f"is {e[key]!r}; a number >= 0 ({FIELDS[key][1]})"))
        if e.get("rms_wave_height_m") is not None and _num(e["rms_wave_height_m"]) and e["rms_wave_height_m"] > 5.0:
            out.append((f"{where}.rms_wave_height_m", f"{e['rms_wave_height_m']} m rms is a significant height of {4 * e['rms_wave_height_m']:g} m: is this the significant height? The field is the rms displacement (a quarter of it)"))
        if e.get("bubble_model") is not None:
            from .sim import surface as surface_model
            if e["bubble_model"] not in surface_model.BUBBLE_LOSS_MODELS:
                out.append((f"{where}.bubble_model", f"{e['bubble_model']!r}: one of {', '.join(sorted(surface_model.BUBBLE_LOSS_MODELS))}, or null"))
        lim = e.get("own_depth_limits_m")
        if lim is not None and not (isinstance(lim, list) and len(lim) == 2 and all(_num(x) for x in lim) and 0 < lim[0] < lim[1]):
            out.append((f"{where}.own_depth_limits_m", f"is {lim!r}; give [shallowest, deepest] in metres"))
        elif lim is not None and depth is not None and lim[0] >= depth:
            out.append((f"{where}.own_depth_limits_m", f"the shallowest operating depth ({lim[0]:g} m) is below the bottom ({depth:g} m)"))
        prof = e.get("profile")
        if prof is not None:
            if isinstance(prof, dict) and prof.get("family") is not None:
                if prof["family"] not in PROFILE_FAMILIES:
                    out.append((f"{where}.profile", f"family {prof['family']!r}: one of {', '.join(PROFILE_FAMILIES)}"))
                elif prof["family"] == "linear" and not all(_num(prof.get(k)) for k in ("surface_speed_m_s", "gradient_per_s")):
                    out.append((f"{where}.profile", "a linear profile needs surface_speed_m_s and gradient_per_s"))
            else:
                ok = isinstance(prof, dict) and isinstance(prof.get("depth_m"), list) and isinstance(prof.get("speed_m_s"), list) and len(prof["depth_m"]) == len(prof["speed_m_s"]) >= 2
                if not ok or any(b <= a for a, b in zip(prof["depth_m"], prof["depth_m"][1:])):
                    out.append((f"{where}.profile", "give {depth_m: [...], speed_m_s: [...]} of equal length, depths increasing, or {family: munk | linear, ...}"))
            if depth is None:
                out.append((f"{where}.profile", "a sound-speed profile needs water_depth_m (Bellhop needs a bottom)"))
        if e.get("profiles_by_range") is not None:
            out += [(f"{where}.profiles_by_range", m) for m in _profiles_by_range_problems(e["profiles_by_range"], depth, prof is not None)]
        for key in ("sound_speed_m_s", "temperature_c", "ph", "shipping"):
            if e.get(key) is not None and not _num(e[key]):
                out.append((f"{where}.{key}", f"is {e[key]!r}; a number"))
        if e.get("sound_speed_m_s") is not None and _num(e["sound_speed_m_s"]) and not 1300.0 <= e["sound_speed_m_s"] <= 1700.0:
            out.append((f"{where}.sound_speed_m_s", f"{e['sound_speed_m_s']} m/s is not a speed of sound in water (1300 to 1700)"))
        if e.get("absorption") is not None and e["absorption"] not in (*acoustics._MODELS, "none"):
            out.append((f"{where}.absorption", f"{e['absorption']!r}: one of {', '.join(acoustics._MODELS)}, none"))
        if e.get("overrides") is not None and not (isinstance(e["overrides"], dict) and all(k in FIELDS for k in e["overrides"])):
            out.append((f"{where}.overrides", "a mapping {field: reason} of this environment's own fields"))
    return out


def _profiles_by_range_problems(sections, depth, beside_profile: bool) -> list:
    """What is wrong with a `profiles_by_range` value (messages; empty: nothing)."""
    form = "give [{range_m, depth_m: [...], speed_m_s: [...]}, ...]: two profiles or more, the first at range_m 0, ranges increasing, all on one depth grid"
    if beside_profile:
        return ["give `profile` (the same water everywhere) or `profiles_by_range` (water that changes with range), not both"]
    if not (isinstance(sections, list) and len(sections) >= 2 and all(isinstance(s, dict) for s in sections)):
        return [form]
    out = []
    for i, s in enumerate(sections):
        extra = sorted(set(s) - {"range_m", "depth_m", "speed_m_s"})
        if extra:
            out.append(f"profile {i}: unknown keys {extra} (range_m, depth_m, speed_m_s)")
        z, c = s.get("depth_m"), s.get("speed_m_s")
        if not (_num(s.get("range_m")) and isinstance(z, list) and isinstance(c, list) and len(z) >= 2 and all(_num(v) for v in z) and all(_num(v) for v in c)):
            out.append(f"profile {i}: {form}")
        elif len(z) != len(c):
            out.append(f"profile {i}: {len(z)} depths and {len(c)} speeds")
        elif any(not 1300.0 <= v <= 1700.0 for v in c):
            out.append(f"profile {i}: a speed outside 1300 to 1700 m/s is not a speed of sound in water")
    if out:
        return out
    ranges = [float(s["range_m"]) for s in sections]
    grid = sections[0]["depth_m"]
    if ranges[0] != 0.0:
        out.append(f"the first profile is at range_m {ranges[0]:g}; it must be at 0: the local profile, the water at the sonar (nothing is extended towards the sonar)")
    if any(b <= a for a, b in zip(ranges, ranges[1:])):
        out.append(f"ranges must increase; they are {ranges}")
    if grid[0] != 0 or any(b <= a for a, b in zip(grid, grid[1:])):
        out.append("depth_m must start at 0 (the surface) and increase")
    for i, s in enumerate(sections[1:], 1):
        if not _same(s["depth_m"], grid):
            out.append(f"profile {i} (range_m {ranges[i]:g}) is on another depth grid than profile 0: all profiles must give their speeds at the same depths (Bellhop's 2-D field has one grid)")
    if depth is None:
        out.append("sound-speed profiles need water_depth_m (Bellhop needs a bottom)")
    elif grid[-1] < depth - 1e-9:
        out.append(f"the profiles end at {grid[-1]:g} m and water_depth_m is {depth:g}: they must reach the bottom (nothing is extended)")
    return out


# --- derived quantities -------------------------------------------------------------------------------

def profile_sections(env: dict):
    """(range [m], depth [m], speed [n_range][n_depth] m/s) of an environment whose water changes with range
    (`profiles_by_range`), or None."""
    sections = env.get("profiles_by_range")
    if sections is None:
        return None
    return ([float(s["range_m"]) for s in sections], [float(v) for v in sections[0]["depth_m"]],
            [[float(v) for v in s["speed_m_s"]] for s in sections])


def profile_arrays(env: dict):
    """(depth [m], speed [m/s]) of the environment's sound-speed profile, or None for isovelocity water. A named
    family is sampled down to the water depth. In water that changes with range (`profiles_by_range`) this is the
    LOCAL profile: the one at the sonar, the first of the list."""
    if env.get("profile") is None and env.get("profiles_by_range") is not None:
        _, z, c = profile_sections(env)
        return z, c[0]
    prof = env.get("profile")
    if prof is None:
        return None
    if prof.get("family") is None:
        return [float(v) for v in prof["depth_m"]], [float(v) for v in prof["speed_m_s"]]
    depth = env.get("water_depth_m")
    if depth is None:
        raise EnvironmentError("environment.profile: a named family needs water_depth_m")
    if prof["family"] == "munk":
        from .sim.bellhop import munk_profile
        z, c = munk_profile(float(depth), int(prof.get("points", 51)), float(prof.get("axis_depth_m", 1300.0)), float(prof.get("axis_speed_m_s", NOMINAL_SOUND_SPEED)))
        return [float(v) for v in z], [float(v) for v in c]
    if prof["family"] == "linear":
        c0, g = float(prof["surface_speed_m_s"]), float(prof["gradient_per_s"])
        return [0.0, float(depth)], [c0, c0 + g * float(depth)]
    raise EnvironmentError(f"environment.profile: unknown family {prof['family']!r}")


def wind_speed(env: dict, default: float | None = None):
    """Wind speed (m/s) of an environment under either layer's field names: `sea_state` means its median wind."""
    if env.get("sea_state") is not None:
        return float(acoustics.sea_state_wind_speed(env["sea_state"]))
    for key in ("wind_speed_m_s", "wind_speed"):
        if env.get(key) is not None:
            return float(env[key])
    return default


def coverage_site(env: dict, where: str = "environment") -> dict:
    """What the live chain's coverage picture (`chain.coverage`; gpusonar.coverage.of_build) takes from a site:
    {water_depth_m, bottom, wind_speed_m_s}. Nothing is defaulted: a site that does not state its water depth, its bottom
    or its wind (`wind_speed_m_s` or `sea_state`) is refused, by name. `bottom` comes back as the chain takes it:
    {sediment: name} or {speed, density, attenuation}."""
    env = normalise(env, where, strict=False)
    wind = wind_speed(env)
    missing = [key for key, value in (("water_depth_m", env.get("water_depth_m")), ("bottom", env.get("bottom")), ("wind_speed_m_s (or sea_state)", wind)) if value is None]
    if missing:
        raise EnvironmentError(f"{where}: the coverage picture needs the site's water depth, bottom and wind, and this environment does not give {', '.join(missing)}. "
                               "They are facts of the site and have no default: without the water depth and the bottom the picture is the deep-water one, in which nothing returns "
                               "from the bottom, and it says that targets the chain detects cannot be seen")
    if not _num(env["water_depth_m"]) or float(env["water_depth_m"]) <= 0.0:
        raise EnvironmentError(f"{where}.water_depth_m: {env['water_depth_m']!r}; a positive number of metres")
    bottom = {"sediment": env["bottom"]} if isinstance(env["bottom"], str) else {k: float(v) for k, v in dict(env["bottom"]).items()}
    return {"water_depth_m": float(env["water_depth_m"]), "bottom": bottom, "wind_speed_m_s": float(wind)}


def resolve(env, depth: float = 0.0, strict: bool = False, where: str = "environment") -> dict:
    """The environment the simulator works with, from what a scenario or a caller gave (a name, a mapping under
    either spelling, or an already resolved mapping): the keys the simulator has always used (`sound_speed`, and
    `temperature`, `salinity`, `depth`, `ph`, `water_depth`, `frame_origin_lla` when given) plus the site
    (`site_water_depth`, `wind_speed`, `rms_wave_height`, `bottom`, `profile`, `bubble_model`, `absorption`, `shipping`, `name`), and
    `defaulted`: the names of the values nobody gave. Water that changes with range (`profiles_by_range`): `profile`
    is the local profile (range 0) and `profile_by_range` = {range, speed} holds all of them on its depth grid.

    Sound speed: the one given; else the profile (the local one) at the reference depth; else Mackenzie/Marczak from the temperature,
    salinity and reference depth (or `depth`, the caller's); else `acoustics.NOMINAL_SOUND_SPEED`."""
    if isinstance(env, dict) and env.get("_resolved"):
        return dict(env)
    e = load(env, where) if strict or isinstance(env, str) else _with_base(normalise(env, where, strict=False))
    if strict:
        found = problems([e], named=False)
        if found:
            raise EnvironmentError("; ".join(f"{w.replace('environments[0]', where).replace('environments.', where + ' ')}: {m}" for w, m in found))
    out = {k: v for k, v in e.items() if k not in FIELDS}             # a caller's own keys pass through (not strict)
    defaulted = []
    legacy = {"temperature_c": "temperature", "salinity_psu": "salinity", "reference_depth_m": "depth", "ph": "ph",
              "water_depth_m": "water_depth", "frame_origin_lla": "frame_origin_lla", "rms_wave_height_m": "rms_wave_height",
              "bottom": "bottom", "bubble_model": "bubble_model", "absorption": "absorption", "shipping": "shipping", "name": "name"}
    for canon, key in legacy.items():
        if e.get(canon) is not None:
            out[key] = copy.deepcopy(e[canon])
    wind = wind_speed(e)
    if wind is not None:
        out["wind_speed"] = wind
    prof = profile_arrays(e)
    if prof is not None:
        out["profile"] = {"depth": prof[0], "speed": prof[1]}
    sections = profile_sections(e)
    if sections is not None:
        out["profile_by_range"] = {"range": sections[0], "speed": sections[2]}
    if e.get("water_depth_m") is not None:
        # `water_depth` is what the simulated vehicle is told (a record carries it; processors read it); the site's own
        # depth, which propagation and reverberation use, is `site_water_depth`. They differ only when
        # `water_depth_known: false`.
        out["site_water_depth"] = out["water_depth"]
        if e.get("water_depth_known") is False:
            del out["water_depth"]
    overrides = e.get("overrides") or {}
    c = e.get("sound_speed_m_s")
    if c is None:
        if prof is not None:
            c = float(np.interp(float(e.get("reference_depth_m", DEFAULT_REFERENCE_DEPTH_M)), prof[0], prof[1]))
        elif e.get("temperature_c") is not None:
            c = acoustics.sound_speed(e["temperature_c"], e.get("salinity_psu", DEFAULT_SALINITY_PSU), e.get("reference_depth_m", depth))
        else:
            c = NOMINAL_SOUND_SPEED
            defaulted.append("sound_speed")
    elif strict:
        if e.get("temperature_c") is not None and prof is None:
            implied = acoustics.sound_speed(e["temperature_c"], e.get("salinity_psu", DEFAULT_SALINITY_PSU), e.get("reference_depth_m", depth))
            seen = (float(c), float(e["temperature_c"]), float(e.get("salinity_psu", DEFAULT_SALINITY_PSU)))
            if abs(float(c) - implied) > SOUND_SPEED_TOLERANCE_M_S and "sound_speed_m_s" not in overrides and seen not in _TOLD:
                _TOLD.add(seen)                        # once per process for one water
                warnings.warn(f"{where}: sound_speed_m_s {float(c):g} beside temperature_c {e['temperature_c']:g} and salinity_psu "
                              f"{e.get('salinity_psu', DEFAULT_SALINITY_PSU):g}, which give {implied:.1f} m/s: geometry uses the first, absorption the "
                              "second. Leave one out, or name sound_speed_m_s under the environment's `overrides` with the reason", stacklevel=3)
        if prof is not None and not (min(prof[1]) - 1e-6 <= float(c) <= max(prof[1]) + 1e-6) and "sound_speed_m_s" not in overrides:
            raise EnvironmentError(f"{where}: sound_speed_m_s {float(c):g} is outside its own profile ({min(prof[1]):g} to {max(prof[1]):g} m/s)")
    out["sound_speed"] = c
    for key, name in (("temperature", "temperature"), ("salinity", "salinity"), ("wind_speed", "wind_speed")):
        if key not in out:
            defaulted.append(name)
    out["defaulted"] = defaulted
    out["_resolved"] = True
    return out


def water(env: dict) -> tuple:
    """(temperature C, salinity PSU, depth m, pH) the absorption formulas take, the stated defaults filled in
    (`DEFAULT_TEMPERATURE_C` and the rest: the one place they are written)."""
    return (env.get("temperature", DEFAULT_TEMPERATURE_C), env.get("salinity", DEFAULT_SALINITY_PSU),
            env.get("depth", DEFAULT_REFERENCE_DEPTH_M), env.get("ph", DEFAULT_PH))


def absorption_db_per_m(env: dict, frequency, model: str | None = None):
    """Absorption (dB/m) at `frequency` in this environment's water, by `model` or the environment's own."""
    return acoustics.absorption(frequency, *water(env), model or env.get("absorption") or DEFAULT_ABSORPTION)


# --- a scenario's use of its environment --------------------------------------------------------------

class _Site:
    """Gives site values to a scenario's blocks and keeps count of what was typed against them."""

    def __init__(self, overrides):
        if isinstance(overrides, list):
            overrides = {str(k): "" for k in overrides}
        self.overrides = dict(overrides or {})
        self.used = set()
        self.problems = []
        self.filled = []

    def take(self, block: dict, key: str, value, path: str, what: str) -> None:
        """`block[key]` from the environment's `value` when the block does not give it; when it does and the two
        differ, a problem unless `path` is a named override."""
        if value is None:
            return
        if key not in block:
            block[key] = copy.deepcopy(value)
            self.filled.append(path)
        elif block[key] is None:
            pass                # an explicit null says "not in this run" (propagation.bottom_z: null: the surface alone)
        elif not _same(block[key], value):
            self.differs(path, f"{block[key]!r} here, {value!r} from {what}")

    def differs(self, path: str, text: str) -> None:
        if path in self.overrides:
            self.used.add(path)
        else:
            self.problems.append(f"{path}: {text}")


def _reverb_boundary(item: dict) -> str | None:
    """Which boundary a reverberation item lies on: `boundary: surface | bottom`, else by its model."""
    if item.get("boundary") is not None:
        if item["boundary"] not in ("surface", "bottom"):
            raise EnvironmentError(f"reverberation: boundary {item['boundary']!r}: surface or bottom")
        return str(item["boundary"])
    model = item.get("model", "lambert")
    if model in SURFACE_REVERB_MODELS:
        return "surface"
    if model in BOTTOM_REVERB_MODELS:
        return "bottom"
    return None


def apply_to_scenario(cfg: dict, env: dict, sonar=None) -> dict:
    """A copy of scenario `cfg` in which every block that needs a site value has it from the resolved environment
    `env` (`resolve`), in the explicit form the simulator has always read. Values typed in a block are kept; one that
    contradicts the environment raises `EnvironmentError` unless `environment_overrides` names it. The result's
    `_environment_filled` lists what came from the environment."""
    cfg = dict(cfg)
    site = _Site(cfg.get("environment_overrides"))
    wind = env.get("wind_speed")
    depth = env.get("site_water_depth", env.get("water_depth"))
    bottom = env.get("bottom")
    tier = cfg.get("fidelity", "free_field")
    surface_z = 0.0
    if env.get("profile_by_range") is not None and tier != "eigenray":
        # Only the eigenray tier (Bellhop) traces through a profile. The other tiers have one sound speed, the
        # environment's (the speed at the sonar), as they do beside a single `profile`; here that leaves out what the
        # scenario was written to show, so it is said.
        warnings.warn(f"scenario {cfg.get('name', '')!r}: the environment's water changes with range (profiles_by_range) and fidelity is {tier!r}, which has one "
                      f"sound speed: {env['sound_speed']:g} m/s (the speed at the sonar) is used everywhere and the change with range is NOT simulated. "
                      "Use fidelity: eigenray (Bellhop)", stacklevel=3)

    # Absorption model: the environment's unless the scenario names one. A model is a choice of formula, not a
    # property of the site (the water it is evaluated in is: temperature, salinity, depth, pH), so a scenario's own
    # `absorption` simply wins: comparisons run one site under several models.
    if env.get("absorption") is not None and cfg.get("absorption") is None:
        cfg["absorption"] = env["absorption"]

    # propagation
    if tier == "image_source":
        prop = dict(cfg.get("propagation") or {})
        surface_z = prop.get("surface_z", 0.0)
        site.take(prop, "bottom_z", depth, "propagation.bottom_z", "environment.water_depth_m")
        if prop.get("bottom_z") is not None and "bottom_coefficient" not in prop and bottom is not None:
            site.take(prop, "bottom", {"sediment": bottom} if isinstance(bottom, str) else bottom, "propagation.bottom", "environment.bottom")
        cfg["propagation"] = prop
    elif tier == "sediment" and (cfg.get("propagation") or {}).get("penetration") in ("fluid", "rough"):
        # Opt-in fluid penetration (G-3; `rough`, G-13, adds roughness scattering): the sediment is the environment's
        # bottom unless the scenario gives one, and the attenuation is taken at the sonar's band centre, as for the
        # eigenray tier. With the default
        # (`penetration` absent or `constant`) nothing is filled in: the tier takes neither.
        prop = dict(cfg["propagation"])
        if bottom is not None:
            site.take(prop, "bottom", {"sediment": bottom} if isinstance(bottom, str) else bottom, "propagation.bottom", "environment.bottom")
        if sonar is not None:
            site.take(prop, "frequency", float(sonar.center_frequency), "propagation.frequency", "the sonar's band centre")
        cfg["propagation"] = prop
    elif tier in ("eigenray", "eigenray3d"):
        # eigenray3d (Bellhop3D, opt-in) takes the same ocean; its bed is `propagation.bathymetry` (3-D), not a profile.
        prop = dict(cfg.get("propagation") or {})
        ocean = dict(prop.get("environment") or {})
        where = "propagation.environment"
        if ocean.get("bathymetry") is not None and "depth" in ocean:
            # A bottom whose depth changes with range: the ocean's `depth` is the deepest point of the profile and the
            # environment's water depth is the depth at the sonar. The first must not be shallower than the second.
            if depth is not None and float(ocean["depth"]) < float(depth) - 1e-9:
                site.differs(f"{where}.depth", f"{ocean['depth']!r} here is shallower than {depth!r} from environment.water_depth_m")
        else:
            site.take(ocean, "depth", depth, f"{where}.depth", "environment.water_depth_m")
        if env.get("profile") is not None:
            site.take(ocean, "ssp_depth", env["profile"]["depth"], f"{where}.ssp_depth", "environment.profile")
            site.take(ocean, "ssp_speed", env["profile"]["speed"], f"{where}.ssp_speed", "environment.profile")
            if env.get("profile_by_range") is not None:
                site.take(ocean, "ssp_range", env["profile_by_range"]["range"], f"{where}.ssp_range", "environment.profiles_by_range")
                site.take(ocean, "ssp_speed_by_range", env["profile_by_range"]["speed"], f"{where}.ssp_speed_by_range", "environment.profiles_by_range")
        else:
            if ocean.get("ssp_speed") is not None:
                # a profile typed here beside the environment's one speed: that speed is the one at the sonar, and must be a speed of the profile
                if not (min(ocean["ssp_speed"]) - 1e-6 <= float(env["sound_speed"]) <= max(ocean["ssp_speed"]) + 1e-6):
                    site.differs(f"{where}.ssp_speed", f"{min(ocean['ssp_speed'])!r} to {max(ocean['ssp_speed'])!r} here, {env['sound_speed']!r} from environment.sound_speed_m_s")
            else:
                site.take(ocean, "sound_speed", env["sound_speed"], f"{where}.sound_speed", "environment.sound_speed_m_s")
        typed_bottom = [k for k in ("bottom_speed", "bottom_density", "bottom_attenuation", "sediment") if k in ocean]
        if isinstance(bottom, str):
            if not typed_bottom:
                ocean["sediment"] = bottom
            elif ocean.get("sediment") != bottom:
                site.differs(f"{where}.bottom", f"{ {k: ocean[k] for k in typed_bottom} } here, sediment {bottom!r} from environment.bottom")
        elif isinstance(bottom, dict):
            if "sediment" in ocean:
                site.differs(f"{where}.bottom", f"sediment {ocean['sediment']!r} here, {bottom!r} from environment.bottom")
            else:
                for key in ("speed", "density", "attenuation"):
                    site.take(ocean, f"bottom_{key}", bottom.get(key), f"{where}.bottom_{key}", "environment.bottom")
        if ocean:
            prop["environment"] = ocean
        if sonar is not None:
            site.take(prop, "frequency", float(sonar.center_frequency), "propagation.frequency", "the sonar's band centre")
            if "max_range" not in prop:
                start, n = sonar.window(env["sound_speed"])
                prop["max_range"] = float(np.ceil(1.1 * 0.5 * env["sound_speed"] * (start + n / sonar.sample_rate)))
                site.filled.append("propagation.max_range")
        site.take(prop, "wind_speed", wind, "propagation.wind_speed", "environment.wind_speed_m_s")
        if tier == "eigenray3d":
            site.take(prop, "bathymetry", cfg.get("bathymetry"), "propagation.bathymetry", "bathymetry")
        cfg["propagation"] = prop

    # One bed (`bathymetry`, a sim/bathymetry.py description at the top of a scenario; D-73's "one description"):
    # the eigenray3d provider (above), every reverberation item on the bottom and every point-scatterer seafloor take
    # it unless they give their own (which must then be the same). Absent, nothing here runs.
    shared_bed = cfg.get("bathymetry")
    if shared_bed is not None:
        if cfg.get("scene"):
            items = []
            for i, item in enumerate(cfg["scene"]):
                item = dict(item)
                if item.get("type") == "seafloor":
                    site.take(item, "bathymetry", shared_bed, f"scene[{i}].bathymetry", "bathymetry")
                items.append(item)
            cfg["scene"] = items
        if cfg.get("reverberation"):
            items = []
            for i, item in enumerate(cfg["reverberation"]):
                item = dict(item)
                if (item.get("model", "lambert") != "volume" and _reverb_boundary(item) != "surface"
                        and "boundary_z" not in item):
                    site.take(item, "bathymetry", shared_bed, f"reverberation[{i}].bathymetry", "bathymetry")
                items.append(item)
            cfg["reverberation"] = items

    # time spreading
    if cfg.get("time_spread"):
        spread = dict(cfg["time_spread"])
        surf = spread.get("surface")
        if surf is True:
            surf = {}
            if wind is None and env.get("rms_wave_height") is None:
                raise EnvironmentError("time_spread.surface: true needs environment.wind_speed_m_s (or sea_state) or rms_wave_height_m")
        if isinstance(surf, dict) and (surf or spread.get("surface") is True):
            surf = dict(surf)
            if not any(k in surf for k in ("rms_height", "mean_square_slope", "rms_vertical_speed")) or "wind_speed" in surf:
                site.take(surf, "wind_speed", wind, "time_spread.surface.wind_speed", "environment.wind_speed_m_s")
            site.take(surf, "rms_height", env.get("rms_wave_height"), "time_spread.surface.rms_height", "environment.rms_wave_height_m")
            spread["surface"] = surf
        cfg["time_spread"] = spread

    # reverberation
    if cfg.get("reverberation"):
        items = []
        for i, item in enumerate(cfg["reverberation"]):
            item = dict(item)
            where = f"reverberation[{i}]"
            side = _reverb_boundary(item)
            model = item.get("model", "lambert")
            if model != "volume" and item.get("bathymetry") is None:
                if "boundary_z" not in item:
                    z = surface_z if side == "surface" else depth
                    if z is not None:
                        item["boundary_z"] = float(z)
                        site.filled.append(f"{where}.boundary_z")
                elif depth is not None:
                    z = float(item["boundary_z"])
                    on_surface, on_bottom = _same(z, float(surface_z)), _same(z, float(depth))
                    if model in SURFACE_REVERB_MODELS and not on_surface:
                        site.differs(f"{where}.boundary_z", f"{z!r} here, a surface law belongs at {surface_z!r}")
                    elif model not in SURFACE_REVERB_MODELS and not (on_bottom or (on_surface and model == "lambert")):
                        site.differs(f"{where}.boundary_z", f"{z!r} here, {depth!r} from environment.water_depth_m")
            if model in SURFACE_REVERB_MODELS:
                site.take(item, "wind_speed", wind, f"{where}.wind_speed", "environment.wind_speed_m_s")
            if model == "fluid_sediment" and isinstance(bottom, str) and "sediment" not in item and not all(k in item for k in ("rho", "nu", "delta", "w2", "gamma2", "sigma2")):
                item["sediment"] = bottom
                site.filled.append(f"{where}.sediment")
            item.pop("boundary", None)
            items.append(item)
        cfg["reverberation"] = items

    # ambient noise
    rx = cfg.get("receiver") or {}
    noise = rx.get("ambient_noise")
    if noise is True or noise == "environment":
        noise = {"model": "wenz"}
    if isinstance(noise, dict) and noise and noise.get("model", "wenz") == "wenz" and "level_db" not in noise:
        noise = dict(noise)
        site.take(noise, "wind_speed", wind, "receiver.ambient_noise.wind_speed", "environment.wind_speed_m_s")
        site.take(noise, "shipping", env.get("shipping"), "receiver.ambient_noise.shipping", "environment.shipping")
        noise.setdefault("wind_speed", DEFAULT_WIND_SPEED_M_S)          # nobody said: the one stated default
        noise.setdefault("shipping", DEFAULT_SHIPPING)
    if noise is not rx.get("ambient_noise"):
        cfg["receiver"] = {**rx, "ambient_noise": noise}

    if site.problems:
        raise EnvironmentError(f"scenario {cfg.get('name', '')!r}: typed values contradict the environment (change them, leave them out, or name each under "
                               "`environment_overrides` with its reason): " + "; ".join(site.problems))
    stale = sorted(set(site.overrides) - site.used)
    if stale:
        warnings.warn(f"scenario {cfg.get('name', '')!r}: environment_overrides names {stale}, which no longer differ from the environment "
                      "(or are not keys that take a site value)", stacklevel=2)
    cfg["_environment_filled"] = site.filled
    return cfg


def expand_scenario(cfg: dict) -> dict:
    """A scenario configuration in the explicit form: the environment resolved (`resolve`, strict) and every block
    holding its site values (`apply_to_scenario`). This is what `config.load_scenario_config` returns, so that code
    which reads a scenario (`cfg["environment"]["sound_speed"]`, `cfg["propagation"]["bottom_z"]`,
    `cfg["reverberation"][0]["wind_speed"]`) finds the values wherever the file took them from. Running the result
    gives the same records as running the file. To change the site of a loaded scenario, pass the change as an
    override to the loader (`load_scenario_config(name, {"environment": {"water_depth_m": 45.0}})`), not by editing
    the expanded mapping: a value edited there contradicts the blocks already filled from it, and is reported."""
    from .sonar import SonarModel
    cfg = acoustics.resolve_sea_state(cfg)
    env = resolve(cfg.get("environment"), strict=True)
    sonar = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides")) if cfg.get("sonar") is not None else None
    out = apply_to_scenario(cfg, env, sonar)
    out["environment"] = env
    return out
