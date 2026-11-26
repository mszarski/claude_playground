"""The coverage picture: what the sonar can see from where it is, in the water it is in.

`chain.coverage` (opt-in) puts one member, `coverage`, into the chain's report line; both chain engines get it from the
same C++ class (libs/live coverage.hpp, which says how each number is made), so this module only
    reads the configuration                `options`, `cpp_options`
    makes the detection-probability table  `pd_table`: `live.detection_probability` for the chain as configured, on a grid of
                                           one row's link-budget SNR. The chain is given the table at its first ping.
    takes levels from a build              `budget_of`: `buildspec.budget`'s numbers as the `budget:` block
    answers "could I have seen it"         `could_have_seen`: from a report's `coverage` member, the detection probability
                                           of each target class at a range (any bearing) at the depth the sonar was at
    and serves tools/coverage_map.py       `field`, `summary`

Configuration (`chain.coverage`; a key the C++ options do not have is an error):

    water_depth_m:     REQUIRED in a chain, no default: the site's water depth, m. Used on a ping for which neither
                       `elevation.water_depth` nor navigation gives one (what a ping carries comes first)
    bottom:            REQUIRED in a chain, no default: {sediment: name in configs/sediments.yaml} or {speed, density,
                       attenuation[, shear_speed, shear_attenuation]}. A row with shear (`rock`) is an elastic solid, as in the
                       simulator (its shear unconfirmed, said once); {sediment: rock, shear_speed: 0} is the row as a fluid.
                       Without the two the picture is the deep-water one, in which nothing comes back from the bottom: in
                       100 m over sand it said a ship at 12 km could not be seen while the chain detected it.
                       A chain given `coverage` without them does not start (`chain_options`); a build takes them from
                       its site (`of_build`: the spec's `operation.coverage.environment`)
    classes:           [{name, ts_db, depth_m: [shallowest, deepest]}, ...]   target classes (a build's: buildspec.target_classes_of)
    budget:            {source_level_db, noise_spectrum_db, array_gain_db, absorption_db_per_km}   no defaults (budget_of)
    pd:                0.5      the detection probability the ranges are stated for
    fluctuation:       none     or swerling1: the echo's ping-to-ping statistics in the table
    range_m:           null     [first, last]; null: the receive window of the ping
    n_ranges, n_depths, n_report                48, 5, 12: the grid, and how many of its ranges the report lists
    recommend:         true     try other own depths
    own_depth_limits_m: [5, 500], bottom_clearance_m: 10, n_own: 10
    refresh:           {depth_m: 5, water_depth_m: 10, speed_m_s: 0.5}   recomputed when one of these has changed by so much
    surface:           {coefficient: -1, rms_height_m: 0, loss_db: 0, bubble_model: null, wind_speed_m_s: null, frequency_hz: null}
                                (`field` and `summary`, the tools' entry points, still take a water without a bottom: rays that
                                reach it are dropped. That is a statement about deep water, made by the caller)
    rays:              {fan: 720, max_bounces: 3, max_arrivals: 6}
    null_db:           -10      a surface-image null: the two-way factor below this where the direct path exists
    reverberation:     null     the boundaries' reverberation in the budget (sonar-w2w3; coverage.hpp says how, and what it
                       leaves out). {items: [...], frequency_hz: F[, beam_integral_deg: B]}: `items` in the scenario's
                       `reverberation:` form ({mu_db: -27} a Lambert bottom, {model: chapman_harris, wind_speed: 5}, any
                       bottom or surface law of sim/simulator.py; `boundary: surface | bottom` where the model does not say),
                       turned into tables by the simulator's own laws at `frequency_hz` (the band centre); `beam_integral_deg`:
                       the beam's integral over azimuth (default: the array's at broadside, from the record); `max_bounces`: boundary
                       reflections a reverberation path may have each way (default: from the range and the water depth, coverage.hpp). Each class's
                       entry then also gives `max_range_noise_only_m` and `limited_by`. null (not given): noise only, and
                       the member says "reverberation": "not included: ..." (ranges optimistic where reverberation limits).
                       false: noise only and nothing said (the picture as before the term existed)

The profile is the one the sound-speed ladder holds (`chain.elevation.profile`; with `ensemble`, each class also gets the
span of its greatest range over the ensemble's profiles).
"""

from __future__ import annotations

import copy
import json

import numpy as np

DEFAULTS = {
    "classes": [],
    "budget": {"source_level_db": None, "noise_spectrum_db": None, "array_gain_db": None, "absorption_db_per_km": None},
    "pd": 0.5,
    "fluctuation": "none",
    "range_m": None,
    "n_ranges": 48, "n_depths": 5, "n_report": 12,
    "recommend": True,
    "own_depth_limits_m": [5.0, 500.0], "bottom_clearance_m": 10.0, "n_own": 10,
    "refresh": {"depth_m": 5.0, "water_depth_m": 10.0, "speed_m_s": 0.5},
    "surface": {"coefficient": -1.0, "rms_height_m": 0.0, "loss_db": 0.0, "bubble_model": None, "wind_speed_m_s": None, "frequency_hz": None},
    "bottom": None,
    "water_depth_m": None,
    "rays": {"fan": 720, "max_bounces": 3, "max_arrivals": 6},
    "null_db": -10.0,
    "surface_z": 0.0,
    "reverberation": None,
}
REVERBERATION_KEYS = ("items", "frequency_hz", "beam_integral_deg", "max_bounces")
PD_SNR_DB = np.arange(-20.0, 60.01, 0.5)          # the table's grid: one row's link-budget SNR, dB


def options(cfg: dict | None) -> dict:
    """`chain.coverage` merged over DEFAULTS (unknown keys are an error)."""
    out = copy.deepcopy(DEFAULTS)
    for key, value in (cfg or {}).items():
        if key not in out:
            raise ValueError(f"coverage: unknown key {key!r}; keys are {sorted(out)}")
        if isinstance(out[key], dict) and isinstance(value, dict):
            unknown = set(value) - set(out[key])
            if unknown:
                raise ValueError(f"coverage.{key}: unknown keys {sorted(unknown)}; keys are {sorted(out[key])}")
            out[key].update(value)
        else:
            out[key] = value
    return out


def bubble_table(surface: dict, sound_speed: float = 1500.0):
    """(grazing rad, loss dB per bounce) of a named bubble law (sim/surface.py BUBBLE_LOSS_MODELS, read only), or None.
    The law needs the wind speed and the frequency: `wind_speed_m_s` and `frequency_hz` beside `bubble_model`."""
    if not surface.get("bubble_model"):
        return None
    if surface.get("wind_speed_m_s") is None or surface.get("frequency_hz") is None:
        raise ValueError("surface.bubble_model needs surface.wind_speed_m_s and surface.frequency_hz (the band centre)")
    import warnings

    from .sim import surface as surface_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grazing, loss = surface_model.bubble_loss_table(str(surface["bubble_model"]), float(surface["wind_speed_m_s"]), float(surface["frequency_hz"]), float(sound_speed))
    loss = np.where(np.isfinite(loss), loss, 200.0)
    return [float(v) for v in grazing], [float(v) for v in loss]


def halfspace(bottom: dict | None, sound_speed: float = 1500.0) -> dict | None:
    """{speed, density, attenuation} of `bottom` ({sediment: name} resolves through configs/sediments.yaml as gpusonar.lloyd does),
    with {shear_speed, shear_attenuation} for an elastic one (a row with shear, `rock`; or typed)."""
    if bottom is None:
        return None
    from .lloyd import Layer
    return Layer(depth=1.0, bottom=dict(bottom), sound_speed=float(sound_speed)).halfspace()


def cpp_options(cfg: dict | None, sound_speed: float = 1500.0) -> dict:
    """`chain.coverage` as the flat options of the C++ class (libs/live CoverageOptions; `coverage_<name>=value` arguments
    of build/bin/sonar_live). Without the detection-probability table: `pd_table` makes it once the sonar is known."""
    c = options(cfg)
    b, s, r, q = c["budget"], c["surface"], c["rays"], c["refresh"]
    for item in c["classes"]:
        if set(item) - {"name", "ts_db", "depth_m"} or not {"name", "ts_db", "depth_m"} <= set(item):
            raise ValueError(f"coverage.classes: each needs name, ts_db and depth_m [shallowest, deepest]; got {item}")
        if any(ch in str(item["name"]) for ch in ':;"\\'):
            raise ValueError(f"coverage.classes: the name {item['name']!r} may not hold : ; \" or \\")
    o = {"pd": float(c["pd"]), "n_ranges": int(c["n_ranges"]), "n_depths": int(c["n_depths"]), "n_report": int(c["n_report"]),
         "recommend": int(bool(c["recommend"])), "own_depth_min": float(c["own_depth_limits_m"][0]), "own_depth_max": float(c["own_depth_limits_m"][1]),
         "bottom_clearance": float(c["bottom_clearance_m"]), "n_own": int(c["n_own"]), "refresh_depth": float(q["depth_m"]),
         "refresh_water_depth": float(q["water_depth_m"]), "refresh_speed": float(q["speed_m_s"]), "surface_coefficient": float(s["coefficient"]),
         "surface_rms_height": float(s["rms_height_m"]), "surface_loss_db": float(s["loss_db"]), "ray_fan": int(r["fan"]),
         "max_bounces": int(r["max_bounces"]), "max_arrivals": int(r["max_arrivals"]), "null_db": float(c["null_db"]), "surface_z": float(c["surface_z"])}
    if c["classes"]:
        o["classes"] = ";".join(f"{i['name']}:{float(i['ts_db'])!r}:{float(i['depth_m'][0])!r}:{float(i['depth_m'][1])!r}" for i in c["classes"])
    for key, value in b.items():
        if value is not None:
            o[key] = float(value)
    if c["water_depth_m"] is not None:
        if not (isinstance(c["water_depth_m"], (int, float)) and not isinstance(c["water_depth_m"], bool) and float(c["water_depth_m"]) > 0.0):
            raise ValueError(f"coverage.water_depth_m: {c['water_depth_m']!r}; it is the site's water depth in metres, a positive number")
        o["water_depth"] = float(c["water_depth_m"])
    if c["range_m"] is not None:
        o.update(range_first=float(c["range_m"][0]), range_last=float(c["range_m"][1]))
    table = bubble_table(s, sound_speed)
    if table is not None:
        o.update(surface_loss_grazing=table[0], surface_loss_table_db=table[1])
    o.update(reverberation_options(c["reverberation"], sound_speed))
    half = halfspace(c["bottom"], sound_speed)
    if half is not None:
        o.update(bottom=1, bottom_speed=float(half["speed"]), bottom_density=float(half["density"]), bottom_attenuation=float(half.get("attenuation", 0.0)))
        if float(half.get("shear_speed", 0.0)) > 0.0:          # an elastic bottom (rock): a fluid's options are as before
            o.update(bottom_shear_speed=float(half["shear_speed"]), bottom_shear_attenuation=float(half.get("shear_attenuation", 0.0)))
    return o


def reverberation_options(block, sound_speed: float = 1500.0) -> dict:
    """The C++ options of `coverage.reverberation` (DEFAULTS' comment): reverb 2 (not given: the member says it is not
    included), 0 (false: as before the term existed) or 1 with the scattering-strength tables of the simulator's laws
    (gpusonar/reverb_budget.py `strength_tables`) on its grazing grid."""
    if block is None:
        return {"reverb": 2}
    if block is False:
        return {"reverb": 0}
    if not isinstance(block, dict) or set(block) - set(REVERBERATION_KEYS) or not {"items", "frequency_hz"} <= set(block):
        raise ValueError(f"coverage.reverberation: give {{items: [...], frequency_hz: F[, beam_integral_deg: B]}} (or false); got {block!r}")
    if not block["items"]:
        return {"reverb": 2}
    from . import reverb_budget
    tables = reverb_budget.strength_tables(block["items"], float(block["frequency_hz"]), float(sound_speed))
    o = {"reverb": 1, "reverb_grazing": [float(v) for v in reverb_budget.GRAZING]}
    for side in ("bottom", "surface"):
        if tables[side] is not None:
            o[f"reverb_{side}_strength"] = [float(v) for v in tables[side].ravel()]
    if block.get("beam_integral_deg") is not None:
        o["reverb_beam_integral"] = float(np.radians(float(block["beam_integral_deg"])))
    if block.get("max_bounces") is not None:
        o["reverb_max_bounces"] = int(block["max_bounces"])
    return o


def missing_site(cfg: dict | None) -> list:
    """The site facts `chain.coverage` must state and does not: of `water_depth_m`, `bottom`."""
    c = cfg or {}
    return [key for key in ("water_depth_m", "bottom") if c.get(key) is None]


def chain_options(cfg: dict | None, sound_speed: float = 1500.0) -> dict:
    """`cpp_options` for a chain (gpusonar.live, both engines): the same, after refusing a `chain.coverage` that does not
    state the site's water depth and bottom. Neither has a default: they are facts of the site."""
    missing = missing_site(cfg)
    if missing:
        raise ValueError("chain.coverage needs the site's water depth and bottom, and " + " and ".join(f"coverage.{k}" for k in missing) + (" is" if len(missing) == 1 else " are")
                         + " not given. Give coverage.water_depth_m (metres) and coverage.bottom ({sediment: name in configs/sediments.yaml} or {speed, density, attenuation}); "
                         "neither has a default. Without them the picture would be the deep-water one, in which nothing returns from the bottom, and it would say that targets the "
                         "chain detects cannot be seen. A generated deployment file takes them from the build spec's site (operation.coverage.environment)")
    return cpp_options(cfg, sound_speed)


def of_build(spec: dict, site: dict) -> dict:
    """`chain.coverage` of a build's deployment file (buildspec.chain_config). `site`: the spec's `operation.coverage`,
    {environment: the name of one of the spec's `environments`, or an environment mapping (gpusonar.environment.FIELDS)}.
    Water depth, bottom and wind are the site's, none defaulted (environment.coverage_site raises naming what is missing);
    the classes are the spec's targets, the levels `buildspec.budget`'s at the site's wind, the surface's rms height the
    site's (its `rms_wave_height_m`, else the wind law's, as the depth planner has it)."""
    from . import buildspec, depthplan, environment
    where = "operation.coverage"
    if not isinstance(site, dict) or set(site) != {"environment"}:
        raise ValueError(f"{where}: give {{environment: <one of the build's environments, or an environment mapping>}}; got {site!r}")
    env = site["environment"]
    if isinstance(env, str):
        known = buildspec.environments_of(spec)
        env = next((e for e in known if e["name"] == site["environment"]), None)
        if env is None:
            raise ValueError(f"{where}.environment: {site['environment']!r} is not one of the build's environments ({', '.join(e['name'] for e in known) or 'it has none'})")
    else:
        env = environment.normalise(env, f"{where}.environment")
    facts = environment.coverage_site(env, f"{where}.environment" + (f" ({env['name']})" if env.get("name") else ""))
    d = buildspec.describe(buildspec.sonar_config(spec))                    # without the deployment file: this is part of it
    classes = classes_of(buildspec.target_classes_of(spec, range_m=float(d["window_m"][1])))
    if not classes:
        raise ValueError(f"{where}: the build has no target classes (`targets:`); the coverage picture is stated per class")
    surf = depthplan.surface_of(env)
    block = {"classes": classes, "budget": {k: round(v, 2) for k, v in budget_of(d, facts["wind_speed_m_s"]).items()},       # a file is read by people: 0.01 dB
             "water_depth_m": facts["water_depth_m"], "bottom": facts["bottom"], "surface": {"rms_height_m": round(float(surf.height()), 3)}}
    if surf.bubble_model:
        block["surface"].update(bubble_model=surf.bubble_model, wind_speed_m_s=float(surf.wind_speed), frequency_hz=float(d["centre_hz"]))
    limit = (spec.get("platform") or {}).get("max_depth_m")
    lim = env.get("own_depth_limits_m")
    if lim is not None or limit is not None:
        lo, hi = (float(v) for v in (lim or [DEFAULTS["own_depth_limits_m"][0], limit]))
        block["own_depth_limits_m"] = [lo, float(min(hi, float(limit))) if limit is not None else hi]
    if env.get("bottom_clearance_m") is not None:
        block["bottom_clearance_m"] = float(env["bottom_clearance_m"])
    return block


def pd_table(record, chain_cfg: dict | None = None, fluctuation: str | None = None, rows: int | None = None):
    """(snr_db, pd): the probability that the chain as configured reports a point echo, against one row's link-budget SNR
    (`live.detection_probability`, the detector's own plan and thresholds). `record`: a ping or the sonar's template."""
    from . import live
    cfg = dict(chain_cfg or {})
    if fluctuation is None:
        fluctuation = options(cfg.get("coverage"))["fluctuation"] if cfg.get("coverage") is not None else "none"
    pd = live.detection_probability(PD_SNR_DB, record, cfg, rows=rows, fluctuation=fluctuation)
    return [float(v) for v in PD_SNR_DB], [float(min(max(v, 0.0), 1.0)) for v in np.atleast_1d(pd)]


def budget_of(d: dict, wind_speed: float) -> dict:
    """The `budget:` block of a build (`d`: buildspec.derive or describe), from buildspec.budget: its source level, the
    ambient noise spectrum at `wind_speed` m/s (the site's: no default), one row's gain against that noise, and its absorption."""
    from . import buildspec
    eq, gains = buildspec.budget(d, wind_speed)
    return {"source_level_db": float(eq.source_level_db), "noise_spectrum_db": float(gains["noise_spectrum_db"]),
            "array_gain_db": float(gains["gain_isotropic_noise_db"]), "absorption_db_per_km": float(eq.absorption_db_per_m() * 1000.0)}


def classes_of(resolved: list) -> list:
    """`classes:` from a build's resolved target classes (buildspec.target_classes_of): the LOW end of each target strength."""
    return [{"name": str(c["name"]), "ts_db": float(c["ts_db"][0]), "depth_m": [float(c["depth_m"][0]), float(c["depth_m"][1])]} for c in resolved]


def water(depth, speed, sigma=None, gradient_sigma: float = 0.0, rung: str = "supplied", ensemble=None) -> dict:
    """A profile as the C++ classes take it."""
    out = {"depth": [float(v) for v in depth], "speed": [float(v) for v in speed], "gradient_sigma": float(gradient_sigma), "rung": str(rung)}
    if sigma is not None:
        out["sigma"] = [float(v) for v in sigma]
    if ensemble is not None:
        out["ensemble"] = [[float(v) for v in member] for member in ensemble]
    return out


def field(profile: dict, water_depth, own_depth: float, target_depths, ranges, cfg: dict | None, sonar: dict, projector_below_m: float = 0.0) -> dict:
    """The two-way factor of the coverage picture on target depths x horizontal ranges (libs/live coverage_field):
    {factor_db, coherent_db, incoherent_db, direct, slant, snr_0db_target_db}, each [depth, range]. `sonar`:
    {f_centre, bandwidth, pulse}. The budget may be left out (the SNR is then NaN)."""
    from . import _gpusonar as core
    return core.coverage_field(profile, float(water_depth or 0.0), float(own_depth), float(own_depth) + float(projector_below_m), [float(r) for r in ranges],
                               [float(z) for z in target_depths], cpp_options(cfg), sonar)


def summary(profile: dict, water_depth, own_depth: float, cfg: dict, sonar: dict, table, range_m, projector_below_m: float = 0.0) -> dict:
    """The report member itself, as a dict, for one situation: what the chain would print (and `seconds`, the time it took)."""
    from . import _gpusonar as core
    cov = core.Coverage(cpp_options(cfg))
    cov.set_pd_table(*table)
    text = cov.update_numbers(sonar, float(range_m[0]), float(range_m[1]), float(own_depth), float(own_depth) + float(projector_below_m),
                              float(water_depth or 0.0), profile)
    out = json.loads("{" + text + "}")["coverage"]
    out["seconds"] = cov.seconds()
    return out


def could_have_seen(coverage: dict, range_m: float) -> dict:
    """{class name: detection probability} at a range (m, horizontal, any bearing) from a report's `coverage` member: the
    probability that a target of the class there would have been reported from the depth the sonar was at (worst over
    the class's depths). Outside the listed ranges: None (the sonar's window does not reach there)."""
    r = np.asarray(coverage.get("ranges_m", []), dtype=float)
    out = {}
    for c in coverage.get("classes", []):
        if "pd" not in c or r.size == 0 or not (r[0] <= range_m <= r[-1]):
            out[c["name"]] = None
        else:
            out[c["name"]] = float(np.interp(range_m, r, np.asarray(c["pd"], dtype=float)))
    return out
