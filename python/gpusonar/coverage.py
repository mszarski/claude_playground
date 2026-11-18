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
    bottom:            null     {sediment: name in configs/sediments.yaml} or {speed, density, attenuation}; null: the bottom
                                returns nothing (rays that reach it are dropped)
    rays:              {fan: 720, max_bounces: 3, max_arrivals: 6}
    null_db:           -10      a surface-image null: the two-way factor below this where the direct path exists

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
    "rays": {"fan": 720, "max_bounces": 3, "max_arrivals": 6},
    "null_db": -10.0,
    "surface_z": 0.0,
}
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
    """{speed, density, attenuation} of `bottom` ({sediment: name} resolves through configs/sediments.yaml as gpusonar.lloyd does)."""
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
    if c["range_m"] is not None:
        o.update(range_first=float(c["range_m"][0]), range_last=float(c["range_m"][1]))
    table = bubble_table(s, sound_speed)
    if table is not None:
        o.update(surface_loss_grazing=table[0], surface_loss_table_db=table[1])
    half = halfspace(c["bottom"], sound_speed)
    if half is not None:
        o.update(bottom=1, bottom_speed=float(half["speed"]), bottom_density=float(half["density"]), bottom_attenuation=float(half.get("attenuation", 0.0)))
    return o


def pd_table(record, chain_cfg: dict | None = None, fluctuation: str | None = None, rows: int | None = None):
    """(snr_db, pd): the probability that the chain as configured reports a point echo, against one row's link-budget SNR
    (`live.detection_probability`, the detector's own plan and thresholds). `record`: a ping or the sonar's template."""
    from . import live
    cfg = dict(chain_cfg or {})
    if fluctuation is None:
        fluctuation = options(cfg.get("coverage"))["fluctuation"] if cfg.get("coverage") is not None else "none"
    pd = live.detection_probability(PD_SNR_DB, record, cfg, rows=rows, fluctuation=fluctuation)
    return [float(v) for v in PD_SNR_DB], [float(min(max(v, 0.0), 1.0)) for v in np.atleast_1d(pd)]


def budget_of(d: dict, wind_speed: float = 5.0) -> dict:
    """The `budget:` block of a build (`d`: buildspec.derive or describe), from buildspec.budget: its source level, the
    ambient noise spectrum at `wind_speed` m/s, one row's gain against that noise, and its absorption."""
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
