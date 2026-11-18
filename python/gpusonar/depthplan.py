"""Own depth as a design and operating lever: detection margin against own depth and range for target classes.

    margin = (free-field link budget of the build at the range) + target strength + two-way propagation factor - threshold

The budget is `buildspec.performance`'s (spherical spreading, absorption, ambient noise, the row's gain against sea
noise, the rows' detection gain); the factor is `gpusonar.lloyd`'s: the surface image with roughness and bubble loss
across the band, and, when the environment has a bottom, the layer's image series (isovelocity), or Bellhop's arrivals
when it has a sound-speed profile. `plan` returns, per target class: the margin grid, the own depth that maximises the
worst margin over the class's depths at the range of interest, the band of own depths that hold it, the reach against
own depth, the target depths left uncovered at one own depth, how many own depths cover all of them, and the band of
own depths that holds two classes at once. tools/popoto_depth_planner.py is the command.

An environment is a mapping (`ENVIRONMENT_FIELDS`); a build spec carries a list of them under `environments:` (or the
name of a file under configs/builds/environments/).
"""

from __future__ import annotations

import warnings

import numpy as np

from . import acoustics, environment, lloyd

# One environment schema for the build layer and the scenarios: gpusonar/environment.py (FIELDS, the loader and the
# checks). The planner reads the fields below; an environment may carry the others (temperature, absorption, ...).
ENVIRONMENT_FIELDS = {k: environment.FIELDS[k] for k in environment.PLANNING_FIELDS}
NO_PATH_DB = -99.0                    # the factor is cut off here: below it there is no path to speak of (a shadow zone)
DEFAULT_LIMITS = [5.0, 300.0]
DEFAULT_CLEARANCE = 10.0


def load_environments(value) -> list:
    """The `environments` value of a build spec as a list: a list as written, or the named file's `environments:` list
    (gpusonar.environment.load_list)."""
    return environment.load_list(value)


def environment_problems(envs) -> list:
    """(field, message) for everything wrong with a list of environments (gpusonar.environment.problems)."""
    return environment.problems(envs)


def wind_of(env: dict, default: float = environment.DEFAULT_WIND_SPEED_M_S) -> float:
    return float(environment.wind_speed(env, default))


def surface_of(env: dict) -> lloyd.Surface:
    return lloyd.Surface.from_config({"wind_speed": wind_of(env), "rms_height": env.get("rms_wave_height_m"), "bubble_model": env.get("bubble_model")},
                                     float(env.get("sound_speed_m_s", acoustics.NOMINAL_SOUND_SPEED)))


def layer_of(env: dict, absorption_db_per_m: float = 0.0) -> lloyd.Layer:
    bottom = env.get("bottom")
    if isinstance(bottom, str):
        bottom = {"sediment": bottom}
    has = env.get("water_depth_m") is not None and bottom is not None
    return lloyd.Layer(depth=float(env["water_depth_m"]) if has else None, bottom=bottom if has else None, sound_speed=float(env.get("sound_speed_m_s", acoustics.NOMINAL_SOUND_SPEED)),
                       max_grazing_deg=30.0, absorption_db_per_m=absorption_db_per_m)


def own_depth_grid(env: dict, n: int = 100, max_depth: float | None = None) -> np.ndarray:
    """Own depths the plan searches: the environment's limits when it states them, else from 5 m to the platform's deepest (`max_depth`, a build's
    `platform.max_depth_m`) or 300 m when neither is known; never deeper than the platform's limit, nor closer to the bottom than the clearance."""
    lim = env.get("own_depth_limits_m")
    lo, hi = (float(x) for x in (lim or [DEFAULT_LIMITS[0], max_depth if max_depth else DEFAULT_LIMITS[1]]))
    if max_depth:
        hi = min(hi, float(max_depth))
    if env.get("water_depth_m") is not None:
        hi = min(hi, float(env["water_depth_m"]) - float(env.get("bottom_clearance_m", DEFAULT_CLEARANCE)))
    if hi <= lo:
        raise ValueError(f"environment {env.get('name')!r}: no operating depth between the shallowest allowed ({lo:g} m) and the bottom clearance")
    step = max(np.ceil((hi - lo) / (n - 1)), 1.0)
    return np.unique(np.r_[np.arange(lo, hi + 1e-9, step), hi])


def factor(env: dict, f_lo: float, f_hi: float, own_depths, target_depths, ranges, projector_below_m: float = 0.0, absorption_db_per_m: float = 0.0) -> dict:
    """The two-way factor (lloyd.summarise's figures, dB) on [own depth, target depth, range], and how it was computed."""
    own, zt, rr = (np.asarray(x, dtype=float) for x in (own_depths, target_depths, ranges))
    surface = surface_of(env)
    keys = ("peak_db", "energy_db", "narrowband_db")
    out = {k: np.zeros((own.size, zt.size, rr.size)) for k in keys}
    if env.get("profile") is not None:
        from .sim.bellhop import OceanEnvironment, find_bellhop, run_bellhop
        if find_bellhop() is None:
            raise RuntimeError(f"environment {env.get('name')!r} has a sound-speed profile: that needs Bellhop (set GPUSONAR_BELLHOP)")
        prof = env["profile"]
        bottom = env.get("bottom") or {"speed": float(prof["speed_m_s"][-1]), "density": 1.0, "attenuation": 0.0}       # no bottom given: one that returns nothing
        kw = {"sediment": bottom} if isinstance(bottom, str) else {"bottom_speed": float(bottom["speed"]), "bottom_density": float(bottom["density"]), "bottom_attenuation": float(bottom.get("attenuation", 0.0))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ocean = OceanEnvironment(depth=float(env["water_depth_m"]), ssp_depth=np.asarray(prof["depth_m"], dtype=float), ssp_speed=np.asarray(prof["speed_m_s"], dtype=float), **kw)
        f = lloyd.band(f_lo, f_hi, 257)
        for j, z in enumerate(zt):                                              # reciprocity: the source at the target's depth
            try:
                arr = run_bellhop(ocean, 0.5 * (f_lo + f_hi), float(z), own, rr, angles=(-30.0, 30.0))
            except StopIteration:
                raise RuntimeError(f"Bellhop's arrivals file was cut short (source at {z:g} m in environment {env.get('name')!r})") from None
            for i in range(own.size):
                for m in range(rr.size):
                    h, s = lloyd.bellhop_transfer(arr, i, m, f, surface, ocean.speed_at(float(z)), ocean.speed_at(0.0))
                    res = lloyd.summarise(h, s)
                    for k in keys:
                        out[k][i, j, m] = float(res[k])
        return {**out, "method": "Bellhop arrivals over the profile (flat bottom, range-independent); the projector taken at the array's depth"}
    layer = layer_of(env, absorption_db_per_m)
    ducted = layer.depth is not None
    f = lloyd.band(f_lo, f_hi, 513 if ducted else 129)
    for i, z in enumerate(own):                                                 # one own depth at a time: bounded memory
        h, s = lloyd.one_way(f, z, zt[:, None], rr[None, :], surface, layer)
        if projector_below_m:
            z_tx = z + projector_below_m if layer.depth is None else min(z + projector_below_m, layer.depth)
            res = lloyd.summarise(h, s, *lloyd.one_way(f, z_tx, zt[:, None], rr[None, :], surface, layer))
        else:
            res = lloyd.summarise(h, s)
        for k in keys:
            out[k][i] = res[k]
    return {**out, "method": ("image series of an isovelocity layer over a fluid bottom (Rayleigh reflection), to 30 degrees of grazing" if ducted
                              else "direct path and surface image (no bottom return), isovelocity")}


def _intervals(x: np.ndarray, ok: np.ndarray) -> list:
    """Runs of True in `ok` as [first x, last x] pairs."""
    out, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(ok) - 1):
            out.append([float(x[start]), float(x[i if v else i - 1])])
            start = None
    return out


def _cover(held: np.ndarray, own: np.ndarray, most: int = 6) -> list:
    """Greedy cover: held[own depth, target depth] -> own depths chosen in turn and the share of target depths covered after each."""
    covered = np.zeros(held.shape[1], dtype=bool)
    steps = []
    for _ in range(most):
        gain = (held & ~covered).sum(axis=1)
        i = int(np.argmax(gain))
        if gain[i] == 0:
            break
        covered |= held[i]
        steps.append({"own_depth_m": float(own[i]), "covered_share": float(covered.mean())})
        if covered.all():
            break
    return steps


def plan(d: dict, env: dict, classes: list, range_of_interest: float = 12000.0, own_depths=None, ranges=None, threshold_db: float | None = None,
         statistic: str = "peak_db", all_target_depths=None, cache: dict | None = None, keep_grid: bool = True) -> dict:
    """The depth plan of a build (`d`: buildspec.derive / describe) in one environment for resolved target classes
    (target_classes.resolve). Margins are at the LOW end of each class's target strength; `margin_high_ts_db` adds the rest."""
    from . import buildspec as bs
    threshold_db = bs.BUDGET_THRESHOLD_DB if threshold_db is None else float(threshold_db)
    wind = wind_of(env)
    eq, gains = bs.budget(d, wind)
    f_lo, f_hi = d["band_hz"]
    max_depth = (d.get("platform") or {}).get("max_depth_m")
    own = own_depth_grid(env, max_depth=max_depth) if own_depths is None else np.asarray(own_depths, dtype=float)
    heavy = env.get("profile") is not None or (env.get("water_depth_m") is not None and env.get("bottom") is not None)      # many paths: a coarser grid
    if own_depths is None and heavy and own.size > 24:
        own = own[::int(np.ceil(own.size / 24))]
    rr = (np.unique(np.r_[np.linspace(max(d["blind_range_m"] * 1.2, 1000.0), max(range_of_interest, d["window_m"][1]), 16 if heavy else 36), range_of_interest])
          if ranges is None else np.asarray(ranges, dtype=float))
    i_r = int(np.argmin(np.abs(rr - range_of_interest)))
    deepest = float(env["water_depth_m"]) - 1.0 if env.get("water_depth_m") is not None else max(max(c["depth_m"][1] for c in classes), float(own[-1]))
    if all_target_depths is None:
        all_target_depths = np.unique(np.round(np.r_[np.geomspace(1.0, deepest, 14 if heavy else 48), [z for c in classes for z in c["depth_m"] if z <= deepest],
                                                     [0.5 * (c["depth_m"][0] + min(c["depth_m"][1], deepest)) for c in classes if c["depth_m"][0] <= deepest]], 2))
    zt = np.asarray(all_target_depths, dtype=float)
    key = (env.get("name"), float(d["projector_m"][2]), f_lo, f_hi, own.tobytes(), zt.tobytes(), rr.tobytes())
    if cache is not None and key in cache:
        fac = cache[key]                                                          # the factor does not depend on the rows
    else:
        fac = factor(env, f_lo, f_hi, own, zt, rr, float(d["projector_m"][2]), eq.absorption_db_per_m())
        fac = {k: (np.maximum(v, NO_PATH_DB) if isinstance(v, np.ndarray) else v) for k, v in fac.items()}
        if cache is not None:
            cache[key] = fac
    free = np.asarray(eq.snr(rr)) - threshold_db                                  # margin of a 0 dB target in free field, rows included
    f3 = fac[statistic]
    out = {"build": d.get("build") or d.get("name"), "rows": d["rows"], "environment": {k: v for k, v in env.items()}, "wind_speed_m_s": wind, "rms_wave_height_m": surface_of(env).height(),
           "platform_max_depth_m": max_depth, "bottom_clearance_m": float(env.get("bottom_clearance_m", DEFAULT_CLEARANCE)) if env.get("water_depth_m") is not None else None,
           "profile_rests_on": ("the sound-speed profile supplied with the environment" if env.get("profile") is not None else "isovelocity water: no profile was supplied (the bottom rung: nothing known of the profile)"),
           "method": fac["method"], "statistic": statistic, "threshold_db": threshold_db, "range_of_interest_m": float(rr[i_r]), "band_hz": [f_lo, f_hi],
           "projector_below_array_m": float(d["projector_m"][2]), "gains": gains, "own_depths_m": own.tolist(), "ranges_m": rr.tolist(), "target_depths_m": zt.tolist(),
           "free_field_margin_0db_target_db": free.tolist(), "classes": {}}
    held_by = {}
    for c in classes:
        lo, hi = c["depth_m"]
        sel = (zt >= lo - 1e-6) & (zt <= min(hi, deepest) + 1e-6)
        if not sel.any():
            out["classes"][c["name"]] = {"skipped": f"its depths ({lo:g}-{hi:g} m) are not in this water"}
            continue
        ts_lo, ts_hi = c["ts_db"]
        worst = f3[:, sel, :].min(axis=1) + free[None, :] + ts_lo                 # [own, range]
        mean = lloyd.depth_average(f3[:, sel, :], axis=1) + free[None, :] + ts_lo
        at = worst[:, i_r]
        best = int(np.argmax(at))
        held = worst >= 0.0
        reach = np.array([rr[np.flatnonzero(h)[-1]] if h.any() else 0.0 for h in held])
        gaps = np.array([bool(h.any() and not h[:np.flatnonzero(h)[-1] + 1].all()) for h in held])
        all_d = f3[best, :, i_r] + free[i_r] + ts_lo                                 # every target depth, at the recommended own depth
        entry = {"ts_db": [ts_lo, ts_hi], "ts_status": c.get("status"), "depth_m": [lo, hi], "target_depths_used_m": zt[sel].tolist(),
                 "free_field_margin_db": float(free[i_r] + ts_lo), "margin_high_ts_db": float(ts_hi - ts_lo),
                 "margin_worst_db": worst.tolist(), "margin_at_range_worst_db": at.tolist(), "margin_at_range_depth_mean_db": mean[:, i_r].tolist(),
                 "factor_at_range_worst_db": (at - free[i_r] - ts_lo).tolist(),
                 "recommended": {"own_depth_m": float(own[best]), "inside_platform_limit": (None if not max_depth else bool(own[best] <= max_depth)),
                                 "at_the_deepest_searched": bool(best == len(own) - 1), "margin_db": float(at[best]), "factor_db": float(at[best] - free[i_r] - ts_lo),
                                 "held_band_m": _intervals(own, at >= 0.0), "within_3db_band_m": _intervals(own, at >= at[best] - 3.0),
                                 "reach_m": float(reach[best]), "nearer_gaps": bool(gaps[best]),
                                 "target_depths_not_held_m": _intervals(zt, all_d < 0.0)},
                 "shallowest_own_depth_held_m": float(own[np.flatnonzero(at >= 0.0)[0]]) if (at >= 0.0).any() else None,
                 "reach_by_own_depth_m": reach.tolist(), "at_own_depth": {}}
        for z in (own[0], 10.0, 30.0, 100.0):
            if own[0] <= z <= own[-1]:
                k = int(np.argmin(np.abs(own - z)))
                entry["at_own_depth"][f"{own[k]:g}"] = {"margin_db": float(at[k]), "reach_m": float(reach[k])}
        # depth diversity: every target depth of the water, this class's target strength, at the range of interest
        m_all = f3[:, :, i_r] + free[i_r] + ts_lo
        entry["depth_diversity"] = {"what": "own depths chosen in turn so that every target depth from 1 m to the deepest considered is held at the range of interest, "
                                            "at this class's low target strength", "deepest_target_m": float(zt[-1]), "steps": _cover(m_all >= 0.0, own),
                                    "one_depth_best_share": float((m_all >= 0.0).mean(axis=1).max())}
        if not keep_grid:
            entry.pop("margin_worst_db")
        out["classes"][c["name"]] = entry
        held_by[c["name"]] = at
    names = list(held_by)
    out["together"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = np.minimum(held_by[a], held_by[b])
            k = int(np.argmax(both))
            out["together"][f"{a} + {b}"] = {"held_band_m": _intervals(own, both >= 0.0), "best_own_depth_m": float(own[k]), "margin_of_the_weaker_db": float(both[k])}
    return out


def elevation_for(d: dict, snr_map_db: float, range_m: float) -> dict | None:
    """What stacked rows add for a target detected with `snr_map_db` at the summed map: the elevation sigma from noise
    (the bound of gpusonar.elevation for this build's rows) and the depth sigma it is at `range_m`. None for one row."""
    el = d.get("elevation")
    if not el:
        return None
    from . import buildspec as bs
    from . import elevation as elev
    n = d["per_row"][0]
    row_snr = snr_map_db - bs.rows_detection_gain_db(d["rows"])
    if row_snr < 10.0:
        return {"row_snr_db": float(row_snr), "sigma_deg": None, "depth_sigma_m": None}
    sig = float(elev.sin_elevation_sigma(np.repeat(el["heights_m"], n), 10.0 ** (min(row_snr, 40.0) / 10.0) / n, d["band_hz"][0], d["band_hz"][1], d["sound_speed"]))
    return {"row_snr_db": float(row_snr), "sigma_deg": float(np.degrees(sig)), "depth_sigma_m": float(range_m * sig)}
