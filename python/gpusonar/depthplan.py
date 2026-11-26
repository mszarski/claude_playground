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

from . import acoustics, environment, lloyd, reverb_budget

# One environment schema for the build layer and the scenarios: gpusonar/environment.py (FIELDS, the loader and the
# checks). The planner reads the fields below; an environment may carry the others (temperature, absorption, ...).
ENVIRONMENT_FIELDS = {k: environment.FIELDS[k] for k in environment.PLANNING_FIELDS}
NO_PATH_DB = -99.0                    # the factor is cut off here: below it there is no path to speak of (a shadow zone)
DEFAULT_LIMITS = [5.0, 300.0]
DEFAULT_CLEARANCE = 10.0
LOCAL_PROFILE_NOTE = "the environment's water changes with range (profiles_by_range): planned with the LOCAL profile (range 0) at every range, as a vehicle that has only its own cast would"


def _local_water(env: dict) -> tuple:
    """(env, ranged): the planner is range independent. An environment whose water changes with range (`profiles_by_range`)
    is planned with its local profile at every range; `ranged` says that this was done (it goes into the plan's words)."""
    if env.get("profile") is None and env.get("profiles_by_range") is not None:
        z, c = environment.profile_arrays(env)
        return {**env, "profile": {"depth_m": z, "speed_m_s": c}}, True
    return env, False


ROWS_GAIN_WHAT = {
    "none": "rows gain nothing against reverberation (bound A, the default: it arrives by the echo's own multipath; ours)",
    "incoherent": "rows gain against reverberation what the build's detector gains from them against noise (buildspec.detection "
                  "rows_gain_db): the rows' reverberation taken to be independent looks, as the noise is; the most the chain's power sum of "
                  "the rows' maps can gain (ours)",
    "number": "rows gain the stated dB against reverberation (an input: e.g. a measurement; docs/design/popoto_trade_study.md Sec. 5 "
              "measured 2.47 dB for 2 rows and 3.38 dB for 3 on the shelf, ours, one site, 3 pings)",
}


def rows_gain_against_reverberation(setting, det: dict, where: str = "reverberation_rows_gain") -> tuple:
    """(dB, what) of a `reverberation_rows_gain` setting for a build whose detection is `det` (buildspec.detection): what
    its stacked rows gain against the site's reverberation, added to one row's echo-over-reverberation ratio.

    - None or "none" (the default): 0 dB (bound A).
    - "incoherent": `det["rows_gain_db"]`, the build's detector's own gain from its rows against noise: the rows'
      reverberation as independent looks, as the noise is (the worth of `rows` effective looks to this detector; 3.99 dB
      for three rows of 32, 2.54 for two). Not 10 log10(rows) (bound B of docs/design/popoto_trade_study.md): adding the
      rows' maps in power gains less than that even against noise.
    - a number (dB, >= 0): used as given.

    A build with one row gains nothing, whatever is given."""
    if setting is None or (isinstance(setting, str) and setting == "none"):
        return 0.0, ROWS_GAIN_WHAT["none"]
    if isinstance(setting, str):
        if setting == "incoherent":
            return (float(det["rows_gain_db"]) if int(det["rows"]) > 1 else 0.0), ROWS_GAIN_WHAT["incoherent"]
        raise ValueError(f"{where}: {setting!r} unknown: give none, incoherent or a number of dB (>= 0)")
    if isinstance(setting, bool) or not isinstance(setting, (int, float, np.integer, np.floating)) or not np.isfinite(float(setting)) or float(setting) < 0.0:
        raise ValueError(f"{where}: {setting!r}: give none, incoherent or a number of dB (>= 0)")
    return (float(setting) if int(det["rows"]) > 1 else 0.0), ROWS_GAIN_WHAT["number"]


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
    env, ranged = _local_water(env)
    surface = surface_of(env)
    keys = ("peak_db", "energy_db", "narrowband_db")
    out = {k: np.zeros((own.size, zt.size, rr.size)) for k in keys}
    if env.get("profile") is not None:
        from .sim.bellhop import OceanEnvironment, find_bellhop, run_bellhop
        if find_bellhop() is None:
            raise RuntimeError(f"environment {env.get('name')!r} has a sound-speed profile: that needs Bellhop (set GPUSONAR_BELLHOP)")
        prof = env["profile"]
        bottom = env.get("bottom") or {"speed": float(prof["speed_m_s"][-1]), "density": 1.0, "attenuation": 0.0}       # no bottom given: one that returns nothing
        # A row with shear (rock) is an elastic solid here as in the simulator (A8): Bellhop's shear column. A typed
        # bottom with shear_speed > 0 the same; a fluid one writes no shear, as before.
        if isinstance(bottom, str):
            kw = {"sediment": bottom}
            lloyd.Layer(depth=float(env["water_depth_m"]), bottom={"sediment": bottom}).halfspace()     # says once if the row's shear is unconfirmed
        else:
            kw = {"bottom_speed": float(bottom["speed"]), "bottom_density": float(bottom["density"]), "bottom_attenuation": float(bottom.get("attenuation", 0.0))}
            lloyd.check_halfspace(bottom, f"environment {env.get('name')!r}.bottom")
            if float(bottom.get("shear_speed", 0.0)) > 0.0:
                kw.update(bottom_shear_speed=float(bottom["shear_speed"]), bottom_shear_attenuation=float(bottom.get("shear_attenuation", 0.0)))
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
        return {**out, "method": "Bellhop arrivals over the profile (flat bottom, range-independent); the projector taken at the array's depth" + (f"; {LOCAL_PROFILE_NOTE}" if ranged else "")}
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
    elastic = ducted and float((layer.halfspace() or {}).get("shear_speed", 0.0)) > 0.0
    return {**out, "method": (f"image series of an isovelocity layer over {'an elastic' if elastic else 'a fluid'} bottom (Rayleigh reflection), to 30 degrees of grazing" if ducted
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


def reverberation_of(d: dict, env: dict, own, rr, absorption_db_per_m: float, cache: dict | None = None):
    """(levels [own, range] re the source level, dB, or None; note): the site's reverberation at the delay of a target at each
    range, by the coverage picture's term (C++, libs/live coverage.hpp) with the boundaries' scattering from the ONE rule
    (gpusonar.environment.boundary_scattering) and a beam of one row of the build (unweighted, broadside)."""
    from . import reverb_budget
    psi = reverb_budget.line_beam_integral(d["per_row"][0], d["pitch_m"], d["centre_hz"], d["sound_speed"])
    key = ("reverb", env.get("name"), tuple(d["band_hz"]), float(psi), float(d["projector_m"][2]), np.asarray(own, float).tobytes(), np.asarray(rr, float).tobytes())
    if cache is not None and key in cache:
        return cache[key]
    term = reverb_budget.site_term(env, d["band_hz"], d["sound_speed"], psi, absorption_db_per_m, f"environment {env.get('name')!r}")
    got = (reverb_budget.site_levels_db(term, own, rr, float(d["projector_m"][2]), d["sound_speed"]), term["note"])
    if cache is not None:
        cache[key] = got
    return got


def plan(d: dict, env: dict, classes: list, range_of_interest: float = 12000.0, own_depths=None, ranges=None, threshold_db: float | None = None,
         statistic: str = "peak_db", all_target_depths=None, cache: dict | None = None, keep_grid: bool = True, reverberation: bool = True,
         reverberation_rows_gain=None) -> dict:
    """The depth plan of a build (`d`: buildspec.derive / describe) in one environment for resolved target classes
    (target_classes.resolve). Margins are at the LOW end of each class's target strength; `margin_high_ts_db` adds the rest.
    A margin is over `threshold_db` of ONE row's SNR: by default what the build's own detector needs for the build's design
    detection probability with its rows added (buildspec.detection); the budget itself holds no row gain.

    Reverberation (`reverberation`, default on): where the site's boundary scattering follows from what it states
    (gpusonar.environment.boundary_scattering: stated `bottom_scattering` / `surface_scattering`, a named sediment, the
    wind), each margin is over the power sum of noise and the boundaries' reverberation at the target's delay
    (`reverberation_of`): margin = -10 log10(10^(-(SNR + G) / 10) + 10^(-SRR / 10)) - T1, with SNR one row's over noise, G
    the rows' gain against noise (`rows_gain_db`), T1 the one-row threshold and SRR the echo over reverberation (the rows
    are taken to gain nothing against reverberation: it comes in by the same multipath as the echo; ours). With nothing
    included, or `reverberation=False`, it is the noise-only margin, computed as before. `reverberation` in the result
    says which, and why.

    `reverberation_rows_gain` (opt-in; default None, bound A, the plan exactly as before): what the rows gain against the
    reverberation, g dB added to SRR (`rows_gain_against_reverberation`: "none", "incoherent" or a number of dB);
    `reverberation.rows_gain_against_reverberation_db` states the g used."""
    from . import buildspec as bs
    env, ranged = _local_water(env)
    wind = wind_of(env)
    eq, gains = bs.budget(d, wind)
    det = bs.detection(d)
    threshold_db = det["threshold_db"] if threshold_db is None else float(threshold_db)
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
    free = np.asarray(eq.snr(rr)) - threshold_db                                  # margin of a 0 dB target in free field: one row's SNR over the detector's need
    f3 = fac[statistic]
    rl, note = reverberation_of(d, env, own, rr, eq.absorption_db_per_m(), cache) if reverberation else (None, "not included: the plan was asked without it")
    gain = float(det["rows_gain_db"])
    g_rev, g_what = rows_gain_against_reverberation(reverberation_rows_gain, det)
    two_tl = 2.0 * np.asarray(eq.transmission_loss(rr), dtype=float)

    def M(fac, ts, i_own=slice(None), i_rng=slice(None)):
        """Margin (dB) of a factor `fac` ([own, ..., range] or [own, range], or with scalars where an index is one) and a
        target strength `ts` at own depths `i_own` and ranges `i_rng`: noise-only, summed in the order it always was
        (fac + free + ts: bit for bit the plan without the term), or with the site's reverberation."""
        noise = fac + free[i_rng] + ts
        if rl is None:
            return noise
        r = rl[i_own, i_rng]
        if np.ndim(fac) == 3:
            r = r[:, None, :]
        elif np.ndim(fac) == 2 and np.ndim(r) == 1:                           # [own, target depth] at one range
            r = r[:, None]
        srr = fac + ts - two_tl[i_rng] - r
        if g_rev != 0.0:                                                      # opt-in: the rows' gain against reverberation
            srr = srr + g_rev
        return reverb_budget.with_reverberation(noise + threshold_db + gain, srr) - threshold_db - gain
    out = {"build": d.get("build") or d.get("name"), "rows": d["rows"], "environment": {k: v for k, v in env.items()}, "wind_speed_m_s": wind, "rms_wave_height_m": surface_of(env).height(),
           "platform_max_depth_m": max_depth, "bottom_clearance_m": float(env.get("bottom_clearance_m", DEFAULT_CLEARANCE)) if env.get("water_depth_m") is not None else None,
           "profile_rests_on": (LOCAL_PROFILE_NOTE if ranged else "the sound-speed profile supplied with the environment" if env.get("profile") is not None else "isovelocity water: no profile was supplied (the bottom rung: nothing known of the profile)"),
           "method": fac["method"], "statistic": statistic, "threshold_db": threshold_db,
           "detection": {**{k: det[k] for k in ("probability", "threshold_db", "threshold_one_row_db", "threshold_high_db", "probability_high", "rows", "rows_gain_db", "statistic")},
                         "threshold_is_the_detectors": bool(abs(threshold_db - det["threshold_db"]) < 1e-9)},
           "range_of_interest_m": float(rr[i_r]), "band_hz": [f_lo, f_hi],
           "projector_below_array_m": float(d["projector_m"][2]), "gains": gains, "own_depths_m": own.tolist(), "ranges_m": rr.tolist(), "target_depths_m": zt.tolist(),
           "free_field_margin_0db_target_db": free.tolist(), "classes": {},
           "reverberation": {"included": rl is not None, "note": note, "rows_gain_against_reverberation_db": g_rev if rl is not None else None,
                             "level_at_range_re_source_db": None if rl is None else rl[:, i_r].tolist()}}       # [own depth], at the range of interest
    if reverberation_rows_gain is not None and reverberation_rows_gain != "none":       # opt-in only: the default plan's keys are as before
        out["reverberation"]["rows_gain_against_reverberation"] = {"setting": reverberation_rows_gain, "what": g_what}
    held_by = {}
    for c in classes:
        lo, hi = c["depth_m"]
        sel = (zt >= lo - 1e-6) & (zt <= min(hi, deepest) + 1e-6)
        if not sel.any():
            out["classes"][c["name"]] = {"skipped": f"its depths ({lo:g}-{hi:g} m) are not in this water"}
            continue
        ts_lo, ts_hi = c["ts_db"]
        worst_f = f3[:, sel, :].min(axis=1)                                       # [own, range]: the worst depth's factor
        worst = M(worst_f, ts_lo)
        mean = M(lloyd.depth_average(f3[:, sel, :], axis=1), ts_lo)
        at = worst[:, i_r]
        best = int(np.argmax(at))
        held = worst >= 0.0
        reach = np.array([rr[np.flatnonzero(h)[-1]] if h.any() else 0.0 for h in held])
        gaps = np.array([bool(h.any() and not h[:np.flatnonzero(h)[-1] + 1].all()) for h in held])
        all_d = M(f3[best, :, i_r], ts_lo, best, i_r)                               # every target depth, at the recommended own depth
        entry = {"ts_db": [ts_lo, ts_hi], "ts_status": c.get("status"), "depth_m": [lo, hi], "target_depths_used_m": zt[sel].tolist(),
                 "free_field_margin_db": float(free[i_r] + ts_lo), "margin_high_ts_db": float(ts_hi - ts_lo),
                 "free_field_detection_probability": float(bs.detection_probability(d, float(free[i_r] + ts_lo + threshold_db))),
                 "margin_worst_db": worst.tolist(), "margin_at_range_worst_db": at.tolist(), "margin_at_range_depth_mean_db": mean[:, i_r].tolist(),
                 "factor_at_range_worst_db": (worst_f[:, i_r] if rl is not None else at - free[i_r] - ts_lo).tolist(),
                 "recommended": {"own_depth_m": float(own[best]), "inside_platform_limit": (None if not max_depth else bool(own[best] <= max_depth)),
                                 "at_the_deepest_searched": bool(best == len(own) - 1), "margin_db": float(at[best]), "factor_db": float(worst_f[best, i_r] if rl is not None else at[best] - free[i_r] - ts_lo),
                                 "detection_probability": float(bs.detection_probability(d, float(at[best] + threshold_db))),
                                 "held_band_m": _intervals(own, at >= 0.0), "within_3db_band_m": _intervals(own, at >= at[best] - 3.0),
                                 "reach_m": float(reach[best]), "nearer_gaps": bool(gaps[best]),
                                 "target_depths_not_held_m": _intervals(zt, all_d < 0.0)},
                 "shallowest_own_depth_held_m": float(own[np.flatnonzero(at >= 0.0)[0]]) if (at >= 0.0).any() else None,
                 "reach_by_own_depth_m": reach.tolist(), "at_own_depth": {}}
        if rl is not None:                                                        # what noise alone would give there, and which limits
            noise_at = worst_f[best, i_r] + free[i_r] + ts_lo
            held_n = (worst_f + free[None, :] + ts_lo)[best] >= 0.0
            entry["recommended"].update(margin_noise_only_db=float(noise_at), reach_noise_only_m=float(rr[np.flatnonzero(held_n)[-1]]) if held_n.any() else 0.0,
                                        limited_by="reverberation" if noise_at - at[best] > 3.0 else "noise")
        for z in (own[0], 10.0, 30.0, 100.0):
            if own[0] <= z <= own[-1]:
                k = int(np.argmin(np.abs(own - z)))
                entry["at_own_depth"][f"{own[k]:g}"] = {"margin_db": float(at[k]), "reach_m": float(reach[k])}
        # depth diversity: every target depth of the water, this class's target strength, at the range of interest
        m_all = M(f3[:, :, i_r], ts_lo, slice(None), i_r)
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


def elevation_for(d: dict, row_snr_db: float, range_m: float) -> dict | None:
    """What stacked rows add for a target whose SNR in ONE row is `row_snr_db` (a plan's margin plus its threshold): the
    elevation sigma from noise (the bound of gpusonar.elevation for this build's rows) and the depth sigma it is at
    `range_m`. None for one row."""
    el = d.get("elevation")
    if not el:
        return None
    from . import elevation as elev
    n = d["per_row"][0]
    row_snr = float(row_snr_db)
    if row_snr < 10.0:
        return {"row_snr_db": float(row_snr), "sigma_deg": None, "depth_sigma_m": None}
    sig = float(elev.sin_elevation_sigma(np.repeat(el["heights_m"], n), 10.0 ** (min(row_snr, 40.0) / 10.0) / n, d["band_hz"][0], d["band_hz"][1], d["sound_speed"]))
    return {"row_snr_db": float(row_snr), "sigma_deg": float(np.degrees(sig)), "depth_sigma_m": float(range_m * sig)}
