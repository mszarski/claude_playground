"""The scenario catalogue as the pictures see it: each scenario's family, the ping a page shows, the platform's
track over the whole run, and the thinning a slow scene gets on a CPU (docs/tasks/VISUALIZATION.md section 2.2, V2).

Families are those of docs/handoff/SIMULATION.md section 4 (the last heading split in two: fjord water and water
that changes with range), written out here by name so that a picture never depends on parsing a document; the test
`test_every_scenario_has_a_family` checks the two agree. Nothing here changes a scenario file: a page's choices are
overrides for that page's own run, and the page says which.
"""

from __future__ import annotations

import copy

import numpy as np

# (family, short name for files, scenarios). Order: the order of SIMULATION.md section 4.
FAMILIES = [
    ("Point targets", "point_targets", (
        "point_grid_l12_5d", "point_grid_p4_2v", "point_grid_fls_300k", "point_grid_fls_750k", "point_grid_sas_auv",
        "point_grid_sas_rail", "point_grid_flank_32", "point_grid_popoto_8", "point_grid_popoto_8_low",
        "point_grid_popoto_4x8", "pool_wires_l12_5d", "stripmap_points", "rail_points", "rail_position_errors")),
    ("Forward-looking sonar", "fls", (
        "pool_fls_l12_5d", "pool_fls_p4_2v", "harbour_fls_750k", "seafloor_objects_fls", "fls_120k_transit",
        "fls_170k_transit", "fls_330k_transit", "fls_120k_cluttered_seabed", "fls_120k_rough_surface",
        "fls_170k_traffic", "fls_glints_on_sediment", "fls_highlight_target_aspect", "fls_120k_quay_multipath",
        "pool_fls_p4_2v_walls")),
    ("Synthetic aperture sonar and buried objects", "sas", (
        "wreck_sas", "minefield_sas", "sas_minsas_survey", "sas_minsas_boulder_field", "sas_minsas_rocky_ripples",
        "sas_minsas_speckle", "rail_objects", "rail_cylinder_glint", "buried_cylinder", "buried_objects",
        "sas_auv_multipath", "sas_auv_multipath_free")),
    ("Flank array, 32 hydrophones", "flank", (
        "flank_target_1km", "flank_target_5km", "flank_target_10km", "flank_seamount_3d", "flank_patrol",
        "flank_patrol_doppler", "flank_patrol_doppler_underway", "flank_patrol_doppler_tone", "flank_sst_shallow",
        "flank_sst_spread", "flank_seamount_3d_opening", "flank_seamount_3d_opening_nx2d", "flank_level_3d",
        "flank_bank_3d", "flank_level_nx2d", "flank_ridge_3d", "flank_ridge_nx2d", "flank_canyon_3d",
        "flank_canyon_nx2d")),
    ("TR 0702 validation and ASIAEX", "validation", (
        "sst_pursuit", "sst_asiaex", "sst_echo_level", "sst_sediment_reverb", "asiaex_dahl_2004")),
    ("Popoto 27.5 kHz and the first long-range design", "popoto", (
        "popoto_trial_100m", "popoto_trial_1000m", "popoto_trial_2000m", "popoto_far_mover", "popoto_far_mover_doppler")),
    ("The 12 km design", "design_12km", (
        "popoto_12km", "popoto_12km_2row", "popoto_12km_3row", "popoto_12km_doppler", "popoto_12km_two_ships",
        "popoto_12km_large_submerged", "popoto_12km_large_submerged_tone", "popoto_12km_deep", "popoto_12km_deep_sea_state_5",
        "popoto_env_winter_sand_200m", "popoto_env_summer_mud_60m", "popoto_env_gravel_30m",
        "popoto_env_deep_silt_1000m")),
    ("Stacked rows", "rows", (
        "popoto_elevation_free", "popoto_elevation_shallow", "popoto_elevation_rehearsal", "popoto_face_2row_shallow")),
    ("Staged builds", "staged", (
        "popoto_testauv_trial_1row", "popoto_static_trial_2row", "popoto_static_trial_3row", "popoto_static_free",
        "popoto_static_passing")),
    ("Direct sound and overload", "direct", (
        "popoto_testauv_direct_overload", "popoto_testauv_direct_overload_fast",
        "popoto_testauv_direct_overload_gain_switch")),
    ("Fjord-type water", "fjord", ("fjord_testauv_trial_1row", "fjord_static_trial_2row", "fjord_static_trial_3row")),
    ("Water that changes with range", "ssp_range", (
        "ssp_front_static_trial_2row", "ssp_front_testauv_trial_1row", "ssp_thermocline_static_trial_3row")),
]

# One page per family is committed (docs/tasks/VISUALIZATION.md question 5): the scenario shown for each. Chosen to
# show the family's propagation at its most particular (ours).
FAMILY_EXAMPLE = {
    "point_targets": "point_grid_flank_32", "fls": "fls_120k_quay_multipath", "sas": "buried_cylinder",
    "flank": "flank_target_1km", "validation": "sst_asiaex", "popoto": "popoto_trial_100m",
    "design_12km": "popoto_env_winter_sand_200m", "rows": "popoto_elevation_shallow", "staged": "popoto_static_passing",
    "direct": "popoto_testauv_direct_overload", "fjord": "fjord_testauv_trial_1row",
    "ssp_range": "ssp_front_testauv_trial_1row",
}


def scenario_family(name: str) -> tuple[str, str]:
    """(family title, short name) of a scenario; ("Other", "other") for a scenario added since this list."""
    for title, short, names in FAMILIES:
        if name in names:
            return title, short
    return "Other", "other"


# Scenes too slow for one page on a 4-core CPU, thinned for the page as tools/gallery.py --quick thins them (its
# factors; ours to reuse here). The page says so. Densities only: one ping is simulated anyway.
QUICK_DENSITY = {
    "pool_fls_l12_5d": 0.1, "pool_fls_p4_2v": 0.1, "harbour_fls_750k": 0.25, "wreck_sas": 0.3, "minefield_sas": 0.3,
    "rail_objects": 0.1, "sas_minsas_survey": 0.5, "sas_minsas_boulder_field": 0.5, "sas_minsas_rocky_ripples": 0.5,
    "sas_minsas_speckle": 0.02, "pool_fls_p4_2v_walls": 0.1,
    "fls_120k_quay_multipath": 0.1,          # not in the gallery: ours (one ping takes 2 min on this CPU unthinned)
}


def thinned_scene(cfg: dict, factor: float) -> list:
    """The scene with every `density` scaled by `factor` (tools/gallery.py `_scale_density`)."""
    scene = copy.deepcopy(cfg.get("scene") or [])
    for item in scene:
        if "density" in item:
            item["density"] = float(item["density"]) * factor
    return scene


def view_plan(cfg: dict, quick: bool = False) -> dict:
    """What a page simulates, from a scenario's config as written (`load_scenario_config(..., expand=False)`):

    {overrides: for run_scenario, ping: the index shown, n_pings, track [N, 3] (the platform's position at every
    ping of the full run, from its platform block; no simulation), note: one sentence for the page}.

    The ping shown: the first, except for a synthetic-aperture run (mode `sas`, more than one ping), whose middle ping
    is shown (the targets lie abreast of the middle of a track; the first ping often sees none of them). That ping is
    simulated alone at its own position (the scene at time 0, which is the same scene for these static runs; for a run
    with platform `motion` the sway and heave realisation of a one-ping run, not the full run's)."""
    from ..sonar import SonarModel
    plat = cfg.get("platform") or {}
    pings = cfg.get("pings") or {}
    interval = float(pings.get("interval", 1.0))
    pos = np.asarray(plat.get("position", [0.0, 0.0, 0.0]), float)
    vel = np.asarray(plat.get("velocity", [0.0, 0.0, 0.0]), float)
    mode = SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides")).config.get("mode")
    mux = len(SonarModel.load(cfg["sonar"], cfg.get("sonar_overrides")).mux_groups)
    over: dict = {}
    if plat.get("legs") is not None:
        speed = float(plat.get("speed", np.linalg.norm(vel)))
        track, leg_of = [], []
        for li, leg in enumerate(plat["legs"]):
            h = np.radians(float(leg["heading_deg"]))
            d = np.array([np.cos(h), np.sin(h), 0.0])
            n = int(leg["pings"]) if "pings" in leg else int(np.floor(float(leg["length"]) / (speed * interval))) + 1
            for i in range(n):
                track.append(np.asarray(leg["start"], float) + d * speed * i * interval)
                leg_of.append(li)
        track = np.array(track)
        n0 = leg_of.count(0)
        k = n0 // 2
        leg = dict(plat["legs"][0], start=track[k].tolist(), pings=1)
        leg.pop("length", None)
        over["platform"] = {"legs": [leg]}
        note = f"ping {k} of {len(track)} (the middle of leg 0 of {len(plat['legs'])}), simulated alone at its position"
    elif plat.get("raster") is not None:
        r = plat["raster"]
        xs, ys = (np.arange(*[float(v) for v in r[a]]) for a in ("x", "y"))
        track = np.array([[x, y, pos[2]] for x in xs for y in ys])
        k = len(track) // 2
        sx, sy = float(r["x"][2]), float(r["y"][2])
        over["platform"] = {"raster": {"x": [track[k, 0], track[k, 0] + 0.5 * sx, sx],
                                       "y": [track[k, 1], track[k, 1] + 0.5 * sy, sy]}}
        note = f"raster position {k} of {len(track)} (the middle), simulated alone"
    else:
        count = int(pings.get("count", 1))
        track = np.array([pos + vel * i * interval for i in range(count)])
        k = count // 2 if (mode == "sas" and count > 1) else 0
        over["pings"] = {"count": 1}
        if k:
            over["platform"] = {"position": track[k].tolist()}
            note = (f"ping {k} of {count} (the middle of the track), simulated alone at its position"
                    + ("; the sway and heave are a one-ping draw, not the full run's" if plat.get("motion") else ""))
        else:
            note = f"ping 0 of {count}" + ("" if count > 1 else " (the only ping)")
    if mux > 1:
        note += f"; the array is read in {mux} multiplexed halves, one a ping: this ping holds one"
    if quick and cfg.get("name") in QUICK_DENSITY:
        f = QUICK_DENSITY[cfg["name"]]
        over["scene"] = thinned_scene(cfg, f)
        note += f"; scene densities x {f:g} for a CPU (as tools/gallery.py --quick)"
    return {"overrides": over, "ping": int(k), "n_pings": len(track), "track": np.asarray(track, float),
            "note": note, "mode": mode}
