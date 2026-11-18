#!/usr/bin/env python3
"""Own depth as a lever: detection margin against own depth and range, and the depth to operate (or hang) at.

    tools/popoto_depth_planner.py --build popoto_face_3x32                       # the spec's targets and environments
    tools/popoto_depth_planner.py --build SPEC.yaml --environment shelf_100m_sand --target ship --range 12000
    tools/popoto_depth_planner.py --build popoto_static_3x32 --water-depth 80 --bottom medium_sand --sea-state 3 \\
        --rms-height 0.15 --own-depth 5 60 --ts 10 --target-depth 20 60 --range 2500 --out /tmp/plan
    tools/popoto_depth_planner.py --shipped                                      # docs/design/popoto_depth_planner.json and .png

Given a build spec (or `--sonar NAME [--deploy NAME]`), an environment (water depth, bottom by name from
configs/sediments.yaml or as numbers, wind or sea state and/or rms wave height, bubble law, a sound-speed profile from
the spec) and target classes (target strength, depth range), it prints for each class: the margin of the free-field
budget, the own depth that maximises the worst margin over the class's depths at the range of interest, the band of
own depths that hold it, the reach there and at reference depths, the target depths left uncovered, how many own
depths cover every target depth (pinging at two depths, or as the vehicle changes depth, fills the nulls), and the band
that holds two classes at once. For a static array "own depth" is the hanging depth. `--out PREFIX` writes PREFIX.json
and PREFIX.png.

margin = free-field budget (buildspec.budget: spherical spreading, absorption, ambient noise at the environment's wind,
the row's gain, the rows' detection gain) + target strength (the class's LOW end) + two-way propagation factor
(gpusonar.lloyd: surface image with roughness and bubble loss across the band; with a bottom, the layer's image series;
with a profile, Bellhop) - threshold (13 dB at the map). Source level: the budget takes the sonar file's
`source_level_db` as it stands, in an energy equation, exactly as docs/DESIGN.md Sec. 5 does; the simulator takes the
same number as a PEAK pressure (docs/HARDCODED_AUDIT.md P-5), so a simulated echo carries 3 dB less than this budget
assumes. The propagation factor is a ratio and has no such convention.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from gpusonar import buildspec as bs  # noqa: E402
from gpusonar import depthplan, lloyd  # noqa: E402
from gpusonar import target_classes as tc  # noqa: E402
from gpusonar.sim.bellhop import find_bellhop  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5fd6"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
SOURCE_LEVEL_CONVENTION = ("source levels are rms (the owner's decision; the usual datasheet convention): the budget puts the build's source level into an energy equation, which is the rms usage. "
                           "Until the simulator is changed to the same convention (HARDCODED_AUDIT P-5: it takes the number as a peak pressure) a simulated echo has 3 dB less power than the budget. The propagation factor is a ratio")
FACE = ["popoto_face_1x32", "popoto_face_2x32", "popoto_face_3x32"]
STATIC = ["popoto_testauv_1x32", "popoto_static_2x32", "popoto_static_3x32"]


def km(v):
    return f"{v / 1e3:.1f}"


def bands(b):
    return ", ".join(f"{lo:g}-{hi:g} m" if hi > lo else f"{lo:g} m" for lo, hi in b) or "none"


def ladder_environments() -> list:
    """The illustrative profiles of the stage-2 profile ladder (tools/popoto_stage2_study.py, docs/design/popoto_stage2.json) as
    environments: its water depth, over the popoto_12km scenario's sand."""
    try:
        import popoto_stage2_study as s2
    except Exception:  # noqa: BLE001
        return []
    out = []
    for i, (name, p) in enumerate(s2.PROFILES.items()):
        z, c = s2.true_profile(p)
        water = float(s2.WATER)
        keep = z < water
        z, c = np.r_[z[keep], water], np.r_[c[keep], np.interp(water, z, c)]
        out.append({"name": f"ladder_{i + 1}", "title": f"profile ladder: {name}; {water:g} m over sand, Bellhop", "water_depth_m": water,
                    "bottom": dict(bs.C_ASSUMED_BOTTOM), "profile": {"depth_m": [float(x) for x in z], "speed_m_s": [float(x) for x in c]},
                    "wind_speed_m_s": 5.0, "bottom_clearance_m": depthplan.DEFAULT_CLEARANCE,
                    "note": "the profile of tools/popoto_stage2_study.py PROFILES: illustrative, defined by parameters, not data"})
    return out


def text(P: dict) -> str:
    env = P["environment"]
    L = [f"Build {P['build']} ({P['rows']} row{'s' if P['rows'] > 1 else ''}); environment {env.get('name')}: {env.get('title', '')}",
         f"  {P['method']}; wind {P['wind_speed_m_s']:g} m/s, rms wave height {P['rms_wave_height_m']:.2f} m" + (f", bubble loss {env['bubble_model']}" if env.get("bubble_model") else "")
         + f"; own depths {P['own_depths_m'][0]:g}-{P['own_depths_m'][-1]:g} m; range of interest {km(P['range_of_interest_m'])} km; threshold {P['threshold_db']:g} dB",
         "", "| Class | TS used (dB) | Depths (m) | Free-field margin | Best own depth | Margin there | Own depths that hold it | Within 3 dB of best | Reach there (km) | Target depths not held there |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, c in P["classes"].items():
        if "skipped" in c:
            L.append(f"| {name} | | | {c['skipped']} | | | | | | |")
            continue
        r = c["recommended"]
        L.append(f"| {name} | {c['ts_db'][0]:+.0f} ({c['ts_status']}) | {c['depth_m'][0]:g}-{c['depth_m'][1]:g} | {c['free_field_margin_db']:+.1f} dB | {r['own_depth_m']:g} m | {r['margin_db']:+.1f} dB | "
                 f"{bands(r['held_band_m'])} | {bands(r['within_3db_band_m'])} | {km(r['reach_m'])}" + (" (gaps nearer)" if r["nearer_gaps"] else "") + f" | {bands(r['target_depths_not_held_m'])} |")
    L.append("")
    if P.get("platform_max_depth_m"):
        L.append(f"  Platform limit {P['platform_max_depth_m']:g} m" + (f", bottom clearance {P['bottom_clearance_m']:g} m" if P.get("bottom_clearance_m") is not None else "") + ": every best depth above is inside it"
                 + ("; where the best is the deepest searched, the limit (or the bottom) is what stops it: " + ", ".join(n for n, c in P["classes"].items() if "recommended" in c and c["recommended"]["at_the_deepest_searched"])
                    if any("recommended" in c and c["recommended"]["at_the_deepest_searched"] for c in P["classes"].values()) else "") + ".")
    L.append(f"  Rests on: {P['profile_rests_on']}.")
    for name, c in P["classes"].items():
        if "skipped" in c:
            continue
        L.append(f"  {name}: at own depth " + "; ".join(f"{z} m: {v['margin_db']:+.1f} dB, reach {km(v['reach_m'])} km" for z, v in c["at_own_depth"].items())
                 + f". With the high end of its target strength add {c['margin_high_ts_db']:.1f} dB.")
        steps = c["depth_diversity"]["steps"]
        if steps:
            L.append(f"    depth diversity (every target depth to {c['depth_diversity']['deepest_target_m']:g} m): " + ", then ".join(f"{s['own_depth_m']:g} m -> {100 * s['covered_share']:.0f}%" for s in steps))
        else:
            L.append("    depth diversity: no own depth holds any target depth at this range")
    for pair, v in P["together"].items():
        L.append(f"  both {pair}: own depths {bands(v['held_band_m'])}; best {v['best_own_depth_m']:g} m ({v['margin_of_the_weaker_db']:+.1f} dB for the weaker)")
    return "\n".join(L)


def figure(plans: list, path: Path, title: str):
    """One row per plan (an environment): the worst margin over each class's depths against range and own depth, and the margin
    at the range of interest against own depth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    names = [n for n in plans[0]["classes"]]
    fig, axes = plt.subplots(len(plans), len(names) + 1, figsize=(4.2 * (len(names) + 1), 3.3 * len(plans)), constrained_layout=True, squeeze=False)
    norm = TwoSlopeNorm(vmin=-30.0, vcenter=0.0, vmax=30.0)
    mesh = None
    for i, P in enumerate(plans):
        own, rr = np.asarray(P["own_depths_m"]), np.asarray(P["ranges_m"]) / 1e3
        for j, name in enumerate(names):
            ax, c = axes[i, j], P["classes"].get(name, {})
            if "margin_worst_db" not in c:
                ax.text(0.5, 0.5, c.get("skipped", "not planned"), ha="center", va="center", transform=ax.transAxes, color=MUTED, wrap=True)
                ax.set_axis_off()
                continue
            m = np.clip(np.asarray(c["margin_worst_db"]), -30.0, 30.0)
            mesh = ax.pcolormesh(rr, own, m, cmap="RdBu", norm=norm, shading="nearest")
            if m.min() < 0.0 < m.max():
                ax.contour(rr, own, m, levels=[0.0], colors=INK, linewidths=1.0)
            ax.plot([P["range_of_interest_m"] / 1e3], [c["recommended"]["own_depth_m"]], "o", color=YELLOW, mec=INK, ms=7)
            ax.invert_yaxis()
            ax.set_title(f"{name}: {c['ts_db'][0]:+.0f} dB at {c['depth_m'][0]:g}-{c['depth_m'][1]:g} m", loc="left", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{P['environment'].get('name')}\nown depth (m)")
            if i == len(plans) - 1:
                ax.set_xlabel("range (km)")
        ax = axes[i, -1]
        for name, col in zip(names, (BLUE, ORANGE, AQUA, VIOLET, YELLOW)):
            c = P["classes"].get(name, {})
            if "margin_at_range_worst_db" in c:
                ax.plot(np.clip(c["margin_at_range_worst_db"], -40, 40), own, color=col, lw=1.8, label=name)
        ax.axvline(0.0, color=INK, lw=0.8)
        ax.invert_yaxis()
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_title(f"margin at {km(P['range_of_interest_m'])} km against own depth", loc="left", fontsize=9)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
        if i == len(plans) - 1:
            ax.set_xlabel("margin (dB), worst over the class's depths")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes[:, :-1], shrink=0.6, label="margin (dB): blue held, red not; line: 0 dB; dot: best own depth")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def slim(P: dict) -> dict:
    """A plan without its two-dimensional grids (for the JSON kept in docs/)."""
    P = copy.deepcopy(P)
    for c in P["classes"].values():
        c.pop("margin_worst_db", None)
    return P


def physics_tables(d: dict, classes: list, r: float = 12000.0) -> dict:
    """The numbers the plan text quotes about the model itself: the first lobe, what the band fills, what roughness and bubbles do."""
    f_lo, f_hi = d["band_hz"]
    fc, c = d["centre_hz"], d["sound_speed"]
    ship = next((x for x in classes if x["name"] == "ship"), classes[0])
    z_ship = 0.5 * (ship["depth_m"][0] + ship["depth_m"][1])
    lobe = lloyd.first_lobe_own_depth(fc, z_ship, r, c)
    own = [10.0, 30.0, round(lobe["zero_db_m"]), 150.0, round(lobe["peak_m"]), 300.0, round(lobe["first_null_m"]), 600.0]
    surfaces = {"flat": lloyd.FLAT, "5 m/s wind (wind-law height)": lloyd.Surface(wind_speed=5.0), "9.8 m/s wind (sea state 5, wind-law height)": lloyd.Surface(wind_speed=9.8),
                "9.8 m/s wind with apl_1994 bubble loss": lloyd.Surface(wind_speed=9.8, bubble_model="apl_1994")}
    rows = []
    for z in own:
        row = {"own_depth_m": float(z)}
        for label, s in surfaces.items():
            t = lloyd.two_way(f_lo, f_hi, z, z_ship, r, s, n_f=257)
            row[label] = {k: float(t[k]) for k in ("narrowband_db", "energy_db", "peak_db")}
        rows.append(row)
    # where the band stops mattering and where roughness starts to
    grazing = np.arcsin((np.array(own) + z_ship) / np.hypot(r, np.array(own) + z_ship))
    rough = [{"own_depth_m": float(z), "grazing_deg": float(np.degrees(g)),
              **{f"coherent_reflection_rms_{h:g}m": float(lloyd.Surface(rms_height=h).coherent(g, fc)) for h in (0.13, 0.5, 1.0)},
              "bubble_loss_db_9p8_m_s": float(lloyd.Surface(wind_speed=9.8, bubble_model="apl_1994").bubble_loss_db(g, fc))} for z, g in zip(own, grazing)]
    deep = []
    for zt in (4.0, 30.0, 100.0, 300.0):
        zo = np.arange(5.0, 301.0, 5.0)
        t = lloyd.two_way(f_lo, f_hi, zo, zt, r, lloyd.Surface(wind_speed=5.0), n_f=257)
        deep.append({"target_depth_m": zt, "peak_db_lowest": float(t["peak_db"].min()), "peak_db_highest": float(t["peak_db"].max()), "peak_db_median": float(np.median(t["peak_db"])),
                     "narrowband_db_lowest": float(t["narrowband_db"].min()), "own_depths_m": [5.0, 300.0]})
    lam = c / fc
    return {"range_m": r, "ship_depth_m": z_ship, "wavelength_m": lam, "first_lobe": lobe, "by_own_depth": rows, "roughness": rough, "by_target_depth": deep,
            "band_fills_beyond_depth_product_m2": float(lam * r * fc / (2.0 * (f_hi - f_lo))),
            "wind_law_rms_height_m": {"5": lloyd.Surface(wind_speed=5.0).height(), "9.8": lloyd.Surface(wind_speed=9.8).height()}}


def run(spec_or_d, envs, classes, range_m, threshold, statistic, cache, log=print, keep_grid=True, bellhop=True):
    d = bs.derive(spec_or_d) if "projector" in spec_or_d else spec_or_d
    out = {}
    for env in envs:
        if env.get("profile") is not None and (not bellhop or find_bellhop() is None):
            out[env["name"]] = {"skipped": "a sound-speed profile needs Bellhop (GPUSONAR_BELLHOP)", "environment": env}
            log(f"{env['name']}: skipped (needs Bellhop)")
            continue
        try:
            P = depthplan.plan(d, env, classes, range_of_interest=range_m, threshold_db=threshold, statistic=statistic, cache=cache, keep_grid=keep_grid)
        except ValueError as exc:
            out[env["name"]] = {"skipped": str(exc), "environment": env}
            log(f"{env['name']}: skipped ({exc})")
            continue
        for name, c in P["classes"].items():
            if "skipped" in c:
                continue
            r = c["recommended"]
            ts = c["ts_db"][0]
            e = depthplan.elevation_for(d, r["margin_db"] + threshold, P["range_of_interest_m"])
            if e:
                c["elevation_at_recommended"] = e
        out[env["name"]] = P
    return out


def shipped(args) -> int:
    dest = ROOT / "docs" / "design"
    R = {"what": "tools/popoto_depth_planner.py --shipped", "source_level_convention": SOURCE_LEVEL_CONVENTION, "bellhop": find_bellhop() is not None, "threshold_db": args.threshold,
         "statistic": args.statistic, "builds": {}, "not_run": []}
    spec3 = bs.load("popoto_face_3x32")
    classes = bs.target_classes_of(spec3, range_m=12000.0)
    R["classes"] = [{k: v for k, v in c.items()} for c in classes]
    R["cylinders"] = {f"{d_:g} m x {l_:g} m": {"by_aspect": [{"aspect_deg": a, "ts_db": float(v), "ts_far_field_db": float(w)} for a, v, w in zip(
        ASPECTS, tc.cylinder_ts_db(l_, d_, ASPECTS, *spec3["projector"]["band_hz"], 12000.0), tc.cylinder_ts_db(l_, d_, ASPECTS, *spec3["projector"]["band_hz"], None))],
        **tc.cylinder_closed_forms(l_, d_, 5000.0, 12000.0)} for d_, l_ in ((2.0, 10.0), (4.0, 30.0), (7.0, 60.0), (10.0, 100.0))}
    R["physics"] = physics_tables(bs.derive(spec3), classes)
    envs = bs.environments_of(spec3) + ladder_environments()
    R["environments"] = envs
    cache, figs = {}, []
    for build in FACE:
        spec = bs.load(build)
        plans = run(spec, envs, classes, 12000.0, args.threshold, args.statistic, cache, bellhop=not args.no_bellhop)
        for name, P in plans.items():
            if "skipped" not in P:
                print(text(P) + "\n", flush=True)
            else:
                R["not_run"].append(f"{build} in {name}: {P['skipped']}")
        if build == FACE[-1]:
            figs = [plans[n] for n in ("deep_no_bottom_return", "shelf_100m_sand", "slope_1000m_silt_profile", "ladder_2") if n in plans and "skipped" not in plans[n]]
        R["builds"][build] = {"range_of_interest_m": 12000.0, "plans": {n: slim(P) for n, P in plans.items()}}
        (dest / "popoto_depth_planner.json").write_text(json.dumps(R, indent=1, default=float) + "\n")
    if figs and not args.no_figure:
        figure(figs, dest / "popoto_depth_planner.png", "popoto_face_3x32: detection margin (dB over 13 dB at the map), worst over each class's depths, against range and own depth")
    # static and test-AUV builds: the hanging (or running) depth, at the end of their own receive window, where there is no profile to trace
    for build in STATIC:
        spec = bs.load(build)
        d = bs.derive(spec)
        r = float(d["window_m"][1])
        mine = [e for e in bs.environments_of(spec) if e.get("profile") is None]
        plans = run(spec, mine, bs.target_classes_of(spec, range_m=r), r, args.threshold, args.statistic, {}, keep_grid=False)
        for P in plans.values():
            if "skipped" not in P:
                print(text(P) + "\n", flush=True)
        R["builds"][build] = {"range_of_interest_m": r, "plans": plans, "spec_depth_m": spec["platform"].get("depth_m"), "trial_own_depth_m": spec["trial"].get("own_depth_m")}
    (dest / "popoto_depth_planner.json").write_text(json.dumps(R, indent=1, default=float) + "\n")
    return 0


ASPECTS = [0.0, 1.0, 2.0, 5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 90.0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", help="a build spec: a name under configs/builds or a path")
    ap.add_argument("--sonar", help="instead of --build: a sonar config name")
    ap.add_argument("--deploy", help="with --sonar: its deployment file's name")
    ap.add_argument("--environment", action="append", help="name of one of the spec's environments (repeatable; default: all of them)")
    ap.add_argument("--water-depth", type=float, help="m; with --bottom: an environment given here instead of the spec's")
    ap.add_argument("--bottom", help="a sediment name of configs/sediments.yaml")
    ap.add_argument("--wind", type=float, help="m/s")
    ap.add_argument("--sea-state", type=int)
    ap.add_argument("--rms-height", type=float, help="rms wave height, m (a quarter of the significant height)")
    ap.add_argument("--bubble-model", help="apl_1994, or leave out for none")
    ap.add_argument("--own-depth", type=float, nargs=2, metavar=("SHALLOWEST", "DEEPEST"), help="operating depth limits, m")
    ap.add_argument("--target", action="append", help="name of one of the spec's target classes (repeatable; default: all)")
    ap.add_argument("--ts", type=float, help="with --target-depth: a target given here (dB re 1 m)")
    ap.add_argument("--target-depth", type=float, nargs=2, metavar=("SHALLOWEST", "DEEPEST"))
    ap.add_argument("--range", type=float, default=12000.0, help="range of interest, m")
    ap.add_argument("--threshold", type=float, default=13.0, help="SNR at the map for detection, dB")
    ap.add_argument("--statistic", default="peak_db", choices=("peak_db", "energy_db", "narrowband_db"), help="which two-way figure of gpusonar.lloyd the margin uses")
    ap.add_argument("--out", help="prefix: writes PREFIX.json and PREFIX.png")
    ap.add_argument("--shipped", action="store_true", help="the shipped builds, into docs/design/")
    ap.add_argument("--no-bellhop", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    if args.shipped:
        return shipped(args)
    if args.build:
        spec = bs.load(args.build)
        issues = bs.validate(spec)
        if any(i.level == "error" for i in issues):
            print("\n".join(str(i) for i in issues if i.level == "error"))
            return 2
        d = bs.derive(spec)
        classes, envs = bs.target_classes_of(spec, range_m=args.range), bs.environments_of(spec)
    elif args.sonar:
        d = bs.describe_shipped(args.sonar, args.deploy)
        classes, envs = [], []
    else:
        ap.error("give --build, --sonar or --shipped")
    inline = {k: v for k, v in (("water_depth_m", args.water_depth), ("bottom", args.bottom), ("wind_speed_m_s", args.wind), ("sea_state", args.sea_state),
                                ("rms_wave_height_m", args.rms_height), ("bubble_model", args.bubble_model), ("own_depth_limits_m", args.own_depth)) if v is not None}
    if inline and not args.environment:
        envs = [dict({"name": "given", "title": "the environment given on the command line"}, **inline)]
    elif args.environment:
        known = {e["name"]: e for e in envs}
        missing = [n for n in args.environment if n not in known]
        if missing:
            ap.error(f"no environment {missing} in the spec; it has {sorted(known)}")
        envs = [dict(known[n], **inline) for n in args.environment]
    problems = depthplan.environment_problems(envs)
    if args.ts is not None:
        if args.target_depth is None:
            ap.error("--ts needs --target-depth")
        given = [{"name": "given", "target_strength_db": args.ts, "depth_m": list(args.target_depth), "status": "assumed"}]
        problems += tc.problems(given)
        classes = tc.resolve(given, *d["band_hz"], args.range) if not problems else []
    elif args.target:
        classes = [c for c in classes if c["name"] in args.target]
    if problems or not envs or not classes:
        print("\n".join(f"error: {w}: {m}" for w, m in problems) or "error: no environment or no target class: give them in the spec (targets:, environments:) or on the command line")
        return 2
    plans = run(d, envs, classes, args.range, args.threshold, args.statistic, {}, bellhop=not args.no_bellhop)
    for P in plans.values():
        print((text(P) if "skipped" not in P else f"{P['environment']['name']}: {P['skipped']}") + "\n")
    print("Source level: " + SOURCE_LEVEL_CONVENTION + ".")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps({"source_level_convention": SOURCE_LEVEL_CONVENTION, "classes": classes, "plans": {n: slim(P) for n, P in plans.items()}}, indent=1, default=float) + "\n")
        good = [P for P in plans.values() if "skipped" not in P]
        if good and not args.no_figure:
            figure(good, out.with_suffix(".png"), f"{good[0]['build']}: detection margin (dB over {args.threshold:g} dB), worst over each class's depths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
