#!/usr/bin/env python3
"""The coverage picture of a build in one water: for each target class, the probability of detecting it against range
and target depth from one own depth, the greatest range at which the worst of its depths is still detected with the
stated probability, where the direct path ends, the blind bands, and the own depth that would reach furthest.

The same numbers the chain prints as `coverage` in its report line (libs/live coverage.hpp; python/gpusonar/coverage.py):
loss from rays through the profile (the chain's ray code), levels from the build's link budget (buildspec.budget),
detection from the chain's own detector (live.detection_probability), not from a typed threshold.

  --build NAME          a build spec (configs/builds): sonar, link budget, target classes, environments
  --environment NAME    one of the build's environments (default: its first); its water depth, bottom, wind, wave height,
                        bubble law and profile are used
  --own-depth M         the array's depth (default 100)
  --profile SOURCE      what the vehicle is taken to know of the water:
                          environment   the environment's profile (isovelocity when it has none): a supplied cast
                          prior         nothing: isovelocity at the environment's sound speed
                          built:LO-HI   a profile built from the vehicle's own samples between LO and HI m of the
                                        environment's profile (--sensor-sigma), continued with no gradient outside
  --ensemble G1,G2,..   with --extent M[,M..]: the profile ensemble (elevation.profile.ensemble) for the unsampled water
  --pd P                the detection probability the ranges are stated for (default 0.5)
  --json PATH, --figure PATH, --chain-config   write the result, draw it, or print the `chain.coverage` block to paste

Usage: tools/coverage_map.py --build popoto_face_1x32 --environment slope_1000m_silt_profile --own-depth 225 --figure out.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import gpusonar as gs  # noqa: E402
from gpusonar import buildspec as bs  # noqa: E402
from gpusonar import coverage, depthplan  # noqa: E402
from gpusonar.sim import make_pose  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

core = gs.core
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"      # the planner's figure style (tools/popoto_depth_planner.py)


def water_of(env: dict, own_depth: float, source: str, sensor_sigma: float, gradient_sigma: float, ensemble=None, extent=None) -> dict:
    """The profile handed to the rays, as the chain's ladder would hold it (core.live_water_profile)."""
    c0 = float(env.get("sound_speed_m_s", 1500.0))
    bottom = float(env["water_depth_m"]) if env.get("water_depth_m") is not None else float("nan")
    prof = env.get("profile")
    depth = [float(v) for v in prof["depth_m"]] if prof else [0.0, 5000.0]
    speed = [float(v) for v in prof["speed_m_s"]] if prof else [c0, c0]
    c_own = float(np.interp(own_depth, depth, speed))
    o = {"sample_sigma": sensor_sigma, "age_sigma_per_hour": 0.0, "distance_sigma_per_km": 0.0}
    if ensemble:
        o.update(ensemble=True, ensemble_gradients=[float(v) for v in ensemble], ensemble_extents=[float(v) for v in extent])
    samples, supplied = (), ((), ())
    if source == "environment":
        o.update(ladder=["supplied"], supplied_sigma=sensor_sigma)
        supplied = (depth, speed)
    elif source == "prior":
        o.update(ladder=["local"])
    elif source.startswith("built:"):
        lo, hi = (float(v) for v in source[6:].split("-"))
        o.update(ladder=["built"])
        samples = np.arange(lo, hi + 0.01, 1.0)
    else:
        raise SystemExit(f"--profile: {source!r} (environment, prior or built:LO-HI)")
    lad = core.SoundSpeedLadder(o)
    for z in samples:
        lad.add(0, 0.0, 0.0, float(z), float(np.interp(z, depth, speed)))
    got = lad.evaluate(0, 0.0, 0.0, own_depth, c_own, True, bottom, list(supplied[0]), list(supplied[1]), 0.0, gradient_sigma)
    return core.live_water_profile(o, got, [], [], False, 0.0, gradient_sigma, own_depth, c_own, bottom)


def chain_block(spec: dict, env: dict, pd: float) -> dict:
    """`chain.coverage` for a build in an environment: classes, budget, surface and bottom."""
    d = bs.derive(spec)
    surf = depthplan.surface_of(env)
    block = {"classes": coverage.classes_of(bs.target_classes_of(spec, range_m=float(d["window_m"][1]))), "budget": coverage.budget_of(d, depthplan.wind_of(env)), "pd": pd,
             "surface": {"rms_height_m": float(surf.height())}}
    if surf.bubble_model:
        block["surface"].update(bubble_model=surf.bubble_model, wind_speed_m_s=float(surf.wind_speed), frequency_hz=float(d["centre_hz"]))
    if env.get("bottom") is not None and env.get("water_depth_m") is not None:
        block["bottom"] = {"sediment": env["bottom"]} if isinstance(env["bottom"], str) else dict(env["bottom"])
        block["water_depth_m"] = float(env["water_depth_m"])        # a chain needs both (gpusonar.coverage.chain_options); a water without them is for this tool only
    limit = (d.get("platform") or {}).get("max_depth_m")
    lim = env.get("own_depth_limits_m") or [depthplan.DEFAULT_LIMITS[0], float(limit or depthplan.DEFAULT_LIMITS[1])]
    block["own_depth_limits_m"] = [float(lim[0]), float(min(lim[1], limit) if limit else lim[1])]
    block["bottom_clearance_m"] = float(env.get("bottom_clearance_m", depthplan.DEFAULT_CLEARANCE))
    # the site's boundary scattering by the one rule (gpusonar.environment.boundary_scattering), as coverage.of_build writes it
    block["reverberation"] = coverage.site_reverberation(env, float(d["centre_hz"]), f"environment {env.get('name')!r}")
    return block


def run(build: str, environment: str | None, own_depth: float, source: str = "environment", pd: float = 0.5, sensor_sigma: float = 0.1,
        gradient_sigma: float = 0.05, ensemble=None, extent=None, n_depths: int = 40) -> dict:
    spec = bs.load(build)
    d = bs.derive(spec)
    envs = bs.environments_of(spec)
    env = envs[0] if environment is None else next((e for e in envs if e["name"] == environment), None)
    if env is None:
        raise SystemExit(f"--environment: {environment!r} is not one of {[e['name'] for e in envs]}")
    block = chain_block(spec, env, pd)
    record = SonarModel.from_config(bs.sonar_config(spec)).make_record(make_pose())
    sonar = {"f_centre": float(d["centre_hz"]), "bandwidth": float(d["band_hz"][1] - d["band_hz"][0]), "pulse": float(d["pulse_s"]),
             "beam_integral": float(gs.core.coverage_beam_integral_of(record))}       # the record's, as the chain has it
    chain_cfg = bs.chain_config(spec)
    table = coverage.pd_table(record, chain_cfg)
    water = water_of(env, own_depth, source, sensor_sigma, gradient_sigma, ensemble, extent)
    water_depth = env.get("water_depth_m")
    below = float(d["projector_m"][2])
    summary = coverage.summary(water, water_depth, own_depth, block, sonar, table, d["window_m"], below)
    # the grid behind it, every target depth of the water
    ranges = np.linspace(max(d["window_m"][0], 50.0), d["window_m"][1], 96)
    deepest = float(water_depth) - 1.0 if water_depth is not None else max(max(c["depth_m"][1] for c in block["classes"]), own_depth) + 50.0
    depths = np.linspace(1.0, deepest, n_depths)
    f = coverage.field(water, water_depth, own_depth, depths, ranges, block, sonar, below)
    grid = {"ranges_m": ranges.tolist(), "target_depths_m": depths.tolist(), "factor_db": np.where(np.isfinite(f["factor_db"]), f["factor_db"], -200.0).tolist(),
            "direct_path": f["direct"].astype(int).tolist(), "pd": {}}
    for c in block["classes"]:
        grid["pd"][c["name"]] = np.interp(f["snr_0db_target_db"] + c["ts_db"], table[0], table[1]).tolist()
    return {"build": build, "environment": env["name"], "own_depth_m": own_depth, "profile_source": source, "profile_rung": water["rung"],
            "ensemble_members": len(water["ensemble"]), "sonar": sonar, "window_m": [float(v) for v in d["window_m"]], "chain_coverage": block,
            "detection": {"what": "live.detection_probability for the build's chain configuration; one row's link-budget SNR at which it is the stated pd",
                          "snr_at_pd_db": float(np.interp(pd, table[1], table[0]))},
            "summary": summary, "grid": grid}


def text(R: dict) -> str:
    s = R["summary"]
    L = [f"{R['build']} in {R['environment']}, array at {R['own_depth_m']:g} m; profile: {R['profile_source']} (rung {R['profile_rung']})"
         + (f", ensemble of {R['ensemble_members'] + 1}" if R["ensemble_members"] else ""),
         f"detection: probability {s.get('pd')} needs {R['detection']['snr_at_pd_db']:.1f} dB of one row's link-budget SNR (the chain's detector); refresh took {s['seconds'] * 1e3:.0f} ms"]
    if "error" in s:
        return "\n".join(L + ["  " + s["error"]])
    L.append(f"  {'class':18s} {'max range':>10s} {'direct to':>10s}  blind bands [range m, depth m]; nulls; ensemble; at the recommended depth")
    for c in s["classes"]:
        if c.get("max_range_m") is None:
            L.append(f"  {c['name']:18s} {c.get('note', '')}")
            continue
        blind = "; ".join(f"{b[0]}-{b[1]} m, {b[2]}-{b[3]} m" for b in c.get("blind_m", [])) or "none"
        L.append(f"  {c['name']:18s} {c['max_range_m']:8d} m {c['direct_to_m']:8d} m  {blind}; nulls {c.get('nulls_m', [])}"
                 + (f"; ensemble {c['max_range_ensemble_m']}" if "max_range_ensemble_m" in c else "") + f"; {c.get('recommended_max_range_m')} m")
    if "recommend" in s:
        L.append(f"  recommended own depth {s['recommend']['own_depth_m']:g} m: the least of the classes' ranges changes by {s['recommend']['gain_m']:+d} m")
    L.append("  pd at " + ", ".join(str(r) for r in s["ranges_m"]) + " m:")
    for c in s["classes"]:
        if "pd" in c:
            L.append(f"    {c['name']:18s} " + " ".join(f"{v:4.2f}" for v in c["pd"]))
    return "\n".join(L)


def figure(R: dict, path: Path) -> None:
    """One panel per class: detection probability against range and target depth (one hue, light to dark), the class's
    depths marked, the greatest range drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    g, classes = R["grid"], [c for c in R["summary"].get("classes", []) if c.get("max_range_m") is not None]
    fig, axes = plt.subplots(1, max(len(classes), 1), figsize=(4.4 * max(len(classes), 1), 3.4), constrained_layout=True, squeeze=False)
    r_km, z = np.array(g["ranges_m"]) / 1000.0, np.array(g["target_depths_m"])
    mesh = None
    for ax, c in zip(axes[0], classes):
        mesh = ax.pcolormesh(r_km, z, np.array(g["pd"][c["name"]]), cmap="Blues", vmin=0.0, vmax=1.0, shading="nearest", rasterized=True)
        lo, hi = next(k["depth_m"] for k in R["chain_coverage"]["classes"] if k["name"] == c["name"])
        ax.plot([r_km[0], r_km[-1]], [lo, lo], color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.plot([r_km[0], r_km[-1]], [hi, hi], color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.axvline(c["max_range_m"] / 1000.0, color=INK, lw=1.2)
        ax.text(c["max_range_m"] / 1000.0, z[0], f" {c['max_range_m'] / 1000.0:.1f} km", va="top", ha="right" if c["max_range_m"] > 0.7 * g["ranges_m"][-1] else "left", fontsize=8)
        ax.plot([r_km[0]], [R["own_depth_m"]], marker=">", color=INK, ms=6, clip_on=False)
        ax.set_ylim(z[-1], 0.0)
        ax.set_title(f"{c['name']}: depths {lo:g}-{hi:g} m (dashed)", loc="left", fontsize=9)
        ax.set_xlabel("range, km")
    axes[0][0].set_ylabel("target depth, m")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes[0].tolist(), label="detection probability, one ping", shrink=0.85)
    fig.suptitle(f"{R['build']} in {R['environment']}: array at {R['own_depth_m']:g} m (marker), profile {R['profile_rung']}; line: greatest range at pd {R['summary'].get('pd')}",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--environment")
    ap.add_argument("--own-depth", type=float, default=100.0)
    ap.add_argument("--profile", default="environment")
    ap.add_argument("--pd", type=float, default=0.5)
    ap.add_argument("--sensor-sigma", type=float, default=0.1, help="m/s: the sound-speed sensor's accuracy taken for a built or supplied profile (illustrative)")
    ap.add_argument("--gradient-sigma", type=float, default=0.05, help="1/s: the prior outside what is known (the deployment files', unconfirmed)")
    ap.add_argument("--ensemble", help="gradients of the unsampled water, 1/s, comma-separated")
    ap.add_argument("--extent", help="m, one or more comma-separated: how far such a gradient may run")
    ap.add_argument("--json")
    ap.add_argument("--figure")
    ap.add_argument("--chain-config", action="store_true")
    a = ap.parse_args(argv)
    if bool(a.ensemble) != bool(a.extent):
        ap.error("--ensemble and --extent go together")
    R = run(a.build, a.environment, a.own_depth, a.profile, a.pd, a.sensor_sigma, a.gradient_sigma,
            [float(v) for v in a.ensemble.split(",")] if a.ensemble else None, [float(v) for v in a.extent.split(",")] if a.extent else None)
    print(text(R))
    if a.chain_config:
        print(json.dumps({"coverage": R["chain_coverage"]}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(R, indent=1) + "\n")
    if a.figure:
        figure(R, Path(a.figure))
    return 0


if __name__ == "__main__":
    sys.exit(main())
