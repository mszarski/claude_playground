#!/usr/bin/env python3
"""Where to hang (or run) the array relative to a sound-speed maximum: the depth planner in fjord-type water.

    GPUSONAR_BELLHOP=.../bellhop.exe tools/fjord_depth_plan.py [--out docs/figures/fjord_depth_plan.md]

`tools/popoto_depth_planner.py` takes its waters from a build spec's `environments:`; this runs the same planner
(gpusonar.depthplan, Bellhop for the profile) for the three builds trialled first in two waters given here:

    fjord_profile   the ILLUSTRATIVE profile of sim/basin.py `stratified_profile` (a maximum of 1495 m/s at 19 m
                    under a fresher surface layer; of the shape of the casts of one site, PHYSICS_NOTES Sec. 21.6)
                    over a flat seabed (the planner's water is flat: no basins or sills), 70 m, the suite's default
                    half-space (the seabed there is not named), a calm night (rms height 1.4 cm, wind 1.9 m/s)
    same_isovelocity the same water without the profile (1492 m/s)

against one target class per build, of the trial scenario's target strength and any depth from 5 to 40 m, at the
trial range. Every number of the water, the target and the limits is a placeholder.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

CASES = [("popoto_testauv_1x32", 500.0, 10.0), ("popoto_static_3x32", 150.0, 10.0), ("popoto_static_2x32", 2600.0, 15.0)]


def waters() -> list:
    from gpusonar.sim import basin
    z, c = basin.stratified_profile(70.0, step=5.0)
    common = {"water_depth_m": 70.0, "bottom": {"speed": 1600.0, "density": 1.8, "attenuation": 0.5}, "wind_speed_m_s": 1.9, "rms_wave_height_m": 0.014,
              "own_depth_limits_m": [5.0, 60.0], "bottom_clearance_m": 10.0, "illustrative": True}
    return [dict(common, name="fjord_profile", title="70 m, sound-speed maximum at 19 m (illustrative), calm", profile={"depth_m": [float(v) for v in z], "speed_m_s": [float(v) for v in c]}),
            dict(common, name="same_isovelocity", title="the same water at one sound speed", sound_speed_m_s=1492.0)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--builds", default=",".join(c[0] for c in CASES))
    args = ap.parse_args(argv)
    import popoto_depth_planner as planner
    from gpusonar import buildspec as bs
    from gpusonar import depthplan
    from gpusonar import target_classes as tc
    lines = ["# Own depth against a sound-speed maximum: the depth planner in fjord-type water", "",
             "Made by `tools/fjord_depth_plan.py` (the planner of `tools/popoto_depth_planner.py` with the waters named below; flat seabed). ILLUSTRATIVE water; placeholder targets.", ""]
    for build, range_m, ts in CASES:
        if build not in args.builds.split(","):
            continue
        d = bs.derive(bs.load(build))
        classes = tc.resolve([{"name": "trial_target", "target_strength_db": ts, "depth_m": [5.0, 40.0], "status": "assumed"}], *d["band_hz"], range_m)
        envs = waters()
        problems = depthplan.environment_problems(envs)
        if problems:
            print(problems)
            return 2
        ranges = np.unique(np.r_[np.linspace(0.5 * range_m, 1.5 * range_m, 5), range_m])
        plans = {}
        for env in envs:
            try:
                plans[env["name"]] = depthplan.plan(d, env, classes, range_of_interest=range_m, ranges=ranges, statistic="peak_db", cache={}, keep_grid=True)
            except Exception as err:                                           # noqa: BLE001
                plans[env["name"]] = {"skipped": f"{type(err).__name__}: {err}", "environment": env}
        lines += [f"## {build}, target of {ts:g} dB between 5 and 40 m at {range_m:g} m", ""]
        for name, P in plans.items():
            lines += ["```", (planner.text(P) if "skipped" not in P else f"{name}: {P['skipped']}"), "```", ""]
            if "skipped" in P:
                continue
            c = P["classes"]["trial_target"]
            own, worst = np.asarray(P.get("own_depths_m", [])), c.get("factor_at_range_worst_db")
            if worst is not None and len(own):
                lines += ["Two-way propagation factor re free field at the range, worst over the target's depths, by own depth, dB: "
                          + "; ".join(f"{z:.0f} m: {w:+.1f}" for z, w in zip(own, worst)), ""]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
