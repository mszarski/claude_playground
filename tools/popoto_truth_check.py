#!/usr/bin/env python3
"""Score the live chain's track classes on the popoto_12km scenario against the simulated truth.

    tools/popoto_truth_check.py recording.h5 [json of extra chain settings] [--scenario popoto_12km]

The recording is the scenario's pings (run_scenario("popoto_12km").records written with RecordWriter); they go
through the C++ chain of configs/deploy/popoto_vehicle.yaml with `doa: map` and `doa: sbl`, and the last ping's tracks
are scored. Truth: the ship's direct echo is at its geometric range (the scenario's `ship` points moved by their
velocity to the last ping); a track within 15 m of it should be a mover, a track further out is a later (bottom- and
surface-bounce) arrival and should be multipath. The source of the track-class table in docs/RESULTS.md. Writes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar.config import load_scenario_config, load_yaml  # noqa: E402


def ship_at(cfg: dict, ping: int) -> np.ndarray:
    """The scenario's `ship` points, [N, 3] in metres, where they are at ping `ping` (0 = the first)."""
    ship = next(item for item in cfg["scene"] if item.get("label") == "ship")
    pos = np.asarray(ship["positions"], dtype=float)
    vel = np.asarray(ship.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    return pos + vel * float(cfg["pings"]["interval"]) * ping


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="the scenario's pings, an .h5 written with RecordWriter")
    ap.add_argument("extra", nargs="?", default="{}", help="JSON of chain settings added to the deployment file's (default: none)")
    ap.add_argument("--scenario", default="popoto_12km", help="the scenario the recording was simulated from (its ship is the truth)")
    args = ap.parse_args(argv)

    import gpusonar as gs
    from gpusonar.live import cpp_chain_options
    rd = gs.RecordReader(args.recording)
    recs = [rd.read(i) for i in range(len(rd))]
    chain = load_yaml(ROOT / "configs" / "deploy" / "popoto_vehicle.yaml")["chain"]
    extra = json.loads(args.extra)
    cfg = load_scenario_config(args.scenario)
    for doa in ("map", "sbl"):
        cpp = gs.core.LiveChainCpp(cpp_chain_options(dict(chain, doa=doa, **extra)))
        reps = [json.loads(cpp.process(r.copy())) for r in recs]
        ship = ship_at(cfg, len(reps) - 1)
        own = np.array(reps[-1]["ownship_xyz"][:2])
        dx, dy = ship[:, 0] - own[0], ship[:, 1] - own[1]
        r_true = np.hypot(dx, dy)
        bearing = np.degrees(np.arctan2(dx, dy))
        print(f"--- doa={doa}: ship direct range {r_true.min():.1f}..{r_true.max():.1f} m, bearing span {bearing.min():.2f}..{bearing.max():.2f} deg")
        by_id = {t["id"]: t for t in reps[-1]["tracks"]}
        ok = bad = 0
        for t in sorted(reps[-1]["tracks"], key=lambda t: t["range_m"]):
            excess = t["range_m"] - r_true.mean()
            truth = "direct" if abs(excess) <= 15.0 else ("multipath" if excess > 15.0 else "none")
            good = (truth == "direct" and t["kind"] == "mover") or (truth == "multipath" and t["kind"] == "multipath")
            ok += good
            bad += not good
            par = by_id.get(t["parent"]) if t["parent"] is not None else None
            print(f"  id {t['id']:2d} excess {excess:+6.1f} m  x-rel {t['x'] - own[0]:+7.1f}  v=({t['vx']:+.1f},{t['vy']:+.1f})  {t['kind']:9s} parent {t['parent']} "
                  f"(parent excess {'' if par is None else round(par['range_m'] - r_true.mean(), 1)})  truth {truth} {'ok' if good else 'WRONG'}")
        print(f"  {ok} right, {bad} wrong")
    return 0


if __name__ == "__main__":
    sys.exit(main())
