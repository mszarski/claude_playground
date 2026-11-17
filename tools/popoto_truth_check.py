#!/usr/bin/env python3
"""Score the live chain's track classes on the popoto_12km scenario against the simulated truth.

    tools/popoto_truth_check.py recording.h5 [json of extra chain settings]

The recording is the scenario's six pings (run_scenario("popoto_12km").records written with RecordWriter).
Truth: the ship's direct echo is at its geometric range; a track within 15 m of it should be a mover,
a track further out is a later (bottom- and surface-bounce) arrival and should be multipath."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import json, sys, numpy as np, gpusonar as gs
from gpusonar.config import load_yaml
from gpusonar.live import cpp_chain_options
core = gs.core
rd = gs.RecordReader(sys.argv[1]); recs = [rd.read(i) for i in range(len(rd))]
chain = load_yaml("configs/deploy/popoto_vehicle.yaml")["chain"]
extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
for doa in ("map", "sbl"):
    cpp = core.LiveChainCpp(cpp_chain_options(dict(chain, doa=doa, **extra)))
    reps = [json.loads(cpp.process(r.copy())) for r in recs]
    k = len(reps) - 1
    own = np.array(reps[-1]["ownship_xyz"][:2])
    ship_x = np.array([740.0, 770.0, 800.0, 830.0, 860.0]) - 6.0 * 19.9 * k
    r_true = np.hypot(ship_x - own[0], 12000.0 - own[1])
    print(f"--- doa={doa}: ship direct range {r_true.min():.1f}..{r_true.max():.1f} m, bearing span {np.degrees(np.arctan2(ship_x - own[0], 12000)).min():.2f}..{np.degrees(np.arctan2(ship_x - own[0], 12000)).max():.2f} deg")
    by_id = {t["id"]: t for t in reps[-1]["tracks"]}
    ok = bad = 0
    for t in sorted(reps[-1]["tracks"], key=lambda t: t["range_m"]):
        excess = t["range_m"] - r_true.mean()
        truth = "direct" if abs(excess) <= 15.0 else ("multipath" if excess > 15.0 else "none")
        good = (truth == "direct" and t["kind"] == "mover") or (truth == "multipath" and t["kind"] == "multipath")
        ok += good; bad += (not good)
        par = by_id.get(t["parent"]) if t["parent"] is not None else None
        print(f"  id {t['id']:2d} excess {excess:+6.1f} m  x-rel {t['x']-own[0]:+7.1f}  v=({t['vx']:+.1f},{t['vy']:+.1f})  {t['kind']:9s} parent {t['parent']} (parent excess {'' if par is None else round(par['range_m']-r_true.mean(),1)})  truth {truth} {'ok' if good else 'WRONG'}")
    print(f"  {ok} right, {bad} wrong")
