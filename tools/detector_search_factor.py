#!/usr/bin/env python3
"""Count the multiscale detector's false detections on noise-only pings: where its measured constants come from.

    tools/detector_search_factor.py [--pings 400] [--rows 1,2,3] [--budgets 3,1,0.3,0.1] [--seed 4000] [--json FILE]
    tools/detector_search_factor.py --verify [--pings 200] [--budgets 3,1] [--seed 9000] [--json FILE]
    tools/detector_search_factor.py --search-factor 2.5 ...       the detector as it was: one flat factor for every test

The detector (libs/live detector.hpp) turns a budget of false detections per ping into a threshold for each of its
tests by counting the independent tests its map holds. The map is sampled finer than a range resolution and a
beamwidth, its largest sample is searched for, and a detection is a group of flagged cells, so that count cannot be
read off the grid. It is

    the single sample    kSearchSample[rows] x beams x extent / resolution x u
    the cell mean        kSearchMean[rows]   x beams x extent / cell       x sqrt(u) x hazard

with u = rows x looks x threshold (how the number of separate maxima of a smooth field, and of upcrossings along
bearing, grows with the threshold: docs/PHYSICS_NOTES.md Sec. 29.3), and the two constants per number of rows MEASURED
here.

What is done: noise-only pings of the long-range array with one, two and three rows (configs/sonars
popoto_long_range, _2row, _3row; 800 m of window, the deployed bearings -60 to 60 degrees every 0.5, white noise on
every hydrophone) go through the Python chain's own map; on each map the detector is run with ONE of its tests at a
time, at the threshold its plan gives for a stated number of false detections per ping from that test, and the
detections it reports are counted. The constant that would have made the count equal what was asked is the plan's own
times found / asked; it is printed per budget and pooled over the budgets of 3 a ping and fewer (at 10 a ping
neighbouring false detections begin to be merged), with the count and its Poisson error.

`--verify`: the whole detector as shipped (both tests, the constants in the code) at loosened budgets on other seeds:
false detections per ping against the budget.

The table in libs/live/include/sonar/live/detector.hpp was made with the default arguments, starting from a flat factor
of 2.5 (`--search-factor 2.5`, growth laws applied to the thresholds used); run it again and compare before changing
the map (bearing step, samples per cell, aperture window) or the grouping. Nothing is measured at the default budget of
one false detection in a thousand pings: that would take a hundred thousand pings.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from gpusonar import core, detect, live  # noqa: E402
from gpusonar.live import LiveChain  # noqa: E402
from gpusonar.sim import Scene, make_pose, simulate_ping  # noqa: E402
from gpusonar.sonar import SonarModel  # noqa: E402

SONARS = {1: "popoto_long_range", 2: "popoto_long_range_2row", 3: "popoto_long_range_3row"}
WINDOW = {"receive": {"kind": "baseband", "sample_rate": 8000.0, "window": {"range": [900.0, 1700.0]}}}
CHAIN = {"bearings_deg": [-60.0, 60.0, 0.5], "backend": "host", "cell_statistic": "multiscale"}
NOISE_RMS = 2e-4
POOLED_BUDGETS = 3.0          # budgets up to this many false detections a ping go into the pooled constant
TESTS = ((0, "sample"), (1, "cell mean"))


def say(text: str) -> None:
    print(text, flush=True)


def noise_ping(sonar, seed: int, sequence: int = 0):
    """One ping with nothing in the water: white noise on every hydrophone (the target is 300 dB down)."""
    scene = Scene()
    scene.add_points([[0.0, 1300.0, 0.0]], target_strength_db=-300.0)
    return simulate_ping(sonar, make_pose(position=(0.0, 0.0, 0.0)), scene, noise_rms=NOISE_RMS, rng=np.random.default_rng(seed), sequence=sequence,
                         environment={"sound_speed": 1500.0}, adc=False, absorption_model="none")


def chain_config(rows: int, **cfar) -> dict:
    cfg = dict(CHAIN, engine="python")
    if rows > 1:
        cfg["elevation"] = {}                       # the rows' maps are added in power
    cfar = {k: v for k, v in cfar.items() if v is not None}
    if cfar:
        cfg["cfar"] = cfar
    return cfg


def template(rows: int):
    return SonarModel.load(SONARS[rows], WINDOW).make_record(core.Pose(), environment={"sound_speed": 1500.0})


def maps(rows: int, pings: int, seed: int):
    """The chain's own map of each noise ping, as its detector is handed it: (fine, cell_mean, fine_ranges, cell_ranges,
    bearings, guard, train, plan)."""
    sonar = SonarModel.load(SONARS[rows], WINDOW)
    seen = []
    plain = detect.multiscale

    def spy(*args):
        seen.append(args)
        return plain(*args)

    detect.multiscale = spy
    try:
        with LiveChain(chain_config(rows)) as chain:
            for i in range(pings):
                chain.process(noise_ping(sonar, seed + i, i))
                yield seen.pop()
    finally:
        detect.multiscale = plain


def one_test(plan: dict, which: int) -> dict:
    """The plan with only test `which` (0: the sample, 1: the cell mean) switched on."""
    scales = [dict(s) for s in plan["scales"][:2]]
    scales[1 - which]["factor"] = float("inf")
    return dict(plan, scales=scales)


def growth(scale: dict, rows: int) -> float:
    """What the test's count of tests grows by with its threshold (detector.hpp): u for the sample, sqrt(u) x the
    statistic's hazard for a mean; u = rows x looks x threshold."""
    x, a = float(scale["factor"]), rows * float(scale["looks"])
    u = a * x
    if int(scale["cells"]) == 0:
        return u
    h = 1e-4 * x
    w = list(scale["weights"])
    hazard = (np.log(core.weighted_exponential_tail(w, rows, x - h)) - np.log(core.weighted_exponential_tail(w, rows, x + h))) / (2.0 * h * a)
    return float(np.sqrt(u) * hazard)


def measure(rows_list, pings: int, budgets, seed: int, search_factor=None, log=say, store: Path | None = None) -> dict:
    """Each test alone at each budget. `search_factor`: a flat factor for the thresholds used (None: the code's table)."""
    out = {}
    for rows in rows_list:
        t0 = time.time()
        record = template(rows)
        # A budget of 2 b shared between the two tests: b false detections a ping asked of each.
        plans = {b: live.detector_plan(record, chain_config(rows, false_alarms_per_ping=2.0 * b, search_factor=search_factor), rows) for b in budgets}
        counts = {(w, b): 0 for w, _ in TESTS for b in budgets}
        for args in maps(rows, pings, seed + 100000 * rows):
            for (which, b) in counts:
                counts[(which, b)] += len(detect.multiscale(*args[:-1], one_test(plans[b], which))[0])
        any_plan = plans[budgets[0]]
        res = {"rows": rows, "pings": pings, "seconds": round(time.time() - t0, 1), "beams": any_plan["beams"], "cell_m": any_plan["cell_m"],
               "resolution_m": any_plan["resolution_m"], "samples_per_cell": any_plan["samples_per_cell"], "tests": {}}
        for which, name in TESTS:
            entries, found, expected = [], 0, 0.0
            for b in budgets:
                s = plans[b]["scales"][which]
                n = counts[(which, b)]
                places = any_plan["beams"] * res["cell_m"] * (len(args[3])) / max(s["length_m"], res["resolution_m"])
                grows = growth(s, rows)
                # the constant that would have made the count what was asked, with the growth laws: tests x found / asked / (places x growth)
                constant = s["tests"] * n / (b * pings) / (places * grows)
                entries.append({"asked_per_ping": b, "found": n, "found_per_ping": n / pings, "ratio": n / (b * pings), "ratio_sigma": np.sqrt(max(n, 1)) / (b * pings),
                                "threshold_db": float(10.0 * np.log10(s["factor"])), "plan_tests": s["tests"], "plan_search": s["search"], "growth": grows,
                                "constant": constant})
                if b <= POOLED_BUDGETS:
                    found += n
                    expected += b * pings * places * grows / s["tests"]
                log(f"rows {rows} {name:9s} asked {b:5.2f} a ping at {entries[-1]['threshold_db']:5.2f} dB: found {n:5d} in {pings} pings = {n / pings:.3f} a ping, "
                    f"{n / (b * pings):.2f} +- {np.sqrt(max(n, 1)) / (b * pings):.2f} of what was asked; constant {constant:.3f}")
            pooled = found / expected if expected > 0 else float("nan")
            res["tests"][name] = {"looks": plans[budgets[0]]["scales"][which]["looks"], "by_budget": entries, "constant": pooled,
                                  "constant_sigma": pooled / np.sqrt(max(found, 1)), "detections": found}
            log(f"rows {rows} {name:9s} constant over the budgets of {POOLED_BUDGETS:g} a ping and fewer: {pooled:.3f} +- {pooled / np.sqrt(max(found, 1)):.3f} ({found} detections)")
        out[str(rows)] = res
        if store is not None:
            store.write_text(json.dumps(out, indent=1, default=float) + "\n")       # after every row count: a long run can be read while it goes
    return out


def verify(rows_list, pings: int, budgets, seed: int, search_factor=None, log=say, store: Path | None = None) -> dict:
    """The detector as shipped (both tests) at loosened budgets, counted on the chain's own maps."""
    out = {}
    for rows in rows_list:
        record = template(rows)
        plans = {b: live.detector_plan(record, chain_config(rows, false_alarms_per_ping=b, search_factor=search_factor), rows) for b in budgets}
        counts = {b: 0 for b in budgets}
        for args in maps(rows, pings, seed + 100000 * rows):
            for b in budgets:
                counts[b] += len(detect.multiscale(*args[:-1], plans[b])[0])
        out[str(rows)] = {str(b): {"found": n, "pings": pings, "per_ping": n / pings, "ratio": n / (b * pings), "ratio_sigma": np.sqrt(max(n, 1)) / (b * pings),
                                   "thresholds_db": [float(10.0 * np.log10(s["factor"])) for s in plans[b]["scales"]]} for b, n in counts.items()}
        for b, n in counts.items():
            log(f"rows {rows} budget {b:5.2f} a ping: {n} false detections in {pings} pings = {n / pings:.3f} a ping, {n / (b * pings):.2f} +- {np.sqrt(max(n, 1)) / (b * pings):.2f} of the budget")
        if store is not None:
            store.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pings", type=int, default=None)
    ap.add_argument("--rows", default="1,2,3")
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--search-factor", type=float, default=None, help="a flat factor for the thresholds used (2.5: the detector as it was)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rows = [int(r) for r in args.rows.split(",") if r]
    store = Path(args.json) if args.json else None
    if args.verify:
        budgets = [float(b) for b in (args.budgets or "3,1").split(",") if b]
        verify(rows, args.pings or 200, budgets, 9000 if args.seed is None else args.seed, args.search_factor, store=store)
    else:
        budgets = [float(b) for b in (args.budgets or "3,1,0.3,0.1").split(",") if b]
        measure(rows, args.pings or 400, budgets, 4000 if args.seed is None else args.seed, args.search_factor, store=store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
